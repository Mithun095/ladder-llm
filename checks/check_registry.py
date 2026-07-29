from src.cascade import CEILING_TIER, STARTING_TIER
from src.classifier import ClassifierResult
from src.metrics import model_cost_usd
from src.registry import MAX_TIER, TASK_TYPES, get_model

for tier in range(1, MAX_TIER + 1):
    for task_type in TASK_TYPES:
        config = get_model(tier, task_type)
        assert config.provider in ("groq", "openrouter")
        assert config.model_id
        assert config.active_params_b > 0
        # Either both prices are published or neither is; a half-populated entry would silently
        # fall back to the active-param estimate and look like a real quote.
        assert (config.usd_per_m_in is None) == (config.usd_per_m_out is None), \
            f"tier={tier} type={task_type} has only one of the two published rates"

# The whole premise of a cascade is that escalating costs more than not escalating. That held by
# assumption until published rates were checked and tiers 2 and 3 turned out to be inverted for
# four of the five task types — the cascade was escalating *down* the price curve, and the
# ceiling was blocking the cheaper, larger model. Asserted here so it cannot silently invert
# again when a model is swapped. See BUILD-LOG.md #20.
for task_type in TASK_TYPES:
    costs = [model_cost_usd(get_model(t, task_type)) for t in range(1, MAX_TIER + 1)]
    for tier in range(1, MAX_TIER):
        assert costs[tier] >= costs[tier - 1], (
            f"{task_type}: tier {tier + 1} (${costs[tier]:.6f}) is cheaper than tier {tier} "
            f"(${costs[tier - 1]:.6f}) — escalation must not move down the price curve"
        )

# Where the ladder is *entered* is a separate question from how it's ordered, and the price
# assertion above does not cover it. `expert` used to start at tier 3, which after the tier 2/3
# swap meant every expert query skipped the cheap 120B tier to open on the expensive dense one —
# a routing bug the monotonicity check happily passed. These three assertions cover it.
DIFFICULTIES = ClassifierResult.model_fields["difficulty"].annotation.__args__
assert set(STARTING_TIER) == set(DIFFICULTIES) == set(CEILING_TIER), (
    "STARTING_TIER/CEILING_TIER must cover exactly the difficulties the classifier can return — "
    "a missing key is a KeyError on a live query"
)
for difficulty in DIFFICULTIES:
    start, ceiling = STARTING_TIER[difficulty], CEILING_TIER[difficulty]
    assert 1 <= start <= ceiling <= MAX_TIER, \
        f"{difficulty}: start={start} ceiling={ceiling} — the cascade loop would never run"
    for task_type in TASK_TYPES:
        # Entering above tier 1 skips tiers, and because price is monotonic every skipped tier is
        # cheaper. That alone is fine — it's the deliberate bet that a cheap model is too weak to
        # be worth a call. What is never fine is skipping a tier that is cheaper AND larger than
        # the one you land on, because then there is no bet: the skipped tier wins on both axes.
        #
        # This is exactly what `expert` did. It started at tier 3 (qwen3.6-27b: 27B total,
        # $0.43/query) and skipped tier 2 (gpt-oss-120b: 117B total, $0.038/query) — cheaper and
        # 4x the parameters. The "too weak to bother with" justification cannot apply to a model
        # that is bigger than the one you chose instead.
        entry = get_model(start, task_type)
        for skipped_tier in range(1, start):
            skipped = get_model(skipped_tier, task_type)
            dominates = (model_cost_usd(skipped) <= model_cost_usd(entry)
                         and skipped.total_params_b > entry.total_params_b)
            assert not dominates, (
                f"{difficulty}/{task_type}: starts at tier {start} ({entry.model_id}, "
                f"{entry.total_params_b}B total, ${model_cost_usd(entry):.5f}/query) but skips "
                f"tier {skipped_tier} ({skipped.model_id}, {skipped.total_params_b}B total, "
                f"${model_cost_usd(skipped):.5f}/query) — the skipped tier is both cheaper and "
                f"larger, so skipping it is strictly worse"
            )

# Active params are NOT asserted monotonic on purpose. A sparse MoE can have fewer active params
# than a smaller dense model while costing more per token, so the two metrics genuinely disagree
# (translation tier 2 is 5.1B active vs. tier 3's 4B, yet tier 3 costs twice as much). Price is
# what the ladder is ordered by; forcing both would mean deleting one of the two honest numbers.
print(f"registry check passed: {MAX_TIER * len(TASK_TYPES)} pairs resolved, "
      f"price ladder monotonic for all {len(TASK_TYPES)} task types.")
