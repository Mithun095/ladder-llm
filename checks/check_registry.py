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

# Active params are NOT asserted monotonic on purpose. A sparse MoE can have fewer active params
# than a smaller dense model while costing more per token, so the two metrics genuinely disagree
# (translation tier 2 is 5.1B active vs. tier 3's 4B, yet tier 3 costs twice as much). Price is
# what the ladder is ordered by; forcing both would mean deleting one of the two honest numbers.
print(f"registry check passed: {MAX_TIER * len(TASK_TYPES)} pairs resolved, "
      f"price ladder monotonic for all {len(TASK_TYPES)} task types.")
