from src.cascade import CascadeResult, TraceStep
from src.formatter import format_answer
from src.metrics import compute_saved_pct, total_active_params_burned, trace_cost_usd
from src.registry import MAX_TIER, TASK_TYPES

trace = [
    TraceStep(tier=1, model_id="llama-3.1-8b-instant", status="escalated", confidence=3, active_params_b=8),
    TraceStep(tier=2, model_id="some-tier2-model", status="accepted", confidence=9, active_params_b=27),
]

burned = total_active_params_burned(trace)
assert burned == 35, f"expected 35, got {burned}"

pct = compute_saved_pct(trace, "qa")
assert pct <= 100, f"pct out of range: {pct}"
print(f"burned={burned}B, saved={pct:.1f}%")

# Savings are deliberately unclamped: a cascade that escalates far enough can burn more than
# the max-tier baseline it's measured against, and that has to show up as a negative number
# rather than being rounded up to "0% saved".
expensive = [
    TraceStep(tier=3, model_id="tier3", status="judged_fail", confidence=9, active_params_b=14),
    TraceStep(tier=4, model_id="tier4", status="accepted", confidence=9, active_params_b=55),
]
assert compute_saved_pct(expensive, "coding") < 0, "cascade costing more than baseline must report negative savings"

# An unavailable tier never ran, so it costs nothing — unlike a malformed response, which did.
free = [TraceStep(tier=1, model_id="down", status="unavailable")]
assert total_active_params_burned(free) == 0

# CascadeResult.accepted gates every "saved" figure the UI and the eval harness report. The bug
# it replaced tested `answer != "<sentinel>"`, which is true for a judge-rejected answer too —
# so a run that burned two tiers and delivered nothing usable still claimed "36% saved".
rejected = CascadeResult(
    answer="The BBP formula computes hex digits of pi...",  # real text, still not an answer
    trace=[
        TraceStep(tier=1, model_id="tier1", status="judged_fail", confidence=8, active_params_b=8),
        TraceStep(tier=2, model_id="tier2", status="judged_fail", confidence=10, active_params_b=27),
    ],
)
assert not rejected.accepted, "a trace ending in judged_fail must not count as accepted"
assert CascadeResult(answer="x", trace=[]).accepted is False, "an empty trace is not an acceptance"
assert CascadeResult(
    answer="9",
    trace=[TraceStep(tier=1, model_id="t1", status="accepted", confidence=9, active_params_b=8)],
).accepted, "a trace ending in accepted must count as accepted"

# trace_cost_usd re-resolves each step's tier against the registry, so a tier/type pair the
# registry doesn't hold would raise KeyError inside the Streamlit render path — after the answer
# is already on screen. Every tier the cascade can emit, for every type it can classify as.
for _type in TASK_TYPES:
    for _tier in range(1, MAX_TIER + 1):
        step = TraceStep(tier=_tier, model_id="m", status="judged_fail", active_params_b=1)
        assert trace_cost_usd([step], _type) > 0, f"no cost for tier={_tier} type={_type}"
    # An unavailable tier never ran, so it must contribute nothing to the bill.
    assert trace_cost_usd([TraceStep(tier=1, model_id="m", status="unavailable")], _type) == 0
    assert trace_cost_usd([], _type) == 0

# Only accepted runs are cached. Caching a judge-rejected answer makes it permanent: the judge
# flips roughly one verdict in five between identical runs, so a retry would often succeed — but
# a cached failure means no retry ever happens.
#
# Driven through the real run_cascade with its three model calls stubbed out, so this exercises
# the actual caching condition rather than restating the property it depends on. No network.
import src.cascade as _cascade
from src.classifier import ClassifierResult


def _run_with_stubbed_models(verdict_word):
    _cascade._CACHE.clear()
    real = (_cascade.classify, _cascade.call_model, _cascade.judge)
    _cascade.classify = lambda q: ClassifierResult(
        difficulty="easy", type="qa", optimized_prompt=q)
    _cascade.call_model = lambda cfg, s, u, schema: _cascade.AnswerResult(
        answer="some answer", confidence=9)
    _cascade.judge = lambda q, a, t=None: type(
        "V", (), {"verdict": verdict_word, "reason": "stub"})()
    try:
        out = _cascade.run_cascade("a stubbed query", use_cache=True)
        return out, len(_cascade._CACHE)
    finally:
        _cascade.classify, _cascade.call_model, _cascade.judge = real
        _cascade._CACHE.clear()


_res, _cached = _run_with_stubbed_models("fail")
assert not _res.accepted, "stub should have produced a rejected run"
assert _cached == 0, "a judge-rejected run must not be cached — a retry could pass"

_res, _cached = _run_with_stubbed_models("pass")
assert _res.accepted and _cached == 1, "an accepted run must be cached"

assert format_answer("def f(): pass", "coding").startswith("```")
assert format_answer("hello", "coding") == "```\nhello\n```" or "```" in format_answer("hello", "coding")

print("metrics/formatter check passed.")
