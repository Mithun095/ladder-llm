from src.cascade import TraceStep
from src.formatter import format_answer
from src.metrics import compute_saved_pct, total_active_params_burned

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
    TraceStep(tier=3, model_id="tier3", status="judged_fail", confidence=9, active_params_b=32),
    TraceStep(tier=4, model_id="tier4", status="accepted", confidence=9, active_params_b=55),
]
assert compute_saved_pct(expensive, "coding") < 0, "cascade costing more than baseline must report negative savings"

# An unavailable tier never ran, so it costs nothing — unlike a malformed response, which did.
free = [TraceStep(tier=1, model_id="down", status="unavailable")]
assert total_active_params_burned(free) == 0

assert format_answer("def f(): pass", "coding").startswith("```")
assert format_answer("hello", "coding") == "```\nhello\n```" or "```" in format_answer("hello", "coding")

print("metrics/formatter check passed.")
