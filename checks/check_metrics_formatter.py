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
assert 0 <= pct <= 100, f"pct out of range: {pct}"
print(f"burned={burned}B, saved={pct:.1f}%")

assert format_answer("def f(): pass", "coding").startswith("```")
assert format_answer("hello", "coding") == "```\nhello\n```" or "```" in format_answer("hello", "coding")

print("metrics/formatter check passed.")
