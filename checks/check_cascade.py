from src.cascade import run_cascade

result = run_cascade("What is a closure in Python?")

print("Answer:", result.answer[:200])
print("Tier used:", result.tier_used)
print("Trace:")
for step in result.trace:
    print(f"  tier={step.tier} model={step.model_id} status={step.status} confidence={step.confidence}")

assert result.answer and result.answer != "No model produced a usable answer."
assert len(result.trace) >= 1
assert result.trace[-1].status in ("accepted", "judged_fail", "unavailable")

print("cascade check passed.")
