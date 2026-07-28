from src.judge import judge

good = judge("What is 2 + 2?", "4")
assert good is not None
print("good answer verdict:", good.verdict, "-", good.reason)
assert good.verdict == "pass"

bad = judge("What is 2 + 2?", "The capital of France is Paris.")
assert bad is not None
print("bad answer verdict:", bad.verdict, "-", bad.reason)
assert bad.verdict == "fail"

print("judge check passed.")
