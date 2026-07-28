"""Measure the judge against known-correct labels, separating its two error types.

Live check (real API calls). Run it after ANY change to the judge's prompt — a change that
raises the main benchmark's pass rate might just be a more permissive referee, and this is the
only thing in the repo that can tell the difference.

Thresholds are deliberately asymmetric: a false pass hands the user a wrong answer, a false
fail only wastes compute.
"""
from eval.judge_ground_truth import JUDGE_CASES
from src.judge import judge

MAX_FALSE_PASS_RATE = 0.35   # harmful error — the tighter budget
MAX_FALSE_FAIL_RATE = 0.45   # wasteful error — tolerated more

false_passes, false_fails, unusable = [], [], []
should_pass_n = sum(1 for case in JUDGE_CASES if case[2])
should_fail_n = len(JUDGE_CASES) - should_pass_n

for question, answer, expected_pass, task_type, why in JUDGE_CASES:
    verdict = judge(question, answer, task_type)
    if verdict is None:
        unusable.append(why)
        continue
    actual_pass = verdict.verdict == "pass"
    if actual_pass == expected_pass:
        mark = "ok  "
    elif actual_pass:
        mark = "FALSE PASS"
        false_passes.append((why, verdict.reason))
    else:
        mark = "FALSE FAIL"
        false_fails.append((why, verdict.reason))
    print(f"  {mark:11s} [{why}] -> {verdict.verdict}")

fp_rate = len(false_passes) / should_fail_n if should_fail_n else 0.0
ff_rate = len(false_fails) / should_pass_n if should_pass_n else 0.0

print(f"\n  false passes (accepted a wrong answer): {len(false_passes)}/{should_fail_n} = {fp_rate:.0%}")
print(f"  false fails  (rejected a right answer): {len(false_fails)}/{should_pass_n} = {ff_rate:.0%}")
if unusable:
    print(f"  no verdict returned: {len(unusable)}")

for label, items in (("FALSE PASSES", false_passes), ("FALSE FAILS", false_fails)):
    if items:
        print(f"\n  {label}:")
        for why, reason in items:
            print(f"    - {why}\n        judge said: {reason}")

assert fp_rate <= MAX_FALSE_PASS_RATE, (
    f"judge accepts too many wrong answers ({fp_rate:.0%} > {MAX_FALSE_PASS_RATE:.0%}) — "
    "a permissive judge inflates the benchmark pass rate while handing users wrong answers"
)
assert ff_rate <= MAX_FALSE_FAIL_RATE, (
    f"judge rejects too many correct answers ({ff_rate:.0%} > {MAX_FALSE_FAIL_RATE:.0%}) — "
    "this burns compute escalating answers that were already right"
)
print("\njudge accuracy check passed.")
