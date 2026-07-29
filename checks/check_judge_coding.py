"""Does the judge agree with the Python interpreter about whether code is correct?

The judge's verdicts are checked against EXECUTION, not against my opinion. Each case ships a
function and assertions; whether the code is right is decided by running it. That makes this a
ground-truth measurement the judge doesn't control — the same escape from circularity as
eval/judge_ground_truth.py (BUILD-LOG #18), applied to code.

Needs a live GROQ_API_KEY. Run after ANY change to the judge prompt.

    .venv/bin/python -m checks.check_judge_coding
"""
import subprocess
import sys
import tempfile
import time

from src.judge import judge

# Groq allows 30 requests/minute. Without pacing, later cases return "judge unavailable" and the
# measurement silently shrinks — which reads as a better score, since unavailable cases aren't
# counted as errors. A rate limit must not be able to flatter the result.
PACE_S = 2.5

QUERY = "Write a Python function `is_palindrome(s)` that returns True if the string is a " \
        "palindrome ignoring case, spaces and punctuation."
# 'abca' is load-bearing: without it the "only checks first and last character" implementation
# passes every other assertion, and this check would score the judge as correct for rejecting
# code the tests called working. A weak oracle produces confident wrong measurements.
TESTS = ("assert is_palindrome('A man, a plan, a canal: Panama')\n"
         "assert not is_palindrome('hello')\n"
         "assert not is_palindrome('abca')\n"
         "assert is_palindrome('')\n")

# Correct implementations written in deliberately different styles, because "looks unlike the
# answer I would have written" is the failure mode being probed.
CORRECT = [
    ("terse one-liner, no comments",
     "import re\n"
     "def is_palindrome(s):\n"
     "    c = re.sub(r'[^a-z0-9]', '', s.lower())\n"
     "    return c == c[::-1]"),
    ("verbose, docstring, heavy comments",
     "def is_palindrome(s):\n"
     '    """Check whether s reads the same forwards and backwards."""\n'
     "    # Step 1: keep only letters and digits\n"
     "    cleaned = ''.join(ch.lower() for ch in s if ch.isalnum())\n"
     "    # Step 2: compare against the reverse. This is O(n) space.\n"
     "    return cleaned == cleaned[::-1]"),
    ("two-pointer, no slicing",
     "def is_palindrome(s):\n"
     "    t = [c.lower() for c in s if c.isalnum()]\n"
     "    i, j = 0, len(t) - 1\n"
     "    while i < j:\n"
     "        if t[i] != t[j]:\n"
     "            return False\n"
     "        i += 1; j -= 1\n"
     "    return True"),
    ("correct but with a WRONG comment claiming O(1) space",
     "def is_palindrome(s):\n"
     "    # Reverses the string in O(1) space.\n"
     "    c = ''.join(ch.lower() for ch in s if ch.isalnum())\n"
     "    return c == c[::-1]"),
]

# Wrong in ways that still look plausible — the false-pass probe.
WRONG = [
    ("forgets to strip punctuation",
     "def is_palindrome(s):\n"
     "    return s.lower() == s.lower()[::-1]"),
    ("off-by-one, drops the last character",
     "def is_palindrome(s):\n"
     "    c = ''.join(ch.lower() for ch in s if ch.isalnum())\n"
     "    return c[:-1] == c[::-1][:-1] and len(c) > 1"),
    ("returns the cleaned string instead of a bool",
     "def is_palindrome(s):\n"
     "    return ''.join(ch.lower() for ch in s if ch.isalnum())"),
    ("checks only that the first and last characters match",
     "def is_palindrome(s):\n"
     "    c = ''.join(ch.lower() for ch in s if ch.isalnum())\n"
     "    return not c or c[0] == c[-1]"),
]


def executes_correctly(code: str) -> bool:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n\n" + TESTS + "\nprint('__PASS__')\n")
        path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=15)
        return "__PASS__" in p.stdout
    except subprocess.TimeoutExpired:
        return False


def main():
    false_fails = false_passes = 0
    graded = 0
    print(f"{'label':<48}{'runs?':<8}{'judge':<8}verdict")
    print("-" * 78)
    unavailable = 0
    for label, code in CORRECT + WRONG:
        runs = executes_correctly(code)
        time.sleep(PACE_S)
        verdict = judge(QUERY, code, "coding")
        if verdict is None:
            unavailable += 1
            print(f"{label:<48}{'yes' if runs else 'no':<8}{'n/a':<8}judge unavailable")
            continue
        graded += 1
        passed = verdict.verdict == "pass"
        if runs and not passed:
            false_fails += 1
            tag = "FALSE FAIL — rejected working code"
        elif not runs and passed:
            false_passes += 1
            tag = "FALSE PASS — approved broken code"
        else:
            tag = "correct"
        print(f"{label:<48}{'yes' if runs else 'no':<8}{'pass' if passed else 'fail':<8}{tag}")
        if tag != "correct":
            print(f"{'':<64}judge said: {verdict.reason[:80]}")

    n_correct = sum(1 for _, c in CORRECT if executes_correctly(c))
    n_wrong = len(WRONG)
    print("-" * 78)
    print(f"false FAILS  (rejected code that runs)   : {false_fails}/{n_correct}")
    print(f"false PASSES (approved code that breaks) : {false_passes}/{n_wrong}")

    if unavailable:
        print(f"\n{unavailable} case(s) were not judged (rate limit) — the rates above are over "
              f"{graded} graded cases, not all {len(CORRECT) + len(WRONG)}.")
    if graded < len(CORRECT) + len(WRONG):
        print("Incomplete run: not asserting on a partial measurement.")
        return

    # Asymmetric thresholds, as in check_judge_accuracy.py: a false pass hands the user broken
    # code and stops the cascade, a false fail only wastes an escalation.
    #
    # These are loose because the instrument is small. Eight cases cannot distinguish a 25% error
    # rate from a 50% one, so this check exists to catch a REGRESSION (the judge grading
    # explanations again, which measured 3/4 false fails), not to certify a rate. Do not tune the
    # judge prompt against this file — that is how the v2 judge in BUILD-LOG #18 was made worse
    # while looking better.
    assert false_passes <= 2, f"judge approved {false_passes}/{n_wrong} broken implementations"
    assert false_fails <= 1, f"judge rejected {false_fails}/{n_correct} working implementations"
    print("\ncoding judge check passed.")


if __name__ == "__main__":
    main()
