"""Rank candidate models for the coding tiers by EXECUTING what they write.

Why this exists
---------------
Every other quality number in this project is scored by the judge, which false-passes about 29%
of wrong answers (BUILD-LOG #18) and contributes a ~12-point run-to-run swing (BUILD-LOG #21).
On five coding queries that noise is larger than any real difference between models, so the
judge cannot rank them.

Generated code can be *run*. Each task ships assertions; a model scores only if its function
actually passes them. No verdicts, no judge, no circularity — the same reason
eval/judge_ground_truth.py exists, applied to code.

Usage
-----
    .venv/bin/python -m eval.compare_coding_models

Models on a provider whose quota is exhausted report `unavailable` rather than scoring 0 —
scoring an outage as a failure is the mistake this project keeps having to un-make.
"""
import json
import subprocess
import sys
import tempfile
import time

from src.cascade import ANSWER_SYSTEM_PROMPT, AnswerResult
from src.llm_client import ModelUnavailable, call_groq, call_json, call_openrouter

# (prompt, assertions). Assertions must be self-contained and must not print.
TASKS = [
    ("Write a Python function `is_palindrome(s)` that returns True if the string is a "
     "palindrome ignoring case, spaces and punctuation.",
     "assert is_palindrome('A man, a plan, a canal: Panama')\n"
     "assert is_palindrome('')\n"
     "assert not is_palindrome('hello')\n"
     "assert is_palindrome('No lemon, no melon')"),

    ("Write a Python function `binary_search(arr, target)` that returns the index of target in "
     "a sorted list, or -1 if absent.",
     "assert binary_search([1,3,5,7,9], 7) == 3\n"
     "assert binary_search([1,3,5,7,9], 1) == 0\n"
     "assert binary_search([], 4) == -1\n"
     "assert binary_search([1,3,5], 4) == -1"),

    ("Write a Python function `merge_sorted(a, b)` that merges two sorted lists into one "
     "sorted list.",
     "assert merge_sorted([1,3,5],[2,4,6]) == [1,2,3,4,5,6]\n"
     "assert merge_sorted([],[1,2]) == [1,2]\n"
     "assert merge_sorted([1,1],[1]) == [1,1,1]\n"
     "assert merge_sorted([],[]) == []"),

    ("Write a Python function `lcs_length(a, b)` returning the length of the longest common "
     "subsequence of two strings.",
     "assert lcs_length('ABCBDAB','BDCABA') == 4\n"
     "assert lcs_length('','abc') == 0\n"
     "assert lcs_length('abc','abc') == 3\n"
     "assert lcs_length('abc','def') == 0"),

    ("Write a Python function `reverse_list(head)` that reverses a singly linked list in O(1) "
     "extra space. Nodes have `.val` and `.next` attributes; return the new head.",
     "class N:\n"
     "    def __init__(s,v): s.val=v; s.next=None\n"
     "def build(vals):\n"
     "    hd=None\n"
     "    for v in reversed(vals):\n"
     "        n=N(v); n.next=hd; hd=n\n"
     "    return hd\n"
     "def tolist(h):\n"
     "    out=[]\n"
     "    while h: out.append(h.val); h=h.next\n"
     "    return out\n"
     "assert tolist(reverse_list(build([1,2,3]))) == [3,2,1]\n"
     "assert reverse_list(None) is None\n"
     "assert tolist(reverse_list(build([1]))) == [1]"),

    ("Write a Python function `word_count(text)` returning a dict mapping each lowercase word "
     "to its count, splitting on whitespace and stripping punctuation.",
     "r = word_count('The cat, the CAT!')\n"
     "assert r.get('the') == 2, r\n"
     "assert r.get('cat') == 2, r"),
]

# (provider, model_id, total_params_b). Registry incumbents plus the Groq alternatives that
# would let coding survive an OpenRouter outage.
CANDIDATES = [
    ("groq", "llama-3.1-8b-instant", 8),
    ("groq", "openai/gpt-oss-20b", 20),
    ("groq", "openai/gpt-oss-120b", 117),
    ("groq", "qwen/qwen3.6-27b", 27),
    ("groq", "llama-3.3-70b-versatile", 70),
    ("openrouter", "cohere/north-mini-code:free", 7),
    ("openrouter", "poolside/laguna-xs-2.1:free", 7),
    ("openrouter", "poolside/laguna-s-2.1:free", 14),
]

REPEATS = 2
PACE_S = 2.2  # Groq allows 30 requests/minute; stay just under it
EXEC_TIMEOUT_S = 15

SUFFIX = (" Return ONLY the function definition. No example usage, no markdown fences, "
          "no explanation.")


def run_generated(code: str, tests: str) -> tuple[bool, str]:
    """Run model-written code in a separate process with a timeout.

    Separate process because generated code can loop forever, call sys.exit, or shadow a builtin
    — none of which should be able to take the harness down with it.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n\n" + tests + "\nprint('__PASS__')\n")
        path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True,
                           timeout=EXEC_TIMEOUT_S)
        err = (p.stderr.strip().splitlines() or [""])[-1][:90]
        return ("__PASS__" in p.stdout, err)
    except subprocess.TimeoutExpired:
        return (False, f"timed out after {EXEC_TIMEOUT_S}s")


def score(provider: str, model_id: str) -> dict:
    call_fn = call_groq if provider == "groq" else call_openrouter
    passed = attempted = unavailable = 0
    failures = []
    for _ in range(REPEATS):
        for prompt, tests in TASKS:
            time.sleep(PACE_S)
            try:
                result = call_json(call_fn, model_id, ANSWER_SYSTEM_PROMPT, prompt + SUFFIX,
                                   AnswerResult)
            except ModelUnavailable:
                unavailable += 1
                continue
            attempted += 1
            if result is None:
                failures.append(f"{prompt[:36]}... -> unparseable response")
                continue
            code = result.answer.replace("```python", "").replace("```", "")
            ok, err = run_generated(code, tests)
            passed += ok
            if not ok:
                failures.append(f"{prompt[:36]}... -> {err}")
    return {"passed": passed, "attempted": attempted, "unavailable": unavailable,
            "failures": failures}


def main():
    results = {}
    for provider, model_id, size in CANDIDATES:
        r = score(provider, model_id)
        r["provider"], r["total_params_b"] = provider, size
        results[model_id] = r
        state = (f"{r['passed']}/{r['attempted']}" if r["attempted"]
                 else f"unavailable ({r['unavailable']} calls refused)")
        print(f"  {provider:<11} {model_id:<30} {state}", flush=True)

    print("\n" + "=" * 78)
    print(f"{'model':<32}{'provider':<12}{'exec pass':<16}{'total'}")
    ranked = sorted(results.items(),
                    key=lambda kv: -(kv[1]["passed"] / kv[1]["attempted"]) if kv[1]["attempted"] else 1)
    for model_id, r in ranked:
        if not r["attempted"]:
            print(f"{model_id:<32}{r['provider']:<12}{'unavailable':<16}{r['total_params_b']}B")
            continue
        pct = r["passed"] / r["attempted"] * 100
        print(f"{model_id:<32}{r['provider']:<12}{r['passed']}/{r['attempted']} = {pct:5.1f}%   "
              f"{r['total_params_b']}B")
        for f in r["failures"][:2]:
            print(f"      {f}")

    skipped = [m for m, r in results.items() if not r["attempted"]]
    if skipped:
        print(f"\n{len(skipped)} model(s) unreachable this run — NOT scored as failures. "
              f"Re-run once the provider quota resets for a complete comparison:")
        for m in skipped:
            print(f"  - {m}")

    with open("eval/coding_model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nWritten to eval/coding_model_comparison.json")


if __name__ == "__main__":
    main()
