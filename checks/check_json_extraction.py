"""Regression check for _extract_json — pure logic, no API calls, runs in CI.

This function has been rewritten three times and silently broken twice, each time in a way
that only showed up as a `malformed_response` in a live run. Every wrapper shape below is one
that a real model actually produced during this project.
"""
import json

from src.llm_client import _extract_json

CASES = [
    # (name, raw model output, expected parsed dict)
    (
        "bare json",
        '{"answer": "4", "confidence": 9}',
        {"answer": "4", "confidence": 9},
    ),
    (
        "markdown fenced",
        '```json\n{"answer": "hi", "confidence": 8}\n```',
        {"answer": "hi", "confidence": 8},
    ),
    (
        "chatty preamble",
        'Sure! Here is the JSON:\n{"answer": "Canberra", "confidence": 10}\nHope that helps.',
        {"answer": "Canberra", "confidence": 10},
    ),
    (
        # qwen3.6 and other reasoning-tuned models emit their chain of thought first, and
        # quote a *draft* of the JSON inside it. The last object emitted is the real one.
        "reasoning block quoting a draft",
        '<think>\nStep 5: Construct JSON: `{"answer": "8", "confidence": 4}`\n</think>\n\n'
        '{"answer": "9", "confidence": 10}',
        {"answer": "9", "confidence": 10},
    ),
    (
        # Regression: taking last-`{`-to-last-`}` grabbed the inner object and dropped the rest.
        "nested object",
        '{"answer": "x", "meta": {"a": 1}}',
        {"answer": "x", "meta": {"a": 1}},
    ),
    (
        # Regression: a coding answer containing braces is the common real case for this.
        "braces inside an answer string",
        '{"answer": "def f(): return {1, 2}", "confidence": 10}',
        {"answer": "def f(): return {1, 2}", "confidence": 10},
    ),
]

for name, raw, expected in CASES:
    got = json.loads(_extract_json(raw))
    assert got == expected, f"{name}: expected {expected}, got {got}"
    print(f"  ok  {name}")

# No JSON at all: return something, let the Pydantic layer above reject it — don't crash here.
assert isinstance(_extract_json("I'm sorry, I can't help with that."), str)

print(f"json extraction check passed: {len(CASES)} wrapper shapes.")
