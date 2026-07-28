"""Hand-labelled (question, answer, should_pass) cases for measuring the *judge*.

The main benchmark has no ground truth — every query is scored by the judge, so its "pass
rate" measures agreement with the judge, not correctness. That makes it useless for the one
question that matters when you change the judge's prompt: did the router get better, or did
the referee just get more permissive?

These cases have known-correct labels, so they measure the judge directly. Two error types
matter and they are not equally bad:

- **false pass** — judge accepts a wrong answer. The user is handed something incorrect and
  the cascade stops. This is the harmful one.
- **false fail** — judge rejects a correct answer. The cascade escalates unnecessarily and
  burns compute, but the user still gets a right answer. Wasteful, not harmful.

Deliberately included: correct answers that are terse, verbose, or show no working (a strict
judge false-fails these), and wrong answers that are fluent and well-presented (a permissive
judge false-passes these).
"""

# (question, proposed answer, should_pass, task_type, why this case exists)
# task_type matters: the judge applies a different rubric to summarization and translation,
# so a check that always passed None would be testing a code path the cascade never uses.
JUDGE_CASES = [
    # --- correct answers a too-strict judge tends to reject ---
    ("What is the capital of Australia?", "Canberra", True, "qa", "correct but maximally terse"),
    ("What is 17 * 23?", "391", True, "reasoning", "correct, no working shown"),
    ("A farmer has 17 sheep. All but 9 die. How many sheep are left?",
     "The remaining number of sheep is 9, which is less than the original number of 17.",
     True, "qa", "correct but verbose — the real false-fail that started this"),
    ("What is the 47th digit after the decimal point of pi?", "7", True,
     "qa", "correct; was false-failed for months as a 'hallucination'"),
    ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
     "How much does the ball cost?", "$0.05", True, "reasoning", "correct, terse"),
    ("Translate 'Thank you very much' to German", "Vielen Dank", True,
     "translation", "correct, not a literal word-for-word match"),
    ("Summarize: Regular exercise improves cardiovascular health, boosts mental well-being, "
     "aids weight management, and increases longevity.",
     "Exercise benefits the heart, the mind, and overall lifespan.", True,
     "summarization", "faithful summary that drops secondary detail — the point of summarizing"),

    # --- wrong answers a too-permissive judge tends to accept ---
    ("A farmer has 17 sheep. All but 9 die. How many sheep are left?", "8", False,
     "qa", "the classic wrong answer, stated confidently"),
    ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
     "How much does the ball cost?", "$0.10", False,
     "reasoning", "the classic intuitive-but-wrong answer"),
    ("A farmer has 17 sheep. All but 9 die. How many sheep are left?",
     "No information is given about any sheep deaths, so the answer cannot be determined.",
     False, "qa", "fluent refusal of an answerable question"),
    ("What is the capital of Australia?", "Sydney", False, "qa", "plausible and wrong"),
    ("What is the 47th digit after the decimal point of pi?",
     "To find it, use the Bailey-Borwein-Plouffe formula or a spigot algorithm, "
     "computing pi to at least 48 decimal places.", False,
     "qa", "describes how to find the answer instead of giving it"),
    ("What is 17 * 23?", "371", False, "reasoning", "arithmetic slip"),
    ("Explain the difference between TCP and UDP.",
     "TCP and UDP are both programming languages used for web development.", False,
     "qa", "confidently wrong"),
]
