from typing import Literal

from pydantic import BaseModel

from src.llm_client import ModelUnavailable, call_groq, call_json

JUDGE_MODEL = "llama-3.1-8b-instant"

# This prompt has been through three versions, each fixing the previous one's measured failure.
# Measure any change with checks/check_judge_accuracy.py — a looser judge raises the benchmark
# pass rate while handing users wrong answers, and the main benchmark cannot detect that.
#
# v1 "You are a STRICT answer judge... decide if the answer is correct and adequately addresses
#    the question" — graded presentation, not substance. Failed correct-but-verbose answers.
# v2 explicit pass/fail conditions, style ruled out of scope — fixed the false fails, but
#    measured a 57% FALSE PASS rate: shown "17 * 23 = 371" it replied "correctly stated".
#    It wasn't verifying at all, it was agreeing with whatever it was shown and inventing a
#    justification afterwards.
# v3 (this one) makes the judge answer the question ITSELF first, in the same JSON, before it
#    is allowed to render a verdict. Committing to an independent answer is what breaks the
#    agree-with-the-input bias — it has to notice its own answer differs before it can approve.
BASE_SYSTEM_PROMPT = """You are an answer judge.

Work in this order, and do not skip step 1:
1. Answer the question YOURSELF, independently, without being influenced by the proposed \
answer. For arithmetic or logic, actually work it out.
2. Compare your answer to the proposed one.
3. Give a verdict.

Verdict rules:
- PASS if the proposed answer agrees with yours and answers what was asked — even if it is \
terse, verbose, informally worded, or shows no working.
- FAIL if it is factually wrong, contradicts your own answer, answers a different question, \
refuses to answer an answerable question, or only describes how one *would* find the answer \
instead of giving it.

Do not fail an answer for style, formatting, length, or missing explanation. Do not assume the \
proposed answer is correct — it frequently is not.

Respond with ONLY this JSON shape, no other text:
{"own_answer": "<your own answer from step 1, as briefly as possible — a number or one short phrase>", "verdict": "pass"|"fail", "reason": "<one sentence>"}"""

# A generic "is this factually correct" rubric fails subjective tasks: it nitpicks a valid
# summary for omitting a secondary detail, or a valid translation for not being a literal
# word-for-word match. Found empirically via eval/run_eval.py — summarization queries were
# failing the judge almost every time despite objectively reasonable answers.
TYPE_GUIDANCE = {
    "summarization": "\n\nThis is a summarization task: judge whether the summary faithfully "
    "captures the main point, not whether it retains every secondary detail — omitting minor "
    "details is the point of summarizing, not a flaw.",
    "translation": "\n\nThis is a translation task: judge whether the meaning is preserved, "
    "not whether the phrasing is a literal word-for-word match.",
    # Coding had no entry here until coding tiers moved to Groq and became reachable enough to
    # measure. Against execution ground truth the judge was false-failing 3 of 4 working
    # implementations, with reasons like "fails to explain the implementation of slicing" — it
    # was grading the explanation, not the code. Step 1 of the base prompt tells it to answer the
    # question itself first, which for code means writing its own implementation; two correct
    # programs rarely look alike, so "differs from mine" was reading as "wrong".
    "coding": "\n\nThis is a coding task. For step 1, do not write your own implementation and "
    "compare them line by line — two correct programs for the same task usually look nothing "
    "alike. Instead, hand-execute what the proposed code DOES on the inputs the question "
    "implies, including an awkward one, and decide whether that behaviour is what was asked "
    "for.\n"
    "PASS working code regardless of style: terse or verbose, commented or bare, any correct "
    "algorithm, any naming, with or without docstrings or type hints. A comment that is wrong or "
    "misleading is not a reason to fail code that behaves correctly.\n"
    "FAIL only if hand-executing it gives a wrong result, it crashes, it solves a different "
    "problem, or it returns the wrong type — for example returning a string where a boolean was "
    "asked for.",
}


class JudgeResult(BaseModel):
    # own_answer isn't consumed anywhere — its job is to force the model to commit to an
    # independent answer before the verdict field, so it can't just ratify what it was shown.
    own_answer: str = ""
    verdict: Literal["pass", "fail"]
    reason: str


def judge(query: str, answer: str, task_type: str | None = None) -> JudgeResult | None:
    """Returns None if the judge can't produce a verdict — either it was rate-limited or it
    returned unparseable JSON. The caller decides what an un-judged answer means; it must not
    crash the request, since the judge is a quality gate, not the answer itself."""
    system = BASE_SYSTEM_PROMPT + TYPE_GUIDANCE.get(task_type, "")
    user = f"Question: {query}\n\nAnswer: {answer}"
    try:
        return call_json(call_groq, JUDGE_MODEL, system, user, JudgeResult)
    except ModelUnavailable:
        return None
