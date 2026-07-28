from typing import Literal

from pydantic import BaseModel

from src.llm_client import ModelUnavailable, call_groq, call_json

JUDGE_MODEL = "llama-3.1-8b-instant"

BASE_SYSTEM_PROMPT = """You are a strict answer judge. Given a question and a proposed answer, decide \
if the answer is correct and adequately addresses the question. Respond with ONLY this JSON \
shape, no other text:
{"verdict": "pass"|"fail", "reason": "<one sentence>"}"""

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
}


class JudgeResult(BaseModel):
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
