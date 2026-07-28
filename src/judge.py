from typing import Literal

from pydantic import BaseModel

from src.llm_client import call_groq, call_json

JUDGE_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are a strict answer judge. Given a question and a proposed answer, decide \
if the answer is correct and adequately addresses the question. Respond with ONLY this JSON \
shape, no other text:
{"verdict": "pass"|"fail", "reason": "<one sentence>"}"""


class JudgeResult(BaseModel):
    verdict: Literal["pass", "fail"]
    reason: str


def judge(query: str, answer: str) -> JudgeResult | None:
    user = f"Question: {query}\n\nAnswer: {answer}"
    return call_json(call_groq, JUDGE_MODEL, SYSTEM_PROMPT, user, JudgeResult)
