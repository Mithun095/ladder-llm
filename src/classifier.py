from typing import Literal

from pydantic import BaseModel

from src.llm_client import call_groq, call_json

CLASSIFIER_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are a query classifier. Given a user query, respond with ONLY this JSON \
shape, no other text:
{"difficulty": "easy"|"medium"|"hard"|"expert", "type": "qa"|"coding"|"reasoning"|"summarization"|"translation", "optimized_prompt": "<the query, rewritten to be clear and unambiguous>"}"""


class ClassifierResult(BaseModel):
    difficulty: Literal["easy", "medium", "hard", "expert"]
    type: Literal["qa", "coding", "reasoning", "summarization", "translation"]
    optimized_prompt: str


def classify(query: str) -> ClassifierResult:
    result = call_json(call_groq, CLASSIFIER_MODEL, SYSTEM_PROMPT, query, ClassifierResult)
    if result is None:
        return ClassifierResult(difficulty="medium", type="qa", optimized_prompt=query)
    return result
