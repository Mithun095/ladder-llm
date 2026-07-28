from typing import Literal

from pydantic import BaseModel

from src.llm_client import ModelUnavailable, call_groq, call_json

CLASSIFIER_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are a query classifier. Given a user query, respond with ONLY this JSON \
shape, no other text:
{"difficulty": "easy"|"medium"|"hard"|"expert", "type": "qa"|"coding"|"reasoning"|"summarization"|"translation", "optimized_prompt": "<the query, rewritten to be clear and unambiguous>"}

type guidance:
- "coding": the user wants code written, debugged, or reviewed (e.g. "write a function that...").
- "qa": the user wants a concept explained, even if the concept is about programming \
(e.g. "what is a closure in Python?" is qa, not coding — no code is being requested).
- "reasoning": math, logic, or proofs.
- "summarization": condensing given text.
- "translation": converting text between languages.

optimized_prompt guidance:
- For "translation" queries, keep the instruction explicit: "Translate '<exact source text>' \
to <language>." Do not rephrase it into a plain question — that erases the instruction to \
translate and makes the downstream model try to answer it instead.
- For other types, rewrite for clarity as needed."""


class ClassifierResult(BaseModel):
    difficulty: Literal["easy", "medium", "hard", "expert"]
    type: Literal["qa", "coding", "reasoning", "summarization", "translation"]
    optimized_prompt: str


def classify(query: str) -> ClassifierResult:
    # The classifier is a hard dependency of every query, so it can't be allowed to fail the
    # request: if the model is rate-limited or returns junk, fall back to a neutral
    # medium/qa classification with the query passed through untouched. That routes to a
    # middle tier and still answers, rather than crashing before any model is even tried.
    try:
        result = call_json(call_groq, CLASSIFIER_MODEL, SYSTEM_PROMPT, query, ClassifierResult)
    except ModelUnavailable:
        result = None
    if result is None:
        return ClassifierResult(difficulty="medium", type="qa", optimized_prompt=query)
    return result
