from typing import Literal

from pydantic import BaseModel

from src.llm_client import ModelUnavailable, call_groq, call_json

# Was llama-3.1-8b-instant, chosen because the classifier is a cheap pre-step and the 8B model is
# the cheapest thing on Groq. That reasoning ignored what a classifier error actually costs: the
# *type* selects an entire model ladder, so misreading a QA question as a coding one routes it to
# a different provider — and when that provider's daily quota is gone, the query fails outright
# with nothing to show. It is the most consequential single call in the system.
#
# Measured both against the benchmark's own type labels (25 queries x 2 repeats). This is ground
# truth, not judge verdicts, so unlike most numbers in this project it isn't circular:
#
#                          type accuracy   type stable   difficulty stable
#   llama-3.1-8b-instant       94% (47/50)     24/25          15/25
#   openai/gpt-oss-120b       100% (50/50)     25/25          22/25
#
# The difficulty column is the one that showed up in use: the same query classified easy, medium,
# hard and expert across repeated runs, changing both its entry tier and its ceiling each time.
#
# Costs ~1.8x more per call ($0.037 vs $0.021 per 1k) and adds ~250ms median latency (711ms vs
# 461ms), against a multi-second end-to-end time. Worth it to stop routing queries into a ladder
# that can't answer them. See BUILD-LOG.md #23.
CLASSIFIER_MODEL = "openai/gpt-oss-120b"

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
