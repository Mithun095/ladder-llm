import json
import os
import time

from dotenv import load_dotenv
from groq import Groq
from groq import APIStatusError as GroqAPIStatusError
from openai import OpenAI
from openai import APIStatusError as OpenAIAPIStatusError
from pydantic import BaseModel, ValidationError

load_dotenv()

_groq = Groq(api_key=os.environ["GROQ_API_KEY"])
_openrouter = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)


class ModelUnavailable(Exception):
    pass


def call_groq(model_id: str, system: str, user: str) -> str:
    resp = _groq.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    if not resp.choices:
        return ""
    return resp.choices[0].message.content or ""


def call_openrouter(model_id: str, system: str, user: str) -> str:
    resp = _openrouter.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    if not resp.choices:
        return ""
    return resp.choices[0].message.content or ""


_DECODER = json.JSONDecoder()


def _extract_json(text: str) -> str:
    """Pull the model's real JSON object out of whatever it wrapped it in.

    Small models bury the JSON three different ways, sometimes at once: markdown code
    fences, a reasoning block (`<think>...</think>`) ahead of the answer, and — inside that
    reasoning — draft copies of the JSON it's about to emit. Rather than pattern-match each
    wrapper, scan for every position where a *complete* JSON object starts, skip past each
    one found, and keep the last top-level object. The real answer is always the last one
    emitted; skipping past a parsed object is what keeps a nested `{...}` (or a brace inside
    an answer string, e.g. a Python snippet) from being mistaken for the start of a new one.
    """
    best = None
    i = 0
    while (i := text.find("{", i)) != -1:
        try:
            _, end = _DECODER.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        best = text[i:end]
        i = end
    return best if best is not None else text.strip()


RETRY_WAIT_S = 2.0


def call_json(call_fn, model_id: str, system: str, user: str, schema: type[BaseModel]):
    """Call an LLM expecting JSON and validate the result against `schema`.

    Handles the two unrelated ways this fails, each with one retry:

    - **Provider says 429/503.** Groq's per-minute rate limit clears in seconds ("try again
      in 2s"), so one short wait recovers it outright. If the second attempt also fails, the
      outage is real (e.g. OpenRouter's *daily* free-tier cap, which no wait will fix) and
      `ModelUnavailable` is raised so the cascade can skip this tier.
    - **Model returns something that isn't valid JSON.** Retry once with a stricter
      instruction appended, then give up and return None so the caller can escalate.

    This lives here, not in `call_model()`, because the classifier and the judge call
    `call_json` directly — putting the 429 handling one layer up left both of them able to
    crash the whole app on a rate limit that the answering path handled gracefully.
    """
    for attempt in range(2):
        try:
            raw = call_fn(model_id, system, user)
        except (GroqAPIStatusError, OpenAIAPIStatusError) as e:
            status = getattr(e, "status_code", None)
            if status not in (429, 503):
                raise  # e.g. a 400 means the registry has a bad model ID — a real bug, crash loudly
            if attempt == 0:
                time.sleep(RETRY_WAIT_S)
                continue
            raise ModelUnavailable(f"{model_id} unavailable ({status})") from e

        try:
            return schema.model_validate_json(_extract_json(raw))
        except (ValidationError, json.JSONDecodeError):
            if attempt == 0:
                system = system + "\n\nReply with ONLY valid JSON, no other text."
    return None


def call_model(model_config, system: str, user: str, schema: type[BaseModel]):
    call_fn = call_groq if model_config.provider == "groq" else call_openrouter
    return call_json(call_fn, model_config.model_id, system, user, schema)
