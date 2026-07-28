import json
import os

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


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


def call_json(call_fn, model_id: str, system: str, user: str, schema: type[BaseModel]):
    """Call an LLM expecting JSON; validate against schema; retry once on failure."""
    for attempt in range(2):
        raw = call_fn(model_id, system, user)
        try:
            return schema.model_validate_json(_strip_fences(raw))
        except (ValidationError, json.JSONDecodeError):
            if attempt == 0:
                system = system + "\n\nReply with ONLY valid JSON, no other text."
    return None


def call_model(model_config, system: str, user: str, schema: type[BaseModel]):
    call_fn = call_groq if model_config.provider == "groq" else call_openrouter
    try:
        return call_json(call_fn, model_config.model_id, system, user, schema)
    except (GroqAPIStatusError, OpenAIAPIStatusError) as e:
        status = getattr(e, "status_code", None)
        if status in (429, 503):
            raise ModelUnavailable(f"{model_config.model_id} unavailable ({status})") from e
        raise
