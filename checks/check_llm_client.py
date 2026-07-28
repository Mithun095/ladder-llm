from pydantic import BaseModel

from src.llm_client import call_groq, call_json


class Greeting(BaseModel):
    message: str


raw = call_groq("llama-3.1-8b-instant", "Reply with plain text.", "Say hello in 3 words.")
assert isinstance(raw, str) and len(raw) > 0, "call_groq returned nothing"

result = call_json(
    call_groq,
    "llama-3.1-8b-instant",
    'Reply with ONLY JSON: {"message": "<a 3 word greeting>"}',
    "Greet me.",
    Greeting,
)
assert result is not None, "call_json failed to parse valid JSON request"
assert isinstance(result.message, str)
print("llm_client checks passed. Sample:", result.message)
