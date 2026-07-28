"""Provider-failure paths, exercised with mocks — no live API calls, runs in CI.

Every one of these is a failure that actually happened during this project. They're mocked
rather than live because you can't ask a provider to rate-limit you on demand, and because a
check that only passes when the free tier happens to be healthy isn't a check.
"""
import httpx
from openai import APIStatusError as OpenAIAPIStatusError
from pydantic import BaseModel

from src import llm_client
from src.llm_client import ModelUnavailable, call_json


class Shape(BaseModel):
    message: str


llm_client.RETRY_WAIT_S = 0  # don't actually sleep through the retry in a test


def _status_error(code: int) -> OpenAIAPIStatusError:
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx.Response(code, request=request)
    return OpenAIAPIStatusError("boom", response=response, body=None)


def _raises(code: int):
    def call_fn(model_id, system, user):
        raise _status_error(code)
    return call_fn


def _fails_then(code: int, then: str):
    calls = {"n": 0}

    def call_fn(model_id, system, user):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _status_error(code)
        return then
    return call_fn, calls


# 1. A sustained 429/503 becomes ModelUnavailable, so the cascade can skip the tier.
for code in (429, 503):
    try:
        call_json(_raises(code), "m", "s", "u", Shape)
        raise AssertionError(f"{code} should have raised ModelUnavailable")
    except ModelUnavailable:
        print(f"  ok  sustained {code} -> ModelUnavailable")

# 2. A *transient* 429 (Groq's per-minute limit, clears in seconds) recovers on the retry
#    instead of taking the tier down.
call_fn, calls = _fails_then(429, '{"message": "recovered"}')
result = call_json(call_fn, "m", "s", "u", Shape)
assert result is not None and result.message == "recovered", result
assert calls["n"] == 2, f"expected exactly one retry, got {calls['n']} calls"
print("  ok  transient 429 -> retried and recovered")

# 3. A 400 is NOT swallowed: it means the registry holds a bad model ID, which is a real bug
#    that should crash loudly rather than be misreported as a routine outage.
try:
    call_json(_raises(400), "m", "s", "u", Shape)
    raise AssertionError("400 should propagate, not be caught")
except OpenAIAPIStatusError:
    print("  ok  400 propagates (bad model ID is a bug, not an outage)")
except ModelUnavailable:
    raise AssertionError("400 was wrongly treated as an outage")

# 4. Unparseable output retries once with a stricter instruction, then returns None so the
#    caller escalates — it must not raise.
assert call_json(lambda *a: "I'm afraid I can't do that.", "m", "s", "u", Shape) is None
print("  ok  unparseable output -> None, no exception")

# 5. Empty/missing content must not crash the client wrappers. Both of these were real
#    TypeErrors: `.content` was None once, then `.choices` itself was None.
class _Resp:
    def __init__(self, choices):
        self.choices = choices


class _Choice:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})()


for name, resp in (("choices=None", _Resp(None)), ("content=None", _Resp([_Choice(None)]))):
    llm_client._groq.chat.completions.create = lambda **kw: resp
    assert llm_client.call_groq("m", "s", "u") == "", name
    print(f"  ok  {name} -> empty string, no crash")

print("error handling check passed.")
