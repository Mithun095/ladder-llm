# LadderLLM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build LadderLLM — a cascade router that classifies a query, starts at the cheapest plausible free LLM tier, and escalates only on low confidence or judge failure, with a Streamlit UI showing the live trace.

**Architecture:** Classifier tags (difficulty, type) and rewrites the prompt → Registry maps (tier, type) to a specific Groq/OpenRouter model → Cascade Runner waterfalls through tiers, calling a shared `llm_client` that hides the provider difference and validates JSON output against Pydantic schemas → Streamlit UI renders the answer and trace.

**Tech Stack:** Python 3.11+, `groq` SDK, `openai` SDK (OpenRouter, OpenAI-compatible), `pydantic` v2, `python-dotenv`, `streamlit`.

## Global Constraints

- No pytest suite — one small assert-based check script per module under `checks/`, matching the spec's testing approach.
- `registry.py` is a plain dict, no class hierarchy.
- Model IDs in `idea.md`'s grid must be verified against live Groq/OpenRouter model lists before being hardcoded (Task 3) — substitute dead IDs with the closest live free equivalent at that tier/type.
- Catch `APIStatusError` 429/503 from both providers and treat as "tier unavailable," never as low confidence.
- `.env` already has both API keys set (`GROQ_API_KEY`, `OPENROUTER_API_KEY`) — never commit it (already in `.gitignore`).
- Full spec: `docs/superpowers/specs/2026-07-28-ladder-llm-design.md`.

---

## Task 1: Project scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `src/__init__.py`
- Create: `checks/check_env.py`

**Interfaces:**
- Produces: a working venv with all deps installed; confirms `.env` loads correctly for every later task.

- [ ] **Step 1: Write `requirements.txt`**

```
streamlit
groq
openai
pydantic>=2
python-dotenv
```

- [ ] **Step 2: Create venv and install**

Run: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
Expected: all 5 packages (and deps) install cleanly.

- [ ] **Step 3: Write `.env.example`**

```
GROQ_API_KEY=your-groq-key-here
OPENROUTER_API_KEY=your-openrouter-key-here
```

- [ ] **Step 4: Create `src/__init__.py`** (empty file, makes `src` a package)

- [ ] **Step 5: Write `checks/check_env.py`**

```python
import os
from dotenv import load_dotenv

load_dotenv()

assert os.environ.get("GROQ_API_KEY"), "GROQ_API_KEY missing from .env"
assert os.environ.get("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY missing from .env"
print("Both API keys present.")
```

- [ ] **Step 6: Run the check**

Run: `python checks/check_env.py`
Expected: `Both API keys present.`

- [ ] **Step 7: Commit**

```bash
git add requirements.txt .env.example src/__init__.py checks/check_env.py
git commit -m "Scaffold project: deps, env check, package layout"
```

---

## Task 2: `llm_client.py` — shared provider abstraction

**Files:**
- Create: `src/llm_client.py`
- Test: `checks/check_llm_client.py`

**Interfaces:**
- Produces:
  - `call_groq(model_id: str, system: str, user: str) -> str`
  - `call_openrouter(model_id: str, system: str, user: str) -> str`
  - `call_json(call_fn, model_id: str, system: str, user: str, schema: type[BaseModel]) -> BaseModel | None` — retries once on invalid JSON, returns `None` if both attempts fail.
  - `ModelUnavailable(Exception)` — raised when a provider returns 429/503.
  - `call_model(model_config, system: str, user: str, schema: type[BaseModel]) -> BaseModel | None` — dispatches to `call_groq`/`call_openrouter` by `model_config.provider`, wraps `call_json`, raises `ModelUnavailable` on 429/503.
- Consumes: nothing (leaf module, only stdlib + SDKs + pydantic).

- [ ] **Step 1: Write `src/llm_client.py`**

```python
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
    return resp.choices[0].message.content


def call_openrouter(model_id: str, system: str, user: str) -> str:
    resp = _openrouter.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content


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
```

- [ ] **Step 2: Write `checks/check_llm_client.py`**

```python
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
```

- [ ] **Step 3: Run it**

Run: `python -m checks.check_llm_client`
Expected: `llm_client checks passed. Sample: <something>`. If it fails on JSON parsing, print `raw` to see what Groq actually returned and adjust `_strip_fences` if needed.

- [ ] **Step 4: Commit**

```bash
git add src/llm_client.py checks/check_llm_client.py
git commit -m "Add llm_client: provider abstraction + JSON retry helper"
```

---

## Task 3: `registry.py` — verify live models, build the grid

**Files:**
- Create: `checks/discover_models.py` (throwaway discovery script, not a permanent check)
- Create: `src/registry.py`
- Test: `checks/check_registry.py`

**Interfaces:**
- Produces:
  - `ModelConfig` (dataclass): `provider: str` (`"groq"|"openrouter"`), `model_id: str`, `active_params_b: float`.
  - `REGISTRY: dict[tuple[int, str], ModelConfig]` keyed by `(tier: int, task_type: str)`.
  - `MAX_TIER: int = 4`
  - `TASK_TYPES: tuple[str, ...] = ("qa", "coding", "reasoning", "summarization", "translation")`
  - `get_model(tier: int, task_type: str) -> ModelConfig`
- Consumes: nothing directly, but the grid must only contain model IDs confirmed live in Step 1.

- [ ] **Step 1: Discover live free models**

Write `checks/discover_models.py`:

```python
import os

import requests
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

print("=== Groq models ===")
groq_models = Groq(api_key=os.environ["GROQ_API_KEY"]).models.list()
for m in groq_models.data:
    print(m.id)

print("\n=== OpenRouter free models ===")
resp = requests.get("https://openrouter.ai/api/v1/models")
for m in resp.json()["data"]:
    if m["id"].endswith(":free"):
        print(m["id"])
```

Run: `python checks/discover_models.py`

Compare the printed lists against `idea.md`'s tier grid (lines 33-36). For each cell, confirm the exact ID is present; if not, pick the closest live `:free` (OpenRouter) or free-tier (Groq) model for that tier's rough size class. Write down the final grid before Step 2 — this is what goes into `REGISTRY`.

This script is exploratory, not part of the permanent check suite — do not add it to `checks/check_registry.py`'s assertions.

- [ ] **Step 2: Write `src/registry.py`**

Use the model IDs confirmed in Step 1. Template (replace any ID that Step 1 showed as dead, and fill in real `active_params_b` — for MoE models like Nemotron Ultra use *active* params, not total, per the spec):

```python
from dataclasses import dataclass

TASK_TYPES = ("qa", "coding", "reasoning", "summarization", "translation")
MAX_TIER = 4


@dataclass(frozen=True)
class ModelConfig:
    provider: str  # "groq" | "openrouter"
    model_id: str
    active_params_b: float


REGISTRY: dict[tuple[int, str], ModelConfig] = {
    (1, "qa"): ModelConfig("groq", "llama-3.1-8b-instant", 8),
    (1, "coding"): ModelConfig("groq", "llama-3.1-8b-instant", 8),
    (1, "reasoning"): ModelConfig("groq", "llama-3.1-8b-instant", 8),
    (1, "summarization"): ModelConfig("groq", "llama-3.1-8b-instant", 8),
    (1, "translation"): ModelConfig("groq", "llama-3.1-8b-instant", 8),
    # Fill tiers 2-4 for all 5 types using Step 1's confirmed live IDs.
    # ...
}


def get_model(tier: int, task_type: str) -> ModelConfig:
    config = REGISTRY.get((tier, task_type))
    if config is None:
        raise KeyError(f"No model registered for tier={tier}, type={task_type}")
    return config
```

Fill in every `(tier, type)` pair for tiers 1-4 — 20 entries total. This step requires the real output from Step 1; do not guess IDs.

- [ ] **Step 3: Write `checks/check_registry.py`**

```python
from src.registry import MAX_TIER, TASK_TYPES, get_model

for tier in range(1, MAX_TIER + 1):
    for task_type in TASK_TYPES:
        config = get_model(tier, task_type)
        assert config.provider in ("groq", "openrouter")
        assert config.model_id
        assert config.active_params_b > 0

print(f"registry check passed: {MAX_TIER * len(TASK_TYPES)} pairs resolved.")
```

- [ ] **Step 4: Run it**

Run: `python -m checks.check_registry`
Expected: `registry check passed: 20 pairs resolved.`

- [ ] **Step 5: Commit**

```bash
git add src/registry.py checks/check_registry.py checks/discover_models.py
git commit -m "Add model registry, verified against live Groq/OpenRouter catalogs"
```

---

## Task 4: `classifier.py`

**Files:**
- Create: `src/classifier.py`
- Test: `checks/check_classifier.py`

**Interfaces:**
- Consumes: `call_json`, `call_groq` from `src/llm_client.py`.
- Produces:
  - `ClassifierResult` (pydantic `BaseModel`): `difficulty: Literal["easy","medium","hard","expert"]`, `type: Literal["qa","coding","reasoning","summarization","translation"]`, `optimized_prompt: str`.
  - `classify(query: str) -> ClassifierResult` — never returns `None`; falls back to `difficulty="medium", type="qa", optimized_prompt=query` if the LLM call fails twice.

- [ ] **Step 1: Write `src/classifier.py`**

```python
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
```

- [ ] **Step 2: Write `checks/check_classifier.py`**

```python
from src.classifier import classify

samples = [
    ("What is a closure in Python?", "qa"),
    ("Write a function to reverse a linked list in O(1) space", "coding"),
    ("Prove that the square root of 2 is irrational", "reasoning"),
    ("Summarize: The quick brown fox jumps over the lazy dog repeatedly.", "summarization"),
    ("Translate 'good morning' to French", "translation"),
]

for query, expected_type in samples:
    result = classify(query)
    print(f"{query!r} -> difficulty={result.difficulty}, type={result.type}")
    assert result.type == expected_type, f"expected type={expected_type}, got {result.type}"
    assert result.difficulty in ("easy", "medium", "hard", "expert")
    assert result.optimized_prompt

print("classifier check passed.")
```

- [ ] **Step 3: Run it**

Run: `python -m checks.check_classifier`
Expected: 5 lines of classification output, then `classifier check passed.` If a type assertion fails, it's a real signal about prompt quality — tighten `SYSTEM_PROMPT` (e.g., add one example per type) rather than loosening the assertion.

- [ ] **Step 4: Commit**

```bash
git add src/classifier.py checks/check_classifier.py
git commit -m "Add classifier: difficulty x type tagging + prompt optimization"
```

---

## Task 5: `judge.py`

**Files:**
- Create: `src/judge.py`
- Test: `checks/check_judge.py`

**Interfaces:**
- Consumes: `call_json`, `call_groq` from `src/llm_client.py`.
- Produces:
  - `JudgeResult` (pydantic `BaseModel`): `verdict: Literal["pass","fail"]`, `reason: str`.
  - `judge(query: str, answer: str) -> JudgeResult | None` — returns `None` if the LLM call fails twice (caller decides the fallback; see cascade.py Task 6, which treats `None` as `fail`).

- [ ] **Step 1: Write `src/judge.py`**

```python
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
```

- [ ] **Step 2: Write `checks/check_judge.py`**

```python
from src.judge import judge

good = judge("What is 2 + 2?", "4")
assert good is not None
print("good answer verdict:", good.verdict, "-", good.reason)
assert good.verdict == "pass"

bad = judge("What is 2 + 2?", "The capital of France is Paris.")
assert bad is not None
print("bad answer verdict:", bad.verdict, "-", bad.reason)
assert bad.verdict == "fail"

print("judge check passed.")
```

- [ ] **Step 3: Run it**

Run: `python -m checks.check_judge`
Expected: both verdicts printed, ending in `judge check passed.`

- [ ] **Step 4: Commit**

```bash
git add src/judge.py checks/check_judge.py
git commit -m "Add judge: binary pass/fail answer evaluation"
```

---

## Task 6: `cascade.py` — the waterfall loop

**Files:**
- Create: `src/cascade.py`
- Test: `checks/check_cascade.py`

**Interfaces:**
- Consumes:
  - `classify` from `src/classifier.py`
  - `judge` from `src/judge.py`
  - `call_model`, `ModelUnavailable` from `src/llm_client.py`
  - `get_model` from `src/registry.py`
- Produces:
  - `TraceStep` (dataclass): `tier: int`, `model_id: str`, `status: Literal["accepted","escalated","judged_fail","unavailable","malformed_response"]`, `confidence: int | None = None`, `judge_reason: str | None = None`, `active_params_b: float = 0`.
  - `CascadeResult` (dataclass): `answer: str`, `trace: list[TraceStep]`, `tier_used: int`, `type: str`.
  - `run_cascade(query: str) -> CascadeResult`

- [ ] **Step 1: Write `src/cascade.py`**

```python
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from src.classifier import classify
from src.judge import judge
from src.llm_client import ModelUnavailable, call_model
from src.registry import get_model

ANSWER_SYSTEM_PROMPT = """Answer the user's question. Respond with ONLY this JSON shape, no \
other text:
{"answer": "<your answer>", "confidence": <integer 1-10, your honest confidence this answer is correct>}"""

STARTING_TIER = {"easy": 1, "medium": 1, "hard": 2, "expert": 3}
CEILING_TIER = {"easy": 2, "medium": 2, "hard": 3, "expert": 4}


class AnswerResult(BaseModel):
    answer: str
    confidence: int = Field(ge=1, le=10)


@dataclass
class TraceStep:
    tier: int
    model_id: str
    status: Literal["accepted", "escalated", "judged_fail", "unavailable", "malformed_response"]
    confidence: int | None = None
    judge_reason: str | None = None
    active_params_b: float = 0


@dataclass
class CascadeResult:
    answer: str
    trace: list[TraceStep] = field(default_factory=list)
    tier_used: int = 0
    type: str = "qa"


def run_cascade(query: str) -> CascadeResult:
    classification = classify(query)
    trace: list[TraceStep] = []
    tier = STARTING_TIER[classification.difficulty]
    ceiling = CEILING_TIER[classification.difficulty]
    last_answer = None

    while tier <= ceiling:
        model_config = get_model(tier, classification.type)
        try:
            result = call_model(model_config, ANSWER_SYSTEM_PROMPT, classification.optimized_prompt, AnswerResult)
        except ModelUnavailable:
            trace.append(TraceStep(tier, model_config.model_id, "unavailable"))
            tier += 1
            continue

        if result is None:
            trace.append(TraceStep(tier, model_config.model_id, "malformed_response"))
            tier += 1
            continue

        last_answer = result.answer

        if result.confidence >= 8:
            trace.append(TraceStep(tier, model_config.model_id, "accepted", result.confidence, active_params_b=model_config.active_params_b))
            break
        if result.confidence <= 4:
            trace.append(TraceStep(tier, model_config.model_id, "escalated", result.confidence, active_params_b=model_config.active_params_b))
            tier += 1
            continue

        verdict = judge(query, result.answer)
        if verdict is not None and verdict.verdict == "pass":
            trace.append(TraceStep(tier, model_config.model_id, "accepted", result.confidence, verdict.reason, model_config.active_params_b))
            break
        reason = verdict.reason if verdict is not None else "judge call failed"
        trace.append(TraceStep(tier, model_config.model_id, "judged_fail", result.confidence, reason, model_config.active_params_b))
        tier += 1

    final_tier = trace[-1].tier if trace else tier
    return CascadeResult(
        answer=last_answer or "No model produced a usable answer.",
        trace=trace,
        tier_used=final_tier,
        type=classification.type,
    )
```

- [ ] **Step 2: Write `checks/check_cascade.py`**

```python
from src.cascade import run_cascade

result = run_cascade("What is a closure in Python?")

print("Answer:", result.answer[:200])
print("Tier used:", result.tier_used)
print("Trace:")
for step in result.trace:
    print(f"  tier={step.tier} model={step.model_id} status={step.status} confidence={step.confidence}")

assert result.answer and result.answer != "No model produced a usable answer."
assert len(result.trace) >= 1
assert result.trace[-1].status in ("accepted", "judged_fail", "unavailable")

print("cascade check passed.")
```

- [ ] **Step 3: Run it**

Run: `python -m checks.check_cascade`
Expected: a real answer, a trace of at least one step, ending `cascade check passed.` If every tier comes back `judged_fail`, that's the "self-reported confidence always high" failure mode from the spec — note it, we'll address the judge-always fallback in Task 9's edge-case pass if it shows up consistently.

- [ ] **Step 4: Commit**

```bash
git add src/cascade.py checks/check_cascade.py
git commit -m "Add cascade runner: waterfall escalation with confidence + judge"
```

---

## Task 7: `metrics.py` + `formatter.py`

**Files:**
- Create: `src/metrics.py`
- Create: `src/formatter.py`
- Test: `checks/check_metrics_formatter.py`

**Interfaces:**
- Consumes: `TraceStep` from `src/cascade.py`, `get_model`/`MAX_TIER` from `src/registry.py`.
- Produces:
  - `total_active_params_burned(trace: list[TraceStep]) -> float`
  - `compute_saved_pct(trace: list[TraceStep], task_type: str) -> float`
  - `format_answer(answer: str, task_type: str) -> str`

- [ ] **Step 1: Write `src/metrics.py`**

```python
from src.cascade import TraceStep
from src.registry import MAX_TIER, get_model


def total_active_params_burned(trace: list[TraceStep]) -> float:
    return sum(step.active_params_b for step in trace if step.status != "unavailable")


def compute_saved_pct(trace: list[TraceStep], task_type: str) -> float:
    baseline = get_model(MAX_TIER, task_type).active_params_b
    used = total_active_params_burned(trace)
    return max(0.0, (baseline - used) / baseline * 100)
```

- [ ] **Step 2: Write `src/formatter.py`**

```python
def format_answer(answer: str, task_type: str) -> str:
    if task_type == "coding" and "```" not in answer:
        return f"```\n{answer}\n```"
    if task_type == "translation" and "\n" in answer.strip():
        lines = [line for line in answer.splitlines() if line.strip()]
        rows = "\n".join(f"| {line} |" for line in lines)
        return f"| Translation |\n|---|\n{rows}"
    return answer
```

- [ ] **Step 3: Write `checks/check_metrics_formatter.py`**

```python
from src.cascade import TraceStep
from src.formatter import format_answer
from src.metrics import compute_saved_pct, total_active_params_burned

trace = [
    TraceStep(tier=1, model_id="llama-3.1-8b-instant", status="escalated", confidence=3, active_params_b=8),
    TraceStep(tier=2, model_id="some-tier2-model", status="accepted", confidence=9, active_params_b=27),
]

burned = total_active_params_burned(trace)
assert burned == 35, f"expected 35, got {burned}"

pct = compute_saved_pct(trace, "qa")
assert 0 <= pct <= 100, f"pct out of range: {pct}"
print(f"burned={burned}B, saved={pct:.1f}%")

assert format_answer("def f(): pass", "coding").startswith("```")
assert format_answer("hello", "coding") == "```\nhello\n```" or "```" in format_answer("hello", "coding")

print("metrics/formatter check passed.")
```

- [ ] **Step 4: Run it**

Run: `python -m checks.check_metrics_formatter`
Expected: `burned=35.0B, saved=X.X%` then `metrics/formatter check passed.`

- [ ] **Step 5: Commit**

```bash
git add src/metrics.py src/formatter.py checks/check_metrics_formatter.py
git commit -m "Add compute-savings metrics and per-type answer formatting"
```

---

## Task 8: `app.py` — Streamlit UI

**Files:**
- Create: `src/app.py`

**Interfaces:**
- Consumes: `run_cascade` from `src/cascade.py`, `compute_saved_pct`/`total_active_params_burned` from `src/metrics.py`, `format_answer` from `src/formatter.py`.
- Produces: a runnable Streamlit app, no importable interface (leaf of the dependency graph).

- [ ] **Step 1: Write `src/app.py`**

```python
import streamlit as st

from src.cascade import run_cascade
from src.formatter import format_answer
from src.metrics import compute_saved_pct, total_active_params_burned

st.set_page_config(page_title="LadderLLM", layout="wide")
st.title("LadderLLM — Adaptive Multi-Tier LLM Cascade Router")

query = st.text_input("Ask a question")
submit = st.button("Submit")

if submit and query:
    with st.spinner("Routing through the cascade..."):
        result = run_cascade(query)
    st.session_state["result"] = result

if "result" in st.session_state:
    result = st.session_state["result"]
    answer_col, trace_col = st.columns(2)

    with answer_col:
        st.subheader("Answer")
        st.markdown(format_answer(result.answer, result.type))

        saved = compute_saved_pct(result.trace, result.type)
        burned = total_active_params_burned(result.trace)
        st.metric("Compute saved vs. max tier", f"{saved:.0f}%")
        st.caption(f"{burned:.0f}B active params burned this run (tier {result.tier_used} used)")

    with trace_col:
        st.subheader("Routing trace")
        for step in result.trace:
            icon = {"accepted": "✅", "escalated": "⬆️", "judged_fail": "❌", "unavailable": "⚠️", "malformed_response": "⚠️"}[step.status]
            line = f"{icon} Tier {step.tier} — `{step.model_id}` — {step.status}"
            if step.confidence is not None:
                line += f" (confidence {step.confidence})"
            st.write(line)
            if step.judge_reason:
                st.caption(f"Judge: {step.judge_reason}")
```

- [ ] **Step 2: Run it manually**

Run: `streamlit run src/app.py`
Expected: browser opens, submit a query (e.g. "What is a closure in Python?"), see an answer on the left and a trace on the right. Try a coding query and a translation query to see `format_answer` branch.

- [ ] **Step 3: Commit**

```bash
git add src/app.py
git commit -m "Add Streamlit UI: two-column answer + live trace"
```

---

## Task 9: Edge-case hardening pass

**Files:**
- Modify: `src/cascade.py` (only if Task 6/8 manual testing surfaced the always-high-confidence problem)
- Modify: `.env.example` (finalize if any new config emerged)

**Interfaces:**
- No new interfaces — this task validates existing ones under failure conditions.

- [ ] **Step 1: Test an OpenRouter 503/429 path**

Temporarily point a registry entry's `model_id` at a clearly invalid OpenRouter model ID (e.g. append garbage to a real free ID) via the Streamlit UI or a scratch script calling `run_cascade` directly. Confirm the trace shows `"unavailable"` for that tier and the cascade proceeds to the next tier rather than hanging or crashing. Revert the registry change afterward.

- [ ] **Step 2: Test the all-low-confidence path**

Run a genuinely hard/ambiguous query through the UI (e.g. an open-ended philosophy question routed as `type=qa`) and confirm that when every tier escalates or judge-fails up to the ceiling, `run_cascade` still returns the last real answer it got (not a crash, not `"No model produced a usable answer."` unless truly every tier was unavailable).

- [ ] **Step 3: Check for the confidence-always-high pattern**

Review the `confidence` values logged across Tasks 6-8's manual runs. If confidence was ≥ 8 on every single call regardless of answer quality (the failure mode the spec calls out), add a module-level `JUDGE_ALWAYS = False` flag to `src/cascade.py` and, when `True`, skip the `>= 8` fast-accept branch and always fire the judge. Flip it to `True` only if the pattern was actually observed — don't add this speculatively.

- [ ] **Step 4: Confirm `.env.example` is complete and `.env` was never staged**

Run: `git log --all --oneline -- .env`
Expected: no output (the file was never committed).

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "Edge-case hardening: verified unavailable-tier and low-confidence paths"
```
