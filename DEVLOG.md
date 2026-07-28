# Dev Log — what got built, in order, and why

A walkthrough of the project as it was built, task by task. For the "what broke and how I
fixed it" record, see `BUILD-LOG.md`. For the original idea/scope, see `idea.md`,
`learning-guide.md`, and the full spec at `docs/superpowers/specs/2026-07-28-ladder-llm-design.md`.

## Setup: spec, plan, git

Before writing code: turned the three source docs (`idea.md`, `learning-guide.md`, `review.md`)
into a single design spec, then a task-by-task implementation plan. Initialized git (this
started as a plain folder, no version control) with a `.gitignore` covering `.env` and
`__pycache__`. Every task below ends in its own commit.

## Task 1 — Project scaffolding

Set up `requirements.txt` (streamlit, groq, openai, pydantic, python-dotenv, requests),
created a venv, confirmed both API keys (already in `.env`) load correctly via a tiny
`checks/check_env.py`. `src/` is the package holding all real code; `checks/` holds one
small assert-based script per module — no pytest, just enough to catch a broken change.

## Task 2 — `llm_client.py`

The shared layer every other module calls through. Two ideas live here:

1. **Provider abstraction.** Groq has its own SDK. OpenRouter just exposes an
   OpenAI-compatible REST API, so the `openai` SDK talks to it by pointing `base_url` at
   `https://openrouter.ai/api/v1`. `call_model()` picks the right one based on a model's
   `provider` field so callers never need to know or care which provider a model lives on.
2. **Structured output via prompting.** Neither provider is used with a dedicated
   structured-output mode here — instead, the system prompt says "reply with ONLY this JSON
   shape," and `call_json()` strips markdown fences, validates the result against a Pydantic
   schema, and retries once with a stricter instruction if it fails. Every JSON-expecting
   caller (classifier, judge, cascade) reuses this one function.

`ModelUnavailable` is a custom exception raised when either provider returns 429/503 — this
is how the cascade later knows to skip a tier instead of treating an outage as a bad answer.

## Task 3 — `registry.py`

The tier x task-type model grid, as a plain dict — no class hierarchy, since a dict lookup
is all this needs. Before writing it, ran `checks/discover_models.py` to pull the *live*
model lists from both providers, and checked every entry from the original design grid
against that live list (all 20 turned out to still be current). Each `ModelConfig` records
`active_params_b` — active parameters per token, not total — which matters for
Mixture-of-Experts models like `openai/gpt-oss-120b` (~117B total, ~5.1B active) where using
the total would badly overstate how "big" a tier actually is.

## Task 4 — `classifier.py`

The entry point of the system. One cheap call to `llama-3.1-8b-instant` looks at the raw
query and returns three things: `difficulty` (easy/medium/hard/expert), `type` (qa/coding/
reasoning/summarization/translation), and `optimized_prompt` (the query rewritten to be
clearer before it's sent further down the pipeline). Tested against 5 hand-picked queries,
one per type — all 5 classified correctly on the first live run.

## Task 5 — `judge.py`

The LLM-as-judge piece, deliberately scoped narrow: it only ever returns `pass` or `fail`,
never a numeric score. A small model like `llama-3.1-8b-instant` can reliably say "does this
look right, yes or no" but isn't capable enough to meaningfully rate answer quality 1-10 —
so the design doesn't ask it to. It only ever fires for the ambiguous confidence band (see
Task 6); clear wins and clear losses skip it.

## Task 6 — `cascade.py`

The core waterfall loop, and the hardest piece to get right:

- **Starting tier / ceiling** depend on classified difficulty: easy/medium start at tier 1
  (ceiling tier 2); hard starts at tier 2 (ceiling tier 3); expert starts at tier 3 (ceiling
  tier 4). Harder queries get both a higher floor and a higher ceiling.
- **Confidence calibration**: the model self-rates its own answer 1-10 in the same JSON
  response as the answer itself. That number is only trusted at the extremes — ≥8 accepts
  immediately, ≤4 escalates immediately without spending a judge call. Only the murky 5-7
  band fires the judge, since self-reported confidence in the middle is known to be an
  unreliable signal on its own.
- **Every iteration is logged to a `TraceStep`** — tier, model, status, confidence, judge
  reason if any — which is what the UI later renders as the "how did we get this answer"
  trace.
- **`ModelUnavailable` is caught per-tier**, logged as `"unavailable"`, and the loop moves on
  — an outage never gets misread as a bad answer.

Hit a real, reproducible-looking failure here on the first test run (both tier 1 and tier 2
came back malformed) that turned out to be the known ~10-15% malformed-JSON rate from small
models, not a code bug — see `BUILD-LOG.md` for the actual debugging steps.

## Task 7 — `metrics.py` + `formatter.py`

`metrics.py` sums the *active* params actually burned across every real call in a trace
(skipping `unavailable` steps, since those never actually executed) and compares that
against the max-tier baseline for that query's type — this is the "honest" MoE-aware compute
savings number. `formatter.py` is pure presentation: wraps coding answers in a code fence if
the model didn't already, and turns multi-line translation answers into a small markdown
table.

## Task 8 — `app.py` (Streamlit UI)

Two-column layout: answer + compute-saved metric on the left, live routing trace on the
right. The one non-obvious Streamlit behavior baked in here: Streamlit reruns the *entire*
script on every interaction, including every keystroke — so the cascade call is gated behind
`if submit and query`, and the result is stashed in `st.session_state` so it survives that
rerun instead of disappearing or re-firing.

---
*(updated as later tasks — edge-case hardening, UI testing — get done)*
