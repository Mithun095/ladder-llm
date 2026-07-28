# LadderLLM — Design Spec

Status: approved 2026-07-28. Source docs: `idea.md`, `learning-guide.md`, `review.md`.

## Problem

Every query today gets routed to one model chosen upfront, usually oversized "just in case."
LadderLLM starts at the smallest plausible model and escalates only when it demonstrably fails.

## Architecture

```
User query
  → Classifier (difficulty × type + prompt optimization)
  → Model Registry (tier × type → specific model)
  → Cascade Runner (waterfall loop with confidence + judge escalation)
  → Streamlit UI (answer panel + live routing trace + compute savings)
```

## Components

| File | Responsibility |
|---|---|
| `registry.py` | Tier × task-type → model config, as a plain dict. No class hierarchy. |
| `classifier.py` | Calls `llama-3.1-8b-instant`, returns `{difficulty, type, optimized_prompt}` validated via Pydantic. |
| `judge.py` | Calls `llama-3.1-8b-instant`, returns `{verdict: pass|fail, reason}`. Binary filter only, not a scorer. |
| `cascade.py` | Waterfall loop: call model → parse self-rated confidence → judge if borderline → escalate or accept. |
| `metrics.py` | `compute_saved_pct`, `total_params_burned` (active params, not total — matters for MoE models). |
| `formatter.py` | Formats answer by task type (code block / bullet list / table / markdown). |
| `app.py` | Streamlit UI: two-column layout, answer left, live trace right. |

## Model grid

As specified in `idea.md`, with one change to process: **before hardcoding `registry.py`, pull
the live model lists from Groq and OpenRouter and confirm every ID in the grid actually resolves.**
Model IDs in the source doc (e.g. `nvidia/nemotron-3-ultra-550b-a55b:free`) may have drifted since
it was written — swap any dead ID for the closest live free equivalent at that tier/type, and note
the substitution in `registry.py` as a comment.

## Escalation logic

Starting tier = one below classified difficulty (easy/medium start at Tier 1, hard at Tier 2,
expert at Tier 3). Per iteration, the model self-rates confidence 1–10:

- ≥ 8 → accept
- ≤ 4 → escalate immediately
- 5–7 → fire judge → pass/fail → accept or escalate

Ceiling per difficulty class as defined in `idea.md`'s escalation table.

## Error handling

- OpenRouter `:free` models can 503/429 on spare capacity. Catch `openai.APIStatusError`
  (status 429/503), log to trace as "model unavailable — skipping tier," escalate. Never treat
  unavailability as a low-confidence answer.
- Classifier JSON malformed (~10-15% with small models): strip markdown fences, retry once with
  a stricter system prompt. Fail twice → default to `difficulty=medium, type=general` and continue.
- If self-reported confidence is empirically always 8-9 regardless of correctness (expected small-model
  behavior — verify during Day 5 judge testing), add a judge-always fallback mode.
- Streamlit reruns the whole script per interaction — gate the cascade call behind
  `if st.button("Submit")`, persist results in `st.session_state`.

## Testing approach

No pytest suite. One small assert-based self-check per module, written alongside it:
- `registry.py`: assert every (tier, type) pair resolves to a valid model config.
- `classifier.py`: run 5-10 hand-crafted queries spanning all 5 task types, eyeball the JSON.
- `judge.py`: one clearly-good and one clearly-bad answer, confirm the verdict distinguishes them.
- `cascade.py`: one hardcoded end-to-end query, print the trace to stdout.

## Build order

Phased, one module at a time, matching `learning-guide.md`:
1. `registry.py` (with live model-ID verification first)
2. `classifier.py`
3. `judge.py`
4. `cascade.py`
5. `metrics.py` + `formatter.py`
6. `app.py` (Streamlit)
7. Edge-case pass (503s, all-low-confidence, ambiguous queries) + `.env.example`

## Out of scope

Persistence/database, auth, multi-user, deployment, paid model tiers. Single-session,
local-run, portfolio-scope project per `review.md`.
