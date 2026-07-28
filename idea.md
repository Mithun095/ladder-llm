# LadderLLM — Adaptive Multi-Tier LLM Cascade Router

## Problem

Every LLM query today gets routed to a single model chosen upfront — usually the largest one "just in case." A question like "what is a closure in Python?" does not need a 550B-parameter reasoning model. This project builds a system that starts at the smallest model that could plausibly answer a query and only escalates to larger models when the smaller one genuinely fails.

## What It Does

LadderLLM classifies each query along two dimensions — **difficulty** (easy / medium / hard / expert) and **task type** (coding / reasoning / general Q&A / summarization / translation) — then routes it through a ladder of free model tiers, escalating only when confidence is low or a judge model flags the answer as insufficient. The final output shows exactly which models were tried, how many iterations it took, and how much compute was saved versus always using the largest model.

## Why It's Resume-Worthy

- Demonstrates **cost-aware system design** — a real, quantified optimization (up to 94% compute saved on easy queries)
- Shows **multi-provider integration** (Groq + OpenRouter) under a single unified registry abstraction
- Uses the **LLM-as-judge pattern** for automated quality evaluation — a technique now standard in production LLM pipelines
- Implements a **prompt compiler** step: restructures raw user input into typed JSON before routing, reducing downstream token consumption
- Entirely free to run — no GPU, no paid API credits

## Architecture

```
User query
  → Classifier (difficulty × type + prompt optimization)
  → Model Registry (tier × type → specific model)
  → Cascade Runner (waterfall loop with confidence + judge escalation)
  → Streamlit UI (answer panel + live routing trace + compute savings)
```

## Tier × Task Type Model Grid (all free)

| Tier | General Q&A | Coding | Reasoning/Math | Summarization | Translation |
|---|---|---|---|---|---|
| **1 — Nano** | Groq `llama-3.1-8b-instant` | OR `cohere/north-mini-code:free` | Groq `llama-3.1-8b-instant` | Groq `llama-3.1-8b-instant` | Groq `llama-3.1-8b-instant` |
| **2 — Small** | Groq `qwen/qwen3.6-27b` | OR `poolside/laguna-xs-2.1:free` | Groq `qwen/qwen3.6-27b` | Groq `llama-3.3-70b-versatile` | OR `google/gemma-4-26b-a4b-it:free` |
| **3 — Large** | Groq `openai/gpt-oss-120b` | OR `poolside/laguna-m.1:free` | OR `nvidia/nemotron-3-super-120b-a12b:free` | Groq `openai/gpt-oss-120b` | Groq `openai/gpt-oss-120b` |
| **4 — Max** | OR `nvidia/nemotron-3-ultra-550b-a55b:free` | OR `nvidia/nemotron-3-ultra-550b-a55b:free` | OR `nvidia/nemotron-3-ultra-550b-a55b:free` | OR `nvidia/nemotron-3-ultra-550b-a55b:free` | OR `nvidia/nemotron-3-ultra-550b-a55b:free` |

**OR = OpenRouter | Groq = Groq free tier**

## Escalation Logic

Starting tier is always one below the classified difficulty:

| Classified | Starts at | Escalates to |
|---|---|---|
| Easy | Tier 1 | Tier 2 (ceiling) |
| Medium | Tier 1 | Tier 1 → Tier 2 |
| Hard | Tier 2 | Tier 2 → Tier 3 |
| Expert | Tier 3 | Tier 3 → Tier 4 |

Per iteration: model answers + self-rates confidence (1–10).
- ≥ 8 → accept
- ≤ 4 → escalate immediately
- 5–7 → fire judge (Groq `llama-3.1-8b-instant`) → pass/fail → accept or escalate

## Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Groq models | `groq` Python SDK |
| OpenRouter models | `openai` SDK (OpenRouter is OpenAI-compatible) |
| Classifier output validation | `pydantic` v2 |
| Environment | `python-dotenv` |
| Python | 3.11+ |

## What You Need to Run It

- Free Groq API key: [console.groq.com](https://console.groq.com) (no credit card)
- Free OpenRouter API key: [openrouter.ai](https://openrouter.ai) (no credit card)
- Add both to `.env` — see `.env.example`
- `pip install -r requirements.txt && streamlit run src/app.py`

## Scope

2–3 weeks, solo, part-time. See `learning-guide.md` for the phased build plan.
Full design spec: `docs/superpowers/specs/ladder-llm-design.md`
