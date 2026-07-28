# Learning Guide: LadderLLM

## Concepts to Learn

1. **LLM confidence calibration** — language models are notoriously overconfident. Self-reported confidence scores (asking the model "rate your confidence 1–10") are a weak signal but cheap to collect. The key insight: use it only to detect the extremes (clearly confident ≥ 8 vs. clearly unsure ≤ 4) and fire a separate judge for the murky middle.

2. **LLM-as-judge pattern** — using one model to evaluate another's output. Understand its failure modes: judge models share the same overconfidence bias; a small judge (llama-3.1-8b) may not be capable enough to evaluate complex reasoning outputs. In this project the judge is used as a binary pass/fail filter, not a nuanced scorer — that's the correct scope for a small judge model.

3. **Structured output via prompt engineering** — reliably extracting JSON from a model response without using a dedicated structured-output API (Groq and OpenRouter expose this differently). Pattern: explicit JSON schema in the system prompt + Pydantic validation with one retry. Know when to use `response_format={"type": "json_object"}` vs. freeform parsing.

4. **Multi-provider API integration** — Groq uses its own Python SDK; OpenRouter uses an OpenAI-compatible endpoint (`base_url="https://openrouter.ai/api/v1"`). The registry abstraction hides this: callers don't need to know which provider a model lives on.

5. **Streamlit session state and live updates** — `st.session_state` for persisting the routing trace across reruns; `st.status()` and `st.empty()` for live iteration-by-iteration trace updates. Understand that Streamlit reruns the entire script on each interaction — the cascade runner must be called inside a single execution path, not across reruns.

6. **MoE (Mixture of Experts) model sizing** — Nemotron Ultra is 550B total parameters but only ~55B active per token. The honest compute savings metric uses active params, not total. Know the difference when talking about this in an interview.

## Phased Weekly Build Plan (~2.5–3 weeks part-time)

### Week 1: Core pipeline (no UI)

- **Days 1–2:** Set up project, install deps, confirm both API keys work. Write `registry.py` — the model grid as a plain dict. Write one test: assert every (tier, type) pair resolves to a valid model config.
- **Days 3–4:** Write `classifier.py`. Prompt `llama-3.1-8b-instant` to return `{difficulty, type, optimized_prompt}` as JSON. Validate with Pydantic. Handle the one-retry path for malformed JSON. Test with 5–10 hand-crafted queries across all 5 types.
- **Day 5:** Write `judge.py`. Prompt `llama-3.1-8b-instant` to return `{verdict: "pass"|"fail", reason: str}`. Test in isolation: given a good answer and a bad answer, does the judge reliably distinguish them?

### Week 2: Cascade runner + metrics

- **Days 6–8:** Write `cascade.py`. Implement the waterfall loop: call model → parse confidence → fire judge if borderline → escalate or accept. Hardcode a simple test query, run it end-to-end in a Python script (no UI yet), print the trace to stdout. This is the hardest part — get it working before touching the UI.
- **Days 9–10:** Write `metrics.py`. Compute `compute_saved_pct` and `total_params_burned`. Write `formatter.py` — format answer by task type (code block, bullet list, side-by-side table for translation, plain markdown for general).

### Week 3: Streamlit UI + polish

- **Days 11–13:** Wire everything into `app.py`. Two-column layout: answer left, trace right. Trace panel updates live after each iteration using `st.status`. Input box + submit button at top.
- **Days 14–15:** Edge cases — what happens when a `:free` OpenRouter model returns 503? What if all tiers return low confidence? Test with deliberately hard, deliberately easy, and deliberately ambiguous queries. Add `.env.example`. Final QA pass.

## What Commonly Goes Wrong

- **OpenRouter `:free` models return 503 or time out without warning.** Free models on OpenRouter are served on spare capacity — they go down. The cascade runner must catch `openai.APIStatusError` with status 503/429, log it to the trace as "model unavailable — skipping tier," and escalate. Do not let an unavailable model look like a low-confidence answer.

- **Self-reported confidence is always 8–9, even for wrong answers.** Models optimised for helpfulness tend to project confidence. When this happens in testing, lower the accept threshold (try ≥ 9 instead of ≥ 8) or disable self-report entirely and always fire the judge. The judge-always path is slower but more reliable — have it as a fallback mode.

- **Classifier JSON is malformed ~10–15% of the time with small models.** `llama-3.1-8b-instant` will occasionally return the JSON wrapped in markdown code fences, or with extra commentary before/after. Strip fences before parsing, and on `json.JSONDecodeError` retry once with a stricter system prompt: "Reply with ONLY valid JSON, no other text." If it fails twice, default to `difficulty=medium, type=general` and continue.

- **Streamlit reruns the whole script on every user interaction.** If the cascade runner is called at the module level, it re-runs on every keystroke in the input box. Fix: gate the cascade call behind `if st.button("Submit")` and store results in `st.session_state` so they persist across the submit-triggered rerun.

- **Compute savings metric looks wrong for Tier 3 > Tier 4 active params.** Poolside Laguna M.1 (Tier 3 coding) is a ~32B active parameter model; Nemotron Ultra Tier 4 has ~55B active. If a hard coding query resolves at Tier 3, the math shows 32B < 55B — technically savings vs. Tier 4, but only 42%. Display the raw numbers ("32B active used, 55B max") rather than percentage-only, so the user understands what they're seeing.

- **Translation at Tier 1 uses a general model with no multilingual specialization.** `llama-3.1-8b-instant` handles common language pairs (EN↔ES, EN↔FR, EN↔DE) reasonably well but will struggle with less common pairs (EN↔HI, EN↔AR, EN↔ZH). This is acceptable for a demo — document it in the UI as a known limitation rather than trying to hide it.
