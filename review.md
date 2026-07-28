# Review: LadderLLM

> Adversarial review to be completed before build starts. Scoring uses the same 6-criterion rubric as the other projects — see `REVIEW-CRITERIA.md`.

## Score (preliminary)

| Criterion | Score /5 | Notes |
|---|---|---|
| Resume impact | 4 | Compute optimization + multi-provider routing is immediately legible to ML engineers. Slightly less "wow" than K8s operator or SAM 3 because routing systems are well-known. |
| Current relevance | 5 | LLM cost optimization and cascade routing are active research/industry topics in 2026. Free model ecosystem (Groq, OpenRouter) is a live, current hook. |
| Learning depth | 4 | Forces real understanding of LLM confidence calibration, judge evaluation, provider API differences. The prompt compiler and tier registry add genuine systems design. |
| Feasibility (1–3 weeks) | 4 | Streamlit + two SDK integrations is manageable. Biggest risk: OpenRouter `:free` model availability is unpredictable. Mitigated by graceful fallback in cascade runner. |
| Uniqueness | 4 | Cascade routing is a known pattern, but the 2D classifier (difficulty × type) + prompt compiler + live compute savings display is a specific, non-generic angle. Few portfolio projects show this. |
| Societal value | 4 | Genuinely reduces compute waste. Makes advanced AI accessible without paid API keys. Honest about what it does — no inflated claims. |

**Total: 25 / 30**

## Verdict: keep as-is

No structural flaw. The two risks worth watching during build:
1. Self-reported confidence is unreliable — the judge layer must be implemented, not skipped.
2. OpenRouter `:free` models can be slow or unavailable — the cascade runner must handle 429/503 gracefully and skip to the next tier rather than hanging.
