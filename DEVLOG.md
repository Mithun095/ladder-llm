# Dev Log — what got built, in order, and why

A walkthrough of the project as it was built, task by task. For the "what broke and how I
fixed it" record, see `BUILD-LOG.md`. For the project overview and architecture, see
`README.md`; for interview-ready framing of all of this, see `INTERVIEW-PREP.md`.

## Setup: spec, plan, git

Before writing code: turned the original project brief (a problem statement, a phased
learning/build plan, and a pre-build scoring rubric) into a single design spec, then a
task-by-task implementation plan. Initialized git (this started as a plain folder, no version
control) with a `.gitignore` covering `.env` and `__pycache__`. Every task below ends in its
own commit.

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

Run it with `PYTHONPATH=<project root> streamlit run src/app.py` — plain `streamlit run
src/app.py` throws `ModuleNotFoundError: No module named 'src'`, because Streamlit launches
the file directly (like `python src/app.py`), which only adds `src/`'s own directory to the
import path, not the project root that `src.cascade` needs to resolve against. See
`BUILD-LOG.md` for the full explanation.

First real end-to-end test in the browser: "can you write a code to find weather the number
is prime or not" (typo and all) classified as `coding`, routed to tier 1's coding model
(`cohere/north-mini-code:free`), accepted immediately at confidence 10, code rendered inside
a fenced block by `format_answer`, 87% compute saved vs. max tier, 7B active params burned.
Whole pipeline — classifier → registry → cascade → metrics → formatter → UI — worked
correctly on the first real query.

## Task 9 — Edge-case hardening

Ran three real probes instead of assuming the design docs' warnings were already handled:

1. **Unavailable-tier path.** Simulating an outage with a fake model ID actually returned a
   `400`, not a `503` — different failure, and it crashed instead of degrading. Decided to
   leave `400` uncaught on purpose (a bad model ID means the registry itself is broken, which
   should surface loudly, not get masked as a routine outage). Simulated a real `503` by
   mocking the provider call directly — confirmed the tier gets logged `"unavailable"` and the
   cascade moves on; if every tier in range is unavailable, it returns a clear fallback message
   instead of crashing.
2. **Confidence calibration, empirically checked.** 5 live cascade runs all came back with
   confidence ≥9, including a hallucinated wrong answer to a pi-digit question. That's the
   overconfidence failure mode expected going in, confirmed empirically rather than assumed. Flipped
   on `JUDGE_ALWAYS = True` in `cascade.py` — every answer now goes through the judge instead
   of fast-accepting on a confidence score alone. Re-ran the same pi-digit question: same wrong
   answer, but now correctly flagged `judged_fail` with an accurate reason.
3. **Classifier flip-flop caught by the full check-suite re-run.** "What is a closure in
   Python?" classified as `coding` instead of `qa` on one run. Confirmed via 6 repeat calls
   that it wasn't a consistent misclassification (genuine sampling variance at a fuzzy
   qa/coding boundary), then fixed the root cause — added one clarifying line per type to the
   classifier's system prompt — rather than loosening the test. 8/8 consistent afterward.

Confirmed `.env` was never staged or committed (`git log --all -- .env` returns nothing), and
ran the entire `checks/` suite end to end as a final gate before the last commit.

---
*(build complete — all 9 planned tasks done; see BUILD-LOG.md for the full error/debugging record)*

## Post-build: eval harness + resume-focused enhancements

Researched what would make this project stand out beyond "it works" — grounded in real cascade
routing literature (FrugalGPT, RouteLLM) and current production guidance. Built the highest-
value items:

- **`eval/` package**: a 25-query benchmark set spanning all 5 task types, `run_eval.py` runs
  each query through both the full cascade *and* an always-max-tier baseline (raw query, no
  classification/prompt optimization/escalation — the "naive" approach), judges both, and
  reports cascade pass rate vs. baseline pass rate vs. compute saved — the same kind of
  quality/cost tradeoff metric RouteLLM's paper reports.
- **`eval/calibration.py`**: turns the empirically-observed confidence overconfidence pattern
  (see BUILD-LOG) into an actual Expected Calibration Error (ECE) metric plus a reliability
  table, computed from every (confidence, judge-verdict) pair collected during an eval sweep.
- **Dollar-cost translation** in `metrics.py`: maps active-param savings to an illustrative
  dollar figure via a documented, approximate per-active-billion-param rate — makes the
  abstract compute-savings number legible to a non-ML audience.
- **CI** (`.github/workflows/checks.yml`): runs the checks that don't need live API keys on
  every push.

Running the eval harness live surfaced 5 real bugs/gaps in one sitting (all logged in detail in
`BUILD-LOG.md`): two crashes from unhandled `None`/empty API responses, a metrics-accounting
bug where a fully-failed run was reported as "100% compute saved," a judge rubric that
punished valid summaries/translations for not being factually literal, and — caught live in
the UI by manual testing, not by any automated check — a classifier prompt-rewrite step that
was silently corrupting translation instructions into unrelated questions. All fixed at their
actual root cause (shared client functions, cascade's trace recording, the judge's prompt, the
classifier's prompt) rather than patched at the call site that happened to surface them.

Also hit a real operational ceiling: OpenRouter's free tier caps unpaid accounts at 50
free-model requests *per day, account-wide* — a day of active testing exhausted it mid-audit.
The `ModelUnavailable` handling degraded correctly under that real, sustained condition with no
crash, which is itself a validation of the design — but it means the eval harness's final
"after all fixes" report needs a re-run after the daily reset. `eval/results_baseline_before_fixes.json`
preserves the pre-fix numbers (68% cascade/baseline pass rate parity, 63.9% avg compute saved,
0.405 ECE) as a before/after reference point once the clean run lands.

**Deliberately not built this round**: a semantic cache in front of the classifier, and an
A/B comparison view in the UI (cascade vs. always-max-tier, side by side, live) — both were
flagged as valuable in the research but are more invasive changes to core routing/UI that
deserve their own focused pass rather than being rushed in alongside everything else.
