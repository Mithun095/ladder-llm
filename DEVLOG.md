# Dev Log — what got built, in what order, and why

A walkthrough of the system one module at a time, in the order it was built. Each section
explains the concept first, then the decision, then what was actually verified. If you want
the failures instead, read [`BUILD-LOG.md`](BUILD-LOG.md); for the interview framing, see
[`INTERVIEW-PREP.md`](INTERVIEW-PREP.md).

**The core idea in one paragraph.** Almost every LLM application picks one model up front and
sends everything to it — usually the biggest one, because that's the safe default. But most
queries don't need it: "what is the capital of Australia?" is answered correctly by an 8B model
in under a second. A *cascade router* starts each query at the cheapest model that could
plausibly handle it, checks whether the answer is actually good, and only escalates to a
bigger model when it isn't. You pay big-model prices only for the queries that genuinely need
a big model. This is the idea behind [FrugalGPT](https://arxiv.org/abs/2305.05176) and
[RouteLLM](https://arxiv.org/abs/2406.18665); LadderLLM is that pattern built on entirely free
infrastructure.

---

## Setup: spec, plan, git

Before writing code, the project brief (a problem statement, a phased build plan, a scoring
rubric) became a single design spec and then a task-by-task implementation plan. Git
initialized from scratch — this started as a plain folder — with `.gitignore` covering `.env`,
`__pycache__`, and `.venv`. Each task below ends in its own commit.

Two structural choices worth naming:

- **`src/` holds all real code; `checks/` holds one assert-based script per module.** No
  pytest. Each check is a plain Python file you run with `python -m checks.check_x` that prints
  what it saw and asserts on it. For a project this size a test framework is ceremony; what
  actually matters is that every module has *something* that fails loudly when it breaks.
- **Plain dicts and dataclasses over class hierarchies.** There's exactly one registry lookup
  behaviour and one trace-step shape. A dict and a `@dataclass` express both completely.

## Task 1 — Scaffolding

`requirements.txt` (streamlit, groq, openai, pydantic, python-dotenv, requests), a venv, and a
tiny `checks/check_env.py` confirming both API keys load from `.env`.

## Task 2 — `llm_client.py`: one door to every model

Everything else in the system calls models through this one module. Three ideas live here.

**Provider abstraction.** Groq ships its own SDK. OpenRouter doesn't need one — it exposes an
OpenAI-compatible REST API, so the `openai` SDK talks to it just by pointing `base_url` at
`https://openrouter.ai/api/v1`. `call_model()` picks the right function from the model's
`provider` field, so nothing downstream ever knows or cares where a model is hosted. Adding a
third provider means adding one function here and nothing anywhere else.

**Structured output by prompting, not by API feature.** These models are asked for JSON the
plain way — the system prompt says "reply with ONLY this JSON shape" — and then the response
is *validated* rather than trusted. `call_json()` extracts the JSON object from whatever the
model wrapped it in, validates it against a Pydantic schema, and retries once with a stricter
instruction if that fails. Every JSON-expecting caller (classifier, judge, cascade) goes
through this one function, which is why fixing extraction once fixed it everywhere.

Getting that extraction right took three attempts and broke twice (`BUILD-LOG.md` #12). Models
bury their JSON in markdown fences, in chatty preambles, and — for reasoning-tuned models —
inside a `<think>` block that quotes *draft copies* of the JSON before emitting the real one.
The working approach doesn't look for any of those wrappers: it scans for every position where
a complete JSON object parses, skips past each one it finds, and keeps the last top-level
object. Six wrapper shapes are pinned in `checks/check_json_extraction.py`.

**Failure classification.** `ModelUnavailable` is raised when a provider returns 429 or 503.
That's what lets the cascade tell "this tier is down" apart from "this tier gave a bad answer"
— two things that must never be confused, since one is worth escalating over and the other
just means try the same rung again later. A 400 is deliberately *not* caught: it means the
registry holds a model ID that doesn't exist, which is a real bug that should crash.

A transient 429 (Groq's per-minute limit clears in seconds) waits briefly and retries once
before giving up; a sustained one (OpenRouter's *daily* free-tier cap) correctly degrades to
skipping the tier. This handling lives in `call_json()` rather than `call_model()` because the
classifier and judge call `call_json` directly — putting it one layer up left both of them able
to crash the app, which they eventually did (`BUILD-LOG.md` #14).

## Task 3 — `registry.py`: the tier × task-type grid

A plain dict mapping `(tier, task_type)` → `ModelConfig`. Four tiers, five task types, 20
entries.

**Why two dimensions and not one.** Task *type* decides which model is good at the work — a
coding-specialised 7B model beats a general 8B model at writing code, and neither ranking is
transferable to translation. Difficulty decides *how far up the ladder to start*. Collapsing
them into a single "quality" axis would mean either wasting compute on easy queries that
happen to be a hard *type*, or under-serving genuinely hard queries of an easy type.

**Why `active_params_b` and not `params_b`.** This is the field the whole savings metric is
built on, and it's the one place a naive number would have quietly inflated every result. Some
models here are Mixture-of-Experts: `openai/gpt-oss-120b` has ~117B total parameters but a
router activates only ~5.1B of them per token. Compute cost tracks *active* params, not total.
Recording 120 instead of 5.1 would have made the savings number look spectacular and be wrong
(`BUILD-LOG.md` #3).

A consequence worth stating rather than hiding: **active params are not monotonic with tier
number.** Tier 3 QA (`gpt-oss-120b`, 5.1B active) has a *smaller* compute footprint than tier 2
QA (a dense 27B). That looks like a mistake and isn't — it's the honest output of ranking tiers
by capability while measuring them by compute. The one case that *was* a genuine bug — tier-2
summarization at 70B dense, i.e. more expensive than the 55B-active tier-4 ceiling — was found
only after the savings metric stopped clamping negative values, and fixed by swapping in a 27B
model (`BUILD-LOG.md` #16).

Before any of this was hardcoded, `checks/discover_models.py` pulled the live model lists from
both providers and every grid entry was checked against what actually existed.

## Task 4 — `classifier.py`: the routing decision

One cheap call to `llama-3.1-8b-instant` returns three things:

- `difficulty` — easy / medium / hard / expert → sets the starting tier and the ceiling
- `type` — qa / coding / reasoning / summarization / translation → selects the model column
- `optimized_prompt` — the query rewritten to be clearer before any answering model sees it

**The classifier must never fail the request**, since it runs on every query. If the model is
rate-limited or returns junk, `classify()` falls back to a neutral `medium`/`qa` classification
with the query passed through untouched — a middle tier still answers, rather than the whole
request dying before a single model is tried.

**Prompt optimization is deliberately not applied to every type.** This is the correction that
came out of the single largest bug in the project (`BUILD-LOG.md` #13). Rewriting a query for
clarity is safe when the query is *purely an instruction*. It is actively destructive when the
query carries a **payload** — the text to be summarized or translated — because a paraphrase is
free to discard that payload and still be a "clearer" sentence:

```
Summarize: The stock market saw significant volatility this week as investors reacted to...
  ↓ rewritten to ↓
Summarize the main points from the paragraph about the stock market this week.
```

There is no paragraph any more. The answering model, given nothing to summarize, fell back to
world knowledge and correctly reported that its training data ended in 2023. So
`PRESERVE_QUERY_TYPES = {"summarization", "translation"}` bypasses the rewrite entirely for
content-bearing types. Fixing this took summarization from 1/5 to 4/5 on the benchmark subset.

## Task 5 — `judge.py`: the escalation signal

The LLM-as-judge piece, scoped deliberately narrow: it returns **`pass` or `fail`**, never a
numeric score. A small model can reliably answer "does this look right, yes or no." Asking the
same model to rate quality 1-10 asks for a discrimination it doesn't have, and you'd get a
number that looks precise and means nothing.

**The prompt names its pass and fail conditions explicitly**, and rules style out of scope. The
original said *"You are a strict answer judge... decide if the answer is correct and adequately
addresses the question"* — and that vagueness made it grade presentation instead of substance.
It failed *"The remaining number of sheep is 9, which is less than the original 17"* for "not
stating the remaining number." Worse, it had been failing a **correct** answer to a pi-digit
question since the start of the project, and I had that incident written up as evidence the
judge worked (`BUILD-LOG.md` #17). Fixing the prompt moved the whole benchmark: pass rate
68% → 72%, compute saved 71.5% → 76.6%, ECE 0.23 → 0.13.

Two further refinements came from measurement, not design:

- **The rubric is task-type-aware.** A generic "is this correct?" prompt suits QA, coding and
  reasoning, where there's one right answer. It misfires on summarization and translation,
  where the standard is faithfulness, not literal correctness — it will fault a good summary
  for omitting a secondary detail, which is the entire point of summarizing. `TYPE_GUIDANCE`
  adds a per-type sentence for those two.
- **An unavailable judge accepts rather than escalates.** If the judge can't return a verdict,
  the answer is accepted and marked *unverified* in the trace. Escalating instead would spend a
  bigger tier only to hit the same broken judge one rung up — strictly more expensive, same
  outcome.

## Task 6 — `cascade.py`: the waterfall

The core loop, and the hardest piece to get right.

**Starting tier and ceiling both come from difficulty:**

| difficulty | starts at | ceiling |
|---|---|---|
| easy / medium | tier 1 | tier 2 |
| hard | tier 2 | tier 3 |
| expert | tier 3 | tier 4 |

Harder queries get both a higher floor (don't waste a round-trip on a model that will obviously
fail) and a higher ceiling (allowed to spend more before giving up).

**Confidence: designed, measured, demoted.** The original design had the answering model
self-rate its confidence 1-10 in the same JSON as the answer, and trusted that number at the
extremes — ≥8 accept immediately, ≤4 escalate immediately, and fire the judge only in the
ambiguous 5-7 band. The appeal is that confidence is nearly free (one extra field in a call
you're already making) while the judge costs a whole second call.

**That design is no longer live.** In testing, confidence came back 9 or 10 on *every single
call*, on correct and incorrect answers alike. The 5-7 band was never reached, so the judge
never fired, so every answer was fast-accepted on a number carrying no information.
`JUDGE_ALWAYS = True` disables both shortcuts and judges every answer (`BUILD-LOG.md` #7).
Confidence is still collected — it's the raw material for the calibration metric below — it's
just no longer trusted as a routing decision. Measured properly later, the top confidence
bucket claims ~0.98 and delivers ~0.75.

The shortcut code stays in the file behind the flag on purpose: "here is the design, and here
is the measurement that killed it" is more useful than a file that pretends the idea was never
tried.

**Three distinct failure modes, three distinct statuses.** This is where most of the
difficulty lives — every trace step records which of these happened:

| status | meaning | compute charged? |
|---|---|---|
| `accepted` | judge passed it (or judge unavailable → unverified) | yes |
| `judged_fail` | model answered, judge rejected it → escalate | yes |
| `escalated` | confidence below threshold (only when `JUDGE_ALWAYS=False`) | yes |
| `malformed_response` | model ran but its output wouldn't parse → escalate | **yes** |
| `unavailable` | provider returned 429/503, model never ran → skip tier | **no** |

The last two rows are the subtle ones and getting them wrong produced a real metrics bug
(`BUILD-LOG.md` #11): a malformed response means the model *did* run and burn tokens, so it
must be charged. Only a provider rejection is genuinely free.

**An exact-match query cache** sits in front of the whole thing. A repeat query costs zero LLM
calls — not even the classifier, which every query otherwise pays for. It's a normalized-string
dict, marked `ponytail:` with its ceiling: the upgrade path is embedding the query and matching
on cosine similarity so paraphrases hit too, backed by a persistent store. The eval harness
runs with the cache off, since a benchmark number that quietly benefits from cache hits isn't
measuring the router.

Each step also records its own latency, and the result records end-to-end latency, which makes
the real tradeoff visible: a cascade that escalates twice is *cheaper* than one max-tier call
but *slower*, because it's three sequential round trips instead of one.

## Task 7 — `metrics.py` and `formatter.py`

`metrics.py` sums the active params actually burned across a trace (skipping `unavailable`
steps, which never ran) and compares that against what a single max-tier call would have cost
for that task type.

**The savings number is deliberately not clamped at zero.** A cascade that escalates far enough
genuinely can cost more than the baseline — expert coding hitting 32B at tier 3 then 55B at
tier 4 is 87B against a 55B baseline, i.e. −58%. Clamping that to "0% saved" would make the
routing's worst case permanently invisible, which is exactly the kind of flattering default
that had already produced two wrong numbers in this project (`BUILD-LOG.md` #16).

There's also an illustrative dollar figure, computed from a documented approximate per-active-
billion-param rate. It's marked as illustrative in the code, the UI and the README, because
these are free models with no real bill — its only job is to make the abstract active-param
number legible to someone who doesn't think in parameters.

`formatter.py` is pure presentation: wrap a coding answer in a code fence if the model didn't,
render a multi-line translation as a small table.

## Task 8 — `app.py`: the Streamlit UI

Two columns — answer on the left, live routing trace on the right — with the classification
decision above them and a cost row below.

**The one Streamlit behaviour you must know:** it re-executes the *entire script*, top to
bottom, on every interaction, including every keystroke in a text box. A bare
`result = run_cascade(query)` in the script body would fire a full multi-model cascade on every
character typed. So the call is gated behind `if submit and query`, and the result is stashed in
`st.session_state` so it survives the rerun that clicking Submit itself causes.

Run it with `PYTHONPATH=. streamlit run src/app.py` — plain `streamlit run src/app.py` raises
`ModuleNotFoundError: No module named 'src'`, for reasons in `BUILD-LOG.md` #5.

The UI shows the trace rather than just the answer on purpose. The answer is what a user wants;
the trace — which models were tried, what the judge said about each, how long each took, what
it cost — is the actual subject of the project.

## Task 9 — Edge-case hardening

Rather than assuming the design's warnings were handled, each was probed directly: the
unavailable-tier path (via a mocked 503, after discovering that a fake model ID produces a 400
and *shouldn't* be caught), the confidence calibration assumption (which failed, and changed
the design), and a full check-suite re-run that caught a classifier flip-flop. All three are
written up in `BUILD-LOG.md` #6, #7 and #8.

Confirmed `.env` was never staged or committed (`git log --all -- .env` returns nothing).

---

## Post-build: measurement and hardening

Everything above makes the system *work*. This round makes its central claim *checkable*.

**`eval/` — the benchmark harness.** 25 queries across all five task types. Each runs through
the full cascade **and** through an always-max-tier baseline (raw query, no classification, no
prompt optimization, no escalation — the naive approach the whole project argues against). Both
are judged, and the harness reports cascade pass rate against baseline pass rate against
compute saved. That's the same quality/cost tradeoff shape RouteLLM's paper reports, and it's
the difference between "it saves compute" as an assertion and as a measurement.

Savings are averaged only over queries that actually produced an answer — a query where every
tier was rate-limited burns zero compute and scores "100% saved", which is true and completely
misleading.

**`eval/calibration.py` — Expected Calibration Error.** ECE measures the gap between stated
confidence and actual accuracy: bucket every (confidence, verdict) pair by confidence, compare
each bucket's average confidence against its actual pass rate, and take the weighted mean
absolute difference. 0 is perfect calibration. This turns the hand-wavy "models are
overconfident" observation into a number that can be tracked and compared across changes — and
it moved measurably as real bugs were fixed, 0.52 → 0.23 → 0.13.

One caveat belongs with it: this ECE compares confidence against the **judge's** verdicts, not
against ground truth. An over-strict judge failing correct answers inflates the apparent
overconfidence, so part of that improvement was the judge getting less wrong rather than the
models getting better calibrated (`BUILD-LOG.md` #17). The metric is only as good as its referee.

**CI** (`.github/workflows/checks.yml`) runs every check that doesn't need live API keys on
each push: registry, calibration, metrics/formatter, JSON extraction, error handling. The last
two exist specifically because they cover code that was broken silently and repeatedly, where
the only symptom was a bad result in a live run days later.

**Running the harness found five bugs in one sitting** — two crashes from unhandled `None`
responses, a metrics-accounting bug reporting a total failure as 100% saved, a judge rubric
misfiring on subjective tasks, and (caught by manual UI testing, not by any check) the
classifier corrupting translation prompts. All are in `BUILD-LOG.md`. The point of an eval
harness turns out to be as much stress test as scoreboard: 25 queries across many models hit
edge conditions no amount of single-query manual testing had.

## What's deliberately not built

- **Semantic cache.** The current cache is exact-match only. Matching paraphrases needs an
  embedding model and a vector store — worth doing, a bigger change than it looks.
- **A/B view in the UI** — cascade and always-max-tier side by side, live. The eval harness
  already computes this comparison offline; the UI version is presentation work that also
  doubles every query's cost.
- **Per-task-type tuned thresholds.** The calibration data needed to tune them is already
  being collected; nothing consumes it yet.
- **Persistent trace storage.** Traces live in session state only, so routing quality can't be
  tracked across sessions.
