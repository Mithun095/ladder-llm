# Research notes — what would actually make LadderLLM better

Eight ranked recommendations from published work and real systems, filtered against this
project's constraints (free tiers, CPU only, one person, no hand-labelling at scale) and against
what `BUILD-LOG.md` already records as tried and rejected.

**Every URL below was fetched and verified.** Two things were checked empirically rather than
recalled, and both changed a recommendation:

- **Groq does not support `logprobs`.** Tested directly against `llama-3.1-8b-instant` and
  `openai/gpt-oss-120b`: both return
  ``400 — '`logprobs` is not supported with this model'``. Every token-probability approach to
  calibration (P(True), perplexity-based deferral, the logprob formulation of semantic entropy)
  is therefore dead on this stack. Item 6 uses the sampling variant instead.
- **Groq does accept `temperature` and `seed`** (same test, no 400 returned) — but
  `src/llm_client.py` sets neither, so every call in the system today runs at the provider
  default. That matters for items 2 and 6 and is called out in both.
- **arXiv 2406.12665 is not the paper it is widely miscited as.** It is *CollabStory*, not
  *Trust or Escalate*. The correct ID is 2407.18370. Verified by fetching both.

**Ranking principle.** Sorted by (value to this project) / (effort), with value weighted toward
measurement quality — because the documented central weakness is that the benchmark is
noise-dominated (two identical sweeps: 72% and 84%) and its quality metric is scored by the same
judge it tunes. Items 1-4 are measurement. Items 5-6 are the system, and **neither can be
evaluated until 1-3 exist** — that is stated as their counterargument, because shipping them
first would produce the fifth unfalsifiable judge revision.

---

## 1. Score the benchmark against gold answers, not against the judge

**What.** Replace (or extend) the 25 self-authored queries in `eval/benchmark_set.py` with items
from three public datasets that carry machine-checkable ground truth:

| type | dataset | gold signal | how to score |
|---|---|---|---|
| reasoning | GSM8K test split | final number after `####` | exact numeric match |
| coding | MBPP (sanitized) | `test_list` of `assert` statements | execute them |
| qa | MMLU test | gold choice index | letter match |

The judge still runs inside the cascade — it is the escalation trigger, it is part of the system
under test. What changes is that the *benchmark's* pass/fail comes from ground truth instead of
from the judge's opinion. That severs the circularity `BUILD-LOG.md` #18 names as the most
general lesson in the log.

**Sources** (all verified):
- GSM8K — Cobbe et al., *Training Verifiers to Solve Math Word Problems*,
  https://arxiv.org/abs/2110.14168. Data: https://github.com/openai/grade-school-math —
  raw JSONL confirmed downloadable at
  `https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl`
  (fetched; `{"question": ..., "answer": ...}` per line, gold after `####`).
- MBPP — Austin et al., *Program Synthesis with Large Language Models*,
  https://arxiv.org/abs/2108.07732. Data confirmed at
  `https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json`
  (fetched; each record has `prompt`, `code`, `test_list`).
- MMLU — Hendrycks et al., *Measuring Massive Multitask Language Understanding*,
  https://arxiv.org/abs/2009.03300. No `datasets` library needed — Hugging Face's
  datasets-server serves rows over plain HTTP with no auth, verified working:
  `https://datasets-server.huggingface.co/rows?dataset=cais/mmlu&config=all&split=test&offset=0&length=100`

**Why it fits.** Three `curl`s and `json.loads`. No new dependency (`requests` is already
pinned), no GPU, no hand-labelling — the labels ship with the data. MBPP scoring reuses the
execution harness `eval/compare_coding_models.py` already has — `run_generated(code, tests)`,
which writes the code to a temp file and runs the assertions in a `subprocess` with a timeout,
verified by reading it — and `BUILD-LOG.md` #24 already
concluded "being able to run the artifact beats any judge" — this generalises that conclusion
from coding to two more task types. It also takes n from 25 to ~150, which shrinks the
confidence interval by about 2.4x on its own.

**Effort.** ~1 day. A loader that caches three files into `eval/data/`, plus three scorers
(~120 lines total).

**Measurably improves.** The headline pass rate stops meaning "share of answers this judge
approved" and starts meaning "share of answers that are correct". Judge nondeterminism —
one of the two documented sources of the 12-point spread — is removed from the score entirely
for 3 of 5 task types.

**Strongest argument against.** Quota and format risk. ~150 queries through the full cascade is
450-750 Groq calls; at 30 req/min that is ~25 minutes of pacing, and any OpenRouter tier hit
burns the 50/day account cap. Mitigable — 22 of 25 queries resolve at tiers 1-2, which are all
Groq. The real risk is the second one: MMLU is multiple-choice and MBPP wants bare code, neither
of which fits `ANSWER_SYSTEM_PROMPT`'s free-text JSON envelope. Get that wrong and you reproduce
#24's trap exactly — a capable model scoring 0% because of the harness, not the model. Validate
the loader on 5 items per type with a known-good model before trusting any number.

Two scope limits worth writing into the README rather than discovering later. Summarization and
translation get no ground truth from any of this; they stay judge-scored, so 2 of 5 task types
keep the old problem. And **MMLU changes the task, not just the answer format**: 4-way multiple
choice has a 25% floor from guessing, and small models do disproportionately better at
recognising an answer than at recalling one. So tier distribution and savings measured on MMLU
will read optimistic against the free-form QA the app actually serves. Treat it as a knowledge
probe that happens to have ground truth, not as a proxy for real QA traffic.

---

## 2. Split the eval into `collect` and `score`, and freeze the artifacts

**What.** `eval/run_eval.py` currently calls the model, the judge, and the tier-4 baseline in one
pass, so changing anything downstream means re-running everything upstream. Persist raw per-query
outputs — answer text, tier, model id, classification, timings — to `eval/runs/<timestamp>.json`,
and make scoring a separate pass that reads that file.

Three things fall out, all of which the project currently says it cannot do:

- **Judge variance becomes measurable for free.** Run the judge 5x over one *frozen* answer set:
  the entire spread is judge nondeterminism, with model sampling and classifier drift held
  constant. Today the 72%/84% spread mixes all three and nothing separates them.
- **Judge/rubric changes cost zero quota** and are evaluated against *identical* answers — a
  paired comparison instead of two independent draws from a wide distribution.
- **The baseline arm stops doubling every sweep.** Collect it once, reuse it across many
  scoring runs.

Two cheap determinism fixes belong with it. First: **`src/llm_client.py` sets no `temperature`
and no `seed` on any call**, so the whole system — answers, classifier, judge — runs at the
provider default sampling temperature. Groq accepts both parameters (verified above, no 400), so
pinning `temperature=0` and a fixed `seed` on eval runs is a three-line change that cuts
model-sampling variance immediately. Determinism still isn't guaranteed across a served fleet,
which is exactly why it should be *measured* with the replicate machinery above rather than
assumed.

Second: cache the classifier's verdict per query in a checked-in JSON. `BUILD-LOG.md` #23
measured difficulty as stable on only 22 of 25 queries, and difficulty sets both entry tier and
ceiling — so routing wobbles between sweeps for reasons that have nothing to do with the change
being tested. Pin it for A/B runs, and report classifier instability as its own separate number
instead of letting it leak into the pass rate.

**Source.** No paper — this is the generate-then-score split that EleutherAI's
`lm-evaluation-harness` implements as `--log_samples`, writing `<task>_eval_samples.json` for
post-hoc analysis without re-running the model:
https://github.com/EleutherAI/lm-evaluation-harness (verified).

**Why it fits.** It is almost entirely moving code that already exists. No new dependency. And it
attacks the binding constraint directly: the project's own README says the full eval can be run
about once a day, which is why so many conclusions here rest on single runs.

**Effort.** 3-5 hours. Probably the best value-per-hour item on this list; it is ranked second
only because item 1 is what makes the collected artifacts worth scoring.

**Measurably improves.** Decomposes the 12-point noise floor into judge variance vs. model
variance vs. classifier variance — the single thing `BUILD-LOG.md` #21 identifies as unknown.
Multiplies effective quota by however many scoring passes you run.

**Strongest argument against.** Frozen classification means the eval no longer measures the
end-to-end system a user experiences, so you need both modes and must label which number is
which — otherwise you have quietly swapped one measurement for a different one, which is this
project's recurring bug (#11, #16, #19). Frozen answers also go stale: the free-tier models
change under you, so an artifact set has a shelf life and needs a date stamp.

---

## 3. Generate the judge's test set from the same gold data

**What.** `eval/judge_ground_truth.py` has 14 hand-written cases — 7 per error type. That is why
`BUILD-LOG.md` #24 could not distinguish false-fail counts of 0, 2, 2 and 0 from each other. At
n=7, a 29% false-pass estimate has a 95% confidence interval of roughly 8-64%; the check cannot
detect anything short of a total fix.

Once item 1 lands, labelled judge cases are free, from two sources:

- **Perturbation.** `(question, gold answer, should_pass=True)` and
  `(question, corrupted gold, should_pass=False)`. For GSM8K corrupt the number (±1, digit swap,
  another problem's answer); for MBPP mutate the reference `code` and *verify the mutation
  actually fails `test_list`*, so the label is derived by execution, not asserted.
- **Harvest.** Better, and completely free once item 2 exists: every collected cascade answer
  already has a ground-truth verdict. A tier-1 answer that failed exact-match is a genuine
  `should_fail` case; one that passed is a genuine `should_pass` case. These are the real error
  distribution the judge faces, in the real answer style, at whatever volume you have run.

Both together give 200+ labelled cases for zero hand-labelling.

**Source.** Same datasets as item 1. The design point — that a judge must be measured against
labels it does not produce — is `BUILD-LOG.md` #18's own conclusion; this just gives it enough n
to be useful.

**Effort.** 2-4 hours on top of items 1 and 2.

**Measurably improves.** False-pass and false-fail rates get error bars narrow enough to A/B a
judge change: a 95% interval roughly 20-38% at n=100, versus today's 8-64%. Without this, every judge improvement in
items 5 and 6 is unfalsifiable.

**Strongest argument against.** Synthetic perturbations are easier to catch than real errors — a
judge can ace "391 vs 392" while still ratifying a fluent, wrong essay, which is precisely the
failure #18 found. So the perturbation half will flatter the judge. Weight the harvested half
higher and never report perturbation accuracy on its own.

---

## 4. Put error bars on the numbers instead of caveats in the prose

**What.** The README reports 92.7%, 83.7% and 88% as point estimates and then explains in three
paragraphs of prose that they are noisy. Evan Miller's paper gives the arithmetic that replaces
the prose. Four things apply directly here:

- **CLT standard error over questions.** At n=25 and p=0.88, SE ≈ 6.5 points → a 95% CI of
  roughly 75-100%. That interval belongs in the results table.
- **Clustered standard errors** when each question is run k times — the variance has a
  within-question term that naive pooling ignores. This is exactly the replicate structure the
  project needs after item 2.
- **Paired differences for two-system comparison.** Cascade-vs-baseline is already paired per
  query (`baseline_ran` / `baseline_passed` are recorded per row) — so use the paired difference
  and its SE, which is far more powerful than comparing two independent rates. The README
  currently does a sign test by hand; formalise it.
- **Power analysis.** How many questions to detect a 5-point improvement at 80% power. Answers
  "is n=150 enough?" *before* spending the quota — the exact question `BUILD-LOG.md` #21 wishes
  it had asked.

**Source.** Evan Miller, *Adding Error Bars to Evals: A Statistical Approach to Language Model
Evaluations*, https://arxiv.org/abs/2411.00640 (verified — title and author confirmed).

**Why it fits.** The project's whole personality is measurement honesty stated in prose. This
converts that into arithmetic, which is both shorter and stronger. Stdlib `statistics` covers
most of it; `numpy` for a bootstrap if you want one.

**Effort.** 2-3 hours.

**Measurably improves.** Nothing about the system — it improves the claims. It also gates future
experiments: a power calculation would have stopped the "72% → 100%" result in #21 before it was
run, not after.

**Strongest argument against.** Error bars do not make the instrument better, they make its
limits legible. Done before items 1-3, you mostly publish wide intervals around a metric you have
already established is measuring the wrong thing. Near-free once item 1 lands; low value before.

---

## 5. Replace the single judge with a panel, and escalate the judge on disagreement

**What.** The judge is one call to `llama-3.1-8b-instant`. Two papers combine into one change:

- **PoLL** finds that a panel of several *smaller, architecturally different* models outperforms
  a single large judge, while reducing intra-model bias and costing over 7x less. This project
  already has three model families on the free tier — `llama-3.1-8b-instant`, `openai/gpt-oss-20b`,
  `qwen/qwen3.6-27b` — so a 3-way majority vote costs nothing but calls.
- **Trust or Escalate** makes it cheaper still: run the cheap judges; if they agree, take the
  verdict; if they disagree, treat that as the uncertainty signal and escalate the *judge* to
  `gpt-oss-120b`. A cascade for the judge, inside a cascade project — thematically the right
  shape for this repo, and it turns judge disagreement into a free non-self-reported confidence
  signal (feeds item 6).

There is a specific failure in the README this should fix: run 3 of the sheep riddle, where the
judge solved the problem itself, got 8, fell for the same trick as the answering model, and
rejected the correct answer 9 twice. That is a single-model reasoning failure. Different model
families fail on different riddles, which is exactly what a panel cancels.

**Sources** (both verified):
- Verga et al., *Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse
  Models*, https://arxiv.org/abs/2404.18796
- Jung et al., *Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreement*,
  https://arxiv.org/abs/2407.18370 — note the correct ID; 2406.12665 is a different paper.

**Why it fits.** `judge.py` grows a list of judge models and a vote; the escalation branch is
~10 lines. Everything stays on Groq's per-minute limit rather than OpenRouter's daily cap. Three
8B-class judge calls remain cheaper on this project's own cost table than one `gpt-oss-120b`
judge call.

**Effort.** 3-4 hours.

**Measurably improves.** The false-pass rate, which is the project's stated weakest number. PoLL
reports the panel beating a single large judge; here the comparison is against a single *small*
judge, so the headroom should be larger.

**Strongest argument against.** **You cannot show it works until item 3 exists.** On 7 cases per
error type, a panel could halve the false-pass rate and the check would not reliably see it —
#24 already recorded four identical runs scoring 0, 2, 2, 0. Ship this before the measurement and
it becomes the fourth judge revision justified by a number the project has itself proven is
noise, which is the exact mistake #24 ends by naming. Also 3x the judge calls per answer is real
latency in a UI that already takes multiple seconds, and it pushes hard on Groq's 30 req/min cap:
`call_json` handles a 429 with a single 2-second sleep and then raises `ModelUnavailable`, so
under a full eval sweep a tripled call volume will start marking judges unavailable — and
`cascade.py` treats an unavailable judge as *accept unverified*, which would silently inflate the
pass rate. Pace the sweep, and count unverified accepts as their own category.

---

## 6. Replace the dead confidence field with agreement across samples

**What.** `BUILD-LOG.md` #7 killed self-reported confidence: 9-10 on every call, ECE 0.279. The
published replacement is entropy over *sampled answers*, not over stated confidence.

The original method needs token probabilities. **Verified unavailable:** Groq returns
``400 — '`logprobs` is not supported with this model'`` for both `llama-3.1-8b-instant` and
`openai/gpt-oss-120b`. So the logprob formulation is out.

The authors published a variant for exactly this situation. From Oxford's OATML group blog on
the Nature paper: *"We introduce an additional 'discrete' variant of semantic entropy that can be
computed without access to token probabilities while still performing well."* (verified by
fetching the page; the Nature article itself is behind an auth redirect, so the implementation
details are cited from the blog, not from the paper text I could read.)

The adaptation for this project — mine, not the paper's, and it should be labelled as such in any
write-up: sample k=3 answers, group them by meaning, use the number of distinct groups (or the
largest group's share) as the uncertainty score. For reasoning and QA the answers are short
enough that normalized string equality is an adequate grouping function — no entailment model, no
embeddings, no torch, no new dependency. And because `llm_client.py` sets no temperature, calls
already run at the provider default and repeated samples already diverge, so **no plumbing is
needed to get diversity** — just call k times.

Three uses: (a) an escalation trigger that does not depend on the judge at all — the most
valuable property on offer, because it is the first quality signal in this system that is not
judge-derived; (b) a real calibration curve, replacing an ECE computed over a field that is
effectively constant; (c) the "should I trust this verdict" input for item 5.

**Sources** (both verified):
- Kuhn, Gal & Farquhar, *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation
  in Natural Language Generation*, https://arxiv.org/abs/2302.09664
- Farquhar, Kossen, Kuhn & Gal, *Detecting hallucinations in large language models using semantic
  entropy*, Nature 630, 625-630 (2024), https://www.nature.com/articles/s41586-024-07421-0 —
  **this URL redirects to a Nature auth page and I could not read the article text.** The
  discrete-variant claim above is sourced from the authors' own group blog, which I did fetch:
  https://oatml.cs.ox.ac.uk/blog/2024/06/19/detecting_hallucinations_2024.html

**Effort.** 4-6 hours, behind a flag.

**Measurably improves.** Gives the project a genuine calibration story to replace the one it had
to abandon, and an escalation signal whose errors are uncorrelated with the judge's — which is
worth more than a better judge, because it is a *second* signal rather than a refinement of the
only one.

**Strongest argument against.** It costs 3x the calls at tier 1, straight out of the savings
headline — the metric the project can actually measure reliably. It is also useless for
summarization (20% of the benchmark), where no cheap grouping function exists for long free text.
And whether it beats the judge as an escalation trigger is unmeasurable until item 1 lands. Gate
it, measure the savings hit, and be prepared to find that 3x8B at tier 1 is not worth it.

---

## 7. Use RouterBench to test routing *policy* offline, at zero quota

**What.** 405k precomputed inference outcomes from 11 LLMs across 8 datasets, with correctness
labels and costs attached. Routing policies — entry tiers, ceilings, thresholds — can be
evaluated by table lookup: no API calls, no quota, no judge, no run-to-run noise at all. The
paper also defines a cost-quality Pareto frontier to compare a policy against.

**Source.** Hu et al., *RouterBench: A Benchmark for Multi-LLM Routing System*,
https://arxiv.org/abs/2403.12031. Code: https://github.com/withmartian/routerbench (verified).
Data: https://huggingface.co/datasets/withmartian/routerbench — verified to contain
`routerbench_0shot.pkl`, `routerbench_5shot.pkl`, `routerbench_raw.pkl`.

**Why it fits.** The binding constraint on this project is that the real eval runs about once a
day. Questions like "is `CEILING_TIER` near-optimal?" — which `BUILD-LOG.md` #21 tried to answer
and could not, because the noise floor swallowed the effect — become deterministic table
lookups.

**Effort.** ~half a day.

**Measurably improves.** Lets `STARTING_TIER` / `CEILING_TIER` be argued from data instead of
from the `ponytail:` comment currently sitting on them, without spending a day of quota per
hypothesis.

**Strongest argument against.** **Its 11 models are not this ladder.** Nothing it produces
transfers to LadderLLM's savings numbers or tier assignments; it can only say whether a
threshold-shaped policy is near-optimal *in general*. Publishing a RouterBench result next to
this project's own numbers would invite exactly the conflation the repo is otherwise careful
about. Also: the data ships as pandas pickles — a new dependency plus unpickling a third-party
file, and HF's datasets-server reports "No (supported) data files found" for it, so there is no
preview API to inspect it safely first.

---

## 8. AutoMix's framing: the judge is a noisy sensor, not a gate — read it, don't build it

**What.** AutoMix addresses precisely this project's problem — a cascade whose verifier is
unreliable — and its answer is not "fix the verifier" but "model the verifier's noise and act
under uncertainty", via a few-shot self-verifier plus a POMDP router over its output.

The POMDP is overkill here. The transferable idea is one line of design: the judge's verdict is
*evidence*, not a decision. This project has measured that the judge is wrong ~29% of the time in
each direction, and `cascade.py` uses that knowledge nowhere — `verdict == "fail"` is treated as
ground truth. Combining judge verdict + panel disagreement (item 5) + sample agreement (item 6)
gives three actions instead of two: accept, escalate, or accept-but-flag.

**Source.** Aggarwal et al., *AutoMix: Automatically Mixing Language Models*,
https://arxiv.org/abs/2310.12963 (verified).

**Effort.** 1-2 days including threshold fitting.

**Strongest argument against — and this is the one item here I would not build.** It requires
items 1, 2, 3, 5 and 6 all in place before any threshold can be fitted; it adds a tuned policy to
a system whose main virtue is that its policy is simple and legible in a dict; and AutoMix's
reported gains come from datasets orders of magnitude larger than this one. Ship the framing as a
comment on `CEILING_TIER` and move on.

---

## Deliberately not recommended

Considered against the task's focus areas and rejected, with reasons:

- **Semantic / embedding cache (GPTCache-style).** Improves demo latency, not measurement — and
  measurement is where the project's real weakness is. The exact-match cache already exists with
  a documented upgrade path in a `ponytail:` comment in `cascade.py`. Any embedding model is a
  new local model plus torch, bought for a benefit the benchmark cannot even detect: 25 distinct
  queries, zero paraphrases.
- **RouteLLM's released routers.** Verified from https://github.com/lm-sys/RouteLLM: the `mf` and
  `sw_ranking` routers require an **OpenAI API key for embeddings** — a paid API, which violates
  the hard constraint — and `bert` / `causal_llm` need transformers checkpoints. The paper
  (https://arxiv.org/abs/2406.18665, verified) is already cited in the README for its framing;
  the library is not usable here. Say so rather than listing it.
- **Distilling the LLM classifier into a local classifier** (the FrugalGPT DistilBERT-scorer
  pattern, https://arxiv.org/abs/2305.05176; the Hybrid LLM BERT router,
  https://arxiv.org/abs/2404.14618 — both verified). The classifier already measures 100% type
  accuracy on the benchmark's own labels (#23), so there is no accuracy headroom to buy. The
  remaining wins are latency and determinism, and item 2 buys determinism for about five lines
  and no dependency.
- **A fifth judge-prompt revision.** #17, #18 and #24 have mined this out, and #24 explicitly
  warns the next person not to tune the prompt against the current check.
- **Anything gated on self-reported confidence.** Rejected with data in #7; item 6 is the
  replacement.
- **Token-logprob calibration / P(True)** (*Language Models (Mostly) Know What They Know*).
  Verified dead on this stack: Groq returns HTTP 400 on `logprobs` for both models tested.

## Suggested order

1 → 2 → 3 → 4 establishes a benchmark that can detect a change. Only then 5, then 6, each
measured against it. 7 is independent and can be done any time as a side quest. 8 is a comment.

The one thing worth saying plainly: items 5 and 6 are the interesting engineering, and they are
ranked below the boring measurement work on purpose. This repo's own log contains four separate
entries about believing a number that could not support the conclusion drawn from it. Building
the panel judge before the test set that could falsify it would be the fifth.
