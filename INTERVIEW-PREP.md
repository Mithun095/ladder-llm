# Interview Prep — LadderLLM

Everything you need to talk about this project confidently: the pitch, the architecture, the
real decisions you made, the real bugs you hit, and a full anticipated Q&A. Read `BUILD-LOG.md`
and `DEVLOG.md` alongside this if you want the raw, blow-by-blow version of anything here.

---

## The 30-second pitch

"I built a router that sends each LLM query to the cheapest model that can plausibly handle
it, and only escalates to a bigger model when a judge model confirms the cheap answer
genuinely failed. It classifies difficulty and task type first, then walks a ladder of free
models — Groq and OpenRouter — instead of always calling the biggest one 'just in case.' On my
eval set it saved about 64% of the compute a naive always-biggest-model approach would burn,
at roughly the same answer-quality pass rate. It's the same idea behind FrugalGPT and
RouteLLM, applied to a fully free-tier stack."

## The 2-minute walkthrough

1. A query comes in. A small model (`llama-3.1-8b-instant`) classifies it along two axes —
   difficulty (easy/medium/hard/expert) and task type (qa/coding/reasoning/summarization/
   translation) — and rewrites the query into a cleaner prompt.
2. A registry (a plain dict, tier × type → model) picks the specific model for the current
   tier. Tiers 1-4 span from an 8B model up to a 550B-parameter MoE model with ~55B active
   params per token.
3. The model answers and self-rates its own confidence 1-10, in the same structured JSON
   response.
4. A judge model (also `llama-3.1-8b-instant`) checks the answer. If it passes, the answer is
   returned with a trace showing every tier tried. If it fails, the cascade escalates to the
   next tier and tries again, up to a difficulty-dependent ceiling.
5. A Streamlit UI shows the answer, the full routing trace (which models were tried and why),
   and how much compute was saved compared to always using the biggest tier.

## Why each design decision, in your own words

- **Why classify difficulty AND type, not just one?** Type determines *which* model is good at
  the task (a coding-specialized small model vs. a general one); difficulty determines *how far
  up the ladder* to start. Conflating them would mean either wasting compute on easy queries of
  a "hard" type, or under-serving genuinely hard queries of a "usually easy" type.
- **Why self-reported confidence AND a separate judge, not just one?** Confidence is nearly
  free (same API call, extra JSON field) but I found empirically it's unreliable — see the
  calibration section below. The judge costs a second API call but is a genuine second
  opinion. Originally I only fired the judge on ambiguous confidence (5-7); I flipped to
  "always judge" after finding confidence was ~9-10 regardless of correctness in live testing.
- **Why is the judge binary pass/fail, not a 1-10 score?** A small model judging on a fine
  numeric scale is asking it to do something it's not well-suited for — nuanced quality scoring
  requires more capability than a small model reliably has. Binary "does this look right"
  is a scope a small model can actually do reasonably well.
- **Why a plain dict for the registry instead of a class hierarchy?** There was never more than
  one behavior (look up a model config by tier and type) — a dict already does that.
  A class hierarchy would be solving a problem I didn't have.
- **Why does compute-saved use active params, not total params?** Several tier-4/tier-3 models
  are Mixture-of-Experts — `gpt-oss-120b` is ~117B total parameters but only ~5.1B active per
  token. If I'd used total params, the compute-saved metric would look artificially generous
  (or in one specific case, artificially *bad* — a hard coding query resolving at a 32B-active
  tier-3 model shows less "savings" than expected relative to the 55B-active tier 4, because
  active-param sizing isn't strictly monotonic with the tier number). Reporting the raw
  numbers alongside the percentage, rather than hiding that, was a deliberate choice.

## Real bugs you found (STAR-format, ready to tell)

### "Tell me about a bug that looked like a real problem but wasn't."

**Situation:** My end-to-end cascade test failed on the very first run — both tier 1 and tier
2 came back with malformed JSON, for the identical query and code.
**Task:** Figure out if this was a real bug in my parsing/retry logic before touching any code.
**Action:** I isolated each layer instead of guessing — called the raw model directly (clean
JSON came back), called my JSON-parsing helper directly (parsed fine), called the full
provider-dispatch path (parsed fine), reproduced the exact classify→optimize→answer chain the
cascade uses (parsed fine, three times), then re-ran the original failing test with zero code
changes.
**Result:** It passed cleanly on re-run. It wasn't a bug — it was the known ~10-15%
malformed-JSON rate from small models, hitting twice in a row by chance. If I'd started
editing `_strip_fences()` or my Pydantic schema the moment I saw the failure, I'd have
"fixed" something that was never broken. The real validation was that my error-handling design
(log it, escalate, degrade gracefully) turned a probabilistic failure mode into a non-event.

### "Tell me about the hardest bug you actually found and fixed."

**Situation:** During eval testing, a translation query — "Translate 'Where is the nearest
train station?' to Spanish" — came back as a Spanish sentence about not having GPS access, at
*both* tiers I tried, on two completely different models.
**Task:** Two different models failing identically pointed away from "one model is bad" and
toward something shared upstream.
**Action:** I traced the actual prompt each model received and found the culprit: my
classifier's "rewrite this query to be clear and unambiguous" step had rewritten the query into
`"where is the nearest train station in spanish translation"` — which deleted the word
"translate" as an instruction and left something that reads exactly like a real navigation
question.
**Result:** Fixed it at the actual source — added explicit guidance in the classifier's system
prompt to preserve the literal "Translate 'X' to Y" structure for translation-type queries,
instead of patching the downstream answer prompt (which would have left the corrupted prompt
in place for anything else that used it). Verified against multiple translation queries
afterward.

### "Tell me about a metrics/measurement bug — a time your own numbers lied to you."

**Situation:** My eval harness reported `100% compute saved` on a query that had actually
*failed* to produce any usable answer at all.
**Task:** Figure out why "saved 100%" and "totally failed" were showing up together — those
shouldn't both be true.
**Action:** Traced it to how I recorded a `malformed_response` trace step — I'd defaulted its
`active_params_b` to 0, same as a genuine provider outage. But a malformed response means the
model *did* run and generate tokens — real compute was spent, the JSON just didn't parse. Only
an actual 429/503 rejection (the model never ran) is legitimately free.
**Result:** Fixed the trace-recording to charge real compute for a malformed response, same as
any other real attempt. This is a good story because it shows you don't just trust a metric
that looks good — you interrogate a number that looks *too* good.

### "Tell me about hitting a real production-style constraint."

**Situation:** Mid-audit, live testing started returning `unavailable` for almost every
OpenRouter-routed query, even ones that had worked minutes before.
**Task:** Determine whether this was a bug in my code or an external condition.
**Action:** Called an OpenRouter model directly outside the app and got the real answer: a
`429 - Rate limit exceeded: free-models-per-day` error. OpenRouter caps unpaid accounts at 50
free-model requests per day, account-wide — not per-model — and a day of active testing had
exhausted it.
**Result:** Not a code bug at all — and a good validation: my `ModelUnavailable` handling
degraded correctly under a real, sustained, account-wide outage, not just a simulated one.
Documented it as a known operational constraint rather than something to "fix" — anyone
reproducing this project needs to know it.

## Anticipated interview questions and prepared answers

**Q: Walk me through what happens when I type a query into the UI.**
A: [Use the 2-minute walkthrough above, said out loud, not read.]

**Q: Why use an LLM to judge another LLM's answer? Isn't that circular?**
A: It can be, and the judge shares the same class of bias — that's a real, documented
limitation, not something I'm hiding. But scoped narrowly (binary pass/fail, not nuanced
scoring) it's a useful signal, especially compared to trusting a model's own self-reported
confidence, which I found empirically to be badly overconfident (see the calibration
section). The judge is a second, independent inference — it's not perfect, but it catches
things confidence alone misses, which I demonstrated directly: the same hallucinated wrong
answer that got waved through at confidence 9-10 got correctly flagged once I made the judge
mandatory instead of confidence-gated.

**Q: What's Expected Calibration Error, and why does it matter for this project?**
A: It measures the gap between a model's stated confidence and its actual accuracy. I
collected (confidence, judge-verdict) pairs across live eval runs and found that in the
top confidence bucket (self-rated 8-10 out of 10), actual accuracy was only around 61% — an
ECE of 0.405, where 0 would be perfect calibration. That's not a guess, it's a measured
number from my own eval harness, and it's exactly why the system doesn't trust self-reported
confidence alone.

**Q: Why not just always use the biggest model? Isn't that simpler and safer?**
A: That's literally the baseline I compare against in my eval harness — an always-tier-4
approach, no classification, no escalation. The cascade matched its pass rate while using
~64% less active-parameter compute on average. "Simpler" isn't free if it's burning 3-10x the
compute for queries that a small model already answers correctly.

**Q: What are the actual limitations of this system? Don't sugarcoat it.**
A: Three real ones. First, OpenRouter's free tier hard-caps at 50 requests/day account-wide,
which I hit mid-testing — this system as-is doesn't scale past that without a paid tier.
Second, the judge is a small model and, even with task-aware prompting, still occasionally
misjudges subjective tasks like summarization — there's a real ceiling to what a small judge
model can evaluate well. Third, the "$ saved" metric is explicitly illustrative — these are
free models, so there's no real bill to compare against; it's a documented approximation, not
a real billing reconciliation.

**Q: How would you scale this to a real production system?**
A: A few directions I'd take, in priority order: (1) a semantic cache in front of the
classifier — a cache hit should skip routing and the LLM call entirely, which is standard
production guidance for this pattern; (2) move from static confidence thresholds to
thresholds tuned per task type from logged calibration data, since I already have the
infrastructure to collect (confidence, verdict) pairs; (3) add retry/backoff and multi-key
rotation to handle the free-tier rate-limit ceiling I hit; (4) persist every query's trace
(SQLite or similar) instead of only in-session state, so routing quality can be monitored
over time, not just per-session.

**Q: Why Groq for some models and OpenRouter for others?**
A: Different model catalogs — Groq hosts a specific set of fast-inference models for free;
OpenRouter aggregates many providers' free-tier models under one OpenAI-compatible API. I
built a small provider-abstraction layer (`llm_client.py`) so the rest of the system — the
registry, the cascade, the judge — never has to know or care which provider a given model
lives on; it just calls `call_model()` and gets an answer back.

**Q: What was the trickiest part of the whole project?**
A: Getting the escalation loop (`cascade.py`) right — juggling three independent failure
modes at once (a provider being genuinely down, a model returning unparseable JSON, and a
model returning a parseable but wrong answer) and making sure each one is handled differently
rather than collapsed into one generic "failed" bucket, since conflating them would mean an
outage gets mistaken for a bad answer, or vice versa.

**Q: If you had one more week, what would you build next?**
A: The semantic cache and an A/B comparison view in the UI (cascade vs. always-biggest-model,
side by side, live) — both flagged as valuable during research but deliberately deferred
because they touch core routing/UI more invasively than I wanted to rush in the same pass as
everything else.

**Q: How do you know your compute-savings number isn't just optimistic marketing math?**
A: Because I built an eval harness specifically to make that claim checkable instead of
asserted — it runs the same 25 queries through both the cascade and a naive always-biggest
baseline, judges both independently, and reports the real pass-rate delta alongside the
compute-saved percentage. I'd rather show "68% pass rate at 64% less compute" than just say
"it saves compute" with nothing to back it.

**Q: What would you say is your single best "shows how you actually work" story from this
project?**
A: Probably the translation bug — not because it was the hardest bug technically, but because
of how I found it: two different models failed identically on the same query, which is a
strong signal the bug isn't in either model but in something they both received. That's a
debugging instinct (blame the shared input before blaming the two independent things that
both broke the same way), not a lucky guess.

## Quick facts to have on the tip of your tongue

- 5 task types, 4 tiers, 20 registered (tier, type) model configurations, all verified live
  before being hardcoded.
- Escalation ceiling: easy/medium → tier 2, hard → tier 3, expert → tier 4.
- Judge fires on every answer (not just ambiguous confidence) — a deliberate change made after
  finding confidence was almost always 9-10 regardless of correctness.
- ECE of 0.405 in the high-confidence bucket, ~61% actual accuracy where confidence implied
  ~90%+.
- OpenRouter's free-tier daily cap: 50 requests/account/day unpaid, 1000/day with $10 credit.
- Zero paid APIs, zero GPUs — Groq's free tier + OpenRouter's `:free` model catalog only.
