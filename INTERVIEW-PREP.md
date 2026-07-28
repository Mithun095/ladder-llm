# Interview Prep — LadderLLM

Everything needed to talk about this project confidently: the pitch, the architecture, the
decisions and *why* they were made, the real bugs, and a full anticipated Q&A.

Read this alongside [`BUILD-LOG.md`](BUILD-LOG.md) (the debugging casebook) and
[`DEVLOG.md`](DEVLOG.md) (what was built and why).

**One rule before anything else: don't oversell.** The strongest thing about this project isn't
the savings number — it's that the savings number is *checkable*, and that several of them were
wrong before they were right. Lead with the measurement discipline. An interviewer who catches
you inflating a number stops listening to everything after it.

---

## The 30-second pitch

> "It's a router that sends each LLM query to the cheapest model that can plausibly handle it,
> and escalates to a bigger one only when a judge model confirms the cheap answer actually
> failed. It classifies difficulty and task type first, then walks a ladder of free models
> across Groq and OpenRouter instead of always calling the biggest one just in case. On my
> benchmark it saves about 77% of the active-parameter compute a naive always-biggest-model
> approach would burn, with two thirds of queries resolving at the cheapest tier. It's the
> FrugalGPT / RouteLLM cascade pattern, built on a fully free-tier stack — and I built an eval
> harness for it specifically so the savings claim is measured rather than asserted."

## The 2-minute walkthrough

1. **A query comes in.** A small model (`llama-3.1-8b-instant`) classifies it on two axes —
   difficulty (easy/medium/hard/expert) and task type (qa/coding/reasoning/summarization/
   translation) — and, for instruction-only queries, rewrites it into a cleaner prompt.
2. **A registry picks the model.** A plain dict keyed on `(tier, task_type)`. Difficulty sets
   the starting tier and the ceiling; type selects the column. Tiers run from an 8B model up to
   a 550B MoE model with ~55B active params per token.
3. **The model answers** and self-rates its confidence 1-10 in the same structured JSON
   response.
4. **A judge model checks the answer** — binary pass/fail, with a task-type-aware rubric. Pass
   returns the answer with the full trace. Fail escalates to the next tier, up to the
   difficulty-dependent ceiling.
5. **A Streamlit UI** shows the answer, the live routing trace (every model tried, what the
   judge said, how long each took), and the compute saved versus always using the biggest tier.

Add if there's time: *"and there's an exact-match cache in front of the whole thing, so a repeat
query costs zero LLM calls — not even the classifier."*

## Why each decision, in your own words

**Why classify difficulty *and* type, not one "quality" score?**
Type decides *which* model is good at the work — a coding-specialised 7B beats a general 8B at
writing code, and that ranking doesn't transfer to translation. Difficulty decides *how far up
the ladder to start*. Collapsing them means either wasting compute on easy queries that happen
to be a hard type, or under-serving hard queries of an easy type.

**Why both self-reported confidence and a separate judge?**
Confidence is nearly free — one extra field in a call you're already making. The judge costs a
whole second API call but is a genuine second opinion. I designed it to trust confidence at the
extremes and only pay for the judge in the ambiguous middle. **Then I measured it, and that
design was wrong** — confidence came back 9 or 10 on every call regardless of correctness, so
the ambiguous band was never reached and the judge never fired. I flipped to judging every
answer. The confidence shortcut code is still in the file behind a flag, because the more
interesting artefact is the design *plus* the measurement that killed it.

**Why is the judge binary pass/fail rather than a 1-10 score?**
A small model can reliably answer "does this look right, yes or no." Asking it to rate quality
on a ten-point scale asks for a discrimination it doesn't have — you'd get a number that looks
precise and means nothing. Scope the judge to what a small model can actually do.

**Why a plain dict for the registry instead of classes?**
There is exactly one behaviour: look up a config by tier and type. A dict does that. A class
hierarchy would be solving a problem I don't have, and every future reader would have to walk
it to learn what one `REGISTRY[(2, "coding")]` tells you at a glance.

**Why measure active params rather than total params?**
Several models here are Mixture-of-Experts — `gpt-oss-120b` is ~117B total but activates only
~5.1B per token. Compute cost tracks active params. If I'd recorded 120, the savings metric
would have been enormous and wrong *in my favour*, and it would have looked great, so nobody
would have questioned it. A consequence I chose to show rather than hide: active params aren't
monotonic with tier number, so tier 3 QA is genuinely *cheaper* than tier 2 QA.

**Why does the savings metric allow negative numbers?**
Because a cascade that escalates far enough really can cost more than one direct max-tier call
— expert coding is 32B at tier 3 plus 55B at tier 4 against a 55B baseline, i.e. −58%. It was
clamped at zero originally, which made the routing's worst case permanently invisible.
Unclamping it immediately exposed a second bug: tier-2 summarization was a 70B dense model, more
expensive than the tier-4 ceiling it was measured against. **The clamp had been hiding a real
routing bug.**

**Why an exact-match cache rather than a semantic one?**
Semantic caching needs an embedding model and a vector store — a real dependency and a real
change. Exact-match on a normalized string is a dict, it's honest about what it does, and it
already removes the largest single win available (a repeat query costs *zero* calls, including
the classifier every query otherwise pays for). It's marked in the code with its ceiling and its
upgrade path.

---

## Real bugs, STAR format

### "Tell me about the hardest bug you found."

**Situation.** Summarization queries were failing the judge almost every time — 0-1 out of 5 —
in both the cascade and the always-max-tier baseline.

**Task.** Both paths failing identically ruled out "the small model is bad at summarizing," so
something upstream and shared was wrong.

**Action.** My first diagnosis was the judge. Its rubric is "decide if the answer is correct,"
which suits QA and coding but misfires on summarization, where the standard is faithfulness, not
literal correctness. I added task-type-specific guidance to the judge's prompt. It helped
slightly, **and I recorded that as the fix.** It was not the fix.

Re-measuring later, summarization was still 1/5. So I stopped reading the judge's verdicts and
read the actual answers. One said: *"I'm not aware of the current events in the stock market as
my knowledge cutoff is December 2023."* That is not a bad summary — that is a model that was
never given the text. The judge had been right every time.

I printed what the classifier actually hands downstream:

| Raw query | What the model received |
|---|---|
| `Summarize: The stock market saw significant volatility this week as investors reacted to new inflation data...` | `Summarize the main points from the paragraph about the stock market.` |

There is no paragraph. The `optimized_prompt` step — meant to rewrite queries for clarity — had
**paraphrased away the payload**, leaving only a description of it.

**Result.** The root cause is that prompt rewriting is only safe when the query is *purely an
instruction*. Summarization and translation queries are instruction **plus payload**, and a
paraphrase can discard the payload while still being a "clearer" sentence. So I fixed it
structurally rather than with another prompt hint — content-bearing task types bypass the
rewrite entirely. On the subset I could run that day, summarization went 1/5 → 4/5, QA 4/5 →
5/5, and calibration error improved from 0.519 to 0.291. On the full benchmark afterwards,
summarization reached 5/5.

*Why this is the one to tell:* three separate lessons. I'd **documented a wrong root cause with
confidence**, and a partial improvement is the best disguise a wrong diagnosis has. I'd been
**reading the judge's opinion of the answer instead of the answer**, which said the problem in
plain English from the first run. And I'd hit this same bug earlier with *translation*, patched
that one instance, and never asked what else shared the mechanism.

### "Tell me about a bug that looked real but wasn't."

**Situation.** My first end-to-end cascade run failed outright — both tier 1 and tier 2 returned
malformed JSON, so my retry-once parsing had failed twice in a row.

**Task.** Determine whether the parsing logic was actually broken before touching it.

**Action.** I isolated layer by layer instead of guessing: raw model call (clean JSON), the
JSON-parsing helper alone (fine), the full provider-dispatch path (fine), the exact
classify→optimize→answer chain three times (fine each time), then re-ran the original failing
test with zero code changes.

**Result.** It passed. It was never a bug — small models return malformed JSON roughly 10-15% of
the time, and two independent ~12% events landing together is about a 1.4% coincidence I hit on
run one. **The LLM-specific lesson: before fixing a failure, establish whether it's
deterministic.** Everything downstream of a model call is probabilistic. Had I "fixed"
`_extract_json` there, I'd have introduced a real bug chasing a phantom.

The mirror image happened later and I used the same test: one specific model failed on *four
out of four* different queries. Deterministic, one common factor — that one was real (a
reasoning model wrapping its answer in a `<think>` block), and I found it in minutes because I
knew which question to ask.

### "Tell me about a time you were wrong about your own project."

**Situation.** Late in the project I noticed my judge failing answers that were obviously fine.
The clearest one: for *"A farmer has 17 sheep, all but 9 die, how many are left?"*, the model
answered *"The remaining number of sheep is 9, which is less than the original number of 17"* —
and the judge failed it with *"the answer does not state the remaining number of sheep."* It
states it. It's the fourth word.

**Task.** Work out whether the judge was wrong occasionally or wrong systematically.

**Action.** I'd been reading the judge's verdicts as data about the *answers*. Reading them as
data about the *judge*, the pattern was immediate: every false failure was about form, not fact
— "doesn't show the calculation," "too simplistic," "lacks context." It was grading essays. The
cause was my own prompt: *"You are a **strict** answer judge... decide if the answer is correct
and **adequately addresses** the question."* "Strict" invites rejection, "adequately addresses"
is undefined, so the model filled the gap with completeness-of-presentation. I replaced it with
explicit pass/fail conditions and an instruction that style, length and missing working are
out of scope.

**Result — and this is the actual story.** With the judge no longer failing things for style, I
re-ran my standard prompt set and one query came back *accepted* that never had before: *"what
is the 47th digit after the decimal point of pi?"*, answer **7**. I had been using that exact
question since my earliest testing as my canonical hallucination example — it's written up in
my build log as *"the model answers 7 with confidence 10, and once the judge became mandatory
it correctly caught it."*

So I finally checked. **The 47th digit of pi is 7.** The model had been right the entire time.
What I'd documented as "my judge correctly catching a hallucination" was my judge **rejecting a
correct answer for not showing its working** — the exact bug I'd just fixed, sitting in my own
documentation, being cited as evidence the system worked.

*Three things I'd draw out of it:*

1. **I never verified my own test case.** I chose that question *because* I assumed small models
   get it wrong, then used the model's answer as evidence for the assumption that made me choose
   it. Circular, and it survived the whole project because checking it took one line of Python
   I never ran.
2. **A component that fails safe still fails.** A judge that wrongly *rejects* looks
   responsible — it produces cautious escalations, not visible errors. It was quietly spending
   compute to replace correct answers, and every false rejection read as appropriate caution.
3. **The conclusion survived because it had a measurement behind it, not just a story.** The
   overconfidence finding the pi example was supporting is still true — but it's supported by
   the ECE number, not the anecdote. If the anecdote had been my only evidence, I'd have had
   nothing.

*(If asked "why didn't you just quietly fix it?" — because the git history and the build log
show the correction, and a project where nothing was ever wrong is a project where nothing was
ever checked.)*

### "Tell me about a time your own metrics lied to you."

**Situation.** The eval harness reported `100% compute saved` on a query that had failed to
produce any answer at all.

**Task.** Work out why "saved everything" and "delivered nothing" were showing up together.

**Action.** Traced it to how I recorded a `malformed_response` step — I'd defaulted its
`active_params_b` to 0, the same as a genuine provider outage. But a malformed response means
the model **did** run and burn tokens; only a 429/503 rejection, where the model never ran, is
genuinely free.

**Result.** Fixed the accounting to charge real compute for a malformed response. What makes
this worth telling is that it's the *third* flattering-but-wrong number this metric pipeline
produced — along with the near-miss of counting MoE models by total parameters, and the clamp
that hid negative savings. None of them crashed. All made the system look better than it was.
**A defensive default in a measurement path isn't defensive, it's a thumb on the scale**, and
you have to interrogate good numbers at least as hard as bad ones because nobody investigates
a result they like.

### "Tell me about a production-style constraint you hit."

**Situation.** Two different rate limits, same day. OpenRouter started returning `unavailable`
for most queries; later a test script died outright on Groq.

**Task.** Determine which was an external condition and which was my bug.

**Action.** Calling OpenRouter directly gave the real answer: `429 - Rate limit exceeded:
free-models-per-day`, capped at 50 requests per day **account-wide across every free model**.
Not a bug — a real ceiling of building on free infrastructure.

The Groq one *was* my bug, and an embarrassing one. I had `ModelUnavailable` handling
specifically so a rate limit couldn't crash anything — but it lived in `call_model()`, and the
classifier and judge call the shared `call_json()` directly, bypassing it. The answering path
degraded gracefully while the classifier, which runs on **every single query**, could take down
the whole app.

**Result.** Moved the handling down into `call_json()`, where all three paths already go. While
there I split the two limits by their actual behaviour: Groq's per-minute limit clears in
seconds, so a transient 429 now waits briefly and retries and usually just succeeds; OpenRouter's
daily cap survives that and correctly degrades to skipping the tier. **The transferable lesson:
when you add error handling, grep every caller of the thing you're protecting.** "I handled
that" was true of one path out of three, and the unprotected ones ran more often.

---

## Anticipated questions

**Q: Walk me through what happens when I type a query into the UI.**
Use the 2-minute walkthrough — said, not recited.

**Q: Isn't using an LLM to judge another LLM circular?**
Partly, yes, and the judge shares a class of bias with the model it's judging — that's a real
limitation I document rather than hide. Scoped narrowly to binary pass/fail it's still a more
useful signal than the alternative, which is trusting the model's own confidence — I measured
that at ~0.98 stated against ~0.75 actual. But I'd add that the judge is the component I trust
least, and I have a specific reason: I found it failing correct answers for stylistic reasons,
and one of those false failures had been sitting in my own documentation for weeks as *proof
the judge worked*. That's the story below.

**Q: What's Expected Calibration Error and why does it matter here?**
It measures the gap between stated confidence and actual accuracy: bucket every
(confidence, verdict) pair by confidence, compare each bucket's mean confidence to its actual
pass rate, take the weighted mean absolute difference. 0 is perfect. Mine went 0.52 → 0.23 →
0.13 as I fixed real bugs; the top bucket currently claims 0.95 and delivers 0.82. It's computed
by my own harness from live runs, and it's why the router doesn't trust self-reported confidence.

**The part I'd volunteer without being asked:** this ECE measures confidence against my
*judge's* verdicts, not against ground truth. So when my judge was over-strict and failing
correct answers, it inflated the apparent overconfidence — a real chunk of that 0.23 → 0.13
improvement was the judge getting less wrong, not the models getting better calibrated. The
metric is only as good as its referee. That's a genuine limitation of the whole
LLM-as-judge approach, and I'd rather state it than have someone find it.

**Q: Why not just always use the biggest model? Simpler and safer.**
That's literally the baseline in my eval harness — always tier 4, raw query, no classification,
no escalation. The cascade uses about 77% less active-parameter compute. "Simpler" isn't free
if it burns 3-10× the compute on queries an 8B model already answers correctly. Worth adding
honestly: the cascade is also *slower* on escalation, since three sequential round trips beats
one, so it's a cost/latency tradeoff rather than a free win.

**Q: What are the real limitations? Don't sugarcoat it.**
Four. **(1)** OpenRouter's free tier caps unpaid accounts at 50 requests/day account-wide; I hit
it mid-testing and it means the full cross-provider eval, including the baseline arm, can only be
run once a day. **(2)** The judge is a small model and still occasionally misjudges subjective
tasks even with task-aware prompting — there's a real ceiling there. **(3)** The dollar figure
is explicitly illustrative; these are free models with no bill to reconcile against. **(4)** The
benchmark is 25 hand-written queries I chose myself, which is enough to catch bugs and not
enough to make a strong statistical claim.

**Q: How do you know the savings number isn't optimistic marketing math?**
Because I built the harness specifically to make it falsifiable, and because it has *caught* me
being wrong three times — the MoE parameter count, the malformed-response accounting, and the
clamp hiding negative savings. A metric that has never contradicted you isn't being checked. I'd
rather present a number with its failure modes attached than a rounder one without.

**Q: How would you scale this to production?**
In priority order: **(1)** a semantic cache in front of the classifier, since a cache hit should
skip routing and inference entirely — the current exact-match version already shows the win;
**(2)** replace the static confidence thresholds with per-task-type thresholds tuned from logged
calibration data, since I'm already collecting the (confidence, verdict) pairs and nothing
consumes them yet; **(3)** persist traces to a real store instead of session state, so routing
quality is monitorable over time rather than per-session; **(4)** multi-key rotation and proper
backoff for the rate-limit ceiling. And honestly, **(5)** replace the LLM judge for objective
task types with real ground-truth evaluation where it exists — for coding that's running the
tests, which beats any judge model.

**Q: Why Groq for some models and OpenRouter for others?**
Different catalogs. Groq hosts a specific set of fast-inference models free; OpenRouter
aggregates many providers' free-tier models behind one OpenAI-compatible API. The abstraction in
`llm_client.py` means nothing downstream knows which is which — adding a third provider is one
function there and no changes anywhere else. It also caught me out once: "OpenAI-compatible"
describes the HTTP contract, not the Python exception hierarchy, so the two SDKs raise entirely
separate `APIStatusError` classes and you have to catch both.

**Q: What was the trickiest part?**
The cascade loop, because of three independent failure modes that must stay distinct: a provider
being down, a model returning unparseable output, and a model returning a parseable but wrong
answer. Collapsing them into one "failed" bucket means an outage gets mistaken for a bad answer
— and that's not hypothetical, it's exactly the bug that produced "100% compute saved" on a
total failure.

**Q: If you had one more week?**
The semantic cache and an A/B view in the UI showing cascade against always-max-tier side by
side and live. The harness already computes that comparison offline; putting it in front of a
user is what makes the argument without needing to read a JSON file.

**Q: What would you do differently if you started over?**
Build the eval harness first. I built it after the system worked, and it found five real bugs in
its first sitting — including two crashes that no amount of manual single-query testing had
surfaced, because they needed 25 queries across many models to hit. I'd been testing by typing
things into a UI and eyeballing whether the answer looked fine, which is exactly as rigorous as
it sounds.

**Q: This is all free-tier. Does any of it transfer to a real system?**
The economics transfer directly — free models still have the compute-cost *ratios* the routing
exploits, and the active-param accounting is the same arithmetic you'd do against a price sheet.
What doesn't transfer is reliability engineering: paid tiers don't hand you a 50-request daily
cap, so the aggressive `ModelUnavailable` degradation matters less. Though it did mean I tested
graceful degradation under genuine sustained outages rather than mocked ones, which most side
projects never do.

---

## Quick facts

- 5 task types × 4 tiers = 20 registered model configurations, all validated against both
  providers' live catalogs by `checks/check_model_ids.py` (which caught one delisted model ID
  mid-project).
- Escalation ceilings: easy/medium → tier 2, hard → tier 3, expert → tier 4.
- The judge fires on **every** answer — a change made after measuring confidence at 9-10
  regardless of correctness.
- ECE improved 0.52 → 0.23 → 0.13 across bug fixes; top confidence bucket claims 0.95,
  delivers 0.82.
- **76.6%** average active-parameter compute saved versus an always-tier-4 baseline.
- **72%** pass rate overall, **90%** (18/20) excluding queries where no model was reachable;
  13 of 20 answered queries resolved at tier 1.
- Free-tier ceilings: OpenRouter 50 requests/day account-wide (1000/day with a $10 credit);
  Groq 30 requests/minute.
- 11 self-checks, 5 of which run in CI without needing API keys (the rest need live API access).
- Zero paid APIs, zero GPUs.

## Things to be honest about if pushed

- The 25-query benchmark is small and self-authored. It is a bug-finding instrument first and a
  statistical claim a distant second.
- The judge is the weakest component and the least rigorous part of the design.
- Several results in this repo have been wrong before they were right, and the git history shows
  it. That's the intended impression, not a thing to explain away.
