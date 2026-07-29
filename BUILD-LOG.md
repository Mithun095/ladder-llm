# Build Log — every bug I hit, and how I found it

My working record of what broke while building LadderLLM. Each entry follows the same shape,
so it's readable as a debugging casebook rather than a diary:

> **Symptom** — what I actually saw
> **How I found the cause** — the steps, including the wrong turns
> **Root cause** — what was actually wrong
> **Fix** — what I changed, and where
> **Takeaway** — the transferable lesson

For what got built and why, see [`DEVLOG.md`](DEVLOG.md). For the interview framing of these
same stories, see [`INTERVIEW-PREP.md`](INTERVIEW-PREP.md).

**The one theme worth noticing before you read on:** of the 25 issues below, only about a third
were bugs in code. The rest were bugs in a *prompt*, in a *metric*, in a *test*, in a *comment*,
or in my own assumption about what the failure meant. In an LLM system the code is often the
least likely thing to be broken, and that changes where you look first.

The single most repeated mistake here is worth naming up front, because it recurs in #11, #16,
#19, #24 and #25: **confusing "some output exists" with "this worked", and losing track of which
population a number is averaged over.** Five separate appearances, each caught only by asking
what exactly the denominator was.

---

## 1. Two SDKs, two different exception classes

**Symptom** — none yet; caught while writing the code.

**How I found the cause** — I wanted one `try/except` around any model call to catch rate
limits (429) and capacity errors (503) from either provider. I went to write
`except openai.APIStatusError` and stopped: OpenRouter is *OpenAI-compatible*, so I use the
`openai` SDK for it — but Groq ships its own SDK, which defines its own separate
`APIStatusError` class.

**Root cause** — "OpenAI-compatible API" means the HTTP contract matches. It says nothing
about Python exception hierarchies. Two libraries, two unrelated class trees, and
`except openai.APIStatusError` would silently not catch a single Groq failure.

**Fix** — import both and catch them as a tuple:

```python
from groq import APIStatusError as GroqAPIStatusError
from openai import APIStatusError as OpenAIAPIStatusError
...
except (GroqAPIStatusError, OpenAIAPIStatusError) as e:
```

**Takeaway** — "compatible" is a claim about a wire protocol, not about your language's type
system. A Groq outage would have crashed the app while an identical OpenRouter outage
degraded gracefully, and nothing in testing would have shown it until Groq actually went down.

---

## 2. Model IDs in my design notes were a runtime landmine

**Symptom** — none yet; avoided deliberately.

**How I found the cause** — I'd sketched the tier × task-type model grid on paper before
writing `registry.py`. Free-tier catalogs on Groq and OpenRouter change constantly. A stale ID
doesn't fail at import, or in a linter, or in a type check — it fails at *runtime*, as a `400`,
on whichever unlucky query first routes to that tier.

**Root cause** — a hardcoded table of external identifiers with no verification step is
config that lies quietly.

**Fix** — wrote a script to hit both providers' live model-list endpoints, and cross-checked all
20 grid entries against what actually existed *that day* before writing a single line of
`registry.py`.

**Takeaway** — when you hardcode identifiers owned by someone else, write the script that
verifies them.

> ⚠️ **I got this half right, and it cost me later.** That first script only *printed* both
> catalogs for me to eyeball. It never compared them to the registry, so it could only ever
> catch a stale ID on the day I happened to run it and read the output carefully. Months later
> `poolside/laguna-m.1:free` was delisted and started returning 404 mid-eval-sweep — exactly the
> failure this script existed to prevent, sailing past a script that was technically running.
> It's now `checks/check_model_ids.py`, which validates every entry and **fails loudly**. A
> check that reports instead of asserting isn't a check, it's a report — and nobody reads a
> report that has always been fine.

---

## 3. Counting MoE models by total parameters would have inflated my headline metric

**Symptom** — none yet; caught while filling in the registry.

**How I found the cause** — I was about to write `120` for `openai/gpt-oss-120b`. The name
says 120B. But it's a Mixture-of-Experts model: a router picks a small subset of "expert"
subnetworks per token, so only ~5.1B parameters are actually *active* for any given token —
the other ~112B sit idle.

**Root cause** — "model size" is two different numbers, and the whole point of this project
is the second one. Compute cost tracks *active* params per token, not total params on disk.

**Fix** — the registry field is named `active_params_b`, not `params_b`, and every entry is
documented as active-not-total.

**Takeaway** — if I'd written 120, my compute-savings number would have been enormous and
wrong, in my own favour, and it would have looked *great*. The most dangerous bugs in a
metrics pipeline are the ones that flatter you — nobody investigates a good number.

---

## 4. The cascade failed on its very first end-to-end run — and there was no bug

**Symptom** —

```
Answer: No model produced a usable answer.
Trace:
  tier=1 model=llama-3.1-8b-instant status=malformed_response
  tier=2 model=qwen/qwen3.6-27b     status=malformed_response
```

Both tiers, two different models, one run. My retry-once JSON parsing had failed twice in a row.

**How I found the cause** — my instinct was "the JSON parsing is broken." Instead of editing
it, I isolated one layer at a time:

1. Raw model call, identical prompt → clean, valid JSON.
2. My JSON-parsing helper, fed that output → parsed fine.
3. Full provider-dispatch function with the real registry config → parsed fine.
4. The exact classify → optimize → answer chain the cascade runs, three times → fine each time.
5. Re-ran the *original failing test*, zero code changes → passed.

**Root cause** — nothing. Small models return malformed JSON maybe 10-15% of the time. Two
independent ~12% events landing together is a ~1.4% coincidence, and I hit it on run one.

**Fix** — none. That's the point.

**Takeaway** — this is the LLM-specific debugging skill: **before you fix a failure, establish
whether it's deterministic.** Everything downstream of a model call is probabilistic. If I'd
"fixed" `_extract_json` here I'd have introduced a real bug chasing a phantom. The rule I
took from it: reproduce it three times, or don't touch it. (Had it failed on *every* run,
that would have been a genuine signal — and later, in #12, it was.)

---

## 5. `ModuleNotFoundError: No module named 'src'` — only in Streamlit

**Symptom** —

```
File "/home/mithun/Desktop/ladder-llm/src/app.py", line 3, in <module>
    from src.cascade import run_cascade
ModuleNotFoundError: No module named 'src'
```

Every other module imports `from src.cascade import ...` happily. All checks pass. Only the
app breaks.

**How I found the cause** — the import is identical everywhere, so the difference had to be
*how the process starts*. Everywhere else I run `python -m checks.check_x` from the project
root, and `-m` puts the current directory on `sys.path`. `streamlit run src/app.py` launches
the file directly, exactly like `python src/app.py` would.

**Root cause** — when Python runs a *script* directly, it puts that script's own directory on
`sys.path` — here, `.../ladder-llm/src`. So Python looks for a `src` package *inside* `src/`.
The `src` it needs is one level up, and that level is never added.

**Fix** — no code change; launch with the project root on the path:

```bash
PYTHONPATH=. streamlit run src/app.py
```

**Takeaway** — an import error that only appears under one launcher is almost never a bad
import. `python -m pkg.mod` and `python path/to/file.py` build `sys.path` differently, and
knowing which one your tool uses saves an hour of restructuring a package that was fine.

---

## 6. Simulating an outage with a fake model ID taught me the wrong lesson (at first)

**Symptom** — to test the "tier unavailable" path I pointed a registry entry at
`totally-fake/does-not-exist:free`. It came back `400 BadRequestError` and crashed the
cascade, instead of being caught as `ModelUnavailable`.

**How I found the cause** — obvious immediately: I only catch 429/503, and this was a 400.

**Root cause** — my *test* was wrong, not the code. A made-up model ID isn't what an outage
looks like. An outage is a real model the provider can't serve right now (429/503). A 400
means "this model does not exist" — which, since `registry.py` is generated from a verified
live model list, can only mean my registry is broken.

**Fix** — deliberately left 400 uncaught, and documented why. A broken registry is a real bug
that should crash loudly in development, not get silently swallowed and logged as a routine
outage. To test the path properly I mocked the provider call to raise a genuine 503, and
confirmed the tier is skipped, logged `unavailable`, and the cascade moves on.

**Takeaway** — "catch more exception types" is usually the wrong reflex. Errors mean different
things, and collapsing them into one bucket destroys information you need. Also: if your
simulated failure doesn't look like the real failure, you're testing a fiction.

---

## 7. Self-reported confidence was 9-10 on every single call

**Symptom** — my original design fired the judge only for "ambiguous" confidence (5-7), on
the theory that ≥8 is clearly fine and ≤4 is clearly not. In 5 live runs, confidence *never
once* landed in 5-7. It was 9 or 10 every time — on correct answers and incorrect ones alike.

**Root cause** — a model's self-rated confidence measures how *fluent and well-formed* its
answer feels, not whether it's true. There's no internal fact-check being consulted. So the
ambiguous band the whole design hinged on was never reached, the judge never fired, and every
answer was fast-accepted on a number that carried no information.

**Fix** — flipped `JUDGE_ALWAYS = True` in `cascade.py`: skip the confidence shortcuts, judge
every answer. Later quantified properly as an Expected Calibration Error of 0.23-0.52 depending
on the run, with the top confidence bucket claiming ~0.98 and delivering ~0.75.

**Takeaway** — I'd *read* that LLMs are overconfident. Reading it and measuring it are
different: measuring it told me a specific branch of my design was dead code before I shipped
it. The confidence shortcuts are still in the file behind the flag, because "here's the design,
and here's the measurement that killed it" is worth more than a clean file.

> ⚠️ **This entry originally contained a wrong example, and the correction is in #17.** I used
> to illustrate it with "I asked for the 47th digit of pi, it answered 7 with confidence 10 —
> a hallucination the judge later caught." The answer was **not** a hallucination. The 47th
> digit of pi is 7. The model was right, my judge was wrong, and I'd written the incident up
> as proof my judge worked. See #17.

---

## 8. The classifier flip-flopped on a query it had gotten right before

**Symptom** — re-running the full check suite before calling the build done,
`"What is a closure in Python?"` came back `type=coding`, failing an assertion expecting `qa`.

**How I found the cause** — tempting fix: loosen the assertion to accept either. I'd written
myself a note earlier that a failed type assertion is a signal about prompt quality, not test
strictness, so instead I ran the same query 6 more times: **6/6 `qa`**. Not consistently
broken — intermittent.

**Root cause** — a genuinely ambiguous boundary. "Closure in Python" *is* a programming topic
(→ coding), but the user wants an explanation, not code (→ qa). My prompt never said which
way to resolve that, so the model resolved it by sampling.

**Fix** — fixed the ambiguity at its source, in the classifier's system prompt: one clarifying
line per type, explicitly stating that conceptual questions *about* programming are `qa`
unless code is actually being requested. Re-ran 8 times: **8/8 `qa`**.

**Takeaway** — an intermittent classification failure is usually an underspecified prompt, not
a flaky model. And relaxing the assertion would have deleted the only signal telling me the
prompt was vague.

---

## 9. A CI check that makes no network calls still needed API keys

**Symptom** — `check_metrics_formatter.py` is pure arithmetic and string formatting, zero
network. In CI it failed with `KeyError: 'GROQ_API_KEY'`.

**How I found the cause** — the traceback pointed into `src/llm_client.py`, which the check
never calls. Following the import chain: `check_metrics_formatter` → `cascade` → `classifier`
→ `llm_client`, whose module body runs `_groq = Groq(api_key=os.environ["GROQ_API_KEY"])` at
**import time**. Importing is enough to require the key.

Locally this is invisible: `.env` always exists and `load_dotenv()` backfills it. I only
reproduced it by moving `.env` aside to simulate a clean CI checkout.

**Root cause** — side effects at module import time. Importing a module for one pure function
drags in every side effect of everything it transitively imports.

**Fix** — scoped to CI: the workflow sets placeholder (non-secret) env values so imports
succeed. Making client construction lazy is the better fix, but it's a change to a
well-tested core module for a CI-only problem, so I documented the tradeoff instead of
taking it. *(Marked in the workflow file, not silently skipped.)*

**Takeaway** — "works on my machine" is often literally "my machine has a file yours doesn't."
Deleting your own config and re-running is the cheapest CI simulator there is.

---

## 10. Two crashes, one line apart, from the same wrong assumption

**Symptom** — first live eval sweep crashed:

```
TypeError: 'NoneType' object is not subscriptable
```

inside `call_openrouter`. Fixed it, re-ran, hit the *same error class* one line up.

**How I found the cause** — first crash: `resp.choices[0].message.content` was `None`. Fixed
with `content or ""`. Second crash: `resp.choices` *itself* was `None` — so
`resp.choices[0]` blew up before `.content` was ever reached.

**Root cause** — an overloaded OpenRouter free model can return **HTTP 200 with an error
payload in the body** instead of a proper error status. The SDK dutifully parses that into a
response object with `choices=None` and raises nothing. My code assumed a 200 meant a
well-formed completion.

**Fix** — guard the whole shape, in both provider wrappers:

```python
if not resp.choices:
    return ""
return resp.choices[0].message.content or ""
```

**Takeaway** — two lessons. Fixing the exact line that threw, rather than the assumption
behind it, gets you the same bug again one line over — the second crash was really the first
one, unfixed. And a 200 status is not a promise about the body's shape.

Neither of these appeared in any manual single-query test. It took a 25-query sweep across
many models to surface them, which is the real argument for an eval harness: it's a stress
test that happens to also produce a number.

---

## 11. "100% compute saved" on a query that completely failed

**Symptom** — the eval report gave `"What is 17 * 23?"` a `compute_saved_pct` of **100.0**,
while also marking it failed. Saved everything, delivered nothing.

**How I found the cause** — a number that good on a query that bad is a metric bug, not a
result. Traced the trace: both tiers returned `malformed_response`, and `cascade.py` recorded
those steps with `active_params_b=0` — the same value used for a genuine `unavailable` outage.
Zero params burned → 100% saved.

**Root cause** — I'd conflated two different failures. `unavailable` means the provider
rejected the request and the model never ran: genuinely free. `malformed_response` means the
model **did** run and generate tokens, and only the parsing failed: that compute was spent
and has to be paid for in the accounting.

**Fix** — `malformed_response` steps now record the model's real `active_params_b`, same as
any other attempt that actually ran.

**Takeaway** — interrogate numbers that look *too good* at least as hard as numbers that look
bad. This one was reporting a total failure as a perfect result. *(A related version of this
survived until later — see #13.)*

---

## 12. One model returned malformed JSON on 100% of queries

**Symptom** — hunting for a query that demonstrates a clean multi-tier escalation,
`qwen/qwen3.6-27b` came back `malformed_response` on all four queries I tried.

**How I found the cause** — the inverse of #4: four different queries, one model, identical
failure. Deterministic, and the model is the common factor. So I printed its raw output:

```
<think>
1. Analyze the request: ...
5. Construct JSON: `{"answer": "9", "confidence": 10}`
</think>

{"answer": "9", "confidence": 10}
```

It's a reasoning-tuned model. It emits its whole chain of thought first, and — crucially —
quotes a **draft copy of the JSON inside that reasoning**.

**Root cause** — my extraction only stripped markdown code fences. It had no concept of a
reasoning block wrapping the answer.

**My first fix made it worse.** I took "everything from the first `{` to the last `}`". The
first `{` landed inside the draft at step 5, the last `}` at the real answer — so the
extracted span included all the reasoning text *between* them. Still unparseable.

Second attempt: last `{` to last `}`. That fixed this model — and quietly broke every coding
answer, because `{"answer": "def f(): return {1, 2}", ...}` has a brace *inside the answer
string*, and the last `{` lands there. I only caught this because I wrote the regression test
before trusting the fix.

**Fix** — stop pattern-matching wrappers. Scan the text for every position where a *complete*
JSON object parses, skip past each one found, and keep the last top-level object:

```python
while (i := text.find("{", i)) != -1:
    try:
        _, end = _DECODER.raw_decode(text, i)
    except ValueError:
        i += 1
        continue
    best = text[i:end]
    i = end   # skip past it, so a nested `{` is never a candidate
```

Skipping to `end` is what makes nested objects and braces-in-strings safe; keeping the *last*
match is what makes reasoning drafts safe. It handles `<think>`, markdown fences, chatty
preambles and any future wrapper convention without knowing about any of them.

**Fix to the process, too** — six wrapper shapes are now pinned in
`checks/check_json_extraction.py`, which runs in CI. This function had been rewritten three
times and silently broken twice; each break only showed up as a `malformed_response` in a live
run, days later.

**Takeaway** — the mirror image of #4: *deterministic* failure across varied inputs means the
bug is real and the common factor is the culprit. And a fix you can't test is a guess — the
brace-in-string regression would have shipped without that check file.

---

## 13. The classifier was deleting the text it was asked to summarize

This is the biggest one, and it corrects a conclusion I'd previously written down wrong.

**Symptom** — summarization queries failed the judge almost every time: **0-1 out of 5**, in
both the cascade and the always-max-tier baseline.

**My first, wrong diagnosis** — both paths failing ruled out "the small model is bad at
summarizing," so I looked at the judge. Its rubric ("decide if the answer is correct") is
built for tasks with one right answer, and summarization has no single ground truth, only
faithfulness. That reasoning was sound, so I added task-type-specific guidance to the judge's
prompt (don't penalise a summary for dropping secondary detail; judge translation on meaning,
not literal wording). It helped slightly. **I recorded that as the fix and moved on. It was
not the fix.**

**How I found the real cause** — re-measuring after unrelated work, summarization was still
1/5. So I stopped reading verdicts and read the *answers*:

> "I'm not aware of the current events in the stock market as my knowledge cutoff is
> December 2023, but I can suggest..."

That is not a bad summary. That is a model that **was never given the text**. The judge had
been right every time.

I printed what `classify()` actually hands downstream:

| Raw query | What the model actually received |
|---|---|
| `Summarize: The stock market saw significant volatility this week as investors reacted to new inflation data...` | `Summarize the main points from the paragraph about the stock market this week.` |
| `Summarize: The quick brown fox jumps over the lazy dog repeatedly, day after day...` | `Summarize the story about a fox and a dog.` |

There is no paragraph. There is no story. The classifier's `optimized_prompt` step — meant to
rewrite queries "to be clear and unambiguous" — had **paraphrased away the payload**, leaving
only a description of it. The model, given nothing to summarize, fell back to world knowledge.
This is the same failure I'd hit earlier with translation, where
`Translate 'Where is the nearest train station?' to Spanish` was rewritten into
`where is the nearest train station in spanish translation` and both tiers tried to *answer*
it. I'd patched translation with a prompt hint and never asked whether the same thing was
happening elsewhere.

**Root cause** — prompt rewriting is only safe when the query is **purely an instruction**.
Summarization and translation queries are instruction **+ payload**, and a paraphrase is free
to discard the payload — it's still a "clearer" sentence, just about nothing.

**Fix** — structural, not another prompt hint:

```python
PRESERVE_QUERY_TYPES = {"summarization", "translation"}
prompt = query if classification.type in PRESERVE_QUERY_TYPES else classification.optimized_prompt
```

Content-bearing task types bypass the rewrite entirely. No amount of prompt tuning makes a
paraphrase reliably payload-preserving; not paraphrasing does.

**Measured effect** (same queries, same judge, before → after, on the Groq-servable
qa + summarization subset — the only arm runnable that day, see #14):

| | before | after |
|---|---|---|
| summarization pass rate | 1/5 | **4/5** |
| qa pass rate | 4/5 | **5/5** |
| confidence ECE | 0.519 | **0.291** |

On the full 25-query benchmark afterwards, summarization reached 5/5 and ECE fell further to
0.131 — though part of *that* later drop was a separate judge fix, not this one (#17).

**Takeaway** — three of them, and this is the entry I'd actually want to be asked about:

1. **I documented a wrong root cause with confidence.** The judge-rubric reasoning was
   plausible, produced a small improvement, and was wrong. A partial improvement is the most
   effective disguise a wrong diagnosis has.
2. **I read the verdicts instead of the outputs.** The answer text said "I was never given the
   text" in plain English from the very first run. I was reading the judge's opinion of the
   answer rather than the answer.
3. **I fixed one symptom of a general bug.** Translation and summarization were the same bug.
   Patching the instance in front of me left the other one live for days. When you fix
   something, ask what else shares the mechanism.

---

## 14. Hitting two different real rate limits

**Symptom A — OpenRouter, mid-audit** — most OpenRouter-routed queries started returning
`unavailable`, having worked minutes earlier.

**How I found the cause** — called an OpenRouter model directly, outside the app:

```
RateLimitError: 429 - Rate limit exceeded: free-models-per-day.
X-RateLimit-Limit: 50, X-RateLimit-Remaining: 0, X-RateLimit-Reset: 2026-07-29T00:00:00Z
```

**Root cause** — not a bug. OpenRouter caps unpaid accounts at **50 free-model requests per
day, account-wide across every `:free` model** — not per model. A day of debugging, live UI
testing and two eval attempts burns that easily.

**Symptom B — Groq, later the same day** — a test script died with a raw traceback:
`groq.RateLimitError: ... requests per minute (RPM): Limit 30, Used 30. Please try again in 2s.`

**How I found the cause** — this one *was* a bug, and an embarrassing one: I had
`ModelUnavailable` handling specifically so a rate limit couldn't crash anything. But it lived
in `call_model()`, and `classify()` and `judge()` call the shared `call_json()` **directly**,
bypassing it. So the answering path degraded gracefully while the classifier — which runs on
*every single query* — could take down the entire app.

**Root cause** — error handling placed at a wrapper that not all callers go through. Two of
three call paths were unprotected, and the unprotected ones ran more often.

**Fix** — moved the 429/503 handling down into `call_json()`, where every path already goes.
While there, I split the two rate limits by their actual behaviour: Groq's per-minute limit
clears in seconds, so a transient 429 now waits briefly and retries once and usually just
succeeds; only a second failure raises `ModelUnavailable`. OpenRouter's daily cap survives the
retry and correctly degrades to skipping the tier. `classify()` falls back to a neutral
medium/qa classification, and an unavailable judge now accepts the answer marked
*unverified* rather than escalating — escalating would burn a bigger tier only to hit the same
broken judge one rung up.

**Takeaway** — when you add error handling, grep every caller of the thing you're protecting.
"I handled that" was true of one path out of three, and I only found out because the busiest
unprotected path finally got unlucky. Also worth stating plainly for anyone reproducing this:
free-tier ceilings are a real architectural constraint here, not a footnote. The upside is
that both limits validated `ModelUnavailable` under genuine sustained outages rather than
mocked ones.

---

## 15. My own check had an incomplete assertion

**Symptom** — after the fixes above, `check_cascade.py` started failing on an assertion I'd
written several tasks earlier.

**How I found the cause** — it accepted only `accepted`, `judged_fail` and `unavailable` as
valid final trace states. A run had legitimately ended on `malformed_response` — every tier up
to the ceiling failed to parse.

**Root cause** — the check, not the cascade. Ending on `malformed_response` had always been a
valid terminal state; my assertion had been incomplete from the day I wrote it, and no run had
happened to land there until now.

**Fix** — corrected the check. Deliberately *not* the cascade.

**Takeaway** — a newly failing test is not proof the code broke. Work out which one is wrong
before you "fix" the one that's shouting.

---

## 16. The savings metric was clamped, hiding its own worst case

**Symptom** — none visible. That was the problem.

**How I found the cause** — reading `metrics.py` during a cleanup pass:

```python
return max(0.0, (baseline - used) / baseline * 100)
```

**Root cause** — a cascade that escalates far enough can burn **more** active params than one
direct max-tier call. Expert-difficulty coding: 32B at tier 3, then 55B at tier 4 = 87B, versus
a 55B baseline. That's −58% "saved". The clamp silently reported it as 0%, so the routing's
genuine worst case could never appear in any number I published.

**Fix** — dropped the clamp; negative savings now show as negative, in the UI and the eval
report, with the reasoning in a comment. Updated the check that asserted `0 <= pct <= 100` and
added one asserting the expensive case reports negative.

**The fix immediately caught a second bug.** With the clamp gone, summarization queries started
reporting **−42%**. Tier 2 for summarization was `llama-3.3-70b-versatile` — 70B *dense*, i.e.
more active params than the 55B-active tier-4 model at the top of the ladder. A tier-2 slot
that costs more than the ceiling isn't a tradeoff, it's a routing bug, and the clamp had been
hiding it since the registry was written. Swapped to a 27B model.

**Takeaway** — same family as #3 and #11, and the reason I keep flagging it: this metric
pipeline has now produced three separate flattering-but-wrong numbers. A clamp, a default of
zero, a plausible-looking parameter count. None crashed. All made the system look better than
it was. Defensive defaults in a *measurement* path aren't defensive — they're a thumb on the
scale, and the one that hid −42% was doing it for months of commits.

---

## 17. My judge was failing correct answers — and the proof that it worked was wrong

The one I'd most want to be asked about, along with #13.

**Symptom** — running my standard prompt set after unrelated work, two obviously-fine answers
came back `judged_fail`:

```
query : A farmer has 17 sheep. All but 9 die. How many sheep are left?
answer: "The remaining number of sheep is 9, which is less than the original number of 17."
judge : FAIL — "The answer does not state the remaining number of sheep."
```

It states the remaining number of sheep. It is the fourth word.

**How I found the cause** — I'd been reading the judge's *verdicts* as data about the answers.
Reading them as data about the **judge** instead, a pattern appeared immediately: every false
failure was about form, not fact — "doesn't show the calculation process," "lacks context,"
"too simplistic," "doesn't provide a numerical solution" (about an answer containing a number).

**Root cause** — my judge prompt:

> *"You are a **strict** answer judge... decide if the answer is correct and **adequately
> addresses** the question."*

"Strict" invites rejection, and "adequately addresses" is undefined, so the model filled the
gap with the only rubric it had: completeness of presentation. It was grading essays.

**Fix** — replaced the vague instruction with explicit conditions, and ruled style out of scope:

> PASS if the answer is factually correct and answers what was asked — even if it is terse,
> verbose, informally worded, or shows no working.
> FAIL if it is factually wrong, answers a different question, refuses, or only describes how
> one *would* find the answer instead of giving it.
> Do not fail an answer for style, formatting, length, or missing explanation.

The "only describes how one would find the answer" clause is deliberate — that's a real failure
mode I'd seen (a model responding to "what is the 47th digit of pi" with a tutorial on the
Chudnovsky algorithm), and it needed to stay a FAIL while terseness stopped being one.

**Then the fix exposed something worse.** With the judge no longer failing things for style, I
re-ran my standard set, and the pi question came back **accepted, answer "7"**. I'd been using
that exact question since the earliest edge-case testing as my canonical hallucination example:
*"it answers 7 with confidence 10, and once the judge was mandatory it correctly caught it."*
It's written up that way in this log and in my interview notes.

So I actually checked:

```
π = 3.14159265358979323846264338327950288419716939937510...
             the 47th digit after the decimal point is 7
```

**The model was right. It had always been right.** And what I had recorded as "my judge
correctly catching a hallucination" was my judge **rejecting a correct answer for not showing
its working** — the exact bug in this entry, sitting in plain sight in my own documentation,
being cited as evidence the system worked.

**What this cost, and what saved it.** The overconfidence conclusion the pi example was
supporting is still correct — but it's supported by the ECE measurement (top confidence bucket
claims 0.98, delivers 0.75), not by that anecdote. The measurement survived; the story didn't.
If I'd only had the story, I'd have had nothing.

**Takeaways** — three, and they're the ones I'd actually defend in an interview:

1. **I never verified my own test case.** I picked "the 47th digit of pi" *because* I assumed
   small models get it wrong, and then used the model's answer as evidence for the assumption
   that made me pick the question. Circular, and it stood for the entire project because
   checking took one line of Python I never ran.
2. **A component that fails safe still fails.** A judge that wrongly rejects looks *responsible*
   — it produces cautious escalations, not visible errors. It was silently pushing correct
   answers up the ladder, spending compute to replace right answers, and every single false
   rejection read as the system being appropriately careful.
3. **Evidence that a thing works is the last place you look for a bug in it** — which is
   exactly why it's where this one lived. Same shape as #13: a wrong conclusion, documented
   confidently, protected by the fact that it *looked* like a success.

---

## 18. The judge wasn't judging — it was agreeing. And my benchmark couldn't tell.

**Symptom** — after fixing the over-strict judge in #17, the benchmark improved across the
board: pass rate 68% → 72%, compute saved 71.5% → 76.6%, ECE 0.23 → 0.13. Every number moved
the right way.

**Why I didn't trust it** — the judge is the *scorer*. I had just made the scorer more
permissive and the score went up. Those two facts are compatible with "the router got better"
and equally compatible with "the grading got easier", and **nothing in my benchmark could
distinguish them**, because every query in it is scored by the judge and nothing else. My
"pass rate" was never measuring correctness. It was measuring agreement with the judge.

**How I found the cause** — I built the thing I was missing: 14 hand-labelled
`(question, answer, should_pass)` cases with known-correct verdicts
(`eval/judge_ground_truth.py`), half correct answers phrased awkwardly, half wrong answers
phrased fluently. Then I measured the judge's two error types separately, because they are not
equally bad:

- **false pass** — accepts a wrong answer. The user is handed something incorrect and the
  cascade stops. Harmful.
- **false fail** — rejects a correct answer. The cascade escalates unnecessarily. Wasteful.

The result was much worse than the benchmark implied:

```
false passes (accepted a wrong answer): 4/7 = 57%
false fails  (rejected a right answer): 2/7 = 29%
```

**Root cause** — shown a wrong answer, the judge didn't verify it, it *ratified* it and
invented a justification afterwards:

| shown | judge said |
|---|---|
| `17 * 23 = 371` | *"The answer to the multiplication of 17 and 23 is correctly stated as 371."* |
| ball costs `$0.10` | *"The cost of the ball is correctly given as $0.10, which is $1.00 less than the bat's cost..."* |
| `"No information is given about any sheep deaths"` | *"The answer correctly states that the information provided is insufficient."* |

That second reason is barely coherent — it's reverse-engineering support for a conclusion it
had already adopted. The judge was never doing the arithmetic. It was pattern-matching
"confident, well-formed answer" to "pass". My #17 fix had told it not to judge on style, and
with style removed it had nothing left but agreement.

**Fix** — make the judge commit to its own answer **before** it's allowed to give a verdict, by
putting an `own_answer` field first in the required JSON:

```
1. Answer the question YOURSELF, independently, without being influenced by the proposed
   answer. For arithmetic or logic, actually work it out.
2. Compare your answer to the proposed one.
3. Give a verdict.
```

`own_answer` is never read by any code. Its entire purpose is to force an independent
commitment into the context before the verdict token is generated, so the model has to notice
its own answer differs before it can approve. Measured effect:

| | v2 (style-blind) | v3 (solve-first) |
|---|---|---|
| **false pass** (harmful) | **57%** | **29%** |
| false fail (wasteful) | 29% | 29% |

Both arithmetic ratifications are now caught. It's not fixed — 29% false passes is still bad,
and a small model judging a small model has a real ceiling — but the harmful error rate halved
at no cost to the wasteful one.

**Takeaways:**

1. **A metric scored by the component you're changing cannot evaluate that change.** This is
   the most general lesson in this whole log. Every "improvement" to the judge automatically
   improved the benchmark, in the same direction, regardless of whether it was an improvement.
   The fix wasn't a better judge, it was *a second measurement the judge doesn't control*.
2. **Split your error types before you set a target.** "72% pass rate" hid the fact that the
   errors were overwhelmingly the harmful kind. One aggregate number let a 57% false-pass rate
   look like a good result.
3. **Adding an unused field changed the model's behaviour**, because generation order is
   causal — what's in the context before a token influences that token. `own_answer` is dead
   data to my code and the most load-bearing line in the prompt.
4. This is the third time in this project the same shape appeared (#11, #16, and now this): a
   number that looked good, wasn't checked because it looked good, and was measuring something
   other than what its name said.

## 19. The same bug, a third time: "36% compute saved" on an answer that was rejected

**Symptom** — a screenshot from the deployed app. Query: *"What is the 47th digit after the
decimal point of pi?"* Both tiers returned an explanation of the Bailey–Borwein–Plouffe formula
instead of a digit. The judge correctly rejected both. The banner said **"No tier produced an
answer the judge accepted."** Directly beneath it, the cost panel said **"36% compute saved."**

**Root cause** — the gate on the savings metric asked the wrong question:

```python
produced_answer = result.answer != "No model produced a usable answer."
```

That asks *"is there a string?"*, not *"did this work?"*. A judge-rejected answer is still a
string, so it sailed through the gate. The run burned 35B active params and delivered nothing,
and the UI congratulated it for spending less than the maximum.

**Why it survived two previous fixes** — this is the same defect as #11, which I fixed in the
eval harness, and #16, which I fixed in the metric itself. Each time I patched the caller in
front of me. The predicate was never defined in one place, so `src/app.py` and
`eval/run_eval.py` had each independently rolled their own copy of it, and each copy was
wrong in the same way.

**Fix** — define it once, on the object every caller already holds:

```python
@dataclass
class CascadeResult:
    ...
    @property
    def accepted(self) -> bool:
        return bool(self.trace) and self.trace[-1].status == "accepted"
```

Both callers now route through it, and `checks/check_metrics_formatter.py` asserts that a trace
ending in `judged_fail` is not accepted.

**The number moved in the direction I didn't expect.** I assumed excluding failures would push
average savings down, because I thought of them as bad results. The opposite happened — savings
went *up*, because failed runs are the ones that escalate through the most tiers and therefore
carry the *highest* cost. Counting them was dragging the average down with the price of
failure. The old number wasn't flattering the router; it was quietly mixing two different
populations.

**Takeaways:**

1. **A bug fixed at three call sites was never fixed — it was reproduced.** The signal I
   ignored twice: I was writing the same predicate in more than one file. That's the moment to
   move it, not the third time it bites.
2. **"Is there output?" and "did it succeed?" are different questions**, and conflating them is
   easy precisely because the happy path makes them agree.
3. Predict which way a metric should move before you re-measure. Being wrong about the
   direction is how you find out you misunderstood what it was averaging over.

---

## 20. The cascade was escalating *down* the price curve

**Prompted by a question I couldn't answer** — "why aren't you using the bigger free models,
like the 120B ones?" I assumed the answer was "I already am, at tier 3." Then I actually
checked what the tiers cost, and found the ladder was inverted.

**What I found** — I had been ranking tiers by *active parameters*, on the reasonable-sounding
theory that active params proxy compute and compute proxies cost. Published per-token rates for
the same open-weight models say otherwise:

| tier | model | active params | $/1M in | $/1M out |
|---|---|---|---|---|
| 1 | `llama-3.1-8b-instant` | 8B | 0.050 | 0.080 |
| 2 | `qwen/qwen3.6-27b` | 27B | **0.300** | **2.000** |
| 3 | `openai/gpt-oss-120b` | 5.1B | **0.037** | **0.170** |
| 4 | `nemotron-3-ultra-550b-a55b` | 55B | 0.500 | 2.200 |

**Tier 3 was roughly 12x cheaper per output token than the tier 2 it escalated up from** — and
it's the larger, generally stronger model. `gpt-oss-120b` is a sparse MoE: ~117B total
parameters, ~5.1B active per token. A 27B *dense* model activates every one of its parameters on
every token; the 120B model activates 4% of its.

**Why this mattered more than it looks** — it compounded with `CEILING_TIER`, which stops easy
and medium queries at tier 2. The two settings together meant the cascade was **structurally
incapable of reaching a model that was simultaneously better and cheaper.** That is the actual
reason the pi query in #19 failed: not that the ceiling was too low, but that the wrong model
was sitting under it.

**Fix** — swap tiers 2 and 3 for the four task types where the inversion existed, so the ladder
is monotonic in real price. Then assert it, in `checks/check_registry.py`:

```python
assert costs[tier] >= costs[tier - 1], (
    f"{task_type}: tier {tier+1} is cheaper than tier {tier} "
    f"— escalation must not move down the price curve")
```

I also replaced the invented `$0.02 per active-billion-params` constant in `metrics.py` with the
actual published rates, because that constant is what had hidden the inversion: it *derived*
price from active params, so by construction it could never disagree with them.

**The two metrics genuinely disagree, and both are kept.** Active params say tier 1 (8B) costs
more than tier 2 (5.1B); price says the opposite. Neither is wrong — they measure different
things, and a sparse model is exactly where they come apart. The ladder is ordered by price, the
UI shows both, and `check_registry.py` deliberately asserts monotonicity only on price.

**A side effect worth having:** the swap moved tier 2 for four task types from OpenRouter onto
Groq. OpenRouter's free tier is capped at 50 requests/day account-wide and was the single most
common cause of the app failing outright; Groq's limit is per-minute and clears on its own.
Cheaper, stronger, and less exposed to the quota that actually breaks things.

**Takeaways:**

1. **"Bigger model" and "more expensive model" stopped being synonyms when MoE arrived.** The
   cheapest model in this ladder by published rate has the most parameters.
2. **A derived metric can't contradict what it's derived from.** My cost estimate was a function
   of active params, so it could never have caught an error in using active params as cost. It
   took an external source of prices to see it.
3. The design invariant — *escalating costs more* — was assumed everywhere and asserted nowhere,
   for the whole life of the project.

---

## 21. Two identical benchmark runs scored 72% and 84%

**How this surfaced** — before changing `CEILING_TIER` I wanted evidence, so I ran the
benchmark twice: once with the shipped ceiling, once with it lifted to tier 4. The lifted run
scored 100% against the shipped run's 72%, and seven queries "improved".

**Why I didn't believe it** — several of the "rescued" queries had been accepted at a *lower*
tier in the lifted run than the tier the shipped run stopped at. A higher ceiling cannot cause a
query to succeed at tier 1. Those weren't rescues; they were the judge returning a different
verdict on an identical query.

**What I measured instead** — I re-ran the *same* configuration twice:

```
sweep A, replicate 1:  18/25 = 72%
sweep A, replicate 2:  21/25 = 84%
```

Same code, same queries, same ceiling. **A 12-point spread, with 5 of 25 queries flipping
verdict.** The classifier is non-deterministic too, so a query's difficulty label — and
therefore which ceiling even applies to it — changes between runs.

**Consequence** — my benchmark cannot resolve any change smaller than about 12 points, and every
single-run before/after comparison in this project's history is weaker evidence than I treated
it as. At n=25 with a stochastic judge and a stochastic classifier, one sweep is an anecdote.

**What I did about the ceiling** — nothing, and that's the finding. The experiment couldn't
justify raising it, and once #20 put the right model underneath it, raising it stopped being
attractive anyway: the ceiling now stops at the strongest cheap model in the ladder rather than
below it. Lifting it would have cost `8 + 5.1 + 27 = 40.1B` per escalation to buy a pass-rate
delta smaller than the noise floor.

**Takeaways:**

1. **Measure your noise floor before you measure your effect.** I nearly shipped a ceiling
   change and a "72% → 100%" claim on top of a 12-point measurement error.
2. **A result that's too good is data about your instrument**, not about your system.
3. The honest headline for a benchmark like this is a range from repeated runs, not a point
   estimate quoted to one decimal place.
4. This is the second time the fix was *not* the change I set out to make (see #19 → #20). The
   investigation was worth more than the hypothesis.


## 22. The tier swap left a second dict pointing at the old ladder

**Symptom** — none visible. Every check passed, the benchmark was fine, and the app worked. This
one was found by reading the code after fixing #20 and asking what *else* assumed the old tier
order.

**What was wrong** — `STARTING_TIER` decides where a query *enters* the ladder:

```python
STARTING_TIER = {"easy": 1, "medium": 1, "hard": 2, "expert": 3}
```

Expert queries started at tier 3. That was correct when tier 3 held the largest model. After the
#20 swap it meant every expert query skipped tier 2 — the sparse 120B model — to open on tier 3,
the dense 27B one. The skipped tier was **cheaper and larger**:

| task type | old entry (tier 3) | new entry (tier 2) | |
|---|---|---|---|
| qa | qwen3.6-27b — 27B total, $0.430/1k | gpt-oss-120b — 117B total, $0.038/1k | **11.4× cheaper** |
| summarization | qwen3.6-27b — 27B total, $0.430/1k | gpt-oss-120b — 117B total, $0.038/1k | **11.4× cheaper** |
| reasoning | nemotron-super — 120B total, $0.089/1k | gpt-oss-120b — 117B total, $0.038/1k | 2.3× cheaper |
| translation | gemma-4-26b — 26B total, $0.075/1k | gpt-oss-120b — 117B total, $0.038/1k | 2.0× cheaper |

Paid on **every** expert query, needed or not, to get a smaller model.

**Why the existing check didn't catch it** — `check_registry.py` asserts the ladder is monotonic
in price. It is. The bug wasn't in the ordering, it was in *where the ladder gets entered*, and
nothing looked at that. A passing check on the adjacent property is worse than no check, because
it reads as coverage.

**The check I wrote first was also useless, and I only found out because I tested it.** My first
attempt asserted the entry tier is the cheapest tier within its own band. Expert's band is
[3, 4], and tier 3 *is* cheaper than tier 4 — so it passed with the bug still present:

```
$ python -m checks.check_registry     # with the buggy expert=3 restored
registry check passed.                # ...it didn't catch anything
```

The bug is about tiers skipped *before* the band, which my assertion never looked at.

**Finding the invariant that actually discriminates** — "never skip a cheaper tier" is wrong: the
price ladder is monotonic, so every skipped tier is cheaper, and that rule would forbid starting
above tier 1 at all. Starting high is a deliberate bet that a cheap model is too weak to be worth
calling.

What makes expert=3 different from hard=2 is that the skipped tier wins on **both** axes:

- `hard` starts at tier 2, skipping tier 1 (8B total, cheaper). Cheaper but *smaller* — a real
  tradeoff, and a legitimate bet.
- `expert` started at tier 3, skipping tier 2 (117B total, cheaper). Cheaper **and bigger** —
  no tradeoff at all, just strictly worse.

So the rule is: **never skip a tier that dominates your entry tier on both cost and size.** That
required adding `total_params_b` to the registry, which the project had deliberately avoided
tracking (#3) because *active* params are what drive cost. Both are needed: active params for
what a token costs, total params for what the model knows. They are answers to different
questions and I had been treating one as a substitute for the other.

```
$ python -m checks.check_registry     # with the buggy expert=3 restored
AssertionError: expert/qa: starts at tier 3 (qwen/qwen3.6-27b, 27B total, $0.00043/query)
but skips tier 2 (openai/gpt-oss-120b, 117B total, $0.00004/query) — the skipped tier is
both cheaper and larger, so skipping it is strictly worse
```

**Takeaways:**

1. **A config change has a blast radius, and it is every other constant that encoded the same
   assumption.** #20 changed what a tier number *means*; `STARTING_TIER` still spoke the old
   language. Grep for what else reads the thing you just redefined.
2. **Test that your check fails on the bug it was written for.** Mine didn't, and it printed
   "passed" while the bug sat two lines away. This is the same lesson as #2, where a
   catalog-printing script that never asserted let a delisted model ID through for months.
3. **Two metrics I'd treated as one.** #3 concluded "track active params, not total" and that was
   right for cost. It quietly became "total params don't matter," which is wrong — they're the
   capability proxy, and the invariant here can't be written without both.


## 23. The cheapest model in the system was making the most expensive mistake

**Symptom** — a screenshot of the deployed app. Query: *"What is the 47th digit after the decimal
point of pi?"*, routed as **`coding` / `hard`**. Coding runs on OpenRouter at every tier, the free
daily cap was exhausted, so every tier returned `unavailable` and the query failed with nothing to
show. A question about a number was sent to a ladder of code-specialised models.

**First instinct was wrong** — I assumed the classifier reliably mislabels that query and went to
fix the prompt. It doesn't. Twelve runs of the same query never once returned `coding`:

```
WITH quotes  {'reasoning/medium': 2, 'qa/hard': 3, 'qa/medium': 1}
no quotes    {'reasoning/expert': 2, 'qa/easy': 1, 'reasoning/medium': 3}
```

The `coding` label was a rare draw. But look at what *is* consistent: nothing. The same query came
back easy, medium, hard **and** expert. Difficulty sets both the entry tier and the ceiling, so
that query was being routed differently on essentially every submission.

**Root cause** — the classifier ran on `llama-3.1-8b-instant`, picked because it's the cheapest
model on Groq and classification is "just a cheap pre-step". That reasoning priced the *call* and
ignored the *consequence*. Classification is the only decision in the system that can send a
query to an entirely different provider — and when that provider is out of quota, a
classification error isn't a slightly worse answer, it's no answer at all.

**Measurement** — the benchmark set is grouped five-per-type, so the intended type is ground
truth. 25 queries x 2 repeats, both candidates:

| classifier | type accuracy | type stable across repeats | difficulty stable |
|---|---|---|---|
| `llama-3.1-8b-instant` | 94% (47/50) | 24/25 | **15/25** |
| `openai/gpt-oss-120b` | **100%** (50/50) | **25/25** | **22/25** |

The 8B model's confusions were `qa->reasoning` twice and `reasoning->qa` once — including
*"How does a hash map achieve O(1) lookup?"*, which it called reasoning on both attempts.

**This is one of the few numbers in the project that isn't circular.** It's scored against the
benchmark's own type labels, not against the judge. Compare #18, where every judge-scored metric
moved together and none of them could distinguish "better router" from "easier grading."

**Cost of the fix** — ~1.8x per classifier call ($0.037 vs $0.021 per 1k) and ~250ms median added
latency (711ms vs 461ms) against a multi-second end-to-end time. Bought with the same sparse-MoE
economics as #20: `gpt-oss-120b` is 117B total but ~5.1B active, so "use the much bigger model
for the most consequential decision" costs cents.

**Takeaways:**

1. **Price the consequence of a call, not the call.** The classifier was optimised as the
   cheapest step in the pipeline while being the step whose errors cost the most. Cost per token
   and cost per mistake are different quantities and I had only been tracking one.
2. **A rare wrong answer and a wildly unstable right answer are the same bug.** Chasing the
   `coding` outlier would have missed that difficulty was unstable on 10 of 25 queries.
3. **Reproduce before fixing.** Twelve runs cost about a minute and disproved the hypothesis I was
   about to write code against.


## 24. A whole task type with no fallback, and a benchmark that measured the wrong thing

**Symptom** — coding queries failed completely, repeatedly, on the deployed app. Every tier
`unavailable`, no answer, nothing to show.

**Root cause** — coding ran on OpenRouter at all four tiers. OpenRouter's free tier caps
`free-models-per-day` **account-wide**, so when it runs out there is no coding path at all. Every
other task type had Groq at tiers 1-2 and degraded gracefully; coding had no fallback and simply
died. A single provider outage took out 20% of the system's functionality.

```
429  cohere/north-mini-code:free      Rate limit exceeded: free-models-per-day
429  poolside/laguna-xs-2.1:free      Rate limit exceeded: free-models-per-day
429  poolside/laguna-s-2.1:free       Rate limit exceeded: free-models-per-day
429  nvidia/nemotron-3-ultra-550b:free Rate limit exceeded: free-models-per-day
```

**Choosing replacements without the judge** — the judge false-passes ~29% of wrong answers (#18)
and swings ~12 points run to run (#21). On five coding queries that noise is larger than any real
difference between models, so it cannot rank them. Generated code, though, can simply be **run**:
`eval/compare_coding_models.py` gives each model tasks with assertions and executes the result.
Ground truth, no verdicts.

**The first version of that benchmark was wrong, in this project's favourite way.** It reported:

```
llama-3.3-70b-versatile   0/12 =  0.0%
llama-3.1-8b-instant      8/12 = 66.7%
openai/gpt-oss-20b       12/12 = 100.0%
```

which reads as "llama-3.3-70b cannot write code". It writes code fine. It never emits the JSON
envelope `ANSWER_SYSTEM_PROMPT` requires, so no response ever parsed. Splitting the two failure
modes apart:

| model | JSON envelope ok | code correct (of parseable) |
|---|---|---|
| `openai/gpt-oss-20b` | **12/12** | 12/12 |
| `openai/gpt-oss-120b` | **12/12** | 12/12 |
| `qwen/qwen3.6-27b` | 11/12 | 11/11 |
| `llama-3.1-8b-instant` | 8/12 | 8/8 |
| `llama-3.3-70b-versatile` | **0/12** | unmeasured |

**Every model that parsed wrote 100% correct code.** So the tasks are too easy to rank coding
ability, and the number I nearly published as a coding ranking was entirely a JSON-compliance
ranking. Both matter — a model that can't produce the envelope is unusable in this cascade no
matter how good its code is — but they are different facts, and one number for both is the same
conflation as #19.

**What this does and doesn't establish.** The OpenRouter code-specialised models were unreachable
throughout, so this shows the Groq models are *viable*, not that they *beat* the incumbents.
`laguna-s` stays at tier 3 rather than being dropped on the assumption that specialised beats
general. A cron job runs the full comparison after the next quota reset.

**And then coding became reachable enough to find the next bug.** With tiers 1-2 on Groq, a
palindrome query got working code rejected at tier 1:

> *"The provided answer is correct in functionality but the proposed answer includes an incorrect
> code comment."*

The judge said the code was functionally correct and failed it anyway — over a comment. `TYPE_GUIDANCE`
had entries for summarization and translation, added after each was measured failing, and none for
coding, because coding had never been reachable enough to measure. Step 1 of the base prompt
("answer the question yourself first") means *write your own implementation* for code, and two
correct programs rarely look alike, so "differs from mine" was reading as "wrong".

`checks/check_judge_coding.py` scores the judge against execution. Baseline: 3 of 4 working
implementations rejected, with reasons like *"fails to explain the implementation of slicing"* —
grading the explanation, not the code.

**I stopped short of claiming the fix works.** Four complete runs of the *identical* prompt gave
false-fail counts of 0, 2, 2 and 0 out of 4, and false-pass counts of 2, 1, 1 and 0. Eight cases
cannot distinguish those from each other, so the check asserts loose bounds, refuses to assert at
all on a partial run, and carries a comment telling the next person not to tune the prompt
against it. The guidance is justified by the mechanism it removes — explanation-grading, which
was reproducible and is now absent from the reasons — not by a number I can't reproduce.

The temptation here was strong and worth naming: the final run came back 0/4 and 0/4, a perfect
score, and it would have been very easy to write that down as the result. It's one draw from a
distribution I had already watched produce 2/4 twice. **A good number from an instrument you have
just proven is noisy is not evidence; it's the same noise, pointing somewhere flattering.** That
is the exact mistake #21 exists to record, and I nearly made it again two entries later.

**Takeaways:**

1. **Every task type needs a path that survives its primary provider.** Four tiers of redundancy
   are zero redundancy if all four are one account's quota.
2. **When a benchmark reports 0%, suspect the harness before the subject.** A capable model
   scoring zero is nearly always a measurement fault.
3. **Being able to run the artifact beats any judge.** Execution is the only ground truth in this
   project that costs nothing and never drifts. Where output is executable, use it.
4. **Pacing is part of correctness in a rate-limited harness.** Unjudged cases weren't counted as
   errors, so hitting the limit made the score look *better*. A measurement that improves when it
   collects less data is broken.


## 25. "This metric doesn't depend on the judge" — the per-query value didn't, the average did

**What the README claimed**, in a table with a column headed *Noise-sensitive?* marked `no`:

> The two savings figures are the ones to trust: they depend only on which tiers ran, not on the
> judge's opinion of anything.

**Why it was wrong.** Each *per-query* savings value really is judge-free — it is computed from
which tiers ran and what those models cost. But the reported number is a *mean*, and
`eval/run_eval.py` accumulates it under `if cascade_ok`. The judge decides which runs enter the
average. The judge therefore selects the population, and the mean inherits every bit of the
judge's instability even though not one of its inputs does.

Worse, I had already documented the proof and not connected it. Entry #19 records that adding
that exact gate **moved the number** — savings went *up*, because rejected runs escalate furthest
and cost most. A number that moves when you change the judge's gate is not independent of the
judge. I wrote both claims within a few hours of each other.

**The general shape**, which is worth more than the fix: *a statistic can be independent of X at
the level of each observation and dependent on X at the level of the sample.* Selection is a
channel. Checking that no term in your formula mentions the judge does not tell you the judge
isn't in the answer — you also have to ask who chose the rows.

**Three smaller instances of the same carelessness, found in the same pass:**

- The tier-distribution row read `14 at tier 1, 8 at tier 2, 0 above` in a table about 25
  queries. It summed to 22 — the accepted-only subset — so the three failures vanished from the
  record of where the cascade actually spent money, and "0 above" was false: one query climbed to
  tier 3 before being rejected. Where a query *ran* and whether it *passed* are different
  questions.
- `Illustrative $ saved: $X across 25 queries`, summed under `if cascade_ok`, i.e. over 22.
- `checks/check_judge_coding.py` incremented its denominators before the `continue` that skips an
  ungraded case, so a rate limit inflated the denominator without being able to reach the
  numerator — biasing the judge's error rates *downward*, in the file that exists to measure the
  judge honestly.

**Takeaways:**

1. **For any rate, name the population out loud.** Three of these four bugs are one question
   unasked: *what exactly is the denominator, and who decided membership?*
2. **"Independent of X" needs checking at the sample level, not just the formula level.**
3. Documentation drifts fastest right after the code is most correct. Every one of these was
   written *while* fixing something real, which is exactly when it feels safe not to re-read.


---

*Entries are appended as new issues surface. Nothing here is retro-edited except where a
conclusion was later proven wrong — those corrections are called out inside the original entry
and cross-linked to the entry that overturned it (see #7 → #17, and #13).*
