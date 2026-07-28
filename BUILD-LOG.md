# Build Log — issues I hit while building LadderLLM

My own record of what actually went wrong (or would've gone wrong) while building this,
and how I reasoned through each one. Kept so I can talk through it later — what broke,
why, how I found the cause, how I fixed it.

## Building `llm_client.py`

**Issue: Groq and OpenRouter don't share an error type, even though OpenRouter's API is
"OpenAI-compatible."**
I wanted one `try/except` around any model call to catch rate limits (429) and
capacity errors (503) from either provider. But the `groq` SDK and the `openai` SDK
(I use `openai` for OpenRouter, since its endpoint is OpenAI-compatible) are two separate
libraries — each raises its *own* `APIStatusError` class. If I'd only caught
`openai.APIStatusError`, a Groq-side outage would've slipped through uncaught and crashed
the app instead of gracefully skipping that tier. Fix: I import both exception classes
(`GroqAPIStatusError`, `OpenAIAPIStatusError`) and catch them as a tuple in `call_model()`.

**Design call: where does JSON-retry logic live?**
Both my classifier and my judge need the same pattern — small models sometimes wrap JSON in
markdown fences or add stray commentary, so I strip fences, validate with Pydantic, and if
that fails, retry once with a stricter "reply with ONLY JSON" instruction. I didn't want to
write that logic twice (a bug fix later would need to happen in two places and drift out of
sync), so I pulled it into one shared `call_json()` helper that both modules call into.

## Building `registry.py`

**Issue I avoided by checking first: model IDs in my own design notes might already be stale.**
I'd sketched out a tier x task-type model grid before writing any code. Free-tier model
catalogs on Groq and OpenRouter change over time — an ID I picked could 400 ("model not
found") the first time I actually called it, with no warning until runtime. Instead of
hardcoding the grid and finding out the hard way, I wrote a small discovery script that hits
both providers' live model-list endpoints first, and cross-checked every grid entry against
what's actually available *right now* before writing a single line into `registry.py`.

**Bug I caught before it shipped: MoE models don't burn their full parameter count per query.**
`openai/gpt-oss-120b` sounds like "the big 120B model," but it's actually a
Mixture-of-Experts model — only ~5.1B parameters are *active* per token, the rest sit idle.
If I'd recorded 120 instead of 5.1 in the registry, my compute-savings metric would've been
quietly wrong — it would've overstated how much compute the system actually saved, in the
model's favor. I caught this by checking what "active params" actually means for MoE
architectures before filling in the numbers, not after. The registry field is explicitly
`active_params_b`, documented as active-not-total, with a comment explaining why per model.

## Building `cascade.py`

**Real bug hunt: my end-to-end cascade test failed on the very first run.**

```
Answer: No model produced a usable answer.
Tier used: 2
Trace:
  tier=1 model=llama-3.1-8b-instant status=malformed_response confidence=None
  tier=2 model=qwen/qwen3.6-27b status=malformed_response confidence=None
```

Both tier 1 and tier 2 came back `malformed_response` — meaning my retry-once JSON parsing
failed *twice in a row*, on two different models, in the same run. My first instinct was
"there's a bug in my JSON parsing." Instead of immediately rewriting code, I isolated each
layer to find out exactly where it broke:

1. Called the raw model directly with the identical prompt → got back clean, valid JSON.
2. Called my JSON-parsing helper with that same output → parsed fine.
3. Called the full provider-dispatch function with the real registry config → parsed fine.
4. Reproduced the exact classify → optimize-prompt → answer chain the cascade uses,
   end-to-end, three times in a row → parsed fine every time.
5. Re-ran the original failing test again, with zero code changes → passed cleanly.

**What I concluded:** it wasn't a bug in my code at all — it was the exact "classifier/answer
JSON is malformed ~10-15% of the time with small models" behavior I'd already read about
before building this. Small models occasionally just return broken JSON, probabilistically,
even with a good prompt. If I'd started editing `_strip_fences()` or my Pydantic schemas the
moment I saw the failure, I'd have "fixed" something that was never broken, and possibly
introduced a real bug chasing a phantom one. The actual validation here was that my
error-handling design — log it as `malformed_response`, move to the next tier, and if
everything fails, return a clear message instead of crashing — means this failure mode is a
non-event for the end user instead of a crash. (If it had failed on *every* run instead of
intermittently, that would've pointed at a real bug in the parsing code — it didn't.)

## Building `app.py` (Streamlit)

**Issue I avoided by knowing Streamlit's execution model up front: it reruns the whole
script on every interaction.**
Streamlit doesn't just re-render — it re-executes the entire Python script top to bottom on
every single interaction, including every keystroke typed into a text box. If I'd written
`result = run_cascade(query)` directly in the script body, it would've fired a full multi-model
LLM cascade on every character typed into the input field, not just on submit. I gated the
call behind `if submit and query`, and stashed the result in `st.session_state` so it survives
the rerun that clicking Submit itself triggers, instead of getting lost or re-triggered.

**Real bug hit: `ModuleNotFoundError: No module named 'src'` the first time I opened the app
in the browser.**

```
File "/home/mithun/Desktop/ladder-llm/src/app.py", line 3, in <module>
    from src.cascade import run_cascade
ModuleNotFoundError: No module named 'src'
```

Every other module imports fine with `from src.cascade import ...` — my checks all pass, my
package layout is correct. The difference is *how* the file gets launched. Everywhere else I
run things as `python -m checks.check_whatever` from the project root, which puts the project
root on Python's import path automatically. But `streamlit run src/app.py` launches the file
directly, the same way `python src/app.py` would — and when Python runs a script directly, it
only adds *that script's own directory* (`.../ladder-llm/src`) to `sys.path`, not the project
root above it. So `src.cascade` can't resolve, because there's no `src` folder *inside*
`src/` — the `src` package Python needs to see is the one one level up, and that level never
got added to the path.

Fix: launch with `PYTHONPATH=<project root> streamlit run src/app.py` instead of just
`streamlit run src/app.py`. Setting `PYTHONPATH` explicitly adds the project root to the
import path regardless of how the script itself gets invoked — no code changes needed, since
this is purely about how the interpreter resolves imports, not a flaw in the package
structure itself.

## Edge-case hardening pass

**Real bug hit while testing the "unavailable tier" path: I broke the cascade with my own
test setup, and it taught me something about my error handling.**
I tried to simulate an OpenRouter outage by pointing a registry entry at a made-up model ID
(`totally-fake/does-not-exist:free`). That's not what a real outage looks like, though — it
came back as `400 BadRequestError` ("not a valid model ID"), not `429`/`503`. My code only
catches `429`/`503` as `ModelUnavailable`; the `400` propagated up and crashed the whole
cascade. My first reaction was "that's a bug, catch more error codes." On reflection, it
isn't — a `400` from a bad model ID means *my own registry is broken*, which is a real bug
that should crash loudly during development, not get silently swallowed the same way as a
transient free-tier outage. Since `registry.py` is built from a live model-list check, a
`400` shouldn't happen in production; if it ever does, I want to see it, not have it quietly
vanish into a trace as "unavailable." I left this uncaught on purpose. To actually test the
outage path, I simulated a real `503` by mocking the provider call directly instead of using
a fake model ID — that confirmed the tier gets skipped, logged as `"unavailable"`, and the
cascade moves on cleanly.

**Real pattern confirmed, not just theorized: self-reported confidence was ≥9 on every single
live call I made, including a hallucinated answer.**
I'd read in my own notes that small models tend to be overconfident, but I wanted to see it
myself before trusting it. I asked "what's the 47th digit after the decimal point of pi?" —
a question small models are known to get wrong — and the model answered "7" with confidence
10. Across 5 separate live queries during this build, confidence never once landed in the
5-7 "fire the judge" band; it was always 9 or 10, correct or not. That's not a coincidence at
5-for-5 — it's the calibration problem stated plainly: a model's self-rating measures how
fluent the answer *sounds*, not whether it's *right*. I flipped on the judge-always fallback
mode (skip the confidence fast-paths entirely, always fire the judge) as a result. Re-running
the same pi-digit question afterward: same wrong answer ("7"), but this time the judge caught
it and marked the tier `judged_fail` with an accurate explanation of why it was wrong, instead
of the confidence score alone waving it through.

**Real bug hit while re-running my full check suite before calling this done: the classifier
flip-flopped on a query it had classified correctly earlier.**
"What is a closure in Python?" came back as `type=coding` this time, failing my own test
that expects `qa`. My first instinct was to loosen the assertion — but I'd already written a
note to myself in the classifier task that a failed type assertion is "a real signal about
prompt quality," not something to paper over by relaxing the check. Before touching anything,
I ran the exact same query through the classifier 6 more times in a row: 6/6 came back `qa`.
So it wasn't broken, and it wasn't consistently broken either — it was genuine sampling
variance on a query that sits right on the boundary between two categories ("closure in
Python" is a programming concept, but the user is asking for an explanation, not code). I
fixed the actual ambiguity: added one clarifying line per type to the classifier's system
prompt, explicitly stating that conceptual questions about programming are `qa`, not `coding`,
unless code is actually being requested. Re-ran the same query 8 times after the prompt
change: 8/8 came back `qa`. Root-caused a fuzzy category boundary in the prompt, rather than
patching the symptom by weakening the test.

## Setting up CI

**Real issue hit: a check that never calls a live API still needed API keys to even import.**
I wanted `check_metrics_formatter.py` (pure logic — no network calls at all) to run in CI
without needing real secrets. It failed anyway, with `KeyError: 'GROQ_API_KEY'`, thrown from
inside `src/llm_client.py`. The reason: `llm_client.py` constructs the Groq and OpenAI clients
at *module import time* (`_groq = Groq(api_key=os.environ["GROQ_API_KEY"])`), and
`check_metrics_formatter.py` transitively imports `cascade.py` → `classifier.py` →
`llm_client.py`, even though it never actually calls either client. Locally this was masked
completely — `.env` is always present, so `load_dotenv()` silently backfills the keys — which
is exactly why I only caught it by deliberately moving `.env` aside and re-running the check,
simulating what a clean CI checkout actually looks like. Rather than refactor client
construction to be lazy (a real fix, but a bigger change to an already-tested core module for
a CI-only problem), I scoped the fix to CI: the workflow sets placeholder env var values
(not real secrets) so the import succeeds, and documented in the workflow file exactly which
checks still need real secrets and aren't run automatically.

## Building and running the eval harness

**Real crash, twice, from the same root cause: I never handled a model returning no content.**
The very first live eval run crashed with `TypeError: 'NoneType' object is not subscriptable`
inside `call_openrouter`. `resp.choices[0].message.content` was `None` — the tier-4 model
(a reasoning-heavy Nemotron variant) apparently returned an empty content field on some query.
I fixed it (`content or ""`), re-ran, and hit the *same error class* one line up:
`resp.choices` itself was `None` this time, not just `.content`. That's a known OpenRouter
quirk — an overloaded free model can return HTTP 200 with an error payload embedded in the
body instead of a proper error status, and the SDK parses that into a response object with
`choices=None` rather than raising. I hardened both `call_groq` and `call_openrouter` against
an empty/missing `choices` list, not just empty content. Neither of these had shown up in any
of my manual single-query tests earlier — it took a 25-query sweep hitting a wider variety of
models and edge conditions to surface them. This is exactly why the eval harness earns its
keep beyond producing a nice number: it's a stress test.

**Real metrics bug, found by the eval harness reporting a nonsensical number.**
One eval query ("What is 17 * 23?") reported `compute_saved_pct: 100.0` while also being
marked as a *failed* run — i.e., "we saved all the compute" on a request that never produced
a usable answer. I traced it: both tiers it tried came back `malformed_response` (the model
ran and generated text, just not parseable JSON), and `cascade.py` was recording those steps
with `active_params_b=0` — the same value used for a genuine `unavailable` outage where the
model never even ran. That's wrong: a malformed response still cost real compute; only a
429/503 rejection is actually free. Fixed by recording the model's real `active_params_b` on
`malformed_response` steps too, same as any other real attempt.

**Real methodology gap, found by a systematic pattern, not a single failure.**
Summarization queries failed the judge almost every time in the first eval run — both the
cascade *and* the always-max-tier baseline, across nearly all 5 summarization queries. That
consistency across both paths ruled out "the small model is bad at summarizing" — it pointed
at the judge itself. I pulled one failing case and read the judge's actual reasoning: it
faulted a perfectly reasonable one-sentence summary for "missing the repetitive behavior"
that was, on inspection, still present in a different phrasing. The judge's rubric
("decide if the answer is correct") is built for a task with one right answer — QA, coding,
reasoning — and doesn't map onto summarization or translation, where there's no single
ground truth to be "correct" against, only "faithful." I added task-type-specific guidance to
the judge's prompt (don't penalize summaries for omitting secondary detail; judge
translations on meaning, not literal phrasing). Re-testing the same 5 summarization queries
afterward: pass rate went from roughly 1-in-5 to 2-in-5 — a real, measurable improvement, but
not a full fix. A small model judging open-ended, subjective output has a real ceiling that
prompt tweaking alone doesn't erase — which is itself the finding, not a bug I kept chasing.

**Real gap in my own test, not in the code it was testing.**
After the fixes above, `check_cascade.py` started failing on an assertion I'd written months
(well, tasks) earlier: it only accepted `"accepted"`, `"judged_fail"`, or `"unavailable"` as
valid final trace states, and a run had legitimately ended on `"malformed_response"` — every
tier up to the ceiling failed to parse. That's always been a valid way for a cascade to end;
the check's assertion was just incomplete from the start, and it took a live run actually
landing on that path to expose it.

**Real bug caught live in the UI, not by any automated check: translation queries phrased as
questions got answered instead of translated.**
I asked the running app to translate "Where is the nearest train station?" to Spanish. Both
tier 1 and tier 2 came back with a Spanish sentence about not having GPS access — the model
had tried to *answer* the question ("where is the nearest station") instead of translating
the sentence. Both tiers failing the same way, on two different models, ruled out "one model
is just bad at this" and pointed at something upstream shared by both: the prompt they were
each given. I traced it to `classify()` — its `optimized_prompt` field is supposed to rewrite
the query "to be clear and unambiguous," but for this query it rewrote `"Translate 'Where is
the nearest train station?' to Spanish"` into `"where is the nearest train station in spanish
translation"`. That rewrite deleted the word "translate" as an instruction and left something
that reads exactly like a real navigation question with "in spanish translation" tacked on at
the end — which is exactly what both downstream models tried to answer. The fix belongs at the
classifier, not the cascade or the answer prompt, because the corrupted prompt is what gets
handed to *every* tier — patching downstream would leave the actual corruption in place. Added
explicit guidance: for translation queries, preserve the literal "Translate 'X' to Y."
structure instead of rephrasing it into a different sentence. Verified against 3 translation
queries afterward — the instruction survives the rewrite intact every time now.

**Real operational limit hit mid-audit: OpenRouter's free tier has a hard *daily* request cap,
not just per-model rate limiting.**
While re-running the eval harness for a clean final report, most OpenRouter-routed coding and
translation queries started coming back `unavailable` even though they'd worked minutes
earlier. I tested one OpenRouter model call directly and got the real answer:

```
RateLimitError: 429 - Rate limit exceeded: free-models-per-day.
Add 10 credits to unlock 1000 free model requests per day.
X-RateLimit-Limit: 50, X-RateLimit-Remaining: 0, X-RateLimit-Reset: 2026-07-29T00:00:00Z
```

OpenRouter's free tier caps unpaid accounts at **50 free-model requests per day, total, across
every `:free` model** — not per-model, per-account. A day of heavy manual testing (debugging
sessions, live UI checks, two earlier eval attempts) burned through it. This is a real
constraint of building on free-tier infrastructure that the original design docs didn't call
out explicitly, and it's worth stating plainly for anyone reproducing this project: expect to
hit this within a single active day of testing unless you add the $10 one-time credit
(1000/day) or spread testing across days. The upside: this validated the `ModelUnavailable`
handling under a real, sustained outage rather than a simulated one — every affected query
degraded to escalate-or-fallback exactly as designed, no crash, at real production-account
scale. `eval/results.json` in this repo reflects the last run completed before the quota was
hit; a fully clean cross-provider sweep needs to wait for the daily reset.

**Real bug, found while looking for a clean screenshot: one specific model was failing JSON
parsing 100% of the time, on every query, for a reason none of my other testing had hit.**
Trying to find a query that shows a clean multi-tier escalation, `qwen/qwen3.6-27b` (Groq,
tier 2 for qa/reasoning) came back `malformed_response` on all 4 different queries I tried it
with. Four different queries failing identically, on one specific model, meant the model
itself — not the query content — was the common factor. I printed its raw output directly and
found it: this is a reasoning-tuned model that wraps its entire chain-of-thought in
`<think>...</think>` tags before the actual JSON answer, e.g.:

```
<think>
1. Analyze the Request: ...
5. Construct JSON: `{"answer": "9", "confidence": 10}`
...
</think>

{"answer": "9", "confidence": 10}
```

My JSON extraction (`_strip_fences`) only stripped markdown code fences — it had no handling
for a reasoning block wrapping the real answer. My first fix attempt (grab everything from the
first `{` to the last `}`) made it *worse*, not better: the model's reasoning trace quotes a
draft copy of the JSON mid-thought (step 5 above), so "first `{`" landed inside that draft, and
"last `}`" landed at the real final answer — the extracted span included all the reasoning
text in between as invalid JSON. Fixed by taking the *last* `{` to the *last* `}` instead of
first-to-last: the real, final answer is always the last complete JSON object emitted,
regardless of how many draft copies the model's own reasoning quotes earlier. This generalizes
past this one model — it handles any reasoning-wrapper convention (`<think>`, `<|thinking|>`,
whatever a future model uses) without special-casing any of them, since it doesn't look for
tags at all, just the last balanced-looking brace pair.

---
*(more entries added below as later steps and eval-run results surface real findings)*
