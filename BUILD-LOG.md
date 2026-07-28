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

---
*(more entries added below as later steps and edge-case testing surface real issues)*
