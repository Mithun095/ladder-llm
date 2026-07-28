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

---
*(more entries added below as later steps and edge-case testing surface real issues)*
