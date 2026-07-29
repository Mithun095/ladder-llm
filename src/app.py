import streamlit as st

from src.cascade import run_cascade
from src.formatter import format_answer
from src.metrics import (
    compute_saved_pct,
    dollar_saved_pct,
    estimate_dollar_saved,
    total_active_params_burned,
    trace_cost_usd,
)
from src.registry import MAX_TIER, get_model

st.set_page_config(page_title="LadderLLM", layout="wide")
st.title("LadderLLM — Adaptive Multi-Tier LLM Cascade Router")
st.caption(
    "Every query starts at the cheapest model that could plausibly answer it and only "
    "escalates when a judge model says the cheap answer actually failed."
)

query = st.text_input("Ask a question", placeholder="e.g. A farmer has 17 sheep. All but 9 die. How many are left?")
submit = st.button("Submit", type="primary")

if submit and query:
    try:
        with st.spinner("Routing through the cascade..."):
            result = run_cascade(query)
        st.session_state["result"] = result
    except Exception as e:
        # call_json re-raises any provider status code it doesn't recognize as retryable
        # (429/503) so a real registry bug (bad model ID) crashes loudly instead of silently
        # returning nothing. But other 4xx codes are reachable too — e.g. 413 on an oversized
        # summarization/translation paste, since those types send the raw query untouched — and
        # those aren't registry bugs, just an operational error the user should see, not an
        # unhandled exception that takes down the whole page.
        st.error(f"**The cascade failed to complete.** {type(e).__name__}: {e}")

if "result" in st.session_state:
    result = st.session_state["result"]
    saved = compute_saved_pct(result.trace, result.type)
    burned = total_active_params_burned(result.trace)
    baseline = get_model(MAX_TIER, result.type).active_params_b

    if result.cached:
        st.success("Cache hit — this exact query was answered earlier, so no models were called at all.")

    # The cascade returns the best answer it found even when no tier was ever accepted. Showing
    # that silently is how a judge-rejected answer ends up looking like a verified one — say
    # plainly that every tier failed, and why the last one did.
    if result.accepted and not result.verified:
        # Shown rather than silently passing it off as reviewed. The cascade returns its best
        # answer when the judge is unreachable, which is the right call for a reader — but the
        # reader should know which of the two they are looking at.
        st.warning(
            "**This answer was not reviewed.** The judge model was rate-limited, so the cascade "
            "returned the answer unchecked rather than withholding it. Treat it as unverified — "
            "and note it is *not* counted as a pass in the benchmark, because an answer nobody "
            "checked is not a verified one."
        )

    if not result.accepted:
        # Find the last judge rejection anywhere in the trace, not just at the final step: a
        # judged_fail tier can be followed by an unavailable one (the next tier's provider is
        # rate-limited) if nothing escalates further, and the final step's status used to be
        # the only one checked — dropping the judge's reason and reporting "no model was
        # reachable", which is false when a model *was* reached and rejected.
        last_rejection = next(
            (s for s in reversed(result.trace) if s.status == "judged_fail"), None
        )
        if last_rejection is not None:
            # "reached its ceiling" uses the trace's actual last tier, not last_rejection.tier —
            # those differ exactly when a later tier was attempted and came back unavailable.
            st.error(
                f"**No tier produced an answer the judge accepted** — the cascade reached its "
                f"ceiling (tier {result.trace[-1].tier}) and stopped. The answer below is the "
                f"best attempt, shown so you can see what was rejected, but it did **not** pass "
                f"review.\n\nJudge's reason (tier {last_rejection.tier}): "
                f"*{last_rejection.judge_reason}*"
            )
        elif result.trace and all(s.status == "unavailable" for s in result.trace):
            # Almost always the free-tier daily cap rather than a real outage. Saying so beats
            # making the reader cross-reference the trace against the README's limitations.
            #
            # Provider comes from the registry, not from guessing at the model ID. The previous
            # version tested `"/" in model_id`, which was right until Groq's own catalogue turned
            # out to use slashes too — it would have labelled `openai/gpt-oss-120b`, a Groq model,
            # as OpenRouter and blamed the wrong quota.
            providers = {get_model(s.tier, result.type).provider for s in result.trace}
            who = ("OpenRouter" if providers == {"openrouter"}
                   else "Groq" if providers == {"groq"} else "both providers")
            limits = {
                "OpenRouter": "**50 requests per day, account-wide** on unpaid accounts, which "
                              "resets at 00:00 UTC",
                "Groq": "**30 requests per minute**, which clears on its own within the minute",
            }.get(who, "a free-tier quota")
            st.error(
                f"**No model was reachable for this query.** Every tier in range returned "
                f"429/503, so nothing ran and no compute was spent.\n\n"
                f"On a free tier this is usually the quota, not an outage — {who} caps you at "
                f"{limits}. This is the cascade degrading as designed rather than crashing; "
                f"see Limitations in the README."
            )
        else:
            st.error(
                "**No tier produced a usable answer.** Every model in range was either "
                "unavailable or returned output that couldn't be parsed — see the trace."
            )

    # Routing decision up top: the classification is what drives everything below it, so it
    # shouldn't be buried in the trace.
    st.markdown(
        f"**Routed as** `{result.type}` · **difficulty** `{result.difficulty}` · "
        f"**resolved at tier {result.tier_used}** of {MAX_TIER} · "
        f"**{len(result.trace)} model call{'s' if len(result.trace) != 1 else ''}**"
    )

    answer_col, trace_col = st.columns([3, 2])

    with answer_col:
        st.subheader("Answer" if result.accepted else "Best attempt (rejected)")
        st.markdown(format_answer(result.answer, result.type))

    with trace_col:
        st.subheader("Routing trace")
        for step in result.trace:
            # .get with a fallback, not [] — this dict is indexed by a status Literal that has
            # gained a member before, and a KeyError here fires inside the render path with the
            # answer already on screen. An unknown status should show as unknown, not crash.
            icon = {
                "accepted": "✅",
                "accepted_unverified": "❓",
                "escalated": "⬆️",
                "judged_fail": "❌",
                "unavailable": "⚠️",
                "malformed_response": "⚠️",
            }.get(step.status, "•")
            line = f"{icon} **Tier {step.tier}** — `{step.model_id}` — {step.status}"
            if step.confidence is not None:
                line += f" · confidence {step.confidence}/10"
            if step.elapsed_ms:
                line += f" · {step.elapsed_ms / 1000:.1f}s"
            st.write(line)
            if step.judge_reason:
                st.caption(f"Judge: {step.judge_reason}")

    st.divider()
    st.subheader("Cost of this answer")

    # "Saved" only means anything if the run delivered a usable answer. Two ways it can fail to:
    # every tier unavailable (0 params burned → a flattering "100% saved" for doing nothing), or
    # every tier judge-rejected (params burned, nothing delivered → "36% saved" on a total
    # failure). Both are gated on result.accepted. See BUILD-LOG.md #11 and #19.
    cost_cols = st.columns(4)
    if result.accepted:
        cost_cols[0].metric(
            "Compute saved vs. max tier",
            f"{saved:.0f}%",
            delta=None if saved >= 0 else "cascade cost more than the baseline",
            delta_color="inverse",
            help=f"Active params burned: {burned:.1f}B vs. an always-tier-{MAX_TIER} baseline "
                 f"of {baseline}B. Measures compute, not price.",
        )
        cost_saved = dollar_saved_pct(result.trace, result.type)
        cost_cols[1].metric(
            "Cost saved vs. max tier",
            f"{cost_saved:.0f}%",
            delta=None if cost_saved >= 0 else "cascade cost more than the baseline",
            delta_color="inverse",
            help="Priced at published $/1M-token rates for the same open-weight models. This is "
                 "a separate number from compute saved, and the two can disagree — a sparse MoE "
                 "can burn fewer active params than a smaller dense model yet cost more per "
                 "token. See registry.py.",
        )
        cost_cols[2].metric(
            "$ saved on this query",
            f"${estimate_dollar_saved(result.trace, result.type):.5f}",
            help=f"This run priced at ${trace_cost_usd(result.trace, result.type):.5f}. Published "
                 f"rates, not a real bill — every model called here is a free endpoint.",
        )
    else:
        for col, label in ((0, "Compute saved vs. max tier"), (1, "Cost saved vs. max tier"),
                           (2, "$ saved on this query")):
            cost_cols[col].metric(
                label, "n/a",
                help=f"No accepted answer came out of this run, so there is nothing to have "
                     f"saved on — the {burned:.1f}B active params it burned bought nothing.",
            )
    cost_cols[3].metric("End-to-end latency", "0.0s (cached)" if result.cached else f"{result.elapsed_ms / 1000:.1f}s")

    if result.optimized_prompt:
        with st.expander("What the classifier actually sent downstream"):
            st.caption("The raw query is rewritten for clarity before any answering model sees it.")
            st.code(result.optimized_prompt, language=None)
