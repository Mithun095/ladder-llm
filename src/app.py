import streamlit as st

from src.cascade import run_cascade
from src.formatter import format_answer
from src.metrics import (
    compute_saved_pct,
    estimate_dollar_saved,
    total_active_params_burned,
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
    with st.spinner("Routing through the cascade..."):
        result = run_cascade(query)
    st.session_state["result"] = result

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
    accepted = bool(result.trace) and result.trace[-1].status == "accepted"
    if not accepted:
        last = result.trace[-1] if result.trace else None
        if last is not None and last.status == "judged_fail":
            st.error(
                f"**No tier produced an answer the judge accepted** — the cascade reached its "
                f"ceiling (tier {last.tier}) and stopped. The answer below is the best attempt, "
                f"shown so you can see what was rejected, but it did **not** pass review.\n\n"
                f"Judge's reason: *{last.judge_reason}*"
            )
        elif result.trace and all(s.status == "unavailable" for s in result.trace):
            # Almost always the free-tier daily cap rather than a real outage. Saying so beats
            # making the reader cross-reference the trace against the README's limitations.
            providers = "OpenRouter" if all("/" in s.model_id for s in result.trace) else "the provider"
            st.error(
                f"**No model was reachable for this query.** Every tier in range returned "
                f"429/503, so nothing ran and no compute was spent.\n\n"
                f"On a free tier this is usually the quota, not an outage — {providers} caps "
                f"unpaid accounts at **50 requests per day, account-wide**, and Groq at 30 per "
                f"minute. Coding queries route to OpenRouter at tiers 1-3, so they're the first "
                f"to fail once the daily cap is gone. This is the cascade degrading as designed "
                f"rather than crashing; see Limitations in the README."
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
        st.subheader("Answer" if accepted else "Best attempt (rejected)")
        st.markdown(format_answer(result.answer, result.type))

    with trace_col:
        st.subheader("Routing trace")
        for step in result.trace:
            icon = {
                "accepted": "✅",
                "escalated": "⬆️",
                "judged_fail": "❌",
                "unavailable": "⚠️",
                "malformed_response": "⚠️",
            }[step.status]
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

    # A run where every tier was unavailable burns zero compute and would score "100% saved".
    # True, and completely meaningless — nothing was delivered. Savings are only a number when
    # something came out; otherwise this metric flatters a total failure. (Same bug as the one
    # fixed in the eval harness — see BUILD-LOG.md #11.)
    produced_answer = result.answer != "No model produced a usable answer."
    cost_cols = st.columns(4)
    if produced_answer:
        cost_cols[0].metric(
            "Compute saved vs. max tier",
            f"{saved:.0f}%",
            delta=None if saved >= 0 else "cascade cost more than the baseline",
            delta_color="inverse",
        )
        cost_cols[2].metric(
            "Illustrative $ saved",
            f"${estimate_dollar_saved(result.trace, result.type):.5f}",
            help="Approximate rate, not a real bill — these are free-tier models. See metrics.py.",
        )
    else:
        cost_cols[0].metric("Compute saved vs. max tier", "n/a",
                            help="No answer was produced, so there is nothing to have saved on.")
        cost_cols[2].metric("Illustrative $ saved", "n/a")
    cost_cols[1].metric("Active params burned", f"{burned:.1f}B", help=f"Always-tier-{MAX_TIER} baseline: {baseline}B")
    cost_cols[3].metric("End-to-end latency", "0.0s (cached)" if result.cached else f"{result.elapsed_ms / 1000:.1f}s")

    if result.optimized_prompt:
        with st.expander("What the classifier actually sent downstream"):
            st.caption("The raw query is rewritten for clarity before any answering model sees it.")
            st.code(result.optimized_prompt, language=None)
