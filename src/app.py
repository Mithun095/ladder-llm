import streamlit as st

from src.cascade import run_cascade
from src.formatter import format_answer
from src.metrics import compute_saved_pct, total_active_params_burned

st.set_page_config(page_title="LadderLLM", layout="wide")
st.title("LadderLLM — Adaptive Multi-Tier LLM Cascade Router")

query = st.text_input("Ask a question")
submit = st.button("Submit")

if submit and query:
    with st.spinner("Routing through the cascade..."):
        result = run_cascade(query)
    st.session_state["result"] = result

if "result" in st.session_state:
    result = st.session_state["result"]
    answer_col, trace_col = st.columns(2)

    with answer_col:
        st.subheader("Answer")
        st.markdown(format_answer(result.answer, result.type))

        saved = compute_saved_pct(result.trace, result.type)
        burned = total_active_params_burned(result.trace)
        st.metric("Compute saved vs. max tier", f"{saved:.0f}%")
        st.caption(f"{burned:.0f}B active params burned this run (tier {result.tier_used} used)")

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
            line = f"{icon} Tier {step.tier} — `{step.model_id}` — {step.status}"
            if step.confidence is not None:
                line += f" (confidence {step.confidence})"
            st.write(line)
            if step.judge_reason:
                st.caption(f"Judge: {step.judge_reason}")
