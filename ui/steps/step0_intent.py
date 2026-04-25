"""
Step 0 — Educational Notice + Planning Path

Robust branching version:
- Persists the selected path in step0_planning_pathway.
- Keeps USER_INTENT compatibility for Step 2/3.
"""

from __future__ import annotations

from typing import Dict
import streamlit as st

from ui.state.keys import CURRENT_STEP, USER_INTENT

STEP0_NOTICE_ACCEPTED = "step0_educational_notice_accepted"
STEP0_PATHWAY = "step0_planning_pathway"

PATHWAY_OPTIONS = [
    ("compare_both", "Compare both approaches"),
    ("savings_only", "Savings only"),
    ("savings_plus_investing", "Savings + investing"),
]

PATHWAY_GUIDANCE: Dict[str, str] = {
    "compare_both": (
        "Recommended for most users: start with the comparison view to understand trade-offs."
    ),
    "savings_only": (
        "Focus on cash-flow stability and long-term saving first. The investment engine is skipped in this path."
    ),
    "savings_plus_investing": (
        "Build a weekly savings plan, then explore how investing could affect longer-term outcomes under uncertainty."
    ),
}

PATHWAY_TO_USER_INTENT = {
    "compare_both": "not_sure_yet",
    "savings_only": "avoid_overspending",
    "savings_plus_investing": "save_more_each_week",
}

DEFAULT_PATHWAY = "compare_both"


def _resolve_pathway() -> str:
    value = str(st.session_state.get(STEP0_PATHWAY, DEFAULT_PATHWAY) or DEFAULT_PATHWAY)
    valid = {key for key, _ in PATHWAY_OPTIONS}
    if value not in valid:
        value = DEFAULT_PATHWAY
    st.session_state[STEP0_PATHWAY] = value
    return value


def _index_for_pathway(pathway: str) -> int:
    for idx, (key, _) in enumerate(PATHWAY_OPTIONS):
        if key == pathway:
            return idx
    return 0


def _render_pathway_selector() -> tuple[str, str, str]:
    st.markdown("#### What path do you want to explore?")

    current_pathway = _resolve_pathway()
    labels = [label for _, label in PATHWAY_OPTIONS]
    values = [key for key, _ in PATHWAY_OPTIONS]

    selected_label = st.radio(
        "Planning path",
        labels,
        index=_index_for_pathway(current_pathway),
        label_visibility="collapsed",
        key="step0_pathway_radio",
    )

    selected_pathway = values[labels.index(selected_label)]

    # CRITICAL: persist canonical branch every render.
    st.session_state[STEP0_PATHWAY] = selected_pathway
    st.session_state["selected_planning_pathway"] = selected_pathway  # harmless alias/debug
    selected_intent = PATHWAY_TO_USER_INTENT.get(selected_pathway, "not_sure_yet")
    st.session_state[USER_INTENT] = selected_intent

    st.info(PATHWAY_GUIDANCE.get(selected_pathway, PATHWAY_GUIDANCE[DEFAULT_PATHWAY]))

    return selected_pathway, selected_label, selected_intent


def _render_educational_notice() -> bool:
    st.markdown("## Educational assumptions")
    st.markdown("### Educational use notice")

    st.warning(
        "LifeBudget Micro is an educational planning and scenario-exploration tool. "
        "It is not financial advice, investment advice, or a guarantee of future outcomes."
    )

    st.markdown(
        "Projections are scenario estimates based on assumptions and historical data, "
        "not predictions. Investing involves risk, including possible loss of capital."
    )

    accepted = bool(
        st.checkbox(
            "I understand this is an educational simulation and not financial advice.",
            value=bool(st.session_state.get(STEP0_NOTICE_ACCEPTED, False)),
            key=STEP0_NOTICE_ACCEPTED,
        )
    )

    if not accepted:
        st.info("Please acknowledge the educational notice to continue.")
        return False

    return True


def render_step_0() -> dict:
    st.markdown("## Life goals planning")
    st.caption("Choose the planning path you want to explore, then confirm the educational notice before continuing.")

    selected_pathway, selected_label, selected_intent = _render_pathway_selector()

    with st.expander("How should I interpret the investment results later?", expanded=False):
        st.markdown(
            """
The investment engine is a **scenario comparison layer**.

It can help compare:
- saving only,
- saving + investing,
- sensitivity to uncertainty.

It is **not**:
- a guaranteed return,
- personalised investment advice,
- a market prediction.
"""
        )

    st.markdown("---")

    notice_ok = _render_educational_notice()

    st.markdown("---")

    if st.button("Start planning", use_container_width=True, key="step0_continue_to_step1", disabled=not notice_ok):
        st.session_state[STEP0_PATHWAY] = selected_pathway
        st.session_state[USER_INTENT] = selected_intent
        st.session_state[CURRENT_STEP] = 1
        st.session_state["current_step"] = 1
        st.rerun()

    return {
        "notice_accepted": bool(notice_ok),
        "pathway": selected_pathway if notice_ok else "",
        "pathway_label": selected_label if notice_ok else "",
        "intent": selected_intent,
        "guidance_text": PATHWAY_GUIDANCE.get(selected_pathway, ""),
    }
