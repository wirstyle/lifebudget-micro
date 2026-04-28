"""
Step 0 — Landing Page + Educational Notice

Final-polish version:
- Presents the app as three user-facing modules instead of three narrow branches.
- Keeps the old step0_planning_pathway and USER_INTENT values for downstream compatibility.
- Step 0 no longer owns the sidebar; app.py should render the shared global sidebar once.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import html

import streamlit as st

from ui.state.keys import CURRENT_STEP, USER_INTENT

STEP0_NOTICE_ACCEPTED = "step0_educational_notice_accepted"
# Dedicated widget key. Do not use the durable acceptance flag as the checkbox
# key, otherwise the checkbox widget can accidentally become the source of truth.
STEP0_NOTICE_CHECKBOX = "step0_educational_notice_checkbox_v1"
STEP0_NOTICE_ACCEPTED_ONCE = "educational_notice_accepted_once_v1"
STEP0_NOTICE_ALIASES = (
    STEP0_NOTICE_ACCEPTED,
    STEP0_NOTICE_ACCEPTED_ONCE,
    "step0_educational_notice_acknowledged",
    "step0_notice_accepted",
)
STEP0_PATHWAY = "step0_planning_pathway"
STEP0_SELECTED_MODULE = "step0_selected_module"

# Keep these original pathway values because later steps still use them as
# compatibility signals. Step 0 now exposes broader modules to the user, then
# maps each module back to the closest existing internal pathway/intention.
PATHWAY_OPTIONS = [
    ("compare_both", "Compare both approaches"),
    ("savings_only", "Savings only"),
    ("savings_plus_investing", "Savings + investing"),
]

PATHWAY_GUIDANCE: Dict[str, str] = {
    "compare_both": (
        "Recommended full-flow mode: compare savings capacity, investment assumptions, "
        "and long-term scenario outcomes."
    ),
    "savings_only": (
        "Focus on cash-flow stability and long-term saving first. The investment engine can be skipped."
    ),
    "savings_plus_investing": (
        "Build from savings capacity into investment testing and long-term scenario exploration."
    ),
}

PATHWAY_TO_USER_INTENT = {
    "compare_both": "not_sure_yet",
    "savings_only": "avoid_overspending",
    "savings_plus_investing": "save_more_each_week",
}

DEFAULT_PATHWAY = "compare_both"

# module_id, title, steps, description, button_label, target_step, pathway, user_intent
MODULE_OPTIONS: List[Tuple[str, str, str, str, str, int, str, str]] = [
    (
        "personal_finance",
        "Personal Finance Planner",
        "Steps 1-3",
        "Build income, expenses, savings capacity, and short-term goals before moving into investment or projection decisions.",
        "Select finance planner",
        1,
        "compare_both",
        "not_sure_yet",
    ),
    (
        "investment_lab",
        "Investment Strategy Lab",
        "Steps 4-5",
        "Prepare a cached investment universe, run the strategy engine, and review historical risk/return diagnostics and suggestions.",
        "Select investment lab",
        4,
        "savings_plus_investing",
        "save_more_each_week",
    ),
    (
        "scenario_explorer",
        "Long-Term Scenario Explorer",
        "Steps 6-7",
        "Explore long-term savings outcomes first, then compare them with either a labelled educational investment proxy or tested strategy returns from the Investment Lab.",
        "Select scenario explorer",
        6,
        "compare_both",
        "not_sure_yet",
    ),
]

DEFAULT_MODULE = "personal_finance"



def _educational_notice_accepted() -> bool:
    """Read the durable educational-notice state across legacy aliases."""
    return any(bool(st.session_state.get(key, False)) for key in STEP0_NOTICE_ALIASES)


def _mark_educational_notice_accepted() -> None:
    """Persist notice acceptance independently of the checkbox widget state."""
    for key in STEP0_NOTICE_ALIASES:
        st.session_state[key] = True
    st.session_state[STEP0_NOTICE_CHECKBOX] = True


def _handle_notice_checkbox_change() -> None:
    """Persist acceptance before the next full rerun.

    app.py renders the global sidebar before Step 0. Using a checkbox callback
    means the durable notice flag is already available when the sidebar renders
    on the rerun triggered by checking the box.
    """
    if bool(st.session_state.get(STEP0_NOTICE_CHECKBOX, False)):
        _mark_educational_notice_accepted()


def _module_lookup() -> Dict[str, dict]:
    return {
        module_id: {
            "id": module_id,
            "title": title,
            "steps": steps,
            "description": description,
            "button_label": button_label,
            "target_step": int(target_step),
            "pathway": pathway,
            "intent": intent,
        }
        for module_id, title, steps, description, button_label, target_step, pathway, intent in MODULE_OPTIONS
    }


def _resolve_module_id() -> str:
    modules = _module_lookup()
    value = str(st.session_state.get(STEP0_SELECTED_MODULE, DEFAULT_MODULE) or DEFAULT_MODULE)
    if value not in modules:
        value = DEFAULT_MODULE
    st.session_state[STEP0_SELECTED_MODULE] = value
    return value


def _resolve_pathway() -> str:
    value = str(st.session_state.get(STEP0_PATHWAY, DEFAULT_PATHWAY) or DEFAULT_PATHWAY)
    valid = {key for key, _ in PATHWAY_OPTIONS}
    if value not in valid:
        value = DEFAULT_PATHWAY
    st.session_state[STEP0_PATHWAY] = value
    return value


def _pathway_label(pathway: str) -> str:
    for key, label in PATHWAY_OPTIONS:
        if key == pathway:
            return label
    return "Compare both approaches"


def _persist_module_choice(module_id: str) -> dict:
    modules = _module_lookup()
    module = modules.get(module_id, modules[DEFAULT_MODULE])

    # CRITICAL: persist both the user-facing module and the internal legacy branch.
    # This avoids breaking Steps 2/3/6 while letting Step 0 use clearer language.
    st.session_state[STEP0_SELECTED_MODULE] = str(module["id"])
    st.session_state[STEP0_PATHWAY] = str(module["pathway"])
    st.session_state["selected_planning_pathway"] = str(module["pathway"])  # harmless alias/debug
    st.session_state["selected_start_module"] = str(module["id"])
    st.session_state[USER_INTENT] = str(module["intent"])
    return module



def _render_module_card(module: dict, *, selected: bool) -> None:
    border_color = "#2E7D32" if selected else "rgba(49, 51, 63, 0.18)"
    badge_text = "Selected" if selected else "Module"
    badge_bg = "rgba(46, 125, 50, 0.10)" if selected else "rgba(49, 51, 63, 0.06)"

    title = html.escape(str(module.get("title", "Module")))
    steps = html.escape(str(module.get("steps", "")))
    description = html.escape(str(module.get("description", "")))
    badge = html.escape(badge_text)

    st.markdown(
        f"""
        <div style="
            border: 1.5px solid {border_color};
            border-radius: 18px;
            padding: 1.2rem 1.15rem;
            min-height: 250px;
            background: rgba(255, 255, 255, 0.72);
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
            margin-bottom: 0.75rem;
        ">
            <div style="display:flex; justify-content:space-between; align-items:center; gap:0.75rem;">
                <p style="font-size: 0.82rem; color: #6b7280; margin: 0;">{steps}</p>
                <span style="font-size: 0.72rem; padding: 0.18rem 0.48rem; border-radius: 999px; background: {badge_bg}; color: #374151;">{badge}</span>
            </div>
            <h3 style="margin-top: 0.7rem; margin-bottom: 0.8rem; line-height:1.2;">{title}</h3>
            <p style="color: #4b5563; line-height: 1.45; margin-bottom: 0;">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(str(module.get("button_label", "Select")), key=f"step0_select_{module['id']}", use_container_width=True):
        _persist_module_choice(str(module["id"]))
        st.rerun()


def _render_module_selector() -> dict:
    st.markdown("### Choose where to start")
    st.caption(
        "The app is organised into three modules. You can start with personal finance, jump into the investment lab, "
        "or explore long-term scenarios using savings-only, educational proxy, or tested engine results."
    )

    modules = _module_lookup()
    selected_id = _resolve_module_id()
    selected_module = modules.get(selected_id, modules[DEFAULT_MODULE])

    cols = st.columns(3)
    for idx, raw in enumerate(MODULE_OPTIONS):
        module_id = raw[0]
        module = modules[module_id]
        with cols[idx]:
            _render_module_card(module, selected=(module_id == selected_id))

    selected_module = _persist_module_choice(str(st.session_state.get(STEP0_SELECTED_MODULE, selected_module["id"])))

    st.info(
        "**Recommended full flow:** Personal Finance Planner -> Investment Strategy Lab -> "
        "Long-Term Scenario Explorer -> Insights Summary."
    )
    st.success(f"Selected start: **{selected_module['title']}** ({selected_module['steps']}).")

    return selected_module


def _render_investment_interpretation_note() -> None:
    with st.expander("How should I interpret investment and projection results later?", expanded=False):
        st.markdown(
            """
LifeBudget Micro uses three evidence levels:

- **Savings-only scenario:** available immediately from your saving capacity.
- **Educational investment proxy:** available before the engine has run; useful for exploration, but not a tested result.
- **Tested strategy projection:** available after the Investment Strategy Lab runs the Step 5 engine.

Investment/projection outputs are **not**:
- a guaranteed return,
- personalised investment advice,
- a market prediction.
"""
        )


def _render_educational_notice() -> bool:
    st.markdown("## Educational assumptions")

    if _educational_notice_accepted():
        _mark_educational_notice_accepted()
        st.success("Educational notice already accepted for this session.")
        with st.expander("Review educational notice", expanded=False):
            st.markdown(
                "LifeBudget Micro is an educational planning and scenario-exploration tool. "
                "It is not financial advice, investment advice, or a guarantee of future outcomes. "
                "Projections are scenario estimates based on assumptions and historical data, not predictions. "
                "Investing involves risk, including possible loss of capital. Past performance refers to the past "
                "and is not a reliable indicator of future results."
            )
        return True

    st.markdown("### Educational use notice")
    st.warning(
        "LifeBudget Micro is an educational planning and scenario-exploration tool. "
        "It is not financial advice, investment advice, or a guarantee of future outcomes."
    )

    st.markdown(
        "Projections are scenario estimates based on assumptions and historical data, "
        "not predictions. Investing involves risk, including possible loss of capital. "
        "Past performance refers to the past and is not a reliable indicator of future results."
    )

    accepted = bool(
        st.checkbox(
            "I understand this is an educational simulation and not financial advice.",
            value=bool(st.session_state.get(STEP0_NOTICE_CHECKBOX, False)),
            key=STEP0_NOTICE_CHECKBOX,
            on_change=_handle_notice_checkbox_change,
        )
    )

    if accepted:
        _mark_educational_notice_accepted()
        return True

    st.info("Please acknowledge the educational notice to continue.")
    return False

def render_step_0() -> dict:
    # Ensure legacy state exists even before the user presses a module button.
    _resolve_pathway()
    selected_module = _persist_module_choice(_resolve_module_id())

    st.markdown("# LifeBudget Micro")
    st.caption(
        "Plan your budget, test an investment strategy, and explore long-term scenarios. "
        "Choose a starting module to begin."
    )

    selected_module = _render_module_selector()

    st.markdown("---")
    notice_ok = _render_educational_notice()
    st.markdown("---")

    start_label = f"Start: {selected_module['title']}"
    if st.button(start_label, use_container_width=True, key="step0_continue_to_selected_module", disabled=not notice_ok):
        _mark_educational_notice_accepted()
        selected_module = _persist_module_choice(str(selected_module["id"]))
        target_step = int(selected_module.get("target_step", 1))
        st.session_state[CURRENT_STEP] = target_step
        st.session_state["current_step"] = target_step
        st.rerun()

    selected_pathway = str(selected_module.get("pathway", DEFAULT_PATHWAY))
    selected_intent = str(selected_module.get("intent", PATHWAY_TO_USER_INTENT.get(selected_pathway, "not_sure_yet")))

    return {
        "notice_accepted": bool(notice_ok),
        "module": str(selected_module.get("id", DEFAULT_MODULE)) if notice_ok else "",
        "module_label": str(selected_module.get("title", "")) if notice_ok else "",
        "target_step": int(selected_module.get("target_step", 1)) if notice_ok else 0,
        "pathway": selected_pathway if notice_ok else "",
        "pathway_label": _pathway_label(selected_pathway) if notice_ok else "",
        "intent": selected_intent,
        "guidance_text": PATHWAY_GUIDANCE.get(selected_pathway, ""),
    }
