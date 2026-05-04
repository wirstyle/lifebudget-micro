"""Home / module-selection screen for LifeBudget Micro.

This screen is the user's entry point into the prototype. It presents the app as
three user-facing modules:

- Personal Finance Planner;
- Investment Strategy Lab;
- Long-Term Scenario Explorer.

It also renders the educational notice gate. Module cards stay disabled until
the notice is accepted, because the app produces educational backtests,
proxies, and scenario comparisons rather than financial advice.

Implementation note
-------------------
Earlier versions of the project used narrower planning pathways and intent
values. The current Home screen keeps ``step0_planning_pathway`` and
``USER_INTENT`` updated for downstream compatibility, while showing clearer
module language to the user.

Step 0 does not render its own sidebar. ``app.py`` renders the shared global
sidebar once after the active screen.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

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
STEP0_PENDING_MODULE_OPEN = "step0_pending_module_open_v1"

# Legacy internal pathway values.
#
# Earlier versions of LifeBudget Micro used narrower planning pathways such as
# "savings only" and "savings + investing". The final Home screen now presents
# broader user-facing modules instead:
#
# - Personal Finance Planner
# - Investment Strategy Lab
# - Long-Term Scenario Explorer
#
# These pathway values are still persisted because later screens and older
# compatibility logic can read them to decide which downstream flow is active.
# Step 0 therefore maps each visible module card back to the closest internal
# pathway and USER_INTENT value.
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


# User-facing Home cards.
#
# Each module has a visible card title/description and a target step. The last
# two fields keep compatibility with the older internal pathway/user-intent
# model. In the final UI, the user sees three broad modules rather than the
# original narrow planning branches.
#
# Tuple shape:
# module_id, title, steps_label, description, button_label, target_step,
# legacy_pathway, legacy_user_intent
MODULE_OPTIONS: List[Tuple[str, str, str, str, str, int, str, str]] = [
    (
        "personal_finance",
        "Personal Finance Planner",
        "Steps 1-3",
        "Build your budget baseline, savings capacity, and short-term feasibility before moving into scenarios.",
        "Open finance planner",
        1,
        "compare_both",
        "not_sure_yet",
    ),
    (
        "investment_lab",
        "Investment Strategy Lab",
        "Steps 4-5",
        "Prepare an investment universe, run the engine, and review risk/return diagnostics.",
        "Open investment lab",
        4,
        "savings_plus_investing",
        "save_more_each_week",
    ),
    (
        "scenario_explorer",
        "Long-Term Scenario Explorer",
        "Steps 6-7",
        "Explore savings-only, educational proxy, and tested-strategy long-term scenarios.",
        "Open scenario explorer",
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

    The checkbox has its own widget key, while the accepted notice is stored in
    durable alias keys. Using a callback keeps the acceptance state consistent
    across the main page and global sidebar after the checkbox interaction.
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


def _navigate_to_module(module: dict) -> None:
    """Open the selected user-facing module.

    This keeps Step 0 as a hub: the card buttons are the real entry points,
    while the educational notice remains the required gate before navigation.
    """
    target_step = int(module.get("target_step", 1))
    st.session_state[CURRENT_STEP] = target_step
    st.session_state["current_step"] = target_step
    st.rerun()


def _module_button_label(module: dict) -> str:
    """Return a native Streamlit button label for a clickable module card.

    Native Streamlit buttons do not support true card layouts or custom
    per-card padding. Keep the whole tile clickable, but use markdown spacing
    and a short divider so each module title has more presence and is clearly
    separated from the description.
    """
    title = str(module.get("title", "Module")).strip()
    description = str(module.get("description", "")).strip()
    divider = "────────────"
    return f"\n**{title}**\n\n{divider}\n\n{description}"


def _render_module_card(module: dict, *, can_open: bool) -> None:
    """Render one clickable module card using a native Streamlit button.

    Streamlit does not provide a true clickable-card component without custom
    HTML/JS. The clean native compromise is to make the entire module tile a
    full-width button and remove the separate Open button underneath.
    """
    button_label = _module_button_label(module)
    module_title = str(module.get("title", "module"))

    if st.button(
        button_label,
        key=f"step0_open_module_card_{module['id']}",
        use_container_width=True,
        disabled=not bool(can_open),
        help=(
            f"Open {module_title}."
            if can_open
            else "Accept the educational notice above to enable module navigation."
        ),
    ):
        selected_module = _persist_module_choice(str(module["id"]))
        _mark_educational_notice_accepted()
        _navigate_to_module(selected_module)

def _render_module_selector(*, can_open: bool) -> dict:
    st.markdown("### Choose a module to open")
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
            _render_module_card(module, can_open=can_open)

    # Persist the currently selected/default module so the sidebar can preview the
    # relevant branch even before a module button is clicked.
    selected_module = _persist_module_choice(str(st.session_state.get(STEP0_SELECTED_MODULE, selected_module["id"])))

    st.info(
        "**Recommended full flow:** Finance Planner → Investment Lab → "
        "Scenario Explorer → Insights Summary."
    )

    return selected_module


# Retained as optional explanatory copy for future Home-screen variants.
# The final submitted layout keeps the landing page shorter, so this block is
# not called by default. The same concepts are covered later in the Strategy
# Engine, Long-Term Scenario, and User Guide.
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


def _render_app_overview() -> None:
    """Explain the product scope before the user chooses a module.

    Keep this block focused on what the product does. Educational caveats and
    real-world limitations live in the notice below so the landing page does
    not repeat the same disclaimer three times.
    """
    st.info(
        "Estimate your personal cash flow, test a savings target, build an educational "
        "investment strategy, and compare long-term savings-only vs savings-plus-investing scenarios."
    )


def _render_important_limitations() -> None:
    """Render compact real-world limitations inside the educational notice area."""
    with st.expander("Important limitations", expanded=False):
        st.markdown(
            """
Real-world decisions would also need to consider tax, platform fees, fund charges, bid–ask spreads, inflation, currency effects, emergency savings, debt, pension/ISA rules, liquidity needs, asset availability, and personal risk tolerance.

The prototype does not check whether a strategy is suitable for a specific person. Backtests are useful for understanding behaviour over a historical period, but future markets can behave very differently.
"""
        )

def _render_educational_notice() -> bool:
    """Render the lightweight Step 0 gate before the module cards.

    The module buttons are intentionally disabled until this notice is accepted.
    This keeps the relationship between the notice and the entry buttons visible
    on the same screen, without needing a separate Start button.
    """
    st.markdown("### Educational assumptions")

    notice_text = (
        "LifeBudget Micro is an educational planning and scenario-comparison prototype. "
        "Investment outputs are backtests or educational proxies: they show how a selected "
        "strategy would have behaved over the available past data, not what it will do in the future. "
        "It is not financial advice, investment advice, or a guarantee of future outcomes."
    )

    if _educational_notice_accepted():
        _mark_educational_notice_accepted()
        st.caption("✅ Educational notice accepted.")
        with st.expander("Review educational assumptions", expanded=False):
            st.markdown(notice_text)
            st.markdown(
                "Real-world decisions would also need to consider tax, platform fees, fund charges, "
                "bid–ask spreads, inflation, currency effects, emergency savings, debt, pension/ISA rules, "
                "liquidity needs, asset availability, and personal risk tolerance."
            )
        return True

    st.warning(notice_text)
    _render_important_limitations()

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

    st.caption("Accept the educational notice above to enable the module buttons.")
    return False

def render_step_0() -> dict:
    # Ensure legacy state exists even before the user opens a module.
    _resolve_pathway()
    selected_module = _persist_module_choice(_resolve_module_id())
    st.session_state.pop(STEP0_PENDING_MODULE_OPEN, None)

    st.markdown("# LifeBudget Micro")

    _render_app_overview()

    # Place the notice before the cards so the disabled/enabled state is visually
    # connected to the three module entry buttons.
    notice_ok = _render_educational_notice()

    st.markdown("---")
    selected_module = _render_module_selector(can_open=notice_ok)

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
