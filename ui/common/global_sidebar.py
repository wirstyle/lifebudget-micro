"""Global sidebar for LifeBudget Micro.

Streamlit-native version: no custom CSS/HTML.

The sidebar is a compact visual navigator plus a lightweight state inspector.
It uses only native Streamlit components, avoids heavy computation, and keeps
expensive checks/actions on the main screen.

Design principles
-----------------
- The main screen owns the workflow and heavy actions.
- The sidebar provides context, navigation and explanation.
- Diagnostics are read-only unless they simply toggle a main-screen panel.
- The sidebar should never refresh market data, run the Strategy Engine, launch
  robustness checks, or apply investment suggestions by itself.

This keeps the app safer for assessment: users can inspect state and move
between modules without accidentally triggering expensive or state-changing
operations.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

try:  # Keep the sidebar resilient if key names change during development.
    from ui.state.keys import CURRENT_STEP, PLANNING_SNAPSHOT
except Exception:  # pragma: no cover - defensive fallback for early imports.
    CURRENT_STEP = "current_step"
    PLANNING_SNAPSHOT = "planning_snapshot"


STEP0_NOTICE_ACCEPTED = "step0_educational_notice_accepted"
STEP0_SELECTED_MODULE = "step0_selected_module"
STEP0_PATHWAY = "step0_planning_pathway"

# Timing keys used by Step 5 suggestion modules.
# They are strings here on purpose so the sidebar does not import recommendation
# modules just to render. This avoids circular UI dependencies and keeps the
# global sidebar lightweight.
PRESET_SUGGESTION_TIMING_KEY = "step5_preset_suggestion_timing_v1"
AUTO_OPT_SUGGESTION_TIMING_KEY = "step5_auto_opt_suggestion_timing_v1"
UNIVERSE_SUGGESTION_TIMING_KEY = "step5_universe_suggestion_timing_v1"
SIZE_SUGGESTION_TIMING_KEY = "step5_size_suggestion_timing_v1"

# Main Step 5 diagnostics panel keys. Defined as strings to avoid importing
# post_run.py from the global sidebar and accidentally creating circular UI
# dependencies. The sidebar only toggles visibility; the main panel owns the
# detailed render.
STEP5_RUN_DIAGNOSTICS_EXPANDED_KEY = "step5_run_diagnostics_expanded_v1"
STEP5_SCROLL_TO_RUN_DIAGNOSTICS_KEY = "step5_scroll_to_run_diagnostics_v1"
STEP5_RELIABILITY_EXPANDED_KEY = "step5_reliability_and_robustness_expanded_v1"
STEP5_SCROLL_TO_RELIABILITY_KEY = "step5_scroll_to_reliability_and_robustness_v1"
STEP5_CURRENT_RESULT_IS_FRESH_KEY = "step5_current_result_is_fresh_v1"
STEP5_POST_RUN_IS_STALE_KEY = "step5_post_run_is_stale"

# Personal Finance live-summary defaults. These mirror the first visible state
# of the main Personal Finance Setup screen, so the sidebar is useful even on
# the first render before the main screen has seeded widget-backed values.
PF_DEFAULT_INCOME_WEEKLY = 460.0
PF_DEFAULT_FIXED_WEEKLY = 185.0
PF_DEFAULT_VARIABLE_WEEKLY = 80.0
PF_DEFAULT_DISCRETIONARY_WEEKLY = 35.0
PF_DEFAULT_HORIZON_WEEKS = 12


STEP_NAV_ITEMS: tuple[dict[str, Any], ...] = (
    {"step": 0, "group": "Home", "icon": "", "title": "Home", "subtitle": "Module selector"},
    {"step": 1, "group": "Personal Finance Planner", "icon": "", "title": "Personal Finance Setup", "subtitle": "Budget, target, feasibility"},
    {"step": 2, "group": "Personal Finance Planner", "icon": "", "title": "Personal Finance Setup", "subtitle": "Budget, target, feasibility"},
    {"step": 3, "group": "Personal Finance Planner", "icon": "", "title": "Personal Finance Setup", "subtitle": "Budget, target, feasibility"},
    {"step": 4, "group": "Investment Strategy Lab", "icon": "", "title": "Risk Profile & Asset Universe", "subtitle": ""},
    {"step": 5, "group": "Investment Strategy Lab", "icon": "", "title": "Strategy Engine", "subtitle": ""},
    {"step": 6, "group": "Long-Term Scenario Explorer", "icon": "", "title": "Long-Term Scenario", "subtitle": ""},
    {"step": 7, "group": "Long-Term Scenario Explorer", "icon": "", "title": "Final Report", "subtitle": ""},
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _coerce_mapping(value: Any) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        return dict(value)
    except Exception:
        return {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _money_weekly(value: Any) -> str:
    try:
        return f"£{float(value):,.0f}/week"
    except Exception:
        return "—"


def _money_plain(value: Any) -> str:
    try:
        return f"£{float(value):,.0f}"
    except Exception:
        return "—"


def _period_amount_to_weekly(amount: Any, period: Any = "Weekly") -> float:
    raw = _safe_float(amount, 0.0)
    label = str(period or "Weekly").strip().lower()
    if label.startswith("month"):
        return raw * 12.0 / 52.0
    if label.startswith("year") or label.startswith("annual"):
        return raw / 52.0
    return raw


def _first_positive_value(*values: Any) -> float:
    for value in values:
        candidate = _safe_float(value, 0.0)
        if candidate > 0.0:
            return float(candidate)
    return 0.0


def _personal_finance_sidebar_summary() -> dict[str, float | int | bool | str]:


    """Read the lightweight Personal Finance state without triggering work.

    The sidebar should remain safe even when a screen has not yet seeded all of
    its widget-backed values, or when the user navigates back from a later module.
    Use this order of evidence:

    1. confirmed planning snapshot;
    2. live preview snapshot, if the main screen produced one on a prior rerun;
    3. current widget-backed values;
    4. the same safe defaults shown by the Personal Finance Setup screen.
    """


    snapshot = _planning_snapshot()
    preview = _coerce_mapping(st.session_state.get("planning_snapshot_preview", {}))
    source_snapshot = snapshot or preview

    income_weekly_from_snapshot = _safe_float(source_snapshot.get("monthly_income", 0.0), 0.0) * 12.0 / 52.0
    fixed_weekly_from_snapshot = _safe_float(source_snapshot.get("fixed_essentials_monthly", 0.0), 0.0) * 12.0 / 52.0
    variable_weekly_from_snapshot = _safe_float(source_snapshot.get("variable_essentials_monthly", 0.0), 0.0) * 12.0 / 52.0
    discretionary_weekly_from_snapshot = _safe_float(source_snapshot.get("discretionary_spending_monthly", 0.0), 0.0) * 12.0 / 52.0
    spending_weekly_from_snapshot = fixed_weekly_from_snapshot + variable_weekly_from_snapshot + discretionary_weekly_from_snapshot

    income_weekly_from_widgets = _period_amount_to_weekly(
        st.session_state.get("step1_income_amount", PF_DEFAULT_INCOME_WEEKLY),
        st.session_state.get("step1_income_period", "Weekly"),
    )
    fixed_weekly_from_widgets = _period_amount_to_weekly(
        st.session_state.get("step1_fixed_amount", PF_DEFAULT_FIXED_WEEKLY),
        st.session_state.get("step1_fixed_period", "Weekly"),
    )
    variable_weekly_from_widgets = _period_amount_to_weekly(
        st.session_state.get("step1_variable_amount", PF_DEFAULT_VARIABLE_WEEKLY),
        st.session_state.get("step1_variable_period", "Weekly"),
    )
    discretionary_weekly_from_widgets = _period_amount_to_weekly(
        st.session_state.get("step1_discretionary_amount", PF_DEFAULT_DISCRETIONARY_WEEKLY),
        st.session_state.get("step1_discretionary_period", "Weekly"),
    )
    spending_weekly_from_widgets = (
        fixed_weekly_from_widgets + variable_weekly_from_widgets + discretionary_weekly_from_widgets
    )

    income_weekly = _first_positive_value(income_weekly_from_snapshot, income_weekly_from_widgets)
    spending_weekly = _first_positive_value(spending_weekly_from_snapshot, spending_weekly_from_widgets)
    margin_weekly = _first_positive_value(
        source_snapshot.get("baseline_savings_weekly", 0.0),
        max(income_weekly - spending_weekly, 0.0),
    )
    target_weekly = _first_positive_value(
        st.session_state.get("step1_target_weekly_savings", 0.0),
        source_snapshot.get("target_a_weekly", 0.0),
        source_snapshot.get("weekly_savings", 0.0),
        round(max(margin_weekly, 0.0) * 0.30),
    )
    horizon_weeks = _safe_int(
        st.session_state.get(
            "step2_planning_horizon_weeks",
            source_snapshot.get("planning_horizon_weeks", PF_DEFAULT_HORIZON_WEEKS),
        ),
        PF_DEFAULT_HORIZON_WEEKS,
    )

    if snapshot:
        source_label = "Saved plan snapshot"
    elif preview:
        source_label = "Live preview from main screen"
    else:
        source_label = "Live estimate from visible defaults"

    return {
        "has_any_values": bool(income_weekly > 0.0 or spending_weekly > 0.0 or target_weekly > 0.0),
        "snapshot_stored": bool(snapshot),
        "source_label": str(source_label),
        "income_weekly": float(income_weekly),
        "spending_weekly": float(spending_weekly),
        "margin_weekly": float(margin_weekly),
        "target_weekly": float(target_weekly),
        "horizon_weeks": int(max(1, horizon_weeks)),
    }

def _pct(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.2f}%"
    except Exception:
        return "—"


def _current_step() -> int:
    for key in (CURRENT_STEP, "current_step"):
        try:
            if key in st.session_state:
                return _safe_int(st.session_state.get(key), 0)
        except Exception:
            continue
    return 0


def _go_to_step(step: int) -> None:
    """Move the single-page app to a step without performing heavy work."""
    step_int = max(0, min(7, _safe_int(step, 0)))
    st.session_state[CURRENT_STEP] = step_int
    st.session_state["current_step"] = step_int
    st.rerun()


def _native_button(
    label: str,
    *,
    key: str,
    use_container_width: bool = True,
    disabled: bool = False,
    help: str | None = None,
    icon: str | None = None,
) -> bool:
    """Render a Streamlit button with a safe fallback for older versions.

    Streamlit's native Material icon support is used when available. No custom
    HTML/CSS is injected. If the installed Streamlit version does not support
    the ``icon`` argument, the button falls back to a plain text label.
    """
    kwargs = {
        "label": label,
        "key": key,
        "use_container_width": use_container_width,
        "disabled": disabled,
        "help": help,
    }
    if icon:
        try:
            return bool(st.button(**kwargs, icon=icon))
        except TypeError:
            return bool(st.button(**kwargs))
    return bool(st.button(**kwargs))


def _module_for_step(step: int) -> tuple[str, str]:
    if step == 0:
        return "Home", "Choose starting module"
    if step in {1, 2, 3}:
        return "Personal Finance Planner", "Budget, savings target, feasibility"
    if step in {4, 5}:
        return "Investment Strategy Lab", "Build the asset universe, run the strategy engine, and review results"
    if step == 6:
        return "Long-Term Scenario Explorer", "Scenario simulation, not forecast"
    if step == 7:
        return "Final Report", "Decision-support interpretation"
    return "LifeBudget Micro", "Educational planning prototype"


def _selected_module_label() -> str:
    module_id = str(st.session_state.get(STEP0_SELECTED_MODULE, "") or "")
    return {
        "personal_finance": "Personal Finance Planner",
        "investment_lab": "Investment Strategy Lab",
        "scenario_explorer": "Long-Term Scenario Explorer",
    }.get(module_id, "Full planning flow")


def _selected_pathway_id() -> str:
    return str(st.session_state.get(STEP0_PATHWAY, "") or "")


def _selected_pathway_label() -> str:
    pathway = _selected_pathway_id()
    return {
        "compare_both": "Compare savings and investing",
        "savings_only": "Savings-only pathway",
        "savings_plus_investing": "Savings + investing pathway",
    }.get(pathway, "Not selected yet")


def _educational_notice_is_accepted(step: int) -> bool:
    """Return a robust educational-notice status for the shared sidebar.

    Earlier Step 0 versions used slightly different state-key names. Also, if
    the user has already moved past Step 0, the notice must have been accepted
    in the normal flow, so the sidebar should not display a false pending state.
    """
    accepted_keys = (
        STEP0_NOTICE_ACCEPTED,
        "step0_educational_notice_acknowledged",
        "step0_notice_accepted",
    )
    accepted = any(bool(st.session_state.get(key, False)) for key in accepted_keys)

    if not accepted and int(step) > 0:
        # Keep the sidebar consistent after navigation from Step 0.
        st.session_state[STEP0_NOTICE_ACCEPTED] = True
        accepted = True

    return bool(accepted)


def _planning_snapshot() -> dict:
    for key in (PLANNING_SNAPSHOT, "planning_snapshot"):
        payload = _coerce_mapping(st.session_state.get(key, {}))
        if payload:
            return payload
    return {}


def _asset_panel_summary() -> dict:
    panel = st.session_state.get("asset_panel_df")
    if isinstance(panel, pd.DataFrame) and not panel.empty:
        assets = 0
        rows = int(len(panel))
        if "asset" in panel.columns:
            try:
                assets = int(panel["asset"].nunique())
            except Exception:
                assets = 0
        return {
            "rows": rows,
            "assets": assets,
            "source": str(st.session_state.get("asset_panel_source_label", "Step 4 panel") or "Step 4 panel"),
        }

    run_map = _coerce_mapping(st.session_state.get("step5_last_run_result", {}))
    if run_map:
        return {
            "rows": _safe_int(run_map.get("asset_panel_n_rows", 0), 0),
            "assets": _safe_int(run_map.get("asset_panel_n_assets", 0), 0),
            "source": str(run_map.get("asset_panel_source_label", "Step 4 panel") or "Step 4 panel"),
        }
    return {}


def _latest_run_result() -> dict:
    for key in ("step5_last_run_result", "last_run_result", "run_result"):
        payload = _coerce_mapping(st.session_state.get(key, {}))
        if payload:
            return payload
    return {}


def _investment_context() -> dict:
    return _coerce_mapping(st.session_state.get("investment_context", {}))


def _has_asset_panel() -> bool:
    panel_meta = _asset_panel_summary()
    return bool(_safe_int(panel_meta.get("assets", 0), 0) > 0 and _safe_int(panel_meta.get("rows", 0), 0) > 0)


def _has_step5_result() -> bool:
    return bool(_latest_run_result())


def _has_projection_bridge() -> bool:
    ctx = _investment_context()
    oos_returns = ctx.get("oos_returns_monthly", [])
    return bool(isinstance(oos_returns, list) and len(oos_returns) > 0)


def _has_projection_result() -> bool:
    """Return True when Long-Term Scenario has produced reportable output.

    Step 6 stores projections under a few legacy/current keys. Keep this
    lightweight: only read session state, never trigger a recomputation.
    """
    likely_keys = (
        "investment_projection_result",
        "investment_projection_compare_results",
        "step6_compare_branch_result",
        "step6_projection_result",
        "step6_last_projection_result",
        "long_term_projection_result",
        "projection_result",
        "step7_report_context",
    )
    for key in likely_keys:
        value = st.session_state.get(key)
        if isinstance(value, dict) and len(value) > 0:
            return True
        if isinstance(value, list) and len(value) > 0:
            return True
        if value is not None and not isinstance(value, (dict, list, str)):
            return bool(value)
        if isinstance(value, str) and value.strip():
            return True
    return False


def _is_investment_pathway(step: int) -> bool:
    pathway = _selected_pathway_id()
    selected_module = str(st.session_state.get(STEP0_SELECTED_MODULE, "") or "")
    return bool(
        pathway in {"compare_both", "savings_plus_investing"}
        or selected_module == "investment_lab"
        or int(step) in {4, 5}
        or _has_asset_panel()
        or _has_step5_result()
    )


def _step_access_state(target_step: int, current_step: int) -> tuple[bool, str]:
    """Return whether the sidebar should allow navigation to target_step.

    This intentionally stays conservative. The main screen remains the canonical
    wizard flow. The sidebar helps users move around without letting them jump
    into screens that are likely missing required state.
    """
    target_step = _safe_int(target_step, 0)
    current_step = _safe_int(current_step, 0)

    if target_step == 0:
        return True, "Return to the module selector."

    notice_ok = _educational_notice_is_accepted(current_step)
    snapshot = _planning_snapshot()

    if target_step in {1, 2, 3}:
        if notice_ok or current_step >= target_step or snapshot:
            return True, "Open this planning step."
        return False, "Accept the educational notice on Home first."

    if target_step == 4:
        if current_step >= 4 or (notice_ok and _is_investment_pathway(current_step)):
            return True, "Open Risk Profile & Asset Universe."
        return False, "Choose an investing pathway before opening Risk Profile & Asset Universe."

    if target_step == 5:
        if current_step >= 5 or _has_asset_panel() or _has_step5_result():
            return True, "Open the engine workspace."
        return False, "Prepare the market-data panel first."

    if target_step == 6:
        if current_step >= 6 or _has_step5_result() or _has_projection_bridge() or snapshot:
            return True, "Open long-term scenario exploration."
        return False, "Complete the planning setup or run Step 5 first."

    if target_step == 7:
        if current_step >= 7 or _has_projection_result():
            return True, "Open insights and reports."
        return False, "Generate a long-term scenario before opening reports."

    return False, "Step unavailable."


def _timing_total(key: str) -> float:
    payload = _coerce_mapping(st.session_state.get(key, {}))
    return _safe_float(payload.get("total_seconds", 0.0), 0.0)


def _format_seconds(value: Any) -> str:
    seconds = _safe_float(value, 0.0)
    if seconds <= 0.0:
        return "—"
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes = seconds / 60.0
    return f"{minutes:.1f} min"


def _suggestion_timing_rows() -> list[tuple[str, float, int, int]]:
    """Return Step 5 suggestion timing rows without importing suggestion modules."""
    rows: list[tuple[str, float, int, int]] = []
    for label, key in (
        ("Preset", PRESET_SUGGESTION_TIMING_KEY),
        ("Engine tuning", AUTO_OPT_SUGGESTION_TIMING_KEY),
        ("Universe", UNIVERSE_SUGGESTION_TIMING_KEY),
        ("Size", SIZE_SUGGESTION_TIMING_KEY),
    ):
        payload = _coerce_mapping(st.session_state.get(key, {}))
        seconds = _safe_float(payload.get("total_seconds", 0.0), 0.0)
        candidates = _safe_int(payload.get("candidate_count", 0), 0)
        accepted = _safe_int(payload.get("accepted_count", 0), 0)
        if seconds > 0.0 or candidates > 0:
            rows.append((label, seconds, candidates, accepted))
    return rows


def _step5_improvement_flow_completed() -> bool:
    """Return True once the terminal Step 5 improvement phase is resolved.

    The main Step 5 post-run panel imports the exact phase-key constants from
    the suggestion modules. The global sidebar deliberately avoids those imports
    to prevent circular UI dependencies, so this helper detects the final
    universe-size decision from session-state keys. A truthy Step 5 size
    applied/skipped marker means the optional improvement flow is complete.
    """
    try:
        items = list(st.session_state.items())
    except Exception:
        items = []

    for key, value in items:
        key_text = str(key).lower()
        if "step5" not in key_text or "size" not in key_text:
            continue
        if not ("applied" in key_text or "skipped" in key_text):
            continue
        if "timing" in key_text or "suggestion_count" in key_text:
            continue
        if isinstance(value, bool):
            if value:
                return True
            continue
        if value is not None and str(value).strip():
            return True

    return False


def _step5_result_is_current_for_sidebar() -> bool:
    """Return True only when the sidebar can safely open reliability checks.

    Completing all optional improvement phases is not enough. If the user has
    applied a suggestion or changed engine inputs after the last executed run,
    reliability and robustness should stay closed until the updated setup is run
    again. The Strategy Engine workspace owns the authoritative freshness flag;
    this sidebar also double-checks the stored run/config fingerprints when they
    are available.
    """
    run_map = _latest_run_result()
    if not run_map:
        return False

    explicit_fresh = st.session_state.get(STEP5_CURRENT_RESULT_IS_FRESH_KEY, None)
    post_run_stale = bool(st.session_state.get(STEP5_POST_RUN_IS_STALE_KEY, False))

    # The global sidebar is rendered before the active Step 5 page. If the
    # previous completed render says the result is fresh and not stale, trust
    # that flag before inspecting transient current-signature keys. Those keys
    # can briefly lag behind the main page and incorrectly disable the button.
    if isinstance(explicit_fresh, bool) and explicit_fresh and not post_run_stale:
        return True

    if post_run_stale:
        return False

    last_signature = str(st.session_state.get("step5_last_run_signature", "") or run_map.get("run_signature", "") or "")
    current_signature = str(
        st.session_state.get("step5_current_run_signature", "")
        or st.session_state.get("step5_current_input_signature", "")
        or ""
    )
    if last_signature and current_signature and last_signature != current_signature:
        return False

    last_config_fp = str(st.session_state.get("step5_last_config_fingerprint", "") or run_map.get("config_fingerprint", "") or "")
    current_config_fp = str(st.session_state.get("step5_current_config_fingerprint", "") or "")
    if last_config_fp and current_config_fp and last_config_fp != current_config_fp:
        return False

    if isinstance(explicit_fresh, bool):
        return bool(explicit_fresh)

    # Legacy fallback for older sessions before the workspace wrote the explicit flag.
    return bool(run_map)


def _toggle_step5_run_timings() -> None:
    """Toggle the main Step 5 run-timings panel from the sidebar.

    Streamlit button callbacks run before the script body is rendered, so the
    sidebar label updates immediately from Show -> Hide (or Hide -> Show) on
    the rerun caused by the click, without needing a manual st.rerun().
    """
    next_visible = not bool(st.session_state.get(STEP5_RUN_DIAGNOSTICS_EXPANDED_KEY, False))
    st.session_state[STEP5_RUN_DIAGNOSTICS_EXPANDED_KEY] = bool(next_visible)
    st.session_state[STEP5_SCROLL_TO_RUN_DIAGNOSTICS_KEY] = bool(next_visible)


def _toggle_step5_reliability() -> None:
    """Toggle the main Step 5 reliability panel from the sidebar.

    Keep this independent from the run-timings toggle. The callback pattern is
    important: it updates session_state before the sidebar is rendered again,
    so the label changes immediately from Show -> Hide without relying on a
    later unrelated rerun.
    """
    can_open = bool(_step5_improvement_flow_completed() and _step5_result_is_current_for_sidebar())
    if not can_open:
        st.session_state[STEP5_RELIABILITY_EXPANDED_KEY] = False
        st.session_state[STEP5_SCROLL_TO_RELIABILITY_KEY] = False
        return

    next_visible = not bool(st.session_state.get(STEP5_RELIABILITY_EXPANDED_KEY, False))
    st.session_state[STEP5_RELIABILITY_EXPANDED_KEY] = bool(next_visible)
    st.session_state[STEP5_SCROLL_TO_RELIABILITY_KEY] = bool(next_visible)


# ---------------------------------------------------------------------------
# Sidebar rendering blocks
# ---------------------------------------------------------------------------


def _branch_items_for_step(step: int) -> tuple[str, tuple[int, ...], str]:
    """Return only the branch currently selected or being used.

    The Home cards define three product branches:
    - Personal Finance Planner -> Steps 1-3
    - Investment Strategy Lab -> Steps 4-5
    - Long-Term Scenario Explorer -> Steps 6-7

    The sidebar should mirror that branch model. It should not show the whole
    seven-step wizard when the user is inside one module.
    """
    step = _safe_int(step, 0)
    selected_module = str(st.session_state.get(STEP0_SELECTED_MODULE, "") or "")
    pathway = _selected_pathway_id()

    # If the user is on Home but has selected one of the module cards, preview
    # only that module's branch instead of showing every possible step.
    if step == 0:
        if selected_module == "personal_finance":
            return (
                "Personal Finance",
                (1, 2, 3),
                "Finance branch: one combined setup screen for budget, target, and feasibility.",
            )
        if selected_module == "investment_lab":
            return (
                "Investment Strategy Lab",
                (4, 5),
                "Investment branch: data universe and engine workspace.",
            )
        if selected_module == "scenario_explorer":
            return (
                "Scenario & Reports",
                (6, 7),
                "Scenario branch: long-term scenario and final insights.",
            )
        return (
            "No active branch yet",
            tuple(),
            "Choose a module on Home to reveal the relevant pathway.",
        )

    # Once the user is inside a step, the current step decides the visible
    # branch. This keeps long-term scenarios/7 visually separate from the investment engine,
    # even if the scenario is using Step 5 results.
    if step in {1, 2, 3}:
        return (
            "Personal Finance",
            (1, 2, 3),
            "Finance branch: one combined setup screen for budget, target, and feasibility.",
        )

    if step in {4, 5} or selected_module == "investment_lab":
        return (
            "Investment Strategy Lab",
            (4, 5),
            "Investment branch: data universe and engine workspace.",
        )

    if step in {6, 7} or selected_module == "scenario_explorer" or pathway == "savings_only":
        return (
            "Scenario & Reports",
            (6, 7),
            "Scenario branch: long-term scenario and final insights.",
        )

    return (
        "Personal Finance",
        (1, 2, 3),
        "Finance branch: one combined setup screen for budget, target, and feasibility.",
    )

def _step_nav_item(target_step: int) -> dict[str, Any]:
    for item in STEP_NAV_ITEMS:
        if _safe_int(item.get("step", -1), -1) == int(target_step):
            return dict(item)
    return {"step": target_step, "title": f"Step {target_step}", "subtitle": ""}


def _render_branch_status_item(item: dict[str, Any], current_step: int) -> None:
    target = _safe_int(item.get("step", 0), 0)
    title = str(item.get("title", "") or "")
    subtitle = str(item.get("subtitle", "") or "")
    allowed, reason = _step_access_state(target, current_step)
    is_active = target == int(current_step)

    body = f"**{title}**  \n{subtitle}" if subtitle else f"**{title}**"

    if is_active:
        # Native Streamlit status card. It is intentionally not a button.
        st.info(body)
        return

    if not allowed:
        st.caption(f"Locked · {title}")
        if subtitle:
            st.caption(subtitle)
        st.caption(reason)
        return

    st.markdown(body)


def _render_personal_finance_module_status(step: int) -> None:
    """Render the Personal Finance branch as one screen, not three fake steps.

    The current UI consolidates the old Step 1-3 flow into a single Personal
    Finance Setup page. Showing Step 1 / Step 2 / Step 3 in the sidebar would
    imply separate navigation targets that no longer exist on the main screen.
    Keep this as a module status summary instead.
    """
    is_active = int(step) in {1, 2, 3}
    body = (
        "**Personal Finance Setup**  \n"
        "Budget estimate, savings target, and feasibility check."
    )

    if is_active:
        st.info(body)
    else:
        st.markdown(body)

    st.caption("This branch is now one combined screen, so it is not split into separate sidebar steps.")


def _render_step_navigation(step: int) -> None:
    st.markdown("### Navigation")

    home_disabled = int(step) == 0
    if _native_button(
        "Home",
        key="global_sidebar_home_button",
        use_container_width=True,
        disabled=home_disabled,
        help="Return to the module selector." if not home_disabled else "You are already on Home.",
        icon=":material/home:",
    ):
        _go_to_step(0)

    branch_title, branch_steps, branch_caption = _branch_items_for_step(step)
    st.markdown(f"**{branch_title}**")
    st.caption(branch_caption)

    if branch_title == "Personal Finance":
        _render_personal_finance_module_status(step)
        return

    if not branch_steps:
        return

    for target_step in branch_steps:
        _render_branch_status_item(_step_nav_item(target_step), step)

def _render_compact_status(step: int) -> None:
    module, _description = _module_for_step(step)
    st.caption(f"Current: {module}")

    if step >= 4:
        panel_meta = _asset_panel_summary()
        if panel_meta:
            st.caption(
                f"Market-data panel: {panel_meta.get('assets', 0)} assets · "
                f"{panel_meta.get('rows', 0)} rows"
            )

    run_map = _latest_run_result()
    if step >= 5 and run_map:
        perf = _coerce_mapping(run_map.get("performance_summary", {}))
        st.caption(
            f"Last run: Sharpe {_safe_float(perf.get('sharpe', 0.0), 0.0):.2f} · "
            f"CAGR {_pct(perf.get('cagr', 0.0))}"
        )


def _readiness_line(label: str, ready: bool, detail: str) -> None:
    status = "Ready" if ready else "Pending"
    st.caption(f"**{label}:** {status} · {detail}")


def _render_readiness_status(step: int) -> None:
    """Show what is already available without running anything heavy."""
    with st.expander("Readiness status", expanded=False):
        notice_ok = _educational_notice_is_accepted(step)
        snapshot = _planning_snapshot()
        panel_meta = _asset_panel_summary()
        run_map = _latest_run_result()
        projection_bridge = _has_projection_bridge()
        projection_result = _has_projection_result()

        _readiness_line(
            "Educational notice",
            notice_ok,
            "accepted" if notice_ok else "pending on Home",
        )
        _readiness_line(
            "Personal finance snapshot",
            bool(snapshot),
            "stored" if snapshot else "not stored yet",
        )
        _readiness_line(
            "Market-data panel",
            _has_asset_panel(),
            (
                f"{panel_meta.get('assets', 0)} assets · {panel_meta.get('rows', 0)} rows"
                if panel_meta
                else "prepare Step 4 first"
            ),
        )
        _readiness_line(
            "Strategy engine result",
            bool(run_map),
            "available for diagnostics/projection" if run_map else "run the Strategy Engine",
        )
        _readiness_line(
            "Scenario projection input",
            projection_bridge or bool(snapshot),
            "ready" if (projection_bridge or bool(snapshot)) else "needs planning or strategy result",
        )
        _readiness_line(
            "Insights report",
            projection_result,
            "projection context available" if projection_result else "generate scenario first",
        )

        st.caption("This panel only reads app state. It does not refresh data, run the engine, or launch robustness checks.")


def _render_current_context(step: int) -> None:
    module, description = _module_for_step(step)
    st.markdown("### Current context")
    st.write(f"**Module:** {module}")
    st.caption(description)

    if step == 0:
        st.write(f"**Selected start:** {_selected_module_label()}")
    else:
        st.write(f"**Pathway:** {_selected_pathway_label()}")

    if _educational_notice_is_accepted(step):
        st.success("Educational notice accepted")
    else:
        st.warning("Educational notice pending")

    if step >= 4:
        panel_meta = _asset_panel_summary()
        if panel_meta:
            st.caption(
                f"Market-data panel: {panel_meta.get('source', 'Step 4 panel')} · "
                f"assets={panel_meta.get('assets', 0)} · rows={panel_meta.get('rows', 0)}"
            )
        else:
            st.caption("Market-data panel: cached deployment panel prepared in Step 4")


def _render_available_checks(step: int) -> None:
    st.markdown("### Available checks")

    if step in {1, 2}:
        st.caption("Budget pressure and savings feasibility are handled inside the Personal Finance Planner.")
    elif step == 3:
        st.caption("Short-term stress assumptions are available in the feasibility panel.")
    elif step == 4:
        st.caption("Universe diagnostics are available after preparing the asset panel.")
    elif step == 5:
        st.caption("Engine diagnostics, suggestion timings, and start-date robustness are available after a real run.")
    elif step == 6:
        st.caption("Scenario comparison and horizon comparison are available in the projection screen.")
    elif step == 7:
        st.caption("Final consistency checks and decision-support caveats are summarised here.")
    else:
        st.caption("Checks appear inside the relevant module once enough inputs exist.")

    st.info("Heavy checks are run from the main screen, not automatically from the sidebar.")


def _render_tools_and_checks(step: int) -> None:
    with st.expander("Checks & shortcuts", expanded=False):
        _render_available_checks(step)

        st.divider()
        st.markdown("**Safe shortcuts**")

        # Keep this list intentionally small. These buttons only navigate; they
        # never run engines, refresh data, apply suggestions, or launch exports.
        if step != 4:
            step4_enabled, step4_reason = _step_access_state(4, step)
            if st.button(
                "Open Risk Profile & Asset Universe",
                key="global_sidebar_tool_open_step4",
                use_container_width=True,
                disabled=not step4_enabled,
                help=step4_reason,
            ):
                _go_to_step(4)

        if step != 5:
            step5_enabled, step5_reason = _step_access_state(5, step)
            if st.button(
                "Open Strategy Engine",
                key="global_sidebar_tool_open_step5",
                use_container_width=True,
                disabled=not step5_enabled,
                help=step5_reason,
            ):
                _go_to_step(5)

        if step != 6:
            step6_enabled, step6_reason = _step_access_state(6, step)
            if st.button(
                "Open Long-Term Scenario",
                key="global_sidebar_tool_open_step6",
                use_container_width=True,
                disabled=not step6_enabled,
                help=step6_reason,
            ):
                _go_to_step(6)

        if step != 7:
            step7_enabled, step7_reason = _step_access_state(7, step)
            if st.button(
                "Open Final Report",
                key="global_sidebar_tool_open_step7",
                use_container_width=True,
                disabled=not step7_enabled,
                help=step7_reason,
            ):
                _go_to_step(7)

        st.caption("Run, improve, refresh, robustness, and export actions stay on the main screen to avoid accidental heavy computation.")


def _render_audit_diagnostics(step: int) -> None:
    with st.expander("Technical diagnostics", expanded=False):
        snapshot = _planning_snapshot()
        if snapshot:
            st.markdown("**Personal finance snapshot**")
            income_weekly = _safe_float(snapshot.get("monthly_income", 0.0), 0.0) * 12.0 / 52.0
            margin_weekly = _safe_float(snapshot.get("baseline_savings_weekly", 0.0), 0.0)
            target_weekly = _safe_float(snapshot.get("target_a_weekly", st.session_state.get("target_a_weekly", 0.0)), 0.0)
            st.caption(
                f"Income≈{_money_weekly(income_weekly)} · "
                f"margin≈{_money_weekly(margin_weekly)} · "
                f"target≈{_money_weekly(target_weekly)}"
            )
        else:
            st.caption("No personal finance snapshot stored yet.")

        panel_meta = _asset_panel_summary()
        if panel_meta:
            st.divider()
            st.markdown("**Market-data panel**")
            st.caption(
                f"source={panel_meta.get('source', 'Step 4 panel')} · "
                f"assets={panel_meta.get('assets', 0)} · rows={panel_meta.get('rows', 0)}"
            )

        run_map = _latest_run_result()
        if run_map:
            st.divider()
            st.markdown("**Last Strategy Engine run**")
            perf = _coerce_mapping(run_map.get("performance_summary", {}))
            engine_timing = _coerce_mapping(run_map.get("engine_timing", {}))
            total_engine = _safe_float(engine_timing.get("total_engine", 0.0), 0.0)
            st.caption(
                f"CAGR={_pct(perf.get('cagr', 0.0))} · "
                f"Sharpe={_safe_float(perf.get('sharpe', 0.0), 0.0):.2f} · "
                f"MaxDD=-{100.0 * abs(_safe_float(perf.get('max_drawdown', 0.0), 0.0)):.2f}%"
            )
            if total_engine > 0.0:
                st.caption(f"Engine timing: {total_engine:.2f}s")
            run_sig = str(run_map.get("run_signature", "") or "")
            cfg_fp = str(run_map.get("config_fingerprint", "") or "")
            if run_sig:
                st.caption(f"run_signature={run_sig}")
            if cfg_fp:
                st.caption(f"config_fp={cfg_fp}")

        suggestion_totals = [
            ("Preset", _timing_total(PRESET_SUGGESTION_TIMING_KEY)),
            ("Engine tuning", _timing_total(AUTO_OPT_SUGGESTION_TIMING_KEY)),
            ("Universe", _timing_total(UNIVERSE_SUGGESTION_TIMING_KEY)),
            ("Size", _timing_total(SIZE_SUGGESTION_TIMING_KEY)),
        ]
        visible = [(name, seconds) for name, seconds in suggestion_totals if seconds > 0.0]
        if visible:
            st.divider()
            st.markdown("**Suggestion timing overhead**")
            for name, seconds in visible:
                st.caption(f"{name}: {seconds:.2f}s")

        investment_context = _investment_context()
        oos_returns = investment_context.get("oos_returns_monthly", [])
        if isinstance(oos_returns, list) and oos_returns:
            st.divider()
            st.markdown("**Projection bridge**")
            st.caption(f"{len(oos_returns)} monthly OOS returns available for long-term scenarios.")

        if not snapshot and not panel_meta and not run_map:
            st.caption("Diagnostics will populate after the planner, investment panel, or engine run has data.")


def _render_help(step: int) -> None:
    with st.expander("Help & explanations", expanded=False):
        st.markdown("**Educational framing**")
        st.caption(
            "LifeBudget Micro is a planning and scenario-exploration prototype. "
            "It is not financial advice and does not predict future returns."
        )

        if step in {5, 6, 7}:
            st.divider()
            st.markdown("**Common investment metrics**")
            st.caption("**CAGR:** historical annualised growth rate in the tested window.")
            st.caption("**Sharpe:** return per unit of volatility; useful for comparing risk-adjusted performance.")
            st.caption("**MaxDD:** maximum historical peak-to-trough decline; a pain-test metric.")
            st.caption("Past performance refers to the past and is not a reliable indicator of future results.")
        elif step in {1, 2, 3}:
            st.divider()
            st.markdown("**Personal finance terms**")
            st.caption("**Free margin:** estimated money left after essential and discretionary weekly spending.")
            st.caption("**Savings target:** weekly amount tested against your current cash-flow estimate.")
            st.caption("**Feasibility:** short-term stress check, not a guarantee that real spending will match the scenario.")


def _render_home_sidebar_minimal() -> None:
    """Render a quiet Home sidebar before the user enters a module."""
    st.markdown("### Navigation")
    _native_button(
        "Home",
        key="global_sidebar_home_button_step0",
        use_container_width=True,
        disabled=True,
        help="You are already on Home.",
        icon=":material/home:",
    )

    st.markdown("**Start here**")
    st.caption("Choose a module from the main page. The sidebar becomes more detailed once a module is open.")

    st.markdown("**Modules**")
    st.caption("Personal Finance Planner")
    st.caption("Investment Strategy Lab")
    st.caption("Long-Term Scenario Explorer")

    with st.expander("About this prototype", expanded=False):
        st.caption(
            "LifeBudget Micro is an educational planning and scenario-exploration prototype. "
            "It is not financial advice and does not predict future returns."
        )


def _render_personal_finance_terms() -> None:
    with st.expander("Personal finance terms", expanded=False):
        st.markdown("**Budget estimate**")
        st.caption("A quick weekly view of income, spending and free margin. It is a planning baseline, not an exact bank statement.")

        st.markdown("**Income**")
        st.caption("Money coming in before weekly spending is subtracted. It can be estimated quickly or edited more exactly in detailed controls.")

        st.markdown("**Essentials**")
        st.caption("Necessary spending such as fixed bills and variable essentials. These are separated from flexible spending so pressure is easier to see.")

        st.markdown("**Discretionary spending**")
        st.caption("Flexible spending that can sometimes be reduced. The model treats this as the first place to test savings capacity.")

        st.markdown("**Free margin**")
        st.caption("Estimated money left after essential and discretionary weekly spending. This is the source of the savings/contribution bridge.")

        st.markdown("**Savings target**")
        st.caption("Weekly amount tested against the current cash-flow estimate. It becomes the contribution assumption for later modules.")

        st.markdown("**Cautious / Balanced / Stretch**")
        st.caption("Preset target levels. Cautious uses less free margin, Balanced uses a middle share, and Stretch tests a more demanding savings habit.")

        st.markdown("**Feasibility check**")
        st.caption("Short-term stress check showing whether the savings target still looks workable under uncertainty. It is not a guarantee.")

        st.markdown("**Conservative / Expected / High case**")
        st.caption("Three short-term outcomes from the feasibility model: lower, central and higher cases for the selected horizon.")

        st.markdown("**Short-term path chart**")
        st.caption("Compares the baseline path with the savings-target path over the selected short-term horizon.")

        st.markdown("**Uncertainty band**")
        st.caption("The shaded 10–90% range around the target plan. Wider bands mean the short-term path is more sensitive to spending variability.")

        st.markdown("**Life event stress test**")
        st.caption("Optional one-off cost added to the short-term plan. It helps test whether the target leaves room for surprises.")


def _render_personal_finance_q_and_a() -> None:
    """Render a compact onboarding Q&A for the Personal Finance module.

    Keep this lightweight. The main screen owns the controls; the sidebar only
    explains how to use them and why the values matter downstream.
    """
    with st.expander("Personal finance Q&A", expanded=False):
        st.markdown("**Do these numbers need to be exact?**")
        st.caption(
            "No. This is a planning baseline for the prototype. Rough weekly estimates "
            "are enough for testing the flow; exact editing remains optional."
        )

        st.markdown("**Why does this screen use weekly values?**")
        st.caption(
            "Weekly numbers make income, spending, free margin and savings target easier "
            "to compare on the same scale."
        )

        st.markdown("**What is free margin?**")
        st.caption(
            "Free margin is the estimated weekly money left after spending. It is the amount the app uses to test whether a savings target is realistic."
        )

        st.markdown("**Why split essentials and discretionary spending?**")
        st.caption(
            "Essentials show the harder-to-change spending base. Discretionary spending shows the flexible part that may absorb savings targets or stress events."
        )

        st.markdown("**What does the feasibility check test?**")
        st.caption(
            "It tests whether the selected savings target still looks workable over the short-term horizon after uncertainty is applied."
        )

        st.markdown("**What do the stress-test controls change?**")
        st.caption(
            "They adjust the short-term horizon, uncertainty level and optional one-off life-event cost. They do not change investment results directly."
        )

        st.markdown("**Why does this affect investing later?**")
        st.caption(
            "The savings target becomes the contribution bridge used by the investment "
            "and Long-Term Scenario modules."
        )


def _render_personal_finance_plan_snapshot() -> None:
    summary = _personal_finance_sidebar_summary()
    st.markdown("### Current plan")

    if not bool(summary.get("has_any_values", False)):
        st.caption("Adjust the budget and savings target on the main screen to populate this summary.")
        return

    st.caption(str(summary.get("source_label", "Live estimate")))
    st.caption(f"Income: **{_money_weekly(summary.get('income_weekly', 0.0))}**")
    st.caption(f"Spending: **{_money_weekly(summary.get('spending_weekly', 0.0))}**")
    st.caption(f"Free margin: **{_money_weekly(summary.get('margin_weekly', 0.0))}**")

    target = _safe_float(summary.get("target_weekly", 0.0), 0.0)
    margin = _safe_float(summary.get("margin_weekly", 0.0), 0.0)
    if target > 0.0:
        st.caption(f"Savings target: **{_money_weekly(target)}**")
        if margin > 0.0:
            st.caption(f"Target uses: **{100.0 * target / margin:.0f}% of free margin**")
    else:
        st.caption("Savings target: not set yet")

    st.caption(f"Horizon: **{_safe_int(summary.get('horizon_weeks', PF_DEFAULT_HORIZON_WEEKS), PF_DEFAULT_HORIZON_WEEKS)} weeks**")

def _render_personal_finance_next_action() -> None:
    st.markdown("### Next")
    st.caption("Use the main screen button to build the comparison branch, or open the investment setup when ready.")

    step4_enabled, step4_reason = _step_access_state(4, _current_step())
    if st.button(
        "Open Investment Strategy Lab",
        key="global_sidebar_personal_finance_open_investment_lab",
        use_container_width=True,
        disabled=not step4_enabled,
        help=step4_reason,
    ):
        _go_to_step(4)


def _render_personal_finance_sidebar(step: int) -> None:
    """Render a focused sidebar for the combined Personal Finance Setup module."""
    st.markdown("### Navigation")
    if _native_button(
        "Home",
        key="global_sidebar_home_button_personal_finance",
        use_container_width=True,
        disabled=False,
        help="Return to the module selector.",
        icon=":material/home:",
    ):
        _go_to_step(0)

    st.markdown("**Personal Finance Planner**")
    st.caption("Set a weekly budget baseline, choose a savings target, and check short-term feasibility.")
    st.info("**Personal Finance Setup**  \nBudget estimate, savings target, and feasibility check.")

    st.divider()
    _render_personal_finance_plan_snapshot()

    st.divider()
    _render_personal_finance_q_and_a()
    _render_personal_finance_terms()


def _format_count(value: Any) -> str:
    try:
        return f"{int(float(value)):,.0f}"
    except Exception:
        return "0"


def _step4_selected_assets_count() -> int:
    """Best-effort count of the currently selected Step 4 primary universe."""
    for key in (
        "last_used_universe_assets",
        "recommended_universe_assets",
        "selected_assets",
        "last_recommendation_candidate_assets",
    ):
        raw = st.session_state.get(key)
        if isinstance(raw, (list, tuple, set)) and len(raw) > 0:
            return int(len(raw))
    return _safe_int(st.session_state.get("universe_size", 0), 0)


def _step4_current_universe_summary() -> dict[str, Any]:
    """Read the Step 4 universe state without importing Step 4 modules."""
    philosophy = str(st.session_state.get("investment_philosophy", "Balanced") or "Balanced")
    strategy = str(st.session_state.get("universe_strategy", "Core multi-asset") or "Core multi-asset")
    size = _safe_int(st.session_state.get("universe_size", 25), 25)
    selected_count = _step4_selected_assets_count()
    custom_enabled = bool(st.session_state.get("universe_custom_enabled", False))

    if custom_enabled:
        mode = "custom basket"
    elif selected_count and selected_count > size:
        mode = "expanded basket"
    else:
        # Keep this simple: Risk & Universe owns the detailed composition note.
        mode = "recommended basket"

    return {
        "philosophy": philosophy,
        "strategy": strategy,
        "size": int(size),
        "selected_count": int(selected_count or size),
        "mode": mode,
        "custom_enabled": custom_enabled,
    }


def _step4_funding_bridge_summary() -> dict[str, Any]:
    """Return the contribution bridge used later by investment/scenario modules."""
    ctx = _investment_context()
    pf = _personal_finance_sidebar_summary()

    monthly = _safe_float(ctx.get("monthly_contribution", 0.0), 0.0)
    weekly = _safe_float(ctx.get("weekly_equivalent", 0.0), 0.0)

    if monthly <= 0.0 and weekly <= 0.0:
        weekly = _safe_float(pf.get("target_weekly", 0.0), 0.0)
        monthly = weekly * 52.0 / 12.0 if weekly > 0.0 else 0.0

    if ctx:
        source = "Personal Finance Setup"
    elif bool(pf.get("has_any_values", False)):
        source = str(pf.get("source_label", "Personal Finance estimate"))
    else:
        source = "Demo fallback"

    return {
        "monthly": float(monthly),
        "weekly": float(weekly),
        "source": source,
    }


def _render_step4_navigation_block() -> None:
    st.markdown("### Navigation")
    if _native_button(
        "Home",
        key="global_sidebar_home_button_step4",
        use_container_width=True,
        disabled=False,
        help="Return to the module selector.",
        icon=":material/home:",
    ):
        _go_to_step(0)

    st.markdown("**Investment Strategy Lab**")
    st.caption("Choose the risk profile and asset universe that will feed the Strategy Engine.")

    # Match the Scenario & Reports sidebar pattern: branch items are passive
    # status labels, not navigation buttons. Home remains the only action here.
    _render_branch_status_item(_step_nav_item(4), 4)
    _render_branch_status_item(_step_nav_item(5), 4)


def _render_step4_current_universe() -> None:
    summary = _step4_current_universe_summary()
    st.markdown("### Current universe")
    st.caption(f"Risk profile: **{summary['philosophy']}**")
    st.caption(f"Basket: **{summary['strategy']}**")
    st.caption(f"Selected size: **{summary['size']} assets**")
    if int(summary.get("selected_count", 0)) != int(summary.get("size", 0)):
        st.caption(f"Prepared basket: **{summary['selected_count']} assets**")
    st.caption(f"Mode: **{summary['mode']}**")


def _render_step4_funding_bridge() -> None:
    bridge = _step4_funding_bridge_summary()
    st.markdown("### Funding bridge")
    monthly = _safe_float(bridge.get("monthly", 0.0), 0.0)
    weekly = _safe_float(bridge.get("weekly", 0.0), 0.0)
    if monthly > 0.0 or weekly > 0.0:
        st.caption(f"Monthly contribution: **{_money_plain(monthly)}/mo**")
        st.caption(f"Weekly equivalent: **{_money_weekly(weekly)}**")
        st.caption(f"Source: {bridge.get('source', 'Personal Finance Setup')}")
    else:
        st.caption("No contribution bridge found yet. The demo can still use a fallback, but Personal Finance gives the cleaner path.")


def _render_step4_market_panel_status() -> None:
    st.markdown("### Market-data status")
    panel_meta = _asset_panel_summary()
    frequency = str(st.session_state.get("asset_return_frequency", "monthly") or "monthly").lower()

    if panel_meta:
        assets = _safe_int(panel_meta.get("assets", 0), 0)
        rows = _safe_int(panel_meta.get("rows", 0), 0)
        source = str(panel_meta.get("source", "Cached panel") or "Cached panel")
        if assets > 0 and rows > 0:
            st.success("Market-data ready for Strategy Engine.")
            st.caption(f"**{_format_count(assets)} assets · {_format_count(rows)} rows**")
            st.caption(f"Returns: **{frequency}**")
            st.caption(f"Source: {source}")
            return

    st.warning("Market-data panel not ready yet.")
    st.caption("Use the main Risk Profile and Asset Universe screen to prepare or refresh the panel before running the Strategy Engine.")


def _render_step4_market_data_tools_toggle() -> None:
    with st.expander("Diagnostics", expanded=False):
        st.caption(
            "Optional market-data preview, audit details, and CSV export controls. "
            "This only shows or hides diagnostics on the main screen; it does not refresh data or run the engine."
        )

        tools_visible = bool(st.session_state.get("step4_show_market_data_tools", False))
        button_label = "Hide diagnostics & downloads" if tools_visible else "Show diagnostics & downloads"
        if st.button(
            button_label,
            key="global_sidebar_step4_toggle_market_data_tools",
            use_container_width=True,
            help="Show or hide the technical market-data preview, audit details, and CSV exports on the main screen.",
        ):
            st.session_state["step4_show_market_data_tools"] = not tools_visible
            st.rerun()


def _render_step4_q_and_a() -> None:
    with st.expander("Risk & Universe Q&A", expanded=False):
        st.markdown("**Does this screen optimise the portfolio?**")
        st.caption("No. It prepares the asset universe and market-data panel. Optimisation happens in the Strategy Engine.")
        st.markdown("**Why can the panel asset count differ from the selected basket?**")
        st.caption("The selected basket is the intended universe; the prepared panel reflects the cached/demo market data available for the engine.")
        st.markdown("**Why use cached market data?**")
        st.caption("It keeps the deployed demo reliable and avoids live Yahoo/rate-limit issues.")


def _render_step4_help() -> None:
    with st.expander("Risk & Universe terms", expanded=False):
        st.markdown("**Risk profile**")
        st.caption("Broad investment posture: Growth, Balanced, or Defensive.")

        st.markdown("**Universe basket**")
        st.caption("The asset list available to the Strategy Engine. This screen chooses the assets; it does not optimise weights.")

        st.markdown("**Equity**")
        st.caption("Shares or equity-market funds; higher growth potential, higher market volatility.")

        st.markdown("**Fixed Income**")
        st.caption("Bond-like funds; often used for stability or income.")

        st.markdown("**Commodity**")
        st.caption("Gold or broad commodity exposure; may behave differently from stocks and bonds.")

        st.markdown("**Real Estate**")
        st.caption("Property/REIT exposure; can diversify the basket, but still has market risk.")

        st.markdown("**Subgroups**")
        st.caption("More specific labels such as US equities, Treasury bonds, gold, or sector equity.")

        st.markdown("**Market-data panel**")
        st.caption("Prepared historical returns and features used by the Strategy Engine.")

        st.markdown("**Funding bridge**")
        st.caption("The contribution inherited from Personal Finance Setup for later simulations.")

        st.caption("These labels describe the basket composition; they are not investment recommendations.")


def _render_step4_sidebar(step: int) -> None:
    """Render a focused sidebar for Risk Profile & Asset Universe."""
    _render_step4_navigation_block()

    st.divider()
    _render_step4_current_universe()

    st.divider()
    _render_step4_funding_bridge()

    st.divider()
    _render_step4_market_panel_status()

    st.divider()
    _render_step4_q_and_a()
    _render_step4_help()
    _render_step4_market_data_tools_toggle()



# ---------------------------------------------------------------------------
# Step 5 focused sidebar
# ---------------------------------------------------------------------------


def _step5_setup_summary() -> dict[str, Any]:
    """Read Strategy Engine setup state without running the engine."""
    philosophy = str(st.session_state.get("investment_philosophy", "Balanced") or "Balanced")
    template = str(st.session_state.get("step5_template", "Balanced Risk-Controlled") or "Balanced Risk-Controlled")
    style = str(st.session_state.get("step5_style", "Balanced") or "Balanced")
    panel_meta = _asset_panel_summary()
    cfg = _coerce_mapping(st.session_state.get("last_engine_config", {}))
    if not cfg:
        cfg = _coerce_mapping(_latest_run_result().get("config", {}))
    return {
        "philosophy": philosophy,
        "template": template,
        "style": style,
        "panel_assets": _safe_int(panel_meta.get("assets", 0), 0),
        "panel_rows": _safe_int(panel_meta.get("rows", 0), 0),
        "top_k": cfg.get("top_k", st.session_state.get("step5_basic_top_k", "—")),
        "signal_mode": cfg.get("signal_mode", st.session_state.get("step5_basic_signal_mode", "—")),
    }


def _render_step5_navigation_block() -> None:
    st.markdown("### Navigation")
    if _native_button(
        "Home",
        key="global_sidebar_home_button_step5",
        use_container_width=True,
        disabled=False,
        help="Return to the module selector.",
        icon=":material/home:",
    ):
        _go_to_step(0)

    st.markdown("**Investment Strategy Lab**")
    st.caption("Run the selected universe through the Strategy Engine and review tested results.")

    # Match the Scenario & Reports sidebar pattern: branch items are passive
    # status labels, not navigation buttons. Home remains the only action here.
    _render_branch_status_item(_step_nav_item(4), 5)
    _render_branch_status_item(_step_nav_item(5), 5)


def _render_step5_engine_setup() -> None:
    setup = _step5_setup_summary()
    st.markdown("### Engine setup")
    st.caption(f"Template: **{setup['template']}**")
    st.caption(f"Style: **{setup['style']}**")
    st.caption(f"Risk profile: **{setup['philosophy']}**")
    assets = _safe_int(setup.get("panel_assets", 0), 0)
    rows = _safe_int(setup.get("panel_rows", 0), 0)
    if assets > 0 and rows > 0:
        st.caption(f"Universe/data: **{_format_count(assets)} panel assets · {_format_count(rows)} rows**")
    else:
        st.caption("Universe/data: pending market-data panel")



def _render_step5_funding_bridge() -> None:
    bridge = _step4_funding_bridge_summary()
    st.markdown("### Funding bridge")
    monthly = _safe_float(bridge.get("monthly", 0.0), 0.0)
    weekly = _safe_float(bridge.get("weekly", 0.0), 0.0)
    if monthly > 0.0 or weekly > 0.0:
        st.caption(f"Monthly contribution: **{_money_plain(monthly)}/mo**")
        st.caption(f"Weekly equivalent: **{_money_weekly(weekly)}**")
        st.caption(f"Source: {bridge.get('source', 'Personal Finance Setup')}")
    else:
        st.caption("No contribution bridge found yet. The demo can still run, but Personal Finance gives the cleaner path.")


def _render_step5_run_status() -> None:
    st.markdown("### Run status")
    run_map = _latest_run_result()
    if not run_map:
        st.warning("Strategy Engine not run yet.")
        st.caption("Use the main screen button to run the portfolio test.")
        return

    perf = _coerce_mapping(run_map.get("performance_summary", {}))
    st.success("Latest result stored.")
    st.caption(f"CAGR: **{_pct(perf.get('cagr', 0.0))}**")
    st.caption(f"Volatility: **{_pct(perf.get('annual_volatility', perf.get('volatility', 0.0)))}**")
    st.caption(f"Max drawdown: **-{100.0 * abs(_safe_float(perf.get('max_drawdown', 0.0), 0.0)):.2f}%**")
    st.caption(f"Sharpe: **{_safe_float(perf.get('sharpe', 0.0), 0.0):.2f}**")
    periods = _safe_int(perf.get("periods", 0), 0)
    if periods > 0:
        st.caption(f"Test periods: **{periods}**")



def _render_step5_q_and_a() -> None:
    with st.expander("Strategy Engine Q&A", expanded=False):
        st.markdown("**Does this guarantee future returns?**")
        st.caption(
            "No. It is a historical strategy test. It helps compare setups inside the app, but future markets can behave differently."
        )

        st.markdown("**Why is the engine needed?**")
        st.caption(
            "Step 4 gives the app a universe of possible assets. The engine turns that universe into an actual tested portfolio "
            "by choosing weights through time. The headline metrics are calculated after that return path exists."
        )

        st.markdown("**What can the engine improve?**")
        st.caption(
            "It can improve diversification, drawdown control, dynamic asset selection, and fit with the chosen risk profile."
        )

        st.markdown("**What can the engine worsen?**")
        st.caption(
            "It can lag simple benchmarks in strong bull markets, reduce upside through risk controls, overfit weak signals, "
            "or add turnover and parameter sensitivity."
        )

        st.markdown("**What does CAGR mean here?**")
        st.caption(
            "CAGR is the annualised growth rate of the tested strategy over the evaluated historical window. "
            "It summarises long-run growth, but it is not a promised future return."
        )

        st.markdown("**What does volatility mean here?**")
        st.caption(
            "Volatility is the bumpiness of the tested return path. Higher volatility usually means the journey feels less stable, "
            "even when the long-term return looks attractive."
        )

        st.markdown("**What does Sharpe mean here?**")
        st.caption(
            "Sharpe compares return with volatility. A higher Sharpe usually means the strategy was paid better for the risk it took, "
            "but it still depends on the tested period and assumptions."
        )

        st.markdown("**What does max drawdown mean?**")
        st.caption(
            "Max drawdown is the worst historical fall from a previous high to a later low. It is the main pain-test metric: "
            "it shows how much discomfort the user would have needed to tolerate before recovery."
        )

        st.markdown("**Why can a good CAGR still feel uncomfortable?**")
        st.caption(
            "Because CAGR describes the whole-period average, while volatility and drawdown describe the journey. "
            "A strategy can finish well and still have difficult periods along the way."
        )

        st.markdown("**Why run this before projection?**")
        st.caption(
            "The projection module needs a return path or a fallback assumption. A real Step 5 run gives it a tested strategy path "
            "instead of a generic educational proxy."
        )

def _render_step5_terms() -> None:
    with st.expander("Strategy Engine terms", expanded=False):
        st.markdown("**Strategy template**")
        st.caption("The broad engine method used to build the portfolio.")

        st.markdown("**Style preset**")
        st.caption("The risk posture applied to the selected template.")

        st.markdown("**Out-of-sample path**")
        st.caption("The tested return series after model decisions; closer to a backtest than an in-sample fit.")

        st.caption("Metric explanations are in the Strategy Engine Q&A above. These terms are not predictions or investment advice.")

def _render_step5_diagnostics() -> None:
    """Render the single collapsed Step 5 diagnostics expander after a real run."""
    run_map = _latest_run_result()
    if not run_map:
        return

    # Keep this as the only Step 5 sidebar diagnostics expander. It is rendered
    # only after a real run, and it is always requested collapsed by default.
    # post_run.py owns the full main-screen diagnostics panel.
    with st.expander("Diagnostics", expanded=False):
        perf = _coerce_mapping(run_map.get("performance_summary", {}))
        risk = _coerce_mapping(run_map.get("risk_summary", {}))
        panel_meta = _asset_panel_summary()
        investment_context = _investment_context()
        oos_returns = investment_context.get("oos_returns_monthly", [])

        periods = _safe_int(perf.get("periods", 0), 0)
        if periods <= 0 and isinstance(oos_returns, list):
            periods = len(oos_returns)

        assets = _safe_int(panel_meta.get("assets", run_map.get("asset_panel_n_assets", 0)), 0)
        reference_assets = _safe_int(
            risk.get("reference_assets", run_map.get("reference_assets", 0)),
            0,
        )

        benchmark_passed = bool(
            run_map.get("benchmark_check_passed", False)
            or run_map.get("benchmark_passed", False)
            or _safe_float(perf.get("sharpe", 0.0), 0.0) > 0.0
        )

        sharpe = _safe_float(perf.get("sharpe", 0.0), 0.0)
        max_dd = abs(_safe_float(perf.get("max_drawdown", 0.0), 0.0))
        if sharpe >= 0.75 and max_dd <= 0.25:
            confidence = "Moderate-to-Strong"
        elif sharpe >= 0.35:
            confidence = "Moderate"
        else:
            confidence = "Low-to-Moderate"

        st.markdown("**Reliability snapshot**")
        st.success(f"Confidence: {confidence}.")
        st.caption(f"**Benchmark check:** {'Passed' if benchmark_passed else 'Needs review'}")
        if reference_assets > 0:
            st.caption(f"**Reference assets:** {_format_count(reference_assets)}")
        elif assets > 0:
            st.caption(f"**Panel assets:** {_format_count(assets)}")
        if periods > 0:
            st.caption(f"**OOS months:** {_format_count(periods)}")

        window_start = str(perf.get("start", run_map.get("window_start", "")) or "").strip()
        window_end = str(perf.get("end", run_map.get("window_end", "")) or "").strip()
        if window_start or window_end:
            st.caption(f"Window: {window_start or '—'} → {window_end or '—'}")

        engine_timing = _coerce_mapping(run_map.get("engine_timing", {}))
        suggestion_rows = _suggestion_timing_rows()
        timings_visible = bool(st.session_state.get(STEP5_RUN_DIAGNOSTICS_EXPANDED_KEY, False))

        st.divider()
        st.markdown("**Timing snapshot**")
        st.caption("Technical timings and run metadata stay hidden from the main page unless opened.")

        total_engine = _safe_float(engine_timing.get("total_engine", 0.0), 0.0)
        walk_forward = _safe_float(engine_timing.get("walk_forward_loop", 0.0), 0.0)
        probabilistic = _safe_float(engine_timing.get("probabilistic_total", 0.0), 0.0)
        n_oos_dates = _safe_int(engine_timing.get("n_oos_dates", periods), periods)

        if total_engine > 0.0:
            st.caption(f"**Engine run:** {_format_seconds(total_engine)}")
        else:
            st.caption("**Engine run:** timing not available in the stored result")
        if walk_forward > 0.0:
            st.caption(f"**Walk-forward:** {_format_seconds(walk_forward)}")
        if probabilistic > 0.0:
            st.caption(f"**Probabilistic:** {_format_seconds(probabilistic)}")
        if n_oos_dates > 0:
            st.caption(f"**OOS months timed:** {_format_count(n_oos_dates)}")

        panel_rows = _safe_int(run_map.get("asset_panel_n_rows", panel_meta.get("rows", 0)), 0)
        panel_assets = _safe_int(run_map.get("asset_panel_n_assets", panel_meta.get("assets", 0)), 0)
        source = str(run_map.get("source", "micro_pipeline_real") or "micro_pipeline_real")
        if panel_rows > 0 or panel_assets > 0:
            st.caption(f"Panel: {panel_assets} assets · {_format_count(panel_rows)} rows")
        st.caption(f"Source: {source}")

        if suggestion_rows:
            total_suggestion_seconds = sum(max(0.0, row[1]) for row in suggestion_rows)
            st.caption(f"**Suggestion overhead:** {_format_seconds(total_suggestion_seconds)}")
            for label, seconds, candidates, accepted in suggestion_rows:
                suffix = ""
                if candidates > 0:
                    suffix = f" · {candidates} tested"
                    if accepted > 0:
                        suffix += f" · {accepted} passed"
                st.caption(f"{label}: {_format_seconds(seconds)}{suffix}")

        timing_button_label = "Hide run timings" if timings_visible else "Show run timings"
        st.button(
            timing_button_label,
            key="global_sidebar_step5_toggle_run_timings",
            use_container_width=True,
            help="Show or hide the detailed Run timings and diagnostics panel on the main Strategy Engine screen.",
            on_click=_toggle_step5_run_timings,
        )

        st.divider()
        st.caption("Full validation details stay in the main post-run panel.")

        improvement_flow_completed = _step5_improvement_flow_completed()
        result_is_current = _step5_result_is_current_for_sidebar()
        reliability_visible = bool(st.session_state.get(STEP5_RELIABILITY_EXPANDED_KEY, False))

        if (not improvement_flow_completed or not result_is_current) and reliability_visible:
            # Do not allow the robustness panel to stay open for an intermediate
            # or stale setup. Accepted/skipped suggestions can change the active
            # engine inputs, so the updated setup must be run before reliability
            # checks are meaningful.
            reliability_visible = False
            st.session_state[STEP5_RELIABILITY_EXPANDED_KEY] = False
            st.session_state[STEP5_SCROLL_TO_RELIABILITY_KEY] = False

        if improvement_flow_completed and result_is_current:
            st.caption("Optional improvement checks completed. Reliability and robustness can now be opened from here.")
        elif improvement_flow_completed and not result_is_current:
            st.info(
                "Optional improvement checks are completed, but the current Strategy Engine inputs differ from the last executed run. "
                "Run the updated setup first, then open reliability and robustness."
            )
        else:
            st.info(
                "Reliability and robustness are best run after the optional improvement checks are completed, "
                "because accepted suggestions can still change the setup being tested."
            )

        reliability_button_disabled = bool(not improvement_flow_completed or not result_is_current)
        reliability_button_label = "Hide reliability & robustness" if reliability_visible else "Show reliability & robustness"
        st.button(
            reliability_button_label,
            key="global_sidebar_step5_toggle_reliability_and_robustness",
            use_container_width=True,
            disabled=reliability_button_disabled,
            help=(
                "Complete or skip all optional improvement checks before opening reliability and robustness."
                if not improvement_flow_completed
                else (
                    "Run the updated Strategy Engine setup first; reliability checks must match the latest executed result."
                    if not result_is_current
                    else "Show or hide the reliability and robustness panel for the final selected setup."
                )
            ),
            on_click=_toggle_step5_reliability,
        )


def _render_step5_sidebar(step: int) -> None:
    """Render a focused sidebar for the Strategy Engine module.

    Before a portfolio run, the sidebar prioritises setup context. After a
    successful run, it prioritises the stored result and headline metrics.
    """
    _render_step5_navigation_block()

    if _has_step5_result():
        st.divider()
        _render_step5_run_status()

        st.divider()
        _render_step5_engine_setup()

        st.divider()
        _render_step5_funding_bridge()
    else:
        st.divider()
        _render_step5_engine_setup()

        st.divider()
        _render_step5_funding_bridge()

        st.divider()
        _render_step5_run_status()

    st.divider()
    _render_step5_q_and_a()
    _render_step5_terms()

    if _has_step5_result():
        _render_step5_diagnostics()


def _render_step6_scenario_status() -> None:
    """Focused Long-Term Scenario sidebar status.

    Long-Term Scenario already contains the scenario controls and comparison chart on the
    main page. The sidebar should summarise state without repeating every
    diagnostic panel from the investment engine.
    """
    st.markdown("### Scenario status")
    st.caption(
        "Long-Term Scenario uses the current contribution plan plus either a tested Strategy Engine path "
        "or an educational proxy/savings-only assumption."
    )

    investment_context = _investment_context()
    snapshot = _planning_snapshot()
    run_map = _latest_run_result()

    weekly = _first_positive_value(
        investment_context.get("weekly_equivalent"),
        investment_context.get("weekly_contribution"),
        investment_context.get("target_weekly"),
        snapshot.get("target_a_weekly") if snapshot else 0.0,
        snapshot.get("target_weekly") if snapshot else 0.0,
        st.session_state.get("target_a_weekly"),
        st.session_state.get("weekly_savings_target"),
    )
    monthly = _first_positive_value(
        investment_context.get("monthly_contribution"),
        investment_context.get("target_a_monthly"),
        snapshot.get("target_a_monthly") if snapshot else 0.0,
        snapshot.get("target_monthly") if snapshot else 0.0,
        weekly * 52.0 / 12.0 if weekly > 0.0 else 0.0,
    )

    if monthly > 0.0:
        st.caption(f"**Contribution:** {_money_plain(monthly)}/mo")
    else:
        st.caption("**Contribution:** resolved on the main scenario screen")

    if weekly > 0.0:
        st.caption(f"**Weekly equivalent:** {_money_weekly(weekly)}")

    philosophy = str(st.session_state.get("investment_philosophy", "Balanced") or "Balanced")
    st.caption(f"**Projection profile:** {philosophy}")

    if run_map:
        perf = _coerce_mapping(run_map.get("performance_summary", {}))
        st.caption("**Return path:** Historical Strategy Engine path")
        st.caption(
            "**Last engine run:** "
            f"CAGR {_pct(perf.get('cagr', 0.0))} · "
            f"Sharpe {_safe_float(perf.get('sharpe', 0.0), 0.0):.2f}"
        )
    else:
        st.caption(f"**Return path:** Demo {philosophy} proxy")
        st.caption("Strategy Engine has not been run yet, so the investment path is labelled as an educational proxy.")

    st.caption(
        "**Projection:** generated on this screen"
        if _has_projection_result()
        else "**Projection:** adjust the scenario on the main screen"
    )



def _render_step6_projection_diagnostics() -> None:
    with st.expander("Projection context", expanded=False):
        panel_meta = _asset_panel_summary()
        if panel_meta:
            st.markdown("**Market-data panel**")
            st.caption(
                f"{panel_meta.get('assets', 0)} assets · {panel_meta.get('rows', 0)} rows · "
                f"{panel_meta.get('source', 'Step 4 panel')}"
            )

        run_map = _latest_run_result()
        if run_map:
            st.divider()
            st.markdown("**Last Strategy Engine run**")
            perf = _coerce_mapping(run_map.get("performance_summary", {}))
            st.caption(
                f"CAGR {_pct(perf.get('cagr', 0.0))} · "
                f"Vol {_pct(perf.get('annual_volatility', perf.get('volatility', 0.0)))} · "
                f"MaxDD -{100.0 * abs(_safe_float(perf.get('max_drawdown', 0.0), 0.0)):.2f}% · "
                f"Sharpe {_safe_float(perf.get('sharpe', 0.0), 0.0):.2f}"
            )

        investment_context = _investment_context()
        oos_returns = investment_context.get("oos_returns_monthly", [])
        if isinstance(oos_returns, list) and oos_returns:
            st.divider()
            st.markdown("**Projection bridge**")
            st.caption(f"{len(oos_returns)} monthly OOS returns available for long-term scenarios.")

        if not panel_meta and not run_map:
            st.caption("Projection context will populate after Personal Finance, Risk Profile & Asset Universe, or Strategy Engine has produced inputs.")


def _render_step6_q_and_a() -> None:
    with st.expander("Scenario Q&A", expanded=False):
        st.markdown("**What is this screen for?**")
        st.caption(
            "It translates a contribution plan and return-path assumption into possible long-term wealth ranges. "
            "It is a scenario explorer, not a forecast."
        )

        st.markdown("**Why is this useful?**")
        st.caption("1. It translates return behaviour into personal contribution outcomes.")
        st.caption("2. It compares investment/proxy paths with a savings-only baseline.")
        st.caption("3. It shows uncertainty through P10 / median / P90 ranges instead of one promised number.")

        st.markdown("**How is this different from the Strategy Engine?**")
        st.caption(
            "The Strategy Engine asks how a selected strategy behaved on the historical market-data panel. "
            "Long-Term Scenario asks what range of contribution outcomes could occur if future returns behaved similarly."
        )

        st.markdown("**Why can this run before the Strategy Engine?**")
        st.caption(
            "Before a tested engine path exists, the investment view can use a clearly labelled educational proxy. "
            "Once the Strategy Engine has run, the scenario can use the tested return path instead."
        )


def _render_step6_terms() -> None:
    with st.expander("Scenario terms", expanded=False):
        st.markdown("**Starting investment pot**")
        st.caption(
            "The amount already available at the start of the Long-Term Scenario. It is separate from future monthly contributions."
        )

        st.markdown("**Optional wealth goal**")
        st.caption(
            "A target amount used to calculate goal probability. If the goal is left at £0, the projection can still run, "
            "but goal probability is not meaningful."
        )

        st.markdown("**Compare across horizons**")
        st.caption(
            "Runs the same contribution and return-path assumptions across several time horizons, such as 20, 30 and 50 years. "
            "It compares time length, not separate market forecasts."
        )

        st.markdown("**Horizon comparison table**")
        st.caption(
            "A table that compares terminal outcomes at each selected horizon, including median wealth, savings-only baseline and uncertainty ranges."
        )

        st.markdown("**P10 / median / P90**")
        st.caption(
            "Scenario range markers: P10 is a lower outcome, median is the middle outcome, and P90 is a higher outcome. "
            "They are not promised results."
        )

        st.markdown("**Monte Carlo paths**")
        st.caption(
            "The number of simulated paths used to build the scenario range. More paths can make the range smoother, "
            "but they do not make the future more predictable."
        )

        st.markdown("**Projection path mode**")
        st.caption(
            "Controls how the scenario path is generated. A monthly path uses monthly returns directly; a hybrid daily path expands each sampled month "
            "into a synthetic within-month journey."
        )

        st.markdown("**Hybrid daily simulation**")
        st.caption(
            "The scenario still samples monthly Strategy Engine returns, but each sampled month is expanded into synthetic daily steps that compound back "
            "to the same monthly return."
        )

        st.markdown("**Synthetic trading days per month**")
        st.caption(
            "The number of artificial daily steps used inside each simulated month when hybrid daily simulation is selected."
        )

        st.markdown("**Daily path variation**")
        st.caption(
            "Controls how much visible movement happens inside each simulated month. It changes the shape of the path, not the sampled monthly return itself."
        )

        st.markdown("**Return path**")
        st.caption(
            "The return source used by Long-Term Scenario. After Strategy Engine has run, this is normally the historical Strategy Engine OOS return series. "
            "Before that, the app can use a clearly labelled educational proxy."
        )

        st.markdown("**Savings-only baseline**")
        st.caption(
            "The same starting pot and contribution path with no investment return, volatility, drawdown or market risk."
        )

        st.markdown("**Educational investment proxy**")
        st.caption(
            "A labelled assumption used when the Strategy Engine has not produced a tested return path yet. "
            "It keeps the scenario screen usable without pretending to be a real backtest."
        )

        st.markdown("**Loss vs contributions**")
        st.caption(
            "The share of simulations where the terminal value ends below the total amount contributed. "
            "It is a scenario diagnostic, not a prediction."
        )

        st.markdown("**Goal probability**")
        st.caption(
            "The share of simulations that reach the optional wealth goal. Treat it as a secondary diagnostic unless the goal is meaningful."
        )

        st.markdown("**Model scope**")
        st.caption(
            "This prototype can adjust discretionary contributions automatically; essential spending is treated as fixed in the model."
        )

        st.markdown("**Contribution allocation**")
        st.caption(
            "Long-Term Scenario works at strategy-return level. It does not yet show month-by-month asset-level purchase percentages."
        )


def _render_step6_sidebar(step: int) -> None:
    """Render a quieter sidebar for Long-Term Scenario."""
    _render_step_navigation(step)
    st.divider()

    _render_step6_scenario_status()

    _render_step6_projection_diagnostics()
    _render_step6_q_and_a()
    _render_step6_terms()


def _projection_result_payload() -> dict:
    """Read the current Long-Term Scenario projection result without running work."""
    for key in (
        "investment_projection_result",
        "step6_projection_result",
        "step6_last_projection_result",
        "long_term_projection_result",
        "projection_result",
    ):
        payload = _coerce_mapping(st.session_state.get(key, {}))
        if payload:
            return payload
    return {}


def _projection_summary_from_payload(payload: dict) -> dict:
    payload = _coerce_mapping(payload)
    summary = _coerce_mapping(payload.get("summary", {}))
    if summary:
        return summary
    result = _coerce_mapping(payload.get("result", {}))
    result_summary = _coerce_mapping(result.get("summary", {}))
    if result_summary:
        return result_summary
    return {}


def _step7_compare_payload() -> dict:
    for key in ("step6_compare_branch_result", "investment_projection_compare_results"):
        payload = _coerce_mapping(st.session_state.get(key, {}))
        if payload:
            return payload
    return {}


def _step7_report_is_complete() -> bool:
    compare_payload = _step7_compare_payload()
    if compare_payload:
        rows = compare_payload.get("rows")
        if isinstance(rows, list):
            return len(rows) > 0
        return True
    return bool(_projection_result_payload() or _has_projection_result())


def _step7_horizon_label() -> str:
    projection_summary = _projection_summary_from_payload(_projection_result_payload())
    current_horizon = _safe_int(
        st.session_state.get("investment_projection_horizon_years", projection_summary.get("horizon_years", 0)),
        0,
    )
    compare_horizons = st.session_state.get("investment_projection_compare_horizons", [])
    if not isinstance(compare_horizons, list):
        compare_horizons = []

    labels: list[str] = []
    if current_horizon > 0:
        labels.append(f"current {current_horizon}y")
    for horizon in compare_horizons:
        try:
            horizon_int = int(horizon)
        except Exception:
            continue
        if horizon_int > 0 and horizon_int != current_horizon:
            labels.append(f"{horizon_int}y")

    deduped = list(dict.fromkeys(labels))
    return " + ".join(deduped) if deduped else "available from Long-Term Scenario"


def _step7_return_path_label() -> str:
    ctx = _investment_context()
    explicit = str(ctx.get("projection_return_source", "") or "").strip()
    if explicit:
        return explicit
    philosophy = str(st.session_state.get("investment_philosophy", "Balanced") or "Balanced")
    if _latest_run_result():
        return "Historical Strategy Engine path"
    return f"Demo {philosophy} proxy"


def _render_step7_current_report() -> None:
    """Focused Final Report status for the sidebar."""
    st.markdown("### Current report")

    report_complete = _step7_report_is_complete()
    if report_complete:
        st.success("Final report available.")
    else:
        st.warning("Generate the Long-Term Scenario first.")
    st.caption(
        "Summarises the selected plan, return path, and Long-Term Scenario comparison for decision support."
    )

    bridge = _step4_funding_bridge_summary()
    monthly = _safe_float(bridge.get("monthly", 0.0), 0.0)
    weekly = _safe_float(bridge.get("weekly", 0.0), 0.0)

    st.caption("**Source:** Long-Term Scenario")
    st.caption(f"**Return path:** {_step7_return_path_label()}")
    if monthly > 0.0:
        st.caption(f"**Monthly contribution:** {_money_plain(monthly)}/mo")
    if weekly > 0.0:
        st.caption(f"**Weekly equivalent:** {_money_weekly(weekly)}")
    st.caption(f"**Horizon view:** {_step7_horizon_label()}")


def _render_step7_readiness() -> None:
    with st.expander("Report inputs", expanded=False):
        snapshot = _planning_snapshot()
        finance_summary = _personal_finance_sidebar_summary()
        finance_ready = bool(snapshot) or bool(finance_summary.get("has_any_values", False))
        panel_ready = _has_asset_panel()
        run_ready = _has_step5_result()
        projection_ready = _step7_report_is_complete()

        _readiness_line(
            "Personal finance",
            finance_ready,
            "contribution bridge available" if finance_ready else "set up finance plan",
        )
        _readiness_line(
            "Market-data panel",
            panel_ready,
            "available" if panel_ready else "optional if using proxy/savings-only report",
        )
        _readiness_line(
            "Strategy/proxy path",
            True,
            "tested engine path" if run_ready else "educational proxy available",
        )
        _readiness_line(
            "Long-Term Scenario",
            projection_ready,
            "completed" if projection_ready else "generate scenario first",
        )
        _readiness_line(
            "Final report",
            projection_ready,
            "available" if projection_ready else "pending scenario output",
        )
        st.caption("This sidebar reads app state only. It does not run projections, refresh data, or launch diagnostics.")


def _render_step7_shortcuts() -> None:
    st.markdown("### Report shortcuts")

    step6_enabled, step6_reason = _step_access_state(6, 7)
    if st.button(
        "← Back to Long-Term Scenario",
        key="global_sidebar_step7_back_to_scenario",
        use_container_width=True,
        disabled=not step6_enabled,
        help=step6_reason,
    ):
        _go_to_step(6)

    step5_enabled, step5_reason = _step_access_state(5, 7)
    if st.button(
        "Open Strategy Engine",
        key="global_sidebar_step7_open_strategy_engine",
        use_container_width=True,
        disabled=not step5_enabled,
        help=step5_reason,
    ):
        _go_to_step(5)

    if st.button(
        "Return Home",
        key="global_sidebar_step7_return_home",
        use_container_width=True,
        help="Return to the module selector.",
    ):
        _go_to_step(0)

    st.caption("Final-report actions are navigation only; heavy reruns stay on their main screens.")


def _render_step7_report_diagnostics() -> None:
    with st.expander("Technical diagnostics", expanded=False):
        st.markdown("**Projection source**")
        st.caption(f"return_path={_step7_return_path_label()}")
        st.caption(f"horizons={_step7_horizon_label()}")

        compare_payload = _step7_compare_payload()
        if compare_payload:
            rows = compare_payload.get("rows")
            if isinstance(rows, list):
                st.caption(f"comparison_rows={len(rows)}")
            else:
                st.caption("comparison_payload=available")

        panel_meta = _asset_panel_summary()
        if panel_meta:
            st.divider()
            st.markdown("**Selected market-data panel**")
            st.caption(
                f"source={panel_meta.get('source', 'selected market-data panel')} · "
                f"assets={panel_meta.get('assets', 0)} · rows={panel_meta.get('rows', 0)}"
            )

        run_map = _latest_run_result()
        technical_ids: list[str] = []
        if run_map:
            st.divider()
            st.markdown("**Last Strategy Engine run**")
            perf = _coerce_mapping(run_map.get("performance_summary", {}))
            st.caption(
                f"CAGR={_pct(perf.get('cagr', 0.0))} · "
                f"Sharpe={_safe_float(perf.get('sharpe', 0.0), 0.0):.2f} · "
                f"MaxDD=-{100.0 * abs(_safe_float(perf.get('max_drawdown', 0.0), 0.0)):.2f}%"
            )
            run_sig = str(run_map.get("run_signature", "") or "")
            cfg_fp = str(run_map.get("config_fingerprint", "") or "")
            if run_sig:
                technical_ids.append(f"run_signature={run_sig}")
            if cfg_fp:
                technical_ids.append(f"config_fp={cfg_fp}")

        if technical_ids:
            st.divider()
            st.markdown("**Technical IDs**")
            for item in technical_ids:
                st.caption(item)

        if not compare_payload and not panel_meta and not run_map:
            st.caption("Diagnostics populate after the scenario, market panel, or engine result exists.")


def _render_step7_report_help() -> None:
    with st.expander("Report Q&A & terms", expanded=False):
        st.markdown("**What is this report for?**")
        st.caption(
            "It summarises the selected plan, tested/proxy return path, and Long-Term Scenario comparison."
        )

        st.markdown("**Savings-only baseline**")
        st.caption("Same contribution path, but no investment return, volatility, drawdown or market risk.")

        st.markdown("**Educational proxy**")
        st.caption("A labelled fallback used before a tested Strategy Engine path exists. It is not backtest evidence.")

        st.markdown("**Tested Strategy Engine path**")
        st.caption("A historical out-of-sample return path generated from the selected universe and engine configuration.")

        st.markdown("**Scenario ranges**")
        st.caption("P10 / median / P90 are modelled ranges under assumptions, not promised future outcomes.")

        st.markdown("**Goal probability and loss vs contributions**")
        st.caption("Goal probability checks whether paths reach the selected goal; loss vs contributions checks whether terminal value falls below total contributions.")


def _render_step7_sidebar(step: int) -> None:
    """Render a report-focused sidebar for Final Report."""
    _render_step_navigation(step)
    st.divider()

    _render_step7_current_report()

    st.divider()
    _render_step7_readiness()
    _render_step7_report_diagnostics()
    _render_step7_report_help()


def render_global_sidebar() -> None:


    """Render a persistent contextual sidebar across the whole Streamlit app.

    Call this once from app.py after the active screen has rendered. Rendering the
    main screen first lets the sidebar read any state seeded or updated by that
    screen during the current rerun.

    Do not call this from individual step files, otherwise the sidebar can be
    duplicated.
    """

    step = _current_step()

    with st.sidebar:
        st.markdown("## LifeBudget Micro")
        st.caption("Educational prototype · Plan. Test. Explore.")

        if int(step) == 0:
            _render_home_sidebar_minimal()
            return

        if int(step) in {1, 2, 3}:
            _render_personal_finance_sidebar(step)
            return

        if int(step) == 4:
            _render_step4_sidebar(step)
            return

        if int(step) == 5:
            _render_step5_sidebar(step)
            return

        if int(step) == 6:
            _render_step6_sidebar(step)
            return

        if int(step) == 7:
            _render_step7_sidebar(step)
            return

        _render_step_navigation(step)
        st.divider()

        _render_compact_status(step)

        with st.expander("Current context", expanded=False):
            _render_current_context(step)

        _render_readiness_status(step)
        _render_tools_and_checks(step)
        _render_audit_diagnostics(step)
        _render_help(step)
