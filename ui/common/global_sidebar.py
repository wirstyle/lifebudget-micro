"""Global sidebar for LifeBudget Micro.

This sidebar is deliberately *not* a step navigator. The main app keeps the
wizard-style Back/Continue flow. The sidebar acts as a persistent inspector:
current context, available checks, lightweight audit diagnostics, and quick help.

Keep this module read-only with respect to heavy computation. It should only
read from st.session_state and expose safe navigation such as returning Home.
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
STEP0_NOTICE_CHECKBOX = "step0_educational_notice_checkbox_v1"
STEP0_NOTICE_ACCEPTED_ONCE = "educational_notice_accepted_once_v1"
STEP0_NOTICE_ALIASES = (
    STEP0_NOTICE_ACCEPTED,
    STEP0_NOTICE_ACCEPTED_ONCE,
    "step0_educational_notice_acknowledged",
    "step0_notice_accepted",
    STEP0_NOTICE_CHECKBOX,
)
STEP0_SELECTED_MODULE = "step0_selected_module"
STEP0_PATHWAY = "step0_planning_pathway"

# Timing keys used by Step 5 suggestion modules. They are strings here on purpose
# so the sidebar does not need to import all recommendation modules just to render.
PRESET_SUGGESTION_TIMING_KEY = "step5_preset_suggestion_timing_v1"
AUTO_OPT_SUGGESTION_TIMING_KEY = "step5_auto_opt_suggestion_timing_v1"
UNIVERSE_SUGGESTION_TIMING_KEY = "step5_universe_suggestion_timing_v1"
SIZE_SUGGESTION_TIMING_KEY = "step5_size_suggestion_timing_v1"


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


def _module_for_step(step: int) -> tuple[str, str]:
    if step == 0:
        return "Home", "Choose starting module"
    if step in {1, 2, 3}:
        return "Personal Finance Planner", "Budget, savings target, feasibility"
    if step in {4, 5}:
        return "Investment Strategy Lab", "Universe setup and engine testing"
    if step == 6:
        return "Long-Term Scenario Explorer", "Scenario simulation, not forecast"
    if step == 7:
        return "Insights Summary", "Decision-support interpretation"
    return "LifeBudget Micro", "Educational planning prototype"


def _selected_module_label() -> str:
    module_id = str(st.session_state.get(STEP0_SELECTED_MODULE, "") or "")
    return {
        "personal_finance": "Personal Finance Planner",
        "investment_lab": "Investment Strategy Lab",
        "scenario_explorer": "Long-Term Scenario Explorer",
    }.get(module_id, "Full planning flow")


def _selected_pathway_label() -> str:
    pathway = str(st.session_state.get(STEP0_PATHWAY, "") or "")
    return {
        "compare_both": "Compare savings and investing",
        "savings_only": "Savings-only pathway",
        "savings_plus_investing": "Savings + investing pathway",
    }.get(pathway, "Not selected yet")


def _educational_notice_is_accepted(step: int) -> bool:
    """Return a robust educational-notice status for the shared sidebar.

    The durable acceptance state is intentionally separated from the Step 0
    checkbox widget. This prevents false "pending" states when the user returns
    to Home or when the sidebar renders before Step 0 in app.py.
    """
    accepted = any(bool(st.session_state.get(key, False)) for key in STEP0_NOTICE_ALIASES)

    if accepted:
        # Normalize all aliases so old/new Step 0 versions and the sidebar agree.
        for key in STEP0_NOTICE_ALIASES:
            st.session_state[key] = True
        return True

    if int(step) > 0:
        # If the user is beyond Home, they passed the notice gate. Persist that
        # state so returning to Home does not incorrectly show "pending".
        for key in STEP0_NOTICE_ALIASES:
            st.session_state[key] = True
        return True

    return False


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
        return {"rows": rows, "assets": assets, "source": str(st.session_state.get("asset_panel_source_label", "Step 4 panel") or "Step 4 panel")}

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


def _timing_total(key: str) -> float:
    payload = _coerce_mapping(st.session_state.get(key, {}))
    return _safe_float(payload.get("total_seconds", 0.0), 0.0)


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
        st.success("Educational notice accepted", icon="✅")
    else:
        st.warning("Educational notice pending", icon="⚠️")

    if step >= 4:
        panel_meta = _asset_panel_summary()
        if panel_meta:
            st.caption(
                f"Data: {panel_meta.get('source', 'Step 4 panel')} · "
                f"assets={panel_meta.get('assets', 0)} · rows={panel_meta.get('rows', 0)}"
            )
        else:
            st.caption("Data: cached deployment panel prepared in Step 4")


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


def _render_audit_diagnostics(step: int) -> None:
    with st.expander("Audit / diagnostics", expanded=False):
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
            st.markdown("**Last Step 5 run**")
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

        investment_context = _coerce_mapping(st.session_state.get("investment_context", {}))
        oos_returns = investment_context.get("oos_returns_monthly", [])
        if isinstance(oos_returns, list) and oos_returns:
            st.divider()
            st.markdown("**Projection bridge**")
            st.caption(f"{len(oos_returns)} monthly OOS returns available for Step 6.")

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


def _render_demo_controls() -> None:
    st.markdown("### Demo controls")
    if st.button("Back to Home", key="global_sidebar_back_home", use_container_width=True):
        st.session_state[CURRENT_STEP] = 0
        st.session_state["current_step"] = 0
        st.rerun()

    st.caption("Use the main screen's Back/Continue buttons for step-by-step navigation.")


def render_global_sidebar() -> None:
    """Render a persistent sidebar across the whole Streamlit app.

    Call this once from app.py, after st.set_page_config and before rendering the
    active step. Do not call it from individual step files, otherwise the sidebar
    can be duplicated.
    """
    step = _current_step()

    with st.sidebar:
        st.markdown("## LifeBudget Micro")
        st.caption("Educational prototype · Plan. Test. Explore.")
        st.divider()

        _render_current_context(step)
        st.divider()

        _render_available_checks(step)
        st.divider()

        _render_audit_diagnostics(step)
        _render_help(step)
        st.divider()

        _render_demo_controls()
