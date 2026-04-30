"""Global sidebar for LifeBudget Micro.

Streamlit-native version: no custom CSS/HTML.

The sidebar is a compact visual navigator plus a lightweight inspector. It uses
only native Streamlit components, avoids heavy computation, and keeps expensive
checks/actions on the main screen.
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

# Timing keys used by Step 5 suggestion modules. They are strings here on purpose
# so the sidebar does not need to import all recommendation modules just to render.
PRESET_SUGGESTION_TIMING_KEY = "step5_preset_suggestion_timing_v1"
AUTO_OPT_SUGGESTION_TIMING_KEY = "step5_auto_opt_suggestion_timing_v1"
UNIVERSE_SUGGESTION_TIMING_KEY = "step5_universe_suggestion_timing_v1"
SIZE_SUGGESTION_TIMING_KEY = "step5_size_suggestion_timing_v1"


STEP_NAV_ITEMS: tuple[dict[str, Any], ...] = (
    {"step": 0, "group": "Home", "icon": "", "title": "Home", "subtitle": "Module selector"},
    {"step": 1, "group": "Personal Finance", "icon": "", "title": "Step 1", "subtitle": "Income & Budget"},
    {"step": 2, "group": "Personal Finance", "icon": "", "title": "Step 2", "subtitle": "Goal Setup"},
    {"step": 3, "group": "Personal Finance", "icon": "", "title": "Step 3", "subtitle": "Savings Feasibility"},
    {"step": 4, "group": "Investment Setup", "icon": "", "title": "Step 4", "subtitle": "Universe & Data"},
    {"step": 5, "group": "Investment Setup", "icon": "", "title": "Step 5", "subtitle": "Engine Workspace"},
    {"step": 6, "group": "Scenario & Reports", "icon": "", "title": "Step 6", "subtitle": "Long-Term Scenario"},
    {"step": 7, "group": "Scenario & Reports", "icon": "", "title": "Step 7", "subtitle": "Insights & Reports"},
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
    likely_keys = (
        "step6_projection_result",
        "step6_last_projection_result",
        "long_term_projection_result",
        "projection_result",
        "step7_report_context",
    )
    return any(bool(st.session_state.get(key)) for key in likely_keys)


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
            return True, "Open universe and market-data setup."
        return False, "Choose an investing pathway before opening Step 4."

    if target_step == 5:
        if current_step >= 5 or _has_asset_panel() or _has_step5_result():
            return True, "Open the engine workspace."
        return False, "Prepare the Step 4 asset panel first."

    if target_step == 6:
        if current_step >= 6 or _has_step5_result() or _has_projection_bridge() or snapshot:
            return True, "Open long-term scenario exploration."
        return False, "Complete the planning setup or run Step 5 first."

    if target_step == 7:
        if current_step >= 7 or _has_projection_result():
            return True, "Open insights and reports."
        return False, "Generate a Step 6 scenario before opening reports."

    return False, "Step unavailable."


def _timing_total(key: str) -> float:
    payload = _coerce_mapping(st.session_state.get(key, {}))
    return _safe_float(payload.get("total_seconds", 0.0), 0.0)


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
                "Investment Setup",
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
    # branch. This keeps Step 6/7 visually separate from the investment engine,
    # even if the scenario is using Step 5 results.
    if step in {1, 2, 3}:
        return (
            "Personal Finance",
            (1, 2, 3),
            "Finance branch: one combined setup screen for budget, target, and feasibility.",
        )

    if step in {4, 5} or selected_module == "investment_lab":
        return (
            "Investment Setup",
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
                f"Data: {panel_meta.get('assets', 0)} assets · "
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
            "Step 5 engine result",
            bool(run_map),
            "available for diagnostics/projection" if run_map else "run portfolio from Step 5",
        )
        _readiness_line(
            "Step 6 projection input",
            projection_bridge or bool(snapshot),
            "ready" if (projection_bridge or bool(snapshot)) else "needs planning or Step 5 result",
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


def _render_tools_and_checks(step: int) -> None:
    with st.expander("Tools & checks", expanded=False):
        _render_available_checks(step)

        st.divider()
        st.markdown("**Safe shortcuts**")

        # Keep this list intentionally small. These buttons only navigate; they
        # never run engines, refresh data, apply suggestions, or launch exports.
        if step != 4:
            step4_enabled, step4_reason = _step_access_state(4, step)
            if st.button(
                "Open Universe & Data",
                key="global_sidebar_tool_open_step4",
                use_container_width=True,
                disabled=not step4_enabled,
                help=step4_reason,
            ):
                _go_to_step(4)

        if step != 5:
            step5_enabled, step5_reason = _step_access_state(5, step)
            if st.button(
                "Open Engine Workspace",
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
                "Open Insights & Reports",
                key="global_sidebar_tool_open_step7",
                use_container_width=True,
                disabled=not step7_enabled,
                help=step7_reason,
            ):
                _go_to_step(7)

        st.caption("Run, improve, refresh, robustness, and export actions stay on the main screen to avoid accidental heavy computation.")


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

        investment_context = _investment_context()
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

        _render_step_navigation(step)
        st.divider()

        _render_compact_status(step)

        with st.expander("Current context", expanded=False):
            _render_current_context(step)

        _render_readiness_status(step)
        _render_tools_and_checks(step)
        _render_audit_diagnostics(step)
        _render_help(step)
