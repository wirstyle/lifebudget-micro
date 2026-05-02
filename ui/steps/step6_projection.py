"""
Long-Term Scenario
"""

from __future__ import annotations

from typing import Any, Dict, List

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from ui.common.messages import section_header
from ui.common.tables import show_table
from ui.state.keys import (
    CURRENT_STEP,
    ENGINE_HAS_RUN,
    INVESTMENT_MONTHLY_CONTRIBUTION,
    INVESTMENT_PROJECTION_COMPARE_RESULTS,
    INVESTMENT_PROJECTION_COMPARE_SIGNATURE,
    INVESTMENT_PROJECTION_RESULT,
    INVESTMENT_PROJECTION_SIGNATURE,
    INVESTMENT_WEEKLY_EQUIVALENT,
    PROJECTION_OPEN,
    PLANNING_SNAPSHOT,
    TARGET_A_WEEKLY,
    WEEKLY_SAVINGS_DERIVED,
)
from services.projection_service import build_projection_workflow_bundle
from services.ui_adapters import build_projection_runtime_payload, build_projection_state_payload
from ui.services.step3_results_service import build_step3_view_model
try:
    from ui.state.updates import queue_and_rerun
except Exception:  # pragma: no cover
    def queue_and_rerun(updates, *, merge_nested=False):
        for _k, _v in dict(updates or {}).items():
            st.session_state[_k] = _v
        st.rerun()


DEFAULT_COMPARE_HORIZONS = [20, 30, 50]
SAVINGS_ONLY_HORIZONS = [20, 30, 50]
DEFAULT_STEP6_DEMO_PROFILE = "Balanced"
DEFAULT_STEP6_DEMO_RETURN_MONTHS = 135

LONG_TERM_SHOCK_OPTIONS = {
    "None": {
        "amount": 0.0,
        "frequency_years": 0,
        "label": "No long-term life-shock drag is applied.",
    },
    "Mild (rare)": {
        "amount": 500.0,
        "frequency_years": 10,
        "label": "Models an occasional £500 shock roughly once per decade as an expected monthly drag.",
    },
    "Moderate": {
        "amount": 1000.0,
        "frequency_years": 7,
        "label": "Models an occasional £1,000 shock roughly every 7 years as an expected monthly drag.",
    },
    "Stress scenario": {
        "amount": 2000.0,
        "frequency_years": 5,
        "label": "Models a heavier £2,000 shock roughly every 5 years as an expected monthly drag.",
    },
}


def _long_term_shock_config(option: str) -> Dict[str, Any]:
    option = str(option or "None")
    if option not in LONG_TERM_SHOCK_OPTIONS:
        option = "None"
    cfg = dict(LONG_TERM_SHOCK_OPTIONS[option])
    cfg["option"] = option

    amount = _safe_float(cfg.get("amount"), 0.0)
    frequency_years = _safe_int(cfg.get("frequency_years"), 0)
    if amount > 0.0 and frequency_years > 0:
        annual_expected_drag = amount / float(frequency_years)
        monthly_expected_drag = annual_expected_drag / 12.0
    else:
        annual_expected_drag = 0.0
        monthly_expected_drag = 0.0

    cfg["annual_expected_drag"] = float(annual_expected_drag)
    cfg["monthly_expected_drag"] = float(monthly_expected_drag)
    return cfg


def _render_long_term_shock_control(*, key_prefix: str) -> Dict[str, Any]:
    with st.expander("Optional: long-term life shocks", expanded=False):
        st.caption(
            "This is separate from the Personal Finance Setup short-term shock. Personal Finance Setup tests fragility over weeks; "
            "this optional setting models occasional long-term setbacks as an expected contribution drag."
        )
        current = str(st.session_state.get(f"{key_prefix}_long_term_shock_option", "None") or "None")
        if current not in LONG_TERM_SHOCK_OPTIONS:
            current = "None"
        selected = st.selectbox(
            "Include occasional life shocks in long-term scenario?",
            options=list(LONG_TERM_SHOCK_OPTIONS.keys()),
            index=list(LONG_TERM_SHOCK_OPTIONS.keys()).index(current),
            key=f"{key_prefix}_long_term_shock_option",
        )
        cfg = _long_term_shock_config(selected)
        st.caption(str(cfg.get("label", "")))
        if float(cfg.get("monthly_expected_drag", 0.0)) > 0.0:
            st.info(
                f"Expected long-term drag: about **{_format_currency(float(cfg['monthly_expected_drag']))}/month** "
                f"({_format_currency(float(cfg['annual_expected_drag']))}/year on average). "
                "This is not copied from Personal Finance Setup and is not treated as an annual guaranteed shock."
            )
        else:
            st.caption("Default: no long-term life-shock adjustment.")
        return cfg


def _apply_long_term_shock_to_contribution(monthly_contribution: float, shock_cfg: Dict[str, Any]) -> tuple[float, float]:
    monthly_drag = _safe_float(shock_cfg.get("monthly_expected_drag"), 0.0)
    adjusted = max(float(monthly_contribution) - monthly_drag, 0.0)
    return adjusted, monthly_drag



def _get_projection_path_recommendation(philosophy: str) -> Dict[str, Any]:
    philosophy_name = str(philosophy or "Balanced").strip().capitalize()
    if philosophy_name == "Growth":
        return {
            "philosophy": "Growth",
            "mode_label": "Hybrid daily simulation",
            "mode": "daily_hybrid",
            "days": 21,
            "variation": 0.30,
            "rationale": "Growth can tolerate a more visibly volatile path, so the projection should show more intramonth movement.",
        }
    if philosophy_name == "Defensive":
        return {
            "philosophy": "Defensive",
            "mode_label": "Hybrid daily simulation",
            "mode": "daily_hybrid",
            "days": 10,
            "variation": 0.10,
            "rationale": "Defensive portfolios should still use the hybrid path, but with a tighter and smoother intramonth profile.",
        }
    return {
        "philosophy": "Balanced",
        "mode_label": "Hybrid daily simulation",
        "mode": "daily_hybrid",
        "days": 21,
        "variation": 0.18,
        "rationale": "Balanced is best represented by a realistic but not overly aggressive hybrid path.",
    }


def _classify_projection_path_coherence(
    *,
    philosophy: str,
    mode: str,
    days: int,
    variation: float,
) -> Dict[str, str]:
    rec = _get_projection_path_recommendation(philosophy)
    target_mode = str(rec.get("mode", "daily_hybrid"))
    target_days = int(rec.get("days", 21) or 21)
    target_variation = float(rec.get("variation", 0.18) or 0.18)

    if str(mode or "monthly") != target_mode:
        return {
            "status": "warning",
            "title": "Projection path is outside the philosophy recommendation.",
            "message": f"{philosophy} currently recommends {rec['mode_label']} rather than monthly bootstrap.",
        }

    day_gap = abs(int(days) - target_days)
    variation_gap = abs(float(variation) - target_variation)

    if day_gap <= 2 and variation_gap <= 0.05:
        return {
            "status": "success",
            "title": "Projection path aligned with mother philosophy.",
            "message": f"{philosophy} recommendation matched: {target_days} synthetic days/month and variation {target_variation:.2f}.",
        }
    if day_gap <= 6 and variation_gap <= 0.12:
        return {
            "status": "info",
            "title": "Projection path is broadly coherent with mother philosophy.",
            "message": f"Current settings are close to the {philosophy} recommendation, but not the primary suggested combination.",
        }
    return {
        "status": "warning",
        "title": "Projection path deviates from the philosophy recommendation.",
        "message": f"{philosophy} recommends {target_days} synthetic days/month and variation {target_variation:.2f}.",
    }


def _coerce_mapping(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            out = to_dict()
            if isinstance(out, dict):
                return dict(out)
        except Exception:
            pass
    try:
        return dict(value)
    except Exception:
        return {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if pd.notnull(out):
            return float(out)
    except Exception:
        pass
    return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _step6_back_target() -> tuple[int, str]:
    """Return the most useful previous-step target for Long-Term Scenario navigation.

    If the user has a real Strategy Engine result, Long-Term Scenario is downstream of
    Strategy Engine. If not, Long-Term Scenario is acting as a savings/proxy scenario screen and
    the most useful previous step is the Personal Finance Setup.
    """
    has_engine_result = bool(
        st.session_state.get(ENGINE_HAS_RUN, False)
        or st.session_state.get("step5_last_run_result")
        or st.session_state.get("step5_run_result")
        or st.session_state.get("last_run_result")
        or st.session_state.get("run_result")
    )
    if has_engine_result:
        return 5, "← Back to Strategy Engine"
    return 1, "← Back to Personal Finance Setup"


def _go_to_step(step: int) -> None:
    st.session_state[CURRENT_STEP] = int(step)
    st.session_state["current_step"] = int(step)
    st.rerun()


def _summary_value(summary: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in summary and summary.get(key) is not None:
            return _safe_float(summary.get(key), default=default)
    return float(default)


def _format_currency(value: float, decimals: int = 0) -> str:
    return f"£{value:,.{decimals}f}"


def _format_probability(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100.0:,.2f}%"
    except Exception:
        return "—"


def _default_investment_wealth_goal(monthly_contribution: float, horizon_years: int = 20) -> float:
    """Return a useful default goal for the investment/proxy branch.

    The default scales with the selected projection horizon, so a short
    one-year scenario does not inherit an unrealistic £100k target. The goal is
    only auto-updated while the user has not manually edited the goal field.
    """
    monthly = max(_safe_float(monthly_contribution, 0.0), 0.0)
    horizon = max(1, min(50, _safe_int(horizon_years, 20)))
    if monthly <= 0.0:
        return 0.0

    if horizon <= 1:
        multiplier = 1.25
    elif horizon <= 5:
        multiplier = 1.50
    elif horizon <= 10:
        multiplier = 1.75
    elif horizon <= 20:
        multiplier = 2.00
    elif horizon <= 30:
        multiplier = 2.35
    else:
        multiplier = 2.80

    raw_goal = monthly * 12.0 * float(horizon) * multiplier
    if raw_goal < 10000.0:
        rounding_unit = 1000.0
    elif raw_goal < 50000.0:
        rounding_unit = 5000.0
    else:
        rounding_unit = 10000.0

    rounded = round(raw_goal / rounding_unit) * rounding_unit
    return float(max(rounding_unit, rounded))


def _horizon_years_from_label(value: Any) -> int:
    """Parse labels such as '20 years' or '1 years (current)' safely."""
    import re

    match = re.search(r"(\d+)", str(value or ""))
    if not match:
        return 0
    try:
        return int(match.group(1))
    except Exception:
        return 0


def _savings_only_terminal_value(*, starting_value: float, monthly_contribution: float, horizon_years: int) -> float:
    years = max(int(horizon_years), 0)
    return float(max(_safe_float(starting_value, 0.0), 0.0) + max(_safe_float(monthly_contribution, 0.0), 0.0) * 12.0 * years)


def _positive_float(value: Any) -> float:
    """Return a positive float or 0.0.

    This helper is intentionally strict for contribution resolution: a stored
    zero should not mask a later positive fallback value.
    """
    try:
        out = float(value)
        if pd.notnull(out) and out > 0.0:
            return float(out)
    except Exception:
        pass
    return 0.0


def _resolve_projection_contribution(*, prefer_plan: bool = False) -> Dict[str, Any]:
    """Resolve the monthly/weekly contribution used by Long-Term Scenario.

    Long-Term Scenario can be reached from several places: Personal Finance, Investment
    Lab, or directly from Home. Older state keys may contain a stored zero, so
    this resolver checks every known source and only accepts positive values.
    The final fallback matches the Risk Profile & Asset Universe hosted-demo default.
    """
    investment_context = _coerce_mapping(st.session_state.get("investment_context", {}))
    snapshot = _coerce_mapping(st.session_state.get(PLANNING_SNAPSHOT, {}))

    plan_weekly_candidates = [
        ("Personal Finance target", snapshot.get("target_a_weekly")),
        ("Personal Finance target", snapshot.get("target_weekly")),
        ("Personal Finance target", snapshot.get("weekly_savings_target")),
        ("Personal Finance target", st.session_state.get(TARGET_A_WEEKLY)),
        ("Personal Finance derived savings", st.session_state.get(WEEKLY_SAVINGS_DERIVED)),
        ("Personal Finance baseline savings", snapshot.get("baseline_savings_weekly")),
    ]
    investment_weekly_candidates = [
        ("Risk Profile & Asset Universe contribution bridge", investment_context.get("weekly_equivalent")),
        ("Risk Profile & Asset Universe contribution bridge", investment_context.get("weekly_contribution")),
        ("Risk Profile & Asset Universe contribution bridge", investment_context.get("target_weekly")),
        ("Investment contribution", st.session_state.get(INVESTMENT_WEEKLY_EQUIVALENT)),
    ]
    plan_monthly_candidates = [
        ("Personal Finance target", snapshot.get("target_a_monthly")),
        ("Personal Finance target", snapshot.get("target_monthly")),
        ("Personal Finance target", snapshot.get("monthly_savings_target")),
    ]
    investment_monthly_candidates = [
        ("Risk Profile & Asset Universe contribution bridge", investment_context.get("monthly_contribution")),
        ("Risk Profile & Asset Universe contribution bridge", investment_context.get("target_a_monthly")),
        ("Investment contribution", st.session_state.get(INVESTMENT_MONTHLY_CONTRIBUTION)),
    ]

    weekly_candidates = (plan_weekly_candidates + investment_weekly_candidates) if prefer_plan else (investment_weekly_candidates + plan_weekly_candidates)
    monthly_candidates = (plan_monthly_candidates + investment_monthly_candidates) if prefer_plan else (investment_monthly_candidates + plan_monthly_candidates)

    for label, raw in weekly_candidates:
        weekly = _positive_float(raw)
        if weekly > 0.0:
            monthly = weekly * 52.0 / 12.0
            return {
                "weekly": float(weekly),
                "monthly": float(monthly),
                "source": str(investment_context.get("contribution_bridge_source") or label),
                "fallback_used": False,
            }

    for label, raw in monthly_candidates:
        monthly = _positive_float(raw)
        if monthly > 0.0:
            weekly = monthly * 12.0 / 52.0
            return {
                "weekly": float(weekly),
                "monthly": float(monthly),
                "source": str(investment_context.get("contribution_bridge_source") or label),
                "fallback_used": False,
            }

    weekly = 48.0
    monthly = weekly * 52.0 / 12.0
    return {
        "weekly": float(weekly),
        "monthly": float(monthly),
        "source": "Demo default",
        "fallback_used": True,
    }


def _sync_projection_contribution_state(contribution: Dict[str, Any]) -> None:
    """Keep common Long-Term Scenario/Insights Summary contribution aliases aligned."""
    monthly = _safe_float(contribution.get("monthly"), 0.0)
    weekly = _safe_float(contribution.get("weekly"), 0.0)
    st.session_state[INVESTMENT_MONTHLY_CONTRIBUTION] = float(monthly)
    st.session_state[INVESTMENT_WEEKLY_EQUIVALENT] = float(weekly)

    ctx = _coerce_mapping(st.session_state.get("investment_context", {}))
    ctx["monthly_contribution"] = float(monthly)
    ctx["weekly_equivalent"] = float(weekly)
    ctx.setdefault("contribution_bridge_source", str(contribution.get("source", "")))
    st.session_state["investment_context"] = ctx


def _projection_view_from_label(label: str) -> str:
    raw = str(label or "").lower()
    if "compare" in raw or "investment" in raw or "proxy" in raw or "strategy" in raw:
        return "savings_plus_investing"
    return "savings_only"


def _default_step6_view() -> str:
    pathway = _current_pathway()
    if pathway == "savings_only":
        return "savings_only"
    if pathway in {"savings_plus_investing", "compare_both"}:
        # The old compare branch is now represented as an investment view with
        # an optional savings-only baseline overlay.
        return "savings_plus_investing"
    return "savings_plus_investing"


def _step6_has_real_engine_return_path() -> bool:
    """Return True only when Strategy Engine has produced a usable strategy return path."""
    investment_context = _coerce_mapping(st.session_state.get("investment_context", {}))
    return bool(st.session_state.get(ENGINE_HAS_RUN, False)) and _has_real_step5_return_path(investment_context)


def _step6_projection_options() -> list[tuple[str, str]]:
    """Label the two Long-Term Scenario modes as one scenario screen.

    Savings-only remains available as the simple accumulation view. The
    investment/proxy mode is now framed as the comparison view: it keeps the
    investment/proxy projection and automatically includes the savings-only
    baseline, so no content from either path is lost.
    """
    if _step6_has_real_engine_return_path():
        return [
            ("savings_only", "Savings-only accumulation"),
            ("savings_plus_investing", "Compare with tested strategy"),
        ]
    return [
        ("savings_only", "Savings-only accumulation"),
        ("savings_plus_investing", "Compare with investment scenario"),
    ]


def _render_step6_evidence_note() -> None:
    if _step6_has_real_engine_return_path():
        st.success(
            "Evidence mode: **tested Strategy Engine returns** are available. "
            "Investment and comparison views use the engine-generated monthly return path."
        )
        return

    st.info(
        "Available now: **savings-only scenarios** and a clearly labelled **educational investment proxy**. "
        "Run the Investment Strategy Lab later to replace the proxy with tested Strategy Engine returns automatically."
    )


def _render_step6_view_selector() -> str:
    options = _step6_projection_options()
    valid = [value for value, _ in options]
    current = str(st.session_state.get("step6_projection_view_mode_v1", _default_step6_view()) or _default_step6_view())
    if current not in valid:
        current = _default_step6_view()
    if current not in valid:
        current = valid[0] if valid else "savings_only"

    labels = [label for _, label in options]
    current_label = dict(options).get(current, labels[0])
    selected = st.radio(
        "Projection mode",
        options=labels,
        index=labels.index(current_label),
        horizontal=True,
        key="step6_projection_view_radio_v1",
        help=(
            "Savings-only accumulation shows the contribution-only path. The comparison mode shows the "
            "investment/proxy path alongside the savings-only baseline, using tested Strategy Engine returns when available "
            "or a clearly labelled educational proxy before Strategy Engine has run."
        ),
    )
    selected_mode = _projection_view_from_label(selected)
    st.session_state["step6_projection_view_mode_v1"] = selected_mode
    return selected_mode


def _render_step6_projection_interpretation_note(
    *,
    monthly_contribution: float,
    weekly_equivalent: float = 0.0,
    using_demo_proxy: bool = False,
    profile_value: str = DEFAULT_STEP6_DEMO_PROFILE,
    return_source_label: str = "",
    comparison_table: pd.DataFrame | None = None,
    goal_amount: float = 0.0,
    metrics: dict[str, Any] | None = None,
) -> None:
    """Render one compact reader that also contains the horizon table and key detailed metrics."""
    title = "How to read this projection and table"
    with st.expander(title, expanded=False):
        if using_demo_proxy:
            st.info(
                f"This view compares your savings-only baseline with a clearly labelled "
                f"**educational investment proxy**. Because Strategy Engine has not been run, "
                f"the investment path currently uses a transparent **{profile_value} proxy assumption**. "
                f"Long-Term Scenario does not predict an exact future amount: it translates your "
                f"**{_format_currency(monthly_contribution)}/month** contribution into possible low / middle / high "
                f"outcome ranges under the selected return-path assumptions. Run Risk Profile & Asset Universe "
                f"and the Strategy Engine later to replace the proxy with tested Strategy Engine returns automatically."
            )
        else:
            st.info(
                f"This view compares your savings-only baseline with the selected investment path. "
                f"The investment path uses the completed Strategy Engine return series. "
                f"Long-Term Scenario does not predict an exact future amount: it translates your "
                f"**{_format_currency(monthly_contribution)}/month** contribution into possible low / middle / high "
                f"outcome ranges under the selected return-path assumptions."
            )

        if isinstance(comparison_table, pd.DataFrame) and not comparison_table.empty:
            st.divider()
            st.markdown("**Horizon comparison table**")
            show_table(comparison_table, title=None)
            if goal_amount <= 0.0 and "goal probability" in comparison_table.columns:
                st.caption("Goal probability uses the optional wealth goal. Set a goal above £0 to populate that column.")

        metrics_map = _coerce_mapping(metrics)
        if metrics_map:
            st.divider()
            st.markdown("**Detailed projection metrics**")
            detail_metrics = st.columns(4)
            with detail_metrics[0]:
                st.metric("Expected terminal wealth", _format_currency(metrics_map.get("expected_terminal", 0.0)))
            with detail_metrics[1]:
                st.metric(
                    "P10–P90 range",
                    f"{_format_currency(metrics_map.get('p10_terminal', 0.0))} ··· {_format_currency(metrics_map.get('p90_terminal', 0.0))}",
                )
            with detail_metrics[2]:
                st.metric("Loss vs contributions", _format_probability(metrics_map.get("loss_probability", 0.0)))
            with detail_metrics[3]:
                goal_display = "—" if goal_amount <= 0.0 else _format_probability(metrics_map.get("goal_probability", 0.0))
                st.metric("Goal probability", goal_display)


def _profile_assumptions_for_demo_proxy(profile: str) -> Dict[str, float]:
    """Return conservative educational profile assumptions for the Long-Term Scenario fallback.

    This is deliberately not presented as a real engine result. It only exists
    so the Investment / Compare pathways can be demonstrated before a Strategy Engine run.
    A real Strategy Engine engine path always takes priority when available.
    """
    name = str(profile or DEFAULT_STEP6_DEMO_PROFILE).strip().capitalize()
    if name == "Growth":
        return {"annual_return": 0.075, "annual_volatility": 0.16}
    if name == "Defensive":
        return {"annual_return": 0.040, "annual_volatility": 0.07}
    return {"annual_return": 0.055, "annual_volatility": 0.11}


def _build_demo_monthly_return_proxy(profile: str, months: int = DEFAULT_STEP6_DEMO_RETURN_MONTHS) -> list[float]:
    """Build a deterministic monthly return path for demo-only projections.

    The pattern is deterministic rather than random so repeated demos are stable.
    It approximates a broad profile-level return/volatility path, not the output
    of the Strategy Engine engine.
    """
    assumptions = _profile_assumptions_for_demo_proxy(profile)
    annual_return = float(assumptions["annual_return"])
    annual_vol = float(assumptions["annual_volatility"])
    monthly_mean = (1.0 + annual_return) ** (1.0 / 12.0) - 1.0
    monthly_vol = annual_vol / float(np.sqrt(12.0))

    base_pattern = np.array(
        [-0.95, 0.15, 0.55, -0.35, 0.90, -0.65, 0.35, 0.05, -1.15, 0.80, 0.45, -0.15],
        dtype=float,
    )
    base_pattern = (base_pattern - float(base_pattern.mean())) / float(base_pattern.std(ddof=0))
    reps = int(np.ceil(max(int(months), 1) / len(base_pattern)))
    z = np.tile(base_pattern, reps)[: max(int(months), 1)]
    returns = monthly_mean + monthly_vol * z
    returns = np.clip(returns, -0.25, 0.25)
    return [float(x) for x in returns]


def _has_real_step5_return_path(investment_context: Dict[str, Any]) -> bool:
    """Check whether Long-Term Scenario has a real Strategy Engine return path to project from."""
    candidates = [
        investment_context.get("oos_returns_monthly"),
        investment_context.get("oos_returns_simple"),
        investment_context.get("portfolio_returns"),
    ]
    for raw in candidates:
        try:
            if hasattr(raw, "tolist"):
                raw = raw.tolist()
        except Exception:
            raw = []
        if isinstance(raw, list):
            clean = []
            for item in raw:
                try:
                    value = float(item)
                    if pd.notnull(value):
                        clean.append(value)
                except Exception:
                    continue
            if len(clean) >= 12:
                return True
    return False


def _projection_context_with_optional_demo_proxy(
    investment_context: Dict[str, Any],
    *,
    profile: str,
) -> tuple[Dict[str, Any], bool, str]:
    """Return a projection context using real Strategy Engine data or a labelled demo proxy.

    Real Strategy Engine results always win. If no engine return path exists yet, Long-Term Scenario
    can still render the Investment and Compare views using a transparent,
    deterministic profile-level proxy. This avoids a dead end in demo navigation
    while keeping the distinction between demo assumption and real engine output.
    """
    ctx = dict(investment_context or {})
    if bool(st.session_state.get(ENGINE_HAS_RUN, False)) and _has_real_step5_return_path(ctx):
        ctx["projection_return_source"] = "Historical Strategy Engine path"
        return ctx, False, "Historical Strategy Engine path"

    ctx["oos_returns_monthly"] = _build_demo_monthly_return_proxy(profile)
    ctx["projection_return_source"] = f"Demo {str(profile or DEFAULT_STEP6_DEMO_PROFILE)} proxy"
    ctx["projection_proxy_is_demo"] = True
    return ctx, True, f"Demo {str(profile or DEFAULT_STEP6_DEMO_PROFILE)} proxy"


def _extract_projection_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    summary = _coerce_mapping(payload.get("summary", {}))
    if summary:
        return summary

    result = _coerce_mapping(payload.get("result", {}))
    result_summary = _coerce_mapping(result.get("summary", {}))
    if result_summary:
        return result_summary

    projection_df = payload.get("projection_df", pd.DataFrame())
    if isinstance(projection_df, pd.DataFrame) and not projection_df.empty:
        value_col = None
        for candidate in ("projected_value", "value", "wealth", "portfolio_value"):
            if candidate in projection_df.columns:
                value_col = candidate
                break
        if value_col is not None:
            terminal = _safe_float(projection_df[value_col].iloc[-1])
            return {
                "expected_terminal": terminal,
                "median_terminal": terminal,
                "p10_terminal": terminal,
                "p90_terminal": terminal,
            }
    return {}


def _extract_metrics(summary: Dict[str, Any], monthly_contribution: float) -> Dict[str, Any]:
    expected_terminal = _summary_value(summary, "expected_terminal", "final_value")
    median_terminal = _summary_value(
        summary,
        "median_terminal",
        "expected_terminal",
        "final_value",
        default=expected_terminal,
    )
    p10_terminal = _summary_value(summary, "p10_terminal", default=median_terminal)
    p90_terminal = _summary_value(summary, "p90_terminal", default=median_terminal)
    total_contributed = _summary_value(summary, "total_contributed")
    expected_profit = _summary_value(summary, "expected_profit", default=(expected_terminal - total_contributed))
    goal_probability = summary.get("probability_of_reaching_goal")
    loss_probability = summary.get("probability_of_loss_vs_contributions")
    horizon_years = _safe_int(summary.get("horizon_years"), 0)
    return {
        "expected_terminal": expected_terminal,
        "median_terminal": median_terminal,
        "p10_terminal": p10_terminal,
        "p90_terminal": p90_terminal,
        "monthly_contribution": _safe_float(monthly_contribution),
        "total_contributed": total_contributed,
        "expected_profit": expected_profit,
        "goal_probability": goal_probability,
        "loss_probability": loss_probability,
        "horizon_years": horizon_years,
    }


def _coerce_dataframe(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if value is None:
        return pd.DataFrame()
    try:
        df = pd.DataFrame(value)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _normalise_wealth_paths(value: Any, horizon_years: int | None = None) -> pd.DataFrame:
    df = _coerce_dataframe(value)
    if df.empty:
        return pd.DataFrame()

    lower_columns = {str(c).lower(): c for c in df.columns}

    if {"month", "median", "p10", "p90"}.issubset(set(lower_columns.keys())):
        work = df[
            [
                lower_columns["month"],
                lower_columns["median"],
                lower_columns["p10"],
                lower_columns["p90"],
            ]
        ].copy()
        work.columns = ["month", "median", "p10", "p90"]
        work["month"] = pd.to_numeric(work["month"], errors="coerce")
        for col in ("median", "p10", "p90"):
            work[col] = pd.to_numeric(work[col], errors="coerce")
        return work.dropna(subset=["month"]).sort_values("month").reset_index(drop=True)

    if {"months", "median", "p10", "p90"}.issubset(set(lower_columns.keys())):
        work = df[
            [
                lower_columns["months"],
                lower_columns["median"],
                lower_columns["p10"],
                lower_columns["p90"],
            ]
        ].copy()
        work.columns = ["month", "median", "p10", "p90"]
        work["month"] = pd.to_numeric(work["month"], errors="coerce")
        for col in ("median", "p10", "p90"):
            work[col] = pd.to_numeric(work[col], errors="coerce")
        return work.dropna(subset=["month"]).sort_values("month").reset_index(drop=True)

    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if numeric_df.empty:
        return pd.DataFrame()

    arr = numeric_df.to_numpy(dtype="float64")
    arr = arr[np.isfinite(arr).any(axis=1)][:, np.isfinite(arr).any(axis=0)]
    if arr.size == 0:
        return pd.DataFrame()

    expected_months = None
    if horizon_years is not None and int(horizon_years) > 0:
        expected_months = int(horizon_years) * 12 + 1

    # Core convention from old app.py:
    # rows = simulations, columns = months
    # If one dimension matches expected months, force that as columns.
    if expected_months is not None:
        if arr.shape[1] == expected_months:
            month_axis = 1
        elif arr.shape[0] == expected_months:
            month_axis = 0
        else:
            month_axis = 1 if arr.shape[1] >= arr.shape[0] else 0
    else:
        month_axis = 1 if arr.shape[1] >= arr.shape[0] else 0

    if month_axis == 0:
        arr = arr.T

    work = pd.DataFrame(
        {
            "month": range(arr.shape[1]),
            "median": np.nanmedian(arr, axis=0),
            "p10": np.nanquantile(arr, 0.10, axis=0),
            "p90": np.nanquantile(arr, 0.90, axis=0),
        }
    )
    return work.reset_index(drop=True)


def _build_main_chart_df(projection_result: Dict[str, Any], horizon_years: int | None = None) -> pd.DataFrame:
    wealth_paths = projection_result.get("wealth_paths")
    chart_df = _normalise_wealth_paths(wealth_paths, horizon_years=horizon_years)
    if not chart_df.empty:
        return chart_df.rename(
            columns={
                "median": "Median",
                "p10": "P10",
                "p90": "P90",
            }
        )

    projection_df = _coerce_dataframe(projection_result.get("projection_df"))
    if projection_df.empty:
        return pd.DataFrame()

    month_col = None
    for candidate in ("month", "months", "t", "period"):
        if candidate in projection_df.columns:
            month_col = candidate
            break

    value_col = None
    for candidate in ("projected_value", "value", "wealth", "portfolio_value"):
        if candidate in projection_df.columns:
            value_col = candidate
            break

    if month_col is not None and value_col is not None:
        work = projection_df[[month_col, value_col]].copy()
        work.columns = ["month", "Median"]
        work["month"] = pd.to_numeric(work["month"], errors="coerce")
        work["Median"] = pd.to_numeric(work["Median"], errors="coerce")
        work = work.dropna(subset=["month", "Median"]).sort_values("month").reset_index(drop=True)
        work["P10"] = work["Median"]
        work["P90"] = work["Median"]
        return work[["month", "Median", "P10", "P90"]]

    if "year" in projection_df.columns and value_col is not None:
        work = projection_df[["year", value_col]].copy()
        work["year"] = pd.to_numeric(work["year"], errors="coerce")
        work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
        work = work.dropna(subset=["year", value_col]).sort_values("year").reset_index(drop=True)
        expanded_rows: List[Dict[str, float]] = []
        for _, row in work.iterrows():
            month_index = int(row["year"] * 12)
            expanded_rows.append({"month": month_index, "Median": float(row[value_col])})
        if expanded_rows:
            out = pd.DataFrame(expanded_rows)
            out["P10"] = out["Median"]
            out["P90"] = out["Median"]
            return out[["month", "Median", "P10", "P90"]]

    return pd.DataFrame()


def _extract_compare_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not payload:
        return {}
    result = _coerce_mapping(payload.get("result", {}))
    if result:
        return result
    return payload


def _build_ordered_horizon_payloads(
    compare_results: Dict[str, Any],
    current_horizon_years: int,
    current_result: Dict[str, Any],
    selected_compare_horizons: List[int] | None = None,
) -> List[tuple[int, Dict[str, Any], bool]]:
    ordered: List[tuple[int, Dict[str, Any], bool]] = []
    seen: set[int] = set()

    current_payload = _coerce_mapping(current_result)
    if current_payload:
        ordered.append((int(current_horizon_years), current_payload, True))
        seen.add(int(current_horizon_years))

    requested = sorted(set(int(x) for x in (selected_compare_horizons or []) if int(x) > 0))
    for horizon in requested:
        payload = _extract_compare_payload(_coerce_mapping(compare_results.get(str(int(horizon)), {})))
        if payload and horizon not in seen:
            ordered.append((int(horizon), payload, False))
            seen.add(int(horizon))

    discovered: List[tuple[int, Dict[str, Any], bool]] = []
    for key, raw_payload in compare_results.items():
        payload = _extract_compare_payload(_coerce_mapping(raw_payload))
        if not payload:
            continue
        summary = _extract_projection_summary(payload)
        horizon_years = _safe_int(summary.get("horizon_years"), _safe_int(key, 0))
        if horizon_years <= 0 or horizon_years in seen:
            continue
        discovered.append((int(horizon_years), payload, False))
        seen.add(int(horizon_years))

    discovered = sorted(discovered, key=lambda x: x[0])
    ordered.extend(discovered)

    ordered = sorted(ordered, key=lambda x: x[0])
    return ordered


def _build_horizon_table(
    compare_results: Dict[str, Any],
    current_horizon_years: int,
    current_result: Dict[str, Any],
    selected_compare_horizons: List[int] | None = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for horizon_years, payload, is_current in _build_ordered_horizon_payloads(
        compare_results,
        current_horizon_years,
        current_result,
        selected_compare_horizons=selected_compare_horizons,
    ):
        summary = _extract_projection_summary(payload)
        if not summary:
            continue
        metrics = _extract_metrics(summary, monthly_contribution=0.0)
        rows.append(
            {
                "horizon": f"{horizon_years} years (current)" if is_current else f"{horizon_years} years",
                "median final wealth": metrics["median_terminal"],
                "p10": metrics["p10_terminal"],
                "p50": metrics["median_terminal"],
                "p90": metrics["p90_terminal"],
                "loss vs contributions": metrics.get("loss_probability"),
                "goal probability": summary.get("probability_of_reaching_goal"),
                "_sort": horizon_years,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)
    return df


def _build_terminal_horizon_chart_df(
    compare_results: Dict[str, Any],
    current_horizon_years: int,
    current_result: Dict[str, Any],
    selected_compare_horizons: List[int] | None = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for horizon_years, payload, _ in _build_ordered_horizon_payloads(
        compare_results,
        current_horizon_years,
        current_result,
        selected_compare_horizons=selected_compare_horizons,
    ):
        summary = _extract_projection_summary(payload)
        if not summary:
            continue
        median_terminal = _summary_value(summary, "median_terminal", "expected_terminal", "final_value")
        rows.append({"horizon_years": horizon_years, "Median terminal wealth": median_terminal})

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("horizon_years").reset_index(drop=True)


def _build_median_paths_comparison_chart(
    compare_results: Dict[str, Any],
    current_horizon_years: int,
    current_result: Dict[str, Any],
    selected_compare_horizons: List[int] | None = None,
) -> pd.DataFrame:
    series_frames: List[pd.DataFrame] = []

    ordered_payloads = _build_ordered_horizon_payloads(
        compare_results,
        current_horizon_years,
        current_result,
        selected_compare_horizons=selected_compare_horizons,
    )

    # Smallest horizon first, largest last.
    for horizon_years, payload, is_current in ordered_payloads:
        chart_df = _build_main_chart_df(payload, horizon_years=horizon_years)
        if chart_df.empty or "Median" not in chart_df.columns:
            continue

        frame = chart_df[["month", "Median"]].copy()
        label = f"{horizon_years}y median (current)" if is_current else f"{horizon_years}y median"
        frame = frame.rename(columns={"Median": label})
        series_frames.append(frame)

    if not series_frames:
        return pd.DataFrame()

    merged = series_frames[0]
    for frame in series_frames[1:]:
        merged = pd.merge(merged, frame, on="month", how="outer")

    merged = merged.sort_values("month").reset_index(drop=True)

    # With st.line_chart, the most reliable approximation of visual layering
    # is to order the series from largest horizon to smallest horizon.
    # That way, the smallest horizon is drawn last and tends to remain most visible.
    ordered_cols = ["month"] + sorted(
        [col for col in merged.columns if col != "month"],
        key=lambda x: int(str(x).split("y")[0]),
        reverse=True,
    )
    return merged[ordered_cols]


def _add_savings_baseline_to_horizon_comparison_chart(
    chart_df: pd.DataFrame,
    *,
    show_savings_baseline: bool,
    starting_value: float,
    monthly_contribution: float,
    goal_amount: float,
) -> pd.DataFrame:
    """Overlay a savings-only baseline on the multi-horizon comparison chart.

    The investment/proxy branch now uses the horizon-comparison chart as the
    main visual. When enabled, this adds one no-market-return savings path
    across the longest displayed horizon instead of rendering a separate
    single-horizon investment chart above it.
    """
    if not isinstance(chart_df, pd.DataFrame) or chart_df.empty or "month" not in chart_df.columns:
        return pd.DataFrame()

    out = chart_df.copy()
    if not show_savings_baseline:
        return out

    try:
        max_month = int(pd.to_numeric(out["month"], errors="coerce").max())
    except Exception:
        max_month = 0
    if max_month <= 0:
        return out

    max_horizon_years = max(1, int(np.ceil(max_month / 12.0)))
    savings_result = _build_savings_only_projection(
        starting_value=float(starting_value),
        monthly_contribution=float(monthly_contribution),
        horizon_years=int(max_horizon_years),
        goal_amount=float(goal_amount),
    )
    savings_chart_df = _build_main_chart_df(savings_result, horizon_years=int(max_horizon_years))
    if not isinstance(savings_chart_df, pd.DataFrame) or savings_chart_df.empty or "Median" not in savings_chart_df.columns:
        return out

    baseline = savings_chart_df[["month", "Median"]].copy()
    baseline["month"] = pd.to_numeric(baseline["month"], errors="coerce")
    baseline["Median"] = pd.to_numeric(baseline["Median"], errors="coerce")
    baseline = baseline.dropna(subset=["month", "Median"])
    baseline = baseline.loc[baseline["month"] <= max_month]
    baseline = baseline.rename(columns={"Median": "Savings-only baseline"})
    if baseline.empty:
        return out

    return pd.merge(out, baseline, on="month", how="outer").sort_values("month").reset_index(drop=True)


def _build_horizon_comparison_altair_chart(chart_df: pd.DataFrame) -> alt.Chart | None:
    if not isinstance(chart_df, pd.DataFrame) or chart_df.empty or "month" not in chart_df.columns:
        return None

    value_cols = [col for col in chart_df.columns if col != "month"]
    if not value_cols:
        return None

    work = chart_df.copy()
    long_df = work.melt(id_vars="month", value_vars=value_cols, var_name="series", value_name="value")
    long_df["month"] = pd.to_numeric(long_df["month"], errors="coerce")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["month", "value"])
    if long_df.empty:
        return None

    horizon_order_raw = long_df["series"].str.extract(r"^(\d+)")[0]
    long_df["horizon_order"] = pd.to_numeric(horizon_order_raw, errors="coerce").fillna(-1.0)
    long_df["is_endpoint"] = False

    endpoint_idx = long_df.groupby("series")["month"].idxmax()
    long_df.loc[endpoint_idx, "is_endpoint"] = True

    # Draw longer horizons first and shorter ones last so smaller horizons stay more visible.
    line_data = long_df.sort_values(["horizon_order", "month"], ascending=[False, True])

    base = alt.Chart(line_data).encode(
        x=alt.X(
            "month:Q",
            title="Month",
            axis=alt.Axis(titlePadding=16, labelPadding=6),
        ),
        y=alt.Y("value:Q", title=None),
        color=alt.Color("series:N", title=None),
        order=alt.Order("horizon_order:Q", sort="descending"),
    )

    lines = base.mark_line().encode(
        detail="series:N"
    )

    points = alt.Chart(line_data[line_data["is_endpoint"]]).mark_point(filled=True, size=90).encode(
        x=alt.X("month:Q"),
        y=alt.Y("value:Q"),
        color=alt.Color("series:N", title=None),
        tooltip=[
            alt.Tooltip("series:N", title="Series"),
            alt.Tooltip("month:Q", title="Final month", format=".0f"),
            alt.Tooltip("value:Q", title="Final value", format=",.0f"),
        ],
    )

    chart = (
        (lines + points)
        .properties(
            height=420,
            padding={"top": 8, "right": 12, "bottom": 48, "left": 4},
        )
        .configure_axisX(titlePadding=16, labelPadding=6)
    )
    return chart


def _build_investment_chart_with_optional_savings_baseline(
    investment_chart_df: pd.DataFrame,
    *,
    show_savings_baseline: bool,
    starting_value: float,
    monthly_contribution: float,
    horizon_years: int,
    goal_amount: float,
) -> pd.DataFrame:
    """Return the investment chart, optionally with a savings-only baseline.

    The baseline is a visual overlay only. It does not replace the investment
    projection calculation and it does not require a separate compare branch.
    """
    if not isinstance(investment_chart_df, pd.DataFrame) or investment_chart_df.empty:
        return pd.DataFrame()

    out = investment_chart_df.copy()
    if show_savings_baseline:
        rename_map = {
            "Median": "Investment median",
            "P10": "Investment P10",
            "P90": "Investment P90",
        }
        out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})
        savings_result = _build_savings_only_projection(
            starting_value=float(starting_value),
            monthly_contribution=float(monthly_contribution),
            horizon_years=int(horizon_years),
            goal_amount=float(goal_amount),
        )
        savings_chart_df = _build_main_chart_df(savings_result, horizon_years=int(horizon_years))
        if isinstance(savings_chart_df, pd.DataFrame) and not savings_chart_df.empty and "Median" in savings_chart_df.columns:
            baseline = savings_chart_df[["month", "Median"]].rename(columns={"Median": "Savings-only baseline"})
            out = pd.merge(out, baseline, on="month", how="outer").sort_values("month").reset_index(drop=True)
    return out


def _render_investment_step6(*, show_header: bool = True, show_navigation: bool = True, render_savings_details_after_table: bool = False) -> None:
    if show_header:
        section_header("Long-Term Scenario — Investment/proxy projection")

    # Keep Long-Term Scenario available even before Strategy Engine has been run. If the real engine
    # path is missing, this branch uses a clearly labelled educational proxy and
    # automatically switches to the real Strategy Engine path once available.
    # Keep Long-Term Scenario open by default in the polished demo. The old closed/open
    # gate made the page look empty even after the engine had run.
    st.session_state[PROJECTION_OPEN] = True

    contribution = _resolve_projection_contribution(prefer_plan=False)
    _sync_projection_contribution_state(contribution)
    investment_context = _coerce_mapping(st.session_state.get("investment_context", {}))
    monthly_contribution = _safe_float(contribution.get("monthly"), 0.0)
    weekly_equivalent = _safe_float(contribution.get("weekly"), 0.0)
    profile_value = str(st.session_state.get("investment_philosophy", DEFAULT_STEP6_DEMO_PROFILE) or DEFAULT_STEP6_DEMO_PROFILE)

    # Default Long-Term Scenario to the projection path recommended for the selected
    # philosophy. This is a one-time migration so users can still manually
    # change advanced settings afterwards without the app overwriting them on
    # every rerun.
    recommended_projection = _get_projection_path_recommendation(profile_value)
    recommended_defaults_key = "step6_recommended_projection_defaults_seeded_v1"
    if not bool(st.session_state.get(recommended_defaults_key, False)):
        st.session_state["investment_projection_simulation_mode_label"] = str(recommended_projection["mode_label"])
        st.session_state["investment_projection_simulation_granularity"] = str(recommended_projection["mode"])
        st.session_state["investment_projection_daily_steps_per_month"] = int(recommended_projection["days"])
        st.session_state["investment_projection_daily_path_noise_scale"] = float(recommended_projection["variation"])
        st.session_state[recommended_defaults_key] = True

    investment_context, using_demo_proxy, return_source_label = _projection_context_with_optional_demo_proxy(
        investment_context,
        profile=profile_value,
    )

    # Keep the investment branch simple: evidence/proxy caveats are available
    # inside the How-to-read expander rather than shown as stacked blue boxes.

    # Keep the investment branch simple: long-term life shocks are not exposed
    # here because they add conceptual noise to the core horizon comparison.
    shock_cfg = _long_term_shock_config("None")
    original_monthly_contribution = float(monthly_contribution)
    monthly_shock_drag = 0.0
    weekly_equivalent = monthly_contribution * 12.0 / 52.0 if monthly_contribution > 0 else 0.0

    # The interpretation guide is shown after a projection/comparison exists.
    # Before the run, Long-Term Scenario should stay focused on inputs and the single action.


    if monthly_contribution <= 0.0:
        st.info("The projection is inactive until your monthly contribution is above £0.")
        left, right = st.columns(2)
        with left:
            back_target, back_label = _step6_back_target()
            if st.button(back_label, key="step6_back_zero_contribution", use_container_width=True):
                _go_to_step(back_target)
        with right:
            if st.button("Continue to Insights Summary →", key="step6_continue_zero_contribution", use_container_width=True):
                _go_to_step(7)
        return

    horizon_default = max(1, min(50, _safe_int(st.session_state.get("investment_projection_horizon_years", 1), 1)))
    horizon_years = _safe_int(
        st.slider(
            "Projection horizon (years)",
            min_value=1,
            max_value=50,
            value=horizon_default,
            step=1,
            key="investment_projection_horizon_years",
        ),
        horizon_default,
    )

    goal_key = "investment_projection_goal_amount"
    auto_goal_key = "step6_auto_investment_wealth_goal_value_v1"
    goal_user_touched_key = "step6_investment_wealth_goal_user_touched_v1"
    desired_auto_goal = _default_investment_wealth_goal(monthly_contribution, horizon_years)
    legacy_auto_goal = _default_investment_wealth_goal(monthly_contribution, 20)
    current_goal_value = _safe_float(st.session_state.get(goal_key, 0.0), 0.0)
    previous_auto_goal = _safe_float(st.session_state.get(auto_goal_key, 0.0), 0.0)

    if previous_auto_goal > 0.0:
        if abs(current_goal_value - previous_auto_goal) > 1.0:
            st.session_state[goal_user_touched_key] = True
    elif current_goal_value <= 0.0 or abs(current_goal_value - legacy_auto_goal) <= 1.0:
        st.session_state[goal_user_touched_key] = False
    else:
        st.session_state[goal_user_touched_key] = True

    if not bool(st.session_state.get(goal_user_touched_key, False)):
        if desired_auto_goal > 0.0 and abs(current_goal_value - desired_auto_goal) > 1.0:
            st.session_state[goal_key] = float(desired_auto_goal)
        st.session_state[auto_goal_key] = float(desired_auto_goal)
    # Investment/proxy is now the comparison-oriented Long-Term Scenario branch by default.
    # Migrate older sessions once so the page opens in the intended state even
    # if previous widgets had stored unchecked/empty values.
    investment_defaults_key = "step6_investment_defaults_migrated_v2"
    baseline_key = "step6_show_savings_baseline_on_investment_chart_v1"
    if not bool(st.session_state.get(investment_defaults_key, False)):
        st.session_state[baseline_key] = True
        st.session_state["investment_projection_compare_enabled"] = True
        st.session_state["investment_projection_compare_horizons"] = list(DEFAULT_COMPARE_HORIZONS)
        st.session_state["investment_projection_compare_horizons_widget"] = list(DEFAULT_COMPARE_HORIZONS)
        st.session_state[investment_defaults_key] = True
    elif baseline_key not in st.session_state:
        st.session_state[baseline_key] = True

    # Horizon comparison is the default investment/proxy workflow. The base
    # projection is still calculated internally, but the UI exposes a single
    # action: run the comparison.
    if "investment_projection_compare_enabled" not in st.session_state:
        st.session_state["investment_projection_compare_enabled"] = True
    current_widget_horizons = st.session_state.get("investment_projection_compare_horizons_widget")
    if not isinstance(current_widget_horizons, list) or len(current_widget_horizons) == 0:
        st.session_state["investment_projection_compare_horizons_widget"] = list(DEFAULT_COMPARE_HORIZONS)
    current_stored_horizons = st.session_state.get("investment_projection_compare_horizons")
    if not isinstance(current_stored_horizons, list) or len(current_stored_horizons) == 0:
        st.session_state["investment_projection_compare_horizons"] = list(DEFAULT_COMPARE_HORIZONS)

    # Projection comparison settings are rendered later with the other
    # secondary expanders. The active values are read from session_state here so
    # the visible output can stay above all optional controls.
    current_savings = _safe_float(st.session_state.get("investment_projection_current_savings", 0.0), 0.0)
    goal_amount = _safe_float(st.session_state.get("investment_projection_goal_amount", desired_auto_goal), 0.0)
    compare_enabled = bool(st.session_state.get("investment_projection_compare_enabled", True))
    show_savings_baseline = bool(st.session_state.get(baseline_key, True))

    default_multiselect = st.session_state.get(
        "investment_projection_compare_horizons_widget",
        st.session_state.get("investment_projection_compare_horizons", DEFAULT_COMPARE_HORIZONS),
    )
    if not isinstance(default_multiselect, list) or len(default_multiselect) == 0:
        default_multiselect = list(DEFAULT_COMPARE_HORIZONS)

    available_horizons = [20, 30, 50]
    selected_compare_horizons: List[int] = []
    if compare_enabled:
        selected_compare_horizons = sorted(
            set(int(x) for x in default_multiselect if int(x) in available_horizons)
        )
    st.session_state["investment_projection_compare_horizons"] = selected_compare_horizons

    effective_compare_horizons = (
        sorted(set([int(horizon_years)] + selected_compare_horizons))
        if compare_enabled
        else []
    )

    mc_paths = _safe_int(st.session_state.get("investment_projection_mc_paths", 100), 100)
    simulation_granularity = str(
        st.session_state.get("investment_projection_simulation_granularity", "monthly") or "monthly"
    )
    daily_steps_per_month = _safe_int(
        st.session_state.get("investment_projection_daily_steps_per_month", 21),
        21,
    )
    daily_path_noise_scale = _safe_float(
        st.session_state.get("investment_projection_daily_path_noise_scale", 0.35),
        0.35,
    )

    runtime_payload = build_projection_runtime_payload(st.session_state, investment_context=investment_context)
    runtime_payload = dict(runtime_payload)
    runtime_payload["current_savings"] = current_savings
    runtime_payload["horizon_years"] = horizon_years
    runtime_payload["goal_amount"] = goal_amount
    runtime_payload["profile"] = profile_value
    runtime_payload["monthly_contribution"] = monthly_contribution
    runtime_payload["long_term_life_shock_option"] = str(shock_cfg.get("option", "None"))
    runtime_payload["long_term_life_shock_monthly_drag"] = float(monthly_shock_drag)
    runtime_payload["compare_horizons"] = effective_compare_horizons
    runtime_payload["n_sims"] = mc_paths
    runtime_payload["simulation_granularity"] = simulation_granularity
    runtime_payload["daily_steps_per_month"] = int(daily_steps_per_month)
    runtime_payload["daily_path_noise_scale"] = float(daily_path_noise_scale)

    historical_path_count = _safe_int(runtime_payload.get("historical_path_count"), 0)

    state_payload = build_projection_state_payload(st.session_state)

    preview_bundle = build_projection_workflow_bundle(
        runtime_payload,
        step5_run_result=st.session_state.get("step5_run_result", {}),
        stored_state=state_payload,
        execute_fresh=False,
    )

    projection_compare_is_current = bool(preview_bundle.get("stored_projection_compare_is_current", False))

    def _run_and_store_horizon_comparison() -> bool:
        """Generate the base projection and horizon comparison for the current Long-Term Scenario inputs."""
        fresh_bundle = build_projection_workflow_bundle(
            runtime_payload,
            step5_run_result=st.session_state.get("step5_run_result", {}),
            stored_state=state_payload,
            execute_fresh=True,
        )

        fresh_projection = _coerce_mapping(fresh_bundle.get("fresh_projection", {}))
        if fresh_projection:
            st.session_state[INVESTMENT_PROJECTION_RESULT] = _coerce_mapping(fresh_projection.get("result", {}))
            st.session_state[INVESTMENT_PROJECTION_SIGNATURE] = fresh_projection.get("signature")

        fresh_compare = _coerce_mapping(fresh_bundle.get("fresh_compare", {}))
        compare_results = _coerce_mapping(fresh_compare.get("compare_results", {}))
        st.session_state[INVESTMENT_PROJECTION_COMPARE_RESULTS] = compare_results
        st.session_state[INVESTMENT_PROJECTION_COMPARE_SIGNATURE] = fresh_compare.get("compare_signature")
        return bool(compare_results or fresh_projection)

    auto_generated_comparison = False
    auto_run_available = bool(compare_enabled and selected_compare_horizons)
    if auto_run_available and not projection_compare_is_current:
        with st.spinner("Generating horizon comparison..."):
            auto_generated_comparison = _run_and_store_horizon_comparison()
        projection_compare_is_current = bool(auto_generated_comparison)

    if not compare_enabled:
        comparison_status_message = "Enable horizon comparison to compare multiple horizons."
    elif not selected_compare_horizons:
        comparison_status_message = "Select at least one horizon to compare."
    elif auto_generated_comparison:
        comparison_status_message = "Horizon comparison updated automatically for the current Long-Term Scenario inputs."
    elif projection_compare_is_current:
        comparison_status_message = "Horizon comparison updates automatically when Long-Term Scenario inputs change."
    else:
        comparison_status_message = "Horizon comparison will update automatically once valid inputs are available."

    def _render_projection_comparison_settings() -> None:
        """Render optional Long-Term Scenario comparison controls at the bottom of the page."""
        # Keep this expander open after the advanced-settings checkbox triggers a Streamlit rerun.
        # Without this, ticking "Show advanced..." immediately collapses the parent expander
        # and makes the page jump back toward the top.
        advanced_settings_key = "step6_show_advanced_projection_settings_v1"
        settings_expanded = bool(st.session_state.get(advanced_settings_key, False))
        with st.expander("Adjust projection comparison settings", expanded=settings_expanded):
            st.markdown("**Basic comparison settings**")
            st.caption(
                "These optional inputs change the generated comparison. The current projection updates automatically when these values change."
            )

            scenario_cols = st.columns(2)
            with scenario_cols[0]:
                st.number_input(
                    "Starting investment pot (£)",
                    min_value=0.0,
                    step=100.0,
                    key="investment_projection_current_savings",
                )
            with scenario_cols[1]:
                st.number_input(
                    "Optional wealth goal (£)",
                    min_value=0.0,
                    step=1000.0,
                    key="investment_projection_goal_amount",
                    help="Set a goal above £0 to populate probability of success.",
                )

            if not bool(st.session_state.get(goal_user_touched_key, False)):
                st.caption(
                    f"Auto-suggested goal for a {int(horizon_years)}-year scenario: "
                    f"**{_format_currency(desired_auto_goal)}**. You can edit it."
                )
            else:
                st.caption("Custom wealth goal is being used. Clear or change it manually if needed.")

            check_cols = st.columns(2)
            with check_cols[0]:
                st.checkbox(
                    "Compare across horizons",
                    value=bool(st.session_state.get("investment_projection_compare_enabled", True)),
                    key="investment_projection_compare_enabled",
                    help="Shows the selected investment/proxy path across 20, 30 and 50 years.",
                )
            with check_cols[1]:
                st.checkbox(
                    "Show savings-only baseline on comparison chart",
                    value=bool(st.session_state.get(baseline_key, True)),
                    key=baseline_key,
                    help="Adds the same contribution path with no investment return, volatility or drawdown.",
                )

            current_default_horizons = st.session_state.get(
                "investment_projection_compare_horizons",
                DEFAULT_COMPARE_HORIZONS,
            )
            if not isinstance(current_default_horizons, list) or len(current_default_horizons) == 0:
                current_default_horizons = list(DEFAULT_COMPARE_HORIZONS)

            if bool(st.session_state.get("investment_projection_compare_enabled", True)):
                st.multiselect(
                    "Horizons to compare",
                    options=available_horizons,
                    default=sorted(set(int(x) for x in current_default_horizons if int(x) in available_horizons)),
                    key="investment_projection_compare_horizons_widget",
                    help="Choose the long-term horizons to compare. The current single-horizon result is generated behind the scenes.",
                )

            show_advanced_projection_settings = bool(
                st.checkbox(
                    "Show advanced Monte Carlo/path settings",
                    value=settings_expanded,
                    key=advanced_settings_key,
                    help=(
                        "Reveals simulation-path controls such as Monte Carlo paths, monthly vs hybrid daily path mode, "
                        "synthetic trading days and daily path variation. Leave collapsed unless deliberately testing projection behaviour."
                    ),
                )
            )

            if show_advanced_projection_settings:
                with st.container(border=True):
                    st.markdown("#### Advanced Monte Carlo/path settings")
                    st.number_input(
                        "Monte Carlo paths",
                        min_value=100,
                        max_value=100000,
                        step=100,
                        key="investment_projection_mc_paths",
                    )
                    current_mode = str(st.session_state.get("investment_projection_simulation_granularity", "monthly") or "monthly")
                    simulation_mode_label = st.selectbox(
                        "Projection path mode",
                        options=["Monthly bootstrap", "Hybrid daily simulation"],
                        index=0 if current_mode != "daily_hybrid" else 1,
                        key="investment_projection_simulation_mode_label",
                        help="Hybrid daily simulation preserves the sampled monthly return from the engine, but expands each month into a synthetic daily path for a richer path shape.",
                    )
                    rendered_granularity = "daily_hybrid" if simulation_mode_label == "Hybrid daily simulation" else "monthly"
                    st.session_state["investment_projection_simulation_granularity"] = rendered_granularity

                    if rendered_granularity == "daily_hybrid":
                        rendered_daily_steps = _safe_int(
                            st.number_input(
                                "Synthetic trading days per month",
                                min_value=5,
                                max_value=31,
                                step=1,
                                key="investment_projection_daily_steps_per_month",
                                help="Used only for the hybrid daily projection path. 21 is a sensible default.",
                            ),
                            21,
                        )
                        rendered_daily_noise = _safe_float(
                            st.slider(
                                "Daily path variation within month",
                                min_value=0.0,
                                max_value=1.0,
                                step=0.05,
                                key="investment_projection_daily_path_noise_scale",
                                help="Controls how much day-to-day movement is allowed inside each sampled month while preserving the final monthly return.",
                            ),
                            0.35,
                        )
                        st.caption(
                            "Hybrid daily simulation: the engine still samples monthly OOS returns, but each sampled month is expanded into a synthetic daily path that exactly compounds back to the same monthly return."
                        )
                    else:
                        rendered_daily_steps = _safe_int(st.session_state.get("investment_projection_daily_steps_per_month", 21), 21)
                        rendered_daily_noise = _safe_float(st.session_state.get("investment_projection_daily_path_noise_scale", 0.35), 0.35)
                        st.caption(
                            "Monthly bootstrap: Long-Term Scenario reuses the engine's realised monthly OOS return path directly."
                        )

                    recommendation = _get_projection_path_recommendation(profile_value)
                    coherence_status = _classify_projection_path_coherence(
                        philosophy=profile_value,
                        mode=rendered_granularity,
                        days=rendered_daily_steps,
                        variation=rendered_daily_noise,
                    )

                    if coherence_status["status"] == "success":
                        st.success(coherence_status["title"])
                    elif coherence_status["status"] == "info":
                        st.info(coherence_status["title"])
                    else:
                        st.warning(coherence_status["title"])

                    st.caption(coherence_status["message"])
                    st.info(
                        "Projection coherence suggestion ({philosophy})\n\n"
                        "Recommended:\n"
                        "• {mode_label}\n"
                        "• {days} synthetic trading days/month\n"
                        "• {variation:.2f} daily path variation\n\n"
                        "{rationale}".format(**recommendation)
                    )

                    if st.button(
                        "Apply recommended projection settings",
                        key="step6_apply_projection_recommendation",
                        use_container_width=False,
                    ):
                        queue_and_rerun(
                            {
                                "investment_projection_simulation_mode_label": str(recommendation["mode_label"]),
                                "investment_projection_simulation_granularity": str(recommendation["mode"]),
                                "investment_projection_daily_steps_per_month": int(recommendation["days"]),
                                "investment_projection_daily_path_noise_scale": float(recommendation["variation"]),
                            }
                        )

            if historical_path_count > 0:
                mode_text = "hybrid daily simulation" if str(runtime_payload.get("simulation_granularity", "monthly")) == "daily_hybrid" else "monthly bootstrap"
                if using_demo_proxy:
                    st.caption(f"Return path: Demo proxy · {historical_path_count} monthly returns · {mode_text}.")
                else:
                    st.caption(f"Return path: Historical Strategy Engine path · {historical_path_count} monthly OOS returns · {mode_text}.")
            else:
                st.warning("No return path detected in Long-Term Scenario. Projection may fall back to profile / summary assumptions.")

            st.caption(comparison_status_message)

    projection_result = _coerce_mapping(st.session_state.get(INVESTMENT_PROJECTION_RESULT, {}))
    projection_summary = _extract_projection_summary(projection_result)
    metrics = _extract_metrics(projection_summary, monthly_contribution=monthly_contribution)
    main_chart_df = _build_main_chart_df(projection_result, horizon_years=horizon_years)

    if not projection_summary:
        pass
    else:
        primary_metrics = st.columns(4)
        with primary_metrics[0]:
            st.metric("Monthly contribution used", f"{_format_currency(metrics['monthly_contribution'])}/mo")
        with primary_metrics[1]:
            st.metric("Total contributed", _format_currency(metrics["total_contributed"]))
        with primary_metrics[2]:
            st.metric("Median terminal wealth", _format_currency(metrics["median_terminal"]))
        with primary_metrics[3]:
            st.metric("Expected profit", _format_currency(metrics["expected_profit"]))

        if compare_enabled:
            horizon_bits = [f"{int(horizon_years)}y current"]
            horizon_bits.extend(f"{int(x)}y" for x in selected_compare_horizons)
            horizon_summary = " + ".join(dict.fromkeys(horizon_bits))
        else:
            horizon_summary = f"{int(horizon_years)}y current only"
        baseline_summary = "savings-only baseline on" if show_savings_baseline else "savings-only baseline off"
        goal_summary = "no wealth goal" if goal_amount <= 0.0 else f"{_format_currency(goal_amount)} goal"
        st.caption(
            "Projection setup: "
            f"{_format_currency(current_savings)} starting pot · {goal_summary} · "
            f"{horizon_summary} · {baseline_summary}."
        )
        _render_projection_comparison_settings()

        def _render_detailed_projection_metrics() -> None:
            with st.expander("Detailed projection metrics", expanded=False):
                detail_metrics = st.columns(4)
                with detail_metrics[0]:
                    st.metric("Expected terminal wealth", _format_currency(metrics["expected_terminal"]))
                with detail_metrics[1]:
                    st.metric(
                        "P10–P90 range",
                        f"{_format_currency(metrics['p10_terminal'])} ··· {_format_currency(metrics['p90_terminal'])}",
                    )
                with detail_metrics[2]:
                    st.metric("Loss vs contributions", _format_probability(metrics["loss_probability"]))
                with detail_metrics[3]:
                    goal_display = "—" if goal_amount <= 0.0 else _format_probability(metrics["goal_probability"])
                    st.metric("Goal probability", goal_display)
                st.caption(
                    "Loss vs contributions means the simulated terminal value ends below the total amount contributed. "
                    "Goal probability uses the optional wealth goal, so treat it as a secondary diagnostic unless the goal is meaningful."
                )



    compare_results = _coerce_mapping(st.session_state.get(INVESTMENT_PROJECTION_COMPARE_RESULTS, {}))
    horizon_table_df = _build_horizon_table(
        compare_results,
        horizon_years,
        projection_result,
        selected_compare_horizons=selected_compare_horizons,
    )
    median_paths_chart_df = _build_median_paths_comparison_chart(
        compare_results,
        horizon_years,
        projection_result,
        selected_compare_horizons=selected_compare_horizons,
    )
    median_paths_chart_df = _add_savings_baseline_to_horizon_comparison_chart(
        median_paths_chart_df,
        show_savings_baseline=show_savings_baseline,
        starting_value=current_savings,
        monthly_contribution=monthly_contribution,
        goal_amount=goal_amount,
    )
    terminal_horizon_chart_df = _build_terminal_horizon_chart_df(
        compare_results,
        horizon_years,
        projection_result,
        selected_compare_horizons=selected_compare_horizons,
    )

    if not horizon_table_df.empty:
        st.markdown("### Horizon comparison")

        display_table = horizon_table_df.copy()
        if show_savings_baseline:
            display_table["savings-only baseline"] = display_table["horizon"].apply(
                lambda label: _savings_only_terminal_value(
                    starting_value=current_savings,
                    monthly_contribution=monthly_contribution,
                    horizon_years=_horizon_years_from_label(label),
                )
            )
            ordered_cols = [
                "horizon",
                "median final wealth",
                "savings-only baseline",
                "p10",
                "p90",
                "loss vs contributions",
                "goal probability",
            ]
            display_table = display_table[[col for col in ordered_cols if col in display_table.columns]]
        for col in ("median final wealth", "savings-only baseline", "p10", "p50", "p90"):
            if col in display_table.columns:
                display_table[col] = display_table[col].apply(_format_currency)
        if "loss vs contributions" in display_table.columns:
            display_table["loss vs contributions"] = display_table["loss vs contributions"].apply(_format_probability)
        if "goal probability" in display_table.columns:
            display_table["goal probability"] = display_table["goal probability"].apply(
                lambda x: "—" if goal_amount <= 0.0 else _format_probability(x)
            )
        elif "probability of success" in display_table.columns:
            display_table["probability of success"] = display_table["probability of success"].apply(
                lambda x: "—" if goal_amount <= 0.0 else _format_probability(x)
            )

        if not median_paths_chart_df.empty:
            comparison_chart = _build_horizon_comparison_altair_chart(median_paths_chart_df)
            if comparison_chart is not None:
                st.altair_chart(comparison_chart, use_container_width=True)
            else:
                st.line_chart(median_paths_chart_df.set_index("month"))
            if show_savings_baseline and "Savings-only baseline" in median_paths_chart_df.columns:
                st.caption(
                    "Savings-only baseline assumes the same starting pot and monthly contribution, "
                    "but applies no investment return, volatility or drawdown."
                )
        elif not terminal_horizon_chart_df.empty:
            st.line_chart(terminal_horizon_chart_df.set_index("horizon_years"))

    if projection_summary or not horizon_table_df.empty:
        _render_step6_projection_interpretation_note(
            monthly_contribution=monthly_contribution,
            weekly_equivalent=weekly_equivalent,
            using_demo_proxy=using_demo_proxy,
            profile_value=profile_value,
            return_source_label=return_source_label,
            comparison_table=display_table if not display_table.empty else None,
            goal_amount=goal_amount,
            metrics=metrics if projection_summary else None,
        )

    if render_savings_details_after_table:
        st.markdown("---")
        _render_savings_only_step6(show_header=False, show_navigation=False, persist_result=False)

    if show_navigation:
        left, right = st.columns(2)
        with left:
            back_target, back_label = _step6_back_target()
            if st.button(back_label, key="step6_back", use_container_width=True):
                _go_to_step(back_target)
        with right:
            if st.button("Continue to Insights Summary →", key="step6_continue", use_container_width=True):
                _go_to_step(7)



# ============================================================
# Branch-aware Long-Term Scenario wrapper: savings-only vs investment paths
# ============================================================

STEP0_PATHWAY = "step0_planning_pathway"
CASH_ONLY_PROJECTION_RESULT = "investment_projection_cash_only_result"


def _current_pathway() -> str:
    valid = {"compare_both", "savings_only", "savings_plus_investing"}

    value = str(st.session_state.get(STEP0_PATHWAY, "") or "").strip()
    if value in valid:
        return value

    alias = str(st.session_state.get("selected_planning_pathway", "") or "").strip()
    if alias in valid:
        st.session_state[STEP0_PATHWAY] = alias
        return alias

    radio_label = str(st.session_state.get("step0_pathway_radio", "") or "").lower()
    if "savings-only" in radio_label or "savings only" in radio_label:
        st.session_state[STEP0_PATHWAY] = "savings_only"
        return "savings_only"
    if "compare" in radio_label:
        st.session_state[STEP0_PATHWAY] = "compare_both"
        return "compare_both"
    if "investing" in radio_label:
        st.session_state[STEP0_PATHWAY] = "savings_plus_investing"
        return "savings_plus_investing"

    return "compare_both"


def _build_savings_only_projection(
    *,
    starting_value: float,
    monthly_contribution: float,
    horizon_years: int,
    goal_amount: float,
    monthly_range: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """Build a savings-only projection with low / expected / high contribution bands.

    If monthly_range is provided, the projection uses it as a contribution
    capacity range. Otherwise it falls back to a fixed contribution path.
    """
    months = max(int(horizon_years) * 12, 1)
    month_values = np.arange(0, months + 1)

    range_payload = dict(monthly_range or {})
    conservative_monthly = max(_safe_float(range_payload.get("conservative"), monthly_contribution), 0.0)
    expected_monthly = max(_safe_float(range_payload.get("expected"), monthly_contribution), 0.0)
    high_monthly = max(_safe_float(range_payload.get("high"), monthly_contribution), 0.0)

    p10_values = float(starting_value) + (month_values * conservative_monthly)
    median_values = float(starting_value) + (month_values * expected_monthly)
    p90_values = float(starting_value) + (month_values * high_monthly)

    terminal = float(median_values[-1])
    p10_terminal = float(p10_values[-1])
    p90_terminal = float(p90_values[-1])
    total_contributed = float(starting_value) + expected_monthly * months
    goal_probability = None if goal_amount <= 0 else (1.0 if terminal >= goal_amount else 0.0)

    projection_df = pd.DataFrame(
        {
            "month": month_values,
            "projected_value": median_values,
            "p10": p10_values,
            "median": median_values,
            "p90": p90_values,
        }
    )
    summary = {
        "expected_terminal": terminal,
        "median_terminal": terminal,
        "p10_terminal": p10_terminal,
        "p90_terminal": p90_terminal,
        "final_value": terminal,
        "starting_value": float(starting_value),
        "monthly_contribution": float(expected_monthly),
        "monthly_contribution_conservative": float(conservative_monthly),
        "monthly_contribution_high": float(high_monthly),
        "total_contributed": total_contributed,
        "expected_profit": 0.0,
        "gain_from_growth": 0.0,
        "probability_of_reaching_goal": goal_probability,
        "probability_of_loss_vs_contributions": 0.0,
        "probability_of_finishing_below_initial": 0.0,
        "goal_amount": float(goal_amount),
        "horizon_years": int(horizon_years),
        "risk_profile": "Savings-only",
        "assumed_annual_return": 0.0,
        "interpretation": "contribution_capacity_range",
        "source": "savings_only_pathway",
    }
    return {"summary": summary, "projection_df": projection_df}


def _resolve_savings_weekly_from_plan() -> tuple[float, str]:
    snapshot = _coerce_mapping(st.session_state.get(PLANNING_SNAPSHOT, {}))

    candidates = [
        ("planned weekly target", snapshot.get("target_a_weekly")),
        ("planned weekly target", st.session_state.get(TARGET_A_WEEKLY)),
        ("baseline weekly savings", snapshot.get("baseline_savings_weekly")),
        ("derived weekly savings", st.session_state.get(WEEKLY_SAVINGS_DERIVED)),
    ]

    for label, raw in candidates:
        value = _safe_float(raw, 0.0)
        if value > 0.0:
            return float(value), label

    return 0.0, "no positive savings target found"


def _resolve_savings_goal_from_plan() -> tuple[float, str]:
    snapshot = _coerce_mapping(st.session_state.get(PLANNING_SNAPSHOT, {}))
    candidates = [
        ("short-term goal", snapshot.get("short_term_goal_amount")),
        ("wealth goal", snapshot.get("goal_amount")),
        ("target balance", snapshot.get("target_balance")),
    ]
    for label, raw in candidates:
        value = _safe_float(raw, 0.0)
        if value > 0.0:
            return float(value), label
    return 0.0, "no explicit goal"


def _resolve_savings_capacity_range(
    *,
    fallback_weekly: float,
    fallback_monthly: float,
    monthly_shock_drag: float = 0.0,
) -> Dict[str, Any]:
    """Resolve conservative / expected / high savings capacity.

    Preferred source is the Personal Finance Setup short-term Monte Carlo result. Personal Finance Setup already
    models spending uncertainty over weeks, so Long-Term Scenario converts the final
    conservative / expected / high cash balances into weekly and monthly saving
    capacity bands. If Personal Finance Setup is not available, use a clearly labelled demo range
    around the current contribution rather than a single exact value.
    """
    snapshot = _coerce_mapping(st.session_state.get(PLANNING_SNAPSHOT, {}))

    if snapshot:
        try:
            view_model = build_step3_view_model(snapshot)
            summary = _coerce_mapping(view_model.get("summary", {}))
            payload = _coerce_mapping(view_model.get("payload", {}))
            weeks = max(
                _safe_int(
                    snapshot.get("planning_horizon_weeks", payload.get("weeks", 12)),
                    12,
                ),
                1,
            )
            conservative_final = _safe_float(summary.get("conservative_final"), 0.0)
            expected_final = _safe_float(summary.get("expected_final"), 0.0)
            optimistic_final = _safe_float(summary.get("optimistic_final"), 0.0)

            conservative_weekly = max(conservative_final / float(weeks), 0.0)
            expected_weekly = max(expected_final / float(weeks), 0.0)
            high_weekly = max(optimistic_final / float(weeks), 0.0)

            if expected_weekly > 0.0:
                monthly = {
                    "conservative": max(conservative_weekly * 52.0 / 12.0 - monthly_shock_drag, 0.0),
                    "expected": max(expected_weekly * 52.0 / 12.0 - monthly_shock_drag, 0.0),
                    "high": max(high_weekly * 52.0 / 12.0 - monthly_shock_drag, 0.0),
                }
                weekly = {key: value * 12.0 / 52.0 for key, value in monthly.items()}
                return {
                    "weekly": weekly,
                    "monthly": monthly,
                    "source": "Personal Finance Setup Monte Carlo cash-flow range",
                    "weeks": int(weeks),
                    "fallback_used": False,
                }
        except Exception as exc:
            # Keep Long-Term Scenario robust; the UI below will fall back to demo range.
            st.caption(f"Personal Finance Setup range unavailable for Long-Term Scenario savings band: {exc}")

    expected_monthly = max(_safe_float(fallback_monthly, 0.0) - monthly_shock_drag, 0.0)
    if expected_monthly <= 0.0:
        expected_monthly = max(_safe_float(fallback_weekly, 0.0) * 52.0 / 12.0 - monthly_shock_drag, 0.0)

    monthly = {
        "conservative": expected_monthly * 0.75,
        "expected": expected_monthly,
        "high": expected_monthly * 1.25,
    }
    weekly = {key: value * 12.0 / 52.0 for key, value in monthly.items()}
    return {
        "weekly": weekly,
        "monthly": monthly,
        "source": "Demo contribution range",
        "weeks": 0,
        "fallback_used": True,
    }


def _build_savings_horizon_rows(
    *,
    starting_value: float,
    monthly_contribution: float,
    goal_amount: float,
    horizons: list[int] | None = None,
    monthly_range: Dict[str, float] | None = None,
) -> pd.DataFrame:
    horizons = list(horizons or SAVINGS_ONLY_HORIZONS)
    rows: list[dict[str, Any]] = []
    for years in horizons:
        result = _build_savings_only_projection(
            starting_value=starting_value,
            monthly_contribution=monthly_contribution,
            horizon_years=int(years),
            goal_amount=goal_amount,
            monthly_range=monthly_range,
        )
        summary = _extract_projection_summary(result)
        conservative = _summary_value(summary, "p10_terminal")
        expected = _summary_value(summary, "median_terminal", "expected_terminal", "final_value")
        high = _summary_value(summary, "p90_terminal")
        if goal_amount > 0.0:
            gap = expected - goal_amount
            if gap >= 0:
                interpretation = "Goal reached in expected case"
            elif high >= goal_amount:
                interpretation = "Possible in high case"
            elif abs(gap) <= max(goal_amount * 0.20, 1.0):
                interpretation = "Close / stretch"
            else:
                interpretation = "Likely shortfall"
        else:
            interpretation = "Accumulation range"
        rows.append(
            {
                "Horizon": f"{int(years)} years",
                "Conservative": conservative,
                "Expected": expected,
                "High case": high,
                "Goal gap": expected - goal_amount if goal_amount > 0.0 else None,
                "Interpretation": interpretation,
                "_years": int(years),
            }
        )
    return pd.DataFrame(rows)


def _render_savings_only_step6(*, show_header: bool = True, show_navigation: bool = True, persist_result: bool = True) -> None:
    if show_header:
        section_header("Long-Term Scenario — Savings-only pathway")

    contribution = _resolve_projection_contribution(prefer_plan=True)
    _sync_projection_contribution_state(contribution)
    weekly_saving = _safe_float(contribution.get("weekly"), 0.0)
    base_monthly_saving = _safe_float(contribution.get("monthly"), 0.0)
    source_label = str(contribution.get("source", "Demo default") or "Demo default")
    resolved_goal_amount, resolved_goal_source = _resolve_savings_goal_from_plan()

    # Keep savings-only deliberately simple. Personal Finance Setup already handles short-term
    # fragility; the old long-term shock control added a second, confusing layer
    # of assumptions by converting occasional costs into a monthly drag.
    shock_cfg = _long_term_shock_config("None")
    monthly_shock_drag = 0.0

    capacity_range = _resolve_savings_capacity_range(
        fallback_weekly=weekly_saving,
        fallback_monthly=base_monthly_saving,
        monthly_shock_drag=monthly_shock_drag,
    )
    monthly_range = dict(capacity_range.get("monthly", {}) or {})
    weekly_range = dict(capacity_range.get("weekly", {}) or {})
    monthly_saving = _safe_float(monthly_range.get("expected"), 0.0)
    weekly_saving = _safe_float(weekly_range.get("expected"), 0.0)

    if weekly_saving <= 0.0:
        st.warning(
            "No positive planned saving target was found from Steps 2–3. "
            "Go back and set a weekly target before using the savings-only pathway."
        )
        if st.button("← Back to Personal Finance Setup", key="step6_savings_no_target_back", use_container_width=True):
            _go_to_step(1)
        return

    if "savings_only_starting_value" not in st.session_state:
        st.session_state["savings_only_starting_value"] = 0.0
    if "savings_only_goal_amount" not in st.session_state:
        st.session_state["savings_only_goal_amount"] = float(resolved_goal_amount)

    # Read setup values from session state before rendering the chart. The widgets
    # themselves live in the single details expander below the main outlook, so
    # changing them reruns the page and updates the headline chart cleanly.
    starting_value = _safe_float(st.session_state.get("savings_only_starting_value", 0.0), 0.0)
    goal_amount = _safe_float(st.session_state.get("savings_only_goal_amount", 0.0), 0.0)
    goal_source = "Long-Term Scenario scenario input" if goal_amount > 0.0 else "no explicit goal"

    horizons_df = _build_savings_horizon_rows(
        starting_value=starting_value,
        monthly_contribution=monthly_saving,
        goal_amount=goal_amount,
        horizons=SAVINGS_ONLY_HORIZONS,
        monthly_range=monthly_range,
    )

    expected_parts: list[str] = []
    if not horizons_df.empty:
        for years in SAVINGS_ONLY_HORIZONS:
            subset = horizons_df.loc[horizons_df["_years"] == years]
            if subset.empty:
                continue
            expected_value = float(subset.iloc[0]["Expected"])
            expected_parts.append(f"{int(years)} years: {_format_currency(expected_value)}")

    prefix = f"With {_format_currency(monthly_saving)}/month"
    if starting_value > 0.0:
        prefix += f" and a starting balance of {_format_currency(starting_value)}"

    primary_goal_horizon = int(max(SAVINGS_ONLY_HORIZONS))
    final_primary = float(horizons_df.loc[horizons_df["_years"] == primary_goal_horizon, "Expected"].iloc[0])
    high_primary = float(horizons_df.loc[horizons_df["_years"] == primary_goal_horizon, "High case"].iloc[0])

    with st.expander("Savings-only baseline and feasibility", expanded=False):
        st.info(
            "This section explains the savings-only baseline used in the main horizon comparison above. "
            "It is not another investment projection: it assumes no investment return, no market volatility and no drawdowns."
        )

        if expected_parts:
            st.caption(prefix + ", savings-only accumulation reaches about " + " · ".join(expected_parts) + ".")

        st.markdown("**Savings-only outlook**")
        cards = st.columns(len(SAVINGS_ONLY_HORIZONS))
        for idx, years in enumerate(SAVINGS_ONLY_HORIZONS):
            row = horizons_df.loc[horizons_df["_years"] == years].iloc[0]
            with cards[idx]:
                st.metric(f"{years} expected", _format_currency(float(row["Expected"])))
                st.caption(
                    f"Range: {_format_currency(float(row['Conservative']))}–{_format_currency(float(row['High case']))}"
                )

        chart_df = horizons_df[["_years", "Conservative", "Expected", "High case"]].copy()
        chart_df = chart_df.rename(columns={"_years": "Years"}).set_index("Years")
        st.line_chart(chart_df)

        st.divider()
        with st.container(border=True):
            st.markdown("### Inputs and assumptions")
            st.caption(
                "These values define the savings-only path. "
                f"Saving range source: **{capacity_range.get('source', source_label)}**. "
                f"Base monthly contribution: **{_format_currency(base_monthly_saving)}/month**."
            )

            c0, c1 = st.columns(2)
            with c0:
                st.metric("Monthly saving used", f"{_format_currency(monthly_saving)}/mo")
            with c1:
                st.metric(
                    "Monthly saving range",
                    f"{_format_currency(_safe_float(monthly_range.get('conservative'), 0.0))}–{_format_currency(_safe_float(monthly_range.get('high'), 0.0))}",
                )

            input_cols = st.columns(2)
            with input_cols[0]:
                st.number_input(
                    "Starting savings balance (£)",
                    min_value=0.0,
                    step=100.0,
                    key="savings_only_starting_value",
                    help="Money you already have saved before this long-term pathway starts.",
                )
            with input_cols[1]:
                st.number_input(
                    "Long-term goal (£)",
                    min_value=0.0,
                    step=1000.0,
                    key="savings_only_goal_amount",
                    help="Optional. Set above £0 if you want the table to show goal feasibility.",
                )

            if goal_amount > 0.0:
                st.caption(f"Long-term goal used for this scenario: **{_format_currency(goal_amount)}**.")

        display_df = horizons_df.drop(columns=["_years", "Goal gap"], errors="ignore").copy()
        for col in ["Conservative", "Expected", "High case"]:
            display_df[col] = display_df[col].apply(lambda x: _format_currency(float(x)))
        show_table(display_df, title="Feasibility by horizon")

        st.caption(
            "Personal Finance Setup tests whether the plan works over weeks. Long-Term Scenario extends the same saving behaviour over years, "
            "so the focus here is long-term pathway feasibility rather than short-term cash-flow volatility."
        )

        st.caption("Savings-only pathway does not model investment growth, volatility, drawdowns or market risk.")

    result = _build_savings_only_projection(
        starting_value=starting_value,
        monthly_contribution=monthly_saving,
        horizon_years=primary_goal_horizon,
        goal_amount=goal_amount,
        monthly_range=monthly_range,
    )
    result["summary"]["weekly_contribution"] = float(weekly_saving)
    result["summary"]["base_monthly_contribution_before_shocks"] = float(base_monthly_saving)
    result["summary"]["long_term_life_shock_option"] = "None"
    result["summary"]["long_term_life_shock_monthly_drag"] = 0.0
    result["summary"]["contribution_source"] = str(source_label)
    result["summary"]["goal_source"] = str(goal_source)
    result["summary"]["primary_savings_horizon_years"] = int(primary_goal_horizon)
    result["summary"]["horizon_table"] = horizons_df.to_dict("records")
    result["summary"]["savings_capacity_source"] = str(capacity_range.get("source", ""))
    result["summary"]["monthly_capacity_range"] = dict(monthly_range)
    result["summary"]["weekly_capacity_range"] = dict(weekly_range)
    if persist_result:
        st.session_state[INVESTMENT_PROJECTION_RESULT] = result
        st.session_state[CASH_ONLY_PROJECTION_RESULT] = result
        st.session_state[INVESTMENT_PROJECTION_SIGNATURE] = (
            f"savings_only_pathway_{weekly_saving}_{starting_value}_{goal_amount}_{primary_goal_horizon}"
        )
    else:
        # In the combined Long-Term Scenario screen, keep the savings-only payload available
        # without overwriting the investment/proxy projection state used by the
        # lower comparison section and Insights Summary hand-off.
        st.session_state[CASH_ONLY_PROJECTION_RESULT] = result

    if show_navigation:
        st.markdown("---")
        left, right = st.columns(2)
        with left:
            if st.button("← Back to Personal Finance Setup", key="step6_savings_back", use_container_width=True):
                _go_to_step(1)
        with right:
            if st.button("Continue to Insights Summary →", key="step6_savings_continue", use_container_width=True):
                _go_to_step(7)



def render_step_6() -> None:
    pathway = _current_pathway()

    if pathway == "savings_only":
        _render_savings_only_step6()
        return

    # Existing Long-Term Scenario investment flow. This branch intentionally still requires
    # a Strategy Engine engine run because it depends on investment engine outputs.
    _render_investment_step6()



# ============================================================
# Compare-both branch override
# ============================================================

COMPARE_BRANCH_RESULT = "step6_compare_branch_result"


def _extract_investment_result_for_horizon(compare_results: Dict[str, Any], horizon: int, fallback_current: Dict[str, Any]) -> Dict[str, Any]:
    if int(horizon) == _safe_int(_extract_projection_summary(fallback_current).get("horizon_years"), 0):
        return _coerce_mapping(fallback_current)

    raw = _coerce_mapping(compare_results.get(str(int(horizon)), {}))
    if not raw:
        raw = _coerce_mapping(compare_results.get(int(horizon), {}))
    return _extract_compare_payload(raw) if raw else {}


def _build_compare_rows(
    *,
    horizons: list[int],
    starting_value: float,
    monthly_contribution: float,
    goal_amount: float,
    investment_current_result: Dict[str, Any],
    investment_compare_results: Dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        cash_result = _build_savings_only_projection(
            starting_value=starting_value,
            monthly_contribution=monthly_contribution,
            horizon_years=int(horizon),
            goal_amount=goal_amount,
        )
        cash_summary = _extract_projection_summary(cash_result)
        cash_terminal = _summary_value(cash_summary, "median_terminal", "expected_terminal", "final_value")

        investment_result = _extract_investment_result_for_horizon(
            investment_compare_results,
            int(horizon),
            investment_current_result,
        )
        investment_summary = _extract_projection_summary(investment_result)

        inv_median = _summary_value(investment_summary, "median_terminal", "expected_terminal", "final_value")
        inv_p10 = _summary_value(investment_summary, "p10_terminal", "median_terminal", "expected_terminal", "final_value")
        inv_p90 = _summary_value(investment_summary, "p90_terminal", "median_terminal", "expected_terminal", "final_value")
        diff = inv_median - cash_terminal

        if not investment_summary:
            interpretation = "Run comparison"
        elif diff > max(cash_terminal * 0.10, 500):
            interpretation = "Investing improves median, with risk"
        elif diff > 0:
            interpretation = "Small median improvement"
        else:
            interpretation = "Savings-only competitive"

        rows.append(
            {
                "Horizon": f"{int(horizon)} years",
                "Savings-only": cash_terminal,
                "Investing median": inv_median if investment_summary else None,
                "Investing P10": inv_p10 if investment_summary else None,
                "Investing P90": inv_p90 if investment_summary else None,
                "Difference": diff if investment_summary else None,
                "Interpretation": interpretation,
                "_years": int(horizon),
            }
        )
    return pd.DataFrame(rows)


def _render_compare_step6(*, show_header: bool = True) -> None:
    if show_header:
        section_header("Long-Term Scenario — Pathway comparison")

    # Compare mode can use a labelled demo proxy before Strategy Engine exists. Real
    # Strategy Engine engine returns always override the proxy automatically.
    contribution = _resolve_projection_contribution(prefer_plan=True)
    _sync_projection_contribution_state(contribution)
    investment_context = _coerce_mapping(st.session_state.get("investment_context", {}))
    profile_value = str(st.session_state.get("investment_philosophy", DEFAULT_STEP6_DEMO_PROFILE) or DEFAULT_STEP6_DEMO_PROFILE)
    investment_context, using_demo_proxy, return_source_label = _projection_context_with_optional_demo_proxy(
        investment_context,
        profile=profile_value,
    )
    source_label = str(contribution.get("source", "Demo default") or "Demo default")
    base_monthly_contribution = _safe_float(contribution.get("monthly"), 0.0)
    shock_cfg = _render_long_term_shock_control(key_prefix="compare_pathway")
    monthly_contribution, monthly_shock_drag = _apply_long_term_shock_to_contribution(base_monthly_contribution, shock_cfg)
    weekly_equivalent = monthly_contribution * 12.0 / 52.0 if monthly_contribution > 0 else 0.0

    if using_demo_proxy:
        st.info(
            "Comparison mode is currently comparing savings-only against a **labelled educational investment proxy**. "
            "This is useful for exploration, but it is not a tested engine result. After Risk Profile & Asset Universe and the Strategy Engine are completed, "
            "the real engine return path overrides the proxy automatically."
        )
    else:
        st.info(
            "Compare mode uses the same starting pot, monthly contribution, horizon and goal for both pathways. "
            "The only difference is return assumption: savings-only uses 0% market return; investing uses the Strategy Engine engine projection. "
            "Optional long-term shocks reduce the shared contribution before both pathways are compared."
        )

    if monthly_contribution <= 0.0:
        st.warning("No positive contribution was found from the plan. Go back to Step 2/3 and set a weekly saving target.")
        return

    c0, c1, c2 = st.columns(3)
    with c0:
        current_savings = _safe_float(
            st.number_input(
                "Starting pot used for both paths (£)",
                min_value=0.0,
                step=100.0,
                key="compare_starting_pot",
            )
        )
    with c1:
        goal_amount, goal_source = _resolve_savings_goal_from_plan()
        goal_amount = _safe_float(
            st.number_input(
                "Goal used for both paths (£)",
                min_value=0.0,
                step=1000.0,
                value=float(goal_amount),
                key="compare_goal_amount",
            )
        )
    with c2:
        mc_paths = _safe_int(
            st.number_input(
                "Monte Carlo paths",
                min_value=100,
                max_value=100000,
                step=100,
                value=_safe_int(st.session_state.get("investment_projection_mc_paths", 1000), 1000),
                key="compare_mc_paths",
            ),
            1000,
        )

    capacity_range = _resolve_savings_capacity_range(
        fallback_weekly=weekly_equivalent,
        fallback_monthly=base_monthly_contribution,
        monthly_shock_drag=monthly_shock_drag,
    )
    savings_monthly_range = dict(capacity_range.get("monthly", {}) or {})
    st.caption(
        f"Monthly contribution used for both paths: **{_format_currency(monthly_contribution)}/month** "
        f"(about **{_format_currency(weekly_equivalent)}/week**). "
        f"Base before long-term shocks: **{_format_currency(base_monthly_contribution)}/month**. "
        f"Contribution range source: **{capacity_range.get('source', source_label)}**."
    )

    horizons = st.multiselect(
        "Horizons to compare",
        options=[10, 20, 30, 50],
        default=[10, 20, 30, 50],
        key="compare_branch_horizons",
        help="The comparison is fairest when both pathways use the same horizons.",
    )
    horizons = sorted(set(int(h) for h in horizons if int(h) > 0))
    if not horizons:
        st.warning("Select at least one horizon.")
        return

    primary_horizon = 20 if 20 in horizons else horizons[0]

    runtime_payload = build_projection_runtime_payload(st.session_state, investment_context=investment_context)
    runtime_payload = dict(runtime_payload)
    runtime_payload.update(
        {
            "current_savings": current_savings,
            "horizon_years": int(primary_horizon),
            "goal_amount": goal_amount,
            "profile": profile_value,
            "monthly_contribution": monthly_contribution,
            "long_term_life_shock_option": str(shock_cfg.get("option", "None")),
            "long_term_life_shock_monthly_drag": float(monthly_shock_drag),
            "compare_horizons": horizons,
            "n_sims": mc_paths,
            "simulation_granularity": st.session_state.get("investment_projection_simulation_granularity", "monthly"),
            "daily_steps_per_month": int(st.session_state.get("investment_projection_daily_steps_per_month", 21) or 21),
            "daily_path_noise_scale": float(st.session_state.get("investment_projection_daily_path_noise_scale", 0.35) or 0.35),
        }
    )

    if st.button("Run fair pathway comparison", key="run_fair_pathway_comparison", use_container_width=True):
        state_payload = build_projection_state_payload(st.session_state)
        fresh_bundle = build_projection_workflow_bundle(
            runtime_payload,
            step5_run_result=st.session_state.get("step5_run_result", {}),
            stored_state=state_payload,
            execute_fresh=True,
        )

        fresh_projection = _coerce_mapping(fresh_bundle.get("fresh_projection", {}))
        investment_result = _coerce_mapping(fresh_projection.get("result", {}))
        st.session_state[INVESTMENT_PROJECTION_RESULT] = investment_result
        st.session_state[INVESTMENT_PROJECTION_SIGNATURE] = fresh_projection.get("signature")

        fresh_compare = _coerce_mapping(fresh_bundle.get("fresh_compare", {}))
        investment_compare_results = _coerce_mapping(fresh_compare.get("compare_results", {}))
        st.session_state[INVESTMENT_PROJECTION_COMPARE_RESULTS] = investment_compare_results
        st.session_state[INVESTMENT_PROJECTION_COMPARE_SIGNATURE] = fresh_compare.get("compare_signature")

        cash_result = _build_savings_only_projection(
            starting_value=current_savings,
            monthly_contribution=monthly_contribution,
            horizon_years=int(primary_horizon),
            goal_amount=goal_amount,
            monthly_range=savings_monthly_range,
        )
        cash_rows = _build_savings_horizon_rows(
            starting_value=current_savings,
            monthly_contribution=monthly_contribution,
            goal_amount=goal_amount,
            horizons=horizons,
            monthly_range=savings_monthly_range,
        )
        cash_result["summary"]["weekly_contribution"] = float(weekly_equivalent)
        cash_result["summary"]["base_monthly_contribution_before_shocks"] = float(base_monthly_contribution)
        cash_result["summary"]["long_term_life_shock_option"] = str(shock_cfg.get("option", "None"))
        cash_result["summary"]["long_term_life_shock_monthly_drag"] = float(monthly_shock_drag)
        cash_result["summary"]["contribution_source"] = str(source_label)
        cash_result["summary"]["goal_source"] = "compare goal"
        cash_result["summary"]["horizon_table"] = cash_rows.to_dict("records")
        st.session_state[CASH_ONLY_PROJECTION_RESULT] = cash_result

        compare_rows = _build_compare_rows(
            horizons=horizons,
            starting_value=current_savings,
            monthly_contribution=_safe_float(savings_monthly_range.get("expected"), monthly_contribution),
            goal_amount=goal_amount,
            investment_current_result=investment_result,
            investment_compare_results=investment_compare_results,
        )
        st.session_state[COMPARE_BRANCH_RESULT] = {
            "rows": compare_rows.to_dict("records"),
            "monthly_contribution": monthly_contribution,
            "base_monthly_contribution_before_shocks": base_monthly_contribution,
            "long_term_life_shock_option": str(shock_cfg.get("option", "None")),
            "long_term_life_shock_monthly_drag": float(monthly_shock_drag),
            "weekly_contribution": weekly_equivalent,
            "starting_value": current_savings,
            "goal_amount": goal_amount,
            "profile": profile_value,
            "horizons": horizons,
        }
        st.rerun()

    compare_payload = _coerce_mapping(st.session_state.get(COMPARE_BRANCH_RESULT, {}))
    rows = compare_payload.get("rows", [])
    if not rows:
        st.info("Click **Run fair pathway comparison** to generate the savings-only vs investing table.")
    else:
        st.markdown("### Fair comparison by horizon")
        compare_df = pd.DataFrame(rows)
        display_df = compare_df.drop(columns=["_years"], errors="ignore").copy()
        for col in ["Savings-only", "Investing median", "Investing P10", "Investing P90", "Difference"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: "—" if pd.isna(x) else _format_currency(float(x)))
        show_table(display_df, title=None)

        st.caption(
            "Use this table comparatively: it asks when investing meaningfully changes the outcome, not whether investing is guaranteed to win."
        )

        chart_df = compare_df.copy()
        if {"_years", "Savings-only", "Investing median"}.issubset(chart_df.columns):
            chart_df = chart_df[["_years", "Savings-only", "Investing median"]].rename(
                columns={"_years": "Years"}
            )
            chart_df = chart_df.set_index("Years")
            st.line_chart(chart_df)

    st.markdown("---")
    left, right = st.columns(2)
    with left:
        back_target, back_label = _step6_back_target()
        if st.button(back_label, key="step6_compare_back", use_container_width=True):
            _go_to_step(back_target)
    with right:
        if st.button("Continue to Insights Summary →", key="step6_compare_continue", use_container_width=True):
            _go_to_step(7)


def render_step_6() -> None:
    """Render Long-Term Scenario as one combined scenario screen.

    The page now shows the savings-only accumulation view and the
    investment/proxy comparison view together. No projection-mode selector is
    shown, and no underlying content is deleted; the old branch renderers are
    reused in sequence.
    """
    section_header("Long-Term Scenario")
    st.info(
        "Educational scenario simulator, not financial advice or a forecast. "
        "It turns your contribution plan into possible long-term wealth ranges. "
        "This view keeps the investment/proxy pathway visible alongside the savings-only baseline, "
        "so the two can be compared on the same screen. "
        "Results are scenario ranges, not promised outcomes; past performance or proxy assumptions are not reliable indicators of future results."
    )

    with st.expander("Why compare savings-only with long-term investing?", expanded=False):
        st.markdown(
            "Savings-only planning is the stability baseline: it is simple, liquid and avoids market drawdowns, "
            "but its growth is limited to what the user can contribute. Over long horizons, inflation, large life goals "
            "and retirement needs can reduce the purchasing power of cash savings."
        )
        st.markdown(
            "The investment/proxy path is not presented as a guaranteed better choice. It shows how a disciplined long-term "
            "strategy could change the range of outcomes if the user can tolerate uncertainty, temporary losses and model risk."
        )
        st.caption(
            "Use this comparison to understand trade-offs: cash-flow stability first, then long-term scenario testing. "
            "The app does not recommend investing and does not predict future returns."
        )

    contribution = _resolve_projection_contribution(prefer_plan=False)
    _sync_projection_contribution_state(contribution)

    # With the selector removed, keep the comparison branch open by default.
    # Existing controls still let the user adjust horizons and advanced settings.
    st.session_state["investment_projection_compare_enabled"] = True
    if not isinstance(st.session_state.get("investment_projection_compare_horizons_widget"), list) or not st.session_state.get("investment_projection_compare_horizons_widget"):
        st.session_state["investment_projection_compare_horizons_widget"] = list(DEFAULT_COMPARE_HORIZONS)
    if not isinstance(st.session_state.get("investment_projection_compare_horizons"), list) or not st.session_state.get("investment_projection_compare_horizons"):
        st.session_state["investment_projection_compare_horizons"] = list(DEFAULT_COMPARE_HORIZONS)

    # Evidence notes matter most for investment/proxy branches, but they now live
    # inside the How-to-read expander to keep the page visually clean.
    _render_investment_step6(show_header=False, render_savings_details_after_table=False)
