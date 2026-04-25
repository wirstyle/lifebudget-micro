"""
Step 6 — Long-term Projection
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
try:
    from ui.state.updates import queue_and_rerun
except Exception:  # pragma: no cover
    def queue_and_rerun(updates, *, merge_nested=False):
        for _k, _v in dict(updates or {}).items():
            st.session_state[_k] = _v
        st.rerun()


DEFAULT_COMPARE_HORIZONS = [20, 30, 50]

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
            "This is separate from the Step 3 short-term shock. Step 3 tests fragility over weeks; "
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
                "This is not copied from Step 3 and is not treated as an annual guaranteed shock."
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
                "probability of success": summary.get("probability_of_reaching_goal"),
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

    long_df["horizon_order"] = long_df["series"].str.extract(r"^(\d+)").astype(float)
    long_df["is_endpoint"] = False

    endpoint_idx = long_df.groupby("series")["month"].idxmax()
    long_df.loc[endpoint_idx, "is_endpoint"] = True

    # Draw longer horizons first and shorter ones last so smaller horizons stay more visible.
    line_data = long_df.sort_values(["horizon_order", "month"], ascending=[False, True])

    base = alt.Chart(line_data).encode(
        x=alt.X("month:Q", title="Month"),
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

    chart = (lines + points).properties(height=420)
    return chart


def _render_investment_step6() -> None:
    section_header("Step 6 — Long-horizon contribution projection")

    if not bool(st.session_state.get(ENGINE_HAS_RUN, False)):
        st.warning("Run the engine in Step 5 before using the long-term projection.")
        if st.button("Back to Step 5", key="step6_back_no_engine"):
            st.session_state[CURRENT_STEP] = 5
            st.rerun()
        return

    if not bool(st.session_state.get(PROJECTION_OPEN, False)):
        st.info("Projection view is currently closed.")
        if st.button("Open projection", key="step6_open_projection"):
            st.session_state[PROJECTION_OPEN] = True
            st.rerun()
        return

    investment_context = _coerce_mapping(st.session_state.get("investment_context", {}))
    monthly_contribution = _safe_float(
        st.session_state.get(
            INVESTMENT_MONTHLY_CONTRIBUTION,
            investment_context.get("monthly_contribution", 0.0),
        )
    )
    weekly_equivalent = _safe_float(
        st.session_state.get(
            INVESTMENT_WEEKLY_EQUIVALENT,
            investment_context.get("weekly_equivalent", 0.0),
        )
    )

    st.caption(
        "Educational / research use only (Step 6 projection): This application does not provide financial advice, "
        "investment recommendations, or an offer to buy or sell any financial instrument. Results are experimental, "
        "may not generalise to real-world market conditions, and past performance is not indicative of future results. "
        "Users remain solely responsible for any investment decisions made using this application."
    )
    shock_cfg = _render_long_term_shock_control(key_prefix="investment_projection")
    original_monthly_contribution = float(monthly_contribution)
    monthly_contribution, monthly_shock_drag = _apply_long_term_shock_to_contribution(monthly_contribution, shock_cfg)
    weekly_equivalent = monthly_contribution * 12.0 / 52.0 if monthly_contribution > 0 else 0.0

    st.caption(
        f"This projection uses the Step 4 contribution bridge as the base: **{_format_currency(original_monthly_contribution)}/month**. "
        f"After optional long-term life-shock drag, projected contribution is **{_format_currency(monthly_contribution)}/month** "
        f"(weekly equivalent: **{_format_currency(weekly_equivalent)}/week**)."
    )

    if monthly_contribution <= 0.0:
        st.info("The projection is inactive until your monthly contribution is above £0.")
        left, right = st.columns(2)
        with left:
            if st.button("Back to Step 5", key="step6_back_zero_contribution"):
                st.session_state[CURRENT_STEP] = 5
                st.rerun()
        with right:
            if st.button("Continue to Step 7", key="step6_continue_zero_contribution"):
                st.session_state[CURRENT_STEP] = 7
                st.rerun()
        return

    input_cols = st.columns(3)
    with input_cols[0]:
        current_savings = _safe_float(
            st.number_input(
                "Starting investment pot (£)",
                min_value=0.0,
                step=100.0,
                key="investment_projection_current_savings",
            )
        )
    with input_cols[1]:
        horizon_years = _safe_int(
            st.number_input(
                "Projection horizon (years)",
                min_value=1,
                max_value=80,
                step=1,
                key="investment_projection_horizon_years",
            ),
            20,
        )
    with input_cols[2]:
        goal_amount = _safe_float(
            st.number_input(
                "Optional wealth goal (£)",
                min_value=0.0,
                step=1000.0,
                key="investment_projection_goal_amount",
                help="Set a goal above £0 to populate probability of success.",
            )
        )

    profile_value = str(
        st.session_state.get("investment_philosophy", "Balanced") or "Balanced"
    )

    st.caption(
        f"Projection profile inferred from your current setup: **{profile_value}** · "
        "Return path source: **Historical engine path**"
    )

    with st.expander("Advanced projection settings", expanded=False):
        mc_paths = _safe_int(
            st.number_input(
                "Monte Carlo paths",
                min_value=100,
                max_value=100000,
                step=100,
                key="investment_projection_mc_paths",
            ),
            1000,
        )
        simulation_mode_label = st.selectbox(
            "Projection path mode",
            options=["Monthly bootstrap", "Hybrid daily simulation"],
            index=0 if str(st.session_state.get("investment_projection_simulation_granularity", "monthly") or "monthly") != "daily_hybrid" else 1,
            key="investment_projection_simulation_mode_label",
            help="Hybrid daily simulation preserves the sampled monthly return from the engine, but expands each month into a synthetic daily path for a richer path shape.",
        )
        simulation_granularity = "daily_hybrid" if simulation_mode_label == "Hybrid daily simulation" else "monthly"
        st.session_state["investment_projection_simulation_granularity"] = simulation_granularity

        if simulation_granularity == "daily_hybrid":
            daily_steps_per_month = _safe_int(
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
            daily_path_noise_scale = _safe_float(
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
            daily_steps_per_month = _safe_int(st.session_state.get("investment_projection_daily_steps_per_month", 21), 21)
            daily_path_noise_scale = _safe_float(st.session_state.get("investment_projection_daily_path_noise_scale", 0.35), 0.35)
            st.caption(
                "Monthly bootstrap: Step 6 reuses the engine's realised monthly OOS return path directly."
            )

        recommendation = _get_projection_path_recommendation(profile_value)
        coherence_status = _classify_projection_path_coherence(
            philosophy=profile_value,
            mode=simulation_granularity,
            days=daily_steps_per_month,
            variation=daily_path_noise_scale,
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

    compare_enabled = bool(
        st.checkbox(
            "Compare across horizons",
            value=bool(st.session_state.get("investment_projection_compare_enabled", True)),
            key="investment_projection_compare_enabled",
        )
    )

    default_multiselect = st.session_state.get("investment_projection_compare_horizons", DEFAULT_COMPARE_HORIZONS)
    if not isinstance(default_multiselect, list):
        default_multiselect = list(DEFAULT_COMPARE_HORIZONS)

    available_horizons = [20, 30, 50]

    selected_compare_horizons: List[int] = []
    if compare_enabled:
        selected_compare_horizons = st.multiselect(
            "Horizons to compare",
            options=available_horizons,
            default=sorted(set(int(x) for x in default_multiselect if int(x) != horizon_years)),
            key="investment_projection_compare_horizons_widget",
            help="The current projection horizon is always included automatically in the outputs below.",
        )
        selected_compare_horizons = sorted(
            set(int(x) for x in selected_compare_horizons if int(x) > 0 and int(x) != horizon_years)
        )

    st.session_state["investment_projection_compare_horizons"] = selected_compare_horizons

    effective_compare_horizons = sorted(set([int(horizon_years)] + selected_compare_horizons)) if compare_enabled else []

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
    runtime_payload["simulation_granularity"] = st.session_state.get("investment_projection_simulation_granularity", "monthly")
    runtime_payload["daily_steps_per_month"] = int(st.session_state.get("investment_projection_daily_steps_per_month", 21) or 21)
    runtime_payload["daily_path_noise_scale"] = float(st.session_state.get("investment_projection_daily_path_noise_scale", 0.35) or 0.35)

    historical_path_count = _safe_int(runtime_payload.get("historical_path_count"), 0)
    if historical_path_count > 0:
        mode_text = "hybrid daily simulation" if str(runtime_payload.get("simulation_granularity", "monthly")) == "daily_hybrid" else "monthly bootstrap"
        st.caption(f"Historical engine path detected: {historical_path_count} monthly OOS returns available for projection · mode: {mode_text}.")
    else:
        st.warning("No historical engine return path detected in Step 6. Projection may fall back to profile / summary assumptions.")

    state_payload = build_projection_state_payload(st.session_state)

    preview_bundle = build_projection_workflow_bundle(
        runtime_payload,
        step5_run_result=st.session_state.get("step5_run_result", {}),
        stored_state=state_payload,
        execute_fresh=False,
    )

    projection_result_is_current = bool(preview_bundle.get("stored_projection_is_current", False))
    projection_compare_is_current = bool(preview_bundle.get("stored_projection_compare_is_current", False))
    comparison_requires_base_projection = not projection_result_is_current

    action_cols = st.columns([1.2, 1.2, 2.6])
    with action_cols[0]:
        run_projection_clicked = st.button(
            "Run long-term projection",
            key="run_long_term_projection_button",
            use_container_width=True,
        )
    with action_cols[1]:
        run_comparison_clicked = st.button(
            "Run horizon comparison",
            key="run_horizon_comparison_button",
            use_container_width=True,
            disabled=(not compare_enabled) or comparison_requires_base_projection or (len(selected_compare_horizons) == 0),
        )
    with action_cols[2]:
        if not compare_enabled:
            st.caption("Enable horizon comparison to compare multiple horizons.")
        elif comparison_requires_base_projection:
            st.caption(
                "Run the base long-term projection first. Then horizon comparison will use that current Step 6 result as the anchor."
            )
        elif not selected_compare_horizons:
            st.caption("Select at least one extra horizon to compare.")
        elif projection_compare_is_current:
            st.caption("Projection results below match the current Step 6 inputs.")
        else:
            st.caption("Base projection is current. Horizon comparison is ready to run.")

    if run_projection_clicked:
        fresh_bundle = build_projection_workflow_bundle(
            runtime_payload,
            step5_run_result=st.session_state.get("step5_run_result", {}),
            stored_state=state_payload,
            execute_fresh=True,
        )
        fresh_projection = _coerce_mapping(fresh_bundle.get("fresh_projection", {}))
        st.session_state[INVESTMENT_PROJECTION_RESULT] = _coerce_mapping(fresh_projection.get("result", {}))
        st.session_state[INVESTMENT_PROJECTION_SIGNATURE] = fresh_projection.get("signature")

    if run_comparison_clicked and compare_enabled and not comparison_requires_base_projection and selected_compare_horizons:
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

    projection_result = _coerce_mapping(st.session_state.get(INVESTMENT_PROJECTION_RESULT, {}))
    projection_summary = _extract_projection_summary(projection_result)
    metrics = _extract_metrics(projection_summary, monthly_contribution=monthly_contribution)
    main_chart_df = _build_main_chart_df(projection_result, horizon_years=horizon_years)

    if not projection_summary:
        st.info("Configure the Step 6 inputs and click **Run long-term projection** to generate the Monte Carlo result.")
    else:
        top_metrics_1 = st.columns(4)
        with top_metrics_1[0]:
            st.metric("Expected terminal wealth", _format_currency(metrics["expected_terminal"]))
        with top_metrics_1[1]:
            st.metric("Median terminal wealth", _format_currency(metrics["median_terminal"]))
        with top_metrics_1[2]:
            st.metric(
                "P10–P90 range",
                f"{_format_currency(metrics['p10_terminal'])} ··· {_format_currency(metrics['p90_terminal'])}",
            )
        with top_metrics_1[3]:
            goal_display = "—" if goal_amount <= 0.0 else _format_probability(metrics["goal_probability"])
            st.metric("Goal probability", goal_display)

        top_metrics_2 = st.columns(4)
        with top_metrics_2[0]:
            st.metric("Monthly contribution used", f"{_format_currency(metrics['monthly_contribution'])}/mo")
        with top_metrics_2[1]:
            st.metric("Total contributed", _format_currency(metrics["total_contributed"]))
        with top_metrics_2[2]:
            st.metric("Expected profit", _format_currency(metrics["expected_profit"]))
        with top_metrics_2[3]:
            st.metric("Loss vs contributions", _format_probability(metrics["loss_probability"]))

        st.caption(
            "Projection source: historical engine OOS returns. Step 6 now reads stored engine returns from "
            "investment_context first, while the monthly contribution is wired directly from Step 4 via the same context."
        )

        if not main_chart_df.empty:
            st.line_chart(main_chart_df.set_index("month"))

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
    terminal_horizon_chart_df = _build_terminal_horizon_chart_df(
        compare_results,
        horizon_years,
        projection_result,
        selected_compare_horizons=selected_compare_horizons,
    )

    if not horizon_table_df.empty:
        st.markdown("### Horizon comparison")

        display_table = horizon_table_df.copy()
        for col in ("median final wealth", "p10", "p50", "p90"):
            display_table[col] = display_table[col].apply(_format_currency)
        display_table["probability of success"] = display_table["probability of success"].apply(
            lambda x: "—" if goal_amount <= 0.0 else _format_probability(x)
        )
        show_table(display_table, title=None)

        if goal_amount <= 0.0:
            st.caption("Probability of success uses the optional wealth goal. Set a goal above £0 to populate that column.")

        if not median_paths_chart_df.empty:
            comparison_chart = _build_horizon_comparison_altair_chart(median_paths_chart_df)
            if comparison_chart is not None:
                st.altair_chart(comparison_chart, use_container_width=True)
            else:
                st.line_chart(median_paths_chart_df.set_index("month"))
        elif not terminal_horizon_chart_df.empty:
            st.line_chart(terminal_horizon_chart_df.set_index("horizon_years"))

    st.caption(
        "MVP note: this version only adjusts discretionary automatically. Essentials are treated as fixed in the model."
    )

    left, right = st.columns(2)
    with left:
        if st.button("Back to Step 5", key="step6_back"):
            st.session_state[CURRENT_STEP] = 5
            st.rerun()
    with right:
        if st.button("Continue to Step 7", key="step6_continue"):
            st.session_state[CURRENT_STEP] = 7
            st.rerun()



# ============================================================
# Branch-aware Step 6 wrapper: savings-only vs investment paths
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
) -> Dict[str, Any]:
    months = max(int(horizon_years) * 12, 1)
    month_values = np.arange(0, months + 1)
    values = float(starting_value) + (month_values * float(monthly_contribution))
    total_contributed = float(starting_value) + float(monthly_contribution) * months
    terminal = float(values[-1])
    goal_probability = None if goal_amount <= 0 else (1.0 if terminal >= goal_amount else 0.0)

    projection_df = pd.DataFrame({"month": month_values, "projected_value": values})
    summary = {
        "expected_terminal": terminal,
        "median_terminal": terminal,
        "p10_terminal": terminal,
        "p90_terminal": terminal,
        "final_value": terminal,
        "starting_value": float(starting_value),
        "monthly_contribution": float(monthly_contribution),
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
        "interpretation": "contribution_only",
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


def _build_savings_horizon_rows(
    *,
    starting_value: float,
    monthly_contribution: float,
    goal_amount: float,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    horizons = list(horizons or [5, 10, 20])
    rows: list[dict[str, Any]] = []
    for years in horizons:
        result = _build_savings_only_projection(
            starting_value=starting_value,
            monthly_contribution=monthly_contribution,
            horizon_years=int(years),
            goal_amount=goal_amount,
        )
        summary = _extract_projection_summary(result)
        terminal = _summary_value(summary, "median_terminal", "expected_terminal", "final_value")
        if goal_amount > 0.0:
            gap = terminal - goal_amount
            if gap >= 0:
                interpretation = "Goal reached"
            elif abs(gap) <= max(goal_amount * 0.20, 1.0):
                interpretation = "Close / stretch"
            else:
                interpretation = "Likely shortfall"
        else:
            interpretation = "Accumulation potential"
        rows.append(
            {
                "Horizon": f"{int(years)} years",
                "Total saved": terminal,
                "Goal gap": terminal - goal_amount if goal_amount > 0.0 else None,
                "Interpretation": interpretation,
                "_years": int(years),
            }
        )
    return pd.DataFrame(rows)


def _render_savings_only_step6() -> None:
    section_header("Step 6 — Long-term savings pathway")

    weekly_saving, source_label = _resolve_savings_weekly_from_plan()
    base_monthly_saving = float(weekly_saving) * 52.0 / 12.0
    goal_amount, goal_source = _resolve_savings_goal_from_plan()
    starting_value = _safe_float(st.session_state.get("savings_only_starting_value", 0.0), 0.0)

    st.info(
        "Savings-only mode translates your short-term weekly plan into a long-term cash pathway. "
        "It skips the investment engine and assumes no market return."
    )

    shock_cfg = _render_long_term_shock_control(key_prefix="savings_only")
    monthly_saving, monthly_shock_drag = _apply_long_term_shock_to_contribution(base_monthly_saving, shock_cfg)
    weekly_saving = monthly_saving * 12.0 / 52.0 if monthly_saving > 0 else 0.0

    c0, c1, c2 = st.columns(3)
    with c0:
        st.metric("Monthly saving used", f"{_format_currency(monthly_saving)}/mo")
    with c1:
        st.metric("Weekly equivalent", f"{_format_currency(weekly_saving)}/week")
    with c2:
        st.metric("Goal used", "—" if goal_amount <= 0.0 else _format_currency(goal_amount))

    st.caption(
        f"Saving amount source: **{source_label}**. "
        f"Base monthly saving before long-term shocks: **{_format_currency(base_monthly_saving)}/month**. "
        f"Goal source: **{goal_source}**."
    )

    if weekly_saving <= 0.0:
        st.warning(
            "No positive planned saving target was found from Steps 2–3. "
            "Go back and set a weekly target before using the savings-only pathway."
        )
        if st.button("Back to Step 3", key="step6_savings_no_target_back"):
            st.session_state[CURRENT_STEP] = 3
            st.session_state["current_step"] = 3
            st.rerun()
        return

    with st.expander("Optional: add current savings pot", expanded=False):
        starting_value = _safe_float(
            st.number_input(
                "Current savings pot (£)",
                min_value=0.0,
                step=100.0,
                value=starting_value,
                key="savings_only_starting_value",
            )
        )

    horizons_df = _build_savings_horizon_rows(
        starting_value=starting_value,
        monthly_contribution=monthly_saving,
        goal_amount=goal_amount,
        horizons=[5, 10, 20],
    )

    st.markdown("### Long-term savings outlook")
    cards = st.columns(3)
    for idx, years in enumerate([5, 10, 20]):
        row = horizons_df.loc[horizons_df["_years"] == years].iloc[0]
        with cards[idx]:
            st.metric(f"{years} years", _format_currency(float(row["Total saved"])))
            st.caption(str(row["Interpretation"]))

    display_df = horizons_df.drop(columns=["_years"]).copy()
    display_df["Total saved"] = display_df["Total saved"].apply(lambda x: _format_currency(float(x)))
    if "Goal gap" in display_df.columns:
        display_df["Goal gap"] = display_df["Goal gap"].apply(
            lambda x: "—" if pd.isna(x) else _format_currency(float(x))
        )
    show_table(display_df, title="Feasibility by horizon")

    st.caption(
        "Step 3 tests whether the plan works over weeks. Step 6 extends the same saving behaviour over years, "
        "so the focus here is long-term pathway feasibility rather than short-term cash-flow volatility."
    )

    final_20 = float(horizons_df.loc[horizons_df["_years"] == 20, "Total saved"].iloc[0])
    result = _build_savings_only_projection(
        starting_value=starting_value,
        monthly_contribution=monthly_saving,
        horizon_years=20,
        goal_amount=goal_amount,
    )
    result["summary"]["weekly_contribution"] = float(weekly_saving)
    result["summary"]["base_monthly_contribution_before_shocks"] = float(base_monthly_saving)
    result["summary"]["long_term_life_shock_option"] = str(shock_cfg.get("option", "None"))
    result["summary"]["long_term_life_shock_monthly_drag"] = float(monthly_shock_drag)
    result["summary"]["contribution_source"] = str(source_label)
    result["summary"]["goal_source"] = str(goal_source)
    result["summary"]["horizon_table"] = horizons_df.to_dict("records")
    st.session_state[INVESTMENT_PROJECTION_RESULT] = result
    st.session_state[CASH_ONLY_PROJECTION_RESULT] = result
    st.session_state[INVESTMENT_PROJECTION_SIGNATURE] = (
        f"savings_only_pathway_{weekly_saving}_{starting_value}_{goal_amount}_{shock_cfg.get('option', 'None')}"
    )

    st.markdown("### Can savings alone reach the goal?")
    if goal_amount <= 0.0:
        st.info("No explicit long-term goal was found, so this branch shows savings accumulation potential.")
    elif final_20 >= goal_amount:
        st.success("Savings alone reaches the goal within the 20-year view under these assumptions.")
    elif final_20 >= goal_amount * 0.80:
        st.info("Savings alone gets close, but the goal is still a stretch under these assumptions.")
    else:
        st.warning(
            "Savings alone appears unlikely to reach the goal within the 20-year view. "
            "This does not mean you should invest; it means the comparison branch may be useful to test whether investment risk changes the gap."
        )

    st.caption("Savings-only pathway does not model investment growth, volatility, drawdowns or market risk.")

    st.markdown("---")
    left, right = st.columns(2)
    with left:
        if st.button("Back to Step 3", key="step6_savings_back"):
            st.session_state[CURRENT_STEP] = 3
            st.session_state["current_step"] = 3
            st.rerun()
    with right:
        if st.button("Continue to Step 7", key="step6_savings_continue"):
            st.session_state[CURRENT_STEP] = 7
            st.session_state["current_step"] = 7
            st.rerun()


def render_step_6() -> None:
    pathway = _current_pathway()

    if pathway == "savings_only":
        _render_savings_only_step6()
        return

    # Existing Step 6 investment flow. This branch intentionally still requires
    # a Step 5 engine run because it depends on investment engine outputs.
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


def _render_compare_step6() -> None:
    section_header("Step 6 — Long-term pathway comparison")

    if not bool(st.session_state.get(ENGINE_HAS_RUN, False)):
        st.warning("Run the engine in Step 5 before comparing savings-only and investing pathways.")
        if st.button("Back to Step 5", key="step6_compare_back_no_engine"):
            st.session_state[CURRENT_STEP] = 5
            st.session_state["current_step"] = 5
            st.rerun()
        return

    investment_context = _coerce_mapping(st.session_state.get("investment_context", {}))
    weekly_saving, source_label = _resolve_savings_weekly_from_plan()
    monthly_from_plan = float(weekly_saving) * 52.0 / 12.0
    monthly_from_investment_bridge = _safe_float(
        st.session_state.get(
            INVESTMENT_MONTHLY_CONTRIBUTION,
            investment_context.get("monthly_contribution", 0.0),
        )
    )
    base_monthly_contribution = monthly_from_plan if monthly_from_plan > 0 else monthly_from_investment_bridge
    shock_cfg = _render_long_term_shock_control(key_prefix="compare_pathway")
    monthly_contribution, monthly_shock_drag = _apply_long_term_shock_to_contribution(base_monthly_contribution, shock_cfg)
    weekly_equivalent = monthly_contribution * 12.0 / 52.0 if monthly_contribution > 0 else 0.0

    st.info(
        "Compare mode uses the same starting pot, monthly contribution, horizon and goal for both pathways. "
        "The only difference is return assumption: savings-only uses 0% market return; investing uses the Step 5 engine projection. "
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

    st.caption(
        f"Monthly contribution used for both paths: **{_format_currency(monthly_contribution)}/month** "
        f"(about **{_format_currency(weekly_equivalent)}/week**). "
        f"Base before long-term shocks: **{_format_currency(base_monthly_contribution)}/month**. Source: **{source_label}**."
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

    profile_value = str(st.session_state.get("investment_philosophy", "Balanced") or "Balanced")
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
        )
        cash_rows = _build_savings_horizon_rows(
            starting_value=current_savings,
            monthly_contribution=monthly_contribution,
            goal_amount=goal_amount,
            horizons=horizons,
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
            monthly_contribution=monthly_contribution,
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
        if st.button("Back to Step 5", key="step6_compare_back"):
            st.session_state[CURRENT_STEP] = 5
            st.session_state["current_step"] = 5
            st.rerun()
    with right:
        if st.button("Continue to Step 7", key="step6_compare_continue"):
            st.session_state[CURRENT_STEP] = 7
            st.session_state["current_step"] = 7
            st.rerun()


def render_step_6() -> None:
    pathway = _current_pathway()

    if pathway == "savings_only":
        _render_savings_only_step6()
        return

    if pathway == "compare_both":
        _render_compare_step6()
        return

    _render_investment_step6()
