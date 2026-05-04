"""
Short-term feasibility service for the Personal Finance Planner.

This module builds the Step 3 view model used by both the legacy Step 3 page
and the combined Personal Finance Planner dashboard. It turns the confirmed
planning snapshot into a baseline cash-flow path, a simulated target-plan path,
summary metrics, explanatory text, and next-action guidance.

The renderer should treat this service as the source of truth for feasibility
outputs. The chart and cards should only display the returned view model rather
than rebuilding scenario logic directly.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.baseline import generate_baseline, validate_baseline_df
from src.compounder import apply_shock_map_to_df, simulate_scenario_df
from src.expenses import events_to_weekly_shock_map
from src.explain import (
    build_step3_next_actions_markdown,
    build_step3_technical_details_markdown,
    build_structural_deficit_tips,
)
from ui.state.keys import (
    DISC_W,
    DISCRETIONARY_W,
    FIXED_ITEMS_ROWS_WEEKLY,
    FIXED_TOTAL_W,
    INCOME_W,
    PLANNING_SNAPSHOT,
    VARIABLE_ITEMS_ROWS_WEEKLY,
    VARIABLE_PARTS,
    WEEKLY_MARGIN,
)

PRESET_CFG = {
    "Quick estimate (default)": {"iterations": 150, "variability_frac": 0.30},
    "Typical spending": {"iterations": 250, "variability_frac": 0.30},
    "Unpredictable weeks": {"iterations": 250, "variability_frac": 0.50},
    "Stress test": {"iterations": 600, "variability_frac": 0.50},
}

INTENT_LABELS = {
    "not_sure_yet": "Not sure yet",
    "avoid_overspending": "Avoid overspending",
    "save_more_each_week": "Save more each week",
    "reach_target_balance": "Reach a target balance",
}


def coerce_snapshot() -> dict:
    """Return the current planning snapshot as a plain dictionary."""
    snapshot = st.session_state.get(PLANNING_SNAPSHOT, {}) or {}
    return dict(snapshot) if isinstance(snapshot, dict) else {}


def safe_float(value, default: float = 0.0) -> float:
    """Convert a value to float without letting bad UI/session values crash."""
    try:
        return float(value)
    except Exception:
        return float(default)


def safe_int(value, default: int = 0) -> int:
    """Convert a value to int without letting bad UI/session values crash."""
    try:
        return int(value)
    except Exception:
        return int(default)


def monthly_to_weekly(value: float) -> float:
    """Convert a monthly amount into an approximate weekly equivalent."""
    return float(value) * 12.0 / 52.0


def resolve_intent(snapshot: dict) -> str:
    """Resolve the current planning intent stored in the snapshot."""
    value = str(snapshot.get("user_intent", "not_sure_yet") or "not_sure_yet")
    return value if value in INTENT_LABELS else "not_sure_yet"


def resolve_simulation_inputs(snapshot: dict) -> dict:
    """Convert the planning snapshot into weekly simulation inputs.

    The snapshot stores canonical monthly values, while the short-term
    feasibility engine works in weeks. This helper resolves the uncertainty
    preset, target gap, discretionary cut, and random seed used by the scenario
    simulation.
    """
    planning_horizon = max(safe_int(snapshot.get("planning_horizon_weeks", 12), 12), 1)
    uncertainty_preset = str(snapshot.get("uncertainty_preset", "Quick estimate (default)") or "Quick estimate (default)")
    preset_cfg = dict(PRESET_CFG.get(uncertainty_preset, PRESET_CFG["Quick estimate (default)"]))

    monthly_income = safe_float(snapshot.get("monthly_income", 0.0), 0.0)
    fixed_essentials_monthly = safe_float(snapshot.get("fixed_essentials_monthly", 0.0), 0.0)
    variable_essentials_monthly = safe_float(snapshot.get("variable_essentials_monthly", 0.0), 0.0)
    discretionary_monthly = safe_float(snapshot.get("discretionary_spending_monthly", 0.0), 0.0)

    income_w = monthly_to_weekly(monthly_income)
    fixed_w = monthly_to_weekly(fixed_essentials_monthly)
    variable_essentials_w = monthly_to_weekly(variable_essentials_monthly)
    discretionary_w = monthly_to_weekly(discretionary_monthly)

    fixed_total_w = float(fixed_w + variable_essentials_w)

    # For simulation purposes, fixed essentials stay in the fixed bucket while
    # variable essentials and discretionary spending share the variable bucket.
    # This lets the uncertainty model vary the spending categories that are less
    # predictable week to week.
    variable_bucket_w = float(variable_essentials_w + discretionary_w)

    margin_w = safe_float(snapshot.get("baseline_savings_weekly", 0.0), 0.0)
    target_a = max(safe_float(snapshot.get("target_a_weekly", 0.0), 0.0), 0.0)
    need_w = max(float(target_a) - max(float(margin_w), 0.0), 0.0)
    cut_a = min(float(need_w), max(float(discretionary_w), 0.0))
    seed = 42 + max(safe_int(snapshot.get("random_run_nonce", 0), 0), 0)

    return {
        "weeks": int(planning_horizon),
        "uncertainty_preset": uncertainty_preset,
        "iterations": int(preset_cfg["iterations"]),
        "variability_frac": float(preset_cfg["variability_frac"]),
        "income_w": float(income_w),
        "fixed_w": float(fixed_w),
        "variable_essentials_w": float(variable_essentials_w),
        "fixed_total_w": float(fixed_total_w),
        "variable_bucket_w": float(variable_bucket_w),
        "discretionary_w": float(discretionary_w),
        "margin_w": float(margin_w),
        "target_a": float(target_a),
        "cut_a": float(cut_a),
        "seed": int(seed),
        "structural_deficit": bool(snapshot.get("structural_deficit", False)),
    }


def build_shock_map(snapshot: dict) -> dict[int, float]:
    """Build a weekly shock map from cleaned one-off events in the snapshot."""
    if not bool(snapshot.get("enable_one_off_events", False)):
        return {}

    raw_events = snapshot.get("one_off_events", []) or []
    normalized_events = []

    for row in raw_events:
        if not isinstance(row, dict):
            continue

        normalized_events.append(
            {
                "name": str(row.get("name", row.get("Event (optional)", "")) or ""),
                "amount": safe_float(row.get("amount", row.get("Amount (£)", 0.0)), 0.0),
                "week": safe_int(row.get("week", row.get("Week", 1)), 1),
            }
        )

    return events_to_weekly_shock_map(normalized_events)


def build_projection_payload(snapshot: dict) -> dict:
    """Build baseline and target-plan dataframes for the short-term chart.

    The baseline path represents the confirmed budget before target adjustments.
    The target-plan path uses the selected savings target and uncertainty preset.
    One-off events are applied to both paths so the comparison remains consistent.
    """
    sim = resolve_simulation_inputs(snapshot)
    shock_map = build_shock_map(snapshot)

    baseline_df = generate_baseline(
        income=float(sim["income_w"]),
        fixed_expenses=float(sim["fixed_w"]),
        variable_expenses=float(sim["variable_bucket_w"]),
        weeks=int(sim["weeks"]),
    )
    validate_baseline_df(baseline_df)

    if shock_map:
        baseline_df = apply_shock_map_to_df(baseline_df, shock_map=shock_map, value_cols=("Balance",))

    plan_a_df, params_a = simulate_scenario_df(
        income=float(sim["income_w"]),
        fixed_expenses=float(sim["fixed_w"]),
        variable_expenses=float(sim["variable_bucket_w"]),
        delta_savings=float(sim["cut_a"]),
        weeks=int(sim["weeks"]),
        iterations=int(sim["iterations"]),
        seed=int(sim["seed"]),
        variability_frac=float(sim["variability_frac"]),
    )

    if shock_map:
        plan_a_df = apply_shock_map_to_df(plan_a_df, shock_map=shock_map, value_cols=("Mean", "Lower", "Upper"))

    return {
        "baseline_df": baseline_df,
        "plan_a_df": plan_a_df,
        "params_a": params_a,
        "shock_map": shock_map,
        **sim,
    }


def build_technical_summary(baseline_df: pd.DataFrame, plan_a_df: pd.DataFrame) -> dict:
    """Summarise final baseline and simulated target-plan outcomes.

    The dashboard maps ``Lower`` / ``Mean`` / ``Upper`` to
    Conservative / Expected / High case.
    """
    if baseline_df.empty or plan_a_df.empty:
        return {
            "baseline_final": 0.0,
            "conservative_final": 0.0,
            "expected_final": 0.0,
            "optimistic_final": 0.0,
            "expected_vs_baseline_abs": 0.0,
            "expected_vs_baseline_pct": 0.0,
            "conservative_vs_baseline_abs": 0.0,
            "conservative_vs_baseline_pct": 0.0,
            "optimistic_vs_baseline_abs": 0.0,
            "optimistic_vs_baseline_pct": 0.0,
            "range_width": 0.0,
        }

    baseline_final = safe_float(baseline_df["Balance"].iloc[-1], 0.0)
    conservative_final = safe_float(plan_a_df["Lower"].iloc[-1], 0.0)
    expected_final = safe_float(plan_a_df["Mean"].iloc[-1], 0.0)
    optimistic_final = safe_float(plan_a_df["Upper"].iloc[-1], 0.0)

    def delta_pct(value: float, base: float) -> float:
        if abs(base) <= 1e-12:
            return 0.0
        return 100.0 * (value - base) / base

    return {
        "baseline_final": baseline_final,
        "conservative_final": conservative_final,
        "expected_final": expected_final,
        "optimistic_final": optimistic_final,
        "expected_vs_baseline_abs": expected_final - baseline_final,
        "expected_vs_baseline_pct": delta_pct(expected_final, baseline_final),
        "conservative_vs_baseline_abs": conservative_final - baseline_final,
        "conservative_vs_baseline_pct": delta_pct(conservative_final, baseline_final),
        "optimistic_vs_baseline_abs": optimistic_final - baseline_final,
        "optimistic_vs_baseline_pct": delta_pct(optimistic_final, baseline_final),
        "range_width": optimistic_final - conservative_final,
    }


def build_structural_deficit_breakdown(payload: dict) -> dict:
    """Build explanatory spending-driver inputs for structural deficit guidance."""
    return {
        "income_w": float(st.session_state.get(INCOME_W, payload["income_w"])),
        "fixed_total_w": float(st.session_state.get(FIXED_TOTAL_W, payload["fixed_total_w"])),
        "discretionary_w": float(st.session_state.get(DISCRETIONARY_W, payload["discretionary_w"])),
        "margin_w": float(st.session_state.get(WEEKLY_MARGIN, payload["margin_w"])),
        "fixed_items_rows": (
            st.session_state.get(FIXED_ITEMS_ROWS_WEEKLY)
            if isinstance(st.session_state.get(FIXED_ITEMS_ROWS_WEEKLY), list)
            else []
        ),
        "variable_items_rows_weekly": (
            st.session_state.get(VARIABLE_ITEMS_ROWS_WEEKLY)
            if isinstance(st.session_state.get(VARIABLE_ITEMS_ROWS_WEEKLY), list)
            else []
        ),
        "variable_parts": st.session_state.get(VARIABLE_PARTS, {}),
    }


def build_step3_view_model(snapshot: dict) -> dict:
    """Build the complete Step 3 feasibility view model.

    Renderer code should consume this dictionary directly for metrics, charts,
    explanatory text, and next-action guidance. This keeps simulation and
    explanation logic out of the Streamlit layout layer.
    """
    intent = resolve_intent(snapshot)
    payload = build_projection_payload(snapshot)
    baseline_df = payload["baseline_df"]
    plan_a_df = payload["plan_a_df"]
    summary = build_technical_summary(baseline_df, plan_a_df)

    structural_deficit_tips = ""
    if bool(payload["structural_deficit"]):
        breakdown = build_structural_deficit_breakdown(payload)
        structural_deficit_tips = build_structural_deficit_tips(
            breakdown=breakdown,
            deficit_w=abs(float(payload["margin_w"])),
            top_n=3,
        )

    next_actions = build_step3_next_actions_markdown(
        intent=intent,
        margin_w=float(payload["margin_w"]),
        target_a=float(payload["target_a"]),
        disc=float(payload["discretionary_w"]),
        structural_deficit=bool(payload["structural_deficit"]),
        short_term_goal_amount=float(snapshot.get("short_term_goal_amount", 0.0) or 0.0),
        short_term_goal_weeks=int(snapshot.get("short_term_goal_weeks", payload["weeks"]) or payload["weeks"]),
        short_term_goal_required_weekly=float(snapshot.get("short_term_goal_required_weekly", 0.0) or 0.0),
        structural_deficit_tips=structural_deficit_tips,
    )

    explanation = build_step3_technical_details_markdown(
        baseline_df=baseline_df,
        plan_df=plan_a_df,
        weeks=int(payload["weeks"]),
        iterations=int(payload["iterations"]),
        variability_frac=float(payload["variability_frac"]),
        uncertainty_preset=str(payload["uncertainty_preset"]),
        seed=int(payload["seed"]),
        target_a=float(payload["target_a"]),
        short_term_goal_amount=float(snapshot.get("short_term_goal_amount", 0.0) or 0.0),
        short_term_goal_weeks=int(snapshot.get("short_term_goal_weeks", payload["weeks"]) or payload["weeks"]),
        short_term_goal_required_weekly=float(snapshot.get("short_term_goal_required_weekly", 0.0) or 0.0),
    )

    return {
        "intent": intent,
        "intent_label": INTENT_LABELS[intent],
        "payload": payload,
        "baseline_df": baseline_df,
        "plan_a_df": plan_a_df,
        "summary": summary,
        "next_actions": next_actions,
        "explanation": explanation,
        "structural_deficit_tips": structural_deficit_tips,
    }