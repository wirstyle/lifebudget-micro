"""
Budget snapshot service for the Personal Finance Planner.

This module converts the Step 1 UI inputs into the canonical planning snapshot
stored in ``st.session_state[PLANNING_SNAPSHOT]``. That snapshot is the hand-off
between the Personal Finance Planner and later modules such as short-term
feasibility, investment setup, and long-term scenarios.

The service also keeps weekly derived values in ``st.session_state`` for legacy
compatibility and for UI cards that need quick access to income, spending, and
free margin. Renderer files should call this service rather than rebuilding
budget maths directly.
"""

from __future__ import annotations

import streamlit as st

from src.baseline import generate_baseline, validate_baseline_df
from src.expenses import to_weekly, weekly_to_monthly
from ui.common.messages import step1_current_situation_locked
from ui.services.step1_breakdown_service import (
    build_variable_items_rows_weekly,
    rows_with_weekly_from_fixed_df,
    safe_float,
)
from ui.state.keys import *


def to_monthly_explicit(amount: float, period: str) -> float:
    """Convert a user-entered amount into a monthly equivalent.

    Step 1 allows users to enter values as weekly, monthly, or yearly figures.
    The planning snapshot stores the canonical cash-flow inputs as monthly
    values because later services use monthly totals for contribution bridges
    and scenario calculations.
    """
    amount = safe_float(amount, 0.0)
    period = str(period or "Monthly")
    if period == "Weekly":
        return float(weekly_to_monthly(amount))
    if period == "Yearly":
        return amount / 12.0
    return amount


def seed_step1_from_snapshot_if_missing() -> None:
    """Restore Step 1 widget defaults from an existing planning snapshot.

    This is used when the user returns to the Personal Finance Planner after a
    snapshot has already been created. Existing widget values are not overwritten,
    because Streamlit widget-backed keys must remain user-controlled once set.
    """
    snapshot = st.session_state.get(PLANNING_SNAPSHOT, {}) or {}
    if not isinstance(snapshot, dict) or not snapshot:
        return

    periods = snapshot.get("input_periods", {}) or {}
    period_map = {
        STEP1_INCOME_PERIOD: str(periods.get("income", "Weekly") or "Weekly"),
        STEP1_FIXED_PERIOD: str(periods.get("fixed", "Weekly") or "Weekly"),
        STEP1_VARIABLE_PERIOD: str(periods.get("variable", "Weekly") or "Weekly"),
        STEP1_DISCRETIONARY_PERIOD: str(periods.get("discretionary", "Weekly") or "Weekly"),
    }
    for key, value in period_map.items():
        if key not in st.session_state:
            st.session_state[key] = value

    amounts = {
        STEP1_INCOME_AMOUNT: safe_float(snapshot.get("monthly_income", 0.0), 0.0),
        STEP1_FIXED_AMOUNT: safe_float(snapshot.get("fixed_essentials_monthly", 0.0), 0.0),
        STEP1_VARIABLE_AMOUNT: safe_float(snapshot.get("variable_essentials_monthly", 0.0), 0.0),
        STEP1_DISCRETIONARY_AMOUNT: safe_float(snapshot.get("discretionary_spending_monthly", 0.0), 0.0),
    }
    for key, monthly_value in amounts.items():
        if key in st.session_state:
            continue
        period = st.session_state.get(
            {
                STEP1_INCOME_AMOUNT: STEP1_INCOME_PERIOD,
                STEP1_FIXED_AMOUNT: STEP1_FIXED_PERIOD,
                STEP1_VARIABLE_AMOUNT: STEP1_VARIABLE_PERIOD,
                STEP1_DISCRETIONARY_AMOUNT: STEP1_DISCRETIONARY_PERIOD,
            }[key],
            "Weekly",
        )
        if period == "Weekly":
            st.session_state[key] = float(to_weekly(monthly_value, "Monthly"))
        elif period == "Monthly":
            st.session_state[key] = float(monthly_value)
        else:
            st.session_state[key] = float(monthly_value * 12.0)


def persist_weekly_artifacts(snapshot: dict) -> None:
    """Persist weekly derived values used by legacy cards and downstream services.

    The canonical data source is still ``PLANNING_SNAPSHOT``. These additional
    keys are maintained because older Step 2/3 components and summary cards read
    weekly values directly from ``st.session_state``.
    """
    income_w = float(to_weekly(safe_float(snapshot.get("monthly_income", 0.0), 0.0), "Monthly"))
    fixed_w = float(to_weekly(safe_float(snapshot.get("fixed_essentials_monthly", 0.0), 0.0), "Monthly"))
    var_w = float(to_weekly(safe_float(snapshot.get("variable_essentials_monthly", 0.0), 0.0), "Monthly"))
    disc_w = float(to_weekly(safe_float(snapshot.get("discretionary_spending_monthly", 0.0), 0.0), "Monthly"))
    fixed_total_w = float(fixed_w + var_w)
    margin_w = float(income_w - fixed_total_w - disc_w)

    st.session_state[INCOME_W] = income_w
    st.session_state[FIXED_TOTAL_W] = fixed_total_w
    st.session_state[DISCRETIONARY_W] = disc_w
    st.session_state[WEEKLY_MARGIN] = margin_w
    st.session_state[FIXED_W] = fixed_w
    st.session_state[VAR_W] = var_w
    st.session_state[DISC_W] = disc_w
    st.session_state[MONTHLY_INCOME_DERIVED] = float(snapshot.get("monthly_income", 0.0) or 0.0)
    st.session_state[ESSENTIAL_SPENDING_MONTHLY_DERIVED] = float(snapshot.get("essential_spending_monthly", 0.0) or 0.0)
    st.session_state[DISCRETIONARY_SPENDING_MONTHLY_DERIVED] = float(
        snapshot.get("discretionary_spending_monthly", 0.0) or 0.0
    )
    st.session_state[WEEKLY_SAVINGS_DERIVED] = float(snapshot.get("weekly_savings", 0.0) or 0.0)

    st.session_state[FIXED_ITEMS_ROWS_WEEKLY] = rows_with_weekly_from_fixed_df(
        st.session_state.get(STEP1_FIXED_BREAKDOWN_DF)
    )
    build_variable_items_rows_weekly()


def invalidate_downstream_from_step1() -> None:
    """Mark baseline and plan outputs stale after Step 1 inputs change.

    Any budget edit can invalidate the baseline, short-term plans, feasibility
    outputs, and later scenario assumptions. Renderer code calls this before
    queuing widget patches that modify income or spending fields.
    """
    st.session_state[BASELINE_READY] = False
    st.session_state[BASELINE_DF] = None
    st.session_state[BASELINE_SIGNATURE] = None
    st.session_state[PLAN_READY] = False
    st.session_state[PLAN_A_DF] = None
    st.session_state[PLAN_B_DF] = None
    st.session_state[DOWNSTREAM_INPUTS_DIRTY] = True
    st.session_state[PLAN_GENERATED] = False


def build_baseline_signature(snapshot: dict) -> dict[str, float | int | str | bool]:
    """Build a compact signature for the current baseline inputs.

    This is used to detect whether the stored baseline still matches the current
    budget, planning horizon, uncertainty seed, and one-off shock setting.
    """
    return {
        "income_w": round(float(st.session_state.get(INCOME_W, 0.0) or 0.0), 6),
        "fixed_total_w": round(float(st.session_state.get(FIXED_TOTAL_W, 0.0) or 0.0), 6),
        "discretionary_w": round(float(st.session_state.get(DISCRETIONARY_W, 0.0) or 0.0), 6),
        "weeks": int(snapshot.get("planning_horizon_weeks", 12) or 12),
        "preset_name": str(snapshot.get("uncertainty_preset", "Quick estimate (default)") or "Quick estimate (default)"),
        "seed": int(snapshot.get("random_run_nonce", 0) or 0) + 42,
        "shock_enabled": bool(snapshot.get("enable_one_off_events", False)),
    }


def sync_confirmed_snapshot(snapshot: dict) -> None:
    """Merge, persist, and confirm the current Step 1 planning snapshot.

    Confirmation writes the canonical snapshot, regenerates the baseline path,
    validates the baseline dataframe, and marks the Step 1 baseline as ready for
    downstream planning modules.
    """
    current_snapshot = st.session_state.get(PLANNING_SNAPSHOT, {}) or {}
    merged_snapshot = dict(current_snapshot) if isinstance(current_snapshot, dict) else {}
    merged_snapshot.update(dict(snapshot))

    persist_weekly_artifacts(merged_snapshot)

    # ``generate_baseline`` keeps the historical parameter name
    # ``variable_expenses`` for flexible/discretionary spending. Fixed and
    # variable essentials are already combined into FIXED_TOTAL_W above.
    baseline_df = generate_baseline(
        income=float(st.session_state.get(INCOME_W, 0.0) or 0.0),
        fixed_expenses=float(st.session_state.get(FIXED_TOTAL_W, 0.0) or 0.0),
        variable_expenses=float(st.session_state.get(DISCRETIONARY_W, 0.0) or 0.0),
        weeks=int(merged_snapshot.get("planning_horizon_weeks", 12) or 12),
    )

    validate_baseline_df(baseline_df)
    st.session_state[PLANNING_SNAPSHOT] = merged_snapshot
    st.session_state[BASELINE_DF] = baseline_df
    st.session_state[BASELINE_SIGNATURE] = build_baseline_signature(merged_snapshot)
    st.session_state[CONFIRMED_SNAPSHOT] = True
    st.session_state[BASELINE_READY] = True
    st.session_state[DOWNSTREAM_INPUTS_DIRTY] = False


def build_snapshot() -> dict:
    """Build the canonical Step 1 cash-flow snapshot from current widget state.

    The snapshot stores monthly canonical values plus weekly convenience fields.
    Step 1 owns the current cash-flow baseline; the selected savings target is
    refined by the Step 2 service layer in the combined dashboard.
    """
    current_snapshot = st.session_state.get(PLANNING_SNAPSHOT, {}) or {}
    current_snapshot = dict(current_snapshot) if isinstance(current_snapshot, dict) else {}

    income_amount = safe_float(st.session_state.get(STEP1_INCOME_AMOUNT, 0.0), 0.0)
    income_period = str(st.session_state.get(STEP1_INCOME_PERIOD, "Weekly"))
    fixed_amount = safe_float(st.session_state.get(STEP1_FIXED_AMOUNT, 0.0), 0.0)
    fixed_period = str(st.session_state.get(STEP1_FIXED_PERIOD, "Weekly"))
    variable_amount = safe_float(st.session_state.get(STEP1_VARIABLE_AMOUNT, 0.0), 0.0)
    variable_period = str(st.session_state.get(STEP1_VARIABLE_PERIOD, "Weekly"))
    discretionary_amount = safe_float(st.session_state.get(STEP1_DISCRETIONARY_AMOUNT, 0.0), 0.0)
    discretionary_period = str(st.session_state.get(STEP1_DISCRETIONARY_PERIOD, "Weekly"))

    # Step 1 preserves the legacy weekly_savings field when present. In the
    # combined Personal Finance Planner, the active target is applied and
    # persisted by the Step 2 goal service after this baseline snapshot exists.
    weekly_savings = safe_float(
        st.session_state.get(WEEKLY_SAVINGS_DERIVED, current_snapshot.get("weekly_savings", 0.0)),
        0.0,
    )

    monthly_income = to_monthly_explicit(income_amount, income_period)
    fixed_monthly = to_monthly_explicit(fixed_amount, fixed_period)
    variable_monthly = to_monthly_explicit(variable_amount, variable_period)
    discretionary_monthly = to_monthly_explicit(discretionary_amount, discretionary_period)
    essential_spending = fixed_monthly + variable_monthly
    baseline_monthly = max(monthly_income - essential_spending - discretionary_monthly, 0.0)
    baseline_weekly = baseline_monthly * 12.0 / 52.0
    plan_a_weekly = max(weekly_savings, 0.0)
    plan_a_monthly = plan_a_weekly * 52.0 / 12.0
    required_cut_monthly = max(plan_a_monthly - baseline_monthly, 0.0)
    structural_deficit = monthly_income < (essential_spending + discretionary_monthly)

    return {
        "weekly_savings": plan_a_weekly,
        "monthly_income": monthly_income,
        "essential_spending_monthly": essential_spending,
        "fixed_essentials_monthly": fixed_monthly,
        "variable_essentials_monthly": variable_monthly,
        "discretionary_spending_monthly": discretionary_monthly,
        "baseline_savings_monthly": baseline_monthly,
        "baseline_savings_weekly": baseline_weekly,
        "target_a_weekly": plan_a_weekly,
        "target_a_monthly": plan_a_monthly,
        "required_cut_a_monthly": required_cut_monthly,
        "structural_deficit": structural_deficit,
        "input_periods": {
            "income": income_period,
            "fixed": fixed_period,
            "variable": variable_period,
            "discretionary": discretionary_period,
        },
    }


def prepare_step1_preview() -> dict:
    """Build and store a non-final preview snapshot for the current inputs.

    The preview lets the UI show live weekly values before the snapshot is
    confirmed or auto-saved into the downstream planning state.
    """
    snapshot = build_snapshot()
    st.session_state[STEP1_PLANNING_SNAPSHOT_PREVIEW] = snapshot
    persist_weekly_artifacts(snapshot)
    return snapshot


def get_weekly_reality_payload(snapshot: dict) -> dict[str, float]:
    """Return weekly values for UI cards after syncing derived session keys."""
    persist_weekly_artifacts(snapshot)
    return {
        "income_weekly": float(st.session_state.get(INCOME_W, 0.0) or 0.0),
        "fixed_weekly": float(st.session_state.get(FIXED_W, 0.0) or 0.0),
        "variable_weekly": float(st.session_state.get(VAR_W, 0.0) or 0.0),
        "discretionary_weekly": float(st.session_state.get(DISC_W, 0.0) or 0.0),
        "margin_weekly": float(st.session_state.get(WEEKLY_MARGIN, 0.0) or 0.0),
    }


def confirm_step1_snapshot(snapshot: dict) -> None:
    """Confirm the Step 1 snapshot and queue the user-facing success message."""
    sync_confirmed_snapshot(snapshot)
    st.session_state[STEP1_PENDING_SUCCESS_MESSAGE] = step1_current_situation_locked()