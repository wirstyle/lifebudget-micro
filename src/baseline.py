"""
Deterministic baseline path helpers for LifeBudget Micro.

This module builds the no-change weekly cash-flow path used as the comparison
anchor in the Personal Finance Planner and short-term feasibility checks.

The baseline calculation is intentionally simple:

    weekly surplus = income - fixed_expenses - variable_expenses

The caller decides what each spending bucket represents. For example, one
service may pass fixed + variable essentials as ``fixed_expenses`` and
discretionary spending as ``variable_expenses``; another service may keep fixed
essentials separate and combine variable essentials with discretionary spending.
The baseline only needs the total weekly spending split across two buckets.

Validation remains deliberately small: this module checks the dataframe schema,
while service/renderer layers decide how to display any user-facing errors.
"""

from __future__ import annotations

import pandas as pd


def validate_baseline_df(df: pd.DataFrame) -> None:
    """Validate the minimum schema expected from a baseline dataframe.

    The charting and feasibility services require at least:

    - ``Week``: 1-indexed week number;
    - ``Balance``: cumulative balance after each week.

    Additional columns, such as ``Weekly Savings``, are allowed.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Baseline output is missing or empty.")

    required = {"Week", "Balance"}
    missing = required - set(df.columns)

    if missing:
        raise KeyError(
            f"Baseline dataframe missing columns: {sorted(missing)}. "
            f"Found columns: {df.columns.tolist()}. "
            "Fix generate_baseline to output at least Week + Balance."
        )


def generate_baseline(
    income: float,
    fixed_expenses: float,
    variable_expenses: float,
    weeks: int = 12,
) -> pd.DataFrame:
    """Generate a deterministic weekly baseline trajectory.

    Parameters:
    - ``income``: weekly take-home income;
    - ``fixed_expenses``: first weekly spending bucket;
    - ``variable_expenses``: second weekly spending bucket;
    - ``weeks``: positive planning horizon in weeks.

    Returns:
    A dataframe with:

    - ``Week``;
    - ``Weekly Savings``: weekly surplus under the no-change baseline;
    - ``Balance``: cumulative balance over the selected horizon.
    """
    horizon = int(weeks)
    if horizon <= 0:
        raise ValueError(f"weeks must be positive, got {weeks}")

    weekly_savings = float(income) - float(fixed_expenses) - float(variable_expenses)

    data = []
    balance = 0.0

    for week in range(1, horizon + 1):
        balance += weekly_savings
        data.append(
            {
                "Week": int(week),
                "Weekly Savings": round(float(weekly_savings), 2),
                "Balance": round(float(balance), 2),
            }
        )

    # Callers validate and decide how to surface errors in the UI.
    return pd.DataFrame(data)