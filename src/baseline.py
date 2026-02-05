# src/baseline.py
"""
Baseline (deterministic) module.

This module provides the "no-change" trajectory used as the comparison anchor.

Important naming note (current MVP):
- `variable_expenses` here is used as the **controllable spending bucket** from Step 1/2.
  In LifeBudget Micro's current design, that bucket corresponds to **discretionary spending**.
- Fixed costs are passed in as `fixed_expenses` (fixed essential + variable essentials combined),
  and are treated as unchanged in the baseline.

Option A chosen:
- Validation lives in app.py (Streamlit-friendly try/except + st.error)
- baseline.py still provides validate_baseline_df() as the single schema rule
"""

from __future__ import annotations
import pandas as pd


def validate_baseline_df(df: pd.DataFrame) -> None:
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
    """
    Generate a deterministic baseline trajectory.

    Parameters:
    - income: weekly income (take-home)
    - fixed_expenses: weekly fixed total (fixed essential + variable essentials combined)
    - variable_expenses: weekly controllable bucket (currently: discretionary spending)
    - weeks: planning horizon

    Returns:
    DataFrame with columns:
    - Week
    - Weekly Savings (weekly surplus given the baseline allocation)
    - Balance (cumulative)
    """
    if int(weeks) <= 0:
        raise ValueError(f"weeks must be positive, got {weeks}")

    weekly_savings = float(income) - float(fixed_expenses) - float(variable_expenses)

    data = []
    balance = 0.0

    for week in range(1, int(weeks) + 1):
        balance += weekly_savings
        data.append(
            {
                "Week": int(week),
                "Weekly Savings": round(float(weekly_savings), 2),
                "Balance": round(float(balance), 2),
            }
        )

    # ✅ No validation here (app.py handles validation + user-facing errors)
    return pd.DataFrame(data)
