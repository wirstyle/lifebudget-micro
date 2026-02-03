# src/baseline.py
import pandas as pd


def generate_baseline(
    income: float,
    fixed_expenses: float,
    variable_expenses: float,
    weeks: int = 12,
) -> pd.DataFrame:
    """
    Deterministic baseline projection (weekly).

    Computes:
    - Weekly Savings = income - fixed_expenses - variable_expenses
    - Balance = cumulative sum of weekly savings across the horizon

    Returns a DataFrame with:
    Week, Weekly Savings, Balance
    """
    weekly_savings = income - fixed_expenses - variable_expenses

    data = []
    balance = 0.0

    for week in range(1, weeks + 1):
        balance += weekly_savings
        data.append(
            {
                "Week": week,
                "Weekly Savings": round(weekly_savings, 2),
                "Balance": round(balance, 2),
            }
        )

    return pd.DataFrame(data)
