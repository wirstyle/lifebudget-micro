# src/baseline.py
import pandas as pd


def generate_baseline(
    income: float,
    fixed_expenses: float,
    variable_expenses: float,
    months: int = 6,
) -> pd.DataFrame:
    """
    Deterministic baseline projection.

    Computes:
    - Monthly Savings = income - fixed_expenses - variable_expenses
    - Balance = cumulative sum of monthly savings across the horizon

    Returns a DataFrame with:
    Month, Monthly Savings, Balance
    """
    monthly_savings = income - fixed_expenses - variable_expenses

    data = []
    balance = 0.0

    for month in range(1, months + 1):
        balance += monthly_savings
        data.append(
            {
                "Month": month,
                "Monthly Savings": round(monthly_savings, 2),
                "Balance": round(balance, 2),
            }
        )

    return pd.DataFrame(data)
