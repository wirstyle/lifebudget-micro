import pandas as pd

def generate_baseline(income, fixed_expenses, variable_expenses, months=6):
    """
    Generates a deterministic baseline projection of savings and balance.

    Parameters:
    - income: monthly income
    - fixed_expenses: fixed monthly expenses
    - variable_expenses: average variable monthly expenses
    - months: number of months to project

    Returns:
    - pandas DataFrame with Month, Monthly Savings, and Balance
    """

    monthly_savings = income - fixed_expenses - variable_expenses

    data = []
    balance = 0

    for month in range(1, months + 1):
        balance += monthly_savings
        data.append({
            "Month": month,
            "Monthly Savings": round(monthly_savings, 2),
            "Balance": round(balance, 2)
        })

    return pd.DataFrame(data)
