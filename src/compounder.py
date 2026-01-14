# src/compounder.py
import numpy as np


def simulate_scenario(
    income: float,
    fixed_expenses: float,
    variable_expenses: float,
    delta_savings: float,
    months: int,
    iterations: int = 200,
    seed: int = 42,
    variability_pct: float = 0.10,
):
    """
    Monte Carlo light simulation for a simple scenario:
    - We model variability on variable expenses (noise).
    - "delta_savings" reduces variable expenses (i.e., saving more each month).
    Returns mean, lower, upper arrays for balances across months.

    Notes:
    - seed fixed for reproducibility (important for assessment + logging).
    - variability_pct controls the standard deviation as a fraction of variable_expenses.
    """
    rng = np.random.default_rng(seed)

    all_runs = np.zeros((iterations, months), dtype=float)

    for i in range(iterations):
        balance = 0.0
        for m in range(months):
            # Noise based on variable expenses variability
            noise = rng.normal(loc=0.0, scale=variable_expenses * variability_pct)

            # Scenario effect: save more => reduce variable expenses by delta_savings
            effective_variable = variable_expenses - delta_savings + noise
            if effective_variable < 0:
                effective_variable = 0.0

            savings = income - fixed_expenses - effective_variable
            balance += savings

            all_runs[i, m] = balance

    mean = all_runs.mean(axis=0)
    lower = np.percentile(all_runs, 10, axis=0)
    upper = np.percentile(all_runs, 90, axis=0)

    return mean, lower, upper
