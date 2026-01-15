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
    variability_frac: float = 0.30,
):
    """
    Monte Carlo–style simulation for a simple behavioural scenario.

    Model:
    - Uncertainty is applied ONLY to variable expenses via Gaussian noise.
    - delta_savings represents additional monthly savings by reducing variable expenses.

    Parameters:
    - variability_frac: fraction of variable_expenses used as the noise std dev.
      Example: variability_frac=0.10 => std dev = 10% of variable_expenses.

    Returns:
    - mean, lower, upper arrays (balances across months)
      where lower/upper are 10th/90th percentiles (uncertainty band).
    """
    rng = np.random.default_rng(seed)
    all_runs = np.zeros((iterations, months), dtype=float)

    for i in range(iterations):
        balance = 0.0
        for m in range(months):
            noise = rng.normal(loc=0.0, scale=variable_expenses * variability_frac)

            effective_variable = variable_expenses - delta_savings + noise
            if effective_variable < 0.0:
                effective_variable = 0.0

            savings = income - fixed_expenses - effective_variable
            balance += savings
            all_runs[i, m] = balance

    mean = all_runs.mean(axis=0)
    lower = np.percentile(all_runs, 10, axis=0)
    upper = np.percentile(all_runs, 90, axis=0)

    return mean, lower, upper
