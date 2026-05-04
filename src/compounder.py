"""
Short-horizon weekly scenario simulation for LifeBudget Micro.

This module supports the Personal Finance Planner feasibility check. It builds
weekly simulated balance paths for a target savings scenario, including:

- uncertainty in variable/discretionary spending;
- deterministic target-plan savings adjustments;
- 10th/mean/90th percentile outputs for charting;
- one-off shock/event adjustments to balances.

Long-horizon savings and investment projections are handled elsewhere. This
module is intentionally limited to short-term weekly budgeting scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.expenses import apply_cumulative_events_to_series


# ============================================================
# Core simulation
# ============================================================

def simulate_scenario(
    income: float,
    fixed_expenses: float,
    variable_expenses: float,
    delta_savings: float,
    weeks: int,
    iterations: int = 200,
    seed: int = 42,
    variability_frac: float = 0.30,
):
    """Run a weekly Monte Carlo-style cash-flow simulation.

    Model assumptions:
    - ``income`` and ``fixed_expenses`` are deterministic weekly values;
    - uncertainty is applied only to ``variable_expenses``;
    - ``variable_expenses`` may represent discretionary spending, or a combined
      variable/discretionary bucket, depending on the caller;
    - ``delta_savings`` is interpreted as a real weekly spending reduction
      applied to the controllable spending bucket.

    Returns:
    - mean weekly cumulative balance path;
    - 10th percentile path;
    - 90th percentile path.
    """
    weeks = int(weeks)
    iterations = int(iterations)

    if weeks <= 0:
        raise ValueError(f"weeks must be positive, got {weeks}")
    if iterations <= 0:
        raise ValueError(f"iterations must be positive, got {iterations}")
    if variability_frac < 0:
        raise ValueError(f"variability_frac must be >= 0, got {variability_frac}")

    rng = np.random.default_rng(seed)
    all_runs = np.zeros((iterations, weeks), dtype=float)

    income = float(income)
    fixed_expenses = float(fixed_expenses)
    variable_expenses = max(float(variable_expenses), 0.0)
    delta_cut = max(float(delta_savings), 0.0)

    for i in range(iterations):
        balance = 0.0

        for week_index in range(weeks):
            noise = rng.normal(
                loc=0.0,
                scale=variable_expenses * float(variability_frac),
            )

            effective_variable = variable_expenses - delta_cut + noise
            if effective_variable < 0.0:
                effective_variable = 0.0

            savings = income - fixed_expenses - effective_variable
            balance += savings
            all_runs[i, week_index] = balance

    mean = all_runs.mean(axis=0)
    lower = np.percentile(all_runs, 10, axis=0)
    upper = np.percentile(all_runs, 90, axis=0)

    return mean, lower, upper


# ============================================================
# One-off shock support — single-event legacy helpers
# ============================================================

def apply_one_off_shock(values, shock_amount: float, shock_week: int):
    """Apply one legacy one-off expense from ``shock_week`` onward.

    ``shock_week`` is 1-indexed. From that week onward, all balances are reduced
    by ``shock_amount``. The newer multi-event pathway uses shock maps, but this
    helper is kept for backward compatibility.
    """
    if values is None:
        return values
    if shock_amount <= 0:
        return [float(value) for value in values]

    start_idx = max(int(shock_week) - 1, 0)
    out = []

    for index, value in enumerate(values):
        numeric_value = float(value)
        out.append(numeric_value - float(shock_amount) if index >= start_idx else numeric_value)

    return out


def apply_one_off_shock_to_df(
    df: pd.DataFrame,
    shock_amount: float,
    shock_week: int,
    value_cols: Tuple[str, ...] = ("Balance",),
) -> pd.DataFrame:
    """Apply one legacy cumulative shock to one or more dataframe columns."""
    if df is None or df.empty or shock_amount <= 0:
        return df

    missing = [column for column in value_cols if column not in df.columns]
    if missing:
        raise KeyError(
            f"Dataframe missing expected columns {missing}. "
            f"Found columns: {df.columns.tolist()}"
        )

    df2 = df.copy()
    start_idx = max(int(shock_week) - 1, 0)

    for column in value_cols:
        values = df2[column].tolist()
        df2[column] = [
            float(value) - float(shock_amount) if index >= start_idx else float(value)
            for index, value in enumerate(values)
        ]

    return df2


# ============================================================
# Multi-event shock map support
# ============================================================

def apply_shock_map_to_df(
    df: pd.DataFrame,
    shock_map: Dict[int, float],
    value_cols: Tuple[str, ...] = ("Balance",),
) -> pd.DataFrame:
    """Apply a cumulative multi-event shock map to dataframe columns.

    ``shock_map`` uses the convention ``{week_1_indexed: shock_amount}``.
    Negative shock amounts reduce balances.

    Semantics:
    - shocks are punctual events, not recurring costs;
    - if a shock occurs at week ``k``, balances are reduced from week ``k``
      onward;
    - multiple shocks accumulate over time.
    """
    if df is None or df.empty or not shock_map:
        return df

    missing = [column for column in value_cols if column not in df.columns]
    if missing:
        raise KeyError(
            f"Dataframe missing expected columns {missing}. "
            f"Found columns: {df.columns.tolist()}"
        )

    df2 = df.copy()

    for column in value_cols:
        values = df2[column].tolist()
        df2[column] = apply_cumulative_events_to_series(values, shock_map)

    return df2


# ============================================================
# Scenario dataframe builder
# ============================================================

@dataclass(frozen=True)
class ScenarioParams:
    """Traceability container for short-term scenario settings.

    The current UI usually consumes the plain ``params`` dictionary returned by
    ``simulate_scenario_df``. This dataclass is kept as a lightweight structured
    representation for compatibility and possible future use.
    """

    delta_savings: float
    weeks: int
    iterations: int
    seed: int
    variability_frac: float
    shock_enabled: bool = False
    shock_amount: float = 0.0
    shock_week: int = 1


def validate_scenario_df(df: pd.DataFrame) -> None:
    """Validate the minimum schema expected from a scenario dataframe."""
    if df is None or df.empty:
        raise ValueError("Scenario dataframe is empty.")

    required = {"Week", "Mean", "Lower", "Upper"}
    missing = required - set(df.columns)

    if missing:
        raise KeyError(
            f"Scenario dataframe missing columns: {sorted(missing)}. "
            f"Found columns: {df.columns.tolist()}"
        )


def simulate_scenario_df(
    income: float,
    fixed_expenses: float,
    variable_expenses: float,
    delta_savings: float,
    weeks: int,
    iterations: int = 200,
    seed: int = 42,
    variability_frac: float = 0.30,
    shock_enabled: bool = False,
    shock_amount: float = 0.0,
    shock_week: int = 1,
) -> Tuple[pd.DataFrame, Dict]:
    """Run a weekly scenario simulation and return UI-friendly outputs.

    Returns:
    - ``scenario_df`` with ``Week``, ``Mean``, ``Lower`` and ``Upper`` columns;
    - ``params`` dictionary for traceability/debugging.

    ``Lower`` and ``Upper`` are the 10th and 90th percentile balance paths.
    """
    weeks = int(weeks)
    iterations = int(iterations)

    mean, lower, upper = simulate_scenario(
        income=income,
        fixed_expenses=fixed_expenses,
        variable_expenses=variable_expenses,
        delta_savings=delta_savings,
        weeks=weeks,
        iterations=iterations,
        seed=seed,
        variability_frac=variability_frac,
    )

    mean_l = [float(value) for value in mean]
    low_l = [float(value) for value in lower]
    up_l = [float(value) for value in upper]

    # Legacy single-event shock pathway. The preferred newer path applies
    # shock maps outside this helper via ``apply_shock_map_to_df``.
    if shock_enabled and shock_amount > 0:
        mean_l = apply_one_off_shock(mean_l, shock_amount, shock_week)
        low_l = apply_one_off_shock(low_l, shock_amount, shock_week)
        up_l = apply_one_off_shock(up_l, shock_amount, shock_week)

    df = pd.DataFrame(
        {
            "Week": list(range(1, weeks + 1)),
            "Mean": mean_l,
            "Lower": low_l,
            "Upper": up_l,
        }
    )

    validate_scenario_df(df)

    params = {
        "delta_savings": float(max(float(delta_savings), 0.0)),
        "weeks": int(weeks),
        "iterations": int(iterations),
        "seed": int(seed),
        "variability_frac": float(variability_frac),
        "shock_enabled": bool(shock_enabled),
        "shock_amount": float(shock_amount),
        "shock_week": int(shock_week),
    }

    return df, params


def simulate_scenario_df_with_seed_offset(
    income: float,
    fixed_expenses: float,
    variable_expenses: float,
    delta_savings: float,
    weeks: int,
    iterations: int = 200,
    seed: int = 42,
    seed_offset: int = 0,
    variability_frac: float = 0.30,
    shock_enabled: bool = False,
    shock_amount: float = 0.0,
    shock_week: int = 1,
) -> Tuple[pd.DataFrame, Dict]:
    """Run ``simulate_scenario_df`` with an added seed offset.

    This is useful when comparing two short-term scenarios while keeping them
    reproducible but not identical.
    """
    return simulate_scenario_df(
        income=income,
        fixed_expenses=fixed_expenses,
        variable_expenses=variable_expenses,
        delta_savings=delta_savings,
        weeks=weeks,
        iterations=iterations,
        seed=int(seed) + int(seed_offset),
        variability_frac=variability_frac,
        shock_enabled=shock_enabled,
        shock_amount=shock_amount,
        shock_week=shock_week,
    )