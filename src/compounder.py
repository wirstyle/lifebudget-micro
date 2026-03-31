# src/compounder.py
"""
Short-horizon weekly scenario simulation for LifeBudget Micro.

Responsibilities
----------------
This module handles the weekly uncertainty layer used by the core budgeting
engine. In particular, it is responsible for:

- weekly scenario simulation
- uncertainty in variable/discretionary spending
- one-off shock/event adjustments to balances
- building scenario DataFrames for app.py visualisation

Scope note
----------
This module is intentionally limited to short-horizon weekly budgeting logic.

Long-horizon savings growth / investment projection is handled separately in
`investment.py`, which was added as an optional module for probabilistic
investment-growth exploration under risk and volatility assumptions.

This separation is intentional:
- `compounder.py` supports weekly budgeting decisions
- `investment.py` supports optional long-term growth exploration
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# NEW: multi-event cumulative shock helper
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
    """
    Monte Carlo–style simulation for a simple behavioural scenario (weekly).

    Model:
    - Uncertainty is applied ONLY to variable expenses via Gaussian noise.
    - delta_savings is interpreted as a REAL weekly spend reduction (a "cut")
      applied to the controllable spending bucket.
    """
    if weeks <= 0:
        raise ValueError(f"weeks must be positive, got {weeks}")
    if iterations <= 0:
        raise ValueError(f"iterations must be positive, got {iterations}")
    if variability_frac < 0:
        raise ValueError(f"variability_frac must be >= 0, got {variability_frac}")

    rng = np.random.default_rng(seed)
    all_runs = np.zeros((iterations, weeks), dtype=float)

    delta_cut = float(delta_savings)

    for i in range(iterations):
        balance = 0.0
        for w in range(weeks):
            noise = rng.normal(loc=0.0, scale=variable_expenses * variability_frac)

            effective_variable = variable_expenses - delta_cut + noise
            if effective_variable < 0.0:
                effective_variable = 0.0

            savings = income - fixed_expenses - effective_variable
            balance += savings
            all_runs[i, w] = balance

    mean = all_runs.mean(axis=0)
    lower = np.percentile(all_runs, 10, axis=0)
    upper = np.percentile(all_runs, 90, axis=0)

    return mean, lower, upper


# ============================================================
# One-off shock (single event – legacy)
# ============================================================

def apply_one_off_shock(values, shock_amount: float, shock_week: int):
    """
    Apply a single unexpected expense ONCE at week shock_week (1-indexed).

    Semantics:
    - From shock_week onward, balances are reduced by shock_amount.
    """
    if values is None:
        return values
    if shock_amount <= 0:
        return [float(v) for v in values]

    start_idx = max(int(shock_week) - 1, 0)
    out = []
    for i, v in enumerate(values):
        vv = float(v)
        out.append(vv - float(shock_amount) if i >= start_idx else vv)
    return out


def apply_one_off_shock_to_df(
    df: pd.DataFrame,
    shock_amount: float,
    shock_week: int,
    value_cols: Tuple[str, ...] = ("Balance",),
) -> pd.DataFrame:
    """
    Apply a single cumulative shock to one or more dataframe columns.
    """
    if df is None or df.empty or shock_amount <= 0:
        return df

    missing = [c for c in value_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Dataframe missing expected columns {missing}. "
            f"Found columns: {df.columns.tolist()}"
        )

    df2 = df.copy()
    start_idx = max(int(shock_week) - 1, 0)

    for col in value_cols:
        vals = df2[col].tolist()
        df2[col] = [
            float(v) - float(shock_amount) if i >= start_idx else float(v)
            for i, v in enumerate(vals)
        ]

    return df2


# ============================================================
# NEW: Multi-event shock (shock_map)
# ============================================================

def apply_shock_map_to_df(
    df: pd.DataFrame,
    shock_map: Dict[int, float],
    value_cols: Tuple[str, ...] = ("Balance",),
) -> pd.DataFrame:
    """
    Apply a cumulative multi-event shock map to one or more dataframe columns.

    shock_map:
      {week (1-indexed) -> total shock amount at that week}

    Semantics:
    - Shocks are punctual (not tracking).
    - If a shock occurs at week k, balance is reduced from week k onward.
    - Multiple shocks accumulate over time.
    """
    if df is None or df.empty or not shock_map:
        return df

    missing = [c for c in value_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Dataframe missing expected columns {missing}. "
            f"Found columns: {df.columns.tolist()}"
        )

    df2 = df.copy()

    for col in value_cols:
        values = df2[col].tolist()
        df2[col] = apply_cumulative_events_to_series(values, shock_map)

    return df2


# ============================================================
# Scenario dataframe builder (UI-friendly)
# ============================================================

@dataclass(frozen=True)
class ScenarioParams:
    delta_savings: float
    weeks: int
    iterations: int
    seed: int
    variability_frac: float
    shock_enabled: bool = False
    shock_amount: float = 0.0
    shock_week: int = 1


def validate_scenario_df(df: pd.DataFrame) -> None:
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
    """
    Runs simulate_scenario and returns:
    - scenario_df: Week, Mean, Lower, Upper
    - params: traceability dict
    """
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

    mean_l = [float(x) for x in mean]
    low_l = [float(x) for x in lower]
    up_l = [float(x) for x in upper]

    # Legacy single-event shock (kept for backward compatibility)
    if shock_enabled and shock_amount > 0:
        mean_l = apply_one_off_shock(mean_l, shock_amount, shock_week)
        low_l = apply_one_off_shock(low_l, shock_amount, shock_week)
        up_l = apply_one_off_shock(up_l, shock_amount, shock_week)

    df = pd.DataFrame(
        {
            "Week": list(range(1, int(weeks) + 1)),
            "Mean": mean_l,
            "Lower": low_l,
            "Upper": up_l,
        }
    )

    validate_scenario_df(df)

    params = {
        "delta_savings": float(delta_savings),
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
    """
    Convenience wrapper:
    - Scenario A: seed + 0
    - Scenario B: seed + 1
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
