"""
Breakdown helpers for the Personal Finance Planner.

This module supports the optional fixed and variable essentials breakdowns used
inside Step 1. It converts detailed user-entered items into weekly totals and
stores lightweight weekly rows in ``st.session_state`` so the budget UI can show
the largest spending drivers.

The canonical planning snapshot is still built by ``step1_budget_service``.
This module only prepares breakdown-level details for estimates, summaries, and
diagnostic guidance.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.expenses import default_fixed_items_rows, to_weekly, total_weekly_from_items, variable_essentials_weekly_total
from ui.state.keys import (
    STEP1_FIXED_BREAKDOWN_DF,
    STEP1_VAR_COMMUTE_COST,
    STEP1_VAR_COMMUTE_DAYS,
    STEP1_VAR_GROCERIES_AMOUNT,
    STEP1_VAR_GROCERIES_PERIOD,
    STEP1_VAR_HOUSEHOLD_AMOUNT,
    STEP1_VAR_HOUSEHOLD_PERIOD,
    STEP1_VAR_SEASON,
    STEP1_VAR_UTILITIES_AMOUNT,
    STEP1_VAR_UTILITIES_PERIOD,
    VARIABLE_ITEMS_ROWS_WEEKLY,
    VARIABLE_PARTS,
    VARIABLE_TOP_DRIVER,
)


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float without letting bad UI/session values crash."""
    try:
        return float(value)
    except Exception:
        return float(default)


def normalize_df(df: Any) -> pd.DataFrame:
    """Return a clean dataframe copy with a reset index, or an empty dataframe."""
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    return df.copy().reset_index(drop=True)


def ensure_fixed_breakdown_df() -> pd.DataFrame:
    """Return the fixed essentials dataframe, creating defaults when missing.

    The fixed breakdown editor expects a dataframe with user-facing column names.
    If the user has not edited it yet, defaults from ``src.expenses`` are seeded
    into session state so the editor and weekly total helper share one source.
    """
    df = st.session_state.get(STEP1_FIXED_BREAKDOWN_DF)
    if isinstance(df, pd.DataFrame) and not df.empty:
        return normalize_df(df)

    rows = [
        {
            "Name": str(row.get("name", "") or ""),
            "Amount (£)": float(row.get("amount", 0.0) or 0.0),
            "Period": str(row.get("period", "Monthly") or "Monthly"),
        }
        for row in default_fixed_items_rows()
    ]
    df = normalize_df(pd.DataFrame(rows))
    st.session_state[STEP1_FIXED_BREAKDOWN_DF] = df
    return df


def fixed_breakdown_weekly_total(df: pd.DataFrame) -> float:
    """Calculate the weekly total for the fixed essentials breakdown."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return 0.0

    items = []
    for _, row in df.iterrows():
        items.append(
            {
                "name": str(row.get("Name", "") or ""),
                "amount": safe_float(row.get("Amount (£)"), 0.0),
                "period": str(row.get("Period", "Monthly") or "Monthly"),
            }
        )

    return float(total_weekly_from_items(items))


def rows_with_weekly_from_fixed_df(df: pd.DataFrame) -> list[dict[str, float | str]]:
    """Convert fixed breakdown rows into weekly item rows for guidance cards.

    These rows are not the canonical budget source. They are lightweight
    explanations used by the UI to identify the largest fixed spending drivers.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []

    rows: list[dict[str, float | str]] = []
    for _, row in df.iterrows():
        try:
            name = str(row.get("Name", "") or "").strip()
            amount = safe_float(row.get("Amount (£)"), 0.0)
            period = str(row.get("Period", "Monthly") or "Monthly")
            weekly = float(to_weekly(amount, period))

            if name and weekly > 0.0:
                rows.append({"name": name, "weekly": weekly})
        except Exception:
            continue

    return rows


def _variable_breakdown() -> dict[str, float]:
    """Build the variable essentials breakdown from current widget state."""
    utilities = safe_float(st.session_state.get(STEP1_VAR_UTILITIES_AMOUNT, 0.0), 0.0)
    utilities_period = str(st.session_state.get(STEP1_VAR_UTILITIES_PERIOD, "Weekly"))
    groceries = safe_float(st.session_state.get(STEP1_VAR_GROCERIES_AMOUNT, 0.0), 0.0)
    groceries_period = str(st.session_state.get(STEP1_VAR_GROCERIES_PERIOD, "Weekly"))
    household = safe_float(st.session_state.get(STEP1_VAR_HOUSEHOLD_AMOUNT, 0.0), 0.0)
    household_period = str(st.session_state.get(STEP1_VAR_HOUSEHOLD_PERIOD, "Weekly"))
    commute_days = int(st.session_state.get(STEP1_VAR_COMMUTE_DAYS, 0) or 0)
    commute_cost = safe_float(st.session_state.get(STEP1_VAR_COMMUTE_COST, 0.0), 0.0)
    season = str(st.session_state.get(STEP1_VAR_SEASON, "Normal") or "Normal")

    # The UI exposes a simple Low/Normal/High control, while the expense helper
    # expects seasonal labels. High approximates winter utility pressure and Low
    # approximates summer utility pressure.
    season_value = "Winter" if season == "High" else ("Summer" if season == "Low" else "Normal")

    return variable_essentials_weekly_total(
        utilities_base=utilities,
        utilities_period=utilities_period,
        season=season_value,
        commute_days=commute_days,
        commute_cost_per_day=commute_cost,
        groceries=groceries,
        groceries_period=groceries_period,
        household=household,
        household_period=household_period,
    )


def variable_breakdown_weekly_total() -> float:
    """Return the total weekly value for the variable essentials breakdown."""
    return float(_variable_breakdown().get("total_weekly", 0.0) or 0.0)


def build_variable_items_rows_weekly() -> list[dict[str, float | str]]:
    """Persist variable essentials rows and top-driver metadata.

    The returned rows support Budget pressure guidance and explanatory UI cards.
    The full breakdown payload is also stored in ``VARIABLE_PARTS`` for debugging
    or future summaries, while ``VARIABLE_TOP_DRIVER`` stores the largest row.
    """
    breakdown = _variable_breakdown()

    rows = [
        {"name": "Utilities", "weekly": float(breakdown.get("utilities_weekly", 0.0) or 0.0)},
        {"name": "Groceries", "weekly": float(breakdown.get("groceries_weekly", 0.0) or 0.0)},
        {"name": "Household basics", "weekly": float(breakdown.get("household_weekly", 0.0) or 0.0)},
        {"name": "Commute", "weekly": float(breakdown.get("commute_weekly", 0.0) or 0.0)},
    ]

    out = [row for row in rows if float(row["weekly"]) > 0.0]
    out.sort(key=lambda r: float(r["weekly"]), reverse=True)

    st.session_state[VARIABLE_PARTS] = breakdown
    st.session_state[VARIABLE_ITEMS_ROWS_WEEKLY] = out
    st.session_state[VARIABLE_TOP_DRIVER] = out[0] if out else None

    return out