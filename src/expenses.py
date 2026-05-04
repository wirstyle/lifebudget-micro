"""
Expense and cash-flow helper functions for LifeBudget Micro.

This module contains shared budgeting utilities used by the Personal Finance
Planner and short-term scenario services:

- weekly/monthly/yearly period conversion helpers;
- fixed essentials default rows and weekly aggregation;
- discretionary spending presets;
- variable essentials breakdown calculations;
- one-off event cleaning and weekly shock-map construction;
- cumulative shock application for balance paths.

The functions are intentionally small and dependency-light so they can be reused
from Streamlit UI services without creating circular imports.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, TypedDict, Union


# --- Conversions kept here to avoid circular imports. ---
WEEKS_PER_YEAR = 52.0
MONTHS_PER_YEAR = 12.0
WEEKS_PER_MONTH = WEEKS_PER_YEAR / MONTHS_PER_YEAR

PERIOD_TO_WEEK = {
    "Weekly": 1.0,
    "Monthly": MONTHS_PER_YEAR / WEEKS_PER_YEAR,
    "Yearly": 1.0 / WEEKS_PER_YEAR,
}


def to_weekly(amount: Union[int, float], period: str) -> float:
    """Convert a supported period amount into a weekly amount.

    The function intentionally raises ``ValueError`` for unknown periods so
    calling code catches invalid widget/session values early.
    """
    if period not in PERIOD_TO_WEEK:
        raise ValueError(f"Unknown period: {period!r}. Expected one of {list(PERIOD_TO_WEEK)}")
    return float(amount) * float(PERIOD_TO_WEEK[period])


# ------------------------------------------------------------
# Period conversion helpers
# ------------------------------------------------------------

def weekly_to_monthly(amount: Union[int, float]) -> float:
    """Convert a weekly amount to a monthly equivalent using 52 / 12."""
    return float(amount) * WEEKS_PER_MONTH


def monthly_to_weekly(amount: Union[int, float]) -> float:
    """Convert a monthly amount to a weekly equivalent using 52 / 12."""
    return float(amount) / WEEKS_PER_MONTH


def monthly_to_yearly(amount: Union[int, float]) -> float:
    """Convert a monthly amount to a yearly equivalent."""
    return float(amount) * MONTHS_PER_YEAR


def yearly_to_monthly(amount: Union[int, float]) -> float:
    """Convert a yearly amount to a monthly equivalent."""
    return float(amount) / MONTHS_PER_YEAR


def weekly_to_yearly(amount: Union[int, float]) -> float:
    """Convert a weekly amount to a yearly equivalent."""
    return float(amount) * WEEKS_PER_YEAR


def yearly_to_weekly(amount: Union[int, float]) -> float:
    """Convert a yearly amount to a weekly equivalent."""
    return float(amount) / WEEKS_PER_YEAR


def _safe_float(x) -> float:
    """Convert to float, treating None, NaN, and invalid values as zero."""
    try:
        if x is None:
            return 0.0
        value = float(x)
        if math.isnan(value):
            return 0.0
        return value
    except Exception:
        return 0.0


def _safe_int(x, default: int = 0) -> int:
    """Convert to int, falling back when values are missing or invalid."""
    try:
        if x is None:
            return default
        value = float(x)
        if math.isnan(value):
            return default
        return int(value)
    except Exception:
        return default


# ============================================================
# Fixed essentials — itemised builder helpers
# ============================================================

class ExpenseItem(TypedDict, total=False):
    name: str
    amount: float
    period: str


def total_weekly_from_items(items: Sequence[dict]) -> float:
    """Sum itemised expense rows into a weekly total.

    Expected fields per row:
    - ``amount``: numeric amount;
    - ``period``: ``Weekly``, ``Monthly`` or ``Yearly``; defaults to Weekly.

    Legacy rows with ``enabled=False`` are ignored for backward compatibility.
    """
    total = 0.0

    for item in items or []:
        try:
            if "enabled" in item and not bool(item.get("enabled", True)):
                continue

            amount = _safe_float(item.get("amount", 0.0))
            period = item.get("period", "Weekly") or "Weekly"

            if amount <= 0:
                continue

            total += to_weekly(amount, str(period))
        except Exception:
            continue

    return float(total)


def default_fixed_items_rows() -> List[ExpenseItem]:
    """Return default fixed-essential rows for the optional Step 1 editor."""
    return [
        {"name": "Rent / housing", "amount": 0.0, "period": "Monthly"},
        {"name": "Utilities (gas/electric/water)", "amount": 0.0, "period": "Monthly"},
        {"name": "Council tax", "amount": 0.0, "period": "Monthly"},
        {"name": "Internet / phone", "amount": 0.0, "period": "Monthly"},
        {"name": "Transport pass", "amount": 0.0, "period": "Weekly"},
        {"name": "Insurance", "amount": 0.0, "period": "Monthly"},
        {"name": "Subscriptions (fixed)", "amount": 0.0, "period": "Monthly"},
    ]


# ============================================================
# Discretionary presets
# ============================================================

def discretionary_preset_value(preset_name: str) -> float:
    """Return a weekly discretionary preset used by the legacy Step 1 buttons."""
    name = (preset_name or "").strip().lower()

    if name == "quiet week":
        return 20.0
    if name == "typical":
        return 35.0
    if name == "social-heavy":
        return 60.0

    return 35.0


# ============================================================
# Variable essentials — breakdown model
# ============================================================

def variable_essentials_weekly_total(
    *,
    utilities_base: Union[int, float],
    utilities_period: str,
    season: str,
    commute_days: Union[int, float],
    commute_cost_per_day: Union[int, float],
    groceries: Union[int, float],
    groceries_period: str,
    household: Union[int, float],
    household_period: str,
) -> Dict[str, float]:
    """Return a transparent weekly breakdown for variable essentials.

    The model is intentionally simple for the prototype: utilities are adjusted
    by a small seasonal multiplier, commute is calculated as days times cost,
    and groceries/household basics are normalised to weekly amounts.
    """
    util_base_w = to_weekly(_safe_float(utilities_base), str(utilities_period or "Weekly"))

    season_l = (season or "Normal").strip().lower()
    if season_l == "winter":
        season_mult = 1.15
    elif season_l == "summer":
        season_mult = 0.95
    else:
        season_mult = 1.00

    utilities_w = util_base_w * season_mult

    commute_days_i = max(_safe_int(commute_days, default=0), 0)
    commute_cost = max(_safe_float(commute_cost_per_day), 0.0)
    commute_w = float(commute_days_i) * commute_cost

    groceries_w = to_weekly(_safe_float(groceries), str(groceries_period or "Weekly"))
    household_w = to_weekly(_safe_float(household), str(household_period or "Weekly"))

    total = float(utilities_w + commute_w + groceries_w + household_w)

    return {
        "utilities_weekly": float(round(utilities_w, 2)),
        "commute_weekly": float(round(commute_w, 2)),
        "groceries_weekly": float(round(groceries_w, 2)),
        "household_weekly": float(round(household_w, 2)),
        "total_weekly": float(round(total, 2)),
    }


# ============================================================
# One-off events -> shock map
# ============================================================

class ShockRow(TypedDict, total=False):
    name: str
    amount: float
    week: int


def clean_events(rows: Sequence[dict], *, weeks: int) -> List[ShockRow]:
    """Clean event rows and return valid one-off expense events.

    Input rows usually come from a data editor or compact stress-test preset.
    Only positive amounts are kept. Weeks are clipped to the selected planning
    horizon so downstream simulations can safely build a weekly shock map.
    """
    cleaned: List[ShockRow] = []
    horizon = max(_safe_int(weeks, default=0), 0)

    if horizon <= 0:
        return cleaned

    for row in rows or []:
        amount = _safe_float(row.get("amount", 0.0))
        if amount <= 0:
            continue

        week = _safe_int(row.get("week", 0), default=0)
        if week < 1 or week > horizon:
            week = max(1, min(horizon, week if week != 0 else 1))

        name = str(row.get("name", "") or "").strip()
        cleaned.append({"name": name, "amount": float(round(amount, 2)), "week": int(week)})

    return cleaned


def events_to_weekly_shock_map(rows: Sequence[ShockRow]) -> Dict[int, float]:
    """Convert cleaned one-off events into a weekly shock map.

    Convention:
    - positive ``amount`` means an expense entered by the user;
    - the returned shock map stores negative values to reduce balances.

    Output shape: ``{week_index_1_based: -total_amount_for_week}``.
    """
    shock_map: Dict[int, float] = {}

    for row in rows or []:
        week = int(row["week"])
        amount = abs(_safe_float(row["amount"]))

        if amount <= 0:
            continue

        shock_map[week] = float(shock_map.get(week, 0.0) - amount)

    return shock_map


# ============================================================
# Apply cumulative shock map to a series
# ============================================================

def apply_cumulative_events_to_series(
    values: Sequence[Union[int, float]],
    shock_map: Dict[int, float],
) -> List[float]:
    """Apply a cumulative shock map to a numeric time series.

    ``shock_map`` uses 1-indexed weeks and negative values for expenses.

    Semantics:
    - if a shock occurs at week ``k``, all values from ``k`` onward shift by
      that amount;
    - multiple shocks accumulate over time.
    """
    if values is None:
        return []

    values_list = list(values)
    if not values_list:
        return []

    if not shock_map:
        return [float(value) for value in values_list]

    out: List[float] = []
    cumulative = 0.0

    for index, value in enumerate(values_list):
        week = index + 1

        if week in shock_map:
            cumulative += float(shock_map[week])

        out.append(float(value) + cumulative)

    return out