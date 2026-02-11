# src/expenses.py
# ------------------------------------------------------------
# LifeBudget Micro — helpers for:
# - Weekly normalisation of itemised expenses
# - Default "fixed essentials" rows for st.data_editor
# - Discretionary preset suggestion (by preset name)
# - Variable essentials breakdown model (utilities/season/commute/groceries/household)
# - One-off events cleaning + shock_map building (weekly)
# - Apply cumulative shock_map to a time series (used by compounder.py)
# ------------------------------------------------------------

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, TypedDict, Union
import math


# --- Conversions (kept here to avoid circular imports) ---
PERIOD_TO_WEEK = {
    "Weekly": 1.0,
    "Monthly": 12.0 / 52.0,  # approx, consistent with app.py
    "Yearly": 1.0 / 52.0,
}


def to_weekly(amount: Union[int, float], period: str) -> float:
    """Convert (amount, period) -> weekly float."""
    if period not in PERIOD_TO_WEEK:
        raise ValueError(f"Unknown period: {period!r}. Expected one of {list(PERIOD_TO_WEEK)}")
    return float(amount) * float(PERIOD_TO_WEEK[period])


def _safe_float(x) -> float:
    try:
        if x is None:
            return 0.0
        v = float(x)
        if math.isnan(v):
            return 0.0
        return v
    except Exception:
        return 0.0


def _safe_int(x, default: int = 0) -> int:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v):
            return default
        return int(v)
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
    """
    Sum a list of itemised expenses into a weekly total.

    Expected fields per dict:
      - amount (numeric)
      - period ("Weekly"|"Monthly"|"Yearly")  [optional; defaults Weekly]

    Back-compat:
      - If legacy 'enabled' exists, it is respected (False => row ignored).
    """
    total = 0.0
    for it in items or []:
        try:
            # Back-compat: respect legacy "enabled" if present
            if "enabled" in it and not bool(it.get("enabled", True)):
                continue

            amount = _safe_float(it.get("amount", 0.0))
            period = it.get("period", "Weekly") or "Weekly"

            if amount <= 0:
                continue

            total += to_weekly(amount, str(period))
        except Exception:
            continue

    return float(total)

def default_fixed_items_rows() -> List[ExpenseItem]:
    """Default rows for a 'Fixed essentials' data_editor."""
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
# Discretionary presets (matches Step 1 UI buttons)
# ============================================================

def discretionary_preset_value(preset_name: str) -> float:
    """
    Return a weekly discretionary preset value based on the UI preset button.
    Matches Step 1: ["Quiet week", "Typical", "Social-heavy"].
    """
    name = (preset_name or "").strip().lower()
    if name == "quiet week":
        return 20.0
    if name == "typical":
        return 35.0
    if name == "social-heavy":
        return 60.0
    # fallback (keeps MVP feel)
    return 35.0


# ============================================================
# Variable essentials — breakdown model (matches your Step 1 expander)
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
    """
    Returns a dict with a transparent breakdown + total_weekly.

    Season multipliers are intentionally simple for an MVP.
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
    """Clean rows from data_editor-like source; returns only valid amount>0 rows."""
    cleaned: List[ShockRow] = []
    W = int(weeks)
    if W <= 0:
        return cleaned

    for r in rows or []:
        amount = _safe_float(r.get("amount", 0.0))
        if amount <= 0:
            continue

        week = _safe_int(r.get("week", 0), default=0)
        if week < 1 or week > W:
            week = max(1, min(W, week if week != 0 else 1))

        name = str(r.get("name", "") or "").strip()
        cleaned.append({"name": name, "amount": float(round(amount, 2)), "week": int(week)})

    return cleaned


def events_to_weekly_shock_map(rows: Sequence[ShockRow]) -> Dict[int, float]:
    """
    Convert cleaned events into a weekly shock map.

    Convention:
      - positive 'amount' means an expense
      - shock_map stores NEGATIVE numbers to reduce balances
    Output: {week_index (1-indexed): -total_amount_for_week}
    """
    shock_map: Dict[int, float] = {}
    for r in rows or []:
        week = int(r["week"])
        amt = abs(_safe_float(r["amount"]))
        if amt <= 0:
            continue
        shock_map[week] = float(shock_map.get(week, 0.0) - amt)
    return shock_map


# ============================================================
# Apply cumulative shock map to a series (used by compounder.py)
# ============================================================

def apply_cumulative_events_to_series(values: Sequence[Union[int, float]], shock_map: Dict[int, float]) -> List[float]:
    """
    Apply a cumulative shock_map to a time series.

    shock_map:
      {week (1-indexed) -> shock amount (NEGATIVE for expenses)}

    Semantics:
    - If a shock occurs at week k, all values from k onward shift by that amount.
    - Multiple shocks accumulate over time.
    """
    if not values:
        return []
    if not shock_map:
        return [float(v) for v in values]

    out: List[float] = []
    cumulative = 0.0

    for i, v in enumerate(values):
        week = i + 1  # 1-indexed
        if week in shock_map:
            cumulative += float(shock_map[week])
        out.append(float(v) + cumulative)

    return out
