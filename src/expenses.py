# src/expenses.py
# ------------------------------------------------------------
# LifeBudget Micro — small helpers for:
# - Weekly normalisation of itemised expenses
# - Default "fixed essentials" rows for st.data_editor
# - Discretionary preset suggestion
# - One-off events cleaning + shock_map building (weekly)
# ------------------------------------------------------------

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, TypedDict, Union
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
        # Handle NaN
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
        # Handle NaN
        v = float(x)
        if math.isnan(v):
            return default
        return int(v)
    except Exception:
        return default


class ExpenseItem(TypedDict, total=False):
    name: str
    amount: float
    period: str
    enabled: bool


def total_weekly_from_items(items: Sequence[dict]) -> float:
    """
    Sum a list of itemised expenses into a weekly total.

    Expected fields (per row / dict):
      - amount (numeric)
      - period ("Weekly"|"Monthly"|"Yearly")  [optional; defaults to Weekly]
      - enabled (bool)                        [optional; defaults True]

    Any missing/invalid rows are treated as 0.
    """
    total = 0.0
    for it in items or []:
        try:
            enabled = bool(it.get("enabled", True))
            if not enabled:
                continue
            amount = _safe_float(it.get("amount", 0.0))
            period = it.get("period", "Weekly") or "Weekly"
            if amount <= 0:
                continue
            total += to_weekly(amount, str(period))
        except Exception:
            # Defensive: never break UI because of one bad row
            continue
    return float(total)


def default_fixed_items_rows() -> List[ExpenseItem]:
    """
    Default rows for a 'Fixed essentials' data_editor.

    These are intentionally generic and safe for an MVP.
    """
    return [
        {"name": "Rent / housing", "amount": 0.0, "period": "Monthly", "enabled": True},
        {"name": "Utilities (gas/electric/water)", "amount": 0.0, "period": "Monthly", "enabled": True},
        {"name": "Council tax", "amount": 0.0, "period": "Monthly", "enabled": True},
        {"name": "Internet / phone", "amount": 0.0, "period": "Monthly", "enabled": True},
        {"name": "Transport pass", "amount": 0.0, "period": "Weekly", "enabled": True},
        {"name": "Insurance", "amount": 0.0, "period": "Monthly", "enabled": True},
        {"name": "Subscriptions (fixed)", "amount": 0.0, "period": "Monthly", "enabled": False},
    ]


def variable_essentials_weekly_total(amount_raw: Union[int, float], period: str) -> float:
    """
    Weekly total for variable essentials (e.g., food, toiletries) when entered as one number.
    """
    amount = _safe_float(amount_raw)
    if amount <= 0:
        return 0.0
    return to_weekly(amount, period)


def discretionary_preset_value(
    income_w: Optional[Union[int, float]] = None,
    *,
    pct_of_income: float = 0.08,
    min_w: float = 10.0,
    max_w: float = 200.0,
    fallback_w: float = 35.0,
) -> float:
    """
    Suggest a weekly discretionary value.

    If income_w provided: use pct_of_income with clamps.
    Otherwise returns fallback_w (keeps previous MVP feel).
    """
    if income_w is None:
        return float(fallback_w)

    inc = _safe_float(income_w)
    if inc <= 0:
        return float(fallback_w)

    suggestion = inc * float(pct_of_income)
    suggestion = max(float(min_w), min(float(max_w), suggestion))
    return float(round(suggestion, 2))


# ----------------------------
# One-off events -> shock map
# ----------------------------

class ShockRow(TypedDict, total=False):
    name: str
    amount: float
    week: int


def clean_events(rows: Sequence[dict], *, weeks: int) -> List[ShockRow]:
    """
    Clean rows from a data_editor-like source.

    Input row keys expected:
      - name (optional)
      - amount (£) : numeric, >= 0
      - week       : integer in [1, weeks]

    Returns only valid rows with amount > 0 and clamped week.
    """
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
            # clamp into range (keeps UX forgiving)
            week = max(1, min(W, week if week != 0 else 1))

        name = str(r.get("name", "") or "").strip()
        cleaned.append({"name": name, "amount": float(round(amount, 2)), "week": int(week)})

    return cleaned


def events_to_weekly_shock_map(rows: Sequence[ShockRow]) -> Dict[int, float]:
    """
    Convert cleaned one-off events into a weekly shock map.

    Convention:
      - positive 'amount' means an expense
      - shock_map stores NEGATIVE numbers to reduce balances

    Output:
      {week_index: -total_amount_for_that_week}
    """
    shock_map: Dict[int, float] = {}
    for r in rows or []:
        week = int(r["week"])
        amt = abs(_safe_float(r["amount"]))
        if amt <= 0:
            continue
        shock_map[week] = float(shock_map.get(week, 0.0) - amt)
    return shock_map
