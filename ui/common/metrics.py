# ui/common/metrics.py

"""
Reusable metric rendering helpers for Streamlit.

Goal:
- Centralise st.metric usage
- Standardise formatting (%, decimals, etc.)
- Avoid duplication across app.py
"""

from __future__ import annotations

from typing import Optional

import streamlit as st


def _format_number(value, decimals: int = 2, suffix: str = "") -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{decimals}f}{suffix}"
    except Exception:
        return str(value)


def safe_metric(
    label: str,
    value: Optional[float],
    delta: Optional[float] = None,
    suffix: str = "",
    decimals: int = 2,
) -> None:
    """Safely render a metric with optional delta."""
    if value is None:
        st.metric(label, "—")
        return

    display_value = _format_number(value, decimals=decimals, suffix=suffix)

    if delta is not None:
        display_delta = _format_number(delta, decimals=decimals, suffix=suffix)
        st.metric(label, display_value, display_delta)
    else:
        st.metric(label, display_value)


def percent_metric(
    label: str,
    value: Optional[float],
    delta: Optional[float] = None,
    decimals: int = 2,
) -> None:
    """Render a fraction (0.12) as a percentage metric (12.00%)."""
    safe_metric(
        label,
        value * 100 if value is not None else None,
        delta * 100 if delta is not None else None,
        suffix="%",
        decimals=decimals,
    )


def percent_points_metric(
    label: str,
    value: Optional[float],
    delta: Optional[float] = None,
    decimals: int = 2,
) -> None:
    """Render already-percentage values (e.g. 12.5) without multiplying by 100."""
    safe_metric(label, value, delta=delta, suffix=" pp", decimals=decimals)


def integer_metric(label: str, value: Optional[float]) -> None:
    """Render integer metric."""
    if value is None:
        st.metric(label, "—")
        return
    try:
        st.metric(label, f"{int(value)}")
    except Exception:
        st.metric(label, str(value))


def currency_metric(
    label: str,
    value: Optional[float],
    delta: Optional[float] = None,
    currency_symbol: str = "£",
    decimals: int = 2,
) -> None:
    """Render a currency metric."""
    if value is None:
        st.metric(label, "—")
        return

    try:
        display_value = f"{currency_symbol}{float(value):,.{decimals}f}"
    except Exception:
        display_value = str(value)

    if delta is not None:
        try:
            display_delta = f"{currency_symbol}{float(delta):,.{decimals}f}"
        except Exception:
            display_delta = str(delta)
        st.metric(label, display_value, display_delta)
    else:
        st.metric(label, display_value)


def robustness_badge(score: float) -> None:
    """Visual classification for robustness score."""
    try:
        score_value = float(score)
    except Exception:
        st.info("Robustness unavailable.")
        return

    if score_value >= 75:
        st.success(f"Robustness: {score_value:.0f}/100 (Stable)")
    elif score_value >= 45:
        st.info(f"Robustness: {score_value:.0f}/100 (Moderate)")
    else:
        st.warning(f"Robustness: {score_value:.0f}/100 (Fragile)")


def robustness_metrics(
    robustness_score: Optional[float],
    instability_penalty: Optional[float],
) -> None:
    """Common two-metric block used in robustness / temporal stability sections."""
    col1, col2 = st.columns(2)
    with col1:
        safe_metric("Robustness score", robustness_score, suffix="/100", decimals=0)
    with col2:
        safe_metric("Temporal penalty", instability_penalty, decimals=3)


def stability_window_metrics(
    stable_count: Optional[int],
    moderate_count: Optional[int],
    high_count: Optional[int],
) -> None:
    """Common three-metric block used for stability window counts."""
    c1, c2, c3 = st.columns(3)
    with c1:
        integer_metric("Stable windows", stable_count)
    with c2:
        integer_metric("Moderate sensitivity", moderate_count)
    with c3:
        integer_metric("High sensitivity", high_count)
