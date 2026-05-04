"""Reusable Streamlit card helpers for LifeBudget Micro.

This module contains small UI wrappers for consistent card-style sections and
the Step 1 weekly cash-flow snapshot. It does not contain business logic; it
only formats already-computed values for display.
"""

from __future__ import annotations

from typing import Callable, Optional

import matplotlib.pyplot as plt
import streamlit as st

from ui.common.messages import step1_no_margin_message, step1_weekly_margin_message


def card(title: str, content_fn: Callable[[], None], subtitle: Optional[str] = None, *, border: bool = True) -> None:
    with st.container(border=border):
        st.markdown(f"### {str(title)}")
        if subtitle:
            st.caption(str(subtitle))
        content_fn()


def summary_card(title: str, summary: str, *, subtitle: Optional[str] = None, level: str = "info") -> None:
    with st.container(border=True):
        st.markdown(f"### {title}")
        if subtitle:
            st.caption(subtitle)
        if level == "success":
            st.success(summary)
        elif level == "warning":
            st.warning(summary)
        elif level == "error":
            st.error(summary)
        else:
            st.info(summary)


def insight_card(title: str, message: str, level: str = "info") -> None:
    with st.container(border=True):
        st.markdown(f"### {title}")

        if level == "success":
            st.success(message)
        elif level == "warning":
            st.warning(message)
        elif level == "error":
            st.error(message)
        else:
            st.info(message)


def weekly_reality_card(
    *,
    income_weekly: float,
    fixed_weekly: float,
    variable_weekly: float,
    discretionary_weekly: float,
    margin_weekly: float,
    show_equivalents: bool = True,
) -> None:
    with st.container(border=True):
        st.markdown("### Your cash-flow snapshot")

        if margin_weekly > 0:
            st.success(step1_weekly_margin_message(margin_weekly))
        else:
            st.warning(step1_no_margin_message())

        total = max(income_weekly, 0.0)

        if total > 0.0:
            fig, ax = plt.subplots(figsize=(10, 2.4))

            left = 0.0
            for _, value in [
                ("Fixed", fixed_weekly),
                ("Variable", variable_weekly),
                ("Discretionary", discretionary_weekly),
                ("Margin", max(margin_weekly, 0.0)),
            ]:
                ax.barh(["Weekly"], [value], left=left)
                left += value

            ax.axvline(total, linestyle="--")
            ax.set_xlabel("£ per week")
            ax.set_title("Where your weekly money goes")

            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Fixed:** £{fixed_weekly:,.0f}")
                st.markdown(f"**Discretionary:** £{discretionary_weekly:,.0f}")
            with c2:
                st.markdown(f"**Variable:** £{variable_weekly:,.0f}")
                st.markdown(f"**Margin:** £{margin_weekly:,.0f}")

        if show_equivalents:
            with st.expander("See monthly/yearly equivalents (optional)", expanded=False):
                income_monthly = income_weekly * 52.0 / 12.0
                essentials_monthly = (fixed_weekly + variable_weekly) * 52.0 / 12.0
                discretionary_monthly = discretionary_weekly * 52.0 / 12.0
                margin_monthly = margin_weekly * 52.0 / 12.0

                st.markdown(
                    f"**Income:** £{income_weekly:,.0f}/w ≈ £{income_monthly:,.0f}/mo ≈ £{income_monthly*12.0:,.0f}/yr"
                )
                st.markdown(
                    f"**Essentials:** £{(fixed_weekly + variable_weekly):,.0f}/w ≈ £{essentials_monthly:,.0f}/mo ≈ £{essentials_monthly*12.0:,.0f}/yr"
                )
                st.markdown(
                    f"**Discretionary:** £{discretionary_weekly:,.0f}/w ≈ £{discretionary_monthly:,.0f}/mo ≈ £{discretionary_monthly*12.0:,.0f}/yr"
                )
                st.markdown(
                    f"**Leftover (Margin):** £{margin_weekly:,.0f}/w ≈ £{margin_monthly:,.0f}/mo ≈ £{margin_monthly*12.0:,.0f}/yr"
                )