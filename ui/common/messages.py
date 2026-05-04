"""Shared Streamlit message and heading helpers for LifeBudget Micro.

This module centralises small UI message wrappers and short Step 1 copy strings
so the app keeps consistent success, warning, error, loading, and section-header
patterns across screens.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

import streamlit as st


def show_success(message: str) -> None:
    if message:
        st.success(str(message))


def show_info(message: str) -> None:
    if message:
        st.info(str(message))


def show_warning(message: str) -> None:
    if message:
        st.warning(str(message))


def show_error(message: str) -> None:
    if message:
        st.error(str(message))


def show_toast_or_success(message: str, *, icon: str = "✅", fallback_level: str = "success") -> None:
    if not message:
        return
    try:
        st.toast(str(message), icon=icon)
        return
    except Exception:
        pass

    {
        "warning": st.warning,
        "error": st.error,
        "info": st.info,
    }.get(str(fallback_level).lower().strip(), st.success)(str(message))


@contextmanager
def loading(message: str = "Processing...") -> Iterator[None]:
    with st.spinner(message):
        yield


def section_header(title: str, subtitle: Optional[str] = None) -> None:
    if title:
        st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def subsection_header(title: str, subtitle: Optional[str] = None) -> None:
    if title:
        st.markdown(f"#### {title}")
    if subtitle:
        st.caption(subtitle)


def step1_intro_message() -> str:
    return "Enter rough numbers, then confirm to create your cash-flow snapshot."


def step1_confirmation_required_message() -> str:
    return "Please confirm your situation first."


def step1_weekly_margin_message(margin_weekly: float) -> str:
    return f"You appear able to save about £{float(margin_weekly):,.0f}/week under this snapshot."


def step1_no_margin_message() -> str:
    return "Right now there is no weekly margin available without changing something."


def step1_preset_applied(label: str) -> str:
    return f"Preset applied: {str(label)}"


def step1_income_estimate_applied() -> str:
    return "Estimate applied to take-home income."


def step1_fixed_breakdown_applied() -> str:
    return "Fixed essentials breakdown applied."


def step1_variable_breakdown_applied() -> str:
    return "Variable essentials breakdown applied."


def step1_current_situation_locked() -> str:
    return "Current situation locked. Building your baseline..."
