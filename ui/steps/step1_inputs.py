"""
Step 1 — User Inputs (Budget / Preferences)

Refactor goals in this version:
- keep renderer UI-focused
- move snapshot/baseline orchestration to ui.services.step1_budget_service
- move breakdown helpers to ui.services.step1_breakdown_service
- reuse ui.common.cards / ui.common.tables / ui.state.updates
"""
from __future__ import annotations

import streamlit as st

from src.expenses import discretionary_preset_value
from ui.common.cards import weekly_reality_card
from ui.common.messages import (
    section_header,
    show_toast_or_success,
    step1_confirmation_required_message,
    step1_fixed_breakdown_applied,
    step1_income_estimate_applied,
    step1_intro_message,
    step1_preset_applied,
    step1_variable_breakdown_applied,
)
from ui.common.tables import stateful_data_editor
from ui.services.step1_breakdown_service import (
    ensure_fixed_breakdown_df,
    fixed_breakdown_weekly_total,
    normalize_df,
    safe_float,
    variable_breakdown_weekly_total,
)
from ui.services.step1_budget_service import (
    confirm_step1_snapshot,
    get_weekly_reality_payload,
    invalidate_downstream_from_step1,
    prepare_step1_preview,
    seed_step1_from_snapshot_if_missing,
)
from ui.state.keys import *
from ui.state.updates import apply_pending_step_patch, queue_step_patch

_PERIOD_OPTIONS = ["Weekly", "Monthly", "Yearly"]
_SEASON_OPTIONS = ["Low", "Normal", "High"]


def _rounded_weekly_target(value: float) -> float:
    try:
        return float(round(max(float(value), 0.0)))
    except Exception:
        return 0.0


def _render_step2_target_seed(snapshot: dict) -> None:
    if not isinstance(snapshot, dict) or not snapshot:
        return

    payload = get_weekly_reality_payload(snapshot)
    margin_weekly = float(payload.get("margin_weekly", 0.0) or 0.0)

    if margin_weekly <= 0.0:
        st.warning(
            "Suggested next step: your current snapshot does not leave a weekly margin yet. "
            "Use Step 2 to test a very cautious target, or go back and adjust the snapshot first."
        )
        return

    suggested_weekly = _rounded_weekly_target(margin_weekly * 0.30)
    st.info(
        f"Suggested next step: try starting Step 2 around **£{suggested_weekly:,.0f}/week**. "
        "You can still adjust it with Safe, Recommended or Ambitious presets."
    )



def _format_gbp_weekly(value: float) -> str:
    try:
        return f"£{float(value):,.0f}/week"
    except Exception:
        return "£0/week"


def _collect_budget_driver_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []

    fixed_rows = st.session_state.get(FIXED_ITEMS_ROWS_WEEKLY, []) or []
    variable_rows = st.session_state.get(VARIABLE_ITEMS_ROWS_WEEKLY, []) or []

    for row in list(fixed_rows):
        try:
            name = str(row.get("name", "") or "").strip()
            weekly = float(row.get("weekly", 0.0) or 0.0)
            if name and weekly > 0:
                rows.append({"name": name, "weekly": weekly, "bucket": "Fixed"})
        except Exception:
            continue

    for row in list(variable_rows):
        try:
            name = str(row.get("name", "") or "").strip()
            weekly = float(row.get("weekly", 0.0) or 0.0)
            if name and weekly > 0:
                rows.append({"name": name, "weekly": weekly, "bucket": "Variable"})
        except Exception:
            continue

    rows.sort(key=lambda r: float(r.get("weekly", 0.0) or 0.0), reverse=True)
    return rows


def _render_budget_pressure_guidance(snapshot: dict) -> None:
    if not isinstance(snapshot, dict) or not snapshot:
        return

    payload = get_weekly_reality_payload(snapshot)
    margin_weekly = float(payload.get("margin_weekly", 0.0) or 0.0)
    discretionary_weekly = float(payload.get("discretionary_weekly", 0.0) or 0.0)
    income_weekly = float(payload.get("income_weekly", 0.0) or 0.0)
    fixed_weekly = float(payload.get("fixed_weekly", 0.0) or 0.0)
    variable_weekly = float(payload.get("variable_weekly", 0.0) or 0.0)

    if margin_weekly > 0.0:
        return

    shortfall_weekly = abs(float(margin_weekly))
    max_possible_margin = margin_weekly + max(discretionary_weekly, 0.0)
    structural_deficit = max_possible_margin < 0.0

    st.markdown("### Budget pressure detected")

    if structural_deficit:
        st.warning(
            f"Your current snapshot is short by about **{_format_gbp_weekly(shortfall_weekly)}**. "
            "Even removing discretionary spending entirely would not fully restore weekly margin, "
            "so the first check should be income accuracy and the largest essential categories."
        )
    else:
        st.info(
            f"Your current snapshot is short by about **{_format_gbp_weekly(shortfall_weekly)}**. "
            "This appears recoverable within discretionary spending, but it is worth checking whether "
            "the current discretionary figure is realistic before setting a savings target."
        )

    essentials_weekly = fixed_weekly + variable_weekly
    essentials_ratio = (essentials_weekly / income_weekly) if income_weekly > 0 else 0.0
    discretionary_ratio = (discretionary_weekly / income_weekly) if income_weekly > 0 else 0.0

    with st.expander("What should I check first?", expanded=False):
        if income_weekly <= 0:
            st.write("• **Income:** take-home income is currently zero or missing. Check the amount and period first.")
        else:
            st.write(
                f"• **Essentials:** currently about **{_format_gbp_weekly(essentials_weekly)}** "
                f"({essentials_ratio * 100:.0f}% of weekly income)."
            )
            st.write(
                f"• **Discretionary spending:** currently about **{_format_gbp_weekly(discretionary_weekly)}** "
                f"({discretionary_ratio * 100:.0f}% of weekly income)."
            )

        driver_rows = _collect_budget_driver_rows()
        if driver_rows:
            st.write("• **Largest itemised drivers to verify:**")
            for row in driver_rows[:3]:
                st.write(
                    f"  - {row['name']} ({row['bucket']}): **{_format_gbp_weekly(float(row['weekly']))}**"
                )
        else:
            st.write(
                "• Add detail in **Need help estimating essentials?** if you want the app to identify the largest fixed or variable drivers."
            )

        if structural_deficit:
            st.write(
                "• Because this looks structural, avoid treating Step 2 as a normal savings target yet. "
                "First, check whether income period, rent/housing, utilities, transport or other essential fields are entered correctly."
            )
        else:
            st.write(
                "• Because this may be recoverable, start by checking flexible/discretionary categories before changing the long-term plan."
            )


def render_step_1() -> None:
    _, success_message = apply_pending_step_patch(
        STEP1_PENDING_WIDGET_PATCH,
        STEP1_PENDING_SUCCESS_MESSAGE,
    )
    if success_message:
        show_toast_or_success(success_message, icon="✅", fallback_level="success")
    seed_step1_from_snapshot_if_missing()

    section_header("Step 1 — Your current situation", "Rough numbers are fine — this is a starting estimate.")

    income_col, income_period_col = st.columns([2.1, 1.0])
    with income_col:
        st.number_input("Take-home income (£)", min_value=0.0, value=safe_float(st.session_state.get(STEP1_INCOME_AMOUNT, 460.0), 460.0), step=10.0, key=STEP1_INCOME_AMOUNT)
    with income_period_col:
        st.selectbox("Income period", _PERIOD_OPTIONS, index=_PERIOD_OPTIONS.index(str(st.session_state.get(STEP1_INCOME_PERIOD, "Weekly"))), key=STEP1_INCOME_PERIOD)

    with st.expander("Not sure about take-home pay? Estimate it (optional)", expanded=False):
        e1, e2 = st.columns(2)
        with e1:
            gross_amount = st.number_input("Gross income (£)", min_value=0.0, value=safe_float(st.session_state.get(STEP1_ESTIMATED_GROSS_INCOME, 30000.0), 30000.0), step=10.0, key=STEP1_ESTIMATED_GROSS_INCOME)
        with e2:
            gross_period = st.selectbox("Gross income period", _PERIOD_OPTIONS, index=_PERIOD_OPTIONS.index(str(st.session_state.get(STEP1_ESTIMATED_GROSS_PERIOD, "Yearly"))), key=STEP1_ESTIMATED_GROSS_PERIOD)
        tax_rate = st.slider("Estimated deduction rate", min_value=0.0, max_value=0.60, value=float(st.session_state.get(STEP1_ESTIMATED_TAX_RATE, 0.30) or 0.30), step=0.01, key=STEP1_ESTIMATED_TAX_RATE)
        estimated_take_home = gross_amount * (1.0 - tax_rate)
        st.caption(f"Estimated take-home: £{estimated_take_home:,.2f} per {gross_period.lower()}")
        if st.button("Use this estimate as my take-home income", key="step1_apply_income_estimate"):
            invalidate_downstream_from_step1()
            queue_step_patch(
                STEP1_PENDING_WIDGET_PATCH,
                {STEP1_INCOME_AMOUNT: float(estimated_take_home), STEP1_INCOME_PERIOD: str(gross_period)},
                success_key=STEP1_PENDING_SUCCESS_MESSAGE,
                success_message=step1_income_estimate_applied(),
            )

    fixed_col, fixed_period_col = st.columns([2.1, 1.0])
    with fixed_col:
        st.number_input("Fixed essentials (£)", min_value=0.0, value=safe_float(st.session_state.get(STEP1_FIXED_AMOUNT, 185.0), 185.0), step=5.0, key=STEP1_FIXED_AMOUNT)
    with fixed_period_col:
        st.selectbox("Period", _PERIOD_OPTIONS, index=_PERIOD_OPTIONS.index(str(st.session_state.get(STEP1_FIXED_PERIOD, "Weekly"))), key=STEP1_FIXED_PERIOD)

    variable_col, variable_period_col = st.columns([2.1, 1.0])
    with variable_col:
        st.number_input("Variable essentials (£)", min_value=0.0, value=safe_float(st.session_state.get(STEP1_VARIABLE_AMOUNT, 80.0), 80.0), step=5.0, key=STEP1_VARIABLE_AMOUNT)
    with variable_period_col:
        st.selectbox("Period ", _PERIOD_OPTIONS, index=_PERIOD_OPTIONS.index(str(st.session_state.get(STEP1_VARIABLE_PERIOD, "Weekly"))), key=STEP1_VARIABLE_PERIOD)

    discretionary_col, discretionary_period_col = st.columns([2.1, 1.0])
    with discretionary_col:
        st.number_input("Discretionary spending (£)", min_value=0.0, value=safe_float(st.session_state.get(STEP1_DISCRETIONARY_AMOUNT, 35.0), 35.0), step=5.0, key=STEP1_DISCRETIONARY_AMOUNT)
    with discretionary_period_col:
        st.selectbox("Period  ", _PERIOD_OPTIONS, index=_PERIOD_OPTIONS.index(str(st.session_state.get(STEP1_DISCRETIONARY_PERIOD, "Weekly"))), key=STEP1_DISCRETIONARY_PERIOD)

    preset_cols = st.columns(3)
    for col, label in zip(preset_cols, ["Quiet week", "Typical", "Social-heavy"]):
        with col:
            if st.button(label, key=f"step1_discretionary_preset_{label}"):
                weekly_amount = discretionary_preset_value(label)
                current_period = str(st.session_state.get(STEP1_DISCRETIONARY_PERIOD, "Weekly"))
                patched_value = weekly_amount if current_period == "Weekly" else weekly_amount * 52.0 / 12.0
                invalidate_downstream_from_step1()
                queue_step_patch(
                    STEP1_PENDING_WIDGET_PATCH,
                    {STEP1_DISCRETIONARY_AMOUNT: float(round(patched_value, 2))},
                    success_key=STEP1_PENDING_SUCCESS_MESSAGE,
                    success_message=step1_preset_applied(label),
                )

    with st.expander("Need help estimating essentials? (optional)", expanded=False):
        st.markdown("### Build fixed essentials (optional)")
        editor_df = ensure_fixed_breakdown_df()
        with st.form("step1_fixed_breakdown_form", clear_on_submit=False):
            edited_fixed_df = stateful_data_editor(
                state_key=STEP1_FIXED_BREAKDOWN_DF,
                editor_key="step1_fixed_breakdown_editor",
                default_df=editor_df,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                column_config={
                    "Name": st.column_config.TextColumn("Name"),
                    "Amount (£)": st.column_config.NumberColumn("Amount (£)", min_value=0.0, step=1.0),
                    "Period": st.column_config.SelectboxColumn("Period", options=_PERIOD_OPTIONS),
                },
            )
            normalized_fixed_df = normalize_df(edited_fixed_df)
            fixed_total_weekly = fixed_breakdown_weekly_total(normalized_fixed_df)
            st.write(f"**Estimated total:** £{fixed_total_weekly:,.2f}/week")
            apply_fixed_breakdown = st.form_submit_button("Use this total as my Fixed essentials")

        if apply_fixed_breakdown:
            st.session_state[STEP1_FIXED_BREAKDOWN_DF] = normalized_fixed_df
            invalidate_downstream_from_step1()
            queue_step_patch(
                STEP1_PENDING_WIDGET_PATCH,
                {STEP1_FIXED_AMOUNT: float(round(fixed_total_weekly, 2)), STEP1_FIXED_PERIOD: "Weekly"},
                success_key=STEP1_PENDING_SUCCESS_MESSAGE,
                success_message=step1_fixed_breakdown_applied(),
            )

        st.markdown("---")
        st.markdown("### Variable essentials breakdown (optional)")
        v1, v2 = st.columns(2)
        with v1:
            st.number_input("Utilities baseline (£)", min_value=0.0, value=safe_float(st.session_state.get(STEP1_VAR_UTILITIES_AMOUNT, 0.0), 0.0), step=5.0, key=STEP1_VAR_UTILITIES_AMOUNT)
            st.selectbox("Utilities period", _PERIOD_OPTIONS, index=_PERIOD_OPTIONS.index(str(st.session_state.get(STEP1_VAR_UTILITIES_PERIOD, "Weekly"))), key=STEP1_VAR_UTILITIES_PERIOD)
            st.selectbox("Season", _SEASON_OPTIONS, index=_SEASON_OPTIONS.index(str(st.session_state.get(STEP1_VAR_SEASON, "Normal"))), key=STEP1_VAR_SEASON)
            st.slider("Commute days/week", min_value=0, max_value=7, value=int(st.session_state.get(STEP1_VAR_COMMUTE_DAYS, 0) or 0), key=STEP1_VAR_COMMUTE_DAYS)
            st.number_input("Cost per commute day (£)", min_value=0.0, value=safe_float(st.session_state.get(STEP1_VAR_COMMUTE_COST, 0.0), 0.0), step=1.0, key=STEP1_VAR_COMMUTE_COST)
        with v2:
            st.number_input("Groceries (£)", min_value=0.0, value=safe_float(st.session_state.get(STEP1_VAR_GROCERIES_AMOUNT, 0.0), 0.0), step=5.0, key=STEP1_VAR_GROCERIES_AMOUNT)
            st.selectbox("Groceries period", _PERIOD_OPTIONS, index=_PERIOD_OPTIONS.index(str(st.session_state.get(STEP1_VAR_GROCERIES_PERIOD, "Weekly"))), key=STEP1_VAR_GROCERIES_PERIOD)
            st.number_input("Household basics (£)", min_value=0.0, value=safe_float(st.session_state.get(STEP1_VAR_HOUSEHOLD_AMOUNT, 0.0), 0.0), step=5.0, key=STEP1_VAR_HOUSEHOLD_AMOUNT)
            st.selectbox("Household period", _PERIOD_OPTIONS, index=_PERIOD_OPTIONS.index(str(st.session_state.get(STEP1_VAR_HOUSEHOLD_PERIOD, "Weekly"))), key=STEP1_VAR_HOUSEHOLD_PERIOD)

        variable_total_weekly = variable_breakdown_weekly_total()
        st.write(f"**Estimated total:** £{variable_total_weekly:,.2f}/week")
        if st.button("Use this total as my Variable essentials", key="step1_apply_variable_breakdown"):
            invalidate_downstream_from_step1()
            queue_step_patch(
                STEP1_PENDING_WIDGET_PATCH,
                {STEP1_VARIABLE_AMOUNT: float(round(variable_total_weekly, 2)), STEP1_VARIABLE_PERIOD: "Weekly"},
                success_key=STEP1_PENDING_SUCCESS_MESSAGE,
                success_message=step1_variable_breakdown_applied(),
            )

    snapshot = prepare_step1_preview()

    st.info(step1_intro_message())
    confirmed = bool(st.session_state.get(CONFIRMED_SNAPSHOT, False))
    if confirmed:
        confirmed_snapshot = st.session_state.get(PLANNING_SNAPSHOT, {}) or {}
        if isinstance(confirmed_snapshot, dict) and confirmed_snapshot:
            weekly_reality_card(**get_weekly_reality_payload(confirmed_snapshot))
            _render_budget_pressure_guidance(confirmed_snapshot)
            _render_step2_target_seed(confirmed_snapshot)

    left, middle, right = st.columns(3)
    with left:
        if st.button("Back to Step 0", key="step1_back_to_step0"):
            st.session_state[CURRENT_STEP] = 0
            st.rerun()
    with middle:
        confirm_label = "Update snapshot" if confirmed else "Confirm my current situation"
        if st.button(confirm_label, key="step1_confirm_snapshot"):
            confirm_step1_snapshot(snapshot)
            st.rerun()
    with right:
        if st.button("Continue to Step 2", key="step1_continue", disabled=not confirmed):
            st.session_state[CURRENT_STEP] = 2
            st.rerun()
        if not confirmed:
            st.caption(step1_confirmation_required_message())
