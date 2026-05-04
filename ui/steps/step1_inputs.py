"""
Personal Finance Planner renderer for LifeBudget Micro.

This module owns the user-facing finance setup flow. In the current prototype,
Step 1 acts as a compact dashboard that combines:

- Step 1: current weekly income/spending estimate;
- Step 2: weekly savings target selection;
- Step 3: short-term feasibility and stress-test preview.

The canonical hand-off is ``st.session_state[PLANNING_SNAPSHOT]``. Later modules
use that snapshot, together with the selected weekly target, as the bridge into
investment and long-term scenario planning.

Most business logic lives in ``ui.services`` modules. This file should remain
primarily responsible for Streamlit layout, widget state, user feedback, and
safe navigation between steps.
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

# Default route for the final prototype: Step 1 renders the combined Personal
# Finance Planner dashboard instead of the older standalone Step 1 page.
# Keeping the flag allows the legacy view to remain available for compatibility.
PERSONAL_FINANCE_COMBINED_MODE_KEY = "step1_personal_finance_combined_mode_enabled"


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


def _personal_finance_snapshot_confirmed() -> bool:
    return bool(st.session_state.get(CONFIRMED_SNAPSHOT, False))


def _personal_finance_has_positive_margin(snapshot: dict) -> bool:
    if not isinstance(snapshot, dict) or not snapshot:
        return False
    try:
        return float(snapshot.get("baseline_savings_weekly", 0.0) or 0.0) > 0.0
    except Exception:
        return False


def _personal_finance_has_target(snapshot: dict) -> bool:
    if STEP1_TARGET_WEEKLY_SAVINGS in st.session_state:
        return True
    if not isinstance(snapshot, dict) or not snapshot:
        return False
    for key in ("target_a_weekly", "target_weekly", "weekly_savings_target"):
        try:
            if float(snapshot.get(key, 0.0) or 0.0) > 0.0:
                return True
        except Exception:
            continue
    return False


def _personal_finance_money(value: float, suffix: str = "") -> str:
    try:
        return f"£{float(value):,.0f}{suffix}"
    except Exception:
        return f"£0{suffix}"


def _personal_finance_target_value(snapshot: dict) -> float:
    if STEP1_TARGET_WEEKLY_SAVINGS in st.session_state:
        try:
            return float(st.session_state.get(STEP1_TARGET_WEEKLY_SAVINGS, 0.0) or 0.0)
        except Exception:
            return 0.0
    if isinstance(snapshot, dict):
        for key in ("target_a_weekly", "target_weekly", "weekly_savings_target"):
            try:
                value = float(snapshot.get(key, 0.0) or 0.0)
                if value > 0.0:
                    return value
            except Exception:
                continue
    return 0.0


def _personal_finance_feasibility_preview(snapshot: dict) -> str:
    """Return a compact expected-case preview for the planner status cards.

    The full Step 3 renderer still owns the chart and detailed feasibility text.
    This helper only computes a small summary so the user can understand the
    state of the Personal Finance Planner without scrolling through every panel.
    """
    if not isinstance(snapshot, dict) or not snapshot:
        return "Locked"
    try:
        from ui.services.step3_results_service import build_step3_view_model

        view_model = build_step3_view_model(snapshot)
        summary = view_model.get("summary", {}) if isinstance(view_model, dict) else {}
        expected = float(summary.get("expected_final", 0.0) or 0.0)
        if expected:
            return f"Expected {_personal_finance_money(expected)}"
    except Exception:
        pass
    return "Ready"


def _render_personal_finance_card(
    *,
    number: int,
    title: str,
    status: str,
    detail: str,
    locked: bool = False,
) -> None:
    border = "rgba(49, 51, 63, 0.18)"
    background = "#ffffff" if not locked else "#f8fafc"
    badge_bg = "#ecfdf5" if "✓" in status or "Ready" in status or "Confirmed" in status else "#f3f4f6"
    badge_color = "#047857" if "✓" in status or "Ready" in status or "Confirmed" in status else "#6b7280"
    st.markdown(
        f"""
        <div style="
            border: 1px solid {border};
            border-radius: 16px;
            padding: 1.0rem 1.0rem;
            min-height: 132px;
            background: {background};
            box-shadow: 0 1px 8px rgba(0,0,0,0.035);
        ">
            <div style="font-size: 0.82rem; color: #6b7280; margin-bottom: 0.35rem;">
                Step {number}
            </div>
            <div style="font-size: 1.05rem; font-weight: 700; margin-bottom: 0.55rem;">
                {title}
            </div>
            <span style="
                display: inline-block;
                padding: 0.18rem 0.5rem;
                border-radius: 999px;
                background: {badge_bg};
                color: {badge_color};
                font-size: 0.78rem;
                font-weight: 650;
                margin-bottom: 0.6rem;
            ">{status}</span>
            <div style="font-size: 0.88rem; color: #4b5563; margin-top: 0.55rem; line-height: 1.35;">
                {detail}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_personal_finance_status_cards(snapshot: dict) -> dict:
    confirmed = _personal_finance_snapshot_confirmed()
    positive_margin = _personal_finance_has_positive_margin(snapshot)
    has_target = _personal_finance_has_target(snapshot)
    target_weekly = _personal_finance_target_value(snapshot)

    margin_weekly = 0.0
    if isinstance(snapshot, dict) and snapshot:
        try:
            payload = get_weekly_reality_payload(snapshot)
            margin_weekly = float(payload.get("margin_weekly", 0.0) or 0.0)
        except Exception:
            margin_weekly = float(snapshot.get("baseline_savings_weekly", 0.0) or 0.0)

    feasibility_ready = bool(confirmed and positive_margin and has_target)

    c1, c2, c3 = st.columns(3)
    with c1:
        _render_personal_finance_card(
            number=1,
            title="Current situation",
            status="Confirmed ✓" if confirmed else "Needs confirmation",
            detail=(
                f"Margin: {_personal_finance_money(margin_weekly, '/week')}"
                if confirmed
                else "Enter income and spending, then confirm your snapshot."
            ),
            locked=False,
        )
    with c2:
        _render_personal_finance_card(
            number=2,
            title="Savings target",
            status=(f"{_personal_finance_money(target_weekly, '/week')}" if has_target else "Locked"),
            detail=(
                str(st.session_state.get(STEP2_SELECTED_PRESET, "Recommended") or "Recommended") + " preset"
                if has_target
                else "Unlocks after the current situation is confirmed."
            ),
            locked=not confirmed,
        )
    with c3:
        _render_personal_finance_card(
            number=3,
            title="Short-term feasibility",
            status="Ready ✓" if feasibility_ready else "Locked",
            detail=(
                _personal_finance_feasibility_preview(snapshot)
                if feasibility_ready
                else "Unlocks after a positive margin and savings target are available."
            ),
            locked=not feasibility_ready,
        )

    return {
        "confirmed": confirmed,
        "positive_margin": positive_margin,
        "has_target": has_target,
        "feasibility_ready": feasibility_ready,
    }


def _personal_finance_next_step_from_pathway() -> tuple[str, int]:
    pathway = str(st.session_state.get("step0_planning_pathway", "compare_both") or "compare_both")
    if pathway == "savings_only":
        return "Continue to Long-Term Scenarios", 6
    if pathway == "savings_plus_investing":
        return "Continue to Investment Lab", 4
    return "Continue to Risk Profile and Asset Universe →", 4


# -----------------------------------------------------------------------------
# Combined Personal Finance Planner state and dashboard helpers
# -----------------------------------------------------------------------------
PERSONAL_FINANCE_ACTIVE_PANEL_KEY = "personal_finance_active_panel_v1"
PERSONAL_FINANCE_LAST_PANEL_KEY = "personal_finance_last_active_panel_v1"
STEP1_QUICK_MARGIN_WEEKLY_KEY = "step1_quick_margin_weekly_v1"
STEP1_QUICK_BASE_MARGIN_WEEKLY_KEY = "step1_quick_base_margin_weekly_v2"
STEP1_QUICK_MARGIN_PENDING_KEY = "step1_quick_margin_pending_v2"
STEP1_QUICK_MARGIN_MIGRATED_KEY = "step1_quick_margin_migrated_v2"
STEP1_SHOW_EXACT_EDITING_KEY = "step1_show_exact_budget_editing_v1"

_STRESS_PRESET_OPTIONS = ["None", "Minor unexpected expense", "Major monthly shock", "Severe emergency shock"]
_STRESS_PRESET_EVENTS = {
    "None": {"amount": 0.0, "label": "No one-off cost is applied."},
    "Minor unexpected expense": {
        "amount": 250.0,
        "label": "Applies a one-off £250 cost around the middle of the selected horizon.",
    },
    "Major monthly shock": {
        "amount": 600.0,
        "label": "Applies a one-off £600 cost around the middle of the selected horizon.",
    },
    "Severe emergency shock": {
        "amount": 1000.0,
        "label": "Applies a one-off £1,000 cost around the middle of the selected horizon.",
    },
}
_LEGACY_STRESS_PRESET_ALIASES = {
    "Typical bump": "Minor unexpected expense",
    "Tough month": "Major monthly shock",
    "Heavy shock": "Severe emergency shock",
}


def _normalize_life_event_stress_preset(value: object) -> str:
    preset = str(value or "None").strip()
    preset = _LEGACY_STRESS_PRESET_ALIASES.get(preset, preset)
    return preset if preset in _STRESS_PRESET_OPTIONS else "None"


def _life_event_stress_payload(*, horizon_weeks: int | None = None) -> dict[str, object]:
    """Build the compact Step 3 one-off life-event shock payload.

    This mirrors the legacy Step 3 assumption control: the selected life-event
    stress preset is a single one-off cost applied around the middle of the
    selected short-term horizon, not an averaged weekly drag.
    """
    raw_horizon = horizon_weeks
    if raw_horizon is None:
        raw_horizon = st.session_state.get(STEP2_PLANNING_HORIZON_WEEKS, 12)
    try:
        horizon = max(1, min(52, int(raw_horizon or 12)))
    except Exception:
        horizon = 12

    selected = _normalize_life_event_stress_preset(st.session_state.get("step3_stress_preset", "None"))

    event_meta = dict(_STRESS_PRESET_EVENTS.get(selected, _STRESS_PRESET_EVENTS["None"]))
    amount = safe_float(event_meta.get("amount", 0.0), 0.0)
    shock_week = max(1, min(horizon, int(round(float(horizon) / 2.0))))
    enabled = bool(amount > 0.0)

    return {
        "stress_preset": selected,
        "stress_label": str(event_meta.get("label", "")),
        "shock_enabled": enabled,
        "shock_amount": float(amount),
        "shock_week": int(shock_week),
        "enable_one_off_events": enabled,
        "one_off_events": ([{"name": selected, "amount": float(amount), "week": int(shock_week)}] if enabled else []),
        "shock_map": ({int(shock_week): -float(amount)} if enabled else {}),
    }


def _apply_dashboard_life_event_stress_to_snapshot(snapshot: dict, *, persist: bool = True) -> dict:
    """Attach the compact life-event stress preset to the planning snapshot.

    The Personal Finance Planner renders Step 3 in compact form, so it must
    explicitly persist the same compatibility fields that the legacy Step 3
    page creates: enable_one_off_events, one_off_events and shock_map.
    """
    updated = dict(snapshot or {}) if isinstance(snapshot, dict) else {}

    raw_horizon = st.session_state.get(
        STEP2_PLANNING_HORIZON_WEEKS,
        updated.get("planning_horizon_weeks", 12),
    )
    try:
        horizon = max(1, min(52, int(raw_horizon or 12)))
    except Exception:
        horizon = 12

    payload = _life_event_stress_payload(horizon_weeks=horizon)
    updated["planning_horizon_weeks"] = int(horizon)
    updated["stress_preset"] = str(payload["stress_preset"])
    updated["enable_one_off_events"] = bool(payload["enable_one_off_events"])
    updated["shock_enabled"] = bool(payload["shock_enabled"])
    updated["shock_amount"] = float(payload["shock_amount"])
    updated["shock_week"] = int(payload["shock_week"])
    updated["one_off_events"] = list(payload["one_off_events"])
    updated["shock_map"] = dict(payload["shock_map"])

    if persist:
        st.session_state[PLANNING_SNAPSHOT] = updated
    return updated


def _life_event_stress_caption(snapshot: dict | None = None) -> str:
    source = dict(snapshot or {}) if isinstance(snapshot, dict) else {}
    preset = _normalize_life_event_stress_preset(
        source.get("stress_preset", st.session_state.get("step3_stress_preset", "None"))
    )
    amount = safe_float(source.get("shock_amount", _STRESS_PRESET_EVENTS.get(preset, {}).get("amount", 0.0)), 0.0)
    week = int(source.get("shock_week", _life_event_stress_payload().get("shock_week", 1)) or 1)
    if amount <= 0.0:
        return "No one-off life-event cost is applied."
    return f"Applies a one-off £{amount:,.0f} cost around week {week}."


def _period_to_weekly(amount: float, period: str) -> float:
    period = str(period or "Weekly")
    amount = safe_float(amount, 0.0)
    if period == "Monthly":
        return float(amount) * 12.0 / 52.0
    if period == "Yearly":
        return float(amount) / 52.0
    return float(amount)


def _weekly_to_period_amount(weekly: float, period: str) -> float:
    period = str(period or "Weekly")
    weekly = safe_float(weekly, 0.0)
    if period == "Monthly":
        return float(weekly) * 52.0 / 12.0
    if period == "Yearly":
        return float(weekly) * 52.0
    return float(weekly)


def _ensure_budget_amount_defaults() -> None:
    """Seed the canonical Step 1 budget fields before autosave/widgets read them.

    The compact dashboard can render the detailed controls collapsed. In that
    state Streamlit has not instantiated the exact number inputs yet, so the
    canonical amount keys may be missing. If autosave reads missing keys as zero,
    the budget bar collapses after a preset click. This helper keeps one source
    of truth alive even when controls are hidden.
    """
    defaults = (
        (STEP1_INCOME_AMOUNT, STEP1_INCOME_PERIOD, 460.0),
        (STEP1_FIXED_AMOUNT, STEP1_FIXED_PERIOD, 185.0),
        (STEP1_VARIABLE_AMOUNT, STEP1_VARIABLE_PERIOD, 80.0),
        (STEP1_DISCRETIONARY_AMOUNT, STEP1_DISCRETIONARY_PERIOD, 35.0),
    )

    for amount_key, period_key, default_weekly in defaults:
        if period_key not in st.session_state:
            st.session_state[period_key] = "Weekly"
        if amount_key not in st.session_state:
            st.session_state[amount_key] = float(default_weekly)
            st.session_state[period_key] = "Weekly"

    current_weeklies = [
        _period_to_weekly(st.session_state.get(amount_key, 0.0), st.session_state.get(period_key, "Weekly"))
        for amount_key, period_key, _default_weekly in defaults
    ]
    if all(float(value or 0.0) <= 0.0 for value in current_weeklies):
        for amount_key, period_key, default_weekly in defaults:
            st.session_state[amount_key] = float(default_weekly)
            st.session_state[period_key] = "Weekly"
        st.session_state[STEP1_QUICK_BASE_MARGIN_WEEKLY_KEY] = 160.0
        st.session_state[STEP1_QUICK_MARGIN_WEEKLY_KEY] = 160.0


def _step1_quick_weekly_values() -> dict[str, float]:
    income = _period_to_weekly(
        st.session_state.get(STEP1_INCOME_AMOUNT, 460.0),
        st.session_state.get(STEP1_INCOME_PERIOD, "Weekly"),
    )
    fixed = _period_to_weekly(
        st.session_state.get(STEP1_FIXED_AMOUNT, 185.0),
        st.session_state.get(STEP1_FIXED_PERIOD, "Weekly"),
    )
    variable = _period_to_weekly(
        st.session_state.get(STEP1_VARIABLE_AMOUNT, 80.0),
        st.session_state.get(STEP1_VARIABLE_PERIOD, "Weekly"),
    )
    discretionary = _period_to_weekly(
        st.session_state.get(STEP1_DISCRETIONARY_AMOUNT, 35.0),
        st.session_state.get(STEP1_DISCRETIONARY_PERIOD, "Weekly"),
    )
    spending = fixed + variable + discretionary
    margin = income - spending
    return {
        "income": float(income),
        "fixed": float(fixed),
        "variable": float(variable),
        "discretionary": float(discretionary),
        "spending": float(spending),
        "margin": float(margin),
        "max_margin_before_discretionary": float(max(0.0, income - fixed - variable)),
    }


def _apply_quick_margin_to_discretionary(desired_margin_weekly: float) -> None:
    values = _step1_quick_weekly_values()
    income = float(values["income"])
    fixed = float(values["fixed"])
    variable = float(values["variable"])
    desired_margin_weekly = float(desired_margin_weekly)

    # Keep income/fixed/variable stable and absorb the quick-estimate change in
    # discretionary spending. This preserves the existing Step 1 data model while
    # giving the user one simple control for the first-pass estimate.
    discretionary_weekly = max(0.0, income - fixed - variable - desired_margin_weekly)
    discretionary_period = str(st.session_state.get(STEP1_DISCRETIONARY_PERIOD, "Weekly") or "Weekly")
    st.session_state[STEP1_DISCRETIONARY_AMOUNT] = float(
        round(_weekly_to_period_amount(discretionary_weekly, discretionary_period), 2)
    )


def _render_budget_colour_bar(values: dict[str, float]) -> None:
    income = max(float(values.get("income", 0.0) or 0.0), 1.0)
    fixed = max(float(values.get("fixed", 0.0) or 0.0), 0.0)
    variable = max(float(values.get("variable", 0.0) or 0.0), 0.0)
    discretionary = max(float(values.get("discretionary", 0.0) or 0.0), 0.0)
    margin = float(values.get("margin", 0.0) or 0.0)
    free_margin = max(margin, 0.0)
    deficit = abs(min(margin, 0.0))

    income_colour = "#f1f5f9"
    fixed_colour = "#64748b"
    variable_colour = "#94a3b8"
    discretionary_colour = "#cbd5e1"
    free_margin_colour = "#22c55e"
    deficit_colour = "#ef4444"

    margin_colour = free_margin_colour if margin >= 0 else deficit_colour
    margin_label = "Free margin" if margin >= 0 else "Deficit"
    margin_help = "Money left after spending" if margin >= 0 else "Spending above income"

    total = max(
        income,
        fixed + variable + discretionary + free_margin,
        fixed + variable + discretionary + deficit,
        1.0,
    )

    segments = [
        ("Fixed essentials", fixed, fixed_colour),
        ("Variable essentials", variable, variable_colour),
        ("Discretionary", discretionary, discretionary_colour),
    ]

    if free_margin > 0:
        segments.append(("Free margin", free_margin, free_margin_colour))
    if deficit > 0:
        segments.append(("Deficit", deficit, deficit_colour))

    pieces = []
    for label, amount, colour in segments:
        width = max(2.0, min(100.0, 100.0 * float(amount) / float(total))) if amount > 0 else 0.0
        if width <= 0:
            continue
        pieces.append(
            f'<div title="{label}: £{amount:,.0f}/week" '
            f'style="width:{width:.2f}%; background:{colour}; height:22px;"></div>'
        )

    fixed_swatch = (
        f'<span style="width:0.74rem; height:0.74rem; border-radius:0.22rem; '
        f'background:{fixed_colour}; display:inline-block;"></span>'
    )
    variable_swatch = (
        f'<span style="width:0.74rem; height:0.74rem; border-radius:0.22rem; '
        f'background:{variable_colour}; display:inline-block;"></span>'
    )
    discretionary_swatch = (
        f'<span style="width:0.74rem; height:0.74rem; border-radius:0.22rem; '
        f'background:{discretionary_colour}; border:1px solid #cbd5e1; display:inline-block;"></span>'
    )
    margin_swatch = (
        f'<span style="width:0.74rem; height:0.74rem; border-radius:0.22rem; '
        f'background:{margin_colour}; display:inline-block;"></span>'
    )

    legend_items = [
        ("Fixed essentials", "Rent, subscriptions, fixed bills", fixed_swatch),
        ("Variable essentials", "Groceries, utilities, transport", variable_swatch),
        ("Discretionary", "Flexible spending", discretionary_swatch),
        (margin_label, margin_help, margin_swatch),
    ]

    legend_html = "".join(
        f'<div style="display:flex; align-items:center; gap:0.45rem; font-size:0.82rem; color:#334155; line-height:1.35;">'
        f'{swatch}'
        f'<span><b>{label}</b> · {help_text}</span>'
        f'</div>'
        for label, help_text, swatch in legend_items
    )

    st.markdown(
        f"""
        <div style="border:1px solid rgba(49,51,63,0.14); border-radius:14px; padding:0.85rem; background:#ffffff; box-shadow:0 1px 6px rgba(0,0,0,0.03);">
            <div style="display:flex; overflow:hidden; border-radius:999px; height:22px; background:{income_colour}; margin-bottom:0.65rem;">
                {''.join(pieces)}
            </div>
            <div style="display:grid; grid-template-columns:1fr; gap:0.4rem;">
                {legend_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _auto_save_current_situation_snapshot() -> None:
    """Persist the current Step 1 estimate without asking for a confirm button."""
    try:
        _ensure_budget_amount_defaults()
        snapshot = prepare_step1_preview()
        confirm_step1_snapshot(snapshot)
        st.session_state[CONFIRMED_SNAPSHOT] = True
        st.session_state["step1_auto_saved_snapshot"] = True
    except Exception:
        # Keep the UI resilient: if the snapshot cannot be built, the existing
        # gating messages will still keep Step 2/3 locked.
        pass


def _queue_quick_margin_and_rerun(new_margin: float, *, min_value: float, max_value: float) -> None:
    clamped = max(float(min_value), min(float(max_value), float(new_margin)))
    st.session_state[STEP1_QUICK_MARGIN_PENDING_KEY] = float(round(clamped))
    st.rerun()


_STEP1_BUDGET_SLIDER_SPECS = (
    (
        "step1_budget_slider_income_weekly_v1",
        STEP1_INCOME_AMOUNT,
        STEP1_INCOME_PERIOD,
        "Take-home income (£/week)",
        0,
        1500,
        10,
        460.0,
    ),
    (
        "step1_budget_slider_fixed_weekly_v1",
        STEP1_FIXED_AMOUNT,
        STEP1_FIXED_PERIOD,
        "Fixed essentials (£/week)",
        0,
        1200,
        5,
        185.0,
    ),
    (
        "step1_budget_slider_variable_weekly_v1",
        STEP1_VARIABLE_AMOUNT,
        STEP1_VARIABLE_PERIOD,
        "Variable essentials (£/week)",
        0,
        700,
        5,
        80.0,
    ),
    (
        "step1_budget_slider_discretionary_weekly_v1",
        STEP1_DISCRETIONARY_AMOUNT,
        STEP1_DISCRETIONARY_PERIOD,
        "Discretionary spending (£/week)",
        0,
        500,
        5,
        35.0,
    ),
)


def _sync_budget_slider_defaults_from_amounts() -> None:
    """Sync the weekly budget sliders from the canonical Step 1 amount keys.

    This runs before the slider widgets are instantiated, so helper buttons and
    exact-number edits stay reflected in the slider positions on the next rerun.
    """
    _ensure_budget_amount_defaults()
    for slider_key, amount_key, period_key, _label, min_value, max_value, _step, default in _STEP1_BUDGET_SLIDER_SPECS:
        weekly_value = _period_to_weekly(
            st.session_state.get(amount_key, default),
            st.session_state.get(period_key, "Weekly"),
        )
        st.session_state[slider_key] = float(max(float(min_value), min(float(max_value), float(weekly_value))))


def _apply_budget_slider_to_amount(slider_key: str, amount_key: str, period_key: str) -> None:
    """Copy a weekly slider value into the existing Step 1 amount model."""
    value = safe_float(st.session_state.get(slider_key, 0.0), 0.0)
    st.session_state[amount_key] = float(round(max(value, 0.0), 2))
    st.session_state[period_key] = "Weekly"
    st.session_state[CONFIRMED_SNAPSHOT] = False
    invalidate_downstream_from_step1()


def _render_budget_value_sliders() -> None:
    """Render the detailed budget controls as weekly sliders."""
    _sync_budget_slider_defaults_from_amounts()

    st.markdown("#### Budget sliders")
    st.caption(
        "Use these to tune the weekly income and spending categories. Free margin is calculated from these values."
    )

    row1 = st.columns(2)
    row2 = st.columns(2)
    slots = [row1[0], row1[1], row2[0], row2[1]]

    for slot, spec in zip(slots, _STEP1_BUDGET_SLIDER_SPECS):
        slider_key, amount_key, period_key, label, min_value, max_value, step, default = spec
        with slot:
            st.slider(
                label,
                min_value=int(min_value),
                max_value=int(max_value),
                step=int(step),
                key=slider_key,
                on_change=_apply_budget_slider_to_amount,
                args=(slider_key, amount_key, period_key),
            )


def _ensure_quick_margin_state() -> tuple[int, int, int, float]:
    """Initialise slider state before the slider widget is instantiated.

    Streamlit does not allow changing a widget-backed session_state key after
    the widget is created. Preset buttons therefore write to a pending key and
    this helper consumes it before rendering the slider.
    """
    values = _step1_quick_weekly_values()
    current_margin = float(values.get("margin", 0.0) or 0.0)
    stored_base_margin = safe_float(st.session_state.get(STEP1_QUICK_BASE_MARGIN_WEEKLY_KEY, current_margin), current_margin)
    if STEP1_QUICK_BASE_MARGIN_WEEKLY_KEY not in st.session_state or (stored_base_margin <= 0.0 and current_margin > 0.0):
        st.session_state[STEP1_QUICK_BASE_MARGIN_WEEKLY_KEY] = float(round(current_margin))

    base_margin = float(st.session_state.get(STEP1_QUICK_BASE_MARGIN_WEEKLY_KEY, current_margin) or current_margin)

    # The quick slider is a *positive weekly surplus* control. Users do not
    # normally think of "having -£40 left"; deficits are still detected when
    # exact income/spending inputs imply spending > income, but the simple
    # first-pass slider starts at zero and moves upward.
    slider_min = 0
    slider_max = int(max(400.0, max(0.0, base_margin) + 250.0))
    default_value = int(round(max(float(slider_min), min(float(slider_max), max(0.0, base_margin)))))

    pending = st.session_state.pop(STEP1_QUICK_MARGIN_PENDING_KEY, None)
    if pending is not None:
        selected_value = int(round(max(float(slider_min), min(float(slider_max), float(pending)))))
        st.session_state[STEP1_QUICK_MARGIN_WEEKLY_KEY] = float(selected_value)
        _apply_quick_margin_to_discretionary(float(selected_value))
        st.session_state[CONFIRMED_SNAPSHOT] = False
    elif STEP1_QUICK_MARGIN_WEEKLY_KEY not in st.session_state:
        st.session_state[STEP1_QUICK_MARGIN_WEEKLY_KEY] = float(default_value)
        _apply_quick_margin_to_discretionary(float(default_value))
    else:
        try:
            stored = float(st.session_state.get(STEP1_QUICK_MARGIN_WEEKLY_KEY, default_value) or default_value)
            if not bool(st.session_state.get(STEP1_QUICK_MARGIN_MIGRATED_KEY, False)):
                # Older quick-slider builds could persist a negative extreme.
                # Reset that once when the stable base margin is positive.
                if base_margin > 0 and stored < 0:
                    stored = float(default_value)
                    st.session_state[STEP1_QUICK_MARGIN_WEEKLY_KEY] = float(default_value)
                    _apply_quick_margin_to_discretionary(float(default_value))
                st.session_state[STEP1_QUICK_MARGIN_MIGRATED_KEY] = True
            if stored < slider_min or stored > slider_max:
                st.session_state[STEP1_QUICK_MARGIN_WEEKLY_KEY] = float(default_value)
                _apply_quick_margin_to_discretionary(float(default_value))
        except Exception:
            st.session_state[STEP1_QUICK_MARGIN_WEEKLY_KEY] = float(default_value)
            _apply_quick_margin_to_discretionary(float(default_value))

    return int(slider_min), int(slider_max), int(default_value), float(base_margin)


def _render_step1_quick_estimate_panel() -> dict[str, float]:
    _, success_message = apply_pending_step_patch(
        STEP1_PENDING_WIDGET_PATCH,
        STEP1_PENDING_SUCCESS_MESSAGE,
    )
    if success_message:
        show_toast_or_success(success_message, icon="✅", fallback_level="success")
    seed_step1_from_snapshot_if_missing()
    _ensure_budget_amount_defaults()

    st.markdown("### Budget estimate")
    st.caption("Start with a quick preset or open detailed budget controls to tune the income and spending categories.")

    slider_min, slider_max, _, base_margin = _ensure_quick_margin_state()
    values = _step1_quick_weekly_values()

    _render_budget_colour_bar(values)

    st.caption(
        "Quick presets adjust discretionary/flexible spending only. Income, fixed essentials, "
        "and variable essentials stay unchanged."
    )

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button(
            "Quiet week",
            key="step1_quick_quiet_week",
            use_container_width=True,
            help="Lower flexible spending and increase the estimated weekly free margin.",
        ):
            _queue_quick_margin_and_rerun(base_margin + 50.0, min_value=slider_min, max_value=slider_max)
    with b2:
        if st.button(
            "Typical week",
            key="step1_quick_typical_week",
            use_container_width=True,
            help="Return flexible spending to the typical estimate.",
        ):
            _queue_quick_margin_and_rerun(base_margin, min_value=slider_min, max_value=slider_max)
    with b3:
        if st.button(
            "Expensive week",
            key="step1_quick_expensive_week",
            use_container_width=True,
            help="Increase flexible spending and reduce the estimated weekly free margin.",
        ):
            _queue_quick_margin_and_rerun(base_margin - 50.0, min_value=slider_min, max_value=slider_max)

    if float(values.get("margin", 0.0) or 0.0) < 0.0:
        st.warning(
            f"This estimate is short by about **£{abs(float(values['margin'])):,.0f}/week**. "
            "Use the presets or open detailed budget controls if the income/spending split looks wrong."
        )

    with st.expander("Detailed budget controls", expanded=False):
        st.caption(
            "Adjust the main weekly budget values first. Optional tools below can estimate income, break down essentials, "
            "or show exact number inputs when needed."
        )

        _render_budget_value_sliders()
        st.divider()
        st.markdown("#### Optional tools")

        toggle_cols = st.columns(2)
        with toggle_cols[0]:
            show_take_home_estimator = st.checkbox(
                "Estimate take-home income from gross pay",
                key="step1_adv_show_take_home_estimator_v1",
            )
            show_fixed_breakdown = st.checkbox(
                "Build fixed essentials breakdown",
                key="step1_adv_show_fixed_breakdown_v1",
            )
        with toggle_cols[1]:
            show_variable_breakdown = st.checkbox(
                "Build variable essentials breakdown",
                key="step1_adv_show_variable_breakdown_v1",
            )
            show_exact_totals = st.checkbox(
                "Edit exact income and spending totals",
                value=False,
                key="step1_adv_show_exact_totals_v2",
            )

        # IMPORTANT: the helper sections below appear before the exact total
        # widgets. Their buttons update the same Step 1 amount keys, so they must
        # run before those widget keys are instantiated in this Streamlit rerun.
        if show_take_home_estimator:
            st.markdown("#### Estimate take-home income")
            e1, e2 = st.columns(2)
            with e1:
                gross_amount = st.number_input(
                    "Gross income (£)",
                    min_value=0.0,
                    value=safe_float(st.session_state.get(STEP1_ESTIMATED_GROSS_INCOME, 30000.0), 30000.0),
                    step=10.0,
                    key=STEP1_ESTIMATED_GROSS_INCOME,
                )
            with e2:
                gross_period = st.selectbox(
                    "Gross income period",
                    _PERIOD_OPTIONS,
                    index=_PERIOD_OPTIONS.index(str(st.session_state.get(STEP1_ESTIMATED_GROSS_PERIOD, "Yearly"))),
                    key=STEP1_ESTIMATED_GROSS_PERIOD,
                )
            tax_rate = st.slider(
                "Estimated deduction rate",
                min_value=0.0,
                max_value=0.60,
                value=float(st.session_state.get(STEP1_ESTIMATED_TAX_RATE, 0.30) or 0.30),
                step=0.01,
                key=STEP1_ESTIMATED_TAX_RATE,
            )
            estimated_take_home = float(gross_amount) * (1.0 - float(tax_rate))
            st.caption(f"Estimated take-home: £{estimated_take_home:,.2f} per {str(gross_period).lower()}")
            if st.button("Use this estimate as my take-home income", key="step1_dashboard_apply_income_estimate"):
                invalidate_downstream_from_step1()
                queue_step_patch(
                    STEP1_PENDING_WIDGET_PATCH,
                    {STEP1_INCOME_AMOUNT: float(estimated_take_home), STEP1_INCOME_PERIOD: str(gross_period)},
                    success_key=STEP1_PENDING_SUCCESS_MESSAGE,
                    success_message=step1_income_estimate_applied(),
                )
                st.rerun()

        if show_fixed_breakdown:
            st.markdown("#### Build fixed essentials")
            editor_df = ensure_fixed_breakdown_df()
            with st.form("step1_dashboard_fixed_breakdown_form", clear_on_submit=False):
                edited_fixed_df = stateful_data_editor(
                    state_key=STEP1_FIXED_BREAKDOWN_DF,
                    editor_key="step1_dashboard_fixed_breakdown_editor",
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
                st.rerun()

        if show_variable_breakdown:
            st.markdown("#### Build variable essentials")
            v1, v2 = st.columns(2)
            with v1:
                st.number_input(
                    "Utilities baseline (£)",
                    min_value=0.0,
                    value=safe_float(st.session_state.get(STEP1_VAR_UTILITIES_AMOUNT, 0.0), 0.0),
                    step=5.0,
                    key=STEP1_VAR_UTILITIES_AMOUNT,
                )
                st.selectbox(
                    "Utilities period",
                    _PERIOD_OPTIONS,
                    index=_PERIOD_OPTIONS.index(str(st.session_state.get(STEP1_VAR_UTILITIES_PERIOD, "Weekly"))),
                    key=STEP1_VAR_UTILITIES_PERIOD,
                )
                st.selectbox(
                    "Season",
                    _SEASON_OPTIONS,
                    index=_SEASON_OPTIONS.index(str(st.session_state.get(STEP1_VAR_SEASON, "Normal"))),
                    key=STEP1_VAR_SEASON,
                )
                st.slider(
                    "Commute days/week",
                    min_value=0,
                    max_value=7,
                    value=int(st.session_state.get(STEP1_VAR_COMMUTE_DAYS, 0) or 0),
                    key=STEP1_VAR_COMMUTE_DAYS,
                )
                st.number_input(
                    "Cost per commute day (£)",
                    min_value=0.0,
                    value=safe_float(st.session_state.get(STEP1_VAR_COMMUTE_COST, 0.0), 0.0),
                    step=1.0,
                    key=STEP1_VAR_COMMUTE_COST,
                )
            with v2:
                st.number_input(
                    "Groceries (£)",
                    min_value=0.0,
                    value=safe_float(st.session_state.get(STEP1_VAR_GROCERIES_AMOUNT, 0.0), 0.0),
                    step=5.0,
                    key=STEP1_VAR_GROCERIES_AMOUNT,
                )
                st.selectbox(
                    "Groceries period",
                    _PERIOD_OPTIONS,
                    index=_PERIOD_OPTIONS.index(str(st.session_state.get(STEP1_VAR_GROCERIES_PERIOD, "Weekly"))),
                    key=STEP1_VAR_GROCERIES_PERIOD,
                )
                st.number_input(
                    "Household basics (£)",
                    min_value=0.0,
                    value=safe_float(st.session_state.get(STEP1_VAR_HOUSEHOLD_AMOUNT, 0.0), 0.0),
                    step=5.0,
                    key=STEP1_VAR_HOUSEHOLD_AMOUNT,
                )
                st.selectbox(
                    "Household period",
                    _PERIOD_OPTIONS,
                    index=_PERIOD_OPTIONS.index(str(st.session_state.get(STEP1_VAR_HOUSEHOLD_PERIOD, "Weekly"))),
                    key=STEP1_VAR_HOUSEHOLD_PERIOD,
                )

            variable_total_weekly = variable_breakdown_weekly_total()
            st.write(f"**Estimated total:** £{variable_total_weekly:,.2f}/week")
            if st.button("Use this total as my Variable essentials", key="step1_dashboard_apply_variable_breakdown"):
                invalidate_downstream_from_step1()
                queue_step_patch(
                    STEP1_PENDING_WIDGET_PATCH,
                    {STEP1_VARIABLE_AMOUNT: float(round(variable_total_weekly, 2)), STEP1_VARIABLE_PERIOD: "Weekly"},
                    success_key=STEP1_PENDING_SUCCESS_MESSAGE,
                    success_message=step1_variable_breakdown_applied(),
                )
                st.rerun()

        if show_exact_totals:
            st.markdown("#### Exact income and spending totals")
            c1, c2 = st.columns([2.1, 1.0])
            with c1:
                st.number_input(
                    "Take-home income (£)",
                    min_value=0.0,
                    value=safe_float(st.session_state.get(STEP1_INCOME_AMOUNT, 460.0), 460.0),
                    step=10.0,
                    key=STEP1_INCOME_AMOUNT,
                )
            with c2:
                st.selectbox(
                    "Income period",
                    _PERIOD_OPTIONS,
                    index=_PERIOD_OPTIONS.index(str(st.session_state.get(STEP1_INCOME_PERIOD, "Weekly"))),
                    key=STEP1_INCOME_PERIOD,
                )

            c3, c4 = st.columns([2.1, 1.0])
            with c3:
                st.number_input(
                    "Fixed essentials (£)",
                    min_value=0.0,
                    value=safe_float(st.session_state.get(STEP1_FIXED_AMOUNT, 185.0), 185.0),
                    step=5.0,
                    key=STEP1_FIXED_AMOUNT,
                )
            with c4:
                st.selectbox(
                    "Fixed period",
                    _PERIOD_OPTIONS,
                    index=_PERIOD_OPTIONS.index(str(st.session_state.get(STEP1_FIXED_PERIOD, "Weekly"))),
                    key=STEP1_FIXED_PERIOD,
                )

            c5, c6 = st.columns([2.1, 1.0])
            with c5:
                st.number_input(
                    "Variable essentials (£)",
                    min_value=0.0,
                    value=safe_float(st.session_state.get(STEP1_VARIABLE_AMOUNT, 80.0), 80.0),
                    step=5.0,
                    key=STEP1_VARIABLE_AMOUNT,
                )
            with c6:
                st.selectbox(
                    "Variable period",
                    _PERIOD_OPTIONS,
                    index=_PERIOD_OPTIONS.index(str(st.session_state.get(STEP1_VARIABLE_PERIOD, "Weekly"))),
                    key=STEP1_VARIABLE_PERIOD,
                )

            c7, c8 = st.columns([2.1, 1.0])
            with c7:
                st.number_input(
                    "Discretionary spending (£)",
                    min_value=0.0,
                    value=safe_float(st.session_state.get(STEP1_DISCRETIONARY_AMOUNT, 35.0), 35.0),
                    step=5.0,
                    key=STEP1_DISCRETIONARY_AMOUNT,
                )
            with c8:
                st.selectbox(
                    "Discretionary period",
                    _PERIOD_OPTIONS,
                    index=_PERIOD_OPTIONS.index(str(st.session_state.get(STEP1_DISCRETIONARY_PERIOD, "Weekly"))),
                    key=STEP1_DISCRETIONARY_PERIOD,
                )

        st.caption("Advanced edits are auto-saved before the target and feasibility cards update.")

    _auto_save_current_situation_snapshot()
    return _step1_quick_weekly_values()


def _render_cashflow_summary(values: dict[str, float], target_weekly: float) -> None:
    """Render the compact cash-flow summary.

    Target and feasibility status live in the cards below, so this top summary
    intentionally stays focused on the current weekly cash-flow baseline.
    The target_weekly parameter is kept for call-site compatibility.
    """
    income = float(values.get("income", 0.0) or 0.0)
    spending = float(values.get("spending", 0.0) or 0.0)
    margin = float(values.get("margin", 0.0) or 0.0)
    margin_colour = "#047857" if margin >= 0 else "#b91c1c"

    st.markdown(
        f"""
        <div style="border:1px solid rgba(49,51,63,0.14); border-radius:14px; padding:0.8rem 1rem; background:#ffffff; margin:0.5rem 0 1rem 0;">
            <div style="font-weight:700; margin-bottom:0.45rem;">Cash-flow summary</div>
            <div style="display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:0.75rem; align-items:start;">
                <div><div style="font-size:0.74rem;color:#64748b;">Income</div><div style="font-weight:700;">£{income:,.0f}/week</div></div>
                <div><div style="font-size:0.74rem;color:#64748b;">Spending</div><div style="font-weight:700;">£{spending:,.0f}/week</div></div>
                <div><div style="font-size:0.74rem;color:#64748b;">Free margin</div><div style="font-weight:700;color:{margin_colour};">£{margin:,.0f}/week</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_period_equivalents_table(values: dict[str, float]) -> None:
    """Render the current weekly baseline as weekly/monthly/yearly equivalents."""
    income_w = float(values.get("income", 0.0) or 0.0)
    fixed_w = float(values.get("fixed", 0.0) or 0.0)
    variable_w = float(values.get("variable", 0.0) or 0.0)
    discretionary_w = float(values.get("discretionary", 0.0) or 0.0)
    spending_w = fixed_w + variable_w + discretionary_w
    margin_w = income_w - spending_w

    rows = [
        ("Income", income_w),
        ("Essentials", fixed_w + variable_w),
        ("Discretionary", discretionary_w),
        ("Free margin", margin_w),
    ]
    body = "".join(
        f"""
        <tr>
            <td style="padding:0.42rem 0.55rem; font-weight:650;">{name}</td>
            <td style="padding:0.42rem 0.55rem; text-align:right;">£{weekly:,.0f}/w</td>
            <td style="padding:0.42rem 0.55rem; text-align:right;">£{weekly * 52.0 / 12.0:,.0f}/mo</td>
            <td style="padding:0.42rem 0.55rem; text-align:right;">£{weekly * 52.0:,.0f}/yr</td>
        </tr>
        """
        for name, weekly in rows
    )
    st.markdown(
        f"""
        <div style="border:1px solid rgba(49,51,63,0.14); border-radius:12px; overflow:hidden; margin:0.5rem 0 1rem 0;">
            <table style="width:100%; border-collapse:collapse; font-size:0.88rem;">
                <thead style="background:#f8fafc; color:#475569;">
                    <tr>
                        <th style="padding:0.45rem 0.55rem; text-align:left;">Category</th>
                        <th style="padding:0.45rem 0.55rem; text-align:right;">Weekly</th>
                        <th style="padding:0.45rem 0.55rem; text-align:right;">Monthly</th>
                        <th style="padding:0.45rem 0.55rem; text-align:right;">Yearly</th>
                    </tr>
                </thead>
                <tbody>{body}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _target_service_payload(snapshot: dict, target_weekly: float | None = None) -> tuple[dict, dict, dict, str, int]:
    """Build and persist the Step 2-compatible target payload.

    The combined dashboard does not render the old Step 2 page directly, but
    downstream modules still expect the same snapshot fields. This helper keeps
    the compact Personal Finance Planner compatible with the Step 2 service layer.
    """
    from ui.services.step2_goal_service import (
        UNCERTAINTY_OPTIONS,
        build_feasibility_view_model,
        build_step2_snapshot_update,
        build_target_context,
        initialize_step2_state,
        persist_step2_snapshot,
        resolve_intent,
    )

    if not isinstance(snapshot, dict):
        snapshot = {}
    intent = resolve_intent(snapshot)
    init_state = initialize_step2_state(snapshot)
    planning_horizon = int(st.session_state.get(STEP2_PLANNING_HORIZON_WEEKS, init_state.get("planning_horizon", 12)) or 12)
    target_context = build_target_context(snapshot, intent, planning_horizon)
    if target_weekly is None:
        target_weekly = float(
            st.session_state.get(STEP1_TARGET_WEEKLY_SAVINGS, target_context.get("recommended_target", 0.0)) or 0.0
        )

    uncertainty_preset = str(st.session_state.get(STEP2_UNCERTAINTY_PRESET, UNCERTAINTY_OPTIONS[0]) or UNCERTAINTY_OPTIONS[0])
    random_run_nonce = int(st.session_state.get(STEP2_RANDOM_RUN_NONCE, 0) or 0)
    feasibility = build_feasibility_view_model(
        snapshot,
        intent=intent,
        planning_horizon=int(planning_horizon),
        target_weekly=float(target_weekly or 0.0),
    )
    updated_snapshot = build_step2_snapshot_update(
        snapshot,
        intent=intent,
        planning_horizon=int(planning_horizon),
        uncertainty_preset=uncertainty_preset,
        random_run_nonce=random_run_nonce,
        feasibility=feasibility,
    )
    updated_snapshot = _apply_dashboard_life_event_stress_to_snapshot(updated_snapshot, persist=False)
    persist_step2_snapshot(updated_snapshot)
    st.session_state[PLANNING_SNAPSHOT] = updated_snapshot
    return updated_snapshot, target_context, feasibility, intent, int(planning_horizon)


def _apply_target_preset_and_rerun(value: float, label: str) -> None:
    """Apply a target preset without mutating an instantiated widget key.

    The compact dashboard renders STEP1_TARGET_WEEKLY_SAVINGS as a
    number_input before the preset buttons. Streamlit forbids assigning to that
    widget key later in the same render, so presets must use the existing
    pending-patch pattern and immediately rerun. Without the explicit rerun,
    the patch is only applied on the next unrelated click, making the sidebar
    look one interaction behind.
    """
    queue_step_patch(
        STEP1_PENDING_WIDGET_PATCH,
        {
            STEP1_TARGET_WEEKLY_SAVINGS: float(round(max(float(value), 0.0), 2)),
            STEP2_SELECTED_PRESET: str(label),
            STEP2_TARGET_USER_TOUCHED_INTERNAL: True,
        },
    )
    st.rerun()


def _render_savings_target_card(snapshot: dict) -> tuple[dict, dict, dict, float, int]:
    st.markdown("### Savings target")
    st.caption("Choose how much of your current free margin you want to set aside each week.")

    updated_snapshot = snapshot
    target_context: dict = {}
    feasibility: dict = {}
    planning_horizon = int(st.session_state.get(STEP2_PLANNING_HORIZON_WEEKS, 12) or 12)

    baseline_weekly = float(snapshot.get("baseline_savings_weekly", 0.0) or 0.0) if isinstance(snapshot, dict) else 0.0
    if baseline_weekly <= 0:
        st.info("Restore a positive weekly margin first. The target card will unlock automatically.")
        return updated_snapshot, target_context, feasibility, 0.0, planning_horizon

    try:
        _, target_context, _, _, planning_horizon = _target_service_payload(snapshot, None)
    except Exception:
        target_context = {
            "safe_target": max(0.0, baseline_weekly * 0.20),
            "recommended_target": max(0.0, baseline_weekly * 0.30),
            "ambitious_target": max(0.0, baseline_weekly * 0.40),
            "baseline_weekly": baseline_weekly,
        }

    if STEP1_TARGET_WEEKLY_SAVINGS not in st.session_state:
        st.session_state[STEP1_TARGET_WEEKLY_SAVINGS] = float(
            round(float(target_context.get("recommended_target", max(0.0, baseline_weekly * 0.30))), 2)
        )
        st.session_state[STEP2_SELECTED_PRESET] = "Recommended"

    max_target_weekly = max(0.0, float(baseline_weekly))
    current_target = float(st.session_state.get(STEP1_TARGET_WEEKLY_SAVINGS, 0.0) or 0.0)
    if current_target > max_target_weekly:
        st.session_state[STEP1_TARGET_WEEKLY_SAVINGS] = float(round(max_target_weekly, 2))

    preset_options = ["Safe", "Recommended", "Ambitious"]
    preset_display_labels = {
        "Safe": "Cautious",
        "Recommended": "Balanced",
        "Ambitious": "Stretch",
    }
    active_preset_before = str(st.session_state.get(STEP2_SELECTED_PRESET, "Recommended") or "Recommended")
    if active_preset_before not in preset_options:
        active_preset_before = "Recommended"
        st.session_state[STEP2_SELECTED_PRESET] = active_preset_before

    target_weekly = float(
        st.number_input(
            "Weekly savings target (£/week)",
            min_value=0.0,
            max_value=float(max_target_weekly),
            step=5.0,
            key=STEP1_TARGET_WEEKLY_SAVINGS,
            help=(
                "This is the amount you aim to set aside each week. "
                "It cannot be higher than the current weekly free margin."
            ),
        )
        or 0.0
    )
    st.session_state[STEP2_TARGET_USER_TOUCHED_INTERNAL] = True

    preset_cols = st.columns(3)
    for idx, preset_name in enumerate(preset_options):
        display_label = preset_display_labels.get(preset_name, preset_name)
        preset_value = {
            "Safe": float(target_context.get("safe_target", baseline_weekly * 0.20)),
            "Recommended": float(target_context.get("recommended_target", baseline_weekly * 0.30)),
            "Ambitious": float(target_context.get("ambitious_target", baseline_weekly * 0.40)),
        }.get(preset_name, float(target_context.get("recommended_target", baseline_weekly * 0.30)))
        preset_value = min(max_target_weekly, max(0.0, float(preset_value)))

        with preset_cols[idx]:
            if st.button(
                display_label,
                key=f"step1_dashboard_target_preset_{preset_name.lower()}_v2",
                use_container_width=True,
                type="primary" if preset_name == active_preset_before else "secondary",
                help=(
                    "Lower-pressure weekly target."
                    if preset_name == "Safe"
                    else "Balanced default weekly target."
                    if preset_name == "Recommended"
                    else "Higher-pressure stretch weekly target."
                ),
            ):
                _apply_target_preset_and_rerun(preset_value, preset_name)

    try:
        updated_snapshot, target_context, feasibility, _, planning_horizon = _target_service_payload(snapshot, target_weekly)
    except Exception:
        updated_snapshot = dict(snapshot)
        updated_snapshot["target_a_weekly"] = float(target_weekly)
        st.session_state[PLANNING_SNAPSHOT] = updated_snapshot
        feasibility = {}

    pct = (target_weekly / baseline_weekly) if baseline_weekly > 0 else 0.0
    remaining_after_target = max(0.0, baseline_weekly - target_weekly)
    st.caption(
        f"This weekly target uses about **{pct * 100:.0f}%** of the current weekly free margin, "
        f"leaving roughly **£{remaining_after_target:,.0f}/week** as a buffer."
    )

    if pct >= 0.85:
        st.warning(
            "This target uses most of the available weekly margin. It may still be possible, "
            "but the feasibility check is important because there is limited room for surprises."
        )

    return updated_snapshot, target_context, feasibility, float(target_weekly), int(planning_horizon)


def _short_money(value: float) -> str:
    try:
        value = float(value)
        if abs(value) >= 1000:
            return f"£{value / 1000.0:.1f}k"
        return f"£{value:,.0f}"
    except Exception:
        return "—"


def _render_dashboard_projection_chart(
    baseline_df,
    plan_a_df,
    *,
    target_weekly: float = 0.0,
    shock_week: int | None = None,
    shock_amount: float = 0.0,
) -> None:
    """Render the short-term feasibility path inside the compact dashboard."""
    if baseline_df is None or plan_a_df is None:
        st.info("Short-term path chart is not available yet.")
        return
    try:
        if baseline_df.empty or plan_a_df.empty:
            st.info("Short-term path chart is not available yet.")
            return
    except Exception:
        st.info("Short-term path chart is not available yet.")
        return

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9.5, 3.6))
        ax.plot(baseline_df["Week"], baseline_df["Balance"], label="Baseline", linewidth=2.2)
        ax.plot(plan_a_df["Week"], plan_a_df["Mean"], label="Target plan", linewidth=2.2)
        ax.fill_between(
            plan_a_df["Week"],
            plan_a_df["Lower"],
            plan_a_df["Upper"],
            alpha=0.15,
            label="Target plan range (10–90%)",
        )

        if float(target_weekly or 0.0) > 0.0:
            final_week = float(baseline_df["Week"].max())
            final_target = final_week * float(target_weekly)
            ax.scatter([final_week], [final_target], s=55, marker="o", label="Savings target", zorder=5)
            ax.annotate("target", xy=(final_week, final_target), xytext=(6, 6), textcoords="offset points", fontsize=9)

        try:
            marker_week = int(shock_week or 0)
            marker_amount = float(shock_amount or 0.0)
        except Exception:
            marker_week = 0
            marker_amount = 0.0
        if marker_week > 0 and marker_amount > 0.0:
            ax.axvline(marker_week, linestyle=":", linewidth=1.3, label="Life event stress")
            ax.annotate(
                f"one-off -£{marker_amount:,.0f}",
                xy=(marker_week, 0),
                xytext=(6, 18),
                textcoords="offset points",
                rotation=90,
                fontsize=8,
                va="bottom",
            )

        ax.axhline(0.0, linestyle="--", linewidth=1.0)
        ax.set_title("Short-term cash-flow path")
        ax.set_xlabel("Week")
        ax.set_ylabel("Balance (£)")
        ax.legend()
        st.pyplot(fig, clear_figure=True)
    except Exception as exc:
        st.info("Short-term path chart is unavailable for this scenario.")
        st.caption(str(exc))


def _build_dashboard_feasibility_view_model(snapshot: dict) -> dict:
    """Build Step 3's view model for the dashboard, returning {} on failure."""
    if not isinstance(snapshot, dict) or not snapshot:
        return {}
    try:
        from ui.services.step3_results_service import build_step3_view_model

        scenario_snapshot = _apply_dashboard_life_event_stress_to_snapshot(snapshot, persist=True)
        view_model = build_step3_view_model(scenario_snapshot)
        return view_model if isinstance(view_model, dict) else {}
    except Exception:
        return {}


def _render_feasibility_card(snapshot: dict, *, target_weekly: float) -> None:
    st.markdown("### Feasibility check")

    try:
        horizon_weeks = int(st.session_state.get(STEP2_PLANNING_HORIZON_WEEKS, 12) or 12)
    except Exception:
        horizon_weeks = 12

    st.caption(
        f"Checks whether the selected weekly savings target remains realistic over a "
        f"{horizon_weeks}-week short-term scenario."
    )

    if not isinstance(snapshot, dict) or not snapshot:
        st.info("Budget estimate is still being prepared.")
        return
    if float(snapshot.get("baseline_savings_weekly", 0.0) or 0.0) <= 0:
        st.info("Restore a positive weekly margin before running feasibility.")
        return
    if target_weekly <= 0:
        st.info("Choose a weekly savings target first.")
        return

    view_model = _build_dashboard_feasibility_view_model(snapshot)
    if not view_model:
        st.info("Feasibility preview is unavailable until all short-term planning assumptions are ready.")
        return

    summary = view_model.get("summary", {}) if isinstance(view_model, dict) else {}
    conservative = float(summary.get("conservative_final", 0.0) or 0.0)
    expected = float(summary.get("expected_final", 0.0) or 0.0)
    optimistic = float(summary.get("optimistic_final", 0.0) or 0.0)
    baseline_final = float(summary.get("baseline_final", 0.0) or 0.0)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Conservative", _short_money(conservative))
    with m2:
        st.metric("Expected", _short_money(expected))
    with m3:
        st.metric("High case", _short_money(optimistic))

    st.caption(
        "Expected is the central simulated outcome for the selected weekly target and short-term assumptions. "
        "Conservative and high case show lower and higher scenario outcomes."
    )

    expected_delta = expected - baseline_final
    conservative_delta = conservative - baseline_final
    if conservative_delta >= -max(25.0, abs(baseline_final) * 0.02) and expected_delta >= 0:
        st.success("This target looks feasible in the short-term test.")
    elif expected_delta >= 0:
        st.info("This target is feasible, but the cautious case leaves less room for surprises.")
    else:
        st.error(
            "This target is fragile under the stress-test assumptions. It still builds savings, "
            "but the expected result no longer clearly improves on your baseline."
        )


def _render_feasibility_chart_card(snapshot: dict, *, target_weekly: float) -> None:
    """Render the feasibility chart as its own dashboard card."""
    if not isinstance(snapshot, dict) or not snapshot:
        return
    if float(snapshot.get("baseline_savings_weekly", 0.0) or 0.0) <= 0 or float(target_weekly or 0.0) <= 0:
        return

    view_model = _build_dashboard_feasibility_view_model(snapshot)
    if not view_model:
        return

    baseline_df = view_model.get("baseline_df")
    plan_a_df = view_model.get("plan_a_df")
    payload = view_model.get("payload", {}) if isinstance(view_model, dict) else {}
    scenario_snapshot = _apply_dashboard_life_event_stress_to_snapshot(snapshot, persist=False)
    shock_amount = float(scenario_snapshot.get("shock_amount", payload.get("shock_amount", 0.0)) or 0.0)
    shock_week = int(scenario_snapshot.get("shock_week", payload.get("shock_week", 0)) or 0)
    with st.container(border=True):
        st.markdown("### Short-term path chart")
        st.caption(
            "Baseline shows the current cash-flow path using the confirmed budget estimate, before applying the selected weekly savings target. "
            "Target plan shows the simulated path after applying the weekly target. "
            "The shaded band shows the 10–90% uncertainty range for possible short-term variation."
        )
        if shock_amount > 0.0 and shock_week > 0:
            st.caption(f"Life-event stress is modelled as a one-off £{shock_amount:,.0f} cost around week {shock_week}.")
        _render_dashboard_projection_chart(
            baseline_df,
            plan_a_df,
            target_weekly=float(target_weekly),
            shock_week=shock_week,
            shock_amount=shock_amount,
        )


def _render_advanced_assumptions_and_diagnostics(snapshot: dict, values: dict[str, float]) -> None:
    from ui.services.step2_goal_service import UNCERTAINTY_OPTIONS

    stress_level_help = {
        "Quick estimate (default)": "Balanced default for a quick feasibility check.",
        "Typical spending": "Allows for normal week-to-week variation in spending.",
        "Unpredictable weeks": "Widens the range for irregular expenses or variable income.",
        "Stress test": "Tests whether the plan still works under tougher short-term conditions.",
    }

    # Seed widget-backed state before rendering widgets, then do not also pass
    # Streamlit a default value/index for the same keys. This avoids the hosted
    # warning: "widget ... was created with a default value but also had its
    # value set via the Session State API".
    try:
        current_horizon = int(st.session_state.get(STEP2_PLANNING_HORIZON_WEEKS, 12) or 12)
    except Exception:
        current_horizon = 12
    st.session_state[STEP2_PLANNING_HORIZON_WEEKS] = max(4, min(52, current_horizon))

    current_uncertainty = str(st.session_state.get(STEP2_UNCERTAINTY_PRESET, UNCERTAINTY_OPTIONS[0]) or UNCERTAINTY_OPTIONS[0])
    if current_uncertainty not in UNCERTAINTY_OPTIONS:
        current_uncertainty = UNCERTAINTY_OPTIONS[0]
    st.session_state[STEP2_UNCERTAINTY_PRESET] = current_uncertainty

    current_stress = _normalize_life_event_stress_preset(st.session_state.get("step3_stress_preset", "None"))
    st.session_state["step3_stress_preset"] = current_stress

    with st.expander("Optional stress-test settings", expanded=False):
        st.caption(
            "Use these controls to test how the plan behaves when normal spending varies or a one-off cost appears. "
            "This matters because groceries, utilities, transport and discretionary spending rarely stay exactly the same every week."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.slider(
                "Short-term horizon (weeks)",
                min_value=4,
                max_value=52,
                step=1,
                key=STEP2_PLANNING_HORIZON_WEEKS,
            )
        with c2:
            selected_uncertainty = st.selectbox(
                "Stress level",
                UNCERTAINTY_OPTIONS,
                key=STEP2_UNCERTAINTY_PRESET,
            )
            st.caption(stress_level_help.get(str(selected_uncertainty), stress_level_help["Quick estimate (default)"]))
        with c3:
            st.selectbox(
                "Life event stress test",
                _STRESS_PRESET_OPTIONS,
                key="step3_stress_preset",
                help="Applies a one-off cost around the middle of the selected horizon. It is not averaged across all weeks.",
            )
            st.caption(_life_event_stress_caption(snapshot))

        if st.button("Redraw uncertainty sample", key="step1_dashboard_redraw_uncertainty", use_container_width=True):
            st.session_state[STEP2_RANDOM_RUN_NONCE] = int(st.session_state.get(STEP2_RANDOM_RUN_NONCE, 0) or 0) + 1
            st.rerun()

        margin = float(values.get("margin", 0.0) or 0.0)
        if margin < 0:
            st.markdown("#### What should I check first?")
            st.write(
                "Check income period, fixed essentials, and whether discretionary spending is realistic. "
                "If the deficit is deliberate for a short period, continue with caution and use the stress test."
            )


def render_personal_finance_planner() -> None:
    """Render Steps 1-3 as a compact dashboard.

    This version keeps the Step 4-style feel: one dashboard page, three main
    user decisions, and optional details tucked away. It avoids rendering the
    full Step 1/2/3 legacy pages vertically.
    """
    _, success_message = apply_pending_step_patch(
        STEP1_PENDING_WIDGET_PATCH,
        STEP1_PENDING_SUCCESS_MESSAGE,
    )
    if success_message:
        show_toast_or_success(success_message, icon="✅", fallback_level="success")

    st.markdown("## Personal Finance Setup")
    st.caption("Set a quick weekly baseline, choose a savings target, and check short-term feasibility.")

    st.info(
        "**Why this matters:** your weekly free margin becomes the contribution bridge used later by the investment "
        "and long-term scenario modules. Rough numbers are enough for the demo; exact editing remains optional."
    )
    st.caption(
        "If you are unsure about a field, check the help icons where available, the Q&A section, "
        "and the Terms/educational notice in the sidebar."
    )

    # 1) Budget estimate full-width card.
    with st.container(border=True):
        values = _render_step1_quick_estimate_panel()

    # Ensure downstream cards read the current auto-saved snapshot.
    snapshot = st.session_state.get(PLANNING_SNAPSHOT, {}) or {}
    if not isinstance(snapshot, dict):
        snapshot = {}
    snapshot = _apply_dashboard_life_event_stress_to_snapshot(snapshot, persist=True)

    existing_target = _personal_finance_target_value(snapshot)

    # Useful conversion table: keep it visible, but without an extra heading.
    _render_period_equivalents_table(values)
    _render_cashflow_summary(values, existing_target)

    # 2) Main action cards: target + feasibility.
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            snapshot, _, _, target_weekly, _ = _render_savings_target_card(snapshot)
    with right:
        with st.container(border=True):
            _render_feasibility_card(snapshot, target_weekly=target_weekly)

    # 3) Keep the chart separate from the feasibility summary. This avoids a
    # cramped card and removes the dependency on Step 3's private chart helper.
    _render_feasibility_chart_card(snapshot, target_weekly=target_weekly)

    # 4) Optional tools are deliberately collapsed so the normal flow stays short.
    _render_advanced_assumptions_and_diagnostics(snapshot, values)

    st.markdown("---")
    button_label, next_step = _personal_finance_next_step_from_pathway()
    can_continue = bool(
        isinstance(snapshot, dict)
        and float(snapshot.get("baseline_savings_weekly", 0.0) or 0.0) > 0
        and float(st.session_state.get(STEP1_TARGET_WEEKLY_SAVINGS, snapshot.get("target_a_weekly", 0.0)) or 0.0) > 0
    )

    left_nav, right_nav = st.columns([1.0, 1.8])
    with left_nav:
        if st.button("Back to Home", key="personal_finance_back_to_step0", use_container_width=True):
            _auto_save_current_situation_snapshot()
            st.session_state[CURRENT_STEP] = 0
            st.session_state["current_step"] = 0
            st.rerun()
    with right_nav:
        if st.button(button_label, key="personal_finance_continue_next_module", use_container_width=True, disabled=not can_continue):
            _auto_save_current_situation_snapshot()
            st.session_state[CURRENT_STEP] = int(next_step)
            st.session_state["current_step"] = int(next_step)
            st.rerun()
        if not can_continue:
            st.caption("Set a positive free margin and weekly savings target before continuing.")


def _render_step_1_legacy(*, embedded: bool = False) -> None:
    """Render the older standalone Step 1 form.

    The final prototype normally uses the combined Personal Finance Planner,
    but this path is kept for embedded/legacy compatibility.
    """
    if not embedded and bool(st.session_state.get(PERSONAL_FINANCE_COMBINED_MODE_KEY, True)):
        render_personal_finance_planner()
        return

    _, success_message = apply_pending_step_patch(
        STEP1_PENDING_WIDGET_PATCH,
        STEP1_PENDING_SUCCESS_MESSAGE,
    )
    if success_message:
        show_toast_or_success(success_message, icon="✅", fallback_level="success")
    seed_step1_from_snapshot_if_missing()

    if embedded:
        st.caption("Rough numbers are fine — this is a starting estimate.")
    else:
        section_header("Step 1 — Your current situation", "Rough numbers are fine — this is a starting estimate.")

    income_col, income_period_col = st.columns([2.1, 1.0])
    with income_col:
        st.number_input(
            "Take-home income (£)",
            min_value=0.0,
            value=safe_float(st.session_state.get(STEP1_INCOME_AMOUNT, 460.0), 460.0),
            step=10.0,
            key=STEP1_INCOME_AMOUNT,
        )
    with income_period_col:
        st.selectbox(
            "Income period",
            _PERIOD_OPTIONS,
            index=_PERIOD_OPTIONS.index(str(st.session_state.get(STEP1_INCOME_PERIOD, "Weekly"))),
            key=STEP1_INCOME_PERIOD,
        )

    with st.expander("Not sure about take-home pay? Estimate it (optional)", expanded=False):
        e1, e2 = st.columns(2)
        with e1:
            gross_amount = st.number_input(
                "Gross income (£)",
                min_value=0.0,
                value=safe_float(st.session_state.get(STEP1_ESTIMATED_GROSS_INCOME, 30000.0), 30000.0),
                step=10.0,
                key=STEP1_ESTIMATED_GROSS_INCOME,
            )
        with e2:
            gross_period = st.selectbox(
                "Gross income period",
                _PERIOD_OPTIONS,
                index=_PERIOD_OPTIONS.index(str(st.session_state.get(STEP1_ESTIMATED_GROSS_PERIOD, "Yearly"))),
                key=STEP1_ESTIMATED_GROSS_PERIOD,
            )
        tax_rate = st.slider(
            "Estimated deduction rate",
            min_value=0.0,
            max_value=0.60,
            value=float(st.session_state.get(STEP1_ESTIMATED_TAX_RATE, 0.30) or 0.30),
            step=0.01,
            key=STEP1_ESTIMATED_TAX_RATE,
        )
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
        st.number_input(
            "Fixed essentials (£)",
            min_value=0.0,
            value=safe_float(st.session_state.get(STEP1_FIXED_AMOUNT, 185.0), 185.0),
            step=5.0,
            key=STEP1_FIXED_AMOUNT,
        )
    with fixed_period_col:
        st.selectbox(
            "Period",
            _PERIOD_OPTIONS,
            index=_PERIOD_OPTIONS.index(str(st.session_state.get(STEP1_FIXED_PERIOD, "Weekly"))),
            key=STEP1_FIXED_PERIOD,
        )

    variable_col, variable_period_col = st.columns([2.1, 1.0])
    with variable_col:
        st.number_input(
            "Variable essentials (£)",
            min_value=0.0,
            value=safe_float(st.session_state.get(STEP1_VARIABLE_AMOUNT, 80.0), 80.0),
            step=5.0,
            key=STEP1_VARIABLE_AMOUNT,
        )
    with variable_period_col:
        st.selectbox(
            "Period ",
            _PERIOD_OPTIONS,
            index=_PERIOD_OPTIONS.index(str(st.session_state.get(STEP1_VARIABLE_PERIOD, "Weekly"))),
            key=STEP1_VARIABLE_PERIOD,
        )

    discretionary_col, discretionary_period_col = st.columns([2.1, 1.0])
    with discretionary_col:
        st.number_input(
            "Discretionary spending (£)",
            min_value=0.0,
            value=safe_float(st.session_state.get(STEP1_DISCRETIONARY_AMOUNT, 35.0), 35.0),
            step=5.0,
            key=STEP1_DISCRETIONARY_AMOUNT,
        )
    with discretionary_period_col:
        st.selectbox(
            "Period  ",
            _PERIOD_OPTIONS,
            index=_PERIOD_OPTIONS.index(str(st.session_state.get(STEP1_DISCRETIONARY_PERIOD, "Weekly"))),
            key=STEP1_DISCRETIONARY_PERIOD,
        )

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
            st.number_input(
                "Utilities baseline (£)",
                min_value=0.0,
                value=safe_float(st.session_state.get(STEP1_VAR_UTILITIES_AMOUNT, 0.0), 0.0),
                step=5.0,
                key=STEP1_VAR_UTILITIES_AMOUNT,
            )
            st.selectbox(
                "Utilities period",
                _PERIOD_OPTIONS,
                index=_PERIOD_OPTIONS.index(str(st.session_state.get(STEP1_VAR_UTILITIES_PERIOD, "Weekly"))),
                key=STEP1_VAR_UTILITIES_PERIOD,
            )
            st.selectbox(
                "Season",
                _SEASON_OPTIONS,
                index=_SEASON_OPTIONS.index(str(st.session_state.get(STEP1_VAR_SEASON, "Normal"))),
                key=STEP1_VAR_SEASON,
            )
            st.slider(
                "Commute days/week",
                min_value=0,
                max_value=7,
                value=int(st.session_state.get(STEP1_VAR_COMMUTE_DAYS, 0) or 0),
                key=STEP1_VAR_COMMUTE_DAYS,
            )
            st.number_input(
                "Cost per commute day (£)",
                min_value=0.0,
                value=safe_float(st.session_state.get(STEP1_VAR_COMMUTE_COST, 0.0), 0.0),
                step=1.0,
                key=STEP1_VAR_COMMUTE_COST,
            )
        with v2:
            st.number_input(
                "Groceries (£)",
                min_value=0.0,
                value=safe_float(st.session_state.get(STEP1_VAR_GROCERIES_AMOUNT, 0.0), 0.0),
                step=5.0,
                key=STEP1_VAR_GROCERIES_AMOUNT,
            )
            st.selectbox(
                "Groceries period",
                _PERIOD_OPTIONS,
                index=_PERIOD_OPTIONS.index(str(st.session_state.get(STEP1_VAR_GROCERIES_PERIOD, "Weekly"))),
                key=STEP1_VAR_GROCERIES_PERIOD,
            )
            st.number_input(
                "Household basics (£)",
                min_value=0.0,
                value=safe_float(st.session_state.get(STEP1_VAR_HOUSEHOLD_AMOUNT, 0.0), 0.0),
                step=5.0,
                key=STEP1_VAR_HOUSEHOLD_AMOUNT,
            )
            st.selectbox(
                "Household period",
                _PERIOD_OPTIONS,
                index=_PERIOD_OPTIONS.index(str(st.session_state.get(STEP1_VAR_HOUSEHOLD_PERIOD, "Weekly"))),
                key=STEP1_VAR_HOUSEHOLD_PERIOD,
            )

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

    confirm_label = "Update snapshot" if confirmed else "Confirm my current situation"
    if embedded:
        confirm_col, note_col = st.columns([1.1, 2.0])
        with confirm_col:
            if st.button(confirm_label, key="step1_confirm_snapshot", use_container_width=True):
                confirm_step1_snapshot(snapshot)
                st.rerun()
        with note_col:
            if not confirmed:
                st.caption(step1_confirmation_required_message())
            else:
                st.caption("Snapshot confirmed. The savings target section is available below.")
        return

    left, middle, right = st.columns(3)
    with left:
        if st.button("Back to Step 0", key="step1_back_to_step0"):
            st.session_state[CURRENT_STEP] = 0
            st.rerun()
    with middle:
        if st.button(confirm_label, key="step1_confirm_snapshot"):
            confirm_step1_snapshot(snapshot)
            st.rerun()
    with right:
        if st.button("Continue to Step 2", key="step1_continue", disabled=not confirmed):
            st.session_state[CURRENT_STEP] = 2
            st.rerun()
        if not confirmed:
            st.caption(step1_confirmation_required_message())


def render_step_1(*, embedded: bool = False) -> None:
    """Public Step 1 entry point used by the app router."""
    if not embedded and bool(st.session_state.get(PERSONAL_FINANCE_COMBINED_MODE_KEY, True)):
        render_personal_finance_planner()
        return

    if embedded:
        _render_step1_quick_estimate_panel()
        return

    _render_step_1_legacy(embedded=False)