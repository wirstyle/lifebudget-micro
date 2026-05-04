"""
Savings-target service for the Personal Finance Planner.

This module contains the Step 2 goal logic used by both the legacy Step 2 page
and the combined Personal Finance Planner dashboard. It derives weekly target
presets from the confirmed Step 1 baseline, stores target-related assumptions in
the planning snapshot, and prepares the feasibility payload consumed by Step 3.

The renderer should keep UI decisions separate from this service. This module
owns target calculation, buffer-goal conversion, one-off event cleaning, and
snapshot updates.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.expenses import clean_events
from ui.state.keys import (
    PLAN_GENERATED,
    PLANNING_SNAPSHOT,
    STEP1_TARGET_WEEKLY_SAVINGS,
    STEP2_BASELINE_WEEKLY_SEEDED,
    STEP2_BUFFER_GOAL_AMOUNT_INPUT,
    STEP2_BUFFER_GOAL_PRIORITY_INPUT,
    STEP2_BUFFER_GOAL_WEEKS_INPUT,
    STEP2_ENABLE_ONE_OFF_EVENTS,
    STEP2_ONE_OFF_EVENTS_DF,
    STEP2_PLANNING_HORIZON_WEEKS,
    STEP2_RANDOM_RUN_NONCE,
    STEP2_SELECTED_PRESET,
    STEP2_SUGGESTION_FEEDBACK,
    STEP2_TARGET_INITIALIZED,
    STEP2_TARGET_USER_TOUCHED_INTERNAL,
    STEP2_UNCERTAINTY_PRESET,
    USER_INTENT,
)

PRIORITY_OPTIONS = ["Balanced", "Fastest possible", "Lowest pressure"]
UNCERTAINTY_OPTIONS = ["Quick estimate (default)", "Typical spending", "Unpredictable weeks", "Stress test"]

INTENT_COPY = {
    "not_sure_yet": {
        "info": "We’ll start from your current weekly margin and give you a balanced short-term target.",
        "tip": "Tip: the suggested target is a moderate default so you can sanity-check what feels sustainable.",
    },
    "avoid_overspending": {
        "info": "The priority here is cash-flow stability first. We’ll keep the default target cautious and focus on making the plan resilient.",
        "tip": "Tip: if your cash flow still feels tight, use the Safe preset and test one-off events before aiming higher.",
    },
    "save_more_each_week": {
        "info": "You want to push savings up from today’s baseline. We’ll bias the default toward a more ambitious weekly target.",
        "tip": "Tip: start from Recommended or Ambitious, then check in Step 3 whether the plan still looks comfortable under uncertainty.",
    },
    "reach_target_balance": {
        "info": "This mode reframes Step 2 around a short-term cash-buffer goal: how much cash you want to build, by when, before any investing logic from later steps.",
        "tip": "Tip: set a cash-buffer goal below and we’ll convert it into an implied weekly pace you can compare with your manual target.",
    },
}


def safe_float(value, default: float = 0.0) -> float:
    """Convert a value to float without letting bad UI/session values crash."""
    try:
        return float(value)
    except Exception:
        return float(default)


def coerce_snapshot() -> dict:
    """Return the current planning snapshot as a plain dictionary."""
    snapshot = st.session_state.get(PLANNING_SNAPSHOT, {}) or {}
    return dict(snapshot) if isinstance(snapshot, dict) else {}


def resolve_intent(snapshot: dict) -> str:
    """Resolve the current planning intent from snapshot or session state."""
    value = str(snapshot.get("user_intent", st.session_state.get(USER_INTENT, "not_sure_yet")) or "not_sure_yet")
    return value if value in INTENT_COPY else "not_sure_yet"


def weekly_to_monthly(value: float) -> float:
    """Convert a weekly amount into an approximate monthly equivalent."""
    return float(value) * 52.0 / 12.0


def monthly_to_weekly(value: float) -> float:
    """Convert a monthly amount into an approximate weekly equivalent."""
    return float(value) * 12.0 / 52.0


def round_target(value: float) -> float:
    """Round a weekly target to a small, user-friendly increment."""
    return float(round(max(float(value), 0.0) / 4.0) * 4.0)


def cap_target_to_margin(value: float, baseline_weekly: float) -> float:
    """Keep selectable weekly targets within the current weekly free margin.

    Buffer-goal requirements may exceed the current free margin, but selectable
    target presets should not ask the user to save more than the confirmed
    weekly margin. The uncapped buffer requirement is still kept separately.
    """
    return float(min(round_target(value), max(float(baseline_weekly), 0.0)))


def default_one_off_events_df() -> pd.DataFrame:
    """Return the editable default table for optional one-off events."""
    return pd.DataFrame(
        [
            {"Event (optional)": "", "Amount (£)": 0.0, "Week": 1},
            {"Event (optional)": "", "Amount (£)": 0.0, "Week": 1},
            {"Event (optional)": "", "Amount (£)": 0.0, "Week": 1},
        ]
    )


def normalize_one_off_events_df(df) -> pd.DataFrame:
    """Normalise the one-off events editor dataframe to the expected schema."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return default_one_off_events_df()

    work = df.copy().reset_index(drop=True)
    expected_cols = ["Event (optional)", "Amount (£)", "Week"]

    for col in expected_cols:
        if col not in work.columns:
            work[col] = "" if col == "Event (optional)" else (0.0 if col == "Amount (£)" else 1)

    work = work[expected_cols].copy()
    work["Event (optional)"] = work["Event (optional)"].astype(str)
    work["Amount (£)"] = pd.to_numeric(work["Amount (£)"], errors="coerce").fillna(0.0)
    work["Week"] = pd.to_numeric(work["Week"], errors="coerce").fillna(1).astype(int).clip(lower=1)

    return work


def seed_buffer_goal_inputs(snapshot: dict, planning_horizon: int) -> None:
    """Seed cash-buffer goal widgets from snapshot values when missing."""
    if STEP2_BUFFER_GOAL_AMOUNT_INPUT not in st.session_state:
        st.session_state[STEP2_BUFFER_GOAL_AMOUNT_INPUT] = float(
            snapshot.get("short_term_goal_amount", 1000.0) or 1000.0
        )

    if STEP2_BUFFER_GOAL_WEEKS_INPUT not in st.session_state:
        default_weeks = int(snapshot.get("short_term_goal_weeks", planning_horizon) or planning_horizon)
        st.session_state[STEP2_BUFFER_GOAL_WEEKS_INPUT] = max(default_weeks, 1)

    if STEP2_BUFFER_GOAL_PRIORITY_INPUT not in st.session_state:
        priority = str(snapshot.get("short_term_goal_priority", "Balanced") or "Balanced")
        st.session_state[STEP2_BUFFER_GOAL_PRIORITY_INPUT] = priority if priority in PRIORITY_OPTIONS else "Balanced"


def buffer_required_weekly(amount: float, weeks: int, priority: str) -> float:
    """Convert a cash-buffer goal into an implied weekly pace."""
    weeks = max(int(weeks), 1)
    required = float(amount) / float(weeks)

    if priority == "Fastest possible":
        required *= 1.10
    elif priority == "Lowest pressure":
        required *= 0.90

    return round_target(required)


def intent_target_triplet(
    intent: str,
    baseline_weekly: float,
    buffer_required_weekly_value: float,
) -> tuple[float, float, float]:
    """Return Safe/Recommended/Ambitious weekly targets for an intent.

    The returned preset values are capped at the current weekly free margin.
    For cash-buffer mode, the uncapped required weekly pace remains available
    separately through ``buffer_required_weekly_value``.
    """
    baseline_weekly = max(float(baseline_weekly), 0.0)

    multipliers = {
        "not_sure_yet": (0.20, 0.30, 0.40),
        "avoid_overspending": (0.05, 0.15, 0.25),
        "save_more_each_week": (0.25, 0.40, 0.55),
        "reach_target_balance": (0.20, 0.30, 0.40),
    }

    safe_mult, rec_mult, amb_mult = multipliers.get(intent, multipliers["not_sure_yet"])

    safe_target = round_target(baseline_weekly * safe_mult)
    recommended_target = round_target(baseline_weekly * rec_mult)
    ambitious_target = round_target(baseline_weekly * amb_mult)

    if intent == "reach_target_balance" and buffer_required_weekly_value > 0.0:
        safe_target = round_target(max(safe_target, buffer_required_weekly_value * 0.85))
        recommended_target = round_target(max(recommended_target, buffer_required_weekly_value))
        ambitious_target = round_target(max(ambitious_target, buffer_required_weekly_value * 1.15))

    return (
        cap_target_to_margin(safe_target, baseline_weekly),
        cap_target_to_margin(recommended_target, baseline_weekly),
        cap_target_to_margin(ambitious_target, baseline_weekly),
    )


def resolve_suggested_target(
    intent: str,
    *,
    safe_target: float,
    recommended_target: float,
    ambitious_target: float,
    buffer_required_weekly_value: float,
) -> tuple[float, str, str]:
    """Resolve the default suggested target and label for the current intent."""
    if intent == "avoid_overspending":
        return safe_target, "Safe", f"Suggested target applied: £{safe_target:,.0f}/week (cautious)"

    if intent == "save_more_each_week":
        return ambitious_target, "Ambitious", f"Suggested target applied: £{ambitious_target:,.0f}/week (stretch)"

    if intent == "reach_target_balance":
        suggested = max(recommended_target, buffer_required_weekly_value)
        return suggested, "Recommended", f"Suggested target applied: £{suggested:,.0f}/week (cash-buffer aligned)"

    return recommended_target, "Recommended", f"Suggested target applied: £{recommended_target:,.0f}/week (balanced)"


def initialize_step2_state(snapshot: dict) -> dict:
    """Initialise Step 2 widget/session defaults without overwriting user edits."""
    if STEP2_PLANNING_HORIZON_WEEKS not in st.session_state:
        st.session_state[STEP2_PLANNING_HORIZON_WEEKS] = int(snapshot.get("planning_horizon_weeks", 12) or 12)

    if STEP2_UNCERTAINTY_PRESET not in st.session_state:
        snapshot_uncertainty = str(snapshot.get("uncertainty_preset", UNCERTAINTY_OPTIONS[0]) or UNCERTAINTY_OPTIONS[0])
        st.session_state[STEP2_UNCERTAINTY_PRESET] = (
            snapshot_uncertainty if snapshot_uncertainty in UNCERTAINTY_OPTIONS else UNCERTAINTY_OPTIONS[0]
        )

    if STEP2_RANDOM_RUN_NONCE not in st.session_state:
        st.session_state[STEP2_RANDOM_RUN_NONCE] = int(snapshot.get("random_run_nonce", 0) or 0)

    if STEP2_ENABLE_ONE_OFF_EVENTS not in st.session_state:
        st.session_state[STEP2_ENABLE_ONE_OFF_EVENTS] = bool(snapshot.get("enable_one_off_events", False))

    if STEP2_ONE_OFF_EVENTS_DF not in st.session_state:
        snapshot_events = snapshot.get("one_off_events", []) or []
        if snapshot_events:
            st.session_state[STEP2_ONE_OFF_EVENTS_DF] = normalize_snapshot_events(snapshot_events)
        else:
            st.session_state[STEP2_ONE_OFF_EVENTS_DF] = default_one_off_events_df()

    planning_horizon = int(
        st.session_state.get(STEP2_PLANNING_HORIZON_WEEKS, snapshot.get("planning_horizon_weeks", 12)) or 12
    )
    seed_buffer_goal_inputs(snapshot, planning_horizon)

    if STEP2_SELECTED_PRESET not in st.session_state:
        st.session_state[STEP2_SELECTED_PRESET] = "Recommended"

    if STEP2_TARGET_INITIALIZED not in st.session_state:
        st.session_state[STEP2_TARGET_INITIALIZED] = False

    if STEP2_TARGET_USER_TOUCHED_INTERNAL not in st.session_state:
        st.session_state[STEP2_TARGET_USER_TOUCHED_INTERNAL] = False

    return {
        "planning_horizon": planning_horizon,
        "uncertainty_options": list(UNCERTAINTY_OPTIONS),
    }


def normalize_snapshot_events(snapshot_events: list[dict]) -> pd.DataFrame:
    """Convert snapshot one-off event dictionaries into the editor dataframe."""
    rows = []

    for row in snapshot_events:
        if not isinstance(row, dict):
            continue

        rows.append(
            {
                "Event (optional)": str(row.get("name", "") or ""),
                "Amount (£)": float(safe_float(row.get("amount", 0.0), 0.0)),
                "Week": int(safe_float(row.get("week", 1), 1.0) or 1.0),
            }
        )

    return normalize_one_off_events_df(pd.DataFrame(rows))


def get_buffer_goal_state(planning_horizon: int) -> dict:
    """Read current cash-buffer goal inputs and derive the required weekly pace."""
    buffer_goal_amount = float(st.session_state.get(STEP2_BUFFER_GOAL_AMOUNT_INPUT, 1000.0) or 1000.0)
    buffer_goal_weeks = int(st.session_state.get(STEP2_BUFFER_GOAL_WEEKS_INPUT, planning_horizon) or planning_horizon)
    buffer_goal_priority = str(st.session_state.get(STEP2_BUFFER_GOAL_PRIORITY_INPUT, "Balanced") or "Balanced")

    if buffer_goal_priority not in PRIORITY_OPTIONS:
        buffer_goal_priority = "Balanced"

    buffer_required_weekly_value = buffer_required_weekly(buffer_goal_amount, buffer_goal_weeks, buffer_goal_priority)

    return {
        "buffer_goal_amount": buffer_goal_amount,
        "buffer_goal_weeks": buffer_goal_weeks,
        "buffer_goal_priority": buffer_goal_priority,
        "buffer_required_weekly_value": buffer_required_weekly_value,
    }


def build_target_context(snapshot: dict, intent: str, planning_horizon: int) -> dict:
    """Build target presets and cash-buffer context for the renderer."""
    baseline_weekly = safe_float(snapshot.get("baseline_savings_weekly", 0.0), 0.0)
    baseline_weekly = max(float(baseline_weekly), 0.0)

    buffer_state = get_buffer_goal_state(planning_horizon)

    safe_target, recommended_target, ambitious_target = intent_target_triplet(
        intent,
        baseline_weekly,
        float(buffer_state["buffer_required_weekly_value"]),
    )

    raw_suggested_target, suggested_label, suggested_feedback = resolve_suggested_target(
        intent,
        safe_target=safe_target,
        recommended_target=recommended_target,
        ambitious_target=ambitious_target,
        buffer_required_weekly_value=float(buffer_state["buffer_required_weekly_value"]),
    )

    suggested_target = cap_target_to_margin(raw_suggested_target, baseline_weekly)
    if raw_suggested_target > suggested_target:
        suggested_feedback = (
            f"Suggested target applied: £{suggested_target:,.0f}/week "
            "(capped to current weekly free margin)"
        )

    return {
        "baseline_weekly": baseline_weekly,
        "safe_target": safe_target,
        "recommended_target": recommended_target,
        "ambitious_target": ambitious_target,
        "suggested_target": suggested_target,
        "suggested_label": suggested_label,
        "suggested_feedback": suggested_feedback,
        **buffer_state,
    }


def apply_target_seed_if_needed(*, baseline_weekly: float, recommended_target: float) -> None:
    """Seed the weekly target once, unless the user has already edited it."""
    baseline_weekly = max(float(baseline_weekly), 0.0)
    seeded_target = cap_target_to_margin(recommended_target, baseline_weekly)

    current_target = safe_float(st.session_state.get(STEP1_TARGET_WEEKLY_SAVINGS, 0.0), 0.0)
    user_touched_target = bool(st.session_state.get(STEP2_TARGET_USER_TOUCHED_INTERNAL, False))
    last_seeded_baseline_weekly = safe_float(st.session_state.get(STEP2_BASELINE_WEEKLY_SEEDED, -1.0), -1.0)
    baseline_changed_since_seed = abs(float(baseline_weekly) - float(last_seeded_baseline_weekly)) > 1e-9

    if baseline_weekly > 0.0 and (not user_touched_target) and (
        current_target <= 0.0
        or not bool(st.session_state.get(STEP2_TARGET_INITIALIZED, False))
        or baseline_changed_since_seed
    ):
        st.session_state[STEP1_TARGET_WEEKLY_SAVINGS] = float(seeded_target)
        st.session_state[STEP2_SELECTED_PRESET] = "Recommended"
        st.session_state[STEP2_TARGET_INITIALIZED] = True
        st.session_state[STEP2_BASELINE_WEEKLY_SEEDED] = float(baseline_weekly)


def build_target_patch_payload(
    value: float,
    preset_label: str,
    *,
    baseline_weekly: float | None = None,
    mark_user_touched: bool = True,
    feedback_message: str | None = None,
) -> dict:
    """Build a session-state patch for applying a weekly target preset."""
    if baseline_weekly is not None:
        target_value = cap_target_to_margin(value, baseline_weekly)
    else:
        target_value = round_target(value)

    payload = {
        STEP1_TARGET_WEEKLY_SAVINGS: target_value,
        STEP2_SELECTED_PRESET: str(preset_label),
        STEP2_TARGET_INITIALIZED: True,
        STEP2_TARGET_USER_TOUCHED_INTERNAL: bool(mark_user_touched),
    }

    if baseline_weekly is not None:
        payload[STEP2_BASELINE_WEEKLY_SEEDED] = float(baseline_weekly)

    if feedback_message:
        payload[STEP2_SUGGESTION_FEEDBACK] = str(feedback_message)

    return payload


def consume_feedback_message() -> str | None:
    """Read and clear the pending target suggestion feedback message."""
    feedback_message = st.session_state.pop(STEP2_SUGGESTION_FEEDBACK, None)
    return str(feedback_message) if feedback_message else None


def clean_one_off_events_from_state(planning_horizon: int) -> list[dict]:
    """Clean editable one-off events and store whether any are enabled."""
    raw_event_rows = normalize_one_off_events_df(st.session_state.get(STEP2_ONE_OFF_EVENTS_DF)).to_dict("records")
    normalized_event_rows = []

    for row in raw_event_rows:
        if not isinstance(row, dict):
            continue

        normalized_event_rows.append(
            {
                "name": str(row.get("Event (optional)", "") or ""),
                "amount": float(safe_float(row.get("Amount (£)", 0.0), 0.0)),
                "week": int(safe_float(row.get("Week", 1), 1.0) or 1.0),
            }
        )

    cleaned_one_off_events = clean_events(normalized_event_rows, weeks=int(planning_horizon))
    st.session_state[STEP2_ENABLE_ONE_OFF_EVENTS] = bool(cleaned_one_off_events)

    return cleaned_one_off_events


def build_feasibility_view_model(
    snapshot: dict,
    *,
    intent: str,
    planning_horizon: int,
    target_weekly: float,
) -> dict:
    """Build the Step 2 feasibility payload consumed by Step 3.

    For ``reach_target_balance``, the required weekly pace may be lifted to the
    cash-buffer requirement. This preserves the goal-comparison behaviour while
    the selectable UI target remains capped to the current weekly free margin.
    """
    cleaned_one_off_events = clean_one_off_events_from_state(planning_horizon)
    buffer_state = get_buffer_goal_state(planning_horizon)
    buffer_required_weekly_value = float(buffer_state["buffer_required_weekly_value"])

    required_weekly = float(target_weekly or 0.0)
    if intent == "reach_target_balance" and buffer_required_weekly_value > 0.0:
        required_weekly = max(required_weekly, buffer_required_weekly_value)

    required_monthly = weekly_to_monthly(required_weekly)
    baseline_margin_weekly = max(safe_float(snapshot.get("baseline_savings_weekly", 0.0), 0.0), 0.0)
    discretionary_weekly = max(
        monthly_to_weekly(safe_float(snapshot.get("discretionary_spending_monthly", 0.0), 0.0)),
        0.0,
    )
    need_weekly = max(float(required_weekly) - baseline_margin_weekly, 0.0)
    cut_weekly = min(need_weekly, discretionary_weekly)
    required_cut_monthly = weekly_to_monthly(cut_weekly)
    expected_cash_only_balance = float(required_weekly) * float(planning_horizon)

    return {
        "cleaned_one_off_events": cleaned_one_off_events,
        "required_weekly": required_weekly,
        "required_monthly": required_monthly,
        "baseline_margin_weekly": baseline_margin_weekly,
        "discretionary_weekly": discretionary_weekly,
        "need_weekly": need_weekly,
        "required_cut_monthly": required_cut_monthly,
        "expected_cash_only_balance": expected_cash_only_balance,
        **buffer_state,
    }


def build_step2_snapshot_update(
    snapshot: dict,
    *,
    intent: str,
    planning_horizon: int,
    uncertainty_preset: str,
    random_run_nonce: int,
    feasibility: dict,
) -> dict:
    """Return the planning snapshot updated with Step 2 target assumptions."""
    updated_snapshot = dict(snapshot)
    updated_snapshot.update(
        {
            "user_intent": str(intent),
            "planning_horizon_weeks": int(planning_horizon),
            "uncertainty_preset": str(uncertainty_preset),
            "random_run_nonce": int(random_run_nonce),
            "enable_one_off_events": bool(st.session_state.get(STEP2_ENABLE_ONE_OFF_EVENTS, False)),
            "one_off_events": feasibility["cleaned_one_off_events"],
            "short_term_goal_amount": float(feasibility["buffer_goal_amount"]),
            "short_term_goal_weeks": int(feasibility["buffer_goal_weeks"]),
            "short_term_goal_priority": str(feasibility["buffer_goal_priority"]),
            "short_term_goal_required_weekly": float(feasibility["buffer_required_weekly_value"]),
            "weekly_savings": float(feasibility["required_weekly"]),
            "target_a_monthly": float(feasibility["required_monthly"]),
            "target_a_weekly": float(feasibility["required_weekly"]),
            "required_cut_a_monthly": float(feasibility["required_cut_monthly"]),
            "structural_deficit": bool(snapshot.get("structural_deficit", False)),
        }
    )
    return updated_snapshot


def persist_step2_snapshot(updated_snapshot: dict) -> None:
    """Persist the Step 2 snapshot and mark the legacy plan state as available."""
    st.session_state[PLANNING_SNAPSHOT] = updated_snapshot
    st.session_state[PLAN_GENERATED] = True