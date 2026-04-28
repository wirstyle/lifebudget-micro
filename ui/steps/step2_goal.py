"""Step 2 — Savings Goal."""
from __future__ import annotations

import streamlit as st

from ui.common.messages import section_header
from ui.services.step2_goal_service import (
    INTENT_COPY,
    UNCERTAINTY_OPTIONS,
    apply_target_seed_if_needed,
    build_feasibility_view_model,
    build_step2_snapshot_update,
    build_target_context,
    build_target_patch_payload,
    coerce_snapshot,
    consume_feedback_message,
    initialize_step2_state,
    persist_step2_snapshot,
    resolve_intent,
)
from ui.state.keys import *
from ui.state.updates import queue_and_rerun


def _queue_target_patch(
    value: float,
    preset_label: str,
    *,
    baseline_weekly: float | None = None,
    mark_user_touched: bool = True,
    feedback_message: str | None = None,
) -> None:
    queue_and_rerun(
        build_target_patch_payload(
            value,
            preset_label,
            baseline_weekly=baseline_weekly,
            mark_user_touched=mark_user_touched,
            feedback_message=feedback_message,
        )
    )


def _mark_target_user_touched() -> None:
    st.session_state[STEP2_TARGET_USER_TOUCHED_INTERNAL] = True


def _safe_money(value: float) -> str:
    try:
        return f"£{float(value):,.0f}"
    except Exception:
        return "£0"


def _render_no_margin_mode(snapshot: dict, *, planning_horizon: int, embedded: bool = False) -> None:
    baseline_margin = float(snapshot.get("baseline_savings_weekly", 0.0) or 0.0)
    discretionary_monthly = float(snapshot.get("discretionary_spending_monthly", 0.0) or 0.0)
    discretionary_weekly = discretionary_monthly * 12.0 / 52.0
    fixed_monthly = float(snapshot.get("fixed_essentials_monthly", 0.0) or 0.0)
    variable_monthly = float(snapshot.get("variable_essentials_monthly", 0.0) or 0.0)
    income_monthly = float(snapshot.get("monthly_income", 0.0) or 0.0)

    weekly_shortfall = abs(min(baseline_margin, 0.0))
    max_possible_margin = baseline_margin + max(discretionary_weekly, 0.0)
    structural_deficit = max_possible_margin < 0.0

    if embedded:
        st.markdown("### Restore weekly margin first")
    else:
        section_header("Step 2 — Restore weekly margin first")
    st.warning(
        "Your current snapshot does not leave a weekly surplus yet. "
        "Before choosing a savings target, the useful next step is to restore break-even."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Weekly surplus available", "£0/week" if baseline_margin <= 0 else f"{_safe_money(baseline_margin)}/week")
    with c2:
        st.metric("Discretionary spending", f"{_safe_money(discretionary_weekly)}/week")
    with c3:
        st.metric("Planning horizon", f"{int(planning_horizon)} weeks")

    st.markdown("### What to review first")
    if income_monthly <= 0:
        st.write("• **Income:** take-home income is zero or missing. Check the amount and period first.")
    else:
        essentials_monthly = fixed_monthly + variable_monthly
        essentials_ratio = essentials_monthly / income_monthly if income_monthly > 0 else 0.0

        if essentials_ratio >= 1.0:
            st.write(
                f"• **Essentials exceed take-home income:** essentials are about "
                f"**{_safe_money(essentials_monthly)}/month** ({essentials_ratio * 100:.0f}% of income). "
                "Check income period and the largest essential categories first."
            )
        else:
            st.write(
                f"• **Essentials:** about **{_safe_money(essentials_monthly)}/month** "
                f"({essentials_ratio * 100:.0f}% of take-home income)."
            )

        if discretionary_monthly > 0:
            st.write(
                f"• **Discretionary spending adds pressure:** currently about "
                f"**{_safe_money(discretionary_monthly)}/month**."
            )

        if structural_deficit:
            st.write(
                "• **Priority:** restore break-even by checking income, rent/housing, utilities, transport "
                "and other essential fields before setting a savings target."
            )
        else:
            st.write(
                "• **Priority:** this may be recoverable through flexible spending, but confirm the estimate "
                "before setting a savings target."
            )

    st.markdown("### Next action")
    if embedded:
        st.caption("Adjust the current situation section above until weekly surplus is restored.")
    else:
        if st.button("Back to Step 1", key="step2_no_margin_back"):
            st.session_state[CURRENT_STEP] = 1
            st.rerun()

    st.caption("Once weekly surplus is restored, Step 2 savings planning will unlock.")


def render_step_2(*, embedded: bool = False) -> None:
    snapshot = coerce_snapshot()
    intent = resolve_intent(snapshot)
    init_state = initialize_step2_state(snapshot)
    planning_horizon = int(init_state["planning_horizon"])

    target_context = build_target_context(snapshot, intent, planning_horizon)
    apply_target_seed_if_needed(
        baseline_weekly=float(target_context["baseline_weekly"]),
        recommended_target=float(target_context["recommended_target"]),
    )

    baseline_weekly = float(target_context.get("baseline_weekly", snapshot.get("baseline_savings_weekly", 0.0)) or 0.0)
    if baseline_weekly <= 0.0:
        _render_no_margin_mode(snapshot, planning_horizon=planning_horizon, embedded=embedded)
        return

    if embedded:
        st.caption("Choose a weekly savings target. The feasibility section updates below.")
    else:
        section_header("Step 2 — Choose your savings goal")
    st.info(INTENT_COPY[intent]["info"])
    st.caption(INTENT_COPY[intent]["tip"])

    feedback_message = consume_feedback_message()
    if feedback_message:
        st.success(feedback_message)

    st.markdown("### My weekly savings target")
    if STEP1_TARGET_WEEKLY_SAVINGS not in st.session_state:
        st.session_state[STEP1_TARGET_WEEKLY_SAVINGS] = float(target_context["recommended_target"])

    target_weekly = st.number_input(
        "Weekly savings target (£)",
        min_value=0.0,
        step=5.0,
        key=STEP1_TARGET_WEEKLY_SAVINGS,
        on_change=_mark_target_user_touched,
    )
    active_preset = str(st.session_state.get(STEP2_SELECTED_PRESET, "Recommended") or "Recommended")
    if active_preset == "Safe":
        st.caption(
            f"Safe starting point: about £{float(target_context['safe_target']):,.0f}/week. This is the more cautious default for your current intent."
        )
    elif active_preset == "Ambitious":
        st.caption(
            f"Ambitious starting point: about £{float(target_context['ambitious_target']):,.0f}/week. This pushes harder and is more sensitive to setbacks."
        )
    else:
        st.caption(
            f"Suggested starting point: about £{float(target_context['recommended_target']):,.0f}/week. This is the balanced default for your current intent."
        )

    p1, p2, p3 = st.columns(3)
    with p1:
        if st.button("Safe", key="step2_preset_safe"):
            _queue_target_patch(
                float(target_context["safe_target"]),
                "Safe",
                baseline_weekly=float(target_context["baseline_weekly"]),
            )
    with p2:
        if st.button("Recommended", key="step2_preset_recommended"):
            _queue_target_patch(
                float(target_context["recommended_target"]),
                "Recommended",
                baseline_weekly=float(target_context["baseline_weekly"]),
            )
    with p3:
        if st.button("Ambitious", key="step2_preset_ambitious"):
            _queue_target_patch(
                float(target_context["ambitious_target"]),
                "Ambitious",
                baseline_weekly=float(target_context["baseline_weekly"]),
            )

    st.caption("Use the presets for quick changes, or adjust the amount manually.")


    planning_horizon = int(st.session_state.get(STEP2_PLANNING_HORIZON_WEEKS, 12) or 12)
    uncertainty_preset = str(st.session_state.get(STEP2_UNCERTAINTY_PRESET, UNCERTAINTY_OPTIONS[0]) or UNCERTAINTY_OPTIONS[0])  # configured in Step 3
    random_run_nonce = int(st.session_state.get(STEP2_RANDOM_RUN_NONCE, 0) or 0)

    feasibility = build_feasibility_view_model(
        snapshot,
        intent=intent,
        planning_horizon=planning_horizon,
        target_weekly=float(target_weekly or 0.0),
    )

    st.markdown("### Quick feasibility check")
    if float(feasibility["required_weekly"]) <= float(feasibility["baseline_margin_weekly"]):
        st.success(
            f"Target: £{float(feasibility['required_weekly']):,.0f}/week is already achievable without changes."
        )
    elif float(feasibility["need_weekly"]) <= float(feasibility["discretionary_weekly"]):
        st.info(
            f"Target: £{float(feasibility['required_weekly']):,.0f}/week needs about £{float(feasibility['required_cut_monthly']):,.0f}/mo more than your current baseline."
        )
    else:
        st.error(
            f"Target: £{float(feasibility['required_weekly']):,.0f}/week is not achievable through discretionary cuts alone. Even using your full discretionary budget only gives about £{float(feasibility['required_cut_monthly']):,.0f}/mo."
        )

    updated_snapshot = build_step2_snapshot_update(
        snapshot,
        intent=intent,
        planning_horizon=planning_horizon,
        uncertainty_preset=uncertainty_preset,
        random_run_nonce=random_run_nonce,
        feasibility=feasibility,
    )
    persist_step2_snapshot(updated_snapshot)

    if embedded:
        st.caption("Savings target saved. Review the short-term feasibility section below.")
        return

    left, right = st.columns(2)
    with left:
        if st.button("Back to Step 1", key="step2_back"):
            st.session_state[CURRENT_STEP] = 1
            st.rerun()
    with right:
        if st.button("Continue to Step 3", key="step2_continue"):
            st.session_state[CURRENT_STEP] = 3
            st.rerun()
