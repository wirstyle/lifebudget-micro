"""
Legacy Step 3 short-term feasibility renderer.

The final prototype usually renders the feasibility summary and chart inside the
combined Personal Finance Planner dashboard in ``step1_inputs.py``. This module
is kept for legacy/embedded compatibility and displays the Step 3 view model
built by ``step3_results_service``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from ui.common.messages import section_header
from ui.common.metrics import currency_metric
from ui.services.step2_goal_service import UNCERTAINTY_OPTIONS
from ui.services.step3_results_service import build_step3_view_model, coerce_snapshot
from ui.state.keys import *

STEP0_PATHWAY = "step0_planning_pathway"


def _current_pathway() -> str:
    """Resolve the current planning pathway with legacy fallback keys.

    Step 0 has used a few different session-state keys during the prototype.
    This helper normalises them into the canonical pathway value used for
    navigation after the short-term feasibility step.
    """
    valid = {"compare_both", "savings_only", "savings_plus_investing"}

    value = str(st.session_state.get(STEP0_PATHWAY, "") or "").strip()
    if value in valid:
        return value

    alias = str(st.session_state.get("selected_planning_pathway", "") or "").strip()
    if alias in valid:
        st.session_state[STEP0_PATHWAY] = alias
        return alias

    radio_label = str(st.session_state.get("step0_pathway_radio", "") or "").lower()
    if "savings-only" in radio_label or "savings only" in radio_label:
        st.session_state[STEP0_PATHWAY] = "savings_only"
        return "savings_only"
    if "compare" in radio_label:
        st.session_state[STEP0_PATHWAY] = "compare_both"
        return "compare_both"
    if "investing" in radio_label:
        st.session_state[STEP0_PATHWAY] = "savings_plus_investing"
        return "savings_plus_investing"

    # Last compatibility fallback.
    intent = str(st.session_state.get(USER_INTENT, "") or "").strip()
    if intent == "avoid_overspending":
        st.session_state[STEP0_PATHWAY] = "savings_only"
        return "savings_only"
    if intent == "save_more_each_week":
        st.session_state[STEP0_PATHWAY] = "savings_plus_investing"
        return "savings_plus_investing"

    st.session_state[STEP0_PATHWAY] = "compare_both"
    return "compare_both"


def _pathway_label(pathway: str) -> str:
    """Return a user-facing label for the selected planning pathway."""
    return {
        "compare_both": "Compare savings vs investing",
        "savings_only": "Savings-only plan",
        "savings_plus_investing": "Savings + investing plan",
    }.get(pathway, "Compare savings vs investing")


def _render_projection_chart(
    baseline_df,
    plan_a_df,
    *,
    target_weekly: float = 0.0,
    shock_week: int | None = None,
    shock_amount: float = 0.0,
) -> None:
    """Render the Step 3 baseline vs target-plan chart.

    The service provides the baseline path and target-plan simulation output.
    The chart maps Mean/Lower/Upper to expected and 10–90% scenario range.
    """
    if baseline_df.empty or plan_a_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.plot(baseline_df["Week"], baseline_df["Balance"], label="Baseline", linewidth=2.5)
    ax.plot(plan_a_df["Week"], plan_a_df["Mean"], label="Target plan (expected)", linewidth=2.5)
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
        ax.scatter([final_week], [final_target], s=70, marker="o", label="Savings target", zorder=5)
        ax.annotate(
            "savings target",
            xy=(final_week, final_target),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )

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
    ax.set_title("Short-term balance path")
    ax.set_xlabel("Week")
    ax.set_ylabel("Balance (£)")
    ax.legend()
    st.pyplot(fig, clear_figure=True)


def _branch_card(pathway: str) -> tuple[str, str, str, int]:
    """Return the next-step navigation copy for the selected pathway."""
    if pathway == "savings_only":
        return (
            "Next pathway",
            "Continue to Step 6 to extend this weekly saving into a long-term savings-only pathway.",
            "Continue to Step 6",
            6,
        )

    if pathway == "savings_plus_investing":
        return (
            "Next pathway",
            "Continue to Step 4 to build the investment branch as an educational scenario layer.",
            "Continue to Step 4",
            4,
        )

    return (
        "Next pathway",
        "Build the investment branch so Step 6 can compare savings-only outcomes against savings + investing outcomes.",
        "Build comparison branch",
        4,
    )


def _go_to_next_branch(pathway: str, next_step: int) -> None:
    """Navigate to the next module while preserving pathway compatibility keys."""
    # Defensive navigation: write both canonical imported key and raw string key.
    st.session_state[STEP0_PATHWAY] = pathway
    st.session_state["selected_planning_pathway"] = pathway

    if pathway == "savings_only":
        st.session_state[PROJECTION_OPEN] = True
        st.session_state[CURRENT_STEP] = 6
        st.session_state["current_step"] = 6
    else:
        st.session_state[CURRENT_STEP] = int(next_step)
        st.session_state["current_step"] = int(next_step)

    st.rerun()


_UNCERTAINTY_LABELS = {
    "Quick estimate (default)": "Balanced uncertainty",
    "Typical spending": "Mild variability",
    "Unpredictable weeks": "Elevated variability",
    "Stress test": "Severe variability",
}

_UNCERTAINTY_REVERSE_LABELS = {label: raw for raw, label in _UNCERTAINTY_LABELS.items()}

_STRESS_PRESET_OPTIONS = ["None", "Minor unexpected expense", "Major monthly shock", "Severe emergency shock"]

_STRESS_PRESET_EVENTS = {
    "None": {
        "amount": 0.0,
        "label": "No one-off cost is applied.",
    },
    "Minor unexpected expense": {
        "amount": 250.0,
        "label": "Adds a modest one-off cost in the middle of the short-term horizon.",
    },
    "Major monthly shock": {
        "amount": 600.0,
        "label": "Adds a more demanding one-off cost in the middle of the short-term horizon.",
    },
    "Severe emergency shock": {
        "amount": 1000.0,
        "label": "Adds a large one-off cost to stress-test the plan.",
    },
}


def _sync_step3_assumption_controls(snapshot: dict) -> dict:
    """Render Step 3 assumption controls and persist them to the snapshot.

    This legacy renderer owns the visible Step 3 controls when the separate Step
    3 page is used. The combined Personal Finance Planner has a compact version
    of the same controls, so this function also writes compatibility payloads
    expected by the shared short-term simulation service.
    """
    if not isinstance(snapshot, dict):
        snapshot = {}

    current_uncertainty = str(
        st.session_state.get(
            STEP2_UNCERTAINTY_PRESET,
            snapshot.get("uncertainty_preset", UNCERTAINTY_OPTIONS[0]),
        )
        or UNCERTAINTY_OPTIONS[0]
    )
    if current_uncertainty not in UNCERTAINTY_OPTIONS:
        current_uncertainty = UNCERTAINTY_OPTIONS[0]

    current_horizon = int(
        st.session_state.get(
            STEP2_PLANNING_HORIZON_WEEKS,
            snapshot.get("planning_horizon_weeks", 12),
        )
        or 12
    )
    current_horizon = max(4, min(52, current_horizon))

    current_stress = str(
        st.session_state.get(
            "step3_stress_preset",
            snapshot.get("stress_preset", "None"),
        )
        or "None"
    )
    legacy_stress_labels = {
        "Typical bump": "Minor unexpected expense",
        "Tough month": "Major monthly shock",
        "Heavy shock": "Severe emergency shock",
    }
    current_stress = legacy_stress_labels.get(current_stress, current_stress)
    if current_stress not in _STRESS_PRESET_OPTIONS:
        current_stress = "None"

    st.session_state[STEP2_UNCERTAINTY_PRESET] = current_uncertainty
    st.session_state[STEP2_PLANNING_HORIZON_WEEKS] = current_horizon
    st.session_state["step3_stress_preset"] = current_stress

    # Seed widget-backed state before rendering widgets. Avoid also passing
    # value/index defaults for these same keys, otherwise hosted Streamlit may
    # display a warning about Session State + widget defaults.
    current_uncertainty_display_seed = _UNCERTAINTY_LABELS.get(current_uncertainty, current_uncertainty)
    display_options_seed = [_UNCERTAINTY_LABELS.get(option, option) for option in UNCERTAINTY_OPTIONS]
    if st.session_state.get("step3_uncertainty_display") not in display_options_seed:
        st.session_state["step3_uncertainty_display"] = current_uncertainty_display_seed

    with st.expander("Scenario assumptions (optional)", expanded=False):
        st.caption(
            "These controls affect the short-term feasibility test shown in this step. "
            "Step 2 chooses the savings target; Step 3 stress-tests it."
        )

        selected_horizon = int(
            st.slider(
                "Short-term horizon (weeks)",
                min_value=4,
                max_value=52,
                step=1,
                key=STEP2_PLANNING_HORIZON_WEEKS,
                help="How many weeks ahead this short-term feasibility view should test.",
            )
        )

        uncertainty_display_options = [_UNCERTAINTY_LABELS.get(option, option) for option in UNCERTAINTY_OPTIONS]
        selected_uncertainty_display = st.selectbox(
            "Scenario stress level",
            uncertainty_display_options,
            key="step3_uncertainty_display",
            help="Controls how cautious the short-term planning assumptions should feel.",
        )
        selected_uncertainty = _UNCERTAINTY_REVERSE_LABELS.get(selected_uncertainty_display, selected_uncertainty_display)
        st.session_state[STEP2_UNCERTAINTY_PRESET] = selected_uncertainty

        selected_stress = st.selectbox(
            "Life event stress test",
            _STRESS_PRESET_OPTIONS,
            key="step3_stress_preset",
            help="Applies a simple one-off cost scenario without asking you to fill in a spreadsheet.",
        )

        event_meta = _STRESS_PRESET_EVENTS.get(str(selected_stress), _STRESS_PRESET_EVENTS["None"])
        st.caption(str(event_meta.get("label", "")))

        st.caption("Randomness is kept stable unless you redraw.")
        if st.button("Redraw uncertainty sample", key="step3_random_run"):
            st.session_state[STEP2_RANDOM_RUN_NONCE] = int(st.session_state.get(STEP2_RANDOM_RUN_NONCE, 0) or 0) + 1
            stored_snapshot = st.session_state.get(PLANNING_SNAPSHOT, {}) or {}
            if isinstance(stored_snapshot, dict):
                stored_snapshot = dict(stored_snapshot)
                stored_snapshot["uncertainty_preset"] = str(selected_uncertainty)
                stored_snapshot["planning_horizon_weeks"] = int(selected_horizon)
                stored_snapshot["stress_preset"] = str(selected_stress)
                stored_snapshot["random_run_nonce"] = int(st.session_state.get(STEP2_RANDOM_RUN_NONCE, 0) or 0)
                st.session_state[PLANNING_SNAPSHOT] = stored_snapshot
            st.rerun()

    selected_horizon = int(st.session_state.get(STEP2_PLANNING_HORIZON_WEEKS, current_horizon) or current_horizon)
    selected_uncertainty = str(st.session_state.get(STEP2_UNCERTAINTY_PRESET, current_uncertainty) or current_uncertainty)
    selected_stress = str(st.session_state.get("step3_stress_preset", current_stress) or current_stress)
    if selected_stress not in _STRESS_PRESET_OPTIONS:
        selected_stress = "None"

    event_meta = _STRESS_PRESET_EVENTS.get(selected_stress, _STRESS_PRESET_EVENTS["None"])
    shock_amount = float(event_meta.get("amount", 0.0) or 0.0)
    shock_week = max(1, int(round(float(selected_horizon) / 2.0)))

    updated = dict(snapshot)
    updated["planning_horizon_weeks"] = int(selected_horizon)
    updated["uncertainty_preset"] = selected_uncertainty
    updated["random_run_nonce"] = int(st.session_state.get(STEP2_RANDOM_RUN_NONCE, snapshot.get("random_run_nonce", 0)) or 0)
    updated["stress_preset"] = selected_stress

    # Compatibility payloads for the existing short-term simulation layer.
    updated["enable_one_off_events"] = bool(shock_amount > 0.0)
    updated["shock_enabled"] = bool(shock_amount > 0.0)
    updated["shock_amount"] = float(shock_amount)
    updated["shock_week"] = int(shock_week)
    updated["one_off_events"] = (
        [{"name": selected_stress, "amount": float(shock_amount), "week": int(shock_week)}]
        if shock_amount > 0.0
        else []
    )
    updated["shock_map"] = ({int(shock_week): -float(shock_amount)} if shock_amount > 0.0 else {})

    st.session_state[PLANNING_SNAPSHOT] = updated
    return updated


def _render_scenario_summary(
    *,
    snapshot: dict,
    planning_horizon: int,
    target_weekly: float,
) -> None:
    """Render a compact summary of the active short-term scenario."""
    stress_preset = str(snapshot.get("stress_preset", "None") or "None")
    raw_uncertainty = str(snapshot.get("uncertainty_preset", UNCERTAINTY_OPTIONS[0]) or UNCERTAINTY_OPTIONS[0])
    uncertainty_label = _UNCERTAINTY_LABELS.get(raw_uncertainty, raw_uncertainty)

    st.info(
        "**Scenario summary**  \n"
        f"• Horizon: **{planning_horizon} weeks**  \n"
        f"• Target: **£{target_weekly:,.0f}/week**  \n"
        f"• Stress: **{uncertainty_label}** · **{stress_preset}**"
    )


def _classify_short_term_result(summary: dict, *, materiality_pct: float = 0.02) -> dict:
    """Classify Step 3 results with tolerance so near-zero deltas do not become false errors."""
    baseline = float(summary.get("baseline_final", 0.0) or 0.0)
    conservative = float(summary.get("conservative_final", 0.0) or 0.0)
    expected = float(summary.get("expected_final", 0.0) or 0.0)

    if baseline <= 0.0:
        expected_pct = 0.0
        conservative_pct = 0.0
    else:
        expected_pct = (expected - baseline) / baseline
        conservative_pct = (conservative - baseline) / baseline

    if conservative_pct >= -materiality_pct and expected_pct >= materiality_pct:
        return {
            "level": "success",
            "title": "robust",
            "message": "Your plan looks robust in this run: the expected case improves and the conservative case stays close to baseline.",
        }

    if expected_pct >= materiality_pct:
        return {
            "level": "warning",
            "title": "upside_sensitive",
            "message": "Your plan has upside, but it is downside-sensitive: expected improves while conservative falls below baseline.",
        }

    if expected_pct > -materiality_pct:
        return {
            "level": "warning",
            "title": "roughly_in_line",
            "message": "Your plan is roughly in line with baseline, but weaker scenarios show downside sensitivity.",
        }

    return {
        "level": "error",
        "title": "stretched",
        "message": "Your plan looks stretched in this scenario: even the expected case is materially below baseline.",
    }


def _render_concise_explanation(summary: dict, *, planning_horizon: int, target_weekly: float) -> None:
    """Render a short plain-English explanation of the Step 3 outcome."""
    baseline_final = float(summary.get("baseline_final", 0.0) or 0.0)
    conservative = float(summary.get("conservative_final", 0.0) or 0.0)
    expected = float(summary.get("expected_final", 0.0) or 0.0)
    optimistic = float(summary.get("optimistic_final", 0.0) or 0.0)
    target_path = float(target_weekly) * float(planning_horizon)

    expected_delta = expected - baseline_final
    conservative_delta = conservative - baseline_final
    if abs(expected_delta) < 0.5:
        expected_delta = 0.0
    if abs(conservative_delta) < 0.5:
        conservative_delta = 0.0

    range_width = optimistic - conservative

    st.markdown("### What happened")
    st.markdown(
        f"- Baseline ends around **£{baseline_final:,.0f}** after **{planning_horizon} weeks**.\n"
        f"- Expected outcome is **£{expected:,.0f}** ({expected_delta:+,.0f} vs baseline).\n"
        f"- Conservative outcome is **£{conservative:,.0f}** ({conservative_delta:+,.0f} vs baseline)."
    )

    st.markdown("### What it means")
    classification = _classify_short_term_result(summary)
    if classification["level"] == "success":
        st.success(classification["message"])
    elif classification["level"] == "warning":
        st.warning(classification["message"])
    else:
        st.error(classification["message"])

    st.markdown("### Main trade-off")
    st.markdown(
        f"- Savings target path: **£{target_path:,.0f}** by week {planning_horizon}.\n"
        f"- 10–90% outcome range: about **£{range_width:,.0f}**.\n"
        "- Wider ranges mean the plan depends more on real-life spending variability."
    )


def _render_action_guidance(summary: dict) -> None:
    """Render high-level guidance based on the classified short-term result."""
    classification = _classify_short_term_result(summary)

    st.markdown("### What to do next")
    if classification["level"] == "success":
        st.success(
            "Your plan looks robust in this short-term test. You could keep the target as a safety buffer, "
            "or increase it slightly in Step 2 if you want a more ambitious plan."
        )
    elif classification["title"] == "roughly_in_line":
        st.warning(
            "Your plan is close to baseline. Keep the target if you want a cautious plan, or try a tougher scenario "
            "before increasing it."
        )
    elif classification["level"] == "warning":
        st.warning(
            "Your plan has upside but is sensitive to weaker weeks. Keep the target, lower it slightly, "
            "or try a tougher scenario before continuing."
        )
    else:
        st.error(
            "Your plan looks stretched in this scenario. Consider lowering the weekly target or returning to Step 1 "
            "to adjust the cash-flow snapshot."
        )


def render_step_3(*, embedded: bool = False) -> None:
    """Render the legacy/embedded Step 3 short-term feasibility page."""
    if embedded:
        st.caption("Stress-test the selected weekly savings target before continuing.")
    else:
        section_header("Step 3 — Short-term feasibility")

    snapshot = coerce_snapshot()
    if not snapshot:
        st.warning("No planning snapshot found yet. Complete the current situation and savings target sections first.")
        if not embedded:
            if st.button("Go back to Step 1", key="step3_go_step1"):
                st.session_state[CURRENT_STEP] = 1
                st.session_state["current_step"] = 1
                st.rerun()
        return

    pathway = _current_pathway()
    snapshot = _sync_step3_assumption_controls(snapshot)

    view_model = build_step3_view_model(snapshot)
    baseline_df = view_model["baseline_df"]
    plan_a_df = view_model["plan_a_df"]
    summary = view_model["summary"]
    next_actions = view_model.get("next_actions", "")
    explanation = view_model.get("explanation", "")

    planning_horizon = int(
        snapshot.get("planning_horizon_weeks", st.session_state.get(STEP2_PLANNING_HORIZON_WEEKS, 12)) or 12
    )
    target_weekly = float(snapshot.get("target_a_weekly", st.session_state.get(TARGET_A_WEEKLY, 0.0)) or 0.0)
    shock_week = int(snapshot.get("shock_week", 0) or 0)
    shock_amount = float(snapshot.get("shock_amount", 0.0) or 0.0)

    _render_scenario_summary(
        snapshot=snapshot,
        planning_horizon=planning_horizon,
        target_weekly=target_weekly,
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        currency_metric("Conservative", float(summary["conservative_final"]), decimals=0)
    with m2:
        currency_metric("Expected", float(summary["expected_final"]), decimals=0)
    with m3:
        currency_metric("High case", float(summary["optimistic_final"]), decimals=0)

    st.caption(
        f"Baseline final balance: **£{float(summary['baseline_final']):,.0f}**. "
        "The figures above show the conservative, expected and high-case outcomes for this scenario."
    )

    st.markdown("## Short-term cash-flow path")
    _render_projection_chart(
        baseline_df,
        plan_a_df,
        target_weekly=target_weekly,
        shock_week=shock_week,
        shock_amount=shock_amount,
    )

    _render_action_guidance(summary)

    if next_actions:
        with st.expander("Detailed next actions", expanded=False):
            if isinstance(next_actions, dict):
                title = str(next_actions.get("title", "Next actions") or "Next actions")
                body = str(next_actions.get("body", "") or "")
                level = str(next_actions.get("level", "info") or "info")

                st.markdown(f"### {title}")

                if level == "success":
                    st.success(body)
                elif level == "warning":
                    st.warning(body)
                elif level == "error":
                    st.error(body)
                else:
                    st.info(body)
            else:
                st.markdown(str(next_actions))

    with st.expander("See concise explanation", expanded=False):
        _render_concise_explanation(summary, planning_horizon=planning_horizon, target_weekly=target_weekly)

    if explanation:
        with st.expander("Technical simulation details", expanded=False):
            st.markdown(str(explanation))

    st.markdown("---")
    card_title, card_body, button_label, next_step = _branch_card(pathway)

    if embedded:
        st.caption(f"**{card_title}:** {card_body}")
        if st.button(button_label, key=f"step3_continue_branch_{pathway}", use_container_width=True):
            _go_to_next_branch(pathway, next_step)
        return

    left, right = st.columns([1.0, 2.0])
    with left:
        if st.button("Back to Step 2", key="step3_back"):
            st.session_state[CURRENT_STEP] = 2
            st.session_state["current_step"] = 2
            st.rerun()
    with right:
        st.caption(f"**{card_title}:** {card_body}")
        if st.button(button_label, key=f"step3_continue_branch_{pathway}", use_container_width=True):
            _go_to_next_branch(pathway, next_step)