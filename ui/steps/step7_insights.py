"""
Step 7 — Interpretation & Insights

Branch-aware version:
- savings_only: no Step 5 engine required.
- compare_both / savings_plus_investing: keep investment-engine interpretation.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import streamlit as st

from ui.common.cards import insight_card
from ui.common.messages import section_header
from ui.state.keys import CURRENT_STEP, ENGINE_HAS_RUN, INVESTMENT_PROJECTION_RESULT


STEP0_PATHWAY = "step0_planning_pathway"
CASH_ONLY_PROJECTION_RESULT = "investment_projection_cash_only_result"


def _current_pathway() -> str:
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

    return "compare_both"


def _coerce_mapping(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            mapped = to_dict()
            if isinstance(mapped, dict):
                return dict(mapped)
        except Exception:
            pass
    try:
        return dict(value)
    except Exception:
        return {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if out == out and out not in (float("inf"), float("-inf")):
            return float(out)
    except Exception:
        pass
    return float(default)


def _fmt_gbp0(value: Any) -> str:
    return f"£{_safe_float(value, 0.0):,.0f}"


def _fmt_gbp_compact(value: Any) -> str:
    """Compact money labels for narrow Streamlit metric cards."""
    amount = _safe_float(value, 0.0)
    sign = "-" if amount < 0 else ""
    amount = abs(float(amount))
    if amount >= 1_000_000:
        return f"{sign}£{amount / 1_000_000:.2f}m"
    if amount >= 100_000:
        return f"{sign}£{amount / 1_000:.0f}k"
    if amount >= 10_000:
        return f"{sign}£{amount / 1_000:.1f}k"
    return f"{sign}£{amount:,.0f}"


def _fmt_pct_from_fraction(value: Any, decimals: int = 1) -> str:
    return f"{_safe_float(value, 0.0) * 100.0:.{decimals}f}%"


def _extract_engine_summary() -> dict:
    run_result = _coerce_mapping(st.session_state.get("step5_run_result", {}))
    return _coerce_mapping(run_result.get("performance_summary", {}))


def _extract_projection_summary(key: str = INVESTMENT_PROJECTION_RESULT) -> dict:
    projection_result = _coerce_mapping(st.session_state.get(key, {}))
    summary = _coerce_mapping(projection_result.get("summary", {}))
    if summary:
        return summary

    df = projection_result.get("projection_df")
    if isinstance(df, pd.DataFrame) and not df.empty:
        value_col = "projected_value" if "projected_value" in df.columns else None
        if value_col:
            final_value = float(df[value_col].iloc[-1])
            return {
                "final_value": final_value,
                "expected_terminal": final_value,
                "median_terminal": final_value,
                "p10_terminal": final_value,
                "p90_terminal": final_value,
            }

    return {}


def _metric(summary: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in summary and summary.get(key) is not None:
            return _safe_float(summary.get(key), default)
    return float(default)


def _render_projection_summary(proj: Dict[str, Any], *, title: str) -> None:
    st.markdown(f"### {title}")

    expected = _metric(proj, "expected_terminal", "final_value")
    median = _metric(proj, "median_terminal", "expected_terminal", "final_value")
    p10 = _metric(proj, "p10_terminal", "median_terminal", "expected_terminal", "final_value")
    p90 = _metric(proj, "p90_terminal", "median_terminal", "expected_terminal", "final_value")
    contributed = _metric(proj, "total_contributed")
    growth = _metric(proj, "expected_profit", "gain_from_growth", default=expected - contributed)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Expected", _fmt_gbp0(expected))
    with c2:
        st.metric("Median", _fmt_gbp0(median))
    with c3:
        st.metric("P10–P90", f"{_fmt_gbp_compact(p10)}–{_fmt_gbp_compact(p90)}")
    with c4:
        st.metric("Growth above contributions", _fmt_gbp0(growth))

    st.caption(
        "Scenario output only: these figures come from the selected simulation assumptions and are not guaranteed forecasts."
    )


def _render_savings_only_insights(proj: Dict[str, Any]) -> None:
    if not proj:
        st.info("Open Step 6 first to generate the savings-only pathway view.")
        if st.button("Back to Step 6", key="step7_savings_missing_back"):
            st.session_state[CURRENT_STEP] = 6
            st.session_state["current_step"] = 6
            st.rerun()
        return

    _render_projection_summary(proj, title="Savings-only pathway readout")

    terminal = _metric(proj, "median_terminal", "expected_terminal", "final_value")
    monthly = _metric(proj, "monthly_contribution")
    weekly = _metric(proj, "weekly_contribution")
    goal = _metric(proj, "goal_amount")
    goal_prob = proj.get("probability_of_reaching_goal")
    contribution_source = str(proj.get("contribution_source", "planned saving target") or "planned saving target")

    insight_card(
        "Decision readout",
        "This path answers a simple question: how far can disciplined saving take you over the long term without adding market risk?",
        level="info",
    )

    if monthly > 0:
        st.write(
            f"The savings-only path is based on sustaining about **{_fmt_gbp0(monthly)}/month** "
            f"(**{_fmt_gbp0(weekly)}/week**) from your **{contribution_source}**."
        )

    if goal > 0:
        gap = terminal - goal
        if goal_prob is not None and float(goal_prob) >= 1.0:
            insight_card(
                "Goal feasibility",
                "Savings alone reaches the stated goal in the 20-year pathway view.",
                level="success",
            )
        elif terminal >= goal * 0.80:
            insight_card(
                "Goal feasibility",
                f"Savings alone gets close, but remains about **{_fmt_gbp0(abs(gap))}** below the goal in the 20-year view.",
                level="info",
            )
        else:
            insight_card(
                "Goal feasibility",
                f"Savings alone appears materially short of the goal, with a gap of about **{_fmt_gbp0(abs(gap))}** in the 20-year view.",
                level="warning",
            )
    else:
        insight_card(
            "Goal feasibility",
            "No explicit long-term goal was found, so this branch should be read as accumulation potential rather than goal success/failure.",
            level="info",
        )

    horizon_rows = proj.get("horizon_table", [])
    if isinstance(horizon_rows, list) and horizon_rows:
        st.markdown("### Horizon interpretation")
        for row in horizon_rows:
            horizon = str(row.get("Horizon", ""))
            total = _safe_float(row.get("Total saved", 0.0))
            interpretation = str(row.get("Interpretation", ""))
            st.write(f"• **{horizon}:** {_fmt_gbp0(total)} — {interpretation}")

    st.markdown("### What this means")
    st.write("• If savings alone is enough, the lower-risk path may already be sufficient for the goal.")
    st.write("• If savings alone is close, small changes to contribution or horizon may matter more than adding complexity.")
    st.write("• If savings alone is far short, the compare branch can test whether investing changes the gap — without treating investing as a recommendation.")

    st.caption(
        "Savings-only pathway insights do not include investment return, volatility, drawdown or market-risk assumptions."
    )


def _render_engine_metrics(engine: Dict[str, Any]) -> tuple[float, float, float]:
    st.markdown("### Key metrics")
    c1, c2, c3 = st.columns(3)

    sharpe = _safe_float(engine.get("sharpe", 0.0))
    cagr = _safe_float(engine.get("cagr", 0.0))
    max_dd = _safe_float(engine.get("max_drawdown", 0.0))

    with c1:
        st.metric("Sharpe", f"{sharpe:.2f}")
    with c2:
        st.metric("CAGR", _fmt_pct_from_fraction(cagr))
    with c3:
        st.metric("Max drawdown", _fmt_pct_from_fraction(max_dd))

    return sharpe, cagr, max_dd


def _render_investing_insights(proj: Dict[str, Any], cash_proj: Dict[str, Any], engine: Dict[str, Any], *, compare: bool) -> None:
    sharpe, cagr, max_dd = _render_engine_metrics(engine)

    st.markdown("### Strategy interpretation")
    if sharpe >= 1.0:
        insight_card("Efficiency", "This simulated run shows strong historical risk-adjusted behaviour.", level="success")
    elif sharpe >= 0.7:
        insight_card("Efficiency", "This simulated run shows reasonable historical efficiency, with room to improve.", level="info")
    else:
        insight_card("Efficiency", "This simulated run shows lower risk-adjusted efficiency under the current assumptions.", level="warning")

    if cagr >= 0.10:
        insight_card("Growth", "This scenario is growth-oriented relative to contributions alone.", level="success")
    elif cagr >= 0.07:
        insight_card("Growth", "This scenario has a balanced historical return profile.", level="info")
    else:
        insight_card("Growth", "This scenario has a lower-return / more defensive historical profile.", level="warning")

    if abs(max_dd) <= 0.12:
        insight_card("Risk", "Drawdowns are relatively contained in the historical path.", level="success")
    elif abs(max_dd) <= 0.20:
        insight_card("Risk", "Drawdowns are moderate in the historical path and may still feel uncomfortable for some users.", level="info")
    else:
        insight_card("Risk", "Historical drawdown risk is high and may be psychologically difficult.", level="warning")

    if proj:
        _render_projection_summary(proj, title="Investment pathway readout")
    else:
        st.info("Run Step 6 projection to unlock deeper long-term interpretation.")

    if compare and proj and cash_proj:
        st.markdown("### Savings-only vs investing decision")
        cash_terminal = _metric(cash_proj, "median_terminal", "expected_terminal", "final_value")
        inv_terminal = _metric(proj, "median_terminal", "expected_terminal", "final_value")
        diff = inv_terminal - cash_terminal

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Savings-only median", _fmt_gbp0(cash_terminal))
        with c2:
            st.metric("Investing median", _fmt_gbp0(inv_terminal))
        with c3:
            st.metric("Scenario difference", _fmt_gbp0(diff))

        if diff > 0:
            insight_card(
                "Decision trade-off",
                "In this scenario, the investing path shows higher central terminal wealth, but that upside comes with market uncertainty and drawdown risk.",
                level="info",
            )
        else:
            insight_card(
                "Decision trade-off",
                "The investing scenario does not clearly beat the savings-only path in this run, so the extra uncertainty may not be justified under these assumptions.",
                level="warning",
            )

    st.markdown("### Where to improve")
    if sharpe < 0.9:
        st.write("• Improve efficiency through better diversification or clearer strategy assumptions.")
    if abs(max_dd) > 0.15:
        st.write("• Reduce drawdowns for a smoother user experience.")
    if cagr < 0.10:
        st.write("• Increase growth exposure only if aligned with risk tolerance.")
    if proj and cash_proj:
        st.write("• Review whether the projected gain over cash-only is large enough to justify additional uncertainty.")
    st.write("• Treat the outputs as scenario comparisons, not guaranteed forecasts.")


def render_step_7() -> None:
    pathway = _current_pathway()

    if pathway == "savings_only":
        section_header("Step 7 — Savings-only pathway insights")
        proj = _extract_projection_summary(INVESTMENT_PROJECTION_RESULT)
        _render_savings_only_insights(proj)

        st.markdown("---")
        left, right = st.columns(2)
        with left:
            if st.button("Back to Step 6", key="step7_savings_back_to_step6"):
                st.session_state[CURRENT_STEP] = 6
                st.session_state["current_step"] = 6
                st.rerun()
        with right:
            st.success("You have completed this planning path.")
        return

    section_header(
        "Step 7 — Savings vs investing insights"
        if pathway == "compare_both"
        else "Step 7 — Investment pathway insights"
    )

    if not bool(st.session_state.get(ENGINE_HAS_RUN, False)):
        st.warning("Run the engine in Step 5 to unlock investment insights.")
        if st.button("Go back to Step 5", key="step7_go_back_step5"):
            st.session_state[CURRENT_STEP] = 5
            st.session_state["current_step"] = 5
            st.rerun()
        return

    engine = _extract_engine_summary()
    proj = _extract_projection_summary(INVESTMENT_PROJECTION_RESULT)
    cash_proj = _extract_projection_summary(CASH_ONLY_PROJECTION_RESULT)

    _render_investing_insights(
        proj,
        cash_proj,
        engine,
        compare=(pathway == "compare_both"),
    )

    st.markdown("---")
    left, right = st.columns(2)
    with left:
        if st.button("Back to Step 6", key="step7_back_to_step6"):
            st.session_state[CURRENT_STEP] = 6
            st.session_state["current_step"] = 6
            st.rerun()
    with right:
        st.success("You have completed the full workflow.")



# ============================================================
# Compare-both insights override
# ============================================================

COMPARE_BRANCH_RESULT = "step6_compare_branch_result"


def _render_compare_branch_insights(engine: Dict[str, Any]) -> None:
    compare_payload = _coerce_mapping(st.session_state.get(COMPARE_BRANCH_RESULT, {}))
    rows = compare_payload.get("rows", [])
    proj = _extract_projection_summary(INVESTMENT_PROJECTION_RESULT)
    cash_proj = _extract_projection_summary(CASH_ONLY_PROJECTION_RESULT)

    _render_engine_metrics(engine)

    if not rows:
        st.info("Run the fair pathway comparison in Step 6 to unlock comparison insights.")
        return

    compare_df = pd.DataFrame(rows)
    valid = compare_df.dropna(subset=["Savings-only", "Investing median"], how="any") if not compare_df.empty else pd.DataFrame()

    st.markdown("### Decision comparison")
    if valid.empty:
        st.warning("No complete comparison rows were found. Re-run Step 6 comparison.")
        return

    valid["Difference"] = pd.to_numeric(valid["Difference"], errors="coerce")
    best_row = valid.iloc[valid["Difference"].abs().idxmax()] if valid["Difference"].notna().any() else valid.iloc[-1]
    last_row = valid.sort_values("_years").iloc[-1] if "_years" in valid.columns else valid.iloc[-1]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Longest savings-only", _fmt_gbp0(last_row.get("Savings-only", 0.0)))
    with c2:
        st.metric("Longest investing median", _fmt_gbp0(last_row.get("Investing median", 0.0)))
    with c3:
        st.metric("Median difference", _fmt_gbp0(last_row.get("Difference", 0.0)))

    positive_rows = valid[pd.to_numeric(valid["Difference"], errors="coerce") > 0]
    negative_rows = valid[pd.to_numeric(valid["Difference"], errors="coerce") <= 0]

    if len(positive_rows) == len(valid):
        insight_card(
            "Comparison readout",
            "Across the tested horizons, the simulated investing pathway has a higher median outcome than savings-only. The key question is whether the extra upside is worth accepting uncertainty and drawdown risk.",
            level="info",
        )
    elif len(positive_rows) > 0:
        first_positive = positive_rows.sort_values("_years").iloc[0]
        insight_card(
            "Comparison readout",
            f"Investing does not dominate equally at every horizon. It first shows a positive median difference around **{first_positive.get('Horizon', 'one of the tested horizons')}**.",
            level="info",
        )
    else:
        insight_card(
            "Comparison readout",
            "In this run, investing does not clearly beat savings-only on median outcome across the tested horizons. That makes the extra uncertainty harder to justify under these assumptions.",
            level="warning",
        )

    st.markdown("### Horizon interpretation")
    for _, row in valid.sort_values("_years").iterrows():
        horizon = str(row.get("Horizon", ""))
        savings = _safe_float(row.get("Savings-only", 0.0))
        investing = _safe_float(row.get("Investing median", 0.0))
        diff = _safe_float(row.get("Difference", 0.0))
        interp = str(row.get("Interpretation", ""))
        st.write(
            f"• **{horizon}:** savings-only {_fmt_gbp0(savings)} vs investing median {_fmt_gbp0(investing)} "
            f"({ _fmt_gbp0(diff) } difference) — {interp}."
        )

    if proj:
        _render_projection_summary(proj, title="Investment pathway readout")
    if cash_proj:
        _render_projection_summary(cash_proj, title="Savings-only pathway readout")

    st.markdown("### What this means")
    st.write("• The comparison is fair only because both branches use the same contribution, starting pot, goal and horizons.")
    st.write("• Savings-only is simpler and avoids market volatility, but has limited upside.")
    st.write("• In the simulated comparison, investing may improve the median outcome, but it introduces uncertainty and potential drawdowns.")
    st.write("• This is a decision-support comparison, not a recommendation to invest.")


def render_step_7() -> None:
    pathway = _current_pathway()

    if pathway == "savings_only":
        section_header("Step 7 — Savings-only pathway insights")
        proj = _extract_projection_summary(INVESTMENT_PROJECTION_RESULT)
        _render_savings_only_insights(proj)

        st.markdown("---")
        left, right = st.columns(2)
        with left:
            if st.button("Back to Step 6", key="step7_savings_back_to_step6"):
                st.session_state[CURRENT_STEP] = 6
                st.session_state["current_step"] = 6
                st.rerun()
        with right:
            st.success("You have completed this planning path.")
        return

    if pathway == "compare_both":
        section_header("Step 7 — Savings vs investing decision insights")
        if not bool(st.session_state.get(ENGINE_HAS_RUN, False)):
            st.warning("Run the engine in Step 5 to unlock comparison insights.")
            if st.button("Go back to Step 5", key="step7_compare_go_back_step5"):
                st.session_state[CURRENT_STEP] = 5
                st.session_state["current_step"] = 5
                st.rerun()
            return

        engine = _extract_engine_summary()
        _render_compare_branch_insights(engine)

        st.markdown("---")
        left, right = st.columns(2)
        with left:
            if st.button("Back to Step 6", key="step7_compare_back_to_step6"):
                st.session_state[CURRENT_STEP] = 6
                st.session_state["current_step"] = 6
                st.rerun()
        with right:
            st.success("You have completed the comparison workflow.")
        return

    section_header("Step 7 — Investment pathway insights")

    if not bool(st.session_state.get(ENGINE_HAS_RUN, False)):
        st.warning("Run the engine in Step 5 to unlock investment insights.")
        if st.button("Go back to Step 5", key="step7_go_back_step5"):
            st.session_state[CURRENT_STEP] = 5
            st.session_state["current_step"] = 5
            st.rerun()
        return

    engine = _extract_engine_summary()
    proj = _extract_projection_summary(INVESTMENT_PROJECTION_RESULT)
    cash_proj = _extract_projection_summary(CASH_ONLY_PROJECTION_RESULT)

    _render_investing_insights(
        proj,
        cash_proj,
        engine,
        compare=False,
    )

    st.markdown("---")
    left, right = st.columns(2)
    with left:
        if st.button("Back to Step 6", key="step7_back_to_step6"):
            st.session_state[CURRENT_STEP] = 6
            st.session_state["current_step"] = 6
            st.rerun()
    with right:
        st.success("You have completed the full workflow.")
