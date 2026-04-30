"""
Investment Pathway Insights.

Proxy-aware final-polish version:
- Savings-only insights work without the Strategy Engine.
- Investment / comparison insights can interpret either:
  1. tested Strategy Engine results, or
  2. a clearly labelled educational investment proxy.
- The page only asks the user to return to the Scenario Explorer when the
  relevant long-term projection has not been generated yet.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

import pandas as pd
import streamlit as st

from ui.common.cards import insight_card
from ui.common.messages import section_header
from ui.state.keys import CURRENT_STEP, ENGINE_HAS_RUN, INVESTMENT_PROJECTION_RESULT


STEP0_PATHWAY = "step0_planning_pathway"
STEP6_VIEW_MODE_KEY = "step6_projection_view_mode_v1"
CASH_ONLY_PROJECTION_RESULT = "investment_projection_cash_only_result"
COMPARE_BRANCH_RESULT = "step6_compare_branch_result"


def _current_pathway() -> str:
    """Resolve the broad route selected by Home / legacy branch state."""
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


def _current_step7_mode() -> str:
    """Resolve the Step 7 report mode without changing existing state keys.

    The broad Home pathway can be ``compare_both`` while Step 6 stores the
    active projection view as ``savings_plus_investing``. When a comparison
    payload exists, or when both projection branches exist under the compare
    pathway, Step 7 should open the comparison report rather than falling back
    to the investment-only report.
    """
    pathway = _current_pathway()
    mode = str(st.session_state.get(STEP6_VIEW_MODE_KEY, "") or "").strip()

    compare_payload = _coerce_mapping(st.session_state.get(COMPARE_BRANCH_RESULT, {}))
    compare_rows = compare_payload.get("rows", [])
    has_compare_rows = isinstance(compare_rows, list) and len(compare_rows) > 0

    if mode == "savings_only":
        return "savings_only"

    if mode == "compare_both":
        return "compare_both"

    if pathway == "compare_both":
        investment_summary = _extract_projection_summary(INVESTMENT_PROJECTION_RESULT)
        savings_summary = _extract_projection_summary(CASH_ONLY_PROJECTION_RESULT)
        if has_compare_rows or (_has_projection(investment_summary) and _has_projection(savings_summary)):
            return "compare_both"

    if mode in {"savings_plus_investing"}:
        return mode
    return pathway


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


def _fmt_pct_from_fraction(value: Any, decimals: int = 1) -> str:
    return f"{_safe_float(value, 0.0) * 100.0:.{decimals}f}%"


def _extract_engine_summary() -> dict:
    """Read the latest Step 5 performance summary from legacy and current keys."""
    for key in ("step5_last_run_result", "step5_run_result", "last_run_result", "run_result"):
        run_result = _coerce_mapping(st.session_state.get(key, {}))
        summary = _coerce_mapping(run_result.get("performance_summary", {}))
        if summary:
            return summary
    return {}


def _extract_projection_payload(key: str = INVESTMENT_PROJECTION_RESULT) -> dict:
    return _coerce_mapping(st.session_state.get(key, {}))


def _extract_projection_summary(key_or_payload: str | Dict[str, Any] = INVESTMENT_PROJECTION_RESULT) -> dict:
    if isinstance(key_or_payload, str):
        projection_result = _extract_projection_payload(key_or_payload)
    else:
        projection_result = _coerce_mapping(key_or_payload)

    summary = _coerce_mapping(projection_result.get("summary", {}))
    if summary:
        return summary

    result = _coerce_mapping(projection_result.get("result", {}))
    result_summary = _coerce_mapping(result.get("summary", {}))
    if result_summary:
        return result_summary

    df = projection_result.get("projection_df")
    if isinstance(df, pd.DataFrame) and not df.empty:
        for value_col in ("projected_value", "value", "wealth", "portfolio_value"):
            if value_col in df.columns:
                final_value = _safe_float(df[value_col].iloc[-1])
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


def _has_projection(summary: Dict[str, Any]) -> bool:
    return any(
        _metric(summary, key, default=0.0) > 0.0
        for key in ("median_terminal", "expected_terminal", "final_value", "p90_terminal")
    )


def _engine_result_is_real() -> bool:
    """Return True only when Step 5 genuinely ran.

    Step 6 can now generate an educational proxy before Step 5. Step 7 should
    interpret that proxy, but should not call it a tested engine result.
    """
    return bool(st.session_state.get(ENGINE_HAS_RUN, False))


def _evidence_label() -> str:
    return "Tested strategy engine result" if _engine_result_is_real() else "Educational investment proxy"


def _render_missing_step6_result(message: str, *, button_key: str) -> None:
    st.info(message)
    if st.button("← Back to Scenario Explorer", key=button_key):
        st.session_state[CURRENT_STEP] = 6
        st.session_state["current_step"] = 6
        st.rerun()


def _render_projection_summary(proj: Dict[str, Any], *, title: str) -> None:
    st.markdown(f"### {title}")
    st.caption(
        "Scenario readout based on the selected contribution, horizon and return assumptions. "
        "Values are model outputs, not guaranteed future outcomes."
    )

    expected = _metric(proj, "expected_terminal", "final_value")
    median = _metric(proj, "median_terminal", "expected_terminal", "final_value")
    p10 = _metric(proj, "p10_terminal", "median_terminal", "expected_terminal", "final_value")
    p90 = _metric(proj, "p90_terminal", "median_terminal", "expected_terminal", "final_value")
    contributed = _metric(proj, "total_contributed")
    growth = _metric(proj, "expected_profit", "gain_from_growth", default=expected - contributed)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Expected terminal", _fmt_gbp0(expected))
    with c2:
        st.metric("Median terminal", _fmt_gbp0(median))
    with c3:
        st.metric("P10 downside case", _fmt_gbp0(p10))
    with c4:
        st.metric("P90 upside case", _fmt_gbp0(p90))
    with c5:
        st.metric("Modelled growth", _fmt_gbp0(growth))


def _render_engine_metrics(engine: Dict[str, Any]) -> tuple[float, float, float]:
    st.markdown("### Strategy engine snapshot")
    c1, c2, c3 = st.columns(3)

    sharpe = _safe_float(engine.get("sharpe", 0.0))
    cagr = _safe_float(engine.get("cagr", 0.0))
    max_dd = _safe_float(engine.get("max_drawdown", 0.0))

    with c1:
        st.metric("Sharpe", f"{sharpe:.2f}")
    with c2:
        st.metric("CAGR", _fmt_pct_from_fraction(cagr))
    with c3:
        st.metric("Drawdown severity", _fmt_pct_from_fraction(abs(max_dd)))

    return sharpe, cagr, max_dd


def _render_proxy_evidence_note(*, compare: bool = False) -> None:
    if _engine_result_is_real():
        st.success(
            "Evidence level: **tested Strategy Engine result** interpreted through the long-term Scenario Explorer."
        )
        return

    if compare:
        st.info(
            "Evidence level: **educational proxy**. This comparison contrasts savings-only outcomes with a labelled investment proxy. "
            "It is useful for demonstrating the full report flow, but it is not a tested Strategy Engine result. "
            "Run the Investment Strategy Lab to replace the proxy automatically."
        )
    else:
        st.info(
            "Evidence level: **educational proxy**. This investment pathway was generated before a tested Strategy Engine run was available. "
            "Treat it as an illustrative scenario; running the Investment Strategy Lab replaces it with tested strategy returns."
        )




def _render_report_summary(
    *,
    mode_label: str,
    proj: Dict[str, Any] | None = None,
    cash_proj: Dict[str, Any] | None = None,
    engine: Dict[str, Any] | None = None,
    compare: bool = False,
) -> None:
    """Render a compact executive-style summary for the final report."""
    proj = _coerce_mapping(proj or {})
    cash_proj = _coerce_mapping(cash_proj or {})
    engine = _coerce_mapping(engine or {})

    if compare:
        inv_terminal = _metric(proj, "median_terminal", "expected_terminal", "final_value")
        cash_terminal = _metric(cash_proj, "median_terminal", "expected_terminal", "final_value")
        diff = inv_terminal - cash_terminal
        evidence = "a tested strategy engine result" if _engine_result_is_real() else "an educational investment proxy"
        if _has_projection(proj) and _has_projection(cash_proj):
            verdict = "higher" if diff > 0 else "lower"
            text = (
                f"This final comparison reads the savings-only path against {evidence}. "
                f"The investment-side median terminal value is **{_fmt_gbp0(inv_terminal)}**, versus **{_fmt_gbp0(cash_terminal)}** "
                f"for savings-only, making the modelled central difference **{_fmt_gbp0(diff)}**. "
                f"In this run, the investment-side central outcome is {verdict} than the savings-only path under the selected assumptions. "
                "The decision still depends on whether the extra uncertainty, drawdown exposure and model risk are acceptable."
            )
        else:
            text = (
                f"This report is designed to compare savings-only planning against {evidence}. "
                "Generate both Scenario Explorer branches to complete the comparison readout."
            )
        insight_card("Report summary", text, level="info")
        return

    if mode_label == "savings_only":
        terminal = _metric(proj, "median_terminal", "expected_terminal", "final_value")
        goal = _metric(proj, "goal_amount")
        if goal > 0:
            gap = terminal - goal
            goal_text = "above" if gap >= 0 else "below"
            text = (
                f"This report summarises the savings-only pathway. The central projected outcome is **{_fmt_gbp0(terminal)}**, "
                f"which is **{_fmt_gbp0(abs(gap))}** {goal_text} the selected goal under the current contribution assumptions. "
                "This route is simpler and avoids market drawdown risk, but it also gives up potential investment upside."
            )
        else:
            text = (
                f"This report summarises the savings-only pathway. The central projected outcome is **{_fmt_gbp0(terminal)}** "
                "under the current contribution assumptions. This route is simpler and avoids market drawdown risk, but it also gives up potential investment upside."
            )
        insight_card("Report summary", text, level="info")
        return

    median = _metric(proj, "median_terminal", "expected_terminal", "final_value")
    p10 = _metric(proj, "p10_terminal", "median_terminal", "expected_terminal", "final_value")
    p90 = _metric(proj, "p90_terminal", "median_terminal", "expected_terminal", "final_value")
    if _engine_result_is_real():
        sharpe = _safe_float(engine.get("sharpe", 0.0))
        cagr = _safe_float(engine.get("cagr", 0.0))
        dd_abs = abs(_safe_float(engine.get("max_drawdown", 0.0)))
        text = (
            f"This report interprets a tested Strategy Engine result through the long-term Scenario Explorer. "
            f"The strategy snapshot shows Sharpe **{sharpe:.2f}**, CAGR near **{_fmt_pct_from_fraction(cagr)}**, "
            f"and drawdown severity near **{_fmt_pct_from_fraction(dd_abs)}**. The modelled median terminal value is "
            f"**{_fmt_gbp0(median)}**, with a scenario range from **{_fmt_gbp0(p10)}** to **{_fmt_gbp0(p90)}**. "
            "This is a tested historical pathway translated into a personal contribution scenario, not a forecast."
        )
        insight_card("Report summary", text, level="info")
    else:
        text = (
            f"This report currently uses an educational investment proxy. The modelled median terminal value is "
            f"**{_fmt_gbp0(median)}**, with a scenario range from **{_fmt_gbp0(p10)}** to **{_fmt_gbp0(p90)}**. "
            "Use this as an interface and scenario demonstration until the Investment Strategy Lab creates a tested strategy return path."
        )
        insight_card("Report summary", text, level="warning")


def _render_recommended_next_actions(*, mode_label: str, compare: bool = False) -> None:
    """Final report actions. These are app-navigation suggestions, not financial advice."""
    st.markdown("### Recommended next actions")
    if compare:
        if _engine_result_is_real():
            st.write("• Use the comparison as a decision filter: higher median outcome only matters if the drawdown and uncertainty are acceptable.")
            st.write("• Return to the Scenario Explorer to stress-test the result with a different horizon, contribution level, starting pot or goal.")
            st.write("• Return to the Strategy Engine if the investment branch needs lower drawdown, a different universe, or a cleaner risk profile.")
        else:
            st.write("• Run the Investment Strategy Lab to replace the educational proxy with a tested Strategy Engine result.")
            st.write("• Re-run the Scenario Explorer after the real strategy result exists so the comparison uses stronger evidence.")
            st.write("• Treat the current comparison as a report-flow demonstration, not as a basis for choosing the investment branch.")
        return

    if mode_label == "savings_only":
        st.write("• Keep the savings-only route if the projected range already meets the goal with acceptable effort and timeline.")
        st.write("• Return to the Scenario Explorer to test a different contribution, horizon, starting pot or goal amount.")
        st.write("• Explore the investment pathway only if the savings-only route leaves a meaningful gap and market uncertainty is acceptable.")
        return

    if _engine_result_is_real():
        st.write("• Treat this as a candidate pathway only if the drawdown level and uncertainty are acceptable.")
        st.write("• Return to the Strategy Engine if the result needs lower drawdown, a smoother risk profile, or a different asset universe.")
        st.write("• Return to the Scenario Explorer to test whether the same strategy still works under different contribution and horizon assumptions.")
    else:
        st.write("• Run the Investment Strategy Lab to replace this proxy with tested strategy returns.")
        st.write("• Re-run the Scenario Explorer after the Strategy Engine has produced a real return path.")
        st.write("• Treat this screen as an educational demonstration until then.")

def _render_decision_support_note(*, compare: bool = False) -> None:
    with st.expander("How to use these outputs safely", expanded=False):
        insight_card(
            "Decision-support only",
            "Use this page to compare pathways, assumptions and trade-offs. It is not financial advice, an investment recommendation, or a promise of future results.",
            level="info",
        )
        insight_card(
            "Past performance warning",
            "Historical returns and educational proxies are not reliable indicators of future results. Scenario ranges should be read as model outputs, not expected real-money outcomes.",
            level="warning",
        )

        if _engine_result_is_real():
            st.write("• The Strategy Engine generated a historical strategy return path from the selected universe and engine configuration.")
        else:
            st.write("• The Scenario Explorer used a labelled educational proxy because no tested strategy return path was available.")
        st.write("• The long-term scenario model translates contribution assumptions into projected ranges.")
        st.write("• These insights interpret plausibility, uncertainty, trade-offs and horizon sensitivity.")
        st.write("• P10 / median / P90 are scenario percentiles under the model assumptions, not guaranteed outcomes.")
        if compare:
            st.write("• The comparison is fair only when both branches use the same contribution, starting pot, goal and horizons.")


def _render_horizon_rows(rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows or [])
    if not rows:
        return

    display = pd.DataFrame(rows)
    if display.empty:
        return

    money_cols = [
        "Total saved",
        "Conservative",
        "Expected",
        "High case",
        "Savings-only",
        "Investing median",
        "Investing P10",
        "Investing P90",
        "Difference",
        "Goal gap",
    ]
    for col in money_cols:
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="coerce").apply(
                lambda x: "—" if pd.isna(x) else _fmt_gbp0(float(x))
            )
    if "_years" in display.columns:
        display = display.drop(columns=["_years"])
    st.dataframe(display, use_container_width=True, hide_index=True)


def _render_savings_only_insights(proj: Dict[str, Any]) -> None:
    if not _has_projection(proj):
        _render_missing_step6_result(
            "Open the Scenario Explorer and generate the savings-only scenario first. This page will then interpret the long-term savings range.",
            button_key="step7_savings_missing_back",
        )
        return

    _render_report_summary(mode_label="savings_only", proj=proj)
    _render_projection_summary(proj, title="Savings-only pathway readout")

    monthly = _metric(proj, "monthly_contribution")
    weekly = _metric(proj, "weekly_contribution")
    goal = _metric(proj, "goal_amount")
    terminal = _metric(proj, "median_terminal", "expected_terminal", "final_value")
    p10 = _metric(proj, "p10_terminal", default=terminal)
    p90 = _metric(proj, "p90_terminal", default=terminal)
    contribution_source = str(proj.get("contribution_source", "planned saving capacity") or "planned saving capacity")

    insight_card(
        "Decision readout",
        "This path asks how far disciplined saving alone can take the user without investment growth, volatility, drawdowns or market risk.",
        level="info",
    )

    if monthly > 0:
        st.write(
            f"The savings-only path is based on about **{_fmt_gbp0(monthly)}/month** "
            f"(**{_fmt_gbp0(weekly)}/week**) from **{contribution_source}**."
        )
    st.write(f"The central long-term outcome is **{_fmt_gbp0(terminal)}**, with an illustrative range of **{_fmt_gbp0(p10)}–{_fmt_gbp0(p90)}**.")

    if goal > 0:
        gap = terminal - goal
        if terminal >= goal:
            insight_card("Goal feasibility", "Savings alone reaches the stated goal in the central scenario.", level="success")
        elif p90 >= goal:
            insight_card(
                "Goal feasibility",
                "Savings alone does not reach the goal in the central scenario, but the high-case saving path could reach it.",
                level="info",
            )
        else:
            insight_card(
                "Goal feasibility",
                f"Savings alone remains about **{_fmt_gbp0(abs(gap))}** below the goal in the central scenario.",
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
        _render_horizon_rows(horizon_rows)

    st.markdown("### What this means")
    st.write("• If savings alone is enough, the lower-risk path may already be sufficient for the goal.")
    st.write("• If savings alone is close, contribution size and horizon may matter more than adding investment complexity.")
    st.write("• If savings alone is far short, the comparison branch can test whether accepting investment uncertainty changes the gap.")
    st.caption("Savings-only insights do not include investment return, volatility, drawdown or market-risk assumptions.")

    _render_recommended_next_actions(mode_label="savings_only")
    _render_decision_support_note(compare=False)


def _render_strategy_interpretation(engine: Dict[str, Any]) -> None:
    if not _engine_result_is_real():
        st.markdown("### Proxy interpretation")
        insight_card(
            "Illustrative investment scenario",
            "This proxy demonstrates the long-term scenario interface before a tested strategy return path exists. It should not be read as evidence that the selected universe or engine configuration performed well.",
            level="info",
        )
        return

    sharpe = _safe_float(engine.get("sharpe", 0.0))
    cagr = _safe_float(engine.get("cagr", 0.0))
    max_dd = _safe_float(engine.get("max_drawdown", 0.0))
    dd_abs = abs(max_dd)

    st.markdown("### Strategy interpretation")
    if sharpe >= 1.0:
        insight_card(
            "Efficiency",
            f"Sharpe {sharpe:.2f} suggests strong risk-adjusted efficiency on the tested historical panel.",
            level="success",
        )
    elif sharpe >= 0.7:
        insight_card(
            "Efficiency",
            f"Sharpe {sharpe:.2f} suggests reasonable risk-adjusted efficiency, with room to improve before calling the pathway highly efficient.",
            level="info",
        )
    else:
        insight_card(
            "Efficiency",
            f"Sharpe {sharpe:.2f} suggests weak risk-adjusted efficiency on this run.",
            level="warning",
        )

    if cagr >= 0.10:
        insight_card(
            "Growth",
            f"CAGR around {_fmt_pct_from_fraction(cagr)} indicates a growth-oriented historical path, assuming the user accepts the associated volatility and drawdowns.",
            level="success",
        )
    elif cagr >= 0.07:
        insight_card(
            "Growth",
            f"CAGR around {_fmt_pct_from_fraction(cagr)} indicates a growth-positive but still balanced historical return profile.",
            level="info",
        )
    else:
        insight_card(
            "Growth",
            f"CAGR around {_fmt_pct_from_fraction(cagr)} suggests a lower-return or more defensive historical profile.",
            level="warning",
        )

    if dd_abs <= 0.12:
        insight_card(
            "Risk",
            f"Drawdown severity near {_fmt_pct_from_fraction(dd_abs)} was relatively contained in the historical test.",
            level="success",
        )
    elif dd_abs <= 0.20:
        insight_card(
            "Risk",
            f"Drawdown severity near {_fmt_pct_from_fraction(dd_abs)} is material but may be tolerable for users comfortable with temporary losses.",
            level="info",
        )
    else:
        insight_card(
            "Risk",
            f"Drawdown severity near {_fmt_pct_from_fraction(dd_abs)} is high and could be psychologically difficult during bad periods.",
            level="warning",
        )


def _render_investing_insights(proj: Dict[str, Any], cash_proj: Dict[str, Any], engine: Dict[str, Any], *, compare: bool) -> None:
    if not _has_projection(proj):
        _render_missing_step6_result(
            "Open the Scenario Explorer and run the investment/proxy projection first. This page will then interpret the long-term investment pathway.",
            button_key="step7_investment_missing_back",
        )
        return

    _render_proxy_evidence_note(compare=compare)
    _render_report_summary(mode_label="investment", proj=proj, cash_proj=cash_proj, engine=engine, compare=compare)
    if _engine_result_is_real():
        _render_engine_metrics(engine)
    readout_title = "Investment scenario readout" if _engine_result_is_real() else "Investment proxy readout"
    _render_projection_summary(proj, title=readout_title)
    _render_strategy_interpretation(engine)

    if compare and _has_projection(cash_proj):
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

        if abs(diff) > max(10000.0, 5.0 * max(abs(inv_terminal), 1.0)):
            st.warning(
                "This comparison shows a very large gap between branches. Check that both branches use comparable horizons, contributions, starting pots and goal assumptions."
            )

        if diff > 0:
            insight_card(
                "Decision trade-off",
                "The investing scenario shows a higher central terminal value, but that upside comes with uncertainty and potential drawdowns.",
                level="info",
            )
        else:
            insight_card(
                "Decision trade-off",
                "The investing scenario does not clearly beat the savings-only path here, so the extra uncertainty may be harder to justify under these assumptions.",
                level="warning",
            )

    st.markdown("### Where this pathway could improve")
    if _engine_result_is_real():
        engine = _coerce_mapping(engine)
        sharpe = _safe_float(engine.get("sharpe", 0.0))
        cagr = _safe_float(engine.get("cagr", 0.0))
        max_dd = _safe_float(engine.get("max_drawdown", 0.0))
        if sharpe < 0.9:
            st.write("• Improve risk-adjusted efficiency if Sharpe remains below the desired threshold.")
        if abs(max_dd) > 0.15:
            st.write(f"• Reduce drawdowns if a fall near {_fmt_pct_from_fraction(abs(max_dd))} would be emotionally or financially hard to tolerate.")
        if cagr < 0.10:
            st.write("• Increase growth exposure only if the user accepts potentially higher volatility and deeper drawdowns.")
    else:
        st.write("• Run the Investment Strategy Lab to replace the proxy with tested strategy returns.")
        st.write("• Use the proxy only to demonstrate the long-term projection interface and compare pathway structure.")
    st.write("• Treat improvements as scenario comparisons, not guaranteed better real-money outcomes.")

    _render_recommended_next_actions(mode_label="investment", compare=compare)
    _render_decision_support_note(compare=compare)


def _render_compare_branch_insights(engine: Dict[str, Any]) -> None:
    compare_payload = _coerce_mapping(st.session_state.get(COMPARE_BRANCH_RESULT, {}))
    rows = compare_payload.get("rows", [])
    proj = _extract_projection_summary(INVESTMENT_PROJECTION_RESULT)
    cash_proj = _extract_projection_summary(CASH_ONLY_PROJECTION_RESULT)

    if not rows:
        # Graceful fallback: if a user generated separate investment and savings
        # projections but did not run the fair comparison table, still show the
        # core comparison. Otherwise send them back to Step 6, not Step 5.
        if _has_projection(proj) and _has_projection(cash_proj):
            _render_investing_insights(proj, cash_proj, engine, compare=True)
            return
        _render_missing_step6_result(
            "Run the comparison view in the Scenario Explorer first. It can use either the educational proxy or tested Strategy Engine returns.",
            button_key="step7_compare_missing_back",
        )
        return

    compare_df = pd.DataFrame(rows)
    valid = compare_df.dropna(subset=["Savings-only", "Investing median"], how="any") if not compare_df.empty else pd.DataFrame()

    _render_proxy_evidence_note(compare=True)
    _render_report_summary(mode_label="comparison", proj=proj, cash_proj=cash_proj, engine=engine, compare=True)
    if _engine_result_is_real():
        _render_engine_metrics(engine)

    st.markdown("### Savings vs investment comparison")
    if valid.empty:
        st.warning("No complete comparison rows were found. Re-run the Scenario Explorer comparison.")
        return

    valid["Difference"] = pd.to_numeric(valid["Difference"], errors="coerce")
    last_row = valid.sort_values("_years").iloc[-1] if "_years" in valid.columns else valid.iloc[-1]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Longest savings-only", _fmt_gbp0(last_row.get("Savings-only", 0.0)))
    with c2:
        st.metric("Longest investing median", _fmt_gbp0(last_row.get("Investing median", 0.0)))
    with c3:
        st.metric("Median difference", _fmt_gbp0(last_row.get("Difference", 0.0)))

    savings_terminal = _safe_float(last_row.get("Savings-only", 0.0), 0.0)
    investing_terminal = _safe_float(last_row.get("Investing median", 0.0), 0.0)
    diff_terminal = _safe_float(last_row.get("Difference", 0.0), 0.0)
    if abs(diff_terminal) > max(10000.0, 5.0 * max(abs(investing_terminal), 1.0)):
        st.warning(
            "This comparison shows a very large gap between branches. Check that both branches use comparable horizons, contributions, starting pots and goal assumptions."
        )

    positive_rows = valid[pd.to_numeric(valid["Difference"], errors="coerce") > 0]
    if len(positive_rows) == len(valid):
        insight_card(
            "Comparison readout",
            "Across the tested horizons, the investment pathway has a higher median outcome than savings-only. The key question is whether the extra upside is worth accepting uncertainty, model risk and drawdown exposure.",
            level="info",
        )
    elif len(positive_rows) > 0:
        first_positive = positive_rows.sort_values("_years").iloc[0]
        insight_card(
            "Comparison readout",
            f"Investment does not dominate equally at every horizon. It first shows a positive median difference around **{first_positive.get('Horizon', 'one of the tested horizons')}**.",
            level="info",
        )
    else:
        insight_card(
            "Comparison readout",
            "In this run, investment does not clearly beat savings-only on median outcome across the tested horizons. That makes the extra uncertainty harder to justify under these assumptions.",
            level="warning",
        )

    st.markdown("### Horizon interpretation")
    _render_horizon_rows(valid.sort_values("_years").to_dict("records") if "_years" in valid.columns else valid.to_dict("records"))

    if _has_projection(proj):
        _render_projection_summary(proj, title="Investment scenario readout" if _engine_result_is_real() else "Investment proxy readout")
    if _has_projection(cash_proj):
        _render_projection_summary(cash_proj, title="Savings-only pathway readout")

    st.markdown("### What this means")
    st.write("• Savings-only is simpler and avoids market volatility, but has limited upside.")
    st.write("• Investment may improve the median outcome, but introduces uncertainty, drawdown risk and dependence on model assumptions.")
    if not _engine_result_is_real():
        st.write("• Because this is currently proxy-based, it demonstrates the concept; the Investment Strategy Lab provides the stronger tested version.")
    st.write("• This is a decision-support comparison, not a recommendation to invest or a promise of future performance.")

    _render_recommended_next_actions(mode_label="comparison", compare=True)
    _render_decision_support_note(compare=True)


def _render_footer(*, complete_text: str) -> None:
    st.markdown("---")
    st.success(complete_text)
    st.caption("You can return to the Scenario Explorer to change assumptions, or go back Home to choose another module.")
    left, right = st.columns(2)
    with left:
        if st.button("← Back to Scenario Explorer", key=f"step7_back_to_step6_{complete_text.lower().replace(' ', '_')}"):
            st.session_state[CURRENT_STEP] = 6
            st.session_state["current_step"] = 6
            st.rerun()
    with right:
        if st.button("Return to Home", key=f"step7_return_home_{complete_text.lower().replace(' ', '_')}", use_container_width=True):
            st.session_state[CURRENT_STEP] = 0
            st.session_state["current_step"] = 0
            st.rerun()


def render_step_7() -> None:
    mode = _current_step7_mode()

    if mode == "savings_only":
        section_header(
            "Savings-Only Pathway Report",
            "Final readout for the contribution-only route: simple, lower-risk and assumption-light."
        )
        proj = _extract_projection_summary(CASH_ONLY_PROJECTION_RESULT)
        if not _has_projection(proj):
            proj = _extract_projection_summary(INVESTMENT_PROJECTION_RESULT)
        _render_savings_only_insights(proj)
        _render_footer(complete_text="Planning path completed.")
        return

    if mode == "compare_both":
        title = "Savings vs Investment Scenario Insights"
        section_header(
            title,
            "Final comparison of the savings-only route against the investment/proxy route under the selected assumptions."
        )
        engine = _extract_engine_summary()
        _render_compare_branch_insights(engine)
        _render_footer(complete_text="Comparison path completed.")
        return

    title = "Investment Pathway Report"
    if not _engine_result_is_real():
        title = "Investment Proxy Report"
    section_header(
        title,
        "Final readout for the investment scenario, using tested Strategy Engine returns when available or a clearly labelled proxy otherwise."
    )
    engine = _extract_engine_summary()
    proj = _extract_projection_summary(INVESTMENT_PROJECTION_RESULT)
    cash_proj = _extract_projection_summary(CASH_ONLY_PROJECTION_RESULT)
    _render_investing_insights(proj, cash_proj, engine, compare=False)
    _render_footer(complete_text="Scenario path completed.")
