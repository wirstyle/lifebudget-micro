"""
Step 7 — Interpretation & Insights

Proxy-aware final-polish version:
- Savings-only insights work without Step 5.
- Investment / comparison insights can interpret either:
  1. tested Step 5 engine results, or
  2. the clearly labelled Step 6 educational proxy.
- Step 7 no longer blocks the Long-Term Scenario Explorer just because
  ENGINE_HAS_RUN is False; it only asks the user to return to Step 6 when the
  relevant Step 6 projection has not been generated yet.
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
    """Prefer the actual Step 6 tab/mode when available.

    The Home module may set the broad pathway to compare_both, but the user can
    still select Savings only / Investment proxy / Compare in Step 6. Step 7
    should interpret the view the user actually generated.
    """
    mode = str(st.session_state.get(STEP6_VIEW_MODE_KEY, "") or "").strip()
    if mode in {"compare_both", "savings_only", "savings_plus_investing"}:
        return mode
    return _current_pathway()


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
    run_result = _coerce_mapping(st.session_state.get("step5_run_result", {}))
    return _coerce_mapping(run_result.get("performance_summary", {}))


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
    return "Tested Step 5 engine result" if _engine_result_is_real() else "Educational investment proxy"


def _render_missing_step6_result(message: str, *, button_key: str) -> None:
    st.info(message)
    if st.button("← Back to Scenario Explorer", key=button_key):
        st.session_state[CURRENT_STEP] = 6
        st.session_state["current_step"] = 6
        st.rerun()


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
        st.metric("Low / high range", f"{_fmt_gbp0(p10)} · {_fmt_gbp0(p90)}")
    with c4:
        st.metric("Above contributions", _fmt_gbp0(growth))


def _render_engine_metrics(engine: Dict[str, Any]) -> tuple[float, float, float]:
    st.markdown("### Engine metrics")
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


def _render_proxy_evidence_note(*, compare: bool = False) -> None:
    if _engine_result_is_real():
        st.success(
            "Evidence level: **tested Step 5 engine result**. Step 7 is interpreting the strategy return path generated by the investment engine."
        )
        return

    if compare:
        st.info(
            "Evidence level: **educational proxy**. This comparison uses savings-only outcomes against a labelled investment proxy. "
            "It is useful for demonstrating the Scenario Explorer, but it is not a tested engine result. Run Step 4 + Step 5 to replace it automatically."
        )
    else:
        st.info(
            "Evidence level: **educational proxy**. This investment pathway was generated without a Step 5 engine run. "
            "Treat it as an illustrative scenario only; running Step 4 + Step 5 replaces it with tested strategy returns."
        )


def _render_decision_support_note(*, compare: bool = False) -> None:
    st.markdown("### How to use these outputs")
    insight_card(
        "Decision-support only",
        "These outputs help compare plausible pathways under model assumptions. They are not financial advice, an investment recommendation, or a promise of future results.",
        level="info",
    )
    insight_card(
        "Past performance warning",
        "Past performance refers to the past and is not a reliable indicator of future results. Scenario ranges should be read as model outputs, not expected real-money outcomes.",
        level="warning",
    )

    if _engine_result_is_real():
        st.write("• **Step 5** generated a historical strategy return path from the selected universe and engine configuration.")
    else:
        st.write("• **Step 6** used a labelled educational proxy because no tested Step 5 return path was available.")
    st.write("• **Step 6** translates contribution assumptions into long-term ranges.")
    st.write("• **Step 7** interprets the result as plausibility, uncertainty, trade-offs and horizon sensitivity.")
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
            "Open Step 6 and generate the savings-only scenario first. Step 7 will then interpret the long-term savings range.",
            button_key="step7_savings_missing_back",
        )
        return

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


def _render_strategy_interpretation(engine: Dict[str, Any]) -> None:
    if not _engine_result_is_real():
        st.markdown("### Proxy interpretation")
        insight_card(
            "Illustrative investment scenario",
            "This is not a real engine-tested strategy. It is a labelled proxy so the Long-Term Scenario Explorer can be demonstrated before Step 5 is run.",
            level="info",
        )
        st.write("• Use this to understand the interface and the type of long-term trade-off the app can show.")
        st.write("• Do not interpret the proxy as evidence that the selected Step 4 universe or Step 5 engine configuration performed well.")
        st.write("• For assessed technical evidence, run Step 4 + Step 5 and return to Step 6/7.")
        return

    sharpe, cagr, max_dd = _render_engine_metrics(engine)
    st.markdown("### Strategy interpretation")
    if sharpe >= 1.0:
        insight_card("Efficiency", "The strategy shows strong risk-adjusted performance on the tested historical panel.", level="success")
    elif sharpe >= 0.7:
        insight_card("Efficiency", "The strategy has reasonable efficiency, but still room to improve.", level="info")
    else:
        insight_card("Efficiency", "The strategy has low risk-adjusted returns on this run.", level="warning")

    if cagr >= 0.10:
        insight_card("Growth", "The strategy is clearly growth-oriented.", level="success")
    elif cagr >= 0.07:
        insight_card("Growth", "The strategy has a balanced return profile.", level="info")
    else:
        insight_card("Growth", "The strategy has a lower-return / more defensive profile.", level="warning")

    if abs(max_dd) <= 0.12:
        insight_card("Risk", "Drawdowns are relatively contained in the historical test.", level="success")
    elif abs(max_dd) <= 0.20:
        insight_card("Risk", "Drawdowns are moderate and may be tolerable for many users.", level="info")
    else:
        insight_card("Risk", "Drawdown risk is high and may be psychologically difficult.", level="warning")


def _render_investing_insights(proj: Dict[str, Any], cash_proj: Dict[str, Any], engine: Dict[str, Any], *, compare: bool) -> None:
    if not _has_projection(proj):
        _render_missing_step6_result(
            "Open Step 6 and run the investment/proxy projection first. Step 7 will then interpret the long-term investment pathway.",
            button_key="step7_investment_missing_back",
        )
        return

    _render_proxy_evidence_note(compare=compare)
    _render_strategy_interpretation(engine)
    _render_projection_summary(proj, title="Investment pathway readout")

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

    _render_decision_support_note(compare=compare)

    st.markdown("### Where to improve")
    if _engine_result_is_real():
        engine = _coerce_mapping(engine)
        sharpe = _safe_float(engine.get("sharpe", 0.0))
        cagr = _safe_float(engine.get("cagr", 0.0))
        max_dd = _safe_float(engine.get("max_drawdown", 0.0))
        if sharpe < 0.9:
            st.write("• Improve efficiency through better diversification or clearer strategy assumptions.")
        if abs(max_dd) > 0.15:
            st.write("• Reduce drawdowns for a smoother user experience.")
        if cagr < 0.10:
            st.write("• Increase growth exposure only if aligned with risk tolerance.")
    else:
        st.write("• Run the Investment Strategy Lab to replace the proxy with tested strategy returns.")
        st.write("• Use the proxy only to demonstrate the long-term projection interface and compare pathway structure.")
    st.write("• Treat the outputs as scenario comparison, not as a guaranteed forecast or expected real-money outcome.")


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
            "Run the comparison view in Step 6 first. It can use either the educational proxy or the tested Step 5 engine returns.",
            button_key="step7_compare_missing_back",
        )
        return

    compare_df = pd.DataFrame(rows)
    valid = compare_df.dropna(subset=["Savings-only", "Investing median"], how="any") if not compare_df.empty else pd.DataFrame()

    _render_proxy_evidence_note(compare=True)
    if _engine_result_is_real():
        _render_engine_metrics(engine)

    st.markdown("### Decision comparison")
    if valid.empty:
        st.warning("No complete comparison rows were found. Re-run the Step 6 comparison.")
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

    positive_rows = valid[pd.to_numeric(valid["Difference"], errors="coerce") > 0]
    if len(positive_rows) == len(valid):
        insight_card(
            "Comparison readout",
            "Across the tested horizons, the investment pathway has a higher median outcome than savings-only. The key question is whether the extra upside is worth accepting uncertainty and drawdown risk.",
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
        _render_projection_summary(proj, title="Investment pathway readout")
    if _has_projection(cash_proj):
        _render_projection_summary(cash_proj, title="Savings-only pathway readout")

    _render_decision_support_note(compare=True)

    st.markdown("### What this means")
    st.write("• Savings-only is simpler and avoids market volatility, but has limited upside.")
    st.write("• Investment may improve the median outcome, but introduces uncertainty and potential drawdowns.")
    if not _engine_result_is_real():
        st.write("• Because this is currently proxy-based, it demonstrates the concept; Step 4 + Step 5 provide the stronger tested version.")
    st.write("• This is a decision-support comparison, not a recommendation to invest or a promise of future performance.")


def _render_footer(*, complete_text: str) -> None:
    st.markdown("---")
    st.success(complete_text)
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
        section_header("Step 7 — Savings-only pathway insights")
        proj = _extract_projection_summary(CASH_ONLY_PROJECTION_RESULT)
        if not _has_projection(proj):
            proj = _extract_projection_summary(INVESTMENT_PROJECTION_RESULT)
        _render_savings_only_insights(proj)
        _render_footer(complete_text="You have completed this planning path.")
        return

    if mode == "compare_both":
        title = "Step 7 — Savings vs investing decision insights"
        if not _engine_result_is_real():
            title = "Step 7 — Savings vs educational proxy insights"
        section_header(title)
        engine = _extract_engine_summary()
        _render_compare_branch_insights(engine)
        _render_footer(complete_text="You have completed the comparison workflow.")
        return

    title = "Step 7 — Investment pathway insights"
    if not _engine_result_is_real():
        title = "Step 7 — Educational investment proxy insights"
    section_header(title)
    engine = _extract_engine_summary()
    proj = _extract_projection_summary(INVESTMENT_PROJECTION_RESULT)
    cash_proj = _extract_projection_summary(CASH_ONLY_PROJECTION_RESULT)
    _render_investing_insights(proj, cash_proj, engine, compare=False)
    _render_footer(complete_text="You have completed this scenario path.")
