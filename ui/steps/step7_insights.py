"""
Investment Pathway Insights.

Proxy-aware final-polish version:
- Savings-only insights work without the Strategy Engine.
- Investment / comparison insights can interpret either:
  1. tested Strategy Engine results, or
  2. a clearly labelled educational investment proxy.
- The page only asks the user to return to the Long-Term Scenario when the
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
try:
    from ui.state.keys import INVESTMENT_PROJECTION_COMPARE_RESULTS
except Exception:  # pragma: no cover - keep Step 7 compatible with older key modules.
    INVESTMENT_PROJECTION_COMPARE_RESULTS = "investment_projection_compare_results"


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
    has_compare_rows = (
        isinstance(compare_rows, list) and len(compare_rows) > 0
    ) or _has_current_horizon_comparison_state()

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


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return int(default)
            return int(float(cleaned.replace(",", "")))
        return int(float(value))
    except Exception:
        return int(default)


def _fmt_gbp0(value: Any) -> str:
    return f"£{_safe_float(value, 0.0):,.0f}"


def _fmt_pct_from_fraction(value: Any, decimals: int = 1) -> str:
    return f"{_safe_float(value, 0.0) * 100.0:.{decimals}f}%"


def _first_non_empty_text(*values: Any) -> str:
    """Return the first useful text value from mixed state/summary fields."""
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in {"none", "nan", "null", "—", "-"}:
            return text
    return ""


def _session_text(*keys: str) -> str:
    """Read the first non-empty text value from Streamlit session state."""
    return _first_non_empty_text(*(st.session_state.get(key) for key in keys))


def _projection_profile_text(proj: Dict[str, Any] | None = None) -> str:
    """Best-effort risk/projection profile label for report copy."""
    proj = _coerce_mapping(proj or {})
    return _first_non_empty_text(
        proj.get("projection_profile"),
        proj.get("risk_profile"),
        proj.get("profile"),
        proj.get("philosophy"),
        st.session_state.get("step6_projection_profile"),
        st.session_state.get("step4_philosophy"),
        st.session_state.get("investment_philosophy"),
        st.session_state.get("selected_philosophy"),
        st.session_state.get("risk_profile"),
        st.session_state.get("mother_philosophy"),
    )


def _set_step(step: int) -> None:
    """Route to an app step using both current and legacy state keys."""
    st.session_state[CURRENT_STEP] = int(step)
    st.session_state["current_step"] = int(step)
    st.rerun()


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



def _has_current_horizon_comparison_state() -> bool:
    """Return True when the newer Step 6 horizon-comparison state exists.

    Older Step 6 versions stored final-report rows in ``step6_compare_branch_result``.
    The current Long-Term Scenario stores the investment/proxy horizon payloads in
    ``INVESTMENT_PROJECTION_COMPARE_RESULTS`` and keeps the savings-only branch in
    ``CASH_ONLY_PROJECTION_RESULT``. Step 7 should accept both contracts.
    """
    compare_results = _coerce_mapping(st.session_state.get(INVESTMENT_PROJECTION_COMPARE_RESULTS, {}))
    investment_summary = _extract_projection_summary(INVESTMENT_PROJECTION_RESULT)
    if compare_results and _has_projection(investment_summary):
        return True
    if compare_results:
        return True
    savings_summary = _extract_projection_summary(CASH_ONLY_PROJECTION_RESULT)
    if _has_projection(investment_summary) and _has_projection(savings_summary):
        return True
    # Current Step 6 may have a valid investment/proxy projection and render the
    # savings-only baseline on the chart/table without persisting a separate cash
    # payload. Treat that as sufficient state; rows can be reconstructed.
    return bool(_has_projection(investment_summary))


def _horizon_years_from_label(value: Any) -> int:
    """Parse labels such as '20 years' or '1 years (current)' safely."""
    import re

    match = re.search(r"(\d+)", str(value or ""))
    if not match:
        return 0
    try:
        return int(match.group(1))
    except Exception:
        return 0


def _safe_money_float(value: Any, default: float = 0.0) -> float:
    """Parse numeric or currency-like values without assuming display format."""
    if isinstance(value, (int, float)):
        return _safe_float(value, default)
    text = str(value or "").strip()
    if not text or text in {"—", "-", "None"}:
        return float(default)
    cleaned = (
        text.replace("£", "")
        .replace(",", "")
        .replace("%", "")
        .strip()
    )
    return _safe_float(cleaned, default)


def _extract_compare_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Support Step 6 compare payloads stored as either raw result or wrapper."""
    payload = _coerce_mapping(payload)
    if not payload:
        return {}
    result = _coerce_mapping(payload.get("result", {}))
    return result if result else payload


def _cash_terminal_by_horizon() -> dict[int, float]:
    """Build a year -> savings-only terminal map from the current cash payload."""
    cash_payload = _extract_projection_payload(CASH_ONLY_PROJECTION_RESULT)
    cash_summary = _extract_projection_summary(cash_payload)
    out: dict[int, float] = {}

    horizon_table = cash_summary.get("horizon_table") or cash_payload.get("horizon_table") or []
    if isinstance(horizon_table, pd.DataFrame):
        horizon_rows = horizon_table.to_dict("records")
    elif isinstance(horizon_table, list):
        horizon_rows = horizon_table
    else:
        horizon_rows = []

    for row in horizon_rows:
        row_map = _coerce_mapping(row)
        years = _safe_int(row_map.get("_years"), 0) or _horizon_years_from_label(
            row_map.get("Horizon") or row_map.get("horizon")
        )
        if years <= 0:
            continue
        value = None
        for key in (
            "Expected",
            "expected",
            "median final wealth",
            "Median terminal",
            "median_terminal",
            "expected_terminal",
            "final_value",
        ):
            if key in row_map and row_map.get(key) is not None:
                value = _safe_money_float(row_map.get(key), 0.0)
                break
        if value is not None and value > 0.0:
            out[int(years)] = float(value)

    cash_horizon = _safe_int(cash_summary.get("horizon_years"), 0)
    cash_terminal = _metric(cash_summary, "median_terminal", "expected_terminal", "final_value", default=0.0)
    if cash_horizon > 0 and cash_terminal > 0.0:
        out.setdefault(int(cash_horizon), float(cash_terminal))

    return out


def _infer_savings_only_terminal_for_horizon(years: Any) -> float:
    """Infer the savings-only baseline from the active Step 6 projection state.

    The current Long-Term Scenario can display the savings-only baseline directly
    in the horizon table/chart without storing a separate
    ``CASH_ONLY_PROJECTION_RESULT`` payload. Final Report therefore needs this
    fallback: same starting pot + same monthly contribution + no investment
    return, volatility or drawdown.
    """
    years_int = _safe_int(years, 0)
    if years_int <= 0:
        return 0.0

    investment_summary = _extract_projection_summary(INVESTMENT_PROJECTION_RESULT)

    starting_value = _metric(
        investment_summary,
        "starting_value",
        default=_safe_float(st.session_state.get("investment_projection_current_savings", 0.0), 0.0),
    )
    if starting_value <= 0.0:
        starting_value = _safe_float(st.session_state.get("investment_projection_current_savings", 0.0), 0.0)

    monthly_contribution = _metric(investment_summary, "monthly_contribution", default=0.0)
    if monthly_contribution <= 0.0:
        total_contributed = _metric(investment_summary, "total_contributed", default=0.0)
        source_horizon = _safe_int(investment_summary.get("horizon_years"), 0)
        if total_contributed > starting_value and source_horizon > 0:
            monthly_contribution = (total_contributed - starting_value) / float(source_horizon * 12)

    if monthly_contribution <= 0.0:
        investment_context = _coerce_mapping(st.session_state.get("investment_context", {}))
        monthly_contribution = _safe_float(investment_context.get("monthly_contribution", 0.0), 0.0)

    if monthly_contribution <= 0.0:
        weekly = _safe_float(st.session_state.get("target_a_weekly", 0.0), 0.0)
        if weekly > 0.0:
            monthly_contribution = weekly * 52.0 / 12.0

    if monthly_contribution <= 0.0:
        return 0.0

    return float(starting_value) + (float(monthly_contribution) * 12.0 * float(years_int))


def _projection_payload_horizon(payload: Dict[str, Any], fallback_key: Any = None) -> int:
    summary = _extract_projection_summary(payload)
    return (
        _safe_int(summary.get("horizon_years"), 0)
        or _safe_int(fallback_key, 0)
        or _horizon_years_from_label(fallback_key)
    )


def _legacy_compare_rows() -> list[dict[str, Any]]:
    compare_payload = _coerce_mapping(st.session_state.get(COMPARE_BRANCH_RESULT, {}))
    rows = compare_payload.get("rows", [])
    if isinstance(rows, pd.DataFrame):
        return rows.to_dict("records")
    if isinstance(rows, list):
        return [dict(row) for row in rows if isinstance(row, dict)]
    return []


def _horizon_compare_rows_from_current_state() -> list[dict[str, Any]]:
    """Translate current Step 6 horizon-comparison payloads into Step 7 rows."""
    cash_by_year = _cash_terminal_by_horizon()
    compare_results = _coerce_mapping(st.session_state.get(INVESTMENT_PROJECTION_COMPARE_RESULTS, {}))

    investment_payloads: list[tuple[int, Dict[str, Any]]] = []
    seen: set[int] = set()

    current_payload = _extract_projection_payload(INVESTMENT_PROJECTION_RESULT)
    current_summary = _extract_projection_summary(current_payload)
    current_years = _projection_payload_horizon(current_payload)
    if current_years > 0 and _has_projection(current_summary):
        investment_payloads.append((int(current_years), current_payload))
        seen.add(int(current_years))

    for key, raw_payload in compare_results.items():
        payload = _extract_compare_payload(_coerce_mapping(raw_payload))
        summary = _extract_projection_summary(payload)
        if not _has_projection(summary):
            continue
        years = _projection_payload_horizon(payload, key)
        if years <= 0 or years in seen:
            continue
        investment_payloads.append((int(years), payload))
        seen.add(int(years))

    rows: list[dict[str, Any]] = []
    for years, payload in sorted(investment_payloads, key=lambda item: item[0]):
        summary = _extract_projection_summary(payload)
        inv_median = _metric(summary, "median_terminal", "expected_terminal", "final_value", default=0.0)
        inv_p10 = _metric(summary, "p10_terminal", "median_terminal", "expected_terminal", "final_value", default=inv_median)
        inv_p90 = _metric(summary, "p90_terminal", "median_terminal", "expected_terminal", "final_value", default=inv_median)
        if inv_median <= 0.0:
            continue

        savings_terminal = cash_by_year.get(int(years))
        if savings_terminal is None:
            # Final fallback: if only a single savings-only projection exists,
            # use it for its matching horizon and leave other rows incomplete.
            cash_summary = _extract_projection_summary(CASH_ONLY_PROJECTION_RESULT)
            cash_horizon = _safe_int(cash_summary.get("horizon_years"), 0)
            if cash_horizon == int(years):
                savings_terminal = _metric(cash_summary, "median_terminal", "expected_terminal", "final_value", default=0.0)

        if savings_terminal is None or savings_terminal <= 0.0:
            savings_terminal = _infer_savings_only_terminal_for_horizon(years)

        if savings_terminal is None or savings_terminal <= 0.0:
            continue

        rows.append(
            {
                "Horizon": f"{int(years)} years" + (" (current)" if int(years) == current_years else ""),
                "Savings-only": float(savings_terminal),
                "Investing median": float(inv_median),
                "Investing P10": float(inv_p10),
                "Investing P90": float(inv_p90),
                "Difference": float(inv_median - savings_terminal),
                "_years": int(years),
            }
        )
    return rows


def _compare_rows_from_available_state() -> list[dict[str, Any]]:
    """Prefer legacy fair-comparison rows, then fall back to current Step 6 state."""
    legacy_rows = _legacy_compare_rows()
    if legacy_rows:
        return legacy_rows
    return _horizon_compare_rows_from_current_state()

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
    if st.button("← Back to Long-Term Scenario", key=button_key):
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
            "Evidence level: **tested Strategy Engine result** interpreted through the long-term Long-Term Scenario."
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
                "Generate both Long-Term Scenario branches to complete the comparison readout."
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
            f"This report interprets a tested Strategy Engine result through the long-term Long-Term Scenario. "
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
            st.write("• Return to the Long-Term Scenario to stress-test the result with a different horizon, contribution level, starting pot or goal.")
            st.write("• Return to the Strategy Engine if the investment branch needs lower drawdown, a different universe, or a cleaner risk profile.")
        else:
            st.write("• Run the Investment Strategy Lab to replace the educational proxy with a tested Strategy Engine result.")
            st.write("• Re-run the Long-Term Scenario after the real strategy result exists so the comparison uses stronger evidence.")
            st.write("• Treat the current comparison as a report-flow demonstration, not as a basis for choosing the investment branch.")
        return

    if mode_label == "savings_only":
        st.write("• Keep the savings-only route if the projected range already meets the goal with acceptable effort and timeline.")
        st.write("• Return to the Long-Term Scenario to test a different contribution, horizon, starting pot or goal amount.")
        st.write("• Explore the investment pathway only if the savings-only route leaves a meaningful gap and market uncertainty is acceptable.")
        return

    if _engine_result_is_real():
        st.write("• Treat this as a candidate pathway only if the drawdown level and uncertainty are acceptable.")
        st.write("• Return to the Strategy Engine if the result needs lower drawdown, a smoother risk profile, or a different asset universe.")
        st.write("• Return to the Long-Term Scenario to test whether the same strategy still works under different contribution and horizon assumptions.")
    else:
        st.write("• Run the Investment Strategy Lab to replace this proxy with tested strategy returns.")
        st.write("• Re-run the Long-Term Scenario after the Strategy Engine has produced a real return path.")
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
            st.write("• The Long-Term Scenario used a labelled educational proxy because no tested strategy return path was available.")
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
            "Open the Long-Term Scenario and generate the savings-only scenario first. This page will then interpret the long-term savings range.",
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
            "Open the Long-Term Scenario and run the investment/proxy projection first. This page will then interpret the long-term investment pathway.",
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


def _render_compare_branch_insights(engine: Dict[str, Any]) -> bool:
    rows = _compare_rows_from_available_state()
    proj_payload = _extract_projection_payload(INVESTMENT_PROJECTION_RESULT)
    proj = _extract_projection_summary(proj_payload)
    cash_proj = _extract_projection_summary(CASH_ONLY_PROJECTION_RESULT)

    # Last-resort bridge for sessions where Step 6 has a valid current
    # investment/proxy projection but did not persist the horizon-comparison
    # rows in the legacy shape expected by Step 7.
    if not rows and _has_projection(proj):
        years = (
            _projection_payload_horizon(proj_payload)
            or _safe_int(st.session_state.get("investment_projection_horizon_years"), 0)
            or 1
        )
        inv_median = _metric(proj, "median_terminal", "expected_terminal", "final_value", default=0.0)
        inv_p10 = _metric(proj, "p10_terminal", "median_terminal", "expected_terminal", "final_value", default=inv_median)
        inv_p90 = _metric(proj, "p90_terminal", "median_terminal", "expected_terminal", "final_value", default=inv_median)
        savings_terminal = _infer_savings_only_terminal_for_horizon(years)
        if inv_median > 0.0 and savings_terminal > 0.0:
            rows = [
                {
                    "Horizon": f"{int(years)} years (current)",
                    "Savings-only": float(savings_terminal),
                    "Investing median": float(inv_median),
                    "Investing P10": float(inv_p10),
                    "Investing P90": float(inv_p90),
                    "Difference": float(inv_median - savings_terminal),
                    "_years": int(years),
                }
            ]

    if not rows:
        _render_missing_step6_result(
            "Generate the comparison view in the Long-Term Scenario first. The Final Report can then summarise your personal finance setup, strategy/proxy pathway, and long-term scenario outputs.",
            button_key="step7_compare_missing_back",
        )
        return False

    compare_df = pd.DataFrame(rows)
    valid = compare_df.dropna(subset=["Savings-only", "Investing median"], how="any") if not compare_df.empty else pd.DataFrame()
    if valid.empty:
        _render_proxy_evidence_note(compare=True)
        st.warning("No complete comparison rows were found. Re-run the Long-Term Scenario comparison.")
        return False

    valid = valid.copy()
    valid["Difference"] = pd.to_numeric(valid["Difference"], errors="coerce")
    if "_years" in valid.columns:
        valid = valid.sort_values("_years")
    last_row = valid.iloc[-1]

    savings_terminal = _safe_float(last_row.get("Savings-only", 0.0), 0.0)
    investing_terminal = _safe_float(last_row.get("Investing median", 0.0), 0.0)
    diff_terminal = _safe_float(last_row.get("Difference", investing_terminal - savings_terminal), 0.0)
    monthly = (
        _metric(proj, "monthly_contribution", default=0.0)
        or _metric(cash_proj, "monthly_contribution", default=0.0)
        or _safe_float(st.session_state.get("investment_projection_monthly_contribution", 0.0), 0.0)
    )
    if monthly <= 0.0:
        investment_context = _coerce_mapping(st.session_state.get("investment_context", {}))
        monthly = _safe_float(investment_context.get("monthly_contribution", 0.0), 0.0)
    if monthly <= 0.0:
        weekly = _safe_float(st.session_state.get("target_a_weekly", 0.0), 0.0)
        monthly = weekly * 52.0 / 12.0 if weekly > 0.0 else 0.0

    evidence = "tested Strategy Engine result" if _engine_result_is_real() else "educational investment proxy"
    horizon_label = str(last_row.get("Horizon", "the longest horizon shown"))
    horizons_text = " + ".join(str(x) for x in valid["Horizon"].astype(str).tolist()) if "Horizon" in valid.columns else "current horizon"

    # Keep the evidence warning once, near the top. Repetition below is avoided.
    _render_proxy_evidence_note(compare=True)
    if _engine_result_is_real():
        _render_engine_metrics(engine)

    st.markdown("### Executive summary")
    longest_years = _safe_int(last_row.get("_years"), 0) or _horizon_years_from_label(horizon_label)
    horizon_prefix = f"{longest_years}y" if longest_years > 0 else "Longest-horizon"

    metric_cols = st.columns(4 if monthly > 0 else 3)
    col_idx = 0
    if monthly > 0:
        with metric_cols[col_idx]:
            st.metric("Monthly contribution", f"{_fmt_gbp0(monthly)}/mo")
        col_idx += 1
    with metric_cols[col_idx]:
        st.metric(f"{horizon_prefix} savings-only", _fmt_gbp0(savings_terminal))
    with metric_cols[col_idx + 1]:
        st.metric(f"{horizon_prefix} investment/proxy", _fmt_gbp0(investing_terminal))
    with metric_cols[col_idx + 2]:
        st.metric(f"{horizon_prefix} difference", _fmt_gbp0(diff_terminal))

    st.markdown("### Setup snapshot")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("**Personal finance plan**")
        if monthly > 0:
            st.caption(f"Contribution capacity used: {_fmt_gbp0(monthly)}/month.")
        else:
            st.caption("Contribution capacity read from the stored personal finance setup.")
    with s2:
        st.markdown("**Investment setup**")
        profile_text = _projection_profile_text(proj)
        setup_bits = [f"Return path: {evidence}."]
        if profile_text:
            setup_bits.append(f"Risk profile: {profile_text}.")
        setup_bits.append(
            "Strategy Engine: tested." if _engine_result_is_real() else "Strategy Engine: not yet tested."
        )
        st.caption(" ".join(setup_bits))
    with s3:
        st.markdown("**Long-term scenario**")
        st.caption(f"Horizon comparison: {horizons_text}.")

    summary_text = (
        f"This final report compares the savings-only baseline against a {evidence}. "
        f"At **{horizon_label}**, the savings-only path reaches **{_fmt_gbp0(savings_terminal)}**, "
        f"while the investment/proxy median reaches **{_fmt_gbp0(investing_terminal)}**, "
        f"a modelled central difference of **{_fmt_gbp0(diff_terminal)}**. "
        "That upside should be read alongside uncertainty, drawdown exposure and model risk."
    )
    insight_card("Executive interpretation", summary_text, level="info")

    if abs(diff_terminal) > max(10000.0, 5.0 * max(abs(investing_terminal), 1.0)):
        st.warning(
            "This comparison shows a very large gap between branches. Check that both branches use comparable horizons, contributions, starting pots and goal assumptions."
        )

    st.markdown("### Final comparison")
    positive_rows = valid[pd.to_numeric(valid["Difference"], errors="coerce") > 0]
    if len(positive_rows) == len(valid):
        comparison_text = (
            "Across the tested horizons, the investment/proxy pathway has a higher median outcome than savings-only. "
            "The key question is whether the extra upside is worth accepting uncertainty, model risk and drawdown exposure."
        )
        comparison_level = "info"
    elif len(positive_rows) > 0:
        first_positive = positive_rows.sort_values("_years").iloc[0] if "_years" in positive_rows.columns else positive_rows.iloc[0]
        comparison_text = (
            f"Investment/proxy does not dominate equally at every horizon. It first shows a positive median difference around "
            f"**{first_positive.get('Horizon', 'one of the tested horizons')}**."
        )
        comparison_level = "info"
    else:
        comparison_text = (
            "In this run, investment/proxy does not clearly beat savings-only on median outcome across the tested horizons. "
            "That makes the extra uncertainty harder to justify under these assumptions."
        )
        comparison_level = "warning"
    insight_card("Comparison readout", comparison_text, level=comparison_level)

    st.markdown("### Horizon interpretation")
    _render_horizon_rows(valid.to_dict("records"))

    st.markdown("### Final interpretation")
    insight_card(
        "Upside vs simplicity",
        "Savings-only is simpler and avoids market volatility, but has limited upside. The investment/proxy path may improve the central outcome, but it depends on return assumptions and introduces uncertainty.",
        level="info",
    )
    insight_card(
        "Risk trade-off",
        "A higher central scenario is not automatically a better decision. It needs to be judged against drawdown exposure, model risk, evidence quality and whether the user could stay invested through bad periods.",
        level="info",
    )
    if _engine_result_is_real():
        next_step_text = (
            "Use the Long-Term Scenario to stress-test the result with different contributions, goals and horizons. "
            "Return to the Strategy Engine only if the investment branch needs lower drawdown, a different universe or a cleaner risk profile."
        )
        next_step_level = "success"
    else:
        next_step_text = (
            "This report currently uses an educational proxy. Run the Investment Strategy Lab to replace it with a tested Strategy Engine result, "
            "then re-run the Long-Term Scenario so the final report uses stronger evidence."
        )
        next_step_level = "warning"
    insight_card("Recommended next step", next_step_text, level=next_step_level)

    with st.expander("Selected-horizon projection details", expanded=False):
        st.caption(
            "Selected-horizon details are kept here so the main report stays focused on the cross-horizon comparison."
        )
        if _has_projection(proj):
            _render_projection_summary(
                proj,
                title="Selected-horizon investment details" if _engine_result_is_real() else "Selected-horizon investment proxy details",
            )
        if _has_projection(cash_proj):
            _render_projection_summary(cash_proj, title="Selected-horizon savings-only details")
        if not _has_projection(proj) and not _has_projection(cash_proj):
            st.caption("No separate selected-horizon projection payload was available; the report above was reconstructed from the comparison rows.")

    _render_decision_support_note(compare=True)
    return True


def _render_footer(*, complete_text: str, show_strategy_lab: bool = False) -> None:
    """Final navigation for the report page.

    ``show_strategy_lab`` is kept for backwards compatibility with existing
    call sites, but the footer intentionally exposes only the two core report
    exits: refine the Long-Term Scenario or return Home.
    """
    st.markdown("---")
    st.success(complete_text)
    st.caption("You can refine the assumptions in Long-Term Scenario, or return Home to choose another module.")

    safe_key = complete_text.lower().replace(" ", "_").replace(".", "")
    left, right = st.columns(2)
    with left:
        if st.button("← Back to Long-Term Scenario", key=f"step7_back_to_step6_{safe_key}", use_container_width=True):
            _set_step(6)
    with right:
        if st.button("Return to Home", key=f"step7_return_home_{safe_key}", use_container_width=True):
            _set_step(0)


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
        section_header(
            "Final Report",
            "Complete summary of your personal finance plan, investment setup, long-term scenario, and decision-support insights."
        )
        engine = _extract_engine_summary()
        report_complete = _render_compare_branch_insights(engine)
        if report_complete:
            _render_footer(complete_text="Final report generated.", show_strategy_lab=True)
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
    _render_footer(complete_text="Scenario report generated.", show_strategy_lab=True)
