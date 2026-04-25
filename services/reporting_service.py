# services/reporting_service.py
"""
Reporting service for LifeBudget Micro.

Gold Stable mode keeps reporting focused on the real engine run. Candidate,
recommendation, and focus-comparison helpers are retained as safe compatibility
stubs, but no longer build UI-facing recommendation artefacts.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List
import math

import pandas as pd

from .ui_adapters import coerce_mapping


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return float(default)


def extract_performance_summary(run_result: Any) -> Dict[str, Any]:
    run_map = coerce_mapping(run_result)
    return coerce_mapping(run_map.get("performance_summary", {}))


def build_metrics_payload(run_result: Any) -> Dict[str, Any]:
    perf = extract_performance_summary(run_result)
    if not perf:
        return {}

    return {
        "sharpe": _safe_float(perf.get("sharpe", 0.0)),
        "cagr": _safe_float(perf.get("cagr", 0.0)),
        "max_drawdown": _safe_float(perf.get("max_drawdown", 0.0)),
        "annual_volatility": _safe_float(perf.get("annual_volatility", perf.get("annualized_volatility", 0.0))),
        "universe_size": int(_safe_float(perf.get("universe_size", 0), 0.0)),
        "risk_penalty": _safe_float(perf.get("risk_penalty", 0.0)),
        "concentration": int(_safe_float(perf.get("concentration", 0), 0.0)),
        "source": str(coerce_mapping(run_result).get("source", "unknown")),
    }


def build_summary_text(run_result: Any) -> str:
    metrics = build_metrics_payload(run_result)
    if not metrics:
        return "No run summary available."

    sharpe = float(metrics.get("sharpe", 0.0))
    cagr = float(metrics.get("cagr", 0.0))
    max_dd = abs(float(metrics.get("max_drawdown", 0.0)))

    parts: List[str] = []
    if sharpe >= 1.0:
        parts.append("strong risk-adjusted efficiency")
    elif sharpe >= 0.7:
        parts.append("reasonable efficiency")
    else:
        parts.append("weak efficiency")

    if cagr >= 0.10:
        parts.append("growth-oriented return profile")
    elif cagr >= 0.07:
        parts.append("balanced return profile")
    else:
        parts.append("defensive / lower-return profile")

    if max_dd <= 0.12:
        parts.append("contained drawdowns")
    elif max_dd <= 0.20:
        parts.append("moderate drawdowns")
    else:
        parts.append("material drawdown risk")

    return " · ".join(parts).capitalize() + "."


# ---------------------------------------------------------------------------
# Retired candidate/report comparison APIs
# ---------------------------------------------------------------------------

def build_focus_metrics_table(current_run: Any, candidate_run: Any) -> pd.DataFrame:
    """Retired in Gold Stable mode. Candidate comparisons are no longer shown."""
    return pd.DataFrame()


def build_candidate_table(
    run_result: Any = None,
    *,
    candidate_evaluations: Iterable[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Retired in Gold Stable mode. Candidate leaderboards are no longer shown."""
    return pd.DataFrame()


def build_report_bundle(
    run_result: Any,
    *,
    candidate_evaluations: Iterable[dict[str, Any]] | None = None,
    recommended_run: Any = None,
    comparison_df: pd.DataFrame | None = None,
) -> Dict[str, Any]:
    """Build a lean report bundle for the real engine run only."""
    return {
        "metrics": build_metrics_payload(run_result),
        "candidate_table": pd.DataFrame(),
        "focus_metrics_table": pd.DataFrame(),
        "comparison_df": pd.DataFrame(),
        "summary_text": build_summary_text(run_result),
    }
