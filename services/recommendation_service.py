# services/recommendation_service.py
"""
Gold Stable Freeze compatibility layer.

The old Step 5 improvement/recommendation system has been retired for the FYP
stable demo path. This module intentionally keeps the public function names used
by older imports, but removes preset/universe recommendation logic, candidate
reruns, and acceptance-wizard behaviour.

Current Step 5 flow:
    Run -> Metrics -> Projection -> Insights
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List
import math

import pandas as pd

from .ui_adapters import coerce_mapping


PRESETS = ["Balanced", "Growth", "Defensive"]

PHILOSOPHY_SCORE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Defensive": {"sharpe": 0.60, "cagr": 0.20, "dd": 0.80},
    "Balanced": {"sharpe": 0.50, "cagr": 0.30, "dd": 0.50},
    "Growth": {"sharpe": 0.40, "cagr": 0.70, "dd": 0.30},
}

# Retained only for backward-compatible imports. No active recommendation flow
# consumes these rules in Gold Stable mode.
PHILOSOPHY_ACCEPTANCE_RULES: Dict[str, Dict[str, float]] = {
    "Defensive": {"max_sharpe_drop": 0.03, "max_drawdown_worsening": 0.01, "min_cagr_improvement": 0.0},
    "Balanced": {"max_sharpe_drop": 0.06, "max_drawdown_worsening": 0.03, "min_cagr_improvement": 0.0},
    "Growth": {"max_sharpe_drop": 0.10, "max_drawdown_worsening": 0.08, "min_cagr_improvement": 0.0025},
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return float(default)


def _normalise_philosophy(value: Any, default: str = "Balanced") -> str:
    text = str(value or "").strip().capitalize()
    return text if text in PHILOSOPHY_SCORE_WEIGHTS else default


def _resolve_target_philosophy(target: Any = None) -> str:
    return _normalise_philosophy(target, default="Balanced")


def extract_performance_summary(run_result: Any) -> Dict[str, Any]:
    run_map = coerce_mapping(run_result)
    return coerce_mapping(run_map.get("performance_summary", {}))


def compute_adjusted_score(perf: Dict[str, Any], philosophy: str = "Balanced") -> float:
    """Small, deterministic metric score retained for summaries/reporting only."""
    weights = PHILOSOPHY_SCORE_WEIGHTS.get(
        _resolve_target_philosophy(philosophy),
        PHILOSOPHY_SCORE_WEIGHTS["Balanced"],
    )
    perf = coerce_mapping(perf)
    sharpe = _safe_float(perf.get("sharpe", 0.0))
    cagr = _safe_float(perf.get("cagr", 0.0))
    dd = abs(_safe_float(perf.get("max_drawdown", 0.0)))
    return float((weights["sharpe"] * sharpe) + (weights["cagr"] * cagr) - (weights["dd"] * dd))


def score_preset(perf: Dict[str, Any], target: str) -> float:
    """Backward-compatible alias for the old preset scorer."""
    return compute_adjusted_score(perf, philosophy=target)


# ---------------------------------------------------------------------------
# Retired recommendation APIs
# ---------------------------------------------------------------------------

def recommend_presets(run_result: dict, current_preset: str) -> List[Dict[str, Any]]:
    """Retired in Gold Stable mode. Returns no suggested presets."""
    return []


def build_preset_improvement_table(
    current_run: Any,
    tested_presets: Iterable[dict[str, Any]] | None = None,
    current_preset: str = "Balanced",
) -> pd.DataFrame:
    """Retired in Gold Stable mode. Returns a current-run row only when possible."""
    perf = extract_performance_summary(current_run)
    if not perf:
        return pd.DataFrame()
    return pd.DataFrame([
        {
            "label": "Current setup",
            "family": "current",
            "strategy_template": "",
            "style_preset": str(current_preset or "Balanced"),
            "preset": str(current_preset or "Balanced"),
            "sharpe": round(_safe_float(perf.get("sharpe")), 3),
            "cagr": round(_safe_float(perf.get("cagr")), 4),
            "max_drawdown": round(_safe_float(perf.get("max_drawdown")), 4),
            "delta_sharpe": 0.0,
            "delta_cagr": 0.0,
            "delta_max_drawdown": 0.0,
            "score_delta": 0.0,
            "adjusted_score": round(score_preset(perf, current_preset), 4),
            "accepted": True,
            "reason": "Gold Stable mode: preset recommendations are retired.",
            "source": "current",
        }
    ])


def recommend_universe(current_universe: Dict[str, Any]) -> pd.DataFrame:
    """Retired in Gold Stable mode. Returns an empty table."""
    return pd.DataFrame()


def build_universe_candidate_table(candidate_evaluations: Iterable[dict[str, Any]] | None = None) -> pd.DataFrame:
    """Retired in Gold Stable mode. Returns an empty table."""
    return pd.DataFrame()


def acceptance_gate(score: float, threshold: float = 0.6) -> str:
    """Legacy helper retained for imports; no longer drives UI routing."""
    return "accepted" if _safe_float(score) >= _safe_float(threshold, 0.6) else "rejected"


def build_acceptance_decision(
    current_run: Any,
    candidate_run: Any,
    *,
    philosophy: str = "Balanced",
) -> Dict[str, Any]:
    """Retired candidate acceptance API. Always returns inactive metadata."""
    return {
        "accepted": False,
        "status": "retired",
        "reason": "Gold Stable mode does not run candidate acceptance decisions.",
        "philosophy": _resolve_target_philosophy(philosophy),
    }


def build_recommendation_bundle(
    run_result: Any,
    *,
    current_preset: str = "Balanced",
    current_universe: Dict[str, Any] | None = None,
    candidate_evaluations: Iterable[dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Retired bundle API. Returns stable, empty recommendation containers."""
    return {
        "preset_recommendations": [],
        "preset_table": build_preset_improvement_table(run_result, current_preset=current_preset),
        "universe_recommendations": pd.DataFrame(),
        "candidate_table": pd.DataFrame(),
        "status": "retired",
        "reason": "Gold Stable mode keeps Step 5 as Run -> Metrics -> Projection -> Insights.",
    }
