# services/projection_service.py

"""
Projection service for LifeBudget Micro.
"""

from __future__ import annotations

from typing import Any, Dict
import json
import math
import pandas as pd

from .ui_adapters import coerce_mapping, coerce_list

try:
    from src.investment import run_investment_projection as _run_investment_projection_core
except Exception:  # pragma: no cover
    _run_investment_projection_core = None


PROFILE_RETURN_MAP = {
    "Conservative": 0.04,
    "Balanced": 0.06,
    "Growth": 0.08,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return float(default)


def _coerce_float_list(value: Any) -> list[float]:
    values = []
    for raw in coerce_list(value):
        try:
            out = float(raw)
            if math.isfinite(out):
                values.append(out)
        except Exception:
            continue
    return values



def _first_present_numeric(mapping: Dict[str, Any], *keys: str) -> float | None:
    data = coerce_mapping(mapping)
    for key in keys:
        if key not in data or data.get(key) is None:
            continue
        try:
            value = float(data.get(key))
            if math.isfinite(value):
                return float(value)
        except Exception:
            continue
    return None


def _build_rich_projection_summary(
    projection_result: Dict[str, Any],
    *,
    runtime: Dict[str, Any],
    fallback_annual_return: float | None = None,
) -> Dict[str, Any]:
    result_map = coerce_mapping(projection_result)
    summary = coerce_mapping(result_map.get("summary", {}))
    projection_df = result_map.get("projection_df")

    current_savings = _safe_float(runtime.get("current_savings", 0.0))
    monthly_contribution = _safe_float(runtime.get("monthly_contribution", 0.0))
    horizon_years = int(runtime.get("horizon_years", 0) or 0)
    goal_amount = _safe_float(runtime.get("goal_amount", 0.0))
    risk_profile = str(runtime.get("profile", "Balanced") or "Balanced")

    expected_terminal = _first_present_numeric(summary, "expected_terminal", "final_value", "expected_final_value")
    median_terminal = _first_present_numeric(summary, "median_terminal", "p50_terminal", "median_final_value")
    p10_terminal = _first_present_numeric(summary, "p10_terminal", "downside_terminal")
    p90_terminal = _first_present_numeric(summary, "p90_terminal", "upside_terminal")
    total_contributed = _first_present_numeric(summary, "total_contributed")
    expected_profit = _first_present_numeric(summary, "expected_profit", "gain_from_growth")
    annual_return_assumption = _first_present_numeric(summary, "annual_return_assumption", "assumed_annual_return")
    annual_vol_assumption = _first_present_numeric(summary, "annual_vol_assumption", "assumed_annual_vol")
    probability_of_reaching_goal = _first_present_numeric(summary, "probability_of_reaching_goal")
    probability_of_loss_vs_contributions = _first_present_numeric(summary, "probability_of_loss_vs_contributions")
    probability_of_finishing_below_initial = _first_present_numeric(summary, "probability_of_finishing_below_initial")
    median_profit = _first_present_numeric(summary, "median_profit")
    p10_profit = _first_present_numeric(summary, "p10_profit")
    p90_profit = _first_present_numeric(summary, "p90_profit")

    if isinstance(projection_df, pd.DataFrame) and not projection_df.empty:
        if expected_terminal is None and "projected_value" in projection_df.columns:
            expected_terminal = _first_present_numeric({"x": projection_df["projected_value"].iloc[-1]}, "x")
        if median_terminal is None and expected_terminal is not None:
            median_terminal = float(expected_terminal)
        if total_contributed is None and "annual_contribution" in projection_df.columns:
            total_contributed = float(pd.to_numeric(projection_df["annual_contribution"], errors="coerce").fillna(0.0).sum())

    if total_contributed is None:
        total_contributed = float(current_savings + (monthly_contribution * 12.0 * max(horizon_years, 0)))

    if expected_terminal is None:
        expected_terminal = float(total_contributed)
    if median_terminal is None:
        median_terminal = float(expected_terminal)
    if p10_terminal is None:
        p10_terminal = float(median_terminal)
    if p90_terminal is None:
        p90_terminal = float(median_terminal)

    if expected_profit is None:
        expected_profit = float(expected_terminal - total_contributed)
    if median_profit is None:
        median_profit = float(median_terminal - total_contributed)
    if p10_profit is None:
        p10_profit = float(p10_terminal - total_contributed)
    if p90_profit is None:
        p90_profit = float(p90_terminal - total_contributed)

    if annual_return_assumption is None:
        annual_return_assumption = (
            float(fallback_annual_return)
            if fallback_annual_return is not None and math.isfinite(float(fallback_annual_return))
            else None
        )

    if goal_amount <= 0.0:
        goal_amount = _first_present_numeric(summary, "goal_amount", "wealth_goal") or 0.0
    if probability_of_reaching_goal is None and goal_amount > 0.0:
        probability_of_reaching_goal = 1.0 if median_terminal >= goal_amount else 0.0
    if probability_of_loss_vs_contributions is None:
        probability_of_loss_vs_contributions = 1.0 if p10_terminal < total_contributed else 0.0
    if probability_of_finishing_below_initial is None:
        probability_of_finishing_below_initial = 1.0 if p10_terminal < current_savings else 0.0

    final_value = _first_present_numeric(summary, "final_value")
    if final_value is None:
        final_value = float(expected_terminal)

    gain_from_growth = _first_present_numeric(summary, "gain_from_growth")
    if gain_from_growth is None:
        gain_from_growth = float(expected_profit)

    if final_value > total_contributed * 1.5:
        interpretation = "compounding_dominant"
    elif final_value > total_contributed:
        interpretation = "compounding_meaningful"
    else:
        interpretation = "contribution_dominant"

    enriched = dict(summary)
    enriched.update(
        {
            "expected_terminal": float(expected_terminal),
            "median_terminal": float(median_terminal),
            "p10_terminal": float(p10_terminal),
            "p90_terminal": float(p90_terminal),
            "expected_profit": float(expected_profit),
            "median_profit": float(median_profit),
            "p10_profit": float(p10_profit),
            "p90_profit": float(p90_profit),
            "probability_of_loss_vs_contributions": (
                float(probability_of_loss_vs_contributions)
                if probability_of_loss_vs_contributions is not None
                else None
            ),
            "probability_of_finishing_below_initial": (
                float(probability_of_finishing_below_initial)
                if probability_of_finishing_below_initial is not None
                else None
            ),
            "probability_of_reaching_goal": (
                float(probability_of_reaching_goal)
                if probability_of_reaching_goal is not None and goal_amount > 0.0
                else None
            ),
            "annual_return_assumption": annual_return_assumption,
            "annual_vol_assumption": annual_vol_assumption,
            "assumed_annual_return": annual_return_assumption,
            "assumed_annual_vol": annual_vol_assumption,
            "horizon_years": int(horizon_years),
            "monthly_contribution": float(monthly_contribution),
            "risk_profile": str(risk_profile),
            "starting_value": float(current_savings),
            "current_savings": float(current_savings),
            "goal_amount": float(goal_amount),
            "wealth_goal": float(goal_amount),
            "total_contributed": float(total_contributed),
            "final_value": float(final_value),
            "gain_from_growth": float(gain_from_growth),
            "interpretation": interpretation,
        }
    )
    return enriched

def resolve_projection_return(profile: str, step5_run_result: Any = None) -> float:
    annual_return = float(PROFILE_RETURN_MAP.get(str(profile or "Balanced"), 0.06))
    perf = coerce_mapping(coerce_mapping(step5_run_result).get("performance_summary", {}))
    cagr = perf.get("cagr")
    if cagr is not None:
        try:
            annual_return = float(cagr)
        except Exception:
            pass
    return float(annual_return)


def build_projection_table(
    monthly_contribution: float,
    annual_return: float,
    horizon_years: int,
    starting_value: float = 0.0,
) -> pd.DataFrame:
    rows = []
    portfolio_value = float(starting_value)
    annual_contribution = float(monthly_contribution) * 12.0

    for year in range(1, int(horizon_years) + 1):
        portfolio_value = (portfolio_value + annual_contribution) * (1.0 + float(annual_return))
        rows.append(
            {
                "year": int(year),
                "annual_contribution": float(annual_contribution),
                "projected_value": float(portfolio_value),
            }
        )

    return pd.DataFrame(rows)


def build_projection_summary(projection_df: pd.DataFrame, annual_return: float) -> Dict[str, Any]:
    if not isinstance(projection_df, pd.DataFrame) or projection_df.empty:
        return {}

    final_value = float(projection_df["projected_value"].iloc[-1])
    total_contributed = float(projection_df["annual_contribution"].sum())
    gain_from_growth = float(final_value - total_contributed)

    if final_value > total_contributed * 1.5:
        interpretation = "compounding_dominant"
    elif final_value > total_contributed:
        interpretation = "compounding_meaningful"
    else:
        interpretation = "contribution_dominant"

    return {
        "final_value": final_value,
        "total_contributed": total_contributed,
        "gain_from_growth": gain_from_growth,
        "assumed_annual_return": float(annual_return),
        "interpretation": interpretation,
    }


def build_projection_signature(payload: Dict[str, Any]) -> str:
    data = {
        "current_savings": _safe_float(payload.get("current_savings", 0.0)),
        "monthly_contribution": _safe_float(payload.get("monthly_contribution", 0.0)),
        "horizon_years": int(payload.get("horizon_years", 0) or 0),
        "goal_amount": _safe_float(payload.get("goal_amount", 0.0)),
        "profile": str(payload.get("profile", "Balanced") or "Balanced"),
        "n_sims": int(payload.get("n_sims", 0) or 0),
        "seed": int(payload.get("seed", 42) or 42),
        "override_annual_return": payload.get("override_annual_return"),
        "override_annual_vol": payload.get("override_annual_vol"),
        "bootstrap_method": str(payload.get("bootstrap_method", "iid") or "iid"),
        "simulation_granularity": str(payload.get("simulation_granularity", "monthly") or "monthly"),
        "daily_steps_per_month": int(payload.get("daily_steps_per_month", 21) or 21),
        "daily_path_noise_scale": payload.get("daily_path_noise_scale"),
        "historical_path_count": len(_coerce_float_list(payload.get("realised_monthly_returns", []))),
    }
    return json.dumps(data, sort_keys=True)


def build_projection_compare_signature(payload: Dict[str, Any]) -> str:
    compare_payload = dict(payload or {})
    compare_payload["compare_horizons"] = sorted(
        int(x) for x in coerce_list(compare_payload.get("compare_horizons", [])) if str(x).strip()
    )
    return build_projection_signature(compare_payload)


def build_projection_bundle(
    monthly_contribution: float,
    horizon_years: int,
    profile: str,
    step5_run_result: Any = None,
    starting_value: float = 0.0,
) -> Dict[str, Any]:
    annual_return = resolve_projection_return(profile, step5_run_result)
    projection_df = build_projection_table(
        monthly_contribution=monthly_contribution,
        annual_return=annual_return,
        horizon_years=horizon_years,
        starting_value=starting_value,
    )
    summary = build_projection_summary(projection_df, annual_return)
    summary = _build_rich_projection_summary(
        {"projection_df": projection_df, "summary": summary},
        runtime={
            "current_savings": starting_value,
            "monthly_contribution": monthly_contribution,
            "horizon_years": horizon_years,
            "profile": profile,
            "goal_amount": 0.0,
        },
        fallback_annual_return=annual_return,
    )

    return {
        "projection_df": projection_df,
        "summary": summary,
        "source": "simple_projection_bundle",
    }


def run_projection(payload: Dict[str, Any], step5_run_result: Any = None) -> Dict[str, Any]:
    runtime = coerce_mapping(payload)
    monthly_contribution = _safe_float(runtime.get("monthly_contribution", 0.0))
    horizon_years = int(runtime.get("horizon_years", 0) or 0)
    profile = str(runtime.get("profile", "Balanced") or "Balanced")

    if monthly_contribution <= 0.0 or horizon_years <= 0:
        return {
            "result": {},
            "signature": build_projection_signature(runtime),
            "is_current": False,
            "source": "inactive_projection",
        }

    if _run_investment_projection_core is not None:
        result = _run_investment_projection_core(
            current_savings=_safe_float(runtime.get("current_savings", 0.0)),
            invest_fraction=1.0,
            monthly_contribution=monthly_contribution,
            horizon_years=horizon_years,
            risk_profile=profile,
            n_sims=int(runtime.get("n_sims", 1000) or 1000),
            seed=int(runtime.get("seed", 42) or 42),
            override_annual_return=(
                _safe_float(runtime.get("override_annual_return"))
                if runtime.get("override_annual_return") is not None
                else None
            ),
            override_annual_vol=(
                _safe_float(runtime.get("override_annual_vol"))
                if runtime.get("override_annual_vol") is not None
                else None
            ),
            realised_monthly_returns=_coerce_float_list(runtime.get("realised_monthly_returns", [])),
            bootstrap_method=str(runtime.get("bootstrap_method", "iid") or "iid"),
            goal_amount=_safe_float(runtime.get("goal_amount", 0.0)),
            simulation_granularity=str(runtime.get("simulation_granularity", "monthly") or "monthly"),
            daily_steps_per_month=int(runtime.get("daily_steps_per_month", 21) or 21),
            daily_path_noise_scale=_safe_float(runtime.get("daily_path_noise_scale", 0.35), 0.35),
        )
        result_map = coerce_mapping(result)
        result_map["summary"] = _build_rich_projection_summary(
            result_map,
            runtime=runtime,
            fallback_annual_return=(
                _safe_float(runtime.get("override_annual_return"))
                if runtime.get("override_annual_return") is not None
                else resolve_projection_return(profile, step5_run_result)
            ),
        )
        return {
            "result": result_map,
            "signature": build_projection_signature(runtime),
            "is_current": True,
            "source": "core_projection",
        }

    bundle = build_projection_bundle(
        monthly_contribution=monthly_contribution,
        horizon_years=horizon_years,
        profile=profile,
        step5_run_result=step5_run_result,
        starting_value=_safe_float(runtime.get("current_savings", 0.0)),
    )
    bundle["summary"] = _build_rich_projection_summary(
        bundle,
        runtime=runtime,
        fallback_annual_return=resolve_projection_return(profile, step5_run_result),
    )
    return {
        "result": bundle,
        "signature": build_projection_signature(runtime),
        "is_current": True,
        "source": "fallback_projection",
    }


def run_projection_compare(payload: Dict[str, Any], step5_run_result: Any = None) -> Dict[str, Any]:
    runtime = coerce_mapping(payload)
    compare_horizons = [
        int(x)
        for x in coerce_list(runtime.get("compare_horizons", []))
        if str(x).strip()
    ]
    if not compare_horizons:
        return {
            "compare_results": {},
            "compare_signature": build_projection_compare_signature(runtime),
            "is_current": False,
        }

    results: Dict[str, Any] = {}
    for horizon in sorted(set(compare_horizons)):
        compare_runtime = dict(runtime)
        compare_runtime["horizon_years"] = int(horizon)
        projection_payload = run_projection(compare_runtime, step5_run_result=step5_run_result)
        results[str(int(horizon))] = projection_payload.get("result", {})

    return {
        "compare_results": results,
        "compare_signature": build_projection_compare_signature(runtime),
        "is_current": True,
    }


def build_projection_workflow_bundle(
    runtime_payload: Dict[str, Any],
    *,
    step5_run_result: Any = None,
    stored_state: Dict[str, Any] | None = None,
    execute_fresh: bool = False,
) -> Dict[str, Any]:
    """
    High-level Step 6 service bundle.

    By default this only computes freshness/signature context and inspects stored
    state. Set execute_fresh=True when the UI button explicitly requests a run.
    """
    runtime = coerce_mapping(runtime_payload)
    state = coerce_mapping(stored_state)

    projection_signature = build_projection_signature(runtime)
    compare_signature = build_projection_compare_signature(runtime)

    stored_result = coerce_mapping(state.get("projection_result", {}))
    stored_result_is_current = bool(stored_result) and str(state.get("projection_signature") or "") == projection_signature

    stored_compare_results = coerce_mapping(state.get("projection_compare_results", {}))
    stored_compare_is_current = bool(stored_compare_results) and str(state.get("projection_compare_signature") or "") == compare_signature

    fresh_projection = {}
    fresh_compare = {}
    if execute_fresh:
        fresh_projection = run_projection(runtime, step5_run_result=step5_run_result)
        fresh_compare = run_projection_compare(runtime, step5_run_result=step5_run_result)

    return {
        "projection_signature": projection_signature,
        "projection_compare_signature": compare_signature,
        "stored_projection_result": stored_result,
        "stored_projection_is_current": stored_result_is_current,
        "stored_projection_compare_results": stored_compare_results,
        "stored_projection_compare_is_current": stored_compare_is_current,
        "fresh_projection": fresh_projection,
        "fresh_compare": fresh_compare,
    }
