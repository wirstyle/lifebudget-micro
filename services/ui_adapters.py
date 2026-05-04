"""
UI adapter helpers for LifeBudget Micro.

This module converts Streamlit session-state objects and service outputs into
small, deterministic dictionaries used by the Step 4, Step 5, Step 6, and final
report workflows.

Some helpers intentionally return empty compatibility payloads for inactive
advanced/recommendation paths. This keeps older saved state and imports safe
without reactivating retired UI flows.
"""

from __future__ import annotations

from typing import Any, Dict, List


PROJECTION_PROFILE_OPTIONS = ["Conservative", "Defensive", "Balanced", "Growth"]

def coerce_mapping(value: Any) -> Dict[str, Any]:
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


def coerce_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _normalise_float_list(value: Any) -> List[float]:
    if value is None:
        return []

    try:
        if hasattr(value, "tolist"):
            value = value.tolist()
    except Exception:
        value = []

    out: List[float] = []
    for item in coerce_list(value):
        try:
            out.append(float(item))
        except Exception:
            continue
    return out


def build_step4_universe_payload(session_state: Any) -> Dict[str, Any]:
    state = coerce_mapping(session_state)

    selected_assets = list(
        state.get("selected_assets")
        or state.get("last_used_universe_assets")
        or state.get("recommended_universe_assets", [])
        or []
    )
    candidate_assets = list(state.get("candidate_assets") or state.get("last_recommendation_candidate_assets", []) or [])
    recommended_assets = list(state.get("recommended_universe_assets", []) or selected_assets)

    return {
        "size": int(state.get("universe_size", 25) or 25),
        "strategy": str(state.get("universe_strategy", "Core multi-asset") or "Core multi-asset"),
        "philosophy": str(state.get("investment_philosophy", state.get("universe_simple_style_preset", "Balanced")) or "Balanced"),
        "custom_enabled": bool(state.get("universe_custom_enabled", False)),
        "custom_text": str(state.get("custom_universe_text", "") or ""),
        "recommended_assets": recommended_assets,
        "selected_assets": selected_assets,
        "candidate_assets": candidate_assets,
        "selection_source": str(state.get("universe_custom_enabled_source", "") or ""),
        "style_preset": str(state.get("universe_simple_style_preset", "Balanced") or "Balanced"),
        "strategy_template": str(state.get("universe_simple_strategy_template", "Balanced Risk-Controlled") or "Balanced Risk-Controlled"),
        "projection_profile": str(state.get("investment_projection_profile_hint", "Balanced") or "Balanced"),
        "simple_mode_enabled": bool(state.get("universe_simple_mode_enabled", True)),
    }


def build_step5_simple_payload(session_state: Any) -> Dict[str, Any]:
    state = coerce_mapping(session_state)

    return {
        "style_preset": str(state.get("universe_simple_style_preset", "Balanced") or "Balanced"),
        "strategy_template": str(state.get("universe_simple_strategy_template", "Balanced Risk-Controlled") or "Balanced Risk-Controlled"),
        "risk_appetite": float(state.get("universe_simple_risk_appetite", 0.5) or 0.5),
        "diversification": float(state.get("universe_simple_diversification", 0.5) or 0.5),
        "stability": float(state.get("universe_simple_stability", 0.5) or 0.5),
        "turnover_pref": float(state.get("universe_simple_turnover_pref", 0.5) or 0.5),
        "drawdown_protection": float(state.get("universe_simple_drawdown_protection", 0.5) or 0.5),
        "overlay_intensity": float(state.get("universe_simple_overlay_intensity", 0.5) or 0.5),
        "signal_confidence": float(state.get("universe_simple_signal_confidence", 0.5) or 0.5),
        "simplicity": float(state.get("universe_simple_simplicity", 0.5) or 0.5),
        "simple_mode_enabled": bool(state.get("universe_simple_mode_enabled", True)),
        "show_advanced": False,
    }


def build_step5_advanced_payload(session_state: Any) -> Dict[str, Any]:
    """Compatibility surface for inactive advanced Step 5 payloads."""
    return {}


def merge_config_layers(simple_cfg: Dict[str, Any], advanced_cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Compatibility helper. Advanced overrides are ignored when absent/empty."""
    payload: Dict[str, Any] = dict(simple_cfg or {})
    if isinstance(advanced_cfg, dict) and advanced_cfg:
        payload.update(advanced_cfg)
    return payload


def build_projection_payload(session_state: Any) -> Dict[str, Any]:
    state = coerce_mapping(session_state)

    return {
        "monthly_contribution": float(state.get("investment_monthly_contribution", 500.0) or 500.0),
        "weekly_equivalent": float(state.get("investment_weekly_equivalent", 0.0) or 0.0),
        "horizon_years": int(state.get("investment_projection_horizon_years", state.get("step6_horizon_years", 20)) or 20),
        "profile": str(state.get("investment_projection_profile", state.get("step6_projection_profile", "Balanced")) or "Balanced"),
    }


def build_projection_runtime_payload(session_state: Any, investment_context: Any = None) -> Dict[str, Any]:
    state = coerce_mapping(session_state)
    context = coerce_mapping(investment_context or state.get("investment_context", {}))

    compare_horizons = []
    for raw in coerce_list(state.get("investment_projection_compare_horizons", state.get("step6_compare_horizons", []))):
        try:
            compare_horizons.append(int(raw))
        except Exception:
            continue

    profile = str(
        state.get(
            "investment_projection_profile",
            context.get("risk_profile", context.get("profile", state.get("investment_projection_profile_hint", "Balanced"))),
        )
        or "Balanced"
    )
    if profile not in PROJECTION_PROFILE_OPTIONS:
        profile = "Balanced"

    current_savings = float(state.get("investment_projection_current_savings", context.get("current_savings", context.get("starting_value", 0.0))) or 0.0)
    monthly_contribution = float(context.get("monthly_contribution", state.get("investment_monthly_contribution", 0.0)) or 0.0)
    weekly_equivalent = float(context.get("weekly_equivalent", state.get("investment_weekly_equivalent", 0.0)) or 0.0)
    horizon_years = int(state.get("investment_projection_horizon_years", context.get("horizon_years", state.get("step6_horizon_years", 20))) or 20)
    goal_amount = float(state.get("investment_projection_goal_amount", context.get("goal_amount", state.get("goal_amount", 0.0))) or 0.0)

    realised_monthly_returns = _normalise_float_list(
        context.get("oos_returns_monthly")
        or context.get("oos_returns_simple")
        or coerce_mapping(state.get("step5_run_result", {})).get("oos_returns_simple")
        or []
    )

    simulation_mode_label = str(state.get("investment_projection_simulation_mode_label", "") or "").strip()
    simulation_granularity = str(state.get("investment_projection_simulation_granularity", "") or "").strip()
    if simulation_granularity not in {"monthly", "daily_hybrid"}:
        simulation_granularity = "daily_hybrid" if simulation_mode_label == "Hybrid daily simulation" else "monthly"

    return {
        "current_savings": current_savings,
        "monthly_contribution": monthly_contribution,
        "weekly_equivalent": weekly_equivalent,
        "horizon_years": horizon_years,
        "goal_amount": goal_amount,
        "profile": profile,
        "risk_profile": profile,
        "n_sims": int(state.get("investment_projection_n_sims", 1000) or 1000),
        "seed": int(state.get("seed", 42) or 42),
        "override_annual_return": state.get("investment_projection_annual_return"),
        "override_annual_vol": state.get("investment_projection_annual_vol"),
        "bootstrap_method": str(state.get("investment_projection_bootstrap_method", "iid") or "iid"),
        "simulation_granularity": simulation_granularity,
        "daily_steps_per_month": int(state.get("investment_projection_daily_steps_per_month", 21) or 21),
        "daily_path_noise_scale": float(state.get("investment_projection_daily_path_noise_scale", 0.35) or 0.35),
        "compare_horizons_enabled": bool(state.get("investment_projection_compare_enabled", False)),
        "compare_horizons": compare_horizons,
        "engine_has_run": bool(state.get("engine_has_run", False)),
        "projection_open": bool(state.get("projection_open", False)),
        "realised_monthly_returns": realised_monthly_returns,
        "historical_path_count": int(len(realised_monthly_returns)),
    }


def build_projection_state_payload(session_state: Any) -> Dict[str, Any]:
    state = coerce_mapping(session_state)
    return {
        "projection_open": bool(state.get("projection_open", False)),
        "engine_has_run": bool(state.get("engine_has_run", False)),
        "projection_result": coerce_mapping(state.get("investment_projection_result", {})),
        "projection_signature": state.get("investment_projection_signature"),
        "projection_compare_results": coerce_mapping(state.get("investment_projection_compare_results", {})),
        "projection_compare_signature": state.get("investment_projection_compare_signature"),
    }


def build_recommendation_state_payload(session_state: Any) -> Dict[str, Any]:
    """Compatibility surface for saved recommendation state from older runs."""
    return {
        "simple_override_patch": {},
        "simple_override_scope_key": "",
        "simple_override_source": "",
        "dynamic_size_options": [],
        "size_optimisation_results": {},
        "size_optimisation_scope": "",
        "start_date_stability_results": {},
        "start_date_stability_signature": "",
    }


def build_current_workflow_snapshot(session_state: Any) -> Dict[str, Any]:
    state = coerce_mapping(session_state)

    return {
        "goal_amount": float(state.get("goal_amount", 0.0) or 0.0),
        "goal_years": int(state.get("goal_years", 0) or 0),
        "goal_priority": str(state.get("goal_priority", "Balanced") or "Balanced"),
        "weekly_savings": float(state.get("weekly_savings", 0.0) or 0.0),
        "universe": build_step4_universe_payload(state),
        "step5_simple": build_step5_simple_payload(state),
        "step5_advanced": {},
        "projection": build_projection_payload(state),
        "projection_runtime": build_projection_runtime_payload(state),
        "projection_state": build_projection_state_payload(state),
        "recommendation_state": build_recommendation_state_payload(state),
        "engine_has_run": bool(state.get("engine_has_run", False)),
        "projection_open": bool(state.get("projection_open", False)),
    }
