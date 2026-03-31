# app.py (FULL) — LifeBudget Micro
# ------------------------------------------------------------
# UX Redesign (Flow B): goal-first, progressive disclosure, auto-baseline,
# advanced controls hidden, Plan A default + Plan B optional, outputs-first.
#
# Core idea:
# - Users want an assistant, not a simulator.
# - Keep all heavy logic modules intact; restructure the UI.
#
# Steps:
# 0) Choose your goal (sets mental model)
# 1) Quick inputs (rough numbers first, detail optional)
#    -> Confirm snapshot (locks + auto-generates baseline)
# 2) Choose your savings goal (Plan A; Plan B optional)
#    -> Auto-plan suggestion
# 3) Results: big numbers + simple chart (technical explanation hidden)
# ------------------------------------------------------------

from pathlib import Path
from typing import Any, Dict, List
from math import log1p
from dataclasses import fields

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from src.baseline import generate_baseline, validate_baseline_df

from src.compounder import (
    simulate_scenario_df,
    apply_shock_map_to_df,
)

from src.explain import (
    ExplanationInputs,
    build_explanation,

    SinglePlanExplanationInputs,
    build_single_plan_explanation,
    compute_reflection_metrics,
    build_human_reflection_text,
    build_structural_deficit_tips,
    InvestmentExplanationInputs,
    build_investment_strategy_explanation,
)

from src.expenses import (
    total_weekly_from_items,
    default_fixed_items_rows,
    variable_essentials_weekly_total,
    discretionary_preset_value,
    clean_events,
    events_to_weekly_shock_map,
    weekly_to_monthly,
    weekly_to_yearly,
)

from src.investment import (
    MicroPipelineConfig,
    SimpleUISpec,
    run_micro_investment_pipeline,
    run_investment_projection,
    compare_run_reports,
    config_to_dict,
    config_fingerprint,
    make_pure_cs_baseline_config,
    resolve_simple_ui_to_micro_cfg,
    sanitize_param_space_for_engine,
    build_engine_param_activity_audit,
    build_asset_group_summary,
    build_asset_group_dataframe,
    aggregate_weights_by_asset_group,
    build_run_report,
    SIMPLE_STRATEGY_TEMPLATES,
    SIMPLE_STYLE_PRESETS,
)

from src.market_data import download_yahoo_return_panel, download_yahoo_macro_feature_panel
from src.features import build_monthly_panel_from_daily, build_weekly_panel_from_daily
from src.evaluation import (
    build_sharpe_turnover_frontier,
    choose_sharpe_turnover_candidate,
    compare_probabilistic_overlay_runs,
    build_probabilistic_mode_audit_table,
    auto_select_signal_mode_by_rank_ic,
    run_alpha_sweep,
    build_plateau_region_table,
    summarize_plateau_region,
    choose_plateau_candidate,
    build_cost_model_summary,
    build_cost_drag_timeseries,
    compare_cost_model_runs,
    build_lambdarank_multiloss_summary,
    build_lambdarank_multiloss_timeseries,
    compare_lambdarank_multiloss_runs,
    run_random_tuning,
    run_bayesian_style_tuning,
    run_surrogate_tuning,
    run_optuna_tuning,
    run_nsga2_tuning,
    run_recursive_multiobjective_search,
    run_global_multiobjective_search,
    run_simple_auto_optimize,
    auto_tune_around_config,
    build_local_search_param_space,
    build_local_search_param_space_from_policy,
    build_tuning_trials_table,
    choose_best_tuning_trial,
    choose_multiobjective_solution_by_policy,
)

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(page_title="LifeBudget Micro", layout="centered")

# ------------------------------------------------------------
# Step 6 gating state (NEW)
# ------------------------------------------------------------
if "projection_open" not in st.session_state:
    st.session_state["projection_open"] = False

if "engine_has_run" not in st.session_state:
    st.session_state["engine_has_run"] = False

# ------------------------------------------------------------
# Helpers: period <-> weekly
# ------------------------------------------------------------
PERIOD_TO_WEEK = {
    "Weekly": 1.0,
    "Monthly": 12.0 / 52.0,
    "Yearly": 1.0 / 52.0,
}


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, pd.Series):
        return value.to_dict()
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


def _row_to_dict(row: Any) -> dict[str, Any]:
    return _coerce_mapping(row)


def to_weekly(amount: float, period: str) -> float:
    return float(amount) * float(PERIOD_TO_WEEK[period])


def weekly_to_period(amount_w: float, period: str) -> float:
    if period == "Weekly":
        return float(amount_w)
    if period == "Monthly":
        return float(amount_w) * (52.0 / 12.0)
    if period == "Yearly":
        return float(amount_w) * 52.0
    raise ValueError(f"Unknown period: {period}")


def _build_investment_context_from_snapshot(plan_snapshot: Any) -> dict[str, float | bool]:
    snapshot = _coerce_mapping(plan_snapshot)
    monthly_contribution = float(max(snapshot.get("target_a_monthly", 0.0) or 0.0, 0.0))
    weekly_equivalent = float(max(snapshot.get("target_a_weekly", 0.0) or 0.0, 0.0))
    baseline_monthly = float(max(snapshot.get("baseline_savings_monthly", 0.0) or 0.0, 0.0))
    required_cut_monthly = float(max(snapshot.get("required_cut_a_monthly", 0.0) or 0.0, 0.0))
    return {
        "monthly_contribution": monthly_contribution,
        "weekly_equivalent": weekly_equivalent,
        "baseline_monthly": baseline_monthly,
        "required_cut_monthly": required_cut_monthly,
        "structural_deficit": bool(snapshot.get("structural_deficit", False)),
    }


def _store_investment_context(plan_snapshot: Any) -> dict[str, float | bool]:
    investment_context = _build_investment_context_from_snapshot(plan_snapshot)
    st.session_state["investment_context"] = investment_context
    st.session_state["investment_monthly_contribution"] = float(investment_context.get("monthly_contribution", 0.0) or 0.0)
    st.session_state["investment_weekly_equivalent"] = float(investment_context.get("weekly_equivalent", 0.0) or 0.0)
    return investment_context


def _map_simple_style_to_risk_profile(style_preset: Any, strategy_template: Any = None) -> str:
    style = str(style_preset or "Balanced").strip().lower()
    strategy = str(strategy_template or "").strip().lower()
    if style in {"conservative", "defensive"}:
        return "Conservative"
    if style in {"growth", "research"}:
        return "Growth"
    if "hybrid research" in strategy:
        return "Growth"
    return "Balanced"


def _extract_monthly_returns_for_projection(run_result: Any) -> np.ndarray | None:
    run_map = _coerce_mapping(run_result)
    raw = run_map.get("oos_returns_simple")
    if raw is None:
        return None
    try:
        returns = pd.Series(raw).copy()
    except Exception:
        return None
    if returns.empty:
        return None
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if returns.empty:
        return None
    try:
        returns.index = pd.to_datetime(returns.index)
    except Exception:
        return None
    returns = returns.sort_index()
    if returns.empty:
        return None
    monthly_returns = (1.0 + returns).groupby(returns.index.to_period("M")).prod() - 1.0
    monthly_returns = pd.to_numeric(monthly_returns, errors="coerce").dropna()
    if monthly_returns.empty:
        return None
    return monthly_returns.to_numpy(dtype="float64")


def _coerce_float_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype="float64").reshape(-1)
    except Exception:
        return None
    if arr.size == 0:
        return None
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return arr


def _read_projection_monthly_returns_from_context(investment_context: Any) -> np.ndarray | None:
    ctx = _coerce_mapping(investment_context)
    return _coerce_float_array(ctx.get("oos_returns_monthly"))


def _update_investment_context_from_run(run_result: Any) -> dict[str, Any]:
    ctx = _coerce_mapping(st.session_state.get("investment_context", {}))
    run_map = _coerce_mapping(run_result)

    monthly_returns = _extract_monthly_returns_for_projection(run_map)
    weekly_returns = _coerce_float_array(run_map.get("oos_returns_simple"))
    if monthly_returns is not None:
        ctx["oos_returns_monthly"] = monthly_returns.tolist()
        ctx["oos_returns_monthly_count"] = int(monthly_returns.size)
    else:
        ctx["oos_returns_monthly"] = []
        ctx["oos_returns_monthly_count"] = 0
    if weekly_returns is not None:
        ctx["oos_returns_weekly"] = weekly_returns.tolist()
        ctx["oos_returns_weekly_count"] = int(weekly_returns.size)
    else:
        ctx["oos_returns_weekly"] = []
        ctx["oos_returns_weekly_count"] = 0

    perf_summary = _coerce_mapping(run_map.get("performance_summary", {}))
    if perf_summary:
        ctx["engine_performance_summary"] = perf_summary
    cfg_payload = _coerce_mapping(run_map.get("config_dict", {}))
    if cfg_payload:
        ctx["engine_config"] = cfg_payload
    cfg_fingerprint_value = run_map.get("config_fingerprint")
    if cfg_fingerprint_value is not None:
        ctx["engine_config_fingerprint"] = str(cfg_fingerprint_value)
    ctx["engine_projection_source"] = "historical_engine_oos" if monthly_returns is not None else "parametric_fallback"

    st.session_state["investment_context"] = ctx
    return ctx


# ------------------------------------------------------------
# Universe helpers (foundation for future investment module)
# ------------------------------------------------------------
UNIVERSE_PRESETS = {
    "U12 Core Multi-Asset": ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG", "GLD", "DBC", "VNQ"],
    "Equities Focus": ["SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ"],
    "Defensive Multi-Asset": ["SPY", "TLT", "IEF", "LQD", "GLD", "DBC"],
    "Inflation / Real Assets": ["GLD", "DBC", "VNQ", "EEM", "SPY"],
}

UNIVERSE_SIZE_OPTIONS = [12, 25, 50, 75, 100, 150, 250]
UNIVERSE_STRATEGY_CORE = "Core multi-asset"
UNIVERSE_STRATEGY_DIVERSIFIED = "Diversified global beta"
UNIVERSE_STRATEGY_EQUITY = "Equity heavy"
UNIVERSE_STRATEGY_DEFENSIVE = "Defensive income"
UNIVERSE_STRATEGY_REAL_ASSETS = "Real assets tilt"
UNIVERSE_STRATEGY_QUALITY = "Quality / Dividend equity"
UNIVERSE_STRATEGY_LOW_VOL = "Low volatility / capital preservation"
UNIVERSE_STRATEGY_LONG_HISTORY = "Long-history / projection-friendly"
UNIVERSE_STRATEGY_OPTIONS = [
    UNIVERSE_STRATEGY_CORE,
    UNIVERSE_STRATEGY_DIVERSIFIED,
    UNIVERSE_STRATEGY_EQUITY,
    UNIVERSE_STRATEGY_DEFENSIVE,
    UNIVERSE_STRATEGY_REAL_ASSETS,
    UNIVERSE_STRATEGY_QUALITY,
    UNIVERSE_STRATEGY_LOW_VOL,
    UNIVERSE_STRATEGY_LONG_HISTORY,
]
UNIVERSE_ALLOWED_STRATEGIES_BY_SIZE = {
    12: [
        UNIVERSE_STRATEGY_CORE,
        UNIVERSE_STRATEGY_EQUITY,
        UNIVERSE_STRATEGY_DEFENSIVE,
        UNIVERSE_STRATEGY_REAL_ASSETS,
        UNIVERSE_STRATEGY_QUALITY,
        UNIVERSE_STRATEGY_LOW_VOL,
        UNIVERSE_STRATEGY_LONG_HISTORY,
    ],
    25: [
        UNIVERSE_STRATEGY_CORE,
        UNIVERSE_STRATEGY_DIVERSIFIED,
        UNIVERSE_STRATEGY_EQUITY,
        UNIVERSE_STRATEGY_DEFENSIVE,
        UNIVERSE_STRATEGY_REAL_ASSETS,
        UNIVERSE_STRATEGY_QUALITY,
        UNIVERSE_STRATEGY_LOW_VOL,
        UNIVERSE_STRATEGY_LONG_HISTORY,
    ],
    50: [
        UNIVERSE_STRATEGY_CORE,
        UNIVERSE_STRATEGY_DIVERSIFIED,
        UNIVERSE_STRATEGY_EQUITY,
        UNIVERSE_STRATEGY_DEFENSIVE,
        UNIVERSE_STRATEGY_REAL_ASSETS,
        UNIVERSE_STRATEGY_QUALITY,
        UNIVERSE_STRATEGY_LOW_VOL,
    ],
    75: [
        UNIVERSE_STRATEGY_CORE,
        UNIVERSE_STRATEGY_DIVERSIFIED,
        UNIVERSE_STRATEGY_EQUITY,
        UNIVERSE_STRATEGY_DEFENSIVE,
        UNIVERSE_STRATEGY_REAL_ASSETS,
        UNIVERSE_STRATEGY_QUALITY,
        UNIVERSE_STRATEGY_LOW_VOL,
    ],
    100: [
        UNIVERSE_STRATEGY_CORE,
        UNIVERSE_STRATEGY_DIVERSIFIED,
        UNIVERSE_STRATEGY_EQUITY,
        UNIVERSE_STRATEGY_DEFENSIVE,
        UNIVERSE_STRATEGY_REAL_ASSETS,
        UNIVERSE_STRATEGY_QUALITY,
        UNIVERSE_STRATEGY_LOW_VOL,
    ],
    150: [
        UNIVERSE_STRATEGY_CORE,
        UNIVERSE_STRATEGY_DIVERSIFIED,
        UNIVERSE_STRATEGY_EQUITY,
        UNIVERSE_STRATEGY_DEFENSIVE,
        UNIVERSE_STRATEGY_REAL_ASSETS,
        UNIVERSE_STRATEGY_QUALITY,
        UNIVERSE_STRATEGY_LOW_VOL,
    ],
    250: [
        UNIVERSE_STRATEGY_CORE,
        UNIVERSE_STRATEGY_DIVERSIFIED,
        UNIVERSE_STRATEGY_EQUITY,
        UNIVERSE_STRATEGY_DEFENSIVE,
        UNIVERSE_STRATEGY_REAL_ASSETS,
        UNIVERSE_STRATEGY_QUALITY,
        UNIVERSE_STRATEGY_LOW_VOL,
    ],
}

LARGE_UNIVERSE_TEST_PRESETS = {
    "50 assets — diversified liquid mix": [
        "SPY", "QQQ", "IWM", "DIA", "MDY", "EFA", "EEM", "VGK", "EWJ", "EWU",
        "XLK", "XLF", "XLV", "XLI", "XLP", "XLY", "XLE", "XLB", "XLU", "VNQ",
        "TLT", "IEF", "SHY", "TIP", "LQD", "HYG", "EMB", "BND", "GLD", "SLV",
        "DBC", "USO", "UNG", "DBA", "XLC", "XBI", "SMH", "SOXX", "ARKK", "ICLN",
        "SCHD", "VIG", "MTUM", "QUAL", "USMV", "VEA", "VWO", "ACWI", "AGG", "MUB",
    ],
    "75 assets — broad research basket": [
        "SPY", "QQQ", "IWM", "DIA", "MDY", "RSP", "EFA", "EEM", "VEA", "VWO",
        "ACWI", "VGK", "EWJ", "EWU", "EWG", "EWZ", "INDA", "FXI", "XLK", "XLF",
        "XLV", "XLI", "XLP", "XLY", "XLE", "XLB", "XLU", "XLRE", "XLC", "XHB",
        "XRT", "XBI", "IBB", "SMH", "SOXX", "ITA", "ICLN", "TAN", "LIT", "ARKK",
        "SCHD", "VIG", "DVY", "MTUM", "QUAL", "USMV", "VLUE", "SIZE", "SPLV", "VNQ",
        "REM", "TLT", "IEF", "IEI", "SHY", "TIP", "LQD", "HYG", "JNK", "EMB",
        "BND", "AGG", "MBB", "MUB", "GLD", "SLV", "PPLT", "DBC", "USO", "UNG",
        "DBA", "CPER", "URA", "BITO", "BIL",
    ],
    "100 assets — stress / scale test": [
        "SPY", "QQQ", "IWM", "DIA", "MDY", "RSP", "VOO", "VTI", "SCHX", "VV",
        "EFA", "EEM", "VEA", "VWO", "ACWI", "VXUS", "VGK", "EWJ", "EWU", "EWG",
        "EWQ", "EWZ", "EWC", "EWA", "INDA", "FXI", "KWEB", "XLK", "XLF", "XLV",
        "XLI", "XLP", "XLY", "XLE", "XLB", "XLU", "XLRE", "XLC", "XHB", "XRT",
        "XBI", "IBB", "IHI", "SMH", "SOXX", "ITA", "PPA", "ICLN", "TAN", "LIT",
        "ARKK", "ARKG", "SCHD", "VIG", "DVY", "NOBL", "MTUM", "QUAL", "USMV", "VLUE",
        "SIZE", "SPLV", "SPYG", "SPYV", "VNQ", "REM", "REET", "TLT", "IEF", "IEI",
        "SHY", "TIP", "LQD", "HYG", "JNK", "EMB", "BND", "AGG", "MBB", "MUB",
        "GLD", "SLV", "PPLT", "PALL", "DBC", "USO", "UNG", "DBA", "DBB", "CPER",
        "URA", "WOOD", "BITO", "BIL", "SHV", "VGSH", "SGOV", "CWB", "SJNK", "HYMB",
    ],
    "200 assets — research scale test": [
        "SPY", "QQQ", "IWM", "DIA", "MDY", "RSP", "VOO", "VTI", "SCHX", "VV",
        "IVV", "IJH", "IJR", "VB", "VUG", "VTV", "IWF", "IWD", "SCHG", "SCHV",
        "EFA", "EEM", "VEA", "VWO", "ACWI", "VXUS", "IEFA", "IEMG", "VGK", "EWJ",
        "EWU", "EWG", "EWQ", "EWZ", "EWC", "EWA", "INDA", "FXI", "KWEB", "MCHI",
        "XLK", "XLF", "XLV", "XLI", "XLP", "XLY", "XLE", "XLB", "XLU", "XLRE",
        "XLC", "XHB", "XRT", "XBI", "IBB", "IHI", "SMH", "SOXX", "ITA", "PPA",
        "ICLN", "TAN", "LIT", "URNM", "ARKK", "ARKG", "ARKF", "SCHD", "VIG", "DVY",
        "NOBL", "MTUM", "QUAL", "USMV", "VLUE", "SIZE", "SPLV", "SPYG", "SPYV", "FNDX",
        "VNQ", "REM", "REET", "SCHH", "TLT", "IEF", "IEI", "SHY", "TIP", "STIP",
        "LQD", "HYG", "JNK", "EMB", "BND", "AGG", "MBB", "MUB", "BSV", "BIV",
        "BLV", "VCIT", "VCSH", "SJNK", "HYMB", "CWB", "GLD", "IAU", "SLV", "PPLT",
        "PALL", "DBC", "USO", "UNG", "DBA", "DBB", "CPER", "URA", "WOOD", "CUT",
        "BITO", "BIL", "SHV", "VGSH", "SGOV", "TFLO", "SCHO", "SCHR", "SCHZ", "BWX",
        "IGSB", "FLOT", "USHY", "ANGL", "MINT", "JPST", "VMBS", "GNMA", "CMBS", "RWX",
        "IFGL", "REET", "RWR", "XLRE", "FREL", "VPU", "IDU", "IYT", "XTN", "KRE",
        "KBE", "IYR", "GDX", "GDXJ", "SIL", "COPX", "XME", "PICK", "MXI", "IGE",
        "MOO", "PHO", "FIW", "TIPX", "LTPZ", "EFV", "EFG", "SCZ", "EWY", "EWT",
        "EWH", "EWS", "EWP", "EWI", "EWN", "EZA", "TUR", "ARGT", "GREK", "EPOL",
        "THD", "ECH", "ERUS", "NORW", "ENZL", "AAXJ", "VNM", "FM", "DXJ", "HEWJ",
    ],
    "250 assets — extended research stress test": [
        "SPY", "QQQ", "IWM", "DIA", "MDY", "RSP", "VOO", "VTI", "SCHX", "VV",
        "IVV", "IJH", "IJR", "VB", "VUG", "VTV", "IWF", "IWD", "SCHG", "SCHV",
        "EFA", "EEM", "VEA", "VWO", "ACWI", "VXUS", "IEFA", "IEMG", "VGK", "EWJ",
        "EWU", "EWG", "EWQ", "EWZ", "EWC", "EWA", "INDA", "FXI", "KWEB", "MCHI",
        "ASHR", "EWY", "EWT", "EWH", "EWS", "EWP", "EWI", "EWN", "EZA", "TUR",
        "ARGT", "GREK", "EPOL", "THD", "ECH", "ERUS", "NORW", "ENZL", "AAXJ", "VNM",
        "FM", "DXJ", "HEWJ", "SCZ", "EFV", "EFG", "VSS", "GWX", "PID", "DEM",
        "XLK", "XLF", "XLV", "XLI", "XLP", "XLY", "XLE", "XLB", "XLU", "XLRE",
        "XLC", "XHB", "XRT", "XBI", "IBB", "IHI", "SMH", "SOXX", "ITA", "PPA",
        "IYT", "XTN", "KRE", "KBE", "KIE", "KCE", "IYR", "RWR", "FREL", "SCHH",
        "ICLN", "TAN", "LIT", "URNM", "REMX", "WOOD", "CUT", "MOO", "PHO", "FIW",
        "ARKK", "ARKG", "ARKF", "ARKW", "SCHD", "VIG", "DVY", "NOBL", "MTUM", "QUAL",
        "USMV", "VLUE", "SIZE", "SPLV", "SPYG", "SPYV", "FNDX", "DGRW", "DGRO", "RDVY",
        "VNQ", "REM", "REET", "RWX", "IFGL", "VNQI", "TLT", "IEF", "IEI", "SHY",
        "TIP", "STIP", "LTPZ", "TIPX", "LQD", "HYG", "JNK", "EMB", "BND", "AGG",
        "MBB", "MUB", "BSV", "BIV", "BLV", "VCIT", "VCSH", "SJNK", "HYMB", "CWB",
        "IGSB", "FLOT", "USHY", "ANGL", "MINT", "JPST", "VMBS", "GNMA", "CMBS", "BWX",
        "BIL", "SHV", "VGSH", "SGOV", "TFLO", "SCHO", "SCHR", "SCHZ", "GLD", "IAU",
        "SLV", "PPLT", "PALL", "GDX", "GDXJ", "SIL", "COPX", "XME", "PICK", "MXI",
        "IGE", "DBC", "USO", "UNG", "DBA", "DBB", "CPER", "URA", "KRBN", "GRN",
        "BITO", "BTF", "WGMI", "BLOK", "MJ", "YOLO", "MSOS", "JETS", "AWAY", "HACK",
        "CIBR", "BUG", "BOTZ", "ROBO", "DRIV", "KARS", "IDRV", "IRBO", "SKYY", "CLOU",
        "WCLD", "FDN", "PNQI", "IBUY", "XSD", "PSI", "XOP", "OIH", "XES", "AMLP",
    ],
}




def _unique_preserve_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        ticker = str(item or "").strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        out.append(ticker)
    return out


_DIVERSIFIED_UNIVERSE_MASTER = _unique_preserve_order([
    *UNIVERSE_PRESETS["U12 Core Multi-Asset"],
    *LARGE_UNIVERSE_TEST_PRESETS["50 assets — diversified liquid mix"],
    *LARGE_UNIVERSE_TEST_PRESETS["75 assets — broad research basket"],
    *LARGE_UNIVERSE_TEST_PRESETS["100 assets — stress / scale test"],
    *LARGE_UNIVERSE_TEST_PRESETS["200 assets — research scale test"],
    *LARGE_UNIVERSE_TEST_PRESETS["250 assets — extended research stress test"],
])

_EQUITY_HEAVY_SEED = [
    "SPY", "QQQ", "VTI", "VOO", "IVV", "SCHX", "VV", "IWM", "IJR", "IJH",
    "RSP", "MDY", "EFA", "VEA", "IEFA", "EEM", "VWO", "IEMG", "ACWI", "VXUS",
    "VGK", "EWJ", "EWU", "INDA", "FXI", "KWEB", "XLK", "XLF", "XLV", "XLI",
    "XLP", "XLY", "XLE", "XLB", "XLU", "XLC", "SCHD", "VIG", "MTUM", "QUAL",
    "USMV", "VLUE", "SIZE", "SPYG", "SPYV", "ICLN", "TAN", "SMH", "SOXX", "VNQ",
]

_DEFENSIVE_INCOME_SEED = [
    "TLT", "IEF", "IEI", "SHY", "VGSH", "SGOV", "BIL", "TFLO", "TIP", "STIP",
    "LQD", "VCIT", "VCSH", "BND", "AGG", "SCHZ", "MUB", "VMBS", "MBB", "BWX",
    "SCHD", "VIG", "NOBL", "USMV", "SPLV", "VNQ", "GLD", "IAU", "SPY", "EFA",
]

_REAL_ASSETS_SEED = [
    "GLD", "IAU", "SLV", "PPLT", "PALL", "DBC", "DBA", "DBB", "CPER", "USO",
    "UNG", "URA", "KRBN", "GRN", "GDX", "GDXJ", "SIL", "COPX", "XME", "PICK",
    "MXI", "IGE", "VNQ", "REET", "RWX", "IFGL", "SCHH", "IYR", "XLRE", "REM",
    "ICLN", "TAN", "LIT", "URNM", "WOOD", "CUT", "MOO", "PHO", "FIW", "SPY",
]

_QUALITY_DIVIDEND_SEED = [
    "SCHD", "VIG", "DGRO", "DGRW", "DVY", "NOBL", "QUAL", "USMV", "SPYV", "VLUE",
    "VTV", "IWD", "SCHV", "FNDX", "RDVY", "VOOV", "XLV", "XLP", "XLU", "VNQ",
    "SPY", "EFA", "VEA", "IEFA", "LQD", "IEF", "VGSH", "GLD", "IAU", "USFR",
]

_LOW_VOL_CAPITAL_PRESERVATION_SEED = [
    "SGOV", "BIL", "SHV", "VGSH", "SCHO", "SCHR", "SHY", "IEI", "IEF", "TIP",
    "STIP", "TFLO", "LQD", "VCIT", "VCSH", "AGG", "BND", "SCHZ", "MUB", "VMBS",
    "MBB", "SCHD", "USMV", "SPLV", "VIG", "NOBL", "GLD", "IAU", "VNQ", "SPY",
]

_LONG_HISTORY_PROJECTION_SEED = [
    "SPY", "QQQ", "DIA", "MDY", "IWM", "EFA", "EEM", "VGK", "EWJ", "EWU",
    "TLT", "IEF", "SHY", "TIP", "LQD", "HYG", "BND", "AGG", "GLD", "SLV",
    "DBC", "VNQ", "XLF", "XLK", "XLV", "XLP", "XLY", "XLI", "XLE", "XLB",
]


def _build_strategy_master_universe(strategy_name: str) -> List[str]:
    strategy = str(strategy_name or UNIVERSE_STRATEGY_CORE).strip()
    if strategy == UNIVERSE_STRATEGY_EQUITY:
        seed = _EQUITY_HEAVY_SEED
    elif strategy == UNIVERSE_STRATEGY_DEFENSIVE:
        seed = _DEFENSIVE_INCOME_SEED
    elif strategy == UNIVERSE_STRATEGY_REAL_ASSETS:
        seed = _REAL_ASSETS_SEED
    elif strategy == UNIVERSE_STRATEGY_QUALITY:
        seed = _QUALITY_DIVIDEND_SEED
    elif strategy == UNIVERSE_STRATEGY_LOW_VOL:
        seed = _LOW_VOL_CAPITAL_PRESERVATION_SEED
    elif strategy == UNIVERSE_STRATEGY_LONG_HISTORY:
        seed = _LONG_HISTORY_PROJECTION_SEED
    elif strategy == UNIVERSE_STRATEGY_DIVERSIFIED:
        seed = []
    else:
        seed = UNIVERSE_PRESETS["U12 Core Multi-Asset"]
    return _unique_preserve_order([*seed, *_DIVERSIFIED_UNIVERSE_MASTER])


def _allowed_universe_strategies_for_size(universe_size: Any) -> List[str]:
    try:
        size_value = int(universe_size)
    except Exception:
        size_value = int(UNIVERSE_SIZE_OPTIONS[0])
    allowed = list(UNIVERSE_ALLOWED_STRATEGIES_BY_SIZE.get(size_value, UNIVERSE_STRATEGY_OPTIONS))
    return [x for x in allowed if x in UNIVERSE_STRATEGY_OPTIONS] or list(UNIVERSE_STRATEGY_OPTIONS)


def _build_generated_universe(universe_size: Any, strategy_name: Any) -> List[str]:
    try:
        requested_size = int(universe_size)
    except Exception:
        requested_size = int(UNIVERSE_SIZE_OPTIONS[0])
    requested_size = max(requested_size, 0)
    master = _build_strategy_master_universe(str(strategy_name or UNIVERSE_STRATEGY_CORE))
    return list(master[:requested_size])


def _build_strategy_candidate_pool(universe_size: Any, strategy_name: Any) -> List[str]:
    try:
        requested_size = int(universe_size)
    except Exception:
        requested_size = int(UNIVERSE_SIZE_OPTIONS[0])
    requested_size = max(requested_size, 0)
    master = _build_strategy_master_universe(str(strategy_name or UNIVERSE_STRATEGY_CORE))
    if requested_size <= 0:
        return list(master)
    extra = max(12, requested_size // 2)
    candidate_size = min(len(master), max(requested_size + extra, requested_size * 2))
    return list(master[:candidate_size])


def _safe_zscore_series(values: pd.Series) -> pd.Series:
    xs = pd.to_numeric(values, errors="coerce")
    if xs.empty:
        return pd.Series(np.nan, index=values.index if hasattr(values, "index") else None, dtype="float64")
    mean_value = float(xs.mean())
    std_value = float(xs.std(ddof=0))
    if not np.isfinite(std_value) or std_value <= 1e-12:
        return pd.Series(0.0, index=xs.index, dtype="float64")
    return (xs - mean_value) / std_value


def _build_strategy_recommended_assets(
    panel_df: pd.DataFrame,
    *,
    universe_size: Any,
    strategy_name: Any,
    baseline_assets: List[str] | None = None,
    min_history: int = 12,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "candidate_assets": [],
        "recommended_assets": [],
        "candidate_score_df": pd.DataFrame(),
        "selection_trace_df": pd.DataFrame(),
        "reason": "",
        "pool_used_count": 0,
    }
    if not isinstance(panel_df, pd.DataFrame) or panel_df.empty:
        result["reason"] = "Panel is empty."
        return result

    try:
        target_size = int(universe_size)
    except Exception:
        target_size = 0
    target_size = max(target_size, 0)
    if target_size <= 1:
        result["reason"] = "Universe size too small for recommendation."
        return result

    candidate_assets = _build_strategy_candidate_pool(target_size, str(strategy_name or UNIVERSE_STRATEGY_CORE))
    result["candidate_assets"] = candidate_assets
    result["pool_used_count"] = len(candidate_assets)
    if len(candidate_assets) <= target_size:
        result["reason"] = "No extra candidate assets available beyond the current preset cut."
        return result

    filtered_panel, info = _filter_asset_panel_to_universe(panel_df, candidate_assets)
    used_assets = list(info.get("used_assets", []))
    if len(used_assets) < target_size:
        result["reason"] = "Not enough candidate assets with return history in the available panel."
        return result

    work = filtered_panel.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["asset"] = work["asset"].astype(str).str.upper()
    work["return"] = pd.to_numeric(work["return"], errors="coerce")
    work = work.dropna(subset=["date", "asset", "return"])
    if work.empty:
        result["reason"] = "Candidate panel became empty after cleaning."
        return result

    stats_df = (
        work.groupby("asset", as_index=False)
        .agg(
            obs_count=("return", "count"),
            mean_return=("return", "mean"),
            vol=("return", "std"),
        )
    )
    stats_df["vol"] = pd.to_numeric(stats_df["vol"], errors="coerce").fillna(0.0)
    stats_df = stats_df[stats_df["obs_count"] >= max(6, int(min_history))].copy()
    if len(stats_df) < target_size:
        result["reason"] = "Too few candidate assets meet the minimum history requirement."
        return result

    stats_df["sharpe_like"] = np.where(stats_df["vol"] > 1e-12, stats_df["mean_return"] / stats_df["vol"], 0.0)

    pivot = work.pivot_table(index="date", columns="asset", values="return")
    candidate_cols = [c for c in stats_df["asset"].astype(str).tolist() if c in pivot.columns]
    if len(candidate_cols) < target_size:
        result["reason"] = "Too few candidate assets survive the return pivot step."
        return result
    pivot = pivot[candidate_cols]
    corr = pivot.corr(min_periods=max(6, int(min_history // 2) or 6)).fillna(0.0)

    avg_abs_corr = {}
    for asset in candidate_cols:
        if asset not in corr.index:
            avg_abs_corr[asset] = 0.0
            continue
        row = corr.loc[asset].drop(labels=[asset], errors="ignore").abs()
        avg_abs_corr[asset] = float(row.mean()) if not row.empty else 0.0
    stats_df["avg_abs_corr"] = stats_df["asset"].map(avg_abs_corr).fillna(0.0)
    stats_df["obs_log"] = stats_df["obs_count"].map(lambda x: log1p(float(x or 0.0)))

    try:
        taxonomy_df = build_asset_group_dataframe(stats_df["asset"].tolist())
    except Exception:
        taxonomy_df = pd.DataFrame(columns=["asset", "group", "subgroup"])
    taxonomy_map: dict[str, dict[str, Any]] = {}
    if isinstance(taxonomy_df, pd.DataFrame) and not taxonomy_df.empty:
        tmp = taxonomy_df.copy()
        if "ticker" in tmp.columns and "asset" not in tmp.columns:
            tmp = tmp.rename(columns={"ticker": "asset"})
        if "asset" in tmp.columns:
            for _, row in tmp.iterrows():
                ticker = str(row.get("asset", "") or "").strip().upper()
                if ticker:
                    taxonomy_map[ticker] = {
                        "group": str(row.get("group", "Other / Unclassified") or "Other / Unclassified"),
                        "subgroup": str(row.get("subgroup", "Other / Unclassified") or "Other / Unclassified"),
                    }

    stats_df["group"] = stats_df["asset"].map(lambda x: taxonomy_map.get(str(x).upper(), {}).get("group", "Other / Unclassified"))
    stats_df["subgroup"] = stats_df["asset"].map(lambda x: taxonomy_map.get(str(x).upper(), {}).get("subgroup", "Other / Unclassified"))

    stats_df["base_score"] = (
        0.45 * _safe_zscore_series(stats_df["sharpe_like"]).fillna(0.0)
        + 0.20 * _safe_zscore_series(stats_df["mean_return"]).fillna(0.0)
        - 0.15 * _safe_zscore_series(stats_df["vol"]).fillna(0.0)
        - 0.35 * _safe_zscore_series(stats_df["avg_abs_corr"]).fillna(0.0)
        + 0.10 * _safe_zscore_series(stats_df["obs_log"]).fillna(0.0)
    )

    stats_df = stats_df.sort_values(["base_score", "asset"], ascending=[False, True]).reset_index(drop=True)
    score_lookup = stats_df.set_index("asset")

    selected: list[str] = []
    selected_groups: set[str] = set()
    selected_subgroups: set[str] = set()
    trace_rows: list[dict[str, Any]] = []

    while len(selected) < target_size:
        best_asset = None
        best_score = None
        best_meta: dict[str, Any] = {}
        for asset in score_lookup.index.tolist():
            if asset in selected:
                continue
            row = score_lookup.loc[asset]
            base_score = float(row.get("base_score", 0.0) or 0.0)
            if selected:
                corr_vals = []
                for prev in selected:
                    try:
                        corr_vals.append(abs(float(corr.loc[asset, prev])))
                    except Exception:
                        pass
                mean_corr_to_selected = float(np.mean(corr_vals)) if corr_vals else 0.0
            else:
                mean_corr_to_selected = 0.0
            group_value = str(row.get("group", "Other / Unclassified") or "Other / Unclassified")
            subgroup_value = str(row.get("subgroup", "Other / Unclassified") or "Other / Unclassified")
            group_bonus = 0.12 if group_value not in selected_groups else 0.0
            subgroup_bonus = 0.05 if subgroup_value not in selected_subgroups else 0.0
            final_score = base_score - 0.50 * mean_corr_to_selected + group_bonus + subgroup_bonus
            if best_score is None or final_score > best_score:
                best_asset = asset
                best_score = final_score
                best_meta = {
                    "base_score": base_score,
                    "mean_corr_to_selected": mean_corr_to_selected,
                    "group": group_value,
                    "subgroup": subgroup_value,
                    "final_score": final_score,
                }
        if best_asset is None:
            break
        selected.append(best_asset)
        selected_groups.add(str(best_meta.get("group", "Other / Unclassified")))
        selected_subgroups.add(str(best_meta.get("subgroup", "Other / Unclassified")))
        trace_rows.append({"step": len(selected), "asset": best_asset, **best_meta})

    recommended_assets = list(selected[:target_size])
    result["recommended_assets"] = recommended_assets
    result["candidate_score_df"] = stats_df
    result["selection_trace_df"] = pd.DataFrame(trace_rows)
    if baseline_assets:
        baseline_set = {str(x).strip().upper() for x in baseline_assets if str(x).strip()}
        recommended_set = set(recommended_assets)
        swap_count = len(recommended_set - baseline_set)
        result["swap_count"] = int(swap_count)
    return result


def _summarize_asset_basket_structure(panel_df: pd.DataFrame, assets: List[str]) -> dict[str, Any]:
    filtered_panel, info = _filter_asset_panel_to_universe(panel_df, assets)
    used_assets = list(info.get("used_assets", []))
    summary: dict[str, Any] = {
        "used_assets": used_assets,
        "n_assets": len(used_assets),
        "group_count": 0,
        "subgroup_count": 0,
        "avg_abs_corr": np.nan,
    }
    if not used_assets or filtered_panel.empty:
        return summary
    try:
        taxonomy_df = build_asset_group_dataframe(used_assets)
    except Exception:
        taxonomy_df = pd.DataFrame()
    if isinstance(taxonomy_df, pd.DataFrame) and not taxonomy_df.empty:
        temp = taxonomy_df.copy()
        if "ticker" in temp.columns and "asset" not in temp.columns:
            temp = temp.rename(columns={"ticker": "asset"})
        if "group" in temp.columns:
            summary["group_count"] = int(temp["group"].astype(str).nunique())
        if "subgroup" in temp.columns:
            summary["subgroup_count"] = int(temp["subgroup"].astype(str).nunique())
    pivot = filtered_panel.copy()
    pivot["date"] = pd.to_datetime(pivot["date"], errors="coerce")
    pivot["return"] = pd.to_numeric(pivot["return"], errors="coerce")
    pivot = pivot.dropna(subset=["date", "asset", "return"]).pivot_table(index="date", columns="asset", values="return")
    if isinstance(pivot, pd.DataFrame) and pivot.shape[1] >= 2:
        corr = pivot.corr().abs()
        mask = ~np.eye(corr.shape[0], dtype=bool)
        vals = corr.where(mask).stack().dropna()
        if not vals.empty:
            summary["avg_abs_corr"] = float(vals.mean())
    return summary

# ------------------------------------------------------------
# Universe selector modes (new unified selector)
# ------------------------------------------------------------
PROBABILISTIC_MODE_CORE = [
    "none",
    "historical",
    "historical_by_regime",
    "historical_by_features",
    "quantile_regression",
]

PROBABILISTIC_MODE_EXPERIMENTAL = [
    "parametric_feature_aware",
    "knn_historical",
    "feature_bucketed_historical",
    "hybrid",
]

PROBABILISTIC_MODE_LABELS = {
    "none": "None",
    "historical": "Historical",
    "historical_by_regime": "Historical by regime",
    "historical_by_features": "Historical by features (core)",
    "quantile_regression": "Quantile regression (core)",
    "parametric_feature_aware": "Parametric feature-aware (experimental)",
    "knn_historical": "kNN historical (experimental / research)",
    "feature_bucketed_historical": "Feature-bucketed historical (experimental)",
    "hybrid": "Hybrid (experimental / not fully validated)",
}

PROBABILISTIC_MODE_NOTES = {
    "parametric_feature_aware": "This variant now has a distinct backend implementation, but it remains experimental / research-grade. Validate results with diagnostics before drawing conclusions.",
    "feature_bucketed_historical": "This variant now has a distinct backend implementation, but it remains experimental / research-grade. Validate results with diagnostics before drawing conclusions.",
    "hybrid": "This mode remains experimental. Keep expectations conservative and validate it against overlay-vs-none and diagnostics before drawing conclusions.",
    "knn_historical": "This mode now has a distinct backend implementation, but it remains research-grade / experimental in the current app surface. Treat results as exploratory unless confirmed by diagnostics.",
}


AUTO_SIGNAL_CONTRACTS = [
    {"label": "mu_sigma | base", "signal_mode": "mu_sigma", "multi_loss_enabled": False, "low_signal_fallback_to_ew": False},
    {"label": "mu_sigma | fallback_enabled", "signal_mode": "mu_sigma", "multi_loss_enabled": False, "low_signal_fallback_to_ew": True},
    {"label": "mu_sigma | multi_loss", "signal_mode": "mu_sigma", "multi_loss_enabled": True, "low_signal_fallback_to_ew": False},
    {"label": "mu_sigma | multi_loss + fallback_enabled", "signal_mode": "mu_sigma", "multi_loss_enabled": True, "low_signal_fallback_to_ew": True},
    {"label": "huber_mu | base", "signal_mode": "huber_mu", "multi_loss_enabled": False, "low_signal_fallback_to_ew": False},
    {"label": "huber_mu | multi_loss", "signal_mode": "huber_mu", "multi_loss_enabled": True, "low_signal_fallback_to_ew": False},
    {"label": "lambdarank_like | base", "signal_mode": "lambdarank_like", "multi_loss_enabled": False, "low_signal_fallback_to_ew": False},
    {"label": "lambdarank_like | multi_loss", "signal_mode": "lambdarank_like", "multi_loss_enabled": True, "low_signal_fallback_to_ew": False},
    {"label": "lambdarank_real | base", "signal_mode": "lambdarank_real", "multi_loss_enabled": False, "low_signal_fallback_to_ew": False},
    {"label": "lambdarank_real | multi_loss", "signal_mode": "lambdarank_real", "multi_loss_enabled": True, "low_signal_fallback_to_ew": False},
    {"label": "directional_classifier | base", "signal_mode": "directional_classifier", "multi_loss_enabled": False, "low_signal_fallback_to_ew": False},
    {"label": "directional_classifier | multi_loss", "signal_mode": "directional_classifier", "multi_loss_enabled": True, "low_signal_fallback_to_ew": False},
    {"label": "logistic_loss | base", "signal_mode": "logistic_loss", "multi_loss_enabled": False, "low_signal_fallback_to_ew": False},
    {"label": "logistic_loss | multi_loss", "signal_mode": "logistic_loss", "multi_loss_enabled": True, "low_signal_fallback_to_ew": False},
    {"label": "top_k_classifier | base", "signal_mode": "top_k_classifier", "multi_loss_enabled": False, "low_signal_fallback_to_ew": False},
    {"label": "top_k_classifier | multi_loss", "signal_mode": "top_k_classifier", "multi_loss_enabled": True, "low_signal_fallback_to_ew": False},
    {"label": "quantile_loss | base", "signal_mode": "quantile_loss", "multi_loss_enabled": False, "low_signal_fallback_to_ew": False},
    {"label": "quantile_loss | multi_loss", "signal_mode": "quantile_loss", "multi_loss_enabled": True, "low_signal_fallback_to_ew": False},
]
AUTO_SIGNAL_CONTRACT_LABELS = [str(x["label"]) for x in AUTO_SIGNAL_CONTRACTS]
AUTO_SIGNAL_CONTRACT_LOOKUP = {str(x["label"]): dict(x) for x in AUTO_SIGNAL_CONTRACTS}

PARETO_SELECTION_POLICY_OPTIONS = [
    "balanced_compromise",
    "knee_point",
    "weighted_hypervolume",
    "fixed_composite_score",
]
COMPOSITE_PROFILE_OPTIONS = [
    "balanced",
    "growth",
    "defensive",
]
DEFAULT_COMPOSITE_PROFILE = "balanced"

PROJECTION_PROFILE_OPTIONS = [
    "Conservative",
    "Balanced",
    "Growth",
]


MANUAL_COMPOSITE_WEIGHT_FIELDS = [
    ("sharpe", "Sharpe"),
    ("cagr", "CAGR"),
    ("max_drawdown", "Max drawdown"),
    ("mean_turnover", "Mean turnover"),
    ("diversification", "Diversification"),
    ("stability", "Stability"),
]

MANUAL_COMPOSITE_PROFILE_DEFAULTS = {
    "balanced": {
        "sharpe": 0.35,
        "cagr": 0.25,
        "max_drawdown": -0.20,
        "mean_turnover": -0.10,
        "diversification": 0.10,
        "stability": 0.00,
    },
    "growth": {
        "sharpe": 0.25,
        "cagr": 0.40,
        "max_drawdown": -0.15,
        "mean_turnover": -0.05,
        "diversification": 0.10,
        "stability": 0.05,
    },
    "defensive": {
        "sharpe": 0.25,
        "cagr": 0.15,
        "max_drawdown": -0.30,
        "mean_turnover": -0.10,
        "diversification": 0.10,
        "stability": 0.10,
    },
}


def _get_manual_composite_profile_seed(profile_name: str) -> dict[str, float]:
    profile_key = str(profile_name or DEFAULT_COMPOSITE_PROFILE).strip().lower() or DEFAULT_COMPOSITE_PROFILE
    seed = MANUAL_COMPOSITE_PROFILE_DEFAULTS.get(profile_key) or MANUAL_COMPOSITE_PROFILE_DEFAULTS[DEFAULT_COMPOSITE_PROFILE]
    return {str(k): float(v) for k, v in dict(seed).items()}


def _render_manual_composite_weights_ui(prefix: str, preset_profile: str) -> tuple[bool, dict[str, float] | None]:
    enabled_key = f"{prefix}_manual_weights_enabled"
    enabled = bool(
        st.checkbox(
            "manual weights",
            value=bool(st.session_state.get(enabled_key, False)),
            key=enabled_key,
            help="Override the preset composite profile with explicit metric weights for the fixed composite score chooser.",
        )
    )
    if not enabled:
        return False, None

    profile_key = str(preset_profile or DEFAULT_COMPOSITE_PROFILE).strip().lower() or DEFAULT_COMPOSITE_PROFILE
    seed = _get_manual_composite_profile_seed(profile_key)
    seed_profile_key = f"{prefix}_manual_weights_seed_profile"
    seeded_once_key = f"{prefix}_manual_weights_seeded_once"
    custom_seed_key = f"{prefix}_manual_weights_custom_seed_profile"
    if not bool(st.session_state.get(seeded_once_key, False)):
        for metric_name, _ in MANUAL_COMPOSITE_WEIGHT_FIELDS:
            widget_key = f"{prefix}_manual_weight_{metric_name}"
            if widget_key not in st.session_state:
                st.session_state[widget_key] = float(seed.get(metric_name, 0.0))
        st.session_state[seed_profile_key] = str(profile_key)
        st.session_state[custom_seed_key] = str(profile_key)
        st.session_state[seeded_once_key] = True

    st.caption("Manual weights override the preset profile for this chooser. Zero-valued fields are treated as omitted, and non-zero active weights are renormalized by the backend.")

    tool_cols = st.columns([1.2, 1.0, 1.0, 1.0])
    with tool_cols[0]:
        manual_seed_profile = st.selectbox(
            "manual weight seed",
            COMPOSITE_PROFILE_OPTIONS,
            index=COMPOSITE_PROFILE_OPTIONS.index(str(st.session_state.get(custom_seed_key, profile_key)).strip().lower()) if str(st.session_state.get(custom_seed_key, profile_key)).strip().lower() in COMPOSITE_PROFILE_OPTIONS else COMPOSITE_PROFILE_OPTIONS.index(DEFAULT_COMPOSITE_PROFILE),
            key=custom_seed_key,
            help="Pick a preset template to prefill the manual weights editor.",
        )
    with tool_cols[1]:
        apply_seed_clicked = st.button("Apply seed preset", key=f"{prefix}_manual_apply_seed")
    with tool_cols[2]:
        reset_current_clicked = st.button("Reset to current profile", key=f"{prefix}_manual_reset_current")
    with tool_cols[3]:
        clear_all_clicked = st.button("Clear all", key=f"{prefix}_manual_clear_all")

    if apply_seed_clicked or reset_current_clicked or clear_all_clicked:
        if clear_all_clicked:
            seed_to_apply = {metric_name: 0.0 for metric_name, _ in MANUAL_COMPOSITE_WEIGHT_FIELDS}
            st.session_state[seed_profile_key] = "manual_zeroed"
        else:
            selected_seed_profile = profile_key if reset_current_clicked else str(manual_seed_profile)
            seed_to_apply = _get_manual_composite_profile_seed(selected_seed_profile)
            st.session_state[seed_profile_key] = str(selected_seed_profile)
            st.session_state[custom_seed_key] = str(selected_seed_profile)
        for metric_name, _ in MANUAL_COMPOSITE_WEIGHT_FIELDS:
            st.session_state[f"{prefix}_manual_weight_{metric_name}"] = float(seed_to_apply.get(metric_name, 0.0))
        st.rerun()

    cols = st.columns(3)
    manual_weights: dict[str, float] = {}
    for idx, (metric_name, label) in enumerate(MANUAL_COMPOSITE_WEIGHT_FIELDS):
        with cols[idx % 3]:
            value = st.number_input(
                label,
                min_value=-1.0,
                max_value=1.0,
                value=float(st.session_state.get(f"{prefix}_manual_weight_{metric_name}", seed.get(metric_name, 0.0))),
                step=0.05,
                format="%.2f",
                key=f"{prefix}_manual_weight_{metric_name}",
                help="Use positive weights for reward metrics and negative weights for cost metrics when following the canonical sign convention.",
            )
            manual_weights[str(metric_name)] = float(value)

    active_weights = {k: float(v) for k, v in manual_weights.items() if np.isfinite(v) and abs(float(v)) > 1e-12}
    signed_sum = float(sum(active_weights.values())) if active_weights else 0.0
    abs_sum = float(sum(abs(v) for v in active_weights.values())) if active_weights else 0.0
    if not active_weights:
        st.warning("At least one manual composite weight must be non-zero. Falling back to the selected preset profile.")
        return False, None

    reward_metrics = {"sharpe", "cagr", "diversification", "stability"}
    cost_metrics = {"max_drawdown", "mean_turnover"}
    reward_sign_issues = [metric for metric in reward_metrics if metric in active_weights and float(active_weights[metric]) < 0.0]
    cost_sign_issues = [metric for metric in cost_metrics if metric in active_weights and float(active_weights[metric]) > 0.0]
    if reward_sign_issues or cost_sign_issues:
        issues = []
        if reward_sign_issues:
            issues.append("reward metrics with negative weights: " + ", ".join(sorted(reward_sign_issues)))
        if cost_sign_issues:
            issues.append("cost metrics with positive weights: " + ", ".join(sorted(cost_sign_issues)))
        st.info("Manual weights will still run, but the sign convention looks unusual (" + "; ".join(issues) + ").")

    omitted_metrics = [metric_name for metric_name, _ in MANUAL_COMPOSITE_WEIGHT_FIELDS if metric_name not in active_weights]
    preview_weights = {k: (abs(float(v)) / abs_sum if abs_sum > 1e-12 else 0.0) for k, v in active_weights.items()}
    st.caption(f"Active weights: {len(active_weights)} · signed sum={signed_sum:.2f} · abs sum={abs_sum:.2f}")
    if preview_weights:
        preview_parts = [f"{metric}={weight:.2f}" for metric, weight in preview_weights.items()]
        st.caption("Backend normalized preview (abs-renormalized): " + " · ".join(preview_parts))
    if omitted_metrics:
        st.caption("Omitted / zero-weight metrics: " + ", ".join(str(x) for x in omitted_metrics))
    return True, manual_weights


def _is_simple_mode_active() -> bool:
    return bool(st.session_state.get("universe_simple_mode_enabled", True))


def _get_selection_policy_options_for_current_mode() -> list[str]:
    if _is_simple_mode_active():
        return ["fixed_composite_score"]
    return list(PARETO_SELECTION_POLICY_OPTIONS)


def _format_signal_contract_label(label: str) -> str:
    return str(label).replace("|", "→")


def _build_signal_selection_contracts(base_cfg: MicroPipelineConfig, selected_contract_labels: List[str]) -> list[dict]:
    base_payload = config_to_dict(base_cfg)
    labels = [str(x) for x in (selected_contract_labels or []) if str(x) in AUTO_SIGNAL_CONTRACT_LOOKUP]
    if not labels:
        labels = list(AUTO_SIGNAL_CONTRACT_LABELS)
    contracts: list[dict] = []
    seen: set[tuple] = set()
    for label in labels:
        spec = dict(AUTO_SIGNAL_CONTRACT_LOOKUP.get(label, {}))
        if not spec:
            continue
        payload = dict(base_payload)
        payload.update({
            "signal_mode": spec.get("signal_mode"),
            "multi_loss_enabled": bool(spec.get("multi_loss_enabled", False)),
            "low_signal_fallback_to_ew": bool(spec.get("low_signal_fallback_to_ew", False)),
        })
        key = (
            str(payload.get("signal_mode")),
            bool(payload.get("multi_loss_enabled", False)),
            bool(payload.get("low_signal_fallback_to_ew", False)),
        )
        if key in seen:
            continue
        seen.add(key)
        contracts.append({
            "label": str(label),
            "signal_mode_requested": str(payload.get("signal_mode")),
            "multi_loss_enabled": bool(payload.get("multi_loss_enabled", False)),
            "low_signal_fallback_to_ew": bool(payload.get("low_signal_fallback_to_ew", False)),
            "cfg_payload": payload,
        })
    return contracts


def _attach_signal_contract_metadata(selection_table: pd.DataFrame, contracts: list[dict]) -> pd.DataFrame:
    if not isinstance(selection_table, pd.DataFrame) or selection_table.empty:
        return selection_table
    meta_df = pd.DataFrame([
        {
            "signal_mode": c.get("label"),
            "contract_label": c.get("label"),
            "signal_mode_requested_contract": c.get("signal_mode_requested"),
            "contract_multi_loss_enabled": c.get("multi_loss_enabled"),
            "contract_low_signal_fallback_to_ew": c.get("low_signal_fallback_to_ew"),
        }
        for c in (contracts or [])
    ])
    if meta_df.empty:
        return selection_table
    return selection_table.merge(meta_df, on="signal_mode", how="left")


def _format_probabilistic_mode_label(mode: str) -> str:
    return PROBABILISTIC_MODE_LABELS.get(str(mode), str(mode))


def _is_experimental_probabilistic_mode(mode: str) -> bool:
    return str(mode) in set(PROBABILISTIC_MODE_EXPERIMENTAL)




def _render_educational_finance_disclaimer(context_label: str = "") -> None:
    context = f" ({str(context_label).strip()})" if str(context_label).strip() else ""
    st.caption(
        "Educational / research use only" + context + ": This application does not provide financial advice, investment recommendations, or an offer to buy or sell any financial instrument. "
        "Results are experimental, may not generalise to real-world market conditions, and past performance is not indicative of future results. "
        "Users remain solely responsible for any investment decisions made using this application."
    )


def _custom_universe_supported_for_size(universe_size: Any) -> bool:
    try:
        return int(universe_size) in {12, 25}
    except Exception:
        return False

def _large_universe_test_summary(asset_count: int) -> tuple[str, str]:
    n = int(asset_count)
    if 200 <= n <= 250:
        return ("ok", f"Formal 200+ research scale-test range active: {n} assets.")
    if 50 <= n <= 100:
        return ("ok", f"Formal 50–100 scale-test range active: {n} assets.")
    if 101 <= n < 200:
        return ("warn", f"Large universe selected: {n} assets. This is between the formal 50–100 UX band and the 200+ research band.")
    if n > 250:
        return ("warn", f"Very large universe selected: {n} assets. This is above the formal 200–250 research stress-test range.")
    if n >= 25:
        return ("info", f"Medium universe selected: {n} assets. Add more names if you want the formal large-universe test.")
    return ("info", f"Current universe size: {n} assets.")


_PENDING_SESSION_UPDATES_KEY = "_pending_session_updates"
_PENDING_TOASTS_KEY = "_pending_toasts"


def _queue_session_updates(updates: dict) -> None:
    pending = dict(st.session_state.get(_PENDING_SESSION_UPDATES_KEY, {}))
    pending.update({k: v for k, v in _coerce_mapping(updates).items()})
    st.session_state[_PENDING_SESSION_UPDATES_KEY] = pending


def _apply_pending_session_updates(*keys: str) -> None:
    pending = dict(st.session_state.get(_PENDING_SESSION_UPDATES_KEY, {}))
    if not pending:
        return
    selected_keys = set(str(k) for k in keys if k is not None)
    if not selected_keys:
        selected_keys = set(pending.keys())
    remaining = {}
    for k, v in pending.items():
        if k in selected_keys:
            st.session_state[k] = v
        else:
            remaining[k] = v
    if remaining:
        st.session_state[_PENDING_SESSION_UPDATES_KEY] = remaining
    elif _PENDING_SESSION_UPDATES_KEY in st.session_state:
        del st.session_state[_PENDING_SESSION_UPDATES_KEY]


def _queue_toast(message: str, icon: str = "✅") -> None:
    toasts = list(st.session_state.get(_PENDING_TOASTS_KEY, []))
    toasts.append({"message": str(message), "icon": str(icon)})
    st.session_state[_PENDING_TOASTS_KEY] = toasts


def _flush_pending_toasts() -> None:
    toasts = list(st.session_state.get(_PENDING_TOASTS_KEY, []))
    if not toasts:
        return
    if _PENDING_TOASTS_KEY in st.session_state:
        del st.session_state[_PENDING_TOASTS_KEY]
    for item in toasts:
        try:
            st.toast(str(item.get("message", "Done")), icon=str(item.get("icon", "✅")))
        except Exception:
            pass


def _clip01_ui(value: Any, default: float = 0.5) -> float:
    try:
        v = float(value)
    except Exception:
        v = float(default)
    if not np.isfinite(v):
        v = float(default)
    return float(np.clip(v, 0.0, 1.0))


_SIMPLE_SLIDER_KEYS = {
    "risk_appetite": "universe_simple_risk_appetite",
    "diversification": "universe_simple_diversification",
    "stability": "universe_simple_stability",
    "turnover_pref": "universe_simple_turnover_pref",
    "drawdown_protection": "universe_simple_drawdown_protection",
    "overlay_intensity": "universe_simple_overlay_intensity",
    "signal_confidence": "universe_simple_signal_confidence",
    "simplicity": "universe_simple_simplicity",
}


def _get_simple_slider_template_defaults(strategy_template: str) -> dict[str, float]:
    strategy = str(strategy_template or "Balanced Risk-Controlled")
    templates = {
        "Core Ranking": {
            "risk_appetite": 0.48,
            "diversification": 0.42,
            "stability": 0.62,
            "turnover_pref": 0.36,
            "drawdown_protection": 0.58,
            "overlay_intensity": 0.18,
            "signal_confidence": 0.62,
            "simplicity": 0.78,
        },
        "Balanced Risk-Controlled": {
            "risk_appetite": 0.50,
            "diversification": 0.55,
            "stability": 0.60,
            "turnover_pref": 0.40,
            "drawdown_protection": 0.62,
            "overlay_intensity": 0.42,
            "signal_confidence": 0.56,
            "simplicity": 0.56,
        },
        "Hybrid Research": {
            "risk_appetite": 0.56,
            "diversification": 0.58,
            "stability": 0.48,
            "turnover_pref": 0.62,
            "drawdown_protection": 0.48,
            "overlay_intensity": 0.70,
            "signal_confidence": 0.66,
            "simplicity": 0.26,
        },
    }
    return dict(templates.get(strategy, templates["Balanced Risk-Controlled"]))


def _get_simple_slider_style_bias(style_preset: str) -> dict[str, float]:
    style = str(style_preset or "Balanced")
    biases = {
        "Conservative": {
            "risk_appetite": -0.18,
            "diversification": 0.10,
            "stability": 0.16,
            "turnover_pref": -0.14,
            "drawdown_protection": 0.20,
            "overlay_intensity": -0.12,
            "signal_confidence": -0.04,
            "simplicity": 0.12,
        },
        "Balanced": {},
        "Growth": {
            "risk_appetite": 0.22,
            "diversification": -0.08,
            "stability": -0.10,
            "turnover_pref": 0.12,
            "drawdown_protection": -0.16,
            "overlay_intensity": 0.08,
            "signal_confidence": 0.06,
            "simplicity": -0.10,
        },
        "Defensive": {
            "risk_appetite": -0.24,
            "diversification": 0.12,
            "stability": 0.20,
            "turnover_pref": -0.18,
            "drawdown_protection": 0.26,
            "overlay_intensity": -0.06,
            "signal_confidence": -0.02,
            "simplicity": 0.10,
        },
        "Research": {
            "risk_appetite": 0.06,
            "diversification": 0.04,
            "stability": 0.02,
            "turnover_pref": 0.08,
            "drawdown_protection": 0.00,
            "overlay_intensity": 0.20,
            "signal_confidence": 0.12,
            "simplicity": -0.22,
        },
    }
    return dict(biases.get(style, {}))


def _build_simple_slider_recommendations(strategy_template: str, style_preset: str) -> dict[str, float]:
    values = _get_simple_slider_template_defaults(strategy_template)
    for metric_name, bias in _get_simple_slider_style_bias(style_preset).items():
        values[metric_name] = _clip01_ui(float(values.get(metric_name, 0.5)) + float(bias), default=0.5)
    return {k: _clip01_ui(v, default=0.5) for k, v in values.items()}


def _maybe_apply_simple_slider_profile_defaults() -> None:
    strategy_template = str(st.session_state.get("universe_simple_strategy_template", "Balanced Risk-Controlled"))
    style_preset = str(st.session_state.get("universe_simple_style_preset", "Balanced"))
    current_signature = f"{strategy_template}||{style_preset}"
    last_signature = str(st.session_state.get("universe_simple_slider_profile_signature", "") or "")
    first_boot = not bool(st.session_state.get("universe_simple_slider_profile_initialized", False))
    if (not first_boot) and current_signature == last_signature:
        return

    recommendations = _build_simple_slider_recommendations(strategy_template, style_preset)
    for metric_name, session_key in _SIMPLE_SLIDER_KEYS.items():
        st.session_state[session_key] = float(recommendations.get(metric_name, 0.5))

    st.session_state["universe_simple_slider_profile_signature"] = current_signature
    st.session_state["universe_simple_slider_profile_initialized"] = True
    st.session_state["universe_simple_slider_profile_last_source"] = current_signature


def _ensure_simple_slider_state_initialized() -> None:
    strategy_template = str(st.session_state.get("universe_simple_strategy_template", "Balanced Risk-Controlled"))
    style_preset = str(st.session_state.get("universe_simple_style_preset", "Balanced"))
    current_signature = f"{strategy_template}||{style_preset}"

    if bool(st.session_state.get("universe_simple_slider_profile_initialized", False)):
        return

    recommendations = _build_simple_slider_recommendations(strategy_template, style_preset)
    for metric_name, session_key in _SIMPLE_SLIDER_KEYS.items():
        if session_key not in st.session_state:
            st.session_state[session_key] = float(recommendations.get(metric_name, 0.5))

    st.session_state["universe_simple_slider_profile_signature"] = current_signature
    st.session_state["universe_simple_slider_profile_initialized"] = True
    st.session_state["universe_simple_slider_profile_last_source"] = "initial_boot"


def _sync_manual_controls_from_simple_cfg(cfg: MicroPipelineConfig) -> None:
    if not isinstance(cfg, MicroPipelineConfig):
        return

    cfg_payload = config_to_dict(cfg)
    for cfg_name, session_key in ENGINE_PRESET_SESSION_MAP.items():
        if cfg_name not in cfg_payload:
            continue
        value = cfg_payload.get(cfg_name)
        if cfg_name == "top_k":
            st.session_state["universe_micro_top_k_none"] = value is None
            if value is not None:
                st.session_state[session_key] = int(value)
            continue
        st.session_state[session_key] = value

    st.session_state["universe_simple_manual_sync_source"] = "simple_semantic_resolve"


def _resolve_and_sync_simple_mode_state(universe_size: int) -> dict[str, Any]:
    spec = SimpleUISpec(
        strategy_template=str(st.session_state.get("universe_simple_strategy_template", "Balanced Risk-Controlled")),
        style_preset=str(st.session_state.get("universe_simple_style_preset", "Balanced")),
        risk_appetite=float(st.session_state.get("universe_simple_risk_appetite", 0.5)),
        diversification=float(st.session_state.get("universe_simple_diversification", 0.5)),
        stability=float(st.session_state.get("universe_simple_stability", 0.5)),
        turnover_pref=float(st.session_state.get("universe_simple_turnover_pref", 0.5)),
        drawdown_protection=float(st.session_state.get("universe_simple_drawdown_protection", 0.5)),
        overlay_intensity=float(st.session_state.get("universe_simple_overlay_intensity", 0.5)),
        signal_confidence=float(st.session_state.get("universe_simple_signal_confidence", 0.5)),
        simplicity=float(st.session_state.get("universe_simple_simplicity", 0.5)),
        universe_size=int(max(int(universe_size or 0), 0)),
    )
    cfg, summary = resolve_simple_ui_to_micro_cfg(spec)
    st.session_state["universe_simple_resolved_summary"] = summary
    st.session_state["universe_simple_resolved_config"] = config_to_dict(cfg)
    return summary


def _build_current_simple_spec(universe_size: int) -> SimpleUISpec:
    return SimpleUISpec(
        strategy_template=str(st.session_state.get("universe_simple_strategy_template", "Balanced Risk-Controlled")),
        style_preset=str(st.session_state.get("universe_simple_style_preset", "Balanced")),
        risk_appetite=float(st.session_state.get("universe_simple_risk_appetite", 0.5)),
        diversification=float(st.session_state.get("universe_simple_diversification", 0.5)),
        stability=float(st.session_state.get("universe_simple_stability", 0.5)),
        turnover_pref=float(st.session_state.get("universe_simple_turnover_pref", 0.5)),
        drawdown_protection=float(st.session_state.get("universe_simple_drawdown_protection", 0.5)),
        overlay_intensity=float(st.session_state.get("universe_simple_overlay_intensity", 0.5)),
        signal_confidence=float(st.session_state.get("universe_simple_signal_confidence", 0.5)),
        simplicity=float(st.session_state.get("universe_simple_simplicity", 0.5)),
        universe_size=int(max(int(universe_size or 0), 0)),
    )


def _simple_spec_with_profile_defaults(universe_size: int, strategy_template: str, style_preset: str) -> SimpleUISpec:
    slider_values = _build_simple_slider_recommendations(str(strategy_template), str(style_preset))
    return SimpleUISpec(
        strategy_template=str(strategy_template),
        style_preset=str(style_preset),
        risk_appetite=float(slider_values.get("risk_appetite", 0.5)),
        diversification=float(slider_values.get("diversification", 0.5)),
        stability=float(slider_values.get("stability", 0.5)),
        turnover_pref=float(slider_values.get("turnover_pref", 0.5)),
        drawdown_protection=float(slider_values.get("drawdown_protection", 0.5)),
        overlay_intensity=float(slider_values.get("overlay_intensity", 0.5)),
        signal_confidence=float(slider_values.get("signal_confidence", 0.5)),
        simplicity=float(slider_values.get("simplicity", 0.5)),
        universe_size=int(max(int(universe_size or 0), 0)),
    )


def _simple_spec_signature(spec: SimpleUISpec) -> tuple:
    return (
        str(spec.strategy_template),
        str(spec.style_preset),
        round(float(spec.risk_appetite), 4),
        round(float(spec.diversification), 4),
        round(float(spec.stability), 4),
        round(float(spec.turnover_pref), 4),
        round(float(spec.drawdown_protection), 4),
        round(float(spec.overlay_intensity), 4),
        round(float(spec.signal_confidence), 4),
        round(float(spec.simplicity), 4),
        int(spec.universe_size),
    )


def _safe_perf_float(value: Any) -> float:
    try:
        v = float(value)
    except Exception:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _recommendation_score_from_perf(perf_summary: Any, run_result: Any = None) -> float:
    perf = _coerce_mapping(perf_summary)
    run_map = _coerce_mapping(run_result)
    sharpe = _safe_perf_float(perf.get("sharpe"))
    cagr = _safe_perf_float(perf.get("cagr"))
    max_dd = abs(_safe_perf_float(perf.get("max_drawdown")))
    turnover = _safe_perf_float(perf.get("mean_turnover"))
    if not np.isfinite(turnover):
        turnover = _safe_perf_float((_coerce_mapping(run_map.get("turnover_summary", {}))).get("mean_turnover"))
    div_ratio = _safe_perf_float((_coerce_mapping(run_map.get("diversification_summary", {}))).get("mean_diversification_ratio"))
    score = 0.0
    score += sharpe if np.isfinite(sharpe) else -0.50
    score += 2.0 * cagr if np.isfinite(cagr) else 0.0
    score -= 1.5 * max_dd if np.isfinite(max_dd) else 0.0
    score -= 0.35 * turnover if np.isfinite(turnover) else 0.0
    score += 0.10 * div_ratio if np.isfinite(div_ratio) else 0.0
    return float(score)


def _build_preset_recommendation_candidates(current_spec: SimpleUISpec) -> list[dict[str, Any]]:
    universe_size = int(max(int(current_spec.universe_size or 0), 0))
    current_template = str(current_spec.strategy_template)
    current_style = str(current_spec.style_preset)
    style_candidates = ["Balanced", "Conservative", "Defensive", "Growth"]
    if current_style == "Research" or current_template == "Hybrid Research":
        style_candidates.append("Research")
    template_candidates = ["Core Ranking", "Balanced Risk-Controlled", "Hybrid Research"]
    candidates: list[dict[str, Any]] = []
    seen: set[tuple] = set()

    def add_candidate(spec: SimpleUISpec, family: str, label: str) -> None:
        sig = _simple_spec_signature(spec)
        if sig in seen:
            return
        seen.add(sig)
        candidates.append({
            "spec": spec,
            "family": str(family),
            "label": str(label),
            "strategy_template": str(spec.strategy_template),
            "style_preset": str(spec.style_preset),
        })

    add_candidate(current_spec, "current", "Current setup")
    for style_name in style_candidates:
        add_candidate(_simple_spec_with_profile_defaults(universe_size, current_template, style_name), "style_preserving", f"{current_template} + {style_name}")
    for template_name in template_candidates:
        add_candidate(_simple_spec_with_profile_defaults(universe_size, template_name, current_style), "template_shift", f"{template_name} + {current_style}")
    for template_name, style_name in [
        ("Balanced Risk-Controlled", "Defensive"),
        ("Balanced Risk-Controlled", "Balanced"),
        ("Core Ranking", "Growth"),
        ("Hybrid Research", "Research"),
    ]:
        add_candidate(_simple_spec_with_profile_defaults(universe_size, template_name, style_name), "opportunity", f"{template_name} + {style_name}")
    return candidates


def _evaluate_simple_preset_recommendations(primary_engine_panel_df: pd.DataFrame, current_spec: SimpleUISpec, current_run: Any, current_cfg: MicroPipelineConfig) -> dict[str, Any]:
    if not isinstance(primary_engine_panel_df, pd.DataFrame) or primary_engine_panel_df.empty:
        return {"rows": pd.DataFrame(), "strategy_preserving": None, "opportunity": None}
    rows: list[dict[str, Any]] = []
    candidates = _build_preset_recommendation_candidates(current_spec)
    current_sig = _simple_spec_signature(current_spec)
    baseline_perf = _coerce_mapping(_coerce_mapping(current_run).get("performance_summary", {}))
    baseline_score = _recommendation_score_from_perf(baseline_perf, current_run)
    resolved_baseline = _coerce_mapping(st.session_state.get("universe_simple_resolved_summary", {}))

    for candidate in candidates:
        spec = candidate.get("spec")
        if not isinstance(spec, SimpleUISpec):
            continue
        sig = _simple_spec_signature(spec)
        if sig == current_sig:
            run = current_run
            cfg = current_cfg
            summary = resolved_baseline
        else:
            try:
                cfg, summary = resolve_simple_ui_to_micro_cfg(spec)
                run = run_micro_investment_pipeline(primary_engine_panel_df, cfg=cfg)
            except Exception:
                continue
        perf = _coerce_mapping(_coerce_mapping(run).get("performance_summary", {}))
        score_value = _recommendation_score_from_perf(perf, run)
        rows.append({
            "family": str(candidate.get("family", "candidate")),
            "label": str(candidate.get("label", "Candidate")),
            "strategy_template": str(spec.strategy_template),
            "style_preset": str(spec.style_preset),
            "sharpe": _safe_perf_float(perf.get("sharpe")),
            "cagr": _safe_perf_float(perf.get("cagr")),
            "max_drawdown": _safe_perf_float(perf.get("max_drawdown")),
            "annual_volatility": _safe_perf_float(perf.get("annual_volatility", perf.get("annualized_volatility"))),
            "mean_turnover": _safe_perf_float(perf.get("mean_turnover", (_coerce_mapping(_coerce_mapping(run).get("turnover_summary", {}))).get("mean_turnover"))),
            "score": score_value,
            "score_delta": score_value - baseline_score,
            "resolved_signal_mode": str(_coerce_mapping(summary).get("resolved_signal_mode", "—")),
            "resolved_top_k": _coerce_mapping(summary).get("resolved_top_k", "—"),
            "resolved_overlay": str(_coerce_mapping(summary).get("resolved_probabilistic_mode", "—")),
            "resolved_temperature": _coerce_mapping(summary).get("resolved_temperature", np.nan),
            "resolved_weight_shrink": _coerce_mapping(summary).get("resolved_weight_shrink", np.nan),
            "run": run,
            "cfg": cfg,
            "summary": summary,
            "spec": spec,
        })

    if not rows:
        return {"rows": pd.DataFrame(), "strategy_preserving": None, "opportunity": None}

    df = pd.DataFrame([{k: v for k, v in row.items() if k not in {"run", "cfg", "summary", "spec"}} for row in rows])
    strategy_preserving = None
    opportunity = None
    current_template = str(current_spec.strategy_template)
    sp = df[(df["family"] == "style_preserving") & (df["strategy_template"] == current_template)].copy()
    if not sp.empty:
        sp = sp.sort_values(["score", "sharpe", "cagr"], ascending=[False, False, False])
        top = sp.iloc[0]
        if float(top.get("score_delta", 0.0) or 0.0) > 0.01 and str(top.get("label")) != "Current setup":
            strategy_preserving = next((row for row in rows if row.get("label") == top.get("label")), None)
    broader = df[df["label"] != "Current setup"].copy()
    if not broader.empty:
        broader = broader.sort_values(["score", "sharpe", "cagr"], ascending=[False, False, False])
        top = broader.iloc[0]
        if float(top.get("score_delta", 0.0) or 0.0) > 0.02:
            opportunity = next((row for row in rows if row.get("label") == top.get("label")), None)
    return {"rows": df, "strategy_preserving": strategy_preserving, "opportunity": opportunity}


def _queue_apply_simple_preset_recommendation(spec: SimpleUISpec) -> None:
    if not isinstance(spec, SimpleUISpec):
        return
    _queue_session_updates({
        "universe_simple_mode_enabled": True,
        "universe_simple_strategy_template": str(spec.strategy_template),
        "universe_simple_style_preset": str(spec.style_preset),
        "universe_simple_risk_appetite": float(spec.risk_appetite),
        "universe_simple_diversification": float(spec.diversification),
        "universe_simple_stability": float(spec.stability),
        "universe_simple_turnover_pref": float(spec.turnover_pref),
        "universe_simple_drawdown_protection": float(spec.drawdown_protection),
        "universe_simple_overlay_intensity": float(spec.overlay_intensity),
        "universe_simple_signal_confidence": float(spec.signal_confidence),
        "universe_simple_simplicity": float(spec.simplicity),
        "universe_simple_slider_profile_signature": f"{spec.strategy_template}||{spec.style_preset}",
        "universe_simple_slider_profile_initialized": True,
        "universe_simple_slider_profile_last_source": "preset_recommendation",
    })

def apply_large_universe_test_preset(preset_name: str, target: str = "primary") -> None:
    preset_assets = list(LARGE_UNIVERSE_TEST_PRESETS.get(str(preset_name), []))
    if not preset_assets:
        st.warning(f"Large-universe preset not found: {preset_name}")
        return

    text_blob = ", ".join(preset_assets)
    target_key = str(target or "primary").strip().lower()

    if target_key == "comparison":
        _queue_session_updates({
            "comparison_enabled": True,
            "comparison_universe_mode": "Custom asset list",
            "comparison_custom_universe_text": text_blob,
            "comparison_universe_preset_name": st.session_state.get(
                "comparison_universe_preset_name", "Defensive Multi-Asset"
            ),
        })
    else:
        _queue_session_updates({
            "universe_mode": "Custom asset list",
            "custom_universe_text": text_blob,
            "universe_preset_name": st.session_state.get(
                "universe_preset_name", "U12 Core Multi-Asset"
            ),
        })

    st.session_state["universe_large_test_last_applied"] = f"{preset_name} → {target_key}"
    _queue_toast(f"Large-universe test preset applied: {preset_name} ({target_key})", icon="✅")
    st.rerun()

ENGINE_PRESETS = {
    "Balanced Research Default": {
        "min_train": 120,
        "lookback_mu": 12,
        "lookback_sigma": 12,
        "sigma_power_alpha": 1.0,
        "temperature": 1.0,
        "weight_shrink": 0.05,
        "inertia": 0.10,
        "deadband": True,
        "deadband_threshold": 0.02,
        "regime_mode": "quantile",
        "mu_regime_mode": "pooled",
        "ewma_sigma": True,
        "ewma_halflife": 6,
        "probabilistic_mode": "historical_by_regime",
        "probabilistic_q_low": 0.25,
        "probabilistic_q_high": 0.75,
        "probabilistic_min_obs": 12,
        "probabilistic_interval_penalty_weight": 0.25,
        "probabilistic_downside_penalty_weight": 0.25,
        "probabilistic_confidence_scale": 1.0,
        "probabilistic_confidence_min_mult": 0.75,
        "probabilistic_confidence_max_mult": 1.25,
        "dispersion_sigma_enabled": True,
        "dispersion_sigma_lookback": 12,
        "dispersion_sigma_strength": 0.50,
        "dispersion_sigma_floor_mult": 0.75,
        "dispersion_sigma_ceiling_mult": 1.50,
        "signal_mode": "mu_sigma",
        "huber_delta": 1.0,
        "signal_score_blend": 1.0,
        "lambdarank_lookback": 12,
        "lambdarank_temperature": 1.0,
        "lambdarank_real_lookback": 18,
        "lambdarank_real_temperature": 1.0,
        "lambdarank_real_gain_power": 1.0,
        "lambdarank_real_pair_power": 1.0,
        "lambdarank_real_l2": 0.0,
        "quantile_loss_lookback": 18,
        "quantile_loss_q": 0.60,
        "quantile_loss_alpha": 0.0,
        "quantile_loss_confidence_scale": 1.0,
        "multi_loss_enabled": False,
        "multi_loss_rank_weight": 0.40,
        "multi_loss_direction_weight": 0.30,
        "multi_loss_return_weight": 0.30,
        "multi_loss_logistic_weight": 0.0,
        "multi_loss_topk_weight": 0.0,
        "multi_loss_temperature": 1.0,
        "multi_loss_l2": 0.0,
        "directional_classifier_lookback": 12,
        "directional_classifier_threshold": 0.50,
        "directional_classifier_confidence_scale": 1.0,
        "asset_weight_cap": 0.20,
        "w_cap": 0.10,
    },
    "Defensive Overlay": {
        "min_train": 132,
        "lookback_mu": 12,
        "lookback_sigma": 15,
        "sigma_power_alpha": 1.20,
        "temperature": 0.80,
        "weight_shrink": 0.12,
        "inertia": 0.35,
        "deadband": True,
        "deadband_threshold": 0.03,
        "regime_mode": "quantile",
        "mu_regime_mode": "pooled",
        "ewma_sigma": True,
        "ewma_halflife": 8,
        "probabilistic_mode": "historical_by_regime",
        "probabilistic_q_low": 0.20,
        "probabilistic_q_high": 0.80,
        "probabilistic_min_obs": 16,
        "probabilistic_interval_penalty_weight": 0.35,
        "probabilistic_downside_penalty_weight": 0.40,
        "probabilistic_confidence_scale": 0.90,
        "probabilistic_confidence_min_mult": 0.75,
        "probabilistic_confidence_max_mult": 1.15,
        "dispersion_sigma_enabled": True,
        "dispersion_sigma_lookback": 15,
        "dispersion_sigma_strength": 0.65,
        "dispersion_sigma_floor_mult": 0.80,
        "dispersion_sigma_ceiling_mult": 1.35,
        "signal_mode": "huber_mu",
        "huber_delta": 1.25,
        "signal_score_blend": 0.90,
        "lambdarank_lookback": 12,
        "lambdarank_temperature": 1.0,
        "lambdarank_real_lookback": 18,
        "lambdarank_real_temperature": 1.0,
        "lambdarank_real_gain_power": 1.0,
        "lambdarank_real_pair_power": 1.0,
        "lambdarank_real_l2": 0.0,
        "quantile_loss_lookback": 18,
        "quantile_loss_q": 0.60,
        "quantile_loss_alpha": 0.0,
        "quantile_loss_confidence_scale": 1.0,
        "multi_loss_enabled": False,
        "multi_loss_rank_weight": 0.40,
        "multi_loss_direction_weight": 0.30,
        "multi_loss_return_weight": 0.30,
        "multi_loss_logistic_weight": 0.0,
        "multi_loss_topk_weight": 0.0,
        "multi_loss_temperature": 1.0,
        "multi_loss_l2": 0.0,
        "directional_classifier_lookback": 12,
        "directional_classifier_threshold": 0.55,
        "directional_classifier_confidence_scale": 0.90,
        "asset_weight_cap": 0.16,
        "w_cap": 0.08,
    },
    "Adaptive Ranking": {
        "min_train": 120,
        "lookback_mu": 9,
        "lookback_sigma": 12,
        "sigma_power_alpha": 0.85,
        "temperature": 1.20,
        "weight_shrink": 0.04,
        "inertia": 0.15,
        "deadband": True,
        "deadband_threshold": 0.015,
        "regime_mode": "quantile",
        "mu_regime_mode": "split",
        "ewma_sigma": True,
        "ewma_halflife": 5,
        "probabilistic_mode": "quantile_regression",
        "probabilistic_q_low": 0.25,
        "probabilistic_q_high": 0.75,
        "probabilistic_min_obs": 12,
        "probabilistic_interval_penalty_weight": 0.20,
        "probabilistic_downside_penalty_weight": 0.25,
        "probabilistic_confidence_scale": 1.10,
        "probabilistic_confidence_min_mult": 0.80,
        "probabilistic_confidence_max_mult": 1.30,
        "probabilistic_qr_alpha": 0.02,
        "probabilistic_qr_solver": "highs",
        "probabilistic_qr_feature_cap": 8,
        "dispersion_sigma_enabled": True,
        "dispersion_sigma_lookback": 12,
        "dispersion_sigma_strength": 0.45,
        "dispersion_sigma_floor_mult": 0.75,
        "dispersion_sigma_ceiling_mult": 1.45,
        "signal_mode": "lambdarank_real",
        "huber_delta": 1.0,
        "signal_score_blend": 1.0,
        "lambdarank_lookback": 12,
        "lambdarank_temperature": 1.10,
        "lambdarank_real_lookback": 18,
        "lambdarank_real_temperature": 1.10,
        "lambdarank_real_gain_power": 1.0,
        "lambdarank_real_pair_power": 1.0,
        "lambdarank_real_l2": 0.0,
        "quantile_loss_lookback": 18,
        "quantile_loss_q": 0.60,
        "quantile_loss_alpha": 0.0,
        "quantile_loss_confidence_scale": 1.0,
        "multi_loss_enabled": True,
        "multi_loss_rank_weight": 0.50,
        "multi_loss_direction_weight": 0.20,
        "multi_loss_return_weight": 0.20,
        "multi_loss_logistic_weight": 0.10,
        "multi_loss_topk_weight": 0.0,
        "multi_loss_temperature": 1.0,
        "multi_loss_l2": 0.0,
        "directional_classifier_lookback": 12,
        "directional_classifier_threshold": 0.50,
        "directional_classifier_confidence_scale": 1.0,
        "asset_weight_cap": 0.22,
        "w_cap": 0.10,
    },
}

ENGINE_PRESET_META = {
    "Balanced Research Default": {
        "summary": "General-purpose research baseline with regime-aware overlay and moderate regularisation.",
        "style": "Balanced",
        "focus": "Stability + broad compatibility",
    },
    "Defensive Overlay": {
        "summary": "More conservative preset: stronger shrinkage, more inertia, tighter caps and heavier downside control.",
        "style": "Defensive",
        "focus": "Lower turnover + more robustness",
    },
    "Adaptive Ranking": {
        "summary": "More research-forward preset: ranking-style signal, quantile-regression overlay and slightly more reactivity.",
        "style": "Adaptive",
        "focus": "Cross-sectional signal exploitation",
    },
}

COST_MODEL_PRESETS = {
    "Off / legacy": {
        "cost_model_enabled": False,
        "transaction_cost_commission_bps": 0.0,
        "transaction_cost_slippage_bps": 0.0,
        "transaction_cost_spread_bps": 0.0,
        "transaction_cost_market_impact_bps": 0.0,
        "transaction_cost_market_impact_power": 1.0,
        "transaction_cost_min_trade_weight": 0.0,
        "holding_cost_annual_bps": 0.0,
        "tax_model_enabled": False,
        "tax_short_term_rate": 0.0,
        "tax_long_term_rate": 0.0,
        "tax_long_term_threshold_months": 12,
        "tax_apply_loss_credit": False,
        "tax_loss_credit_rate": 0.0,
    },
    "Low-cost ETF / tax-sheltered": {
        "cost_model_enabled": True,
        "transaction_cost_commission_bps": 0.0,
        "transaction_cost_slippage_bps": 2.0,
        "transaction_cost_spread_bps": 1.0,
        "transaction_cost_market_impact_bps": 1.0,
        "transaction_cost_market_impact_power": 1.0,
        "transaction_cost_min_trade_weight": 0.001,
        "holding_cost_annual_bps": 5.0,
        "tax_model_enabled": False,
        "tax_short_term_rate": 0.0,
        "tax_long_term_rate": 0.0,
        "tax_long_term_threshold_months": 12,
        "tax_apply_loss_credit": False,
        "tax_loss_credit_rate": 0.0,
    },
    "Realistic taxable retail": {
        "cost_model_enabled": True,
        "transaction_cost_commission_bps": 1.0,
        "transaction_cost_slippage_bps": 4.0,
        "transaction_cost_spread_bps": 2.0,
        "transaction_cost_market_impact_bps": 3.0,
        "transaction_cost_market_impact_power": 1.0,
        "transaction_cost_min_trade_weight": 0.002,
        "holding_cost_annual_bps": 10.0,
        "tax_model_enabled": True,
        "tax_short_term_rate": 0.30,
        "tax_long_term_rate": 0.15,
        "tax_long_term_threshold_months": 12,
        "tax_apply_loss_credit": True,
        "tax_loss_credit_rate": 0.15,
    },
    "Stress / high-friction": {
        "cost_model_enabled": True,
        "transaction_cost_commission_bps": 2.0,
        "transaction_cost_slippage_bps": 8.0,
        "transaction_cost_spread_bps": 4.0,
        "transaction_cost_market_impact_bps": 8.0,
        "transaction_cost_market_impact_power": 1.2,
        "transaction_cost_min_trade_weight": 0.003,
        "holding_cost_annual_bps": 20.0,
        "tax_model_enabled": True,
        "tax_short_term_rate": 0.35,
        "tax_long_term_rate": 0.20,
        "tax_long_term_threshold_months": 12,
        "tax_apply_loss_credit": True,
        "tax_loss_credit_rate": 0.20,
    },
}

COST_MODEL_PRESET_META = {
    "Off / legacy": "Pure legacy behaviour with no implementation drag.",
    "Low-cost ETF / tax-sheltered": "Small frictions, low carry cost, no tax drag.",
    "Realistic taxable retail": "Reasonable all-in retail assumptions with taxation and optional loss credit.",
    "Stress / high-friction": "Research stress test for fragile high-turnover strategies.",
}

def apply_cost_model_preset(preset_name: str) -> None:
    preset = COST_MODEL_PRESETS.get(str(preset_name), {})
    if not preset:
        st.warning(f"Cost-model preset not found: {preset_name}")
        return
    session_map = {
        "cost_model_enabled": "universe_cost_model_enabled",
        "transaction_cost_commission_bps": "universe_transaction_cost_commission_bps",
        "transaction_cost_slippage_bps": "universe_transaction_cost_slippage_bps",
        "transaction_cost_spread_bps": "universe_transaction_cost_spread_bps",
        "transaction_cost_market_impact_bps": "universe_transaction_cost_market_impact_bps",
        "transaction_cost_market_impact_power": "universe_transaction_cost_market_impact_power",
        "transaction_cost_min_trade_weight": "universe_transaction_cost_min_trade_weight",
        "holding_cost_annual_bps": "universe_holding_cost_annual_bps",
        "tax_model_enabled": "universe_tax_model_enabled",
        "tax_short_term_rate": "universe_tax_short_term_rate",
        "tax_long_term_rate": "universe_tax_long_term_rate",
        "tax_long_term_threshold_months": "universe_tax_long_term_threshold_months",
        "tax_apply_loss_credit": "universe_tax_apply_loss_credit",
        "tax_loss_credit_rate": "universe_tax_loss_credit_rate",
    }
    for k, v in preset.items():
        ss_key = session_map.get(k)
        if ss_key is not None:
            st.session_state[ss_key] = v
    st.session_state["universe_cost_model_preset"] = str(preset_name)
    st.toast(f"Cost-model preset applied: {preset_name}", icon="✅")

ENGINE_PRESET_SESSION_MAP = {
    "min_train": "universe_micro_min_train",
    "lookback_mu": "universe_micro_lookback_mu",
    "lookback_sigma": "universe_micro_lookback_sigma",
    "sigma_power_alpha": "universe_micro_sigma_power_alpha",
    "temperature": "universe_micro_temperature",
    "top_k": "universe_micro_top_k",
    "weight_shrink": "universe_micro_weight_shrink",
    "inertia": "universe_micro_inertia",
    "deadband": "universe_micro_deadband",
    "deadband_threshold": "universe_micro_deadband_threshold",
    "regime_mode": "universe_micro_regime_mode",
    "mu_regime_mode": "universe_micro_mu_regime_mode",
    "ewma_sigma": "universe_micro_ewma_sigma",
    "ewma_halflife": "universe_micro_ewma_halflife",
    "probabilistic_mode": "universe_micro_prob_mode",
    "probabilistic_q_low": "universe_micro_prob_q_low",
    "probabilistic_q_high": "universe_micro_prob_q_high",
    "probabilistic_min_obs": "universe_micro_prob_min_obs",
    "probabilistic_interval_penalty_weight": "universe_micro_prob_interval_penalty_weight",
    "probabilistic_downside_penalty_weight": "universe_micro_prob_downside_penalty_weight",
    "probabilistic_confidence_scale": "universe_micro_prob_confidence_scale",
    "probabilistic_confidence_min_mult": "universe_micro_prob_confidence_min_mult",
    "probabilistic_confidence_max_mult": "universe_micro_prob_confidence_max_mult",
    "probabilistic_qr_alpha": "universe_micro_prob_qr_alpha",
    "probabilistic_qr_solver": "universe_micro_prob_qr_solver",
    "probabilistic_qr_feature_cap": "universe_micro_prob_qr_feature_cap",
    "dispersion_sigma_enabled": "universe_micro_dispersion_sigma_enabled",
    "dispersion_sigma_lookback": "universe_micro_dispersion_sigma_lookback",
    "dispersion_sigma_strength": "universe_micro_dispersion_sigma_strength",
    "dispersion_sigma_floor_mult": "universe_micro_dispersion_sigma_floor_mult",
    "dispersion_sigma_ceiling_mult": "universe_micro_dispersion_sigma_ceiling_mult",
    "signal_mode": "universe_micro_signal_mode",
    "huber_delta": "universe_micro_huber_delta",
    "signal_score_blend": "universe_micro_signal_score_blend",
    "lambdarank_lookback": "universe_micro_lambdarank_lookback",
    "lambdarank_temperature": "universe_micro_lambdarank_temperature",
    "lambdarank_real_lookback": "universe_micro_lambdarank_real_lookback",
    "lambdarank_real_temperature": "universe_micro_lambdarank_real_temperature",
    "lambdarank_real_gain_power": "universe_micro_lambdarank_real_gain_power",
    "lambdarank_real_pair_power": "universe_micro_lambdarank_real_pair_power",
    "lambdarank_real_l2": "universe_micro_lambdarank_real_l2",
    "multi_loss_enabled": "universe_multi_loss_enabled",
    "multi_loss_rank_weight": "universe_multi_loss_rank_weight",
    "multi_loss_direction_weight": "universe_multi_loss_direction_weight",
    "multi_loss_return_weight": "universe_multi_loss_return_weight",
    "multi_loss_logistic_weight": "universe_multi_loss_logistic_weight",
    "multi_loss_topk_weight": "universe_multi_loss_topk_weight",
    "multi_loss_temperature": "universe_multi_loss_temperature",
    "multi_loss_l2": "universe_multi_loss_l2",
    "directional_classifier_lookback": "universe_micro_directional_classifier_lookback",
    "directional_classifier_threshold": "universe_micro_directional_classifier_threshold",
    "directional_classifier_confidence_scale": "universe_micro_directional_classifier_confidence_scale",
    "logistic_loss_lookback": "universe_micro_logistic_loss_lookback",
    "logistic_loss_l2": "universe_micro_logistic_loss_l2",
    "logistic_loss_threshold": "universe_micro_logistic_loss_threshold",
    "logistic_loss_confidence_scale": "universe_micro_logistic_loss_confidence_scale",
    "top_k_classifier_lookback": "universe_micro_top_k_classifier_lookback",
    "top_k_classifier_k": "universe_micro_top_k_classifier_k",
    "top_k_classifier_threshold": "universe_micro_top_k_classifier_threshold",
    "top_k_classifier_confidence_scale": "universe_micro_top_k_classifier_confidence_scale",
    "quantile_loss_lookback": "universe_micro_quantile_loss_lookback",
    "quantile_loss_q": "universe_micro_quantile_loss_q",
    "quantile_loss_alpha": "universe_micro_quantile_loss_alpha",
    "quantile_loss_confidence_scale": "universe_micro_quantile_loss_confidence_scale",
    "covariance_mode": "universe_micro_covariance_mode",
    "regime_dependent_covariance": "universe_micro_regime_dependent_covariance",
    "correlation_lookback": "universe_micro_correlation_lookback",
    "correlation_min_periods": "universe_micro_correlation_min_periods",
    "correlation_shrink_to_identity": "universe_micro_correlation_shrink_to_identity",
    "covariance_shrink_to_diagonal": "universe_micro_covariance_shrink_to_diagonal",
    "covariance_jitter": "universe_micro_covariance_jitter",
    "covariance_lookback_low": "universe_micro_covariance_lookback_low",
    "covariance_lookback_mid": "universe_micro_covariance_lookback_mid",
    "covariance_lookback_high": "universe_micro_covariance_lookback_high",
    "covariance_halflife_low": "universe_micro_covariance_halflife_low",
    "covariance_halflife_mid": "universe_micro_covariance_halflife_mid",
    "covariance_halflife_high": "universe_micro_covariance_halflife_high",
    "correlation_aware_allocation": "universe_micro_correlation_aware_allocation",
    "correlation_allocator_method": "universe_micro_correlation_allocator_method",
    "correlation_allocator_blend": "universe_micro_correlation_allocator_blend",
    "correlation_penalty_strength": "universe_micro_correlation_penalty_strength",
    "correlation_penalty_power": "universe_micro_correlation_penalty_power",
    "correlation_use_abs": "universe_micro_correlation_use_abs",
    "mean_variance_risk_aversion": "universe_micro_mean_variance_risk_aversion",
    "risk_budget_strength": "universe_micro_risk_budget_strength",
    "cluster_corr_threshold": "universe_micro_cluster_corr_threshold",
    # Phase 3 preset wiring — vol targeting / regime derisk
    "vol_targeting": "universe_micro_vol_targeting",
    "covariance_aware_vol_targeting": "universe_micro_covariance_aware_vol_targeting",
    "target_portfolio_vol_monthly": "universe_micro_target_portfolio_vol_monthly",
    "vol_target_floor_mult": "universe_micro_vol_target_floor_mult",
    "vol_target_ceiling_mult": "universe_micro_vol_target_ceiling_mult",
    "regime_derisk_low": "universe_micro_regime_derisk_low",
    "regime_derisk_mid": "universe_micro_regime_derisk_mid",
    "regime_derisk_high": "universe_micro_regime_derisk_high",
    # Phase 1 preset wiring — adaptive caps
    "vol_dependent_cap_enabled": "universe_micro_vol_cap_enabled",
    "vol_dependent_cap_threshold_low": "universe_micro_vol_cap_threshold_low",
    "vol_dependent_cap_threshold_high": "universe_micro_vol_cap_threshold_high",
    "vol_dependent_cap_low_mult": "universe_micro_vol_cap_low_mult",
    "vol_dependent_cap_high_mult": "universe_micro_vol_cap_high_mult",
    "vol_dependent_cap_min": "universe_micro_vol_cap_min",
    "vol_dependent_cap_max": "universe_micro_vol_cap_max",
    "corr_dependent_cap_enabled": "universe_micro_corr_cap_enabled",
    "corr_dependent_cap_threshold_low": "universe_micro_corr_cap_threshold_low",
    "corr_dependent_cap_threshold_high": "universe_micro_corr_cap_threshold_high",
    "corr_dependent_cap_low_mult": "universe_micro_corr_cap_low_mult",
    "corr_dependent_cap_high_mult": "universe_micro_corr_cap_high_mult",
    "corr_dependent_cap_min": "universe_micro_corr_cap_min",
    "corr_dependent_cap_max": "universe_micro_corr_cap_max",
    "dispersion_dependent_cap_enabled": "universe_micro_disp_cap_enabled",
    "dispersion_dependent_cap_threshold_low": "universe_micro_disp_cap_threshold_low",
    "dispersion_dependent_cap_threshold_high": "universe_micro_disp_cap_threshold_high",
    "dispersion_dependent_cap_low_mult": "universe_micro_disp_cap_low_mult",
    "dispersion_dependent_cap_high_mult": "universe_micro_disp_cap_high_mult",
    "dispersion_dependent_cap_min": "universe_micro_disp_cap_min",
    "dispersion_dependent_cap_max": "universe_micro_disp_cap_max",
    "regime_dependent_cap_enabled": "universe_micro_regime_cap_enabled",
    "regime_dependent_cap_low_mult": "universe_micro_regime_cap_low_mult",
    "regime_dependent_cap_mid_mult": "universe_micro_regime_cap_mid_mult",
    "regime_dependent_cap_high_mult": "universe_micro_regime_cap_high_mult",
    "regime_dependent_cap_min": "universe_micro_regime_cap_min",
    "regime_dependent_cap_max": "universe_micro_regime_cap_max",
    # Phase 1 preset wiring — turnover
    "turnover_penalty_strength": "universe_micro_turnover_penalty_strength",
    "turnover_penalty_power": "universe_micro_turnover_penalty_power",
    "turnover_penalty_target": "universe_micro_turnover_penalty_target",
    "turnover_penalty_max_turnover": "universe_micro_turnover_penalty_max_turnover",
    "turnover_constraint_max_turnover": "universe_micro_turnover_constraint_max_turnover",
    # Phase 1 preset wiring — dispersion governance
    "dispersion_gate": "universe_micro_dispersion_gate",
    "dispersion_gate_threshold": "universe_micro_dispersion_gate_threshold",
    "dispersion_gate_min_active_weight": "universe_micro_dispersion_gate_min_active_weight",
    "dispersion_risk_model_enabled": "universe_micro_dispersion_risk_model_enabled",
    "dispersion_risk_strength": "universe_micro_dispersion_risk_strength",
    "dispersion_risk_floor_mult": "universe_micro_dispersion_risk_floor_mult",
    "dispersion_risk_ceiling_mult": "universe_micro_dispersion_risk_ceiling_mult",
    "dispersion_risk_apply_to_covariance": "universe_micro_dispersion_risk_apply_to_covariance",
    "dispersion_top_k_enabled": "universe_micro_dispersion_top_k_enabled",
    "dispersion_top_k_threshold_low": "universe_micro_dispersion_top_k_threshold_low",
    "dispersion_top_k_threshold_high": "universe_micro_dispersion_top_k_threshold_high",
    "dispersion_top_k_low_mult": "universe_micro_dispersion_top_k_low_mult",
    "dispersion_top_k_high_mult": "universe_micro_dispersion_top_k_high_mult",
    "dispersion_top_k_min_k": "universe_micro_dispersion_top_k_min_k",
    "dispersion_top_k_max_k": "universe_micro_dispersion_top_k_max_k",
    "dispersion_vol_target_enabled": "universe_micro_dispersion_vol_target_enabled",
    "dispersion_vol_target_threshold_low": "universe_micro_dispersion_vol_target_threshold_low",
    "dispersion_vol_target_threshold_high": "universe_micro_dispersion_vol_target_threshold_high",
    "dispersion_vol_target_low_mult": "universe_micro_dispersion_vol_target_low_mult",
    "dispersion_vol_target_high_mult": "universe_micro_dispersion_vol_target_high_mult",
    "dispersion_vol_target_min": "universe_micro_dispersion_vol_target_min",
    "dispersion_vol_target_max": "universe_micro_dispersion_vol_target_max",
    "low_signal_fallback_to_ew": "universe_micro_low_signal_fallback_to_ew",
    "low_signal_fallback_threshold": "universe_micro_low_signal_fallback_threshold",
    "low_signal_fallback_mode": "universe_micro_low_signal_fallback_mode",
    "low_signal_fallback_min_model_weight": "universe_micro_low_signal_fallback_min_model_weight",
    # Phase 4 preset wiring — probabilistic overlay advanced
    "probabilistic_regime_aware": "universe_micro_prob_regime_aware",
    "probabilistic_regime_min_obs": "universe_micro_prob_regime_min_obs",
    "probabilistic_feature_filter_enabled": "universe_micro_prob_feature_filter_enabled",
    "probabilistic_feature_filter_min_obs": "universe_micro_prob_feature_filter_min_obs",
    "probabilistic_feature_filter_k": "universe_micro_prob_feature_filter_k",
    "probabilistic_knn_k": "universe_micro_prob_knn_k",
    "probabilistic_knn_min_obs": "universe_micro_prob_knn_min_obs",
    "probabilistic_knn_distance_power": "universe_micro_prob_knn_distance_power",
    "probabilistic_knn_weighted_quantiles": "universe_micro_prob_knn_weighted_quantiles",
    "probabilistic_knn_weight_eps": "universe_micro_prob_knn_weight_eps",
    "probabilistic_bucket_n_bins": "universe_micro_prob_bucket_n_bins",
    "probabilistic_bucket_min_obs": "universe_micro_prob_bucket_min_obs",
    "probabilistic_bucket_match_min_features": "universe_micro_prob_bucket_match_min_features",
    "probabilistic_bucket_max_features": "universe_micro_prob_bucket_max_features",
    "probabilistic_parametric_min_sigma": "universe_micro_prob_parametric_min_sigma",
    "probabilistic_parametric_max_sigma_mult": "universe_micro_prob_parametric_max_sigma_mult",
    "probabilistic_parametric_use_neighbor_weights": "universe_micro_prob_parametric_use_neighbor_weights",
    "probabilistic_hybrid_weight": "universe_micro_prob_hybrid_weight",
    "probabilistic_hybrid_min_qr_weight": "universe_micro_prob_hybrid_min_qr_weight",
    "probabilistic_hybrid_max_qr_weight": "universe_micro_prob_hybrid_max_qr_weight",
    "probabilistic_hybrid_use_confidence": "universe_micro_prob_hybrid_use_confidence",
    # Phase 5 preset wiring — feature-conditioned mu + research switches
    "feature_mu_enabled": "universe_micro_feature_mu_enabled",
    "feature_mu_blend": "universe_micro_feature_mu_blend",
    "feature_mu_k": "universe_micro_feature_mu_k",
    "feature_mu_min_obs": "universe_micro_feature_mu_min_obs",
    "feature_mu_cols_max": "universe_micro_feature_mu_cols_max",
    "pure_cs_baseline": "universe_micro_pure_cs_baseline",
    # Final config governance — remaining internal/default fields
    "sigma_floor": "universe_micro_sigma_floor",
    "long_only": "universe_micro_long_only",
    "regime_lookback": "universe_micro_regime_lookback",
    "regime_quantile_low": "universe_micro_regime_quantile_low",
    "regime_quantile_high": "universe_micro_regime_quantile_high",
    "score_normalize": "universe_micro_score_normalize",
    "score_clip": "universe_micro_score_clip",
    "store_correlation_snapshots": "universe_micro_store_correlation_snapshots",
    "store_sigma_fwd_snapshots": "universe_micro_store_sigma_fwd_snapshots",
}


CRASH_OVERLAY_PRESETS = {
    "Crash-aware balanced": {
        "probabilistic_mode": "historical_by_regime",
        "probabilistic_q_low": 0.15,
        "probabilistic_q_high": 0.75,
        "probabilistic_min_obs": 12,
        "probabilistic_interval_penalty_weight": 0.30,
        "probabilistic_downside_penalty_weight": 0.90,
        "probabilistic_confidence_scale": 1.10,
        "probabilistic_confidence_min_mult": 0.70,
        "probabilistic_confidence_max_mult": 1.15,
        "probabilistic_overlay_strength": 1.00,
        "probabilistic_overlay_blend": 0.70,
    },
    "Crash-aware defensive": {
        "probabilistic_mode": "historical_by_regime",
        "probabilistic_q_low": 0.10,
        "probabilistic_q_high": 0.70,
        "probabilistic_min_obs": 16,
        "probabilistic_interval_penalty_weight": 0.40,
        "probabilistic_downside_penalty_weight": 1.20,
        "probabilistic_confidence_scale": 1.20,
        "probabilistic_confidence_min_mult": 0.65,
        "probabilistic_confidence_max_mult": 1.10,
        "probabilistic_overlay_strength": 1.20,
        "probabilistic_overlay_blend": 0.85,
    },
    "Crash-aware severe stress": {
        "probabilistic_mode": "historical_by_regime",
        "probabilistic_q_low": 0.05,
        "probabilistic_q_high": 0.65,
        "probabilistic_min_obs": 18,
        "probabilistic_interval_penalty_weight": 0.55,
        "probabilistic_downside_penalty_weight": 1.60,
        "probabilistic_confidence_scale": 1.35,
        "probabilistic_confidence_min_mult": 0.60,
        "probabilistic_confidence_max_mult": 1.05,
        "probabilistic_overlay_strength": 1.40,
        "probabilistic_overlay_blend": 1.00,
    },
}

CRASH_OVERLAY_PRESET_META = {
    "Crash-aware balanced": "Balanced downside-sensitive overlay: stronger left-tail awareness without fully dominating the base signal.",
    "Crash-aware defensive": "Defensive crash preset: heavier downside penalty and tighter confidence bounds for more conservative overlay behaviour.",
    "Crash-aware severe stress": "Severe stress preset: maximum downside focus and overlay influence for crash / left-tail scenario testing.",
}


def _parse_custom_tickers(raw_text: str) -> List[str]:
    normalized = (raw_text or "").replace("\n", ",").replace(";", ",")
    tokens = [str(tok).strip().upper() for tok in normalized.split(",") if str(tok).strip()]
    seen = set()
    out: List[str] = []
    for tok in tokens:
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


@st.cache_data(show_spinner=False)
def _read_uploaded_asset_panel(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    suffix = Path(file_name).suffix.lower()
    import io
    if suffix == ".parquet":
        return pd.read_parquet(io.BytesIO(file_bytes))
    return pd.read_csv(io.BytesIO(file_bytes))



@st.cache_data(show_spinner=False)
def _download_yahoo_asset_panel_cached(
    tickers: tuple[str, ...],
    start_date: str,
    end_date: str,
    frequency: str,
    interval: str,
    auto_adjust: bool,
) -> pd.DataFrame:
    return download_yahoo_return_panel(
        list(tickers),
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
        interval=interval,
        auto_adjust=auto_adjust,
    )


@st.cache_data(show_spinner=False)
def _download_yahoo_macro_feature_panel_cached(
    start_date: str,
    end_date: str,
    frequency: str,
    auto_adjust: bool,
) -> pd.DataFrame:
    return download_yahoo_macro_feature_panel(
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
        auto_adjust=auto_adjust,
    )


@st.cache_data(show_spinner=False)
def _download_yahoo_daily_asset_panel_cached(
    tickers: tuple[str, ...],
    start_date: str,
    end_date: str,
    auto_adjust: bool,
) -> pd.DataFrame:
    return download_yahoo_return_panel(
        list(tickers),
        start_date=start_date,
        end_date=end_date,
        frequency="daily",
        interval="1d",
        auto_adjust=auto_adjust,
    )


def _validate_asset_panel(df: pd.DataFrame) -> pd.DataFrame:
    need = {"date", "asset", "return"}
    missing = sorted(need - set(df.columns))
    if missing:
        raise ValueError(f"Asset panel is missing required columns: {missing}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["asset"] = out["asset"].astype(str).str.upper().str.strip()
    out["return"] = pd.to_numeric(out["return"], errors="coerce")
    out = out.dropna(subset=["date", "asset", "return"]).sort_values(["date", "asset"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("Asset panel is empty after cleaning.")
    return out


def _filter_asset_panel_to_universe(panel_df: pd.DataFrame, selected_assets: List[str]) -> tuple[pd.DataFrame, dict]:
    panel = panel_df.copy()
    panel["asset"] = panel["asset"].astype(str).str.upper()
    available_assets = sorted(panel["asset"].unique().tolist())
    requested_assets = [str(x).upper() for x in (selected_assets or [])]
    if not requested_assets:
        return panel, {"requested_assets": [], "available_assets": available_assets, "used_assets": available_assets, "missing_assets": []}
    available_set = set(available_assets)
    used_assets = [a for a in requested_assets if a in available_set]
    missing_assets = [a for a in requested_assets if a not in available_set]
    filtered = panel[panel["asset"].isin(used_assets)].copy() if used_assets else panel.copy()
    return filtered, {"requested_assets": requested_assets, "available_assets": available_assets, "used_assets": used_assets, "missing_assets": missing_assets}


def _build_engine_asset_panel(
    base_panel_df: pd.DataFrame,
    *,
    source_mode: str,
    requested_frequency: str | None = None,
    daily_panel_df: pd.DataFrame | None = None,
    macro_panel_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Build the panel actually passed into investment.py.

    - For Yahoo monthly/weekly runs, try to enrich the engine input from the daily
      panel using src.features so the probabilistic overlay can become feature-aware.
    - If enrichment fails or is not applicable, fall back cleanly to the base panel.
    - For uploaded files, preserve the uploaded panel as-is; if it already carries
      engineered columns, they remain available to the engine because validation only
      requires date/asset/return and keeps additional columns.
    """
    base_panel = _validate_asset_panel(base_panel_df)
    meta = {
        "panel_kind": "base",
        "feature_enriched": False,
        "engine_rows": int(len(base_panel)),
        "engine_cols": int(base_panel.shape[1]),
        "feature_cols": [],
        "message": "Using base return panel.",
    }

    freq = str(requested_frequency or "").lower()
    try:
        if source_mode == "Yahoo Finance (recommended)" and daily_panel_df is not None and freq in {"monthly", "weekly"}:
            daily_clean = _validate_asset_panel(daily_panel_df)
            if freq == "monthly":
                enriched = build_monthly_panel_from_daily(daily_clean, macro_panel_df=macro_panel_df)
            else:
                enriched = build_weekly_panel_from_daily(daily_clean, macro_panel_df=macro_panel_df)
            enriched = _validate_asset_panel(enriched)
            feature_cols = [c for c in enriched.columns if c not in {"date", "asset", "return"}]
            if feature_cols:
                meta.update({
                    "panel_kind": f"{freq}_feature_enriched",
                    "feature_enriched": True,
                    "engine_rows": int(len(enriched)),
                    "engine_cols": int(enriched.shape[1]),
                    "feature_cols": feature_cols,
                    "message": f"Using feature-enriched {freq} panel built from daily Yahoo returns.",
                })
                return enriched, meta
            meta.update({
                "panel_kind": f"{freq}_base_fallback",
                "message": f"Feature builder returned no extra columns; falling back to base {freq} panel.",
            })
    except Exception as e:
        meta.update({
            "panel_kind": f"{freq or 'base'}_fallback_error",
            "message": f"Feature enrichment failed ({e}); falling back to base return panel.",
        })

    feature_cols = [c for c in base_panel.columns if c not in {"date", "asset", "return"}]
    if feature_cols:
        meta.update({
            "panel_kind": "uploaded_or_pre_enriched",
            "feature_enriched": True,
            "engine_rows": int(len(base_panel)),
            "engine_cols": int(base_panel.shape[1]),
            "feature_cols": feature_cols,
            "message": "Using input panel with pre-existing feature columns.",
        })
    return base_panel, meta


def _build_universe_micro_cfg(universe_size: int = 0) -> MicroPipelineConfig:
    if bool(st.session_state.get("universe_simple_mode_enabled", True)):
        spec = SimpleUISpec(
            strategy_template=str(st.session_state.get("universe_simple_strategy_template", "Balanced Risk-Controlled")),
            style_preset=str(st.session_state.get("universe_simple_style_preset", "Balanced")),
            risk_appetite=float(st.session_state.get("universe_simple_risk_appetite", 0.5)),
            diversification=float(st.session_state.get("universe_simple_diversification", 0.5)),
            stability=float(st.session_state.get("universe_simple_stability", 0.5)),
            turnover_pref=float(st.session_state.get("universe_simple_turnover_pref", 0.5)),
            drawdown_protection=float(st.session_state.get("universe_simple_drawdown_protection", 0.5)),
            overlay_intensity=float(st.session_state.get("universe_simple_overlay_intensity", 0.5)),
            signal_confidence=float(st.session_state.get("universe_simple_signal_confidence", 0.5)),
            simplicity=float(st.session_state.get("universe_simple_simplicity", 0.5)),
            universe_size=int(max(universe_size, 0)),
        )
        cfg, summary = resolve_simple_ui_to_micro_cfg(spec)
        st.session_state["universe_simple_resolved_summary"] = summary
        return cfg

    cfg_payload = dict(
        min_train=int(st.session_state.get("universe_micro_min_train", 120)),
        lookback_mu=int(st.session_state.get("universe_micro_lookback_mu", 12)),
        lookback_sigma=int(st.session_state.get("universe_micro_lookback_sigma", 12)),
        sigma_power_alpha=float(st.session_state.get("universe_micro_sigma_power_alpha", 1.0)),
        temperature=float(st.session_state.get("universe_micro_temperature", 1.0)),
        top_k=(None if st.session_state.get("universe_micro_top_k_none", True) else int(st.session_state.get("universe_micro_top_k", 5))),
        weight_shrink=float(st.session_state.get("universe_micro_weight_shrink", 0.05)),
        inertia=float(st.session_state.get("universe_micro_inertia", 0.0)),
        deadband=bool(st.session_state.get("universe_micro_deadband", True)),
        deadband_threshold=float(st.session_state.get("universe_micro_deadband_threshold", 0.02)),
        regime_mode=str(st.session_state.get("universe_micro_regime_mode", "quantile")),
        mu_regime_mode=str(st.session_state.get("universe_micro_mu_regime_mode", "pooled")),
        ewma_sigma=bool(st.session_state.get("universe_micro_ewma_sigma", True)),
        ewma_halflife=int(st.session_state.get("universe_micro_ewma_halflife", 6)),
        covariance_mode=str(st.session_state.get("universe_micro_covariance_mode", "ewma_cov")),
        regime_dependent_covariance=bool(st.session_state.get("universe_micro_regime_dependent_covariance", True)),
        correlation_lookback=int(st.session_state.get("universe_micro_correlation_lookback", 24)),
        correlation_min_periods=int(st.session_state.get("universe_micro_correlation_min_periods", 6)),
        correlation_shrink_to_identity=float(st.session_state.get("universe_micro_correlation_shrink_to_identity", 0.10)),
        covariance_shrink_to_diagonal=float(st.session_state.get("universe_micro_covariance_shrink_to_diagonal", 0.10)),
        covariance_jitter=float(st.session_state.get("universe_micro_covariance_jitter", 1e-8)),
        covariance_lookback_low=int(st.session_state.get("universe_micro_covariance_lookback_low", 18)),
        covariance_lookback_mid=int(st.session_state.get("universe_micro_covariance_lookback_mid", 24)),
        covariance_lookback_high=int(st.session_state.get("universe_micro_covariance_lookback_high", 36)),
        covariance_halflife_low=int(st.session_state.get("universe_micro_covariance_halflife_low", 4)),
        covariance_halflife_mid=int(st.session_state.get("universe_micro_covariance_halflife_mid", 6)),
        covariance_halflife_high=int(st.session_state.get("universe_micro_covariance_halflife_high", 9)),
        correlation_aware_allocation=bool(st.session_state.get("universe_micro_correlation_aware_allocation", True)),
        correlation_allocator_method=str(st.session_state.get("universe_micro_correlation_allocator_method", "score_penalty")),
        correlation_allocator_blend=float(st.session_state.get("universe_micro_correlation_allocator_blend", 0.35)),
        correlation_penalty_strength=float(st.session_state.get("universe_micro_correlation_penalty_strength", 1.0)),
        correlation_penalty_power=float(st.session_state.get("universe_micro_correlation_penalty_power", 1.0)),
        correlation_use_abs=bool(st.session_state.get("universe_micro_correlation_use_abs", True)),
        mean_variance_risk_aversion=float(st.session_state.get("universe_micro_mean_variance_risk_aversion", 4.0)),
        risk_budget_strength=float(st.session_state.get("universe_micro_risk_budget_strength", 0.5)),
        cluster_corr_threshold=float(st.session_state.get("universe_micro_cluster_corr_threshold", 0.70)),
        vol_targeting=bool(st.session_state.get("universe_micro_vol_targeting", False)),
        covariance_aware_vol_targeting=bool(st.session_state.get("universe_micro_covariance_aware_vol_targeting", True)),
        target_portfolio_vol_monthly=float(st.session_state.get("universe_micro_target_portfolio_vol_monthly", 0.04)),
        vol_target_floor_mult=float(st.session_state.get("universe_micro_vol_target_floor_mult", 0.50)),
        vol_target_ceiling_mult=float(st.session_state.get("universe_micro_vol_target_ceiling_mult", 1.50)),
        regime_derisk_low=float(st.session_state.get("universe_micro_regime_derisk_low", 1.00)),
        regime_derisk_mid=float(st.session_state.get("universe_micro_regime_derisk_mid", 0.95)),
        regime_derisk_high=float(st.session_state.get("universe_micro_regime_derisk_high", 0.80)),
        probabilistic_mode=str(st.session_state.get("universe_micro_prob_mode", "none")),
        probabilistic_q_low=float(st.session_state.get("universe_micro_prob_q_low", 0.25)),
        probabilistic_q_high=float(st.session_state.get("universe_micro_prob_q_high", 0.75)),
        probabilistic_min_obs=int(st.session_state.get("universe_micro_prob_min_obs", 12)),
        probabilistic_interval_penalty_weight=float(st.session_state.get("universe_micro_prob_interval_penalty_weight", 0.25)),
        probabilistic_downside_penalty_weight=float(st.session_state.get("universe_micro_prob_downside_penalty_weight", 0.25)),
        probabilistic_confidence_scale=float(st.session_state.get("universe_micro_prob_confidence_scale", 1.0)),
        probabilistic_confidence_min_mult=float(st.session_state.get("universe_micro_prob_confidence_min_mult", 0.75)),
        probabilistic_confidence_max_mult=float(st.session_state.get("universe_micro_prob_confidence_max_mult", 1.25)),
        probabilistic_overlay_strength=float(st.session_state.get("universe_micro_prob_overlay_strength", 1.0)),
        probabilistic_overlay_blend=float(st.session_state.get("universe_micro_prob_overlay_blend", 1.0)),
        dispersion_sigma_enabled=bool(st.session_state.get("universe_micro_dispersion_sigma_enabled", False)),
        dispersion_sigma_lookback=int(st.session_state.get("universe_micro_dispersion_sigma_lookback", 12)),
        dispersion_sigma_strength=float(st.session_state.get("universe_micro_dispersion_sigma_strength", 0.50)),
        dispersion_sigma_floor_mult=float(st.session_state.get("universe_micro_dispersion_sigma_floor_mult", 0.75)),
        dispersion_sigma_ceiling_mult=float(st.session_state.get("universe_micro_dispersion_sigma_ceiling_mult", 1.50)),
        probabilistic_qr_alpha=float(st.session_state.get("universe_micro_prob_qr_alpha", 0.0)),
        probabilistic_qr_solver=str(st.session_state.get("universe_micro_prob_qr_solver", "highs")),
        probabilistic_qr_feature_cap=int(st.session_state.get("universe_micro_prob_qr_feature_cap", 8)),
        probabilistic_regime_aware=bool(st.session_state.get("universe_micro_prob_regime_aware", False)),
        probabilistic_regime_min_obs=int(st.session_state.get("universe_micro_prob_regime_min_obs", 12)),
        probabilistic_feature_filter_enabled=bool(st.session_state.get("universe_micro_prob_feature_filter_enabled", False)),
        probabilistic_feature_filter_min_obs=int(st.session_state.get("universe_micro_prob_feature_filter_min_obs", 12)),
        probabilistic_feature_filter_k=(None if bool(st.session_state.get("universe_micro_prob_feature_filter_k_none", True)) else int(st.session_state.get("universe_micro_prob_feature_filter_k", 24))),
        probabilistic_knn_k=(None if bool(st.session_state.get("universe_micro_prob_knn_k_none", True)) else int(st.session_state.get("universe_micro_prob_knn_k", 24))),
        probabilistic_knn_min_obs=int(st.session_state.get("universe_micro_prob_knn_min_obs", 12)),
        probabilistic_knn_distance_power=float(st.session_state.get("universe_micro_prob_knn_distance_power", 2.0)),
        probabilistic_knn_weighted_quantiles=bool(st.session_state.get("universe_micro_prob_knn_weighted_quantiles", True)),
        probabilistic_knn_weight_eps=float(st.session_state.get("universe_micro_prob_knn_weight_eps", 1e-6)),
        probabilistic_bucket_n_bins=int(st.session_state.get("universe_micro_prob_bucket_n_bins", 4)),
        probabilistic_bucket_min_obs=int(st.session_state.get("universe_micro_prob_bucket_min_obs", 12)),
        probabilistic_bucket_match_min_features=int(st.session_state.get("universe_micro_prob_bucket_match_min_features", 2)),
        probabilistic_bucket_max_features=int(st.session_state.get("universe_micro_prob_bucket_max_features", 6)),
        probabilistic_parametric_min_sigma=float(st.session_state.get("universe_micro_prob_parametric_min_sigma", 1e-4)),
        probabilistic_parametric_max_sigma_mult=float(st.session_state.get("universe_micro_prob_parametric_max_sigma_mult", 3.0)),
        probabilistic_parametric_use_neighbor_weights=bool(st.session_state.get("universe_micro_prob_parametric_use_neighbor_weights", True)),
        probabilistic_hybrid_weight=float(st.session_state.get("universe_micro_prob_hybrid_weight", 0.50)),
        probabilistic_hybrid_min_qr_weight=float(st.session_state.get("universe_micro_prob_hybrid_min_qr_weight", 0.25)),
        probabilistic_hybrid_max_qr_weight=float(st.session_state.get("universe_micro_prob_hybrid_max_qr_weight", 0.75)),
        probabilistic_hybrid_use_confidence=bool(st.session_state.get("universe_micro_prob_hybrid_use_confidence", True)),
        feature_mu_enabled=bool(st.session_state.get("universe_micro_feature_mu_enabled", True)),
        feature_mu_blend=float(st.session_state.get("universe_micro_feature_mu_blend", 0.25)),
        feature_mu_k=int(st.session_state.get("universe_micro_feature_mu_k", 24)),
        feature_mu_min_obs=int(st.session_state.get("universe_micro_feature_mu_min_obs", 12)),
        feature_mu_cols_max=int(st.session_state.get("universe_micro_feature_mu_cols_max", 8)),
        pure_cs_baseline=bool(st.session_state.get("universe_micro_pure_cs_baseline", False)),
        sigma_floor=float(st.session_state.get("universe_micro_sigma_floor", 1e-8)),
        long_only=bool(st.session_state.get("universe_micro_long_only", True)),
        regime_lookback=int(st.session_state.get("universe_micro_regime_lookback", 24)),
        regime_quantile_low=float(st.session_state.get("universe_micro_regime_quantile_low", 0.33)),
        regime_quantile_high=float(st.session_state.get("universe_micro_regime_quantile_high", 0.67)),
        score_normalize=bool(st.session_state.get("universe_micro_score_normalize", True)),
        score_clip=(float(st.session_state.get("universe_micro_score_clip", 5.0)) if bool(st.session_state.get("universe_micro_score_clip_enabled", False)) else None),
        store_correlation_snapshots=bool(st.session_state.get("universe_micro_store_correlation_snapshots", False)),
        store_sigma_fwd_snapshots=bool(st.session_state.get("universe_micro_store_sigma_fwd_snapshots", False)),
        signal_mode=str(st.session_state.get("universe_micro_signal_mode", "mu_sigma")),
        huber_delta=float(st.session_state.get("universe_micro_huber_delta", 1.0)),
        signal_score_blend=float(st.session_state.get("universe_micro_signal_score_blend", 1.0)),
        lambdarank_lookback=int(st.session_state.get("universe_micro_lambdarank_lookback", 12)),
        lambdarank_temperature=float(st.session_state.get("universe_micro_lambdarank_temperature", 1.0)),
        lambdarank_real_lookback=int(st.session_state.get("universe_micro_lambdarank_real_lookback", 18)),
        lambdarank_real_temperature=float(st.session_state.get("universe_micro_lambdarank_real_temperature", 1.0)),
        lambdarank_real_gain_power=float(st.session_state.get("universe_micro_lambdarank_real_gain_power", 1.0)),
        lambdarank_real_pair_power=float(st.session_state.get("universe_micro_lambdarank_real_pair_power", 1.0)),
        lambdarank_real_l2=float(st.session_state.get("universe_micro_lambdarank_real_l2", 0.0)),
        multi_loss_enabled=bool(st.session_state.get("universe_multi_loss_enabled", False)),
        multi_loss_rank_weight=float(st.session_state.get("universe_multi_loss_rank_weight", 0.40)),
        multi_loss_direction_weight=float(st.session_state.get("universe_multi_loss_direction_weight", 0.30)),
        multi_loss_return_weight=float(st.session_state.get("universe_multi_loss_return_weight", 0.30)),
        multi_loss_logistic_weight=float(st.session_state.get("universe_multi_loss_logistic_weight", 0.0)),
        multi_loss_topk_weight=float(st.session_state.get("universe_multi_loss_topk_weight", 0.0)),
        multi_loss_temperature=float(st.session_state.get("universe_multi_loss_temperature", 1.0)),
        multi_loss_l2=float(st.session_state.get("universe_multi_loss_l2", 0.0)),
        directional_classifier_lookback=int(st.session_state.get("universe_micro_directional_classifier_lookback", 12)),
        directional_classifier_threshold=float(st.session_state.get("universe_micro_directional_classifier_threshold", 0.50)),
        directional_classifier_confidence_scale=float(st.session_state.get("universe_micro_directional_classifier_confidence_scale", 1.0)),
        logistic_loss_lookback=int(st.session_state.get("universe_micro_logistic_loss_lookback", 18)),
        logistic_loss_l2=float(st.session_state.get("universe_micro_logistic_loss_l2", 1e-3)),
        logistic_loss_threshold=float(st.session_state.get("universe_micro_logistic_loss_threshold", 0.50)),
        logistic_loss_confidence_scale=float(st.session_state.get("universe_micro_logistic_loss_confidence_scale", 1.0)),
        top_k_classifier_lookback=int(st.session_state.get("universe_micro_top_k_classifier_lookback", 12)),
        top_k_classifier_k=int(st.session_state.get("universe_micro_top_k_classifier_k", 5)),
        top_k_classifier_threshold=float(st.session_state.get("universe_micro_top_k_classifier_threshold", 0.50)),
        top_k_classifier_confidence_scale=float(st.session_state.get("universe_micro_top_k_classifier_confidence_scale", 1.0)),
        quantile_loss_lookback=int(st.session_state.get("universe_micro_quantile_loss_lookback", 18)),
        quantile_loss_q=float(st.session_state.get("universe_micro_quantile_loss_q", 0.60)),
        quantile_loss_alpha=float(st.session_state.get("universe_micro_quantile_loss_alpha", 0.0)),
        quantile_loss_confidence_scale=float(st.session_state.get("universe_micro_quantile_loss_confidence_scale", 1.0)),
        regime_dependent_universe_enabled=bool(st.session_state.get("universe_regime_universe_enabled", False)),
        regime_universe_selection_metric=str(st.session_state.get("universe_regime_universe_metric", "mu_over_sigma")),
        regime_universe_keep_frac_low=float(st.session_state.get("universe_regime_keep_low", 1.00)),
        regime_universe_keep_frac_mid=float(st.session_state.get("universe_regime_keep_mid", 0.85)),
        regime_universe_keep_frac_high=float(st.session_state.get("universe_regime_keep_high", 0.60)),
        regime_universe_min_assets=int(st.session_state.get("universe_regime_min_assets", 2)),
        regime_universe_max_assets=(int(st.session_state.get("universe_regime_max_assets", 25)) if bool(st.session_state.get("universe_regime_max_assets_enabled", False)) else None),
        regime_universe_low_offensive_vol_tilt=float(st.session_state.get("universe_regime_low_vol_tilt", 0.25)),
        regime_universe_high_defensive_vol_tilt=float(st.session_state.get("universe_regime_high_vol_tilt", 0.75)),
    )

    valid_fields = {f.name for f in fields(MicroPipelineConfig)}

    if st.session_state.get("universe_micro_asset_weight_cap_enabled", False) and "asset_weight_cap" in valid_fields:
        cfg_payload["asset_weight_cap"] = float(st.session_state.get("universe_micro_asset_weight_cap", 0.20))

    if st.session_state.get("universe_micro_w_cap_enabled", False) and "w_cap" in valid_fields:
        cfg_payload["w_cap"] = float(st.session_state.get("universe_micro_w_cap", 0.20))

    if "vol_dependent_cap_enabled" in valid_fields:
        cfg_payload["vol_dependent_cap_enabled"] = bool(st.session_state.get("universe_micro_vol_cap_enabled", False))
    if "vol_dependent_cap_threshold_low" in valid_fields:
        cfg_payload["vol_dependent_cap_threshold_low"] = float(st.session_state.get("universe_micro_vol_cap_threshold_low", 0.03))
    if "vol_dependent_cap_threshold_high" in valid_fields:
        cfg_payload["vol_dependent_cap_threshold_high"] = float(st.session_state.get("universe_micro_vol_cap_threshold_high", 0.06))
    if "vol_dependent_cap_low_mult" in valid_fields:
        cfg_payload["vol_dependent_cap_low_mult"] = float(st.session_state.get("universe_micro_vol_cap_low_mult", 1.15))
    if "vol_dependent_cap_high_mult" in valid_fields:
        cfg_payload["vol_dependent_cap_high_mult"] = float(st.session_state.get("universe_micro_vol_cap_high_mult", 0.80))
    if "vol_dependent_cap_min" in valid_fields:
        cfg_payload["vol_dependent_cap_min"] = (float(st.session_state.get("universe_micro_vol_cap_min", 0.0)) if bool(st.session_state.get("universe_micro_vol_cap_min_enabled", False)) else None)
    if "vol_dependent_cap_max" in valid_fields:
        cfg_payload["vol_dependent_cap_max"] = (float(st.session_state.get("universe_micro_vol_cap_max", 1.0)) if bool(st.session_state.get("universe_micro_vol_cap_max_enabled", False)) else None)

    if "corr_dependent_cap_enabled" in valid_fields:
        cfg_payload["corr_dependent_cap_enabled"] = bool(st.session_state.get("universe_micro_corr_cap_enabled", False))
    if "corr_dependent_cap_threshold_low" in valid_fields:
        cfg_payload["corr_dependent_cap_threshold_low"] = float(st.session_state.get("universe_micro_corr_cap_threshold_low", 0.20))
    if "corr_dependent_cap_threshold_high" in valid_fields:
        cfg_payload["corr_dependent_cap_threshold_high"] = float(st.session_state.get("universe_micro_corr_cap_threshold_high", 0.60))
    if "corr_dependent_cap_low_mult" in valid_fields:
        cfg_payload["corr_dependent_cap_low_mult"] = float(st.session_state.get("universe_micro_corr_cap_low_mult", 1.10))
    if "corr_dependent_cap_high_mult" in valid_fields:
        cfg_payload["corr_dependent_cap_high_mult"] = float(st.session_state.get("universe_micro_corr_cap_high_mult", 0.80))
    if "corr_dependent_cap_min" in valid_fields:
        cfg_payload["corr_dependent_cap_min"] = (float(st.session_state.get("universe_micro_corr_cap_min", 0.0)) if bool(st.session_state.get("universe_micro_corr_cap_min_enabled", False)) else None)
    if "corr_dependent_cap_max" in valid_fields:
        cfg_payload["corr_dependent_cap_max"] = (float(st.session_state.get("universe_micro_corr_cap_max", 1.0)) if bool(st.session_state.get("universe_micro_corr_cap_max_enabled", False)) else None)

    if "dispersion_dependent_cap_enabled" in valid_fields:
        cfg_payload["dispersion_dependent_cap_enabled"] = bool(st.session_state.get("universe_micro_disp_cap_enabled", False))
    if "dispersion_dependent_cap_threshold_low" in valid_fields:
        cfg_payload["dispersion_dependent_cap_threshold_low"] = float(st.session_state.get("universe_micro_disp_cap_threshold_low", 0.10))
    if "dispersion_dependent_cap_threshold_high" in valid_fields:
        cfg_payload["dispersion_dependent_cap_threshold_high"] = float(st.session_state.get("universe_micro_disp_cap_threshold_high", 0.30))
    if "dispersion_dependent_cap_low_mult" in valid_fields:
        cfg_payload["dispersion_dependent_cap_low_mult"] = float(st.session_state.get("universe_micro_disp_cap_low_mult", 1.10))
    if "dispersion_dependent_cap_high_mult" in valid_fields:
        cfg_payload["dispersion_dependent_cap_high_mult"] = float(st.session_state.get("universe_micro_disp_cap_high_mult", 0.80))
    if "dispersion_dependent_cap_min" in valid_fields:
        cfg_payload["dispersion_dependent_cap_min"] = (float(st.session_state.get("universe_micro_disp_cap_min", 0.0)) if bool(st.session_state.get("universe_micro_disp_cap_min_enabled", False)) else None)
    if "dispersion_dependent_cap_max" in valid_fields:
        cfg_payload["dispersion_dependent_cap_max"] = (float(st.session_state.get("universe_micro_disp_cap_max", 1.0)) if bool(st.session_state.get("universe_micro_disp_cap_max_enabled", False)) else None)

    if "regime_dependent_cap_enabled" in valid_fields:
        cfg_payload["regime_dependent_cap_enabled"] = bool(st.session_state.get("universe_micro_regime_cap_enabled", False))
    if "regime_dependent_cap_low_mult" in valid_fields:
        cfg_payload["regime_dependent_cap_low_mult"] = float(st.session_state.get("universe_micro_regime_cap_low_mult", 1.05))
    if "regime_dependent_cap_mid_mult" in valid_fields:
        cfg_payload["regime_dependent_cap_mid_mult"] = float(st.session_state.get("universe_micro_regime_cap_mid_mult", 0.95))
    if "regime_dependent_cap_high_mult" in valid_fields:
        cfg_payload["regime_dependent_cap_high_mult"] = float(st.session_state.get("universe_micro_regime_cap_high_mult", 0.80))
    if "regime_dependent_cap_min" in valid_fields:
        cfg_payload["regime_dependent_cap_min"] = (float(st.session_state.get("universe_micro_regime_cap_min", 0.0)) if bool(st.session_state.get("universe_micro_regime_cap_min_enabled", False)) else None)
    if "regime_dependent_cap_max" in valid_fields:
        cfg_payload["regime_dependent_cap_max"] = (float(st.session_state.get("universe_micro_regime_cap_max", 1.0)) if bool(st.session_state.get("universe_micro_regime_cap_max_enabled", False)) else None)

    if "turnover_penalty_strength" in valid_fields:
        cfg_payload["turnover_penalty_strength"] = float(st.session_state.get("universe_micro_turnover_penalty_strength", 0.0))
    if "turnover_penalty_power" in valid_fields:
        cfg_payload["turnover_penalty_power"] = float(st.session_state.get("universe_micro_turnover_penalty_power", 1.0))
    if "turnover_penalty_target" in valid_fields:
        cfg_payload["turnover_penalty_target"] = float(st.session_state.get("universe_micro_turnover_penalty_target", 0.20))
    if "turnover_penalty_max_turnover" in valid_fields:
        cfg_payload["turnover_penalty_max_turnover"] = (float(st.session_state.get("universe_micro_turnover_penalty_max_turnover", 0.35)) if bool(st.session_state.get("universe_micro_turnover_penalty_max_turnover_enabled", False)) else None)
    if "turnover_constraint_max_turnover" in valid_fields:
        cfg_payload["turnover_constraint_max_turnover"] = (float(st.session_state.get("universe_micro_turnover_constraint_max_turnover", 0.35)) if bool(st.session_state.get("universe_micro_turnover_constraint_enabled", False)) else None)

    if "dispersion_gate" in valid_fields:
        cfg_payload["dispersion_gate"] = bool(st.session_state.get("universe_micro_dispersion_gate", True))
    if "dispersion_gate_threshold" in valid_fields:
        cfg_payload["dispersion_gate_threshold"] = float(st.session_state.get("universe_micro_dispersion_gate_threshold", 0.10))
    if "dispersion_gate_min_active_weight" in valid_fields:
        cfg_payload["dispersion_gate_min_active_weight"] = float(st.session_state.get("universe_micro_dispersion_gate_min_active_weight", 0.25))
    if "dispersion_risk_model_enabled" in valid_fields:
        cfg_payload["dispersion_risk_model_enabled"] = bool(st.session_state.get("universe_micro_dispersion_risk_model_enabled", False))
    if "dispersion_risk_strength" in valid_fields:
        cfg_payload["dispersion_risk_strength"] = float(st.session_state.get("universe_micro_dispersion_risk_strength", 0.50))
    if "dispersion_risk_floor_mult" in valid_fields:
        cfg_payload["dispersion_risk_floor_mult"] = float(st.session_state.get("universe_micro_dispersion_risk_floor_mult", 0.85))
    if "dispersion_risk_ceiling_mult" in valid_fields:
        cfg_payload["dispersion_risk_ceiling_mult"] = float(st.session_state.get("universe_micro_dispersion_risk_ceiling_mult", 1.25))
    if "dispersion_risk_apply_to_covariance" in valid_fields:
        cfg_payload["dispersion_risk_apply_to_covariance"] = bool(st.session_state.get("universe_micro_dispersion_risk_apply_to_covariance", True))
    if "dispersion_top_k_enabled" in valid_fields:
        cfg_payload["dispersion_top_k_enabled"] = bool(st.session_state.get("universe_micro_dispersion_top_k_enabled", False))
    if "dispersion_top_k_threshold_low" in valid_fields:
        cfg_payload["dispersion_top_k_threshold_low"] = float(st.session_state.get("universe_micro_dispersion_top_k_threshold_low", 0.10))
    if "dispersion_top_k_threshold_high" in valid_fields:
        cfg_payload["dispersion_top_k_threshold_high"] = float(st.session_state.get("universe_micro_dispersion_top_k_threshold_high", 0.30))
    if "dispersion_top_k_low_mult" in valid_fields:
        cfg_payload["dispersion_top_k_low_mult"] = float(st.session_state.get("universe_micro_dispersion_top_k_low_mult", 1.50))
    if "dispersion_top_k_high_mult" in valid_fields:
        cfg_payload["dispersion_top_k_high_mult"] = float(st.session_state.get("universe_micro_dispersion_top_k_high_mult", 0.75))
    if "dispersion_top_k_min_k" in valid_fields:
        cfg_payload["dispersion_top_k_min_k"] = (int(st.session_state.get("universe_micro_dispersion_top_k_min_k", 1)) if bool(st.session_state.get("universe_micro_dispersion_top_k_min_k_enabled", False)) else None)
    if "dispersion_top_k_max_k" in valid_fields:
        cfg_payload["dispersion_top_k_max_k"] = (int(st.session_state.get("universe_micro_dispersion_top_k_max_k", 10)) if bool(st.session_state.get("universe_micro_dispersion_top_k_max_k_enabled", False)) else None)
    if "dispersion_vol_target_enabled" in valid_fields:
        cfg_payload["dispersion_vol_target_enabled"] = bool(st.session_state.get("universe_micro_dispersion_vol_target_enabled", False))
    if "dispersion_vol_target_threshold_low" in valid_fields:
        cfg_payload["dispersion_vol_target_threshold_low"] = float(st.session_state.get("universe_micro_dispersion_vol_target_threshold_low", 0.10))
    if "dispersion_vol_target_threshold_high" in valid_fields:
        cfg_payload["dispersion_vol_target_threshold_high"] = float(st.session_state.get("universe_micro_dispersion_vol_target_threshold_high", 0.30))
    if "dispersion_vol_target_low_mult" in valid_fields:
        cfg_payload["dispersion_vol_target_low_mult"] = float(st.session_state.get("universe_micro_dispersion_vol_target_low_mult", 0.75))
    if "dispersion_vol_target_high_mult" in valid_fields:
        cfg_payload["dispersion_vol_target_high_mult"] = float(st.session_state.get("universe_micro_dispersion_vol_target_high_mult", 1.15))
    if "dispersion_vol_target_min" in valid_fields:
        cfg_payload["dispersion_vol_target_min"] = (float(st.session_state.get("universe_micro_dispersion_vol_target_min", 0.01)) if bool(st.session_state.get("universe_micro_dispersion_vol_target_min_enabled", False)) else None)
    if "dispersion_vol_target_max" in valid_fields:
        cfg_payload["dispersion_vol_target_max"] = (float(st.session_state.get("universe_micro_dispersion_vol_target_max", 0.25)) if bool(st.session_state.get("universe_micro_dispersion_vol_target_max_enabled", False)) else None)
    if "low_signal_fallback_to_ew" in valid_fields:
        cfg_payload["low_signal_fallback_to_ew"] = bool(st.session_state.get("universe_micro_low_signal_fallback_to_ew", False))
    if "low_signal_fallback_threshold" in valid_fields:
        cfg_payload["low_signal_fallback_threshold"] = float(st.session_state.get("universe_micro_low_signal_fallback_threshold", 0.05))
    if "low_signal_fallback_mode" in valid_fields:
        cfg_payload["low_signal_fallback_mode"] = str(st.session_state.get("universe_micro_low_signal_fallback_mode", "blend"))
    if "low_signal_fallback_min_model_weight" in valid_fields:
        cfg_payload["low_signal_fallback_min_model_weight"] = float(st.session_state.get("universe_micro_low_signal_fallback_min_model_weight", 0.25))

    if "cost_model_enabled" in valid_fields:
        cfg_payload["cost_model_enabled"] = bool(st.session_state.get("universe_cost_model_enabled", False))
    if "transaction_cost_commission_bps" in valid_fields:
        cfg_payload["transaction_cost_commission_bps"] = float(st.session_state.get("universe_transaction_cost_commission_bps", 0.0))
    if "transaction_cost_slippage_bps" in valid_fields:
        cfg_payload["transaction_cost_slippage_bps"] = float(st.session_state.get("universe_transaction_cost_slippage_bps", 0.0))
    if "transaction_cost_spread_bps" in valid_fields:
        cfg_payload["transaction_cost_spread_bps"] = float(st.session_state.get("universe_transaction_cost_spread_bps", 0.0))
    if "transaction_cost_market_impact_bps" in valid_fields:
        cfg_payload["transaction_cost_market_impact_bps"] = float(st.session_state.get("universe_transaction_cost_market_impact_bps", 0.0))
    if "transaction_cost_market_impact_power" in valid_fields:
        cfg_payload["transaction_cost_market_impact_power"] = float(st.session_state.get("universe_transaction_cost_market_impact_power", 1.0))
    if "transaction_cost_min_trade_weight" in valid_fields:
        cfg_payload["transaction_cost_min_trade_weight"] = float(st.session_state.get("universe_transaction_cost_min_trade_weight", 0.0))
    if "holding_cost_annual_bps" in valid_fields:
        cfg_payload["holding_cost_annual_bps"] = float(st.session_state.get("universe_holding_cost_annual_bps", 0.0))
    if "tax_model_enabled" in valid_fields:
        cfg_payload["tax_model_enabled"] = bool(st.session_state.get("universe_tax_model_enabled", False))
    if "tax_short_term_rate" in valid_fields:
        cfg_payload["tax_short_term_rate"] = float(st.session_state.get("universe_tax_short_term_rate", 0.0))
    if "tax_long_term_rate" in valid_fields:
        long_rate = float(st.session_state.get("universe_tax_long_term_rate", 0.0))
        cfg_payload["tax_long_term_rate"] = long_rate
    if "tax_long_term_threshold_months" in valid_fields:
        cfg_payload["tax_long_term_threshold_months"] = int(st.session_state.get("universe_tax_long_term_threshold_months", 12))
    if "tax_apply_loss_credit" in valid_fields:
        cfg_payload["tax_apply_loss_credit"] = bool(st.session_state.get("universe_tax_apply_loss_credit", False))
    if "tax_loss_credit_rate" in valid_fields:
        cfg_payload["tax_loss_credit_rate"] = float(st.session_state.get("universe_tax_loss_credit_rate", 0.0))

    if "factor_model_active" in valid_fields:
        cfg_payload["factor_model_active"] = bool(st.session_state.get("universe_factor_model_enabled", False))
    if "factor_covariance_active" in valid_fields:
        cfg_payload["factor_covariance_active"] = bool(st.session_state.get("universe_factor_covariance_enabled", False))
    if "factor_model_n_factors" in valid_fields:
        cfg_payload["factor_model_n_factors"] = int(st.session_state.get("universe_factor_model_n_factors", 3))
    if "factor_model_min_obs" in valid_fields:
        cfg_payload["factor_model_min_obs"] = int(st.session_state.get("universe_factor_model_min_obs", 24))
    if "factor_model_mu_blend" in valid_fields:
        cfg_payload["factor_model_mu_blend"] = float(st.session_state.get("universe_factor_model_mu_blend", 0.35))
    if "factor_model_residual_blend" in valid_fields:
        cfg_payload["factor_model_residual_blend"] = float(st.session_state.get("universe_factor_model_residual_blend", 0.50))
    if "factor_model_covariance_blend" in valid_fields:
        cfg_payload["factor_model_covariance_blend"] = float(st.session_state.get("universe_factor_model_covariance_blend", 0.60))
    if "factor_model_shrink_to_diagonal" in valid_fields:
        cfg_payload["factor_model_shrink_to_diagonal"] = float(st.session_state.get("universe_factor_model_shrink_to_diagonal", 0.10))
    if "factor_covariance_n_factors" in valid_fields:
        cfg_payload["factor_covariance_n_factors"] = int(st.session_state.get("universe_factor_covariance_n_factors", 3))
    if "factor_covariance_min_obs" in valid_fields:
        cfg_payload["factor_covariance_min_obs"] = int(st.session_state.get("universe_factor_covariance_min_obs", 24))
    if "factor_covariance_blend" in valid_fields:
        cfg_payload["factor_covariance_blend"] = float(st.session_state.get("universe_factor_covariance_blend", 0.50))
    if "factor_covariance_shrink_to_diagonal" in valid_fields:
        cfg_payload["factor_covariance_shrink_to_diagonal"] = float(st.session_state.get("universe_factor_covariance_shrink_to_diagonal", 0.10))

    return MicroPipelineConfig(**{k: v for k, v in cfg_payload.items() if k in valid_fields})


def apply_engine_preset(preset_name: str) -> None:
    preset = ENGINE_PRESETS.get(str(preset_name), {})
    if not preset:
        st.warning(f"Preset not found: {preset_name}")
        return

    pending_updates = {}
    for cfg_name, session_key in ENGINE_PRESET_SESSION_MAP.items():
        if cfg_name in preset:
            pending_updates[session_key] = preset[cfg_name]

    if "asset_weight_cap" in preset:
        pending_updates["universe_micro_asset_weight_cap_enabled"] = True
        pending_updates["universe_micro_asset_weight_cap"] = float(preset["asset_weight_cap"])
    if "w_cap" in preset:
        pending_updates["universe_micro_w_cap_enabled"] = True
        pending_updates["universe_micro_w_cap"] = float(preset["w_cap"])

    if "top_k" in preset:
        top_k_val = preset.get("top_k")
        pending_updates["universe_micro_top_k_none"] = top_k_val is None
        if top_k_val is not None:
            pending_updates["universe_micro_top_k"] = int(top_k_val)
    else:
        pending_updates["universe_micro_top_k_none"] = True

    pending_updates["universe_engine_preset_last_applied"] = str(preset_name)
    _queue_session_updates(pending_updates)
    _queue_toast(f"Engine preset applied: {preset_name}", icon="✅")
    st.rerun()




def _cfg_payload_from_trial_row(base_payload: dict, trial_row: pd.Series | dict | None) -> dict:
    valid_fields = {f.name for f in fields(MicroPipelineConfig)}
    payload = {k: v for k, v in _coerce_mapping(base_payload).items() if k in valid_fields}
    if trial_row is None:
        return payload
    row = _row_to_dict(trial_row)
    tuned_param = str(row.get("tuned_param", "") or "")
    candidate_kind = str(row.get("candidate_kind", "") or "")
    if tuned_param and tuned_param != "baseline" and candidate_kind != "baseline":
        payload[tuned_param] = row.get("candidate_value")
    return payload


def _payload_json(payload: dict) -> str:
    try:
        return json.dumps(_coerce_mapping(payload), sort_keys=True, default=str)
    except Exception:
        return "{}"


def _payload_from_json(raw: str | None) -> dict:
    try:
        obj = json.loads(str(raw or "{}"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _annotate_local_search_stage(local_table: pd.DataFrame, *, stage_name: str, base_payload: dict) -> pd.DataFrame:
    if not isinstance(local_table, pd.DataFrame) or local_table.empty:
        return pd.DataFrame()
    out = local_table.copy()
    payload_jsons = []
    fingerprints = []
    for _, row in out.iterrows():
        payload = _cfg_payload_from_trial_row(base_payload, row)
        payload_jsons.append(_payload_json(payload))
        try:
            fingerprints.append(config_fingerprint(MicroPipelineConfig(**payload)))
        except Exception:
            fingerprints.append(None)
    out["refinement_stage"] = str(stage_name)
    out["effective_cfg_payload_json"] = payload_jsons
    out["effective_config_fingerprint"] = fingerprints
    return out




TUNING_OBJECTIVE_OPTIONS = [
    "sharpe",
    "cagr",
    "information_ratio",
    "min_mean_turnover",
    "-annual_volatility",
    "composite_balanced",
    "composite_growth",
    "composite_defensive",
    "composite_quality",
    "composite_low_turnover",
    "composite_diversified",
    "composite_robust",
]


def _is_composite_objective(name: str | None) -> bool:
    key = str(name or "").strip().lower()
    return key.startswith("composite_")


def _format_tuning_objective_label(name: str) -> str:
    labels = {
        "sharpe": "Sharpe",
        "cagr": "CAGR",
        "information_ratio": "Information ratio",
        "min_mean_turnover": "Min mean turnover",
        "-annual_volatility": "Min annual volatility",
        "composite_balanced": "Composite — Balanced",
        "composite_growth": "Composite — Growth",
        "composite_defensive": "Composite — Defensive",
        "composite_quality": "Composite — Quality",
        "composite_low_turnover": "Composite — Low turnover",
        "composite_diversified": "Composite — Diversified",
        "composite_robust": "Composite — Robust",
    }
    key = str(name or "").strip()
    return labels.get(key, key.replace("_", " ").title())

def _search_budget_to_intensity(search_budget: str | None, *, default: str = "Quick") -> str:
    budget = str(search_budget or "").strip().lower()
    if budget == "expanded":
        return "Standard"
    if budget in {"light", "standard"}:
        return "Quick" if budget == "light" else "Standard"
    return str(default)


def _build_local_search_space_for_intensity(
    cfg: MicroPipelineConfig,
    panel_df: pd.DataFrame,
    *,
    intensity: str = "Quick",
    include_weight_shrink: bool = False,
) -> tuple[dict, int]:
    """Build a compact local-search grid and a budget from a UI intensity label.

    This helper stays in app.py because the UI still exposes a simple intensity
    control (Quick / Standard). The actual search logic remains in evaluation.py.
    """
    intensity_key = str(intensity or "Quick").strip().lower()
    if intensity_key == "standard":
        param_space = build_local_search_param_space(
            config_to_dict(cfg),
            panel_df=panel_df,
            temperature_multipliers=(0.8, 1.0, 1.2),
            top_k_offsets=(-2, 0, 2),
            overlay_multipliers=(0.7, 1.0, 1.3),
            target_vol_multipliers=(0.8, 1.0, 1.2),
            turnover_penalty_multipliers=(0.7, 1.0, 1.3),
            weight_shrink_multipliers=((0.7, 1.0, 1.3) if include_weight_shrink else ()),
        )
        return param_space, 32

    param_space = build_local_search_param_space(
        config_to_dict(cfg),
        panel_df=panel_df,
        temperature_multipliers=(0.9, 1.0, 1.1),
        top_k_offsets=(-1, 0, 1),
        overlay_multipliers=(0.85, 1.0, 1.15),
        target_vol_multipliers=(0.9, 1.0, 1.1),
        turnover_penalty_multipliers=(0.85, 1.0, 1.15),
        weight_shrink_multipliers=((0.85, 1.0, 1.15) if include_weight_shrink else ()),
    )
    return param_space, 18


def _maybe_apply_simple_mode_local_search_defaults() -> dict:
    if not bool(st.session_state.get("universe_simple_mode_enabled", True)):
        return {}
    resolved = _coerce_mapping(st.session_state.get("universe_simple_resolved_summary", {}))
    tuning_policy = _coerce_mapping(_coerce_mapping(resolved).get("tuning_policy", {}))
    if not tuning_policy:
        return {}
    st.session_state["universe_local_search_policy_objective"] = str(tuning_policy.get("objective", "") or "")
    st.session_state["universe_local_search_policy_two_stage_search"] = bool(tuning_policy.get("two_stage_search", False))
    st.session_state["universe_local_search_policy_search_budget"] = str(tuning_policy.get("search_budget", ""))
    st.session_state["universe_local_search_policy_max_candidates_hint"] = int(tuning_policy.get("max_candidates_hint", 0) or 0)
    return tuning_policy


def _resolve_simple_mode_composite_profile() -> str:
    style = str(st.session_state.get("universe_simple_style_preset", DEFAULT_COMPOSITE_PROFILE)).strip().lower()
    style_to_profile = {
        "balanced": "balanced",
        "growth": "growth",
        "defensive": "defensive",
        "conservative": "defensive",
        "research": "growth",
    }
    resolved = str(style_to_profile.get(style, DEFAULT_COMPOSITE_PROFILE)).strip().lower() or DEFAULT_COMPOSITE_PROFILE
    if resolved not in COMPOSITE_PROFILE_OPTIONS:
        resolved = DEFAULT_COMPOSITE_PROFILE
    return resolved


def _maybe_apply_simple_mode_selection_policy_defaults() -> dict:
    if not bool(st.session_state.get("universe_simple_mode_enabled", True)):
        return {}
    composite_profile = _resolve_simple_mode_composite_profile()
    selection_defaults = {
        "selection_policy": "fixed_composite_score",
        "composite_profile": composite_profile,
    }
    st.session_state["universe_local_search_mode"] = "multiobjective_pareto"
    st.session_state["universe_local_search_solution_picker"] = selection_defaults["selection_policy"]
    st.session_state["universe_local_search_composite_profile"] = selection_defaults["composite_profile"]
    st.session_state["universe_adv_tuning_nsga2_selection_policy"] = selection_defaults["selection_policy"]
    st.session_state["universe_adv_tuning_nsga2_composite_profile"] = selection_defaults["composite_profile"]
    st.session_state["universe_simple_auto_selection_policy"] = selection_defaults["selection_policy"]
    st.session_state["universe_simple_auto_composite_profile"] = selection_defaults["composite_profile"]
    return selection_defaults


def _run_local_refinement_search(
    panel_df: pd.DataFrame,
    cfg: MicroPipelineConfig,
    *,
    objective: str = "sharpe",
    intensity: str = "Quick",
    include_weight_shrink: bool = False,
    two_stage: bool = False,
    local_search_policy: dict | None = None,
    max_candidates_override: int | None = None,
) -> dict:
    base_payload = config_to_dict(cfg)
    policy = _coerce_mapping(local_search_policy)
    objective_effective = str(policy.get("objective") or objective)
    two_stage_effective = bool(policy.get("two_stage_search", two_stage))
    local_space_raw, local_budget = _build_local_search_space_for_intensity(
        cfg,
        panel_df,
        intensity=intensity,
        include_weight_shrink=include_weight_shrink,
    )
    local_space_effective = build_local_search_param_space_from_policy(
        base_payload,
        policy=policy,
        panel_df=panel_df,
        raw_param_space=local_space_raw,
        include_weight_shrink=include_weight_shrink,
    )
    local_budget_effective = int(max_candidates_override) if max_candidates_override is not None else int(local_budget)
    stage1_df = auto_tune_around_config(
        panel_df,
        base_cfg_payload=base_payload,
        objective=objective_effective,
        param_space=local_space_effective,
        include_weight_shrink=include_weight_shrink,
        max_candidates=local_budget_effective,
        local_search_policy=policy or None,
    )
    stage1_table = _annotate_local_search_stage(stage1_df, stage_name="stage_1", base_payload=base_payload) if isinstance(stage1_df, pd.DataFrame) and not stage1_df.empty else pd.DataFrame()
    stage2_table = pd.DataFrame()
    stage2_base_payload = None

    best_stage1 = None
    if isinstance(stage1_table, pd.DataFrame) and not stage1_table.empty:
        best_stage1, stage1_table = choose_best_tuning_trial(stage1_table, objective_col="objective_value", higher_is_better=True)

    if bool(two_stage_effective) and best_stage1 is not None:
        stage2_base_payload = _payload_from_json(best_stage1.get("effective_cfg_payload_json"))
        if stage2_base_payload:
            try:
                stage2_cfg = MicroPipelineConfig(**stage2_base_payload)
                stage2_space_raw, stage2_budget = _build_local_search_space_for_intensity(
                    stage2_cfg,
                    panel_df,
                    intensity="Quick",
                    include_weight_shrink=include_weight_shrink,
                )
                stage2_space_effective = build_local_search_param_space_from_policy(
                    stage2_base_payload,
                    policy=policy,
                    panel_df=panel_df,
                    raw_param_space=stage2_space_raw,
                    include_weight_shrink=include_weight_shrink,
                )
                stage2_budget_effective = min(int(local_budget_effective), int(stage2_budget), 12)
                stage2_df = auto_tune_around_config(
                    panel_df,
                    base_cfg_payload=stage2_base_payload,
                    objective=objective_effective,
                    param_space=stage2_space_effective,
                    include_weight_shrink=include_weight_shrink,
                    max_candidates=stage2_budget_effective,
                    local_search_policy=policy or None,
                )
                if isinstance(stage2_df, pd.DataFrame) and not stage2_df.empty:
                    stage2_table = _annotate_local_search_stage(stage2_df, stage_name="stage_2", base_payload=stage2_base_payload)
            except Exception:
                stage2_table = pd.DataFrame()

    tables = [df for df in [stage1_table, stage2_table] if isinstance(df, pd.DataFrame) and not df.empty]
    combined = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    best_overall = None
    if isinstance(combined, pd.DataFrame) and not combined.empty:
        best_overall, combined = choose_best_tuning_trial(combined, objective_col="objective_value", higher_is_better=True)

    stage1_baseline = pd.DataFrame()
    if isinstance(stage1_table, pd.DataFrame) and not stage1_table.empty:
        stage1_baseline = stage1_table[stage1_table.get("candidate_kind", pd.Series(dtype=str)).astype(str).eq("baseline")].head(1)

    return {
        "combined_table": combined,
        "stage1_table": stage1_table,
        "stage2_table": stage2_table,
        "best_overall": best_overall,
        "best_stage1": best_stage1,
        "stage1_baseline": stage1_baseline.iloc[0] if not stage1_baseline.empty else None,
        "base_payload": base_payload,
        "stage2_base_payload": stage2_base_payload,
        "objective_effective": objective_effective,
        "two_stage_effective": bool(two_stage_effective),
        "policy": policy,
        "local_space_effective": local_space_effective,
        "max_candidates_effective": int(local_budget_effective),
    }


def _summarize_cfg_payload_diff(base_payload: dict, candidate_payload: dict, *, limit: int = 8) -> list[dict[str, object]]:
    valid_fields = {f.name for f in fields(MicroPipelineConfig)}
    base = {k: v for k, v in _coerce_mapping(base_payload).items() if k in valid_fields}
    cand = {k: v for k, v in _coerce_mapping(candidate_payload).items() if k in valid_fields}
    rows: list[dict[str, object]] = []
    for key in sorted(set(base) | set(cand)):
        left = base.get(key)
        right = cand.get(key)
        same = False
        try:
            if pd.isna(left) and pd.isna(right):
                same = True
        except Exception:
            pass
        if not same:
            same = left == right
        if same:
            continue
        rows.append({"param": str(key), "baseline": left, "optimized": right})
    return rows[: max(int(limit), 0)]


def _maybe_auto_optimize_simple_config(
    panel_df: pd.DataFrame,
    cfg_base: MicroPipelineConfig,
    simple_summary: dict | None,
) -> dict:
    base_payload = config_to_dict(cfg_base)
    summary = _coerce_mapping(simple_summary)
    tuning_policy = _coerce_mapping(summary.get("tuning_policy", {}))
    result = {
        "final_cfg": cfg_base,
        "baseline_cfg": cfg_base,
        "baseline_payload": base_payload,
        "best_payload": {},
        "tuning_result": {},
        "was_optimized": False,
        "used_auto_opt": False,
        "objective_name": str(tuning_policy.get("objective", "") or ""),
        "objective_improvement": np.nan,
        "trials_explored": 0,
        "changed_params": [],
        "selection_policy": str(tuning_policy.get("simple_auto_selection_policy") or tuning_policy.get("selection_policy") or "fixed_composite_score"),
        "composite_profile": tuning_policy.get("composite_profile", _resolve_simple_mode_composite_profile()),
        "failure_reason": "",
    }

    if not bool(st.session_state.get("universe_simple_mode_enabled", True)):
        result["failure_reason"] = "simple_mode_off"
        return result
    if not bool(st.session_state.get("universe_simple_auto_optimize_enabled", False)):
        result["failure_reason"] = "auto_opt_off"
        return result

    result["used_auto_opt"] = True
    if panel_df is None or not isinstance(panel_df, pd.DataFrame) or panel_df.empty:
        result["failure_reason"] = "empty_panel"
        return result
    if not tuning_policy:
        result["failure_reason"] = "missing_tuning_policy"
        return result

    try:
        tuning_result = run_simple_auto_optimize(
            panel_df,
            base_cfg_payload=base_payload,
            simple_summary=summary,
            include_weight_shrink=False,
        )
    except Exception as exc:
        result["failure_reason"] = f"search_failed: {exc}"
        return result

    result["tuning_result"] = tuning_result
    result["objective_name"] = str(tuning_result.get("objective_name") or result["objective_name"] or "")
    result["selection_policy"] = str(
        tuning_policy.get("simple_auto_selection_policy")
        or tuning_policy.get("selection_policy")
        or "fixed_composite_score"
    )
    result["composite_profile"] = tuning_policy.get("composite_profile", result["composite_profile"])

    baseline_payload = _coerce_mapping(tuning_result.get("baseline_payload")) or base_payload
    final_payload = _coerce_mapping(tuning_result.get("final_cfg_payload")) or baseline_payload
    best_payload = _coerce_mapping(tuning_result.get("best_payload"))
    result["baseline_payload"] = baseline_payload
    result["best_payload"] = best_payload

    trials_df = tuning_result.get("trials_df", pd.DataFrame())
    selection_table = tuning_result.get("selection_table", pd.DataFrame())
    candidate_rows = selection_table if isinstance(selection_table, pd.DataFrame) and not selection_table.empty else trials_df
    if isinstance(candidate_rows, pd.DataFrame) and not candidate_rows.empty:
        n_rows = int(len(candidate_rows))
        if "candidate_kind" in candidate_rows.columns:
            try:
                baseline_mask = candidate_rows["candidate_kind"].astype(str).eq("baseline")
                n_rows = max(int(n_rows - int(baseline_mask.sum())), 0)
            except Exception:
                pass
        result["trials_explored"] = int(n_rows)

    baseline_row = tuning_result.get("baseline_row")
    best_row = tuning_result.get("best_row")
    base_objective = _safe_float(_row_to_dict(baseline_row).get("objective_value", np.nan))
    best_objective = _safe_float(_row_to_dict(best_row).get("objective_value", np.nan))
    improvement = best_objective - base_objective if np.isfinite(best_objective) and np.isfinite(base_objective) else np.nan
    if not np.isfinite(improvement):
        improvement = _safe_float(tuning_result.get("objective_improvement"))
    result["objective_improvement"] = improvement

    try:
        result["baseline_cfg"] = MicroPipelineConfig(**baseline_payload)
    except Exception:
        result["baseline_cfg"] = cfg_base

    was_optimized = bool(tuning_result.get("was_optimized", False))
    if not was_optimized:
        result["failure_reason"] = str(tuning_result.get("optimization_reason") or "no_material_improvement")
        return result

    try:
        final_cfg = MicroPipelineConfig(**final_payload)
    except Exception as exc:
        result["failure_reason"] = f"coerce_failed: {exc}"
        return result

    result["final_cfg"] = final_cfg
    result["was_optimized"] = True
    result["changed_params"] = _summarize_cfg_payload_diff(baseline_payload, final_payload, limit=8)
    return result


def _queue_promote_cfg_payload_to_manual_state(cfg_payload: dict, *, source_label: str = "Local refinement") -> None:
    payload = {k: v for k, v in _coerce_mapping(cfg_payload).items() if k in {f.name for f in fields(MicroPipelineConfig)}}
    pending_updates = {"universe_simple_mode_enabled": False}

    special_nullable_fields = {
        "top_k",
        "asset_weight_cap",
        "w_cap",
        "score_clip",
        "probabilistic_feature_filter_k",
        "probabilistic_knn_k",
        "top_k_classifier_k",
    }
    for cfg_name, session_key in ENGINE_PRESET_SESSION_MAP.items():
        if cfg_name in payload and cfg_name not in special_nullable_fields:
            pending_updates[session_key] = payload[cfg_name]

    top_k_val = payload.get("top_k")
    pending_updates["universe_micro_top_k_none"] = top_k_val is None
    if top_k_val is not None:
        pending_updates["universe_micro_top_k"] = int(top_k_val)

    feature_filter_k_val = payload.get("probabilistic_feature_filter_k")
    pending_updates["universe_micro_prob_feature_filter_k_none"] = feature_filter_k_val is None
    if feature_filter_k_val is not None:
        pending_updates["universe_micro_prob_feature_filter_k"] = int(feature_filter_k_val)

    knn_k_val = payload.get("probabilistic_knn_k")
    pending_updates["universe_micro_prob_knn_k_none"] = knn_k_val is None
    if knn_k_val is not None:
        pending_updates["universe_micro_prob_knn_k"] = int(knn_k_val)

    top_k_classifier_k_val = payload.get("top_k_classifier_k")
    if top_k_classifier_k_val is not None:
        pending_updates["universe_micro_top_k_classifier_k"] = int(top_k_classifier_k_val)

    if "asset_weight_cap" in payload:
        pending_updates["universe_micro_asset_weight_cap_enabled"] = payload.get("asset_weight_cap") is not None
        if payload.get("asset_weight_cap") is not None:
            pending_updates["universe_micro_asset_weight_cap"] = float(payload.get("asset_weight_cap"))
    if "w_cap" in payload:
        pending_updates["universe_micro_w_cap_enabled"] = payload.get("w_cap") is not None
        if payload.get("w_cap") is not None:
            pending_updates["universe_micro_w_cap"] = float(payload.get("w_cap"))
    if "score_clip" in payload:
        pending_updates["universe_micro_score_clip_enabled"] = payload.get("score_clip") is not None
        if payload.get("score_clip") is not None:
            pending_updates["universe_micro_score_clip"] = float(payload.get("score_clip"))

    pending_updates["universe_local_search_last_applied_fingerprint"] = config_fingerprint(MicroPipelineConfig(**payload))
    _queue_session_updates(pending_updates)
    _queue_toast(f"Applied best local candidate to manual config: {source_label}", icon="✅")


def _append_refinement_history(entry: dict) -> None:
    history = list(st.session_state.get("universe_local_refinement_history", []))
    history.append(_coerce_mapping(entry))
    st.session_state["universe_local_refinement_history"] = history[-25:]


def _maybe_store_refinement_history(entry: dict) -> None:
    signature = _payload_json(entry)
    if signature == str(st.session_state.get("universe_local_refinement_history_last_signature", "")):
        return
    _append_refinement_history(entry)
    st.session_state["universe_local_refinement_history_last_signature"] = signature

def _safe_float(x: object, default: float = float("nan")) -> float:
    try:
        val = float(x)
    except Exception:
        return default
    return val if np.isfinite(val) else default

PARETO_OBJECTIVE_OPTIONS = [
    "sharpe",
    "cagr",
    "max_drawdown",
    "mean_turnover",
    "diversification",
    "stability",
]

PARETO_OBJECTIVE_LABELS = {
    "sharpe": "Sharpe",
    "cagr": "CAGR",
    "max_drawdown": "Drawdown",
    "mean_turnover": "Turnover",
    "diversification": "Diversification",
    "stability": "Stability",
}


def _format_pareto_objective_label(name: str) -> str:
    return PARETO_OBJECTIVE_LABELS.get(str(name), str(name))


def _build_local_pareto_input(trials_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(trials_df).copy()
    if df.empty:
        return df

    if "diversification" not in df.columns:
        for candidate in [
            "mean_diversification_ratio",
            "mean_effective_risk_bets",
            "mean_effective_breadth",
            "mean_effective_n_assets",
            "inverse_concentration",
            "mean_active_assets",
        ]:
            if candidate in df.columns:
                df["diversification"] = pd.to_numeric(df[candidate], errors="coerce")
                break
        else:
            df["diversification"] = np.nan

    if "stability" not in df.columns:
        sharpe = pd.to_numeric(df.get("sharpe"), errors="coerce")
        dd = pd.to_numeric(df.get("max_drawdown"), errors="coerce").abs()
        turnover = pd.to_numeric(df.get("mean_turnover"), errors="coerce")
        vol = pd.to_numeric(df.get("annual_volatility"), errors="coerce")
        breadth = pd.to_numeric(df.get("mean_effective_breadth"), errors="coerce")
        base = sharpe.fillna(0.0)
        penalty = dd.fillna(0.0) + turnover.fillna(0.0) + 0.5 * vol.fillna(0.0)
        breadth_bonus = 0.1 * breadth.fillna(0.0)
        df["stability"] = base - penalty + breadth_bonus

    return df


def _normalize_pareto_objective_matrix(
    df: pd.DataFrame,
    objective_specs: dict[str, dict] | None,
) -> tuple[pd.DataFrame, list[str]]:
    work = pd.DataFrame(df).copy()
    specs = {str(k): _coerce_mapping(v) for k, v in _coerce_mapping(objective_specs).items() if str(k) in work.columns}
    normalized_cols: list[str] = []
    if work.empty or not specs:
        return work, normalized_cols

    for objective_name, meta in specs.items():
        raw = pd.to_numeric(work.get(objective_name), errors="coerce")
        sense = str(meta.get("sense", "max")).strip().lower()
        if sense == "min":
            raw = -raw
        valid = raw[np.isfinite(raw)]
        norm_col = f"_norm_{objective_name}"
        if valid.empty:
            work[norm_col] = 0.0
        else:
            lo = float(valid.min())
            hi = float(valid.max())
            if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) <= 1e-12:
                work[norm_col] = np.where(np.isfinite(raw), 1.0, 0.0)
            else:
                norm = (raw - lo) / (hi - lo)
                work[norm_col] = norm.clip(lower=0.0, upper=1.0).fillna(0.0)
        normalized_cols.append(norm_col)
    return work, normalized_cols


def _attach_hypervolume_proxy_metrics(
    df: pd.DataFrame,
    objective_specs: dict[str, dict] | None,
) -> pd.DataFrame:
    work, norm_cols = _normalize_pareto_objective_matrix(df, objective_specs)
    if work.empty:
        return work

    has_backend_total = False
    backend_total = np.nan
    if "hypervolume_total" in work.columns:
        existing_total = pd.to_numeric(work.get("hypervolume_total"), errors="coerce")
        valid_total = existing_total[np.isfinite(existing_total)]
        if not valid_total.empty:
            backend_total = float(valid_total.iloc[0])
            has_backend_total = True
    if (not has_backend_total) and "frontier_hypervolume" in work.columns:
        frontier_total = pd.to_numeric(work.get("frontier_hypervolume"), errors="coerce")
        valid_frontier_total = frontier_total[np.isfinite(frontier_total)]
        if not valid_frontier_total.empty:
            backend_total = float(valid_frontier_total.iloc[0])
            has_backend_total = True

    if not norm_cols:
        if has_backend_total:
            work["hypervolume_total"] = backend_total
            if "hypervolume_contribution" not in work.columns:
                work["hypervolume_contribution"] = np.nan
            if "hypervolume_contribution_share" not in work.columns:
                work["hypervolume_contribution_share"] = np.nan
            work["hypervolume_source"] = "backend"
        else:
            work["hypervolume_total"] = np.nan
            work["hypervolume_contribution"] = np.nan
            work["hypervolume_contribution_share"] = np.nan
            work["hypervolume_source"] = "unavailable"
        return work

    norm_matrix = work[norm_cols].fillna(0.0).clip(lower=0.0, upper=1.0)
    hv_proxy = norm_matrix.prod(axis=1)
    hv_proxy = pd.to_numeric(hv_proxy, errors="coerce").fillna(0.0).clip(lower=0.0)
    hv_proxy_total = float(hv_proxy.sum()) if len(hv_proxy) else 0.0

    if has_backend_total:
        work["hypervolume_total"] = backend_total
        existing_contribution = pd.to_numeric(work.get("hypervolume_contribution"), errors="coerce") if "hypervolume_contribution" in work.columns else pd.Series(np.nan, index=work.index, dtype=float)
        if not np.isfinite(existing_contribution).any():
            work["hypervolume_contribution"] = hv_proxy
            work["hypervolume_contribution_share"] = (hv_proxy / hv_proxy_total) if hv_proxy_total > 0 else 0.0
        elif "hypervolume_contribution_share" not in work.columns:
            contrib_total = float(pd.to_numeric(work.get("hypervolume_contribution"), errors="coerce").fillna(0.0).sum())
            work["hypervolume_contribution_share"] = (pd.to_numeric(work.get("hypervolume_contribution"), errors="coerce").fillna(0.0) / contrib_total) if contrib_total > 0 else 0.0
        work["hypervolume_source"] = "backend"
    else:
        work["hypervolume_contribution"] = hv_proxy
        work["hypervolume_contribution_share"] = (hv_proxy / hv_proxy_total) if hv_proxy_total > 0 else 0.0
        work["hypervolume_total"] = hv_proxy_total
        work["hypervolume_source"] = "proxy"

    work["selection_score_balanced_compromise"] = norm_matrix.mean(axis=1)
    work["selection_score_knee_point"] = norm_matrix.mean(axis=1) - norm_matrix.std(axis=1, ddof=0)
    work["selection_score_weighted_hypervolume"] = hv_proxy
    return work


def _pareto_metadata_app(
    df: pd.DataFrame,
    *,
    objective_cols: list[str],
    objective_specs: dict[str, dict] | None,
) -> pd.DataFrame:
    work = _attach_hypervolume_proxy_metrics(df, objective_specs)
    if not isinstance(work, pd.DataFrame) or work.empty:
        return work

    use_cols = [str(c) for c in (objective_cols or []) if str(c) in work.columns]
    if not use_cols:
        work["is_pareto_efficient"] = False
        work["pareto_rank"] = np.nan
        work["crowding_distance"] = np.nan
        work["dominates_count"] = 0
        work["dominated_by_count"] = 0
        return work

    scores = {}
    for col in use_cols:
        vals = pd.to_numeric(work.get(col), errors="coerce")
        sense = str(_coerce_mapping(_coerce_mapping(objective_specs).get(col, {})).get("sense", "max")).strip().lower()
        if sense == "min":
            vals = -vals
        vals = vals.replace([np.inf, -np.inf], np.nan)
        vals = vals.where(np.isfinite(vals), -1e18)
        scores[col] = vals.astype(float).to_numpy()
    mat = np.column_stack([scores[c] for c in use_cols]) if use_cols else np.empty((len(work), 0))
    n = mat.shape[0]
    dominates_count = np.zeros(n, dtype=int)
    dominated_by_count = np.zeros(n, dtype=int)
    domination_sets = [set() for _ in range(n)]

    for i in range(n):
        xi = mat[i]
        for j in range(i + 1, n):
            xj = mat[j]
            i_dom_j = np.all(xi >= xj) and np.any(xi > xj)
            j_dom_i = np.all(xj >= xi) and np.any(xj > xi)
            if i_dom_j and not j_dom_i:
                dominates_count[i] += 1
                dominated_by_count[j] += 1
                domination_sets[i].add(j)
            elif j_dom_i and not i_dom_j:
                dominates_count[j] += 1
                dominated_by_count[i] += 1
                domination_sets[j].add(i)

    ranks = np.full(n, np.nan, dtype=float)
    remaining = set(range(n))
    current_rank = 1
    current_front = [i for i in range(n) if dominated_by_count[i] == 0]
    temp_dom_count = dominated_by_count.copy()
    fronts: list[list[int]] = []
    while current_front:
        fronts.append(list(current_front))
        next_front = []
        for i in current_front:
            if i in remaining:
                remaining.remove(i)
            ranks[i] = float(current_rank)
            for j in domination_sets[i]:
                temp_dom_count[j] -= 1
                if temp_dom_count[j] == 0:
                    next_front.append(j)
        current_rank += 1
        current_front = [j for j in next_front if j in remaining]
    for i in sorted(remaining):
        ranks[i] = float(current_rank)

    crowd = np.zeros(n, dtype=float)
    for front in fronts:
        if not front:
            continue
        if len(front) <= 2:
            for idx in front:
                crowd[idx] = np.inf
            continue
        front_mat = mat[front]
        front_crowd = np.zeros(len(front), dtype=float)
        for k in range(front_mat.shape[1]):
            vals = front_mat[:, k]
            order = np.argsort(vals)
            front_crowd[order[0]] = np.inf
            front_crowd[order[-1]] = np.inf
            vmin = vals[order[0]]
            vmax = vals[order[-1]]
            denom = vmax - vmin
            if not np.isfinite(denom) or denom <= 1e-12:
                continue
            for pos in range(1, len(order) - 1):
                if np.isinf(front_crowd[order[pos]]):
                    continue
                front_crowd[order[pos]] += (vals[order[pos + 1]] - vals[order[pos - 1]]) / denom
        for local_idx, global_idx in enumerate(front):
            crowd[global_idx] = front_crowd[local_idx]

    work["is_pareto_efficient"] = pd.Series(ranks, index=work.index).eq(1.0)
    work["pareto_rank"] = pd.Series(ranks, index=work.index)
    work["crowding_distance"] = pd.Series(crowd, index=work.index)
    work["dominates_count"] = pd.Series(dominates_count, index=work.index)
    work["dominated_by_count"] = pd.Series(dominated_by_count, index=work.index)
    return work


def _build_hypervolume_caption(df: pd.DataFrame | None) -> str:
    if isinstance(df, pd.DataFrame) and not df.empty and "hypervolume_source" in df.columns:
        source_values = {str(x).strip().lower() for x in df.get("hypervolume_source", pd.Series(dtype=object)).dropna().astype(str)}
        if "backend" in source_values:
            return "Hypervolume figures are sourced from the multiobjective backend for this result path."
        if "proxy" in source_values:
            return "Hypervolume figures in this view use a normalized frontier-volume proxy for compatibility with result paths that do not expose backend hypervolume totals."
    if isinstance(df, pd.DataFrame) and not df.empty:
        if "frontier_hypervolume" in df.columns:
            series = pd.to_numeric(df.get("frontier_hypervolume"), errors="coerce")
            if np.isfinite(series).any():
                return "Hypervolume figures are sourced from the multiobjective backend for this result path."
        if "hypervolume_total" in df.columns:
            series = pd.to_numeric(df.get("hypervolume_total"), errors="coerce")
            if np.isfinite(series).any():
                return "Hypervolume figures may come from the multiobjective backend or a compatibility fallback, depending on the current result path."
    return "Hypervolume figures are unavailable for this view."


def _choose_pareto_solution_app(
    candidate_df: pd.DataFrame,
    *,
    policy: str,
    objective_specs: dict[str, dict] | None,
    composite_profile: Any = DEFAULT_COMPOSITE_PROFILE,
) -> tuple[dict, pd.DataFrame]:
    prepared = _attach_hypervolume_proxy_metrics(candidate_df, objective_specs)
    if not isinstance(prepared, pd.DataFrame) or prepared.empty:
        return {}, prepared

    policy_name = str(policy or "balanced_compromise").strip().lower()
    if isinstance(composite_profile, dict):
        composite_profile_name = {
            str(k): float(v)
            for k, v in dict(composite_profile).items()
            if str(k) and np.isfinite(pd.to_numeric(v, errors="coerce"))
        }
        if not composite_profile_name:
            composite_profile_name = DEFAULT_COMPOSITE_PROFILE
    else:
        composite_profile_name = str(composite_profile or DEFAULT_COMPOSITE_PROFILE).strip().lower() or DEFAULT_COMPOSITE_PROFILE
    score_col = {
        "balanced_compromise": "selection_score_balanced_compromise",
        "knee_point": "selection_score_knee_point",
        "weighted_hypervolume": "selection_score_weighted_hypervolume",
        "fixed_composite_score": "fixed_composite_score",
        "composite_balanced": "fixed_composite_score",
        "composite_growth": "fixed_composite_score",
        "composite_defensive": "fixed_composite_score",
    }.get(policy_name, "selection_score_balanced_compromise")

    picked: dict[str, Any] = {}
    try:
        picked = dict(
            choose_multiobjective_solution_by_policy(
                prepared,
                policy=policy_name,
                objective_specs=objective_specs,
                composite_profile=composite_profile_name,
            )
            or {}
        )
    except TypeError:
        try:
            picked = dict(
                choose_multiobjective_solution_by_policy(
                    prepared,
                    policy=policy_name,
                    objective_specs=objective_specs,
                )
                or {}
            )
        except Exception:
            picked = {}
    except Exception:
        picked = {}

    selected_row = _coerce_mapping(picked.get("selected_row"))
    picked_frontier = pd.DataFrame(picked.get("frontier_df", prepared)).copy() if picked else prepared.copy()
    if isinstance(picked_frontier, pd.DataFrame) and not picked_frontier.empty:
        prepared = _attach_hypervolume_proxy_metrics(picked_frontier, objective_specs)

    selected_idx = None
    if selected_row:
        selected_fp = str(selected_row.get("effective_config_fingerprint") or selected_row.get("config_fingerprint") or "")
        if selected_fp:
            fp_series = prepared.get("effective_config_fingerprint")
            if fp_series is None:
                fp_series = prepared.get("config_fingerprint")
            if fp_series is not None:
                matches = prepared.index[(fp_series.fillna("").astype(str) == selected_fp)]
                if len(matches):
                    selected_idx = matches[0]
        if selected_idx is None and len(prepared):
            try:
                selected_idx = prepared.index[0]
            except Exception:
                selected_idx = None
        if selected_idx is not None:
            selected_row = _row_to_dict(prepared.loc[selected_idx])

    if not selected_row:
        score = pd.to_numeric(prepared.get(score_col), errors="coerce")
        if score is None or not isinstance(score, pd.Series):
            score = pd.Series(np.nan, index=prepared.index, dtype=float)
        if "crowding_distance" in prepared.columns:
            crowd = pd.to_numeric(prepared.get("crowding_distance"), errors="coerce")
        else:
            crowd = pd.Series(np.zeros(len(prepared)), index=prepared.index, dtype=float)
        rank = pd.to_numeric(prepared.get("pareto_rank"), errors="coerce") if "pareto_rank" in prepared.columns else pd.Series(np.ones(len(prepared)), index=prepared.index, dtype=float)
        chooser = pd.DataFrame({"score": score, "crowding_distance": crowd, "pareto_rank": rank}, index=prepared.index)
        chooser = chooser.replace([np.inf, -np.inf], np.nan)
        chooser["score"] = chooser["score"].fillna(-1.0)
        chooser["crowding_distance"] = chooser["crowding_distance"].fillna(0.0)
        chooser["pareto_rank"] = chooser["pareto_rank"].fillna(999999.0)
        chooser = chooser.sort_values(["score", "crowding_distance", "pareto_rank"], ascending=[False, False, True])
        if not chooser.empty:
            selected_idx = chooser.index[0]
            selected_row = _row_to_dict(prepared.loc[selected_idx])

    return selected_row, prepared


def _make_pareto_candidate_label(row: pd.Series, position: int) -> str:
    rank = row.get("pareto_rank", "—")
    stage = str(row.get("refinement_stage", "trial")).replace("_", " ")
    param = row.get("tuned_param", "candidate")
    sharpe = _format_num_or_dash(row.get("sharpe"))
    cagr = _format_pct_or_dash(row.get("cagr"))
    turnover = _format_pct_or_dash(row.get("mean_turnover"))
    return f"#{position + 1} | rank {rank} | {stage} | {param} | Sharpe {sharpe} | CAGR {cagr} | Turnover {turnover}"


def _format_pct_or_dash(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{100.0 * float(x):.2f}%"


def _format_num_or_dash(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{float(x):.2f}"


def _format_scalar_compact(x: object) -> str:
    if x is None:
        return "—"
    try:
        if pd.isna(x):
            return "—"
    except Exception:
        pass
    if isinstance(x, (bool, np.bool_)):
        return "True" if bool(x) else "False"
    if isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_)):
        return str(int(x))
    try:
        val = float(x)
        if np.isfinite(val):
            if abs(val) >= 100:
                return f"{val:.2f}"
            if abs(val) >= 1:
                return f"{val:.3g}"
            return f"{val:.4g}"
    except Exception:
        pass
    return str(x)


def _format_candidate_value_or_dash(x: object) -> str:
    if x is None:
        return "—"
    try:
        if pd.isna(x):
            return "—"
    except Exception:
        pass
    if isinstance(x, dict):
        items = list(x.items())
        return ", ".join(f"{k}={_format_scalar_compact(v)}" for k, v in items) if items else "—"
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return "—"
        if s.startswith('{') or s.startswith('['):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    items = list(parsed.items())
                    return ", ".join(f"{k}={_format_scalar_compact(v)}" for k, v in items) if items else "—"
                if isinstance(parsed, list):
                    return ", ".join(_format_scalar_compact(v) for v in parsed) if parsed else "—"
            except Exception:
                return s
        try:
            val = float(s)
            return _format_num_or_dash(val)
        except Exception:
            return s
    try:
        return _format_num_or_dash(float(x))
    except Exception:
        return str(x)


def _format_delta_pct_or_dash(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    sign = "+" if float(x) > 0 else ""
    return f"{sign}{100.0 * float(x):.2f}%"


def _format_bps_or_dash(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    sign = "+" if float(x) > 0 else ""
    return f"{sign}{float(x):.1f} bps"


def _prepare_cost_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df is None or not isinstance(summary_df, pd.DataFrame) or summary_df.empty:
        return pd.DataFrame()
    out = summary_df.copy()
    keep = [c for c in ["metric", "value"] if c in out.columns]
    if keep:
        out = out[keep].copy()
    if "value" in out.columns:
        out["value"] = pd.to_numeric(out["value"], errors="ignore")
    return out


def _prepare_factor_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df is None or not isinstance(summary_df, pd.DataFrame) or summary_df.empty:
        return pd.DataFrame()
    out = summary_df.copy()
    keep = [c for c in ["metric", "value"] if c in out.columns]
    if keep:
        out = out[keep].copy()
    if "value" in out.columns:
        out["value"] = pd.to_numeric(out["value"], errors="ignore")
    return out


def _safe_perf_metric(run: dict, *keys: str) -> float | None:
    if not isinstance(run, dict):
        return None
    perf = _coerce_mapping(run.get("performance_summary", {}))
    if not isinstance(perf, dict):
        return None
    for key in keys:
        if key in perf:
            try:
                val = float(perf.get(key))
                if np.isfinite(val):
                    return val
            except Exception:
                pass
    return None


def _safe_diag_mean(run: dict, *cols: str) -> float | None:
    if not isinstance(run, dict):
        return None
    diag = run.get("diagnostics_df", pd.DataFrame())
    if not isinstance(diag, pd.DataFrame) or diag.empty:
        return None
    for col in cols:
        if col in diag.columns:
            s = pd.to_numeric(diag[col], errors="coerce")
            if s.notna().any():
                return float(s.mean())
    return None


def _build_factor_model_summary_compat(run: dict) -> pd.DataFrame:
    if not isinstance(run, dict):
        return pd.DataFrame()
    rows = []
    def add(metric, value):
        rows.append({"metric": str(metric), "value": value})
    add("factor_model_active", _safe_diag_mean(run, "factor_model_active"))
    add("factor_covariance_active", _safe_diag_mean(run, "factor_covariance_active"))
    add("factor_overlay_active_share", _safe_diag_mean(run, "factor_overlay_active"))
    add("factor_model_n_factors_mean", _safe_diag_mean(run, "factor_model_n_factors"))
    add("factor_model_n_obs_mean", _safe_diag_mean(run, "factor_model_n_obs"))
    add("factor_model_explained_variance_share_mean", _safe_diag_mean(run, "factor_model_explained_variance_share"))
    add("factor_model_residual_variance_share_mean", _safe_diag_mean(run, "factor_model_residual_variance_share"))
    add("factor_mu_abs_tilt_mean", _safe_diag_mean(run, "factor_mu_abs_tilt_mean"))
    add("factor_mu_abs_tilt_max", _safe_diag_mean(run, "factor_mu_abs_tilt_max"))
    add("factor_covariance_blend_used_mean", _safe_diag_mean(run, "factor_covariance_blend_used", "factor_model_covariance_blend_used"))
    add("factor_residual_blend_used_mean", _safe_diag_mean(run, "factor_residual_blend_used"))
    add("factor_covariance_trace_ratio_mean", _safe_diag_mean(run, "factor_covariance_trace_ratio"))
    add("cagr", _safe_perf_metric(run, "cagr"))
    add("sharpe", _safe_perf_metric(run, "sharpe"))
    add("max_drawdown", _safe_perf_metric(run, "max_drawdown"))
    add("annual_volatility", _safe_perf_metric(run, "annual_volatility", "annualized_volatility"))
    add("information_ratio", _safe_perf_metric(run, "information_ratio"))
    add("mean_turnover", _safe_perf_metric(run, "mean_turnover") or _safe_diag_mean(run, "turnover"))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out[out["value"].notna()].reset_index(drop=True)


def _build_factor_model_timeseries_compat(run: dict) -> pd.DataFrame:
    if not isinstance(run, dict):
        return pd.DataFrame()
    diag = run.get("diagnostics_df", pd.DataFrame())
    if not isinstance(diag, pd.DataFrame) or diag.empty:
        return pd.DataFrame()
    cols = [
        "date", "factor_model_active", "factor_covariance_active", "factor_overlay_active",
        "factor_model_n_factors", "factor_model_n_obs", "factor_model_explained_variance_share",
        "factor_model_residual_variance_share", "factor_mu_abs_tilt_mean", "factor_mu_abs_tilt_max",
        "factor_covariance_blend_used", "factor_model_covariance_blend_used", "factor_residual_blend_used",
        "factor_covariance_trace_ratio", "gross_portfolio_return_simple", "portfolio_return_simple",
        "cumulative_gross_nav", "cumulative_net_nav", "nav_end",
    ]
    keep = [c for c in cols if c in diag.columns]
    if not keep:
        return pd.DataFrame()
    out = diag[keep].copy()
    if "portfolio_return_simple" in out.columns and "net_portfolio_return_simple" not in out.columns:
        out = out.rename(columns={"portfolio_return_simple": "net_portfolio_return_simple"})
    if "factor_model_covariance_blend_used" in out.columns and "factor_covariance_blend_used" not in out.columns:
        out = out.rename(columns={"factor_model_covariance_blend_used": "factor_covariance_blend_used"})
    if "cumulative_gross_nav" not in out.columns and "gross_portfolio_return_simple" in out.columns:
        gross = pd.to_numeric(out["gross_portfolio_return_simple"], errors="coerce").fillna(0.0)
        out["cumulative_gross_nav"] = (1.0 + gross).cumprod()
    if "cumulative_net_nav" not in out.columns:
        if "net_portfolio_return_simple" in out.columns:
            net = pd.to_numeric(out["net_portfolio_return_simple"], errors="coerce").fillna(0.0)
            out["cumulative_net_nav"] = (1.0 + net).cumprod()
        elif "nav_end" in out.columns:
            out["cumulative_net_nav"] = pd.to_numeric(out["nav_end"], errors="coerce")
    return out


def _compare_factor_model_runs_compat(baseline_run: dict, factor_run: dict, *, baseline_name: str = "no-factor", factor_name: str = "factor-industrial") -> dict:
    base_summary = _build_factor_model_summary_compat(baseline_run)
    factor_summary = _build_factor_model_summary_compat(factor_run)
    if not base_summary.empty or not factor_summary.empty:
        merged = pd.merge(
            base_summary.rename(columns={"value": "baseline_value"}),
            factor_summary.rename(columns={"value": "factor_value"}),
            on="metric",
            how="outer",
        )
        merged["baseline_value"] = pd.to_numeric(merged["baseline_value"], errors="coerce")
        merged["factor_value"] = pd.to_numeric(merged["factor_value"], errors="coerce")
        merged["delta"] = merged["factor_value"] - merged["baseline_value"]
    else:
        merged = pd.DataFrame(columns=["metric", "baseline_value", "factor_value", "delta"])
    better_low = {"max_drawdown", "annual_volatility", "mean_turnover", "factor_mu_abs_tilt_mean", "factor_mu_abs_tilt_max"}
    rows = []
    for _, r in merged.iterrows():
        metric = str(r.get("metric"))
        delta = r.get("delta")
        better_when = "lower" if metric in better_low else "higher"
        direction = "neutral"
        if pd.notna(delta):
            if metric == "factor_covariance_trace_ratio_mean":
                b = r.get("baseline_value")
                f = r.get("factor_value")
                if pd.notna(b) and pd.notna(f):
                    if abs(float(f) - 1.0) + 1e-12 < abs(float(b) - 1.0):
                        direction = "improved"
                    elif abs(float(f) - 1.0) > abs(float(b) - 1.0) + 1e-12:
                        direction = "worsened"
            elif better_when == "higher":
                direction = "improved" if float(delta) > 1e-12 else ("worsened" if float(delta) < -1e-12 else "neutral")
            else:
                direction = "improved" if float(delta) < -1e-12 else ("worsened" if float(delta) > 1e-12 else "neutral")
        rows.append({
            "metric": metric,
            "baseline_value": r.get("baseline_value"),
            "factor_value": r.get("factor_value"),
            "delta": delta,
            "better_when": better_when,
            "direction": direction,
            "headline": f"{metric}: {factor_name} {direction} versus {baseline_name}.",
        })
    comp = pd.DataFrame(rows)
    focus_metrics = ["cagr", "sharpe", "max_drawdown", "annual_volatility", "mean_turnover", "information_ratio", "factor_model_explained_variance_share_mean", "factor_mu_abs_tilt_mean", "factor_covariance_trace_ratio_mean"]
    focus = comp[comp["metric"].isin(focus_metrics)].copy() if not comp.empty else pd.DataFrame()
    ts = _build_factor_model_timeseries_compat(factor_run)
    improved = int((comp.get("direction") == "improved").sum()) if not comp.empty else 0
    worsened = int((comp.get("direction") == "worsened").sum()) if not comp.empty else 0
    return {
        "factor_summary_baseline": base_summary,
        "factor_summary_factor_run": factor_summary,
        "factor_comparison": comp,
        "factor_focus": focus,
        "factor_timeseries": ts,
        "decision_summary": {"headline": f"{factor_name} comparison complete: {improved} improved, {worsened} worsened versus {baseline_name}.", "improved": improved, "worsened": worsened},
    }


def _overlay_decision_kind(direction: str | None) -> str:
    direction = str(direction or "").strip().lower()
    if direction == "improved":
        return "ok"
    if direction == "worsened":
        return "bad"
    return "warn"


def _overlay_decision_text(direction: str | None) -> str:
    direction = str(direction or "").strip().lower()
    if direction == "improved":
        return "Overlay helped economically in this matched run."
    if direction == "worsened":
        return "Overlay hurt economically in this matched run."
    return "Overlay impact looks mixed or neutral in this matched run."


def _prepare_overlay_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df is None or not isinstance(summary_df, pd.DataFrame) or summary_df.empty:
        return pd.DataFrame()
    out = summary_df.copy()
    if "metric" in out.columns and "value" in out.columns:
        keep = [c for c in ["metric", "value"] if c in out.columns]
        out = out[keep].copy()
        if "value" in out.columns:
            out["value"] = pd.to_numeric(out["value"], errors="ignore")
    return out


def _overlay_metric_lookup_from_summary(summary_df: pd.DataFrame) -> dict:
    if summary_df is None or not isinstance(summary_df, pd.DataFrame) or summary_df.empty:
        return {}
    if not {"metric", "value"}.issubset(summary_df.columns):
        return {}
    return dict(zip(summary_df["metric"].astype(str), summary_df["value"]))


def _overlay_metric_float(metric_lookup: dict, key: str) -> float | None:
    if not isinstance(metric_lookup, dict):
        return None
    val = pd.to_numeric(pd.Series([metric_lookup.get(key)]), errors="coerce").iloc[0]
    return float(val) if pd.notna(val) else None


def _overlay_metric_text(metric_lookup: dict, key: str) -> str:
    if not isinstance(metric_lookup, dict):
        return "—"
    val = metric_lookup.get(key)
    if val is None:
        return "—"
    txt = str(val).strip()
    return txt if txt else "—"


def _build_overlay_rich_focus_cards(metric_lookup: dict) -> list[dict]:
    cards: list[dict] = []
    prob_mode_effective = _overlay_metric_text(metric_lookup, "probabilistic_mode_effective")
    if prob_mode_effective != "—":
        cards.append({
            "label": "Prob mode effective",
            "value": prob_mode_effective,
            "delta": None,
            "help": "Real backend probabilistic mode actually used by the overlay.",
        })
    prob_source = _overlay_metric_text(metric_lookup, "prob_source")
    if prob_source != "—":
        cards.append({
            "label": "Prob source",
            "value": prob_source,
            "delta": None,
            "help": "Primary probabilistic source reported by the run diagnostics.",
        })
    feature_mu_active_share = _overlay_metric_float(metric_lookup, "feature_mu_active_share")
    if feature_mu_active_share is not None:
        cards.append({
            "label": "Feature μ active share",
            "value": _format_pct_or_dash(feature_mu_active_share),
            "delta": None,
            "help": "Share of rebalance windows where feature-conditioned mu was active.",
        })
    feature_mu_distance_mean = _overlay_metric_float(metric_lookup, "feature_mu_distance_mean")
    if feature_mu_distance_mean is not None:
        cards.append({
            "label": "Feature μ distance",
            "value": _format_num_or_dash(feature_mu_distance_mean),
            "delta": None,
            "help": "Average analogue distance used by the feature-conditioned mu block.",
        })
    regime_enabled = _overlay_metric_float(metric_lookup, "regime_universe_enabled_rate")
    if regime_enabled is not None:
        cards.append({
            "label": "Regime universe enabled",
            "value": _format_pct_or_dash(regime_enabled),
            "delta": None,
            "help": "Share of rebalance windows where regime-dependent universe filtering was enabled.",
        })
    regime_selected_assets = _overlay_metric_float(metric_lookup, "regime_universe_selected_assets_mean")
    if regime_selected_assets is not None:
        cards.append({
            "label": "Regime selected assets",
            "value": _format_num_or_dash(regime_selected_assets),
            "delta": None,
            "help": "Average number of assets kept after regime-dependent universe filtering.",
        })
    return cards


def _prepare_overlay_internal_metrics_table(metric_base_df: pd.DataFrame, metric_overlay_df: pd.DataFrame) -> pd.DataFrame:
    internal_blocks = []
    if isinstance(metric_base_df, pd.DataFrame) and not metric_base_df.empty:
        tmp = metric_base_df.copy().rename(columns={"value": "none"})
        internal_blocks.append(tmp)
    if isinstance(metric_overlay_df, pd.DataFrame) and not metric_overlay_df.empty:
        tmp = metric_overlay_df.copy().rename(columns={"value": "overlay"})
        internal_blocks.append(tmp)
    if not internal_blocks:
        return pd.DataFrame()
    internal_df = internal_blocks[0]
    for extra in internal_blocks[1:]:
        internal_df = internal_df.merge(extra, on="metric", how="outer")
    if internal_df.empty or "metric" not in internal_df.columns:
        return internal_df

    preferred_order = [
        "prob_source",
        "probabilistic_mode_effective",
        "signal_mode_effective",
        "probabilistic_active_rate",
        "probabilistic_feature_aware_share",
        "probabilistic_feature_match_n_mean",
        "probabilistic_feature_distance_mean",
        "probabilistic_interval_width_mean",
        "probabilistic_downside_mean",
        "probabilistic_confidence_mean",
        "probabilistic_mu_penalty_mean_abs",
        "probabilistic_mu_penalty_max_abs",
        "feature_mu_active_share",
        "feature_mu_match_n_mean",
        "feature_mu_cols_used_mean",
        "feature_mu_distance_mean",
        "feature_mu_abs_tilt_mean",
        "feature_mu_abs_tilt_max",
        "regime_universe_enabled_rate",
        "regime_universe_keep_frac_mean",
        "regime_universe_keep_frac_low_mean",
        "regime_universe_keep_frac_mid_mean",
        "regime_universe_keep_frac_high_mean",
        "regime_universe_selected_assets_mean",
        "regime_universe_filtered_assets_mean",
        "regime_universe_selection_metric_mode",
        "regime_universe_regime_mode",
        "regime_universe_low_offensive_vol_tilt_mean",
        "regime_universe_high_defensive_vol_tilt_mean",
        "regime_universe_min_assets_mean",
        "regime_universe_max_assets_mean",
        "requested_asset_weight_cap_mean",
        "requested_w_cap_mean",
        "effective_cap_base_mean",
        "effective_cap_final_mean",
        "dispersion_risk_model_enabled_rate",
        "dispersion_risk_model_active_rate",
        "dispersion_risk_model_mult_mean",
        "dispersion_risk_model_z_mean",
        "dispersion_risk_model_trace_ratio_mean",
        "dispersion_risk_model_diag_ratio_mean",
        "dispersion_risk_model_apply_target_mode",
        "dispersion_risk_model_reason_mode",
    ]
    order_map = {metric: i for i, metric in enumerate(preferred_order)}
    internal_df["__order"] = internal_df["metric"].astype(str).map(order_map).fillna(len(order_map) + 999)
    internal_df = internal_df.sort_values(["__order", "metric"]).drop(columns="__order").reset_index(drop=True)
    return internal_df


def _overlay_validation_kind(status: str | None) -> str:
    status = str(status or "").strip().lower()
    if status == "ok":
        return "ok"
    if status == "fail":
        return "bad"
    return "info"


def _overlay_validation_text(status: str | None) -> str:
    status = str(status or "").strip().lower()
    if status == "ok":
        return "Diagnostics coverage looks complete for this mode."
    if status == "fail":
        return "Critical diagnostics gap: required diagnostics are missing or empty for this mode."
    return "Diagnostics coverage is partial for this run: some optional diagnostics were unavailable or sparse, but the main economic comparison is still available."


def _overlay_validation_label(status: str | None) -> str:
    status = str(status or "").strip().lower()
    if status == "ok":
        return "Complete"
    if status == "fail":
        return "Critical gap"
    return "Partial"


def _prepare_overlay_validation_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in ["diagnostics_rows", "missing_required_count", "missing_optional_count", "sparse_required_count"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _prepare_overlay_validation_warnings(warnings_obj) -> list[str]:
    if warnings_obj is None:
        return []
    if isinstance(warnings_obj, (list, tuple, set)):
        return [str(x) for x in warnings_obj if str(x).strip()]
    txt = str(warnings_obj).strip()
    return [txt] if txt else []



def _probabilistic_mode_audit_kind(status: str | None, *, fallback_triggered: bool = False, backend_distinct: bool | None = None) -> str:
    status = str(status or "").strip().lower()
    if status == "fail":
        return "bad"
    if fallback_triggered or backend_distinct is False:
        return "warn"
    if status == "warn":
        return "info"
    return "ok"



def _probabilistic_mode_audit_headline(row: dict) -> tuple[str, str]:
    status = str(_row_to_dict(row).get("semantic_status", "warn") or "warn").strip().lower()
    fallback_triggered = bool(_row_to_dict(row).get("prob_fallback_triggered", False))
    backend_distinct = _row_to_dict(row).get("prob_backend_distinct")
    requested_mode = str(_row_to_dict(row).get("requested_mode") or "none")
    effective_mode = str(_row_to_dict(row).get("effective_mode") or requested_mode)
    accepted_ok = bool(_row_to_dict(row).get("accepted_prob_source_ok", False))
    backend_ok = bool(_row_to_dict(row).get("backend_family_ok", False))
    interval_ok = bool(_row_to_dict(row).get("interval_cols_ok", False))
    required_ok = bool(_row_to_dict(row).get("required_groups_ok", False))

    if status == "fail" or not accepted_ok or not backend_ok or not interval_ok:
        return (
            "Requested mode is not semantically aligned with realised backend / diagnostics.",
            "bad",
        )
    if fallback_triggered or backend_distinct is False or not required_ok or requested_mode != effective_mode:
        return (
            "Run is usable, but semantics show fallback / partial distinctness / contract warnings.",
            "warn",
        )
    return (
        "Distinct probabilistic backend validated and contract looks aligned.",
        "ok",
    )



def _prepare_probabilistic_mode_audit_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    bool_cols = [
        "prob_backend_distinct",
        "required_groups_ok",
        "accepted_prob_source_ok",
        "interval_cols_ok",
        "backend_family_ok",
        "interval_backend_ok",
        "backend_distinct_ok",
        "prob_fallback_triggered",
    ]
    for col in bool_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda x: bool(x) if pd.notna(x) else x)
    text_cols = [
        "requested_mode",
        "effective_mode",
        "prob_source",
        "prob_backend_family",
        "prob_backend_name",
        "prob_backend_variant",
        "prob_interval_backend",
        "semantic_status",
        "warnings",
        "prob_fallback_reason",
    ]
    keep_cols = [c for c in [
        "run_name",
        *text_cols,
        "prob_backend_distinct",
        "required_groups_ok",
        "missing_required_groups",
        "accepted_prob_source_ok",
        "interval_cols_ok",
        "backend_family_ok",
        "interval_backend_ok",
        "backend_distinct_ok",
        "prob_fallback_triggered",
        "prob_contract_status",
    ] if c in out.columns]

    keep_cols = list(dict.fromkeys(keep_cols))

    return out[keep_cols].copy()



def _render_probabilistic_mode_audit(run: dict, *, title: str = "Probabilistic mode audit", key_prefix: str = "primary") -> pd.DataFrame:
    try:
        audit_df = build_probabilistic_mode_audit_table(run)
    except Exception as e:
        st.warning(f"Could not build probabilistic mode audit: {e}")
        return pd.DataFrame()
    audit_df = _prepare_probabilistic_mode_audit_table(audit_df)
    if audit_df.empty:
        return audit_df
    row = _row_to_dict(audit_df.iloc[0])
    headline, headline_kind = _probabilistic_mode_audit_headline(row)

    st.markdown(f"#### {title}")
    badge(str((row.get("semantic_status") or "warn")).upper(), _probabilistic_mode_audit_kind(
        row.get("semantic_status"),
        fallback_triggered=bool(row.get("prob_fallback_triggered", False)),
        backend_distinct=row.get("prob_backend_distinct"),
    ))
    st.caption(headline)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Requested mode", _format_probabilistic_mode_label(str(row.get("requested_mode") or "none")))
    with c2:
        st.metric("Effective mode", _format_probabilistic_mode_label(str(row.get("effective_mode") or row.get("requested_mode") or "none")))
    with c3:
        st.metric("prob_source", str(row.get("prob_source") or "—"))
    with c4:
        st.metric("Backend family", str(row.get("prob_backend_family") or "—"))

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric("Backend distinct", "Yes" if bool(row.get("prob_backend_distinct", False)) else "No")
    with c6:
        st.metric("Fallback triggered", "Yes" if bool(row.get("prob_fallback_triggered", False)) else "No")
    with c7:
        st.metric("Required groups", "OK" if bool(row.get("required_groups_ok", False)) else "Missing")
    with c8:
        st.metric("Accepted source", "OK" if bool(row.get("accepted_prob_source_ok", False)) else "Mismatch")

    fallback_reason = str(row.get("prob_fallback_reason") or "").strip()
    if fallback_reason:
        st.caption(f"Fallback reason: {fallback_reason}")

    warnings_text = str(row.get("warnings") or "").strip()
    if warnings_text:
        with st.expander("Probabilistic mode audit warnings", expanded=(headline_kind == "bad")):
            for bit in [x.strip() for x in warnings_text.split(" | ") if str(x).strip()]:
                st.markdown(f"- {bit}")

    with st.expander("Probabilistic mode audit table", expanded=False):
        st.dataframe(audit_df, use_container_width=True, hide_index=True)
        _download_dataframe_button(
            f"Download {key_prefix} probabilistic mode audit",
            audit_df,
            f"{key_prefix}_probabilistic_mode_audit.csv",
            key=f"{key_prefix}_probabilistic_mode_audit_download",
        )
    return audit_df



def _build_overlay_mode_audit_summary(base_run: dict, overlay_run: dict, *, base_name: str, overlay_name: str) -> pd.DataFrame:
    frames = []
    for run_name, run_obj in [(base_name, base_run), (overlay_name, overlay_run)]:
        try:
            audit_df = build_probabilistic_mode_audit_table(run_obj)
        except Exception:
            audit_df = pd.DataFrame()
        audit_df = _prepare_probabilistic_mode_audit_table(audit_df)
        if audit_df.empty:
            continue
        local = audit_df.copy()
        local.insert(0, "run_name", str(run_name))
        frames.append(local)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _infer_overlay_direction_from_bundle(overlay_bundle: dict) -> str:
    decision_summary = overlay_bundle.get("decision_summary", {}) if isinstance(overlay_bundle, dict) else {}
    direction = str(_coerce_mapping(decision_summary).get("direction", "")).strip().lower()
    if direction in {"improved", "worsened", "neutral"}:
        return direction
    delta = (overlay_bundle.get("economic_delta", {}) if isinstance(overlay_bundle, dict) else {}) or {}
    delta_sharpe = pd.to_numeric(pd.Series([delta.get("delta_sharpe")]), errors="coerce").iloc[0]
    delta_cagr = pd.to_numeric(pd.Series([delta.get("delta_cagr")]), errors="coerce").iloc[0]
    delta_turnover = pd.to_numeric(pd.Series([delta.get("delta_turnover")]), errors="coerce").iloc[0]
    if pd.notna(delta_sharpe) and delta_sharpe > 0 and (pd.isna(delta_cagr) or delta_cagr >= 0):
        return "improved"
    if pd.notna(delta_sharpe) and delta_sharpe < 0 and (pd.isna(delta_turnover) or delta_turnover >= 0):
        return "worsened"
    return "neutral"


def _build_overlay_interpretation_lines(overlay_bundle: dict) -> list[str]:
    lines: list[str] = []
    if not isinstance(overlay_bundle, dict):
        return lines

    decision_summary = _coerce_mapping(overlay_bundle.get("decision_summary", {}))
    economic_delta = _coerce_mapping(
        overlay_bundle.get("economic_delta", {})
    )

    direction = _infer_overlay_direction_from_bundle(overlay_bundle)
    improved_count = int(decision_summary.get("improved_metric_count", 0) or 0)
    worsened_count = int(decision_summary.get("worsened_metric_count", 0) or 0)

    delta_sharpe = pd.to_numeric(
        pd.Series([economic_delta.get("delta_sharpe")]), errors="coerce"
    ).iloc[0]
    delta_cagr = pd.to_numeric(
        pd.Series([economic_delta.get("delta_cagr")]), errors="coerce"
    ).iloc[0]
    delta_mdd = pd.to_numeric(
        pd.Series([economic_delta.get("delta_max_drawdown")]), errors="coerce"
    ).iloc[0]
    delta_turnover = pd.to_numeric(
        pd.Series([economic_delta.get("delta_turnover")]), errors="coerce"
    ).iloc[0]

    if direction == "improved":
        lines.append(
            "Economic reading: the overlay improves the realised risk-adjusted profile versus the matched no-overlay baseline."
        )
    elif direction == "worsened":
        lines.append(
            "Economic reading: the overlay weakens the realised economic profile versus the matched no-overlay baseline."
        )
    else:
        lines.append(
            "Economic reading: the overlay changes the portfolio profile, but the realised outcome is mixed rather than decisively better."
        )

    if pd.notna(delta_sharpe):
        if delta_sharpe > 0:
            lines.append(
                f"Sharpe moved up by {float(delta_sharpe):.2f}, which supports the case that the overlay added useful signal conditioning."
            )
        elif delta_sharpe < 0:
            lines.append(
                f"Sharpe moved down by {float(delta_sharpe):.2f}, which suggests the overlay may be over-filtering or distorting the base signal."
            )
        else:
            lines.append(
                "Sharpe was essentially unchanged, so the overlay did not materially improve risk-adjusted performance."
            )

    if pd.notna(delta_cagr):
        if delta_cagr > 0:
            lines.append(
                f"CAGR improved by {100.0 * float(delta_cagr):.2f} percentage points, so the overlay added return in realised portfolio terms."
            )
        elif delta_cagr < 0:
            lines.append(
                f"CAGR fell by {abs(100.0 * float(delta_cagr)):.2f} percentage points, so the overlay gave up realised growth."
            )

    if pd.notna(delta_mdd):
        if delta_mdd > 0:
            lines.append(
                f"Max drawdown became less severe by {100.0 * float(delta_mdd):.2f} percentage points, which improves downside robustness."
            )
        elif delta_mdd < 0:
            lines.append(
                f"Max drawdown worsened by {abs(100.0 * float(delta_mdd)):.2f} percentage points, so the overlay did not help capital preservation in this run."
            )

    if pd.notna(delta_turnover):
        if delta_turnover < 0:
            lines.append(
                f"Turnover fell by {abs(100.0 * float(delta_turnover)):.2f} percentage points, which strengthens the implementation case once trading frictions matter."
            )
        elif delta_turnover > 0:
            lines.append(
                f"Turnover rose by {100.0 * float(delta_turnover):.2f} percentage points, so some of the overlay benefit may be offset by implementation drag."
            )

    lines.append(
        f"Across the core economic metrics, {improved_count} improved and {worsened_count} worsened versus the no-overlay benchmark."
    )
    lines.append(
        "Coursework interpretation: this block should be discussed as an economic validation of whether probabilistic conditioning actually improves realised portfolio outcomes, not just forecast aesthetics."
    )
    return lines



def _build_overlay_coursework_paragraph(overlay_bundle: dict) -> str:
    lines = _build_overlay_interpretation_lines(overlay_bundle)
    if not lines:
        return ""
    return "\n\n".join(lines)


def _coerce_numeric_columns(df: pd.DataFrame, skip: set[str] | None = None) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame() if df is None else df
    skip = skip or set()
    out = df.copy()
    for col in out.columns:
        if col in skip:
            continue
        converted = pd.to_numeric(out[col], errors="coerce")
        if converted.notna().any():
            out[col] = converted
    return out


def _prepare_download_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").astype(str)
    return out




def apply_crash_overlay_preset(preset_name: str) -> None:
    preset = CRASH_OVERLAY_PRESETS.get(str(preset_name), {})
    if not preset:
        st.warning(f"Crash overlay preset not found: {preset_name}")
        return

    st.session_state["universe_micro_prob_mode"] = str(preset.get("probabilistic_mode", "historical_by_regime"))
    st.session_state["universe_micro_prob_q_low"] = float(preset.get("probabilistic_q_low", 0.15))
    st.session_state["universe_micro_prob_q_high"] = float(preset.get("probabilistic_q_high", 0.75))
    st.session_state["universe_micro_prob_min_obs"] = int(preset.get("probabilistic_min_obs", 12))
    st.session_state["universe_micro_prob_interval_penalty_weight"] = float(preset.get("probabilistic_interval_penalty_weight", 0.25))
    st.session_state["universe_micro_prob_downside_penalty_weight"] = float(preset.get("probabilistic_downside_penalty_weight", 0.90))
    st.session_state["universe_micro_prob_confidence_scale"] = float(preset.get("probabilistic_confidence_scale", 1.10))
    st.session_state["universe_micro_prob_confidence_min_mult"] = float(preset.get("probabilistic_confidence_min_mult", 0.70))
    st.session_state["universe_micro_prob_confidence_max_mult"] = float(preset.get("probabilistic_confidence_max_mult", 1.15))
    st.session_state["universe_micro_prob_overlay_strength"] = float(preset.get("probabilistic_overlay_strength", 1.00))
    st.session_state["universe_micro_prob_overlay_blend"] = float(preset.get("probabilistic_overlay_blend", 0.70))
    st.session_state["universe_overlay_compare_enabled"] = True
    st.session_state["universe_overlay_report_show_summary"] = True
    st.session_state["universe_overlay_report_show_internal_metrics"] = True
    st.session_state["universe_crash_overlay_last_applied"] = str(preset_name)
    st.toast(f"Crash overlay preset applied: {preset_name}", icon="🛡️")
def _download_dataframe_button(label: str, df: pd.DataFrame, file_name: str, *, key: str) -> None:
    out = _prepare_download_df(df)
    if out is None or out.empty:
        return
    st.download_button(
        label=label,
        data=out.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        key=key,
    )



@st.cache_data(show_spinner=False)
def _run_preset_sweep_cached(
    panel_df: pd.DataFrame,
    cfg_payload_json: str,
    preset_names: tuple[str, ...],
) -> pd.DataFrame:
    payload = json.loads(cfg_payload_json)
    rows = []
    for preset_name in preset_names:
        preset = ENGINE_PRESETS.get(str(preset_name), {})
        local = dict(payload)
        for k, v in preset.items():
            local[k] = v
        cfg = MicroPipelineConfig(**local)
        run = run_micro_investment_pipeline(panel_df, cfg=cfg)
        perf = _coerce_mapping(run.get("performance_summary", {}))
        risk = _coerce_mapping(run.get("risk_summary", {}))
        div = _coerce_mapping(run.get("diversification_summary", {}))
        uni = _coerce_mapping(run.get("universe_summary", {}))
        rows.append({
            "preset_name": str(preset_name),
            "sharpe": perf.get("sharpe"),
            "cagr": perf.get("cagr"),
            "annual_volatility": perf.get("annual_volatility", perf.get("annualized_volatility")),
            "max_drawdown": perf.get("max_drawdown"),
            "mean_turnover": perf.get("mean_turnover"),
            "active_return_annual": perf.get("active_return_annual"),
            "tracking_error_annual": perf.get("tracking_error_annual"),
            "information_ratio": perf.get("information_ratio"),
            "mean_effective_breadth": div.get("mean_effective_breadth"),
            "mean_diversification_ratio": div.get("mean_diversification_ratio"),
            "mean_effective_risk_bets": risk.get("mean_effective_risk_bets"),
            "mean_active_assets": uni.get("mean_active_assets"),
            "config_fingerprint": config_fingerprint(cfg),
        })
    return pd.DataFrame(rows)




def _build_parameter_stability_outputs(grid_df: pd.DataFrame, *, near_best_tol: float = 0.10) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if grid_df is None or not isinstance(grid_df, pd.DataFrame) or grid_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = grid_df.copy()
    for col in ["sharpe", "cagr", "mean_turnover", "temperature", "weight_shrink", "inertia"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "sharpe" not in df.columns or df["sharpe"].dropna().empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    best_sharpe = float(df["sharpe"].max())
    tol = max(float(near_best_tol), 0.0)
    threshold = best_sharpe - tol
    df["near_best_sharpe"] = pd.to_numeric(df["sharpe"], errors="coerce") >= threshold

    overall = pd.DataFrame([{
        "grid_rows": int(len(df)),
        "best_sharpe": best_sharpe,
        "median_sharpe": float(df["sharpe"].median()),
        "sharpe_std": float(df["sharpe"].std(ddof=1)) if len(df) > 1 else 0.0,
        "positive_sharpe_rate": float((df["sharpe"] > 0).mean()),
        "near_best_sharpe_rate": float(df["near_best_sharpe"].mean()),
        "median_cagr": float(pd.to_numeric(df.get("cagr"), errors="coerce").median()) if "cagr" in df.columns else np.nan,
        "median_turnover": float(pd.to_numeric(df.get("mean_turnover"), errors="coerce").median()) if "mean_turnover" in df.columns else np.nan,
    }])

    sharpe_std = float(df["sharpe"].std(ddof=1)) if len(df) > 1 else 0.0
    denom = sharpe_std if np.isfinite(sharpe_std) and sharpe_std > 1e-12 else 1.0
    resilient = df.copy()
    resilient["stability_score"] = (
        0.55 * (pd.to_numeric(resilient["sharpe"], errors="coerce") / denom)
        + 0.25 * resilient["near_best_sharpe"].astype(float)
        - 0.20 * pd.to_numeric(resilient.get("mean_turnover"), errors="coerce").fillna(0.0)
    )
    resilient = resilient.sort_values(["stability_score", "sharpe", "cagr"], ascending=[False, False, False]).reset_index(drop=True)

    rows = []
    for param in ["temperature", "weight_shrink", "inertia"]:
        if param not in df.columns:
            continue
        for value, sub in df.groupby(param, dropna=False):
            s = pd.to_numeric(sub["sharpe"], errors="coerce")
            rows.append({
                "parameter": param,
                "value": value,
                "n": int(len(sub)),
                "best_sharpe": float(s.max()) if not s.dropna().empty else np.nan,
                "median_sharpe": float(s.median()) if not s.dropna().empty else np.nan,
                "sharpe_std": float(s.std(ddof=1)) if len(s.dropna()) > 1 else 0.0,
                "positive_sharpe_rate": float((s > 0).mean()) if len(s.dropna()) else np.nan,
                "near_best_sharpe_rate": float(sub["near_best_sharpe"].mean()),
                "median_cagr": float(pd.to_numeric(sub.get("cagr"), errors="coerce").median()) if "cagr" in sub.columns else np.nan,
                "median_turnover": float(pd.to_numeric(sub.get("mean_turnover"), errors="coerce").median()) if "mean_turnover" in sub.columns else np.nan,
            })
    param_df = pd.DataFrame(rows)
    if not param_df.empty:
        param_df = param_df.sort_values(["parameter", "median_sharpe", "near_best_sharpe_rate"], ascending=[True, False, False]).reset_index(drop=True)

    return overall, resilient, param_df


def _build_sensitivity_analysis_outputs(
    grid_df: pd.DataFrame,
    *,
    metric_col: str = "sharpe",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if grid_df is None or not isinstance(grid_df, pd.DataFrame) or grid_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = grid_df.copy()
    numeric_cols = [
        "temperature", "weight_shrink", "inertia",
        "sharpe", "cagr", "information_ratio", "mean_turnover",
        "annual_volatility", "max_drawdown",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    metric = str(metric_col or "sharpe")
    if metric not in df.columns or df[metric].dropna().empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    global_metric = pd.to_numeric(df[metric], errors="coerce")
    global_median = float(global_metric.median()) if not global_metric.dropna().empty else np.nan
    global_best = float(global_metric.max()) if not global_metric.dropna().empty else np.nan

    rows = []
    detail_rows = []
    for param in ["temperature", "weight_shrink", "inertia"]:
        if param not in df.columns:
            continue
        grouped = []
        for value, sub in df.groupby(param, dropna=False):
            metric_series = pd.to_numeric(sub[metric], errors="coerce")
            if metric_series.dropna().empty:
                continue
            med = float(metric_series.median())
            best = float(metric_series.max())
            sd = float(metric_series.std(ddof=1)) if metric_series.dropna().shape[0] > 1 else 0.0
            hit = float((metric_series > 0).mean()) if metric == "sharpe" else np.nan
            detail_rows.append({
                "parameter": param,
                "value": value,
                f"median_{metric}": med,
                f"best_{metric}": best,
                f"std_{metric}": sd,
                "positive_rate": hit,
                "median_cagr": float(pd.to_numeric(sub.get("cagr"), errors="coerce").median()) if "cagr" in sub.columns else np.nan,
                "median_turnover": float(pd.to_numeric(sub.get("mean_turnover"), errors="coerce").median()) if "mean_turnover" in sub.columns else np.nan,
                "n": int(len(sub)),
            })
            grouped.append((value, med, best, sd))

        if not grouped:
            continue
        vals = pd.Series([g[1] for g in grouped], dtype="float64")
        range_med = float(vals.max() - vals.min()) if not vals.dropna().empty else np.nan
        std_med = float(vals.std(ddof=1)) if vals.dropna().shape[0] > 1 else 0.0
        rel = float(range_med / max(abs(global_median), 1e-9)) if np.isfinite(range_med) and np.isfinite(global_median) else np.nan
        best_idx = int(vals.idxmax()) if not vals.dropna().empty else 0
        stable_idx = int(pd.Series([g[3] for g in grouped], dtype="float64").idxmin()) if grouped else 0
        rows.append({
            "parameter": param,
            "grid_points": int(len(grouped)),
            f"range_median_{metric}": range_med,
            f"std_median_{metric}": std_med,
            "relative_sensitivity": rel,
            "best_value_by_median": grouped[best_idx][0] if grouped else np.nan,
            "most_stable_value": grouped[stable_idx][0] if grouped else np.nan,
        })

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty and "relative_sensitivity" in summary_df.columns:
        summary_df = summary_df.sort_values(["relative_sensitivity", f"range_median_{metric}"], ascending=[False, False]).reset_index(drop=True)

    overall = pd.DataFrame([{
        "metric": metric,
        "grid_rows": int(len(df)),
        "best_metric": global_best,
        "median_metric": global_median,
        "metric_std": float(global_metric.std(ddof=1)) if global_metric.dropna().shape[0] > 1 else 0.0,
        "most_sensitive_parameter": (summary_df.iloc[0]["parameter"] if not summary_df.empty else None),
        "max_relative_sensitivity": (float(summary_df.iloc[0]["relative_sensitivity"]) if not summary_df.empty else np.nan),
    }])

    detail_df = pd.DataFrame(detail_rows)
    if not detail_df.empty:
        detail_df = detail_df.sort_values(["parameter", f"median_{metric}"], ascending=[True, False]).reset_index(drop=True)

    return overall, summary_df, detail_df


def _build_breadth_vs_estimation_outputs(
    grid_df: pd.DataFrame,
    *,
    objective_col: str = "sharpe",
    n_bins: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if grid_df is None or not isinstance(grid_df, pd.DataFrame) or grid_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = grid_df.copy()
    numeric_cols = [
        "temperature", "weight_shrink", "inertia",
        "sharpe", "cagr", "information_ratio", "mean_turnover",
        "annual_volatility", "max_drawdown",
        "mean_effective_breadth", "mean_diversification_ratio",
        "mean_effective_risk_bets", "mean_active_assets",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    objective = str(objective_col or "sharpe")
    if objective not in df.columns or df[objective].dropna().empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    breadth_col = None
    for cand in ["mean_effective_breadth", "mean_effective_risk_bets", "mean_active_assets"]:
        if cand in df.columns and df[cand].dropna().shape[0] > 0:
            breadth_col = cand
            break
    if breadth_col is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def _z(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        mu = s.mean()
        sd = s.std(ddof=1)
        if pd.isna(sd) or sd <= 1e-12:
            return pd.Series(0.0, index=s.index, dtype="float64")
        return ((s - mu) / sd).astype("float64")

    temp_rank = pd.to_numeric(df.get("temperature"), errors="coerce").rank(pct=True)
    shrink_rank = pd.to_numeric(df.get("weight_shrink"), errors="coerce").rank(pct=True)
    inertia_rank = pd.to_numeric(df.get("inertia"), errors="coerce").rank(pct=True)
    df["estimation_load_proxy"] = pd.concat([temp_rank, 1.0 - shrink_rank, 1.0 - inertia_rank], axis=1).mean(axis=1)

    df["objective_z"] = _z(df[objective])
    df["breadth_z"] = _z(df[breadth_col])
    df["turnover_z"] = _z(df.get("mean_turnover", pd.Series(index=df.index, dtype="float64"))).fillna(0.0)
    df["estimation_load_z"] = _z(df["estimation_load_proxy"]).fillna(0.0)
    df["breadth_estimation_balance_score"] = (
        0.55 * df["objective_z"]
        + 0.30 * df["breadth_z"]
        - 0.30 * df["estimation_load_z"]
        - 0.20 * df["turnover_z"]
    )

    def _corr(a: str, b: str) -> float:
        if a not in df.columns or b not in df.columns:
            return float("nan")
        sub = df[[a, b]].dropna()
        if len(sub) < 2:
            return float("nan")
        return float(sub[a].corr(sub[b]))

    overall = pd.DataFrame([{
        "grid_rows": int(len(df)),
        "objective": objective,
        "breadth_metric": breadth_col,
        "best_objective": float(pd.to_numeric(df[objective], errors="coerce").max()),
        "median_objective": float(pd.to_numeric(df[objective], errors="coerce").median()),
        "median_breadth": float(pd.to_numeric(df[breadth_col], errors="coerce").median()),
        "median_estimation_load_proxy": float(pd.to_numeric(df["estimation_load_proxy"], errors="coerce").median()),
        "corr_breadth_objective": _corr(breadth_col, objective),
        "corr_breadth_turnover": _corr(breadth_col, "mean_turnover"),
        "corr_estimationload_objective": _corr("estimation_load_proxy", objective),
        "corr_estimationload_turnover": _corr("estimation_load_proxy", "mean_turnover"),
        "best_balance_score": float(pd.to_numeric(df["breadth_estimation_balance_score"], errors="coerce").max()),
    }])

    ranked = df.sort_values(["breadth_estimation_balance_score", objective, breadth_col], ascending=[False, False, False]).reset_index(drop=True)

    bins = max(int(n_bins), 2)
    bucket_df = pd.DataFrame()
    try:
        valid = pd.to_numeric(df[breadth_col], errors="coerce").dropna()
        if valid.shape[0] >= bins:
            tmp = df.copy()
            tmp["breadth_bucket"] = pd.qcut(pd.to_numeric(tmp[breadth_col], errors="coerce"), q=bins, duplicates="drop")
            rows = []
            for bucket, sub in tmp.groupby("breadth_bucket", dropna=True):
                rows.append({
                    "breadth_bucket": str(bucket),
                    "n": int(len(sub)),
                    f"median_{objective}": float(pd.to_numeric(sub[objective], errors="coerce").median()),
                    "median_turnover": float(pd.to_numeric(sub.get("mean_turnover"), errors="coerce").median()) if "mean_turnover" in sub.columns else np.nan,
                    "median_breadth": float(pd.to_numeric(sub[breadth_col], errors="coerce").median()),
                    "median_effective_risk_bets": float(pd.to_numeric(sub.get("mean_effective_risk_bets"), errors="coerce").median()) if "mean_effective_risk_bets" in sub.columns else np.nan,
                    "median_diversification_ratio": float(pd.to_numeric(sub.get("mean_diversification_ratio"), errors="coerce").median()) if "mean_diversification_ratio" in sub.columns else np.nan,
                    "median_estimation_load_proxy": float(pd.to_numeric(sub["estimation_load_proxy"], errors="coerce").median()),
                    "median_balance_score": float(pd.to_numeric(sub["breadth_estimation_balance_score"], errors="coerce").median()),
                })
            bucket_df = pd.DataFrame(rows)
    except Exception:
        bucket_df = pd.DataFrame()

    detail_cols = [c for c in [
        "temperature", "weight_shrink", "inertia", objective,
        breadth_col, "mean_turnover", "mean_effective_risk_bets",
        "mean_diversification_ratio", "mean_active_assets",
        "estimation_load_proxy", "breadth_estimation_balance_score", "config_fingerprint"
    ] if c in ranked.columns]
    detail_df = ranked[detail_cols].copy()

    rows = []
    for pname in ["temperature", "weight_shrink", "inertia"]:
        if pname not in df.columns:
            continue
        for value, sub in df.groupby(pname, dropna=False):
            rows.append({
                "parameter": pname,
                "value": value,
                f"median_{objective}": float(pd.to_numeric(sub[objective], errors="coerce").median()),
                "median_breadth": float(pd.to_numeric(sub[breadth_col], errors="coerce").median()),
                "median_turnover": float(pd.to_numeric(sub.get("mean_turnover"), errors="coerce").median()) if "mean_turnover" in sub.columns else np.nan,
                "median_estimation_load_proxy": float(pd.to_numeric(sub["estimation_load_proxy"], errors="coerce").median()),
                "median_balance_score": float(pd.to_numeric(sub["breadth_estimation_balance_score"], errors="coerce").median()),
                "n": int(len(sub)),
            })
    summary_df = pd.DataFrame(rows)
    return overall, summary_df, bucket_df, detail_df


def _parse_float_grid(raw: str, *, default: list[float]) -> list[float]:
    txt = str(raw or "").replace(";", ",").replace("\n", ",")
    vals = []
    for tok in txt.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            vals.append(float(tok))
        except Exception:
            continue
    if not vals:
        vals = list(default)
    out = []
    seen = set()
    for v in vals:
        key = round(float(v), 10)
        if key not in seen:
            seen.add(key)
            out.append(float(v))
    return out


@st.cache_data(show_spinner=False)
def _run_param_grid_cached(
    panel_df: pd.DataFrame,
    cfg_payload_json: str,
    temp_grid: tuple[float, ...],
    shrink_grid: tuple[float, ...],
    inertia_grid: tuple[float, ...],
) -> pd.DataFrame:
    payload = json.loads(cfg_payload_json)
    rows = []
    for temp in temp_grid:
        for shrink in shrink_grid:
            for inertia in inertia_grid:
                local = dict(payload)
                local["temperature"] = float(temp)
                local["weight_shrink"] = float(shrink)
                local["inertia"] = float(inertia)
                cfg = MicroPipelineConfig(**local)
                run = run_micro_investment_pipeline(panel_df, cfg=cfg)
                perf = _coerce_mapping(run.get("performance_summary", {}))
                risk = _coerce_mapping(run.get("risk_summary", {}))
                div = _coerce_mapping(run.get("diversification_summary", {}))
                uni = _coerce_mapping(run.get("universe_summary", {}))
                rows.append({
                    "temperature": float(temp),
                    "weight_shrink": float(shrink),
                    "inertia": float(inertia),
                    "sharpe": perf.get("sharpe"),
                    "cagr": perf.get("cagr"),
                    "annual_volatility": perf.get("annual_volatility", perf.get("annualized_volatility")),
                    "max_drawdown": perf.get("max_drawdown"),
                    "mean_turnover": perf.get("mean_turnover"),
                    "active_return_annual": perf.get("active_return_annual"),
                    "tracking_error_annual": perf.get("tracking_error_annual"),
                    "information_ratio": perf.get("information_ratio"),
                    "mean_effective_breadth": div.get("mean_effective_breadth"),
                    "mean_diversification_ratio": div.get("mean_diversification_ratio"),
                    "mean_effective_risk_bets": risk.get("mean_effective_risk_bets"),
                    "mean_active_assets": uni.get("mean_active_assets"),
                    "config_fingerprint": config_fingerprint(cfg),
                })
    return pd.DataFrame(rows)


def _parse_float_grid(raw: str, *, default: list[float]) -> list[float]:
    txt = str(raw or "").replace(";", ",").replace("\n", ",")
    vals = []
    for tok in txt.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            vals.append(float(tok))
        except Exception:
            continue
    if not vals:
        vals = list(default)
    out = []
    seen = set()
    for v in vals:
        key = round(float(v), 10)
        if key not in seen:
            seen.add(key)
            out.append(float(v))
    return out


@st.cache_data(show_spinner=False)
def _run_alpha_sweep_cached(
    panel_df: pd.DataFrame,
    cfg_payload_json: str,
    alpha_values: tuple[float, ...],
) -> pd.DataFrame:
    payload = json.loads(cfg_payload_json)
    return run_alpha_sweep(
        panel_df,
        base_cfg_payload=payload,
        alpha_values=list(alpha_values),
    )


@st.cache_data(show_spinner=False)
def _run_sharpe_turnover_frontier_cached(
    panel_df: pd.DataFrame,
    cfg_payload_json: str,
    temp_grid: tuple[float, ...],
    shrink_grid: tuple[float, ...],
    inertia_grid: tuple[float, ...],
) -> pd.DataFrame:
    payload = json.loads(cfg_payload_json)
    rows = []
    for temp in temp_grid:
        for shrink in shrink_grid:
            for inertia in inertia_grid:
                local = dict(payload)
                local["temperature"] = float(temp)
                local["weight_shrink"] = float(shrink)
                local["inertia"] = float(inertia)
                cfg = MicroPipelineConfig(**local)
                run = run_micro_investment_pipeline(panel_df, cfg=cfg)
                perf = _coerce_mapping(run.get("performance_summary", {}))
                risk = _coerce_mapping(run.get("risk_summary", {}))
                div = _coerce_mapping(run.get("diversification_summary", {}))
                rows.append({
                    "temperature": float(temp),
                    "weight_shrink": float(shrink),
                    "inertia": float(inertia),
                    "sharpe": perf.get("sharpe"),
                    "cagr": perf.get("cagr"),
                    "annual_volatility": perf.get("annual_volatility", perf.get("annualized_volatility")),
                    "max_drawdown": perf.get("max_drawdown"),
                    "mean_turnover": perf.get("mean_turnover"),
                    "active_return_annual": perf.get("active_return_annual"),
                    "tracking_error_annual": perf.get("tracking_error_annual"),
                    "information_ratio": perf.get("information_ratio"),
                    "mean_effective_breadth": div.get("mean_effective_breadth"),
                    "mean_diversification_ratio": div.get("mean_diversification_ratio"),
                    "mean_effective_risk_bets": risk.get("mean_effective_risk_bets"),
                    "config_fingerprint": config_fingerprint(cfg),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return build_sharpe_turnover_frontier(df, sharpe_col="sharpe", turnover_col="mean_turnover")


def _normalize_tuning_space_spec(spec: object) -> tuple[object | None, dict]:
    meta = {"status": "ok", "message": "", "effective_n": np.nan}

    if isinstance(spec, list):
        cleaned = []
        seen = set()
        for item in spec:
            key = json.dumps(item, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                cleaned.append(item)
        meta["effective_n"] = int(len(cleaned))
        if len(cleaned) == 0:
            meta.update({"status": "drop", "message": "empty_choice_list"})
            return None, meta
        if len(cleaned) == 1:
            meta.update({"status": "drop", "message": "single_choice_only"})
            return None, meta
        return cleaned, meta

    if isinstance(spec, tuple) and len(spec) == 2:
        lo, hi = spec
        try:
            lo_f = float(lo)
            hi_f = float(hi)
        except Exception:
            meta.update({"status": "drop", "message": "non_numeric_range"})
            return None, meta
        if not np.isfinite(lo_f) or not np.isfinite(hi_f):
            meta.update({"status": "drop", "message": "non_finite_range"})
            return None, meta
        if hi_f < lo_f:
            lo, hi = hi, lo
            lo_f, hi_f = hi_f, lo_f
        meta["effective_n"] = 2
        if abs(hi_f - lo_f) <= 1e-12:
            meta.update({"status": "drop", "message": "zero_width_range"})
            return None, meta
        if isinstance(lo, int) and isinstance(hi, int):
            return (int(lo), int(hi)), meta
        return (float(lo_f), float(hi_f)), meta

    meta.update({"status": "drop", "message": "fixed_or_unsupported_spec", "effective_n": 1})
    return None, meta


def _prepare_tuning_param_space(
    base_cfg_payload: dict,
    raw_param_space: dict,
    *,
    method: str = "bayesian_style",
) -> tuple[dict, pd.DataFrame, list[str]]:
    valid_fields = {f.name for f in fields(MicroPipelineConfig)}
    cleaned: dict = {}
    rows = []
    warnings: list[str] = []

    method_name = str(method or "bayesian_style").strip().lower()
    signal_mode = str(_coerce_mapping(base_cfg_payload).get("signal_mode", "") or "")
    probabilistic_mode = str(_coerce_mapping(base_cfg_payload).get("probabilistic_mode", "") or "")

    heuristic_drop = {
        "sigma_power_alpha": signal_mode not in {"mu_sigma", "huber_mu", "mu_sigma_fallback"},
        "top_k": bool(_coerce_mapping(base_cfg_payload).get("top_k") is None and method_name == "surrogate"),
    }
    heuristic_reason = {
        "sigma_power_alpha": f"inactive_for_signal_mode:{signal_mode or 'unknown'}",
        "top_k": "top_k_currently_none_for_surrogate",
    }

    for key, spec in (raw_param_space or {}).items():
        row = {
            "param": str(key),
            "status": "keep",
            "reason": "",
            "spec_type": type(spec).__name__,
            "current_value": _coerce_mapping(base_cfg_payload).get(key),
            "effective_n": np.nan,
            "signal_mode": signal_mode,
            "probabilistic_mode": probabilistic_mode,
        }
        if key not in valid_fields:
            row.update({"status": "drop", "reason": "not_in_micro_config"})
            rows.append(row)
            warnings.append(f"Dropped tuning param '{key}' because it is not a MicroPipelineConfig field.")
            continue
        normalized, meta = _normalize_tuning_space_spec(spec)
        row["effective_n"] = meta.get("effective_n")
        if normalized is None:
            row.update({"status": "drop", "reason": str(meta.get("message", "invalid_spec"))})
            rows.append(row)
            warnings.append(f"Dropped tuning param '{key}' because its search spec is degenerate ({row['reason']}).")
            continue
        if heuristic_drop.get(key, False):
            row.update({"status": "drop", "reason": heuristic_reason.get(key, "heuristic_inactive")})
            rows.append(row)
            warnings.append(f"Dropped tuning param '{key}' because it looks inactive for the current config ({row['reason']}).")
            continue
        cleaned[key] = normalized
        rows.append(row)

    audit_df = pd.DataFrame(rows)

    engine_audit_df = pd.DataFrame()
    try:
        cfg = MicroPipelineConfig(**_coerce_mapping(base_cfg_payload))
        engine_result = sanitize_param_space_for_engine(cleaned, cfg)
        engine_cleaned = _coerce_mapping(engine_result.get("sanitized_param_space", {}))
        engine_audit_df = engine_result.get("activity_audit", pd.DataFrame())
        if not isinstance(engine_audit_df, pd.DataFrame) or engine_audit_df.empty:
            engine_audit_df = build_engine_param_activity_audit(cfg, param_space=cleaned)
        for note in list(engine_result.get("notes", []) or []):
            warnings.append(str(note))
        for item in list(engine_result.get("dropped_params", []) or []):
            param = str(item.get("param", ""))
            reason = str(item.get("reason", "inactive in current engine context"))
            warnings.append(f"Dropped tuning param '{param}' because the engine marks it inactive ({reason}).")
        cleaned = engine_cleaned
    except Exception as e:
        warnings.append(f"Engine-aware tuning sanitisation failed; using local sanitised param space ({e}).")

    if isinstance(engine_audit_df, pd.DataFrame) and not engine_audit_df.empty:
        engine_cols = engine_audit_df.rename(columns={
            "active": "engine_active",
            "status": "engine_status",
            "reason": "engine_reason",
            "controller": "engine_controller",
        })[[c for c in ["param", "engine_active", "engine_status", "engine_reason", "engine_controller"] if c in engine_audit_df.rename(columns={"active": "engine_active", "status": "engine_status", "reason": "engine_reason", "controller": "engine_controller"}).columns]]
        if isinstance(audit_df, pd.DataFrame) and not audit_df.empty:
            audit_df = audit_df.merge(engine_cols, on="param", how="outer")
            audit_df["status"] = np.where(
                audit_df["param"].isin(list(cleaned.keys())),
                "keep",
                audit_df["status"].fillna("drop"),
            )
            audit_df["reason"] = audit_df["reason"].fillna("")
            drop_mask = (~audit_df["param"].isin(list(cleaned.keys()))) & (audit_df["reason"].astype(str).str.len() == 0)
            if "engine_reason" in audit_df.columns:
                audit_df.loc[drop_mask, "reason"] = audit_df.loc[drop_mask, "engine_reason"].fillna("")
        else:
            audit_df = engine_cols.copy()
            audit_df["status"] = np.where(audit_df["param"].isin(list(cleaned.keys())), "keep", "drop")
            audit_df["reason"] = np.where(audit_df["status"].eq("drop"), audit_df.get("engine_reason", ""), "")

    if len(cleaned) == 0:
        warnings.append("No active tuning parameters remain after param-space sanitisation.")
    elif len(cleaned) == 1:
        warnings.append("Only one active tuning parameter remains after param-space sanitisation; surrogate guidance may add limited value.")
    if method_name == "surrogate" and len(cleaned) < 2:
        warnings.append("Surrogate tuning usually benefits from at least two active dimensions; current search space is very small.")
    if isinstance(audit_df, pd.DataFrame) and not audit_df.empty:
        audit_df = audit_df.sort_values(["status", "param"], ascending=[True, True]).reset_index(drop=True)
    return cleaned, audit_df, warnings


def _summarize_random_trials(trials_df: pd.DataFrame) -> dict:
    if not isinstance(trials_df, pd.DataFrame) or trials_df.empty:
        return {}

    out: dict = {}
    for col in [
        "random_search_space_dim",
        "random_search_raw_param_count",
        "random_search_dropped_param_count",
        "random_search_unique_sample_count",
        "random_search_duplicate_rate",
    ]:
        if col in trials_df.columns:
            vals = pd.to_numeric(trials_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if not vals.empty:
                out[col] = float(vals.iloc[-1])

    if "random_search_space_status" in trials_df.columns:
        xs = trials_df["random_search_space_status"].dropna().astype(str)
        if not xs.empty:
            out["random_search_space_status"] = str(xs.iloc[-1])

    if "random_search_has_active_variation" in trials_df.columns:
        xs = pd.to_numeric(trials_df["random_search_has_active_variation"], errors="coerce").dropna()
        if not xs.empty:
            out["random_search_has_active_variation"] = float(xs.iloc[-1])

    if "random_search_dropped_params" in trials_df.columns:
        xs = trials_df["random_search_dropped_params"].dropna().astype(str)
        if not xs.empty:
            out["random_search_dropped_params"] = str(xs.iloc[-1])

    if "random_search_fixed_like_params" in trials_df.columns:
        xs = trials_df["random_search_fixed_like_params"].dropna().astype(str)
        if not xs.empty:
            out["random_search_fixed_like_params"] = str(xs.iloc[-1])

    if "random_search_invalid_like_params" in trials_df.columns:
        xs = trials_df["random_search_invalid_like_params"].dropna().astype(str)
        if not xs.empty:
            out["random_search_invalid_like_params"] = str(xs.iloc[-1])

    if "random_search_duplicate_sample" in trials_df.columns:
        dup = pd.to_numeric(trials_df["random_search_duplicate_sample"], errors="coerce").dropna()
        if not dup.empty:
            out["random_search_duplicate_sample_rate"] = float(dup.mean())

    return out


def _summarize_bayesian_trials(trials_df: pd.DataFrame) -> dict:
    if not isinstance(trials_df, pd.DataFrame) or trials_df.empty:
        return {}

    out: dict = {}
    numeric_last_cols = [
        "bayesian_search_space_dim",
        "bayesian_search_raw_param_count",
        "bayesian_search_dropped_param_count",
        "bayesian_search_requested_warmup",
        "bayesian_search_effective_warmup",
        "bayesian_search_local_width",
        "bayesian_search_unique_sample_count",
        "bayesian_search_duplicate_rate",
        "bayesian_search_incumbent_updates",
        "bayesian_search_best_val_so_far",
    ]
    for col in numeric_last_cols:
        if col in trials_df.columns:
            vals = pd.to_numeric(trials_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if not vals.empty:
                out[col] = float(vals.iloc[-1])

    if "bayesian_search_has_active_variation" in trials_df.columns:
        xs = pd.to_numeric(trials_df["bayesian_search_has_active_variation"], errors="coerce").dropna()
        if not xs.empty:
            out["bayesian_search_has_active_variation"] = float(xs.iloc[-1])

    for col in [
        "bayesian_search_space_status",
        "bayesian_search_dropped_params",
        "bayesian_search_fixed_like_params",
        "bayesian_search_invalid_like_params",
        "bayesian_search_phase",
    ]:
        if col in trials_df.columns:
            xs = trials_df[col].dropna().astype(str)
            if not xs.empty:
                out[col] = str(xs.iloc[-1])

    if "bayesian_search_used_incumbent" in trials_df.columns:
        xs = pd.to_numeric(trials_df["bayesian_search_used_incumbent"], errors="coerce").dropna()
        if not xs.empty:
            out["bayesian_search_used_incumbent_rate"] = float(xs.mean())

    if "bayesian_search_incumbent_improved" in trials_df.columns:
        xs = pd.to_numeric(trials_df["bayesian_search_incumbent_improved"], errors="coerce").dropna()
        if not xs.empty:
            out["bayesian_search_incumbent_improved_rate"] = float(xs.mean())

    if "bayesian_search_duplicate_sample" in trials_df.columns:
        xs = pd.to_numeric(trials_df["bayesian_search_duplicate_sample"], errors="coerce").dropna()
        if not xs.empty:
            out["bayesian_search_duplicate_sample_rate"] = float(xs.mean())

    return out


def _summarize_surrogate_trials(trials_df: pd.DataFrame) -> dict:
    if not isinstance(trials_df, pd.DataFrame) or trials_df.empty:
        return {}

    out = {}
    if "surrogate_used" in trials_df.columns:
        out["surrogate_used_rate"] = float(pd.to_numeric(trials_df["surrogate_used"], errors="coerce").fillna(0.0).mean())
    if "surrogate_ready" in trials_df.columns:
        out["surrogate_ready_rate"] = float(pd.to_numeric(trials_df["surrogate_ready"], errors="coerce").fillna(0.0).mean())
    if "surrogate_reason" in trials_df.columns:
        vc = trials_df["surrogate_reason"].astype(str).value_counts(dropna=False)
        out["surrogate_top_reason"] = str(vc.index[0]) if len(vc) else "-"
        out["surrogate_reason_counts"] = vc.to_dict()
    if "surrogate_feature_dim" in trials_df.columns:
        out["surrogate_feature_dim"] = float(pd.to_numeric(trials_df["surrogate_feature_dim"], errors="coerce").median())
    if "surrogate_n_fit" in trials_df.columns:
        out["surrogate_n_fit"] = float(pd.to_numeric(trials_df["surrogate_n_fit"], errors="coerce").max())
    if "surrogate_effective_warmup" in trials_df.columns:
        out["surrogate_effective_warmup"] = float(pd.to_numeric(trials_df["surrogate_effective_warmup"], errors="coerce").max())
    if "surrogate_candidate_pool_size" in trials_df.columns:
        out["surrogate_candidate_pool_size"] = float(pd.to_numeric(trials_df["surrogate_candidate_pool_size"], errors="coerce").median())
    if "surrogate_resid_std" in trials_df.columns:
        out["surrogate_resid_std"] = float(pd.to_numeric(trials_df["surrogate_resid_std"], errors="coerce").median())
    return out


def _summarize_optuna_trials(trials_df: pd.DataFrame) -> dict:
    if not isinstance(trials_df, pd.DataFrame) or trials_df.empty:
        return {}

    out = {}
    for c in ["optuna_available", "optuna_fallback_active", "optuna_has_active_variation"]:
        if c in trials_df.columns:
            out[f"{c}_rate"] = float(pd.to_numeric(trials_df[c], errors="coerce").fillna(0.0).mean())

    for c in [
        "optuna_space_dim", "optuna_raw_param_count", "optuna_dropped_param_count",
        "optuna_requested_trial_count", "optuna_trial_number", "optuna_best_value",
        "optuna_best_trial_number",
    ]:
        if c in trials_df.columns:
            out[c] = float(pd.to_numeric(trials_df[c], errors="coerce").median())

    for c in [
        "optuna_effective_method", "optuna_space_status", "optuna_requested_direction",
        "optuna_study_direction", "optuna_fallback_reason", "method",
    ]:
        if c in trials_df.columns:
            vc = trials_df[c].astype(str).value_counts(dropna=False)
            out[c] = str(vc.index[0]) if len(vc) else "-"

    if "optuna_available" in trials_df.columns:
        xs = pd.to_numeric(trials_df["optuna_available"], errors="coerce").fillna(0.0)
        out["optuna_any_available"] = bool((xs > 0).any())
    if "optuna_fallback_active" in trials_df.columns:
        xs = pd.to_numeric(trials_df["optuna_fallback_active"], errors="coerce").fillna(0.0)
        out["optuna_any_fallback"] = bool((xs > 0).any())
    return out


def _resolve_global_search_method(seed_strategy: str, exploitation_ratio: float) -> str:
    strategy = str(seed_strategy or "hybrid").strip().lower()
    ratio = float(np.clip(_safe_float(exploitation_ratio, 0.5), 0.0, 1.0))
    if strategy not in {"hybrid", "sobol", "lhs", "random"}:
        strategy = "hybrid"
    if strategy == "hybrid":
        if ratio <= 0.15:
            return "sobol"
        if ratio <= 0.30:
            return "lhs"
        return "hybrid"
    return strategy


def _apply_global_search_controls_to_schema(
    param_space: dict,
    *,
    base_cfg_payload: dict,
    categorical_breadth: str = "medium",
    exploitation_ratio: float = 0.5,
) -> dict:
    schema = _coerce_mapping(param_space)
    if not schema:
        return {}

    breadth = str(categorical_breadth or "medium").strip().lower()
    if breadth not in {"narrow", "medium", "wide"}:
        breadth = "medium"
    ratio = float(np.clip(_safe_float(exploitation_ratio, 0.5), 0.0, 1.0))
    base_payload = _coerce_mapping(base_cfg_payload)
    out: dict = {}

    def _is_number(x) -> bool:
        try:
            return np.isfinite(float(x))
        except Exception:
            return False

    for key, spec in schema.items():
        if isinstance(spec, dict):
            out[str(key)] = dict(spec)
            continue

        values = list(spec or [])
        if not values:
            continue
        current = base_payload.get(str(key))
        unique_vals = []
        seen = set()
        for v in values:
            token = json.dumps(v, sort_keys=True, default=str)
            if token in seen:
                continue
            seen.add(token)
            unique_vals.append(v)
        values = unique_vals
        if not values:
            continue

        if all(_is_number(v) for v in values):
            ordered = sorted(values, key=lambda v: (abs(float(v) - float(current)) if _is_number(current) else 0.0, float(v)))
            if ratio >= 0.80:
                keep_n = max(2, min(len(ordered), int(np.ceil(len(ordered) * 0.50))))
            elif ratio >= 0.60:
                keep_n = max(2, min(len(ordered), int(np.ceil(len(ordered) * 0.67))))
            else:
                keep_n = len(ordered)
            kept = ordered[:keep_n]
            kept = sorted(kept, key=lambda v: float(v))
            out[str(key)] = kept
        else:
            if breadth == "wide":
                keep_n = len(values)
            elif breadth == "narrow":
                keep_n = 2 if len(values) > 1 else 1
            else:
                keep_n = 3 if len(values) > 2 else len(values)
            ordered = list(values)
            if current in ordered:
                ordered = [current] + [v for v in ordered if v != current]
            out[str(key)] = ordered[:keep_n]
    return out


def _build_global_search_param_coverage_df(param_schema: dict, population_df: pd.DataFrame) -> pd.DataFrame:
    schema = _coerce_mapping(param_schema)
    pop = pd.DataFrame(population_df).copy()
    if not schema:
        return pd.DataFrame()

    rows = []
    for param, spec in schema.items():
        param = str(param)
        available = None
        if isinstance(spec, dict):
            vals = spec.get("values")
            if isinstance(vals, (list, tuple, set)):
                available = len(list(vals))
        elif isinstance(spec, (list, tuple, set)):
            available = len(list(spec))

        if param in pop.columns:
            observed = pop[param].dropna()
            if observed.empty:
                unique_n = 0
                sample_values = []
            else:
                as_str = observed.astype(str)
                unique_n = int(as_str.nunique(dropna=True))
                sample_values = list(as_str.drop_duplicates().head(8))
        else:
            unique_n = 0
            sample_values = []

        coverage_ratio = np.nan
        if available is not None and available > 0:
            coverage_ratio = float(unique_n) / float(available)

        rows.append({
            "param": param,
            "available_choices": available,
            "observed_unique": unique_n,
            "coverage_ratio": coverage_ratio,
            "sample_values": ", ".join(sample_values),
        })

    df = pd.DataFrame(rows)
    if not df.empty and "coverage_ratio" in df.columns:
        df = df.sort_values(["coverage_ratio", "observed_unique", "param"], ascending=[False, False, True], na_position="last").reset_index(drop=True)
    return df



@st.cache_data(show_spinner=False)
def _run_advanced_tuning_cached(
    panel_df: pd.DataFrame,
    cfg_payload_json: str,
    method: str,
    objective: str,
    n_trials: int,
    warmup: int,
    param_space_json: str,
    nsga2_objectives_json: str = "[]",
    nsga2_population_size: int = 24,
    nsga2_generations: int = 8,
    nsga2_mutation_rate: float = 0.15,
    nsga2_crossover_rate: float = 0.90,
    nsga2_seed: int = 42,
    recursive_search_enabled: bool = False,
    recursive_max_depth: int = 3,
    recursive_shrink_factor: float = 0.5,
    recursive_convergence_tol: float = 1e-3,
    recursive_seed_frontier_size: int = 8,
    global_total_budget: int = 200,
    global_seed_strategy: str = "hybrid",
):
    payload = json.loads(cfg_payload_json)
    param_space = json.loads(param_space_json)
    method_name = str(method or "bayesian_style").strip().lower()

    if method_name == "random":
        return run_random_tuning(
            panel_df,
            base_cfg_payload=payload,
            param_space=param_space,
            n_trials=int(n_trials),
            objective=str(objective),
        )
    if method_name == "surrogate":
        return run_surrogate_tuning(
            panel_df,
            base_cfg_payload=payload,
            param_space=param_space,
            n_trials=int(n_trials),
            objective=str(objective),
        )
    if method_name == "optuna":
        return run_optuna_tuning(
            panel_df,
            base_cfg_payload=payload,
            param_space=param_space,
            n_trials=int(n_trials),
            objective=str(objective),
        )
    if method_name == "global_multiobjective":
        try:
            objective_names = json.loads(nsga2_objectives_json or "[]")
        except Exception:
            objective_names = []
        objective_names = [str(x) for x in (objective_names or []) if str(x)]
        if not objective_names:
            objective_names = ["sharpe", "cagr", "max_drawdown", "mean_turnover", "diversification", "stability"]
        return run_global_multiobjective_search(
            panel_df,
            base_cfg_payload=payload,
            param_schema=param_space,
            objective_names=objective_names,
            budget=int(global_total_budget),
            method=str(global_seed_strategy or "hybrid"),
        )
    if method_name == "nsga2":
        try:
            objective_names = json.loads(nsga2_objectives_json or "[]")
        except Exception:
            objective_names = []
        objective_names = [str(x) for x in (objective_names or []) if str(x)]
        if not objective_names:
            objective_names = ["sharpe", "cagr", "max_drawdown", "mean_turnover", "diversification", "stability"]
        if bool(recursive_search_enabled):
            return run_recursive_multiobjective_search(
                panel_df,
                base_cfg_payload=payload,
                initial_param_space=param_space,
                objective_names=objective_names,
                search_method="nsga2",
                recursion_depth=int(recursive_max_depth),
                per_round_budget=int(nsga2_population_size),
                shrink_factor=float(recursive_shrink_factor),
                convergence_tol=float(recursive_convergence_tol),
                seed_frontier_size=int(recursive_seed_frontier_size),
            )
        return run_nsga2_tuning(
            panel_df,
            base_cfg_payload=payload,
            param_space=param_space,
            objective_names=objective_names,
            population_size=int(nsga2_population_size),
            generations=int(nsga2_generations),
            crossover_rate=float(nsga2_crossover_rate),
            mutation_rate=float(nsga2_mutation_rate),
            seed=int(nsga2_seed),
        )
    return run_bayesian_style_tuning(
        panel_df,
        base_cfg_payload=payload,
        param_space=param_space,
        n_trials=int(n_trials),
        warmup=int(warmup),
        objective=str(objective),
    )

def _render_run_report(run_name: str, result: dict, *, prefix: str) -> None:
    perf = result.get("performance_summary", {}) or {}
    risk = result.get("risk_summary", {}) or {}
    div = result.get("diversification_summary", {}) or {}
    uni = result.get("universe_summary", {}) or {}
    config_summary = _coerce_mapping(result.get("config_summary", {}))
    summary_df = result.get("summary_df", pd.DataFrame())

    if (
        not div
        or not uni
        or not config_summary
        or not isinstance(summary_df, pd.DataFrame)
        or summary_df.empty
    ):
        try:
            enriched_report = build_run_report(result, run_name=run_name)
        except Exception:
            enriched_report = {}
        sections = _coerce_mapping(enriched_report.get("sections", {}))
        if not div:
            div = _coerce_mapping(sections.get("diversification", {}))
        if not uni:
            uni = _coerce_mapping(sections.get("universe", {}))
        if not config_summary:
            config_summary = _coerce_mapping(sections.get("config", {}))
        if not isinstance(summary_df, pd.DataFrame) or summary_df.empty:
            summary_df = enriched_report.get("summary_df", pd.DataFrame())

    st.markdown(f"#### {run_name} detailed report")
    a, b, c, d = st.columns(4)
    with a:
        st.metric("Periods", int(perf.get("n_months", 0) or 0))
    with b:
        st.metric("Mean turnover", _format_pct_or_dash(perf.get("mean_turnover")))
    with c:
        st.metric("Eff. breadth", _format_num_or_dash(div.get("mean_effective_breadth")))
    with d:
        st.metric("Risk bets", _format_num_or_dash(risk.get("mean_effective_risk_bets")))

    e, f, g, h = st.columns(4)
    with e:
        st.metric("Active return (ann.)", _format_pct_or_dash(perf.get("active_return_annual")))
    with f:
        st.metric("Tracking error (ann.)", _format_pct_or_dash(perf.get("tracking_error_annual")))
    with g:
        st.metric("Information ratio", _format_num_or_dash(perf.get("information_ratio")))
    with h:
        st.metric("Div. ratio", _format_num_or_dash(div.get("mean_diversification_ratio")))

    i, j = st.columns(2)
    with i:
        st.metric("Mean active assets", _format_num_or_dash(uni.get("mean_active_assets")))
    with j:
        st.metric("Mean active return", _format_pct_or_dash(perf.get("mean_active_return")))

    if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
        with st.expander(f"{run_name} summary table", expanded=False):
            st.dataframe(summary_df, use_container_width=True)

    diag = result.get("diagnostics_df", pd.DataFrame())
    weights = result.get("weights_df", pd.DataFrame())
    group_weights_df = pd.DataFrame()
    if isinstance(weights, pd.DataFrame) and not weights.empty:
        try:
            weight_cols = [c for c in ["date", "asset", "weight"] if c in weights.columns]
            if {"asset", "weight"}.issubset(weights.columns):
                group_base = weights[weight_cols].copy()
                has_date = "date" in group_base.columns
                grouped_rows = []
                if has_date:
                    for dt, block in group_base.groupby("date", dropna=False):
                        agg = aggregate_weights_by_asset_group(
                            block[["asset", "weight"]],
                            asset_col="asset",
                            weight_col="weight",
                            include_subgroup=False,
                        )
                        if isinstance(agg, pd.DataFrame) and not agg.empty:
                            agg = agg.copy()
                            agg["date"] = dt
                            grouped_rows.append(agg)
                    if grouped_rows:
                        grouped_all = pd.concat(grouped_rows, ignore_index=True)
                        group_weights_df = (
                            grouped_all.groupby("group", as_index=False)["weight"]
                            .mean()
                            .rename(columns={"weight": "portfolio_weight"})
                        )
                else:
                    agg = aggregate_weights_by_asset_group(
                        group_base[["asset", "weight"]],
                        asset_col="asset",
                        weight_col="weight",
                        include_subgroup=False,
                    )
                    if isinstance(agg, pd.DataFrame) and not agg.empty:
                        group_weights_df = agg.rename(columns={"weight": "portfolio_weight"})

                if isinstance(group_weights_df, pd.DataFrame) and not group_weights_df.empty:
                    group_weights_df = group_weights_df.copy()
                    group_weights_df["portfolio_weight_pct"] = pd.to_numeric(
                        group_weights_df["portfolio_weight"], errors="coerce"
                    )
                    group_weights_df = group_weights_df.dropna(subset=["portfolio_weight_pct"])
                    group_weights_df = group_weights_df.sort_values(
                        ["portfolio_weight_pct", "group"], ascending=[False, True]
                    ).reset_index(drop=True)
        except Exception:
            group_weights_df = pd.DataFrame()
    warnings = result.get("universe_warnings", []) or []
    config_payload = config_to_dict(result.get("config", MicroPipelineConfig()))

    with st.expander(f"{run_name} diagnostics / config / downloads", expanded=False):
        fp = str(config_summary.get("config_fingerprint", config_fingerprint(result.get("config", MicroPipelineConfig()))))
        st.caption(f"Config fingerprint: {fp}")
        st.json(config_payload)
        if warnings:
            st.markdown("**Warnings**")
            for w in warnings:
                st.markdown(f"- {w}")
        if isinstance(diag, pd.DataFrame) and not diag.empty:
            st.markdown("**Diagnostics preview**")
            st.dataframe(diag.head(50), use_container_width=True)
        if isinstance(group_weights_df, pd.DataFrame) and not group_weights_df.empty:
            st.markdown("**Portfolio composition by group**")
            st.dataframe(
                group_weights_df[["group", "portfolio_weight_pct"]].style.format({"portfolio_weight_pct": "{:.2%}"}),
                use_container_width=True,
                hide_index=True,
            )
        if isinstance(weights, pd.DataFrame) and not weights.empty:
            st.markdown("**Weights preview**")
            st.dataframe(weights.head(50), use_container_width=True)

        d1, d2, d3, d4 = st.columns(4)
        with d1:
            _download_dataframe_button(f"Download {run_name} summary", summary_df, f"{prefix}_summary.csv", key=f"{prefix}_summary_download")
        with d2:
            _download_dataframe_button(f"Download {run_name} diagnostics", diag, f"{prefix}_diagnostics.csv", key=f"{prefix}_diag_download")
        with d3:
            _download_dataframe_button(f"Download {run_name} weights", weights, f"{prefix}_weights.csv", key=f"{prefix}_weights_download")
        with d4:
            st.download_button(
                label=f"Download {run_name} config",
                data=json.dumps(config_payload, indent=2, sort_keys=True).encode("utf-8"),
                file_name=f"{prefix}_config.json",
                mime="application/json",
                key=f"{prefix}_config_download",
            )


# ------------------------------------------------------------
# Session-state helper
# ------------------------------------------------------------
def ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default


def invalidate_position():
    """Any change to Step 1 inputs invalidates the confirmed snapshot & baseline."""
    st.session_state["position_confirmed"] = False
    st.session_state["baseline_ready"] = False
    st.session_state["baseline_df"] = None
    st.session_state["baseline_signature"] = None
    st.session_state["plan_ready"] = False
    st.session_state["plan_a_df"] = None
    st.session_state["plan_b_df"] = None


def make_baseline_signature():
    required = ["income_w", "fixed_total_w", "discretionary_w"]
    missing = [k for k in required if k not in st.session_state]
    if missing:
        raise RuntimeError(f"make_baseline_signature() called before weekly values exist: {missing}")

    # Signature includes advanced bits that affect trajectories
    shock_events = tuple(
        (float(r.get("amount", 0.0)), int(r.get("week", 0)))
        for r in st.session_state.get("shock_events_rows", [])
        if float(r.get("amount", 0.0)) > 0
    )

    return {
        "income_w": round(float(st.session_state["income_w"]), 6),
        "fixed_total_w": round(float(st.session_state["fixed_total_w"]), 6),
        "discretionary_w": round(float(st.session_state["discretionary_w"]), 6),
        "weeks": int(st.session_state.get("weeks", 12)),
        "preset_name": str(st.session_state.get("preset_name", "Quick estimate (default)")),
        "seed": int(st.session_state.get("seed", 42)),
        "shock_enabled": bool(st.session_state.get("shock_enabled", False)),
        "shock_events": shock_events,
    }


# ------------------------------------------------------------
# Session state defaults
# ------------------------------------------------------------
# Navigation / mode
ss("goal_mode", "Not sure yet")
ss("goal_selected", False)

# Raw inputs (widget-backed)
ss("income_raw", 460.0)
ss("income_period", "Weekly")

ss("fixed_essential_raw", 185.0)
ss("fixed_essential_period", "Weekly")

ss("variable_essential_raw", 80.0)
ss("variable_essential_period", "Weekly")

ss("discretionary_raw", 35.0)
ss("discretionary_period", "Weekly")
ss("discretionary_period_prev", st.session_state.get("discretionary_period", "Weekly"))

# Estimator state (optional)
ss("gross_annual_est", 30000.0)
ss("net_ratio_pct", 70)
ss("income_est_applied", False)

# Expense builder state
ss("fixed_items_rows", default_fixed_items_rows())

# Variable essentials breakdown state
ss("var_utilities_base", 0.0)
ss("var_utilities_period", "Weekly")
ss("var_season", "Normal")
ss("var_commute_days", 0)
ss("var_commute_cost", 0.0)
ss("var_groceries", 0.0)
ss("var_groceries_period", "Weekly")
ss("var_household", 0.0)
ss("var_household_period", "Weekly")
ss("variable_parts", {})
ss("variable_items_rows_weekly", [])
ss("variable_top_driver", None)

# Weekly normalised snapshot
ss("income_w", 0.0)
ss("fixed_total_w", 0.0)
ss("discretionary_w", 0.0)
ss("weekly_margin", 0.0)
ss("fixed_w", 0.0)
ss("var_w", 0.0)
ss("disc_w", 0.0)

# Confirm gate + baseline
ss("position_confirmed", False)
ss("baseline_df", None)
ss("baseline_ready", False)
ss("baseline_signature", None)

# Advanced controls
ss("weeks", 12)
ss("seed", 42)

ss("shock_enabled", False)
ss("shock_events_rows", [
    {"name": "", "amount": 0.0, "week": 1},
    {"name": "", "amount": 0.0, "week": 1},
    {"name": "", "amount": 0.0, "week": 1},
])

# Uncertainty presets
PRESETS = {
    "Quick estimate (default)": {"variability_pct": 30, "iterations": 150},
    "Typical spending": {"variability_pct": 30, "iterations": 250},
    "Unpredictable weeks": {"variability_pct": 50, "iterations": 250},
    "Stress test": {"variability_pct": 50, "iterations": 600},
}
ss("preset_name", "Quick estimate (default)")

# Plans
ss("show_plan_b", False)
ss("target_a_weekly", 0.0)
ss("target_b_weekly", 0.0)

ss("plan_ready", False)
ss("investment_setup_open", False)
ss("plan_a_df", None)
ss("plan_b_df", None)

UNIVERSE_MODE_STANDARD = "Generated universe"
UNIVERSE_MODE_LARGE = "Legacy large-universe preset"
UNIVERSE_MODE_CUSTOM = "Custom asset list"
UNIVERSE_MODE_OPTIONS = [
    UNIVERSE_MODE_STANDARD,
    UNIVERSE_MODE_LARGE,
    UNIVERSE_MODE_CUSTOM,
]

ss("universe_mode", UNIVERSE_MODE_STANDARD)
ss("universe_size", 12)
ss("universe_strategy", UNIVERSE_STRATEGY_CORE)
ss("universe_custom_enabled", False)
ss("universe_preset_name", "U12 Core Multi-Asset")
ss("custom_universe_text", ", ".join(UNIVERSE_PRESETS["U12 Core Multi-Asset"]))
ss("last_used_universe_assets", [])
ss("last_missing_universe_assets", [])
ss("comparison_enabled", False)
ss("comparison_universe_mode", UNIVERSE_MODE_STANDARD)
ss("comparison_universe_size", 12)
ss("comparison_universe_strategy", UNIVERSE_STRATEGY_DEFENSIVE)
ss("comparison_universe_custom_enabled", False)
ss("comparison_universe_preset_name", "Defensive Multi-Asset")
ss("comparison_custom_universe_text", "SPY, TLT, GLD, DBC")
ss("universe_large_preset_name", "50 assets — diversified liquid mix")
ss("comparison_large_preset_name", "50 assets — diversified liquid mix")
ss("last_comparison_universe_assets", [])
ss("last_comparison_missing_universe_assets", [])

ss("universe_large_test_preset_name", "50 assets — diversified liquid mix")
ss("universe_research_scale_preset_name", "200 assets — research scale test")
ss("universe_research_scale_target", "Primary universe")
ss("universe_research_scale_last_applied", "Not applied")
ss("universe_large_test_target", "Primary universe")
ss("universe_large_test_last_applied", "Not applied")
ss("universe_regime_universe_enabled", False)
ss("universe_regime_universe_metric", "mu_over_sigma")
ss("universe_regime_keep_low", 1.00)
ss("universe_regime_keep_mid", 0.85)
ss("universe_regime_keep_high", 0.60)
ss("universe_regime_min_assets", 2)
ss("universe_regime_max_assets_enabled", False)
ss("universe_regime_max_assets", 25)
ss("universe_regime_low_vol_tilt", 0.25)
ss("universe_regime_high_vol_tilt", 0.75)

ss("universe_micro_min_train", 120)
ss("universe_micro_lookback_mu", 12)
ss("universe_micro_lookback_sigma", 12)
ss("universe_micro_sigma_power_alpha", 1.0)
ss("universe_micro_temperature", 1.0)
ss("universe_micro_weight_shrink", 0.05)
ss("universe_micro_inertia", 0.0)
ss("universe_micro_deadband", True)
ss("universe_micro_deadband_threshold", 0.02)
ss("universe_micro_regime_mode", "quantile")
ss("universe_micro_mu_regime_mode", "pooled")
ss("universe_micro_ewma_sigma", True)
ss("universe_micro_ewma_halflife", 6)
ss("universe_micro_top_k_none", True)
ss("universe_micro_top_k", 5)

ss("universe_micro_asset_weight_cap_enabled", False)
ss("universe_micro_asset_weight_cap", 0.20)
ss("universe_micro_w_cap_enabled", False)
ss("universe_micro_w_cap", 0.10)
ss("universe_overlay_compare_enabled", True)
ss("universe_overlay_compare_show_detail", False)
ss("universe_overlay_report_show_summary", True)
ss("universe_cost_model_preset", "Off / legacy")
ss("universe_cost_model_enabled", False)
ss("universe_cost_model_compare_enabled", True)
ss("universe_cost_report_show_summary", True)
ss("universe_cost_report_show_timeseries", True)
ss("universe_cost_report_show_nav_chart", True)
ss("universe_cost_report_show_detail", False)
ss("universe_cost_report_show_diagnostics", False)
ss("universe_local_search_enabled", False)
ss("universe_local_search_objective", "composite_balanced")
ss("universe_local_search_intensity", "Quick")
ss("universe_local_search_include_weight_shrink", False)
ss("universe_local_search_show_detail", False)
ss("universe_local_search_two_stage", False)
ss("universe_local_refinement_history", [])
ss("universe_local_refinement_history_last_signature", "")
ss("universe_local_search_last_applied_fingerprint", "")
ss("universe_simple_auto_optimize_enabled", False)
ss("universe_simple_auto_opt_last_result", {})
ss("universe_simple_auto_opt_used", False)
ss("universe_simple_auto_opt_last_fingerprint", "")
ss("universe_transaction_cost_commission_bps", 0.0)
ss("universe_transaction_cost_slippage_bps", 0.0)
ss("universe_transaction_cost_spread_bps", 0.0)
ss("universe_transaction_cost_market_impact_bps", 0.0)
ss("universe_transaction_cost_market_impact_power", 1.0)
ss("universe_transaction_cost_min_trade_weight", 0.0)
ss("universe_holding_cost_annual_bps", 0.0)
ss("universe_tax_model_enabled", False)
ss("universe_tax_short_term_rate", 0.0)
ss("universe_tax_long_term_rate", 0.0)
ss("universe_tax_long_term_threshold_months", 12)
ss("universe_tax_apply_loss_credit", False)
ss("universe_tax_loss_credit_rate", 0.0)

ss("universe_factor_model_enabled", False)
ss("universe_factor_covariance_enabled", False)
ss("universe_factor_model_n_factors", 3)
ss("universe_factor_model_min_obs", 24)
ss("universe_factor_model_mu_blend", 0.35)
ss("universe_factor_model_residual_blend", 0.50)
ss("universe_factor_model_covariance_blend", 0.60)
ss("universe_factor_model_shrink_to_diagonal", 0.10)
ss("universe_factor_covariance_n_factors", 3)
ss("universe_factor_covariance_min_obs", 24)
ss("universe_factor_covariance_blend", 0.50)
ss("universe_factor_covariance_shrink_to_diagonal", 0.10)
ss("universe_factor_compare_enabled", True)
ss("universe_factor_report_show_summary", True)
ss("universe_factor_report_show_timeseries", True)
ss("universe_factor_report_show_nav_chart", True)
ss("universe_factor_report_show_detail", False)
ss("universe_factor_report_show_diagnostics", False)

ss("universe_micro_prob_mode", "none")
ss("universe_micro_prob_q_low", 0.25)
ss("universe_micro_prob_q_high", 0.75)
ss("universe_micro_prob_min_obs", 12)
ss("universe_micro_prob_interval_penalty_weight", 0.25)
ss("universe_micro_prob_downside_penalty_weight", 0.25)
ss("universe_micro_prob_confidence_scale", 1.0)
ss("universe_micro_prob_confidence_min_mult", 0.75)
ss("universe_micro_prob_confidence_max_mult", 1.25)
ss("universe_micro_prob_overlay_strength", 1.0)
ss("universe_micro_prob_overlay_blend", 1.0)
ss("universe_crash_overlay_enabled", False)
ss("universe_crash_overlay_preset_name", "Crash-aware balanced")
ss("universe_crash_overlay_last_applied", "Custom / manual")
ss("universe_preset_sweep_enabled", False)
ss("universe_preset_sweep_selected", list(ENGINE_PRESETS.keys()) if "ENGINE_PRESETS" in globals() else [])
ss("universe_param_sweep_enabled", False)
ss("universe_param_sweep_temp_grid_text", "0.7, 1.0, 1.3")
ss("universe_param_sweep_shrink_grid_text", "0.0, 0.05, 0.10")
ss("universe_param_sweep_inertia_grid_text", "0.0, 0.2, 0.4")
ss("universe_parameter_stability_enabled", False)
ss("universe_parameter_stability_temp_grid_text", "0.7, 1.0, 1.3")
ss("universe_parameter_stability_shrink_grid_text", "0.0, 0.05, 0.10")
ss("universe_parameter_stability_inertia_grid_text", "0.0, 0.2, 0.4")
ss("universe_parameter_stability_near_best_tol", 0.10)
ss("universe_parameter_stability_show_detail", False)
ss("universe_sensitivity_analysis_enabled", False)
ss("universe_sensitivity_temp_grid_text", "0.7, 1.0, 1.3")
ss("universe_sensitivity_shrink_grid_text", "0.0, 0.05, 0.10")
ss("universe_sensitivity_inertia_grid_text", "0.0, 0.2, 0.4")
ss("universe_sensitivity_metric", "sharpe")
ss("universe_sensitivity_show_detail", False)
ss("universe_robust_region_enabled", False)
ss("universe_robust_region_temp_grid_text", "0.7, 1.0, 1.3")
ss("universe_robust_region_shrink_grid_text", "0.0, 0.05, 0.10")
ss("universe_robust_region_inertia_grid_text", "0.0, 0.2, 0.4")
ss("universe_robust_region_sharpe_rel_tol", 0.95)
ss("universe_robust_region_max_turnover_enabled", False)
ss("universe_robust_region_max_turnover", 0.35)
ss("universe_robust_region_show_detail", False)
ss("universe_breadth_estimation_enabled", False)
ss("universe_breadth_estimation_temp_grid_text", "0.7, 1.0, 1.3")
ss("universe_breadth_estimation_shrink_grid_text", "0.0, 0.05, 0.10")
ss("universe_breadth_estimation_inertia_grid_text", "0.0, 0.2, 0.4")
ss("universe_breadth_estimation_metric", "sharpe")
ss("universe_breadth_estimation_bins", 3)
ss("universe_breadth_estimation_show_detail", False)
ss("universe_alpha_sweep_enabled", False)
ss("universe_alpha_sweep_grid_text", "0.5, 0.75, 1.0, 1.25, 1.5")
ss("universe_alpha_sweep_show_frontier", True)
ss("universe_frontier_temp_grid_text", "0.7, 1.0, 1.3")
ss("universe_frontier_shrink_grid_text", "0.0, 0.05, 0.10")
ss("universe_frontier_inertia_grid_text", "0.0, 0.2, 0.4")
ss("universe_frontier_max_turnover", 0.35)
ss("universe_frontier_enabled", False)
ss("universe_adv_tuning_enabled", False)
ss("universe_adv_tuning_method", "bayesian_style")
ss("universe_adv_tuning_objective", "sharpe")
ss("universe_adv_tuning_n_trials", 20)
ss("universe_adv_tuning_warmup", 5)
ss("universe_adv_tuning_temp_grid_text", "0.7, 1.0, 1.3")
ss("universe_adv_tuning_shrink_grid_text", "0.0, 0.05, 0.10")
ss("universe_adv_tuning_inertia_grid_text", "0.0, 0.2, 0.4")
ss("universe_adv_tuning_alpha_grid_text", "0.75, 1.0, 1.25")
ss("universe_adv_tuning_topk_grid_text", "")
ss("universe_adv_tuning_show_detail", False)
ss("universe_adv_tuning_nsga2_population_size", 24)
ss("universe_adv_tuning_nsga2_generations", 8)
ss("universe_adv_tuning_nsga2_mutation_rate", 0.15)
ss("universe_adv_tuning_nsga2_crossover_rate", 0.90)
ss("universe_adv_tuning_nsga2_seed", 42)
ss("universe_adv_tuning_nsga2_objectives", ["sharpe", "cagr", "max_drawdown", "mean_turnover", "diversification", "stability"])
ss("universe_adv_global_total_budget", 120)
ss("universe_adv_global_seed_strategy", "hybrid")
ss("universe_adv_global_categorical_breadth", "medium")
ss("universe_adv_global_exploitation_ratio", 0.50)
ss("universe_adv_global_resume_mode", "restart")
ss("universe_workspace_mode", "Simple")
ss("universe_global_search_run_nonce", 0)
ss("universe_micro_dispersion_sigma_enabled", False)
ss("universe_micro_dispersion_sigma_lookback", 12)
ss("universe_micro_dispersion_sigma_strength", 0.50)
ss("universe_micro_dispersion_sigma_floor_mult", 0.75)
ss("universe_micro_dispersion_sigma_ceiling_mult", 1.50)
ss("universe_micro_prob_qr_alpha", 0.0)
ss("universe_micro_prob_qr_solver", "highs")
ss("universe_micro_prob_qr_feature_cap", 8)
ss("universe_engine_preset_name", "Balanced Research Default")
ss("universe_engine_preset_last_applied", "Custom / manual")
ss("universe_micro_signal_mode", "mu_sigma")
ss("universe_micro_huber_delta", 1.0)
ss("universe_micro_signal_score_blend", 1.0)
ss("universe_micro_lambdarank_lookback", 12)
ss("universe_micro_lambdarank_temperature", 1.0)
ss("universe_micro_lambdarank_real_lookback", 18)
ss("universe_micro_lambdarank_real_temperature", 1.0)
ss("universe_micro_lambdarank_real_gain_power", 1.0)
ss("universe_micro_lambdarank_real_pair_power", 1.0)
ss("universe_micro_lambdarank_real_l2", 0.0)
ss("universe_multi_loss_enabled", False)
ss("universe_multi_loss_rank_weight", 0.40)
ss("universe_multi_loss_direction_weight", 0.30)
ss("universe_multi_loss_return_weight", 0.30)
ss("universe_multi_loss_logistic_weight", 0.0)
ss("universe_multi_loss_topk_weight", 0.0)
ss("universe_multi_loss_temperature", 1.0)
ss("universe_multi_loss_l2", 0.0)
ss("universe_micro_directional_classifier_lookback", 12)
ss("universe_micro_directional_classifier_threshold", 0.50)
ss("universe_micro_directional_classifier_confidence_scale", 1.0)
ss("universe_micro_logistic_loss_lookback", 18)
ss("universe_micro_logistic_loss_l2", 1e-3)
ss("universe_micro_logistic_loss_threshold", 0.50)
ss("universe_micro_logistic_loss_confidence_scale", 1.0)
ss("universe_micro_top_k_classifier_lookback", 12)
ss("universe_micro_top_k_classifier_k", 5)
ss("universe_micro_top_k_classifier_threshold", 0.50)
ss("universe_micro_top_k_classifier_confidence_scale", 1.0)
ss("universe_micro_quantile_loss_lookback", 18)
ss("universe_micro_quantile_loss_q", 0.60)
ss("universe_micro_quantile_loss_alpha", 0.0)
ss("universe_micro_quantile_loss_confidence_scale", 1.0)
ss("universe_run_pure_cs_baseline_compare", False)
ss("universe_show_pure_cs_report", False)
ss("universe_micro_signal_mode", "mu_sigma")
ss("universe_micro_huber_delta", 1.0)
ss("universe_micro_signal_score_blend", 1.0)
ss("universe_micro_lambdarank_lookback", 12)
ss("universe_micro_lambdarank_temperature", 1.0)

# ------------------------------------------------------------
# Small UI helpers
# ------------------------------------------------------------
def badge(text: str, kind: str = "info"):
    styles = {
        "info": ("rgba(59, 130, 246, 0.10)", "rgba(59, 130, 246, 0.25)", "rgba(30, 64, 175, 1)"),
        "ok": ("rgba(46, 204, 113, 0.14)", "rgba(46, 204, 113, 0.35)", "rgba(22, 101, 52, 1)"),
        "warn": ("rgba(245, 158, 11, 0.12)", "rgba(245, 158, 11, 0.28)", "rgba(146, 64, 14, 1)"),
        "bad": ("rgba(239, 68, 68, 0.10)", "rgba(239, 68, 68, 0.22)", "rgba(127, 29, 29, 1)"),
    }
    bg, border, fg = styles.get(kind, styles["info"])
    st.markdown(
        f"""
        <div style="
            width: 100%;
            box-sizing: border-box;
            padding: 0.65rem 0.9rem;
            border-radius: 0.6rem;
            background: {bg};
            border: 1px solid {border};
            color: {fg};
            font-weight: 600;
            margin-top: 0.55rem;
        ">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_discretionary_preset(preset_name: str):
    st.session_state["discretionary_raw"] = float(discretionary_preset_value(preset_name))
    st.session_state["discretionary_period"] = "Weekly"
    st.session_state["discretionary_period_prev"] = "Weekly"
    invalidate_position()
    st.toast(f"Preset applied: {preset_name}", icon="✅")


def on_discretionary_period_change():
    prev = st.session_state.get("discretionary_period_prev", "Weekly")
    curr = st.session_state.get("discretionary_period", "Weekly")
    raw = float(st.session_state.get("discretionary_raw", 0.0))
    weekly = to_weekly(raw, prev)
    new_raw = weekly_to_period(weekly, curr)
    st.session_state["discretionary_raw"] = round(float(new_raw), 2)
    st.session_state["discretionary_period_prev"] = curr
    invalidate_position()


def confirm_position():
    st.session_state["position_confirmed"] = True
    st.toast("Current situation locked. Building your baseline…", icon="✅")


# ------------------------------------------------------------
# Header (Goal-first)
# ------------------------------------------------------------
st.title("How much can I really save?")
st.caption("Answer 3 quick questions. Get a clear weekly plan. (You can refine later.)")

# ============================================================
# STEP 0 — GOAL (sets mental model)
# ============================================================
st.header("Step 0 — What are you trying to do?")

goal = st.radio(
    "Choose one (this only changes guidance, not your numbers).",
    options=[
        "Not sure yet",
        "Avoid overspending (break-even)",
        "Save more each week",
        "Reach a target balance in X weeks",
    ],
    index=0 if not st.session_state.get("goal_selected", False) else [
        "Not sure yet",
        "Avoid overspending (break-even)",
        "Save more each week",
        "Reach a target balance in X weeks",
    ].index(st.session_state.get("goal_mode", "Not sure yet")),
    key="goal_mode_radio",
)

st.session_state["goal_mode"] = goal
st.session_state["goal_selected"] = True

if goal == "Avoid overspending (break-even)":
    badge("Goal: get your weekly spending ≤ your weekly income. Then you can plan savings.", "info")
elif goal == "Save more each week":
    badge("Goal: pick a weekly savings target. We’ll tell you if it’s realistic and how much you must cut.", "info")
elif goal == "Reach a target balance in X weeks":
    badge("Goal: choose a target final balance and horizon. We’ll convert it into a weekly target.", "info")
else:
    badge("If you're not sure, we’ll start with your current weekly margin and suggest a realistic target.", "info")

# ============================================================
# STEP 1 — QUICK INPUTS (rough first, detail optional)
# ============================================================
st.divider()
st.header("Step 1 — Your current situation (rough numbers are fine)")

# ---- Income
c1, c2 = st.columns([2, 1])
with c1:
    st.number_input(
        "Take-home income (£)",
        min_value=0.0,
        step=10.0,
        key="income_raw",
        help="Net (after tax) income.",
        on_change=invalidate_position,
    )
with c2:
    st.selectbox(
        "Income period",
        ["Weekly", "Monthly", "Yearly"],
        key="income_period",
        on_change=invalidate_position,
    )

# Optional estimator (shorter copy)
with st.expander("Estimate take-home income (optional)"):
    st.caption("Rough estimator if you only know gross annual income.")
    st.number_input("Gross income (£/year)", min_value=0.0, step=500.0, key="gross_annual_est")
    st.slider("Estimated net ratio (%)", 40, 90, 70, 1, key="net_ratio_pct", format="%d%%")
    net_ratio = float(st.session_state["net_ratio_pct"]) / 100.0
    net_annual = float(st.session_state["gross_annual_est"]) * net_ratio
    net_weekly = net_annual / 52.0
    net_monthly = net_annual / 12.0
    st.markdown(f"**Estimated take-home:** £{net_weekly:,.2f}/week · £{net_monthly:,.2f}/month")
    if st.button("Use this estimate for my income", use_container_width=True):
        st.session_state["income_raw"] = float(net_weekly)
        st.session_state["income_period"] = "Weekly"
        st.session_state["income_est_applied"] = True
        invalidate_position()
        st.toast("Income updated from estimator", icon="✅")

# ---- Essentials (keep simple labels)
e1, e2 = st.columns([2, 1])
with e1:
    st.number_input(
        "Fixed essentials (£)",
        min_value=0.0,
        step=10.0,
        key="fixed_essential_raw",
        help="Rent, council tax, insurance, subscriptions you consider non-negotiable.",
        on_change=invalidate_position,
    )
with e2:
    st.selectbox(
        "Period",
        ["Weekly", "Monthly", "Yearly"],
        key="fixed_essential_period",
        on_change=invalidate_position,
    )

e3, e4 = st.columns([2, 1])
with e3:
    st.number_input(
        "Variable essentials (£)",
        min_value=0.0,
        step=5.0,
        key="variable_essential_raw",
        help="Groceries, commuting, utilities, household basics (things that vary).",
        on_change=invalidate_position,
    )
with e4:
    st.selectbox(
        "Period ",
        ["Weekly", "Monthly", "Yearly"],
        key="variable_essential_period",
        on_change=invalidate_position,
    )

# ---- Discretionary (presets keep as-is)
e5, e6 = st.columns([2, 1])
with e5:
    st.number_input(
        "Discretionary spending (£)",
        min_value=0.0,
        step=5.0,
        key="discretionary_raw",
        help="Flexible spending you can reduce if needed (social, shopping, entertainment, etc.).",
        on_change=invalidate_position,
    )
with e6:
    st.selectbox(
        "Period",
        ["Weekly", "Monthly", "Yearly"],
        key="discretionary_period",
        on_change=on_discretionary_period_change,
    )

p1, p2, p3 = st.columns(3)
for col, name in zip([p1, p2, p3], ["Quiet week", "Typical", "Social-heavy"]):
    with col:
        st.button(name, key=f"disc_preset_{name}", on_click=apply_discretionary_preset, args=(name,))

# ---- Optional detail (progressive disclosure)
with st.expander("Want to break essentials down in more detail? (optional)"):
    st.markdown("#### Build fixed essentials (optional)")
    fixed_rows = st.data_editor(
        st.session_state["fixed_items_rows"],
        hide_index=True,
        num_rows="dynamic",
        key="fixed_items_editor",
        column_config={
            "name": st.column_config.TextColumn("Name"),
            "amount": st.column_config.NumberColumn("Amount (£)", min_value=0.0),
            "period": st.column_config.SelectboxColumn("Period", options=["Weekly", "Monthly", "Yearly"]),
        },
        use_container_width=True,
    )
    st.session_state["fixed_items_rows"] = fixed_rows
    total_fixed = total_weekly_from_items(fixed_rows)
    st.markdown(f"**Estimated total:** £{total_fixed:,.2f}/week")
    if st.button("Use this total as my Fixed essentials", use_container_width=True):
        st.session_state["fixed_essential_raw"] = float(total_fixed)
        st.session_state["fixed_essential_period"] = "Weekly"
        invalidate_position()
        st.toast("Fixed essentials updated", icon="✅")

    st.markdown("---")
    st.markdown("#### Rough variable essentials breakdown (optional)")
    cA, cB = st.columns(2)
    with cA:
        st.number_input("Utilities baseline (£)", min_value=0.0, step=5.0, key="var_utilities_base")
        st.selectbox("Utilities period", ["Weekly", "Monthly", "Yearly"], key="var_utilities_period")
        st.selectbox("Season", ["Winter", "Normal", "Summer"], key="var_season")
        st.slider("Commute days/week", 0, 7, key="var_commute_days")
        st.number_input("Cost per commute day (£)", min_value=0.0, step=0.5, key="var_commute_cost")
    with cB:
        st.number_input("Groceries (£)", min_value=0.0, step=5.0, key="var_groceries")
        st.selectbox("Groceries period", ["Weekly", "Monthly", "Yearly"], key="var_groceries_period")
        st.number_input("Household basics (£)", min_value=0.0, step=5.0, key="var_household")
        st.selectbox("Household period", ["Weekly", "Monthly", "Yearly"], key="var_household_period")

    parts = variable_essentials_weekly_total(
        utilities_base=st.session_state["var_utilities_base"],
        utilities_period=st.session_state["var_utilities_period"],
        season=st.session_state["var_season"],
        commute_days=st.session_state["var_commute_days"],
        commute_cost_per_day=st.session_state["var_commute_cost"],
        groceries=st.session_state["var_groceries"],
        groceries_period=st.session_state["var_groceries_period"],
        household=st.session_state["var_household"],
        household_period=st.session_state["var_household_period"],
    )
    st.session_state["variable_parts"] = parts

    # Build a cleaned list for smarter tips
    LABEL_MAP = {
        "Utilities": "Utilities",
        "Commute": "Commuting",
        "Groceries": "Groceries",
        "Household": "Household basics",
    }
    variable_items_weekly = []
    for k, v in parts.items():
        if k.endswith("_weekly") and k != "total_weekly":
            raw = k.replace("_weekly", "")
            label = LABEL_MAP.get(raw.title(), raw.title())
            if float(v) > 0:
                variable_items_weekly.append({"name": label, "weekly": float(v)})

    # Merge duplicates
    merged = {}
    for row in variable_items_weekly:
        merged[row["name"]] = merged.get(row["name"], 0.0) + float(row["weekly"])
    variable_items_weekly = [{"name": k, "weekly": v} for k, v in merged.items()]
    variable_items_weekly.sort(key=lambda r: r["weekly"], reverse=True)

    st.session_state["variable_items_rows_weekly"] = variable_items_weekly
    st.session_state["variable_top_driver"] = variable_items_weekly[0] if variable_items_weekly else None

    st.markdown(f"**Estimated total:** £{parts['total_weekly']:,.2f}/week")
    if st.button("Use this total as my Variable essentials", use_container_width=True):
        st.session_state["variable_essential_raw"] = float(parts["total_weekly"])
        st.session_state["variable_essential_period"] = "Weekly"
        invalidate_position()
        st.toast("Variable essentials updated", icon="✅")

# ---- Confirm snapshot
st.button("Confirm my current situation", use_container_width=True, on_click=confirm_position)

if not st.session_state.get("position_confirmed", False):
    st.info("Enter rough numbers first, then confirm to see your weekly reality.")
    st.stop()

# ------------------------------------------------------------
# Weekly snapshot (single source of truth for the rest of the app)
# ------------------------------------------------------------
income_w = to_weekly(st.session_state["income_raw"], st.session_state["income_period"])
fixed_w = to_weekly(st.session_state["fixed_essential_raw"], st.session_state["fixed_essential_period"])
var_w = to_weekly(st.session_state["variable_essential_raw"], st.session_state["variable_essential_period"])
disc_w = to_weekly(st.session_state["discretionary_raw"], st.session_state["discretionary_period"])

fixed_total_w = fixed_w + var_w
margin_w = income_w - fixed_total_w - disc_w

st.session_state["income_w"] = float(income_w)
st.session_state["fixed_total_w"] = float(fixed_total_w)
st.session_state["discretionary_w"] = float(disc_w)
st.session_state["weekly_margin"] = float(margin_w)

st.session_state["fixed_w"] = float(fixed_w)
st.session_state["var_w"] = float(var_w)
st.session_state["disc_w"] = float(disc_w)

# Persist itemised fixed rows weekly for smarter deficit tips
def _rows_with_weekly(rows):
    out = []
    for r in (rows or []):
        try:
            name = str(r.get("name", "")).strip()
            amount = float(r.get("amount", 0.0) or 0.0)
            period = str(r.get("period", "Weekly") or "Weekly")
            weekly = float(to_weekly(amount, period))
            if weekly > 0:
                out.append({"name": name, "weekly": weekly})
        except Exception:
            continue
    return out


st.session_state["fixed_items_rows_weekly"] = _rows_with_weekly(st.session_state.get("fixed_items_rows", []))

# ============================================================
# STEP 1 OUTPUT — "Weekly reality" (big meaning + simple chart)
# ============================================================
st.subheader("Your weekly reality")

if margin_w >= 0:
    st.success(f"You can safely save about **£{margin_w:,.0f}/week** (without changing anything).")
else:
    st.error(f"You are overspending by about **£{abs(margin_w):,.0f}/week**.")

# Stacked bar (keep it visual)
segments = []
if margin_w >= 0:
    segments = [("Fixed", fixed_w), ("Variable", var_w), ("Discretionary", disc_w), ("Margin", margin_w)]
else:
    segments = [("Fixed", fixed_w), ("Variable", var_w), ("Discretionary", disc_w), ("Deficit", abs(margin_w))]
segments = [(label, value) for label, value in segments if value > 0]

COLOR_MAP = {
    "Fixed": "#4C72B0",
    "Variable": "#55A868",
    "Discretionary": "#DD8452",
    "Margin": "#64B5CD",
    "Deficit": "#C44E52",
}

stack_total = sum(v for _, v in segments)
fig, ax = plt.subplots(figsize=(9.5, 3.0))

left = 0.0
for label, value in segments:
    ax.barh(y=["Weekly"], width=[value], left=[left], color=COLOR_MAP[label])
    left += value

ax.axvline(income_w, linestyle="--", linewidth=1.5)
ax.set_xlabel("£ per week")
ax.set_title("Where your weekly money goes", pad=10)
ax.legend().remove()

max_x = max(float(income_w), float(stack_total))
ax.set_xlim(0, max_x * 1.05 if max_x > 0 else 1)

fig.subplots_adjust(bottom=0.48)
x_cols = [0.22, 0.62]
y_rows = [0.14, 0.06]


def _pct(v: float) -> float:
    return (v / income_w * 100.0) if income_w > 0 else 0.0


def _draw_item(i: int, label: str, value: float):
    row = 0 if i < 2 else 1
    col = i % 2
    x = x_cols[col]
    y = y_rows[row]

    fig.text(x - 0.045, y, "■", ha="right", va="center", fontsize=15, color=COLOR_MAP[label])
    fig.text(
        x,
        y,
        f"{label}: £{value:,.0f} ({_pct(value):.0f}%)",
        ha="left",
        va="center",
        fontsize=12,
        color="black",
    )


for i, (label, value) in enumerate(segments[:4]):
    _draw_item(i, label, value)

st.pyplot(fig, clear_figure=True)

with st.expander("See monthly/yearly equivalents (optional)"):
    st.write(f"Income: **£{income_w:,.0f}/w** ≈ **£{weekly_to_period(income_w, 'Monthly'):,.0f}/mo**")
    st.write(f"Essentials: **£{fixed_total_w:,.0f}/w** ≈ **£{weekly_to_period(fixed_total_w, 'Monthly'):,.0f}/mo**")
    st.write(f"Discretionary: **£{disc_w:,.0f}/w** ≈ **£{weekly_to_period(disc_w, 'Monthly'):,.0f}/mo**")
    st.write(f"Leftover (Margin): **£{margin_w:,.0f}/w** ≈ **£{weekly_to_period(margin_w, 'Monthly'):,.0f}/mo**")

# ============================================================
# STEP 2 — PLAN (Plan A default + Plan B optional)
# ============================================================
st.divider()
st.header("Step 2 — Choose your savings goal")

baseline_margin = max(float(margin_w), 0.0)
disc = float(disc_w)
max_possible_margin = float(margin_w) + float(disc_w)
structural_deficit = max_possible_margin < 0  # even cutting disc to 0 can't break-even

if structural_deficit:
    badge(
        "You have a structural deficit: even cutting discretionary to £0/week won’t fully fix it. "
        "This tool can still estimate, but you’ll likely need a structural change (income ↑ or essentials ↓).",
        "bad",
    )
else:
    if margin_w < 0:
        badge(
            "First priority: reach break-even. Any savings goal assumes you’ve eliminated the deficit.",
            "warn",
        )
    else:
        badge(
            "Your baseline already saves your positive margin automatically. Now pick a weekly target to improve on it.",
            "info",
        )

# Auto-suggest button
cL, cR = st.columns([2, 1])
with cL:
    st.caption("Tip: if you’re unsure, start with a realistic target.")
with cR:
    if st.button("Suggest a realistic target", use_container_width=True):
        st.session_state["target_a_weekly"] = round(float(max(margin_w, 0.0)), 2)
        st.toast("Suggested target applied to Plan A.", icon="✅")

# Plan A input
st.markdown("### Plan A — My main goal")
st.markdown('<div id="plan-a-target-anchor"></div>', unsafe_allow_html=True)
st.number_input(
    "I want to save (£ per week)",
    min_value=0.0,
    step=5.0,
    key="target_a_weekly",
)
plan_a_target_weekly = float(st.session_state.get("target_a_weekly", 0.0) or 0.0)
plan_a_target_ready_for_investment = bool(np.isfinite(plan_a_target_weekly) and plan_a_target_weekly > 0.0)
if plan_a_target_ready_for_investment:
    st.session_state["investment_setup_gate_attempted"] = False

# Plan B removed from the streamlined Step 2 single-plan flow
st.session_state["show_plan_b"] = False
st.session_state["target_b_weekly"] = 0.0
st.session_state["plan_b_df"] = None

# ------------------------------------------------------------
# Advanced settings (hide complexity)
# ------------------------------------------------------------
with st.expander("Advanced settings (optional)"):
    st.slider(
        "Planning horizon (weeks)",
        min_value=4,
        max_value=52,
        value=int(st.session_state.get("weeks", 12)),
        key="weeks",
        help="Shorter horizons are easier to interpret.",
    )

    st.selectbox(
        "Uncertainty preset",
        list(PRESETS.keys()),
        index=list(PRESETS.keys()).index(st.session_state.get("preset_name", "Quick estimate (default)")),
        key="preset_name",
        help="Controls how unpredictable discretionary spending is in the simulation.",
    )

    st.caption("Randomness is kept stable unless you redraw.")
    if st.button("Try another random run", use_container_width=True):
        st.session_state["seed"] = int(st.session_state.get("seed", 42)) + 1
        st.toast("New random draw applied.", icon="🎲")

    st.checkbox("Enable one-off events (unexpected expenses)", key="shock_enabled")
    shock_rows = st.data_editor(
        st.session_state.get("shock_events_rows", []),
        key="shock_events_editor",
        hide_index=True,
        num_rows=3,
        disabled=not st.session_state.get("shock_enabled", False),
        column_config={
            "name": st.column_config.TextColumn("Event (optional)"),
            "amount": st.column_config.NumberColumn("Amount (£)", min_value=0.0),
            "week": st.column_config.NumberColumn("Week", min_value=1, max_value=int(st.session_state.get("weeks", 12))),
        },
        use_container_width=True,
    )
    st.session_state["shock_events_rows"] = shock_rows

# ------------------------------------------------------------
# Build shock map (if enabled)
# ------------------------------------------------------------
weeks = int(st.session_state.get("weeks", 12))
shock_map = {}
if st.session_state.get("shock_enabled", False):
    rows = clean_events(st.session_state.get("shock_events_rows", []), weeks=weeks)
    if rows:
        shock_map = events_to_weekly_shock_map(rows)

# ------------------------------------------------------------
# Auto-generate baseline if needed (no button)
# ------------------------------------------------------------
sig_now = make_baseline_signature()
if (not st.session_state.get("baseline_ready", False)) or (st.session_state.get("baseline_signature") != sig_now):
    baseline_raw = generate_baseline(
        float(st.session_state["income_w"]),
        float(st.session_state["fixed_total_w"]),
        float(st.session_state["discretionary_w"]),
        int(weeks),
    ).round(2)

    try:
        validate_baseline_df(baseline_raw)
    except (ValueError, KeyError) as e:
        st.error(f"Baseline validation failed: {e}")
        st.stop()

    baseline_adj = apply_shock_map_to_df(
        baseline_raw,
        shock_map=shock_map,
        value_cols=("Balance",),
    )

    st.session_state["baseline_df"] = baseline_adj
    st.session_state["baseline_ready"] = True
    st.session_state["baseline_signature"] = sig_now

badge(f"Baseline built for **{weeks} weeks**. Ready to evaluate your plan.", "ok")

# ============================================================
# Compute cuts needed for targets (Plan semantics)
# ------------------------------------------------------------
def compute_required_cut(target_weekly: float) -> float:
    """
    Target semantics:
    - baseline already keeps positive margin as savings
    - to reach a higher target, we must increase weekly savings by:
        need = max(target - max(margin,0), 0)
      achieved by cutting discretionary (only lever in MVP)
    """
    need = max(float(target_weekly) - max(float(margin_w), 0.0), 0.0)
    return float(need)


def cap_cut_to_discretionary(cut_weekly: float) -> float:
    return float(min(max(cut_weekly, 0.0), max(disc, 0.0)))


target_a = float(st.session_state.get("target_a_weekly", 0.0))
cut_a = cap_cut_to_discretionary(compute_required_cut(target_a))

target_b = float(st.session_state.get("target_b_weekly", 0.0)) if st.session_state.get("show_plan_b", False) else 0.0
cut_b = cap_cut_to_discretionary(compute_required_cut(target_b)) if st.session_state.get("show_plan_b", False) else 0.0

baseline_savings_weekly = float(max(margin_w, 0.0))
target_a_monthly = float(weekly_to_period(target_a, "Monthly"))
baseline_savings_monthly = float(weekly_to_period(baseline_savings_weekly, "Monthly"))
cut_a_monthly = float(weekly_to_period(cut_a, "Monthly"))
target_b_monthly = float(weekly_to_period(target_b, "Monthly")) if st.session_state.get("show_plan_b", False) else 0.0
cut_b_monthly = float(weekly_to_period(cut_b, "Monthly")) if st.session_state.get("show_plan_b", False) else 0.0

st.session_state["planning_snapshot"] = {
    "margin_weekly": float(margin_w),
    "baseline_savings_weekly": float(baseline_savings_weekly),
    "target_a_weekly": float(target_a),
    "required_cut_a_weekly": float(cut_a),
    "target_b_weekly": float(target_b),
    "required_cut_b_weekly": float(cut_b),
    "margin_monthly": float(weekly_to_period(margin_w, "Monthly")),
    "baseline_savings_monthly": float(baseline_savings_monthly),
    "target_a_monthly": float(target_a_monthly),
    "required_cut_a_monthly": float(cut_a_monthly),
    "target_b_monthly": float(target_b_monthly),
    "required_cut_b_monthly": float(cut_b_monthly),
    "structural_deficit": bool(structural_deficit),
}
_store_investment_context(st.session_state["planning_snapshot"])

# Inform feasibility (assistant-style)
def plan_feasibility_line(label: str, target: float, cut: float):
    if structural_deficit:
        st.warning(f"{label}: Target set to £{target:,.0f}/week, but you’re in structural deficit territory.")
        return
    if margin_w < 0:
        # In deficit: interpret target as "after break-even"
        st.info(
            f"{label}: You want £{target:,.0f}/week. First you must eliminate the deficit "
            f"(≈ £{abs(margin_w):,.0f}/week)."
        )
    else:
        if target <= max(margin_w, 0.0):
            st.success(f"{label}: £{target:,.0f}/week is already achievable without changes.")
        else:
            need = target - max(margin_w, 0.0)
            if need <= disc:
                st.warning(
                    f"{label}: To reach £{target:,.0f}/week, you must cut discretionary by about **£{need:,.0f}/week**."
                )
            else:
                st.error(
                    f"{label}: To reach £{target:,.0f}/week you'd need to cut **£{need:,.0f}/week**, "
                    f"but your discretionary is only **£{disc:,.0f}/week**."
                )


st.markdown("#### Quick feasibility check")
plan_feasibility_line("Plan A", target_a, cut_a)
if st.session_state.get("show_plan_b", False):
    plan_feasibility_line("Plan B", target_b, cut_b)

# ============================================================
# Run simulations (only if baseline exists; keep it lightweight)
# ============================================================
preset_cfg = PRESETS[str(st.session_state.get("preset_name", "Quick estimate (default)"))]
iterations = int(preset_cfg["iterations"])
variability_pct = int(preset_cfg["variability_pct"])
variability_frac = float(variability_pct) / 100.0
seed = int(st.session_state.get("seed", 42))

# Simulate Plan A
plan_a_df, params_a = simulate_scenario_df(
    income=float(st.session_state["income_w"]),
    fixed_expenses=float(st.session_state["fixed_total_w"]),
    variable_expenses=float(st.session_state["discretionary_w"]),
    delta_savings=float(cut_a),  # simulator expects a cut in variable spending
    weeks=int(weeks),
    iterations=int(iterations),
    seed=int(seed),
    variability_frac=float(variability_frac),
)

# Apply shocks to scenario columns too (so charts line up)
plan_a_df = apply_shock_map_to_df(plan_a_df, shock_map=shock_map, value_cols=("Mean", "Lower", "Upper"))
st.session_state["plan_a_df"] = plan_a_df

plan_b_df = None
if st.session_state.get("show_plan_b", False):
    plan_b_df, params_b = simulate_scenario_df(
        income=float(st.session_state["income_w"]),
        fixed_expenses=float(st.session_state["fixed_total_w"]),
        variable_expenses=float(st.session_state["discretionary_w"]),
        delta_savings=float(cut_b),
        weeks=int(weeks),
        iterations=int(iterations),
        seed=int(seed + 1),  # slight offset for variety
        variability_frac=float(variability_frac),
    )
    plan_b_df = apply_shock_map_to_df(plan_b_df, shock_map=shock_map, value_cols=("Mean", "Lower", "Upper"))
    st.session_state["plan_b_df"] = plan_b_df

# ============================================================
# STEP 3 — RESULTS (numbers first, chart second, tech hidden)
# ============================================================
st.divider()
st.header("Step 3 — Your plan outcome")

baseline_df = st.session_state["baseline_df"]
base_final = float(baseline_df["Balance"].iloc[-1])

a_final_mean = float(plan_a_df["Mean"].iloc[-1])
a_final_low = float(plan_a_df["Lower"].iloc[-1])
a_final_high = float(plan_a_df["Upper"].iloc[-1])

# Big numbers first
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Baseline final balance", f"£{base_final:,.0f}")
with col2:
    st.metric("Conservative", f"£{a_final_low:,.0f}")
with col3:
    st.metric("Expected", f"£{a_final_mean:,.0f}")
with col4:
    st.metric("Optimistic", f"£{a_final_high:,.0f}")

st.caption("These are your plan outcomes across the 10th percentile, median path, and 90th percentile of the simulation.")

# Simple chart: baseline + mean lines (+ optional bands)
st.markdown("### What happens over time")

fig2, ax2 = plt.subplots(figsize=(9.5, 4.2))

ax2.plot(baseline_df["Week"], baseline_df["Balance"], label="Baseline", linewidth=2.5)
ax2.plot(plan_a_df["Week"], plan_a_df["Mean"], label="Plan A (mean)", linewidth=2.5)
ax2.fill_between(plan_a_df["Week"], plan_a_df["Lower"], plan_a_df["Upper"], alpha=0.15, label="Plan A (10–90%)")

if plan_b_df is not None:
    ax2.plot(plan_b_df["Week"], plan_b_df["Mean"], label="Plan B (mean)", linewidth=2.5)
    ax2.fill_between(plan_b_df["Week"], plan_b_df["Lower"], plan_b_df["Upper"], alpha=0.12, label="Plan B (10–90%)")

ax2.set_xlabel("Week")
ax2.set_ylabel("Balance (£)")
ax2.set_title("Projected balance over time")
ax2.axhline(0, linewidth=1.0, linestyle="--")
ax2.legend()
st.pyplot(fig2, clear_figure=True)

# ============================================================
# Assistant-style advice (short), technical explanation hidden
# ============================================================
st.markdown("### What to do next (short)")

if structural_deficit:
    breakdown = {
        "income_w": float(st.session_state.get("income_w", 0.0)),
        "fixed_total_w": float(st.session_state.get("fixed_total_w", 0.0)),
        "discretionary_w": float(st.session_state.get("discretionary_w", 0.0)),
        "margin_w": float(st.session_state.get("weekly_margin", 0.0)),
        "fixed_items_rows": st.session_state.get("fixed_items_rows_weekly", []),
        "variable_items_rows_weekly": st.session_state.get("variable_items_rows_weekly", []),
        "variable_parts": st.session_state.get("variable_parts", {}),
    }
    deficit_w = abs(float(margin_w))
    tips = build_structural_deficit_tips(breakdown=breakdown, deficit_w=deficit_w, top_n=3)
    st.warning("You’re in structural deficit territory. Here are the highest-leverage fixes based on your breakdown.")
    st.markdown(tips)
else:
    if margin_w >= 0 and target_a <= max(margin_w, 0.0):
        st.success("Plan A is already covered by your current margin. Consider setting a slightly higher target.")
    else:
        need = max(target_a - max(margin_w, 0.0), 0.0)
        if need <= disc:
            st.info(
                f"To reach Plan A (£{target_a:,.0f}/week), aim to reduce discretionary by about **£{need:,.0f}/week**. "
                "Start with a small, consistent cut rather than a perfect plan."
            )
        else:
            st.error(
                "Plan A is not achievable through discretionary cuts alone. "
                "You’ll need a structural change (income ↑ or essentials ↓)."
            )


# Technical details (hidden)
with st.expander("See detailed explanation (technical)"):
    # Build technical explanation inputs (delta = cuts used)
    inputs = ExplanationInputs(
        income=float(st.session_state["income_w"]),
        fixed_expenses=float(st.session_state["fixed_total_w"]),
        variable_expenses=float(st.session_state["discretionary_w"]),
        weeks=int(weeks),
        delta_a=float(cut_a),
        delta_b=float(cut_b if plan_b_df is not None else 0.0),
        variability_pct=float(variability_pct),
        seed=int(seed),
        iters=int(iterations),
    )

    # Compute reflection metrics + human text (if Plan B exists)
    if plan_b_df is not None:
        b_final_mean = float(plan_b_df["Mean"].iloc[-1])
        b_final_low = float(plan_b_df["Lower"].iloc[-1])
        b_final_high = float(plan_b_df["Upper"].iloc[-1])

        rm = compute_reflection_metrics(
            weeks=int(weeks),
            base_final=float(base_final),
            a_mean=float(a_final_mean),
            a_low=float(a_final_low),
            a_high=float(a_final_high),
            b_mean=float(b_final_mean),
            b_low=float(b_final_low),
            b_high=float(b_final_high),
            target_a_weekly=float(target_a),
            target_b_weekly=float(target_b),
            cut_a_weekly=float(cut_a),
            cut_b_weekly=float(cut_b),
            margin_a_weekly=float(max(margin_w, 0.0)),
            margin_b_weekly=float(max(margin_w, 0.0)),
            delta_a_weekly=float(cut_a),
            delta_b_weekly=float(cut_b),
        )

        st.markdown(build_human_reflection_text(rm))

        st.markdown("---")
        st.markdown("#### Technical explanation")
        st.markdown(
            build_explanation(
                base_final=float(base_final),
                a_final_mean=float(a_final_mean),
                a_final_low=float(a_final_low),
                a_final_high=float(a_final_high),
                b_final_mean=float(b_final_mean),
                b_final_low=float(b_final_low),
                b_final_high=float(b_final_high),
                inputs=inputs,
            )
        )
    else:
        # Single-plan explanation aligned to Baseline / Conservative / Expected / Optimistic
        single_plan_inputs = SinglePlanExplanationInputs(
            weeks=int(weeks),
            variability_pct=float(variability_pct),
            seed=int(seed),
            iters=int(iterations),
        )
        st.markdown("#### Technical explanation (Baseline vs your plan)")
        st.markdown(
            build_single_plan_explanation(
                baseline_final=float(base_final),
                conservative_final=float(a_final_low),
                expected_final=float(a_final_mean),
                optimistic_final=float(a_final_high),
                inputs=single_plan_inputs,
            )
        )


# ============================================================
# Progressive disclosure gate: open advanced investment module
# ============================================================
st.markdown("---")
st.markdown("### Continue to the investment module")
st.caption("Optional: use this to configure the investment universe, compare universes, and run deeper allocation analysis.")

investment_gate_attempted_key = "investment_setup_gate_attempted"
investment_gate_toast_key = "investment_setup_gate_toast_pending"
investment_gate_scroll_key = "investment_setup_gate_scroll_pending"
if investment_gate_attempted_key not in st.session_state:
    st.session_state[investment_gate_attempted_key] = False
if investment_gate_toast_key not in st.session_state:
    st.session_state[investment_gate_toast_key] = False
if investment_gate_scroll_key not in st.session_state:
    st.session_state[investment_gate_scroll_key] = False

if bool(st.session_state.get(investment_gate_toast_key, False)):
    st.toast("You need to fill in this field before continuing.", icon="⚠️")
    st.session_state[investment_gate_toast_key] = False

if bool(st.session_state.get(investment_gate_attempted_key, False)) and not plan_a_target_ready_for_investment:
    st.markdown(
        """
        <style>
        div[data-testid="stNumberInput"]:has(label div p[data-testid="stWidgetLabel"]),
        div[data-testid="stNumberInput"]:has(label) {
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

if not bool(st.session_state.get("investment_setup_open", False)):
    continue_to_investment_clicked = st.button(
        "Continue to investment setup",
        key="open_investment_setup",
        use_container_width=True,
    )
    if continue_to_investment_clicked:
        st.session_state[investment_gate_attempted_key] = True
        if plan_a_target_ready_for_investment:
            st.session_state["investment_setup_open"] = True
            st.session_state[investment_gate_attempted_key] = False
            st.session_state[investment_gate_toast_key] = False
            st.session_state[investment_gate_scroll_key] = False
            st.rerun()
        else:
            st.session_state[investment_gate_toast_key] = False
            st.session_state[investment_gate_scroll_key] = True
            st.toast("You need to fill in this field before continuing.", icon="⚠️")

    if bool(st.session_state.get(investment_gate_attempted_key, False)) and not plan_a_target_ready_for_investment:
        st.caption("Please fill in the Plan A savings field to continue.")
        if bool(st.session_state.get(investment_gate_scroll_key, False)):
            components.html(
                """
                <script>
                const findPlanAInput = () => {
                  const labels = Array.from(window.parent.document.querySelectorAll('label'));
                  for (const label of labels) {
                    const text = (label.innerText || '').trim();
                    if (text.includes('I want to save (£ per week)')) {
                      const wrapper = label.closest('div[data-testid="stNumberInput"]');
                      if (wrapper) {
                        wrapper.style.border = '2px solid #dc2626';
                        wrapper.style.borderRadius = '12px';
                        wrapper.style.padding = '0.35rem';
                        wrapper.style.boxShadow = '0 0 0 1px rgba(220,38,38,0.12)';
                      }
                      const input = label.parentElement?.querySelector('input');
                      if (input) {
                        input.focus();
                        input.setAttribute('aria-invalid', 'true');
                      }
                      const anchor = window.parent.document.getElementById('plan-a-target-anchor');
                      if (anchor) {
                        anchor.scrollIntoView({ behavior: 'smooth', block: 'center' });
                      } else if (wrapper) {
                        wrapper.scrollIntoView({ behavior: 'smooth', block: 'center' });
                      }
                      return;
                    }
                  }
                };
                findPlanAInput();
                </script>
                """,
                height=0,
                width=0,
            )
            st.session_state[investment_gate_scroll_key] = False
    st.stop()

# ============================================================
# STEP 4 — UNIVERSE SETUP (foundation for investment module)
# ============================================================
st.divider()
st.header("Step 4 — Investment universe setup (foundation)")
st.caption("This step prepares the universe-selection layer for the investment module: preset universes, large-universe research baskets, custom asset lists, and a later data-aware recommendation step that can suggest a stronger asset mix within the same strategy.")

plan_snapshot = _coerce_mapping(st.session_state.get("planning_snapshot", {}))
investment_context = _store_investment_context(plan_snapshot)
step4_monthly_contribution = float(investment_context.get("monthly_contribution", 0.0) or 0.0)
step4_weekly_equivalent = float(investment_context.get("weekly_equivalent", 0.0) or 0.0)
step4_baseline_monthly = float(investment_context.get("baseline_monthly", 0.0) or 0.0)
step4_required_cut_monthly = float(investment_context.get("required_cut_monthly", 0.0) or 0.0)

st.markdown("### Monthly contribution bridge")
b1, b2, b3 = st.columns(3)
with b1:
    st.metric("Plan A monthly contribution", f"£{step4_monthly_contribution:,.0f}/mo")
with b2:
    st.metric("Weekly equivalent", f"£{step4_weekly_equivalent:,.0f}/w")
with b3:
    st.metric("Baseline monthly savings", f"£{step4_baseline_monthly:,.0f}/mo")

if bool(plan_snapshot.get("structural_deficit", False)):
    st.warning("Step 4 uses a monthly investing view, but your underlying budgeting logic remains weekly. Fix the structural deficit first before treating this as a realistic long-term contribution.")
else:
    st.caption(
        f"For the investment module, Step 4 now uses the monthly contribution view as the main unit: **£{step4_monthly_contribution:,.0f}/month**. "
        f"Equivalent to **£{step4_weekly_equivalent:,.0f}/week** in the earlier budgeting steps. "
        f"Extra cut needed vs baseline: **£{step4_required_cut_monthly:,.0f}/month**."
    )



def _normalize_universe_mode(mode: Any) -> str:
    raw = str(mode or "").strip()
    if raw in {"Preset universe", "Standard preset", "Large-universe preset"}:
        return UNIVERSE_MODE_STANDARD
    if raw in UNIVERSE_MODE_OPTIONS:
        return raw
    return UNIVERSE_MODE_STANDARD


def _resolve_universe_selection(
    universe_size: Any,
    strategy_name: Any,
    custom_enabled: Any = False,
    custom_text: str = "",
) -> List[str]:
    if bool(custom_enabled):
        parsed_custom = _parse_custom_tickers(custom_text)
        if parsed_custom:
            return parsed_custom
    return _build_generated_universe(universe_size, strategy_name)


def _build_universe_preview_text(assets: List[str], *, max_items: int = 20) -> str:
    items = [str(x) for x in list(assets or []) if str(x).strip()]
    if not items:
        return ""
    suffix = " ..." if len(items) > max_items else ""
    return ", ".join(items[:max_items]) + suffix


def _handle_simple_preset_change() -> None:
    strategy_template = str(st.session_state.get("universe_simple_strategy_template", "Balanced Risk-Controlled"))
    style_preset = str(st.session_state.get("universe_simple_style_preset", "Balanced"))
    current_signature = f"{strategy_template}||{style_preset}"

    recommendations = _build_simple_slider_recommendations(strategy_template, style_preset)
    for metric_name, session_key in _SIMPLE_SLIDER_KEYS.items():
        st.session_state[session_key] = float(recommendations.get(metric_name, 0.5))

    st.session_state["universe_simple_slider_profile_signature"] = current_signature
    st.session_state["universe_simple_slider_profile_initialized"] = True
    st.session_state["universe_simple_slider_profile_last_source"] = "preset_change"


def _render_universe_group_mix(title: str, assets: List[str], *, prefix: str) -> None:
    tickers = [str(x).strip().upper() for x in list(assets or []) if str(x).strip()]
    if not tickers:
        return
    try:
        summary = _coerce_mapping(build_asset_group_summary(tickers))
        group_counts = _coerce_mapping(summary.get("group_counts", {}))
    except Exception as exc:
        st.caption(f"Could not build asset-group mix for {title.lower()}: {exc}")
        return

    if not group_counts:
        return

    rows = [
        {"group": str(group), "count": int(count or 0)}
        for group, count in group_counts.items()
        if int(count or 0) > 0
    ]
    if not rows:
        return

    mix_df = pd.DataFrame(rows).sort_values(["count", "group"], ascending=[False, True]).reset_index(drop=True)
    classified_share = float(summary.get("classified_share", 0.0) or 0.0)

    st.markdown(f"#### {title}")
    st.dataframe(mix_df, use_container_width=True, hide_index=True)
    st.caption(
        f"{int(summary.get('n_assets', len(tickers)) or len(tickers))} assets across "
        f"{int(summary.get('n_groups', len(mix_df)) or len(mix_df))} groups. "
        f"Classified share: {classified_share:.0%}."
    )

    try:
        detail_df = build_asset_group_dataframe(tickers)
    except Exception:
        detail_df = pd.DataFrame()
    if isinstance(detail_df, pd.DataFrame) and not detail_df.empty:
        with st.expander(f"{title} detail", expanded=False):
            st.dataframe(detail_df, use_container_width=True, hide_index=True)


def _render_universe_selector(
    *,
    section_title: str,
    size_key: str,
    strategy_key: str,
    custom_enabled_key: str,
    custom_text_key: str,
    default_size: int,
    default_strategy: str,
    custom_label: str,
    preview_label: str,
) -> List[str]:
    try:
        current_size = int(st.session_state.get(size_key, default_size))
    except Exception:
        current_size = int(default_size)
    if current_size not in UNIVERSE_SIZE_OPTIONS:
        current_size = int(default_size)
    st.session_state[size_key] = current_size

    allowed_strategies = _allowed_universe_strategies_for_size(current_size)
    current_strategy = str(st.session_state.get(strategy_key, default_strategy) or default_strategy)
    if current_strategy not in allowed_strategies:
        current_strategy = allowed_strategies[0]
    st.session_state[strategy_key] = current_strategy

    st.markdown(f"### {section_title}")
    c1, c2 = st.columns(2)
    with c1:
        st.selectbox(
            f"{section_title} size",
            options=UNIVERSE_SIZE_OPTIONS,
            key=size_key,
            help="Choose how many assets the generated universe should contain.",
        )
    selected_size = int(st.session_state.get(size_key, default_size) or default_size)
    allowed_strategies = _allowed_universe_strategies_for_size(selected_size)
    current_strategy = str(st.session_state.get(strategy_key, default_strategy) or default_strategy)
    if current_strategy not in allowed_strategies:
        st.session_state[strategy_key] = allowed_strategies[0]
    with c2:
        st.selectbox(
            f"{section_title} strategy",
            options=allowed_strategies,
            key=strategy_key,
            help="Choose how the selected universe size should be constructed.",
        )

    generated_assets = _build_generated_universe(
        st.session_state.get(size_key, default_size),
        st.session_state.get(strategy_key, default_strategy),
    )
    candidate_pool = _build_strategy_candidate_pool(
        st.session_state.get(size_key, default_size),
        st.session_state.get(strategy_key, default_strategy),
    )
    custom_supported = _custom_universe_supported_for_size(selected_size)
    multiselect_key = f"{custom_text_key}_multiselect"

    override_source_key = f"{custom_enabled_key}_source"
    override_source = str(st.session_state.get(override_source_key, "") or "")
    custom_text_value = str(st.session_state.get(custom_text_key, "") or "")
    parsed_custom_assets = _parse_custom_tickers(custom_text_value)
    custom_enabled = bool(st.session_state.get(custom_enabled_key, False))
    custom_override_active = custom_enabled and bool(parsed_custom_assets)

    if custom_supported:
        if parsed_custom_assets and multiselect_key not in st.session_state:
            st.session_state[multiselect_key] = list(parsed_custom_assets)
        elif multiselect_key not in st.session_state:
            st.session_state[multiselect_key] = list(generated_assets)

        st.checkbox(
            "Custom asset list",
            key=custom_enabled_key,
            help="Edit the generated universe using a constrained candidate pool for the selected strategy.",
        )
        custom_enabled = bool(st.session_state.get(custom_enabled_key, False))
        if custom_enabled:
            current_multiselect = [
                str(x).strip().upper()
                for x in list(st.session_state.get(multiselect_key, generated_assets) or [])
                if str(x).strip()
            ]
            valid_multiselect = [x for x in current_multiselect if x in candidate_pool]
            if not valid_multiselect:
                valid_multiselect = [x for x in generated_assets if x in candidate_pool]
            if valid_multiselect != current_multiselect:
                st.session_state[multiselect_key] = list(valid_multiselect)
            selected_custom_assets = st.multiselect(
                custom_label,
                options=candidate_pool,
                default=list(st.session_state.get(multiselect_key, valid_multiselect or generated_assets) or generated_assets),
                key=multiselect_key,
                help="Add or remove assets from the generated basket. The final custom size can differ from the preset size in this MVP.",
            )
            normalized_custom_assets = [str(x).strip().upper() for x in list(selected_custom_assets or []) if str(x).strip()]
            st.session_state[custom_text_key] = ", ".join(normalized_custom_assets)
            parsed_custom_assets = list(normalized_custom_assets)
            custom_override_active = bool(parsed_custom_assets)
            if normalized_custom_assets:
                if override_source not in {"recommendation", "broader_recommendation"}:
                    st.session_state[override_source_key] = "manual_custom"
            else:
                st.session_state[override_source_key] = ""
            preset_size = int(selected_size)
            custom_size = len(parsed_custom_assets)
            st.caption(f"Preset size: {preset_size} · Custom selected size: {custom_size}")
            st.caption("Turning this off returns to the preset universe, but keeps your last custom selection in case you turn it back on.")
        else:
            custom_override_active = False
            if parsed_custom_assets:
                st.caption("Custom editing is off, so the preset universe is active. Your last custom selection is being kept in memory.")
            if override_source not in {"recommendation", "broader_recommendation"}:
                st.session_state[override_source_key] = ""
    else:
        st.checkbox(
            "Custom asset list",
            value=False,
            key=f"{custom_enabled_key}_disabled_notice",
            disabled=True,
            help="Custom asset editing is currently available for 12- and 25-asset primary universes only.",
        )
        st.caption("Custom asset editing is currently available for 12- and 25-asset primary universes only. Switch the primary universe size to 12 or 25 to enable manual asset selection.")
        st.info("Advanced path: for larger universes, keep using the generated universe builder for diversification and data coverage. A future extension could support a core+satellite workflow, where you choose a small set of preferred assets and the system fills the rest automatically.")

    if custom_override_active:
        if override_source == "recommendation":
            badge_text = "Applied recommendation active: this universe is currently using an override basket instead of the generated preset."
        elif override_source == "broader_recommendation":
            badge_text = "Applied broader recommendation active: this universe is currently using an override basket instead of the generated preset."
        else:
            badge_text = "Custom universe override active: this universe is currently using an override basket instead of the generated preset."
        badge(badge_text, "ok")
        reset_cols = st.columns([1.15, 2.85])
        with reset_cols[0]:
            clear_override_clicked = st.button(
                "Return to preset universe",
                key=f"{custom_enabled_key}_clear_override",
                help="Clear the currently applied override basket and go back to the generated size + strategy preset.",
            )
        with reset_cols[1]:
            if override_source == "recommendation":
                st.caption("This basket came from the Step 5 recommendation block. Clear it to go back to the generated preset universe.")
            elif override_source == "broader_recommendation":
                st.caption("This basket came from the broader Step 5 recommendation block. Clear it to go back to the generated preset universe.")
            else:
                st.caption("This universe is currently running from a custom override basket. Clear it to go back to the generated preset universe.")
        if clear_override_clicked:
            _queue_session_updates({
                custom_enabled_key: False,
                custom_text_key: "",
                override_source_key: "",
                multiselect_key: list(generated_assets),
            })
            _queue_toast(f"{section_title} reverted to the generated preset universe.", icon="✅")
            st.rerun()
    else:
        st.caption("This new selector replaces the old standard vs large-universe split.")

    selected_assets = _resolve_universe_selection(
        st.session_state.get(size_key, default_size),
        st.session_state.get(strategy_key, default_strategy),
        custom_override_active,
        str(st.session_state.get(custom_text_key, "")),
    )

    requested_size = int(st.session_state.get(size_key, default_size) or default_size)
    actual_size = len(selected_assets)
    strategy_name = str(st.session_state.get(strategy_key, default_strategy) or default_strategy)
    if selected_assets:
        if actual_size == requested_size:
            st.caption(f"{preview_label} ({actual_size} assets | {strategy_name}): {_build_universe_preview_text(selected_assets)}")
        else:
            st.caption(f"{preview_label} ({actual_size}/{requested_size} assets | {strategy_name}): {_build_universe_preview_text(selected_assets)}")
    else:
        st.warning(f"No valid assets found yet for the {section_title.lower()}.")

    return selected_assets


_apply_pending_session_updates()
_flush_pending_toasts()


selected_assets = _render_universe_selector(
    section_title="Primary universe",
    size_key="universe_size",
    strategy_key="universe_strategy",
    custom_enabled_key="universe_custom_enabled",
    custom_text_key="custom_universe_text",
    default_size=12,
    default_strategy=UNIVERSE_STRATEGY_CORE,
    custom_label="Primary custom asset list",
    preview_label="Primary universe",
)
st.session_state["last_used_universe_assets"] = selected_assets

_render_universe_group_mix("Universe mix", selected_assets, prefix="primary_universe_mix")

# Optional comparison universe removed from the UI
st.session_state["comparison_enabled"] = False
comparison_enabled = False
comparison_assets: List[str] = []
st.session_state["last_comparison_universe_assets"] = []

recommendation_candidate_assets = _build_strategy_candidate_pool(
    st.session_state.get("universe_size", len(selected_assets) or 0),
    st.session_state.get("universe_strategy", UNIVERSE_STRATEGY_CORE),
)
st.session_state["last_recommendation_candidate_assets"] = list(recommendation_candidate_assets)

st.markdown("### Asset panel source")
data_source_mode = st.radio(
    "Choose how to provide market data",
    ["Yahoo Finance (recommended)", "Upload CSV/Parquet"],
    key="universe_data_source_mode",
    horizontal=True,
)

panel_df = None
panel_source_label = None
yahoo_daily_panel_df = None
yahoo_macro_panel_df = None

if data_source_mode == "Yahoo Finance (recommended)":
    universe_union = sorted(set(selected_assets) | set(comparison_assets) | set(recommendation_candidate_assets))
    st.caption("Yahoo Finance will be used as the primary data source. CSV/Parquet upload remains available as an optional fallback.")
    y1, y2 = st.columns(2)
    with y1:
        yahoo_start_date = st.date_input("Yahoo start date", value=pd.Timestamp("2010-01-01").date(), key="yahoo_start_date")
        yahoo_frequency = st.selectbox("Return frequency", ["monthly", "weekly", "daily"], index=0, key="yahoo_return_frequency")
    with y2:
        yahoo_end_date = st.date_input("Yahoo end date", value=pd.Timestamp.today().date(), key="yahoo_end_date")
        yahoo_auto_adjust = st.checkbox("Use Yahoo auto-adjusted prices", value=False, key="yahoo_auto_adjust")

    if universe_union:
        st.caption(f"Tickers to download ({len(universe_union)}): {', '.join(universe_union)}")
    else:
        st.warning("No valid tickers have been selected yet for Yahoo download.")

    if universe_union:
        union_kind, union_text = _large_universe_test_summary(len(universe_union))
        badge(f"Combined Yahoo download basket: {union_text}", union_kind)
        if len(universe_union) >= 50:
            st.caption("Large-universe mode is active. Expect heavier download / feature-building time, especially with daily enrichment.")

    if universe_union:
        yahoo_interval = "1d"
        if yahoo_frequency == "weekly":
            yahoo_interval = "1d"
        elif yahoo_frequency == "monthly":
            yahoo_interval = "1d"

        try:
            with st.spinner("Downloading market data from Yahoo Finance..."):
                panel_df = _download_yahoo_asset_panel_cached(
                    tuple(universe_union),
                    str(yahoo_start_date),
                    str(yahoo_end_date),
                    str(yahoo_frequency),
                    str(yahoo_interval),
                    bool(yahoo_auto_adjust),
                )
                yahoo_daily_panel_df = _download_yahoo_daily_asset_panel_cached(
                    tuple(universe_union),
                    str(yahoo_start_date),
                    str(yahoo_end_date),
                    bool(yahoo_auto_adjust),
                )
            panel_df = _validate_asset_panel(panel_df)
            yahoo_daily_panel_df = _validate_asset_panel(yahoo_daily_panel_df)
            panel_source_label = f"Yahoo Finance ({yahoo_frequency})"
            st.success(
                f"Yahoo panel loaded: {len(panel_df):,} rows across {panel_df['asset'].nunique()} assets. "
                f"Date range: {panel_df['date'].min().date()} → {panel_df['date'].max().date()}"
            )
            with st.expander("Preview Yahoo asset panel", expanded=False):
                st.dataframe(panel_df.head(50), use_container_width=True)
            _download_dataframe_button(
                "Download Yahoo asset panel as CSV",
                panel_df,
                "yahoo_asset_panel.csv",
                key="download_yahoo_asset_panel_csv",
            )
        except Exception as e:
            st.error(f"Could not download data from Yahoo Finance: {e}")
            panel_df = None
else:
    uploaded_asset_panel = st.file_uploader(
        "Optional: upload an asset panel (CSV or Parquet with date, asset, return)",
        type=["csv", "parquet"],
        key="universe_asset_panel_upload",
    )
    if uploaded_asset_panel is not None:
        try:
            panel_df = _read_uploaded_asset_panel(uploaded_asset_panel.getvalue(), uploaded_asset_panel.name)
            panel_df = _validate_asset_panel(panel_df)
            panel_source_label = f"Uploaded file ({uploaded_asset_panel.name})"
            st.success(
                f"Uploaded panel loaded: {len(panel_df):,} rows across {panel_df['asset'].nunique()} assets. "
                f"Date range: {panel_df['date'].min().date()} → {panel_df['date'].max().date()}"
            )
            with st.expander("Preview uploaded asset panel", expanded=False):
                st.dataframe(panel_df.head(50), use_container_width=True)
        except Exception as e:
            st.error(f"Could not load the uploaded asset panel: {e}")
            panel_df = None

asset_panel_ready = isinstance(panel_df, pd.DataFrame) and not panel_df.empty
if not asset_panel_ready:
    st.session_state["universe_engine_setup_open"] = False

if asset_panel_ready:
    st.markdown("---")
    st.caption("Portfolio and market data ready.")
    if not bool(st.session_state.get("universe_engine_setup_open", False)):
        continue_to_engine_clicked = st.button(
            "Continue to engine setup",
            key="open_engine_workspace",
            use_container_width=True,
        )
        if continue_to_engine_clicked:
            st.session_state["universe_engine_setup_open"] = True
            st.rerun()

if asset_panel_ready and not bool(st.session_state.get("universe_engine_setup_open", False)):
    st.info("Finish this step by continuing to the engine workspace.")
elif not asset_panel_ready:
    st.info("Load market data to continue to the engine workspace.")

if not (asset_panel_ready and bool(st.session_state.get("universe_engine_setup_open", False))):
    st.stop()

st.markdown("### Step 5 — Engine workspace")
_render_educational_finance_disclaimer("Step 5 engine workspace")
if str(st.session_state.get("universe_workspace_mode", "Simple")) == "Advanced":
    st.session_state["universe_workspace_mode"] = "Simple"
workspace_mode = st.radio(
    "Mode",
    ["Simple", "Global Search"],
    key="universe_workspace_mode",
    horizontal=True,
    help="Simple = semantic controls with advanced manual controls still available below. Global Search = dedicated schema-guided global multiobjective exploration.",
)
st.session_state["universe_simple_mode_enabled"] = bool(workspace_mode == "Simple")
if workspace_mode == "Simple":
    st.caption("Simple mode keeps the research engine behind a smaller semantic control surface. Advanced manual controls remain available below, but they now follow a clearer Configure → Run → Improve → Project flow.")
else:
    st.caption("Global Search is a parallel exploration module: schema-guided, resumable and explicitly multiobjective.")

if st.session_state.get("universe_simple_mode_enabled", True):
    st.markdown("#### Configure this setup")
    st.caption("Global strategy comes first here: template + style define the intent; auto-opt stays in the setup layer before the engine run.")
    _ensure_simple_slider_state_initialized()

    s1, s2 = st.columns(2)
    with s1:
        st.selectbox(
            "Strategy template",
            SIMPLE_STRATEGY_TEMPLATES,
            key="universe_simple_strategy_template",
            on_change=_handle_simple_preset_change,
        )
    with s2:
        st.selectbox(
            "Style preset",
            SIMPLE_STYLE_PRESETS,
            key="universe_simple_style_preset",
            on_change=_handle_simple_preset_change,
        )


    st.caption("Strategy template defines the base behaviour. Style preset applies a global bias. Sliders then fine-tune from that starting point.")

    u_assets = len(selected_assets) if isinstance(selected_assets, list) else 0
    if u_assets <= 15:
        bucket_label = "Small universe"
    elif u_assets <= 50:
        bucket_label = "Medium universe"
    else:
        bucket_label = "Large universe"
    st.caption(f"Detected universe size: {bucket_label} ({u_assets} assets)")

    c1, c2 = st.columns(2)
    with c1:
        st.slider("Risk appetite", 0.0, 1.0, 0.5, key="universe_simple_risk_appetite")
        st.slider("Diversification ↔ concentration", 0.0, 1.0, 0.5, key="universe_simple_diversification")
        st.slider("Stability ↔ responsiveness", 0.0, 1.0, 0.5, key="universe_simple_stability")
        st.slider("Low turnover ↔ adaptive", 0.0, 1.0, 0.5, key="universe_simple_turnover_pref")
    with c2:
        st.slider("Drawdown protection", 0.0, 1.0, 0.5, key="universe_simple_drawdown_protection")
        st.slider("Overlay intensity", 0.0, 1.0, 0.5, key="universe_simple_overlay_intensity")
        st.slider("Confidence in signal", 0.0, 1.0, 0.5, key="universe_simple_signal_confidence")
        st.slider("Simplicity ↔ sophistication", 0.0, 1.0, 0.5, key="universe_simple_simplicity")

    st.checkbox(
        "Auto optimize",
        key="universe_simple_auto_optimize_enabled",
        help="Runs a short local refinement around the resolved Simple-mode config before the final pipeline run. Falls back to the baseline config if no usable improvement is found.",
    )

    _resolve_and_sync_simple_mode_state(u_assets)

if st.session_state.get("universe_simple_mode_enabled", True):
    _simple_local_search_policy = _maybe_apply_simple_mode_local_search_defaults()
    _simple_selection_policy = _maybe_apply_simple_mode_selection_policy_defaults()
    resolved = _coerce_mapping(st.session_state.get("universe_simple_resolved_summary", {}))
    if resolved:
        st.markdown("#### Resolved internal configuration")
        st.caption("This is the current internal implementation inferred from your high-level setup before the run.")
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Top k", str(resolved.get("resolved_top_k", "None")))
        with r2:
            st.metric("Overlay", str(resolved.get("resolved_probabilistic_mode", "none")))
        with r3:
            st.metric("Universe bucket", str(resolved.get("universe_bucket", "unknown")))
        st.caption(
            f"Vol targeting={resolved.get('resolved_vol_targeting')} · "
            f"turnover_penalty={resolved.get('resolved_turnover_penalty_strength')} · "
            f"target_vol={resolved.get('resolved_target_portfolio_vol_monthly')}"
        )
        tuning_policy = dict(resolved.get("tuning_policy", {}) or {})
        if tuning_policy:
            selection_policy = str(st.session_state.get("universe_simple_auto_selection_policy", "fixed_composite_score") or "fixed_composite_score")
            composite_profile = str(st.session_state.get("universe_simple_auto_composite_profile", DEFAULT_COMPOSITE_PROFILE) or DEFAULT_COMPOSITE_PROFILE)
            st.caption(
                f"Semantic tuning policy → objective={tuning_policy.get('objective', '—')} · "
                f"two-stage={'on' if bool(tuning_policy.get('two_stage_search', False)) else 'off'} · "
                f"search_budget={tuning_policy.get('search_budget', '—')} · "
                f"selection_policy={selection_policy} · composite_profile={composite_profile} · "
                f"preferred_dims={', '.join(list(tuning_policy.get('preferred_dims', [])[:6])) or '—'}"
            )

st.markdown("#### Run")
st.caption("Engine setup ready. Run the current configuration first; refinement and recommendation blocks come afterwards.")
run_portfolio_clicked = st.button(
    "Run portfolio & view results",
    key="open_engine_results",
    use_container_width=True,
)
if run_portfolio_clicked:
    st.session_state["engine_results_open"] = True
    st.session_state["engine_has_run"] = False
    st.session_state["projection_open"] = False
    st.rerun()

if bool(st.session_state.get("engine_results_open", False)):
    with st.expander("Advanced manual controls", expanded=False):
        if bool(st.session_state.get("universe_simple_mode_enabled", True)):
            st.caption("Simple mode is currently syncing the affected manual controls below from the semantic resolve.")

        st.markdown("---")
        st.caption("Auto-selection & validation")
        st.caption("Meta-controls live first here: use them to validate or auto-select the signal contract before touching lower-level engine modules.")
        st.markdown("---")
        st.caption("Automatic signal-contract selection by Rank IC")
        st.checkbox(
            "Enable automatic signal-contract selection by Rank IC",
            key="universe_auto_signal_select_enabled",
            help="Runs multiple requested signal contracts on the same panel, measures their ex-post cross-sectional Rank IC, and automatically uses the best candidate for the primary run.",
        )
        st.multiselect(
            "Requested signal contracts to explore",
            options=AUTO_SIGNAL_CONTRACT_LABELS,
            format_func=_format_signal_contract_label,
            key="universe_auto_signal_select_contracts",
            help="This Route-B exploration evaluates explicit requested signal contracts rather than starting from base mode names and expanding them afterwards.",
        )
        st.caption("Automatic selection now evaluates explicit requested signal contracts. The comparison table reports requested contracts and realised effective modes side by side.")
        st.checkbox(
            "Show full auto-selection report",
            key="universe_auto_signal_select_show_report",
            help="If enabled, renders the full Rank IC comparison table and detailed per-contract diagnostics.",
        )

        st.markdown("---")
        st.caption("Core engine definition")
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("min_train", min_value=24, max_value=240, step=12, key="universe_micro_min_train")
            st.number_input("lookback_mu", min_value=3, max_value=60, step=1, key="universe_micro_lookback_mu")
            st.number_input("lookback_sigma", min_value=3, max_value=60, step=1, key="universe_micro_lookback_sigma")
            st.number_input(
                "sigma_power_alpha",
                min_value=0.10,
                max_value=5.0,
                step=0.05,
                key="universe_micro_sigma_power_alpha",
                help="Controls how strongly sigma enters the score normalisation. Engine support may depend on signal_mode, so treat this as advanced / engine-dependent until the motor refactor is complete.",
            )
            st.number_input("temperature", min_value=0.1, max_value=5.0, step=0.1, key="universe_micro_temperature")
            st.number_input("weight_shrink", min_value=0.0, max_value=1.0, step=0.01, key="universe_micro_weight_shrink")
            st.number_input("inertia", min_value=0.0, max_value=1.0, step=0.01, key="universe_micro_inertia")
        with c2:
            st.checkbox("deadband", key="universe_micro_deadband")
            st.number_input("deadband_threshold", min_value=0.0, max_value=0.20, step=0.01, key="universe_micro_deadband_threshold")
            st.checkbox(
                "Use default top_k (all assets / no hard cutoff)",
                key="universe_micro_top_k_none",
                help="Base allocator concentration control. When enabled, no hard top_k cutoff is applied. When disabled, only the top-k scored assets survive before softmax allocation.",
            )
            if not st.session_state.get("universe_micro_top_k_none", True):
                st.number_input(
                    "top_k",
                    min_value=1,
                    max_value=50,
                    step=1,
                    key="universe_micro_top_k",
                    help="Hard cutoff for the base allocator. This is different from top_k_classifier_k and from dispersion_top_k_* controls.",
                )
            st.selectbox("regime_mode", ["none", "quantile"], key="universe_micro_regime_mode")
            st.selectbox("mu_regime_mode", ["pooled", "split"], key="universe_micro_mu_regime_mode")
            st.checkbox("ewma_sigma", key="universe_micro_ewma_sigma")
            st.number_input("ewma_halflife", min_value=1, max_value=24, step=1, key="universe_micro_ewma_halflife")

        if str(st.session_state.get("universe_micro_signal_mode", "mu_sigma")) not in {"mu_sigma", "huber_mu", "mu_sigma_fallback"}:
            st.info("sigma_power_alpha is currently engine-dependent and may be inactive for the selected signal_mode. The tuning audit below will drop it automatically when inactive.")

        st.markdown("---")
        st.caption("Risk shaping — per-asset caps")
        cap1, cap2 = st.columns(2)
        valid_cfg_fields = {f.name for f in fields(MicroPipelineConfig)}
        with cap1:
            st.checkbox("Enable asset_weight_cap", key="universe_micro_asset_weight_cap_enabled")
            if st.session_state.get("universe_micro_asset_weight_cap_enabled", False):
                st.number_input("asset_weight_cap", min_value=0.01, max_value=1.0, step=0.01, key="universe_micro_asset_weight_cap")
        with cap2:
            if "w_cap" in valid_cfg_fields:
                st.checkbox("Enable W_CAP", key="universe_micro_w_cap_enabled")
                if st.session_state.get("universe_micro_w_cap_enabled", False):
                    st.number_input("w_cap", min_value=0.01, max_value=1.0, step=0.01, key="universe_micro_w_cap")
            else:
                st.caption("This investment.py version exposes asset_weight_cap but not W_CAP.")

        # Requested base cap transparency
        asset_weight_cap_requested = None
        if bool(st.session_state.get("universe_micro_asset_weight_cap_enabled", False)):
            asset_weight_cap_requested = float(st.session_state.get("universe_micro_asset_weight_cap", 0.20))

        w_cap_requested = None
        if bool(st.session_state.get("universe_micro_w_cap_enabled", False)):
            w_cap_requested = float(st.session_state.get("universe_micro_w_cap", 0.20))

        requested_caps = [x for x in [asset_weight_cap_requested, w_cap_requested] if x is not None]
        effective_cap_base = min(requested_caps) if requested_caps else None

        def _fmt_cap(x):
            return "disabled" if x is None else f"{float(x):.4f}"

        st.markdown("**Requested base cap transparency**")
        st.caption(
            f"asset_weight_cap requested = {_fmt_cap(asset_weight_cap_requested)} | "
            f"w_cap requested = {_fmt_cap(w_cap_requested)} | "
            f"effective base cap = {_fmt_cap(effective_cap_base)}"
        )

        if effective_cap_base is not None:
            st.info(
                "The engine uses the most restrictive requested base cap "
                "(min(asset_weight_cap, w_cap)) before adaptive caps are applied."
            )
        else:
            st.info(
                "No explicit base cap is active. The allocator may still be constrained later "
                "by adaptive caps, turnover logic, or other governance rules."
            )

        st.caption(
            "Final effective cap can become tighter after vol-, correlation-, dispersion-, "
            "or regime-dependent cap adjustments."
        )

        st.markdown("---")
        st.caption("Adaptive caps (Phase 1 wiring)")
        ac1, ac2 = st.columns(2)
        with ac1:
            st.checkbox("Enable vol-dependent cap", key="universe_micro_vol_cap_enabled")
            st.number_input("vol cap threshold low", min_value=0.0, max_value=1.0, step=0.01, key="universe_micro_vol_cap_threshold_low")
            st.number_input("vol cap threshold high", min_value=0.0, max_value=1.0, step=0.01, key="universe_micro_vol_cap_threshold_high")
            st.number_input("vol cap low mult", min_value=0.0, max_value=5.0, step=0.05, key="universe_micro_vol_cap_low_mult")
            st.number_input("vol cap high mult", min_value=0.0, max_value=5.0, step=0.05, key="universe_micro_vol_cap_high_mult")
            st.checkbox("Enable vol cap min", key="universe_micro_vol_cap_min_enabled")
            if st.session_state.get("universe_micro_vol_cap_min_enabled", False):
                st.number_input("vol cap min", min_value=0.0, max_value=1.0, step=0.01, key="universe_micro_vol_cap_min")
            st.checkbox("Enable vol cap max", key="universe_micro_vol_cap_max_enabled")
            if st.session_state.get("universe_micro_vol_cap_max_enabled", False):
                st.number_input("vol cap max", min_value=0.0, max_value=1.0, step=0.01, key="universe_micro_vol_cap_max")
        with ac2:
            st.checkbox("Enable corr-dependent cap", key="universe_micro_corr_cap_enabled")
            st.number_input("corr cap threshold low", min_value=0.0, max_value=1.0, step=0.01, key="universe_micro_corr_cap_threshold_low")
            st.number_input("corr cap threshold high", min_value=0.0, max_value=1.0, step=0.01, key="universe_micro_corr_cap_threshold_high")
            st.number_input("corr cap low mult", min_value=0.0, max_value=5.0, step=0.05, key="universe_micro_corr_cap_low_mult")
            st.number_input("corr cap high mult", min_value=0.0, max_value=5.0, step=0.05, key="universe_micro_corr_cap_high_mult")
            st.checkbox("Enable corr cap min", key="universe_micro_corr_cap_min_enabled")
            if st.session_state.get("universe_micro_corr_cap_min_enabled", False):
                st.number_input("corr cap min", min_value=0.0, max_value=1.0, step=0.01, key="universe_micro_corr_cap_min")
            st.checkbox("Enable corr cap max", key="universe_micro_corr_cap_max_enabled")
            if st.session_state.get("universe_micro_corr_cap_max_enabled", False):
                st.number_input("corr cap max", min_value=0.0, max_value=1.0, step=0.01, key="universe_micro_corr_cap_max")

        ac3, ac4 = st.columns(2)
        with ac3:
            st.checkbox("Enable dispersion-dependent cap", key="universe_micro_disp_cap_enabled")
            st.number_input("disp cap threshold low", min_value=0.0, max_value=1.0, step=0.01, key="universe_micro_disp_cap_threshold_low")
            st.number_input("disp cap threshold high", min_value=0.0, max_value=1.0, step=0.01, key="universe_micro_disp_cap_threshold_high")
            st.number_input("disp cap low mult", min_value=0.0, max_value=5.0, step=0.05, key="universe_micro_disp_cap_low_mult")
            st.number_input("disp cap high mult", min_value=0.0, max_value=5.0, step=0.05, key="universe_micro_disp_cap_high_mult")
            st.checkbox("Enable disp cap min", key="universe_micro_disp_cap_min_enabled")
            if st.session_state.get("universe_micro_disp_cap_min_enabled", False):
                st.number_input("disp cap min", min_value=0.0, max_value=1.0, step=0.01, key="universe_micro_disp_cap_min")
            st.checkbox("Enable disp cap max", key="universe_micro_disp_cap_max_enabled")
            if st.session_state.get("universe_micro_disp_cap_max_enabled", False):
                st.number_input("disp cap max", min_value=0.0, max_value=1.0, step=0.01, key="universe_micro_disp_cap_max")
        with ac4:
            st.checkbox("Enable regime-dependent cap", key="universe_micro_regime_cap_enabled")
            st.number_input("regime cap low mult", min_value=0.0, max_value=5.0, step=0.05, key="universe_micro_regime_cap_low_mult")
            st.number_input("regime cap mid mult", min_value=0.0, max_value=5.0, step=0.05, key="universe_micro_regime_cap_mid_mult")
            st.number_input("regime cap high mult", min_value=0.0, max_value=5.0, step=0.05, key="universe_micro_regime_cap_high_mult")
            st.checkbox("Enable regime cap min", key="universe_micro_regime_cap_min_enabled")
            if st.session_state.get("universe_micro_regime_cap_min_enabled", False):
                st.number_input("regime cap min", min_value=0.0, max_value=1.0, step=0.01, key="universe_micro_regime_cap_min")
            st.checkbox("Enable regime cap max", key="universe_micro_regime_cap_max_enabled")
            if st.session_state.get("universe_micro_regime_cap_max_enabled", False):
                st.number_input("regime cap max", min_value=0.0, max_value=1.0, step=0.01, key="universe_micro_regime_cap_max")

        st.markdown("---")
        st.caption("Turnover governance (Phase 1 wiring)")
        to1, to2 = st.columns(2)
        with to1:
            st.number_input("turnover penalty strength", min_value=0.0, max_value=20.0, step=0.05, key="universe_micro_turnover_penalty_strength")
            st.number_input("turnover penalty power", min_value=0.1, max_value=5.0, step=0.1, key="universe_micro_turnover_penalty_power")
            st.number_input("turnover penalty target", min_value=0.0, max_value=1.0, step=0.01, key="universe_micro_turnover_penalty_target")
        with to2:
            st.checkbox("Enable turnover penalty max", key="universe_micro_turnover_penalty_max_turnover_enabled")
            if st.session_state.get("universe_micro_turnover_penalty_max_turnover_enabled", False):
                st.number_input("turnover penalty max", min_value=0.0, max_value=2.0, step=0.01, key="universe_micro_turnover_penalty_max_turnover")
            st.checkbox("Enable turnover hard constraint", key="universe_micro_turnover_constraint_enabled")
            if st.session_state.get("universe_micro_turnover_constraint_enabled", False):
                st.number_input("turnover hard constraint max", min_value=0.0, max_value=2.0, step=0.01, key="universe_micro_turnover_constraint_max_turnover")

        st.markdown("---")
        st.caption("Risk model / covariance (Phase 2 wiring)")
        rc1, rc2 = st.columns(2)
        with rc1:
            st.selectbox(
                "covariance_mode",
                ["ewma_cov", "corr_sigma"],
                key="universe_micro_covariance_mode",
                help="Selects the base covariance estimator used by the engine.",
            )
            st.checkbox(
                "Regime-dependent covariance",
                key="universe_micro_regime_dependent_covariance",
                help="Lets covariance lookback / halflife adapt by regime while preserving default behaviour when left unchanged.",
            )
            st.number_input("correlation_lookback", min_value=3, max_value=240, step=1, key="universe_micro_correlation_lookback")
            st.number_input("correlation_min_periods", min_value=1, max_value=120, step=1, key="universe_micro_correlation_min_periods")
            st.number_input("correlation_shrink_to_identity", min_value=0.0, max_value=1.0, step=0.05, key="universe_micro_correlation_shrink_to_identity")
            st.number_input("covariance_shrink_to_diagonal", min_value=0.0, max_value=1.0, step=0.05, key="universe_micro_covariance_shrink_to_diagonal")
            st.number_input("covariance_jitter", min_value=0.0, max_value=1e-2, step=1e-8, format="%.8f", key="universe_micro_covariance_jitter")
        with rc2:
            st.markdown("**Regime covariance windows**")
            st.number_input("covariance_lookback_low", min_value=3, max_value=240, step=1, key="universe_micro_covariance_lookback_low")
            st.number_input("covariance_lookback_mid", min_value=3, max_value=240, step=1, key="universe_micro_covariance_lookback_mid")
            st.number_input("covariance_lookback_high", min_value=3, max_value=240, step=1, key="universe_micro_covariance_lookback_high")
            st.number_input("covariance_halflife_low", min_value=1, max_value=120, step=1, key="universe_micro_covariance_halflife_low")
            st.number_input("covariance_halflife_mid", min_value=1, max_value=120, step=1, key="universe_micro_covariance_halflife_mid")
            st.number_input("covariance_halflife_high", min_value=1, max_value=120, step=1, key="universe_micro_covariance_halflife_high")

        st.markdown("---")
        st.caption("Correlation-aware allocator (Phase 2 wiring)")
        ca1, ca2 = st.columns(2)
        with ca1:
            st.checkbox(
                "Enable correlation-aware allocation",
                key="universe_micro_correlation_aware_allocation",
                help="Lets the allocator penalise concentration in highly correlated names using the engine's built-in allocator block.",
            )
            st.selectbox(
                "correlation_allocator_method",
                ["score_penalty", "mean_variance_light", "risk_budget"],
                key="universe_micro_correlation_allocator_method",
            )
            st.number_input("correlation_allocator_blend", min_value=0.0, max_value=1.0, step=0.05, key="universe_micro_correlation_allocator_blend")
            st.number_input("correlation_penalty_strength", min_value=0.0, max_value=20.0, step=0.05, key="universe_micro_correlation_penalty_strength")
            st.number_input("correlation_penalty_power", min_value=0.1, max_value=5.0, step=0.1, key="universe_micro_correlation_penalty_power")
        with ca2:
            st.checkbox("Use absolute correlation", key="universe_micro_correlation_use_abs")
            st.number_input("mean_variance_risk_aversion", min_value=0.0, max_value=100.0, step=0.25, key="universe_micro_mean_variance_risk_aversion")
            st.number_input("risk_budget_strength", min_value=0.0, max_value=10.0, step=0.05, key="universe_micro_risk_budget_strength")
            st.number_input("cluster_corr_threshold", min_value=0.0, max_value=1.0, step=0.01, key="universe_micro_cluster_corr_threshold")
            st.caption("These controls are ignored unless the selected allocator method uses them, so default behaviour stays unchanged until activated.")

        st.markdown("---")
        st.caption("Vol targeting / regime derisk (Phase 3 wiring)")
        vt1, vt2 = st.columns(2)
        with vt1:
            st.checkbox(
                "Enable vol targeting",
                key="universe_micro_vol_targeting",
                help="Scales final portfolio exposure toward a target monthly volatility.",
            )
            st.checkbox(
                "Covariance-aware vol targeting",
                key="universe_micro_covariance_aware_vol_targeting",
                help="Uses the covariance-aware portfolio risk estimate when applying vol targeting.",
            )
            st.number_input(
                "target_portfolio_vol_monthly",
                min_value=0.001,
                max_value=0.50,
                step=0.005,
                key="universe_micro_target_portfolio_vol_monthly",
                help="Target monthly portfolio volatility used by the engine when vol targeting is enabled.",
            )
            st.number_input(
                "vol_target_floor_mult",
                min_value=0.0,
                max_value=5.0,
                step=0.05,
                key="universe_micro_vol_target_floor_mult",
                help="Lower bound on the scaling multiplier applied by vol targeting.",
            )
            st.number_input(
                "vol_target_ceiling_mult",
                min_value=0.0,
                max_value=5.0,
                step=0.05,
                key="universe_micro_vol_target_ceiling_mult",
                help="Upper bound on the scaling multiplier applied by vol targeting.",
            )
        with vt2:
            st.markdown("**Regime derisk multipliers**")
            st.number_input(
                "regime_derisk_low",
                min_value=0.0,
                max_value=2.0,
                step=0.05,
                key="universe_micro_regime_derisk_low",
                help="Exposure multiplier applied in low-risk regimes.",
            )
            st.number_input(
                "regime_derisk_mid",
                min_value=0.0,
                max_value=2.0,
                step=0.05,
                key="universe_micro_regime_derisk_mid",
                help="Exposure multiplier applied in mid-risk regimes.",
            )
            st.number_input(
                "regime_derisk_high",
                min_value=0.0,
                max_value=2.0,
                step=0.05,
                key="universe_micro_regime_derisk_high",
                help="Exposure multiplier applied in high-risk regimes.",
            )

        st.markdown("---")
        st.caption("Frictions / implementation realism")
        cp1, cp2 = st.columns([2, 1])
        with cp1:
            st.selectbox(
                "Cost-model preset",
                list(COST_MODEL_PRESETS.keys()),
                key="universe_cost_model_preset",
                help="Quick realistic presets. They only change cost/tax inputs and remain backward-compatible.",
            )
            st.caption(COST_MODEL_PRESET_META.get(str(st.session_state.get("universe_cost_model_preset", "Off / legacy")), ""))
        with cp2:
            st.write("")
            st.write("")
            if st.button("Apply cost preset", key="apply_cost_model_preset_btn", use_container_width=True):
                apply_cost_model_preset(str(st.session_state.get("universe_cost_model_preset", "Off / legacy")))

        st.checkbox(
            "Enable full cost model",
            key="universe_cost_model_enabled",
            help="Turns on transaction, spread, market-impact and holding-cost drag inside the engine. Defaults remain OFF for backward compatibility.",
        )
        cst1, cst2 = st.columns(2)
        with cst1:
            st.number_input("commission (bps)", min_value=0.0, max_value=500.0, step=1.0, key="universe_transaction_cost_commission_bps")
            st.number_input("slippage (bps)", min_value=0.0, max_value=500.0, step=1.0, key="universe_transaction_cost_slippage_bps")
            st.number_input("spread (bps)", min_value=0.0, max_value=500.0, step=1.0, key="universe_transaction_cost_spread_bps")
            st.number_input("holding cost annual (bps)", min_value=0.0, max_value=500.0, step=1.0, key="universe_holding_cost_annual_bps")
        with cst2:
            st.number_input("market impact (bps)", min_value=0.0, max_value=500.0, step=1.0, key="universe_transaction_cost_market_impact_bps")
            st.number_input("market impact power", min_value=0.1, max_value=5.0, step=0.1, key="universe_transaction_cost_market_impact_power")
            st.number_input("min trade weight", min_value=0.0, max_value=1.0, step=0.001, format="%.4f", key="universe_transaction_cost_min_trade_weight")

        tax1, tax2 = st.columns(2)
        with tax1:
            st.checkbox(
                "Enable tax model",
                key="universe_tax_model_enabled",
                help="Applies realised-gain taxation on top of the cost model using the engine's average-cost approximation.",
            )
            st.number_input("tax short-term rate", min_value=0.0, max_value=1.0, step=0.01, key="universe_tax_short_term_rate")
            st.number_input("tax long-term rate", min_value=0.0, max_value=1.0, step=0.01, key="universe_tax_long_term_rate")
        with tax2:
            st.number_input("tax long-term threshold (months)", min_value=1, max_value=120, step=1, key="universe_tax_long_term_threshold_months")
            st.checkbox("Apply tax loss credit", key="universe_tax_apply_loss_credit")
            st.number_input("tax loss-credit rate", min_value=0.0, max_value=1.0, step=0.01, key="universe_tax_loss_credit_rate")

        cst_r1, cst_r2 = st.columns(2)
        with cst_r1:
            st.checkbox(
                "Run no-cost baseline comparison",
                key="universe_cost_model_compare_enabled",
                help="Runs the same engine with costs/taxes disabled so you can quantify implementation drag cleanly.",
            )
            st.checkbox("Show compact cost summary cards", key="universe_cost_report_show_summary")
            st.checkbox("Show cost drag timeseries", key="universe_cost_report_show_timeseries")
            st.checkbox("Show gross vs net NAV chart", key="universe_cost_report_show_nav_chart")
        with cst_r2:
            st.checkbox("Show detailed cost-model report", key="universe_cost_report_show_detail")
            st.checkbox("Show cost diagnostics preview", key="universe_cost_report_show_diagnostics")

        st.markdown("---")
        st.caption("Factor model industrial")
        fm1, fm2 = st.columns(2)
        with fm1:
            st.checkbox(
                "Enable factor-model industrial overlay",
                key="universe_factor_model_enabled",
                help="Activates a statistical industrial factor overlay that can tilt mu, residual alpha and covariance while keeping legacy behaviour intact when OFF.",
            )
            st.number_input("factor model n_factors", min_value=1, max_value=20, step=1, key="universe_factor_model_n_factors")
            st.number_input("factor model min_obs", min_value=6, max_value=240, step=1, key="universe_factor_model_min_obs")
            st.number_input("factor mu blend", min_value=0.0, max_value=1.0, step=0.05, key="universe_factor_model_mu_blend")
            st.number_input("factor residual blend", min_value=0.0, max_value=1.0, step=0.05, key="universe_factor_model_residual_blend")
            st.number_input("factor model covariance blend", min_value=0.0, max_value=1.0, step=0.05, key="universe_factor_model_covariance_blend")
            st.number_input("factor model shrink-to-diagonal", min_value=0.0, max_value=1.0, step=0.05, key="universe_factor_model_shrink_to_diagonal")
        with fm2:
            st.checkbox(
                "Enable factor-covariance overlay",
                key="universe_factor_covariance_enabled",
                help="Uses a factor-structured covariance estimate even if the broader factor-model mu overlay is disabled.",
            )
            st.number_input("factor covariance n_factors", min_value=1, max_value=20, step=1, key="universe_factor_covariance_n_factors")
            st.number_input("factor covariance min_obs", min_value=6, max_value=240, step=1, key="universe_factor_covariance_min_obs")
            st.number_input("factor covariance blend", min_value=0.0, max_value=1.0, step=0.05, key="universe_factor_covariance_blend")
            st.number_input("factor covariance shrink-to-diagonal", min_value=0.0, max_value=1.0, step=0.05, key="universe_factor_covariance_shrink_to_diagonal")
            st.checkbox(
                "Run no-factor baseline comparison",
                key="universe_factor_compare_enabled",
                help="Runs the same engine with the factor overlay disabled so you can measure incremental value cleanly.",
            )

        fr1, fr2 = st.columns(2)
        with fr1:
            st.checkbox("Show compact factor summary cards", key="universe_factor_report_show_summary")
            st.checkbox("Show factor timeseries", key="universe_factor_report_show_timeseries")
            st.checkbox("Show factor NAV chart", key="universe_factor_report_show_nav_chart")
        with fr2:
            st.checkbox("Show detailed factor report", key="universe_factor_report_show_detail")
            st.checkbox("Show factor diagnostics preview", key="universe_factor_report_show_diagnostics")

        st.markdown("---")
        st.caption("Overlay / probabilistic")
        current_prob_mode = str(st.session_state.get("universe_micro_prob_mode", "none") or "none")
        prob_mode_options = list(PROBABILISTIC_MODE_CORE)
        if current_prob_mode not in prob_mode_options and current_prob_mode:
            prob_mode_options.append(current_prob_mode)
        p1, p2 = st.columns(2)
        with p1:
            selected_prob_mode = st.selectbox(
                "probabilistic_mode (core)",
                prob_mode_options,
                index=max(0, prob_mode_options.index(current_prob_mode if current_prob_mode in prob_mode_options else "none")),
                format_func=_format_probabilistic_mode_label,
                help="Core probabilistic overlay choices. Experimental / alias variants have been moved below so the main UI does not overstate backend distinctness.",
            )
            st.session_state["universe_micro_prob_mode"] = str(selected_prob_mode)
            st.number_input(
                "probabilistic_q_low",
                min_value=0.01,
                max_value=0.49,
                step=0.01,
                key="universe_micro_prob_q_low",
            )
            st.number_input(
                "probabilistic_min_obs",
                min_value=3,
                max_value=120,
                step=1,
                key="universe_micro_prob_min_obs",
            )
            st.number_input(
                "prob_interval_penalty_weight",
                min_value=0.0,
                max_value=5.0,
                step=0.05,
                key="universe_micro_prob_interval_penalty_weight",
            )
            st.number_input(
                "prob_confidence_scale",
                min_value=0.0,
                max_value=5.0,
                step=0.05,
                key="universe_micro_prob_confidence_scale",
            )
        with p2:
            st.number_input(
                "probabilistic_q_high",
                min_value=0.51,
                max_value=0.99,
                step=0.01,
                key="universe_micro_prob_q_high",
            )
            st.number_input(
                "prob_downside_penalty_weight",
                min_value=0.0,
                max_value=5.0,
                step=0.05,
                key="universe_micro_prob_downside_penalty_weight",
            )
            st.number_input(
                "prob_confidence_min_mult",
                min_value=0.0,
                max_value=2.0,
                step=0.05,
                key="universe_micro_prob_confidence_min_mult",
            )
            st.number_input(
                "prob_confidence_max_mult",
                min_value=0.0,
                max_value=3.0,
                step=0.05,
                key="universe_micro_prob_confidence_max_mult",
            )

        with st.expander("Experimental probabilistic variants", expanded=_is_experimental_probabilistic_mode(st.session_state.get("universe_micro_prob_mode", "none"))):
            st.caption("These variants are exposed for research use. Some are now distinct backends but remain experimental or not fully validated.")
            exp_current_mode = str(st.session_state.get("universe_micro_prob_mode", "none") or "none")
            exp_default = exp_current_mode if exp_current_mode in PROBABILISTIC_MODE_EXPERIMENTAL else PROBABILISTIC_MODE_EXPERIMENTAL[0]
            experimental_choice = st.selectbox(
                "Experimental probabilistic mode",
                PROBABILISTIC_MODE_EXPERIMENTAL,
                index=PROBABILISTIC_MODE_EXPERIMENTAL.index(exp_default),
                format_func=_format_probabilistic_mode_label,
                key="universe_micro_prob_mode_experimental_choice",
            )
            st.caption(PROBABILISTIC_MODE_NOTES.get(str(experimental_choice), ""))
            if st.button("Use selected experimental probabilistic variant", key="universe_micro_apply_prob_experimental"):
                st.session_state["universe_micro_prob_mode"] = str(experimental_choice)
                st.rerun()

        prob_mode_ui = str(st.session_state.get("universe_micro_prob_mode", "none"))
        if _is_experimental_probabilistic_mode(prob_mode_ui):
            st.warning(
                f"Current probabilistic mode: {_format_probabilistic_mode_label(prob_mode_ui)}. "
                + PROBABILISTIC_MODE_NOTES.get(prob_mode_ui, "Treat this mode as experimental / research-oriented and validate it carefully."),
                icon="⚠️",
            )

        if prob_mode_ui == "quantile_regression":
            st.markdown("**Quantile regression controls**")
            q1, q2, q3 = st.columns(3)
            with q1:
                st.number_input(
                    "probabilistic_qr_alpha",
                    min_value=0.0,
                    max_value=10.0,
                    step=0.01,
                    key="universe_micro_prob_qr_alpha",
                    help="Regularisation strength for QuantileRegressor. Lower alpha fits more freely; higher alpha shrinks coefficients and usually gives more stable but less reactive quantiles.",
                )
            with q2:
                st.selectbox(
                    "probabilistic_qr_solver",
                    ["highs", "interior-point"],
                    key="universe_micro_prob_qr_solver",
                    help="Linear programming solver used by QuantileRegressor.",
                )
            with q3:
                st.number_input(
                    "probabilistic_qr_feature_cap",
                    min_value=1,
                    max_value=64,
                    step=1,
                    key="universe_micro_prob_qr_feature_cap",
                    help="Maximum number of feature columns passed into the QR fit.",
                )
            st.caption("Tip: alpha=0.0 is the least regularised. Values around 0.01–0.10 are a sensible starting range for stability tests.")


        st.markdown("---")
        st.caption("Crash overlay preset")
        st.checkbox(
            "Enable crash-aware overlay UI",
            key="universe_crash_overlay_enabled",
            help="Shows formal downside-focused overlay presets for crash / left-tail stress testing without changing the engine backend.",
        )
        if bool(st.session_state.get("universe_crash_overlay_enabled", False)):
            co1, co2 = st.columns([1.4, 0.8])
            with co1:
                st.selectbox(
                    "Crash overlay preset",
                    list(CRASH_OVERLAY_PRESETS.keys()),
                    key="universe_crash_overlay_preset_name",
                    help="Preset bundles that tilt the probabilistic overlay toward stronger crash / downside awareness using existing overlay knobs.",
                )
            with co2:
                st.write("")
                st.write("")
                if st.button("Apply crash overlay preset", key="apply_crash_overlay_preset_btn"):
                    apply_crash_overlay_preset(str(st.session_state.get("universe_crash_overlay_preset_name", "Crash-aware balanced")))
            active_crash_preset = str(st.session_state.get("universe_crash_overlay_preset_name", "Crash-aware balanced"))
            st.caption(CRASH_OVERLAY_PRESET_META.get(active_crash_preset, ""))
            cmeta1, cmeta2, cmeta3 = st.columns(3)
            with cmeta1:
                st.metric("Overlay strength", f"{float(st.session_state.get('universe_micro_prob_overlay_strength', 1.0)):.2f}")
            with cmeta2:
                st.metric("Overlay blend", f"{float(st.session_state.get('universe_micro_prob_overlay_blend', 1.0)):.2f}")
            with cmeta3:
                st.metric("Last applied", str(st.session_state.get("universe_crash_overlay_last_applied", "Custom / manual")))
            st.info("Crash overlay presets reuse the existing probabilistic overlay engine: they raise downside sensitivity, strengthen overlay influence and keep the economic overlay-vs-none validation active.")

        st.markdown("**Overlay blending controls**")
        ob1, ob2 = st.columns(2)
        with ob1:
            st.number_input(
                "prob_overlay_strength",
                min_value=0.0,
                max_value=5.0,
                step=0.05,
                key="universe_micro_prob_overlay_strength",
                help="Scales how aggressively the probabilistic penalties/confidence reshape the base signal.",
            )
        with ob2:
            st.number_input(
                "prob_overlay_blend",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                key="universe_micro_prob_overlay_blend",
                help="Blend between base mu and overlay-adjusted mu. 0 = base only, 1 = full overlay-adjusted signal.",
            )

        st.markdown("**Probabilistic overlay advanced (Phase 4 wiring)**")

        st.markdown("**Regime-aware**")
        pra1, pra2 = st.columns(2)
        with pra1:
            st.checkbox("probabilistic_regime_aware", key="universe_micro_prob_regime_aware")
        with pra2:
            st.number_input("probabilistic_regime_min_obs", min_value=1, max_value=240, step=1, key="universe_micro_prob_regime_min_obs")

        st.markdown("**Feature filter**")
        pff1, pff2, pff3 = st.columns(3)
        with pff1:
            st.checkbox("probabilistic_feature_filter_enabled", key="universe_micro_prob_feature_filter_enabled")
        with pff2:
            st.number_input("probabilistic_feature_filter_min_obs", min_value=1, max_value=240, step=1, key="universe_micro_prob_feature_filter_min_obs")
        with pff3:
            st.checkbox("feature_filter_k = auto", key="universe_micro_prob_feature_filter_k_none")
            if not st.session_state.get("universe_micro_prob_feature_filter_k_none", True):
                st.number_input("probabilistic_feature_filter_k", min_value=1, max_value=500, step=1, key="universe_micro_prob_feature_filter_k")

        st.markdown("**KNN**")
        pknn1, pknn2, pknn3 = st.columns(3)
        with pknn1:
            st.checkbox("knn_k = auto", key="universe_micro_prob_knn_k_none")
            if not st.session_state.get("universe_micro_prob_knn_k_none", True):
                st.number_input("probabilistic_knn_k", min_value=1, max_value=500, step=1, key="universe_micro_prob_knn_k")
        with pknn2:
            st.number_input("probabilistic_knn_min_obs", min_value=1, max_value=240, step=1, key="universe_micro_prob_knn_min_obs")
            st.checkbox("probabilistic_knn_weighted_quantiles", key="universe_micro_prob_knn_weighted_quantiles")
        with pknn3:
            st.number_input("probabilistic_knn_distance_power", min_value=0.1, max_value=10.0, step=0.1, key="universe_micro_prob_knn_distance_power")
            st.number_input("probabilistic_knn_weight_eps", min_value=0.0, max_value=0.1, step=1e-6, format="%.6f", key="universe_micro_prob_knn_weight_eps")

        st.markdown("**Bucket**")
        pb1, pb2, pb3, pb4 = st.columns(4)
        with pb1:
            st.number_input("probabilistic_bucket_n_bins", min_value=2, max_value=20, step=1, key="universe_micro_prob_bucket_n_bins")
        with pb2:
            st.number_input("probabilistic_bucket_min_obs", min_value=1, max_value=240, step=1, key="universe_micro_prob_bucket_min_obs")
        with pb3:
            st.number_input("probabilistic_bucket_match_min_features", min_value=1, max_value=50, step=1, key="universe_micro_prob_bucket_match_min_features")
        with pb4:
            st.number_input("probabilistic_bucket_max_features", min_value=1, max_value=50, step=1, key="universe_micro_prob_bucket_max_features")

        st.markdown("**Parametric**")
        pp1, pp2, pp3 = st.columns(3)
        with pp1:
            st.number_input("probabilistic_parametric_min_sigma", min_value=0.0, max_value=1.0, step=1e-4, format="%.4f", key="universe_micro_prob_parametric_min_sigma")
        with pp2:
            st.number_input("probabilistic_parametric_max_sigma_mult", min_value=0.1, max_value=20.0, step=0.1, key="universe_micro_prob_parametric_max_sigma_mult")
        with pp3:
            st.checkbox("probabilistic_parametric_use_neighbor_weights", key="universe_micro_prob_parametric_use_neighbor_weights")

        st.markdown("**Hybrid**")
        ph1, ph2, ph3, ph4 = st.columns(4)
        with ph1:
            st.number_input("probabilistic_hybrid_weight", min_value=0.0, max_value=1.0, step=0.05, key="universe_micro_prob_hybrid_weight")
        with ph2:
            st.number_input("probabilistic_hybrid_min_qr_weight", min_value=0.0, max_value=1.0, step=0.05, key="universe_micro_prob_hybrid_min_qr_weight")
        with ph3:
            st.number_input("probabilistic_hybrid_max_qr_weight", min_value=0.0, max_value=1.0, step=0.05, key="universe_micro_prob_hybrid_max_qr_weight")
        with ph4:
            st.checkbox("probabilistic_hybrid_use_confidence", key="universe_micro_prob_hybrid_use_confidence")

        st.markdown("---")
        st.caption("Feature-conditioned mu + research switches (Phase 5 wiring)")
        fm1, fm2 = st.columns(2)
        with fm1:
            st.checkbox(
                "Enable feature-conditioned mu",
                key="universe_micro_feature_mu_enabled",
                help="Blends baseline mu with a feature-conditioned nearest-neighbour estimate when sufficient feature context exists.",
            )
            st.number_input("feature_mu_blend", min_value=0.0, max_value=1.0, step=0.05, key="universe_micro_feature_mu_blend")
            st.number_input("feature_mu_k", min_value=1, max_value=500, step=1, key="universe_micro_feature_mu_k")
        with fm2:
            st.number_input("feature_mu_min_obs", min_value=1, max_value=500, step=1, key="universe_micro_feature_mu_min_obs")
            st.number_input("feature_mu_cols_max", min_value=1, max_value=100, step=1, key="universe_micro_feature_mu_cols_max")
            st.checkbox(
                "pure_cs_baseline",
                key="universe_micro_pure_cs_baseline",
                help="Runs the current config through the engine's PURE_CS baseline mode for research comparisons.",
            )

        st.checkbox(
            "Run economic overlay validation vs none",
            key="universe_overlay_compare_enabled",
            help="Runs a matched no-overlay baseline and compares realised CAGR / Sharpe / drawdown / turnover against the active probabilistic overlay.",
        )
        st.checkbox(
            "Show detailed overlay validation report",
            key="universe_overlay_compare_show_detail",
        )
        ov_ui_1, ov_ui_2 = st.columns(2)
        with ov_ui_1:
            st.checkbox(
                "Show compact overlay summary cards",
                key="universe_overlay_report_show_summary",
                help="Shows clear no-overlay vs overlay metric cards for quicker interpretation.",
            )
        with ov_ui_2:
            st.checkbox(
                "Show overlay internal metrics in report",
                key="universe_overlay_report_show_internal_metrics",
                help="Includes interval width, downside risk, confidence and feature-aware share when available.",
            )
        st.checkbox(
            "Show overlay diagnostics tables",
            key="universe_overlay_report_show_diagnostics",
            help="Adds summary/diagnostic tables for deeper inspection of the probabilistic overlay.",
        )

        st.markdown("---")
        st.caption("Research-only tools — param grid runner / param sweep")
        st.checkbox(
            "Enable param grid runner",
            key="universe_param_sweep_enabled",
            help="Run a grid over temperature, weight_shrink and inertia and inspect the full parameter table.",
        )
        st.text_input(
            "param sweep temperature grid",
            key="universe_param_sweep_temp_grid_text",
            help="Comma-separated values. Example: 0.7, 1.0, 1.3",
        )
        st.text_input(
            "param sweep weight_shrink grid",
            key="universe_param_sweep_shrink_grid_text",
            help="Comma-separated values. Example: 0.0, 0.05, 0.10",
        )
        st.text_input(
            "param sweep inertia grid",
            key="universe_param_sweep_inertia_grid_text",
            help="Comma-separated values. Example: 0.0, 0.2, 0.4",
        )

        st.markdown("---")
        st.caption("α sweep (sigma_power_alpha)")
        st.checkbox(
            "Enable α sweep",
            key="universe_alpha_sweep_enabled",
            help="Runs the engine across a grid of sigma_power_alpha values. In the current engine, α controls how strongly volatility is penalised in the allocation score, approximately through mu / sigma^alpha.",
        )
        st.caption("Interpretation: higher α penalises volatility more strongly in the allocator score; lower α makes the engine more tolerant of high-volatility assets.")


        st.markdown("---")
        st.caption("Parameter stability module")
        st.checkbox(
            "Enable parameter stability module",
            key="universe_parameter_stability_enabled",
            help="Runs a local grid around temperature, weight_shrink and inertia, then ranks candidates by robustness rather than just the single best Sharpe point.",
        )
        ps1, ps2 = st.columns(2)
        with ps1:
            st.text_input(
                "stability temperature grid",
                key="universe_parameter_stability_temp_grid_text",
                help="Comma-separated values. Example: 0.7, 1.0, 1.3",
            )
            st.text_input(
                "stability weight_shrink grid",
                key="universe_parameter_stability_shrink_grid_text",
                help="Comma-separated values. Example: 0.0, 0.05, 0.10",
            )
        with ps2:
            st.text_input(
                "stability inertia grid",
                key="universe_parameter_stability_inertia_grid_text",
                help="Comma-separated values. Example: 0.0, 0.2, 0.4",
            )
            st.number_input(
                "near-best Sharpe tolerance",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                key="universe_parameter_stability_near_best_tol",
                help="A candidate is treated as near-best if its Sharpe is within this absolute distance of the best Sharpe in the local grid.",
            )
        st.checkbox(
            "Show detailed stability tables",
            key="universe_parameter_stability_show_detail",
        )

        st.markdown("---")
        st.caption("Sensitivity analysis")
        st.checkbox(
            "Enable sensitivity analysis",
            key="universe_sensitivity_analysis_enabled",
            help="Runs a local parameter grid and quantifies how sensitive the chosen objective is to temperature, weight_shrink and inertia.",
        )
        sa1, sa2 = st.columns(2)
        with sa1:
            st.text_input(
                "sensitivity temperature grid",
                key="universe_sensitivity_temp_grid_text",
                help="Comma-separated values. Example: 0.7, 1.0, 1.3",
            )
            st.text_input(
                "sensitivity weight_shrink grid",
                key="universe_sensitivity_shrink_grid_text",
                help="Comma-separated values. Example: 0.0, 0.05, 0.10",
            )
        with sa2:
            st.text_input(
                "sensitivity inertia grid",
                key="universe_sensitivity_inertia_grid_text",
                help="Comma-separated values. Example: 0.0, 0.2, 0.4",
            )
            st.selectbox(
                "sensitivity metric",
                ["sharpe", "cagr", "information_ratio"],
                key="universe_sensitivity_metric",
                help="Primary metric used to quantify sensitivity across the local grid.",
            )
        st.checkbox(
            "Show detailed sensitivity tables",
            key="universe_sensitivity_show_detail",
        )

        st.markdown("---")
        st.caption("Robust region detection")
        st.checkbox(
            "Enable robust region detection",
            key="universe_robust_region_enabled",
            help="Uses the plateau-region helpers from evaluation.py to detect a robust parameter region around the best Sharpe candidate instead of focusing only on a single optimum.",
        )
        rr1, rr2 = st.columns(2)
        with rr1:
            st.text_input(
                "robust region temperature grid",
                key="universe_robust_region_temp_grid_text",
                help="Comma-separated values. Example: 0.7, 1.0, 1.3",
            )
            st.text_input(
                "robust region weight_shrink grid",
                key="universe_robust_region_shrink_grid_text",
                help="Comma-separated values. Example: 0.0, 0.05, 0.10",
            )
            st.number_input(
                "robust region Sharpe tolerance",
                min_value=0.50,
                max_value=1.00,
                step=0.01,
                key="universe_robust_region_sharpe_rel_tol",
                help="Rows with Sharpe >= tolerance × best Sharpe are flagged as belonging to the plateau region.",
            )
        with rr2:
            st.text_input(
                "robust region inertia grid",
                key="universe_robust_region_inertia_grid_text",
                help="Comma-separated values. Example: 0.0, 0.2, 0.4",
            )
            st.checkbox(
                "Apply turnover rule inside robust region",
                key="universe_robust_region_max_turnover_enabled",
                help="If enabled, plateau membership also requires turnover <= the threshold below.",
            )
            st.number_input(
                "robust region max turnover",
                min_value=0.0,
                max_value=2.0,
                step=0.05,
                key="universe_robust_region_max_turnover",
                help="Optional turnover ceiling used when flagging the plateau region and choosing the stable candidate.",
            )
        st.checkbox(
            "Show detailed robust-region tables",
            key="universe_robust_region_show_detail",
        )

        st.markdown("---")
        st.caption("Breadth vs estimation formal")
        st.checkbox(
            "Enable breadth vs estimation formal",
            key="universe_breadth_estimation_enabled",
            help="Formalises the trade-off between effective breadth and estimation load using the local parameter grid.",
        )
        be1, be2 = st.columns(2)
        with be1:
            st.text_input(
                "breadth/estimation temperature grid",
                key="universe_breadth_estimation_temp_grid_text",
                help="Comma-separated values. Example: 0.7, 1.0, 1.3",
            )
            st.text_input(
                "breadth/estimation weight_shrink grid",
                key="universe_breadth_estimation_shrink_grid_text",
                help="Comma-separated values. Example: 0.0, 0.05, 0.10",
            )
        with be2:
            st.text_input(
                "breadth/estimation inertia grid",
                key="universe_breadth_estimation_inertia_grid_text",
                help="Comma-separated values. Example: 0.0, 0.2, 0.4",
            )
            st.selectbox(
                "breadth/estimation objective",
                ["sharpe", "information_ratio", "cagr"],
                key="universe_breadth_estimation_metric",
                help="Primary performance metric used when balancing breadth against estimation load.",
            )
        be3, be4 = st.columns(2)
        with be3:
            st.number_input(
                "breadth buckets",
                min_value=2,
                max_value=5,
                step=1,
                key="universe_breadth_estimation_bins",
                help="Number of quantile buckets used to summarise the breadth trade-off.",
            )
        with be4:
            st.checkbox(
                "Show detailed breadth/estimation tables",
                key="universe_breadth_estimation_show_detail",
            )

        st.text_input(
            "alpha grid",
            key="universe_alpha_sweep_grid_text",
            help="Comma-separated values. Example: 0.5, 0.75, 1.0, 1.25, 1.5",
        )
        st.checkbox(
            "Show α sweep frontier",
            key="universe_alpha_sweep_show_frontier",
            help="Adds a simple Sharpe-vs-turnover frontier view over the alpha sweep results.",
        )

        st.markdown("---")
        st.caption("Sharpe vs turnover frontier")
        st.checkbox(
            "Enable Sharpe vs turnover frontier",
            key="universe_frontier_enabled",
            help="Run a small grid over temperature, weight_shrink and inertia and trace the non-dominated Sharpe/turnover frontier.",
        )
        st.text_input(
            "frontier temperature grid",
            key="universe_frontier_temp_grid_text",
            help="Comma-separated values. Example: 0.7, 1.0, 1.3",
        )
        st.text_input(
            "frontier weight_shrink grid",
            key="universe_frontier_shrink_grid_text",
            help="Comma-separated values. Example: 0.0, 0.05, 0.10",
        )
        st.text_input(
            "frontier inertia grid",
            key="universe_frontier_inertia_grid_text",
            help="Comma-separated values. Example: 0.0, 0.2, 0.4",
        )
        st.number_input(
            "frontier max turnover rule",
            min_value=0.0,
            max_value=2.0,
            step=0.05,
            key="universe_frontier_max_turnover",
            help="Used to choose a candidate from the frontier.",
        )

        st.markdown("---")
        if workspace_mode != "Global Search":
            st.caption("Bayesian / Optuna / surrogate / NSGA-II tuning")
            st.checkbox(
                "Enable advanced tuning research block",
                key="universe_adv_tuning_enabled",
                help="Runs a compact research-level tuner over a local parameter space and reports the best candidate.",
            )
            at1, at2 = st.columns(2)
            with at1:
                st.selectbox(
                    "tuning engine",
                    ["bayesian_style", "random", "surrogate", "optuna", "nsga2", "global_multiobjective"],
                    key="universe_adv_tuning_method",
                    help="Optuna falls back gracefully if the library is not installed. NSGA-II runs a Pareto multi-objective search over the same sanitised param space. Global multiobjective search uses the evaluation.py global schema-guided search backend.",
                )
                st.selectbox(
                    "tuning objective",
                    TUNING_OBJECTIVE_OPTIONS,
                    key="universe_adv_tuning_objective",
                    format_func=_format_tuning_objective_label,
                )
                st.number_input(
                    "tuning trials",
                    min_value=4,
                    max_value=200,
                    step=1,
                    key="universe_adv_tuning_n_trials",
                    help="Used by random / surrogate / bayesian-style / optuna. NSGA-II uses population size × generations instead.",
                )
            with at2:
                st.number_input(
                    "bayesian warmup",
                    min_value=1,
                    max_value=100,
                    step=1,
                    key="universe_adv_tuning_warmup",
                    help="Used by the Bayesian-style tuner before local exploitation starts.",
                )
                st.checkbox(
                    "Show detailed tuning tables",
                    key="universe_adv_tuning_show_detail",
                )
                st.multiselect(
                    "NSGA-II Pareto objectives",
                    PARETO_OBJECTIVE_OPTIONS,
                    default=st.session_state.get("universe_adv_tuning_nsga2_objectives", ["sharpe", "cagr", "max_drawdown", "mean_turnover", "diversification", "stability"]),
                    key="universe_adv_tuning_nsga2_objectives",
                    format_func=_format_pareto_objective_label,
                    help="Used only when tuning engine = nsga2.",
                )
            nsga_col1, nsga_col2, nsga_col3 = st.columns(3)
            with nsga_col1:
                st.number_input(
                    "NSGA-II population size",
                    min_value=4,
                    max_value=200,
                    step=1,
                    key="universe_adv_tuning_nsga2_population_size",
                )
                st.number_input(
                    "NSGA-II generations",
                    min_value=1,
                    max_value=100,
                    step=1,
                    key="universe_adv_tuning_nsga2_generations",
                )
            with nsga_col2:
                st.number_input(
                    "NSGA-II mutation rate",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    key="universe_adv_tuning_nsga2_mutation_rate",
                )
                st.number_input(
                    "NSGA-II crossover rate",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    key="universe_adv_tuning_nsga2_crossover_rate",
                )
            with nsga_col3:
                st.number_input(
                    "NSGA-II seed",
                    min_value=0,
                    max_value=999999,
                    step=1,
                    key="universe_adv_tuning_nsga2_seed",
                )
                st.caption("NSGA-II controls are only used when the tuning engine is set to nsga2.")

            st.markdown("---")
            st.caption("Global multiobjective search")
            gm1, gm2, gm3 = st.columns(3)
            with gm1:
                st.number_input(
                    "global total budget",
                    min_value=8,
                    max_value=2000,
                    step=4,
                    key="universe_adv_global_total_budget",
                    help="Total candidate budget for the global multiobjective search backend.",
                )
                st.selectbox(
                    "seed strategy",
                    ["hybrid", "sobol", "lhs", "random"],
                    key="universe_adv_global_seed_strategy",
                    help="Controls the initial global exploration method used by run_global_multiobjective_search().",
                )
            with gm2:
                st.selectbox(
                    "categorical breadth",
                    ["narrow", "medium", "wide"],
                    key="universe_adv_global_categorical_breadth",
                    help="Controls how many categorical choices remain active in the schema before the global search starts.",
                )
                st.number_input(
                    "exploitation ratio",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    key="universe_adv_global_exploitation_ratio",
                    help="Higher values shrink numeric grids around the current config before the global search runs.",
                )
            with gm3:
                st.selectbox(
                    "resume / restart",
                    ["restart", "resume"],
                    key="universe_adv_global_resume_mode",
                    help="Resume reuses the last compatible global-search result when the current signature matches and the stored budget already covers the requested budget.",
                )
                st.caption("Global multiobjective controls are only used when tuning engine = global_multiobjective.")

            rec1, rec2, rec3 = st.columns(3)
            with rec1:
                st.checkbox(
                    "recursive search enabled",
                    key="universe_adv_tuning_recursive_enabled",
                    help="Runs recursive multi-run search on top of NSGA-II: explore, refine around the frontier, then stop when convergence is detected.",
                )
                st.number_input(
                    "max recursion depth",
                    min_value=1,
                    max_value=10,
                    step=1,
                    key="universe_adv_tuning_recursive_depth",
                    help="Maximum number of recursive search rounds when recursive search is enabled.",
                )
            with rec2:
                st.number_input(
                    "shrink factor",
                    min_value=0.05,
                    max_value=1.0,
                    step=0.05,
                    key="universe_adv_tuning_recursive_shrink_factor",
                    help="Base shrink factor used when narrowing the parameter space around the frontier between rounds.",
                )
                st.number_input(
                    "convergence tolerance",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.0005,
                    format="%.4f",
                    key="universe_adv_tuning_recursive_convergence_tol",
                    help="Stops recursive search when frontier improvement becomes too small.",
                )
            with rec3:
                st.number_input(
                    "seed frontier size",
                    min_value=1,
                    max_value=200,
                    step=1,
                    key="universe_adv_tuning_recursive_seed_frontier_size",
                    help="How many frontier seeds to carry forward across recursive rounds.",
                )
                st.caption("Recursive controls are only used when tuning engine = nsga2 and recursive search is enabled.")
        else:
            st.caption("Global multiobjective search")
            st.session_state["universe_adv_tuning_method"] = "global_multiobjective"
            st.checkbox(
                "Enable global search module",
                key="universe_adv_tuning_enabled",
                help="Runs the dedicated global multiobjective search module and renders the frontier, history, fingerprints and parameter coverage below.",
            )
            st.multiselect(
                "Global search objectives",
                PARETO_OBJECTIVE_OPTIONS,
                default=st.session_state.get("universe_adv_tuning_nsga2_objectives", ["sharpe", "cagr", "max_drawdown", "mean_turnover", "diversification", "stability"]),
                key="universe_adv_tuning_nsga2_objectives",
                format_func=_format_pareto_objective_label,
                help="Objectives used by the schema-guided global search frontier.",
            )
            gm1, gm2, gm3 = st.columns(3)
            with gm1:
                st.number_input(
                    "total budget",
                    min_value=8,
                    max_value=2000,
                    step=4,
                    key="universe_adv_global_total_budget",
                )
                st.selectbox(
                    "seed strategy",
                    ["hybrid", "sobol", "lhs", "random"],
                    key="universe_adv_global_seed_strategy",
                )
            with gm2:
                st.selectbox(
                    "categorical breadth",
                    ["narrow", "medium", "wide"],
                    key="universe_adv_global_categorical_breadth",
                )
                st.number_input(
                    "exploitation ratio",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    key="universe_adv_global_exploitation_ratio",
                )
            with gm3:
                st.selectbox(
                    "resume / restart",
                    ["restart", "resume"],
                    key="universe_adv_global_resume_mode",
                )
                if st.button("Run global search", key="universe_global_search_run_button", use_container_width=True):
                    st.session_state["universe_global_search_run_nonce"] = int(st.session_state.get("universe_global_search_run_nonce", 0)) + 1
                    st.session_state["universe_adv_tuning_enabled"] = True
                    st.rerun()
            st.caption("Results below: frontier, history, explored fingerprints, parameter coverage and best compromise candidate.")

        at3, at4 = st.columns(2)
        with at3:
            st.text_input(
                "tuning temperature grid",
                key="universe_adv_tuning_temp_grid_text",
                help="Comma-separated values. Example: 0.7, 1.0, 1.3",
            )
            st.text_input(
                "tuning weight_shrink grid",
                key="universe_adv_tuning_shrink_grid_text",
                help="Comma-separated values. Example: 0.0, 0.05, 0.10",
            )
            st.text_input(
                "tuning sigma_power_alpha grid",
                key="universe_adv_tuning_alpha_grid_text",
                help="Comma-separated values. Example: 0.75, 1.0, 1.25. This dimension is engine-dependent and may be dropped automatically when inactive for the current signal_mode.",
            )
            st.caption("sigma_power_alpha is engine-dependent; the sanitiser will remove it from tuning when the current signal_mode makes it inactive.")
        with at4:
            st.text_input(
                "tuning inertia grid",
                key="universe_adv_tuning_inertia_grid_text",
                help="Comma-separated values. Example: 0.0, 0.2, 0.4",
            )
            st.text_input(
                "tuning top_k grid (optional)",
                key="universe_adv_tuning_topk_grid_text",
                help="Comma-separated integers. Leave blank to keep top_k fixed.",
            )

        st.markdown("---")
        st.caption("Dispersion → risk model")
        ds1, ds2 = st.columns(2)
        with ds1:
            st.checkbox(
                "dispersion_sigma_enabled",
                key="universe_micro_dispersion_sigma_enabled",
                help="Use recent cross-sectional dispersion to scale sigma_hat in the risk model.",
            )
            st.number_input(
                "dispersion_sigma_lookback",
                min_value=3,
                max_value=60,
                step=1,
                key="universe_micro_dispersion_sigma_lookback",
            )
            st.number_input(
                "dispersion_sigma_strength",
                min_value=0.0,
                max_value=5.0,
                step=0.05,
                key="universe_micro_dispersion_sigma_strength",
            )
        with ds2:
            st.number_input(
                "dispersion_sigma_floor_mult",
                min_value=0.10,
                max_value=2.0,
                step=0.05,
                key="universe_micro_dispersion_sigma_floor_mult",
            )
            st.number_input(
                "dispersion_sigma_ceiling_mult",
                min_value=0.10,
                max_value=3.0,
                step=0.05,
                key="universe_micro_dispersion_sigma_ceiling_mult",
            )

        st.markdown("---")
        st.caption("Dispersion governance (Phase 1 wiring)")
        dg1, dg2 = st.columns(2)
        with dg1:
            st.checkbox("dispersion_gate", key="universe_micro_dispersion_gate")
            st.number_input("dispersion_gate_threshold", min_value=0.0, max_value=1.0, step=0.01, key="universe_micro_dispersion_gate_threshold")
            st.number_input("dispersion_gate_min_active_weight", min_value=0.0, max_value=1.0, step=0.01, key="universe_micro_dispersion_gate_min_active_weight")
        with dg2:
            st.checkbox("dispersion_top_k_enabled", key="universe_micro_dispersion_top_k_enabled")
            st.number_input("dispersion_top_k_threshold_low", min_value=0.0, max_value=1.0, step=0.01, key="universe_micro_dispersion_top_k_threshold_low")
            st.number_input("dispersion_top_k_threshold_high", min_value=0.0, max_value=1.0, step=0.01, key="universe_micro_dispersion_top_k_threshold_high")
            st.number_input("dispersion_top_k_low_mult", min_value=0.0, max_value=5.0, step=0.05, key="universe_micro_dispersion_top_k_low_mult")
            st.number_input("dispersion_top_k_high_mult", min_value=0.0, max_value=5.0, step=0.05, key="universe_micro_dispersion_top_k_high_mult")
            st.checkbox("Enable dispersion top_k min_k", key="universe_micro_dispersion_top_k_min_k_enabled")
            if st.session_state.get("universe_micro_dispersion_top_k_min_k_enabled", False):
                st.number_input("dispersion_top_k_min_k", min_value=1, max_value=100, step=1, key="universe_micro_dispersion_top_k_min_k")
            st.checkbox("Enable dispersion top_k max_k", key="universe_micro_dispersion_top_k_max_k_enabled")
            if st.session_state.get("universe_micro_dispersion_top_k_max_k_enabled", False):
                st.number_input("dispersion_top_k_max_k", min_value=1, max_value=100, step=1, key="universe_micro_dispersion_top_k_max_k")

        with st.expander("Experimental / engine-dependent controls", expanded=bool(st.session_state.get("universe_micro_dispersion_risk_model_enabled", False))):
            st.caption("These controls are wired in the current engine and remain exposed for research continuity. Use the tuning/activity audit and overlay summary to verify when the dispersion risk model is active and how strongly it changes the risk model.")
            ed1, ed2 = st.columns(2)
            with ed1:
                st.checkbox("dispersion_risk_model_enabled", key="universe_micro_dispersion_risk_model_enabled")
                st.number_input("dispersion_risk_strength", min_value=0.0, max_value=5.0, step=0.05, key="universe_micro_dispersion_risk_strength")
                st.number_input("dispersion_risk_floor_mult", min_value=0.0, max_value=5.0, step=0.05, key="universe_micro_dispersion_risk_floor_mult")
                st.number_input("dispersion_risk_ceiling_mult", min_value=0.0, max_value=5.0, step=0.05, key="universe_micro_dispersion_risk_ceiling_mult")
            with ed2:
                st.checkbox("dispersion_risk_apply_to_covariance", key="universe_micro_dispersion_risk_apply_to_covariance")
                st.caption("sigma_power_alpha is also treated as engine-dependent in tuning and may be dropped automatically when inactive for the current signal mode.")

        dg3, dg4 = st.columns(2)
        with dg3:
            st.checkbox("dispersion_vol_target_enabled", key="universe_micro_dispersion_vol_target_enabled")
            st.number_input("dispersion_vol_target_threshold_low", min_value=0.0, max_value=1.0, step=0.01, key="universe_micro_dispersion_vol_target_threshold_low")
            st.number_input("dispersion_vol_target_threshold_high", min_value=0.0, max_value=1.0, step=0.01, key="universe_micro_dispersion_vol_target_threshold_high")
            st.number_input("dispersion_vol_target_low_mult", min_value=0.0, max_value=5.0, step=0.05, key="universe_micro_dispersion_vol_target_low_mult")
            st.number_input("dispersion_vol_target_high_mult", min_value=0.0, max_value=5.0, step=0.05, key="universe_micro_dispersion_vol_target_high_mult")
            st.checkbox("Enable dispersion vol target min", key="universe_micro_dispersion_vol_target_min_enabled")
            if st.session_state.get("universe_micro_dispersion_vol_target_min_enabled", False):
                st.number_input("dispersion_vol_target_min", min_value=0.0, max_value=1.0, step=0.005, format="%.3f", key="universe_micro_dispersion_vol_target_min")
            st.checkbox("Enable dispersion vol target max", key="universe_micro_dispersion_vol_target_max_enabled")
            if st.session_state.get("universe_micro_dispersion_vol_target_max_enabled", False):
                st.number_input("dispersion_vol_target_max", min_value=0.0, max_value=1.0, step=0.005, format="%.3f", key="universe_micro_dispersion_vol_target_max")
        with dg4:
            st.checkbox("low_signal_fallback_to_ew", key="universe_micro_low_signal_fallback_to_ew")
            st.number_input("low_signal_fallback_threshold", min_value=0.0, max_value=1.0, step=0.01, key="universe_micro_low_signal_fallback_threshold")
            st.selectbox("low_signal_fallback_mode", ["blend", "hard"], key="universe_micro_low_signal_fallback_mode")
            st.number_input("low_signal_fallback_min_model_weight", min_value=0.0, max_value=1.0, step=0.01, key="universe_micro_low_signal_fallback_min_model_weight")

        st.markdown("---")
        st.caption("Signal model extensions")
        sm1, sm2 = st.columns(2)
        with sm1:
            st.selectbox(
                "signal_mode",
                ["mu_sigma", "huber_mu", "lambdarank_like", "lambdarank_real", "directional_classifier", "logistic_loss", "top_k_classifier", "quantile_loss"],
                key="universe_micro_signal_mode",
                help="mu_sigma is the default. huber_mu applies a robust Huber-style location estimate. lambdarank_like is the legacy ranking-style layer. lambdarank_real adds a stronger pairwise ranking signal. directional_classifier builds a probability-style up/down signal from recent sign persistence and standardized returns. logistic_loss fits a simple logistic directional layer. top_k_classifier learns a probability-style top-k winner layer. quantile_loss fits a one-dimensional quantile-style signal that can be useful when you want asymmetric return conditioning.",
            )
            st.number_input(
                "signal_score_blend",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                key="universe_micro_signal_score_blend",
            )
            st.number_input(
                "directional_classifier_threshold",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                key="universe_micro_directional_classifier_threshold",
            )
            st.number_input(
                "logistic_loss_threshold",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                key="universe_micro_logistic_loss_threshold",
            )
            st.number_input(
                "top_k_classifier_threshold",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                key="universe_micro_top_k_classifier_threshold",
            )
            st.number_input(
                "quantile_loss_q",
                min_value=0.01,
                max_value=0.99,
                step=0.01,
                key="universe_micro_quantile_loss_q",
            )
            st.checkbox(
                "Enable multi-loss training",
                key="universe_multi_loss_enabled",
                help="Blends ranking, directional and return-style sub-signals into a single research signal. Default behaviour is unchanged when disabled.",
            )
        with sm2:
            st.number_input(
                "huber_delta",
                min_value=0.1,
                max_value=10.0,
                step=0.1,
                key="universe_micro_huber_delta",
            )
            st.number_input(
                "lambdarank_lookback",
                min_value=3,
                max_value=60,
                step=1,
                key="universe_micro_lambdarank_lookback",
            )
            st.number_input(
                "lambdarank_temperature",
                min_value=0.1,
                max_value=5.0,
                step=0.1,
                key="universe_micro_lambdarank_temperature",
            )
            st.number_input(
                "lambdarank_real_lookback",
                min_value=3,
                max_value=120,
                step=1,
                key="universe_micro_lambdarank_real_lookback",
            )
            st.number_input(
                "lambdarank_real_temperature",
                min_value=0.1,
                max_value=5.0,
                step=0.1,
                key="universe_micro_lambdarank_real_temperature",
            )
            st.number_input(
                "directional_classifier_lookback",
                min_value=3,
                max_value=60,
                step=1,
                key="universe_micro_directional_classifier_lookback",
            )
            st.number_input(
                "directional_classifier_confidence_scale",
                min_value=0.1,
                max_value=5.0,
                step=0.1,
                key="universe_micro_directional_classifier_confidence_scale",
            )
            st.number_input(
                "logistic_loss_lookback",
                min_value=3,
                max_value=120,
                step=1,
                key="universe_micro_logistic_loss_lookback",
            )
            st.number_input(
                "logistic_loss_l2",
                min_value=0.0,
                max_value=10.0,
                step=0.05,
                key="universe_micro_logistic_loss_l2",
            )
            st.number_input(
                "logistic_loss_confidence_scale",
                min_value=0.1,
                max_value=5.0,
                step=0.1,
                key="universe_micro_logistic_loss_confidence_scale",
            )
            st.number_input(
                "top_k_classifier_lookback",
                min_value=3,
                max_value=120,
                step=1,
                key="universe_micro_top_k_classifier_lookback",
            )
            st.number_input(
                "top_k_classifier_k",
                min_value=1,
                max_value=100,
                step=1,
                key="universe_micro_top_k_classifier_k",
            )
            st.number_input(
                "top_k_classifier_confidence_scale",
                min_value=0.1,
                max_value=5.0,
                step=0.1,
                key="universe_micro_top_k_classifier_confidence_scale",
            )
            st.number_input(
                "quantile_loss_lookback",
                min_value=3,
                max_value=120,
                step=1,
                key="universe_micro_quantile_loss_lookback",
            )
            st.number_input(
                "quantile_loss_alpha",
                min_value=0.0,
                max_value=10.0,
                step=0.05,
                key="universe_micro_quantile_loss_alpha",
            )
            st.number_input(
                "quantile_loss_confidence_scale",
                min_value=0.1,
                max_value=5.0,
                step=0.1,
                key="universe_micro_quantile_loss_confidence_scale",
            )
            st.number_input(
                "lambdarank_real_gain_power",
                min_value=0.0,
                max_value=5.0,
                step=0.1,
                key="universe_micro_lambdarank_real_gain_power",
            )
            st.number_input(
                "lambdarank_real_pair_power",
                min_value=0.0,
                max_value=5.0,
                step=0.1,
                key="universe_micro_lambdarank_real_pair_power",
            )
            st.number_input(
                "lambdarank_real_l2",
                min_value=0.0,
                max_value=10.0,
                step=0.05,
                key="universe_micro_lambdarank_real_l2",
            )

        if bool(st.session_state.get("universe_multi_loss_enabled", False)):
            st.caption("Multi-loss blend")
            ml1, ml2 = st.columns(2)
            with ml1:
                st.number_input("multi_loss_rank_weight", min_value=0.0, max_value=5.0, step=0.05, key="universe_multi_loss_rank_weight")
                st.number_input("multi_loss_direction_weight", min_value=0.0, max_value=5.0, step=0.05, key="universe_multi_loss_direction_weight")
                st.number_input("multi_loss_return_weight", min_value=0.0, max_value=5.0, step=0.05, key="universe_multi_loss_return_weight")
            with ml2:
                st.number_input("multi_loss_logistic_weight", min_value=0.0, max_value=5.0, step=0.05, key="universe_multi_loss_logistic_weight")
                st.number_input("multi_loss_topk_weight", min_value=0.0, max_value=5.0, step=0.05, key="universe_multi_loss_topk_weight")
                st.number_input("multi_loss_temperature", min_value=0.1, max_value=5.0, step=0.1, key="universe_multi_loss_temperature")
                st.number_input("multi_loss_l2", min_value=0.0, max_value=10.0, step=0.05, key="universe_multi_loss_l2")

        st.markdown("---")
        st.caption("Regime-dependent universe")
        st.checkbox(
            "Enable regime-dependent universe",
            key="universe_regime_universe_enabled",
            help="Lets the eligible asset universe change with the detected regime before allocation is built. High regime can become more defensive, low regime can stay broader or tilt more offensive.",
        )
        ru1, ru2 = st.columns(2)
        with ru1:
            st.selectbox(
                "Universe selection metric",
                ["mu_over_sigma", "score", "mu"],
                key="universe_regime_universe_metric",
                help="Controls how assets are ranked before the regime-specific keep fraction is applied.",
            )
            st.number_input(
                "Keep fraction LOW regime",
                min_value=0.05,
                max_value=1.00,
                step=0.05,
                key="universe_regime_keep_low",
                help="Fraction of candidate assets kept in low-risk / favourable regimes.",
            )
            st.number_input(
                "Keep fraction MID regime",
                min_value=0.05,
                max_value=1.00,
                step=0.05,
                key="universe_regime_keep_mid",
                help="Fraction of candidate assets kept in mid / neutral regimes.",
            )
            st.number_input(
                "Keep fraction HIGH regime",
                min_value=0.05,
                max_value=1.00,
                step=0.05,
                key="universe_regime_keep_high",
                help="Fraction of candidate assets kept in stressed / high-risk regimes.",
            )
        with ru2:
            st.number_input(
                "Minimum assets kept",
                min_value=2,
                max_value=500,
                step=1,
                key="universe_regime_min_assets",
                help="Hard lower bound so the allocator still has enough names to work with.",
            )
            st.checkbox(
                "Enable max-assets cap for regime universe",
                key="universe_regime_max_assets_enabled",
                help="Optional hard cap on the number of names kept after regime filtering.",
            )
            if st.session_state.get("universe_regime_max_assets_enabled", False):
                st.number_input(
                    "Maximum assets kept",
                    min_value=2,
                    max_value=500,
                    step=1,
                    key="universe_regime_max_assets",
                )
            st.number_input(
                "LOW regime offensive vol tilt",
                min_value=0.0,
                max_value=3.0,
                step=0.05,
                key="universe_regime_low_vol_tilt",
                help="Positive values make low regimes slightly more willing to keep higher-vol names when ranking the universe.",
            )
            st.number_input(
                "HIGH regime defensive vol tilt",
                min_value=0.0,
                max_value=3.0,
                step=0.05,
                key="universe_regime_high_vol_tilt",
                help="Positive values make stressed regimes favour lower-vol names more strongly.",
            )
        if bool(st.session_state.get("universe_regime_universe_enabled", False)):
            low_keep = float(st.session_state.get("universe_regime_keep_low", 1.00))
            mid_keep = float(st.session_state.get("universe_regime_keep_mid", 0.85))
            high_keep = float(st.session_state.get("universe_regime_keep_high", 0.60))
            min_assets = int(st.session_state.get("universe_regime_min_assets", 2))
            max_assets = int(st.session_state.get("universe_regime_max_assets", 25)) if bool(st.session_state.get("universe_regime_max_assets_enabled", False)) else None
            metric_name = str(st.session_state.get("universe_regime_universe_metric", "mu_over_sigma"))
            max_assets_text = str(max_assets) if max_assets is not None else "none"
            st.caption(
                f"Active regime universe rule → metric={metric_name} | keep low/mid/high = {low_keep:.2f}/{mid_keep:.2f}/{high_keep:.2f} | min assets = {min_assets} | max assets = {max_assets_text}."
            )


    if isinstance(panel_df, pd.DataFrame) and not panel_df.empty:
        try:
            primary_panel_df, primary_info = _filter_asset_panel_to_universe(panel_df, selected_assets)
            primary_daily_panel_df = None
            if yahoo_daily_panel_df is not None:
                primary_daily_panel_df, _ = _filter_asset_panel_to_universe(yahoo_daily_panel_df, selected_assets)
            primary_engine_panel_df, primary_engine_meta = _build_engine_asset_panel(
                primary_panel_df,
                source_mode=data_source_mode,
                requested_frequency=(str(yahoo_frequency) if data_source_mode == "Yahoo Finance (recommended)" else None),
                daily_panel_df=primary_daily_panel_df,
                macro_panel_df=yahoo_macro_panel_df,
            )
            st.session_state["last_missing_universe_assets"] = primary_info["missing_assets"]

            comparison_panel_df = None
            comparison_info = None
            comparison_engine_panel_df = None
            comparison_engine_meta = {}

            st.success(
                f"Source ready ({panel_source_label or 'asset panel'}): {len(panel_df):,} rows across {panel_df['asset'].nunique()} assets. "
                f"Primary filtered universe uses {primary_panel_df['asset'].nunique()} assets."
            )
            primary_scale_kind, primary_scale_text = _large_universe_test_summary(len(primary_info["used_assets"]))
            badge(f"Primary universe scale check: {primary_scale_text}", primary_scale_kind)
            if comparison_info is None and len(primary_info["used_assets"]) >= 50:
                st.caption("This run is already operating in the formal large-universe regime on the primary basket.")
            if recommendation_candidate_assets:
                st.caption(f"Strategy candidate pool available for later recommendation testing: {len(recommendation_candidate_assets)} assets.")

            if yahoo_macro_panel_df is not None and not yahoo_macro_panel_df.empty:
                macro_cols = [c for c in yahoo_macro_panel_df.columns if c != "date"]
                st.caption(f"Macro context downloaded: {len(macro_cols)} columns (VIX / rates).")
            if primary_info["missing_assets"]:
                st.caption(f"Primary assets not found in the panel: {', '.join(primary_info['missing_assets'])}")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Primary universe panel")
                st.caption(
                    f"Assets used: {len(primary_info['used_assets'])} | Rows: {len(primary_panel_df):,} | "
                    f"Date range: {primary_panel_df['date'].min().date()} → {primary_panel_df['date'].max().date()}"
                )
                with st.expander("Preview primary filtered asset panel"):
                    st.dataframe(primary_panel_df.head(50), use_container_width=True)
                if primary_engine_meta.get("feature_enriched", False):
                    st.caption(
                        f"Engine panel: {primary_engine_meta.get('panel_kind')} | features: {len(primary_engine_meta.get('feature_cols', []))} | rows: {primary_engine_meta.get('engine_rows')}"
                    )
                    with st.expander("Preview primary engine panel (feature-enriched)", expanded=False):
                        st.dataframe(primary_engine_panel_df.head(50), use_container_width=True)
                else:
                    st.caption(primary_engine_meta.get("message", "Using base return panel."))

            if st.session_state.get("comparison_enabled", False) and comparison_assets:
                comparison_panel_df, comparison_info = _filter_asset_panel_to_universe(panel_df, comparison_assets)
                comparison_daily_panel_df = None
                if yahoo_daily_panel_df is not None:
                    comparison_daily_panel_df, _ = _filter_asset_panel_to_universe(yahoo_daily_panel_df, comparison_assets)
                comparison_engine_panel_df, comparison_engine_meta = _build_engine_asset_panel(
                    comparison_panel_df,
                    source_mode=data_source_mode,
                    requested_frequency=(str(yahoo_frequency) if data_source_mode == "Yahoo Finance (recommended)" else None),
                    daily_panel_df=comparison_daily_panel_df,
                    macro_panel_df=yahoo_macro_panel_df,
                )
                st.session_state["last_comparison_missing_universe_assets"] = comparison_info["missing_assets"]

                with c2:
                    st.markdown("#### Comparison universe panel")
                    st.caption(
                        f"Assets used: {len(comparison_info['used_assets'])} | Rows: {len(comparison_panel_df):,} | "
                        f"Date range: {comparison_panel_df['date'].min().date()} → {comparison_panel_df['date'].max().date()}"
                    )
                    with st.expander("Preview comparison filtered asset panel"):
                        st.dataframe(comparison_panel_df.head(50), use_container_width=True)
                    if comparison_engine_meta.get("feature_enriched", False):
                        st.caption(
                            f"Engine panel: {comparison_engine_meta.get('panel_kind')} | features: {len(comparison_engine_meta.get('feature_cols', []))} | rows: {comparison_engine_meta.get('engine_rows')}"
                        )
                        with st.expander("Preview comparison engine panel (feature-enriched)", expanded=False):
                            st.dataframe(comparison_engine_panel_df.head(50), use_container_width=True)
                    else:
                        st.caption(comparison_engine_meta.get("message", "Using base return panel."))

                if comparison_info["missing_assets"]:
                    st.caption(
                        f"Comparison assets not found in the panel: {', '.join(comparison_info['missing_assets'])}"
                    )

                primary_set = set(primary_info["used_assets"])
                comparison_set = set(comparison_info["used_assets"])
                overlap = sorted(primary_set & comparison_set)
                only_primary = sorted(primary_set - comparison_set)
                only_comparison = sorted(comparison_set - primary_set)

                st.markdown("#### Primary vs comparison summary")
                s1, s2, s3 = st.columns(3)
                with s1:
                    st.metric("Overlap assets", len(overlap))
                with s2:
                    st.metric("Primary-only assets", len(only_primary))
                with s3:
                    st.metric("Comparison-only assets", len(only_comparison))

                if overlap:
                    st.caption(f"Overlap: {', '.join(overlap)}")
                if only_primary:
                    st.caption(f"Primary only: {', '.join(only_primary)}")
                if only_comparison:
                    st.caption(f"Comparison only: {', '.join(only_comparison)}")

                comp_kind, comp_text = _large_universe_test_summary(len(comparison_info["used_assets"]))
                badge(f"Comparison universe scale check: {comp_text}", comp_kind)
            else:
                st.session_state["last_comparison_missing_universe_assets"] = []

            st.markdown("### Universe performance comparison")
            st.caption("Optional research baseline: PURE_CS strips overlay / regime / covariance-aware layers while keeping the core cross-sectional allocator.")
            st.checkbox(
                "Run PURE_CS baseline comparison",
                key="universe_run_pure_cs_baseline_compare",
                help="Runs a clean PURE_CS baseline alongside the current engine so you can justify whether the extra research layers add value.",
            )
            st.checkbox(
                "Show full PURE_CS report",
                key="universe_show_pure_cs_report",
                help="If enabled, renders the full detailed report for the PURE_CS baseline run.",
            )
            primary_ready = len(primary_info["used_assets"]) >= 2 and len(primary_engine_panel_df) > 0
            comparison_ready = comparison_panel_df is not None and comparison_info is not None and len(comparison_info["used_assets"]) >= 2 and len(comparison_engine_panel_df) > 0

            if primary_ready:
                cfg = _build_universe_micro_cfg(universe_size=len(primary_info["used_assets"]))
                simple_auto_opt_result = {}
                if bool(st.session_state.get("universe_simple_mode_enabled", True)):
                    simple_resolved_summary = _coerce_mapping(st.session_state.get("universe_simple_resolved_summary", {}))
                    simple_auto_opt_result = _maybe_auto_optimize_simple_config(
                        primary_engine_panel_df,
                        cfg,
                        simple_resolved_summary,
                    )
                    cfg = simple_auto_opt_result.get("final_cfg", cfg)
                    st.session_state["universe_simple_auto_opt_last_result"] = simple_auto_opt_result
                    st.session_state["simple_auto_opt_result"] = simple_auto_opt_result
                    st.session_state["simple_auto_opt_best_payload"] = _coerce_mapping(simple_auto_opt_result.get("best_payload", {}))
                    st.session_state["simple_auto_opt_baseline_payload"] = _coerce_mapping(simple_auto_opt_result.get("baseline_payload", {}))
                    st.session_state["simple_auto_opt_used"] = bool(simple_auto_opt_result.get("was_optimized", False))
                    st.session_state["universe_simple_auto_opt_used"] = bool(simple_auto_opt_result.get("was_optimized", False))
                    try:
                        st.session_state["universe_simple_auto_opt_last_fingerprint"] = config_fingerprint(cfg)
                    except Exception:
                        st.session_state["universe_simple_auto_opt_last_fingerprint"] = ""
                else:
                    st.session_state["universe_simple_auto_opt_last_result"] = {}
                    st.session_state["simple_auto_opt_result"] = {}
                    st.session_state["simple_auto_opt_best_payload"] = {}
                    st.session_state["simple_auto_opt_baseline_payload"] = {}
                    st.session_state["simple_auto_opt_used"] = False
                    st.session_state["universe_simple_auto_opt_used"] = False
                    st.session_state["universe_simple_auto_opt_last_fingerprint"] = ""

                if bool(st.session_state.get("universe_simple_mode_enabled", True)):
                    auto_opt_info = _coerce_mapping(st.session_state.get("universe_simple_auto_opt_last_result", {}))
                    auto_opt_enabled = bool(st.session_state.get("universe_simple_auto_optimize_enabled", False))
                    policy_name = str(auto_opt_info.get("objective_name") or ((st.session_state.get("universe_simple_resolved_summary", {}) or {}).get("resolved_tuning_objective", "—")))
                    trials_explored = int(auto_opt_info.get("trials_explored", 0) or 0)
                    was_optimized = bool(auto_opt_info.get("was_optimized", False))
                    baseline_fp = ""
                    final_fp = ""
                    try:
                        baseline_fp = config_fingerprint(auto_opt_info.get("baseline_cfg", cfg))
                    except Exception:
                        baseline_fp = ""
                    try:
                        final_fp = config_fingerprint(cfg)
                    except Exception:
                        final_fp = ""
                    st.markdown("#### Simple auto-opt summary")
                    a1, a2, a3, a4 = st.columns(4)
                    with a1:
                        st.metric("Auto optimize", "ON" if auto_opt_enabled else "OFF")
                    with a2:
                        st.metric("Optimized config used", "Yes" if was_optimized else "No")
                    with a3:
                        st.metric("Trials explored", trials_explored)
                    with a4:
                        st.metric("Objective", policy_name or "—")
                    if auto_opt_enabled:
                        st.caption(
                            f"Baseline `{baseline_fp[:10] if baseline_fp else '—'}` → Final `{final_fp[:10] if final_fp else '—'}` | "
                            f"selection_policy={auto_opt_info.get('selection_policy', 'fixed_composite_score')} | "
                            f"composite_profile={auto_opt_info.get('composite_profile', 'balanced')}"
                        )
                        changed_params = list(auto_opt_info.get("changed_params", []) or [])
                        baseline_cfg_payload = _coerce_mapping(auto_opt_info.get("baseline_cfg_payload", {}))
                        final_cfg_payload = _coerce_mapping(auto_opt_info.get("final_cfg_payload", {}))
                        if not final_cfg_payload:
                            try:
                                final_cfg_payload = _coerce_mapping(config_to_dict(cfg))
                            except Exception:
                                final_cfg_payload = {}
                        if not baseline_cfg_payload:
                            try:
                                baseline_cfg_payload = _coerce_mapping(config_to_dict(auto_opt_info.get("baseline_cfg", cfg)))
                            except Exception:
                                baseline_cfg_payload = {}
                        if was_optimized and changed_params:
                            st.dataframe(pd.DataFrame(changed_params), use_container_width=True, hide_index=True)
                        else:
                            failure_reason = str(auto_opt_info.get("failure_reason", "") or "")
                            if failure_reason and failure_reason != "auto_opt_off":
                                st.caption(f"Auto-opt fallback to baseline: {failure_reason}")

                        if baseline_cfg_payload or final_cfg_payload:
                            with st.expander("Show auto-opt final config", expanded=False):
                                changed_cfg_payload = {}
                                final_keys = set(final_cfg_payload.keys())
                                baseline_keys = set(baseline_cfg_payload.keys())
                                for key in sorted(final_keys | baseline_keys):
                                    baseline_value = baseline_cfg_payload.get(key)
                                    final_value = final_cfg_payload.get(key)
                                    if baseline_value != final_value:
                                        changed_cfg_payload[key] = {
                                            "baseline": baseline_value,
                                            "final": final_value,
                                        }

                                if changed_cfg_payload:
                                    st.markdown("**Changed config values used by the run**")
                                    st.json(changed_cfg_payload)
                                else:
                                    st.caption("No config delta was detected between baseline and final payloads.")

                                cfg_col1, cfg_col2 = st.columns(2)
                                with cfg_col1:
                                    st.markdown("**Baseline config payload**")
                                    st.json(baseline_cfg_payload)
                                with cfg_col2:
                                    st.markdown("**Final config payload used by the run**")
                                    st.json(final_cfg_payload)


                selected_signal_bundle = None
                selected_signal_mode = str(getattr(cfg, "signal_mode", "mu_sigma"))
                if bool(st.session_state.get("universe_auto_signal_select_enabled", False)):
                    try:
                        candidate_contracts = _build_signal_selection_contracts(
                            cfg,
                            [str(x) for x in st.session_state.get("universe_auto_signal_select_contracts", [])],
                        )
                        selection_runs = {}
                        selection_cfg_lookup = {}
                        for contract in candidate_contracts:
                            cfg_payload = dict(contract.get("cfg_payload", {}))
                            local_cfg = MicroPipelineConfig(**cfg_payload)
                            label = str(contract.get("label"))
                            selection_cfg_lookup[label] = local_cfg
                            selection_runs[label] = run_micro_investment_pipeline(primary_engine_panel_df, cfg=local_cfg)
                        selected_signal_bundle = auto_select_signal_mode_by_rank_ic(selection_runs)
                        selection_table_raw = selected_signal_bundle.get("selection_table", pd.DataFrame())
                        if isinstance(selection_table_raw, pd.DataFrame) and not selection_table_raw.empty:
                            selection_table_raw = _attach_signal_contract_metadata(selection_table_raw, candidate_contracts)
                            selected_signal_bundle["selection_table"] = selection_table_raw
                        selected_signal_mode = str(selected_signal_bundle.get("selected_signal_mode") or selected_signal_mode)
                        st.session_state["universe_auto_selected_signal_mode_last"] = selected_signal_mode
                        primary_run = selection_runs.get(selected_signal_mode)
                        selected_cfg = selection_cfg_lookup.get(selected_signal_mode)
                        if selected_cfg is not None:
                            cfg = selected_cfg
                    except Exception as e:
                        st.warning(f"Could not auto-select signal mode by Rank IC: {e}")
                        primary_run = run_micro_investment_pipeline(primary_engine_panel_df, cfg=cfg)
                else:
                    primary_run = run_micro_investment_pipeline(primary_engine_panel_df, cfg=cfg)

                st.session_state["engine_has_run"] = True

                primary_perf = primary_run.get("performance_summary", {})
                investment_context = _update_investment_context_from_run(primary_run)
                primary_warnings = primary_run.get("universe_warnings", []) or []


                if primary_warnings:
                    st.warning("Primary universe warnings detected:")
                    for w in primary_warnings:
                        st.markdown(f"- {w}")

                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("Primary CAGR", _format_pct_or_dash(primary_perf.get("cagr")))
                with metric_cols[1]:
                    st.metric("Primary Vol", _format_pct_or_dash(primary_perf.get("annual_volatility", primary_perf.get("annualized_volatility"))))
                with metric_cols[2]:
                    st.metric("Primary Sharpe", _format_num_or_dash(primary_perf.get("sharpe")))
                with metric_cols[3]:
                    st.metric("Primary MaxDD", _format_pct_or_dash(primary_perf.get("max_drawdown")))

                primary_prob_mode = str(getattr(cfg, "probabilistic_mode", "none") or "none")
                primary_prob_audit_df = pd.DataFrame()
                if primary_prob_mode != "none":
                    primary_prob_audit_df = _render_probabilistic_mode_audit(
                        primary_run,
                        title="Probabilistic mode audit",
                        key_prefix="primary",
                    )

                if selected_signal_bundle is not None:
                    st.markdown("#### Automatic signal-contract selection by Rank IC")
                    selection_table = _coerce_numeric_columns(
                        selected_signal_bundle.get("selection_table", pd.DataFrame()),
                        skip={"signal_mode", "signal_col_used", "contract_label", "signal_mode_requested_contract", "signal_mode_requested", "signal_mode_effective"},
                    )
                    a1, a2, a3, a4 = st.columns(4)
                    with a1:
                        st.metric("Selected contract", _format_signal_contract_label(selected_signal_mode))
                    with a2:
                        d = pd.to_numeric(selection_table.loc[selection_table["signal_mode"] == selected_signal_mode, "rank_ic_mean"], errors="coerce") if isinstance(selection_table, pd.DataFrame) and not selection_table.empty else pd.Series(dtype="float64")
                        st.metric("Selected Rank IC", _format_num_or_dash(d.iloc[0] if not d.empty else np.nan))
                    with a3:
                        d = pd.to_numeric(selection_table.loc[selection_table["signal_mode"] == selected_signal_mode, "rank_ic_ir"], errors="coerce") if isinstance(selection_table, pd.DataFrame) and not selection_table.empty else pd.Series(dtype="float64")
                        st.metric("Selected IC IR", _format_num_or_dash(d.iloc[0] if not d.empty else np.nan))
                    with a4:
                        d = pd.to_numeric(selection_table.loc[selection_table["signal_mode"] == selected_signal_mode, "sharpe"], errors="coerce") if isinstance(selection_table, pd.DataFrame) and not selection_table.empty else pd.Series(dtype="float64")
                        st.metric("Selected Sharpe", _format_num_or_dash(d.iloc[0] if not d.empty else np.nan))
                    if isinstance(selection_table, pd.DataFrame) and not selection_table.empty:
                        st.caption("Selection order: Rank IC mean → Rank IC IR → Sharpe → CAGR. Route-B exploration evaluates requested signal contracts and then reports the realised effective mode for each contract.")
                        focus_cols = [
                            c for c in [
                                "contract_label",
                                "signal_mode_requested_contract",
                                "contract_multi_loss_enabled",
                                "contract_low_signal_fallback_to_ew",
                                "signal_mode_requested",
                                "signal_mode_effective",
                                "signal_col_used",
                                "rank_ic_mean",
                                "rank_ic_ir",
                                "rank_ic_hit_rate",
                                "rank_ic_t_stat",
                                "n_dates",
                                "sharpe",
                                "cagr",
                                "max_drawdown",
                                "mean_turnover",
                            ] if c in selection_table.columns
                        ]
                        st.dataframe(selection_table[focus_cols], use_container_width=True)
                        if bool(st.session_state.get("universe_auto_signal_select_show_report", False)):
                            with st.expander("Detailed automatic signal-contract selection report", expanded=False):
                                st.dataframe(selection_table, use_container_width=True)
                                _download_dataframe_button(
                                    "Download auto signal-contract selection table",
                                    selection_table,
                                    "auto_signal_contract_selection.csv",
                                    key="auto_signal_mode_selection_download",
                                )

                overlay_bundle = None
                overlay_none_run = None
                if (
                    str(getattr(cfg, "probabilistic_mode", "none")) != "none"
                    and bool(st.session_state.get("universe_overlay_compare_enabled", True))
                ):
                    try:
                        overlay_cfg_payload = config_to_dict(cfg)
                        overlay_cfg_payload["probabilistic_mode"] = "none"
                        if "probabilistic_overlay_strength" in overlay_cfg_payload:
                            overlay_cfg_payload["probabilistic_overlay_strength"] = 0.0
                        if "probabilistic_overlay_blend" in overlay_cfg_payload:
                            overlay_cfg_payload["probabilistic_overlay_blend"] = 0.0
                        overlay_none_cfg = MicroPipelineConfig(**{k: v for k, v in overlay_cfg_payload.items() if k in {f.name for f in fields(MicroPipelineConfig)}})
                        overlay_none_run = run_micro_investment_pipeline(primary_engine_panel_df, cfg=overlay_none_cfg)
                        overlay_bundle = compare_probabilistic_overlay_runs(
                            overlay_none_run,
                            primary_run,
                            base_name="none",
                            overlay_name=str(getattr(cfg, "probabilistic_mode", "overlay")),
                        )

                        st.markdown("#### Overlay evaluated economically")
                        decision_summary = _coerce_mapping(overlay_bundle.get("decision_summary", {}))
                        decision_direction = _infer_overlay_direction_from_bundle(overlay_bundle)
                        decision_headline = str(decision_summary.get("headline", "Overlay vs none economic validation."))

                        outcome_label_map = {
                            "better": "Beneficial",
                            "worse": "Adverse",
                            "mixed": "Mixed / neutral",
                            "neutral": "Mixed / neutral",
                            "unknown": "Mixed / neutral",
                        }
                        outcome_label = outcome_label_map.get(str(decision_direction), "Mixed / neutral")
                        st.markdown(f"**Economic outcome:** {outcome_label}")
                        st.caption(decision_headline)

                        overlay_validation_summary = _prepare_overlay_validation_summary_table(
                            overlay_bundle.get("overlay_validation_summary", pd.DataFrame())
                        )
                        overlay_validation_warnings = _prepare_overlay_validation_warnings(
                            overlay_bundle.get("overlay_validation_warnings", [])
                        )
                        base_validation = overlay_bundle.get("overlay_validation_base", {}) or {}
                        overlay_validation = overlay_bundle.get("overlay_validation_overlay", {}) or {}
                        base_validation_status = str(
                            decision_summary.get("base_validation_status", base_validation.get("status", "warn"))
                        )
                        overlay_validation_status = str(
                            decision_summary.get("overlay_validation_status", overlay_validation.get("status", "warn"))
                        )

                        st.markdown("**Mode-aware diagnostics validation**")
                        v1, v2, v3, v4 = st.columns(4)
                        with v1:
                            st.markdown(f"**No-overlay diagnostics:** {_overlay_validation_label(base_validation_status)}")
                        with v2:
                            st.markdown(f"**Overlay diagnostics:** {_overlay_validation_label(overlay_validation_status)}")
                        with v3:
                            st.metric(
                                "Validation notes",
                                int(decision_summary.get("validation_warning_count", len(overlay_validation_warnings)) or 0),
                            )
                        with v4:
                            st.metric(
                                "Overlay missing required",
                                int(len(overlay_validation.get("missing_required_cols", [])) or 0),
                            )

                        coverage_text = _overlay_validation_text(overlay_validation_status)
                        if overlay_validation_status == "ok" and base_validation_status == "ok":
                            st.caption(coverage_text)
                        else:
                            st.markdown(coverage_text)

                        if overlay_validation_warnings:
                            with st.expander("Overlay diagnostics notes", expanded=(overlay_validation_status == "fail")):
                                for line in overlay_validation_warnings:
                                    st.markdown(f"- {line}")

                        if isinstance(overlay_validation_summary, pd.DataFrame) and not overlay_validation_summary.empty:
                            with st.expander("Overlay diagnostics validation summary", expanded=False):
                                st.dataframe(overlay_validation_summary, use_container_width=True, hide_index=True)

                        overlay_mode_audit_summary = _build_overlay_mode_audit_summary(
                            overlay_none_run,
                            primary_run,
                            base_name="none",
                            overlay_name=str(getattr(cfg, "probabilistic_mode", "overlay")),
                        )
                        if isinstance(overlay_mode_audit_summary, pd.DataFrame) and not overlay_mode_audit_summary.empty:
                            st.markdown("**Mode-aware diagnostics validation — contract vs reality**")
                            audit_cols = st.columns(3)
                            overlay_audit_row = overlay_mode_audit_summary.loc[
                                overlay_mode_audit_summary["run_name"].astype(str) == str(getattr(cfg, "probabilistic_mode", "overlay"))
                            ] if "run_name" in overlay_mode_audit_summary.columns else pd.DataFrame()
                            overlay_audit_row = _row_to_dict(overlay_audit_row.iloc[0]) if not overlay_audit_row.empty else {}
                            with audit_cols[0]:
                                st.markdown(
                                    f"**No-overlay contract check:** {_overlay_validation_label(base_validation.get('status', 'warn'))}"
                                )
                            with audit_cols[1]:
                                st.markdown(
                                    f"**Overlay contract check:** {_overlay_validation_label(overlay_audit_row.get('semantic_status', overlay_validation_status))}"
                                )
                            with audit_cols[2]:
                                st.metric("Overlay fallback", "Yes" if bool(overlay_audit_row.get("prob_fallback_triggered", False)) else "No")

                            st.dataframe(overlay_mode_audit_summary, use_container_width=True, hide_index=True)
                            _download_dataframe_button(
                                "Download mode-aware diagnostics validation audit",
                                overlay_mode_audit_summary,
                                "overlay_mode_aware_diagnostics_validation.csv",
                                key="overlay_mode_aware_diagnostics_validation_download",
                            )

                            economic_delta = _coerce_mapping(
                                overlay_bundle.get("economic_delta", {})
                            )

                        od1, od2, od3, od4 = st.columns(4)
                        with od1:
                            st.metric("Δ Sharpe", _format_num_or_dash(economic_delta.get("delta_sharpe")))
                        with od2:
                            st.metric("Δ CAGR", _format_delta_pct_or_dash(economic_delta.get("delta_cagr")))
                        with od3:
                            st.metric("Δ MaxDD", _format_delta_pct_or_dash(economic_delta.get("delta_max_drawdown")))
                        with od4:
                            st.metric("Δ Turnover", _format_delta_pct_or_dash(economic_delta.get("delta_turnover")))


                        overlay_focus = _coerce_numeric_columns(overlay_bundle.get("overlay_focus", pd.DataFrame()), skip={"metric", "better_when", "economic_note", "direction"})
                        overlay_cmp = _coerce_numeric_columns(overlay_bundle.get("overlay_comparison", pd.DataFrame()), skip={"metric", "better_when", "economic_note", "direction"})
                        overlay_base = _prepare_overlay_summary_table(_coerce_numeric_columns(overlay_bundle.get("overlay_summary_base", pd.DataFrame()), skip={"name", "metric"}))
                        overlay_on = _prepare_overlay_summary_table(_coerce_numeric_columns(overlay_bundle.get("overlay_summary_overlay", pd.DataFrame()), skip={"name", "metric"}))
                        overlay_metric_base = _prepare_overlay_summary_table(_coerce_numeric_columns(overlay_bundle.get("overlay_metric_summary_base", pd.DataFrame()), skip={"metric"}))
                        overlay_metric_on = _prepare_overlay_summary_table(_coerce_numeric_columns(overlay_bundle.get("overlay_metric_summary_overlay", pd.DataFrame()), skip={"metric"}))
                        overlay_metric_lookup = _overlay_metric_lookup_from_summary(overlay_metric_on)
                        overlay_metric_lookup_base = _overlay_metric_lookup_from_summary(overlay_metric_base)

                        if bool(st.session_state.get("universe_overlay_report_show_summary", True)):
                            st.markdown("**No-overlay vs overlay at a glance**")
                            s1, s2, s3, s4 = st.columns(4)
                            base_lookup = {}
                            on_lookup = {}
                            if isinstance(overlay_base, pd.DataFrame) and not overlay_base.empty and {"metric", "value"}.issubset(overlay_base.columns):
                                base_lookup = dict(zip(overlay_base["metric"].astype(str), overlay_base["value"]))
                            if isinstance(overlay_on, pd.DataFrame) and not overlay_on.empty and {"metric", "value"}.issubset(overlay_on.columns):
                                on_lookup = dict(zip(overlay_on["metric"].astype(str), overlay_on["value"]))

                            with s1:
                                st.metric("Sharpe", _format_num_or_dash(on_lookup.get("sharpe")), delta=_format_num_or_dash(economic_delta.get("delta_sharpe")))
                            with s2:
                                st.metric("CAGR", _format_pct_or_dash(on_lookup.get("cagr")), delta=_format_delta_pct_or_dash(economic_delta.get("delta_cagr")))
                            with s3:
                                st.metric("MaxDD", _format_pct_or_dash(on_lookup.get("max_drawdown")), delta=_format_delta_pct_or_dash(economic_delta.get("delta_max_drawdown")))
                            with s4:
                                st.metric("Turnover", _format_pct_or_dash(on_lookup.get("mean_turnover")), delta=_format_delta_pct_or_dash(economic_delta.get("delta_turnover")))

                            summary_compare_rows = []
                            for metric, label, formatter in [
                                ("sharpe", "Sharpe", _format_num_or_dash),
                                ("cagr", "CAGR", _format_pct_or_dash),
                                ("max_drawdown", "MaxDD", _format_pct_or_dash),
                                ("mean_turnover", "Turnover", _format_pct_or_dash),
                            ]:
                                summary_compare_rows.append({
                                    "metric": label,
                                    "none": formatter(base_lookup.get(metric)),
                                    "overlay": formatter(on_lookup.get(metric)),
                                })
                            st.dataframe(pd.DataFrame(summary_compare_rows), use_container_width=True, hide_index=True)

                            rich_focus_cards = _build_overlay_rich_focus_cards(overlay_metric_lookup)
                            if rich_focus_cards:
                                st.markdown("**Overlay telemetry highlights**")
                                rich_cols = st.columns(min(4, len(rich_focus_cards)))
                                for idx, card in enumerate(rich_focus_cards):
                                    with rich_cols[idx % len(rich_cols)]:
                                        st.metric(card.get("label", "metric"), card.get("value", "—"), delta=card.get("delta"), help=card.get("help"))

                            regime_summary_available = any(
                                metric in overlay_metric_lookup
                                for metric in [
                                    "regime_universe_enabled_rate",
                                    "regime_universe_selected_assets_mean",
                                    "regime_universe_filtered_assets_mean",
                                    "regime_universe_regime_mode",
                                    "regime_universe_selection_metric_mode",
                                ]
                            )
                            if regime_summary_available:
                                st.markdown("**Regime-dependent universe summary**")
                                ru1, ru2, ru3, ru4 = st.columns(4)
                                with ru1:
                                    st.metric(
                                        "Enabled rate",
                                        _format_pct_or_dash(_overlay_metric_float(overlay_metric_lookup, "regime_universe_enabled_rate")),
                                    )
                                with ru2:
                                    st.metric(
                                        "Selected assets",
                                        _format_num_or_dash(_overlay_metric_float(overlay_metric_lookup, "regime_universe_selected_assets_mean")),
                                    )
                                with ru3:
                                    st.metric(
                                        "Filtered assets",
                                        _format_num_or_dash(_overlay_metric_float(overlay_metric_lookup, "regime_universe_filtered_assets_mean")),
                                    )
                                with ru4:
                                    st.metric(
                                        "Realised regime",
                                        _overlay_metric_text(overlay_metric_lookup, "regime_universe_regime_mode"),
                                    )
                                regime_detail_rows = []
                                for metric, label, formatter in [
                                    ("regime_universe_keep_frac_mean", "Keep fraction", _format_pct_or_dash),
                                    ("regime_universe_selection_metric_mode", "Selection metric", lambda x: _overlay_metric_text(overlay_metric_lookup, metric)),
                                    ("regime_universe_low_offensive_vol_tilt_mean", "Low offensive vol tilt", _format_num_or_dash),
                                    ("regime_universe_high_defensive_vol_tilt_mean", "High defensive vol tilt", _format_num_or_dash),
                                ]:
                                    if metric == "regime_universe_selection_metric_mode":
                                        value = formatter(None)
                                    else:
                                        value = formatter(_overlay_metric_float(overlay_metric_lookup, metric))
                                    if str(value) != "—":
                                        regime_detail_rows.append({"metric": label, "overlay": value})
                                if regime_detail_rows:
                                    st.dataframe(pd.DataFrame(regime_detail_rows), use_container_width=True, hide_index=True)

                            dispersion_summary_available = any(
                                metric in overlay_metric_lookup
                                for metric in [
                                    "dispersion_risk_model_active_rate",
                                    "dispersion_risk_model_mult_mean",
                                    "dispersion_risk_model_apply_target_mode",
                                    "dispersion_risk_model_reason_mode",
                                ]
                            )
                            if dispersion_summary_available:
                                st.markdown("**Dispersion risk model summary**")
                                dr1, dr2, dr3, dr4 = st.columns(4)
                                with dr1:
                                    st.metric(
                                        "Active rate",
                                        _format_pct_or_dash(overlay_metric_lookup.get("dispersion_risk_model_active_rate")),
                                    )
                                with dr2:
                                    st.metric(
                                        "Mean multiplier",
                                        _format_num_or_dash(overlay_metric_lookup.get("dispersion_risk_model_mult_mean")),
                                    )
                                with dr3:
                                    st.metric(
                                        "Target mode",
                                        str(overlay_metric_lookup.get("dispersion_risk_model_apply_target_mode", "—")),
                                    )
                                with dr4:
                                    st.metric(
                                        "Reason mode",
                                        str(overlay_metric_lookup.get("dispersion_risk_model_reason_mode", "—")),
                                    )

                        if isinstance(overlay_focus, pd.DataFrame) and not overlay_focus.empty:
                            st.markdown("**Economic interpretation by metric**")
                            st.dataframe(overlay_focus, use_container_width=True)

                        st.markdown("**Overlay interpretation for report / coursework**")
                        for line in _build_overlay_interpretation_lines(overlay_bundle):
                            st.markdown(f"- {line}")
                        overlay_coursework_text = _build_overlay_coursework_paragraph(overlay_bundle)
                        if overlay_coursework_text:
                            with st.expander("Coursework-ready overlay paragraph", expanded=False):
                                st.text_area(
                                    "Ready-to-use interpretation",
                                    value=overlay_coursework_text,
                                    height=220,
                                    key="overlay_coursework_paragraph_text",
                                )
                                st.download_button(
                                    label="Download overlay interpretation",
                                    data=overlay_coursework_text.encode("utf-8"),
                                    file_name="overlay_interpretation.txt",
                                    mime="text/plain",
                                    key="overlay_interpretation_download",
                                )

                        if bool(st.session_state.get("universe_overlay_report_show_internal_metrics", True)):
                            internal_df = _prepare_overlay_internal_metrics_table(overlay_metric_base, overlay_metric_on)
                            if isinstance(internal_df, pd.DataFrame) and not internal_df.empty:
                                st.markdown("**Overlay internals**")
                                st.caption("Internal metrics are reordered so the semantic contract comes first: backend/source, feature-conditioned μ, regime-dependent universe, then governance and dispersion controls.")
                                st.dataframe(internal_df, use_container_width=True, hide_index=True)

                        if bool(st.session_state.get("universe_overlay_compare_show_detail", False)):
                            with st.expander("Detailed overlay vs none report", expanded=False):
                                if isinstance(overlay_base, pd.DataFrame) and not overlay_base.empty:
                                    st.markdown("**No-overlay summary**")
                                    st.dataframe(overlay_base, use_container_width=True, hide_index=True)
                                if isinstance(overlay_on, pd.DataFrame) and not overlay_on.empty:
                                    st.markdown("**Overlay summary**")
                                    st.dataframe(overlay_on, use_container_width=True, hide_index=True)
                                if isinstance(overlay_cmp, pd.DataFrame) and not overlay_cmp.empty:
                                    st.markdown("**Metric-by-metric comparison**")
                                    st.dataframe(overlay_cmp, use_container_width=True)
                                    _download_dataframe_button(
                                        "Download overlay vs none comparison",
                                        overlay_cmp,
                                        "overlay_vs_none_comparison.csv",
                                        key="overlay_vs_none_download",
                                    )
                                if bool(st.session_state.get("universe_overlay_report_show_diagnostics", False)):
                                    overlay_diag_base = _coerce_numeric_columns(overlay_none_run.get("diagnostics_df", pd.DataFrame()) if isinstance(overlay_none_run, dict) else pd.DataFrame())
                                    overlay_diag_on = _coerce_numeric_columns(primary_run.get("diagnostics_df", pd.DataFrame()) if isinstance(primary_run, dict) else pd.DataFrame())
                                    if isinstance(overlay_diag_base, pd.DataFrame) and not overlay_diag_base.empty:
                                        st.markdown("**No-overlay diagnostics preview**")
                                        st.dataframe(overlay_diag_base.head(50), use_container_width=True)
                                    if isinstance(overlay_diag_on, pd.DataFrame) and not overlay_diag_on.empty:
                                        st.markdown("**Overlay diagnostics preview**")
                                        st.dataframe(overlay_diag_on.head(50), use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not evaluate overlay economically vs none: {e}")

                cost_bundle = None
                cost_none_run = None
                if bool(getattr(cfg, "cost_model_enabled", False)) and bool(st.session_state.get("universe_cost_model_compare_enabled", True)):
                    try:
                        cost_cfg_payload = config_to_dict(cfg)
                        for k, v in {
                            "cost_model_enabled": False,
                            "transaction_cost_commission_bps": 0.0,
                            "transaction_cost_slippage_bps": 0.0,
                            "transaction_cost_spread_bps": 0.0,
                            "transaction_cost_market_impact_bps": 0.0,
                            "holding_cost_annual_bps": 0.0,
                            "tax_model_enabled": False,
                            "tax_short_term_rate": 0.0,
                            "tax_long_term_rate": 0.0,
                            "tax_apply_loss_credit": False,
                            "tax_loss_credit_rate": 0.0,
                        }.items():
                            if k in {f.name for f in fields(MicroPipelineConfig)}:
                                cost_cfg_payload[k] = v
                        cost_none_cfg = MicroPipelineConfig(**{k: v for k, v in cost_cfg_payload.items() if k in {f.name for f in fields(MicroPipelineConfig)}})
                        cost_none_run = run_micro_investment_pipeline(primary_engine_panel_df, cfg=cost_none_cfg)
                        cost_bundle = compare_cost_model_runs(cost_none_run, primary_run, baseline_name="gross/no-cost", cost_name="net/cost-aware")

                        st.markdown("#### Full cost model evaluated economically")
                        cost_comparison = _coerce_numeric_columns(cost_bundle.get("cost_comparison", pd.DataFrame()), skip={"metric", "better_when", "direction", "headline"})
                        cost_focus = _coerce_numeric_columns(cost_bundle.get("cost_focus", pd.DataFrame()), skip={"metric", "better_when", "direction", "headline"})
                        cost_summary_base = _prepare_cost_summary_table(_coerce_numeric_columns(cost_bundle.get("cost_summary_baseline", pd.DataFrame()), skip={"metric"}))
                        cost_summary_on = _prepare_cost_summary_table(_coerce_numeric_columns(cost_bundle.get("cost_summary_cost_run", pd.DataFrame()), skip={"metric"}))
                        cost_ts = _coerce_numeric_columns(cost_bundle.get("cost_drag_timeseries", pd.DataFrame()))
                        decision_summary = cost_bundle.get("decision_summary", {}) or {}
                        st.caption(str(decision_summary.get("headline", "Gross/no-cost vs net/cost-aware validation.")))

                        cmp_lookup = {}
                        if isinstance(cost_comparison, pd.DataFrame) and not cost_comparison.empty and {"metric", "delta"}.issubset(cost_comparison.columns):
                            cmp_lookup = dict(zip(cost_comparison["metric"].astype(str), pd.to_numeric(cost_comparison["delta"], errors="coerce")))
                        on_lookup = {}
                        if isinstance(cost_summary_on, pd.DataFrame) and not cost_summary_on.empty and {"metric", "value"}.issubset(cost_summary_on.columns):
                            on_lookup = dict(zip(cost_summary_on["metric"].astype(str), cost_summary_on["value"]))

                        if bool(st.session_state.get("universe_cost_report_show_summary", True)):
                            c1, c2, c3, c4 = st.columns(4)
                            with c1:
                                st.metric("Net Sharpe", _format_num_or_dash(on_lookup.get("net_sharpe")), delta=_format_num_or_dash(cmp_lookup.get("net_sharpe")))
                            with c2:
                                st.metric("Net CAGR", _format_pct_or_dash(on_lookup.get("net_cagr")), delta=_format_delta_pct_or_dash(cmp_lookup.get("net_cagr")))
                            with c3:
                                st.metric("Mean total cost", _format_pct_or_dash(on_lookup.get("mean_total_cost_rate")))
                            with c4:
                                st.metric("Mean cost drag", _format_bps_or_dash(on_lookup.get("mean_cost_drag_bps")))
                            focus_rows=[]
                            for metric,label,formatter in [
                                ("gross_cagr","Gross CAGR",_format_pct_or_dash),
                                ("net_cagr","Net CAGR",_format_pct_or_dash),
                                ("gross_sharpe","Gross Sharpe",_format_num_or_dash),
                                ("net_sharpe","Net Sharpe",_format_num_or_dash),
                                ("mean_total_cost_rate","Mean total cost rate",_format_pct_or_dash),
                                ("mean_cost_drag_bps","Mean cost drag",_format_bps_or_dash),
                                ("mean_transaction_cost_rate","Mean transaction cost",_format_pct_or_dash),
                                ("mean_tax_cost_rate","Mean tax cost",_format_pct_or_dash),
                                ("mean_holding_cost_rate","Mean holding cost",_format_pct_or_dash),
                            ]:
                                focus_rows.append({"metric": label, "value": formatter(on_lookup.get(metric))})
                            st.dataframe(pd.DataFrame(focus_rows), use_container_width=True, hide_index=True)

                        if isinstance(cost_focus, pd.DataFrame) and not cost_focus.empty:
                            st.markdown("**Economic cost interpretation by metric**")
                            st.dataframe(cost_focus, use_container_width=True)

                        if isinstance(cost_ts, pd.DataFrame) and not cost_ts.empty:
                            if bool(st.session_state.get("universe_cost_report_show_nav_chart", True)) and {"cumulative_gross_nav", "cumulative_net_nav"}.issubset(cost_ts.columns):
                                st.markdown("**Gross vs net NAV path**")
                                nav_plot_df = cost_ts[[c for c in ["date", "cumulative_gross_nav", "cumulative_net_nav", "cumulative_cost_gap_pct"] if c in cost_ts.columns]].copy()
                                if "date" in nav_plot_df.columns:
                                    nav_plot_df = nav_plot_df.sort_values("date").set_index("date")
                                st.line_chart(nav_plot_df[[c for c in ["cumulative_gross_nav", "cumulative_net_nav"] if c in nav_plot_df.columns]], use_container_width=True)
                                nav_last = nav_plot_df.tail(1)
                                if not nav_last.empty:
                                    nv1, nv2, nv3 = st.columns(3)
                                    last_row = nav_last.iloc[0]
                                    with nv1:
                                        st.metric("Final gross NAV", _format_num_or_dash(last_row.get("cumulative_gross_nav")))
                                    with nv2:
                                        st.metric("Final net NAV", _format_num_or_dash(last_row.get("cumulative_net_nav")))
                                    with nv3:
                                        st.metric("Final cost gap", _format_delta_pct_or_dash(last_row.get("cumulative_cost_gap_pct")))

                            if bool(st.session_state.get("universe_cost_report_show_timeseries", True)):
                                st.markdown("**Cost drag timeseries**")
                                preview_cols = [c for c in ["date", "gross_portfolio_return_simple", "net_portfolio_return_simple", "total_cost_rate", "cost_drag_bps", "turnover_buy", "turnover_sell", "cumulative_gross_nav", "cumulative_net_nav", "cumulative_cost_gap_pct"] if c in cost_ts.columns]
                                if preview_cols:
                                    preview_df = cost_ts[preview_cols].copy()
                                    st.dataframe(preview_df, use_container_width=True, hide_index=True)
                                _download_dataframe_button(
                                    "Download cost drag timeseries",
                                    cost_ts,
                                    "cost_drag_timeseries.csv",
                                    key="cost_drag_timeseries_download",
                                )

                        if bool(st.session_state.get("universe_cost_report_show_detail", False)):
                            with st.expander("Detailed full cost model report", expanded=False):
                                if isinstance(cost_summary_base, pd.DataFrame) and not cost_summary_base.empty:
                                    st.markdown("**Gross / no-cost summary**")
                                    st.dataframe(cost_summary_base, use_container_width=True, hide_index=True)
                                if isinstance(cost_summary_on, pd.DataFrame) and not cost_summary_on.empty:
                                    st.markdown("**Net / cost-aware summary**")
                                    st.dataframe(cost_summary_on, use_container_width=True, hide_index=True)
                                if isinstance(cost_comparison, pd.DataFrame) and not cost_comparison.empty:
                                    st.markdown("**Metric-by-metric cost comparison**")
                                    st.dataframe(cost_comparison, use_container_width=True)
                                    _download_dataframe_button(
                                        "Download cost comparison",
                                        cost_comparison,
                                        "cost_model_comparison.csv",
                                        key="cost_model_comparison_download",
                                    )
                                if bool(st.session_state.get("universe_cost_report_show_diagnostics", False)):
                                    cost_diag_base = _coerce_numeric_columns(cost_none_run.get("diagnostics_df", pd.DataFrame()) if isinstance(cost_none_run, dict) else pd.DataFrame())
                                    cost_diag_on = _coerce_numeric_columns(primary_run.get("diagnostics_df", pd.DataFrame()) if isinstance(primary_run, dict) else pd.DataFrame())
                                    if isinstance(cost_diag_base, pd.DataFrame) and not cost_diag_base.empty:
                                        st.markdown("**Gross / no-cost diagnostics preview**")
                                        st.dataframe(cost_diag_base.head(50), use_container_width=True)
                                    if isinstance(cost_diag_on, pd.DataFrame) and not cost_diag_on.empty:
                                        st.markdown("**Net / cost-aware diagnostics preview**")
                                        st.dataframe(cost_diag_on.head(50), use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not evaluate full cost model economically vs no-cost baseline: {e}")

                lambdarank_bundle = None
                lambdarank_baseline_run = None
                if str(getattr(cfg, "signal_mode", "mu_sigma")) == "lambdarank_real" or bool(getattr(cfg, "multi_loss_enabled", False)):
                    try:
                        ranked_cfg_payload = config_to_dict(cfg)
                        base_cfg_payload = dict(ranked_cfg_payload)
                        if "signal_mode" in {f.name for f in fields(MicroPipelineConfig)}:
                            base_cfg_payload["signal_mode"] = "mu_sigma"
                        if "multi_loss_enabled" in {f.name for f in fields(MicroPipelineConfig)}:
                            base_cfg_payload["multi_loss_enabled"] = False
                        lambdarank_baseline_cfg = MicroPipelineConfig(**{k: v for k, v in base_cfg_payload.items() if k in {f.name for f in fields(MicroPipelineConfig)}})
                        lambdarank_baseline_run = run_micro_investment_pipeline(primary_engine_panel_df, cfg=lambdarank_baseline_cfg)
                        lambdarank_bundle = compare_lambdarank_multiloss_runs(
                            lambdarank_baseline_run,
                            primary_run,
                            baseline_name="mu_sigma baseline",
                            ranked_name="lambdarank/multi-loss",
                        )

                        st.markdown("#### LambdaRank real / multi-loss validation")
                        lr_summary = _coerce_numeric_columns(lambdarank_bundle.get("lambdarank_summary_ranked_run", pd.DataFrame()), skip={"metric"})
                        lr_focus = _coerce_numeric_columns(lambdarank_bundle.get("lambdarank_focus", pd.DataFrame()), skip={"metric", "better_when", "direction"})
                        lr_cmp = _coerce_numeric_columns(lambdarank_bundle.get("lambdarank_comparison", pd.DataFrame()), skip={"metric", "better_when", "direction"})
                        lr_ts = _coerce_numeric_columns(lambdarank_bundle.get("lambdarank_timeseries", pd.DataFrame()))
                        lr_decision = lambdarank_bundle.get("decision_summary", {}) or {}
                        st.caption(str(lr_decision.get("headline", "LambdaRank / multi-loss validation vs baseline.")))

                        lr_lookup = {}
                        if isinstance(lr_summary, pd.DataFrame) and not lr_summary.empty and {"metric", "value"}.issubset(lr_summary.columns):
                            lr_lookup = dict(zip(lr_summary["metric"].astype(str), lr_summary["value"]))

                        l1, l2, l3, l4 = st.columns(4)
                        with l1:
                            st.metric("Signal mode", str(lr_lookup.get("signal_mode", getattr(cfg, "signal_mode", "—"))))
                        with l2:
                            st.metric("Rank IC mean", _format_num_or_dash(lr_lookup.get("signal_rank_ic_mean")))
                        with l3:
                            st.metric("Rank IC IR", _format_num_or_dash(lr_lookup.get("signal_rank_ic_ir")))
                        with l4:
                            st.metric("Multi-loss active", "Yes" if bool(lr_lookup.get("multi_loss_active", False)) else "No")

                        focus_rows = []
                        for metric, label, formatter in [
                            ("cagr", "CAGR", _format_pct_or_dash),
                            ("annual_volatility", "Volatility", _format_pct_or_dash),
                            ("sharpe", "Sharpe", _format_num_or_dash),
                            ("max_drawdown", "Max drawdown", _format_pct_or_dash),
                            ("mean_turnover", "Mean turnover", _format_num_or_dash),
                            ("information_ratio", "Information ratio", _format_num_or_dash),
                            ("multi_loss_score_std", "Multi-loss score std", _format_num_or_dash),
                        ]:
                            focus_rows.append({"metric": label, "value": formatter(lr_lookup.get(metric))})
                        st.dataframe(pd.DataFrame(focus_rows), use_container_width=True, hide_index=True)

                        if isinstance(lr_focus, pd.DataFrame) and not lr_focus.empty:
                            st.markdown("**LambdaRank / multi-loss comparison focus**")
                            st.dataframe(lr_focus, use_container_width=True)

                        if isinstance(lr_ts, pd.DataFrame) and not lr_ts.empty:
                            plot_cols = [c for c in ["cumulative_gross_nav", "cumulative_net_nav"] if c in lr_ts.columns]
                            if plot_cols:
                                st.markdown("**LambdaRank / multi-loss NAV path**")
                                plot_df = lr_ts[[c for c in ["date"] + plot_cols if c in lr_ts.columns]].copy()
                                if "date" in plot_df.columns:
                                    plot_df = plot_df.sort_values("date").set_index("date")
                                st.line_chart(plot_df[plot_cols], use_container_width=True)
                            with st.expander("LambdaRank / multi-loss diagnostics / downloads", expanded=False):
                                st.dataframe(lr_ts, use_container_width=True)
                                _download_dataframe_button(
                                    "Download LambdaRank / multi-loss timeseries",
                                    lr_ts,
                                    "lambdarank_multiloss_timeseries.csv",
                                    key="lambdarank_multiloss_timeseries_download",
                                )
                                if isinstance(lr_cmp, pd.DataFrame) and not lr_cmp.empty:
                                    st.markdown("**Metric-by-metric LambdaRank comparison**")
                                    st.dataframe(lr_cmp, use_container_width=True)
                                    _download_dataframe_button(
                                        "Download LambdaRank / multi-loss comparison",
                                        lr_cmp,
                                        "lambdarank_multiloss_comparison.csv",
                                        key="lambdarank_multiloss_comparison_download",
                                    )
                    except Exception as e:
                        st.warning(f"Could not evaluate LambdaRank real / multi-loss formally: {e}")


                _render_run_report("Primary", primary_run, prefix="primary")

                if bool(st.session_state.get("universe_simple_mode_enabled", True)):
                    st.markdown("#### Refinement tools")
                    st.info("Simple mode keeps auto-opt in the setup layer. Manual/refinement research tools stay below as optional post-run exploration, while the recommendation assistants remain in the Improve this setup section.")
                else:
                    st.markdown("#### Refinement tools")
                    st.caption("Diagnostics come first; refinement tools then explore small, local improvements around the current resolved config without sending you back into the low-level manual controls.")

                    with st.container(border=True):
                        st.caption("Local preset refinement")
                        st.checkbox(
                            "Enable local auto-search around current config",
                            key="universe_local_search_enabled",
                            help="Runs a compact local search around the current resolved config. It keeps the neighbourhood small so the UX stays fast and safe.",
                        )
                        ls1, ls2, ls3 = st.columns(3)
                        with ls1:
                            st.selectbox(
                                "local search objective",
                                TUNING_OBJECTIVE_OPTIONS,
                                key="universe_local_search_objective",
                                format_func=_format_tuning_objective_label,
                            )
                        with ls2:
                            st.selectbox(
                                "local search intensity",
                                ["Quick", "Standard"],
                                key="universe_local_search_intensity",
                                help="Quick = smaller neighbourhood. Standard = slightly wider local sweep.",
                            )
                        with ls3:
                            st.checkbox(
                                "two-stage local refinement",
                                key="universe_local_search_two_stage",
                                help="Runs a second micro-refinement around the best stage-1 candidate using the same local-search engine.",
                            )

                        ls4, ls5 = st.columns(2)
                        with ls4:
                            st.checkbox(
                                "include weight_shrink in local search",
                                key="universe_local_search_include_weight_shrink",
                                help="Keeps the local search focused by default. Enable this to also perturb weight_shrink around the current config.",
                            )
                        with ls5:
                            st.checkbox(
                                "Show detailed local-search table",
                                key="universe_local_search_show_detail",
                            )

                        if bool(st.session_state.get("universe_local_search_enabled", False)):
                            try:
                                simple_resolved_summary = st.session_state.get("universe_simple_resolved_summary", {}) if bool(st.session_state.get("universe_simple_mode_enabled", True)) else {}
                                simple_tuning_policy = _coerce_mapping(_coerce_mapping(simple_resolved_summary).get("tuning_policy", {}))

                                objective_name = str(st.session_state.get("universe_local_search_objective", "composite_balanced"))
                                include_shrink = bool(st.session_state.get("universe_local_search_include_weight_shrink", False))
                                intensity = str(st.session_state.get("universe_local_search_intensity", "Quick"))
                                two_stage_enabled = bool(st.session_state.get("universe_local_search_two_stage", False))
                                max_candidates_override = None

                                if simple_tuning_policy:
                                    max_hint = simple_tuning_policy.get("max_candidates_hint")
                                    if max_hint not in {None, ""}:
                                        try:
                                            max_candidates_override = int(max_hint)
                                        except Exception:
                                            max_candidates_override = None

                                refinement_bundle = _run_local_refinement_search(
                                    primary_engine_panel_df,
                                    cfg,
                                    objective=objective_name,
                                    intensity=intensity,
                                    include_weight_shrink=include_shrink,
                                    two_stage=two_stage_enabled,
                                    local_search_policy=simple_tuning_policy or None,
                                    max_candidates_override=max_candidates_override,
                                )
                                local_table = refinement_bundle.get("combined_table", pd.DataFrame())
                                best_local = refinement_bundle.get("best_overall")
                                base_row = refinement_bundle.get("stage1_baseline")

                                st.markdown("##### Local preset refinement")
                                if isinstance(local_table, pd.DataFrame) and not local_table.empty:
                                    base_val = _safe_float(base_row.get("objective_value") if base_row is not None else np.nan)
                                    best_val = _safe_float(best_local.get("objective_value") if best_local is not None else np.nan)
                                    delta_val = best_val - base_val if np.isfinite(best_val) and np.isfinite(base_val) else np.nan
                                    best_payload = _payload_from_json(best_local.get("effective_cfg_payload_json")) if best_local is not None else {}
                                    best_stage = str(best_local.get("refinement_stage", "stage_1")) if best_local is not None else "stage_1"
                                    best_fingerprint = str(best_local.get("effective_config_fingerprint", "")) if best_local is not None else ""
                                    applied_fingerprint = str(st.session_state.get("universe_local_search_last_applied_fingerprint", ""))
                                    is_applied_best = bool(best_fingerprint) and best_fingerprint == applied_fingerprint

                                    m1, m2, m3, m4 = st.columns(4)
                                    with m1:
                                        st.metric("Candidates", int(max(len(local_table) - (1 if base_row is not None else 0), 0)))
                                    with m2:
                                        st.metric("Active local dims", int(pd.Series(local_table.get("tuned_param", pd.Series(dtype=str))).replace("baseline", np.nan).dropna().nunique()))
                                    with m3:
                                        st.metric("Base objective", _format_num_or_dash(base_val))
                                    with m4:
                                        st.metric("Best objective", _format_num_or_dash(best_val))

                                    policy_used = _coerce_mapping(refinement_bundle.get("policy", {}))
                                    search_budget_used = str(policy_used.get("search_budget", "manual")) if policy_used else "manual"
                                    preferred_dims_used = list(policy_used.get("preferred_dims", []) or []) if policy_used else []
                                    st.caption(
                                        f"Objective active: {_format_tuning_objective_label(objective_name)} | intensity: {intensity} | "
                                        f"two-stage: {'on' if two_stage_enabled else 'off'} | search_budget: {search_budget_used}"
                                    )
                                    if preferred_dims_used:
                                        st.caption(f"Policy-guided local dims: {', '.join(preferred_dims_used[:8])}")

                                    if best_local is not None and base_row is not None:
                                        c1, c2, c3, c4, c5 = st.columns(5)
                                        with c1:
                                            st.metric("Best tuned param", str(best_local.get("tuned_param", "—")))
                                        with c2:
                                            st.metric("Best candidate value", _format_candidate_value_or_dash(best_local.get("candidate_value")))
                                        with c3:
                                            st.metric("Δ objective", _format_num_or_dash(delta_val))
                                        with c4:
                                            st.metric("Best Sharpe", _format_num_or_dash(best_local.get("sharpe")))
                                        with c5:
                                            st.metric("Best stage", str(best_stage).replace("_", " ").title())

                                    s1, s2, s3, s4 = st.columns(4)
                                    with s1:
                                        st.metric("Baseline fingerprint", str(base_row.get("effective_config_fingerprint", cfg_fingerprint if (cfg_fingerprint := config_fingerprint(cfg)) else "—"))[:10] if base_row is not None else config_fingerprint(cfg)[:10])
                                    with s2:
                                        st.metric("Refined fingerprint", (best_fingerprint[:10] if best_fingerprint else "—"))
                                    with s3:
                                        st.metric("Applied", "Yes" if is_applied_best else "No")
                                    with s4:
                                        st.metric("Applied fingerprint", applied_fingerprint[:10] if applied_fingerprint else "—")

                                    if _is_composite_objective(objective_name) and best_local is not None:
                                        st.caption("Composite score breakdown")
                                        b1, b2, b3, b4, b5 = st.columns(5)
                                        with b1:
                                            st.metric("Sharpe contribution", _format_num_or_dash(best_local.get("composite_contribution_sharpe")))
                                        with b2:
                                            st.metric("CAGR contribution", _format_num_or_dash(best_local.get("composite_contribution_cagr")))
                                        with b3:
                                            st.metric("DD contribution", _format_num_or_dash(best_local.get("composite_contribution_max_drawdown")))
                                        with b4:
                                            st.metric("Turnover contribution", _format_num_or_dash(best_local.get("composite_contribution_mean_turnover")))
                                        with b5:
                                            st.metric("Diversification contribution", _format_num_or_dash(best_local.get("composite_contribution_diversification")))

                                        bd1, bd2, bd3, bd4 = st.columns(4)
                                        with bd1:
                                            st.metric("Composite score", _format_num_or_dash(best_local.get("composite_objective_score")))
                                        with bd2:
                                            st.metric("Diversification source", str(best_local.get("composite_diversification_source", "—")))
                                        with bd3:
                                            st.metric("Active components", str(best_local.get("composite_objective_active_components", "—")))
                                        with bd4:
                                            st.metric("Missing components", str(best_local.get("composite_objective_missing_components", "—")))

                                    history_entry = {
                                        "timestamp": pd.Timestamp.utcnow().isoformat(),
                                        "objective": objective_name,
                                        "intensity": intensity,
                                        "two_stage": bool(two_stage_enabled),
                                        "candidate_count": int(max(len(local_table) - (1 if base_row is not None else 0), 0)),
                                        "base_objective": base_val,
                                        "best_objective": best_val,
                                        "objective_improvement": delta_val,
                                        "best_param": str(best_local.get("tuned_param", "")) if best_local is not None else "",
                                        "best_value": best_local.get("candidate_value") if best_local is not None else np.nan,
                                        "best_stage": best_stage,
                                        "best_config_fingerprint": best_fingerprint,
                                    }
                                    _maybe_store_refinement_history(history_entry)

                                    search_mode_col, picker_col, profile_col, history_col = st.columns([1.05, 1.2, 0.95, 1.0])
                                    with search_mode_col:
                                        search_mode = st.selectbox(
                                            "search mode",
                                            ["single_objective", "multiobjective_pareto"],
                                            key="universe_local_search_mode",
                                            help="Single-objective keeps the existing best-candidate flow. Multiobjective Pareto builds a non-dominated frontier over the local search trials.",
                                        )
                                    local_policy_options = _get_selection_policy_options_for_current_mode()
                                    local_policy_locked = _is_simple_mode_active()
                                    with picker_col:
                                        if local_policy_locked:
                                            solution_picker = "fixed_composite_score"
                                            st.session_state["universe_local_search_solution_picker"] = solution_picker
                                            st.text_input(
                                                "selection policy",
                                                value=solution_picker,
                                                disabled=True,
                                                key="universe_local_search_solution_picker_locked_display",
                                                help="Simple / Auto mode uses the fixed composite score chooser as the tested default.",
                                            )
                                        else:
                                            solution_picker = st.selectbox(
                                                "selection policy",
                                                local_policy_options,
                                                key="universe_local_search_solution_picker",
                                                help="Used to preselect one candidate from the Pareto frontier and to drive the default row selection.",
                                            )
                                    local_composite_profile: Any = DEFAULT_COMPOSITE_PROFILE
                                    with profile_col:
                                        if str(solution_picker).strip().lower() == "fixed_composite_score":
                                            if local_policy_locked:
                                                local_composite_profile = _resolve_simple_mode_composite_profile()
                                                st.session_state["universe_local_search_composite_profile"] = local_composite_profile
                                                st.text_input(
                                                    "composite profile",
                                                    value=local_composite_profile,
                                                    disabled=True,
                                                    key="universe_local_search_composite_profile_locked_display",
                                                    help="Simple / Auto mode resolves the composite profile from the semantic style preset.",
                                                )
                                            else:
                                                local_composite_profile = st.selectbox(
                                                    "composite profile",
                                                    COMPOSITE_PROFILE_OPTIONS,
                                                    index=COMPOSITE_PROFILE_OPTIONS.index(DEFAULT_COMPOSITE_PROFILE),
                                                    key="universe_local_search_composite_profile",
                                                    help="Profile used by the fixed composite score chooser.",
                                                )
                                                local_manual_enabled, local_manual_weights = _render_manual_composite_weights_ui(
                                                    "universe_local_search",
                                                    str(local_composite_profile),
                                                )
                                                if local_manual_enabled and local_manual_weights:
                                                    local_composite_profile = dict(local_manual_weights)
                                        else:
                                            st.caption("")
                                    with history_col:
                                        st.caption("Baseline vs refined vs applied")
                                        st.write(
                                            f"Baseline `{config_fingerprint(cfg)[:10]}` → Refined `{best_fingerprint[:10] if best_fingerprint else '—'}` → Applied `{applied_fingerprint[:10] if applied_fingerprint else '—'}`"
                                        )

                                    selected_objectives = st.multiselect(
                                        "Pareto objectives",
                                        PARETO_OBJECTIVE_OPTIONS,
                                        default=["sharpe", "cagr", "max_drawdown", "mean_turnover", "diversification", "stability"],
                                        key="universe_local_search_pareto_objectives",
                                        format_func=_format_pareto_objective_label,
                                    )

                                    if str(search_mode) == "multiobjective_pareto":
                                        pareto_input = _build_local_pareto_input(local_table)
                                        pareto_specs = {
                                            str(obj): {"sense": ("min" if str(obj) in {"max_drawdown", "mean_turnover"} else "max")}
                                            for obj in (selected_objectives or [])
                                        }
                                        pareto_candidates = pareto_input.copy()
                                        if "refinement_stage" in pareto_candidates.columns:
                                            pareto_candidates = pareto_candidates.loc[
                                                pareto_candidates["refinement_stage"].fillna("").astype(str).str.lower() != "baseline"
                                            ].copy()
                                        if pareto_candidates.empty:
                                            st.info("No Pareto candidates available yet after removing the baseline row.")
                                        elif not pareto_specs:
                                            st.info("Choose at least one Pareto objective to build the frontier.")
                                        else:
                                            pareto_table = _pareto_metadata_app(
                                                pareto_candidates,
                                                objective_cols=list(pareto_specs.keys()),
                                                objective_specs=pareto_specs,
                                            )
                                            pareto_frontier = pareto_table.loc[pareto_table.get("is_pareto_efficient", False).astype(bool)].copy()
                                            if pareto_frontier.empty and not pareto_table.empty and "pareto_rank" in pareto_table.columns:
                                                min_rank = pd.to_numeric(pareto_table["pareto_rank"], errors="coerce").min()
                                                if np.isfinite(min_rank):
                                                    pareto_frontier = pareto_table.loc[pd.to_numeric(pareto_table["pareto_rank"], errors="coerce") == min_rank].copy()
                                            frontier_for_selection = pareto_frontier if not pareto_frontier.empty else pareto_table
                                            auto_choice, frontier_for_selection = _choose_pareto_solution_app(
                                                frontier_for_selection,
                                                policy=solution_picker,
                                                objective_specs=pareto_specs,
                                                composite_profile=local_composite_profile,
                                            )
                                            auto_fingerprint = str(auto_choice.get("effective_config_fingerprint") or auto_choice.get("config_fingerprint") or "")
                                            hypervolume_total = _safe_float(frontier_for_selection.get("hypervolume_total").iloc[0]) if isinstance(frontier_for_selection, pd.DataFrame) and not frontier_for_selection.empty and "hypervolume_total" in frontier_for_selection.columns else np.nan

                                            if not pareto_frontier.empty:
                                                st.markdown("**Pareto frontier**")
                                                pf1, pf2, pf3, pf4, pf5 = st.columns(5)
                                                with pf1:
                                                    st.metric("Frontier size", int(len(pareto_frontier)))
                                                with pf2:
                                                    st.metric("Auto-picked Sharpe", _format_num_or_dash(auto_choice.get("sharpe")))
                                                with pf3:
                                                    st.metric("Auto-picked CAGR", _format_pct_or_dash(auto_choice.get("cagr")))
                                                with pf4:
                                                    st.metric("Auto-picked turnover", _format_pct_or_dash(auto_choice.get("mean_turnover")))
                                                with pf5:
                                                    st.metric("Hypervolume total", _format_num_or_dash(hypervolume_total))
                                                st.caption(_build_hypervolume_caption(frontier_for_selection))

                                            candidate_options = []
                                            candidate_lookup = {}
                                            for pos, (_, row) in enumerate(frontier_for_selection.reset_index(drop=True).iterrows()):
                                                label = _make_pareto_candidate_label(row, pos)
                                                candidate_options.append(label)
                                                candidate_lookup[label] = _row_to_dict(row)
                                            default_idx = 0
                                            if auto_fingerprint and candidate_options:
                                                for i, label in enumerate(candidate_options):
                                                    row_fp = str(candidate_lookup[label].get("effective_config_fingerprint") or candidate_lookup[label].get("config_fingerprint") or "")
                                                    if row_fp and row_fp == auto_fingerprint:
                                                        default_idx = i
                                                        break
                                            selected_label = st.selectbox(
                                                "Pareto candidate row",
                                                candidate_options,
                                                index=default_idx if candidate_options else None,
                                                key="universe_local_search_pareto_candidate_row",
                                                help="Select one efficient candidate from the frontier to apply.",
                                            ) if candidate_options else None
                                            selected_row = _coerce_mapping(candidate_lookup.get(selected_label, auto_choice))
                                            selected_payload = _payload_from_json(selected_row.get("effective_cfg_payload_json")) if selected_row else {}
                                            selected_fingerprint = str(selected_row.get("effective_config_fingerprint") or selected_row.get("config_fingerprint") or "")

                                            apply_col, picker_info_col = st.columns([1.2, 1.0])
                                            with apply_col:
                                                selected_improvement = _safe_float(selected_row.get("objective_value")) - base_val if selected_row else np.nan
                                                apply_disabled = not bool(selected_payload)
                                                if st.button(
                                                    "Apply selected Pareto solution",
                                                    key="universe_local_search_apply_pareto_selected",
                                                    use_container_width=True,
                                                    disabled=apply_disabled,
                                                    help="Promote the selected Pareto-efficient candidate into the active manual config for the next run.",
                                                ):
                                                    selected_stage = str(selected_row.get("refinement_stage", "pareto"))
                                                    _queue_promote_cfg_payload_to_manual_state(selected_payload, source_label=f"pareto / {solution_picker} / {selected_stage}")
                                                    applied_entry = dict(history_entry)
                                                    applied_entry.update({
                                                        "applied": True,
                                                        "applied_fingerprint": selected_fingerprint,
                                                        "best_stage": selected_stage,
                                                        "best_param": str(selected_row.get("tuned_param", "pareto")),
                                                        "best_value": selected_row.get("candidate_value"),
                                                        "best_objective": selected_row.get("objective_value"),
                                                        "objective_improvement": selected_improvement,
                                                        "selection_policy": solution_picker,
                                                        "search_mode": search_mode,
                                                    })
                                                    _append_refinement_history(applied_entry)
                                                    st.session_state["universe_local_search_last_applied_fingerprint"] = selected_fingerprint
                                                    st.rerun()
                                            with picker_info_col:
                                                st.caption("Selected Pareto solution")
                                                st.write(
                                                    f"Fingerprint `{selected_fingerprint[:10] if selected_fingerprint else '—'}` | "
                                                    f"policy `{solution_picker}` | objectives `{', '.join([_format_pareto_objective_label(x) for x in selected_objectives])}`"
                                                )

                                            frontier_cols = [
                                                "pareto_rank", "crowding_distance", "dominates_count", "dominated_by_count",
                                                "hypervolume_contribution", "hypervolume_contribution_share",
                                                "refinement_stage", "tuned_param", "candidate_value", "candidate_delta_pct",
                                                "objective_value", "sharpe", "cagr", "max_drawdown", "mean_turnover",
                                                "diversification", "stability", "annual_volatility",
                                                "effective_config_fingerprint", "config_fingerprint", "status",
                                            ]
                                            frontier_cols = [c for c in frontier_cols if c in frontier_for_selection.columns]
                                            st.dataframe(frontier_for_selection[frontier_cols], use_container_width=True, hide_index=True)
                                    else:
                                        apply_col, history_only_col = st.columns([1.2, 1.0])
                                        with apply_col:
                                            apply_disabled = not bool(best_payload) or not np.isfinite(delta_val) or delta_val <= 1e-12
                                            if st.button(
                                                "Apply best candidate",
                                                key="universe_local_search_apply_best",
                                                use_container_width=True,
                                                disabled=apply_disabled,
                                                help="Promote the best refined candidate into the active manual config for the next run.",
                                            ):
                                                _queue_promote_cfg_payload_to_manual_state(best_payload, source_label=f"{objective_name} / {best_stage}")
                                                applied_entry = dict(history_entry)
                                                applied_entry.update({"applied": True, "applied_fingerprint": best_fingerprint, "search_mode": search_mode})
                                                _append_refinement_history(applied_entry)
                                                st.session_state["universe_local_search_last_applied_fingerprint"] = best_fingerprint
                                                st.rerun()
                                        with history_only_col:
                                            st.caption("Single-objective apply path")
                                            st.write(f"Best stage `{best_stage}` | Δ objective `{_format_num_or_dash(delta_val)}`")

                                        focus_cols = [
                                            "refinement_stage", "tuned_param", "candidate_value", "candidate_delta_pct",
                                            "objective_value", "sharpe", "cagr", "annual_volatility",
                                            "max_drawdown", "mean_turnover", "information_ratio",
                                            "mean_effective_breadth", "mean_effective_risk_bets",
                                            "mean_active_assets",
                                            "composite_objective_score", "composite_objective_preset",
                                            "composite_contribution_sharpe", "composite_contribution_cagr",
                                            "composite_contribution_max_drawdown", "composite_contribution_mean_turnover",
                                            "composite_contribution_diversification", "composite_diversification_source",
                                            "effective_config_fingerprint", "config_fingerprint", "status",
                                        ]
                                        focus_cols = [c for c in focus_cols if c in local_table.columns]
                                        st.dataframe(local_table[focus_cols], use_container_width=True, hide_index=True)

                                    history_df = pd.DataFrame(list(st.session_state.get("universe_local_refinement_history", [])))
                                    if not history_df.empty:
                                        with st.expander("Refinement history", expanded=False):
                                            show_cols = [
                                                "timestamp", "objective", "intensity", "two_stage", "candidate_count",
                                                "base_objective", "best_objective", "objective_improvement",
                                                "best_param", "best_value", "best_stage", "best_config_fingerprint", "applied",
                                            ]
                                            show_cols = [c for c in show_cols if c in history_df.columns]
                                            st.dataframe(history_df[show_cols].iloc[::-1], use_container_width=True, hide_index=True)
                                            if st.button("Clear refinement history", key="universe_local_search_clear_history"):
                                                st.session_state["universe_local_refinement_history"] = []
                                                st.session_state["universe_local_refinement_history_last_signature"] = ""
                                                st.rerun()

                                    if bool(st.session_state.get("universe_local_search_show_detail", False)):
                                        with st.expander("Local search diagnostics / downloads", expanded=False):
                                            _download_dataframe_button(
                                                "Download local-search table",
                                                local_table,
                                                "primary_local_search.csv",
                                                key="primary_local_search_download",
                                            )
                                            st.dataframe(local_table, use_container_width=True)
                                else:
                                    st.info("Local preset refinement did not produce any active candidates in the current engine context.")
                            except Exception as e:
                                st.warning(f"Could not run local preset refinement: {e}")

                if bool(st.session_state.get("universe_run_pure_cs_baseline_compare", False)):
                    try:
                        pure_cs_cfg = make_pure_cs_baseline_config(cfg)
                        pure_cs_run = run_micro_investment_pipeline(primary_engine_panel_df, cfg=pure_cs_cfg)
                        pure_cs_perf = pure_cs_run.get("performance_summary", {}) or {}

                        st.markdown("#### PURE_CS baseline comparison")
                        pc1, pc2, pc3, pc4 = st.columns(4)
                        with pc1:
                            st.metric("PURE_CS CAGR", _format_pct_or_dash(pure_cs_perf.get("cagr")))
                        with pc2:
                            st.metric("PURE_CS Vol", _format_pct_or_dash(pure_cs_perf.get("annual_volatility", pure_cs_perf.get("annualized_volatility"))))
                        with pc3:
                            st.metric("PURE_CS Sharpe", _format_num_or_dash(pure_cs_perf.get("sharpe")))
                        with pc4:
                            st.metric("PURE_CS MaxDD", _format_pct_or_dash(pure_cs_perf.get("max_drawdown")))

                        pure_bundle = compare_run_reports(pure_cs_run, primary_run, left_name="pure_cs", right_name="engine")
                        pure_metrics_cmp = _coerce_numeric_columns(pure_bundle.get("metrics_comparison", pd.DataFrame()), skip={"section", "metric"})
                        if isinstance(pure_metrics_cmp, pd.DataFrame) and not pure_metrics_cmp.empty:
                            pure_focus = pure_metrics_cmp[pure_metrics_cmp["metric"].isin([
                                "cagr",
                                "sharpe",
                                "max_drawdown",
                                "annual_volatility",
                                "mean_turnover",
                                "active_return_annual",
                                "tracking_error_annual",
                                "information_ratio",
                                "mean_effective_breadth",
                                "mean_effective_risk_bets",
                                "mean_diversification_ratio",
                                "mean_active_assets",
                            ])].copy()
                            if not pure_focus.empty:
                                st.dataframe(pure_focus, use_container_width=True)

                        ep1, ep2, ep3, ep4 = st.columns(4)
                        with ep1:
                            d = pd.to_numeric(pure_metrics_cmp.loc[pure_metrics_cmp["metric"] == "sharpe", "delta"], errors="coerce") if isinstance(pure_metrics_cmp, pd.DataFrame) and not pure_metrics_cmp.empty else pd.Series(dtype="float64")
                            st.metric("Engine Δ Sharpe vs PURE_CS", _format_num_or_dash(d.iloc[0] if not d.empty else np.nan))
                        with ep2:
                            d = pd.to_numeric(pure_metrics_cmp.loc[pure_metrics_cmp["metric"] == "cagr", "delta"], errors="coerce") if isinstance(pure_metrics_cmp, pd.DataFrame) and not pure_metrics_cmp.empty else pd.Series(dtype="float64")
                            st.metric("Engine Δ CAGR vs PURE_CS", _format_delta_pct_or_dash(d.iloc[0] if not d.empty else np.nan))
                        with ep3:
                            d = pd.to_numeric(pure_metrics_cmp.loc[pure_metrics_cmp["metric"] == "max_drawdown", "delta"], errors="coerce") if isinstance(pure_metrics_cmp, pd.DataFrame) and not pure_metrics_cmp.empty else pd.Series(dtype="float64")
                            st.metric("Engine Δ MaxDD vs PURE_CS", _format_delta_pct_or_dash(d.iloc[0] if not d.empty else np.nan))
                        with ep4:
                            d = pd.to_numeric(pure_metrics_cmp.loc[pure_metrics_cmp["metric"] == "mean_turnover", "delta"], errors="coerce") if isinstance(pure_metrics_cmp, pd.DataFrame) and not pure_metrics_cmp.empty else pd.Series(dtype="float64")
                            st.metric("Engine Δ Turnover vs PURE_CS", _format_delta_pct_or_dash(d.iloc[0] if not d.empty else np.nan))

                        if bool(st.session_state.get("universe_show_pure_cs_report", False)):
                            _render_run_report("PURE_CS baseline", pure_cs_run, prefix="pure_cs")

                        with st.expander("Detailed engine vs PURE_CS baseline", expanded=False):
                            st.dataframe(pure_metrics_cmp, use_container_width=True)
                            pure_diag_cmp = _coerce_numeric_columns(pure_bundle.get("diagnostics_comparison", pd.DataFrame()), skip={"metric"})
                            pure_w_cmp = _coerce_numeric_columns(pure_bundle.get("weights_comparison", pd.DataFrame()), skip={"asset"})
                            if isinstance(pure_diag_cmp, pd.DataFrame) and not pure_diag_cmp.empty:
                                st.markdown("**Diagnostics mean comparison**")
                                st.dataframe(pure_diag_cmp, use_container_width=True)
                            if isinstance(pure_w_cmp, pd.DataFrame) and not pure_w_cmp.empty:
                                st.markdown("**Average weight comparison**")
                                st.dataframe(pure_w_cmp, use_container_width=True)
                            dl1, dl2, dl3 = st.columns(3)
                            with dl1:
                                _download_dataframe_button("Download engine vs PURE_CS metrics", pure_metrics_cmp, "engine_vs_pure_cs_metrics.csv", key="engine_pure_cs_metrics_download")
                            with dl2:
                                _download_dataframe_button("Download engine vs PURE_CS diagnostics", pure_diag_cmp, "engine_vs_pure_cs_diagnostics.csv", key="engine_pure_cs_diag_download")
                            with dl3:
                                _download_dataframe_button("Download engine vs PURE_CS weights", pure_w_cmp, "engine_vs_pure_cs_weights.csv", key="engine_pure_cs_weights_download")
                    except Exception as e:
                        st.warning(f"Could not run PURE_CS baseline comparison: {e}")

                if bool(st.session_state.get("universe_preset_sweep_enabled", False)):
                    try:
                        selected_presets = tuple(
                            p for p in st.session_state.get("universe_preset_sweep_selected", list(ENGINE_PRESETS.keys()))
                            if p in ENGINE_PRESETS
                        )
                        st.markdown("#### Automatic preset sweep")
                        if selected_presets:
                            cfg_payload = config_to_dict(cfg)
                            sweep_df = _run_preset_sweep_cached(
                                primary_engine_panel_df,
                                json.dumps(cfg_payload, sort_keys=True),
                                selected_presets,
                            )
                            if isinstance(sweep_df, pd.DataFrame) and not sweep_df.empty:
                                sort_cols = [c for c in ["sharpe", "information_ratio", "cagr"] if c in sweep_df.columns]
                                if sort_cols:
                                    sweep_df = sweep_df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)

                                best_row = sweep_df.iloc[0]
                                s1, s2, s3, s4 = st.columns(4)
                                with s1:
                                    st.metric("Best preset", str(best_row.get("preset_name", "—")))
                                with s2:
                                    st.metric("Best Sharpe", _format_num_or_dash(best_row.get("sharpe")))
                                with s3:
                                    st.metric("Best CAGR", _format_pct_or_dash(best_row.get("cagr")))
                                with s4:
                                    st.metric("Best turnover", _format_pct_or_dash(best_row.get("mean_turnover")))

                                focus_cols = [
                                    "preset_name",
                                    "sharpe",
                                    "cagr",
                                    "annual_volatility",
                                    "max_drawdown",
                                    "mean_turnover",
                                    "active_return_annual",
                                    "tracking_error_annual",
                                    "information_ratio",
                                    "mean_effective_breadth",
                                    "mean_diversification_ratio",
                                    "mean_effective_risk_bets",
                                    "mean_active_assets",
                                    "config_fingerprint",
                                ]
                                focus_cols = [c for c in focus_cols if c in sweep_df.columns]
                                st.dataframe(sweep_df[focus_cols], use_container_width=True)

                                with st.expander("Preset sweep diagnostics / downloads", expanded=False):
                                    _download_dataframe_button(
                                        "Download preset sweep results",
                                        sweep_df,
                                        "primary_preset_sweep.csv",
                                        key="primary_preset_sweep_download",
                                    )

                                try:
                                    frontier_df = build_sharpe_turnover_frontier(
                                        sweep_df.rename(columns={"preset_name": "label"}),
                                        sharpe_col="sharpe",
                                        turnover_col="mean_turnover",
                                    )
                                    frontier_only = frontier_df[frontier_df["on_frontier"] == True].copy() if "on_frontier" in frontier_df.columns else frontier_df.copy()
                                    st.markdown("**Preset frontier snapshot**")
                                    frontier_cols = [c for c in ["preset_name", "sharpe", "mean_turnover", "frontier_rank", "on_frontier"] if c in frontier_only.columns]
                                    if frontier_cols:
                                        st.dataframe(frontier_only[frontier_cols], use_container_width=True)
                                except Exception:
                                    pass
                            else:
                                st.info("Preset sweep produced an empty result table.")
                        else:
                            st.info("Select at least one preset to run the sweep.")
                    except Exception as e:
                        st.warning(f"Could not build automatic preset sweep: {e}")


                if bool(st.session_state.get("universe_param_sweep_enabled", False)):
                    try:
                        temp_grid = tuple(_parse_float_grid(
                            st.session_state.get("universe_param_sweep_temp_grid_text", ""),
                            default=[0.7, 1.0, 1.3],
                        ))
                        shrink_grid = tuple(_parse_float_grid(
                            st.session_state.get("universe_param_sweep_shrink_grid_text", ""),
                            default=[0.0, 0.05, 0.10],
                        ))
                        inertia_grid = tuple(_parse_float_grid(
                            st.session_state.get("universe_param_sweep_inertia_grid_text", ""),
                            default=[0.0, 0.2, 0.4],
                        ))

                        cfg_payload = config_to_dict(cfg)
                        grid_df = _run_param_grid_cached(
                            primary_engine_panel_df,
                            json.dumps(cfg_payload, sort_keys=True),
                            temp_grid,
                            shrink_grid,
                            inertia_grid,
                        )

                        st.markdown("#### Param grid runner / param sweep")
                        if isinstance(grid_df, pd.DataFrame) and not grid_df.empty:
                            sort_cols = [c for c in ["sharpe", "information_ratio", "cagr"] if c in grid_df.columns]
                            if sort_cols:
                                grid_df = grid_df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)

                            g1, g2, g3, g4 = st.columns(4)
                            with g1:
                                st.metric("Grid rows", int(len(grid_df)))
                            with g2:
                                st.metric("Best Sharpe", _format_num_or_dash(pd.to_numeric(grid_df.get("sharpe"), errors="coerce").max() if "sharpe" in grid_df.columns else None))
                            with g3:
                                st.metric("Best CAGR", _format_pct_or_dash(pd.to_numeric(grid_df.get("cagr"), errors="coerce").max() if "cagr" in grid_df.columns else None))
                            with g4:
                                st.metric("Lowest MaxDD", _format_pct_or_dash(pd.to_numeric(grid_df.get("max_drawdown"), errors="coerce").max() if "max_drawdown" in grid_df.columns else None))

                            focus_cols = [
                                "temperature", "weight_shrink", "inertia",
                                "sharpe", "cagr", "annual_volatility", "max_drawdown",
                                "mean_turnover", "active_return_annual", "tracking_error_annual",
                                "information_ratio", "mean_effective_breadth",
                                "mean_diversification_ratio", "mean_effective_risk_bets",
                                "mean_active_assets", "config_fingerprint",
                            ]
                            focus_cols = [c for c in focus_cols if c in grid_df.columns]
                            st.dataframe(grid_df[focus_cols], use_container_width=True)

                            with st.expander("Param sweep diagnostics / downloads", expanded=False):
                                _download_dataframe_button(
                                    "Download param sweep grid",
                                    grid_df,
                                    "primary_param_sweep_grid.csv",
                                    key="primary_param_sweep_grid_download",
                                )

                            try:
                                frontier_df = build_sharpe_turnover_frontier(grid_df, sharpe_col="sharpe", turnover_col="mean_turnover")
                                chosen, frontier_view = choose_sharpe_turnover_candidate(frontier_df, max_turnover=0.35)
                                st.markdown("**Sweep-derived frontier snapshot**")
                                if chosen is not None:
                                    f1, f2, f3, f4 = st.columns(4)
                                    with f1:
                                        st.metric("Frontier Sharpe", _format_num_or_dash(chosen.get("sharpe")))
                                    with f2:
                                        st.metric("Frontier turnover", _format_pct_or_dash(chosen.get("mean_turnover")))
                                    with f3:
                                        st.metric("Frontier temperature", _format_num_or_dash(chosen.get("temperature")))
                                    with f4:
                                        st.metric("Frontier shrink / inertia", f"{chosen.get('weight_shrink', '—')} / {chosen.get('inertia', '—')}")
                                if isinstance(frontier_view, pd.DataFrame) and not frontier_view.empty:
                                    frontier_cols = [c for c in ["temperature","weight_shrink","inertia","sharpe","mean_turnover","frontier_rank","on_frontier"] if c in frontier_view.columns]
                                    st.dataframe(frontier_view[frontier_cols], use_container_width=True)
                            except Exception:
                                pass
                        else:
                            st.info("Param sweep produced an empty grid.")
                    except Exception as e:
                        st.warning(f"Could not build param sweep grid: {e}")


                if bool(st.session_state.get("universe_parameter_stability_enabled", False)):
                    try:
                        temp_grid = tuple(_parse_float_grid(
                            st.session_state.get("universe_parameter_stability_temp_grid_text", ""),
                            default=[0.7, 1.0, 1.3],
                        ))
                        shrink_grid = tuple(_parse_float_grid(
                            st.session_state.get("universe_parameter_stability_shrink_grid_text", ""),
                            default=[0.0, 0.05, 0.10],
                        ))
                        inertia_grid = tuple(_parse_float_grid(
                            st.session_state.get("universe_parameter_stability_inertia_grid_text", ""),
                            default=[0.0, 0.2, 0.4],
                        ))

                        cfg_payload = config_to_dict(cfg)
                        stability_grid_df = _run_param_grid_cached(
                            primary_engine_panel_df,
                            json.dumps(cfg_payload, sort_keys=True),
                            temp_grid,
                            shrink_grid,
                            inertia_grid,
                        )
                        overall_df, resilient_df, param_df = _build_parameter_stability_outputs(
                            stability_grid_df,
                            near_best_tol=float(st.session_state.get("universe_parameter_stability_near_best_tol", 0.10)),
                        )

                        st.markdown("#### Parameter stability module")
                        if isinstance(overall_df, pd.DataFrame) and not overall_df.empty:
                            row = overall_df.iloc[0]
                            s1, s2, s3, s4 = st.columns(4)
                            with s1:
                                st.metric("Best Sharpe", _format_num_or_dash(row.get("best_sharpe")))
                            with s2:
                                st.metric("Median Sharpe", _format_num_or_dash(row.get("median_sharpe")))
                            with s3:
                                st.metric("Sharpe std", _format_num_or_dash(row.get("sharpe_std")))
                            with s4:
                                st.metric("Near-best rate", _format_pct_or_dash(row.get("near_best_sharpe_rate")))

                            s5, s6, s7, s8 = st.columns(4)
                            with s5:
                                st.metric("Positive Sharpe rate", _format_pct_or_dash(row.get("positive_sharpe_rate")))
                            with s6:
                                st.metric("Median CAGR", _format_pct_or_dash(row.get("median_cagr")))
                            with s7:
                                st.metric("Median turnover", _format_pct_or_dash(row.get("median_turnover")))
                            with s8:
                                st.metric("Grid rows", int(row.get("grid_rows", 0)))

                        if isinstance(resilient_df, pd.DataFrame) and not resilient_df.empty:
                            st.markdown("**Most resilient candidates**")
                            resilient_cols = [c for c in [
                                "temperature", "weight_shrink", "inertia", "stability_score", "sharpe", "cagr",
                                "mean_turnover", "near_best_sharpe", "config_fingerprint"
                            ] if c in resilient_df.columns]
                            st.dataframe(resilient_df[resilient_cols].head(12), use_container_width=True)

                        if bool(st.session_state.get("universe_parameter_stability_show_detail", False)) and isinstance(param_df, pd.DataFrame) and not param_df.empty:
                            st.markdown("**Parameter sensitivity tables**")
                            for pname in ["temperature", "weight_shrink", "inertia"]:
                                sub = param_df[param_df["parameter"] == pname].copy()
                                if sub.empty:
                                    continue
                                show_cols = [c for c in ["value", "n", "best_sharpe", "median_sharpe", "sharpe_std", "positive_sharpe_rate", "near_best_sharpe_rate", "median_cagr", "median_turnover"] if c in sub.columns]
                                st.markdown(f"*{pname}*")
                                st.dataframe(sub[show_cols], use_container_width=True)

                        with st.expander("Parameter stability downloads", expanded=False):
                            if isinstance(stability_grid_df, pd.DataFrame) and not stability_grid_df.empty:
                                _download_dataframe_button(
                                    "Download stability grid",
                                    stability_grid_df,
                                    "primary_parameter_stability_grid.csv",
                                    key="primary_parameter_stability_grid_download",
                                )
                            if isinstance(resilient_df, pd.DataFrame) and not resilient_df.empty:
                                _download_dataframe_button(
                                    "Download resilient candidates",
                                    resilient_df,
                                    "primary_parameter_stability_resilient.csv",
                                    key="primary_parameter_stability_resilient_download",
                                )
                            if isinstance(param_df, pd.DataFrame) and not param_df.empty:
                                _download_dataframe_button(
                                    "Download parameter sensitivity summary",
                                    param_df,
                                    "primary_parameter_stability_summary.csv",
                                    key="primary_parameter_stability_summary_download",
                                )
                    except Exception as e:
                        st.warning(f"Could not build parameter stability module: {e}")

                if bool(st.session_state.get("universe_sensitivity_analysis_enabled", False)):
                    try:
                        temp_grid = tuple(_parse_float_grid(
                            st.session_state.get("universe_sensitivity_temp_grid_text", ""),
                            default=[0.7, 1.0, 1.3],
                        ))
                        shrink_grid = tuple(_parse_float_grid(
                            st.session_state.get("universe_sensitivity_shrink_grid_text", ""),
                            default=[0.0, 0.05, 0.10],
                        ))
                        inertia_grid = tuple(_parse_float_grid(
                            st.session_state.get("universe_sensitivity_inertia_grid_text", ""),
                            default=[0.0, 0.2, 0.4],
                        ))
                        metric_name = str(st.session_state.get("universe_sensitivity_metric", "sharpe"))
                        cfg_payload = config_to_dict(cfg)
                        sensitivity_grid_df = _run_param_grid_cached(
                            primary_engine_panel_df,
                            json.dumps(cfg_payload, sort_keys=True),
                            temp_grid,
                            shrink_grid,
                            inertia_grid,
                        )
                        sens_overall_df, sens_summary_df, sens_detail_df = _build_sensitivity_analysis_outputs(
                            sensitivity_grid_df,
                            metric_col=metric_name,
                        )

                        st.markdown("#### Sensitivity analysis")
                        if isinstance(sens_overall_df, pd.DataFrame) and not sens_overall_df.empty:
                            row = sens_overall_df.iloc[0]
                            q1, q2, q3, q4 = st.columns(4)
                            with q1:
                                st.metric("Metric", str(row.get("metric", metric_name)))
                            with q2:
                                st.metric("Best value", _format_num_or_dash(row.get("best_metric")))
                            with q3:
                                st.metric("Median value", _format_num_or_dash(row.get("median_metric")))
                            with q4:
                                st.metric("Metric std", _format_num_or_dash(row.get("metric_std")))

                            q5, q6 = st.columns(2)
                            with q5:
                                st.metric("Most sensitive parameter", str(row.get("most_sensitive_parameter", "—")))
                            with q6:
                                st.metric("Max relative sensitivity", _format_num_or_dash(row.get("max_relative_sensitivity")))

                        if isinstance(sens_summary_df, pd.DataFrame) and not sens_summary_df.empty:
                            st.markdown("**Sensitivity by parameter**")
                            st.dataframe(sens_summary_df, use_container_width=True)

                        if bool(st.session_state.get("universe_sensitivity_show_detail", False)) and isinstance(sens_detail_df, pd.DataFrame) and not sens_detail_df.empty:
                            st.markdown("**Detailed value-level sensitivity**")
                            for pname in ["temperature", "weight_shrink", "inertia"]:
                                sub = sens_detail_df[sens_detail_df["parameter"] == pname].copy()
                                if sub.empty:
                                    continue
                                st.markdown(f"*{pname}*")
                                st.dataframe(sub, use_container_width=True)

                        with st.expander("Sensitivity analysis downloads", expanded=False):
                            if isinstance(sensitivity_grid_df, pd.DataFrame) and not sensitivity_grid_df.empty:
                                _download_dataframe_button(
                                    "Download sensitivity grid",
                                    sensitivity_grid_df,
                                    "primary_sensitivity_grid.csv",
                                    key="primary_sensitivity_grid_download",
                                )
                            if isinstance(sens_summary_df, pd.DataFrame) and not sens_summary_df.empty:
                                _download_dataframe_button(
                                    "Download sensitivity summary",
                                    sens_summary_df,
                                    "primary_sensitivity_summary.csv",
                                    key="primary_sensitivity_summary_download",
                                )
                            if isinstance(sens_detail_df, pd.DataFrame) and not sens_detail_df.empty:
                                _download_dataframe_button(
                                    "Download sensitivity detail",
                                    sens_detail_df,
                                    "primary_sensitivity_detail.csv",
                                    key="primary_sensitivity_detail_download",
                                )
                    except Exception as e:
                        st.warning(f"Could not build sensitivity analysis: {e}")
                if bool(st.session_state.get("universe_robust_region_enabled", False)):
                    try:
                        temp_grid = tuple(_parse_float_grid(
                            st.session_state.get("universe_robust_region_temp_grid_text", ""),
                            default=[0.7, 1.0, 1.3],
                        ))
                        shrink_grid = tuple(_parse_float_grid(
                            st.session_state.get("universe_robust_region_shrink_grid_text", ""),
                            default=[0.0, 0.05, 0.10],
                        ))
                        inertia_grid = tuple(_parse_float_grid(
                            st.session_state.get("universe_robust_region_inertia_grid_text", ""),
                            default=[0.0, 0.2, 0.4],
                        ))
                        sharpe_rel_tol = float(st.session_state.get("universe_robust_region_sharpe_rel_tol", 0.95))
                        max_turnover = None
                        if bool(st.session_state.get("universe_robust_region_max_turnover_enabled", False)):
                            max_turnover = float(st.session_state.get("universe_robust_region_max_turnover", 0.35))

                        cfg_payload = config_to_dict(cfg)
                        robust_grid_df = _run_param_grid_cached(
                            primary_engine_panel_df,
                            json.dumps(cfg_payload, sort_keys=True),
                            temp_grid,
                            shrink_grid,
                            inertia_grid,
                        )
                        plateau_summary_df = summarize_plateau_region(
                            robust_grid_df,
                            sharpe_col="sharpe",
                            turnover_col="mean_turnover",
                            sharpe_rel_tol=sharpe_rel_tol,
                            max_turnover=max_turnover,
                        )
                        plateau_candidate, plateau_flagged_df = choose_plateau_candidate(
                            robust_grid_df,
                            sharpe_col="sharpe",
                            turnover_col="mean_turnover",
                            sharpe_rel_tol=sharpe_rel_tol,
                            max_turnover=max_turnover,
                        )

                        st.markdown("#### Robust region detection")
                        if isinstance(plateau_summary_df, pd.DataFrame) and not plateau_summary_df.empty:
                            summary_map = {str(r["metric"]): r.get("value") for _, r in plateau_summary_df.iterrows() if "metric" in r}
                            r1, r2, r3, r4 = st.columns(4)
                            with r1:
                                st.metric("Best Sharpe", _format_num_or_dash(summary_map.get("best_sharpe")))
                            with r2:
                                st.metric("Plateau size", int(summary_map.get("plateau_size", 0) or 0))
                            with r3:
                                st.metric("Plateau exists", "Yes" if int(summary_map.get("plateau_exists", 0) or 0) == 1 else "No")
                            with r4:
                                st.metric("Sharpe cut", _format_num_or_dash(summary_map.get("plateau_sharpe_cut")))

                            r5, r6, r7 = st.columns(3)
                            with r5:
                                st.metric("Temp range", f"{_format_num_or_dash(summary_map.get('temperature_min'))} → {_format_num_or_dash(summary_map.get('temperature_max'))}")
                            with r6:
                                st.metric("Shrink range", f"{_format_num_or_dash(summary_map.get('weight_shrink_min'))} → {_format_num_or_dash(summary_map.get('weight_shrink_max'))}")
                            with r7:
                                st.metric("Inertia range", f"{_format_num_or_dash(summary_map.get('inertia_min'))} → {_format_num_or_dash(summary_map.get('inertia_max'))}")

                        if plateau_candidate is not None and isinstance(plateau_candidate, pd.Series):
                            st.markdown("**Stable plateau candidate**")
                            c1, c2, c3, c4, c5 = st.columns(5)
                            with c1:
                                st.metric("temperature", _format_num_or_dash(plateau_candidate.get("temperature")))
                            with c2:
                                st.metric("weight_shrink", _format_num_or_dash(plateau_candidate.get("weight_shrink")))
                            with c3:
                                st.metric("inertia", _format_num_or_dash(plateau_candidate.get("inertia")))
                            with c4:
                                st.metric("Sharpe", _format_num_or_dash(plateau_candidate.get("sharpe")))
                            with c5:
                                st.metric("Turnover", _format_pct_or_dash(plateau_candidate.get("mean_turnover")))

                        plateau_only_df = pd.DataFrame()
                        if isinstance(plateau_flagged_df, pd.DataFrame) and not plateau_flagged_df.empty and "plateau_region" in plateau_flagged_df.columns:
                            plateau_only_df = plateau_flagged_df[plateau_flagged_df["plateau_region"] == True].copy()
                            if not plateau_only_df.empty:
                                st.markdown("**Plateau-region candidates**")
                                show_cols = [c for c in ["temperature", "weight_shrink", "inertia", "sharpe", "mean_turnover", "plateau_region", "config_fingerprint"] if c in plateau_only_df.columns]
                                st.dataframe(plateau_only_df[show_cols].sort_values(["mean_turnover", "sharpe"], ascending=[True, False]).reset_index(drop=True), use_container_width=True)
                            else:
                                st.info("No robust plateau rows were found under the current tolerance / turnover rule.")

                        if bool(st.session_state.get("universe_robust_region_show_detail", False)) and isinstance(plateau_flagged_df, pd.DataFrame) and not plateau_flagged_df.empty:
                            st.markdown("**Detailed robust-region table**")
                            show_cols = [c for c in ["temperature", "weight_shrink", "inertia", "sharpe", "cagr", "mean_turnover", "plateau_region", "plateau_sharpe_cut", "best_sharpe", "config_fingerprint"] if c in plateau_flagged_df.columns]
                            st.dataframe(plateau_flagged_df[show_cols].sort_values(["plateau_region", "sharpe", "mean_turnover"], ascending=[False, False, True]).reset_index(drop=True), use_container_width=True)

                        with st.expander("Robust-region downloads", expanded=False):
                            if isinstance(robust_grid_df, pd.DataFrame) and not robust_grid_df.empty:
                                _download_dataframe_button(
                                    "Download robust-region base grid",
                                    robust_grid_df,
                                    "primary_robust_region_grid.csv",
                                    key="primary_robust_region_grid_download",
                                )
                            if isinstance(plateau_flagged_df, pd.DataFrame) and not plateau_flagged_df.empty:
                                _download_dataframe_button(
                                    "Download robust-region flagged table",
                                    plateau_flagged_df,
                                    "primary_robust_region_flagged.csv",
                                    key="primary_robust_region_flagged_download",
                                )
                            if isinstance(plateau_summary_df, pd.DataFrame) and not plateau_summary_df.empty:
                                _download_dataframe_button(
                                    "Download robust-region summary",
                                    plateau_summary_df,
                                    "primary_robust_region_summary.csv",
                                    key="primary_robust_region_summary_download",
                                )
                            if isinstance(plateau_only_df, pd.DataFrame) and not plateau_only_df.empty:
                                _download_dataframe_button(
                                    "Download plateau-only candidates",
                                    plateau_only_df,
                                    "primary_robust_region_plateau_only.csv",
                                    key="primary_robust_region_plateau_only_download",
                                )
                    except Exception as e:
                        st.warning(f"Could not build robust region detection: {e}")
                if bool(st.session_state.get("universe_breadth_estimation_enabled", False)):
                    try:
                        temp_grid = tuple(_parse_float_grid(
                            st.session_state.get("universe_breadth_estimation_temp_grid_text", ""),
                            default=[0.7, 1.0, 1.3],
                        ))
                        shrink_grid = tuple(_parse_float_grid(
                            st.session_state.get("universe_breadth_estimation_shrink_grid_text", ""),
                            default=[0.0, 0.05, 0.10],
                        ))
                        inertia_grid = tuple(_parse_float_grid(
                            st.session_state.get("universe_breadth_estimation_inertia_grid_text", ""),
                            default=[0.0, 0.2, 0.4],
                        ))
                        objective_name = str(st.session_state.get("universe_breadth_estimation_metric", "sharpe"))
                        n_bins = int(st.session_state.get("universe_breadth_estimation_bins", 3))
                        cfg_payload = config_to_dict(cfg)
                        breadth_grid_df = _run_param_grid_cached(
                            primary_engine_panel_df,
                            json.dumps(cfg_payload, sort_keys=True),
                            temp_grid,
                            shrink_grid,
                            inertia_grid,
                        )
                        be_overall_df, be_summary_df, be_bucket_df, be_detail_df = _build_breadth_vs_estimation_outputs(
                            breadth_grid_df,
                            objective_col=objective_name,
                            n_bins=n_bins,
                        )

                        st.markdown("#### Breadth vs estimation formal")
                        st.caption("Uses effective breadth metrics from the engine together with an estimation-load proxy inferred from parameter aggressiveness.")
                        if isinstance(be_overall_df, pd.DataFrame) and not be_overall_df.empty:
                            row = be_overall_df.iloc[0]
                            b1, b2, b3, b4 = st.columns(4)
                            with b1:
                                st.metric("Objective", str(row.get("objective", objective_name)))
                            with b2:
                                st.metric("Breadth metric", str(row.get("breadth_metric", "—")))
                            with b3:
                                st.metric("Best objective", _format_num_or_dash(row.get("best_objective")))
                            with b4:
                                st.metric("Median breadth", _format_num_or_dash(row.get("median_breadth")))
                            b5, b6, b7, b8 = st.columns(4)
                            with b5:
                                st.metric("Corr breadth↔objective", _format_num_or_dash(row.get("corr_breadth_objective")))
                            with b6:
                                st.metric("Corr breadth↔turnover", _format_num_or_dash(row.get("corr_breadth_turnover")))
                            with b7:
                                st.metric("Corr est.load↔objective", _format_num_or_dash(row.get("corr_estimationload_objective")))
                            with b8:
                                st.metric("Corr est.load↔turnover", _format_num_or_dash(row.get("corr_estimationload_turnover")))

                        if isinstance(be_bucket_df, pd.DataFrame) and not be_bucket_df.empty:
                            st.markdown("**Breadth buckets**")
                            st.dataframe(be_bucket_df, use_container_width=True)

                        if isinstance(be_detail_df, pd.DataFrame) and not be_detail_df.empty:
                            st.markdown("**Balanced candidates**")
                            head_cols = [c for c in [
                                "temperature", "weight_shrink", "inertia", objective_name,
                                "mean_effective_breadth", "mean_effective_risk_bets", "mean_turnover",
                                "estimation_load_proxy", "breadth_estimation_balance_score", "config_fingerprint"
                            ] if c in be_detail_df.columns]
                            st.dataframe(be_detail_df[head_cols].head(12), use_container_width=True)

                        if bool(st.session_state.get("universe_breadth_estimation_show_detail", False)) and isinstance(be_summary_df, pd.DataFrame) and not be_summary_df.empty:
                            st.markdown("**Parameter-level breadth/estimation summary**")
                            st.dataframe(be_summary_df, use_container_width=True)
                            if isinstance(be_detail_df, pd.DataFrame) and not be_detail_df.empty:
                                with st.expander("Detailed breadth/estimation table", expanded=False):
                                    st.dataframe(be_detail_df, use_container_width=True)

                        with st.expander("Breadth vs estimation downloads", expanded=False):
                            if isinstance(breadth_grid_df, pd.DataFrame) and not breadth_grid_df.empty:
                                _download_dataframe_button(
                                    "Download breadth/estimation base grid",
                                    breadth_grid_df,
                                    "primary_breadth_estimation_grid.csv",
                                    key="primary_breadth_estimation_grid_download",
                                )
                            if isinstance(be_bucket_df, pd.DataFrame) and not be_bucket_df.empty:
                                _download_dataframe_button(
                                    "Download breadth buckets",
                                    be_bucket_df,
                                    "primary_breadth_estimation_buckets.csv",
                                    key="primary_breadth_estimation_buckets_download",
                                )
                            if isinstance(be_summary_df, pd.DataFrame) and not be_summary_df.empty:
                                _download_dataframe_button(
                                    "Download breadth/estimation parameter summary",
                                    be_summary_df,
                                    "primary_breadth_estimation_summary.csv",
                                    key="primary_breadth_estimation_summary_download",
                                )
                            if isinstance(be_detail_df, pd.DataFrame) and not be_detail_df.empty:
                                _download_dataframe_button(
                                    "Download breadth/estimation detailed candidates",
                                    be_detail_df,
                                    "primary_breadth_estimation_candidates.csv",
                                    key="primary_breadth_estimation_candidates_download",
                                )
                    except Exception as e:
                        st.warning(f"Could not build breadth vs estimation formal: {e}")
                if bool(st.session_state.get("universe_alpha_sweep_enabled", False)):
                    try:
                        alpha_grid = tuple(_parse_float_grid(
                            st.session_state.get("universe_alpha_sweep_grid_text", ""),
                            default=[0.5, 0.75, 1.0, 1.25, 1.5],
                        ))
                        cfg_payload = config_to_dict(cfg)
                        alpha_df = _run_alpha_sweep_cached(
                            primary_engine_panel_df,
                            json.dumps(cfg_payload, sort_keys=True),
                            alpha_grid,
                        )

                        st.markdown("#### α sweep (sigma_power_alpha)")
                        st.info("α controls the volatility penalty inside the engine scoring rule. In the current allocator, the cross-sectional score is approximately computed as μ / σ^α before the allocation step. Higher α makes the engine more conservative toward volatile assets; lower α makes it more tolerant of volatility.")
                        if isinstance(alpha_df, pd.DataFrame) and not alpha_df.empty:
                            sort_cols = [c for c in ["sharpe", "information_ratio", "cagr"] if c in alpha_df.columns]
                            if sort_cols:
                                alpha_df = alpha_df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)

                            best_row = alpha_df.iloc[0]
                            a1, a2, a3, a4 = st.columns(4)
                            with a1:
                                st.metric("Best α", _format_num_or_dash(best_row.get("sigma_power_alpha")))
                            with a2:
                                st.metric("Best Sharpe", _format_num_or_dash(best_row.get("sharpe")))
                            with a3:
                                st.metric("Best CAGR", _format_pct_or_dash(best_row.get("cagr")))
                            with a4:
                                st.metric("Best turnover", _format_pct_or_dash(best_row.get("mean_turnover")))

                            focus_cols = [
                                "sigma_power_alpha",
                                "sharpe",
                                "cagr",
                                "annual_volatility",
                                "max_drawdown",
                                "mean_turnover",
                                "active_return_annual",
                                "tracking_error_annual",
                                "information_ratio",
                                "mean_effective_breadth",
                                "mean_diversification_ratio",
                                "mean_effective_risk_bets",
                                "mean_active_assets",
                                "config_fingerprint",
                            ]
                            focus_cols = [c for c in focus_cols if c in alpha_df.columns]
                            st.dataframe(alpha_df[focus_cols], use_container_width=True)

                            with st.expander("α sweep diagnostics / downloads", expanded=False):
                                _download_dataframe_button(
                                    "Download α sweep results",
                                    alpha_df,
                                    "primary_alpha_sweep.csv",
                                    key="primary_alpha_sweep_download",
                                )
                                if bool(st.session_state.get("universe_alpha_sweep_show_frontier", True)):
                                    try:
                                        alpha_frontier = build_sharpe_turnover_frontier(alpha_df, sharpe_col="sharpe", turnover_col="mean_turnover")
                                        frontier_only = alpha_frontier[alpha_frontier["on_frontier"] == True].copy() if "on_frontier" in alpha_frontier.columns else alpha_frontier.copy()
                                        st.markdown("**α-sweep frontier snapshot**")
                                        frontier_cols = [c for c in ["sigma_power_alpha", "sharpe", "mean_turnover", "frontier_rank", "on_frontier"] if c in frontier_only.columns]
                                        if frontier_cols:
                                            st.dataframe(frontier_only[frontier_cols], use_container_width=True)
                                    except Exception:
                                        pass
                        else:
                            st.info("α sweep produced an empty result table.")
                    except Exception as e:
                        st.warning(f"Could not build α sweep: {e}")

                if bool(st.session_state.get("universe_frontier_enabled", False)):
                    try:
                        temp_grid = tuple(_parse_float_grid(
                            st.session_state.get("universe_frontier_temp_grid_text", ""),
                            default=[0.7, 1.0, 1.3],
                        ))
                        shrink_grid = tuple(_parse_float_grid(
                            st.session_state.get("universe_frontier_shrink_grid_text", ""),
                            default=[0.0, 0.05, 0.10],
                        ))
                        inertia_grid = tuple(_parse_float_grid(
                            st.session_state.get("universe_frontier_inertia_grid_text", ""),
                            default=[0.0, 0.2, 0.4],
                        ))

                        cfg_payload = config_to_dict(cfg)
                        frontier_df = _run_sharpe_turnover_frontier_cached(
                            primary_engine_panel_df,
                            json.dumps(cfg_payload, sort_keys=True),
                            temp_grid,
                            shrink_grid,
                            inertia_grid,
                        )

                        st.markdown("#### Sharpe vs turnover frontier")
                        if isinstance(frontier_df, pd.DataFrame) and not frontier_df.empty:
                            chosen, frontier_view = choose_sharpe_turnover_candidate(
                                frontier_df,
                                max_turnover=float(st.session_state.get("universe_frontier_max_turnover", 0.35)),
                            )

                            if chosen is not None:
                                f1, f2, f3, f4 = st.columns(4)
                                with f1:
                                    st.metric("Chosen Sharpe", _format_num_or_dash(chosen.get("sharpe")))
                                with f2:
                                    st.metric("Chosen turnover", _format_pct_or_dash(chosen.get("mean_turnover")))
                                with f3:
                                    st.metric("Chosen temperature", _format_num_or_dash(chosen.get("temperature")))
                                with f4:
                                    st.metric("Chosen shrink / inertia", f"{chosen.get('weight_shrink', '—')} / {chosen.get('inertia', '—')}")

                            focus_cols = [
                                "temperature", "weight_shrink", "inertia",
                                "sharpe", "mean_turnover", "cagr", "max_drawdown",
                                "annual_volatility", "active_return_annual", "information_ratio",
                                "on_frontier", "frontier_rank",
                            ]
                            focus_cols = [c for c in focus_cols if c in frontier_df.columns]
                            st.dataframe(frontier_df[focus_cols], use_container_width=True)

                            with st.expander("Frontier diagnostics / downloads", expanded=False):
                                if isinstance(frontier_view, pd.DataFrame) and not frontier_view.empty:
                                    fv = frontier_view.copy()
                                    if "on_frontier" in fv.columns:
                                        fv = fv[fv["on_frontier"] == True].copy()
                                    st.markdown("**Frontier-only view**")
                                    st.dataframe(fv[focus_cols], use_container_width=True)
                                _download_dataframe_button(
                                    "Download frontier grid",
                                    frontier_df,
                                    "primary_sharpe_turnover_frontier.csv",
                                    key="primary_sharpe_turnover_frontier_download",
                                )
                        else:
                            st.info("Sharpe vs turnover frontier could not produce a non-empty grid.")
                    except Exception as e:
                        st.warning(f"Could not build Sharpe vs turnover frontier: {e}")

                if bool(st.session_state.get("universe_adv_tuning_enabled", False)):
                    try:
                        temp_grid = tuple(_parse_float_grid(
                            st.session_state.get("universe_adv_tuning_temp_grid_text", ""),
                            default=[0.7, 1.0, 1.3],
                        ))
                        shrink_grid = tuple(_parse_float_grid(
                            st.session_state.get("universe_adv_tuning_shrink_grid_text", ""),
                            default=[0.0, 0.05, 0.10],
                        ))
                        inertia_grid = tuple(_parse_float_grid(
                            st.session_state.get("universe_adv_tuning_inertia_grid_text", ""),
                            default=[0.0, 0.2, 0.4],
                        ))
                        alpha_grid = tuple(_parse_float_grid(
                            st.session_state.get("universe_adv_tuning_alpha_grid_text", ""),
                            default=[0.75, 1.0, 1.25],
                        ))
                        topk_raw = str(st.session_state.get("universe_adv_tuning_topk_grid_text", "") or "").replace(";", ",")
                        topk_vals = []
                        if topk_raw.strip():
                            seen_topk = set()
                            for part in topk_raw.split(","):
                                token = str(part).strip()
                                if not token:
                                    continue
                                try:
                                    kval = int(float(token))
                                except Exception:
                                    continue
                                if kval not in seen_topk:
                                    seen_topk.add(kval)
                                    topk_vals.append(kval)

                        raw_param_space = {
                            "temperature": list(temp_grid),
                            "weight_shrink": list(shrink_grid),
                            "inertia": list(inertia_grid),
                            "sigma_power_alpha": list(alpha_grid),
                        }
                        if topk_vals:
                            raw_param_space["top_k"] = list(topk_vals)

                        tuning_method = str(st.session_state.get("universe_adv_tuning_method", "bayesian_style"))
                        cfg_payload = config_to_dict(cfg)
                        param_space, param_space_audit_df, param_space_warnings = _prepare_tuning_param_space(
                            cfg_payload,
                            raw_param_space,
                            method=tuning_method,
                        )
                        global_seed_strategy_requested = str(st.session_state.get("universe_adv_global_seed_strategy", "hybrid"))
                        global_seed_strategy_effective = _resolve_global_search_method(
                            global_seed_strategy_requested,
                            float(st.session_state.get("universe_adv_global_exploitation_ratio", 0.50)),
                        )
                        global_param_schema = _apply_global_search_controls_to_schema(
                            param_space,
                            base_cfg_payload=cfg_payload,
                            categorical_breadth=str(st.session_state.get("universe_adv_global_categorical_breadth", "medium")),
                            exploitation_ratio=float(st.session_state.get("universe_adv_global_exploitation_ratio", 0.50)),
                        ) if str(tuning_method).strip().lower() == "global_multiobjective" else param_space

                        st.markdown("#### Global Search" if (workspace_mode == "Global Search" and str(tuning_method).strip().lower() == "global_multiobjective") else "#### Advanced tuning research")
                        if param_space_warnings:
                            st.info("Param-space sanitisation notes:")
                            for _w in param_space_warnings:
                                st.markdown(f"- {_w}")

                        if isinstance(param_space_audit_df, pd.DataFrame) and not param_space_audit_df.empty:
                            method_label = str(tuning_method).strip().lower()
                            kept_mask = param_space_audit_df.get("status", pd.Series(dtype=str)).astype(str).eq("keep")
                            drop_mask = param_space_audit_df.get("status", pd.Series(dtype=str)).astype(str).eq("drop")
                            kept_count = int(kept_mask.sum())
                            dropped_count = int(drop_mask.sum())
                            a1, a2, a3 = st.columns(3)
                            with a1:
                                st.metric("Kept tuning params", kept_count)
                            with a2:
                                st.metric("Dropped tuning params", dropped_count)
                            with a3:
                                st.metric("Effective tuning dims", len(param_space))

                            st.caption(f"Audit shown for tuning engine: {method_label}")

                            dropped_view = param_space_audit_df[drop_mask].copy()
                            kept_view = param_space_audit_df[kept_mask].copy()

                            if not dropped_view.empty:
                                st.warning("Some tuning dimensions were dropped because they are inactive, degenerate, or engine-dependent in the current context.")
                                reason_col = "engine_reason" if "engine_reason" in dropped_view.columns else "reason"
                                summary_parts = []
                                if reason_col in dropped_view.columns:
                                    reason_counts = (
                                        dropped_view[reason_col]
                                        .fillna("")
                                        .astype(str)
                                        .replace("", "unspecified")
                                        .value_counts()
                                    )
                                    for reason, count in reason_counts.items():
                                        summary_parts.append(f"{reason}: {int(count)}")
                                if summary_parts:
                                    st.caption("Dropped reasons — " + " | ".join(summary_parts))

                            k1, k2 = st.columns(2)
                            with k1:
                                if not kept_view.empty:
                                    kept_cols = [c for c in ["param", "current_value", "effective_n", "signal_mode", "probabilistic_mode"] if c in kept_view.columns]
                                    st.caption("Kept tuning dimensions")
                                    st.dataframe(kept_view[kept_cols] if kept_cols else kept_view, use_container_width=True, hide_index=True)
                            with k2:
                                if not dropped_view.empty:
                                    dropped_cols = [c for c in ["param", "reason", "engine_reason", "engine_controller", "current_value", "effective_n"] if c in dropped_view.columns]
                                    st.caption("Dropped tuning dimensions and reasons")
                                    st.dataframe(dropped_view[dropped_cols] if dropped_cols else dropped_view, use_container_width=True, hide_index=True)

                            with st.expander("Param-space activity audit", expanded=bool(dropped_count)):
                                audit_cols = [c for c in ["param", "status", "reason", "engine_status", "engine_reason", "engine_controller", "signal_mode", "probabilistic_mode", "current_value", "effective_n"] if c in param_space_audit_df.columns]
                                st.dataframe(param_space_audit_df[audit_cols] if audit_cols else param_space_audit_df, use_container_width=True)

                        effective_param_space = global_param_schema if str(tuning_method).strip().lower() == "global_multiobjective" else param_space
                        if not effective_param_space:
                            st.warning("Advanced tuning was skipped because no active tuning dimensions remain after param-space sanitisation.")
                        else:
                            global_resume_mode = str(st.session_state.get("universe_adv_global_resume_mode", "restart")).strip().lower()
                            global_signature = None
                            tuning_result = None
                            if str(tuning_method).strip().lower() == "global_multiobjective":
                                global_signature = json.dumps({
                                    "cfg_payload": cfg_payload,
                                    "param_schema": effective_param_space,
                                    "objective_names": list(st.session_state.get("universe_adv_tuning_nsga2_objectives", [])),
                                    "budget": int(st.session_state.get("universe_adv_global_total_budget", 120)),
                                    "seed_strategy": global_seed_strategy_effective,
                                }, sort_keys=True, default=str)
                                last_signature = st.session_state.get("universe_adv_global_last_signature")
                                last_budget = int(st.session_state.get("universe_adv_global_last_budget", 0) or 0)
                                if global_resume_mode == "resume" and last_signature == global_signature and last_budget >= int(st.session_state.get("universe_adv_global_total_budget", 120)):
                                    tuning_result = st.session_state.get("universe_adv_global_last_result")
                                    st.info("Global multiobjective search resumed from the last compatible result already stored in this session.")

                            if tuning_result is None:
                                tuning_result = _run_advanced_tuning_cached(
                                    primary_engine_panel_df,
                                    json.dumps(cfg_payload, sort_keys=True),
                                    tuning_method,
                                    str(st.session_state.get("universe_adv_tuning_objective", "sharpe")),
                                    int(st.session_state.get("universe_adv_tuning_n_trials", 20)),
                                    int(st.session_state.get("universe_adv_tuning_warmup", 5)),
                                    json.dumps(effective_param_space, sort_keys=True),
                                    json.dumps(list(st.session_state.get("universe_adv_tuning_nsga2_objectives", [])), sort_keys=True),
                                    int(st.session_state.get("universe_adv_tuning_nsga2_population_size", 24)),
                                    int(st.session_state.get("universe_adv_tuning_nsga2_generations", 8)),
                                    float(st.session_state.get("universe_adv_tuning_nsga2_mutation_rate", 0.15)),
                                    float(st.session_state.get("universe_adv_tuning_nsga2_crossover_rate", 0.90)),
                                    int(st.session_state.get("universe_adv_tuning_nsga2_seed", 42)),
                                    bool(st.session_state.get("universe_adv_tuning_recursive_enabled", False)),
                                    int(st.session_state.get("universe_adv_tuning_recursive_depth", 3)),
                                    float(st.session_state.get("universe_adv_tuning_recursive_shrink_factor", 0.50)),
                                    float(st.session_state.get("universe_adv_tuning_recursive_convergence_tol", 1e-3)),
                                    int(st.session_state.get("universe_adv_tuning_recursive_seed_frontier_size", 8)),
                                    int(st.session_state.get("universe_adv_global_total_budget", 120)),
                                    str(global_seed_strategy_effective),
                                )
                                if str(tuning_method).strip().lower() == "global_multiobjective":
                                    st.session_state["universe_adv_global_last_signature"] = global_signature
                                    st.session_state["universe_adv_global_last_budget"] = int(st.session_state.get("universe_adv_global_total_budget", 120))
                                    st.session_state["universe_adv_global_last_result"] = tuning_result

                            if str(tuning_method).strip().lower() == "global_multiobjective":
                                global_result = tuning_result if isinstance(tuning_result, dict) else {}
                                population_df = pd.DataFrame((global_result or {}).get("population_df", pd.DataFrame())).copy()
                                frontier_df = pd.DataFrame((global_result or {}).get("frontier_df", pd.DataFrame())).copy()
                                history_df = pd.DataFrame((global_result or {}).get("history_df", pd.DataFrame())).copy()
                                best_compromise_payload = _coerce_mapping(_coerce_mapping(global_result).get("best_compromise_payload", {}))
                                frontier_hypervolume = _safe_float((global_result or {}).get("frontier_hypervolume"))
                                search_method_effective = str((global_result or {}).get("search_method_effective", global_seed_strategy_effective))
                                coverage_df = _build_global_search_param_coverage_df(effective_param_space, population_df)
                                fingerprint_df = pd.DataFrame()
                                if isinstance(population_df, pd.DataFrame) and not population_df.empty and "candidate_fingerprint" in population_df.columns:
                                    fp_cols = [c for c in ["candidate_fingerprint", "sample_method", "status", "cache_hit"] if c in population_df.columns]
                                    fingerprint_df = population_df[fp_cols].drop_duplicates().reset_index(drop=True)

                                if isinstance(population_df, pd.DataFrame) and not population_df.empty:
                                    g1, g2, g3, g4 = st.columns(4)
                                    with g1:
                                        st.metric("Explored rows", int(len(population_df)))
                                    with g2:
                                        explored_fps = int(fingerprint_df["candidate_fingerprint"].nunique()) if not fingerprint_df.empty and "candidate_fingerprint" in fingerprint_df.columns else int(population_df.get("candidate_fingerprint", pd.Series(dtype=str)).astype(str).nunique())
                                        st.metric("Explored fingerprints", explored_fps)
                                    with g3:
                                        st.metric("Global frontier size", int(len(frontier_df)) if isinstance(frontier_df, pd.DataFrame) else 0)
                                    with g4:
                                        st.metric("Frontier hypervolume", _format_num_or_dash(frontier_hypervolume))

                                    st.caption(f"Global search method effective: {search_method_effective} | seed strategy requested: {global_seed_strategy_requested}")

                                    if best_compromise_payload:
                                        ac1, ac2 = st.columns([1.2, 1.0])
                                        with ac1:
                                            if st.button(
                                                "Apply best global compromise",
                                                key="universe_adv_tuning_apply_global_multiobjective",
                                                use_container_width=True,
                                            ):
                                                _queue_promote_cfg_payload_to_manual_state(best_compromise_payload, source_label="advanced tuning / global multiobjective")
                                                st.rerun()
                                        with ac2:
                                            st.caption("Best compromise available")
                                            st.write(f"`{str(best_compromise_payload.get('config_fingerprint', ''))[:12] or 'ready'}`")

                                    st.markdown("**Global frontier**")
                                    frontier_cols = [c for c in [
                                        "trial", "sample_method", "pareto_rank", "hypervolume_contribution", "hypervolume_contribution_share",
                                        "sharpe", "cagr", "max_drawdown", "mean_turnover", "diversification", "stability",
                                        "temperature", "weight_shrink", "inertia", "sigma_power_alpha", "top_k", "candidate_fingerprint",
                                    ] if c in frontier_df.columns]
                                    st.dataframe(frontier_df[frontier_cols] if frontier_cols else frontier_df, use_container_width=True, hide_index=True)

                                    fcol, ccol = st.columns(2)
                                    with fcol:
                                        st.markdown("**Explored fingerprints**")
                                        if not fingerprint_df.empty:
                                            st.dataframe(fingerprint_df, use_container_width=True, hide_index=True)
                                        else:
                                            st.info("No fingerprint table available.")
                                    with ccol:
                                        st.markdown("**Coverage por parámetro**")
                                        if not coverage_df.empty:
                                            st.dataframe(coverage_df, use_container_width=True, hide_index=True)
                                        else:
                                            st.info("No parameter coverage table available.")

                                    with st.expander("Global search diagnostics / downloads", expanded=False):
                                        st.markdown("**Effective global param schema**")
                                        st.json(effective_param_space)
                                        if isinstance(history_df, pd.DataFrame) and not history_df.empty:
                                            st.markdown("**Global search history**")
                                            st.dataframe(history_df, use_container_width=True)
                                            _download_dataframe_button(
                                                "Download global search history",
                                                history_df,
                                                "primary_global_multiobjective_history.csv",
                                                key="primary_global_multiobjective_history_download",
                                            )
                                        _download_dataframe_button(
                                            "Download global search population",
                                            population_df,
                                            "primary_global_multiobjective_population.csv",
                                            key="primary_global_multiobjective_population_download",
                                        )
                                        _download_dataframe_button(
                                            "Download global search frontier",
                                            frontier_df,
                                            "primary_global_multiobjective_frontier.csv",
                                            key="primary_global_multiobjective_frontier_download",
                                        )
                                        if not coverage_df.empty:
                                            _download_dataframe_button(
                                                "Download parameter coverage",
                                                coverage_df,
                                                "primary_global_multiobjective_param_coverage.csv",
                                                key="primary_global_multiobjective_param_coverage_download",
                                            )
                                else:
                                    st.info("Global multiobjective search did not produce a non-empty population table.")

                            elif str(tuning_method).strip().lower() == "nsga2":
                                nsga2_result = tuning_result if isinstance(tuning_result, dict) else {}
                                population_df = pd.DataFrame((nsga2_result or {}).get("population_df", pd.DataFrame())).copy()
                                frontier_df = pd.DataFrame((nsga2_result or {}).get("frontier_df", pd.DataFrame())).copy()
                                history_df = pd.DataFrame((nsga2_result or {}).get("history_df", pd.DataFrame())).copy()
                                round_summaries_df = pd.DataFrame((nsga2_result or {}).get("round_summaries_df", pd.DataFrame())).copy()
                                best_frontier_df = pd.DataFrame((nsga2_result or {}).get("best_frontier_df", pd.DataFrame())).copy()
                                best_population_df = pd.DataFrame((nsga2_result or {}).get("best_population_df", pd.DataFrame())).copy()
                                best_compromise_payload = _coerce_mapping(_coerce_mapping(nsga2_result).get("best_compromise_payload", {}))
                                recursive_best_round = (nsga2_result or {}).get("best_round")
                                recursive_converged = bool((nsga2_result or {}).get("converged", False))
                                recursive_convergence_reason = str((nsga2_result or {}).get("convergence_reason") or "")
                                recursive_hv = (nsga2_result or {}).get("frontier_hypervolume")
                                nsga2_objectives = [str(x) for x in (st.session_state.get("universe_adv_tuning_nsga2_objectives", []) or []) if str(x)]
                                nsga2_specs = {
                                    str(obj): {"sense": ("min" if str(obj) in {"max_drawdown", "mean_turnover"} else "max")}
                                    for obj in nsga2_objectives
                                }

                                if isinstance(population_df, pd.DataFrame) and not population_df.empty:
                                    nsga2_policy_col, nsga2_profile_col = st.columns([1.2, 0.95])
                                    nsga2_policy_options = _get_selection_policy_options_for_current_mode()
                                    nsga2_policy_locked = _is_simple_mode_active()
                                    with nsga2_policy_col:
                                        if nsga2_policy_locked:
                                            nsga2_selection_policy = "fixed_composite_score"
                                            st.session_state["universe_adv_tuning_nsga2_selection_policy"] = nsga2_selection_policy
                                            st.text_input(
                                                "selection policy",
                                                value=nsga2_selection_policy,
                                                disabled=True,
                                                key="universe_adv_tuning_nsga2_selection_policy_locked_display",
                                                help="Simple / Auto mode keeps NSGA-II selection on the fixed composite score chooser.",
                                            )
                                        else:
                                            nsga2_selection_policy = st.selectbox(
                                                "selection policy",
                                                nsga2_policy_options,
                                                key="universe_adv_tuning_nsga2_selection_policy",
                                                help="Controls the default NSGA-II Pareto candidate selection and the row highlighted for application.",
                                            )
                                    nsga2_composite_profile: Any = DEFAULT_COMPOSITE_PROFILE
                                    with nsga2_profile_col:
                                        if str(nsga2_selection_policy).strip().lower() == "fixed_composite_score":
                                            if nsga2_policy_locked:
                                                nsga2_composite_profile = _resolve_simple_mode_composite_profile()
                                                st.session_state["universe_adv_tuning_nsga2_composite_profile"] = nsga2_composite_profile
                                                st.text_input(
                                                    "composite profile",
                                                    value=nsga2_composite_profile,
                                                    disabled=True,
                                                    key="universe_adv_tuning_nsga2_composite_profile_locked_display",
                                                    help="Simple / Auto mode resolves the composite profile from the semantic style preset.",
                                                )
                                            else:
                                                nsga2_composite_profile = st.selectbox(
                                                    "composite profile",
                                                    COMPOSITE_PROFILE_OPTIONS,
                                                    index=COMPOSITE_PROFILE_OPTIONS.index(DEFAULT_COMPOSITE_PROFILE),
                                                    key="universe_adv_tuning_nsga2_composite_profile",
                                                    help="Profile used by the fixed composite score chooser.",
                                                )
                                                nsga2_manual_enabled, nsga2_manual_weights = _render_manual_composite_weights_ui(
                                                    "universe_adv_tuning_nsga2",
                                                    str(nsga2_composite_profile),
                                                )
                                                if nsga2_manual_enabled and nsga2_manual_weights:
                                                    nsga2_composite_profile = dict(nsga2_manual_weights)
                                        else:
                                            st.caption("")
                                    frontier_source_df = best_frontier_df if isinstance(best_frontier_df, pd.DataFrame) and not best_frontier_df.empty else frontier_df
                                    frontier_for_selection = frontier_source_df if isinstance(frontier_source_df, pd.DataFrame) and not frontier_source_df.empty else population_df
                                    auto_selected, frontier_for_selection = _choose_pareto_solution_app(
                                        frontier_for_selection,
                                        policy=nsga2_selection_policy,
                                        objective_specs=nsga2_specs or None,
                                        composite_profile=nsga2_composite_profile,
                                    ) if not frontier_for_selection.empty else ({}, frontier_for_selection)
                                    best_row = _coerce_mapping(auto_selected)
                                    hypervolume_total = _safe_float(frontier_for_selection.get("hypervolume_total").iloc[0]) if isinstance(frontier_for_selection, pd.DataFrame) and not frontier_for_selection.empty and "hypervolume_total" in frontier_for_selection.columns else np.nan
                                    t1, t2, t3, t4, t5 = st.columns(5)
                                    with t1:
                                        st.metric("Final population", int(len(population_df)))
                                    with t2:
                                        st.metric("Frontier size", int(len(frontier_df)) if isinstance(frontier_df, pd.DataFrame) else 0)
                                    with t3:
                                        st.metric("Best Sharpe", _format_num_or_dash(best_row.get("sharpe")))
                                    with t4:
                                        st.metric("Best CAGR", _format_pct_or_dash(best_row.get("cagr")))
                                    with t5:
                                        st.metric("Hypervolume total", _format_num_or_dash(hypervolume_total))
                                    st.caption(_build_hypervolume_caption(frontier_for_selection))

                                    if isinstance(round_summaries_df, pd.DataFrame) and not round_summaries_df.empty:
                                        st.markdown("**Recursive search summary**")
                                        rr1, rr2, rr3, rr4 = st.columns(4)
                                        best_round_display = recursive_best_round if recursive_best_round is not None else "—"
                                        requested_seed_frontier_size = int(st.session_state.get("universe_adv_tuning_recursive_seed_frontier_size", 8))
                                        effective_seed_frontier_size = requested_seed_frontier_size
                                        round_view = round_summaries_df.copy()
                                        if "seed_frontier_size_effective" in round_view.columns:
                                            eff_series = pd.to_numeric(round_view["seed_frontier_size_effective"], errors="coerce").dropna()
                                            if not eff_series.empty:
                                                effective_seed_frontier_size = int(eff_series.iloc[-1])
                                        elif "seed_count" in round_view.columns:
                                            eff_series = pd.to_numeric(round_view["seed_count"], errors="coerce").dropna()
                                            if not eff_series.empty:
                                                effective_seed_frontier_size = int(eff_series.iloc[-1])
                                        with rr1:
                                            st.metric("Rounds", int(len(round_summaries_df)))
                                        with rr2:
                                            st.metric("Best round", best_round_display)
                                        with rr3:
                                            st.metric("Recursive converged", "Yes" if recursive_converged else "No")
                                        with rr4:
                                            st.metric("Stop reason", recursive_convergence_reason or "—")

                                        cfg1, cfg2, cfg3, cfg4 = st.columns(4)
                                        with cfg1:
                                            st.metric("Seed frontier size (requested)", requested_seed_frontier_size)
                                        with cfg2:
                                            st.metric("Seed frontier size (effective)", effective_seed_frontier_size)
                                        with cfg3:
                                            st.metric("Shrink factor", f"{float(st.session_state.get('universe_adv_tuning_recursive_shrink_factor', 0.50)):.2f}")
                                        with cfg4:
                                            st.metric("Convergence tol", f"{float(st.session_state.get('universe_adv_tuning_recursive_convergence_tol', 1e-3)):.4f}")

                                        if "frontier_hypervolume" not in round_view.columns and "hypervolume_total" in round_view.columns:
                                            round_view["frontier_hypervolume"] = round_view["hypervolume_total"]
                                        round_cols = [
                                            "round",
                                            "frontier_size",
                                            "frontier_hypervolume",
                                            "delta_hv_abs",
                                            "delta_hv_rel",
                                            "frontier_spread",
                                            "seed_frontier_size_requested",
                                            "seed_frontier_size_effective",
                                            "seed_count",
                                            "stop_reason",
                                        ]
                                        round_cols = [c for c in round_cols if c in round_view.columns]
                                        st.caption("Frontier by round / hypervolume by round")
                                        st.dataframe(round_view[round_cols] if round_cols else round_view, use_container_width=True, hide_index=True)

                                    candidate_options = []
                                    candidate_lookup = {}
                                    for pos, (_, row) in enumerate(frontier_for_selection.reset_index(drop=True).iterrows()):
                                        label = _make_pareto_candidate_label(row, pos)
                                        candidate_options.append(label)
                                        candidate_lookup[label] = _row_to_dict(row)

                                    auto_fp = str(auto_selected.get("effective_config_fingerprint") or auto_selected.get("config_fingerprint") or "")
                                    default_idx = 0
                                    if auto_fp and candidate_options:
                                        for i, label in enumerate(candidate_options):
                                            row_fp = str(candidate_lookup[label].get("effective_config_fingerprint") or candidate_lookup[label].get("config_fingerprint") or "")
                                            if row_fp and row_fp == auto_fp:
                                                default_idx = i
                                                break

                                    selected_label = st.selectbox(
                                        "NSGA-II Pareto candidate row",
                                        candidate_options,
                                        index=default_idx if candidate_options else None,
                                        key="universe_adv_tuning_nsga2_candidate_row",
                                        help="Select one Pareto-efficient NSGA-II candidate to apply.",
                                    ) if candidate_options else None
                                    selected_row = _coerce_mapping(candidate_lookup.get(selected_label, auto_selected))
                                    selected_payload = _payload_from_json(selected_row.get("effective_cfg_payload_json")) if selected_row else {}
                                    if not selected_payload:
                                        selected_payload = _coerce_mapping(selected_row.get("effective_cfg_payload"))
                                    if not selected_payload and best_compromise_payload:
                                        selected_payload = _coerce_mapping(best_compromise_payload)
                                    selected_fingerprint = str(selected_row.get("effective_config_fingerprint") or selected_row.get("config_fingerprint") or "")

                                    apply_col, picker_info_col = st.columns([1.2, 1.0])
                                    with apply_col:
                                        if st.button(
                                            "Apply selected Pareto solution",
                                            key="universe_adv_tuning_apply_nsga2_selected",
                                            use_container_width=True,
                                            disabled=not bool(selected_payload),
                                            help="Promote the selected NSGA-II Pareto solution into the active manual config for the next run.",
                                        ):
                                            _queue_promote_cfg_payload_to_manual_state(selected_payload, source_label="advanced tuning / nsga2 / pareto")
                                            st.session_state["universe_adv_tuning_last_applied_fingerprint"] = selected_fingerprint
                                            st.rerun()
                                    with picker_info_col:
                                        st.caption("Selected NSGA-II Pareto solution")
                                        st.write(
                                            f"Fingerprint `{selected_fingerprint[:10] if selected_fingerprint else '—'}` | policy `{nsga2_selection_policy}` | objectives `{', '.join([_format_pareto_objective_label(x) for x in nsga2_objectives])}`"
                                        )

                                    winner_cols = [
                                        "trial", "generation", "pareto_rank", "hypervolume_contribution",
                                        "sharpe", "cagr", "max_drawdown", "mean_turnover",
                                        "diversification", "stability", "effective_config_fingerprint",
                                    ]
                                    winner_cols = [c for c in winner_cols if c in selected_row]
                                    st.markdown("**winner final**")
                                    if selected_row:
                                        st.dataframe(pd.DataFrame([{k: selected_row.get(k) for k in winner_cols}] if winner_cols else [selected_row]), use_container_width=True, hide_index=True)
                                    else:
                                        st.info("No final winner row is available yet.")

                                    pop_cols = [
                                        "trial", "generation", "candidate_origin", "pareto_rank", "crowding_distance",
                                        "objective_value", "sharpe", "cagr", "max_drawdown", "mean_turnover",
                                        "diversification", "stability", "annual_volatility",
                                        "temperature", "weight_shrink", "inertia", "sigma_power_alpha", "top_k",
                                        "effective_config_fingerprint", "status",
                                    ]
                                    pop_cols = [c for c in pop_cols if c in population_df.columns]
                                    st.markdown("**NSGA-II final population**")
                                    st.dataframe(population_df[pop_cols] if pop_cols else population_df, use_container_width=True, hide_index=True)

                                    frontier_cols = [
                                        "trial", "generation", "pareto_rank", "crowding_distance", "dominates_count", "dominated_by_count",
                                        "hypervolume_contribution", "hypervolume_contribution_share",
                                        "objective_value", "sharpe", "cagr", "max_drawdown", "mean_turnover",
                                        "diversification", "stability", "annual_volatility",
                                        "temperature", "weight_shrink", "inertia", "sigma_power_alpha", "top_k",
                                        "effective_config_fingerprint", "status",
                                    ]
                                    frontier_cols = [c for c in frontier_cols if c in frontier_for_selection.columns]
                                    st.markdown("**NSGA-II Pareto frontier**")
                                    st.dataframe(frontier_for_selection[frontier_cols] if frontier_cols else frontier_for_selection, use_container_width=True, hide_index=True)

                                    with st.expander("NSGA-II diagnostics / downloads", expanded=False):
                                        st.markdown("**Effective tuning param space**")
                                        st.json(param_space)
                                        if isinstance(history_df, pd.DataFrame) and not history_df.empty:
                                            st.markdown("**NSGA-II history**")
                                            st.dataframe(history_df, use_container_width=True)
                                            _download_dataframe_button(
                                                "Download NSGA-II history",
                                                history_df,
                                                "primary_advanced_tuning_nsga2_history.csv",
                                                key="primary_advanced_tuning_nsga2_history_download",
                                            )
                                        if isinstance(round_summaries_df, pd.DataFrame) and not round_summaries_df.empty:
                                            st.markdown("**Recursive round summaries**")
                                            st.dataframe(round_summaries_df, use_container_width=True)
                                            _download_dataframe_button(
                                                "Download recursive round summaries",
                                                round_summaries_df,
                                                "primary_advanced_tuning_recursive_round_summaries.csv",
                                                key="primary_advanced_tuning_recursive_round_summaries_download",
                                            )
                                        _download_dataframe_button(
                                            "Download NSGA-II final population",
                                            population_df,
                                            "primary_advanced_tuning_nsga2_population.csv",
                                            key="primary_advanced_tuning_nsga2_population_download",
                                        )
                                        _download_dataframe_button(
                                            "Download NSGA-II frontier",
                                            frontier_for_selection,
                                            "primary_advanced_tuning_nsga2_frontier.csv",
                                            key="primary_advanced_tuning_nsga2_frontier_download",
                                        )
                                else:
                                    st.info("NSGA-II did not produce a non-empty final population table.")

                            elif isinstance(tuning_result, pd.DataFrame) and not tuning_result.empty:
                                trials_df = tuning_result
                                trials_view = build_tuning_trials_table(trials_df)
                                best_trial = choose_best_tuning_trial(
                                    trials_view if isinstance(trials_view, pd.DataFrame) and not trials_view.empty else trials_df,
                                    objective_col="objective_value",
                                )
                                if isinstance(best_trial, pd.Series) and not best_trial.empty:
                                    t1, t2, t3, t4 = st.columns(4)
                                    with t1:
                                        st.metric("Best objective", _format_num_or_dash(best_trial.get("objective_value")))
                                    with t2:
                                        st.metric("Best Sharpe", _format_num_or_dash(best_trial.get("sharpe")))
                                    with t3:
                                        st.metric("Best CAGR", _format_pct_or_dash(best_trial.get("cagr")))
                                    with t4:
                                        st.metric("Best turnover", _format_pct_or_dash(best_trial.get("mean_turnover")))

                                    pcols = [c for c in ["temperature", "weight_shrink", "inertia", "sigma_power_alpha", "top_k"] if c in best_trial.index]
                                    if pcols:
                                        st.caption("Best candidate parameters")
                                        st.json({c: best_trial.get(c) for c in pcols})

                                if str(tuning_method).strip().lower() == "surrogate":
                                    surrogate_summary = _summarize_surrogate_trials(trials_df)
                                    if surrogate_summary:
                                        s1, s2, s3, s4 = st.columns(4)
                                        with s1:
                                            st.metric("Surrogate used rate", _format_pct_or_dash(surrogate_summary.get("surrogate_used_rate")))
                                        with s2:
                                            st.metric("Surrogate ready rate", _format_pct_or_dash(surrogate_summary.get("surrogate_ready_rate")))
                                        with s3:
                                            st.metric("Eff. warmup", _format_num_or_dash(surrogate_summary.get("surrogate_effective_warmup")))
                                        with s4:
                                            st.metric("Top surrogate reason", str(surrogate_summary.get("surrogate_top_reason", "-")))

                                        st.caption(
                                            f"Surrogate diagnostics: feature_dim={_format_num_or_dash(surrogate_summary.get('surrogate_feature_dim'))}, "
                                            f"n_fit={_format_num_or_dash(surrogate_summary.get('surrogate_n_fit'))}, "
                                            f"candidate_pool={_format_num_or_dash(surrogate_summary.get('surrogate_candidate_pool_size'))}, "
                                            f"median resid std={_format_num_or_dash(surrogate_summary.get('surrogate_resid_std'))}"
                                        )

                                if str(tuning_method).strip().lower() == "random":
                                    random_summary = _summarize_random_trials(trials_df)
                                    if random_summary:
                                        r1, r2, r3, r4 = st.columns(4)
                                        with r1:
                                            st.metric("Random space dim", _format_num_or_dash(random_summary.get("random_search_space_dim")))
                                        with r2:
                                            st.metric("Unique samples", _format_num_or_dash(random_summary.get("random_search_unique_sample_count")))
                                        with r3:
                                            st.metric("Duplicate rate", _format_pct_or_dash(random_summary.get("random_search_duplicate_rate")))
                                        with r4:
                                            st.metric("Dropped params", _format_num_or_dash(random_summary.get("random_search_dropped_param_count")))

                                        st.caption(
                                            f"Random-search diagnostics: raw_param_count={_format_num_or_dash(random_summary.get('random_search_raw_param_count'))}, "
                                            f"space_status={str(random_summary.get('random_search_space_status', '-'))}, "
                                            f"active_variation={_format_pct_or_dash(random_summary.get('random_search_has_active_variation'))}"
                                        )

                                if str(tuning_method).strip().lower() == "bayesian_style":
                                    bayesian_summary = _summarize_bayesian_trials(trials_df)
                                    if bayesian_summary:
                                        b1, b2, b3, b4 = st.columns(4)
                                        with b1:
                                            st.metric("Bayesian space dim", _format_num_or_dash(bayesian_summary.get("bayesian_search_space_dim")))
                                        with b2:
                                            st.metric("Eff. warmup", _format_num_or_dash(bayesian_summary.get("bayesian_search_effective_warmup")))
                                        with b3:
                                            st.metric("Incumbent used", _format_pct_or_dash(bayesian_summary.get("bayesian_search_used_incumbent_rate")))
                                        with b4:
                                            st.metric("Duplicate rate", _format_pct_or_dash(bayesian_summary.get("bayesian_search_duplicate_rate")))

                                        st.caption(
                                            f"Bayesian diagnostics: raw_param_count={_format_num_or_dash(bayesian_summary.get('bayesian_search_raw_param_count'))}, "
                                            f"space_status={str(bayesian_summary.get('bayesian_search_space_status', '-'))}, "
                                            f"phase={str(bayesian_summary.get('bayesian_search_phase', '-'))}, "
                                            f"active_variation={_format_pct_or_dash(bayesian_summary.get('bayesian_search_has_active_variation'))}, "
                                            f"incumbent_updates={_format_num_or_dash(bayesian_summary.get('bayesian_search_incumbent_updates'))}"
                                        )

                                if str(tuning_method).strip().lower() == "optuna":
                                    optuna_summary = _summarize_optuna_trials(trials_df)
                                    if optuna_summary:
                                        o1, o2, o3, o4 = st.columns(4)
                                        with o1:
                                            st.metric("Optuna available", "Yes" if optuna_summary.get("optuna_any_available") else "No")
                                        with o2:
                                            st.metric("Fallback active", "Yes" if optuna_summary.get("optuna_any_fallback") else "No")
                                        with o3:
                                            st.metric("Optuna space dim", _format_num_or_dash(optuna_summary.get("optuna_space_dim")))
                                        with o4:
                                            st.metric("Best value", _format_num_or_dash(optuna_summary.get("optuna_best_value")))

                                        effective_method = str(optuna_summary.get("optuna_effective_method", optuna_summary.get("method", "-")))
                                        fallback_reason = str(optuna_summary.get("optuna_fallback_reason", "-"))
                                        requested_direction = str(optuna_summary.get("optuna_requested_direction", "-"))
                                        study_direction = str(optuna_summary.get("optuna_study_direction", "-"))
                                        space_status = str(optuna_summary.get("optuna_space_status", "-"))

                                        if optuna_summary.get("optuna_any_fallback"):
                                            st.warning(
                                                f"Optuna real no estuvo disponible y este tuning corrió en fallback bayesian-style. "
                                                f"effective_method={effective_method}; fallback_reason={fallback_reason}."
                                            )
                                        elif optuna_summary.get("optuna_any_available"):
                                            st.success(
                                                f"Este tuning sí usó Optuna real. effective_method={effective_method}; "
                                                f"requested_direction={requested_direction}; study_direction={study_direction}."
                                            )
                                        else:
                                            st.info(
                                                f"Optuna diagnostics: effective_method={effective_method}; fallback_reason={fallback_reason}."
                                            )

                                        st.caption(
                                            f"Optuna diagnostics: raw_param_count={_format_num_or_dash(optuna_summary.get('optuna_raw_param_count'))}, "
                                            f"space_status={space_status}, "
                                            f"requested_direction={requested_direction}, "
                                            f"study_direction={study_direction}, "
                                            f"fallback_active={_format_pct_or_dash(optuna_summary.get('optuna_fallback_active_rate'))}, "
                                            f"active_variation={_format_pct_or_dash(optuna_summary.get('optuna_has_active_variation_rate'))}"
                                        )

                                focus_cols = [
                                    "trial", "objective_value", "sharpe", "cagr", "annual_volatility", "max_drawdown",
                                    "mean_turnover", "information_ratio", "temperature", "weight_shrink", "inertia",
                                    "sigma_power_alpha", "top_k", "status", "surrogate_used", "surrogate_ready",
                                    "surrogate_reason", "surrogate_effective_warmup",
                                    "random_search_space_dim", "random_search_raw_param_count",
                                    "random_search_dropped_param_count", "random_search_has_active_variation",
                                    "random_search_duplicate_sample", "random_search_unique_sample_count",
                                    "random_search_duplicate_rate", "random_search_space_status",
                                    "bayesian_search_space_dim", "bayesian_search_raw_param_count",
                                    "bayesian_search_dropped_param_count", "bayesian_search_has_active_variation",
                                    "bayesian_search_requested_warmup", "bayesian_search_effective_warmup",
                                    "bayesian_search_phase", "bayesian_search_used_incumbent",
                                    "bayesian_search_incumbent_improved", "bayesian_search_incumbent_updates",
                                    "bayesian_search_duplicate_sample", "bayesian_search_duplicate_rate",
                                    "bayesian_search_space_status", "bayesian_search_best_val_so_far",
                                    "optuna_available", "optuna_requested_direction",
                                    "optuna_fallback_to_bayesian_style_requested", "optuna_fallback_active",
                                    "optuna_effective_method", "optuna_requested_trial_count",
                                    "optuna_raw_param_count", "optuna_space_dim",
                                    "optuna_dropped_param_count", "optuna_space_status",
                                    "optuna_has_active_variation", "optuna_fallback_reason",
                                    "optuna_trial_number", "optuna_study_direction",
                                    "optuna_best_value", "optuna_best_trial_number",
                                    "config_fingerprint",
                                ]
                                table_to_show = trials_view if isinstance(trials_view, pd.DataFrame) and not trials_view.empty else trials_df
                                focus_cols = [c for c in focus_cols if c in table_to_show.columns]
                                if focus_cols:
                                    st.dataframe(table_to_show[focus_cols], use_container_width=True)
                                else:
                                    st.dataframe(table_to_show, use_container_width=True)

                                with st.expander("Advanced tuning diagnostics / downloads", expanded=False):
                                    if isinstance(param_space_audit_df, pd.DataFrame) and not param_space_audit_df.empty:
                                        st.markdown("**Param-space audit**")
                                        st.dataframe(param_space_audit_df, use_container_width=True)
                                    st.markdown("**Effective tuning param space**")
                                    st.json(param_space)
                                    _download_dataframe_button(
                                        "Download tuning trials",
                                        table_to_show,
                                        "primary_advanced_tuning_trials.csv",
                                        key="primary_advanced_tuning_trials_download",
                                    )
                                    if isinstance(param_space_audit_df, pd.DataFrame) and not param_space_audit_df.empty:
                                        _download_dataframe_button(
                                            "Download param-space audit",
                                            param_space_audit_df,
                                            "primary_advanced_tuning_param_space_audit.csv",
                                            key="primary_advanced_tuning_param_audit_download",
                                        )
                                    if bool(st.session_state.get("universe_adv_tuning_show_detail", False)) and isinstance(trials_df, pd.DataFrame):
                                        st.markdown("**Raw tuning trials**")
                                        st.dataframe(trials_df, use_container_width=True)
                            else:
                                st.info("Advanced tuning did not produce a non-empty trials table.")
                    except Exception as e:
                        st.warning(f"Could not run advanced tuning research block: {e}")

                if st.session_state.get("engine_has_run", False):
                    st.markdown("#### Improve this setup")
                    st.caption("Use the post-run assistant blocks below to refine the current engine result before moving on to long-horizon projection.")

                if bool(st.session_state.get("universe_simple_mode_enabled", True)) and len(primary_info.get("used_assets", [])) >= 2:
                    try:
                        current_simple_spec = _build_current_simple_spec(int(st.session_state.get("universe_size", len(selected_assets)) or len(selected_assets)))
                        preset_recommendations = _evaluate_simple_preset_recommendations(
                            primary_engine_panel_df,
                            current_simple_spec,
                            primary_run,
                            cfg,
                        )
                        preset_rows = preset_recommendations.get("rows", pd.DataFrame())
                        strategy_preserving_rec = preset_recommendations.get("strategy_preserving")
                        opportunity_rec = preset_recommendations.get("opportunity")
                        if isinstance(preset_rows, pd.DataFrame) and not preset_rows.empty:
                            if strategy_preserving_rec is not None or opportunity_rec is not None:
                                st.markdown("#### Suggested preset improvements")
                                st.caption(
                                    "These suggestions keep the same primary universe and test nearby Simple-mode preset combinations to see whether a different posture improves final engine performance."
                                )

                                def _render_preset_recommendation_card(title: str, rec: dict[str, Any], *, button_key: str) -> None:
                                    rec_perf = _coerce_mapping(_coerce_mapping(rec.get("run", {})).get("performance_summary", {}))
                                    delta_sharpe = _safe_perf_float(rec_perf.get("sharpe")) - _safe_perf_float(primary_perf.get("sharpe"))
                                    delta_cagr = _safe_perf_float(rec_perf.get("cagr")) - _safe_perf_float(primary_perf.get("cagr"))
                                    delta_dd = _safe_perf_float(rec_perf.get("max_drawdown")) - _safe_perf_float(primary_perf.get("max_drawdown"))
                                    st.markdown(f"**{title}**")
                                    st.caption(
                                        f"Switch to template={rec.get('strategy_template', '—')} · style={rec.get('style_preset', '—')} · "
                                        f"signal={rec.get('resolved_signal_mode', '—')} · overlay={rec.get('resolved_overlay', '—')} · top_k={rec.get('resolved_top_k', '—')}"
                                    )
                                    c1, c2, c3, c4 = st.columns(4)
                                    with c1:
                                        st.metric("Sharpe", _format_num_or_dash(rec_perf.get("sharpe")), delta=_format_num_or_dash(delta_sharpe))
                                    with c2:
                                        st.metric("CAGR", _format_pct_or_dash(rec_perf.get("cagr")), delta=_format_delta_pct_or_dash(delta_cagr))
                                    with c3:
                                        st.metric("MaxDD", _format_pct_or_dash(rec_perf.get("max_drawdown")), delta=_format_delta_pct_or_dash(delta_dd))
                                    with c4:
                                        st.metric("Score delta", _format_num_or_dash(rec.get("score_delta")))
                                    if st.button("Apply suggested setup", key=button_key, use_container_width=True):
                                        _queue_apply_simple_preset_recommendation(rec.get("spec"))
                                        _queue_toast(f"Suggested preset applied: {rec.get('strategy_template')} + {rec.get('style_preset')}", icon="✅")
                                        st.rerun()

                                card_cols = st.columns(2)
                                with card_cols[0]:
                                    if strategy_preserving_rec is not None:
                                        _render_preset_recommendation_card(
                                            "Strategy-preserving suggestion",
                                            strategy_preserving_rec,
                                            button_key="apply_strategy_preserving_preset_recommendation",
                                        )
                                    else:
                                        st.markdown("**Strategy-preserving suggestion**")
                                        st.caption("The current style already looks close to the best nearby preset under the same template.")
                                with card_cols[1]:
                                    if opportunity_rec is not None:
                                        _render_preset_recommendation_card(
                                            "Broader opportunity-set suggestion",
                                            opportunity_rec,
                                            button_key="apply_broader_preset_recommendation",
                                        )
                                    else:
                                        st.markdown("**Broader opportunity-set suggestion**")
                                        st.caption("No materially better nearby preset combination was found in the assistant search.")

                            with st.expander("Preset recommendation diagnostics", expanded=False):
                                diag_cols = [c for c in [
                                    "label", "family", "strategy_template", "style_preset", "sharpe", "cagr", "max_drawdown",
                                    "annual_volatility", "mean_turnover", "score", "score_delta", "resolved_signal_mode",
                                    "resolved_top_k", "resolved_overlay", "resolved_temperature", "resolved_weight_shrink"
                                ] if c in preset_rows.columns]
                                st.dataframe(preset_rows[diag_cols].sort_values(["score", "sharpe"], ascending=[False, False]), use_container_width=True, hide_index=True)
                    except Exception as preset_recommendation_error:
                        st.warning(f"Could not run the preset recommendation assistant: {preset_recommendation_error}")

                recommendation_bundle = None
                recommendation_run = None
                if (
                    st.session_state.get("engine_has_run", False)
                    and len(primary_info.get("used_assets", [])) >= 2
                    and recommendation_candidate_assets
                ):
                    try:
                        recommendation_result = _build_strategy_recommended_assets(
                            panel_df,
                            universe_size=st.session_state.get("universe_size", len(selected_assets)),
                            strategy_name=st.session_state.get("universe_strategy", UNIVERSE_STRATEGY_CORE),
                            baseline_assets=primary_info.get("used_assets", []),
                        )
                        recommended_assets = list(recommendation_result.get("recommended_assets", []) or [])
                        if recommended_assets:
                            current_set = {str(x).strip().upper() for x in primary_info.get("used_assets", []) if str(x).strip()}
                            recommended_set = {str(x).strip().upper() for x in recommended_assets if str(x).strip()}
                            added_assets = sorted(recommended_set - current_set)
                            removed_assets = sorted(current_set - recommended_set)

                            st.markdown("#### Strategy-preserving universe recommendation")
                            r1, r2, r3, r4 = st.columns(4)
                            with r1:
                                st.metric("Current strategy", str(st.session_state.get("universe_strategy", UNIVERSE_STRATEGY_CORE)))
                            with r2:
                                st.metric("Universe size", int(st.session_state.get("universe_size", len(selected_assets)) or len(selected_assets)))
                            with r3:
                                st.metric("Candidate pool", int(recommendation_result.get("pool_used_count", 0) or 0))
                            with r4:
                                st.metric("Suggested swaps", int(recommendation_result.get("swap_count", len(added_assets)) or 0))

                            if added_assets or removed_assets:
                                st.caption(
                                    "The app kept the same strategy but searched a larger strategy-compatible pool and found a potentially stronger mix. "
                                    "This recommendation is based on asset-level return quality, correlation, and group diversification before rerunning the engine."
                                )
                                if added_assets:
                                    st.caption(f"Suggested additions: {', '.join(added_assets[:20])}")
                                if removed_assets:
                                    st.caption(f"Suggested removals: {', '.join(removed_assets[:20])}")
                            else:
                                st.caption("The current preset cut already looks close to the best strategy-preserving mix in the available candidate pool.")

                            recommended_panel_df, recommended_info = _filter_asset_panel_to_universe(panel_df, recommended_assets)
                            recommended_daily_panel_df = None
                            if yahoo_daily_panel_df is not None:
                                recommended_daily_panel_df, _ = _filter_asset_panel_to_universe(yahoo_daily_panel_df, recommended_assets)
                            recommended_engine_panel_df, recommended_engine_meta = _build_engine_asset_panel(
                                recommended_panel_df,
                                source_mode=data_source_mode,
                                requested_frequency=(str(yahoo_frequency) if data_source_mode == "Yahoo Finance (recommended)" else None),
                                daily_panel_df=recommended_daily_panel_df,
                                macro_panel_df=yahoo_macro_panel_df,
                            )

                            if len(recommended_info.get("used_assets", [])) >= 2 and not recommended_engine_panel_df.empty:
                                recommendation_cfg_payload = config_to_dict(cfg)
                                recommendation_cfg = MicroPipelineConfig(**recommendation_cfg_payload)
                                recommendation_run = run_micro_investment_pipeline(recommended_engine_panel_df, cfg=recommendation_cfg)
                                recommendation_bundle = compare_run_reports(primary_run, recommendation_run, left_name="current", right_name="recommended")

                                current_structure = _summarize_asset_basket_structure(panel_df, primary_info.get("used_assets", []))
                                recommended_structure = _summarize_asset_basket_structure(panel_df, recommended_info.get("used_assets", []))
                                recommendation_perf = recommendation_run.get("performance_summary", {}) or {}

                                recommendation_df = pd.DataFrame([
                                    {
                                        "Universe": "Current",
                                        "AssetsUsed": int(len(primary_info.get("used_assets", []))),
                                        "CAGR": primary_perf.get("cagr"),
                                        "Volatility": primary_perf.get("annual_volatility", primary_perf.get("annualized_volatility")),
                                        "Sharpe": primary_perf.get("sharpe"),
                                        "MaxDrawdown": primary_perf.get("max_drawdown"),
                                        "MeanDivRatio": (primary_run.get("diversification_summary", {}) or {}).get("mean_diversification_ratio"),
                                        "MeanBreadth": (primary_run.get("diversification_summary", {}) or {}).get("mean_effective_breadth"),
                                        "AvgAbsCorr": current_structure.get("avg_abs_corr"),
                                        "Groups": current_structure.get("group_count"),
                                        "Subgroups": current_structure.get("subgroup_count"),
                                    },
                                    {
                                        "Universe": "Recommended",
                                        "AssetsUsed": int(len(recommended_info.get("used_assets", []))),
                                        "CAGR": recommendation_perf.get("cagr"),
                                        "Volatility": recommendation_perf.get("annual_volatility", recommendation_perf.get("annualized_volatility")),
                                        "Sharpe": recommendation_perf.get("sharpe"),
                                        "MaxDrawdown": recommendation_perf.get("max_drawdown"),
                                        "MeanDivRatio": (recommendation_run.get("diversification_summary", {}) or {}).get("mean_diversification_ratio"),
                                        "MeanBreadth": (recommendation_run.get("diversification_summary", {}) or {}).get("mean_effective_breadth"),
                                        "AvgAbsCorr": recommended_structure.get("avg_abs_corr"),
                                        "Groups": recommended_structure.get("group_count"),
                                        "Subgroups": recommended_structure.get("subgroup_count"),
                                    },
                                ])
                                st.dataframe(recommendation_df, use_container_width=True, hide_index=True)

                                rec_metrics = _coerce_numeric_columns(recommendation_bundle.get("metrics_comparison", pd.DataFrame()), skip={"section", "metric"})
                                focus_rec_metrics = pd.DataFrame()
                                if isinstance(rec_metrics, pd.DataFrame) and not rec_metrics.empty and "metric" in rec_metrics.columns:
                                    focus_rec_metrics = rec_metrics[
                                        rec_metrics["metric"].isin([
                                            "cagr",
                                            "sharpe",
                                            "max_drawdown",
                                            "annual_volatility",
                                            "mean_turnover",
                                            "mean_effective_breadth",
                                            "mean_diversification_ratio",
                                            "mean_active_assets",
                                        ])
                                    ].copy()
                                if isinstance(focus_rec_metrics, pd.DataFrame) and not focus_rec_metrics.empty:
                                    st.markdown("**Current vs recommended (engine rerun)**")
                                    st.dataframe(focus_rec_metrics, use_container_width=True, hide_index=True)

                                with st.expander("Recommendation diagnostics / candidate scores", expanded=False):
                                    candidate_score_df = recommendation_result.get("candidate_score_df", pd.DataFrame())
                                    selection_trace_df = recommendation_result.get("selection_trace_df", pd.DataFrame())
                                    if isinstance(candidate_score_df, pd.DataFrame) and not candidate_score_df.empty:
                                        keep_cols = [c for c in [
                                            "asset", "group", "subgroup", "obs_count", "mean_return", "vol",
                                            "sharpe_like", "avg_abs_corr", "base_score"
                                        ] if c in candidate_score_df.columns]
                                        st.markdown("**Candidate asset ranking**")
                                        st.dataframe(candidate_score_df[keep_cols].head(50), use_container_width=True, hide_index=True)
                                    if isinstance(selection_trace_df, pd.DataFrame) and not selection_trace_df.empty:
                                        st.markdown("**Greedy selection trace**")
                                        st.dataframe(selection_trace_df, use_container_width=True, hide_index=True)
                                    _render_universe_group_mix("Recommended universe mix", recommended_assets, prefix="recommended_universe_mix")
                                    _download_dataframe_button(
                                        "Download recommendation comparison",
                                        recommendation_df,
                                        "current_vs_recommended_universe.csv",
                                        key="recommended_universe_compare_download",
                                    )

                                apply_recommendation_cols = st.columns([1.2, 2.8])
                                with apply_recommendation_cols[0]:
                                    apply_recommendation_clicked = st.button(
                                        "Apply recommended universe",
                                        key="apply_recommended_universe_button",
                                        help="Replace the current primary universe selection with the recommended asset mix and rerun the app.",
                                    )
                                with apply_recommendation_cols[1]:
                                    st.caption("This applies the recommended asset basket to Step 4 as the new primary universe.")

                                if apply_recommendation_clicked:
                                    _queue_session_updates(
                                        {
                                            "universe_custom_enabled": True,
                                            "universe_custom_enabled_source": "recommendation",
                                            "custom_universe_text": ", ".join(recommended_assets),
                                            "last_used_universe_assets": list(recommended_assets),
                                            "universe_size": int(len(recommended_assets)),
                                        }
                                    )
                                    _queue_toast("Recommended universe applied to Step 4.", icon="✅")
                                    st.rerun()
                            else:
                                st.info("The app could not build a fully usable recommended universe from the available panel yet.")
                        else:
                            reason = str(recommendation_result.get("reason", "") or "No recommendation produced.")
                            st.info(f"No within-strategy recommendation yet: {reason}")
                    except Exception as recommendation_error:
                        st.warning(f"Could not run the within-strategy recommendation block: {recommendation_error}")

                if (
                    st.session_state.get("engine_has_run", False)
                    and not st.session_state.get("projection_open", False)
                ):
                    st.markdown("#### Continue")
                    st.caption("You have now seen the current run and the assistant refinement suggestions. Move to Step 6 only when you are happy with this setup.")
                    if st.button(
                        "Continue to long-term projection",
                        key="open_projection_step",
                        use_container_width=True,
                    ):
                        st.session_state["projection_open"] = True
                        st.rerun()

                if (
                    st.session_state.get("engine_has_run", False)
                    and st.session_state.get("projection_open", False)
                ):
                    st.markdown("#### Step 6 — Long-horizon contribution projection")
                    _render_educational_finance_disclaimer("Step 6 projection")

                    projection_context = _coerce_mapping(st.session_state.get("investment_context", {}))
                    projection_monthly_contribution = float(
                        projection_context.get("monthly_contribution", st.session_state.get("investment_monthly_contribution", 0.0)) or 0.0
                    )
                    projection_weekly_equivalent = float(
                        projection_context.get("weekly_equivalent", st.session_state.get("investment_weekly_equivalent", 0.0)) or 0.0
                    )

                    projection_realised_monthly_returns = _read_projection_monthly_returns_from_context(projection_context)
                    projection_has_historical_path = (
                        projection_realised_monthly_returns is not None
                        and len(projection_realised_monthly_returns) > 0
                    )

                    projection_default_profile = _map_simple_style_to_risk_profile(
                        st.session_state.get("universe_simple_style_preset", "Balanced"),
                        st.session_state.get("universe_simple_strategy_template", "Balanced Risk-Controlled"),
                    )

                    projection_default_profile_idx = (
                        PROJECTION_PROFILE_OPTIONS.index(projection_default_profile)
                        if projection_default_profile in PROJECTION_PROFILE_OPTIONS
                        else 1
                    )

                    st.caption(
                        f"This projection now uses the Step 4 contribution bridge directly: **£{projection_monthly_contribution:,.0f}/month** "
                        f"(weekly equivalent: **£{projection_weekly_equivalent:,.0f}/week**)."
                    )

                    if projection_monthly_contribution <= 0.0:
                        st.info("Your current investment contribution is £0/month, so the long-horizon projection is inactive until Plan A monthly contribution is above zero.")
                        st.session_state["investment_projection_result"] = {}
                    else:
                        projection_risk_profile = str(projection_default_profile)
                        projection_bootstrap_method = "block" if projection_has_historical_path else "iid"

                        p1, p2, p3 = st.columns(3)
                        with p1:
                            projection_current_savings = float(st.number_input(
                                "Starting investment pot (£)",
                                min_value=0.0,
                                step=100.0,
                                key="investment_projection_current_savings",
                            ))
                        with p2:
                            projection_horizon_years = int(st.number_input(
                                "Projection horizon (years)",
                                min_value=1,
                                max_value=50,
                                step=1,
                                key="investment_projection_horizon_years",
                            ))
                        with p3:
                            projection_goal_amount_raw = float(st.number_input(
                                "Optional wealth goal (£)",
                                min_value=0.0,
                                step=1000.0,
                                key="investment_projection_goal_amount",
                                help="Set to 0 to disable the goal probability metric.",
                            ))
                        projection_goal_amount = projection_goal_amount_raw if projection_goal_amount_raw > 0.0 else None

                        st.caption(
                            f"Projection profile inferred from your current setup: **{projection_risk_profile}** · "
                            f"Return path source: **{'Historical engine path' if projection_has_historical_path else 'Parametric fallback'}**"
                        )

                        with st.expander("Advanced projection settings", expanded=False):
                            projection_n_sims = int(st.number_input(
                                "Monte Carlo paths",
                                min_value=100,
                                max_value=20000,
                                step=100,
                                key="investment_projection_n_sims",
                            ))
                            st.caption(
                                "Path construction is resolved automatically from the engine context to keep the main Step 6 flow simple."
                            )

                    projection_engine_summary = _coerce_mapping(projection_context.get("engine_performance_summary", {}))
                    projection_annual_return = pd.to_numeric(
                        pd.Series([projection_engine_summary.get("cagr", primary_perf.get("cagr"))]),
                        errors="coerce",
                    ).iloc[0]
                    projection_annual_vol = pd.to_numeric(
                        pd.Series([
                            projection_engine_summary.get(
                                "annual_volatility",
                                projection_engine_summary.get(
                                    "annualized_volatility",
                                    primary_perf.get("annual_volatility", primary_perf.get("annualized_volatility")),
                                ),
                            )
                        ]),
                        errors="coerce",
                    ).iloc[0]
                    projection_result = run_investment_projection(
                        current_savings=float(projection_current_savings),
                        invest_fraction=1.0,
                        monthly_contribution=float(projection_monthly_contribution),
                        horizon_years=int(projection_horizon_years),
                        risk_profile=str(projection_risk_profile),
                        n_sims=int(projection_n_sims),
                        seed=int(st.session_state.get("seed", 42)),
                        override_annual_return=float(projection_annual_return) if pd.notna(projection_annual_return) else None,
                        override_annual_vol=float(projection_annual_vol) if pd.notna(projection_annual_vol) else None,
                        realised_monthly_returns=projection_realised_monthly_returns,
                        bootstrap_method=str(projection_bootstrap_method),
                        goal_amount=projection_goal_amount,
                    )
                    st.session_state["investment_projection_result"] = projection_result
                    projection_summary = _coerce_mapping(projection_result.get("summary", {}))

                    ps1, ps2, ps3, ps4 = st.columns(4)
                    with ps1:
                        st.metric("Expected terminal wealth", f"£{float(projection_summary.get('expected_terminal', 0.0) or 0.0):,.0f}")
                    with ps2:
                        st.metric("Median terminal wealth", f"£{float(projection_summary.get('median_terminal', 0.0) or 0.0):,.0f}")
                    with ps3:
                        st.metric("P10–P90 range", f"£{float(projection_summary.get('p10_terminal', 0.0) or 0.0):,.0f} → £{float(projection_summary.get('p90_terminal', 0.0) or 0.0):,.0f}")
                    with ps4:
                        goal_prob = projection_summary.get("probability_of_reaching_goal")
                        goal_text = _format_pct_or_dash(goal_prob) if goal_prob is not None else "—"
                        st.metric("Goal probability", goal_text)

                    ps5, ps6, ps7, ps8 = st.columns(4)
                    with ps5:
                        st.metric("Monthly contribution used", f"£{float(projection_summary.get('monthly_contribution', projection_monthly_contribution) or 0.0):,.0f}/mo")
                    with ps6:
                        st.metric("Total contributed", f"£{float(projection_summary.get('total_contributed', 0.0) or 0.0):,.0f}")
                    with ps7:
                        st.metric("Expected profit", f"£{float(projection_summary.get('expected_profit', 0.0) or 0.0):,.0f}")
                    with ps8:
                        st.metric("Loss vs contributions", _format_pct_or_dash(projection_summary.get("probability_of_loss_vs_contributions")))

                    projection_source_code = str(projection_context.get("engine_projection_source", "historical_engine_oos" if projection_has_historical_path else "parametric_fallback"))
                    projection_source_label = "historical engine OOS returns" if projection_source_code == "historical_engine_oos" else "parametric fallback"
                    st.caption(
                        f"Projection source: {projection_source_label}. Step 6 now reads stored engine returns from investment_context first, while the monthly contribution is wired directly from Step 4 via the same context."
                    )

                    wealth_paths = projection_result.get("wealth_paths")
                    if isinstance(wealth_paths, np.ndarray) and wealth_paths.ndim == 2 and wealth_paths.shape[1] >= 2:
                        x_axis = np.arange(wealth_paths.shape[1], dtype=int)
                        wealth_df = pd.DataFrame({
                            "Median": np.nanmedian(wealth_paths, axis=0),
                            "P10": np.nanquantile(wealth_paths, 0.10, axis=0),
                            "P90": np.nanquantile(wealth_paths, 0.90, axis=0),
                        }, index=x_axis)
                        wealth_df.index.name = "Month"
                        st.line_chart(wealth_df, use_container_width=True)


                    st.markdown("#### Step 7 — Understand your strategy")
                    _render_educational_finance_disclaimer("Step 7 explanation")

                    step7_projection_context = _coerce_mapping(st.session_state.get("investment_projection_result", {}))
                    step7_explanation_inputs = InvestmentExplanationInputs(
                        performance_summary=_coerce_mapping(primary_run.get("performance_summary", {})),
                        risk_summary=_coerce_mapping(primary_run.get("risk_summary", {})),
                        investment_context=_coerce_mapping(st.session_state.get("investment_context", {})),
                        projection_summary=_coerce_mapping(step7_projection_context.get("summary", {})),
                    )
                    step7_text = build_investment_strategy_explanation(step7_explanation_inputs)
                    st.markdown(step7_text)

                if comparison_ready:
                    comparison_run = run_micro_investment_pipeline(comparison_engine_panel_df, cfg=cfg)
                    comparison_perf = comparison_run.get("performance_summary", {})
                    comparison_warnings = comparison_run.get("universe_warnings", []) or []

                    if comparison_warnings:
                        st.warning("Comparison universe warnings detected:")
                        for w in comparison_warnings:
                            st.markdown(f"- {w}")

                    st.markdown("#### Primary vs comparison metrics")
                    comparison_df = pd.DataFrame([
                        {
                            "Universe": "Primary",
                            "AssetsUsed": int(primary_engine_panel_df["asset"].nunique()),
                            "CAGR": primary_perf.get("cagr"),
                            "Volatility": primary_perf.get("annual_volatility", primary_perf.get("annualized_volatility")),
                            "Sharpe": primary_perf.get("sharpe"),
                            "MaxDrawdown": primary_perf.get("max_drawdown"),
                            "MeanTurnover": primary_perf.get("mean_turnover"),
                            "ActiveReturnAnn": primary_perf.get("active_return_annual"),
                            "TrackingErrorAnn": primary_perf.get("tracking_error_annual"),
                            "InformationRatio": primary_perf.get("information_ratio"),
                            "MeanDivRatio": (primary_run.get("diversification_summary", {}) or {}).get("mean_diversification_ratio"),
                            "Periods": primary_perf.get("n_months", primary_perf.get("n_periods")),
                            "Warnings": len(primary_warnings),
                        },
                        {
                            "Universe": "Comparison",
                            "AssetsUsed": int(comparison_engine_panel_df["asset"].nunique()),
                            "CAGR": comparison_perf.get("cagr"),
                            "Volatility": comparison_perf.get("annual_volatility", comparison_perf.get("annualized_volatility")),
                            "Sharpe": comparison_perf.get("sharpe"),
                            "MaxDrawdown": comparison_perf.get("max_drawdown"),
                            "MeanTurnover": comparison_perf.get("mean_turnover"),
                            "ActiveReturnAnn": comparison_perf.get("active_return_annual"),
                            "TrackingErrorAnn": comparison_perf.get("tracking_error_annual"),
                            "InformationRatio": comparison_perf.get("information_ratio"),
                            "MeanDivRatio": (comparison_run.get("diversification_summary", {}) or {}).get("mean_diversification_ratio"),
                            "Periods": comparison_perf.get("n_months", comparison_perf.get("n_periods")),
                            "Warnings": len(comparison_warnings),
                        },
                    ])
                    st.dataframe(comparison_df, use_container_width=True)

                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Best CAGR", "Primary" if (primary_perf.get("cagr", float("-inf")) >= comparison_perf.get("cagr", float("-inf"))) else "Comparison")
                    with m2:
                        st.metric("Best Sharpe", "Primary" if (primary_perf.get("sharpe", float("-inf")) >= comparison_perf.get("sharpe", float("-inf"))) else "Comparison")
                    with m3:
                        pvol = primary_perf.get("annual_volatility", primary_perf.get("annualized_volatility", np.inf))
                        cvol = comparison_perf.get("annual_volatility", comparison_perf.get("annualized_volatility", np.inf))
                        st.metric("Lower Vol", "Primary" if pvol <= cvol else "Comparison")
                    with m4:
                        pdd = abs(primary_perf.get("max_drawdown", np.inf))
                        cdd = abs(comparison_perf.get("max_drawdown", np.inf))
                        st.metric("Shallower MaxDD", "Primary" if pdd <= cdd else "Comparison")

                    comparison_bundle = compare_run_reports(primary_run, comparison_run, left_name="primary", right_name="comparison")
                    metrics_cmp = _coerce_numeric_columns(comparison_bundle.get("metrics_comparison", pd.DataFrame()), skip={"section", "metric"})
                    diagnostics_cmp = _coerce_numeric_columns(comparison_bundle.get("diagnostics_comparison", pd.DataFrame()), skip={"metric"})
                    weights_cmp = _coerce_numeric_columns(comparison_bundle.get("weights_comparison", pd.DataFrame()), skip={"asset"})

                    if isinstance(metrics_cmp, pd.DataFrame) and not metrics_cmp.empty:
                        focus_metrics = metrics_cmp[
                            metrics_cmp["metric"].isin([
                                "cagr",
                                "sharpe",
                                "max_drawdown",
                                "annual_volatility",
                                "mean_turnover",
                                "active_return_annual",
                                "tracking_error_annual",
                                "information_ratio",
                                "mean_effective_breadth",
                                "mean_effective_risk_bets",
                                "mean_diversification_ratio",
                                "mean_active_assets",
                            ])
                        ].copy()
                        if not focus_metrics.empty:
                            st.markdown("#### Research comparison snapshot")
                            st.dataframe(focus_metrics, use_container_width=True)

                    with st.expander("Detailed primary vs comparison report", expanded=False):
                        st.markdown("**Metric-by-metric comparison**")
                        if isinstance(metrics_cmp, pd.DataFrame) and not metrics_cmp.empty:
                            st.dataframe(metrics_cmp, use_container_width=True)
                        st.markdown("**Diagnostics mean comparison**")
                        if isinstance(diagnostics_cmp, pd.DataFrame) and not diagnostics_cmp.empty:
                            st.dataframe(diagnostics_cmp, use_container_width=True)
                        st.markdown("**Average weight comparison**")
                        if isinstance(weights_cmp, pd.DataFrame) and not weights_cmp.empty:
                            st.dataframe(weights_cmp, use_container_width=True)

                        cdl1, cdl2, cdl3 = st.columns(3)
                        with cdl1:
                            _download_dataframe_button("Download metrics comparison", metrics_cmp, "primary_vs_comparison_metrics.csv", key="cmp_metrics_download")
                        with cdl2:
                            _download_dataframe_button("Download diagnostics comparison", diagnostics_cmp, "primary_vs_comparison_diagnostics.csv", key="cmp_diag_download")
                        with cdl3:
                            _download_dataframe_button("Download weights comparison", weights_cmp, "primary_vs_comparison_weights.csv", key="cmp_weights_download")

                    _render_run_report("Comparison", comparison_run, prefix="comparison")
                elif st.session_state.get("comparison_enabled", False):
                    st.info("Comparison universe is not ready yet. Check the filtered panel coverage and the selected assets.")
            else:
                st.warning("Primary universe is not ready for the micro-pipeline. You need at least 2 assets with valid return history in the selected data source.")

        except Exception as e:
            st.error(f"Could not process the selected asset panel source: {e}")
    else:
        if data_source_mode == "Yahoo Finance (recommended)":
            st.info(
                "Choose a primary universe and Yahoo date range. The app will then download prices from Yahoo Finance, "
                "build a return panel automatically, and run the investment micro-pipeline."
            )
        else:
            st.info(
                "No asset panel uploaded yet. You can already define a primary universe here. "
                "Once a panel is uploaded, the app will filter it, run the investment micro-pipeline, and then test whether a better asset mix exists within the same strategy."
            )


    # Footer
    st.caption("MVP note: this version only adjusts discretionary automatically. Essentials are treated as fixed in the model.")


