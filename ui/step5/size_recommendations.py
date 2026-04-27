from __future__ import annotations

"""Step 5 universe-size assistant.

Phase 4 scope:
- keep the active Step 5 strategy preset unchanged;
- keep the active technical engine configuration unchanged;
- keep the current universe-composition result as the baseline;
- test only a capped coarse-to-fine set of universe sizes;
- apply by promoting an already rerun-tested candidate result.

This module deliberately does not rerun preset, engine-tuning, or universe-
composition search. It only changes the number of assets after Phase 3 has been
resolved.
"""

import hashlib
import json
import time
from dataclasses import fields
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st

from src.investment import MicroPipelineConfig, run_micro_investment_pipeline
from ui.services.step4_universe_service import (
    STEP4_PANEL_TIMINGS_KEY,
    asset_display_label,
    build_strategy_candidate_pool,
    build_step4_universe_payload_from_state,
    build_universe_mix_detail,
    normalize_asset_ticker,
    resolve_step4_asset_panel,
)
from ui.state.keys import (
    ASSET_AUTO_ADJUST,
    ASSET_END_DATE,
    ASSET_PANEL_DF,
    ASSET_PANEL_READY,
    ASSET_PANEL_SOURCE_LABEL,
    ASSET_RETURN_FREQUENCY,
    ASSET_SOURCE_MODE,
    ASSET_START_DATE,
    ASSET_UPLOADED_FILE,
    CUSTOM_UNIVERSE_TEXT,
    LAST_RECOMMENDATION_CANDIDATE_ASSETS,
    LAST_USED_UNIVERSE_ASSETS,
    RECOMMENDED_UNIVERSE_ASSETS,
    UNIVERSE_CUSTOM_ENABLED,
    UNIVERSE_CUSTOM_ENABLED_SOURCE,
    UNIVERSE_SIZE,
    UNIVERSE_STRATEGY,
)

try:
    from ui.state.updates import queue_and_rerun
except Exception:  # pragma: no cover
    queue_and_rerun = None


SIZE_SUGGESTION_STATE_KEY = "step5_size_suggestion_v1"
SIZE_SUGGESTION_SCOPE_KEY = "step5_size_suggestion_scope_v1"
SIZE_APPLIED_SIGNATURE_KEY = "step5_size_applied_run_signature_v1"
SIZE_APPLIED_LABEL_KEY = "step5_size_applied_label_v1"
SIZE_SKIPPED_SCOPE_KEY = "step5_size_skipped_scope_v1"
SIZE_SKIPPED_LABEL_KEY = "step5_size_skipped_label_v1"
SIZE_SKIPPED_RUN_SIGNATURE_KEY = "step5_size_skipped_run_signature_v1"
SIZE_SUGGESTION_TIMING_KEY = "step5_size_suggestion_timing_v1"
SIZE_RECOMMENDATION_CONTEXT_KEY = "step5_recommended_size_context_v1"
UNIVERSE_RECOMMENDATION_CONTEXT_KEY = "step5_recommended_universe_context_v1"
STEP5_SCROLL_TO_RESULT_AFTER_APPLY_KEY = "step5_scroll_to_real_run_result_after_apply_v1"

TECHNICAL_WIDGET_KEYS: dict[str, str] = {
    "top_k": "step5_basic_top_k",
    "lookback_mu": "step5_basic_lookback_mu",
    "lookback_sigma": "step5_basic_lookback_sigma",
    "signal_mode": "step5_basic_signal_mode",
    "temperature": "step5_basic_temperature",
    "weight_shrink": "step5_basic_weight_shrink",
    "inertia": "step5_basic_inertia",
    "feature_mu_enabled": "step5_basic_feature_mu_enabled",
    "feature_mu_blend": "step5_basic_feature_mu_blend",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _coerce_mapping(value: Any) -> dict:
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
        return out if pd.notna(out) else float(default)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _format_pct(value: Any, decimals: int = 2) -> str:
    try:
        return f"{100.0 * float(value):.{decimals}f}%"
    except Exception:
        return "—"


def _format_delta_pct(value: Any, decimals: int = 2) -> str:
    try:
        return f"{100.0 * float(value):+.{decimals}f}%"
    except Exception:
        return "—"


def _normalise_perf(perf: Any) -> dict:
    p = _coerce_mapping(perf)
    return {
        "cagr": _safe_float(p.get("cagr", 0.0), 0.0),
        "sharpe": _safe_float(p.get("sharpe", 0.0), 0.0),
        "annual_volatility": _safe_float(p.get("annual_volatility", p.get("volatility", 0.0)), 0.0),
        "volatility": _safe_float(p.get("volatility", p.get("annual_volatility", 0.0)), 0.0),
        "max_drawdown": abs(_safe_float(p.get("max_drawdown", 0.0), 0.0)),
        "periods": _safe_int(p.get("periods", 0), 0),
    }


def _coerce_cfg_payload(cfg_payload: Any) -> dict:
    payload = _coerce_mapping(cfg_payload)
    valid = {f.name for f in fields(MicroPipelineConfig)}
    return {key: value for key, value in payload.items() if key in valid}


def _coerce_cfg(cfg_payload: Any) -> MicroPipelineConfig:
    return MicroPipelineConfig(**_coerce_cfg_payload(cfg_payload))


def _hash_scope(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(encoded.encode("utf-8")).hexdigest()[:16]


def _asset_list(value: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in list(value or []):
        ticker = normalize_asset_ticker(raw)
        if ticker and ticker not in seen:
            seen.add(ticker)
            out.append(ticker)
    return out


def _asset_signature(assets: list[str]) -> str:
    encoded = json.dumps(sorted(_asset_list(assets)), sort_keys=True)
    return hashlib.md5(encoded.encode("utf-8")).hexdigest()[:12]


def _stable_panel_fingerprint(panel_df: pd.DataFrame) -> str:
    if panel_df is None or not isinstance(panel_df, pd.DataFrame) or panel_df.empty:
        return "empty"
    payload = {
        "rows": int(len(panel_df)),
        "cols": list(map(str, panel_df.columns)),
        "assets": sorted(panel_df["asset"].dropna().astype(str).str.upper().unique().tolist()) if "asset" in panel_df.columns else [],
        "min_date": str(panel_df["date"].min()) if "date" in panel_df.columns else "",
        "max_date": str(panel_df["date"].max()) if "date" in panel_df.columns else "",
    }
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(encoded.encode("utf-8")).hexdigest()[:16]


def _resolve_run_signature_for_size(
    cfg_final: dict,
    asset_panel_df: Any,
    *,
    universe_size: int,
    universe_strategy: str,
    selected_assets: list[str],
) -> str:
    payload = {
        "cfg_final": _coerce_mapping(cfg_final),
        "universe_size": int(universe_size),
        "universe_strategy": str(universe_strategy),
        "selected_assets": list(_asset_list(selected_assets)),
        "panel_fp": _stable_panel_fingerprint(asset_panel_df) if isinstance(asset_panel_df, pd.DataFrame) else "empty",
    }
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(encoded.encode("utf-8")).hexdigest()[:12]


def _resolve_config_fingerprint_for_cfg(cfg_final: dict) -> str:
    encoded = json.dumps(_coerce_mapping(cfg_final), sort_keys=True, default=str)
    return hashlib.md5(encoded.encode("utf-8")).hexdigest()[:12]


def _philosophy() -> str:
    raw = str(st.session_state.get("investment_philosophy", "Balanced") or "Balanced").strip()
    return raw if raw in {"Growth", "Balanced", "Defensive"} else "Balanced"


def _current_template_style() -> tuple[str, str]:
    template = str(st.session_state.get("step5_template", "") or "")
    style = str(st.session_state.get("step5_style", "") or "")
    return template, style


def _panel_assets(panel_df: pd.DataFrame) -> list[str]:
    if isinstance(panel_df, pd.DataFrame) and not panel_df.empty and "asset" in panel_df.columns:
        return sorted(panel_df["asset"].dropna().astype(str).str.upper().unique().tolist())
    return []


def _filter_panel_to_assets(panel_df: pd.DataFrame, assets: list[str]) -> pd.DataFrame:
    if panel_df is None or not isinstance(panel_df, pd.DataFrame) or panel_df.empty:
        return pd.DataFrame()
    if "asset" not in panel_df.columns:
        return pd.DataFrame()
    keep = set(_asset_list(assets))
    if not keep:
        return pd.DataFrame()
    work = panel_df.copy()
    work["asset"] = work["asset"].astype(str).str.upper().str.strip()
    return work.loc[work["asset"].isin(keep)].copy().reset_index(drop=True)


def _base_cfg_payload_from_run(run_result: dict) -> dict:
    run_map = _coerce_mapping(run_result)
    for raw in (
        run_map.get("config_dict"),
        st.session_state.get("last_engine_config"),
        st.session_state.get("step5_last_cfg_final"),
        run_map.get("config"),
    ):
        payload = _coerce_cfg_payload(raw)
        if payload:
            return payload
    return _coerce_cfg_payload({})


# ---------------------------------------------------------------------------
# Scoring / acceptance gate
# ---------------------------------------------------------------------------


def _score_perf(perf: dict, philosophy: str) -> float:
    p = _normalise_perf(perf)
    profile = str(philosophy or "Balanced").strip().lower()
    sharpe = p["sharpe"]
    cagr = p["cagr"]
    vol = abs(p["annual_volatility"])
    maxdd = abs(p["max_drawdown"])
    if profile == "growth":
        return float((0.45 * sharpe) + (1.20 * cagr) - (0.20 * maxdd) - (0.08 * vol))
    if profile == "defensive":
        return float((0.70 * sharpe) + (0.20 * cagr) - (1.10 * maxdd) - (0.60 * vol))
    return float((0.55 * sharpe) + (0.50 * cagr) - (0.70 * maxdd) - (0.35 * vol))


def _acceptance_gate(candidate_perf: dict, current_perf: dict, philosophy: str) -> tuple[bool, str]:
    c = _normalise_perf(candidate_perf)
    b = _normalise_perf(current_perf)
    profile = str(philosophy or "Balanced").strip().lower()

    delta_score = _score_perf(c, philosophy) - _score_perf(b, philosophy)
    delta_sharpe = c["sharpe"] - b["sharpe"]
    delta_cagr = c["cagr"] - b["cagr"]
    delta_vol = c["annual_volatility"] - b["annual_volatility"]  # positive means more volatile
    delta_dd = c["max_drawdown"] - b["max_drawdown"]  # positive means worse drawdown

    if profile == "defensive":
        passed = (
            delta_score > 0.006
            and delta_dd <= 0.004
            and delta_vol <= 0.006
            and delta_sharpe >= -0.015
            and delta_cagr >= -0.012
        ) or (
            (delta_dd <= -0.012 or delta_vol <= -0.008)
            and delta_sharpe >= -0.010
            and delta_cagr >= -0.015
        )
        reason = "Defensive size gate prioritises lower drawdown/volatility and avoids sacrificing Sharpe or too much CAGR."
    elif profile == "growth":
        passed = (
            delta_score > 0.006
            and delta_cagr >= -0.004
            and delta_sharpe >= -0.060
            and delta_dd <= 0.050
            and delta_vol <= 0.035
        ) or (
            delta_cagr >= 0.010
            and delta_sharpe >= -0.040
            and delta_dd <= 0.050
        )
        reason = "Growth size gate gives more weight to CAGR, while rejecting larger universes that damage Sharpe or drawdown too much."
    else:
        passed = (
            delta_score > 0.006
            and delta_sharpe >= -0.030
            and delta_cagr >= -0.008
            and delta_dd <= 0.020
            and delta_vol <= 0.018
        ) or (
            delta_sharpe >= 0.055
            and delta_dd <= 0.020
            and delta_vol <= 0.020
        )
        reason = "Balanced size gate looks for a better overall trade-off without materially worsening Sharpe, drawdown, or volatility."

    return bool(passed), reason


def _asset_metric_table(panel_df: pd.DataFrame) -> pd.DataFrame:
    if panel_df.empty or not {"date", "asset", "return"}.issubset(set(panel_df.columns)):
        return pd.DataFrame()
    work = panel_df[["date", "asset", "return"]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["asset"] = work["asset"].astype(str).str.upper().str.strip()
    work["return"] = pd.to_numeric(work["return"], errors="coerce")
    work = work.dropna(subset=["date", "asset", "return"])
    if work.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for asset, g in work.groupby("asset", sort=False):
        s = pd.to_numeric(g.sort_values("date")["return"], errors="coerce").dropna().clip(lower=-0.999999)
        if len(s) < 12:
            continue
        wealth = (1.0 + s).cumprod()
        years = max(float(len(s)) / 12.0, 1.0 / 12.0)
        final_wealth = float(wealth.iloc[-1]) if len(wealth) else 0.0
        cagr = float(final_wealth ** (1.0 / years) - 1.0) if final_wealth > 0 else 0.0
        vol = float(s.std(ddof=1) * (12.0 ** 0.5)) if len(s) > 1 else 0.0
        annual_mean = float(s.mean() * 12.0) if len(s) else 0.0
        sharpe = float(annual_mean / vol) if vol > 1e-12 else 0.0
        drawdown = (wealth / wealth.cummax()) - 1.0
        maxdd = abs(float(drawdown.min())) if len(drawdown) else 0.0
        trailing_12 = s.tail(12)
        trailing_return = float((1.0 + trailing_12).prod() - 1.0) if len(trailing_12) else 0.0
        rows.append(
            {
                "asset": str(asset),
                "n_obs": int(len(s)),
                "cagr": cagr,
                "annual_volatility": vol,
                "sharpe": sharpe,
                "max_drawdown": maxdd,
                "trailing_12m_return": trailing_return,
                "balanced_score": (0.55 * sharpe) + (0.50 * cagr) - (0.70 * maxdd) - (0.35 * vol),
                "growth_score": (0.45 * sharpe) + (1.20 * cagr) - (0.20 * maxdd) - (0.08 * vol),
                "defensive_score": (0.70 * sharpe) + (0.20 * cagr) - (1.10 * maxdd) - (0.60 * vol),
            }
        )
    return pd.DataFrame(rows)


def _size_cap_for_philosophy(philosophy: str) -> int:
    profile = str(philosophy or "Balanced").strip().lower()
    if profile == "defensive":
        return 25
    if profile == "growth":
        return 75
    return 50


def _coarse_sizes(baseline_size: int, cap_size: int) -> list[int]:
    baseline_size = int(baseline_size)
    cap_size = int(cap_size)
    if cap_size <= baseline_size:
        return []
    span = cap_size - baseline_size
    if span <= 4:
        raw = [cap_size]
    else:
        raw = [baseline_size + round(span * 0.45), baseline_size + round(span * 0.75), cap_size]
    out: list[int] = []
    for value in raw:
        size = int(max(baseline_size + 1, min(cap_size, value)))
        if size not in out:
            out.append(size)
    return out[:3]


def _refinement_size(baseline_size: int, cap_size: int, best_size: int, tested_sizes: set[int]) -> int | None:
    baseline_size = int(baseline_size)
    cap_size = int(cap_size)
    best_size = int(best_size)
    if cap_size <= baseline_size:
        return None

    local_step = max(2, int(round((cap_size - baseline_size) / 8.0)))
    raw_candidates: list[int]
    if best_size >= cap_size:
        raw_candidates = [cap_size - local_step, cap_size - (2 * local_step)]
    elif best_size <= baseline_size + local_step:
        raw_candidates = [best_size + local_step, best_size + (2 * local_step)]
    else:
        raw_candidates = [best_size + local_step, best_size - local_step, best_size + (2 * local_step), best_size - (2 * local_step)]

    for raw in raw_candidates:
        size = int(max(baseline_size + 1, min(cap_size, raw)))
        if size > baseline_size and size <= cap_size and size not in tested_sizes:
            return size
    return None


def _search_pool_assets(current_assets: list[str], strategy: str, cap_size: int) -> list[str]:
    target_n = max(int(cap_size), len(current_assets))
    seed = _asset_list(current_assets + build_strategy_candidate_pool(cap_size, strategy))
    return seed[:target_n]


def _resolve_temporary_search_panel(search_assets: list[str], current_panel: pd.DataFrame, step4_payload: dict) -> tuple[pd.DataFrame, str]:
    """Return a non-persistent panel for size testing.

    If the active Step 4 panel already covers the capped size-search pool, reuse
    it. Otherwise, for Yahoo-backed universes only, build a temporary panel from
    the same Step 4 data settings without updating the permanent Step 4 panel.
    """
    clean_assets = _asset_list(search_assets)
    if not clean_assets:
        return pd.DataFrame(), "No candidate assets were available for size search."

    current_panel = current_panel.copy() if isinstance(current_panel, pd.DataFrame) else pd.DataFrame()
    available = set(_panel_assets(current_panel))
    if available and set(clean_assets).issubset(available):
        return current_panel, "Reused the current Step 4 asset panel."

    source_mode = str(step4_payload.get("asset_source_mode", st.session_state.get(ASSET_SOURCE_MODE, "yahoo")) or "yahoo").lower()
    if source_mode != "yahoo":
        missing = sorted(set(clean_assets) - available)
        return current_panel, f"Only the already-loaded panel can be used for uploaded data. Missing assets: {', '.join(missing[:12])}."

    old_timings = st.session_state.get(STEP4_PANEL_TIMINGS_KEY)
    try:
        panel_df, _label, _caption = resolve_step4_asset_panel(
            selected_assets=clean_assets,
            candidate_assets=[],
            source_mode="yahoo",
            start_date=step4_payload.get("asset_start_date", st.session_state.get(ASSET_START_DATE)),
            end_date=step4_payload.get("asset_end_date", st.session_state.get(ASSET_END_DATE)),
            frequency=str(step4_payload.get("asset_return_frequency", st.session_state.get(ASSET_RETURN_FREQUENCY, "monthly")) or "monthly"),
            auto_adjust=bool(step4_payload.get("auto_adjust_asset_prices", st.session_state.get(ASSET_AUTO_ADJUST, True))),
            uploaded_file=step4_payload.get("asset_uploaded_file", st.session_state.get(ASSET_UPLOADED_FILE)),
        )
        if old_timings is not None:
            st.session_state[STEP4_PANEL_TIMINGS_KEY] = old_timings
        else:
            st.session_state.pop(STEP4_PANEL_TIMINGS_KEY, None)
        if isinstance(panel_df, pd.DataFrame) and not panel_df.empty:
            loaded = set(_panel_assets(panel_df))
            if set(clean_assets).issubset(loaded):
                return panel_df, "Built a temporary Yahoo panel for the capped size-search pool."
            missing = sorted(set(clean_assets) - loaded)
            return panel_df, f"Temporary Yahoo panel loaded partially. Missing assets: {', '.join(missing[:12])}."
    except Exception as exc:
        if old_timings is not None:
            st.session_state[STEP4_PANEL_TIMINGS_KEY] = old_timings
        else:
            st.session_state.pop(STEP4_PANEL_TIMINGS_KEY, None)
        return current_panel, f"Temporary larger-size panel could not be built: {exc}"

    return current_panel, "Temporary larger-size panel was unavailable; reused the current panel only."


def _rank_assets_for_size(panel_df: pd.DataFrame, current_assets: list[str], target_size: int, philosophy: str, strategy: str) -> list[str]:
    current = _asset_list(current_assets)
    target_size = int(target_size)
    available = set(_panel_assets(panel_df))
    kept_current = [asset for asset in current if asset in available]
    if target_size <= len(kept_current):
        return kept_current[:target_size]

    metrics = _asset_metric_table(panel_df)
    if metrics.empty:
        extras = [asset for asset in _panel_assets(panel_df) if asset not in set(kept_current)]
        return _asset_list(kept_current + extras)[:target_size]

    profile = str(philosophy or "Balanced").strip()
    score_col = "growth_score" if profile == "Growth" else "defensive_score" if profile == "Defensive" else "balanced_score"
    preferred_pool = set(_search_pool_assets(current, strategy, target_size))
    work = metrics.copy()
    if preferred_pool:
        preferred = work.loc[work["asset"].isin(preferred_pool)].copy()
        if len(preferred) >= max(0, target_size - len(kept_current)):
            work = preferred
    work[score_col] = pd.to_numeric(work.get(score_col), errors="coerce")
    work = work.dropna(subset=["asset", score_col]).sort_values(score_col, ascending=False)
    extras = [asset for asset in work["asset"].astype(str).str.upper().tolist() if asset not in set(kept_current)]
    return _asset_list(kept_current + extras)[:target_size]


def _run_candidate(cfg_payload: dict, panel_df: pd.DataFrame) -> dict:
    if panel_df is None or not isinstance(panel_df, pd.DataFrame) or panel_df.empty:
        return {}
    try:
        raw = run_micro_investment_pipeline(panel_df, cfg=_coerce_cfg(cfg_payload))
        raw_map = _coerce_mapping(raw)
        perf = _normalise_perf(raw_map.get("performance_summary", {}))
        if not perf:
            return {}
        return {"raw": raw_map, "performance_summary": perf}
    except Exception as exc:
        return {"error": str(exc)}


def _build_scope(run_result: dict) -> str:
    run_map = _coerce_mapping(run_result)
    panel = st.session_state.get(ASSET_PANEL_DF, st.session_state.get("asset_panel_df"))
    panel_df = panel.copy() if isinstance(panel, pd.DataFrame) else pd.DataFrame()
    step4_payload = _coerce_mapping(build_step4_universe_payload_from_state())
    base_cfg = _base_cfg_payload_from_run(run_map)
    template, style = _current_template_style()
    current_assets = _asset_list(
        run_map.get("selected_assets")
        or step4_payload.get("selected_assets")
        or st.session_state.get(LAST_USED_UNIVERSE_ASSETS, [])
        or st.session_state.get("selected_assets", [])
    )
    baseline_size = _safe_int(run_map.get("universe_size", step4_payload.get("size", st.session_state.get(UNIVERSE_SIZE, len(current_assets) or 25))), len(current_assets) or 25)
    current_strategy = str(run_map.get("universe_strategy", step4_payload.get("strategy", st.session_state.get(UNIVERSE_STRATEGY, ""))) or "")
    philosophy = _philosophy()
    return _hash_scope(
        {
            "run_signature": str(run_map.get("run_signature", "") or ""),
            "config_fingerprint": str(run_map.get("config_fingerprint", "") or ""),
            "philosophy": philosophy,
            "template": template,
            "style": style,
            "technical_cfg": base_cfg,
            "baseline_size": int(baseline_size),
            "cap_size": int(_size_cap_for_philosophy(philosophy)),
            "universe_strategy": current_strategy,
            "current_assets": current_assets,
            "panel_assets": _panel_assets(panel_df),
            "panel_fp": _stable_panel_fingerprint(panel_df),
            "phase": "universe_size_v1",
        }
    )


def _not_testable_eval(size: int, reason: str, *, stage: str) -> dict:
    return {
        "label": f"Size {int(size)}",
        "family": "size_search",
        "stage": str(stage),
        "universe_size": int(size),
        "assets": [],
        "performance_summary": {},
        "raw_result": {},
        "score": None,
        "score_delta": None,
        "accepted": False,
        "not_testable": True,
        "gate_reason": str(reason or "Candidate size was not testable."),
        "error": str(reason or "Candidate size was not testable."),
        "elapsed_sec": 0.0,
    }


def _evaluate_size(
    *,
    size: int,
    stage: str,
    search_panel: pd.DataFrame,
    current_assets: list[str],
    current_perf: dict,
    philosophy: str,
    universe_strategy: str,
    base_cfg: dict,
) -> dict:
    candidate_started = time.perf_counter()
    target_size = int(size)
    assets = _rank_assets_for_size(search_panel, current_assets, target_size, philosophy, universe_strategy)
    if len(assets) != target_size:
        return _not_testable_eval(
            target_size,
            f"Not enough loaded assets to test size {target_size}. Available ranked assets: {len(assets)}.",
            stage=stage,
        )

    candidate_panel = _filter_panel_to_assets(search_panel, assets)
    panel_asset_count = int(candidate_panel["asset"].nunique()) if isinstance(candidate_panel, pd.DataFrame) and "asset" in candidate_panel.columns else 0
    if candidate_panel.empty or panel_asset_count < target_size:
        return _not_testable_eval(
            target_size,
            f"Size {target_size} was not testable because the panel only contained {panel_asset_count} assets after filtering.",
            stage=stage,
        )

    run = _run_candidate(base_cfg, candidate_panel)
    candidate_elapsed_sec = float(time.perf_counter() - candidate_started)
    perf = _normalise_perf(run.get("performance_summary", {}))
    accepted = False
    gate_reason = "Candidate could not be evaluated."
    if perf and not run.get("error"):
        accepted, gate_reason = _acceptance_gate(perf, current_perf, philosophy)

    return {
        "label": f"Optimised size {target_size}",
        "family": "size_search",
        "stage": str(stage),
        "universe_size": int(target_size),
        "universe_strategy": str(universe_strategy),
        "assets": list(assets),
        "asset_signature": _asset_signature(assets),
        "panel_fingerprint": _stable_panel_fingerprint(candidate_panel),
        "panel_rows": int(len(candidate_panel)),
        "panel_assets": panel_asset_count,
        "candidate_panel": candidate_panel,
        "cfg_payload": dict(base_cfg),
        "performance_summary": dict(perf),
        "raw_result": _coerce_mapping(run.get("raw", {})),
        "score": _score_perf(perf, philosophy) if perf else None,
        "score_delta": (_score_perf(perf, philosophy) - _score_perf(current_perf, philosophy)) if perf else None,
        "accepted": bool(accepted),
        "not_testable": False,
        "gate_reason": gate_reason,
        "error": str(run.get("error", "") or ""),
        "elapsed_sec": float(candidate_elapsed_sec),
    }


def _run_size_search(run_result: dict, *, max_engine_tests: int = 4) -> dict:
    run_map = _coerce_mapping(run_result)
    current_perf = _normalise_perf(run_map.get("performance_summary", {}))
    philosophy = _philosophy()
    template, style = _current_template_style()
    base_cfg = _base_cfg_payload_from_run(run_map)

    panel = st.session_state.get(ASSET_PANEL_DF, st.session_state.get("asset_panel_df"))
    current_panel = panel.copy() if isinstance(panel, pd.DataFrame) else pd.DataFrame()
    if current_panel.empty:
        return {"scope": _build_scope(run_result), "evaluations": [], "error": "Step 4 asset panel is missing."}

    step4_payload = _coerce_mapping(build_step4_universe_payload_from_state())
    current_assets = _asset_list(
        run_map.get("selected_assets")
        or step4_payload.get("selected_assets")
        or st.session_state.get(LAST_USED_UNIVERSE_ASSETS, [])
        or st.session_state.get("selected_assets", [])
    )
    if not current_assets:
        current_assets = _panel_assets(current_panel)

    baseline_size = _safe_int(
        run_map.get("universe_size", step4_payload.get("size", st.session_state.get(UNIVERSE_SIZE, len(current_assets) or 25))),
        len(current_assets) or 25,
    )
    if len(current_assets) > 0:
        baseline_size = min(int(baseline_size), len(current_assets)) if int(baseline_size) > len(current_assets) else int(baseline_size)
    current_assets = current_assets[: int(baseline_size)] if baseline_size > 0 else current_assets

    universe_strategy = str(run_map.get("universe_strategy", step4_payload.get("strategy", st.session_state.get(UNIVERSE_STRATEGY, ""))) or "")
    cap_size = int(_size_cap_for_philosophy(philosophy))
    coarse = _coarse_sizes(int(baseline_size), cap_size)
    search_assets = _search_pool_assets(current_assets, universe_strategy, cap_size)

    started = time.perf_counter()
    search_panel, panel_note = _resolve_temporary_search_panel(search_assets, current_panel, step4_payload)

    evaluations: list[dict] = []
    tested_count = 0
    for size in coarse:
        if tested_count >= int(max_engine_tests):
            break
        item = _evaluate_size(
            size=int(size),
            stage="coarse",
            search_panel=search_panel,
            current_assets=current_assets,
            current_perf=current_perf,
            philosophy=philosophy,
            universe_strategy=universe_strategy,
            base_cfg=base_cfg,
        )
        evaluations.append(item)
        if not bool(item.get("not_testable", False)):
            tested_count += 1

    testable_stage1 = [x for x in evaluations if not bool(_coerce_mapping(x).get("not_testable", False))]
    if testable_stage1 and tested_count < int(max_engine_tests):
        best_stage1 = max(
            testable_stage1,
            key=lambda item: (
                _safe_float(_coerce_mapping(item).get("score_delta"), -999.0),
                _safe_float(_normalise_perf(_coerce_mapping(item).get("performance_summary", {})).get("sharpe"), -999.0),
            ),
        )
        tested_sizes = {_safe_int(_coerce_mapping(x).get("universe_size"), 0) for x in evaluations}
        refine = _refinement_size(int(baseline_size), cap_size, _safe_int(best_stage1.get("universe_size"), int(baseline_size)), tested_sizes)
        if refine is not None:
            item = _evaluate_size(
                size=int(refine),
                stage="local refinement",
                search_panel=search_panel,
                current_assets=current_assets,
                current_perf=current_perf,
                philosophy=philosophy,
                universe_strategy=universe_strategy,
                base_cfg=base_cfg,
            )
            evaluations.append(item)
            if not bool(item.get("not_testable", False)):
                tested_count += 1

    if not coarse:
        evaluations.append(
            _not_testable_eval(
                int(baseline_size),
                f"Current size {int(baseline_size)} is already at or above the practical {philosophy} cap ({cap_size}).",
                stage="range check",
            )
        )

    def _sort_key(item: dict) -> tuple[int, int, float, float]:
        m = _coerce_mapping(item)
        return (
            0 if bool(m.get("not_testable", False)) else 1,
            1 if bool(m.get("accepted", False)) else 0,
            _safe_float(m.get("score_delta"), -999.0),
            _safe_float(_normalise_perf(m.get("performance_summary", {})).get("sharpe"), -999.0),
        )

    evaluations = sorted(evaluations, key=_sort_key, reverse=True)
    total_elapsed_sec = float(time.perf_counter() - started)
    accepted_count = int(sum(1 for item in evaluations if bool(_coerce_mapping(item).get("accepted", False))))
    real_tested_count = int(sum(1 for item in evaluations if not bool(_coerce_mapping(item).get("not_testable", False))))

    timing_rows = []
    accepted_seen = 0
    for item in evaluations:
        item_map = _coerce_mapping(item)
        accepted = bool(item_map.get("accepted", False))
        not_testable = bool(item_map.get("not_testable", False))
        accepted_seen += 1 if accepted else 0
        if not_testable:
            status = "not testable"
        elif accepted and accepted_seen == 1:
            status = "recommended"
        elif accepted:
            status = "passed gate"
        else:
            status = "not selected"
        timing_rows.append(
            {
                "candidate": str(item_map.get("label", f"Size {item_map.get('universe_size', '—')}") or "Candidate"),
                "status": status,
                "seconds": round(_safe_float(item_map.get("elapsed_sec", 0.0), 0.0), 2),
            }
        )

    scope = _build_scope(run_result)
    return {
        "scope": scope,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": total_elapsed_sec,
        "philosophy": philosophy,
        "strategy_template": template,
        "style_preset": style,
        "baseline_size": int(baseline_size),
        "cap_size": int(cap_size),
        "universe_strategy": universe_strategy,
        "current_assets": list(current_assets),
        "current_perf": dict(current_perf),
        "panel_note": str(panel_note or ""),
        "evaluations": evaluations,
        "candidate_count": int(real_tested_count),
        "planned_candidate_count": int(len(evaluations)),
        "timing_summary": {
            "scope": scope,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_seconds": round(total_elapsed_sec, 2),
            "candidate_count": int(real_tested_count),
            "planned_candidate_count": int(len(evaluations)),
            "accepted_count": accepted_count,
            "candidate_seconds": timing_rows,
        },
    }


# ---------------------------------------------------------------------------
# Apply and render
# ---------------------------------------------------------------------------


def _normalise_promoted_candidate_result(candidate: dict, cfg_payload: dict) -> dict:
    candidate_map = _coerce_mapping(candidate)
    raw_result = _coerce_mapping(candidate_map.get("raw_result", {}))
    if not raw_result:
        return {}

    assets = _asset_list(candidate_map.get("assets", []))
    universe_size = _safe_int(candidate_map.get("universe_size", len(assets)), len(assets))
    universe_strategy = str(candidate_map.get("universe_strategy", st.session_state.get(UNIVERSE_STRATEGY, "")) or "")
    candidate_panel = candidate_map.get("candidate_panel")
    if not isinstance(candidate_panel, pd.DataFrame) or candidate_panel.empty:
        panel = st.session_state.get(ASSET_PANEL_DF, st.session_state.get("asset_panel_df"))
        source_panel = panel.copy() if isinstance(panel, pd.DataFrame) else pd.DataFrame()
        candidate_panel = _filter_panel_to_assets(source_panel, assets)
    if candidate_panel.empty:
        return {}

    run_signature = _resolve_run_signature_for_size(
        cfg_payload,
        candidate_panel,
        universe_size=universe_size,
        universe_strategy=universe_strategy,
        selected_assets=assets,
    )
    config_fingerprint = _resolve_config_fingerprint_for_cfg(cfg_payload)
    run_timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    perf = _normalise_perf(candidate_map.get("performance_summary", raw_result.get("performance_summary", {})))

    promoted = dict(raw_result)
    promoted.update(
        {
            "source": "micro_pipeline_real",
            "promoted_from_size_candidate": True,
            "size_candidate_label": str(candidate_map.get("label", "Universe size candidate") or "Universe size candidate"),
            "asset_panel_source_label": "Step 5 optimised universe size",
            "asset_panel_n_rows": int(len(candidate_panel)),
            "asset_panel_n_assets": int(candidate_panel["asset"].nunique()) if "asset" in candidate_panel.columns else 0,
            "performance_summary": dict(perf),
            "config": dict(cfg_payload),
            "config_dict": dict(cfg_payload),
            "run_signature": run_signature,
            "config_fingerprint": config_fingerprint,
            "run_timestamp": run_timestamp,
            "panel_shape": (int(len(candidate_panel)), int(len(candidate_panel.columns))),
            "evaluation_period_label": "Walk-forward evaluated period",
            "search_eval_split": {"enabled": False, "reason": "size_candidate_promoted_after_rerun_test"},
            "universe_size": int(universe_size),
            "universe_strategy": str(universe_strategy),
            "selected_assets": list(assets),
            "candidate_panel_fingerprint": _stable_panel_fingerprint(candidate_panel),
        }
    )
    promoted.setdefault(
        "timing_summary",
        {
            "base_run_sec": 0.0,
            "display_base_run_sec": 0.0,
            "base_from_cache": True,
            "total_step5_sec": 0.0,
            "display_total_step5_sec": 0.0,
            "cache_hits": 1,
            "engine_runs": 0,
            "promoted_from_size_candidate": True,
            "breakdown_df": pd.DataFrame(),
        },
    )
    return promoted


def _technical_widget_patch(cfg_payload: dict) -> dict:
    patch: dict[str, Any] = {}
    payload = _coerce_mapping(cfg_payload)
    for field, widget_key in TECHNICAL_WIDGET_KEYS.items():
        if field in payload:
            patch[widget_key] = payload.get(field)
    return patch


def _apply_candidate(candidate: dict) -> None:
    candidate_map = _coerce_mapping(candidate)
    assets = _asset_list(candidate_map.get("assets", []))
    if not assets:
        st.warning("This size candidate does not expose an asset list.")
        return

    cfg_payload = _coerce_cfg_payload(candidate_map.get("cfg_payload", {}))
    promoted_result = _normalise_promoted_candidate_result(candidate_map, cfg_payload)
    universe_size = _safe_int(candidate_map.get("universe_size", len(assets)), len(assets))
    universe_strategy = str(candidate_map.get("universe_strategy", st.session_state.get(UNIVERSE_STRATEGY, "")) or "")

    candidate_panel = candidate_map.get("candidate_panel")
    if not isinstance(candidate_panel, pd.DataFrame) or candidate_panel.empty:
        source_panel = st.session_state.get(ASSET_PANEL_DF, st.session_state.get("asset_panel_df"))
        candidate_panel = _filter_panel_to_assets(source_panel.copy() if isinstance(source_panel, pd.DataFrame) else pd.DataFrame(), assets)

    recommendation_context = {
        "source": "step5_size_suggestion",
        "universe_size": int(universe_size),
        "universe_strategy": str(universe_strategy),
        "assets": list(assets),
        "asset_signature": _asset_signature(assets),
        "label": str(candidate_map.get("label", "Universe size candidate") or "Universe size candidate"),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    patch = {
        UNIVERSE_SIZE: int(universe_size),
        "universe_size_input": int(universe_size),
        UNIVERSE_STRATEGY: str(universe_strategy),
        "universe_strategy_input": str(universe_strategy),
        UNIVERSE_CUSTOM_ENABLED: False,
        "universe_custom_enabled_input": False,
        CUSTOM_UNIVERSE_TEXT: "",
        UNIVERSE_CUSTOM_ENABLED_SOURCE: "recommendation",
        RECOMMENDED_UNIVERSE_ASSETS: list(assets),
        LAST_USED_UNIVERSE_ASSETS: list(assets),
        LAST_RECOMMENDATION_CANDIDATE_ASSETS: _asset_list(build_strategy_candidate_pool(universe_size, universe_strategy)),
        "selected_assets": list(assets),
        SIZE_RECOMMENDATION_CONTEXT_KEY: recommendation_context,
        UNIVERSE_RECOMMENDATION_CONTEXT_KEY: recommendation_context,
        ASSET_PANEL_DF: candidate_panel if isinstance(candidate_panel, pd.DataFrame) and not candidate_panel.empty else st.session_state.get(ASSET_PANEL_DF),
        "asset_panel_df": candidate_panel if isinstance(candidate_panel, pd.DataFrame) and not candidate_panel.empty else st.session_state.get("asset_panel_df"),
        ASSET_PANEL_READY: bool(isinstance(candidate_panel, pd.DataFrame) and not candidate_panel.empty),
        ASSET_PANEL_SOURCE_LABEL: "Step 5 optimised universe size",
    }
    patch.update(_technical_widget_patch(cfg_payload))

    if promoted_result:
        run_signature = str(promoted_result.get("run_signature", "") or "")
        config_fingerprint = str(promoted_result.get("config_fingerprint", "") or "")
        run_timestamp = str(promoted_result.get("run_timestamp", "") or "")
        patch.update(
            {
                "last_engine_config": dict(cfg_payload),
                "step5_last_cfg_final": dict(cfg_payload),
                "step5_last_run_signature": run_signature,
                "step5_current_run_signature": run_signature,
                "step5_current_input_signature": run_signature,
                "step5_last_config_fingerprint": config_fingerprint,
                "step5_current_config_fingerprint": config_fingerprint,
                "step5_last_run_timestamp": run_timestamp,
                "step5_last_run_result": dict(promoted_result),
                "step5_run_result": dict(promoted_result),
                "engine_has_run": True,
                "step5_force_run_once": False,
                "step5_auto_run_reason": "",
                "step5_auto_run_pending_patch_keys": [],
                SIZE_SUGGESTION_STATE_KEY: {},
                SIZE_SUGGESTION_SCOPE_KEY: "",
                SIZE_APPLIED_SIGNATURE_KEY: run_signature,
                SIZE_APPLIED_LABEL_KEY: str(candidate_map.get("label", "Universe size candidate") or "Universe size candidate"),
                "step5_size_apply_message_v1": (
                    f"Universe size suggestion applied using the rerun-tested candidate result: "
                    f"{candidate_map.get('label', 'Universe size candidate')}."
                ),
                STEP5_SCROLL_TO_RESULT_AFTER_APPLY_KEY: True,
            }
        )
    else:
        patch.update(
            {
                "engine_has_run": False,
                "step5_last_run_result": None,
                "step5_run_result": None,
                SIZE_APPLIED_SIGNATURE_KEY: "",
                SIZE_APPLIED_LABEL_KEY: "",
                "step5_size_apply_message_v1": (
                    f"Universe size suggestion applied: {candidate_map.get('label', 'Universe size candidate')}. "
                    "Run the engine again to confirm the updated result."
                ),
            }
        )

    if callable(queue_and_rerun):
        queue_and_rerun(patch)
        return
    for key, value in patch.items():
        st.session_state[key] = value
    st.rerun()


def _skip_current_size_candidate(scope: str, label: str, run_signature: str) -> None:
    patch = {
        SIZE_SKIPPED_SCOPE_KEY: str(scope or ""),
        SIZE_SKIPPED_LABEL_KEY: str(label or "current size"),
        SIZE_SKIPPED_RUN_SIGNATURE_KEY: str(run_signature or ""),
        "step5_size_apply_message_v1": f"Universe size kept unchanged for this run. Skipped recommendation: {label}.",
        STEP5_SCROLL_TO_RESULT_AFTER_APPLY_KEY: True,
    }
    if callable(queue_and_rerun):
        queue_and_rerun(patch)
        return
    for key, value in patch.items():
        st.session_state[key] = value
    st.rerun()


def _candidate_table(evaluations: list[dict], current_perf: dict, baseline_size: int) -> pd.DataFrame:
    rows: list[dict] = []
    base = _normalise_perf(current_perf)
    accepted_seen = 0
    for raw in evaluations:
        item = _coerce_mapping(raw)
        not_testable = bool(item.get("not_testable", False))
        perf = _normalise_perf(item.get("performance_summary", {}))
        accepted = bool(item.get("accepted", False))
        accepted_seen += 1 if accepted else 0
        if not_testable:
            status = "Not testable"
        else:
            status = "Recommended" if accepted and accepted_seen == 1 else ("Passed gate" if accepted else "Not selected")
        target_size = _safe_int(item.get("universe_size"), baseline_size)
        vol_delta = perf.get("annual_volatility", 0.0) - base.get("annual_volatility", 0.0)
        maxdd_improvement = base.get("max_drawdown", 0.0) - perf.get("max_drawdown", 0.0)
        rows.append(
            {
                "candidate": str(item.get("label", f"Size {target_size}") or f"Size {target_size}"),
                "status": status,
                "stage": str(item.get("stage", "") or ""),
                "size": int(target_size),
                "Δ size": int(target_size) - int(baseline_size),
                "CAGR": "—" if not_testable else _format_pct(perf.get("cagr", 0.0)),
                "Δ CAGR": "—" if not_testable else _format_delta_pct(perf.get("cagr", 0.0) - base.get("cagr", 0.0)),
                "Vol": "—" if not_testable else _format_pct(perf.get("annual_volatility", 0.0)),
                "Δ Vol": "—" if not_testable else _format_delta_pct(vol_delta),
                "MaxDD": "—" if not_testable else f"-{100.0 * abs(perf.get('max_drawdown', 0.0)):.2f}%",
                "DD improvement": "—" if not_testable else _format_delta_pct(maxdd_improvement),
                "Sharpe": "—" if not_testable else f"{perf.get('sharpe', 0.0):.2f}",
                "Δ Sharpe": "—" if not_testable else f"{perf.get('sharpe', 0.0) - base.get('sharpe', 0.0):+.2f}",
                "score delta": "—" if not_testable else f"{_safe_float(item.get('score_delta'), 0.0):+.3f}",
                "seconds": "—" if not_testable else f"{_safe_float(item.get('elapsed_sec'), 0.0):.2f}s",
                "gate": str(item.get("gate_reason", "") or ""),
                "error": str(item.get("error", "") or ""),
            }
        )
    return pd.DataFrame(rows)


def _render_recommended_candidate(candidate: dict, current_perf: dict, baseline_size: int) -> None:
    item = _coerce_mapping(candidate)
    perf = _normalise_perf(item.get("performance_summary", {}))
    current = _normalise_perf(current_perf)
    assets = _asset_list(item.get("assets", []))
    target_size = _safe_int(item.get("universe_size"), len(assets))

    st.markdown("### Recommended universe size")
    st.markdown(f"**Optimised size {target_size}**")
    st.caption(
        f"This keeps the current preset and technical engine config, starts from the current {baseline_size}-asset universe, "
        f"and only changes the tested universe size."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("CAGR", _format_pct(perf["cagr"]), delta=_format_delta_pct(perf["cagr"] - current["cagr"]))
    with c2:
        st.metric(
            "Vol",
            _format_pct(perf["annual_volatility"]),
            delta=_format_delta_pct(perf["annual_volatility"] - current["annual_volatility"]),
            delta_color="inverse",
        )
    with c3:
        dd_improvement = current["max_drawdown"] - perf["max_drawdown"]
        st.metric("MaxDD", f"-{100.0 * perf['max_drawdown']:.2f}%", delta=_format_delta_pct(dd_improvement))
    with c4:
        st.metric("Sharpe", f"{perf['sharpe']:.2f}", delta=f"{perf['sharpe'] - current['sharpe']:+.2f}")

    st.success("This candidate passed the universe-size acceptance gate.")
    gate_reason = str(item.get("gate_reason", "") or "")
    if gate_reason:
        st.caption(gate_reason)

    with st.expander("Optimised universe assets", expanded=False):
        if assets:
            preview = ", ".join(asset_display_label(x) for x in assets[:50])
            if len(assets) > 50:
                preview += f" ... +{len(assets) - 50} more"
            st.write(preview)
        st.caption(
            f"baseline_size={int(baseline_size)} · tested_size={int(target_size)} · "
            f"score_delta={_safe_float(item.get('score_delta'), 0.0):+.3f}"
        )
        detail = build_universe_mix_detail(assets)
        if isinstance(detail, pd.DataFrame) and not detail.empty:
            st.dataframe(detail, use_container_width=True, hide_index=True)


def render_size_improvement(run_result: dict) -> None:
    """Render the fourth Step 5 improvement phase: universe size."""
    run_map = _coerce_mapping(run_result)
    perf = _normalise_perf(run_map.get("performance_summary", {}))
    if not perf:
        return

    st.markdown("### Universe size suggestion")
    st.caption(
        "This keeps the current strategy preset, technical engine configuration, and current universe-composition baseline, "
        "then tests whether a different universe size improves the result."
    )

    msg = st.session_state.pop("step5_size_apply_message_v1", "")

    current_run_signature = str(run_map.get("run_signature", "") or "")
    applied_signature = str(st.session_state.get(SIZE_APPLIED_SIGNATURE_KEY, "") or "")
    applied_label = str(st.session_state.get(SIZE_APPLIED_LABEL_KEY, "") or "")
    if applied_signature and current_run_signature and applied_signature == current_run_signature:
        label = applied_label or "the accepted universe size suggestion"
        st.success(f"Universe size applied: {label}. The rerun-tested candidate is now the current Step 5 result.")
        st.caption(
            "Size testing is hidden for this run to avoid suggesting the same loop again. "
            "Change Step 4 or run a new baseline if you want to test a different size."
        )
        return

    if msg:
        st.success(str(msg))

    scope = _build_scope(run_map)
    saved_scope = str(st.session_state.get(SIZE_SUGGESTION_SCOPE_KEY, "") or "")
    payload = _coerce_mapping(st.session_state.get(SIZE_SUGGESTION_STATE_KEY, {}))
    evaluations = list(payload.get("evaluations", []) or []) if saved_scope == scope else []

    if saved_scope != scope or not evaluations:
        with st.spinner("Testing universe-size candidates with the real engine..."):
            payload = _run_size_search(run_map, max_engine_tests=4)
        st.session_state[SIZE_SUGGESTION_STATE_KEY] = payload
        st.session_state[SIZE_SUGGESTION_SCOPE_KEY] = str(payload.get("scope", scope))
        st.session_state[SIZE_SUGGESTION_TIMING_KEY] = _coerce_mapping(payload.get("timing_summary", {}))
        evaluations = list(payload.get("evaluations", []) or [])

    if not evaluations:
        error = str(payload.get("error", "") or "")
        if error:
            st.info(f"No safe size candidates were available to test for this run. {error}")
        else:
            st.info("No safe size candidates were available to test for this run.")
        return

    baseline_size = _safe_int(payload.get("baseline_size", run_map.get("universe_size", st.session_state.get(UNIVERSE_SIZE, 25))), 25)
    cap_size = _safe_int(payload.get("cap_size", _size_cap_for_philosophy(_philosophy())), _size_cap_for_philosophy(_philosophy()))
    panel_note = str(payload.get("panel_note", "") or "")
    if panel_note:
        st.caption(panel_note)
    st.caption(f"Size-search range: current baseline {baseline_size} → practical cap {cap_size}. Baseline is not rerun.")

    accepted_items = [dict(x) for x in evaluations if bool(_coerce_mapping(x).get("accepted", False))]
    table = _candidate_table(evaluations, perf, baseline_size)
    skipped_scope = str(st.session_state.get(SIZE_SKIPPED_SCOPE_KEY, "") or "")
    skipped_label = str(st.session_state.get(SIZE_SKIPPED_LABEL_KEY, "") or "")
    skipped_run_signature = str(st.session_state.get(SIZE_SKIPPED_RUN_SIGNATURE_KEY, "") or "")
    size_was_skipped = bool(
        skipped_scope
        and skipped_scope == scope
        and (not skipped_run_signature or skipped_run_signature == current_run_signature)
    )

    if not accepted_items:
        st.success("Current universe size has converged: no tested alternative materially improved this run.")
        st.caption(
            "The diagnostics table shows tested and unsupported sizes. Unsupported sizes are labelled Not testable rather than rejected."
        )
    elif size_was_skipped:
        st.info("Current universe size kept for this run. The improvement flow is complete.")
        if skipped_label:
            st.caption(f"Skipped size recommendation: {skipped_label}.")
    else:
        best_candidate = accepted_items[0]
        st.success("Recommended universe size found. The best accepted candidate is shown below.")
        _render_recommended_candidate(best_candidate, perf, baseline_size)
        st.caption(
            "The diagnostics table shows every planned universe-size candidate. Only the best accepted candidate "
            "is offered as the main action; rejected or unsupported sizes are shown for transparency."
        )

    with st.expander("Universe size diagnostics", expanded=False):
        st.caption(
            f"real_engine_tests={_safe_int(payload.get('candidate_count', 0), 0)} · "
            f"planned_candidates={_safe_int(payload.get('planned_candidate_count', len(evaluations)), len(evaluations))} · "
            f"elapsed={_safe_float(payload.get('elapsed_sec', 0.0), 0.0):.2f}s · scope={str(payload.get('scope', scope))}"
        )
        if not table.empty:
            st.dataframe(table, use_container_width=True, hide_index=True)

    action_label = str(accepted_items[0].get("label", "recommended size") if accepted_items else "current size")
    if accepted_items and not size_was_skipped:
        best_candidate = accepted_items[0]
        left, right = st.columns(2)
        with left:
            if st.button("Apply recommended size", key="step5_apply_best_size_candidate_v1", use_container_width=True):
                _apply_candidate(best_candidate)
        with right:
            if st.button("Keep current size and finish", key="step5_skip_best_size_candidate_v1", use_container_width=True):
                _skip_current_size_candidate(scope, action_label, current_run_signature)
    elif not size_was_skipped:
        if st.button("Keep current size and finish", key="step5_keep_current_size_finish_v1", use_container_width=True):
            _skip_current_size_candidate(scope, action_label, current_run_signature)


# Compatibility wrapper for older imports.
def render_size_recommendations(run_result: dict | None = None, *args: Any, **kwargs: Any) -> None:
    if isinstance(run_result, dict):
        render_size_improvement(run_result)
    return None
