from __future__ import annotations

"""Step 5 universe-composition assistant.

Phase 3 scope:
- keep the active Step 5 strategy preset unchanged;
- keep the active technical engine configuration unchanged;
- keep the current universe size unchanged;
- test only small candidate asset compositions already present in the Step 4
  market-data panel;
- apply by promoting an already rerun-tested candidate result.

This module deliberately does not implement universe-size search. Size remains a
separate later phase.
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
    allowed_universe_strategies_for_size,
    asset_display_label,
    build_generated_universe,
    build_step4_universe_payload_from_state,
    build_strategy_candidate_pool,
    build_universe_mix_detail,
    normalize_asset_ticker,
)
from ui.state.keys import (
    ASSET_PANEL_DF,
    ASSET_PANEL_READY,
    ASSET_PANEL_SOURCE_LABEL,
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


UNIVERSE_SUGGESTION_STATE_KEY = "step5_universe_suggestion_v1"
UNIVERSE_SUGGESTION_SCOPE_KEY = "step5_universe_suggestion_scope_v1"
UNIVERSE_APPLIED_SIGNATURE_KEY = "step5_universe_applied_run_signature_v1"
UNIVERSE_APPLIED_LABEL_KEY = "step5_universe_applied_label_v1"
UNIVERSE_SKIPPED_SCOPE_KEY = "step5_universe_skipped_scope_v1"
UNIVERSE_SKIPPED_LABEL_KEY = "step5_universe_skipped_label_v1"
UNIVERSE_SKIPPED_RUN_SIGNATURE_KEY = "step5_universe_skipped_run_signature_v1"
UNIVERSE_SUGGESTION_TIMING_KEY = "step5_universe_suggestion_timing_v1"
UNIVERSE_RECOMMENDATION_CONTEXT_KEY = "step5_recommended_universe_context_v1"
STEP5_SCROLL_TO_RESULT_AFTER_APPLY_KEY = "step5_scroll_to_real_run_result_after_apply_v1"


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


def _resolve_run_signature_for_universe(
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
# Candidate construction and scoring
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
        reason = "Defensive gate prioritises lower drawdown/volatility and avoids sacrificing Sharpe or too much CAGR."
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
        reason = "Growth gate gives more weight to CAGR, while rejecting candidates that damage Sharpe or drawdown too much."
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
        reason = "Balanced gate looks for a better overall trade-off without materially worsening Sharpe, drawdown, or volatility."

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


def _take_top_assets(metrics: pd.DataFrame, score_col: str, universe_size: int, *, ascending: bool = False) -> list[str]:
    if metrics.empty or score_col not in metrics.columns:
        return []
    work = metrics.copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work = work.dropna(subset=["asset", score_col]).sort_values(score_col, ascending=ascending)
    return _asset_list(work["asset"].head(int(universe_size)).tolist())


def _diversified_assets(metrics: pd.DataFrame, universe_size: int, philosophy: str) -> list[str]:
    if metrics.empty:
        return []
    score_col = "growth_score" if philosophy == "Growth" else "defensive_score" if philosophy == "Defensive" else "balanced_score"
    detail = build_universe_mix_detail(metrics["asset"].tolist())
    if not isinstance(detail, pd.DataFrame) or detail.empty or "ticker" not in detail.columns:
        return _take_top_assets(metrics, score_col, universe_size)
    work = metrics.merge(detail.rename(columns={"ticker": "asset"}), on="asset", how="left")
    work["group"] = work.get("group", "Other / Unclassified").fillna("Other / Unclassified").astype(str)
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work = work.dropna(subset=[score_col]).sort_values(["group", score_col], ascending=[True, False])
    grouped = {group: df.sort_values(score_col, ascending=False)["asset"].tolist() for group, df in work.groupby("group")}
    selected: list[str] = []
    while len(selected) < int(universe_size) and grouped:
        made_progress = False
        for group in list(grouped.keys()):
            bucket = grouped.get(group, [])
            if not bucket:
                grouped.pop(group, None)
                continue
            asset = bucket.pop(0)
            if asset not in selected:
                selected.append(asset)
                made_progress = True
            if len(selected) >= int(universe_size):
                break
        if not made_progress:
            break
    if len(selected) < int(universe_size):
        fill = _take_top_assets(metrics, score_col, universe_size)
        for asset in fill:
            if asset not in selected:
                selected.append(asset)
            if len(selected) >= int(universe_size):
                break
    return _asset_list(selected[: int(universe_size)])


def _candidate_specs(
    *,
    panel_df: pd.DataFrame,
    current_assets: list[str],
    current_strategy: str,
    universe_size: int,
    philosophy: str,
    max_candidates: int = 4,
) -> list[dict]:
    available_assets = set(_panel_assets(panel_df))
    current_set = set(_asset_list(current_assets))
    current_sig = _asset_signature(list(current_set))
    candidates: list[dict] = []
    seen = {current_sig}

    def _add(label: str, family: str, assets: list[str], strategy_name: str, caption: str) -> None:
        clean_assets = _asset_list(assets)
        if len(clean_assets) != int(universe_size):
            return
        if not set(clean_assets).issubset(available_assets):
            return
        sig = _asset_signature(clean_assets)
        if sig in seen:
            return
        seen.add(sig)
        candidates.append(
            {
                "label": label,
                "family": family,
                "assets": clean_assets,
                "strategy_name": str(strategy_name or current_strategy),
                "caption": caption,
                "asset_signature": sig,
            }
        )

    # 1) Prefer semantically explainable Step 4 strategy seeds when the assets are
    # already present in the existing Step 4 panel. This avoids downloading or
    # applying untested assets.
    for strategy in allowed_universe_strategies_for_size(universe_size):
        if len(candidates) >= int(max_candidates):
            break
        if str(strategy) == str(current_strategy):
            continue
        generated = build_generated_universe(universe_size, strategy)
        _add(
            label=f"{strategy} composition",
            family="strategy_seed",
            assets=generated,
            strategy_name=strategy,
            caption="Same universe size, generated from an alternative Step 4 composition seed already available in the data panel.",
        )

    # 2) Fallback / complement: build small same-size compositions from the loaded
    # candidate pool only. These are still test-before-apply and require no new data.
    metrics = _asset_metric_table(panel_df)
    if not metrics.empty:
        candidate_pool = _asset_list(build_strategy_candidate_pool(universe_size, current_strategy))
        pool_set = set(candidate_pool) & available_assets
        if pool_set:
            metrics = metrics.loc[metrics["asset"].isin(pool_set)].copy()

    if not metrics.empty and len(candidates) < int(max_candidates):
        profile_score = "growth_score" if philosophy == "Growth" else "defensive_score" if philosophy == "Defensive" else "balanced_score"
        _add(
            label="Risk-adjusted available-pool composition",
            family="risk_adjusted_pool",
            assets=_take_top_assets(metrics, profile_score, universe_size),
            strategy_name=current_strategy,
            caption="Same size, selected from loaded Step 4 candidate assets by the current philosophy score.",
        )
    if not metrics.empty and len(candidates) < int(max_candidates):
        _add(
            label="Lower-volatility available-pool composition",
            family="risk_control_pool",
            assets=_take_top_assets(metrics, "annual_volatility", universe_size, ascending=True),
            strategy_name=current_strategy,
            caption="Same size, selected from loaded Step 4 candidate assets with lower realised volatility.",
        )
    if not metrics.empty and len(candidates) < int(max_candidates):
        _add(
            label="Momentum available-pool composition",
            family="momentum_pool",
            assets=_take_top_assets(metrics, "trailing_12m_return", universe_size),
            strategy_name=current_strategy,
            caption="Same size, selected from loaded Step 4 candidate assets with stronger trailing return behaviour.",
        )
    if not metrics.empty and len(candidates) < int(max_candidates):
        _add(
            label="Diversified available-pool composition",
            family="diversified_pool",
            assets=_diversified_assets(metrics, universe_size, philosophy),
            strategy_name=current_strategy,
            caption="Same size, selected from loaded Step 4 candidate assets while spreading picks across asset groups where possible.",
        )

    return candidates[: int(max_candidates)]


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
    current_assets = _asset_list(step4_payload.get("selected_assets") or st.session_state.get(LAST_USED_UNIVERSE_ASSETS, []))
    return _hash_scope(
        {
            "run_signature": str(run_map.get("run_signature", "") or ""),
            "config_fingerprint": str(run_map.get("config_fingerprint", "") or ""),
            "philosophy": _philosophy(),
            "template": template,
            "style": style,
            "technical_cfg": base_cfg,
            "universe_size": int(step4_payload.get("size", st.session_state.get(UNIVERSE_SIZE, 25)) or 25),
            "universe_strategy": str(step4_payload.get("strategy", st.session_state.get(UNIVERSE_STRATEGY, "")) or ""),
            "current_assets": current_assets,
            "panel_assets": _panel_assets(panel_df),
            "panel_fp": _stable_panel_fingerprint(panel_df),
            "phase": "universe_composition_v1",
        }
    )


def _run_universe_search(run_result: dict, *, max_candidates: int = 4) -> dict:
    run_map = _coerce_mapping(run_result)
    current_perf = _normalise_perf(run_map.get("performance_summary", {}))
    philosophy = _philosophy()
    template, style = _current_template_style()
    base_cfg = _base_cfg_payload_from_run(run_map)

    panel = st.session_state.get(ASSET_PANEL_DF, st.session_state.get("asset_panel_df"))
    panel_df = panel.copy() if isinstance(panel, pd.DataFrame) else pd.DataFrame()
    if panel_df.empty:
        return {"scope": _build_scope(run_result), "evaluations": [], "error": "Step 4 asset panel is missing."}

    step4_payload = _coerce_mapping(build_step4_universe_payload_from_state())
    universe_size = _safe_int(step4_payload.get("size", st.session_state.get(UNIVERSE_SIZE, 25)), 25)
    current_strategy = str(step4_payload.get("strategy", st.session_state.get(UNIVERSE_STRATEGY, "")) or "")
    current_assets = _asset_list(step4_payload.get("selected_assets") or st.session_state.get(LAST_USED_UNIVERSE_ASSETS, []))
    if not current_assets:
        current_assets = _panel_assets(panel_df)[:universe_size]

    candidates = _candidate_specs(
        panel_df=panel_df,
        current_assets=current_assets,
        current_strategy=current_strategy,
        universe_size=universe_size,
        philosophy=philosophy,
        max_candidates=max_candidates,
    )

    evaluations: list[dict] = []
    started = time.perf_counter()

    for candidate in candidates:
        candidate_started = time.perf_counter()
        candidate_map = _coerce_mapping(candidate)
        assets = _asset_list(candidate_map.get("assets", []))
        candidate_panel = _filter_panel_to_assets(panel_df, assets)
        run = _run_candidate(base_cfg, candidate_panel)
        candidate_elapsed_sec = float(time.perf_counter() - candidate_started)
        perf = _normalise_perf(run.get("performance_summary", {}))
        accepted = False
        gate_reason = "Candidate could not be evaluated."
        if perf and not run.get("error"):
            accepted, gate_reason = _acceptance_gate(perf, current_perf, philosophy)

        overlap = 0.0
        try:
            union = set(current_assets) | set(assets)
            overlap = float(len(set(current_assets) & set(assets)) / len(union)) if union else 0.0
        except Exception:
            overlap = 0.0

        evaluations.append(
            {
                "label": str(candidate_map.get("label", "Universe composition candidate") or "Universe composition candidate"),
                "family": str(candidate_map.get("family", "universe") or "universe"),
                "caption": str(candidate_map.get("caption", "") or ""),
                "strategy_template": template,
                "style_preset": style,
                "universe_size": int(universe_size),
                "universe_strategy": str(candidate_map.get("strategy_name", current_strategy) or current_strategy),
                "assets": list(assets),
                "asset_signature": _asset_signature(assets),
                "overlap_vs_current": float(overlap),
                "panel_fingerprint": _stable_panel_fingerprint(candidate_panel),
                "panel_rows": int(len(candidate_panel)) if isinstance(candidate_panel, pd.DataFrame) else 0,
                "panel_assets": int(candidate_panel["asset"].nunique()) if isinstance(candidate_panel, pd.DataFrame) and "asset" in candidate_panel.columns else 0,
                "cfg_payload": dict(base_cfg),
                "performance_summary": dict(perf),
                "raw_result": _coerce_mapping(run.get("raw", {})),
                "score": _score_perf(perf, philosophy) if perf else None,
                "score_delta": (_score_perf(perf, philosophy) - _score_perf(current_perf, philosophy)) if perf else None,
                "accepted": bool(accepted),
                "gate_reason": gate_reason,
                "error": str(run.get("error", "") or ""),
                "elapsed_sec": float(candidate_elapsed_sec),
            }
        )

    evaluations = sorted(
        evaluations,
        key=lambda item: (
            1 if bool(item.get("accepted")) else 0,
            _safe_float(item.get("score_delta"), -999.0),
            _safe_float(_coerce_mapping(item.get("performance_summary", {})).get("sharpe"), -999.0),
        ),
        reverse=True,
    )
    total_elapsed_sec = float(time.perf_counter() - started)
    accepted_count = int(sum(1 for item in evaluations if bool(_coerce_mapping(item).get("accepted", False))))
    timing_rows = []
    accepted_seen = 0
    for item in evaluations:
        item_map = _coerce_mapping(item)
        accepted = bool(item_map.get("accepted", False))
        accepted_seen += 1 if accepted else 0
        timing_rows.append(
            {
                "candidate": str(item_map.get("label", "Candidate") or "Candidate"),
                "status": "recommended" if accepted and accepted_seen == 1 else ("passed gate" if accepted else "not selected"),
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
        "universe_size": int(universe_size),
        "current_strategy": current_strategy,
        "current_assets": list(current_assets),
        "current_perf": dict(current_perf),
        "evaluations": evaluations,
        "candidate_count": int(len(evaluations)),
        "timing_summary": {
            "scope": scope,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_seconds": round(total_elapsed_sec, 2),
            "candidate_count": int(len(evaluations)),
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
    panel = st.session_state.get(ASSET_PANEL_DF, st.session_state.get("asset_panel_df"))
    source_panel = panel.copy() if isinstance(panel, pd.DataFrame) else pd.DataFrame()
    candidate_panel = _filter_panel_to_assets(source_panel, assets)
    if candidate_panel.empty:
        return {}

    run_signature = _resolve_run_signature_for_universe(
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
            "promoted_from_universe_candidate": True,
            "universe_candidate_label": str(candidate_map.get("label", "Universe composition candidate") or "Universe composition candidate"),
            "asset_panel_source_label": "Step 5 recommended universe composition",
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
            "search_eval_split": {"enabled": False, "reason": "universe_candidate_promoted_after_rerun_test"},
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
            "promoted_from_universe_candidate": True,
            "breakdown_df": pd.DataFrame(),
        },
    )
    return promoted


def _apply_candidate(candidate: dict) -> None:
    candidate_map = _coerce_mapping(candidate)
    assets = _asset_list(candidate_map.get("assets", []))
    if not assets:
        st.warning("This universe candidate does not expose an asset list.")
        return

    cfg_payload = _coerce_cfg_payload(candidate_map.get("cfg_payload", {}))
    promoted_result = _normalise_promoted_candidate_result(candidate_map, cfg_payload)
    universe_size = _safe_int(candidate_map.get("universe_size", len(assets)), len(assets))
    universe_strategy = str(candidate_map.get("universe_strategy", st.session_state.get(UNIVERSE_STRATEGY, "")) or "")

    source_panel = st.session_state.get(ASSET_PANEL_DF, st.session_state.get("asset_panel_df"))
    candidate_panel = _filter_panel_to_assets(source_panel.copy() if isinstance(source_panel, pd.DataFrame) else pd.DataFrame(), assets)
    recommendation_context = {
        "source": "step5_universe_suggestion",
        "universe_size": int(universe_size),
        "universe_strategy": str(universe_strategy),
        "assets": list(assets),
        "asset_signature": _asset_signature(assets),
        "label": str(candidate_map.get("label", "Universe composition candidate") or "Universe composition candidate"),
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
        UNIVERSE_RECOMMENDATION_CONTEXT_KEY: recommendation_context,
        ASSET_PANEL_DF: candidate_panel if isinstance(candidate_panel, pd.DataFrame) and not candidate_panel.empty else source_panel,
        "asset_panel_df": candidate_panel if isinstance(candidate_panel, pd.DataFrame) and not candidate_panel.empty else source_panel,
        ASSET_PANEL_READY: bool(isinstance(candidate_panel, pd.DataFrame) and not candidate_panel.empty),
        ASSET_PANEL_SOURCE_LABEL: "Step 5 recommended universe composition",
    }

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
                UNIVERSE_SUGGESTION_STATE_KEY: {},
                UNIVERSE_SUGGESTION_SCOPE_KEY: "",
                UNIVERSE_APPLIED_SIGNATURE_KEY: run_signature,
                UNIVERSE_APPLIED_LABEL_KEY: str(candidate_map.get("label", "Universe composition candidate") or "Universe composition candidate"),
                "step5_universe_apply_message_v1": (
                    f"Universe composition suggestion applied using the rerun-tested candidate result: "
                    f"{candidate_map.get('label', 'Universe composition candidate')}."
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
                UNIVERSE_APPLIED_SIGNATURE_KEY: "",
                UNIVERSE_APPLIED_LABEL_KEY: "",
                "step5_universe_apply_message_v1": (
                    f"Universe composition suggestion applied: {candidate_map.get('label', 'Universe composition candidate')}. "
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


def _skip_current_universe_candidate(scope: str, label: str, run_signature: str) -> None:
    """Mark the current universe-composition recommendation as skipped.

    This completes the improvement flow while preserving diagnostics/timing and
    leaving the existing Step 4 universe/panel untouched.
    """
    patch = {
        UNIVERSE_SKIPPED_SCOPE_KEY: str(scope or ""),
        UNIVERSE_SKIPPED_LABEL_KEY: str(label or "recommended universe"),
        UNIVERSE_SKIPPED_RUN_SIGNATURE_KEY: str(run_signature or ""),
        "step5_universe_apply_message_v1": (
            f"Universe composition kept unchanged for this run. Skipped recommendation: {label}."
        ),
        STEP5_SCROLL_TO_RESULT_AFTER_APPLY_KEY: True,
    }
    if callable(queue_and_rerun):
        queue_and_rerun(patch)
        return
    for key, value in patch.items():
        st.session_state[key] = value
    st.rerun()


def _candidate_table(evaluations: list[dict], current_perf: dict) -> pd.DataFrame:
    rows: list[dict] = []
    base = _normalise_perf(current_perf)
    accepted_seen = 0
    for raw in evaluations:
        item = _coerce_mapping(raw)
        perf = _normalise_perf(item.get("performance_summary", {}))
        accepted = bool(item.get("accepted", False))
        accepted_seen += 1 if accepted else 0
        status = "Recommended" if accepted and accepted_seen == 1 else ("Passed gate" if accepted else "Not selected")
        vol_delta = perf.get("annual_volatility", 0.0) - base.get("annual_volatility", 0.0)
        maxdd_improvement = base.get("max_drawdown", 0.0) - perf.get("max_drawdown", 0.0)
        assets = _asset_list(item.get("assets", []))
        rows.append(
            {
                "candidate": str(item.get("label", "Candidate") or "Candidate"),
                "status": status,
                "family": str(item.get("family", "") or ""),
                "assets": int(len(assets)),
                "overlap vs current": f"{100.0 * _safe_float(item.get('overlap_vs_current'), 0.0):.0f}%",
                "CAGR": _format_pct(perf.get("cagr", 0.0)),
                "Δ CAGR": _format_delta_pct(perf.get("cagr", 0.0) - base.get("cagr", 0.0)),
                "Vol": _format_pct(perf.get("annual_volatility", 0.0)),
                "Δ Vol": _format_delta_pct(vol_delta),
                "MaxDD": f"-{100.0 * abs(perf.get('max_drawdown', 0.0)):.2f}%",
                "DD improvement": _format_delta_pct(maxdd_improvement),
                "Sharpe": f"{perf.get('sharpe', 0.0):.2f}",
                "Δ Sharpe": f"{perf.get('sharpe', 0.0) - base.get('sharpe', 0.0):+.2f}",
                "score delta": f"{_safe_float(item.get('score_delta'), 0.0):+.3f}",
                "seconds": f"{_safe_float(item.get('elapsed_sec'), 0.0):.2f}s",
                "gate": str(item.get("gate_reason", "") or ""),
                "error": str(item.get("error", "") or ""),
            }
        )
    return pd.DataFrame(rows)


def _render_recommended_candidate(candidate: dict, current_perf: dict) -> None:
    item = _coerce_mapping(candidate)
    perf = _normalise_perf(item.get("performance_summary", {}))
    current = _normalise_perf(current_perf)
    assets = _asset_list(item.get("assets", []))

    st.markdown("### Recommended universe composition")
    st.markdown(f"**{item.get('label', 'Universe composition candidate')}**")
    caption = str(item.get("caption", "") or "")
    if caption:
        st.caption(caption)

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

    st.success("This candidate passed the universe-composition acceptance gate.")
    gate_reason = str(item.get("gate_reason", "") or "")
    if gate_reason:
        st.caption(gate_reason)
    if item.get("error"):
        st.warning(str(item.get("error")))

    with st.expander("Recommended universe assets", expanded=False):
        if assets:
            preview = ", ".join(asset_display_label(x) for x in assets[:40])
            if len(assets) > 40:
                preview += f" ... +{len(assets) - 40} more"
            st.write(preview)
        st.caption(
            f"size={item.get('universe_size', '—')} · strategy={item.get('universe_strategy', '—')} · "
            f"overlap_vs_current={100.0 * _safe_float(item.get('overlap_vs_current'), 0.0):.0f}% · "
            f"score_delta={_safe_float(item.get('score_delta'), 0.0):+.3f}"
        )


def render_universe_improvement(run_result: dict) -> None:
    """Render the third Step 5 improvement phase: universe composition.

    The tested candidates change only the asset composition. They keep the active
    Step 5 strategy preset, technical engine config, and universe size unchanged.
    """
    run_map = _coerce_mapping(run_result)
    perf = _normalise_perf(run_map.get("performance_summary", {}))
    if not perf:
        return

    st.markdown("### Universe composition suggestion")
    st.caption(
        "This keeps the current strategy preset, technical engine configuration, and universe size, "
        "then tests a small number of alternative asset compositions using only assets already present in the Step 4 data panel."
    )

    msg = st.session_state.pop("step5_universe_apply_message_v1", "")

    current_run_signature = str(run_map.get("run_signature", "") or "")
    applied_signature = str(st.session_state.get(UNIVERSE_APPLIED_SIGNATURE_KEY, "") or "")
    applied_label = str(st.session_state.get(UNIVERSE_APPLIED_LABEL_KEY, "") or "")
    if applied_signature and current_run_signature and applied_signature == current_run_signature:
        label = applied_label or "the accepted universe composition suggestion"
        st.success(
            f"Universe composition applied: {label}. The rerun-tested candidate is now the current Step 5 result."
        )
        st.caption(
            "Universe testing is hidden for this run to avoid suggesting the same loop again. "
            "Change Step 4 or run a new baseline if you want to test a different universe."
        )
        return

    if msg:
        st.success(str(msg))

    scope = _build_scope(run_map)
    saved_scope = str(st.session_state.get(UNIVERSE_SUGGESTION_SCOPE_KEY, "") or "")
    payload = _coerce_mapping(st.session_state.get(UNIVERSE_SUGGESTION_STATE_KEY, {}))
    evaluations = list(payload.get("evaluations", []) or []) if saved_scope == scope else []

    if saved_scope != scope or not evaluations:
        with st.spinner("Testing alternative universe compositions with the real engine..."):
            payload = _run_universe_search(run_map, max_candidates=4)
        st.session_state[UNIVERSE_SUGGESTION_STATE_KEY] = payload
        st.session_state[UNIVERSE_SUGGESTION_SCOPE_KEY] = str(payload.get("scope", scope))
        st.session_state[UNIVERSE_SUGGESTION_TIMING_KEY] = _coerce_mapping(payload.get("timing_summary", {}))
        evaluations = list(payload.get("evaluations", []) or [])

    if not evaluations:
        error = str(payload.get("error", "") or "")
        if error:
            st.info(f"No safe universe candidates were available to test for this run. {error}")
        else:
            st.info("No safe universe candidates were available to test for this run.")
        return

    accepted_items = [dict(x) for x in evaluations if bool(_coerce_mapping(x).get("accepted", False))]
    table = _candidate_table(evaluations, perf)
    skipped_scope = str(st.session_state.get(UNIVERSE_SKIPPED_SCOPE_KEY, "") or "")
    skipped_label = str(st.session_state.get(UNIVERSE_SKIPPED_LABEL_KEY, "") or "")
    skipped_run_signature = str(st.session_state.get(UNIVERSE_SKIPPED_RUN_SIGNATURE_KEY, "") or "")
    universe_was_skipped = bool(
        skipped_scope
        and skipped_scope == scope
        and (not skipped_run_signature or skipped_run_signature == current_run_signature)
    )

    if not accepted_items:
        st.success("Current universe composition has converged: no tested alternative materially improved this run.")
        st.caption(
            "The tested universe candidates are kept below for transparency, but none is offered as an action because "
            "the acceptance gate did not find a better trade-off."
        )
    elif universe_was_skipped:
        st.info("Current universe composition kept for this run. The improvement flow is complete.")
        if skipped_label:
            st.caption(f"Skipped universe recommendation: {skipped_label}.")
    else:
        best_candidate = accepted_items[0]
        st.success("Recommended universe composition found. The best accepted candidate is shown below.")
        _render_recommended_candidate(best_candidate, perf)
        st.caption(
            "The diagnostics table shows every universe candidate tested by the engine. Only the best accepted candidate "
            "is offered as the main action; rejected candidates are shown for transparency, not as recommendations."
        )

    with st.expander("Universe composition diagnostics", expanded=False):
        st.caption(
            f"tested_candidates={len(evaluations)} · elapsed={_safe_float(payload.get('elapsed_sec', 0.0), 0.0):.2f}s · "
            f"scope={str(payload.get('scope', scope))}"
        )
        if not table.empty:
            st.dataframe(table, use_container_width=True, hide_index=True)

    if accepted_items and not universe_was_skipped:
        best_candidate = accepted_items[0]
        left, right = st.columns(2)
        with left:
            if st.button("Apply recommended universe", key="step5_apply_best_universe_candidate_v1", use_container_width=True):
                _apply_candidate(best_candidate)
        with right:
            if st.button("Keep current universe and continue", key="step5_skip_best_universe_candidate_v1", use_container_width=True):
                _skip_current_universe_candidate(
                    scope,
                    str(best_candidate.get("label", "recommended universe") or "recommended universe"),
                    current_run_signature,
                )


# Compatibility wrapper for older imports.
def render_universe_recommendations(run_result: dict | None = None, *args: Any, **kwargs: Any) -> None:
    if isinstance(run_result, dict):
        render_universe_improvement(run_result)
    return None
