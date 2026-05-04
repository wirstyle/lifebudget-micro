"""Step 5 engine-tuning recommendation phase.

This module implements Phase 2 of the optional Strategy Engine improvement flow.
It keeps the selected universe, asset list, universe size, strategy template and
style preset unchanged, then tests a small whitelist of technical engine knobs
with the real engine.

Accepted candidates are applied by promoting an already rerun-tested result, so
the user does not need to manually run the same candidate again.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import fields
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st

from src.investment import MicroPipelineConfig, run_micro_investment_pipeline

try:
    from ui.state.updates import queue_and_rerun
except Exception:  # pragma: no cover
    queue_and_rerun = None


AUTO_OPT_SUGGESTION_STATE_KEY = "step5_auto_opt_suggestion_v1"
AUTO_OPT_SUGGESTION_SCOPE_KEY = "step5_auto_opt_suggestion_scope_v1"
AUTO_OPT_APPLIED_SIGNATURE_KEY = "step5_auto_opt_applied_run_signature_v1"
AUTO_OPT_APPLIED_LABEL_KEY = "step5_auto_opt_applied_label_v1"
AUTO_OPT_SKIPPED_SCOPE_KEY = "step5_auto_opt_skipped_scope_v1"
AUTO_OPT_SKIPPED_LABEL_KEY = "step5_auto_opt_skipped_label_v1"
AUTO_OPT_SKIPPED_RUN_SIGNATURE_KEY = "step5_auto_opt_skipped_run_signature_v1"
AUTO_OPT_SUGGESTION_TIMING_KEY = "step5_auto_opt_suggestion_timing_v1"
STEP5_SCROLL_TO_RESULT_AFTER_APPLY_KEY = "step5_scroll_to_real_run_result_after_apply_v1"
AUTO_OPT_SUGGESTION_COUNT_KEY = "step5_suggestion_count_auto_opt_v1"
AUTO_OPT_SECONDS_PER_TEST = 30.0


def _suggestion_candidate_count(default: int = 2) -> int:
    try:
        raw = int(st.session_state.get(AUTO_OPT_SUGGESTION_COUNT_KEY, default))
    except Exception:
        raw = int(default)
    return int(max(1, min(6, raw)))


def _format_runtime_estimate(seconds: float) -> str:
    try:
        seconds = float(seconds)
    except Exception:
        seconds = 60.0
    if seconds <= 40:
        return "~30s"
    if seconds <= 70:
        return "~1 min"
    if seconds <= 105:
        return "~90s"
    if seconds <= 150:
        return "~2 min"
    minutes = seconds / 60.0
    rounded = round(minutes * 2.0) / 2.0
    return f"~{rounded:g} min"

# Only these low-level fields may be changed by Phase 2. Strategy preset,
# universe composition and universe size are reserved for other phases.
SAFE_TUNING_FIELDS: tuple[str, ...] = (
    "top_k",
    "lookback_mu",
    "lookback_sigma",
    "temperature",
    "weight_shrink",
    "inertia",
)

FIELD_TO_WIDGET_KEY: dict[str, str] = {
    "top_k": "step5_basic_top_k",
    "lookback_mu": "step5_basic_lookback_mu",
    "lookback_sigma": "step5_basic_lookback_sigma",
    "temperature": "step5_basic_temperature",
    "weight_shrink": "step5_basic_weight_shrink",
    "inertia": "step5_basic_inertia",
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


def _clip_int(value: Any, low: int, high: int, default: int) -> int:
    raw = _safe_int(value, default)
    return int(max(int(low), min(int(high), raw)))


def _clip_float(value: Any, low: float, high: float, default: float) -> float:
    raw = _safe_float(value, default)
    return float(max(float(low), min(float(high), raw)))


def _stable_panel_fingerprint(panel_df: pd.DataFrame) -> str:
    if panel_df is None or not isinstance(panel_df, pd.DataFrame) or panel_df.empty:
        return "empty"
    payload = {
        "rows": int(len(panel_df)),
        "cols": list(map(str, panel_df.columns)),
        "assets": sorted(panel_df["asset"].dropna().astype(str).unique().tolist()) if "asset" in panel_df.columns else [],
        "min_date": str(panel_df["date"].min()) if "date" in panel_df.columns else "",
        "max_date": str(panel_df["date"].max()) if "date" in panel_df.columns else "",
    }
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(encoded.encode("utf-8")).hexdigest()[:16]


def _resolve_run_signature_for_cfg(cfg_final: dict, asset_panel_df: Any) -> str:
    payload = {
        "cfg_final": _coerce_mapping(cfg_final),
        "universe_size": st.session_state.get("universe_size"),
        "universe_strategy": st.session_state.get("universe_strategy"),
        "selected_assets": list(st.session_state.get("selected_assets", []) or []),
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
        return sorted(panel_df["asset"].dropna().astype(str).unique().tolist())
    return []


def _base_cfg_payload_from_run(run_result: dict) -> dict:
    """Prefer the full engine config over the UI-only config.

    run_panel stores the full MicroPipelineConfig as config_dict/last_engine_config.
    Using it here avoids accidentally resetting hidden engine defaults while changing
    only the SAFE_TUNING_FIELDS.
    """
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


def _technical_summary(cfg_payload: dict) -> dict:
    payload = _coerce_mapping(cfg_payload)
    return {field: payload.get(field) for field in SAFE_TUNING_FIELDS}


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
            delta_score > 0.004
            and delta_dd <= 0.004
            and delta_vol <= 0.006
            and delta_sharpe >= -0.015
            and delta_cagr >= -0.012
        ) or (
            (delta_dd <= -0.010 or delta_vol <= -0.006)
            and delta_sharpe >= -0.010
            and delta_cagr >= -0.015
        )
        reason = "Defensive gate prioritises lower drawdown/volatility and avoids sacrificing Sharpe or too much CAGR."
    elif profile == "growth":
        passed = (
            delta_score > 0.004
            and delta_cagr >= -0.004
            and delta_sharpe >= -0.060
            and delta_dd <= 0.050
            and delta_vol <= 0.030
        ) or (
            delta_cagr >= 0.008
            and delta_sharpe >= -0.040
            and delta_dd <= 0.050
        )
        reason = "Growth gate gives more weight to CAGR, while rejecting candidates that damage Sharpe or drawdown too much."
    else:
        passed = (
            delta_score > 0.004
            and delta_sharpe >= -0.030
            and delta_cagr >= -0.008
            and delta_dd <= 0.020
            and delta_vol <= 0.015
        ) or (
            delta_sharpe >= 0.050
            and delta_dd <= 0.020
            and delta_vol <= 0.020
        )
        reason = "Balanced gate looks for a better overall trade-off without materially worsening Sharpe, drawdown, or volatility."

    return bool(passed), reason


def _apply_patch_to_cfg(base_cfg: dict, patch: dict, *, universe_size: int) -> dict:
    out = dict(base_cfg or {})
    max_top_k = max(1, int(universe_size or 1))
    if "top_k" in patch:
        out["top_k"] = _clip_int(patch["top_k"], 1, max_top_k, _safe_int(out.get("top_k", min(12, max_top_k)), min(12, max_top_k)))
    if "lookback_mu" in patch:
        out["lookback_mu"] = _clip_int(patch["lookback_mu"], 3, 60, _safe_int(out.get("lookback_mu", 12), 12))
    if "lookback_sigma" in patch:
        out["lookback_sigma"] = _clip_int(patch["lookback_sigma"], 3, 60, _safe_int(out.get("lookback_sigma", 12), 12))
    if "temperature" in patch:
        out["temperature"] = round(_clip_float(patch["temperature"], 0.01, 5.0, _safe_float(out.get("temperature", 1.0), 1.0)), 4)
    if "weight_shrink" in patch:
        out["weight_shrink"] = round(_clip_float(patch["weight_shrink"], 0.0, 1.0, _safe_float(out.get("weight_shrink", 0.05), 0.05)), 4)
    if "inertia" in patch:
        out["inertia"] = round(_clip_float(patch["inertia"], 0.0, 1.0, _safe_float(out.get("inertia", 0.0), 0.0)), 4)
    return _coerce_cfg_payload(out)


def _candidate_specs(base_cfg: dict, philosophy: str, universe_size: int, *, max_candidates: int = 3) -> list[dict]:
    max_top_k = max(1, int(universe_size or 1))
    current_top_k = _clip_int(base_cfg.get("top_k", min(12, max_top_k)), 1, max_top_k, min(12, max_top_k))
    current_mu = _clip_int(base_cfg.get("lookback_mu", 12), 3, 60, 12)
    current_sigma = _clip_int(base_cfg.get("lookback_sigma", 12), 3, 60, 12)
    current_temp = _clip_float(base_cfg.get("temperature", 1.0), 0.01, 5.0, 1.0)
    current_shrink = _clip_float(base_cfg.get("weight_shrink", 0.05), 0.0, 1.0, 0.05)
    current_inertia = _clip_float(base_cfg.get("inertia", 0.0), 0.0, 1.0, 0.0)

    def spec(label: str, family: str, patch: dict, caption: str) -> dict:
        # max_top_k is the effective upper bound for top_k under the current universe.
        cfg_payload = _apply_patch_to_cfg(base_cfg, patch, universe_size=max_top_k)
        changed = {
            field: cfg_payload.get(field)
            for field in SAFE_TUNING_FIELDS
            if cfg_payload.get(field) != base_cfg.get(field)
        }
        return {
            "label": label,
            "family": family,
            "caption": caption,
            "technical_patch": changed,
            "cfg_payload": cfg_payload,
        }

    smoother = spec(
        "Smoother risk-control tuning",
        "risk_control",
        {
            "top_k": min(max_top_k, max(current_top_k + 2, int(round(current_top_k * 1.15)))),
            "lookback_mu": current_mu + 6,
            "lookback_sigma": current_sigma + 6,
            "temperature": current_temp + 0.15,
            "weight_shrink": current_shrink + 0.05,
            "inertia": current_inertia + 0.04,
        },
        "Tests slightly broader selection, longer lookbacks, more equal-weight shrink and a little more inertia.",
    )
    responsive = spec(
        "More responsive signal tuning",
        "responsiveness",
        {
            "top_k": max(1, min(max_top_k, current_top_k - 2)),
            "lookback_mu": current_mu - 3,
            "lookback_sigma": current_sigma - 2,
            "temperature": current_temp - 0.12,
            "weight_shrink": current_shrink - 0.03,
            "inertia": current_inertia - 0.03,
        },
        "Tests a slightly faster and more concentrated technical posture without changing the strategy preset.",
    )
    balanced = spec(
        "Balanced middle-ground tuning",
        "middle_ground",
        {
            "top_k": min(max_top_k, max(current_top_k, min(max_top_k, 12))),
            "lookback_mu": 18 if current_mu <= 12 else 12,
            "lookback_sigma": 18 if current_sigma <= 12 else 12,
            "temperature": 1.05 if current_temp <= 1.0 else 0.95,
            "weight_shrink": 0.08 if current_shrink < 0.08 else max(0.04, current_shrink - 0.03),
            "inertia": 0.06 if current_inertia < 0.06 else max(0.02, current_inertia - 0.03),
        },
        "Tests a neutral technical variation around the current engine setup.",
    )
    defensive = spec(
        "Lower-volatility technical tuning",
        "defensive_smoothing",
        {
            "top_k": min(max_top_k, current_top_k + 4),
            "lookback_mu": max(current_mu + 6, 18),
            "lookback_sigma": max(current_sigma + 9, 21),
            "temperature": current_temp + 0.25,
            "weight_shrink": current_shrink + 0.08,
            "inertia": current_inertia + 0.08,
        },
        "Tests a more diversified, smoother configuration designed to reduce volatility and drawdown sensitivity.",
    )
    growth = spec(
        "Higher-conviction growth tuning",
        "growth_conviction",
        {
            "top_k": max(1, current_top_k - 3),
            "lookback_mu": max(6, current_mu - 4),
            "lookback_sigma": max(9, current_sigma),
            "temperature": current_temp - 0.18,
            "weight_shrink": current_shrink - 0.04,
            "inertia": current_inertia - 0.04,
        },
        "Tests a more focused technical posture intended to preserve upside while still passing the risk gate.",
    )

    profile = str(philosophy or "Balanced").strip().lower()
    if profile == "growth":
        ordered = [growth, responsive, balanced, smoother]
    elif profile == "defensive":
        ordered = [defensive, smoother, balanced, responsive]
    else:
        ordered = [balanced, smoother, responsive, defensive]

    unique: list[dict] = []
    seen: set[str] = set()
    base_summary = _technical_summary(base_cfg)
    for item in ordered:
        item_map = _coerce_mapping(item)
        cfg_payload = _coerce_cfg_payload(item_map.get("cfg_payload", {}))
        summary = _technical_summary(cfg_payload)
        if summary == base_summary:
            continue
        fp = json.dumps(summary, sort_keys=True, default=str)
        if fp in seen:
            continue
        seen.add(fp)
        item_map["cfg_payload"] = cfg_payload
        item_map["technical_patch"] = {
            field: cfg_payload.get(field)
            for field in SAFE_TUNING_FIELDS
            if cfg_payload.get(field) != base_cfg.get(field)
        }
        unique.append(item_map)
        if len(unique) >= max(1, int(max_candidates)):
            break
    return unique[: max(1, int(max_candidates))]


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
    panel = st.session_state.get("asset_panel_df")
    panel_df = panel.copy() if isinstance(panel, pd.DataFrame) else pd.DataFrame()
    base_cfg = _base_cfg_payload_from_run(run_map)
    template, style = _current_template_style()
    return _hash_scope({
        "run_signature": str(run_map.get("run_signature", "") or ""),
        "config_fingerprint": str(run_map.get("config_fingerprint", "") or ""),
        "philosophy": _philosophy(),
        "template": template,
        "style": style,
        "technical": _technical_summary(base_cfg),
        "suggestion_candidates": _suggestion_candidate_count(2),
        "assets": _panel_assets(panel_df),
        "rows": int(len(panel_df)) if isinstance(panel_df, pd.DataFrame) else 0,
        "panel_fp": _stable_panel_fingerprint(panel_df),
        "safe_fields": SAFE_TUNING_FIELDS,
    })


def _run_auto_opt_search(run_result: dict, *, max_candidates: int = 3) -> dict:
    run_map = _coerce_mapping(run_result)
    current_perf = _normalise_perf(run_map.get("performance_summary", {}))
    philosophy = _philosophy()
    template, style = _current_template_style()

    panel = st.session_state.get("asset_panel_df")
    panel_df = panel.copy() if isinstance(panel, pd.DataFrame) else pd.DataFrame()
    if panel_df.empty:
        return {"scope": _build_scope(run_result), "evaluations": [], "error": "Step 4 asset panel is missing."}

    base_cfg = _base_cfg_payload_from_run(run_map)
    universe_size = len(_panel_assets(panel_df)) or _safe_int(st.session_state.get("universe_size", 25), 25)
    candidates = _candidate_specs(base_cfg, philosophy, universe_size, max_candidates=max_candidates)

    evaluations: list[dict] = []
    started = time.perf_counter()

    for candidate in candidates:
        candidate_started = time.perf_counter()
        candidate_map = _coerce_mapping(candidate)
        cfg_payload = _coerce_cfg_payload(candidate_map.get("cfg_payload", {}))
        run = _run_candidate(cfg_payload, panel_df)
        candidate_elapsed_sec = float(time.perf_counter() - candidate_started)
        perf = _normalise_perf(run.get("performance_summary", {}))
        accepted = False
        gate_reason = "Candidate could not be evaluated."
        if perf and not run.get("error"):
            accepted, gate_reason = _acceptance_gate(perf, current_perf, philosophy)

        evaluations.append(
            {
                "label": str(candidate_map.get("label", "Technical tuning candidate") or "Technical tuning candidate"),
                "family": str(candidate_map.get("family", "technical") or "technical"),
                "caption": str(candidate_map.get("caption", "") or ""),
                "strategy_template": template,
                "style_preset": style,
                "technical_patch": _coerce_mapping(candidate_map.get("technical_patch", {})),
                "cfg_payload": dict(cfg_payload),
                "performance_summary": dict(perf),
                "raw_result": _coerce_mapping(run.get("raw", {})),
                "score": _score_perf(perf, philosophy) if perf else None,
                "score_delta": (_score_perf(perf, philosophy) - _score_perf(current_perf, philosophy)) if perf else None,
                "accepted": bool(accepted),
                "gate_reason": gate_reason,
                "error": str(run.get("error", "") or ""),
                "elapsed_sec": float(candidate_elapsed_sec),
                "apply_patch": _build_apply_patch(cfg_payload),
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
        "current_perf": dict(current_perf),
        "current_technical": _technical_summary(base_cfg),
        "evaluations": evaluations,
        "candidate_count": int(len(evaluations)),
        "safe_tuning_fields": list(SAFE_TUNING_FIELDS),
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


def _build_apply_patch(cfg_payload: Any) -> dict:
    payload = _coerce_cfg_payload(cfg_payload)
    return {FIELD_TO_WIDGET_KEY[field]: payload[field] for field in SAFE_TUNING_FIELDS if field in payload and field in FIELD_TO_WIDGET_KEY}


def _normalise_promoted_candidate_result(candidate: dict, cfg_payload: dict) -> dict:
    candidate_map = _coerce_mapping(candidate)
    raw_result = _coerce_mapping(candidate_map.get("raw_result", {}))
    if not raw_result:
        return {}

    panel = st.session_state.get("asset_panel_df")
    panel_df = panel.copy() if isinstance(panel, pd.DataFrame) else pd.DataFrame()
    run_signature = _resolve_run_signature_for_cfg(cfg_payload, panel_df)
    config_fingerprint = _resolve_config_fingerprint_for_cfg(cfg_payload)
    run_timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    perf = _normalise_perf(candidate_map.get("performance_summary", raw_result.get("performance_summary", {})))

    promoted = dict(raw_result)
    promoted.update(
        {
            "source": "micro_pipeline_real",
            "promoted_from_auto_opt_candidate": True,
            "auto_opt_candidate_label": str(candidate_map.get("label", "Technical tuning candidate") or "Technical tuning candidate"),
            "asset_panel_source_label": str(st.session_state.get("asset_panel_source_label", "Step 4 asset panel") or "Step 4 asset panel"),
            "asset_panel_n_rows": int(len(panel_df)) if isinstance(panel_df, pd.DataFrame) else 0,
            "asset_panel_n_assets": int(panel_df["asset"].nunique()) if isinstance(panel_df, pd.DataFrame) and "asset" in panel_df.columns else 0,
            "performance_summary": dict(perf),
            "config": dict(cfg_payload),
            "config_dict": dict(cfg_payload),
            "run_signature": run_signature,
            "config_fingerprint": config_fingerprint,
            "run_timestamp": run_timestamp,
            "panel_shape": (int(len(panel_df)), int(len(panel_df.columns))) if isinstance(panel_df, pd.DataFrame) else (0, 0),
            "evaluation_period_label": "Walk-forward evaluated period",
            "search_eval_split": {"enabled": False, "reason": "auto_opt_candidate_promoted_after_rerun_test"},
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
            "promoted_from_auto_opt_candidate": True,
            "breakdown_df": pd.DataFrame(),
        },
    )
    return promoted


def _apply_candidate(candidate: dict) -> None:
    candidate_map = _coerce_mapping(candidate)
    patch = _coerce_mapping(candidate_map.get("apply_patch", {}))
    if not patch:
        st.warning("This candidate does not expose an apply patch.")
        return

    cfg_payload = _coerce_cfg_payload(candidate_map.get("cfg_payload", {}))
    promoted_result = _normalise_promoted_candidate_result(candidate_map, cfg_payload)

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
                AUTO_OPT_SUGGESTION_STATE_KEY: {},
                AUTO_OPT_SUGGESTION_SCOPE_KEY: "",
                AUTO_OPT_APPLIED_SIGNATURE_KEY: run_signature,
                AUTO_OPT_APPLIED_LABEL_KEY: str(candidate_map.get("label", "Technical tuning candidate") or "Technical tuning candidate"),
                "step5_basic_engine_controls_touched": True,
                "step5_basic_engine_controls_source": "auto_opt_suggestion",
                "step5_auto_opt_apply_message_v1": (
                    f"Engine tuning suggestion applied using the rerun-tested candidate result: "
                    f"{candidate_map.get('label', 'Technical tuning candidate')}."
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
                AUTO_OPT_APPLIED_SIGNATURE_KEY: "",
                AUTO_OPT_APPLIED_LABEL_KEY: "",
                "step5_basic_engine_controls_touched": True,
                "step5_basic_engine_controls_source": "auto_opt_suggestion",
                "step5_auto_opt_apply_message_v1": (
                    f"Engine tuning suggestion applied: {candidate_map.get('label', 'Technical tuning candidate')}. "
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


def _skip_current_auto_opt_candidate(scope: str, label: str, run_signature: str) -> None:
    """Mark the current engine-tuning recommendation as skipped for this run.

    This preserves the tested diagnostics/timing but allows the sequential flow to
    continue to Universe composition without applying technical changes.
    """
    patch = {
        AUTO_OPT_SKIPPED_SCOPE_KEY: str(scope or ""),
        AUTO_OPT_SKIPPED_LABEL_KEY: str(label or "recommended tuning"),
        AUTO_OPT_SKIPPED_RUN_SIGNATURE_KEY: str(run_signature or ""),
        "step5_auto_opt_apply_message_v1": (
            f"Engine tuning kept unchanged for this run. Skipped recommendation: {label}."
        ),
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
        technical_patch = _coerce_mapping(item.get("technical_patch", {}))
        rows.append(
            {
                "candidate": str(item.get("label", "Candidate") or "Candidate"),
                "status": status,
                "family": str(item.get("family", "") or ""),
                "changed knobs": ", ".join(technical_patch.keys()) if technical_patch else "—",
                "CAGR": _format_pct(perf.get("cagr", 0.0)),
                "Δ CAGR": _format_delta_pct(perf.get("cagr", 0.0) - base.get("cagr", 0.0)),
                "Vol": _format_pct(perf.get("annual_volatility", 0.0)),
                "Δ Vol": _format_delta_pct(vol_delta),
                "MaxDD": f"-{100.0 * abs(perf.get('max_drawdown', 0.0)):.2f}%",
                "Drawdown change": _format_delta_pct(maxdd_improvement),
                "Sharpe": f"{perf.get('sharpe', 0.0):.2f}",
                "Δ Sharpe": f"{perf.get('sharpe', 0.0) - base.get('sharpe', 0.0):+.2f}",
                "Δ Score": f"{_safe_float(item.get('score_delta'), 0.0):+.3f}",
                "Seconds": f"{_safe_float(item.get('elapsed_sec'), 0.0):.2f}s",
            }
        )
    return pd.DataFrame(rows)


def _render_recommended_candidate(candidate: dict, current_perf: dict) -> None:
    """Render the actionable engine-tuning candidate as a compact card."""
    item = _coerce_mapping(candidate)
    perf = _normalise_perf(item.get("performance_summary", {}))
    current = _normalise_perf(current_perf)

    st.markdown("### Recommended engine tuning")
    st.markdown(f"**{item.get('label', 'Technical tuning candidate')}**")

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


def render_auto_opt_improvement(run_result: dict) -> None:
    """Render the second Step 5 improvement phase: technical engine tuning.

    The tested candidates change only SAFE_TUNING_FIELDS. They keep the current
    selected universe, asset list, universe size, strategy template, and style
    preset unchanged.
    """
    run_map = _coerce_mapping(run_result)
    perf = _normalise_perf(run_map.get("performance_summary", {}))
    if not perf:
        return

    # The phase rail already labels this as Engine tuning. Keep this panel action-first.

    msg = st.session_state.pop("step5_auto_opt_apply_message_v1", "")

    current_run_signature = str(run_map.get("run_signature", "") or "")
    applied_signature = str(st.session_state.get(AUTO_OPT_APPLIED_SIGNATURE_KEY, "") or "")
    applied_label = str(st.session_state.get(AUTO_OPT_APPLIED_LABEL_KEY, "") or "")
    if applied_signature and current_run_signature and applied_signature == current_run_signature:
        label = applied_label or "the accepted engine tuning suggestion"
        st.success(
            f"Engine tuning applied: {label}. The rerun-tested candidate is now the current strategy engine result."
        )
        st.caption(
            "Engine tuning is hidden for this run to avoid suggesting the same loop again. "
            "Change the strategy setup or technical controls if you want to test tuning again."
        )
        return

    if msg:
        st.success(str(msg))

    scope = _build_scope(run_map)
    saved_scope = str(st.session_state.get(AUTO_OPT_SUGGESTION_SCOPE_KEY, "") or "")
    payload = _coerce_mapping(st.session_state.get(AUTO_OPT_SUGGESTION_STATE_KEY, {}))
    evaluations = list(payload.get("evaluations", []) or []) if saved_scope == scope else []

    skipped_scope = str(st.session_state.get(AUTO_OPT_SKIPPED_SCOPE_KEY, "") or "")
    skipped_label = str(st.session_state.get(AUTO_OPT_SKIPPED_LABEL_KEY, "") or "")
    skipped_run_signature = str(st.session_state.get(AUTO_OPT_SKIPPED_RUN_SIGNATURE_KEY, "") or "")
    tuning_was_skipped = bool(
        skipped_scope
        and skipped_scope == scope
        and (not skipped_run_signature or skipped_run_signature == current_run_signature)
    )
    if tuning_was_skipped and not evaluations:
        st.warning("Engine tuning skipped: current technical setup kept for this run. Universe composition can now be tested on the existing setup.")
        if skipped_label:
            st.caption(f"Skipped engine tuning check: {skipped_label}.")
        return

    if saved_scope != scope or not evaluations:
        max_candidates = _suggestion_candidate_count(2)
        estimate = _format_runtime_estimate(AUTO_OPT_SECONDS_PER_TEST * max_candidates)
        label = f"{max_candidates} engine-tuning alternative" if max_candidates == 1 else f"{max_candidates} engine-tuning alternatives"
        with st.spinner(f"Testing {label} ({estimate})..."):
            payload = _run_auto_opt_search(run_map, max_candidates=max_candidates)
        st.session_state[AUTO_OPT_SUGGESTION_STATE_KEY] = payload
        st.session_state[AUTO_OPT_SUGGESTION_SCOPE_KEY] = str(payload.get("scope", scope))
        st.session_state[AUTO_OPT_SUGGESTION_TIMING_KEY] = _coerce_mapping(payload.get("timing_summary", {}))
        evaluations = list(payload.get("evaluations", []) or [])

    if not evaluations:
        st.info("No safe technical tuning candidates were available to test for this run.")
        if st.button("Continue with current tuning", key="step5_continue_auto_opt_no_candidates_v1", use_container_width=True):
            _skip_current_auto_opt_candidate(scope, "no safe tuning candidates", current_run_signature)
        return

    accepted_items = [dict(x) for x in evaluations if bool(_coerce_mapping(x).get("accepted", False))]
    table = _candidate_table(evaluations, perf)
    skipped_scope = str(st.session_state.get(AUTO_OPT_SKIPPED_SCOPE_KEY, "") or "")
    skipped_label = str(st.session_state.get(AUTO_OPT_SKIPPED_LABEL_KEY, "") or "")
    skipped_run_signature = str(st.session_state.get(AUTO_OPT_SKIPPED_RUN_SIGNATURE_KEY, "") or "")
    tuning_was_skipped = bool(
        skipped_scope
        and skipped_scope == scope
        and (not skipped_run_signature or skipped_run_signature == current_run_signature)
    )

    if not accepted_items:
        st.success("Current engine tuning has converged: no tested technical variation materially improved this run.")
        st.caption(
            "The tested technical candidates are kept below for transparency, but none is offered as an action because "
            "the acceptance gate did not find a better trade-off."
        )
    elif tuning_was_skipped:
        st.warning("Engine tuning skipped: current technical setup kept for this run. Universe composition can now be tested on the existing setup.")
        if skipped_label:
            st.caption(f"Skipped engine tuning recommendation: {skipped_label}.")
    else:
        best_candidate = accepted_items[0]
        _render_recommended_candidate(best_candidate, perf)

        with st.expander("How to decide on this engine-tuning recommendation", expanded=False):
            st.info(
                "**Decision guide:** this keeps the same strategy preset, risk style, Step 4 universe and universe size. "
                "It only adjusts small technical engine knobs. Apply it only if the rerun-tested trade-off feels better overall, "
                "not because one isolated metric improved."
            )
            st.divider()
            st.success("This candidate passed the engine-tuning acceptance gate.")

            if not table.empty:
                clean_cols = [
                    "candidate",
                    "status",
                    "CAGR",
                    "Δ CAGR",
                    "Vol",
                    "Δ Vol",
                    "MaxDD",
                    "Drawdown change",
                    "Sharpe",
                    "Δ Sharpe",
                ]
                clean_cols = [col for col in clean_cols if col in table.columns]
                st.dataframe(table[clean_cols], use_container_width=True, hide_index=True)

            caption = str(best_candidate.get("caption", "") or "").strip()
            gate_reason = str(best_candidate.get("gate_reason", "") or "").strip()
            rationale_bits = []
            if caption:
                rationale_bits.append(caption.rstrip("."))
            if gate_reason:
                rationale_bits.append(gate_reason.rstrip("."))
            if rationale_bits:
                st.write(". ".join(rationale_bits) + ".")
            else:
                st.write(
                    "This tests a small technical variation around the current engine setup. "
                    "The acceptance gate passed it because the overall trade-off improved without materially worsening Sharpe, drawdown or volatility."
                )

            if best_candidate.get("error"):
                st.warning(str(best_candidate.get("error")))

            technical_patch = _coerce_mapping(best_candidate.get("technical_patch", {}))
            if technical_patch:
                st.markdown("**Recommended technical settings**")
                rows = [
                    {"Technical setting": key, "Recommended value": value}
                    for key, value in technical_patch.items()
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        left, right = st.columns(2)
        with left:
            if st.button("Apply recommended tuning", key="step5_apply_best_auto_opt_candidate_v1", use_container_width=True):
                _apply_candidate(best_candidate)
        with right:
            if st.button("Keep current tuning and continue", key="step5_skip_best_auto_opt_candidate_v1", use_container_width=True):
                _skip_current_auto_opt_candidate(
                    scope,
                    str(best_candidate.get("label", "recommended tuning") or "recommended tuning"),
                    current_run_signature,
                )


    # Advanced engine-tuning diagnostics intentionally stay out of the main decision surface.
    # Timing/candidate diagnostics are available from the Step 5 run diagnostics area.

    if not accepted_items and not tuning_was_skipped:
        if st.button("Continue with current tuning", key="step5_continue_current_auto_opt_v1", use_container_width=True):
            _skip_current_auto_opt_candidate(scope, "no accepted tuning candidate", current_run_signature)


# Compatibility wrapper for older imports.
def render_auto_opt_recommendations(run_result: dict, simple_cfg: dict | None = None) -> None:
    render_auto_opt_improvement(run_result)
