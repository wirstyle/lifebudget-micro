from __future__ import annotations

"""Step 5 preset-improvement assistant.

Scope of this first restored suggestion button:
- test only high-level strategy preset alternatives;
- do not change Step 4 universe, asset list, or universe size;
- do not run local auto-optimisation;
- apply changes through queued session-state updates to avoid Streamlit widget-key
  mutation errors.
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
    SEMANTIC_LAST_SIGNATURE,
    SEMANTIC_SLIDER_KEYS,
    SEMANTIC_TOUCHED_FLAG,
    allowed_style_presets_for_philosophy,
    allowed_strategy_templates_for_philosophy,
    recommended_strategy_combo_for_philosophy,
    resolve_semantic_slider_defaults,
    semantic_seed_signature,
    strategy_combo_status,
)
from ui.step5.governance import resolve_governance_status

try:
    from ui.state.updates import queue_and_rerun
except Exception:  # pragma: no cover
    queue_and_rerun = None


PRESET_SUGGESTION_STATE_KEY = "step5_preset_suggestion_v2"
PRESET_SUGGESTION_SCOPE_KEY = "step5_preset_suggestion_scope_v2"
PRESET_APPLIED_SIGNATURE_KEY = "step5_preset_applied_run_signature_v2"
PRESET_APPLIED_LABEL_KEY = "step5_preset_applied_label_v2"
PRESET_SKIPPED_SCOPE_KEY = "step5_preset_skipped_scope_v1"
PRESET_SKIPPED_LABEL_KEY = "step5_preset_skipped_label_v1"
PRESET_SUGGESTION_TIMING_KEY = "step5_preset_suggestion_timing_v1"
STEP5_SCROLL_TO_RESULT_AFTER_APPLY_KEY = "step5_scroll_to_real_run_result_after_apply_v1"
STEP5_DEMO_SPEED_MODE_KEY = "step5_demo_speed_mode_v1"


# Keep these as string constants instead of importing auto_opt_recommendations here.
# Importing the second-phase module from the first-phase module is unnecessary and
# would make the suggestion phases more tightly coupled. These keys match the
# public state keys defined by ui.step5.auto_opt_recommendations.
AUTO_OPT_STATE_KEYS_TO_CLEAR = (
    "step5_auto_opt_suggestion_v1",
    "step5_auto_opt_suggestion_scope_v1",
    "step5_auto_opt_suggestion_timing_v1",
    "step5_auto_opt_applied_run_signature_v1",
    "step5_auto_opt_applied_label_v1",
)


def _demo_speed_mode_enabled() -> bool:
    """Use smaller candidate budgets by default in hosted/demo mode."""
    if STEP5_DEMO_SPEED_MODE_KEY not in st.session_state:
        st.session_state[STEP5_DEMO_SPEED_MODE_KEY] = True
    return bool(st.session_state.get(STEP5_DEMO_SPEED_MODE_KEY, True))


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
        return float(value)
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


# ---------------------------------------------------------------------------
# Match the current Step 5 resolver without importing step5_workspace.
# Importing step5_workspace here would create a circular dependency because
# workspace -> post_run -> preset_recommendations.
# ---------------------------------------------------------------------------


def _resolve_overlay_label(simple_cfg: dict, advanced_cfg: dict) -> str:
    overlay_intensity = float(simple_cfg.get("overlay_intensity", 0.5) or 0.5)
    simplicity = float(simple_cfg.get("simplicity_vs_sophistication", 0.5) or 0.5)
    if overlay_intensity < 0.35:
        return "light"
    if overlay_intensity > 0.70 and simplicity < 0.45:
        return "adaptive"
    return "historical"


def _resolve_universe_bucket() -> str:
    try:
        universe_size = int(st.session_state.get("universe_size", 25) or 25)
    except Exception:
        universe_size = 25
    if universe_size <= 12:
        return "small"
    if universe_size <= 25:
        return "medium"
    if universe_size <= 75:
        return "large"
    return "research"


def _resolve_top_k(simple_cfg: dict) -> int:
    concentration = float(simple_cfg.get("diversification_vs_concentration", 0.55) or 0.55)
    try:
        universe_size = int(st.session_state.get("universe_size", 25) or 25)
    except Exception:
        universe_size = 25
    if concentration >= 0.75:
        return max(4, min(8, universe_size))
    if concentration >= 0.60:
        return max(6, min(10, universe_size))
    if concentration >= 0.45:
        return max(8, min(12, universe_size))
    return max(10, min(16, universe_size))


def _build_resolved_cfg(simple_cfg: dict, advanced_cfg: dict) -> dict:
    resolved = dict(simple_cfg or {})
    resolved.update(dict(advanced_cfg or {}))
    resolved["top_k"] = _resolve_top_k(simple_cfg)
    resolved["overlay_label"] = _resolve_overlay_label(simple_cfg, advanced_cfg)
    resolved["universe_bucket"] = _resolve_universe_bucket()
    resolved["signal_mode"] = "mu_sigma" if float(simple_cfg.get("confidence_in_signal", 0.5) or 0.5) >= 0.50 else "broad_beta"
    resolved["selection_policy"] = "fixed_composite_score"
    return resolved


def _resolve_candidate_cfg_final(simple_cfg: dict) -> tuple[dict, dict]:
    resolved_cfg = _build_resolved_cfg(simple_cfg, {})
    try:
        gov = resolve_governance_status(simple_cfg, {}, resolved_cfg)
    except Exception as exc:
        gov = {"state": "coherent", "cfg_final": dict(resolved_cfg), "warning": str(exc)}

    cfg_final = dict(_coerce_mapping(gov).get("cfg_final", {}) or {})
    if not cfg_final:
        governed_result = _coerce_mapping(gov).get("governed_result")
        final_obj = getattr(governed_result, "final_cfg", None)
        if final_obj is not None:
            try:
                cfg_final = _coerce_mapping(final_obj)
            except Exception:
                cfg_final = dict(getattr(final_obj, "__dict__", {}) or {})
    if not cfg_final:
        cfg_final = dict(resolved_cfg)
    return _coerce_cfg_payload(cfg_final), _coerce_mapping(gov)


# ---------------------------------------------------------------------------
# Candidate construction and scoring
# ---------------------------------------------------------------------------


def _current_simple_state() -> dict:
    philosophy = str(st.session_state.get("investment_philosophy", "Balanced") or "Balanced")
    rec_template, rec_style = recommended_strategy_combo_for_philosophy(philosophy)
    template = str(st.session_state.get("step5_template", rec_template) or rec_template)
    style = str(st.session_state.get("step5_style", rec_style) or rec_style)

    def _slider_value(logical_key: str, fallback: float) -> float:
        widget_key = SEMANTIC_SLIDER_KEYS.get(logical_key, "")
        return _safe_float(st.session_state.get(widget_key, fallback), fallback)

    return _build_simple_cfg(
        philosophy=philosophy,
        template=template,
        style=style,
        slider_values={
            "risk_appetite": _slider_value("risk_appetite", 0.50),
            "drawdown_protection": _slider_value("drawdown_protection", 0.62),
            "diversification_vs_concentration": _slider_value("diversification_vs_concentration", 0.55),
            "overlay_intensity": _slider_value("overlay_intensity", 0.42),
            "stability_vs_responsiveness": _slider_value("stability_vs_responsiveness", 0.60),
            "confidence_in_signal": _slider_value("confidence_in_signal", 0.56),
            "low_turnover_vs_adaptive": _slider_value("low_turnover_vs_adaptive", 0.40),
            "simplicity_vs_sophistication": _slider_value("simplicity_vs_sophistication", 0.56),
        },
    )


def _build_simple_cfg(*, philosophy: str, template: str, style: str, slider_values: dict[str, float]) -> dict:
    risk = _safe_float(slider_values.get("risk_appetite", 0.50), 0.50)
    drawdown = _safe_float(slider_values.get("drawdown_protection", 0.62), 0.62)
    divers = _safe_float(slider_values.get("diversification_vs_concentration", 0.55), 0.55)
    risk_penalty = max(0.0, min(1.0, (1.0 - risk) * 0.40 + drawdown * 0.60))
    concentration = 5 + int(round(divers * 20))
    rec_template, rec_style = recommended_strategy_combo_for_philosophy(philosophy)
    return {
        "preset": style,
        "template": template,
        "risk_appetite": risk,
        "drawdown_protection": drawdown,
        "diversification_vs_concentration": divers,
        "overlay_intensity": _safe_float(slider_values.get("overlay_intensity", 0.42), 0.42),
        "stability_vs_responsiveness": _safe_float(slider_values.get("stability_vs_responsiveness", 0.60), 0.60),
        "confidence_in_signal": _safe_float(slider_values.get("confidence_in_signal", 0.56), 0.56),
        "low_turnover_vs_adaptive": _safe_float(slider_values.get("low_turnover_vs_adaptive", 0.40), 0.40),
        "simplicity_vs_sophistication": _safe_float(slider_values.get("simplicity_vs_sophistication", 0.56), 0.56),
        "risk_penalty": risk_penalty,
        "concentration": concentration,
        "combo_status": strategy_combo_status(philosophy, template, style),
        "recommended_template": rec_template,
        "recommended_style": rec_style,
        "philosophy": philosophy,
    }


def _allowed_combos(philosophy: str) -> list[tuple[str, str]]:
    combos: list[tuple[str, str]] = []
    for template in allowed_strategy_templates_for_philosophy(philosophy):
        for style in allowed_style_presets_for_philosophy(philosophy, template):
            combo = (str(template), str(style))
            if combo not in combos:
                combos.append(combo)
    return combos


def _candidate_combos(philosophy: str, current_template: str, current_style: str, *, max_candidates: int = 2) -> list[tuple[str, str, str]]:
    current = (str(current_template), str(current_style))
    rec_template, rec_style = recommended_strategy_combo_for_philosophy(philosophy)
    allowed = _allowed_combos(philosophy)

    ordered: list[tuple[str, str, str]] = []

    # First, if the current setup has drifted from the philosophy default, test the
    # canonical bundle as the safest nearby restore path.
    recommended = (rec_template, rec_style)
    if recommended != current and recommended in allowed:
        ordered.append((rec_template, rec_style, "recommended_restore"))

    # Then test nearby styles under the same template. This is the cleanest preset
    # improvement because it changes posture without changing the engine family.
    for template, style in allowed:
        if (template, style) == current:
            continue
        if template == current[0] and all((template, style) != (x[0], x[1]) for x in ordered):
            ordered.append((template, style, "nearby_style"))

    # Finally allow one broader but still philosophy-approved alternative.
    for template, style in allowed:
        if (template, style) == current:
            continue
        if all((template, style) != (x[0], x[1]) for x in ordered):
            ordered.append((template, style, "broader_allowed"))

    return ordered[: max(1, int(max_candidates))]


def _score_perf(perf: dict, philosophy: str) -> float:
    p = _normalise_perf(perf)
    profile = str(philosophy or "Balanced").strip().lower()
    sharpe = p["sharpe"]
    cagr = p["cagr"]
    vol = abs(p["annual_volatility"])
    maxdd = abs(p["max_drawdown"])
    if profile == "growth":
        return float((0.50 * sharpe) + (1.10 * cagr) - (0.25 * maxdd) - (0.10 * vol))
    if profile == "defensive":
        return float((0.65 * sharpe) + (0.25 * cagr) - (0.95 * maxdd) - (0.45 * vol))
    return float((0.55 * sharpe) + (0.55 * cagr) - (0.65 * maxdd) - (0.25 * vol))


def _acceptance_gate(candidate_perf: dict, current_perf: dict, philosophy: str) -> tuple[bool, str]:
    c = _normalise_perf(candidate_perf)
    b = _normalise_perf(current_perf)
    profile = str(philosophy or "Balanced").strip().lower()

    delta_score = _score_perf(c, philosophy) - _score_perf(b, philosophy)
    delta_sharpe = c["sharpe"] - b["sharpe"]
    delta_cagr = c["cagr"] - b["cagr"]
    delta_vol = c["annual_volatility"] - b["annual_volatility"]
    delta_dd = c["max_drawdown"] - b["max_drawdown"]  # positive means worse drawdown

    if profile == "defensive":
        passed = (
            delta_score > 0.005
            and delta_dd <= 0.005
            and delta_vol <= 0.010
            and delta_cagr >= -0.010
        ) or (delta_dd <= -0.020 and delta_sharpe >= -0.03)
        reason = "Defensive gate prioritises lower drawdown/volatility and avoids sacrificing too much CAGR."
    elif profile == "growth":
        passed = (
            delta_score > 0.005
            and delta_cagr >= -0.005
            and delta_dd <= 0.050
        ) or (delta_cagr >= 0.010 and delta_sharpe >= -0.03)
        reason = "Growth gate allows some extra drawdown only if return/risk-adjusted performance improves."
    else:
        passed = (
            delta_score > 0.005
            and delta_cagr >= -0.010
            and delta_dd <= 0.020
        ) or (delta_sharpe >= 0.050 and delta_dd <= 0.020)
        reason = "Balanced gate looks for a better trade-off without allowing drawdown to drift too far."

    return bool(passed), reason


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
    panel = st.session_state.get("asset_panel_df")
    assets: list[str] = []
    rows = 0
    if isinstance(panel, pd.DataFrame) and not panel.empty:
        rows = int(len(panel))
        if "asset" in panel.columns:
            assets = sorted(panel["asset"].dropna().astype(str).unique().tolist())
    simple = _current_simple_state()
    return _hash_scope({
        "run_signature": str(_coerce_mapping(run_result).get("run_signature", "") or ""),
        "philosophy": simple.get("philosophy"),
        "template": simple.get("template"),
        "preset": simple.get("preset"),
        "assets": assets,
        "rows": rows,
    })


def _run_preset_search(run_result: dict, *, max_candidates: int = 2) -> dict:
    run_map = _coerce_mapping(run_result)
    current_perf = _normalise_perf(run_map.get("performance_summary", {}))
    simple_current = _current_simple_state()
    philosophy = str(simple_current.get("philosophy", "Balanced") or "Balanced")
    current_template = str(simple_current.get("template", "Balanced Risk-Controlled") or "Balanced Risk-Controlled")
    current_style = str(simple_current.get("preset", "Balanced") or "Balanced")

    panel = st.session_state.get("asset_panel_df")
    panel_df = panel.copy() if isinstance(panel, pd.DataFrame) else pd.DataFrame()
    if panel_df.empty:
        return {"scope": _build_scope(run_result), "evaluations": [], "error": "Step 4 asset panel is missing."}

    combos = _candidate_combos(philosophy, current_template, current_style, max_candidates=max_candidates)
    evaluations: list[dict] = []
    started = time.perf_counter()

    for template, style, family in combos:
        candidate_started = time.perf_counter()
        defaults = resolve_semantic_slider_defaults(template, style)
        simple_candidate = _build_simple_cfg(
            philosophy=philosophy,
            template=template,
            style=style,
            slider_values=defaults,
        )
        cfg_payload, gov = _resolve_candidate_cfg_final(simple_candidate)
        run = _run_candidate(cfg_payload, panel_df)
        candidate_elapsed_sec = float(time.perf_counter() - candidate_started)
        perf = _normalise_perf(run.get("performance_summary", {}))
        accepted = False
        gate_reason = "Candidate could not be evaluated."
        if perf and not run.get("error"):
            accepted, gate_reason = _acceptance_gate(perf, current_perf, philosophy)

        evaluations.append({
            "label": f"{template} + {style}",
            "family": family,
            "strategy_template": template,
            "style_preset": style,
            "slider_defaults": dict(defaults),
            "cfg_payload": dict(cfg_payload),
            "performance_summary": dict(perf),
            # Keep the real candidate engine output so Apply can promote the already
            # rerun-tested result instead of forcing the user to run the same candidate
            # manually again. This is safe because the candidate was evaluated against
            # the same Step 4 panel and the exact cfg_payload below.
            "raw_result": _coerce_mapping(run.get("raw", {})),
            "score": _score_perf(perf, philosophy) if perf else None,
            "score_delta": (_score_perf(perf, philosophy) - _score_perf(current_perf, philosophy)) if perf else None,
            "accepted": bool(accepted),
            "gate_reason": gate_reason,
            "governance_state": str(_coerce_mapping(gov).get("state", "coherent") or "coherent"),
            "error": str(run.get("error", "") or ""),
            "elapsed_sec": float(candidate_elapsed_sec),
            "apply_patch": _build_apply_patch(template, style, defaults, cfg_payload),
        })

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
        "current_template": current_template,
        "current_style": current_style,
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


def _clear_auto_opt_state_patch() -> dict:
    """Clear second-phase cached results when the preset phase changes state.

    Preset changes alter the strategic setup. Any technical-tuning candidates
    computed before that decision are no longer the right comparison target.
    """
    patch: dict[str, Any] = {}
    for key in AUTO_OPT_STATE_KEYS_TO_CLEAR:
        if key.endswith("suggestion_v1") or key.endswith("timing_v1"):
            patch[key] = {}
        else:
            patch[key] = ""
    return patch


def _skip_current_preset_candidate(scope: str, label: str = "current preset") -> None:
    patch = {
        PRESET_SKIPPED_SCOPE_KEY: str(scope or ""),
        PRESET_SKIPPED_LABEL_KEY: str(label or "current preset"),
    }
    patch.update(_clear_auto_opt_state_patch())

    if callable(queue_and_rerun):
        queue_and_rerun(patch)
        return
    for key, value in patch.items():
        st.session_state[key] = value
    st.rerun()


def _build_widget_patch_from_cfg_payload(cfg_payload: Any) -> dict:
    payload = _coerce_mapping(cfg_payload)
    field_to_widget = {
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
    return {widget_key: payload[field_name] for field_name, widget_key in field_to_widget.items() if field_name in payload}


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


def _normalise_promoted_candidate_result(candidate: dict, cfg_payload: dict) -> dict:
    """Build a Step-5-compatible run_result from a tested preset candidate.

    The preset test has already called the real investment engine for this
    candidate. Applying the preset can therefore promote that tested result to
    the current run, avoiding a duplicated manual rerun.
    """
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
            "promoted_from_preset_candidate": True,
            "preset_candidate_label": str(candidate_map.get("label", "Preset candidate") or "Preset candidate"),
            "asset_panel_source_label": str(st.session_state.get("asset_panel_source_label", "Step 4 asset panel") or "Step 4 asset panel"),
            "asset_panel_n_rows": int(len(panel_df)) if isinstance(panel_df, pd.DataFrame) else 0,
            "asset_panel_n_assets": int(panel_df["asset"].nunique()) if isinstance(panel_df, pd.DataFrame) and "asset" in panel_df.columns else 0,
            "performance_summary": dict(perf),
            "config": dict(cfg_payload),
            "governance_state": str(candidate_map.get("governance_state", "coherent") or "coherent"),
            "run_signature": run_signature,
            "config_fingerprint": config_fingerprint,
            "run_timestamp": run_timestamp,
            "panel_shape": (int(len(panel_df)), int(len(panel_df.columns))) if isinstance(panel_df, pd.DataFrame) else (0, 0),
            "evaluation_period_label": "Walk-forward evaluated period",
            "search_eval_split": {"enabled": False, "reason": "preset_candidate_promoted_after_rerun_test"},
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
            "promoted_from_preset_candidate": True,
            "breakdown_df": pd.DataFrame(),
        },
    )
    return promoted


def _build_apply_patch(template: str, style: str, defaults: dict, cfg_payload: dict) -> dict:
    patch = {
        "step5_template": str(template),
        "step5_style": str(style),
        SEMANTIC_TOUCHED_FLAG: False,
        SEMANTIC_LAST_SIGNATURE: semantic_seed_signature(template, style),
    }
    for logical_key, widget_key in SEMANTIC_SLIDER_KEYS.items():
        if logical_key in defaults:
            patch[widget_key] = float(defaults.get(logical_key, 0.5))
    patch.update(_build_widget_patch_from_cfg_payload(cfg_payload))
    return patch


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
                PRESET_SUGGESTION_STATE_KEY: {},
                PRESET_SUGGESTION_SCOPE_KEY: "",
                PRESET_SKIPPED_SCOPE_KEY: "",
                PRESET_SKIPPED_LABEL_KEY: "",
                **_clear_auto_opt_state_patch(),
                PRESET_APPLIED_SIGNATURE_KEY: run_signature,
                PRESET_APPLIED_LABEL_KEY: str(candidate_map.get("label", "Preset candidate") or "Preset candidate"),
                "step5_preset_apply_message_v2": (
                    f"Preset suggestion applied using the rerun-tested candidate result: "
                    f"{candidate_map.get('label', 'Preset candidate')}."
                ),
                STEP5_SCROLL_TO_RESULT_AFTER_APPLY_KEY: True,
            }
        )
    else:
        # Safety fallback: if the candidate result is unavailable, keep the previous
        # manual rerun behaviour rather than pretending the result is current.
        patch.update(
            {
                "engine_has_run": False,
                "step5_last_run_result": None,
                "step5_run_result": None,
                PRESET_APPLIED_SIGNATURE_KEY: "",
                PRESET_APPLIED_LABEL_KEY: "",
                PRESET_SKIPPED_SCOPE_KEY: "",
                PRESET_SKIPPED_LABEL_KEY: "",
                **_clear_auto_opt_state_patch(),
                "step5_preset_apply_message_v2": (
                    f"Preset suggestion applied: {candidate_map.get('label', 'Preset candidate')}. "
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
        # Positive means the drawdown became less severe; negative means it worsened.
        maxdd_improvement = base.get("max_drawdown", 0.0) - perf.get("max_drawdown", 0.0)
        rows.append({
            "candidate": str(item.get("label", "Candidate") or "Candidate"),
            "status": status,
            "family": str(item.get("family", "") or ""),
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
        })
    return pd.DataFrame(rows)


def _candidate_context_caption(item: dict) -> str:
    family = str(item.get("family", "") or "")
    if family == "recommended_restore":
        return "Restores the philosophy-recommended strategy bundle."
    if family == "nearby_style":
        return "Tests a nearby style under the current strategy template."
    return "Tests another philosophy-approved preset combination."


def _render_recommended_candidate(candidate: dict, current_perf: dict) -> None:
    """Render only the best accepted preset as the actionable recommendation."""
    item = _coerce_mapping(candidate)
    perf = _normalise_perf(item.get("performance_summary", {}))
    current = _normalise_perf(current_perf)

    st.markdown("### Recommended preset improvement")
    st.markdown(f"**{item.get('label', 'Preset candidate')}**")
    st.caption(_candidate_context_caption(item))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("CAGR", _format_pct(perf["cagr"]), delta=_format_delta_pct(perf["cagr"] - current["cagr"]))
    with c2:
        # Lower volatility is better, so use inverse colour semantics.
        st.metric(
            "Vol",
            _format_pct(perf["annual_volatility"]),
            delta=_format_delta_pct(perf["annual_volatility"] - current["annual_volatility"]),
            delta_color="inverse",
        )
    with c3:
        # MaxDD is stored as absolute drawdown. Positive display delta means drawdown improved.
        dd_improvement = current["max_drawdown"] - perf["max_drawdown"]
        st.metric("MaxDD", f"-{100.0 * perf['max_drawdown']:.2f}%", delta=_format_delta_pct(dd_improvement))
    with c4:
        st.metric("Sharpe", f"{perf['sharpe']:.2f}", delta=f"{perf['sharpe'] - current['sharpe']:+.2f}")

    st.success("This candidate passed the preset acceptance gate.")
    gate_reason = str(item.get("gate_reason", "") or "")
    if gate_reason:
        st.caption(gate_reason)
    if item.get("error"):
        st.warning(str(item.get("error")))

    with st.expander("Recommended setup details", expanded=False):
        st.caption(
            f"template={item.get('strategy_template', '—')} · style={item.get('style_preset', '—')} · "
            f"governance={item.get('governance_state', '—')} · score_delta={_safe_float(item.get('score_delta'), 0.0):+.3f}"
        )


def render_preset_improvement(run_result: dict) -> dict:
    """Render the first restored Step 5 preset-improvement assistant.

    Returns a small flow-state payload so post_run.py can decide whether the
    second phase (technical engine tuning) should run now or wait.
    """
    flow_state = {
        "blocks_auto_opt": False,
        "status": "not_available",
        "scope": "",
        "has_recommendation": False,
    }

    run_map = _coerce_mapping(run_result)
    perf = _normalise_perf(run_map.get("performance_summary", {}))
    if not perf:
        return flow_state

    st.markdown("### Preset suggestion")
    st.caption(
        "Optional phase 1: tests nearby strategy presets after the real engine run. "
        "This does not change the Step 4 universe, assets, size, or market-data panel."
    )

    # After applying an accepted preset we keep this section intentionally quiet:
    # one green confirmation is enough. Showing both the transient apply message
    # and the persistent "already applied" state felt visually redundant.
    msg = st.session_state.pop("step5_preset_apply_message_v2", "")

    current_run_signature = str(run_map.get("run_signature", "") or "")
    applied_signature = str(st.session_state.get(PRESET_APPLIED_SIGNATURE_KEY, "") or "")
    applied_label = str(st.session_state.get(PRESET_APPLIED_LABEL_KEY, "") or "")
    if applied_signature and current_run_signature and applied_signature == current_run_signature:
        label = applied_label or "the accepted preset suggestion"
        st.success(
            f"Preset improvement applied: {label}. The rerun-tested candidate is now the current Step 5 result."
        )
        st.caption(
            "Preset testing is hidden for this run to avoid suggesting the same loop again. "
            "Engine tuning can now test small technical variations on top of this accepted strategy setup."
        )
        flow_state.update({"status": "applied", "has_recommendation": False})
        return flow_state

    if msg:
        st.success(str(msg))

    scope = _build_scope(run_map)
    flow_state["scope"] = str(scope)
    saved_scope = str(st.session_state.get(PRESET_SUGGESTION_SCOPE_KEY, "") or "")
    payload = _coerce_mapping(st.session_state.get(PRESET_SUGGESTION_STATE_KEY, {}))
    evaluations = list(payload.get("evaluations", []) or []) if saved_scope == scope else []

    skipped_scope = str(st.session_state.get(PRESET_SKIPPED_SCOPE_KEY, "") or "")
    skipped_label = str(st.session_state.get(PRESET_SKIPPED_LABEL_KEY, "") or "")
    if skipped_scope and skipped_scope == scope and not evaluations:
        st.info("Preset check skipped for this run. Engine tuning can now be reviewed or skipped.")
        if skipped_label:
            st.caption(f"Skipped preset check: {skipped_label}.")
        flow_state.update({"status": "skipped", "has_recommendation": False, "blocks_auto_opt": False})
        return flow_state

    # Product flow: after the user runs the real engine, do not ask them to run a
    # second diagnostic button. Test the nearby preset candidates once for the
    # current run scope, cache them in session_state, and then show the Apply
    # choice directly.
    if saved_scope != scope or not evaluations:
        quick_mode = _demo_speed_mode_enabled()
        max_candidates = 1 if quick_mode else 2
        estimate = "~30s" if quick_mode else "~60s+"
        st.info(
            f"Preset suggestion is optional and reruns {max_candidates} candidate"
            f"{'s' if max_candidates != 1 else ''} with the real engine. Estimated time: {estimate}."
        )
        left, right = st.columns(2)
        should_run = False
        with left:
            should_run = st.button(
                "Run quick preset check" if quick_mode else "Run full preset check",
                key="step5_run_preset_suggestion_check_v1",
                use_container_width=True,
            )
        with right:
            if st.button("Skip preset check and continue", key="step5_skip_preset_check_not_run_v1", use_container_width=True):
                _skip_current_preset_candidate(scope, "preset check skipped")
        if not should_run:
            flow_state.update({"status": "waiting_for_user", "has_recommendation": False, "blocks_auto_opt": True})
            return flow_state

        with st.spinner("Testing nearby preset alternatives with the real engine..."):
            payload = _run_preset_search(run_map, max_candidates=max_candidates)
        st.session_state[PRESET_SUGGESTION_STATE_KEY] = payload
        st.session_state[PRESET_SUGGESTION_SCOPE_KEY] = str(payload.get("scope", scope))
        st.session_state[PRESET_SUGGESTION_TIMING_KEY] = _coerce_mapping(payload.get("timing_summary", {}))
        evaluations = list(payload.get("evaluations", []) or [])

    if not evaluations:
        st.info("No nearby preset alternatives were available to test for this run.")
        flow_state.update({"status": "no_candidates"})
        return flow_state

    accepted_items = [dict(x) for x in evaluations if bool(_coerce_mapping(x).get("accepted", False))]
    table = _candidate_table(evaluations, perf)
    skipped_scope = str(st.session_state.get(PRESET_SKIPPED_SCOPE_KEY, "") or "")
    skipped_label = str(st.session_state.get(PRESET_SKIPPED_LABEL_KEY, "") or "")
    preset_was_skipped = bool(skipped_scope and skipped_scope == scope)

    if not accepted_items:
        st.success("Current preset loop has converged: no tested preset materially improved this run.")
        st.caption(
            "The tested alternatives are kept below for transparency, but none is offered as an action because "
            "the acceptance gate did not find a better trade-off. Engine tuning can continue on the current preset."
        )
        flow_state.update({"status": "converged", "has_recommendation": False})
    elif preset_was_skipped:
        st.info(
            "Current preset kept for this run. Engine tuning can now test small technical variations on the existing strategy setup."
        )
        if skipped_label:
            st.caption(f"Skipped preset recommendation: {skipped_label}.")
        flow_state.update({"status": "skipped", "has_recommendation": True, "blocks_auto_opt": False})
    else:
        best_candidate = accepted_items[0]
        flow_state.update({"status": "pending_action", "has_recommendation": True, "blocks_auto_opt": True})
        st.success("Recommended preset improvement found. The best accepted candidate is shown below.")
        _render_recommended_candidate(best_candidate, perf)
        st.caption(
            "The table below shows every preset candidate tested by the engine. Only the best accepted candidate "
            "is offered as the main action; rejected candidates are shown for transparency, not as recommendations."
        )

    with st.expander("Preset test diagnostics", expanded=False):
        st.caption(
            f"tested_candidates={len(evaluations)} · elapsed={_safe_float(payload.get('elapsed_sec', 0.0), 0.0):.2f}s · "
            f"scope={str(payload.get('scope', scope))}"
        )
        if not table.empty:
            st.dataframe(table, use_container_width=True, hide_index=True)

    if accepted_items and not preset_was_skipped:
        best_candidate = accepted_items[0]
        left, right = st.columns(2)
        with left:
            if st.button("Apply recommended preset", key="step5_apply_best_preset_candidate_v3", use_container_width=True):
                _apply_candidate(best_candidate)
        with right:
            if st.button("Keep current preset and continue", key="step5_skip_best_preset_candidate_v1", use_container_width=True):
                _skip_current_preset_candidate(scope, str(best_candidate.get("label", "recommended preset") or "recommended preset"))

    return flow_state


# Compatibility names used by older Step 5 imports. Keep these harmless until the
# second and third suggestion buttons are rebuilt deliberately.

def render_preset_recommendations(run_result: dict, simple_cfg: dict | None = None) -> None:
    render_preset_improvement(run_result)


def render_auto_opt_recommendations(simple_cfg: dict | None = None) -> None:
    st.info("Technical auto-tune suggestions are intentionally disabled in this preset-only rebuild.")


def render_size_recommendations() -> None:
    st.info("Universe/size suggestions are intentionally disabled until the preset loop is stable.")
