from __future__ import annotations

import hashlib
import json
import time
from dataclasses import fields
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st

from src.investment import MicroPipelineConfig, config_to_dict, run_micro_investment_pipeline


STEP5_TIMING_CACHE_KEY = "step5_timing_cache"
STEP5_LAST_TIMINGS_KEY = "step5_last_timings"

# Legacy Step 5 state keys from the removed exploration layer.
# Keeping one list avoids duplicated cleanup loops across workspace/post-run.
RETIRED_STEP5_STATE_KEYS = (
    "step5_show_improvements",
    "step5_request_improvements",
    "step5_improvements_requested_signature",
    "step5_continue_improvements_after_apply",
    "step5_resume_improvement_step",
    "step5_cumulative_wizard_active",
    "step5_request_universe_recommendation",
    "step5_universe_recommendation_requested_signature",
    "step5_preset_recommendations",
    "step5_preset_evaluations",
    "step5_best_preset_eval",
    "step5_candidate_table",
    "step5_candidate_frontier_df",
    "step5_universe_recommendation_result",
    "step5_tested_candidate_evaluations",
    "step5_best_candidate_eval",
    "step5_selected_candidate_eval",
    "step5_pending_apply_patch",
    "step5_pending_apply_cfg_final",
    "step5_pending_apply_simple_cfg",
    "step5_improve_wizard",
    "step5_auto_run_reason",
    "step5_auto_run_pending_patch_keys",
)


def _coerce_mapping(value: Any) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            out = to_dict()
            if isinstance(out, dict):
                return dict(out)
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


def _coerce_cfg(cfg_payload: dict) -> MicroPipelineConfig:
    valid = {f.name for f in fields(MicroPipelineConfig)}
    filtered = {k: v for k, v in dict(cfg_payload or {}).items() if k in valid}
    return MicroPipelineConfig(**filtered)


def _extract_feature_mu_selection_payload(raw_result: dict) -> dict:
    raw_map = _coerce_mapping(raw_result)
    payload = {
        "feature_mu_selected_cols_n": 0,
        "feature_mu_selected_cols": "",
        "feature_mu_selected_family_counts": "{}",
    }

    def _normalise_counts(value: Any) -> str:
        if isinstance(value, dict):
            try:
                return json.dumps(value, sort_keys=True)
            except Exception:
                return "{}"
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return json.dumps(parsed, sort_keys=True)
            except Exception:
                return "{}"
        return "{}"

    for key in payload:
        if key in raw_map:
            payload[key] = raw_map.get(key, payload[key])

    payload["feature_mu_selected_cols_n"] = _safe_int(payload.get("feature_mu_selected_cols_n", 0), 0)
    payload["feature_mu_selected_cols"] = str(payload.get("feature_mu_selected_cols", "") or "")
    payload["feature_mu_selected_family_counts"] = _normalise_counts(payload.get("feature_mu_selected_family_counts", "{}"))
    return payload


def _normalise_run_result(
    raw_result: dict,
    simple_cfg: dict,
    advanced_cfg: dict,
    cfg_final: dict,
    governance_status: dict,
    asset_panel_df: pd.DataFrame,
) -> dict:
    raw_result = _coerce_mapping(raw_result)
    perf = _coerce_mapping(raw_result.get("performance_summary", {}))
    feature_mu_payload = _extract_feature_mu_selection_payload(raw_result)

    out = dict(raw_result)
    out.update(
        {
            "source": "micro_pipeline_real",
            "asset_panel_source_label": str(st.session_state.get("asset_panel_source_label", "Step 4 asset panel") or "Step 4 asset panel"),
            "asset_panel_n_rows": int(len(asset_panel_df)) if isinstance(asset_panel_df, pd.DataFrame) else 0,
            "asset_panel_n_assets": int(asset_panel_df["asset"].nunique()) if isinstance(asset_panel_df, pd.DataFrame) and "asset" in asset_panel_df.columns else 0,
            "performance_summary": {
                "cagr": _safe_float(perf.get("cagr", 0.0), 0.0),
                "annual_volatility": _safe_float(perf.get("annual_volatility", perf.get("volatility", 0.0)), 0.0),
                "volatility": _safe_float(perf.get("volatility", perf.get("annual_volatility", 0.0)), 0.0),
                "sharpe": _safe_float(perf.get("sharpe", 0.0), 0.0),
                "max_drawdown": abs(_safe_float(perf.get("max_drawdown", 0.0), 0.0)),
                "periods": _safe_int(perf.get("periods", 0), 0),
            },
            "config": {**dict(simple_cfg or {}), **dict(advanced_cfg or {}), **dict(cfg_final or {})},
            "governance_state": str((governance_status or {}).get("state", "coherent") or "coherent"),
            "feature_mu_selected_cols_n": _safe_int(feature_mu_payload.get("feature_mu_selected_cols_n", 0), 0),
            "feature_mu_selected_cols": str(feature_mu_payload.get("feature_mu_selected_cols", "") or ""),
            "feature_mu_selected_family_counts": str(feature_mu_payload.get("feature_mu_selected_family_counts", "{}") or "{}"),
        }
    )
    return out


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


def _resolve_run_signature(cfg_final: dict, asset_panel_df: Any) -> str:
    payload = {
        "cfg_final": _coerce_mapping(cfg_final),
        "universe_size": st.session_state.get("universe_size"),
        "universe_strategy": st.session_state.get("universe_strategy"),
        "selected_assets": list(st.session_state.get("selected_assets", []) or []),
        "panel_fp": _stable_panel_fingerprint(asset_panel_df) if isinstance(asset_panel_df, pd.DataFrame) else "empty",
    }
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(encoded.encode("utf-8")).hexdigest()[:12]


def _resolve_config_fingerprint(cfg_final: dict) -> str:
    encoded = json.dumps(_coerce_mapping(cfg_final), sort_keys=True, default=str)
    return hashlib.md5(encoded.encode("utf-8")).hexdigest()[:12]


def _build_cache_key(asset_panel_df: pd.DataFrame, cfg_payload: dict) -> str:
    payload = {"panel_fp": _stable_panel_fingerprint(asset_panel_df), "cfg": _coerce_mapping(cfg_payload)}
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(encoded.encode("utf-8")).hexdigest()[:24]


def _run_pipeline_cached(asset_panel_df: pd.DataFrame, cfg_payload: dict) -> tuple[dict, float, bool]:
    cache = st.session_state.get(STEP5_TIMING_CACHE_KEY, {})
    cache = cache if isinstance(cache, dict) else {}
    cache_key = _build_cache_key(asset_panel_df, cfg_payload)
    cached = cache.get(cache_key)
    if isinstance(cached, dict) and "result" in cached:
        return _coerce_mapping(cached.get("result", {})), 0.0, True

    t0 = time.perf_counter()
    raw_result = run_micro_investment_pipeline(asset_panel_df, cfg=_coerce_cfg(cfg_payload))
    elapsed = time.perf_counter() - t0
    cache[cache_key] = {"result": _coerce_mapping(raw_result)}
    st.session_state[STEP5_TIMING_CACHE_KEY] = cache
    return _coerce_mapping(raw_result), float(elapsed), False


def _render_timing_summary(timing_summary: dict) -> None:
    if not timing_summary:
        return
    with st.expander("Step 5 timing", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Engine run", f"{_safe_float(timing_summary.get('base_run_sec', 0.0)):.2f}s")
        with c2:
            st.metric("Cache", "hit" if timing_summary.get("base_from_cache") else "miss")
        with c3:
            st.metric("Total Step 5", f"{_safe_float(timing_summary.get('total_step5_sec', 0.0)):.2f}s")


def clear_retired_step5_state() -> None:
    """Remove legacy Step 5 exploration state from session_state."""
    for key in RETIRED_STEP5_STATE_KEYS:
        st.session_state.pop(key, None)


def clear_retired_step5_improvement_state() -> None:
    """Backward-compatible alias for older imports."""
    clear_retired_step5_state()


def render_run_panel(
    simple_cfg: dict,
    advanced_cfg: dict,
    governance_status: dict | None = None,
    cfg_final: dict | None = None,
    *,
    compact: bool = False,
):
    """Gold Stable runner: one real engine run and one stored result.

    compact=True is used by the cleaned Step 5 workspace so the run button can live
    inside the first "Ready to run" card without repeating the full technical run block.
    """
    if not compact:
        st.markdown("## Run engine")
        st.caption("Runs the real micro pipeline and stores the result for metrics, projection, and insights.")

    clear_retired_step5_improvement_state()

    governance_status = dict(governance_status or {})
    cfg_final = dict(cfg_final or {})
    state = str(governance_status.get("state", "coherent") or "coherent")
    disabled = state == "blocked"

    asset_panel_df = st.session_state.get("asset_panel_df")
    asset_panel_df = asset_panel_df.copy() if isinstance(asset_panel_df, pd.DataFrame) else pd.DataFrame()
    disabled = bool(disabled or asset_panel_df.empty)

    current_signature = _resolve_run_signature(cfg_final, asset_panel_df)
    current_config_fingerprint = _resolve_config_fingerprint(cfg_final)
    st.session_state["step5_current_run_signature"] = current_signature
    st.session_state["step5_current_config_fingerprint"] = current_config_fingerprint

    last_signature = str(st.session_state.get("step5_last_run_signature", "") or "")
    last_config_fingerprint = str(st.session_state.get("step5_last_config_fingerprint", "") or "")
    last_timestamp = str(st.session_state.get("step5_last_run_timestamp", "") or "")

    if not compact:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Panel rows", f"{int(len(asset_panel_df)):,}")
        with c2:
            st.metric("Panel assets", int(asset_panel_df["asset"].nunique()) if "asset" in asset_panel_df.columns and not asset_panel_df.empty else 0)
        with c3:
            st.metric("Engine status", "Ready" if not disabled and not asset_panel_df.empty else "Blocked")

    if last_signature and last_config_fingerprint and (
        last_signature != current_signature or last_config_fingerprint != current_config_fingerprint
    ):
        st.info("The current Step 5 inputs differ from the last executed run. Press 'Run portfolio & view results' to refresh the metrics.")

    if disabled:
        st.error("Execution is blocked until the governance issues are resolved.")
    elif not compact:
        st.info("Using the Step 4 asset panel for real execution.")

    if not compact:
        with st.expander("Run reproducibility details", expanded=False):
            st.caption(f"run_signature={current_signature} · config_fp={current_config_fingerprint}")
            if last_timestamp:
                st.caption(f"Last completed run → {last_timestamp}")

    manual_run_clicked = st.button(
        "Run portfolio & view results",
        key="step5_run_portfolio_view_results",
        use_container_width=True,
        disabled=disabled,
    )

    if not manual_run_clicked:
        return None

    step5_t0 = time.perf_counter()

    if asset_panel_df.empty:
        st.error("Step 4 asset panel is missing or empty. Load a valid panel before running the engine.")
        return None

    try:
        cfg_obj = _coerce_cfg(cfg_final)
        cfg_payload = dict(config_to_dict(cfg_obj))
        run_timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

        st.session_state["last_engine_config"] = cfg_payload
        st.session_state["step5_last_run_signature"] = current_signature
        st.session_state["step5_last_config_fingerprint"] = current_config_fingerprint
        st.session_state["step5_last_run_timestamp"] = run_timestamp

        with st.spinner("Running real micro investment pipeline..."):
            raw_result, base_run_sec, base_from_cache = _run_pipeline_cached(asset_panel_df, cfg_payload)

        run_result = _normalise_run_result(
            raw_result,
            dict(simple_cfg or {}),
            dict(advanced_cfg or {}),
            cfg_final,
            governance_status,
            asset_panel_df,
        )
        run_result["run_signature"] = current_signature
        run_result["config_fingerprint"] = current_config_fingerprint
        run_result["run_timestamp"] = run_timestamp
        run_result["panel_shape"] = (int(len(asset_panel_df)), int(len(asset_panel_df.columns)))
        run_result["evaluation_period_label"] = "Full period"
        run_result["search_eval_split"] = {"enabled": False, "reason": "gold_stable_minimal_runner"}

        elapsed_total = float(time.perf_counter() - step5_t0)
        timing_summary = {
            "base_run_sec": float(base_run_sec),
            "display_base_run_sec": float(base_run_sec),
            "base_from_cache": bool(base_from_cache),
            "total_step5_sec": elapsed_total,
            "display_total_step5_sec": elapsed_total,
            "cache_hits": int(bool(base_from_cache)),
            "engine_runs": 1,
            "breakdown_df": pd.DataFrame(),
        }
        st.session_state[STEP5_LAST_TIMINGS_KEY] = timing_summary
        run_result["timing_summary"] = timing_summary

        st.session_state["step5_last_run_result"] = run_result
        st.session_state["step5_run_result"] = run_result
        st.session_state["engine_has_run"] = True
        st.session_state["step5_force_run_once"] = False
        st.session_state["step5_auto_run_reason"] = ""
        st.session_state["step5_auto_run_pending_patch_keys"] = []
        clear_retired_step5_improvement_state()

        st.success("Run completed and stored as the current real result.")
        if not compact:
            _render_timing_summary(timing_summary)
        return run_result

    except Exception as exc:
        st.session_state["engine_has_run"] = False
        st.error(f"Engine run failed: {exc}")
        return None
