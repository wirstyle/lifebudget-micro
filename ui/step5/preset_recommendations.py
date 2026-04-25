from __future__ import annotations

from typing import Any
import hashlib
import json
import time
from dataclasses import fields
import pandas as pd
import streamlit as st

from src.investment import MicroPipelineConfig, run_micro_investment_pipeline
from src.evaluation import run_simple_auto_optimize
from ui.services.step4_universe_service import build_strategy_candidate_pool, recommended_universe_sizes_for_philosophy, resolve_step4_asset_panel
from ui.state.keys import ASSET_SOURCE_MODE, ASSET_START_DATE, ASSET_END_DATE, ASSET_RETURN_FREQUENCY, ASSET_AUTO_ADJUST, ASSET_UPLOADED_FILE

try:
    from ui.state.updates import queue_and_rerun
except Exception:  # pragma: no cover
    queue_and_rerun = None


def _coerce_mapping(value: Any) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        return dict(value)
    except Exception:
        return {}


def _coerce_dataframe(value: Any) -> pd.DataFrame:
    return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


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


def _family_title(family: str) -> str:
    key = str(family or "").strip()
    if key == "strategy_preserving":
        return "Best nearby option"
    if key == "broader_opportunity":
        return "Best broader alternative"
    return "Preset suggestion"


def _family_description(family: str) -> str:
    key = str(family or "").strip()
    if key == "strategy_preserving":
        return "Keeps your overall posture close to the current setup and looks for a cleaner nearby improvement."
    if key == "broader_opportunity":
        return "Tests a broader alternative that may improve the final trade-off, even if it changes the setup more noticeably."
    return "Rerun-tested preset suggestion."


def _resolve_top_candidate(family: str) -> dict:
    payload = _coerce_mapping(st.session_state.get("step5_preset_recommendations", {}))
    evaluations = list(payload.get("evaluations", []) or st.session_state.get("step5_preset_evaluations", []) or [])
    family_matches = []
    for raw in evaluations:
        item = _coerce_mapping(raw)
        if str(item.get("family", "") or "").strip() == str(family or "").strip():
            family_matches.append(item)
    if not family_matches:
        return {}

    def _sort_key(item: dict) -> tuple:
        perf = _coerce_mapping(item.get("performance_summary", item.get("perf", {})))
        return (
            1 if bool(item.get("accepted", False)) else 0,
            _safe_float(item.get("adjusted_score", 0.0), 0.0),
            _safe_float(perf.get("sharpe", 0.0), 0.0),
            _safe_float(perf.get("cagr", 0.0), 0.0),
        )

    return dict(sorted(family_matches, key=_sort_key, reverse=True)[0])


def _apply_patch(patch: dict) -> None:
    patch = dict(patch or {})
    if not patch:
        st.warning("This suggested setup does not expose an apply patch.")
        return

    pending_patch = dict(patch)
    rerun_patch = {
        **dict(patch),
        "step5_pending_apply_patch": dict(pending_patch),
        "step5_force_run_once": True,
        "step5_auto_run_reason": "preset_apply",
        "step5_auto_run_pending_patch_keys": list(pending_patch.keys()),
        "step5_continue_improvements_after_apply": True,
        "step5_show_improvements": True,
        "step5_request_improvements": False,
        "step5_last_applied_improvement_step": "preset",
        "step5_resume_improvement_step": "auto_opt",
        "step5_improve_success_message": "Step 1 completed. Preset improvement applied. You can continue with Step 2.",
        "engine_has_run": False,
    }

    if callable(queue_and_rerun):
        queue_and_rerun(rerun_patch)
        return
    for key, value in rerun_patch.items():
        st.session_state[key] = value
    st.rerun()




def _queue_updates(patch: dict) -> None:
    patch = dict(patch or {})
    if not patch:
        return
    if callable(queue_and_rerun):
        queue_and_rerun(patch)
        return
    for key, value in patch.items():
        st.session_state[key] = value
    st.rerun()


def _build_cfg_patch_delta(base_payload: Any, target_payload: Any) -> dict:
    base_map = _coerce_mapping(base_payload)
    target_map = _coerce_mapping(target_payload)
    delta = {}
    for key, value in target_map.items():
        if base_map.get(key) != value:
            delta[key] = value
    return delta


def _build_widget_patch_from_cfg_payload(cfg_payload: Any) -> dict:
    payload = _coerce_mapping(cfg_payload)
    patch = {}
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
    for field_name, widget_key in field_to_widget.items():
        if field_name in payload:
            patch[widget_key] = payload.get(field_name)
    return patch



def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _row_to_dict(row: Any) -> dict:
    if row is None:
        return {}
    to_dict = getattr(row, "to_dict", None)
    if callable(to_dict):
        try:
            out = to_dict()
            if isinstance(out, dict):
                return dict(out)
        except Exception:
            pass
    if isinstance(row, dict):
        return dict(row)
    try:
        return dict(row)
    except Exception:
        return {}


def _coerce_cfg_payload(cfg_payload: Any) -> dict:
    payload = _coerce_mapping(cfg_payload)
    valid_fields = {f.name for f in fields(MicroPipelineConfig)}
    return {k: v for k, v in payload.items() if k in valid_fields}


def _current_cfg_payload() -> dict:
    return _coerce_cfg_payload(
        st.session_state.get("step5_last_cfg_final", st.session_state.get("last_engine_config", {}))
    )


def _current_panel_df() -> pd.DataFrame:
    panel = st.session_state.get("asset_panel_df", pd.DataFrame())
    return panel.copy() if isinstance(panel, pd.DataFrame) else pd.DataFrame()


def _score_candidate_from_perf(perf: Any, philosophy: Any = None) -> float:
    perf_map = _coerce_mapping(perf)
    sharpe = _safe_float(perf_map.get("sharpe", 0.0), 0.0)
    cagr = _safe_float(perf_map.get("cagr", 0.0), 0.0)
    maxdd = abs(_safe_float(perf_map.get("max_drawdown", 0.0), 0.0))
    profile = str(philosophy or st.session_state.get("investment_philosophy", "Balanced") or "Balanced").strip().lower()
    if profile == "growth":
        return float((0.40 * sharpe) + (0.70 * cagr) - (0.30 * maxdd))
    if profile == "defensive":
        return float((0.60 * sharpe) + (0.20 * cagr) - (0.80 * maxdd))
    return float((0.50 * sharpe) + (0.30 * cagr) - (0.50 * maxdd))


def _hash_scope(payload: Any) -> str:
    encoded = json.dumps(_coerce_mapping(payload), sort_keys=True, default=str)
    return hashlib.md5(encoded.encode("utf-8")).hexdigest()[:16]


def _build_universe_opt_scope() -> str:
    payload = {
        "philosophy": str(st.session_state.get("investment_philosophy", "Balanced") or "Balanced"),
        "strategy": str(st.session_state.get("universe_strategy", "Core multi-asset") or "Core multi-asset"),
        "current_size": _safe_int(st.session_state.get("universe_size", 25), 25),
        "cfg_fp": _hash_scope(_current_cfg_payload()),
        "asset_count": int(_current_panel_df()["asset"].nunique()) if isinstance(_current_panel_df(), pd.DataFrame) and "asset" in _current_panel_df().columns else 0,
    }
    return _hash_scope(payload)


def _build_auto_opt_scope() -> str:
    payload = {
        "philosophy": str(st.session_state.get("investment_philosophy", "Balanced") or "Balanced"),
        "current_size": _safe_int(st.session_state.get("universe_size", 25), 25),
        "cfg_fp": _hash_scope(_current_cfg_payload()),
        "asset_count": int(_current_panel_df()["asset"].nunique()) if isinstance(_current_panel_df(), pd.DataFrame) and "asset" in _current_panel_df().columns else 0,
    }
    return _hash_scope(payload)


def _run_cfg_on_panel(cfg_payload: Any, asset_panel_df: Any) -> dict:
    panel = asset_panel_df.copy() if isinstance(asset_panel_df, pd.DataFrame) else pd.DataFrame()
    if panel.empty:
        return {}
    payload = _coerce_cfg_payload(cfg_payload)
    if not payload:
        return {}
    try:
        cfg_obj = MicroPipelineConfig(**payload)
        result = run_micro_investment_pipeline(panel, cfg=cfg_obj)
        result_map = _coerce_mapping(result)
        perf = _coerce_mapping(result_map.get("performance_summary", {}))
        if not perf:
            return {}
        return {"run": result_map, "perf": perf}
    except Exception:
        return {}




def _build_size_search_candidate_assets(size_value: int, strategy_name: Any) -> list[str]:
    """
    Build a broader candidate pool for size search without letting refinement
    sizes collapse back to the canonical buckets.
    """
    try:
        size_n = int(size_value)
    except Exception:
        size_n = 0
    if size_n < 2:
        return []

    pool_size = max(size_n * 2, 50)
    raw_assets = build_strategy_candidate_pool(pool_size, strategy_name)

    cleaned: list[str] = []
    seen = set()
    for raw in list(raw_assets or []):
        asset = str(raw).strip().upper()
        if asset and asset not in seen:
            seen.add(asset)
            cleaned.append(asset)

    return cleaned[:size_n]


def _resolve_size_candidate_panel(candidate_assets: list[str]) -> tuple[pd.DataFrame, str, float]:
    """
    Rebuild a real candidate panel for the requested size when possible.
    Falls back to the current panel filter only if rebuild fails.
    """
    desired_assets = [str(x).strip().upper() for x in list(candidate_assets or []) if str(x).strip()]
    if len(desired_assets) < 2:
        return pd.DataFrame(), "", 0.0

    source_mode = str(st.session_state.get(ASSET_SOURCE_MODE, "yahoo") or "yahoo").lower()
    frequency = str(st.session_state.get(ASSET_RETURN_FREQUENCY, "monthly") or "monthly")
    auto_adjust = bool(st.session_state.get(ASSET_AUTO_ADJUST, True))
    uploaded_file = st.session_state.get(ASSET_UPLOADED_FILE) if source_mode != "yahoo" else None

    t0 = time.perf_counter()
    try:
        rebuilt_panel, source_label, _ = resolve_step4_asset_panel(
            selected_assets=list(desired_assets),
            candidate_assets=[],
            source_mode=source_mode,
            start_date=st.session_state.get(ASSET_START_DATE) if source_mode == "yahoo" else None,
            end_date=st.session_state.get(ASSET_END_DATE) if source_mode == "yahoo" else None,
            frequency=frequency,
            auto_adjust=auto_adjust,
            uploaded_file=uploaded_file,
        )
        panel = rebuilt_panel.copy() if isinstance(rebuilt_panel, pd.DataFrame) else pd.DataFrame()
        if isinstance(panel, pd.DataFrame) and not panel.empty and "asset" in panel.columns:
            panel["asset"] = panel["asset"].astype(str).str.upper().str.strip()
            panel = panel.loc[panel["asset"].isin(set(desired_assets))].copy()
        return panel, str(source_label or ""), float(time.perf_counter() - t0)
    except Exception:
        fallback = _current_panel_df()
        if isinstance(fallback, pd.DataFrame) and not fallback.empty and "asset" in fallback.columns:
            fallback["asset"] = fallback["asset"].astype(str).str.upper().str.strip()
            fallback = fallback.loc[fallback["asset"].isin(set(desired_assets))].copy()
        return fallback if isinstance(fallback, pd.DataFrame) else pd.DataFrame(), "fallback_current_panel", float(time.perf_counter() - t0)

def _run_universe_size_search_local(*, asset_panel_df: pd.DataFrame, strategy_name: Any, cfg_payload: dict, philosophy: Any, sizes: list[int]) -> tuple[list[dict], dict]:
    base_panel = asset_panel_df.copy() if isinstance(asset_panel_df, pd.DataFrame) else pd.DataFrame()
    results = []
    timing_rows = []
    block_t0 = time.perf_counter()

    for raw_size in list(sizes or []):
        try:
            size_value = int(raw_size)
        except Exception:
            continue
        if size_value < 2:
            continue

        candidate_assets = _build_size_search_candidate_assets(size_value, strategy_name)
        if len(candidate_assets) < 2:
            continue

        candidate_panel, panel_source, panel_elapsed = _resolve_size_candidate_panel(candidate_assets)
        effective_assets = []
        if isinstance(candidate_panel, pd.DataFrame) and not candidate_panel.empty and "asset" in candidate_panel.columns:
            effective_assets = sorted(candidate_panel["asset"].dropna().astype(str).unique().tolist())

        if len(effective_assets) < 2:
            continue

        candidate_cfg_payload = dict(cfg_payload or {})
        candidate_top_k = int(size_value)
        candidate_cfg_payload["top_k"] = candidate_top_k
        candidate_cfg_payload["size_search_mode"] = "exact"

        run_t0 = time.perf_counter()
        run_payload = _run_cfg_on_panel(candidate_cfg_payload, candidate_panel)
        run_elapsed = time.perf_counter() - run_t0

        perf = _coerce_mapping(run_payload.get("perf", {}))
        if not perf:
            continue

        final_score = _score_candidate_from_perf(perf, philosophy)
        results.append(
            {
                "size": int(size_value),
                "assets": list(effective_assets),
                "asset_count": int(len(effective_assets)),
                "run": _coerce_mapping(run_payload.get("run", {})),
                "perf": perf,
                "score": float(final_score),
                "coherence_score": float("nan"),
                "coherence_penalty": 0.0,
                "final_score": float(final_score),
                "strategy": str(strategy_name or "Core multi-asset"),
                "effective_top_k": int(candidate_top_k),
                "cfg_payload": dict(candidate_cfg_payload),
                "panel_source": str(panel_source or ""),
            }
        )
        timing_rows.append(
            {
                "block": "size",
                "size": int(size_value),
                "assets_used": int(len(effective_assets)),
                "panel_source": str(panel_source or ""),
                "panel_sec": float(panel_elapsed),
                "run_sec": float(run_elapsed),
                "total_sec": float(panel_elapsed + run_elapsed),
            }
        )

    sorted_results = sorted(results, key=lambda item: _safe_float(item.get("final_score", float("-inf")), float("-inf")), reverse=True)
    meta = {
        "elapsed_sec": float(time.perf_counter() - block_t0),
        "candidates_tested": int(len(results)),
        "timing_rows": pd.DataFrame(timing_rows),
    }
    return sorted_results, meta

def _run_and_store_universe_size_optimisation() -> None:
    panel = _current_panel_df()
    cfg_payload = _current_cfg_payload()
    if panel.empty or not cfg_payload:
        st.warning("A valid Step 5 run is required before universe size optimisation can be launched.")
        return
    philosophy = str(st.session_state.get("investment_philosophy", "Balanced") or "Balanced")
    strategy_name = str(st.session_state.get("universe_strategy", "Core multi-asset") or "Core multi-asset")
    current_size = _safe_int(st.session_state.get("universe_size", 25), 25)
    coarse_sizes = [int(x) for x in recommended_universe_sizes_for_philosophy(philosophy) if int(x) >= 2]
    if current_size >= 2 and current_size not in coarse_sizes:
        coarse_sizes.append(int(current_size))
    coarse_sizes = list(dict.fromkeys(coarse_sizes))
    coarse_results, coarse_meta = _run_universe_size_search_local(
        asset_panel_df=panel,
        strategy_name=strategy_name,
        cfg_payload=cfg_payload,
        philosophy=philosophy,
        sizes=coarse_sizes,
    )
    coarse_results = [dict(item, phase="coarse") for item in coarse_results]
    coarse_best = coarse_results[0] if coarse_results else None
    refinement_sizes = []
    if coarse_best:
        best_size = _safe_int(coarse_best.get("size", 0), 0)
        for offset in (-6, -3, 3, 6):
            candidate_size = int(best_size + offset)
            if candidate_size >= 2 and candidate_size not in coarse_sizes and candidate_size not in refinement_sizes:
                refinement_sizes.append(candidate_size)
    refinement_results, refinement_meta = _run_universe_size_search_local(
        asset_panel_df=panel,
        strategy_name=strategy_name,
        cfg_payload=cfg_payload,
        philosophy=philosophy,
        sizes=refinement_sizes,
    ) if refinement_sizes else ([], {"elapsed_sec": 0.0, "candidates_tested": 0, "timing_rows": pd.DataFrame()})
    refinement_results = [dict(item, phase="refinement") for item in refinement_results]
    combined_map = {}
    for item in list(coarse_results) + list(refinement_results):
        size_value = _safe_int(item.get("size", 0), 0)
        if size_value <= 0:
            continue
        existing = combined_map.get(size_value)
        if existing is None or _safe_float(item.get("final_score", float("-inf")), float("-inf")) > _safe_float(existing.get("final_score", float("-inf")), float("-inf")):
            combined_map[size_value] = dict(item)
    combined_results = sorted(list(combined_map.values()), key=lambda item: _safe_float(item.get("final_score", float("-inf")), float("-inf")), reverse=True)
    best_result = dict(combined_results[0]) if combined_results else {}
    timing_parts = []
    if isinstance(coarse_meta.get("timing_rows"), pd.DataFrame) and not coarse_meta.get("timing_rows").empty:
        timing_parts.append(coarse_meta.get("timing_rows"))
    if isinstance(refinement_meta.get("timing_rows"), pd.DataFrame) and not refinement_meta.get("timing_rows").empty:
        timing_parts.append(refinement_meta.get("timing_rows"))
    size_timing_df = pd.concat(timing_parts, ignore_index=True) if timing_parts else pd.DataFrame()

    st.session_state["universe_size_optimisation_results"] = {
        "coarse_sizes": list(coarse_sizes),
        "refinement_sizes": list(refinement_sizes),
        "coarse_results": list(coarse_results),
        "refinement_results": list(refinement_results),
        "combined_results": list(combined_results),
        "best_result": best_result,
        "elapsed_sec": float(_safe_float(coarse_meta.get("elapsed_sec", 0.0), 0.0) + _safe_float(refinement_meta.get("elapsed_sec", 0.0), 0.0)),
        "candidates_tested": int(_safe_int(coarse_meta.get("candidates_tested", 0), 0) + _safe_int(refinement_meta.get("candidates_tested", 0), 0)),
        "timing_rows": size_timing_df,
    }
    st.session_state["universe_size_optimisation_scope"] = _build_universe_opt_scope()
    st.session_state["step5_show_improvements"] = True
    st.session_state["step5_request_improvements"] = False
    current_sig = str(st.session_state.get("step5_current_input_signature", "") or "")
    if current_sig:
        st.session_state["step5_improvements_requested_signature"] = current_sig
    st.rerun()


def _summarize_cfg_payload_diff(base_payload: Any, target_payload: Any) -> list[dict]:
    base_map = _coerce_mapping(base_payload)
    target_map = _coerce_mapping(target_payload)
    rows = []
    for key, value in target_map.items():
        if base_map.get(key) == value:
            continue
        rows.append({"parameter": str(key), "current": base_map.get(key), "optimised": value})
    return rows


def _run_and_store_auto_optimisation(simple_cfg: Any = None) -> None:
    auto_t0 = time.perf_counter()
    panel = _current_panel_df()
    base_cfg_payload = _current_cfg_payload()
    if panel.empty or not base_cfg_payload:
        st.warning("A valid Step 5 run is required before local optimisation can be launched.")
        return
    simple_summary = _coerce_mapping(simple_cfg or st.session_state.get("step5_last_simple_cfg", {}))
    philosophy = str(st.session_state.get("investment_philosophy", "Balanced") or "Balanced")
    try:
        tuning_result = run_simple_auto_optimize(
            panel,
            base_cfg_payload=base_cfg_payload,
            simple_summary=simple_summary,
            include_weight_shrink=False,
            coherence_philosophy=philosophy,
        )
        baseline_payload = _coerce_mapping(tuning_result.get("baseline_payload", {})) or dict(base_cfg_payload)
        final_payload = _coerce_mapping(tuning_result.get("final_cfg_payload", {})) or dict(baseline_payload)
        auto_opt_result = {
            "elapsed_sec": float(time.perf_counter() - auto_t0),
            "baseline_payload": dict(baseline_payload),
            "baseline_cfg_payload": dict(baseline_payload),
            "final_cfg_payload": dict(final_payload),
            "best_payload": _coerce_mapping(tuning_result.get("best_payload", {})),
            "tuning_result": tuning_result,
            "was_optimized": bool(tuning_result.get("was_optimized", False)),
            "used_auto_opt": True,
            "objective_name": str(tuning_result.get("objective_name", "") or ""),
            "objective_improvement": tuning_result.get("objective_improvement"),
            "trials_explored": _safe_int(tuning_result.get("trials_explored", 0), 0),
            "changed_params": _summarize_cfg_payload_diff(baseline_payload, final_payload),
            "selection_policy": str(tuning_result.get("selection_policy", "") or "fixed_composite_score"),
            "composite_profile": tuning_result.get("composite_profile"),
            "failure_reason": "" if bool(tuning_result.get("was_optimized", False)) else str(tuning_result.get("failure_reason", "no_improvement_found") or "no_improvement_found"),
            "best_row": _row_to_dict(tuning_result.get("best_row")),
            "baseline_row": _row_to_dict(tuning_result.get("baseline_row")),
        }
    except Exception as exc:
        auto_opt_result = {
            "elapsed_sec": float(time.perf_counter() - auto_t0),
            "baseline_payload": dict(base_cfg_payload),
            "baseline_cfg_payload": dict(base_cfg_payload),
            "final_cfg_payload": dict(base_cfg_payload),
            "best_payload": {},
            "tuning_result": {},
            "was_optimized": False,
            "used_auto_opt": True,
            "objective_name": "",
            "objective_improvement": None,
            "trials_explored": 0,
            "changed_params": [],
            "selection_policy": "fixed_composite_score",
            "composite_profile": None,
            "failure_reason": f"search_failed: {exc}",
            "best_row": {},
            "baseline_row": {},
        }
    # ------------------------------------------------------------
    # Sequential flow behaviour
    # ------------------------------------------------------------
    # Step 2 should not leave the user on the generic Run panel. Once the
    # local search has completed, this step is considered resolved:
    # - if a useful payload was found, apply it and force one real engine run;
    # - if no useful payload was found, keep the current config and unlock Step 3.
    # This keeps the Improve wizard cumulative: Preset -> Auto-opt -> Universe -> Size.
    current_sig = str(st.session_state.get("step5_current_input_signature", "") or "")
    baseline_payload = _coerce_mapping(auto_opt_result.get("baseline_cfg_payload", auto_opt_result.get("baseline_payload", {})))
    final_payload = _coerce_mapping(auto_opt_result.get("final_cfg_payload", {})) or dict(baseline_payload)
    persisted_patch = _build_cfg_patch_delta(baseline_payload, final_payload)
    has_real_change = bool(auto_opt_result.get("was_optimized", False)) and bool(persisted_patch)

    patch = {
        "auto_optimisation_result": dict(auto_opt_result),
        "auto_optimization_result": dict(auto_opt_result),
        "auto_optimisation_scope": _build_auto_opt_scope(),
        "step5_continue_improvements_after_apply": True,
        "step5_show_improvements": True,
        "step5_request_improvements": False,
        "step5_last_applied_improvement_step": "auto_opt",
        "step5_resume_improvement_step": "universe",
        "step5_improvements_requested_signature": current_sig,
    }

    if has_real_change:
        pending_widget_patch = _build_widget_patch_from_cfg_payload(final_payload)
        patch.update(
            {
                "engine_has_run": False,
                "step5_force_run_once": True,
                "step5_auto_run_reason": "auto_opt_apply",
                "step5_auto_run_pending_patch_keys": list(pending_widget_patch.keys()),
                "step5_improve_success_message": "Step 2 completed. Auto-opt setup applied. You can continue with Step 3.",
                "step5_last_cfg_final": dict(final_payload),
                "last_applied_engine_config": dict(final_payload),
                "last_applied_engine_source": "Post-run auto-optimisation",
                "universe_simple_override_patch": dict(persisted_patch),
                "universe_simple_override_source": "Post-run auto-optimisation",
                "universe_simple_override_scope_key": str(st.session_state.get("universe_size", 25) or 25),
            }
        )
        patch.update(pending_widget_patch)
    else:
        patch.update(
            {
                "step5_improve_success_message": "Step 2 completed. No useful auto-opt change was needed. You can continue with Step 3.",
            }
        )

    _queue_updates(patch)

def _apply_universe_size_best_result(best_result: Any) -> None:
    result = _coerce_mapping(best_result)
    try:
        size_value = int(result.get("size", 0) or 0)
    except Exception:
        size_value = 0
    assets = [str(x).strip().upper() for x in list(result.get("assets", []) or []) if str(x).strip()]
    if size_value <= 0:
        st.warning("This universe-size candidate does not expose a valid target size.")
        return

    # The size search evaluates candidates with a real candidate cfg/panel.
    # Applying the size must therefore do more than update `universe_size`: it
    # must also force a fresh Step 5 run on the selected assets and then return
    # to the Improve workflow. Otherwise Streamlit lands on the generic Run
    # panel with a stale panel/signature after the portfolio structure changed.
    candidate_cfg_payload = _coerce_cfg_payload(result.get("cfg_payload", {}))
    widget_patch = _build_widget_patch_from_cfg_payload(candidate_cfg_payload) if candidate_cfg_payload else {}

    patch = {
        # --- portfolio structure ---
        "universe_size": int(size_value),
        "universe_custom_enabled_source": "size_optimization",
        "recommended_universe_assets": list(assets),
        "last_recommendation_candidate_assets": list(assets),

        # --- force a real rerun with the updated universe/panel ---
        "engine_has_run": False,
        "step5_force_run_once": True,
        "step5_force_panel_refresh": True,
        "step5_auto_run_reason": "size_apply",
        "step5_auto_run_pending_patch_keys": ["universe_size", "selected_assets"] + list(widget_patch.keys()),

        # --- keep the user inside the sequential Improve wizard ---
        "step5_continue_improvements_after_apply": True,
        "step5_show_improvements": True,
        "step5_request_improvements": False,
        "step5_request_universe_recommendation": False,
        "step5_last_applied_improvement_step": "size",
        "step5_resume_improvement_step": "done",
        "step5_improve_success_message": "Step 4 completed. Portfolio size update applied and rerun. Improvement workflow is complete.",
    }

    if assets:
        patch["selected_assets"] = list(assets)

    if candidate_cfg_payload:
        patch["step5_last_cfg_final"] = dict(candidate_cfg_payload)
        patch["last_applied_engine_config"] = dict(candidate_cfg_payload)
        patch["last_applied_engine_source"] = "Post-run universe-size optimisation"
        patch.update(widget_patch)

    _queue_updates(patch)


def _apply_auto_optimised_payload(auto_opt_info: Any) -> None:
    info = _coerce_mapping(auto_opt_info)
    final_payload = _coerce_mapping(info.get("final_cfg_payload", {}))
    baseline_payload = _coerce_mapping(info.get("baseline_cfg_payload", info.get("baseline_payload", {})))
    if not final_payload:
        st.warning("No optimised config payload is available to apply.")
        return

    persisted_patch = _build_cfg_patch_delta(baseline_payload, final_payload)
    if not persisted_patch:
        persisted_patch = dict(final_payload)

    widget_patch = _build_widget_patch_from_cfg_payload(final_payload)
    patch = {
        "engine_has_run": False,
        "step5_continue_improvements_after_apply": True,
        "step5_show_improvements": True,
        "step5_request_improvements": False,
        "step5_last_applied_improvement_step": "auto_opt",
        "step5_resume_improvement_step": "universe",
        "step5_improve_success_message": "Step 2 completed. Auto-opt setup applied. You can continue with Step 3.",
        "step5_last_cfg_final": dict(final_payload),
        "last_applied_engine_config": dict(final_payload),
        "last_applied_engine_source": "Post-run auto-optimisation",
        "universe_simple_override_patch": dict(persisted_patch),
        "universe_simple_override_source": "Post-run auto-optimisation",
        "universe_simple_override_scope_key": str(st.session_state.get("universe_size", 25) or 25),
        "step5_force_run_once": True,
        "step5_auto_run_reason": "auto_opt_apply",
        "step5_auto_run_pending_patch_keys": list(widget_patch.keys()),
    }
    patch.update(widget_patch)
    _queue_updates(patch)



def _render_universe_size_optimisation_block() -> None:
    optimisation_payload = _coerce_mapping(st.session_state.get("universe_size_optimisation_results", {}))
    best_result = _coerce_mapping(optimisation_payload.get("best_result", {}))
    combined_results = list(optimisation_payload.get("combined_results", []) or [])
    current_scope = _build_universe_opt_scope()
    saved_scope = str(st.session_state.get("universe_size_optimisation_scope", "") or "")

    with st.expander("Universe size optimisation (advanced)", expanded=False):
        coarse_sizes = [int(x) for x in recommended_universe_sizes_for_philosophy(st.session_state.get("investment_philosophy", "Balanced")) if int(x) >= 2]
        current_size = int(st.session_state.get("universe_size", 25) or 25)
        current_strategy = str(st.session_state.get("universe_strategy", "Core multi-asset") or "Core multi-asset")
        philosophy = str(st.session_state.get("investment_philosophy", "Balanced") or "Balanced")

        st.caption("Strategy change · New structure · Change magnitude: HIGH")

        info_cols = st.columns(4)
        with info_cols[0]:
            st.metric("Philosophy", philosophy)
        with info_cols[1]:
            st.metric("Current size", current_size)
        with info_cols[2]:
            st.metric("Coarse sizes", ", ".join(str(x) for x in coarse_sizes) if coarse_sizes else "—")
        with info_cols[3]:
            display_strategy = current_strategy if len(current_strategy) <= 12 else f"{current_strategy[:9]}…"
            st.metric("Current strategy", display_strategy)

        if st.button("Run universe size optimisation", key="run_universe_size_optimisation", use_container_width=True):
            _run_and_store_universe_size_optimisation()
            return

        if not optimisation_payload:
            st.info("No universe-size optimisation results are stored yet.")
            return
        if saved_scope and saved_scope != current_scope:
            st.info("Saved universe-size optimisation results are stale for the current setup. Run the search again to refresh them.")
            return

        if combined_results:
            rows = []
            for item in combined_results:
                row = _coerce_mapping(item)
                perf = _coerce_mapping(row.get("perf", {}))
                rows.append({
                    "phase": str(row.get("phase", "search") or "search"),
                    "size": int(row.get("size", 0) or 0),
                    "assets": int(len(row.get("assets", []) or [])),
                    "sharpe": _safe_float(perf.get("sharpe", 0.0), 0.0),
                    "cagr": _safe_float(perf.get("cagr", 0.0), 0.0),
                    "max_drawdown": _safe_float(perf.get("max_drawdown", 0.0), 0.0),
                    "final_score": _safe_float(row.get("final_score", 0.0), 0.0),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if best_result:
            best_perf = _coerce_mapping(best_result.get("perf", {}))
            st.success(f"Best candidate found: {int(best_result.get('size', 0) or 0)} assets.")
            best_cols = st.columns(5)
            with best_cols[0]:
                st.metric("Best size", int(best_result.get("size", 0) or 0))
            with best_cols[1]:
                st.metric("Sharpe", f"{_safe_float(best_perf.get('sharpe', 0.0), 0.0):.2f}")
            with best_cols[2]:
                st.metric("CAGR", _format_pct(best_perf.get("cagr", 0.0)))
            with best_cols[3]:
                st.metric("MaxDD", _format_pct(best_perf.get("max_drawdown", 0.0)))
            with best_cols[4]:
                st.metric("Final score", f"{_safe_float(best_result.get('final_score', 0.0), 0.0):.2f}")
            if st.button("Apply best size", key="apply_best_universe_size_candidate", use_container_width=True):
                _apply_universe_size_best_result(best_result)


def _render_auto_optimisation_block(simple_cfg: Any = None) -> None:
    auto_opt_info = _coerce_mapping(st.session_state.get("auto_optimisation_result", st.session_state.get("auto_optimization_result", {})))
    current_scope = _build_auto_opt_scope()
    saved_scope = str(st.session_state.get("auto_optimisation_scope", "") or "")

    with st.expander("Local optimisation (advanced)", expanded=False):
        if st.button("Run auto-optimisation", key="run_auto_optimisation", use_container_width=True):
            _run_and_store_auto_optimisation(simple_cfg=simple_cfg)
            return

        if not auto_opt_info:
            st.info("No auto-optimisation results are stored yet.")
            return
        if saved_scope and saved_scope != current_scope:
            st.info("Saved local-optimisation results are stale for the current setup. Run the search again to refresh them.")
            return

        was_optimized = bool(auto_opt_info.get("was_optimized", False))
        best_row = _coerce_mapping(auto_opt_info.get("best_row", {}))
        baseline_row = _coerce_mapping(auto_opt_info.get("baseline_row", {}))
        changed_params = list(auto_opt_info.get("changed_params", []) or [])
        final_payload = _coerce_mapping(auto_opt_info.get("final_cfg_payload", {}))

        st.caption("Fine-tune current strategy parameters")
        st.caption("Parameter tuning · Small adjustments · Change magnitude: LOW")

        summary_cols = st.columns(4)
        with summary_cols[0]:
            st.metric("Trials", int(auto_opt_info.get("trials_explored", 0) or 0))
        with summary_cols[1]:
            st.metric("Selection policy", str(auto_opt_info.get("selection_policy", "—") or "—"))
        with summary_cols[2]:
            st.metric("Objective", str(auto_opt_info.get("objective_name", "—") or "—"))
        with summary_cols[3]:
            st.metric("Optimised", "Yes" if was_optimized else "No")

        if best_row:
            best_cols = st.columns(4)
            with best_cols[0]:
                st.metric("Best objective", f"{_safe_float(best_row.get('objective_value_adjusted', best_row.get('objective_value', 0.0)), 0.0):.3f}")
            with best_cols[1]:
                st.metric("Coherence", f"{_safe_float(best_row.get('coherence_score', 0.0), 0.0):.2f}")
            with best_cols[2]:
                st.metric("Penalty", f"{_safe_float(best_row.get('coherence_penalty', 0.0), 0.0):.2f}")
            with best_cols[3]:
                st.metric("Label", str(best_row.get("coherence_label", "—") or "—"))
            if baseline_row:
                st.caption(
                    f"Baseline adjusted objective={_safe_float(baseline_row.get('objective_value_adjusted', baseline_row.get('objective_value', 0.0)), 0.0):.3f} · "
                    f"Best adjusted objective={_safe_float(best_row.get('objective_value_adjusted', best_row.get('objective_value', 0.0)), 0.0):.3f}"
                )

        if was_optimized and changed_params:
            st.dataframe(pd.DataFrame(changed_params), use_container_width=True, hide_index=True)
        elif str(auto_opt_info.get("failure_reason", "") or "").strip():
            st.caption(f"Auto-optimisation kept the baseline config: {str(auto_opt_info.get('failure_reason', '') or '').strip()}")

        if final_payload:
            apply_cols = st.columns([1.2, 2.8])
            with apply_cols[0]:
                if st.button("Apply optimised setup", key="apply_auto_optimised_setup", use_container_width=True):
                    _apply_auto_optimised_payload(auto_opt_info)
            with apply_cols[1]:
                st.caption("This applies the optimised configuration to the current Step 5 setup and reruns the workspace.")

def _build_current_score(perf: dict) -> float:
    sharpe = _safe_float(perf.get("sharpe", 0.0), 0.0)
    cagr = _safe_float(perf.get("cagr", 0.0), 0.0)
    max_dd = abs(_safe_float(perf.get("max_drawdown", 0.0), 0.0))
    return float(sharpe + (0.50 * cagr) - (0.75 * max_dd))



def _has_material_improvement(candidate_eval: dict, current_perf: dict) -> bool:
    candidate_eval = _coerce_mapping(candidate_eval)
    perf = _coerce_mapping(candidate_eval.get("performance_summary", candidate_eval.get("perf", {})))

    if not perf:
        return False

    sharpe = _safe_float(perf.get("sharpe", 0.0), 0.0)
    cagr = _safe_float(perf.get("cagr", 0.0), 0.0)
    max_dd = abs(_safe_float(perf.get("max_drawdown", 0.0), 0.0))

    current_sharpe = _safe_float(current_perf.get("sharpe", 0.0), 0.0)
    current_cagr = _safe_float(current_perf.get("cagr", 0.0), 0.0)
    current_dd = abs(_safe_float(current_perf.get("max_drawdown", 0.0), 0.0))

    delta_sharpe = sharpe - current_sharpe
    delta_cagr = cagr - current_cagr
    delta_dd = max_dd - current_dd

    return (
        (delta_sharpe > 0.005)
        or (delta_cagr > 0.0005)
        or (delta_dd < -0.0005)
    )

def _render_candidate_card(candidate_eval: dict, current_perf: dict, button_key: str) -> None:
    candidate_eval = _coerce_mapping(candidate_eval)
    perf = _coerce_mapping(candidate_eval.get("performance_summary", candidate_eval.get("perf", {})))
    family = str(candidate_eval.get("family", "") or "").strip()

    title = _family_title(family)
    description = _family_description(family)

    st.markdown(f"**{title}**")
    st.caption(description)

    if not perf:
        st.info("No rerun-tested suggestion is available for this family in this run.")
        return

    sharpe = _safe_float(perf.get("sharpe", 0.0), 0.0)
    cagr = _safe_float(perf.get("cagr", 0.0), 0.0)
    max_dd = abs(_safe_float(perf.get("max_drawdown", 0.0), 0.0))

    current_sharpe = _safe_float(current_perf.get("sharpe", 0.0), 0.0)
    current_cagr = _safe_float(current_perf.get("cagr", 0.0), 0.0)
    current_dd = abs(_safe_float(current_perf.get("max_drawdown", 0.0), 0.0))

    delta_sharpe = sharpe - current_sharpe
    delta_cagr = cagr - current_cagr
    delta_dd = max_dd - current_dd

    label = str(candidate_eval.get("label", candidate_eval.get("candidate_label", title)) or title)
    config_summary = str(candidate_eval.get("config_summary", "") or "").strip()
    reason = str(candidate_eval.get("reason", "") or "").strip()
    gov_state = str(candidate_eval.get("governance_state", "") or "").strip()
    accepted = bool(candidate_eval.get("accepted", False))

    candidate_score = _safe_float(candidate_eval.get("adjusted_score", 0.0), 0.0)
    score_delta = candidate_score - _build_current_score(current_perf)

    st.caption(label)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Sharpe", f"{sharpe:.2f}", delta=f"{delta_sharpe:+.2f}")
    with m2:
        st.metric("CAGR", _format_pct(cagr), delta=_format_delta_pct(delta_cagr))
    with m3:
        st.metric("MaxDD", f"-{100.0 * max_dd:.2f}%", delta=_format_delta_pct(-delta_dd))
    with m4:
        st.metric("Score delta", f"{score_delta:+.2f}")

    if config_summary:
        with st.expander("Setup details", expanded=False):
            st.caption(config_summary)

    if accepted:
        st.success("This rerun-tested setup passed the current acceptance gate.")
    else:
        st.info("This rerun-tested setup is shown for inspection, but it did not pass the current acceptance gate.")

    if gov_state:
        st.caption(f"Governance state: {gov_state}")
    if reason:
        st.caption(reason)

    if st.button("Apply suggested setup", key=button_key, use_container_width=True):
        _apply_patch(_coerce_mapping(candidate_eval.get("apply_patch", {})))


def _prepare_diagnostics_table(rows: pd.DataFrame) -> pd.DataFrame:
    work = _coerce_dataframe(rows)
    if work.empty:
        return pd.DataFrame()

    rename_map = {
        "template": "strategy_template",
        "preset": "style_preset",
        "max_drawdown": "maxdd",
    }
    for old, new in rename_map.items():
        if old in work.columns and new not in work.columns:
            work = work.rename(columns={old: new})

    desired = [
        "label",
        "family",
        "strategy_template",
        "style_preset",
        "sharpe",
        "cagr",
        "maxdd",
        "adjusted_score",
        "accepted",
        "source",
    ]
    keep_cols = [c for c in desired if c in work.columns]
    if not keep_cols:
        return work
    return work[keep_cols].copy()


def render_preset_recommendations(run_result: dict, simple_cfg: dict | None = None) -> None:
    run_map = _coerce_mapping(run_result)
    perf = _coerce_mapping(run_map.get("performance_summary", {}))
    if not perf:
        return

    payload = _coerce_mapping(st.session_state.get("step5_preset_recommendations", {}))
    rows = _coerce_dataframe(payload.get("rows", pd.DataFrame()))
    candidate_count = int(payload.get("candidate_count", 0) or 0)
    source = str(payload.get("source", "") or "").strip()

    st.markdown("### Step 1 — Preset improvement")
    st.caption(
        "Start with high-level strategy logic. Review one or two rerun-tested preset suggestions first, "
        "then inspect the diagnostics table below if you want more detail."
    )

    nearby_candidate = _resolve_top_candidate("strategy_preserving")
    broader_candidate = _resolve_top_candidate("broader_opportunity")

    has_nearby_raw = bool(_coerce_mapping(nearby_candidate).get("performance_summary") or _coerce_mapping(nearby_candidate).get("perf"))
    has_broader_raw = bool(_coerce_mapping(broader_candidate).get("performance_summary") or _coerce_mapping(broader_candidate).get("perf"))

    has_nearby = has_nearby_raw and _has_material_improvement(nearby_candidate, perf)
    has_broader = has_broader_raw and _has_material_improvement(broader_candidate, perf)

    if has_nearby and has_broader:
        c1, c2 = st.columns(2)
        with c1:
            _render_candidate_card(
                nearby_candidate,
                perf,
                "step5_apply_strategy_preserving_preset",
            )
        with c2:
            _render_candidate_card(
                broader_candidate,
                perf,
                "step5_apply_broader_preset",
            )
    elif has_nearby:
        _render_candidate_card(
            nearby_candidate,
            perf,
            "step5_apply_strategy_preserving_preset",
        )
    elif has_broader:
        _render_candidate_card(
            broader_candidate,
            perf,
            "step5_apply_broader_preset",
        )
    else:
        st.success("Current preset loop has converged. No further material preset improvement found.")

    diagnostics_df = _prepare_diagnostics_table(rows)

    with st.expander("Preset recommendation diagnostics", expanded=False):
        debug_fast_rows = _coerce_dataframe(payload.get("debug_fast_rows", pd.DataFrame()))
        if not diagnostics_df.empty:
            st.caption(f"source={source or 'preset_confirmed_rerun'} · confirmed_candidates={candidate_count}")
            st.dataframe(diagnostics_df, use_container_width=True, hide_index=True)
        else:
            st.info("No confirmed preset candidates are available for this run.")

        if not debug_fast_rows.empty:
            st.caption("FAST screening debug (internal ranking only)")
            st.dataframe(debug_fast_rows, use_container_width=True, hide_index=True)

    st.caption("Preset-only restore mode: universe and local optimisation assistants are temporarily disabled while the preset loop is being rebuilt.")

def render_auto_opt_recommendations(simple_cfg: dict | None = None) -> None:
    st.markdown("### Step 2 — Auto-opt")
    st.caption(
        "After choosing the best high-level preset, run a local search around the current engine parameters "
        "to see whether nearby values improve performance without changing the overall setup."
    )
    _render_auto_optimisation_block(simple_cfg=simple_cfg)



def _render_improvement_timing_summary() -> None:
    step5_timings = _coerce_mapping(st.session_state.get("step5_last_timings", {}))
    auto_opt_info = _coerce_mapping(st.session_state.get("auto_optimisation_result", st.session_state.get("auto_optimization_result", {})))
    size_info = _coerce_mapping(st.session_state.get("universe_size_optimisation_results", {}))

    preset_sec = _safe_float(step5_timings.get("preset_total_sec", 0.0), 0.0)
    universe_sec = _safe_float(step5_timings.get("universe_total_sec", 0.0), 0.0)
    auto_opt_sec = _safe_float(auto_opt_info.get("elapsed_sec", 0.0), 0.0)
    size_sec = _safe_float(size_info.get("elapsed_sec", 0.0), 0.0)
    total_sec = preset_sec + auto_opt_sec + universe_sec + size_sec

    if total_sec <= 0:
        return

    st.markdown("### Improvement timing summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Step 1 Preset", f"{preset_sec:.2f}s")
    with c2:
        st.metric("Step 2 Auto-opt", f"{auto_opt_sec:.2f}s")
    with c3:
        st.metric("Step 3 Universe", f"{universe_sec:.2f}s")
    with c4:
        st.metric("Step 4 Size", f"{size_sec:.2f}s")
    with c5:
        st.metric("Total", f"{total_sec:.2f}s")

    st.caption(
        f"tested candidates → preset={_safe_int(step5_timings.get('preset_candidates_tested', 0), 0)} · "
        f"universe={_safe_int(step5_timings.get('universe_candidates_tested', 0), 0)} · "
        f"size={_safe_int(size_info.get('candidates_tested', 0), 0)}"
    )

    timing_rows = size_info.get("timing_rows", pd.DataFrame())
    if isinstance(timing_rows, pd.DataFrame) and not timing_rows.empty:
        with st.expander("Size timing breakdown", expanded=False):
            st.dataframe(timing_rows, use_container_width=True, hide_index=True)


def render_size_recommendations() -> None:
    _render_improvement_timing_summary()
    st.markdown("### Step 4 — Portfolio size")
    st.caption(
        "Only after preset, local engine tuning, and same-size universe suggestions should you consider changing "
        "the overall portfolio size. This is the most structural change in the sequence."
    )
    _render_universe_size_optimisation_block()

