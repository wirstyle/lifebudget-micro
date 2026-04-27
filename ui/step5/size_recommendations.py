from __future__ import annotations

"""Step 5 size recommendation assistant.

Phase 4 scope:
- keep the active Step 5 strategy preset unchanged;
- keep the active technical engine configuration unchanged;
- keep the current universe-composition logic as the baseline;
- test only a small number of philosophy-compatible universe sizes;
- apply by promoting an already rerun-tested candidate result.

This module deliberately does not re-run preset, engine-tuning, or universe-
composition search. It changes the requested universe size plus the generated
asset list/panel only after the user explicitly applies the accepted candidate.
"""

import hashlib
import json
import math
import time
from dataclasses import fields
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st

from src.investment import MicroPipelineConfig, run_micro_investment_pipeline
from ui.services.step4_universe_service import (
    UNIVERSE_SIZES,
    allowed_universe_strategies_for_size,
    asset_display_label,
    build_generated_universe,
    build_step4_universe_payload_from_state,
    build_strategy_candidate_pool,
    normalize_asset_ticker,
    philosophy_size_test_cap,
    recommended_universe_sizes_for_philosophy,
    resolve_recommended_strategy_for_size,
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


UNIVERSE_SIZE_SUGGESTION_STATE_KEY = "step5_universe_size_suggestion_v1"
UNIVERSE_SIZE_SUGGESTION_SCOPE_KEY = "step5_universe_size_suggestion_scope_v1"
UNIVERSE_SIZE_APPLIED_SIGNATURE_KEY = "step5_universe_size_applied_run_signature_v1"
UNIVERSE_SIZE_APPLIED_LABEL_KEY = "step5_universe_size_applied_label_v1"
UNIVERSE_SIZE_SKIPPED_SCOPE_KEY = "step5_universe_size_skipped_scope_v1"
UNIVERSE_SIZE_SKIPPED_LABEL_KEY = "step5_universe_size_skipped_label_v1"
UNIVERSE_SIZE_SKIPPED_RUN_SIGNATURE_KEY = "step5_universe_size_skipped_run_signature_v1"
UNIVERSE_SIZE_SUGGESTION_TIMING_KEY = "step5_universe_size_suggestion_timing_v1"
UNIVERSE_SIZE_RECOMMENDATION_CONTEXT_KEY = "step5_recommended_universe_size_context_v1"
UNIVERSE_BASE_PANEL_FOR_SIZE_KEY = "step5_universe_base_panel_for_size_v1"
UNIVERSE_BASE_PANEL_FOR_SIZE_SIGNATURE_KEY = "step5_universe_base_panel_for_size_signature_v1"
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


def _panel_assets(panel_df: pd.DataFrame) -> list[str]:
    if isinstance(panel_df, pd.DataFrame) and not panel_df.empty and "asset" in panel_df.columns:
        return sorted(panel_df["asset"].dropna().astype(str).str.upper().unique().tolist())
    return []


def _get_search_panel(run_result: dict | None = None) -> pd.DataFrame:
    """Return the broad panel used for Phase 4 size tests.

    If Phase 3 applied a same-size composition candidate, the visible Step 5
    panel may have been narrowed to that promoted asset set. Size testing still
    needs the broader Step 4 candidate pool, so Phase 3 preserves that source
    panel under UNIVERSE_BASE_PANEL_FOR_SIZE_KEY. The signature guard prevents
    reusing it after an unrelated fresh run.
    """
    run_map = _coerce_mapping(run_result)
    current_signature = str(run_map.get("run_signature", "") or "")
    stored_signature = str(st.session_state.get(UNIVERSE_BASE_PANEL_FOR_SIZE_SIGNATURE_KEY, "") or "")
    stored_panel = st.session_state.get(UNIVERSE_BASE_PANEL_FOR_SIZE_KEY)
    if (
        current_signature
        and stored_signature == current_signature
        and isinstance(stored_panel, pd.DataFrame)
        and not stored_panel.empty
    ):
        return stored_panel.copy()

    panel = st.session_state.get(ASSET_PANEL_DF, st.session_state.get("asset_panel_df"))
    return panel.copy() if isinstance(panel, pd.DataFrame) else pd.DataFrame()


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
    return {}


def _all_universe_sizes() -> list[int]:
    out: list[int] = []
    for raw in list(UNIVERSE_SIZES or []):
        value = _safe_int(raw, 0)
        if value > 0 and value not in out:
            out.append(value)
    return out or [12, 25, 50, 75, 100, 150, 250]


def _dedupe_sizes(values: list[int]) -> list[int]:
    out: list[int] = []
    for raw in list(values or []):
        value = _safe_int(raw, 0)
        if value > 0 and value not in out:
            out.append(int(value))
    return out


def _philosophy_size_band(philosophy: str) -> tuple[int, int, list[int]]:
    """Return the capped automatic size-search band for the current philosophy.

    This is intentionally narrower than the full Step 4 "show all sizes" space:
    - Defensive: 12-25
    - Balanced: 25-50
    - Growth: 50-100
    """
    philosophy_name = str(philosophy or "Balanced").strip()
    preferred = _dedupe_sizes([_safe_int(x, 0) for x in list(recommended_universe_sizes_for_philosophy(philosophy_name) or [])])
    if not preferred:
        preferred = [50, 75, 100] if philosophy_name == "Growth" else ([12, 25, 50] if philosophy_name == "Defensive" else [25, 50, 75])

    cap = _safe_int(philosophy_size_test_cap(philosophy_name), 50)
    lower = int(min(preferred))
    upper = int(max(lower, cap))

    # Keep canonical checkpoints only inside the automatic philosophy cap. For
    # example Balanced keeps 25/50 and does not automatically scan 75.
    in_band = [int(x) for x in preferred if int(x) >= lower and int(x) <= upper]
    if upper in set(_all_universe_sizes()) and upper not in in_band:
        in_band.append(int(upper))
    if lower not in in_band:
        in_band.insert(0, int(lower))
    return int(lower), int(upper), _dedupe_sizes(in_band)

def _build_assets_for_size(candidate_size: int, strategy_name: str) -> list[str]:
    size_value = max(1, _safe_int(candidate_size, 25))
    if size_value in set(_all_universe_sizes()):
        assets = _asset_list(build_generated_universe(size_value, strategy_name))
    else:
        assets = _asset_list(build_strategy_candidate_pool(size_value, strategy_name))[:size_value]
    return list(assets)[:size_value]


def _local_refinement_sizes(
    *,
    current_size: int,
    philosophy: str,
    max_available_assets: int,
    max_refinements: int = 3,
) -> list[int]:
    """Return a tiny, philosophy-capped local size search.

    Example for Balanced with current=25 and available=37: roughly [30, 34, 37].
    """
    current = max(1, _safe_int(current_size, 25))
    available = max(0, _safe_int(max_available_assets, 0))
    band_min, band_max, _preferred = _philosophy_size_band(philosophy)
    out: list[int] = []

    upper = min(int(available), int(band_max))
    if upper > current:
        span = int(upper - current)
        for frac in (0.42, 0.75, 1.00):
            value = current + max(1, int(math.ceil(span * frac)))
            value = min(value, upper)
            if value != current and value not in out:
                out.append(int(value))
            if len(out) >= int(max_refinements):
                return out[: int(max_refinements)]

    lower = max(1, int(band_min))
    if current > lower:
        span = int(current - lower)
        for frac in (0.42, 0.75, 1.00):
            value = current - max(1, int(math.ceil(span * frac)))
            value = max(value, lower)
            if value != current and value not in out:
                out.append(int(value))
            if len(out) >= int(max_refinements):
                return out[: int(max_refinements)]

    return out[: int(max_refinements)]


# ---------------------------------------------------------------------------
# Candidate selection and evaluation
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
    delta_vol = c["annual_volatility"] - b["annual_volatility"]
    delta_dd = c["max_drawdown"] - b["max_drawdown"]

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
        reason = "Growth gate gives more weight to CAGR, while rejecting sizes that damage Sharpe or drawdown too much."
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


def _candidate_sizes(current_size: int, philosophy: str, *, max_candidates: int = 3) -> list[int]:
    """Compatibility helper kept for older imports/tests."""
    _band_min, _band_max, preferred = _philosophy_size_band(philosophy)
    out = [int(x) for x in preferred if int(x) != int(current_size)]
    return out[: int(max_candidates)]


def _resolve_strategy_for_size(philosophy: str, current_strategy: str, candidate_size: int) -> str:
    allowed = [str(x) for x in list(allowed_universe_strategies_for_size(candidate_size) or [])]
    if str(current_strategy) in allowed:
        return str(current_strategy)
    recommended = str(resolve_recommended_strategy_for_size(philosophy, candidate_size) or "")
    if recommended in allowed:
        return recommended
    return allowed[0] if allowed else str(current_strategy or "Core Balanced")


def _make_candidate_spec(
    *,
    candidate_size: int,
    current_size: int,
    current_strategy: str,
    philosophy: str,
    stage: str,
) -> dict:
    """Build one size-candidate spec without evaluating it.

    Stage naming is intentionally product-facing:
    - coarse_canonical: official philosophy cap/checkpoint, e.g. 50 for Balanced;
    - coarse_probe: first-pass probe inside the philosophy band, e.g. 36/44;
    - local_refinement: second-pass probe around the best first-pass result.
    """
    size_value = max(1, _safe_int(candidate_size, 0))
    strategy_name = _resolve_strategy_for_size(philosophy, current_strategy, size_value)
    assets = _build_assets_for_size(size_value, strategy_name)

    if int(size_value) < int(current_size):
        direction = "Smaller"
    elif int(size_value) > int(current_size):
        direction = "Larger"
    else:
        direction = "Same-size"

    stage_name = str(stage or "coarse_probe")
    if stage_name == "local_refinement":
        label = f"Fine refinement universe ({int(size_value)} assets)"
        caption = (
            f"Second-pass test around the best coarse size result, using {int(size_value)} assets. "
            "The current strategy preset and technical engine settings are kept unchanged."
        )
    elif stage_name == "coarse_probe":
        label = f"Coarse probe universe ({int(size_value)} assets)"
        caption = (
            f"First-pass intermediate test inside the current philosophy-capped size band, using {int(size_value)} assets. "
            "The current strategy preset and technical engine settings are kept unchanged."
        )
    else:
        label = f"{direction} universe ({int(size_value)} assets)"
        caption = (
            f"Tests the Step 4 {strategy_name} universe at {int(size_value)} assets. "
            "The current strategy preset and technical engine settings are kept unchanged."
        )

    return {
        "label": label,
        "family": "universe_size",
        "stage": stage_name,
        "candidate_size": int(size_value),
        "strategy_name": str(strategy_name),
        "assets": list(assets),
        "asset_signature": _asset_signature(assets),
        "caption": caption,
    }


def _candidate_specs(
    *,
    current_size: int,
    current_strategy: str,
    philosophy: str,
    available_asset_count: int,
    max_engine_candidates: int = 3,
) -> list[dict]:
    """Return the first-pass, philosophy-capped coarse size candidates.

    The current size is not repeated here because it is already the baseline real
    Step 5 result. For Balanced at 25 with a 50-asset data pool, this usually
    returns 50 plus two broad probes around the upper half of the 25-50 band.
    """
    specs: list[dict] = []
    seen_sizes: set[int] = set()

    def _add(candidate_size: int, *, stage: str) -> None:
        size_value = max(1, _safe_int(candidate_size, 0))
        if size_value <= 0 or size_value == int(current_size) or size_value in seen_sizes:
            return
        seen_sizes.add(int(size_value))
        specs.append(
            _make_candidate_spec(
                candidate_size=int(size_value),
                current_size=int(current_size),
                current_strategy=str(current_strategy),
                philosophy=str(philosophy),
                stage=str(stage),
            )
        )

    _band_min, _band_max, preferred = _philosophy_size_band(philosophy)
    for size in preferred:
        if int(size) != int(current_size):
            _add(int(size), stage="coarse_canonical")

    for size in _local_refinement_sizes(
        current_size=int(current_size),
        philosophy=philosophy,
        max_available_assets=int(available_asset_count),
        max_refinements=int(max_engine_candidates),
    ):
        _add(int(size), stage="coarse_probe")

    return list(specs)


def _sort_evaluations_for_selection(evaluations: list[dict]) -> list[dict]:
    return sorted(
        [dict(_coerce_mapping(item)) for item in list(evaluations or [])],
        key=lambda item: (
            1 if bool(item.get("accepted")) else 0,
            _safe_float(item.get("score_delta"), -999.0),
            _safe_float(_coerce_mapping(item.get("performance_summary", {})).get("sharpe"), -999.0),
        ),
        reverse=True,
    )


def _fine_refinement_sizes(
    *,
    best_size: int,
    current_size: int,
    tested_sizes: list[int],
    philosophy: str,
    available_asset_count: int,
    max_refinements: int = 1,
) -> list[int]:
    """Return second-pass sizes around the best first-pass result.

    This is deliberately not a full grid search. It takes the best coarse result
    and tests one or two midpoints between that size and its nearest tested
    neighbours. Example: current=25, tested=[36, 44, 50], best=44 -> 40/47.
    """
    max_ref = max(0, _safe_int(max_refinements, 0))
    if max_ref <= 0:
        return []

    best = max(1, _safe_int(best_size, 0))
    current = max(1, _safe_int(current_size, 0))
    available = max(0, _safe_int(available_asset_count, 0))
    _band_min, band_max, _preferred = _philosophy_size_band(philosophy)
    upper_cap = min(int(band_max), int(available))
    if upper_cap <= 0:
        return []

    tested = sorted({int(x) for x in list(tested_sizes or []) if _safe_int(x, 0) > 0} | {int(current)})
    lower_neighbours = [x for x in tested if x < best]
    upper_neighbours = [x for x in tested if x > best]

    candidates: list[int] = []
    lower = max(lower_neighbours) if lower_neighbours else None
    upper = min(upper_neighbours) if upper_neighbours else None

    # Prefer the side with the largest gap first because it contains more
    # untested information. Then try the other side if the runtime cap allows.
    options: list[tuple[int, int]] = []
    if lower is not None and best - lower >= 2:
        mid = int(round((best + lower) / 2.0))
        if lower < mid < best:
            options.append((best - lower, mid))
    if upper is not None and upper - best >= 2:
        mid = int(round((best + upper) / 2.0))
        if best < mid < upper:
            options.append((upper - best, mid))

    options = sorted(options, key=lambda pair: pair[0], reverse=True)
    blocked = set(tested) | {int(current), int(best)}
    for _gap, value in options:
        size_value = max(1, min(int(value), int(upper_cap)))
        if size_value not in blocked and size_value not in candidates:
            candidates.append(size_value)
        if len(candidates) >= max_ref:
            break

    return candidates[:max_ref]


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
    panel_df = _get_search_panel(run_result)
    step4_payload = _coerce_mapping(build_step4_universe_payload_from_state())
    base_cfg = _base_cfg_payload_from_run(run_map)
    current_assets = _asset_list(step4_payload.get("selected_assets") or st.session_state.get(LAST_USED_UNIVERSE_ASSETS, []))
    current_size = _safe_int(step4_payload.get("size", st.session_state.get(UNIVERSE_SIZE, 25)), 25)
    current_strategy = str(step4_payload.get("strategy", st.session_state.get(UNIVERSE_STRATEGY, "")) or "")
    return _hash_scope(
        {
            "run_signature": str(run_map.get("run_signature", "") or ""),
            "config_fingerprint": str(run_map.get("config_fingerprint", "") or ""),
            "philosophy": _philosophy(),
            "technical_cfg": base_cfg,
            "current_universe_size": int(current_size),
            "current_universe_strategy": str(current_strategy),
            "current_assets": current_assets,
            "panel_assets": _panel_assets(panel_df),
            "panel_fp": _stable_panel_fingerprint(panel_df),
            "phase": "universe_size_v1",
        }
    )


def _run_size_search(run_result: dict, *, max_candidates: int = 4) -> dict:
    """Run a capped coarse-to-fine universe-size search.

    The current Step 5 run is the baseline, so the current size is not rerun.
    Stage 1 tests a tiny philosophy-capped coarse set. Stage 2 optionally tests
    one local midpoint around the best first-pass result, keeping runtime under
    control while still behaving like a real search instead of a one-shot grid.
    """
    run_map = _coerce_mapping(run_result)
    current_perf = _normalise_perf(run_map.get("performance_summary", {}))
    philosophy = _philosophy()
    base_cfg = _base_cfg_payload_from_run(run_map)

    panel_df = _get_search_panel(run_result)
    if panel_df.empty:
        return {"scope": _build_scope(run_result), "evaluations": [], "error": "Step 4 asset panel is missing."}

    step4_payload = _coerce_mapping(build_step4_universe_payload_from_state())
    current_size = _safe_int(step4_payload.get("size", st.session_state.get(UNIVERSE_SIZE, 25)), 25)
    current_strategy = str(step4_payload.get("strategy", st.session_state.get(UNIVERSE_STRATEGY, "")) or "")
    current_assets = _asset_list(step4_payload.get("selected_assets") or st.session_state.get(LAST_USED_UNIVERSE_ASSETS, []))
    available_assets = set(_panel_assets(panel_df))
    max_engine_tests = max(1, _safe_int(max_candidates, 4))

    evaluations: list[dict] = []
    engine_tests_used = 0
    started = time.perf_counter()

    def _evaluate_candidate(candidate: dict) -> dict:
        nonlocal engine_tests_used
        candidate_started = time.perf_counter()
        candidate_map = _coerce_mapping(candidate)
        candidate_size = _safe_int(candidate_map.get("candidate_size", 0), 0)
        strategy_name = str(candidate_map.get("strategy_name", current_strategy) or current_strategy)
        assets = _asset_list(candidate_map.get("assets", []))
        missing_assets = sorted(set(assets) - available_assets)
        candidate_panel = _filter_panel_to_assets(panel_df, assets)

        run: dict = {}
        perf: dict = {}
        accepted = False
        gate_reason = "Candidate could not be evaluated."
        error = ""
        test_status = "not_testable"

        if not assets:
            error = "No generated assets were available for this size."
        elif len(assets) != int(candidate_size):
            error = f"Generated asset count was {len(assets)} for requested size {candidate_size}."
        elif missing_assets:
            shown = ", ".join(missing_assets[:8])
            more = f" +{len(missing_assets) - 8} more" if len(missing_assets) > 8 else ""
            error = f"Missing assets in the current Step 4 panel: {shown}{more}."
        elif candidate_panel.empty:
            error = "Filtered candidate panel is empty."
        elif engine_tests_used >= int(max_engine_tests):
            error = "Runtime cap reached: this candidate was considered but not engine-tested."
        else:
            engine_tests_used += 1
            test_status = "engine_tested"
            run = _run_candidate(base_cfg, candidate_panel)
            perf = _normalise_perf(run.get("performance_summary", {}))
            error = str(run.get("error", "") or "")
            if error:
                test_status = "not_testable"
            if perf and not error:
                accepted, gate_reason = _acceptance_gate(perf, current_perf, philosophy)

        candidate_elapsed_sec = float(time.perf_counter() - candidate_started)
        score = _score_perf(perf, philosophy) if perf else None
        base_score = _score_perf(current_perf, philosophy) if current_perf else None
        return {
            "label": str(candidate_map.get("label", "Universe size candidate") or "Universe size candidate"),
            "family": str(candidate_map.get("family", "universe_size") or "universe_size"),
            "stage": str(candidate_map.get("stage", "") or ""),
            "test_status": str(test_status),
            "caption": str(candidate_map.get("caption", "") or ""),
            "current_universe_size": int(current_size),
            "universe_size": int(candidate_size),
            "universe_strategy": str(strategy_name),
            "assets": list(assets),
            "asset_signature": _asset_signature(assets),
            "missing_assets": list(missing_assets),
            "panel_fingerprint": _stable_panel_fingerprint(candidate_panel),
            "panel_rows": int(len(candidate_panel)) if isinstance(candidate_panel, pd.DataFrame) else 0,
            "panel_assets": int(candidate_panel["asset"].nunique()) if isinstance(candidate_panel, pd.DataFrame) and "asset" in candidate_panel.columns else 0,
            "cfg_payload": dict(base_cfg),
            "performance_summary": dict(perf),
            "raw_result": _coerce_mapping(run.get("raw", {})),
            "score": score,
            "score_delta": (float(score) - float(base_score)) if score is not None and base_score is not None else None,
            "accepted": bool(accepted),
            "gate_reason": gate_reason,
            "error": str(error or ""),
            "elapsed_sec": float(candidate_elapsed_sec),
        }

    # Stage 1: broad, philosophy-capped candidates.
    stage1_specs = _candidate_specs(
        current_size=current_size,
        current_strategy=current_strategy,
        philosophy=philosophy,
        available_asset_count=len(available_assets),
        max_engine_candidates=3,
    )
    seen_sizes: set[int] = set()
    for candidate in stage1_specs:
        candidate_size = _safe_int(_coerce_mapping(candidate).get("candidate_size", 0), 0)
        if candidate_size <= 0 or candidate_size in seen_sizes:
            continue
        seen_sizes.add(int(candidate_size))
        evaluations.append(_evaluate_candidate(candidate))

    # Stage 2: one local refinement around the best first-pass engine-tested size.
    engine_evaluated = [
        dict(_coerce_mapping(item))
        for item in evaluations
        if str(_coerce_mapping(item).get("test_status", "")) == "engine_tested"
        and _coerce_mapping(item).get("performance_summary")
    ]
    fine_refinement_run = False
    if engine_tests_used < int(max_engine_tests) and engine_evaluated:
        ranked_stage1 = _sort_evaluations_for_selection(engine_evaluated)
        best_stage1 = ranked_stage1[0]
        best_size = _safe_int(best_stage1.get("universe_size", current_size), current_size)
        tested_sizes = [_safe_int(item.get("universe_size", 0), 0) for item in engine_evaluated]
        remaining = int(max_engine_tests - engine_tests_used)
        fine_sizes = _fine_refinement_sizes(
            best_size=int(best_size),
            current_size=int(current_size),
            tested_sizes=tested_sizes,
            philosophy=philosophy,
            available_asset_count=len(available_assets),
            max_refinements=min(1, remaining),
        )
        for size_value in fine_sizes:
            if int(size_value) in seen_sizes:
                continue
            seen_sizes.add(int(size_value))
            fine_spec = _make_candidate_spec(
                candidate_size=int(size_value),
                current_size=int(current_size),
                current_strategy=str(current_strategy),
                philosophy=str(philosophy),
                stage="local_refinement",
            )
            evaluations.append(_evaluate_candidate(fine_spec))
            fine_refinement_run = True

    ranked = _sort_evaluations_for_selection(evaluations)
    best_accepted = next((item for item in ranked if bool(_coerce_mapping(item).get("accepted", False))), None)
    if best_accepted is not None:
        best_sig = str(_coerce_mapping(best_accepted).get("asset_signature", ""))
        evaluations = [best_accepted] + [item for item in evaluations if str(_coerce_mapping(item).get("asset_signature", "")) != best_sig]

    total_elapsed_sec = float(time.perf_counter() - started)
    accepted_count = int(sum(1 for item in evaluations if bool(_coerce_mapping(item).get("accepted", False))))
    engine_tested_count = int(sum(1 for item in evaluations if str(_coerce_mapping(item).get("test_status", "")) == "engine_tested"))
    not_testable_count = int(len(evaluations) - engine_tested_count)
    timing_rows = []
    accepted_seen = 0
    for item in evaluations:
        item_map = _coerce_mapping(item)
        accepted = bool(item_map.get("accepted", False))
        engine_tested = str(item_map.get("test_status", "")) == "engine_tested"
        accepted_seen += 1 if accepted else 0
        if not engine_tested:
            status = "not testable"
        elif accepted and accepted_seen == 1:
            status = "recommended"
        elif accepted:
            status = "passed gate"
        else:
            status = "not selected"
        timing_rows.append(
            {
                "candidate": str(item_map.get("label", "Candidate") or "Candidate"),
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
        "current_universe_size": int(current_size),
        "current_strategy": current_strategy,
        "current_assets": list(current_assets),
        "current_perf": dict(current_perf),
        "evaluations": evaluations,
        "candidate_count": int(len(evaluations)),
        "engine_tested_count": int(engine_tested_count),
        "not_testable_count": int(not_testable_count),
        "fine_refinement_run": bool(fine_refinement_run),
        "search_mode": "coarse_to_fine",
        "max_engine_tests": int(max_engine_tests),
        "timing_summary": {
            "scope": scope,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_seconds": round(total_elapsed_sec, 2),
            "candidate_count": int(engine_tested_count),
            "considered_count": int(len(evaluations)),
            "not_testable_count": int(not_testable_count),
            "accepted_count": accepted_count,
            "fine_refinement_run": bool(fine_refinement_run),
            "search_mode": "coarse_to_fine",
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
    source_panel = _get_search_panel()
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
            "promoted_from_universe_size_candidate": True,
            "universe_size_candidate_label": str(candidate_map.get("label", "Universe size candidate") or "Universe size candidate"),
            "asset_panel_source_label": "Step 5 recommended universe size",
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
            "search_eval_split": {"enabled": False, "reason": "universe_size_candidate_promoted_after_rerun_test"},
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
            "promoted_from_universe_size_candidate": True,
            "breakdown_df": pd.DataFrame(),
        },
    )
    return promoted


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

    source_panel = _get_search_panel()
    candidate_panel = _filter_panel_to_assets(source_panel, assets)
    recommendation_context = {
        "source": "step5_universe_size_suggestion",
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
        UNIVERSE_SIZE_RECOMMENDATION_CONTEXT_KEY: recommendation_context,
        ASSET_PANEL_DF: candidate_panel if isinstance(candidate_panel, pd.DataFrame) and not candidate_panel.empty else source_panel,
        "asset_panel_df": candidate_panel if isinstance(candidate_panel, pd.DataFrame) and not candidate_panel.empty else source_panel,
        ASSET_PANEL_READY: bool(isinstance(candidate_panel, pd.DataFrame) and not candidate_panel.empty),
        ASSET_PANEL_SOURCE_LABEL: "Step 5 recommended universe size",
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
                UNIVERSE_SIZE_SUGGESTION_STATE_KEY: {},
                UNIVERSE_SIZE_SUGGESTION_SCOPE_KEY: "",
                UNIVERSE_SIZE_APPLIED_SIGNATURE_KEY: run_signature,
                UNIVERSE_SIZE_APPLIED_LABEL_KEY: str(candidate_map.get("label", "Universe size candidate") or "Universe size candidate"),
                "step5_universe_size_apply_message_v1": (
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
                UNIVERSE_SIZE_APPLIED_SIGNATURE_KEY: "",
                UNIVERSE_SIZE_APPLIED_LABEL_KEY: "",
                "step5_universe_size_apply_message_v1": (
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
        UNIVERSE_SIZE_SKIPPED_SCOPE_KEY: str(scope or ""),
        UNIVERSE_SIZE_SKIPPED_LABEL_KEY: str(label or "recommended size"),
        UNIVERSE_SIZE_SKIPPED_RUN_SIGNATURE_KEY: str(run_signature or ""),
        "step5_universe_size_apply_message_v1": (
            f"Universe size kept unchanged for this run. Skipped recommendation: {label}."
        ),
        STEP5_SCROLL_TO_RESULT_AFTER_APPLY_KEY: True,
    }
    if callable(queue_and_rerun):
        queue_and_rerun(patch)
        return
    for key, value in patch.items():
        st.session_state[key] = value
    st.rerun()


def _display_size_stage(stage: Any) -> str:
    raw = str(stage or "").strip().lower()
    return {
        "coarse_canonical": "Philosophy cap check",
        "coarse_probe": "Coarse probe",
        "local_refinement": "Local refinement",
    }.get(raw, str(stage or ""))


def _candidate_table(evaluations: list[dict], current_perf: dict) -> pd.DataFrame:
    rows: list[dict] = []
    base = _normalise_perf(current_perf)
    accepted_seen = 0
    for raw in evaluations:
        item = _coerce_mapping(raw)
        perf = _normalise_perf(item.get("performance_summary", {}))
        accepted = bool(item.get("accepted", False))
        accepted_seen += 1 if accepted else 0
        engine_tested = str(item.get("test_status", "")) == "engine_tested"
        if not engine_tested:
            status = "Not testable"
        else:
            status = "Recommended" if accepted and accepted_seen == 1 else ("Passed gate" if accepted else "Not selected")
        vol_delta = perf.get("annual_volatility", 0.0) - base.get("annual_volatility", 0.0)
        maxdd_improvement = base.get("max_drawdown", 0.0) - perf.get("max_drawdown", 0.0)
        rows.append(
            {
                "candidate": str(item.get("label", "Candidate") or "Candidate"),
                "status": status,
                "stage": _display_size_stage(item.get("stage", "")),
                "size": _safe_int(item.get("universe_size", 0), 0),
                "strategy": str(item.get("universe_strategy", "") or ""),
                "assets": int(len(_asset_list(item.get("assets", [])))),
                "panel assets": _safe_int(item.get("panel_assets", 0), 0),
                "missing": int(len(list(item.get("missing_assets", []) or []))),
                "CAGR": _format_pct(perf.get("cagr", 0.0)),
                "Δ CAGR": _format_delta_pct(perf.get("cagr", 0.0) - base.get("cagr", 0.0)),
                "Vol": _format_pct(perf.get("annual_volatility", 0.0)),
                "Δ Vol": _format_delta_pct(vol_delta),
                "MaxDD": f"-{100.0 * abs(perf.get('max_drawdown', 0.0)):.2f}%",
                "DD improvement": _format_delta_pct(maxdd_improvement),
                "Sharpe": f"{perf.get('sharpe', 0.0):.2f}",
                "Δ Sharpe": f"{perf.get('sharpe', 0.0) - base.get('sharpe', 0.0):+.2f}",
                "score delta": f"{_safe_float(item.get('score_delta'), 0.0):+.3f}",
                "seconds": f"{_safe_float(item.get('elapsed_sec', 0.0), 0.0):.2f}s",
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

    st.markdown("### Recommended universe size")
    st.markdown(f"**{item.get('label', 'Universe size candidate')}**")
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

    st.success("This candidate passed the universe-size acceptance gate.")
    gate_reason = str(item.get("gate_reason", "") or "")
    if gate_reason:
        st.caption(gate_reason)
    if item.get("error"):
        st.warning(str(item.get("error")))

    with st.expander("Recommended size assets", expanded=False):
        if assets:
            preview = ", ".join(asset_display_label(x) for x in assets[:40])
            if len(assets) > 40:
                preview += f" ... +{len(assets) - 40} more"
            st.write(preview)
        st.caption(
            f"size={item.get('universe_size', '—')} · strategy={item.get('universe_strategy', '—')} · "
            f"score_delta={_safe_float(item.get('score_delta'), 0.0):+.3f}"
        )


def render_universe_size_improvement(run_result: dict) -> None:
    """Render the fourth Step 5 improvement phase: universe size."""
    run_map = _coerce_mapping(run_result)
    perf = _normalise_perf(run_map.get("performance_summary", {}))
    if not perf:
        return

    st.markdown("### Universe size suggestion")
    st.caption(
        "This keeps the current strategy preset and technical engine configuration, then runs a capped coarse-to-fine universe-size search. It first tests a small set of philosophy-compatible sizes, then uses one local refinement around the best first-pass result when possible."
    )

    msg = st.session_state.pop("step5_universe_size_apply_message_v1", "")

    current_run_signature = str(run_map.get("run_signature", "") or "")
    applied_signature = str(st.session_state.get(UNIVERSE_SIZE_APPLIED_SIGNATURE_KEY, "") or "")
    applied_label = str(st.session_state.get(UNIVERSE_SIZE_APPLIED_LABEL_KEY, "") or "")
    if applied_signature and current_run_signature and applied_signature == current_run_signature:
        label = applied_label or "the accepted universe size suggestion"
        st.success(
            f"Universe size applied: {label}. The rerun-tested candidate is now the current Step 5 result."
        )
        st.caption(
            "Universe-size testing is hidden for this run to avoid suggesting the same loop again. "
            "The improvement flow is complete."
        )
        return

    if msg:
        st.success(str(msg))

    scope = _build_scope(run_map)
    saved_scope = str(st.session_state.get(UNIVERSE_SIZE_SUGGESTION_SCOPE_KEY, "") or "")
    payload = _coerce_mapping(st.session_state.get(UNIVERSE_SIZE_SUGGESTION_STATE_KEY, {}))
    evaluations = list(payload.get("evaluations", []) or []) if saved_scope == scope else []

    if saved_scope != scope or not evaluations:
        with st.spinner("Testing alternative universe sizes with the real engine..."):
            payload = _run_size_search(run_map, max_candidates=4)
        st.session_state[UNIVERSE_SIZE_SUGGESTION_STATE_KEY] = payload
        st.session_state[UNIVERSE_SIZE_SUGGESTION_SCOPE_KEY] = str(payload.get("scope", scope))
        st.session_state[UNIVERSE_SIZE_SUGGESTION_TIMING_KEY] = _coerce_mapping(payload.get("timing_summary", {}))
        evaluations = list(payload.get("evaluations", []) or [])

    if not evaluations:
        error = str(payload.get("error", "") or "")
        if error:
            st.info(f"No safe universe-size candidates were available to test for this run. {error}")
        else:
            st.info("No safe universe-size candidates were available to test for this run.")
        if st.button("Finish with current size", key="step5_finish_current_size_no_candidates_v1", use_container_width=True):
            _skip_current_size_candidate(scope, "no available size candidate", current_run_signature)
        return

    engine_tested_count = int(payload.get("engine_tested_count", sum(1 for x in evaluations if str(_coerce_mapping(x).get("test_status", "")) == "engine_tested")))
    not_testable_count = int(payload.get("not_testable_count", sum(1 for x in evaluations if str(_coerce_mapping(x).get("test_status", "")) != "engine_tested")))
    if not_testable_count > 0:
        st.info(
            f"Size search used the current Step 4 data pool: {engine_tested_count} candidate(s) were engine-tested and {not_testable_count} candidate(s) could not be tested with the available panel."
        )

    accepted_items = [dict(x) for x in evaluations if bool(_coerce_mapping(x).get("accepted", False))]
    table = _candidate_table(evaluations, perf)
    skipped_scope = str(st.session_state.get(UNIVERSE_SIZE_SKIPPED_SCOPE_KEY, "") or "")
    skipped_label = str(st.session_state.get(UNIVERSE_SIZE_SKIPPED_LABEL_KEY, "") or "")
    skipped_run_signature = str(st.session_state.get(UNIVERSE_SIZE_SKIPPED_RUN_SIGNATURE_KEY, "") or "")
    size_was_skipped = bool(
        skipped_scope
        and skipped_scope == scope
        and (not skipped_run_signature or skipped_run_signature == current_run_signature)
    )

    if not accepted_items:
        st.success("Current universe size remains best within the philosophy-capped Step 4 data pool: no tested alternative materially improved this run.")
        st.caption(
            "The tested size candidates are kept below for transparency, but none is offered as an Apply action because "
            "the acceptance gate did not find a better trade-off."
        )
    elif size_was_skipped:
        st.info("Current universe size kept for this run. The improvement flow is complete.")
        if skipped_label:
            st.caption(f"Skipped universe-size recommendation: {skipped_label}.")
    else:
        best_candidate = accepted_items[0]
        st.success("Recommended universe size found. The best accepted candidate is shown below.")
        _render_recommended_candidate(best_candidate, perf)
        st.caption(
            "The diagnostics table shows every size candidate tested by the engine. Only the best accepted candidate "
            "is offered as the main action; rejected candidates are shown for transparency, not as recommendations."
        )

    with st.expander("Universe size diagnostics", expanded=False):
        st.caption(
            f"mode={str(payload.get('search_mode', 'coarse_to_fine'))} · "
            f"engine_tested={engine_tested_count} · considered={len(evaluations)} · not_testable={not_testable_count} · "
            f"fine_refinement_run={bool(payload.get('fine_refinement_run', False))} · "
            f"elapsed={_safe_float(payload.get('elapsed_sec', 0.0), 0.0):.2f}s · scope={str(payload.get('scope', scope))}"
        )
        if not table.empty:
            st.dataframe(table, use_container_width=True, hide_index=True)

    if accepted_items and not size_was_skipped:
        best_candidate = accepted_items[0]
        left, right = st.columns(2)
        with left:
            if st.button("Apply recommended size", key="step5_apply_best_universe_size_candidate_v1", use_container_width=True):
                _apply_candidate(best_candidate)
        with right:
            if st.button("Keep current size and finish", key="step5_skip_best_universe_size_candidate_v1", use_container_width=True):
                _skip_current_size_candidate(
                    scope,
                    str(best_candidate.get("label", "recommended size") or "recommended size"),
                    current_run_signature,
                )
    elif not accepted_items and not size_was_skipped:
        if st.button("Keep current size and finish", key="step5_finish_current_size_no_acceptance_v1", use_container_width=True):
            _skip_current_size_candidate(scope, "no accepted universe-size candidate", current_run_signature)


# Compatibility wrapper for older/shorter imports.
def render_size_recommendations(run_result: dict | None = None, *args: Any, **kwargs: Any) -> None:
    if isinstance(run_result, dict):
        render_universe_size_improvement(run_result)
    return None


# Preferred short name for Phase 4 imports.
def render_size_improvement(run_result: dict | None = None, *args: Any, **kwargs: Any) -> None:
    if isinstance(run_result, dict):
        render_universe_size_improvement(run_result)
    return None
