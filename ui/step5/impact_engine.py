from __future__ import annotations

import copy
from dataclasses import fields
from typing import Any

import pandas as pd

from src.investment import MicroPipelineConfig, run_micro_investment_pipeline


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


def _coerce_cfg(cfg_payload: dict) -> MicroPipelineConfig:
    valid = {f.name for f in fields(MicroPipelineConfig)}
    payload = {k: v for k, v in dict(cfg_payload or {}).items() if k in valid}
    return MicroPipelineConfig(**payload)


def _run_metrics(cfg_payload: dict, asset_panel_df: pd.DataFrame) -> dict | None:
    if asset_panel_df is None or not isinstance(asset_panel_df, pd.DataFrame) or asset_panel_df.empty:
        return None
    try:
        cfg_obj = _coerce_cfg(cfg_payload)
        result = run_micro_investment_pipeline(asset_panel_df, cfg=cfg_obj)
        perf = _coerce_mapping(_coerce_mapping(result).get("performance_summary", {}))
        return {
            "sharpe": _safe_float(perf.get("sharpe", 0.0), 0.0),
            "cagr": _safe_float(perf.get("cagr", 0.0), 0.0),
            "max_drawdown": _safe_float(perf.get("max_drawdown", 0.0), 0.0),
        }
    except Exception:
        return None


def compute_local_impact(
    cfg_base: dict,
    overrides: dict,
    asset_panel_df: pd.DataFrame,
    *,
    delta_scale: float = 0.10,
    max_params: int = 3,
) -> pd.DataFrame:
    """
    Estimate REAL local impact of changed numeric overrides via small +/- reruns.

    For each changed numeric parameter:
    - perturb down
    - perturb up
    - rerun engine
    - measure marginal change vs base metrics
    """
    base_cfg = dict(cfg_base or {})
    override_map = dict(overrides or {})
    if not base_cfg or asset_panel_df is None or not isinstance(asset_panel_df, pd.DataFrame) or asset_panel_df.empty:
        return pd.DataFrame()

    changed_numeric: list[tuple[str, float, float]] = []
    for key, new_value in override_map.items():
        if not isinstance(new_value, (int, float)) or isinstance(new_value, bool):
            continue
        old_value = base_cfg.get(key)
        try:
            old_num = float(old_value)
            new_num = float(new_value)
        except Exception:
            continue
        if old_num == new_num:
            continue
        changed_numeric.append((str(key), old_num, new_num))

    changed_numeric = changed_numeric[: max(1, int(max_params))]
    if not changed_numeric:
        return pd.DataFrame()

    base_metrics = _run_metrics(base_cfg, asset_panel_df)
    if base_metrics is None:
        return pd.DataFrame()

    rows: list[dict] = []
    for param, old_value, new_value in changed_numeric:
        delta = max(abs(new_value) * float(delta_scale), 1e-3)
        for direction in (-1.0, 1.0):
            perturbed_value = float(new_value + direction * delta)
            cfg_test = copy.deepcopy(base_cfg)
            cfg_test[param] = perturbed_value
            metrics = _run_metrics(cfg_test, asset_panel_df)
            if not metrics:
                continue
            rows.append(
                {
                    "parameter": param,
                    "base": old_value,
                    "override": new_value,
                    "test_value": perturbed_value,
                    "direction": "up" if direction > 0 else "down",
                    "delta": float(direction * delta),
                    "sharpe_impact": float(metrics["sharpe"] - base_metrics["sharpe"]),
                    "cagr_impact": float(metrics["cagr"] - base_metrics["cagr"]),
                    "drawdown_impact": float(metrics["max_drawdown"] - base_metrics["max_drawdown"]),
                }
            )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    sort_cols = [c for c in ["parameter", "direction"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def _composite_local_score(metrics: dict | None, *, profile: str = "balanced") -> float:
    metrics = dict(metrics or {})
    sharpe = _safe_float(metrics.get("sharpe", 0.0), 0.0)
    cagr = _safe_float(metrics.get("cagr", 0.0), 0.0)
    maxdd = abs(_safe_float(metrics.get("max_drawdown", 0.0), 0.0))
    profile_key = str(profile or "balanced").strip().lower()
    if profile_key == "growth":
        return float((1.00 * sharpe) + (0.75 * cagr) - (0.60 * maxdd))
    if profile_key in {"defensive", "conservative"}:
        return float((1.10 * sharpe) + (0.30 * cagr) - (1.10 * maxdd))
    return float((1.00 * sharpe) + (0.50 * cagr) - (0.75 * maxdd))


def compute_gradient_map(
    impact_df: pd.DataFrame,
    *,
    min_abs_signal: float = 1e-6,
) -> pd.DataFrame:
    if impact_df is None or not isinstance(impact_df, pd.DataFrame) or impact_df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    work = impact_df.copy()
    for param, g in work.groupby("parameter", dropna=False):
        down = g.loc[g["direction"].astype(str).str.lower() == "down"]
        up = g.loc[g["direction"].astype(str).str.lower() == "up"]

        def _metric(df: pd.DataFrame, col: str) -> float:
            if df.empty or col not in df.columns:
                return 0.0
            return _safe_float(df[col].mean(), 0.0)

        sharpe_up = _metric(up, "sharpe_impact")
        sharpe_down = _metric(down, "sharpe_impact")
        cagr_up = _metric(up, "cagr_impact")
        cagr_down = _metric(down, "cagr_impact")
        dd_up = _metric(up, "drawdown_impact")
        dd_down = _metric(down, "drawdown_impact")

        up_score = sharpe_up + (0.50 * cagr_up) - (0.75 * dd_up)
        down_score = sharpe_down + (0.50 * cagr_down) - (0.75 * dd_down)

        if abs(up_score - down_score) < float(min_abs_signal):
            direction = "flat"
        else:
            direction = "up" if up_score > down_score else "down"

        confidence = abs(up_score - down_score)
        rows.append(
            {
                "parameter": str(param),
                "recommended_direction": direction,
                "up_score": float(up_score),
                "down_score": float(down_score),
                "confidence": float(confidence),
                "best_local_signal": float(max(up_score, down_score)),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["confidence", "parameter"], ascending=[False, True]).reset_index(drop=True)


def compute_local_autotune(
    cfg_base: dict,
    gradient_df: pd.DataFrame,
    asset_panel_df: pd.DataFrame,
    *,
    profile: str = "balanced",
    max_params: int = 3,
    step_scale: float = 0.10,
    max_trials: int = 8,
) -> dict:
    base_cfg = dict(cfg_base or {})
    if not base_cfg or asset_panel_df is None or not isinstance(asset_panel_df, pd.DataFrame) or asset_panel_df.empty:
        return {
            "baseline_metrics": {},
            "baseline_score": 0.0,
            "candidates_df": pd.DataFrame(),
            "best_candidate_cfg": {},
            "best_candidate_metrics": {},
            "best_candidate_score": 0.0,
            "best_delta_score": 0.0,
        }

    baseline_metrics = _run_metrics(base_cfg, asset_panel_df)
    if not baseline_metrics:
        return {
            "baseline_metrics": {},
            "baseline_score": 0.0,
            "candidates_df": pd.DataFrame(),
            "best_candidate_cfg": {},
            "best_candidate_metrics": {},
            "best_candidate_score": 0.0,
            "best_delta_score": 0.0,
        }

    baseline_score = _composite_local_score(baseline_metrics, profile=profile)

    if gradient_df is None or not isinstance(gradient_df, pd.DataFrame) or gradient_df.empty:
        return {
            "baseline_metrics": baseline_metrics,
            "baseline_score": baseline_score,
            "candidates_df": pd.DataFrame(),
            "best_candidate_cfg": dict(base_cfg),
            "best_candidate_metrics": baseline_metrics,
            "best_candidate_score": baseline_score,
            "best_delta_score": 0.0,
        }

    selected = gradient_df.copy().head(max(1, int(max_params)))
    candidate_rows: list[dict] = []
    best_cfg = dict(base_cfg)
    best_metrics = dict(baseline_metrics)
    best_score = float(baseline_score)

    trials_done = 0
    for _, row in selected.iterrows():
        if trials_done >= int(max_trials):
            break
        param = str(row.get("parameter", "") or "")
        direction = str(row.get("recommended_direction", "flat") or "flat").lower()
        if not param or direction == "flat":
            continue
        try:
            base_value = float(base_cfg.get(param))
        except Exception:
            continue
        delta = max(abs(base_value) * float(step_scale), 1e-3)
        signed_delta = delta if direction == "up" else -delta

        candidate_cfg = copy.deepcopy(best_cfg)
        candidate_cfg[param] = float(base_value + signed_delta)

        metrics = _run_metrics(candidate_cfg, asset_panel_df)
        trials_done += 1
        if not metrics:
            continue

        score = _composite_local_score(metrics, profile=profile)
        delta_score = float(score - baseline_score)
        candidate_rows.append(
            {
                "parameter": param,
                "direction": direction,
                "base_value": base_value,
                "candidate_value": candidate_cfg[param],
                "score": float(score),
                "delta_score": delta_score,
                "sharpe": _safe_float(metrics.get("sharpe", 0.0), 0.0),
                "cagr": _safe_float(metrics.get("cagr", 0.0), 0.0),
                "max_drawdown": _safe_float(metrics.get("max_drawdown", 0.0), 0.0),
            }
        )

        if score > best_score:
            best_score = float(score)
            best_cfg = dict(candidate_cfg)
            best_metrics = dict(metrics)

    candidates_df = pd.DataFrame(candidate_rows)
    if not candidates_df.empty:
        candidates_df = candidates_df.sort_values(["delta_score", "score"], ascending=[False, False]).reset_index(drop=True)

    return {
        "baseline_metrics": baseline_metrics,
        "baseline_score": float(baseline_score),
        "candidates_df": candidates_df,
        "best_candidate_cfg": best_cfg,
        "best_candidate_metrics": best_metrics,
        "best_candidate_score": float(best_score),
        "best_delta_score": float(best_score - baseline_score),
    }


def _dominates(a: dict, b: dict) -> bool:
    # Objectives: sharpe ↑, cagr ↑, max_drawdown ↓
    a_sh = _safe_float(a.get("sharpe", 0.0), 0.0)
    a_cg = _safe_float(a.get("cagr", 0.0), 0.0)
    a_dd = _safe_float(a.get("max_drawdown", 0.0), 0.0)
    b_sh = _safe_float(b.get("sharpe", 0.0), 0.0)
    b_cg = _safe_float(b.get("cagr", 0.0), 0.0)
    b_dd = _safe_float(b.get("max_drawdown", 0.0), 0.0)

    no_worse = (a_sh >= b_sh) and (a_cg >= b_cg) and (a_dd <= b_dd)
    strictly_better = (a_sh > b_sh) or (a_cg > b_cg) or (a_dd < b_dd)
    return bool(no_worse and strictly_better)


def compute_local_pareto(
    cfg_base: dict,
    gradient_df: pd.DataFrame,
    asset_panel_df: pd.DataFrame,
    *,
    max_params: int = 3,
    step_scale: float = 0.10,
    max_candidates: int = 12,
) -> dict:
    """
    Build a small local Pareto frontier around the current config.
    Uses gradient-informed directional candidates and evaluates:
      - sharpe (maximize)
      - cagr (maximize)
      - max_drawdown (minimize)
    """
    base_cfg = dict(cfg_base or {})
    if not base_cfg or asset_panel_df is None or not isinstance(asset_panel_df, pd.DataFrame) or asset_panel_df.empty:
        return {
            "baseline_metrics": {},
            "candidates_df": pd.DataFrame(),
            "frontier_df": pd.DataFrame(),
            "frontier_candidates": [],
        }

    baseline_metrics = _run_metrics(base_cfg, asset_panel_df)
    if not baseline_metrics:
        return {
            "baseline_metrics": {},
            "candidates_df": pd.DataFrame(),
            "frontier_df": pd.DataFrame(),
            "frontier_candidates": [],
        }

    if gradient_df is None or not isinstance(gradient_df, pd.DataFrame) or gradient_df.empty:
        return {
            "baseline_metrics": baseline_metrics,
            "candidates_df": pd.DataFrame(),
            "frontier_df": pd.DataFrame(),
            "frontier_candidates": [],
        }

    selected = gradient_df.copy().head(max(1, int(max_params)))
    candidate_rows = []
    candidate_cfgs = []
    candidate_id = 1

    # baseline candidate
    candidate_rows.append({
        "candidate_id": "base",
        "parameter": "base",
        "direction": "base",
        "candidate_value": None,
        "sharpe": _safe_float(baseline_metrics.get("sharpe", 0.0), 0.0),
        "cagr": _safe_float(baseline_metrics.get("cagr", 0.0), 0.0),
        "max_drawdown": _safe_float(baseline_metrics.get("max_drawdown", 0.0), 0.0),
    })
    candidate_cfgs.append({"candidate_id": "base", "cfg": dict(base_cfg)})

    # single-parameter candidates around gradient direction
    trials = 1
    for _, row in selected.iterrows():
        if trials >= int(max_candidates):
            break
        param = str(row.get("parameter", "") or "")
        direction = str(row.get("recommended_direction", "flat") or "flat").lower()
        if not param or direction == "flat":
            continue
        try:
            base_value = float(base_cfg.get(param))
        except Exception:
            continue

        signed = 1.0 if direction == "up" else -1.0
        # produce two local candidates: 1x step and 2x step
        for mult in (1.0, 2.0):
            if trials >= int(max_candidates):
                break
            delta = max(abs(base_value) * float(step_scale) * float(mult), 1e-3)
            cfg = copy.deepcopy(base_cfg)
            cfg[param] = float(base_value + signed * delta)
            metrics = _run_metrics(cfg, asset_panel_df)
            trials += 1
            if not metrics:
                continue
            cid = f"cand_{candidate_id}"
            candidate_id += 1
            candidate_rows.append({
                "candidate_id": cid,
                "parameter": param,
                "direction": direction,
                "candidate_value": cfg[param],
                "sharpe": _safe_float(metrics.get("sharpe", 0.0), 0.0),
                "cagr": _safe_float(metrics.get("cagr", 0.0), 0.0),
                "max_drawdown": _safe_float(metrics.get("max_drawdown", 0.0), 0.0),
            })
            candidate_cfgs.append({"candidate_id": cid, "cfg": cfg})

    candidates_df = pd.DataFrame(candidate_rows)
    if candidates_df.empty:
        return {
            "baseline_metrics": baseline_metrics,
            "candidates_df": pd.DataFrame(),
            "frontier_df": pd.DataFrame(),
            "frontier_candidates": [],
        }

    # Pareto filter
    rows = candidates_df.to_dict(orient="records")
    frontier_ids = []
    for i, a in enumerate(rows):
        dominated = False
        for j, b in enumerate(rows):
            if i == j:
                continue
            if _dominates(b, a):
                dominated = True
                break
        if not dominated:
            frontier_ids.append(a["candidate_id"])

    candidates_df["is_frontier"] = candidates_df["candidate_id"].isin(frontier_ids)
    frontier_df = candidates_df.loc[candidates_df["is_frontier"]].copy().reset_index(drop=True)

    frontier_candidates = []
    frontier_id_set = set(frontier_ids)
    for item in candidate_cfgs:
        if item["candidate_id"] in frontier_id_set:
            frontier_candidates.append(item)

    return {
        "baseline_metrics": baseline_metrics,
        "candidates_df": candidates_df,
        "frontier_df": frontier_df,
        "frontier_candidates": frontier_candidates,
    }
