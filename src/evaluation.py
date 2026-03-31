from __future__ import annotations

"""
src/evaluation.py

Research / evaluation layer for cross-sectional signal assessment.

Purpose
-------
This module is intentionally separate from feature engineering (features.py)
and portfolio construction / walk-forward execution (investment.py).

It provides tools to evaluate whether a signal contains useful cross-sectional
information before, or alongside, translating it into allocation decisions.

Main capabilities
-----------------
- MAE / MSE style forecast evaluation
- Rank IC by date
- Spearman IC by date
- Kendall IC by date
- rolling IC summaries
- IC by asset (time-series predictive relationship)
- IC by regime
- ranking vs error comparison tables across multiple signal columns

Expected input format
---------------------
Most functions work with a long panel DataFrame containing at least:
- date
- asset
- current signal column(s)
- realised next-period return target, or a current return column from which the
  forward return can be built via per-asset shift

The module is frequency-agnostic: monthly, weekly, etc.
"""

from dataclasses import dataclass, asdict
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Dataclasses
# ============================================================

@dataclass(frozen=True)
class ICSummary:
    metric: str
    n_dates: int
    mean_ic: float
    std_ic: float
    ir: float
    hit_rate: float
    positive_rate: float
    t_stat: float
    median_ic: float
    q10_ic: float
    q90_ic: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProbabilisticSummary:
    n_obs: int
    coverage: float
    nominal_coverage: float
    coverage_error: float
    avg_interval_width: float
    median_interval_width: float
    avg_pinball_low: float
    avg_pinball_mid: float
    avg_pinball_high: float
    avg_interval_score: float
    avg_calibration_gap: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# Small helpers
# ============================================================


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan



def _coerce_panel(
    panel: pd.DataFrame,
    *,
    date_col: str,
    asset_col: str,
) -> pd.DataFrame:
    if panel is None or not isinstance(panel, pd.DataFrame) or panel.empty:
        raise ValueError("panel must be a non-empty DataFrame")
    missing = [c for c in [date_col, asset_col] if c not in panel.columns]
    if missing:
        raise KeyError(f"panel missing required columns: {missing}")
    out = panel.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[asset_col] = out[asset_col].astype(str)
    out = out.dropna(subset=[date_col, asset_col]).sort_values([date_col, asset_col]).reset_index(drop=True)
    if out.empty:
        raise ValueError("panel is empty after cleaning date/asset columns")
    return out



def add_forward_return_column(
    panel: pd.DataFrame,
    *,
    return_col: str = "return",
    date_col: str = "date",
    asset_col: str = "asset",
    horizon: int = 1,
    forward_return_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a next-period return column via per-asset shift.

    Example:
    - monthly panel with return_t -> future_return_1 = return_{t+1}
    - weekly panel with return_t -> future_return_1 = return_{t+1}
    """
    if int(horizon) <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if return_col not in panel.columns:
        raise KeyError(f"return column not found: {return_col}")

    out = _coerce_panel(panel, date_col=date_col, asset_col=asset_col)
    col = forward_return_col or f"future_return_{int(horizon)}"
    out[return_col] = pd.to_numeric(out[return_col], errors="coerce")
    out[col] = out.groupby(asset_col, sort=False)[return_col].shift(-int(horizon))
    return out



def _rank_series(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").rank(method="average", pct=True)



def _corr_pair(x: pd.Series, y: pd.Series, *, method: str) -> float:
    df = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if df.shape[0] < 2:
        return np.nan
    if method == "rank_ic":
        xr = _rank_series(df["x"])
        yr = _rank_series(df["y"])
        return float(xr.corr(yr, method="pearson"))
    if method in {"pearson", "spearman", "kendall"}:
        return float(df["x"].corr(df["y"], method=method))
    raise ValueError(f"Unsupported method: {method}")



def _summarize_ic(series: pd.Series, *, metric: str) -> ICSummary:
    xs = pd.to_numeric(series, errors="coerce").dropna()
    n = int(xs.shape[0])
    if n == 0:
        return ICSummary(
            metric=metric,
            n_dates=0,
            mean_ic=np.nan,
            std_ic=np.nan,
            ir=np.nan,
            hit_rate=np.nan,
            positive_rate=np.nan,
            t_stat=np.nan,
            median_ic=np.nan,
            q10_ic=np.nan,
            q90_ic=np.nan,
        )
    mean_ic = float(xs.mean())
    std_ic = float(xs.std(ddof=1)) if n > 1 else 0.0
    ir = float(mean_ic / std_ic) if std_ic > 1e-12 else np.nan
    t_stat = float(mean_ic / (std_ic / np.sqrt(n))) if std_ic > 1e-12 and n > 1 else np.nan
    hit_rate = float((xs > 0).mean())
    return ICSummary(
        metric=metric,
        n_dates=n,
        mean_ic=mean_ic,
        std_ic=std_ic,
        ir=ir,
        hit_rate=hit_rate,
        positive_rate=hit_rate,
        t_stat=t_stat,
        median_ic=float(xs.median()),
        q10_ic=float(xs.quantile(0.10)),
        q90_ic=float(xs.quantile(0.90)),
    )



def _validate_signal_and_target(panel: pd.DataFrame, signal_col: str, target_col: str) -> None:
    missing = [c for c in [signal_col, target_col] if c not in panel.columns]
    if missing:
        raise KeyError(f"panel missing required columns: {missing}")


# ============================================================
# Point forecast error metrics
# ============================================================


def compute_error_metrics(
    panel: pd.DataFrame,
    *,
    prediction_col: str,
    target_col: str,
    by: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute MAE / MSE / RMSE / bias.

    If `by` is provided (e.g. date or asset), returns one row per group.
    Otherwise returns a single-row DataFrame.
    """
    _validate_signal_and_target(panel, prediction_col, target_col)
    df = panel[[c for c in [by, prediction_col, target_col] if c is not None]].copy()
    df[prediction_col] = pd.to_numeric(df[prediction_col], errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[prediction_col, target_col])
    if df.empty:
        return pd.DataFrame(columns=[c for c in [by, "n", "mae", "mse", "rmse", "bias"] if c is not None])

    def _agg(g: pd.DataFrame) -> pd.Series:
        err = pd.to_numeric(g[prediction_col], errors="coerce") - pd.to_numeric(g[target_col], errors="coerce")
        mse = float((err ** 2).mean())
        return pd.Series({
            "n": int(err.shape[0]),
            "mae": float(err.abs().mean()),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "bias": float(err.mean()),
        })

    if by is None:
        return _agg(df).to_frame().T.reset_index(drop=True)
    out = df.groupby(by, dropna=False).apply(_agg).reset_index()
    return out




# ============================================================
# Probabilistic forecast evaluation
# ============================================================


def _pinball_loss(y_true: pd.Series, y_pred: pd.Series, q: float) -> pd.Series:
    qq = float(np.clip(q, 1e-6, 1.0 - 1e-6))
    err = pd.to_numeric(y_true, errors="coerce") - pd.to_numeric(y_pred, errors="coerce")
    return pd.Series(np.maximum(qq * err, (qq - 1.0) * err), index=err.index, dtype="float64")


def compute_pinball_loss(
    panel: pd.DataFrame,
    *,
    prediction_col: str,
    target_col: str,
    q: float,
    by: Optional[str] = None,
) -> pd.DataFrame:
    _validate_signal_and_target(panel, prediction_col, target_col)
    df = panel[[c for c in [by, prediction_col, target_col] if c is not None]].copy()
    df[prediction_col] = pd.to_numeric(df[prediction_col], errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[prediction_col, target_col])
    if df.empty:
        return pd.DataFrame(columns=[c for c in [by, "n", "pinball_loss"] if c is not None])

    def _agg(g: pd.DataFrame) -> pd.Series:
        pl = _pinball_loss(g[target_col], g[prediction_col], q=q)
        return pd.Series({"n": int(pl.shape[0]), "pinball_loss": float(pl.mean())})

    if by is None:
        return _agg(df).to_frame().T.reset_index(drop=True)
    return df.groupby(by, dropna=False).apply(_agg).reset_index()


def compute_interval_metrics(
    panel: pd.DataFrame,
    *,
    target_col: str,
    pred_low_col: str,
    pred_mid_col: Optional[str] = None,
    pred_high_col: str,
    q_low: float = 0.10,
    q_high: float = 0.90,
    by: Optional[str] = None,
) -> pd.DataFrame:
    need = [target_col, pred_low_col, pred_high_col] + ([pred_mid_col] if pred_mid_col is not None else [])
    missing = [c for c in need if c not in panel.columns]
    if missing:
        raise KeyError(f"panel missing required columns: {missing}")

    cols = [c for c in [by, target_col, pred_low_col, pred_mid_col, pred_high_col] if c is not None]
    df = panel[cols].copy()
    for c in [target_col, pred_low_col, pred_high_col] + ([pred_mid_col] if pred_mid_col is not None else []):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[target_col, pred_low_col, pred_high_col])
    if df.empty:
        return pd.DataFrame(columns=[c for c in [by, "n", "coverage", "nominal_coverage", "coverage_error", "avg_interval_width", "median_interval_width", "avg_pinball_low", "avg_pinball_mid", "avg_pinball_high", "avg_interval_score", "avg_calibration_gap"] if c is not None])

    nominal = float(np.clip(q_high, 0.0, 1.0) - np.clip(q_low, 0.0, 1.0))

    def _agg(g: pd.DataFrame) -> pd.Series:
        y = g[target_col]
        lo = g[pred_low_col]
        hi = g[pred_high_col]
        width = hi - lo
        covered = ((y >= lo) & (y <= hi)).astype(float)
        pl_lo = _pinball_loss(y, lo, q=q_low)
        pl_hi = _pinball_loss(y, hi, q=q_high)
        if pred_mid_col is not None:
            mid = g[pred_mid_col]
            pl_mid = _pinball_loss(y, mid, q=0.50)
            cal_gap = (mid - y).abs()
        else:
            pl_mid = pd.Series(np.nan, index=g.index, dtype="float64")
            cal_gap = pd.Series(np.nan, index=g.index, dtype="float64")
        alpha = max(1.0 - nominal, 1e-6)
        interval_score = width + (2.0 / alpha) * (lo - y).clip(lower=0.0) + (2.0 / alpha) * (y - hi).clip(lower=0.0)
        return pd.Series({
            "n": int(g.shape[0]),
            "coverage": float(covered.mean()),
            "nominal_coverage": nominal,
            "coverage_error": float(covered.mean() - nominal),
            "avg_interval_width": float(width.mean()),
            "median_interval_width": float(width.median()),
            "avg_pinball_low": float(pl_lo.mean()),
            "avg_pinball_mid": float(pl_mid.mean()) if pred_mid_col is not None else np.nan,
            "avg_pinball_high": float(pl_hi.mean()),
            "avg_interval_score": float(interval_score.mean()),
            "avg_calibration_gap": float(cal_gap.mean()) if pred_mid_col is not None else np.nan,
        })

    if by is None:
        return _agg(df).to_frame().T.reset_index(drop=True)
    return df.groupby(by, dropna=False).apply(_agg).reset_index()


def summarize_probabilistic_forecast(
    panel: pd.DataFrame,
    *,
    target_col: str,
    pred_low_col: str,
    pred_mid_col: Optional[str] = None,
    pred_high_col: str,
    q_low: float = 0.10,
    q_high: float = 0.90,
) -> pd.DataFrame:
    metrics = compute_interval_metrics(
        panel,
        target_col=target_col,
        pred_low_col=pred_low_col,
        pred_mid_col=pred_mid_col,
        pred_high_col=pred_high_col,
        q_low=q_low,
        q_high=q_high,
    )
    if metrics.empty:
        return pd.DataFrame([ProbabilisticSummary(0, np.nan, q_high-q_low, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan).to_dict()])
    row = metrics.iloc[0].to_dict()
    return pd.DataFrame([ProbabilisticSummary(
        n_obs=int(row.get("n", 0) or 0),
        coverage=_safe_float(row.get("coverage")),
        nominal_coverage=_safe_float(row.get("nominal_coverage")),
        coverage_error=_safe_float(row.get("coverage_error")),
        avg_interval_width=_safe_float(row.get("avg_interval_width")),
        median_interval_width=_safe_float(row.get("median_interval_width")),
        avg_pinball_low=_safe_float(row.get("avg_pinball_low")),
        avg_pinball_mid=_safe_float(row.get("avg_pinball_mid")),
        avg_pinball_high=_safe_float(row.get("avg_pinball_high")),
        avg_interval_score=_safe_float(row.get("avg_interval_score")),
        avg_calibration_gap=_safe_float(row.get("avg_calibration_gap")),
    ).to_dict()])

# ============================================================
# Cross-sectional IC by date
# ============================================================


def compute_cross_sectional_ic_by_date(
    panel: pd.DataFrame,
    *,
    signal_col: str,
    target_col: str,
    date_col: str = "date",
    asset_col: str = "asset",
    method: str = "rank_ic",
    min_assets: int = 3,
) -> pd.DataFrame:
    """
    Compute cross-sectional IC per date.

    Methods:
    - rank_ic   : Pearson correlation of cross-sectional percentile ranks
    - spearman  : pandas Spearman correlation
    - kendall   : pandas Kendall correlation
    - pearson   : raw Pearson correlation
    """
    panel = _coerce_panel(panel, date_col=date_col, asset_col=asset_col)
    _validate_signal_and_target(panel, signal_col, target_col)

    rows: List[Dict[str, Any]] = []
    for dt, g in panel.groupby(date_col, sort=True):
        gg = g[[asset_col, signal_col, target_col]].copy()
        gg[signal_col] = pd.to_numeric(gg[signal_col], errors="coerce")
        gg[target_col] = pd.to_numeric(gg[target_col], errors="coerce")
        gg = gg.dropna(subset=[signal_col, target_col])
        n = int(gg.shape[0])
        ic = np.nan
        if n >= int(min_assets):
            ic = _corr_pair(gg[signal_col], gg[target_col], method=method)
        rows.append({
            "date": pd.to_datetime(dt),
            "metric": method,
            "signal_col": signal_col,
            "target_col": target_col,
            "n_assets": n,
            "ic": _safe_float(ic),
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)



def compute_rank_ic_by_date(**kwargs) -> pd.DataFrame:
    return compute_cross_sectional_ic_by_date(method="rank_ic", **kwargs)



def compute_spearman_ic_by_date(**kwargs) -> pd.DataFrame:
    return compute_cross_sectional_ic_by_date(method="spearman", **kwargs)



def compute_kendall_ic_by_date(**kwargs) -> pd.DataFrame:
    return compute_cross_sectional_ic_by_date(method="kendall", **kwargs)



def summarize_ic_by_date(ic_by_date: pd.DataFrame, *, metric: Optional[str] = None) -> pd.DataFrame:
    if ic_by_date is None or not isinstance(ic_by_date, pd.DataFrame) or ic_by_date.empty:
        return pd.DataFrame([_summarize_ic(pd.Series(dtype="float64"), metric=metric or "ic").to_dict()])
    use_metric = metric or str(ic_by_date.get("metric", pd.Series(["ic"])) .iloc[0])
    return pd.DataFrame([_summarize_ic(pd.to_numeric(ic_by_date["ic"], errors="coerce"), metric=use_metric).to_dict()])



def compute_rolling_ic(
    ic_by_date: pd.DataFrame,
    *,
    window: int = 12,
) -> pd.DataFrame:
    if ic_by_date is None or not isinstance(ic_by_date, pd.DataFrame) or ic_by_date.empty:
        return pd.DataFrame(columns=["date", "rolling_mean_ic", "rolling_std_ic", "rolling_ir", "rolling_hit_rate"])
    if int(window) <= 1:
        raise ValueError(f"window must be > 1, got {window}")

    df = ic_by_date.copy().sort_values("date").reset_index(drop=True)
    xs = pd.to_numeric(df["ic"], errors="coerce")
    df["rolling_mean_ic"] = xs.rolling(window=window, min_periods=max(3, window // 3)).mean()
    df["rolling_std_ic"] = xs.rolling(window=window, min_periods=max(3, window // 3)).std(ddof=1)
    df["rolling_ir"] = df["rolling_mean_ic"] / df["rolling_std_ic"].replace(0.0, np.nan)
    df["rolling_hit_rate"] = (xs > 0).astype(float).rolling(window=window, min_periods=max(3, window // 3)).mean()
    return df


# ============================================================
# IC by asset (time-series relationship)
# ============================================================


def compute_ic_by_asset(
    panel: pd.DataFrame,
    *,
    signal_col: str,
    target_col: str,
    date_col: str = "date",
    asset_col: str = "asset",
    method: str = "spearman",
    min_periods: int = 12,
) -> pd.DataFrame:
    panel = _coerce_panel(panel, date_col=date_col, asset_col=asset_col)
    _validate_signal_and_target(panel, signal_col, target_col)

    rows: List[Dict[str, Any]] = []
    for asset, g in panel.groupby(asset_col, sort=True):
        gg = g[[date_col, signal_col, target_col]].copy().sort_values(date_col)
        gg[signal_col] = pd.to_numeric(gg[signal_col], errors="coerce")
        gg[target_col] = pd.to_numeric(gg[target_col], errors="coerce")
        gg = gg.dropna(subset=[signal_col, target_col])
        n = int(gg.shape[0])
        ic = np.nan
        if n >= int(min_periods):
            ic = _corr_pair(gg[signal_col], gg[target_col], method=method if method != "rank_ic" else "rank_ic")
        rows.append({
            "asset": str(asset),
            "metric": method,
            "signal_col": signal_col,
            "target_col": target_col,
            "n_periods": n,
            "ic": _safe_float(ic),
        })
    out = pd.DataFrame(rows).sort_values(["ic", "asset"], ascending=[False, True]).reset_index(drop=True)
    return out


# ============================================================
# IC by regime
# ============================================================


def compute_ic_by_regime(
    panel: pd.DataFrame,
    *,
    signal_col: str,
    target_col: str,
    regime_col: str,
    date_col: str = "date",
    asset_col: str = "asset",
    method: str = "rank_ic",
    min_assets: int = 3,
) -> Dict[str, pd.DataFrame]:
    panel = _coerce_panel(panel, date_col=date_col, asset_col=asset_col)
    _validate_signal_and_target(panel, signal_col, target_col)
    if regime_col not in panel.columns:
        raise KeyError(f"regime column not found: {regime_col}")

    date_level = compute_cross_sectional_ic_by_date(
        panel,
        signal_col=signal_col,
        target_col=target_col,
        date_col=date_col,
        asset_col=asset_col,
        method=method,
        min_assets=min_assets,
    )

    regime_map = (
        panel[[date_col, regime_col]]
        .dropna()
        .drop_duplicates(subset=[date_col])
        .rename(columns={date_col: "date"})
    )
    merged = date_level.merge(regime_map, on="date", how="left")

    summary_rows: List[Dict[str, Any]] = []
    for regime, g in merged.groupby(regime_col, dropna=False):
        summ = _summarize_ic(pd.to_numeric(g["ic"], errors="coerce"), metric=f"{method}|{regime}").to_dict()
        summ["regime"] = regime
        summary_rows.append(summ)
    summary = pd.DataFrame(summary_rows)
    return {
        "ic_by_date": merged,
        "summary_by_regime": summary,
    }


# ============================================================
# Ranking-vs-error comparison across signals
# ============================================================


def compare_signals_ranking_vs_error(
    panel: pd.DataFrame,
    *,
    signal_cols: Sequence[str],
    target_col: str,
    date_col: str = "date",
    asset_col: str = "asset",
    ic_method: str = "rank_ic",
    min_assets: int = 3,
) -> pd.DataFrame:
    panel = _coerce_panel(panel, date_col=date_col, asset_col=asset_col)
    if not signal_cols:
        raise ValueError("signal_cols must contain at least one signal column")
    if target_col not in panel.columns:
        raise KeyError(f"target column not found: {target_col}")

    rows: List[Dict[str, Any]] = []
    for sig in signal_cols:
        if sig not in panel.columns:
            continue
        errors = compute_error_metrics(panel, prediction_col=sig, target_col=target_col)
        ic_by_date = compute_cross_sectional_ic_by_date(
            panel,
            signal_col=sig,
            target_col=target_col,
            date_col=date_col,
            asset_col=asset_col,
            method=ic_method,
            min_assets=min_assets,
        )
        rank_ic_summary = summarize_ic_by_date(ic_by_date, metric=ic_method)
        spearman_summary = summarize_ic_by_date(
            compute_spearman_ic_by_date(
                panel=panel,
                signal_col=sig,
                target_col=target_col,
                date_col=date_col,
                asset_col=asset_col,
                min_assets=min_assets,
            ),
            metric="spearman",
        )
        kendall_summary = summarize_ic_by_date(
            compute_kendall_ic_by_date(
                panel=panel,
                signal_col=sig,
                target_col=target_col,
                date_col=date_col,
                asset_col=asset_col,
                min_assets=min_assets,
            ),
            metric="kendall",
        )
        er = errors.iloc[0].to_dict() if not errors.empty else {"n": np.nan, "mae": np.nan, "mse": np.nan, "rmse": np.nan, "bias": np.nan}
        r1 = rank_ic_summary.iloc[0].to_dict() if not rank_ic_summary.empty else {}
        r2 = spearman_summary.iloc[0].to_dict() if not spearman_summary.empty else {}
        r3 = kendall_summary.iloc[0].to_dict() if not kendall_summary.empty else {}
        rows.append({
            "signal_col": sig,
            "n_obs": er.get("n", np.nan),
            "mae": er.get("mae", np.nan),
            "mse": er.get("mse", np.nan),
            "rmse": er.get("rmse", np.nan),
            "bias": er.get("bias", np.nan),
            "rank_ic_mean": r1.get("mean_ic", np.nan),
            "rank_ic_ir": r1.get("ir", np.nan),
            "rank_ic_hit_rate": r1.get("hit_rate", np.nan),
            "spearman_ic_mean": r2.get("mean_ic", np.nan),
            "spearman_ic_ir": r2.get("ir", np.nan),
            "kendall_ic_mean": r3.get("mean_ic", np.nan),
            "kendall_ic_ir": r3.get("ir", np.nan),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Heuristic composite ranks for comparison layer.
    out["rank_mae"] = out["mae"].rank(method="min", ascending=True)
    out["rank_rmse"] = out["rmse"].rank(method="min", ascending=True)
    out["rank_rank_ic"] = out["rank_ic_mean"].rank(method="min", ascending=False)
    out["rank_spearman_ic"] = out["spearman_ic_mean"].rank(method="min", ascending=False)
    out["rank_composite"] = (
        out[["rank_mae", "rank_rmse", "rank_rank_ic", "rank_spearman_ic"]]
        .mean(axis=1)
    )
    return out.sort_values(["rank_composite", "rank_ic_mean", "mae"], ascending=[True, False, True]).reset_index(drop=True)


# ============================================================
# Probabilistic convenience wrapper
# ============================================================


def evaluate_probabilistic_forecast_research_layer(
    panel: pd.DataFrame,
    *,
    target_col: str,
    pred_low_col: str,
    pred_mid_col: Optional[str] = None,
    pred_high_col: str,
    probabilistic_q_low: float = 0.10,
    probabilistic_q_high: float = 0.90,
    date_col: str = "date",
    asset_col: str = "asset",
    regime_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    End-to-end probabilistic forecast evaluation: overall interval metrics,
    summary, by-date diagnostics, and optional by-regime diagnostics.
    """
    panel = _coerce_panel(panel, date_col=date_col, asset_col=asset_col)
    out = {
        "probabilistic_interval_metrics": compute_interval_metrics(
            panel,
            target_col=target_col,
            pred_low_col=pred_low_col,
            pred_mid_col=pred_mid_col if pred_mid_col is not None and pred_mid_col in panel.columns else None,
            pred_high_col=pred_high_col,
            q_low=probabilistic_q_low,
            q_high=probabilistic_q_high,
        ),
        "probabilistic_summary": summarize_probabilistic_forecast(
            panel,
            target_col=target_col,
            pred_low_col=pred_low_col,
            pred_mid_col=pred_mid_col if pred_mid_col is not None and pred_mid_col in panel.columns else None,
            pred_high_col=pred_high_col,
            q_low=probabilistic_q_low,
            q_high=probabilistic_q_high,
        ),
        "probabilistic_by_date": compute_interval_metrics(
            panel,
            by=date_col,
            target_col=target_col,
            pred_low_col=pred_low_col,
            pred_mid_col=pred_mid_col if pred_mid_col is not None and pred_mid_col in panel.columns else None,
            pred_high_col=pred_high_col,
            q_low=probabilistic_q_low,
            q_high=probabilistic_q_high,
        ),
    }
    if regime_col is not None and regime_col in panel.columns:
        out["probabilistic_by_regime"] = compute_interval_metrics(
            panel,
            by=regime_col,
            target_col=target_col,
            pred_low_col=pred_low_col,
            pred_mid_col=pred_mid_col if pred_mid_col is not None and pred_mid_col in panel.columns else None,
            pred_high_col=pred_high_col,
            q_low=probabilistic_q_low,
            q_high=probabilistic_q_high,
        )
    else:
        out["probabilistic_by_regime"] = pd.DataFrame()
    return out


# ============================================================
# High-level convenience wrapper
# ============================================================


def evaluate_signal_research_layer(
    panel: pd.DataFrame,
    *,
    signal_col: str,
    target_col: Optional[str] = None,
    return_col: Optional[str] = "return",
    horizon: int = 1,
    date_col: str = "date",
    asset_col: str = "asset",
    regime_col: Optional[str] = None,
    min_assets: int = 3,
    rolling_window: int = 12,
    pred_low_col: Optional[str] = None,
    pred_mid_col: Optional[str] = None,
    pred_high_col: Optional[str] = None,
    probabilistic_q_low: float = 0.10,
    probabilistic_q_high: float = 0.90,
) -> Dict[str, Any]:
    """
    End-to-end research evaluation for a single signal.

    If target_col is omitted, the function creates a forward return target from
    return_col using a per-asset shift by `horizon`.
    """
    panel = _coerce_panel(panel, date_col=date_col, asset_col=asset_col)

    use_target = target_col
    work = panel.copy()
    if use_target is None:
        if return_col is None:
            raise ValueError("Either target_col or return_col must be provided")
        work = add_forward_return_column(
            work,
            return_col=return_col,
            date_col=date_col,
            asset_col=asset_col,
            horizon=horizon,
            forward_return_col=f"future_return_{int(horizon)}",
        )
        use_target = f"future_return_{int(horizon)}"

    rank_ic_by_date = compute_rank_ic_by_date(
        panel=work,
        signal_col=signal_col,
        target_col=use_target,
        date_col=date_col,
        asset_col=asset_col,
        min_assets=min_assets,
    )
    spearman_ic_by_date = compute_spearman_ic_by_date(
        panel=work,
        signal_col=signal_col,
        target_col=use_target,
        date_col=date_col,
        asset_col=asset_col,
        min_assets=min_assets,
    )
    kendall_ic_by_date = compute_kendall_ic_by_date(
        panel=work,
        signal_col=signal_col,
        target_col=use_target,
        date_col=date_col,
        asset_col=asset_col,
        min_assets=min_assets,
    )

    out = {
        "panel": work,
        "error_metrics": compute_error_metrics(work, prediction_col=signal_col, target_col=use_target),
        "rank_ic_by_date": rank_ic_by_date,
        "spearman_ic_by_date": spearman_ic_by_date,
        "kendall_ic_by_date": kendall_ic_by_date,
        "rank_ic_summary": summarize_ic_by_date(rank_ic_by_date, metric="rank_ic"),
        "spearman_ic_summary": summarize_ic_by_date(spearman_ic_by_date, metric="spearman"),
        "kendall_ic_summary": summarize_ic_by_date(kendall_ic_by_date, metric="kendall"),
        "rolling_rank_ic": compute_rolling_ic(rank_ic_by_date, window=rolling_window),
        "ic_by_asset": compute_ic_by_asset(
            work,
            signal_col=signal_col,
            target_col=use_target,
            date_col=date_col,
            asset_col=asset_col,
            method="spearman",
            min_periods=max(6, rolling_window),
        ),
    }

    if regime_col is not None and regime_col in work.columns:
        out["ic_by_regime"] = compute_ic_by_regime(
            work,
            signal_col=signal_col,
            target_col=use_target,
            regime_col=regime_col,
            date_col=date_col,
            asset_col=asset_col,
            method="rank_ic",
            min_assets=min_assets,
        )
    else:
        out["ic_by_regime"] = {"ic_by_date": pd.DataFrame(), "summary_by_regime": pd.DataFrame()}

    if pred_low_col is not None and pred_high_col is not None and pred_low_col in work.columns and pred_high_col in work.columns:
        prob = evaluate_probabilistic_forecast_research_layer(
            work,
            target_col=use_target,
            pred_low_col=pred_low_col,
            pred_mid_col=pred_mid_col if pred_mid_col is not None and pred_mid_col in work.columns else None,
            pred_high_col=pred_high_col,
            probabilistic_q_low=probabilistic_q_low,
            probabilistic_q_high=probabilistic_q_high,
            date_col=date_col,
            asset_col=asset_col,
            regime_col=regime_col,
        )
        out.update(prob)
    else:
        out["probabilistic_interval_metrics"] = pd.DataFrame()
        out["probabilistic_summary"] = pd.DataFrame()
        out["probabilistic_by_date"] = pd.DataFrame()
        out["probabilistic_by_regime"] = pd.DataFrame()

    return out




# ============================================================
# Automatic signal-mode selection by Rank IC
# ============================================================


def _get_cfg_value(cfg: Any, *names: str) -> Any:
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        for name in names:
            if name in cfg:
                return cfg.get(name)
        return None
    for name in names:
        if hasattr(cfg, name):
            return getattr(cfg, name)
    return None


def _mode_from_series(diag: pd.DataFrame, cols: Sequence[str]) -> Optional[str]:
    if diag is None or not isinstance(diag, pd.DataFrame) or diag.empty:
        return None
    for col in cols:
        if col not in diag.columns:
            continue
        xs = diag[col].dropna().astype(str).str.strip()
        xs = xs[xs.ne("")]
        if xs.empty:
            continue
        mode = xs.mode(dropna=True)
        if not mode.empty:
            return str(mode.iloc[0])
    return None


def _normalize_signal_mode(signal_mode: Any) -> Optional[str]:
    if signal_mode is None:
        return None
    mode = str(signal_mode).strip().lower()
    if not mode:
        return None
    if mode.endswith("_multi_loss"):
        mode = mode[: -len("_multi_loss")]
    alias_map = {
        "mean": "mu_sigma",
        "mu": "mu_sigma",
        "mu_only": "mu_sigma",
        "gaussian_mu": "mu_sigma",
        "mu_sigma_point": "mu_sigma",
        "robust_mu": "huber_mu",
        "huber": "huber_mu",
        "rank": "lambdarank_real",
        "lambdarank": "lambdarank_real",
        "lambda_rank": "lambdarank_real",
        "lambdarank_like_score": "lambdarank_like",
        "directional": "directional_classifier",
        "classifier_directional": "directional_classifier",
        "logistic": "logistic_loss",
        "logit": "logistic_loss",
        "topk": "top_k_classifier",
        "top_k": "top_k_classifier",
        "topk_classifier": "top_k_classifier",
        "quantile": "quantile_loss",
        "quantile_regression": "quantile_loss",
        "quantile_model": "quantile_loss",
        "mu_sigma_fallback": "mu_sigma",
    }
    return alias_map.get(mode, mode)


def _normalize_signal_output_col(col: Any) -> Optional[str]:
    if col is None:
        return None
    name = str(col).strip()
    if not name:
        return None
    alias_map = {
        "signal_prob_up_series": "signal_prob_up",
        "signal_top_k_prob_series": "signal_top_k_prob",
        "signal_quantile_pred_series": "signal_quantile_pred",
        "signal_primary_output_series_key": None,
    }
    if name in alias_map:
        return alias_map[name]
    if name.endswith("_series"):
        trimmed = name[: -len("_series")]
        return trimmed or name
    return name


def _first_non_empty_text_from_cols(diag: pd.DataFrame, cols: Sequence[str]) -> Optional[str]:
    if diag is None or not isinstance(diag, pd.DataFrame) or diag.empty:
        return None
    for col in cols:
        if col not in diag.columns:
            continue
        xs = diag[col].dropna().astype(str).str.strip()
        xs = xs[xs.ne("")]
        if xs.empty:
            continue
        return str(xs.iloc[0])
    return None


def _split_signal_candidate_text(value: Any) -> Tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, (list, tuple, set)):
        raw_parts = list(value)
    else:
        text = str(value).strip()
        if not text:
            return tuple()
        normalized = text.replace(";", "|").replace(",", "|")
        raw_parts = normalized.split("|")
    out: List[str] = []
    for part in raw_parts:
        col = _normalize_signal_output_col(part)
        if col:
            out.append(col)
    return tuple(dict.fromkeys(out))



def _get_run_signal_modes(run_report: Dict[str, Any]) -> Dict[str, Any]:
    cfg = (run_report or {}).get("config")
    diagnostics = (run_report or {}).get("diagnostics_df", pd.DataFrame())
    requested = _get_cfg_value(cfg, "signal_mode", "signal_mode_requested", "requested_signal_mode")
    effective = _mode_from_series(
        diagnostics,
        cols=["signal_mode_effective", "effective_signal_mode", "signal_mode_used", "signal_mode"],
    )
    if effective is None:
        effective = _get_cfg_value(cfg, "signal_mode_effective", "effective_signal_mode", "signal_mode")

    contract_base = _first_non_empty_text_from_cols(
        diagnostics,
        cols=["signal_mode_contract_base", "signal_contract_base"],
    )
    if contract_base is None:
        contract_base = _get_cfg_value(cfg, "signal_mode_contract_base", "signal_contract_base")

    preferred = _first_non_empty_text_from_cols(
        diagnostics,
        cols=["signal_eval_col_preferred", "signal_primary_output_label", "signal_primary_output_series_key"],
    )
    if preferred is None:
        preferred = _get_cfg_value(
            cfg,
            "signal_eval_col_preferred",
            "signal_primary_output_label",
            "signal_primary_output_series_key",
        )
    preferred_norm = _normalize_signal_output_col(preferred)

    candidates_text = _first_non_empty_text_from_cols(
        diagnostics,
        cols=["signal_eval_col_candidates", "signal_primary_output_candidates"],
    )
    if candidates_text is None:
        candidates_text = _get_cfg_value(cfg, "signal_eval_col_candidates", "signal_primary_output_candidates")
    candidate_cols = _split_signal_candidate_text(candidates_text)
    if preferred_norm is not None:
        candidate_cols = tuple(dict.fromkeys((preferred_norm,) + candidate_cols))

    primary_series_key = _first_non_empty_text_from_cols(
        diagnostics,
        cols=["signal_primary_output_series_key"],
    )
    if primary_series_key is None:
        primary_series_key = _get_cfg_value(cfg, "signal_primary_output_series_key")
    primary_series_key_norm = _normalize_signal_output_col(primary_series_key)

    primary_label = _first_non_empty_text_from_cols(
        diagnostics,
        cols=["signal_primary_output_label"],
    )
    if primary_label is None:
        primary_label = _get_cfg_value(cfg, "signal_primary_output_label")
    primary_label_norm = _normalize_signal_output_col(primary_label)

    primary_kind = _first_non_empty_text_from_cols(
        diagnostics,
        cols=["signal_primary_output_kind"],
    )
    if primary_kind is None:
        primary_kind = _get_cfg_value(cfg, "signal_primary_output_kind")

    requested_norm = _normalize_signal_mode(requested)
    effective_norm = _normalize_signal_mode(effective)
    contract_base_norm = _normalize_signal_mode(contract_base)
    if effective_norm is None:
        effective_norm = contract_base_norm or requested_norm
    if contract_base_norm is None:
        contract_base_norm = effective_norm or requested_norm

    return {
        "signal_mode_requested": requested_norm,
        "signal_mode_effective": effective_norm,
        "signal_mode_contract_base": contract_base_norm,
        "signal_eval_col_preferred": preferred_norm,
        "signal_eval_col_candidates": candidate_cols,
        "signal_primary_output_series_key": primary_series_key_norm,
        "signal_primary_output_label": primary_label_norm,
        "signal_primary_output_kind": None if primary_kind is None else str(primary_kind),
        "signal_mode_requested_raw": None if requested is None else str(requested),
        "signal_mode_effective_raw": None if effective is None else str(effective),
        "signal_mode_contract_base_raw": None if contract_base is None else str(contract_base),
        "signal_eval_col_preferred_raw": None if preferred is None else str(preferred),
        "signal_primary_output_series_key_raw": None if primary_series_key is None else str(primary_series_key),
        "signal_primary_output_label_raw": None if primary_label is None else str(primary_label),
    }


def _signal_mode_preferred_cols(signal_mode: Any) -> Tuple[str, ...]:
    mode = _normalize_signal_mode(signal_mode)
    default_cols = ("mu_hat_used", "score", "mu_hat", "weight")
    mapping = {
        "mu_sigma": ("mu_hat_used", "mu_hat", "score", "weight"),
        "huber_mu": ("mu_hat_used", "mu_hat", "score", "weight"),
        "lambdarank_like": ("score", "mu_hat_used", "mu_hat", "weight"),
        "lambdarank_real": ("score", "mu_hat_used", "mu_hat", "weight"),
        "directional_classifier": (
            "signal_prob_up", "score", "mu_hat_used", "mu_hat", "weight"
        ),
        "logistic_loss": (
            "signal_prob_up", "score", "mu_hat_used", "mu_hat", "weight"
        ),
        "top_k_classifier": (
            "signal_top_k_prob", "score", "mu_hat_used", "mu_hat", "weight"
        ),
        "quantile_loss": ("signal_quantile_pred", "mu_hat_used", "score", "mu_hat", "weight"),
    }
    cols = list(mapping.get(mode, default_cols))
    if mode in {"directional_classifier", "logistic_loss"}:
        cols.extend(["signal_logistic_prob_mean", "signal_directional_prob_mean"])
    if mode == "top_k_classifier":
        cols.extend(["signal_top_k_prob_mean"])
    if mode == "quantile_loss":
        cols.extend(["signal_quantile_pred_mean"])
    return tuple(dict.fromkeys(cols))


def _empty_rank_ic_quality_result(**extra: Any) -> Dict[str, Any]:
    base = {
        "signal_mode_requested": None,
        "signal_mode_effective": None,
        "signal_col_preferred": None,
        "signal_col_used": None,
        "signal_col_resolution": "unresolved",
        "signal_col_fallback_used": False,
        "signal_col_resolution_note": "No diagnostics available for semantic signal-column resolution.",
        "rank_ic_mean": np.nan,
        "rank_ic_ir": np.nan,
        "rank_ic_hit_rate": np.nan,
        "rank_ic_t_stat": np.nan,
        "n_dates": 0,
        "ic_by_date": pd.DataFrame(),
        "rank_ic_summary": pd.DataFrame(),
    }
    base.update(extra)
    return base


def _evaluate_rank_ic_for_signal_col(
    diagnostics: pd.DataFrame,
    *,
    signal_col: str,
    target_col: str,
    date_col: str,
    asset_col: str,
    min_assets: int,
) -> Dict[str, Any]:
    if signal_col not in diagnostics.columns:
        return {"signal_col": signal_col, "n_valid_rows": 0, "n_dates": 0, "ic_by_date": pd.DataFrame(), "rank_ic_summary": pd.DataFrame()}
    work = diagnostics[[c for c in [date_col, asset_col, signal_col, target_col] if c in diagnostics.columns]].copy()
    if signal_col not in work.columns or target_col not in work.columns:
        return {"signal_col": signal_col, "n_valid_rows": 0, "n_dates": 0, "ic_by_date": pd.DataFrame(), "rank_ic_summary": pd.DataFrame()}
    work[signal_col] = pd.to_numeric(work[signal_col], errors="coerce")
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=[signal_col, target_col])
    if work.empty:
        return {"signal_col": signal_col, "n_valid_rows": 0, "n_dates": 0, "ic_by_date": pd.DataFrame(), "rank_ic_summary": pd.DataFrame()}
    ic_by_date = compute_rank_ic_by_date(
        panel=work,
        signal_col=signal_col,
        target_col=target_col,
        date_col=date_col,
        asset_col=asset_col,
        min_assets=min_assets,
    )
    summary = summarize_ic_by_date(ic_by_date, metric="rank_ic")
    srow = summary.iloc[0].to_dict() if not summary.empty else {}
    n_dates = int(_safe_float(srow.get("n_dates")) or 0)
    return {
        "signal_col": signal_col,
        "n_valid_rows": int(work.shape[0]),
        "n_dates": n_dates,
        "ic_by_date": ic_by_date,
        "rank_ic_summary": summary,
        "rank_ic_mean": _safe_float(srow.get("mean_ic")),
        "rank_ic_ir": _safe_float(srow.get("ir")),
        "rank_ic_hit_rate": _safe_float(srow.get("hit_rate")),
        "rank_ic_t_stat": _safe_float(srow.get("t_stat")),
    }


def resolve_rank_ic_signal_col(
    run_report: Dict[str, Any],
    *,
    candidate_signal_cols: Sequence[str] = ("mu_hat_used", "score", "mu_hat", "weight"),
    target_col: str = "realised_return",
    date_col: str = "date",
    asset_col: str = "asset",
    min_assets: int = 3,
) -> Dict[str, Any]:
    weights_df = run_report.get("weights_df", pd.DataFrame())
    diagnostics_df = run_report.get("diagnostics_df", pd.DataFrame())

    source_name = None
    source_df = pd.DataFrame()
    if isinstance(weights_df, pd.DataFrame) and not weights_df.empty:
        source_name = "weights_df"
        source_df = weights_df
    elif isinstance(diagnostics_df, pd.DataFrame) and not diagnostics_df.empty:
        source_name = "diagnostics_df"
        source_df = diagnostics_df
    else:
        return _empty_rank_ic_quality_result()

    if target_col not in source_df.columns:
        if source_name == "weights_df" and isinstance(diagnostics_df, pd.DataFrame) and not diagnostics_df.empty and target_col in diagnostics_df.columns:
            source_name = "diagnostics_df"
            source_df = diagnostics_df
        else:
            raise KeyError(f"run {source_name or 'report'} missing required target column: {target_col}")

    mode_info = _get_run_signal_modes(run_report)
    requested_mode = mode_info.get("signal_mode_requested")
    effective_mode = mode_info.get("signal_mode_effective")
    explicit_preferred = mode_info.get("signal_eval_col_preferred")
    preferred_effective = list(_signal_mode_preferred_cols(effective_mode))
    preferred_requested = list(_signal_mode_preferred_cols(requested_mode))
    preferred_union = list(dict.fromkeys(([explicit_preferred] if explicit_preferred else []) + preferred_effective + preferred_requested))
    fallback_candidates = [str(c) for c in candidate_signal_cols if str(c) not in preferred_union]
    evaluation_order = [c for c in preferred_union + fallback_candidates if c in source_df.columns]

    signal_col_preferred = explicit_preferred or (preferred_effective[0] if preferred_effective else None)
    resolution = "no_candidate_available"
    note = f"No preferred or fallback signal column is available in {source_name}."
    fallback_used = False
    used = None
    ic_by_date = pd.DataFrame()
    summary = pd.DataFrame()
    stats = {"rank_ic_mean": np.nan, "rank_ic_ir": np.nan, "rank_ic_hit_rate": np.nan, "rank_ic_t_stat": np.nan, "n_dates": 0}

    for col in evaluation_order:
        detail = _evaluate_rank_ic_for_signal_col(
            source_df,
            signal_col=col,
            target_col=target_col,
            date_col=date_col,
            asset_col=asset_col,
            min_assets=min_assets,
        )
        if int(detail.get("n_dates", 0) or 0) <= 0:
            continue
        used = col
        ic_by_date = detail.get("ic_by_date", pd.DataFrame())
        summary = detail.get("rank_ic_summary", pd.DataFrame())
        stats = {
            "rank_ic_mean": detail.get("rank_ic_mean"),
            "rank_ic_ir": detail.get("rank_ic_ir"),
            "rank_ic_hit_rate": detail.get("rank_ic_hit_rate"),
            "rank_ic_t_stat": detail.get("rank_ic_t_stat"),
            "n_dates": detail.get("n_dates"),
        }
        if explicit_preferred is not None and col == explicit_preferred:
            resolution = "explicit_contract_preferred"
            note = f"Resolved using explicit diagnostics contract preferred column '{col}' from {source_name}."
            fallback_used = False
        elif col in preferred_effective:
            resolution = "effective_mode_preferred"
            note = f"Resolved using effective signal_mode='{effective_mode}' with preferred column '{col}' from {source_name}."
            fallback_used = False
        elif col in preferred_requested:
            resolution = "requested_mode_preferred"
            note = f"Effective preferred columns were unavailable; resolved using requested signal_mode='{requested_mode}' with column '{col}' from {source_name}."
            fallback_used = False
        else:
            resolution = "fallback_candidate"
            note = (
                f"Preferred columns for effective signal_mode='{effective_mode}'"
                f" and requested signal_mode='{requested_mode}' were unavailable or unusable; fell back to '{col}' from {source_name}."
            )
            fallback_used = True
        break

    if used is None:
        preferred_text = ", ".join(preferred_union) if preferred_union else "<none>"
        fallback_text = ", ".join(fallback_candidates) if fallback_candidates else "<none>"
        return _empty_rank_ic_quality_result(
            signal_mode_requested=requested_mode,
            signal_mode_effective=effective_mode,
            signal_col_preferred=signal_col_preferred,
            signal_col_resolution=resolution,
            signal_col_fallback_used=False,
            signal_col_resolution_note=(
                f"No usable Rank-IC signal column found in {source_name}. Preferred set: [{preferred_text}]. Fallback set: [{fallback_text}]."
            ),
        )

    return {
        "signal_mode_requested": requested_mode,
        "signal_mode_effective": effective_mode,
        "signal_col_preferred": signal_col_preferred,
        "signal_col_used": used,
        "signal_col_resolution": resolution,
        "signal_col_fallback_used": bool(fallback_used),
        "signal_col_resolution_note": note,
        "rank_ic_source": source_name,
        "ic_by_date": ic_by_date,
        "rank_ic_summary": summary,
        **stats,
    }


def evaluate_run_rank_ic_quality(
    run_report: Dict[str, Any],
    *,
    candidate_signal_cols: Sequence[str] = ("mu_hat_used", "score", "mu_hat", "weight"),
    target_col: str = "realised_return",
    date_col: str = "date",
    asset_col: str = "asset",
    min_assets: int = 3,
) -> Dict[str, Any]:
    return resolve_rank_ic_signal_col(
        run_report,
        candidate_signal_cols=candidate_signal_cols,
        target_col=target_col,
        date_col=date_col,
        asset_col=asset_col,
        min_assets=min_assets,
    )


def auto_select_signal_mode_by_rank_ic(
    run_reports: Dict[str, Dict[str, Any]],
    *,
    candidate_signal_cols: Sequence[str] = ("mu_hat_used", "score", "mu_hat", "weight"),
    target_col: str = "realised_return",
    date_col: str = "date",
    asset_col: str = "asset",
    min_assets: int = 3,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    details: Dict[str, Dict[str, Any]] = {}

    selection_space_kind = "requested_contracts"
    selection_space_restricted_to_base_modes = False

    for mode_name, run_report in (run_reports or {}).items():
        if run_report is None:
            continue
        rank_detail = evaluate_run_rank_ic_quality(
            run_report,
            candidate_signal_cols=candidate_signal_cols,
            target_col=target_col,
            date_col=date_col,
            asset_col=asset_col,
            min_assets=min_assets,
        )
        perf = run_report.get("performance_summary", {}) or {}
        requested_mode = rank_detail.get("signal_mode_requested")
        effective_mode = rank_detail.get("signal_mode_effective")
        effective_variant_explored = str(requested_mode) != str(effective_mode)
        if effective_variant_explored:
            selection_space_note = "effective variant inferred post hoc from requested contract"
        else:
            selection_space_note = "effective mode matches requested contract"
        row = {
            "signal_mode": str(mode_name),
            "signal_mode_requested": requested_mode,
            "signal_mode_effective": effective_mode,
            "effective_variant_explored": bool(effective_variant_explored),
            "selection_space_kind": selection_space_kind,
            "selection_space_restricted_to_base_modes": bool(selection_space_restricted_to_base_modes),
            "selection_space_note": selection_space_note,
            "signal_col_preferred": rank_detail.get("signal_col_preferred"),
            "signal_col_used": rank_detail.get("signal_col_used"),
            "signal_col_resolution": rank_detail.get("signal_col_resolution"),
            "signal_col_fallback_used": rank_detail.get("signal_col_fallback_used"),
            "signal_col_resolution_note": rank_detail.get("signal_col_resolution_note"),
            "rank_ic_mean": rank_detail.get("rank_ic_mean"),
            "rank_ic_ir": rank_detail.get("rank_ic_ir"),
            "rank_ic_hit_rate": rank_detail.get("rank_ic_hit_rate"),
            "rank_ic_t_stat": rank_detail.get("rank_ic_t_stat"),
            "n_dates": rank_detail.get("n_dates"),
            "cagr": perf.get("cagr"),
            "sharpe": perf.get("sharpe"),
            "annual_volatility": perf.get("annual_volatility", perf.get("annualized_volatility")),
            "max_drawdown": perf.get("max_drawdown"),
            "mean_turnover": perf.get("mean_turnover"),
        }
        rows.append(row)
        details[str(mode_name)] = rank_detail

    table = pd.DataFrame(rows)
    if table.empty:
        return {
            "selected_signal_mode": None,
            "selection_table": pd.DataFrame(),
            "selection_details": details,
            "selection_space_kind": selection_space_kind,
            "selection_space_restricted_to_base_modes": bool(selection_space_restricted_to_base_modes),
            "effective_variant_explored": False,
            "selection_space_note": "selection metadata not available",
        }

    for col in ["rank_ic_mean", "rank_ic_ir", "rank_ic_hit_rate", "rank_ic_t_stat", "n_dates", "cagr", "sharpe", "annual_volatility", "max_drawdown", "mean_turnover"]:
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce")

    table = table.sort_values(
        ["rank_ic_mean", "rank_ic_ir", "sharpe", "cagr"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    selected = str(table.iloc[0]["signal_mode"]) if not table.empty else None
    best_row = table.iloc[0].to_dict() if not table.empty else {}
    return {
        "selected_signal_mode": selected,
        "selection_table": table,
        "selection_details": details,
        "selection_space_kind": selection_space_kind,
        "selection_space_restricted_to_base_modes": bool(selection_space_restricted_to_base_modes),
        "effective_variant_explored": bool(best_row.get("effective_variant_explored", False)),
        "selection_space_note": best_row.get("selection_space_note", "selection metadata not available"),
    }


# ============================================================
# Smoke test
# ============================================================


def _make_toy_panel(n_dates: int = 24, n_assets: int = 8, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-31", periods=n_dates, freq="M")
    assets = [f"A{i:02d}" for i in range(n_assets)]
    rows: List[Dict[str, Any]] = []

    for asset_idx, asset in enumerate(assets):
        latent = rng.normal(scale=0.03, size=n_dates + 1)
        noise = rng.normal(scale=0.05, size=n_dates + 1)
        signal = 0.6 * latent[:-1] + 0.4 * rng.normal(scale=0.02, size=n_dates)
        future_ret = 0.5 * latent[1:] + noise[1:]
        regime = np.where(np.arange(n_dates) % 2 == 0, "low_corr", "high_corr")
        for i, dt in enumerate(dates):
            rows.append({
                "date": dt,
                "asset": asset,
                "signal": float(signal[i] + 0.02 * asset_idx / max(1, n_assets - 1)),
                "return": float(future_ret[i]),
                "regime": regime[i],
            })
    return pd.DataFrame(rows)



def _smoke_test() -> None:
    panel = _make_toy_panel()
    # Explicit forward target from current return to next-period return
    work = add_forward_return_column(panel, return_col="return", horizon=1)
    # Also create a model prediction column for error metrics / comparison.
    work["signal_alt"] = work["signal"] * 0.8 + 0.02
    work["prob_low"] = work["signal"] - 0.05
    work["prob_mid"] = work["signal"]
    work["prob_high"] = work["signal"] + 0.05

    rank_ic = compute_rank_ic_by_date(panel=work, signal_col="signal", target_col="future_return_1")
    assert not rank_ic.empty and {"date", "ic", "n_assets"}.issubset(rank_ic.columns)

    rolling = compute_rolling_ic(rank_ic, window=6)
    assert "rolling_mean_ic" in rolling.columns

    by_asset = compute_ic_by_asset(work, signal_col="signal", target_col="future_return_1")
    assert not by_asset.empty and "asset" in by_asset.columns

    by_regime = compute_ic_by_regime(work, signal_col="signal", target_col="future_return_1", regime_col="regime")
    assert "summary_by_regime" in by_regime and isinstance(by_regime["summary_by_regime"], pd.DataFrame)

    cmp_df = compare_signals_ranking_vs_error(work, signal_cols=["signal", "signal_alt"], target_col="future_return_1")
    assert not cmp_df.empty and "rank_ic_mean" in cmp_df.columns and "mae" in cmp_df.columns

    report = evaluate_signal_research_layer(
        work,
        signal_col="signal",
        target_col="future_return_1",
        regime_col="regime",
        pred_low_col="prob_low",
        pred_mid_col="prob_mid",
        pred_high_col="prob_high",
    )
    assert "rank_ic_summary" in report and "ic_by_asset" in report
    assert "probabilistic_summary" in report and isinstance(report["probabilistic_summary"], pd.DataFrame)
    assert "probabilistic_by_regime" in report and isinstance(report["probabilistic_by_regime"], pd.DataFrame)


if __name__ == "__main__":
    _smoke_test()
    print("evaluation.py smoke test passed")


# ============================================================
# Probabilistic overlay validation
# ============================================================

def _run_perf_dict(run: Dict[str, Any]) -> Dict[str, Any]:
    perf = run.get("performance_summary", {}) or {}
    if not perf and isinstance(run.get("run_report"), dict):
        perf = ((run.get("run_report") or {}).get("sections") or {}).get("performance", {}) or {}
    return perf


def _run_diag_mean(run: Dict[str, Any], col: str) -> float:
    diag = run.get("diagnostics_df", pd.DataFrame())
    if not isinstance(diag, pd.DataFrame) or diag.empty or col not in diag.columns:
        return np.nan
    return float(pd.to_numeric(diag[col], errors="coerce").mean())


def _run_diag_mode_text(run: Dict[str, Any], col: str) -> Optional[str]:
    diag = run.get("diagnostics_df", pd.DataFrame())
    if not isinstance(diag, pd.DataFrame) or diag.empty or col not in diag.columns:
        return None
    xs = diag[col].dropna().astype(str)
    if xs.empty:
        return None
    mode = xs.mode(dropna=True)
    if mode.empty:
        return None
    return str(mode.iloc[0])


def _build_cap_governance_label(run: Dict[str, Any]) -> str:
    cap_source = _run_diag_mode_text(run, "cap_source")
    if cap_source is None:
        return "cap governance unavailable"

    source_map = {
        "w_cap": "final cap governed by w_cap",
        "asset_weight_cap": "final cap governed by asset_weight_cap",
        "combined_min_cap": "final cap governed by min(asset_weight_cap, w_cap)",
        "uncapped": "no explicit requested base cap active",
    }
    base_label = source_map.get(str(cap_source), f"final cap governed by {cap_source}")

    tighten_bits = []
    checks = [
        ("vol_dependent_cap_multiplier", "vol"),
        ("corr_dependent_cap_multiplier", "corr"),
        ("dispersion_dependent_cap_multiplier", "dispersion"),
        ("regime_dependent_cap_multiplier", "regime"),
    ]
    for col, name in checks:
        v = _run_diag_mean(run, col)
        if np.isfinite(v) and float(v) < 0.999999:
            tighten_bits.append(name)

    if tighten_bits:
        return f"{base_label}; tightened by {'/'.join(tighten_bits)} cap"
    return base_label


def _normalize_probabilistic_mode(probabilistic_mode: Any) -> str:
    mode = str(probabilistic_mode or "none").strip().lower()
    alias_map = {
        "off": "none",
        "disabled": "none",
        "historical_regime_filtered": "historical_by_regime",
        "regime_filtered_historical": "historical_by_regime",
        "historical_filtered_by_features": "historical_by_features",
        "feature_aware_regime_filtered": "historical_by_features",
        "knn_historical_by_regime": "knn_historical",
        "parametric": "parametric_feature_aware",
        "bucketed_historical": "feature_bucketed_historical",
    }
    return alias_map.get(mode, mode or "none")


def _get_prob_source(run: Dict[str, Any]) -> Optional[str]:
    source = _run_diag_mode_text(run, "prob_source")
    if source is None:
        cfg = run.get("config")
        source = _get_cfg_value(cfg, "prob_source", "probabilistic_source")
    return None if source is None else str(source)


def _get_probabilistic_mode(run: Dict[str, Any]) -> str:
    cfg = run.get("config")
    mode = _run_diag_mode_text(
        run,
        "probabilistic_mode_effective",
    )
    if mode is None:
        mode = _run_diag_mode_text(run, "probabilistic_mode")
    if mode is None:
        mode = _get_cfg_value(cfg, "probabilistic_mode", "probabilistic_mode_effective")
    mode = _normalize_probabilistic_mode(mode)

    prob_source = _normalize_probabilistic_mode(_get_prob_source(run))
    if mode == "none" and prob_source in {
        "historical", "historical_by_regime", "historical_by_features",
        "parametric_feature_aware", "knn_historical",
        "feature_bucketed_historical", "quantile_regression", "hybrid",
    }:
        mode = prob_source
    return mode


def _get_run_diagnostics_df(run: Dict[str, Any]) -> pd.DataFrame:
    diag = (run or {}).get("diagnostics_df", pd.DataFrame())
    if not isinstance(diag, pd.DataFrame) or diag.empty:
        return pd.DataFrame()
    return diag.copy()


def _run_diag_non_na_rate(run: Dict[str, Any], col: str) -> float:
    diag = _get_run_diagnostics_df(run)
    if diag.empty or col not in diag.columns:
        return 0.0
    xs = diag[col]
    return float(xs.notna().mean())


def _run_diag_present_cols(run: Dict[str, Any], cols: Sequence[str]) -> List[str]:
    diag = _get_run_diagnostics_df(run)
    if diag.empty:
        return []
    return [str(col) for col in cols if str(col) in diag.columns]


def _run_diag_missing_cols(run: Dict[str, Any], cols: Sequence[str]) -> List[str]:
    diag = _get_run_diagnostics_df(run)
    if diag.empty:
        return [str(col) for col in cols]
    return [str(col) for col in cols if str(col) not in diag.columns]


def _cfg_numeric_value(run: Dict[str, Any], *names: str) -> float:
    cfg = (run or {}).get("config")
    value = _get_cfg_value(cfg, *names)
    return _safe_float(value)


def _cfg_text_value(run: Dict[str, Any], *names: str) -> Optional[str]:
    cfg = (run or {}).get("config")
    value = _get_cfg_value(cfg, *names)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _diag_mean_or_cfg(run: Dict[str, Any], diag_cols: Sequence[str], cfg_names: Sequence[str] = ()) -> float:
    value = _diag_mean_fallback(run, *diag_cols)
    if np.isfinite(value):
        return float(value)
    for name in cfg_names:
        value = _cfg_numeric_value(run, name)
        if np.isfinite(value):
            return float(value)
    return np.nan


def _diag_mode_or_cfg(run: Dict[str, Any], diag_cols: Sequence[str], cfg_names: Sequence[str] = ()) -> Optional[str]:
    for col in diag_cols:
        value = _run_diag_mode_text(run, col)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    for name in cfg_names:
        value = _cfg_text_value(run, name)
        if value is not None:
            return value
    return None


def _build_overlay_feature_mu_rows(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"metric": "feature_mu_active_share", "value": _diag_mean_fallback(run, "feature_mu_active_share", "feature_mu_active")},
        {"metric": "feature_mu_match_n_mean", "value": _diag_mean_fallback(run, "feature_mu_match_n_mean", "feature_mu_match_n")},
        {"metric": "feature_mu_cols_used_mean", "value": _diag_mean_fallback(run, "feature_mu_cols_used_mean", "feature_mu_cols_used")},
        {"metric": "feature_mu_distance_mean", "value": _diag_mean_fallback(run, "feature_mu_distance_mean")},
        {"metric": "feature_mu_abs_tilt_mean", "value": _diag_mean_fallback(run, "feature_mu_abs_tilt_mean", "feature_mu_abs_tilt")},
        {"metric": "feature_mu_abs_tilt_max", "value": _diag_mean_fallback(run, "feature_mu_abs_tilt_max")},
    ]


def _build_overlay_regime_universe_rows(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    keep_low = _diag_mean_or_cfg(
        run,
        diag_cols=("regime_universe_keep_frac_low",),
        cfg_names=("regime_universe_keep_frac_low",),
    )
    keep_mid = _diag_mean_or_cfg(
        run,
        diag_cols=("regime_universe_keep_frac_mid",),
        cfg_names=("regime_universe_keep_frac_mid",),
    )
    keep_high = _diag_mean_or_cfg(
        run,
        diag_cols=("regime_universe_keep_frac_high",),
        cfg_names=("regime_universe_keep_frac_high",),
    )
    keep_mean = _diag_mean_or_cfg(
        run,
        diag_cols=("regime_universe_keep_frac",),
        cfg_names=("regime_universe_keep_frac",),
    )
    if not np.isfinite(keep_mean):
        keep_vals = [x for x in [keep_low, keep_mid, keep_high] if np.isfinite(x)]
        keep_mean = float(np.mean(keep_vals)) if keep_vals else np.nan

    enabled_cfg = _cfg_text_value(run, "regime_dependent_universe_enabled")
    enabled_rate = _diag_mean_or_cfg(
        run,
        diag_cols=("regime_dependent_universe_enabled", "regime_universe_enabled", "regime_universe_active"),
        cfg_names=("regime_dependent_universe_enabled",),
    )
    if not np.isfinite(enabled_rate) and enabled_cfg is not None:
        enabled_rate = 1.0 if str(enabled_cfg).strip().lower() in {"1", "true", "yes", "on"} else 0.0

    selection_metric_mode = _diag_mode_or_cfg(
        run,
        ("regime_universe_selection_metric", "regime_universe_metric_mode"),
        ("regime_universe_selection_metric", "regime_universe_metric_mode"),
    )
    regime_mode = _diag_mode_or_cfg(
        run,
        ("regime_universe_regime", "regime", "regime_state", "regime_bucket"),
        ("regime_universe_regime", "regime_mode"),
    )

    return [
        {"metric": "regime_universe_enabled_rate", "value": enabled_rate},
        {"metric": "regime_universe_keep_frac_mean", "value": keep_mean},
        {"metric": "regime_universe_keep_frac_low_mean", "value": keep_low},
        {"metric": "regime_universe_keep_frac_mid_mean", "value": keep_mid},
        {"metric": "regime_universe_keep_frac_high_mean", "value": keep_high},
        {"metric": "regime_universe_selected_assets_mean", "value": _diag_mean_fallback(run, "regime_universe_selected_assets", "regime_selected_assets", "selected_assets_after_regime_filter")},
        {"metric": "regime_universe_filtered_assets_mean", "value": _diag_mean_fallback(run, "regime_universe_filtered_assets", "regime_filtered_assets", "filtered_assets_by_regime")},
        {"metric": "regime_universe_selection_metric_mode", "value": selection_metric_mode},
        {"metric": "regime_universe_metric_mode", "value": selection_metric_mode},
        {"metric": "regime_universe_regime_mode", "value": regime_mode},
        {"metric": "regime_universe_low_offensive_vol_tilt_mean", "value": _diag_mean_or_cfg(run, ("regime_universe_low_offensive_vol_tilt",), ("regime_universe_low_offensive_vol_tilt",))},
        {"metric": "regime_universe_high_defensive_vol_tilt_mean", "value": _diag_mean_or_cfg(run, ("regime_universe_high_defensive_vol_tilt",), ("regime_universe_high_defensive_vol_tilt",))},
        {"metric": "regime_universe_min_assets_mean", "value": _diag_mean_or_cfg(run, ("regime_universe_min_assets",), ("regime_universe_min_assets",))},
        {"metric": "regime_universe_max_assets_mean", "value": _diag_mean_or_cfg(run, ("regime_universe_max_assets",), ("regime_universe_max_assets",))},
    ]




def _resolve_group_column(run: Dict[str, Any], alternatives: Sequence[str]) -> Optional[str]:
    diag = _get_run_diagnostics_df(run)
    if diag.empty:
        return None
    best = None
    best_rate = -1.0
    for col in alternatives:
        if col not in diag.columns:
            continue
        rate = _run_diag_non_na_rate(run, col)
        if rate > best_rate:
            best = col
            best_rate = rate
    return best


def _safe_boolish(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(value):
            return None
        return bool(int(value))
    txt = str(value).strip().lower()
    if txt in {"true", "1", "yes", "y", "on", "ok"}:
        return True
    if txt in {"false", "0", "no", "n", "off", "none", "nan", ""}:
        return False
    return None


def _diag_or_cfg_text(run: Dict[str, Any], diag_cols: Sequence[str], cfg_names: Sequence[str] = ()) -> Optional[str]:
    return _diag_mode_or_cfg(run, diag_cols, cfg_names)


def _diag_or_cfg_bool(run: Dict[str, Any], diag_cols: Sequence[str], cfg_names: Sequence[str] = ()) -> Optional[bool]:
    for col in diag_cols:
        v = _run_diag_mode_text(run, col)
        b = _safe_boolish(v)
        if b is not None:
            return b
    cfg = (run or {}).get("config")
    for name in cfg_names:
        b = _safe_boolish(_get_cfg_value(cfg, name))
        if b is not None:
            return b
    return None


def _probabilistic_mode_contract(probabilistic_mode: str, prob_source: Optional[str] = None) -> Dict[str, Any]:
    mode = _normalize_probabilistic_mode(probabilistic_mode)
    source = None if prob_source is None else str(prob_source).strip().lower()

    common_groups = {
        "probabilistic_active": ("probabilistic_active",),
        "interval_width": ("probabilistic_interval_width_mean", "prob_interval_width", "interval_width"),
        "downside": ("probabilistic_downside_mean", "prob_downside", "downside"),
        "confidence": ("probabilistic_confidence_mean", "prob_confidence", "confidence"),
        "mu_penalty_mean_abs": ("probabilistic_mu_penalty_mean_abs", "prob_mu_penalty_mean_abs"),
        "mu_penalty_max_abs": ("probabilistic_mu_penalty_max_abs", "prob_mu_penalty_max_abs"),
    }
    feature_groups = {
        "feature_aware_share": ("probabilistic_feature_aware_share", "prob_feature_aware_share"),
        "feature_match_n": ("probabilistic_feature_match_n_mean", "prob_feature_match_n", "feature_match_n"),
        "feature_distance": ("probabilistic_feature_distance_mean", "prob_feature_distance", "feature_distance"),
    }
    regime_groups = {
        "regime_label": ("regime", "regime_used", "regime_state", "regime_bucket"),
    }
    bucket_groups = {
        "bucket_match_features": ("probabilistic_bucket_match_features", "prob_bucket_match_features", "bucket_match_features"),
    }
    qr_groups = {
        "qr_active": ("prob_qr_active", "qr_active"),
        "qr_match_n": ("prob_qr_match_n", "qr_match_n", "qr_n_obs"),
    }
    hybrid_groups = {
        "hybrid_weight_qr": ("prob_hybrid_qr_weight",),
        "hybrid_weight_knn": ("prob_hybrid_knn_weight",),
        "hybrid_component_agreement": ("prob_hybrid_component_agreement",),
    }
    interval_triplet_groups = {
        "pred_low": ("pred_low", "prob_q_low"),
        "pred_mid": ("pred_mid", "prob_q_mid"),
        "pred_high": ("pred_high", "prob_q_high"),
    }
    semantic_cols = [
        "prob_source",
        "probabilistic_mode_requested",
        "probabilistic_mode_effective",
        "prob_backend_family",
        "prob_backend_name",
        "prob_backend_variant",
        "prob_backend_distinct",
        "prob_interval_backend",
        "prob_contract_status",
        "prob_contract_note",
        "prob_backend_parent_mode",
        "prob_fallback_triggered",
        "prob_fallback_reason",
    ]
    optional_cols = semantic_cols + [
        "signal_mode_effective",
        "probabilistic_feature_aware_share",
        "probabilistic_feature_match_n_mean",
        "probabilistic_feature_distance_mean",
        "probabilistic_bucket_match_features",
        "qr_active",
        "qr_match_n",
        "regime",
    ]
    contracts = {
        "none": {
            "required_groups": {},
            "optional_cols": optional_cols,
            "accepted_prob_sources": ["none", None],
            "expected_backend_families": ["none"],
            "accepted_interval_backends": ["none"],
            "require_backend_distinct": False,
            "semantic_required_cols": ["prob_source", "probabilistic_mode_effective"],
            "notes": ["No probabilistic overlay expected."],
        },
        "historical": {
            "required_groups": {**common_groups, **interval_triplet_groups},
            "optional_cols": optional_cols,
            "accepted_prob_sources": ["historical"],
            "expected_backend_families": ["historical_quantiles"],
            "accepted_interval_backends": ["historical_quantiles"],
            "require_backend_distinct": True,
            "semantic_required_cols": semantic_cols,
            "notes": ["Historical overlay should expose core interval/downside/confidence diagnostics."],
        },
        "historical_by_regime": {
            "required_groups": {**common_groups, **interval_triplet_groups, **regime_groups},
            "optional_cols": optional_cols,
            "accepted_prob_sources": ["historical_by_regime"],
            "expected_backend_families": ["historical_quantiles_regime_filtered"],
            "accepted_interval_backends": ["historical_quantiles_regime_filtered", "historical_quantiles"],
            "require_backend_distinct": True,
            "semantic_required_cols": semantic_cols,
            "notes": ["Regime-aware historical overlay should expose regime labels and core diagnostics."],
        },
        "historical_by_features": {
            "required_groups": {**common_groups, **interval_triplet_groups, "feature_aware_share": feature_groups["feature_aware_share"], "feature_match_n": feature_groups["feature_match_n"]},
            "optional_cols": optional_cols,
            "accepted_prob_sources": ["historical_by_features"],
            "expected_backend_families": ["historical_quantiles_feature_filtered"],
            "accepted_interval_backends": ["historical_quantiles_feature_filtered", "historical_quantiles"],
            "require_backend_distinct": True,
            "semantic_required_cols": semantic_cols,
            "notes": ["Feature-conditioned historical overlay should expose feature-aware share and match depth."],
        },
        "parametric_feature_aware": {
            "required_groups": {**common_groups, **interval_triplet_groups, "feature_distance": feature_groups["feature_distance"], "feature_match_n": feature_groups["feature_match_n"]},
            "optional_cols": optional_cols,
            "accepted_prob_sources": ["parametric_feature_aware"],
            "expected_backend_families": ["feature_weighted_gaussian"],
            "accepted_interval_backends": ["weighted_parametric", "feature_weighted_gaussian"],
            "require_backend_distinct": True,
            "semantic_required_cols": semantic_cols,
            "notes": ["Parametric feature-aware overlay should expose local distance diagnostics and gaussian-like backend semantics."],
        },
        "knn_historical": {
            "required_groups": {**common_groups, **interval_triplet_groups, "feature_match_n": feature_groups["feature_match_n"], "feature_distance": feature_groups["feature_distance"]},
            "optional_cols": optional_cols,
            "accepted_prob_sources": ["knn_historical"],
            "expected_backend_families": ["distance_weighted_knn_quantiles"],
            "accepted_interval_backends": ["distance_weighted_knn_quantiles", "knn_quantiles"],
            "require_backend_distinct": True,
            "semantic_required_cols": semantic_cols,
            "notes": ["KNN overlay should expose neighbourhood count/distance diagnostics and a KNN-like backend family."],
        },
        "feature_bucketed_historical": {
            "required_groups": {**common_groups, **interval_triplet_groups, "feature_match_n": feature_groups["feature_match_n"]},
            "optional_cols": optional_cols,
            "accepted_prob_sources": ["feature_bucketed_historical"],
            "expected_backend_families": ["bucket_match_quantiles"],
            "accepted_interval_backends": ["bucket_match_quantiles", "historical_quantiles_feature_filtered"],
            "require_backend_distinct": True,
            "semantic_required_cols": semantic_cols,
            "advisory_groups": bucket_groups,
            "notes": ["Bucketed historical overlay should expose bucket/feature match evidence where available."],
        },
        "quantile_regression": {
            "required_groups": {**common_groups, **interval_triplet_groups, **qr_groups},
            "optional_cols": optional_cols,
            "accepted_prob_sources": ["quantile_regression"],
            "expected_backend_families": ["quantile_regression"],
            "accepted_interval_backends": ["qr", "quantile_regression"],
            "require_backend_distinct": True,
            "semantic_required_cols": semantic_cols,
            "notes": ["Quantile-regression overlay should expose QR activation/fit diagnostics and QR-like backend semantics."],
        },
        "hybrid": {
            "required_groups": {**common_groups, **interval_triplet_groups, **hybrid_groups},
            "optional_cols": optional_cols,
            "accepted_prob_sources": ["hybrid"],
            "expected_backend_families": ["hybrid_mix"],
            "accepted_interval_backends": ["hybrid_mix"],
            "require_backend_distinct": True,
            "semantic_required_cols": semantic_cols,
            "notes": ["Hybrid overlay should expose mixture weights or component diagnostics, not just generic intervals."],
        },
    }
    contract = contracts.get(mode, {
        "required_groups": {**common_groups, **interval_triplet_groups},
        "optional_cols": optional_cols,
        "accepted_prob_sources": [mode],
        "expected_backend_families": [mode],
        "accepted_interval_backends": [mode],
        "require_backend_distinct": True,
        "semantic_required_cols": semantic_cols,
        "notes": [f"Unknown probabilistic mode '{mode}'; falling back to generic probabilistic contract."],
    }).copy()
    contract["probabilistic_mode"] = mode
    contract["prob_source"] = source
    return contract


def build_probabilistic_mode_audit_table(run: Dict[str, Any]) -> pd.DataFrame:
    mode = _get_probabilistic_mode(run)
    prob_source = _get_prob_source(run)
    contract = _probabilistic_mode_contract(mode, prob_source=prob_source)
    diag = _get_run_diagnostics_df(run)
    diag_rows = int(diag.shape[0]) if isinstance(diag, pd.DataFrame) else 0

    effective_mode = _normalize_probabilistic_mode(_diag_or_cfg_text(run, ("probabilistic_mode_effective", "probabilistic_mode"), ("probabilistic_mode_effective", "probabilistic_mode")))
    requested_mode = _normalize_probabilistic_mode(_cfg_text_value(run, "probabilistic_mode", "probabilistic_mode_requested") or mode)
    backend_family = _diag_or_cfg_text(run, ("prob_backend_family",), ("prob_backend_family",))
    backend_name = _diag_or_cfg_text(run, ("prob_backend_name",), ("prob_backend_name",))
    backend_variant = _diag_or_cfg_text(run, ("prob_backend_variant",), ("prob_backend_variant",))
    interval_backend = _diag_or_cfg_text(run, ("prob_interval_backend",), ("prob_interval_backend",))
    backend_distinct = _diag_or_cfg_bool(run, ("prob_backend_distinct",), ("prob_backend_distinct",))
    fallback_triggered = _diag_or_cfg_bool(run, ("prob_fallback_triggered",), ("prob_fallback_triggered",))
    fallback_reason = _diag_or_cfg_text(run, ("prob_fallback_reason",), ("prob_fallback_reason",))
    contract_status = _diag_or_cfg_text(run, ("prob_contract_status",), ("prob_contract_status",))
    backend_parent_mode = _diag_or_cfg_text(run, ("prob_backend_parent_mode",), ("prob_backend_parent_mode",))

    required_groups = dict(contract.get("required_groups", {}))
    resolved_required_cols = {}
    missing_required_groups = []
    sparse_required_groups = []
    for group, alternatives in required_groups.items():
        resolved = _resolve_group_column(run, alternatives)
        if resolved is None:
            missing_required_groups.append(group)
            continue
        resolved_required_cols[group] = resolved
        if _run_diag_non_na_rate(run, resolved) < 0.25:
            sparse_required_groups.append(group)

    advisory_groups = dict(contract.get("advisory_groups", {}))
    missing_advisory_groups = []
    for group, alternatives in advisory_groups.items():
        if _resolve_group_column(run, alternatives) is None:
            missing_advisory_groups.append(group)

    semantic_required_cols = list(contract.get("semantic_required_cols", []))
    semantic_present_cols = _run_diag_present_cols(run, semantic_required_cols)
    semantic_missing_cols = _run_diag_missing_cols(run, semantic_required_cols)
    if diag_rows == 0:
        semantic_present_cols = []
        semantic_missing_cols = semantic_required_cols

    accepted_prob_sources = list(contract.get("accepted_prob_sources", []))
    expected_backend_families = [str(x) for x in contract.get("expected_backend_families", []) if x is not None]
    accepted_interval_backends = [str(x) for x in contract.get("accepted_interval_backends", []) if x is not None]
    require_backend_distinct = bool(contract.get("require_backend_distinct", False))

    accepted_prob_source_ok = prob_source in accepted_prob_sources if accepted_prob_sources else True
    backend_family_ok = (backend_family in expected_backend_families) if expected_backend_families else True
    interval_backend_ok = (interval_backend in accepted_interval_backends) if accepted_interval_backends else True
    backend_distinct_ok = True if not require_backend_distinct else (backend_distinct is True)

    interval_cols_ok = all(_resolve_group_column(run, alternatives) is not None for alternatives in {
        "pred_low": ("pred_low", "prob_q_low"),
        "pred_mid": ("pred_mid", "prob_q_mid"),
        "pred_high": ("pred_high", "prob_q_high"),
    }.values())

    required_groups_ok = (len(missing_required_groups) == 0 and len(sparse_required_groups) == 0)
    semantic_required_ok = len(semantic_missing_cols) == 0

    warnings = []
    if mode != "none" and diag_rows == 0:
        warnings.append(f"Mode '{mode}' is active but diagnostics_df is empty.")
    if missing_required_groups:
        warnings.append(f"Missing required diagnostics groups: {', '.join(missing_required_groups)}.")
    if sparse_required_groups:
        warnings.append(f"Sparse required diagnostics groups: {', '.join(sparse_required_groups)}.")
    if not accepted_prob_source_ok:
        warnings.append(f"prob_source='{prob_source}' is outside accepted sources {accepted_prob_sources} for mode '{mode}'.")
    if not backend_family_ok:
        warnings.append(f"prob_backend_family='{backend_family}' is inconsistent with expected families {expected_backend_families} for mode '{mode}'.")
    if not interval_backend_ok:
        warnings.append(f"prob_interval_backend='{interval_backend}' is inconsistent with accepted interval backends {accepted_interval_backends} for mode '{mode}'.")
    if not backend_distinct_ok:
        warnings.append(f"Mode '{mode}' is marketed as distinct but prob_backend_distinct={backend_distinct}.")
    if not semantic_required_ok:
        warnings.append(f"Missing semantic contract columns: {', '.join(semantic_missing_cols)}.")
    if not interval_cols_ok:
        warnings.append("Canonical interval triplet columns are missing (pred_low/pred_mid/pred_high or prob_q_*).")
    if bool(fallback_triggered) and not str(fallback_reason or "").strip():
        warnings.append("Fallback was triggered but prob_fallback_reason is empty.")
    if mode == "historical_by_regime" and _resolve_group_column(run, ("regime", "regime_used", "regime_state", "regime_bucket")) is None:
        warnings.append("historical_by_regime should expose a realised regime label.")
    if mode == "historical_by_features" and _resolve_group_column(run, feature_groups := ("probabilistic_feature_aware_share", "prob_feature_aware_share")) is None:
        warnings.append("historical_by_features should expose probabilistic_feature_aware_share.")
    if mode == "feature_bucketed_historical" and missing_advisory_groups:
        warnings.append(f"feature_bucketed_historical lacks advisory bucket diagnostics: {', '.join(missing_advisory_groups)}.")
    if mode == "hybrid" and _resolve_group_column(run, ("prob_hybrid_qr_weight",)) is None and _resolve_group_column(run, ("prob_hybrid_knn_weight",)) is None:
        warnings.append("hybrid should expose at least one hybrid component weight diagnostic.")
    if mode == "quantile_regression" and _resolve_group_column(run, ("prob_qr_active", "qr_active")) is None:
        warnings.append("quantile_regression should expose qr_active or equivalent diagnostic.")

    semantic_status = "ok"
    if mode != "none" and (diag_rows == 0 or missing_required_groups or not accepted_prob_source_ok or not backend_family_ok or not interval_backend_ok or not semantic_required_ok or not interval_cols_ok or not backend_distinct_ok):
        semantic_status = "fail"
    elif warnings:
        semantic_status = "warn"

    row = {
        "requested_mode": requested_mode,
        "effective_mode": effective_mode,
        "prob_source": prob_source,
        "prob_backend_family": backend_family,
        "prob_backend_name": backend_name,
        "prob_backend_variant": backend_variant,
        "prob_backend_distinct": backend_distinct,
        "prob_interval_backend": interval_backend,
        "prob_backend_parent_mode": backend_parent_mode,
        "prob_fallback_triggered": fallback_triggered,
        "prob_fallback_reason": fallback_reason,
        "prob_contract_status": contract_status,
        "required_groups_ok": bool(required_groups_ok),
        "missing_required_groups": " | ".join(missing_required_groups),
        "sparse_required_groups": " | ".join(sparse_required_groups),
        "accepted_prob_source_ok": bool(accepted_prob_source_ok),
        "backend_family_ok": bool(backend_family_ok),
        "interval_backend_ok": bool(interval_backend_ok),
        "backend_distinct_ok": bool(backend_distinct_ok),
        "semantic_required_cols_ok": bool(semantic_required_ok),
        "missing_semantic_cols": " | ".join(semantic_missing_cols),
        "interval_cols_ok": bool(interval_cols_ok),
        "diagnostics_rows": diag_rows,
        "semantic_status": semantic_status,
        "warnings": " | ".join(warnings),
    }
    return pd.DataFrame([row])


def validate_probabilistic_overlay_run(run: Dict[str, Any]) -> Dict[str, Any]:
    mode = _get_probabilistic_mode(run)
    prob_source = _get_prob_source(run)
    contract = _probabilistic_mode_contract(mode, prob_source=prob_source)
    required_groups = dict(contract.get("required_groups", {}))
    optional_cols = list(contract.get("optional_cols", []))
    notes = list(contract.get("notes", []))
    accepted_prob_sources = list(contract.get("accepted_prob_sources", []))

    diag = _get_run_diagnostics_df(run)
    diag_rows = int(diag.shape[0]) if isinstance(diag, pd.DataFrame) else 0

    resolved_required_cols = {}
    missing_required_groups = []
    sparse_required_groups = []
    present_required_cols = []
    missing_required_cols = []
    non_na_rate_by_col = {}
    for group, alternatives in required_groups.items():
        resolved = _resolve_group_column(run, alternatives)
        if resolved is None:
            missing_required_groups.append(group)
            missing_required_cols.extend(list(alternatives))
            continue
        resolved_required_cols[group] = resolved
        present_required_cols.append(resolved)
        rate = _run_diag_non_na_rate(run, resolved)
        non_na_rate_by_col[resolved] = rate
        if rate < 0.25:
            sparse_required_groups.append(group)

    present_optional_cols = _run_diag_present_cols(run, optional_cols)
    missing_optional_cols = _run_diag_missing_cols(run, optional_cols)
    for col in present_optional_cols:
        non_na_rate_by_col[col] = _run_diag_non_na_rate(run, col)

    audit_df = build_probabilistic_mode_audit_table(run)
    audit_row = audit_df.iloc[0].to_dict() if not audit_df.empty else {}
    semantic_status = str(audit_row.get("semantic_status", "ok") or "ok")
    audit_warnings = [w for w in str(audit_row.get("warnings", "")).split(" | ") if str(w).strip()]

    warnings = []
    if mode != "none" and diag_rows == 0:
        warnings.append(f"Mode '{mode}' is active but diagnostics_df is empty.")
    if missing_required_groups:
        warnings.append(
            f"Mode '{mode}' is missing required diagnostics groups: {', '.join(missing_required_groups)}."
        )
    if sparse_required_groups:
        warnings.append(
            f"Mode '{mode}' has required diagnostics groups with low non-NaN coverage: {', '.join(sparse_required_groups)}."
        )
    if missing_optional_cols:
        warnings.append(
            f"Mode '{mode}' is missing optional diagnostics columns: {', '.join(missing_optional_cols)}."
        )
    if prob_source is not None and accepted_prob_sources and prob_source not in accepted_prob_sources:
        warnings.append(
            f"Mode '{mode}' produced prob_source='{prob_source}', outside accepted sources: {', '.join([str(x) for x in accepted_prob_sources if x is not None])}."
        )
    warnings.extend([w for w in audit_warnings if w not in warnings])

    status = semantic_status
    if status not in {"ok", "warn", "fail"}:
        status = "ok"
    if missing_required_groups or (mode != "none" and diag_rows == 0):
        status = "fail"
    elif status == "ok" and (sparse_required_groups or missing_optional_cols):
        status = "warn"

    return {
        "probabilistic_mode": mode,
        "prob_source": prob_source,
        "status": status,
        "diagnostics_rows": diag_rows,
        "required_groups": required_groups,
        "resolved_required_cols": resolved_required_cols,
        "missing_required_groups": missing_required_groups,
        "sparse_required_groups": sparse_required_groups,
        "accepted_prob_sources": accepted_prob_sources,
        "required_cols": list(dict.fromkeys([c for cols in required_groups.values() for c in cols])),
        "optional_cols": optional_cols,
        "present_required_cols": present_required_cols,
        "present_optional_cols": present_optional_cols,
        "missing_required_cols": sorted(set(missing_required_cols)),
        "missing_optional_cols": missing_optional_cols,
        "sparse_required_cols": [resolved_required_cols[g] for g in sparse_required_groups if g in resolved_required_cols],
        "non_na_rate_by_col": non_na_rate_by_col,
        "notes": notes,
        "warnings": warnings,
        "audit_table": audit_df,
        "requested_mode": audit_row.get("requested_mode"),
        "effective_mode": audit_row.get("effective_mode"),
        "prob_backend_family": audit_row.get("prob_backend_family"),
        "prob_backend_name": audit_row.get("prob_backend_name"),
        "prob_backend_variant": audit_row.get("prob_backend_variant"),
        "prob_backend_distinct": audit_row.get("prob_backend_distinct"),
        "prob_interval_backend": audit_row.get("prob_interval_backend"),
        "prob_backend_parent_mode": audit_row.get("prob_backend_parent_mode"),
        "prob_fallback_triggered": audit_row.get("prob_fallback_triggered"),
        "prob_fallback_reason": audit_row.get("prob_fallback_reason"),
        "prob_contract_status": audit_row.get("prob_contract_status"),
        "accepted_prob_source_ok": audit_row.get("accepted_prob_source_ok"),
        "backend_family_ok": audit_row.get("backend_family_ok"),
        "interval_backend_ok": audit_row.get("interval_backend_ok"),
        "backend_distinct_ok": audit_row.get("backend_distinct_ok"),
        "semantic_required_cols_ok": audit_row.get("semantic_required_cols_ok"),
        "interval_cols_ok": audit_row.get("interval_cols_ok"),
        "semantic_status": semantic_status,
    }



def _diag_mean_fallback(run: Dict[str, Any], *cols: str) -> float:
    for col in cols:
        v = _run_diag_mean(run, col)
        if np.isfinite(v):
            return float(v)
    return np.nan


def _run_diag_ratio_mean(run: Dict[str, Any], numerator_col: str, denominator_col: str) -> float:
    diag = run.get("diagnostics_df", pd.DataFrame())
    if not isinstance(diag, pd.DataFrame) or diag.empty:
        return np.nan
    if numerator_col not in diag.columns or denominator_col not in diag.columns:
        return np.nan
    num = pd.to_numeric(diag[numerator_col], errors="coerce")
    den = pd.to_numeric(diag[denominator_col], errors="coerce")
    ratio = num / den.replace(0.0, np.nan)
    return float(ratio.replace([np.inf, -np.inf], np.nan).mean())


def build_probabilistic_overlay_summary(run: Dict[str, Any]) -> pd.DataFrame:
    perf = _run_perf_dict(run)
    probabilistic_mode = _get_probabilistic_mode(run)
    probabilistic_mode_effective = _diag_mode_or_cfg(
        run,
        ("probabilistic_mode_effective", "probabilistic_mode"),
        ("probabilistic_mode_effective", "probabilistic_mode"),
    )
    prob_source = _get_prob_source(run)
    signal_mode_effective = _run_diag_mode_text(run, "signal_mode_effective") or _run_diag_mode_text(run, "signal_mode")
    validation = validate_probabilistic_overlay_run(run)

    rows = [
        {"metric": "probabilistic_mode", "value": probabilistic_mode},
        {"metric": "probabilistic_mode_effective", "value": probabilistic_mode_effective},
        {"metric": "prob_source", "value": prob_source},
        {"metric": "prob_backend_family", "value": validation.get("prob_backend_family")},
        {"metric": "prob_backend_name", "value": validation.get("prob_backend_name")},
        {"metric": "prob_backend_variant", "value": validation.get("prob_backend_variant")},
        {"metric": "prob_backend_distinct", "value": validation.get("prob_backend_distinct")},
        {"metric": "prob_interval_backend", "value": validation.get("prob_interval_backend")},
        {"metric": "prob_backend_parent_mode", "value": validation.get("prob_backend_parent_mode")},
        {"metric": "prob_fallback_triggered", "value": validation.get("prob_fallback_triggered")},
        {"metric": "prob_fallback_reason", "value": validation.get("prob_fallback_reason")},
        {"metric": "prob_contract_status", "value": validation.get("prob_contract_status")},
        {"metric": "signal_mode_effective", "value": signal_mode_effective},
        {"metric": "probabilistic_validation_status", "value": validation.get("status")},
        {"metric": "probabilistic_semantic_status", "value": validation.get("semantic_status")},
        {"metric": "probabilistic_backend_family_ok", "value": validation.get("backend_family_ok")},
        {"metric": "probabilistic_interval_backend_ok", "value": validation.get("interval_backend_ok")},
        {"metric": "probabilistic_backend_distinct_ok", "value": validation.get("backend_distinct_ok")},
        {"metric": "probabilistic_interval_cols_ok", "value": validation.get("interval_cols_ok")},
        {"metric": "probabilistic_accepted_prob_source_ok", "value": validation.get("accepted_prob_source_ok")},
        {"metric": "probabilistic_warning_count", "value": int(len(validation.get("warnings", [])))},
        {"metric": "probabilistic_missing_required_count", "value": int(len(validation.get("missing_required_groups", [])))},
        {"metric": "probabilistic_missing_optional_count", "value": int(len(validation.get("missing_optional_cols", [])))},
        {"metric": "probabilistic_sparse_required_count", "value": int(len(validation.get("sparse_required_groups", [])))},
        {"metric": "cap_governance_label", "value": _build_cap_governance_label(run)},
        {"metric": "cagr", "value": _safe_float(perf.get("cagr"))},
        {"metric": "annual_volatility", "value": _safe_float(perf.get("annual_volatility", perf.get("annualized_volatility")))},
        {"metric": "sharpe", "value": _safe_float(perf.get("sharpe"))},
        {"metric": "max_drawdown", "value": _safe_float(perf.get("max_drawdown"))},
        {"metric": "mean_turnover", "value": _safe_float(perf.get("mean_turnover"))},
        {"metric": "active_return_annual", "value": _safe_float(perf.get("active_return_annual"))},
        {"metric": "tracking_error_annual", "value": _safe_float(perf.get("tracking_error_annual"))},
        {"metric": "information_ratio", "value": _safe_float(perf.get("information_ratio"))},
        {"metric": "probabilistic_active_rate", "value": _diag_mean_fallback(run, "probabilistic_active", "prob_active")},
        {"metric": "probabilistic_interval_width_mean", "value": _diag_mean_fallback(run, "probabilistic_interval_width_mean", "prob_interval_width", "interval_width")},
        {"metric": "probabilistic_downside_mean", "value": _diag_mean_fallback(run, "probabilistic_downside_mean", "prob_downside", "downside")},
        {"metric": "probabilistic_confidence_mean", "value": _diag_mean_fallback(run, "probabilistic_confidence_mean", "prob_confidence", "confidence")},
        {"metric": "probabilistic_mu_penalty_mean_abs", "value": _diag_mean_fallback(run, "probabilistic_mu_penalty_mean_abs", "prob_mu_penalty_mean_abs")},
        {"metric": "probabilistic_mu_penalty_max_abs", "value": _diag_mean_fallback(run, "probabilistic_mu_penalty_max_abs", "prob_mu_penalty_max_abs")},
        {"metric": "probabilistic_feature_aware_share", "value": _diag_mean_fallback(run, "probabilistic_feature_aware_share", "prob_feature_aware_share")},
        {"metric": "probabilistic_feature_match_n_mean", "value": _diag_mean_fallback(run, "probabilistic_feature_match_n_mean", "prob_feature_match_n", "feature_match_n")},
        {"metric": "probabilistic_feature_distance_mean", "value": _diag_mean_fallback(run, "probabilistic_feature_distance_mean", "prob_feature_distance", "feature_distance")},
        {"metric": "requested_asset_weight_cap_mean", "value": _run_diag_mean(run, "requested_asset_weight_cap")},
        {"metric": "requested_w_cap_mean", "value": _run_diag_mean(run, "requested_w_cap")},
        {"metric": "effective_cap_base_mean", "value": _diag_mean_fallback(run, "base_requested_weight_cap", "effective_cap_base_mean")},
        {"metric": "effective_cap_final_mean", "value": _diag_mean_fallback(run, "effective_requested_weight_cap", "effective_cap_final_mean")},
        {"metric": "dispersion_risk_model_enabled_rate", "value": _diag_mean_fallback(run, "dispersion_risk_model_enabled")},
        {"metric": "dispersion_risk_model_active_rate", "value": _diag_mean_fallback(run, "dispersion_risk_model_active")},
        {"metric": "dispersion_risk_model_mult_mean", "value": _diag_mean_fallback(run, "dispersion_risk_model_mult")},
        {"metric": "dispersion_risk_model_z_mean", "value": _diag_mean_fallback(run, "dispersion_risk_model_z")},
        {"metric": "dispersion_risk_model_trace_ratio_mean", "value": _run_diag_ratio_mean(run, "dispersion_risk_model_cov_trace_after", "dispersion_risk_model_cov_trace_before")},
        {"metric": "dispersion_risk_model_diag_ratio_mean", "value": _run_diag_ratio_mean(run, "dispersion_risk_model_diag_mean_after", "dispersion_risk_model_diag_mean_before")},
        {"metric": "dispersion_risk_model_apply_target_mode", "value": _run_diag_mode_text(run, "dispersion_risk_model_apply_target")},
        {"metric": "dispersion_risk_model_reason_mode", "value": _run_diag_mode_text(run, "dispersion_risk_model_reason")},
    ]
    rows.extend(_build_overlay_feature_mu_rows(run))
    rows.extend(_build_overlay_regime_universe_rows(run))
    return pd.DataFrame(rows)


def _performance_dict_to_table_row(name: str, perf: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": str(name),
        "cagr": perf.get("cagr"),
        "sharpe": perf.get("sharpe"),
        "max_drawdown": perf.get("max_drawdown"),
        "annual_volatility": perf.get("annual_volatility", perf.get("annualized_volatility")),
        "mean_turnover": perf.get("mean_turnover"),
        "active_return_annual": perf.get("active_return_annual"),
        "tracking_error_annual": perf.get("tracking_error_annual"),
        "information_ratio": perf.get("information_ratio"),
    }


def compare_probabilistic_overlay_runs(
    base_run: Dict[str, Any],
    overlay_run: Dict[str, Any],
    *,
    base_name: str = "none",
    overlay_name: str = "overlay",
) -> Dict[str, Any]:
    """Economically validate a probabilistic overlay against a no-overlay baseline,
    while also validating mode-specific diagnostics coverage.
    """
    base_validation = validate_probabilistic_overlay_run(base_run)
    overlay_validation = validate_probabilistic_overlay_run(overlay_run)

    base_metric_summary = build_probabilistic_overlay_summary(base_run).rename(columns={"value": base_name})
    overlay_metric_summary = build_probabilistic_overlay_summary(overlay_run).rename(columns={"value": overlay_name})
    comparison = base_metric_summary.merge(overlay_metric_summary, on="metric", how="outer")

    base_num = pd.to_numeric(comparison[base_name], errors="coerce")
    overlay_num = pd.to_numeric(comparison[overlay_name], errors="coerce")
    comparison["delta"] = overlay_num - base_num
    comparison["delta_pct"] = comparison["delta"] / base_num.replace(0.0, np.nan).abs()

    better_when = {
        "cagr": "higher",
        "sharpe": "higher",
        "information_ratio": "higher",
        "active_return_annual": "higher",
        "annual_volatility": "lower",
        "max_drawdown": "higher",
        "mean_turnover": "lower",
        "tracking_error_annual": "lower",
        "probabilistic_interval_width_mean": "lower",
        "probabilistic_downside_mean": "lower",
        "probabilistic_confidence_mean": "higher",
        "probabilistic_mu_penalty_mean_abs": "lower",
        "probabilistic_mu_penalty_max_abs": "lower",
        "probabilistic_feature_aware_share": "higher",
        "probabilistic_feature_match_n_mean": "higher",
        "probabilistic_feature_distance_mean": "lower",
        "probabilistic_active_rate": "higher",
        "feature_mu_active_share": "higher",
        "feature_mu_match_n_mean": "higher",
        "feature_mu_cols_used_mean": "neutral",
        "feature_mu_distance_mean": "lower",
        "feature_mu_abs_tilt_mean": "neutral",
        "feature_mu_abs_tilt_max": "neutral",
        "regime_universe_enabled_rate": "higher",
        "regime_universe_keep_frac_mean": "neutral",
        "regime_universe_keep_frac_low_mean": "neutral",
        "regime_universe_keep_frac_mid_mean": "neutral",
        "regime_universe_keep_frac_high_mean": "neutral",
        "regime_universe_selected_assets_mean": "neutral",
        "regime_universe_filtered_assets_mean": "neutral",
        "regime_universe_low_offensive_vol_tilt_mean": "neutral",
        "regime_universe_high_defensive_vol_tilt_mean": "neutral",
        "regime_universe_min_assets_mean": "neutral",
        "regime_universe_max_assets_mean": "neutral",
        "requested_asset_weight_cap_mean": "lower",
        "requested_w_cap_mean": "lower",
        "effective_cap_base_mean": "lower",
        "effective_cap_final_mean": "lower",
        "dispersion_risk_model_enabled_rate": "higher",
        "dispersion_risk_model_active_rate": "higher",
        "dispersion_risk_model_mult_mean": "lower",
        "dispersion_risk_model_z_mean": "neutral",
        "dispersion_risk_model_trace_ratio_mean": "lower",
        "dispersion_risk_model_diag_ratio_mean": "lower",
    }
    economic_note = {
        "cagr": "Compound growth improved if positive.",
        "sharpe": "Risk-adjusted performance improved if positive.",
        "max_drawdown": "Less negative is better, so a positive delta is good.",
        "annual_volatility": "Lower realised volatility is usually preferable.",
        "mean_turnover": "Lower turnover means lower implementation drag.",
        "information_ratio": "Higher active return per unit of tracking error is better.",
        "active_return_annual": "Higher active return improves economic value vs benchmark.",
        "tracking_error_annual": "Lower tracking error means the overlay is not adding as much instability.",
        "probabilistic_interval_width_mean": "Narrower intervals imply more concentrated probabilistic conviction.",
        "probabilistic_downside_mean": "Lower estimated downside is preferable.",
        "probabilistic_confidence_mean": "Higher confidence scale implies stronger overlay conviction.",
        "probabilistic_mu_penalty_mean_abs": "Lower signal penalty means the overlay is less aggressively distorting the base signal.",
        "probabilistic_feature_aware_share": "Higher share means the overlay is using feature-aware conditioning more often.",
        "probabilistic_mode_effective": "Descriptive: realised backend probabilistic mode actually used by the overlay.",
        "feature_mu_active_share": "Higher share means feature-conditioned mu was active in more rebalance windows.",
        "feature_mu_match_n_mean": "Higher neighbour count means feature-conditioned mu had a deeper local analogue set on average.",
        "feature_mu_cols_used_mean": "Descriptive: average number of feature columns used by the feature-conditioned mu block.",
        "feature_mu_distance_mean": "Lower distance implies feature-conditioned mu is drawing from closer analogues.",
        "feature_mu_abs_tilt_mean": "Descriptive: average absolute tilt applied by feature-conditioned mu versus the base mu.",
        "feature_mu_abs_tilt_max": "Descriptive: upper-tail absolute tilt applied by feature-conditioned mu across rebalance windows.",
        "regime_universe_enabled_rate": "Higher enabled rate means regime-dependent universe filtering was switched on more often.",
        "regime_universe_keep_frac_mean": "Descriptive: average keep fraction implied by the regime-dependent universe settings.",
        "regime_universe_keep_frac_low_mean": "Descriptive: keep fraction applied in the low-risk regime state.",
        "regime_universe_keep_frac_mid_mean": "Descriptive: keep fraction applied in the mid-risk regime state.",
        "regime_universe_keep_frac_high_mean": "Descriptive: keep fraction applied in the high-risk regime state.",
        "regime_universe_selected_assets_mean": "Descriptive: average number of assets kept after regime-dependent universe filtering.",
        "regime_universe_filtered_assets_mean": "Descriptive: average number of assets removed by regime-dependent universe filtering.",
        "regime_universe_selection_metric_mode": "Descriptive: dominant ranking metric used by the regime-dependent universe block.",
        "regime_universe_regime_mode": "Descriptive: dominant realised regime label seen by the run.",
        "regime_universe_low_offensive_vol_tilt_mean": "Descriptive: offensive volatility tilt configured for low-risk regime selection.",
        "regime_universe_high_defensive_vol_tilt_mean": "Descriptive: defensive volatility tilt configured for high-risk regime selection.",
        "regime_universe_min_assets_mean": "Descriptive: minimum asset floor imposed by regime-dependent universe selection.",
        "regime_universe_max_assets_mean": "Descriptive: maximum asset ceiling imposed by regime-dependent universe selection.",
        "requested_asset_weight_cap_mean": "Lower requested asset_weight_cap means tighter per-asset governance.",
        "requested_w_cap_mean": "Lower requested w_cap means tighter direct weight governance.",
        "effective_cap_base_mean": "Lower base requested cap means the strategy is governed by a stricter requested cap before adaptives.",
        "effective_cap_final_mean": "Lower final effective cap means adaptive governance tightened the position limit more strongly.",
        "dispersion_risk_model_enabled_rate": "Higher enabled rate means the configuration actually exposed the dispersion risk model more often.",
        "dispersion_risk_model_active_rate": "Higher active rate means the dispersion risk model was live in more rebalance windows.",
        "dispersion_risk_model_mult_mean": "Lower mean multiplier implies stronger average risk tightening; above 1 implies loosening.",
        "dispersion_risk_model_z_mean": "The mean z-score is descriptive: positive values indicate recent dispersion above history.",
        "dispersion_risk_model_trace_ratio_mean": "Lower trace ratio means the covariance risk model was tightened on average after the dispersion adjustment.",
        "dispersion_risk_model_diag_ratio_mean": "Lower diagonal ratio means average asset variances were tightened after the dispersion adjustment.",
    }
    comparison["better_when"] = comparison["metric"].map(better_when)
    comparison["economic_note"] = comparison["metric"].map(economic_note)

    def _direction(metric: str, delta: float) -> str:
        rule = better_when.get(str(metric))
        if not np.isfinite(delta) or rule is None or abs(float(delta)) <= 1e-12:
            return "neutral"
        if rule == "higher":
            return "improved" if float(delta) > 0 else "worsened"
        if rule == "lower":
            return "improved" if float(delta) < 0 else "worsened"
        if rule == "neutral":
            return "descriptive"
        return "neutral"

    comparison["direction"] = [
        _direction(metric, delta)
        for metric, delta in zip(comparison["metric"], pd.to_numeric(comparison["delta"], errors="coerce"))
    ]

    focus_metrics = [
        "cagr",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "mean_turnover",
        "active_return_annual",
        "tracking_error_annual",
        "information_ratio",
        "probabilistic_interval_width_mean",
        "probabilistic_downside_mean",
        "probabilistic_confidence_mean",
        "probabilistic_mu_penalty_mean_abs",
        "probabilistic_feature_aware_share",
        "effective_cap_base_mean",
        "effective_cap_final_mean",
        "requested_asset_weight_cap_mean",
        "requested_w_cap_mean",
        "dispersion_risk_model_active_rate",
        "dispersion_risk_model_mult_mean",
        "dispersion_risk_model_trace_ratio_mean",
        "dispersion_risk_model_diag_ratio_mean",
        "dispersion_risk_model_reason_mode",
        "dispersion_risk_model_apply_target_mode",
        "probabilistic_warning_count",
        "probabilistic_missing_required_count",
        "probabilistic_sparse_required_count",
    ]
    economic_focus_metrics = [
        "cagr",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "mean_turnover",
        "active_return_annual",
        "tracking_error_annual",
        "information_ratio",
    ]

    focus_df = comparison[comparison["metric"].isin(focus_metrics)].copy().reset_index(drop=True)
    econ_focus_df = comparison[comparison["metric"].isin(economic_focus_metrics)].copy().reset_index(drop=True)

    improved_count = int((econ_focus_df["direction"] == "improved").sum()) if not econ_focus_df.empty else 0
    worsened_count = int((econ_focus_df["direction"] == "worsened").sum()) if not econ_focus_df.empty else 0

    delta_lookup = dict(zip(comparison["metric"], pd.to_numeric(comparison["delta"], errors="coerce")))
    delta_sharpe = _safe_float(delta_lookup.get("sharpe"))
    delta_cagr = _safe_float(delta_lookup.get("cagr"))
    delta_turnover = _safe_float(delta_lookup.get("mean_turnover"))
    delta_max_drawdown = _safe_float(delta_lookup.get("max_drawdown"))

    validation_rows = []
    for run_name, validation in [(base_name, base_validation), (overlay_name, overlay_validation)]:
        validation_rows.append({
            "run_name": run_name,
            "probabilistic_mode": validation.get("probabilistic_mode"),
            "status": validation.get("status"),
            "diagnostics_rows": validation.get("diagnostics_rows"),
            "missing_required_count": len(validation.get("missing_required_groups", [])),
            "missing_optional_count": len(validation.get("missing_optional_cols", [])),
            "sparse_required_count": len(validation.get("sparse_required_cols", [])),
            "warnings": " | ".join(validation.get("warnings", [])),
            "prob_source": validation.get("prob_source"),
            "prob_backend_family": validation.get("prob_backend_family"),
            "prob_backend_distinct": validation.get("prob_backend_distinct"),
            "prob_interval_backend": validation.get("prob_interval_backend"),
            "semantic_status": validation.get("semantic_status"),
            "backend_family_ok": validation.get("backend_family_ok"),
            "interval_backend_ok": validation.get("interval_backend_ok"),
            "backend_distinct_ok": validation.get("backend_distinct_ok"),
            "accepted_prob_source_ok": validation.get("accepted_prob_source_ok"),
            "interval_cols_ok": validation.get("interval_cols_ok"),
            "notes": " | ".join(validation.get("notes", [])),
        })
    validation_summary = pd.DataFrame(validation_rows)

    validation_warnings = []
    validation_warnings.extend([f"{base_name}: {w}" for w in base_validation.get("warnings", [])])
    validation_warnings.extend([f"{overlay_name}: {w}" for w in overlay_validation.get("warnings", [])])

    if np.isfinite(delta_sharpe) and delta_sharpe > 0 and improved_count >= worsened_count:
        headline = f"Overlay validation looks positive: {overlay_name} improved Sharpe vs {base_name}."
    elif np.isfinite(delta_sharpe) and delta_sharpe < 0 and worsened_count > improved_count:
        headline = f"Overlay validation looks weak: {overlay_name} worsened Sharpe vs {base_name}."
    else:
        headline = f"Overlay validation is mixed: {overlay_name} changes the economic profile but without a clean dominant win."

    if overlay_validation.get("status") == "fail":
        headline = f"{headline} Mode-aware diagnostics validation failed for {overlay_name}."
    elif overlay_validation.get("status") == "warn":
        headline = f"{headline} Mode-aware diagnostics validation raised warnings for {overlay_name}."

    decision_summary = {
        "base_name": str(base_name),
        "overlay_name": str(overlay_name),
        "headline": headline,
        "base_cap_governance_label": _build_cap_governance_label(base_run),
        "overlay_cap_governance_label": _build_cap_governance_label(overlay_run),
        "improved_metric_count": improved_count,
        "worsened_metric_count": worsened_count,
        "delta_sharpe": delta_sharpe,
        "delta_cagr": delta_cagr,
        "delta_turnover": delta_turnover,
        "delta_max_drawdown": delta_max_drawdown,
        "base_validation_status": base_validation.get("status"),
        "overlay_validation_status": overlay_validation.get("status"),
        "validation_warning_count": len(validation_warnings),
    }

    base_perf = _run_perf_dict(base_run)
    overlay_perf = _run_perf_dict(overlay_run)
    overlay_summary_base = pd.DataFrame([_performance_dict_to_table_row(base_name, base_perf)])
    overlay_summary_overlay = pd.DataFrame([_performance_dict_to_table_row(overlay_name, overlay_perf)])

    return {
        "overlay_summary_base": overlay_summary_base,
        "overlay_summary_overlay": overlay_summary_overlay,
        "overlay_metric_summary_base": base_metric_summary,
        "overlay_metric_summary_overlay": overlay_metric_summary,
        "overlay_comparison": comparison,
        "overlay_focus": focus_df,
        "overlay_validation_base": base_validation,
        "overlay_validation_overlay": overlay_validation,
        "overlay_validation_summary": validation_summary,
        "overlay_validation_audit_base": base_validation.get("audit_table", pd.DataFrame()),
        "overlay_validation_audit_overlay": overlay_validation.get("audit_table", pd.DataFrame()),
        "overlay_validation_warnings": validation_warnings,
        "decision_summary": decision_summary,
        "economic_delta": {
            "delta_sharpe": delta_sharpe,
            "delta_cagr": delta_cagr,
            "delta_turnover": delta_turnover,
            "delta_max_drawdown": delta_max_drawdown,
        },
    }


# ============================================================
# Alpha sweep helpers
# ============================================================

def run_alpha_sweep(
    panel_df: pd.DataFrame,
    *,
    base_cfg_payload: Dict[str, Any],
    alpha_values: Sequence[float],
) -> pd.DataFrame:
    """Run a research sweep over sigma_power_alpha.

    Parameters
    ----------
    panel_df
        Asset panel consumed by run_micro_investment_pipeline.
    base_cfg_payload
        Dictionary payload compatible with MicroPipelineConfig / config_from_dict.
    alpha_values
        Iterable of candidate sigma_power_alpha values.
    """
    if panel_df is None or not isinstance(panel_df, pd.DataFrame) or panel_df.empty:
        return pd.DataFrame()

    try:
        from src.investment import (
            MicroPipelineConfig,
            config_from_dict,
            config_fingerprint,
            run_micro_investment_pipeline,
        )
    except Exception:
        # fallback for local non-package execution
        from investment import (
            MicroPipelineConfig,
            config_from_dict,
            config_fingerprint,
            run_micro_investment_pipeline,
        )

    values: List[float] = []
    seen = set()
    for a in alpha_values or []:
        aa = _safe_float(a)
        if not np.isfinite(aa):
            continue
        key = round(float(aa), 10)
        if key in seen:
            continue
        seen.add(key)
        values.append(float(aa))
    if not values:
        values = [1.0]

    valid_fields = {f.name for f in MicroPipelineConfig.__dataclass_fields__.values()}
    payload0 = {k: v for k, v in dict(base_cfg_payload or {}).items() if k in valid_fields}
    rows: List[Dict[str, Any]] = []

    for alpha in values:
        local = dict(payload0)
        local["sigma_power_alpha"] = float(alpha)
        cfg = config_from_dict(local)
        run = run_micro_investment_pipeline(panel_df, cfg=cfg)
        perf = run.get("performance_summary", {}) or {}
        risk = run.get("risk_summary", {}) or {}
        div = run.get("diversification_summary", {}) or {}
        uni = run.get("universe_summary", {}) or {}
        rows.append({
            "sigma_power_alpha": float(alpha),
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

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["sigma_power_alpha"] = pd.to_numeric(out["sigma_power_alpha"], errors="coerce")
    sort_cols = [c for c in ["sharpe", "information_ratio", "cagr", "sigma_power_alpha"] if c in out.columns]
    ascending = [False, False, False, True][:len(sort_cols)] if sort_cols else True
    return out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


# ============================================================
# Sharpe / turnover frontier helpers
# ============================================================

def build_sharpe_turnover_frontier(
    df: pd.DataFrame,
    *,
    sharpe_col: str = "sharpe",
    turnover_col: str = "mean_turnover",
) -> pd.DataFrame:
    """Annotate a simple 2D non-dominated Sharpe-vs-turnover frontier.

    Higher Sharpe is preferred and lower turnover is preferred. The function
    coerces both columns to numeric, drops invalid rows, identifies frontier
    points by scanning from low to high turnover, and adds light diagnostics so
    downstream reporting can distinguish frontier members from dominated points.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    if sharpe_col not in df.columns or turnover_col not in df.columns:
        return pd.DataFrame()

    out = df.copy()
    out[sharpe_col] = pd.to_numeric(out[sharpe_col], errors="coerce")
    out[turnover_col] = pd.to_numeric(out[turnover_col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=[sharpe_col, turnover_col]).reset_index(drop=True)
    if out.empty:
        return out

    # Stable ordering matters: among equal turnover, higher Sharpe should appear
    # first so only the best point at that turnover can enter the frontier.
    out = out.sort_values([turnover_col, sharpe_col], ascending=[True, False], kind="mergesort").reset_index(drop=True)

    best_sharpe_so_far = -np.inf
    frontier_turnover_best = np.inf
    on_frontier: List[bool] = []
    frontier_rank: List[float] = []
    frontier_best_sharpe_so_far: List[float] = []
    sharpe_gap_to_frontier: List[float] = []
    turnover_gap_to_frontier: List[float] = []
    dominance_flag: List[str] = []
    rank = 0
    eps = 1e-12

    for _, row in out.iterrows():
        s = float(row[sharpe_col])
        t = float(row[turnover_col])
        is_frontier = s > best_sharpe_so_far + eps
        if is_frontier:
            best_sharpe_so_far = s
            frontier_turnover_best = t
            rank += 1
            on_frontier.append(True)
            frontier_rank.append(float(rank))
            sharpe_gap_to_frontier.append(0.0)
            turnover_gap_to_frontier.append(0.0)
            dominance_flag.append("frontier")
        else:
            on_frontier.append(False)
            frontier_rank.append(np.nan)
            sharpe_gap_to_frontier.append(float(best_sharpe_so_far - s) if np.isfinite(best_sharpe_so_far) else np.nan)
            turnover_gap_to_frontier.append(float(t - frontier_turnover_best) if np.isfinite(frontier_turnover_best) else np.nan)
            dominance_flag.append("dominated")
        frontier_best_sharpe_so_far.append(float(best_sharpe_so_far) if np.isfinite(best_sharpe_so_far) else np.nan)

    out["on_frontier"] = on_frontier
    out["frontier_rank"] = frontier_rank
    out["frontier_best_sharpe_so_far"] = frontier_best_sharpe_so_far
    out["sharpe_gap_to_frontier"] = sharpe_gap_to_frontier
    out["turnover_gap_to_frontier"] = turnover_gap_to_frontier
    out["frontier_status"] = dominance_flag
    out["frontier_n_points"] = int(sum(on_frontier))

    # User-facing default ordering: strongest Sharpe first, lower turnover as
    # tie-break, while preserving frontier annotations.
    return out.sort_values(["on_frontier", sharpe_col, turnover_col], ascending=[False, False, True], kind="mergesort").reset_index(drop=True)


def choose_sharpe_turnover_candidate(
    df: pd.DataFrame,
    *,
    sharpe_col: str = "sharpe",
    turnover_col: str = "mean_turnover",
    max_turnover: Optional[float] = None,
) -> Tuple[Optional[pd.Series], pd.DataFrame]:
    """Choose the best candidate by maximum Sharpe, optionally subject to turnover cap."""
    frontier = build_sharpe_turnover_frontier(df, sharpe_col=sharpe_col, turnover_col=turnover_col)
    if frontier.empty:
        return None, frontier
    eligible = frontier.copy()
    if max_turnover is not None:
        eligible = eligible[pd.to_numeric(eligible[turnover_col], errors="coerce") <= float(max_turnover)].copy()
    if eligible.empty:
        eligible = frontier.copy()
    eligible = eligible.sort_values([sharpe_col, turnover_col], ascending=[False, True]).reset_index(drop=True)
    return eligible.iloc[0], frontier


# ============================================================
# Plateau / robust-region helpers
# ============================================================

def build_plateau_region_table(
    df: pd.DataFrame,
    *,
    sharpe_col: str = "sharpe",
    turnover_col: str = "mean_turnover",
    sharpe_rel_tol: float = 0.95,
    max_turnover: Optional[float] = None,
) -> pd.DataFrame:
    """Flag a robust plateau region around the best Sharpe candidate.

    A row is considered part of the plateau if:
    - sharpe >= sharpe_rel_tol * best_sharpe
    - and optionally turnover <= max_turnover
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out[sharpe_col] = pd.to_numeric(out[sharpe_col], errors="coerce")
    if turnover_col in out.columns:
        out[turnover_col] = pd.to_numeric(out[turnover_col], errors="coerce")
    out = out.dropna(subset=[sharpe_col]).reset_index(drop=True)
    if out.empty:
        return out

    best_sharpe = float(out[sharpe_col].max())
    sharpe_cut = float(best_sharpe) * float(sharpe_rel_tol)

    plateau_mask = out[sharpe_col] >= sharpe_cut
    if max_turnover is not None and turnover_col in out.columns:
        plateau_mask = plateau_mask & (out[turnover_col] <= float(max_turnover))

    out["plateau_region"] = plateau_mask.astype(bool)
    out["plateau_sharpe_cut"] = sharpe_cut
    out["best_sharpe"] = best_sharpe
    return out


def summarize_plateau_region(
    df: pd.DataFrame,
    *,
    sharpe_col: str = "sharpe",
    turnover_col: str = "mean_turnover",
    sharpe_rel_tol: float = 0.95,
    max_turnover: Optional[float] = None,
) -> pd.DataFrame:
    """Summarise the robust plateau region from a parameter grid."""
    flagged = build_plateau_region_table(
        df,
        sharpe_col=sharpe_col,
        turnover_col=turnover_col,
        sharpe_rel_tol=sharpe_rel_tol,
        max_turnover=max_turnover,
    )
    if flagged.empty:
        return pd.DataFrame(columns=["metric", "value"])

    plateau = flagged[flagged["plateau_region"] == True].copy()
    if plateau.empty:
        return pd.DataFrame([
            {"metric": "best_sharpe", "value": float(flagged[sharpe_col].max())},
            {"metric": "plateau_size", "value": 0},
            {"metric": "plateau_exists", "value": 0},
        ])

    rows = [
        {"metric": "best_sharpe", "value": float(flagged[sharpe_col].max())},
        {"metric": "plateau_size", "value": int(len(plateau))},
        {"metric": "plateau_exists", "value": 1},
        {"metric": "plateau_sharpe_cut", "value": float(plateau["plateau_sharpe_cut"].iloc[0])},
    ]

    for col in ["temperature", "weight_shrink", "inertia", sharpe_col, turnover_col]:
        if col in plateau.columns:
            vals = pd.to_numeric(plateau[col], errors="coerce").dropna()
            if not vals.empty:
                rows.extend([
                    {"metric": f"{col}_min", "value": float(vals.min())},
                    {"metric": f"{col}_max", "value": float(vals.max())},
                    {"metric": f"{col}_mean", "value": float(vals.mean())},
                ])
    return pd.DataFrame(rows)


def choose_plateau_candidate(
    df: pd.DataFrame,
    *,
    sharpe_col: str = "sharpe",
    turnover_col: str = "mean_turnover",
    sharpe_rel_tol: float = 0.95,
    max_turnover: Optional[float] = None,
) -> Tuple[Optional[pd.Series], pd.DataFrame]:
    """Choose a stable candidate from the plateau region.

    Rule:
    - build plateau region
    - prefer minimum turnover inside plateau
    - tie-break by higher Sharpe
    """
    flagged = build_plateau_region_table(
        df,
        sharpe_col=sharpe_col,
        turnover_col=turnover_col,
        sharpe_rel_tol=sharpe_rel_tol,
        max_turnover=max_turnover,
    )
    if flagged.empty:
        return None, flagged

    plateau = flagged[flagged["plateau_region"] == True].copy()
    if plateau.empty:
        flagged = flagged.sort_values([sharpe_col, turnover_col], ascending=[False, True]).reset_index(drop=True)
        return flagged.iloc[0], flagged

    if turnover_col in plateau.columns:
        plateau = plateau.sort_values([turnover_col, sharpe_col], ascending=[True, False]).reset_index(drop=True)
    else:
        plateau = plateau.sort_values([sharpe_col], ascending=[False]).reset_index(drop=True)
    return plateau.iloc[0], flagged


# ============================================================
# Nested validation helpers
# ============================================================

def split_panel_for_nested_validation(
    panel: pd.DataFrame,
    *,
    date_col: str = "date",
    tuning_fraction: float = 0.60,
    min_tuning_periods: int = 36,
    min_test_periods: int = 12,
) -> Dict[str, Any]:
    """Chronologically split a long panel into tuning and test subsets.

    Split is performed on unique dates, not raw rows, to preserve the cross-sectional panel structure.
    """
    if panel is None or not isinstance(panel, pd.DataFrame) or panel.empty:
        raise ValueError("panel must be a non-empty DataFrame")
    if date_col not in panel.columns:
        raise KeyError(f"panel missing required date column: {date_col}")

    out = panel.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col]).sort_values([date_col]).reset_index(drop=True)
    unique_dates = pd.Series(out[date_col].drop_duplicates().sort_values().tolist(), dtype="datetime64[ns]")

    n_dates = int(unique_dates.shape[0])
    if n_dates < max(int(min_tuning_periods) + int(min_test_periods), 8):
        raise ValueError(
            f"Not enough unique dates for nested validation: got {n_dates}, "
            f"need at least {int(min_tuning_periods) + int(min_test_periods)}"
        )

    raw_split = int(np.floor(float(tuning_fraction) * n_dates))
    split_idx = max(int(min_tuning_periods), raw_split)
    split_idx = min(split_idx, n_dates - int(min_test_periods))
    if split_idx <= 0 or split_idx >= n_dates:
        raise ValueError("Could not produce a valid chronological tuning/test split")

    split_date = pd.Timestamp(unique_dates.iloc[split_idx - 1])
    tune_dates = set(unique_dates.iloc[:split_idx].tolist())
    test_dates = set(unique_dates.iloc[split_idx:].tolist())

    tuning_panel = out[out[date_col].isin(tune_dates)].copy().reset_index(drop=True)
    test_panel = out[out[date_col].isin(test_dates)].copy().reset_index(drop=True)

    return {
        "tuning_panel": tuning_panel,
        "test_panel": test_panel,
        "split_date": split_date,
        "n_tuning_dates": int(len(tune_dates)),
        "n_test_dates": int(len(test_dates)),
        "tuning_start": pd.Timestamp(min(tune_dates)),
        "tuning_end": pd.Timestamp(max(tune_dates)),
        "test_start": pd.Timestamp(min(test_dates)),
        "test_end": pd.Timestamp(max(test_dates)),
    }


def build_nested_validation_comparison(
    *,
    untuned_test_run: Dict[str, Any],
    tuned_test_run: Dict[str, Any],
    tuning_choice: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Simple comparison table for nested-validation test results."""
    left = (untuned_test_run or {}).get("performance_summary", {}) or {}
    right = (tuned_test_run or {}).get("performance_summary", {}) or {}

    rows = [
        {
            "metric": "cagr",
            "untuned_test": left.get("cagr"),
            "tuned_test": right.get("cagr"),
            "delta": _safe_float(right.get("cagr")) - _safe_float(left.get("cagr")),
        },
        {
            "metric": "sharpe",
            "untuned_test": left.get("sharpe"),
            "tuned_test": right.get("sharpe"),
            "delta": _safe_float(right.get("sharpe")) - _safe_float(left.get("sharpe")),
        },
        {
            "metric": "max_drawdown",
            "untuned_test": left.get("max_drawdown"),
            "tuned_test": right.get("max_drawdown"),
            "delta": _safe_float(right.get("max_drawdown")) - _safe_float(left.get("max_drawdown")),
        },
        {
            "metric": "mean_turnover",
            "untuned_test": left.get("mean_turnover"),
            "tuned_test": right.get("mean_turnover"),
            "delta": _safe_float(right.get("mean_turnover")) - _safe_float(left.get("mean_turnover")),
        },
        {
            "metric": "information_ratio",
            "untuned_test": left.get("information_ratio"),
            "tuned_test": right.get("information_ratio"),
            "delta": _safe_float(right.get("information_ratio")) - _safe_float(left.get("information_ratio")),
        },
    ]
    if tuning_choice is not None:
        for k in ["temperature", "weight_shrink", "inertia", "sharpe", "mean_turnover"]:
            if k in tuning_choice:
                rows.append({"metric": f"tuning_choice.{k}", "untuned_test": np.nan, "tuned_test": tuning_choice.get(k), "delta": np.nan})
    return pd.DataFrame(rows)


# ============================================================
# Bayesian tuning helpers
# ============================================================

def summarize_bayesian_tuning_results(
    trials_df: pd.DataFrame,
    *,
    objective_col: str = "objective_value",
) -> pd.DataFrame:
    """Summarise Optuna/random-search style tuning results."""
    if trials_df is None or not isinstance(trials_df, pd.DataFrame) or trials_df.empty:
        return pd.DataFrame(columns=["metric", "value"])

    out = trials_df.copy()
    if objective_col in out.columns:
        out[objective_col] = pd.to_numeric(out[objective_col], errors="coerce")
    rows = [{"metric": "n_trials", "value": int(len(out))}]

    if objective_col in out.columns and out[objective_col].notna().any():
        vals = pd.to_numeric(out[objective_col], errors="coerce").dropna()
        rows.extend([
            {"metric": "best_objective", "value": float(vals.max())},
            {"metric": "mean_objective", "value": float(vals.mean())},
            {"metric": "median_objective", "value": float(vals.median())},
            {"metric": "objective_std", "value": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0},
        ])

    for col in ["sharpe", "cagr", "mean_turnover", "max_drawdown"]:
        if col in out.columns and pd.to_numeric(out[col], errors="coerce").notna().any():
            vals = pd.to_numeric(out[col], errors="coerce").dropna()
            rows.append({"metric": f"best_{col}", "value": float(vals.max() if col != 'max_drawdown' else vals.max())})

    return pd.DataFrame(rows)

# ============================================================
# Model selection based on ranking
# ============================================================

def build_model_selection_ranking_table(
    candidates_df: pd.DataFrame,
    *,
    higher_is_better: Optional[Sequence[str]] = None,
    lower_is_better: Optional[Sequence[str]] = None,
    metric_weights: Optional[Dict[str, float]] = None,
    model_col: str = "model_name",
) -> pd.DataFrame:
    """Build a weighted ranking table for model selection.

    Parameters
    ----------
    candidates_df
        Table with one row per candidate model / signal mode.
    higher_is_better
        Metrics to rank descending (best value gets rank 1).
    lower_is_better
        Metrics to rank ascending (best value gets rank 1).
    metric_weights
        Optional weights for the composite rank. Unspecified metrics default to 1.0.
    model_col
        Name of the identifier column.
    """
    if candidates_df is None or not isinstance(candidates_df, pd.DataFrame) or candidates_df.empty:
        return pd.DataFrame()

    out = candidates_df.copy()
    if model_col not in out.columns:
        if "signal_mode" in out.columns:
            out = out.rename(columns={"signal_mode": model_col})
        else:
            out[model_col] = [f"candidate_{i}" for i in range(len(out))]

    higher = list(higher_is_better or [
        "rank_ic_mean",
        "rank_ic_ir",
        "rank_ic_hit_rate",
        "rank_ic_t_stat",
        "n_dates",
        "sharpe",
        "cagr",
        "information_ratio",
        "max_drawdown",
    ])
    lower = list(lower_is_better or [
        "mean_turnover",
        "annual_volatility",
        "tracking_error_annual",
    ])
    weights = {str(k): float(v) for k, v in dict(metric_weights or {}).items()}

    used_metrics: List[str] = []
    for col in higher + lower:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            used_metrics.append(col)

    if not used_metrics:
        return out

    rank_cols: List[str] = []
    for col in higher:
        if col not in out.columns:
            continue
        rcol = f"rank__{col}"
        out[rcol] = out[col].rank(method="average", ascending=False, na_option="bottom")
        rank_cols.append(rcol)
    for col in lower:
        if col not in out.columns:
            continue
        rcol = f"rank__{col}"
        out[rcol] = out[col].rank(method="average", ascending=True, na_option="bottom")
        rank_cols.append(rcol)

    if not rank_cols:
        return out

    raw_score = pd.Series(0.0, index=out.index, dtype="float64")
    weight_sum = 0.0
    for rcol in rank_cols:
        metric = rcol.replace("rank__", "", 1)
        w = float(weights.get(metric, 1.0))
        raw_score = raw_score + pd.to_numeric(out[rcol], errors="coerce").fillna(out.shape[0] + 1.0) * w
        weight_sum += w
    out["ranking_score"] = raw_score / max(weight_sum, 1e-12)
    out["ranking_selected"] = False
    out = out.sort_values(
        ["ranking_score", "rank__rank_ic_mean" if "rank__rank_ic_mean" in out.columns else model_col,
         "rank__sharpe" if "rank__sharpe" in out.columns else model_col],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    if not out.empty:
        out.loc[0, "ranking_selected"] = True
    return out



def auto_select_model_by_ranking(
    run_reports: Dict[str, Dict[str, Any]],
    *,
    candidate_signal_cols: Sequence[str] = ("mu_hat_used", "score", "mu_hat", "weight"),
    target_col: str = "realised_return",
    date_col: str = "date",
    asset_col: str = "asset",
    min_assets: int = 3,
    metric_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Select the best candidate run using a weighted ranking table.

    This is broader than pure Rank-IC selection: it ranks candidates using
    both cross-sectional ranking quality and portfolio-level economic metrics.
    """
    rows: List[Dict[str, Any]] = []
    details: Dict[str, Dict[str, Any]] = {}

    for model_name, run_report in (run_reports or {}).items():
        if run_report is None:
            continue
        rank_detail = evaluate_run_rank_ic_quality(
            run_report,
            candidate_signal_cols=candidate_signal_cols,
            target_col=target_col,
            date_col=date_col,
            asset_col=asset_col,
            min_assets=min_assets,
        )
        perf = run_report.get("performance_summary", {}) or {}
        row = {
            "model_name": str(model_name),
            "signal_mode_requested": rank_detail.get("signal_mode_requested"),
            "signal_mode_effective": rank_detail.get("signal_mode_effective"),
            "signal_col_preferred": rank_detail.get("signal_col_preferred"),
            "signal_col_used": rank_detail.get("signal_col_used"),
            "signal_col_resolution": rank_detail.get("signal_col_resolution"),
            "signal_col_fallback_used": rank_detail.get("signal_col_fallback_used"),
            "signal_col_resolution_note": rank_detail.get("signal_col_resolution_note"),
            "rank_ic_mean": rank_detail.get("rank_ic_mean"),
            "rank_ic_ir": rank_detail.get("rank_ic_ir"),
            "rank_ic_hit_rate": rank_detail.get("rank_ic_hit_rate"),
            "rank_ic_t_stat": rank_detail.get("rank_ic_t_stat"),
            "n_dates": rank_detail.get("n_dates"),
            "cagr": perf.get("cagr"),
            "sharpe": perf.get("sharpe"),
            "annual_volatility": perf.get("annual_volatility", perf.get("annualized_volatility")),
            "max_drawdown": perf.get("max_drawdown"),
            "mean_turnover": perf.get("mean_turnover"),
            "active_return_annual": perf.get("active_return_annual"),
            "tracking_error_annual": perf.get("tracking_error_annual"),
            "information_ratio": perf.get("information_ratio"),
        }
        rows.append(row)
        details[str(model_name)] = {
            **rank_detail,
            "performance_summary": perf,
        }

    table = build_model_selection_ranking_table(
        pd.DataFrame(rows),
        metric_weights=metric_weights,
        model_col="model_name",
    )
    selected = None if table.empty else str(table.iloc[0]["model_name"])
    selected_detail = details.get(selected, {}) if selected is not None else {}
    return {
        "selected_model": selected,
        "selection_table": table,
        "selection_details": details,
        "selected_detail": selected_detail,
    }


# ============================================================
# Breadth vs estimation helpers
# ============================================================

def _coerce_numeric_copy(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def build_breadth_estimation_table(
    df: pd.DataFrame,
    *,
    objective_col: str = "sharpe",
    breadth_col: str = "mean_effective_breadth",
    turnover_col: str = "mean_turnover",
    breadth_bucket_count: int = 4,
) -> pd.DataFrame:
    """Formalise the breadth-vs-estimation trade-off from a parameter grid.

    Expected input is a parameter grid / sweep table already containing
    portfolio performance metrics and breadth diagnostics. The function adds:
    - estimation_load_proxy
    - breadth_quantile_bucket
    - balanced_candidate flag
    - objective_percentile / breadth_percentile / turnover_percentile
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    out = df.copy()
    numeric_cols = [
        objective_col,
        breadth_col,
        turnover_col,
        "temperature",
        "weight_shrink",
        "inertia",
        "information_ratio",
        "cagr",
        "mean_active_assets",
        "mean_effective_risk_bets",
        "mean_diversification_ratio",
    ]
    out = _coerce_numeric_copy(out, [c for c in numeric_cols if c in out.columns])

    need = [c for c in [objective_col, breadth_col] if c in out.columns]
    if len(need) < 2:
        return pd.DataFrame()
    out = out.dropna(subset=[objective_col, breadth_col]).reset_index(drop=True)
    if out.empty:
        return out

    # Estimation-load proxy: hotter temperature + lower shrink + lower inertia -> higher estimation burden.
    temp = pd.to_numeric(out.get("temperature", pd.Series(np.nan, index=out.index)), errors="coerce")
    shrink = pd.to_numeric(out.get("weight_shrink", pd.Series(np.nan, index=out.index)), errors="coerce")
    inertia = pd.to_numeric(out.get("inertia", pd.Series(np.nan, index=out.index)), errors="coerce")

    temp_term = temp.fillna(temp.median() if temp.notna().any() else 1.0).clip(lower=0.0)
    shrink_term = (1.0 - shrink.fillna(shrink.median() if shrink.notna().any() else 0.0)).clip(lower=0.0)
    inertia_term = (1.0 - inertia.fillna(inertia.median() if inertia.notna().any() else 0.0)).clip(lower=0.0)
    out["estimation_load_proxy"] = (temp_term + shrink_term + inertia_term) / 3.0

    # Percentiles for multi-objective balancing.
    out["objective_percentile"] = pd.to_numeric(out[objective_col], errors="coerce").rank(method="average", pct=True)
    out["breadth_percentile"] = pd.to_numeric(out[breadth_col], errors="coerce").rank(method="average", pct=True)
    if turnover_col in out.columns:
        out["turnover_percentile"] = 1.0 - pd.to_numeric(out[turnover_col], errors="coerce").rank(method="average", pct=True)
    else:
        out["turnover_percentile"] = np.nan
    out["estimation_light_percentile"] = 1.0 - pd.to_numeric(out["estimation_load_proxy"], errors="coerce").rank(method="average", pct=True)

    score_parts = ["objective_percentile", "breadth_percentile", "estimation_light_percentile"]
    if turnover_col in out.columns and out[turnover_col].notna().any():
        score_parts.append("turnover_percentile")
    out["breadth_estimation_balance_score"] = out[score_parts].mean(axis=1)

    # Near-frontier balanced candidates.
    obj_cut = float(pd.to_numeric(out[objective_col], errors="coerce").quantile(0.80))
    breadth_cut = float(pd.to_numeric(out[breadth_col], errors="coerce").quantile(0.50))
    est_cut = float(pd.to_numeric(out["estimation_load_proxy"], errors="coerce").quantile(0.50))
    balanced_mask = (out[objective_col] >= obj_cut) & (out[breadth_col] >= breadth_cut) & (out["estimation_load_proxy"] <= est_cut)
    if turnover_col in out.columns and out[turnover_col].notna().any():
        turnover_cut = float(pd.to_numeric(out[turnover_col], errors="coerce").quantile(0.60))
        balanced_mask = balanced_mask & (pd.to_numeric(out[turnover_col], errors="coerce") <= turnover_cut)
    out["balanced_candidate"] = balanced_mask.astype(bool)

    bucket_n = max(int(breadth_bucket_count), 2)
    if out[breadth_col].nunique(dropna=True) >= bucket_n:
        out["breadth_quantile_bucket"] = pd.qcut(out[breadth_col], q=bucket_n, duplicates="drop")
    else:
        out["breadth_quantile_bucket"] = pd.Series(["all"] * len(out), index=out.index, dtype="object")

    sort_cols = ["breadth_estimation_balance_score", objective_col]
    ascending = [False, False]
    if turnover_col in out.columns:
        sort_cols.append(turnover_col)
        ascending.append(True)
    return out.sort_values(sort_cols, ascending=ascending, na_position="last").reset_index(drop=True)



def summarize_breadth_estimation_tradeoff(
    df: pd.DataFrame,
    *,
    objective_col: str = "sharpe",
    breadth_col: str = "mean_effective_breadth",
    turnover_col: str = "mean_turnover",
    breadth_bucket_count: int = 4,
) -> pd.DataFrame:
    """Summarise the breadth / estimation trade-off from a parameter grid."""
    table = build_breadth_estimation_table(
        df,
        objective_col=objective_col,
        breadth_col=breadth_col,
        turnover_col=turnover_col,
        breadth_bucket_count=breadth_bucket_count,
    )
    if table.empty:
        return pd.DataFrame(columns=["metric", "value"])

    rows = []
    obj = pd.to_numeric(table[objective_col], errors="coerce")
    breadth = pd.to_numeric(table[breadth_col], errors="coerce")
    est = pd.to_numeric(table["estimation_load_proxy"], errors="coerce")
    turn = pd.to_numeric(table[turnover_col], errors="coerce") if turnover_col in table.columns else pd.Series(np.nan, index=table.index)

    rows.extend([
        {"metric": "n_candidates", "value": int(len(table))},
        {"metric": "balanced_candidate_count", "value": int(table["balanced_candidate"].sum())},
        {"metric": f"corr_{breadth_col}_vs_{objective_col}", "value": _safe_float(breadth.corr(obj))},
        {"metric": f"corr_estimation_load_vs_{objective_col}", "value": _safe_float(est.corr(obj))},
        {"metric": f"corr_{breadth_col}_vs_{turnover_col}", "value": _safe_float(breadth.corr(turn)) if turnover_col in table.columns else np.nan},
        {"metric": f"corr_estimation_load_vs_{turnover_col}", "value": _safe_float(est.corr(turn)) if turnover_col in table.columns else np.nan},
        {"metric": f"best_{objective_col}", "value": _safe_float(obj.max())},
        {"metric": f"median_{objective_col}", "value": _safe_float(obj.median())},
        {"metric": f"median_{breadth_col}", "value": _safe_float(breadth.median())},
        {"metric": "median_estimation_load_proxy", "value": _safe_float(est.median())},
        {"metric": "best_balance_score", "value": _safe_float(pd.to_numeric(table["breadth_estimation_balance_score"], errors="coerce").max())},
    ])

    bucket_col = "breadth_quantile_bucket"
    if bucket_col in table.columns:
        grp = table.groupby(bucket_col, dropna=False)
        bucket_summary = grp.agg(
            n=(objective_col, "size"),
            objective_median=(objective_col, "median"),
            objective_best=(objective_col, "max"),
            breadth_median=(breadth_col, "median"),
            estimation_load_median=("estimation_load_proxy", "median"),
            balance_score_median=("breadth_estimation_balance_score", "median"),
        ).reset_index()
        if turnover_col in table.columns:
            bucket_turn = grp[turnover_col].median().reset_index(name="turnover_median")
            bucket_summary = bucket_summary.merge(bucket_turn, on=bucket_col, how="left")
        rows.append({"metric": "bucket_summary_rows", "value": int(len(bucket_summary))})

    return pd.DataFrame(rows)



def choose_balanced_breadth_candidate(
    df: pd.DataFrame,
    *,
    objective_col: str = "sharpe",
    breadth_col: str = "mean_effective_breadth",
    turnover_col: str = "mean_turnover",
    breadth_bucket_count: int = 4,
    min_objective_quantile: float = 0.80,
) -> Tuple[Optional[pd.Series], pd.DataFrame]:
    """Choose a balanced candidate that respects both breadth and estimation discipline."""
    table = build_breadth_estimation_table(
        df,
        objective_col=objective_col,
        breadth_col=breadth_col,
        turnover_col=turnover_col,
        breadth_bucket_count=breadth_bucket_count,
    )
    if table.empty:
        return None, table

    obj = pd.to_numeric(table[objective_col], errors="coerce")
    obj_cut = float(obj.quantile(float(np.clip(min_objective_quantile, 0.0, 1.0))))

    eligible = table[table["balanced_candidate"] == True].copy()
    eligible = eligible[pd.to_numeric(eligible[objective_col], errors="coerce") >= obj_cut].copy()
    if eligible.empty:
        eligible = table.copy()

    sort_cols = ["breadth_estimation_balance_score", objective_col]
    ascending = [False, False]
    if turnover_col in eligible.columns:
        sort_cols.append(turnover_col)
        ascending.append(True)
    chosen = eligible.sort_values(sort_cols, ascending=ascending, na_position="last").reset_index(drop=True)
    return chosen.iloc[0], table


# ============================================================
# Full cost model / tax / slippage helpers
# ============================================================

def _run_cost_dict(run: Dict[str, Any]) -> Dict[str, Any]:
    cost = run.get("cost_summary", {}) or {}
    if not cost and isinstance(run.get("run_report"), dict):
        cost = (((run.get("run_report") or {}).get("sections") or {}).get("costs") or {})
    return cost


def build_cost_model_summary(run: Dict[str, Any]) -> pd.DataFrame:
    """Build a user-facing summary table for the full implementation-cost model.

    Works with the run object returned by investment.py and stays backward-compatible:
    if the cost model is absent, it simply returns NaNs / disabled flags.
    """
    perf = _run_perf_dict(run)
    cost = _run_cost_dict(run)
    cfg = run.get("config")

    cost_enabled = bool(cost.get("cost_model_enabled", getattr(cfg, "cost_model_enabled", False)))
    tax_enabled = bool(cost.get("tax_model_enabled", getattr(cfg, "tax_model_enabled", False)))

    rows = [
        {"metric": "cost_model_enabled", "value": cost_enabled},
        {"metric": "tax_model_enabled", "value": tax_enabled},
        {"metric": "gross_cagr", "value": _safe_float(cost.get("gross_cagr", perf.get("gross_cagr")))},
        {"metric": "net_cagr", "value": _safe_float(cost.get("net_cagr", perf.get("cagr")))},
        {"metric": "cost_impact_on_cagr", "value": _safe_float(cost.get("cost_impact_on_cagr"))},
        {"metric": "gross_sharpe", "value": _safe_float(perf.get("gross_sharpe"))},
        {"metric": "net_sharpe", "value": _safe_float(perf.get("sharpe"))},
        {
            "metric": "cost_impact_on_sharpe",
            "value": _safe_float(perf.get("sharpe")) - _safe_float(perf.get("gross_sharpe")),
        },
        {"metric": "mean_total_cost_rate", "value": _safe_float(cost.get("mean_total_cost_rate", perf.get("mean_total_cost_rate")))},
        {"metric": "mean_transaction_cost_rate", "value": _safe_float(cost.get("mean_transaction_cost_rate", perf.get("mean_transaction_cost_rate")))},
        {"metric": "mean_tax_cost_rate", "value": _safe_float(cost.get("mean_tax_cost_rate", perf.get("mean_tax_cost_rate")))},
        {"metric": "mean_holding_cost_rate", "value": _safe_float(cost.get("mean_holding_cost_rate"))},
        {"metric": "cumulative_total_cost_rate", "value": _safe_float(cost.get("cumulative_total_cost_rate"))},
        {"metric": "mean_cost_drag_bps", "value": _safe_float(cost.get("mean_cost_drag_bps"))},
        {"metric": "mean_transaction_drag_bps", "value": _safe_float(cost.get("mean_transaction_drag_bps"))},
        {"metric": "mean_tax_drag_bps", "value": _safe_float(cost.get("mean_tax_drag_bps"))},
        {"metric": "mean_holding_drag_bps", "value": _safe_float(cost.get("mean_holding_drag_bps"))},
        {"metric": "mean_turnover", "value": _safe_float(perf.get("mean_turnover"))},
        {"metric": "mean_turnover_buy", "value": _safe_float(cost.get("mean_turnover_buy"))},
        {"metric": "mean_turnover_sell", "value": _safe_float(cost.get("mean_turnover_sell"))},
        {"metric": "mean_realised_gain_positive", "value": _safe_float(cost.get("mean_realised_gain_positive"))},
    ]
    return pd.DataFrame(rows)



def build_cost_drag_timeseries(run: Dict[str, Any]) -> pd.DataFrame:
    """Create a period-by-period implementation-drag table from diagnostics_df."""
    diag = run.get("diagnostics_df", pd.DataFrame())
    if not isinstance(diag, pd.DataFrame) or diag.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "gross_portfolio_return_simple",
                "net_portfolio_return_simple",
                "transaction_cost_rate",
                "holding_cost_rate",
                "tax_cost_rate",
                "total_cost_rate",
                "cost_drag_bps",
                "cumulative_gross_nav",
                "cumulative_net_nav",
                "cumulative_cost_gap",
            ]
        )

    out = diag.copy()
    keep = [
        c for c in [
            "date",
            "gross_portfolio_return_simple",
            "portfolio_return_simple",
            "transaction_cost_rate",
            "holding_cost_rate",
            "tax_cost_rate",
            "total_cost_rate",
            "cost_drag_bps",
            "transaction_cost_drag_bps",
            "tax_drag_bps",
            "holding_cost_drag_bps",
            "turnover",
            "turnover_buy",
            "turnover_sell",
            "realised_gain_value",
            "realised_gain_positive_value",
            "realised_gain_negative_value",
            "tax_credit_value",
            "nav_pre_cost",
            "nav_post_cost_pre_return",
            "nav_end",
        ] if c in out.columns
    ]
    out = out[keep].copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for c in out.columns:
        if c != "date":
            out[c] = pd.to_numeric(out[c], errors="coerce")

    gross = pd.to_numeric(out.get("gross_portfolio_return_simple", pd.Series(dtype="float64")), errors="coerce").fillna(0.0)
    net = pd.to_numeric(out.get("portfolio_return_simple", pd.Series(dtype="float64")), errors="coerce").fillna(0.0)
    out = out.rename(columns={"portfolio_return_simple": "net_portfolio_return_simple"})
    out["cumulative_gross_nav"] = (1.0 + gross).cumprod()
    out["cumulative_net_nav"] = (1.0 + net).cumprod()
    out["cumulative_cost_gap"] = out["cumulative_net_nav"] - out["cumulative_gross_nav"]
    out["cumulative_cost_gap_pct"] = out["cumulative_cost_gap"] / out["cumulative_gross_nav"].replace(0.0, np.nan)
    return out.reset_index(drop=True)



def compare_cost_model_runs(
    baseline_run: Dict[str, Any],
    cost_run: Dict[str, Any],
    *,
    baseline_name: str = "no_cost_model",
    cost_name: str = "cost_model",
) -> Dict[str, Any]:
    """Compare a baseline run against a cost-aware run.

    Intended usage:
    - baseline_run: existing legacy or no-cost run
    - cost_run: run with cost_model_enabled and optionally tax_model_enabled
    """
    left_perf = _run_perf_dict(baseline_run)
    right_perf = _run_perf_dict(cost_run)
    left_cost = _run_cost_dict(baseline_run)
    right_cost = _run_cost_dict(cost_run)

    rows = [
        {
            "metric": "cagr",
            baseline_name: _safe_float(left_perf.get("cagr")),
            cost_name: _safe_float(right_perf.get("cagr")),
            "delta": _safe_float(right_perf.get("cagr")) - _safe_float(left_perf.get("cagr")),
            "better_when": "higher",
            "note": "Net CAGR after implementation costs.",
        },
        {
            "metric": "sharpe",
            baseline_name: _safe_float(left_perf.get("sharpe")),
            cost_name: _safe_float(right_perf.get("sharpe")),
            "delta": _safe_float(right_perf.get("sharpe")) - _safe_float(left_perf.get("sharpe")),
            "better_when": "higher",
            "note": "Net Sharpe after implementation costs.",
        },
        {
            "metric": "max_drawdown",
            baseline_name: _safe_float(left_perf.get("max_drawdown")),
            cost_name: _safe_float(right_perf.get("max_drawdown")),
            "delta": _safe_float(right_perf.get("max_drawdown")) - _safe_float(left_perf.get("max_drawdown")),
            "better_when": "higher",
            "note": "Less negative drawdown is better.",
        },
        {
            "metric": "mean_turnover",
            baseline_name: _safe_float(left_perf.get("mean_turnover")),
            cost_name: _safe_float(right_perf.get("mean_turnover")),
            "delta": _safe_float(right_perf.get("mean_turnover")) - _safe_float(left_perf.get("mean_turnover")),
            "better_when": "lower",
            "note": "Turnover drives transaction cost drag.",
        },
        {
            "metric": "gross_cagr_internal",
            baseline_name: _safe_float(left_cost.get("gross_cagr", left_perf.get("gross_cagr"))),
            cost_name: _safe_float(right_cost.get("gross_cagr", right_perf.get("gross_cagr"))),
            "delta": _safe_float(right_cost.get("gross_cagr", right_perf.get("gross_cagr"))) - _safe_float(left_cost.get("gross_cagr", left_perf.get("gross_cagr"))),
            "better_when": "higher",
            "note": "Gross engine CAGR before implementation drag.",
        },
        {
            "metric": "mean_total_cost_rate",
            baseline_name: _safe_float(left_cost.get("mean_total_cost_rate", left_perf.get("mean_total_cost_rate"))),
            cost_name: _safe_float(right_cost.get("mean_total_cost_rate", right_perf.get("mean_total_cost_rate"))),
            "delta": _safe_float(right_cost.get("mean_total_cost_rate", right_perf.get("mean_total_cost_rate"))) - _safe_float(left_cost.get("mean_total_cost_rate", left_perf.get("mean_total_cost_rate"))),
            "better_when": "lower",
            "note": "All-in periodic cost drag rate.",
        },
        {
            "metric": "mean_transaction_cost_rate",
            baseline_name: _safe_float(left_cost.get("mean_transaction_cost_rate", left_perf.get("mean_transaction_cost_rate"))),
            cost_name: _safe_float(right_cost.get("mean_transaction_cost_rate", right_perf.get("mean_transaction_cost_rate"))),
            "delta": _safe_float(right_cost.get("mean_transaction_cost_rate", right_perf.get("mean_transaction_cost_rate"))) - _safe_float(left_cost.get("mean_transaction_cost_rate", left_perf.get("mean_transaction_cost_rate"))),
            "better_when": "lower",
            "note": "Commission + slippage + spread + impact drag.",
        },
        {
            "metric": "mean_tax_cost_rate",
            baseline_name: _safe_float(left_cost.get("mean_tax_cost_rate", left_perf.get("mean_tax_cost_rate"))),
            cost_name: _safe_float(right_cost.get("mean_tax_cost_rate", right_perf.get("mean_tax_cost_rate"))),
            "delta": _safe_float(right_cost.get("mean_tax_cost_rate", right_perf.get("mean_tax_cost_rate"))) - _safe_float(left_cost.get("mean_tax_cost_rate", left_perf.get("mean_tax_cost_rate"))),
            "better_when": "lower",
            "note": "Tax drag from realised gains net of optional loss credit.",
        },
        {
            "metric": "mean_holding_cost_rate",
            baseline_name: _safe_float(left_cost.get("mean_holding_cost_rate")),
            cost_name: _safe_float(right_cost.get("mean_holding_cost_rate")),
            "delta": _safe_float(right_cost.get("mean_holding_cost_rate")) - _safe_float(left_cost.get("mean_holding_cost_rate")),
            "better_when": "lower",
            "note": "Ongoing annualised carry / holding cost component.",
        },
        {
            "metric": "cost_impact_on_cagr_internal",
            baseline_name: _safe_float(left_cost.get("cost_impact_on_cagr")),
            cost_name: _safe_float(right_cost.get("cost_impact_on_cagr")),
            "delta": _safe_float(right_cost.get("cost_impact_on_cagr")) - _safe_float(left_cost.get("cost_impact_on_cagr")),
            "better_when": "higher",
            "note": "Internal gross-to-net CAGR gap; closer to zero is better.",
        },
        {
            "metric": "mean_cost_drag_bps",
            baseline_name: _safe_float(left_cost.get("mean_cost_drag_bps")),
            cost_name: _safe_float(right_cost.get("mean_cost_drag_bps")),
            "delta": _safe_float(right_cost.get("mean_cost_drag_bps")) - _safe_float(left_cost.get("mean_cost_drag_bps")),
            "better_when": "lower",
            "note": "Average all-in cost drag in basis points per period.",
        },
    ]
    comparison = pd.DataFrame(rows)

    def _direction(metric: str, delta: float, better_when: str) -> str:
        if not np.isfinite(delta):
            return "neutral"
        if better_when == "higher":
            return "improved" if delta > 0 else ("worsened" if delta < 0 else "neutral")
        if better_when == "lower":
            return "improved" if delta < 0 else ("worsened" if delta > 0 else "neutral")
        return "neutral"

    comparison["direction"] = [
        _direction(m, d, bw)
        for m, d, bw in zip(
            comparison["metric"],
            pd.to_numeric(comparison["delta"], errors="coerce"),
            comparison["better_when"],
        )
    ]

    focus_metrics = [
        "cagr", "sharpe", "max_drawdown", "mean_turnover",
        "mean_total_cost_rate", "mean_transaction_cost_rate", "mean_tax_cost_rate",
        "mean_holding_cost_rate", "mean_cost_drag_bps", "cost_impact_on_cagr_internal",
    ]
    focus = comparison[comparison["metric"].isin(focus_metrics)].copy().reset_index(drop=True)

    delta_sharpe = _safe_float(focus.loc[focus["metric"] == "sharpe", "delta"].iloc[0]) if (focus["metric"] == "sharpe").any() else np.nan
    delta_cagr = _safe_float(focus.loc[focus["metric"] == "cagr", "delta"].iloc[0]) if (focus["metric"] == "cagr").any() else np.nan
    delta_cost = _safe_float(focus.loc[focus["metric"] == "mean_total_cost_rate", "delta"].iloc[0]) if (focus["metric"] == "mean_total_cost_rate").any() else np.nan

    if np.isfinite(delta_sharpe) and delta_sharpe < 0 and np.isfinite(delta_cost) and delta_cost > 0:
        headline = f"{cost_name} shows the expected implementation drag versus {baseline_name}: net Sharpe falls once costs are applied."
    elif np.isfinite(delta_sharpe) and delta_sharpe >= 0 and np.isfinite(delta_cost) and delta_cost > 0:
        headline = f"{cost_name} remains economically resilient versus {baseline_name}: cost drag is present but net Sharpe holds up."
    else:
        headline = f"Cost-model comparison between {baseline_name} and {cost_name} is mixed or incomplete."

    decision_summary = {
        "baseline_name": str(baseline_name),
        "cost_name": str(cost_name),
        "headline": headline,
        "delta_sharpe": delta_sharpe,
        "delta_cagr": delta_cagr,
        "delta_mean_total_cost_rate": delta_cost,
        "improved_metric_count": int((focus["direction"] == "improved").sum()) if not focus.empty else 0,
        "worsened_metric_count": int((focus["direction"] == "worsened").sum()) if not focus.empty else 0,
        "cost_model_enabled": bool(right_cost.get("cost_model_enabled", False)),
        "tax_model_enabled": bool(right_cost.get("tax_model_enabled", False)),
    }

    return {
        "cost_summary_baseline": build_cost_model_summary(baseline_run).rename(columns={"value": baseline_name}),
        "cost_summary_cost_run": build_cost_model_summary(cost_run).rename(columns={"value": cost_name}),
        "cost_comparison": comparison,
        "cost_focus": focus,
        "cost_drag_timeseries": build_cost_drag_timeseries(cost_run),
        "decision_summary": decision_summary,
    }


# ============================================================
# Industrial factor-model helpers
# ============================================================

def _run_factor_dict(run: Dict[str, Any]) -> Dict[str, Any]:
    factor = run.get("factor_model_summary", {}) or {}
    if not factor and isinstance(run.get("run_report"), dict):
        factor = (((run.get("run_report") or {}).get("sections") or {}).get("factor_model") or {})
    return factor


def build_factor_model_summary(run: Dict[str, Any]) -> pd.DataFrame:
    """Build a user-facing summary table for the industrial factor-model overlay.

    Stays backward-compatible: if the factor block is absent, disabled flags and NaNs are returned.
    """
    perf = _run_perf_dict(run)
    factor = _run_factor_dict(run)
    cfg = run.get("config")

    factor_enabled = bool(factor.get("factor_model_active", getattr(cfg, "factor_model_active", False)))
    factor_cov_enabled = bool(factor.get("factor_covariance_active", getattr(cfg, "factor_covariance_active", False)))

    rows = [
        {"metric": "factor_model_active", "value": factor_enabled},
        {"metric": "factor_covariance_active", "value": factor_cov_enabled},
        {"metric": "factor_overlay_active_share", "value": _safe_float(factor.get("factor_overlay_active_share"))},
        {"metric": "factor_model_n_factors_mean", "value": _safe_float(factor.get("factor_model_n_factors_mean"))},
        {"metric": "factor_model_n_obs_mean", "value": _safe_float(factor.get("factor_model_n_obs_mean"))},
        {"metric": "factor_model_explained_variance_share_mean", "value": _safe_float(factor.get("factor_model_explained_variance_share_mean"))},
        {"metric": "factor_model_residual_variance_share_mean", "value": _safe_float(factor.get("factor_model_residual_variance_share_mean"))},
        {"metric": "factor_mu_abs_tilt_mean", "value": _safe_float(factor.get("factor_mu_abs_tilt_mean"))},
        {"metric": "factor_mu_abs_tilt_max", "value": _safe_float(factor.get("factor_mu_abs_tilt_max"))},
        {"metric": "factor_covariance_blend_used_mean", "value": _safe_float(factor.get("factor_covariance_blend_used_mean"))},
        {"metric": "factor_residual_blend_used_mean", "value": _safe_float(factor.get("factor_residual_blend_used_mean"))},
        {"metric": "factor_covariance_trace_ratio_mean", "value": _safe_float(factor.get("factor_covariance_trace_ratio_mean"))},
        {"metric": "gross_cagr", "value": _safe_float(perf.get("gross_cagr", perf.get("cagr")))},
        {"metric": "cagr", "value": _safe_float(perf.get("cagr"))},
        {"metric": "sharpe", "value": _safe_float(perf.get("sharpe"))},
        {"metric": "max_drawdown", "value": _safe_float(perf.get("max_drawdown"))},
        {"metric": "mean_turnover", "value": _safe_float(perf.get("mean_turnover"))},
        {"metric": "information_ratio", "value": _safe_float(perf.get("information_ratio"))},
    ]
    return pd.DataFrame(rows)


def build_factor_model_timeseries(run: Dict[str, Any]) -> pd.DataFrame:
    """Create a period-by-period factor-overlay diagnostic table from diagnostics_df."""
    diag = run.get("diagnostics_df", pd.DataFrame())
    if not isinstance(diag, pd.DataFrame) or diag.empty:
        return pd.DataFrame(columns=[
            "date",
            "factor_overlay_active",
            "factor_model_n_factors",
            "factor_model_n_obs",
            "factor_model_explained_variance_share",
            "factor_mu_abs_tilt_mean",
            "factor_mu_abs_tilt_max",
            "factor_covariance_blend_used",
            "factor_residual_blend_used",
            "factor_covariance_trace_ratio",
        ])

    out = diag.copy()
    keep = [c for c in [
        "date",
        "factor_model_active",
        "factor_covariance_active",
        "factor_overlay_active",
        "factor_model_method",
        "factor_model_n_factors",
        "factor_model_n_obs",
        "factor_model_explained_variance_share",
        "factor_model_residual_variance_share",
        "factor_model_factor_strength",
        "factor_mu_blend_used",
        "factor_covariance_blend_used",
        "factor_residual_blend_used",
        "factor_mu_abs_tilt_mean",
        "factor_mu_abs_tilt_max",
        "factor_covariance_trace_ratio",
        "factor_common_mu_mean",
        "factor_common_mu_std",
        "factor_common_cov_trace",
        "factor_residual_cov_trace",
        "factor_total_cov_trace",
        "portfolio_return_simple",
        "gross_portfolio_return_simple",
        "turnover",
    ] if c in out.columns]
    out = out[keep].copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for c in out.columns:
        if c not in {"date", "factor_model_method"}:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "gross_portfolio_return_simple" in out.columns:
        out["cumulative_gross_nav"] = (1.0 + pd.to_numeric(out["gross_portfolio_return_simple"], errors="coerce").fillna(0.0)).cumprod()
    if "portfolio_return_simple" in out.columns:
        out["cumulative_net_nav"] = (1.0 + pd.to_numeric(out["portfolio_return_simple"], errors="coerce").fillna(0.0)).cumprod()
    if "factor_overlay_active" in out.columns:
        out["factor_overlay_active_cum_share"] = pd.to_numeric(out["factor_overlay_active"], errors="coerce").fillna(0.0).expanding().mean()
    return out.reset_index(drop=True)


def compare_factor_model_runs(
    baseline_run: Dict[str, Any],
    factor_run: Dict[str, Any],
    *,
    baseline_name: str = "baseline",
    factor_name: str = "factor_model",
) -> Dict[str, Any]:
    """Compare a baseline run against an industrial factor-model run.

    Intended usage:
    - baseline_run: no factor overlay / legacy run
    - factor_run: run with factor_model_active and/or factor_covariance_active
    """
    left_perf = _run_perf_dict(baseline_run)
    right_perf = _run_perf_dict(factor_run)
    left_factor = _run_factor_dict(baseline_run)
    right_factor = _run_factor_dict(factor_run)

    rows = [
        {
            "metric": "cagr",
            baseline_name: _safe_float(left_perf.get("cagr")),
            factor_name: _safe_float(right_perf.get("cagr")),
            "delta": _safe_float(right_perf.get("cagr")) - _safe_float(left_perf.get("cagr")),
            "better_when": "higher",
            "note": "Compound growth after applying the factor overlay.",
        },
        {
            "metric": "sharpe",
            baseline_name: _safe_float(left_perf.get("sharpe")),
            factor_name: _safe_float(right_perf.get("sharpe")),
            "delta": _safe_float(right_perf.get("sharpe")) - _safe_float(left_perf.get("sharpe")),
            "better_when": "higher",
            "note": "Risk-adjusted performance under the factor overlay.",
        },
        {
            "metric": "information_ratio",
            baseline_name: _safe_float(left_perf.get("information_ratio")),
            factor_name: _safe_float(right_perf.get("information_ratio")),
            "delta": _safe_float(right_perf.get("information_ratio")) - _safe_float(left_perf.get("information_ratio")),
            "better_when": "higher",
            "note": "Higher IR suggests the overlay improves active efficiency.",
        },
        {
            "metric": "max_drawdown",
            baseline_name: _safe_float(left_perf.get("max_drawdown")),
            factor_name: _safe_float(right_perf.get("max_drawdown")),
            "delta": _safe_float(right_perf.get("max_drawdown")) - _safe_float(left_perf.get("max_drawdown")),
            "better_when": "higher",
            "note": "Less negative drawdown is better.",
        },
        {
            "metric": "mean_turnover",
            baseline_name: _safe_float(left_perf.get("mean_turnover")),
            factor_name: _safe_float(right_perf.get("mean_turnover")),
            "delta": _safe_float(right_perf.get("mean_turnover")) - _safe_float(left_perf.get("mean_turnover")),
            "better_when": "lower",
            "note": "Overlay should not destroy implementability via turnover.",
        },
        {
            "metric": "factor_overlay_active_share",
            baseline_name: _safe_float(left_factor.get("factor_overlay_active_share")),
            factor_name: _safe_float(right_factor.get("factor_overlay_active_share")),
            "delta": _safe_float(right_factor.get("factor_overlay_active_share")) - _safe_float(left_factor.get("factor_overlay_active_share")),
            "better_when": "higher",
            "note": "Share of dates where the industrial factor overlay was active.",
        },
        {
            "metric": "factor_model_explained_variance_share_mean",
            baseline_name: _safe_float(left_factor.get("factor_model_explained_variance_share_mean")),
            factor_name: _safe_float(right_factor.get("factor_model_explained_variance_share_mean")),
            "delta": _safe_float(right_factor.get("factor_model_explained_variance_share_mean")) - _safe_float(left_factor.get("factor_model_explained_variance_share_mean")),
            "better_when": "higher",
            "note": "Higher explained common variance means the factor block captures more systematic structure.",
        },
        {
            "metric": "factor_model_n_factors_mean",
            baseline_name: _safe_float(left_factor.get("factor_model_n_factors_mean")),
            factor_name: _safe_float(right_factor.get("factor_model_n_factors_mean")),
            "delta": _safe_float(right_factor.get("factor_model_n_factors_mean")) - _safe_float(left_factor.get("factor_model_n_factors_mean")),
            "better_when": "higher",
            "note": "Average number of industrial/statistical factors used when active.",
        },
        {
            "metric": "factor_mu_abs_tilt_mean",
            baseline_name: _safe_float(left_factor.get("factor_mu_abs_tilt_mean")),
            factor_name: _safe_float(right_factor.get("factor_mu_abs_tilt_mean")),
            "delta": _safe_float(right_factor.get("factor_mu_abs_tilt_mean")) - _safe_float(left_factor.get("factor_mu_abs_tilt_mean")),
            "better_when": "lower",
            "note": "Average absolute tilt applied to base mu by the industrial overlay.",
        },
        {
            "metric": "factor_covariance_blend_used_mean",
            baseline_name: _safe_float(left_factor.get("factor_covariance_blend_used_mean")),
            factor_name: _safe_float(right_factor.get("factor_covariance_blend_used_mean")),
            "delta": _safe_float(right_factor.get("factor_covariance_blend_used_mean")) - _safe_float(left_factor.get("factor_covariance_blend_used_mean")),
            "better_when": "lower",
            "note": "Higher blend means stronger replacement of base covariance by factor covariance.",
        },
        {
            "metric": "factor_covariance_trace_ratio_mean",
            baseline_name: _safe_float(left_factor.get("factor_covariance_trace_ratio_mean")),
            factor_name: _safe_float(right_factor.get("factor_covariance_trace_ratio_mean")),
            "delta": _safe_float(right_factor.get("factor_covariance_trace_ratio_mean")) - _safe_float(left_factor.get("factor_covariance_trace_ratio_mean")),
            "better_when": "lower",
            "note": "Trace ratio near 1 suggests the overlay preserves overall variance scale.",
        },
    ]
    comparison = pd.DataFrame(rows)

    def _direction(metric: str, delta: float, better_when: str) -> str:
        if not np.isfinite(delta):
            return "neutral"
        if metric == "factor_covariance_trace_ratio_mean":
            left = _safe_float(comparison.loc[comparison["metric"] == metric, baseline_name].iloc[0])
            right = _safe_float(comparison.loc[comparison["metric"] == metric, factor_name].iloc[0])
            if np.isfinite(left) and np.isfinite(right):
                left_gap = abs(left - 1.0)
                right_gap = abs(right - 1.0)
                if right_gap < left_gap - 1e-12:
                    return "improved"
                if right_gap > left_gap + 1e-12:
                    return "worsened"
                return "neutral"
        if better_when == "higher":
            return "improved" if delta > 0 else ("worsened" if delta < 0 else "neutral")
        if better_when == "lower":
            return "improved" if delta < 0 else ("worsened" if delta > 0 else "neutral")
        return "neutral"

    comparison["direction"] = [
        _direction(m, d, bw)
        for m, d, bw in zip(
            comparison["metric"],
            pd.to_numeric(comparison["delta"], errors="coerce"),
            comparison["better_when"],
        )
    ]

    focus_metrics = [
        "cagr", "sharpe", "information_ratio", "max_drawdown", "mean_turnover",
        "factor_overlay_active_share", "factor_model_explained_variance_share_mean",
        "factor_mu_abs_tilt_mean", "factor_covariance_blend_used_mean", "factor_covariance_trace_ratio_mean",
    ]
    focus = comparison[comparison["metric"].isin(focus_metrics)].copy().reset_index(drop=True)

    delta_sharpe = _safe_float(focus.loc[focus["metric"] == "sharpe", "delta"].iloc[0]) if (focus["metric"] == "sharpe").any() else np.nan
    delta_cagr = _safe_float(focus.loc[focus["metric"] == "cagr", "delta"].iloc[0]) if (focus["metric"] == "cagr").any() else np.nan
    delta_ir = _safe_float(focus.loc[focus["metric"] == "information_ratio", "delta"].iloc[0]) if (focus["metric"] == "information_ratio").any() else np.nan
    delta_turnover = _safe_float(focus.loc[focus["metric"] == "mean_turnover", "delta"].iloc[0]) if (focus["metric"] == "mean_turnover").any() else np.nan

    improved = int((focus["direction"] == "improved").sum()) if not focus.empty else 0
    worsened = int((focus["direction"] == "worsened").sum()) if not focus.empty else 0

    if np.isfinite(delta_sharpe) and delta_sharpe > 0 and improved >= worsened:
        headline = f"{factor_name} looks economically promising versus {baseline_name}: Sharpe improves while the industrial factor block remains interpretable."
    elif np.isfinite(delta_sharpe) and delta_sharpe < 0 and worsened > improved:
        headline = f"{factor_name} looks weak versus {baseline_name}: the industrial factor overlay worsens the portfolio profile."
    else:
        headline = f"Factor-model comparison between {baseline_name} and {factor_name} is mixed or regime-dependent."

    decision_summary = {
        "baseline_name": str(baseline_name),
        "factor_name": str(factor_name),
        "headline": headline,
        "delta_sharpe": delta_sharpe,
        "delta_cagr": delta_cagr,
        "delta_information_ratio": delta_ir,
        "delta_mean_turnover": delta_turnover,
        "improved_metric_count": improved,
        "worsened_metric_count": worsened,
        "factor_model_active": bool(right_factor.get("factor_model_active", False)),
        "factor_covariance_active": bool(right_factor.get("factor_covariance_active", False)),
    }

    return {
        "factor_summary_baseline": build_factor_model_summary(baseline_run).rename(columns={"value": baseline_name}),
        "factor_summary_factor_run": build_factor_model_summary(factor_run).rename(columns={"value": factor_name}),
        "factor_comparison": comparison,
        "factor_focus": focus,
        "factor_timeseries": build_factor_model_timeseries(factor_run),
        "decision_summary": decision_summary,
    }


# ============================================================
# Advanced tuning engines (Bayesian / Optuna / surrogate)
# ============================================================

import random


def _tuning_valid_cfg_payload(base_cfg_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Filter a raw payload down to valid MicroPipelineConfig fields when possible."""
    try:
        from src.investment import MicroPipelineConfig
    except Exception:
        try:
            from investment import MicroPipelineConfig
        except Exception:
            return dict(base_cfg_payload or {})

    valid_fields = {f.name for f in MicroPipelineConfig.__dataclass_fields__.values()}
    return {k: v for k, v in dict(base_cfg_payload or {}).items() if k in valid_fields}



def _coerce_sampled_param_value(spec: Any, value: Any) -> Any:
    """Coerce a proposed value back into the type/range implied by the param spec."""
    if isinstance(spec, tuple) and len(spec) == 2:
        lo, hi = spec
        if isinstance(lo, int) and isinstance(hi, int):
            try:
                vv = int(round(float(value)))
            except Exception:
                vv = int(lo)
            return int(min(max(vv, int(lo)), int(hi)))
        try:
            vv = float(value)
        except Exception:
            vv = float(lo)
        return float(min(max(vv, float(lo)), float(hi)))
    if isinstance(spec, list):
        if not spec:
            return np.nan
        # if value already matches one candidate, keep it
        for cand in spec:
            if value == cand:
                return cand
        # numeric nearest fallback
        try:
            v = float(value)
            nums = [(cand, _safe_float(cand)) for cand in spec]
            nums = [(cand, num) for cand, num in nums if np.isfinite(num)]
            if nums:
                return min(nums, key=lambda x: abs(x[1] - v))[0]
        except Exception:
            pass
        return random.choice(spec)
    return value



def _sample_param_dict(param_space: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sample one parameter dict from a search space.

    param_space format:

    {
        "temperature": (0.5, 2.0),
        "weight_shrink": (0.0, 0.8),
        "top_k": [10, 20, 30]
    }
    """
    params: Dict[str, Any] = {}
    for k, v in (param_space or {}).items():
        if isinstance(v, tuple) and len(v) == 2:
            lo, hi = v
            if isinstance(lo, int) and isinstance(hi, int):
                params[k] = random.randint(int(lo), int(hi))
            else:
                params[k] = random.uniform(float(lo), float(hi))
        elif isinstance(v, list):
            if len(v) == 0:
                continue
            params[k] = random.choice(v)
        else:
            params[k] = v
    return params



def _extract_inverse_concentration_proxy(run: Dict[str, Any]) -> float:
    diag = (run or {}).get("diagnostics_df", pd.DataFrame()) if isinstance(run, dict) else pd.DataFrame()
    if isinstance(diag, pd.DataFrame) and not diag.empty:
        if "weight_herfindahl" in diag.columns:
            xs = pd.to_numeric(diag["weight_herfindahl"], errors="coerce")
            xs = xs.where(xs > 0.0)
            inv = 1.0 / xs
            if inv.notna().any():
                return float(inv.mean())
        if "effective_n_assets" in diag.columns:
            xs = pd.to_numeric(diag["effective_n_assets"], errors="coerce")
            if xs.notna().any():
                return float(xs.mean())
    div = (run.get("diversification_summary", {}) or {}) if isinstance(run, dict) else {}
    return _safe_float(div.get("mean_effective_n_assets"))


def _extract_tuning_metric_bundle(run: Dict[str, Any]) -> Dict[str, Any]:
    perf = _run_perf_dict(run)
    risk = (run.get("risk_summary", {}) or {}) if isinstance(run, dict) else {}
    div = (run.get("diversification_summary", {}) or {}) if isinstance(run, dict) else {}
    uni = (run.get("universe_summary", {}) or {}) if isinstance(run, dict) else {}

    inverse_concentration = _extract_inverse_concentration_proxy(run)
    diversification_candidates = [
        ("mean_diversification_ratio", _safe_float(div.get("mean_diversification_ratio"))),
        ("mean_effective_risk_bets", _safe_float(risk.get("mean_effective_risk_bets"))),
        ("mean_effective_breadth", _safe_float(div.get("mean_effective_breadth"))),
        ("inverse_concentration", _safe_float(inverse_concentration)),
        ("mean_active_assets", _safe_float(uni.get("mean_active_assets"))),
    ]
    diversification_value = np.nan
    diversification_source = None
    for name, value in diversification_candidates:
        if np.isfinite(value):
            diversification_value = float(value)
            diversification_source = str(name)
            break

    lookup = {
        "sharpe": _safe_float(perf.get("sharpe")),
        "cagr": _safe_float(perf.get("cagr")),
        "information_ratio": _safe_float(perf.get("information_ratio")),
        "active_return_annual": _safe_float(perf.get("active_return_annual")),
        "max_drawdown": _safe_float(perf.get("max_drawdown")),
        "annual_volatility": _safe_float(perf.get("annual_volatility", perf.get("annualized_volatility"))),
        "mean_turnover": _safe_float(perf.get("mean_turnover")),
        "tracking_error_annual": _safe_float(perf.get("tracking_error_annual")),
        "mean_effective_breadth": _safe_float(div.get("mean_effective_breadth")),
        "mean_diversification_ratio": _safe_float(div.get("mean_diversification_ratio")),
        "mean_effective_risk_bets": _safe_float(risk.get("mean_effective_risk_bets")),
        "mean_active_assets": _safe_float(uni.get("mean_active_assets")),
        "mean_effective_n_assets": _safe_float(div.get("mean_effective_n_assets")),
        "inverse_concentration": _safe_float(inverse_concentration),
        "diversification": _safe_float(diversification_value),
    }
    lookup["diversification_source"] = diversification_source
    return lookup


def _normalize_higher_better_metric(value: Any, *, center: float, scale: float) -> float:
    x = _safe_float(value)
    if not np.isfinite(x):
        return np.nan
    scl = max(abs(float(scale)), 1e-12)
    return float(0.5 * (1.0 + np.tanh((x - float(center)) / scl)))


def _normalize_lower_better_metric(value: Any, *, center: float, scale: float) -> float:
    x = _safe_float(value)
    if not np.isfinite(x):
        return np.nan
    scl = max(abs(float(scale)), 1e-12)
    return float(0.5 * (1.0 + np.tanh((float(center) - x) / scl)))


def _normalize_drawdown_metric(value: Any) -> float:
    x = _safe_float(value)
    if not np.isfinite(x):
        return np.nan
    dd_abs = abs(float(x))
    return _normalize_lower_better_metric(dd_abs, center=0.20, scale=0.12)


def _normalize_diversification_metric(value: Any, source: Optional[str]) -> float:
    x = _safe_float(value)
    if not np.isfinite(x):
        return np.nan
    src = str(source or "").strip().lower()
    if src == "mean_diversification_ratio":
        return _normalize_higher_better_metric(x, center=1.30, scale=0.60)
    if src in {
        "mean_effective_risk_bets",
        "mean_effective_breadth",
        "mean_effective_n_assets",
        "inverse_concentration",
        "mean_active_assets",
    }:
        clipped = max(float(x), 1.0)
        return float(np.clip(np.log(clipped) / np.log(10.0), 0.0, 1.0))
    return _normalize_higher_better_metric(x, center=2.0, scale=1.0)


def _resolve_multi_objective_preset(objective: str) -> str:
    obj = str(objective or "multi_objective").strip().lower()
    alias_map = {
        "multi_objective": "composite_balanced",
        "composite": "composite_balanced",
        "balanced": "composite_balanced",
        "growth": "composite_growth",
        "defensive": "composite_defensive",
        "quality": "composite_quality",
        "turnover": "composite_low_turnover",
        "low_turnover": "composite_low_turnover",
        "diversified": "composite_diversified",
        "diversification": "composite_diversified",
        "robust": "composite_robust",
    }
    return alias_map.get(obj, obj)


def _multi_objective_weights(preset: str) -> Dict[str, float]:
    name = _resolve_multi_objective_preset(preset)
    presets = {
        "composite_balanced": {
            "sharpe": 0.32,
            "cagr": 0.22,
            "max_drawdown": 0.20,
            "mean_turnover": 0.12,
            "diversification": 0.14,
        },
        "composite_growth": {
            "sharpe": 0.24,
            "cagr": 0.38,
            "max_drawdown": 0.16,
            "mean_turnover": 0.08,
            "diversification": 0.14,
        },
        "composite_defensive": {
            "sharpe": 0.26,
            "cagr": 0.14,
            "max_drawdown": 0.30,
            "mean_turnover": 0.16,
            "diversification": 0.14,
        },
        "composite_quality": {
            "sharpe": 0.38,
            "cagr": 0.18,
            "max_drawdown": 0.20,
            "mean_turnover": 0.12,
            "diversification": 0.12,
        },
        "composite_low_turnover": {
            "sharpe": 0.24,
            "cagr": 0.16,
            "max_drawdown": 0.16,
            "mean_turnover": 0.28,
            "diversification": 0.16,
        },
        "composite_diversified": {
            "sharpe": 0.22,
            "cagr": 0.16,
            "max_drawdown": 0.18,
            "mean_turnover": 0.10,
            "diversification": 0.34,
        },
        "composite_robust": {
            "sharpe": 0.30,
            "cagr": 0.16,
            "max_drawdown": 0.24,
            "mean_turnover": 0.18,
            "diversification": 0.12,
        },
    }
    return dict(presets.get(name, presets["composite_balanced"]))


def compute_multi_objective_score(
    run: Dict[str, Any],
    *,
    objective: str = "multi_objective",
) -> Dict[str, Any]:
    preset = _resolve_multi_objective_preset(objective)
    metrics = _extract_tuning_metric_bundle(run)
    diversification_source = metrics.get("diversification_source")

    normalized = {
        "sharpe": _normalize_higher_better_metric(metrics.get("sharpe"), center=0.75, scale=0.75),
        "cagr": _normalize_higher_better_metric(metrics.get("cagr"), center=0.08, scale=0.08),
        "max_drawdown": _normalize_drawdown_metric(metrics.get("max_drawdown")),
        "mean_turnover": _normalize_lower_better_metric(metrics.get("mean_turnover"), center=0.18, scale=0.12),
        "diversification": _normalize_diversification_metric(metrics.get("diversification"), diversification_source),
    }

    weights = _multi_objective_weights(preset)
    weighted_sum = 0.0
    total_weight = 0.0
    active_components: List[str] = []
    missing_components: List[str] = []
    for metric_name, weight in weights.items():
        score = _safe_float(normalized.get(metric_name))
        if np.isfinite(score):
            weighted_sum += float(weight) * float(score)
            total_weight += float(weight)
            active_components.append(str(metric_name))
        else:
            missing_components.append(str(metric_name))

    composite_score = float(weighted_sum / total_weight) if total_weight > 0 else np.nan

    out = {
        "objective_name": str(objective),
        "composite_objective_preset": preset,
        "composite_objective_score": composite_score,
        "composite_objective_weight_sum": float(total_weight),
        "composite_objective_active_components": "|".join(active_components),
        "composite_objective_missing_components": "|".join(missing_components),
        "composite_diversification_source": diversification_source,
        "composite_raw_sharpe": _safe_float(metrics.get("sharpe")),
        "composite_raw_cagr": _safe_float(metrics.get("cagr")),
        "composite_raw_max_drawdown": _safe_float(metrics.get("max_drawdown")),
        "composite_raw_mean_turnover": _safe_float(metrics.get("mean_turnover")),
        "composite_raw_diversification": _safe_float(metrics.get("diversification")),
        "composite_sharpe_score": _safe_float(normalized.get("sharpe")),
        "composite_cagr_score": _safe_float(normalized.get("cagr")),
        "composite_drawdown_score": _safe_float(normalized.get("max_drawdown")),
        "composite_turnover_score": _safe_float(normalized.get("mean_turnover")),
        "composite_diversification_score": _safe_float(normalized.get("diversification")),
    }
    for metric_name, weight in weights.items():
        out[f"composite_weight_{metric_name}"] = float(weight)
    return out


def _extract_tuning_objective_value(run: Dict[str, Any], objective: str = "sharpe") -> float:
    """Extract a tuning objective from performance/risk/diversification/run report blocks."""
    obj = str(objective or "sharpe")
    obj_norm = _resolve_multi_objective_preset(obj)
    if obj_norm.startswith("composite_"):
        return _safe_float(compute_multi_objective_score(run, objective=obj_norm).get("composite_objective_score"))

    lookup = _extract_tuning_metric_bundle(run)
    if obj in lookup:
        return _safe_float(lookup.get(obj))

    if obj.startswith("-"):
        base = _extract_tuning_objective_value(run, obj[1:])
        return -base if np.isfinite(base) else np.nan

    if obj.startswith("min_"):
        base = _extract_tuning_objective_value(run, obj[4:])
        return -base if np.isfinite(base) else np.nan

    perf = _run_perf_dict(run)
    return _safe_float(perf.get(obj))


def _run_objective_once(
    panel_df,
    *,
    base_cfg_payload: Dict[str, Any],
    param_dict: Dict[str, Any],
    objective: str = "sharpe",
):
    """Run pipeline once and compute objective value."""
    try:
        from src.investment import (
            MicroPipelineConfig,
            config_from_dict,
            config_fingerprint,
            run_micro_investment_pipeline,
        )
    except Exception:
        from investment import (
            MicroPipelineConfig,
            config_from_dict,
            config_fingerprint,
            run_micro_investment_pipeline,
        )

    valid_fields = {f.name for f in MicroPipelineConfig.__dataclass_fields__.values()}
    payload = {k: v for k, v in dict(base_cfg_payload or {}).items() if k in valid_fields}
    for k, v in dict(param_dict or {}).items():
        if k in valid_fields:
            payload[k] = v

    cfg = config_from_dict(payload)
    run = run_micro_investment_pipeline(panel_df, cfg=cfg)
    val = _extract_tuning_objective_value(run, objective=objective)
    perf = _run_perf_dict(run)
    risk = run.get("risk_summary", {}) or {}
    div = run.get("diversification_summary", {}) or {}
    uni = run.get("universe_summary", {}) or {}

    composite_summary = compute_multi_objective_score(run, objective=objective)

    summary = {
        "objective_name": str(objective),
        "objective_value": val,
        "config_fingerprint": config_fingerprint(cfg),
        "sharpe": _safe_float(perf.get("sharpe")),
        "cagr": _safe_float(perf.get("cagr")),
        "max_drawdown": _safe_float(perf.get("max_drawdown")),
        "annual_volatility": _safe_float(perf.get("annual_volatility", perf.get("annualized_volatility"))),
        "mean_turnover": _safe_float(perf.get("mean_turnover")),
        "information_ratio": _safe_float(perf.get("information_ratio")),
        "active_return_annual": _safe_float(perf.get("active_return_annual")),
        "tracking_error_annual": _safe_float(perf.get("tracking_error_annual")),
        "mean_effective_breadth": _safe_float(div.get("mean_effective_breadth")),
        "mean_diversification_ratio": _safe_float(div.get("mean_diversification_ratio")),
        "mean_effective_risk_bets": _safe_float(risk.get("mean_effective_risk_bets")),
        "mean_active_assets": _safe_float(uni.get("mean_active_assets")),
        "mean_effective_n_assets": _safe_float(div.get("mean_effective_n_assets")),
        "inverse_concentration": _safe_float(_extract_inverse_concentration_proxy(run)),
    }
    summary.update(composite_summary)
    return val, run, summary



def _trial_row_from_result(
    trial_idx: int,
    *,
    method: str,
    params: Dict[str, Any],
    result_summary: Dict[str, Any],
    status: str = "ok",
    error: Optional[str] = None,
) -> Dict[str, Any]:
    row = dict(params or {})
    row.update(dict(result_summary or {}))
    row["trial"] = int(trial_idx)
    row["method"] = str(method)
    row["status"] = str(status)
    row["error"] = None if error in {None, ""} else str(error)
    return row



def _random_search_normalize_space_spec(spec: Any) -> Tuple[Optional[Any], Dict[str, Any]]:
    meta: Dict[str, Any] = {"status": "ok", "message": "", "effective_n": np.nan}

    if isinstance(spec, list):
        cleaned: List[Any] = []
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



def _prepare_random_search_param_space(param_space: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    raw = dict(param_space or {})
    cleaned: Dict[str, Any] = {}
    dropped: List[str] = []
    fixed_like: List[str] = []
    invalid_like: List[str] = []

    for key, spec in raw.items():
        normalized, meta = _random_search_normalize_space_spec(spec)
        if normalized is None:
            dropped.append(str(key))
            msg = str(meta.get("message", "invalid_spec"))
            if msg in {"single_choice_only", "zero_width_range", "fixed_or_unsupported_spec"}:
                fixed_like.append(str(key))
            else:
                invalid_like.append(str(key))
            continue
        cleaned[str(key)] = normalized

    summary = {
        "raw_param_count": int(len(raw)),
        "space_dim": int(len(cleaned)),
        "dropped_param_count": int(len(dropped)),
        "dropped_params": dropped,
        "fixed_like_params": fixed_like,
        "invalid_like_params": invalid_like,
        "has_active_variation": bool(len(cleaned) > 0),
        "space_status": "active" if len(cleaned) > 0 else "degenerate",
    }
    return cleaned, summary



def run_random_tuning(
    panel_df,
    *,
    base_cfg_payload: Dict[str, Any],
    param_space: Dict[str, Any],
    n_trials: int = 20,
    objective: str = "sharpe",
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """Simple random search tuner with light search-space validation and telemetry."""
    if panel_df is None or not isinstance(panel_df, pd.DataFrame) or panel_df.empty:
        return pd.DataFrame()
    if random_seed is not None:
        random.seed(int(random_seed))
        np.random.seed(int(random_seed))

    sanitized_space, space_info = _prepare_random_search_param_space(param_space)
    if not bool(space_info.get("has_active_variation", False)):
        skipped = _trial_row_from_result(
            0,
            method="random",
            params={},
            result_summary={"objective_name": str(objective), "objective_value": np.nan},
            status="skipped",
            error="random_search_no_active_param_space",
        )
        skipped.update({
            "random_search_space_dim": int(space_info.get("space_dim", 0)),
            "random_search_raw_param_count": int(space_info.get("raw_param_count", 0)),
            "random_search_dropped_param_count": int(space_info.get("dropped_param_count", 0)),
            "random_search_has_active_variation": bool(space_info.get("has_active_variation", False)),
            "random_search_duplicate_sample": False,
            "random_search_unique_sample_count": 0,
            "random_search_space_status": str(space_info.get("space_status", "degenerate")),
            "random_search_dropped_params": "|".join(space_info.get("dropped_params", [])),
            "random_search_fixed_like_params": "|".join(space_info.get("fixed_like_params", [])),
            "random_search_invalid_like_params": "|".join(space_info.get("invalid_like_params", [])),
            "random_search_duplicate_rate": 0.0,
        })
        return pd.DataFrame([skipped])

    rows: List[Dict[str, Any]] = []
    seen_param_keys: set = set()
    unique_sample_count = 0
    total_trials = int(max(0, n_trials))

    for i in range(total_trials):
        params = _sample_param_dict(sanitized_space)
        param_key = json.dumps(params, sort_keys=True, default=str)
        duplicate_sample = param_key in seen_param_keys
        if not duplicate_sample:
            seen_param_keys.add(param_key)
            unique_sample_count += 1
        try:
            _, _, summary = _run_objective_once(
                panel_df,
                base_cfg_payload=base_cfg_payload,
                param_dict=params,
                objective=objective,
            )
            row = _trial_row_from_result(i, method="random", params=params, result_summary=summary)
        except Exception as exc:
            row = _trial_row_from_result(
                i,
                method="random",
                params=params,
                result_summary={"objective_name": str(objective), "objective_value": np.nan},
                status="error",
                error=exc,
            )
        row.update({
            "random_search_space_dim": int(space_info.get("space_dim", 0)),
            "random_search_raw_param_count": int(space_info.get("raw_param_count", 0)),
            "random_search_dropped_param_count": int(space_info.get("dropped_param_count", 0)),
            "random_search_has_active_variation": bool(space_info.get("has_active_variation", False)),
            "random_search_duplicate_sample": bool(duplicate_sample),
            "random_search_unique_sample_count": int(unique_sample_count),
            "random_search_space_status": str(space_info.get("space_status", "active")),
            "random_search_dropped_params": "|".join(space_info.get("dropped_params", [])),
            "random_search_fixed_like_params": "|".join(space_info.get("fixed_like_params", [])),
            "random_search_invalid_like_params": "|".join(space_info.get("invalid_like_params", [])),
        })
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    duplicate_rate = float(pd.to_numeric(df.get("random_search_duplicate_sample"), errors="coerce").fillna(0.0).mean()) if "random_search_duplicate_sample" in df.columns else np.nan
    df["random_search_duplicate_rate"] = duplicate_rate
    return df



def run_bayesian_style_tuning(
    panel_df,
    *,
    base_cfg_payload: Dict[str, Any],
    param_space: Dict[str, Any],
    n_trials: int = 30,
    warmup: int = 5,
    objective: str = "sharpe",
    local_width: float = 0.25,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """Simple Bayesian-like tuner with light search-space validation and telemetry.

    The objective convention remains maximisation-based; minimisation objectives are
    already converted by _extract_tuning_objective_value via sign inversion.
    """
    if panel_df is None or not isinstance(panel_df, pd.DataFrame) or panel_df.empty:
        return pd.DataFrame()
    if random_seed is not None:
        random.seed(int(random_seed))
        np.random.seed(int(random_seed))

    sanitized_space, space_info = _prepare_random_search_param_space(param_space)
    requested_warmup = int(max(0, warmup))
    total_trials = int(max(0, n_trials))
    effective_warmup = int(min(requested_warmup, max(total_trials - 1, 0))) if total_trials > 0 else 0

    if not bool(space_info.get("has_active_variation", False)):
        skipped = _trial_row_from_result(
            0,
            method="bayesian_style",
            params={},
            result_summary={"objective_name": str(objective), "objective_value": np.nan},
            status="skipped",
            error="bayesian_style_no_active_param_space",
        )
        skipped.update({
            "bayesian_search_space_dim": int(space_info.get("space_dim", 0)),
            "bayesian_search_raw_param_count": int(space_info.get("raw_param_count", 0)),
            "bayesian_search_dropped_param_count": int(space_info.get("dropped_param_count", 0)),
            "bayesian_search_has_active_variation": bool(space_info.get("has_active_variation", False)),
            "bayesian_search_space_status": str(space_info.get("space_status", "degenerate")),
            "bayesian_search_dropped_params": "|".join(space_info.get("dropped_params", [])),
            "bayesian_search_fixed_like_params": "|".join(space_info.get("fixed_like_params", [])),
            "bayesian_search_invalid_like_params": "|".join(space_info.get("invalid_like_params", [])),
            "bayesian_search_requested_warmup": requested_warmup,
            "bayesian_search_effective_warmup": effective_warmup,
            "bayesian_search_local_width": float(max(local_width, 1e-6)),
            "bayesian_search_phase": "skipped",
            "bayesian_search_used_incumbent": False,
            "bayesian_search_best_val_so_far": np.nan,
            "bayesian_search_incumbent_updates": 0,
            "bayesian_search_duplicate_sample": False,
            "bayesian_search_unique_sample_count": 0,
            "bayesian_search_duplicate_rate": 0.0,
        })
        return pd.DataFrame([skipped])

    rows: List[Dict[str, Any]] = []
    best_params: Optional[Dict[str, Any]] = None
    best_val = -np.inf
    incumbent_updates = 0
    seen_param_keys: set = set()
    unique_sample_count = 0
    local_width_eff = float(max(local_width, 1e-6))

    for i in range(total_trials):
        use_random_warmup = i < effective_warmup or best_params is None
        phase = "warmup" if use_random_warmup else "local"
        used_incumbent = False

        if use_random_warmup:
            params = _sample_param_dict(sanitized_space)
        else:
            params = {}
            for k, v in (sanitized_space or {}).items():
                if isinstance(v, tuple) and len(v) == 2:
                    lo, hi = v
                    center = best_params.get(k, (lo + hi) / 2)
                    width = max(float(hi) - float(lo), 0.0) * local_width_eff
                    a = max(float(lo), float(center) - width)
                    b = min(float(hi), float(center) + width)
                    if isinstance(lo, int) and isinstance(hi, int):
                        params[k] = random.randint(int(np.floor(a)), int(np.ceil(b)))
                    else:
                        params[k] = random.uniform(float(a), float(b))
                elif isinstance(v, list):
                    if not v:
                        continue
                    incumbent = best_params.get(k)
                    if incumbent in v and random.random() < 0.60:
                        params[k] = incumbent
                        used_incumbent = True
                    else:
                        params[k] = random.choice(v)
                else:
                    params[k] = v

        param_key = json.dumps(params, sort_keys=True, default=str)
        duplicate_sample = param_key in seen_param_keys
        if not duplicate_sample:
            seen_param_keys.add(param_key)
            unique_sample_count += 1

        incumbent_improved = False
        try:
            val, _, summary = _run_objective_once(
                panel_df,
                base_cfg_payload=base_cfg_payload,
                param_dict=params,
                objective=objective,
            )
            row = _trial_row_from_result(i, method="bayesian_style", params=params, result_summary=summary)
            if np.isfinite(_safe_float(val)) and float(val) > float(best_val):
                best_val = float(val)
                best_params = dict(params)
                incumbent_updates += 1
                incumbent_improved = True
        except Exception as exc:
            row = _trial_row_from_result(
                i,
                method="bayesian_style",
                params=params,
                result_summary={"objective_name": str(objective), "objective_value": np.nan},
                status="error",
                error=exc,
            )

        row.update({
            "bayesian_search_space_dim": int(space_info.get("space_dim", 0)),
            "bayesian_search_raw_param_count": int(space_info.get("raw_param_count", 0)),
            "bayesian_search_dropped_param_count": int(space_info.get("dropped_param_count", 0)),
            "bayesian_search_has_active_variation": bool(space_info.get("has_active_variation", False)),
            "bayesian_search_space_status": str(space_info.get("space_status", "active")),
            "bayesian_search_dropped_params": "|".join(space_info.get("dropped_params", [])),
            "bayesian_search_fixed_like_params": "|".join(space_info.get("fixed_like_params", [])),
            "bayesian_search_invalid_like_params": "|".join(space_info.get("invalid_like_params", [])),
            "bayesian_search_requested_warmup": requested_warmup,
            "bayesian_search_effective_warmup": effective_warmup,
            "bayesian_search_local_width": local_width_eff,
            "bayesian_search_phase": phase,
            "bayesian_search_used_incumbent": bool(used_incumbent),
            "bayesian_search_best_val_so_far": float(best_val) if np.isfinite(best_val) else np.nan,
            "bayesian_search_incumbent_updates": int(incumbent_updates),
            "bayesian_search_incumbent_improved": bool(incumbent_improved),
            "bayesian_search_duplicate_sample": bool(duplicate_sample),
            "bayesian_search_unique_sample_count": int(unique_sample_count),
        })
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    duplicate_rate = float(pd.to_numeric(df.get("bayesian_search_duplicate_sample"), errors="coerce").fillna(0.0).mean()) if "bayesian_search_duplicate_sample" in df.columns else np.nan
    df["bayesian_search_duplicate_rate"] = duplicate_rate
    return df



def _surrogate_feature_names(keys: Sequence[str], param_space: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for k in keys:
        spec = param_space.get(k)
        if isinstance(spec, list):
            values = list(spec)
            if len(values) <= 1:
                names.append(str(k))
            else:
                for cand in values:
                    names.append(f"{k}__{cand}")
        else:
            names.append(str(k))
    return names



def _encode_single_param_dict_for_surrogate(
    params: Dict[str, Any],
    keys: Sequence[str],
    param_space: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], bool]:
    vec: List[float] = []
    for k in keys:
        spec = param_space.get(k)
        value = params.get(k)
        if isinstance(spec, tuple) and len(spec) == 2:
            lo, hi = spec
            vv = _safe_float(value)
            vlo = _safe_float(lo)
            vhi = _safe_float(hi)
            if not np.isfinite(vv) or not np.isfinite(vlo) or not np.isfinite(vhi):
                return None, False
            width = float(vhi) - float(vlo)
            if abs(width) <= 1e-12:
                vec.append(0.0)
            else:
                vec.append(float((float(vv) - float(vlo)) / width))
        elif isinstance(spec, list):
            values = list(spec)
            if len(values) <= 1:
                vec.append(1.0)
            else:
                matched = False
                for cand in values:
                    bit = 1.0 if value == cand else 0.0
                    vec.append(bit)
                    matched = matched or (bit > 0.5)
                if not matched:
                    return None, False
        else:
            vv = _safe_float(value)
            if np.isfinite(vv):
                vec.append(float(vv))
            else:
                vec.append(1.0 if value == spec else 0.0)
    x = np.asarray(vec, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        return None, False
    return x, True



def _encode_params_for_surrogate(
    trials_df: pd.DataFrame,
    keys: Sequence[str],
    param_space: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    X_rows: List[np.ndarray] = []
    valid_index: List[int] = []
    for idx, row in trials_df.iterrows():
        params = {k: row.get(k) for k in keys}
        x, ok = _encode_single_param_dict_for_surrogate(params, keys, param_space)
        if ok and x is not None:
            X_rows.append(x)
            valid_index.append(idx)
    if not X_rows:
        return np.empty((0, 0), dtype=float), np.asarray([], dtype=int)
    return np.vstack(X_rows).astype(float), np.asarray(valid_index, dtype=int)



def _fit_linear_surrogate_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    ridge_lambda: float = 1e-3,
) -> Optional[Dict[str, Any]]:
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0] or X.shape[0] < 3:
        return None
    x_mean = np.nanmean(X, axis=0)
    x_std = np.nanstd(X, axis=0)
    x_std = np.where(np.isfinite(x_std) & (x_std > 1e-12), x_std, 1.0)
    Xs = (X - x_mean) / x_std
    Xd = np.column_stack([np.ones(Xs.shape[0]), Xs])
    penalty = float(max(ridge_lambda, 0.0))
    eye = np.eye(Xd.shape[1], dtype=float)
    eye[0, 0] = 0.0
    try:
        beta = np.linalg.solve(Xd.T @ Xd + penalty * eye, Xd.T @ y)
    except Exception:
        try:
            beta = np.linalg.lstsq(Xd, y, rcond=None)[0]
        except Exception:
            return None
    pred = Xd @ beta
    resid = y - pred
    resid_std = float(np.nanstd(resid, ddof=1)) if resid.shape[0] > 1 else 0.0
    return {
        "beta": beta,
        "x_mean": x_mean,
        "x_std": x_std,
        "resid_std": resid_std if np.isfinite(resid_std) else np.nan,
        "n_fit": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }



def _predict_with_linear_surrogate(model: Dict[str, Any], x: np.ndarray) -> float:
    try:
        x_mean = np.asarray(model.get("x_mean"), dtype=float)
        x_std = np.asarray(model.get("x_std"), dtype=float)
        beta = np.asarray(model.get("beta"), dtype=float)
        xs = (np.asarray(x, dtype=float) - x_mean) / np.where(np.abs(x_std) > 1e-12, x_std, 1.0)
        xd = np.concatenate([[1.0], xs])
        return float(np.dot(xd, beta)) if np.all(np.isfinite(xd)) and np.all(np.isfinite(beta)) else np.nan
    except Exception:
        return np.nan



def _min_candidate_distance(x: np.ndarray, X_hist: np.ndarray) -> float:
    if X_hist.ndim != 2 or X_hist.shape[0] == 0:
        return np.nan
    try:
        d = np.sqrt(((X_hist - x.reshape(1, -1)) ** 2).sum(axis=1))
        return float(np.nanmin(d)) if d.size else np.nan
    except Exception:
        return np.nan



def run_surrogate_tuning(
    panel_df,
    *,
    base_cfg_payload: Dict[str, Any],
    param_space: Dict[str, Any],
    n_trials: int = 40,
    objective: str = "sharpe",
    warmup: int = 8,
    candidate_pool_size: int = 64,
    exploration_prob: float = 0.25,
    random_seed: Optional[int] = None,
    ridge_lambda: float = 1e-3,
    distance_bonus_strength: float = 0.10,
    exploit_top_k: int = 5,
) -> pd.DataFrame:
    """Light surrogate tuner with one-hot categorical encoding and distance-aware candidate selection."""
    if panel_df is None or not isinstance(panel_df, pd.DataFrame) or panel_df.empty:
        return pd.DataFrame()
    if random_seed is not None:
        random.seed(int(random_seed))
        np.random.seed(int(random_seed))

    rows: List[Dict[str, Any]] = []
    keys = list((param_space or {}).keys())
    feature_names = _surrogate_feature_names(keys, param_space)
    feature_dim = len(feature_names)
    min_fit_obs = max(6, feature_dim + 2)
    effective_warmup = max(int(max(0, warmup)), min_fit_obs)
    candidate_pool_n = int(max(12, candidate_pool_size))
    exploration_prob = float(np.clip(exploration_prob, 0.0, 1.0))
    distance_bonus_strength = float(max(distance_bonus_strength, 0.0))
    exploit_top_k = int(max(1, exploit_top_k))

    for i in range(int(max(0, n_trials))):
        params = _sample_param_dict(param_space)
        surrogate_used = False
        surrogate_ready = False
        surrogate_feature_dim = feature_dim
        surrogate_n_fit = 0
        surrogate_resid_std = np.nan
        surrogate_reason = "warmup"

        if i >= effective_warmup and rows:
            trials_so_far = pd.DataFrame(rows)
            trials_ok = trials_so_far[trials_so_far.get("status", "ok").astype(str) == "ok"].copy() if "status" in trials_so_far.columns else trials_so_far.copy()
            X_hist, valid_idx = _encode_params_for_surrogate(trials_ok, keys, param_space)
            y_all = pd.to_numeric(trials_ok.get("objective_value"), errors="coerce")
            if X_hist.shape[0] >= min_fit_obs and y_all.shape[0] > 0:
                y_fit_all = y_all.iloc[valid_idx].to_numpy(dtype=float)
                good = np.isfinite(y_fit_all)
                X_fit = X_hist[good]
                y_fit = y_fit_all[good]
                if X_fit.shape[0] >= min_fit_obs:
                    model = _fit_linear_surrogate_model(X_fit, y_fit, ridge_lambda=ridge_lambda)
                    surrogate_ready = model is not None
                    if model is not None:
                        surrogate_n_fit = int(model.get("n_fit", 0) or 0)
                        surrogate_resid_std = _safe_float(model.get("resid_std"))
                        if random.random() <= (1.0 - exploration_prob):
                            scored_candidates: List[Tuple[float, float, float, Dict[str, Any]]] = []
                            for _ in range(candidate_pool_n):
                                cand = _sample_param_dict(param_space)
                                x_cand, ok = _encode_single_param_dict_for_surrogate(cand, keys, param_space)
                                if not ok or x_cand is None:
                                    continue
                                pred = _predict_with_linear_surrogate(model, x_cand)
                                if not np.isfinite(pred):
                                    continue
                                novelty = _min_candidate_distance(x_cand, X_fit)
                                novelty = 0.0 if not np.isfinite(novelty) else float(novelty)
                                score = float(pred) + distance_bonus_strength * novelty
                                scored_candidates.append((score, float(pred), novelty, cand))
                            if scored_candidates:
                                scored_candidates.sort(key=lambda z: z[0], reverse=True)
                                top_pool = scored_candidates[: min(len(scored_candidates), exploit_top_k)]
                                params = random.choice(top_pool)[3]
                                surrogate_used = True
                                surrogate_reason = "surrogate_guided"
                            else:
                                surrogate_reason = "candidate_encoding_failed"
                        else:
                            surrogate_reason = "exploration_branch"
                    else:
                        surrogate_reason = "surrogate_fit_failed"
                else:
                    surrogate_reason = "insufficient_finite_objective_obs"
            else:
                surrogate_reason = "insufficient_fit_rows"

        try:
            _, _, summary = _run_objective_once(
                panel_df,
                base_cfg_payload=base_cfg_payload,
                param_dict=params,
                objective=objective,
            )
            row = _trial_row_from_result(i, method="surrogate", params=params, result_summary=summary)
            row["surrogate_used"] = bool(surrogate_used)
            row["surrogate_ready"] = bool(surrogate_ready)
            row["surrogate_reason"] = str(surrogate_reason)
            row["surrogate_feature_dim"] = int(surrogate_feature_dim)
            row["surrogate_n_fit"] = int(surrogate_n_fit)
            row["surrogate_effective_warmup"] = int(effective_warmup)
            row["surrogate_candidate_pool_size"] = int(candidate_pool_n)
            row["surrogate_resid_std"] = _safe_float(surrogate_resid_std)
            rows.append(row)
        except Exception as exc:
            row = _trial_row_from_result(i, method="surrogate", params=params, result_summary={"objective_name": str(objective), "objective_value": np.nan}, status="error", error=exc)
            row["surrogate_used"] = bool(surrogate_used)
            row["surrogate_ready"] = bool(surrogate_ready)
            row["surrogate_reason"] = str(surrogate_reason)
            row["surrogate_feature_dim"] = int(surrogate_feature_dim)
            row["surrogate_n_fit"] = int(surrogate_n_fit)
            row["surrogate_effective_warmup"] = int(effective_warmup)
            row["surrogate_candidate_pool_size"] = int(candidate_pool_n)
            row["surrogate_resid_std"] = _safe_float(surrogate_resid_std)
            rows.append(row)
    return pd.DataFrame(rows)



def run_optuna_tuning(
    panel_df,
    *,
    base_cfg_payload: Dict[str, Any],
    param_space: Dict[str, Any],
    n_trials: int = 30,
    objective: str = "sharpe",
    study_name: Optional[str] = None,
    random_seed: Optional[int] = None,
    direction: Optional[str] = None,
    fallback_to_bayesian_style: bool = True,
) -> pd.DataFrame:
    """Optuna wrapper with graceful fallback when optuna is unavailable.

    Adds explicit telemetry so reporting can distinguish real Optuna runs from a
    Bayesian-style fallback, instead of relying only on the `method` column.
    """
    if panel_df is None or not isinstance(panel_df, pd.DataFrame) or panel_df.empty:
        return pd.DataFrame()
    if random_seed is not None:
        random.seed(int(random_seed))
        np.random.seed(int(random_seed))

    sanitized_space, space_info = _prepare_random_search_param_space(param_space)
    inferred_direction = str(direction or ("maximize" if not str(objective).startswith(("-", "min_")) else "minimize"))

    def _apply_optuna_meta(df: pd.DataFrame, *, available: bool, fallback_active: bool, effective_method: str, fallback_reason: str = "") -> pd.DataFrame:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        out = df.copy()
        out["optuna_available"] = bool(available)
        out["optuna_requested_direction"] = str(inferred_direction)
        out["optuna_fallback_to_bayesian_style_requested"] = bool(fallback_to_bayesian_style)
        out["optuna_fallback_active"] = bool(fallback_active)
        out["optuna_effective_method"] = str(effective_method)
        out["optuna_requested_trial_count"] = int(max(0, n_trials))
        out["optuna_raw_param_count"] = int(space_info.get("raw_param_count", 0))
        out["optuna_space_dim"] = int(space_info.get("space_dim", 0))
        out["optuna_dropped_param_count"] = int(space_info.get("dropped_param_count", 0))
        out["optuna_space_status"] = str(space_info.get("space_status", "degenerate"))
        out["optuna_dropped_params"] = "|".join(space_info.get("dropped_params", []))
        out["optuna_fixed_like_params"] = "|".join(space_info.get("fixed_like_params", []))
        out["optuna_invalid_like_params"] = "|".join(space_info.get("invalid_like_params", []))
        out["optuna_has_active_variation"] = bool(space_info.get("has_active_variation", False))
        out["optuna_fallback_reason"] = str(fallback_reason or "")
        return out

    if not bool(space_info.get("has_active_variation", False)):
        skipped = _trial_row_from_result(
            0,
            method="optuna_skipped",
            params={},
            result_summary={"objective_name": str(objective), "objective_value": np.nan},
            status="skipped",
            error="optuna_no_active_param_space",
        )
        out = pd.DataFrame([skipped])
        out = _apply_optuna_meta(out, available=False, fallback_active=False, effective_method="skipped", fallback_reason="no_active_param_space")
        return out

    try:
        import optuna  # type: ignore
    except Exception:
        if fallback_to_bayesian_style:
            out = run_bayesian_style_tuning(
                panel_df,
                base_cfg_payload=base_cfg_payload,
                param_space=sanitized_space,
                n_trials=n_trials,
                objective=objective,
                random_seed=random_seed,
            )
            if out is None or not isinstance(out, pd.DataFrame):
                out = pd.DataFrame()
            if not out.empty:
                out["method"] = "optuna_fallback_bayesian_style"
                out = _apply_optuna_meta(
                    out,
                    available=False,
                    fallback_active=True,
                    effective_method="bayesian_style",
                    fallback_reason="optuna_import_failed",
                )
            return out
        skipped = _trial_row_from_result(
            0,
            method="optuna_unavailable",
            params={},
            result_summary={"objective_name": str(objective), "objective_value": np.nan},
            status="skipped",
            error="optuna_import_failed_no_fallback",
        )
        out = pd.DataFrame([skipped])
        out = _apply_optuna_meta(out, available=False, fallback_active=False, effective_method="unavailable", fallback_reason="optuna_import_failed")
        return out

    sampler = optuna.samplers.TPESampler(seed=int(random_seed)) if random_seed is not None else optuna.samplers.TPESampler()
    study = optuna.create_study(direction=inferred_direction, study_name=study_name, sampler=sampler)
    rows: List[Dict[str, Any]] = []

    def _suggest(trial, name: str, spec: Any):
        if isinstance(spec, tuple) and len(spec) == 2:
            lo, hi = spec
            if isinstance(lo, int) and isinstance(hi, int):
                return trial.suggest_int(name, int(lo), int(hi))
            return trial.suggest_float(name, float(lo), float(hi))
        if isinstance(spec, list):
            return trial.suggest_categorical(name, spec)
        return spec

    def objective_fn(trial):
        params = {k: _suggest(trial, k, spec) for k, spec in (sanitized_space or {}).items()}
        try:
            val, _, summary = _run_objective_once(
                panel_df,
                base_cfg_payload=base_cfg_payload,
                param_dict=params,
                objective=objective,
            )
            row = _trial_row_from_result(len(rows), method="optuna", params=params, result_summary=summary)
            row["optuna_trial_number"] = int(trial.number)
            row["optuna_study_direction"] = str(inferred_direction)
            rows.append(row)
            return float(val) if np.isfinite(_safe_float(val)) else -1e18
        except Exception as exc:
            row = _trial_row_from_result(
                len(rows),
                method="optuna",
                params=params,
                result_summary={"objective_name": str(objective), "objective_value": np.nan},
                status="error",
                error=exc,
            )
            row["optuna_trial_number"] = int(trial.number)
            row["optuna_study_direction"] = str(inferred_direction)
            rows.append(row)
            return -1e18

    study.optimize(objective_fn, n_trials=int(max(0, n_trials)))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = _apply_optuna_meta(out, available=True, fallback_active=False, effective_method="optuna", fallback_reason="")
    try:
        out["optuna_best_value"] = float(study.best_value)
        out["optuna_best_trial_number"] = int(study.best_trial.number)
    except Exception:
        pass
    return out



def _coerce_local_search_center(base_value: Any, default: float = 0.0) -> float:
    try:
        v = float(base_value)
    except Exception:
        v = float(default)
    return v if np.isfinite(v) else float(default)



def _coerce_local_search_grid(values: Sequence[Any], *, as_int: bool = False) -> List[Any]:
    cleaned: List[Any] = []
    seen: set = set()
    for item in list(values or []):
        try:
            v = int(round(float(item))) if as_int else float(item)
        except Exception:
            continue
        if not np.isfinite(float(v)):
            continue
        key = ("int", int(v)) if as_int else ("float", round(float(v), 12))
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(int(v) if as_int else float(v))
    return cleaned



def _infer_panel_asset_count(panel_df: pd.DataFrame, asset_col: str = "asset") -> int:
    if panel_df is None or not isinstance(panel_df, pd.DataFrame) or panel_df.empty or asset_col not in panel_df.columns:
        return 0
    try:
        return int(pd.Series(panel_df[asset_col]).dropna().astype(str).nunique())
    except Exception:
        return 0



def build_local_search_param_space(
    base_cfg_payload: Dict[str, Any],
    *,
    panel_df: Optional[pd.DataFrame] = None,
    temperature_multipliers: Sequence[float] = (0.8, 1.0, 1.2),
    top_k_offsets: Sequence[int] = (-2, 0, 2),
    overlay_multipliers: Sequence[float] = (0.7, 1.0, 1.3),
    target_vol_multipliers: Sequence[float] = (0.8, 1.0, 1.2),
    turnover_penalty_multipliers: Sequence[float] = (0.7, 1.0, 1.3),
    weight_shrink_multipliers: Sequence[float] = (),
) -> Dict[str, List[Any]]:
    base = dict(base_cfg_payload or {})
    n_assets = max(_infer_panel_asset_count(panel_df), 1)

    out: Dict[str, List[Any]] = {}

    temp_center = max(_coerce_local_search_center(base.get("temperature"), 1.0), 0.05)
    temp_vals = [max(temp_center * float(m), 0.05) for m in list(temperature_multipliers or [])]
    temp_vals = _coerce_local_search_grid(temp_vals, as_int=False)
    if len(temp_vals) >= 2:
        out["temperature"] = temp_vals

    raw_topk = base.get("top_k")
    if raw_topk is not None:
        try:
            topk_center = int(round(float(raw_topk)))
        except Exception:
            topk_center = None
        if topk_center is not None:
            topk_vals = []
            for offset in list(top_k_offsets or []):
                try:
                    candidate = int(round(topk_center + int(offset)))
                except Exception:
                    continue
                candidate = int(np.clip(candidate, 1, n_assets))
                topk_vals.append(candidate)
            topk_vals = _coerce_local_search_grid(topk_vals, as_int=True)
            if len(topk_vals) >= 2:
                out["top_k"] = topk_vals

    overlay_center = max(_coerce_local_search_center(base.get("probabilistic_overlay_strength"), 0.0), 0.0)
    overlay_vals = [max(overlay_center * float(m), 0.0) for m in list(overlay_multipliers or [])]
    overlay_vals = _coerce_local_search_grid(overlay_vals, as_int=False)
    if len(overlay_vals) >= 2:
        out["probabilistic_overlay_strength"] = overlay_vals

    target_vol_center = max(_coerce_local_search_center(base.get("target_portfolio_vol_monthly"), 0.04), 1e-4)
    target_vol_vals = [max(target_vol_center * float(m), 1e-4) for m in list(target_vol_multipliers or [])]
    target_vol_vals = _coerce_local_search_grid(target_vol_vals, as_int=False)
    if len(target_vol_vals) >= 2:
        out["target_portfolio_vol_monthly"] = target_vol_vals

    turnover_center = max(_coerce_local_search_center(base.get("turnover_penalty_strength"), 0.0), 0.0)
    turnover_vals = [max(turnover_center * float(m), 0.0) for m in list(turnover_penalty_multipliers or [])]
    turnover_vals = _coerce_local_search_grid(turnover_vals, as_int=False)
    if len(turnover_vals) >= 2:
        out["turnover_penalty_strength"] = turnover_vals

    shrink_mults = list(weight_shrink_multipliers or [])
    if shrink_mults:
        shrink_center = float(np.clip(_coerce_local_search_center(base.get("weight_shrink"), 0.05), 0.0, 1.0))
        shrink_vals = [float(np.clip(shrink_center * float(m), 0.0, 1.0)) for m in shrink_mults]
        shrink_vals = _coerce_local_search_grid(shrink_vals, as_int=False)
        if len(shrink_vals) >= 2:
            out["weight_shrink"] = shrink_vals

    return out



def build_local_search_param_space_from_policy(
    base_cfg_payload: Dict[str, Any],
    *,
    policy: Optional[Dict[str, Any]] = None,
    panel_df: Optional[pd.DataFrame] = None,
    raw_param_space: Optional[Dict[str, Sequence[Any]]] = None,
    include_weight_shrink: bool = False,
    temperature_multipliers: Sequence[float] = (0.8, 1.0, 1.2),
    top_k_offsets: Sequence[int] = (-2, 0, 2),
    overlay_multipliers: Sequence[float] = (0.7, 1.0, 1.3),
    target_vol_multipliers: Sequence[float] = (0.8, 1.0, 1.2),
    turnover_penalty_multipliers: Sequence[float] = (0.7, 1.0, 1.3),
    weight_shrink_multipliers: Sequence[float] = (0.7, 1.0, 1.3),
) -> Dict[str, List[Any]]:
    """Build a local-search param space and filter it according to a semantic policy.

    The policy is expected to come from ``resolve_simple_ui_tuning_policy(...)``
    in ``investment.py``. This helper does not execute search; it only converts
    the policy intent (mainly ``preferred_dims``) into a concrete filtered search
    space that ``auto_tune_around_config(...)`` can consume.

    Behaviour:
    - if ``raw_param_space`` is provided, it is filtered directly;
    - otherwise a default local grid is built around ``base_cfg_payload``;
    - if the policy has no usable ``preferred_dims`` overlap with the available
      space, the unfiltered space is returned as a safe fallback.
    """
    base_payload = dict(base_cfg_payload or {})
    policy_dict = dict(policy or {})

    available_space: Dict[str, List[Any]]
    if raw_param_space:
        available_space = {str(k): list(v) for k, v in dict(raw_param_space).items() if v is not None}
    else:
        available_space = build_local_search_param_space(
            base_payload,
            panel_df=panel_df,
            temperature_multipliers=temperature_multipliers,
            top_k_offsets=top_k_offsets,
            overlay_multipliers=overlay_multipliers,
            target_vol_multipliers=target_vol_multipliers,
            turnover_penalty_multipliers=turnover_penalty_multipliers,
            weight_shrink_multipliers=weight_shrink_multipliers if include_weight_shrink else (),
        )

    if not available_space:
        return {}

    preferred_dims_raw = policy_dict.get("preferred_dims", [])
    if isinstance(preferred_dims_raw, str):
        preferred_dims = [preferred_dims_raw]
    else:
        preferred_dims = [str(x) for x in list(preferred_dims_raw or []) if str(x).strip()]
    preferred_dims = list(dict.fromkeys(preferred_dims))
    if not preferred_dims:
        return dict(available_space)

    alias_map = {
        "overlay": "probabilistic_overlay_strength",
        "overlay_strength": "probabilistic_overlay_strength",
        "vol_target": "target_portfolio_vol_monthly",
        "target_vol": "target_portfolio_vol_monthly",
        "turnover_penalty": "turnover_penalty_strength",
        "shrink": "weight_shrink",
    }
    resolved_preferred: List[str] = []
    for dim in preferred_dims:
        resolved = alias_map.get(str(dim), str(dim))
        if resolved in available_space and resolved not in resolved_preferred:
            resolved_preferred.append(resolved)

    if not resolved_preferred:
        return dict(available_space)

    filtered = {name: list(available_space[name]) for name in resolved_preferred if name in available_space}
    return filtered or dict(available_space)



def _coerce_search_stage_enabled(value: Any, *, default: bool = True) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(value):
            return bool(default)
        return bool(int(value))
    txt = str(value).strip().lower()
    if txt in {"1", "true", "yes", "y", "on"}:
        return True
    if txt in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)



def _local_search_spec_values(spec: Any) -> List[Any]:
    if isinstance(spec, list):
        return list(spec)
    if isinstance(spec, tuple):
        return list(spec)
    return [spec]



def _param_value_equal(a: Any, b: Any) -> bool:
    try:
        af = float(a)
        bf = float(b)
        if np.isfinite(af) and np.isfinite(bf):
            return bool(abs(af - bf) <= 1e-12)
    except Exception:
        pass
    return a == b



def _param_value_distance(candidate: Any, base: Any) -> float:
    try:
        c = float(candidate)
        b = float(base)
        if np.isfinite(c) and np.isfinite(b):
            scale = max(abs(b), 1.0)
            return float(abs(c - b) / scale)
    except Exception:
        pass
    return 0.0 if candidate == base else 1.0



def _json_compact(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except Exception:
        return json.dumps(str(value))



def _candidate_params_key(params: Dict[str, Any]) -> str:
    return _json_compact(dict(sorted(dict(params or {}).items(), key=lambda kv: str(kv[0]))))



def _local_search_value_choices(local_space: Dict[str, Any], base_payload: Dict[str, Any], param_name: str) -> List[Any]:
    values = _local_search_spec_values(local_space.get(param_name))
    base_value = base_payload.get(param_name)
    out: List[Any] = []
    for value in values:
        if _param_value_equal(value, base_value):
            continue
        if not any(_param_value_equal(value, existing) for existing in out):
            out.append(value)
    return out



def _build_local_search_candidate_catalog(local_space: Dict[str, Any], base_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for param_name in local_space.keys():
        base_value = base_payload.get(param_name)
        for value in _local_search_value_choices(local_space, base_payload, param_name):
            rows.append({
                "param_name": str(param_name),
                "candidate_value": value,
                "base_value": base_value,
                "candidate_delta": _safe_float(value) - _safe_float(base_value) if np.isfinite(_safe_float(value)) and np.isfinite(_safe_float(base_value)) else np.nan,
                "distance_from_base": _param_value_distance(value, base_value),
            })
    return rows



def _evaluate_local_search_candidate(
    panel_df,
    *,
    trial_idx: int,
    method: str,
    stage: str,
    candidate_kind: str,
    base_cfg_payload: Dict[str, Any],
    objective: str,
    candidate_params: Dict[str, Any],
    baseline_val: Any,
    local_space: Dict[str, Any],
    local_search_budget: int,
    stage_rank: Optional[int] = None,
    seed_params: Optional[Dict[str, Any]] = None,
    tuned_param_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    try:
        _, _, summary = _run_objective_once(
            panel_df,
            base_cfg_payload=base_cfg_payload,
            param_dict=candidate_params,
            objective=objective,
        )
        row = _trial_row_from_result(
            trial_idx,
            method=method,
            params=candidate_params,
            result_summary=summary,
            status="ok",
        )
    except Exception as exc:
        row = _trial_row_from_result(
            trial_idx,
            method=method,
            params=candidate_params,
            result_summary={"objective_name": str(objective), "objective_value": np.nan},
            status="error",
            error=exc,
        )

    tuned_names = list(tuned_param_names or candidate_params.keys())
    base_value_map = {name: base_cfg_payload.get(name) for name in tuned_names}
    value_delta_map = {}
    value_distance_map = {}
    for name in tuned_names:
        cand_val = candidate_params.get(name)
        base_val = base_value_map.get(name)
        cand_num = _safe_float(cand_val)
        base_num = _safe_float(base_val)
        value_delta_map[str(name)] = cand_num - base_num if np.isfinite(cand_num) and np.isfinite(base_num) else np.nan
        value_distance_map[str(name)] = _param_value_distance(cand_val, base_val)

    objective_value = _safe_float(row.get("objective_value"))
    baseline_value = _safe_float(baseline_val)
    improvement = objective_value - baseline_value if np.isfinite(objective_value) and np.isfinite(baseline_value) else np.nan
    improvement_pct = (improvement / abs(baseline_value)) if np.isfinite(improvement) and np.isfinite(baseline_value) and abs(baseline_value) > 1e-12 else np.nan

    row.update({
        "tuned_param": "|".join(str(x) for x in tuned_names) if tuned_names else "baseline",
        "tuned_param_count": int(len(tuned_names)),
        "candidate_kind": str(candidate_kind),
        "candidate_stage": str(stage),
        "candidate_params_json": _json_compact(candidate_params),
        "candidate_param_names": "|".join(str(x) for x in tuned_names),
        "candidate_param_count": int(len(candidate_params)),
        "base_value": base_value_map[tuned_names[0]] if len(tuned_names) == 1 else _json_compact(base_value_map),
        "candidate_value": candidate_params.get(tuned_names[0]) if len(tuned_names) == 1 else _json_compact({k: candidate_params.get(k) for k in tuned_names}),
        "candidate_delta": value_delta_map.get(str(tuned_names[0]), np.nan) if len(tuned_names) == 1 else np.nan,
        "candidate_delta_pct": (value_delta_map.get(str(tuned_names[0]), np.nan) / abs(_safe_float(base_value_map[tuned_names[0]]))) if len(tuned_names) == 1 and np.isfinite(value_delta_map.get(str(tuned_names[0]), np.nan)) and np.isfinite(_safe_float(base_value_map[tuned_names[0]])) and abs(_safe_float(base_value_map[tuned_names[0]])) > 1e-12 else np.nan,
        "candidate_distance_from_base": float(sum(value_distance_map.values())),
        "candidate_delta_map_json": _json_compact(value_delta_map),
        "candidate_distance_map_json": _json_compact(value_distance_map),
        "local_search_budget": int(local_search_budget),
        "local_search_dim": int(len(local_space)),
        "local_search_objective": str(objective),
        "local_search_base_objective": baseline_value,
        "local_search_improvement": improvement,
        "local_search_improvement_pct": improvement_pct,
        "local_search_seed_params_json": _json_compact(seed_params or {}),
        "local_search_stage_rank": np.nan if stage_rank is None else int(stage_rank),
    })
    return row



def _select_stage1_refinement_seeds(
    stage1_df: pd.DataFrame,
    *,
    objective_col: str = "objective_value",
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    if stage1_df is None or not isinstance(stage1_df, pd.DataFrame) or stage1_df.empty:
        return []
    eligible = stage1_df.copy()
    if "status" in eligible.columns:
        eligible = eligible[eligible["status"].fillna("ok").astype(str).eq("ok")].copy()
    if objective_col in eligible.columns:
        eligible = eligible[pd.to_numeric(eligible[objective_col], errors="coerce").notna()].copy()
    if "candidate_stage" in eligible.columns:
        eligible = eligible[eligible["candidate_stage"].astype(str).eq("stage1")].copy()
    if eligible.empty:
        return []
    eligible = eligible.sort_values([objective_col, "trial"], ascending=[False, True], na_position="last").reset_index(drop=True)
    seeds: List[Dict[str, Any]] = []
    seen: set = set()
    for _, row in eligible.head(max(int(top_n), 0)).iterrows():
        params = extract_best_candidate_payload_from_row(row, base_cfg_payload={}).get("candidate_overrides", {})
        if not params:
            continue
        key = _candidate_params_key(params)
        if key in seen:
            continue
        seen.add(key)
        seeds.append(params)
    return seeds



def extract_best_candidate_payload_from_row(
    best_row: Any,
    *,
    base_cfg_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if best_row is None:
        return {
            "candidate_overrides": {},
            "applied_cfg_payload": dict(_tuning_valid_cfg_payload(base_cfg_payload or {})),
            "candidate_params_json": "{}",
        }

    if isinstance(best_row, pd.Series):
        row = best_row.to_dict()
    elif isinstance(best_row, dict):
        row = dict(best_row)
    else:
        try:
            row = dict(best_row)
        except Exception:
            row = {}

    try:
        from src.investment import MicroPipelineConfig
    except Exception:
        from investment import MicroPipelineConfig

    valid_fields = {f.name for f in MicroPipelineConfig.__dataclass_fields__.values()}
    base_payload = dict(_tuning_valid_cfg_payload(base_cfg_payload or {}))

    overrides: Dict[str, Any] = {}
    payload_text = row.get("candidate_params_json")
    if payload_text not in {None, "", np.nan}:
        try:
            parsed = json.loads(payload_text)
            if isinstance(parsed, dict):
                overrides.update({str(k): v for k, v in parsed.items() if str(k) in valid_fields})
        except Exception:
            pass

    if not overrides:
        for key, value in row.items():
            if str(key) in valid_fields:
                overrides[str(key)] = value

    clean_overrides: Dict[str, Any] = {}
    for key, value in overrides.items():
        if key not in valid_fields:
            continue
        if key in base_payload and _param_value_equal(value, base_payload.get(key)):
            continue
        clean_overrides[key] = value

    applied_cfg = dict(base_payload)
    applied_cfg.update(clean_overrides)

    payload = {
        "candidate_overrides": clean_overrides,
        "applied_cfg_payload": applied_cfg,
        "candidate_params_json": _json_compact(clean_overrides),
        "config_patch": clean_overrides,
        "config_patch_count": int(len(clean_overrides)),
        "config_fingerprint": row.get("config_fingerprint"),
        "objective_name": row.get("objective_name"),
        "objective_value": _safe_float(row.get("objective_value")),
        "candidate_kind": row.get("candidate_kind"),
        "candidate_stage": row.get("candidate_stage"),
        "candidate_param_names": row.get("candidate_param_names"),
        "trial": _safe_float(row.get("trial")),
    }
    return payload



def choose_best_tuning_candidate_payload(
    trials_df: pd.DataFrame,
    *,
    base_cfg_payload: Optional[Dict[str, Any]] = None,
    objective_col: str = "objective_value",
    higher_is_better: bool = True,
    status_col: str = "status",
) -> Dict[str, Any]:
    best_row, table = choose_best_tuning_trial(
        trials_df,
        objective_col=objective_col,
        higher_is_better=higher_is_better,
        status_col=status_col,
    )
    payload = extract_best_candidate_payload_from_row(best_row, base_cfg_payload=base_cfg_payload)
    payload["best_row"] = None if best_row is None else best_row.to_dict()
    payload["selection_table"] = table
    return payload


SIMPLE_AUTO_OPT_SAFE_DIMS: Tuple[str, ...] = (
    "top_k",
    "temperature",
    "weight_shrink",
    "inertia",
    "deadband_threshold",
    "turnover_penalty_strength",
    "turnover_penalty_target",
    "turnover_constraint_max_turnover",
    "target_portfolio_vol_monthly",
    "correlation_penalty_strength",
    "probabilistic_overlay_strength",
)


def _resolve_simple_auto_opt_policy(simple_summary: Optional[Dict[str, Any]] = None, *, policy: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Resolve the semantic local-search policy for Simple auto-opt.

    Accepts either a full ``simple_summary`` bundle (with an optional
    ``tuning_policy`` sub-dict) or a raw policy dict directly.
    """
    summary = dict(simple_summary or {})
    summary_policy = summary.get("tuning_policy", {})
    out: Dict[str, Any] = {}
    if isinstance(summary_policy, dict):
        out.update(dict(summary_policy))
    if isinstance(policy, dict):
        out.update(dict(policy))
    preferred_dims_raw = out.get("preferred_dims", [])
    if isinstance(preferred_dims_raw, str):
        preferred_dims = [preferred_dims_raw]
    else:
        preferred_dims = [str(x) for x in list(preferred_dims_raw or []) if str(x).strip()]
    preferred_dims = [d for d in list(dict.fromkeys(preferred_dims)) if d in SIMPLE_AUTO_OPT_SAFE_DIMS]
    out["preferred_dims"] = preferred_dims
    return out


def _policy_search_budget_to_intensity(search_budget: Any, *, default: str = "Quick") -> str:
    budget = str(search_budget or "").strip().lower()
    if budget == "expanded":
        return "Standard"
    if budget == "standard":
        return "Standard"
    if budget == "light":
        return "Quick"
    return str(default)


def _build_simple_auto_opt_param_space(
    base_cfg_payload: Dict[str, Any],
    *,
    panel_df: Optional[pd.DataFrame] = None,
    search_budget: Any = None,
    include_weight_shrink: bool = False,
) -> Tuple[Dict[str, List[Any]], int, str]:
    """Build a compact, safe local-search grid for Simple auto-opt.

    Returns ``(raw_param_space, max_candidates, intensity_label)``.
    """
    intensity = _policy_search_budget_to_intensity(search_budget, default="Quick")
    intensity_key = str(intensity or "Quick").strip().lower()

    if intensity_key == "standard":
        raw_space = build_local_search_param_space(
            dict(base_cfg_payload or {}),
            panel_df=panel_df,
            temperature_multipliers=(0.8, 1.0, 1.2),
            top_k_offsets=(-2, 0, 2),
            overlay_multipliers=(0.7, 1.0, 1.3),
            target_vol_multipliers=(0.8, 1.0, 1.2),
            turnover_penalty_multipliers=(0.7, 1.0, 1.3),
            weight_shrink_multipliers=((0.7, 1.0, 1.3) if include_weight_shrink else ()),
        )
        return raw_space, 24, "Standard"

    raw_space = build_local_search_param_space(
        dict(base_cfg_payload or {}),
        panel_df=panel_df,
        temperature_multipliers=(0.9, 1.0, 1.1),
        top_k_offsets=(-1, 0, 1),
        overlay_multipliers=(0.85, 1.0, 1.15),
        target_vol_multipliers=(0.9, 1.0, 1.1),
        turnover_penalty_multipliers=(0.85, 1.0, 1.15),
        weight_shrink_multipliers=((0.85, 1.0, 1.15) if include_weight_shrink else ()),
    )
    return raw_space, 12, "Quick"


def _extract_baseline_tuning_row(trials_df: pd.DataFrame) -> Optional[pd.Series]:
    if trials_df is None or not isinstance(trials_df, pd.DataFrame) or trials_df.empty:
        return None
    if "candidate_kind" in trials_df.columns:
        baseline = trials_df[trials_df["candidate_kind"].astype(str).eq("baseline")].copy()
        if not baseline.empty:
            return baseline.iloc[0]
    return trials_df.iloc[0]


def _is_material_tuning_improvement(
    best_row: Optional[pd.Series],
    baseline_row: Optional[pd.Series],
    *,
    objective_col: str = "objective_value",
    higher_is_better: bool = True,
    min_abs_improvement: float = 1e-9,
    min_rel_improvement: float = 1e-4,
) -> bool:
    if best_row is None or baseline_row is None:
        return False
    best_val = _safe_float(best_row.get(objective_col))
    base_val = _safe_float(baseline_row.get(objective_col))
    if not np.isfinite(best_val) or not np.isfinite(base_val):
        return False
    delta = float(best_val - base_val) if bool(higher_is_better) else float(base_val - best_val)
    if not np.isfinite(delta) or delta <= float(min_abs_improvement):
        return False
    if abs(base_val) > 1e-12:
        rel = float(delta / abs(base_val))
        if np.isfinite(rel) and rel >= float(min_rel_improvement):
            return True
    return delta > float(min_abs_improvement)


def run_simple_auto_optimize(
    panel_df,
    *,
    base_cfg_payload: Dict[str, Any],
    simple_summary: Optional[Dict[str, Any]] = None,
    policy: Optional[Dict[str, Any]] = None,
    raw_param_space: Optional[Dict[str, Sequence[Any]]] = None,
    include_weight_shrink: bool = False,
    objective_col: str = "objective_value",
    higher_is_better: bool = True,
    status_col: str = "status",
    min_abs_improvement: float = 1e-9,
    min_rel_improvement: float = 1e-4,
) -> Dict[str, Any]:
    """Run the backend for Phase 5 Simple-mode auto optimization.

    This helper keeps the contract UI-agnostic: it resolves the semantic tuning
    policy, builds a small local neighbourhood, runs the existing local search,
    selects the best valid candidate, and only promotes it when the improvement
    is material and usable.
    """
    base_payload = dict(_tuning_valid_cfg_payload(base_cfg_payload or {}))
    resolved_policy = _resolve_simple_auto_opt_policy(simple_summary, policy=policy)
    objective_name = str(resolved_policy.get("objective") or "sharpe")
    two_stage_search = _coerce_search_stage_enabled(resolved_policy.get("two_stage_search"), default=False)
    search_budget = resolved_policy.get("search_budget")

    result: Dict[str, Any] = {
        "baseline_payload": dict(base_payload),
        "final_cfg_payload": dict(base_payload),
        "best_payload": {},
        "best_row": None,
        "baseline_row": None,
        "trials_df": pd.DataFrame(),
        "selection_table": pd.DataFrame(),
        "policy": dict(resolved_policy),
        "objective_name": objective_name,
        "search_budget": search_budget,
        "search_intensity": _policy_search_budget_to_intensity(search_budget, default="Quick"),
        "two_stage_search": bool(two_stage_search),
        "was_optimized": False,
        "usable_improvement": False,
        "optimization_status": "baseline_only",
        "optimization_reason": "auto_opt_not_run",
    }

    if panel_df is None or not isinstance(panel_df, pd.DataFrame) or panel_df.empty:
        result["optimization_reason"] = "empty_panel_df"
        return result
    if not base_payload:
        result["optimization_reason"] = "empty_base_payload"
        return result

    try:
        if raw_param_space is not None:
            raw_space = {str(k): list(v) for k, v in dict(raw_param_space).items() if v is not None}
            default_budget = 12
            intensity = _policy_search_budget_to_intensity(search_budget, default="Quick")
        else:
            raw_space, default_budget, intensity = _build_simple_auto_opt_param_space(
                base_payload,
                panel_df=panel_df,
                search_budget=search_budget,
                include_weight_shrink=include_weight_shrink,
            )
        result["search_intensity"] = intensity

        effective_space = build_local_search_param_space_from_policy(
            base_payload,
            policy=resolved_policy,
            panel_df=panel_df,
            raw_param_space=raw_space,
            include_weight_shrink=include_weight_shrink,
        )
        result["raw_param_space"] = raw_space
        result["effective_param_space"] = effective_space
        if not effective_space:
            result["optimization_reason"] = "empty_param_space"
            return result

        max_candidates_hint = resolved_policy.get("max_candidates_hint")
        max_candidates = default_budget
        try:
            if max_candidates_hint not in {None, ""}:
                max_candidates = int(max(1, int(max_candidates_hint)))
        except Exception:
            max_candidates = default_budget
        result["max_candidates"] = int(max_candidates)

        trials_df = auto_tune_around_config(
            panel_df,
            base_cfg_payload=base_payload,
            objective=objective_name,
            param_space=effective_space,
            local_search_policy=resolved_policy or None,
            include_weight_shrink=include_weight_shrink,
            max_candidates=int(max_candidates),
            two_stage_search=bool(two_stage_search),
        )
        result["trials_df"] = trials_df if isinstance(trials_df, pd.DataFrame) else pd.DataFrame()
        if not isinstance(trials_df, pd.DataFrame) or trials_df.empty:
            result["optimization_reason"] = "no_trials"
            return result

        best_payload_bundle = choose_best_tuning_candidate_payload(
            trials_df,
            base_cfg_payload=base_payload,
            objective_col=objective_col,
            higher_is_better=higher_is_better,
            status_col=status_col,
        )
        selection_table = pd.DataFrame(best_payload_bundle.get("selection_table", pd.DataFrame())).copy()
        best_row_dict = best_payload_bundle.get("best_row")
        best_row = pd.Series(best_row_dict) if isinstance(best_row_dict, dict) and best_row_dict else None
        baseline_row = _extract_baseline_tuning_row(selection_table if not selection_table.empty else trials_df)
        result["selection_table"] = selection_table
        result["best_payload"] = dict(best_payload_bundle)
        result["best_row"] = best_row
        result["baseline_row"] = baseline_row

        if best_row is None:
            result["optimization_reason"] = "no_valid_best_row"
            return result

        usable_improvement = _is_material_tuning_improvement(
            best_row,
            baseline_row,
            objective_col=objective_col,
            higher_is_better=higher_is_better,
            min_abs_improvement=min_abs_improvement,
            min_rel_improvement=min_rel_improvement,
        )
        result["usable_improvement"] = bool(usable_improvement)

        applied_payload = dict(best_payload_bundle.get("applied_cfg_payload", {}) or {})
        config_patch = dict(best_payload_bundle.get("config_patch", {}) or {})
        if usable_improvement and applied_payload and config_patch:
            result["final_cfg_payload"] = dict(applied_payload)
            result["was_optimized"] = True
            result["optimization_status"] = "optimized"
            result["optimization_reason"] = "material_improvement_found"
        else:
            result["optimization_reason"] = "no_material_improvement" if applied_payload else "invalid_best_payload"

        if baseline_row is not None and best_row is not None:
            base_obj = _safe_float(baseline_row.get(objective_col))
            best_obj = _safe_float(best_row.get(objective_col))
            delta = best_obj - base_obj if np.isfinite(best_obj) and np.isfinite(base_obj) else np.nan
            rel = delta / abs(base_obj) if np.isfinite(delta) and np.isfinite(base_obj) and abs(base_obj) > 1e-12 else np.nan
            result["baseline_objective_value"] = base_obj
            result["best_objective_value"] = best_obj
            result["objective_improvement"] = delta
            result["objective_improvement_pct"] = rel

        return result
    except Exception as exc:
        result["optimization_status"] = "error"
        result["optimization_reason"] = f"auto_opt_error: {exc}"
        result["error"] = str(exc)
        result["final_cfg_payload"] = dict(base_payload)
        result["was_optimized"] = False
        return result



def auto_tune_around_config(
    panel_df,
    *,
    base_cfg_payload: Dict[str, Any],
    objective: str = "sharpe",
    param_space: Optional[Dict[str, Sequence[Any]]] = None,
    local_search_policy: Optional[Dict[str, Any]] = None,
    include_weight_shrink: bool = False,
    temperature_multipliers: Sequence[float] = (0.8, 1.0, 1.2),
    top_k_offsets: Sequence[int] = (-2, 0, 2),
    overlay_multipliers: Sequence[float] = (0.7, 1.0, 1.3),
    target_vol_multipliers: Sequence[float] = (0.8, 1.0, 1.2),
    turnover_penalty_multipliers: Sequence[float] = (0.7, 1.0, 1.3),
    weight_shrink_multipliers: Sequence[float] = (0.7, 1.0, 1.3),
    max_candidates: int = 32,
    two_stage_search: bool = True,
    stage2_top_n: int = 3,
    stage2_max_pairs: int = 12,
    stage2_include_seed_triplets: bool = False,
    refine_from_best_stage1: bool = True,
) -> pd.DataFrame:
    """Run a real local search around the current config.

    Stage 1 performs one-at-a-time local perturbations.
    Stage 2 optionally performs small multi-parameter refinement using the best
    Stage-1 seeds, so the engine can evaluate genuine combinations rather than
    only isolated knobs.
    """
    if panel_df is None or not isinstance(panel_df, pd.DataFrame) or panel_df.empty:
        return pd.DataFrame()

    local_budget = int(max(0, max_candidates))
    if local_budget <= 0:
        return pd.DataFrame()

    base_payload = dict(_tuning_valid_cfg_payload(base_cfg_payload))
    raw_space = build_local_search_param_space_from_policy(
        base_payload,
        policy=local_search_policy,
        panel_df=panel_df,
        raw_param_space=param_space,
        include_weight_shrink=include_weight_shrink,
        temperature_multipliers=temperature_multipliers,
        top_k_offsets=top_k_offsets,
        overlay_multipliers=overlay_multipliers,
        target_vol_multipliers=target_vol_multipliers,
        turnover_penalty_multipliers=turnover_penalty_multipliers,
        weight_shrink_multipliers=weight_shrink_multipliers,
    )

    try:
        from src.investment import MicroPipelineConfig, sanitize_param_space_for_engine
    except Exception:
        from investment import MicroPipelineConfig, sanitize_param_space_for_engine

    cfg = MicroPipelineConfig(**base_payload)
    engine_info = sanitize_param_space_for_engine(raw_space, cfg)
    local_space = dict(engine_info.get("sanitized_param_space", {}) or {})
    if not local_space:
        return pd.DataFrame()

    baseline_val, _, baseline_summary = _run_objective_once(
        panel_df,
        base_cfg_payload=base_payload,
        param_dict={},
        objective=objective,
    )

    rows: List[Dict[str, Any]] = []
    baseline_row = _trial_row_from_result(
        0,
        method="local_search_base",
        params={},
        result_summary=baseline_summary,
        status="ok",
    )
    baseline_row.update({
        "tuned_param": "baseline",
        "tuned_param_count": 0,
        "candidate_kind": "baseline",
        "candidate_stage": "baseline",
        "candidate_value": np.nan,
        "candidate_delta": np.nan,
        "candidate_delta_pct": np.nan,
        "candidate_distance_from_base": 0.0,
        "candidate_params_json": "{}",
        "candidate_param_names": "",
        "candidate_param_count": 0,
        "candidate_delta_map_json": "{}",
        "candidate_distance_map_json": "{}",
        "base_value": np.nan,
        "local_search_budget": local_budget,
        "local_search_dim": int(len(local_space)),
        "local_search_objective": str(objective),
        "local_search_base_objective": _safe_float(baseline_val),
        "local_search_improvement": 0.0,
        "local_search_improvement_pct": 0.0,
        "local_search_stage_rank": 0,
        "local_search_seed_params_json": "{}",
    })
    rows.append(baseline_row)

    candidate_catalog = _build_local_search_candidate_catalog(local_space, base_payload)
    trial_idx = 1
    stage1_rows: List[Dict[str, Any]] = []
    for item in candidate_catalog:
        if trial_idx > local_budget:
            break
        candidate_params = {item["param_name"]: item["candidate_value"]}
        row = _evaluate_local_search_candidate(
            panel_df,
            trial_idx=trial_idx,
            method="local_search_oat",
            stage="stage1",
            candidate_kind="one_at_a_time",
            base_cfg_payload=base_payload,
            objective=objective,
            candidate_params=candidate_params,
            baseline_val=baseline_val,
            local_space=local_space,
            local_search_budget=local_budget,
            stage_rank=len(stage1_rows) + 1,
            tuned_param_names=[item["param_name"]],
        )
        rows.append(row)
        stage1_rows.append(row)
        trial_idx += 1

    stage1_df = pd.DataFrame(stage1_rows)
    stage2_enabled = _coerce_search_stage_enabled(two_stage_search, default=True)
    stage2_budget_remaining = max(0, local_budget - (trial_idx - 1))

    if stage2_enabled and stage2_budget_remaining > 0 and not stage1_df.empty:
        stage1_valid = stage1_df.copy()
        if "status" in stage1_valid.columns:
            stage1_valid = stage1_valid[stage1_valid["status"].fillna("ok").astype(str).eq("ok")].copy()
        if "objective_value" in stage1_valid.columns:
            stage1_valid = stage1_valid[pd.to_numeric(stage1_valid["objective_value"], errors="coerce").notna()].copy()
        if not stage1_valid.empty:
            stage1_valid = stage1_valid.sort_values(["objective_value", "trial"], ascending=[False, True], na_position="last").reset_index(drop=True)
            stage1_valid["stage1_rank"] = np.arange(1, len(stage1_valid) + 1)

            seeds = _select_stage1_refinement_seeds(
                stage1_valid,
                objective_col="objective_value",
                top_n=max(1, int(stage2_top_n)),
            )
            if refine_from_best_stage1 and seeds:
                base_refine_payload = dict(base_payload)
                base_refine_payload.update(seeds[0])
            else:
                base_refine_payload = dict(base_payload)

            stage2_candidates: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
            seen_stage2: set = set()

            for seed_rank, seed_params in enumerate(seeds, start=1):
                seed_names = list(seed_params.keys())
                active_names = [name for name in local_space.keys() if name not in seed_params]
                for extra_name in active_names:
                    for extra_value in _local_search_value_choices(local_space, base_payload, extra_name):
                        candidate_params = dict(seed_params)
                        candidate_params[str(extra_name)] = extra_value
                        key = _candidate_params_key(candidate_params)
                        if key in seen_stage2:
                            continue
                        seen_stage2.add(key)
                        stage2_candidates.append((candidate_params, {
                            "seed_rank": seed_rank,
                            "seed_params": dict(seed_params),
                            "tuned_param_names": list(candidate_params.keys()),
                            "candidate_kind": "multi_param_refine",
                        }))
                        if len(stage2_candidates) >= int(stage2_max_pairs):
                            break
                    if len(stage2_candidates) >= int(stage2_max_pairs):
                        break
                if len(stage2_candidates) >= int(stage2_max_pairs):
                    break

            if stage2_include_seed_triplets and stage2_candidates and len(local_space) >= 3:
                for seed_rank, seed_params in enumerate(seeds, start=1):
                    available_names = [name for name in local_space.keys() if name not in seed_params]
                    if len(available_names) < 2:
                        continue
                    for i_name in range(len(available_names)):
                        if len(stage2_candidates) >= int(stage2_max_pairs) + stage2_budget_remaining:
                            break
                        for j_name in range(i_name + 1, len(available_names)):
                            name_i = available_names[i_name]
                            name_j = available_names[j_name]
                            vals_i = _local_search_value_choices(local_space, base_payload, name_i)
                            vals_j = _local_search_value_choices(local_space, base_payload, name_j)
                            if not vals_i or not vals_j:
                                continue
                            candidate_params = dict(seed_params)
                            candidate_params[name_i] = vals_i[0]
                            candidate_params[name_j] = vals_j[0]
                            key = _candidate_params_key(candidate_params)
                            if key in seen_stage2:
                                continue
                            seen_stage2.add(key)
                            stage2_candidates.append((candidate_params, {
                                "seed_rank": seed_rank,
                                "seed_params": dict(seed_params),
                                "tuned_param_names": list(candidate_params.keys()),
                                "candidate_kind": "multi_param_refine_triplet",
                            }))
                            break
                        if len(stage2_candidates) >= int(stage2_max_pairs) + stage2_budget_remaining:
                            break

            for stage_rank, (candidate_params, meta) in enumerate(stage2_candidates[:stage2_budget_remaining], start=1):
                if trial_idx > local_budget:
                    break
                row = _evaluate_local_search_candidate(
                    panel_df,
                    trial_idx=trial_idx,
                    method="local_search_refine",
                    stage="stage2",
                    candidate_kind=str(meta.get("candidate_kind", "multi_param_refine")),
                    base_cfg_payload=base_refine_payload if refine_from_best_stage1 else base_payload,
                    objective=objective,
                    candidate_params=candidate_params,
                    baseline_val=baseline_val,
                    local_space=local_space,
                    local_search_budget=local_budget,
                    stage_rank=stage_rank,
                    seed_params=meta.get("seed_params", {}),
                    tuned_param_names=meta.get("tuned_param_names"),
                )
                row["local_search_refine_from_best_stage1"] = bool(refine_from_best_stage1)
                row["local_search_refine_seed_rank"] = int(meta.get("seed_rank", stage_rank))
                rows.append(row)
                trial_idx += 1

    out = build_tuning_trials_table(pd.DataFrame(rows), objective_col="objective_value", sort_desc=True)
    if out.empty:
        return out

    best_payload = choose_best_tuning_candidate_payload(
        out,
        base_cfg_payload=base_payload,
        objective_col="objective_value",
        higher_is_better=True,
        status_col="status",
    )
    best_params_json = best_payload.get("candidate_params_json", "{}")
    out["best_candidate_params_json"] = best_params_json
    out["best_candidate_param_count"] = int(best_payload.get("config_patch_count", 0) or 0)
    out["best_candidate_stage"] = best_payload.get("candidate_stage")
    out["best_candidate_kind"] = best_payload.get("candidate_kind")
    out["two_stage_search_enabled"] = bool(stage2_enabled)
    out["two_stage_search_stage2_top_n"] = int(max(1, stage2_top_n))
    out["two_stage_search_stage2_max_pairs"] = int(max(0, stage2_max_pairs))
    out["two_stage_search_stage2_executed"] = bool((out.get("candidate_stage") == "stage2").any()) if "candidate_stage" in out.columns else False
    return out

def build_tuning_trials_table(
    trials_df: pd.DataFrame,
    *,
    objective_col: str = "objective_value",
    sort_desc: bool = True,
) -> pd.DataFrame:
    """Common reporting table for random/Bayesian/Optuna/surrogate tuning outputs."""
    if trials_df is None or not isinstance(trials_df, pd.DataFrame) or trials_df.empty:
        return pd.DataFrame()
    out = trials_df.copy()
    for col in [
        "trial", "objective_value", "sharpe", "cagr", "max_drawdown", "annual_volatility",
        "mean_turnover", "information_ratio", "active_return_annual", "tracking_error_annual",
        "mean_effective_breadth", "mean_diversification_ratio", "mean_effective_risk_bets", "mean_active_assets",
        "mean_effective_n_assets", "inverse_concentration",
        "composite_objective_score", "composite_objective_weight_sum",
        "composite_raw_sharpe", "composite_raw_cagr", "composite_raw_max_drawdown",
        "composite_raw_mean_turnover", "composite_raw_diversification",
        "composite_sharpe_score", "composite_cagr_score", "composite_drawdown_score",
        "composite_turnover_score", "composite_diversification_score",
        "composite_weight_sharpe", "composite_weight_cagr", "composite_weight_max_drawdown",
        "composite_weight_mean_turnover", "composite_weight_diversification",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "status" not in out.columns:
        out["status"] = "ok"
    out["is_valid_trial"] = out["status"].fillna("ok").astype(str).eq("ok") & pd.to_numeric(out.get(objective_col), errors="coerce").notna()
    if objective_col in out.columns:
        out["objective_rank"] = pd.to_numeric(out[objective_col], errors="coerce").rank(method="min", ascending=not bool(sort_desc))
    else:
        out["objective_rank"] = np.nan
    sort_cols = [c for c in ["is_valid_trial", objective_col, "sharpe", "cagr", "trial"] if c in out.columns]
    ascending = []
    for c in sort_cols:
        if c == "is_valid_trial":
            ascending.append(False)
        elif c == objective_col:
            ascending.append(not bool(sort_desc))
        elif c in {"trial"}:
            ascending.append(True)
        else:
            ascending.append(False)
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=ascending, na_position="last").reset_index(drop=True)
    return out



def choose_best_tuning_trial(
    trials_df: pd.DataFrame,
    *,
    objective_col: str = "objective_value",
    higher_is_better: bool = True,
    status_col: str = "status",
) -> Tuple[Optional[pd.Series], pd.DataFrame]:
    """Choose the best valid tuning trial and return (best_row, sorted_table)."""
    table = build_tuning_trials_table(trials_df, objective_col=objective_col, sort_desc=higher_is_better)
    if table.empty:
        return None, table
    eligible = table.copy()
    if status_col in eligible.columns:
        eligible = eligible[eligible[status_col].fillna("ok").astype(str).eq("ok")].copy()
    if objective_col in eligible.columns:
        eligible = eligible[pd.to_numeric(eligible[objective_col], errors="coerce").notna()].copy()
    if eligible.empty:
        return None, table
    eligible = eligible.sort_values(
        [objective_col, "sharpe", "cagr", "trial"],
        ascending=[not bool(higher_is_better), False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    return eligible.iloc[0], table

# ============================================================
# Classifier signal reporting
# ============================================================

def _run_signal_mode(run: Dict[str, Any]) -> str:
    cfg = (run or {}).get("config")
    if cfg is not None:
        try:
            return str(getattr(cfg, "signal_mode", "") or "")
        except Exception:
            pass
    diag = (run or {}).get("diagnostics_df", pd.DataFrame())
    if isinstance(diag, pd.DataFrame) and not diag.empty and "signal_mode_effective" in diag.columns:
        vals = diag["signal_mode_effective"].dropna().astype(str)
        if not vals.empty:
            return str(vals.iloc[-1])
    return "unknown"

def _run_diag_last_or_mean(run: Dict[str, Any], col: str, *, use_last: bool = False) -> float:
    diag = (run or {}).get("diagnostics_df", pd.DataFrame())
    if not isinstance(diag, pd.DataFrame) or diag.empty or col not in diag.columns:
        return np.nan
    vals = pd.to_numeric(diag[col], errors="coerce").dropna()
    if vals.empty:
        return np.nan
    return float(vals.iloc[-1] if use_last else vals.mean())

def build_classifier_signal_summary(run: Dict[str, Any]) -> pd.DataFrame:
    """Compact summary for classifier-style signal layers.

    Works defensively with any subset of diagnostics currently exposed by the engine.
    It is compatible with directional/logistic/top-k modes and does not require all
    columns to be present.
    """
    perf = _run_perf_dict(run)
    signal_mode = _run_signal_mode(run)

    rows = [
        {"metric": "signal_mode", "value": signal_mode},
        {"metric": "cagr", "value": _safe_float(perf.get("cagr"))},
        {"metric": "sharpe", "value": _safe_float(perf.get("sharpe"))},
        {"metric": "max_drawdown", "value": _safe_float(perf.get("max_drawdown"))},
        {"metric": "annual_volatility", "value": _safe_float(perf.get("annual_volatility", perf.get("annualized_volatility")))},
        {"metric": "mean_turnover", "value": _safe_float(perf.get("mean_turnover"))},
        {"metric": "information_ratio", "value": _safe_float(perf.get("information_ratio"))},
        {"metric": "signal_directional_lookback", "value": _run_diag_last_or_mean(run, "signal_directional_lookback", use_last=True)},
        {"metric": "signal_directional_threshold", "value": _run_diag_last_or_mean(run, "signal_directional_threshold", use_last=True)},
        {"metric": "signal_directional_prob_mean", "value": _run_diag_last_or_mean(run, "signal_directional_prob_mean")},
        {"metric": "signal_directional_prob_cross_section_std", "value": _run_diag_last_or_mean(run, "signal_directional_prob_cross_section_std")},
        {"metric": "signal_logistic_lookback", "value": _run_diag_last_or_mean(run, "signal_logistic_lookback", use_last=True)},
        {"metric": "signal_logistic_threshold", "value": _run_diag_last_or_mean(run, "signal_logistic_threshold", use_last=True)},
        {"metric": "signal_logistic_l2", "value": _run_diag_last_or_mean(run, "signal_logistic_l2", use_last=True)},
        {"metric": "signal_logistic_prob_mean", "value": _run_diag_last_or_mean(run, "signal_logistic_prob_mean")},
        {"metric": "signal_logistic_prob_cross_section_std", "value": _run_diag_last_or_mean(run, "signal_logistic_prob_cross_section_std")},
        {"metric": "signal_top_k_lookback", "value": _run_diag_last_or_mean(run, "signal_top_k_lookback", use_last=True)},
        {"metric": "signal_top_k_classifier_k", "value": _run_diag_last_or_mean(run, "signal_top_k_classifier_k", use_last=True)},
        {"metric": "signal_top_k_threshold", "value": _run_diag_last_or_mean(run, "signal_top_k_threshold", use_last=True)},
        {"metric": "signal_top_k_prob_mean", "value": _run_diag_last_or_mean(run, "signal_top_k_prob_mean")},
        {"metric": "signal_top_k_prob_cross_section_std", "value": _run_diag_last_or_mean(run, "signal_top_k_prob_cross_section_std")},
        {"metric": "signal_top_k_positive_share", "value": _run_diag_last_or_mean(run, "signal_top_k_positive_share")},
    ]
    out = pd.DataFrame(rows)
    return out[out["value"].notna() | out["metric"].eq("signal_mode")].reset_index(drop=True)

def build_top_k_classifier_timeseries(run: Dict[str, Any]) -> pd.DataFrame:
    """Per-date diagnostics for classifier-style modes, focused on top-k when present."""
    diag = (run or {}).get("diagnostics_df", pd.DataFrame())
    if not isinstance(diag, pd.DataFrame) or diag.empty:
        return pd.DataFrame()

    cols = [
        "date",
        "signal_mode_effective",
        "signal_directional_lookback",
        "signal_directional_threshold",
        "signal_directional_prob_mean",
        "signal_directional_prob_cross_section_std",
        "signal_logistic_lookback",
        "signal_logistic_threshold",
        "signal_logistic_l2",
        "signal_logistic_prob_mean",
        "signal_logistic_prob_cross_section_std",
        "signal_top_k_lookback",
        "signal_top_k_classifier_k",
        "signal_top_k_threshold",
        "signal_top_k_prob_mean",
        "signal_top_k_prob_cross_section_std",
        "signal_top_k_positive_share",
        "gross_portfolio_return_simple",
        "net_portfolio_return_simple",
        "portfolio_return_simple",
        "cum_return",
    ]
    use_cols = [c for c in cols if c in diag.columns]
    if not use_cols:
        return pd.DataFrame()

    out = diag[use_cols].copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.sort_values("date").reset_index(drop=True)

    gross_col = "gross_portfolio_return_simple" if "gross_portfolio_return_simple" in out.columns else None
    net_col = "net_portfolio_return_simple" if "net_portfolio_return_simple" in out.columns else None
    if net_col is None and "portfolio_return_simple" in out.columns:
        net_col = "portfolio_return_simple"

    if gross_col is not None:
        g = pd.to_numeric(out[gross_col], errors="coerce").fillna(0.0)
        out["cumulative_gross_nav"] = (1.0 + g).cumprod()
    if net_col is not None:
        n = pd.to_numeric(out[net_col], errors="coerce").fillna(0.0)
        out["cumulative_net_nav"] = (1.0 + n).cumprod()
    if "cumulative_gross_nav" in out.columns and "cumulative_net_nav" in out.columns:
        out["classifier_nav_gap"] = out["cumulative_net_nav"] - out["cumulative_gross_nav"]
    return out

def compare_top_k_classifier_runs(
    baseline_run: Dict[str, Any],
    classifier_run: Dict[str, Any],
    *,
    baseline_name: str = "baseline",
    classifier_name: str = "classifier",
) -> Dict[str, Any]:
    """Formal comparison between a baseline run and a classifier-signal run.

    Despite the name, it is robust to top-k/logistic/directional modes and simply
    reports whatever classifier-related diagnostics are available.
    """
    base_summary = build_classifier_signal_summary(baseline_run).rename(columns={"value": baseline_name})
    cls_summary = build_classifier_signal_summary(classifier_run).rename(columns={"value": classifier_name})
    comparison = base_summary.merge(cls_summary, on="metric", how="outer")

    base_num = pd.to_numeric(comparison.get(baseline_name), errors="coerce")
    cls_num = pd.to_numeric(comparison.get(classifier_name), errors="coerce")
    comparison["delta"] = cls_num - base_num
    comparison["delta_pct"] = comparison["delta"] / base_num.replace(0.0, np.nan).abs()

    better_when = {
        "cagr": "higher",
        "sharpe": "higher",
        "information_ratio": "higher",
        "annual_volatility": "lower",
        "mean_turnover": "lower",
        "max_drawdown": "higher",
        "signal_directional_prob_cross_section_std": "higher",
        "signal_logistic_prob_cross_section_std": "higher",
        "signal_top_k_prob_cross_section_std": "higher",
        "signal_top_k_positive_share": "higher",
    }
    comparison["better_when"] = comparison["metric"].map(better_when)

    def _direction(metric: str, delta: float) -> str:
        rule = better_when.get(str(metric))
        if not np.isfinite(delta) or rule is None or abs(float(delta)) <= 1e-12:
            return "neutral"
        if rule == "higher":
            return "improved" if float(delta) > 0 else "worsened"
        if rule == "lower":
            return "improved" if float(delta) < 0 else "worsened"
        if rule == "neutral":
            return "descriptive"
        return "neutral"

    comparison["direction"] = [
        _direction(metric, delta)
        for metric, delta in zip(comparison["metric"], pd.to_numeric(comparison["delta"], errors="coerce"))
    ]

    focus_metrics = [
        "cagr",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "mean_turnover",
        "information_ratio",
        "signal_top_k_prob_mean",
        "signal_top_k_prob_cross_section_std",
        "signal_top_k_positive_share",
        "signal_logistic_prob_mean",
        "signal_logistic_prob_cross_section_std",
        "signal_directional_prob_mean",
    ]
    focus_df = comparison[comparison["metric"].isin(focus_metrics)].copy().reset_index(drop=True)

    delta_lookup = dict(zip(comparison["metric"], pd.to_numeric(comparison["delta"], errors="coerce")))
    delta_sharpe = _safe_float(delta_lookup.get("sharpe"))
    delta_cagr = _safe_float(delta_lookup.get("cagr"))
    delta_turnover = _safe_float(delta_lookup.get("mean_turnover"))
    delta_information_ratio = _safe_float(delta_lookup.get("information_ratio"))

    improved_count = int((focus_df["direction"] == "improved").sum()) if not focus_df.empty else 0
    worsened_count = int((focus_df["direction"] == "worsened").sum()) if not focus_df.empty else 0
    mode_name = _run_signal_mode(classifier_run)

    if np.isfinite(delta_sharpe) and delta_sharpe > 0 and improved_count >= worsened_count:
        headline = f"{mode_name} looks positive: Sharpe improved vs {baseline_name}."
    elif np.isfinite(delta_sharpe) and delta_sharpe < 0 and worsened_count > improved_count:
        headline = f"{mode_name} looks weak: Sharpe worsened vs {baseline_name}."
    else:
        headline = f"{mode_name} changes the signal profile, but the economic result is mixed."

    return {
        "classifier_summary_baseline": base_summary,
        "classifier_summary_classifier_run": cls_summary,
        "classifier_comparison": comparison,
        "classifier_focus": focus_df,
        "classifier_timeseries": build_top_k_classifier_timeseries(classifier_run),
        "decision_summary": {
            "baseline_name": str(baseline_name),
            "classifier_name": str(classifier_name),
            "signal_mode": mode_name,
            "headline": headline,
            "improved_metric_count": improved_count,
            "worsened_metric_count": worsened_count,
            "delta_sharpe": delta_sharpe,
            "delta_cagr": delta_cagr,
            "delta_turnover": delta_turnover,
            "delta_information_ratio": delta_information_ratio,
        },
    }

# ============================================================
# LambdaRank real / multi-loss validation
# ============================================================

def build_lambdarank_multiloss_summary(run: Dict[str, Any]) -> pd.DataFrame:
    """Compact summary for LambdaRank-real and multi-loss research modes.

    The function is defensive: it works even when some diagnostics are absent,
    so it can be used safely across legacy and partially upgraded runs.
    """
    perf = _run_perf_dict(run)
    cfg = run.get("config")
    signal_mode = _run_signal_mode(run)

    def _cfg(name: str, default: Any = np.nan) -> Any:
        if cfg is None:
            return default
        return getattr(cfg, name, default)

    rows = [
        {"metric": "signal_mode", "value": signal_mode},
        {"metric": "cagr", "value": _safe_float(perf.get("cagr"))},
        {"metric": "sharpe", "value": _safe_float(perf.get("sharpe"))},
        {"metric": "max_drawdown", "value": _safe_float(perf.get("max_drawdown"))},
        {"metric": "annual_volatility", "value": _safe_float(perf.get("annual_volatility", perf.get("annualized_volatility")))},
        {"metric": "mean_turnover", "value": _safe_float(perf.get("mean_turnover"))},
        {"metric": "information_ratio", "value": _safe_float(perf.get("information_ratio"))},
        {"metric": "multi_loss_active", "value": bool(_run_diag_last_or_mean(run, "multi_loss_active", use_last=True)) if np.isfinite(_run_diag_last_or_mean(run, "multi_loss_active", use_last=True)) else bool(_cfg("multi_loss_enabled", False))},
        {"metric": "multi_loss_rank_weight", "value": _run_diag_last_or_mean(run, "multi_loss_rank_weight", use_last=True) if np.isfinite(_run_diag_last_or_mean(run, "multi_loss_rank_weight", use_last=True)) else _safe_float(_cfg("multi_loss_rank_weight"))},
        {"metric": "multi_loss_direction_weight", "value": _run_diag_last_or_mean(run, "multi_loss_direction_weight", use_last=True) if np.isfinite(_run_diag_last_or_mean(run, "multi_loss_direction_weight", use_last=True)) else _safe_float(_cfg("multi_loss_direction_weight"))},
        {"metric": "multi_loss_return_weight", "value": _run_diag_last_or_mean(run, "multi_loss_return_weight", use_last=True) if np.isfinite(_run_diag_last_or_mean(run, "multi_loss_return_weight", use_last=True)) else _safe_float(_cfg("multi_loss_return_weight"))},
        {"metric": "multi_loss_logistic_weight", "value": _run_diag_last_or_mean(run, "multi_loss_logistic_weight", use_last=True) if np.isfinite(_run_diag_last_or_mean(run, "multi_loss_logistic_weight", use_last=True)) else _safe_float(_cfg("multi_loss_logistic_weight"))},
        {"metric": "multi_loss_topk_weight", "value": _run_diag_last_or_mean(run, "multi_loss_topk_weight", use_last=True) if np.isfinite(_run_diag_last_or_mean(run, "multi_loss_topk_weight", use_last=True)) else _safe_float(_cfg("multi_loss_topk_weight"))},
        {"metric": "multi_loss_temperature", "value": _run_diag_last_or_mean(run, "multi_loss_temperature", use_last=True) if np.isfinite(_run_diag_last_or_mean(run, "multi_loss_temperature", use_last=True)) else _safe_float(_cfg("multi_loss_temperature"))},
        {"metric": "multi_loss_l2", "value": _run_diag_last_or_mean(run, "multi_loss_l2", use_last=True) if np.isfinite(_run_diag_last_or_mean(run, "multi_loss_l2", use_last=True)) else _safe_float(_cfg("multi_loss_l2"))},
        {"metric": "multi_loss_score_std", "value": _run_diag_last_or_mean(run, "multi_loss_score_std")},
        {"metric": "signal_lambdarank_real_lookback", "value": _run_diag_last_or_mean(run, "signal_lambdarank_real_lookback", use_last=True) if np.isfinite(_run_diag_last_or_mean(run, "signal_lambdarank_real_lookback", use_last=True)) else _safe_float(_cfg("lambdarank_real_lookback"))},
        {"metric": "signal_lambdarank_real_temperature", "value": _run_diag_last_or_mean(run, "signal_lambdarank_real_temperature", use_last=True) if np.isfinite(_run_diag_last_or_mean(run, "signal_lambdarank_real_temperature", use_last=True)) else _safe_float(_cfg("lambdarank_real_temperature"))},
        {"metric": "signal_lambdarank_real_gain_power", "value": _run_diag_last_or_mean(run, "signal_lambdarank_real_gain_power", use_last=True) if np.isfinite(_run_diag_last_or_mean(run, "signal_lambdarank_real_gain_power", use_last=True)) else _safe_float(_cfg("lambdarank_real_gain_power"))},
        {"metric": "signal_lambdarank_real_pair_power", "value": _run_diag_last_or_mean(run, "signal_lambdarank_real_pair_power", use_last=True) if np.isfinite(_run_diag_last_or_mean(run, "signal_lambdarank_real_pair_power", use_last=True)) else _safe_float(_cfg("lambdarank_real_pair_power"))},
        {"metric": "signal_lambdarank_real_l2", "value": _run_diag_last_or_mean(run, "signal_lambdarank_real_l2", use_last=True) if np.isfinite(_run_diag_last_or_mean(run, "signal_lambdarank_real_l2", use_last=True)) else _safe_float(_cfg("lambdarank_real_l2"))},
        {"metric": "signal_score_std", "value": _run_diag_last_or_mean(run, "score_cross_section_std")},
        {"metric": "signal_rank_ic_mean", "value": _safe_float((evaluate_run_rank_ic_quality(run) or {}).get("rank_ic_mean"))},
        {"metric": "signal_rank_ic_ir", "value": _safe_float((evaluate_run_rank_ic_quality(run) or {}).get("rank_ic_ir"))},
    ]
    return pd.DataFrame(rows)


def build_lambdarank_multiloss_timeseries(run: Dict[str, Any]) -> pd.DataFrame:
    """Per-date diagnostics for LambdaRank real / multi-loss runs."""
    diag = run.get("diagnostics_df", pd.DataFrame())
    if not isinstance(diag, pd.DataFrame) or diag.empty:
        return pd.DataFrame()

    preferred_cols = [
        "date",
        "signal_mode",
        "score_cross_section_std",
        "multi_loss_active",
        "multi_loss_rank_weight",
        "multi_loss_direction_weight",
        "multi_loss_return_weight",
        "multi_loss_logistic_weight",
        "multi_loss_topk_weight",
        "multi_loss_temperature",
        "multi_loss_l2",
        "multi_loss_score_std",
        "signal_lambdarank_real_lookback",
        "signal_lambdarank_real_temperature",
        "signal_lambdarank_real_gain_power",
        "signal_lambdarank_real_pair_power",
        "signal_lambdarank_real_l2",
        "gross_portfolio_return_simple",
        "net_portfolio_return_simple",
        "portfolio_return_simple",
        "mean_selected_prob_up",
        "mean_selected_score",
        "realised_return",
    ]
    cols = [c for c in preferred_cols if c in diag.columns]
    if not cols:
        return pd.DataFrame()

    out = diag[cols].copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.sort_values("date").reset_index(drop=True)

    gross_col = "gross_portfolio_return_simple" if "gross_portfolio_return_simple" in out.columns else None
    net_col = "net_portfolio_return_simple" if "net_portfolio_return_simple" in out.columns else None
    if net_col is None and "portfolio_return_simple" in out.columns:
        net_col = "portfolio_return_simple"

    if gross_col is not None:
        g = pd.to_numeric(out[gross_col], errors="coerce").fillna(0.0)
        out["cumulative_gross_nav"] = (1.0 + g).cumprod()
    if net_col is not None:
        n = pd.to_numeric(out[net_col], errors="coerce").fillna(0.0)
        out["cumulative_net_nav"] = (1.0 + n).cumprod()
    if "cumulative_gross_nav" in out.columns and "cumulative_net_nav" in out.columns:
        out["lambdarank_nav_gap"] = out["cumulative_net_nav"] - out["cumulative_gross_nav"]
    return out


def compare_lambdarank_multiloss_runs(
    baseline_run: Dict[str, Any],
    ranked_run: Dict[str, Any],
    *,
    baseline_name: str = "baseline",
    ranked_name: str = "lambdarank_multiloss",
) -> Dict[str, Any]:
    """Formal comparison between a baseline run and a LambdaRank/multi-loss run."""
    base_summary = build_lambdarank_multiloss_summary(baseline_run).rename(columns={"value": baseline_name})
    ranked_summary = build_lambdarank_multiloss_summary(ranked_run).rename(columns={"value": ranked_name})
    comparison = base_summary.merge(ranked_summary, on="metric", how="outer")

    base_num = pd.to_numeric(comparison.get(baseline_name), errors="coerce")
    ranked_num = pd.to_numeric(comparison.get(ranked_name), errors="coerce")
    comparison["delta"] = ranked_num - base_num
    comparison["delta_pct"] = comparison["delta"] / base_num.replace(0.0, np.nan).abs()

    better_when = {
        "cagr": "higher",
        "sharpe": "higher",
        "information_ratio": "higher",
        "annual_volatility": "lower",
        "mean_turnover": "lower",
        "max_drawdown": "higher",
        "signal_rank_ic_mean": "higher",
        "signal_rank_ic_ir": "higher",
        "multi_loss_score_std": "higher",
        "signal_score_std": "higher",
    }
    comparison["better_when"] = comparison["metric"].map(better_when)

    def _direction(metric: str, delta: float) -> str:
        rule = better_when.get(str(metric))
        if not np.isfinite(delta) or rule is None or abs(float(delta)) <= 1e-12:
            return "neutral"
        if rule == "higher":
            return "improved" if float(delta) > 0 else "worsened"
        if rule == "lower":
            return "improved" if float(delta) < 0 else "worsened"
        if rule == "neutral":
            return "descriptive"
        return "neutral"

    comparison["direction"] = [
        _direction(metric, delta)
        for metric, delta in zip(comparison["metric"], pd.to_numeric(comparison["delta"], errors="coerce"))
    ]

    focus_metrics = [
        "cagr",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "mean_turnover",
        "information_ratio",
        "signal_rank_ic_mean",
        "signal_rank_ic_ir",
        "multi_loss_active",
        "multi_loss_rank_weight",
        "multi_loss_direction_weight",
        "multi_loss_return_weight",
        "multi_loss_logistic_weight",
        "multi_loss_topk_weight",
        "multi_loss_temperature",
        "multi_loss_l2",
        "multi_loss_score_std",
        "signal_lambdarank_real_lookback",
        "signal_lambdarank_real_temperature",
        "signal_lambdarank_real_gain_power",
        "signal_lambdarank_real_pair_power",
        "signal_lambdarank_real_l2",
    ]
    focus_df = comparison[comparison["metric"].isin(focus_metrics)].copy().reset_index(drop=True)

    delta_lookup = dict(zip(comparison["metric"], pd.to_numeric(comparison["delta"], errors="coerce")))
    delta_sharpe = _safe_float(delta_lookup.get("sharpe"))
    delta_cagr = _safe_float(delta_lookup.get("cagr"))
    delta_turnover = _safe_float(delta_lookup.get("mean_turnover"))
    delta_information_ratio = _safe_float(delta_lookup.get("information_ratio"))
    delta_rank_ic = _safe_float(delta_lookup.get("signal_rank_ic_mean"))

    improved_count = int((focus_df["direction"] == "improved").sum()) if not focus_df.empty else 0
    worsened_count = int((focus_df["direction"] == "worsened").sum()) if not focus_df.empty else 0
    mode_name = _run_signal_mode(ranked_run)

    if np.isfinite(delta_sharpe) and delta_sharpe > 0 and improved_count >= worsened_count:
        headline = f"{mode_name} looks positive: Sharpe improved vs {baseline_name}."
    elif np.isfinite(delta_rank_ic) and delta_rank_ic > 0 and improved_count >= worsened_count:
        headline = f"{mode_name} improved the ranking profile vs {baseline_name}, with mixed economic impact."
    elif np.isfinite(delta_sharpe) and delta_sharpe < 0 and worsened_count > improved_count:
        headline = f"{mode_name} looks weak: Sharpe worsened vs {baseline_name}."
    else:
        headline = f"{mode_name} changes the ranking profile, but the overall result is mixed."

    return {
        "lambdarank_summary_baseline": base_summary,
        "lambdarank_summary_ranked_run": ranked_summary,
        "lambdarank_comparison": comparison,
        "lambdarank_focus": focus_df,
        "lambdarank_timeseries": build_lambdarank_multiloss_timeseries(ranked_run),
        "decision_summary": {
            "baseline_name": str(baseline_name),
            "ranked_name": str(ranked_name),
            "signal_mode": mode_name,
            "headline": headline,
            "improved_metric_count": improved_count,
            "worsened_metric_count": worsened_count,
            "delta_sharpe": delta_sharpe,
            "delta_cagr": delta_cagr,
            "delta_turnover": delta_turnover,
            "delta_information_ratio": delta_information_ratio,
            "delta_rank_ic": delta_rank_ic,
        },
    }


# ============================================================
# Weighted adaptive hypervolume helpers
# ============================================================

def _coerce_objective_specs_for_hv(
    objective_cols: Sequence[str],
    objective_specs: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, Dict[str, str]]:
    specs_in = dict(objective_specs or {})
    out: Dict[str, Dict[str, str]] = {}
    for col in [str(x) for x in list(objective_cols or []) if str(x).strip()]:
        spec = dict(specs_in.get(col, {}))
        sense = str(spec.get("sense", "min" if col in {"max_drawdown", "mean_turnover"} else "max")).strip().lower()
        if sense not in {"min", "max"}:
            sense = "max"
        out[col] = {"sense": sense}
    return out


def _adaptive_weight_profile_for_preset(
    preset: str,
    objective_cols: Sequence[str],
) -> Dict[str, float]:
    preset_norm = str(preset or "balanced").strip().lower()
    base = {str(col): 1.0 for col in objective_cols}
    if preset_norm in {"growth", "aggressive"}:
        base.update({
            "sharpe": 1.25,
            "cagr": 1.45,
            "max_drawdown": 0.80,
            "mean_turnover": 0.60,
            "diversification": 0.80,
            "stability": 0.85,
        })
    elif preset_norm in {"defensive", "capital_preservation", "conservative"}:
        base.update({
            "sharpe": 1.05,
            "cagr": 0.85,
            "max_drawdown": 1.55,
            "mean_turnover": 1.20,
            "diversification": 1.05,
            "stability": 1.20,
        })
    else:  # balanced
        base.update({
            "sharpe": 1.15,
            "cagr": 1.05,
            "max_drawdown": 1.15,
            "mean_turnover": 0.85,
            "diversification": 0.95,
            "stability": 0.95,
        })

    selected = {str(col): float(base.get(str(col), 1.0)) for col in objective_cols}
    vals = np.array(list(selected.values()), dtype="float64")
    vals[~np.isfinite(vals)] = 1.0
    vals = np.clip(vals, 1e-12, None)
    vals = vals * (max(len(vals), 1) / float(vals.sum()))
    return {col: float(v) for col, v in zip(selected.keys(), vals)}


def normalize_objective_matrix(
    df: pd.DataFrame,
    *,
    objective_cols: Sequence[str],
    objective_specs: Optional[Dict[str, Dict[str, str]]] = None,
    clip: Tuple[float, float] = (0.0, 1.0),
    prefix: str = "norm_",
) -> pd.DataFrame:
    """Normalize a matrix of objectives to 0-1, converting all objectives to maximization.

    For objectives with sense='min', values are inverted so that higher normalized values are always better.
    Constant columns are mapped to 1.0 for finite rows (all points tie on that objective).
    """
    work = pd.DataFrame(df).copy()
    if work.empty:
        return work
    obj_cols = [str(x) for x in list(objective_cols or []) if str(x).strip() and str(x) in work.columns]
    if not obj_cols:
        return work
    specs = _coerce_objective_specs_for_hv(obj_cols, objective_specs=objective_specs)
    lo_clip, hi_clip = float(clip[0]), float(clip[1])
    if hi_clip < lo_clip:
        lo_clip, hi_clip = hi_clip, lo_clip

    for col in obj_cols:
        xs = pd.to_numeric(work[col], errors="coerce")
        finite = xs[np.isfinite(xs)]
        out = pd.Series(np.nan, index=work.index, dtype="float64")
        if finite.empty:
            work[f"{prefix}{col}"] = out
            continue
        x_min = float(finite.min())
        x_max = float(finite.max())
        if abs(x_max - x_min) <= 1e-12:
            out.loc[np.isfinite(xs)] = 1.0
        else:
            if str(specs.get(col, {}).get("sense", "max")).lower() == "min":
                out = (x_max - xs) / (x_max - x_min)
            else:
                out = (xs - x_min) / (x_max - x_min)
        work[f"{prefix}{col}"] = out.clip(lo_clip, hi_clip)
    return work


def build_reference_point(
    objective_df: pd.DataFrame,
    *,
    objective_cols: Sequence[str],
    margin: float = 0.05,
) -> Dict[str, float]:
    """Build a reference point slightly worse than the worst observed point.

    Assumes the objective matrix is already normalized so that larger is better.
    """
    work = pd.DataFrame(objective_df)
    obj_cols = [str(x) for x in list(objective_cols or []) if str(x).strip() and str(x) in work.columns]
    ref: Dict[str, float] = {}
    margin = max(float(margin), 0.0)
    for col in obj_cols:
        xs = pd.to_numeric(work[col], errors="coerce")
        finite = xs[np.isfinite(xs)]
        if finite.empty:
            ref[col] = 0.0
            continue
        worst = float(finite.min())
        best = float(finite.max())
        span = max(best - worst, 1e-6)
        ref[col] = float(worst - margin * span)
    return ref


def _prepare_hv_points(
    objective_df: pd.DataFrame,
    *,
    objective_cols: Sequence[str],
    reference_point: Optional[Dict[str, float]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, List[str], Dict[str, float], Dict[str, float]]:
    work = pd.DataFrame(objective_df).copy()
    obj_cols = [str(x) for x in list(objective_cols or []) if str(x).strip() and str(x) in work.columns]
    if not obj_cols:
        return np.zeros((0, 0), dtype="float64"), [], {}, {}
    ref = dict(reference_point or build_reference_point(work, objective_cols=obj_cols))
    w_raw = {str(col): _safe_float((weights or {}).get(str(col), 1.0)) for col in obj_cols}
    w_vals = np.array([w_raw.get(col, 1.0) for col in obj_cols], dtype="float64")
    w_vals[~np.isfinite(w_vals)] = 1.0
    w_vals = np.clip(w_vals, 1e-12, None)
    w_vals = w_vals * (max(len(w_vals), 1) / float(w_vals.sum()))
    w = {col: float(v) for col, v in zip(obj_cols, w_vals)}

    rows: List[List[float]] = []
    for _, row in work.iterrows():
        vals: List[float] = []
        valid = True
        for col in obj_cols:
            v = _safe_float(row.get(col))
            r = _safe_float(ref.get(col))
            if not np.isfinite(v) or not np.isfinite(r):
                valid = False
                break
            vals.append(max((v - r) * w[col], 0.0))
        if valid and any(v > 0.0 for v in vals):
            rows.append(vals)
    if not rows:
        return np.zeros((0, len(obj_cols)), dtype="float64"), obj_cols, ref, w
    pts = np.asarray(rows, dtype="float64")
    return pts, obj_cols, ref, w


def _hypervolume_union_from_origin(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype="float64")
    if pts.size == 0:
        return 0.0
    if pts.ndim == 1:
        pts = pts.reshape(-1, 1)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    pts = pts[np.all(pts >= 0.0, axis=1)]
    if pts.size == 0:
        return 0.0
    d = int(pts.shape[1])
    if d == 1:
        return float(np.max(pts[:, 0])) if pts.shape[0] else 0.0

    z = np.sort(np.unique(pts[:, -1]))
    z = z[z > 0.0]
    if z.size == 0:
        return 0.0
    hv = 0.0
    prev = 0.0
    for level in z:
        if level <= prev:
            continue
        mask = pts[:, -1] >= level
        slice_pts = pts[mask][:, :-1]
        width = float(level - prev)
        if width > 0.0 and slice_pts.size > 0:
            hv += width * _hypervolume_union_from_origin(slice_pts)
        prev = float(level)
    return float(hv)


def compute_hypervolume(
    objective_df: pd.DataFrame,
    *,
    objective_cols: Sequence[str],
    reference_point: Optional[Dict[str, float]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Compute weighted hypervolume on a normalized maximization objective matrix."""
    pts, _, _, _ = _prepare_hv_points(
        objective_df,
        objective_cols=objective_cols,
        reference_point=reference_point,
        weights=weights,
    )
    return _hypervolume_union_from_origin(pts)


def compute_hypervolume_contributions(
    objective_df: pd.DataFrame,
    *,
    objective_cols: Sequence[str],
    reference_point: Optional[Dict[str, float]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """Compute leave-one-out hypervolume contribution for each row."""
    work = pd.DataFrame(objective_df).copy()
    if work.empty:
        return pd.Series(dtype="float64")
    base_hv = compute_hypervolume(
        work,
        objective_cols=objective_cols,
        reference_point=reference_point,
        weights=weights,
    )
    contrib: List[float] = []
    for idx in work.index:
        reduced = work.drop(index=idx)
        hv_without = compute_hypervolume(
            reduced,
            objective_cols=objective_cols,
            reference_point=reference_point,
            weights=weights,
        )
        contrib.append(max(float(base_hv - hv_without), 0.0))
    return pd.Series(contrib, index=work.index, dtype="float64")


def choose_solution_by_weighted_hypervolume(
    frontier_df: pd.DataFrame,
    *,
    objective_cols: Optional[Sequence[str]] = None,
    objective_specs: Optional[Dict[str, Dict[str, str]]] = None,
    preset: str = "balanced",
    reference_point: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Choose a frontier solution using weighted hypervolume contribution.

    Returns the chosen row plus annotated frontier information:
    - frontier_hypervolume
    - hv_contribution
    - adaptive_weight_profile
    """
    df = pd.DataFrame(frontier_df).copy()
    if df.empty:
        return {
            "selected_row": {},
            "frontier_df": pd.DataFrame(),
            "frontier_hypervolume": 0.0,
            "reference_point": {},
            "adaptive_weight_profile": {},
        }

    if "is_pareto_efficient" in df.columns and df["is_pareto_efficient"].astype(bool).any():
        work = df.loc[df["is_pareto_efficient"].astype(bool)].copy()
    elif "pareto_rank" in df.columns and pd.to_numeric(df["pareto_rank"], errors="coerce").notna().any():
        min_rank = pd.to_numeric(df["pareto_rank"], errors="coerce").min()
        work = df.loc[pd.to_numeric(df["pareto_rank"], errors="coerce") == min_rank].copy()
    else:
        work = df.copy()

    obj_cols = [str(x) for x in list(objective_cols or []) if str(x).strip() and str(x) in work.columns]
    if not obj_cols:
        candidate_defaults = ["sharpe", "cagr", "max_drawdown", "mean_turnover", "diversification", "stability"]
        obj_cols = [col for col in candidate_defaults if col in work.columns]
    if not obj_cols:
        first_row = dict(work.iloc[0].to_dict())
        return {
            "selected_row": first_row,
            "frontier_df": work,
            "frontier_hypervolume": np.nan,
            "reference_point": {},
            "adaptive_weight_profile": {},
        }

    specs = _coerce_objective_specs_for_hv(obj_cols, objective_specs=objective_specs)
    adaptive_weight_profile = _adaptive_weight_profile_for_preset(preset, obj_cols)
    norm = normalize_objective_matrix(work, objective_cols=obj_cols, objective_specs=specs, prefix="hv_norm_")
    norm_cols = [f"hv_norm_{col}" for col in obj_cols]
    ref = dict(reference_point or build_reference_point(norm, objective_cols=norm_cols))
    hv_total = compute_hypervolume(norm, objective_cols=norm_cols, reference_point=ref, weights={f"hv_norm_{k}": v for k, v in adaptive_weight_profile.items()})
    hv_contrib = compute_hypervolume_contributions(norm, objective_cols=norm_cols, reference_point=ref, weights={f"hv_norm_{k}": v for k, v in adaptive_weight_profile.items()})

    out = work.copy()
    out["frontier_hypervolume"] = float(hv_total)
    out["hv_contribution"] = pd.to_numeric(hv_contrib, errors="coerce")
    out["adaptive_weight_profile"] = json.dumps(adaptive_weight_profile, sort_keys=True)
    out["hypervolume_reference_point"] = json.dumps(ref, sort_keys=True)

    pick_idx = out["hv_contribution"].astype(float).idxmax()
    selected = dict(out.loc[pick_idx].to_dict())
    selected["frontier_hypervolume"] = float(hv_total)
    selected["adaptive_weight_profile"] = dict(adaptive_weight_profile)
    selected["reference_point"] = dict(ref)
    selected["objective_cols_used"] = list(obj_cols)
    return {
        "selected_row": selected,
        "frontier_df": out,
        "frontier_hypervolume": float(hv_total),
        "reference_point": dict(ref),
        "adaptive_weight_profile": dict(adaptive_weight_profile),
    }



# ============================================================
# Fixed composite score helpers (Phase 4 canonical selector)
# ============================================================


def get_fixed_composite_weight_profile(profile: str = "composite_balanced") -> Dict[str, float]:
    """Return canonical fixed composite weights for frontier/population selection.

    These profiles are intentionally simple and explicit so the product can
    explain a stable composite objective without depending on knee-point or
    hypervolume selection logic. We keep the first version close to the
    requested canonical formula and reuse the existing composite naming scheme.
    """
    name = _resolve_multi_objective_preset(profile)
    profiles: Dict[str, Dict[str, float]] = {
        "composite_balanced": {
            "sharpe": 0.35,
            "cagr": 0.25,
            "max_drawdown": 0.20,
            "mean_turnover": 0.10,
            "diversification": 0.10,
        },
        "composite_growth": {
            "sharpe": 0.24,
            "cagr": 0.40,
            "max_drawdown": 0.16,
            "mean_turnover": 0.08,
            "diversification": 0.12,
        },
        "composite_defensive": {
            "sharpe": 0.24,
            "cagr": 0.12,
            "max_drawdown": 0.34,
            "mean_turnover": 0.14,
            "diversification": 0.16,
        },
        "composite_quality": {
            "sharpe": 0.42,
            "cagr": 0.18,
            "max_drawdown": 0.20,
            "mean_turnover": 0.08,
            "diversification": 0.12,
        },
        "composite_low_turnover": {
            "sharpe": 0.24,
            "cagr": 0.16,
            "max_drawdown": 0.16,
            "mean_turnover": 0.30,
            "diversification": 0.14,
        },
        "composite_diversified": {
            "sharpe": 0.22,
            "cagr": 0.16,
            "max_drawdown": 0.18,
            "mean_turnover": 0.08,
            "diversification": 0.36,
        },
        "composite_robust": {
            "sharpe": 0.28,
            "cagr": 0.16,
            "max_drawdown": 0.24,
            "mean_turnover": 0.12,
            "diversification": 0.10,
            "stability": 0.10,
        },
    }
    out = dict(profiles.get(name, profiles["composite_balanced"]))
    total = float(sum(v for v in out.values() if np.isfinite(v)))
    if total > 0:
        out = {k: float(v) / total for k, v in out.items()}
    return out


def get_composite_objective_weight_profile(profile: str = "composite_balanced") -> Dict[str, float]:
    """Backward/semantic alias for the canonical fixed composite weight helper.

    Keeps compatibility with earlier naming proposals without changing the
    existing fixed-composite implementation contract.
    """
    return get_fixed_composite_weight_profile(profile)


def _normalize_active_weight_mapping(
    weights: Dict[str, float],
    *,
    floor: float = 1e-12,
) -> Dict[str, float]:
    """Normalize active weights by absolute magnitude.

    Manual profiles may arrive with arbitrary signs or scales from UI inputs.
    We keep only active finite entries and renormalize them so the active
    magnitudes sum to 1.0. Objective direction is still controlled separately
    by objective_specs / metric sense, so negative manual inputs do not invert
    the intended meaning of cost metrics unexpectedly.
    """
    active: Dict[str, float] = {}
    for key, value in (weights or {}).items():
        name = str(key).strip()
        if not name:
            continue
        v = _safe_float(value)
        if not np.isfinite(v):
            continue
        mag = abs(float(v))
        if mag <= float(floor):
            continue
        active[name] = mag
    total = float(sum(active.values()))
    if total <= float(floor):
        return {}
    return {k: float(v) / total for k, v in active.items()}



def _coerce_fixed_composite_weight_profile(
    profile: Any,
    *,
    fallback_profile: str = "composite_balanced",
    objective_specs: Optional[Dict[str, Dict[str, str]]] = None,
) -> Tuple[Dict[str, float], str]:
    """Coerce a fixed composite profile into a stable active-weight mapping.

    Accepts canonical profile names or arbitrary dicts coming from UI/manual
    inputs. Arbitrary dicts are sanitized entry-by-entry, invalid or zero-like
    weights are dropped, and only active components are renormalized. If the
    resulting mapping is empty, we fall back cleanly to the canonical profile.
    """
    if isinstance(profile, dict):
        cleaned = _normalize_active_weight_mapping(dict(profile))
        if cleaned:
            return cleaned, "custom"
        return get_fixed_composite_weight_profile(fallback_profile), str(fallback_profile)

    resolved_name = _resolve_fixed_composite_profile_name(profile)
    if isinstance(resolved_name, dict):
        return _coerce_fixed_composite_weight_profile(
            resolved_name,
            fallback_profile=fallback_profile,
            objective_specs=objective_specs,
        )
    return get_fixed_composite_weight_profile(str(resolved_name)), str(resolved_name)


def _fixed_composite_objective_specs(
    metric_names: Sequence[str],
    objective_specs: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, Dict[str, str]]:
    base_specs = {
        "sharpe": {"sense": "max"},
        "cagr": {"sense": "max"},
        "diversification": {"sense": "max"},
        "stability": {"sense": "max"},
        "max_drawdown": {"sense": "min"},
        "mean_turnover": {"sense": "min"},
    }
    specs = {str(k): dict(v) for k, v in (objective_specs or {}).items()}
    for name in metric_names:
        specs.setdefault(str(name), dict(base_specs.get(str(name), {"sense": "max"})))
    return specs



def normalize_fixed_composite_objective_matrix(
    df: pd.DataFrame,
    *,
    weight_profile: str | Dict[str, float] = "composite_balanced",
    objective_specs: Optional[Dict[str, Dict[str, str]]] = None,
    clip: Tuple[float, float] = (0.0, 1.0),
    prefix: str = "fixed_comp_norm_",
    constant_fill_value: float = 0.5,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Dict[str, str]]]:
    """Normalize the metrics needed by a fixed composite profile.

    Returns the annotated DataFrame, the effective weight profile, and the
    resolved objective specs. Unlike the run-level composite score helper, this
    operates on a frontier/population DataFrame so selection is explicit and
    reproducible across rows.
    """
    work = pd.DataFrame(df).copy()
    weights, _ = _coerce_fixed_composite_weight_profile(
        weight_profile,
        fallback_profile="composite_balanced",
        objective_specs=objective_specs,
    )
    metric_names = [str(k) for k, v in weights.items() if np.isfinite(_safe_float(v)) and _safe_float(v) > 0]
    if not metric_names or work.empty:
        return work, weights, _fixed_composite_objective_specs(metric_names, objective_specs=objective_specs)

    specs = _fixed_composite_objective_specs(metric_names, objective_specs=objective_specs)
    work = normalize_objective_matrix(
        work,
        objective_cols=metric_names,
        objective_specs=specs,
        clip=clip,
        prefix=prefix,
    )

    norm_cols = [f"{prefix}{name}" for name in metric_names]
    for col in norm_cols:
        if col not in work.columns:
            work[col] = np.nan
        xs = pd.to_numeric(work[col], errors="coerce")
        finite = xs[np.isfinite(xs)]
        if finite.empty:
            work[col] = xs
            continue
        x_min = float(finite.min())
        x_max = float(finite.max())
        if abs(x_max - x_min) <= 1e-12:
            xs.loc[np.isfinite(xs)] = float(constant_fill_value)
        work[col] = xs.clip(float(clip[0]), float(clip[1]))
    return work, weights, specs


def compute_fixed_composite_score(
    df: pd.DataFrame,
    *,
    weight_profile: str | Dict[str, float] = "composite_balanced",
    objective_specs: Optional[Dict[str, Dict[str, str]]] = None,
    score_col: str = "fixed_composite_score",
    prefix: str = "fixed_comp_norm_",
) -> pd.DataFrame:
    """Add a canonical fixed composite score to a frontier/population DataFrame."""
    work, weights, _ = normalize_fixed_composite_objective_matrix(
        df,
        weight_profile=weight_profile,
        objective_specs=objective_specs,
        prefix=prefix,
    )
    if work.empty:
        work[score_col] = pd.Series(dtype="float64")
        work["fixed_composite_profile"] = str(weight_profile if not isinstance(weight_profile, dict) else "custom")
        work["fixed_composite_active_components"] = ""
        work["fixed_composite_missing_components"] = ""
        return work

    weights, profile_name = _coerce_fixed_composite_weight_profile(
        weight_profile,
        fallback_profile="composite_balanced",
        objective_specs=objective_specs,
    )
    metric_names = [str(k) for k, v in weights.items() if np.isfinite(_safe_float(v)) and _safe_float(v) > 0]
    score = pd.Series(np.nan, index=work.index, dtype="float64")
    active_components: List[str] = []
    missing_components: List[str] = []
    numer = pd.Series(0.0, index=work.index, dtype="float64")
    denom = pd.Series(0.0, index=work.index, dtype="float64")

    for name in metric_names:
        col = f"{prefix}{name}"
        if col not in work.columns:
            missing_components.append(str(name))
            continue
        xs = pd.to_numeric(work[col], errors="coerce")
        w = float(_safe_float(weights.get(name)))
        valid = np.isfinite(xs)
        if bool(valid.any()):
            numer.loc[valid] += w * xs.loc[valid].astype(float)
            denom.loc[valid] += w
            active_components.append(str(name))
        else:
            missing_components.append(str(name))
    valid_rows = denom > 0
    score.loc[valid_rows] = numer.loc[valid_rows] / denom.loc[valid_rows]
    work[score_col] = pd.to_numeric(score, errors="coerce")
    work["fixed_composite_profile"] = profile_name
    work["fixed_composite_active_components"] = "|".join(active_components)
    work["fixed_composite_missing_components"] = "|".join(missing_components)
    for name, w in weights.items():
        work[f"fixed_composite_weight_{name}"] = float(_safe_float(w))
    return work


def choose_solution_by_fixed_composite_score(
    frontier_df: pd.DataFrame,
    *,
    profile: str | Dict[str, float] = "composite_balanced",
    objective_specs: Optional[Dict[str, Dict[str, str]]] = None,
    score_col: str = "fixed_composite_score",
) -> Dict[str, Any]:
    """Choose the best row by explicit fixed composite score.

    This is intentionally simpler than knee-point / hypervolume selection and is
    designed to provide a stable, explainable contract for Simple/Auto mode.
    """
    df = pd.DataFrame(frontier_df).copy()
    if df.empty:
        return {
            "selected_row": {},
            "frontier_df": pd.DataFrame(),
            "score_col": score_col,
            "weight_profile": {},
            "weight_profile_name": None,
        }

    if "is_pareto_efficient" in df.columns and df["is_pareto_efficient"].astype(bool).any():
        work = df.loc[df["is_pareto_efficient"].astype(bool)].copy()
    elif "pareto_rank" in df.columns and pd.to_numeric(df["pareto_rank"], errors="coerce").notna().any():
        min_rank = pd.to_numeric(df["pareto_rank"], errors="coerce").min()
        work = df.loc[pd.to_numeric(df["pareto_rank"], errors="coerce") == min_rank].copy()
    else:
        work = df.copy()

    weights, weight_profile_name = _coerce_fixed_composite_weight_profile(
        profile,
        fallback_profile="composite_balanced",
        objective_specs=objective_specs,
    )
    scored = compute_fixed_composite_score(
        work,
        weight_profile=weights,
        objective_specs=objective_specs,
        score_col=score_col,
    )
    xs = pd.to_numeric(scored.get(score_col), errors="coerce")
    if not xs.notna().any():
        first_row = dict(scored.iloc[0].to_dict())
        return {
            "selected_row": first_row,
            "frontier_df": scored,
            "score_col": score_col,
            "weight_profile": dict(weights),
            "weight_profile_name": weight_profile_name,
        }

    pick_idx = xs.astype(float).idxmax()
    selected = dict(scored.loc[pick_idx].to_dict())
    selected["weight_profile"] = dict(weights)
    selected["weight_profile_name"] = weight_profile_name
    selected["score_col"] = str(score_col)
    return {
        "selected_row": selected,
        "frontier_df": scored,
        "score_col": str(score_col),
        "weight_profile": dict(weights),
        "weight_profile_name": weight_profile_name,
    }


def _resolve_fixed_composite_profile_name(profile: Any) -> str | Dict[str, float]:
    if isinstance(profile, dict):
        resolved: Dict[str, float] = {}
        for k, v in dict(profile).items():
            name = str(k).strip()
            if not name:
                continue
            resolved[name] = _safe_float(v)
        return resolved
    raw = str(profile or "balanced").strip().lower()
    alias_map = {
        "balanced": "composite_balanced",
        "growth": "composite_growth",
        "defensive": "composite_defensive",
        "quality": "composite_quality",
        "low_turnover": "composite_low_turnover",
        "diversified": "composite_diversified",
        "robust": "composite_robust",
        "fixed": "composite_balanced",
        "default": "composite_balanced",
    }
    return alias_map.get(raw, raw or "composite_balanced")


def choose_multiobjective_solution_by_policy(
    frontier_df: pd.DataFrame,
    *,
    policy: str = "balanced_compromise",
    objective_cols: Optional[Sequence[str]] = None,
    objective_specs: Optional[Dict[str, Dict[str, str]]] = None,
    composite_profile: str | Dict[str, float] = "balanced",
    hypervolume_preset: str = "balanced",
    score_col: str = "fixed_composite_score",
) -> Dict[str, Any]:
    """Choose one row from a frontier/population DataFrame using a named policy.

    Supported policies:
    - balanced_compromise
    - knee_point
    - weighted_hypervolume
    - fixed_composite_score

    Convenience aliases:
    - composite_balanced
    - composite_growth
    - composite_defensive
    """
    df = pd.DataFrame(frontier_df).copy()
    policy_raw = str(policy or "balanced_compromise").strip().lower()
    if df.empty:
        return {
            "selected_row": {},
            "frontier_df": df,
            "selection_policy": policy_raw,
            "composite_profile": _resolve_fixed_composite_profile_name(composite_profile) if policy_raw == "fixed_composite_score" else None,
        }

    alias_to_profile = {
        "composite_balanced": "composite_balanced",
        "composite_growth": "composite_growth",
        "composite_defensive": "composite_defensive",
    }
    if policy_raw in alias_to_profile:
        resolved_profile = alias_to_profile[policy_raw]
        fixed = choose_solution_by_fixed_composite_score(
            df,
            profile=resolved_profile,
            objective_specs=objective_specs,
            score_col=score_col,
        )
        fixed["selection_policy"] = "fixed_composite_score"
        fixed["selection_policy_requested"] = policy_raw
        fixed["composite_profile"] = resolved_profile
        return fixed

    if policy_raw == "fixed_composite_score":
        resolved_profile = _resolve_fixed_composite_profile_name(composite_profile)
        fixed = choose_solution_by_fixed_composite_score(
            df,
            profile=resolved_profile,
            objective_specs=objective_specs,
            score_col=score_col,
        )
        fixed["selection_policy"] = "fixed_composite_score"
        fixed["selection_policy_requested"] = policy_raw
        fixed["composite_profile"] = resolved_profile
        return fixed

    if policy_raw == "weighted_hypervolume":
        cols = list(objective_cols or [c for c in ["sharpe", "cagr", "max_drawdown", "mean_turnover", "diversification", "stability"] if c in df.columns])
        picked = choose_solution_by_weighted_hypervolume(
            df,
            objective_cols=cols,
            objective_specs=objective_specs,
            preset=hypervolume_preset,
        )
        picked["selection_policy"] = "weighted_hypervolume"
        picked["selection_policy_requested"] = policy_raw
        picked["composite_profile"] = None
        frontier_scored = pd.DataFrame(picked.get("frontier_df", df)).copy()
        if frontier_scored.empty:
            frontier_scored = df.copy()
        xs = pd.to_numeric(frontier_scored.get("hv_contribution"), errors="coerce")
        if xs.notna().any():
            pick_idx = xs.astype(float).idxmax()
            selected = dict(frontier_scored.loc[pick_idx].to_dict())
        else:
            selected = dict((picked.get("selected_row") or {}))
            if not selected and not frontier_scored.empty:
                selected = dict(frontier_scored.iloc[0].to_dict())
        picked["selected_row"] = selected
        return picked

    cols = list(objective_cols or [c for c in ["sharpe", "cagr", "max_drawdown", "mean_turnover", "diversification", "stability"] if c in df.columns])
    if cols:
        work = normalize_objective_matrix(df, objective_cols=cols, objective_specs=objective_specs, prefix="pick_norm_")
        norm_cols = [f"pick_norm_{c}" for c in cols if f"pick_norm_{c}" in work.columns]
    else:
        work = df.copy()
        norm_cols = []

    crowd = pd.to_numeric(work.get("crowding_distance"), errors="coerce")
    finite_crowd = crowd.replace([np.inf, -np.inf], np.nan)
    fill_val = float(finite_crowd.max()) if finite_crowd.notna().any() else 0.0
    crowd = crowd.replace(np.inf, fill_val).replace(-np.inf, 0.0).fillna(0.0)
    work["__crowding_rank"] = crowd

    if norm_cols:
        work["balanced_compromise_score"] = pd.to_numeric(work[norm_cols], errors="coerce").mean(axis=1)
        dist_to_ideal = np.sqrt(np.square(pd.to_numeric(work[norm_cols], errors="coerce").fillna(0.0) - 1.0).sum(axis=1))
        work["knee_point_score"] = (1.0 - dist_to_ideal) + 0.10 * crowd
    else:
        fallback_numeric = [c for c in ["sharpe", "cagr", "diversification", "stability"] if c in work.columns]
        work["balanced_compromise_score"] = pd.to_numeric(work[fallback_numeric], errors="coerce").mean(axis=1) if fallback_numeric else 0.0
        work["knee_point_score"] = pd.to_numeric(work["balanced_compromise_score"], errors="coerce").fillna(0.0) + 0.10 * crowd

    if policy_raw == "knee_point":
        local_score_col = "knee_point_score"
    else:
        local_score_col = "balanced_compromise_score"
        policy_raw = "balanced_compromise"

    xs = pd.to_numeric(work.get(local_score_col), errors="coerce")
    if xs.notna().any():
        pick_idx = xs.astype(float).idxmax()
        selected = dict(work.loc[pick_idx].to_dict())
    else:
        selected = dict(work.iloc[0].to_dict()) if not work.empty else {}

    selected["score_col"] = local_score_col
    return {
        "selected_row": selected,
        "frontier_df": work,
        "score_col": local_score_col,
        "selection_policy": policy_raw,
        "selection_policy_requested": str(policy or "balanced_compromise"),
        "composite_profile": None,
    }


# ============================================================
# Recursive multi-objective search (Phase 4E)
# ============================================================


def _recursive_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except Exception:
        return json.dumps(str(value))



def _recursive_unique_preserve(seq: Sequence[Any]) -> List[Any]:
    out: List[Any] = []
    seen = set()
    for item in list(seq or []):
        key = _recursive_json_dumps(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out



def _recursive_candidate_params_key(payload: Dict[str, Any]) -> str:
    return _recursive_json_dumps(dict(payload or {}))



def _recursive_merge_payloads(base_cfg_payload: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base_cfg_payload or {})
    merged.update(dict(patch or {}))
    try:
        from src.investment import coerce_candidate_payload_for_config
    except Exception:
        try:
            from investment import coerce_candidate_payload_for_config
        except Exception:
            coerce_candidate_payload_for_config = None
    if callable(coerce_candidate_payload_for_config):
        try:
            merged = dict(coerce_candidate_payload_for_config(merged) or merged)
        except Exception:
            merged = dict(merged)
    return merged



def _recursive_param_space_sanitized(param_space: Dict[str, Any], base_cfg_payload: Dict[str, Any]) -> Dict[str, List[Any]]:
    raw = dict(param_space or {})
    try:
        from src.investment import MicroPipelineConfig, sanitize_param_space_for_engine
    except Exception:
        try:
            from investment import MicroPipelineConfig, sanitize_param_space_for_engine
        except Exception:
            MicroPipelineConfig = None
            sanitize_param_space_for_engine = None
    if MicroPipelineConfig is not None and callable(sanitize_param_space_for_engine):
        try:
            cfg = MicroPipelineConfig(**dict(_tuning_valid_cfg_payload(base_cfg_payload or {})))
            engine_info = sanitize_param_space_for_engine(raw, cfg)
            sanitized = dict(engine_info.get("sanitized_param_space", {}) or {})
            if sanitized:
                return {str(k): _recursive_unique_preserve(list(v or [])) for k, v in sanitized.items()}
        except Exception:
            pass
    out: Dict[str, List[Any]] = {}
    for key, values in raw.items():
        if isinstance(values, (list, tuple, set, np.ndarray, pd.Series)):
            vals = _recursive_unique_preserve(list(values))
        else:
            vals = [values]
        if vals:
            out[str(key)] = vals
    return out



def _extract_candidate_patch_from_row(row: Any, initial_param_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, pd.Series):
        data = row.to_dict()
    elif isinstance(row, dict):
        data = dict(row)
    else:
        return {}

    for key in ["candidate_params_json", "candidate_delta_map_json", "config_patch_json", "param_patch_json"]:
        raw = data.get(key)
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and parsed:
                    return dict(parsed)
            except Exception:
                pass
        elif isinstance(raw, dict) and raw:
            return dict(raw)

    out: Dict[str, Any] = {}
    for key in list((initial_param_space or {}).keys()):
        if key in data:
            out[str(key)] = data.get(key)
    return out



def _dominates_max(x: Sequence[float], y: Sequence[float]) -> bool:
    xa = np.asarray(x, dtype="float64")
    ya = np.asarray(y, dtype="float64")
    if xa.shape != ya.shape:
        return False
    if not np.all(np.isfinite(xa)) or not np.all(np.isfinite(ya)):
        return False
    return bool(np.all(xa >= ya) and np.any(xa > ya))



def _fast_non_dominated_sort(objective_matrix: np.ndarray) -> List[List[int]]:
    pts = np.asarray(objective_matrix, dtype="float64")
    n = int(pts.shape[0]) if pts.ndim == 2 else 0
    if n <= 0:
        return []
    dominates = [set() for _ in range(n)]
    dominated_count = np.zeros(n, dtype=int)
    fronts: List[List[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates_max(pts[p], pts[q]):
                dominates[p].add(q)
            elif _dominates_max(pts[q], pts[p]):
                dominated_count[p] += 1
        if dominated_count[p] == 0:
            fronts[0].append(p)
    i = 0
    while i < len(fronts) and fronts[i]:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in dominates[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        i += 1
    return fronts



def _compute_crowding_distance(objective_matrix: np.ndarray, front_indices: Sequence[int]) -> Dict[int, float]:
    pts = np.asarray(objective_matrix, dtype="float64")
    front = [int(i) for i in list(front_indices or []) if int(i) >= 0]
    if not front:
        return {}
    if len(front) <= 2:
        return {idx: float("inf") for idx in front}
    m = int(pts.shape[1]) if pts.ndim == 2 else 0
    distances = {idx: 0.0 for idx in front}
    front_pts = pts[front, :]
    for j in range(m):
        vals = front_pts[:, j]
        order = np.argsort(vals)
        sorted_front = [front[k] for k in order]
        distances[sorted_front[0]] = float("inf")
        distances[sorted_front[-1]] = float("inf")
        vmin = float(vals[order[0]])
        vmax = float(vals[order[-1]])
        span = vmax - vmin
        if span <= 1e-12:
            continue
        for pos in range(1, len(sorted_front) - 1):
            prev_val = float(vals[order[pos - 1]])
            next_val = float(vals[order[pos + 1]])
            idx = sorted_front[pos]
            if np.isfinite(distances[idx]):
                distances[idx] += (next_val - prev_val) / span
    return distances



def _annotate_multiobjective_population(
    trials_df: pd.DataFrame,
    *,
    objective_names: Sequence[str],
    objective_specs: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, Any]:
    work = pd.DataFrame(trials_df).copy()
    if work.empty:
        return {
            "population_df": pd.DataFrame(),
            "frontier_df": pd.DataFrame(),
            "objective_cols_used": [],
            "objective_specs": {},
            "frontier_hypervolume": np.nan,
        }
    obj_cols = [str(x) for x in list(objective_names or []) if str(x).strip() and str(x) in work.columns]
    if not obj_cols:
        fallback = ["sharpe", "cagr", "max_drawdown", "mean_turnover", "diversification", "stability"]
        obj_cols = [c for c in fallback if c in work.columns]
    if not obj_cols:
        out = work.copy()
        out["pareto_rank"] = np.nan
        out["crowding_distance"] = np.nan
        out["is_pareto_efficient"] = False
        return {
            "population_df": out,
            "frontier_df": pd.DataFrame(),
            "objective_cols_used": [],
            "objective_specs": {},
            "frontier_hypervolume": np.nan,
        }

    specs = _coerce_objective_specs_for_hv(obj_cols, objective_specs=objective_specs)
    norm = normalize_objective_matrix(work, objective_cols=obj_cols, objective_specs=specs, prefix="mo_norm_")
    norm_cols = [f"mo_norm_{c}" for c in obj_cols]
    valid_mask = norm[norm_cols].notna().all(axis=1)
    valid_index = list(norm.index[valid_mask])
    pareto_rank = pd.Series(np.nan, index=norm.index, dtype="float64")
    crowding = pd.Series(np.nan, index=norm.index, dtype="float64")
    is_efficient = pd.Series(False, index=norm.index, dtype="bool")
    if valid_index:
        matrix = norm.loc[valid_index, norm_cols].to_numpy(dtype="float64")
        fronts = _fast_non_dominated_sort(matrix)
        for rank, front in enumerate(fronts):
            front_global = [valid_index[i] for i in front]
            for idx in front_global:
                pareto_rank.loc[idx] = float(rank)
            crowd_map = _compute_crowding_distance(matrix, front)
            for local_idx, dist in crowd_map.items():
                crowding.loc[valid_index[int(local_idx)]] = float(dist)
            if rank == 0:
                for idx in front_global:
                    is_efficient.loc[idx] = True
    out = norm.copy()
    out["pareto_rank"] = pareto_rank
    out["crowding_distance"] = crowding
    out["is_pareto_efficient"] = is_efficient
    frontier = out.loc[out["is_pareto_efficient"].astype(bool)].copy()
    frontier_hv = np.nan
    if not frontier.empty:
        hv_info = choose_solution_by_weighted_hypervolume(
            frontier,
            objective_cols=obj_cols,
            objective_specs=specs,
            preset="balanced",
        )
        frontier = pd.DataFrame(hv_info.get("frontier_df", frontier)).copy()
        frontier_hv = _safe_float(hv_info.get("frontier_hypervolume"))
        if "hv_contribution" in frontier.columns:
            contrib = pd.to_numeric(frontier["hv_contribution"], errors="coerce").fillna(0.0)
            total = float(contrib.sum())
            frontier["hv_contribution_share"] = contrib / total if total > 1e-12 else 0.0
        frontier_idx = list(frontier.index)
        if frontier_idx:
            aligned = frontier[[c for c in frontier.columns if c in out.columns or c in {"hv_contribution", "hv_contribution_share", "frontier_hypervolume", "adaptive_weight_profile", "hypervolume_reference_point"}]].copy()
            for col in aligned.columns:
                out.loc[frontier_idx, col] = aligned[col]
    return {
        "population_df": out,
        "frontier_df": frontier,
        "objective_cols_used": list(obj_cols),
        "objective_specs": specs,
        "frontier_hypervolume": frontier_hv,
    }



def _sample_candidate_from_param_space(
    param_space: Dict[str, Sequence[Any]],
    *,
    rng: np.random.Generator,
    seed_payloads: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    space = {str(k): list(v or []) for k, v in dict(param_space or {}).items() if list(v or [])}
    if not space:
        return {}
    seed_pool = [dict(x) for x in list(seed_payloads or []) if isinstance(x, dict) and x]
    out: Dict[str, Any] = {}
    use_seed = bool(seed_pool) and float(rng.random()) < 0.50
    seed = dict(seed_pool[int(rng.integers(0, len(seed_pool)))]) if use_seed else {}
    for key, values in space.items():
        if key in seed and seed.get(key) in values and float(rng.random()) < 0.65:
            out[key] = seed.get(key)
        else:
            out[key] = values[int(rng.integers(0, len(values)))]
    return out



def _initialize_population(
    param_space: Dict[str, Sequence[Any]],
    *,
    population_size: int,
    rng: np.random.Generator,
    seed_payloads: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    n = max(1, int(population_size))
    pop: List[Dict[str, Any]] = []
    seen = set()
    for seed in list(seed_payloads or []):
        if len(pop) >= n:
            break
        cand = {str(k): v for k, v in dict(seed or {}).items() if k in param_space}
        key = _recursive_candidate_params_key(cand)
        if key not in seen:
            seen.add(key)
            pop.append(cand)
    while len(pop) < n:
        cand = _sample_candidate_from_param_space(param_space, rng=rng, seed_payloads=seed_payloads)
        key = _recursive_candidate_params_key(cand)
        if key in seen and len(seen) < max(4 * n, n + 1):
            continue
        seen.add(key)
        pop.append(cand)
    return pop



def _mutate_candidate(
    candidate: Dict[str, Any],
    param_space: Dict[str, Sequence[Any]],
    *,
    mutation_rate: float,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    out = dict(candidate or {})
    rate = float(np.clip(_safe_float(mutation_rate), 0.0, 1.0)) if np.isfinite(_safe_float(mutation_rate)) else 0.15
    for key, values in dict(param_space or {}).items():
        vals = list(values or [])
        if not vals:
            continue
        if float(rng.random()) < rate or key not in out:
            if len(vals) == 1:
                out[key] = vals[0]
                continue
            current = out.get(key)
            alts = [v for v in vals if _recursive_json_dumps(v) != _recursive_json_dumps(current)]
            pool = alts if alts else vals
            out[key] = pool[int(rng.integers(0, len(pool)))]
    return out



def _crossover_candidates(
    left: Dict[str, Any],
    right: Dict[str, Any],
    param_space: Dict[str, Sequence[Any]],
    *,
    crossover_rate: float,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    rate = float(np.clip(_safe_float(crossover_rate), 0.0, 1.0)) if np.isfinite(_safe_float(crossover_rate)) else 0.9
    if float(rng.random()) > rate:
        return dict(left or {})
    out: Dict[str, Any] = {}
    for key in dict(param_space or {}).keys():
        lv = dict(left or {}).get(key)
        rv = dict(right or {}).get(key)
        if lv is None and rv is None:
            continue
        if lv is None:
            out[key] = rv
        elif rv is None:
            out[key] = lv
        else:
            out[key] = lv if float(rng.random()) < 0.5 else rv
    return out



def _select_nsga2_survivors(
    candidate_df: pd.DataFrame,
    *,
    objective_names: Sequence[str],
    objective_specs: Optional[Dict[str, Dict[str, str]]] = None,
    survivor_count: int,
) -> pd.DataFrame:
    annotated = _annotate_multiobjective_population(candidate_df, objective_names=objective_names, objective_specs=objective_specs)
    pop = pd.DataFrame(annotated.get("population_df", pd.DataFrame())).copy()
    if pop.empty:
        return pop
    pop["pareto_rank"] = pd.to_numeric(pop.get("pareto_rank"), errors="coerce")
    pop["crowding_distance"] = pd.to_numeric(pop.get("crowding_distance"), errors="coerce")
    pop = pop.sort_values(
        ["pareto_rank", "crowding_distance", "trial"],
        ascending=[True, False, True],
        na_position="last",
    ).reset_index(drop=True)
    keep = max(1, int(survivor_count))
    return pop.head(keep).copy()



def _tournament_pick(population_df: pd.DataFrame, rng: np.random.Generator) -> Dict[str, Any]:
    if population_df is None or population_df.empty:
        return {}
    n = int(population_df.shape[0])
    k = 2 if n >= 2 else 1
    idx = rng.choice(np.arange(n), size=k, replace=False)
    subset = population_df.iloc[list(idx)].copy()
    subset["pareto_rank"] = pd.to_numeric(subset.get("pareto_rank"), errors="coerce")
    subset["crowding_distance"] = pd.to_numeric(subset.get("crowding_distance"), errors="coerce")
    subset = subset.sort_values(["pareto_rank", "crowding_distance"], ascending=[True, False], na_position="last")
    row = subset.iloc[0]
    return _extract_candidate_patch_from_row(row)



def run_nsga2_tuning(
    panel_df,
    *,
    base_cfg_payload: dict,
    param_space: dict,
    objective_names: Sequence[str],
    population_size: int = 24,
    generations: int = 8,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.15,
    seed: int = 42,
    objective_specs: Optional[Dict[str, Dict[str, str]]] = None,
    seed_payloads: Optional[Sequence[Dict[str, Any]]] = None,
) -> dict:
    """Minimal functional NSGA-II tuner over discrete param-space candidates."""
    if panel_df is None or not isinstance(panel_df, pd.DataFrame) or panel_df.empty:
        return {"population_df": pd.DataFrame(), "frontier_df": pd.DataFrame(), "history_df": pd.DataFrame(), "best_compromise_payload": {}}
    space = _recursive_param_space_sanitized(param_space, base_cfg_payload=base_cfg_payload)
    if not space:
        return {"population_df": pd.DataFrame(), "frontier_df": pd.DataFrame(), "history_df": pd.DataFrame(), "best_compromise_payload": {}}

    rng = np.random.default_rng(int(seed))
    pop_size = max(4, int(population_size))
    n_gen = max(1, int(generations))
    obj_cols = [str(x) for x in list(objective_names or []) if str(x).strip()]
    specs = _coerce_objective_specs_for_hv(obj_cols, objective_specs=objective_specs)

    eval_cache: Dict[str, Dict[str, Any]] = {}
    trial_counter = 0
    history_rows: List[Dict[str, Any]] = []

    def evaluate_candidate(candidate_patch: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal trial_counter
        key = _recursive_candidate_params_key(candidate_patch)
        if key in eval_cache:
            return dict(eval_cache[key])
        trial_counter += 1
        merged = _recursive_merge_payloads(base_cfg_payload, candidate_patch)
        try:
            _, _, summary = _run_objective_once(
                panel_df,
                base_cfg_payload=merged,
                param_dict={},
                objective=obj_cols[0] if obj_cols else "sharpe",
            )
            row = _trial_row_from_result(
                trial_counter,
                method="nsga2",
                params=dict(candidate_patch),
                result_summary=summary,
                status="ok",
            )
        except Exception as exc:
            row = {
                "trial": int(trial_counter),
                "method": "nsga2",
                "status": "error",
                "error": str(exc),
            }
            for k, v in dict(candidate_patch).items():
                row[k] = v
            for col in obj_cols:
                row[col] = np.nan
        row["candidate_params_json"] = _recursive_json_dumps(dict(candidate_patch))
        row["full_candidate_payload_json"] = _recursive_json_dumps(merged)
        row["candidate_param_count"] = int(len(candidate_patch))
        eval_cache[key] = dict(row)
        return dict(row)

    population = _initialize_population(space, population_size=pop_size, rng=rng, seed_payloads=seed_payloads)
    population_df = pd.DataFrame([evaluate_candidate(c) for c in population])
    population_df = _select_nsga2_survivors(population_df, objective_names=obj_cols, objective_specs=specs, survivor_count=pop_size)

    for gen in range(1, n_gen + 1):
        annotated = _annotate_multiobjective_population(population_df, objective_names=obj_cols, objective_specs=specs)
        current_pop = pd.DataFrame(annotated.get("population_df", population_df)).copy()
        frontier_df = pd.DataFrame(annotated.get("frontier_df", pd.DataFrame())).copy()
        hv_total = _safe_float(annotated.get("frontier_hypervolume"))
        history_rows.append({
            "generation": int(gen),
            "population_size": int(current_pop.shape[0]),
            "frontier_size": int(frontier_df.shape[0]),
            "frontier_hypervolume": hv_total,
            "trial_count_cumulative": int(len(eval_cache)),
        })
        if gen >= n_gen:
            population_df = current_pop
            break
        parents = current_pop.copy()
        offspring_patches: List[Dict[str, Any]] = []
        seen = set()
        while len(offspring_patches) < pop_size:
            left = _tournament_pick(parents, rng)
            right = _tournament_pick(parents, rng)
            child = _crossover_candidates(left, right, space, crossover_rate=crossover_rate, rng=rng)
            child = _mutate_candidate(child, space, mutation_rate=mutation_rate, rng=rng)
            key = _recursive_candidate_params_key(child)
            if key in seen and len(seen) < 4 * pop_size:
                continue
            seen.add(key)
            offspring_patches.append(child)
        offspring_df = pd.DataFrame([evaluate_candidate(c) for c in offspring_patches])
        combined = pd.concat([parents, offspring_df], ignore_index=True, sort=False)
        population_df = _select_nsga2_survivors(combined, objective_names=obj_cols, objective_specs=specs, survivor_count=pop_size)

    final_annotated = _annotate_multiobjective_population(population_df, objective_names=obj_cols, objective_specs=specs)
    final_population = pd.DataFrame(final_annotated.get("population_df", population_df)).copy()
    final_frontier = pd.DataFrame(final_annotated.get("frontier_df", pd.DataFrame())).copy()
    history_df = pd.DataFrame(history_rows)

    best_compromise_payload: Dict[str, Any] = {}
    if not final_frontier.empty:
        frontier_norm = normalize_objective_matrix(final_frontier, objective_cols=obj_cols, objective_specs=specs, prefix="pick_norm_")
        norm_cols = [f"pick_norm_{c}" for c in obj_cols]
        frontier_norm["balanced_compromise_score"] = pd.to_numeric(frontier_norm[norm_cols], errors="coerce").mean(axis=1)
        pick_idx = frontier_norm["balanced_compromise_score"].astype(float).idxmax()
        best_patch = _extract_candidate_patch_from_row(final_frontier.loc[pick_idx], initial_param_space=space)
        best_compromise_payload = _recursive_merge_payloads(base_cfg_payload, best_patch)

    return {
        "population_df": final_population,
        "frontier_df": final_frontier,
        "history_df": history_df,
        "best_compromise_payload": best_compromise_payload,
    }




def _first_n_primes(n: int) -> List[int]:
    out: List[int] = []
    candidate = 2
    while len(out) < max(0, int(n)):
        is_prime = True
        for p in out:
            if p * p > candidate:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            out.append(candidate)
        candidate += 1
    return out



def _halton_value(index: int, base: int) -> float:
    i = max(1, int(index))
    b = max(2, int(base))
    f = 1.0
    r = 0.0
    while i > 0:
        f /= float(b)
        r += f * float(i % b)
        i //= b
    return float(r)



def _sobol_unit_vector(dim: int, sample_index: int, *, seed: int = 42) -> np.ndarray:
    d = max(1, int(dim))
    n = max(1, int(sample_index))
    try:
        from scipy.stats import qmc
        engine = qmc.Sobol(d=d, scramble=True, seed=int(seed))
        engine.fast_forward(n - 1)
        vec = np.asarray(engine.random(1)[0], dtype="float64")
        if vec.shape[0] == d:
            return np.clip(vec, 0.0, 1.0)
    except Exception:
        pass
    primes = _first_n_primes(d + 3)
    return np.asarray([_halton_value(n + j, primes[j]) for j in range(d)], dtype="float64")



def _lhs_unit_vector(dim: int, sample_index: int, *, seed: int = 42) -> np.ndarray:
    d = max(1, int(dim))
    n = max(1, int(sample_index))
    try:
        from scipy.stats import qmc
        engine = qmc.LatinHypercube(d=d, seed=int(seed) + n)
        vec = np.asarray(engine.random(1)[0], dtype="float64")
        if vec.shape[0] == d:
            return np.clip(vec, 0.0, 1.0)
    except Exception:
        pass
    primes = _first_n_primes(d + 3)
    return np.asarray([_halton_value(n + j, primes[j]) for j in range(d)], dtype="float64")



def _schema_values_from_spec(spec: Any) -> List[Any]:
    if isinstance(spec, dict):
        for key in ["values", "choices", "grid", "candidates", "options"]:
            vals = spec.get(key)
            if isinstance(vals, (list, tuple, set, np.ndarray, pd.Series)):
                return _recursive_unique_preserve(list(vals))
        if all(k in spec for k in ["min", "max"]):
            lo = spec.get("min")
            hi = spec.get("max")
            step = spec.get("step")
            kind = str(spec.get("type", spec.get("dtype", "float")) or "float").strip().lower()
            try:
                if kind in {"int", "integer"}:
                    lo_i = int(round(float(lo)))
                    hi_i = int(round(float(hi)))
                    if step is None:
                        step_i = 1
                    else:
                        step_i = max(1, int(round(float(step))))
                    return list(range(lo_i, hi_i + 1, step_i))
                if step is not None and float(step) > 0:
                    arr = np.arange(float(lo), float(hi) + 0.5 * float(step), float(step), dtype="float64")
                    return [float(x) for x in arr.tolist()]
                return [float(lo), float(hi)]
            except Exception:
                return []
        return []
    if isinstance(spec, tuple) and len(spec) == 2:
        lo, hi = spec
        if isinstance(lo, int) and isinstance(hi, int):
            if hi < lo:
                lo, hi = hi, lo
            return list(range(int(lo), int(hi) + 1))
        try:
            return [float(lo), float(hi)]
        except Exception:
            return []
    if isinstance(spec, (list, tuple, set, np.ndarray, pd.Series)):
        return _recursive_unique_preserve(list(spec))
    return []



def _schema_choice_from_unit(spec: Any, u: float, *, rng: np.random.Generator) -> Any:
    uu = float(np.clip(_safe_float(u) if np.isfinite(_safe_float(u)) else 0.5, 0.0, 1.0))
    if isinstance(spec, dict):
        vals = _schema_values_from_spec(spec)
        if vals:
            idx = min(len(vals) - 1, int(np.floor(uu * len(vals))))
            return vals[idx]
        if all(k in spec for k in ["min", "max"]):
            lo = spec.get("min")
            hi = spec.get("max")
            kind = str(spec.get("type", spec.get("dtype", "float")) or "float").strip().lower()
            log_scale = bool(spec.get("log", False))
            try:
                lo_f = float(lo)
                hi_f = float(hi)
                if hi_f < lo_f:
                    lo_f, hi_f = hi_f, lo_f
                if log_scale and lo_f > 0 and hi_f > 0:
                    x = float(np.exp(np.log(lo_f) + uu * (np.log(hi_f) - np.log(lo_f))))
                else:
                    x = float(lo_f + uu * (hi_f - lo_f))
                if kind in {"int", "integer"}:
                    step = spec.get("step")
                    if step is not None:
                        step_i = max(1, int(round(float(step))))
                        q = int(round((x - lo_f) / step_i))
                        return int(min(max(int(lo_f), int(lo_f) + q * step_i), int(hi_f)))
                    return int(round(x))
                step = spec.get("step")
                if step is not None:
                    step_f = float(step)
                    if step_f > 0:
                        q = round((x - lo_f) / step_f)
                        x = float(lo_f + q * step_f)
                        x = min(max(x, lo_f), hi_f)
                return float(x)
            except Exception:
                pass
    vals = _schema_values_from_spec(spec)
    if vals:
        idx = min(len(vals) - 1, int(np.floor(uu * len(vals))))
        return vals[idx]
    return None



def _schema_is_categorical(spec: Any) -> bool:
    if isinstance(spec, dict):
        typ = str(spec.get("type", spec.get("dtype", "")) or "").strip().lower()
        if typ in {"category", "categorical", "enum", "str", "string", "bool", "boolean"}:
            return True
        vals = _schema_values_from_spec(spec)
        if vals:
            return not all(isinstance(v, (int, float, np.integer, np.floating)) or _safe_float(v) == _safe_float(v) for v in vals)
        return False
    vals = _schema_values_from_spec(spec)
    if not vals:
        return False
    return not all(np.isfinite(_safe_float(v)) for v in vals)



def _condition_is_satisfied(payload: Dict[str, Any], condition: Any) -> bool:
    if condition in {None, '', {}}:
        return True
    if isinstance(condition, str):
        key = str(condition).strip()
        val = payload.get(key)
        if val is None:
            return False
        if isinstance(val, str):
            return str(val).strip().lower() not in {'', '0', 'false', 'none', 'off'}
        try:
            return bool(val)
        except Exception:
            return False
    if isinstance(condition, dict):
        for key, expected in condition.items():
            actual = payload.get(str(key))
            if isinstance(expected, (list, tuple, set)):
                if actual not in set(expected):
                    return False
            elif isinstance(expected, dict):
                if 'not' in expected and actual == expected.get('not'):
                    return False
                if 'in' in expected and actual not in set(expected.get('in') or []):
                    return False
                if 'neq' in expected and actual == expected.get('neq'):
                    return False
                if 'eq' in expected and actual != expected.get('eq'):
                    return False
                if 'min' in expected:
                    aval = _safe_float(actual)
                    if not np.isfinite(aval) or aval < float(expected.get('min')):
                        return False
                if 'max' in expected:
                    aval = _safe_float(actual)
                    if not np.isfinite(aval) or aval > float(expected.get('max')):
                        return False
            else:
                if actual != expected:
                    return False
        return True
    return True



def _sample_global_candidate_from_schema(
    param_schema: Dict[str, Any],
    *,
    rng: np.random.Generator,
    method: str = "hybrid",
    sample_index: int = 1,
    base_payload: Optional[Dict[str, Any]] = None,
    seed_payloads: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    schema = dict(param_schema or {})
    if not schema:
        return {}
    keys = [str(k) for k in schema.keys()]
    base = dict(base_payload or {})
    seeds = [dict(x) for x in list(seed_payloads or []) if isinstance(x, dict) and x]
    mode = str(method or "hybrid").strip().lower()
    if mode not in {"random", "lhs", "sobol", "hybrid", "mutate", "branch"}:
        mode = "hybrid"

    if mode == "hybrid":
        draw = float(rng.random())
        if seeds and draw < 0.35:
            mode_eff = "mutate"
        elif seeds and draw < 0.50:
            mode_eff = "branch"
        elif draw < 0.72:
            mode_eff = "sobol"
        elif draw < 0.90:
            mode_eff = "lhs"
        else:
            mode_eff = "random"
    else:
        mode_eff = mode

    candidate: Dict[str, Any] = {}
    use_seed = mode_eff in {"mutate", "branch"} and bool(seeds)
    seed = dict(seeds[int(rng.integers(0, len(seeds)))]) if use_seed else {}
    unit_vec: Optional[np.ndarray] = None
    if mode_eff == "sobol":
        unit_vec = _sobol_unit_vector(len(keys), sample_index, seed=42)
    elif mode_eff == "lhs":
        unit_vec = _lhs_unit_vector(len(keys), sample_index, seed=42)

    for dim_idx, key in enumerate(keys):
        spec = schema.get(key)
        if isinstance(spec, dict):
            default = spec.get("default")
        else:
            default = None

        parent_val = seed.get(key, base.get(key, default))
        if mode_eff == "mutate" and (key in seed or key in base) and float(rng.random()) < 0.65:
            if _schema_is_categorical(spec):
                choices = [x for x in _schema_values_from_spec(spec) if x != parent_val]
                if choices and float(rng.random()) < 0.55:
                    candidate[key] = choices[int(rng.integers(0, len(choices)))]
                else:
                    candidate[key] = parent_val
            else:
                val = _schema_choice_from_unit(spec, float(rng.random()), rng=rng)
                candidate[key] = parent_val if val is None and parent_val is not None else (val if val is not None else parent_val)
            continue

        if mode_eff == "branch" and _schema_is_categorical(spec):
            choices = _schema_values_from_spec(spec)
            if choices:
                if parent_val in choices and float(rng.random()) < 0.50 and len(choices) > 1:
                    others = [x for x in choices if x != parent_val]
                    candidate[key] = others[int(rng.integers(0, len(others)))]
                else:
                    candidate[key] = choices[int(rng.integers(0, len(choices)))]
                continue

        if mode_eff in {"lhs", "sobol"} and unit_vec is not None:
            u = float(unit_vec[min(dim_idx, len(unit_vec) - 1)])
            if seeds and parent_val is not None and float(rng.random()) < 0.20:
                candidate[key] = parent_val
            else:
                val = _schema_choice_from_unit(spec, u, rng=rng)
                candidate[key] = parent_val if val is None and parent_val is not None else (val if val is not None else parent_val)
            continue

        choices = _schema_values_from_spec(spec)
        if choices:
            if parent_val in choices and float(rng.random()) < 0.25:
                candidate[key] = parent_val
            else:
                candidate[key] = choices[int(rng.integers(0, len(choices)))]
            continue

        val = _schema_choice_from_unit(spec, float(rng.random()), rng=rng)
        candidate[key] = parent_val if val is None and parent_val is not None else val

    return {str(k): v for k, v in candidate.items() if v is not None or base.get(str(k)) is None}



def _prune_inactive_or_incompatible_dims(candidate_payload: Dict[str, Any], *, base_cfg_payload: Dict[str, Any], param_schema: Dict[str, Any]) -> Dict[str, Any]:
    base_payload = dict(_tuning_valid_cfg_payload(base_cfg_payload or {}))
    merged = _recursive_merge_payloads(base_payload, dict(candidate_payload or {}))
    schema = dict(param_schema or {})

    pruned: Dict[str, Any] = {}
    for key, value in dict(candidate_payload or {}).items():
        spec = schema.get(key)
        active = True
        if isinstance(spec, dict):
            active = _condition_is_satisfied(merged, spec.get("active_if", spec.get("enabled_if", spec.get("depends_on", spec.get("conditional_on")))))
            if active:
                incompatible = spec.get("inactive_if", spec.get("disabled_if"))
                if incompatible not in {None, '', {}} and _condition_is_satisfied(merged, incompatible):
                    active = False
        if active:
            pruned[str(key)] = value

    try:
        from src.investment import MicroPipelineConfig, sanitize_param_space_for_engine
    except Exception:
        try:
            from investment import MicroPipelineConfig, sanitize_param_space_for_engine
        except Exception:
            MicroPipelineConfig = None
            sanitize_param_space_for_engine = None

    if MicroPipelineConfig is not None and callable(sanitize_param_space_for_engine):
        try:
            cfg = MicroPipelineConfig(**dict(_tuning_valid_cfg_payload(merged)))
            singleton_space = {str(k): [v] for k, v in pruned.items()}
            engine_info = sanitize_param_space_for_engine(singleton_space, cfg)
            sanitized = dict(engine_info.get("sanitized_param_space", {}) or {})
            if sanitized:
                sanitized_patch = {}
                for key, values in sanitized.items():
                    vals = list(values or [])
                    if vals:
                        sanitized_patch[str(key)] = vals[0]
                pruned = sanitized_patch
        except Exception:
            pass

    final_patch: Dict[str, Any] = {}
    for key, value in pruned.items():
        spec = schema.get(key)
        try:
            final_patch[str(key)] = _coerce_sampled_param_value(spec, value)
        except Exception:
            final_patch[str(key)] = value
    return final_patch



def _candidate_fingerprint_from_payload(candidate_payload: Dict[str, Any], *, base_cfg_payload: Optional[Dict[str, Any]] = None) -> str:
    merged = _recursive_merge_payloads(dict(base_cfg_payload or {}), dict(candidate_payload or {}))
    valid = _tuning_valid_cfg_payload(merged)
    return _recursive_candidate_params_key(valid)



def _dedupe_fingerprints(
    candidate_payloads: Sequence[Dict[str, Any]],
    *,
    base_cfg_payload: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for payload in list(candidate_payloads or []):
        if not isinstance(payload, dict):
            continue
        fp = _candidate_fingerprint_from_payload(payload, base_cfg_payload=base_cfg_payload)
        if fp in seen:
            continue
        seen.add(fp)
        out.append(dict(payload))
    return out



def _cache_run_result_by_fingerprint(
    cache: Dict[str, Dict[str, Any]],
    *,
    fingerprint: str,
    runner,
) -> Dict[str, Any]:
    key = str(fingerprint or "")
    if key in cache:
        row = dict(cache[key])
        row["cache_hit"] = True
        return row
    row = dict(runner())
    row["cache_hit"] = False
    cache[key] = dict(row)
    return row



def _run_multiobjective_once(
    panel_df,
    *,
    base_cfg_payload: Dict[str, Any],
    param_dict: Dict[str, Any],
    objective_names: Sequence[str],
) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
    """Run pipeline once and extract all requested objectives explicitly from the same run."""
    obj_cols = [str(x) for x in list(objective_names or []) if str(x).strip()]
    primary_objective = obj_cols[0] if obj_cols else "sharpe"

    try:
        from src.investment import (
            MicroPipelineConfig,
            config_from_dict,
            config_fingerprint,
            run_micro_investment_pipeline,
        )
    except Exception:
        from investment import (
            MicroPipelineConfig,
            config_from_dict,
            config_fingerprint,
            run_micro_investment_pipeline,
        )

    valid_fields = {f.name for f in MicroPipelineConfig.__dataclass_fields__.values()}
    payload = {k: v for k, v in dict(base_cfg_payload or {}).items() if k in valid_fields}
    for k, v in dict(param_dict or {}).items():
        if k in valid_fields:
            payload[k] = v

    cfg = config_from_dict(payload)
    run = run_micro_investment_pipeline(panel_df, cfg=cfg)
    perf = _run_perf_dict(run)
    risk = run.get("risk_summary", {}) or {}
    div = run.get("diversification_summary", {}) or {}
    uni = run.get("universe_summary", {}) or {}

    summary = {
        "objective_name": str(primary_objective),
        "objective_names": tuple(obj_cols),
        "objective_value": _extract_tuning_objective_value(run, objective=primary_objective),
        "config_fingerprint": config_fingerprint(cfg),
        "sharpe": _safe_float(perf.get("sharpe")),
        "cagr": _safe_float(perf.get("cagr")),
        "max_drawdown": _safe_float(perf.get("max_drawdown")),
        "annual_volatility": _safe_float(perf.get("annual_volatility", perf.get("annualized_volatility"))),
        "mean_turnover": _safe_float(perf.get("mean_turnover")),
        "information_ratio": _safe_float(perf.get("information_ratio")),
        "active_return_annual": _safe_float(perf.get("active_return_annual")),
        "tracking_error_annual": _safe_float(perf.get("tracking_error_annual")),
        "mean_effective_breadth": _safe_float(div.get("mean_effective_breadth")),
        "mean_diversification_ratio": _safe_float(div.get("mean_diversification_ratio")),
        "mean_effective_risk_bets": _safe_float(risk.get("mean_effective_risk_bets")),
        "mean_active_assets": _safe_float(uni.get("mean_active_assets")),
        "mean_effective_n_assets": _safe_float(div.get("mean_effective_n_assets")),
        "inverse_concentration": _safe_float(_extract_inverse_concentration_proxy(run)),
    }
    for obj in obj_cols:
        summary[str(obj)] = _safe_float(_extract_tuning_objective_value(run, objective=obj))
        if str(obj).startswith("composite_"):
            summary.update(compute_multi_objective_score(run, objective=str(obj)))
    if primary_objective not in summary:
        summary[str(primary_objective)] = _safe_float(summary.get("objective_value"))
    return _safe_float(summary.get("objective_value")), run, summary



def _objective_score_from_population_row(
    row: pd.Series,
    *,
    objective_names: Sequence[str],
    objective_specs: Optional[Dict[str, Dict[str, str]]] = None,
) -> float:
    one = pd.DataFrame([dict(row.to_dict())])
    obj_cols = [str(x) for x in list(objective_names or []) if str(x).strip() and str(x) in one.columns]
    if not obj_cols:
        return np.nan
    specs = _coerce_objective_specs_for_hv(obj_cols, objective_specs=objective_specs)
    norm = normalize_objective_matrix(one, objective_cols=obj_cols, objective_specs=specs, prefix="global_pick_norm_")
    norm_cols = [f"global_pick_norm_{c}" for c in obj_cols if f"global_pick_norm_{c}" in norm.columns]
    if not norm_cols:
        return np.nan
    return float(pd.to_numeric(norm.iloc[0][norm_cols], errors="coerce").mean())



def run_global_multiobjective_search(
    panel_df,
    *,
    base_cfg_payload: dict,
    param_schema: dict,
    objective_names: Sequence[str],
    budget: int = 200,
    method: str = "hybrid",
) -> dict:
    """Global discrete/quasi-random multi-objective search guided by param_schema.

    Minimal pragmatic backend:
    - global schema-guided sampling (random / quasi-LHS via Halton)
    - mutation from good points
    - categorical branching
    - engine-aware pruning using sanitize_param_space_for_engine when available
    - run-result cache keyed by config fingerprint
    """
    if panel_df is None or not isinstance(panel_df, pd.DataFrame) or panel_df.empty:
        return {
            "population_df": pd.DataFrame(),
            "frontier_df": pd.DataFrame(),
            "history_df": pd.DataFrame(),
            "best_compromise_payload": {},
            "frontier_hypervolume": np.nan,
            "search_method_effective": str(method or "hybrid"),
        }

    schema = dict(param_schema or {})
    obj_cols = [str(x) for x in list(objective_names or []) if str(x).strip()]
    if not schema or not obj_cols:
        return {
            "population_df": pd.DataFrame(),
            "frontier_df": pd.DataFrame(),
            "history_df": pd.DataFrame(),
            "best_compromise_payload": dict(_tuning_valid_cfg_payload(base_cfg_payload or {})),
            "frontier_hypervolume": np.nan,
            "search_method_effective": str(method or "hybrid"),
        }

    base_payload = dict(_tuning_valid_cfg_payload(base_cfg_payload or {}))
    total_budget = max(1, int(budget))
    method_req = str(method or "hybrid").strip().lower() or "hybrid"
    if method_req not in {"hybrid", "random", "lhs", "sobol", "mutate", "branch"}:
        method_req = "hybrid"

    objective_specs = _coerce_objective_specs_for_hv(obj_cols, objective_specs=None)
    rng = np.random.default_rng(42)
    eval_cache: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    history_rows: List[Dict[str, Any]] = []
    seed_payloads: List[Dict[str, Any]] = []

    def _evaluate_candidate_patch(candidate_patch: Dict[str, Any], *, sample_method: str, sample_step: int) -> Dict[str, Any]:
        pruned_patch = _prune_inactive_or_incompatible_dims(candidate_patch, base_cfg_payload=base_payload, param_schema=schema)
        fingerprint = _candidate_fingerprint_from_payload(pruned_patch, base_cfg_payload=base_payload)

        def _runner() -> Dict[str, Any]:
            merged = _recursive_merge_payloads(base_payload, pruned_patch)
            try:
                _, _, summary = _run_multiobjective_once(
                    panel_df,
                    base_cfg_payload=merged,
                    param_dict={},
                    objective_names=obj_cols,
                )
                row = _trial_row_from_result(
                    sample_step,
                    method="global_multiobjective_search",
                    params=dict(pruned_patch),
                    result_summary=summary,
                    status="ok",
                )
            except Exception as exc:
                row = {"trial": int(sample_step), "method": "global_multiobjective_search", "status": "error", "error": str(exc)}
                for k, v in dict(pruned_patch).items():
                    row[k] = v
                for col in obj_cols:
                    row[col] = np.nan
            row["candidate_params_json"] = _recursive_json_dumps(dict(pruned_patch))
            row["full_candidate_payload_json"] = _recursive_json_dumps(merged)
            row["candidate_param_count"] = int(len(pruned_patch))
            row["candidate_fingerprint"] = str(fingerprint)
            row["sample_method"] = str(sample_method)
            row["sample_step"] = int(sample_step)
            return row

        row = _cache_run_result_by_fingerprint(eval_cache, fingerprint=fingerprint, runner=_runner)
        row["candidate_fingerprint"] = str(fingerprint)
        row["sample_method"] = str(sample_method)
        row["sample_step"] = int(sample_step)
        row["candidate_params_json"] = _recursive_json_dumps(dict(pruned_patch))
        row["full_candidate_payload_json"] = _recursive_json_dumps(_recursive_merge_payloads(base_payload, pruned_patch))
        row["candidate_param_count"] = int(len(pruned_patch))
        row["pruned_param_count"] = int(len(pruned_patch))
        row["cache_hit"] = bool(row.get("cache_hit", False))
        return row

    step = 0
    pending: List[Tuple[Dict[str, Any], str]] = []
    initial_n = min(total_budget, max(8, int(round(total_budget * 0.35))))
    for idx in range(initial_n):
        if len(pending) >= total_budget:
            break
        if method_req == "hybrid":
            sample_method = "sobol" if idx % 2 == 0 else "lhs"
        else:
            sample_method = method_req
        cand = _sample_global_candidate_from_schema(schema, rng=rng, method=sample_method, sample_index=idx + 1, base_payload=base_payload)
        pending.append((cand, sample_method))

    while step < total_budget and pending:
        deduped = _dedupe_fingerprints([x for x, _ in pending], base_cfg_payload=base_payload)
        dedup_map = {_candidate_fingerprint_from_payload(x, base_cfg_payload=base_payload): x for x in deduped}
        ordered_pending: List[Tuple[Dict[str, Any], str]] = []
        seen_pending = set()
        for cand, smethod in pending:
            fp = _candidate_fingerprint_from_payload(cand, base_cfg_payload=base_payload)
            if fp in seen_pending or fp not in dedup_map:
                continue
            seen_pending.add(fp)
            ordered_pending.append((dedup_map[fp], smethod))
        pending = []
        for cand, smethod in ordered_pending:
            if step >= total_budget:
                break
            step += 1
            row = _evaluate_candidate_patch(cand, sample_method=smethod, sample_step=step)
            rows.append(dict(row))

        population_now = pd.DataFrame(rows)
        annotated_now = _annotate_multiobjective_population(population_now, objective_names=obj_cols, objective_specs=objective_specs)
        pop_now = pd.DataFrame(annotated_now.get("population_df", population_now)).copy()
        frontier_now = pd.DataFrame(annotated_now.get("frontier_df", pd.DataFrame())).copy()
        frontier_hv_now = _safe_float(annotated_now.get("frontier_hypervolume"))

        elite_df = pop_now.copy()
        if not elite_df.empty:
            elite_df["__global_score"] = elite_df.apply(lambda r: _objective_score_from_population_row(r, objective_names=obj_cols, objective_specs=objective_specs), axis=1)
            elite_df["pareto_rank"] = pd.to_numeric(elite_df.get("pareto_rank"), errors="coerce")
            elite_df = elite_df.sort_values(["pareto_rank", "__global_score", "crowding_distance"], ascending=[True, False, False], na_position="last")
            elite_df = elite_df.head(max(3, min(12, max(1, total_budget // 10))))
            seed_payloads = []
            for _, r in elite_df.iterrows():
                patch = _extract_candidate_patch_from_row(r, initial_param_space=schema)
                if patch:
                    seed_payloads.append(patch)
            seed_payloads = _dedupe_fingerprints(seed_payloads, base_cfg_payload=base_payload)

        history_rows.append({
            "step": int(step),
            "evaluated_count": int(len(rows)),
            "unique_fingerprints": int(len(eval_cache)),
            "frontier_size": int(frontier_now.shape[0]),
            "frontier_hypervolume": frontier_hv_now,
            "cache_hits_cumulative": int(sum(1 for r in rows if bool(r.get("cache_hit", False)))),
        })

        if step >= total_budget:
            break

        remaining = total_budget - step
        refill_n = min(max(4, min(12, remaining)), remaining)
        next_pending: List[Tuple[Dict[str, Any], str]] = []
        for idx in range(refill_n):
            if seed_payloads and idx < max(1, refill_n // 2):
                sample_method = "mutate" if idx % 2 == 0 else "branch"
            else:
                if method_req == "hybrid":
                    sample_method = "sobol" if idx % 2 == 0 else "lhs"
                elif method_req in {"lhs", "sobol"}:
                    sample_method = method_req
                else:
                    sample_method = "random"
            cand = _sample_global_candidate_from_schema(
                schema,
                rng=rng,
                method=sample_method,
                sample_index=step + idx + 1,
                base_payload=base_payload,
                seed_payloads=seed_payloads,
            )
            next_pending.append((cand, sample_method))
        pending = next_pending

    population_df = pd.DataFrame(rows)
    annotated = _annotate_multiobjective_population(population_df, objective_names=obj_cols, objective_specs=objective_specs)
    population_df = pd.DataFrame(annotated.get("population_df", population_df)).copy()
    frontier_df = pd.DataFrame(annotated.get("frontier_df", pd.DataFrame())).copy()
    frontier_hypervolume = _safe_float(annotated.get("frontier_hypervolume"))
    history_df = pd.DataFrame(history_rows)

    best_compromise_payload: Dict[str, Any] = {}
    if not frontier_df.empty:
        picked = choose_multiobjective_solution_by_policy(
            frontier_df,
            policy="weighted_hypervolume",
            objective_cols=obj_cols,
            objective_specs=objective_specs,
            hypervolume_preset="balanced",
        )
        picked_frontier = pd.DataFrame(picked.get("frontier_df", frontier_df)).copy()
        best_row_dict = dict(picked.get("selected_row") or {})
        if best_row_dict:
            best_row = pd.Series(best_row_dict)
        elif not picked_frontier.empty:
            best_row = picked_frontier.iloc[0]
        else:
            best_row = pd.Series(dtype="object")
        best_patch = _extract_candidate_patch_from_row(best_row, initial_param_space=schema)
        best_compromise_payload = _recursive_merge_payloads(base_payload, best_patch)
        frontier_hypervolume = _safe_float(picked.get("frontier_hypervolume")) if np.isfinite(_safe_float(picked.get("frontier_hypervolume"))) else frontier_hypervolume
        frontier_df = picked_frontier if not picked_frontier.empty else frontier_df

    return {
        "population_df": population_df,
        "frontier_df": frontier_df,
        "history_df": history_df,
        "best_compromise_payload": best_compromise_payload,
        "frontier_hypervolume": frontier_hypervolume,
        "search_method_effective": method_req,
    }


def _select_seed_configs_from_frontier_details(
    frontier_df: pd.DataFrame,
    *,
    base_cfg_payload: Dict[str, Any],
    initial_param_space: Dict[str, Any],
    max_seeds: int = 3,
    objective_names: Optional[Sequence[str]] = None,
    objective_specs: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, Any]:
    frontier = pd.DataFrame(frontier_df).copy()
    if frontier.empty:
        return {
            "seed_payloads": [],
            "seed_table": pd.DataFrame(),
            "seed_reason": "empty_frontier",
            "selection_mix": {},
        }

    target_n = max(1, int(max_seeds))
    obj_cols = [str(x) for x in list(objective_names or []) if str(x).strip() and str(x) in frontier.columns]
    if not obj_cols:
        fallback_cols = [c for c in ["sharpe", "cagr", "max_drawdown", "mean_turnover", "diversification", "stability"] if c in frontier.columns]
        obj_cols = fallback_cols
    specs = _coerce_objective_specs_for_hv(obj_cols, objective_specs=objective_specs) if obj_cols else {}

    work = frontier.copy()
    if obj_cols:
        norm = normalize_objective_matrix(work, objective_cols=obj_cols, objective_specs=specs, prefix="seed_norm_")
        for col in norm.columns:
            if col not in work.columns:
                work[col] = norm[col]
        norm_cols = [f"seed_norm_{c}" for c in obj_cols if f"seed_norm_{c}" in work.columns]
    else:
        norm_cols = []

    work["__hv_rank"] = pd.to_numeric(work.get("hv_contribution"), errors="coerce").fillna(0.0)
    work["__crowding_rank"] = pd.to_numeric(work.get("crowding_distance"), errors="coerce")
    if work["__crowding_rank"].notna().any():
        finite_crowd = work["__crowding_rank"].replace([np.inf, -np.inf], np.nan)
        fill_val = float(finite_crowd.max()) if finite_crowd.notna().any() else 0.0
        work["__crowding_rank"] = work["__crowding_rank"].replace(np.inf, fill_val).replace(-np.inf, 0.0).fillna(0.0)
    else:
        work["__crowding_rank"] = 0.0

    if norm_cols:
        work["__balanced_score"] = pd.to_numeric(work[norm_cols], errors="coerce").mean(axis=1)
        dist_to_ideal = np.sqrt(np.square(pd.to_numeric(work[norm_cols], errors="coerce").fillna(0.0) - 1.0).sum(axis=1))
        crowd = pd.to_numeric(work["__crowding_rank"], errors="coerce").fillna(0.0)
        work["__knee_score"] = (1.0 - dist_to_ideal) + 0.10 * crowd
    else:
        fallback_numeric = [c for c in ["sharpe", "cagr", "diversification", "stability"] if c in work.columns]
        work["__balanced_score"] = pd.to_numeric(work[fallback_numeric], errors="coerce").mean(axis=1) if fallback_numeric else 0.0
        work["__knee_score"] = work["__balanced_score"] + 0.10 * pd.to_numeric(work["__crowding_rank"], errors="coerce").fillna(0.0)

    hv_n = max(1, int(round(target_n * 0.40)))
    extreme_n = max(1, int(round(target_n * 0.40)))
    diverse_n = max(1, target_n - hv_n - extreme_n)
    total_target = hv_n + extreme_n + diverse_n
    if total_target > target_n:
        diverse_n = max(1, diverse_n - (total_target - target_n))

    chosen_idx: List[Any] = []
    chosen_reasons: Dict[Any, str] = {}

    def _take_indices(indexes: Sequence[Any], reason: str, max_take: int) -> None:
        taken = 0
        for idx in indexes:
            if idx in chosen_reasons:
                continue
            chosen_idx.append(idx)
            chosen_reasons[idx] = reason
            taken += 1
            if taken >= max_take:
                break

    hv_ranked = work.sort_values(["__hv_rank", "trial"], ascending=[False, True], na_position="last")
    _take_indices(list(hv_ranked.index), "hv_contribution", hv_n)

    if obj_cols:
        extreme_candidates: List[Any] = []
        for col in obj_cols:
            ncol = f"seed_norm_{col}"
            if ncol not in work.columns:
                continue
            ranked = work.sort_values([ncol, "__crowding_rank"], ascending=[False, False], na_position="last")
            if not ranked.empty:
                extreme_candidates.append(ranked.index[0])
        if not extreme_candidates:
            extreme_candidates = list(work.sort_values(["__balanced_score"], ascending=[False], na_position="last").index)
        _take_indices(extreme_candidates, "objective_extreme", extreme_n)
    else:
        _take_indices(list(work.sort_values(["__balanced_score"], ascending=[False], na_position="last").index), "objective_extreme", extreme_n)

    remaining = work.loc[[idx for idx in work.index if idx not in chosen_reasons]].copy()
    if diverse_n > 0 and not remaining.empty:
        if norm_cols:
            feature_mat = pd.to_numeric(remaining[norm_cols], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
        else:
            feature_mat = pd.to_numeric(remaining[["__balanced_score", "__crowding_rank"]], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
        selected_local: List[int] = []
        if feature_mat.shape[0] > 0:
            knee_order = list(np.argsort(-pd.to_numeric(remaining["__knee_score"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")))
            if knee_order:
                selected_local.append(int(knee_order[0]))
            while len(selected_local) < min(diverse_n, feature_mat.shape[0]):
                best_i = None
                best_d = -np.inf
                for i in range(feature_mat.shape[0]):
                    if i in selected_local:
                        continue
                    if not selected_local:
                        d = float(np.linalg.norm(feature_mat[i]))
                    else:
                        d = float(min(np.linalg.norm(feature_mat[i] - feature_mat[j]) for j in selected_local))
                    d += 0.05 * float(pd.to_numeric(remaining.iloc[i]["__knee_score"], errors="coerce") or 0.0)
                    if d > best_d:
                        best_d = d
                        best_i = i
                if best_i is None:
                    break
                selected_local.append(int(best_i))
        diverse_indices = [remaining.index[i] for i in selected_local]
        _take_indices(diverse_indices, "diverse_knee", diverse_n)

    if len(chosen_idx) < target_n:
        filler = work.sort_values(["__balanced_score", "__hv_rank", "__crowding_rank"], ascending=[False, False, False], na_position="last")
        _take_indices(list(filler.index), "balanced_fill", target_n - len(chosen_idx))

    selected = work.loc[chosen_idx].copy() if chosen_idx else work.head(target_n).copy()
    if not selected.empty:
        selected["seed_reason"] = [chosen_reasons.get(idx, "selected") for idx in selected.index]
        selected["seed_order"] = np.arange(1, selected.shape[0] + 1)

    out_payloads: List[Dict[str, Any]] = []
    seen = set()
    for _, row in selected.iterrows():
        patch = _extract_candidate_patch_from_row(row, initial_param_space=initial_param_space)
        payload = _recursive_merge_payloads(base_cfg_payload, patch)
        key = _recursive_candidate_params_key({k: payload.get(k) for k in initial_param_space.keys() if k in payload})
        if key in seen:
            continue
        seen.add(key)
        out_payloads.append(payload)

    return {
        "seed_payloads": out_payloads,
        "seed_table": selected.reset_index(drop=True),
        "seed_reason": "hybrid_hv_extreme_diverse",
        "selection_mix": {
            "hv_contribution": hv_n,
            "objective_extreme": extreme_n,
            "diverse_knee": diverse_n,
            "actual_selected": int(len(out_payloads)),
        },
    }



def _select_seed_configs_from_frontier(
    frontier_df: pd.DataFrame,
    *,
    base_cfg_payload: Dict[str, Any],
    initial_param_space: Dict[str, Any],
    max_seeds: int = 3,
    objective_names: Optional[Sequence[str]] = None,
    objective_specs: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    details = _select_seed_configs_from_frontier_details(
        frontier_df,
        base_cfg_payload=base_cfg_payload,
        initial_param_space=initial_param_space,
        max_seeds=max_seeds,
        objective_names=objective_names,
        objective_specs=objective_specs,
    )
    return list(details.get("seed_payloads", []) or [])



def _narrow_param_space_around_frontier(
    param_space: Dict[str, Any],
    *,
    frontier_df: pd.DataFrame,
    seed_payloads: Sequence[Dict[str, Any]],
    base_cfg_payload: Optional[Dict[str, Any]] = None,
    shrink_factor: float = 0.5,
) -> Dict[str, List[Any]]:
    space = {str(k): _recursive_unique_preserve(list(v or [])) for k, v in dict(param_space or {}).items() if list(v or [])}
    if not space:
        return {}
    frontier = pd.DataFrame(frontier_df).copy()
    base_shrink = float(np.clip(_safe_float(shrink_factor), 0.05, 1.0)) if np.isfinite(_safe_float(shrink_factor)) else 0.5
    seeds = [dict(x) for x in list(seed_payloads or []) if isinstance(x, dict)]
    out: Dict[str, List[Any]] = {}

    for key, values in space.items():
        values = list(values)
        if len(values) <= 1:
            out[key] = list(values)
            continue

        seed_vals = [seed.get(key) for seed in seeds if key in seed]
        seed_vals = [v for v in seed_vals if v is not None]
        if not seed_vals:
            base_v = dict(base_cfg_payload or {}).get(key)
            if base_v is not None:
                seed_vals = [base_v]

        numeric_vals = pd.to_numeric(pd.Series(values), errors="coerce")
        all_numeric = bool(numeric_vals.notna().all())
        adaptive_shrink = base_shrink

        if all_numeric and not frontier.empty and key in frontier.columns:
            frontier_num = pd.to_numeric(frontier[key], errors="coerce").dropna().astype(float)
            full_min = float(numeric_vals.min())
            full_max = float(numeric_vals.max())
            full_span = max(full_max - full_min, 1e-9)
            if frontier_num.shape[0] >= 2:
                frontier_span = float(frontier_num.max() - frontier_num.min())
                span_ratio = float(np.clip(frontier_span / full_span, 0.0, 1.0))
                concentration = 1.0 - span_ratio
                adaptive_shrink = float(np.clip(base_shrink * (0.60 + 0.80 * span_ratio), 0.10, 1.0))
                if concentration > 0.75:
                    adaptive_shrink = float(np.clip(base_shrink * 0.55, 0.10, 1.0))
            elif frontier_num.shape[0] == 1:
                adaptive_shrink = float(np.clip(base_shrink * 0.45, 0.10, 1.0))
        elif not all_numeric and seed_vals:
            unique_seed_ratio = len({_recursive_json_dumps(v) for v in seed_vals}) / max(len(values), 1)
            adaptive_shrink = float(np.clip(base_shrink * (0.50 + 1.00 * unique_seed_ratio), 0.10, 1.0))

        if not seed_vals:
            keep_n = max(1, int(np.ceil(len(values) * adaptive_shrink)))
            out[key] = list(values[:keep_n])
            continue

        if all_numeric:
            ordered_pairs = sorted([(float(v), raw) for v, raw in zip(numeric_vals.tolist(), values)], key=lambda x: x[0])
            ordered_numeric = np.asarray([v for v, _ in ordered_pairs], dtype="float64")
            ordered_raw = [raw for _, raw in ordered_pairs]
            seed_num = pd.to_numeric(pd.Series(seed_vals), errors="coerce").dropna().astype(float).tolist()
            if not seed_num:
                keep_n = max(2, int(np.ceil(len(values) * adaptive_shrink)))
                out[key] = ordered_raw[:keep_n]
                continue
            distances = np.min(np.abs(ordered_numeric[:, None] - np.asarray(seed_num, dtype="float64")[None, :]), axis=1)
            keep_n = max(2, int(np.ceil(len(values) * adaptive_shrink)))
            keep_idx = np.argsort(distances)[:keep_n]
            keep = [ordered_raw[int(i)] for i in keep_idx]
            keep = _recursive_unique_preserve(keep + list(seed_vals[:2]))
            out[key] = keep
        else:
            keep = [v for v in values if any(_recursive_json_dumps(v) == _recursive_json_dumps(sv) for sv in seed_vals)]
            if dict(base_cfg_payload or {}).get(key) is not None:
                keep.append(dict(base_cfg_payload or {}).get(key))
            keep = _recursive_unique_preserve(keep)
            if not keep:
                keep_n = max(1, int(np.ceil(len(values) * adaptive_shrink)))
                keep = list(values[:keep_n])
            out[key] = keep
    return out



def _payload_signature(payload: Optional[Dict[str, Any]], *, keys: Optional[Sequence[str]] = None) -> str:
    payload = dict(payload or {})
    if keys is not None:
        payload = {str(k): payload.get(k) for k in list(keys) if str(k) in payload}
    return _recursive_json_dumps(payload)



def _detect_hypervolume_convergence(
    history_df: pd.DataFrame,
    *,
    convergence_tol: float = 1e-3,
    objective_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    hist = pd.DataFrame(history_df).copy()
    default = {
        "converged": False,
        "reason": "continue",
        "delta_hv_abs": np.nan,
        "delta_hv_rel": np.nan,
        "frontier_stability": np.nan,
        "best_payload_changed": True,
        "objective_improvement_mean": np.nan,
    }
    if hist.shape[0] < 2:
        return default

    last = hist.iloc[-1]
    prev = hist.iloc[-2]
    tol = max(float(convergence_tol), 0.0)

    prev_hv = _safe_float(prev.get("frontier_hypervolume"))
    curr_hv = _safe_float(last.get("frontier_hypervolume"))
    hv_ok = np.isfinite(prev_hv) and np.isfinite(curr_hv)
    delta_hv_abs = float(curr_hv - prev_hv) if hv_ok else np.nan
    delta_hv_rel = float(delta_hv_abs / max(abs(prev_hv), 1e-9)) if hv_ok else np.nan

    prev_size = max(_safe_float(prev.get("frontier_size")), 0.0)
    curr_size = max(_safe_float(last.get("frontier_size")), 0.0)
    frontier_stability = np.nan
    if np.isfinite(prev_size) and np.isfinite(curr_size):
        frontier_stability = 1.0 - abs(curr_size - prev_size) / max(max(prev_size, curr_size), 1.0)

    prev_payload = str(prev.get("best_compromise_payload_json", "") or "")
    curr_payload = str(last.get("best_compromise_payload_json", "") or "")
    best_payload_changed = prev_payload != curr_payload

    objective_improvements = []
    use_obj = [str(x) for x in list(objective_names or []) if str(x).strip()]
    for col in use_obj:
        p = _safe_float(prev.get(f"best_{col}"))
        c = _safe_float(last.get(f"best_{col}"))
        if np.isfinite(p) and np.isfinite(c):
            objective_improvements.append(float(c - p))
    objective_improvement_mean = float(np.nanmean(objective_improvements)) if objective_improvements else np.nan

    hv_stalled = bool(hv_ok and delta_hv_abs <= tol and delta_hv_rel <= max(tol, 1e-6))
    frontier_stable = bool(np.isfinite(frontier_stability) and frontier_stability >= 0.90)
    objective_stalled = bool(not objective_improvements or (np.nanmax(objective_improvements) <= tol and abs(objective_improvement_mean) <= tol))

    if hv_stalled and frontier_stable and (not best_payload_changed) and objective_stalled:
        return {
            "converged": True,
            "reason": "multi_criteria_converged",
            "delta_hv_abs": delta_hv_abs,
            "delta_hv_rel": delta_hv_rel,
            "frontier_stability": frontier_stability,
            "best_payload_changed": best_payload_changed,
            "objective_improvement_mean": objective_improvement_mean,
        }
    return {
        "converged": False,
        "reason": "continue",
        "delta_hv_abs": delta_hv_abs,
        "delta_hv_rel": delta_hv_rel,
        "frontier_stability": frontier_stability,
        "best_payload_changed": best_payload_changed,
        "objective_improvement_mean": objective_improvement_mean,
    }


def run_recursive_multiobjective_search(
    panel_df,
    *,
    base_cfg_payload: dict,
    initial_param_space: dict,
    objective_names: Sequence[str],
    search_method: str = "nsga2",
    recursion_depth: int = 3,
    per_round_budget: int = 24,
    shrink_factor: float = 0.5,
    convergence_tol: float = 1e-3,
    seed_frontier_size: int = 3,
) -> dict:
    """Recursive frontier-level multi-objective search.

    Advanced pragmatic version:
    - hybrid seed selection (HV contributors + extremes + diverse/knee-like)
    - adaptive per-parameter shrink around the frontier
    - warm-start across rounds via seed_payloads injection
    - multi-criteria convergence (HV + frontier stability + best-payload stability + objective improvements)
    - explicit round telemetry for UI / debugging
    """
    method = str(search_method or "nsga2").strip().lower()

    def _build_recursive_search_result(
        *,
        population_df: Optional[pd.DataFrame] = None,
        frontier_df: Optional[pd.DataFrame] = None,
        history_df: Optional[pd.DataFrame] = None,
        round_summaries_df: Optional[pd.DataFrame] = None,
        best_compromise_payload: Optional[Dict[str, Any]] = None,
        best_round: Optional[int] = None,
        search_method_effective: Optional[str] = None,
        frontier_hypervolume: Any = np.nan,
        converged: bool = False,
        convergence_reason: Optional[str] = None,
        best_frontier_df: Optional[pd.DataFrame] = None,
        best_population_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        pop = pd.DataFrame(population_df).copy() if isinstance(population_df, pd.DataFrame) else pd.DataFrame()
        frontier = pd.DataFrame(frontier_df).copy() if isinstance(frontier_df, pd.DataFrame) else pd.DataFrame()
        history = pd.DataFrame(history_df).copy() if isinstance(history_df, pd.DataFrame) else pd.DataFrame()
        round_summary = pd.DataFrame(round_summaries_df).copy() if isinstance(round_summaries_df, pd.DataFrame) else history.copy()
        best_frontier = pd.DataFrame(best_frontier_df).copy() if isinstance(best_frontier_df, pd.DataFrame) else pd.DataFrame()
        best_population = pd.DataFrame(best_population_df).copy() if isinstance(best_population_df, pd.DataFrame) else pd.DataFrame()

        hv_value = _safe_float(frontier_hypervolume)
        if not np.isfinite(hv_value) and not history.empty and "frontier_hypervolume" in history.columns:
            hv_series = pd.to_numeric(history["frontier_hypervolume"], errors="coerce")
            if hv_series.notna().any():
                hv_value = float(hv_series.max())
        if not np.isfinite(hv_value) and not frontier.empty and "frontier_hypervolume" in frontier.columns:
            hv_series = pd.to_numeric(frontier["frontier_hypervolume"], errors="coerce")
            if hv_series.notna().any():
                hv_value = float(hv_series.iloc[0])

        return {
            "population_df": pop,
            "frontier_df": frontier,
            "history_df": history,
            "round_summaries_df": round_summary,
            "best_compromise_payload": dict(best_compromise_payload or {}),
            "best_round": (None if best_round is None or not pd.notna(best_round) else int(best_round)),
            "search_method_effective": str(search_method_effective or method),
            "frontier_hypervolume": hv_value,
            "converged": bool(converged),
            "convergence_reason": str(convergence_reason or ""),
            "best_frontier_df": best_frontier,
            "best_population_df": best_population,
        }

    if panel_df is None or not isinstance(panel_df, pd.DataFrame) or panel_df.empty:
        return _build_recursive_search_result(
            convergence_reason="empty_panel",
            search_method_effective=method,
        )

    depth = max(1, int(recursion_depth))
    budget = max(4, int(per_round_budget))
    requested_seed_frontier_size = _safe_float(seed_frontier_size)
    if np.isfinite(requested_seed_frontier_size):
        requested_seed_frontier_size = int(requested_seed_frontier_size)
    else:
        requested_seed_frontier_size = 3
    requested_seed_frontier_size = max(1, requested_seed_frontier_size)
    obj_cols = [str(x) for x in list(objective_names or []) if str(x).strip()]
    specs = _coerce_objective_specs_for_hv(obj_cols, objective_specs=None)
    current_space = _recursive_param_space_sanitized(initial_param_space, base_cfg_payload=base_cfg_payload)

    all_population_parts: List[pd.DataFrame] = []
    all_frontier_parts: List[pd.DataFrame] = []
    round_rows: List[Dict[str, Any]] = []
    best_frontier = pd.DataFrame()
    best_population = pd.DataFrame()
    best_payload: Dict[str, Any] = {}
    best_hv = -np.inf
    best_round: Optional[int] = None
    seed_payloads: List[Dict[str, Any]] = [dict(base_cfg_payload or {})]
    converged = False
    convergence_reason = "max_depth_reached"

    for round_idx in range(1, depth + 1):
        if not current_space:
            convergence_reason = "empty_param_space"
            break

        if best_payload:
            seed_payloads = _recursive_unique_preserve([dict(best_payload)] + list(seed_payloads or []))

        if method == "nsga2":
            nsga_population = max(6, min(int(budget), 32))
            nsga_generations = max(2, int(np.ceil(float(budget) / max(nsga_population, 1))) + 1)
            round_result = run_nsga2_tuning(
                panel_df,
                base_cfg_payload=base_cfg_payload,
                param_space=current_space,
                objective_names=obj_cols,
                population_size=nsga_population,
                generations=nsga_generations,
                mutation_rate=0.15,
                crossover_rate=0.90,
                seed=42 + round_idx,
                objective_specs=specs,
                seed_payloads=seed_payloads,
            )
            population_df = pd.DataFrame(round_result.get("population_df", pd.DataFrame())).copy()
            frontier_df = pd.DataFrame(round_result.get("frontier_df", pd.DataFrame())).copy()
            best_round_payload = dict(round_result.get("best_compromise_payload", {}) or {})
        else:
            local_df = auto_tune_around_config(
                panel_df,
                base_cfg_payload=base_cfg_payload,
                objective=obj_cols[0] if obj_cols else "sharpe",
                param_space=current_space,
                max_candidates=budget,
                two_stage_search=True,
            )
            annotated = _annotate_multiobjective_population(local_df, objective_names=obj_cols, objective_specs=specs)
            population_df = pd.DataFrame(annotated.get("population_df", pd.DataFrame())).copy()
            frontier_df = pd.DataFrame(annotated.get("frontier_df", pd.DataFrame())).copy()
            best_round_payload = {}
            if not frontier_df.empty:
                picked = choose_solution_by_weighted_hypervolume(frontier_df, objective_cols=obj_cols, objective_specs=specs, preset="balanced")
                frontier_df = pd.DataFrame(picked.get("frontier_df", frontier_df)).copy()
                selected_row = dict(picked.get("selected_row", {}) or {})
                best_round_payload = _recursive_merge_payloads(base_cfg_payload, _extract_candidate_patch_from_row(selected_row, initial_param_space=current_space))

        round_hv = np.nan
        if not frontier_df.empty:
            if "frontier_hypervolume" in frontier_df.columns:
                hv_vals = pd.to_numeric(frontier_df["frontier_hypervolume"], errors="coerce").dropna()
                round_hv = _safe_float(hv_vals.iloc[0] if hv_vals.shape[0] else np.nan)
            if not np.isfinite(round_hv):
                picked = choose_solution_by_weighted_hypervolume(frontier_df, objective_cols=obj_cols, objective_specs=specs, preset="balanced")
                frontier_df = pd.DataFrame(picked.get("frontier_df", frontier_df)).copy()
                round_hv = _safe_float(picked.get("frontier_hypervolume"))

        if not population_df.empty:
            population_df = population_df.copy()
            population_df["recursive_round"] = int(round_idx)
            all_population_parts.append(population_df)
        if not frontier_df.empty:
            frontier_df = frontier_df.copy()
            frontier_df["recursive_round"] = int(round_idx)
            all_frontier_parts.append(frontier_df)

        frontier_size = int(frontier_df.shape[0])
        population_size = int(population_df.shape[0])
        prev_hv = _safe_float(round_rows[-1].get("frontier_hypervolume")) if round_rows else np.nan
        delta_hv_abs = float(round_hv - prev_hv) if np.isfinite(round_hv) and np.isfinite(prev_hv) else np.nan
        delta_hv_rel = float(delta_hv_abs / max(abs(prev_hv), 1e-9)) if np.isfinite(delta_hv_abs) and np.isfinite(prev_hv) else np.nan

        frontier_spread = np.nan
        if not frontier_df.empty and obj_cols:
            norm = normalize_objective_matrix(frontier_df, objective_cols=obj_cols, objective_specs=specs, prefix="round_norm_")
            ncols = [f"round_norm_{c}" for c in obj_cols if f"round_norm_{c}" in norm.columns]
            if ncols:
                frontier_spread = float(pd.to_numeric(norm[ncols], errors="coerce").std(ddof=0).mean())

        best_obj_metrics = {}
        for col in obj_cols:
            if col in frontier_df.columns:
                vals = pd.to_numeric(frontier_df[col], errors="coerce").dropna()
                if vals.shape[0]:
                    best_obj_metrics[f"best_{col}"] = float(vals.max())
                else:
                    best_obj_metrics[f"best_{col}"] = np.nan
            else:
                best_obj_metrics[f"best_{col}"] = np.nan

        round_row = {
            "round": int(round_idx),
            "search_method": method,
            "population_size": population_size,
            "frontier_size": frontier_size,
            "frontier_hypervolume": round_hv,
            "delta_hv_abs": delta_hv_abs,
            "delta_hv_rel": delta_hv_rel,
            "frontier_spread": frontier_spread,
            "param_dim": int(len(current_space)),
            "search_space_cardinality_hint": int(sum(len(v) for v in current_space.values())),
            "best_compromise_payload_json": _recursive_json_dumps(best_round_payload),
            **best_obj_metrics,
        }
        round_rows.append(round_row)

        if np.isfinite(round_hv) and round_hv > best_hv:
            best_hv = float(round_hv)
            best_frontier = frontier_df.copy()
            best_population = population_df.copy()
            best_payload = dict(best_round_payload or best_payload)
            best_round = int(round_idx)

        conv = _detect_hypervolume_convergence(
            pd.DataFrame(round_rows),
            convergence_tol=convergence_tol,
            objective_names=obj_cols,
        )
        round_rows[-1]["frontier_stability"] = conv.get("frontier_stability")
        round_rows[-1]["best_payload_changed"] = conv.get("best_payload_changed")
        round_rows[-1]["objective_improvement_mean"] = conv.get("objective_improvement_mean")
        round_rows[-1]["convergence_signal"] = conv.get("reason")

        if bool(conv.get("converged", False)):
            converged = True
            convergence_reason = str(conv.get("reason") or "multi_criteria_converged")
            round_rows[-1]["stop_reason"] = convergence_reason
            break

        effective_seed_frontier_size = min(
            max(1, requested_seed_frontier_size),
            max(1, frontier_size),
        ) if frontier_size > 0 else max(1, requested_seed_frontier_size)

        seed_details = _select_seed_configs_from_frontier_details(
            frontier_df,
            base_cfg_payload=base_cfg_payload,
            initial_param_space=current_space,
            max_seeds=effective_seed_frontier_size,
            objective_names=obj_cols,
            objective_specs=specs,
        )
        seed_payloads = list(seed_details.get("seed_payloads", []) or [])
        if best_round_payload:
            seed_payloads = _recursive_unique_preserve([dict(best_round_payload)] + seed_payloads)
        if not seed_payloads:
            seed_payloads = [dict(base_cfg_payload or {})]

        current_space = _narrow_param_space_around_frontier(
            current_space,
            frontier_df=frontier_df,
            seed_payloads=seed_payloads,
            base_cfg_payload=base_cfg_payload,
            shrink_factor=shrink_factor,
        )

        round_rows[-1]["seed_count"] = int(len(seed_payloads))
        round_rows[-1]["seed_frontier_size_requested"] = int(requested_seed_frontier_size)
        round_rows[-1]["seed_frontier_size_effective"] = int(effective_seed_frontier_size)
        round_rows[-1]["seed_reason"] = str(seed_details.get("seed_reason") or "base_only")
        mix = dict(seed_details.get("selection_mix", {}) or {})
        round_rows[-1]["seed_mix_json"] = _recursive_json_dumps(mix)
        round_rows[-1]["stop_reason"] = "continue"

    history_df = pd.DataFrame(round_rows)
    population_all = pd.concat(all_population_parts, ignore_index=True, sort=False) if all_population_parts else pd.DataFrame()
    frontier_all = pd.concat(all_frontier_parts, ignore_index=True, sort=False) if all_frontier_parts else pd.DataFrame()

    if best_frontier.empty and not frontier_all.empty:
        frontier_all = frontier_all.sort_values(["recursive_round"], ascending=[True])
        last_round = int(pd.to_numeric(frontier_all["recursive_round"], errors="coerce").dropna().max())
        best_frontier = frontier_all.loc[pd.to_numeric(frontier_all["recursive_round"], errors="coerce") == last_round].copy()
        if not population_all.empty:
            best_population = population_all.loc[pd.to_numeric(population_all["recursive_round"], errors="coerce") == last_round].copy()
        if best_round is None:
            best_round = last_round

    if not best_payload and not best_frontier.empty:
        picked = choose_solution_by_weighted_hypervolume(best_frontier, objective_cols=obj_cols, objective_specs=specs, preset="balanced")
        best_payload = _recursive_merge_payloads(base_cfg_payload, _extract_candidate_patch_from_row(picked.get("selected_row", {}), initial_param_space=initial_param_space))

    if best_round is None and not history_df.empty and "frontier_hypervolume" in history_df.columns:
        hv_series = pd.to_numeric(history_df["frontier_hypervolume"], errors="coerce")
        if hv_series.notna().any():
            best_round = int(history_df.loc[hv_series.idxmax(), "round"])

    frontier_hypervolume = best_hv if np.isfinite(best_hv) else np.nan
    if not np.isfinite(frontier_hypervolume) and not history_df.empty and "frontier_hypervolume" in history_df.columns:
        hv_series = pd.to_numeric(history_df["frontier_hypervolume"], errors="coerce")
        if hv_series.notna().any():
            frontier_hypervolume = float(hv_series.max())

    if not converged and convergence_reason == "max_depth_reached" and not history_df.empty:
        history_df.loc[history_df.index[-1], "stop_reason"] = "max_depth_reached"

    return _build_recursive_search_result(
        population_df=population_all,
        frontier_df=frontier_all,
        history_df=history_df,
        round_summaries_df=history_df,
        best_compromise_payload=best_payload,
        best_round=best_round,
        search_method_effective=method,
        frontier_hypervolume=frontier_hypervolume,
        converged=converged,
        convergence_reason=convergence_reason,
        best_frontier_df=best_frontier,
        best_population_df=best_population,
    )

