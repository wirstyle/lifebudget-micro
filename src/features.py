"""Daily-to-monthly and daily-to-weekly feature engineering helpers.

This module sits upstream from ``src.investment``. It converts daily asset
returns into weekly or monthly asset panels with leakage-safe, backward-looking
features that the Strategy Engine can consume.

Input assumptions
-----------------
The input panel should contain at least:
- ``date``;
- ``asset``;
- ``return`` as a daily simple return in decimal form.

Output contract
---------------
The generated panels always preserve the Strategy Engine contract:
- ``date``;
- ``asset``;
- ``return`` as a simple return.

Additional feature columns may include intramonth volatility, momentum,
cross-sectional dispersion, macro context, and feature-based mu proxies. These
features are historical diagnostics/signals for educational backtests; they are
not forecasts or investment recommendations.

Design principles
-----------------
- keep daily-to-period aggregation explicit;
- avoid using information from future periods;
- keep compatibility with the current monthly micro-pipeline;
- keep richer research-style features optional and upstream from the engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================
# Config
# ============================================================

@dataclass(frozen=True)
class DailyFeatureConfig:
    date_col: str = "date"
    asset_col: str = "asset"
    return_col: str = "return"

    monthly_return_name: str = "return"
    weekly_return_name: str = "return"

    ewma_halflife_short: int = 5
    ewma_halflife_medium: int = 10
    ewma_halflife_long: int = 21

    rolling_short: int = 5
    rolling_medium: int = 10
    rolling_long: int = 21

    momentum_short: int = 5
    momentum_medium: int = 10
    momentum_long: int = 21

    min_obs_per_month: int = 5
    min_obs_for_window: int = 3

    annualisation_days: int = 252
    monthly_annualisation_days: int = 21

    clip_simple_return_low: float = -0.999999


# ============================================================
# Validation / cleaning
# ============================================================


def validate_daily_asset_panel(df: pd.DataFrame, cfg: DailyFeatureConfig | None = None) -> pd.DataFrame:
    cfg = cfg or DailyFeatureConfig()
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Daily asset panel must be a non-empty pandas DataFrame")

    need = {cfg.date_col, cfg.asset_col, cfg.return_col}
    missing = sorted(need - set(df.columns))
    if missing:
        raise ValueError(f"Daily asset panel is missing required columns: {missing}")

    out = df.copy()
    out[cfg.date_col] = pd.to_datetime(out[cfg.date_col], errors="coerce")
    out[cfg.asset_col] = out[cfg.asset_col].astype(str).str.upper().str.strip()
    out[cfg.return_col] = pd.to_numeric(out[cfg.return_col], errors="coerce")
    out = out.dropna(subset=[cfg.date_col, cfg.asset_col, cfg.return_col]).copy()
    out[cfg.return_col] = out[cfg.return_col].clip(lower=cfg.clip_simple_return_low)
    out = out.sort_values([cfg.asset_col, cfg.date_col]).reset_index(drop=True)

    if out.empty:
        raise ValueError("Daily asset panel is empty after cleaning")
    return out


# ============================================================
# Small helpers
# ============================================================


def _safe_annualised_vol(x: pd.Series, annualisation_days: int) -> float:
    arr = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype="float64")
    if arr.size <= 1:
        return np.nan
    return float(np.std(arr, ddof=1) * np.sqrt(float(annualisation_days)))


def _simple_to_log(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    x = x.clip(lower=-0.999999)
    return np.log1p(x)


def _ewma_vol_snapshot(x: pd.Series, halflife: int, annualisation_days: int) -> float:
    s = pd.to_numeric(x, errors="coerce").dropna()
    if s.shape[0] < 2:
        return np.nan
    ewm_std = s.ewm(halflife=max(int(halflife), 1), adjust=False).std(bias=False)
    last = ewm_std.iloc[-1]
    if pd.isna(last):
        return np.nan
    return float(last * np.sqrt(float(annualisation_days)))


def _rolling_vol_snapshot(x: pd.Series, window: int, annualisation_days: int, min_periods: int) -> float:
    s = pd.to_numeric(x, errors="coerce").dropna()
    if s.shape[0] < max(2, min_periods):
        return np.nan
    last_window = s.iloc[-int(window):]
    if last_window.shape[0] < max(2, min_periods):
        return np.nan
    return float(last_window.std(ddof=1) * np.sqrt(float(annualisation_days)))


def _rolling_compounded_return(x: pd.Series, window: int, min_periods: int) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce").clip(lower=-0.999999)
    log_s = np.log1p(s)

    def _compound(arr: np.ndarray) -> float:
        arr = np.asarray(arr, dtype="float64")
        arr = arr[np.isfinite(arr)]
        if arr.size < max(int(min_periods), 1):
            return np.nan
        return float(np.expm1(arr.sum()))

    return log_s.rolling(window=max(int(window), 1), min_periods=max(int(min_periods), 1)).apply(_compound, raw=True)


def _vol_adjusted_signal(momentum: pd.Series, vol_ann: pd.Series) -> pd.Series:
    m = pd.to_numeric(momentum, errors="coerce")
    v = pd.to_numeric(vol_ann, errors="coerce")
    denom = v.replace(0.0, np.nan)
    out = m / denom
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _monthly_simple_return_from_daily(x: pd.Series) -> float:
    s = pd.to_numeric(x, errors="coerce").dropna().clip(lower=-0.999999)
    if s.empty:
        return np.nan
    return float(np.exp(np.log1p(s).sum()) - 1.0)


def _rolling_efficiency_ratio(x: pd.Series, window: int, min_periods: int) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce").astype("float64")
    net = s.rolling(window=max(int(window), 1), min_periods=max(int(min_periods), 1)).sum().abs()
    gross = s.abs().rolling(window=max(int(window), 1), min_periods=max(int(min_periods), 1)).sum()
    out = net / gross.replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _time_zscore(x: pd.Series, *, min_std_obs: int = 5) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    mu = float(x.mean()) if x.notna().any() else 0.0
    sd = float(x.std(ddof=1)) if int(x.notna().sum()) >= int(min_std_obs) else 0.0
    if np.isfinite(sd) and sd > 1e-12:
        return ((x - mu) / sd).fillna(0.0)
    return (x - mu).fillna(0.0)

def _month_end_from_dates(x: pd.Series) -> pd.Timestamp:
    s = pd.to_datetime(x, errors="coerce").dropna()
    if s.empty:
        return pd.NaT
    return pd.Timestamp(s.max()).to_period("M").to_timestamp("M")


def _week_end_from_dates(x: pd.Series, week_freq: str = "W-FRI") -> pd.Timestamp:
    s = pd.to_datetime(x, errors="coerce").dropna()
    if s.empty:
        return pd.NaT
    return pd.Timestamp(s.max()).to_period(week_freq).end_time.normalize()


def _cross_sectional_zscore(x: pd.Series) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce")
    mu = float(s.mean()) if s.notna().any() else np.nan
    sigma = float(s.std(ddof=1)) if s.notna().sum() > 1 else np.nan
    if not np.isfinite(sigma) or sigma <= 1e-12:
        return pd.Series(np.where(s.notna(), 0.0, np.nan), index=s.index, dtype="float64")
    out = (s - mu) / sigma
    return out.astype("float64")


def _cross_sectional_rank_centered(x: pd.Series) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=s.index, dtype="float64")
    if valid.shape[0] == 1:
        out = pd.Series(np.nan, index=s.index, dtype="float64")
        out.loc[valid.index] = 0.0
        return out
    pct = valid.rank(method="average", pct=True)
    centered = 2.0 * (pct - 0.5)
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    out.loc[centered.index] = centered.astype("float64")
    return out


def add_cross_sectional_features(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    asset_col: str = "asset",
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Add cross-sectional z-scores and centered ranks per date.

    For each feature:
        - feature_cs_z
        - feature_cs_rank

    Computed across assets for each date.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df

    out = df.copy()

    if feature_cols is None:
        exclude = {date_col, asset_col, "return"}
        feature_cols = [
            c for c in out.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(out[c])
        ]

    if not feature_cols:
        return out

    numeric = out[feature_cols].apply(pd.to_numeric, errors="coerce")
    grouped = numeric.groupby(out[date_col])

    means = grouped.transform("mean")
    stds = grouped.transform("std")
    counts = grouped.transform("count")

    z = (numeric - means).div(stds.replace(0.0, np.nan))
    z = z.where(stds.gt(1e-12), 0.0)
    z = z.where(numeric.notna(), np.nan)

    ranks = grouped.rank(method="average", pct=True)
    centered = 2.0 * (ranks - 0.5)
    centered = centered.where(counts.gt(1), 0.0)
    centered = centered.where(numeric.notna(), np.nan)

    for col in feature_cols:
        out[f"{col}_cs_z"] = z[col].astype("float64")
        out[f"{col}_cs_rank"] = centered[col].astype("float64")

    return out

# ============================================================
# Daily intramonth feature engine
# ============================================================


def add_daily_intramonth_state_features(df_daily: pd.DataFrame, cfg: DailyFeatureConfig | None = None) -> pd.DataFrame:
    """
    Add per-day state variables that are later snapshotted at month end.

    This keeps the logic explicit:
    - daily simple return
    - daily log return
    - EWMA vol (5/10/21 style)
    - rolling realised vol (5/10/21 style)
    - rolling daily momentum (5/10/21 style)
    - volatility-adjusted momentum
    - short-term reversal proxy
    - cumulative month-to-date log/simple return
    - additional intramonth signal-state variables:
      efficiency, momentum spreads, semivol asymmetry, breakout/channel state,
      rolling hit-rates and volatility compression

    Output is still daily panel.
    """
    cfg = cfg or DailyFeatureConfig()
    df = validate_daily_asset_panel(df_daily, cfg)

    out_parts: List[pd.DataFrame] = []
    for asset, g in df.groupby(cfg.asset_col, sort=False):
        h = g.copy()
        h["month"] = h[cfg.date_col].dt.to_period("M")
        h["daily_log_return"] = _simple_to_log(h[cfg.return_col])
        h["ret_pos"] = pd.to_numeric(h[cfg.return_col], errors="coerce").clip(lower=0.0)
        h["ret_neg_abs"] = (-pd.to_numeric(h[cfg.return_col], errors="coerce").clip(upper=0.0)).abs()

        for name, halflife in [
            ("ewma_vol_5d_ann", cfg.ewma_halflife_short),
            ("ewma_vol_10d_ann", cfg.ewma_halflife_medium),
            ("ewma_vol_21d_ann", cfg.ewma_halflife_long),
        ]:
            h[name] = (
                h[cfg.return_col]
                .ewm(halflife=max(int(halflife), 1), adjust=False)
                .std(bias=False)
                * np.sqrt(float(cfg.annualisation_days))
            )

        for name, window in [
            ("roll_vol_5d_ann", cfg.rolling_short),
            ("roll_vol_10d_ann", cfg.rolling_medium),
            ("roll_vol_21d_ann", cfg.rolling_long),
        ]:
            h[name] = (
                h[cfg.return_col]
                .rolling(window=max(int(window), 1), min_periods=max(int(cfg.min_obs_for_window), 2))
                .std(ddof=1)
                * np.sqrt(float(cfg.annualisation_days))
            )

        for name, window in [
            ("mom_5d", cfg.momentum_short),
            ("mom_10d", cfg.momentum_medium),
            ("mom_21d", cfg.momentum_long),
        ]:
            h[name] = _rolling_compounded_return(h[cfg.return_col], window=window, min_periods=max(int(cfg.min_obs_for_window), 2))

        h["mom_5d_over_vol_21d"] = _vol_adjusted_signal(h["mom_5d"], h["roll_vol_21d_ann"])
        h["mom_10d_over_vol_21d"] = _vol_adjusted_signal(h["mom_10d"], h["roll_vol_21d_ann"])
        h["mom_21d_over_vol_21d"] = _vol_adjusted_signal(h["mom_21d"], h["roll_vol_21d_ann"])
        h["reversal_5d"] = -h["mom_5d"]

        # Advanced intramonth alpha state variables
        h["alpha_mom_spread_5_21"] = pd.to_numeric(h["mom_5d"], errors="coerce") - pd.to_numeric(h["mom_21d"], errors="coerce")
        h["alpha_mom_spread_10_21"] = pd.to_numeric(h["mom_10d"], errors="coerce") - pd.to_numeric(h["mom_21d"], errors="coerce")
        h["alpha_mom_accel_5_10"] = pd.to_numeric(h["mom_5d"], errors="coerce") - pd.to_numeric(h["mom_10d"], errors="coerce")

        h["alpha_efficiency_10d"] = _rolling_efficiency_ratio(h[cfg.return_col], window=cfg.momentum_medium, min_periods=max(int(cfg.min_obs_for_window), 2))
        h["alpha_efficiency_21d"] = _rolling_efficiency_ratio(h[cfg.return_col], window=cfg.momentum_long, min_periods=max(int(cfg.min_obs_for_window), 2))

        h["alpha_pos_rate_5d"] = (
            (pd.to_numeric(h[cfg.return_col], errors="coerce") > 0).astype(float)
            .rolling(window=max(int(cfg.momentum_short), 1), min_periods=max(int(cfg.min_obs_for_window), 2))
            .mean()
        )
        h["alpha_pos_rate_10d"] = (
            (pd.to_numeric(h[cfg.return_col], errors="coerce") > 0).astype(float)
            .rolling(window=max(int(cfg.momentum_medium), 1), min_periods=max(int(cfg.min_obs_for_window), 2))
            .mean()
        )

        h["alpha_upside_vol_10d_ann"] = (
            h["ret_pos"].rolling(window=max(int(cfg.rolling_medium), 1), min_periods=max(int(cfg.min_obs_for_window), 2)).std(ddof=1)
            * np.sqrt(float(cfg.annualisation_days))
        )
        h["alpha_downside_vol_10d_ann"] = (
            h["ret_neg_abs"].rolling(window=max(int(cfg.rolling_medium), 1), min_periods=max(int(cfg.min_obs_for_window), 2)).std(ddof=1)
            * np.sqrt(float(cfg.annualisation_days))
        )
        h["alpha_upside_vol_21d_ann"] = (
            h["ret_pos"].rolling(window=max(int(cfg.rolling_long), 1), min_periods=max(int(cfg.min_obs_for_window), 2)).std(ddof=1)
            * np.sqrt(float(cfg.annualisation_days))
        )
        h["alpha_downside_vol_21d_ann"] = (
            h["ret_neg_abs"].rolling(window=max(int(cfg.rolling_long), 1), min_periods=max(int(cfg.min_obs_for_window), 2)).std(ddof=1)
            * np.sqrt(float(cfg.annualisation_days))
        )
        h["alpha_down_up_vol_ratio_21d"] = (
            pd.to_numeric(h["alpha_downside_vol_21d_ann"], errors="coerce")
            / pd.to_numeric(h["alpha_upside_vol_21d_ann"], errors="coerce").replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        h["alpha_vol_compression_5v21"] = (
            pd.to_numeric(h["roll_vol_5d_ann"], errors="coerce")
            / pd.to_numeric(h["roll_vol_21d_ann"], errors="coerce").replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)

        h["alpha_price_index"] = np.exp(h["daily_log_return"].cumsum())
        roll_max_21 = h["alpha_price_index"].rolling(window=max(int(cfg.momentum_long), 1), min_periods=max(int(cfg.min_obs_for_window), 2)).max()
        roll_min_21 = h["alpha_price_index"].rolling(window=max(int(cfg.momentum_long), 1), min_periods=max(int(cfg.min_obs_for_window), 2)).min()
        h["alpha_dist_from_21d_high"] = (h["alpha_price_index"] / roll_max_21) - 1.0
        h["alpha_dist_from_21d_low"] = (h["alpha_price_index"] / roll_min_21) - 1.0
        channel_denom = (roll_max_21 - roll_min_21).replace(0.0, np.nan)
        h["alpha_channel_pos_21d"] = ((h["alpha_price_index"] - roll_min_21) / channel_denom).clip(lower=0.0, upper=1.0)

        h["mtd_log_return"] = h.groupby("month")["daily_log_return"].cumsum()
        h["mtd_simple_return"] = np.expm1(h["mtd_log_return"])
        h["mtd_abs_return"] = h.groupby("month")[cfg.return_col].transform(lambda x: pd.to_numeric(x, errors="coerce").abs().cumsum())
        out_parts.append(h)

    out = pd.concat(out_parts, axis=0, ignore_index=True)
    return out.sort_values([cfg.asset_col, cfg.date_col]).reset_index(drop=True)



def build_intramonth_monthly_features(df_daily: pd.DataFrame, cfg: DailyFeatureConfig | None = None) -> pd.DataFrame:
    """
    Aggregate daily data into a monthly asset panel with leakage-safe intramonth features.

    Important leakage note
    ----------------------
    All features for month t are built only from daily observations inside month t.
    They can be used safely to form signals *after month-end* for a t+1 rebalance.
    They should not be interpreted as features known at the start of month t.
    """
    cfg = cfg or DailyFeatureConfig()
    daily = add_daily_intramonth_state_features(df_daily, cfg).copy()
    daily["month"] = daily[cfg.date_col].dt.to_period("M")

    group_cols = [cfg.asset_col, "month"]
    n_obs = daily.groupby(group_cols, sort=True, observed=True).size().rename("n_daily_obs")
    valid_index = n_obs[n_obs >= int(cfg.min_obs_per_month)].index
    if len(valid_index) == 0:
        raise ValueError("Monthly intramonth feature panel is empty after aggregation")

    daily = daily.set_index(group_cols).loc[valid_index].reset_index()
    daily["_ret_num"] = pd.to_numeric(daily[cfg.return_col], errors="coerce")
    daily["_ret_abs"] = daily["_ret_num"].abs()
    daily["_ret_log"] = np.log1p(daily["_ret_num"].clip(lower=cfg.clip_simple_return_low))
    daily["_ret_pos_flag"] = (daily["_ret_num"] > 0).astype(float)
    daily["_ret_neg_flag"] = (daily["_ret_num"] < 0).astype(float)

    grouped = daily.groupby(group_cols, sort=True, observed=True)

    stats = grouped["_ret_num"].agg(["mean", "std", "min", "max", "skew"])
    stats = stats.rename(
        columns={
            "mean": "intramonth_mean_daily",
            "std": "intramonth_std_daily",
            "min": "intramonth_min_daily",
            "max": "intramonth_max_daily",
            "skew": "intramonth_skew_daily",
        }
    )
    stats["intramonth_kurtosis_daily"] = grouped["_ret_num"].agg(pd.Series.kurt)
    stats["intramonth_abs_mean_daily"] = grouped["_ret_abs"].mean()
    stats["intramonth_pos_day_rate"] = grouped["_ret_pos_flag"].mean()
    stats["intramonth_neg_day_rate"] = grouped["_ret_neg_flag"].mean()
    stats["intramonth_range_daily"] = stats["intramonth_max_daily"] - stats["intramonth_min_daily"]

    log_sum = grouped["_ret_log"].sum()
    stats[cfg.monthly_return_name] = np.expm1(log_sum)

    counts = grouped.size().astype("float64")
    std_daily = pd.to_numeric(stats["intramonth_std_daily"], errors="coerce")
    stats["intramonth_realized_vol_ann"] = np.where(
        counts > 1,
        std_daily * np.sqrt(float(cfg.annualisation_days)),
        np.nan,
    )
    stats["intramonth_realized_vol_monthly"] = np.where(
        counts > 1,
        std_daily * np.sqrt(float(cfg.monthly_annualisation_days)),
        np.nan,
    )

    first_dates = grouped[cfg.date_col].first().rename("month_start_date")
    last_dates = grouped[cfg.date_col].last().rename("month_end_date")

    snapshot_cols = [
        "mtd_abs_return",
        "ewma_vol_5d_ann",
        "ewma_vol_10d_ann",
        "ewma_vol_21d_ann",
        "roll_vol_5d_ann",
        "roll_vol_10d_ann",
        "roll_vol_21d_ann",
        "mom_5d",
        "mom_10d",
        "mom_21d",
        "mom_5d_over_vol_21d",
        "mom_10d_over_vol_21d",
        "mom_21d_over_vol_21d",
        "reversal_5d",
        "alpha_mom_spread_5_21",
        "alpha_mom_spread_10_21",
        "alpha_mom_accel_5_10",
        "alpha_efficiency_10d",
        "alpha_efficiency_21d",
        "alpha_pos_rate_5d",
        "alpha_pos_rate_10d",
        "alpha_upside_vol_10d_ann",
        "alpha_downside_vol_10d_ann",
        "alpha_upside_vol_21d_ann",
        "alpha_downside_vol_21d_ann",
        "alpha_down_up_vol_ratio_21d",
        "alpha_vol_compression_5v21",
        "alpha_dist_from_21d_high",
        "alpha_dist_from_21d_low",
        "alpha_channel_pos_21d",
        cfg.return_col,
    ]
    last_snapshot = grouped[snapshot_cols].last().rename(
        columns={
            "mtd_abs_return": "intramonth_cum_abs_return",
            "mom_5d": "mom_5d_eom",
            "mom_10d": "mom_10d_eom",
            "mom_21d": "mom_21d_eom",
            "mom_5d_over_vol_21d": "mom_5d_over_vol_21d_eom",
            "mom_10d_over_vol_21d": "mom_10d_over_vol_21d_eom",
            "mom_21d_over_vol_21d": "mom_21d_over_vol_21d_eom",
            "reversal_5d": "reversal_5d_eom",
            "alpha_mom_spread_5_21": "alpha_mom_spread_5_21_eom",
            "alpha_mom_spread_10_21": "alpha_mom_spread_10_21_eom",
            "alpha_mom_accel_5_10": "alpha_mom_accel_5_10_eom",
            "alpha_efficiency_10d": "alpha_efficiency_10d_eom",
            "alpha_efficiency_21d": "alpha_efficiency_21d_eom",
            "alpha_pos_rate_5d": "alpha_pos_rate_5d_eom",
            "alpha_pos_rate_10d": "alpha_pos_rate_10d_eom",
            "alpha_upside_vol_10d_ann": "alpha_upside_vol_10d_ann_eom",
            "alpha_downside_vol_10d_ann": "alpha_downside_vol_10d_ann_eom",
            "alpha_upside_vol_21d_ann": "alpha_upside_vol_21d_ann_eom",
            "alpha_downside_vol_21d_ann": "alpha_downside_vol_21d_ann_eom",
            "alpha_down_up_vol_ratio_21d": "alpha_down_up_vol_ratio_21d_eom",
            "alpha_vol_compression_5v21": "alpha_vol_compression_5v21_eom",
            "alpha_dist_from_21d_high": "alpha_dist_from_21d_high_eom",
            "alpha_dist_from_21d_low": "alpha_dist_from_21d_low_eom",
            "alpha_channel_pos_21d": "alpha_channel_pos_21d_eom",
            cfg.return_col: "ret_1d_last_m",
        }
    )

    out = pd.concat(
        [
            counts.rename("n_daily_obs"),
            stats,
            last_snapshot,
            first_dates,
            last_dates,
        ],
        axis=1,
    ).reset_index()

    out["date"] = out["month"].dt.to_timestamp("M")
    out = out.drop(columns=["month"])
    out = out.rename(columns={cfg.asset_col: "asset"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["asset"] = out["asset"].astype(str).str.upper().str.strip()
    out = out.sort_values(["date", "asset"]).reset_index(drop=True)

    if out.empty:
        raise ValueError("Monthly intramonth feature panel is empty after aggregation")
    return out

# ============================================================
# Cross-sectional daily aggregates -> monthly features
# ============================================================


def build_cross_sectional_intramonth_features(df_daily: pd.DataFrame, cfg: DailyFeatureConfig | None = None) -> pd.DataFrame:
    """
    Build month-level cross-sectional diagnostics from daily returns and momentum states.

    This is the daily-dispersion layer that sits upstream from the allocator:
    - cross-asset daily return dispersion
    - mean absolute cross-sectional movement
    - cross-sectional range / IQR
    - fraction of assets up / down on a given day
    - cross-sectional momentum dispersion / strength
    - average pairwise correlation of daily asset returns within month

    Output is month-level only (one row per month).
    """
    cfg = cfg or DailyFeatureConfig()
    daily = add_daily_intramonth_state_features(df_daily, cfg).copy()
    daily["month"] = daily[cfg.date_col].dt.to_period("M")

    value_cols = [
        cfg.return_col,
        "mom_5d",
        "mom_10d",
        "mom_21d",
        "mom_5d_over_vol_21d",
        "mom_10d_over_vol_21d",
        "mom_21d_over_vol_21d",
        "reversal_5d",
    ]
    wide_map: Dict[str, pd.DataFrame] = {}
    for col in value_cols:
        wide_map[col] = (
            daily.pivot_table(index=cfg.date_col, columns=cfg.asset_col, values=col, aggfunc="first")
            .sort_index()
        )

    wide = wide_map[cfg.return_col]
    if wide.empty:
        raise ValueError("Could not pivot daily asset panel into wide form")

    def _row_iqr(df_wide: pd.DataFrame) -> pd.Series:
        q75 = df_wide.quantile(0.75, axis=1, interpolation="linear")
        q25 = df_wide.quantile(0.25, axis=1, interpolation="linear")
        return q75 - q25

    def _summarise_daily_series(prefix: str, s: pd.Series) -> Dict[str, float]:
        s = pd.to_numeric(s, errors="coerce")
        return {
            f"{prefix}_mean_m": float(s.mean()) if not s.empty else np.nan,
            f"{prefix}_max_m": float(s.max()) if not s.empty else np.nan,
            f"{prefix}_last_m": float(s.iloc[-1]) if not s.empty else np.nan,
        }

    month_rows: List[Dict[str, object]] = []
    month_index = pd.Series(wide.index, index=wide.index).dt.to_period("M")
    for month, daily_block in wide.groupby(month_index):
        if daily_block.empty:
            continue

        daily_disp = daily_block.std(axis=1, ddof=1)
        daily_mean_abs = daily_block.abs().mean(axis=1)
        daily_range = daily_block.max(axis=1) - daily_block.min(axis=1)
        daily_iqr = _row_iqr(daily_block)
        daily_up_frac = (daily_block > 0).mean(axis=1)
        daily_down_frac = (daily_block < 0).mean(axis=1)
        daily_n_assets = daily_block.notna().sum(axis=1)

        corr = daily_block.corr(min_periods=max(int(cfg.min_obs_for_window), 2))
        avg_pairwise_corr = np.nan
        if isinstance(corr, pd.DataFrame) and corr.shape[0] >= 2:
            tri = corr.where(~np.eye(corr.shape[0], dtype=bool))
            vals = tri.stack().astype(float)
            if not vals.empty:
                avg_pairwise_corr = float(vals.mean())

        row: Dict[str, object] = {
            "date": pd.Period(month, freq="M").to_timestamp("M"),
            "cs_avg_pairwise_corr": avg_pairwise_corr,
            "cs_n_assets_mean": float(daily_n_assets.mean()) if not daily_n_assets.empty else np.nan,
            "cs_n_assets_min": float(daily_n_assets.min()) if not daily_n_assets.empty else np.nan,
            "cs_n_assets_last": float(daily_n_assets.iloc[-1]) if not daily_n_assets.empty else np.nan,
        }
        row.update(_summarise_daily_series("cs_ret_dispersion", daily_disp))
        row.update(_summarise_daily_series("cs_ret_mean_abs", daily_mean_abs))
        row.update(_summarise_daily_series("cs_ret_range", daily_range))
        row.update(_summarise_daily_series("cs_ret_iqr", daily_iqr))
        row.update(_summarise_daily_series("cs_up_frac", daily_up_frac))
        row.update(_summarise_daily_series("cs_down_frac", daily_down_frac))

        for source_col, prefix in [
            ("mom_5d", "cs_mom_5d"),
            ("mom_10d", "cs_mom_10d"),
            ("mom_21d", "cs_mom_21d"),
            ("mom_5d_over_vol_21d", "cs_mom_5d_voladj"),
            ("mom_10d_over_vol_21d", "cs_mom_10d_voladj"),
            ("mom_21d_over_vol_21d", "cs_mom_21d_voladj"),
            ("reversal_5d", "cs_reversal_5d"),
        ]:
            wide_feat = wide_map[source_col]
            feat_block = wide_feat.loc[wide_feat.index.to_period("M") == month]
            if feat_block.empty:
                continue
            feat_std = feat_block.std(axis=1, ddof=1)
            feat_mean_abs = feat_block.abs().mean(axis=1)
            row.update(_summarise_daily_series(f"{prefix}_dispersion", feat_std))
            row.update(_summarise_daily_series(f"{prefix}_mean_abs", feat_mean_abs))

        row["cs_avg_daily_dispersion"] = row.get("cs_ret_dispersion_mean_m", np.nan)
        row["cs_max_daily_dispersion"] = row.get("cs_ret_dispersion_max_m", np.nan)

        month_rows.append(row)

    out = pd.DataFrame(month_rows)
    if out.empty:
        raise ValueError("Cross-sectional intramonth feature panel is empty")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.sort_values("date").reset_index(drop=True)


def add_asset_relative_cross_sectional_monthly_features(df_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-asset cross-sectional monthly features so investment.py can tilt mu.

    These are *cross-sectional* transforms across assets within the same month-end:
    - z-scores of momentum snapshots
    - z-scores of volatility-adjusted momentum snapshots
    - rank/z-score of last daily return in month
    - composite asset-vs-cross-section strength signal
    """
    if df_monthly is None or not isinstance(df_monthly, pd.DataFrame) or df_monthly.empty:
        raise ValueError("df_monthly must be a non-empty pandas DataFrame")
    req = {"date", "asset"}
    missing = sorted(req - set(df_monthly.columns))
    if missing:
        raise ValueError(f"Monthly panel is missing required columns for cross-sectional feature engineering: {missing}")

    out = df_monthly.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    base_cols = [
        "mom_5d_eom",
        "mom_10d_eom",
        "mom_21d_eom",
        "mom_5d_over_vol_21d_eom",
        "mom_10d_over_vol_21d_eom",
        "mom_21d_over_vol_21d_eom",
        "reversal_5d_eom",
        "alpha_mom_spread_5_21_eom",
        "alpha_mom_spread_10_21_eom",
        "alpha_mom_accel_5_10_eom",
        "alpha_efficiency_10d_eom",
        "alpha_efficiency_21d_eom",
        "alpha_pos_rate_5d_eom",
        "alpha_pos_rate_10d_eom",
        "alpha_down_up_vol_ratio_21d_eom",
        "alpha_vol_compression_5v21_eom",
        "alpha_dist_from_21d_high_eom",
        "alpha_dist_from_21d_low_eom",
        "alpha_channel_pos_21d_eom",
    ]
    for col in base_cols:
        if col in out.columns:
            z_name = col.replace("_eom", "_cs_z_eom")
            rank_name = col.replace("_eom", "_cs_rank_eom")
            out[z_name] = out.groupby("date", group_keys=False)[col].apply(_cross_sectional_zscore)
            out[rank_name] = out.groupby("date", group_keys=False)[col].apply(_cross_sectional_rank_centered)

    if "ret_1d_last_m" in out.columns:
        out["ret_1d_cs_z_last_m"] = out.groupby("date", group_keys=False)["ret_1d_last_m"].apply(_cross_sectional_zscore)
        out["ret_1d_cs_rank_last_m"] = out.groupby("date", group_keys=False)["ret_1d_last_m"].apply(_cross_sectional_rank_centered)

    strength_cols = [
        "mom_21d_cs_z_eom",
        "mom_10d_cs_z_eom",
        "mom_5d_cs_z_eom",
        "mom_21d_over_vol_21d_cs_z_eom",
        "mom_10d_over_vol_21d_cs_z_eom",
        "mom_5d_over_vol_21d_cs_z_eom",
        "alpha_mom_spread_5_21_cs_z_eom",
        "alpha_mom_spread_10_21_cs_z_eom",
        "alpha_mom_accel_5_10_cs_z_eom",
        "alpha_efficiency_21d_cs_z_eom",
        "alpha_channel_pos_21d_cs_z_eom",
        "alpha_pos_rate_10d_cs_z_eom",
        "ret_1d_cs_z_last_m",
    ]
    available_strength_cols = [c for c in strength_cols if c in out.columns]
    if available_strength_cols:
        strength_df = out[available_strength_cols].apply(pd.to_numeric, errors="coerce")
        out["asset_vs_cross_section_strength_eom"] = strength_df.mean(axis=1)
        out["asset_vs_cross_section_abs_strength_eom"] = strength_df.abs().mean(axis=1)
    else:
        out["asset_vs_cross_section_strength_eom"] = np.nan
        out["asset_vs_cross_section_abs_strength_eom"] = np.nan

    return out.sort_values(["date", "asset"]).reset_index(drop=True)






def _safe_weighted_row_mean(df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    if df is None or df.empty or not weights:
        return pd.Series(np.nan, index=df.index if isinstance(df, pd.DataFrame) else None, dtype="float64")
    parts = []
    total_w = 0.0
    for col, w in weights.items():
        if col not in df.columns:
            continue
        ww = float(w)
        if not np.isfinite(ww) or abs(ww) <= 1e-12:
            continue
        parts.append(pd.to_numeric(df[col], errors="coerce") * ww)
        total_w += abs(ww)
    if not parts or total_w <= 1e-12:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    stacked = pd.concat(parts, axis=1)
    return stacked.sum(axis=1) / float(total_w)


def add_feature_based_mu_features(df_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature-based expected-return proxy columns for the engine.

    These are heuristic signal features, not forecasts. They combine available cross-sectional momentum, volatility-adjusted momentum, reversal and quality signals into smoothed per-asset columns that the Strategy Engine may use as additional context.
    """
    if df_panel is None or not isinstance(df_panel, pd.DataFrame) or df_panel.empty:
        raise ValueError("df_panel must be a non-empty pandas DataFrame")
    req = {"date", "asset"}
    missing = sorted(req - set(df_panel.columns))
    if missing:
        raise ValueError(f"Panel is missing required columns for mu-feature engineering: {missing}")

    out = df_panel.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["asset"] = out["asset"].astype(str).str.upper().str.strip()
    out = out.sort_values(["asset", "date"]).reset_index(drop=True)

    positive_weights: Dict[str, float] = {
        "mom_21d_over_vol_21d_cs_z_eom": 1.50,
        "mom_10d_over_vol_21d_cs_z_eom": 1.10,
        "mom_5d_over_vol_21d_cs_z_eom": 0.70,
        "mom_21d_cs_z_eom": 1.25,
        "mom_10d_cs_z_eom": 0.90,
        "mom_5d_cs_z_eom": 0.60,
        "alpha_mom_spread_5_21_cs_z_eom": 0.95,
        "alpha_mom_spread_10_21_cs_z_eom": 0.75,
        "alpha_mom_accel_5_10_cs_z_eom": 0.45,
        "alpha_efficiency_21d_cs_z_eom": 0.85,
        "alpha_channel_pos_21d_cs_z_eom": 0.55,
        "alpha_pos_rate_10d_cs_z_eom": 0.40,
        "asset_vs_cross_section_strength_eom": 1.00,
        "ret_1d_cs_z_last_m": 0.25,
    }
    negative_weights: Dict[str, float] = {
        "reversal_5d_cs_z_eom": 0.60,
        "alpha_down_up_vol_ratio_21d_cs_z_eom": 0.40,
        "alpha_dist_from_21d_high_cs_z_eom": 0.15,
    }

    available_cols = [c for c in list(positive_weights) + list(negative_weights) if c in out.columns]
    if not available_cols:
        for c in [
            "mu_feature_base_raw","mu_feature_smooth_3","mu_feature_smooth_6","mu_feature_composite_raw",
            "mu_feature_composite_cs_z","mu_feature_composite_cs_rank","mu_feature_positive_block",
            "mu_feature_negative_block","mu_feature_net_strength","mu_feature_quality_count",
            "mu_feature_abs_strength","mu_feature_time_z"
        ]:
            out[c] = np.nan
        out["mu_feature_quality_count"] = 0.0
        return out

    pos_df = out[[c for c in positive_weights if c in out.columns]].copy() if any(c in out.columns for c in positive_weights) else pd.DataFrame(index=out.index)
    neg_df = out[[c for c in negative_weights if c in out.columns]].copy() if any(c in out.columns for c in negative_weights) else pd.DataFrame(index=out.index)

    out["mu_feature_positive_block"] = _safe_weighted_row_mean(pos_df, {c: positive_weights[c] for c in pos_df.columns}) if not pos_df.empty else np.nan
    out["mu_feature_negative_block"] = _safe_weighted_row_mean(neg_df, {c: negative_weights[c] for c in neg_df.columns}) if not neg_df.empty else np.nan

    pos = pd.to_numeric(out["mu_feature_positive_block"], errors="coerce")
    neg = pd.to_numeric(out["mu_feature_negative_block"], errors="coerce")
    out["mu_feature_base_raw"] = pos - neg.fillna(0.0)
    out["mu_feature_quality_count"] = out[available_cols].notna().sum(axis=1).astype(float)

    g = out.groupby("asset", sort=False)["mu_feature_base_raw"]
    out["mu_feature_smooth_3"] = g.transform(lambda s: pd.to_numeric(s, errors="coerce").rolling(3, min_periods=1).mean())
    out["mu_feature_smooth_6"] = g.transform(lambda s: pd.to_numeric(s, errors="coerce").rolling(6, min_periods=2).mean())

    quality_penalty = 1.0 - 0.10 * (out["mu_feature_quality_count"] < 3).astype(float)
    out["mu_feature_composite_raw"] = (
        0.50 * pd.to_numeric(out["mu_feature_base_raw"], errors="coerce").fillna(0.0)
        + 0.30 * pd.to_numeric(out["mu_feature_smooth_3"], errors="coerce").fillna(0.0)
        + 0.20 * pd.to_numeric(out["mu_feature_smooth_6"], errors="coerce").fillna(0.0)
    ) * quality_penalty

    out["mu_feature_time_z"] = out.groupby("asset", group_keys=False)["mu_feature_composite_raw"].apply(_time_zscore)
    out["mu_feature_composite_cs_z"] = out.groupby("date", group_keys=False)["mu_feature_composite_raw"].apply(_cross_sectional_zscore)
    out["mu_feature_composite_cs_rank"] = out.groupby("date", group_keys=False)["mu_feature_composite_raw"].apply(_cross_sectional_rank_centered)

    z = pd.to_numeric(out["mu_feature_composite_cs_z"], errors="coerce")
    r = pd.to_numeric(out["mu_feature_composite_cs_rank"], errors="coerce")
    tz = pd.to_numeric(out["mu_feature_time_z"], errors="coerce")
    out["mu_feature_net_strength"] = pd.concat([z, r, 0.5 * tz], axis=1).mean(axis=1)
    out["mu_feature_abs_strength"] = pd.to_numeric(out["mu_feature_net_strength"], errors="coerce").abs()

    return out.sort_values(["date", "asset"]).reset_index(drop=True)


def add_conditional_rebalance_context_features(df_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Add precomputed date-level stress scores / flags for the conditional monthly
    rebalance overlay used in investment.py.

    These features are not strictly necessary for the overlay to work, but they
    make the signal construction cleaner and more research-friendly by moving the
    repetitive feature-combination logic upstream into the monthly panel builder.
    """
    if df_monthly is None or not isinstance(df_monthly, pd.DataFrame) or df_monthly.empty:
        raise ValueError("df_monthly must be a non-empty pandas DataFrame")
    req = {"date", "asset"}
    missing = sorted(req - set(df_monthly.columns))
    if missing:
        raise ValueError(f"Monthly panel is missing required columns for conditional rebalance context features: {missing}")

    out = df_monthly.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    # Build date-level aggregates from whatever columns are available.
    by_date = out.groupby("date", sort=True)
    def _gmean(col: str) -> pd.Series:
        if col in out.columns:
            return by_date[col].mean()
        return pd.Series(np.nan, index=by_date.size().index, dtype="float64")
    agg = pd.DataFrame({
        "date": by_date.size().index,
        "vol_level": _gmean("intramonth_realized_vol_monthly").to_numpy(),
        "vol_ewma": _gmean("ewma_vol_21d_ann").to_numpy(),
        "vol_roll": _gmean("roll_vol_21d_ann").to_numpy(),
        "vol_abs_move": _gmean("intramonth_cum_abs_return").to_numpy(),
        "corr_level": _gmean("cs_avg_pairwise_corr").to_numpy(),
        "disp_level": _gmean("cs_ret_dispersion_last_m").to_numpy(),
        "disp_mean": _gmean("cs_ret_dispersion_mean_m").to_numpy(),
        "mom_strength": _gmean("asset_vs_cross_section_abs_strength_eom").to_numpy(),
        "mom_level": _gmean("mom_21d_over_vol_21d_eom").to_numpy(),
        "pos_day_rate": _gmean("intramonth_pos_day_rate").to_numpy(),
        "neg_day_rate": _gmean("intramonth_neg_day_rate").to_numpy(),
    })

    # Composite raw context metrics.
    agg["conditional_rebalance_vol_raw"] = agg[["vol_level", "vol_ewma", "vol_roll", "vol_abs_move"]].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    agg["conditional_rebalance_corr_raw"] = pd.to_numeric(agg["corr_level"], errors="coerce")
    agg["conditional_rebalance_dispersion_raw"] = agg[["disp_level", "disp_mean"]].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    agg["conditional_rebalance_momentum_raw"] = agg[["mom_strength", "mom_level"]].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    # Standardised badness scores. Higher is worse.
    agg["conditional_rebalance_vol_stress_score"] = _time_zscore(agg["conditional_rebalance_vol_raw"])
    agg["conditional_rebalance_corr_stress_score"] = _time_zscore(agg["conditional_rebalance_corr_raw"])
    agg["conditional_rebalance_dispersion_collapse_score"] = -_time_zscore(agg["conditional_rebalance_dispersion_raw"])
    agg["conditional_rebalance_momentum_break_score"] = -_time_zscore(agg["conditional_rebalance_momentum_raw"])

    # Binary flags using moderately conservative defaults aligned with investment.py.
    agg["conditional_rebalance_vol_spike_flag"] = (agg["conditional_rebalance_vol_stress_score"] >= 0.75).astype(float)
    agg["conditional_rebalance_corr_stress_flag"] = (agg["conditional_rebalance_corr_stress_score"] >= 0.75).astype(float)
    agg["conditional_rebalance_dispersion_collapse_flag"] = (agg["conditional_rebalance_dispersion_collapse_score"] >= 0.50).astype(float)
    agg["conditional_rebalance_momentum_break_flag"] = (agg["conditional_rebalance_momentum_break_score"] >= 0.50).astype(float)
    flag_cols = [
        "conditional_rebalance_vol_spike_flag",
        "conditional_rebalance_corr_stress_flag",
        "conditional_rebalance_dispersion_collapse_flag",
        "conditional_rebalance_momentum_break_flag",
    ]
    score_cols = [
        "conditional_rebalance_vol_stress_score",
        "conditional_rebalance_corr_stress_score",
        "conditional_rebalance_dispersion_collapse_score",
        "conditional_rebalance_momentum_break_score",
    ]
    agg["conditional_rebalance_trigger_count_feature"] = agg[flag_cols].sum(axis=1)
    agg["conditional_rebalance_stress_score_feature"] = agg[score_cols].clip(lower=0.0).mean(axis=1)

    keep = [
        "date",
        "conditional_rebalance_vol_raw",
        "conditional_rebalance_corr_raw",
        "conditional_rebalance_dispersion_raw",
        "conditional_rebalance_momentum_raw",
        *score_cols,
        *flag_cols,
        "conditional_rebalance_trigger_count_feature",
        "conditional_rebalance_stress_score_feature",
    ]
    return out.merge(agg[keep], on="date", how="left").sort_values(["date", "asset"]).reset_index(drop=True)




# ============================================================
# Weekly panel builder
# ============================================================


def build_intraweek_weekly_features(
    df_daily: pd.DataFrame,
    cfg: DailyFeatureConfig | None = None,
) -> pd.DataFrame:
    """Aggregate daily data into a weekly asset panel with end-of-week snapshots.

    The output intentionally keeps many column names aligned with the monthly
    panel so the current micro-pipeline can reuse the same feature preferences.
    """
    cfg = cfg or DailyFeatureConfig()
    daily = add_daily_intramonth_state_features(df_daily, cfg)
    daily["week"] = daily[cfg.date_col].dt.to_period("W-FRI")

    rows: List[Dict[str, object]] = []
    for (asset, week), g in daily.groupby([cfg.asset_col, "week"], sort=True):
        g = g.sort_values(cfg.date_col).copy()
        n_obs = int(g.shape[0])
        if n_obs < max(2, int(cfg.min_obs_for_window)):
            continue

        rets = pd.to_numeric(g[cfg.return_col], errors="coerce").dropna()
        if rets.empty:
            continue

        row: Dict[str, object] = {
            "date": _week_end_from_dates(g[cfg.date_col]),
            "asset": str(asset),
            cfg.weekly_return_name: _monthly_simple_return_from_daily(rets),
            "n_daily_obs": n_obs,
            "intramonth_mean_daily": float(rets.mean()),
            "intramonth_std_daily": float(rets.std(ddof=1)) if rets.shape[0] > 1 else np.nan,
            "intramonth_realized_vol_ann": _safe_annualised_vol(rets, cfg.annualisation_days),
            "intramonth_realized_vol_monthly": _safe_annualised_vol(rets, min(int(cfg.monthly_annualisation_days), max(n_obs, 1))),
            "intramonth_abs_mean_daily": float(rets.abs().mean()),
            "intramonth_min_daily": float(rets.min()),
            "intramonth_max_daily": float(rets.max()),
            "intramonth_range_daily": float(rets.max() - rets.min()),
            "intramonth_skew_daily": float(rets.skew()) if rets.shape[0] > 2 else np.nan,
            "intramonth_kurtosis_daily": float(rets.kurt()) if rets.shape[0] > 3 else np.nan,
            "intramonth_pos_day_rate": float((rets > 0).mean()),
            "intramonth_neg_day_rate": float((rets < 0).mean()),
            "intramonth_cum_abs_return": float(np.abs(rets).sum()),
            "ewma_vol_5d_ann": float(pd.to_numeric(g["ewma_vol_5d_ann"], errors="coerce").iloc[-1]),
            "ewma_vol_10d_ann": float(pd.to_numeric(g["ewma_vol_10d_ann"], errors="coerce").iloc[-1]),
            "ewma_vol_21d_ann": float(pd.to_numeric(g["ewma_vol_21d_ann"], errors="coerce").iloc[-1]),
            "roll_vol_5d_ann": float(pd.to_numeric(g["roll_vol_5d_ann"], errors="coerce").iloc[-1]),
            "roll_vol_10d_ann": float(pd.to_numeric(g["roll_vol_10d_ann"], errors="coerce").iloc[-1]),
            "roll_vol_21d_ann": float(pd.to_numeric(g["roll_vol_21d_ann"], errors="coerce").iloc[-1]),
            "mom_5d_eom": float(pd.to_numeric(g["mom_5d"], errors="coerce").iloc[-1]),
            "mom_10d_eom": float(pd.to_numeric(g["mom_10d"], errors="coerce").iloc[-1]),
            "mom_21d_eom": float(pd.to_numeric(g["mom_21d"], errors="coerce").iloc[-1]),
            "mom_5d_over_vol_21d_eom": float(pd.to_numeric(g["mom_5d_over_vol_21d"], errors="coerce").iloc[-1]),
            "mom_10d_over_vol_21d_eom": float(pd.to_numeric(g["mom_10d_over_vol_21d"], errors="coerce").iloc[-1]),
            "mom_21d_over_vol_21d_eom": float(pd.to_numeric(g["mom_21d_over_vol_21d"], errors="coerce").iloc[-1]),
            "reversal_5d_eom": float(pd.to_numeric(g["reversal_5d"], errors="coerce").iloc[-1]),
            "alpha_mom_spread_5_21_eom": float(pd.to_numeric(g["alpha_mom_spread_5_21"], errors="coerce").iloc[-1]),
            "alpha_mom_spread_10_21_eom": float(pd.to_numeric(g["alpha_mom_spread_10_21"], errors="coerce").iloc[-1]),
            "alpha_mom_accel_5_10_eom": float(pd.to_numeric(g["alpha_mom_accel_5_10"], errors="coerce").iloc[-1]),
            "alpha_efficiency_10d_eom": float(pd.to_numeric(g["alpha_efficiency_10d"], errors="coerce").iloc[-1]),
            "alpha_efficiency_21d_eom": float(pd.to_numeric(g["alpha_efficiency_21d"], errors="coerce").iloc[-1]),
            "alpha_pos_rate_5d_eom": float(pd.to_numeric(g["alpha_pos_rate_5d"], errors="coerce").iloc[-1]),
            "alpha_pos_rate_10d_eom": float(pd.to_numeric(g["alpha_pos_rate_10d"], errors="coerce").iloc[-1]),
            "alpha_upside_vol_10d_ann_eom": float(pd.to_numeric(g["alpha_upside_vol_10d_ann"], errors="coerce").iloc[-1]),
            "alpha_downside_vol_10d_ann_eom": float(pd.to_numeric(g["alpha_downside_vol_10d_ann"], errors="coerce").iloc[-1]),
            "alpha_upside_vol_21d_ann_eom": float(pd.to_numeric(g["alpha_upside_vol_21d_ann"], errors="coerce").iloc[-1]),
            "alpha_downside_vol_21d_ann_eom": float(pd.to_numeric(g["alpha_downside_vol_21d_ann"], errors="coerce").iloc[-1]),
            "alpha_down_up_vol_ratio_21d_eom": float(pd.to_numeric(g["alpha_down_up_vol_ratio_21d"], errors="coerce").iloc[-1]),
            "alpha_vol_compression_5v21_eom": float(pd.to_numeric(g["alpha_vol_compression_5v21"], errors="coerce").iloc[-1]),
            "alpha_dist_from_21d_high_eom": float(pd.to_numeric(g["alpha_dist_from_21d_high"], errors="coerce").iloc[-1]),
            "alpha_dist_from_21d_low_eom": float(pd.to_numeric(g["alpha_dist_from_21d_low"], errors="coerce").iloc[-1]),
            "alpha_channel_pos_21d_eom": float(pd.to_numeric(g["alpha_channel_pos_21d"], errors="coerce").iloc[-1]),
            "ret_1d_last_m": float(pd.to_numeric(g[cfg.return_col], errors="coerce").iloc[-1]),
            "week_start_date": pd.to_datetime(g[cfg.date_col].iloc[0]),
            "week_end_date": pd.to_datetime(g[cfg.date_col].iloc[-1]),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("Weekly feature panel is empty after aggregation")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["asset"] = out["asset"].astype(str).str.upper().str.strip()
    return out.sort_values(["date", "asset"]).reset_index(drop=True)

def build_intrAweek_weekly_features(
    df_daily: pd.DataFrame,
    cfg: DailyFeatureConfig | None = None,
) -> pd.DataFrame:
    """Backward-compatible alias for the original misspelled helper name."""
    return build_intraweek_weekly_features(df_daily, cfg)

def build_cross_sectional_intraweek_features(df_daily: pd.DataFrame, cfg: DailyFeatureConfig | None = None) -> pd.DataFrame:
    """Build week-level cross-sectional diagnostics from daily returns and momentum states."""
    cfg = cfg or DailyFeatureConfig()
    daily = add_daily_intramonth_state_features(df_daily, cfg).copy()
    daily["week"] = daily[cfg.date_col].dt.to_period("W-FRI")

    value_cols = [
        cfg.return_col,
        "mom_5d",
        "mom_10d",
        "mom_21d",
        "mom_5d_over_vol_21d",
        "mom_10d_over_vol_21d",
        "mom_21d_over_vol_21d",
        "reversal_5d",
    ]
    wide_map: Dict[str, pd.DataFrame] = {}
    for col in value_cols:
        wide_map[col] = daily.pivot_table(index=cfg.date_col, columns=cfg.asset_col, values=col, aggfunc="first").sort_index()

    wide = wide_map[cfg.return_col]
    if wide.empty:
        raise ValueError("Could not pivot daily asset panel into wide weekly form")

    def _row_iqr(df_wide: pd.DataFrame) -> pd.Series:
        q75 = df_wide.quantile(0.75, axis=1, interpolation="linear")
        q25 = df_wide.quantile(0.25, axis=1, interpolation="linear")
        return q75 - q25

    def _summarise(prefix: str, s: pd.Series) -> Dict[str, float]:
        s = pd.to_numeric(s, errors="coerce")
        return {
            f"{prefix}_mean_m": float(s.mean()) if not s.empty else np.nan,
            f"{prefix}_max_m": float(s.max()) if not s.empty else np.nan,
            f"{prefix}_last_m": float(s.iloc[-1]) if not s.empty else np.nan,
        }

    rows: List[Dict[str, object]] = []
    week_index = pd.Series(wide.index, index=wide.index).dt.to_period("W-FRI")
    for week, block in wide.groupby(week_index):
        if block.empty:
            continue
        daily_disp = block.std(axis=1, ddof=1)
        daily_mean_abs = block.abs().mean(axis=1)
        daily_range = block.max(axis=1) - block.min(axis=1)
        daily_iqr = _row_iqr(block)
        daily_up_frac = (block > 0).mean(axis=1)
        daily_down_frac = (block < 0).mean(axis=1)
        daily_n_assets = block.notna().sum(axis=1)

        corr = block.corr(min_periods=max(int(cfg.min_obs_for_window), 2))
        avg_pairwise_corr = np.nan
        if isinstance(corr, pd.DataFrame) and corr.shape[0] >= 2:
            tri = corr.where(~np.eye(corr.shape[0], dtype=bool))
            vals = tri.stack().astype(float)
            if not vals.empty:
                avg_pairwise_corr = float(vals.mean())

        row: Dict[str, object] = {
            "date": pd.Period(week, freq="W-FRI").end_time.normalize(),
            "cs_avg_pairwise_corr": avg_pairwise_corr,
            "cs_n_assets_mean": float(daily_n_assets.mean()) if not daily_n_assets.empty else np.nan,
            "cs_n_assets_min": float(daily_n_assets.min()) if not daily_n_assets.empty else np.nan,
            "cs_n_assets_last": float(daily_n_assets.iloc[-1]) if not daily_n_assets.empty else np.nan,
        }
        row.update(_summarise("cs_ret_dispersion", daily_disp))
        row.update(_summarise("cs_ret_mean_abs", daily_mean_abs))
        row.update(_summarise("cs_ret_range", daily_range))
        row.update(_summarise("cs_ret_iqr", daily_iqr))
        row.update(_summarise("cs_up_frac", daily_up_frac))
        row.update(_summarise("cs_down_frac", daily_down_frac))
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("Cross-sectional intraweek feature panel is empty")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.sort_values("date").reset_index(drop=True)


def build_weekly_panel_from_daily(
    df_daily: pd.DataFrame,
    cfg: DailyFeatureConfig | None = None,
    *,
    include_cross_sectional: bool = True,
    macro_panel_df: Optional[pd.DataFrame] = None,
    out_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Build a weekly asset panel compatible with the current micro-pipeline."""
    cfg = cfg or DailyFeatureConfig()
    weekly_asset = build_intraweek_weekly_features(df_daily, cfg)
    if include_cross_sectional:
        cs = build_cross_sectional_intraweek_features(df_daily, cfg)
        weekly_asset = weekly_asset.merge(cs, on="date", how="left")
    weekly_asset = add_asset_relative_cross_sectional_monthly_features(weekly_asset)
    weekly_asset = add_conditional_rebalance_context_features(weekly_asset)
    weekly_asset = merge_macro_context_into_asset_panel(weekly_asset, macro_panel_df)
    weekly_asset = add_feature_based_mu_features(weekly_asset)
    weekly_asset = weekly_asset.sort_values(["date", "asset"]).reset_index(drop=True)
    if out_csv is not None:
        weekly_asset.to_csv(out_csv, index=False)
    return weekly_asset

# ============================================================
# Final monthly panel builder
# ============================================================

def build_monthly_panel_from_daily(
    df_daily: pd.DataFrame,
    cfg: DailyFeatureConfig | None = None,
    *,
    include_cross_sectional: bool = True,
    macro_panel_df: Optional[pd.DataFrame] = None,
    out_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Build a monthly asset panel compatible with the Strategy Engine.

    The returned panel always contains:
    - ``date``;
    - ``asset``;
    - ``return``.

    Additional engineered columns are retained for feature-aware configurations,
    diagnostics, and future research-style extensions. The monthly ``return`` is
    computed from daily simple returns within each month.
    """


    cfg = cfg or DailyFeatureConfig()
    monthly_asset = build_intramonth_monthly_features(df_daily, cfg)

    if include_cross_sectional:
        cs = build_cross_sectional_intramonth_features(df_daily, cfg)
        monthly_asset = monthly_asset.merge(cs, on="date", how="left")

    monthly_asset = add_asset_relative_cross_sectional_monthly_features(monthly_asset)
    monthly_asset = add_conditional_rebalance_context_features(monthly_asset)
    monthly_asset = merge_macro_context_into_asset_panel(monthly_asset, macro_panel_df)
    monthly_asset = add_feature_based_mu_features(monthly_asset)
    monthly_asset = monthly_asset.sort_values(["date", "asset"]).reset_index(drop=True)

    if out_csv is not None:
        monthly_asset.to_csv(out_csv, index=False)

    return monthly_asset


# ============================================================
# Convenience selectors
# ============================================================


def select_micro_pipeline_columns(df_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Return only the minimum columns required by the current micro-pipeline.

    Useful when you want:
    - one rich monthly panel for research
    - one minimal panel for the current production micro-pipeline
    """
    if df_monthly is None or not isinstance(df_monthly, pd.DataFrame) or df_monthly.empty:
        raise ValueError("df_monthly must be a non-empty pandas DataFrame")

    need = ["date", "asset", "return"]
    missing = [c for c in need if c not in df_monthly.columns]
    if missing:
        raise ValueError(f"Monthly panel is missing required columns: {missing}")

    out = df_monthly[need].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["asset"] = out["asset"].astype(str).str.upper().str.strip()
    out["return"] = pd.to_numeric(out["return"], errors="coerce")
    return out.dropna(subset=need).sort_values(["date", "asset"]).reset_index(drop=True)


# ============================================================
# Internal smoke-test helpers
# ============================================================


def _build_toy_daily_panel() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    dates = pd.bdate_range("2019-01-01", periods=700)
    assets = ["SPY", "QQQ", "TLT", "GLD"]

    rows: List[Dict[str, object]] = []
    for d in dates:
        market = rng.normal(0.0002, 0.008)
        for asset in assets:
            beta_scale = {
                "SPY": 1.00,
                "QQQ": 1.20,
                "TLT": 0.45,
                "GLD": 0.35,
            }[asset]
            alpha = rng.normal(0.0, 0.0015)
            noise = rng.normal(0.0, 0.0065)
            r = beta_scale * market + alpha + noise
            rows.append({"date": d, "asset": asset, "return": r})
    return pd.DataFrame(rows)



def _smoke_test() -> None:
    daily = _build_toy_daily_panel()
    monthly = build_monthly_panel_from_daily(daily)
    weekly = build_weekly_panel_from_daily(daily)
    assert not monthly.empty
    assert not weekly.empty
    assert {"date", "asset", "return"}.issubset(monthly.columns)
    assert {"intramonth_realized_vol_ann", "ewma_vol_21d_ann", "cs_avg_pairwise_corr", "cs_ret_dispersion_mean_m", "cs_mom_21d_voladj_dispersion_mean_m", "mom_5d_eom", "mom_21d_over_vol_21d_eom", "reversal_5d_eom", "mom_21d_cs_z_eom", "mom_21d_over_vol_21d_cs_z_eom", "ret_1d_cs_z_last_m", "asset_vs_cross_section_strength_eom", "conditional_rebalance_vol_stress_score", "conditional_rebalance_corr_stress_score", "conditional_rebalance_dispersion_collapse_score", "conditional_rebalance_momentum_break_score", "conditional_rebalance_trigger_count_feature", "mu_feature_composite_raw", "mu_feature_composite_cs_z", "mu_feature_net_strength", "alpha_efficiency_21d_eom", "alpha_channel_pos_21d_eom", "alpha_mom_spread_5_21_eom", "alpha_down_up_vol_ratio_21d_eom", "alpha_efficiency_21d_cs_z_eom"}.issubset(monthly.columns)

    weekly_cols = select_micro_pipeline_columns(weekly)
    assert list(weekly_cols.columns) == ["date", "asset", "return"]
    assert not weekly_cols.empty

    micro_cols = select_micro_pipeline_columns(monthly)
    assert list(micro_cols.columns) == ["date", "asset", "return"]
    assert not micro_cols.empty




# ============================================================
# Macro context merge helpers
# ============================================================

def merge_macro_context_into_asset_panel(asset_panel_df: pd.DataFrame, macro_panel_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Merge date-keyed macro context features into an asset panel.

    Keeps the current production contract intact: date / asset / return remain untouched,
    while macro_* columns are appended and shared across assets on the same date.
    """
    if macro_panel_df is None or not isinstance(macro_panel_df, pd.DataFrame) or macro_panel_df.empty:
        return asset_panel_df
    if asset_panel_df is None or not isinstance(asset_panel_df, pd.DataFrame) or asset_panel_df.empty:
        return asset_panel_df
    if 'date' not in macro_panel_df.columns:
        return asset_panel_df
    out = asset_panel_df.copy()
    macro = macro_panel_df.copy()
    out['date'] = pd.to_datetime(out['date'], errors='coerce')
    macro['date'] = pd.to_datetime(macro['date'], errors='coerce')
    macro = macro.sort_values('date').drop_duplicates(subset=['date'], keep='last')
    macro_cols = [c for c in macro.columns if c != 'date']
    if not macro_cols:
        return out
    merged = out.merge(macro[['date'] + macro_cols], on='date', how='left')
    return merged.sort_values(['date', 'asset']).reset_index(drop=True)


if __name__ == "__main__":
    _smoke_test()
