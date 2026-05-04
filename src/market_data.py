"""Market-data download and return-panel helpers for LifeBudget Micro.

This module provides the app's live Yahoo Finance data path. In deployed mode,
LifeBudget Micro can also use cached market-data panels generated elsewhere in
the project; this file is the reusable service layer for downloading fresh price
data and converting it into the long-form return panels consumed by the Strategy
Engine.

Main responsibilities:
- normalise ticker lists;
- download adjusted/close price panels from Yahoo Finance via yfinance;
- fall back from bulk download to chunked and per-ticker downloads when needed;
- convert price panels into daily, weekly, or monthly return panels;
- build a small optional macro-feature panel for volatility/rate context.

The functions here only retrieve and transform historical market data. They do
not forecast prices, provide advice, or decide whether an asset should be used.
"""

from __future__ import annotations

from typing import Iterable, List, Literal, Optional

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency in some local/test environments.
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


PriceField = Literal["Adj Close", "Close"]


def _normalize_tickers(tickers: Iterable[str]) -> List[str]:
    """Return uppercase, deduplicated ticker symbols while preserving order."""
    out: List[str] = []
    seen = set()
    for t in tickers or []:
        s = str(t).strip().upper()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _pick_price_frame(df: pd.DataFrame, preferred_field: PriceField = "Adj Close") -> pd.DataFrame:
    """Extract a clean price frame from yfinance output.

    ``yfinance.download`` can return either:
    - a simple single-ticker frame; or
    - a multi-index column frame for multiple tickers.

    The app prefers adjusted close prices when available, falling back to close
    prices when adjusted close is not present.
    """
    if df is None or df.empty:
        raise ValueError("Yahoo download returned an empty price frame.")

    if isinstance(df.columns, pd.MultiIndex):
        level0 = [str(x) for x in df.columns.get_level_values(0)]
        field = preferred_field if preferred_field in set(level0) else ("Close" if "Close" in set(level0) else None)
        if field is None:
            raise ValueError("Could not find Adj Close or Close in Yahoo download output.")
        out = df[field].copy()
        out.columns = [str(c).upper() for c in out.columns]
        return out

    # Single-ticker case.
    if preferred_field in df.columns:
        out = df[[preferred_field]].copy()
    elif "Close" in df.columns:
        out = df[["Close"]].copy()
    else:
        raise ValueError("Could not find Adj Close or Close in Yahoo download output.")
    out.columns = ["SINGLE_ASSET"]
    return out


def _download_yahoo_price_panel_chunked(
    tickers: Iterable[str],
    *,
    start_date: str,
    end_date: str,
    interval: Literal["1d", "1wk", "1mo"] = "1d",
    auto_adjust: bool = False,
    preferred_field: PriceField = "Adj Close",
    chunk_size: int = 75,
) -> pd.DataFrame:
    """Download a price panel in batches to reduce bulk Yahoo failure risk."""
    if yf is None:
        raise ImportError(
            "yfinance is not installed. Install it with `pip install yfinance` to use Yahoo Finance download in the app."
        )

    tickers_list = _normalize_tickers(tickers)
    if not tickers_list:
        raise ValueError("No tickers were provided for Yahoo download.")

    safe_chunk_size = max(1, int(chunk_size or 75))
    parts: list[pd.DataFrame] = []

    for i in range(0, len(tickers_list), safe_chunk_size):
        batch = tickers_list[i:i + safe_chunk_size]
        raw = yf.download(
            tickers=batch,
            start=str(start_date),
            end=str(end_date),
            interval=str(interval),
            auto_adjust=bool(auto_adjust),
            progress=False,
            group_by="column",
            threads=True,
        )
        prices = _pick_price_frame(raw, preferred_field=preferred_field)
        if prices.columns.tolist() == ["SINGLE_ASSET"] and len(batch) == 1:
            prices.columns = batch
        prices.index = pd.to_datetime(prices.index, errors="coerce")
        prices = prices.sort_index().dropna(how="all")
        if not prices.empty:
            parts.append(prices)

    if not parts:
        raise ValueError("Yahoo chunked download returned no usable prices after cleaning.")

    merged = pd.concat(parts, axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()].sort_index()
    merged = merged.dropna(how="all")
    if merged.empty:
        raise ValueError("Yahoo chunked download returned an empty merged price frame.")
    return merged


def download_yahoo_price_panel(
    tickers: Iterable[str],
    *,
    start_date: str,
    end_date: str,
    interval: Literal["1d", "1wk", "1mo"] = "1d",
    auto_adjust: bool = False,
    preferred_field: PriceField = "Adj Close",
    chunk_size: Optional[int] = None,
) -> pd.DataFrame:
    """Download a Yahoo price panel with bulk, chunked, and per-ticker fallbacks.

    The fallback order is deliberately conservative:
    1. attempt a normal bulk yfinance download;
    2. if that fails, retry in ticker chunks;
    3. if that fails, try each ticker individually and merge successful assets.

    This keeps the app more resilient when Yahoo intermittently throttles,
    returns partial responses, or fails on larger universes.
    """
    if yf is None:
        raise ImportError(
            "yfinance is not installed. Install it with `pip install yfinance` to use Yahoo Finance download in the app."
        )

    tickers_list = _normalize_tickers(tickers)
    if not tickers_list:
        raise ValueError("No tickers were provided for Yahoo download.")

    last_error: Exception | None = None

    def _finalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
        if prices.columns.tolist() == ["SINGLE_ASSET"] and len(tickers_list) == 1:
            prices.columns = tickers_list
        prices.index = pd.to_datetime(prices.index, errors="coerce")
        prices = prices.sort_index().dropna(how="all")
        if prices.empty:
            raise ValueError("Yahoo download returned no usable prices after cleaning.")
        return prices

    try:
        raw = yf.download(
            tickers=tickers_list,
            start=str(start_date),
            end=str(end_date),
            interval=str(interval),
            auto_adjust=bool(auto_adjust),
            progress=False,
            group_by="column",
            threads=True,
        )
        prices = _pick_price_frame(raw, preferred_field=preferred_field)
        return _finalize_prices(prices)
    except Exception as exc:
        last_error = exc

    try:
        prices = _download_yahoo_price_panel_chunked(
            tickers_list,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            auto_adjust=auto_adjust,
            preferred_field=preferred_field,
            chunk_size=max(1, int(chunk_size or 50)),
        )
        return _finalize_prices(prices)
    except Exception as exc:
        last_error = exc

    successful_parts: list[pd.DataFrame] = []
    for ticker in tickers_list:
        try:
            raw = yf.download(
                tickers=[ticker],
                start=str(start_date),
                end=str(end_date),
                interval=str(interval),
                auto_adjust=bool(auto_adjust),
                progress=False,
                group_by="column",
                threads=False,
            )
            prices = _pick_price_frame(raw, preferred_field=preferred_field)
            prices = _finalize_prices(prices)
            if prices.columns.tolist() == ["SINGLE_ASSET"]:
                prices.columns = [ticker]
            if not prices.empty:
                successful_parts.append(prices)
        except Exception as exc:
            last_error = exc
            continue

    if successful_parts:
        merged = pd.concat(successful_parts, axis=1)
        merged = merged.loc[:, ~merged.columns.duplicated()].sort_index().dropna(how="all")
        if not merged.empty:
            return merged

    raise ValueError(f"Yahoo download failed for the requested universe. Last error: {last_error}")


def build_return_panel_from_prices(
    price_df: pd.DataFrame,
    *,
    frequency: Literal["daily", "weekly", "monthly"] = "monthly",
) -> pd.DataFrame:
    """Convert a wide price panel into a long-form Strategy Engine return panel.

    Output columns:
    - ``date``
    - ``asset``
    - ``return``

    Returns are simple percentage changes in decimal form, for example 0.01 for
    +1%. The output shape is intentionally aligned with ``MicroPipelineConfig``.
    """
    if price_df is None or price_df.empty:
        raise ValueError("price_df is empty.")

    prices = price_df.copy()
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    prices = prices.sort_index().dropna(how="all")

    freq = str(frequency).lower()
    if freq == "daily":
        sampled = prices.copy()
    elif freq == "weekly":
        sampled = prices.resample("W-FRI").last()
    elif freq == "monthly":
        sampled = prices.resample("M").last()
    else:
        raise ValueError("frequency must be one of: daily, weekly, monthly")

    rets = sampled.pct_change().dropna(how="all")
    if rets.empty:
        raise ValueError("Return panel is empty after pct_change; try a wider Yahoo date range.")

    long_df = rets.stack(dropna=True).rename("return").reset_index()
    if long_df.shape[1] < 3:
        raise ValueError("Unexpected Yahoo return panel shape after stacking.")

    date_col = str(long_df.columns[0])
    asset_col = str(long_df.columns[1])
    long_df = long_df.rename(columns={date_col: "date", asset_col: "asset"})
    long_df["asset"] = long_df["asset"].astype(str).str.upper()
    long_df["date"] = pd.to_datetime(long_df["date"], errors="coerce")
    long_df["return"] = pd.to_numeric(long_df["return"], errors="coerce")
    long_df = long_df.dropna(subset=["date", "asset", "return"]).sort_values(["date", "asset"]).reset_index(drop=True)
    if long_df.empty:
        raise ValueError("Return panel is empty after cleaning.")
    return long_df


def download_yahoo_return_panel(
    tickers: Iterable[str],
    *,
    start_date: str,
    end_date: str,
    frequency: Literal["daily", "weekly", "monthly"] = "monthly",
    interval: Literal["1d", "1wk", "1mo"] = "1d",
    auto_adjust: bool = False,
    preferred_field: PriceField = "Adj Close",
    chunk_size: Optional[int] = None,
) -> pd.DataFrame:
    """Download Yahoo prices and convert them into a long-form return panel."""
    prices = download_yahoo_price_panel(
        tickers,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        auto_adjust=auto_adjust,
        preferred_field=preferred_field,
        chunk_size=chunk_size,
    )
    return build_return_panel_from_prices(prices, frequency=frequency)


MACRO_TICKER_MAP = {
    "^VIX": "macro_vix",
    "^TNX": "macro_rate_10y",
    "^IRX": "macro_rate_3m",
}


def download_yahoo_macro_feature_panel(
    *,
    start_date: str,
    end_date: str,
    frequency: Literal["daily", "weekly", "monthly"] = "daily",
    auto_adjust: bool = False,
    ticker_map: Optional[dict[str, str]] = None,
    preferred_field: PriceField = "Adj Close",
) -> pd.DataFrame:
    """Download a small macro context panel from Yahoo.

    The default macro set contains VIX, 10-year Treasury yield proxy and
    3-month Treasury bill proxy. The output is date-keyed and contains level,
    change, return and rolling z-score features.

    These features are contextual inputs for analysis/diagnostics. They should
    not be interpreted as forecasts or recommendations.
    """
    tmap = dict(ticker_map or MACRO_TICKER_MAP)
    prices = download_yahoo_price_panel(
        tmap.keys(),
        start_date=start_date,
        end_date=end_date,
        interval="1d",
        auto_adjust=auto_adjust,
        preferred_field=preferred_field,
    )
    prices = prices.rename(columns={k: v for k, v in tmap.items() if k in prices.columns})
    prices = prices.sort_index().ffill()

    freq = str(frequency).lower()
    if freq == "daily":
        sampled = prices.copy()
    elif freq == "weekly":
        sampled = prices.resample("W-FRI").last()
    elif freq == "monthly":
        sampled = prices.resample("M").last()
    else:
        raise ValueError("frequency must be one of: daily, weekly, monthly")

    out = pd.DataFrame(index=sampled.index)
    for col in sampled.columns:
        s = pd.to_numeric(sampled[col], errors="coerce").astype(float)
        out[f"{col}_level"] = s
        out[f"{col}_chg_1"] = s.diff(1)
        out[f"{col}_chg_5"] = s.diff(5)
        out[f"{col}_ret_1"] = s.pct_change(1).replace([np.inf, -np.inf], np.nan)
        out[f"{col}_ret_5"] = s.pct_change(5).replace([np.inf, -np.inf], np.nan)
        out[f"{col}_z_21"] = (
            (s - s.rolling(21, min_periods=5).mean())
            / s.rolling(21, min_periods=5).std(ddof=1)
        ).replace([np.inf, -np.inf], np.nan)

    if {"macro_rate_10y_level", "macro_rate_3m_level"}.issubset(out.columns):
        out["macro_rates_slope_level"] = out["macro_rate_10y_level"] - out["macro_rate_3m_level"]
        out["macro_rates_slope_chg_1"] = out["macro_rates_slope_level"].diff(1)
        out["macro_rates_slope_z_21"] = (
            (out["macro_rates_slope_level"] - out["macro_rates_slope_level"].rolling(21, min_periods=5).mean())
            / out["macro_rates_slope_level"].rolling(21, min_periods=5).std(ddof=1)
        ).replace([np.inf, -np.inf], np.nan)

    out = out.reset_index().rename(columns={out.index.name or "index": "date"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.sort_values("date").reset_index(drop=True)
    return out