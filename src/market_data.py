from __future__ import annotations

from io import BytesIO
from typing import Iterable, List, Literal, Optional
import os
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

try:  # pragma: no cover
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


PriceField = Literal["Adj Close", "Close"]

# Streamlit Community Cloud / shared cloud IPs can be slow or rate-limited by Yahoo.
# Keep calls bounded so Step 4 does not hang for minutes before falling back.
YAHOO_TIMEOUT_SECONDS = float(os.getenv("LIFEBUDGET_YAHOO_TIMEOUT_SECONDS", "12"))
YAHOO_DEFAULT_CHUNK_SIZE = int(os.getenv("LIFEBUDGET_YAHOO_CHUNK_SIZE", "10"))
STOOQ_TIMEOUT_SECONDS = float(os.getenv("LIFEBUDGET_STOOQ_TIMEOUT_SECONDS", "10"))
ENABLE_STOOQ_FALLBACK = str(os.getenv("LIFEBUDGET_ENABLE_STOOQ_FALLBACK", "1")).strip().lower() not in {
    "0",
    "false",
    "no",
}


def _normalize_tickers(tickers: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for t in tickers or []:
        s = str(t).strip().upper()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _safe_date_yyyymmdd(value: str) -> str:
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        return str(value or "")
    return dt.strftime("%Y%m%d")


def _pick_price_frame(df: pd.DataFrame, preferred_field: PriceField = "Adj Close") -> pd.DataFrame:
    """
    yfinance.download can return:
    - a simple frame for one ticker
    - a column MultiIndex for multiple tickers

    This helper supports both common MultiIndex layouts:
    - level 0 = price field, level 1 = ticker   (group_by='column')
    - level 0 = ticker, level 1 = price field   (some yfinance versions)
    """
    if df is None or df.empty:
        raise ValueError("Yahoo download returned an empty price frame.")

    if isinstance(df.columns, pd.MultiIndex):
        level0 = [str(x) for x in df.columns.get_level_values(0)]
        level1 = [str(x) for x in df.columns.get_level_values(1)]

        # Standard group_by='column': ("Close", "SPY")
        if preferred_field in set(level0) or "Close" in set(level0):
            field = preferred_field if preferred_field in set(level0) else "Close"
            out = df[field].copy()
            out.columns = [str(c).upper() for c in out.columns]
            return out

        # Defensive support for group_by='ticker': ("SPY", "Close")
        if preferred_field in set(level1) or "Close" in set(level1):
            field = preferred_field if preferred_field in set(level1) else "Close"
            parts: dict[str, pd.Series] = {}
            for ticker in sorted(set(level0)):
                try:
                    series = df[(ticker, field)]
                    parts[str(ticker).upper()] = pd.to_numeric(series, errors="coerce")
                except Exception:
                    continue
            if parts:
                return pd.DataFrame(parts)

        raise ValueError("Could not find Adj Close or Close in Yahoo download output.")

    # single ticker case
    if preferred_field in df.columns:
        out = df[[preferred_field]].copy()
    elif "Close" in df.columns:
        out = df[["Close"]].copy()
    else:
        raise ValueError("Could not find Adj Close or Close in Yahoo download output.")
    out.columns = ["SINGLE_ASSET"]
    return out


def _clean_price_frame(prices: pd.DataFrame, *, require_non_empty: bool = True) -> pd.DataFrame:
    if prices is None or not isinstance(prices, pd.DataFrame):
        if require_non_empty:
            raise ValueError("Price frame is not a valid DataFrame.")
        return pd.DataFrame()

    out = prices.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[pd.notna(out.index)].sort_index()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(axis=1, how="all").dropna(how="all")
    out = out.loc[:, ~out.columns.duplicated()]
    if require_non_empty and out.empty:
        raise ValueError("Yahoo download returned no usable prices after cleaning.")
    return out


def _yf_download_safe(
    tickers: list[str],
    *,
    start_date: str,
    end_date: str,
    interval: Literal["1d", "1wk", "1mo"],
    auto_adjust: bool,
    threads: bool,
) -> pd.DataFrame:
    if yf is None:
        raise ImportError(
            "yfinance is not installed. Install it with `pip install yfinance` to use Yahoo Finance download in the app."
        )

    kwargs = dict(
        tickers=tickers if len(tickers) != 1 else tickers[0],
        start=str(start_date),
        end=str(end_date),
        interval=str(interval),
        auto_adjust=bool(auto_adjust),
        progress=False,
        group_by="column",
        threads=bool(threads),
    )
    try:
        return yf.download(**kwargs, timeout=float(YAHOO_TIMEOUT_SECONDS))
    except TypeError:
        # Older yfinance builds may not accept timeout.
        return yf.download(**kwargs)


def _download_single_yahoo_history(
    ticker: str,
    *,
    start_date: str,
    end_date: str,
    interval: Literal["1d", "1wk", "1mo"],
    auto_adjust: bool,
    preferred_field: PriceField,
) -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance is not installed.")

    t = str(ticker or "").strip().upper()
    if not t:
        raise ValueError("Empty ticker.")

    obj = yf.Ticker(t)
    kwargs = dict(
        start=str(start_date),
        end=str(end_date),
        interval=str(interval),
        auto_adjust=bool(auto_adjust),
        actions=False,
    )
    try:
        raw = obj.history(**kwargs, timeout=float(YAHOO_TIMEOUT_SECONDS))
    except TypeError:
        raw = obj.history(**kwargs)

    prices = _pick_price_frame(raw, preferred_field=preferred_field)
    prices = _clean_price_frame(prices)
    if prices.columns.tolist() == ["SINGLE_ASSET"]:
        prices.columns = [t]
    else:
        prices = prices.iloc[:, :1].copy()
        prices.columns = [t]
    return prices


def _download_yahoo_price_panel_chunked(
    tickers: Iterable[str],
    *,
    start_date: str,
    end_date: str,
    interval: Literal["1d", "1wk", "1mo"] = "1d",
    auto_adjust: bool = False,
    preferred_field: PriceField = "Adj Close",
    chunk_size: int = YAHOO_DEFAULT_CHUNK_SIZE,
) -> pd.DataFrame:
    tickers_list = _normalize_tickers(tickers)
    if not tickers_list:
        raise ValueError("No tickers were provided for Yahoo download.")

    parts: list[pd.DataFrame] = []
    last_error: Exception | None = None
    effective_chunk_size = max(1, int(chunk_size or YAHOO_DEFAULT_CHUNK_SIZE))

    for i in range(0, len(tickers_list), effective_chunk_size):
        batch = tickers_list[i:i + effective_chunk_size]
        try:
            raw = _yf_download_safe(
                batch,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                auto_adjust=auto_adjust,
                threads=False,
            )
            prices = _pick_price_frame(raw, preferred_field=preferred_field)
            if prices.columns.tolist() == ["SINGLE_ASSET"] and len(batch) == 1:
                prices.columns = batch
            prices = _clean_price_frame(prices, require_non_empty=False)
            if not prices.empty:
                parts.append(prices)
        except Exception as exc:
            last_error = exc
            continue

    if not parts:
        raise ValueError(f"Yahoo chunked download returned no usable prices after cleaning. Last error: {last_error}")

    merged = pd.concat(parts, axis=1)
    merged = _clean_price_frame(merged)
    if merged.empty:
        raise ValueError("Yahoo chunked download returned an empty merged price frame.")
    return merged


def _stooq_symbol(ticker: str) -> str:
    # Stooq generally serves US ETFs/stocks as lower-case ticker + ".us".
    # Symbols with "^" are Yahoo index symbols and are intentionally skipped.
    symbol = str(ticker or "").strip().upper()
    if not symbol or symbol.startswith("^") or symbol.endswith("=F") or "/" in symbol:
        return ""
    return symbol.replace("-", ".").lower() + ".us"


def _download_single_stooq_price(
    ticker: str,
    *,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    symbol = _stooq_symbol(ticker)
    if not symbol:
        raise ValueError(f"Ticker {ticker!r} is not supported by the Stooq fallback.")

    d1 = _safe_date_yyyymmdd(str(start_date))
    d2 = _safe_date_yyyymmdd(str(end_date))
    query = urllib.parse.urlencode({"s": symbol, "i": "d", "d1": d1, "d2": d2})
    url = f"https://stooq.com/q/d/l/?{query}"

    with urllib.request.urlopen(url, timeout=float(STOOQ_TIMEOUT_SECONDS)) as response:
        raw = response.read()

    df = pd.read_csv(BytesIO(raw))
    if df is None or df.empty or "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError(f"Stooq fallback returned no usable data for {ticker}.")

    out = df[["Date", "Close"]].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.dropna(subset=["Date", "Close"]).sort_values("Date")
    if out.empty:
        raise ValueError(f"Stooq fallback returned an empty cleaned frame for {ticker}.")

    return pd.DataFrame({str(ticker).strip().upper(): out["Close"].to_numpy()}, index=out["Date"])


def _download_stooq_price_panel(
    tickers: Iterable[str],
    *,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    if not ENABLE_STOOQ_FALLBACK:
        raise ValueError("Stooq fallback is disabled by LIFEBUDGET_ENABLE_STOOQ_FALLBACK=0.")

    tickers_list = _normalize_tickers(tickers)
    parts: list[pd.DataFrame] = []
    last_error: Exception | None = None

    for ticker in tickers_list:
        try:
            prices = _download_single_stooq_price(ticker, start_date=start_date, end_date=end_date)
            prices = _clean_price_frame(prices, require_non_empty=False)
            if not prices.empty:
                parts.append(prices)
        except Exception as exc:
            last_error = exc
            continue

    if not parts:
        raise ValueError(f"Stooq fallback returned no usable prices. Last error: {last_error}")

    merged = pd.concat(parts, axis=1)
    merged = _clean_price_frame(merged)
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
    """Download a price panel for the selected universe.

    Primary source remains Yahoo Finance through yfinance. For Streamlit Cloud,
    where Yahoo often returns empty frames from shared IPs, this function uses
    bounded fallbacks and finally a Stooq daily-price fallback for US-listed ETFs.
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
        prices = _clean_price_frame(prices)
        if prices.columns.tolist() == ["SINGLE_ASSET"] and len(tickers_list) == 1:
            prices.columns = tickers_list
        return prices

    # 1) Batch Yahoo.
    try:
        raw = _yf_download_safe(
            tickers_list,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            auto_adjust=auto_adjust,
            threads=False,
        )
        prices = _pick_price_frame(raw, preferred_field=preferred_field)
        return _finalize_prices(prices)
    except Exception as exc:
        last_error = exc

    # 2) Small Yahoo chunks. Cloud-safe: avoid one huge request and avoid threads.
    try:
        prices = _download_yahoo_price_panel_chunked(
            tickers_list,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            auto_adjust=auto_adjust,
            preferred_field=preferred_field,
            chunk_size=int(chunk_size or YAHOO_DEFAULT_CHUNK_SIZE),
        )
        return _finalize_prices(prices)
    except Exception as exc:
        last_error = exc

    # 3) Per-ticker Yahoo download. This recovers when one bad ticker breaks a batch.
    successful_parts: list[pd.DataFrame] = []
    for ticker in tickers_list:
        try:
            raw = _yf_download_safe(
                [ticker],
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                auto_adjust=auto_adjust,
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
        merged = _clean_price_frame(merged)
        if not merged.empty:
            return merged

    # 4) Per-ticker Ticker.history. Some yfinance versions succeed here when download() fails.
    history_parts: list[pd.DataFrame] = []
    for ticker in tickers_list:
        try:
            prices = _download_single_yahoo_history(
                ticker,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                auto_adjust=auto_adjust,
                preferred_field=preferred_field,
            )
            if not prices.empty:
                history_parts.append(prices)
        except Exception as exc:
            last_error = exc
            continue

    if history_parts:
        merged = pd.concat(history_parts, axis=1)
        merged = _clean_price_frame(merged)
        if not merged.empty:
            return merged

    # 5) Deployment fallback: Stooq daily close data for US-listed ETFs/stocks.
    # This avoids blocking the public demo when Yahoo returns empty frames from Streamlit Cloud.
    try:
        prices = _download_stooq_price_panel(tickers_list, start_date=start_date, end_date=end_date)
        return _finalize_prices(prices)
    except Exception as exc:
        last_error = exc

    raise ValueError(f"Yahoo download failed for the requested universe. Last error: {last_error}")


def build_return_panel_from_prices(
    price_df: pd.DataFrame,
    *,
    frequency: Literal["daily", "weekly", "monthly"] = "monthly",
) -> pd.DataFrame:
    if price_df is None or price_df.empty:
        raise ValueError("price_df is empty.")

    prices = price_df.copy()
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    prices = prices.loc[pd.notna(prices.index)].sort_index().dropna(how="all")
    prices = prices.dropna(axis=1, how="all")

    freq = str(frequency).lower()
    if freq == "daily":
        sampled = prices.copy()
    elif freq == "weekly":
        sampled = prices.resample("W-FRI").last()
    elif freq == "monthly":
        sampled = prices.resample("M").last()
    else:
        raise ValueError("frequency must be one of: daily, weekly, monthly")

    rets = sampled.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if rets.empty:
        raise ValueError("Return panel is empty after pct_change; try a wider Yahoo date range.")

    # Avoid passing stack(dropna=...) because pandas' new stack implementation rejects
    # explicit dropna/sort arguments in some dependency combinations.
    long_df = rets.stack().rename("return").reset_index()
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
    """Download a small macro context panel (VIX / rates) from Yahoo and return date-keyed features.

    Output columns are date plus macro_* feature columns. Levels are forward-filled before computing
    daily/weekly/monthly change features. For rates tickers like ^TNX and ^IRX, raw quoted levels are kept.
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
