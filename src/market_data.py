from __future__ import annotations

from io import BytesIO, StringIO
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

# -----------------------------------------------------------------------------
# Deployment controls
# -----------------------------------------------------------------------------
# Streamlit Community Cloud often rate-limits Yahoo/yfinance from shared IPs.
# Therefore the public deployment path uses Stooq first by default for standard
# US ETF/equity panels. Yahoo is retained as an opt-in fallback for local/dev use.

STOOQ_TIMEOUT_SECONDS = float(os.getenv("LIFEBUDGET_STOOQ_TIMEOUT_SECONDS", "10"))
ENABLE_STOOQ_FALLBACK = str(os.getenv("LIFEBUDGET_ENABLE_STOOQ_FALLBACK", "1")).strip().lower() not in {
    "0",
    "false",
    "no",
}
PREFER_STOOQ_FIRST = str(os.getenv("LIFEBUDGET_PREFER_STOOQ_FIRST", "1")).strip().lower() not in {
    "0",
    "false",
    "no",
}
MIN_STOOQ_ASSETS = int(os.getenv("LIFEBUDGET_MIN_STOOQ_ASSETS", "12"))

# Yahoo fallback is deliberately OFF by default on cloud. If you want local Yahoo
# behaviour, set LIFEBUDGET_ALLOW_YAHOO_FALLBACK=1 in your environment.
ALLOW_YAHOO_FALLBACK = str(os.getenv("LIFEBUDGET_ALLOW_YAHOO_FALLBACK", "0")).strip().lower() in {
    "1",
    "true",
    "yes",
}
ALLOW_YAHOO_MACRO = str(os.getenv("LIFEBUDGET_ALLOW_YAHOO_MACRO", "0")).strip().lower() in {
    "1",
    "true",
    "yes",
}
YAHOO_TIMEOUT_SECONDS = float(os.getenv("LIFEBUDGET_YAHOO_TIMEOUT_SECONDS", "12"))
YAHOO_DEFAULT_CHUNK_SIZE = int(os.getenv("LIFEBUDGET_YAHOO_CHUNK_SIZE", "10"))


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def _normalize_tickers(tickers: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
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
        raise ValueError("Price download returned no usable prices after cleaning.")
    return out


def _looks_like_yahoo_rate_limit(exc: Exception | None) -> bool:
    text = str(exc or "").lower()
    return any(token in text for token in ("too many requests", "ratelimit", "rate limit", "429"))


# -----------------------------------------------------------------------------
# Stooq download path — default for Streamlit Cloud deployment
# -----------------------------------------------------------------------------

def _stooq_symbol(ticker: str) -> str:
    """Map common US ETF/equity tickers to Stooq CSV symbols."""
    symbol = str(ticker or "").strip().upper()
    if not symbol or symbol.startswith("^") or symbol.endswith("=F") or "/" in symbol:
        return ""
    return symbol.replace("-", ".").lower() + ".us"


def _parse_stooq_csv_response(raw: bytes, ticker: str) -> pd.DataFrame:
    """Parse Stooq CSV robustly for Streamlit Cloud deployment.

    Stooq normally returns a clean CSV starting with Date,Open,High,Low,Close,Volume.
    In cloud/server contexts it can sometimes prepend text/status lines or return
    a malformed body. A direct pd.read_csv(BytesIO(raw)) can then fail with
    tokenizing errors. This helper finds the real CSV header first and skips
    malformed rows instead of killing Step 4.
    """
    text = raw.decode("utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if str(line).strip()]
    if not lines:
        raise ValueError(f"Stooq returned an empty response for {ticker}.")

    header_idx: int | None = None
    sep = ","
    for i, line in enumerate(lines):
        lower = line.lower()
        if "date" in lower and "close" in lower:
            sep = ";" if line.count(";") > line.count(",") else ","
            header_idx = i
            break

    if header_idx is None:
        preview = " | ".join(lines[:4])[:300]
        raise ValueError(f"Stooq response for {ticker} did not contain a Date/Close header. Preview: {preview}")

    csv_text = "\n".join(lines[header_idx:])
    try:
        df = pd.read_csv(StringIO(csv_text), sep=sep, engine="python", on_bad_lines="skip")
    except Exception as exc:
        preview = " | ".join(lines[header_idx:header_idx + 4])[:300]
        raise ValueError(f"Could not parse Stooq CSV for {ticker}. Last error: {exc}. Preview: {preview}")

    if df is None or df.empty:
        raise ValueError(f"Stooq fallback returned an empty parsed frame for {ticker}.")

    normalised = {str(c).strip().lower(): c for c in df.columns}
    date_col = normalised.get("date")
    close_col = normalised.get("close")
    if date_col is None or close_col is None:
        raise ValueError(f"Stooq parsed frame for {ticker} is missing Date/Close columns: {list(df.columns)}")

    out = df[[date_col, close_col]].copy()
    out.columns = ["Date", "Close"]
    return out


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

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; LifeBudgetMicro/1.0; +https://streamlit.app)",
            "Accept": "text/csv,text/plain,*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=float(STOOQ_TIMEOUT_SECONDS)) as response:
        raw = response.read()

    df = _parse_stooq_csv_response(raw, ticker)

    out = df[["Date", "Close"]].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.dropna(subset=["Date", "Close"]).sort_values("Date")
    if out.empty:
        raise ValueError(f"Stooq fallback returned an empty cleaned frame for {ticker}.")

    ticker_name = str(ticker).strip().upper()
    return pd.DataFrame({ticker_name: out["Close"].to_numpy()}, index=out["Date"])

def _download_stooq_price_panel(
    tickers: Iterable[str],
    *,
    start_date: str,
    end_date: str,
    min_assets: int | None = None,
) -> pd.DataFrame:
    if not ENABLE_STOOQ_FALLBACK:
        raise ValueError("Stooq fallback is disabled by LIFEBUDGET_ENABLE_STOOQ_FALLBACK=0.")

    tickers_list = _normalize_tickers(tickers)
    if not tickers_list:
        raise ValueError("No tickers were provided for Stooq download.")

    parts: list[pd.DataFrame] = []
    failed: list[str] = []
    last_error: Exception | None = None

    for ticker in tickers_list:
        try:
            prices = _download_single_stooq_price(ticker, start_date=start_date, end_date=end_date)
            prices = _clean_price_frame(prices, require_non_empty=False)
            if not prices.empty:
                parts.append(prices)
            else:
                failed.append(ticker)
        except Exception as exc:
            last_error = exc
            failed.append(ticker)
            continue

    if not parts:
        raise ValueError(f"Stooq returned no usable prices. Last error: {last_error}")

    merged = pd.concat(parts, axis=1)
    merged = _clean_price_frame(merged)

    required = int(min_assets if min_assets is not None else min(int(MIN_STOOQ_ASSETS), len(tickers_list)))
    if int(merged.shape[1]) < required:
        raise ValueError(
            f"Stooq returned only {int(merged.shape[1])} usable assets; required at least {required}. "
            f"Failed tickers: {failed[:12]}"
        )

    return merged


# -----------------------------------------------------------------------------
# Yahoo helpers — retained only as an opt-in/local fallback
# -----------------------------------------------------------------------------

def _pick_price_frame(df: pd.DataFrame, preferred_field: PriceField = "Adj Close") -> pd.DataFrame:
    """Extract Adj Close/Close from common yfinance output layouts."""
    if df is None or df.empty:
        raise ValueError("Yahoo download returned an empty price frame.")

    if isinstance(df.columns, pd.MultiIndex):
        level0 = [str(x) for x in df.columns.get_level_values(0)]
        level1 = [str(x) for x in df.columns.get_level_values(1)]

        if preferred_field in set(level0) or "Close" in set(level0):
            field = preferred_field if preferred_field in set(level0) else "Close"
            out = df[field].copy()
            out.columns = [str(c).upper() for c in out.columns]
            return out

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

    if preferred_field in df.columns:
        out = df[[preferred_field]].copy()
    elif "Close" in df.columns:
        out = df[["Close"]].copy()
    else:
        raise ValueError("Could not find Adj Close or Close in Yahoo download output.")
    out.columns = ["SINGLE_ASSET"]
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
        raise ImportError("yfinance is not installed.")

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
        return yf.download(**kwargs)


def _download_yahoo_price_panel_minimal(
    tickers: Iterable[str],
    *,
    start_date: str,
    end_date: str,
    interval: Literal["1d", "1wk", "1mo"] = "1d",
    auto_adjust: bool = False,
    preferred_field: PriceField = "Adj Close",
) -> pd.DataFrame:
    """Single Yahoo batch attempt. No ticker-by-ticker retry on cloud."""
    if yf is None:
        raise ImportError("yfinance is not installed.")

    tickers_list = _normalize_tickers(tickers)
    if not tickers_list:
        raise ValueError("No tickers were provided for Yahoo download.")

    raw = _yf_download_safe(
        tickers_list,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        auto_adjust=auto_adjust,
        threads=False,
    )
    prices = _pick_price_frame(raw, preferred_field=preferred_field)
    prices = _clean_price_frame(prices)
    if prices.columns.tolist() == ["SINGLE_ASSET"] and len(tickers_list) == 1:
        prices.columns = tickers_list
    return prices


# -----------------------------------------------------------------------------
# Public API used by Step 4 / feature pipeline
# -----------------------------------------------------------------------------

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

    Despite the historical function name, the deployment-safe default is Stooq.
    Yahoo/yfinance is available only when LIFEBUDGET_ALLOW_YAHOO_FALLBACK=1.
    This avoids repeated Yahoo rate-limit failures on Streamlit Community Cloud.
    """
    tickers_list = _normalize_tickers(tickers)
    if not tickers_list:
        raise ValueError("No tickers were provided for market-data download.")

    last_error: Exception | None = None

    if ENABLE_STOOQ_FALLBACK and PREFER_STOOQ_FIRST:
        try:
            return _download_stooq_price_panel(
                tickers_list,
                start_date=start_date,
                end_date=end_date,
                min_assets=min(int(MIN_STOOQ_ASSETS), len(tickers_list)),
            )
        except Exception as exc:
            last_error = exc
            if not ALLOW_YAHOO_FALLBACK:
                raise ValueError(f"Stooq deployment data failed and Yahoo fallback is disabled. Last error: {last_error}")

    if ALLOW_YAHOO_FALLBACK:
        try:
            return _download_yahoo_price_panel_minimal(
                tickers_list,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                auto_adjust=auto_adjust,
                preferred_field=preferred_field,
            )
        except Exception as exc:
            last_error = exc
            if _looks_like_yahoo_rate_limit(exc):
                raise ValueError(f"Yahoo is rate-limiting this cloud app. Last error: {last_error}")

    raise ValueError(f"Market-data download failed. Last error: {last_error}")


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
        raise ValueError("Return panel is empty after pct_change; try a wider date range.")

    # Avoid stack(dropna=...) because newer pandas stack implementations reject
    # explicit dropna/sort args in some dependency combinations.
    long_df = rets.stack().rename("return").reset_index()
    if long_df.shape[1] < 3:
        raise ValueError("Unexpected return panel shape after stacking.")

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


# -----------------------------------------------------------------------------
# Macro features — optional; never block Step 4 on cloud
# -----------------------------------------------------------------------------

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
    """Download optional macro context features.

    On Streamlit Cloud this returns an empty date-only frame by default so Yahoo
    rate limits cannot block the Step 4 asset panel. Enable with
    LIFEBUDGET_ALLOW_YAHOO_MACRO=1 when running locally.
    """
    if not ALLOW_YAHOO_MACRO:
        return pd.DataFrame({"date": pd.to_datetime([], errors="coerce")})

    tmap = dict(ticker_map or MACRO_TICKER_MAP)

    try:
        prices = _download_yahoo_price_panel_minimal(
            _normalize_tickers(tmap.keys()),
            start_date=start_date,
            end_date=end_date,
            interval="1d",
            auto_adjust=auto_adjust,
            preferred_field=preferred_field,
        )
    except Exception:
        return pd.DataFrame({"date": pd.to_datetime([], errors="coerce")})

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
