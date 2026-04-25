from __future__ import annotations

"""
src/investment.py

Optional long-horizon savings growth module for LifeBudget Micro.

This version includes two layers:

1) Long-horizon savings-growth simulation
   - bootstrap from realised monthly returns
   - parametric Gaussian simulation
   - summary metrics and explanation layer

2) Optional micro-pipeline for live OOS generation
   - compact walk-forward over monthly asset-panel data
   - per-asset mu / sigma estimation
   - cross-asset weighting with coursework-style controls
   - realised OOS portfolio return generation

Important
---------
This is a MICRO pipeline, not the full coursework research stack.
It is meant to be:
- compact
- maintainable
- usable inside the FYP
- able to regenerate a fresh OOS return series when newer data arrives

It is NOT:
- the full notebook
- a full benchmark/tuning framework
- a full research environment

The micro-pipeline below is deliberately closer to the final coursework spirit than
an ultra-minimal toy implementation. In particular it supports:
- longer training windows
- risk-adjusted cross-sectional scores
- softmax allocation with temperature
- shrinkage toward equal weight
- optional inertia versus previous weights
- optional deadband to suppress tiny reallocations
- dispersion gating when cross-sectional signal strength is weak
- simple regime diagnostics

It still does NOT guarantee exact reproduction of the coursework outputs unless:
- the input panel matches the coursework panel
- the same universe, dates, and preprocessing are used
- the same feature engineering and exact model choices are replicated upstream
"""

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import QuantileRegressor
except Exception:
    QuantileRegressor = None

import hashlib
import json
from pathlib import Path
from statistics import NormalDist
import time

from src.expenses import weekly_to_monthly


# ============================================================
# Constants and profile templates
# ============================================================

DEFAULT_BLOCK_LEN = 6
MIN_ANNUAL_VOL = 0.0001
MAX_REASONABLE_ANNUAL_VOL = 0.80
MAX_REASONABLE_ANNUAL_RETURN = 0.30

PROFILE_LIBRARY: Dict[str, Dict[str, float]] = {
    "Conservative": {
        "mean_scale": 0.70,
        "vol_scale": 0.60,
        "fallback_return_annual": 0.04,
        "fallback_vol_annual": 0.08,
    },
    "Balanced": {
        "mean_scale": 1.00,
        "vol_scale": 1.00,
        "fallback_return_annual": 0.06,
        "fallback_vol_annual": 0.12,
    },
    "Growth": {
        "mean_scale": 1.20,
        "vol_scale": 1.30,
        "fallback_return_annual": 0.08,
        "fallback_vol_annual": 0.17,
    },
    "Aggressive": {
        "mean_scale": 1.35,
        "vol_scale": 1.60,
        "fallback_return_annual": 0.10,
        "fallback_vol_annual": 0.24,
    },
}

_VALID_BOOTSTRAP = {"iid", "block"}
_VALID_CONTRIB_TIMING = {"start", "end"}
_VALID_REGIME_MODE = {"none", "quantile"}
_VALID_MU_REGIME_MODE = {"pooled", "split"}

COMPOSITE_WEIGHT_KEYS: Tuple[str, ...] = (
    "sharpe",
    "cagr",
    "max_drawdown",
    "mean_turnover",
    "diversification",
    "stability",
)


# ============================================================
# Asset-group taxonomy helpers
# ============================================================

ASSET_GROUP_ORDER: Tuple[str, ...] = (
    "Equity",
    "Fixed Income",
    "Commodity",
    "Real Estate",
    "Cash / Short Duration",
    "Alternative / Multi-asset",
    "Crypto",
    "Other / Unclassified",
)

_ASSET_GROUP_FALLBACK = "Other / Unclassified"


def _normalize_ticker_symbol(ticker: Any) -> str:
    raw = str(ticker or "").strip().upper()
    if not raw:
        return ""
    return raw.replace(" ", "").replace("/", "-")


_ASSET_GROUP_MAP: Dict[str, Dict[str, str]] = {
    # Core multi-asset / broad equity beta
    "SPY": {"group": "Equity", "subgroup": "US Large Cap"},
    "VOO": {"group": "Equity", "subgroup": "US Large Cap"},
    "IVV": {"group": "Equity", "subgroup": "US Large Cap"},
    "VTI": {"group": "Equity", "subgroup": "US Total Market"},
    "SCHX": {"group": "Equity", "subgroup": "US Large Cap"},
    "VV": {"group": "Equity", "subgroup": "US Large Cap"},
    "QQQ": {"group": "Equity", "subgroup": "US Large Cap / Growth"},
    "DIA": {"group": "Equity", "subgroup": "US Large Cap"},
    "MDY": {"group": "Equity", "subgroup": "US Mid Cap"},
    "RSP": {"group": "Equity", "subgroup": "US Equal Weight"},
    "IWM": {"group": "Equity", "subgroup": "US Small Cap"},
    "IJH": {"group": "Equity", "subgroup": "US Mid Cap"},
    "IJR": {"group": "Equity", "subgroup": "US Small Cap"},
    "VB": {"group": "Equity", "subgroup": "US Small Cap"},
    "VUG": {"group": "Equity", "subgroup": "US Large Cap / Growth"},
    "VTV": {"group": "Equity", "subgroup": "US Large Cap / Value"},
    "IWF": {"group": "Equity", "subgroup": "US Large Cap / Growth"},
    "IWD": {"group": "Equity", "subgroup": "US Large Cap / Value"},
    "SCHG": {"group": "Equity", "subgroup": "US Large Cap / Growth"},
    "SCHV": {"group": "Equity", "subgroup": "US Large Cap / Value"},

    # Developed / international / EM equity
    "EFA": {"group": "Equity", "subgroup": "Developed ex-US"},
    "VEA": {"group": "Equity", "subgroup": "Developed ex-US"},
    "IEFA": {"group": "Equity", "subgroup": "Developed ex-US"},
    "EEM": {"group": "Equity", "subgroup": "Emerging Markets"},
    "VWO": {"group": "Equity", "subgroup": "Emerging Markets"},
    "IEMG": {"group": "Equity", "subgroup": "Emerging Markets"},
    "ACWI": {"group": "Equity", "subgroup": "Global All-Country"},
    "VXUS": {"group": "Equity", "subgroup": "Global ex-US"},
    "VSS": {"group": "Equity", "subgroup": "International Small Cap"},
    "GWX": {"group": "Equity", "subgroup": "International Small Cap"},
    "SCZ": {"group": "Equity", "subgroup": "International Small Cap"},
    "VGK": {"group": "Equity", "subgroup": "Europe"},
    "EWJ": {"group": "Equity", "subgroup": "Japan"},
    "HEWJ": {"group": "Equity", "subgroup": "Japan Hedged"},
    "DXJ": {"group": "Equity", "subgroup": "Japan Hedged"},
    "EWU": {"group": "Equity", "subgroup": "United Kingdom"},
    "EWG": {"group": "Equity", "subgroup": "Germany"},
    "EWQ": {"group": "Equity", "subgroup": "France"},
    "EWZ": {"group": "Equity", "subgroup": "Brazil"},
    "EWC": {"group": "Equity", "subgroup": "Canada"},
    "EWA": {"group": "Equity", "subgroup": "Australia"},
    "INDA": {"group": "Equity", "subgroup": "India"},
    "FXI": {"group": "Equity", "subgroup": "China Large Cap"},
    "KWEB": {"group": "Equity", "subgroup": "China Internet"},
    "MCHI": {"group": "Equity", "subgroup": "China Broad"},
    "ASHR": {"group": "Equity", "subgroup": "China A-Shares"},
    "EWY": {"group": "Equity", "subgroup": "South Korea"},
    "EWT": {"group": "Equity", "subgroup": "Taiwan"},
    "EWH": {"group": "Equity", "subgroup": "Hong Kong"},
    "EWS": {"group": "Equity", "subgroup": "Singapore"},
    "EWP": {"group": "Equity", "subgroup": "Spain"},
    "EWI": {"group": "Equity", "subgroup": "Italy"},
    "EWN": {"group": "Equity", "subgroup": "Netherlands"},
    "EZA": {"group": "Equity", "subgroup": "South Africa"},
    "TUR": {"group": "Equity", "subgroup": "Turkey"},
    "ARGT": {"group": "Equity", "subgroup": "Argentina"},
    "GREK": {"group": "Equity", "subgroup": "Greece"},
    "EPOL": {"group": "Equity", "subgroup": "Poland"},
    "THD": {"group": "Equity", "subgroup": "Thailand"},
    "ECH": {"group": "Equity", "subgroup": "Chile"},
    "ERUS": {"group": "Equity", "subgroup": "Russia"},
    "NORW": {"group": "Equity", "subgroup": "Norway"},
    "ENZL": {"group": "Equity", "subgroup": "New Zealand"},
    "AAXJ": {"group": "Equity", "subgroup": "Asia ex-Japan"},
    "VNM": {"group": "Equity", "subgroup": "Vietnam"},
    "FM": {"group": "Equity", "subgroup": "Frontier Markets"},

    # Sector / industry equity
    "XLK": {"group": "Equity", "subgroup": "Sector Equity"},
    "XLF": {"group": "Equity", "subgroup": "Sector Equity"},
    "XLV": {"group": "Equity", "subgroup": "Sector Equity"},
    "XLI": {"group": "Equity", "subgroup": "Sector Equity"},
    "XLP": {"group": "Equity", "subgroup": "Sector Equity"},
    "XLY": {"group": "Equity", "subgroup": "Sector Equity"},
    "XLE": {"group": "Equity", "subgroup": "Sector Equity"},
    "XLB": {"group": "Equity", "subgroup": "Sector Equity"},
    "XLU": {"group": "Equity", "subgroup": "Sector Equity"},
    "XLC": {"group": "Equity", "subgroup": "Sector Equity"},
    "XLRE": {"group": "Real Estate", "subgroup": "REIT / Real Estate"},
    "XHB": {"group": "Equity", "subgroup": "Sector Equity"},
    "XRT": {"group": "Equity", "subgroup": "Sector Equity"},
    "XBI": {"group": "Equity", "subgroup": "Biotech"},
    "IBB": {"group": "Equity", "subgroup": "Biotech"},
    "IHI": {"group": "Equity", "subgroup": "Healthcare Equipment"},
    "SMH": {"group": "Equity", "subgroup": "Semiconductors"},
    "SOXX": {"group": "Equity", "subgroup": "Semiconductors"},
    "ITA": {"group": "Equity", "subgroup": "Aerospace & Defense"},
    "PPA": {"group": "Equity", "subgroup": "Aerospace & Defense"},
    "IYT": {"group": "Equity", "subgroup": "Transportation"},
    "XTN": {"group": "Equity", "subgroup": "Transportation"},
    "KRE": {"group": "Equity", "subgroup": "Regional Banks"},
    "KBE": {"group": "Equity", "subgroup": "Banks"},
    "KIE": {"group": "Equity", "subgroup": "Insurance"},
    "KCE": {"group": "Equity", "subgroup": "Capital Markets"},

    # Dividend / factor / smart beta equity
    "SCHD": {"group": "Equity", "subgroup": "Dividend / Quality"},
    "VIG": {"group": "Equity", "subgroup": "Dividend / Quality"},
    "DVY": {"group": "Equity", "subgroup": "Dividend / Quality"},
    "NOBL": {"group": "Equity", "subgroup": "Dividend / Quality"},
    "MTUM": {"group": "Equity", "subgroup": "Factor / Smart Beta"},
    "QUAL": {"group": "Equity", "subgroup": "Factor / Smart Beta"},
    "USMV": {"group": "Equity", "subgroup": "Factor / Smart Beta"},
    "VLUE": {"group": "Equity", "subgroup": "Factor / Smart Beta"},
    "SIZE": {"group": "Equity", "subgroup": "Factor / Smart Beta"},
    "SPLV": {"group": "Equity", "subgroup": "Factor / Smart Beta"},
    "SPYG": {"group": "Equity", "subgroup": "Factor / Smart Beta"},
    "SPYV": {"group": "Equity", "subgroup": "Factor / Smart Beta"},
    "FNDX": {"group": "Equity", "subgroup": "Factor / Smart Beta"},
    "DGRW": {"group": "Equity", "subgroup": "Dividend / Quality"},
    "DGRO": {"group": "Equity", "subgroup": "Dividend / Quality"},
    "RDVY": {"group": "Equity", "subgroup": "Dividend / Quality"},
    "PID": {"group": "Equity", "subgroup": "Dividend / Quality"},
    "DEM": {"group": "Equity", "subgroup": "Dividend / Quality"},

    # Thematic / thematic equity
    "ICLN": {"group": "Equity", "subgroup": "Thematic Equity"},
    "TAN": {"group": "Equity", "subgroup": "Thematic Equity"},
    "LIT": {"group": "Equity", "subgroup": "Thematic Equity"},
    "URNM": {"group": "Equity", "subgroup": "Thematic Equity"},
    "REMX": {"group": "Equity", "subgroup": "Thematic Equity"},
    "ARKK": {"group": "Equity", "subgroup": "Thematic Equity"},
    "ARKG": {"group": "Equity", "subgroup": "Thematic Equity"},
    "ARKF": {"group": "Equity", "subgroup": "Thematic Equity"},
    "ARKW": {"group": "Equity", "subgroup": "Thematic Equity"},
    "MJ": {"group": "Equity", "subgroup": "Thematic Equity"},
    "YOLO": {"group": "Equity", "subgroup": "Thematic Equity"},
    "MSOS": {"group": "Equity", "subgroup": "Thematic Equity"},
    "JETS": {"group": "Equity", "subgroup": "Thematic Equity"},
    "AWAY": {"group": "Equity", "subgroup": "Thematic Equity"},
    "HACK": {"group": "Equity", "subgroup": "Thematic Equity"},
    "CIBR": {"group": "Equity", "subgroup": "Thematic Equity"},
    "BUG": {"group": "Equity", "subgroup": "Thematic Equity"},
    "BOTZ": {"group": "Equity", "subgroup": "Thematic Equity"},
    "ROBO": {"group": "Equity", "subgroup": "Thematic Equity"},
    "DRIV": {"group": "Equity", "subgroup": "Thematic Equity"},
    "KARS": {"group": "Equity", "subgroup": "Thematic Equity"},
    "IDRV": {"group": "Equity", "subgroup": "Thematic Equity"},
    "IRBO": {"group": "Equity", "subgroup": "Thematic Equity"},
    "SKYY": {"group": "Equity", "subgroup": "Thematic Equity"},
    "CLOU": {"group": "Equity", "subgroup": "Thematic Equity"},
    "WCLD": {"group": "Equity", "subgroup": "Thematic Equity"},
    "FDN": {"group": "Equity", "subgroup": "Internet / Thematic Equity"},
    "PNQI": {"group": "Equity", "subgroup": "Internet / Thematic Equity"},
    "IBUY": {"group": "Equity", "subgroup": "Internet / Thematic Equity"},
    "XSD": {"group": "Equity", "subgroup": "Semiconductors"},
    "PSI": {"group": "Equity", "subgroup": "Semiconductors"},
    "XOP": {"group": "Equity", "subgroup": "Energy Exploration"},
    "OIH": {"group": "Equity", "subgroup": "Energy Services"},
    "XES": {"group": "Equity", "subgroup": "Energy Services"},
    "AMLP": {"group": "Equity", "subgroup": "Energy Infrastructure"},

    # Fixed income / cash
    "TLT": {"group": "Fixed Income", "subgroup": "Treasuries"},
    "IEF": {"group": "Fixed Income", "subgroup": "Treasuries"},
    "IEI": {"group": "Fixed Income", "subgroup": "Treasuries"},
    "SHY": {"group": "Cash / Short Duration", "subgroup": "Treasuries / Cash"},
    "VGSH": {"group": "Cash / Short Duration", "subgroup": "Treasuries / Cash"},
    "BIL": {"group": "Cash / Short Duration", "subgroup": "Treasuries / Cash"},
    "SHV": {"group": "Cash / Short Duration", "subgroup": "Treasuries / Cash"},
    "SGOV": {"group": "Cash / Short Duration", "subgroup": "Treasuries / Cash"},
    "TFLO": {"group": "Cash / Short Duration", "subgroup": "Treasuries / Cash"},
    "SCHO": {"group": "Cash / Short Duration", "subgroup": "Treasuries / Cash"},
    "SCHR": {"group": "Fixed Income", "subgroup": "Treasuries"},
    "TIP": {"group": "Fixed Income", "subgroup": "Inflation-Linked Bonds"},
    "STIP": {"group": "Fixed Income", "subgroup": "Inflation-Linked Bonds"},
    "LTPZ": {"group": "Fixed Income", "subgroup": "Inflation-Linked Bonds"},
    "TIPX": {"group": "Fixed Income", "subgroup": "Inflation-Linked Bonds"},
    "LQD": {"group": "Fixed Income", "subgroup": "Investment Grade Credit"},
    "VCIT": {"group": "Fixed Income", "subgroup": "Investment Grade Credit"},
    "VCSH": {"group": "Fixed Income", "subgroup": "Investment Grade Credit"},
    "HYG": {"group": "Fixed Income", "subgroup": "High Yield Credit"},
    "JNK": {"group": "Fixed Income", "subgroup": "High Yield Credit"},
    "SJNK": {"group": "Fixed Income", "subgroup": "High Yield Credit"},
    "USHY": {"group": "Fixed Income", "subgroup": "High Yield Credit"},
    "ANGL": {"group": "Fixed Income", "subgroup": "High Yield Credit"},
    "EMB": {"group": "Fixed Income", "subgroup": "Emerging Market Debt"},
    "BND": {"group": "Fixed Income", "subgroup": "Broad Aggregate Bonds"},
    "AGG": {"group": "Fixed Income", "subgroup": "Broad Aggregate Bonds"},
    "SCHZ": {"group": "Fixed Income", "subgroup": "Broad Aggregate Bonds"},
    "MBB": {"group": "Fixed Income", "subgroup": "Mortgage Bonds"},
    "VMBS": {"group": "Fixed Income", "subgroup": "Mortgage Bonds"},
    "GNMA": {"group": "Fixed Income", "subgroup": "Mortgage Bonds"},
    "CMBS": {"group": "Fixed Income", "subgroup": "Mortgage Bonds"},
    "MUB": {"group": "Fixed Income", "subgroup": "Municipal Bonds"},
    "HYMB": {"group": "Fixed Income", "subgroup": "Municipal Bonds"},
    "BSV": {"group": "Fixed Income", "subgroup": "Short-Term Bonds"},
    "BIV": {"group": "Fixed Income", "subgroup": "Intermediate Bonds"},
    "BLV": {"group": "Fixed Income", "subgroup": "Long-Duration Bonds"},
    "BWX": {"group": "Fixed Income", "subgroup": "International Bonds"},
    "IGSB": {"group": "Cash / Short Duration", "subgroup": "Ultra-Short Credit"},
    "FLOT": {"group": "Cash / Short Duration", "subgroup": "Floating Rate / Ultra-Short"},
    "MINT": {"group": "Cash / Short Duration", "subgroup": "Ultra-Short"},
    "JPST": {"group": "Cash / Short Duration", "subgroup": "Ultra-Short"},
    "CWB": {"group": "Fixed Income", "subgroup": "Convertible Bonds"},

    # Real estate
    "VNQ": {"group": "Real Estate", "subgroup": "REIT / Real Estate"},
    "REM": {"group": "Real Estate", "subgroup": "Mortgage REIT"},
    "REET": {"group": "Real Estate", "subgroup": "Global REIT"},
    "SCHH": {"group": "Real Estate", "subgroup": "REIT / Real Estate"},
    "RWX": {"group": "Real Estate", "subgroup": "Global REIT"},
    "IFGL": {"group": "Real Estate", "subgroup": "Global REIT"},
    "RWR": {"group": "Real Estate", "subgroup": "REIT / Real Estate"},
    "FREL": {"group": "Real Estate", "subgroup": "REIT / Real Estate"},
    "IYR": {"group": "Real Estate", "subgroup": "REIT / Real Estate"},
    "VNQI": {"group": "Real Estate", "subgroup": "Global REIT"},

    # Commodities / precious metals / broad resources
    "GLD": {"group": "Commodity", "subgroup": "Gold"},
    "IAU": {"group": "Commodity", "subgroup": "Gold"},
    "SLV": {"group": "Commodity", "subgroup": "Silver"},
    "PPLT": {"group": "Commodity", "subgroup": "Precious Metals"},
    "PALL": {"group": "Commodity", "subgroup": "Precious Metals"},
    "GDX": {"group": "Equity", "subgroup": "Gold Miners"},
    "GDXJ": {"group": "Equity", "subgroup": "Gold Miners"},
    "SIL": {"group": "Equity", "subgroup": "Metals & Mining Equity"},
    "COPX": {"group": "Equity", "subgroup": "Metals & Mining Equity"},
    "XME": {"group": "Equity", "subgroup": "Metals & Mining Equity"},
    "PICK": {"group": "Equity", "subgroup": "Metals & Mining Equity"},
    "MXI": {"group": "Equity", "subgroup": "Materials Equity"},
    "IGE": {"group": "Equity", "subgroup": "Natural Resources Equity"},
    "DBC": {"group": "Commodity", "subgroup": "Broad Commodities"},
    "USO": {"group": "Commodity", "subgroup": "Energy"},
    "UNG": {"group": "Commodity", "subgroup": "Energy"},
    "DBA": {"group": "Commodity", "subgroup": "Agriculture"},
    "DBB": {"group": "Commodity", "subgroup": "Industrial Metals"},
    "CPER": {"group": "Commodity", "subgroup": "Industrial Metals"},
    "URA": {"group": "Equity", "subgroup": "Uranium / Nuclear Equity"},
    "KRBN": {"group": "Commodity", "subgroup": "Carbon Allowances"},
    "GRN": {"group": "Commodity", "subgroup": "Carbon / Green Commodity"},

    # Real-asset / infrastructure / natural resource equity
    "WOOD": {"group": "Equity", "subgroup": "Natural Resources Equity"},
    "CUT": {"group": "Equity", "subgroup": "Natural Resources Equity"},
    "MOO": {"group": "Equity", "subgroup": "Agriculture Equity"},
    "PHO": {"group": "Equity", "subgroup": "Water / Utilities Equity"},
    "FIW": {"group": "Equity", "subgroup": "Water / Utilities Equity"},
    "VPU": {"group": "Equity", "subgroup": "Utilities Equity"},
    "IDU": {"group": "Equity", "subgroup": "Utilities Equity"},

    # Crypto / alternative wrappers
    "BITO": {"group": "Crypto", "subgroup": "Bitcoin Futures"},
    "BTF": {"group": "Crypto", "subgroup": "Bitcoin Futures"},
    "WGMI": {"group": "Crypto", "subgroup": "Crypto Equity"},
    "BLOK": {"group": "Crypto", "subgroup": "Blockchain Equity"},
}


def get_asset_group_info(ticker: Any) -> Dict[str, str]:
    symbol = _normalize_ticker_symbol(ticker)
    if not symbol:
        return {"ticker": "", "group": _ASSET_GROUP_FALLBACK, "subgroup": _ASSET_GROUP_FALLBACK}
    meta = _ASSET_GROUP_MAP.get(symbol)
    if meta is not None:
        return {"ticker": symbol, "group": str(meta.get("group") or _ASSET_GROUP_FALLBACK), "subgroup": str(meta.get("subgroup") or _ASSET_GROUP_FALLBACK)}

    sector_prefixes = ("XL", "XL", "KR")
    if symbol.startswith(("XLK", "XLF", "XLV", "XLI", "XLP", "XLY", "XLE", "XLB", "XLU", "XLC", "XHB", "XRT")):
        return {"ticker": symbol, "group": "Equity", "subgroup": "Sector Equity"}
    if symbol in {"XLRE", "VNQ", "IYR", "RWR", "SCHH", "FREL"} or symbol.startswith("RE"):
        return {"ticker": symbol, "group": "Real Estate", "subgroup": "REIT / Real Estate"}
    if symbol.startswith(("EW", "IEFA", "IEMG")):
        return {"ticker": symbol, "group": "Equity", "subgroup": "International / Country Equity"}
    if symbol in {"VEA", "EFA", "VXUS", "ACWI", "VWO", "EEM", "VGK", "SCZ", "VSS", "GWX", "AAXJ", "FM"}:
        return {"ticker": symbol, "group": "Equity", "subgroup": "International / Country Equity"}
    if symbol.startswith(("SPY", "VOO", "IVV", "SCH", "VTI", "QQQ", "IWM", "IJ", "VB", "VU", "VT", "RSP")):
        return {"ticker": symbol, "group": "Equity", "subgroup": "Broad / Style Equity"}
    if symbol in {"TLT", "IEF", "IEI", "TIP", "STIP", "LTPZ", "TIPX", "LQD", "HYG", "JNK", "EMB", "BND", "AGG", "MUB", "BSV", "BIV", "BLV", "VCIT", "VCSH", "SJNK", "HYMB", "CWB", "BWX", "SCHZ", "MBB", "VMBS", "GNMA", "CMBS"}:
        subgroup = "Fixed Income"
        if symbol in {"TLT", "IEF", "IEI"}:
            subgroup = "Treasuries"
        elif symbol in {"TIP", "STIP", "LTPZ", "TIPX"}:
            subgroup = "Inflation-Linked Bonds"
        elif symbol in {"LQD", "VCIT", "VCSH"}:
            subgroup = "Investment Grade Credit"
        elif symbol in {"HYG", "JNK", "SJNK", "USHY", "ANGL"}:
            subgroup = "High Yield Credit"
        return {"ticker": symbol, "group": "Fixed Income", "subgroup": subgroup}
    if symbol in {"BIL", "SHV", "VGSH", "SGOV", "TFLO", "SCHO", "IGSB", "FLOT", "MINT", "JPST", "SHY"}:
        return {"ticker": symbol, "group": "Cash / Short Duration", "subgroup": "Cash / Ultra-Short"}
    if symbol in {"GLD", "IAU", "SLV", "PPLT", "PALL", "DBC", "USO", "UNG", "DBA", "DBB", "CPER", "KRBN", "GRN"}:
        return {"ticker": symbol, "group": "Commodity", "subgroup": "Commodity"}
    if symbol in {"BITO", "BTF", "WGMI"} or "BTC" in symbol:
        return {"ticker": symbol, "group": "Crypto", "subgroup": "Crypto"}
    if symbol.startswith(("ARK", "BOT", "ROB", "SKY", "CLOU", "HACK", "BUG")):
        return {"ticker": symbol, "group": "Equity", "subgroup": "Thematic Equity"}
    return {"ticker": symbol, "group": _ASSET_GROUP_FALLBACK, "subgroup": _ASSET_GROUP_FALLBACK}


def get_asset_group(ticker: Any) -> str:
    return str(get_asset_group_info(ticker).get("group") or _ASSET_GROUP_FALLBACK)


def get_asset_subgroup(ticker: Any) -> str:
    return str(get_asset_group_info(ticker).get("subgroup") or _ASSET_GROUP_FALLBACK)


def group_assets_by_class(tickers: Sequence[Any]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {k: [] for k in ASSET_GROUP_ORDER}
    seen: set[str] = set()
    for raw in list(tickers or []):
        symbol = _normalize_ticker_symbol(raw)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        group = get_asset_group(symbol)
        if group not in grouped:
            grouped[group] = []
        grouped[group].append(symbol)
    return {k: v for k, v in grouped.items() if v}


def build_asset_group_summary(tickers: Sequence[Any]) -> Dict[str, Any]:
    grouped = group_assets_by_class(tickers)
    flat = [t for members in grouped.values() for t in members]
    subgroup_counts: Dict[str, int] = {}
    subgroup_members: Dict[str, List[str]] = {}
    for symbol in flat:
        subgroup = get_asset_subgroup(symbol)
        subgroup_counts[subgroup] = int(subgroup_counts.get(subgroup, 0) + 1)
        subgroup_members.setdefault(subgroup, []).append(symbol)
    n_total = int(len(flat))
    n_classified = int(sum(len(v) for k, v in grouped.items() if k != _ASSET_GROUP_FALLBACK))
    return {
        "n_assets": n_total,
        "n_groups": int(len(grouped)),
        "classified_share": float(n_classified / n_total) if n_total > 0 else 0.0,
        "group_counts": {k: int(len(v)) for k, v in grouped.items()},
        "group_members": grouped,
        "subgroup_counts": dict(sorted(subgroup_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "subgroup_members": subgroup_members,
    }


def build_asset_group_dataframe(tickers: Sequence[Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for raw in list(tickers or []):
        symbol = _normalize_ticker_symbol(raw)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        meta = get_asset_group_info(symbol)
        rows.append({
            "ticker": symbol,
            "group": meta.get("group", _ASSET_GROUP_FALLBACK),
            "subgroup": meta.get("subgroup", _ASSET_GROUP_FALLBACK),
        })
    if not rows:
        return pd.DataFrame(columns=["ticker", "group", "subgroup"])
    out = pd.DataFrame(rows)
    out["group_sort"] = out["group"].map({g: i for i, g in enumerate(ASSET_GROUP_ORDER)}).fillna(len(ASSET_GROUP_ORDER))
    out = out.sort_values(["group_sort", "group", "subgroup", "ticker"]).drop(columns=["group_sort"]).reset_index(drop=True)
    return out


def aggregate_weights_by_asset_group(
    weights: pd.DataFrame,
    *,
    asset_col: str = "asset",
    weight_col: str = "weight",
    include_subgroup: bool = True,
) -> pd.DataFrame:
    if weights is None or not isinstance(weights, pd.DataFrame) or weights.empty:
        cols = ["group", "weight"]
        if include_subgroup:
            cols.insert(1, "subgroup")
        return pd.DataFrame(columns=cols)
    if asset_col not in weights.columns or weight_col not in weights.columns:
        raise KeyError(f"weights must contain columns {asset_col!r} and {weight_col!r}")

    base = weights[[asset_col, weight_col]].copy()
    base[asset_col] = base[asset_col].map(_normalize_ticker_symbol)
    base[weight_col] = pd.to_numeric(base[weight_col], errors="coerce")
    base = base.dropna(subset=[asset_col, weight_col])
    if base.empty:
        cols = ["group", "weight"]
        if include_subgroup:
            cols.insert(1, "subgroup")
        return pd.DataFrame(columns=cols)

    meta_df = pd.DataFrame([get_asset_group_info(t) for t in base[asset_col]])
    meta_df.index = base.index
    merged = pd.concat([base, meta_df[["group", "subgroup"]]], axis=1)
    group_cols = ["group", "subgroup"] if include_subgroup else ["group"]
    out = (
        merged.groupby(group_cols, dropna=False, as_index=False)[weight_col]
        .sum()
        .rename(columns={weight_col: "weight"})
    )
    out["group_sort"] = out["group"].map({g: i for i, g in enumerate(ASSET_GROUP_ORDER)}).fillna(len(ASSET_GROUP_ORDER))
    sort_cols = ["group_sort", "group"] + (["subgroup"] if include_subgroup else [])
    out = out.sort_values(sort_cols).drop(columns=["group_sort"]).reset_index(drop=True)
    return out


# ============================================================
# Dataclasses
# ============================================================

@dataclass(frozen=True)
class RiskProfileParams:
    profile: str
    mean_scale: float
    vol_scale: float
    annual_return: float
    annual_vol: float
    monthly_mean: float
    monthly_vol: float
    source: str


@dataclass(frozen=True)
class GrowthSummary:
    profile: str
    source: str
    horizon_months: int
    n_sims: int
    initial_invested: float
    monthly_contribution: float
    total_contributed: float
    expected_terminal: float
    median_terminal: float
    p10_terminal: float
    p90_terminal: float
    expected_profit: float
    median_profit: float
    p10_profit: float
    p90_profit: float
    probability_of_loss_vs_contributions: float
    probability_of_finishing_below_initial: float
    probability_of_reaching_goal: Optional[float]
    expected_max_drawdown: float
    drawdown_p10: float
    annual_return_assumption: float
    annual_vol_assumption: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MicroPipelineConfig:
    sigma_power_alpha: float = 1.0

    date_col: str = "date"
    asset_col: str = "asset"
    return_col: str = "return"

    # Closer-to-coursework defaults
    min_train: int = 120
    lookback_mu: int = 12
    lookback_sigma: int = 12
    sigma_floor: float = 0.02
    max_oos_points: Optional[int] = None
    enable_probabilistic: bool = True
    enable_diagnostics: bool = True

    # Allocation controls
    temperature: float = 1.0
    top_k: Optional[int] = None
    long_only: bool = True
    weight_shrink: float = 0.05
    inertia: float = 0.0
    deadband: bool = True
    deadband_threshold: float = 0.02
    asset_weight_cap: Optional[float] = None
    w_cap: Optional[float] = None
    vol_dependent_cap_enabled: bool = False
    vol_dependent_cap_threshold_low: float = 0.03
    vol_dependent_cap_threshold_high: float = 0.06
    vol_dependent_cap_low_mult: float = 1.15
    vol_dependent_cap_high_mult: float = 0.80
    vol_dependent_cap_min: Optional[float] = None
    vol_dependent_cap_max: Optional[float] = None
    corr_dependent_cap_enabled: bool = False
    corr_dependent_cap_threshold_low: float = 0.20
    corr_dependent_cap_threshold_high: float = 0.60
    corr_dependent_cap_low_mult: float = 1.10
    corr_dependent_cap_high_mult: float = 0.80
    corr_dependent_cap_min: Optional[float] = None
    corr_dependent_cap_max: Optional[float] = None
    dispersion_dependent_cap_enabled: bool = False
    dispersion_dependent_cap_threshold_low: float = 0.10
    dispersion_dependent_cap_threshold_high: float = 0.30
    dispersion_dependent_cap_low_mult: float = 1.10
    dispersion_dependent_cap_high_mult: float = 0.80
    dispersion_dependent_cap_min: Optional[float] = None
    dispersion_dependent_cap_max: Optional[float] = None
    regime_dependent_cap_enabled: bool = False
    regime_dependent_cap_low_mult: float = 1.05
    regime_dependent_cap_mid_mult: float = 0.95
    regime_dependent_cap_high_mult: float = 0.80
    regime_dependent_cap_min: Optional[float] = None
    regime_dependent_cap_max: Optional[float] = None

    # Turnover control
    turnover_penalty_strength: float = 0.0
    turnover_penalty_power: float = 1.0
    turnover_penalty_target: float = 0.20
    turnover_penalty_max_turnover: Optional[float] = None
    turnover_constraint_max_turnover: Optional[float] = None

    # Full cost / tax model (defaults OFF for backward compatibility)
    cost_model_enabled: bool = False
    transaction_cost_commission_bps: float = 0.0
    transaction_cost_slippage_bps: float = 0.0
    transaction_cost_spread_bps: float = 0.0
    transaction_cost_market_impact_bps: float = 0.0
    transaction_cost_market_impact_power: float = 1.0
    transaction_cost_min_trade_weight: float = 0.0
    holding_cost_annual_bps: float = 0.0
    tax_model_enabled: bool = False
    tax_short_term_rate: float = 0.0
    tax_long_term_rate: Optional[float] = None
    tax_long_term_threshold_months: int = 12
    tax_apply_loss_credit: bool = False
    tax_loss_credit_rate: Optional[float] = None

    # Dispersion gating
    dispersion_gate: bool = True
    dispersion_gate_threshold: float = 0.10
    dispersion_gate_min_active_weight: float = 0.25

    # Dispersion -> risk model
    dispersion_sigma_enabled: bool = False
    dispersion_sigma_lookback: int = 12
    dispersion_sigma_strength: float = 0.50
    dispersion_sigma_floor_mult: float = 0.75
    dispersion_sigma_ceiling_mult: float = 1.50
    dispersion_risk_model_enabled: bool = False
    dispersion_risk_strength: float = 0.50
    dispersion_risk_floor_mult: float = 0.85
    dispersion_risk_ceiling_mult: float = 1.25
    dispersion_risk_apply_to_covariance: bool = True
    dispersion_top_k_enabled: bool = False
    dispersion_top_k_threshold_low: float = 0.10
    dispersion_top_k_threshold_high: float = 0.30
    dispersion_top_k_low_mult: float = 1.50
    dispersion_top_k_high_mult: float = 0.75
    dispersion_top_k_min_k: Optional[int] = None
    dispersion_top_k_max_k: Optional[int] = None
    dispersion_vol_target_enabled: bool = False
    dispersion_vol_target_threshold_low: float = 0.10
    dispersion_vol_target_threshold_high: float = 0.30
    dispersion_vol_target_low_mult: float = 0.75
    dispersion_vol_target_high_mult: float = 1.15
    dispersion_vol_target_min: Optional[float] = None
    dispersion_vol_target_max: Optional[float] = None
    low_signal_fallback_to_ew: bool = False
    low_signal_fallback_threshold: float = 0.05
    low_signal_fallback_mode: Literal["hard", "blend"] = "blend"
    low_signal_fallback_min_model_weight: float = 0.25

    # Simple regime diagnostics / optional split mu estimation
    regime_mode: Literal["none", "quantile"] = "quantile"
    mu_regime_mode: Literal["pooled", "split"] = "pooled"
    regime_lookback: int = 24
    regime_quantile_low: float = 0.33
    regime_quantile_high: float = 0.67

    # Volatility estimator
    ewma_sigma: bool = True
    ewma_halflife: int = 6

    # Score shaping
    score_normalize: bool = True
    score_clip: Optional[float] = 4.0

    # Covariance / correlation controls
    covariance_mode: Literal["corr_sigma", "ewma_cov"] = "ewma_cov"
    regime_dependent_covariance: bool = True
    correlation_lookback: int = 24
    correlation_min_periods: int = 6
    correlation_shrink_to_identity: float = 0.10
    covariance_shrink_to_diagonal: float = 0.10
    covariance_jitter: float = 1e-8
    covariance_lookback_low: int = 18
    covariance_lookback_mid: int = 24
    covariance_lookback_high: int = 36
    covariance_halflife_low: int = 4
    covariance_halflife_mid: int = 6
    covariance_halflife_high: int = 9
    factor_covariance_active: bool = False
    factor_covariance_n_factors: int = 3
    factor_covariance_min_obs: int = 24
    factor_covariance_shrink_to_diagonal: float = 0.10
    factor_covariance_blend: float = 0.50
    factor_model_active: bool = False
    factor_model_n_factors: int = 3
    factor_model_min_obs: int = 24
    factor_model_mu_blend: float = 0.35
    factor_model_residual_blend: float = 0.50
    factor_model_covariance_blend: float = 0.60
    factor_model_shrink_to_diagonal: float = 0.10

    # Correlation-aware allocation
    correlation_aware_allocation: bool = True
    correlation_allocator_method: Literal["score_penalty", "mean_variance_light", "risk_budget"] = "score_penalty"
    correlation_allocator_blend: float = 0.35
    correlation_penalty_strength: float = 1.0
    correlation_penalty_power: float = 1.0
    correlation_use_abs: bool = True
    mean_variance_risk_aversion: float = 4.0
    risk_budget_strength: float = 0.5
    cluster_corr_threshold: float = 0.70

    # Vol targeting
    vol_targeting: bool = True
    covariance_aware_vol_targeting: bool = True
    target_portfolio_vol_monthly: float = 0.04
    vol_target_floor_mult: float = 0.5
    vol_target_ceiling_mult: float = 1.5

    # Regime derisk
    regime_derisk_low: float = 1.00
    regime_derisk_mid: float = 0.95
    regime_derisk_high: float = 0.80

    # Snapshots
    store_correlation_snapshots: bool = False
    store_sigma_fwd_snapshots: bool = False

    # Probabilistic overlay
    probabilistic_mode: Literal["none", "historical", "historical_by_regime", "historical_by_features", "parametric_feature_aware", "knn_historical", "feature_bucketed_historical", "quantile_regression", "hybrid"] = "none"
    probabilistic_q_low: float = 0.25
    probabilistic_q_high: float = 0.75
    probabilistic_min_obs: int = 12
    probabilistic_interval_penalty_weight: float = 0.25
    probabilistic_downside_penalty_weight: float = 0.25
    probabilistic_confidence_scale: float = 1.0
    probabilistic_confidence_min_mult: float = 0.75
    probabilistic_confidence_max_mult: float = 1.25
    probabilistic_overlay_strength: float = 1.0
    probabilistic_overlay_blend: float = 1.0
    probabilistic_qr_alpha: float = 0.0
    probabilistic_qr_solver: Literal["highs", "interior-point"] = "highs"
    probabilistic_qr_feature_cap: int = 8
    probabilistic_regime_aware: bool = False
    probabilistic_regime_min_obs: int = 12
    probabilistic_feature_filter_enabled: bool = False
    probabilistic_feature_filter_min_obs: int = 12
    probabilistic_feature_filter_k: Optional[int] = None
    probabilistic_knn_k: Optional[int] = None
    probabilistic_knn_min_obs: int = 12
    probabilistic_knn_distance_power: float = 2.0
    probabilistic_knn_weighted_quantiles: bool = True
    probabilistic_knn_weight_eps: float = 1e-6
    probabilistic_bucket_n_bins: int = 4
    probabilistic_bucket_min_obs: int = 12
    probabilistic_bucket_match_min_features: int = 2
    probabilistic_bucket_max_features: int = 6
    probabilistic_parametric_min_sigma: float = 1e-4
    probabilistic_parametric_max_sigma_mult: float = 3.0
    probabilistic_parametric_use_neighbor_weights: bool = True
    probabilistic_hybrid_weight: float = 0.50
    probabilistic_hybrid_min_qr_weight: float = 0.25
    probabilistic_hybrid_max_qr_weight: float = 0.75
    probabilistic_hybrid_use_confidence: bool = True

    # Signal model extensions
    signal_mode: Literal["mu_sigma", "huber_mu", "lambdarank_like", "lambdarank_real", "directional_classifier", "logistic_loss", "top_k_classifier", "quantile_loss"] = "mu_sigma"
    huber_delta: float = 1.0
    signal_score_blend: float = 1.0
    lambdarank_lookback: int = 12
    lambdarank_temperature: float = 1.0
    directional_classifier_lookback: int = 12
    directional_classifier_threshold: float = 0.50
    directional_classifier_confidence_scale: float = 1.0
    logistic_loss_lookback: int = 18
    logistic_loss_l2: float = 1e-3
    logistic_loss_threshold: float = 0.50
    logistic_loss_confidence_scale: float = 1.0
    quantile_loss_lookback: int = 18
    quantile_loss_q: float = 0.50
    quantile_loss_alpha: float = 0.0
    quantile_loss_confidence_scale: float = 1.0
    top_k_classifier_lookback: int = 12
    top_k_classifier_k: Optional[int] = None
    top_k_classifier_threshold: float = 0.50
    top_k_classifier_confidence_scale: float = 1.0
    lambdarank_real_lookback: int = 18
    lambdarank_real_temperature: float = 1.0
    lambdarank_real_gain_power: float = 1.0
    lambdarank_real_pair_power: float = 1.0
    lambdarank_real_l2: float = 0.0
    multi_loss_enabled: bool = False
    multi_loss_rank_weight: float = 0.40
    multi_loss_direction_weight: float = 0.30
    multi_loss_return_weight: float = 0.30
    multi_loss_logistic_weight: float = 0.0
    multi_loss_topk_weight: float = 0.0
    multi_loss_temperature: float = 1.0
    multi_loss_l2: float = 0.0

    # Feature-conditioned mu
    feature_mu_enabled: bool = True
    feature_mu_blend: float = 0.25
    feature_mu_k: int = 24
    feature_mu_min_obs: int = 12
    feature_mu_cols_max: int = 8
    feature_mu_scale_mode: Literal["none", "zscore", "vol_adjusted"] = "none"
    feature_mu_apply_quantile: Optional[float] = None
    feature_mu_rank_aware: bool = False
    feature_mu_sigma_epsilon: float = 1e-6

    # Regime-dependent universe selection
    regime_dependent_universe_enabled: bool = False
    regime_universe_selection_metric: Literal["mu", "score", "mu_over_sigma"] = "mu_over_sigma"
    regime_universe_keep_frac_low: float = 1.00
    regime_universe_keep_frac_mid: float = 0.85
    regime_universe_keep_frac_high: float = 0.60
    regime_universe_min_assets: int = 2
    regime_universe_max_assets: Optional[int] = None
    regime_universe_low_offensive_vol_tilt: float = 0.25
    regime_universe_high_defensive_vol_tilt: float = 0.75

    # Research profile
    pure_cs_baseline: bool = False

    # Selection / chooser contract (kept lightweight for persistence / replay)
    selection_policy: str = "fixed_composite_score"
    composite_profile: Optional[str] = None
    manual_composite_weights_enabled: bool = False
    manual_composite_weights: Optional[Dict[str, float]] = None


SIMPLE_STRATEGY_TEMPLATES = [
    "Core Ranking",
    "Balanced Risk-Controlled",
    "Hybrid Research",
]

SIMPLE_STYLE_PRESETS = [
    "Conservative",
    "Balanced",
    "Growth",
    "Defensive",
    "Research",
]


@dataclass(frozen=True)
class SimpleUISpec:
    strategy_template: str
    style_preset: str
    risk_appetite: float
    diversification: float
    stability: float
    turnover_pref: float
    drawdown_protection: float
    overlay_intensity: float
    signal_confidence: float
    simplicity: float
    universe_size: int


def _clip01(x: Any, default: float = 0.5) -> float:
    try:
        v = float(x)
    except Exception:
        v = default
    if not np.isfinite(v):
        v = default
    return float(np.clip(v, 0.0, 1.0))


def _detect_universe_bucket(n_assets: int) -> str:
    n = max(int(n_assets), 1)
    if n <= 15:
        return "small"
    if n <= 50:
        return "medium"
    return "large"


def _base_cfg_for_strategy_template(strategy_template: str) -> Dict[str, Any]:
    s = str(strategy_template or "Balanced Risk-Controlled")
    if s == "Core Ranking":
        return {
            "signal_mode": "mu_sigma",
            "probabilistic_mode": "none",
            "correlation_aware_allocation": False,
            "vol_targeting": False,
            "feature_mu_enabled": False,
            "factor_model_active": False,
            "factor_covariance_active": False,
            "regime_dependent_universe_enabled": False,
            "pure_cs_baseline": False,
        }
    if s == "Hybrid Research":
        return {
            "signal_mode": "lambdarank_like",
            "probabilistic_mode": "none",
            "correlation_aware_allocation": True,
            "vol_targeting": True,
            "feature_mu_enabled": True,
            "factor_model_active": False,
            "factor_covariance_active": False,
            "regime_dependent_universe_enabled": False,
            "pure_cs_baseline": False,
        }
    # Balanced Risk-Controlled
    return {
        "signal_mode": "mu_sigma",
        "probabilistic_mode": "none",
        "correlation_aware_allocation": True,
        "vol_targeting": True,
        "feature_mu_enabled": False,
        "factor_model_active": False,
        "factor_covariance_active": False,
        "regime_dependent_universe_enabled": False,
        "pure_cs_baseline": False,
    }


def _style_preset_overrides(style_preset: str) -> Dict[str, Any]:
    p = str(style_preset or "Balanced")
    if p == "Conservative":
        return {
            "target_portfolio_vol_monthly": 0.022,
            "regime_derisk_high": 0.68,
            "regime_derisk_mid": 0.88,
            "regime_derisk_low": 0.98,
            "probabilistic_mode": "historical_by_regime",
            "vol_targeting": True,
        }
    if p == "Growth":
        return {
            "target_portfolio_vol_monthly": 0.052,
            "regime_derisk_high": 0.92,
            "regime_derisk_mid": 0.99,
            "regime_derisk_low": 1.00,
            "probabilistic_mode": "historical",
            "vol_targeting": True,
        }
    if p == "Defensive":
        return {
            "target_portfolio_vol_monthly": 0.018,
            "regime_derisk_high": 0.60,
            "regime_derisk_mid": 0.82,
            "regime_derisk_low": 0.94,
            "probabilistic_mode": "historical_by_regime",
            "vol_targeting": True,
        }
    if p == "Research":
        return {
            "target_portfolio_vol_monthly": 0.042,
            "regime_derisk_high": 0.86,
            "regime_derisk_mid": 0.96,
            "regime_derisk_low": 1.00,
            "probabilistic_mode": "none",
            "vol_targeting": False,
        }
    return {
        "target_portfolio_vol_monthly": 0.035,
        "regime_derisk_high": 0.80,
        "regime_derisk_mid": 0.94,
        "regime_derisk_low": 1.00,
        "probabilistic_mode": "historical",
        "vol_targeting": True,
    }


def _resolve_simple_top_k(n_assets: int, style_preset: str, diversification: float) -> Optional[int]:
    n = max(int(n_assets or 0), 1)
    style = str(style_preset or "Balanced")
    if n <= 15:
        base_frac = 0.42
    elif n <= 50:
        base_frac = 0.30
    elif n <= 100:
        base_frac = 0.24
    else:
        base_frac = 0.18
    style_adj = {
        "Conservative": 0.08,
        "Defensive": 0.10,
        "Balanced": 0.02,
        "Growth": -0.05,
        "Research": -0.08,
    }.get(style, 0.0)
    div_adj = 0.18 * (float(diversification) - 0.5)
    frac = float(np.clip(base_frac + style_adj + div_adj, 0.10, 0.80))
    k = int(round(frac * n))
    k = int(np.clip(k, 2, max(n - 1, 2)))
    if k >= n:
        return None
    return k


def _resolve_simple_weight_shrink(style_preset: str, universe_bucket: str, diversification: float, simplicity: float) -> float:
    style = str(style_preset or "Balanced")
    base = {
        "Conservative": 0.18,
        "Defensive": 0.22,
        "Balanced": 0.12,
        "Growth": 0.06,
        "Research": 0.02,
    }.get(style, 0.12)
    bucket_adj = {"small": -0.03, "medium": 0.00, "large": 0.05}.get(str(universe_bucket), 0.0)
    value = base + bucket_adj + 0.10 * float(diversification) + 0.04 * float(simplicity)
    return float(np.clip(value, 0.0, 0.35))


def _resolve_simple_temperature(style_preset: str, universe_bucket: str, diversification: float, risk_appetite: float) -> float:
    style = str(style_preset or "Balanced")
    base = {
        "Conservative": 1.45,
        "Defensive": 1.60,
        "Balanced": 1.10,
        "Growth": 0.78,
        "Research": 0.70,
    }.get(style, 1.10)
    bucket_adj = {"small": -0.10, "medium": 0.00, "large": 0.15}.get(str(universe_bucket), 0.0)
    value = base + bucket_adj + 0.35 * float(diversification) - 0.25 * float(risk_appetite)
    return float(np.clip(value, 0.40, 2.00))


def _resolve_simple_overlay_mode(style_preset: str, strategy_template: str, overlay_intensity: float, signal_confidence: float) -> str:
    style = str(style_preset or "Balanced")
    template = str(strategy_template or "Balanced Risk-Controlled")
    intensity = float(overlay_intensity)
    confidence = float(signal_confidence)
    if style == "Research" or intensity <= 0.08:
        return "none"
    if template == "Hybrid Research" and intensity >= 0.45 and confidence >= 0.45:
        return "quantile_regression"
    if style in {"Conservative", "Defensive"}:
        return "historical_by_regime"
    if style == "Growth":
        return "historical"
    return "historical"


def resolve_simple_auto_composite_profile(style_preset: str) -> str:
    """
    Resolve the canonical fixed-composite chooser profile for Simple/Auto mode.

    This is intentionally tiny and stable so both app.py and downstream summaries
    can rely on one semantic mapping instead of open-coding it in multiple places.
    """
    raw = str(style_preset or "Balanced").strip().lower()
    key = raw.replace("-", "_").replace(" ", "_")
    if key in {"conservative", "defensive", "composite_defensive"}:
        return "defensive"
    if key in {"growth", "aggressive", "composite_growth"}:
        return "growth"
    if key in {"research", "robust", "composite_robust"}:
        return "growth"
    return "balanced"

def get_profile_objective_weights(profile_name: str) -> Dict[str, float]:
    """
    Return a small semantic objective-weight profile for downstream evaluators.

    This is intentionally *not* a hypervolume weighting scheme. The goal is to
    give local search / Pareto selection layers a simple and consistent hint
    about what to prioritise for broad profile families such as balanced,
    growth, and defensive.

    The weights are normalized to sum to 1.0 and emphasize a pragmatic set of
    objectives that already exist elsewhere in the tuning stack:
    - sharpe / quality of risk-adjusted return
    - cagr / growth tilt
    - max_drawdown / drawdown control
    - mean_turnover / implementation frictions
    - diversification / breadth / concentration control
    - stability / robustness / smoothness preference

    Accepted aliases are permissive on purpose so the UI and evaluator can pass
    names like "Balanced", "balanced", "composite_balanced", or
    "balanced_compromise" without extra glue code.
    """
    raw = str(profile_name or "balanced").strip().lower()
    key = raw.replace("-", "_").replace(" ", "_")

    alias_map = {
        "balanced": "balanced",
        "balanced_compromise": "balanced",
        "composite_balanced": "balanced",
        "growth": "growth",
        "composite_growth": "growth",
        "defensive": "defensive",
        "conservative": "defensive",
        "composite_defensive": "defensive",
        "research": "balanced",
        "robust": "balanced",
        "composite_robust": "balanced",
        "aggressive": "growth",
    }
    resolved = alias_map.get(key, "balanced")

    library: Dict[str, Dict[str, float]] = {
        "balanced": {
            "sharpe": 0.30,
            "cagr": 0.22,
            "max_drawdown": 0.20,
            "mean_turnover": 0.10,
            "diversification": 0.10,
            "stability": 0.08,
        },
        "growth": {
            "sharpe": 0.20,
            "cagr": 0.38,
            "max_drawdown": 0.14,
            "mean_turnover": 0.08,
            "diversification": 0.08,
            "stability": 0.12,
        },
        "defensive": {
            "sharpe": 0.28,
            "cagr": 0.12,
            "max_drawdown": 0.32,
            "mean_turnover": 0.12,
            "diversification": 0.08,
            "stability": 0.08,
        },
    }

    weights = dict(library[resolved])
    total = float(sum(max(float(v), 0.0) for v in weights.values()))
    if not np.isfinite(total) or total <= 0.0:
        return {"sharpe": 1.0}
    return {k: float(max(float(v), 0.0) / total) for k, v in weights.items()}



def resolve_simple_ui_tuning_policy(spec: SimpleUISpec) -> Dict[str, Any]:
    """
    Resolve a semantic local-search policy from the Simple UI preset.

    This does NOT execute refinement and does NOT build a concrete param space.
    It only returns search intent defaults that downstream UI / evaluation layers
    can consume more elegantly.
    """
    universe_bucket = _detect_universe_bucket(spec.universe_size)
    risk_appetite = _clip01(spec.risk_appetite)
    diversification = _clip01(spec.diversification)
    stability = _clip01(spec.stability)
    turnover_pref = _clip01(spec.turnover_pref)
    drawdown_protection = _clip01(spec.drawdown_protection)
    overlay_intensity = _clip01(spec.overlay_intensity)
    signal_confidence = _clip01(spec.signal_confidence)
    simplicity = _clip01(spec.simplicity)

    strategy_template = str(spec.strategy_template or "Balanced Risk-Controlled")
    style_preset = str(spec.style_preset or "Balanced")

    simple_auto_selection_policy = "fixed_composite_score"
    composite_profile = resolve_simple_auto_composite_profile(style_preset)

    objective = "composite_balanced"
    if style_preset in {"Conservative", "Defensive"}:
        objective = "composite_defensive"
    elif style_preset == "Growth":
        objective = "composite_growth"
    elif style_preset == "Research":
        objective = "composite_robust"

    if strategy_template == "Core Ranking" and style_preset not in {"Growth", "Conservative", "Defensive"}:
        objective = "composite_quality"
    elif strategy_template == "Hybrid Research" and style_preset == "Balanced":
        objective = "composite_robust"

    preferred_dims: List[str] = []

    # Core ranking / concentration controls
    preferred_dims.extend([
        "top_k",
        "temperature",
        "weight_shrink",
    ])

    # Stability / turnover controls
    if stability >= 0.45 or turnover_pref <= 0.55:
        preferred_dims.extend([
            "inertia",
            "deadband_threshold",
            "turnover_penalty_strength",
            "turnover_constraint_max_turnover",
        ])
    else:
        preferred_dims.extend([
            "turnover_penalty_strength",
            "turnover_penalty_target",
        ])

    # Risk / drawdown controls
    if drawdown_protection >= 0.40 or style_preset in {"Conservative", "Defensive"}:
        preferred_dims.extend([
            "target_portfolio_vol_monthly",
            "correlation_penalty_strength",
        ])
    elif risk_appetite >= 0.60 or style_preset == "Growth":
        preferred_dims.extend([
            "target_portfolio_vol_monthly",
            "top_k",
            "temperature",
        ])

    # Note: correlation_allocator_blend is intentionally excluded from the
    # Simple-mode neighbourhood. We keep only the penalty-strength knob here so
    # the auto-opt remains predictable and avoids more research-like allocator
    # reshaping in v1.

    # Overlay / confidence controls (kept intentionally narrow for Simple mode)
    if overlay_intensity > 0.10:
        preferred_dims.extend([
            "probabilistic_overlay_strength",
        ])

    # Extra flexibility for research-like presets is intentionally suppressed in
    # Simple mode v1. We keep the auto-opt neighbourhood focused on robust,
    # user-understandable knobs rather than research-oriented structural changes.

    # Dedupe preserving order and keep only known config fields.
    valid_cfg_fields = {f.name for f in fields(MicroPipelineConfig)}
    preferred_dims = [d for i, d in enumerate(preferred_dims) if d in valid_cfg_fields and d not in preferred_dims[:i]]

    # Budget / stage-2 bias
    if universe_bucket == "small":
        stage2_top_n = 2
        stage2_max_pairs = 4
        max_candidates = 24
    elif universe_bucket == "medium":
        stage2_top_n = 3
        stage2_max_pairs = 6
        max_candidates = 36
    else:
        stage2_top_n = 3
        stage2_max_pairs = 8
        max_candidates = 48

    if strategy_template == "Hybrid Research" or style_preset == "Research":
        stage2_top_n = max(stage2_top_n, 3)
        stage2_max_pairs += 2
        max_candidates += 12

    if style_preset in {"Conservative", "Defensive"}:
        stage2_bias = "risk_control"
    elif style_preset == "Growth":
        stage2_bias = "growth_tilt"
    elif style_preset == "Research" or strategy_template == "Hybrid Research":
        stage2_bias = "robustness"
    else:
        stage2_bias = "balanced"

    stage2_include_seed_triplets = bool(
        (strategy_template == "Hybrid Research")
        or (style_preset == "Research")
        or (simplicity <= 0.30 and universe_bucket != "small")
    )

    two_stage_search = False
    refine_from_best_stage1 = False

    intensity_score = float(np.clip(
        0.20 * (1.0 - simplicity)
        + 0.20 * overlay_intensity
        + 0.20 * drawdown_protection
        + 0.20 * abs(risk_appetite - 0.5) * 2.0
        + 0.20 * (1.0 if strategy_template == "Hybrid Research" else 0.5),
        0.0,
        1.0,
    ))
    if intensity_score >= 0.67:
        search_budget = "expanded"
    elif intensity_score >= 0.34:
        search_budget = "standard"
    else:
        search_budget = "light"

    if search_budget == "light":
        stage2_top_n = min(stage2_top_n, 2)
        stage2_max_pairs = min(stage2_max_pairs, 4)
        max_candidates = min(max_candidates, 24)
        stage2_include_seed_triplets = False
    elif search_budget == "expanded":
        stage2_top_n += 1 if universe_bucket != "small" else 0
        stage2_max_pairs += 2
        max_candidates += 12

    return {
        "strategy_template": strategy_template,
        "style_preset": style_preset,
        "universe_bucket": universe_bucket,
        "simple_auto_selection_policy": simple_auto_selection_policy,
        "selection_policy": simple_auto_selection_policy,
        "composite_profile": composite_profile,
        "simple_auto_composite_profile": composite_profile,
        "objective": objective,
        "preferred_dims": preferred_dims,
        "two_stage_search": two_stage_search,
        "refine_from_best_stage1": refine_from_best_stage1,
        "stage2_bias": stage2_bias,
        "stage2_top_n": int(stage2_top_n),
        "stage2_max_pairs": int(stage2_max_pairs),
        "stage2_include_seed_triplets": bool(stage2_include_seed_triplets),
        "search_budget": search_budget,
        "max_candidates_hint": int(max_candidates),
    }



def resolve_simple_ui_to_micro_cfg(
    spec: SimpleUISpec,
) -> Tuple[MicroPipelineConfig, Dict[str, Any]]:
    """Resolve the Simple UI semantic preset into a concrete MicroPipelineConfig.

    This is intentionally conservative: it only maps the current Simple-mode
    controls into a coherent base config and applies a small UX stabilisation so
    the default preset does not start in an obviously stretched state.
    """
    if not isinstance(spec, SimpleUISpec):
        raise TypeError("spec must be a SimpleUISpec")

    universe_bucket = _detect_universe_bucket(int(spec.universe_size or 0))
    risk_appetite = _clip01(spec.risk_appetite)
    diversification = _clip01(spec.diversification)
    stability = _clip01(spec.stability)
    turnover_pref = _clip01(spec.turnover_pref)
    drawdown_protection = _clip01(spec.drawdown_protection)
    overlay_intensity = _clip01(spec.overlay_intensity)
    signal_confidence = _clip01(spec.signal_confidence)
    simplicity = _clip01(spec.simplicity)

    base = dict(_base_cfg_for_strategy_template(spec.strategy_template))
    base.update(_style_preset_overrides(spec.style_preset))

    style_name = str(spec.style_preset or "Balanced")
    template_name = str(spec.strategy_template or "Balanced Risk-Controlled")

    top_k = _resolve_simple_top_k(int(spec.universe_size or 0), style_name, diversification)
    if top_k is not None:
        base["top_k"] = int(top_k)

    base["weight_shrink"] = _resolve_simple_weight_shrink(style_name, universe_bucket, diversification, simplicity)
    base["temperature"] = _resolve_simple_temperature(style_name, universe_bucket, diversification, risk_appetite)
    base["probabilistic_mode"] = _resolve_simple_overlay_mode(style_name, template_name, overlay_intensity, signal_confidence)

    if template_name == "Hybrid Research":
        signal_mode = "lambdarank_like" if simplicity <= 0.70 else "mu_sigma"
    elif style_name in {"Conservative", "Defensive"}:
        signal_mode = "huber_mu"
    else:
        signal_mode = "mu_sigma"
    base["signal_mode"] = signal_mode

    base["feature_mu_enabled"] = bool(template_name == "Hybrid Research" or signal_confidence <= 0.45)
    base["regime_dependent_universe_enabled"] = bool(
        int(spec.universe_size or 0) >= 50 and (drawdown_protection >= 0.65 or style_name in {"Conservative", "Defensive"})
    )

    overlay_strength = 0.45 + 0.45 * overlay_intensity
    if style_name in {"Conservative", "Defensive"}:
        overlay_strength += 0.05
    elif style_name == "Research":
        overlay_strength -= 0.10
    if base.get("probabilistic_mode") == "none":
        overlay_strength = 0.0
    base["probabilistic_overlay_strength"] = float(np.clip(overlay_strength, 0.0, 0.90))
    base["probabilistic_overlay_blend"] = float(np.clip(0.70 + 0.20 * overlay_intensity, 0.55, 0.90))

    base["dispersion_gate"] = True
    base["dispersion_gate_threshold"] = float(np.clip(0.09 + 0.06 * (0.5 - risk_appetite) + 0.04 * drawdown_protection, 0.06, 0.16))

    correlation_penalty = 0.20 + 0.70 * drawdown_protection + 0.15 * diversification
    if style_name in {"Conservative", "Defensive"}:
        correlation_penalty += 0.10
    elif style_name == "Growth":
        correlation_penalty -= 0.10
    base["correlation_penalty_strength"] = float(np.clip(correlation_penalty, 0.10, 0.95))

    if bool(base.get("vol_targeting", False)):
        current_target_vol = float(base.get("target_portfolio_vol_monthly", 0.04) or 0.04)
        target_vol = current_target_vol
        target_vol += 0.010 * (risk_appetite - 0.5)
        target_vol -= 0.008 * drawdown_protection
        target_vol -= 0.004 * stability
        if style_name in {"Conservative", "Defensive"}:
            target_vol = min(target_vol, 0.028)
        elif style_name == "Balanced":
            target_vol = min(target_vol, 0.040)
        else:
            target_vol = min(target_vol, 0.055)
        target_vol = max(target_vol, 0.018 if style_name in {"Conservative", "Defensive"} else 0.024)
        base["target_portfolio_vol_monthly"] = float(target_vol)

    # Keep the base presets operationally calm by default. The app can still
    # move into more aggressive regions via manual overrides or tuning.
    adaptive_bias = turnover_pref
    stabiliser = 0.5 * stability + 0.5 * drawdown_protection
    if style_name == "Defensive":
        # Defensive defaults should start in a calm operational zone so
        # governance does not flag the untouched base preset as stretched.
        turnover_penalty_strength = 0.12 + 0.05 * stabiliser + 0.03 * (1.0 - adaptive_bias)
        turnover_constraint_max_turnover = 0.44 - 0.02 * stabiliser
        inertia = 0.12 + 0.12 * stability + 0.03 * (1.0 - adaptive_bias)
    elif style_name == "Conservative":
        turnover_penalty_strength = 0.14 + 0.06 * stabiliser + 0.04 * (1.0 - adaptive_bias)
        turnover_constraint_max_turnover = 0.43 - 0.02 * stabiliser
        inertia = 0.10 + 0.12 * stability + 0.04 * (1.0 - adaptive_bias)
    elif style_name == "Growth":
        turnover_penalty_strength = 0.08 + 0.06 * stability + 0.04 * (1.0 - adaptive_bias)
        turnover_constraint_max_turnover = 0.45
        inertia = 0.02 + 0.06 * stability
    elif style_name == "Research":
        turnover_penalty_strength = 0.10 + 0.05 * stability + 0.03 * (1.0 - adaptive_bias)
        turnover_constraint_max_turnover = 0.45
        inertia = 0.03 + 0.05 * stability
    else:
        turnover_penalty_strength = 0.14 + 0.08 * stabiliser + 0.04 * (1.0 - adaptive_bias)
        turnover_constraint_max_turnover = 0.44 - 0.02 * stabiliser
        inertia = 0.05 + 0.08 * stability + 0.02 * (1.0 - adaptive_bias)

    base["turnover_penalty_strength"] = float(np.clip(turnover_penalty_strength, 0.06, 0.26))
    base["turnover_constraint_max_turnover"] = float(np.clip(turnover_constraint_max_turnover, 0.32, 0.45))

    if style_name == "Defensive":
        # Coherence currently classifies turnover_control as medium when the
        # turnover ceiling is <= 0.40. Keep untouched Defensive presets above
        # that boundary so the base preset starts green by default.
        base["turnover_penalty_strength"] = float(min(base.get("turnover_penalty_strength", 0.20) or 0.20, 0.20))
        base["turnover_constraint_max_turnover"] = float(max(base.get("turnover_constraint_max_turnover", 0.42) or 0.42, 0.41))
    base["turnover_penalty_target"] = float(np.clip(base.get("turnover_penalty_target", 0.20) or 0.20, 0.12, 0.25))
    base["inertia"] = float(np.clip(inertia, 0.00, 0.18))
    base["deadband"] = bool(stability >= 0.45 or drawdown_protection >= 0.55)
    base["deadband_threshold"] = float(np.clip(0.008 + 0.015 * stability + 0.005 * drawdown_protection, 0.008, 0.020))

    tuning_policy = resolve_simple_ui_tuning_policy(spec)
    base["selection_policy"] = str(tuning_policy.get("selection_policy", "fixed_composite_score") or "fixed_composite_score")
    base["composite_profile"] = str(tuning_policy.get("composite_profile", "balanced") or "balanced")
    base["manual_composite_weights_enabled"] = False
    base["manual_composite_weights"] = None

    cfg_fields = {f.name for f in fields(MicroPipelineConfig)}
    cfg = MicroPipelineConfig(**{k: v for k, v in base.items() if k in cfg_fields})

    summary = {
        "strategy_template": spec.strategy_template,
        "style_preset": spec.style_preset,
        "universe_bucket": universe_bucket,
        "resolved_simple_auto_selection_policy": tuning_policy.get("simple_auto_selection_policy", "fixed_composite_score"),
        "resolved_simple_auto_composite_profile": tuning_policy.get("composite_profile", "balanced"),
        "resolved_selection_policy": cfg.selection_policy,
        "resolved_composite_profile": cfg.composite_profile,
        "resolved_manual_composite_weights_enabled": bool(cfg.manual_composite_weights_enabled),
        "resolved_manual_composite_weights": cfg.manual_composite_weights,
        "resolved_top_k": cfg.top_k,
        "resolved_signal_mode": cfg.signal_mode,
        "resolved_feature_mu_enabled": bool(cfg.feature_mu_enabled),
        "resolved_regime_dependent_universe_enabled": bool(cfg.regime_dependent_universe_enabled),
        "resolved_probabilistic_mode": cfg.probabilistic_mode,
        "resolved_probabilistic_overlay_strength": cfg.probabilistic_overlay_strength,
        "resolved_probabilistic_overlay_blend": cfg.probabilistic_overlay_blend,
        "resolved_dispersion_gate": bool(cfg.dispersion_gate),
        "resolved_dispersion_gate_threshold": cfg.dispersion_gate_threshold,
        "resolved_temperature": cfg.temperature,
        "resolved_weight_shrink": cfg.weight_shrink,
        "resolved_correlation_penalty_strength": cfg.correlation_penalty_strength,
        "resolved_vol_targeting": cfg.vol_targeting,
        "resolved_turnover_penalty_strength": cfg.turnover_penalty_strength,
        "resolved_target_portfolio_vol_monthly": cfg.target_portfolio_vol_monthly,
        "resolved_base_config": config_to_dict(cfg),
        "tuning_policy": tuning_policy,
        "resolved_tuning_objective": tuning_policy.get("objective"),
        "resolved_tuning_preferred_dims": tuning_policy.get("preferred_dims", []),
        "resolved_tuning_two_stage_search": bool(tuning_policy.get("two_stage_search", True)),
        "resolved_tuning_stage2_bias": tuning_policy.get("stage2_bias"),
        "resolved_tuning_search_budget": tuning_policy.get("search_budget"),
    }

    return cfg, summary


# ============================================================
# Philosophy-aware config normalization (5C / 5D)
# ============================================================

def _coerce_philosophy_name_for_simple_flow(value: Any) -> str:
    raw = str(value or "Balanced").strip().capitalize()
    return raw if raw in {"Growth", "Balanced", "Defensive"} else "Balanced"


def _caps_strength_to_cap_value(strength: Any) -> float | None:
    raw = str(strength or "").strip().lower()
    if raw == "strong":
        return 0.10
    if raw == "medium":
        return 0.20
    if raw == "soft":
        return 0.30
    return None


def _normalise_top_k_to_range(current_top_k: Any, top_k_range: Any) -> int | None:
    if not isinstance(top_k_range, (list, tuple)) or len(top_k_range) != 2:
        return current_top_k if current_top_k is None else int(current_top_k)
    try:
        low_k = max(int(top_k_range[0]), 2)
        high_k = max(int(top_k_range[1]), low_k)
    except Exception:
        return current_top_k if current_top_k is None else int(current_top_k)
    if current_top_k is None:
        midpoint = int(round((low_k + high_k) / 2.0))
        return int(np.clip(midpoint, low_k, high_k))
    try:
        return int(np.clip(int(current_top_k), low_k, high_k))
    except Exception:
        midpoint = int(round((low_k + high_k) / 2.0))
        return int(np.clip(midpoint, low_k, high_k))


def generate_coherent_config(
    philosophy: str,
    base_cfg: MicroPipelineConfig,
    constraints: Dict[str, Any] | None = None,
) -> Tuple[MicroPipelineConfig, Dict[str, Any]]:
    """
    Normalize a resolved Simple-mode config into a philosophy-coherent region.

    This is intentionally lightweight and non-destructive:
    - reuse the resolved base config
    - only clamp / correct / complete a few structural fields
    - do not execute the engine
    - keep backward-compatible defaults whenever already coherent
    """
    philosophy_name = _coerce_philosophy_name_for_simple_flow(philosophy)
    constraints_map = dict(constraints or {})
    payload = config_to_dict(base_cfg if isinstance(base_cfg, MicroPipelineConfig) else MicroPipelineConfig())
    changes: Dict[str, Dict[str, Any]] = {}

    def _set_field(name: str, value: Any) -> None:
        old = payload.get(name)
        if old != value:
            changes[str(name)] = {"old": old, "new": value}
            payload[name] = value

    top_k_range = constraints_map.get("top_k_range")
    new_top_k = _normalise_top_k_to_range(payload.get("top_k"), top_k_range)
    if new_top_k is not None:
        _set_field("top_k", int(new_top_k))

    allowed_overlay_modes = [str(x) for x in list(constraints_map.get("allowed_overlay_modes", [])) if str(x)]
    preferred_overlay_modes = [str(x) for x in list(constraints_map.get("preferred_overlay_modes", [])) if str(x)]
    current_overlay = str(payload.get("probabilistic_mode", "none") or "none")
    if allowed_overlay_modes and current_overlay not in allowed_overlay_modes:
        fallback_overlay = preferred_overlay_modes[0] if preferred_overlay_modes else allowed_overlay_modes[0]
        _set_field("probabilistic_mode", fallback_overlay)
        current_overlay = str(fallback_overlay)
    if philosophy_name == "Defensive":
        defensive_overlay = "historical_by_regime" if "historical_by_regime" in allowed_overlay_modes else (preferred_overlay_modes[0] if preferred_overlay_modes else current_overlay)
        _set_field("probabilistic_mode", defensive_overlay)
        _set_field("probabilistic_downside_penalty_weight", max(float(payload.get("probabilistic_downside_penalty_weight", 0.0) or 0.0), 0.35))
        _set_field("probabilistic_interval_penalty_weight", max(float(payload.get("probabilistic_interval_penalty_weight", 0.0) or 0.0), 0.20))
    elif philosophy_name == "Growth" and current_overlay == "historical_by_regime" and "historical" in allowed_overlay_modes:
        _set_field("probabilistic_mode", "historical")

    preferred_signal_modes = [str(x) for x in list(constraints_map.get("preferred_signal_modes", [])) if str(x)]
    current_signal = str(payload.get("signal_mode", "mu_sigma") or "mu_sigma")
    if preferred_signal_modes and current_signal not in preferred_signal_modes:
        _set_field("signal_mode", preferred_signal_modes[0])

    covariance_required = bool(constraints_map.get("covariance_required", False))
    preferred_covariance_models = [str(x) for x in list(constraints_map.get("preferred_covariance_models", [])) if str(x)]
    current_cov_mode = str(payload.get("covariance_mode", "ewma_cov") or "ewma_cov")
    if covariance_required:
        _set_field("vol_targeting", True)
        _set_field("covariance_aware_vol_targeting", True)
        _set_field("correlation_aware_allocation", True)
        if current_cov_mode not in {"corr_sigma", "ewma_cov"}:
            _set_field("covariance_mode", "ewma_cov")
        if philosophy_name == "Defensive":
            _set_field("regime_mode", "quantile")
            _set_field("regime_dependent_covariance", True)
            _set_field("correlation_penalty_strength", max(float(payload.get("correlation_penalty_strength", 0.0) or 0.0), 1.0))
            _set_field("target_portfolio_vol_monthly", min(float(payload.get("target_portfolio_vol_monthly", 0.04) or 0.04), 0.025))
        elif philosophy_name == "Balanced":
            _set_field("correlation_penalty_strength", max(float(payload.get("correlation_penalty_strength", 0.0) or 0.0), 0.6))
    elif philosophy_name == "Growth":
        _set_field("target_portfolio_vol_monthly", max(float(payload.get("target_portfolio_vol_monthly", 0.04) or 0.04), 0.035))

    caps_strength = str(constraints_map.get("caps_strength", "") or "")
    cap_value = _caps_strength_to_cap_value(caps_strength)
    current_cap = payload.get("asset_weight_cap")
    try:
        current_cap_float = float(current_cap) if current_cap is not None else None
    except Exception:
        current_cap_float = None
    if cap_value is not None and (current_cap_float is None or current_cap_float > cap_value):
        _set_field("asset_weight_cap", float(cap_value))
        _set_field("w_cap", float(cap_value))
    elif cap_value is not None and payload.get("w_cap") is None:
        _set_field("w_cap", float(cap_value))

    turnover_tolerance = str(constraints_map.get("turnover_tolerance", "medium") or "medium").lower()
    if turnover_tolerance == "low":
        # Important UX fix:
        # coherence.py currently labels turnover_control as "medium" when
        # turnover_penalty_strength >= 0.30 or the turnover ceiling <= 0.40.
        # For untouched Defensive / low-turnover-tolerance presets we want the
        # governed base to remain in the calm / green zone by default, so keep
        # the normalized values below those boundaries instead of tightening them
        # into a structurally stretched state.
        current_strength = float(payload.get("turnover_penalty_strength", 0.0) or 0.0)
        current_limit = float(payload.get("turnover_constraint_max_turnover", 0.45) or 0.45)
        _set_field("turnover_penalty_strength", float(np.clip(max(current_strength, 0.12), 0.12, 0.20)))
        _set_field("turnover_constraint_max_turnover", float(np.clip(max(current_limit, 0.41), 0.41, 0.50)))
        _set_field("inertia", max(float(payload.get("inertia", 0.0) or 0.0), 0.10))
        _set_field("deadband", True)
        _set_field("deadband_threshold", max(float(payload.get("deadband_threshold", 0.0) or 0.0), 0.02))
    elif turnover_tolerance == "medium":
        _set_field("turnover_penalty_strength", max(float(payload.get("turnover_penalty_strength", 0.0) or 0.0), 0.20))
        _set_field("turnover_constraint_max_turnover", min(float(payload.get("turnover_constraint_max_turnover", 0.50) or 0.50), 0.45))
    else:
        _set_field("turnover_penalty_strength", min(float(payload.get("turnover_penalty_strength", 0.0) or 0.0), 0.50))

    concentration_level = str(constraints_map.get("concentration_level", "medium") or "medium").lower()
    if concentration_level == "high":
        _set_field("weight_shrink", min(float(payload.get("weight_shrink", 0.05) or 0.05), 0.10))
        _set_field("temperature", min(float(payload.get("temperature", 1.0) or 1.0), 1.0))
    elif concentration_level == "low":
        _set_field("weight_shrink", max(float(payload.get("weight_shrink", 0.05) or 0.05), 0.12))
        _set_field("temperature", max(float(payload.get("temperature", 1.0) or 1.0), 1.10))
    else:
        _set_field("weight_shrink", min(max(float(payload.get("weight_shrink", 0.05) or 0.05), 0.05), 0.18))

    cfg_fields = {f.name for f in fields(MicroPipelineConfig)}
    coherent_cfg = MicroPipelineConfig(**{k: v for k, v in payload.items() if k in cfg_fields})
    meta = {
        "philosophy": philosophy_name,
        "constraints": constraints_map,
        "changes": changes,
        "changed": bool(changes),
        "n_changes": int(len(changes)),
    }
    return coherent_cfg, meta


def resolve_simple_ui_with_philosophy(
    spec: SimpleUISpec,
) -> Tuple[MicroPipelineConfig, Dict[str, Any], Dict[str, Any]]:
    """
    Extended Simple UI resolution that exposes philosophy-driven constraints
    and returns a final coherent config.

    Flow:
        spec -> resolve_simple_ui_to_micro_cfg -> constraints -> coherent config
    """
    cfg_base, summary = resolve_simple_ui_to_micro_cfg(spec)

    try:
        from src.coherence import resolve_philosophy_to_constraints
    except Exception:
        return cfg_base, summary, {}

    philosophy = _coerce_philosophy_name_for_simple_flow(spec.style_preset)
    try:
        constraints = resolve_philosophy_to_constraints(
            philosophy=philosophy,
            universe=spec.universe_size,
            strategy_template=spec.strategy_template,
            style_preset=spec.style_preset,
        )
    except Exception:
        constraints = {}

    coherent_cfg, coherence_meta = generate_coherent_config(
        philosophy=philosophy,
        base_cfg=cfg_base,
        constraints=constraints,
    )
    coherent_summary = dict(summary or {})
    coherent_summary["resolved_base_config"] = config_to_dict(cfg_base)
    coherent_summary["resolved_coherent_config"] = config_to_dict(coherent_cfg)
    coherent_summary["resolved_constraints"] = dict(constraints or {})
    coherent_summary["resolved_philosophy"] = philosophy
    coherent_summary["resolved_coherence_normalization"] = dict(coherence_meta or {})
    coherent_summary["resolved_coherence_adjusted"] = bool((coherence_meta or {}).get("changed", False))
    coherent_summary["resolved_coherence_adjustment_count"] = int((coherence_meta or {}).get("n_changes", 0))
    return coherent_cfg, coherent_summary, dict(constraints or {})


@dataclass(frozen=True)
class GovernedConfigResolutionResult:
    philosophy: str
    status: str
    base_intention: Dict[str, Any]
    constraints: Dict[str, Any]
    base_cfg: MicroPipelineConfig
    coherent_base_cfg: MicroPipelineConfig
    final_cfg: MicroPipelineConfig
    overrides_applied: Dict[str, Any]
    recommendation_patch_applied: Dict[str, Any]
    repair_patch_applied: Dict[str, Any]
    coherence: Dict[str, Any]
    repairs: Dict[str, Any]
    trace: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "philosophy": self.philosophy,
            "status": self.status,
            "base_intention": _json_safe(self.base_intention),
            "constraints": _json_safe(self.constraints),
            "base_cfg": _json_safe(config_to_dict(self.base_cfg)),
            "coherent_base_cfg": _json_safe(config_to_dict(self.coherent_base_cfg)),
            "final_cfg": _json_safe(config_to_dict(self.final_cfg)),
            "overrides_applied": _json_safe(self.overrides_applied),
            "recommendation_patch_applied": _json_safe(self.recommendation_patch_applied),
            "repair_patch_applied": _json_safe(self.repair_patch_applied),
            "coherence": _json_safe(self.coherence),
            "repairs": _json_safe(self.repairs),
            "trace": _json_safe(self.trace),
        }

    def to_governance_payload(self, *, results: Any = None) -> GovernancePayload:
        merged_overrides: Dict[str, Any] = {}
        for patch in [self.overrides_applied, self.recommendation_patch_applied, self.repair_patch_applied]:
            merged_overrides.update(coerce_candidate_payload_for_config(_coerce_mapping_for_governance(patch)))
        return build_standard_governance_payload(
            philosophy=self.philosophy,
            base_intention=self.base_intention,
            coherent_base_cfg=self.coherent_base_cfg,
            final_cfg=self.final_cfg,
            overrides_applied=merged_overrides,
            coherence=self.coherence,
            repairs=self.repairs,
            trace=self.trace,
            results=results,
        )


@dataclass(frozen=True)
class GovernancePayload:
    philosophy_effective: str
    base_intention: Dict[str, Any]
    coherent_base_cfg: Dict[str, Any]
    overrides_applied: Dict[str, Any]
    final_effective_cfg: Dict[str, Any]
    coherence_score: Optional[float]
    coherence_status: str
    warnings: List[str]
    suggested_repairs: Dict[str, Any]
    short_tradeoff_explanation: str
    actionable_recommendations: List[Dict[str, Any]]
    trace: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "philosophy_effective": self.philosophy_effective,
            "base_intention": _json_safe(self.base_intention),
            "coherent_base_cfg": _json_safe(self.coherent_base_cfg),
            "overrides_applied": _json_safe(self.overrides_applied),
            "final_effective_cfg": _json_safe(self.final_effective_cfg),
            "coherence_score": _json_safe(self.coherence_score),
            "coherence_status": self.coherence_status,
            "warnings": _json_safe(self.warnings),
            "suggested_repairs": _json_safe(self.suggested_repairs),
            "short_tradeoff_explanation": self.short_tradeoff_explanation,
            "actionable_recommendations": _json_safe(self.actionable_recommendations),
            "trace": _json_safe(self.trace),
        }


@dataclass(frozen=True)
class GovernedOverrideResult:
    philosophy: str
    status: str
    label: str
    base_coherent_cfg: MicroPipelineConfig
    final_cfg: MicroPipelineConfig
    manual_overrides_applied: Dict[str, Any]
    coherence_before: Dict[str, Any]
    coherence_after: Dict[str, Any]
    repairs: Dict[str, Any]
    warnings: List[str]
    suggested_repair_patch: Dict[str, Any]
    trace: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "philosophy": self.philosophy,
            "status": self.status,
            "label": self.label,
            "base_coherent_cfg": _json_safe(config_to_dict(self.base_coherent_cfg)),
            "final_cfg": _json_safe(config_to_dict(self.final_cfg)),
            "manual_overrides_applied": _json_safe(self.manual_overrides_applied),
            "coherence_before": _json_safe(self.coherence_before),
            "coherence_after": _json_safe(self.coherence_after),
            "repairs": _json_safe(self.repairs),
            "warnings": _json_safe(self.warnings),
            "suggested_repair_patch": _json_safe(self.suggested_repair_patch),
            "trace": _json_safe(self.trace),
        }


def _coerce_mapping_for_governance(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, pd.Series):
        return value.to_dict()
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




def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def apply_config_patch(base_cfg: MicroPipelineConfig, patch: Optional[Dict[str, Any]] = None) -> MicroPipelineConfig:
    """Apply a shallow patch onto a MicroPipelineConfig safely.

    Only known MicroPipelineConfig fields are accepted. Unknown keys are ignored
    to preserve backward compatibility with partially-built governance patches.
    """
    payload = config_to_dict(base_cfg if isinstance(base_cfg, MicroPipelineConfig) else MicroPipelineConfig())
    patch_map = _coerce_mapping_for_governance(patch)
    if not patch_map:
        return base_cfg if isinstance(base_cfg, MicroPipelineConfig) else MicroPipelineConfig(**payload)
    valid_fields = {f.name for f in fields(MicroPipelineConfig)}
    for k, v in patch_map.items():
        key = str(k)
        if key in valid_fields:
            payload[key] = v
    return MicroPipelineConfig(**{k: v for k, v in payload.items() if k in valid_fields})


def _build_governed_status(coherence_result: Dict[str, Any], repair_plan: Dict[str, Any]) -> str:
    coherence = dict(coherence_result or {})
    repairs = dict(repair_plan or {})
    explicit = str(repairs.get("status", coherence.get("status", "")) or "").strip().lower()
    if explicit in {"incompatible", "discouraged", "repairable", "ok", "coherent", "stretched", "auto_repair_available"}:
        if explicit == "incompatible":
            return "discouraged"
        if explicit == "repairable":
            return "auto_repair_available"
        if explicit == "ok":
            return "coherent"
        return explicit

    label = str(coherence.get("label", "unavailable") or "unavailable").strip().lower()
    try:
        score = float(coherence.get("score_continuous", 0.5))
    except Exception:
        score = 0.5
    score = float(np.clip(score, 0.0, 1.0))
    has_patch = bool(_coerce_mapping_for_governance(repairs.get("suggested_patch", {})))
    high_sev = int(repairs.get("n_high_severity", 0) or 0)
    n_issues = int(repairs.get("n_issues", 0) or 0)

    if bool(repairs.get("incompatible", False)) or label == "incoherent" or score < 0.35 or high_sev > 0:
        return "discouraged"
    if has_patch and (n_issues > 0 or label in {"mixed", "unavailable"} or score < 0.70):
        return "auto_repair_available"
    if n_issues > 0 or label == "mixed" or score < 0.60:
        return "stretched"
    if label == "coherent" and score >= 0.75:
        return "coherent"
    return "stretched"


def _build_governed_override_label(status: str) -> str:
    raw = str(status or "stretched").strip().lower()
    mapping = {
        "coherent": "coherent",
        "stretched": "stretched",
        "discouraged": "discouraged",
        "auto_repair_available": "auto-repair available",
    }
    return mapping.get(raw, raw or "stretched")


def _safe_governance_status(value: Any, *, default: str = "coherent") -> str:
    raw = str(value or default).strip().lower()
    allowed = {"coherent", "stretched", "discouraged", "auto_repair_available"}
    return raw if raw in allowed else default


def _build_actionable_governance_recommendations(
    *,
    philosophy: str,
    coherence_result: Dict[str, Any],
    repair_plan: Dict[str, Any],
    explanation_payload: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    coherence_map = dict(coherence_result or {})
    repairs = dict(repair_plan or {})
    explanation = dict(explanation_payload or {})
    recommendations: List[Dict[str, Any]] = []

    suggested_patch = _coerce_mapping_for_governance(repairs.get("suggested_patch", {}))
    actionable_warnings = [str(x) for x in list(repairs.get("actionable_warnings", []) or []) if str(x).strip()]
    tradeoffs = [str(x) for x in list(explanation.get("tradeoffs", []) or []) if str(x).strip()]
    drivers = [str(x) for x in list(explanation.get("drivers", []) or []) if str(x).strip()]
    label = str(coherence_map.get("label") or "unavailable")

    if suggested_patch:
        recommendations.append({
            "kind": "repair",
            "title": "Apply coherence repair",
            "why": actionable_warnings[0] if actionable_warnings else f"Improve structural alignment with {philosophy}.",
            "tradeoff": tradeoffs[0] if tradeoffs else "May sacrifice some flexibility to recover a cleaner structural posture.",
            "patch": suggested_patch,
        })

    if label in {"mixed", "unavailable", "incoherent"}:
        recommendations.append({
            "kind": "governance_review",
            "title": "Review overridden blocks",
            "why": f"The current configuration is {label} relative to the {philosophy} philosophy.",
            "tradeoff": tradeoffs[0] if tradeoffs else "Keeping the current override path may preserve local upside but increases structural fragility.",
            "patch": {},
        })

    if drivers:
        recommendations.append({
            "kind": "hold_or_document",
            "title": "Document why this config is being kept",
            "why": drivers[0],
            "tradeoff": tradeoffs[0] if tradeoffs else "Even a coherent configuration should be justified in terms of its main trade-off.",
            "patch": {},
        })

    out: List[Dict[str, Any]] = []
    seen = set()
    for rec in recommendations:
        title = str(rec.get("title", "")).strip().lower()
        if not title or title in seen:
            continue
        seen.add(title)
        out.append(rec)
    return out[:4]


def build_standard_governance_payload(
    *,
    philosophy: str,
    base_intention: Optional[Dict[str, Any]] = None,
    coherent_base_cfg: Optional[MicroPipelineConfig] = None,
    final_cfg: Optional[MicroPipelineConfig] = None,
    overrides_applied: Optional[Dict[str, Any]] = None,
    coherence: Optional[Dict[str, Any]] = None,
    repairs: Optional[Dict[str, Any]] = None,
    trace: Optional[List[Dict[str, Any]]] = None,
    results: Any = None,
) -> GovernancePayload:
    philosophy_name = _coerce_philosophy_name_for_simple_flow(philosophy)
    coherence_map = dict(coherence or {})
    repairs_map = dict(repairs or {})
    trace_payload = list(trace or [])

    warnings: List[str] = []
    warnings.extend([str(x) for x in list(coherence_map.get("warnings", []) or []) if str(x).strip()])
    warnings.extend([str(x) for x in list(repairs_map.get("actionable_warnings", []) or []) if str(x).strip()])
    seen_warn = set()
    deduped_warnings: List[str] = []
    for warning in warnings:
        if warning in seen_warn:
            continue
        seen_warn.add(warning)
        deduped_warnings.append(warning)

    explanation_payload: Dict[str, Any] = {}
    short_tradeoff_explanation = ""
    try:
        from src.explain import explain_why_config_works
        explanation_payload = dict(explain_why_config_works(
            cfg=config_to_dict(final_cfg if isinstance(final_cfg, MicroPipelineConfig) else MicroPipelineConfig()),
            results=results or {},
            coherence=coherence_map,
            philosophy=philosophy_name,
        ) or {})
    except Exception:
        explanation_payload = {}

    summary_lines = [str(x) for x in list(explanation_payload.get("summary", []) or []) if str(x).strip()]
    tradeoffs = [str(x) for x in list(explanation_payload.get("tradeoffs", []) or []) if str(x).strip()]
    if tradeoffs:
        short_tradeoff_explanation = tradeoffs[0]
    elif summary_lines:
        short_tradeoff_explanation = summary_lines[-1]
    elif deduped_warnings:
        short_tradeoff_explanation = deduped_warnings[0]
    else:
        short_tradeoff_explanation = f"Configuration currently reads as {_safe_governance_status(coherence_map.get('status'), default='coherent')} for the {philosophy_name} philosophy."

    actionable_recommendations = _build_actionable_governance_recommendations(
        philosophy=philosophy_name,
        coherence_result=coherence_map,
        repair_plan=repairs_map,
        explanation_payload=explanation_payload,
    )

    return GovernancePayload(
        philosophy_effective=philosophy_name,
        base_intention=_coerce_mapping_for_governance(base_intention),
        coherent_base_cfg=config_to_dict(coherent_base_cfg if isinstance(coherent_base_cfg, MicroPipelineConfig) else MicroPipelineConfig()),
        overrides_applied=coerce_candidate_payload_for_config(_coerce_mapping_for_governance(overrides_applied)),
        final_effective_cfg=config_to_dict(final_cfg if isinstance(final_cfg, MicroPipelineConfig) else MicroPipelineConfig()),
        coherence_score=_safe_float(coherence_map.get("score_continuous")),
        coherence_status=_safe_governance_status(coherence_map.get("status") or repairs_map.get("status") or _build_governed_status(coherence_map, repairs_map)),
        warnings=deduped_warnings,
        suggested_repairs={
            "status": str(repairs_map.get("status", "") or ""),
            "suggested_patch": _coerce_mapping_for_governance(repairs_map.get("suggested_patch", {})),
            "actionable_warnings": [str(x) for x in list(repairs_map.get("actionable_warnings", []) or []) if str(x).strip()],
            "n_issues": int(repairs_map.get("n_issues", 0) or 0),
            "n_high_severity": int(repairs_map.get("n_high_severity", 0) or 0),
        },
        short_tradeoff_explanation=short_tradeoff_explanation,
        actionable_recommendations=actionable_recommendations,
        trace=trace_payload,
    )


def serialize_governance_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, GovernancePayload):
        return payload.to_dict()
    if isinstance(payload, GovernedConfigResolutionResult):
        return payload.to_governance_payload().to_dict()
    if isinstance(payload, dict):
        return _json_safe(payload)
    return _json_safe(_coerce_mapping_for_governance(payload))


def _resolve_governance_constraints(
    *,
    philosophy: str,
    universe: Any = None,
    strategy_template: Any = None,
    style_preset: Any = None,
    explicit_constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    constraints = dict(explicit_constraints or {})
    if constraints:
        return constraints
    try:
        from src.coherence import resolve_philosophy_to_constraints
    except Exception:
        return {}
    try:
        return dict(resolve_philosophy_to_constraints(
            philosophy=philosophy,
            universe=universe,
            strategy_template=strategy_template,
            style_preset=style_preset,
        ) or {})
    except Exception:
        return {}


def _evaluate_governed_coherence(
    cfg: MicroPipelineConfig,
    *,
    philosophy: str,
    universe: Any = None,
    strategy_template: Any = None,
    style_preset: Any = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    payload = config_to_dict(cfg if isinstance(cfg, MicroPipelineConfig) else MicroPipelineConfig())
    coherence_result: Dict[str, Any]
    repair_plan: Dict[str, Any]
    try:
        from src.coherence import evaluate_config_coherence, suggest_coherence_repairs
    except Exception:
        coherence_result = {
            "label": "unavailable",
            "score": 0.0,
            "score_continuous": 0.5,
            "reasons": [],
            "warnings": ["coherence_module_unavailable"],
            "philosophy": philosophy,
        }
        repair_plan = {"n_issues": 0, "suggested_patch": {}, "incompatible": False}
        return coherence_result, repair_plan

    coherence_kwargs = {
        "universe": universe if universe is not None else "medium",
        "strategy_template": strategy_template,
        "style_preset": style_preset,
    }
    try:
        coherence_result = dict(evaluate_config_coherence(payload, philosophy, **coherence_kwargs) or {})
    except Exception as exc:
        coherence_result = {
            "label": "unavailable",
            "score": 0.0,
            "score_continuous": 0.5,
            "reasons": [],
            "warnings": [f"coherence_eval_failed: {exc}"],
        }
    try:
        repair_plan = dict(suggest_coherence_repairs(payload, philosophy, **coherence_kwargs) or {})
    except Exception as exc:
        repair_plan = {
            "n_issues": 0,
            "suggested_patch": {},
            "incompatible": False,
            "warnings": [f"coherence_repair_failed: {exc}"],
        }

    coherence_result["philosophy"] = philosophy
    try:
        coherence_result["score_continuous"] = float(np.clip(float(coherence_result.get("score_continuous", 0.5)), 0.0, 1.0))
    except Exception:
        coherence_result["score_continuous"] = 0.5
    return coherence_result, repair_plan


def apply_governed_overrides(
    *,
    base_coherent_cfg: MicroPipelineConfig,
    manual_overrides: Optional[Dict[str, Any]] = None,
    philosophy: Optional[str] = None,
    universe: Any = None,
    strategy_template: Any = None,
    style_preset: Any = None,
    recheck_immediately: bool = True,
) -> GovernedOverrideResult:
    """
    Governed manual-edit layer for advanced controls.

    Treat any user edit as:
    - an explicit override on top of a coherent base config
    - followed by a structural re-check
    - with warnings / status / suggested repair when coherence degrades
    """
    philosophy_name = _coerce_philosophy_name_for_simple_flow(philosophy or style_preset)
    base_cfg = base_coherent_cfg if isinstance(base_coherent_cfg, MicroPipelineConfig) else MicroPipelineConfig()
    overrides_payload = coerce_candidate_payload_for_config(_coerce_mapping_for_governance(manual_overrides))
    trace: List[Dict[str, Any]] = [
        {
            "stage": "manual_override_input",
            "n_override_keys": int(len(overrides_payload)),
            "override_keys": sorted(list(overrides_payload.keys())),
        }
    ]

    coherence_before, repairs_before = _evaluate_governed_coherence(
        base_cfg,
        philosophy=philosophy_name,
        universe=universe,
        strategy_template=strategy_template,
        style_preset=style_preset,
    )
    trace.append({
        "stage": "coherence_before_overrides",
        "status": _build_governed_status(coherence_before, repairs_before),
        "score_continuous": _safe_float(coherence_before.get("score_continuous")),
        "label": coherence_before.get("label"),
    })

    final_cfg = base_cfg if not overrides_payload else apply_config_patch(base_cfg, overrides_payload)
    trace.append({
        "stage": "manual_override_merge",
        "changed": bool(overrides_payload),
        "override_count": int(len(overrides_payload)),
    })

    if recheck_immediately:
        coherence_after, repairs_after = _evaluate_governed_coherence(
            final_cfg,
            philosophy=philosophy_name,
            universe=universe,
            strategy_template=strategy_template,
            style_preset=style_preset,
        )
    else:
        coherence_after, repairs_after = coherence_before, repairs_before

    status = _build_governed_status(coherence_after, repairs_after)
    label = _build_governed_override_label(status)
    warnings: List[str] = []
    warnings.extend([str(x) for x in list(coherence_after.get("warnings", []) or []) if str(x).strip()])
    warnings.extend([str(x) for x in list(coherence_after.get("actionable_warnings", []) or []) if str(x).strip()])
    if status == "stretched":
        warnings.append("Manual overrides stretch the current philosophy; review suggested repair.")
    elif status == "discouraged":
        warnings.append("Manual overrides push the configuration into a discouraged region.")
    elif status == "auto_repair_available":
        warnings.append("Manual overrides remain usable, but a safer repaired version is available.")
    seen_warn = set()
    deduped_warnings: List[str] = []
    for w in warnings:
        key = str(w).strip()
        if not key or key in seen_warn:
            continue
        seen_warn.add(key)
        deduped_warnings.append(key)

    trace.append({
        "stage": "coherence_after_overrides",
        "status": status,
        "label": label,
        "score_continuous": _safe_float(coherence_after.get("score_continuous")),
        "repair_patch_available": bool(_coerce_mapping_for_governance(repairs_after.get("suggested_patch", {}))),
    })

    return GovernedOverrideResult(
        philosophy=philosophy_name,
        status=status,
        label=label,
        base_coherent_cfg=base_cfg,
        final_cfg=final_cfg,
        manual_overrides_applied=overrides_payload,
        coherence_before=coherence_before,
        coherence_after=coherence_after,
        repairs=repairs_after,
        warnings=deduped_warnings,
        suggested_repair_patch=_coerce_mapping_for_governance(repairs_after.get("suggested_patch", {})),
        trace=trace,
    )


def resolve_user_intention_to_governed_config(
    *,
    simple_spec: Optional[SimpleUISpec] = None,
    base_cfg: Optional[MicroPipelineConfig] = None,
    philosophy: Optional[str] = None,
    universe: Any = None,
    strategy_template: Any = None,
    style_preset: Any = None,
    overrides: Optional[Dict[str, Any]] = None,
    recommendation_patch: Optional[Dict[str, Any]] = None,
    repair_patch: Optional[Dict[str, Any]] = None,
    constraints: Optional[Dict[str, Any]] = None,
    trace_source: Any = None,
) -> GovernedConfigResolutionResult:
    """
    Unified governed-resolution contract for configuration changes.

    Official flow:
        1) receive base intention
        2) build coherent config
        3) apply governed overrides / recommendation / repair patches
        4) re-evaluate coherence
        5) return final cfg + coherence + repairs + status + traceability

    This function is intentionally lightweight and non-executing:
    - it does not run the engine
    - it does not change predictive-selection methodology
    - it only resolves and validates config state
    """
    trace: List[Dict[str, Any]] = []
    base_intention = {
        "source": str(trace_source or ("simple_spec" if isinstance(simple_spec, SimpleUISpec) else "base_cfg")),
        "philosophy_requested": philosophy,
        "universe": universe,
        "strategy_template": strategy_template,
        "style_preset": style_preset,
    }

    if isinstance(simple_spec, SimpleUISpec):
        philosophy_name = _coerce_philosophy_name_for_simple_flow(philosophy or simple_spec.style_preset)
        base_intention.update({
            "simple_spec": _json_safe(asdict(simple_spec)),
            "strategy_template": simple_spec.strategy_template,
            "style_preset": simple_spec.style_preset,
            "universe": simple_spec.universe_size,
        })
        cfg_base, summary = resolve_simple_ui_to_micro_cfg(simple_spec)
        trace.append({
            "stage": "base_resolution",
            "kind": "simple_spec",
            "philosophy": philosophy_name,
            "summary": _json_safe(summary),
        })
        if universe is None:
            universe = simple_spec.universe_size
        if strategy_template is None:
            strategy_template = simple_spec.strategy_template
        if style_preset is None:
            style_preset = simple_spec.style_preset
    else:
        cfg_base = base_cfg if isinstance(base_cfg, MicroPipelineConfig) else MicroPipelineConfig()
        philosophy_name = _coerce_philosophy_name_for_simple_flow(philosophy or style_preset)
        base_intention["base_cfg"] = config_to_dict(cfg_base)
        trace.append({
            "stage": "base_resolution",
            "kind": "base_cfg",
            "philosophy": philosophy_name,
        })

    resolved_constraints = _resolve_governance_constraints(
        philosophy=philosophy_name,
        universe=universe,
        strategy_template=strategy_template,
        style_preset=style_preset,
        explicit_constraints=constraints,
    )
    trace.append({
        "stage": "constraints_resolution",
        "constraints_available": bool(resolved_constraints),
        "constraints": _json_safe(resolved_constraints),
    })

    coherent_base_cfg, coherent_meta = generate_coherent_config(
        philosophy=philosophy_name,
        base_cfg=cfg_base,
        constraints=resolved_constraints,
    )
    trace.append({
        "stage": "coherent_base_generation",
        "changed": bool((coherent_meta or {}).get("changed", False)),
        "n_changes": int((coherent_meta or {}).get("n_changes", 0) or 0),
        "changes": _json_safe((coherent_meta or {}).get("changes", {})),
    })

    applied_overrides = coerce_candidate_payload_for_config(_coerce_mapping_for_governance(overrides))
    applied_recommendation_patch = coerce_candidate_payload_for_config(_coerce_mapping_for_governance(recommendation_patch))
    applied_repair_patch = coerce_candidate_payload_for_config(_coerce_mapping_for_governance(repair_patch))

    combined_patch: Dict[str, Any] = {}
    for patch_name, patch_payload in [
        ("overrides", applied_overrides),
        ("recommendation_patch", applied_recommendation_patch),
        ("repair_patch", applied_repair_patch),
    ]:
        if not patch_payload:
            continue
        combined_patch.update(patch_payload)
        trace.append({
            "stage": "patch_application",
            "patch_kind": patch_name,
            "patch_payload": _json_safe(patch_payload),
        })

    override_result = apply_governed_overrides(
        base_coherent_cfg=coherent_base_cfg,
        manual_overrides=combined_patch,
        philosophy=philosophy_name,
        universe=universe,
        strategy_template=strategy_template,
        style_preset=style_preset,
        recheck_immediately=True,
    )
    final_cfg = override_result.final_cfg
    coherence_result = dict(override_result.coherence_after or {})
    repair_plan = dict(override_result.repairs or {})
    status = str(override_result.status or _build_governed_status(coherence_result, repair_plan))
    trace.extend(list(override_result.trace or []))
    trace.append({
        "stage": "final_coherence_evaluation",
        "status": status,
        "coherence_label": coherence_result.get("label"),
        "coherence_score": coherence_result.get("score_continuous"),
        "repair_issue_count": int(repair_plan.get("n_issues", 0) or 0),
        "repair_patch_available": bool(_coerce_mapping_for_governance(repair_plan.get("suggested_patch", {}))),
    })

    return GovernedConfigResolutionResult(
        philosophy=philosophy_name,
        status=status,
        base_intention=base_intention,
        constraints=resolved_constraints,
        base_cfg=cfg_base,
        coherent_base_cfg=coherent_base_cfg,
        final_cfg=final_cfg,
        overrides_applied=applied_overrides,
        recommendation_patch_applied=applied_recommendation_patch,
        repair_patch_applied=applied_repair_patch,
        coherence=coherence_result,
        repairs=repair_plan,
        trace=trace,
    )


@dataclass(frozen=True)
class SimplifiedGlobalParams:
    training_window_months: int
    mu_lookback_months: int
    sigma_lookback_months: int
    covariance_mode: str
    signal_mode: str
    probabilistic_mode: str
    target_vol_monthly: float
    requested_top_k: Optional[int]
    effective_top_k: Optional[int]
    requested_temperature: float
    effective_temperature: float
    long_only: bool
    base_weight_cap: Optional[float]
    effective_weight_cap: Optional[float]
    feature_mu_enabled: bool
    regime_mode: str
    regime_dependent_universe_enabled: bool
    n_assets_input: Optional[int]
    universe_bucket: str
    dispersion_gate_enabled: bool
    vol_targeting_enabled: bool
    factor_model_active: bool
    factor_covariance_active: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def resolve_simplified_global_params(
    cfg: MicroPipelineConfig,
    *,
    n_assets: Optional[int] = None,
    regime: Optional[str] = None,
    score_dispersion: Optional[float] = None,
    estimated_portfolio_vol_monthly: Optional[float] = None,
    corr_mat: Optional[pd.DataFrame] = None,
) -> SimplifiedGlobalParams:
    n_assets_eff = None if n_assets is None else max(int(n_assets), 1)
    universe_scaling = _resolve_universe_scaling(n_assets_eff or 1, cfg)
    adaptive_meta = _resolve_adaptive_allocation(
        score_dispersion=float(score_dispersion) if score_dispersion is not None and np.isfinite(score_dispersion) else np.nan,
        cfg=cfg,
    )
    requested_top_k = cfg.top_k if cfg.top_k is None else int(cfg.top_k)
    effective_top_k = _resolve_effective_top_k_by_universe_size(
        n_assets=n_assets_eff or 1,
        requested_top_k=adaptive_meta.get("top_k", requested_top_k),
        universe_scaling=universe_scaling,
        cfg=cfg,
    )
    requested_temperature = float(adaptive_meta.get("temperature", cfg.temperature))
    effective_temperature = _resolve_effective_temperature_by_universe_size(
        requested_temperature=requested_temperature,
        universe_scaling=universe_scaling,
        cfg=cfg,
    )
    resolved_cap = None
    if bool(cfg.long_only):
        resolved_cap, _ = _resolve_effective_long_only_weight_cap(
            cfg,
            n_assets=n_assets_eff or 1,
            estimated_portfolio_vol_monthly=float(estimated_portfolio_vol_monthly) if estimated_portfolio_vol_monthly is not None and np.isfinite(estimated_portfolio_vol_monthly) else np.nan,
            corr_mat=corr_mat,
            score_dispersion=float(score_dispersion) if score_dispersion is not None and np.isfinite(score_dispersion) else np.nan,
            regime=regime,
        )
    base_cap = _first_finite_optional(getattr(cfg, "asset_weight_cap", None), getattr(cfg, "w_cap", None))
    effective_cap = float(resolved_cap) if resolved_cap is not None and np.isfinite(float(resolved_cap)) else None
    return SimplifiedGlobalParams(
        training_window_months=int(cfg.min_train),
        mu_lookback_months=int(cfg.lookback_mu),
        sigma_lookback_months=int(cfg.lookback_sigma),
        covariance_mode=str(cfg.covariance_mode),
        signal_mode=str(cfg.signal_mode),
        probabilistic_mode=str(cfg.probabilistic_mode),
        target_vol_monthly=float(cfg.target_portfolio_vol_monthly),
        requested_top_k=requested_top_k,
        effective_top_k=effective_top_k,
        requested_temperature=float(requested_temperature),
        effective_temperature=float(effective_temperature),
        long_only=bool(cfg.long_only),
        base_weight_cap=float(base_cap) if base_cap is not None and np.isfinite(float(base_cap)) else None,
        effective_weight_cap=effective_cap,
        feature_mu_enabled=bool(cfg.feature_mu_enabled),
        regime_mode=str(cfg.regime_mode),
        regime_dependent_universe_enabled=bool(getattr(cfg, "regime_dependent_universe_enabled", False)),
        n_assets_input=int(n_assets_eff) if n_assets_eff is not None else None,
        universe_bucket=str(universe_scaling.get("bucket", "unknown")),
        dispersion_gate_enabled=bool(cfg.dispersion_gate),
        vol_targeting_enabled=bool(cfg.vol_targeting),
        factor_model_active=bool(cfg.factor_model_active),
        factor_covariance_active=bool(cfg.factor_covariance_active),
    )


def build_simplified_global_params_summary(
    result: Dict[str, Any],
    *,
    score_dispersion_fallback: Optional[float] = None,
) -> Dict[str, Any]:
    cfg = result.get("config", MicroPipelineConfig())
    if not isinstance(cfg, MicroPipelineConfig):
        return {}
    diag = result.get("diagnostics_df", pd.DataFrame())
    weights_df = result.get("weights_df", pd.DataFrame())
    corr_snaps = result.get("correlation_snapshots", {}) or {}

    n_assets = None
    regime = None
    score_dispersion = score_dispersion_fallback
    est_vol = None
    corr_mat = None

    if isinstance(diag, pd.DataFrame) and not diag.empty:
        row = diag.tail(1).iloc[0]
        if "n_assets" in diag.columns:
            try:
                n_assets = int(row.get("n_assets"))
            except Exception:
                n_assets = None
        regime = row.get("regime", None) if "regime" in diag.columns else None
        if "score_dispersion" in diag.columns:
            try:
                score_dispersion = float(row.get("score_dispersion"))
            except Exception:
                pass
        if "est_portfolio_vol_monthly" in diag.columns:
            try:
                est_vol = float(row.get("est_portfolio_vol_monthly"))
            except Exception:
                pass

    if n_assets is None and isinstance(weights_df, pd.DataFrame) and not weights_df.empty and "asset" in weights_df.columns:
        try:
            n_assets = int(pd.Series(weights_df["asset"]).nunique())
        except Exception:
            n_assets = None

    if isinstance(corr_snaps, dict) and corr_snaps:
        try:
            last_key = sorted(corr_snaps.keys())[-1]
            corr_mat = corr_snaps[last_key]
        except Exception:
            corr_mat = None

    simp = resolve_simplified_global_params(
        cfg,
        n_assets=n_assets,
        regime=regime,
        score_dispersion=score_dispersion,
        estimated_portfolio_vol_monthly=est_vol,
        corr_mat=corr_mat if isinstance(corr_mat, pd.DataFrame) else None,
    )
    return simp.to_dict()


# ============================================================
# Config / reporting / persistence helpers
# ============================================================

def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Series):
        return {str(k): _json_safe(v) for k, v in value.to_dict().items()}
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def config_to_dict(cfg: MicroPipelineConfig) -> Dict[str, Any]:
    payload = {f.name: _json_safe(getattr(cfg, f.name)) for f in fields(MicroPipelineConfig)}
    payload["selection_policy"] = _normalize_selection_policy_value(payload.get("selection_policy"))
    payload["composite_profile"] = _normalize_composite_profile_value(payload.get("composite_profile"))
    payload["manual_composite_weights"] = _json_safe(_sanitize_manual_composite_weights(payload.get("manual_composite_weights")))
    payload["manual_composite_weights_enabled"] = bool(payload.get("manual_composite_weights_enabled", False))
    return payload


def _looks_like_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "nan", "na"}:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _extract_literal_choices(type_str: str) -> Optional[List[Any]]:
    if "Literal[" not in type_str:
        return None
    start = type_str.find("Literal[") + len("Literal[")
    end = type_str.rfind("]")
    if start < len("Literal[") or end <= start:
        return None
    raw = type_str[start:end]
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    out: List[Any] = []
    for p in parts:
        if (p.startswith("'") and p.endswith("'")) or (p.startswith('"') and p.endswith('"')):
            out.append(p[1:-1])
            continue
        if p in {"True", "False"}:
            out.append(p == "True")
            continue
        try:
            if any(ch in p for ch in [".", "e", "E"]):
                out.append(float(p))
            else:
                out.append(int(p))
            continue
        except Exception:
            out.append(p)
    return out or None


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return bool(int(value))
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(float(value)):
            raise ValueError("non-finite bool-like value")
        return bool(int(round(float(value))))
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "1", "yes", "y", "on"}:
            return True
        if s in {"false", "0", "no", "n", "off"}:
            return False
    return bool(value)


def _coerce_mapping_like(raw_value: Any) -> Dict[str, Any]:
    if isinstance(raw_value, dict):
        return dict(raw_value)
    if isinstance(raw_value, str):
        s = raw_value.strip()
        if not s:
            return {}
        try:
            parsed = json.loads(s)
        except Exception as exc:
            raise ValueError("invalid mapping/json value") from exc
        if isinstance(parsed, dict):
            return dict(parsed)
    raise ValueError("value is not a mapping")


def _sanitize_manual_composite_weights(raw_value: Any) -> Optional[Dict[str, float]]:
    if _looks_like_null(raw_value):
        return None
    payload = _coerce_mapping_like(raw_value)
    cleaned: Dict[str, float] = {}
    for key, value in payload.items():
        name = str(key).strip()
        if name not in COMPOSITE_WEIGHT_KEYS:
            continue
        try:
            weight = float(value)
        except Exception:
            continue
        if not np.isfinite(weight):
            continue
        cleaned[name] = float(weight)
    return cleaned or None


def _normalize_selection_policy_value(raw_value: Any) -> str:
    raw = str(raw_value or "fixed_composite_score").strip().lower()
    alias_map = {
        "fixed_composite_score": "fixed_composite_score",
        "composite_balanced": "fixed_composite_score",
        "composite_growth": "fixed_composite_score",
        "composite_defensive": "fixed_composite_score",
        "balanced_compromise": "balanced_compromise",
        "knee_point": "knee_point",
        "weighted_hypervolume": "weighted_hypervolume",
    }
    return alias_map.get(raw, raw or "fixed_composite_score")


def _normalize_composite_profile_value(raw_value: Any) -> Optional[str]:
    if _looks_like_null(raw_value):
        return None
    raw = str(raw_value).strip().lower().replace("-", "_").replace(" ", "_")
    alias_map = {
        "balanced": "balanced",
        "growth": "growth",
        "defensive": "defensive",
        "conservative": "defensive",
        "research": "growth",
        "composite_balanced": "balanced",
        "composite_growth": "growth",
        "composite_defensive": "defensive",
    }
    return alias_map.get(raw, raw or None)


def _coerce_field_value(raw_value: Any, field_type: Any) -> Any:
    type_str = str(field_type)
    nullable = ("Optional" in type_str) or ("NoneType" in type_str)
    if _looks_like_null(raw_value):
        if nullable:
            return None
        raise ValueError("null provided for non-nullable field")

    literal_choices = _extract_literal_choices(type_str)
    if literal_choices:
        if raw_value in literal_choices:
            return raw_value
        if isinstance(raw_value, str):
            raw_norm = raw_value.strip().lower()
            for choice in literal_choices:
                if isinstance(choice, str) and raw_norm == str(choice).strip().lower():
                    return choice
        if any(isinstance(choice, bool) for choice in literal_choices):
            value_bool = _coerce_bool(raw_value)
            if value_bool in literal_choices:
                return value_bool
        if any(isinstance(choice, int) and not isinstance(choice, bool) for choice in literal_choices):
            try:
                value_int = int(round(float(raw_value)))
                if value_int in literal_choices:
                    return value_int
            except Exception:
                pass
        if any(isinstance(choice, float) for choice in literal_choices):
            try:
                value_float = float(raw_value)
                for choice in literal_choices:
                    if isinstance(choice, float) and np.isfinite(value_float) and np.isclose(value_float, choice):
                        return choice
            except Exception:
                pass
        raise ValueError(f"value {raw_value!r} not in literal choices")

    if "bool" in type_str:
        return _coerce_bool(raw_value)
    if "int" in type_str and "float" not in type_str:
        if isinstance(raw_value, (np.integer, int)) and not isinstance(raw_value, bool):
            return int(raw_value)
        if isinstance(raw_value, (np.floating, float)):
            if not np.isfinite(float(raw_value)):
                raise ValueError("non-finite int value")
            return int(round(float(raw_value)))
        if isinstance(raw_value, str):
            s = raw_value.strip()
            if s == "":
                if nullable:
                    return None
                raise ValueError("empty string for int field")
            return int(round(float(s)))
        return int(raw_value)
    if "float" in type_str:
        value = float(raw_value)
        if not np.isfinite(value):
            raise ValueError("non-finite float value")
        return value
    if "str" in type_str:
        return str(raw_value)
    return raw_value


def coerce_candidate_payload_for_config(payload: dict) -> dict:
    """
    Safely coerce a candidate/override payload so it can be applied to
    MicroPipelineConfig without crashing on Optional[int]/Optional[float]
    fields or leaking inactive dependent parameters.

    Rules:
    - cast ints / floats / bools / literals according to dataclass field types
    - preserve nullable fields as None when the field actually supports it
    - never pass None into non-nullable config fields
    - drop invalid keys and invalid values rather than forcing bad casts
    - remove engine-inactive dependent fields after provisional config resolution
    """
    if not isinstance(payload, dict):
        return {}

    field_map = {f.name: f for f in fields(MicroPipelineConfig)}
    coerced: Dict[str, Any] = {}
    for key, raw_value in dict(payload).items():
        name = str(key)
        fld = field_map.get(name)
        if fld is None:
            continue
        try:
            if name == "manual_composite_weights":
                coerced[name] = _sanitize_manual_composite_weights(raw_value)
            elif name == "selection_policy":
                coerced[name] = _normalize_selection_policy_value(raw_value)
            elif name == "composite_profile":
                coerced[name] = _normalize_composite_profile_value(raw_value)
            else:
                coerced[name] = _coerce_field_value(raw_value, fld.type)
        except Exception:
            continue

    if not coerced:
        return {}

    if "manual_composite_weights" in coerced and "manual_composite_weights_enabled" not in coerced:
        coerced["manual_composite_weights_enabled"] = bool(coerced.get("manual_composite_weights"))

    defaults = config_to_dict(MicroPipelineConfig())
    provisional_payload = dict(defaults)
    provisional_payload.update(coerced)
    provisional_cfg = MicroPipelineConfig(**provisional_payload)

    try:
        audit = build_engine_param_activity_audit(provisional_cfg)
        active_lookup = {str(r["param"]): bool(r["active"]) for _, r in audit.iterrows()}
        cleaned = {
            k: v for k, v in coerced.items()
            if bool(active_lookup.get(str(k), True))
        }
    except Exception:
        cleaned = dict(coerced)

    return cleaned


def config_from_dict(data: Dict[str, Any]) -> MicroPipelineConfig:
    payload = coerce_candidate_payload_for_config(dict(data or {}))
    return MicroPipelineConfig(**payload)


def save_config_json(cfg: MicroPipelineConfig, path: str) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config_to_dict(cfg), indent=2, sort_keys=True), encoding="utf-8")
    return str(out)


def load_config_json(path: str) -> MicroPipelineConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config JSON must decode to a dictionary")
    return config_from_dict(payload)


def config_fingerprint(cfg: MicroPipelineConfig, *, length: int = 12) -> str:
    blob = json.dumps(config_to_dict(cfg), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[: max(int(length), 6)]


def build_tunable_param_schema() -> Dict[str, Dict[str, Any]]:
    """
    Build a formal tunable-parameter schema for global / combinatorial search.

    This is intentionally broader than the engine activity audit: it describes
    the *catalog* of candidate tuning dimensions, not whether they are active in
    a specific run. Downstream search layers can still combine this schema with
    ``build_engine_param_activity_audit(...)`` or
    ``sanitize_param_space_for_engine(...)`` to respect current engine context.

    Per parameter the schema includes:
    - dtype
    - nullable
    - kind: continuous / int / categorical / bool
    - low/high or values when inferable
    - log_scale heuristic
    - active_if (controller hint, not executable code)
    - group
    - mutation_step
    - safe_default
    """
    default_cfg = MicroPipelineConfig()
    schema: Dict[str, Dict[str, Any]] = {}

    group_overrides: Dict[str, str] = {
        'date_col': 'io', 'asset_col': 'io', 'return_col': 'io',
        'min_train': 'data_window', 'lookback_mu': 'data_window', 'lookback_sigma': 'data_window', 'sigma_floor': 'risk_model',
        'temperature': 'allocation', 'top_k': 'allocation', 'long_only': 'allocation', 'weight_shrink': 'allocation', 'inertia': 'allocation',
        'deadband': 'allocation', 'deadband_threshold': 'allocation', 'asset_weight_cap': 'allocation', 'w_cap': 'allocation',
        'turnover_penalty_strength': 'turnover', 'turnover_penalty_power': 'turnover', 'turnover_penalty_target': 'turnover',
        'turnover_penalty_max_turnover': 'turnover', 'turnover_constraint_max_turnover': 'turnover',
        'cost_model_enabled': 'costs_taxes', 'transaction_cost_commission_bps': 'costs_taxes', 'transaction_cost_slippage_bps': 'costs_taxes',
        'transaction_cost_spread_bps': 'costs_taxes', 'transaction_cost_market_impact_bps': 'costs_taxes', 'transaction_cost_market_impact_power': 'costs_taxes',
        'transaction_cost_min_trade_weight': 'costs_taxes', 'holding_cost_annual_bps': 'costs_taxes', 'tax_model_enabled': 'costs_taxes',
        'tax_short_term_rate': 'costs_taxes', 'tax_long_term_rate': 'costs_taxes', 'tax_long_term_threshold_months': 'costs_taxes',
        'tax_apply_loss_credit': 'costs_taxes', 'tax_loss_credit_rate': 'costs_taxes',
        'dispersion_gate': 'dispersion', 'dispersion_gate_threshold': 'dispersion', 'dispersion_gate_min_active_weight': 'dispersion',
        'ewma_sigma': 'risk_model', 'ewma_halflife': 'risk_model', 'score_normalize': 'signal', 'score_clip': 'signal',
        'covariance_mode': 'covariance', 'regime_dependent_covariance': 'covariance', 'correlation_lookback': 'covariance',
        'correlation_min_periods': 'covariance', 'correlation_shrink_to_identity': 'covariance', 'covariance_shrink_to_diagonal': 'covariance',
        'covariance_jitter': 'covariance', 'covariance_lookback_low': 'covariance', 'covariance_lookback_mid': 'covariance',
        'covariance_lookback_high': 'covariance', 'covariance_halflife_low': 'covariance', 'covariance_halflife_mid': 'covariance',
        'covariance_halflife_high': 'covariance', 'factor_covariance_active': 'covariance', 'factor_covariance_n_factors': 'covariance',
        'factor_covariance_min_obs': 'covariance', 'factor_covariance_shrink_to_diagonal': 'covariance', 'factor_covariance_blend': 'covariance',
        'factor_model_active': 'factor_model', 'factor_model_n_factors': 'factor_model', 'factor_model_min_obs': 'factor_model',
        'factor_model_mu_blend': 'factor_model', 'factor_model_residual_blend': 'factor_model', 'factor_model_covariance_blend': 'factor_model',
        'factor_model_shrink_to_diagonal': 'factor_model',
        'correlation_aware_allocation': 'correlation_allocator', 'correlation_allocator_method': 'correlation_allocator', 'correlation_allocator_blend': 'correlation_allocator',
        'correlation_penalty_strength': 'correlation_allocator', 'correlation_penalty_power': 'correlation_allocator', 'correlation_use_abs': 'correlation_allocator',
        'mean_variance_risk_aversion': 'correlation_allocator', 'risk_budget_strength': 'correlation_allocator', 'cluster_corr_threshold': 'correlation_allocator',
        'vol_targeting': 'vol_targeting', 'covariance_aware_vol_targeting': 'vol_targeting', 'target_portfolio_vol_monthly': 'vol_targeting',
        'vol_target_floor_mult': 'vol_targeting', 'vol_target_ceiling_mult': 'vol_targeting', 'regime_derisk_low': 'vol_targeting',
        'regime_derisk_mid': 'vol_targeting', 'regime_derisk_high': 'vol_targeting',
        'store_correlation_snapshots': 'diagnostics', 'store_sigma_fwd_snapshots': 'diagnostics',
        'probabilistic_mode': 'probabilistic', 'probabilistic_q_low': 'probabilistic', 'probabilistic_q_high': 'probabilistic',
        'probabilistic_min_obs': 'probabilistic', 'probabilistic_interval_penalty_weight': 'probabilistic', 'probabilistic_downside_penalty_weight': 'probabilistic',
        'probabilistic_confidence_scale': 'probabilistic', 'probabilistic_confidence_min_mult': 'probabilistic', 'probabilistic_confidence_max_mult': 'probabilistic',
        'probabilistic_overlay_strength': 'probabilistic', 'probabilistic_overlay_blend': 'probabilistic', 'signal_mode': 'signal', 'huber_delta': 'signal',
        'signal_score_blend': 'signal', 'feature_mu_enabled': 'feature_mu', 'feature_mu_blend': 'feature_mu', 'feature_mu_k': 'feature_mu',
        'feature_mu_min_obs': 'feature_mu', 'feature_mu_cols_max': 'feature_mu', 'regime_mode': 'regime', 'mu_regime_mode': 'regime',
        'regime_lookback': 'regime', 'regime_quantile_low': 'regime', 'regime_quantile_high': 'regime', 'regime_dependent_universe_enabled': 'regime_universe',
        'regime_universe_selection_metric': 'regime_universe', 'regime_universe_keep_frac_low': 'regime_universe',
        'regime_universe_keep_frac_mid': 'regime_universe', 'regime_universe_keep_frac_high': 'regime_universe', 'regime_universe_min_assets': 'regime_universe',
        'regime_universe_max_assets': 'regime_universe', 'regime_universe_low_offensive_vol_tilt': 'regime_universe',
        'regime_universe_high_defensive_vol_tilt': 'regime_universe', 'pure_cs_baseline': 'research',
        'selection_policy': 'chooser', 'composite_profile': 'chooser', 'manual_composite_weights_enabled': 'chooser', 'manual_composite_weights': 'chooser',
    }

    active_if_map: Dict[str, str] = {
        'asset_weight_cap': 'long_only == True', 'w_cap': 'long_only == True',
        'vol_dependent_cap_threshold_low': 'vol_dependent_cap_enabled == True', 'vol_dependent_cap_threshold_high': 'vol_dependent_cap_enabled == True',
        'vol_dependent_cap_low_mult': 'vol_dependent_cap_enabled == True', 'vol_dependent_cap_high_mult': 'vol_dependent_cap_enabled == True',
        'vol_dependent_cap_min': 'vol_dependent_cap_enabled == True', 'vol_dependent_cap_max': 'vol_dependent_cap_enabled == True',
        'corr_dependent_cap_threshold_low': 'corr_dependent_cap_enabled == True', 'corr_dependent_cap_threshold_high': 'corr_dependent_cap_enabled == True',
        'corr_dependent_cap_low_mult': 'corr_dependent_cap_enabled == True', 'corr_dependent_cap_high_mult': 'corr_dependent_cap_enabled == True',
        'corr_dependent_cap_min': 'corr_dependent_cap_enabled == True', 'corr_dependent_cap_max': 'corr_dependent_cap_enabled == True',
        'dispersion_dependent_cap_threshold_low': 'dispersion_dependent_cap_enabled == True', 'dispersion_dependent_cap_threshold_high': 'dispersion_dependent_cap_enabled == True',
        'dispersion_dependent_cap_low_mult': 'dispersion_dependent_cap_enabled == True', 'dispersion_dependent_cap_high_mult': 'dispersion_dependent_cap_enabled == True',
        'dispersion_dependent_cap_min': 'dispersion_dependent_cap_enabled == True', 'dispersion_dependent_cap_max': 'dispersion_dependent_cap_enabled == True',
        'regime_dependent_cap_low_mult': 'regime_dependent_cap_enabled == True', 'regime_dependent_cap_mid_mult': 'regime_dependent_cap_enabled == True',
        'regime_dependent_cap_high_mult': 'regime_dependent_cap_enabled == True', 'regime_dependent_cap_min': 'regime_dependent_cap_enabled == True',
        'regime_dependent_cap_max': 'regime_dependent_cap_enabled == True',
        'dispersion_sigma_lookback': 'dispersion_sigma_enabled == True', 'dispersion_sigma_strength': 'dispersion_sigma_enabled == True',
        'dispersion_sigma_floor_mult': 'dispersion_sigma_enabled == True', 'dispersion_sigma_ceiling_mult': 'dispersion_sigma_enabled == True',
        'dispersion_risk_strength': 'dispersion_risk_model_enabled == True', 'dispersion_risk_floor_mult': 'dispersion_risk_model_enabled == True',
        'dispersion_risk_ceiling_mult': 'dispersion_risk_model_enabled == True', 'dispersion_risk_apply_to_covariance': 'dispersion_risk_model_enabled == True',
        'dispersion_top_k_threshold_low': 'dispersion_top_k_enabled == True', 'dispersion_top_k_threshold_high': 'dispersion_top_k_enabled == True',
        'dispersion_top_k_low_mult': 'dispersion_top_k_enabled == True', 'dispersion_top_k_high_mult': 'dispersion_top_k_enabled == True',
        'dispersion_top_k_min_k': 'dispersion_top_k_enabled == True', 'dispersion_top_k_max_k': 'dispersion_top_k_enabled == True',
        'dispersion_vol_target_threshold_low': 'dispersion_vol_target_enabled == True and vol_targeting == True',
        'dispersion_vol_target_threshold_high': 'dispersion_vol_target_enabled == True and vol_targeting == True',
        'dispersion_vol_target_low_mult': 'dispersion_vol_target_enabled == True and vol_targeting == True',
        'dispersion_vol_target_high_mult': 'dispersion_vol_target_enabled == True and vol_targeting == True',
        'dispersion_vol_target_min': 'dispersion_vol_target_enabled == True and vol_targeting == True',
        'dispersion_vol_target_max': 'dispersion_vol_target_enabled == True and vol_targeting == True',
        'low_signal_fallback_threshold': 'low_signal_fallback_to_ew == True', 'low_signal_fallback_mode': 'low_signal_fallback_to_ew == True',
        'low_signal_fallback_min_model_weight': 'low_signal_fallback_to_ew == True',
        'regime_lookback': 'regime_mode == "quantile"', 'regime_quantile_low': 'regime_mode == "quantile"', 'regime_quantile_high': 'regime_mode == "quantile"',
        'covariance_lookback_low': 'regime_dependent_covariance == True', 'covariance_lookback_mid': 'regime_dependent_covariance == True',
        'covariance_lookback_high': 'regime_dependent_covariance == True', 'covariance_halflife_low': 'regime_dependent_covariance == True',
        'covariance_halflife_mid': 'regime_dependent_covariance == True', 'covariance_halflife_high': 'regime_dependent_covariance == True',
        'factor_covariance_n_factors': 'factor_covariance_active == True', 'factor_covariance_min_obs': 'factor_covariance_active == True',
        'factor_covariance_shrink_to_diagonal': 'factor_covariance_active == True', 'factor_covariance_blend': 'factor_covariance_active == True',
        'factor_model_n_factors': 'factor_model_active == True', 'factor_model_min_obs': 'factor_model_active == True',
        'factor_model_mu_blend': 'factor_model_active == True', 'factor_model_residual_blend': 'factor_model_active == True',
        'factor_model_covariance_blend': 'factor_model_active == True', 'factor_model_shrink_to_diagonal': 'factor_model_active == True',
        'covariance_aware_vol_targeting': 'vol_targeting == True', 'target_portfolio_vol_monthly': 'vol_targeting == True',
        'vol_target_floor_mult': 'vol_targeting == True', 'vol_target_ceiling_mult': 'vol_targeting == True',
        'regime_derisk_low': 'vol_targeting == True and regime_mode != "none"', 'regime_derisk_mid': 'vol_targeting == True and regime_mode != "none"',
        'regime_derisk_high': 'vol_targeting == True and regime_mode != "none"',
        'probabilistic_q_low': 'probabilistic_mode != "none"', 'probabilistic_q_high': 'probabilistic_mode != "none"',
        'probabilistic_min_obs': 'probabilistic_mode != "none"', 'probabilistic_interval_penalty_weight': 'probabilistic_mode != "none"',
        'probabilistic_downside_penalty_weight': 'probabilistic_mode != "none"', 'probabilistic_confidence_scale': 'probabilistic_mode != "none"',
        'probabilistic_confidence_min_mult': 'probabilistic_mode != "none"', 'probabilistic_confidence_max_mult': 'probabilistic_mode != "none"',
        'probabilistic_overlay_strength': 'probabilistic_mode != "none"', 'probabilistic_overlay_blend': 'probabilistic_mode != "none"',
        'probabilistic_qr_alpha': 'probabilistic_mode in {"quantile_regression", "hybrid"}', 'probabilistic_qr_solver': 'probabilistic_mode in {"quantile_regression", "hybrid"}',
        'probabilistic_qr_feature_cap': 'probabilistic_mode in {"quantile_regression", "hybrid"}',
        'probabilistic_regime_min_obs': 'probabilistic_regime_aware == True or probabilistic_mode in {"historical_by_regime", "hybrid"}',
        'probabilistic_feature_filter_min_obs': 'probabilistic_feature_filter_enabled == True', 'probabilistic_feature_filter_k': 'probabilistic_feature_filter_enabled == True',
        'probabilistic_knn_k': 'probabilistic_mode == "knn_historical"', 'probabilistic_knn_min_obs': 'probabilistic_mode == "knn_historical"',
        'probabilistic_knn_distance_power': 'probabilistic_mode == "knn_historical"', 'probabilistic_knn_weighted_quantiles': 'probabilistic_mode == "knn_historical"',
        'probabilistic_knn_weight_eps': 'probabilistic_mode == "knn_historical"',
        'probabilistic_bucket_n_bins': 'probabilistic_mode == "feature_bucketed_historical"', 'probabilistic_bucket_min_obs': 'probabilistic_mode == "feature_bucketed_historical"',
        'probabilistic_bucket_match_min_features': 'probabilistic_mode == "feature_bucketed_historical"', 'probabilistic_bucket_max_features': 'probabilistic_mode == "feature_bucketed_historical"',
        'probabilistic_parametric_min_sigma': 'probabilistic_mode == "parametric_feature_aware"', 'probabilistic_parametric_max_sigma_mult': 'probabilistic_mode == "parametric_feature_aware"',
        'probabilistic_parametric_use_neighbor_weights': 'probabilistic_mode == "parametric_feature_aware"',
        'probabilistic_hybrid_weight': 'probabilistic_mode == "hybrid"', 'probabilistic_hybrid_min_qr_weight': 'probabilistic_mode == "hybrid"',
        'probabilistic_hybrid_max_qr_weight': 'probabilistic_mode == "hybrid"', 'probabilistic_hybrid_use_confidence': 'probabilistic_mode == "hybrid"',
        'huber_delta': 'signal_mode == "huber_mu"',
        'lambdarank_lookback': 'signal_mode == "lambdarank_like"', 'lambdarank_temperature': 'signal_mode == "lambdarank_like"',
        'directional_classifier_lookback': 'signal_mode == "directional_classifier"', 'directional_classifier_threshold': 'signal_mode == "directional_classifier"',
        'directional_classifier_confidence_scale': 'signal_mode == "directional_classifier"',
        'logistic_loss_lookback': 'signal_mode == "logistic_loss"', 'logistic_loss_l2': 'signal_mode == "logistic_loss"',
        'logistic_loss_threshold': 'signal_mode == "logistic_loss"', 'logistic_loss_confidence_scale': 'signal_mode == "logistic_loss"',
        'quantile_loss_lookback': 'signal_mode == "quantile_loss"', 'quantile_loss_q': 'signal_mode == "quantile_loss"',
        'quantile_loss_alpha': 'signal_mode == "quantile_loss"', 'quantile_loss_confidence_scale': 'signal_mode == "quantile_loss"',
        'top_k_classifier_lookback': 'signal_mode == "top_k_classifier"', 'top_k_classifier_k': 'signal_mode == "top_k_classifier"',
        'top_k_classifier_threshold': 'signal_mode == "top_k_classifier"', 'top_k_classifier_confidence_scale': 'signal_mode == "top_k_classifier"',
        'lambdarank_real_lookback': 'signal_mode == "lambdarank_real"', 'lambdarank_real_temperature': 'signal_mode == "lambdarank_real"',
        'lambdarank_real_gain_power': 'signal_mode == "lambdarank_real"', 'lambdarank_real_pair_power': 'signal_mode == "lambdarank_real"',
        'lambdarank_real_l2': 'signal_mode == "lambdarank_real"',
        'multi_loss_rank_weight': 'multi_loss_enabled == True', 'multi_loss_direction_weight': 'multi_loss_enabled == True',
        'multi_loss_return_weight': 'multi_loss_enabled == True', 'multi_loss_logistic_weight': 'multi_loss_enabled == True',
        'multi_loss_topk_weight': 'multi_loss_enabled == True', 'multi_loss_temperature': 'multi_loss_enabled == True', 'multi_loss_l2': 'multi_loss_enabled == True',
        'feature_mu_blend': 'feature_mu_enabled == True', 'feature_mu_k': 'feature_mu_enabled == True',
        'feature_mu_min_obs': 'feature_mu_enabled == True', 'feature_mu_cols_max': 'feature_mu_enabled == True',
        'regime_universe_selection_metric': 'regime_dependent_universe_enabled == True', 'regime_universe_keep_frac_low': 'regime_dependent_universe_enabled == True',
        'regime_universe_keep_frac_mid': 'regime_dependent_universe_enabled == True', 'regime_universe_keep_frac_high': 'regime_dependent_universe_enabled == True',
        'regime_universe_min_assets': 'regime_dependent_universe_enabled == True', 'regime_universe_max_assets': 'regime_dependent_universe_enabled == True',
        'regime_universe_low_offensive_vol_tilt': 'regime_dependent_universe_enabled == True', 'regime_universe_high_defensive_vol_tilt': 'regime_dependent_universe_enabled == True',
        'tax_long_term_rate': 'tax_model_enabled == True', 'tax_long_term_threshold_months': 'tax_model_enabled == True',
        'tax_apply_loss_credit': 'tax_model_enabled == True', 'tax_loss_credit_rate': 'tax_apply_loss_credit == True',
        'composite_profile': 'selection_policy == "fixed_composite_score"',
        'manual_composite_weights_enabled': 'selection_policy == "fixed_composite_score"',
        'manual_composite_weights': 'selection_policy == "fixed_composite_score" and manual_composite_weights_enabled == True',
    }

    def _infer_group(name: str) -> str:
        if name in group_overrides:
            return group_overrides[name]
        prefixes = (
            ('vol_dependent_cap_', 'cap_controls'), ('corr_dependent_cap_', 'cap_controls'), ('dispersion_dependent_cap_', 'cap_controls'),
            ('regime_dependent_cap_', 'cap_controls'), ('dispersion_', 'dispersion'), ('probabilistic_', 'probabilistic'),
            ('directional_classifier_', 'signal'), ('logistic_loss_', 'signal'), ('quantile_loss_', 'signal'), ('top_k_classifier_', 'signal'),
            ('lambdarank_', 'signal'), ('lambdarank_real_', 'signal'), ('multi_loss_', 'signal'), ('feature_mu_', 'feature_mu'),
            ('regime_universe_', 'regime_universe'), ('factor_covariance_', 'covariance'), ('factor_model_', 'factor_model'),
            ('correlation_', 'correlation_allocator'), ('covariance_', 'covariance'), ('transaction_cost_', 'costs_taxes'), ('tax_', 'costs_taxes'),
        )
        for prefix, group in prefixes:
            if name.startswith(prefix):
                return group
        return 'general'

    def _infer_kind_and_values(field_obj: Any) -> Tuple[str, Optional[List[Any]]]:
        type_str = str(field_obj.type)
        literals = _extract_literal_choices(type_str)
        if literals is not None:
            if all(isinstance(v, bool) for v in literals):
                return 'bool', [False, True]
            return 'categorical', list(literals)
        if 'bool' in type_str:
            return 'bool', [False, True]
        if 'int' in type_str and 'float' not in type_str:
            return 'int', None
        if 'float' in type_str:
            return 'continuous', None
        if 'Dict[' in type_str or type_str.startswith('dict'):
            return 'mapping', None
        return 'categorical', None

    def _infer_bounds(name: str, kind: str, default: Any) -> Tuple[Optional[Any], Optional[Any]]:
        if kind not in {'continuous', 'int'}:
            return None, None
        explicit_ranges: Dict[str, Tuple[Any, Any]] = {
            'sigma_power_alpha': (0.25, 3.0), 'min_train': (24, 240), 'lookback_mu': (3, 60), 'lookback_sigma': (3, 60),
            'sigma_floor': (1e-8, 0.25), 'temperature': (0.10, 5.0), 'top_k': (1, 50), 'weight_shrink': (0.0, 1.0), 'inertia': (0.0, 1.0),
            'deadband_threshold': (0.0, 0.25), 'asset_weight_cap': (0.01, 1.0), 'w_cap': (0.01, 1.0), 'turnover_penalty_strength': (0.0, 5.0),
            'turnover_penalty_power': (0.5, 4.0), 'turnover_penalty_target': (0.0, 1.0), 'turnover_penalty_max_turnover': (0.0, 2.0),
            'turnover_constraint_max_turnover': (0.0, 2.0), 'target_portfolio_vol_monthly': (0.005, 0.20), 'vol_target_floor_mult': (0.1, 2.0),
            'vol_target_ceiling_mult': (0.5, 3.0), 'regime_derisk_low': (0.0, 1.5), 'regime_derisk_mid': (0.0, 1.5), 'regime_derisk_high': (0.0, 1.5),
            'probabilistic_q_low': (0.01, 0.49), 'probabilistic_q_high': (0.51, 0.99), 'probabilistic_min_obs': (3, 120),
            'probabilistic_interval_penalty_weight': (0.0, 3.0), 'probabilistic_downside_penalty_weight': (0.0, 3.0),
            'probabilistic_confidence_scale': (0.0, 5.0), 'probabilistic_confidence_min_mult': (0.25, 1.5),
            'probabilistic_confidence_max_mult': (0.5, 3.0), 'probabilistic_overlay_strength': (0.0, 3.0), 'probabilistic_overlay_blend': (0.0, 1.0),
            'probabilistic_qr_alpha': (0.0, 10.0), 'probabilistic_qr_feature_cap': (1, 32), 'probabilistic_regime_min_obs': (3, 120),
            'probabilistic_feature_filter_min_obs': (3, 120), 'probabilistic_feature_filter_k': (1, 64), 'probabilistic_knn_k': (1, 128),
            'probabilistic_knn_min_obs': (3, 120), 'probabilistic_knn_distance_power': (0.5, 4.0), 'probabilistic_knn_weight_eps': (1e-9, 1e-2),
            'probabilistic_bucket_n_bins': (2, 10), 'probabilistic_bucket_min_obs': (3, 120), 'probabilistic_bucket_match_min_features': (1, 12),
            'probabilistic_bucket_max_features': (1, 16), 'probabilistic_parametric_min_sigma': (1e-8, 0.10), 'probabilistic_parametric_max_sigma_mult': (1.0, 10.0),
            'probabilistic_hybrid_weight': (0.0, 1.0), 'probabilistic_hybrid_min_qr_weight': (0.0, 1.0), 'probabilistic_hybrid_max_qr_weight': (0.0, 1.0),
            'huber_delta': (0.1, 10.0), 'signal_score_blend': (0.0, 2.0), 'lambdarank_lookback': (3, 60), 'lambdarank_temperature': (0.1, 5.0),
            'directional_classifier_lookback': (3, 60), 'directional_classifier_threshold': (0.0, 1.0), 'directional_classifier_confidence_scale': (0.0, 5.0),
            'logistic_loss_lookback': (3, 60), 'logistic_loss_l2': (1e-8, 10.0), 'logistic_loss_threshold': (0.0, 1.0),
            'logistic_loss_confidence_scale': (0.0, 5.0), 'quantile_loss_lookback': (3, 60), 'quantile_loss_q': (0.01, 0.99),
            'quantile_loss_alpha': (0.0, 10.0), 'quantile_loss_confidence_scale': (0.0, 5.0), 'top_k_classifier_lookback': (3, 60),
            'top_k_classifier_k': (1, 64), 'top_k_classifier_threshold': (0.0, 1.0), 'top_k_classifier_confidence_scale': (0.0, 5.0),
            'lambdarank_real_lookback': (3, 60), 'lambdarank_real_temperature': (0.1, 5.0), 'lambdarank_real_gain_power': (0.1, 5.0),
            'lambdarank_real_pair_power': (0.1, 5.0), 'lambdarank_real_l2': (0.0, 10.0), 'multi_loss_rank_weight': (0.0, 1.0),
            'multi_loss_direction_weight': (0.0, 1.0), 'multi_loss_return_weight': (0.0, 1.0), 'multi_loss_logistic_weight': (0.0, 1.0),
            'multi_loss_topk_weight': (0.0, 1.0), 'multi_loss_temperature': (0.1, 5.0), 'multi_loss_l2': (0.0, 10.0), 'feature_mu_blend': (0.0, 1.0),
            'feature_mu_k': (1, 120), 'feature_mu_min_obs': (3, 120), 'feature_mu_cols_max': (1, 32), 'regime_lookback': (6, 120),
            'regime_quantile_low': (0.01, 0.49), 'regime_quantile_high': (0.51, 0.99), 'correlation_lookback': (3, 120), 'correlation_min_periods': (2, 60),
            'correlation_shrink_to_identity': (0.0, 1.0), 'covariance_shrink_to_diagonal': (0.0, 1.0), 'covariance_jitter': (1e-12, 1e-2),
            'covariance_lookback_low': (3, 120), 'covariance_lookback_mid': (3, 120), 'covariance_lookback_high': (3, 180), 'covariance_halflife_low': (1, 60),
            'covariance_halflife_mid': (1, 60), 'covariance_halflife_high': (1, 120), 'factor_covariance_n_factors': (1, 20), 'factor_covariance_min_obs': (6, 240),
            'factor_covariance_shrink_to_diagonal': (0.0, 1.0), 'factor_covariance_blend': (0.0, 1.0), 'factor_model_n_factors': (1, 20),
            'factor_model_min_obs': (6, 240), 'factor_model_mu_blend': (0.0, 1.0), 'factor_model_residual_blend': (0.0, 1.0),
            'factor_model_covariance_blend': (0.0, 1.0), 'factor_model_shrink_to_diagonal': (0.0, 1.0), 'correlation_allocator_blend': (0.0, 1.0),
            'correlation_penalty_strength': (0.0, 5.0), 'correlation_penalty_power': (0.1, 5.0), 'mean_variance_risk_aversion': (0.0, 20.0),
            'risk_budget_strength': (0.0, 5.0), 'cluster_corr_threshold': (0.0, 0.99), 'ewma_halflife': (1, 60), 'score_clip': (0.1, 20.0),
            'vol_dependent_cap_threshold_low': (0.0, 0.50), 'vol_dependent_cap_threshold_high': (0.0, 1.0), 'vol_dependent_cap_low_mult': (0.1, 3.0),
            'vol_dependent_cap_high_mult': (0.1, 3.0), 'vol_dependent_cap_min': (0.01, 1.0), 'vol_dependent_cap_max': (0.01, 1.0),
            'corr_dependent_cap_threshold_low': (0.0, 0.99), 'corr_dependent_cap_threshold_high': (0.0, 0.99), 'corr_dependent_cap_low_mult': (0.1, 3.0),
            'corr_dependent_cap_high_mult': (0.1, 3.0), 'corr_dependent_cap_min': (0.01, 1.0), 'corr_dependent_cap_max': (0.01, 1.0),
            'dispersion_dependent_cap_threshold_low': (0.0, 5.0), 'dispersion_dependent_cap_threshold_high': (0.0, 5.0), 'dispersion_dependent_cap_low_mult': (0.1, 3.0),
            'dispersion_dependent_cap_high_mult': (0.1, 3.0), 'dispersion_dependent_cap_min': (0.01, 1.0), 'dispersion_dependent_cap_max': (0.01, 1.0),
            'regime_dependent_cap_low_mult': (0.1, 3.0), 'regime_dependent_cap_mid_mult': (0.1, 3.0), 'regime_dependent_cap_high_mult': (0.1, 3.0),
            'regime_dependent_cap_min': (0.01, 1.0), 'regime_dependent_cap_max': (0.01, 1.0), 'dispersion_gate_threshold': (0.0, 5.0),
            'dispersion_gate_min_active_weight': (0.0, 1.0), 'dispersion_sigma_lookback': (3, 120), 'dispersion_sigma_strength': (0.0, 5.0),
            'dispersion_sigma_floor_mult': (0.1, 3.0), 'dispersion_sigma_ceiling_mult': (0.1, 5.0), 'dispersion_risk_strength': (0.0, 5.0),
            'dispersion_risk_floor_mult': (0.1, 3.0), 'dispersion_risk_ceiling_mult': (0.1, 5.0), 'dispersion_top_k_threshold_low': (0.0, 5.0),
            'dispersion_top_k_threshold_high': (0.0, 5.0), 'dispersion_top_k_low_mult': (0.1, 3.0), 'dispersion_top_k_high_mult': (0.1, 3.0),
            'dispersion_top_k_min_k': (1, 64), 'dispersion_top_k_max_k': (1, 128), 'dispersion_vol_target_threshold_low': (0.0, 5.0),
            'dispersion_vol_target_threshold_high': (0.0, 5.0), 'dispersion_vol_target_low_mult': (0.1, 3.0), 'dispersion_vol_target_high_mult': (0.1, 3.0),
            'dispersion_vol_target_min': (0.005, 0.20), 'dispersion_vol_target_max': (0.005, 0.30), 'low_signal_fallback_threshold': (0.0, 1.0),
            'low_signal_fallback_min_model_weight': (0.0, 1.0), 'regime_universe_keep_frac_low': (0.0, 1.0), 'regime_universe_keep_frac_mid': (0.0, 1.0),
            'regime_universe_keep_frac_high': (0.0, 1.0), 'regime_universe_min_assets': (1, 128), 'regime_universe_max_assets': (1, 256),
            'regime_universe_low_offensive_vol_tilt': (0.0, 2.0), 'regime_universe_high_defensive_vol_tilt': (0.0, 2.0), 'transaction_cost_commission_bps': (0.0, 200.0),
            'transaction_cost_slippage_bps': (0.0, 200.0), 'transaction_cost_spread_bps': (0.0, 200.0), 'transaction_cost_market_impact_bps': (0.0, 500.0),
            'transaction_cost_market_impact_power': (0.1, 5.0), 'transaction_cost_min_trade_weight': (0.0, 1.0), 'holding_cost_annual_bps': (0.0, 500.0),
            'tax_short_term_rate': (0.0, 1.0), 'tax_long_term_rate': (0.0, 1.0), 'tax_long_term_threshold_months': (1, 120), 'tax_loss_credit_rate': (0.0, 1.0),
        }
        if name in explicit_ranges:
            return explicit_ranges[name]
        try:
            value = float(default)
        except Exception:
            return None, None
        span = max(abs(value) * 0.5, 1.0 if kind == 'int' else 0.25)
        low = 0.0 if value >= 0 else value - span
        high = value + span
        if kind == 'int':
            return int(max(0, np.floor(low))), int(max(np.ceil(high), np.floor(low) + 1))
        return float(low), float(high)

    def _infer_log_scale(name: str, kind: str, default: Any) -> bool:
        if kind not in {'continuous', 'int'}:
            return False
        log_names = ('alpha', 'l2', 'jitter', 'eps', 'floor', 'halflife', 'bps', 'penalty_strength')
        if any(tok in name for tok in log_names):
            try:
                return float(default) > 0
            except Exception:
                return True
        return False

    def _infer_mutation_step(name: str, kind: str, low: Any, high: Any, default: Any, values: Optional[List[Any]]) -> Any:
        if kind == 'bool':
            return 1
        if kind == 'mapping':
            return None
        if kind == 'categorical':
            if isinstance(values, list):
                return 1 if len(values) > 1 else 0
            return 1
        if kind == 'int':
            try:
                lo = int(low) if low is not None else int(default)
                hi = int(high) if high is not None else int(default)
                return max(1, int(np.ceil((hi - lo) / 10.0)))
            except Exception:
                return 1
        try:
            lo = float(low) if low is not None else float(default)
            hi = float(high) if high is not None else float(default)
            span = max(hi - lo, 1e-12)
            return float(max(span / 10.0, 1e-4))
        except Exception:
            return 0.1

    for field_obj in fields(MicroPipelineConfig):
        name = str(field_obj.name)
        default = _json_safe(getattr(default_cfg, name))
        type_str = str(field_obj.type)
        nullable = ('Optional' in type_str) or ('NoneType' in type_str)
        kind, values = _infer_kind_and_values(field_obj)
        low, high = _infer_bounds(name, kind, default)
        schema[name] = {
            'dtype': type_str.replace('typing.', ''),
            'nullable': bool(nullable),
            'kind': kind,
            'values': list(values) if isinstance(values, list) else None,
            'low': _json_safe(low) if low is not None else None,
            'high': _json_safe(high) if high is not None else None,
            'log_scale': bool(_infer_log_scale(name, kind, default)),
            'active_if': active_if_map.get(name),
            'group': _infer_group(name),
            'mutation_step': _json_safe(_infer_mutation_step(name, kind, low, high, default, values)),
            'safe_default': default,
        }
    return schema


# ============================================================
# Tuning-space activity helpers
# ============================================================

def _param_spec_has_variation(spec: Any) -> bool:
    if isinstance(spec, dict):
        if "values" in spec and isinstance(spec.get("values"), (list, tuple)):
            vals = list(spec.get("values") or [])
            return len({_json_safe(v) for v in vals}) > 1
        low = spec.get("low", None)
        high = spec.get("high", None)
        if low is not None and high is not None:
            try:
                return float(low) != float(high)
            except Exception:
                return str(low) != str(high)
    if isinstance(spec, (list, tuple)):
        vals = list(spec)
        return len({_json_safe(v) for v in vals}) > 1
    return False


def _param_space_can_activate(param_space: Optional[Dict[str, Any]], key: str, *, truthy_only: bool = False) -> bool:
    if not isinstance(param_space, dict) or key not in param_space:
        return False
    spec = param_space.get(key)
    values: List[Any] = []
    if isinstance(spec, dict):
        if "values" in spec and isinstance(spec.get("values"), (list, tuple)):
            values = list(spec.get("values") or [])
        elif spec.get("low", None) is not None and spec.get("high", None) is not None:
            values = [spec.get("low"), spec.get("high")]
    elif isinstance(spec, (list, tuple)):
        values = list(spec)
    if not values:
        return _param_spec_has_variation(spec)
    if truthy_only:
        return any(bool(v) for v in values)
    return len({_json_safe(v) for v in values}) > 1


def build_engine_param_activity_audit(
    cfg: MicroPipelineConfig,
    *,
    param_space: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    valid_fields = [f.name for f in fields(MicroPipelineConfig)]
    mode_prob = str(getattr(cfg, "probabilistic_mode", "none") or "none")
    mode_signal = str(getattr(cfg, "signal_mode", "mu_sigma") or "mu_sigma")
    regime_mode = str(getattr(cfg, "regime_mode", "none") or "none")

    active: set[str] = set()
    inactive_reason: Dict[str, str] = {}
    controllers: Dict[str, str] = {}

    def mark_active(*names: str) -> None:
        for name in names:
            if name in valid_fields:
                active.add(name)
                inactive_reason.pop(name, None)

    def mark_group(names: List[str], is_active: bool, reason: str, controller: Optional[str] = None) -> None:
        for name in names:
            if name not in valid_fields:
                continue
            if is_active:
                active.add(name)
                inactive_reason.pop(name, None)
            else:
                inactive_reason[name] = reason
                if controller:
                    controllers[name] = controller

    core_always = [
        "sigma_power_alpha", "date_col", "asset_col", "return_col", "min_train", "lookback_mu", "lookback_sigma",
        "sigma_floor", "temperature", "top_k", "long_only", "weight_shrink", "inertia", "deadband",
        "deadband_threshold", "regime_mode", "mu_regime_mode", "ewma_sigma", "ewma_halflife",
        "score_normalize", "score_clip", "covariance_mode", "correlation_lookback", "correlation_min_periods",
        "correlation_shrink_to_identity", "covariance_shrink_to_diagonal", "covariance_jitter",
        "covariance_lookback_low", "covariance_lookback_mid", "covariance_lookback_high",
        "covariance_halflife_low", "covariance_halflife_mid", "covariance_halflife_high",
        "correlation_aware_allocation", "correlation_allocator_method", "correlation_allocator_blend",
        "correlation_penalty_strength", "correlation_penalty_power", "correlation_use_abs",
        "mean_variance_risk_aversion", "risk_budget_strength", "cluster_corr_threshold",
        "signal_mode", "signal_score_blend", "pure_cs_baseline",
        "selection_policy", "manual_composite_weights_enabled",
    ]
    mark_active(*core_always)

    mark_group(["asset_weight_cap", "w_cap"], bool(cfg.long_only), "inactive when long_only=False", controller="long_only")

    chooser_active = str(getattr(cfg, "selection_policy", "fixed_composite_score") or "fixed_composite_score") == "fixed_composite_score" or _param_space_can_activate(param_space, "selection_policy")
    manual_weights_active = (chooser_active and bool(getattr(cfg, "manual_composite_weights_enabled", False))) or _param_space_can_activate(param_space, "manual_composite_weights_enabled", truthy_only=True)
    mark_group(["composite_profile"], chooser_active, "requires selection_policy='fixed_composite_score'", controller="selection_policy")
    mark_group(["manual_composite_weights"], manual_weights_active, "requires selection_policy='fixed_composite_score' and manual_composite_weights_enabled=True", controller="manual_composite_weights_enabled")

    vol_cap_group = ["vol_dependent_cap_threshold_low", "vol_dependent_cap_threshold_high", "vol_dependent_cap_low_mult", "vol_dependent_cap_high_mult", "vol_dependent_cap_min", "vol_dependent_cap_max"]
    corr_cap_group = ["corr_dependent_cap_threshold_low", "corr_dependent_cap_threshold_high", "corr_dependent_cap_low_mult", "corr_dependent_cap_high_mult", "corr_dependent_cap_min", "corr_dependent_cap_max"]
    disp_cap_group = ["dispersion_dependent_cap_threshold_low", "dispersion_dependent_cap_threshold_high", "dispersion_dependent_cap_low_mult", "dispersion_dependent_cap_high_mult", "dispersion_dependent_cap_min", "dispersion_dependent_cap_max"]
    regime_cap_group = ["regime_dependent_cap_low_mult", "regime_dependent_cap_mid_mult", "regime_dependent_cap_high_mult", "regime_dependent_cap_min", "regime_dependent_cap_max"]

    mark_active("vol_dependent_cap_enabled", "corr_dependent_cap_enabled", "dispersion_dependent_cap_enabled", "regime_dependent_cap_enabled")
    mark_group(vol_cap_group, bool(cfg.vol_dependent_cap_enabled) or _param_space_can_activate(param_space, "vol_dependent_cap_enabled", truthy_only=True), "requires vol_dependent_cap_enabled=True", controller="vol_dependent_cap_enabled")
    mark_group(corr_cap_group, bool(cfg.corr_dependent_cap_enabled) or _param_space_can_activate(param_space, "corr_dependent_cap_enabled", truthy_only=True), "requires corr_dependent_cap_enabled=True", controller="corr_dependent_cap_enabled")
    mark_group(disp_cap_group, bool(cfg.dispersion_dependent_cap_enabled) or _param_space_can_activate(param_space, "dispersion_dependent_cap_enabled", truthy_only=True), "requires dispersion_dependent_cap_enabled=True", controller="dispersion_dependent_cap_enabled")
    mark_group(regime_cap_group, bool(cfg.regime_dependent_cap_enabled) or _param_space_can_activate(param_space, "regime_dependent_cap_enabled", truthy_only=True), "requires regime_dependent_cap_enabled=True", controller="regime_dependent_cap_enabled")

    mark_active("turnover_penalty_strength", "turnover_penalty_power", "turnover_penalty_target", "turnover_penalty_max_turnover", "turnover_constraint_max_turnover")

    mark_active("dispersion_gate", "dispersion_gate_threshold", "dispersion_gate_min_active_weight")
    mark_active("dispersion_sigma_enabled")
    mark_group(["dispersion_sigma_lookback", "dispersion_sigma_strength", "dispersion_sigma_floor_mult", "dispersion_sigma_ceiling_mult"], bool(cfg.dispersion_sigma_enabled) or _param_space_can_activate(param_space, "dispersion_sigma_enabled", truthy_only=True), "requires dispersion_sigma_enabled=True", controller="dispersion_sigma_enabled")

    mark_active("dispersion_risk_model_enabled")
    mark_group(["dispersion_risk_strength", "dispersion_risk_floor_mult", "dispersion_risk_ceiling_mult", "dispersion_risk_apply_to_covariance"], bool(cfg.dispersion_risk_model_enabled) or _param_space_can_activate(param_space, "dispersion_risk_model_enabled", truthy_only=True), "requires dispersion_risk_model_enabled=True", controller="dispersion_risk_model_enabled")

    mark_active("dispersion_top_k_enabled")
    mark_group(["dispersion_top_k_threshold_low", "dispersion_top_k_threshold_high", "dispersion_top_k_low_mult", "dispersion_top_k_high_mult", "dispersion_top_k_min_k", "dispersion_top_k_max_k"], bool(cfg.dispersion_top_k_enabled) or _param_space_can_activate(param_space, "dispersion_top_k_enabled", truthy_only=True), "requires dispersion_top_k_enabled=True", controller="dispersion_top_k_enabled")

    disp_vol_target_controller_active = (bool(cfg.dispersion_vol_target_enabled) and bool(cfg.vol_targeting)) or _param_space_can_activate(param_space, "dispersion_vol_target_enabled", truthy_only=True)
    mark_active("dispersion_vol_target_enabled")
    mark_group(["dispersion_vol_target_threshold_low", "dispersion_vol_target_threshold_high", "dispersion_vol_target_low_mult", "dispersion_vol_target_high_mult", "dispersion_vol_target_min", "dispersion_vol_target_max"], disp_vol_target_controller_active, "requires dispersion_vol_target_enabled=True and vol_targeting=True", controller="dispersion_vol_target_enabled")

    mark_active("low_signal_fallback_to_ew")
    mark_group(["low_signal_fallback_threshold", "low_signal_fallback_mode", "low_signal_fallback_min_model_weight"], bool(cfg.low_signal_fallback_to_ew) or _param_space_can_activate(param_space, "low_signal_fallback_to_ew", truthy_only=True), "requires low_signal_fallback_to_ew=True", controller="low_signal_fallback_to_ew")

    regime_quantile_active = regime_mode == "quantile" or _param_space_can_activate(param_space, "regime_mode")
    mark_group(["regime_lookback", "regime_quantile_low", "regime_quantile_high"], regime_quantile_active, "requires regime_mode='quantile'", controller="regime_mode")

    reg_cov_active = bool(cfg.regime_dependent_covariance) or _param_space_can_activate(param_space, "regime_dependent_covariance", truthy_only=True)
    mark_active("regime_dependent_covariance")
    mark_group(["covariance_lookback_low", "covariance_lookback_mid", "covariance_lookback_high", "covariance_halflife_low", "covariance_halflife_mid", "covariance_halflife_high"], reg_cov_active, "requires regime_dependent_covariance=True", controller="regime_dependent_covariance")

    factor_cov_active = bool(cfg.factor_covariance_active) or _param_space_can_activate(param_space, "factor_covariance_active", truthy_only=True)
    mark_active("factor_covariance_active")
    mark_group(["factor_covariance_n_factors", "factor_covariance_min_obs", "factor_covariance_shrink_to_diagonal", "factor_covariance_blend"], factor_cov_active, "requires factor_covariance_active=True", controller="factor_covariance_active")

    factor_model_active = bool(cfg.factor_model_active) or _param_space_can_activate(param_space, "factor_model_active", truthy_only=True)
    mark_active("factor_model_active")
    mark_group(["factor_model_n_factors", "factor_model_min_obs", "factor_model_mu_blend", "factor_model_residual_blend", "factor_model_covariance_blend", "factor_model_shrink_to_diagonal"], factor_model_active, "requires factor_model_active=True", controller="factor_model_active")

    vol_target_active = bool(cfg.vol_targeting) or _param_space_can_activate(param_space, "vol_targeting", truthy_only=True)
    mark_active("vol_targeting")
    mark_group(["covariance_aware_vol_targeting", "target_portfolio_vol_monthly", "vol_target_floor_mult", "vol_target_ceiling_mult"], vol_target_active, "requires vol_targeting=True", controller="vol_targeting")

    regime_derisk_active = vol_target_active and (regime_mode != "none" or _param_space_can_activate(param_space, "regime_mode"))
    mark_group(["regime_derisk_low", "regime_derisk_mid", "regime_derisk_high"], regime_derisk_active, "requires vol_targeting/regime context", controller="regime_mode")

    mark_active("store_correlation_snapshots", "store_sigma_fwd_snapshots")

    prob_group_base = [
        "probabilistic_q_low", "probabilistic_q_high", "probabilistic_min_obs", "probabilistic_interval_penalty_weight",
        "probabilistic_downside_penalty_weight", "probabilistic_confidence_scale", "probabilistic_confidence_min_mult",
        "probabilistic_confidence_max_mult", "probabilistic_overlay_strength", "probabilistic_overlay_blend"
    ]
    prob_any_active = mode_prob != "none" or _param_space_can_activate(param_space, "probabilistic_mode")
    mark_active("probabilistic_mode")
    mark_group(prob_group_base, prob_any_active, "requires probabilistic_mode!='none'", controller="probabilistic_mode")

    qr_modes = {"quantile_regression", "hybrid"}
    feature_modes = {"historical_by_features", "parametric_feature_aware", "feature_bucketed_historical", "knn_historical", "hybrid"}
    regime_modes = {"historical_by_regime", "hybrid"}
    knn_modes = {"knn_historical"}
    bucket_modes = {"feature_bucketed_historical"}
    parametric_modes = {"parametric_feature_aware"}
    hybrid_modes = {"hybrid"}

    qr_active = (mode_prob in qr_modes) or (_param_space_can_activate(param_space, "probabilistic_mode") and any(m in list((param_space or {}).get("probabilistic_mode", {}).get("values", [])) for m in qr_modes))
    feature_mode_active = (mode_prob in feature_modes) or (_param_space_can_activate(param_space, "probabilistic_mode") and any(m in list((param_space or {}).get("probabilistic_mode", {}).get("values", [])) for m in feature_modes))
    regime_prob_active = (bool(cfg.probabilistic_regime_aware) and prob_any_active) or (mode_prob in regime_modes) or _param_space_can_activate(param_space, "probabilistic_regime_aware", truthy_only=True)
    feature_filter_active = feature_mode_active and (bool(cfg.probabilistic_feature_filter_enabled) or _param_space_can_activate(param_space, "probabilistic_feature_filter_enabled", truthy_only=True))
    knn_active = (mode_prob in knn_modes) or (_param_space_can_activate(param_space, "probabilistic_mode") and any(m in list((param_space or {}).get("probabilistic_mode", {}).get("values", [])) for m in knn_modes))
    bucket_active = (mode_prob in bucket_modes) or (_param_space_can_activate(param_space, "probabilistic_mode") and any(m in list((param_space or {}).get("probabilistic_mode", {}).get("values", [])) for m in bucket_modes))
    parametric_active = (mode_prob in parametric_modes) or (_param_space_can_activate(param_space, "probabilistic_mode") and any(m in list((param_space or {}).get("probabilistic_mode", {}).get("values", [])) for m in parametric_modes))
    hybrid_active = (mode_prob in hybrid_modes) or (_param_space_can_activate(param_space, "probabilistic_mode") and any(m in list((param_space or {}).get("probabilistic_mode", {}).get("values", [])) for m in hybrid_modes))

    mark_group(["probabilistic_qr_alpha", "probabilistic_qr_solver", "probabilistic_qr_feature_cap"], qr_active, "requires probabilistic quantile-regression style mode", controller="probabilistic_mode")
    mark_active("probabilistic_regime_aware")
    mark_group(["probabilistic_regime_min_obs"], regime_prob_active, "requires regime-aware probabilistic overlay", controller="probabilistic_regime_aware")
    mark_active("probabilistic_feature_filter_enabled")
    mark_group(["probabilistic_feature_filter_min_obs", "probabilistic_feature_filter_k"], feature_filter_active, "requires probabilistic feature filter enabled in feature-aware mode", controller="probabilistic_feature_filter_enabled")
    mark_group(["probabilistic_knn_k", "probabilistic_knn_min_obs", "probabilistic_knn_distance_power", "probabilistic_knn_weighted_quantiles", "probabilistic_knn_weight_eps"], knn_active, "requires probabilistic_mode='knn_historical'", controller="probabilistic_mode")
    mark_group(["probabilistic_bucket_n_bins", "probabilistic_bucket_min_obs", "probabilistic_bucket_match_min_features", "probabilistic_bucket_max_features"], bucket_active, "requires probabilistic_mode='feature_bucketed_historical'", controller="probabilistic_mode")
    mark_group(["probabilistic_parametric_min_sigma", "probabilistic_parametric_max_sigma_mult", "probabilistic_parametric_use_neighbor_weights"], parametric_active, "requires probabilistic_mode='parametric_feature_aware'", controller="probabilistic_mode")
    mark_group(["probabilistic_hybrid_weight", "probabilistic_hybrid_min_qr_weight", "probabilistic_hybrid_max_qr_weight", "probabilistic_hybrid_use_confidence"], hybrid_active, "requires probabilistic_mode='hybrid'", controller="probabilistic_mode")

    signal_groups = {
        "huber_mu": ["huber_delta"],
        "lambdarank_like": ["lambdarank_lookback", "lambdarank_temperature"],
        "lambdarank_real": ["lambdarank_real_lookback", "lambdarank_real_temperature", "lambdarank_real_gain_power", "lambdarank_real_pair_power", "lambdarank_real_l2"],
        "directional_classifier": ["directional_classifier_lookback", "directional_classifier_threshold", "directional_classifier_confidence_scale"],
        "logistic_loss": ["logistic_loss_lookback", "logistic_loss_l2", "logistic_loss_threshold", "logistic_loss_confidence_scale"],
        "top_k_classifier": ["top_k_classifier_lookback", "top_k_classifier_k", "top_k_classifier_threshold", "top_k_classifier_confidence_scale"],
        "quantile_loss": ["quantile_loss_lookback", "quantile_loss_q", "quantile_loss_alpha", "quantile_loss_confidence_scale"],
    }
    for mode_name, names in signal_groups.items():
        mode_active = (mode_signal == mode_name) or (_param_space_can_activate(param_space, "signal_mode") and any(m == mode_name for m in list((param_space or {}).get("signal_mode", {}).get("values", []))))
        mark_group(names, mode_active, f"requires signal_mode='{mode_name}'", controller="signal_mode")

    multi_loss_active = bool(cfg.multi_loss_enabled) or _param_space_can_activate(param_space, "multi_loss_enabled", truthy_only=True)
    mark_active("multi_loss_enabled")
    mark_group(["multi_loss_rank_weight", "multi_loss_direction_weight", "multi_loss_return_weight", "multi_loss_logistic_weight", "multi_loss_topk_weight", "multi_loss_temperature", "multi_loss_l2"], multi_loss_active, "requires multi_loss_enabled=True", controller="multi_loss_enabled")

    feature_mu_active = bool(cfg.feature_mu_enabled) or _param_space_can_activate(param_space, "feature_mu_enabled", truthy_only=True)
    mark_active("feature_mu_enabled")
    mark_group(["feature_mu_blend", "feature_mu_k", "feature_mu_min_obs", "feature_mu_cols_max"], feature_mu_active, "requires feature_mu_enabled=True", controller="feature_mu_enabled")

    regime_univ_active = bool(cfg.regime_dependent_universe_enabled) or _param_space_can_activate(param_space, "regime_dependent_universe_enabled", truthy_only=True)
    mark_active("regime_dependent_universe_enabled")
    mark_group(["regime_universe_selection_metric", "regime_universe_keep_frac_low", "regime_universe_keep_frac_mid", "regime_universe_keep_frac_high", "regime_universe_min_assets", "regime_universe_max_assets", "regime_universe_low_offensive_vol_tilt", "regime_universe_high_defensive_vol_tilt"], regime_univ_active, "requires regime_dependent_universe_enabled=True", controller="regime_dependent_universe_enabled")

    cost_active = bool(cfg.cost_model_enabled) or _param_space_can_activate(param_space, "cost_model_enabled", truthy_only=True)
    mark_active("cost_model_enabled")
    mark_group(["transaction_cost_commission_bps", "transaction_cost_slippage_bps", "transaction_cost_spread_bps", "transaction_cost_market_impact_bps", "transaction_cost_market_impact_power", "transaction_cost_min_trade_weight", "holding_cost_annual_bps"], cost_active, "requires cost_model_enabled=True", controller="cost_model_enabled")

    tax_active = bool(cfg.tax_model_enabled) or _param_space_can_activate(param_space, "tax_model_enabled", truthy_only=True)
    mark_active("tax_model_enabled")
    mark_group(["tax_short_term_rate", "tax_long_term_rate", "tax_long_term_threshold_months", "tax_apply_loss_credit", "tax_loss_credit_rate"], tax_active, "requires tax_model_enabled=True", controller="tax_model_enabled")

    rows: List[Dict[str, Any]] = []
    for name in valid_fields:
        is_active = name in active
        rows.append({
            "param": name,
            "active": bool(is_active),
            "status": "active" if is_active else "inactive",
            "reason": "" if is_active else inactive_reason.get(name, "inactive in current engine context"),
            "controller": controllers.get(name, ""),
            "current_value": _json_safe(getattr(cfg, name)),
            "in_param_space": bool(isinstance(param_space, dict) and name in param_space),
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["active", "param"], ascending=[False, True]).reset_index(drop=True)




def build_engine_aware_tunable_catalog(
    cfg: MicroPipelineConfig,
    param_space: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Build a compact engine-aware catalog of tunable config fields.

    This is a thin wrapper around the existing activity audit and engine-aware
    param-space sanitiser. It is meant for UI/reporting and search-schema
    inspection, not for executing any optimisation logic.
    """
    audit = build_engine_param_activity_audit(cfg, param_space=param_space)
    if not isinstance(audit, pd.DataFrame) or audit.empty:
        return pd.DataFrame(
            columns=[
                "param",
                "dtype",
                "nullable",
                "active",
                "controller",
                "reason",
                "current_value",
                "in_param_space",
                "kept_after_sanitization",
            ]
        )

    sanitized_info = sanitize_param_space_for_engine(dict(param_space or {}), cfg) if isinstance(param_space, dict) else {
        "sanitized_param_space": {},
        "kept_params": [],
    }
    kept_params = set(str(k) for k in list(sanitized_info.get("kept_params", []) or []))

    field_map = {f.name: f for f in fields(MicroPipelineConfig)}

    def _dtype_name(field_name: str) -> str:
        fld = field_map.get(field_name)
        if fld is None:
            return "unknown"
        typ = fld.type
        s = str(typ)
        if "Literal" in s:
            return "literal"
        if "bool" in s:
            return "bool"
        if "int" in s and "float" not in s:
            return "int"
        if "float" in s:
            return "float"
        if "str" in s:
            return "str"
        return s.replace("typing.", "")

    def _is_nullable(field_name: str) -> bool:
        fld = field_map.get(field_name)
        if fld is None:
            return False
        s = str(fld.type)
        return ("Optional" in s) or ("NoneType" in s) or ("None" in s and "Literal" not in s)

    catalog = audit.copy()
    catalog["dtype"] = catalog["param"].map(_dtype_name)
    catalog["nullable"] = catalog["param"].map(_is_nullable).astype(bool)
    if "controller" not in catalog.columns:
        catalog["controller"] = ""
    if "reason" not in catalog.columns:
        catalog["reason"] = ""
    if "current_value" not in catalog.columns:
        catalog["current_value"] = catalog["param"].map(lambda p: _json_safe(getattr(cfg, str(p), None)))
    if "in_param_space" not in catalog.columns:
        catalog["in_param_space"] = catalog["param"].map(lambda p: bool(isinstance(param_space, dict) and str(p) in param_space))
    catalog["kept_after_sanitization"] = catalog["param"].map(lambda p: str(p) in kept_params)

    ordered_cols = [
        "param",
        "dtype",
        "nullable",
        "active",
        "controller",
        "reason",
        "current_value",
        "in_param_space",
        "kept_after_sanitization",
    ]
    for col in ordered_cols:
        if col not in catalog.columns:
            catalog[col] = np.nan
    catalog = catalog[ordered_cols].sort_values(["active", "param"], ascending=[False, True]).reset_index(drop=True)
    return catalog



def refine_param_space_around_payloads(
    base_param_space: dict,
    seed_payloads: Sequence[dict],
    *,
    shrink_factor: float = 0.5,
) -> dict:
    """
    Refine a parameter space around one or more seed payloads.

    Supported param-space shapes
    ----------------------------
    - discrete grids as lists / tuples
    - dict specs with ``{"values": [...]}``
    - numeric range specs with ``{"low": ..., "high": ...}``

    The helper is deliberately conservative:
    - it preserves unknown spec keys
    - it only narrows when it can do so safely
    - it keeps at least a small amount of variation
    - it respects MicroPipelineConfig field typing where practical
    """
    if not isinstance(base_param_space, dict) or not base_param_space:
        return {}

    try:
        shrink = float(shrink_factor)
    except Exception:
        shrink = 0.5
    if not np.isfinite(shrink):
        shrink = 0.5
    shrink = float(np.clip(shrink, 0.05, 1.0))

    field_map = {f.name: f for f in fields(MicroPipelineConfig)}

    cleaned_seed_payloads: List[Dict[str, Any]] = []
    for raw in list(seed_payloads or []):
        if not isinstance(raw, dict):
            continue
        cleaned = coerce_candidate_payload_for_config(dict(raw))
        if cleaned:
            cleaned_seed_payloads.append(cleaned)

    def _is_int_field(name: str) -> bool:
        fld = field_map.get(str(name))
        if fld is None:
            return False
        type_str = str(fld.type)
        return ("int" in type_str) and ("float" not in type_str)

    def _coerce_like_field(name: str, value: Any) -> Any:
        fld = field_map.get(str(name))
        if fld is None:
            return value
        try:
            return _coerce_field_value(value, fld.type)
        except Exception:
            return value

    def _json_key(v: Any) -> str:
        return json.dumps(_json_safe(v), sort_keys=True, default=str)

    def _unique_preserve(seq: Sequence[Any]) -> List[Any]:
        out: List[Any] = []
        seen: set[str] = set()
        for item in list(seq or []):
            key = _json_key(item)
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    def _seed_values_for(name: str) -> List[Any]:
        vals: List[Any] = []
        for payload in cleaned_seed_payloads:
            if str(name) not in payload:
                continue
            val = payload.get(str(name))
            if _looks_like_null(val):
                continue
            vals.append(_coerce_like_field(str(name), val))
        return _unique_preserve(vals)

    refined: Dict[str, Any] = {}

    for key, spec in dict(base_param_space).items():
        name = str(key)
        seed_vals = _seed_values_for(name)

        if isinstance(spec, dict):
            spec_out = dict(spec)
            if "values" in spec and isinstance(spec.get("values"), (list, tuple)):
                raw_values = [_coerce_like_field(name, v) for v in list(spec.get("values") or [])]
                raw_values = _unique_preserve(raw_values)
                if len(raw_values) <= 1:
                    refined[name] = spec_out
                    continue

                numeric_vals = pd.to_numeric(pd.Series(raw_values), errors="coerce")
                all_numeric = bool(numeric_vals.notna().all())

                if not seed_vals:
                    keep_n = max(1, int(np.ceil(len(raw_values) * shrink)))
                    spec_out["values"] = list(raw_values[:keep_n])
                    refined[name] = spec_out
                    continue

                if all_numeric:
                    ordered_pairs = sorted(
                        [(float(v), raw) for v, raw in zip(numeric_vals.tolist(), raw_values)],
                        key=lambda x: x[0],
                    )
                    ordered_numeric = np.asarray([v for v, _ in ordered_pairs], dtype="float64")
                    ordered_raw = [raw for _, raw in ordered_pairs]
                    seed_num = pd.to_numeric(pd.Series(seed_vals), errors="coerce").dropna().astype(float).tolist()
                    if seed_num:
                        keep_n = max(2, int(np.ceil(len(raw_values) * shrink)))
                        distances = np.min(
                            np.abs(ordered_numeric[:, None] - np.asarray(seed_num, dtype="float64")[None, :]),
                            axis=1,
                        )
                        keep_idx = np.argsort(distances)[:keep_n]
                        keep = [ordered_raw[int(i)] for i in keep_idx]
                        keep = _unique_preserve(keep + list(seed_vals))
                        spec_out["values"] = keep
                    else:
                        keep_n = max(1, int(np.ceil(len(raw_values) * shrink)))
                        spec_out["values"] = list(raw_values[:keep_n])
                else:
                    keep = [v for v in raw_values if any(_json_key(v) == _json_key(sv) for sv in seed_vals)]
                    if not keep:
                        keep_n = max(1, int(np.ceil(len(raw_values) * shrink)))
                        keep = list(raw_values[:keep_n])
                    spec_out["values"] = _unique_preserve(keep)
                refined[name] = spec_out
                continue

            if ("low" in spec) and ("high" in spec):
                low_raw = spec.get("low")
                high_raw = spec.get("high")
                try:
                    low = float(low_raw)
                    high = float(high_raw)
                except Exception:
                    refined[name] = spec_out
                    continue
                if not np.isfinite(low) or not np.isfinite(high):
                    refined[name] = spec_out
                    continue
                if high < low:
                    low, high = high, low
                full_span = max(high - low, 1e-12)

                seed_num = pd.to_numeric(pd.Series(seed_vals), errors="coerce").dropna().astype(float).tolist()
                if seed_num:
                    center = float(np.mean(seed_num))
                    seed_min = float(min(seed_num))
                    seed_max = float(max(seed_num))
                    seed_span = max(seed_max - seed_min, 0.0)
                    target_span = max(seed_span + full_span * shrink * 0.5, full_span * 0.05)
                    target_span = min(target_span, full_span)
                    new_low = max(low, center - 0.5 * target_span)
                    new_high = min(high, center + 0.5 * target_span)
                    new_low = min(new_low, seed_min)
                    new_high = max(new_high, seed_max)
                    new_low = max(low, new_low)
                    new_high = min(high, new_high)
                else:
                    center = 0.5 * (low + high)
                    target_span = max(full_span * shrink, full_span * 0.05)
                    new_low = max(low, center - 0.5 * target_span)
                    new_high = min(high, center + 0.5 * target_span)

                if _is_int_field(name):
                    new_low = int(np.floor(new_low))
                    new_high = int(np.ceil(new_high))
                    if new_high <= new_low:
                        new_high = int(new_low + 1)
                spec_out["low"] = _coerce_like_field(name, new_low)
                spec_out["high"] = _coerce_like_field(name, new_high)
                refined[name] = spec_out
                continue

            refined[name] = spec_out
            continue

        if isinstance(spec, (list, tuple)):
            raw_values = [_coerce_like_field(name, v) for v in list(spec or [])]
            raw_values = _unique_preserve(raw_values)
            if len(raw_values) <= 1:
                refined[name] = list(raw_values)
                continue

            numeric_vals = pd.to_numeric(pd.Series(raw_values), errors="coerce")
            all_numeric = bool(numeric_vals.notna().all())
            if not seed_vals:
                keep_n = max(1, int(np.ceil(len(raw_values) * shrink)))
                refined[name] = list(raw_values[:keep_n])
                continue

            if all_numeric:
                ordered_pairs = sorted(
                    [(float(v), raw) for v, raw in zip(numeric_vals.tolist(), raw_values)],
                    key=lambda x: x[0],
                )
                ordered_numeric = np.asarray([v for v, _ in ordered_pairs], dtype="float64")
                ordered_raw = [raw for _, raw in ordered_pairs]
                seed_num = pd.to_numeric(pd.Series(seed_vals), errors="coerce").dropna().astype(float).tolist()
                if seed_num:
                    keep_n = max(2, int(np.ceil(len(raw_values) * shrink)))
                    distances = np.min(
                        np.abs(ordered_numeric[:, None] - np.asarray(seed_num, dtype="float64")[None, :]),
                        axis=1,
                    )
                    keep_idx = np.argsort(distances)[:keep_n]
                    keep = [ordered_raw[int(i)] for i in keep_idx]
                    refined[name] = _unique_preserve(keep + list(seed_vals))
                else:
                    keep_n = max(1, int(np.ceil(len(raw_values) * shrink)))
                    refined[name] = list(raw_values[:keep_n])
            else:
                keep = [v for v in raw_values if any(_json_key(v) == _json_key(sv) for sv in seed_vals)]
                if not keep:
                    keep_n = max(1, int(np.ceil(len(raw_values) * shrink)))
                    keep = list(raw_values[:keep_n])
                refined[name] = _unique_preserve(keep)
            continue

        refined[name] = spec

    return refined

def sanitize_param_space_for_engine(
    param_space: Dict[str, Any],
    cfg: MicroPipelineConfig,
) -> Dict[str, Any]:
    if not isinstance(param_space, dict) or not param_space:
        return {
            "sanitized_param_space": {},
            "activity_audit": build_engine_param_activity_audit(cfg),
            "dropped_params": [],
            "kept_params": [],
            "notes": [],
        }
    audit = build_engine_param_activity_audit(cfg, param_space=param_space)
    active_lookup = {str(r["param"]): bool(r["active"]) for _, r in audit.iterrows()}
    reason_lookup = {str(r["param"]): str(r["reason"]) for _, r in audit.iterrows()}
    out: Dict[str, Any] = {}
    dropped: List[Dict[str, str]] = []
    kept: List[str] = []
    notes: List[str] = []
    for key, spec in dict(param_space).items():
        if key not in active_lookup:
            dropped.append({"param": str(key), "reason": "not a valid MicroPipelineConfig field"})
            continue
        if not bool(active_lookup.get(key, False)):
            dropped.append({"param": str(key), "reason": reason_lookup.get(key, "inactive in current engine context")})
            continue
        if not _param_spec_has_variation(spec):
            dropped.append({"param": str(key), "reason": "no effective variation in param spec"})
            continue
        out[str(key)] = spec
        kept.append(str(key))
    if not out:
        notes.append("No active tuning dimensions remained after engine-aware sanitisation.")
    return {
        "sanitized_param_space": out,
        "activity_audit": audit,
        "dropped_params": dropped,
        "kept_params": kept,
        "notes": notes,
    }


def _summary_dict_to_frame(summary: Dict[str, Any], *, section: str) -> pd.DataFrame:
    rows = [{"section": section, "metric": str(k), "value": _json_safe(v)} for k, v in summary.items()]
    return pd.DataFrame(rows)


def _safe_cost_rate(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if not np.isfinite(v):
        return 0.0
    return max(v, 0.0)


def _initialize_cost_tax_state(assets: Iterable[str], *, initial_nav: float = 1.0) -> Dict[str, Any]:
    idx = pd.Index([str(a) for a in assets], dtype="object")
    zero = pd.Series(0.0, index=idx, dtype="float64")
    return {
        "holdings_value": zero.copy(),
        "cost_basis_value": zero.copy(),
        "holding_months": zero.copy(),
        "cash_value": float(max(initial_nav, 0.0)),
        "nav": float(max(initial_nav, 0.0)),
    }


def _evaluate_period_cost_tax(
    *,
    target_weights: pd.Series,
    realised_returns: pd.Series,
    prev_state: Optional[Dict[str, Any]],
    cfg: MicroPipelineConfig,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    assets = pd.Index(pd.Series(target_weights.index, dtype="string").astype(str).tolist(), dtype="object")
    w = pd.to_numeric(target_weights.reindex(assets), errors="coerce").fillna(0.0).astype(float)
    realised = pd.to_numeric(realised_returns.reindex(assets), errors="coerce").fillna(0.0).astype(float)

    state = prev_state if isinstance(prev_state, dict) else _initialize_cost_tax_state(assets, initial_nav=1.0)
    holdings_prev = pd.to_numeric(state.get("holdings_value", pd.Series(0.0, index=assets)), errors="coerce").reindex(assets).fillna(0.0).astype(float)
    basis_prev = pd.to_numeric(state.get("cost_basis_value", pd.Series(0.0, index=assets)), errors="coerce").reindex(assets).fillna(0.0).astype(float)
    age_prev = pd.to_numeric(state.get("holding_months", pd.Series(0.0, index=assets)), errors="coerce").reindex(assets).fillna(0.0).astype(float)
    cash_prev = float(_safe_cost_rate(state.get("cash_value", 0.0))) if np.isfinite(float(state.get("cash_value", 0.0))) else 0.0

    nav_pre = float(holdings_prev.sum() + cash_prev)
    if not np.isfinite(nav_pre) or nav_pre <= 1e-12:
        nav_pre = 1.0
        holdings_prev = pd.Series(0.0, index=assets, dtype="float64")
        basis_prev = pd.Series(0.0, index=assets, dtype="float64")
        age_prev = pd.Series(0.0, index=assets, dtype="float64")
        cash_prev = 1.0

    target_notional_pre = w * nav_pre
    sell_notional_pre = (holdings_prev - target_notional_pre).clip(lower=0.0)
    buy_notional_pre = (target_notional_pre - holdings_prev).clip(lower=0.0)

    min_trade_weight = max(_safe_cost_rate(getattr(cfg, "transaction_cost_min_trade_weight", 0.0)), 0.0)
    min_trade_value = float(min_trade_weight * nav_pre)
    if min_trade_value > 0.0:
        sell_notional_pre = sell_notional_pre.where(sell_notional_pre >= min_trade_value, 0.0)
        buy_notional_pre = buy_notional_pre.where(buy_notional_pre >= min_trade_value, 0.0)

    turnover_sell = float(sell_notional_pre.sum() / nav_pre)
    turnover_buy = float(buy_notional_pre.sum() / nav_pre)
    turnover_total = float((sell_notional_pre.sum() + buy_notional_pre.sum()) / nav_pre)

    commission_rate = _safe_cost_rate(getattr(cfg, "transaction_cost_commission_bps", 0.0)) / 10000.0
    slippage_rate = _safe_cost_rate(getattr(cfg, "transaction_cost_slippage_bps", 0.0)) / 10000.0
    spread_rate = _safe_cost_rate(getattr(cfg, "transaction_cost_spread_bps", 0.0)) / 10000.0
    impact_bps = _safe_cost_rate(getattr(cfg, "transaction_cost_market_impact_bps", 0.0))
    impact_power = max(float(getattr(cfg, "transaction_cost_market_impact_power", 1.0) or 1.0), 0.1)
    holding_fee_annual = _safe_cost_rate(getattr(cfg, "holding_cost_annual_bps", 0.0)) / 10000.0

    trade_rate_linear = (commission_rate + slippage_rate + spread_rate) * turnover_total
    impact_rate = (impact_bps / 10000.0) * (max(turnover_total, 0.0) ** impact_power) if impact_bps > 0.0 else 0.0
    holding_cost_rate = holding_fee_annual / 12.0
    transaction_cost_rate = float(trade_rate_linear + impact_rate) if bool(getattr(cfg, "cost_model_enabled", False)) else 0.0
    ongoing_cost_rate = float(holding_cost_rate * max(float(w.abs().sum()), 0.0)) if bool(getattr(cfg, "cost_model_enabled", False)) else 0.0

    # Average-cost tax approximation on sold notional.
    sell_fraction = sell_notional_pre / holdings_prev.replace(0.0, np.nan)
    sell_fraction = sell_fraction.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=1.0)
    sold_basis = basis_prev * sell_fraction
    realised_gain_value = sell_notional_pre - sold_basis

    short_rate = max(float(getattr(cfg, "tax_short_term_rate", 0.0) or 0.0), 0.0)
    long_rate_raw = getattr(cfg, "tax_long_term_rate", None)
    long_rate = short_rate if long_rate_raw is None or not np.isfinite(float(long_rate_raw)) else max(float(long_rate_raw), 0.0)
    long_threshold = max(int(getattr(cfg, "tax_long_term_threshold_months", 12) or 12), 1)
    loss_credit_raw = getattr(cfg, "tax_loss_credit_rate", None)
    loss_credit_rate = short_rate if loss_credit_raw is None or not np.isfinite(float(loss_credit_raw)) else max(float(loss_credit_raw), 0.0)

    tax_rate_by_asset = pd.Series(np.where(age_prev >= float(long_threshold), long_rate, short_rate), index=assets, dtype="float64")
    tax_due_value = float((realised_gain_value.clip(lower=0.0) * tax_rate_by_asset).sum()) if bool(getattr(cfg, "tax_model_enabled", False)) else 0.0
    tax_credit_value = float((-realised_gain_value.clip(upper=0.0)) * loss_credit_rate).sum() if bool(getattr(cfg, "tax_model_enabled", False) and getattr(cfg, "tax_apply_loss_credit", False)) else 0.0
    tax_net_value = max(tax_due_value - tax_credit_value, 0.0)
    tax_rate_total = float(tax_net_value / nav_pre) if bool(getattr(cfg, "tax_model_enabled", False)) else 0.0

    total_cost_rate = max(transaction_cost_rate + ongoing_cost_rate + tax_rate_total, 0.0)
    total_cost_rate = min(total_cost_rate, 0.999999)
    nav_after_cost = float(nav_pre * (1.0 - total_cost_rate))

    holdings_remaining_pre = (holdings_prev - sell_notional_pre).clip(lower=0.0)
    basis_remaining = (basis_prev - sold_basis).clip(lower=0.0)
    target_holdings_post_cost = w * nav_after_cost
    additional_buy_post_cost = (target_holdings_post_cost - holdings_remaining_pre).clip(lower=0.0)
    cost_basis_post_trade = (basis_remaining + additional_buy_post_cost).clip(lower=0.0)

    remaining_age_num = age_prev * holdings_remaining_pre
    new_age_num = pd.Series(0.0, index=assets, dtype="float64")
    age_den = (holdings_remaining_pre + additional_buy_post_cost).replace(0.0, np.nan)
    holding_age_post_trade = ((remaining_age_num + new_age_num) / age_den).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    cash_post_trade = float(max(nav_after_cost - target_holdings_post_cost.sum(), 0.0))
    gross_portfolio_return = float((w * realised).sum())
    net_portfolio_return = float((1.0 - total_cost_rate) * (1.0 + gross_portfolio_return) - 1.0)

    holdings_end = (target_holdings_post_cost * (1.0 + realised)).clip(lower=0.0)
    holding_age_end = holding_age_post_trade.where(holdings_end > 1e-12, 0.0)
    holding_age_end = holding_age_end + (holdings_end > 1e-12).astype(float)
    basis_end = cost_basis_post_trade.where(holdings_end > 1e-12, 0.0)
    nav_end = float(cash_post_trade + holdings_end.sum())

    next_state = {
        "holdings_value": holdings_end.astype(float),
        "cost_basis_value": basis_end.astype(float),
        "holding_months": holding_age_end.astype(float),
        "cash_value": cash_post_trade,
        "nav": nav_end,
    }

    meta = {
        "cost_model_enabled": bool(getattr(cfg, "cost_model_enabled", False)),
        "tax_model_enabled": bool(getattr(cfg, "tax_model_enabled", False)),
        "gross_portfolio_return_simple": gross_portfolio_return,
        "net_portfolio_return_simple": net_portfolio_return,
        "transaction_cost_rate": float(transaction_cost_rate),
        "holding_cost_rate": float(ongoing_cost_rate),
        "tax_cost_rate": float(tax_rate_total),
        "total_cost_rate": float(total_cost_rate),
        "transaction_cost_value": float(nav_pre * transaction_cost_rate),
        "holding_cost_value": float(nav_pre * ongoing_cost_rate),
        "tax_cost_value": float(nav_pre * tax_rate_total),
        "total_cost_value": float(nav_pre * total_cost_rate),
        "turnover_buy": turnover_buy,
        "turnover_sell": turnover_sell,
        "turnover_total_cost_basis": turnover_total,
        "realised_gain_value": float(realised_gain_value.sum()),
        "realised_gain_positive_value": float(realised_gain_value.clip(lower=0.0).sum()),
        "realised_gain_negative_value": float(realised_gain_value.clip(upper=0.0).sum()),
        "tax_credit_value": float(tax_credit_value),
        "nav_pre_cost": float(nav_pre),
        "nav_post_cost_pre_return": float(nav_after_cost),
        "nav_end": float(nav_end),
        "cost_drag_bps": float(total_cost_rate * 10000.0),
        "transaction_cost_drag_bps": float(transaction_cost_rate * 10000.0),
        "tax_drag_bps": float(tax_rate_total * 10000.0),
        "holding_cost_drag_bps": float(ongoing_cost_rate * 10000.0),
    }
    return meta, next_state


def build_cost_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    diag = result.get("diagnostics_df", pd.DataFrame())
    cfg = result.get("config", MicroPipelineConfig())
    if not isinstance(diag, pd.DataFrame) or diag.empty:
        return {
            "cost_model_enabled": bool(getattr(cfg, "cost_model_enabled", False)) if isinstance(cfg, MicroPipelineConfig) else False,
            "tax_model_enabled": bool(getattr(cfg, "tax_model_enabled", False)) if isinstance(cfg, MicroPipelineConfig) else False,
            "gross_cagr": np.nan,
            "net_cagr": np.nan,
            "cost_impact_on_cagr": np.nan,
            "mean_total_cost_rate": np.nan,
            "mean_transaction_cost_rate": np.nan,
            "mean_tax_cost_rate": np.nan,
            "mean_holding_cost_rate": np.nan,
            "cumulative_total_cost_rate": np.nan,
            "mean_turnover_buy": np.nan,
            "mean_turnover_sell": np.nan,
            "mean_realised_gain_positive": np.nan,
        }

    def s(col: str) -> pd.Series:
        return pd.to_numeric(diag[col], errors="coerce") if col in diag.columns else pd.Series(dtype="float64")

    net_series = pd.Series(result.get("oos_returns_simple", pd.Series(dtype="float64")), dtype="float64").dropna()
    gross_series = s("gross_portfolio_return_simple").dropna()

    def _cagr(xs: pd.Series) -> float:
        if xs.empty:
            return np.nan
        eq = _equity_curve_from_simple(xs.to_numpy(dtype="float64"))
        years = xs.shape[0] / 12.0
        if eq.size == 0 or years <= 0 or eq[-1] <= 0:
            return np.nan
        return float(eq[-1] ** (1.0 / years) - 1.0)

    gross_cagr = _cagr(gross_series)
    net_cagr = _cagr(net_series)
    total_cost = s("total_cost_rate")
    txn_cost = s("transaction_cost_rate")
    tax_cost = s("tax_cost_rate")
    hold_cost = s("holding_cost_rate")

    return {
        "cost_model_enabled": bool(getattr(cfg, "cost_model_enabled", False)) if isinstance(cfg, MicroPipelineConfig) else False,
        "tax_model_enabled": bool(getattr(cfg, "tax_model_enabled", False)) if isinstance(cfg, MicroPipelineConfig) else False,
        "gross_cagr": gross_cagr,
        "net_cagr": net_cagr,
        "cost_impact_on_cagr": float(net_cagr - gross_cagr) if np.isfinite(net_cagr) and np.isfinite(gross_cagr) else np.nan,
        "mean_total_cost_rate": float(total_cost.mean()) if not total_cost.empty else np.nan,
        "mean_transaction_cost_rate": float(txn_cost.mean()) if not txn_cost.empty else np.nan,
        "mean_tax_cost_rate": float(tax_cost.mean()) if not tax_cost.empty else np.nan,
        "mean_holding_cost_rate": float(hold_cost.mean()) if not hold_cost.empty else np.nan,
        "cumulative_total_cost_rate": float(total_cost.sum()) if not total_cost.empty else np.nan,
        "mean_cost_drag_bps": float(s("cost_drag_bps").mean()) if "cost_drag_bps" in diag.columns else np.nan,
        "mean_transaction_drag_bps": float(s("transaction_cost_drag_bps").mean()) if "transaction_cost_drag_bps" in diag.columns else np.nan,
        "mean_tax_drag_bps": float(s("tax_drag_bps").mean()) if "tax_drag_bps" in diag.columns else np.nan,
        "mean_holding_drag_bps": float(s("holding_cost_drag_bps").mean()) if "holding_cost_drag_bps" in diag.columns else np.nan,
        "mean_turnover_buy": float(s("turnover_buy").mean()) if "turnover_buy" in diag.columns else np.nan,
        "mean_turnover_sell": float(s("turnover_sell").mean()) if "turnover_sell" in diag.columns else np.nan,
        "mean_realised_gain_positive": float(s("realised_gain_positive_value").mean()) if "realised_gain_positive_value" in diag.columns else np.nan,
    }


def build_performance_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    oos = pd.Series(result.get("oos_returns_simple", pd.Series(dtype="float64")), dtype="float64").dropna()
    diag = result.get("diagnostics_df", pd.DataFrame())
    if oos.empty:
        return {
            "n_months": 0,
            "cagr": np.nan,
            "annual_return_arith": np.nan,
            "annual_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "best_month": np.nan,
            "worst_month": np.nan,
            "positive_month_rate": np.nan,
            "mean_turnover": np.nan,
            "mean_active_return": np.nan,
            "active_return_annual": np.nan,
            "tracking_error_annual": np.nan,
            "information_ratio": np.nan,
            "gross_cagr": np.nan,
            "gross_annual_return_arith": np.nan,
            "gross_sharpe": np.nan,
            "mean_total_cost_rate": np.nan,
            "mean_transaction_cost_rate": np.nan,
            "mean_tax_cost_rate": np.nan,
        }

    n = int(oos.shape[0])
    eq = _equity_curve_from_simple(oos.to_numpy(dtype="float64"))
    total_return = float(eq[-1] - 1.0) if eq.size else np.nan
    years = n / 12.0
    cagr = float(eq[-1] ** (1.0 / years) - 1.0) if eq.size and years > 0 and eq[-1] > 0 else np.nan
    ann_ret = float(oos.mean() * 12.0)
    ann_vol = float(oos.std(ddof=1) * np.sqrt(12.0)) if n > 1 else 0.0
    sharpe = float(ann_ret / ann_vol) if ann_vol > 1e-12 else np.nan
    max_dd = _max_drawdown_from_simple(oos.to_numpy(dtype="float64"))

    gross_series = pd.to_numeric(diag.get("gross_portfolio_return_simple", pd.Series(dtype="float64")), errors="coerce").dropna() if isinstance(diag, pd.DataFrame) and not diag.empty else pd.Series(dtype="float64")
    gross_eq = _equity_curve_from_simple(gross_series.to_numpy(dtype="float64")) if not gross_series.empty else np.array([])
    gross_cagr = float(gross_eq[-1] ** (1.0 / (gross_series.shape[0] / 12.0)) - 1.0) if gross_eq.size and gross_series.shape[0] > 0 and gross_eq[-1] > 0 else np.nan
    gross_ann_ret = float(gross_series.mean() * 12.0) if not gross_series.empty else np.nan
    gross_ann_vol = float(gross_series.std(ddof=1) * np.sqrt(12.0)) if gross_series.shape[0] > 1 else np.nan
    gross_sharpe = float(gross_ann_ret / gross_ann_vol) if np.isfinite(gross_ann_vol) and gross_ann_vol > 1e-12 else np.nan

    active = pd.Series(dtype="float64")
    if isinstance(diag, pd.DataFrame) and not diag.empty and "active_return_simple" in diag.columns:
        active = pd.to_numeric(diag["active_return_simple"], errors="coerce").dropna()
    active_return_annual = float(active.mean() * 12.0) if not active.empty else np.nan
    tracking_error_annual = float(active.std(ddof=1) * np.sqrt(12.0)) if active.shape[0] > 1 else np.nan
    information_ratio = float(active_return_annual / tracking_error_annual) if np.isfinite(tracking_error_annual) and tracking_error_annual > 1e-12 else np.nan

    return {
        "n_months": n,
        "n_periods": n,
        "total_return": total_return,
        "cagr": cagr,
        "annual_return_arith": ann_ret,
        "annual_volatility": ann_vol,
        "annualized_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "best_month": float(oos.max()),
        "worst_month": float(oos.min()),
        "positive_month_rate": float((oos > 0).mean()),
        "mean_turnover": float(pd.to_numeric(diag.get("turnover", pd.Series(dtype="float64")), errors="coerce").mean()) if isinstance(diag, pd.DataFrame) and not diag.empty else np.nan,
        "mean_active_return": float(active.mean()) if not active.empty else np.nan,
        "active_return_annual": active_return_annual,
        "tracking_error_annual": tracking_error_annual,
        "information_ratio": information_ratio,
        "gross_cagr": gross_cagr,
        "gross_annual_return_arith": gross_ann_ret,
        "gross_sharpe": gross_sharpe,
        "mean_total_cost_rate": float(pd.to_numeric(diag.get("total_cost_rate", pd.Series(dtype="float64")), errors="coerce").mean()) if isinstance(diag, pd.DataFrame) and not diag.empty else np.nan,
        "mean_transaction_cost_rate": float(pd.to_numeric(diag.get("transaction_cost_rate", pd.Series(dtype="float64")), errors="coerce").mean()) if isinstance(diag, pd.DataFrame) and not diag.empty else np.nan,
        "mean_tax_cost_rate": float(pd.to_numeric(diag.get("tax_cost_rate", pd.Series(dtype="float64")), errors="coerce").mean()) if isinstance(diag, pd.DataFrame) and not diag.empty else np.nan,
    }


def build_risk_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    diag = result.get("diagnostics_df", pd.DataFrame())
    if not isinstance(diag, pd.DataFrame) or diag.empty:
        return {
            "mean_est_portfolio_vol_monthly": np.nan,
            "mean_est_portfolio_vol_annual": np.nan,
            "mean_pre_target_vol_monthly": np.nan,
            "mean_vol_target_mult": np.nan,
            "vol_target_hit_rate": np.nan,
            "mean_effective_risk_bets": np.nan,
            "mean_top_risk_contributor_share": np.nan,
            "mean_risk_concentration_herfindahl": np.nan,
            "mean_cluster_high_corr_share": np.nan,
            "mean_portfolio_vol_from_cov": np.nan,
        }

    def s(col: str) -> pd.Series:
        return pd.to_numeric(diag[col], errors="coerce") if col in diag.columns else pd.Series(dtype="float64")

    post_vol = s("est_portfolio_vol_monthly")
    pre_vol = s("est_portfolio_vol_monthly_pre_target")
    return {
        "mean_est_portfolio_vol_monthly": float(post_vol.mean()) if not post_vol.empty else np.nan,
        "mean_est_portfolio_vol_annual": float(post_vol.mean() * np.sqrt(12.0)) if not post_vol.empty else np.nan,
        "mean_pre_target_vol_monthly": float(pre_vol.mean()) if not pre_vol.empty else np.nan,
        "mean_vol_target_mult": float(s("vol_target_mult").mean()) if "vol_target_mult" in diag.columns else np.nan,
        "vol_target_hit_rate": float(pd.to_numeric(diag.get("vol_target_active", pd.Series(dtype="float64")), errors="coerce").mean()) if "vol_target_active" in diag.columns else np.nan,
        "mean_effective_risk_bets": float(s("effective_risk_bets").mean()) if "effective_risk_bets" in diag.columns else np.nan,
        "mean_top_risk_contributor_share": float(s("top_risk_contributor_share").mean()) if "top_risk_contributor_share" in diag.columns else np.nan,
        "mean_risk_concentration_herfindahl": float(s("risk_concentration_herfindahl").mean()) if "risk_concentration_herfindahl" in diag.columns else np.nan,
        "mean_cluster_high_corr_share": float(s("cluster_high_corr_share").mean()) if "cluster_high_corr_share" in diag.columns else np.nan,
        "mean_portfolio_vol_from_cov": float(s("portfolio_vol_from_cov").mean()) if "portfolio_vol_from_cov" in diag.columns else np.nan,
    }


def build_diversification_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    diag = result.get("diagnostics_df", pd.DataFrame())
    if not isinstance(diag, pd.DataFrame) or diag.empty:
        return {
            "mean_effective_n_assets": np.nan,
            "mean_effective_breadth": np.nan,
            "mean_effective_breadth_universe": np.nan,
            "mean_breadth_utilization_ratio": np.nan,
            "mean_diversification_ratio": np.nan,
            "mean_weight_herfindahl": np.nan,
            "mean_avg_pairwise_corr": np.nan,
            "mean_weighted_avg_corr": np.nan,
        }

    def s(col: str) -> pd.Series:
        return pd.to_numeric(diag[col], errors="coerce") if col in diag.columns else pd.Series(dtype="float64")

    return {
        "mean_effective_n_assets": float(s("effective_n_assets").mean()) if "effective_n_assets" in diag.columns else np.nan,
        "mean_effective_breadth": float(s("effective_breadth").mean()) if "effective_breadth" in diag.columns else np.nan,
        "mean_effective_breadth_universe": float(s("effective_breadth_universe").mean()) if "effective_breadth_universe" in diag.columns else np.nan,
        "mean_breadth_utilization_ratio": float(s("breadth_utilization_ratio").mean()) if "breadth_utilization_ratio" in diag.columns else np.nan,
        "mean_diversification_ratio": float(s("diversification_ratio").mean()) if "diversification_ratio" in diag.columns else np.nan,
        "mean_weight_herfindahl": float(s("weight_herfindahl").mean()) if "weight_herfindahl" in diag.columns else np.nan,
        "mean_avg_pairwise_corr": float(s("avg_pairwise_corr").mean()) if "avg_pairwise_corr" in diag.columns else np.nan,
        "mean_weighted_avg_corr": float(s("weighted_avg_corr").mean()) if "weighted_avg_corr" in diag.columns else np.nan,
    }


def build_universe_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    weights_df = result.get("weights_df", pd.DataFrame())
    diag = result.get("diagnostics_df", pd.DataFrame())
    assets: List[str] = []
    if isinstance(weights_df, pd.DataFrame) and not weights_df.empty and "asset" in weights_df.columns:
        assets = sorted(pd.Series(weights_df["asset"], dtype="string").dropna().astype(str).unique().tolist())

    active_assets_per_date = pd.Series(dtype="float64")
    if isinstance(weights_df, pd.DataFrame) and not weights_df.empty and {"date", "weight"}.issubset(weights_df.columns):
        tmp = weights_df.copy()
        tmp["weight"] = pd.to_numeric(tmp["weight"], errors="coerce").fillna(0.0)
        active_assets_per_date = tmp.groupby("date")["weight"].apply(lambda x: float((x.abs() > 1e-10).sum()))

    n_assets_series = pd.to_numeric(diag.get("n_assets", pd.Series(dtype="float64")), errors="coerce") if isinstance(diag, pd.DataFrame) and not diag.empty else pd.Series(dtype="float64")

    return {
        "n_unique_assets": int(len(assets)),
        "assets": assets,
        "mean_available_assets": float(n_assets_series.mean()) if not n_assets_series.empty else np.nan,
        "min_available_assets": float(n_assets_series.min()) if not n_assets_series.empty else np.nan,
        "max_available_assets": float(n_assets_series.max()) if not n_assets_series.empty else np.nan,
        "mean_active_assets": float(active_assets_per_date.mean()) if not active_assets_per_date.empty else np.nan,
        "min_active_assets": float(active_assets_per_date.min()) if not active_assets_per_date.empty else np.nan,
        "max_active_assets": float(active_assets_per_date.max()) if not active_assets_per_date.empty else np.nan,
    }


def build_config_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    cfg = result.get("config", MicroPipelineConfig())
    cfg_dict = config_to_dict(cfg) if isinstance(cfg, MicroPipelineConfig) else _json_safe(cfg)
    fp = config_fingerprint(cfg) if isinstance(cfg, MicroPipelineConfig) else np.nan
    out = {"config_fingerprint": fp}
    out.update(cfg_dict)
    return out


def build_global_params_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    return build_simplified_global_params_summary(result)


def build_factor_model_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    diag = result.get("diagnostics_df", pd.DataFrame())
    cfg = result.get("config", MicroPipelineConfig())
    if not isinstance(diag, pd.DataFrame) or diag.empty:
        return {
            "factor_model_active": bool(getattr(cfg, "factor_model_active", False)) if isinstance(cfg, MicroPipelineConfig) else False,
            "factor_covariance_active": bool(getattr(cfg, "factor_covariance_active", False)) if isinstance(cfg, MicroPipelineConfig) else False,
            "factor_overlay_active_share": np.nan,
            "factor_model_n_factors_mean": np.nan,
            "factor_model_explained_variance_share_mean": np.nan,
            "factor_mu_abs_tilt_mean": np.nan,
            "factor_covariance_blend_used_mean": np.nan,
        }

    def s(col: str) -> pd.Series:
        return pd.to_numeric(diag[col], errors="coerce") if col in diag.columns else pd.Series(dtype="float64")

    factor_active = s("factor_overlay_active") if "factor_overlay_active" in diag.columns else pd.Series(dtype="float64")
    return {
        "factor_model_active": bool(getattr(cfg, "factor_model_active", False)) if isinstance(cfg, MicroPipelineConfig) else False,
        "factor_covariance_active": bool(getattr(cfg, "factor_covariance_active", False)) if isinstance(cfg, MicroPipelineConfig) else False,
        "factor_overlay_active_share": float(factor_active.mean()) if not factor_active.empty else np.nan,
        "factor_model_n_factors_mean": float(s("factor_model_n_factors").mean()) if "factor_model_n_factors" in diag.columns else np.nan,
        "factor_model_n_obs_mean": float(s("factor_model_n_obs").mean()) if "factor_model_n_obs" in diag.columns else np.nan,
        "factor_model_explained_variance_share_mean": float(s("factor_model_explained_variance_share").mean()) if "factor_model_explained_variance_share" in diag.columns else np.nan,
        "factor_model_residual_variance_share_mean": float(s("factor_model_residual_variance_share").mean()) if "factor_model_residual_variance_share" in diag.columns else np.nan,
        "factor_mu_abs_tilt_mean": float(s("factor_mu_abs_tilt_mean").mean()) if "factor_mu_abs_tilt_mean" in diag.columns else np.nan,
        "factor_mu_abs_tilt_max": float(s("factor_mu_abs_tilt_max").max()) if "factor_mu_abs_tilt_max" in diag.columns else np.nan,
        "factor_covariance_blend_used_mean": float(s("factor_covariance_blend_used").mean()) if "factor_covariance_blend_used" in diag.columns else np.nan,
        "factor_residual_blend_used_mean": float(s("factor_residual_blend_used").mean()) if "factor_residual_blend_used" in diag.columns else np.nan,
        "factor_covariance_trace_ratio_mean": float(s("factor_covariance_trace_ratio").mean()) if "factor_covariance_trace_ratio" in diag.columns else np.nan,
    }


def build_run_report(result: Dict[str, Any], *, run_name: str = "run") -> Dict[str, Any]:
    performance = build_performance_summary(result)
    risk = build_risk_summary(result)
    diversification = build_diversification_summary(result)
    universe = build_universe_summary(result)
    costs = build_cost_summary(result)
    factor_model = build_factor_model_summary(result)
    global_params = build_global_params_summary(result)
    config = build_config_summary(result)

    sections = {
        "performance": performance,
        "risk": risk,
        "diversification": diversification,
        "universe": universe,
        "costs": costs,
        "factor_model": factor_model,
        "global_params": global_params,
        "config": config,
    }
    summary_frames = [_summary_dict_to_frame(v, section=k) for k, v in sections.items()]
    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame(columns=["section", "metric", "value"])
    flat_metrics = {}
    for sec, payload in sections.items():
        for k, v in payload.items():
            if isinstance(v, (list, dict)):
                continue
            flat_metrics[f"{sec}.{k}"] = v

    return {
        "run_name": run_name,
        "sections": sections,
        "summary_df": summary_df,
        "flat_metrics": flat_metrics,
    }


def compare_metric_frames(left: pd.DataFrame, right: pd.DataFrame, *, left_name: str = "primary", right_name: str = "comparison") -> pd.DataFrame:
    req = {"section", "metric", "value"}
    if not req.issubset(left.columns) or not req.issubset(right.columns):
        raise ValueError("Both metric frames must contain section, metric, value")
    l = left.rename(columns={"value": left_name})[["section", "metric", left_name]]
    r = right.rename(columns={"value": right_name})[["section", "metric", right_name]]
    out = l.merge(r, on=["section", "metric"], how="outer")
    out["delta"] = pd.to_numeric(out[right_name], errors="coerce") - pd.to_numeric(out[left_name], errors="coerce")
    denom = pd.to_numeric(out[left_name], errors="coerce").replace(0.0, np.nan).abs()
    out["delta_pct"] = out["delta"] / denom
    return out


def compare_run_reports(left_result: Dict[str, Any], right_result: Dict[str, Any], *, left_name: str = "primary", right_name: str = "comparison") -> Dict[str, Any]:
    left_report = build_run_report(left_result, run_name=left_name)
    right_report = build_run_report(right_result, run_name=right_name)
    metrics_comparison = compare_metric_frames(left_report["summary_df"], right_report["summary_df"], left_name=left_name, right_name=right_name)

    weights_cmp = pd.DataFrame()
    left_weights = left_result.get("weights_df", pd.DataFrame())
    right_weights = right_result.get("weights_df", pd.DataFrame())
    if isinstance(left_weights, pd.DataFrame) and isinstance(right_weights, pd.DataFrame) and not left_weights.empty and not right_weights.empty:
        l = left_weights.groupby("asset")["weight"].mean().rename(left_name) if {"asset", "weight"}.issubset(left_weights.columns) else pd.Series(dtype="float64")
        r = right_weights.groupby("asset")["weight"].mean().rename(right_name) if {"asset", "weight"}.issubset(right_weights.columns) else pd.Series(dtype="float64")
        if not l.empty or not r.empty:
            weights_cmp = pd.concat([l, r], axis=1).reset_index()
            weights_cmp["delta"] = pd.to_numeric(weights_cmp[right_name], errors="coerce") - pd.to_numeric(weights_cmp[left_name], errors="coerce")

    diagnostics_cmp = pd.DataFrame()
    left_diag = left_result.get("diagnostics_df", pd.DataFrame())
    right_diag = right_result.get("diagnostics_df", pd.DataFrame())
    if isinstance(left_diag, pd.DataFrame) and isinstance(right_diag, pd.DataFrame) and not left_diag.empty and not right_diag.empty:
        common_numeric = sorted(set(left_diag.select_dtypes(include=[np.number]).columns).intersection(right_diag.select_dtypes(include=[np.number]).columns))
        rows = []
        for col in common_numeric:
            lv = float(pd.to_numeric(left_diag[col], errors="coerce").mean())
            rv = float(pd.to_numeric(right_diag[col], errors="coerce").mean())
            rows.append({"metric": col, left_name: lv, right_name: rv, "delta": rv - lv})
        diagnostics_cmp = pd.DataFrame(rows)

    return {
        "left_report": left_report,
        "right_report": right_report,
        "metrics_comparison": metrics_comparison,
        "weights_comparison": weights_cmp,
        "diagnostics_comparison": diagnostics_cmp,
    }


def save_run_outputs(result: Dict[str, Any], folder: str, *, run_name: str = "run") -> Dict[str, str]:
    out_dir = Path(folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_run_report(result, run_name=run_name)
    paths: Dict[str, str] = {}

    cfg = result.get("config")
    if isinstance(cfg, MicroPipelineConfig):
        cfg_path = out_dir / f"{run_name}_config.json"
        save_config_json(cfg, str(cfg_path))
        paths["config_json"] = str(cfg_path)

    summary_path = out_dir / f"{run_name}_summary.csv"
    report["summary_df"].to_csv(summary_path, index=False)
    paths["summary_csv"] = str(summary_path)

    for key, suffix in [("diagnostics_df", "diagnostics.csv"), ("weights_df", "weights.csv")]:
        frame = result.get(key)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            p = out_dir / f"{run_name}_{suffix}"
            frame.to_csv(p, index=False)
            paths[key] = str(p)

    for key, suffix in [("oos_returns_simple", "oos_returns_simple.csv"), ("oos_returns_log", "oos_returns_log.csv")]:
        ser = result.get(key)
        if isinstance(ser, pd.Series) and not ser.empty:
            p = out_dir / f"{run_name}_{suffix}"
            ser.rename(key).to_frame().to_csv(p, index=True)
            paths[key] = str(p)

    manifest = {
        "run_name": run_name,
        "report": _json_safe(report["sections"]),
        "paths": paths,
    }
    manifest_path = out_dir / f"{run_name}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    paths["manifest_json"] = str(manifest_path)
    return paths

# ============================================================
# Small numerical helpers
# ============================================================

def annual_return_to_monthly(annual_return: float) -> float:
    return float((1.0 + float(annual_return)) ** (1.0 / 12.0) - 1.0)


def monthly_return_to_annual(monthly_return: float) -> float:
    return float((1.0 + float(monthly_return)) ** 12.0 - 1.0)


def annual_vol_to_monthly(annual_vol: float) -> float:
    return float(annual_vol) / np.sqrt(12.0)


def monthly_vol_to_annual(monthly_vol: float) -> float:
    return float(monthly_vol) * np.sqrt(12.0)


def _expand_monthly_return_to_daily_path(
    monthly_return: float,
    daily_steps: int,
    rng: np.random.Generator,
    *,
    noise_scale: float = 0.35,
) -> np.ndarray:
    """Expand one monthly simple return into a synthetic daily path.

    The daily path is constructed in log-return space and then re-centred so
    that the compounded daily returns exactly match the original monthly
    simple return. This keeps the Step 6 daily-hybrid projection compatible
    with the monthly OOS engine path without changing the monthly result.
    """
    n_steps = max(int(daily_steps or 1), 1)
    try:
        monthly = float(monthly_return)
    except Exception:
        monthly = 0.0
    if not np.isfinite(monthly):
        monthly = 0.0

    monthly = float(np.clip(monthly, -0.999999, None))
    total_log_return = float(np.log1p(monthly))
    base_log_return = total_log_return / float(n_steps)

    if n_steps == 1 or float(noise_scale or 0.0) <= 0.0:
        if n_steps == 1:
            return np.asarray([float(np.expm1(total_log_return))], dtype="float64")
        return np.full(n_steps, float(np.expm1(base_log_return)), dtype="float64")

    scale = max(abs(base_log_return) * float(noise_scale), 1e-8)
    log_path = base_log_return + rng.normal(0.0, scale, size=n_steps)

    # Re-centre the noisy log path so the compounded daily path preserves the
    # sampled monthly return exactly up to floating point precision.
    log_path = log_path + ((total_log_return - float(np.sum(log_path))) / float(n_steps))
    daily_returns = np.expm1(log_path).astype("float64")
    return np.clip(daily_returns, -0.999999, None)


def _coerce_1d_float_array(values: Iterable[float], name: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype="float64").reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError(f"{name} contains no finite observations")
    return arr


def _equity_curve_from_simple(simple_returns: np.ndarray) -> np.ndarray:
    x = np.asarray(simple_returns, dtype="float64").reshape(-1)
    if x.size == 0:
        return np.asarray([], dtype="float64")
    x = np.clip(x, -0.999999, None)
    return np.cumprod(1.0 + x)


def _max_drawdown_from_simple(simple_returns: np.ndarray) -> float:
    eq = _equity_curve_from_simple(simple_returns)
    if eq.size == 0:
        return np.nan
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak) - 1.0
    return float(dd.min())


def _moving_block_bootstrap_path(
    xs_monthly: np.ndarray,
    horizon_months: int,
    rng: np.random.Generator,
    block_len: int,
) -> np.ndarray:
    n = len(xs_monthly)
    if n == 0:
        raise ValueError("xs_monthly must not be empty")
    if block_len < 1 or block_len > n:
        raise ValueError(f"block_len must be in [1, {n}], got {block_len}")

    starts = np.arange(0, n - block_len + 1)
    out: List[float] = []
    while len(out) < horizon_months:
        st = int(rng.choice(starts))
        out.extend(xs_monthly[st: st + block_len].tolist())
    return np.asarray(out[:horizon_months], dtype="float64")


def _validate_reasonable_assumptions(
    annual_return: float,
    annual_vol: float,
) -> Tuple[float, float]:
    ar = float(annual_return)
    av = float(annual_vol)

    ar = min(ar, MAX_REASONABLE_ANNUAL_RETURN)
    av = max(MIN_ANNUAL_VOL, min(av, MAX_REASONABLE_ANNUAL_VOL))
    return ar, av


# ============================================================
# Human-readable helpers
# ============================================================

def describe_volatility_level(annual_vol: float) -> str:
    vol = float(annual_vol)
    if vol < 0.08:
        return "Low"
    if vol < 0.14:
        return "Moderate"
    if vol < 0.22:
        return "High"
    return "Very high"


def describe_profile_tradeoff(profile: str) -> str:
    profile = str(profile)
    mapping = {
        "Conservative": (
            "This profile prioritises lower uncertainty and shallower downside, "
            "but usually offers lower long-term growth potential."
        ),
        "Balanced": (
            "This profile aims for a middle ground between growth and risk, "
            "with moderate uncertainty and moderate long-term upside."
        ),
        "Growth": (
            "This profile targets stronger long-term growth, but outcomes can vary "
            "more and temporary losses may be materially larger."
        ),
        "Aggressive": (
            "This profile accepts wide outcome dispersion and deeper drawdowns in exchange "
            "for higher upside potential."
        ),
    }
    return mapping.get(
        profile,
        "Higher expected growth usually comes with higher volatility and a wider range of outcomes.",
    )


def describe_capital_risk(summary: GrowthSummary) -> str:
    loss_prob = float(summary.probability_of_loss_vs_contributions)
    avg_dd = abs(float(summary.expected_max_drawdown))

    if loss_prob <= 0.10 and avg_dd <= 0.10:
        return "Downside appears relatively contained, although uncertainty still exists."
    if loss_prob <= 0.25 and avg_dd <= 0.20:
        return "There is a meaningful trade-off between upside and downside, but risk remains moderate."
    if loss_prob <= 0.40 and avg_dd <= 0.30:
        return "Material downside is possible, and outcomes may vary considerably across paths."
    return "Risk is substantial here: the range of outcomes is wide and deeper temporary losses are plausible."


# ============================================================
# MICRO PIPELINE — compact live OOS generator
# ============================================================

def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(x, dtype="float64") / max(float(temperature), 1e-8)
    z = z - np.nanmax(z)
    ez = np.exp(z)
    s = ez.sum()
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(z) / len(z)
    return ez / s


def _prepare_asset_panel(
    asset_panel_df: pd.DataFrame,
    cfg: MicroPipelineConfig,
) -> pd.DataFrame:
    need = {cfg.date_col, cfg.asset_col, cfg.return_col}
    missing = [c for c in need if c not in asset_panel_df.columns]
    if missing:
        raise ValueError(f"asset_panel_df is missing required columns: {missing}")

    df = asset_panel_df.copy()
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
    df[cfg.asset_col] = df[cfg.asset_col].astype(str)
    df[cfg.return_col] = pd.to_numeric(df[cfg.return_col], errors="coerce")
    df = df.dropna(subset=[cfg.date_col, cfg.asset_col, cfg.return_col]).sort_values(
        [cfg.date_col, cfg.asset_col]
    )

    counts = df.groupby(cfg.date_col)[cfg.asset_col].nunique()
    if counts.empty:
        raise ValueError("asset_panel_df contains no valid rows after cleaning")
    if counts.max() < 2:
        raise ValueError("asset_panel_df must contain at least 2 assets per date in some periods")

    return df.reset_index(drop=True)


def _panel_to_return_matrix(
    asset_panel_df: pd.DataFrame,
    cfg: MicroPipelineConfig,
) -> pd.DataFrame:
    pivot = asset_panel_df.pivot_table(
        index=cfg.date_col,
        columns=cfg.asset_col,
        values=cfg.return_col,
        aggfunc="last",
    ).sort_index()

    pivot = pivot.dropna(how="all")
    if pivot.shape[0] < cfg.min_train + 1:
        raise ValueError(
            f"Not enough monthly rows for micro pipeline. Need at least {cfg.min_train + 1}, got {pivot.shape[0]}"
        )

    # closer to coursework behaviour: only keep dates where enough assets are alive
    enough_assets = pivot.notna().sum(axis=1) >= max(2, int(np.ceil(pivot.shape[1] * 0.5)))
    pivot = pivot.loc[enough_assets]
    if pivot.shape[0] < cfg.min_train + 1:
        raise ValueError(
            f"Not enough valid monthly rows after coverage filter. Need at least {cfg.min_train + 1}, got {pivot.shape[0]}"
        )

    return pivot


def _ewma_sigma(series_df: pd.DataFrame, halflife: int) -> pd.Series:
    sq = series_df.pow(2)
    ewma_var = sq.ewm(halflife=max(int(halflife), 1), adjust=False).mean().iloc[-1]
    return np.sqrt(ewma_var)

def _resolve_adaptive_allocation(*, score_dispersion: float, cfg: MicroPipelineConfig) -> Dict[str, Any]:
    return {
        "bucket": "static",
        "temperature": float(cfg.temperature),
        "top_k": cfg.top_k,
        "score_dispersion": float(score_dispersion),
    }


def _resolve_effective_temperature_by_universe_size(requested_temperature: float, universe_scaling: Dict[str, float], cfg: MicroPipelineConfig) -> float:
    return max(float(requested_temperature) * float(universe_scaling.get("temperature_scale", 1.0)), 1e-6)


def _resolve_effective_top_k_by_universe_size(n_assets: int, requested_top_k: Optional[int], universe_scaling: Dict[str, float], cfg: MicroPipelineConfig) -> Optional[int]:
    if requested_top_k is None:
        return None
    return int(max(1, min(int(requested_top_k), int(n_assets))))


def _apply_regime_derisk(
    weights: pd.Series,
    *,
    regime: str,
    cfg: MicroPipelineConfig,
) -> Tuple[pd.Series, float]:

    # Only meaningful for long-only portfolios
    if not bool(cfg.long_only):
        return weights, 1.0

    rg = str(regime or "pooled").lower()

    # ----- resolve multiplier -----
    if rg == "high":
        mult = float(cfg.regime_derisk_high)
    elif rg == "mid":
        mult = float(cfg.regime_derisk_mid)
    elif rg == "low":
        mult = float(cfg.regime_derisk_low)
    else:
        # pooled / none / unknown / disabled → neutral
        mult = 1.0

    mult = float(np.clip(mult, 0.0, 1.25))

    # ----- neutral → do nothing -----
    if abs(mult - 1.0) < 1e-12:
        return weights, 1.0

    # ----- derisk towards equal weight -----
    ew = pd.Series(
        np.ones(len(weights)) / len(weights),
        index=weights.index,
    )

    out = ew + mult * (weights - ew)

    out = out.clip(lower=0.0)
    s = out.sum()

    if not np.isfinite(s) or s <= 0:
        return ew, mult

    out = out / s

    return out, mult


def _compute_turnover_to_prev(weights: pd.Series, prev_weights: Optional[pd.Series]) -> float:
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0).astype(float)
    if prev_weights is None:
        return float(np.abs(w).sum())
    prev = pd.to_numeric(prev_weights.reindex(w.index), errors="coerce").fillna(0.0).astype(float)
    return float((w - prev).abs().sum())


def _apply_turnover_penalty(weights: pd.Series, prev_weights: Optional[pd.Series], cfg: MicroPipelineConfig) -> Tuple[pd.Series, Dict[str, Any]]:
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0).astype(float).copy()
    meta = {
        "turnover_penalty_active": False,
        "turnover_penalty_strength": float(getattr(cfg, "turnover_penalty_strength", 0.0)),
        "turnover_penalty_power": float(getattr(cfg, "turnover_penalty_power", 1.0)),
        "turnover_penalty_target": float(getattr(cfg, "turnover_penalty_target", 0.20)),
        "turnover_penalty_max_turnover": float(getattr(cfg, "turnover_penalty_max_turnover")) if getattr(cfg, "turnover_penalty_max_turnover", None) is not None else np.nan,
        "turnover_pre_penalty": float(np.abs(w).sum()) if prev_weights is None else np.nan,
        "turnover_post_penalty": float(np.abs(w).sum()) if prev_weights is None else np.nan,
        "turnover_penalty_excess": np.nan,
        "turnover_penalty_scale": 1.0,
        "turnover_constraint_binding": False,
    }
    if prev_weights is None:
        return w, meta

    prev = pd.to_numeric(prev_weights.reindex(w.index), errors="coerce").fillna(0.0).astype(float)
    pre_turnover = float((w - prev).abs().sum())
    meta["turnover_pre_penalty"] = pre_turnover

    strength = max(float(getattr(cfg, "turnover_penalty_strength", 0.0)), 0.0)
    power = max(float(getattr(cfg, "turnover_penalty_power", 1.0)), 0.1)
    target = max(float(getattr(cfg, "turnover_penalty_target", 0.20)), 1e-8)
    max_turnover = getattr(cfg, "turnover_penalty_max_turnover", None)
    max_turnover = None if max_turnover is None or not np.isfinite(float(max_turnover)) else max(float(max_turnover), 0.0)

    scale = 1.0
    excess = max(pre_turnover - target, 0.0)
    if strength > 0.0 and excess > 0.0:
        scale = 1.0 / (1.0 + strength * ((excess / target) ** power))

    if max_turnover is not None and pre_turnover > max_turnover + 1e-12:
        scale = min(scale, float(max_turnover / max(pre_turnover, 1e-12)))
        meta["turnover_constraint_binding"] = True

    scale = float(np.clip(scale, 0.0, 1.0))
    if scale < 1.0 - 1e-12:
        w = prev + scale * (w - prev)
        if bool(cfg.long_only):
            w = w.clip(lower=0.0)
            s = float(w.sum())
            if np.isfinite(s) and s > 0:
                w = w / s
            else:
                w = pd.Series(np.ones(len(prev)) / len(prev), index=prev.index, dtype="float64")
        else:
            denom = float(np.abs(w).sum())
            if np.isfinite(denom) and denom > 0:
                w = w / denom
            else:
                w = prev.copy()
        meta["turnover_penalty_active"] = True

    post_turnover = float((w - prev).abs().sum())
    meta.update({
        "turnover_post_penalty": post_turnover,
        "turnover_penalty_excess": excess,
        "turnover_penalty_scale": scale,
    })
    return w.astype(float), meta


def _apply_turnover_constraint(weights: pd.Series, prev_weights: Optional[pd.Series], cfg: MicroPipelineConfig) -> Tuple[pd.Series, Dict[str, Any]]:
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0).astype(float).copy()
    requested = getattr(cfg, "turnover_constraint_max_turnover", None)
    max_turnover = None if requested is None or not np.isfinite(float(requested)) else max(float(requested), 0.0)
    meta = {
        "turnover_constraint_active": max_turnover is not None,
        "turnover_constraint_max_turnover": float(max_turnover) if max_turnover is not None else np.nan,
        "turnover_pre_constraint": float(np.abs(w).sum()) if prev_weights is None else np.nan,
        "turnover_post_constraint": float(np.abs(w).sum()) if prev_weights is None else np.nan,
        "turnover_constraint_scale": 1.0,
        "turnover_constraint_binding": False,
        "turnover_constraint_distance_clipped": 0.0,
    }
    if prev_weights is None or max_turnover is None:
        return w, meta

    prev = pd.to_numeric(prev_weights.reindex(w.index), errors="coerce").fillna(0.0).astype(float)
    pre_turnover = float((w - prev).abs().sum())
    meta["turnover_pre_constraint"] = pre_turnover
    if pre_turnover <= max_turnover + 1e-12:
        meta["turnover_post_constraint"] = pre_turnover
        return w, meta

    scale = float(np.clip(max_turnover / max(pre_turnover, 1e-12), 0.0, 1.0))
    w = prev + scale * (w - prev)
    if bool(cfg.long_only):
        w = w.clip(lower=0.0)
        s = float(w.sum())
        if np.isfinite(s) and s > 0:
            w = w / s
        else:
            w = pd.Series(np.ones(len(prev)) / len(prev), index=prev.index, dtype="float64")
    else:
        denom = float(np.abs(w).sum())
        if np.isfinite(denom) and denom > 0:
            w = w / denom
        else:
            w = prev.copy()

    post_turnover = float((w - prev).abs().sum())
    meta.update({
        "turnover_post_constraint": post_turnover,
        "turnover_constraint_scale": scale,
        "turnover_constraint_binding": True,
        "turnover_constraint_distance_clipped": float(max(pre_turnover - post_turnover, 0.0)),
    })
    return w.astype(float), meta


def _fit_quantile_regression_triplet(
    block: pd.DataFrame,
    current_row: pd.Series,
    *,
    q_low: float,
    q_high: float,
    alpha: float,
    solver: str,
    feature_cap: int,
) -> tuple[Optional[dict], Dict[str, Any]]:
    meta = {
        "qr_active": False,
        "qr_match_n": np.nan,
        "qr_feature_cols_used": 0,
        "qr_solver_used": str(solver),
    }
    if QuantileRegressor is None:
        meta["qr_solver_used"] = "sklearn_missing"
        return None, meta
    if block is None or block.empty:
        return None, meta
    if "next_return" not in block.columns:
        return None, meta

    y = pd.to_numeric(block["next_return"], errors="coerce")
    feature_cols = [c for c in block.columns if c != "next_return"]
    usable_cols = []
    for c in feature_cols:
        s = pd.to_numeric(block[c], errors="coerce")
        cur = pd.to_numeric(pd.Series([current_row.get(c, np.nan)]), errors="coerce").iloc[0]
        if s.notna().sum() >= 5 and np.isfinite(cur):
            usable_cols.append(c)
    usable_cols = usable_cols[: max(int(feature_cap), 1)]
    if not usable_cols:
        return None, meta

    X = block[usable_cols].apply(pd.to_numeric, errors="coerce")
    cur_x = pd.Series({c: pd.to_numeric(pd.Series([current_row.get(c, np.nan)]), errors="coerce").iloc[0] for c in usable_cols})
    tmp = pd.concat([X, y.rename("next_return")], axis=1).dropna()
    if tmp.shape[0] < 8:
        return None, meta

    X_fit = tmp[usable_cols].copy()
    y_fit = pd.to_numeric(tmp["next_return"], errors="coerce").astype(float)

    # simple standardization
    mu = X_fit.mean(axis=0)
    sd = X_fit.std(axis=0, ddof=1).replace(0.0, np.nan)
    sd = sd.fillna(1.0)
    Xs = ((X_fit - mu) / sd).astype(float)
    cur_scaled = ((cur_x[usable_cols] - mu) / sd).astype(float)

    try:
        mdl_lo = QuantileRegressor(quantile=float(q_low), alpha=float(alpha), solver=str(solver))
        mdl_mid = QuantileRegressor(quantile=0.50, alpha=float(alpha), solver=str(solver))
        mdl_hi = QuantileRegressor(quantile=float(q_high), alpha=float(alpha), solver=str(solver))
        mdl_lo.fit(Xs, y_fit)
        mdl_mid.fit(Xs, y_fit)
        mdl_hi.fit(Xs, y_fit)
        x0 = cur_scaled.to_numpy(dtype="float64").reshape(1, -1)
        ql = float(mdl_lo.predict(x0)[0])
        qm = float(mdl_mid.predict(x0)[0])
        qh = float(mdl_hi.predict(x0)[0])
        vals = sorted([ql, qm, qh])
        out = {"q_low": vals[0], "q_mid": vals[1], "q_high": vals[2]}
        meta.update({
            "qr_active": True,
            "qr_match_n": int(len(Xs)),
            "qr_feature_cols_used": int(len(usable_cols)),
            "qr_solver_used": str(solver),
        })
        return out, meta
    except Exception:
        meta["qr_solver_used"] = f"{solver}_failed"
        return None, meta

def _weighted_quantile(values: np.ndarray, quantile: float, weights: Optional[np.ndarray] = None) -> float:
    vals = np.asarray(values, dtype="float64")
    if vals.size == 0:
        return np.nan
    q = float(np.clip(quantile, 0.0, 1.0))
    if weights is None:
        return float(np.quantile(vals, q))
    w = np.asarray(weights, dtype="float64")
    mask = np.isfinite(vals) & np.isfinite(w) & (w > 0)
    vals = vals[mask]
    w = w[mask]
    if vals.size == 0:
        return np.nan
    order = np.argsort(vals)
    vals = vals[order]
    w = w[order]
    cum = np.cumsum(w)
    total = float(cum[-1])
    if total <= 0:
        return float(np.quantile(vals, q))
    target = q * total
    idx = int(np.searchsorted(cum, target, side="left"))
    idx = min(max(idx, 0), len(vals) - 1)
    return float(vals[idx])


def _probabilistic_backend_defaults(mode: Optional[str]) -> Dict[str, Any]:
    mode_key = str(mode or "none")
    mapping: Dict[str, Dict[str, Any]] = {
        "historical": {
            "prob_source": "historical",
            "prob_backend_family": "historical_quantiles",
            "prob_backend_name": "historical_quantiles",
            "prob_backend_variant": "historical_base",
            "prob_backend_distinct": True,
            "prob_interval_backend": "historical_quantiles",
        },
        "historical_by_regime": {
            "prob_source": "historical_by_regime",
            "prob_backend_family": "historical_quantiles_regime_filtered",
            "prob_backend_name": "historical_quantiles_regime_filtered",
            "prob_backend_variant": "regime_filtered",
            "prob_backend_distinct": True,
            "prob_interval_backend": "historical_quantiles_regime_filtered",
        },
        "historical_by_features": {
            "prob_source": "historical_by_features",
            "prob_backend_family": "historical_quantiles_feature_filtered",
            "prob_backend_name": "historical_quantiles_feature_filtered",
            "prob_backend_variant": "feature_filtered",
            "prob_backend_distinct": True,
            "prob_interval_backend": "historical_quantiles_feature_filtered",
        },
        "parametric_feature_aware": {
            "prob_source": "parametric_feature_aware",
            "prob_backend_family": "feature_weighted_gaussian",
            "prob_backend_name": "feature_weighted_gaussian",
            "prob_backend_variant": "feature_weighted_gaussian",
            "prob_backend_distinct": True,
            "prob_interval_backend": "weighted_parametric",
        },
        "knn_historical": {
            "prob_source": "knn_historical",
            "prob_backend_family": "distance_weighted_knn_quantiles",
            "prob_backend_name": "distance_weighted_knn_quantiles",
            "prob_backend_variant": "knn_historical",
            "prob_backend_distinct": True,
            "prob_interval_backend": "distance_weighted_knn_quantiles",
        },
        "feature_bucketed_historical": {
            "prob_source": "feature_bucketed_historical",
            "prob_backend_family": "bucket_match_quantiles",
            "prob_backend_name": "bucket_match_quantiles",
            "prob_backend_variant": "feature_bucketed_historical",
            "prob_backend_distinct": True,
            "prob_interval_backend": "bucket_match_quantiles",
        },
        "quantile_regression": {
            "prob_source": "quantile_regression",
            "prob_backend_family": "quantile_regression",
            "prob_backend_name": "quantile_regression",
            "prob_backend_variant": "quantile_regression",
            "prob_backend_distinct": True,
            "prob_interval_backend": "qr",
        },
        "hybrid": {
            "prob_source": "hybrid",
            "prob_backend_family": "hybrid_mix",
            "prob_backend_name": "hybrid_mix",
            "prob_backend_variant": "hybrid_mix",
            "prob_backend_distinct": True,
            "prob_interval_backend": "hybrid_mix",
        },
        "none": {
            "prob_source": "none",
            "prob_backend_family": "none",
            "prob_backend_name": "none",
            "prob_backend_variant": "none",
            "prob_backend_distinct": False,
            "prob_interval_backend": "none",
        },
    }
    return dict(mapping.get(mode_key, mapping["none"]))


def _resolve_probabilistic_mode_effective(
    requested_mode: str,
    *,
    effective_mode: Optional[str] = None,
    fallback_reason: Optional[str] = None,
    prob_source: Optional[str] = None,
    prob_backend_family: Optional[str] = None,
    prob_backend_distinct: Optional[Any] = None,
    prob_backend_name: Optional[str] = None,
    prob_backend_variant: Optional[str] = None,
    prob_interval_backend: Optional[str] = None,
    prob_backend_parent_mode: Optional[str] = None,
    prob_contract_status: Optional[str] = None,
    prob_contract_note: Optional[str] = None,
) -> Dict[str, Any]:
    requested = str(requested_mode or "none")
    effective = str(effective_mode or requested)
    requested_defaults = _probabilistic_backend_defaults(requested)
    effective_defaults = _probabilistic_backend_defaults(effective)
    fallback_triggered = bool(fallback_reason) and effective != requested
    backend_parent = str(prob_backend_parent_mode or requested)
    source = str(prob_source or effective_defaults.get("prob_source") or effective)
    interval_backend = str(prob_interval_backend or effective_defaults.get("prob_interval_backend") or source)
    backend_family_value = str(prob_backend_family or effective_defaults.get("prob_backend_family") or source)
    backend_name_value = str(prob_backend_name or effective_defaults.get("prob_backend_name") or backend_family_value)
    backend_variant_value = str(prob_backend_variant or effective_defaults.get("prob_backend_variant") or backend_name_value)
    if prob_backend_distinct is None:
        backend_distinct = bool(effective_defaults.get("prob_backend_distinct", True))
    else:
        backend_distinct = bool(prob_backend_distinct)
    status = str(prob_contract_status or ("fallback" if fallback_triggered else "ok"))
    note = str(prob_contract_note or (fallback_reason or ""))
    return {
        "probabilistic_mode_requested": requested,
        "probabilistic_mode_effective": effective,
        "prob_source": source,
        "prob_backend_family": backend_family_value,
        "prob_backend_name": backend_name_value,
        "prob_backend_variant": backend_variant_value,
        "prob_backend_distinct": backend_distinct,
        "prob_interval_backend": interval_backend,
        "prob_contract_status": status,
        "prob_contract_note": note,
        "prob_backend_parent_mode": backend_parent,
        "prob_fallback_triggered": bool(fallback_triggered),
        "prob_fallback_reason": str(fallback_reason or ""),
    }


def _compute_probabilistic_confidence(mid: float, width: float, cfg: MicroPipelineConfig) -> float:
    denom = width if np.isfinite(width) and width > 1e-12 else np.nan
    confidence_raw = 0.0 if not np.isfinite(denom) else abs(float(mid)) / denom
    conf = 1.0 + float(cfg.probabilistic_confidence_scale) * float(confidence_raw)
    return float(np.clip(conf, float(cfg.probabilistic_confidence_min_mult), float(cfg.probabilistic_confidence_max_mult)))


def _build_probabilistic_summary_from_triplet(
    low: Any,
    mid: Any,
    high: Any,
    cfg: MicroPipelineConfig,
    *,
    requested_mode: str,
    effective_mode: str,
    prob_source: Optional[str] = None,
    prob_backend_family: Optional[str] = None,
    prob_backend_distinct: Optional[Any] = None,
    prob_backend_name: Optional[str] = None,
    prob_backend_variant: Optional[str] = None,
    prob_interval_backend: Optional[str] = None,
    prob_backend_parent_mode: Optional[str] = None,
    prob_contract_status: Optional[str] = None,
    prob_contract_note: Optional[str] = None,
    prob_fallback_reason: Optional[str] = None,
    extras: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    low_f = float(low)
    mid_f = float(mid)
    high_f = float(high)
    vals = [low_f, mid_f, high_f]
    if not all(np.isfinite(v) for v in vals):
        return {}
    ordered = sorted(vals)
    low_f, mid_f, high_f = ordered[0], ordered[1], ordered[2]
    width = float(abs(high_f - low_f))
    downside = float(max(-low_f, 0.0))
    conf = _compute_probabilistic_confidence(mid_f, width, cfg)
    out = {
        "prob_q_low": low_f,
        "prob_q_mid": mid_f,
        "prob_q_high": high_f,
        "pred_low": low_f,
        "pred_mid": mid_f,
        "pred_high": high_f,
        "prob_interval_width": width,
        "prob_downside": downside,
        "prob_confidence": conf,
        "probabilistic_active": True,
    }
    out.update(_resolve_probabilistic_mode_effective(
        requested_mode,
        effective_mode=effective_mode,
        fallback_reason=prob_fallback_reason,
        prob_source=prob_source,
        prob_backend_family=prob_backend_family,
        prob_backend_distinct=prob_backend_distinct,
        prob_backend_name=prob_backend_name,
        prob_backend_variant=prob_backend_variant,
        prob_interval_backend=prob_interval_backend,
        prob_backend_parent_mode=prob_backend_parent_mode,
        prob_contract_status=prob_contract_status,
        prob_contract_note=prob_contract_note,
    ))
    if extras:
        out.update(extras)
    return out


def _build_base_probabilistic_summary_from_sample(
    sample: pd.Series,
    cfg: MicroPipelineConfig,
    *,
    prob_source: str,
    requested_mode: Optional[str] = None,
    effective_mode: Optional[str] = None,
    prob_backend_family: Optional[str] = None,
    prob_backend_distinct: Optional[Any] = None,
    prob_backend_name: Optional[str] = None,
    prob_backend_variant: Optional[str] = None,
    prob_interval_backend: Optional[str] = None,
    prob_backend_parent_mode: Optional[str] = None,
    prob_contract_status: Optional[str] = None,
    prob_contract_note: Optional[str] = None,
    prob_fallback_reason: Optional[str] = None,
    extras: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    s = pd.to_numeric(sample, errors="coerce").dropna().astype(float)
    ql = float(cfg.probabilistic_q_low)
    qh = float(cfg.probabilistic_q_high)
    if s.shape[0] < max(int(cfg.probabilistic_min_obs), 3) or not (0.0 < ql < qh < 1.0):
        return {}
    low = float(s.quantile(ql))
    mid = float(s.quantile(0.5))
    high = float(s.quantile(qh))
    return _build_probabilistic_summary_from_triplet(
        low,
        mid,
        high,
        cfg,
        requested_mode=str(requested_mode or prob_source),
        effective_mode=str(effective_mode or prob_source),
        prob_source=str(prob_source),
        prob_backend_family=prob_backend_family,
        prob_backend_distinct=prob_backend_distinct,
        prob_backend_name=prob_backend_name,
        prob_backend_variant=prob_backend_variant,
        prob_interval_backend=prob_interval_backend,
        prob_backend_parent_mode=prob_backend_parent_mode or str(requested_mode or prob_source),
        prob_contract_status=prob_contract_status,
        prob_contract_note=prob_contract_note,
        prob_fallback_reason=prob_fallback_reason,
        extras=extras,
    )


def _finalize_probabilistic_overlay_output(
    row: Optional[Dict[str, Any]],
    cfg: MicroPipelineConfig,
    *,
    requested_mode: str,
    asset: Optional[str] = None,
) -> Dict[str, Any]:
    if not row:
        return {}
    out = dict(row)
    if "prob_q_low" not in out and "pred_low" in out:
        out["prob_q_low"] = out.get("pred_low")
    if "prob_q_mid" not in out and "pred_mid" in out:
        out["prob_q_mid"] = out.get("pred_mid")
    if "prob_q_high" not in out and "pred_high" in out:
        out["prob_q_high"] = out.get("pred_high")
    out["pred_low"] = float(pd.to_numeric(pd.Series([out.get("pred_low", out.get("prob_q_low", np.nan))]), errors="coerce").iloc[0])
    out["pred_mid"] = float(pd.to_numeric(pd.Series([out.get("pred_mid", out.get("prob_q_mid", np.nan))]), errors="coerce").iloc[0])
    out["pred_high"] = float(pd.to_numeric(pd.Series([out.get("pred_high", out.get("prob_q_high", np.nan))]), errors="coerce").iloc[0])
    for col in ["prob_q_low", "prob_q_mid", "prob_q_high", "pred_low", "pred_mid", "pred_high"]:
        out[col] = float(pd.to_numeric(pd.Series([out.get(col, np.nan)]), errors="coerce").iloc[0])
    width = pd.to_numeric(pd.Series([out.get("prob_interval_width", abs(out["prob_q_high"] - out["prob_q_low"]))]), errors="coerce").iloc[0]
    downside = pd.to_numeric(pd.Series([out.get("prob_downside", max(-out["prob_q_low"], 0.0))]), errors="coerce").iloc[0]
    conf = pd.to_numeric(pd.Series([out.get("prob_confidence", _compute_probabilistic_confidence(out["prob_q_mid"], float(width) if np.isfinite(width) else np.nan, cfg))]), errors="coerce").iloc[0]
    out["prob_interval_width"] = float(max(width, 0.0)) if np.isfinite(width) else np.nan
    out["prob_downside"] = float(max(downside, 0.0)) if np.isfinite(downside) else np.nan
    out["prob_confidence"] = float(conf) if np.isfinite(conf) else np.nan
    out["probabilistic_active"] = bool(out.get("probabilistic_active", True))
    contract = _resolve_probabilistic_mode_effective(
        requested_mode,
        effective_mode=str(out.get("probabilistic_mode_effective", out.get("prob_source", requested_mode))),
        fallback_reason=str(out.get("prob_fallback_reason", "") or "") or None,
        prob_source=str(out.get("prob_source", out.get("probabilistic_mode_effective", requested_mode))),
        prob_backend_family=str(out.get("prob_backend_family", out.get("prob_source", out.get("probabilistic_mode_effective", requested_mode)))),
        prob_backend_distinct=bool(out.get("prob_backend_distinct", True)),
        prob_backend_name=str(out.get("prob_backend_name", out.get("prob_backend_family", out.get("prob_source", requested_mode)))),
        prob_backend_variant=str(out.get("prob_backend_variant", out.get("prob_backend_name", out.get("prob_source", requested_mode)))),
        prob_interval_backend=str(out.get("prob_interval_backend", out.get("prob_backend_family", out.get("prob_source", requested_mode)))),
        prob_backend_parent_mode=str(out.get("prob_backend_parent_mode", requested_mode)),
        prob_contract_status=str(out.get("prob_contract_status", "fallback" if bool(out.get("prob_fallback_triggered", False)) else "ok")),
        prob_contract_note=str(out.get("prob_contract_note", out.get("prob_fallback_reason", ""))),
    )
    out.update(contract)
    out.setdefault("prob_dispatch_path", "canonical_dispatch")
    out.setdefault("prob_dispatch_version", "v2")
    if asset is not None:
        out[cfg.asset_col] = str(asset)
    return out


def _build_probabilistic_fallback_row(sample: pd.Series, cfg: MicroPipelineConfig, *, requested_mode: str, reason: str) -> Dict[str, Any]:
    return _build_base_probabilistic_summary_from_sample(
        sample,
        cfg,
        prob_source="historical",
        requested_mode=requested_mode,
        effective_mode="historical",
        prob_backend_family="historical_quantiles",
        prob_backend_distinct=True,
        prob_backend_name="historical_quantiles",
        prob_backend_variant="historical_fallback",
        prob_interval_backend="historical_quantiles",
        prob_backend_parent_mode=requested_mode,
        prob_contract_status="fallback",
        prob_contract_note=reason,
        prob_fallback_reason=reason,
    )


def _select_probabilistic_feature_block(asset: str, historical_feature_pairs: Optional[pd.DataFrame], current_feature_state: Optional[pd.DataFrame], cfg: MicroPipelineConfig, *, candidate_cols: Optional[List[str]] = None) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    meta = {
        "feature_match_n": np.nan,
        "feature_distance": np.nan,
        "feature_cols_used": 0.0,
        "feature_filter_active": False,
    }
    if historical_feature_pairs is None or current_feature_state is None or historical_feature_pairs.empty or current_feature_state.empty:
        return None, meta
    if str(asset) not in set(current_feature_state.index.astype(str)):
        return None, meta
    asset_pairs = historical_feature_pairs.loc[historical_feature_pairs[cfg.asset_col].astype(str) == str(asset)].copy()
    if asset_pairs.empty:
        return None, meta
    current_row = current_feature_state.loc[str(asset)] if str(asset) in current_feature_state.index.astype(str) else current_feature_state.loc[asset]
    if isinstance(current_row, pd.DataFrame):
        current_row = current_row.iloc[-1]
    feature_cols = [c for c in (candidate_cols or []) if c in asset_pairs.columns and c in current_feature_state.columns]
    usable_cols = []
    for col in feature_cols:
        cur = pd.to_numeric(pd.Series([current_row.get(col, np.nan)]), errors="coerce").iloc[0]
        s = pd.to_numeric(asset_pairs[col], errors="coerce")
        min_obs = max(int(getattr(cfg, "probabilistic_feature_filter_min_obs", cfg.probabilistic_min_obs)), 3)
        if np.isfinite(cur) and s.notna().sum() >= min_obs:
            usable_cols.append(col)
    if not usable_cols:
        return None, meta
    block = asset_pairs[[*usable_cols, "next_return"]].copy().dropna()
    if block.shape[0] < max(int(cfg.probabilistic_min_obs), 3):
        return None, meta
    X = block[usable_cols].to_numpy(dtype="float64")
    cur_vec = np.array([float(pd.to_numeric(pd.Series([current_row.get(c, np.nan)]), errors="coerce").iloc[0]) for c in usable_cols], dtype="float64")
    scale = np.nanstd(X, axis=0, ddof=1)
    scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
    d = np.sqrt(np.mean(((X - cur_vec) / scale) ** 2, axis=1))
    block["_feature_distance"] = d
    meta.update({
        "feature_match_n": float(block.shape[0]),
        "feature_distance": float(np.nanmean(d)) if len(d) else np.nan,
        "feature_cols_used": float(len(usable_cols)),
        "feature_filter_active": True,
    })
    return block, meta


def _build_prob_overlay_historical(asset: str, sample: pd.Series, cfg: MicroPipelineConfig, **_: Any) -> Dict[str, Any]:
    return _build_base_probabilistic_summary_from_sample(
        sample,
        cfg,
        prob_source="historical",
        requested_mode="historical",
        effective_mode="historical",
        prob_backend_family="historical_quantiles",
        prob_backend_distinct=True,
        prob_backend_name="historical_quantiles",
        prob_backend_variant="historical_base",
        prob_interval_backend="historical_quantiles",
        prob_backend_parent_mode="historical",
    )


def _build_prob_overlay_historical_by_regime(asset: str, sample: pd.Series, cfg: MicroPipelineConfig, **_: Any) -> Dict[str, Any]:
    requested_mode = "historical_by_regime"
    recent_n = max(int(getattr(cfg, "probabilistic_regime_min_obs", cfg.probabilistic_min_obs)), 3)
    regime_sample = sample.tail(recent_n)
    if regime_sample.shape[0] >= max(int(cfg.probabilistic_min_obs), 3):
        return _build_base_probabilistic_summary_from_sample(
            regime_sample,
            cfg,
            prob_source="historical_by_regime",
            requested_mode=requested_mode,
            effective_mode="historical_by_regime",
            prob_backend_family="historical_quantiles_regime_filtered",
            prob_backend_distinct=True,
            prob_backend_name="historical_quantiles_regime_filtered",
            prob_backend_variant="regime_filtered",
            prob_interval_backend="historical_quantiles_regime_filtered",
            prob_backend_parent_mode=requested_mode,
        )
    return _build_probabilistic_fallback_row(sample, cfg, requested_mode=requested_mode, reason="insufficient regime-specific observations")


def _build_prob_overlay_historical_by_features(
    asset: str,
    sample: pd.Series,
    cfg: MicroPipelineConfig,
    *,
    historical_feature_pairs: Optional[pd.DataFrame] = None,
    current_feature_state: Optional[pd.DataFrame] = None,
    base_feature_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    requested_mode = "historical_by_features"
    block, meta = _select_probabilistic_feature_block(str(asset), historical_feature_pairs, current_feature_state, cfg, candidate_cols=base_feature_cols)
    if block is not None:
        k = getattr(cfg, "probabilistic_feature_filter_k", None)
        k = min(int(k), len(block)) if k is not None and np.isfinite(float(k)) else min(len(block), max(int(cfg.probabilistic_min_obs), 12))
        block = block.sort_values("_feature_distance", ascending=True).head(max(k, max(int(cfg.probabilistic_min_obs), 3)))
        row = _build_base_probabilistic_summary_from_sample(
            block["next_return"],
            cfg,
            prob_source=requested_mode,
            requested_mode=requested_mode,
            effective_mode=requested_mode,
            prob_backend_family="historical_quantiles_feature_filtered",
            prob_backend_distinct=True,
            prob_backend_name="historical_quantiles_feature_filtered",
            prob_backend_variant="feature_filtered",
            prob_interval_backend="historical_quantiles_feature_filtered",
            prob_backend_parent_mode=requested_mode,
            extras={
                "probabilistic_feature_match_n": float(len(block)),
                "probabilistic_feature_distance": float(block["_feature_distance"].mean()) if len(block) else np.nan,
                "probabilistic_feature_cols_used": float(meta.get("feature_cols_used", 0.0)),
                "probabilistic_feature_filter_active": True,
            },
        )
        if row:
            return row
    return _build_probabilistic_fallback_row(sample, cfg, requested_mode=requested_mode, reason="feature-conditioned historical block unavailable")


def _build_prob_overlay_knn_historical(
    asset: str,
    sample: pd.Series,
    cfg: MicroPipelineConfig,
    *,
    historical_feature_pairs: Optional[pd.DataFrame] = None,
    current_feature_state: Optional[pd.DataFrame] = None,
    base_feature_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    requested_mode = "knn_historical"
    ql = float(cfg.probabilistic_q_low)
    qh = float(cfg.probabilistic_q_high)
    block, meta = _select_probabilistic_feature_block(str(asset), historical_feature_pairs, current_feature_state, cfg, candidate_cols=base_feature_cols)
    if block is not None:
        k_raw = getattr(cfg, "probabilistic_knn_k", None)
        k = min(int(k_raw), len(block)) if k_raw is not None and np.isfinite(float(k_raw)) else min(len(block), max(int(cfg.probabilistic_knn_min_obs), 12))
        k = max(k, max(int(cfg.probabilistic_knn_min_obs), 3))
        block = block.sort_values("_feature_distance", ascending=True).head(min(k, len(block)))
        vals = block["next_return"].to_numpy(dtype="float64")
        d = block["_feature_distance"].to_numpy(dtype="float64")
        weighted = bool(getattr(cfg, "probabilistic_knn_weighted_quantiles", True))
        eps = max(float(getattr(cfg, "probabilistic_knn_weight_eps", 1e-6)), 1e-12)
        pwr = max(float(getattr(cfg, "probabilistic_knn_distance_power", 2.0)), 0.1)
        weights = 1.0 / np.power(np.maximum(d, eps), pwr) if weighted else None
        row = _build_probabilistic_summary_from_triplet(
            _weighted_quantile(vals, ql, weights),
            _weighted_quantile(vals, 0.5, weights),
            _weighted_quantile(vals, qh, weights),
            cfg,
            requested_mode=requested_mode,
            effective_mode=requested_mode,
            prob_source=requested_mode,
            prob_backend_family="distance_weighted_knn_quantiles",
            prob_backend_distinct=True,
            prob_backend_name="distance_weighted_knn_quantiles",
            prob_backend_variant="knn_historical",
            prob_interval_backend="distance_weighted_knn_quantiles",
            prob_backend_parent_mode=requested_mode,
            extras={
                "probabilistic_feature_match_n": float(len(block)),
                "probabilistic_feature_distance": float(np.nanmean(d)) if len(d) else np.nan,
                "probabilistic_feature_cols_used": float(meta.get("feature_cols_used", 0.0)),
                "probabilistic_feature_filter_active": True,
                "prob_knn_weighted": bool(weighted),
            },
        )
        if row:
            return row
    return _build_probabilistic_fallback_row(sample, cfg, requested_mode=requested_mode, reason="kNN historical neighborhood unavailable")


def _build_prob_overlay_feature_bucketed_historical(
    asset: str,
    sample: pd.Series,
    cfg: MicroPipelineConfig,
    *,
    historical_feature_pairs: Optional[pd.DataFrame] = None,
    current_feature_state: Optional[pd.DataFrame] = None,
    base_feature_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    requested_mode = "feature_bucketed_historical"
    block, meta = _select_probabilistic_feature_block(str(asset), historical_feature_pairs, current_feature_state, cfg, candidate_cols=base_feature_cols)
    if block is not None and current_feature_state is not None and not current_feature_state.empty and str(asset) in set(current_feature_state.index.astype(str)):
        current_row = current_feature_state.loc[str(asset)] if str(asset) in current_feature_state.index.astype(str) else current_feature_state.loc[asset]
        if isinstance(current_row, pd.DataFrame):
            current_row = current_row.iloc[-1]
        usable_cols = [c for c in (base_feature_cols or []) if c in block.columns][: max(int(getattr(cfg, "probabilistic_bucket_max_features", 6)), 1)]
        matched = block.copy()
        bins = max(int(getattr(cfg, "probabilistic_bucket_n_bins", 4)), 2)
        min_match = max(int(getattr(cfg, "probabilistic_bucket_match_min_features", 2)), 1)
        match_mask = np.zeros(len(matched), dtype=int)
        used = 0
        for col in usable_cols:
            cur = pd.to_numeric(pd.Series([current_row.get(col, np.nan)]), errors="coerce").iloc[0]
            s = pd.to_numeric(matched[col], errors="coerce")
            if not np.isfinite(cur) or s.notna().sum() < max(int(cfg.probabilistic_bucket_min_obs), 3):
                continue
            try:
                edges = np.unique(np.nanquantile(s.to_numpy(dtype="float64"), np.linspace(0.0, 1.0, bins + 1)))
            except Exception:
                continue
            if len(edges) < 3:
                continue
            cur_bin = np.digitize([cur], edges[1:-1], right=True)[0]
            row_bins = np.digitize(s.to_numpy(dtype="float64"), edges[1:-1], right=True)
            match_mask += (row_bins == cur_bin).astype(int)
            used += 1
        if used > 0:
            matched = matched.loc[match_mask >= min(min_match, used)].copy()
            if matched.shape[0] >= max(int(cfg.probabilistic_bucket_min_obs), 3):
                row = _build_base_probabilistic_summary_from_sample(
                    matched["next_return"],
                    cfg,
                    prob_source=requested_mode,
                    requested_mode=requested_mode,
                    effective_mode=requested_mode,
            prob_backend_family="bucket_match_quantiles",
            prob_backend_distinct=True,
            prob_backend_name="bucket_match_quantiles",
            prob_backend_variant="feature_bucketed_historical",
            prob_interval_backend="bucket_match_quantiles",
                    prob_backend_parent_mode=requested_mode,
                    extras={
                        "probabilistic_feature_match_n": float(len(matched)),
                        "probabilistic_feature_distance": float(matched.get("_feature_distance", pd.Series(dtype=float)).mean()) if "_feature_distance" in matched.columns else np.nan,
                        "probabilistic_feature_cols_used": float(used),
                        "probabilistic_feature_filter_active": True,
                        "probabilistic_bucket_match_features": float(min(min_match, used)),
                    },
                )
                if row:
                    return row
    return _build_probabilistic_fallback_row(sample, cfg, requested_mode=requested_mode, reason="feature buckets unavailable or too sparse")


def _build_prob_overlay_parametric_feature_aware(
    asset: str,
    sample: pd.Series,
    cfg: MicroPipelineConfig,
    *,
    historical_feature_pairs: Optional[pd.DataFrame] = None,
    current_feature_state: Optional[pd.DataFrame] = None,
    base_feature_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    requested_mode = "parametric_feature_aware"
    block, meta = _select_probabilistic_feature_block(str(asset), historical_feature_pairs, current_feature_state, cfg, candidate_cols=base_feature_cols)
    if block is not None:
        k = getattr(cfg, "probabilistic_feature_filter_k", None)
        k = min(int(k), len(block)) if k is not None and np.isfinite(float(k)) else min(len(block), max(int(cfg.probabilistic_min_obs), 12))
        block = block.sort_values("_feature_distance", ascending=True).head(max(k, max(int(cfg.probabilistic_min_obs), 3)))
        vals = pd.to_numeric(block["next_return"], errors="coerce").dropna().to_numpy(dtype="float64")
        if vals.size >= max(int(cfg.probabilistic_min_obs), 3):
            d = block.loc[pd.to_numeric(block["next_return"], errors="coerce").notna(), "_feature_distance"].to_numpy(dtype="float64")
            use_w = bool(getattr(cfg, "probabilistic_parametric_use_neighbor_weights", True))
            if use_w:
                eps = max(float(getattr(cfg, "probabilistic_knn_weight_eps", 1e-6)), 1e-12)
                pwr = max(float(getattr(cfg, "probabilistic_knn_distance_power", 2.0)), 0.1)
                weights = 1.0 / np.power(np.maximum(d, eps), pwr)
                mu = float(np.average(vals, weights=weights))
                var = float(np.average((vals - mu) ** 2, weights=weights))
            else:
                mu = float(np.mean(vals))
                var = float(np.var(vals, ddof=1)) if vals.size > 1 else 0.0
            sigma = float(np.sqrt(max(var, 0.0)))
            sigma = max(sigma, float(getattr(cfg, "probabilistic_parametric_min_sigma", 1e-4)))
            sigma = min(sigma, float(sample.std(ddof=1) if sample.shape[0] > 1 else sigma) * float(getattr(cfg, "probabilistic_parametric_max_sigma_mult", 3.0))) if sample.shape[0] > 1 else sigma
            nd = NormalDist(mu=mu, sigma=max(sigma, 1e-8))
            row = _build_probabilistic_summary_from_triplet(
                nd.inv_cdf(float(cfg.probabilistic_q_low)),
                mu,
                nd.inv_cdf(float(cfg.probabilistic_q_high)),
                cfg,
                requested_mode=requested_mode,
                effective_mode=requested_mode,
                prob_source=requested_mode,
                prob_backend_family="feature_weighted_gaussian",
                prob_backend_distinct=True,
                prob_backend_name="feature_weighted_gaussian",
                prob_backend_variant="feature_weighted_gaussian",
                prob_interval_backend="weighted_parametric",
                prob_backend_parent_mode=requested_mode,
                extras={
                    "prob_parametric_sigma": float(sigma),
                    "probabilistic_feature_match_n": float(len(vals)),
                    "probabilistic_feature_distance": float(np.nanmean(d)) if len(d) else np.nan,
                    "probabilistic_feature_cols_used": float(meta.get("feature_cols_used", 0.0)),
                    "probabilistic_feature_filter_active": True,
                },
            )
            if row:
                return row
    return _build_probabilistic_fallback_row(sample, cfg, requested_mode=requested_mode, reason="parametric feature-aware fit unavailable")


def _build_prob_overlay_quantile_regression(
    asset: str,
    sample: pd.Series,
    cfg: MicroPipelineConfig,
    *,
    historical_feature_pairs: Optional[pd.DataFrame] = None,
    current_feature_state: Optional[pd.DataFrame] = None,
    base_feature_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    requested_mode = "quantile_regression"
    block, _ = _select_probabilistic_feature_block(str(asset), historical_feature_pairs, current_feature_state, cfg, candidate_cols=base_feature_cols)
    if block is not None and current_feature_state is not None and not current_feature_state.empty and str(asset) in set(current_feature_state.index.astype(str)):
        current_row = current_feature_state.loc[str(asset)] if str(asset) in current_feature_state.index.astype(str) else current_feature_state.loc[asset]
        if isinstance(current_row, pd.DataFrame):
            current_row = current_row.iloc[-1]
        qr_out, qr_meta = _fit_quantile_regression_triplet(
            block,
            current_row,
            q_low=float(cfg.probabilistic_q_low),
            q_high=float(cfg.probabilistic_q_high),
            alpha=float(getattr(cfg, "probabilistic_qr_alpha", 0.0)),
            solver=str(getattr(cfg, "probabilistic_qr_solver", "highs")),
            feature_cap=int(getattr(cfg, "probabilistic_qr_feature_cap", 8)),
        )
        if qr_out is not None:
            row = _build_probabilistic_summary_from_triplet(
                qr_out["q_low"],
                qr_out["q_mid"],
                qr_out["q_high"],
                cfg,
                requested_mode=requested_mode,
                effective_mode=requested_mode,
                prob_source=requested_mode,
                prob_backend_family="quantile_regression",
                prob_backend_distinct=True,
                prob_backend_name="quantile_regression",
                prob_backend_variant="quantile_regression",
                prob_interval_backend="qr",
                prob_backend_parent_mode=requested_mode,
                extras={
                    "probabilistic_feature_match_n": float(qr_meta.get("qr_match_n", np.nan)),
                    "probabilistic_feature_cols_used": float(qr_meta.get("qr_feature_cols_used", np.nan)),
                    "prob_qr_active": bool(qr_meta.get("qr_active", False)),
                    "prob_qr_solver_used": str(qr_meta.get("qr_solver_used", "")),
                },
            )
            if row:
                return row
    return _build_probabilistic_fallback_row(sample, cfg, requested_mode=requested_mode, reason="quantile regression fit unavailable")


def _build_prob_overlay_hybrid(
    asset: str,
    sample: pd.Series,
    cfg: MicroPipelineConfig,
    *,
    historical_feature_pairs: Optional[pd.DataFrame] = None,
    current_feature_state: Optional[pd.DataFrame] = None,
    base_feature_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    requested_mode = "hybrid"
    ql = float(cfg.probabilistic_q_low)
    qh = float(cfg.probabilistic_q_high)
    knn_row = None
    qr_row = None
    block, meta = _select_probabilistic_feature_block(str(asset), historical_feature_pairs, current_feature_state, cfg, candidate_cols=base_feature_cols)
    if block is not None:
        k_raw = getattr(cfg, "probabilistic_knn_k", None)
        k = min(int(k_raw), len(block)) if k_raw is not None and np.isfinite(float(k_raw)) else min(len(block), max(int(cfg.probabilistic_knn_min_obs), 12))
        block_knn = block.sort_values("_feature_distance", ascending=True).head(max(min(k, len(block)), max(int(cfg.probabilistic_knn_min_obs), 3)))
        vals = block_knn["next_return"].to_numpy(dtype="float64")
        d = block_knn["_feature_distance"].to_numpy(dtype="float64")
        weights = None
        if bool(getattr(cfg, "probabilistic_knn_weighted_quantiles", True)):
            eps = max(float(getattr(cfg, "probabilistic_knn_weight_eps", 1e-6)), 1e-12)
            pwr = max(float(getattr(cfg, "probabilistic_knn_distance_power", 2.0)), 0.1)
            weights = 1.0 / np.power(np.maximum(d, eps), pwr)
        knn_row = {
            "low": _weighted_quantile(vals, ql, weights),
            "mid": _weighted_quantile(vals, 0.5, weights),
            "high": _weighted_quantile(vals, qh, weights),
            "match_n": float(len(block_knn)),
            "distance": float(np.nanmean(d)) if len(d) else np.nan,
        }
        if current_feature_state is not None and not current_feature_state.empty and str(asset) in set(current_feature_state.index.astype(str)):
            current_row = current_feature_state.loc[str(asset)] if str(asset) in current_feature_state.index.astype(str) else current_feature_state.loc[asset]
            if isinstance(current_row, pd.DataFrame):
                current_row = current_row.iloc[-1]
            qr_out, qr_meta = _fit_quantile_regression_triplet(
                block,
                current_row,
                q_low=ql,
                q_high=qh,
                alpha=float(getattr(cfg, "probabilistic_qr_alpha", 0.0)),
                solver=str(getattr(cfg, "probabilistic_qr_solver", "highs")),
                feature_cap=int(getattr(cfg, "probabilistic_qr_feature_cap", 8)),
            )
            if qr_out is not None:
                qr_row = {"low": float(qr_out["q_low"]), "mid": float(qr_out["q_mid"]), "high": float(qr_out["q_high"]), "meta": qr_meta}
    if knn_row is not None:
        base_weight = float(np.clip(getattr(cfg, "probabilistic_hybrid_weight", 0.50), 0.0, 1.0))
        qr_weight = base_weight if qr_row is not None else 0.0
        if bool(getattr(cfg, "probabilistic_hybrid_use_confidence", True)) and qr_row is not None:
            knn_width = abs(float(knn_row["high"]) - float(knn_row["low"]))
            qr_width = abs(float(qr_row["high"]) - float(qr_row["low"]))
            width_total = knn_width + qr_width
            if width_total > 1e-12:
                qr_weight = float(np.clip(knn_width / width_total, float(getattr(cfg, "probabilistic_hybrid_min_qr_weight", 0.25)), float(getattr(cfg, "probabilistic_hybrid_max_qr_weight", 0.75))))
        qr_mid = float(qr_row["mid"]) if qr_row is not None else np.nan
        knn_mid = float(knn_row["mid"]) if knn_row is not None else np.nan
        qr_width_component = float(abs(float(qr_row["high"]) - float(qr_row["low"]))) if qr_row is not None else np.nan
        knn_width_component = float(abs(float(knn_row["high"]) - float(knn_row["low"]))) if knn_row is not None else np.nan
        eps = 1e-12
        if np.isfinite(qr_mid) and np.isfinite(knn_mid):
            denom_agree = abs(qr_width_component) + abs(knn_width_component) + eps
            agreement = 1.0 - min(abs(qr_mid - knn_mid) / denom_agree, 1.0)
        else:
            agreement = np.nan
        knn_weight = float(1.0 - qr_weight)
        row = _build_probabilistic_summary_from_triplet(
            knn_weight * float(knn_row["low"]) + qr_weight * float(qr_row["low"] if qr_row else knn_row["low"]),
            knn_weight * float(knn_row["mid"]) + qr_weight * float(qr_row["mid"] if qr_row else knn_row["mid"]),
            knn_weight * float(knn_row["high"]) + qr_weight * float(qr_row["high"] if qr_row else knn_row["high"]),
            cfg,
            requested_mode=requested_mode,
            effective_mode=requested_mode,
            prob_source=requested_mode,
            prob_backend_family="hybrid_mix",
            prob_backend_distinct=True,
            prob_backend_name="hybrid_mix",
            prob_backend_variant="hybrid_mix",
            prob_interval_backend="hybrid_mix",
            prob_backend_parent_mode=requested_mode,
            extras={
                "probabilistic_feature_match_n": float(knn_row.get("match_n", np.nan)),
                "probabilistic_feature_distance": float(knn_row.get("distance", np.nan)),
                "probabilistic_feature_cols_used": float(meta.get("feature_cols_used", 0.0)) if block is not None else np.nan,
                "prob_hybrid_qr_weight": float(qr_weight),
                "prob_hybrid_knn_weight": float(knn_weight),
                "prob_hybrid_qr_mid": qr_mid,
                "prob_hybrid_knn_mid": knn_mid,
                "prob_hybrid_qr_width": qr_width_component,
                "prob_hybrid_knn_width": knn_width_component,
                "prob_hybrid_component_agreement": float(agreement) if np.isfinite(agreement) else np.nan,
                "prob_qr_active": bool(qr_row is not None),
            },
        )
        if row:
            return row
    return _build_probabilistic_fallback_row(sample, cfg, requested_mode=requested_mode, reason="hybrid overlay components unavailable")


def _build_probabilistic_overlay_dispatch(
    mode: str,
    asset: str,
    sample: pd.Series,
    cfg: MicroPipelineConfig,
    *,
    historical_feature_pairs: Optional[pd.DataFrame] = None,
    current_feature_state: Optional[pd.DataFrame] = None,
    base_feature_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    requested_mode = str(mode or getattr(cfg, "probabilistic_mode", "none") or "none")
    s = pd.to_numeric(sample, errors="coerce").dropna().astype(float)
    if requested_mode == "none" or s.shape[0] < max(int(cfg.probabilistic_min_obs), 3):
        return {}

    dispatch_map = {
        "historical": _build_prob_overlay_historical,
        "historical_by_regime": _build_prob_overlay_historical_by_regime,
        "historical_by_features": _build_prob_overlay_historical_by_features,
        "parametric_feature_aware": _build_prob_overlay_parametric_feature_aware,
        "knn_historical": _build_prob_overlay_knn_historical,
        "feature_bucketed_historical": _build_prob_overlay_feature_bucketed_historical,
        "quantile_regression": _build_prob_overlay_quantile_regression,
        "hybrid": _build_prob_overlay_hybrid,
    }
    builder = dispatch_map.get(requested_mode, _build_prob_overlay_historical)
    row = builder(
        str(asset),
        s,
        cfg,
        historical_feature_pairs=historical_feature_pairs,
        current_feature_state=current_feature_state,
        base_feature_cols=base_feature_cols,
    )
    row = _finalize_probabilistic_overlay_output(
        row,
        cfg,
        requested_mode=requested_mode,
        asset=str(asset),
    )
    if row:
        row["prob_dispatch_path"] = "canonical_dispatch"
        row["prob_dispatch_version"] = "v2"
    return row


def _build_probabilistic_forecast_from_window(
    use: pd.DataFrame,
    cfg: MicroPipelineConfig,
    *,
    historical_feature_pairs: Optional[pd.DataFrame] = None,
    current_feature_state: Optional[pd.DataFrame] = None,
    base_feature_cols: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    if use is None or not isinstance(use, pd.DataFrame) or use.empty:
        return None
    mode = str(getattr(cfg, "probabilistic_mode", "none") or "none")
    if mode == "none":
        return None
    rows: List[Dict[str, Any]] = []
    for asset in use.columns:
        sample = pd.to_numeric(use[asset], errors="coerce").dropna().astype(float)
        if sample.shape[0] < max(int(cfg.probabilistic_min_obs), 3):
            continue
        row = _build_probabilistic_overlay_dispatch(
            mode,
            str(asset),
            sample,
            cfg,
            historical_feature_pairs=historical_feature_pairs,
            current_feature_state=current_feature_state,
            base_feature_cols=base_feature_cols,
        )
        if row:
            rows.append(row)
    if not rows:
        return None
    prob_df = pd.DataFrame(rows)
    if cfg.asset_col not in prob_df.columns:
        prob_df[cfg.asset_col] = prob_df.index.astype(str)
    prob_df = prob_df.set_index(cfg.asset_col)
    return prob_df


def _integrate_probabilistic_into_mu(mu_hat: pd.Series, prob_df: Optional[pd.DataFrame], cfg: MicroPipelineConfig) -> Tuple[pd.Series, Dict[str, float]]:
    if prob_df is None or prob_df.empty or str(cfg.probabilistic_mode) == "none":
        return mu_hat, {
            "probabilistic_active": False,
            "probabilistic_interval_width_mean": np.nan,
            "probabilistic_downside_mean": np.nan,
            "probabilistic_confidence_mean": np.nan,
            "probabilistic_mu_penalty_mean_abs": np.nan,
            "probabilistic_mu_penalty_max_abs": np.nan,
            "probabilistic_mode_requested": str(getattr(cfg, "probabilistic_mode", "none")),
            "probabilistic_mode_effective": str(getattr(cfg, "probabilistic_mode", "none")),
            "prob_source": str(getattr(cfg, "probabilistic_mode", "none")),
            "prob_contract_status": "inactive",
            "prob_dispatch_path": "canonical_dispatch",
            "prob_dispatch_version": "v2",
            "pred_low": np.nan,
            "pred_mid": np.nan,
            "pred_high": np.nan,
            "prob_q_low": np.nan,
            "prob_q_mid": np.nan,
            "prob_q_high": np.nan,
            "prob_fallback_triggered": np.nan,
        }

    aligned = prob_df.reindex(mu_hat.index).copy()
    for src, dst in (("pred_low", "prob_q_low"), ("pred_mid", "prob_q_mid"), ("pred_high", "prob_q_high")):
        if dst not in aligned.columns and src in aligned.columns:
            aligned[dst] = aligned[src]
        if src not in aligned.columns and dst in aligned.columns:
            aligned[src] = aligned[dst]

    q_low = pd.to_numeric(aligned.get("prob_q_low", aligned.get("pred_low")), errors="coerce")
    q_mid = pd.to_numeric(aligned.get("prob_q_mid", aligned.get("pred_mid")), errors="coerce")
    q_high = pd.to_numeric(aligned.get("prob_q_high", aligned.get("pred_high")), errors="coerce")
    mid = q_mid.fillna(mu_hat)
    width = pd.to_numeric(aligned.get("prob_interval_width"), errors="coerce").fillna((q_high - q_low).abs()).fillna(0.0).clip(lower=0.0)
    downside = pd.to_numeric(aligned.get("prob_downside"), errors="coerce").fillna((0.0 - q_low).clip(lower=0.0)).fillna(0.0).clip(lower=0.0)
    conf = pd.to_numeric(aligned.get("prob_confidence"), errors="coerce").fillna(1.0).clip(lower=0.0)

    penalty = float(cfg.probabilistic_interval_penalty_weight) * width + float(cfg.probabilistic_downside_penalty_weight) * downside
    prob_signal = (mid - penalty) * conf

    blend = float(getattr(cfg, "probabilistic_overlay_blend", 1.0) or 0.0)
    strength = float(getattr(cfg, "probabilistic_overlay_strength", 1.0) or 0.0)
    blend = float(np.clip(blend, 0.0, 1.0))
    strength = float(np.clip(strength, 0.0, 1.0))
    overlay_target = (1.0 - blend) * mu_hat + blend * prob_signal
    mu_used = mu_hat + strength * (overlay_target - mu_hat)
    delta = (mu_used - mu_hat).abs()

    def _num_mean(name: str, fallback: Optional[pd.Series] = None) -> float:
        if name in aligned.columns:
            return float(pd.to_numeric(aligned[name], errors="coerce").mean())
        if fallback is not None:
            return float(pd.to_numeric(fallback, errors="coerce").mean())
        return np.nan

    meta = {
        "probabilistic_active": True,
        "probabilistic_interval_width_mean": float(width.mean()),
        "probabilistic_downside_mean": float(downside.mean()),
        "probabilistic_confidence_mean": float(conf.mean()),
        "probabilistic_mu_penalty_mean_abs": float(delta.mean()),
        "probabilistic_mu_penalty_max_abs": float(delta.max()),
        "probabilistic_overlay_blend": float(blend),
        "probabilistic_overlay_strength": float(strength),
        "probabilistic_prob_signal_mean": float(prob_signal.mean()),
        "pred_low": _num_mean("pred_low", q_low),
        "pred_mid": _num_mean("pred_mid", q_mid),
        "pred_high": _num_mean("pred_high", q_high),
        "prob_q_low": _num_mean("prob_q_low", q_low),
        "prob_q_mid": _num_mean("prob_q_mid", q_mid),
        "prob_q_high": _num_mean("prob_q_high", q_high),
        "prob_pred_low_mean": _num_mean("pred_low", q_low),
        "prob_pred_mid_mean": _num_mean("pred_mid", q_mid),
        "prob_pred_high_mean": _num_mean("pred_high", q_high),
        "prob_q_low_mean": _num_mean("prob_q_low", q_low),
        "prob_q_mid_mean": _num_mean("prob_q_mid", q_mid),
        "prob_q_high_mean": _num_mean("prob_q_high", q_high),
        "prob_fallback_triggered": _num_mean("prob_fallback_triggered"),
        "prob_dispatch_path": "canonical_dispatch",
        "prob_dispatch_version": "v2",
    }

    def _col_mode(*names: str) -> Optional[str]:
        for name in names:
            if name in aligned.columns:
                vals = aligned[name].dropna().astype(str)
                if not vals.empty:
                    try:
                        return str(vals.mode().iloc[0])
                    except Exception:
                        return str(vals.iloc[-1])
        return None

    def _col_mean(name: str) -> float:
        if name not in aligned.columns:
            return np.nan
        return float(pd.to_numeric(aligned[name], errors="coerce").mean())

    meta.update({
        "probabilistic_mode_requested": _col_mode("probabilistic_mode_requested") or str(getattr(cfg, "probabilistic_mode", "none")),
        "probabilistic_mode_effective": _col_mode("probabilistic_mode_effective", "prob_source") or str(getattr(cfg, "probabilistic_mode", "none")),
        "prob_source": _col_mode("prob_source") or str(getattr(cfg, "probabilistic_mode", "none")),
        "prob_backend_family": _col_mode("prob_backend_family"),
        "prob_backend_name": _col_mode("prob_backend_name", "prob_backend_family"),
        "prob_backend_variant": _col_mode("prob_backend_variant", "prob_backend_name", "prob_backend_family"),
        "prob_backend_distinct": bool(round(_col_mean("prob_backend_distinct"))) if np.isfinite(_col_mean("prob_backend_distinct")) else False,
        "prob_interval_backend": _col_mode("prob_interval_backend"),
        "prob_contract_status": _col_mode("prob_contract_status") or "ok",
        "prob_contract_note": _col_mode("prob_contract_note") or "",
        "prob_backend_parent_mode": _col_mode("prob_backend_parent_mode") or str(getattr(cfg, "probabilistic_mode", "none")),
        "prob_fallback_triggered_share": _col_mean("prob_fallback_triggered"),
        "prob_fallback_reason": _col_mode("prob_fallback_reason") or "",
    })
    if "probabilistic_feature_match_n" in aligned.columns:
        meta["probabilistic_feature_match_n_mean"] = _col_mean("probabilistic_feature_match_n")
    if "probabilistic_feature_distance" in aligned.columns:
        meta["probabilistic_feature_distance_mean"] = _col_mean("probabilistic_feature_distance")
    if "prob_qr_active" in aligned.columns:
        meta["prob_qr_active_share"] = _col_mean("prob_qr_active")
    if "prob_hybrid_qr_weight" in aligned.columns:
        meta["prob_hybrid_qr_weight_mean"] = _col_mean("prob_hybrid_qr_weight")
    return mu_used, meta


def _force_psd_matrix(mat: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    arr = np.asarray(mat, dtype="float64")
    arr = (arr + arr.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(arr)
    eigvals = np.clip(eigvals, float(jitter), None)
    out = eigvecs @ np.diag(eigvals) @ eigvecs.T
    out = (out + out.T) / 2.0
    return out


def _resolve_covariance_regime_params(cfg: MicroPipelineConfig, regime: str) -> Dict[str, int]:
    if not bool(cfg.regime_dependent_covariance):
        return {
            "lookback": int(cfg.correlation_lookback),
            "halflife": int(cfg.ewma_halflife),
        }

    if regime == "high":
        return {
            "lookback": int(cfg.covariance_lookback_high),
            "halflife": int(cfg.covariance_halflife_high),
        }
    if regime == "low":
        return {
            "lookback": int(cfg.covariance_lookback_low),
            "halflife": int(cfg.covariance_halflife_low),
        }
    return {
        "lookback": int(cfg.covariance_lookback_mid),
        "halflife": int(cfg.covariance_halflife_mid),
    }


def _estimate_rolling_correlation_matrix(
    train_window: pd.DataFrame,
    cfg: MicroPipelineConfig,
    regime: str,
) -> pd.DataFrame:
    assets = list(train_window.columns)
    n_assets = len(assets)
    if n_assets == 0:
        raise ValueError("train_window must contain at least one asset")
    if n_assets == 1:
        return pd.DataFrame([[1.0]], index=assets, columns=assets, dtype="float64")

    params = _resolve_covariance_regime_params(cfg, regime)
    lookback = max(int(params["lookback"]), 2)
    min_periods = max(int(cfg.correlation_min_periods), 2)
    window = train_window.tail(lookback).copy()

    corr = window.corr(min_periods=min_periods).reindex(index=assets, columns=assets)
    corr = corr.fillna(0.0)
    np.fill_diagonal(corr.values, 1.0)
    corr = ((corr + corr.T) / 2.0).clip(lower=-0.999, upper=0.999)
    np.fill_diagonal(corr.values, 1.0)

    shrink = min(max(float(cfg.correlation_shrink_to_identity), 0.0), 1.0)
    if shrink > 0.0:
        ident = pd.DataFrame(np.eye(n_assets), index=assets, columns=assets, dtype="float64")
        corr = (1.0 - shrink) * corr + shrink * ident
        np.fill_diagonal(corr.values, 1.0)

    return corr.astype("float64")


def _summarise_correlation_matrix(corr: pd.DataFrame) -> Dict[str, float]:
    arr = corr.to_numpy(dtype="float64")
    n = arr.shape[0]
    if n <= 1:
        return {
            "avg_corr": 0.0,
            "avg_abs_corr": 0.0,
            "max_abs_corr": 0.0,
            "effective_breadth_proxy": 1.0,
        }

    mask = ~np.eye(n, dtype=bool)
    off = arr[mask]
    avg_corr = float(np.mean(off)) if off.size else 0.0
    avg_abs_corr = float(np.mean(np.abs(off))) if off.size else 0.0
    max_abs_corr = float(np.max(np.abs(off))) if off.size else 0.0
    denom = 1.0 + max(n - 1, 0) * max(avg_abs_corr, 0.0)
    effective_breadth_proxy = float(n / denom) if denom > 0 else 1.0
    return {
        "avg_corr": avg_corr,
        "avg_abs_corr": avg_abs_corr,
        "max_abs_corr": max_abs_corr,
        "effective_breadth_proxy": effective_breadth_proxy,
    }


def _estimate_ewma_covariance_matrix(window_df: pd.DataFrame, halflife: int) -> pd.DataFrame:
    assets = list(window_df.columns)
    x = window_df.fillna(0.0).to_numpy(dtype="float64")
    n = x.shape[0]
    if n == 0:
        return pd.DataFrame(np.eye(len(assets)), index=assets, columns=assets, dtype="float64")
    hl = max(int(halflife), 1)
    lam = np.exp(np.log(0.5) / hl)
    raw_w = lam ** np.arange(n - 1, -1, -1, dtype="float64")
    raw_w = raw_w / raw_w.sum()
    mean = np.average(x, axis=0, weights=raw_w)
    xc = x - mean
    cov = (xc * raw_w[:, None]).T @ xc
    return pd.DataFrame(cov, index=assets, columns=assets, dtype="float64")



def _covariance_to_correlation(cov: pd.DataFrame) -> pd.DataFrame:
    if cov is None or not isinstance(cov, pd.DataFrame) or cov.empty:
        return pd.DataFrame()
    assets = list(cov.index)
    arr = cov.reindex(index=assets, columns=assets).fillna(0.0).to_numpy(dtype="float64")
    diag = np.clip(np.diag(arr), 0.0, None)
    vol = np.sqrt(diag)
    denom = np.outer(vol, vol)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.divide(arr, denom, out=np.zeros_like(arr), where=denom > 0)
    corr = np.clip((corr + corr.T) / 2.0, -0.999, 0.999)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=assets, columns=assets, dtype="float64")


def _estimate_statistical_factor_model(
    train_window: pd.DataFrame,
    *,
    n_factors: int,
    min_obs: int,
    shrink_to_diagonal: float,
    jitter: float,
) -> Dict[str, Any]:
    assets = list(train_window.columns)
    n_assets = len(assets)
    if n_assets == 0:
        return {"factor_model_active": False}

    req_obs = max(int(min_obs), int(n_factors) + 2, 6)
    window = train_window.tail(max(req_obs, int(min_obs))).copy()
    window = window.loc[:, assets].fillna(0.0)
    n_obs = int(window.shape[0])
    if n_obs < req_obs or n_assets < 2:
        return {
            "factor_model_active": False,
            "factor_model_reason": "insufficient_history",
            "factor_model_n_obs": n_obs,
            "factor_model_n_assets": n_assets,
            "factor_model_n_factors": 0,
        }

    x = window.to_numpy(dtype="float64")
    mean_vec = x.mean(axis=0, keepdims=True)
    xc = x - mean_vec
    sample_cov = np.cov(xc, rowvar=False, ddof=1)
    if sample_cov.ndim == 0:
        sample_cov = np.array([[float(sample_cov)]], dtype="float64")
    sample_cov = np.asarray(sample_cov, dtype="float64")
    sample_cov = (sample_cov + sample_cov.T) / 2.0

    evals, evecs = np.linalg.eigh(sample_cov)
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], 0.0, None)
    evecs = evecs[:, order]
    k = int(min(max(int(n_factors), 1), n_assets, n_obs - 1))
    if k <= 0:
        return {"factor_model_active": False, "factor_model_reason": "no_factors"}

    top_vals = evals[:k]
    top_vecs = evecs[:, :k]
    common_cov = top_vecs @ np.diag(top_vals) @ top_vecs.T
    residual_diag = np.clip(np.diag(sample_cov - common_cov), float(jitter), None)
    residual_cov = np.diag(residual_diag)
    total_cov = common_cov + residual_cov

    shrink = min(max(float(shrink_to_diagonal), 0.0), 1.0)
    if shrink > 0.0:
        total_cov = (1.0 - shrink) * total_cov + shrink * np.diag(np.diag(total_cov))

    total_cov = _force_psd_matrix(total_cov, jitter=float(jitter))
    common_cov = _force_psd_matrix(common_cov, jitter=float(jitter))

    loadings = top_vecs * np.sqrt(np.clip(top_vals, float(jitter), None))[None, :]
    factor_returns = xc @ top_vecs
    factor_premia = factor_returns.mean(axis=0)
    common_mu = top_vecs @ factor_premia
    factor_cov = np.cov(factor_returns, rowvar=False, ddof=1)
    if np.ndim(factor_cov) == 0:
        factor_cov = np.array([[float(factor_cov)]], dtype="float64")
    factor_cov = _force_psd_matrix(np.asarray(factor_cov, dtype="float64"), jitter=float(jitter))

    total_var = float(np.trace(sample_cov))
    common_var = float(np.trace(common_cov))
    explained_share = common_var / total_var if total_var > float(jitter) else np.nan

    return {
        "factor_model_active": True,
        "factor_model_method": "pca_statistical",
        "factor_model_n_obs": n_obs,
        "factor_model_n_assets": n_assets,
        "factor_model_n_factors": k,
        "factor_model_explained_variance_share": float(explained_share) if np.isfinite(explained_share) else np.nan,
        "factor_model_residual_variance_share": float(1.0 - explained_share) if np.isfinite(explained_share) else np.nan,
        "factor_model_factor_strength": float(np.sqrt(np.clip(top_vals, 0.0, None)).sum()),
        "loadings": pd.DataFrame(loadings, index=assets, columns=[f"factor_{i+1}" for i in range(k)], dtype="float64"),
        "factor_returns": pd.DataFrame(factor_returns, index=window.index, columns=[f"factor_{i+1}" for i in range(k)], dtype="float64"),
        "factor_cov": pd.DataFrame(factor_cov, index=[f"factor_{i+1}" for i in range(k)], columns=[f"factor_{i+1}" for i in range(k)], dtype="float64"),
        "common_cov": pd.DataFrame(common_cov, index=assets, columns=assets, dtype="float64"),
        "residual_cov": pd.DataFrame(residual_cov, index=assets, columns=assets, dtype="float64"),
        "total_cov": pd.DataFrame(total_cov, index=assets, columns=assets, dtype="float64"),
        "common_mu": pd.Series(common_mu, index=assets, dtype="float64"),
    }


def _build_industrial_factor_payload(
    train_window: pd.DataFrame,
    cfg: MicroPipelineConfig,
) -> Dict[str, Any]:
    factor_active = bool(getattr(cfg, "factor_model_active", False)) or bool(getattr(cfg, "factor_covariance_active", False))
    if not factor_active:
        return {"factor_model_active": False, "factor_covariance_active": False, "factor_overlay_active": False}

    n_factors = max(int(getattr(cfg, "factor_model_n_factors", getattr(cfg, "factor_covariance_n_factors", 3)) or 3), 1)
    n_factors = min(n_factors, max(int(train_window.shape[1]), 1))
    min_obs = max(
        int(getattr(cfg, "factor_model_min_obs", 24) or 24),
        int(getattr(cfg, "factor_covariance_min_obs", 24) or 24),
    )
    shrink = float(_first_finite_optional(
        getattr(cfg, "factor_model_shrink_to_diagonal", None),
        getattr(cfg, "factor_covariance_shrink_to_diagonal", None),
        getattr(cfg, "covariance_shrink_to_diagonal", 0.0),
    ) or 0.0)
    payload = _estimate_statistical_factor_model(
        train_window,
        n_factors=n_factors,
        min_obs=min_obs,
        shrink_to_diagonal=shrink,
        jitter=float(getattr(cfg, "covariance_jitter", 1e-8) or 1e-8),
    )
    payload["factor_covariance_active"] = bool(getattr(cfg, "factor_covariance_active", False))
    payload["factor_overlay_active"] = bool(payload.get("factor_model_active", False))
    return payload


def _apply_industrial_factor_overlay(
    *,
    base_mu_hat: pd.Series,
    base_sigma_fwd: pd.DataFrame,
    base_corr_mat: pd.DataFrame,
    assets: List[str],
    factor_payload: Optional[Dict[str, Any]],
    cfg: MicroPipelineConfig,
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    base_mu = pd.to_numeric(base_mu_hat.reindex(assets), errors="coerce").fillna(0.0).astype(float)
    sigma_used = base_sigma_fwd.reindex(index=assets, columns=assets).fillna(0.0).astype(float)
    corr_used = base_corr_mat.reindex(index=assets, columns=assets).fillna(0.0).astype(float)
    if factor_payload is None or not bool(factor_payload.get("factor_model_active", False)):
        return base_mu, sigma_used, corr_used, {
            "factor_model_active": False,
            "factor_covariance_active": bool(getattr(cfg, "factor_covariance_active", False)),
            "factor_overlay_active": False,
            "factor_model_n_factors": 0,
            "factor_model_n_obs": np.nan,
            "factor_model_explained_variance_share": np.nan,
            "factor_mu_blend_used": 0.0,
            "factor_covariance_blend_used": 0.0,
            "factor_residual_blend_used": 0.0,
            "factor_mu_abs_tilt_mean": 0.0,
            "factor_mu_abs_tilt_max": 0.0,
            "factor_covariance_trace_ratio": 1.0,
        }

    common_mu = pd.to_numeric(factor_payload.get("common_mu", pd.Series(index=assets, dtype="float64")).reindex(assets), errors="coerce").fillna(0.0).astype(float)
    common_cov = factor_payload.get("common_cov", pd.DataFrame(index=assets, columns=assets))
    residual_cov = factor_payload.get("residual_cov", pd.DataFrame(index=assets, columns=assets))
    total_cov = factor_payload.get("total_cov", pd.DataFrame(index=assets, columns=assets))
    common_cov = common_cov.reindex(index=assets, columns=assets).fillna(0.0).astype(float)
    residual_cov = residual_cov.reindex(index=assets, columns=assets).fillna(0.0).astype(float)
    total_cov = total_cov.reindex(index=assets, columns=assets).fillna(0.0).astype(float)

    mu_blend = min(max(float(getattr(cfg, "factor_model_mu_blend", 0.0) or 0.0), 0.0), 1.0) if bool(getattr(cfg, "factor_model_active", False)) else 0.0
    residual_blend = min(max(float(getattr(cfg, "factor_model_residual_blend", 1.0) or 1.0), 0.0), 1.0)
    covariance_blend = 0.0
    if bool(getattr(cfg, "factor_model_active", False)):
        covariance_blend = min(max(float(getattr(cfg, "factor_model_covariance_blend", 0.0) or 0.0), 0.0), 1.0)
    elif bool(getattr(cfg, "factor_covariance_active", False)):
        covariance_blend = min(max(float(getattr(cfg, "factor_covariance_blend", 0.0) or 0.0), 0.0), 1.0)

    industrial_mu = common_mu + residual_blend * (base_mu - common_mu)
    mu_used = (1.0 - mu_blend) * base_mu + mu_blend * industrial_mu

    industrial_cov = common_cov + residual_blend * residual_cov
    industrial_cov = (industrial_cov + industrial_cov.T) / 2.0
    industrial_cov = _force_psd_matrix(industrial_cov.to_numpy(dtype="float64"), jitter=float(getattr(cfg, "covariance_jitter", 1e-8) or 1e-8))
    industrial_cov = pd.DataFrame(industrial_cov, index=assets, columns=assets, dtype="float64")
    sigma_used = ((1.0 - covariance_blend) * sigma_used + covariance_blend * industrial_cov).astype(float)
    sigma_used = pd.DataFrame(
        _force_psd_matrix(sigma_used.to_numpy(dtype="float64"), jitter=float(getattr(cfg, "covariance_jitter", 1e-8) or 1e-8)),
        index=assets,
        columns=assets,
        dtype="float64",
    )
    corr_used = _covariance_to_correlation(sigma_used) if covariance_blend > 0.0 else corr_used

    delta_mu = (mu_used - base_mu).abs()
    trace_base = float(np.trace(base_sigma_fwd.reindex(index=assets, columns=assets).fillna(0.0).to_numpy(dtype="float64")))
    trace_used = float(np.trace(sigma_used.to_numpy(dtype="float64")))
    return mu_used, sigma_used, corr_used, {
        "factor_model_active": bool(getattr(cfg, "factor_model_active", False)),
        "factor_covariance_active": bool(getattr(cfg, "factor_covariance_active", False)),
        "factor_overlay_active": True,
        "factor_model_method": str(factor_payload.get("factor_model_method", "pca_statistical")),
        "factor_model_n_factors": int(factor_payload.get("factor_model_n_factors", 0) or 0),
        "factor_model_n_obs": float(factor_payload.get("factor_model_n_obs", np.nan)),
        "factor_model_explained_variance_share": float(factor_payload.get("factor_model_explained_variance_share", np.nan)),
        "factor_model_residual_variance_share": float(factor_payload.get("factor_model_residual_variance_share", np.nan)),
        "factor_model_factor_strength": float(factor_payload.get("factor_model_factor_strength", np.nan)),
        "factor_mu_blend_used": float(mu_blend),
        "factor_covariance_blend_used": float(covariance_blend),
        "factor_residual_blend_used": float(residual_blend),
        "factor_mu_abs_tilt_mean": float(delta_mu.mean()) if not delta_mu.empty else 0.0,
        "factor_mu_abs_tilt_max": float(delta_mu.max()) if not delta_mu.empty else 0.0,
        "factor_covariance_trace_ratio": float(trace_used / trace_base) if np.isfinite(trace_base) and trace_base > 1e-12 else np.nan,
        "factor_common_mu_mean": float(common_mu.mean()) if not common_mu.empty else np.nan,
        "factor_common_mu_std": float(common_mu.std(ddof=1)) if common_mu.shape[0] > 1 else np.nan,
        "factor_common_cov_trace": float(np.trace(common_cov.to_numpy(dtype="float64"))) if not common_cov.empty else np.nan,
        "factor_residual_cov_trace": float(np.trace(residual_cov.to_numpy(dtype="float64"))) if not residual_cov.empty else np.nan,
        "factor_total_cov_trace": float(np.trace(total_cov.to_numpy(dtype="float64"))) if not total_cov.empty else np.nan,
    }


def _build_sigma_fwd(
    train_window: pd.DataFrame,
    sigma_hat: pd.Series,
    corr_mat: pd.DataFrame,
    cfg: MicroPipelineConfig,
    regime: str,
) -> pd.DataFrame:
    assets = list(train_window.columns)
    aligned_sigma = sigma_hat.reindex(assets).fillna(float(cfg.sigma_floor)).clip(lower=float(cfg.sigma_floor))

    if str(cfg.covariance_mode) == "ewma_cov":
        params = _resolve_covariance_regime_params(cfg, regime)
        sigma_fwd = _estimate_ewma_covariance_matrix(train_window.tail(max(int(params["lookback"]), 6)), halflife=int(params["halflife"]))
        sigma_fwd = sigma_fwd.reindex(index=assets, columns=assets).fillna(0.0)
        # blend toward corr*sigma structure so diagonal matches current sigma_hat more closely
        d = np.diag(aligned_sigma.to_numpy(dtype="float64"))
        corr_cov = pd.DataFrame(d @ corr_mat.reindex(index=assets, columns=assets).fillna(0.0).to_numpy(dtype="float64") @ d, index=assets, columns=assets)
        sigma_fwd = 0.5 * sigma_fwd + 0.5 * corr_cov
    else:
        d = np.diag(aligned_sigma.to_numpy(dtype="float64"))
        corr = corr_mat.reindex(index=assets, columns=assets).fillna(0.0).to_numpy(dtype="float64")
        np.fill_diagonal(corr, 1.0)
        sigma_fwd = pd.DataFrame(d @ corr @ d, index=assets, columns=assets)

    shrink = min(max(float(cfg.covariance_shrink_to_diagonal), 0.0), 1.0)
    if shrink > 0.0:
        arr = sigma_fwd.to_numpy(dtype="float64")
        diag = np.diag(np.diag(arr))
        arr = (1.0 - shrink) * arr + shrink * diag
        sigma_fwd = pd.DataFrame(arr, index=assets, columns=assets)

    sigma_psd = _force_psd_matrix(sigma_fwd.to_numpy(dtype="float64"), jitter=float(cfg.covariance_jitter))
    sigma_fwd = pd.DataFrame(sigma_psd, index=assets, columns=assets, dtype="float64")
    return sigma_fwd


def _summarise_sigma_fwd(sigma_fwd: pd.DataFrame) -> Dict[str, float]:
    arr = sigma_fwd.to_numpy(dtype="float64")
    n = arr.shape[0]
    if n == 0:
        return {
            "avg_variance": np.nan,
            "avg_cov_offdiag": np.nan,
            "max_cov_offdiag": np.nan,
            "sigma_trace": np.nan,
        }
    diag = np.diag(arr)
    mask = ~np.eye(n, dtype=bool)
    off = arr[mask]
    return {
        "avg_variance": float(np.mean(diag)) if diag.size else np.nan,
        "avg_cov_offdiag": float(np.mean(off)) if off.size else np.nan,
        "max_cov_offdiag": float(np.max(np.abs(off))) if off.size else np.nan,
        "sigma_trace": float(np.trace(arr)),
    }


def _estimate_portfolio_vol_monthly(weights: pd.Series, sigma_hat: pd.Series, sigma_fwd: Optional[pd.DataFrame]) -> float:
    assets = list(weights.index)
    if len(assets) == 0:
        return 0.0
    if sigma_fwd is not None and not sigma_fwd.empty:
        mat = sigma_fwd.reindex(index=assets, columns=assets).fillna(0.0).to_numpy(dtype="float64")
    else:
        sig = sigma_hat.reindex(assets).fillna(0.0).clip(lower=0.0).to_numpy(dtype="float64")
        mat = np.diag(np.square(sig))
    wv = weights.to_numpy(dtype="float64")
    var = float(wv.T @ mat @ wv)
    return float(np.sqrt(max(var, 0.0)))


def _apply_vol_targeting(weights: pd.Series, sigma_hat: pd.Series, sigma_fwd: Optional[pd.DataFrame], cfg: MicroPipelineConfig) -> Tuple[pd.Series, Dict[str, float]]:
    pre_vol = _estimate_portfolio_vol_monthly(weights, sigma_hat, sigma_fwd if bool(cfg.covariance_aware_vol_targeting) else None)
    if not bool(cfg.vol_targeting):
        return weights, {
            "pre_target_vol": float(pre_vol),
            "post_target_vol": float(pre_vol),
            "vol_target_mult": 1.0,
            "vol_target_active": False,
        }

    target = max(float(cfg.target_portfolio_vol_monthly), 1e-8)
    raw_mult = target / max(pre_vol, 1e-8)
    vol_mult = min(max(raw_mult, float(cfg.vol_target_floor_mult)), float(cfg.vol_target_ceiling_mult))
    active = abs(vol_mult - 1.0) > 1e-12
    if not active:
        return weights, {
            "pre_target_vol": float(pre_vol),
            "post_target_vol": float(pre_vol),
            "vol_target_mult": 1.0,
            "vol_target_active": False,
        }

    n = len(weights)
    ew = pd.Series(np.ones(n) / n, index=weights.index)
    out = ew + vol_mult * (weights - ew)
    if cfg.long_only:
        out = out.clip(lower=0.0)
        s = out.sum()
        out = out / s if np.isfinite(s) and s > 0 else ew
    else:
        denom = np.abs(out).sum()
        out = out / denom if np.isfinite(denom) and denom > 0 else weights

    post_vol = _estimate_portfolio_vol_monthly(out, sigma_hat, sigma_fwd if bool(cfg.covariance_aware_vol_targeting) else None)
    return out, {
        "pre_target_vol": float(pre_vol),
        "post_target_vol": float(post_vol),
        "vol_target_mult": float(vol_mult),
        "vol_target_active": True,
    }


def _compute_diversification_diagnostics(train_window: pd.DataFrame, weights: pd.Series, corr_mat: Optional[pd.DataFrame] = None, sigma_fwd: Optional[pd.DataFrame] = None, lookback: int = 24) -> Dict[str, float]:
    if train_window is None or train_window.empty or len(weights) == 0:
        return {
            "weight_herfindahl": np.nan,
            "effective_n_assets": np.nan,
            "diversification_ratio": np.nan,
            "avg_pairwise_corr": np.nan,
            "weighted_avg_corr": np.nan,
            "corr_effective_rank": np.nan,
            "effective_breadth_universe": np.nan,
            "effective_breadth": np.nan,
            "breadth_utilization_ratio": np.nan,
        }
    assets = list(weights.index)
    use_window = train_window.tail(max(int(lookback), 6)).reindex(columns=assets)
    corr = corr_mat.reindex(index=assets, columns=assets).fillna(0.0) if corr_mat is not None else use_window.corr().reindex(index=assets, columns=assets).fillna(0.0)
    cov = sigma_fwd.reindex(index=assets, columns=assets).fillna(0.0) if sigma_fwd is not None else use_window.cov().reindex(index=assets, columns=assets).fillna(0.0)

    w = weights.reindex(assets).fillna(0.0).to_numpy(dtype="float64")
    n = len(w)
    weight_herfindahl = float(np.sum(np.square(w)))
    effective_n_assets = float(1.0 / weight_herfindahl) if weight_herfindahl > 0 else np.nan
    cov_arr = cov.to_numpy(dtype="float64")
    standalone_vols = np.sqrt(np.clip(np.diag(cov_arr), 0.0, None))
    weighted_standalone = float(np.sum(np.abs(w) * standalone_vols))
    port_var = max(float(w.T @ cov_arr @ w), 0.0)
    port_vol = float(np.sqrt(port_var))
    diversification_ratio = float(weighted_standalone / port_vol) if port_vol > 0 else np.nan

    corr_arr = corr.to_numpy(dtype="float64")
    if n > 1:
        mask = ~np.eye(n, dtype=bool)
        off_corr = corr_arr[mask]
        avg_pairwise_corr = float(np.mean(off_corr)) if off_corr.size else np.nan
        outer = np.outer(np.abs(w), np.abs(w))
        weighted_avg_corr = float(np.sum(corr_arr[mask] * outer[mask]) / np.sum(outer[mask])) if np.sum(outer[mask]) > 0 else np.nan
        eigvals = np.linalg.eigvalsh((corr_arr + corr_arr.T) / 2.0)
        eigvals = np.clip(eigvals, 0.0, None)
        eig_sum = float(np.sum(eigvals))
        corr_effective_rank = float((eig_sum ** 2) / np.sum(np.square(eigvals))) if eig_sum > 0 and np.sum(np.square(eigvals)) > 0 else np.nan
    else:
        avg_pairwise_corr = np.nan
        weighted_avg_corr = np.nan
        corr_effective_rank = 1.0

    effective_breadth_universe = float(corr_effective_rank) if np.isfinite(corr_effective_rank) else np.nan
    effective_breadth = float(min(effective_n_assets, effective_breadth_universe)) if np.isfinite(effective_n_assets) and np.isfinite(effective_breadth_universe) else np.nan
    breadth_utilization_ratio = float(effective_breadth / effective_breadth_universe) if np.isfinite(effective_breadth) and np.isfinite(effective_breadth_universe) and effective_breadth_universe > 0 else np.nan

    return {
        "weight_herfindahl": weight_herfindahl,
        "effective_n_assets": effective_n_assets,
        "diversification_ratio": diversification_ratio,
        "avg_pairwise_corr": avg_pairwise_corr,
        "weighted_avg_corr": weighted_avg_corr,
        "corr_effective_rank": corr_effective_rank,
        "effective_breadth_universe": effective_breadth_universe,
        "effective_breadth": effective_breadth,
        "breadth_utilization_ratio": breadth_utilization_ratio,
    }


def _compute_risk_decomposition(weights: pd.Series, sigma_fwd: Optional[pd.DataFrame], corr_mat: Optional[pd.DataFrame], cfg: MicroPipelineConfig) -> Tuple[pd.DataFrame, Dict[str, float]]:
    assets = list(weights.index)
    if len(assets) == 0 or sigma_fwd is None or sigma_fwd.empty:
        empty = pd.DataFrame(index=assets)
        return empty, {
            "portfolio_vol_from_cov": np.nan,
            "top_risk_contributor_share": np.nan,
            "effective_risk_bets": np.nan,
            "risk_concentration_herfindahl": np.nan,
            "cluster_high_corr_share": np.nan,
            "avg_abs_corr_top_weights": np.nan,
        }
    sigma = sigma_fwd.reindex(index=assets, columns=assets).fillna(0.0).to_numpy(dtype="float64")
    w = weights.reindex(assets).fillna(0.0).to_numpy(dtype="float64")
    port_vol = float(np.sqrt(max(float(w.T @ sigma @ w), 0.0)))
    if port_vol <= 1e-12:
        empty = pd.DataFrame(index=assets)
        return empty, {
            "portfolio_vol_from_cov": 0.0,
            "top_risk_contributor_share": np.nan,
            "effective_risk_bets": np.nan,
            "risk_concentration_herfindahl": np.nan,
            "cluster_high_corr_share": np.nan,
            "avg_abs_corr_top_weights": np.nan,
        }
    mrc = sigma @ w / port_vol
    rc = w * mrc
    rc_share = rc / max(port_vol, 1e-12)
    rc_share_abs = np.abs(rc_share)
    herf = float(np.sum(np.square(rc_share_abs))) if rc_share_abs.size else np.nan
    effective_risk_bets = float(1.0 / herf) if np.isfinite(herf) and herf > 0 else np.nan
    top_share = float(np.max(rc_share_abs)) if rc_share_abs.size else np.nan

    cluster_high_corr_share = np.nan
    avg_abs_corr_top_weights = np.nan
    if corr_mat is not None and not corr_mat.empty and len(assets) > 1:
        corr = corr_mat.reindex(index=assets, columns=assets).fillna(0.0).to_numpy(dtype="float64")
        mask = ~np.eye(len(assets), dtype=bool)
        if mask.sum() > 0:
            avg_abs_corr_top_weights = float(np.mean(np.abs(corr[mask])))
            cluster_high_corr_share = float(np.mean(np.abs(corr[mask]) >= float(cfg.cluster_corr_threshold)))

    details = pd.DataFrame({
        "asset": assets,
        "marginal_risk_contribution": mrc,
        "contribution_to_variance": rc,
        "risk_contribution_share": rc_share,
    }).set_index("asset")
    summary = {
        "portfolio_vol_from_cov": port_vol,
        "top_risk_contributor_share": top_share,
        "effective_risk_bets": effective_risk_bets,
        "risk_concentration_herfindahl": herf,
        "cluster_high_corr_share": cluster_high_corr_share,
        "avg_abs_corr_top_weights": avg_abs_corr_top_weights,
    }
    return details, summary


def _apply_correlation_aware_allocator(
    base_weights: pd.Series,
    score: pd.Series,
    sigma_fwd: Optional[pd.DataFrame],
    corr_mat: Optional[pd.DataFrame],
    cfg: MicroPipelineConfig,
) -> Tuple[pd.Series, Dict[str, Any]]:
    if (not bool(cfg.correlation_aware_allocation)) or sigma_fwd is None or sigma_fwd.empty:
        return base_weights, {"correlation_allocator_active": False, "correlation_allocator_method": "none", "correlation_allocator_blend": 0.0}

    assets = list(base_weights.index)
    sigma = sigma_fwd.reindex(index=assets, columns=assets).fillna(0.0).to_numpy(dtype="float64")
    sigma = _force_psd_matrix(sigma, jitter=float(cfg.covariance_jitter))
    score_vec = score.reindex(assets).fillna(0.0).to_numpy(dtype="float64")
    method = str(cfg.correlation_allocator_method)

    alt = None
    if method == "mean_variance_light":
        lam = max(float(cfg.mean_variance_risk_aversion), 1e-6)
        ridge = np.eye(len(assets)) * (float(cfg.covariance_jitter) + 1e-6)
        rhs = np.clip(score_vec, 0.0, None) if bool(cfg.long_only) else score_vec
        try:
            alt_vec = np.linalg.solve(lam * sigma + ridge, rhs)
        except np.linalg.LinAlgError:
            alt_vec = np.linalg.pinv(lam * sigma + ridge) @ rhs
        if bool(cfg.long_only):
            alt_vec = np.clip(alt_vec, 0.0, None)
            denom = alt_vec.sum()
            alt = pd.Series(np.ones(len(assets)) / len(assets), index=assets) if denom <= 0 or not np.isfinite(denom) else pd.Series(alt_vec / denom, index=assets)
        else:
            denom = np.abs(alt_vec).sum()
            alt = pd.Series(base_weights.values, index=assets) if denom <= 0 or not np.isfinite(denom) else pd.Series(alt_vec / denom, index=assets)
    elif method == "risk_budget":
        diag = np.sqrt(np.clip(np.diag(sigma), 1e-12, None))
        inv_risk = 1.0 / diag
        pos_score = np.clip(score_vec, 0.0, None)
        burden = np.ones_like(pos_score)
        if corr_mat is not None and not corr_mat.empty:
            corr = corr_mat.reindex(index=assets, columns=assets).fillna(0.0).to_numpy(dtype="float64")
            burden = 1.0 + float(cfg.risk_budget_strength) * np.mean(np.abs(corr), axis=1)
        alt_vec = pos_score * inv_risk / burden
        denom = alt_vec.sum()
        alt = pd.Series(np.ones(len(assets)) / len(assets), index=assets) if denom <= 0 or not np.isfinite(denom) else pd.Series(alt_vec / denom, index=assets)
    else:  # score_penalty
        if corr_mat is None or corr_mat.empty:
            return base_weights, {"correlation_allocator_active": False, "correlation_allocator_method": method, "correlation_allocator_blend": 0.0}
        corr = corr_mat.reindex(index=assets, columns=assets).fillna(0.0).to_numpy(dtype="float64")
        burden_raw = np.mean(np.abs(corr) if bool(cfg.correlation_use_abs) else np.clip(corr, 0.0, None), axis=1)
        burden = 1.0 + float(cfg.correlation_penalty_strength) * np.power(np.clip(burden_raw, 0.0, None), float(cfg.correlation_penalty_power))
        adj = score_vec / burden
        if bool(cfg.long_only):
            adj = np.clip(adj, 0.0, None)
            denom = adj.sum()
            alt = pd.Series(np.ones(len(assets)) / len(assets), index=assets) if denom <= 0 or not np.isfinite(denom) else pd.Series(adj / denom, index=assets)
        else:
            denom = np.abs(adj).sum()
            alt = pd.Series(base_weights.values, index=assets) if denom <= 0 or not np.isfinite(denom) else pd.Series(adj / denom, index=assets)

    blend = min(max(float(cfg.correlation_allocator_blend), 0.0), 1.0)
    out = (1.0 - blend) * base_weights.reindex(assets).fillna(0.0) + blend * alt.reindex(assets).fillna(0.0)
    if bool(cfg.long_only):
        out = out.clip(lower=0.0)
        s = out.sum()
        out = out / s if np.isfinite(s) and s > 0 else pd.Series(np.ones(len(assets))/len(assets), index=assets)
    else:
        denom = np.abs(out).sum()
        out = out / denom if np.isfinite(denom) and denom > 0 else base_weights
    return out, {
        "correlation_allocator_active": True,
        "correlation_allocator_method": method,
        "correlation_allocator_blend": blend,
    }


def _classify_regime(train_window: pd.DataFrame, cfg: MicroPipelineConfig) -> str:
    if cfg.regime_mode == "none":
        return "pooled"

    market_proxy = train_window.mean(axis=1).dropna()
    if len(market_proxy) < max(cfg.regime_lookback, 6):
        return "pooled"

    recent = market_proxy.tail(int(cfg.regime_lookback))
    vol_now = float(recent.std(ddof=1)) if len(recent) > 1 else 0.0
    rolling_ref = market_proxy.rolling(int(cfg.regime_lookback)).std(ddof=1).dropna()
    if len(rolling_ref) < 10:
        return "pooled"

    q_low = float(rolling_ref.quantile(float(cfg.regime_quantile_low)))
    q_high = float(rolling_ref.quantile(float(cfg.regime_quantile_high)))

    if vol_now <= q_low:
        return "low"
    if vol_now >= q_high:
        return "high"
    return "mid"




def _apply_dispersion_to_sigma_hat(
    train_window: pd.DataFrame,
    sigma_hat: pd.Series,
    cfg: MicroPipelineConfig,
) -> tuple[pd.Series, Dict[str, float]]:
    """
    Use recent cross-sectional return dispersion to scale the sigma forecast.

    Intuition:
    - when cross-sectional dispersion is unusually high, increase sigma_hat modestly
    - when unusually low, allow a mild reduction
    """
    meta = {
        "dispersion_sigma_active": False,
        "dispersion_sigma_recent": np.nan,
        "dispersion_sigma_hist_mean": np.nan,
        "dispersion_sigma_hist_std": np.nan,
        "dispersion_sigma_z": np.nan,
        "dispersion_sigma_mult": 1.0,
    }

    if not bool(getattr(cfg, "dispersion_sigma_enabled", False)):
        return sigma_hat, meta

    if train_window is None or train_window.empty:
        return sigma_hat, meta

    cs_disp = train_window.std(axis=1, ddof=1).dropna()
    lookback = max(int(getattr(cfg, "dispersion_sigma_lookback", 12)), 3)
    if cs_disp.shape[0] < max(lookback, 6):
        return sigma_hat, meta

    recent = float(cs_disp.tail(lookback).mean())
    hist = cs_disp.iloc[:-1] if cs_disp.shape[0] > 1 else cs_disp
    hist_mean = float(hist.mean()) if not hist.empty else np.nan
    hist_std = float(hist.std(ddof=1)) if hist.shape[0] > 1 else 0.0

    if not np.isfinite(hist_mean):
        return sigma_hat, meta

    if not np.isfinite(hist_std) or hist_std <= 1e-12:
        z = 0.0
    else:
        z = (recent - hist_mean) / hist_std

    strength = float(max(getattr(cfg, "dispersion_sigma_strength", 0.50), 0.0))
    mult = 1.0 + strength * 0.10 * float(z)
    mult = float(np.clip(
        mult,
        float(getattr(cfg, "dispersion_sigma_floor_mult", 0.75)),
        float(getattr(cfg, "dispersion_sigma_ceiling_mult", 1.50)),
    ))

    out = pd.to_numeric(sigma_hat, errors="coerce").fillna(float(cfg.sigma_floor)).clip(lower=float(cfg.sigma_floor))
    out = out * mult
    out = out.fillna(float(cfg.sigma_floor)).clip(lower=float(cfg.sigma_floor))

    meta.update({
        "dispersion_sigma_active": True,
        "dispersion_sigma_recent": recent,
        "dispersion_sigma_hist_mean": hist_mean,
        "dispersion_sigma_hist_std": hist_std,
        "dispersion_sigma_z": float(z),
        "dispersion_sigma_mult": mult,
    })
    return out.astype(float), meta

def _forecast_mu_sigma_from_window(
    window_df: pd.DataFrame,
    cfg: MicroPipelineConfig,
) -> Tuple[pd.Series, pd.Series, str]:
    regime = _classify_regime(window_df, cfg)

    mu_window = window_df.tail(int(cfg.lookback_mu))
    if cfg.mu_regime_mode == "split" and cfg.regime_mode == "quantile":
        # simple regime split: filter months by market-vol regime label approximation
        market_proxy = window_df.mean(axis=1).dropna()
        rolling_vol = market_proxy.rolling(int(cfg.regime_lookback)).std(ddof=1)
        if len(rolling_vol.dropna()) >= 10:
            q_low = float(rolling_vol.dropna().quantile(float(cfg.regime_quantile_low)))
            q_high = float(rolling_vol.dropna().quantile(float(cfg.regime_quantile_high)))
            if regime == "low":
                mask = rolling_vol <= q_low
            elif regime == "high":
                mask = rolling_vol >= q_high
            else:
                mask = (rolling_vol > q_low) & (rolling_vol < q_high)
            filtered_idx = rolling_vol.index[mask.fillna(False)]
            filtered = window_df.loc[window_df.index.intersection(filtered_idx)]
            if len(filtered) >= max(6, int(cfg.lookback_mu // 2)):
                mu_window = filtered.tail(int(cfg.lookback_mu))

    mu_hat = mu_window.mean(axis=0)

    sigma_window = window_df.tail(int(cfg.lookback_sigma))
    if cfg.ewma_sigma:
        sigma_hat = _ewma_sigma(sigma_window.fillna(0.0), halflife=int(cfg.ewma_halflife))
    else:
        sigma_hat = sigma_window.std(axis=0, ddof=1)

    sigma_hat = sigma_hat.fillna(float(cfg.sigma_floor)).clip(lower=float(cfg.sigma_floor))
    sigma_hat, _dispersion_sigma_meta = _apply_dispersion_to_sigma_hat(window_df, sigma_hat, cfg)
    return mu_hat, sigma_hat, regime


def _apply_deadband(new_w: pd.Series, prev_w: Optional[pd.Series], threshold: float) -> pd.Series:
    if prev_w is None:
        return new_w
    aligned_prev = prev_w.reindex(new_w.index).fillna(0.0)
    diff = (new_w - aligned_prev).abs()
    out = new_w.copy()
    out.loc[diff < float(threshold)] = aligned_prev.loc[diff < float(threshold)]
    s = out.sum()
    if not np.isfinite(s) or s <= 0:
        return pd.Series(np.ones(len(out)) / len(out), index=out.index)
    return out / s


def _apply_dispersion_gate(
    weights: pd.Series,
    *,
    score_dispersion: float,
    cfg: MicroPipelineConfig,
) -> Tuple[pd.Series, bool, float]:
    """
    Shrink allocation toward equal weight when cross-sectional score dispersion
    is too low, because the ranking signal is likely weak / noisy.

    Returns
    -------
    gated_weights
    gate_active
    model_mix
        Share of the original model allocation kept after gating.
        1.0 means no gating, values closer to 0 imply stronger shrink toward EW.
    """
    if (not bool(cfg.dispersion_gate)) or (not cfg.long_only):
        return weights, False, 1.0

    threshold = max(float(cfg.dispersion_gate_threshold), 1e-8)
    dispersion = max(float(score_dispersion), 0.0)

    if dispersion >= threshold:
        return weights, False, 1.0

    model_mix_min = float(np.clip(cfg.dispersion_gate_min_active_weight, 0.0, 1.0))
    ratio = float(np.clip(dispersion / threshold, 0.0, 1.0))
    model_mix = model_mix_min + (1.0 - model_mix_min) * ratio

    ew = pd.Series(np.ones(len(weights)) / len(weights), index=weights.index)
    gated = model_mix * weights + (1.0 - model_mix) * ew
    s = gated.sum()
    if not np.isfinite(s) or s <= 0:
        gated = ew
    else:
        gated = gated / s

    return gated.astype(float), True, float(model_mix)




def _resolve_universe_scaling(n_assets: int, cfg: MicroPipelineConfig) -> Dict[str, float]:
    n = max(int(n_assets), 1)
    # keep intentionally mild; app can override more directly via cfg knobs
    if n <= 4:
        bucket = "small"
        temp_scale = 0.90
        topk_default = float(n)
    elif n <= 8:
        bucket = "medium"
        temp_scale = 1.00
        topk_default = float(min(n, 4))
    elif n <= 16:
        bucket = "large"
        temp_scale = 1.05
        topk_default = float(min(n, 6))
    else:
        bucket = "xlarge"
        temp_scale = 1.10
        topk_default = float(min(n, 8))
    return {
        "bucket": bucket,
        "temperature_scale": temp_scale,
        "weight_shrink_scale": 1.0,
        "inertia_scale": 1.0,
        "deadband_scale": 1.0,
        "dispersion_gate_scale": 1.0,
        "target_vol_scale": 1.0,
        "topk_default": topk_default,
    }


# Canonical implementations above already include the compatible adaptive allocation, vol-target, weights and probabilistic-forecast logic.
# Additional late override redefinitions were removed to avoid shadowing core engine functions.

# ============================================================
# Restored canonical micro-pipeline layer
# The original tail of the file contained multiple late overrides.
# This restored section provides a single live implementation for
# the public engine API used by app.py.
# ============================================================

from dataclasses import replace


def _safe_zscore_series(x: pd.Series) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mu = float(s.mean()) if s.notna().any() else 0.0
    sd = float(s.std(ddof=1)) if int(s.notna().sum()) >= 2 else 0.0
    if np.isfinite(sd) and sd > 1e-12:
        out = (s - mu) / sd
    else:
        out = s * 0.0
    return out.fillna(0.0).astype(float)


def _resolve_regime_dependent_cap_multiplier(regime: Optional[str], cfg: MicroPipelineConfig) -> Dict[str, Any]:
    rg = str(regime or "mid")
    if not bool(getattr(cfg, "regime_dependent_cap_enabled", False)):
        return {
            "regime_dependent_cap_enabled": False,
            "regime_dependent_cap_regime": rg,
            "regime_dependent_cap_multiplier": 1.0,
        }
    if rg == "high":
        mult = float(getattr(cfg, "regime_dependent_cap_high_mult", 0.80))
    elif rg == "low":
        mult = float(getattr(cfg, "regime_dependent_cap_low_mult", 1.05))
    else:
        mult = float(getattr(cfg, "regime_dependent_cap_mid_mult", 0.95))
    return {
        "regime_dependent_cap_enabled": True,
        "regime_dependent_cap_regime": rg,
        "regime_dependent_cap_multiplier": mult,
        "regime_dependent_cap_low_mult": float(getattr(cfg, "regime_dependent_cap_low_mult", np.nan)),
        "regime_dependent_cap_mid_mult": float(getattr(cfg, "regime_dependent_cap_mid_mult", np.nan)),
        "regime_dependent_cap_high_mult": float(getattr(cfg, "regime_dependent_cap_high_mult", np.nan)),
        "regime_dependent_cap_min": float(getattr(cfg, "regime_dependent_cap_min", np.nan)) if getattr(cfg, "regime_dependent_cap_min", None) is not None else np.nan,
        "regime_dependent_cap_max": float(getattr(cfg, "regime_dependent_cap_max", np.nan)) if getattr(cfg, "regime_dependent_cap_max", None) is not None else np.nan,
    }


def _resolve_effective_long_only_weight_cap(
    cfg: MicroPipelineConfig,
    n_assets: int,
    *,
    estimated_portfolio_vol_monthly: Optional[float] = None,
    corr_mat: Optional[pd.DataFrame] = None,
    score_dispersion: Optional[float] = None,
    regime: Optional[str] = None,
) -> Tuple[Optional[float], Dict[str, Any]]:
    requested_asset_weight_cap = getattr(cfg, "asset_weight_cap", None)
    requested_w_cap = getattr(cfg, "w_cap", None)
    base_requested = _first_finite_optional(requested_asset_weight_cap, requested_w_cap)
    meta: Dict[str, Any] = {
        "requested_asset_weight_cap": float(requested_asset_weight_cap) if requested_asset_weight_cap is not None else np.nan,
        "requested_w_cap": float(requested_w_cap) if requested_w_cap is not None else np.nan,
        "w_cap_active": requested_w_cap is not None,
        "combined_weight_cap_active": requested_asset_weight_cap is not None and requested_w_cap is not None,
        "base_requested_weight_cap": float(base_requested) if base_requested is not None else np.nan,
        "effective_requested_weight_cap": float(base_requested) if base_requested is not None else np.nan,
        "cap_source": "none" if base_requested is None else ("combined" if requested_asset_weight_cap is not None and requested_w_cap is not None else ("asset_weight_cap" if requested_asset_weight_cap is not None else "w_cap")),
        "vol_dependent_cap_enabled": bool(getattr(cfg, "vol_dependent_cap_enabled", False)),
        "vol_dependent_cap_vol_value": float(estimated_portfolio_vol_monthly) if estimated_portfolio_vol_monthly is not None and np.isfinite(float(estimated_portfolio_vol_monthly)) else np.nan,
        "vol_dependent_cap_multiplier": 1.0,
        "vol_dependent_cap_regime": "inactive",
        "corr_dependent_cap_enabled": bool(getattr(cfg, "corr_dependent_cap_enabled", False)),
        "corr_dependent_cap_corr_value": np.nan,
        "corr_dependent_cap_multiplier": 1.0,
        "corr_dependent_cap_regime": "inactive",
        "dispersion_dependent_cap_enabled": bool(getattr(cfg, "dispersion_dependent_cap_enabled", False)),
        "dispersion_dependent_cap_dispersion_value": float(score_dispersion) if score_dispersion is not None and np.isfinite(float(score_dispersion)) else np.nan,
        "dispersion_dependent_cap_multiplier": 1.0,
        "dispersion_dependent_cap_regime": "inactive",
        "combined_cap_multiplier": 1.0,
    }
    if base_requested is None or not np.isfinite(float(base_requested)):
        return None, meta

    mult = 1.0
    if bool(getattr(cfg, "vol_dependent_cap_enabled", False)) and estimated_portfolio_vol_monthly is not None and np.isfinite(float(estimated_portfolio_vol_monthly)):
        v = float(estimated_portfolio_vol_monthly)
        lo = float(getattr(cfg, "vol_dependent_cap_threshold_low", 0.03))
        hi = float(getattr(cfg, "vol_dependent_cap_threshold_high", 0.06))
        if v <= lo:
            m = float(getattr(cfg, "vol_dependent_cap_low_mult", 1.15)); rg = "low"
        elif v >= hi:
            m = float(getattr(cfg, "vol_dependent_cap_high_mult", 0.80)); rg = "high"
        else:
            frac = (v - lo) / max(hi - lo, 1e-12)
            m = float(getattr(cfg, "vol_dependent_cap_low_mult", 1.15)) + frac * (float(getattr(cfg, "vol_dependent_cap_high_mult", 0.80)) - float(getattr(cfg, "vol_dependent_cap_low_mult", 1.15)))
            rg = "mid"
        mult *= m
        meta.update({"vol_dependent_cap_multiplier": float(m), "vol_dependent_cap_regime": rg})

    if bool(getattr(cfg, "corr_dependent_cap_enabled", False)) and corr_mat is not None and not corr_mat.empty:
        arr = corr_mat.to_numpy(dtype="float64")
        mask = ~np.eye(arr.shape[0], dtype=bool)
        avg_abs_corr = float(np.mean(np.abs(arr[mask]))) if mask.any() else 0.0
        lo = float(getattr(cfg, "corr_dependent_cap_threshold_low", 0.20))
        hi = float(getattr(cfg, "corr_dependent_cap_threshold_high", 0.60))
        if avg_abs_corr <= lo:
            m = float(getattr(cfg, "corr_dependent_cap_low_mult", 1.10)); rg = "low"
        elif avg_abs_corr >= hi:
            m = float(getattr(cfg, "corr_dependent_cap_high_mult", 0.80)); rg = "high"
        else:
            frac = (avg_abs_corr - lo) / max(hi - lo, 1e-12)
            m = float(getattr(cfg, "corr_dependent_cap_low_mult", 1.10)) + frac * (float(getattr(cfg, "corr_dependent_cap_high_mult", 0.80)) - float(getattr(cfg, "corr_dependent_cap_low_mult", 1.10)))
            rg = "mid"
        mult *= m
        meta.update({"corr_dependent_cap_corr_value": avg_abs_corr, "corr_dependent_cap_multiplier": float(m), "corr_dependent_cap_regime": rg})

    if bool(getattr(cfg, "dispersion_dependent_cap_enabled", False)) and score_dispersion is not None and np.isfinite(float(score_dispersion)):
        d = float(score_dispersion)
        lo = float(getattr(cfg, "dispersion_dependent_cap_threshold_low", 0.10))
        hi = float(getattr(cfg, "dispersion_dependent_cap_threshold_high", 0.30))
        if d <= lo:
            m = float(getattr(cfg, "dispersion_dependent_cap_low_mult", 1.10)); rg = "low"
        elif d >= hi:
            m = float(getattr(cfg, "dispersion_dependent_cap_high_mult", 0.80)); rg = "high"
        else:
            frac = (d - lo) / max(hi - lo, 1e-12)
            m = float(getattr(cfg, "dispersion_dependent_cap_low_mult", 1.10)) + frac * (float(getattr(cfg, "dispersion_dependent_cap_high_mult", 0.80)) - float(getattr(cfg, "dispersion_dependent_cap_low_mult", 1.10)))
            rg = "mid"
        mult *= m
        meta.update({"dispersion_dependent_cap_multiplier": float(m), "dispersion_dependent_cap_regime": rg})

    regime_meta = _resolve_regime_dependent_cap_multiplier(regime, cfg)
    mult *= float(regime_meta.get("regime_dependent_cap_multiplier", 1.0))
    meta.update(regime_meta)
    meta["combined_cap_multiplier"] = float(mult)

    resolved = float(base_requested) * float(mult)
    if getattr(cfg, "vol_dependent_cap_min", None) is not None:
        resolved = max(resolved, float(getattr(cfg, "vol_dependent_cap_min")))
    if getattr(cfg, "vol_dependent_cap_max", None) is not None:
        resolved = min(resolved, float(getattr(cfg, "vol_dependent_cap_max")))
    if getattr(cfg, "corr_dependent_cap_min", None) is not None:
        resolved = max(resolved, float(getattr(cfg, "corr_dependent_cap_min")))
    if getattr(cfg, "corr_dependent_cap_max", None) is not None:
        resolved = min(resolved, float(getattr(cfg, "corr_dependent_cap_max")))
    if getattr(cfg, "dispersion_dependent_cap_min", None) is not None:
        resolved = max(resolved, float(getattr(cfg, "dispersion_dependent_cap_min")))
    if getattr(cfg, "dispersion_dependent_cap_max", None) is not None:
        resolved = min(resolved, float(getattr(cfg, "dispersion_dependent_cap_max")))
    if getattr(cfg, "regime_dependent_cap_min", None) is not None:
        resolved = max(resolved, float(getattr(cfg, "regime_dependent_cap_min")))
    if getattr(cfg, "regime_dependent_cap_max", None) is not None:
        resolved = min(resolved, float(getattr(cfg, "regime_dependent_cap_max")))
    feasible_floor = 1.0 / max(int(n_assets), 1)
    resolved = float(np.clip(resolved, feasible_floor, 1.0))
    meta["effective_requested_weight_cap"] = float(resolved)
    meta["cap_feasible_floor"] = float(feasible_floor)
    return resolved, meta


def _apply_long_only_weight_cap(weights: pd.Series, cap: Optional[float]) -> Tuple[pd.Series, Dict[str, Any]]:
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0).astype(float)
    if cap is None or not np.isfinite(float(cap)):
        m = float(w.max()) if len(w) else np.nan
        return w / max(float(w.sum()), 1e-12), {
            "asset_weight_cap_active": False,
            "effective_asset_weight_cap": np.nan,
            "pre_cap_max_weight": m,
            "post_cap_max_weight": m,
            "cap_binding_assets_count": 0,
            "cap_binding_assets_share": 0.0,
            "cap_total_excess_redistributed": 0.0,
            "cap_iterations": 0,
            "cap_violation_after_apply": 0.0,
        }
    cap = float(cap)
    out = w.copy()
    pre_max = float(out.max()) if len(out) else np.nan
    total_excess = 0.0
    iterations = 0
    for _ in range(20):
        iterations += 1
        excess = (out - cap).clip(lower=0.0)
        total_excess += float(excess.sum())
        if float(excess.sum()) <= 1e-12:
            break
        out = out.clip(upper=cap)
        slack_mask = out < (cap - 1e-12)
        slack_total = float(out.loc[slack_mask].sum())
        redist = float(excess.sum())
        if slack_total <= 1e-12:
            break
        out.loc[slack_mask] = out.loc[slack_mask] + redist * (out.loc[slack_mask] / slack_total)
    s = float(out.sum())
    if np.isfinite(s) and s > 0:
        out = out / s
    violation = float((out - cap).clip(lower=0.0).sum())
    binding = int((out >= cap - 1e-10).sum())
    return out.astype(float), {
        "asset_weight_cap_active": True,
        "effective_asset_weight_cap": float(cap),
        "pre_cap_max_weight": pre_max,
        "post_cap_max_weight": float(out.max()) if len(out) else np.nan,
        "cap_binding_assets_count": int(binding),
        "cap_binding_assets_share": float(binding / max(len(out), 1)),
        "cap_total_excess_redistributed": float(total_excess),
        "cap_iterations": int(iterations),
        "cap_violation_after_apply": float(violation),
    }


def _build_weights_from_mu_sigma(
    mu_hat: pd.Series,
    sigma_hat: pd.Series,
    *,
    cfg: MicroPipelineConfig,
    prev_weights: Optional[pd.Series] = None,
    n_assets: Optional[int] = None,
    corr_mat: Optional[pd.DataFrame] = None,
    sigma_fwd: Optional[pd.DataFrame] = None,
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    mu_num = pd.to_numeric(mu_hat, errors="coerce").fillna(0.0).astype(float)
    sigma_num = pd.to_numeric(sigma_hat, errors="coerce").astype(float)
    sigma_floor = max(float(getattr(cfg, "sigma_floor", 0.0) or 0.0), 1e-12)
    sigma_base = sigma_num.replace([np.inf, -np.inf], np.nan).fillna(sigma_floor).clip(lower=sigma_floor)

    alpha = float(getattr(cfg, "sigma_power_alpha", 1.0) or 1.0)
    if not np.isfinite(alpha):
        alpha = 1.0

    sigma_for_score = sigma_base.pow(alpha)
    sigma_for_score = sigma_for_score.replace([np.inf, -np.inf], np.nan).fillna(sigma_floor).clip(lower=sigma_floor)

    raw_score = (mu_num / sigma_for_score).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    score_dispersion = float(raw_score.std(ddof=1)) if len(raw_score) > 1 else 0.0

    score = raw_score.astype(float).copy()
    if bool(cfg.score_normalize):
        mu = float(score.mean())
        sd = float(score.std(ddof=1)) if len(score) > 1 else 0.0
        score = ((score - mu) / sd) if np.isfinite(sd) and sd > 1e-12 else (score - mu)

    if cfg.score_clip is not None:
        score = score.clip(lower=-abs(float(cfg.score_clip)), upper=abs(float(cfg.score_clip)))

    universe_scaling = _resolve_universe_scaling(int(n_assets) if n_assets is not None else len(score), cfg)
    allocation_meta = _resolve_adaptive_allocation(score_dispersion=score_dispersion, cfg=cfg)

    requested_temperature = float(allocation_meta.get("temperature", cfg.temperature))
    effective_temperature = _resolve_effective_temperature_by_universe_size(
        requested_temperature, universe_scaling, cfg
    )

    requested_top_k = allocation_meta.get("top_k", cfg.top_k)
    effective_top_k = _resolve_effective_top_k_by_universe_size(
        int(n_assets) if n_assets is not None else len(score),
        requested_top_k,
        universe_scaling,
        cfg,
    )

    score_for_alloc = score.copy()

    # --- top_k telemetry start ---
    top_asset_before = None
    top_score_before = np.nan
    second_asset_before = None
    second_score_before = np.nan
    selected_assets_pre_softmax = list(score_for_alloc.index.astype(str))
    selected_assets_after_topk = list(score_for_alloc.index.astype(str))
    topk_binding = False

    if len(score_for_alloc) > 0:
        ordered_before = score_for_alloc.sort_values(ascending=False)
        if len(ordered_before) >= 1:
            top_asset_before = str(ordered_before.index[0])
            top_score_before = float(ordered_before.iloc[0])
        if len(ordered_before) >= 2:
            second_asset_before = str(ordered_before.index[1])
            second_score_before = float(ordered_before.iloc[1])
    # --- top_k telemetry end ---

    if effective_top_k is not None and int(effective_top_k) > 0 and int(effective_top_k) < len(score_for_alloc):
        keep = score_for_alloc.nlargest(int(effective_top_k)).index
        selected_assets_after_topk = [str(x) for x in keep]
        topk_binding = True
        score_for_alloc = score_for_alloc.where(score_for_alloc.index.isin(keep), other=-1e9)
    else:
        selected_assets_after_topk = list(score_for_alloc.index.astype(str))

    if bool(cfg.long_only):
        weights = pd.Series(
            _softmax(score_for_alloc.values, temperature=effective_temperature),
            index=score_for_alloc.index,
            dtype="float64",
        )
    else:
        centered = score_for_alloc - score_for_alloc.mean()
        denom = float(np.abs(centered).sum())
        weights = (
            centered / denom
            if np.isfinite(denom) and denom > 0
            else pd.Series(np.ones(len(centered)) / max(len(centered), 1), index=centered.index, dtype="float64")
        )

    weights, corr_alloc_meta = _apply_correlation_aware_allocator(weights, score_for_alloc, sigma_fwd, corr_mat, cfg)

    shrink = min(max(float(cfg.weight_shrink) * float(universe_scaling.get("weight_shrink_scale", 1.0)), 0.0), 0.95)
    if bool(cfg.long_only) and 0.0 < shrink < 1.0:
        ew = pd.Series(np.ones(len(weights)) / max(len(weights), 1), index=weights.index, dtype="float64")
        weights = (1.0 - shrink) * weights + shrink * ew
        weights = weights / max(float(weights.sum()), 1e-12)

    effective_inertia = min(max(float(cfg.inertia) * float(universe_scaling.get("inertia_scale", 1.0)), 0.0), 0.95)
    if prev_weights is not None and effective_inertia > 0.0:
        aligned_prev = pd.to_numeric(prev_weights.reindex(weights.index), errors="coerce").fillna(0.0)
        weights = (1.0 - effective_inertia) * weights + effective_inertia * aligned_prev
        weights = weights / max(float(weights.sum()), 1e-12)

    eff_deadband = float(cfg.deadband_threshold)
    if bool(cfg.long_only) and bool(cfg.deadband):
        weights = _apply_deadband(weights, prev_weights, threshold=eff_deadband)

    weights, turnover_penalty_meta = _apply_turnover_penalty(weights, prev_weights, cfg)
    weights, turnover_constraint_meta = _apply_turnover_constraint(weights, prev_weights, cfg)
    turnover_meta = {**turnover_penalty_meta, **turnover_constraint_meta}

    est_vol = _estimate_portfolio_vol_monthly(
        weights,
        sigma_hat,
        sigma_fwd if bool(cfg.covariance_aware_vol_targeting) else None,
    )

    resolved_cap, cap_request_meta = _resolve_effective_long_only_weight_cap(
        cfg,
        len(weights),
        estimated_portfolio_vol_monthly=est_vol,
        corr_mat=corr_mat,
        score_dispersion=score_dispersion,
    )

    weights, cap_meta = _apply_long_only_weight_cap(weights, resolved_cap) if bool(cfg.long_only) else (
        weights,
        {
            "asset_weight_cap_active": False,
            "effective_asset_weight_cap": np.nan,
            "pre_cap_max_weight": float(np.max(np.abs(weights.values))) if len(weights) else np.nan,
            "post_cap_max_weight": float(np.max(np.abs(weights.values))) if len(weights) else np.nan,
            "cap_binding_assets_count": 0,
            "cap_binding_assets_share": 0.0,
            "cap_total_excess_redistributed": 0.0,
            "cap_iterations": 0,
            "cap_violation_after_apply": 0.0,
        },
    )
    cap_meta = {**cap_request_meta, **cap_meta}

    threshold = float(getattr(cfg, "low_signal_fallback_threshold", 0.05))
    low_signal_active = False
    low_signal_model_weight = 1.0
    if bool(getattr(cfg, "low_signal_fallback_to_ew", False)) and score_dispersion < threshold and bool(cfg.long_only):
        ew = pd.Series(np.ones(len(weights)) / max(len(weights), 1), index=weights.index, dtype="float64")
        mode = str(getattr(cfg, "low_signal_fallback_mode", "blend") or "blend")
        if mode == "hard":
            weights = ew
            low_signal_model_weight = 0.0
        else:
            frac = score_dispersion / max(threshold, 1e-12)
            min_model = float(getattr(cfg, "low_signal_fallback_min_model_weight", 0.25))
            low_signal_model_weight = min_model + (1.0 - min_model) * float(np.clip(frac, 0.0, 1.0))
            weights = low_signal_model_weight * weights + (1.0 - low_signal_model_weight) * ew
            weights = weights / max(float(weights.sum()), 1e-12)
        low_signal_active = True

    sigma_transform_ratio = (sigma_for_score / sigma_base.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)

    allocation_meta = {
        **allocation_meta,
        **corr_alloc_meta,
        **turnover_meta,
        **cap_meta,
        "requested_temperature": requested_temperature,
        "effective_temperature": effective_temperature,
        "requested_top_k": requested_top_k,
        "effective_top_k": effective_top_k,
        "score_dispersion": score_dispersion,
        "sigma_power_alpha": float(alpha),
        "sigma_power_alpha_active": bool(abs(alpha - 1.0) > 1e-12),
        "sigma_score_floor": float(sigma_floor),
        "sigma_score_denom_mean": float(sigma_for_score.mean()) if len(sigma_for_score) else np.nan,
        "sigma_score_denom_median": float(sigma_for_score.median()) if len(sigma_for_score) else np.nan,
        "sigma_score_transform_ratio_mean": float(sigma_transform_ratio.mean()) if len(sigma_transform_ratio.dropna()) else np.nan,
        "sigma_score_transform_ratio_max": float(sigma_transform_ratio.max()) if len(sigma_transform_ratio.dropna()) else np.nan,
        "effective_weight_shrink": float(shrink),
        "effective_inertia": float(effective_inertia),
        "effective_deadband_threshold": float(eff_deadband),
        "low_signal_fallback_active": bool(low_signal_active),
        "low_signal_fallback_regime": "low" if low_signal_active else "inactive",
        "low_signal_fallback_model_weight": float(low_signal_model_weight),
        "low_signal_fallback_threshold": float(threshold),

        # --- top_k telemetry ---
        "topk_binding": bool(topk_binding),
        "n_assets_score_input": int(len(score)),
        "n_assets_after_topk": int(len(selected_assets_after_topk)),
        "top_asset_before_topk": top_asset_before,
        "top_score_before_topk": top_score_before,
        "second_asset_before_topk": second_asset_before,
        "second_score_before_topk": second_score_before,
        "selected_assets_pre_softmax": ",".join(selected_assets_pre_softmax),
        "selected_assets_after_topk": ",".join(selected_assets_after_topk),
    }

    return weights.astype(float), raw_score.astype(float), allocation_meta


def _resolve_probabilistic_feature_columns(asset_panel_df: pd.DataFrame) -> List[str]:
    candidates = [
        "mom_5d_eom", "mom_10d_eom", "mom_21d_eom", "mom_5d_over_vol_21d_eom", "mom_10d_over_vol_21d_eom",
        "mom_21d_over_vol_21d_eom", "intramonth_realized_vol_ann", "ewma_vol_21d_ann", "roll_vol_21d_ann",
        "mom_21d_cs_z_eom", "asset_vs_cross_section_strength_eom", "cs_ret_dispersion_mean_m", "cs_avg_pairwise_corr",
        "conditional_rebalance_vol_stress_score", "conditional_rebalance_corr_stress_score",
    ]
    if asset_panel_df is None or not isinstance(asset_panel_df, pd.DataFrame) or asset_panel_df.empty:
        return []
    cols = set(asset_panel_df.columns)
    return [c for c in candidates if c in cols]


def _ensure_feature_mu_fallback_columns(asset_panel_df: pd.DataFrame, cfg: MicroPipelineConfig) -> pd.DataFrame:
    if asset_panel_df is None or not isinstance(asset_panel_df, pd.DataFrame) or asset_panel_df.empty:
        return asset_panel_df

    req = {cfg.date_col, cfg.asset_col, cfg.return_col}
    if not req.issubset(asset_panel_df.columns):
        return asset_panel_df

    fallback_cols = [
        "feature_mu_ret_1m_lag",
        "feature_mu_mom_3m",
        "feature_mu_mom_6m",
        "feature_mu_mom_12m",
        "feature_mu_vol_3m",
        "feature_mu_vol_6m",
        "feature_mu_vol_12m",
    ]

    existing = set(asset_panel_df.columns)
    if all(col in existing for col in fallback_cols):
        return asset_panel_df

    tmp = asset_panel_df.copy()
    tmp[cfg.date_col] = pd.to_datetime(tmp[cfg.date_col], errors="coerce")
    tmp[cfg.asset_col] = tmp[cfg.asset_col].astype(str)
    tmp[cfg.return_col] = pd.to_numeric(tmp[cfg.return_col], errors="coerce")
    tmp = tmp.sort_values([cfg.asset_col, cfg.date_col]).reset_index(drop=True)

    g = tmp.groupby(cfg.asset_col)[cfg.return_col]
    tmp["feature_mu_ret_1m_lag"] = g.shift(1)

    for win in (3, 6, 12):
        lagged = g.shift(1)
        tmp[f"feature_mu_mom_{win}m"] = lagged.groupby(tmp[cfg.asset_col]).rolling(win, min_periods=max(2, min(win, 3))).mean().reset_index(level=0, drop=True)
        tmp[f"feature_mu_vol_{win}m"] = lagged.groupby(tmp[cfg.asset_col]).rolling(win, min_periods=max(2, min(win, 3))).std(ddof=1).reset_index(level=0, drop=True)

    return tmp


def _feature_mu_family_and_priority(col: str) -> Tuple[str, int]:
    name = str(col or "").strip().lower()
    if not name:
        return "other", 999
    if any(tok in name for tok in ("regime", "stress", "dispersion", "pairwise_corr", "corr_stress", "vol_stress")):
        return "regime", 60
    if any(tok in name for tok in ("mom_", "momentum", "alpha_mom", "ret_1d", "ret_1m_lag", "mom_3m", "mom_6m", "mom_12m")):
        return "momentum", 10
    if any(tok in name for tok in ("over_vol", "efficiency", "strength", "channel", "dist_from", "pos_rate", "accel", "spread")):
        return "quality", 20
    if "reversal" in name:
        return "reversal", 30
    if any(tok in name for tok in ("vol", "risk", "range", "skew", "kurt", "downside", "upside")):
        return "risk", 40
    return "other", 50


def _compute_feature_family_counts(selected_cols: Sequence[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not selected_cols:
        return counts
    for raw_col in list(selected_cols):
        col = str(raw_col or "").strip()
        if not col:
            continue
        family, _ = _feature_mu_family_and_priority(col)
        family_key = str(family or "other")
        counts[family_key] = int(counts.get(family_key, 0) + 1)
    return counts


_DEF_FEATURE_MU_FAMILY_CAPS: Dict[str, int] = {
    "momentum": 4,
    "quality": 3,
    "reversal": 1,
    "risk": 2,
    "regime": 1,
    "other": 2,
}


def _inspect_feature_mu_selection(asset_panel_df: pd.DataFrame, candidate_cols: Sequence[str]) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "candidate_cols": [],
        "selected_cols": [],
        "selected_family_counts": {},
        "excluded_low_quality_n": 0,
        "excluded_preview": [],
        "excluded_reasons": {},
    }
    if asset_panel_df is None or not isinstance(asset_panel_df, pd.DataFrame) or asset_panel_df.empty or not candidate_cols:
        return info

    info["candidate_cols"] = [str(c) for c in list(candidate_cols) if c is not None]
    seen: set[str] = set()
    family_counts: Dict[str, int] = {k: 0 for k in _DEF_FEATURE_MU_FAMILY_CAPS}
    selected_rows: List[Tuple[str, int, float, float]] = []
    excluded_preview: List[str] = []
    excluded_reasons: Dict[str, int] = {}

    def _mark_excluded(col: str, reason: str) -> None:
        reason_key = str(reason or "unknown")
        excluded_reasons[reason_key] = int(excluded_reasons.get(reason_key, 0) + 1)
        if len(excluded_preview) < 12:
            excluded_preview.append(f"{col}:{reason_key}")

    for raw_col in list(candidate_cols):
        col = str(raw_col)
        if col in seen:
            _mark_excluded(col, "duplicate")
            continue
        seen.add(col)
        if col not in asset_panel_df.columns:
            _mark_excluded(col, "missing_column")
            continue

        s = pd.to_numeric(asset_panel_df[col], errors="coerce")
        valid = s.dropna()
        if valid.empty:
            _mark_excluded(col, "all_nan")
            continue

        n_obs = int(valid.shape[0])
        if n_obs < 12:
            _mark_excluded(col, "insufficient_obs")
            continue

        missing_share = float(1.0 - (n_obs / max(int(len(s)), 1)))
        if missing_share > 0.60:
            _mark_excluded(col, "high_missing_share")
            continue

        std = float(valid.std(ddof=1)) if n_obs > 1 else 0.0
        if (not np.isfinite(std)) or std <= 1e-12:
            _mark_excluded(col, "low_variance")
            continue

        nunique = int(valid.nunique(dropna=True))
        if nunique <= 1:
            _mark_excluded(col, "constant")
            continue

        family, priority = _feature_mu_family_and_priority(col)
        family_cap = int(_DEF_FEATURE_MU_FAMILY_CAPS.get(family, 1))
        if family_counts.get(family, 0) >= family_cap:
            _mark_excluded(col, f"family_cap_{family}")
            continue

        family_counts[family] = int(family_counts.get(family, 0) + 1)
        selected_rows.append((col, int(priority), float(missing_share), -float(std)))

    selected_rows = sorted(selected_rows, key=lambda x: (x[1], x[2], x[3], x[0]))
    selected_cols = [col for col, _, _, _ in selected_rows]
    selected_family_counts: Dict[str, int] = {}
    for col in selected_cols:
        fam, _ = _feature_mu_family_and_priority(col)
        selected_family_counts[fam] = int(selected_family_counts.get(fam, 0) + 1)

    info["selected_cols"] = selected_cols
    info["selected_family_counts"] = selected_family_counts
    info["excluded_low_quality_n"] = int(sum(excluded_reasons.values()))
    info["excluded_preview"] = excluded_preview
    info["excluded_reasons"] = excluded_reasons
    return info


def _resolve_feature_mu_selection_debug(asset_panel_df: pd.DataFrame) -> Dict[str, Any]:
    empty = {
        "selection_source": "none",
        "candidate_cols": [],
        "selected_cols": [],
        "selected_family_counts": {},
        "excluded_low_quality_n": 0,
        "excluded_preview": [],
        "excluded_reasons": {},
    }
    if asset_panel_df is None or not isinstance(asset_panel_df, pd.DataFrame) or asset_panel_df.empty:
        return empty

    cols = list(asset_panel_df.columns)
    cs_z = [c for c in cols if str(c).endswith("_cs_z")]
    info = _inspect_feature_mu_selection(asset_panel_df, cs_z)
    if info.get("selected_cols"):
        info["selection_source"] = "cs_z"
        return info

    preferred_candidates = _resolve_probabilistic_feature_columns(asset_panel_df)
    info = _inspect_feature_mu_selection(asset_panel_df, preferred_candidates)
    if info.get("selected_cols"):
        info["selection_source"] = "preferred"
        return info

    fallback = [
        "feature_mu_ret_1m_lag",
        "feature_mu_mom_3m",
        "feature_mu_mom_6m",
        "feature_mu_mom_12m",
        "feature_mu_vol_3m",
        "feature_mu_vol_6m",
        "feature_mu_vol_12m",
    ]
    info = _inspect_feature_mu_selection(asset_panel_df, fallback)
    info["selection_source"] = "fallback"
    return info


def _resolve_feature_mu_columns(asset_panel_df: pd.DataFrame) -> List[str]:
    return list(_resolve_feature_mu_selection_debug(asset_panel_df).get("selected_cols", []))


def _build_probabilistic_feature_pairs(asset_panel_df: pd.DataFrame, cfg: MicroPipelineConfig, feature_cols: List[str]) -> pd.DataFrame:
    if asset_panel_df is None or asset_panel_df.empty or not feature_cols:
        return pd.DataFrame(columns=[cfg.date_col, cfg.asset_col, "next_return", *feature_cols])
    use_cols = [cfg.date_col, cfg.asset_col, cfg.return_col, *feature_cols]
    tmp = asset_panel_df[use_cols].copy()
    tmp[cfg.date_col] = pd.to_datetime(tmp[cfg.date_col], errors="coerce")
    tmp[cfg.asset_col] = tmp[cfg.asset_col].astype(str)
    tmp[cfg.return_col] = pd.to_numeric(tmp[cfg.return_col], errors="coerce")
    for col in feature_cols:
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
    tmp = tmp.sort_values([cfg.asset_col, cfg.date_col]).reset_index(drop=True)
    tmp["next_return"] = tmp.groupby(cfg.asset_col)[cfg.return_col].shift(-1)
    tmp = tmp.dropna(subset=[cfg.date_col, cfg.asset_col, "next_return"])
    return tmp.reset_index(drop=True)


def _build_current_feature_state(asset_panel_df: pd.DataFrame, dt: pd.Timestamp, cfg: MicroPipelineConfig, feature_cols: List[str]) -> pd.DataFrame:
    if asset_panel_df is None or asset_panel_df.empty or not feature_cols:
        return pd.DataFrame(columns=feature_cols)
    tmp = asset_panel_df.loc[pd.to_datetime(asset_panel_df[cfg.date_col], errors="coerce") == pd.Timestamp(dt), [cfg.asset_col, *feature_cols]].copy()
    if tmp.empty:
        return pd.DataFrame(columns=feature_cols)
    tmp[cfg.asset_col] = tmp[cfg.asset_col].astype(str)
    for col in feature_cols:
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
    return tmp.drop_duplicates(subset=[cfg.asset_col]).set_index(cfg.asset_col)


def _apply_feature_conditioned_mu(
    mu_hat: pd.Series,
    sigma_hat: Optional[pd.Series] = None,
    *,
    historical_feature_pairs: Optional[pd.DataFrame],
    current_feature_state: Optional[pd.DataFrame],
    cfg: MicroPipelineConfig,
) -> Tuple[pd.Series, Dict[str, Any], pd.DataFrame]:
    base = pd.to_numeric(mu_hat, errors="coerce").fillna(0.0).astype(float)
    empty_details = pd.DataFrame(index=base.index)
    if not bool(getattr(cfg, "feature_mu_enabled", False)) or historical_feature_pairs is None or current_feature_state is None or historical_feature_pairs.empty or current_feature_state.empty:
        meta = {"feature_mu_active_share": 0.0, "feature_mu_match_n_mean": np.nan, "feature_mu_cols_used_mean": np.nan, "feature_mu_distance_mean": np.nan, "feature_mu_abs_tilt_mean": 0.0, "feature_mu_abs_tilt_max": 0.0}
        return base, meta, empty_details

    out = base.copy()
    rows = []
    blend = float(np.clip(getattr(cfg, "feature_mu_blend", 0.25), 0.0, 1.0))
    scale_mode = str(getattr(cfg, "feature_mu_scale_mode", "none") or "none").strip().lower()
    if scale_mode not in {"none", "zscore", "vol_adjusted"}:
        scale_mode = "none"
    epsilon = float(getattr(cfg, "feature_mu_sigma_epsilon", 1e-6) or 1e-6)
    if not np.isfinite(epsilon) or epsilon < 0.0:
        epsilon = 1e-6
    sigma_floor = max(float(getattr(cfg, "sigma_floor", 0.02) or 0.02), 1e-12)
    selection_debug = _resolve_feature_mu_selection_debug(
        historical_feature_pairs.drop(columns=["next_return"], errors="ignore")
        if isinstance(historical_feature_pairs, pd.DataFrame)
        else pd.DataFrame()
    )
    feature_cols_all = [c for c in selection_debug.get("candidate_cols", []) if c in current_feature_state.columns and c in historical_feature_pairs.columns]
    raw_selected_cols = [c for c in selection_debug.get("selected_cols", []) if c in current_feature_state.columns and c in historical_feature_pairs.columns]
    feature_cols = raw_selected_cols[: max(int(getattr(cfg, "feature_mu_cols_max", 8)), 1)]
    selected_family_counts = _compute_feature_family_counts(feature_cols)
    selected_family_counts_json = json.dumps(selected_family_counts, sort_keys=True)
    feature_mu_pred = pd.Series(np.nan, index=base.index, dtype="float64")
    feature_mu_distance_mean = pd.Series(np.nan, index=base.index, dtype="float64")
    feature_mu_match_n = pd.Series(np.nan, index=base.index, dtype="float64")
    feature_mu_cols_used = pd.Series(np.nan, index=base.index, dtype="float64")

    for asset in base.index:
        pred = np.nan
        dist_mean = np.nan
        match_n = 0
        cols_used = 0
        if asset in set(current_feature_state.index):
            asset_pairs = historical_feature_pairs.loc[historical_feature_pairs[cfg.asset_col].astype(str) == str(asset)].copy()
            if not asset_pairs.empty and feature_cols:
                current_row = current_feature_state.loc[str(asset)] if str(asset) in current_feature_state.index else current_feature_state.loc[asset]
                if isinstance(current_row, pd.DataFrame):
                    current_row = current_row.iloc[-1]
                candidate_cols = []
                for col in feature_cols:
                    cur = pd.to_numeric(pd.Series([current_row.get(col, np.nan)]), errors="coerce").iloc[0]
                    s = pd.to_numeric(asset_pairs[col], errors="coerce") if col in asset_pairs.columns else pd.Series(dtype="float64")
                    if np.isfinite(cur) and s.notna().sum() >= max(int(getattr(cfg, "feature_mu_min_obs", 12)), 3):
                        candidate_cols.append(col)
                if candidate_cols:
                    cols_used = len(candidate_cols)
                    block = asset_pairs[[*candidate_cols, "next_return"]].copy().dropna()
                    if len(block) >= max(int(getattr(cfg, "feature_mu_min_obs", 12)), 3):
                        cur_vec = np.array([float(pd.to_numeric(pd.Series([current_row.get(c, np.nan)]), errors="coerce").iloc[0]) for c in candidate_cols], dtype="float64")
                        X = block[candidate_cols].to_numpy(dtype="float64")
                        scale = np.nanstd(X, axis=0, ddof=1)
                        scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
                        d = np.sqrt(np.mean(((X - cur_vec) / scale) ** 2, axis=1))
                        k = min(max(int(getattr(cfg, "feature_mu_k", 24)), 1), len(d))
                        nn = np.argsort(d)[:k]
                        pred = float(pd.to_numeric(block.iloc[nn]["next_return"], errors="coerce").mean())
                        dist_mean = float(np.mean(d[nn])) if len(nn) else np.nan
                        match_n = int(len(nn))
        feature_mu_pred.loc[asset] = pred
        feature_mu_distance_mean.loc[asset] = dist_mean
        feature_mu_match_n.loc[asset] = float(match_n) if match_n else np.nan
        feature_mu_cols_used.loc[asset] = float(cols_used) if cols_used else np.nan

    feature_mu_adjustment = (pd.to_numeric(feature_mu_pred, errors="coerce") - base).astype(float)
    feature_mu_adjustment = feature_mu_adjustment.replace([np.inf, -np.inf], np.nan)

    if scale_mode == "zscore":
        adj = pd.to_numeric(feature_mu_adjustment, errors="coerce")
        adj_mean = float(adj.mean())
        adj_std = max(float(adj.std(ddof=0)), 1e-8)
        adjustment_scaled = (adj - adj_mean) / adj_std
    elif scale_mode == "vol_adjusted":
        sigma_safe = pd.to_numeric(sigma_hat.reindex(base.index) if isinstance(sigma_hat, pd.Series) else pd.Series(index=base.index, dtype="float64"), errors="coerce")
        sigma_safe = sigma_safe.replace([np.inf, -np.inf], np.nan).fillna(sigma_floor)
        sigma_safe = pd.Series(np.maximum(sigma_safe.to_numpy(dtype="float64"), sigma_floor), index=base.index, dtype="float64")
        adjustment_scaled = pd.to_numeric(feature_mu_adjustment, errors="coerce") / (sigma_safe + epsilon)
    else:
        adjustment_scaled = feature_mu_adjustment

    # Step 16 — optional quantile filter (post-scaling)
    feature_mu_apply_quantile = getattr(cfg, "feature_mu_apply_quantile", None)
    if feature_mu_apply_quantile is not None:
        try:
            q = float(feature_mu_apply_quantile)
        except Exception:
            q = None
        if q is not None and 0.0 < q < 1.0:
            adj_abs = pd.to_numeric(adjustment_scaled, errors="coerce").abs()
            thr = float(adj_abs.quantile(q))
            mask = adj_abs >= thr
            adjustment_scaled = pd.to_numeric(adjustment_scaled, errors="coerce").where(mask, 0.0)

    feature_mu_rank_aware = bool(getattr(cfg, "feature_mu_rank_aware", False))
    if feature_mu_rank_aware:
        adj_rank = pd.to_numeric(adjustment_scaled, errors="coerce")
        valid_adj = adj_rank.dropna()
        if len(valid_adj) >= 2:
            rank_scaled = (valid_adj.rank(method="average", pct=True) - 0.5) * 2.0
            adjustment_scaled = adj_rank.copy()
            adjustment_scaled.loc[rank_scaled.index] = rank_scaled.astype(float)
        else:
            adjustment_scaled = adj_rank
    else:
        adjustment_scaled = pd.to_numeric(adjustment_scaled, errors="coerce")

    active_mask = feature_mu_pred.notna() & np.isfinite(base) & np.isfinite(pd.to_numeric(adjustment_scaled, errors="coerce")) & (blend > 0)
    out.loc[active_mask] = base.loc[active_mask] + blend * pd.to_numeric(adjustment_scaled.loc[active_mask], errors="coerce")

    for asset in base.index:
        rows.append({
            "asset": str(asset),
            "feature_mu_pred": float(feature_mu_pred.loc[asset]) if np.isfinite(feature_mu_pred.loc[asset]) else np.nan,
            "feature_mu_distance_mean": float(feature_mu_distance_mean.loc[asset]) if np.isfinite(feature_mu_distance_mean.loc[asset]) else np.nan,
            "feature_mu_match_n": float(feature_mu_match_n.loc[asset]) if np.isfinite(feature_mu_match_n.loc[asset]) else np.nan,
            "feature_mu_cols_used": float(feature_mu_cols_used.loc[asset]) if np.isfinite(feature_mu_cols_used.loc[asset]) else np.nan,
            "feature_mu_active": bool(active_mask.loc[asset]),
            "mu_hat_base": float(base.loc[asset]),
            "mu_hat_feature_tilted": float(out.loc[asset]),
            "feature_mu_adjustment": float(feature_mu_adjustment.loc[asset]) if np.isfinite(feature_mu_adjustment.loc[asset]) else np.nan,
            "feature_mu_adjustment_scaled": float(adjustment_scaled.loc[asset]) if np.isfinite(adjustment_scaled.loc[asset]) else np.nan,
            "feature_mu_scale_mode": scale_mode,
            "feature_mu_abs_tilt": float(abs(out.loc[asset] - base.loc[asset])) if np.isfinite(base.loc[asset]) and np.isfinite(out.loc[asset]) else np.nan,
            "feature_mu_candidate_cols_n": int(len(feature_cols_all)),
            "feature_mu_selected_cols_n": int(len(feature_cols)),
            "feature_mu_selected_cols": "|".join([str(c) for c in feature_cols]),
            "feature_mu_selected_family_counts": selected_family_counts_json,
            "feature_mu_excluded_low_quality_n": int(selection_debug.get("excluded_low_quality_n", 0)),
            "feature_mu_excluded_preview": "|".join([str(x) for x in selection_debug.get("excluded_preview", [])]),
            "feature_mu_excluded_reasons": json.dumps(selection_debug.get("excluded_reasons", {}), sort_keys=True),
            "feature_mu_candidate_cols": "|".join([str(c) for c in feature_cols_all]),
            "feature_mu_selection_source": str(selection_debug.get("selection_source", "none") or "none"),
            "n_feature_candidates": int(len(feature_cols_all)),
            "n_feature_selected": int(len(feature_cols)),
            "selected_feature_columns": "|".join([str(c) for c in feature_cols]),
            "feature_family_counts": selected_family_counts_json,
            "excluded_feature_columns_preview": "|".join([str(x) for x in selection_debug.get("excluded_preview", [])]),
        })
    details = pd.DataFrame(rows).set_index("asset") if rows else empty_details
    try:
        details.to_csv("debug_feature_mu_details.csv")
    except Exception:
        pass
    meta = {
        "feature_mu_active_share": float(pd.to_numeric(details.get("feature_mu_active", pd.Series(dtype="float64")), errors="coerce").mean()) if not details.empty else 0.0,
        "feature_mu_match_n_mean": float(pd.to_numeric(details.get("feature_mu_match_n", pd.Series(dtype="float64")), errors="coerce").mean()) if not details.empty else np.nan,
        "feature_mu_cols_used_mean": float(pd.to_numeric(details.get("feature_mu_cols_used", pd.Series(dtype="float64")), errors="coerce").mean()) if not details.empty else np.nan,
        "feature_mu_distance_mean": float(pd.to_numeric(details.get("feature_mu_distance_mean", pd.Series(dtype="float64")), errors="coerce").mean()) if not details.empty else np.nan,
        "feature_mu_abs_tilt_mean": float(pd.to_numeric(details.get("feature_mu_abs_tilt", pd.Series(dtype="float64")), errors="coerce").mean()) if not details.empty else 0.0,
        "feature_mu_abs_tilt_max": float(pd.to_numeric(details.get("feature_mu_abs_tilt", pd.Series(dtype="float64")), errors="coerce").max()) if not details.empty else 0.0,
        "feature_mu_candidate_cols_n": int(len(feature_cols_all)),
        "feature_mu_selected_cols_n": int(len(feature_cols)),
        "feature_mu_selected_cols": "|".join([str(c) for c in feature_cols]),
        "feature_mu_selected_family_counts": selected_family_counts_json,
        "feature_mu_excluded_low_quality_n": int(selection_debug.get("excluded_low_quality_n", 0)),
        "feature_mu_excluded_preview": "|".join([str(x) for x in selection_debug.get("excluded_preview", [])]),
        "feature_mu_excluded_reasons": json.dumps(selection_debug.get("excluded_reasons", {}), sort_keys=True),
        "feature_mu_candidate_cols": "|".join([str(c) for c in feature_cols_all]),
        "feature_mu_selection_source": str(selection_debug.get("selection_source", "none") or "none"),
        "feature_mu_scale_mode": scale_mode,
        "n_feature_candidates": int(len(feature_cols_all)),
        "n_feature_selected": int(len(feature_cols)),
        "selected_feature_columns": "|".join([str(c) for c in feature_cols]),
        "feature_family_counts": selected_family_counts_json,
        "excluded_feature_columns_preview": "|".join([str(x) for x in selection_debug.get("excluded_preview", [])]),
    }
    return out.astype(float), meta, details


def _apply_regime_dependent_universe_filter(
    mu_signal: pd.Series,
    sigma_hat: pd.Series,
    *,
    regime: str,
    cfg: MicroPipelineConfig,
) -> Tuple[pd.Series, pd.Series, Dict[str, Any], pd.DataFrame]:
    mu = pd.to_numeric(mu_signal, errors="coerce").fillna(0.0).astype(float)
    sigma = pd.to_numeric(sigma_hat, errors="coerce").replace(0.0, np.nan).fillna(float(getattr(cfg, "sigma_floor", 0.02) or 0.02)).astype(float)
    details_empty = pd.DataFrame(index=mu.index)
    keep_low = float(np.clip(getattr(cfg, "regime_universe_keep_frac_low", 1.0), 0.0, 1.0))
    keep_mid = float(np.clip(getattr(cfg, "regime_universe_keep_frac_mid", 0.85), 0.0, 1.0))
    keep_high = float(np.clip(getattr(cfg, "regime_universe_keep_frac_high", 0.60), 0.0, 1.0))
    selection_metric = str(getattr(cfg, "regime_universe_selection_metric", "mu_over_sigma") or "mu_over_sigma")
    enabled = bool(getattr(cfg, "regime_dependent_universe_enabled", False))
    low_offensive_vol_tilt = float(getattr(cfg, "regime_universe_low_offensive_vol_tilt", 0.25) or 0.25)
    high_defensive_vol_tilt = float(getattr(cfg, "regime_universe_high_defensive_vol_tilt", 0.75) or 0.75)
    min_assets = int(max(getattr(cfg, "regime_universe_min_assets", 2) or 2, 1))
    max_assets_cfg = getattr(cfg, "regime_universe_max_assets", None)
    max_assets_value = float(max_assets_cfg) if max_assets_cfg is not None else np.nan

    if regime == "low":
        keep_frac_config = keep_low
    elif regime == "high":
        keep_frac_config = keep_high
    else:
        keep_frac_config = keep_mid

    meta_base = {
        "regime_dependent_universe_enabled": enabled,
        "regime_universe_enabled": enabled,
        "regime_universe_active": False,
        "regime_universe_keep_frac": 1.0 if not enabled else float(keep_frac_config),
        "regime_universe_keep_frac_low": keep_low,
        "regime_universe_keep_frac_mid": keep_mid,
        "regime_universe_keep_frac_high": keep_high,
        "regime_universe_selection_metric": selection_metric,
        "regime_universe_metric_mode": selection_metric,
        "regime_universe_regime": str(regime),
        "regime_universe_low_offensive_vol_tilt": low_offensive_vol_tilt,
        "regime_universe_high_defensive_vol_tilt": high_defensive_vol_tilt,
        "regime_universe_min_assets": min_assets,
        "regime_universe_max_assets": max_assets_value,
        "regime_universe_selected_assets": float(len(mu)),
        "regime_universe_filtered_assets": 0.0,
        "selected_assets_after_regime_filter": float(len(mu)),
        "filtered_assets_by_regime": 0.0,
    }
    if not enabled or len(mu) <= 0:
        return mu.astype(float), sigma.astype(float), meta_base, details_empty

    keep_frac = float(keep_frac_config)
    n_assets = int(len(mu))
    max_assets = n_assets if max_assets_cfg is None else int(max(max_assets_cfg, 1))
    n_keep = int(np.ceil(float(np.clip(keep_frac, 0.0, 1.0)) * n_assets))
    n_keep = max(min_assets, min(n_keep, max_assets, n_assets))

    base_metric = mu.copy()
    sigma_safe = sigma.replace(0.0, np.nan).fillna(float(getattr(cfg, "sigma_floor", 0.02) or 0.02))
    if selection_metric == "score":
        ranking_signal = (mu / sigma_safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    elif selection_metric == "mu_over_sigma":
        ranking_signal = (mu / sigma_safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        ranking_signal = base_metric.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    sigma_z = ((sigma_safe - sigma_safe.mean()) / max(float(sigma_safe.std(ddof=1)) if len(sigma_safe) > 1 else 0.0, 1e-12)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ranking_adj = ranking_signal.copy()
    if regime == "low":
        ranking_adj = ranking_adj + low_offensive_vol_tilt * sigma_z
    elif regime == "high":
        ranking_adj = ranking_adj - high_defensive_vol_tilt * sigma_z

    selected_assets = ranking_adj.sort_values(ascending=False).head(n_keep).index.astype(str)
    selected_set = set(selected_assets)
    details = pd.DataFrame({
        "asset": mu.index.astype(str),
        "regime_universe_selected": [str(a) in selected_set for a in mu.index.astype(str)],
        "regime_universe_rank_signal": pd.to_numeric(ranking_signal.reindex(mu.index), errors="coerce").to_numpy(dtype="float64"),
        "regime_universe_rank_signal_adjusted": pd.to_numeric(ranking_adj.reindex(mu.index), errors="coerce").to_numpy(dtype="float64"),
        "regime_universe_sigma_z": pd.to_numeric(sigma_z.reindex(mu.index), errors="coerce").to_numpy(dtype="float64"),
        "regime_universe_keep_frac": keep_frac,
        "regime_universe_keep_frac_low": keep_low,
        "regime_universe_keep_frac_mid": keep_mid,
        "regime_universe_keep_frac_high": keep_high,
        "regime_universe_selection_metric": selection_metric,
        "regime_universe_metric_mode": selection_metric,
        "regime_universe_regime": str(regime),
        "regime_universe_low_offensive_vol_tilt": low_offensive_vol_tilt,
        "regime_universe_high_defensive_vol_tilt": high_defensive_vol_tilt,
        "regime_universe_min_assets": min_assets,
        "regime_universe_max_assets": max_assets_value,
    }).set_index("asset")

    mu_f = mu.reindex(selected_assets).astype(float)
    sigma_f = sigma.reindex(selected_assets).astype(float)
    meta = {
        **meta_base,
        "regime_universe_active": True,
        "regime_universe_keep_frac": keep_frac,
        "regime_universe_selected_assets": float(len(selected_assets)),
        "regime_universe_filtered_assets": float(max(n_assets - len(selected_assets), 0)),
        "selected_assets_after_regime_filter": float(len(selected_assets)),
        "filtered_assets_by_regime": float(max(n_assets - len(selected_assets), 0)),
    }
    return mu_f, sigma_f, meta, details


def _build_run_overlay_telemetry_summary(diag_df: pd.DataFrame, weights_df: pd.DataFrame, cfg: MicroPipelineConfig) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    if not isinstance(diag_df, pd.DataFrame) or diag_df.empty:
        summary["probabilistic_mode_requested"] = str(getattr(cfg, "probabilistic_mode", "none"))
        summary["prob_source"] = str(getattr(cfg, "probabilistic_mode", "none"))
        summary["probabilistic_mode_effective"] = str(getattr(cfg, "probabilistic_mode", "none"))
        defaults = _probabilistic_backend_defaults(str(getattr(cfg, "probabilistic_mode", "none")))
        summary["prob_backend_family"] = str(defaults.get("prob_backend_family", "none"))
        summary["prob_backend_name"] = str(defaults.get("prob_backend_name", summary["prob_backend_family"]))
        summary["prob_backend_variant"] = str(defaults.get("prob_backend_variant", summary["prob_backend_name"]))
        summary["prob_backend_distinct"] = bool(defaults.get("prob_backend_distinct", False))
        summary["prob_interval_backend"] = str(defaults.get("prob_interval_backend", "none"))
        summary["prob_contract_status"] = "inactive" if str(getattr(cfg, "probabilistic_mode", "none")) == "none" else "unobserved"
        summary["signal_mode_effective"] = str(getattr(cfg, "signal_mode", "mu_sigma"))
        summary["feature_mu_active_share"] = 0.0
        summary["feature_mu_match_n_mean"] = np.nan
        summary["feature_mu_cols_used_mean"] = np.nan
        summary["feature_mu_distance_mean"] = np.nan
        summary["feature_mu_abs_tilt_mean"] = 0.0
        summary["feature_mu_selected_cols"] = ""
        summary["feature_mu_selected_family_counts"] = "{}"
        summary["regime_universe_enabled_rate"] = 1.0 if bool(getattr(cfg, "regime_dependent_universe_enabled", False)) else 0.0
        return summary

    def _col_mean(*names: str) -> float:
        for name in names:
            if name in diag_df.columns:
                s = pd.to_numeric(diag_df[name], errors="coerce")
                if s.notna().any():
                    return float(s.mean())
        return np.nan

    def _col_mode(*names: str) -> Optional[str]:
        for name in names:
            if name in diag_df.columns:
                vals = diag_df[name].dropna().astype(str)
                if not vals.empty:
                    try:
                        return str(vals.mode().iloc[0])
                    except Exception:
                        return str(vals.iloc[-1])
        return None

    summary.update({
        "probabilistic_mode_requested": _col_mode("probabilistic_mode_requested") or str(getattr(cfg, "probabilistic_mode", "none")),
        "prob_source": _col_mode("prob_source") or str(getattr(cfg, "probabilistic_mode", "none")),
        "probabilistic_mode_effective": _col_mode("probabilistic_mode_effective", "prob_source") or str(getattr(cfg, "probabilistic_mode", "none")),
        "prob_backend_family": _col_mode("prob_backend_family") or _probabilistic_backend_defaults(str(getattr(cfg, "probabilistic_mode", "none"))).get("prob_backend_family") or _col_mode("prob_source") or str(getattr(cfg, "probabilistic_mode", "none")),
        "prob_backend_name": _col_mode("prob_backend_name", "prob_backend_family") or _probabilistic_backend_defaults(str(getattr(cfg, "probabilistic_mode", "none"))).get("prob_backend_name"),
        "prob_backend_variant": _col_mode("prob_backend_variant", "prob_backend_name", "prob_backend_family") or _probabilistic_backend_defaults(str(getattr(cfg, "probabilistic_mode", "none"))).get("prob_backend_variant"),
        "prob_backend_distinct": bool(round(_col_mean("prob_backend_distinct"))) if np.isfinite(_col_mean("prob_backend_distinct")) else bool(_probabilistic_backend_defaults(str(getattr(cfg, "probabilistic_mode", "none"))).get("prob_backend_distinct", False)),
        "prob_interval_backend": _col_mode("prob_interval_backend", "prob_backend_family", "prob_source") or _probabilistic_backend_defaults(str(getattr(cfg, "probabilistic_mode", "none"))).get("prob_interval_backend") or str(getattr(cfg, "probabilistic_mode", "none")),
        "prob_contract_status": _col_mode("prob_contract_status") or ("inactive" if str(getattr(cfg, "probabilistic_mode", "none")) == "none" else "ok"),
        "prob_contract_note": _col_mode("prob_contract_note") or "",
        "prob_backend_parent_mode": _col_mode("prob_backend_parent_mode") or str(getattr(cfg, "probabilistic_mode", "none")),
        "prob_fallback_triggered_share": _col_mean("prob_fallback_triggered_share", "prob_fallback_triggered"),
        "prob_fallback_reason": _col_mode("prob_fallback_reason") or "",
        "signal_mode_effective": _col_mode("signal_mode_effective") or str(getattr(cfg, "signal_mode", "mu_sigma")),
        "feature_mu_active_share": _col_mean("feature_mu_active_share", "feature_mu_active"),
        "feature_mu_match_n_mean": _col_mean("feature_mu_match_n_mean", "feature_mu_match_n"),
        "feature_mu_cols_used_mean": _col_mean("feature_mu_cols_used_mean", "feature_mu_cols_used"),
        "feature_mu_distance_mean": _col_mean("feature_mu_distance_mean"),
        "feature_mu_abs_tilt_mean": _col_mean("feature_mu_abs_tilt_mean", "feature_mu_abs_tilt"),
        "feature_mu_selected_cols": _col_mode("feature_mu_selected_cols") or "",
        "feature_mu_selected_family_counts": _col_mode("feature_mu_selected_family_counts") or "{}",
        "regime_universe_enabled_rate": _col_mean("regime_dependent_universe_enabled", "regime_universe_enabled", "regime_universe_active"),
        "regime_universe_keep_frac_mean": _col_mean("regime_universe_keep_frac"),
        "regime_universe_selected_assets_mean": _col_mean("regime_universe_selected_assets", "selected_assets_after_regime_filter"),
        "regime_universe_filtered_assets_mean": _col_mean("regime_universe_filtered_assets", "filtered_assets_by_regime"),
        "regime_universe_selection_metric_mode": _col_mode("regime_universe_selection_metric") or str(getattr(cfg, "regime_universe_selection_metric", "mu_over_sigma")),
        "regime_universe_regime_mode": _col_mode("regime_universe_regime", "regime") or str(getattr(cfg, "regime_mode", "none")),
        "regime_universe_low_offensive_vol_tilt_mean": _col_mean("regime_universe_low_offensive_vol_tilt"),
        "regime_universe_high_defensive_vol_tilt_mean": _col_mean("regime_universe_high_defensive_vol_tilt"),
        "regime_universe_min_assets_mean": _col_mean("regime_universe_min_assets"),
        "regime_universe_max_assets_mean": _col_mean("regime_universe_max_assets"),
    })
    if isinstance(weights_df, pd.DataFrame) and not weights_df.empty and "prob_source" in weights_df.columns:
        vals = weights_df["prob_source"].dropna().astype(str)
        if not vals.empty:
            summary["prob_source_from_weights"] = str(vals.mode().iloc[0])
    return summary


def _apply_signal_model_extension(mu_hat_used: pd.Series, sigma_hat: pd.Series, *, cfg: MicroPipelineConfig) -> Tuple[pd.Series, Dict[str, Any], pd.DataFrame]:
    mode = str(getattr(cfg, "signal_mode", "mu_sigma") or "mu_sigma")
    mu = pd.to_numeric(mu_hat_used, errors="coerce").fillna(0.0).astype(float)
    sigma = pd.to_numeric(sigma_hat, errors="coerce").replace(0.0, np.nan).fillna(float(cfg.sigma_floor)).clip(lower=float(cfg.sigma_floor))
    base_score = (mu / sigma).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    out = mu.copy()
    details = pd.DataFrame(index=mu.index)
    effective_mode = mode

    def _safe_standardize(series: pd.Series) -> pd.Series:
        xs = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        std = float(xs.std(ddof=0)) if len(xs) else 0.0
        if std <= 1e-12:
            return pd.Series(0.0, index=xs.index, dtype="float64")
        return ((xs - float(xs.mean())) / std).astype(float)

    def _score_to_mu_signal(score_like: pd.Series, *, temperature: float = 1.0, blend: float = 1.0) -> pd.Series:
        score_z = _safe_standardize(score_like)
        temp = max(float(temperature), 1e-6)
        bounded = np.tanh(score_z / temp)
        transformed = pd.Series(bounded, index=score_z.index, dtype="float64") * sigma.reindex(score_z.index).fillna(float(cfg.sigma_floor))
        bb = float(np.clip(blend, 0.0, 1.0))
        return (((1.0 - bb) * mu.reindex(transformed.index).fillna(0.0)) + (bb * transformed)).astype(float)

    if mode == "huber_mu":
        delta = max(float(getattr(cfg, "huber_delta", 1.0)), 1e-8)
        out = mu.clip(lower=-delta * sigma, upper=delta * sigma)
    elif mode in {"lambdarank_like", "lambdarank_real"}:
        rank_pct = base_score.rank(method="average", pct=True)
        rank_centered = (rank_pct - 0.5) * 2.0
        details["signal_rank_pct"] = rank_pct.astype(float)
        details["signal_rank_centered"] = rank_centered.astype(float)
        score_like = rank_centered.astype(float)
        if mode == "lambdarank_real":
            gain_power = max(float(getattr(cfg, "lambdarank_real_gain_power", 1.0)), 1e-6)
            pair_power = max(float(getattr(cfg, "lambdarank_real_pair_power", 1.0)), 1e-6)
            score_like = np.sign(rank_centered) * (np.abs(rank_centered) ** gain_power)
            score_like = pd.Series(score_like, index=rank_centered.index, dtype="float64")
            score_like = score_like * (1.0 + np.abs(_safe_standardize(base_score)) ** (pair_power - 1.0))
        if bool(getattr(cfg, "multi_loss_enabled", False)):
            rank_w = max(float(getattr(cfg, "multi_loss_rank_weight", 0.0)), 0.0)
            dir_w = max(float(getattr(cfg, "multi_loss_direction_weight", 0.0)), 0.0)
            ret_w = max(float(getattr(cfg, "multi_loss_return_weight", 0.0)), 0.0)
            log_w = max(float(getattr(cfg, "multi_loss_logistic_weight", 0.0)), 0.0)
            topk_w = max(float(getattr(cfg, "multi_loss_topk_weight", 0.0)), 0.0)
            weight_sum = rank_w + dir_w + ret_w + log_w + topk_w
            rank_component = _safe_standardize(score_like)
            direction_component = _safe_standardize(np.sign(mu) * np.abs(rank_centered))
            return_component = _safe_standardize(base_score)
            logistic_component = _safe_standardize(1.0 / (1.0 + np.exp(-base_score.clip(-20, 20))) - 0.5)
            topk_component = _safe_standardize(rank_pct)
            details["multi_loss_rank_component"] = rank_component.astype(float)
            details["multi_loss_direction_component"] = direction_component.astype(float)
            details["multi_loss_return_component"] = return_component.astype(float)
            details["multi_loss_logistic_component"] = logistic_component.astype(float)
            details["multi_loss_topk_component"] = topk_component.astype(float)
            if weight_sum > 1e-12:
                score_like = (
                    rank_w * rank_component
                    + dir_w * direction_component
                    + ret_w * return_component
                    + log_w * logistic_component
                    + topk_w * topk_component
                ) / weight_sum
                score_like = pd.Series(score_like, index=mu.index, dtype="float64")
                details["multi_loss_score"] = score_like.astype(float)
                details["multi_loss_score_std"] = float(pd.to_numeric(score_like, errors="coerce").std(ddof=1)) if len(score_like) > 1 else 0.0
        temp_name = "lambdarank_real_temperature" if mode == "lambdarank_real" else "lambdarank_temperature"
        score_for_output = pd.Series(score_like, index=mu.index, dtype="float64")
        out = _score_to_mu_signal(
            score_for_output,
            temperature=float(getattr(cfg, temp_name, 1.0)),
            blend=float(getattr(cfg, "signal_score_blend", 1.0)),
        )
        details["score"] = score_for_output.astype(float)
    elif mode in {"directional_classifier", "logistic_loss"}:
        z = base_score * float(getattr(cfg, f"{mode}_confidence_scale", 1.0) if hasattr(cfg, f"{mode}_confidence_scale") else 1.0)
        prob_up = 1.0 / (1.0 + np.exp(-z.clip(-20, 20)))
        details["signal_prob_up"] = prob_up.astype(float)
        out = (prob_up - 0.5) * mu.abs()
    elif mode == "top_k_classifier":
        ranks = base_score.rank(method="average", pct=True)
        details["signal_top_k_prob"] = ranks.astype(float)
        out = (ranks - 0.5) * sigma
    elif mode == "quantile_loss":
        details["signal_quantile_pred"] = mu.astype(float)
        out = mu
    if "score" not in details.columns:
        details["score"] = base_score.astype(float)
    meta = {
        "signal_mode_requested": mode,
        "signal_mode_effective": effective_mode,
        "signal_score_mean": float(pd.to_numeric(details["score"], errors="coerce").mean()) if "score" in details.columns and len(details) else np.nan,
        "signal_score_std": float(pd.to_numeric(details["score"], errors="coerce").std(ddof=1)) if "score" in details.columns and len(details) > 1 else 0.0,
    }
    return out.astype(float), meta, details


def make_pure_cs_baseline_config(cfg: MicroPipelineConfig) -> MicroPipelineConfig:
    return replace(
        cfg,
        pure_cs_baseline=True,
        feature_mu_enabled=False,
        factor_model_active=False,
        factor_covariance_active=False,
        probabilistic_mode="none",
        probabilistic_overlay_strength=0.0,
        probabilistic_overlay_blend=0.0,
        signal_mode="mu_sigma",
        regime_dependent_universe_enabled=False,
        correlation_aware_allocation=False,
    )


def _resolve_pipeline_research_profile(cfg: MicroPipelineConfig) -> MicroPipelineConfig:
    return make_pure_cs_baseline_config(cfg) if bool(getattr(cfg, "pure_cs_baseline", False)) else cfg


def run_micro_investment_pipeline(asset_panel_df: pd.DataFrame, *, cfg: Optional[MicroPipelineConfig] = None) -> Dict[str, Any]:
    engine_t0 = time.perf_counter()
    engine_timing: Dict[str, float] = {
        "prepare_panel": 0.0,
        "feature_pair_prep": 0.0,
        "walk_forward_loop": 0.0,
        "mu_sigma_total": 0.0,
        "probabilistic_total": 0.0,
        "feature_mu_total": 0.0,
        "signal_model_total": 0.0,
        "regime_filter_total": 0.0,
        "covariance_sigma_total": 0.0,
        "weight_build_total": 0.0,
        "post_weights_total": 0.0,
        "diagnostics_total": 0.0,
        "finalize_total": 0.0,
        "n_oos_dates": 0.0,
        "n_loop_iterations": 0.0,
    }

    cfg = _resolve_pipeline_research_profile(cfg or MicroPipelineConfig())
    if asset_panel_df is None or not isinstance(asset_panel_df, pd.DataFrame) or asset_panel_df.empty:
        raise ValueError("asset_panel_df must be a non-empty DataFrame")
    req = {cfg.date_col, cfg.asset_col, cfg.return_col}
    if not req.issubset(asset_panel_df.columns):
        raise ValueError(f"asset_panel_df must contain columns {sorted(req)}")

    t_prepare = time.perf_counter()
    df = asset_panel_df.copy()
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="coerce")
    df[cfg.asset_col] = df[cfg.asset_col].astype(str)
    df[cfg.return_col] = pd.to_numeric(df[cfg.return_col], errors="coerce")
    df = df.dropna(subset=[cfg.date_col, cfg.asset_col, cfg.return_col]).sort_values([cfg.date_col, cfg.asset_col]).reset_index(drop=True)
    if df.empty:
        raise ValueError("asset_panel_df has no valid rows after cleaning")

    df = _ensure_feature_mu_fallback_columns(df, cfg)

    ret_mat = df.pivot(index=cfg.date_col, columns=cfg.asset_col, values=cfg.return_col).sort_index()
    assets = list(ret_mat.columns)
    if len(ret_mat) <= int(cfg.min_train):
        raise ValueError("Not enough observations for min_train")
    engine_timing["prepare_panel"] = float(time.perf_counter() - t_prepare)

    t_feat_pairs = time.perf_counter()
    feature_cols = _resolve_feature_mu_columns(df)
    historical_feature_pairs = _build_probabilistic_feature_pairs(df, cfg, feature_cols)
    engine_timing["feature_pair_prep"] = float(time.perf_counter() - t_feat_pairs)

    oos_dates = ret_mat.index[int(cfg.min_train):]
    max_oos_points = getattr(cfg, "max_oos_points", None)
    try:
        max_oos_points = int(max_oos_points) if max_oos_points is not None else None
    except Exception:
        max_oos_points = None
    if max_oos_points is not None and max_oos_points > 0 and len(oos_dates) > max_oos_points:
        oos_dates = oos_dates[-max_oos_points:]
    prev_weights = None
    cost_state = None
    oos_returns = []
    diag_rows: List[Dict[str, Any]] = []
    weight_rows: List[Dict[str, Any]] = []
    risk_rows: List[pd.DataFrame] = []
    corr_snaps: Dict[pd.Timestamp, pd.DataFrame] = {}
    sigma_snaps: Dict[pd.Timestamp, pd.DataFrame] = {}
    engine_timing["n_oos_dates"] = float(len(oos_dates))

    t_loop = time.perf_counter()
    for dt in oos_dates:
        engine_timing["n_loop_iterations"] += 1.0
        loc = ret_mat.index.get_loc(dt)
        train_window = ret_mat.iloc[:loc].copy()
        train_window = train_window.dropna(how="all", axis=1)
        if train_window.shape[0] < int(cfg.min_train):
            continue
        assets_now = list(train_window.columns)

        t_block = time.perf_counter()
        mu_hat, sigma_hat, regime = _forecast_mu_sigma_from_window(train_window, cfg)
        engine_timing["mu_sigma_total"] += float(time.perf_counter() - t_block)

        current_feature_state = _build_current_feature_state(df, pd.Timestamp(dt), cfg, feature_cols)

        t_block = time.perf_counter()
        probabilistic_enabled = bool(getattr(cfg, "enable_probabilistic", True)) and str(cfg.probabilistic_mode) != "none"
        prob_df = _build_probabilistic_forecast_from_window(
            train_window[assets_now],
            cfg,
            historical_feature_pairs=historical_feature_pairs,
            current_feature_state=current_feature_state,
        ) if probabilistic_enabled else None
        mu_prob, prob_meta = _integrate_probabilistic_into_mu(mu_hat.reindex(assets_now), prob_df, cfg)
        engine_timing["probabilistic_total"] += float(time.perf_counter() - t_block)

        t_block = time.perf_counter()
        mu_feat, feature_meta, feature_details = _apply_feature_conditioned_mu(
            mu_prob,
            sigma_hat=sigma_hat.reindex(assets_now),
            historical_feature_pairs=historical_feature_pairs,
            current_feature_state=current_feature_state,
            cfg=cfg,
        )
        engine_timing["feature_mu_total"] += float(time.perf_counter() - t_block)

        t_block = time.perf_counter()
        mu_signal_full, signal_meta, signal_details = _apply_signal_model_extension(mu_feat, sigma_hat.reindex(assets_now), cfg=cfg)
        engine_timing["signal_model_total"] += float(time.perf_counter() - t_block)

        t_block = time.perf_counter()
        mu_signal, sigma_hat_sub, regime_universe_meta, regime_universe_details = _apply_regime_dependent_universe_filter(
            mu_signal_full.reindex(assets_now),
            sigma_hat.reindex(assets_now),
            regime=regime,
            cfg=cfg,
        )
        engine_timing["regime_filter_total"] += float(time.perf_counter() - t_block)
        assets_selected = list(mu_signal.index.astype(str))

        t_block = time.perf_counter()
        corr_mat = _estimate_rolling_correlation_matrix(train_window[assets_selected], cfg, regime)
        sigma_fwd = _build_sigma_fwd(train_window[assets_selected], sigma_hat_sub.reindex(assets_selected), corr_mat, cfg, regime)
        engine_timing["covariance_sigma_total"] += float(time.perf_counter() - t_block)
        if bool(cfg.store_correlation_snapshots):
            corr_snaps[pd.Timestamp(dt)] = corr_mat.copy()
        if bool(cfg.store_sigma_fwd_snapshots):
            sigma_snaps[pd.Timestamp(dt)] = sigma_fwd.copy()

        prev_w_sub = None if prev_weights is None else prev_weights.reindex(assets_selected).fillna(0.0)
        t_block = time.perf_counter()
        weights_sub, raw_score, alloc_meta = _build_weights_from_mu_sigma(
            mu_signal.reindex(assets_selected),
            sigma_hat_sub.reindex(assets_selected),
            cfg=cfg,
            prev_weights=prev_w_sub,
            n_assets=len(assets_now),
            corr_mat=corr_mat,
            sigma_fwd=sigma_fwd,
        )
        engine_timing["weight_build_total"] += float(time.perf_counter() - t_block)

        feature_rank_base = pd.Series(np.nan, index=pd.Index(assets_now, dtype="object"), dtype="float64")
        feature_rank_tilted = pd.Series(np.nan, index=pd.Index(assets_now, dtype="object"), dtype="float64")
        feature_rank_signal = pd.Series(np.nan, index=pd.Index(assets_now, dtype="object"), dtype="float64")
        feature_rank_delta = pd.Series(np.nan, index=pd.Index(assets_now, dtype="object"), dtype="float64")
        feature_topk_base_assets: List[str] = []
        feature_topk_tilted_assets: List[str] = []
        feature_topk_signal_assets: List[str] = []
        feature_topk_overlap_vs_base = np.nan
        feature_topk_overlap_vs_tilted = np.nan
        feature_topk_changed_vs_base_n = np.nan
        feature_topk_changed_vs_tilted_n = np.nan

        try:
            rank_sigma = pd.to_numeric(sigma_hat.reindex(assets_now), errors="coerce").replace(0.0, np.nan).fillna(float(cfg.sigma_floor)).clip(lower=float(cfg.sigma_floor))
            base_score_for_rank = (
                pd.to_numeric(mu_prob.reindex(assets_now), errors="coerce").fillna(0.0).astype(float) / rank_sigma
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            tilted_score_for_rank = (
                pd.to_numeric(mu_feat.reindex(assets_now), errors="coerce").fillna(0.0).astype(float) / rank_sigma
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            signal_score_for_rank = (
                pd.to_numeric(mu_signal_full.reindex(assets_now), errors="coerce").fillna(0.0).astype(float) / rank_sigma
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

            feature_rank_base = base_score_for_rank.rank(method="min", ascending=False).astype(float)
            feature_rank_tilted = tilted_score_for_rank.rank(method="min", ascending=False).astype(float)
            feature_rank_signal = signal_score_for_rank.rank(method="min", ascending=False).astype(float)
            feature_rank_delta = (feature_rank_tilted - feature_rank_base).astype(float)

            effective_top_k_debug = alloc_meta.get("effective_top_k")
            if effective_top_k_debug is not None:
                k_debug = int(effective_top_k_debug)
                if k_debug > 0:
                    feature_topk_base_assets = [str(x) for x in base_score_for_rank.nlargest(min(k_debug, len(base_score_for_rank))).index]
                    feature_topk_tilted_assets = [str(x) for x in tilted_score_for_rank.nlargest(min(k_debug, len(tilted_score_for_rank))).index]
                    feature_topk_signal_assets = [str(x) for x in signal_score_for_rank.nlargest(min(k_debug, len(signal_score_for_rank))).index]
                    base_set = set(feature_topk_base_assets)
                    tilted_set = set(feature_topk_tilted_assets)
                    signal_set = set(feature_topk_signal_assets)
                    feature_topk_overlap_vs_base = float(len(base_set & signal_set) / k_debug)
                    feature_topk_overlap_vs_tilted = float(len(tilted_set & signal_set) / k_debug)
                    feature_topk_changed_vs_base_n = float(len(base_set ^ signal_set))
                    feature_topk_changed_vs_tilted_n = float(len(tilted_set ^ signal_set))
        except Exception:
            pass

        t_block = time.perf_counter()
        weights_sub, regime_derisk_mult = _apply_regime_derisk(weights_sub, regime=regime, cfg=cfg)
        weights_sub, vol_meta = _apply_vol_targeting(weights_sub, sigma_hat_sub.reindex(assets_selected), sigma_fwd, cfg)

        w = pd.Series(0.0, index=assets, dtype="float64")
        w.loc[weights_sub.index] = pd.to_numeric(weights_sub, errors="coerce").fillna(0.0).astype(float)
        realised = pd.to_numeric(ret_mat.loc[dt].reindex(assets), errors="coerce").fillna(0.0).astype(float)
        active_return_simple = float((w * realised).sum())
        cost_meta, cost_state = _evaluate_period_cost_tax(target_weights=w, realised_returns=realised, prev_state=cost_state, cfg=cfg)
        net_return_simple = float(cost_meta.get("net_portfolio_return_simple", active_return_simple))
        oos_returns.append(net_return_simple)
        engine_timing["post_weights_total"] += float(time.perf_counter() - t_block)

        diagnostics_enabled = bool(getattr(cfg, "enable_diagnostics", True))
        t_block = time.perf_counter()
        if diagnostics_enabled:
            corr_stats = _summarise_correlation_matrix(corr_mat)
            sigma_stats = _summarise_sigma_fwd(sigma_fwd)
            div_diag = _compute_diversification_diagnostics(train_window[assets_selected], weights_sub, corr_mat=corr_mat, sigma_fwd=sigma_fwd)
            risk_detail, risk_summary = _compute_risk_decomposition(weights_sub, sigma_fwd, corr_mat, cfg)
            if isinstance(risk_detail, pd.DataFrame) and not risk_detail.empty:
                tmp = risk_detail.copy()
                tmp.insert(0, "date", pd.Timestamp(dt))
                risk_rows.append(tmp.reset_index(drop=False).rename(columns={"index": "asset"}))

            selected_asset_returns = pd.to_numeric(realised.reindex(assets_selected), errors="coerce").dropna()
            diag_row = {
                "date": pd.Timestamp(dt),
                "regime": regime,
                "n_assets": int(len(assets_now)),
                "signal_mode": str(signal_meta.get("signal_mode_effective", getattr(cfg, "signal_mode", "mu_sigma"))),
                "feature_mu_topk_base_assets": "|".join(feature_topk_base_assets),
                "feature_mu_topk_tilted_assets": "|".join(feature_topk_tilted_assets),
                "feature_mu_topk_signal_assets": "|".join(feature_topk_signal_assets),
                "feature_mu_topk_overlap_vs_base": feature_topk_overlap_vs_base,
                "feature_mu_topk_overlap_vs_tilted": feature_topk_overlap_vs_tilted,
                "feature_mu_topk_changed_vs_base_n": feature_topk_changed_vs_base_n,
                "feature_mu_topk_changed_vs_tilted_n": feature_topk_changed_vs_tilted_n,
                "gross_portfolio_return_simple": active_return_simple,
                "net_portfolio_return_simple": net_return_simple,
                "portfolio_return_simple": active_return_simple,
                "active_return_simple": active_return_simple,
                "realised_return": active_return_simple,
                "mean_selected_realised_return": float(selected_asset_returns.mean()) if not selected_asset_returns.empty else np.nan,
                "turnover": float((w - (prev_weights.reindex(assets).fillna(0.0) if prev_weights is not None else 0.0)).abs().sum() / 2.0) if prev_weights is not None else np.nan,
                "regime_derisk_mult": float(regime_derisk_mult),
                **prob_meta,
                **feature_meta,
                **signal_meta,
                **regime_universe_meta,
                **alloc_meta,
                **vol_meta,
                **corr_stats,
                **sigma_stats,
                **div_diag,
                **risk_summary,
                **cost_meta,
            }
            diag_rows.append(diag_row)

            for asset in assets_now:
                asset_str = str(asset)
                row = {
                    "date": pd.Timestamp(dt),
                    "asset": asset_str,
                    "weight": float(w.loc[asset]),
                    "mu_hat_prob": float(pd.to_numeric(mu_prob.reindex([asset]), errors="coerce").iloc[0]) if asset in mu_prob.index else np.nan,
                    "mu_hat_feature_tilted_pre_signal": float(pd.to_numeric(mu_feat.reindex([asset]), errors="coerce").iloc[0]) if asset in mu_feat.index else np.nan,
                    "feature_mu_rank_base": float(pd.to_numeric(feature_rank_base.reindex([asset]), errors="coerce").iloc[0]) if asset in feature_rank_base.index else np.nan,
                    "feature_mu_rank_tilted": float(pd.to_numeric(feature_rank_tilted.reindex([asset]), errors="coerce").iloc[0]) if asset in feature_rank_tilted.index else np.nan,
                    "feature_mu_rank_signal": float(pd.to_numeric(feature_rank_signal.reindex([asset]), errors="coerce").iloc[0]) if asset in feature_rank_signal.index else np.nan,
                    "feature_mu_rank_delta": float(pd.to_numeric(feature_rank_delta.reindex([asset]), errors="coerce").iloc[0]) if asset in feature_rank_delta.index else np.nan,
                    "feature_mu_topk_base_flag": bool(asset_str in set(feature_topk_base_assets)),
                    "feature_mu_topk_tilted_flag": bool(asset_str in set(feature_topk_tilted_assets)),
                    "feature_mu_topk_signal_flag": bool(asset_str in set(feature_topk_signal_assets)),
                    "mu_hat": float(pd.to_numeric(mu_hat.reindex([asset]), errors="coerce").iloc[0]) if asset in mu_hat.index else np.nan,
                    "mu_hat_used": float(pd.to_numeric(mu_signal.reindex([asset]), errors="coerce").iloc[0]) if asset in mu_signal.index else np.nan,
                    "sigma_hat": float(pd.to_numeric(sigma_hat.reindex([asset]), errors="coerce").iloc[0]) if asset in sigma_hat.index else np.nan,
                    "score": float(pd.to_numeric(raw_score.reindex([asset]), errors="coerce").iloc[0]) if asset in raw_score.index else np.nan,
                    "realised_return": float(pd.to_numeric(realised.reindex([asset]), errors="coerce").iloc[0]) if asset in realised.index else np.nan,
                    "signal_mode_requested": str(getattr(cfg, "signal_mode", "mu_sigma")),
                    "signal_mode_effective": str(signal_meta.get("signal_mode_effective", getattr(cfg, "signal_mode", "mu_sigma"))),
                    "prob_source": str(prob_df.loc[asset, "prob_source"]) if isinstance(prob_df, pd.DataFrame) and asset in prob_df.index and "prob_source" in prob_df.columns else str(getattr(cfg, "probabilistic_mode", "none")),
                    "probabilistic_mode_effective": str(prob_df.loc[asset, "probabilistic_mode_effective"]) if isinstance(prob_df, pd.DataFrame) and asset in prob_df.index and "probabilistic_mode_effective" in prob_df.columns else str(prob_meta.get("probabilistic_mode_effective", prob_meta.get("prob_source", getattr(cfg, "probabilistic_mode", "none")))),
                    "regime_universe_selected": bool(asset_str in set(pd.Index(mu_signal.index).astype(str))),
                    "effective_top_k": alloc_meta.get("effective_top_k"),
                    "topk_binding": alloc_meta.get("topk_binding"),
                }
                if isinstance(prob_df, pd.DataFrame) and asset in prob_df.index:
                    for col in prob_df.columns:
                        row[col] = prob_df.loc[asset, col]
                if asset in signal_details.index:
                    for col in signal_details.columns:
                        target_col = f"signal_{col}" if str(col) == "score" else col
                        row[target_col] = signal_details.loc[asset, col]
                if not feature_details.empty and str(asset) in feature_details.index.astype(str):
                    feat_row = feature_details.loc[str(asset)] if str(asset) in feature_details.index else feature_details.loc[asset]
                    if isinstance(feat_row, pd.DataFrame):
                        feat_row = feat_row.iloc[-1]
                    for col in feature_details.columns:
                        row[col] = feat_row.get(col, np.nan)
                if not regime_universe_details.empty and asset_str in regime_universe_details.index.astype(str):
                    reg_row = regime_universe_details.loc[asset_str] if asset_str in regime_universe_details.index else regime_universe_details.loc[asset]
                    if isinstance(reg_row, pd.DataFrame):
                        reg_row = reg_row.iloc[-1]
                    for col in regime_universe_details.columns:
                        row[col] = reg_row.get(col, np.nan)
                weight_rows.append(row)
        engine_timing["diagnostics_total"] += float(time.perf_counter() - t_block)
        prev_weights = w.copy()

    engine_timing["walk_forward_loop"] = float(time.perf_counter() - t_loop)

    t_finalize = time.perf_counter()
    oos_index = pd.Index(oos_dates[: len(oos_returns)], name="date")
    result = {
        "config": cfg,
        "config_dict": config_to_dict(cfg),
        "config_fingerprint": config_fingerprint(cfg),
        "oos_returns_simple": pd.Series(oos_returns, index=oos_index, dtype="float64"),
        "weights_df": pd.DataFrame(weight_rows),
        "diagnostics_df": pd.DataFrame(diag_rows),
        "risk_contributions_df": pd.concat(risk_rows, ignore_index=True) if risk_rows else pd.DataFrame(),
        "correlation_snapshots": corr_snaps,
        "sigma_fwd_snapshots": sigma_snaps,
    }
    result["performance_summary"] = build_performance_summary(result)
    result["risk_summary"] = build_risk_summary(result)
    result["cost_summary"] = build_cost_summary(result)
    result["overlay_telemetry_summary"] = _build_run_overlay_telemetry_summary(result.get("diagnostics_df", pd.DataFrame()), result.get("weights_df", pd.DataFrame()), cfg)
    result["simplified_global_params_summary"] = build_simplified_global_params_summary(result)

    try:
        report = build_run_report(result, run_name="run")
    except Exception:
        report = {}
    result["report"] = report
    engine_timing["finalize_total"] = float(time.perf_counter() - t_finalize)
    engine_timing["total_engine"] = float(time.perf_counter() - engine_t0)
    result["engine_timing"] = {k: float(v) for k, v in engine_timing.items()}
    return result

def run_investment_projection(
    *,
    current_savings: float,
    invest_fraction: float,
    weekly_savings: Optional[float] = None,
    monthly_contribution: Optional[float] = None,
    horizon_years: int,
    risk_profile: str,
    n_sims: int = 1000,
    seed: Optional[int] = None,
    override_annual_return: Optional[float] = None,
    override_annual_vol: Optional[float] = None,
    realised_monthly_returns: Optional[np.ndarray] = None,
    bootstrap_method: str = "iid",
    block_len: int = DEFAULT_BLOCK_LEN,
    goal_amount: Optional[float] = None,
    micro_asset_panel_df: Optional[pd.DataFrame] = None,
    micro_cfg: Optional[MicroPipelineConfig] = None,
    simulation_granularity: Literal["monthly", "daily_hybrid"] = "monthly",
    daily_steps_per_month: int = 21,
    daily_path_noise_scale: float = 0.35,
) -> Dict[str, Any]:
    monthly_contrib = float(monthly_contribution) if monthly_contribution is not None else float(weekly_to_monthly(float(weekly_savings or 0.0)))
    horizon_months = max(int(horizon_years) * 12, 1)
    rng = np.random.default_rng(seed)
    annual_return = float(override_annual_return if override_annual_return is not None else PROFILE_LIBRARY.get(risk_profile, PROFILE_LIBRARY["Balanced"])["fallback_return_annual"])
    annual_vol = float(override_annual_vol if override_annual_vol is not None else PROFILE_LIBRARY.get(risk_profile, PROFILE_LIBRARY["Balanced"])["fallback_vol_annual"])
    mu_m = annual_return / 12.0
    sigma_m = annual_vol / np.sqrt(12.0)

    use_daily_hybrid = (
        str(simulation_granularity or "monthly").strip().lower() == "daily_hybrid"
        and realised_monthly_returns is not None
        and len(realised_monthly_returns) > 0
    )
    effective_daily_steps = max(int(daily_steps_per_month), 1)

    def _sample_monthly_draws() -> np.ndarray:
        if realised_monthly_returns is not None and len(realised_monthly_returns) > 0:
            rets = np.asarray(realised_monthly_returns, dtype="float64")
            if str(bootstrap_method).strip().lower() == "block":
                out = np.empty((n_sims, horizon_months), dtype="float64")
                for i in range(n_sims):
                    t = 0
                    while t < horizon_months:
                        start = int(rng.integers(0, max(len(rets) - max(int(block_len), 1), 1)))
                        block = rets[start:start + max(int(block_len), 1)]
                        for r in block:
                            if t >= horizon_months:
                                break
                            out[i, t] = float(r)
                            t += 1
                return out
            return rng.choice(rets, size=(n_sims, horizon_months), replace=True).astype("float64")
        return rng.normal(mu_m, sigma_m, size=(n_sims, horizon_months)).astype("float64")

    monthly_draws = _sample_monthly_draws()
    monthly_paths = np.empty((n_sims, horizon_months + 1), dtype="float64")
    monthly_paths[:, 0] = float(current_savings)
    for t in range(horizon_months):
        monthly_paths[:, t + 1] = (monthly_paths[:, t] + monthly_contrib) * (1.0 + monthly_draws[:, t])

    daily_paths = None
    if use_daily_hybrid:
        horizon_days = horizon_months * effective_daily_steps
        daily_paths = np.empty((n_sims, horizon_days + 1), dtype="float64")
        daily_paths[:, 0] = float(current_savings)
        for i in range(n_sims):
            wealth = float(current_savings)
            day_idx = 0
            for month_idx in range(horizon_months):
                wealth = wealth + monthly_contrib
                daily_returns = _expand_monthly_return_to_daily_path(
                    float(monthly_draws[i, month_idx]),
                    effective_daily_steps,
                    rng,
                    noise_scale=float(daily_path_noise_scale),
                )
                for d_ret in daily_returns:
                    wealth = wealth * (1.0 + float(d_ret))
                    day_idx += 1
                    daily_paths[i, day_idx] = wealth

    terminal = monthly_paths[:, -1]
    total_contrib = float(current_savings + monthly_contrib * horizon_months)

    observed_annual_vol = annual_vol
    if realised_monthly_returns is not None and len(realised_monthly_returns) > 1:
        try:
            observed_annual_vol = float(np.nanstd(np.asarray(realised_monthly_returns, dtype="float64"), ddof=1) * np.sqrt(12.0))
        except Exception:
            observed_annual_vol = annual_vol

    summary_source = "historical_engine_oos_daily_hybrid" if use_daily_hybrid else ("historical_engine_oos" if realised_monthly_returns is not None and len(realised_monthly_returns) > 0 else ("micro_pipeline" if micro_asset_panel_df is not None else "parametric"))

    summary = GrowthSummary(
        profile=risk_profile,
        source=summary_source,
        horizon_months=horizon_months,
        n_sims=int(n_sims),
        initial_invested=float(current_savings),
        monthly_contribution=float(monthly_contrib),
        total_contributed=total_contrib,
        expected_terminal=float(np.mean(terminal)),
        median_terminal=float(np.median(terminal)),
        p10_terminal=float(np.quantile(terminal, 0.10)),
        p90_terminal=float(np.quantile(terminal, 0.90)),
        expected_profit=float(np.mean(terminal) - total_contrib),
        median_profit=float(np.median(terminal) - total_contrib),
        p10_profit=float(np.quantile(terminal, 0.10) - total_contrib),
        p90_profit=float(np.quantile(terminal, 0.90) - total_contrib),
        probability_of_loss_vs_contributions=float(np.mean(terminal < total_contrib)),
        probability_of_finishing_below_initial=float(np.mean(terminal < float(current_savings))),
        probability_of_reaching_goal=float(np.mean(terminal >= float(goal_amount))) if goal_amount is not None else None,
        expected_max_drawdown=np.nan,
        drawdown_p10=np.nan,
        annual_return_assumption=float(annual_return),
        annual_vol_assumption=float(observed_annual_vol),
    )
    out = {
        "wealth_paths": monthly_paths,
        "summary": summary.to_dict(),
        "source": summary.source,
        "simulation_granularity": "daily_hybrid" if use_daily_hybrid else "monthly",
        "daily_steps_per_month": int(effective_daily_steps),
    }
    if daily_paths is not None:
        out["wealth_paths_daily"] = daily_paths
        out["daily_hybrid_metadata"] = {
            "daily_steps_per_month": int(effective_daily_steps),
            "daily_path_noise_scale": float(daily_path_noise_scale),
            "monthly_paths_shape": tuple(int(x) for x in monthly_paths.shape),
            "daily_paths_shape": tuple(int(x) for x in daily_paths.shape),
        }
    if micro_asset_panel_df is not None:
        out["micro_pipeline_result"] = run_micro_investment_pipeline(micro_asset_panel_df, cfg=micro_cfg or MicroPipelineConfig())
    return out


def run_contribution_scenarios(*, contributions: List[float], **kwargs: Any) -> pd.DataFrame:
    rows = []
    for c in contributions:
        out = run_investment_projection(monthly_contribution=float(c), **kwargs)
        summ = out["summary"]
        rows.append({"Contribution": float(c), "MedianTerminal": summ["median_terminal"], "ExpectedTerminal": summ["expected_terminal"]})
    return pd.DataFrame(rows)


def _first_finite_optional(*values: Any) -> Optional[float]:
    for v in values:
        try:
            f = float(v)
        except Exception:
            continue
        if np.isfinite(f):
            return f
    return None
