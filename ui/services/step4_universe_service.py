"""
Step 4 universe and market-data service for LifeBudget Micro.

This module contains the non-UI logic behind the Investment Strategy Lab setup.
It resolves investment philosophy defaults, universe sizes, universe strategies,
asset labels, custom asset baskets, cached Yahoo deployment panels, feature
engineering, diagnostics payloads, and the Step 4 payload handed to Step 5.

The Step 4 renderer should stay mostly layout-focused. This service owns the
state transformations, asset-universe construction, panel cache resolution and
session_state updates needed by the Strategy Engine.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple
import hashlib
import json
import os
import time

import pandas as pd
import streamlit as st

from src.features import (
    DailyFeatureConfig,
    add_cross_sectional_features,
    add_daily_intramonth_state_features,
    build_intramonth_monthly_features,
    validate_daily_asset_panel,
)
from src.investment import build_asset_group_dataframe, build_asset_group_summary
from src.market_data import build_return_panel_from_prices, download_yahoo_price_panel
from ui.state.keys import (
    ASSET_AUTO_ADJUST,
    ASSET_END_DATE,
    ASSET_PANEL_DF,
    ASSET_PANEL_ERROR,
    ASSET_PANEL_READY,
    ASSET_PANEL_SOURCE_LABEL,
    ASSET_RETURN_FREQUENCY,
    ASSET_SOURCE_MODE,
    ASSET_START_DATE,
    ASSET_UPLOADED_FILE,
    CUSTOM_UNIVERSE_TEXT,
    INVESTMENT_CONTEXT,
    INVESTMENT_MONTHLY_CONTRIBUTION,
    INVESTMENT_PHILOSOPHY,
    INVESTMENT_PHILOSOPHY_BUNDLE_SUMMARY,
    INVESTMENT_PROJECTION_PROFILE_HINT,
    INVESTMENT_WEEKLY_EQUIVALENT,
    LAST_RECOMMENDATION_CANDIDATE_ASSETS,
    LAST_USED_UNIVERSE_ASSETS,
    PLANNING_SNAPSHOT,
    RECOMMENDED_UNIVERSE_ASSETS,
    UNIVERSE_CUSTOM_ENABLED,
    UNIVERSE_CUSTOM_ENABLED_SOURCE,
    UNIVERSE_SIMPLE_STRATEGY_TEMPLATE,
    UNIVERSE_SIMPLE_STYLE_PRESET,
    UNIVERSE_SIZE,
    UNIVERSE_STRATEGY,
)


STEP4_PANEL_SIGNATURE_KEY = "step4_panel_input_signature"
STEP4_PANEL_TIMINGS_KEY = "step4_panel_timings"
STEP4_PANEL_BUILD_NONCE_KEY = "step4_panel_build_nonce"
STEP4_PANEL_LAST_BUILD_TRIGGER_KEY = "step4_panel_last_build_trigger"

# Cached deployment panels.
# Streamlit Cloud should use these files first so the public demo does not
# depend on live Yahoo/yfinance availability or rate limits. Local/dev can force
# live Yahoo by setting LIFEBUDGET_USE_DEPLOYMENT_PANEL_FIRST=0.
DEPLOYMENT_ASSET_PANEL_PATH = Path(
    os.getenv("LIFEBUDGET_DEPLOYMENT_ASSET_PANEL", "data/deployment_asset_panel.csv.gz")
)
DEPLOYMENT_DAILY_RETURNS_PATH = Path(
    os.getenv("LIFEBUDGET_DEPLOYMENT_DAILY_RETURNS", "data/deployment_daily_returns.csv.gz")
)
DEPLOYMENT_WEEKLY_RETURNS_PATH = Path(
    os.getenv("LIFEBUDGET_DEPLOYMENT_WEEKLY_RETURNS", "data/deployment_weekly_returns.csv.gz")
)
USE_DEPLOYMENT_PANEL_FIRST = str(
    os.getenv("LIFEBUDGET_USE_DEPLOYMENT_PANEL_FIRST", "1")
).strip().lower() not in {"0", "false", "no"}

# Raw return panels are diagnostics/export artifacts only; Step 5 continues to
# consume the prepared panel stored in ASSET_PANEL_DF.
STEP4_RAW_DAILY_PANEL_KEY = "_step4_raw_daily_return_panel_df"
STEP4_RAW_WEEKLY_PANEL_KEY = "_step4_raw_weekly_return_panel_df"
STEP4_RAW_PANEL_SOURCE_KEY = "_step4_raw_return_panel_source"


def _serialize_feature_cfg(cfg: DailyFeatureConfig) -> str:
    """Serialize the feature-engineering config for cache/signature stability."""
    payload = {
        "date_col": cfg.date_col,
        "asset_col": cfg.asset_col,
        "return_col": cfg.return_col,
        "monthly_return_name": cfg.monthly_return_name,
        "weekly_return_name": cfg.weekly_return_name,
        "ewma_halflife_short": cfg.ewma_halflife_short,
        "ewma_halflife_medium": cfg.ewma_halflife_medium,
        "ewma_halflife_long": cfg.ewma_halflife_long,
        "rolling_short": cfg.rolling_short,
        "rolling_medium": cfg.rolling_medium,
        "rolling_long": cfg.rolling_long,
        "momentum_short": cfg.momentum_short,
        "momentum_medium": cfg.momentum_medium,
        "momentum_long": cfg.momentum_long,
        "min_obs_per_month": cfg.min_obs_per_month,
        "min_obs_for_window": cfg.min_obs_for_window,
        "annualisation_days": cfg.annualisation_days,
        "monthly_annualisation_days": cfg.monthly_annualisation_days,
        "clip_simple_return_low": cfg.clip_simple_return_low,
    }
    return json.dumps(payload, sort_keys=True, default=str)


def build_step4_panel_input_signature(
    *,
    selected_assets: List[str],
    candidate_assets: List[str],
    source_mode: str,
    start_date: Any,
    end_date: Any,
    frequency: str,
    auto_adjust: bool,
    uploaded_file: Any,
) -> str:
    """Build a compact hash for the market-data panel inputs.

    Step 4 uses this signature to decide whether the cached/prepared panel can
    be reused or whether the asset panel must be resolved again. The signature
    includes selected assets, candidate assets, data source, date range,
    frequency, upload metadata and the feature-engineering configuration.
    """
    uploaded_name = str(getattr(uploaded_file, "name", "") or "")
    uploaded_size = int(getattr(uploaded_file, "size", 0) or 0) if uploaded_file is not None else 0
    payload = {
        "selected_assets": [str(x).strip().upper() for x in list(selected_assets or []) if str(x).strip()],
        "candidate_assets": [str(x).strip().upper() for x in list(candidate_assets or []) if str(x).strip()],
        "source_mode": str(source_mode or "yahoo").lower(),
        "start_date": str(start_date or ""),
        "end_date": str(end_date or ""),
        "frequency": str(frequency or "monthly").lower(),
        "auto_adjust": bool(auto_adjust),
        "uploaded_name": uploaded_name,
        "uploaded_size": uploaded_size,
        "feature_cfg": _serialize_feature_cfg(DailyFeatureConfig()),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(encoded.encode("utf-8")).hexdigest()[:16]


def _coerce_timing_value(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def format_step4_timing_summary(timings: Dict[str, Any] | None) -> str:
    """Format Step 4 panel timing metadata for compact diagnostics.

    This is intentionally text-based because it is used in captions, debug
    blocks and cached-panel traceability notes.
    """
    t = dict(timings or {})
    if not t:
        return ""

    parts: List[str] = []
    seconds_keys = [
        "deployment_panel_read_seconds",
        "deployment_raw_daily_read_seconds",
        "deployment_raw_weekly_read_seconds",
        "deployment_panel_total_seconds",
        "download_yahoo_seconds",
        "return_panel_seconds",
        "validate_seconds",
        "validate_asset_panel_seconds",
        "intramonth_features_seconds",
        "monthly_aggregation_seconds",
        "cross_sectional_seconds",
        "total_feature_pipeline_seconds",
        "resolve_total_seconds",
    ]
    for key in seconds_keys:
        if key in t:
            parts.append(f"{key}={_coerce_timing_value(t.get(key, 0.0)):.2f}s")

    count_keys = [
        "deployment_panel_rows",
        "deployment_panel_assets",
        "deployment_raw_daily_rows",
        "deployment_raw_daily_assets",
        "deployment_raw_weekly_rows",
        "deployment_raw_weekly_assets",
        "raw_daily_rows",
        "raw_daily_assets",
        "raw_weekly_rows",
        "raw_weekly_assets",
    ]
    for key in count_keys:
        if key in t:
            try:
                parts.append(f"{key}={int(float(t.get(key, 0) or 0))}")
            except Exception:
                parts.append(f"{key}={t.get(key)}")

    cache_parts = []
    if "feature_pipeline_cache_reused" in t:
        cache_parts.append(f"feature_cache_reused={bool(t.get('feature_pipeline_cache_reused', False))}")
    if "reuse_existing_panel" in t:
        cache_parts.append(f"reuse_existing_panel={bool(t.get('reuse_existing_panel', False))}")
    if "deployment_cache_mode" in t:
        cache_parts.append(f"deployment_cache_mode={t.get('deployment_cache_mode')}")
    if cache_parts:
        parts.extend(cache_parts)

    return " · ".join(parts)


UNIVERSE_SIZES = [12, 25, 50, 75, 100, 150, 250]

UNIVERSE_STRATEGY_CORE = "Core multi-asset"
UNIVERSE_STRATEGY_DIVERSIFIED = "Diversified global beta"
UNIVERSE_STRATEGY_EQUITY = "Equity heavy"
UNIVERSE_STRATEGY_DEFENSIVE = "Defensive income"
UNIVERSE_STRATEGY_REAL_ASSETS = "Real assets tilt"
UNIVERSE_STRATEGY_QUALITY = "Quality / Dividend equity"
UNIVERSE_STRATEGY_LOW_VOL = "Low volatility / capital preservation"
UNIVERSE_STRATEGY_LONG_HISTORY = "Long-history / projection-friendly"

STRATEGIES = [
    UNIVERSE_STRATEGY_CORE,
    UNIVERSE_STRATEGY_DIVERSIFIED,
    UNIVERSE_STRATEGY_EQUITY,
    UNIVERSE_STRATEGY_DEFENSIVE,
    UNIVERSE_STRATEGY_REAL_ASSETS,
    UNIVERSE_STRATEGY_QUALITY,
    UNIVERSE_STRATEGY_LOW_VOL,
    UNIVERSE_STRATEGY_LONG_HISTORY,
]

INVESTMENT_PHILOSOPHY_OPTIONS = ["Growth", "Balanced", "Defensive"]

STRATEGY_TEMPLATE_OPTIONS = [
    "Balanced Risk-Controlled",
    "Core Ranking",
    "Hybrid Research",
]

STYLE_PRESET_OPTIONS = [
    "Balanced",
    "Growth",
    "Defensive",
    "Research",
]

PHILOSOPHY_STRATEGY_COMBOS = {
    "Growth": {
        "recommended": ("Core Ranking", "Growth"),
        "allowed": [
            ("Core Ranking", "Growth"),
            ("Core Ranking", "Balanced"),
            ("Balanced Risk-Controlled", "Growth"),
            ("Balanced Risk-Controlled", "Balanced"),
            ("Hybrid Research", "Growth"),
            ("Hybrid Research", "Research"),
            ("Hybrid Research", "Balanced"),
        ],
    },
    "Balanced": {
        "recommended": ("Balanced Risk-Controlled", "Balanced"),
        "allowed": [
            ("Balanced Risk-Controlled", "Defensive"),
            ("Balanced Risk-Controlled", "Balanced"),
            ("Balanced Risk-Controlled", "Growth"),
            ("Core Ranking", "Balanced"),
            ("Core Ranking", "Growth"),
            ("Hybrid Research", "Balanced"),
            ("Hybrid Research", "Growth"),
            ("Hybrid Research", "Research"),
        ],
    },
    "Defensive": {
        "recommended": ("Balanced Risk-Controlled", "Defensive"),
        "allowed": [
            ("Balanced Risk-Controlled", "Defensive"),
            ("Balanced Risk-Controlled", "Balanced"),
            ("Core Ranking", "Defensive"),
            ("Core Ranking", "Balanced"),
            ("Hybrid Research", "Defensive"),
            ("Hybrid Research", "Research"),
        ],
    },
}


SEMANTIC_SLIDER_KEYS = {
    "risk_appetite": "step5_sem_risk_appetite",
    "drawdown_protection": "step5_sem_drawdown_protection",
    "diversification_vs_concentration": "step5_sem_diversification",
    "overlay_intensity": "step5_sem_overlay_intensity",
    "stability_vs_responsiveness": "step5_sem_stability",
    "confidence_in_signal": "step5_sem_confidence_signal",
    "low_turnover_vs_adaptive": "step5_sem_turnover_style",
    "simplicity_vs_sophistication": "step5_sem_simplicity",
}
SEMANTIC_TOUCHED_FLAG = "step5_semantic_user_touched"
SEMANTIC_LAST_SIGNATURE = "step5_semantic_last_seed_signature"

TEMPLATE_BASE = {
    "Core Ranking": {
        "risk_appetite": 0.68,
        "drawdown_protection": 0.34,
        "diversification_vs_concentration": 0.68,
        "overlay_intensity": 0.32,
        "stability_vs_responsiveness": 0.46,
        "confidence_in_signal": 0.72,
        "low_turnover_vs_adaptive": 0.62,
        "simplicity_vs_sophistication": 0.54,
    },
    "Balanced Risk-Controlled": {
        "risk_appetite": 0.50,
        "drawdown_protection": 0.62,
        "diversification_vs_concentration": 0.55,
        "overlay_intensity": 0.42,
        "stability_vs_responsiveness": 0.60,
        "confidence_in_signal": 0.56,
        "low_turnover_vs_adaptive": 0.40,
        "simplicity_vs_sophistication": 0.56,
    },
    "Hybrid Research": {
        "risk_appetite": 0.56,
        "drawdown_protection": 0.44,
        "diversification_vs_concentration": 0.50,
        "overlay_intensity": 0.72,
        "stability_vs_responsiveness": 0.52,
        "confidence_in_signal": 0.62,
        "low_turnover_vs_adaptive": 0.60,
        "simplicity_vs_sophistication": 0.38,
    },
}

STYLE_MODIFIERS = {
    "Conservative": {
        "risk_appetite": -0.16,
        "drawdown_protection": 0.16,
        "diversification_vs_concentration": -0.08,
        "stability_vs_responsiveness": 0.10,
        "confidence_in_signal": -0.06,
        "low_turnover_vs_adaptive": -0.10,
        "simplicity_vs_sophistication": 0.06,
    },
    "Balanced": {},
    "Growth": {
        "risk_appetite": 0.16,
        "drawdown_protection": -0.12,
        "diversification_vs_concentration": 0.10,
        "stability_vs_responsiveness": -0.08,
        "confidence_in_signal": 0.10,
        "low_turnover_vs_adaptive": 0.10,
        "simplicity_vs_sophistication": 0.04,
    },
    "Defensive": {
        "risk_appetite": -0.24,
        "drawdown_protection": 0.22,
        "diversification_vs_concentration": -0.10,
        "overlay_intensity": -0.06,
        "stability_vs_responsiveness": 0.18,
        "confidence_in_signal": -0.06,
        "low_turnover_vs_adaptive": -0.12,
        "simplicity_vs_sophistication": 0.08,
    },
    "Research": {
        "risk_appetite": 0.04,
        "drawdown_protection": -0.04,
        "overlay_intensity": 0.18,
        "stability_vs_responsiveness": -0.06,
        "confidence_in_signal": 0.06,
        "low_turnover_vs_adaptive": 0.14,
        "simplicity_vs_sophistication": -0.18,
    },
}

INVESTMENT_PHILOSOPHY_BUNDLES = {
    "Growth": {
        "universe_strategy_preferences": [
            UNIVERSE_STRATEGY_EQUITY,
            UNIVERSE_STRATEGY_QUALITY,
            UNIVERSE_STRATEGY_DIVERSIFIED,
            UNIVERSE_STRATEGY_CORE,
        ],
        "strategy_template": "Core Ranking",
        "style_preset": "Growth",
        "projection_profile": "Growth",
        "default_universe_size": 50,
        "recommended_sizes": [50, 75, 100],
        "description": "Higher-upside posture using a more equity-led universe and a more offensive strategy bundle.",
    },
    "Balanced": {
        "universe_strategy_preferences": [
            UNIVERSE_STRATEGY_CORE,
            UNIVERSE_STRATEGY_DIVERSIFIED,
            UNIVERSE_STRATEGY_LONG_HISTORY,
            UNIVERSE_STRATEGY_QUALITY,
        ],
        "strategy_template": "Balanced Risk-Controlled",
        "style_preset": "Balanced",
        "projection_profile": "Balanced",
        "default_universe_size": 25,
        "recommended_sizes": [25, 50, 75],
        "description": "Default all-round posture balancing return-seeking, diversification and risk control.",
    },
    "Defensive": {
        "universe_strategy_preferences": [
            UNIVERSE_STRATEGY_DEFENSIVE,
            UNIVERSE_STRATEGY_LOW_VOL,
            UNIVERSE_STRATEGY_LONG_HISTORY,
            UNIVERSE_STRATEGY_CORE,
        ],
        "strategy_template": "Balanced Risk-Controlled",
        "style_preset": "Defensive",
        "projection_profile": "Conservative",
        "default_universe_size": 12,
        "recommended_sizes": [12, 25, 50],
        "description": "More risk-aware posture with a more defensive universe bias and tighter strategy bundle.",
    },
}

UNIVERSE_ALLOWED_STRATEGIES_BY_SIZE = {
    12: [
        UNIVERSE_STRATEGY_CORE,
        UNIVERSE_STRATEGY_EQUITY,
        UNIVERSE_STRATEGY_DEFENSIVE,
        UNIVERSE_STRATEGY_REAL_ASSETS,
        UNIVERSE_STRATEGY_QUALITY,
        UNIVERSE_STRATEGY_LOW_VOL,
        UNIVERSE_STRATEGY_LONG_HISTORY,
    ],
    25: STRATEGIES,
    50: [
        UNIVERSE_STRATEGY_CORE,
        UNIVERSE_STRATEGY_DIVERSIFIED,
        UNIVERSE_STRATEGY_EQUITY,
        UNIVERSE_STRATEGY_DEFENSIVE,
        UNIVERSE_STRATEGY_REAL_ASSETS,
        UNIVERSE_STRATEGY_QUALITY,
        UNIVERSE_STRATEGY_LOW_VOL,
    ],
    75: [
        UNIVERSE_STRATEGY_CORE,
        UNIVERSE_STRATEGY_DIVERSIFIED,
        UNIVERSE_STRATEGY_EQUITY,
        UNIVERSE_STRATEGY_DEFENSIVE,
        UNIVERSE_STRATEGY_REAL_ASSETS,
        UNIVERSE_STRATEGY_QUALITY,
        UNIVERSE_STRATEGY_LOW_VOL,
    ],
    100: [
        UNIVERSE_STRATEGY_CORE,
        UNIVERSE_STRATEGY_DIVERSIFIED,
        UNIVERSE_STRATEGY_EQUITY,
        UNIVERSE_STRATEGY_DEFENSIVE,
        UNIVERSE_STRATEGY_REAL_ASSETS,
        UNIVERSE_STRATEGY_QUALITY,
        UNIVERSE_STRATEGY_LOW_VOL,
    ],
    150: [
        UNIVERSE_STRATEGY_CORE,
        UNIVERSE_STRATEGY_DIVERSIFIED,
        UNIVERSE_STRATEGY_EQUITY,
        UNIVERSE_STRATEGY_DEFENSIVE,
        UNIVERSE_STRATEGY_REAL_ASSETS,
        UNIVERSE_STRATEGY_QUALITY,
        UNIVERSE_STRATEGY_LOW_VOL,
    ],
    250: [
        UNIVERSE_STRATEGY_CORE,
        UNIVERSE_STRATEGY_DIVERSIFIED,
        UNIVERSE_STRATEGY_EQUITY,
        UNIVERSE_STRATEGY_DEFENSIVE,
        UNIVERSE_STRATEGY_REAL_ASSETS,
        UNIVERSE_STRATEGY_QUALITY,
        UNIVERSE_STRATEGY_LOW_VOL,
    ],
}

STRATEGY_SEEDS = {
    UNIVERSE_STRATEGY_CORE: ["SPY","QQQ","IWM","EFA","EEM","TLT","IEF","LQD","HYG","GLD","DBC","VNQ","DIA","MDY","VGK","EWJ","EWU","XLK","XLF","XLV","XLI","XLP","XLY","XLE","XLB","XLU","VWO","BND","AGG","MUB","TIP","SHY","SOXX","SMH","QUAL","SCHD","VIG","USMV","MTUM","VEA","ACWI","EMB","ICLN","SLV","IAU","REET","VNQI","RWX","DBA","USO","UNG","URA","LIT","TAN","XBI","IBB","RSP","VOO","VTI"],
    UNIVERSE_STRATEGY_DIVERSIFIED: ["ACWI","VEA","VWO","SPY","QQQ","IWM","EFA","EEM","VGK","EWJ","EWU","TLT","IEF","TIP","LQD","HYG","BND","AGG","MUB","GLD","SLV","DBC","DBA","USO","UNG","VNQ","REET","XLK","XLF","XLV","XLI","XLP","XLY","XLE","XLB","XLU","SCHD","VIG","QUAL","USMV","MTUM","EMB","SMH","SOXX","ICLN","LIT","TAN","DIA","MDY","RSP","VTI","VOO","EWG","EWC","EWA","INDA","FXI","KWEB"],
    UNIVERSE_STRATEGY_EQUITY: ["SPY","VOO","VTI","QQQ","IWM","MDY","DIA","RSP","EFA","VEA","EEM","VWO","ACWI","VGK","EWJ","EWU","INDA","FXI","KWEB","XLK","XLF","XLV","XLI","XLP","XLY","XLE","XLB","XLU","XLC","SMH","SOXX","XBI","IBB","ICLN","LIT","TAN","SCHD","VIG","QUAL","USMV","MTUM","VLUE","SPYG","SPYV","SCHG","SCHV"],
    UNIVERSE_STRATEGY_DEFENSIVE: ["TLT","IEF","IEI","SHY","VGSH","SGOV","BIL","TIP","STIP","LQD","VCIT","VCSH","BND","AGG","SCHZ","MUB","VMBS","MBB","SCHD","VIG","NOBL","USMV","SPLV","VNQ","GLD","IAU","SPY","EFA","XLU","XLP","XLV"],
    UNIVERSE_STRATEGY_REAL_ASSETS: ["GLD","IAU","SLV","PPLT","DBC","DBA","DBB","CPER","USO","UNG","URA","KRBN","VNQ","REET","RWX","SCHH","IYR","XLRE","ICLN","TAN","LIT","WOOD","CUT","MOO","PHO","FIW","XLE","XLB","SPY","EEM"],
    UNIVERSE_STRATEGY_QUALITY: ["SCHD","VIG","DGRO","DGRW","DVY","NOBL","QUAL","USMV","SPYV","VLUE","VTV","IWD","SCHV","FNDX","RDVY","XLV","XLP","XLU","VNQ","SPY","EFA","VEA","LQD","IEF","VGSH","GLD","IAU","QQQ","VOO","VTI"],
    UNIVERSE_STRATEGY_LOW_VOL: ["SGOV","BIL","SHV","VGSH","SCHO","SCHR","SHY","IEI","IEF","TIP","STIP","TFLO","LQD","VCIT","VCSH","AGG","BND","SCHZ","MUB","VMBS","MBB","SCHD","USMV","SPLV","VIG","NOBL","GLD","IAU","VNQ","SPY"],
    UNIVERSE_STRATEGY_LONG_HISTORY: ["SPY","QQQ","DIA","MDY","IWM","EFA","EEM","VGK","EWJ","EWU","TLT","IEF","SHY","TIP","LQD","HYG","BND","AGG","GLD","SLV","DBC","VNQ","XLF","XLK","XLV","XLP","XLY","XLI","XLE","XLB"],
}


ASSET_DISPLAY_NAMES: Dict[str, str] = {
    "ACWI": "Global equities",
    "AGG": "Aggregate bonds",
    "BIL": "1-3 month Treasury bills",
    "BND": "Total bond market",
    "CPER": "Copper",
    "CUT": "Timber & forestry",
    "DBA": "Agriculture commodities",
    "DBB": "Base metals",
    "DBC": "Broad commodities",
    "DGRO": "Dividend growth equities",
    "DGRW": "Dividend growth equities",
    "DIA": "Dow Jones Industrial Average",
    "DVY": "High dividend yield equities",
    "EEM": "Emerging markets equities",
    "EFA": "Developed markets ex-US",
    "EMB": "Emerging market bonds",
    "EWA": "Australia equities",
    "EWC": "Canada equities",
    "EWG": "Germany equities",
    "EWJ": "Japan equities",
    "EWU": "UK equities",
    "FIW": "Water industry",
    "FNDX": "Fundamental large-cap equities",
    "FXI": "China large-cap equities",
    "GLD": "Gold",
    "HYG": "High-yield corporate bonds",
    "IAU": "Gold",
    "IBB": "Biotechnology",
    "ICLN": "Clean energy",
    "IEF": "7-10 year Treasury bonds",
    "IEI": "3-7 year Treasury bonds",
    "INDA": "India equities",
    "IWD": "Russell 1000 value equities",
    "IWM": "Russell 2000 small-cap equities",
    "IYR": "US real estate",
    "KRBN": "Carbon allowances",
    "KWEB": "China internet equities",
    "LIT": "Lithium & battery technology",
    "LQD": "Investment-grade corporate bonds",
    "MBB": "Mortgage-backed bonds",
    "MDY": "US mid-cap equities",
    "MOO": "Agribusiness",
    "MTUM": "Momentum factor equities",
    "MUB": "Municipal bonds",
    "NOBL": "Dividend aristocrats",
    "PHO": "Water infrastructure",
    "PPLT": "Platinum",
    "QQQ": "Nasdaq-100 equities",
    "QUAL": "Quality factor equities",
    "RDVY": "Rising dividend equities",
    "REET": "Global real estate",
    "RSP": "Equal-weight S&P 500",
    "RWX": "International real estate",
    "SCHD": "Dividend equity",
    "SCHG": "Large-cap growth equities",
    "SCHH": "US REITs",
    "SCHO": "Short-term Treasury bonds",
    "SCHR": "Intermediate Treasury bonds",
    "SCHV": "Large-cap value equities",
    "SCHZ": "Aggregate bond market",
    "SGOV": "0-3 month Treasury bills",
    "SHV": "Short-term Treasury bills",
    "SHY": "1-3 year Treasury bonds",
    "SLV": "Silver",
    "SMH": "Semiconductors",
    "SOXX": "Semiconductors",
    "SPLV": "Low-volatility S&P 500",
    "SPY": "S&P 500 equities",
    "SPYG": "S&P 500 growth equities",
    "SPYV": "S&P 500 value equities",
    "STIP": "Short-term inflation-linked bonds",
    "TAN": "Solar energy",
    "TFLO": "Floating-rate Treasury bonds",
    "TIP": "Inflation-linked Treasury bonds",
    "TLT": "20+ year Treasury bonds",
    "UNG": "Natural gas",
    "URA": "Uranium & nuclear energy",
    "USMV": "Minimum-volatility equities",
    "USO": "Oil",
    "VCIT": "Intermediate corporate bonds",
    "VCSH": "Short-term corporate bonds",
    "VEA": "Developed markets ex-US",
    "VGK": "Europe equities",
    "VGSH": "Short-term Treasury bonds",
    "VIG": "Dividend growth equities",
    "VLUE": "Value factor equities",
    "VMBS": "Mortgage-backed bonds",
    "VNQ": "US real estate",
    "VNQI": "Global ex-US real estate",
    "VOO": "S&P 500 equities",
    "VTI": "Total US stock market",
    "VTV": "Large-cap value equities",
    "VWO": "Emerging markets equities",
    "WOOD": "Global timber & forestry",
    "XBI": "Biotechnology",
    "XLB": "Materials sector",
    "XLC": "Communication services sector",
    "XLE": "Energy sector",
    "XLF": "Financials sector",
    "XLI": "Industrials sector",
    "XLK": "Technology sector",
    "XLP": "Consumer staples sector",
    "XLRE": "Real estate sector",
    "XLU": "Utilities sector",
    "XLV": "Healthcare sector",
    "XLY": "Consumer discretionary sector",
}


def normalize_asset_ticker(value: Any) -> str:
    """Return the canonical ticker even if the UI label is pasted back as 'GLD (Gold)'."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    if "(" in raw:
        raw = raw.split("(", 1)[0].strip()
    return raw.upper()


def asset_display_name(ticker: Any) -> str:
    """Return the plain-English asset name used in the Step 4 UI."""
    symbol = normalize_asset_ticker(ticker)
    return str(ASSET_DISPLAY_NAMES.get(symbol, "")).strip()


def asset_display_label(ticker: Any) -> str:
    """Display helper for Streamlit widgets: 'GLD (Gold)' while value stays 'GLD'."""
    symbol = normalize_asset_ticker(ticker)
    if not symbol:
        return ""
    name = asset_display_name(symbol)
    return f"{symbol} ({name})" if name else symbol


def _unique_preserve_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items or []:
        ticker = normalize_asset_ticker(item)
        if ticker and ticker not in seen:
            seen.add(ticker)
            out.append(ticker)
    return out


def coerce_snapshot() -> dict:
    snapshot = st.session_state.get(PLANNING_SNAPSHOT, {}) or {}
    return dict(snapshot) if isinstance(snapshot, dict) else {}


def store_investment_context(plan_snapshot: dict) -> dict:
    """Persist the Step 1-3 contribution bridge used by Step 4 and later modules."""
    monthly_contribution = float(max(plan_snapshot.get("target_a_monthly", 0.0) or 0.0, 0.0))
    weekly_equivalent = float(max(plan_snapshot.get("target_a_weekly", 0.0) or 0.0, 0.0))
    baseline_monthly = float(max(plan_snapshot.get("baseline_savings_monthly", 0.0) or 0.0, 0.0))
    required_cut_monthly = float(max(plan_snapshot.get("required_cut_a_monthly", 0.0) or 0.0, 0.0))
    ctx = {
        "monthly_contribution": monthly_contribution,
        "weekly_equivalent": weekly_equivalent,
        "baseline_monthly": baseline_monthly,
        "required_cut_monthly": required_cut_monthly,
        "structural_deficit": bool(plan_snapshot.get("structural_deficit", False)),
    }
    st.session_state[INVESTMENT_CONTEXT] = ctx
    st.session_state[INVESTMENT_MONTHLY_CONTRIBUTION] = monthly_contribution
    st.session_state[INVESTMENT_WEEKLY_EQUIVALENT] = weekly_equivalent
    return ctx


def get_canonical_investment_philosophy() -> str:
    value = str(st.session_state.get(INVESTMENT_PHILOSOPHY, "Balanced") or "Balanced")
    if value not in INVESTMENT_PHILOSOPHY_OPTIONS:
        value = "Balanced"
        st.session_state[INVESTMENT_PHILOSOPHY] = value
    return value


def build_bundle_summary(philosophy: Any) -> dict:
    philosophy_name = str(philosophy or "Balanced")
    return dict(INVESTMENT_PHILOSOPHY_BUNDLES.get(philosophy_name, INVESTMENT_PHILOSOPHY_BUNDLES["Balanced"]))


def strategy_combo_status(philosophy: Any, template: Any, style: Any) -> str:
    philosophy_name = get_canonical_investment_philosophy() if philosophy is None else str(philosophy or "Balanced")
    rules = PHILOSOPHY_STRATEGY_COMBOS.get(philosophy_name, PHILOSOPHY_STRATEGY_COMBOS["Balanced"])
    combo = (str(template or ""), str(style or ""))
    if combo == tuple(rules.get("recommended", ())):
        return "recommended"
    if combo in list(rules.get("allowed", [])):
        return "allowed"
    return "blocked"


def recommended_strategy_combo_for_philosophy(philosophy: Any) -> Tuple[str, str]:
    philosophy_name = str(philosophy or "Balanced")
    rules = PHILOSOPHY_STRATEGY_COMBOS.get(philosophy_name, PHILOSOPHY_STRATEGY_COMBOS["Balanced"])
    rec = tuple(rules.get("recommended", ("Balanced Risk-Controlled", "Balanced")))
    return str(rec[0]), str(rec[1])


def allowed_strategy_templates_for_philosophy(philosophy: Any) -> List[str]:
    philosophy_name = str(philosophy or "Balanced")
    rules = PHILOSOPHY_STRATEGY_COMBOS.get(philosophy_name, PHILOSOPHY_STRATEGY_COMBOS["Balanced"])
    out: List[str] = []
    for template, _style in list(rules.get("allowed", [])):
        if template not in out:
            out.append(str(template))
    return out or list(STRATEGY_TEMPLATE_OPTIONS)


def allowed_style_presets_for_philosophy(philosophy: Any, template: Any | None = None) -> List[str]:
    philosophy_name = str(philosophy or "Balanced")
    template_name = str(template or "").strip()
    rules = PHILOSOPHY_STRATEGY_COMBOS.get(philosophy_name, PHILOSOPHY_STRATEGY_COMBOS["Balanced"])
    out: List[str] = []
    for t_name, style in list(rules.get("allowed", [])):
        if template_name and str(t_name) != template_name:
            continue
        if style not in out:
            out.append(str(style))
    if not out and template_name:
        rec_template, rec_style = recommended_strategy_combo_for_philosophy(philosophy_name)
        if template_name == rec_template:
            out = [rec_style]
    return out or list(STYLE_PRESET_OPTIONS)


def resolve_semantic_slider_defaults(template: Any, style: Any) -> Dict[str, float]:
    template_name = str(template or "Balanced Risk-Controlled")
    style_name = str(style or "Balanced")
    base = dict(TEMPLATE_BASE.get(template_name, TEMPLATE_BASE["Balanced Risk-Controlled"]))
    mods = dict(STYLE_MODIFIERS.get(style_name, {}))
    out: Dict[str, float] = {}
    for key, value in base.items():
        out[key] = max(0.0, min(1.0, float(value) + float(mods.get(key, 0.0))))
    return out


def semantic_seed_signature(template: Any, style: Any) -> str:
    return f"{str(template or '')}||{str(style or '')}"


def mark_semantic_sliders_dirty() -> None:
    st.session_state[SEMANTIC_TOUCHED_FLAG] = True


def reset_semantic_slider_touch_state() -> None:
    st.session_state[SEMANTIC_TOUCHED_FLAG] = False


def apply_semantic_slider_defaults(template: Any, style: Any, force: bool = False) -> Dict[str, float]:
    defaults = resolve_semantic_slider_defaults(template, style)
    touched = bool(st.session_state.get(SEMANTIC_TOUCHED_FLAG, False))
    signature = semantic_seed_signature(template, style)
    last_signature = str(st.session_state.get(SEMANTIC_LAST_SIGNATURE, "") or "")
    should_apply = bool(force or (not touched) or (signature != last_signature))
    if should_apply:
        for logical_key, widget_key in SEMANTIC_SLIDER_KEYS.items():
            st.session_state[widget_key] = float(defaults.get(logical_key, 0.5))
        st.session_state[SEMANTIC_LAST_SIGNATURE] = signature
        if force:
            st.session_state[SEMANTIC_TOUCHED_FLAG] = False
    return defaults


def maybe_apply_initial_universe_size_default() -> None:
    if UNIVERSE_SIZE in st.session_state:
        return
    philosophy = get_canonical_investment_philosophy()
    st.session_state[UNIVERSE_SIZE] = int(build_bundle_summary(philosophy)["default_universe_size"])


def recommended_universe_sizes_for_philosophy(philosophy: Any) -> List[int]:
    bundle = build_bundle_summary(philosophy)
    out: List[int] = []
    for raw in list(bundle.get("recommended_sizes", [])):
        try:
            value = int(raw)
        except Exception:
            continue
        if value in UNIVERSE_SIZES and value not in out:
            out.append(value)
    return out or [int(bundle.get("default_universe_size", 25) or 25)]


def preferred_universe_strategies_for_philosophy(philosophy: Any) -> List[str]:
    bundle = build_bundle_summary(philosophy)
    prefs: List[str] = []
    for raw in list(bundle.get("universe_strategy_preferences", [])):
        name = str(raw or "").strip()
        if name and name in STRATEGIES and name not in prefs:
            prefs.append(name)
    return prefs or [UNIVERSE_STRATEGY_CORE]


def resolve_recommended_strategy_for_size(philosophy: Any, universe_size: Any) -> str:
    allowed = allowed_universe_strategies_for_size(universe_size)
    prefs = preferred_universe_strategies_for_philosophy(philosophy)
    return next((x for x in prefs if x in allowed), allowed[0] if allowed else UNIVERSE_STRATEGY_CORE)


def build_universe_size_status(philosophy: Any, universe_size: Any) -> Dict[str, Any]:
    bundle = build_bundle_summary(philosophy)
    try:
        size_value = int(universe_size)
    except Exception:
        size_value = int(bundle.get("default_universe_size", 25) or 25)
    ideal = int(bundle.get("default_universe_size", 25) or 25)
    recommended = recommended_universe_sizes_for_philosophy(philosophy)
    if size_value == ideal:
        status = "Ideal default"
    elif size_value in recommended:
        status = "Recommended subset"
    else:
        status = "Outside recommended subset"
    return {
        "current_size": int(size_value),
        "ideal_size": int(ideal),
        "recommended_sizes": list(recommended),
        "status": str(status),
    }


def apply_investment_philosophy_bundle(philosophy: Any) -> dict:
    """Apply the coherent Step 4 defaults for the selected investment philosophy.

    The bundle seeds Step 5 strategy template/style, projection profile and
    semantic slider defaults so later screens stay aligned with the selected
    Growth/Balanced/Defensive posture.
    """
    philosophy_name = str(philosophy or "Balanced")
    if philosophy_name not in INVESTMENT_PHILOSOPHY_OPTIONS:
        philosophy_name = "Balanced"
    bundle = build_bundle_summary(philosophy_name)
    st.session_state[INVESTMENT_PHILOSOPHY] = philosophy_name
    rec_template, rec_style = recommended_strategy_combo_for_philosophy(philosophy_name)
    st.session_state[UNIVERSE_SIMPLE_STRATEGY_TEMPLATE] = str(rec_template)
    st.session_state[UNIVERSE_SIMPLE_STYLE_PRESET] = str(rec_style)
    st.session_state["step5_template"] = str(rec_template)
    st.session_state["step5_style"] = str(rec_style)
    st.session_state[INVESTMENT_PROJECTION_PROFILE_HINT] = str(bundle["projection_profile"])
    st.session_state[INVESTMENT_PHILOSOPHY_BUNDLE_SUMMARY] = dict(bundle)
    reset_semantic_slider_touch_state()
    apply_semantic_slider_defaults(rec_template, rec_style, force=True)
    st.session_state[SEMANTIC_LAST_SIGNATURE] = semantic_seed_signature(rec_template, rec_style)
    allowed = allowed_universe_strategies_for_size(st.session_state.get(UNIVERSE_SIZE, bundle["default_universe_size"]))
    preferred = next(
        (x for x in bundle["universe_strategy_preferences"] if x in allowed),
        allowed[0] if allowed else UNIVERSE_STRATEGY_CORE,
    )
    current_strategy = str(st.session_state.get(UNIVERSE_STRATEGY, preferred) or preferred)
    if current_strategy not in allowed or UNIVERSE_STRATEGY not in st.session_state:
        st.session_state[UNIVERSE_STRATEGY] = preferred
    return bundle


def allowed_universe_strategies_for_size(universe_size: Any) -> List[str]:
    try:
        size_value = int(universe_size)
    except Exception:
        size_value = 25
    return list(UNIVERSE_ALLOWED_STRATEGIES_BY_SIZE.get(size_value, STRATEGIES))


def validate_universe_inputs(universe_size: Any, strategy_name: Any) -> Tuple[int, str]:
    try:
        size_value = int(universe_size)
    except Exception:
        size_value = 25
    if size_value not in UNIVERSE_SIZES:
        size_value = 25
    strategy_value = str(strategy_name or UNIVERSE_STRATEGY_CORE)
    allowed = allowed_universe_strategies_for_size(size_value)
    if strategy_value not in allowed:
        strategy_value = allowed[0] if allowed else UNIVERSE_STRATEGY_CORE
    return size_value, strategy_value


def parse_custom_assets(raw_text: Any) -> List[str]:
    parts = str(raw_text or "").replace("\n", ",").split(",")
    return _unique_preserve_order([normalize_asset_ticker(x) for x in parts if normalize_asset_ticker(x)])


def _strategy_expansion_order(strategy_value: str) -> List[str]:
    """Return strategy-compatible fallback order for filling larger target baskets.

    Some focused strategy seeds are smaller than the public target sizes. The
    UI should not show a target such as 50 or 100 and then silently return a
    partial basket, so larger baskets are filled with compatible strategies in
    a deterministic order.
    """
    strategy = str(strategy_value or UNIVERSE_STRATEGY_CORE)
    expansion_map: Dict[str, List[str]] = {
        UNIVERSE_STRATEGY_CORE: [
            UNIVERSE_STRATEGY_DIVERSIFIED,
            UNIVERSE_STRATEGY_QUALITY,
            UNIVERSE_STRATEGY_LONG_HISTORY,
            UNIVERSE_STRATEGY_DEFENSIVE,
            UNIVERSE_STRATEGY_EQUITY,
            UNIVERSE_STRATEGY_REAL_ASSETS,
            UNIVERSE_STRATEGY_LOW_VOL,
        ],
        UNIVERSE_STRATEGY_DIVERSIFIED: [
            UNIVERSE_STRATEGY_CORE,
            UNIVERSE_STRATEGY_EQUITY,
            UNIVERSE_STRATEGY_QUALITY,
            UNIVERSE_STRATEGY_DEFENSIVE,
            UNIVERSE_STRATEGY_REAL_ASSETS,
            UNIVERSE_STRATEGY_LONG_HISTORY,
            UNIVERSE_STRATEGY_LOW_VOL,
        ],
        UNIVERSE_STRATEGY_EQUITY: [
            UNIVERSE_STRATEGY_QUALITY,
            UNIVERSE_STRATEGY_DIVERSIFIED,
            UNIVERSE_STRATEGY_CORE,
            UNIVERSE_STRATEGY_REAL_ASSETS,
            UNIVERSE_STRATEGY_LONG_HISTORY,
        ],
        UNIVERSE_STRATEGY_DEFENSIVE: [
            UNIVERSE_STRATEGY_LOW_VOL,
            UNIVERSE_STRATEGY_QUALITY,
            UNIVERSE_STRATEGY_LONG_HISTORY,
            UNIVERSE_STRATEGY_CORE,
            UNIVERSE_STRATEGY_DIVERSIFIED,
        ],
        UNIVERSE_STRATEGY_REAL_ASSETS: [
            UNIVERSE_STRATEGY_CORE,
            UNIVERSE_STRATEGY_DIVERSIFIED,
            UNIVERSE_STRATEGY_EQUITY,
            UNIVERSE_STRATEGY_QUALITY,
            UNIVERSE_STRATEGY_LONG_HISTORY,
        ],
        UNIVERSE_STRATEGY_QUALITY: [
            UNIVERSE_STRATEGY_DEFENSIVE,
            UNIVERSE_STRATEGY_LOW_VOL,
            UNIVERSE_STRATEGY_LONG_HISTORY,
            UNIVERSE_STRATEGY_CORE,
            UNIVERSE_STRATEGY_EQUITY,
            UNIVERSE_STRATEGY_DIVERSIFIED,
        ],
        UNIVERSE_STRATEGY_LOW_VOL: [
            UNIVERSE_STRATEGY_DEFENSIVE,
            UNIVERSE_STRATEGY_LONG_HISTORY,
            UNIVERSE_STRATEGY_QUALITY,
            UNIVERSE_STRATEGY_CORE,
            UNIVERSE_STRATEGY_DIVERSIFIED,
        ],
        UNIVERSE_STRATEGY_LONG_HISTORY: [
            UNIVERSE_STRATEGY_CORE,
            UNIVERSE_STRATEGY_DEFENSIVE,
            UNIVERSE_STRATEGY_QUALITY,
            UNIVERSE_STRATEGY_LOW_VOL,
            UNIVERSE_STRATEGY_DIVERSIFIED,
            UNIVERSE_STRATEGY_EQUITY,
        ],
    }
    fallbacks = list(expansion_map.get(strategy, []))
    for name in STRATEGIES:
        if name != strategy and name not in fallbacks:
            fallbacks.append(name)
    return fallbacks


def _expanded_universe_source(universe_size: Any, strategy_name: Any) -> Tuple[List[str], int, str]:
    size_value, strategy_value = validate_universe_inputs(universe_size, strategy_name)
    primary_seed = _unique_preserve_order(
        list(STRATEGY_SEEDS.get(strategy_value, STRATEGY_SEEDS[UNIVERSE_STRATEGY_CORE]))
    )
    source: List[str] = list(primary_seed)
    for fallback_strategy in _strategy_expansion_order(strategy_value):
        source.extend(list(STRATEGY_SEEDS.get(fallback_strategy, []) or []))
    source = _unique_preserve_order(source)
    return source, min(len(primary_seed), int(size_value)), strategy_value


def build_generated_universe(universe_size: Any, strategy_name: Any) -> List[str]:
    """Build the primary Step 4 asset universe for the selected size/strategy."""
    size_value, strategy_value = validate_universe_inputs(universe_size, strategy_name)
    source, _primary_count, _strategy_value = _expanded_universe_source(size_value, strategy_value)
    return list(source[: int(size_value)])


def build_universe_generation_summary(universe_size: Any, strategy_name: Any) -> Dict[str, Any]:
    size_value, strategy_value = validate_universe_inputs(universe_size, strategy_name)
    generated = build_generated_universe(size_value, strategy_value)
    _source, primary_count, _strategy_value = _expanded_universe_source(size_value, strategy_value)
    prepared_count = len(generated)
    filler_count = max(0, prepared_count - primary_count)
    return {
        "requested_size": int(size_value),
        "strategy": str(strategy_value),
        "prepared_count": int(prepared_count),
        "primary_count": int(min(primary_count, prepared_count)),
        "filler_count": int(filler_count),
        "expanded": bool(filler_count > 0),
        "complete": bool(prepared_count >= int(size_value)),
    }


def build_strategy_candidate_pool(universe_size: Any, strategy_name: Any) -> List[str]:
    """Build the wider candidate pool used by Step 5 universe-size search.

    Important:
    - canonical UI sizes remain validated elsewhere (12/25/50/75/...);
    - the optimisation layer may pass intermediate refinement sizes such as
      44, 47, 53 or 56;
    - the requested size itself is preserved for the target pool size. We only
      use validate_universe_inputs() to coerce the strategy name into a known
      strategy.
    """
    try:
        requested_size = int(universe_size)
    except Exception:
        requested_size = 25
    requested_size = max(1, requested_size)

    _canonical_size, strategy_value = validate_universe_inputs(requested_size, strategy_name)

    primary_seed = list(STRATEGY_SEEDS.get(strategy_value, STRATEGY_SEEDS[UNIVERSE_STRATEGY_CORE]))
    fallback_union: List[str] = []
    for strategy_seed in STRATEGY_SEEDS.values():
        fallback_union.extend(list(strategy_seed or []))

    pool_source = _unique_preserve_order(primary_seed + fallback_union)
    if not pool_source:
        return []

    extra = max(12, requested_size // 2)
    target_size = requested_size + extra
    return pool_source[: min(len(pool_source), target_size)]


def resolve_universe_selection(
    universe_size: Any,
    strategy_name: Any,
    custom_enabled: bool,
    custom_text: Any,
) -> Tuple[List[str], str]:
    """Resolve the active universe assets and explain where they came from.

    Priority:
    1. Manual custom basket when enabled and valid.
    2. Step 5 recommendation-preserved basket when it matches current size/style.
    3. Generated Step 4 universe.
    """
    generated = build_generated_universe(universe_size, strategy_name)
    size_value, strategy_value = validate_universe_inputs(universe_size, strategy_name)

    if bool(custom_enabled):
        custom_assets = parse_custom_assets(custom_text)
        if custom_assets:
            return custom_assets, "custom"

    try:
        source = str(st.session_state.get(UNIVERSE_CUSTOM_ENABLED_SOURCE, "") or "")
        rec_ctx = st.session_state.get("step5_recommended_universe_context_v1", {})
        rec_ctx = dict(rec_ctx) if isinstance(rec_ctx, dict) else {}
        rec_assets = _unique_preserve_order(
            [normalize_asset_ticker(x) for x in list(st.session_state.get(RECOMMENDED_UNIVERSE_ASSETS, []) or [])]
        )
        ctx_size = int(rec_ctx.get("universe_size", size_value) or size_value)
        ctx_strategy = str(rec_ctx.get("universe_strategy", strategy_value) or strategy_value)
        if (
            source == "recommendation"
            and rec_assets
            and len(rec_assets) == int(size_value)
            and ctx_size == int(size_value)
            and ctx_strategy == str(strategy_value)
        ):
            return rec_assets, "recommendation"
    except Exception:
        pass

    return generated, "generated"


def build_universe_preview_text(selected_assets: List[str], preview_limit: int = 8) -> str:
    assets = [normalize_asset_ticker(x) for x in list(selected_assets or []) if normalize_asset_ticker(x)]
    if not assets:
        return "—"
    labels = [asset_display_label(x) for x in assets]
    if len(labels) <= preview_limit:
        return ", ".join(labels)
    return f"{', '.join(labels[:preview_limit])} ..."


def build_universe_mix(selected_assets: List[str]) -> Tuple[pd.DataFrame, dict]:
    assets = [str(x).strip().upper() for x in list(selected_assets or []) if str(x).strip()]
    if not assets:
        return pd.DataFrame(columns=["group", "count"]), {"n_assets": 0, "group_count": 0, "classified_share": 0.0}
    mix_df = build_asset_group_dataframe(assets)
    if not isinstance(mix_df, pd.DataFrame) or mix_df.empty:
        return pd.DataFrame(columns=["group", "count"]), {"n_assets": len(assets), "group_count": 0, "classified_share": 0.0}
    if "ticker" in mix_df.columns and "asset" not in mix_df.columns:
        mix_df = mix_df.rename(columns={"ticker": "asset"})
    summary = build_asset_group_summary(assets)
    work = (
        mix_df.groupby("group", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "group"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return work, dict(summary or {})


def build_universe_mix_detail(selected_assets: List[str]) -> pd.DataFrame:
    assets = [normalize_asset_ticker(x) for x in list(selected_assets or []) if normalize_asset_ticker(x)]
    if not assets:
        return pd.DataFrame(columns=["ticker", "name", "group", "subgroup"])
    detail_df = build_asset_group_dataframe(assets)
    if not isinstance(detail_df, pd.DataFrame) or detail_df.empty:
        return pd.DataFrame(columns=["ticker", "name", "group", "subgroup"])
    work = detail_df.copy()
    if "asset" in work.columns and "ticker" not in work.columns:
        work = work.rename(columns={"asset": "ticker"})
    keep_cols = [c for c in ["ticker", "group", "subgroup"] if c in work.columns]
    work = work[keep_cols].copy()
    if "ticker" in work.columns:
        work["ticker"] = work["ticker"].apply(normalize_asset_ticker)
        work["name"] = work["ticker"].apply(asset_display_name)
    for col in ["group", "subgroup"]:
        if col in work.columns:
            work[col] = work[col].astype(str)
    final_cols = [c for c in ["ticker", "name", "group", "subgroup"] if c in work.columns]
    sort_cols = [c for c in ["group", "subgroup", "ticker"] if c in work.columns]
    return work[final_cols].sort_values(sort_cols, ascending=True).reset_index(drop=True)


def sync_step4_state(
    *,
    philosophy: str,
    universe_size: int,
    strategy_name: str,
    custom_enabled: bool,
    custom_text: str,
    selected_assets: List[str],
    selection_source: str,
) -> None:
    """Persist Step 4 universe choices and candidate pools into session state."""
    st.session_state[INVESTMENT_PHILOSOPHY] = str(philosophy)
    st.session_state[UNIVERSE_SIZE] = int(universe_size)
    st.session_state[UNIVERSE_STRATEGY] = str(strategy_name)
    st.session_state[UNIVERSE_CUSTOM_ENABLED] = bool(custom_enabled)
    st.session_state[CUSTOM_UNIVERSE_TEXT] = str(custom_text or "")
    st.session_state[UNIVERSE_CUSTOM_ENABLED_SOURCE] = str(selection_source)
    st.session_state[LAST_USED_UNIVERSE_ASSETS] = list(selected_assets)
    st.session_state[RECOMMENDED_UNIVERSE_ASSETS] = list(selected_assets)
    st.session_state[LAST_RECOMMENDATION_CANDIDATE_ASSETS] = list(
        build_strategy_candidate_pool(universe_size, strategy_name)
    )


def build_bridge_explanation(context: dict) -> str:
    monthly = float(context.get("monthly_contribution", 0.0) or 0.0)
    weekly = float(context.get("weekly_equivalent", 0.0) or 0.0)
    extra = float(context.get("required_cut_monthly", 0.0) or 0.0)
    return (
        "The investment module uses the monthly contribution view as the main unit: "
        f"£{monthly:,.0f}/month. Equivalent to £{weekly:,.0f}/week in the earlier budgeting steps. "
        f"Extra cut needed vs baseline: £{extra:,.0f}/month."
    )


@st.cache_data(show_spinner=False)
def _cached_download_yahoo_asset_panel(
    tickers: Tuple[str, ...],
    start_date: str,
    end_date: str,
    frequency: str,
    auto_adjust: bool,
) -> pd.DataFrame:
    last_error: Exception | None = None
    tickers = tuple(str(x).strip().upper() for x in tickers if str(x).strip())
    try:
        prices = download_yahoo_price_panel(
            tickers,
            start_date=start_date,
            end_date=end_date,
            interval="1d",
            auto_adjust=bool(auto_adjust),
            chunk_size=50,
        )
        return build_return_panel_from_prices(prices, frequency=str(frequency).lower())
    except Exception as exc:
        last_error = exc

    try:
        working_panels = []
        for ticker in tickers:
            try:
                prices = download_yahoo_price_panel(
                    [ticker],
                    start_date=start_date,
                    end_date=end_date,
                    interval="1d",
                    auto_adjust=bool(auto_adjust),
                )
                panel = build_return_panel_from_prices(prices, frequency=str(frequency).lower())
                if isinstance(panel, pd.DataFrame) and not panel.empty:
                    working_panels.append(panel)
            except Exception as exc:
                last_error = exc
                continue
        if working_panels:
            merged = pd.concat(working_panels, ignore_index=True)
            merged = (
                merged.dropna(subset=["date", "asset", "return"])
                .sort_values(["date", "asset"])
                .reset_index(drop=True)
            )
            if not merged.empty:
                return merged
    except Exception as exc:
        last_error = exc

    raise ValueError(f"Yahoo download failed for the selected universe. Last error: {last_error}")


@st.cache_data(show_spinner=False)
def _cached_feature_engineered_asset_panel(
    panel_df: pd.DataFrame,
    feature_cfg_json: str,
    apply_monthly_intramonth: bool,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    timings: Dict[str, float] = {}
    cfg_payload = json.loads(str(feature_cfg_json or "{}"))
    cfg = DailyFeatureConfig(**cfg_payload) if isinstance(cfg_payload, dict) and cfg_payload else DailyFeatureConfig()

    t0 = time.perf_counter()
    tmp = validate_daily_asset_panel(panel_df, cfg)
    timings["validate_seconds"] = float(time.perf_counter() - t0)

    if bool(apply_monthly_intramonth):
        t1 = time.perf_counter()
        tmp = add_daily_intramonth_state_features(tmp, cfg)
        timings["intramonth_features_seconds"] = float(time.perf_counter() - t1)

        t2 = time.perf_counter()
        out = build_intramonth_monthly_features(tmp, cfg)
        timings["monthly_aggregation_seconds"] = float(time.perf_counter() - t2)
    else:
        timings["intramonth_features_seconds"] = 0.0
        timings["monthly_aggregation_seconds"] = 0.0
        out = tmp

    t3 = time.perf_counter()
    out = add_cross_sectional_features(out)
    timings["cross_sectional_seconds"] = float(time.perf_counter() - t3)

    timings["total_feature_pipeline_seconds"] = float(
        timings.get("validate_seconds", 0.0)
        + timings.get("intramonth_features_seconds", 0.0)
        + timings.get("monthly_aggregation_seconds", 0.0)
        + timings.get("cross_sectional_seconds", 0.0)
    )
    return out, timings


def _read_uploaded_asset_panel_bytes(file_bytes: bytes, filename: str) -> pd.DataFrame:
    lower = str(filename or "").lower()
    if lower.endswith(".parquet"):
        df = pd.read_parquet(BytesIO(file_bytes))
    else:
        df = pd.read_csv(BytesIO(file_bytes))
    return df


def validate_asset_panel(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise the common asset-return panel schema."""
    if panel_df is None or not isinstance(panel_df, pd.DataFrame) or panel_df.empty:
        raise ValueError("Asset panel is empty.")
    need = {"date", "asset", "return"}
    missing = sorted(need - set(panel_df.columns))
    if missing:
        raise ValueError(f"Asset panel is missing required columns: {missing}")
    out = panel_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["asset"] = out["asset"].astype(str).str.upper().str.strip()
    out["return"] = pd.to_numeric(out["return"], errors="coerce")
    out = out.dropna(subset=["date", "asset", "return"]).sort_values(["date", "asset"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("Asset panel is empty after cleaning.")
    return out


def build_weekly_return_panel_from_daily_returns(daily_panel_df: pd.DataFrame) -> pd.DataFrame:
    """Compound a raw daily return panel into weekly asset returns.

    The investment engine still receives the prepared monthly panel. Weekly
    returns are stored for diagnostics and reproducible deployment-cache exports.
    """
    daily = validate_asset_panel(daily_panel_df)
    work = daily[["date", "asset", "return"]].copy()
    work["_gross"] = 1.0 + pd.to_numeric(work["return"], errors="coerce")
    work = work.dropna(subset=["date", "asset", "_gross"])
    if work.empty:
        raise ValueError("Daily panel is empty after cleaning for weekly aggregation.")

    weekly = (
        work.groupby(["asset", pd.Grouper(key="date", freq="W-FRI")], dropna=False)["_gross"]
        .prod()
        .reset_index()
    )
    weekly["return"] = weekly["_gross"] - 1.0
    weekly = weekly[["date", "asset", "return"]]
    return validate_asset_panel(weekly)


def _clear_raw_return_panel_state() -> None:
    for key in (STEP4_RAW_DAILY_PANEL_KEY, STEP4_RAW_WEEKLY_PANEL_KEY, STEP4_RAW_PANEL_SOURCE_KEY):
        if key in st.session_state:
            del st.session_state[key]


def _store_raw_return_panels(
    raw_daily_panel: pd.DataFrame,
    *,
    source_label: str = "Yahoo live/local daily returns",
) -> Dict[str, Any]:
    """Store raw daily and weekly panels from the latest live/local or cached run."""
    daily = validate_asset_panel(raw_daily_panel)
    weekly = build_weekly_return_panel_from_daily_returns(daily)
    st.session_state[STEP4_RAW_DAILY_PANEL_KEY] = daily.copy()
    st.session_state[STEP4_RAW_WEEKLY_PANEL_KEY] = weekly.copy()
    st.session_state[STEP4_RAW_PANEL_SOURCE_KEY] = str(source_label or "Yahoo live/local daily returns")
    return {
        "raw_daily_rows": int(len(daily)),
        "raw_daily_assets": int(daily["asset"].nunique()),
        "raw_weekly_rows": int(len(weekly)),
        "raw_weekly_assets": int(weekly["asset"].nunique()),
    }


def _deployment_panel_file() -> Path:
    """Return the configured cached prepared monthly deployment panel path."""
    return Path(DEPLOYMENT_ASSET_PANEL_PATH)


def _deployment_daily_returns_file() -> Path:
    """Return the configured cached raw daily return panel path."""
    return Path(DEPLOYMENT_DAILY_RETURNS_PATH)


def _deployment_weekly_returns_file() -> Path:
    """Return the configured cached raw weekly return panel path."""
    return Path(DEPLOYMENT_WEEKLY_RETURNS_PATH)


def _deployment_raw_file_for_frequency(frequency: str) -> Path | None:
    freq = str(frequency or "").lower()
    if freq == "daily":
        return _deployment_daily_returns_file()
    if freq == "weekly":
        return _deployment_weekly_returns_file()
    return None


@st.cache_data(show_spinner=False)
def _cached_read_deployment_asset_panel(path_str: str, mtime_ns: int, file_size: int) -> pd.DataFrame:
    """Read the cached prepared monthly deployment panel."""
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Cached deployment panel not found: {path}")
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, compression="infer")
    return validate_asset_panel(df)


@st.cache_data(show_spinner=False)
def _cached_read_deployment_raw_return_panel(path_str: str, mtime_ns: int, file_size: int) -> pd.DataFrame:
    """Read a cached raw daily or weekly return panel."""
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Cached raw return panel not found: {path}")
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, compression="infer")
    return validate_asset_panel(df)


def _requested_asset_list(selected_assets: List[str], candidate_assets: List[str]) -> List[str]:
    requested_assets = sorted(
        set(normalize_asset_ticker(x) for x in list(selected_assets or []) + list(candidate_assets or []))
    )
    return [x for x in requested_assets if x]


def _filter_panel_to_request(
    panel_df: pd.DataFrame,
    *,
    selected_assets: List[str],
    candidate_assets: List[str],
    start_date: Any,
    end_date: Any,
) -> tuple[pd.DataFrame, List[str], List[str]]:
    panel_df = validate_asset_panel(panel_df)
    requested_assets = _requested_asset_list(selected_assets, candidate_assets)

    if requested_assets:
        panel_df = panel_df[panel_df["asset"].astype(str).str.upper().isin(set(requested_assets))].copy()

    start_ts = pd.to_datetime(start_date, errors="coerce")
    end_ts = pd.to_datetime(end_date, errors="coerce")
    if pd.notna(start_ts):
        panel_df = panel_df[panel_df["date"] >= start_ts].copy()
    if pd.notna(end_ts):
        panel_df = panel_df[panel_df["date"] <= end_ts].copy()

    panel_df = validate_asset_panel(panel_df)
    loaded_assets = sorted(set(panel_df["asset"].astype(str).str.upper()))
    missing_assets = [ticker for ticker in requested_assets if ticker not in set(loaded_assets)]
    return panel_df, requested_assets, missing_assets


def _cache_caption(
    *,
    label: str,
    path: Path,
    panel_df: pd.DataFrame,
    requested_assets: List[str],
    missing_assets: List[str],
) -> str:
    loaded_assets = sorted(set(panel_df["asset"].astype(str).str.upper()))
    min_date = pd.to_datetime(panel_df["date"], errors="coerce").min()
    max_date = pd.to_datetime(panel_df["date"], errors="coerce").max()
    min_date_str = min_date.strftime("%Y-%m-%d") if pd.notna(min_date) else "unknown"
    max_date_str = max_date.strftime("%Y-%m-%d") if pd.notna(max_date) else "unknown"

    caption = (
        f"{label} · file={path.as_posix()} · "
        f"rows={len(panel_df):,} · assets={len(loaded_assets)} · "
        f"date_range={min_date_str} to {max_date_str}"
    )
    if requested_assets:
        caption += f" · requested_assets={len(requested_assets)}"
    if missing_assets:
        preview_missing = ", ".join(asset_display_label(x) for x in missing_assets[:12])
        if len(missing_assets) > 12:
            preview_missing += f", +{len(missing_assets) - 12} more"
        caption += f" · missing_from_cache={preview_missing}"
    return caption


def _load_deployment_asset_panel(
    *,
    selected_assets: List[str],
    candidate_assets: List[str],
    start_date: Any,
    end_date: Any,
) -> tuple[pd.DataFrame, str, str, Dict[str, Any]]:
    """Load and filter the cached prepared monthly deployment panel."""
    timings: Dict[str, Any] = {}
    path = _deployment_panel_file()
    if not path.exists():
        raise FileNotFoundError(f"Cached deployment panel not found: {path}")

    stat = path.stat()
    t_read = time.perf_counter()
    panel_df = _cached_read_deployment_asset_panel(str(path), int(stat.st_mtime_ns), int(stat.st_size))
    timings["deployment_panel_read_seconds"] = float(time.perf_counter() - t_read)

    panel_df, requested_assets, missing_assets = _filter_panel_to_request(
        panel_df,
        selected_assets=selected_assets,
        candidate_assets=candidate_assets,
        start_date=start_date,
        end_date=end_date,
    )

    loaded_assets = sorted(set(panel_df["asset"].astype(str).str.upper()))
    caption = _cache_caption(
        label="Cached Yahoo prepared monthly deployment panel",
        path=path,
        panel_df=panel_df,
        requested_assets=requested_assets,
        missing_assets=missing_assets,
    )
    timings["deployment_panel_rows"] = int(len(panel_df))
    timings["deployment_panel_assets"] = int(len(loaded_assets))
    timings["deployment_cache_mode"] = "monthly_prepared"

    return panel_df, "Cached Yahoo deployment panel (monthly prepared)", caption, timings


def _load_deployment_raw_return_panel(
    *,
    frequency: str,
    selected_assets: List[str],
    candidate_assets: List[str],
    start_date: Any,
    end_date: Any,
) -> tuple[pd.DataFrame, str, str, Dict[str, Any]]:
    """Load and filter a cached raw daily or weekly return panel."""
    freq = str(frequency or "").lower()
    path = _deployment_raw_file_for_frequency(freq)
    if path is None:
        raise ValueError(f"No raw cached panel is configured for frequency={frequency!r}.")
    if not path.exists():
        raise FileNotFoundError(f"Cached raw {freq} return panel not found: {path}")

    timings: Dict[str, Any] = {}
    stat = path.stat()
    t_read = time.perf_counter()
    panel_df = _cached_read_deployment_raw_return_panel(str(path), int(stat.st_mtime_ns), int(stat.st_size))
    timings[f"deployment_raw_{freq}_read_seconds"] = float(time.perf_counter() - t_read)

    panel_df, requested_assets, missing_assets = _filter_panel_to_request(
        panel_df,
        selected_assets=selected_assets,
        candidate_assets=candidate_assets,
        start_date=start_date,
        end_date=end_date,
    )

    loaded_assets = sorted(set(panel_df["asset"].astype(str).str.upper()))
    caption = _cache_caption(
        label=f"Cached Yahoo raw {freq} return panel",
        path=path,
        panel_df=panel_df,
        requested_assets=requested_assets,
        missing_assets=missing_assets,
    )
    timings[f"deployment_raw_{freq}_rows"] = int(len(panel_df))
    timings[f"deployment_raw_{freq}_assets"] = int(len(loaded_assets))
    timings["deployment_cache_mode"] = f"raw_{freq}"
    return panel_df, f"Cached Yahoo raw {freq} return panel", caption, timings


def _hydrate_raw_return_panel_exports_from_cache(
    *,
    selected_assets: List[str],
    candidate_assets: List[str],
    start_date: Any,
    end_date: Any,
) -> Dict[str, Any]:
    """Populate raw daily/weekly export panels from cached files when available."""
    timings: Dict[str, Any] = {}
    daily_panel: pd.DataFrame | None = None
    weekly_panel: pd.DataFrame | None = None
    source_parts: List[str] = []

    try:
        if _deployment_daily_returns_file().exists():
            daily_panel, _label, _caption, daily_timings = _load_deployment_raw_return_panel(
                frequency="daily",
                selected_assets=selected_assets,
                candidate_assets=candidate_assets,
                start_date=start_date,
                end_date=end_date,
            )
            timings.update(dict(daily_timings or {}))
            source_parts.append("cached daily")
    except Exception as exc:
        timings["deployment_raw_daily_error"] = str(exc)

    try:
        if _deployment_weekly_returns_file().exists():
            weekly_panel, _label, _caption, weekly_timings = _load_deployment_raw_return_panel(
                frequency="weekly",
                selected_assets=selected_assets,
                candidate_assets=candidate_assets,
                start_date=start_date,
                end_date=end_date,
            )
            timings.update(dict(weekly_timings or {}))
            source_parts.append("cached weekly")
    except Exception as exc:
        timings["deployment_raw_weekly_error"] = str(exc)

    if weekly_panel is None and isinstance(daily_panel, pd.DataFrame) and not daily_panel.empty:
        try:
            weekly_panel = build_weekly_return_panel_from_daily_returns(daily_panel)
            timings["deployment_raw_weekly_rows"] = int(len(weekly_panel))
            timings["deployment_raw_weekly_assets"] = int(weekly_panel["asset"].nunique())
            source_parts.append("weekly derived from cached daily")
        except Exception as exc:
            timings["deployment_raw_weekly_derived_error"] = str(exc)

    if isinstance(daily_panel, pd.DataFrame) and not daily_panel.empty:
        st.session_state[STEP4_RAW_DAILY_PANEL_KEY] = daily_panel.copy()
    elif STEP4_RAW_DAILY_PANEL_KEY in st.session_state:
        del st.session_state[STEP4_RAW_DAILY_PANEL_KEY]

    if isinstance(weekly_panel, pd.DataFrame) and not weekly_panel.empty:
        st.session_state[STEP4_RAW_WEEKLY_PANEL_KEY] = weekly_panel.copy()
    elif STEP4_RAW_WEEKLY_PANEL_KEY in st.session_state:
        del st.session_state[STEP4_RAW_WEEKLY_PANEL_KEY]

    if source_parts:
        st.session_state[STEP4_RAW_PANEL_SOURCE_KEY] = "Cached Yahoo deployment return panels (" + ", ".join(source_parts) + ")"
    elif STEP4_RAW_PANEL_SOURCE_KEY in st.session_state:
        del st.session_state[STEP4_RAW_PANEL_SOURCE_KEY]

    return timings


def _load_deployment_panel_for_frequency(
    *,
    frequency: str,
    selected_assets: List[str],
    candidate_assets: List[str],
    start_date: Any,
    end_date: Any,
) -> tuple[pd.DataFrame, str, str, Dict[str, Any]]:
    """Resolve the best cached deployment path for the selected frequency.

    - monthly: use the prepared monthly panel directly.
    - daily: rebuild the prepared monthly panel from cached raw daily returns.
    - weekly: keep Step 5 stable by using the prepared monthly panel, while
      loading the cached weekly panel into diagnostics/export state.
    """
    freq = str(frequency or "monthly").lower()
    timings: Dict[str, Any] = {}
    t_cache = time.perf_counter()

    if freq == "daily" and _deployment_daily_returns_file().exists():
        raw_daily, _raw_label, raw_caption, raw_timings = _load_deployment_raw_return_panel(
            frequency="daily",
            selected_assets=selected_assets,
            candidate_assets=candidate_assets,
            start_date=start_date,
            end_date=end_date,
        )
        timings.update(dict(raw_timings or {}))
        timings.update(
            _store_raw_return_panels(
                raw_daily,
                source_label="Cached Yahoo deployment daily returns · auto-adjust already resolved at export time",
            )
        )

        cfg = DailyFeatureConfig()
        feature_cfg_json = _serialize_feature_cfg(cfg)
        t_feat = time.perf_counter()
        panel_df, feature_timings = _cached_feature_engineered_asset_panel(
            raw_daily,
            feature_cfg_json,
            True,
        )
        timings.update(dict(feature_timings or {}))
        timings["feature_pipeline_resolve_seconds"] = float(time.perf_counter() - t_feat)
        timings["feature_pipeline_cache_reused"] = bool(
            timings["feature_pipeline_resolve_seconds"] + 1e-6 < timings.get("total_feature_pipeline_seconds", 0.0)
        )
        timings["deployment_panel_total_seconds"] = float(time.perf_counter() - t_cache)
        timings["deployment_cache_mode"] = "daily_raw_to_monthly_prepared"
        caption = f"{raw_caption} · rebuilt monthly prepared engine panel from cached daily returns"
        return panel_df, "Cached Yahoo daily returns → monthly prepared panel", caption, timings

    panel_df, source_label, caption, panel_timings = _load_deployment_asset_panel(
        selected_assets=selected_assets,
        candidate_assets=candidate_assets,
        start_date=start_date,
        end_date=end_date,
    )
    timings.update(dict(panel_timings or {}))
    timings.update(
        _hydrate_raw_return_panel_exports_from_cache(
            selected_assets=selected_assets,
            candidate_assets=candidate_assets,
            start_date=start_date,
            end_date=end_date,
        )
    )
    timings["deployment_panel_total_seconds"] = float(time.perf_counter() - t_cache)
    if freq == "weekly" and _deployment_weekly_returns_file().exists():
        source_label = "Cached Yahoo weekly diagnostics + monthly prepared engine panel"
        caption = (
            f"{caption} · cached weekly raw returns loaded for diagnostics/export; "
            "Step 5 still uses the prepared monthly engine panel"
        )
        timings["deployment_cache_mode"] = "weekly_raw_plus_monthly_prepared"
    return panel_df, source_label, caption, timings


def _should_use_deployment_panel_first(frequency: str) -> bool:
    if not USE_DEPLOYMENT_PANEL_FIRST:
        return False
    freq = str(frequency or "monthly").lower()
    if freq == "daily":
        return _deployment_daily_returns_file().exists() or _deployment_panel_file().exists()
    if freq == "weekly":
        return _deployment_weekly_returns_file().exists() or _deployment_panel_file().exists()
    return _deployment_panel_file().exists()


def resolve_step4_asset_panel(
    *,
    selected_assets: List[str],
    candidate_assets: List[str],
    source_mode: str,
    start_date: Any,
    end_date: Any,
    frequency: str,
    auto_adjust: bool,
    uploaded_file: Any,
) -> Tuple[pd.DataFrame | None, str, str]:
    """Resolve the market-data panel used by the Step 5 Strategy Engine.

    Resolution order:
    1. Uploaded panel when source_mode is upload.
    2. Cached deployment panel first when configured/available.
    3. Live/local Yahoo download with cached deployment fallback.

    The returned dataframe is the prepared engine panel. Raw daily/weekly panels
    are stored separately for diagnostics and downloads.
    """
    source_mode = str(source_mode or "yahoo").lower()
    frequency = str(frequency or "monthly").lower()
    overall_t0 = time.perf_counter()
    timings: Dict[str, Any] = {}

    if source_mode == "upload":
        _clear_raw_return_panel_state()
        if uploaded_file is None:
            return None, "", ""
        raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else None
        if raw is None:
            raise ValueError("Uploaded file could not be read.")

        t_read = time.perf_counter()
        panel_df = _read_uploaded_asset_panel_bytes(raw, getattr(uploaded_file, "name", "uploaded_panel.csv"))
        timings["uploaded_read_seconds"] = float(time.perf_counter() - t_read)

        t_basic = time.perf_counter()
        panel_df = validate_asset_panel(panel_df)
        timings["validate_asset_panel_seconds"] = float(time.perf_counter() - t_basic)

        try:
            cfg = DailyFeatureConfig()
            feature_cfg_json = _serialize_feature_cfg(cfg)
            tmp = validate_daily_asset_panel(panel_df, cfg)
            obs_per_month = tmp.assign(_month=tmp["date"].dt.to_period("M")).groupby(["asset", "_month"]).size()
            median_obs_per_asset_month = float(obs_per_month.median()) if not obs_per_month.empty else 0.0
            apply_monthly_intramonth = bool(median_obs_per_asset_month >= 5.0)

            t_feat = time.perf_counter()
            panel_df, feature_timings = _cached_feature_engineered_asset_panel(
                panel_df,
                feature_cfg_json,
                apply_monthly_intramonth,
            )
            timings.update(dict(feature_timings or {}))
            timings["feature_pipeline_resolve_seconds"] = float(time.perf_counter() - t_feat)
            timings["feature_pipeline_cache_reused"] = bool(
                timings["feature_pipeline_resolve_seconds"] + 1e-6 < timings.get("total_feature_pipeline_seconds", 0.0)
            )
        except Exception as e:
            raise ValueError(f"Feature engineering failed: {e}")

        timings["resolve_total_seconds"] = float(time.perf_counter() - overall_t0)
        st.session_state[STEP4_PANEL_TIMINGS_KEY] = dict(timings)
        return panel_df, f"Uploaded file ({getattr(uploaded_file, 'name', 'uploaded panel')})", format_step4_timing_summary(timings)

    universe_union = sorted(set(selected_assets) | set(candidate_assets))
    if not universe_union:
        raise ValueError("No valid tickers have been selected yet for Yahoo download.")

    deployment_error: Exception | None = None

    if _should_use_deployment_panel_first(frequency):
        try:
            panel_df, source_label, union_caption, cache_timings = _load_deployment_panel_for_frequency(
                frequency=frequency,
                selected_assets=list(selected_assets or []),
                candidate_assets=list(candidate_assets or []),
                start_date=start_date,
                end_date=end_date,
            )
            timings.update(dict(cache_timings or {}))
            timings["download_yahoo_seconds"] = 0.0
            timings["return_panel_seconds"] = 0.0
            timings["resolve_total_seconds"] = float(time.perf_counter() - overall_t0)
            st.session_state[STEP4_PANEL_TIMINGS_KEY] = dict(timings)
            timing_caption = format_step4_timing_summary(timings)
            if timing_caption:
                union_caption = f"{union_caption} · {timing_caption}"
            return panel_df, source_label, union_caption
        except Exception as exc:
            deployment_error = exc
            timings["deployment_panel_error"] = str(exc)

    t_dl = time.perf_counter()
    try:
        raw_daily_panel = _cached_download_yahoo_asset_panel(
            tuple(universe_union),
            str(start_date),
            str(end_date),
            "daily",
            bool(auto_adjust),
        )
    except Exception as yahoo_exc:
        if _deployment_panel_file().exists():
            try:
                panel_df, source_label, union_caption, cache_timings = _load_deployment_panel_for_frequency(
                    frequency=frequency,
                    selected_assets=list(selected_assets or []),
                    candidate_assets=list(candidate_assets or []),
                    start_date=start_date,
                    end_date=end_date,
                )
                timings.update(dict(cache_timings or {}))
                timings["download_yahoo_seconds"] = float(time.perf_counter() - t_dl)
                timings["return_panel_seconds"] = 0.0
                timings["yahoo_live_error"] = str(yahoo_exc)
                timings["resolve_total_seconds"] = float(time.perf_counter() - overall_t0)
                st.session_state[STEP4_PANEL_TIMINGS_KEY] = dict(timings)
                timing_caption = format_step4_timing_summary(timings)
                if timing_caption:
                    union_caption = f"{union_caption} · live_yahoo_failed=True · {timing_caption}"
                return panel_df, source_label, union_caption
            except Exception as cache_exc:
                deployment_error = cache_exc

        if deployment_error is not None:
            raise ValueError(
                f"Yahoo download failed and cached deployment panel could not be used. "
                f"Yahoo error: {yahoo_exc}. Cached panel error: {deployment_error}"
            )
        raise

    timings["download_yahoo_seconds"] = float(time.perf_counter() - t_dl)
    timings["return_panel_seconds"] = 0.0

    t_basic = time.perf_counter()
    raw_daily_panel = validate_asset_panel(raw_daily_panel)
    timings["validate_asset_panel_seconds"] = float(time.perf_counter() - t_basic)

    try:
        timings.update(
            _store_raw_return_panels(
                raw_daily_panel,
                source_label=f"Yahoo live/local daily returns · auto_adjust={bool(auto_adjust)}",
            )
        )
    except Exception as raw_exc:
        timings["raw_panel_export_error"] = str(raw_exc)

    try:
        cfg = DailyFeatureConfig()
        feature_cfg_json = _serialize_feature_cfg(cfg)

        t_feat = time.perf_counter()
        panel_df, feature_timings = _cached_feature_engineered_asset_panel(
            raw_daily_panel,
            feature_cfg_json,
            True,
        )
        timings.update(dict(feature_timings or {}))
        timings["feature_pipeline_resolve_seconds"] = float(time.perf_counter() - t_feat)
        timings["feature_pipeline_cache_reused"] = bool(
            timings["feature_pipeline_resolve_seconds"] + 1e-6 < timings.get("total_feature_pipeline_seconds", 0.0)
        )
    except Exception as e:
        raise ValueError(f"Feature engineering failed: {e}")

    loaded_assets = (
        sorted(set(panel_df["asset"].astype(str).str.upper()))
        if isinstance(panel_df, pd.DataFrame) and not panel_df.empty
        else []
    )
    missing_assets = [ticker for ticker in universe_union if ticker not in set(loaded_assets)]
    union_caption = f"Tickers requested ({len(universe_union)}): {', '.join(asset_display_label(x) for x in universe_union)}"
    if loaded_assets and len(loaded_assets) != len(universe_union):
        union_caption += f" · Loaded subset ({len(loaded_assets)}): {', '.join(asset_display_label(x) for x in loaded_assets)}"
    if missing_assets:
        union_caption += f" · Missing from Yahoo: {', '.join(asset_display_label(x) for x in missing_assets)}"

    timings["resolve_total_seconds"] = float(time.perf_counter() - overall_t0)
    st.session_state[STEP4_PANEL_TIMINGS_KEY] = dict(timings)
    timing_caption = format_step4_timing_summary(timings)
    if timing_caption:
        union_caption = f"{union_caption} · {timing_caption}"
    return panel_df, "Yahoo Finance live daily → monthly prepared panel", union_caption


def update_asset_panel_state(
    panel_df: pd.DataFrame | None,
    source_label: str,
    error_message: str = "",
    *,
    panel_signature: str = "",
    timings: Dict[str, Any] | None = None,
    build_trigger: str = "",
) -> None:
    """Persist the resolved market-data panel and readiness metadata."""
    st.session_state[ASSET_PANEL_DF] = panel_df if isinstance(panel_df, pd.DataFrame) and not panel_df.empty else None
    st.session_state[ASSET_PANEL_READY] = bool(isinstance(panel_df, pd.DataFrame) and not panel_df.empty)
    st.session_state[ASSET_PANEL_SOURCE_LABEL] = str(source_label or "")
    st.session_state[ASSET_PANEL_ERROR] = str(error_message or "")
    if panel_signature:
        st.session_state[STEP4_PANEL_SIGNATURE_KEY] = str(panel_signature)
    if timings is not None:
        st.session_state[STEP4_PANEL_TIMINGS_KEY] = dict(timings or {})
    if build_trigger:
        st.session_state[STEP4_PANEL_LAST_BUILD_TRIGGER_KEY] = str(build_trigger)


def build_step4_universe_payload_from_state() -> Dict[str, Any]:
    """Build the Step 4 payload consumed by the Strategy Engine screen."""
    size_value, strategy_value = validate_universe_inputs(
        st.session_state.get(UNIVERSE_SIZE, 25),
        st.session_state.get(UNIVERSE_STRATEGY, UNIVERSE_STRATEGY_CORE),
    )
    selected_assets, selection_source = resolve_universe_selection(
        size_value,
        strategy_value,
        bool(st.session_state.get(UNIVERSE_CUSTOM_ENABLED, False)),
        st.session_state.get(CUSTOM_UNIVERSE_TEXT, ""),
    )
    payload = {
        "size": int(size_value),
        "strategy": str(strategy_value),
        "custom_enabled": bool(st.session_state.get(UNIVERSE_CUSTOM_ENABLED, False)),
        "custom_text": str(st.session_state.get(CUSTOM_UNIVERSE_TEXT, "") or ""),
        "selection_source": str(selection_source),
        "selected_assets": list(selected_assets),
        "candidate_assets": build_strategy_candidate_pool(size_value, strategy_value),
        "philosophy": get_canonical_investment_philosophy(),
        "style_preset": str(st.session_state.get(UNIVERSE_SIMPLE_STYLE_PRESET, "Balanced") or "Balanced"),
        "strategy_template": str(
            st.session_state.get(UNIVERSE_SIMPLE_STRATEGY_TEMPLATE, "Balanced Risk-Controlled")
            or "Balanced Risk-Controlled"
        ),
        "projection_profile": str(st.session_state.get(INVESTMENT_PROJECTION_PROFILE_HINT, "Balanced") or "Balanced"),
        "asset_source_mode": str(st.session_state.get(ASSET_SOURCE_MODE, "yahoo") or "yahoo"),
        "asset_start_date": st.session_state.get(ASSET_START_DATE),
        "asset_end_date": st.session_state.get(ASSET_END_DATE),
        "asset_return_frequency": str(st.session_state.get(ASSET_RETURN_FREQUENCY, "monthly") or "monthly"),
        "auto_adjust_asset_prices": bool(st.session_state.get(ASSET_AUTO_ADJUST, True)),
        "asset_uploaded_file": st.session_state.get(ASSET_UPLOADED_FILE),
        "asset_panel_df": st.session_state.get(ASSET_PANEL_DF),
        "asset_panel_ready": bool(st.session_state.get(ASSET_PANEL_READY, False)),
        "asset_panel_source_label": str(st.session_state.get(ASSET_PANEL_SOURCE_LABEL, "") or ""),
        "asset_panel_error": str(st.session_state.get(ASSET_PANEL_ERROR, "") or ""),
    }
    return payload


def panel_df_to_csv_bytes(panel_df: pd.DataFrame) -> bytes:
    """Serialize a panel dataframe into CSV bytes for download buttons."""
    if panel_df is None or panel_df.empty:
        return b""
    df = panel_df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df.to_csv(index=False).encode("utf-8")


def panel_download_filename(source_label: str, strategy: str, frequency: str) -> str:
    """Build a stable, filesystem-safe filename for a panel download."""
    return f"{source_label}_{strategy}_{frequency}_panel.csv".replace(" ", "_").replace("/", "_").lower()