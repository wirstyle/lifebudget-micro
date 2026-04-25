from __future__ import annotations

import json
from typing import Any

import streamlit as st
import pandas as pd

from ui.step5.run_panel import clear_retired_step5_state



def _coerce_mapping(value: Any) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        return dict(value)
    except Exception:
        return {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _pct(v: float) -> str:
    return f"{100.0 * float(v):.2f}%"


def _parse_feature_family_counts(raw: Any) -> dict[str, int]:
    if isinstance(raw, dict):
        source = raw
    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            source = parsed if isinstance(parsed, dict) else {}
        except Exception:
            source = {}
    else:
        source = {}

    out: dict[str, int] = {}
    for key, value in source.items():
        name = str(key).strip()
        if not name:
            continue
        out[name] = _safe_int(value, 0)
    return out


def _parse_feature_cols(raw: Any) -> list[str]:
    if isinstance(raw, list):
        items = raw
    else:
        items = str(raw or "").split("|")
    return [str(x).strip() for x in items if str(x).strip()]


def _render_feature_mu_block(run_map: dict) -> None:
    selected_cols_n = _safe_int(run_map.get("feature_mu_selected_cols_n", 0), 0)
    selected_cols = _parse_feature_cols(run_map.get("feature_mu_selected_cols", ""))
    family_counts = _parse_feature_family_counts(run_map.get("feature_mu_selected_family_counts", "{}"))

    # If the engine did not return feature_mu metadata, do not render an empty-looking section.
    has_metadata = bool(selected_cols_n > 0 or len(selected_cols) > 0 or len(family_counts) > 0)
    if not has_metadata:
        return

    st.markdown("### Feature_mu selection")
    m1, m2 = st.columns([1.0, 2.0])
    with m1:
        st.metric("Feature_mu cols used", int(selected_cols_n))
    with m2:
        if family_counts:
            family_text = " · ".join(f"{name}: {count}" for name, count in family_counts.items())
            st.caption("Families")
            st.write(family_text)
        else:
            st.caption("Families")
            st.write("No family summary available.")

    if selected_cols:
        with st.expander("Selected feature_mu columns", expanded=False):
            st.write(selected_cols)



def _extract_oos_returns(run_map: dict) -> list[float]:
    candidates = [
        run_map.get("oos_returns_monthly"),
        run_map.get("oos_returns_simple"),
        run_map.get("portfolio_returns"),
        run_map.get("oos_returns"),
        run_map.get("returns"),
    ]

    for raw in candidates:
        if raw is None:
            continue
        try:
            if hasattr(raw, "tolist"):
                raw = raw.tolist()
        except Exception:
            raw = []

        if not isinstance(raw, list):
            continue

        out: list[float] = []
        for item in raw:
            try:
                out.append(float(item))
            except Exception:
                continue
        if out:
            return out

    return []


def _store_projection_bridge_context(run_map: dict) -> int:
    ctx = st.session_state.get("investment_context", {})
    if not isinstance(ctx, dict):
        ctx = {}

    oos_returns = _extract_oos_returns(run_map)
    ctx["oos_returns_monthly"] = list(oos_returns)
    ctx.setdefault("run_signature", str(run_map.get("run_signature", "") or ""))
    st.session_state["investment_context"] = ctx
    return int(len(oos_returns))




def _render_engine_timing_block(run_map: dict) -> None:
    engine_timing = _coerce_mapping(run_map.get("engine_timing", {}))
    if not engine_timing:
        return

    st.markdown("### Engine timing diagnostics")

    total_engine = _safe_float(engine_timing.get("total_engine", 0.0), 0.0)
    walk_forward = _safe_float(engine_timing.get("walk_forward_loop", 0.0), 0.0)
    cov_sigma = _safe_float(engine_timing.get("covariance_sigma_total", 0.0), 0.0)
    mu_sigma = _safe_float(engine_timing.get("mu_sigma_total", 0.0), 0.0)
    weight_build = _safe_float(engine_timing.get("weight_build_total", 0.0), 0.0)
    signal_model = _safe_float(engine_timing.get("signal_model_total", 0.0), 0.0)
    feature_mu = _safe_float(engine_timing.get("feature_mu_total", 0.0), 0.0)
    probabilistic = _safe_float(engine_timing.get("probabilistic_total", 0.0), 0.0)
    n_oos_dates = _safe_int(engine_timing.get("n_oos_dates", 0), 0)
    n_loop_iterations = _safe_int(engine_timing.get("n_loop_iterations", 0), 0)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total engine", f"{total_engine:.2f}s")
    with c2:
        st.metric("Walk-forward", f"{walk_forward:.2f}s")
    with c3:
        st.metric("Cov/Sigma", f"{cov_sigma:.2f}s")
    with c4:
        st.metric("Mu/Sigma", f"{mu_sigma:.2f}s")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric("Weight build", f"{weight_build:.2f}s")
    with c6:
        st.metric("Signal model", f"{signal_model:.2f}s")
    with c7:
        st.metric("Feature_mu", f"{feature_mu:.2f}s")
    with c8:
        st.metric("Probabilistic", f"{probabilistic:.2f}s")

    st.caption(
        f"n_oos_dates={n_oos_dates} · "
        f"n_loop_iterations={n_loop_iterations}"
    )

    detail_rows = []
    preferred_order = [
        "prepare_panel",
        "feature_pair_prep",
        "walk_forward_loop",
        "mu_sigma_total",
        "probabilistic_total",
        "feature_mu_total",
        "signal_model_total",
        "regime_filter_total",
        "covariance_sigma_total",
        "weight_build_total",
        "post_weights_total",
        "diagnostics_total",
        "finalize_total",
        "total_engine",
    ]

    for key in preferred_order:
        if key in engine_timing:
            value = engine_timing.get(key)
            if isinstance(value, (int, float)):
                detail_rows.append({"component": key, "seconds": float(value)})

    if detail_rows:
        with st.expander("Engine timing breakdown", expanded=False):
            st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)



def _result_interpretation(perf: dict) -> tuple[str, str]:
    cagr = _safe_float(perf.get("cagr", 0.0), 0.0)
    sharpe = _safe_float(perf.get("sharpe", 0.0), 0.0)
    vol = _safe_float(perf.get("annual_volatility", perf.get("volatility", 0.0)), 0.0)
    maxdd = abs(_safe_float(perf.get("max_drawdown", 0.0), 0.0))

    if sharpe >= 1.0:
        sharpe_note = "strong risk-adjusted performance"
    elif sharpe >= 0.60:
        sharpe_note = "reasonable risk-adjusted performance"
    elif sharpe > 0:
        sharpe_note = "positive but modest risk-adjusted performance"
    else:
        sharpe_note = "weak risk-adjusted performance"

    if maxdd >= 0.30:
        risk_note = "a large historical drawdown, so the strategy would require high tolerance for temporary losses"
    elif maxdd >= 0.15:
        risk_note = "a meaningful historical drawdown, so the user would need to tolerate uncomfortable periods"
    elif maxdd > 0:
        risk_note = "a contained historical drawdown relative to more aggressive portfolios"
    else:
        risk_note = "no meaningful drawdown recorded in the available summary"

    headline = "Result interpretation"
    body = (
        f"This run shows {sharpe_note}: CAGR was {_pct(cagr)}, volatility was {_pct(vol)}, "
        f"and maximum drawdown was -{100.0 * maxdd:.2f}%. The main trade-off is {risk_note}."
    )
    return headline, body


def _philosophy_fit_message(perf: dict, philosophy: Any) -> tuple[str, str]:
    profile = str(philosophy or "Balanced").strip() or "Balanced"
    profile_key = profile.lower()
    cagr = _safe_float(perf.get("cagr", 0.0), 0.0)
    sharpe = _safe_float(perf.get("sharpe", 0.0), 0.0)
    vol = _safe_float(perf.get("annual_volatility", perf.get("volatility", 0.0)), 0.0)
    maxdd = abs(_safe_float(perf.get("max_drawdown", 0.0), 0.0))

    if profile_key == "defensive":
        if maxdd >= 0.18 or vol >= 0.13:
            return (
                "warning",
                f"For a Defensive profile, this result may be too uncomfortable: the historical drawdown is around -{100.0 * maxdd:.2f}% and volatility is {_pct(vol)}. The return may be useful, but the risk profile deserves caution.",
            )
        return (
            "success",
            f"For a Defensive profile, this looks relatively aligned: the drawdown is around -{100.0 * maxdd:.2f}% and volatility is {_pct(vol)}, so the setup appears more controlled than aggressive.",
        )

    if profile_key == "growth":
        if cagr >= 0.08 and sharpe >= 0.50:
            return (
                "success",
                f"For a Growth profile, this looks directionally aligned: CAGR is {_pct(cagr)} and the strategy accepts some volatility in exchange for higher upside potential.",
            )
        return (
            "info",
            f"For a Growth profile, the result is usable but not clearly aggressive: CAGR is {_pct(cagr)} and Sharpe is {sharpe:.2f}. You may want to check whether the setup is too defensive or too diversified.",
        )

    # Balanced / default
    if maxdd >= 0.25:
        return (
            "warning",
            f"For a Balanced profile, the return profile may be attractive, but a drawdown around -{100.0 * maxdd:.2f}% is high enough to question whether the setup still feels balanced.",
        )
    if sharpe >= 0.60 and cagr > 0:
        return (
            "success",
            f"For a Balanced profile, this is broadly aligned if the user can tolerate a drawdown near -{100.0 * maxdd:.2f}%. It is growth-positive, but not defensive.",
        )
    return (
        "info",
        f"For a Balanced profile, this result needs review: CAGR is {_pct(cagr)}, Sharpe is {sharpe:.2f}, and drawdown is around -{100.0 * maxdd:.2f}%.",
    )


def _render_metric_explainer(perf: dict) -> None:
    cagr = _safe_float(perf.get("cagr", 0.0), 0.0)
    sharpe = _safe_float(perf.get("sharpe", 0.0), 0.0)
    vol = _safe_float(perf.get("annual_volatility", perf.get("volatility", 0.0)), 0.0)
    maxdd = abs(_safe_float(perf.get("max_drawdown", 0.0), 0.0))

    with st.expander("How to read these metrics", expanded=False):
        st.markdown(
            f"- **CAGR ({_pct(cagr)})** — historical average annual growth in this backtest. Higher is better, but it is not guaranteed.\n"
            f"- **Sharpe ({sharpe:.2f})** — return per unit of risk. Higher usually means the return compensated better for volatility.\n"
            f"- **Volatility ({_pct(vol)})** — how bumpy the portfolio was historically. Lower usually feels more stable.\n"
            f"- **MaxDD (-{100.0 * maxdd:.2f}%)** — worst historical peak-to-trough fall. This is the main pain-test metric."
        )


def _render_what_to_watch(perf: dict, philosophy: Any) -> None:
    profile = str(philosophy or "Balanced").strip() or "Balanced"
    cagr = _safe_float(perf.get("cagr", 0.0), 0.0)
    sharpe = _safe_float(perf.get("sharpe", 0.0), 0.0)
    vol = _safe_float(perf.get("annual_volatility", perf.get("volatility", 0.0)), 0.0)
    maxdd = abs(_safe_float(perf.get("max_drawdown", 0.0), 0.0))

    watch_items = []
    if cagr < 0.05:
        watch_items.append("If CAGR feels too low, the setup may be too defensive, too diversified, or using weak signals for this universe.")
    else:
        watch_items.append("CAGR is positive; the next question is whether the drawdown and volatility are acceptable for the selected philosophy.")
    if sharpe < 0.50:
        watch_items.append("Sharpe is modest; the portfolio may not be earning enough return for the risk it takes.")
    else:
        watch_items.append("Sharpe is usable; compare it against alternative internal configurations if you later add a benchmark module.")
    if maxdd >= 0.18:
        watch_items.append("MaxDD is the key risk flag: a drawdown near 20% can be psychologically hard even if the long-run return is positive.")
    if vol >= 0.15:
        watch_items.append("Volatility is elevated; reduce risk appetite or increase drawdown protection if the strategy should feel smoother.")
    if profile.lower() == "defensive" and (maxdd >= 0.15 or vol >= 0.12):
        watch_items.append("Because the selected philosophy is Defensive, prioritise drawdown and volatility over chasing higher CAGR.")
    elif profile.lower() == "growth":
        watch_items.append("Because the selected philosophy is Growth, some volatility is acceptable, but Sharpe should still justify the risk.")

    with st.expander("What to watch before changing the strategy", expanded=False):
        st.markdown("\n".join(f"- {item}" for item in watch_items))
        st.caption(
            "Benchmark comparisons against SPY, QQQ, bonds, commodities or crypto should only be added if the app calculates them over the same period and methodology."
        )



def _render_reliability_note() -> None:
    st.info(
        "**Reliability note:** these figures come from a historical walk-forward backtest using the selected Step 4 asset panel. "
        "They are useful for comparing configurations inside the app, but they are not forecasts or guarantees. "
        "Results depend on the date range, asset universe, data quality, and engine assumptions. Step 6 should be used to explore future uncertainty rather than treating this run as a prediction."
    )


BENCHMARK_CONTEXT_ASSETS = [
    {"ticker": "SPY", "reference": "SPY — S&P 500", "type": "US large-cap equities", "reading": "Equity-growth reference"},
    {"ticker": "QQQ", "reference": "QQQ — Nasdaq-100", "type": "Growth / technology-heavy equities", "reading": "Higher-growth / higher-volatility reference"},
    {"ticker": "GLD", "reference": "GLD — Gold", "type": "Gold / alternative diversifier", "reading": "Defensive diversifier / inflation hedge reference"},
    {"ticker": "AGG", "reference": "AGG — US aggregate bonds", "type": "Diversified bonds", "reading": "Defensive bond reference"},
    {"ticker": "TLT", "reference": "TLT — Long-term US Treasuries", "type": "Long-duration Treasury bonds", "reading": "Rate-sensitive defensive reference"},
    {"ticker": "IEF", "reference": "IEF — 7–10 year US Treasuries", "type": "Intermediate Treasury bonds", "reading": "Intermediate defensive bond reference"},
    {"ticker": "VNQ", "reference": "VNQ — US real estate", "type": "US REITs / real estate", "reading": "Real-assets / income-oriented reference"},
]

SHORTER_HISTORY_CONTEXT = [
    "DBC / USO — commodities and oil ETFs usually start after 2005 in Yahoo ETF history",
    "BND — total bond market ETF starts after the 2005 window used here",
    "BTC-USD — crypto history is much shorter and would require a separate shorter-window comparison",
]


def _format_pct_signed(value: Any, decimals: int = 2) -> str:
    try:
        return f"{100.0 * float(value):.{decimals}f}%"
    except Exception:
        return "—"


def _safe_return_series(raw: pd.Series) -> pd.Series:
    series = pd.to_numeric(raw, errors="coerce").dropna().astype(float)
    if series.empty:
        return series
    try:
        if float(series.abs().median()) > 1.0:
            series = series / 100.0
    except Exception:
        pass
    return series


def _compute_return_metrics(returns: pd.Series, *, periods_per_year: int = 12) -> dict:
    r = _safe_return_series(returns)
    n = int(len(r))
    if n <= 1:
        return {}

    wealth = (1.0 + r).cumprod()
    final_wealth = float(wealth.iloc[-1]) if len(wealth) else 0.0
    years = float(n) / float(periods_per_year)
    cagr = float(final_wealth ** (1.0 / years) - 1.0) if final_wealth > 0.0 and years > 0.0 else 0.0
    vol = float(r.std(ddof=1) * (periods_per_year ** 0.5)) if n > 1 else 0.0
    ann_mean = float(r.mean() * periods_per_year) if n > 0 else 0.0
    sharpe = float(ann_mean / vol) if vol > 1e-12 else 0.0

    peak = wealth.cummax()
    drawdown = (wealth / peak) - 1.0
    maxdd = float(drawdown.min()) if len(drawdown) else 0.0

    return {"cagr": cagr, "annual_volatility": vol, "sharpe": sharpe, "max_drawdown": maxdd, "periods": n}


def _strategy_context_zone(perf: dict, philosophy: Any) -> str:
    profile = str(philosophy or "Balanced").strip() or "Balanced"
    cagr = _safe_float(perf.get("cagr", 0.0), 0.0)
    vol = _safe_float(perf.get("annual_volatility", perf.get("volatility", 0.0)), 0.0)
    maxdd = abs(_safe_float(perf.get("max_drawdown", 0.0), 0.0))

    if vol >= 0.22 or maxdd >= 0.35:
        return "High-risk / speculative-like zone"
    if cagr >= 0.08 and vol >= 0.13:
        return "Growth-leaning zone"
    if cagr >= 0.05 and vol <= 0.16 and maxdd <= 0.25:
        return f"{profile} / multi-asset zone"
    if vol <= 0.09 and maxdd <= 0.15:
        return "Defensive / smoother-return zone"
    return f"{profile} context zone"


def _benchmark_context_rows(perf: dict, run_map: dict) -> tuple[pd.DataFrame, list[str]]:
    panel = st.session_state.get("asset_panel_df")
    if not isinstance(panel, pd.DataFrame) or panel.empty:
        return pd.DataFrame(), ["Step 4 asset panel is not available in session_state."]

    required = {"date", "asset", "return"}
    if not required.issubset(set(map(str, panel.columns))):
        return pd.DataFrame(), ["Step 4 asset panel does not contain date, asset and return columns."]

    work = panel.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["asset"] = work["asset"].astype(str).str.upper().str.strip()
    work["return"] = pd.to_numeric(work["return"], errors="coerce")
    work = work.dropna(subset=["date", "asset", "return"]).sort_values(["asset", "date"])

    oos_returns = _extract_oos_returns(run_map)
    target_periods = int(len(oos_returns)) if oos_returns else 0
    philosophy = str(st.session_state.get("investment_philosophy", "Balanced") or "Balanced")

    rows = [{
        "Reference": "Your strategy",
        "Type": "Engine portfolio",
        "Window": f"Engine OOS ({target_periods} months)" if target_periods else "Engine backtest",
        "CAGR": _format_pct_signed(_safe_float(perf.get("cagr", 0.0))),
        "Sharpe": f"{_safe_float(perf.get('sharpe', 0.0)):.2f}",
        "Vol": _format_pct_signed(_safe_float(perf.get("annual_volatility", perf.get("volatility", 0.0)))),
        "MaxDD": f"-{100.0 * abs(_safe_float(perf.get('max_drawdown', 0.0))):.2f}%",
        "Reading": _strategy_context_zone(perf, philosophy),
        "Comparability": "Primary result",
    }]

    notes: list[str] = []

    for item in BENCHMARK_CONTEXT_ASSETS:
        ticker = str(item["ticker"]).upper()
        subset = work.loc[work["asset"] == ticker, ["date", "return"]].dropna().sort_values("date")
        if subset.empty:
            notes.append(f"{ticker} was not available in the current Step 4 asset panel, so it was skipped.")
            continue

        full_len = int(len(subset))
        comparable = "Same trailing OOS length"
        if target_periods > 0 and full_len >= target_periods:
            subset = subset.tail(target_periods).copy()
        elif target_periods > 0 and full_len < target_periods:
            comparable = f"Shorter available history ({full_len}/{target_periods} months)"
        else:
            comparable = "Available Step 4 history"

        metrics = _compute_return_metrics(subset["return"], periods_per_year=12)
        if not metrics:
            notes.append(f"{ticker} did not have enough clean returns to compute benchmark metrics.")
            continue

        start_date = subset["date"].min()
        end_date = subset["date"].max()
        window = f"{start_date:%Y-%m} → {end_date:%Y-%m}" if pd.notna(start_date) and pd.notna(end_date) else comparable

        rows.append({
            "Reference": item["reference"],
            "Type": item["type"],
            "Window": window,
            "CAGR": _format_pct_signed(metrics["cagr"]),
            "Sharpe": f"{metrics['sharpe']:.2f}",
            "Vol": _format_pct_signed(metrics["annual_volatility"]),
            "MaxDD": f"{100.0 * metrics['max_drawdown']:.2f}%",
            "Reading": item["reading"],
            "Comparability": comparable,
        })

    return pd.DataFrame(rows), notes


def _render_benchmark_context(perf: dict, run_map: dict) -> None:
    st.markdown("### Where this result sits")
    st.caption(
        "These references are calculated from the Step 4 market-data panel using monthly returns. "
        "When possible, they use the same trailing OOS length as the engine result, so the comparison is contextual rather than a generic historical claim."
    )

    bench_df, notes = _benchmark_context_rows(perf, run_map)
    if isinstance(bench_df, pd.DataFrame) and not bench_df.empty:
        st.dataframe(bench_df, use_container_width=True, hide_index=True)
    else:
        st.info("Benchmark context is unavailable for this run because the Step 4 panel could not be read.")

    with st.expander("Benchmark methodology and exclusions", expanded=False):
        st.markdown(
            "- Benchmarks are **not hardcoded historical ranges**; they are recomputed from the current Step 4 panel.\n"
            "- The comparison uses the same monthly-return convention and, when available, the same trailing OOS length as the engine result.\n"
            "- This is educational context, not an investment recommendation and not a forecast.\n"
            "- Direct benchmark claims should always use the same date range, source, frequency and metric definitions."
        )
        if notes:
            st.caption("Panel notes")
            for note in notes:
                st.write(f"- {note}")

        st.caption("Shorter-history references deliberately left out of the main same-window table:")
        for item in SHORTER_HISTORY_CONTEXT:
            st.write(f"- {item}")

def render_post_run(run_result: dict) -> None:
    """Render the Gold Stable post-run surface.

    Active Step 5 flow:
    - real engine metrics
    - optional engine timing diagnostics
    - feature_mu metadata
    - OOS-return bridge into Step 6
    """
    run_map = _coerce_mapping(run_result)
    if not run_map:
        st.warning("Run result is missing or empty.")
        return

    perf = _coerce_mapping(run_map.get("performance_summary", {}))
    if not perf:
        st.warning("Run result payload is missing a performance summary.")
        return

    st.markdown("## 3. Real run result")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("CAGR", _pct(_safe_float(perf.get("cagr", 0.0))))
    with c2:
        st.metric("Sharpe", f"{_safe_float(perf.get('sharpe', 0.0)):.2f}")
    with c3:
        st.metric("Vol", _pct(_safe_float(perf.get("annual_volatility", perf.get("volatility", 0.0)))))
    with c4:
        st.metric("MaxDD", f"-{100.0 * abs(_safe_float(perf.get('max_drawdown', 0.0))):.2f}%")

    headline, body = _result_interpretation(perf)
    st.info(f"**{headline}:** {body}")

    _render_metric_explainer(perf)

    philosophy = str(st.session_state.get("investment_philosophy", "Balanced") or "Balanced")
    fit_level, fit_message = _philosophy_fit_message(perf, philosophy)
    if fit_level == "success":
        st.success(f"**Philosophy fit:** {fit_message}")
    elif fit_level == "warning":
        st.warning(f"**Philosophy fit:** {fit_message}")
    else:
        st.info(f"**Philosophy fit:** {fit_message}")

    _render_benchmark_context(perf, run_map)
    _render_what_to_watch(perf, philosophy)
    _render_reliability_note()

    run_signature = str(run_map.get("run_signature", "") or "")
    config_fingerprint = str(run_map.get("config_fingerprint", "") or "")
    run_timestamp = str(run_map.get("run_timestamp", "") or "")
    source = str(run_map.get("source", "micro_pipeline_real") or "micro_pipeline_real")
    panel_label = str(run_map.get("asset_panel_source_label", "Step 4 asset panel") or "Step 4 asset panel")
    panel_rows = int(_safe_float(run_map.get("asset_panel_n_rows", 0), 0))
    panel_assets = int(_safe_float(run_map.get("asset_panel_n_assets", 0), 0))
    panel_shape = run_map.get("panel_shape", None)

    st.markdown("## 4. Ready for projection")
    n_oos = _store_projection_bridge_context(run_map)
    if n_oos > 0:
        st.success(f"Projection bridge ready: {n_oos} monthly OOS returns stored for Step 6.")
    else:
        st.info("Projection bridge note: no OOS return series was found in this run payload; Step 6 can still use fallback profile assumptions.")

    with st.expander("Diagnostics (advanced)", expanded=False):
        meta_parts = [
            f"source={source}",
            f"asset_panel={panel_label}",
            f"rows={panel_rows}",
            f"assets={panel_assets}",
        ]
        if run_signature:
            meta_parts.append(f"run_signature={run_signature}")
        if config_fingerprint:
            meta_parts.append(f"config_fp={config_fingerprint}")
        if panel_shape:
            meta_parts.append(f"panel_shape={panel_shape}")
        st.caption(" · ".join(meta_parts))
        if run_timestamp:
            st.caption(f"run_timestamp={run_timestamp}")
        _render_engine_timing_block(run_map)
        _render_feature_mu_block(run_map)

    clear_retired_step5_state()

    st.markdown("---")
    left, right = st.columns(2)
    with left:
        if st.button("Back to Step 4", key="step5_back_to_step4"):
            st.session_state["current_step"] = 4
            st.rerun()
    with right:
        if st.button("Continue to Projection", key="step5_continue_to_step6", use_container_width=True):
            st.session_state["current_step"] = 6
            st.rerun()
