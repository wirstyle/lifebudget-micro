from __future__ import annotations

"""Strategy Engine result-reliability assessment.

This module adds a lightweight confidence layer for the Strategy Engine result without
running extra engines or downloading new data. It deliberately validates what can
be validated safely:
- real market-data basis;
- walk-forward/OOS structure;
- same-period benchmark context;
- metric-calculation sanity checks for known market assets;
- sensitivity evidence from already-tested engine-tuning candidates.

It does not claim that the strategy is a live fund track record or a forecast.
"""

import json
import math
import time
from dataclasses import fields
from typing import Any

import pandas as pd
import streamlit as st

from src.investment import MicroPipelineConfig, run_micro_investment_pipeline


AUTO_OPT_SUGGESTION_TIMING_KEY = "step5_auto_opt_suggestion_timing_v1"
AUTO_OPT_SUGGESTION_STATE_KEY = "step5_auto_opt_suggestion_v1"
PRESET_SUGGESTION_TIMING_KEY = "step5_preset_suggestion_timing_v1"
START_DATE_ROBUSTNESS_STATE_KEY = "step5_start_date_robustness_v1"
START_DATE_ROBUSTNESS_SCOPE_KEY = "step5_start_date_robustness_scope_v1"

REFERENCE_TICKERS: tuple[str, ...] = ("SPY", "QQQ", "GLD", "AGG", "TLT", "IEF", "VNQ")


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
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _parse_pct(value: Any) -> float | None:
    """Parse '13.49%' into 0.1349. Returns None on failure."""
    raw = str(value or "").strip().replace("%", "").replace(",", "")
    if not raw or raw in {"—", "-"}:
        return None
    try:
        return float(raw) / 100.0
    except Exception:
        return None


def _parse_float(value: Any) -> float | None:
    raw = str(value or "").strip().replace(",", "")
    if not raw or raw in {"—", "-"}:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _fmt_pct(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.2f}%"
    except Exception:
        return "—"


def _fmt_pp(value: Any) -> str:
    try:
        return f"{100.0 * float(value):+.2f}pp"
    except Exception:
        return "—"


def _fmt_sharpe_delta(value: Any) -> str:
    try:
        return f"{float(value):+.2f}"
    except Exception:
        return "—"


def _safe_return_series(raw: Any) -> pd.Series:
    try:
        series = pd.to_numeric(raw, errors="coerce").dropna().astype(float)
    except Exception:
        return pd.Series(dtype="float64")
    if series.empty:
        return series
    try:
        # Defensive fallback in case uploaded panels ever use percentages instead of decimals.
        if float(series.abs().median()) > 1.0:
            series = series / 100.0
    except Exception:
        pass
    return series


def _independent_return_metrics(returns: Any, *, periods_per_year: int = 12) -> dict:
    """Independent metric recomputation used only for validation.

    This intentionally duplicates the metric logic locally rather than importing
    the benchmark renderer. The goal is a small calculation sanity check, not a
    second model or a future-performance test.
    """
    r = _safe_return_series(returns)
    n = int(len(r))
    if n <= 1:
        return {}

    wealth = (1.0 + r).cumprod()
    final_wealth = float(wealth.iloc[-1]) if len(wealth) else 0.0
    years = float(n) / float(periods_per_year)
    cagr = float(final_wealth ** (1.0 / years) - 1.0) if final_wealth > 0.0 and years > 0.0 else 0.0
    vol = float(r.std(ddof=1) * (periods_per_year ** 0.5)) if n > 1 else 0.0
    annual_mean = float(r.mean() * periods_per_year) if n > 0 else 0.0
    sharpe = float(annual_mean / vol) if vol > 1e-12 else 0.0
    peak = wealth.cummax()
    drawdown = (wealth / peak) - 1.0
    maxdd = float(drawdown.min()) if len(drawdown) else 0.0
    return {
        "cagr": cagr,
        "annual_volatility": vol,
        "max_drawdown": maxdd,
        "sharpe": sharpe,
        "periods": n,
    }


def _extract_oos_returns(run_map: dict) -> list[float]:
    for key in [
        "oos_returns_monthly",
        "oos_returns_simple",
        "portfolio_returns",
        "oos_returns",
        "returns",
    ]:
        raw = run_map.get(key)
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


def _panel_df() -> pd.DataFrame:
    panel = st.session_state.get("asset_panel_df")
    if isinstance(panel, pd.DataFrame) and not panel.empty:
        return panel.copy()
    return pd.DataFrame()


def _period_bounds_from_window_meta(window_meta: dict, panel_df: pd.DataFrame, run_map: dict) -> tuple[pd.Period | None, pd.Period | None, int]:
    target_periods = _safe_int(window_meta.get("target_periods", 0), 0)
    if target_periods <= 0:
        target_periods = len(_extract_oos_returns(run_map))
    if target_periods <= 0 or panel_df.empty or "date" not in panel_df.columns:
        return None, None, 0

    dates = pd.to_datetime(panel_df["date"], errors="coerce").dropna()
    if dates.empty:
        return None, None, 0
    periods = sorted(pd.Series(dates.dt.to_period("M")).dropna().unique().tolist())
    if not periods:
        return None, None, 0
    selected = periods[-target_periods:] if len(periods) >= target_periods else periods
    return selected[0], selected[-1], len(selected)


def _filter_asset_returns_for_evaluation_window(panel_df: pd.DataFrame, ticker: str, start_p: pd.Period | None, end_p: pd.Period | None) -> pd.Series:
    if panel_df.empty or start_p is None or end_p is None:
        return pd.Series(dtype="float64")
    if not {"date", "asset", "return"}.issubset(set(panel_df.columns)):
        return pd.Series(dtype="float64")
    work = panel_df.copy()
    work["asset"] = work["asset"].astype(str).str.upper()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date", "asset", "return"])
    work["period"] = work["date"].dt.to_period("M")
    subset = work.loc[(work["asset"] == str(ticker).upper()) & (work["period"] >= start_p) & (work["period"] <= end_p)]
    subset = subset.sort_values("date")
    return _safe_return_series(subset["return"])


def _ticker_from_reference(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw or raw.lower().startswith("your strategy"):
        return ""
    if "—" in raw:
        raw = raw.split("—", 1)[0].strip()
    else:
        raw = raw.split(" ", 1)[0].strip()
    raw = raw.upper()
    return raw if raw in REFERENCE_TICKERS else ""


def _benchmark_validation_table(run_map: dict, benchmark_payload: dict) -> tuple[pd.DataFrame, dict]:
    payload = _coerce_mapping(benchmark_payload)
    bench_df = payload.get("bench_df")
    if not isinstance(bench_df, pd.DataFrame) or bench_df.empty:
        return pd.DataFrame(), {"status": "Unavailable", "mean_abs_pp": None, "max_abs_pp": None}

    panel_df = _panel_df()
    window_meta = _coerce_mapping(payload.get("window_meta", {}))
    start_p, end_p, selected_months = _period_bounds_from_window_meta(window_meta, panel_df, run_map)
    if start_p is None or end_p is None:
        return pd.DataFrame(), {"status": "Unavailable", "mean_abs_pp": None, "max_abs_pp": None}

    rows: list[dict] = []
    pp_errors: list[float] = []
    sharpe_errors: list[float] = []

    for _, row in bench_df.iterrows():
        ticker = _ticker_from_reference(row.get("Reference", ""))
        if not ticker:
            continue
        returns = _filter_asset_returns_for_evaluation_window(panel_df, ticker, start_p, end_p)
        check = _independent_return_metrics(returns)
        if not check:
            continue

        app_cagr = _parse_pct(row.get("CAGR"))
        app_vol = _parse_pct(row.get("Vol"))
        app_maxdd = _parse_pct(row.get("MaxDD"))
        app_sharpe = _parse_float(row.get("Sharpe"))
        if app_cagr is None or app_vol is None or app_maxdd is None or app_sharpe is None:
            continue

        delta_cagr = check["cagr"] - app_cagr
        delta_vol = check["annual_volatility"] - app_vol
        delta_maxdd = check["max_drawdown"] - app_maxdd
        delta_sharpe = check["sharpe"] - app_sharpe
        local_pp = max(abs(delta_cagr), abs(delta_vol), abs(delta_maxdd))
        pp_errors.append(local_pp)
        sharpe_errors.append(abs(delta_sharpe))
        status = "Passed" if local_pp <= 0.0015 and abs(delta_sharpe) <= 0.025 else "Review"

        rows.append(
            {
                "Reference asset": ticker,
                "App CAGR": _fmt_pct(app_cagr),
                "Check CAGR": _fmt_pct(check["cagr"]),
                "Δ CAGR": _fmt_pp(delta_cagr),
                "App Vol": _fmt_pct(app_vol),
                "Check Vol": _fmt_pct(check["annual_volatility"]),
                "Δ Vol": _fmt_pp(delta_vol),
                "App MaxDD": _fmt_pct(app_maxdd),
                "Check MaxDD": _fmt_pct(check["max_drawdown"]),
                "Δ MaxDD": _fmt_pp(delta_maxdd),
                "App Sharpe": f"{app_sharpe:.2f}",
                "Check Sharpe": f"{check['sharpe']:.2f}",
                "Δ Sharpe": _fmt_sharpe_delta(delta_sharpe),
                "Status": status,
            }
        )

    if not rows:
        return pd.DataFrame(), {"status": "Unavailable", "mean_abs_pp": None, "max_abs_pp": None}

    mean_abs_pp = float(sum(pp_errors) / len(pp_errors)) if pp_errors else 0.0
    max_abs_pp = float(max(pp_errors)) if pp_errors else 0.0
    max_abs_sharpe = float(max(sharpe_errors)) if sharpe_errors else 0.0
    status = "Passed" if max_abs_pp <= 0.0015 and max_abs_sharpe <= 0.025 else "Review"
    return pd.DataFrame(rows), {
        "status": status,
        "mean_abs_pp": mean_abs_pp,
        "max_abs_pp": max_abs_pp,
        "max_abs_sharpe": max_abs_sharpe,
        "n_assets": len(rows),
        "months": selected_months,
        "window": f"{start_p} → {end_p}",
    }


def _walk_forward_status(run_map: dict, benchmark_payload: dict) -> tuple[str, str]:
    payload = _coerce_mapping(benchmark_payload)
    window_meta = _coerce_mapping(payload.get("window_meta", {}))
    n_oos = _safe_int(window_meta.get("target_periods", 0), 0)
    if n_oos <= 0:
        n_oos = len(_extract_oos_returns(run_map))
    warmup = window_meta.get("warmup_periods")
    warmup_n = _safe_int(warmup, 0) if warmup is not None else 0
    eval_window = str(window_meta.get("evaluation_window", "") or "")

    if n_oos >= 120 and warmup_n >= 60:
        return "Strong", f"Uses {n_oos} OOS monthly returns after approximately {warmup_n} warm-up/training months ({eval_window})."
    if n_oos >= 60:
        detail = f"Uses {n_oos} OOS monthly returns"
        if warmup_n > 0:
            detail += f" after approximately {warmup_n} warm-up/training months"
        return "Moderate", detail + "."
    if n_oos > 0:
        return "Limited", f"Uses {n_oos} OOS monthly returns, which is useful but relatively short."
    return "Unavailable", "OOS return length was not found in the run payload."


def _data_basis_status(run_map: dict) -> tuple[str, str]:
    panel_df = _panel_df()
    label = str(run_map.get("asset_panel_source_label", st.session_state.get("asset_panel_source_label", "selected market-data panel")) or "selected market-data panel")
    rows = _safe_int(run_map.get("asset_panel_n_rows", len(panel_df) if isinstance(panel_df, pd.DataFrame) else 0), 0)
    assets = _safe_int(run_map.get("asset_panel_n_assets", panel_df["asset"].nunique() if isinstance(panel_df, pd.DataFrame) and "asset" in panel_df.columns else 0), 0)
    if rows > 0 and assets > 0:
        if "yahoo" in label.lower():
            return "Strong", f"Uses real historical returns from the selected market-data panel ({label}; {rows:,} rows, {assets} assets). In deployed mode this is a fixed Yahoo-generated cached panel, not a live market-data pull."
        return "Available", f"Uses the selected market-data panel ({rows:,} rows, {assets} assets)."
    return "Unavailable", "The selected market-data panel could not be read for this reliability check."


def _same_period_status(benchmark_payload: dict) -> tuple[str, str]:
    payload = _coerce_mapping(benchmark_payload)
    window_meta = _coerce_mapping(payload.get("window_meta", {}))
    target_periods = _safe_int(window_meta.get("target_periods", 0), 0)
    eval_window = str(window_meta.get("evaluation_window", "") or "")
    bench_df = payload.get("bench_df")
    n_refs = int(len(bench_df)) - 1 if isinstance(bench_df, pd.DataFrame) and not bench_df.empty else 0
    if target_periods > 0 and n_refs > 0:
        return "Strong", f"Compares the strategy and {n_refs} reference assets over the same evaluated period ({eval_window}, {target_periods} months)."
    if n_refs > 0:
        return "Available", f"Reference assets are present, but the exact evaluated-period length was not resolved."
    return "Unavailable", "Same-period benchmark rows were not available."


def _benchmark_relative_status(run_map: dict, benchmark_payload: dict) -> tuple[str, str]:
    perf = _coerce_mapping(run_map.get("performance_summary", {}))
    strategy_cagr = _safe_float(perf.get("cagr", 0.0), 0.0)
    strategy_vol = _safe_float(perf.get("annual_volatility", perf.get("volatility", 0.0)), 0.0)
    strategy_dd = abs(_safe_float(perf.get("max_drawdown", 0.0), 0.0))

    payload = _coerce_mapping(benchmark_payload)
    bench_df = payload.get("bench_df")
    if not isinstance(bench_df, pd.DataFrame) or bench_df.empty:
        return "Unavailable", "Benchmark rows were not available for profile consistency."

    lookup: dict[str, dict] = {}
    for _, row in bench_df.iterrows():
        ticker = _ticker_from_reference(row.get("Reference", ""))
        if ticker:
            lookup[ticker] = dict(row)

    spy = lookup.get("SPY", {})
    qqq = lookup.get("QQQ", {})
    spy_vol = _parse_pct(spy.get("Vol")) if spy else None
    spy_dd = abs(_parse_pct(spy.get("MaxDD")) or 0.0) if spy else None
    qqq_cagr = _parse_pct(qqq.get("CAGR")) if qqq else None
    qqq_vol = _parse_pct(qqq.get("Vol")) if qqq else None
    qqq_dd = abs(_parse_pct(qqq.get("MaxDD")) or 0.0) if qqq else None

    checks: list[str] = []
    if spy_vol is not None and strategy_vol <= spy_vol:
        checks.append("lower volatility than SPY")
    if spy_dd is not None and strategy_dd <= spy_dd:
        checks.append("lower drawdown than SPY")
    if qqq_cagr is not None and strategy_cagr <= qqq_cagr and qqq_vol is not None and strategy_vol <= qqq_vol:
        checks.append("lower-return/lower-volatility profile than QQQ")
    if qqq_dd is not None and strategy_dd <= qqq_dd:
        checks.append("lower drawdown than QQQ")

    if len(checks) >= 2:
        return "Consistent", "The strategy behaves like a risk-controlled portfolio: " + "; ".join(checks[:3]) + "."
    if checks:
        return "Mixed", "Some benchmark-relative properties are consistent, but the profile should still be reviewed."
    return "Review", "The benchmark-relative risk/return profile is not clearly aligned from the available benchmark rows."


def _parameter_sensitivity_status() -> tuple[str, str]:
    timing = _coerce_mapping(st.session_state.get(AUTO_OPT_SUGGESTION_TIMING_KEY, {}))
    candidate_count = _safe_int(timing.get("candidate_count", 0), 0)
    passed_count = _safe_int(timing.get("accepted_count", 0), 0)
    if candidate_count <= 0:
        return "Not run", "No engine-tuning candidate test has been completed for this result."
    if passed_count >= 2:
        return "Moderate", f"{candidate_count} technical candidates were tested and {passed_count} passed the gate; the final result uses the highest-scoring passed candidate, but multiple nearby settings were viable."
    if passed_count == 1:
        return "Selective", f"{candidate_count} technical candidates were tested and one passed the gate; this gives useful but narrow sensitivity evidence."
    return "Stable/Converged", f"{candidate_count} technical candidates were tested and none materially improved the current setup."


def _overall_confidence(component_statuses: list[str]) -> str:
    lowered = [str(x).lower() for x in component_statuses]
    if any(x in {"unavailable"} for x in lowered[:3]):
        return "Limited"
    strong_like = sum(1 for x in lowered if x in {"strong", "passed", "consistent"})
    review_like = sum(1 for x in lowered if x in {"review", "limited", "unavailable"})
    if strong_like >= 4 and review_like == 0:
        return "Moderate-to-Strong"
    if strong_like >= 3:
        return "Moderate"
    return "Limited-to-Moderate"



# ---------------------------------------------------------------------------
# Optional start-date robustness check (manual, no new downloads)
# ---------------------------------------------------------------------------


def _coerce_cfg_payload(cfg_payload: Any) -> dict:
    payload = _coerce_mapping(cfg_payload)
    try:
        valid = {f.name for f in fields(MicroPipelineConfig)}
    except Exception:
        valid = set()
    return {key: value for key, value in payload.items() if key in valid}


def _base_cfg_payload_from_run(run_map: dict) -> dict:
    """Resolve the full engine configuration used by the current Strategy Engine run."""
    run_map = _coerce_mapping(run_map)
    for raw in (
        run_map.get("config_dict"),
        st.session_state.get("last_engine_config"),
        st.session_state.get("step5_last_cfg_final"),
        run_map.get("config"),
    ):
        payload = _coerce_cfg_payload(raw)
        if payload:
            return payload
    return _coerce_cfg_payload({})


def _coerce_micro_cfg(cfg_payload: Any) -> MicroPipelineConfig:
    payload = _coerce_cfg_payload(cfg_payload)
    return MicroPipelineConfig(**payload)


def _date_period_label(value: Any) -> str:
    try:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.notna(ts):
            return f"{ts:%Y-%m}"
    except Exception:
        pass
    return "—"


def _clean_asset_panel_for_robustness(panel_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(panel_df, pd.DataFrame) or panel_df.empty:
        return pd.DataFrame()
    required = {"date", "asset", "return"}
    if not required.issubset(set(panel_df.columns)):
        return pd.DataFrame()
    work = panel_df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["asset"] = work["asset"].astype(str).str.upper().str.strip()
    work["return"] = pd.to_numeric(work["return"], errors="coerce")
    work = work.dropna(subset=["date", "asset", "return"]).sort_values(["date", "asset"]).reset_index(drop=True)
    return work


def _panel_month_bounds(panel_df: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None, int]:
    if not isinstance(panel_df, pd.DataFrame) or panel_df.empty or "date" not in panel_df.columns:
        return None, None, 0
    dates = pd.to_datetime(panel_df["date"], errors="coerce").dropna()
    if dates.empty:
        return None, None, 0
    periods = pd.Series(dates.dt.to_period("M")).dropna().drop_duplicates().sort_values()
    if periods.empty:
        return None, None, 0
    start_ts = periods.iloc[0].to_timestamp()
    end_ts = periods.iloc[-1].to_timestamp()
    return start_ts, end_ts, int(len(periods))


def _robustness_scope(run_map: dict, panel_df: pd.DataFrame, cfg_payload: dict) -> str:
    run_map = _coerce_mapping(run_map)
    start_ts, end_ts, months = _panel_month_bounds(panel_df)
    payload = {
        "run_signature": str(run_map.get("run_signature", "") or ""),
        "config_fingerprint": str(run_map.get("config_fingerprint", "") or ""),
        "cfg": _coerce_cfg_payload(cfg_payload),
        "panel_start": _date_period_label(start_ts),
        "panel_end": _date_period_label(end_ts),
        "panel_months": months,
        "rows": int(len(panel_df)) if isinstance(panel_df, pd.DataFrame) else 0,
        "assets": sorted(panel_df["asset"].dropna().astype(str).unique().tolist()) if isinstance(panel_df, pd.DataFrame) and "asset" in panel_df.columns else [],
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _build_start_date_candidates(panel_df: pd.DataFrame, cfg_payload: dict) -> list[dict]:
    """Return baseline + selected candidate starts with a safe OOS-length guard."""
    start_ts, end_ts, _months = _panel_month_bounds(panel_df)
    if start_ts is None or end_ts is None:
        return []
    min_train = _safe_int(_coerce_mapping(cfg_payload).get("min_train", 120), 120)
    raw_max_oos = _coerce_mapping(cfg_payload).get("max_oos_points", None)
    try:
        max_oos = int(raw_max_oos) if raw_max_oos is not None else None
    except Exception:
        max_oos = None

    desired_starts = [
        start_ts,
        pd.Timestamp("2008-01-01"),
        pd.Timestamp("2010-01-01"),
        pd.Timestamp("2012-01-01"),
        pd.Timestamp("2015-01-01"),
    ]
    unique: list[pd.Timestamp] = []
    for raw in desired_starts:
        ts = pd.to_datetime(raw, errors="coerce")
        if pd.isna(ts):
            continue
        ts = pd.Timestamp(ts).to_period("M").to_timestamp()
        if ts < start_ts or ts > end_ts:
            continue
        if not any(existing.to_period("M") == ts.to_period("M") for existing in unique):
            unique.append(ts)

    rows: list[dict] = []
    for idx, start in enumerate(unique):
        subset = panel_df.loc[pd.to_datetime(panel_df["date"], errors="coerce") >= start].copy()
        _sub_start, _sub_end, sub_months = _panel_month_bounds(subset)
        potential_oos = max(0, int(sub_months) - int(min_train))
        if max_oos is not None and max_oos > 0:
            potential_oos = min(potential_oos, int(max_oos))
        is_baseline = idx == 0 and start.to_period("M") == start_ts.to_period("M")
        if potential_oos >= 60:
            read = "Robustness candidate"
            should_run = not is_baseline
        elif potential_oos >= 36:
            read = "Short but usable"
            should_run = not is_baseline
        elif potential_oos > 0:
            read = "Too short / diagnostic only"
            should_run = False
        else:
            read = "Too short"
            should_run = False
        rows.append(
            {
                "start": start,
                "label": _date_period_label(start),
                "is_baseline": bool(is_baseline),
                "panel_months": int(sub_months),
                "potential_oos": int(potential_oos),
                "should_run": bool(should_run),
                "read": "Current baseline" if is_baseline else read,
            }
        )
    return rows


def _perf_summary_from_run(run_map: dict) -> dict:
    perf = _coerce_mapping(_coerce_mapping(run_map).get("performance_summary", {}))
    return {
        "cagr": _safe_float(perf.get("cagr", 0.0), 0.0),
        "annual_volatility": _safe_float(perf.get("annual_volatility", perf.get("volatility", 0.0)), 0.0),
        "max_drawdown": -abs(_safe_float(perf.get("max_drawdown", 0.0), 0.0)),
        "sharpe": _safe_float(perf.get("sharpe", 0.0), 0.0),
        "periods": _safe_int(perf.get("periods", 0), 0),
    }


def _robustness_row_from_perf(candidate: dict, perf: dict, *, elapsed_sec: float = 0.0, status: str = "Completed", error: str = "") -> dict:
    start = candidate.get("start")
    fallback_oos = _safe_int(candidate.get("potential_oos", 0), 0)
    oos = _safe_int(perf.get("periods", fallback_oos), fallback_oos)
    # Some engine payloads do not expose periods for candidate reruns even though
    # the rerun completed. In that case, show the expected OOS length from the
    # filtered panel rather than a misleading zero.
    if status == "Completed" and oos <= 0 and fallback_oos > 0:
        oos = int(fallback_oos)
    eval_start = "—"
    try:
        if start is not None and oos > 0:
            min_train = max(0, _safe_int(candidate.get("panel_months", 0), 0) - int(candidate.get("potential_oos", oos)))
            # The engine evaluation starts after min_train monthly observations in the filtered panel.
            eval_start_ts = pd.Timestamp(start) + pd.DateOffset(months=int(min_train))
            eval_start = _date_period_label(eval_start_ts)
    except Exception:
        eval_start = "—"
    return {
        "Start date": str(candidate.get("label", "—")),
        "Evaluated period": f"{eval_start} → latest" if eval_start != "—" else "—",
        "OOS months": int(oos),
        "CAGR": _fmt_pct(perf.get("cagr", 0.0)) if status == "Completed" else "—",
        "Vol": _fmt_pct(perf.get("annual_volatility", 0.0)) if status == "Completed" else "—",
        "MaxDD": _fmt_pct(perf.get("max_drawdown", 0.0)) if status == "Completed" else "—",
        "Sharpe": f"{_safe_float(perf.get('sharpe', 0.0), 0.0):.2f}" if status == "Completed" else "—",
        "Reliability read": str(candidate.get("read", "")) if status == "Completed" else str(candidate.get("read", status)),
        "Status": status,
        "Seconds": round(float(elapsed_sec), 2) if elapsed_sec > 0 else 0.0,
        "Error": str(error or ""),
        "_cagr": _safe_float(perf.get("cagr", 0.0), 0.0) if status == "Completed" else None,
        "_vol": _safe_float(perf.get("annual_volatility", 0.0), 0.0) if status == "Completed" else None,
        "_maxdd_abs": abs(_safe_float(perf.get("max_drawdown", 0.0), 0.0)) if status == "Completed" else None,
        "_sharpe": _safe_float(perf.get("sharpe", 0.0), 0.0) if status == "Completed" else None,
    }

def _run_start_date_robustness(run_map: dict) -> dict:
    run_map = _coerce_mapping(run_map)
    panel_df = _clean_asset_panel_for_robustness(_panel_df())
    cfg_payload = _base_cfg_payload_from_run(run_map)
    scope = _robustness_scope(run_map, panel_df, cfg_payload)
    started = time.perf_counter()
    rows: list[dict] = []

    if panel_df.empty:
        return {"scope": scope, "rows": [], "summary": {"status": "Unavailable", "message": "selected market-data panel is unavailable."}, "elapsed_sec": 0.0}
    if not cfg_payload:
        return {"scope": scope, "rows": [], "summary": {"status": "Unavailable", "message": "Engine configuration is unavailable."}, "elapsed_sec": 0.0}

    cfg = _coerce_micro_cfg(cfg_payload)
    candidates = _build_start_date_candidates(panel_df, cfg_payload)
    if not candidates:
        return {"scope": scope, "rows": [], "summary": {"status": "Unavailable", "message": "No valid start-date candidates were available."}, "elapsed_sec": 0.0}

    for candidate in candidates:
        if bool(candidate.get("is_baseline", False)):
            rows.append(_robustness_row_from_perf(candidate, _perf_summary_from_run(run_map), status="Completed"))
            continue
        if not bool(candidate.get("should_run", False)):
            rows.append(_robustness_row_from_perf(candidate, {}, status="Skipped"))
            continue
        subset = panel_df.loc[pd.to_datetime(panel_df["date"], errors="coerce") >= pd.Timestamp(candidate["start"])].copy()
        t0 = time.perf_counter()
        try:
            result = run_micro_investment_pipeline(subset, cfg=cfg)
            elapsed = float(time.perf_counter() - t0)
            rows.append(_robustness_row_from_perf(candidate, _perf_summary_from_run(_coerce_mapping(result)), elapsed_sec=elapsed, status="Completed"))
        except Exception as exc:
            elapsed = float(time.perf_counter() - t0)
            rows.append(_robustness_row_from_perf(candidate, {}, elapsed_sec=elapsed, status="Error", error=str(exc)))

    summary = _summarise_start_date_robustness(rows)
    return {
        "scope": scope,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": float(time.perf_counter() - started),
        "rows": rows,
        "summary": summary,
    }


def _summarise_start_date_robustness(rows: list[dict]) -> dict:
    valid = [r for r in list(rows or []) if str(r.get("Status", "")) == "Completed" and r.get("_cagr") is not None]
    if len(valid) < 2:
        return {
            "status": "Limited",
            "message": "Fewer than two completed start-date windows were available, so robustness cannot be assessed strongly.",
        }
    cagrs = [_safe_float(r.get("_cagr"), 0.0) for r in valid]
    sharpes = [_safe_float(r.get("_sharpe"), 0.0) for r in valid]
    maxdds = [_safe_float(r.get("_maxdd_abs"), 0.0) for r in valid]
    vols = [_safe_float(r.get("_vol"), 0.0) for r in valid]
    cagr_range = max(cagrs) - min(cagrs)
    sharpe_range = max(sharpes) - min(sharpes)
    maxdd_range = max(maxdds) - min(maxdds)
    vol_range = max(vols) - min(vols)
    all_cagr_positive = all(float(x) > 0.0 for x in cagrs)
    downside_stable = maxdd_range <= 0.05
    moderate_return_sensitivity = cagr_range <= 0.060 and sharpe_range <= 0.45

    if cagr_range <= 0.030 and sharpe_range <= 0.25 and maxdd_range <= 0.10:
        status = "Strong"
        message = "The result is reasonably stable across the tested start dates."
        downside_status = "Strong"
        return_sensitivity_status = "Strong"
    elif all_cagr_positive and downside_stable and moderate_return_sensitivity:
        status = "Moderate-to-Strong"
        message = (
            "The result is structurally stable across start dates: drawdown remains controlled and CAGR stays positive. "
            "The main sensitivity is return strength, not downside risk, so the result is not labelled fully Strong."
        )
        downside_status = "Strong"
        return_sensitivity_status = "Moderate"
    elif cagr_range <= 0.060 and sharpe_range <= 0.45 and maxdd_range <= 0.18:
        status = "Moderate"
        message = "The result changes across start dates, but not enough to invalidate the current interpretation."
        downside_status = "Moderate"
        return_sensitivity_status = "Moderate"
    else:
        status = "Sensitive"
        message = "The result is materially sensitive to the selected historical start date and should be treated cautiously."
        downside_status = "Sensitive"
        return_sensitivity_status = "Sensitive"
    return {
        "status": status,
        "message": message,
        "downside_status": downside_status,
        "return_sensitivity_status": return_sensitivity_status,
        "completed_windows": len(valid),
        "cagr_min": min(cagrs),
        "cagr_max": max(cagrs),
        "sharpe_min": min(sharpes),
        "sharpe_max": max(sharpes),
        "maxdd_min": min(maxdds),
        "maxdd_max": max(maxdds),
        "vol_min": min(vols),
        "vol_max": max(vols),
        "cagr_range": cagr_range,
        "sharpe_range": sharpe_range,
        "maxdd_range": maxdd_range,
        "vol_range": vol_range,
    }


def _render_start_date_robustness_check(run_map: dict, *, inline_details: bool = False) -> None:
    run_map = _coerce_mapping(run_map)
    panel_df = _clean_asset_panel_for_robustness(_panel_df())
    cfg_payload = _base_cfg_payload_from_run(run_map)
    scope = _robustness_scope(run_map, panel_df, cfg_payload)
    saved_scope = str(st.session_state.get(START_DATE_ROBUSTNESS_SCOPE_KEY, "") or "")
    saved_payload = _coerce_mapping(st.session_state.get(START_DATE_ROBUSTNESS_STATE_KEY, {}))

    if inline_details:
        st.markdown("**Optional start-date robustness check**")
    ctx = st.container() if inline_details else st.expander("Optional start-date robustness check", expanded=False)
    with ctx:
        st.write(
            "This optional check reruns the same current strategy configuration using different historical start dates from the existing selected market-data panel. "
            "It checks whether the result depends too heavily on one specific historical window. It does not download new data and it does not predict future returns."
        )
        st.caption(
            "Default starts are the current panel start plus 2008, 2010 and 2012 when enough OOS history remains. 2015 is shown only as a too-short diagnostic when applicable. "
            "The current baseline row is reused from the existing result, so it is not rerun."
        )

        if saved_scope != scope and saved_payload:
            st.info("The saved robustness check belongs to a previous result/configuration. Run the check again to update it for the current setup.")

        c1, c2 = st.columns([1.2, 1.0])
        with c1:
            run_clicked = st.button("Run start-date robustness check", key="step5_run_start_date_robustness")
        with c2:
            clear_clicked = st.button("Clear robustness check", key="step5_clear_start_date_robustness")

        if clear_clicked:
            st.session_state[START_DATE_ROBUSTNESS_STATE_KEY] = {}
            st.session_state[START_DATE_ROBUSTNESS_SCOPE_KEY] = ""
            st.rerun()

        if run_clicked:
            if panel_df.empty:
                st.warning("Cannot run robustness check because the selected market-data panel is unavailable.")
            elif not cfg_payload:
                st.warning("Cannot run robustness check because the current engine configuration could not be resolved.")
            else:
                with st.spinner("Running start-date robustness check. This may take around 60–90 seconds..."):
                    payload = _run_start_date_robustness(run_map)
                st.session_state[START_DATE_ROBUSTNESS_STATE_KEY] = dict(payload)
                st.session_state[START_DATE_ROBUSTNESS_SCOPE_KEY] = str(payload.get("scope", scope) or scope)
                st.rerun()

        payload = _coerce_mapping(st.session_state.get(START_DATE_ROBUSTNESS_STATE_KEY, {}))
        if str(st.session_state.get(START_DATE_ROBUSTNESS_SCOPE_KEY, "") or "") != scope:
            payload = {}

        if not payload:
            st.caption("No start-date robustness check has been run for the current result yet.")
            return

        summary = _coerce_mapping(payload.get("summary", {}))
        status = str(summary.get("status", "Unavailable") or "Unavailable")
        message = str(summary.get("message", "") or "")
        elapsed = _safe_float(payload.get("elapsed_sec", 0.0), 0.0)
        if status == "Strong":
            st.success(f"Start-date robustness: **{status}** — {message}")
        elif status == "Moderate-to-Strong":
            st.info(f"Start-date robustness: **{status}** — {message}")
        elif status == "Moderate":
            st.info(f"Start-date robustness: **{status}** — {message}")
        elif status == "Sensitive":
            st.warning(f"Start-date robustness: **{status}** — {message}")
        else:
            st.info(f"Start-date robustness: **{status}** — {message}")

        downside_status = str(summary.get("downside_status", "") or "")
        return_sensitivity_status = str(summary.get("return_sensitivity_status", "") or "")
        if downside_status or return_sensitivity_status:
            st.caption(
                f"Robustness split: downside robustness={downside_status or '—'} · "
                f"return sensitivity={return_sensitivity_status or '—'}."
            )

        if summary.get("completed_windows"):
            st.caption(
                f"CAGR range: {_fmt_pct(summary.get('cagr_min'))} → {_fmt_pct(summary.get('cagr_max'))} · "
                f"Sharpe range: {_safe_float(summary.get('sharpe_min'), 0.0):.2f} → {_safe_float(summary.get('sharpe_max'), 0.0):.2f} · "
                f"MaxDD range: -{100.0 * _safe_float(summary.get('maxdd_min'), 0.0):.2f}% → -{100.0 * _safe_float(summary.get('maxdd_max'), 0.0):.2f}% · "
                f"runtime={elapsed:.2f}s."
            )

        rows = list(payload.get("rows", []) or [])
        if rows:
            display_rows = []
            for row in rows:
                row_map = _coerce_mapping(row)
                display_rows.append(
                    {
                        "Start date": row_map.get("Start date", "—"),
                        "Evaluated period": row_map.get("Evaluated period", "—"),
                        "OOS months": row_map.get("OOS months", 0),
                        "CAGR": row_map.get("CAGR", "—"),
                        "Vol": row_map.get("Vol", "—"),
                        "MaxDD": row_map.get("MaxDD", "—"),
                        "Sharpe": row_map.get("Sharpe", "—"),
                        "Reliability read": row_map.get("Reliability read", ""),
                        "Status": row_map.get("Status", ""),
                        "Seconds": row_map.get("Seconds", 0.0),
                    }
                )
            st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)
            st.caption(
                "The selected configuration was not only evaluated on one fixed historical start date. "
                "A start-date robustness diagnostic was run across alternative historical windows. "
                "The result remained positive across valid windows; if the shorter 2012 window shows lower Sharpe and CAGR, "
                "the app treats that as return sensitivity rather than automatically invalidating the result."
            )
            st.caption(
                "The main sensitivity is return strength, not drawdown: CAGR and Sharpe can fall in the shorter 2012 window, "
                "while drawdown may remain similar or better. The 2012 row is useful as a short-window stress check, "
                "but it has fewer OOS months than the 2005/2008/2010 rows, so it should not dominate the interpretation."
            )

def _current_start_date_robustness_payload(run_map: dict | None = None) -> dict:
    """Return the saved robustness payload only if it belongs to the current result."""
    run_map = _coerce_mapping(run_map or {})
    payload = _coerce_mapping(st.session_state.get(START_DATE_ROBUSTNESS_STATE_KEY, {}))
    if not payload:
        return {}

    panel_df = _clean_asset_panel_for_robustness(_panel_df())
    cfg_payload = _base_cfg_payload_from_run(run_map)
    current_scope = _robustness_scope(run_map, panel_df, cfg_payload)
    saved_scope = str(st.session_state.get(START_DATE_ROBUSTNESS_SCOPE_KEY, "") or "")
    if saved_scope and current_scope and saved_scope != current_scope:
        return {}
    return dict(payload)


def _robustness_timing_rows(payload: dict) -> list[dict]:
    rows: list[dict] = []
    for raw in list(_coerce_mapping(payload).get("rows", []) or []):
        row = _coerce_mapping(raw)
        status_raw = str(row.get("Status", "") or "").strip().lower()
        read_raw = str(row.get("Reliability read", "") or "").strip().lower()
        seconds = _safe_float(row.get("Seconds", 0.0), 0.0)
        if "current baseline" in read_raw:
            status = "baseline reused"
        elif status_raw == "completed" and seconds > 0:
            status = "executed rerun"
        elif status_raw == "skipped":
            status = "too short / diagnostic"
        elif status_raw == "error":
            status = "error"
        else:
            status = status_raw or "not run"
        rows.append({"start date": str(row.get("Start date", "—") or "—"), "status": status, "seconds": round(seconds, 2)})
    return rows


def render_start_date_robustness_timing_block(run_map: dict | None = None) -> None:
    """Render optional start-date robustness timing in Strategy Engine diagnostics."""
    payload = _current_start_date_robustness_payload(run_map)
    if not payload:
        return

    rows = _robustness_timing_rows(payload)
    total_seconds = _safe_float(payload.get("elapsed_sec", 0.0), 0.0)
    checked_count = int(len(rows))
    executed_count = int(sum(1 for row in rows if str(row.get("status", "")).lower() == "executed rerun"))
    if total_seconds <= 0.0 and checked_count <= 0:
        return

    st.markdown("### Start-date robustness timing")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Robustness test", f"{total_seconds:.2f}s" if total_seconds > 0 else "—")
    with c2:
        st.metric("Start dates checked", int(checked_count))
    with c3:
        st.metric("Executed reruns", int(executed_count))

    st.caption(
        "This is the extra time used by the optional Strategy Engine start-date robustness check. "
        "It is separate from the main engine run and from the preset / engine-tuning suggestion tests. "
        "The current baseline row is reused, so it is not rerun."
    )

    if rows:
        with st.container(border=True):
            st.caption("Start-date robustness timing breakdown")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

def _reliability_component_rows(run_map: dict, benchmark_payload: dict) -> tuple[list[dict[str, str]], str, pd.DataFrame, dict]:
    """Build the reliability component rows and benchmark-validation payload."""
    validation_df, validation_meta = _benchmark_validation_table(run_map, benchmark_payload)
    data_status, data_meaning = _data_basis_status(run_map)
    walk_status, walk_meaning = _walk_forward_status(run_map, benchmark_payload)
    same_status, same_meaning = _same_period_status(benchmark_payload)
    calc_status = str(validation_meta.get("status", "Unavailable") or "Unavailable")
    if calc_status == "Passed":
        mean_abs_pp = validation_meta.get("mean_abs_pp")
        max_abs_pp = validation_meta.get("max_abs_pp")
        calc_meaning = (
            "Known benchmark assets were recomputed independently from the current panel. "
            f"Mean metric difference was about {100.0 * _safe_float(mean_abs_pp, 0.0):.2f}pp; "
            f"max difference was about {100.0 * _safe_float(max_abs_pp, 0.0):.2f}pp."
        )
    elif calc_status == "Review":
        calc_meaning = "Known benchmark assets were recomputed, but at least one metric difference deserves review."
    else:
        calc_meaning = "Benchmark calculation validation could not be completed for this run."

    profile_status, profile_meaning = _benchmark_relative_status(run_map, benchmark_payload)
    sensitivity_status, sensitivity_meaning = _parameter_sensitivity_status()
    future_status = "Not claimed"
    future_meaning = "Historical walk-forward backtest only; this is not a live fund track record and not a forecast."

    components = [
        {"Reliability component": "Real market-data snapshot", "Status": data_status, "Meaning": data_meaning},
        {"Reliability component": "Walk-forward evaluation", "Status": walk_status, "Meaning": walk_meaning},
        {"Reliability component": "Same-period benchmark comparison", "Status": same_status, "Meaning": same_meaning},
        {"Reliability component": "Benchmark calculation validation", "Status": calc_status, "Meaning": calc_meaning},
        {"Reliability component": "Benchmark-relative profile", "Status": profile_status, "Meaning": profile_meaning},
        {"Reliability component": "Parameter sensitivity", "Status": sensitivity_status, "Meaning": sensitivity_meaning},
        {"Reliability component": "Future guarantee", "Status": future_status, "Meaning": future_meaning},
    ]
    overall = _overall_confidence([row["Status"] for row in components])
    return components, overall, validation_df, validation_meta


def _start_date_robustness_visible_summary(run_map: dict) -> tuple[dict, dict, str, str]:
    """Return saved robustness payload plus display status/message."""
    payload = _current_start_date_robustness_payload(run_map)
    summary = _coerce_mapping(payload.get("summary", {})) if payload else {}
    status = str(summary.get("status", "Not run yet") or "Not run yet")
    message = str(summary.get("message", "") or "")
    return payload, summary, status, message


def _render_status_callout(status: str, message: str) -> None:
    label = f"Start-date robustness: **{status}**"
    text = f"{label} — {message}" if message else label
    if status in {"Strong"}:
        st.success(text)
    elif status in {"Moderate-to-Strong", "Moderate"}:
        st.info(text)
    elif status in {"Sensitive"}:
        st.warning(text)
    elif status == "Not run yet":
        st.caption("Start-date robustness has not been run yet for this result.")
    else:
        st.info(text)


def _render_start_date_robustness_actions(run_map: dict) -> None:
    """Render robustness action buttons outside the detailed evidence expander."""
    run_map = _coerce_mapping(run_map)
    panel_df = _clean_asset_panel_for_robustness(_panel_df())
    cfg_payload = _base_cfg_payload_from_run(run_map)
    scope = _robustness_scope(run_map, panel_df, cfg_payload)
    saved_scope = str(st.session_state.get(START_DATE_ROBUSTNESS_SCOPE_KEY, "") or "")
    saved_payload = _coerce_mapping(st.session_state.get(START_DATE_ROBUSTNESS_STATE_KEY, {}))

    if saved_scope != scope and saved_payload:
        st.info("The saved robustness check belongs to a previous result/configuration. Run the check again to update it for the current Strategy Engine result.")

    c1, c2 = st.columns([1.2, 1.0])
    with c1:
        run_clicked = st.button(
            "Run robustness check",
            key="step5_run_start_date_robustness",
            use_container_width=True,
            help="Tests the same Strategy Engine setup across alternative historical start dates from the selected market-data panel.",
        )
    with c2:
        clear_clicked = st.button(
            "Clear robustness result",
            key="step5_clear_start_date_robustness",
            use_container_width=True,
            disabled=not bool(saved_payload),
            help="Remove the saved start-date robustness result for this Strategy Engine run.",
        )

    st.caption("The robustness check uses the selected market-data panel already loaded in the app. It does not download new data and it does not predict future returns.")

    if clear_clicked:
        st.session_state[START_DATE_ROBUSTNESS_STATE_KEY] = {}
        st.session_state[START_DATE_ROBUSTNESS_SCOPE_KEY] = ""
        st.rerun()

    if run_clicked:
        if panel_df.empty:
            st.warning("Cannot run robustness check because the selected market-data panel is unavailable.")
        elif not cfg_payload:
            st.warning("Cannot run robustness check because the current engine configuration could not be resolved.")
        else:
            with st.spinner("Running start-date robustness check. This may take around 60–90 seconds..."):
                payload = _run_start_date_robustness(run_map)
            st.session_state[START_DATE_ROBUSTNESS_STATE_KEY] = dict(payload)
            st.session_state[START_DATE_ROBUSTNESS_SCOPE_KEY] = str(payload.get("scope", scope) or scope)
            st.rerun()


def _render_start_date_robustness_details(payload: dict) -> None:
    """Render saved start-date robustness evidence inside the single details expander."""
    if not payload:
        st.caption("Start-date robustness has not been run yet. Use the button above to test alternative historical start windows.")
        return

    summary = _coerce_mapping(payload.get("summary", {}))
    downside_status = str(summary.get("downside_status", "") or "")
    return_sensitivity_status = str(summary.get("return_sensitivity_status", "") or "")
    if downside_status or return_sensitivity_status:
        st.caption(
            f"Robustness split: downside robustness={downside_status or '—'} · "
            f"return sensitivity={return_sensitivity_status or '—'}."
        )

    elapsed = _safe_float(payload.get("elapsed_sec", 0.0), 0.0)
    if summary.get("completed_windows"):
        st.caption(
            f"CAGR range: {_fmt_pct(summary.get('cagr_min'))} → {_fmt_pct(summary.get('cagr_max'))} · "
            f"Sharpe range: {_safe_float(summary.get('sharpe_min'), 0.0):.2f} → {_safe_float(summary.get('sharpe_max'), 0.0):.2f} · "
            f"MaxDD range: -{100.0 * _safe_float(summary.get('maxdd_min'), 0.0):.2f}% → -{100.0 * _safe_float(summary.get('maxdd_max'), 0.0):.2f}% · "
            f"runtime={elapsed:.2f}s."
        )

    rows = list(payload.get("rows", []) or [])
    if rows:
        display_rows = []
        for row in rows:
            row_map = _coerce_mapping(row)
            display_rows.append(
                {
                    "Start date": row_map.get("Start date", "—"),
                    "Evaluated period": row_map.get("Evaluated period", "—"),
                    "OOS months": row_map.get("OOS months", 0),
                    "CAGR": row_map.get("CAGR", "—"),
                    "Vol": row_map.get("Vol", "—"),
                    "MaxDD": row_map.get("MaxDD", "—"),
                    "Sharpe": row_map.get("Sharpe", "—"),
                    "Reliability read": row_map.get("Reliability read", ""),
                    "Status": row_map.get("Status", ""),
                    "Seconds": row_map.get("Seconds", 0.0),
                }
            )
        st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

    timing_rows = _robustness_timing_rows(payload)
    if timing_rows:
        total_seconds = _safe_float(payload.get("elapsed_sec", 0.0), 0.0)
        checked_count = int(len(timing_rows))
        executed_count = int(sum(1 for row in timing_rows if str(row.get("status", "")).lower() == "executed rerun"))
        st.markdown("**Start-date robustness timing**")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Robustness test", f"{total_seconds:.2f}s" if total_seconds > 0 else "—")
        with c2:
            st.metric("Start dates checked", int(checked_count))
        with c3:
            st.metric("Executed reruns", int(executed_count))
        st.dataframe(pd.DataFrame(timing_rows), use_container_width=True, hide_index=True)

    st.caption(
        "The selected configuration was not only evaluated on one fixed historical start date. "
        "Alternative historical windows are used as a sensitivity check. Shorter windows should inform interpretation, not automatically invalidate the result."
    )


def render_result_reliability_assessment(run_map: dict, *, benchmark_payload: dict | None = None, inline_details: bool = False) -> None:
    """Render a compact reliability assessment for the current Strategy Engine result."""
    run_map = _coerce_mapping(run_map)
    benchmark_payload = _coerce_mapping(benchmark_payload or {})

    components, overall, validation_df, validation_meta = _reliability_component_rows(run_map, benchmark_payload)
    robustness_payload, robustness_summary, robustness_status, robustness_message = _start_date_robustness_visible_summary(run_map)

    st.markdown("### Result reliability assessment")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Overall reliability", overall)
    with c2:
        st.metric("Start-date robustness", robustness_status)

    st.info(
        f"**Overall confidence: {overall}.** The Strategy Engine result is based on a historical market-data snapshot and a simulated walk-forward portfolio. "
        "It is useful for comparing configurations inside the app, but it is not a forecast, guarantee, or live fund track record."
    )

    if robustness_payload:
        _render_status_callout(robustness_status, robustness_message)
        if robustness_summary:
            completed = _safe_int(robustness_summary.get("completed_windows", 0), 0)
            if completed > 0:
                st.caption(
                    f"Tested across {completed} valid alternative start windows. "
                    f"CAGR range {_fmt_pct(robustness_summary.get('cagr_min'))} → {_fmt_pct(robustness_summary.get('cagr_max'))}; "
                    f"Sharpe range {_safe_float(robustness_summary.get('sharpe_min'), 0.0):.2f} → {_safe_float(robustness_summary.get('sharpe_max'), 0.0):.2f}."
                )
    else:
        st.caption("Start-date robustness has not been run yet. Run it if you want to test whether this result is sensitive to the chosen historical start window.")

    _render_start_date_robustness_actions(run_map)

    details_ctx = st.container(border=True) if inline_details else st.expander("Detailed evidence and validation", expanded=False)
    with details_ctx:
        if inline_details:
            st.markdown("#### Detailed evidence and validation")
            st.caption(
                "Detailed reliability evidence is shown here instead of inside a nested expander, "
                "because Streamlit does not allow expanders inside expanders."
            )

        st.markdown("**Reliability components**")
        st.dataframe(pd.DataFrame(components), use_container_width=True, hide_index=True)

        st.markdown("**What is real here?**")
        st.markdown(
            "- **Benchmark assets** are real historical market assets from the selected market-data panel.\n"
            "- **The Strategy Engine result** is a walk-forward backtest created from those real asset returns.\n"
            "- **The Long-Term Scenario** is a future uncertainty simulation based on the historical strategy return series."
        )

        st.markdown("**Benchmark calculation validation**")
        st.write(
            "This check compares benchmark metrics shown by the app against an independent recomputation from the same selected market-data panel and the same evaluated period. "
            "It validates the metric calculation pipeline, not future predictive accuracy and not Yahoo data correctness."
        )
        if isinstance(validation_df, pd.DataFrame) and not validation_df.empty:
            st.dataframe(validation_df, use_container_width=True, hide_index=True)
            st.caption(
                f"Validation window: {validation_meta.get('window', '—')} · assets checked={validation_meta.get('n_assets', len(validation_df))} · "
                f"months={validation_meta.get('months', '—')} · status={validation_meta.get('status', 'Unavailable')}."
            )
        else:
            st.info("Benchmark calculation validation is unavailable for this run.")

        st.markdown("**Start-date robustness details**")
        st.write(
            "This check reruns the same Strategy Engine configuration using different historical start dates from the existing selected market-data panel. "
            "It checks whether the result depends too heavily on one specific historical window."
        )
        _render_start_date_robustness_details(robustness_payload)

        st.markdown("**Method notes and limits**")
        st.info(
            "These figures come from a historical walk-forward backtest using the selected asset panel. "
            "They are useful for comparing configurations inside the app, but they are not forecasts or guarantees. "
            "Results depend on the date range, asset universe, data quality, and engine assumptions."
        )
