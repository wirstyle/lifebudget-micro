from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd
import streamlit as st

from src.investment import config_to_dict
from ui.step5.simple_mode import render_simple_mode, resolve_simple_mode_state
from ui.step5.run_panel import clear_retired_step5_state, render_run_panel
from ui.step5.post_run import render_post_run
from ui.step5.governance import resolve_governance_status




SUGGESTION_DEPTH_MODE_KEY = "step5_suggestion_testing_depth_mode_v1"
SUGGESTION_DEPTH_COUNT_KEYS: dict[str, str] = {
    "preset": "step5_suggestion_count_preset_v1",
    "auto_opt": "step5_suggestion_count_auto_opt_v1",
    "universe": "step5_suggestion_count_universe_v1",
    "size": "step5_suggestion_count_size_v1",
}
SUGGESTION_DEPTH_PRESETS: dict[str, dict[str, int]] = {
    "Fast": {"preset": 1, "auto_opt": 1, "universe": 2, "size": 1},
    "Balanced": {"preset": 1, "auto_opt": 2, "universe": 4, "size": 1},
    "Thorough": {"preset": 2, "auto_opt": 3, "universe": 6, "size": 2},
}
SUGGESTION_DEPTH_DEFAULT = "Balanced"
SUGGESTION_SECONDS_PER_TEST: dict[str, float] = {
    "preset": 30.0,
    "auto_opt": 30.0,
    "universe": 22.0,
    "size": 30.0,
}
SHOW_ADVANCED_SUGGESTION_DEPTH_KEY = "step5_show_advanced_suggestion_testing_controls_v1"


def _clamp_suggestion_count(value: Any, *, default: int = 1, low: int = 1, high: int = 8) -> int:
    try:
        raw = int(value)
    except Exception:
        raw = int(default)
    return int(max(low, min(high, raw)))


def _format_runtime_estimate(seconds: float) -> str:
    try:
        seconds = float(seconds)
    except Exception:
        seconds = 30.0
    if seconds <= 40:
        return "~30s"
    if seconds <= 70:
        return "~1 min"
    if seconds <= 105:
        return "~90s"
    if seconds <= 150:
        return "~2 min"
    minutes = seconds / 60.0
    if minutes < 10:
        rounded = round(minutes * 2.0) / 2.0
        return f"~{rounded:g} min"
    return f"~{round(minutes):.0f} min"


def _ensure_suggestion_depth_defaults() -> None:
    defaults = SUGGESTION_DEPTH_PRESETS[SUGGESTION_DEPTH_DEFAULT]
    if SUGGESTION_DEPTH_MODE_KEY not in st.session_state:
        st.session_state[SUGGESTION_DEPTH_MODE_KEY] = SUGGESTION_DEPTH_DEFAULT
    for phase, key in SUGGESTION_DEPTH_COUNT_KEYS.items():
        if key not in st.session_state:
            st.session_state[key] = int(defaults.get(phase, 1))


def _current_suggestion_counts() -> dict[str, int]:
    defaults = SUGGESTION_DEPTH_PRESETS[SUGGESTION_DEPTH_DEFAULT]
    out: dict[str, int] = {}
    for phase, key in SUGGESTION_DEPTH_COUNT_KEYS.items():
        out[phase] = _clamp_suggestion_count(
            st.session_state.get(key, defaults.get(phase, 1)),
            default=defaults.get(phase, 1),
        )
    return out


def _apply_suggestion_depth_preset(mode: str) -> dict[str, int]:
    if mode not in SUGGESTION_DEPTH_PRESETS:
        return _current_suggestion_counts()
    counts = dict(SUGGESTION_DEPTH_PRESETS[mode])
    for phase, value in counts.items():
        st.session_state[SUGGESTION_DEPTH_COUNT_KEYS[phase]] = int(value)
    return counts


def _render_pre_run_suggestion_depth_controls() -> None:
    """Let users opt into deeper suggestion searches before the portfolio run."""
    _ensure_suggestion_depth_defaults()
    show_controls = bool(
        st.checkbox(
            "Show advanced suggestion testing controls",
            key=SHOW_ADVANCED_SUGGESTION_DEPTH_KEY,
            help=(
                "Optional runtime control for the improvement suggestions that may be tested "
                "after the portfolio run. Balanced depth is used by default."
            ),
        )
    )

    if not show_controls:
        st.caption(
            "After the portfolio run, optional improvement suggestions use Balanced testing depth by default. "
            "Open this only if you want to trade more runtime for more candidate coverage."
        )
        return

    with st.container(border=True):
        st.caption("Higher depth tests more alternatives, but each extra candidate adds real engine runtime after the portfolio run.")
        mode_options = ["Fast", "Balanced", "Thorough", "Custom"]
        current_mode = str(st.session_state.get(SUGGESTION_DEPTH_MODE_KEY, SUGGESTION_DEPTH_DEFAULT) or SUGGESTION_DEPTH_DEFAULT)
        if current_mode not in mode_options:
            current_mode = SUGGESTION_DEPTH_DEFAULT
            st.session_state[SUGGESTION_DEPTH_MODE_KEY] = current_mode

        mode = st.selectbox(
            "Testing depth",
            options=mode_options,
            index=mode_options.index(current_mode),
            key=SUGGESTION_DEPTH_MODE_KEY,
            help="Controls how many rerun-tested alternatives are evaluated in each optional suggestion phase.",
        )

        if mode in SUGGESTION_DEPTH_PRESETS:
            counts = _apply_suggestion_depth_preset(mode)
            st.caption(
                f"{mode}: {counts['preset']} preset · {counts['auto_opt']} tuning · "
                f"{counts['universe']} universe-mix · {counts['size']} size test(s)."
            )
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.number_input(
                    "Preset",
                    min_value=1,
                    max_value=4,
                    step=1,
                    key=SUGGESTION_DEPTH_COUNT_KEYS["preset"],
                    help="Strategy preset alternatives to rerun-test.",
                )
            with c2:
                st.number_input(
                    "Engine tuning",
                    min_value=1,
                    max_value=6,
                    step=1,
                    key=SUGGESTION_DEPTH_COUNT_KEYS["auto_opt"],
                    help="Technical tuning alternatives to rerun-test.",
                )
            with c3:
                st.number_input(
                    "Universe mix",
                    min_value=1,
                    max_value=8,
                    step=1,
                    key=SUGGESTION_DEPTH_COUNT_KEYS["universe"],
                    help="Same-size universe compositions to rerun-test.",
                )
            with c4:
                st.number_input(
                    "Universe size",
                    min_value=1,
                    max_value=6,
                    step=1,
                    key=SUGGESTION_DEPTH_COUNT_KEYS["size"],
                    help="Universe-size alternatives to rerun-test.",
                )
            counts = _current_suggestion_counts()

        estimates = {
            phase: counts[phase] * SUGGESTION_SECONDS_PER_TEST.get(phase, 30.0)
            for phase in SUGGESTION_DEPTH_COUNT_KEYS
        }
        total_estimate = sum(estimates.values())
        st.caption(
            "Estimated extra runtime if all phases run: "
            f"{_format_runtime_estimate(total_estimate)} "
            f"({counts['preset']}/{counts['auto_opt']}/{counts['universe']}/{counts['size']} tests)."
        )
        st.caption(
            "This is approximate and depends on the deployed environment, cache state, and panel size. "
            "Timings are recorded later in Advanced run diagnostics and timings."
        )

def _resolve_overlay_label(simple_cfg: dict, advanced_cfg: dict) -> str:
    overlay_intensity = float(simple_cfg.get("overlay_intensity", 0.5) or 0.5)
    simplicity = float(simple_cfg.get("simplicity_vs_sophistication", 0.5) or 0.5)
    if overlay_intensity < 0.35:
        return "light"
    if overlay_intensity > 0.70 and simplicity < 0.45:
        return "adaptive"
    return "historical"


def _resolve_universe_bucket() -> str:
    try:
        universe_size = int(st.session_state.get("universe_size", 25) or 25)
    except Exception:
        universe_size = 25
    if universe_size <= 12:
        return "small"
    if universe_size <= 25:
        return "medium"
    if universe_size <= 75:
        return "large"
    return "research"


def _resolve_top_k(simple_cfg: dict) -> int:
    concentration = float(simple_cfg.get("diversification_vs_concentration", 0.55) or 0.55)
    try:
        universe_size = int(st.session_state.get("universe_size", 25) or 25)
    except Exception:
        universe_size = 25
    if concentration >= 0.75:
        return max(4, min(8, universe_size))
    if concentration >= 0.60:
        return max(6, min(10, universe_size))
    if concentration >= 0.45:
        return max(8, min(12, universe_size))
    return max(10, min(16, universe_size))


def _build_resolved_cfg(simple_cfg: dict, advanced_cfg: dict) -> dict:
    resolved = dict(simple_cfg or {})
    resolved.update(dict(advanced_cfg or {}))
    resolved["top_k"] = _resolve_top_k(simple_cfg)
    resolved["overlay_label"] = _resolve_overlay_label(simple_cfg, advanced_cfg)
    resolved["universe_bucket"] = _resolve_universe_bucket()
    resolved["signal_mode"] = "mu_sigma" if float(simple_cfg.get("confidence_in_signal", 0.5) or 0.5) >= 0.50 else "broad_beta"
    resolved["selection_policy"] = "fixed_composite_score"
    return resolved



def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _stable_panel_fingerprint(panel_df: pd.DataFrame) -> str:
    if panel_df is None or not isinstance(panel_df, pd.DataFrame) or panel_df.empty:
        return "empty"
    payload = {
        "rows": int(len(panel_df)),
        "cols": list(map(str, panel_df.columns)),
        "assets": sorted(panel_df["asset"].dropna().astype(str).unique().tolist()) if "asset" in panel_df.columns else [],
        "min_date": str(panel_df["date"].min()) if "date" in panel_df.columns else "",
        "max_date": str(panel_df["date"].max()) if "date" in panel_df.columns else "",
    }
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(encoded.encode("utf-8")).hexdigest()[:16]


def _resolve_run_signature_for_ready_card(cfg_final: dict, asset_panel_df: Any) -> str:
    payload = {
        "cfg_final": dict(cfg_final or {}),
        "universe_size": st.session_state.get("universe_size"),
        "universe_strategy": st.session_state.get("universe_strategy"),
        "selected_assets": list(st.session_state.get("selected_assets", []) or []),
        "panel_fp": _stable_panel_fingerprint(asset_panel_df) if isinstance(asset_panel_df, pd.DataFrame) else "empty",
    }
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(encoded.encode("utf-8")).hexdigest()[:12]


def _resolve_config_fingerprint_for_ready_card(cfg_final: dict) -> str:
    encoded = json.dumps(dict(cfg_final or {}), sort_keys=True, default=str)
    return hashlib.md5(encoded.encode("utf-8")).hexdigest()[:12]


def _current_result_is_fresh_for_ready_card(cfg_final: dict, asset_panel_df: Any) -> tuple[bool, str]:
    current_signature = _resolve_run_signature_for_ready_card(cfg_final, asset_panel_df)
    current_config_fp = _resolve_config_fingerprint_for_ready_card(cfg_final)
    last_signature = str(st.session_state.get("step5_last_run_signature", "") or "")
    last_config_fp = str(st.session_state.get("step5_last_config_fingerprint", "") or "")
    stored_result = st.session_state.get("step5_last_run_result")
    fresh = bool(
        stored_result is not None
        and current_signature
        and current_config_fp
        and last_signature == current_signature
        and last_config_fp == current_config_fp
    )
    return fresh, current_signature


def _latest_run_performance_summary() -> dict:
    run_map = _coerce_mapping(st.session_state.get("step5_last_run_result", {}))
    perf = _coerce_mapping(run_map.get("performance_summary", {}))
    if not perf:
        return {}
    return {
        "cagr": _safe_float(perf.get("cagr", 0.0), 0.0),
        "sharpe": _safe_float(perf.get("sharpe", 0.0), 0.0),
        "annual_volatility": _safe_float(perf.get("annual_volatility", perf.get("volatility", 0.0)), 0.0),
        "max_drawdown": abs(_safe_float(perf.get("max_drawdown", 0.0), 0.0)),
    }


def _technical_pct(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.2f}%"
    except Exception:
        return "—"


def _technical_current_result_text(perf: dict) -> str:
    if not perf:
        return "No completed run is available yet."
    return (
        f"Current run: CAGR {_technical_pct(perf.get('cagr', 0.0))}, "
        f"Vol {_technical_pct(perf.get('annual_volatility', 0.0))}, "
        f"MaxDD -{100.0 * abs(_safe_float(perf.get('max_drawdown', 0.0), 0.0)):.2f}%, "
        f"Sharpe {_safe_float(perf.get('sharpe', 0.0), 0.0):.2f}."
    )


def _technical_improvement_focus(current_philosophy: str) -> str:
    """Return a short, dynamic explanation of what improvement should mean now."""
    perf = _latest_run_performance_summary()
    if not perf:
        return (
            "Before a run, these controls are available for deliberate technical experiments. "
            "After a run, this guide adapts to the latest CAGR, volatility, MaxDD, Sharpe, and philosophy."
        )

    profile = str(current_philosophy or "Balanced").strip().lower()
    cagr = _safe_float(perf.get("cagr", 0.0), 0.0)
    sharpe = _safe_float(perf.get("sharpe", 0.0), 0.0)
    vol = _safe_float(perf.get("annual_volatility", 0.0), 0.0)
    maxdd = abs(_safe_float(perf.get("max_drawdown", 0.0), 0.0))

    if profile == "growth":
        if cagr < 0.06:
            return (
                "For this Growth setup, the priority is better return capture. The test should look for higher "
                "CAGR or Sharpe, but reject changes that let MaxDD or volatility expand too far."
            )
        if maxdd >= 0.25 or vol >= 0.20:
            return (
                "For this Growth setup, upside is present but risk is becoming the constraint. The useful test is "
                "whether Sharpe can improve while keeping most of the CAGR case intact."
            )
        if sharpe < 0.70:
            return (
                "For this Growth setup, the main opportunity is efficiency: improve Sharpe without making the "
                "engine so defensive that it removes the growth case."
            )
        return (
            "For this Growth setup, the result already has a reasonable growth profile. Only accept technical "
            "changes if they improve Sharpe or drawdown without materially reducing CAGR."
        )

    if profile in {"defensive", "conservative"}:
        if maxdd >= 0.12 or vol >= 0.12:
            return (
                "For this Defensive setup, the priority is lower volatility and MaxDD. A small CAGR sacrifice may "
                "be acceptable only if the path becomes meaningfully smoother and Sharpe remains reasonable."
            )
        if sharpe < 0.60:
            return (
                "For this Defensive setup, risk is the priority, but the engine should still earn enough return per "
                "unit of risk. The test should look for better Sharpe without adding much drawdown."
            )
        if cagr < 0.04:
            return (
                "For this Defensive setup, the path may be controlled but too conservative. The test should check "
                "whether CAGR can improve without giving back the defensive risk profile."
            )
        return (
            "For this Defensive setup, the result already looks controlled. Only accept changes that preserve low "
            "volatility and MaxDD while improving Sharpe or modestly improving CAGR."
        )

    if maxdd >= 0.18 or vol >= 0.15:
        return (
            "For this Balanced setup, the main opportunity is improving the drawdown/Sharpe trade-off rather than "
            "simply chasing higher CAGR."
        )
    if sharpe < 0.65:
        return (
            "For this Balanced setup, the main opportunity is Sharpe improvement: reduce noise, volatility, or "
            "drawdown while keeping CAGR close to the current result."
        )
    if cagr < 0.06:
        return (
            "For this Balanced setup, the result may be too cautious or diluted. The test should check whether CAGR "
            "can improve without materially worsening volatility or MaxDD."
        )
    return (
        "For this Balanced setup, the result already looks broadly acceptable. Only accept technical changes if "
        "the full trade-off improves, not just one isolated metric."
    )


def _technical_first_test_hint(current_philosophy: str) -> str:
    """Summarise the first improvement hypothesis using latest metrics and philosophy.

    Keep this deliberately high-level. The control-level actions are explained
    separately in the metric bullets below, so this text should state the test
    hypothesis rather than repeating parameter names.
    """
    perf = _latest_run_performance_summary()
    if not perf:
        return (
            "test whether a small, reversible engine change improves the full trade-off across CAGR, Vol, "
            "MaxDD, and Sharpe."
        )

    profile = str(current_philosophy or "Balanced").strip().lower()
    cagr = _safe_float(perf.get("cagr", 0.0), 0.0)
    sharpe = _safe_float(perf.get("sharpe", 0.0), 0.0)
    vol = _safe_float(perf.get("annual_volatility", 0.0), 0.0)
    maxdd = abs(_safe_float(perf.get("max_drawdown", 0.0), 0.0))

    if profile == "growth":
        if cagr < 0.06:
            return (
                "test whether the setup is missing too much upside, while rejecting any change that buys CAGR "
                "through a disproportionate increase in Vol or MaxDD."
            )
        if maxdd >= 0.25 or vol >= 0.20:
            return (
                "test whether the growth case can be made less fragile, reducing risk while preserving most of "
                "the existing CAGR case."
            )
        return (
            "test whether the growth setup can improve Sharpe without simply adding more raw risk."
        )

    if profile in {"defensive", "conservative"}:
        if maxdd >= 0.12 or vol >= 0.12:
            return (
                "test whether the path can become smoother, reducing Vol and MaxDD while keeping Sharpe acceptable."
            )
        if cagr < 0.04:
            return (
                "test whether the setup is too conservative and can recover some CAGR without giving back the "
                "defensive risk profile."
            )
        return (
            "test whether the defensive setup can improve Sharpe or modestly improve CAGR without weakening its "
            "risk-control role."
        )

    if maxdd >= 0.18 or vol >= 0.15:
        return (
            "test whether a broader and smoother version of this setup can reduce Vol and MaxDD while keeping "
            "Sharpe and CAGR close to the current result."
        )
    if sharpe < 0.65:
        return (
            "test whether noisy allocation behaviour is holding Sharpe down, without sacrificing too much CAGR."
        )
    if cagr < 0.06:
        return (
            "test whether return capture is too muted, while rejecting changes that improve CAGR only by taking "
            "too much extra drawdown."
        )
    return (
        "test only small technical variations and accept them only if the rerun improves the full trade-off."
    )


def _technical_metric_bullets(current_philosophy: str) -> list[str]:
    profile = str(current_philosophy or "Balanced").strip().lower()
    if profile == "growth":
        return [
            "**CAGR:** test by switching `signal_mode` to a more return-sensitive option, decreasing `lookback_mu`, decreasing `temperature`, or enabling `feature_mu_enabled` so stronger return signals can influence selection.",
            "**Volatility:** test by increasing `top_k`, `lookback_sigma`, and `weight_shrink` to broaden selection and smooth risk estimates.",
            "**MaxDD:** test by increasing `top_k`, `inertia`, `weight_shrink`, and `lookback_mu` to reduce concentration and slow unstable reallocations without fully turning defensive.",
            "**Sharpe:** test by increasing `weight_shrink`, `lookback_sigma`, `temperature`, and `inertia` so the growth setup becomes less noisy without removing too much upside.",
        ]
    if profile in {"defensive", "conservative"}:
        return [
            "**CAGR:** test by switching `signal_mode` to a slightly more return-sensitive option, decreasing `lookback_mu`, or enabling `feature_mu_enabled`, but only if the defensive risk profile survives.",
            "**Volatility:** test by increasing `lookback_sigma`, `top_k`, `weight_shrink`, and `inertia` to smooth estimates, broaden allocation, and reduce noisy reallocations.",
            "**MaxDD:** test by increasing `top_k`, `inertia`, `lookback_sigma`, and `weight_shrink` to lower concentration and slow down unstable portfolio changes.",
            "**Sharpe:** test by increasing `lookback_mu` and `weight_shrink`, or switching `signal_mode` to a cleaner/robuster option, so the smoother path still earns enough return per unit of risk.",
        ]
    return [
        "**CAGR:** test by switching `signal_mode` to a more return-sensitive option, decreasing `lookback_mu`, decreasing `temperature`, or enabling `feature_mu_enabled` so stronger return signals can add upside.",
        "**Volatility:** test by increasing `top_k`, `lookback_sigma`, and `weight_shrink` to broaden selection and smooth risk estimates.",
        "**MaxDD:** test by increasing `top_k`, `inertia`, `lookback_mu`, and `weight_shrink` to reduce concentration and slow unstable reallocations.",
        "**Sharpe:** test by increasing `top_k`, `weight_shrink`, `lookback_sigma`, and `inertia` so volatility or drawdown may fall while CAGR stays close to the current result.",
    ]


def _render_technical_improvement_guide(current_philosophy: str) -> None:
    perf = _latest_run_performance_summary()
    with st.container(border=True):
        st.markdown("**Technical improvement guide**")
        if perf:
            st.caption(_technical_current_result_text(perf))
        st.markdown(f"**Current improvement focus:** {_technical_improvement_focus(current_philosophy)}")
        st.markdown(f"**Initial test hypothesis:** {_technical_first_test_hint(current_philosophy)}")
        st.markdown("**What the controls mainly test:**")
        for bullet in _technical_metric_bullets(current_philosophy):
            st.markdown(f"- {bullet}")
        st.caption(
            "These are tests, not guarantees: the engine only recommends a change if the rerun-tested trade-off "
            "passes the acceptance gate."
        )

def _render_basic_engine_controls(cfg_final: dict) -> dict:
    universe_size = _safe_int(st.session_state.get("universe_size", 25), 25)
    current_top_k = _safe_int(cfg_final.get("top_k", 12), 12)
    current_top_k = max(1, min(current_top_k, max(1, universe_size)))

    current_lookback_mu = _safe_int(cfg_final.get("lookback_mu", 12), 12)
    current_lookback_sigma = _safe_int(cfg_final.get("lookback_sigma", 12), 12)
    current_signal_mode = str(cfg_final.get("signal_mode", "mu_sigma") or "mu_sigma")
    current_temperature = float(cfg_final.get("temperature", 1.0) or 1.0)
    current_weight_shrink = float(cfg_final.get("weight_shrink", 0.05) or 0.05)
    current_inertia = float(cfg_final.get("inertia", 0.0) or 0.0)
    current_feature_mu_enabled = bool(cfg_final.get("feature_mu_enabled", False))
    current_feature_mu_blend = float(cfg_final.get("feature_mu_blend", 0.25) or 0.25)

    c1, c2 = st.columns(2)
    with c1:
        top_k = int(
            st.number_input(
                "top_k",
                min_value=1,
                max_value=max(1, universe_size),
                step=1,
                value=current_top_k,
                key="step5_basic_top_k",
                help="Maximum number of active assets selected by the micro pipeline.",
            )
        )
        lookback_mu = int(
            st.number_input(
                "lookback_mu",
                min_value=3,
                max_value=60,
                step=1,
                value=current_lookback_mu,
                key="step5_basic_lookback_mu",
                help="Lookback window used for the expected-return estimate.",
            )
        )
    with c2:
        lookback_sigma = int(
            st.number_input(
                "lookback_sigma",
                min_value=3,
                max_value=60,
                step=1,
                value=current_lookback_sigma,
                key="step5_basic_lookback_sigma",
                help="Lookback window used for the volatility estimate.",
            )
        )
        signal_mode_options = [
            "mu_sigma",
            "huber_mu",
            "lambdarank_like",
            "directional_classifier",
            "top_k_classifier",
        ]
        safe_signal_mode = current_signal_mode if current_signal_mode in set(signal_mode_options) else "mu_sigma"
        signal_mode = st.selectbox(
            "signal_mode",
            options=signal_mode_options,
            index=signal_mode_options.index(safe_signal_mode),
            key="step5_basic_signal_mode",
            help="Minimal signal contract exposed in Fase 6.0, now including more distinct signal-model branches.",
        )

    c3, c4 = st.columns(2)
    with c3:
        temperature = float(
            st.number_input(
                "temperature",
                min_value=0.01,
                max_value=5.0,
                step=0.01,
                value=current_temperature,
                key="step5_basic_temperature",
                help="Softmax temperature controlling concentration of weights.",
            )
        )
        weight_shrink = float(
            st.number_input(
                "weight_shrink",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                value=current_weight_shrink,
                key="step5_basic_weight_shrink",
                help="Shrink weights toward equal-weight.",
            )
        )
        feature_mu_blend = float(
            st.number_input(
                "feature_mu_blend",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                value=current_feature_mu_blend,
                key="step5_basic_feature_mu_blend",
                help="Controls the intensity of the feature-conditioned expected-return adjustment.",
            )
        )
    with c4:
        inertia = float(
            st.number_input(
                "inertia",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                value=current_inertia,
                key="step5_basic_inertia",
                help="Portfolio inertia / turnover smoothing.",
            )
        )
        feature_mu_enabled = bool(
            st.checkbox(
                "feature_mu_enabled",
                value=current_feature_mu_enabled,
                key="step5_basic_feature_mu_enabled",
                help="Enable feature-conditioned expected-return adjustment.",
            )
        )

    feature_mu_blend_display = feature_mu_blend if feature_mu_enabled else "inactive"
    st.caption(
        f"Effective basic controls → top_k={top_k} · lookback_mu={lookback_mu} · "
        f"lookback_sigma={lookback_sigma} · signal_mode={signal_mode} · "
        f"temperature={temperature} · weight_shrink={weight_shrink} · inertia={inertia} · "
        f"feature_mu_enabled={feature_mu_enabled} · feature_mu_blend={feature_mu_blend_display}"
    )

    _render_technical_improvement_guide(str(st.session_state.get("investment_philosophy", "Balanced") or "Balanced"))

    return {
        "top_k": int(top_k),
        "lookback_mu": int(lookback_mu),
        "lookback_sigma": int(lookback_sigma),
        "signal_mode": str(signal_mode),
        "temperature": float(temperature),
        "weight_shrink": float(weight_shrink),
        "inertia": float(inertia),
        "feature_mu_enabled": bool(feature_mu_enabled),
        "feature_mu_blend": float(feature_mu_blend),
    }


def _render_technical_engine_overrides(cfg_final: dict) -> dict:
    """Render optional technical overrides inside the fine-tune settings block."""
    show_overrides = bool(
        st.checkbox(
            "Show advanced technical engine overrides",
            key="step5_show_technical_engine_overrides",
            help="Optional low-level engine parameters for diagnostics or controlled experimentation.",
        )
    )

    if not show_overrides:
        st.caption("Presets and posture sliders control the engine unless technical overrides are edited.")
        return {}

    st.caption("Optional technical overrides. Leave unchanged unless deliberately testing engine behaviour. Not sure what a parameter means? Hover over the ? icon before changing it.")
    with st.container(border=True):
        return _render_basic_engine_controls(cfg_final)


def _coerce_mapping(value: Any) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        return dict(value)
    except Exception:
        return {}


def _build_step5_input_signature(cfg_final: dict, asset_panel_df: Any) -> str:
    payload = {
        "cfg_final": _coerce_mapping(cfg_final),
        "universe_size": st.session_state.get("universe_size"),
        "universe_strategy": st.session_state.get("universe_strategy"),
        "selected_assets": list(st.session_state.get("selected_assets", []) or []),
        "asset_panel_rows": int(len(asset_panel_df)) if isinstance(asset_panel_df, pd.DataFrame) else 0,
        "asset_panel_assets": int(asset_panel_df["asset"].nunique()) if isinstance(asset_panel_df, pd.DataFrame) and "asset" in asset_panel_df.columns else 0,
        "asset_panel_min_date": str(asset_panel_df["date"].min()) if isinstance(asset_panel_df, pd.DataFrame) and "date" in asset_panel_df.columns and not asset_panel_df.empty else "",
        "asset_panel_max_date": str(asset_panel_df["date"].max()) if isinstance(asset_panel_df, pd.DataFrame) and "date" in asset_panel_df.columns and not asset_panel_df.empty else "",
    }
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(encoded.encode("utf-8")).hexdigest()[:12]


def _resolve_cfg_final(simple_cfg: dict, advanced_cfg: dict) -> tuple[dict, dict]:
    resolved_cfg = _build_resolved_cfg(simple_cfg, advanced_cfg)
    try:
        gov = resolve_governance_status(simple_cfg, advanced_cfg, resolved_cfg)
    except Exception:
        gov = {"state": "coherent", "cfg_final": dict(resolved_cfg)}
    cfg_final = dict(gov.get("cfg_final", {}) or {})
    if not cfg_final:
        governed_result = gov.get("governed_result")
        final_obj = getattr(governed_result, "final_cfg", None)
        if final_obj is not None:
            try:
                cfg_final = dict(config_to_dict(final_obj))
            except Exception:
                cfg_final = dict(getattr(final_obj, "__dict__", {}) or {})
    if not cfg_final:
        cfg_final = dict(resolved_cfg or {})
    return cfg_final, dict(gov or {})


def _shorten(value: Any, max_chars: int = 24) -> str:
    text = str(value or "—")
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


def _render_compact_value(label: str, value: Any) -> None:
    st.caption(str(label))
    st.markdown(f"**{str(value or '—')}**")


def _render_engine_transparency_body() -> None:
    """Shared copy explaining the strategy engine without adding another top-level banner."""
    st.write(
        "The Risk Profile and Asset Universe screen prepares a market-data panel for individual assets such as "
        "SPY, QQQ or GLD. Your strategy does not exist as a single ticker in Yahoo Finance. The strategy engine "
        "creates it by selecting and weighting assets over time. Once that monthly portfolio return series exists, "
        "CAGR, volatility, Sharpe and max drawdown can be calculated from it."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**What the engine can improve**")
        st.markdown(
            "- diversification across the selected universe\n"
            "- risk and drawdown control\n"
            "- dynamic asset selection and weighting\n"
            "- alignment with the chosen risk profile"
        )
    with c2:
        st.markdown("**What the engine can worsen**")
        st.markdown(
            "- it may lag simple assets in strong bull markets\n"
            "- risk controls can reduce upside\n"
            "- signals can overfit if assumptions are weak\n"
            "- extra complexity can add turnover and parameter sensitivity"
        )

    st.info(
        "You do not pass the portfolio through the engine to calculate CAGR. "
        "You pass it through the engine to create the portfolio. The metrics come afterwards."
    )


def _render_engine_transparency_intro() -> None:
    """Keep the detailed engine explanation available without making the page top-heavy."""
    with st.expander("Why the engine is needed", expanded=False):
        _render_engine_transparency_body()


def _render_strategy_engine_about_expander() -> None:
    """Collapsed post-run purpose note. Keeps the result surface clean after execution."""
    with st.expander("About the Strategy Engine", expanded=False):
        st.write(
            "**Purpose:** turn the selected risk profile, asset universe and market-data panel into a tested "
            "strategy return series. This is the execution stage, not a new data-preparation step."
        )
        _render_engine_transparency_body()

def _render_workspace_summary(cfg_final: dict, asset_panel_df: Any, current_philosophy: str, simple_cfg: dict) -> None:
    panel_rows = int(len(asset_panel_df)) if isinstance(asset_panel_df, pd.DataFrame) else 0
    panel_assets = int(asset_panel_df["asset"].nunique()) if isinstance(asset_panel_df, pd.DataFrame) and "asset" in asset_panel_df.columns else 0
    universe_size = int(st.session_state.get("universe_size", 25) or 25)
    panel_ready = bool(st.session_state.get("asset_panel_ready", False)) and panel_rows > 0

    with st.container(border=True):
        st.markdown("### Current setup")
        st.caption("The strategy engine will use the selected universe and market-data panel with the current strategy preset.")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            _render_compact_value("Philosophy", current_philosophy)
        with c2:
            _render_compact_value("Universe", f"{universe_size} assets")
        with c3:
            _render_compact_value("Template", _shorten(simple_cfg.get("template", "—")))
        with c4:
            _render_compact_value("Style", simple_cfg.get("preset", "—"))

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            _render_compact_value("Panel rows", f"{panel_rows:,}")
        with c6:
            _render_compact_value("Panel assets", panel_assets)
        with c7:
            _render_compact_value("Top k", cfg_final.get("top_k", "—"))
        with c8:
            _render_compact_value("Signal", cfg_final.get("signal_mode", "—"))

        if panel_ready:
            st.success("Ready to run: the selected market-data panel is loaded and the current setup is coherent enough for execution.")
        else:
            st.warning("Market data is missing or not ready. Return to Risk Profile & Asset Universe before running the engine.")
        with st.expander("Show technical setup details", expanded=False):
            st.caption(
                f"lookback_mu={cfg_final.get('lookback_mu', '—')} · "
                f"lookback_sigma={cfg_final.get('lookback_sigma', '—')} · "
                f"temperature={cfg_final.get('temperature', '—')} · "
                f"weight_shrink={cfg_final.get('weight_shrink', '—')} · "
                f"overlay={cfg_final.get('probabilistic_mode', cfg_final.get('overlay_label', '—'))} · "
                f"universe_bucket={cfg_final.get('universe_bucket', '—')}"
            )




def _resolve_basic_engine_state(cfg_final: dict) -> dict:
    """Read basic engine override widget state without rendering the widgets.

    This lets the page show the Run button before the advanced controls while still
    respecting any advanced values the user selected on a previous rerun.
    """
    universe_size = _safe_int(st.session_state.get("universe_size", 25), 25)

    def _num(key: str, fallback: Any, cast=float):
        raw = st.session_state.get(key, fallback)
        try:
            return cast(raw)
        except Exception:
            try:
                return cast(fallback)
            except Exception:
                return fallback

    top_k = _num("step5_basic_top_k", cfg_final.get("top_k", 12), int)
    top_k = max(1, min(int(top_k), max(1, universe_size)))
    signal_mode = str(st.session_state.get("step5_basic_signal_mode", cfg_final.get("signal_mode", "mu_sigma")) or "mu_sigma")

    return {
        "top_k": int(top_k),
        "lookback_mu": int(_num("step5_basic_lookback_mu", cfg_final.get("lookback_mu", 12), int)),
        "lookback_sigma": int(_num("step5_basic_lookback_sigma", cfg_final.get("lookback_sigma", 12), int)),
        "signal_mode": signal_mode,
        "temperature": float(_num("step5_basic_temperature", cfg_final.get("temperature", 1.0), float)),
        "weight_shrink": float(_num("step5_basic_weight_shrink", cfg_final.get("weight_shrink", 0.05), float)),
        "inertia": float(_num("step5_basic_inertia", cfg_final.get("inertia", 0.0), float)),
        "feature_mu_enabled": bool(st.session_state.get("step5_basic_feature_mu_enabled", cfg_final.get("feature_mu_enabled", False))),
        "feature_mu_blend": float(_num("step5_basic_feature_mu_blend", cfg_final.get("feature_mu_blend", 0.25), float)),
    }


def _render_ready_to_run_section(
    *,
    cfg_final: dict,
    asset_panel_df: Any,
    current_philosophy: str,
    simple_cfg: dict,
    pre_run_advanced_cfg: dict,
    governance_status: dict,
    bordered: bool = True,
    show_detail_expanders: bool = True,
):
    panel_rows = int(len(asset_panel_df)) if isinstance(asset_panel_df, pd.DataFrame) else 0
    panel_assets = int(asset_panel_df["asset"].nunique()) if isinstance(asset_panel_df, pd.DataFrame) and "asset" in asset_panel_df.columns else 0
    universe_size = int(st.session_state.get("universe_size", 25) or 25)
    source_label = str(st.session_state.get("asset_panel_source_label", "Selected market-data panel") or "Selected market-data panel")
    panel_ready = bool(st.session_state.get("asset_panel_ready", False)) and panel_rows > 0
    gov_state = str((governance_status or {}).get("state", "coherent") or "coherent")
    engine_status = "Blocked" if gov_state == "blocked" or not panel_ready else "Ready"
    current_result_is_fresh, current_run_signature = _current_result_is_fresh_for_ready_card(cfg_final, asset_panel_df)

    ctx = st.container(border=True) if bordered else st.container()
    with ctx:
        if not panel_ready:
            st.warning("Market data is missing or not ready. Go back to Risk Profile & Asset Universe before running the engine.")
        elif gov_state == "blocked":
            st.error("This setup is blocked by governance. Open the setup controls above or return to Risk Profile & Asset Universe to repair it.")
        elif current_result_is_fresh:
            if bordered:
                st.caption(
                    "Current setup already run. Change the preset, sliders, or technical controls to enable a new portfolio test."
                )
        else:
            setup_status_summary = str(simple_cfg.get("setup_status_summary", "") or "").strip()
            if setup_status_summary:
                st.success(f"Ready to run: {setup_status_summary}, and the selected universe plus market-data panel are loaded.")
            else:
                st.success("Ready to run: the selected universe plus market-data panel are loaded.")

        if panel_ready and gov_state != "blocked" and not current_result_is_fresh:
            _render_pre_run_suggestion_depth_controls()
        else:
            _ensure_suggestion_depth_defaults()

        run_result = render_run_panel(
            simple_cfg,
            pre_run_advanced_cfg,
            governance_status=governance_status,
            cfg_final=cfg_final,
            compact=True,
        )

        return run_result

def _build_executed_setup_summary(
    *,
    cfg_final: dict,
    asset_panel_df: Any,
    current_philosophy: str,
    simple_cfg: dict,
) -> str:
    """Return a compact description of the active executed setup.

    Post-run, keep this detail inside the collapsed setup/rerun area so the
    result metrics appear immediately after the main title.
    """
    panel_assets = int(asset_panel_df["asset"].nunique()) if isinstance(asset_panel_df, pd.DataFrame) and "asset" in asset_panel_df.columns else 0
    universe_size = int(st.session_state.get("universe_size", 25) or 25)
    stored_run = _coerce_mapping(st.session_state.get("step5_last_run_result", {}))
    oos_raw = stored_run.get("oos_returns_monthly", stored_run.get("oos_returns_simple", []))
    try:
        oos_months = len(oos_raw.tolist() if hasattr(oos_raw, "tolist") else list(oos_raw or []))
    except Exception:
        oos_months = int(_safe_int(_coerce_mapping(stored_run.get("performance_summary", {})).get("periods", 0), 0))

    template = str(simple_cfg.get("template", "—") or "—")
    style = str(simple_cfg.get("preset", "—") or "—")
    active_parts = [
        f"Risk profile: {current_philosophy}",
        f"Selected universe: {universe_size} assets",
        f"Template: {template}",
        f"Style: {style}",
        f"Panel data: {panel_assets} assets",
    ]
    if oos_months > 0:
        active_parts.append(f"Evaluation: {oos_months} OOS months")
    return " · ".join(active_parts)

def render_step_5() -> None:
    # Gold Stable: remove legacy exploration state before rendering.
    clear_retired_step5_state()

    current_philosophy = str(st.session_state.get("investment_philosophy", "Balanced") or "Balanced")
    pre_run_advanced_cfg: dict[str, Any] = {}
    asset_panel_df = st.session_state.get("asset_panel_df")

    # Resolve without rendering the full setup first. If the latest run is still
    # fresh, keep setup/rerun controls collapsed so the result becomes primary.
    simple_cfg_state = resolve_simple_mode_state()
    cfg_state, gov_state = _resolve_cfg_final(simple_cfg_state, pre_run_advanced_cfg)
    cfg_state = {**dict(cfg_state or {}), **_resolve_basic_engine_state(cfg_state)}
    current_result_is_fresh, current_run_signature = _current_result_is_fresh_for_ready_card(cfg_state, asset_panel_df)
    stored_run_result = st.session_state.get("step5_last_run_result")
    has_fresh_stored_result = bool(stored_run_result is not None and current_result_is_fresh)

    # Before a run, this screen is the Strategy Engine setup. After a fresh run,
    # make the result state explicit and avoid a second large "result overview"
    # title immediately above the metrics.
    if has_fresh_stored_result:
        st.markdown("## Strategy Engine Results")
        st.success(
            "Strategy Engine run completed successfully. Results are ready for review and long-term scenario projection."
        )
    else:
        st.markdown("## Strategy Engine")

    new_run_result = None
    current_signature = current_run_signature
    simple_cfg = simple_cfg_state
    cfg_final = dict(cfg_state or {})
    gov = dict(gov_state or {})

    if has_fresh_stored_result:
        active_setup_summary = _build_executed_setup_summary(
            cfg_final=cfg_final,
            asset_panel_df=asset_panel_df,
            current_philosophy=current_philosophy,
            simple_cfg=simple_cfg,
        )

        with st.expander("Change or rerun setup", expanded=False):
            if active_setup_summary:
                st.markdown(f"**Active result:** {active_setup_summary}.")
            st.caption(
                "Current setup already run. Change the preset, sliders, or technical controls to enable a new portfolio test."
            )
            technical_engine_overrides: dict[str, Any] = {}

            def _post_run_technical_footer(simple_cfg_payload: dict) -> None:
                footer_cfg, _footer_gov = _resolve_cfg_final(simple_cfg_payload, pre_run_advanced_cfg)
                technical_engine_overrides.clear()
                technical_engine_overrides.update(_render_technical_engine_overrides(footer_cfg))

            simple_cfg = render_simple_mode(
                use_internal_expanders=False,
                posture_footer_renderer=_post_run_technical_footer,
            )
            cfg_final, gov = _resolve_cfg_final(simple_cfg, pre_run_advanced_cfg)
            cfg_final = {**dict(cfg_final or {}), **dict(technical_engine_overrides or {})}

            current_signature = _build_step5_input_signature(cfg_final, asset_panel_df)
            st.session_state["step5_current_input_signature"] = current_signature
            clear_retired_step5_state()

            new_run_result = _render_ready_to_run_section(
                cfg_final=cfg_final,
                asset_panel_df=asset_panel_df,
                current_philosophy=current_philosophy,
                simple_cfg=simple_cfg,
                pre_run_advanced_cfg=pre_run_advanced_cfg,
                governance_status=gov,
                bordered=False,
                show_detail_expanders=False,
            )

    else:
        st.info(
            "**Purpose:** turn the selected risk profile, asset universe and market-data panel into a tested "
            "strategy return series. The Strategy Engine creates the portfolio first; CAGR, volatility, "
            "Sharpe and max drawdown are calculated afterwards."
        )
        # First-run or stale-run view: keep the setup card focused on editable
        # strategy choices. Readiness, input transparency and the run action sit
        # outside the card so they read as execution controls rather than setup
        # fields.
        with st.container(border=True):
            technical_engine_overrides: dict[str, Any] = {}

            def _first_run_technical_footer(simple_cfg_payload: dict) -> None:
                footer_cfg, _footer_gov = _resolve_cfg_final(simple_cfg_payload, pre_run_advanced_cfg)
                technical_engine_overrides.clear()
                technical_engine_overrides.update(_render_technical_engine_overrides(footer_cfg))

            simple_cfg = render_simple_mode(
                posture_footer_renderer=_first_run_technical_footer,
            )
            cfg_final, gov = _resolve_cfg_final(simple_cfg, pre_run_advanced_cfg)
            cfg_final = {**dict(cfg_final or {}), **dict(technical_engine_overrides or {})}

            current_signature = _build_step5_input_signature(cfg_final, asset_panel_df)
            st.session_state["step5_current_input_signature"] = current_signature
            clear_retired_step5_state()

        new_run_result = _render_ready_to_run_section(
            cfg_final=cfg_final,
            asset_panel_df=asset_panel_df,
            current_philosophy=current_philosophy,
            simple_cfg=simple_cfg,
            pre_run_advanced_cfg=pre_run_advanced_cfg,
            governance_status=gov,
            bordered=False,
        )

    # render_run_panel publishes the canonical signature when it is mounted.
    current_signature = str(
        st.session_state.get("step5_current_run_signature", current_signature)
        or current_signature
    )
    st.session_state["step5_current_input_signature"] = current_signature

    last_run_signature = str(st.session_state.get("step5_last_run_signature", "") or "")
    stored_run_result = st.session_state.get("step5_last_run_result")
    can_reuse_stored_post_run = bool(
        stored_run_result is not None
        and (last_run_signature == current_signature or has_fresh_stored_result)
    )
    run_result = new_run_result if new_run_result is not None else (
        stored_run_result if can_reuse_stored_post_run else None
    )

    execution_changed = bool(last_run_signature and last_run_signature != current_signature and not has_fresh_stored_result)
    post_run_is_stale = bool(new_run_result is None and stored_run_result is not None and execution_changed)
    st.session_state["step5_post_run_is_stale"] = False

    if post_run_is_stale:
        st.info(
            "The current strategy engine inputs differ from the last executed run. "
            "Run the updated setup to refresh the metrics."
        )

    if run_result:
        render_post_run(run_result)

    # Render fallback navigation only before a run exists. Once render_post_run()
    # is visible, it owns the strategy engine navigation buttons.
    if not run_result:
        st.markdown("---")
        nav_left, nav_right = st.columns(2)
        with nav_left:
            if st.button("← Back to Risk Profile & Asset Universe", key="step5_back_to_step4", use_container_width=True):
                st.session_state["current_step"] = 4
                st.rerun()
        with nav_right:
            if st.button(
                "Continue to Long-Term Scenario →",
                key="step5_continue_to_step6",
                use_container_width=True,
                disabled=not bool(st.session_state.get("engine_has_run", False)),
            ):
                st.session_state["current_step"] = 6
                st.rerun()
