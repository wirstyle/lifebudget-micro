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


def _render_basic_engine_controls(cfg_final: dict) -> dict:
    st.markdown("### Advanced engine controls")
    st.caption("Optional low-level parameters. Leave these unchanged unless you are deliberately testing engine behaviour.")

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

    st.caption(
        f"Effective basic controls → top_k={top_k} · lookback_mu={lookback_mu} · "
        f"lookback_sigma={lookback_sigma} · signal_mode={signal_mode} · "
        f"temperature={temperature} · weight_shrink={weight_shrink} · inertia={inertia} · "
        f"feature_mu_enabled={feature_mu_enabled} · feature_mu_blend={feature_mu_blend}"
    )

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
            st.warning("Market data is missing or not ready. Return to Risk Profile and Universe before running the engine.")
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
            st.warning("Market data is missing or not ready. Go back to Risk Profile and Universe before running the engine.")
        elif gov_state == "blocked":
            st.error("This setup is blocked by governance. Open the setup controls above or return to Risk Profile and Universe to repair it.")
        elif current_result_is_fresh:
            st.success("Result up to date: this setup has already been executed by the real engine.")
        else:
            setup_status_summary = str(simple_cfg.get("setup_status_summary", "") or "").strip()
            if setup_status_summary:
                st.success(f"Ready to run: {setup_status_summary}, and the selected universe plus market-data panel are loaded.")
            else:
                st.success("Ready to run: the selected universe plus market-data panel are loaded.")

        run_result = render_run_panel(
            simple_cfg,
            pre_run_advanced_cfg,
            governance_status=governance_status,
            cfg_final=cfg_final,
            compact=True,
        )

        if show_detail_expanders:
            _render_engine_transparency_intro()

            with st.expander("Strategy engine inputs", expanded=False):
                setup_line = (
                    f"**{current_philosophy}** · **{universe_size}-asset universe** · "
                    f"**{simple_cfg.get('template', '—')}** · **{simple_cfg.get('preset', '—')}**"
                )
                st.write(setup_line)

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    _render_compact_value("Market data", source_label)
                with c2:
                    _render_compact_value("Panel rows", f"{panel_rows:,}")
                with c3:
                    _render_compact_value("Panel assets", panel_assets)
                with c4:
                    _render_compact_value("Engine status", engine_status)

                st.caption(
                    f"top_k={cfg_final.get('top_k', '—')} · "
                    f"signal_mode={cfg_final.get('signal_mode', '—')} · "
                    f"lookback_mu={cfg_final.get('lookback_mu', '—')} · "
                    f"lookback_sigma={cfg_final.get('lookback_sigma', '—')} · "
                    f"temperature={cfg_final.get('temperature', '—')} · "
                    f"weight_shrink={cfg_final.get('weight_shrink', '—')} · "
                    f"overlay={cfg_final.get('probabilistic_mode', cfg_final.get('overlay_label', '—'))}"
                )
                applied_tuning_signature = str(st.session_state.get("step5_auto_opt_applied_run_signature_v1", "") or "")
                applied_tuning_label = str(st.session_state.get("step5_auto_opt_applied_label_v1", "") or "")
                if current_result_is_fresh and current_run_signature and applied_tuning_signature == current_run_signature:
                    label_text = f": {applied_tuning_label}" if applied_tuning_label else ""
                    st.caption(f"These technical values include the applied engine tuning suggestion{label_text}.")

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

    template = simple_cfg.get("template", "â")
    style = simple_cfg.get("preset", "â")
    active_line = (
        f"{current_philosophy} Â· {universe_size}-asset universe Â· "
        f"{template} Â· {style} Â· {panel_assets} panel assets"
    )
    if oos_months > 0:
        active_line += f" Â· {oos_months} OOS months"
    return active_line

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
                "Open this only if you want to change the preset, semantic posture, technical controls, "
                "or run a new portfolio test."
            )
            simple_cfg = render_simple_mode(use_internal_expanders=False)
            cfg_final, gov = _resolve_cfg_final(simple_cfg, pre_run_advanced_cfg)

            st.markdown("**Optional advanced engine controls**")
            with st.container(border=True):
                st.caption(
                    "Low-level parameters resolved from the preset above. Normal users can leave these unchanged; "
                    "use this only for diagnostics or controlled experimentation."
                )
                basic_engine_overrides = _render_basic_engine_controls(cfg_final)
            cfg_final = {**dict(cfg_final or {}), **dict(basic_engine_overrides or {})}

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
            "strategy return series. This is the execution stage, not a new data-preparation step."
        )
        # First-run or stale-run view: keep the setup card focused on editable
        # strategy choices. Readiness, input transparency and the run action sit
        # outside the card so they read as execution controls rather than setup
        # fields.
        with st.container(border=True):
            simple_cfg = render_simple_mode()
            cfg_final, gov = _resolve_cfg_final(simple_cfg, pre_run_advanced_cfg)

            with st.expander("Optional advanced engine controls", expanded=False):
                st.caption(
                    "Low-level parameters resolved from the preset above. Normal users can leave these unchanged; "
                    "use this only for diagnostics or controlled experimentation."
                )
                basic_engine_overrides = _render_basic_engine_controls(cfg_final)
            cfg_final = {**dict(cfg_final or {}), **dict(basic_engine_overrides or {})}

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
            if st.button("← Back to Risk Profile and Universe", key="step5_back_to_step4", use_container_width=True):
                st.session_state["current_step"] = 4
                st.rerun()
        with nav_right:
            if st.button(
                "Continue to Long-Term Scenario Explorer →",
                key="step5_continue_to_step6",
                use_container_width=True,
                disabled=not bool(st.session_state.get("engine_has_run", False)),
            ):
                st.session_state["current_step"] = 6
                st.rerun()
