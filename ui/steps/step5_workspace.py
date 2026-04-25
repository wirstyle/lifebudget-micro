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


def _render_basic_engine_controls(cfg_final: dict) -> dict:
    st.markdown("### Basic engine controls")
    st.caption("Advanced controls. Defaults are already resolved from the Step 4 philosophy and strategy setup; change these only for diagnostics or experimentation.")

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


def _render_workspace_summary(cfg_final: dict, asset_panel_df: Any, current_philosophy: str, simple_cfg: dict) -> None:
    panel_rows = int(len(asset_panel_df)) if isinstance(asset_panel_df, pd.DataFrame) else 0
    panel_assets = int(asset_panel_df["asset"].nunique()) if isinstance(asset_panel_df, pd.DataFrame) and "asset" in asset_panel_df.columns else 0
    universe_size = int(st.session_state.get("universe_size", 25) or 25)
    panel_ready = bool(st.session_state.get("asset_panel_ready", False)) and panel_rows > 0

    with st.container(border=True):
        st.markdown("### 1. Current setup")
        st.caption("The engine will use the Step 4 universe and market-data panel with the current strategy preset.")
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
            st.success("Ready to run: Step 4 market data is loaded and the current setup is coherent enough for execution.")
        else:
            st.warning("Step 4 market data is missing or not ready. Go back to Step 4 before running the engine.")
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
):
    panel_rows = int(len(asset_panel_df)) if isinstance(asset_panel_df, pd.DataFrame) else 0
    panel_assets = int(asset_panel_df["asset"].nunique()) if isinstance(asset_panel_df, pd.DataFrame) and "asset" in asset_panel_df.columns else 0
    universe_size = int(st.session_state.get("universe_size", 25) or 25)
    source_label = str(st.session_state.get("asset_panel_source_label", "Yahoo Finance monthly panel") or "Yahoo Finance monthly panel")
    panel_ready = bool(st.session_state.get("asset_panel_ready", False)) and panel_rows > 0
    gov_state = str((governance_status or {}).get("state", "coherent") or "coherent")
    engine_status = "Blocked" if gov_state == "blocked" or not panel_ready else "Ready"

    with st.container(border=True):
        st.markdown("### 2. Ready to run")
        st.caption(
            "This is the final execution summary after the strategy preset and any optional technical controls. "
            "Press Run only when the setup below matches what you want to test."
        )

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

        if not panel_ready:
            st.warning("Step 4 market data is missing or not ready. Go back to Step 4 before running the engine.")
        elif gov_state == "blocked":
            st.error("This setup is blocked by governance. Open the setup controls above or return to Step 4 to repair it.")
        else:
            st.success("Ready to run: this setup can now be executed by the real engine.")

        with st.expander("Technical execution details", expanded=False):
            st.caption(
                f"top_k={cfg_final.get('top_k', '—')} · "
                f"signal_mode={cfg_final.get('signal_mode', '—')} · "
                f"lookback_mu={cfg_final.get('lookback_mu', '—')} · "
                f"lookback_sigma={cfg_final.get('lookback_sigma', '—')} · "
                f"temperature={cfg_final.get('temperature', '—')} · "
                f"weight_shrink={cfg_final.get('weight_shrink', '—')} · "
                f"overlay={cfg_final.get('probabilistic_mode', cfg_final.get('overlay_label', '—'))}"
            )

        st.caption("This button runs the real micro pipeline and stores the result for metrics, Step 6 projection, and Step 7 insights.")
        return render_run_panel(
            simple_cfg,
            pre_run_advanced_cfg,
            governance_status=governance_status,
            cfg_final=cfg_final,
            compact=True,
        )

def render_step_5() -> None:
    # Gold Stable: remove legacy exploration state before rendering.
    clear_retired_step5_state()

    st.markdown("## Step 5 — Engine workspace")
    st.caption(
        "This step converts the Step 4 universe and market-data panel into a real investment-engine run. "
        "Start with the strategy preset, optionally review technical controls, then run the engine and interpret the result before moving to projection."
    )

    current_philosophy = str(st.session_state.get("investment_philosophy", "Balanced") or "Balanced")

    # 1) Main user-facing setup layer. This is intentionally visible because it is the
    # product-friendly alternative to exposing dozens of low-level engine knobs first.
    with st.container(border=True):
        simple_cfg = render_simple_mode()

    # Gold Stable: advanced/global-search controls are intentionally not mounted.
    # Keep the argument as an empty dict so downstream function signatures stay stable.
    pre_run_advanced_cfg: dict[str, Any] = {}

    cfg_final, gov = _resolve_cfg_final(simple_cfg, pre_run_advanced_cfg)

    # 2) Optional low-level controls. They are closed by default, but because they can
    # affect the execution they must appear before the final Ready-to-run confirmation.
    with st.expander("Optional technical engine controls (diagnostics)", expanded=False):
        st.caption(
            "These are low-level engine parameters resolved from the strategy preset above. "
            "Normal users do not need to change them; use this only for diagnostics or experimentation."
        )
        basic_engine_overrides = _render_basic_engine_controls(cfg_final)
    cfg_final = {**dict(cfg_final or {}), **dict(basic_engine_overrides or {})}

    asset_panel_df = st.session_state.get("asset_panel_df")
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
    )

    last_run_signature = str(st.session_state.get("step5_last_run_signature", "") or "")
    stored_run_result = st.session_state.get("step5_last_run_result")
    can_reuse_stored_post_run = stored_run_result is not None and last_run_signature == current_signature
    run_result = new_run_result if new_run_result is not None else (
        stored_run_result if can_reuse_stored_post_run else None
    )

    execution_changed = bool(last_run_signature and last_run_signature != current_signature)
    post_run_is_stale = bool(new_run_result is None and stored_run_result is not None and execution_changed)
    st.session_state["step5_post_run_is_stale"] = False

    if post_run_is_stale:
        st.info(
            "The current Step 5 inputs differ from the last executed run. "
            "Press 'Run portfolio & view results' to refresh the metrics."
        )

    if run_result:
        render_post_run(run_result)

    # Gold Stable: render fallback navigation only before a run exists.
    # Once render_post_run() is visible, it owns the Step 5 navigation buttons,
    # avoiding duplicate Streamlit widget keys.
    if not run_result:
        st.markdown("---")
        nav_left, nav_right = st.columns(2)
        with nav_left:
            if st.button("← Back to Step 4", key="step5_back_to_step4", use_container_width=True):
                st.session_state["current_step"] = 4
                st.rerun()
        with nav_right:
            if st.button(
                "Continue to Step 6 →",
                key="step5_continue_to_step6",
                use_container_width=True,
                disabled=not bool(st.session_state.get("engine_has_run", False)),
            ):
                st.session_state["current_step"] = 6
                st.rerun()
