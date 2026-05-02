from typing import Any, Callable

import streamlit as st

from ui.services.step4_universe_service import (
    SEMANTIC_SLIDER_KEYS,
    SEMANTIC_TOUCHED_FLAG,
    allowed_style_presets_for_philosophy,
    allowed_strategy_templates_for_philosophy,
    apply_semantic_slider_defaults,
    get_canonical_investment_philosophy,
    mark_semantic_sliders_dirty,
    recommended_strategy_combo_for_philosophy,
    strategy_combo_status,
)


TEMPLATE_DESCRIPTIONS = {
    "Balanced Risk-Controlled": "Risk-aware engine with drawdown control.",
    "Core Ranking": "Ranks assets mainly by expected return and risk.",
    "Hybrid Research": "More experimental blend of signals and overlays.",
}

STYLE_DESCRIPTIONS = {
    "Conservative": "Lower-risk posture with more protection.",
    "Balanced": "Middle-ground posture between growth and stability.",
    "Growth": "Higher-upside posture with more volatility.",
    "Defensive": "More cautious posture focused on stability.",
    "Research": "Experimental setup for testing signal behaviour.",
}


def _template_description(template: str) -> str:
    return TEMPLATE_DESCRIPTIONS.get(str(template or ""), "Strategy behaviour used by the engine run.")


def _style_description(style: str) -> str:
    return STYLE_DESCRIPTIONS.get(str(style or ""), "Risk posture applied to the selected template.")


def _setup_status_summary(
    *,
    combo_status: str,
    philosophy: str,
    template: str,
    style: str,
    rec_template: str,
    rec_style: str,
    applied_preset_label: str,
) -> str:
    current_combo_label = f"{template} + {style}"
    default_combo_label = f"{rec_template} + {rec_style}"
    if combo_status == "recommended":
        return f"recommended {philosophy} setup selected"
    if applied_preset_label == current_combo_label:
        return f"rerun-tested preset selected ({current_combo_label}); original {philosophy} default is {default_combo_label}"
    return f"allowed {philosophy} setup selected ({current_combo_label}); default is {default_combo_label}"


def _sync_combo_to_philosophy_space(philosophy: str) -> tuple[str, str]:
    """Return a valid template/style pair without mutating widget-backed keys.

    Streamlit warns when a widget key is assigned through session_state in the
    same render in which the widget is created. Keep this resolver read-only and
    sync the public Step 5 keys only after the selectboxes have rendered.
    """
    rec_template, rec_style = recommended_strategy_combo_for_philosophy(philosophy)
    allowed_templates = allowed_strategy_templates_for_philosophy(philosophy)
    current_template = str(st.session_state.get("step5_template", rec_template) or rec_template)
    if current_template not in allowed_templates:
        current_template = rec_template

    allowed_styles = allowed_style_presets_for_philosophy(philosophy, current_template)
    current_style = str(st.session_state.get("step5_style", rec_style) or rec_style)
    if current_style not in allowed_styles:
        current_style = rec_style if rec_style in allowed_styles else allowed_styles[0]
    return current_template, current_style


def _slider(widget_key: str, label: str, default: float) -> float:
    if widget_key not in st.session_state:
        st.session_state[widget_key] = float(default)
    return float(
        st.slider(
            label,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key=widget_key,
            on_change=mark_semantic_sliders_dirty,
        )
    )



def _safe_slider_state(logical_key: str, fallback: float) -> float:
    widget_key = SEMANTIC_SLIDER_KEYS.get(logical_key, "")
    try:
        return float(st.session_state.get(widget_key, fallback))
    except Exception:
        return float(fallback)


def resolve_simple_mode_state() -> dict:
    """Return the current Step 5 strategy state without rendering controls."""
    philosophy = get_canonical_investment_philosophy()
    rec_template, rec_style = recommended_strategy_combo_for_philosophy(philosophy)
    current_template = str(st.session_state.get("step5_template", rec_template) or rec_template)
    allowed_templates = allowed_strategy_templates_for_philosophy(philosophy)
    if current_template not in allowed_templates:
        current_template = rec_template

    allowed_styles = allowed_style_presets_for_philosophy(philosophy, current_template)
    current_style = str(st.session_state.get("step5_style", rec_style) or rec_style)
    if current_style not in allowed_styles:
        current_style = rec_style if rec_style in allowed_styles else allowed_styles[0]

    risk = _safe_slider_state("risk_appetite", 0.50)
    drawdown = _safe_slider_state("drawdown_protection", 0.62)
    divers = _safe_slider_state("diversification_vs_concentration", 0.55)
    overlay = _safe_slider_state("overlay_intensity", 0.42)
    stability = _safe_slider_state("stability_vs_responsiveness", 0.60)
    confidence = _safe_slider_state("confidence_in_signal", 0.56)
    turnover_style = _safe_slider_state("low_turnover_vs_adaptive", 0.40)
    simplicity = _safe_slider_state("simplicity_vs_sophistication", 0.56)
    combo_status = strategy_combo_status(philosophy, current_template, current_style)
    applied_preset_label = str(st.session_state.get("step5_preset_applied_label_v2", "") or "")
    setup_status_summary = _setup_status_summary(
        combo_status=combo_status,
        philosophy=philosophy,
        template=current_template,
        style=current_style,
        rec_template=rec_template,
        rec_style=rec_style,
        applied_preset_label=applied_preset_label,
    )
    risk_penalty = max(0.0, min(1.0, (1.0 - risk) * 0.40 + drawdown * 0.60))
    concentration = 5 + int(round(divers * 20))

    return {
        "preset": current_style,
        "template": current_template,
        "risk_appetite": risk,
        "drawdown_protection": drawdown,
        "diversification_vs_concentration": divers,
        "overlay_intensity": overlay,
        "stability_vs_responsiveness": stability,
        "confidence_in_signal": confidence,
        "low_turnover_vs_adaptive": turnover_style,
        "simplicity_vs_sophistication": simplicity,
        "risk_penalty": risk_penalty,
        "concentration": concentration,
        "combo_status": combo_status,
        "setup_status_summary": setup_status_summary,
        "recommended_template": rec_template,
        "recommended_style": rec_style,
        "philosophy": philosophy,
    }

def render_simple_mode(*, use_internal_expanders: bool = True, posture_footer_renderer: Callable[[dict], Any] | None = None):
    philosophy = get_canonical_investment_philosophy()
    rec_template, rec_style = recommended_strategy_combo_for_philosophy(philosophy)

    # Gold Stable: keep semantic defaults stable during a forced rerun,
    # then return to normal user-controlled interactions.
    freeze_after_apply = bool(st.session_state.get("step5_force_run_once", False))

    if freeze_after_apply:
        current_template = str(st.session_state.get("step5_template", rec_template) or rec_template)
        current_style = str(st.session_state.get("step5_style", rec_style) or rec_style)
    else:
        current_template, current_style = _sync_combo_to_philosophy_space(philosophy)

    template_options = allowed_strategy_templates_for_philosophy(philosophy)
    if current_template not in template_options:
        current_template = rec_template if rec_template in template_options else template_options[0]

    # Keep the selectboxes keyed directly to the public Step 5 state keys.
    # Previously the widgets were unkeyed and then copied into session_state
    # after rendering. That can leave Streamlit's internal widget value one
    # rerun behind the public key, making the user select a preset twice.
    st.session_state["step5_template"] = current_template

    left, right = st.columns(2)
    with left:
        template = st.selectbox(
            "Strategy template",
            template_options,
            key="step5_template",
        )
        st.caption(_template_description(template))

    style_options = allowed_style_presets_for_philosophy(philosophy, template)
    current_style_after_template = str(st.session_state.get("step5_style", current_style) or current_style)
    if current_style_after_template not in style_options:
        current_style_after_template = rec_style if rec_style in style_options else style_options[0]
    st.session_state["step5_style"] = current_style_after_template

    with right:
        style = st.selectbox(
            "Style preset",
            style_options,
            key="step5_style",
        )
        st.caption(_style_description(style))

    if not freeze_after_apply:
        apply_semantic_slider_defaults(template, style, force=False)

    combo_status = strategy_combo_status(philosophy, template, style)
    current_combo_label = f"{template} + {style}"
    default_combo_label = f"{rec_template} + {rec_style}"
    applied_preset_label = str(st.session_state.get("step5_preset_applied_label_v2", "") or "")

    setup_status_summary = _setup_status_summary(
        combo_status=combo_status,
        philosophy=philosophy,
        template=template,
        style=style,
        rec_template=rec_template,
        rec_style=rec_style,
        applied_preset_label=applied_preset_label,
    )

    if use_internal_expanders:
        posture_ctx = st.expander("Fine-tune strategy settings (optional)", expanded=False)
    else:
        st.markdown("**Fine-tune strategy settings (optional)**")
        posture_ctx = st.container(border=True)

    with posture_ctx:
        r1, r2 = st.columns(2)
        with r1:
            _slider(SEMANTIC_SLIDER_KEYS["risk_appetite"], "Risk appetite", 0.50)
            _slider(SEMANTIC_SLIDER_KEYS["diversification_vs_concentration"], "Diversification ↔ concentration", 0.55)
            _slider(SEMANTIC_SLIDER_KEYS["stability_vs_responsiveness"], "Stability ↔ responsiveness", 0.60)
            _slider(SEMANTIC_SLIDER_KEYS["low_turnover_vs_adaptive"], "Low turnover ↔ adaptive", 0.40)
        with r2:
            _slider(SEMANTIC_SLIDER_KEYS["drawdown_protection"], "Drawdown protection", 0.62)
            _slider(SEMANTIC_SLIDER_KEYS["overlay_intensity"], "Overlay intensity", 0.42)
            _slider(SEMANTIC_SLIDER_KEYS["confidence_in_signal"], "Confidence in signal", 0.56)
            _slider(SEMANTIC_SLIDER_KEYS["simplicity_vs_sophistication"], "Simplicity ↔ sophistication", 0.56)

        touched = bool(st.session_state.get(SEMANTIC_TOUCHED_FLAG, False))
        if touched:
            st.caption(
                "These sliders have been manually customised. Resetting returns them to the current preset defaults."
            )
        else:
            st.caption(
                "These sliders currently match the selected preset defaults. Use reset after experimenting to return to the preset baseline."
            )

        reset_col, _ = st.columns([1.25, 2.75])
        with reset_col:
            if st.button("Reset preset sliders", key="step5_reset_sliders_to_preset_defaults", use_container_width=True):
                apply_semantic_slider_defaults(template, style, force=True)
                st.rerun()

        # Read values while still inside the optional settings block so callers
        # can append related controls, such as technical overrides, in the same
        # visual group.
        risk = _safe_slider_state("risk_appetite", 0.50)
        drawdown = _safe_slider_state("drawdown_protection", 0.62)
        divers = _safe_slider_state("diversification_vs_concentration", 0.55)
        overlay = _safe_slider_state("overlay_intensity", 0.42)
        stability = _safe_slider_state("stability_vs_responsiveness", 0.60)
        confidence = _safe_slider_state("confidence_in_signal", 0.56)
        turnover_style = _safe_slider_state("low_turnover_vs_adaptive", 0.40)
        simplicity = _safe_slider_state("simplicity_vs_sophistication", 0.56)

        risk_penalty = max(0.0, min(1.0, (1.0 - risk) * 0.40 + drawdown * 0.60))
        concentration = 5 + int(round(divers * 20))

        simple_cfg_payload = {
            "preset": style,
            "template": template,
            "risk_appetite": risk,
            "drawdown_protection": drawdown,
            "diversification_vs_concentration": divers,
            "overlay_intensity": overlay,
            "stability_vs_responsiveness": stability,
            "confidence_in_signal": confidence,
            "low_turnover_vs_adaptive": turnover_style,
            "simplicity_vs_sophistication": simplicity,
            "risk_penalty": risk_penalty,
            "concentration": concentration,
            "combo_status": combo_status,
            "setup_status_summary": setup_status_summary,
            "recommended_template": rec_template,
            "recommended_style": rec_style,
            "philosophy": philosophy,
        }

        if callable(posture_footer_renderer):
            st.divider()
            posture_footer_renderer(dict(simple_cfg_payload))

    return simple_cfg_payload
