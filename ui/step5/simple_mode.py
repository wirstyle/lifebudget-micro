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


def _sync_combo_to_philosophy_space(philosophy: str) -> tuple[str, str]:
    rec_template, rec_style = recommended_strategy_combo_for_philosophy(philosophy)
    allowed_templates = allowed_strategy_templates_for_philosophy(philosophy)
    current_template = str(st.session_state.get("step5_template", rec_template) or rec_template)
    if current_template not in allowed_templates:
        current_template = rec_template
        st.session_state["step5_template"] = current_template

    allowed_styles = allowed_style_presets_for_philosophy(philosophy, current_template)
    current_style = str(st.session_state.get("step5_style", rec_style) or rec_style)
    if current_style not in allowed_styles:
        fallback_style = rec_style if rec_style in allowed_styles else allowed_styles[0]
        current_style = fallback_style
        st.session_state["step5_style"] = current_style
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
        "recommended_template": rec_template,
        "recommended_style": rec_style,
        "philosophy": philosophy,
    }

def render_simple_mode():
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

    st.markdown("### 1. Strategy preset setup")
    st.caption(
        "This is the user-friendly control layer for the engine. Template and style define the main behaviour, "
        "while the semantic sliders fine-tune the posture without exposing every technical knob."
    )

    combo_status = strategy_combo_status(philosophy, current_template, current_style)
    if combo_status == "recommended":
        st.caption(f"Current combo is the recommended bundle for {philosophy}: {current_template} + {current_style}.")
    else:
        st.caption(f"Current combo is allowed for {philosophy}, but the recommended bundle is {rec_template} + {rec_style}.")

    c_action1, c_action2 = st.columns([1.2, 2.8])
    with c_action1:
        if st.button("Reset sliders to preset defaults", key="step5_reset_sliders_to_preset_defaults"):
            apply_semantic_slider_defaults(current_template, current_style, force=True)
            st.rerun()
    with c_action2:
        touched = bool(st.session_state.get(SEMANTIC_TOUCHED_FLAG, False))
        if touched:
            st.caption("Semantic sliders are manually customised and no longer strictly match the current preset defaults.")
        else:
            st.caption("Semantic sliders are currently aligned with the current preset defaults.")

    col1, col2 = st.columns(2)
    with col1:
        template_options = allowed_strategy_templates_for_philosophy(philosophy)
        template = st.selectbox(
            "Strategy template",
            template_options,
            index=template_options.index(current_template),
            key="step5_template",
        )

    current_style_after_template = str(st.session_state.get("step5_style", current_style) or current_style)
    style_options = allowed_style_presets_for_philosophy(philosophy, template)
    if current_style_after_template not in style_options:
        current_style_after_template = rec_style if rec_style in style_options else style_options[0]
        st.session_state["step5_style"] = current_style_after_template

    with col2:
        style = st.selectbox(
            "Style preset",
            style_options,
            index=style_options.index(current_style_after_template),
            key="step5_style",
        )

    if not freeze_after_apply:
        apply_semantic_slider_defaults(template, style, force=False)

    st.caption(
        "These sliders translate investment intent into the detailed engine configuration. "
        "Leave them unchanged for the Step 4 recommended preset behaviour."
    )

    r1, r2 = st.columns(2)
    with r1:
        risk = _slider(SEMANTIC_SLIDER_KEYS["risk_appetite"], "Risk appetite", 0.50)
        divers = _slider(SEMANTIC_SLIDER_KEYS["diversification_vs_concentration"], "Diversification ↔ concentration", 0.55)
        stability = _slider(SEMANTIC_SLIDER_KEYS["stability_vs_responsiveness"], "Stability ↔ responsiveness", 0.60)
        turnover_style = _slider(SEMANTIC_SLIDER_KEYS["low_turnover_vs_adaptive"], "Low turnover ↔ adaptive", 0.40)
    with r2:
        drawdown = _slider(SEMANTIC_SLIDER_KEYS["drawdown_protection"], "Drawdown protection", 0.62)
        overlay = _slider(SEMANTIC_SLIDER_KEYS["overlay_intensity"], "Overlay intensity", 0.42)
        confidence = _slider(SEMANTIC_SLIDER_KEYS["confidence_in_signal"], "Confidence in signal", 0.56)
        simplicity = _slider(SEMANTIC_SLIDER_KEYS["simplicity_vs_sophistication"], "Simplicity ↔ sophistication", 0.56)

    risk_penalty = max(0.0, min(1.0, (1.0 - risk) * 0.40 + drawdown * 0.60))
    concentration = 5 + int(round(divers * 20))

    return {
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
        "recommended_template": rec_template,
        "recommended_style": rec_style,
        "philosophy": philosophy,
    }
