
import streamlit as st

from src.investment import SimpleUISpec, config_to_dict, resolve_user_intention_to_governed_config


def _build_simple_spec(simple_cfg: dict) -> SimpleUISpec:
    payload = dict(simple_cfg or {})
    try:
        universe_size = int(st.session_state.get("universe_size", 25) or 25)
    except Exception:
        universe_size = 25
    return SimpleUISpec(
        strategy_template=str(payload.get("template", "Balanced Risk-Controlled") or "Balanced Risk-Controlled"),
        style_preset=str(payload.get("preset", "Balanced") or "Balanced"),
        risk_appetite=float(payload.get("risk_appetite", 0.50) or 0.50),
        diversification=float(payload.get("diversification_vs_concentration", 0.55) or 0.55),
        stability=float(payload.get("stability_vs_responsiveness", 0.60) or 0.60),
        turnover_pref=float(payload.get("low_turnover_vs_adaptive", 0.40) or 0.40),
        drawdown_protection=float(payload.get("drawdown_protection", 0.62) or 0.62),
        overlay_intensity=float(payload.get("overlay_intensity", 0.42) or 0.42),
        signal_confidence=float(payload.get("confidence_in_signal", 0.56) or 0.56),
        simplicity=float(payload.get("simplicity_vs_sophistication", 0.56) or 0.56),
        universe_size=universe_size,
    )


def resolve_governance_status(simple_cfg, advanced_cfg, resolved_cfg=None):
    simple_cfg = dict(simple_cfg or {})
    advanced_cfg = dict(advanced_cfg or {})
    resolved_cfg = dict(resolved_cfg or {})

    philosophy = str(
        simple_cfg.get(
            "philosophy",
            st.session_state.get("investment_philosophy", "Balanced"),
        )
        or "Balanced"
    )

    simple_spec = _build_simple_spec(simple_cfg)

    overrides = dict(resolved_cfg or {})
    overrides.update(dict(advanced_cfg or {}))

    governed = resolve_user_intention_to_governed_config(
        simple_spec=simple_spec,
        philosophy=philosophy,
        universe=simple_spec.universe_size,
        strategy_template=simple_spec.strategy_template,
        style_preset=simple_spec.style_preset,
        overrides=overrides,
        trace_source="step5_workspace",
    )

    governance_payload = governed.to_governance_payload().to_dict()
    cfg_final = config_to_dict(governed.final_cfg)

    score = float(governance_payload.get("coherence_score") or 0.0)
    status = str(governance_payload.get("coherence_status", governed.status) or governed.status or "coherent")
    warnings = list(governance_payload.get("warnings", []) or [])
    suggested_repairs = dict(governance_payload.get("suggested_repairs", {}) or {})
    repair_patch = dict(suggested_repairs.get("suggested_patch", {}) or {})

    if status in {"blocked", "discouraged"} or score < 0.55:
        state = "blocked"
        operational = "Blocked"
        message = "Governance blocked this configuration. Structural inconsistencies detected."
    elif status in {"stretched", "auto_repair_available"} or score < 0.72:
        state = "stretched"
        operational = "Stretched"
        message = "Configuration is valid but stretched relative to its philosophy."
    else:
        state = "coherent"
        operational = "Operational"
        message = "Configuration is coherent and ready for execution."

    explanation = str(governance_payload.get("short_tradeoff_explanation", "") or "")

    return {
        "state": state,
        "operational_status": operational,
        "coherence_score": score,
        "coherence_score_display": f"{score:.2f}",
        "issues_count": len(warnings),
        "warnings": warnings,
        "repairs": repair_patch,
        "repair_summary": suggested_repairs,
        "message": message,
        "explanation": explanation,
        "cfg_final": cfg_final,
        "governed_result": governed,
        "governance_payload": governance_payload,
    }


def render_governance(simple_cfg, advanced_cfg):
    gov = resolve_governance_status(simple_cfg, advanced_cfg)
    st.markdown("### Governance & philosophy")
    st.caption(
        f"State: {gov.get('operational_status', '—')} · "
        f"Score: {gov.get('coherence_score_display', '—')}"
    )

    warnings = list(gov.get("warnings", []) or [])
    if warnings:
        st.warning("Warnings:")
        for w in warnings:
            st.write(f"- {w}")

    repair_summary = dict(gov.get("repair_summary", {}) or {})
    suggested_patch = dict(repair_summary.get("suggested_patch", {}) or {})
    actionable_warnings = list(repair_summary.get("actionable_warnings", []) or [])

    if actionable_warnings:
        st.info("Suggested repairs:")
        for item in actionable_warnings:
            st.write(f"- {item}")
    elif suggested_patch:
        st.info("Suggested repair patch available.")
