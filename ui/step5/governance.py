"""Step 5 governance helpers for LifeBudget Micro.

This module resolves the Simple-mode Strategy Engine intent plus any advanced
manual overrides into a governed engine configuration. It also renders a compact
governance summary for advanced Step 5 views.

The governance layer is used to check whether manual changes remain coherent
with the selected investment philosophy before the Strategy Engine is executed.
"""

from __future__ import annotations

import math
from typing import Any

import streamlit as st

from src.investment import (
    SimpleUISpec,
    config_to_dict,
    resolve_user_intention_to_governed_config,
)


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


def _first_present(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return default


def _build_simple_spec(simple_cfg: dict) -> SimpleUISpec:
    """Build the governed Simple UI spec from current Step 5 simple settings.

    Supports both the current service payload keys and older semantic slider
    names so saved state and older callers remain compatible.
    """
    payload = dict(simple_cfg or {})
    universe_size = _safe_int(st.session_state.get("universe_size", 25), 25)

    return SimpleUISpec(
        strategy_template=str(
            _first_present(
                payload,
                "strategy_template",
                "template",
                default="Balanced Risk-Controlled",
            )
            or "Balanced Risk-Controlled"
        ),
        style_preset=str(
            _first_present(
                payload,
                "style_preset",
                "preset",
                default="Balanced",
            )
            or "Balanced"
        ),
        risk_appetite=_safe_float(
            _first_present(payload, "risk_appetite", default=0.50),
            0.50,
        ),
        diversification=_safe_float(
            _first_present(
                payload,
                "diversification",
                "diversification_vs_concentration",
                default=0.55,
            ),
            0.55,
        ),
        stability=_safe_float(
            _first_present(
                payload,
                "stability",
                "stability_vs_responsiveness",
                default=0.60,
            ),
            0.60,
        ),
        turnover_pref=_safe_float(
            _first_present(
                payload,
                "turnover_pref",
                "low_turnover_vs_adaptive",
                default=0.40,
            ),
            0.40,
        ),
        drawdown_protection=_safe_float(
            _first_present(payload, "drawdown_protection", default=0.62),
            0.62,
        ),
        overlay_intensity=_safe_float(
            _first_present(payload, "overlay_intensity", default=0.42),
            0.42,
        ),
        signal_confidence=_safe_float(
            _first_present(
                payload,
                "signal_confidence",
                "confidence_in_signal",
                default=0.56,
            ),
            0.56,
        ),
        simplicity=_safe_float(
            _first_present(
                payload,
                "simplicity",
                "simplicity_vs_sophistication",
                default=0.56,
            ),
            0.56,
        ),
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
        trace_source="step5_governance",
    )

    governance_payload = governed.to_governance_payload().to_dict()
    cfg_final = config_to_dict(governed.final_cfg)

    score = _safe_float(governance_payload.get("coherence_score"), 0.0)
    status = str(governance_payload.get("coherence_status", governed.status) or governed.status or "coherent")
    warnings = list(governance_payload.get("warnings", []) or [])
    suggested_repairs = dict(governance_payload.get("suggested_repairs", {}) or {})
    repair_patch = dict(suggested_repairs.get("suggested_patch", {}) or {})

    if status in {"blocked", "discouraged"} or score < 0.55:
        state = "blocked"
        operational = "Blocked"
        message = "Governance blocked this configuration because structural inconsistencies were detected."
    elif status in {"stretched", "auto_repair_available"} or score < 0.72:
        state = "stretched"
        operational = "Stretched"
        message = "Configuration is valid but stretched relative to its selected philosophy."
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
        "suggested_repair_patch": repair_patch,
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

    message = str(gov.get("message", "") or "")
    state = str(gov.get("state", "coherent") or "coherent")

    if message:
        if state == "blocked":
            st.error(message)
        elif state == "stretched":
            st.info(message)
        else:
            st.success(message)

    warnings = list(gov.get("warnings", []) or [])
    if warnings:
        st.warning("Warnings:")
        for warning in warnings:
            st.write(f"- {warning}")

    repair_summary = dict(gov.get("repair_summary", {}) or {})
    suggested_patch = dict(repair_summary.get("suggested_patch", {}) or {})
    actionable_warnings = list(repair_summary.get("actionable_warnings", []) or [])

    if actionable_warnings:
        st.info("Suggested repairs:")
        for item in actionable_warnings:
            st.write(f"- {item}")
    elif suggested_patch:
        st.info("Suggested repair patch available.")