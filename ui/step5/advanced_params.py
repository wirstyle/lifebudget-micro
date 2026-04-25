from __future__ import annotations

import streamlit as st


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _state_default(key: str, fallback):
    current = st.session_state.get(key, None)
    return fallback if current is None else current


def _bool_default(key: str, fallback: bool) -> bool:
    return bool(_state_default(key, fallback))



def _fmt_scalar(value) -> str:
    if value is None:
        return "—"
    try:
        if isinstance(value, bool):
            return "Yes" if value else "No"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)
    except Exception:
        return str(value)


def _normalize_status_label(governance_status: dict | None = None) -> str:
    gov = dict(governance_status or {})
    for key in ("label", "operational_status", "state", "status"):
        raw = str(gov.get(key, "") or "").strip()
        if raw:
            return raw
    return "—"


def _normalize_score_display(governance_status: dict | None = None) -> str:
    gov = dict(governance_status or {})
    raw = gov.get("coherence_score_display", None)
    if raw not in (None, ""):
        return str(raw)
    coherence_after = dict(gov.get("coherence_after", {}) or {})
    score = coherence_after.get("score_continuous", coherence_after.get("score", None))
    try:
        return f"{float(score):.2f}"
    except Exception:
        return "—"


def _collect_visible_warnings(governance_status: dict | None = None) -> list[str]:
    gov = dict(governance_status or {})
    warnings = []
    for item in list(gov.get("warnings", []) or []):
        if item:
            warnings.append(str(item))
    coherence_after = dict(gov.get("coherence_after", {}) or {})
    for key in ("warning", "message"):
        value = coherence_after.get(key)
        if value:
            warnings.append(str(value))
    repair_summary = gov.get("repair_summary") or gov.get("repairs") or {}
    if isinstance(repair_summary, dict):
        reason = repair_summary.get("reason") or repair_summary.get("message")
        if reason:
            warnings.append(str(reason))
    deduped = []
    seen = set()
    for item in warnings:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def _resolve_block_rows(cfg_final: dict | None = None, override_cfg: dict | None = None):
    cfg = dict(cfg_final or {})
    override = dict(override_cfg or {})
    block_defs = [
        {
            "block": "Signal block",
            "role": "Core",
            "keys": ["signal_mode", "signal_score_blend", "sigma_power_alpha", "top_k", "temperature"],
            "base": f"on · {cfg.get('signal_mode', 'mu_sigma')}",
            "current": f"on · {override.get('signal_mode', cfg.get('signal_mode', 'mu_sigma'))}",
            "expected_effect": "signal expression, robustness",
        },
        {
            "block": "Overlay block",
            "role": "Supporting",
            "keys": ["probabilistic_mode", "probabilistic_overlay_strength", "probabilistic_overlay_blend"],
            "base": f"on · {cfg.get('probabilistic_mode', cfg.get('overlay_label', 'historical'))}",
            "current": f"on · {override.get('probabilistic_mode', cfg.get('probabilistic_mode', cfg.get('overlay_label', 'historical')))}",
            "expected_effect": "robustness, downside moderation",
        },
        {
            "block": "Covariance / risk block",
            "role": "Core",
            "keys": [
                "covariance_mode",
                "correlation_penalty_strength",
                "covariance_shrink_to_diagonal",
                "correlation_shrink_to_identity",
                "target_portfolio_vol_monthly",
            ],
            "base": f"on · {cfg.get('covariance_mode', 'ewma_cov')}",
            "current": f"on · {override.get('covariance_mode', cfg.get('covariance_mode', 'ewma_cov'))}",
            "expected_effect": "drawdown control, robustness",
        },
        {
            "block": "Caps / diversification block",
            "role": "Core",
            "keys": ["asset_weight_cap", "w_cap", "weight_shrink"],
            "base": f"on · {_fmt_scalar(cfg.get('asset_weight_cap', cfg.get('w_cap', 'medium')))}",
            "current": (
                f"off · none"
                if override.get("asset_weight_cap", cfg.get("asset_weight_cap")) is None
                and override.get("w_cap", cfg.get("w_cap")) is None
                else f"on · {_fmt_scalar(override.get('asset_weight_cap', override.get('w_cap', cfg.get('asset_weight_cap', cfg.get('w_cap', 'medium')))))}"
            ),
            "expected_effect": "diversification, concentration control",
        },
        {
            "block": "Turnover / stability block",
            "role": "Supporting",
            "keys": ["turnover_penalty_strength", "turnover_penalty_power", "turnover_penalty_target"],
            "base": f"off · {_fmt_scalar(cfg.get('turnover_penalty_strength', 'low'))}",
            "current": f"off · {_fmt_scalar(override.get('turnover_penalty_strength', cfg.get('turnover_penalty_strength', 'low')))}",
            "expected_effect": "turnover control, stability",
        },
        {
            "block": "Allocator block",
            "role": "Core",
            "keys": ["correlation_allocator_method", "correlation_allocator_blend", "mean_variance_risk_aversion"],
            "base": f"on · {cfg.get('correlation_allocator_method', 'score_penalty')}",
            "current": f"on · {override.get('correlation_allocator_method', cfg.get('correlation_allocator_method', 'score_penalty'))}",
            "expected_effect": "portfolio construction, robustness",
        },
    ]

    rows = []
    changed_blocks = 0
    for item in block_defs:
        changed = any(override.get(key, cfg.get(key)) != cfg.get(key) for key in item["keys"] if key in cfg or key in override)
        if changed:
            changed_blocks += 1
        rows.append(
            {
                "Block": item["block"],
                "Role": item["role"],
                "Base": item["base"],
                "Current": item["current"],
                "Changed": "Yes" if changed else "No",
                "Active": "No" if str(item["current"]).startswith("off") else "Yes",
                "Expected effect": item["expected_effect"],
            }
        )
    return rows, changed_blocks


def _resolve_parameter_rows(cfg_final: dict | None = None, override_cfg: dict | None = None):
    cfg = dict(cfg_final or {})
    override = dict(override_cfg or {})
    rows = []
    parameter_defs = [
        ("top_k", "Signal block", "Core"),
        ("asset_weight_cap", "Caps / diversification block", "Core"),
        ("turnover_penalty_strength", "Turnover / stability block", "Supporting"),
        ("target_portfolio_vol_monthly", "Covariance / risk block", "Core"),
        ("correlation_penalty_strength", "Covariance / risk block", "Core"),
        ("sigma_power_alpha", "Signal block", "Core"),
        ("probabilistic_overlay_strength", "Overlay block", "Supporting"),
        ("covariance_shrink_to_diagonal", "Covariance / risk block", "Core"),
    ]
    for parameter, block, role in parameter_defs:
        base_value = cfg.get(parameter)
        current_value = override.get(parameter, base_value)
        changed = base_value != current_value
        rows.append(
            {
                "Parameter": parameter,
                "Block": block,
                "Role": role,
                "Base": _fmt_scalar(base_value),
                "Current": _fmt_scalar(current_value),
                "Changed": "Yes" if changed else "No",
                "Status": "Changed" if changed else "Coherent",
            }
        )
    return rows


def _render_manual_override_governance_header(
    governance_status: dict | None = None,
    cfg_final: dict | None = None,
    override_cfg: dict | None = None,
) -> None:
    import pandas as pd

    gov = dict(governance_status or {})
    cfg = dict(cfg_final or {})
    override = dict(override_cfg or {})
    block_rows, changed_blocks = _resolve_block_rows(cfg, override)
    parameter_rows = _resolve_parameter_rows(cfg, override)
    warnings = _collect_visible_warnings(gov)
    status_label = _normalize_status_label(gov)
    score_display = _normalize_score_display(gov)

    st.markdown("### Manual override governance")
    st.caption(
        "Advanced controls remain available, but they are now read against the coherent base config for the current philosophy. "
        "This block shows what changed at the structural block level and whether the current manual state stretches the mother philosophy."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Override status", status_label)
    with c2:
        st.metric("Coherence score", score_display)
    with c3:
        st.metric("Changed block groups", changed_blocks)

    state_raw = str(gov.get("state", gov.get("status", gov.get("operational_status", ""))) or "").strip().lower()
    if state_raw in {"blocked", "dangerous", "discouraged"}:
        st.error("Manual state is outside the safe governance region.")
    elif state_raw in {"stretched", "auto_repair_available", "preview_failed"}:
        st.info("Manual state is usable, but a safer repaired version is available.")
    else:
        st.success("Manual state remains aligned with the coherent base for the current philosophy.")

    st.markdown("#### Block-level diff vs coherent base")
    st.dataframe(pd.DataFrame([{k: row[k] for k in ['Block', 'Role', 'Base', 'Current', 'Changed']} for row in block_rows]), use_container_width=True, hide_index=True)

    st.markdown("#### Visible warnings")
    if warnings:
        for item in warnings:
            st.warning(item)
    else:
        st.success("No visible governance warnings for the current manual state.")

    st.markdown("#### Governed block reading for current philosophy")
    st.dataframe(
        pd.DataFrame([{k: row[k] for k in ['Block', 'Role', 'Active', 'Current', 'Expected effect']} for row in block_rows]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Parameter-level diff vs coherent base")
    st.dataframe(pd.DataFrame(parameter_rows), use_container_width=True, hide_index=True)

    st.markdown("#### Parameter trade-off reading")
    changed_rows = [row for row in parameter_rows if row["Changed"] == "Yes"]
    if changed_rows:
        first = changed_rows[0]
        st.markdown(
            f"- **{first['Parameter']} = {first['Current']}** — manual override is pulling this away from the coherent base value."
        )
    else:
        st.markdown("- **No parameter drift detected** — manual values remain aligned with the coherent base.")

    st.markdown("#### Repair CTA")
    repair_patch = dict(gov.get("suggested_repair_patch", {}) or {})
    if not repair_patch:
        repairs = gov.get("repair_summary") or gov.get("repairs") or {}
        if isinstance(repairs, dict):
            repair_patch = dict(repairs.get("suggested_patch", {}) or {})
    if warnings:
        st.caption(warnings[0])
    if repair_patch:
        st.caption(f"{len(repair_patch)} structural adjustment available: {', '.join(list(repair_patch.keys())[:4])}.")
    else:
        st.caption("No explicit repair patch is available yet, but the button is shown now so the full top block is visible.")
    if st.button("Apply minimal coherence repair", key="step5_apply_minimal_coherence_repair", use_container_width=True):
        if repair_patch:
            repaired = dict(override)
            repaired.update(repair_patch)
            st.session_state["step5_post_run_advanced_cfg"] = repaired
            st.success("Minimal coherence repair applied to the override layer.")
            st.rerun()
        else:
            st.info("Repair CTA is visible now. Wiring is partial in this pass because the goal was to surface the block first.")
    st.caption(
        "Applies the smallest structural fix needed to restore alignment with your current philosophy. This does not change "
        "your overall strategy intent."
    )
    st.caption("Full recommendations available in Strategy Coherence.")
    st.caption("Simple mode is currently syncing the affected manual controls below from the semantic resolve.")
    st.markdown("---")


def _render_manual_override_governance(
    governance_status: dict | None = None,
    cfg_final: dict | None = None,
    override_cfg: dict | None = None,
) -> None:
    _ = dict(override_cfg or {})
    _render_manual_override_governance_header(
        governance_status=governance_status,
        cfg_final=cfg_final,
        override_cfg=override_cfg,
    )


def render_advanced_params(
    cfg_final: dict | None = None,
    governance_status: dict | None = None,
    run_result: dict | None = None,
    *,
    post_run: bool = False,
):
    cfg_final = dict(cfg_final or {})
    _ = run_result

    if post_run:
        _render_manual_override_governance_header(
            governance_status=governance_status,
            cfg_final=cfg_final,
            override_cfg=dict(st.session_state.get("step5_post_run_advanced_cfg", {}) or {}),
        )
    else:
        st.markdown("### Advanced manual controls")
        st.caption("Expose deeper engine controls. Use cautiously.")

    st.caption("Auto-selection & validation")
    st.caption(
        "Meta-controls live first here: use them to validate or auto-select the signal contract before touching lower-level engine modules."
    )

    auto_signal_enabled = st.checkbox(
        "Enable automatic signal-contract selection by Rank IC",
        value=_bool_default("universe_auto_signal_select_enabled", False),
        key="universe_auto_signal_select_enabled",
        help="Runs multiple requested signal contracts on the same panel, measures ex-post cross-sectional Rank IC, and keeps the strongest candidate.",
    )
    requested_contracts = st.multiselect(
        "Requested signal contracts to explore",
        options=[
            "mu_sigma → base",
            "mu_sigma → fallback_enabled",
            "mu_sigma → multi_loss",
            "mu_sigma → multi_loss + fallback_enabled",
            "huber_mu → base",
            "huber_mu → multi_loss",
            "lambdarank_like → base",
            "lambdarank_like → multi_loss",
            "lambdarank_real → base",
            "lambdarank_real → multi_loss",
            "directional_classifier → base",
            "directional_classifier → multi_loss",
            "logistic_loss → base",
            "logistic_loss → multi_loss",
            "top_k_classifier → base",
            "top_k_classifier → multi_loss",
            "quantile_loss → base",
            "quantile_loss → multi_loss",
        ],
        default=_state_default(
            "universe_auto_signal_select_contracts",
            [f"{cfg_final.get('signal_mode', 'mu_sigma')} → base"],
        ),
        key="universe_auto_signal_select_contracts",
        help="Automatic selection evaluates explicit requested signal-contract variants and compares realised effective modes side by side.",
    )
    st.caption(
        "Automatic selection now evaluates explicit requested signal contracts. "
        "The comparison table reports requested contracts and realised effective modes side by side."
    )
    show_auto_report = st.checkbox(
        "Show full auto-selection report",
        value=_bool_default("universe_auto_signal_select_show_report", False),
        key="universe_auto_signal_select_show_report",
    )

    st.markdown("---")
    st.caption("Core engine definition")
    c1, c2 = st.columns(2)
    with c1:
        min_train = st.number_input(
            "min_train",
            min_value=24,
            max_value=240,
            step=12,
            value=_safe_int(_state_default("universe_micro_min_train", cfg_final.get("min_train", 120)), 120),
            key="universe_micro_min_train",
        )
        lookback_mu = st.number_input(
            "lookback_mu",
            min_value=3,
            max_value=60,
            step=1,
            value=_safe_int(_state_default("universe_micro_lookback_mu", cfg_final.get("lookback_mu", 12)), 12),
            key="universe_micro_lookback_mu",
        )
        lookback_sigma = st.number_input(
            "lookback_sigma",
            min_value=3,
            max_value=60,
            step=1,
            value=_safe_int(_state_default("universe_micro_lookback_sigma", cfg_final.get("lookback_sigma", 12)), 12),
            key="universe_micro_lookback_sigma",
        )
        sigma_power_alpha = st.number_input(
            "sigma_power_alpha",
            min_value=0.10,
            max_value=5.0,
            step=0.05,
            value=_safe_float(_state_default("universe_micro_sigma_power_alpha", cfg_final.get("sigma_power_alpha", 1.0)), 1.0),
            key="universe_micro_sigma_power_alpha",
        )
        sigma_floor = st.number_input(
            "sigma_floor",
            min_value=0.0001,
            max_value=0.25,
            step=0.005,
            value=_safe_float(_state_default("universe_micro_sigma_floor", cfg_final.get("sigma_floor", 0.02)), 0.02),
            key="universe_micro_sigma_floor",
            format="%.4f",
        )
    with c2:
        temperature = st.number_input(
            "temperature",
            min_value=0.10,
            max_value=5.0,
            step=0.05,
            value=_safe_float(_state_default("universe_micro_temperature", cfg_final.get("temperature", 1.0)), 1.0),
            key="universe_micro_temperature",
        )
        top_k = st.number_input(
            "top_k",
            min_value=1,
            max_value=max(1, _safe_int(st.session_state.get("universe_size", 25), 25)),
            step=1,
            value=max(1, _safe_int(_state_default("universe_micro_top_k", cfg_final.get("top_k", 10) or 10), 10)),
            key="universe_micro_top_k",
        )
        inertia = st.slider(
            "inertia",
            0.0,
            1.0,
            value=_safe_float(_state_default("universe_micro_inertia", cfg_final.get("inertia", 0.0)), 0.0),
            step=0.01,
            key="universe_micro_inertia",
        )
        deadband = st.checkbox(
            "deadband",
            value=_bool_default("universe_micro_deadband", bool(cfg_final.get("deadband", True))),
            key="universe_micro_deadband",
        )
        deadband_threshold = st.slider(
            "deadband_threshold",
            0.0,
            0.10,
            value=_safe_float(_state_default("universe_micro_deadband_threshold", cfg_final.get("deadband_threshold", 0.02)), 0.02),
            step=0.005,
            key="universe_micro_deadband_threshold",
        )

    st.markdown("---")
    st.caption("Risk shaping — per-asset caps")

    c_cap_top1, c_cap_top2 = st.columns(2)
    with c_cap_top1:
        asset_weight_cap_enabled = st.checkbox(
            "Enable asset_weight_cap",
            value=_bool_default(
                "universe_micro_asset_cap_enabled",
                cfg_final.get("asset_weight_cap") is not None,
            ),
            key="universe_micro_asset_cap_enabled",
        )
    with c_cap_top2:
        w_cap_enabled = st.checkbox(
            "Enable W_CAP",
            value=_bool_default(
                "universe_micro_w_cap_enabled",
                cfg_final.get("w_cap") is not None,
            ),
            key="universe_micro_w_cap_enabled",
        )

    st.markdown("#### Requested base cap transparency")
    asset_weight_cap_requested = "enabled" if asset_weight_cap_enabled else "disabled"
    w_cap_requested = "enabled" if w_cap_enabled else "disabled"
    if asset_weight_cap_enabled:
        effective_base_cap = "asset_weight_cap"
    elif w_cap_enabled:
        effective_base_cap = "w_cap"
    else:
        effective_base_cap = "disabled"
    st.caption(
        f"asset_weight_cap requested = {asset_weight_cap_requested} | "
        f"w_cap requested = {w_cap_requested} | "
        f"effective base cap = {effective_base_cap}"
    )
    if not asset_weight_cap_enabled and not w_cap_enabled:
        st.info(
            "No explicit base cap is active. The allocator may still be constrained later by adaptive caps, "
            "turnover logic, or other governance rules."
        )
    elif asset_weight_cap_enabled and w_cap_enabled:
        st.info(
            "Both base cap routes are active. Downstream logic will read both requests before the final effective cap is resolved."
        )
    elif asset_weight_cap_enabled:
        st.info("asset_weight_cap is the active requested base cap before later adaptive or governance adjustments.")
    else:
        st.info("w_cap is the active requested base cap before later adaptive or governance adjustments.")
    st.caption(
        "Final effective cap can become tighter after vol-, correlation-, dispersion-, or regime-dependent cap adjustments."
    )

    c1, c2 = st.columns(2)
    with c1:
        weight_shrink = st.slider(
            "weight_shrink",
            0.0,
            0.50,
            value=_safe_float(_state_default("universe_micro_weight_shrink", cfg_final.get("weight_shrink", 0.05)), 0.05),
            step=0.01,
            key="universe_micro_weight_shrink",
        )
        asset_weight_cap = st.slider(
            "asset_weight_cap",
            0.01,
            0.50,
            value=_safe_float(
                _state_default(
                    "universe_micro_asset_weight_cap",
                    cfg_final.get("asset_weight_cap", 0.15) or 0.15,
                ),
                0.15,
            ),
            step=0.01,
            key="universe_micro_asset_weight_cap",
            disabled=not asset_weight_cap_enabled,
        )
        w_cap = st.slider(
            "w_cap",
            0.01,
            0.50,
            value=_safe_float(
                _state_default(
                    "universe_micro_w_cap",
                    cfg_final.get("w_cap", cfg_final.get("asset_weight_cap", 0.15)) or 0.15,
                ),
                0.15,
            ),
            step=0.01,
            key="universe_micro_w_cap",
            disabled=not w_cap_enabled,
        )
    with c2:
        st.markdown("#### Adaptive caps (Phase 1 wiring)")
        caps_left, caps_right = st.columns(2)
        with caps_left:
            vol_dependent_cap_enabled = st.checkbox(
                "Enable vol-dependent cap",
                value=_bool_default("universe_micro_vol_dependent_cap_enabled", bool(cfg_final.get("vol_dependent_cap_enabled", False))),
                key="universe_micro_vol_dependent_cap_enabled",
            )
            vol_cap_threshold_low = st.number_input(
                "vol cap threshold low",
                min_value=0.0, max_value=1.0, step=0.01,
                value=_safe_float(_state_default("universe_micro_vol_cap_threshold_low", cfg_final.get("vol_cap_threshold_low", 0.03)), 0.03),
                key="universe_micro_vol_cap_threshold_low",
            )
            vol_cap_threshold_high = st.number_input(
                "vol cap threshold high",
                min_value=0.0, max_value=1.0, step=0.01,
                value=_safe_float(_state_default("universe_micro_vol_cap_threshold_high", cfg_final.get("vol_cap_threshold_high", 0.06)), 0.06),
                key="universe_micro_vol_cap_threshold_high",
            )
            vol_cap_low_mult = st.number_input(
                "vol cap low mult",
                min_value=0.0, max_value=3.0, step=0.01,
                value=_safe_float(_state_default("universe_micro_vol_cap_low_mult", cfg_final.get("vol_cap_low_mult", 1.15)), 1.15),
                key="universe_micro_vol_cap_low_mult",
            )
            vol_cap_high_mult = st.number_input(
                "vol cap high mult",
                min_value=0.0, max_value=3.0, step=0.01,
                value=_safe_float(_state_default("universe_micro_vol_cap_high_mult", cfg_final.get("vol_cap_high_mult", 0.80)), 0.80),
                key="universe_micro_vol_cap_high_mult",
            )
            vol_cap_min_enabled = st.checkbox(
                "Enable vol cap min",
                value=_bool_default("universe_micro_vol_cap_min_enabled", bool(cfg_final.get("vol_cap_min_enabled", False))),
                key="universe_micro_vol_cap_min_enabled",
            )
            vol_cap_max_enabled = st.checkbox(
                "Enable vol cap max",
                value=_bool_default("universe_micro_vol_cap_max_enabled", bool(cfg_final.get("vol_cap_max_enabled", False))),
                key="universe_micro_vol_cap_max_enabled",
            )
            dispersion_dependent_cap_enabled = st.checkbox(
                "Enable dispersion-dependent cap",
                value=_bool_default("universe_micro_dispersion_dependent_cap_enabled", bool(cfg_final.get("dispersion_dependent_cap_enabled", False))),
                key="universe_micro_dispersion_dependent_cap_enabled",
            )
            disp_cap_threshold_low = st.number_input(
                "disp cap threshold low",
                min_value=0.0, max_value=1.0, step=0.01,
                value=_safe_float(_state_default("universe_micro_disp_cap_threshold_low", cfg_final.get("disp_cap_threshold_low", 0.10)), 0.10),
                key="universe_micro_disp_cap_threshold_low",
            )
            disp_cap_threshold_high = st.number_input(
                "disp cap threshold high",
                min_value=0.0, max_value=1.0, step=0.01,
                value=_safe_float(_state_default("universe_micro_disp_cap_threshold_high", cfg_final.get("disp_cap_threshold_high", 0.30)), 0.30),
                key="universe_micro_disp_cap_threshold_high",
            )
            disp_cap_low_mult = st.number_input(
                "disp cap low mult",
                min_value=0.0, max_value=3.0, step=0.01,
                value=_safe_float(_state_default("universe_micro_disp_cap_low_mult", cfg_final.get("disp_cap_low_mult", 1.10)), 1.10),
                key="universe_micro_disp_cap_low_mult",
            )
            disp_cap_high_mult = st.number_input(
                "disp cap high mult",
                min_value=0.0, max_value=3.0, step=0.01,
                value=_safe_float(_state_default("universe_micro_disp_cap_high_mult", cfg_final.get("disp_cap_high_mult", 0.80)), 0.80),
                key="universe_micro_disp_cap_high_mult",
            )
            disp_cap_min_enabled = st.checkbox(
                "Enable disp cap min",
                value=_bool_default("universe_micro_disp_cap_min_enabled", bool(cfg_final.get("disp_cap_min_enabled", False))),
                key="universe_micro_disp_cap_min_enabled",
            )
            disp_cap_max_enabled = st.checkbox(
                "Enable disp cap max",
                value=_bool_default("universe_micro_disp_cap_max_enabled", bool(cfg_final.get("disp_cap_max_enabled", False))),
                key="universe_micro_disp_cap_max_enabled",
            )
        with caps_right:
            corr_dependent_cap_enabled = st.checkbox(
                "Enable corr-dependent cap",
                value=_bool_default("universe_micro_corr_dependent_cap_enabled", bool(cfg_final.get("corr_dependent_cap_enabled", False))),
                key="universe_micro_corr_dependent_cap_enabled",
            )
            corr_cap_threshold_low = st.number_input(
                "corr cap threshold low",
                min_value=0.0, max_value=1.0, step=0.01,
                value=_safe_float(_state_default("universe_micro_corr_cap_threshold_low", cfg_final.get("corr_cap_threshold_low", 0.20)), 0.20),
                key="universe_micro_corr_cap_threshold_low",
            )
            corr_cap_threshold_high = st.number_input(
                "corr cap threshold high",
                min_value=0.0, max_value=1.0, step=0.01,
                value=_safe_float(_state_default("universe_micro_corr_cap_threshold_high", cfg_final.get("corr_cap_threshold_high", 0.60)), 0.60),
                key="universe_micro_corr_cap_threshold_high",
            )
            corr_cap_low_mult = st.number_input(
                "corr cap low mult",
                min_value=0.0, max_value=3.0, step=0.01,
                value=_safe_float(_state_default("universe_micro_corr_cap_low_mult", cfg_final.get("corr_cap_low_mult", 1.10)), 1.10),
                key="universe_micro_corr_cap_low_mult",
            )
            corr_cap_high_mult = st.number_input(
                "corr cap high mult",
                min_value=0.0, max_value=3.0, step=0.01,
                value=_safe_float(_state_default("universe_micro_corr_cap_high_mult", cfg_final.get("corr_cap_high_mult", 0.80)), 0.80),
                key="universe_micro_corr_cap_high_mult",
            )
            corr_cap_min_enabled = st.checkbox(
                "Enable corr cap min",
                value=_bool_default("universe_micro_corr_cap_min_enabled", bool(cfg_final.get("corr_cap_min_enabled", False))),
                key="universe_micro_corr_cap_min_enabled",
            )
            corr_cap_max_enabled = st.checkbox(
                "Enable corr cap max",
                value=_bool_default("universe_micro_corr_cap_max_enabled", bool(cfg_final.get("corr_cap_max_enabled", False))),
                key="universe_micro_corr_cap_max_enabled",
            )
            regime_dependent_cap_enabled = st.checkbox(
                "Enable regime-dependent cap",
                value=_bool_default("universe_micro_regime_dependent_cap_enabled", bool(cfg_final.get("regime_dependent_cap_enabled", False))),
                key="universe_micro_regime_dependent_cap_enabled",
            )
            regime_cap_low_mult = st.number_input(
                "regime cap low mult",
                min_value=0.0, max_value=3.0, step=0.01,
                value=_safe_float(_state_default("universe_micro_regime_cap_low_mult", cfg_final.get("regime_cap_low_mult", 1.05)), 1.05),
                key="universe_micro_regime_cap_low_mult",
            )
            regime_cap_mid_mult = st.number_input(
                "regime cap mid mult",
                min_value=0.0, max_value=3.0, step=0.01,
                value=_safe_float(_state_default("universe_micro_regime_cap_mid_mult", cfg_final.get("regime_cap_mid_mult", 0.95)), 0.95),
                key="universe_micro_regime_cap_mid_mult",
            )
            regime_cap_high_mult = st.number_input(
                "regime cap high mult",
                min_value=0.0, max_value=3.0, step=0.01,
                value=_safe_float(_state_default("universe_micro_regime_cap_high_mult", cfg_final.get("regime_cap_high_mult", 0.80)), 0.80),
                key="universe_micro_regime_cap_high_mult",
            )
            regime_cap_min_enabled = st.checkbox(
                "Enable regime cap min",
                value=_bool_default("universe_micro_regime_cap_min_enabled", bool(cfg_final.get("regime_cap_min_enabled", False))),
                key="universe_micro_regime_cap_min_enabled",
            )
            regime_cap_max_enabled = st.checkbox(
                "Enable regime cap max",
                value=_bool_default("universe_micro_regime_cap_max_enabled", bool(cfg_final.get("regime_cap_max_enabled", False))),
                key="universe_micro_regime_cap_max_enabled",
            )

    st.markdown("---")
    st.caption("Turnover governance")
    c1, c2 = st.columns(2)
    with c1:
        turnover_penalty_strength = st.slider(
            "turnover_penalty_strength",
            0.0,
            3.0,
            value=_safe_float(
                _state_default("universe_micro_turnover_penalty_strength", cfg_final.get("turnover_penalty_strength", 0.0)),
                0.0,
            ),
            step=0.05,
            key="universe_micro_turnover_penalty_strength",
        )
        turnover_penalty_power = st.slider(
            "turnover_penalty_power",
            0.5,
            3.0,
            value=_safe_float(_state_default("universe_micro_turnover_penalty_power", cfg_final.get("turnover_penalty_power", 1.0)), 1.0),
            step=0.1,
            key="universe_micro_turnover_penalty_power",
        )
        turnover_penalty_target = st.slider(
            "turnover_penalty_target",
            0.0,
            1.0,
            value=_safe_float(_state_default("universe_micro_turnover_penalty_target", cfg_final.get("turnover_penalty_target", 0.20)), 0.20),
            step=0.01,
            key="universe_micro_turnover_penalty_target",
        )
    with c2:
        turnover_penalty_max_turnover = st.slider(
            "turnover_penalty_max_turnover",
            0.0,
            1.0,
            value=_safe_float(
                _state_default("universe_micro_turnover_penalty_max_turnover", cfg_final.get("turnover_penalty_max_turnover", 0.0) or 0.0),
                0.0,
            ),
            step=0.01,
            key="universe_micro_turnover_penalty_max_turnover",
        )
        turnover_constraint_max_turnover = st.slider(
            "turnover_constraint_max_turnover",
            0.0,
            1.0,
            value=_safe_float(
                _state_default(
                    "universe_micro_turnover_constraint_max_turnover",
                    cfg_final.get("turnover_constraint_max_turnover", 0.0) or 0.0,
                ),
                0.0,
            ),
            step=0.01,
            key="universe_micro_turnover_constraint_max_turnover",
        )

    st.markdown("---")
    st.caption("Risk model / covariance")
    c1, c2 = st.columns(2)
    with c1:
        covariance_mode_options = ["corr_sigma", "ewma_cov"]
        covariance_mode_value = str(_state_default("universe_micro_covariance_mode", cfg_final.get("covariance_mode", "ewma_cov")))
        covariance_mode = st.selectbox(
            "covariance_mode",
            options=covariance_mode_options,
            index=covariance_mode_options.index(covariance_mode_value) if covariance_mode_value in covariance_mode_options else 1,
            key="universe_micro_covariance_mode",
        )
        ewma_sigma = st.checkbox(
            "ewma_sigma",
            value=_bool_default("universe_micro_ewma_sigma", bool(cfg_final.get("ewma_sigma", True))),
            key="universe_micro_ewma_sigma",
        )
        ewma_halflife = st.number_input(
            "ewma_halflife",
            min_value=2,
            max_value=24,
            step=1,
            value=_safe_int(_state_default("universe_micro_ewma_halflife", cfg_final.get("ewma_halflife", 6)), 6),
            key="universe_micro_ewma_halflife",
        )
        covariance_shrink_to_diagonal = st.slider(
            "covariance_shrink_to_diagonal",
            0.0,
            1.0,
            value=_safe_float(
                _state_default("universe_micro_covariance_shrink_to_diagonal", cfg_final.get("covariance_shrink_to_diagonal", 0.10)),
                0.10,
            ),
            step=0.01,
            key="universe_micro_covariance_shrink_to_diagonal",
        )
    with c2:
        correlation_shrink_to_identity = st.slider(
            "correlation_shrink_to_identity",
            0.0,
            1.0,
            value=_safe_float(
                _state_default(
                    "universe_micro_correlation_shrink_to_identity",
                    cfg_final.get("correlation_shrink_to_identity", 0.10),
                ),
                0.10,
            ),
            step=0.01,
            key="universe_micro_correlation_shrink_to_identity",
        )
        regime_dependent_covariance = st.checkbox(
            "regime_dependent_covariance",
            value=_bool_default(
                "universe_micro_regime_dependent_covariance",
                bool(cfg_final.get("regime_dependent_covariance", True)),
            ),
            key="universe_micro_regime_dependent_covariance",
        )
        correlation_lookback = st.number_input(
            "correlation_lookback",
            min_value=3,
            max_value=60,
            step=1,
            value=_safe_int(_state_default("universe_micro_correlation_lookback", cfg_final.get("correlation_lookback", 24)), 24),
            key="universe_micro_correlation_lookback",
        )
        covariance_jitter = st.number_input(
            "covariance_jitter",
            min_value=0.0,
            max_value=0.01,
            step=0.000001,
            value=_safe_float(_state_default("universe_micro_covariance_jitter", cfg_final.get("covariance_jitter", 1e-8)), 1e-8),
            key="universe_micro_covariance_jitter",
            format="%.8f",
        )

    st.markdown("---")
    st.caption("Correlation-aware allocator")
    c1, c2 = st.columns(2)
    with c1:
        correlation_aware_allocation = st.checkbox(
            "correlation_aware_allocation",
            value=_bool_default(
                "universe_micro_correlation_aware_allocation",
                bool(cfg_final.get("correlation_aware_allocation", True)),
            ),
            key="universe_micro_correlation_aware_allocation",
        )
        allocator_options = ["score_penalty", "mean_variance_light", "risk_budget"]
        allocator_value = str(_state_default("universe_micro_correlation_allocator_method", cfg_final.get("correlation_allocator_method", "score_penalty")))
        correlation_allocator_method = st.selectbox(
            "correlation_allocator_method",
            options=allocator_options,
            index=allocator_options.index(allocator_value) if allocator_value in allocator_options else 0,
            key="universe_micro_correlation_allocator_method",
        )
        correlation_allocator_blend = st.slider(
            "correlation_allocator_blend",
            0.0,
            1.0,
            value=_safe_float(
                _state_default("universe_micro_correlation_allocator_blend", cfg_final.get("correlation_allocator_blend", 0.35)),
                0.35,
            ),
            step=0.01,
            key="universe_micro_correlation_allocator_blend",
        )
    with c2:
        correlation_penalty_strength = st.slider(
            "correlation_penalty_strength",
            0.0,
            5.0,
            value=_safe_float(
                _state_default("universe_micro_correlation_penalty_strength", cfg_final.get("correlation_penalty_strength", 1.0)),
                1.0,
            ),
            step=0.05,
            key="universe_micro_correlation_penalty_strength",
        )
        mean_variance_risk_aversion = st.slider(
            "mean_variance_risk_aversion",
            0.0,
            10.0,
            value=_safe_float(
                _state_default("universe_micro_mean_variance_risk_aversion", cfg_final.get("mean_variance_risk_aversion", 4.0)),
                4.0,
            ),
            step=0.1,
            key="universe_micro_mean_variance_risk_aversion",
        )
        risk_budget_strength = st.slider(
            "risk_budget_strength",
            0.0,
            2.0,
            value=_safe_float(_state_default("universe_micro_risk_budget_strength", cfg_final.get("risk_budget_strength", 0.5)), 0.5),
            step=0.05,
            key="universe_micro_risk_budget_strength",
        )

    st.markdown("---")
    st.caption("Vol targeting / regime derisk")
    c1, c2 = st.columns(2)
    with c1:
        vol_targeting = st.checkbox(
            "vol_targeting",
            value=_bool_default("universe_micro_vol_targeting", bool(cfg_final.get("vol_targeting", True))),
            key="universe_micro_vol_targeting",
        )
        covariance_aware_vol_targeting = st.checkbox(
            "covariance_aware_vol_targeting",
            value=_bool_default(
                "universe_micro_covariance_aware_vol_targeting",
                bool(cfg_final.get("covariance_aware_vol_targeting", True)),
            ),
            key="universe_micro_covariance_aware_vol_targeting",
        )
        target_portfolio_vol_monthly = st.slider(
            "target_portfolio_vol_monthly",
            0.005,
            0.10,
            value=_safe_float(
                _state_default(
                    "universe_micro_target_portfolio_vol_monthly",
                    cfg_final.get("target_portfolio_vol_monthly", 0.04),
                ),
                0.04,
            ),
            step=0.001,
            key="universe_micro_target_portfolio_vol_monthly",
        )
    with c2:
        regime_derisk_low = st.slider(
            "regime_derisk_low",
            0.0,
            1.25,
            value=_safe_float(_state_default("universe_micro_regime_derisk_low", cfg_final.get("regime_derisk_low", 1.00)), 1.00),
            step=0.01,
            key="universe_micro_regime_derisk_low",
        )
        regime_derisk_mid = st.slider(
            "regime_derisk_mid",
            0.0,
            1.25,
            value=_safe_float(_state_default("universe_micro_regime_derisk_mid", cfg_final.get("regime_derisk_mid", 0.95)), 0.95),
            step=0.01,
            key="universe_micro_regime_derisk_mid",
        )
        regime_derisk_high = st.slider(
            "regime_derisk_high",
            0.0,
            1.25,
            value=_safe_float(_state_default("universe_micro_regime_derisk_high", cfg_final.get("regime_derisk_high", 0.80)), 0.80),
            step=0.01,
            key="universe_micro_regime_derisk_high",
        )

    st.markdown("---")
    st.caption("Cost model")
    c1, c2 = st.columns(2)
    with c1:
        cost_model_enabled = st.checkbox(
            "cost_model_enabled",
            value=_bool_default("universe_micro_cost_model_enabled", bool(cfg_final.get("cost_model_enabled", False))),
            key="universe_micro_cost_model_enabled",
        )
        transaction_cost_commission_bps = st.number_input(
            "transaction_cost_commission_bps",
            min_value=0.0,
            max_value=100.0,
            step=0.5,
            value=_safe_float(
                _state_default(
                    "universe_micro_transaction_cost_commission_bps",
                    cfg_final.get("transaction_cost_commission_bps", 0.0),
                ),
                0.0,
            ),
            key="universe_micro_transaction_cost_commission_bps",
        )
        transaction_cost_slippage_bps = st.number_input(
            "transaction_cost_slippage_bps",
            min_value=0.0,
            max_value=100.0,
            step=0.5,
            value=_safe_float(
                _state_default(
                    "universe_micro_transaction_cost_slippage_bps",
                    cfg_final.get("transaction_cost_slippage_bps", 0.0),
                ),
                0.0,
            ),
            key="universe_micro_transaction_cost_slippage_bps",
        )
    with c2:
        transaction_cost_spread_bps = st.number_input(
            "transaction_cost_spread_bps",
            min_value=0.0,
            max_value=100.0,
            step=0.5,
            value=_safe_float(
                _state_default("universe_micro_transaction_cost_spread_bps", cfg_final.get("transaction_cost_spread_bps", 0.0)),
                0.0,
            ),
            key="universe_micro_transaction_cost_spread_bps",
        )
        transaction_cost_market_impact_bps = st.number_input(
            "transaction_cost_market_impact_bps",
            min_value=0.0,
            max_value=100.0,
            step=0.5,
            value=_safe_float(
                _state_default(
                    "universe_micro_transaction_cost_market_impact_bps",
                    cfg_final.get("transaction_cost_market_impact_bps", 0.0),
                ),
                0.0,
            ),
            key="universe_micro_transaction_cost_market_impact_bps",
        )
        holding_cost_annual_bps = st.number_input(
            "holding_cost_annual_bps",
            min_value=0.0,
            max_value=100.0,
            step=0.5,
            value=_safe_float(_state_default("universe_micro_holding_cost_annual_bps", cfg_final.get("holding_cost_annual_bps", 0.0)), 0.0),
            key="universe_micro_holding_cost_annual_bps",
        )

    st.markdown("---")
    st.caption("Factor model / overlays")
    c1, c2 = st.columns(2)
    with c1:
        factor_model_active = st.checkbox(
            "factor_model_active",
            value=_bool_default("universe_micro_factor_model_active", bool(cfg_final.get("factor_model_active", False))),
            key="universe_micro_factor_model_active",
        )
        factor_model_n_factors = st.number_input(
            "factor_model_n_factors",
            min_value=1,
            max_value=12,
            step=1,
            value=_safe_int(_state_default("universe_micro_factor_model_n_factors", cfg_final.get("factor_model_n_factors", 3)), 3),
            key="universe_micro_factor_model_n_factors",
        )
        factor_model_mu_blend = st.slider(
            "factor_model_mu_blend",
            0.0,
            1.0,
            value=_safe_float(_state_default("universe_micro_factor_model_mu_blend", cfg_final.get("factor_model_mu_blend", 0.35)), 0.35),
            step=0.01,
            key="universe_micro_factor_model_mu_blend",
        )
    with c2:
        factor_covariance_active = st.checkbox(
            "factor_covariance_active",
            value=_bool_default("universe_micro_factor_covariance_active", bool(cfg_final.get("factor_covariance_active", False))),
            key="universe_micro_factor_covariance_active",
        )
        factor_covariance_n_factors = st.number_input(
            "factor_covariance_n_factors",
            min_value=1,
            max_value=12,
            step=1,
            value=_safe_int(_state_default("universe_micro_factor_covariance_n_factors", cfg_final.get("factor_covariance_n_factors", 3)), 3),
            key="universe_micro_factor_covariance_n_factors",
        )
        factor_covariance_blend = st.slider(
            "factor_covariance_blend",
            0.0,
            1.0,
            value=_safe_float(_state_default("universe_micro_factor_covariance_blend", cfg_final.get("factor_covariance_blend", 0.50)), 0.50),
            step=0.01,
            key="universe_micro_factor_covariance_blend",
        )

    st.markdown("---")
    st.caption("Probabilistic / overlay advanced")
    c1, c2 = st.columns(2)
    with c1:
        probabilistic_options = [
            "none",
            "historical",
            "historical_by_regime",
            "historical_by_features",
            "parametric_feature_aware",
            "knn_historical",
            "feature_bucketed_historical",
            "quantile_regression",
            "hybrid",
        ]
        probabilistic_value = str(_state_default("universe_micro_probabilistic_mode", cfg_final.get("probabilistic_mode", "none")))
        probabilistic_mode = st.selectbox(
            "probabilistic_mode",
            options=probabilistic_options,
            index=probabilistic_options.index(probabilistic_value) if probabilistic_value in probabilistic_options else 0,
            key="universe_micro_probabilistic_mode",
        )
        probabilistic_overlay_strength = st.slider(
            "probabilistic_overlay_strength",
            0.0,
            2.0,
            value=_safe_float(
                _state_default(
                    "universe_micro_probabilistic_overlay_strength",
                    cfg_final.get("probabilistic_overlay_strength", 1.0),
                ),
                1.0,
            ),
            step=0.05,
            key="universe_micro_probabilistic_overlay_strength",
        )
        probabilistic_overlay_blend = st.slider(
            "probabilistic_overlay_blend",
            0.0,
            1.0,
            value=_safe_float(_state_default("universe_micro_probabilistic_overlay_blend", cfg_final.get("probabilistic_overlay_blend", 1.0)), 1.0),
            step=0.01,
            key="universe_micro_probabilistic_overlay_blend",
        )
    with c2:
        probabilistic_q_low = st.slider(
            "probabilistic_q_low",
            0.01,
            0.49,
            value=_safe_float(_state_default("universe_micro_probabilistic_q_low", cfg_final.get("probabilistic_q_low", 0.25)), 0.25),
            step=0.01,
            key="universe_micro_probabilistic_q_low",
        )
        probabilistic_q_high = st.slider(
            "probabilistic_q_high",
            0.51,
            0.99,
            value=_safe_float(_state_default("universe_micro_probabilistic_q_high", cfg_final.get("probabilistic_q_high", 0.75)), 0.75),
            step=0.01,
            key="universe_micro_probabilistic_q_high",
        )
        probabilistic_min_obs = st.number_input(
            "probabilistic_min_obs",
            min_value=3,
            max_value=120,
            step=1,
            value=_safe_int(_state_default("universe_micro_probabilistic_min_obs", cfg_final.get("probabilistic_min_obs", 12)), 12),
            key="universe_micro_probabilistic_min_obs",
        )

    st.markdown("---")
    st.caption("Feature-conditioned mu")
    c1, c2 = st.columns(2)
    with c1:
        feature_mu_enabled = st.checkbox(
            "feature_mu_enabled",
            value=_bool_default("universe_micro_feature_mu_enabled", bool(cfg_final.get("feature_mu_enabled", True))),
            key="universe_micro_feature_mu_enabled",
        )
        feature_mu_blend = st.slider(
            "feature_mu_blend",
            0.0,
            1.0,
            value=_safe_float(_state_default("universe_micro_feature_mu_blend", cfg_final.get("feature_mu_blend", 0.25)), 0.25),
            step=0.01,
            key="universe_micro_feature_mu_blend",
        )
    with c2:
        feature_mu_k = st.number_input(
            "feature_mu_k",
            min_value=3,
            max_value=120,
            step=1,
            value=_safe_int(_state_default("universe_micro_feature_mu_k", cfg_final.get("feature_mu_k", 24)), 24),
            key="universe_micro_feature_mu_k",
        )
        feature_mu_min_obs = st.number_input(
            "feature_mu_min_obs",
            min_value=3,
            max_value=120,
            step=1,
            value=_safe_int(_state_default("universe_micro_feature_mu_min_obs", cfg_final.get("feature_mu_min_obs", 12)), 12),
            key="universe_micro_feature_mu_min_obs",
        )

    st.markdown("---")
    st.caption("Dispersion governance")
    c1, c2 = st.columns(2)
    with c1:
        dispersion_gate = st.checkbox(
            "dispersion_gate",
            value=_bool_default("universe_micro_dispersion_gate", bool(cfg_final.get("dispersion_gate", True))),
            key="universe_micro_dispersion_gate",
        )
        dispersion_gate_threshold = st.slider(
            "dispersion_gate_threshold",
            0.0,
            1.0,
            value=_safe_float(_state_default("universe_micro_dispersion_gate_threshold", cfg_final.get("dispersion_gate_threshold", 0.10)), 0.10),
            step=0.01,
            key="universe_micro_dispersion_gate_threshold",
        )
        dispersion_sigma_enabled = st.checkbox(
            "dispersion_sigma_enabled",
            value=_bool_default("universe_micro_dispersion_sigma_enabled", bool(cfg_final.get("dispersion_sigma_enabled", False))),
            key="universe_micro_dispersion_sigma_enabled",
        )
    with c2:
        dispersion_risk_model_enabled = st.checkbox(
            "dispersion_risk_model_enabled",
            value=_bool_default(
                "universe_micro_dispersion_risk_model_enabled",
                bool(cfg_final.get("dispersion_risk_model_enabled", False)),
            ),
            key="universe_micro_dispersion_risk_model_enabled",
        )
        dispersion_top_k_enabled = st.checkbox(
            "dispersion_top_k_enabled",
            value=_bool_default("universe_micro_dispersion_top_k_enabled", bool(cfg_final.get("dispersion_top_k_enabled", False))),
            key="universe_micro_dispersion_top_k_enabled",
        )
        dispersion_vol_target_enabled = st.checkbox(
            "dispersion_vol_target_enabled",
            value=_bool_default(
                "universe_micro_dispersion_vol_target_enabled",
                bool(cfg_final.get("dispersion_vol_target_enabled", False)),
            ),
            key="universe_micro_dispersion_vol_target_enabled",
        )

    st.markdown("---")
    st.caption("Signal model extensions")
    c1, c2 = st.columns(2)
    with c1:
        signal_options = [
            "mu_sigma",
            "huber_mu",
            "lambdarank_like",
            "lambdarank_real",
            "directional_classifier",
            "logistic_loss",
            "top_k_classifier",
            "quantile_loss",
        ]
        signal_value = str(_state_default("universe_micro_signal_mode", cfg_final.get("signal_mode", "mu_sigma")))
        signal_mode = st.selectbox(
            "signal_mode",
            options=signal_options,
            index=signal_options.index(signal_value) if signal_value in signal_options else 0,
            key="universe_micro_signal_mode",
        )
        signal_score_blend = st.slider(
            "signal_score_blend",
            0.0,
            2.0,
            value=_safe_float(_state_default("universe_micro_signal_score_blend", cfg_final.get("signal_score_blend", 1.0)), 1.0),
            step=0.05,
            key="universe_micro_signal_score_blend",
        )
    with c2:
        huber_delta = st.slider(
            "huber_delta",
            0.1,
            5.0,
            value=_safe_float(_state_default("universe_micro_huber_delta", cfg_final.get("huber_delta", 1.0)), 1.0),
            step=0.1,
            key="universe_micro_huber_delta",
        )
        lambdarank_temperature = st.slider(
            "lambdarank_temperature",
            0.1,
            5.0,
            value=_safe_float(_state_default("universe_micro_lambdarank_temperature", cfg_final.get("lambdarank_temperature", 1.0)), 1.0),
            step=0.1,
            key="universe_micro_lambdarank_temperature",
        )

    st.markdown("---")
    st.caption("Regime-dependent universe")
    c1, c2 = st.columns(2)
    with c1:
        regime_dependent_universe_enabled = st.checkbox(
            "regime_dependent_universe_enabled",
            value=_bool_default(
                "universe_micro_regime_dependent_universe_enabled",
                bool(cfg_final.get("regime_dependent_universe_enabled", False)),
            ),
            key="universe_micro_regime_dependent_universe_enabled",
        )
        metric_options = ["mu", "score", "mu_over_sigma"]
        metric_value = str(
            _state_default(
                "universe_micro_regime_universe_selection_metric",
                cfg_final.get("regime_universe_selection_metric", "mu_over_sigma"),
            )
        )
        regime_universe_selection_metric = st.selectbox(
            "regime_universe_selection_metric",
            options=metric_options,
            index=metric_options.index(metric_value) if metric_value in metric_options else 2,
            key="universe_micro_regime_universe_selection_metric",
        )
    with c2:
        regime_universe_keep_frac_low = st.slider(
            "regime_universe_keep_frac_low",
            0.0,
            1.0,
            value=_safe_float(
                _state_default(
                    "universe_micro_regime_universe_keep_frac_low",
                    cfg_final.get("regime_universe_keep_frac_low", 1.00),
                ),
                1.00,
            ),
            step=0.01,
            key="universe_micro_regime_universe_keep_frac_low",
        )
        regime_universe_keep_frac_high = st.slider(
            "regime_universe_keep_frac_high",
            0.0,
            1.0,
            value=_safe_float(
                _state_default(
                    "universe_micro_regime_universe_keep_frac_high",
                    cfg_final.get("regime_universe_keep_frac_high", 0.60),
                ),
                0.60,
            ),
            step=0.01,
            key="universe_micro_regime_universe_keep_frac_high",
        )

    st.markdown("---")
    st.caption("Research / tuning")
    c1, c2 = st.columns(2)
    with c1:
        pure_cs_baseline = st.checkbox(
            "pure_cs_baseline",
            value=_bool_default("universe_micro_pure_cs_baseline", bool(cfg_final.get("pure_cs_baseline", False))),
            key="universe_micro_pure_cs_baseline",
        )
        selection_options = ["fixed_composite_score", "balanced_compromise", "knee_point", "weighted_hypervolume"]
        selection_value = str(_state_default("universe_micro_selection_policy", cfg_final.get("selection_policy", "fixed_composite_score")))
        selection_policy = st.selectbox(
            "selection_policy",
            options=selection_options,
            index=selection_options.index(selection_value) if selection_value in selection_options else 0,
            key="universe_micro_selection_policy",
        )
    with c2:
        profile_options = ["balanced", "growth", "defensive"]
        profile_value = str(_state_default("universe_micro_composite_profile", cfg_final.get("composite_profile", "balanced") or "balanced"))
        composite_profile = st.selectbox(
            "composite_profile",
            options=profile_options,
            index=profile_options.index(profile_value) if profile_value in profile_options else 0,
            key="universe_micro_composite_profile",
        )
        show_activity_audit = st.checkbox(
            "Show engine-parameter activity audit after rerun",
            value=_bool_default("universe_micro_show_activity_audit", False),
            key="universe_micro_show_activity_audit",
        )

    return {
        "auto_signal_select_enabled": auto_signal_enabled,
        "auto_signal_select_contracts": requested_contracts,
        "auto_signal_select_show_report": show_auto_report,
        "min_train": int(min_train),
        "lookback_mu": int(lookback_mu),
        "lookback_sigma": int(lookback_sigma),
        "sigma_power_alpha": float(sigma_power_alpha),
        "sigma_floor": float(sigma_floor),
        "temperature": float(temperature),
        "top_k": int(top_k),
        "inertia": float(inertia),
        "deadband": bool(deadband),
        "deadband_threshold": float(deadband_threshold),
        "weight_shrink": float(weight_shrink),
        "asset_weight_cap": float(asset_weight_cap) if asset_weight_cap_enabled else None,
        "w_cap": float(w_cap) if asset_weight_cap_enabled else None,
        "vol_dependent_cap_enabled": bool(vol_dependent_cap_enabled),
        "vol_cap_threshold_low": float(vol_cap_threshold_low),
        "vol_cap_threshold_high": float(vol_cap_threshold_high),
        "vol_cap_low_mult": float(vol_cap_low_mult),
        "vol_cap_high_mult": float(vol_cap_high_mult),
        "vol_cap_min_enabled": bool(vol_cap_min_enabled),
        "vol_cap_max_enabled": bool(vol_cap_max_enabled),
        "corr_dependent_cap_enabled": bool(corr_dependent_cap_enabled),
        "corr_cap_threshold_low": float(corr_cap_threshold_low),
        "corr_cap_threshold_high": float(corr_cap_threshold_high),
        "corr_cap_low_mult": float(corr_cap_low_mult),
        "corr_cap_high_mult": float(corr_cap_high_mult),
        "corr_cap_min_enabled": bool(corr_cap_min_enabled),
        "corr_cap_max_enabled": bool(corr_cap_max_enabled),
        "dispersion_dependent_cap_enabled": bool(dispersion_dependent_cap_enabled),
        "disp_cap_threshold_low": float(disp_cap_threshold_low),
        "disp_cap_threshold_high": float(disp_cap_threshold_high),
        "disp_cap_low_mult": float(disp_cap_low_mult),
        "disp_cap_high_mult": float(disp_cap_high_mult),
        "disp_cap_min_enabled": bool(disp_cap_min_enabled),
        "disp_cap_max_enabled": bool(disp_cap_max_enabled),
        "regime_dependent_cap_enabled": bool(regime_dependent_cap_enabled),
        "regime_cap_low_mult": float(regime_cap_low_mult),
        "regime_cap_mid_mult": float(regime_cap_mid_mult),
        "regime_cap_high_mult": float(regime_cap_high_mult),
        "regime_cap_min_enabled": bool(regime_cap_min_enabled),
        "regime_cap_max_enabled": bool(regime_cap_max_enabled),
        "turnover_penalty_strength": float(turnover_penalty_strength),
        "turnover_penalty_power": float(turnover_penalty_power),
        "turnover_penalty_target": float(turnover_penalty_target),
        "turnover_penalty_max_turnover": float(turnover_penalty_max_turnover) if turnover_penalty_max_turnover > 0 else None,
        "turnover_constraint_max_turnover": float(turnover_constraint_max_turnover) if turnover_constraint_max_turnover > 0 else None,
        "covariance_mode": str(covariance_mode),
        "ewma_sigma": bool(ewma_sigma),
        "ewma_halflife": int(ewma_halflife),
        "covariance_shrink_to_diagonal": float(covariance_shrink_to_diagonal),
        "correlation_shrink_to_identity": float(correlation_shrink_to_identity),
        "regime_dependent_covariance": bool(regime_dependent_covariance),
        "correlation_lookback": int(correlation_lookback),
        "covariance_jitter": float(covariance_jitter),
        "correlation_aware_allocation": bool(correlation_aware_allocation),
        "correlation_allocator_method": str(correlation_allocator_method),
        "correlation_allocator_blend": float(correlation_allocator_blend),
        "correlation_penalty_strength": float(correlation_penalty_strength),
        "mean_variance_risk_aversion": float(mean_variance_risk_aversion),
        "risk_budget_strength": float(risk_budget_strength),
        "vol_targeting": bool(vol_targeting),
        "covariance_aware_vol_targeting": bool(covariance_aware_vol_targeting),
        "target_portfolio_vol_monthly": float(target_portfolio_vol_monthly),
        "regime_derisk_low": float(regime_derisk_low),
        "regime_derisk_mid": float(regime_derisk_mid),
        "regime_derisk_high": float(regime_derisk_high),
        "cost_model_enabled": bool(cost_model_enabled),
        "transaction_cost_commission_bps": float(transaction_cost_commission_bps),
        "transaction_cost_slippage_bps": float(transaction_cost_slippage_bps),
        "transaction_cost_spread_bps": float(transaction_cost_spread_bps),
        "transaction_cost_market_impact_bps": float(transaction_cost_market_impact_bps),
        "holding_cost_annual_bps": float(holding_cost_annual_bps),
        "factor_model_active": bool(factor_model_active),
        "factor_model_n_factors": int(factor_model_n_factors),
        "factor_model_mu_blend": float(factor_model_mu_blend),
        "factor_covariance_active": bool(factor_covariance_active),
        "factor_covariance_n_factors": int(factor_covariance_n_factors),
        "factor_covariance_blend": float(factor_covariance_blend),
        "probabilistic_mode": str(probabilistic_mode),
        "probabilistic_overlay_strength": float(probabilistic_overlay_strength),
        "probabilistic_overlay_blend": float(probabilistic_overlay_blend),
        "probabilistic_q_low": float(probabilistic_q_low),
        "probabilistic_q_high": float(probabilistic_q_high),
        "probabilistic_min_obs": int(probabilistic_min_obs),
        "feature_mu_enabled": bool(feature_mu_enabled),
        "feature_mu_blend": float(feature_mu_blend),
        "feature_mu_k": int(feature_mu_k),
        "feature_mu_min_obs": int(feature_mu_min_obs),
        "dispersion_gate": bool(dispersion_gate),
        "dispersion_gate_threshold": float(dispersion_gate_threshold),
        "dispersion_sigma_enabled": bool(dispersion_sigma_enabled),
        "dispersion_risk_model_enabled": bool(dispersion_risk_model_enabled),
        "dispersion_top_k_enabled": bool(dispersion_top_k_enabled),
        "dispersion_vol_target_enabled": bool(dispersion_vol_target_enabled),
        "signal_mode": str(signal_mode),
        "signal_score_blend": float(signal_score_blend),
        "huber_delta": float(huber_delta),
        "lambdarank_temperature": float(lambdarank_temperature),
        "regime_dependent_universe_enabled": bool(regime_dependent_universe_enabled),
        "regime_universe_selection_metric": str(regime_universe_selection_metric),
        "regime_universe_keep_frac_low": float(regime_universe_keep_frac_low),
        "regime_universe_keep_frac_high": float(regime_universe_keep_frac_high),
        "pure_cs_baseline": bool(pure_cs_baseline),
        "selection_policy": str(selection_policy),
        "composite_profile": str(composite_profile),
        "show_activity_audit": bool(show_activity_audit),
    }



def render_config_diff(base_cfg: dict | None = None, override_cfg: dict | None = None):
    st.markdown("### Config diff")
    base_map = dict(base_cfg or {})
    override_map = dict(override_cfg or {})
    changes = []
    for key, override_value in override_map.items():
        if override_value is None:
            continue
        base_value = base_map.get(key)
        if base_value != override_value:
            changes.append({"parameter": str(key), "base": base_value, "override": override_value})
    if not changes:
        st.success("No changes relative to the governed base config.")
        return []
    import pandas as pd
    df = pd.DataFrame(changes)
    st.dataframe(df, use_container_width=True, hide_index=True)
    return changes



# ------------------------------------------------------------------
# Compatibility helpers required by step5_workspace variants
# ------------------------------------------------------------------

def render_config_diff(base_cfg: dict | None = None, override_cfg: dict | None = None):
    base_cfg = dict(base_cfg or {})
    override_cfg = dict(override_cfg or {})
    rows = []
    for key in list(dict.fromkeys(list(base_cfg.keys()) + list(override_cfg.keys()))):
        base_value = base_cfg.get(key)
        current_value = override_cfg.get(key, base_value)
        if base_value != current_value:
            rows.append(
                {
                    "parameter": key,
                    "base": base_value,
                    "override": current_value,
                }
            )
    if not rows:
        st.caption("No config diff relative to the governed base.")
        return []
    import pandas as pd
    st.markdown("### Config diff")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    return rows


def render_override_impact_heatmap(base_cfg: dict | None = None, override_cfg: dict | None = None):
    base_cfg = dict(base_cfg or {})
    override_cfg = dict(override_cfg or {})
    rows = []
    for key in list(dict.fromkeys(list(base_cfg.keys()) + list(override_cfg.keys()))):
        base_value = base_cfg.get(key)
        current_value = override_cfg.get(key, base_value)
        if base_value != current_value:
            rows.append(
                {
                    "parameter": key,
                    "base": base_value,
                    "override": current_value,
                    "impact_hint": "changed",
                }
            )
    if rows:
        import pandas as pd
        st.markdown("### Override impact preview")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No override impact to preview.")
    return rows


def render_live_multiobjective_score(
    run_result: dict | None = None,
    preview_meta: dict | None = None,
    impact_rows=None,
    cfg_for_profile: dict | None = None,
):
    run_result = dict(run_result or {})
    preview_meta = dict(preview_meta or {})
    perf = dict(run_result.get("performance_summary", {}) or {})
    try:
        sharpe = float(perf.get("sharpe", 0.0) or 0.0)
    except Exception:
        sharpe = 0.0
    try:
        cagr = float(perf.get("cagr", 0.0) or 0.0)
    except Exception:
        cagr = 0.0
    try:
        maxdd = abs(float(perf.get("max_drawdown", 0.0) or 0.0))
    except Exception:
        maxdd = 0.0
    score = (1.0 * sharpe) + (0.5 * cagr) - (0.75 * maxdd)
    st.markdown("### Live multiobjective score")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Sharpe", f"{sharpe:.3f}")
    with c2:
        st.metric("CAGR", f"{cagr:.3f}")
    with c3:
        st.metric("Composite score", f"{score:.3f}")
    return {
        "sharpe": sharpe,
        "cagr": cagr,
        "max_drawdown": maxdd,
        "score": score,
    }


def render_safe_zone(preview_meta: dict | None = None):
    preview_meta = dict(preview_meta or {})
    status = str(preview_meta.get("status", "coherent") or "coherent").strip().lower()
    st.markdown("### Safe zone")
    if status in {"blocked", "dangerous", "discouraged"}:
        st.error("Danger zone: current override posture is not safely aligned.")
    elif status in {"stretched", "auto_repair_available"}:
        st.warning("Edge zone: current override posture is stretched.")
    else:
        st.success("Safe zone: current override posture looks coherent.")


def render_override_explanation(
    base_cfg: dict | None = None,
    preview_cfg: dict | None = None,
    run_result: dict | None = None,
    preview_meta: dict | None = None,
    philosophy: str | None = None,
):
    base_cfg = dict(base_cfg or {})
    preview_cfg = dict(preview_cfg or {})
    preview_meta = dict(preview_meta or {})
    philosophy = str(philosophy or "Balanced")
    changed = [k for k in preview_cfg.keys() if base_cfg.get(k) != preview_cfg.get(k)]
    st.markdown("### Override explanation")
    if changed:
        st.write(
            f"The current override layer changes **{len(changed)}** parameter(s) relative to the governed base "
            f"under the **{philosophy}** philosophy."
        )
    else:
        st.write(f"No manual parameter drift detected relative to the governed base under **{philosophy}**.")
    if preview_meta.get("warnings"):
        st.caption("Warnings are present for the current override state.")
    return {
        "changed_params": changed,
        "philosophy": philosophy,
    }


def render_local_autotune_controls():
    st.markdown("### Local auto-tune")
    c1, c2, c3 = st.columns(3)
    with c1:
        max_params = st.number_input("Max params", min_value=1, max_value=5, value=3, step=1, key="step5_local_autotune_max_params")
    with c2:
        step_scale = st.slider("Step scale", 0.02, 0.30, value=0.10, step=0.01, key="step5_local_autotune_step_scale")
    with c3:
        max_trials = st.number_input("Max trials", min_value=2, max_value=12, value=6, step=1, key="step5_local_autotune_max_trials")
    return {"max_params": int(max_params), "step_scale": float(step_scale), "max_trials": int(max_trials)}


def render_local_autotune_result(result: dict | None = None):
    result = dict(result or {})
    st.markdown("### Local auto-tune result")
    candidates_df = result.get("candidates_df")
    if candidates_df is not None and hasattr(candidates_df, "empty") and not candidates_df.empty:
        st.dataframe(candidates_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No local auto-tune candidates available.")


def render_local_pareto_controls():
    st.markdown("### Local Pareto")
    c1, c2 = st.columns(2)
    with c1:
        max_params = st.number_input("Pareto max params", min_value=1, max_value=5, value=3, step=1, key="step5_local_pareto_max_params")
    with c2:
        max_candidates = st.number_input("Pareto max candidates", min_value=4, max_value=20, value=10, step=1, key="step5_local_pareto_max_candidates")
    step_scale = st.slider("Pareto step scale", 0.02, 0.30, value=0.10, step=0.01, key="step5_local_pareto_step_scale")
    return {"max_params": int(max_params), "max_candidates": int(max_candidates), "step_scale": float(step_scale)}


def render_local_pareto_result(result: dict | None = None):
    result = dict(result or {})
    st.markdown("### Local Pareto result")
    frontier_df = result.get("frontier_df")
    if frontier_df is not None and hasattr(frontier_df, "empty") and not frontier_df.empty:
        st.dataframe(frontier_df, use_container_width=True, hide_index=True)
        ids = list(frontier_df["candidate_id"].astype(str))
        return st.selectbox("Select Pareto candidate to apply", options=ids, key="step5_local_pareto_selected_candidate_id")
    st.caption("No local Pareto frontier available.")
    return None
