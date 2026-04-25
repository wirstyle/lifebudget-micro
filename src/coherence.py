# coherence.py

from __future__ import annotations

from typing import Any, Dict, List, Optional


VALID_PHILOSOPHIES = ("Growth", "Balanced", "Defensive")
_VALID_OVERLAY_MODES = {
    'none',
    'historical',
    'historical_by_regime',
    'historical_by_features',
    'parametric_feature_aware',
    'knn_historical',
    'feature_bucketed_historical',
    'quantile_regression',
    'hybrid',
}
_VALID_SIGNAL_MODES = {
    'mu_sigma',
    'huber_mu',
    'lambdarank_like',
    'lambdarank_real',
    'directional_classifier',
    'logistic_loss',
    'top_k_classifier',
    'quantile_loss',
}
_VALID_UNIVERSE_BUCKETS = {'small', 'medium', 'large'}


def _safe_float(value: Any) -> float | None:
    try:
        v = float(value)
    except Exception:
        return None
    return v if v == v and v not in (float('inf'), float('-inf')) else None


def _coerce_mode_name(value: Any, default: str = 'none') -> str:
    raw = str(value or default).strip().lower()
    return raw or default


def _coerce_philosophy_name(value: Any) -> str:
    raw = str(value or 'Balanced').strip().capitalize()
    return raw if raw in VALID_PHILOSOPHIES else 'Balanced'


def _normalize_universe_bucket(universe: Any) -> str:
    if isinstance(universe, str):
        raw = universe.strip().lower()
        if raw in _VALID_UNIVERSE_BUCKETS:
            return raw
        try:
            universe = int(raw)
        except Exception:
            return 'medium'
    try:
        n_assets = int(universe)
    except Exception:
        return 'medium'
    if n_assets <= 15:
        return 'small'
    if n_assets <= 75:
        return 'medium'
    return 'large'


def _derive_covariance_model(cfg: Dict[str, Any]) -> str:
    explicit = _coerce_mode_name(cfg.get('covariance_model'), default='')
    if explicit:
        return explicit

    ewma_sigma = bool(cfg.get('ewma_sigma', False))
    correlation_penalty_strength = _safe_float(cfg.get('correlation_penalty_strength')) or 0.0
    target_portfolio_vol_monthly = _safe_float(cfg.get('target_portfolio_vol_monthly'))
    risk_toggle = bool(cfg.get('dispersion_sigma_enabled', False))

    if ewma_sigma and correlation_penalty_strength > 1e-8:
        return 'ewma_corr_penalized'
    if ewma_sigma and target_portfolio_vol_monthly is not None:
        return 'ewma_vol_targeted'
    if ewma_sigma:
        return 'ewma_sigma'
    if correlation_penalty_strength > 1e-8:
        return 'correlation_penalty_only'
    if target_portfolio_vol_monthly is not None or risk_toggle:
        return 'risk_guardrails'
    return 'none'


def _derive_caps_strength(cfg: Dict[str, Any]) -> str:
    explicit = _coerce_mode_name(cfg.get('caps_mode'), default='')
    if explicit:
        return explicit

    caps = [x for x in (_safe_float(cfg.get('asset_weight_cap')), _safe_float(cfg.get('w_cap'))) if x is not None]
    if not caps:
        return 'none'
    effective_cap = min(caps)
    if effective_cap <= 0.10:
        return 'strong'
    if effective_cap <= 0.20:
        return 'medium'
    return 'soft'


def _derive_turnover_control(cfg: Dict[str, Any]) -> str:
    explicit = _coerce_mode_name(cfg.get('turnover_penalty'), default='')
    if explicit:
        return explicit

    strength = _safe_float(cfg.get('turnover_penalty_strength')) or 0.0
    max_turnover = _safe_float(cfg.get('turnover_penalty_max_turnover'))
    constraint = _safe_float(cfg.get('turnover_constraint_max_turnover'))

    tight_limit = None
    for value in (max_turnover, constraint):
        if value is not None:
            tight_limit = value if tight_limit is None else min(tight_limit, value)

    # --- RELAXED THRESHOLDS (UX fix) ---
    if strength >= 1.2 or (tight_limit is not None and tight_limit <= 0.20):
        return 'high'
    if strength >= 0.30 or (tight_limit is not None and tight_limit <= 0.40):
        return 'medium'
    return 'low'


def _coerce_overlay_modes(values: Any) -> List[str]:
    raw_values = values if isinstance(values, (list, tuple, set)) else [values]
    out: List[str] = []
    for raw in raw_values:
        name = _coerce_mode_name(raw, default='none')
        if name in _VALID_OVERLAY_MODES and name not in out:
            out.append(name)
    return out or ['none']


def _coerce_signal_modes(values: Any) -> List[str]:
    raw_values = values if isinstance(values, (list, tuple, set)) else [values]
    out: List[str] = []
    for raw in raw_values:
        name = _coerce_mode_name(raw, default='mu_sigma')
        if name in _VALID_SIGNAL_MODES and name not in out:
            out.append(name)
    return out or ['mu_sigma']


def _bucket_specific_top_k_range(base_range: tuple[int, int], universe_bucket: str) -> tuple[int, int]:
    low, high = int(base_range[0]), int(base_range[1])
    if universe_bucket == 'small':
        return (max(2, min(low, 3)), max(4, min(high, 8)))
    if universe_bucket == 'large':
        return (max(low, 10), max(high, 18))
    return (low, high)


def _style_constraint_adjustments(style_preset: Any) -> Dict[str, Any]:
    style = str(style_preset or 'Balanced').strip().capitalize()
    if style == 'Growth':
        return {
            'top_k_shift': (1, 3),
            'overlay_bias': ['historical', 'none'],
            'signal_bias': ['mu_sigma', 'lambdarank_like'],
            'caps_floor': 'soft',
        }
    if style in {'Conservative', 'Defensive'}:
        return {
            'top_k_shift': (-2, -1),
            'overlay_bias': ['historical_by_regime', 'quantile_regression'],
            'signal_bias': ['huber_mu', 'mu_sigma'],
            'caps_floor': 'medium',
        }
    if style == 'Research':
        return {
            'top_k_shift': (0, 2),
            'overlay_bias': ['quantile_regression', 'historical'],
            'signal_bias': ['lambdarank_like', 'lambdarank_real', 'mu_sigma'],
            'caps_floor': 'soft',
        }
    return {
        'top_k_shift': (0, 0),
        'overlay_bias': [],
        'signal_bias': [],
        'caps_floor': None,
    }


def _caps_rank(value: str) -> int:
    return {'none': 0, 'soft': 1, 'medium': 2, 'strong': 3}.get(str(value), 0)


def _merge_caps_floor(current: str, floor_value: Optional[str]) -> str:
    if not floor_value:
        return current
    return floor_value if _caps_rank(floor_value) > _caps_rank(current) else current


def build_philosophy_spec(philosophy: str) -> dict:
    philosophy_name = _coerce_philosophy_name(philosophy)
    specs = {
        'Growth': {
            'philosophy': 'Growth',
            'signal_strength': 'high',
            'risk_priority': 'low',
            'diversification': 'low_to_medium',
            'concentration_allowed': True,
            'concentration_level': 'high',
            'top_k_range': (6, 14),
            'overlay_role': 'optional',
            'allowed_overlay_modes': ['none', 'historical', 'historical_by_features'],
            'preferred_overlay_modes': ['historical', 'none'],
            'covariance_importance': 'low',
            'covariance_required': False,
            'preferred_covariance_models': ['ewma_sigma', 'ewma_corr_penalized', 'correlation_penalty_only'],
            'caps_strength': 'soft',
            'allowed_caps_strength': ['soft', 'medium'],
            'turnover_tolerance': 'high',
            'preferred_signal_modes': ['mu_sigma', 'lambdarank_like', 'lambdarank_real'],
        },
        'Balanced': {
            'philosophy': 'Balanced',
            'signal_strength': 'medium',
            'risk_priority': 'medium',
            'diversification': 'medium',
            'concentration_allowed': False,
            'concentration_level': 'medium',
            'top_k_range': (8, 15),
            'overlay_role': 'supporting',
            'allowed_overlay_modes': ['historical', 'historical_by_regime', 'historical_by_features', 'quantile_regression'],
            'preferred_overlay_modes': ['historical', 'historical_by_regime'],
            'covariance_importance': 'medium',
            'covariance_required': True,
            'preferred_covariance_models': ['ewma_corr_penalized', 'ewma_vol_targeted', 'ewma_sigma'],
            'caps_strength': 'medium',
            'allowed_caps_strength': ['soft', 'medium', 'strong'],
            'turnover_tolerance': 'medium',
            'preferred_signal_modes': ['mu_sigma', 'huber_mu', 'lambdarank_like'],
        },
        'Defensive': {
            'philosophy': 'Defensive',
            'signal_strength': 'controlled',
            'risk_priority': 'high',
            'diversification': 'high',
            'concentration_allowed': False,
            'concentration_level': 'low',
            'top_k_range': (10, 20),
            'overlay_role': 'important',
            'allowed_overlay_modes': ['historical_by_regime', 'quantile_regression', 'historical'],
            'preferred_overlay_modes': ['historical_by_regime', 'quantile_regression'],
            'covariance_importance': 'high',
            'covariance_required': True,
            'preferred_covariance_models': ['ewma_corr_penalized', 'ewma_vol_targeted', 'risk_guardrails'],
            'caps_strength': 'strong',
            'allowed_caps_strength': ['medium', 'strong'],
            'turnover_tolerance': 'low',
            'preferred_signal_modes': ['huber_mu', 'mu_sigma', 'directional_classifier'],
        },
    }
    return dict(specs[philosophy_name])


def resolve_philosophy_to_constraints(
    philosophy: str,
    universe: Any,
    strategy_template: Any = None,
    style_preset: Any = None,
) -> dict:
    philosophy_name = _coerce_philosophy_name(philosophy)
    base_spec = build_philosophy_spec(philosophy_name)
    universe_bucket = _normalize_universe_bucket(universe)
    template = str(strategy_template or '').strip()
    style = str(style_preset or '').strip()

    top_k_low, top_k_high = _bucket_specific_top_k_range(base_spec['top_k_range'], universe_bucket)
    style_adj = _style_constraint_adjustments(style)
    low_shift, high_shift = style_adj.get('top_k_shift', (0, 0))
    top_k_range = (max(2, top_k_low + int(low_shift)), max(2, top_k_high + int(high_shift)))
    if top_k_range[1] < top_k_range[0]:
        top_k_range = (top_k_range[0], top_k_range[0])

    allowed_overlay_modes = list(base_spec.get('allowed_overlay_modes', ['none']))
    preferred_overlay_modes = list(base_spec.get('preferred_overlay_modes', allowed_overlay_modes))
    for mode in style_adj.get('overlay_bias', []):
        if mode in _VALID_OVERLAY_MODES and mode not in allowed_overlay_modes:
            allowed_overlay_modes.append(mode)
        if mode in _VALID_OVERLAY_MODES and mode not in preferred_overlay_modes:
            preferred_overlay_modes.insert(0, mode)

    preferred_signal_modes = list(base_spec.get('preferred_signal_modes', ['mu_sigma']))
    for mode in style_adj.get('signal_bias', []):
        if mode in _VALID_SIGNAL_MODES and mode not in preferred_signal_modes:
            preferred_signal_modes.insert(0, mode)

    if template == 'Hybrid Research':
        for mode in ['lambdarank_like', 'lambdarank_real']:
            if mode not in preferred_signal_modes:
                preferred_signal_modes.insert(0, mode)
        if 'quantile_regression' not in allowed_overlay_modes:
            allowed_overlay_modes.append('quantile_regression')
        if 'quantile_regression' not in preferred_overlay_modes:
            preferred_overlay_modes.insert(0, 'quantile_regression')
    elif template == 'Core Ranking':
        if 'mu_sigma' in preferred_signal_modes:
            preferred_signal_modes.remove('mu_sigma')
        preferred_signal_modes.insert(0, 'mu_sigma')

    caps_strength = _merge_caps_floor(str(base_spec.get('caps_strength', 'medium')), style_adj.get('caps_floor'))
    covariance_required = bool(base_spec.get('covariance_required', False)) or philosophy_name in {'Balanced', 'Defensive'}

    return {
        'philosophy': philosophy_name,
        'universe_bucket': universe_bucket,
        'strategy_template': template or None,
        'style_preset': style or None,
        'top_k_range': top_k_range,
        'covariance_required': covariance_required,
        'covariance_importance': str(base_spec.get('covariance_importance', 'medium')),
        'preferred_covariance_models': list(base_spec.get('preferred_covariance_models', [])),
        'allowed_overlay_modes': _coerce_overlay_modes(allowed_overlay_modes),
        'preferred_overlay_modes': _coerce_overlay_modes(preferred_overlay_modes),
        'preferred_signal_modes': _coerce_signal_modes(preferred_signal_modes),
        'caps_strength': caps_strength,
        'allowed_caps_strength': list(base_spec.get('allowed_caps_strength', ['medium'])),
        'turnover_tolerance': str(base_spec.get('turnover_tolerance', 'medium')),
        'concentration_allowed': bool(base_spec.get('concentration_allowed', False)),
        'concentration_level': str(base_spec.get('concentration_level', 'medium')),
        'overlay_role': str(base_spec.get('overlay_role', 'supporting')),
        'signal_strength': str(base_spec.get('signal_strength', 'medium')),
        'risk_priority': str(base_spec.get('risk_priority', 'medium')),
        'diversification': str(base_spec.get('diversification', 'medium')),
    }




def _turnover_rank(value: str) -> int:
    return {'low': 0, 'medium': 1, 'high': 2}.get(str(value), 0)


def _covariance_compatible(blocks: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
    if not bool(constraints.get('covariance_required', False)):
        return True
    model = str(blocks.get('covariance_model') or 'none')
    return model != 'none'


def _caps_compatible(blocks: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
    current = str(blocks.get('caps_strength') or 'none')
    allowed = [str(x) for x in list(constraints.get('allowed_caps_strength', []))]
    if allowed:
        return current in allowed
    required_floor = str(constraints.get('caps_strength') or 'none')
    return _caps_rank(current) >= _caps_rank(required_floor)


def _turnover_compatible(blocks: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
    current = str(blocks.get('turnover_control') or 'low')
    tolerance = str(constraints.get('turnover_tolerance') or 'medium')
    if tolerance == 'high':
        return True
    if tolerance == 'medium':
        return _turnover_rank(current) <= _turnover_rank('medium')
    return _turnover_rank(current) <= _turnover_rank('low')


def _derive_repair_status(*, incompatible: bool, n_issues: int, n_high_severity: int) -> str:
    if bool(incompatible) or int(n_high_severity) > 0:
        return 'incompatible'
    if int(n_issues) > 0:
        return 'repairable'
    return 'ok'


def _summarize_repair_severity(issues: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {'high': 0, 'medium': 0, 'low': 0}
    for item in list(issues or []):
        sev = str((item or {}).get('severity') or '').strip().lower()
        if sev in counts:
            counts[sev] += 1
    return counts


def detect_structural_incompatibilities(
    cfg: dict,
    philosophy: str,
    *,
    universe: Any = 'medium',
    strategy_template: Any = None,
    style_preset: Any = None,
) -> dict:
    payload = dict(cfg or {})
    philosophy_name = _coerce_philosophy_name(philosophy)
    constraints = resolve_philosophy_to_constraints(
        philosophy_name,
        universe=universe,
        strategy_template=strategy_template,
        style_preset=style_preset,
    )
    blocks = extract_config_blocks(payload)

    issues: List[Dict[str, Any]] = []
    actionable_warnings: List[str] = []
    suggested_patch: Dict[str, Any] = {}

    def _add_issue(code: str, severity: str, message: str, *, suggested_value: Any = None, patch: Optional[Dict[str, Any]] = None) -> None:
        issues.append({
            'code': str(code),
            'severity': str(severity),
            'message': str(message),
            'suggested_value': suggested_value,
        })
        actionable_warnings.append(str(message))
        if isinstance(patch, dict):
            suggested_patch.update(dict(patch))

    low_k, high_k = constraints.get('top_k_range', (0, 999))
    top_k_value = int(blocks.get('top_k', 0) or 0)
    if top_k_value > 0 and top_k_value < int(low_k):
        _add_issue('top_k_below_band', 'medium', f"top_k={top_k_value} is below the {philosophy_name} compatibility band [{low_k}, {high_k}]", suggested_value=int(low_k), patch={'top_k': int(low_k)})
    elif top_k_value > int(high_k):
        _add_issue('top_k_above_band', 'medium', f"top_k={top_k_value} is above the {philosophy_name} compatibility band [{low_k}, {high_k}]", suggested_value=int(high_k), patch={'top_k': int(high_k)})

    preferred_signal_modes = list(constraints.get('preferred_signal_modes', []))
    signal_mode = _coerce_mode_name(blocks.get('signal_mode'), default='mu_sigma')
    if preferred_signal_modes and signal_mode not in preferred_signal_modes:
        preferred_signal = preferred_signal_modes[0]
        _add_issue('signal_mode_not_preferred', 'medium', f"signal_mode='{signal_mode}' is outside the preferred {philosophy_name} set", suggested_value=preferred_signal, patch={'signal_mode': preferred_signal})

    allowed_overlay_modes = list(constraints.get('allowed_overlay_modes', []))
    preferred_overlay_modes = list(constraints.get('preferred_overlay_modes', allowed_overlay_modes))
    overlay_mode = _coerce_mode_name(blocks.get('overlay_mode'), default='none')
    if allowed_overlay_modes and overlay_mode not in allowed_overlay_modes:
        preferred_overlay = preferred_overlay_modes[0] if preferred_overlay_modes else allowed_overlay_modes[0]
        _add_issue('overlay_mode_incompatible', 'high' if philosophy_name == 'Defensive' else 'medium', f"overlay_mode='{overlay_mode}' is outside the {philosophy_name} compatibility set", suggested_value=preferred_overlay, patch={'probabilistic_mode': preferred_overlay})

    if not _covariance_compatible(blocks, constraints):
        preferred_covariance_models = list(constraints.get('preferred_covariance_models', []))
        suggested_model = preferred_covariance_models[0] if preferred_covariance_models else 'ewma_sigma'
        patch = {'ewma_sigma': True}
        if philosophy_name in {'Balanced', 'Defensive'}:
            patch['correlation_penalty_strength'] = max(0.75, _safe_float(payload.get('correlation_penalty_strength')) or 0.0)
        _add_issue('covariance_required_missing', 'high', f"{philosophy_name} requires an active covariance / risk-control layer", suggested_value=suggested_model, patch=patch)

    if not _caps_compatible(blocks, constraints):
        required_caps = str(constraints.get('caps_strength') or 'medium')
        cap_patch: Dict[str, Any] = {}
        if required_caps == 'strong':
            cap_patch['asset_weight_cap'] = min(0.10, _safe_float(payload.get('asset_weight_cap')) or 0.10)
        elif required_caps == 'medium':
            cap_patch['asset_weight_cap'] = min(0.20, _safe_float(payload.get('asset_weight_cap')) or 0.20)
        _add_issue('caps_strength_incompatible', 'high' if philosophy_name == 'Defensive' else 'medium', f"caps_strength='{blocks.get('caps_strength')}' is too loose for {philosophy_name}", suggested_value=required_caps, patch=cap_patch)

    if not _turnover_compatible(blocks, constraints):
        tolerance = str(constraints.get('turnover_tolerance') or 'medium')
        turnover_patch: Dict[str, Any] = {}
        if tolerance == 'low':
            turnover_patch['turnover_penalty_strength'] = max(1.0, _safe_float(payload.get('turnover_penalty_strength')) or 0.0)
            current_limit = _safe_float(payload.get('turnover_constraint_max_turnover'))
            turnover_patch['turnover_constraint_max_turnover'] = min(current_limit, 0.25) if current_limit is not None else 0.25
        elif tolerance == 'medium':
            turnover_patch['turnover_penalty_strength'] = max(0.20, _safe_float(payload.get('turnover_penalty_strength')) or 0.0)
        _add_issue('turnover_control_too_aggressive', 'medium', f"turnover_control='{blocks.get('turnover_control')}' is too aggressive for {philosophy_name}", suggested_value=tolerance, patch=turnover_patch)

    if philosophy_name == 'Growth' and str(blocks.get('caps_strength')) == 'strong':
        _add_issue('growth_overconstrained_caps', 'low', 'Growth profile looks over-constrained by strong caps', suggested_value='soft', patch={'asset_weight_cap': 0.20})

    incompatible = any(str(x.get('severity')) == 'high' for x in issues)
    n_issues = int(len(issues))
    severity_counts = _summarize_repair_severity(issues)
    n_high_severity = int(severity_counts.get('high', 0))
    status = _derive_repair_status(
        incompatible=bool(incompatible),
        n_issues=n_issues,
        n_high_severity=n_high_severity,
    )
    return {
        'philosophy': philosophy_name,
        'constraints': constraints,
        'blocks': blocks,
        'issues': issues,
        'actionable_warnings': actionable_warnings,
        'suggested_patch': suggested_patch,
        'incompatible': bool(incompatible),
        'n_issues': n_issues,
        'n_high_severity': n_high_severity,
        'severity_counts': severity_counts,
        'status': status,
        'governance_summary': {
            'status': status,
            'n_issues': n_issues,
            'n_high_severity': n_high_severity,
            'has_suggested_patch': bool(suggested_patch),
        },
    }


def suggest_coherence_repairs(
    cfg: dict,
    philosophy: str,
    *,
    universe: Any = 'medium',
    strategy_template: Any = None,
    style_preset: Any = None,
) -> dict:
    return detect_structural_incompatibilities(cfg, philosophy, universe=universe, strategy_template=strategy_template, style_preset=style_preset)


def apply_suggested_coherence_repairs(
    cfg: dict,
    philosophy: str,
    *,
    universe: Any = 'medium',
    strategy_template: Any = None,
    style_preset: Any = None,
) -> dict:
    payload = dict(cfg or {})
    repairs = suggest_coherence_repairs(payload, philosophy, universe=universe, strategy_template=strategy_template, style_preset=style_preset)
    merged = dict(payload)
    merged.update(dict(repairs.get('suggested_patch', {}) or {}))
    return merged


def build_coherence_precheck(
    cfg: dict,
    philosophy: str,
    *,
    universe: Any = 'medium',
    strategy_template: Any = None,
    style_preset: Any = None,
) -> dict:
    guidance = suggest_coherence_repairs(cfg, philosophy, universe=universe, strategy_template=strategy_template, style_preset=style_preset)
    return {**guidance, 'status': str(guidance.get('status', 'ok') or 'ok')}

def extract_config_blocks(cfg: dict) -> dict:
    payload = dict(cfg or {})
    top_k_raw = payload.get('top_k', 0)
    try:
        top_k = int(top_k_raw or 0)
    except Exception:
        top_k = 0
    return {
        'signal_mode': payload.get('signal_mode'),
        'top_k': top_k,
        'covariance_model': _derive_covariance_model(payload),
        'overlay_mode': _coerce_mode_name(payload.get('probabilistic_mode'), default='none'),
        'caps_strength': _derive_caps_strength(payload),
        'turnover_control': _derive_turnover_control(payload),
        'asset_weight_cap': _safe_float(payload.get('asset_weight_cap')),
        'w_cap': _safe_float(payload.get('w_cap')),
        'turnover_penalty_strength': _safe_float(payload.get('turnover_penalty_strength')),
        'correlation_penalty_strength': _safe_float(payload.get('correlation_penalty_strength')),
        'ewma_sigma': bool(payload.get('ewma_sigma', False)),
    }


def evaluate_config_coherence(
    cfg: dict,
    philosophy: str,
    *,
    universe: Any = 'medium',
    strategy_template: Any = None,
    style_preset: Any = None,
) -> dict:
    blocks = extract_config_blocks(cfg)
    philosophy_name = _coerce_philosophy_name(philosophy)

    # ✅ NUEVO: constraints reales (antes usabas spec fijo)
    constraints = resolve_philosophy_to_constraints(
        philosophy_name,
        universe=universe,
        strategy_template=strategy_template,
        style_preset=style_preset,
    )

    score = 0
    reasons = []
    warnings = []

    # --- lógica original mantenida ---
    if philosophy_name == 'Growth' and blocks['covariance_model'] == 'none':
        warnings.append('Growth without covariance control')

    if philosophy_name == 'Balanced' and blocks['covariance_model'] != 'none':
        score += 1
        reasons.append('Balanced risk structure OK')

    if philosophy_name == 'Defensive' and blocks['covariance_model'] == 'none':
        score -= 2
        warnings.append('Defensive without covariance model')

    # --- SIGNAL ---
    preferred_signals = set(constraints.get('preferred_signal_modes', []))
    if blocks['signal_mode'] in preferred_signals:
        score += 1
        reasons.append('Signal mode aligned with philosophy')
    elif blocks['signal_mode'] is not None:
        warnings.append('Signal mode outside preferred philosophy set')

    # --- OVERLAY ---
    allowed_overlays = set(constraints.get('allowed_overlay_modes', []))
    if blocks['overlay_mode'] not in allowed_overlays:
        score -= 1
        warnings.append('Overlay mode outside philosophy compatibility set')

    # =========================================================
    # ✅ FIX REAL: TOP_K
    # =========================================================

    top_k_value = int(blocks.get('top_k', 0) or 0)
    low_k, high_k = constraints.get('top_k_range', (0, 999))

    if top_k_value > 0:
        if top_k_value < low_k:
            score -= 2
            warnings.append(f"top_k below philosophy band [{low_k}, {high_k}]")
        elif top_k_value > high_k:
            score -= 2
            warnings.append(f"top_k above philosophy band [{low_k}, {high_k}]")
        else:
            score += 1
            reasons.append("top_k within philosophy band")

    # penalización fuerte
    if top_k_value > 0:
        if top_k_value < low_k * 0.5 or top_k_value > high_k * 1.5:
            score -= 1
            warnings.append("top_k strongly outside safe range")

    # --- LABEL ---
    if score >= 1:
        label = 'coherent'
    elif score >= -1:
        label = 'mixed'
    else:
        label = 'incoherent'

    guidance = suggest_coherence_repairs(
        cfg,
        philosophy_name,
        universe=universe,
        strategy_template=strategy_template,
        style_preset=style_preset,
    )

    return {
        'label': label,
        'score': score,
        'score_continuous': compute_coherence_score(
            cfg,
            philosophy_name,
            universe=universe,
            strategy_template=strategy_template,
            style_preset=style_preset,
        ),
        'reasons': reasons,
        'warnings': warnings,
        'actionable_warnings': list(guidance.get('actionable_warnings', [])),
        'suggested_patch': dict(guidance.get('suggested_patch', {}) or {}),
        'repair_plan': guidance,
        'blocks': blocks,
        'philosophy': philosophy_name,
        'philosophy_spec': constraints,  # 🔥 ahora correcto
        'status': str(guidance.get('status', 'ok') or 'ok'),
        'severity_counts': dict(guidance.get('severity_counts', {}) or {}),
        'governance_summary': dict(guidance.get('governance_summary', {}) or {}),
    }


def compute_coherence_score(
    cfg: dict,
    philosophy: str,
    *,
    universe: Any = 'medium',
    strategy_template: Any = None,
    style_preset: Any = None,
) -> float:
    blocks = extract_config_blocks(cfg)
    score = 0.5
    philosophy_name = _coerce_philosophy_name(philosophy)

    # --- lógica existente (igual) ---
    if philosophy_name == 'Defensive':
        if blocks['covariance_model'] == 'none':
            score -= 0.25
        if blocks['caps_strength'] in ['none', 'soft']:
            score -= 0.25
        if blocks['turnover_control'] == 'high':
            score -= 0.15
        if blocks['overlay_mode'] != 'none':
            score += 0.10

    top_k = int(blocks.get('top_k', 0) or 0)

    if philosophy_name == 'Growth':
        if blocks['caps_strength'] == 'strong':
            score -= 0.15
        elif blocks['caps_strength'] == 'medium':
            score -= 0.05

        if blocks['covariance_model'] != 'none':
            score += 0.05
        else:
            score -= 0.10

        if top_k > 30:
            score -= 0.10
        elif 0 < top_k <= 12:
            score += 0.05

        if blocks['overlay_mode'] != 'none':
            score += 0.05

    if philosophy_name == 'Balanced':
        if blocks['covariance_model'] != 'none':
            score += 0.10
        if blocks['caps_strength'] == 'medium':
            score += 0.05
        elif blocks['caps_strength'] == 'strong':
            score += 0.02
        if blocks['overlay_mode'] != 'none':
            score += 0.05
        if blocks['turnover_control'] == 'medium':
            score += 0.03
        elif blocks['turnover_control'] == 'high':
            score -= 0.05

    # =========================================================
    # ✅ FIX: usar constraints reales SIEMPRE
    # =========================================================

    constraints = resolve_philosophy_to_constraints(
        philosophy_name,
        universe=universe,
        strategy_template=strategy_template,
        style_preset=style_preset,
    )

    low_k, high_k = constraints.get('top_k_range', (0, 999))

    top_k = int(blocks.get('top_k', 0) or 0)

    if top_k > 0:
        if top_k < low_k:
            score -= 0.12
        elif top_k > high_k:
            score -= 0.12
        else:
            score += 0.05

    # coherencia general
    if blocks['signal_mode'] in set(constraints.get('preferred_signal_modes', [])):
        score += 0.04

    if blocks['overlay_mode'] in set(constraints.get('allowed_overlay_modes', [])):
        score += 0.03
    else:
        score -= 0.06

    return max(0.0, min(1.0, score))

def classify_governed_override_status(
    coherence_result: dict,
    repair_plan: dict,
) -> dict:
    coherence = dict(coherence_result or {})
    repairs = dict(repair_plan or {})
    explicit = str(repairs.get('status', coherence.get('status', '')) or '').strip().lower()
    if explicit == 'incompatible':
        status = 'discouraged'
    elif explicit == 'repairable':
        status = 'auto_repair_available'
    elif explicit == 'ok':
        status = 'coherent'
    elif explicit in {'coherent', 'stretched', 'discouraged', 'auto_repair_available'}:
        status = explicit
    else:
        label = str(coherence.get('label', 'unavailable') or 'unavailable').strip().lower()
        try:
            score = float(coherence.get('score_continuous', 0.5) or 0.5)
        except Exception:
            score = 0.5
        has_patch = bool(dict(repairs.get('suggested_patch', {}) or {}))
        n_issues = int(repairs.get('n_issues', 0) or 0)
        n_high = int(repairs.get('n_high_severity', 0) or 0)
        if bool(repairs.get('incompatible', False)) or label == 'incoherent' or score < 0.35 or n_high > 0:
            status = 'discouraged'
        elif has_patch and (n_issues > 0 or label in {'mixed', 'unavailable'} or score < 0.70):
            status = 'auto_repair_available'
        elif n_issues > 0 or label == 'mixed' or score < 0.60:
            status = 'stretched'
        elif label == 'coherent' and score >= 0.75:
            status = 'coherent'
        else:
            status = 'stretched'
    label_map = {
        'coherent': 'coherent',
        'stretched': 'stretched',
        'discouraged': 'discouraged',
        'auto_repair_available': 'auto-repair available',
    }
    return {
        'status': status,
        'label': label_map.get(status, status),
        'has_suggested_patch': bool(dict(repairs.get('suggested_patch', {}) or {})),
        'n_issues': int(repairs.get('n_issues', 0) or 0),
        'n_high_severity': int(repairs.get('n_high_severity', 0) or 0),
    }


def build_coherence_governance_payload(
    cfg: dict,
    philosophy: str,
    *,
    universe: Any = 'medium',
    strategy_template: Any = None,
    style_preset: Any = None,
) -> dict:
    philosophy_name = _coerce_philosophy_name(philosophy)
    repairs = suggest_coherence_repairs(
        cfg,
        philosophy_name,
        universe=universe,
        strategy_template=strategy_template,
        style_preset=style_preset,
    )
    coherence = evaluate_config_coherence(
        cfg,
        philosophy_name,
        universe=universe,
        strategy_template=strategy_template,
        style_preset=style_preset,
    )
    return {
        'philosophy': philosophy_name,
        'constraints': dict(repairs.get('constraints', {}) or {}),
        'blocks': dict(repairs.get('blocks', {}) or {}),
        'coherence': coherence,
        'repairs': repairs,
        'status': str(repairs.get('status', coherence.get('status', 'ok')) or 'ok'),
        'suggested_patch': dict(repairs.get('suggested_patch', {}) or {}),
        'actionable_warnings': list(repairs.get('actionable_warnings', []) or []),
        'governance_summary': dict(repairs.get('governance_summary', {}) or {}),
    }


_DEF_BLOCK_TITLES = {
    'signal': 'Signal block',
    'overlay': 'Overlay block',
    'covariance_risk': 'Covariance / risk block',
    'caps_diversification': 'Caps / diversification block',
    'turnover_stability': 'Turnover / stability block',
    'allocator': 'Allocator block',
}


def _governed_block_role_matrix(philosophy: str) -> dict:
    philosophy_name = _coerce_philosophy_name(philosophy)
    matrix = {
        'Growth': {
            'signal': ('Core', 'Signal should lead the portfolio logic here; stronger signal expression is expected.', ['upside capture', 'signal expression']),
            'overlay': ('Optional', 'Overlay can help, but it should not dominate Growth.', ['robustness', 'downside moderation']),
            'covariance_risk': ('Supporting', 'Risk structure helps keep Growth usable, but should not suffocate edge capture.', ['drawdown control', 'robustness']),
            'caps_diversification': ('Supporting', 'Soft caps help avoid accidental concentration, but very hard caps can mute upside.', ['diversification', 'concentration control']),
            'turnover_stability': ('Supporting', 'Some turnover discipline is useful, but too much suppression can reduce responsiveness.', ['turnover control', 'stability']),
            'allocator': ('Core', 'Allocator should translate signal into weights without becoming overly defensive.', ['portfolio construction', 'signal transmission']),
        },
        'Balanced': {
            'signal': ('Core', 'Balanced still needs signal to matter, but not in a purely aggressive way.', ['signal expression', 'robustness']),
            'overlay': ('Supporting', 'Overlay is useful when it supports risk-adjusted behaviour without overwhelming the base signal.', ['robustness', 'downside moderation']),
            'covariance_risk': ('Core', 'Balanced depends on an explicit risk structure to remain structurally believable.', ['drawdown control', 'robustness']),
            'caps_diversification': ('Core', 'Moderate caps and diversification are part of the identity of Balanced.', ['diversification', 'concentration control']),
            'turnover_stability': ('Supporting', 'Balanced usually benefits from moderate turnover discipline.', ['turnover control', 'stability']),
            'allocator': ('Core', 'Allocator should blend signal capture with discipline rather than pushing into extremes.', ['portfolio construction', 'robustness']),
        },
        'Defensive': {
            'signal': ('Supporting', 'Signal still matters, but it should serve a risk-governed posture rather than dominate it.', ['robustness', 'signal filtering']),
            'overlay': ('Core', 'A downside-aware or regime-aware overlay is often structurally helpful for Defensive.', ['drawdown control', 'downside moderation']),
            'covariance_risk': ('Core', 'Risk modelling is a central pillar of Defensive rather than an optional extra.', ['drawdown control', 'robustness']),
            'caps_diversification': ('Core', 'Caps and diversification guardrails are foundational for Defensive coherence.', ['diversification', 'concentration control']),
            'turnover_stability': ('Core', 'Defensive usually wants explicit stability / turnover discipline.', ['turnover control', 'stability']),
            'allocator': ('Supporting', 'Allocator should translate the defensive posture into actual weight discipline.', ['portfolio construction', 'robustness']),
        },
    }
    return dict(matrix.get(philosophy_name, matrix['Balanced']))


def _describe_allocator_block(cfg: dict, philosophy: str) -> tuple[bool, str, list[str]]:
    payload = dict(cfg or {})
    correlation_aware = bool(payload.get('correlation_aware_allocation', False))
    method = str(payload.get('correlation_allocator_method', 'base_allocator') or 'base_allocator')
    active = bool(correlation_aware or method not in {'', 'base_allocator', 'none'})
    warnings: list[str] = []
    if _coerce_philosophy_name(philosophy) == 'Defensive' and not correlation_aware:
        warnings.append('Defensive can benefit from allocator-level discipline beyond the base scorer.')
    if _coerce_philosophy_name(philosophy) == 'Growth' and method == 'risk_budget':
        warnings.append('Risk-budget allocator may mute upside capture in Growth.')
    return active, method if active else 'base_allocator', warnings


def describe_governed_config_blocks(
    cfg: dict,
    philosophy: str,
    *,
    universe: Any = 'medium',
    strategy_template: Any = None,
    style_preset: Any = None,
) -> list[dict]:
    payload = dict(cfg or {})
    philosophy_name = _coerce_philosophy_name(philosophy)
    constraints = resolve_philosophy_to_constraints(
        philosophy_name,
        universe=universe,
        strategy_template=strategy_template,
        style_preset=style_preset,
    )
    blocks = extract_config_blocks(payload)
    coherence = evaluate_config_coherence(
        payload,
        philosophy_name,
        universe=universe,
        strategy_template=strategy_template,
        style_preset=style_preset,
    )
    repair_plan = suggest_coherence_repairs(
        payload,
        philosophy_name,
        universe=universe,
        strategy_template=strategy_template,
        style_preset=style_preset,
    )
    matrix = _governed_block_role_matrix(philosophy_name)
    preferred_signals = set(constraints.get('preferred_signal_modes', []))

    allowed_overlays = set(constraints.get('allowed_overlay_modes', []))
    allocator_active, allocator_value, allocator_warnings = _describe_allocator_block(payload, philosophy_name)

    catalog = [
        ('signal', bool(str(blocks.get('signal_mode', '') or '').strip()), str(blocks.get('signal_mode', 'none') or 'none')),
        ('overlay', str(blocks.get('overlay_mode', 'none') or 'none') != 'none', str(blocks.get('overlay_mode', 'none') or 'none')),
        ('covariance_risk', str(blocks.get('covariance_model', 'none') or 'none') != 'none', str(blocks.get('covariance_model', 'none') or 'none')),
        ('caps_diversification', str(blocks.get('caps_strength', 'none') or 'none') != 'none', str(blocks.get('caps_strength', 'none') or 'none')),
        ('turnover_stability', str(blocks.get('turnover_control', 'low') or 'low') not in {'none', 'low'}, str(blocks.get('turnover_control', 'low') or 'low')),
        ('allocator', allocator_active, allocator_value),
    ]

    rows: list[dict] = []
    for block_key, active, current_value in catalog:
        role, explanation, effects = matrix.get(block_key, ('Optional', '', []))
        warnings: list[str] = []
        if block_key == 'signal' and current_value not in preferred_signals and current_value not in {'none', ''}:
            warnings.append('Current signal family sits outside the preferred philosophy set.')
        if block_key == 'overlay' and current_value not in allowed_overlays:
            warnings.append('Current overlay mode sits outside the philosophy compatibility set.')
        if block_key == 'covariance_risk' and philosophy_name in {'Balanced', 'Defensive'} and current_value == 'none':
            warnings.append(f'{philosophy_name} usually wants explicit covariance / risk structure.')
        if block_key == 'caps_diversification' and philosophy_name == 'Defensive' and str(current_value) in {'none', 'soft'}:
            warnings.append('Defensive usually wants stronger caps and diversification guardrails.')
        if block_key == 'caps_diversification' and philosophy_name == 'Growth' and str(current_value) == 'strong':
            warnings.append('Strong caps can become restrictive for Growth.')
        if block_key == 'turnover_stability' and philosophy_name == 'Growth' and str(current_value) == 'high':
            warnings.append('Heavy turnover suppression may reduce edge capture in Growth.')
        if block_key == 'turnover_stability' and philosophy_name == 'Defensive' and str(current_value) == 'low':
            warnings.append('Defensive usually benefits from more explicit stability control.')
        if block_key == 'allocator':
            warnings.extend(list(allocator_warnings))
        if role == 'Core' and not active and block_key in {'covariance_risk', 'caps_diversification', 'turnover_stability'}:
            warnings.append('This block is currently inactive even though it plays a structurally important role for this philosophy.')

        rows.append({
            'block_key': block_key,
            'title': _DEF_BLOCK_TITLES.get(block_key, block_key.replace('_', ' ').title()),
            'role': role,
            'active': bool(active),
            'current_value': current_value,
            'expected_effects': list(effects),
            'explanation': explanation,
            'warnings': warnings,
            'coherence_label': str(coherence.get('label', 'unavailable') or 'unavailable'),
            'coherence_score': float(coherence.get('score_continuous', 0.5) or 0.5),
            'repair_patch_available': bool(dict(repair_plan.get('suggested_patch', {}) or {})),
        })
    return rows


def _fmt_governed_param_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "on" if value else "off"
    try:
        num = float(value)
        if not (num == num and num not in (float("inf"), float("-inf"))):
            return str(value)
        if abs(num) >= 10:
            return f"{num:.2f}"
        if abs(num) >= 1:
            return f"{num:.3f}"
        return f"{num:.4f}"
    except Exception:
        return str(value)


def _governed_parameter_specs() -> list[dict]:
    return [
        {
            'param_key': 'top_k',
            'title': 'top_k',
            'block_key': 'signal',
            'lower_reading': 'more concentration, stronger signal expression, less diversification',
            'higher_reading': 'broader diversification, lower concentration, slightly softer signal expression',
            'safe_hint': 'move back toward the coherent base cutoff unless you intentionally want a different concentration regime',
        },
        {
            'param_key': 'asset_weight_cap',
            'title': 'asset_weight_cap',
            'block_key': 'caps_diversification',
            'lower_reading': 'stronger caps, more diversification discipline, less single-name concentration',
            'higher_reading': 'looser caps, more concentration freedom, potentially higher upside but more fragility',
            'safe_hint': 'bring caps closer to the coherent base unless you explicitly want a different concentration posture',
        },
        {
            'param_key': 'turnover_penalty_strength',
            'title': 'turnover_penalty_strength',
            'block_key': 'turnover_stability',
            'lower_reading': 'more responsive to fresh signal, but potentially noisier and more cost-sensitive',
            'higher_reading': 'more stable and cost-aware, but potentially slower to react to new signal',
            'safe_hint': 'use the coherent base as the default and only push harder if you knowingly want more stability discipline',
        },
        {
            'param_key': 'target_portfolio_vol_monthly',
            'title': 'target_portfolio_vol_monthly',
            'block_key': 'covariance_risk',
            'lower_reading': 'more defensive risk budget and tighter path control',
            'higher_reading': 'more aggressive risk budget and greater upside tolerance, but potentially heavier drawdowns',
            'safe_hint': 'keep the vol target close to the coherent base unless you explicitly want to shift risk appetite',
        },
        {
            'param_key': 'correlation_penalty_strength',
            'title': 'correlation_penalty_strength',
            'block_key': 'covariance_risk',
            'lower_reading': 'less correlation discipline and more raw signal freedom',
            'higher_reading': 'stronger correlation discipline and more robust diversification control',
            'safe_hint': 'rebalance toward the coherent base if you want philosophy-consistent risk structure',
        },
        {
            'param_key': 'sigma_power_alpha',
            'title': 'sigma_power_alpha',
            'block_key': 'signal',
            'lower_reading': 'more aggressive raw-signal expression with less sigma penalty',
            'higher_reading': 'more sigma-aware scoring and a more risk-sensitive signal posture',
            'safe_hint': 'keep alpha close to the coherent base unless you explicitly want a stronger or weaker sigma penalty',
        },
        {
            'param_key': 'probabilistic_overlay_strength',
            'title': 'probabilistic_overlay_strength',
            'block_key': 'overlay',
            'lower_reading': 'lighter overlay influence, leaving more control to the base signal',
            'higher_reading': 'heavier overlay influence, adding more probabilistic moderation to the portfolio',
            'safe_hint': 'bring overlay strength back toward the coherent base if overlay is starting to dominate the strategy identity',
        },
        {
            'param_key': 'covariance_shrink_to_diagonal',
            'title': 'covariance_shrink_to_diagonal',
            'block_key': 'covariance_risk',
            'lower_reading': 'more expressive covariance structure and less shrinkage-based regularisation',
            'higher_reading': 'more shrinkage and a more regularised, robustness-oriented covariance estimate',
            'safe_hint': 'use the coherent base as the anchor unless you have a deliberate reason to change the covariance regularisation regime',
        },
    ]


def describe_governed_parameter_overrides(
    base_cfg: dict,
    current_cfg: dict,
    philosophy: str,
    *,
    universe: Any = 'medium',
    strategy_template: Any = None,
    style_preset: Any = None,
) -> list[dict]:
    base_payload = dict(base_cfg or {})
    current_payload = dict(current_cfg or {})
    philosophy_name = _coerce_philosophy_name(philosophy)
    block_roles = _governed_block_role_matrix(philosophy_name)
    rows: list[dict] = []

    for spec in _governed_parameter_specs():
        key = str(spec.get('param_key'))
        block_key = str(spec.get('block_key', 'signal'))
        role, _, _ = block_roles.get(block_key, ('Optional', '', []))
        base_val = base_payload.get(key)
        current_val = current_payload.get(key)
        if key == 'asset_weight_cap' and current_val is None:
            current_val = current_payload.get('w_cap')
        if key == 'asset_weight_cap' and base_val is None:
            base_val = base_payload.get('w_cap')
        if base_val is None and current_val is None:
            continue

        changed = base_val != current_val
        try:
            base_num = float(base_val) if base_val is not None else None
        except Exception:
            base_num = None
        try:
            current_num = float(current_val) if current_val is not None else None
        except Exception:
            current_num = None

        diff = None
        rel_shift = None
        direction = 'unchanged'
        reading = 'Aligned with the coherent base value.'
        structural_shift = 'aligned'
        safe_suggestion = ''
        status = 'coherent'

        if current_num is not None and base_num is not None:
            diff = float(current_num - base_num)
            denom = max(abs(base_num), 1e-9)
            rel_shift = float(abs(diff) / denom)
            if abs(diff) <= 1e-12:
                direction = 'unchanged'
            elif diff > 0:
                direction = 'higher'
            else:
                direction = 'lower'
            reading = spec['higher_reading'] if direction == 'higher' else spec['lower_reading'] if direction == 'lower' else reading
            structural_shift = 'more defensive / robust' if direction == 'higher' and key in {'turnover_penalty_strength', 'correlation_penalty_strength', 'sigma_power_alpha', 'probabilistic_overlay_strength', 'covariance_shrink_to_diagonal'} else structural_shift
            if direction == 'lower' and key in {'turnover_penalty_strength', 'correlation_penalty_strength', 'sigma_power_alpha', 'probabilistic_overlay_strength', 'covariance_shrink_to_diagonal'}:
                structural_shift = 'more aggressive / less guarded'
            if key == 'top_k':
                structural_shift = 'more diversified / less concentrated' if direction == 'higher' else 'more concentrated / more aggressive' if direction == 'lower' else 'aligned'
            elif key == 'asset_weight_cap':
                structural_shift = 'looser concentration guardrails' if direction == 'higher' else 'tighter concentration guardrails' if direction == 'lower' else 'aligned'
            elif key == 'target_portfolio_vol_monthly':
                structural_shift = 'higher risk appetite' if direction == 'higher' else 'lower risk appetite' if direction == 'lower' else 'aligned'

            if rel_shift is not None:
                if rel_shift >= 0.50:
                    status = 'discouraged'
                elif rel_shift >= 0.20:
                    status = 'stretched'
                else:
                    status = 'coherent'

            # Philosophy-aware guardrails
            if philosophy_name == 'Defensive':
                if key in {'top_k', 'target_portfolio_vol_monthly', 'asset_weight_cap'} and direction == 'higher':
                    status = 'discouraged' if rel_shift is not None and rel_shift >= 0.25 else 'stretched'
                if key in {'turnover_penalty_strength', 'correlation_penalty_strength', 'probabilistic_overlay_strength', 'sigma_power_alpha'} and direction == 'lower':
                    status = 'discouraged' if rel_shift is not None and rel_shift >= 0.25 else 'stretched'
            elif philosophy_name == 'Balanced':
                if key in {'top_k', 'target_portfolio_vol_monthly', 'asset_weight_cap'} and direction == 'higher' and rel_shift is not None and rel_shift >= 0.35:
                    status = 'stretched'
                if key in {'turnover_penalty_strength', 'correlation_penalty_strength'} and direction == 'lower' and rel_shift is not None and rel_shift >= 0.35:
                    status = 'stretched'
                if key == 'top_k' and direction == 'lower' and rel_shift is not None and rel_shift >= 0.35:
                    status = 'stretched'
            elif philosophy_name == 'Growth':
                if key in {'turnover_penalty_strength', 'correlation_penalty_strength', 'sigma_power_alpha', 'probabilistic_overlay_strength', 'covariance_shrink_to_diagonal'} and direction == 'higher' and rel_shift is not None and rel_shift >= 0.35:
                    status = 'stretched'
                if key in {'top_k', 'asset_weight_cap', 'target_portfolio_vol_monthly'} and direction == 'lower' and rel_shift is not None and rel_shift >= 0.35:
                    status = 'stretched'

            safe_suggestion = f"Consider moving **{key}** back toward the coherent base value ({_fmt_governed_param_value(base_num)}); {spec['safe_hint']}." if status in {'stretched', 'discouraged'} else ''

        rows.append({
            'param_key': key,
            'title': spec['title'],
            'block_key': block_key,
            'block_title': _DEF_BLOCK_TITLES.get(block_key, block_key.replace('_', ' ').title()),
            'role': role,
            'base_value': base_val,
            'base_value_display': _fmt_governed_param_value(base_val),
            'current_value': current_val,
            'current_value_display': _fmt_governed_param_value(current_val),
            'changed': bool(changed),
            'direction': direction,
            'diff': diff,
            'relative_shift': rel_shift,
            'status': status,
            'structural_shift': structural_shift,
            'reading': reading,
            'safe_suggestion': safe_suggestion,
        })

    return rows
