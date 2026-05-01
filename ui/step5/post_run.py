from __future__ import annotations

import json
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from ui.step5.run_panel import clear_retired_step5_state
from ui.step5.preset_recommendations import (
    PRESET_APPLIED_LABEL_KEY,
    PRESET_APPLIED_SIGNATURE_KEY,
    PRESET_SKIPPED_LABEL_KEY,
    PRESET_SKIPPED_SCOPE_KEY,
    render_preset_improvement,
)
from ui.step5.auto_opt_recommendations import (
    AUTO_OPT_APPLIED_LABEL_KEY,
    AUTO_OPT_APPLIED_SIGNATURE_KEY,
    AUTO_OPT_SKIPPED_LABEL_KEY,
    AUTO_OPT_SKIPPED_RUN_SIGNATURE_KEY,
    AUTO_OPT_SKIPPED_SCOPE_KEY,
    AUTO_OPT_SUGGESTION_SCOPE_KEY,
    AUTO_OPT_SUGGESTION_STATE_KEY,
    AUTO_OPT_SUGGESTION_TIMING_KEY,
    render_auto_opt_improvement,
)
from ui.step5.universe_recommendations import (
    UNIVERSE_APPLIED_LABEL_KEY,
    UNIVERSE_APPLIED_SIGNATURE_KEY,
    UNIVERSE_SKIPPED_LABEL_KEY,
    UNIVERSE_SKIPPED_RUN_SIGNATURE_KEY,
    UNIVERSE_SKIPPED_SCOPE_KEY,
    UNIVERSE_SUGGESTION_SCOPE_KEY,
    UNIVERSE_SUGGESTION_STATE_KEY,
    UNIVERSE_SUGGESTION_TIMING_KEY,
    render_universe_improvement,
)
from ui.step5.size_recommendations import (
    SIZE_APPLIED_LABEL_KEY,
    SIZE_APPLIED_SIGNATURE_KEY,
    SIZE_SKIPPED_LABEL_KEY,
    SIZE_SKIPPED_RUN_SIGNATURE_KEY,
    SIZE_SKIPPED_SCOPE_KEY,
    SIZE_SUGGESTION_TIMING_KEY,
    render_size_improvement,
)
from ui.step5.reliability_assessment import render_result_reliability_assessment, render_start_date_robustness_timing_block



STEP5_SCROLL_TO_RESULT_AFTER_APPLY_KEY = "step5_scroll_to_real_run_result_after_apply_v1"
PRESET_SUGGESTION_TIMING_KEY = "step5_preset_suggestion_timing_v1"
STEP5_REAL_RUN_RESULT_ANCHOR_ID = "step5-real-run-result-anchor"
STEP5_IMPROVEMENT_CHECKS_ANCHOR_ID = "step5-improvement-checks-anchor"
STEP5_RELIABILITY_ANCHOR_ID = "step5-reliability-robustness-anchor"
STEP5_RELIABILITY_EXPANDED_KEY = "step5_reliability_and_robustness_expanded_v1"
STEP5_SCROLL_TO_RELIABILITY_KEY = "step5_scroll_to_reliability_and_robustness_v1"
STEP5_RUN_DIAGNOSTICS_ANCHOR_ID = "step5-run-diagnostics-anchor"
STEP5_RUN_DIAGNOSTICS_EXPANDED_KEY = "step5_run_diagnostics_expanded_v1"
STEP5_SCROLL_TO_RUN_DIAGNOSTICS_KEY = "step5_scroll_to_run_diagnostics_v1"


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


def _clamp_suggestion_count(value: Any, *, default: int = 1, low: int = 1, high: int = 8) -> int:
    try:
        raw = int(value)
    except Exception:
        raw = int(default)
    return int(max(low, min(high, raw)))


def _format_runtime_estimate(seconds: float) -> str:
    """Human-sized runtime estimate for suggestion-test spinners and captions."""
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


def _current_suggestion_counts() -> dict[str, int]:
    defaults = SUGGESTION_DEPTH_PRESETS[SUGGESTION_DEPTH_DEFAULT]
    out: dict[str, int] = {}
    for phase, key in SUGGESTION_DEPTH_COUNT_KEYS.items():
        out[phase] = _clamp_suggestion_count(st.session_state.get(key, defaults.get(phase, 1)), default=defaults.get(phase, 1))
    return out


def _apply_suggestion_depth_preset(mode: str) -> dict[str, int]:
    if mode not in SUGGESTION_DEPTH_PRESETS:
        return _current_suggestion_counts()
    counts = dict(SUGGESTION_DEPTH_PRESETS[mode])
    for phase, value in counts.items():
        st.session_state[SUGGESTION_DEPTH_COUNT_KEYS[phase]] = int(value)
    return counts


def _render_suggestion_depth_controls() -> None:
    """Optional runtime/coverage control for the sequential suggestion tests."""
    defaults = SUGGESTION_DEPTH_PRESETS[SUGGESTION_DEPTH_DEFAULT]
    if SUGGESTION_DEPTH_MODE_KEY not in st.session_state:
        st.session_state[SUGGESTION_DEPTH_MODE_KEY] = SUGGESTION_DEPTH_DEFAULT
    for phase, key in SUGGESTION_DEPTH_COUNT_KEYS.items():
        if key not in st.session_state:
            st.session_state[key] = int(defaults.get(phase, 1))

    with st.expander("Suggestion testing depth", expanded=False):
        st.caption("Higher depth tests more alternatives, but each extra candidate adds real engine runtime.")
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
                f"{mode}: "
                f"{counts['preset']} preset · {counts['auto_opt']} tuning · "
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
            "The estimate is approximate and depends on the deployed environment, cache state and panel size. "
            "Timings are recorded later in Run timings and diagnostics."
        )


def _maybe_scroll_to_improvement_checks() -> None:
    """Consume the legacy scroll flag without moving the viewport.

    The Strategy Engine surface is now compact enough that forced scroll jumps
    after Apply/Skip/Test actions feel disorienting. Older modules may still set
    the flag, so this function keeps compatibility while making the UI stable.
    """
    st.session_state.pop(STEP5_SCROLL_TO_RESULT_AFTER_APPLY_KEY, None)
    return None


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
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _pct(v: float) -> str:
    return f"{100.0 * float(v):.2f}%"


def _display_candidate_status(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("_", " ")
    if raw in {"recommended", "selected", "best"}:
        return "recommended"
    if raw in {"accepted", "passed", "passed gate", "pass gate"}:
        return "passed gate"
    if raw in {"not selected", "rejected", "not accepted"}:
        return "not selected"
    return str(value or "")


def _parse_feature_family_counts(raw: Any) -> dict[str, int]:
    if isinstance(raw, dict):
        source = raw
    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            source = parsed if isinstance(parsed, dict) else {}
        except Exception:
            source = {}
    else:
        source = {}

    out: dict[str, int] = {}
    for key, value in source.items():
        name = str(key).strip()
        if not name:
            continue
        out[name] = _safe_int(value, 0)
    return out


def _parse_feature_cols(raw: Any) -> list[str]:
    if isinstance(raw, list):
        items = raw
    else:
        items = str(raw or "").split("|")
    return [str(x).strip() for x in items if str(x).strip()]


def _render_feature_mu_block(run_map: dict) -> None:
    selected_cols_n = _safe_int(run_map.get("feature_mu_selected_cols_n", 0), 0)
    selected_cols = _parse_feature_cols(run_map.get("feature_mu_selected_cols", ""))
    family_counts = _parse_feature_family_counts(run_map.get("feature_mu_selected_family_counts", "{}"))

    # If the engine did not return feature_mu metadata, do not render an empty-looking section.
    has_metadata = bool(selected_cols_n > 0 or len(selected_cols) > 0 or len(family_counts) > 0)
    if not has_metadata:
        return

    st.markdown("### Feature_mu selection")
    m1, m2 = st.columns([1.0, 2.0])
    with m1:
        st.metric("Feature_mu cols used", int(selected_cols_n))
    with m2:
        if family_counts:
            family_text = " · ".join(f"{name}: {count}" for name, count in family_counts.items())
            st.caption("Families")
            st.write(family_text)
        else:
            st.caption("Families")
            st.write("No family summary available.")

    if selected_cols:
        with st.container(border=True):
            st.caption("Selected feature_mu columns")
            st.write(selected_cols)



def _extract_oos_returns(run_map: dict) -> list[float]:
    candidates = [
        run_map.get("oos_returns_monthly"),
        run_map.get("oos_returns_simple"),
        run_map.get("portfolio_returns"),
        run_map.get("oos_returns"),
        run_map.get("returns"),
    ]

    for raw in candidates:
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


def _store_projection_bridge_context(run_map: dict) -> int:
    ctx = st.session_state.get("investment_context", {})
    if not isinstance(ctx, dict):
        ctx = {}

    oos_returns = _extract_oos_returns(run_map)
    ctx["oos_returns_monthly"] = list(oos_returns)
    ctx.setdefault("run_signature", str(run_map.get("run_signature", "") or ""))
    st.session_state["investment_context"] = ctx
    return int(len(oos_returns))




def _render_engine_timing_block(run_map: dict) -> None:
    engine_timing = _coerce_mapping(run_map.get("engine_timing", {}))
    if not engine_timing:
        return

    st.markdown("### Engine timing summary")

    total_engine = _safe_float(engine_timing.get("total_engine", 0.0), 0.0)
    walk_forward = _safe_float(engine_timing.get("walk_forward_loop", 0.0), 0.0)
    probabilistic = _safe_float(engine_timing.get("probabilistic_total", 0.0), 0.0)
    n_oos_dates = _safe_int(engine_timing.get("n_oos_dates", 0), 0)
    n_loop_iterations = _safe_int(engine_timing.get("n_loop_iterations", 0), 0)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total engine", f"{total_engine:.2f}s")
    with c2:
        st.metric("Walk-forward", f"{walk_forward:.2f}s")
    with c3:
        st.metric("Probabilistic", f"{probabilistic:.2f}s")
    with c4:
        st.metric("OOS months", int(n_oos_dates))

    st.caption(f"Loop iterations={n_loop_iterations}. Timing is diagnostic only and can vary by environment.")

    detail_rows = []
    preferred_order = [
        "prepare_panel",
        "feature_pair_prep",
        "walk_forward_loop",
        "mu_sigma_total",
        "probabilistic_total",
        "feature_mu_total",
        "signal_model_total",
        "regime_filter_total",
        "covariance_sigma_total",
        "weight_build_total",
        "post_weights_total",
        "diagnostics_total",
        "finalize_total",
        "total_engine",
    ]

    for key in preferred_order:
        if key in engine_timing:
            value = engine_timing.get(key)
            if isinstance(value, (int, float)):
                detail_rows.append({"component": key, "seconds": float(value)})

    if detail_rows:
        with st.container(border=True):
            st.caption("Engine timing breakdown")
            st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)


def _render_preset_suggestion_timing_block() -> None:
    """Render timing for the automatic preset-suggestion test, if available."""
    timing = _coerce_mapping(st.session_state.get(PRESET_SUGGESTION_TIMING_KEY, {}))
    if not timing:
        return

    total_seconds = _safe_float(timing.get("total_seconds", 0.0), 0.0)
    candidate_count = _safe_int(timing.get("candidate_count", 0), 0)
    accepted_count = _safe_int(timing.get("accepted_count", 0), 0)
    if total_seconds <= 0 and candidate_count <= 0:
        return

    st.markdown("### Preset suggestion timing")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Preset test", f"{total_seconds:.2f}s")
    with c2:
        st.metric("Candidates tested", int(candidate_count))
    with c3:
        st.metric("Passed gate", int(accepted_count))

    st.caption(
        "This is the extra time used by the automatic strategy-preset suggestion test. "
        "It is separate from the main portfolio engine run shown above."
    )

    rows = timing.get("candidate_seconds", [])
    if isinstance(rows, list) and rows:
        clean_rows = []
        for row in rows:
            row_map = _coerce_mapping(row)
            clean_rows.append(
                {
                    "candidate": str(row_map.get("candidate", "Candidate") or "Candidate"),
                    "status": _display_candidate_status(row_map.get("status", "")),
                    "seconds": _safe_float(row_map.get("seconds", 0.0), 0.0),
                }
            )
        if clean_rows:
            with st.container(border=True):
                st.caption("Preset suggestion timing breakdown")
                st.dataframe(pd.DataFrame(clean_rows), use_container_width=True, hide_index=True)


def _render_auto_opt_suggestion_timing_block() -> None:
    """Render timing for the automatic engine-tuning test, if available."""
    timing = _coerce_mapping(st.session_state.get(AUTO_OPT_SUGGESTION_TIMING_KEY, {}))
    if not timing:
        return

    total_seconds = _safe_float(timing.get("total_seconds", 0.0), 0.0)
    candidate_count = _safe_int(timing.get("candidate_count", 0), 0)
    accepted_count = _safe_int(timing.get("accepted_count", 0), 0)
    if total_seconds <= 0 and candidate_count <= 0:
        return

    st.markdown("### Engine tuning suggestion timing")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Tuning test", f"{total_seconds:.2f}s")
    with c2:
        st.metric("Candidates tested", int(candidate_count))
    with c3:
        st.metric("Passed gate", int(accepted_count))

    st.caption(
        "This is the extra time used by the automatic engine-tuning suggestion test. "
        "It is separate from the main portfolio engine run and the preset suggestion test."
    )

    rows = timing.get("candidate_seconds", [])
    if isinstance(rows, list) and rows:
        clean_rows = []
        for row in rows:
            row_map = _coerce_mapping(row)
            clean_rows.append(
                {
                    "candidate": str(row_map.get("candidate", "Candidate") or "Candidate"),
                    "status": _display_candidate_status(row_map.get("status", "")),
                    "seconds": _safe_float(row_map.get("seconds", 0.0), 0.0),
                }
            )
        if clean_rows:
            with st.container(border=True):
                st.caption("Engine tuning timing breakdown")
                st.dataframe(pd.DataFrame(clean_rows), use_container_width=True, hide_index=True)


def _clear_auto_opt_suggestion_state() -> None:
    """Avoid showing stale phase-2 results while phase 1 is still pending."""
    st.session_state[AUTO_OPT_SUGGESTION_STATE_KEY] = {}
    st.session_state[AUTO_OPT_SUGGESTION_SCOPE_KEY] = ""
    st.session_state[AUTO_OPT_SUGGESTION_TIMING_KEY] = {}
    st.session_state[AUTO_OPT_APPLIED_SIGNATURE_KEY] = ""
    st.session_state[AUTO_OPT_APPLIED_LABEL_KEY] = ""
    st.session_state[AUTO_OPT_SKIPPED_SCOPE_KEY] = ""
    st.session_state[AUTO_OPT_SKIPPED_LABEL_KEY] = ""
    st.session_state[AUTO_OPT_SKIPPED_RUN_SIGNATURE_KEY] = ""
    _clear_universe_suggestion_state()


def _clear_universe_suggestion_state() -> None:
    """Avoid showing stale phase-3 results while earlier phases are unresolved."""
    st.session_state[UNIVERSE_SUGGESTION_STATE_KEY] = {}
    st.session_state[UNIVERSE_SUGGESTION_SCOPE_KEY] = ""
    st.session_state[UNIVERSE_SUGGESTION_TIMING_KEY] = {}
    st.session_state[UNIVERSE_APPLIED_SIGNATURE_KEY] = ""
    st.session_state[UNIVERSE_APPLIED_LABEL_KEY] = ""
    st.session_state[UNIVERSE_SKIPPED_SCOPE_KEY] = ""
    st.session_state[UNIVERSE_SKIPPED_LABEL_KEY] = ""
    st.session_state[UNIVERSE_SKIPPED_RUN_SIGNATURE_KEY] = ""


def _preset_blocks_engine_tuning(preset_flow_state: Any) -> bool:
    return bool(_coerce_mapping(preset_flow_state).get("blocks_auto_opt", False))


def _auto_opt_applied_for_current_run(run_map: dict) -> bool:
    # Earlier phases can promote a new Step 5 run result, which changes the run
    # signature. Keep the wizard state tied to the stored decision, not only to
    # the latest promoted signature; otherwise completed phases jump back to
    # Active/Locked after a later Apply action.
    return bool(
        str(st.session_state.get(AUTO_OPT_APPLIED_LABEL_KEY, "") or "")
        or str(st.session_state.get(AUTO_OPT_APPLIED_SIGNATURE_KEY, "") or "")
    )


def _universe_applied_for_current_run(run_map: dict) -> bool:
    return bool(
        str(st.session_state.get(UNIVERSE_APPLIED_LABEL_KEY, "") or "")
        or str(st.session_state.get(UNIVERSE_APPLIED_SIGNATURE_KEY, "") or "")
    )


def _auto_opt_skipped_for_current_run(run_map: dict) -> bool:
    return bool(
        str(st.session_state.get(AUTO_OPT_SKIPPED_SCOPE_KEY, "") or "")
        or str(st.session_state.get(AUTO_OPT_SKIPPED_LABEL_KEY, "") or "")
        or str(st.session_state.get(AUTO_OPT_SKIPPED_RUN_SIGNATURE_KEY, "") or "")
    )


def _universe_skipped_for_current_run(run_map: dict) -> bool:
    return bool(
        str(st.session_state.get(UNIVERSE_SKIPPED_SCOPE_KEY, "") or "")
        or str(st.session_state.get(UNIVERSE_SKIPPED_LABEL_KEY, "") or "")
        or str(st.session_state.get(UNIVERSE_SKIPPED_RUN_SIGNATURE_KEY, "") or "")
    )


def _universe_decision_completed_for_current_run(run_map: dict) -> bool:
    return bool(_universe_applied_for_current_run(run_map) or _universe_skipped_for_current_run(run_map))


def _size_applied_for_current_run(run_map: dict) -> bool:
    return bool(
        str(st.session_state.get(SIZE_APPLIED_LABEL_KEY, "") or "")
        or str(st.session_state.get(SIZE_APPLIED_SIGNATURE_KEY, "") or "")
    )


def _size_skipped_for_current_run(run_map: dict) -> bool:
    return bool(
        str(st.session_state.get(SIZE_SKIPPED_SCOPE_KEY, "") or "")
        or str(st.session_state.get(SIZE_SKIPPED_LABEL_KEY, "") or "")
        or str(st.session_state.get(SIZE_SKIPPED_RUN_SIGNATURE_KEY, "") or "")
    )


def _size_decision_completed_for_current_run(run_map: dict) -> bool:
    return bool(_size_applied_for_current_run(run_map) or _size_skipped_for_current_run(run_map))


def _preset_applied_for_current_run(run_map: dict) -> bool:
    return bool(
        str(st.session_state.get(PRESET_APPLIED_LABEL_KEY, "") or "")
        or str(st.session_state.get(PRESET_APPLIED_SIGNATURE_KEY, "") or "")
    )


def _preset_skipped_for_current_run(run_map: dict) -> bool:
    return bool(
        str(st.session_state.get(PRESET_SKIPPED_SCOPE_KEY, "") or "")
        or str(st.session_state.get(PRESET_SKIPPED_LABEL_KEY, "") or "")
    )


def _preset_decision_completed_for_current_run(run_map: dict) -> bool:
    return bool(_preset_applied_for_current_run(run_map) or _preset_skipped_for_current_run(run_map))


def _phase_state(*, applied: bool, skipped: bool, active: bool) -> tuple[str, str]:
    """Resolve the visual state shown in the Step 5 improvement rail."""
    if applied:
        return "applied", "Applied"
    if skipped:
        return "skipped", "Skipped"
    if active:
        return "active", "Active"
    return "locked", "Locked"


def _render_improvement_phase_row(run_map: dict) -> None:
    """Render the sequential improvement flow as a compact phase rail.

    Applied phases are green, skipped phases are amber, active phases are blue,
    and locked phases are faded. This avoids pretending that a skipped phase was
    successfully optimised while still showing that the wizard can continue.
    """
    preset_applied = _preset_applied_for_current_run(run_map)
    preset_skipped = _preset_skipped_for_current_run(run_map)

    tuning_applied = _auto_opt_applied_for_current_run(run_map)
    tuning_skipped = _auto_opt_skipped_for_current_run(run_map)

    universe_applied = _universe_applied_for_current_run(run_map)
    universe_skipped = _universe_skipped_for_current_run(run_map)

    size_applied = _size_applied_for_current_run(run_map)
    size_skipped = _size_skipped_for_current_run(run_map)

    # Defensive dependency resolution: if a later phase has already been resolved,
    # earlier phases must not appear active/locked even if an old session-state key
    # was cleared by a rerun. This keeps the rail coherent after Apply/Skip actions.
    size_done = bool(size_applied or size_skipped)
    universe_done = bool(universe_applied or universe_skipped or size_done)
    tuning_done = bool(tuning_applied or tuning_skipped or universe_done)
    preset_done = bool(preset_applied or preset_skipped or tuning_done)

    effective_preset_applied = bool(preset_applied or (preset_done and not preset_skipped))
    effective_tuning_applied = bool(tuning_applied or (tuning_done and not tuning_skipped))
    effective_universe_applied = bool(universe_applied or (universe_done and not universe_skipped))

    phase_specs = [
        {"num": "1", "label": "Strategy preset", "caption": "Template/style", "applied": effective_preset_applied, "skipped": preset_skipped, "active": not preset_done},
        {"num": "2", "label": "Engine tuning", "caption": "Technical knobs", "applied": effective_tuning_applied, "skipped": tuning_skipped, "active": preset_done and not tuning_done},
        {"num": "3", "label": "Universe mix", "caption": "Same-size assets", "applied": effective_universe_applied, "skipped": universe_skipped, "active": tuning_done and not universe_done},
        {"num": "4", "label": "Universe size", "caption": "Basket breadth", "applied": size_applied, "skipped": size_skipped, "active": universe_done and not size_done},
    ]

    phases = []
    for spec in phase_specs:
        state, status = _phase_state(applied=bool(spec["applied"]), skipped=bool(spec["skipped"]), active=bool(spec["active"]))
        phases.append({**spec, "state": state, "status": status})

    def _card_html(phase: dict) -> str:
        state = str(phase.get("state", "locked"))
        aria_disabled = "true" if state == "locked" else "false"
        return (
            f'<div class="lb-phase-card lb-phase-{state}" aria-disabled="{aria_disabled}">'
            f'<div class="lb-phase-topline">'
            f'<span>Phase {phase.get("num", "")}</span>'
            f'<span class="lb-phase-status">{phase.get("status", "")}</span>'
            f'</div>'
            f'<div class="lb-phase-title">{phase.get("label", "")}</div>'
            f'<div class="lb-phase-caption">{phase.get("caption", "")}</div>'
            f'</div>'
        )

    cards = "".join(_card_html(phase) for phase in phases)
    st.markdown(
        f"""
        <style>
            .lb-phase-rail {{
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 0.85rem;
                margin: 0.6rem 0 1.25rem 0;
            }}
            .lb-phase-card {{
                border: 1px solid rgba(49, 51, 63, 0.16);
                border-radius: 0.75rem;
                padding: 0.95rem 1rem;
                min-height: 7.25rem;
                background: #ffffff;
                box-shadow: 0 1px 2px rgba(49, 51, 63, 0.04);
            }}
            .lb-phase-topline {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 0.5rem;
                color: rgba(49, 51, 63, 0.62);
                font-size: 0.86rem;
                margin-bottom: 0.75rem;
            }}
            .lb-phase-status {{
                border-radius: 999px;
                padding: 0.18rem 0.55rem;
                font-size: 0.76rem;
                background: rgba(49, 51, 63, 0.06);
                white-space: nowrap;
            }}
            .lb-phase-title {{
                font-size: 1.05rem;
                font-weight: 700;
                color: rgb(49, 51, 63);
                margin-bottom: 0.65rem;
            }}
            .lb-phase-caption {{
                color: rgba(49, 51, 63, 0.60);
                font-size: 0.88rem;
                line-height: 1.35;
            }}
            .lb-phase-active {{
                border-color: rgba(49, 101, 195, 0.55);
                background: rgba(236, 242, 255, 0.95);
            }}
            .lb-phase-active .lb-phase-status {{
                color: rgb(45, 86, 166);
                background: rgba(49, 101, 195, 0.12);
                font-weight: 700;
            }}
            .lb-phase-applied {{
                border-color: rgba(65, 135, 54, 0.28);
                background: rgba(240, 249, 238, 0.92);
            }}
            .lb-phase-applied .lb-phase-status {{
                color: rgb(60, 122, 48);
                background: rgba(65, 135, 54, 0.12);
                font-weight: 700;
            }}
            .lb-phase-skipped {{
                border-color: rgba(153, 111, 16, 0.34);
                background: rgba(255, 248, 224, 0.82);
            }}
            .lb-phase-skipped .lb-phase-status {{
                color: rgb(125, 88, 8);
                background: rgba(153, 111, 16, 0.13);
                font-weight: 700;
            }}
            .lb-phase-locked {{
                opacity: 0.48;
                background: rgba(249, 250, 252, 0.72);
                box-shadow: none;
            }}
            .lb-phase-locked .lb-phase-status {{ color: rgba(49, 51, 63, 0.58); }}
            @media (max-width: 800px) {{ .lb-phase-rail {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
            @media (max-width: 520px) {{ .lb-phase-rail {{ grid-template-columns: 1fr; }} }}
        </style>
        <div class="lb-phase-rail">{cards}</div>
        """,
        unsafe_allow_html=True,
    )

def _auto_opt_blocks_universe(run_map: dict) -> bool:
    """Return True when phase 2 has a pending accepted candidate.

    render_auto_opt_improvement() owns the UI and state creation. This helper is
    deliberately state-based so phase 3 can remain sequential without changing
    the phase-2 public function signature.
    """
    if _auto_opt_applied_for_current_run(run_map) or _auto_opt_skipped_for_current_run(run_map):
        return False

    payload = _coerce_mapping(st.session_state.get(AUTO_OPT_SUGGESTION_STATE_KEY, {}))
    evaluations = list(payload.get("evaluations", []) or [])
    accepted_items = [x for x in evaluations if bool(_coerce_mapping(x).get("accepted", False))]
    return bool(accepted_items)


def _render_universe_waiting_for_engine_tuning() -> None:
    """Keep later locked phases quiet; the phase rail already explains the sequence."""
    _clear_universe_suggestion_state()
    return

def _render_universe_suggestion_timing_block() -> None:
    timing = _coerce_mapping(st.session_state.get(UNIVERSE_SUGGESTION_TIMING_KEY, {}))
    if not timing:
        return

    total_seconds = _safe_float(timing.get("total_seconds", 0.0), 0.0)
    candidate_count = _safe_int(timing.get("candidate_count", 0), 0)
    accepted_count = _safe_int(timing.get("accepted_count", 0), 0)
    if total_seconds <= 0 and candidate_count <= 0:
        return

    st.markdown("### Universe suggestion timing")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Universe test", f"{total_seconds:.2f}s")
    with c2:
        st.metric("Candidates tested", int(candidate_count))
    with c3:
        st.metric("Passed gate", int(accepted_count))

    st.caption(
        "This is the extra time used by the automatic universe-mix suggestion. "
        "It tests only a small number of same-size asset compositions from the selected market-data panel."
    )

    candidate_rows = timing.get("candidate_seconds", [])
    if isinstance(candidate_rows, list) and candidate_rows:
        clean_rows = []
        for row in candidate_rows:
            row_map = _coerce_mapping(row)
            clean_rows.append(
                {
                    "candidate": str(row_map.get("candidate", "Candidate") or "Candidate"),
                    "status": _display_candidate_status(row_map.get("status", "")),
                    "seconds": _safe_float(row_map.get("seconds", 0.0), 0.0),
                }
            )
        if clean_rows:
            with st.container(border=True):
                st.caption("Universe suggestion timing breakdown")
                st.dataframe(pd.DataFrame(clean_rows), use_container_width=True, hide_index=True)


def _render_size_suggestion_timing_block() -> None:
    timing = _coerce_mapping(st.session_state.get(SIZE_SUGGESTION_TIMING_KEY, {}))
    if not timing:
        return

    total_seconds = _safe_float(timing.get("total_seconds", 0.0), 0.0)
    candidate_count = _safe_int(timing.get("candidate_count", 0), 0)
    planned_count = _safe_int(timing.get("planned_candidate_count", candidate_count), candidate_count)
    accepted_count = _safe_int(timing.get("accepted_count", 0), 0)
    if total_seconds <= 0 and planned_count <= 0:
        return

    st.markdown("### Universe size suggestion timing")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Size test", f"{total_seconds:.2f}s")
    with c2:
        st.metric("Engine-tested sizes", int(candidate_count))
    with c3:
        st.metric("Planned sizes", int(planned_count))
    with c4:
        st.metric("Passed gate", int(accepted_count))

    st.caption(
        "This is the extra time used by the automatic universe-size suggestion. "
        "It tests coarse/refinement sizes after the universe-mix decision has been resolved."
    )

    candidate_rows = timing.get("candidate_seconds", [])
    if isinstance(candidate_rows, list) and candidate_rows:
        clean_rows = []
        for row in candidate_rows:
            row_map = _coerce_mapping(row)
            clean_rows.append(
                {
                    "candidate": str(row_map.get("candidate", "Candidate") or "Candidate"),
                    "status": _display_candidate_status(row_map.get("status", "")),
                    "seconds": _safe_float(row_map.get("seconds", 0.0), 0.0),
                }
            )
        if clean_rows:
            with st.container(border=True):
                st.caption("Universe size suggestion timing breakdown")
                st.dataframe(pd.DataFrame(clean_rows), use_container_width=True, hide_index=True)


def _render_completed_improvement_timing_summary() -> None:
    """Keep historical suggestion timing available without making it part of the main page."""
    preset_timing = _coerce_mapping(st.session_state.get(PRESET_SUGGESTION_TIMING_KEY, {}))
    tuning_timing = _coerce_mapping(st.session_state.get(AUTO_OPT_SUGGESTION_TIMING_KEY, {}))
    universe_timing = _coerce_mapping(st.session_state.get(UNIVERSE_SUGGESTION_TIMING_KEY, {}))
    size_timing = _coerce_mapping(st.session_state.get(SIZE_SUGGESTION_TIMING_KEY, {}))

    preset_total = _safe_float(preset_timing.get("total_seconds", 0.0), 0.0)
    tuning_total = _safe_float(tuning_timing.get("total_seconds", 0.0), 0.0)
    universe_total = _safe_float(universe_timing.get("total_seconds", 0.0), 0.0)
    size_total = _safe_float(size_timing.get("total_seconds", 0.0), 0.0)
    total = float(max(0.0, preset_total) + max(0.0, tuning_total) + max(0.0, universe_total) + max(0.0, size_total))
    if total <= 0.0:
        return

    with st.expander("Completed suggestion timing diagnostics", expanded=False):
        st.caption(
            "Historical timing only: these suggestion tests were not re-run after Apply; "
            "the current result was promoted from the previously tested candidate."
        )

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Preset test", f"{preset_total:.2f}s" if preset_total > 0 else "—")
        with c2:
            st.metric("Engine tuning test", f"{tuning_total:.2f}s" if tuning_total > 0 else "—")
        with c3:
            st.metric("Universe test", f"{universe_total:.2f}s" if universe_total > 0 else "—")
        with c4:
            st.metric("Size test", f"{size_total:.2f}s" if size_total > 0 else "—")
        with c5:
            st.metric("Total overhead", f"{total:.2f}s")

        preset_candidates = _safe_int(preset_timing.get("candidate_count", 0), 0)
        preset_accepted = _safe_int(preset_timing.get("accepted_count", 0), 0)
        tuning_candidates = _safe_int(tuning_timing.get("candidate_count", 0), 0)
        tuning_accepted = _safe_int(tuning_timing.get("accepted_count", 0), 0)
        universe_candidates = _safe_int(universe_timing.get("candidate_count", 0), 0)
        universe_accepted = _safe_int(universe_timing.get("accepted_count", 0), 0)
        size_candidates = _safe_int(size_timing.get("candidate_count", 0), 0)
        size_accepted = _safe_int(size_timing.get("accepted_count", 0), 0)
        st.caption(
            f"Preset candidates={preset_candidates}, passed_gate={preset_accepted} · "
            f"Engine tuning candidates={tuning_candidates}, passed_gate={tuning_accepted} · "
            f"Universe candidates={universe_candidates}, passed_gate={universe_accepted} · "
            f"Size candidates={size_candidates}, passed_gate={size_accepted}. "
            "Only the highest-scoring passed-gate candidate was promoted."
        )

        rows = []
        for phase_name, timing in [
            ("Preset suggestion", preset_timing),
            ("Engine tuning", tuning_timing),
            ("Universe composition", universe_timing),
            ("Universe size", size_timing),
        ]:
            candidate_rows = timing.get("candidate_seconds", [])
            if not isinstance(candidate_rows, list):
                continue
            for row in candidate_rows:
                row_map = _coerce_mapping(row)
                rows.append(
                    {
                        "phase": phase_name,
                        "candidate": str(row_map.get("candidate", "Candidate") or "Candidate"),
                        "status": _display_candidate_status(row_map.get("status", "")),
                        "seconds": _safe_float(row_map.get("seconds", 0.0), 0.0),
                    }
                )

        if rows:
            st.markdown("**Candidate timing breakdown**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

def _render_completed_improvement_flow(run_map: dict) -> None:
    """Compact terminal state for the improvement wizard."""
    size_applied = _size_applied_for_current_run(run_map)
    size_skipped = _size_skipped_for_current_run(run_map)

    if size_applied:
        label = str(st.session_state.get(SIZE_APPLIED_LABEL_KEY, "") or "the accepted universe size suggestion")
        st.success(f"Improvement flow completed: {label}. The rerun-tested candidate is now the current strategy engine result.")
    elif size_skipped:
        label = str(st.session_state.get(SIZE_SKIPPED_LABEL_KEY, "") or "the recommended universe size")
        st.warning(f"Universe size skipped: current universe size kept. Skipped recommendation: {label}.")
    else:
        st.info("Improvement flow completed for this run.")

    st.caption(
        "Change the strategy setup, technical controls, selected universe, or run a new baseline if you want to start a fresh improvement cycle."
    )


def _render_resolved_phase_history(run_map: dict) -> None:
    """Show completed improvement decisions once, without re-rendering old phase bodies."""
    decisions: list[str] = []

    if _preset_applied_for_current_run(run_map):
        label = str(st.session_state.get(PRESET_APPLIED_LABEL_KEY, "") or "preset improvement")
        decisions.append(f"Preset applied: {label}")
    elif _preset_skipped_for_current_run(run_map):
        label = str(st.session_state.get(PRESET_SKIPPED_LABEL_KEY, "") or "current preset kept")
        decisions.append(f"Preset skipped: {label}")

    if _auto_opt_applied_for_current_run(run_map):
        label = str(st.session_state.get(AUTO_OPT_APPLIED_LABEL_KEY, "") or "engine tuning improvement")
        decisions.append(f"Engine tuning applied: {label}")
    elif _auto_opt_skipped_for_current_run(run_map):
        label = str(st.session_state.get(AUTO_OPT_SKIPPED_LABEL_KEY, "") or "current tuning kept")
        decisions.append(f"Engine tuning skipped: {label}")

    if _universe_applied_for_current_run(run_map):
        label = str(st.session_state.get(UNIVERSE_APPLIED_LABEL_KEY, "") or "universe mix improvement")
        decisions.append(f"Universe mix applied: {label}")
    elif _universe_skipped_for_current_run(run_map):
        label = str(st.session_state.get(UNIVERSE_SKIPPED_LABEL_KEY, "") or "current universe kept")
        decisions.append(f"Universe mix skipped: {label}")

    if decisions:
        st.caption("Resolved so far: " + " · ".join(decisions))


def _render_engine_tuning_waiting_for_preset() -> None:
    """Keep later locked phases quiet; the phase rail already explains the sequence."""
    _clear_auto_opt_suggestion_state()
    return

def _result_interpretation(perf: dict) -> tuple[str, str]:
    cagr = _safe_float(perf.get("cagr", 0.0), 0.0)
    sharpe = _safe_float(perf.get("sharpe", 0.0), 0.0)
    vol = _safe_float(perf.get("annual_volatility", perf.get("volatility", 0.0)), 0.0)
    maxdd = abs(_safe_float(perf.get("max_drawdown", 0.0), 0.0))

    if sharpe >= 1.0:
        sharpe_note = "strong risk-adjusted performance"
    elif sharpe >= 0.60:
        sharpe_note = "reasonable risk-adjusted performance"
    elif sharpe > 0:
        sharpe_note = "positive but modest risk-adjusted performance"
    else:
        sharpe_note = "weak risk-adjusted performance"

    if maxdd >= 0.30:
        risk_note = "a large historical drawdown, so the strategy would require high tolerance for temporary losses"
    elif maxdd >= 0.15:
        risk_note = "a meaningful historical drawdown, so the user would need to tolerate uncomfortable periods"
    elif maxdd > 0:
        risk_note = "a contained historical drawdown relative to more aggressive portfolios"
    else:
        risk_note = "no meaningful drawdown recorded in the available summary"

    headline = "Result interpretation"
    body = (
        f"This run shows {sharpe_note}: CAGR was {_pct(cagr)}, volatility was {_pct(vol)}, "
        f"and maximum drawdown was -{100.0 * maxdd:.2f}%. The main trade-off is {risk_note}."
    )
    return headline, body


def _philosophy_fit_message(perf: dict, philosophy: Any) -> tuple[str, str]:
    profile = str(philosophy or "Balanced").strip() or "Balanced"
    profile_key = profile.lower()
    cagr = _safe_float(perf.get("cagr", 0.0), 0.0)
    sharpe = _safe_float(perf.get("sharpe", 0.0), 0.0)
    vol = _safe_float(perf.get("annual_volatility", perf.get("volatility", 0.0)), 0.0)
    maxdd = abs(_safe_float(perf.get("max_drawdown", 0.0), 0.0))

    if profile_key == "defensive":
        if maxdd >= 0.18 or vol >= 0.13:
            return (
                "warning",
                f"For a Defensive profile, this result may be too uncomfortable: the historical drawdown is around -{100.0 * maxdd:.2f}% and volatility is {_pct(vol)}. The return may be useful, but the risk profile deserves caution.",
            )
        return (
            "success",
            f"For a Defensive profile, this looks relatively aligned: the drawdown is around -{100.0 * maxdd:.2f}% and volatility is {_pct(vol)}, so the setup appears more controlled than aggressive.",
        )

    if profile_key == "growth":
        if cagr >= 0.08 and sharpe >= 0.50:
            return (
                "success",
                f"For a Growth profile, this looks directionally aligned: CAGR is {_pct(cagr)} and the strategy accepts some volatility in exchange for higher upside potential.",
            )
        return (
            "info",
            f"For a Growth profile, the result is usable but not clearly aggressive: CAGR is {_pct(cagr)} and Sharpe is {sharpe:.2f}. You may want to check whether the setup is too defensive or too diversified.",
        )

    # Balanced / default
    style = str(st.session_state.get("step5_style", "") or "").strip()
    style_note = ""
    if style in {"Conservative", "Defensive"}:
        style_note = f" The applied {style} style makes the setup slightly more risk-controlled, but it is still not a fully Defensive mother philosophy."
    elif style and style != "Balanced":
        style_note = f" The current {style} style changes the posture inside the Balanced mother philosophy."

    if maxdd >= 0.25:
        return (
            "warning",
            f"For a Balanced mother philosophy, the return profile may be attractive, but a drawdown around -{100.0 * maxdd:.2f}% is high enough to question whether the setup still feels balanced.{style_note}",
        )
    if sharpe >= 0.60 and cagr > 0:
        return (
            "success",
            f"For a Balanced mother philosophy, this remains broadly aligned if the user can tolerate a drawdown near -{100.0 * maxdd:.2f}%. It is growth-positive, but not fully defensive.{style_note}",
        )
    return (
        "info",
        f"For a Balanced mother philosophy, this result needs review: CAGR is {_pct(cagr)}, Sharpe is {sharpe:.2f}, and drawdown is around -{100.0 * maxdd:.2f}%.{style_note}",
    )


def _render_metric_explainer(perf: dict) -> None:
    cagr = _safe_float(perf.get("cagr", 0.0), 0.0)
    sharpe = _safe_float(perf.get("sharpe", 0.0), 0.0)
    vol = _safe_float(perf.get("annual_volatility", perf.get("volatility", 0.0)), 0.0)
    maxdd = abs(_safe_float(perf.get("max_drawdown", 0.0), 0.0))

    with st.expander("How to read these metrics", expanded=False):
        st.markdown(
            f"- **CAGR ({_pct(cagr)})** — historical average annual growth in this backtest. Higher is better, but it is not guaranteed.\n"
            f"- **Volatility ({_pct(vol)})** — how bumpy the portfolio was historically. Lower usually feels more stable.\n"
            f"- **MaxDD (-{100.0 * maxdd:.2f}%)** — worst historical peak-to-trough fall. This is the main pain-test metric.\n"
            f"- **Sharpe ({sharpe:.2f})** — return per unit of risk. Higher usually means the return compensated better for volatility."
        )



def _decision_guide_message(perf: dict, philosophy: Any) -> str:
    """Return the single post-run decision guide tailored to the selected risk profile."""
    profile = str(philosophy or "Balanced").strip() or "Balanced"
    profile_key = profile.lower()
    cagr = _safe_float(perf.get("cagr", 0.0), 0.0)
    sharpe = _safe_float(perf.get("sharpe", 0.0), 0.0)
    vol = _safe_float(perf.get("annual_volatility", perf.get("volatility", 0.0)), 0.0)
    maxdd = abs(_safe_float(perf.get("max_drawdown", 0.0), 0.0))

    if profile_key in {"defensive", "conservative"}:
        philosophy_rule = (
            "For a Defensive/Conservative profile, prioritise lower drawdown and lower volatility, "
            "but do not accept a change that weakens Sharpe so much that the smoother path stops being worth it."
        )
    elif profile_key == "growth":
        philosophy_rule = (
            "For a Growth profile, the useful improvement is better Sharpe or lower drawdown without killing upside; "
            "do not accept a safer-looking rerun if it removes too much of the CAGR case."
        )
    else:
        philosophy_rule = (
            "For a Balanced profile, judge the full trade-off: lower drawdown is useful, but not if Sharpe or CAGR "
            "falls enough to make the strategy less balanced overall."
        )

    return (
        "**Decision guide:** only apply a suggestion if the rerun-tested result improves the decision trade-off, "
        "not just one isolated metric. "
        f"Current result: CAGR {_pct(cagr)}, volatility {_pct(vol)}, MaxDD -{100.0 * maxdd:.2f}%, "
        f"Sharpe {sharpe:.2f}. {philosophy_rule} "
        "If this result already feels acceptable, the next step is to continue to the Long-Term Scenario; "
        "benchmark, reliability, and improvement checks are optional review layers."
    )


def _render_what_to_watch(perf: dict, philosophy: Any, *, as_expander: bool = True) -> None:
    profile = str(philosophy or "Balanced").strip() or "Balanced"
    cagr = _safe_float(perf.get("cagr", 0.0), 0.0)
    sharpe = _safe_float(perf.get("sharpe", 0.0), 0.0)
    vol = _safe_float(perf.get("annual_volatility", perf.get("volatility", 0.0)), 0.0)
    maxdd = abs(_safe_float(perf.get("max_drawdown", 0.0), 0.0))

    watch_items = []
    if cagr < 0.05:
        watch_items.append("If CAGR feels too low, the setup may be too defensive, too diversified, or using weak signals for this universe.")
    else:
        watch_items.append("CAGR is positive; the next question is whether the drawdown and volatility are acceptable for the selected philosophy.")
    if sharpe < 0.50:
        watch_items.append("Sharpe is modest; the portfolio may not be earning enough return for the risk it takes.")
    else:
        watch_items.append("Sharpe is usable; the key question is whether any suggested change improves the trade-off enough to justify extra engine complexity.")
    if maxdd >= 0.18:
        watch_items.append("MaxDD is the key risk flag: a drawdown near 20% can be psychologically hard even if the long-run return is positive.")
    if vol >= 0.15:
        watch_items.append("Volatility is elevated; reduce risk appetite or increase drawdown protection if the strategy should feel smoother.")
    if profile.lower() == "defensive" and (maxdd >= 0.15 or vol >= 0.12):
        watch_items.append("Because the selected philosophy is Defensive, prioritise drawdown and volatility over chasing higher CAGR.")
    elif profile.lower() == "growth":
        watch_items.append("Because the selected philosophy is Growth, some volatility is acceptable, but Sharpe should still justify the risk.")

    def _body() -> None:
        st.markdown("**What to watch before changing the strategy**")
        st.markdown("\n".join(f"- {item}" for item in watch_items))
        st.caption(
            "Use benchmark context as external reference only; the most direct comparison is still between rerun-tested internal configurations."
        )

    if as_expander:
        with st.expander("What to watch before changing the strategy", expanded=False):
            _body()
    else:
        _body()


def _current_engine_knobs(run_map: dict) -> dict:
    """Resolve the active low-level engine settings for the explanatory lever block."""
    for raw in (
        run_map.get("config_dict"),
        run_map.get("config"),
        st.session_state.get("step5_last_cfg_final"),
        st.session_state.get("last_engine_config"),
    ):
        payload = _coerce_mapping(raw)
        if payload:
            return payload
    return {}


def _fmt_knob(value: Any) -> str:
    if value is None or value == "":
        return "—"
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return f"{number:.3f}".rstrip("0").rstrip(".")
    except Exception:
        return str(value)


def _engine_improvement_target(perf: dict, philosophy: Any) -> str:
    profile = str(philosophy or "Balanced").strip() or "Balanced"
    cagr = _safe_float(perf.get("cagr", 0.0), 0.0)
    sharpe = _safe_float(perf.get("sharpe", 0.0), 0.0)
    vol = _safe_float(perf.get("annual_volatility", perf.get("volatility", 0.0)), 0.0)
    maxdd = abs(_safe_float(perf.get("max_drawdown", 0.0), 0.0))

    if profile.lower() == "defensive":
        if maxdd >= 0.15 or vol >= 0.12:
            return (
                "For this Defensive setup, the first improvement target is a smoother risk profile: "
                "try reducing drawdown and volatility before chasing extra CAGR."
            )
        return (
            "For this Defensive setup, the risk profile is already relatively controlled; the useful test is whether "
            "Sharpe can improve without letting drawdown drift upward."
        )

    if profile.lower() == "growth":
        if cagr < 0.08:
            return (
                "For this Growth setup, the useful improvement target is higher CAGR without a large Sharpe or drawdown penalty."
            )
        return (
            "For this Growth setup, growth is already present; the useful test is whether Sharpe can improve without removing too much upside."
        )

    if maxdd >= 0.18 and sharpe >= 0.60:
        return (
            "For this Balanced setup, the main opportunity is the drawdown/Sharpe trade-off, not simply raw CAGR: "
            "try reducing MaxDD toward the high-teens while keeping Sharpe near or above the current level."
        )
    if sharpe < 0.50:
        return (
            "For this Balanced setup, the main opportunity is signal quality: improve Sharpe before making the strategy more aggressive."
        )
    if cagr < 0.05:
        return (
            "For this Balanced setup, the main opportunity is return capture: test whether the engine is too defensive or too diluted."
        )
    return (
        "For this Balanced setup, the useful improvement target is incremental: improve Sharpe or drawdown without materially increasing volatility."
    )



def _render_engine_levers_to_try(perf: dict, philosophy: Any, run_map: dict, *, as_expander: bool = True) -> None:
    """Explain which engine controls map to the issues flagged by the result."""
    cfg = _current_engine_knobs(run_map)
    cagr = _safe_float(perf.get("cagr", 0.0), 0.0)
    sharpe = _safe_float(perf.get("sharpe", 0.0), 0.0)
    vol = _safe_float(perf.get("annual_volatility", perf.get("volatility", 0.0)), 0.0)
    maxdd = abs(_safe_float(perf.get("max_drawdown", 0.0), 0.0))

    rows = [
        {
            "Goal": "Broader diversification",
            "Engine levers": "top_k, temperature, weight_shrink",
            "How to test it": "Increase top_k or temperature slightly; increase weight_shrink to reduce concentration.",
            "Trade-off": "Can reduce drawdown, but may dilute strong signals and lower CAGR.",
        },
        {
            "Goal": "Smoother risk estimates",
            "Engine levers": "lookback_mu, lookback_sigma",
            "How to test it": "Use longer lookbacks so return/risk estimates react less to short-term noise.",
            "Trade-off": "Usually smoother, but slower to adapt after market regime changes.",
        },
        {
            "Goal": "Lower turnover / more stability",
            "Engine levers": "inertia",
            "How to test it": "Increase inertia so the engine changes allocation more gradually.",
            "Trade-off": "Can make the path steadier, but may react late to new information.",
        },
        {
            "Goal": "Signal quality check",
            "Engine levers": "signal_mode, feature_mu_enabled, feature_mu_blend",
            "How to test it": "Compare the current signal setup against a slightly simpler or feature-assisted signal configuration.",
            "Trade-off": "Can improve Sharpe if signals help, but extra features can overfit if the sample is weak.",
        },
    ]

    if maxdd >= 0.18 or vol >= 0.15:
        suggested_direction = (
            "Sensible first test: slightly broader selection, longer lookbacks, more weight shrink, and more inertia. "
            "That is the most direct way to test whether the current drawdown can be reduced without destroying Sharpe."
        )
    elif sharpe < 0.50:
        suggested_direction = (
            "Sensible first test: keep risk roughly stable and test signal/lookback changes before increasing risk appetite."
        )
    elif cagr < 0.05:
        suggested_direction = (
            "Sensible first test: check whether the strategy is too defensive or too diluted before changing the universe."
        )
    else:
        suggested_direction = (
            "Sensible first test: make small technical changes only, then accept them only if the real rerun improves the trade-off."
        )

    def _body() -> None:
        st.markdown("**Engine levers considered by the improvement checks**")
        st.markdown(f"**Improvement target:** {_engine_improvement_target(perf, philosophy)}")
        st.markdown(f"**Suggested first test:** {suggested_direction}")

        current_bits = [
            f"top_k={_fmt_knob(cfg.get('top_k'))}",
            f"lookback_mu={_fmt_knob(cfg.get('lookback_mu'))}",
            f"lookback_sigma={_fmt_knob(cfg.get('lookback_sigma'))}",
            f"temperature={_fmt_knob(cfg.get('temperature'))}",
            f"weight_shrink={_fmt_knob(cfg.get('weight_shrink'))}",
            f"inertia={_fmt_knob(cfg.get('inertia'))}",
            f"signal_mode={_fmt_knob(cfg.get('signal_mode'))}",
        ]
        st.caption("Current engine levers: " + " · ".join(current_bits))
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if as_expander:
        with st.expander("Improvement test diagnostics", expanded=False):
            st.caption(
                "Optional diagnostic: shows which engine controls the improvement checks may adjust. "
                "Suggestions below are still rerun with the real engine before they can be applied."
            )
            _body()
    else:
        _body()


def _render_reliability_note() -> None:
    with st.expander("Reliability note and limits", expanded=False):
        st.info(
            "These figures come from a historical walk-forward backtest using the selected asset panel. "
            "They are useful for comparing configurations inside the app, but they are not forecasts or guarantees. "
            "Results depend on the date range, asset universe, data quality, and engine assumptions. Long-Term Scenario should be used to explore future uncertainty rather than treating this run as a prediction."
        )


BENCHMARK_CONTEXT_ASSETS = [
    {"ticker": "SPY", "reference": "SPY — S&P 500", "type": "US large-cap equities", "reading": "Equity-growth reference"},
    {"ticker": "QQQ", "reference": "QQQ — Nasdaq-100", "type": "Growth / technology-heavy equities", "reading": "Higher-growth / higher-volatility reference"},
    {"ticker": "GLD", "reference": "GLD — Gold", "type": "Gold / alternative diversifier", "reading": "Defensive diversifier / inflation hedge reference"},
    {"ticker": "AGG", "reference": "AGG — US aggregate bonds", "type": "Diversified bonds", "reading": "Defensive bond reference"},
    {"ticker": "TLT", "reference": "TLT — Long-term US Treasuries", "type": "Long-duration Treasury bonds", "reading": "Rate-sensitive defensive reference"},
    {"ticker": "IEF", "reference": "IEF — 7–10 year US Treasuries", "type": "Intermediate Treasury bonds", "reading": "Intermediate defensive bond reference"},
    {"ticker": "VNQ", "reference": "VNQ — US real estate", "type": "US REITs / real estate", "reading": "Real-assets / income-oriented reference"},
]

SHORTER_HISTORY_CONTEXT = [
    "DBC / USO — commodities and oil ETFs usually start after 2005 in Yahoo ETF history",
    "BND — total bond market ETF starts after the 2005 window used here",
    "BTC-USD — crypto history is much shorter and would require a separate shorter-window comparison",
]


def _format_pct_signed(value: Any, decimals: int = 2) -> str:
    try:
        return f"{100.0 * float(value):.{decimals}f}%"
    except Exception:
        return "—"


def _safe_return_series(raw: pd.Series) -> pd.Series:
    series = pd.to_numeric(raw, errors="coerce").dropna().astype(float)
    if series.empty:
        return series
    try:
        if float(series.abs().median()) > 1.0:
            series = series / 100.0
    except Exception:
        pass
    return series


def _compute_return_metrics(returns: pd.Series, *, periods_per_year: int = 12) -> dict:
    r = _safe_return_series(returns)
    n = int(len(r))
    if n <= 1:
        return {}

    wealth = (1.0 + r).cumprod()
    final_wealth = float(wealth.iloc[-1]) if len(wealth) else 0.0
    years = float(n) / float(periods_per_year)
    cagr = float(final_wealth ** (1.0 / years) - 1.0) if final_wealth > 0.0 and years > 0.0 else 0.0
    vol = float(r.std(ddof=1) * (periods_per_year ** 0.5)) if n > 1 else 0.0
    ann_mean = float(r.mean() * periods_per_year) if n > 0 else 0.0
    sharpe = float(ann_mean / vol) if vol > 1e-12 else 0.0

    peak = wealth.cummax()
    drawdown = (wealth / peak) - 1.0
    maxdd = float(drawdown.min()) if len(drawdown) else 0.0

    return {"cagr": cagr, "annual_volatility": vol, "sharpe": sharpe, "max_drawdown": maxdd, "periods": n}


def _strategy_context_zone(perf: dict, philosophy: Any) -> str:
    profile = str(philosophy or "Balanced").strip() or "Balanced"
    cagr = _safe_float(perf.get("cagr", 0.0), 0.0)
    vol = _safe_float(perf.get("annual_volatility", perf.get("volatility", 0.0)), 0.0)
    maxdd = abs(_safe_float(perf.get("max_drawdown", 0.0), 0.0))

    if vol >= 0.22 or maxdd >= 0.35:
        return "High-risk / speculative-like zone"
    if cagr >= 0.08 and vol >= 0.13:
        return "Growth-leaning zone"
    if cagr >= 0.05 and vol <= 0.16 and maxdd <= 0.25:
        return f"{profile} / multi-asset zone"
    if vol <= 0.09 and maxdd <= 0.15:
        return "Defensive / smoother-return zone"
    return f"{profile} context zone"


def _prepare_benchmark_panel() -> tuple[pd.DataFrame, list[str]]:
    panel = st.session_state.get("asset_panel_df")
    if not isinstance(panel, pd.DataFrame) or panel.empty:
        return pd.DataFrame(), ["Step 4 asset panel is not available in session_state."]

    required = {"date", "asset", "return"}
    if not required.issubset(set(map(str, panel.columns))):
        return pd.DataFrame(), ["Step 4 asset panel does not contain date, asset and return columns."]

    work = panel.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["asset"] = work["asset"].astype(str).str.upper().str.strip()
    work["return"] = pd.to_numeric(work["return"], errors="coerce")
    work = work.dropna(subset=["date", "asset", "return"]).sort_values(["asset", "date"])
    if work.empty:
        return pd.DataFrame(), ["Step 4 asset panel is empty after cleaning date, asset and return values."]
    return work, []


def _format_window(start_date: Any, end_date: Any, fallback: str = "—") -> str:
    try:
        start = pd.to_datetime(start_date, errors="coerce")
        end = pd.to_datetime(end_date, errors="coerce")
        if pd.notna(start) and pd.notna(end):
            return f"{start:%Y-%m} → {end:%Y-%m}"
    except Exception:
        pass
    return fallback


def _resolve_engine_evaluation_window(work: pd.DataFrame, target_periods: int) -> dict:
    """Infer the visible benchmark window from the OOS return length.

    The engine uses the full Step 4 panel as historical input, but the performance
    summary is based on walk-forward/OOS returns. The raw result currently exposes
    the OOS return series length but not always the exact OOS date vector, so this
    helper aligns benchmark context to the trailing monthly panel dates.
    """
    dates = pd.Series(pd.to_datetime(work.get("date"), errors="coerce")).dropna().drop_duplicates().sort_values()
    panel_start = dates.iloc[0] if len(dates) else None
    panel_end = dates.iloc[-1] if len(dates) else None
    total_periods = int(len(dates))
    target_periods = int(target_periods or 0)

    if target_periods > 0 and total_periods >= target_periods:
        eval_dates = dates.tail(target_periods)
        eval_start = eval_dates.iloc[0]
        eval_end = eval_dates.iloc[-1]
        warmup_periods = max(0, total_periods - target_periods)
        basis = "Same engine evaluated window"
    elif target_periods > 0 and total_periods > 0:
        eval_start = panel_start
        eval_end = panel_end
        warmup_periods = 0
        basis = f"Available panel is shorter than OOS length ({total_periods}/{target_periods})"
    else:
        eval_start = panel_start
        eval_end = panel_end
        warmup_periods = None
        basis = "Available Step 4 panel history"

    return {
        "panel_start": panel_start,
        "panel_end": panel_end,
        "panel_window": _format_window(panel_start, panel_end),
        "evaluation_start": eval_start,
        "evaluation_end": eval_end,
        "evaluation_window": _format_window(eval_start, eval_end),
        "target_periods": target_periods,
        "total_periods": total_periods,
        "warmup_periods": warmup_periods,
        "basis": basis,
    }


def _benchmark_context_rows(perf: dict, run_map: dict) -> tuple[pd.DataFrame, list[str], dict]:
    work, notes = _prepare_benchmark_panel()
    if work.empty:
        return pd.DataFrame(), notes, {}

    oos_returns = _extract_oos_returns(run_map)
    target_periods = int(len(oos_returns)) if oos_returns else 0
    window_meta = _resolve_engine_evaluation_window(work, target_periods)
    philosophy = str(st.session_state.get("investment_philosophy", "Balanced") or "Balanced")

    strategy_window = window_meta.get("evaluation_window", "Engine evaluated period")
    strategy_period_label = f"{strategy_window} ({target_periods} months)" if target_periods else strategy_window

    rows = [{
        "Reference": "Your strategy",
        "Type": "Engine portfolio",
        "Window": strategy_period_label,
        "CAGR": _format_pct_signed(_safe_float(perf.get("cagr", 0.0))),
        "Vol": _format_pct_signed(_safe_float(perf.get("annual_volatility", perf.get("volatility", 0.0)))),
        "MaxDD": f"-{100.0 * abs(_safe_float(perf.get('max_drawdown', 0.0))):.2f}%",
        "Sharpe": f"{_safe_float(perf.get('sharpe', 0.0)):.2f}",
        "Reading": _strategy_context_zone(perf, philosophy),
        "Comparability": "Primary engine result",
    }]

    eval_start = window_meta.get("evaluation_start")
    eval_end = window_meta.get("evaluation_end")

    for item in BENCHMARK_CONTEXT_ASSETS:
        ticker = str(item["ticker"]).upper()
        subset_full = work.loc[work["asset"] == ticker, ["date", "return"]].dropna().sort_values("date")
        if subset_full.empty:
            notes.append(f"{ticker} was not available in the current Step 4 asset panel, so it was skipped.")
            continue

        subset = subset_full.copy()
        comparable = "Same engine evaluated window"
        if target_periods > 0 and pd.notna(eval_start) and pd.notna(eval_end):
            subset = subset.loc[(subset["date"] >= eval_start) & (subset["date"] <= eval_end)].copy()
            if subset.empty and int(len(subset_full)) >= target_periods:
                subset = subset_full.tail(target_periods).copy()
                comparable = "Same trailing OOS length"
            elif int(len(subset)) < max(2, int(0.90 * target_periods)):
                comparable = f"Partial data in evaluated window ({len(subset)}/{target_periods} months)"
        elif target_periods > 0 and int(len(subset_full)) >= target_periods:
            subset = subset_full.tail(target_periods).copy()
            comparable = "Same trailing OOS length"
        else:
            comparable = "Available Step 4 history"

        metrics = _compute_return_metrics(subset["return"], periods_per_year=12)
        if not metrics:
            notes.append(f"{ticker} did not have enough clean returns to compute benchmark metrics for the evaluated window.")
            continue

        rows.append({
            "Reference": item["reference"],
            "Type": item["type"],
            "Window": _format_window(subset["date"].min(), subset["date"].max(), comparable),
            "CAGR": _format_pct_signed(metrics["cagr"]),
            "Vol": _format_pct_signed(metrics["annual_volatility"]),
            "MaxDD": f"{100.0 * metrics['max_drawdown']:.2f}%",
            "Sharpe": f"{metrics['sharpe']:.2f}",
            "Reading": item["reading"],
            "Comparability": comparable,
        })

    return pd.DataFrame(rows), notes, window_meta


def _parse_percent_label(value: Any) -> float | None:
    """Convert labels such as '9.73%' or '-19.43%' back to decimal values."""
    try:
        text = str(value or "").strip().replace("%", "").replace("+", "")
        if not text or text == "—":
            return None
        return float(text) / 100.0
    except Exception:
        return None


def _benchmark_row_for(bench_df: pd.DataFrame, needle: str) -> dict:
    if not isinstance(bench_df, pd.DataFrame) or bench_df.empty or "Reference" not in bench_df.columns:
        return {}
    try:
        rows = bench_df.loc[bench_df["Reference"].astype(str).str.contains(needle, case=False, regex=False)]
        if rows.empty:
            return {}
        return dict(rows.iloc[0])
    except Exception:
        return {}


def _render_compact_same_period_context(perf: dict, run_map: dict) -> None:
    """Show a small same-period benchmark context inside the result-reading expander.

    This block keeps the main interpretation compact by showing only the strategy,
    S&P 500, Nasdaq-100 and Gold over the same evaluated period.
    """
    bench_df, _, _ = _benchmark_context_rows(perf, run_map)
    if not isinstance(bench_df, pd.DataFrame) or bench_df.empty:
        return

    strategy = _benchmark_row_for(bench_df, "Your strategy")
    sp500 = _benchmark_row_for(bench_df, "SPY")
    nasdaq = _benchmark_row_for(bench_df, "QQQ")
    gold = _benchmark_row_for(bench_df, "GLD")
    if not strategy or not sp500 or not nasdaq or not gold:
        return

    strategy_cagr = _parse_percent_label(strategy.get("CAGR"))
    sp500_cagr = _parse_percent_label(sp500.get("CAGR"))
    nasdaq_cagr = _parse_percent_label(nasdaq.get("CAGR"))
    strategy_vol = _parse_percent_label(strategy.get("Vol"))
    nasdaq_vol = _parse_percent_label(nasdaq.get("Vol"))
    strategy_dd = _parse_percent_label(strategy.get("MaxDD"))
    nasdaq_dd = _parse_percent_label(nasdaq.get("MaxDD"))

    if (
        strategy_cagr is not None
        and sp500_cagr is not None
        and nasdaq_cagr is not None
        and strategy_vol is not None
        and nasdaq_vol is not None
        and strategy_dd is not None
        and nasdaq_dd is not None
        and strategy_cagr < sp500_cagr
        and strategy_cagr < nasdaq_cagr
        and strategy_vol < nasdaq_vol
        and abs(strategy_dd) < abs(nasdaq_dd)
    ):
        context_text = (
            "**Same-period context:** Over the same evaluated period, this strategy had lower CAGR "
            "than the S&P 500 and Nasdaq-100, but also lower volatility and a less severe max drawdown "
            "than Nasdaq-100. Gold is included as a defensive diversifier reference. This suggests a more "
            "risk-controlled profile rather than a pure growth benchmark profile."
        )
    else:
        context_text = (
            "**Same-period context:** The table below compares this strategy with the S&P 500, "
            "Nasdaq-100 and Gold over the same evaluated period. Use it as context for the "
            "risk/return profile, not as a replacement for the active strategy result."
        )

    st.markdown(context_text)

    rows = [
        {
            "Reference": "Your strategy",
            "CAGR": strategy.get("CAGR", "—"),
            "Vol": strategy.get("Vol", "—"),
            "MaxDD": strategy.get("MaxDD", "—"),
            "Sharpe": strategy.get("Sharpe", "—"),
        },
        {
            "Reference": "S&P 500",
            "CAGR": sp500.get("CAGR", "—"),
            "Vol": sp500.get("Vol", "—"),
            "MaxDD": sp500.get("MaxDD", "—"),
            "Sharpe": sp500.get("Sharpe", "—"),
        },
        {
            "Reference": "Nasdaq-100",
            "CAGR": nasdaq.get("CAGR", "—"),
            "Vol": nasdaq.get("Vol", "—"),
            "MaxDD": nasdaq.get("MaxDD", "—"),
            "Sharpe": nasdaq.get("Sharpe", "—"),
        },
        {
            "Reference": "Gold",
            "CAGR": gold.get("CAGR", "—"),
            "Vol": gold.get("Vol", "—"),
            "MaxDD": gold.get("MaxDD", "—"),
            "Sharpe": gold.get("Sharpe", "—"),
        },
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

def _profile_family_for_verdict(philosophy: Any) -> str:
    raw = str(philosophy or "Balanced").strip().lower()
    if "defensive" in raw or "conservative" in raw:
        return "defensive"
    if "growth" in raw:
        return "growth"
    return "balanced"


def _higher_is_better_verdict(value: float, thresholds: tuple[float, float, float, float]) -> str:
    very_good, good, normal, bad = thresholds
    if value >= very_good:
        return "Very good"
    if value >= good:
        return "Good"
    if value >= normal:
        return "Normal"
    if value >= bad:
        return "Bad"
    return "Very bad"


def _lower_is_better_verdict(value: float, thresholds: tuple[float, float, float, float]) -> str:
    very_good, good, normal, bad = thresholds
    if value <= very_good:
        return "Very good"
    if value <= good:
        return "Good"
    if value <= normal:
        return "Normal"
    if value <= bad:
        return "Bad"
    return "Very bad"


def _render_metric_verdicts_for_profile(perf: dict, philosophy: Any) -> None:
    """Read the four headline metrics in the context of the selected risk profile."""
    profile_label = str(philosophy or "Balanced").strip() or "Balanced"
    profile_family = _profile_family_for_verdict(profile_label)

    cagr = _safe_float(perf.get("cagr", 0.0), 0.0)
    vol = _safe_float(perf.get("annual_volatility", perf.get("volatility", 0.0)), 0.0)
    maxdd_abs = abs(_safe_float(perf.get("max_drawdown", 0.0), 0.0))
    sharpe = _safe_float(perf.get("sharpe", 0.0), 0.0)

    cagr_thresholds = {
        "defensive": (0.090, 0.060, 0.035, 0.010),
        "balanced": (0.120, 0.080, 0.050, 0.020),
        "growth": (0.150, 0.100, 0.060, 0.020),
    }[profile_family]
    vol_thresholds = {
        "defensive": (0.080, 0.110, 0.150, 0.200),
        "balanced": (0.100, 0.130, 0.170, 0.220),
        "growth": (0.140, 0.180, 0.220, 0.280),
    }[profile_family]
    maxdd_thresholds = {
        "defensive": (0.100, 0.150, 0.220, 0.320),
        "balanced": (0.120, 0.180, 0.250, 0.350),
        "growth": (0.180, 0.250, 0.350, 0.450),
    }[profile_family]

    cagr_verdict = _higher_is_better_verdict(cagr, cagr_thresholds)
    vol_verdict = _lower_is_better_verdict(vol, vol_thresholds)
    maxdd_verdict = _lower_is_better_verdict(maxdd_abs, maxdd_thresholds)
    sharpe_verdict = _higher_is_better_verdict(sharpe, (1.20, 0.75, 0.40, 0.10))

    st.markdown("**Metric verdict for this risk profile**")
    st.caption(
        f"Verdicts are read against the selected {profile_label} risk profile, "
        "not as universal investment ratings."
    )
    st.markdown(
        f"- **CAGR ({_pct(cagr)}) — {cagr_verdict}.** Annualised growth rate of the tested strategy over "
        f"the historical period. It is not a guaranteed future return. For a {profile_label} setup, this indicates "
        f"the strength of the long-run growth side of the trade-off.\n"
        f"- **Volatility ({_pct(vol)}) — {vol_verdict}.** How much the strategy return path fluctuated. "
        f"Higher volatility usually means a rougher ride. For a {profile_label} setup, lower bumpiness usually makes "
        f"the strategy easier to hold through time.\n"
        f"- **MaxDD (-{100.0 * maxdd_abs:.2f}%) — {maxdd_verdict}.** The largest peak-to-trough loss during "
        f"the tested period. For a {profile_label} setup, this is the main pain-test metric to watch closely.\n"
        f"- **Sharpe ({sharpe:.2f}) — {sharpe_verdict}.** Risk-adjusted return measure. Higher can be better, "
        f"but it depends on the tested period and assumptions. For a {profile_label} setup, this shows whether "
        f"the return compensated the investor for the bumpiness."
    )


def _plain_english_result_reading(perf: dict, philosophy: Any) -> str:
    """Return a dynamic plain-English interpretation of the run result."""
    profile_label = str(philosophy or "Balanced").strip() or "Balanced"
    profile_family = _profile_family_for_verdict(profile_label)
    cagr = _safe_float(perf.get("cagr", 0.0), 0.0)
    vol = _safe_float(perf.get("annual_volatility", perf.get("volatility", 0.0)), 0.0)
    maxdd_abs = abs(_safe_float(perf.get("max_drawdown", 0.0), 0.0))
    sharpe = _safe_float(perf.get("sharpe", 0.0), 0.0)

    if profile_family == "growth":
        if cagr >= 0.08 and sharpe >= 0.50:
            reading = (
                "This run is doing the main job of a Growth setup: it captures meaningful upside while keeping "
                "risk-adjusted return in a usable range."
            )
        elif cagr < 0.05:
            reading = (
                "This run may be too muted for a Growth setup: the path may be controlled, but the growth case "
                "is not especially strong."
            )
        else:
            reading = (
                "This run is usable for a Growth setup, but it needs comparison against the suggestions to check "
                "whether the upside is worth the risk taken."
            )
        priority = "The key question is whether Sharpe and drawdown are acceptable without removing too much upside."
    elif profile_family == "defensive":
        if maxdd_abs <= 0.15 and vol <= 0.12:
            reading = (
                "This run broadly fits a Defensive/Conservative setup: the historical path looks more controlled "
                "than a pure growth benchmark."
            )
        else:
            reading = (
                "This run may feel uncomfortable for a Defensive/Conservative setup because the historical risk "
                "side is still visible."
            )
        priority = "The key question is whether drawdown and volatility are low enough for the user to stay invested."
    else:
        if sharpe >= 0.60 and cagr > 0.0:
            reading = (
                "This run is broadly balanced: it has positive long-run growth and a usable risk-adjusted profile, "
                "but the drawdown still matters."
            )
        elif maxdd_abs >= 0.25:
            reading = (
                "This run has a return case, but the historical drawdown may be too heavy for a Balanced setup."
            )
        else:
            reading = (
                "This run is mixed rather than clearly bad or clearly excellent; the decision depends on the "
                "CAGR, Sharpe and drawdown trade-off."
            )
        priority = "The key question is whether the drawdown/Sharpe/CAGR balance feels worth accepting."

    return (
        f"**Plain-English reading:** {reading} "
        f"For this {profile_label} profile, CAGR is {_pct(cagr)}, volatility is {_pct(vol)}, "
        f"MaxDD is -{100.0 * maxdd_abs:.2f}%, and Sharpe is {sharpe:.2f}. {priority}"
    )


def _benchmark_full_history_rows() -> tuple[pd.DataFrame, list[str], dict]:
    work, notes = _prepare_benchmark_panel()
    if work.empty:
        return pd.DataFrame(), notes, {}

    window_meta = _resolve_engine_evaluation_window(work, 0)
    rows: list[dict] = []
    for item in BENCHMARK_CONTEXT_ASSETS:
        ticker = str(item["ticker"]).upper()
        subset = work.loc[work["asset"] == ticker, ["date", "return"]].dropna().sort_values("date")
        if subset.empty:
            notes.append(f"{ticker} was not available in the current Step 4 asset panel, so it was skipped from full-history context.")
            continue

        metrics = _compute_return_metrics(subset["return"], periods_per_year=12)
        if not metrics:
            notes.append(f"{ticker} did not have enough clean returns to compute full-history metrics.")
            continue

        rows.append({
            "Reference": item["reference"],
            "Type": item["type"],
            "Window": _format_window(subset["date"].min(), subset["date"].max()),
            "Months": int(metrics.get("periods", len(subset))),
            "CAGR": _format_pct_signed(metrics["cagr"]),
            "Vol": _format_pct_signed(metrics["annual_volatility"]),
            "MaxDD": f"{100.0 * metrics['max_drawdown']:.2f}%",
            "Sharpe": f"{metrics['sharpe']:.2f}",
            "Reading": item["reading"],
        })

    return pd.DataFrame(rows), notes, window_meta



def _render_benchmark_context(perf: dict, run_map: dict, *, inline_details: bool = False) -> dict:
    """Render optional benchmark context without duplicating the main result interpretation."""
    bench_df, notes, window_meta = _benchmark_context_rows(perf, run_map)
    panel_window = str(window_meta.get("panel_window", "—") or "—")
    eval_window = str(window_meta.get("evaluation_window", "—") or "—")
    target_periods = _safe_int(window_meta.get("target_periods", 0), 0)
    warmup_periods = window_meta.get("warmup_periods", None)

    if target_periods > 0:
        warmup_text = (
            f" · warm-up before evaluation: {int(warmup_periods)} monthly observations"
            if isinstance(warmup_periods, int) and warmup_periods > 0
            else ""
        )
        st.caption(
            f"Evaluated period: {eval_window} · {target_periods} monthly OOS returns{warmup_text}. "
            f"Full Step 4 panel: {panel_window}."
        )
    elif panel_window != "—":
        st.caption(f"Full Step 4 panel: {panel_window}. Exact OOS return length was not found in the run payload.")

    def _render_benchmark_tables_and_methodology() -> None:
        if isinstance(bench_df, pd.DataFrame) and not bench_df.empty:
            compact_cols = [
                col
                for col in ["Reference", "Type", "CAGR", "Vol", "MaxDD", "Sharpe", "Reading"]
                if col in bench_df.columns
            ]
            st.markdown("**Same evaluated period**")
            st.dataframe(bench_df[compact_cols], use_container_width=True, hide_index=True)
            st.caption(
                "Note: TLT tracks 20+ year US Treasury bonds. Long-duration bonds can show weak or negative returns "
                "when interest rates rise, because fixed-rate bond prices generally move inversely to rates."
            )
        else:
            st.info("Same-window benchmark details are unavailable for this run.")

        st.divider()
        st.markdown("**Methodology and exclusions**")
        st.caption(
            "Compact method: each benchmark is recomputed from the current Step 4 monthly-return panel and aligned "
            "to the same walk-forward evaluated period where data is available. This is a sanity check, not a "
            "forecast, recommendation, or replacement for the active Strategy Engine result."
        )

        if notes:
            st.caption("Panel notes / exclusions")
            for note in notes:
                st.write(f"- {note}")

        if SHORTER_HISTORY_CONTEXT:
            st.caption("Excluded from this same-period table because their histories are shorter or less comparable:")
            for item in SHORTER_HISTORY_CONTEXT:
                st.write(f"- {item}")

    if inline_details:
        _render_benchmark_tables_and_methodology()
    else:
        with st.expander("Benchmark comparison", expanded=False):
            _render_benchmark_tables_and_methodology()

    return {"bench_df": bench_df, "window_meta": window_meta}


def _benchmark_asset_count(benchmark_payload: dict) -> int:
    bench_df = benchmark_payload.get("bench_df") if isinstance(benchmark_payload, dict) else pd.DataFrame()
    if not isinstance(bench_df, pd.DataFrame) or bench_df.empty or "Reference" not in bench_df.columns:
        return 0
    try:
        refs = bench_df["Reference"].astype(str)
        return int((~refs.str.contains("Your strategy", case=False, regex=False)).sum())
    except Exception:
        return max(0, int(len(bench_df)) - 1)


def _benchmark_status_label(benchmark_payload: dict) -> str:
    assets_checked = _benchmark_asset_count(benchmark_payload)
    if assets_checked >= 3:
        return "Passed"
    if assets_checked > 0:
        return "Limited"
    return "Unavailable"


def _reliability_confidence_label(run_map: dict, benchmark_payload: dict) -> str:
    window_meta = _coerce_mapping(benchmark_payload.get("window_meta", {}) if isinstance(benchmark_payload, dict) else {})
    target_periods = _safe_int(window_meta.get("target_periods", 0), 0)
    if target_periods <= 0:
        target_periods = len(_extract_oos_returns(run_map))
    assets_checked = _benchmark_asset_count(benchmark_payload)

    if target_periods >= 120 and assets_checked >= 3:
        return "Moderate-to-Strong"
    if target_periods >= 60 and assets_checked >= 2:
        return "Moderate"
    if target_periods > 0:
        return "Limited"
    return "Unavailable"


def _render_sidebar_diagnostics(run_map: dict, benchmark_payload: dict, *, improvement_flow_completed: bool = True) -> None:
    """Deprecated no-op.

    The Step 5 sidebar Diagnostics expander is owned by
    ``ui.common.global_sidebar``. Keeping this function as a no-op preserves
    compatibility with older local imports while preventing duplicate sidebar
    expanders after a portfolio run.
    """
    return None

def _maybe_scroll_to_reliability_panel() -> None:
    if not st.session_state.pop(STEP5_SCROLL_TO_RELIABILITY_KEY, False):
        return

    components.html(
        f"""
        <script>
            const target = window.parent.document.getElementById("{STEP5_RELIABILITY_ANCHOR_ID}");
            if (target) {{
                setTimeout(() => target.scrollIntoView({{behavior: "smooth", block: "start"}}), 120);
            }}
        </script>
        """,
        height=0,
    )


def _maybe_scroll_to_run_diagnostics_panel() -> None:
    if not st.session_state.pop(STEP5_SCROLL_TO_RUN_DIAGNOSTICS_KEY, False):
        return

    components.html(
        f"""
        <script>
            const target = window.parent.document.getElementById("{STEP5_RUN_DIAGNOSTICS_ANCHOR_ID}");
            if (target) {{
                setTimeout(() => target.scrollIntoView({{behavior: "smooth", block: "start"}}), 120);
            }}
        </script>
        """,
        height=0,
    )


def render_post_run(run_result: dict) -> None:
    """Render the Gold Stable post-run surface.

    Active strategy engine flow:
    - real engine metrics
    - compact validation context
    - optional improvement checks
    - OOS-return bridge into Long-Term Scenario
    """
    run_map = _coerce_mapping(run_result)
    if not run_map:
        st.warning("Run result is missing or empty.")
        return

    perf = _coerce_mapping(run_map.get("performance_summary", {}))
    if not perf:
        st.warning("Run result payload is missing a performance summary.")
        return

    st.markdown(f'<div id="{STEP5_REAL_RUN_RESULT_ANCHOR_ID}"></div>', unsafe_allow_html=True)
    philosophy = str(st.session_state.get("investment_philosophy", "Balanced") or "Balanced")
    try:
        universe_size = int(st.session_state.get("universe_size", 25) or 25)
    except Exception:
        universe_size = 25
    template = str(st.session_state.get("step5_template", "") or "—")
    style = str(st.session_state.get("step5_style", "") or "—")

    run_signature = str(run_map.get("run_signature", "") or "")
    config_fingerprint = str(run_map.get("config_fingerprint", "") or "")
    run_timestamp = str(run_map.get("run_timestamp", "") or "")
    source = str(run_map.get("source", "micro_pipeline_real") or "micro_pipeline_real")
    panel_label = str(run_map.get("asset_panel_source_label", "Step 4 asset panel") or "Step 4 asset panel")
    panel_rows = int(_safe_float(run_map.get("asset_panel_n_rows", 0), 0))
    panel_assets = int(_safe_float(run_map.get("asset_panel_n_assets", 0), 0))
    panel_shape = run_map.get("panel_shape", None)
    oos_months = len(_extract_oos_returns(run_map))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("CAGR", _pct(_safe_float(perf.get("cagr", 0.0))))
    with c2:
        st.metric("Vol", _pct(_safe_float(perf.get("annual_volatility", perf.get("volatility", 0.0)))))
    with c3:
        st.metric("MaxDD", f"-{100.0 * abs(_safe_float(perf.get('max_drawdown', 0.0))):.2f}%")
    with c4:
        st.metric("Sharpe", f"{_safe_float(perf.get('sharpe', 0.0)):.2f}")

    st.caption(
        "Historical backtest result for comparison only. Not a forecast, guarantee, or live fund track record."
    )

    with st.expander("How to read this result", expanded=False):
        _render_metric_verdicts_for_profile(perf, philosophy)
        st.info(_plain_english_result_reading(perf, philosophy))
        st.divider()
        _render_compact_same_period_context(perf, run_map)

    bench_df, _, window_meta = _benchmark_context_rows(perf, run_map)
    benchmark_payload = {"bench_df": bench_df, "window_meta": window_meta}

    size_flow_completed = _size_decision_completed_for_current_run(run_map)
    universe_flow_completed = _universe_decision_completed_for_current_run(run_map)
    auto_opt_already_applied = _auto_opt_applied_for_current_run(run_map)
    auto_opt_already_skipped = _auto_opt_skipped_for_current_run(run_map)
    preset_flow_completed = _preset_decision_completed_for_current_run(run_map)
    improvement_flow_completed = bool(size_flow_completed)

    if not improvement_flow_completed and bool(st.session_state.get(STEP5_RELIABILITY_EXPANDED_KEY, False)):
        st.session_state[STEP5_RELIABILITY_EXPANDED_KEY] = False
        st.session_state[STEP5_SCROLL_TO_RELIABILITY_KEY] = False

    # Sidebar diagnostics are rendered once by ui.common.global_sidebar.
    # Do not render another st.sidebar Diagnostics expander here, otherwise
    # Step 5 shows duplicate diagnostics and the post-run copy can reopen by
    # default after a portfolio run.

    st.markdown(f'<div id="{STEP5_IMPROVEMENT_CHECKS_ANCHOR_ID}"></div>', unsafe_allow_html=True)
    _maybe_scroll_to_improvement_checks()
    st.markdown("## Optional improvement checks")
    _render_improvement_phase_row(run_map)

    if size_flow_completed:
        _render_resolved_phase_history(run_map)
        _render_completed_improvement_flow(run_map)
    elif universe_flow_completed:
        _render_resolved_phase_history(run_map)
        render_size_improvement(run_map)
    elif auto_opt_already_applied or auto_opt_already_skipped:
        _render_resolved_phase_history(run_map)
        render_universe_improvement(run_map)
    elif preset_flow_completed:
        _render_resolved_phase_history(run_map)
        render_auto_opt_improvement(run_map)
    else:
        preset_flow_state = render_preset_improvement(run_map)
        if _preset_blocks_engine_tuning(preset_flow_state):
            _render_engine_tuning_waiting_for_preset()
        else:
            render_auto_opt_improvement(run_map)


    reliability_visible = bool(st.session_state.get(STEP5_RELIABILITY_EXPANDED_KEY, False))
    if reliability_visible and improvement_flow_completed:
        st.markdown(f'<div id="{STEP5_RELIABILITY_ANCHOR_ID}"></div>', unsafe_allow_html=True)
        _maybe_scroll_to_reliability_panel()

        with st.expander("Reliability and robustness", expanded=True):
            if st.button(
                "Hide reliability & robustness",
                key="step5_main_hide_reliability_and_robustness",
                use_container_width=True,
            ):
                st.session_state[STEP5_RELIABILITY_EXPANDED_KEY] = False
                st.session_state[STEP5_SCROLL_TO_RELIABILITY_KEY] = False
                st.rerun()

            render_result_reliability_assessment(run_map, benchmark_payload=benchmark_payload, inline_details=True)
            render_start_date_robustness_timing_block(run_map)
            st.info(
                "These figures come from a historical walk-forward backtest using the selected asset panel. "
                "They are useful for comparing configurations inside the app, but they are not forecasts or guarantees. "
                "Results depend on the date range, asset universe, data quality, and engine assumptions."
            )
    else:
        st.session_state.pop(STEP5_SCROLL_TO_RELIABILITY_KEY, None)

    # Technical improvement guidance now lives next to the technical engine controls
    # inside Change or rerun setup. Keeping it there avoids a second post-run
    # diagnostics expander after the recommendation flow.

    n_oos = _store_projection_bridge_context(run_map)

    run_diagnostics_visible = bool(st.session_state.get(STEP5_RUN_DIAGNOSTICS_EXPANDED_KEY, False))
    if run_diagnostics_visible:
        st.markdown(f'<div id="{STEP5_RUN_DIAGNOSTICS_ANCHOR_ID}"></div>', unsafe_allow_html=True)
        _maybe_scroll_to_run_diagnostics_panel()

        with st.expander("Run timings and diagnostics", expanded=True):
            if st.button(
                "Hide run timings",
                key="step5_main_hide_run_timings",
                use_container_width=True,
            ):
                st.session_state[STEP5_RUN_DIAGNOSTICS_EXPANDED_KEY] = False
                st.session_state[STEP5_SCROLL_TO_RUN_DIAGNOSTICS_KEY] = False
                st.rerun()

            if not improvement_flow_completed:
                st.info(
                    "Timing diagnostics are available now, but suggestion timings are partial until all optional improvement checks are completed."
                )

            meta_parts = [
                f"source={source}",
                f"asset_panel={panel_label}",
                f"rows={panel_rows}",
                f"assets={panel_assets}",
            ]
            if run_signature:
                meta_parts.append(f"run_signature={run_signature}")
            if config_fingerprint:
                meta_parts.append(f"config_fp={config_fingerprint}")
            if panel_shape:
                meta_parts.append(f"panel_shape={panel_shape}")
            st.caption(" · ".join(meta_parts))
            if run_timestamp:
                st.caption(f"run_timestamp={run_timestamp}")
            _render_engine_timing_block(run_map)
            if size_flow_completed:
                st.caption(
                    "Suggestion timing below is historical for the completed improvement flow. "
                    "It is not re-run after Apply; the current result was promoted from the previously tested candidate."
                )
            _render_preset_suggestion_timing_block()
            _render_auto_opt_suggestion_timing_block()
            _render_universe_suggestion_timing_block()
            _render_size_suggestion_timing_block()
            _render_feature_mu_block(run_map)
    else:
        st.session_state.pop(STEP5_SCROLL_TO_RUN_DIAGNOSTICS_KEY, None)

    st.markdown("---")
    nav_left, nav_right = st.columns(2)
    with nav_left:
        if st.button("← Back to Risk Profile & Asset Universe", key="step5_bridge_back_to_investment_setup", use_container_width=True):
            st.session_state["current_step"] = 4
            st.rerun()
    with nav_right:
        if st.button("Continue to Long-Term Scenario →", key="step5_bridge_continue_to_long_term", use_container_width=True):
            st.session_state["current_step"] = 6
            st.rerun()
