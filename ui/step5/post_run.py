from __future__ import annotations

import json
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from ui.step5.run_panel import clear_retired_step5_state
from ui.step5.preset_recommendations import render_preset_improvement
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
from ui.step5.reliability_assessment import render_result_reliability_assessment, render_start_date_robustness_timing_block



STEP5_SCROLL_TO_RESULT_AFTER_APPLY_KEY = "step5_scroll_to_real_run_result_after_apply_v1"
PRESET_SUGGESTION_TIMING_KEY = "step5_preset_suggestion_timing_v1"
STEP5_REAL_RUN_RESULT_ANCHOR_ID = "step5-real-run-result-anchor"


def _maybe_scroll_to_real_run_result() -> None:
    """Scroll back to the real result after applying a rerun-tested preset.

    Streamlit has no native scroll-to-anchor API, so this tiny hidden component is
    intentionally limited to one job: move the viewport back to section 3 after
    an Apply action promotes a candidate result.
    """
    if not bool(st.session_state.pop(STEP5_SCROLL_TO_RESULT_AFTER_APPLY_KEY, False)):
        return

    components.html(
        f"""
        <script>
        const anchorId = {STEP5_REAL_RUN_RESULT_ANCHOR_ID!r};
        function scrollToRealRunResult() {{
            const doc = window.parent.document;
            const el = doc.getElementById(anchorId);
            if (el) {{
                el.scrollIntoView({{ behavior: "smooth", block: "start" }});
            }}
        }}
        setTimeout(scrollToRealRunResult, 250);
        setTimeout(scrollToRealRunResult, 800);
        </script>
        """,
        height=0,
    )


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
        with st.expander("Selected feature_mu columns", expanded=False):
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

    st.markdown("### Engine timing diagnostics")

    total_engine = _safe_float(engine_timing.get("total_engine", 0.0), 0.0)
    walk_forward = _safe_float(engine_timing.get("walk_forward_loop", 0.0), 0.0)
    cov_sigma = _safe_float(engine_timing.get("covariance_sigma_total", 0.0), 0.0)
    mu_sigma = _safe_float(engine_timing.get("mu_sigma_total", 0.0), 0.0)
    weight_build = _safe_float(engine_timing.get("weight_build_total", 0.0), 0.0)
    signal_model = _safe_float(engine_timing.get("signal_model_total", 0.0), 0.0)
    feature_mu = _safe_float(engine_timing.get("feature_mu_total", 0.0), 0.0)
    probabilistic = _safe_float(engine_timing.get("probabilistic_total", 0.0), 0.0)
    n_oos_dates = _safe_int(engine_timing.get("n_oos_dates", 0), 0)
    n_loop_iterations = _safe_int(engine_timing.get("n_loop_iterations", 0), 0)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total engine", f"{total_engine:.2f}s")
    with c2:
        st.metric("Walk-forward", f"{walk_forward:.2f}s")
    with c3:
        st.metric("Cov/Sigma", f"{cov_sigma:.2f}s")
    with c4:
        st.metric("Mu/Sigma", f"{mu_sigma:.2f}s")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric("Weight build", f"{weight_build:.2f}s")
    with c6:
        st.metric("Signal model", f"{signal_model:.2f}s")
    with c7:
        st.metric("Feature_mu", f"{feature_mu:.2f}s")
    with c8:
        st.metric("Probabilistic", f"{probabilistic:.2f}s")

    st.caption(
        f"n_oos_dates={n_oos_dates} · "
        f"n_loop_iterations={n_loop_iterations}"
    )

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
        with st.expander("Engine timing breakdown", expanded=False):
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
        "This is the extra time used by the automatic Step 5 preset suggestion test. "
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
            with st.expander("Preset suggestion timing breakdown", expanded=False):
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
        "This is the extra time used by the automatic Step 5 engine tuning suggestion test. "
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
            with st.expander("Engine tuning timing breakdown", expanded=False):
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
    current_run_signature = str(_coerce_mapping(run_map).get("run_signature", "") or "")
    applied_signature = str(st.session_state.get(AUTO_OPT_APPLIED_SIGNATURE_KEY, "") or "")
    return bool(current_run_signature and applied_signature and current_run_signature == applied_signature)


def _universe_applied_for_current_run(run_map: dict) -> bool:
    current_run_signature = str(_coerce_mapping(run_map).get("run_signature", "") or "")
    applied_signature = str(st.session_state.get(UNIVERSE_APPLIED_SIGNATURE_KEY, "") or "")
    return bool(current_run_signature and applied_signature and current_run_signature == applied_signature)


def _auto_opt_skipped_for_current_run(run_map: dict) -> bool:
    run_map = _coerce_mapping(run_map)
    current_run_signature = str(run_map.get("run_signature", "") or "")
    skipped_run_signature = str(st.session_state.get(AUTO_OPT_SKIPPED_RUN_SIGNATURE_KEY, "") or "")
    skipped_scope = str(st.session_state.get(AUTO_OPT_SKIPPED_SCOPE_KEY, "") or "")
    return bool(
        skipped_scope
        and current_run_signature
        and (not skipped_run_signature or skipped_run_signature == current_run_signature)
    )


def _universe_skipped_for_current_run(run_map: dict) -> bool:
    run_map = _coerce_mapping(run_map)
    current_run_signature = str(run_map.get("run_signature", "") or "")
    skipped_run_signature = str(st.session_state.get(UNIVERSE_SKIPPED_RUN_SIGNATURE_KEY, "") or "")
    skipped_scope = str(st.session_state.get(UNIVERSE_SKIPPED_SCOPE_KEY, "") or "")
    return bool(
        skipped_scope
        and current_run_signature
        and (not skipped_run_signature or skipped_run_signature == current_run_signature)
    )


def _universe_decision_completed_for_current_run(run_map: dict) -> bool:
    return bool(_universe_applied_for_current_run(run_map) or _universe_skipped_for_current_run(run_map))


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
    _clear_universe_suggestion_state()
    st.markdown("### Universe composition suggestion")
    st.info(
        "Universe composition becomes available after you apply the engine tuning suggestion, or after engine tuning converges with no accepted candidate."
    )
    st.caption(
        "This avoids testing asset-composition changes on top of technical settings that may still be replaced in the previous phase."
    )


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
        "This is the extra time used by the automatic Step 5 universe-composition suggestion. "
        "It tests only a small number of same-size asset compositions from the existing Step 4 data panel."
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
            with st.expander("Universe suggestion timing breakdown", expanded=False):
                st.dataframe(pd.DataFrame(clean_rows), use_container_width=True, hide_index=True)


def _render_completed_improvement_timing_summary() -> None:
    """Keep historical suggestion timing visible after Apply without re-running tests."""
    preset_timing = _coerce_mapping(st.session_state.get(PRESET_SUGGESTION_TIMING_KEY, {}))
    tuning_timing = _coerce_mapping(st.session_state.get(AUTO_OPT_SUGGESTION_TIMING_KEY, {}))
    universe_timing = _coerce_mapping(st.session_state.get(UNIVERSE_SUGGESTION_TIMING_KEY, {}))

    preset_total = _safe_float(preset_timing.get("total_seconds", 0.0), 0.0)
    tuning_total = _safe_float(tuning_timing.get("total_seconds", 0.0), 0.0)
    universe_total = _safe_float(universe_timing.get("total_seconds", 0.0), 0.0)
    total = float(max(0.0, preset_total) + max(0.0, tuning_total) + max(0.0, universe_total))
    if total <= 0.0:
        return

    st.markdown("### Completed improvement flow timing")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Preset test", f"{preset_total:.2f}s" if preset_total > 0 else "—")
    with c2:
        st.metric("Engine tuning test", f"{tuning_total:.2f}s" if tuning_total > 0 else "—")
    with c3:
        st.metric("Universe test", f"{universe_total:.2f}s" if universe_total > 0 else "—")
    with c4:
        st.metric("Total overhead", f"{total:.2f}s")

    preset_candidates = _safe_int(preset_timing.get("candidate_count", 0), 0)
    preset_accepted = _safe_int(preset_timing.get("accepted_count", 0), 0)
    tuning_candidates = _safe_int(tuning_timing.get("candidate_count", 0), 0)
    tuning_accepted = _safe_int(tuning_timing.get("accepted_count", 0), 0)
    universe_candidates = _safe_int(universe_timing.get("candidate_count", 0), 0)
    universe_accepted = _safe_int(universe_timing.get("accepted_count", 0), 0)
    st.caption(
        "Historical timing only: these suggestion tests were not re-run after Apply; "
        "the current result was promoted from the previously tested candidate. "
        f"Preset candidates={preset_candidates}, passed_gate={preset_accepted} · "
        f"Engine tuning candidates={tuning_candidates}, passed_gate={tuning_accepted} · "
        f"Universe candidates={universe_candidates}, passed_gate={universe_accepted}. "
        "Only the highest-scoring passed-gate candidate was promoted."
    )

    rows = []
    for phase_name, timing in [
        ("Preset suggestion", preset_timing),
        ("Engine tuning", tuning_timing),
        ("Universe composition", universe_timing),
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
        with st.expander("Completed suggestion timing breakdown", expanded=False):
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_completed_improvement_flow(run_map: dict) -> None:
    universe_applied = _universe_applied_for_current_run(run_map)
    universe_skipped = _universe_skipped_for_current_run(run_map)
    if universe_applied:
        label = str(st.session_state.get(UNIVERSE_APPLIED_LABEL_KEY, "") or "the accepted universe composition suggestion")
        message = f"Improvement flow completed: {label}. The rerun-tested candidate is now the current Step 5 result."
    elif universe_skipped:
        label = str(st.session_state.get(UNIVERSE_SKIPPED_LABEL_KEY, "") or "the recommended universe composition")
        message = f"Improvement flow completed: current universe kept. Skipped recommendation: {label}."
    else:
        tuning_label = str(st.session_state.get(AUTO_OPT_APPLIED_LABEL_KEY, "") or "the accepted improvement suggestion")
        message = f"Improvement flow completed: {tuning_label}. The rerun-tested candidate is now the current Step 5 result."

    st.markdown("## 4. Improve this setup (optional)")
    st.success(message)
    st.caption(
        "Preset, engine-tuning, and universe-composition tests are hidden for this result to avoid re-testing the same loop. "
        "Change the strategy setup, technical controls, Step 4 universe, or run a new baseline if you want to start a fresh improvement cycle."
    )
    _render_completed_improvement_timing_summary()



def _render_engine_tuning_waiting_for_preset() -> None:
    _clear_auto_opt_suggestion_state()
    st.markdown("### Engine tuning suggestion")
    st.info(
        "Engine tuning becomes available after you apply the preset suggestion or choose to keep the current preset."
    )
    st.caption(
        "This avoids testing technical knobs on a strategy preset that may be replaced in the previous phase. "
        "Once the preset decision is resolved, this phase will test small variations around the active engine configuration."
    )


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


def _render_what_to_watch(perf: dict, philosophy: Any) -> None:
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
        watch_items.append("Sharpe is usable; the key question is whether the improvement over the baseline justifies the extra engine complexity.")
    if maxdd >= 0.18:
        watch_items.append("MaxDD is the key risk flag: a drawdown near 20% can be psychologically hard even if the long-run return is positive.")
    if vol >= 0.15:
        watch_items.append("Volatility is elevated; reduce risk appetite or increase drawdown protection if the strategy should feel smoother.")
    if profile.lower() == "defensive" and (maxdd >= 0.15 or vol >= 0.12):
        watch_items.append("Because the selected philosophy is Defensive, prioritise drawdown and volatility over chasing higher CAGR.")
    elif profile.lower() == "growth":
        watch_items.append("Because the selected philosophy is Growth, some volatility is acceptable, but Sharpe should still justify the risk.")

    with st.expander("What to watch before changing the strategy", expanded=False):
        st.markdown("\n".join(f"- {item}" for item in watch_items))
        st.caption(
            "Use the benchmark context above as external reference only; the most direct comparison is still between the tested internal configurations."
        )



def _render_reliability_note() -> None:
    st.info(
        "**Reliability note:** these figures come from a historical walk-forward backtest using the selected Step 4 asset panel. "
        "The panel start date provides historical input for the engine; the displayed performance is based on the evaluated OOS returns produced after the engine has enough prior history. "
        "They are useful for comparing configurations inside the app, but they are not forecasts or guarantees. "
        "Results depend on the date range, asset universe, data quality, and engine assumptions. Step 6 should be used to explore future uncertainty rather than treating this run as a prediction."
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


def _render_benchmark_context(perf: dict, run_map: dict) -> None:
    st.markdown("### Benchmark context — same evaluated period")

    bench_df, notes, window_meta = _benchmark_context_rows(perf, run_map)
    panel_window = str(window_meta.get("panel_window", "—") or "—")
    eval_window = str(window_meta.get("evaluation_window", "—") or "—")
    target_periods = _safe_int(window_meta.get("target_periods", 0), 0)
    warmup_periods = window_meta.get("warmup_periods", None)

    st.caption(
        "The Step 4 start date defines the historical market-data panel used by the engine. "
        "The metrics above are the engine's walk-forward evaluated returns, so the main benchmark table below uses that same evaluated period."
    )

    if target_periods > 0:
        warmup_text = (
            f" · Approx. warm-up/training before evaluation: {int(warmup_periods)} monthly observations"
            if isinstance(warmup_periods, int) and warmup_periods > 0
            else ""
        )
        st.info(
            f"Step 4 panel: {panel_window} · Engine evaluated period: {eval_window} "
            f"({target_periods} monthly OOS returns){warmup_text}."
        )
    elif panel_window != "—":
        st.info(f"Step 4 panel: {panel_window}. Exact OOS return length was not found in the run payload.")

    if isinstance(bench_df, pd.DataFrame) and not bench_df.empty:
        st.dataframe(bench_df, use_container_width=True, hide_index=True)
        st.caption(
            "Note: TLT tracks 20+ year US Treasury bonds. Long-duration bonds can show weak or negative returns "
            "when interest rates rise, because fixed-rate bond prices generally move inversely to rates."
        )
    else:
        st.info("Benchmark context is unavailable for this run because the Step 4 panel could not be read.")

    with st.expander("Benchmark details and methodology", expanded=False):
        st.markdown("**Why can long-term Treasury bonds show negative CAGR?**")
        st.write(
            "TLT tracks long-duration US Treasury bonds, specifically bonds with more than 20 years "
            "remaining to maturity. Long-duration fixed-rate bonds are sensitive to interest-rate changes: "
            "when market interest rates rise, existing bond prices generally fall."
        )
        st.caption("Sources: iShares/BlackRock TLT fund description; SEC Investor Bulletin on Interest Rate Risk.")

        st.divider()
        st.markdown("**Full-history benchmark context from Step 4 panel**")
        st.caption(
            "This table uses the full available Step 4 history for each reference asset, usually starting around 2005 for the selected benchmark set. "
            "It is long-run context only, not a direct comparison with the engine result unless the strategy is also evaluated over the same full window."
        )
        full_df, full_notes, _ = _benchmark_full_history_rows()
        if isinstance(full_df, pd.DataFrame) and not full_df.empty:
            st.dataframe(full_df, use_container_width=True, hide_index=True)
        else:
            st.info("Full-history benchmark context is unavailable for this run.")
        if full_notes:
            st.caption("Full-history panel notes")
            for note in full_notes:
                st.write(f"- {note}")

        st.divider()
        st.markdown("**Benchmark methodology and exclusions**")
        st.markdown(
            "- The **main benchmark table** compares like with like: your engine result and reference assets over the same walk-forward evaluated period.\n"
            "- The **full-history table** is contextual only. It shows what well-known assets did across the full Step 4 panel, but it is not the direct scorecard for your strategy.\n"
            "- Benchmarks are **not hardcoded historical ranges**; they are recomputed from the current Step 4 panel using the same monthly-return convention.\n"
            "- This is educational context, not an investment recommendation and not a forecast."
        )
        if notes:
            st.caption("Main benchmark panel notes")
            for note in notes:
                st.write(f"- {note}")

        if SHORTER_HISTORY_CONTEXT:
            st.caption("Shorter-history references deliberately left out of the main same-window table:")
            for item in SHORTER_HISTORY_CONTEXT:
                st.write(f"- {item}")

    return {"bench_df": bench_df, "window_meta": window_meta}


def render_post_run(run_result: dict) -> None:
    """Render the Gold Stable post-run surface.

    Active Step 5 flow:
    - real engine metrics
    - optional engine timing diagnostics
    - feature_mu metadata
    - OOS-return bridge into Step 6
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
    _maybe_scroll_to_real_run_result()
    st.markdown("## 3. Real run result")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("CAGR", _pct(_safe_float(perf.get("cagr", 0.0))))
    with c2:
        st.metric("Vol", _pct(_safe_float(perf.get("annual_volatility", perf.get("volatility", 0.0)))))
    with c3:
        st.metric("MaxDD", f"-{100.0 * abs(_safe_float(perf.get('max_drawdown', 0.0))):.2f}%")
    with c4:
        st.metric("Sharpe", f"{_safe_float(perf.get('sharpe', 0.0)):.2f}")

    headline, body = _result_interpretation(perf)
    st.info(f"**{headline}:** {body}")

    _render_metric_explainer(perf)

    philosophy = str(st.session_state.get("investment_philosophy", "Balanced") or "Balanced")
    fit_level, fit_message = _philosophy_fit_message(perf, philosophy)
    if fit_level == "success":
        st.success(f"**Philosophy fit:** {fit_message}")
    elif fit_level == "warning":
        st.warning(f"**Philosophy fit:** {fit_message}")
    else:
        st.info(f"**Philosophy fit:** {fit_message}")

    benchmark_payload = _render_benchmark_context(perf, run_map)
    render_result_reliability_assessment(run_map, benchmark_payload=benchmark_payload)
    _render_what_to_watch(perf, philosophy)
    _render_reliability_note()

    run_signature = str(run_map.get("run_signature", "") or "")
    config_fingerprint = str(run_map.get("config_fingerprint", "") or "")
    run_timestamp = str(run_map.get("run_timestamp", "") or "")
    source = str(run_map.get("source", "micro_pipeline_real") or "micro_pipeline_real")
    panel_label = str(run_map.get("asset_panel_source_label", "Step 4 asset panel") or "Step 4 asset panel")
    panel_rows = int(_safe_float(run_map.get("asset_panel_n_rows", 0), 0))
    panel_assets = int(_safe_float(run_map.get("asset_panel_n_assets", 0), 0))
    panel_shape = run_map.get("panel_shape", None)

    improvement_flow_completed = _universe_decision_completed_for_current_run(run_map)
    auto_opt_already_applied = _auto_opt_applied_for_current_run(run_map)
    auto_opt_already_skipped = _auto_opt_skipped_for_current_run(run_map)

    if improvement_flow_completed:
        _render_completed_improvement_flow(run_map)
    elif auto_opt_already_applied or auto_opt_already_skipped:
        # Phase 2 has already been resolved for this run. Do not send the
        # current signature back through Phase 1, otherwise the preset search
        # is tested again before Phase 3 becomes available.
        render_auto_opt_improvement(run_map)
        render_universe_improvement(run_map)
    else:
        preset_flow_state = render_preset_improvement(run_map)
        if _preset_blocks_engine_tuning(preset_flow_state):
            _render_engine_tuning_waiting_for_preset()
        else:
            render_auto_opt_improvement(run_map)
            if _auto_opt_blocks_universe(run_map):
                _render_universe_waiting_for_engine_tuning()
            else:
                render_universe_improvement(run_map)

    st.markdown("## 5. Ready for projection")
    n_oos = _store_projection_bridge_context(run_map)
    if n_oos > 0:
        st.success(f"Projection bridge ready: {n_oos} monthly OOS returns stored for Step 6.")
    else:
        st.info("Projection bridge note: no OOS return series was found in this run payload; Step 6 can still use fallback profile assumptions.")

    with st.expander("Diagnostics (advanced)", expanded=False):
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
        if improvement_flow_completed:
            st.caption(
                "Suggestion timing below is historical for the completed improvement flow. "
                "It is not re-run after Apply; the current result was promoted from the previously tested candidate."
            )
        _render_preset_suggestion_timing_block()
        _render_auto_opt_suggestion_timing_block()
        _render_universe_suggestion_timing_block()
        render_start_date_robustness_timing_block(run_map)
        _render_feature_mu_block(run_map)

    clear_retired_step5_state()

    st.markdown("---")
    left, right = st.columns(2)
    with left:
        if st.button("Back to Step 4", key="step5_back_to_step4"):
            st.session_state["current_step"] = 4
            st.rerun()
    with right:
        if st.button("Continue to Projection", key="step5_continue_to_step6", use_container_width=True):
            st.session_state["current_step"] = 6
            st.rerun()
