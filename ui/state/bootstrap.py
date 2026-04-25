"""Session state bootstrap for LifeBudget Micro."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import pandas as pd
import streamlit as st

from .keys import *


def _default_fixed_breakdown_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Name": "Rent / housing", "Amount (£)": 0.0, "Period": "Monthly"},
            {"Name": "Utilities (gas/electric/water)", "Amount (£)": 0.0, "Period": "Monthly"},
            {"Name": "Council tax", "Amount (£)": 0.0, "Period": "Monthly"},
            {"Name": "Internet / phone", "Amount (£)": 0.0, "Period": "Monthly"},
            {"Name": "Transport pass", "Amount (£)": 0.0, "Period": "Weekly"},
            {"Name": "Insurance", "Amount (£)": 0.0, "Period": "Monthly"},
            {"Name": "Subscriptions (fixed)", "Amount (£)": 0.0, "Period": "Monthly"},
        ]
    )


def _default_one_off_events_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Event (optional)": "", "Amount (£)": 0.0, "Week": 1},
            {"Event (optional)": "", "Amount (£)": 0.0, "Week": 1},
            {"Event (optional)": "", "Amount (£)": 0.0, "Week": 1},
        ]
    )


DEFAULT_STATE = {
    CURRENT_STEP: 0,
    USER_INTENT: "not_sure_yet",
    STEP2_OPEN: False,
    GOAL_MODE: "",
    GOAL_SELECTED: False,
    POSITION_CONFIRMED: False,
    CONFIRMED_SNAPSHOT: False,
    PLAN_READY: False,
    PLAN_GENERATED: False,
    SHOW_PLAN_B: False,
    PLANNING_SNAPSHOT: {},
    DOWNSTREAM_INPUTS_DIRTY: False,
    PENDING_UPDATES: {},
    STEP1_PENDING_WIDGET_PATCH: {},
    STEP1_PENDING_SUCCESS_MESSAGE: "",
    STEP1_INCOME_AMOUNT: 460.0,
    STEP1_INCOME_PERIOD: "Weekly",
    STEP1_ESTIMATED_GROSS_INCOME: 30000.0,
    STEP1_ESTIMATED_GROSS_PERIOD: "Weekly",
    STEP1_ESTIMATED_TAX_RATE: 0.30,
    STEP1_FIXED_AMOUNT: 185.0,
    STEP1_FIXED_PERIOD: "Weekly",
    STEP1_VARIABLE_AMOUNT: 80.0,
    STEP1_VARIABLE_PERIOD: "Weekly",
    STEP1_DISCRETIONARY_AMOUNT: 35.0,
    STEP1_DISCRETIONARY_PERIOD: "Weekly",
    STEP1_TARGET_WEEKLY_SAVINGS: 0.0,
    STEP1_FIXED_BREAKDOWN_DF: _default_fixed_breakdown_df(),
    STEP1_VAR_UTILITIES_AMOUNT: 0.0,
    STEP1_VAR_UTILITIES_PERIOD: "Weekly",
    STEP1_VAR_GROCERIES_AMOUNT: 0.0,
    STEP1_VAR_GROCERIES_PERIOD: "Weekly",
    STEP1_VAR_HOUSEHOLD_AMOUNT: 0.0,
    STEP1_VAR_HOUSEHOLD_PERIOD: "Weekly",
    STEP1_VAR_COMMUTE_DAYS: 0,
    STEP1_VAR_COMMUTE_COST: 0.0,
    STEP1_VAR_SEASON: "Normal",
    STEP1_PLANNING_SNAPSHOT_PREVIEW: {},
    INCOME_W: 0.0,
    FIXED_TOTAL_W: 0.0,
    DISCRETIONARY_W: 0.0,
    WEEKLY_MARGIN: 0.0,
    FIXED_W: 0.0,
    VAR_W: 0.0,
    DISC_W: 0.0,
    FIXED_ITEMS_ROWS_WEEKLY: [],
    VARIABLE_ITEMS_ROWS_WEEKLY: [],
    VARIABLE_TOP_DRIVER: None,
    VARIABLE_PARTS: {},
    MONTHLY_INCOME_DERIVED: 0.0,
    ESSENTIAL_SPENDING_MONTHLY_DERIVED: 0.0,
    DISCRETIONARY_SPENDING_MONTHLY_DERIVED: 0.0,
    WEEKLY_SAVINGS_DERIVED: 0.0,
    STEP2_PLANNING_HORIZON_WEEKS: 12,
    STEP2_UNCERTAINTY_PRESET: "Quick estimate (default)",
    STEP2_RANDOM_RUN_NONCE: 0,
    STEP2_ENABLE_ONE_OFF_EVENTS: False,
    STEP2_ONE_OFF_EVENTS_DF: _default_one_off_events_df(),
    STEP2_SELECTED_PRESET: "Recommended",
    STEP2_TARGET_INITIALIZED: False,
    STEP2_TARGET_USER_TOUCHED_INTERNAL: False,
    STEP2_BASELINE_WEEKLY_SEEDED: -1.0,
    STEP2_BUFFER_GOAL_AMOUNT_INPUT: 1000.0,
    STEP2_BUFFER_GOAL_WEEKS_INPUT: 12,
    STEP2_BUFFER_GOAL_PRIORITY_INPUT: "Balanced",
    STEP2_SUGGESTION_FEEDBACK: "",
    BASELINE_DF: None,
    BASELINE_READY: False,
    BASELINE_SIGNATURE: None,
    PLAN_A_DF: None,
    PLAN_B_DF: None,
    TARGET_A_WEEKLY: 0.0,
    TARGET_A_WEEKLY_USER_TOUCHED: False,
    TARGET_B_WEEKLY: 0.0,
    ENGINE_HAS_RUN: False,
    PROJECTION_OPEN: False,
    ENGINE_RESULTS_OPEN: False,
    INVESTMENT_SETUP_OPEN: False,
    INVESTMENT_SETUP_GATE_ATTEMPTED: False,
    INVESTMENT_CONTEXT: {},
    INVESTMENT_MONTHLY_CONTRIBUTION: 0.0,
    INVESTMENT_WEEKLY_EQUIVALENT: 0.0,
    INVESTMENT_PHILOSOPHY: "Balanced",
    INVESTMENT_PHILOSOPHY_BUNDLE_SUMMARY: {},
    INVESTMENT_PROJECTION_PROFILE_HINT: "Balanced",
    INVESTMENT_PROJECTION_RESULT: {},
    INVESTMENT_PROJECTION_SIGNATURE: None,
    INVESTMENT_PROJECTION_COMPARE_ENABLED: False,
    INVESTMENT_PROJECTION_COMPARE_HORIZONS: [20, 30, 50],
    INVESTMENT_PROJECTION_COMPARE_RESULTS: {},
    INVESTMENT_PROJECTION_COMPARE_SIGNATURE: None,
    UNIVERSE_SIZE: 25,
    UNIVERSE_STRATEGY: "Core multi-asset",
    UNIVERSE_CUSTOM_ENABLED: False,
    UNIVERSE_CUSTOM_ENABLED_SOURCE: "",
    UNIVERSE_CUSTOM_ENABLED_RECOMMENDED_ASSETS: [],
    CUSTOM_UNIVERSE_TEXT: "",
    RECOMMENDED_UNIVERSE_ASSETS: [],
    LAST_USED_UNIVERSE_ASSETS: [],
    LAST_MISSING_UNIVERSE_ASSETS: [],
    LAST_RECOMMENDATION_CANDIDATE_ASSETS: [],
    LAST_COMPARISON_UNIVERSE_ASSETS: [],
    LAST_COMPARISON_MISSING_UNIVERSE_ASSETS: [],
    UNIVERSE_STRATEGY_LAST_RECOMMENDED: "",
    COMPARISON_ENABLED: False,
    UNIVERSE_WORKSPACE_MODE: "simple",
    UNIVERSE_ENGINE_SETUP_OPEN: False,
    UNIVERSE_SIMPLE_MODE_ENABLED: True,
    UNIVERSE_SIMPLE_STRATEGY_TEMPLATE: "Balanced Risk-Controlled",
    UNIVERSE_SIMPLE_STYLE_PRESET: "Balanced",
    UNIVERSE_SIMPLE_RESOLVED_CONFIG: {},
    UNIVERSE_SIMPLE_RESOLVED_CONSTRAINTS: {},
    UNIVERSE_SIMPLE_RESOLVED_SUMMARY: {},
    UNIVERSE_SIMPLE_GOVERNANCE_PAYLOAD: {},
    UNIVERSE_SIMPLE_GOVERNED_RESOLUTION: {},
    UNIVERSE_SIMPLE_RISK_APPETITE: 0.5,
    UNIVERSE_SIMPLE_DIVERSIFICATION: 0.5,
    UNIVERSE_SIMPLE_STABILITY: 0.5,
    UNIVERSE_SIMPLE_TURNOVER_PREF: 0.5,
    UNIVERSE_SIMPLE_DRAWDOWN_PROTECTION: 0.5,
    UNIVERSE_SIMPLE_OVERLAY_INTENSITY: 0.5,
    UNIVERSE_SIMPLE_SIGNAL_CONFIDENCE: 0.5,
    UNIVERSE_SIMPLE_SIMPLICITY: 0.5,
    UNIVERSE_SIMPLE_OVERRIDE_PATCH: {},
    UNIVERSE_SIMPLE_OVERRIDE_SCOPE_KEY: "",
    UNIVERSE_SIMPLE_OVERRIDE_SOURCE: "",
    LAST_PRE_RUN_ENGINE_CONFIG: {},
    LAST_PRE_RUN_ENGINE_FINGERPRINT: None,
    LAST_ENGINE_CONFIG: {},
    LAST_ENGINE_PHILOSOPHY: "",
    LAST_ENGINE_COHERENCE: {},
    LAST_APPLIED_ENGINE_CONFIG: {},
    LAST_APPLIED_ENGINE_FINGERPRINT: None,
    ADVANCED_GOVERNANCE_LAST_REPAIR_NOTE: "",
    UNIVERSE_PRE_RUN_GOVERNANCE_SNAPSHOT: {},
    UNIVERSE_DYNAMIC_SIZE_OPTIONS: [],
    UNIVERSE_SIZE_OPTIMISATION_RESULTS: {},
    UNIVERSE_SIZE_OPTIMISATION_SCOPE: "",
    UNIVERSE_SIMPLE_AUTO_OPTIMIZE_ENABLED: False,
    UNIVERSE_SIMPLE_AUTO_OPT_USED: False,
    UNIVERSE_SIMPLE_AUTO_OPT_LAST_RESULT: {},
    UNIVERSE_SIMPLE_AUTO_OPT_LAST_FINGERPRINT: None,
    SIMPLE_AUTO_OPT_RESULT: {},
    SIMPLE_AUTO_OPT_BASELINE_PAYLOAD: {},
    SIMPLE_AUTO_OPT_BEST_PAYLOAD: {},
    SIMPLE_AUTO_OPT_USED: False,
    UNIVERSE_LOCAL_REFINEMENT_HISTORY: [],
    UNIVERSE_LOCAL_REFINEMENT_HISTORY_LAST_SIGNATURE: None,
    UNIVERSE_GLOBAL_SEARCH_RUN_NONCE: 0,
    ASSET_SOURCE_MODE: "yahoo",
    ASSET_START_DATE: None,
    ASSET_END_DATE: None,
    ASSET_RETURN_FREQUENCY: "monthly",
    ASSET_AUTO_ADJUST: True,
    ASSET_UPLOADED_FILE: None,
    ASSET_PANEL_DF: None,
    ASSET_PANEL_READY: False,
    ASSET_PANEL_SOURCE_LABEL: "",
    ASSET_PANEL_ERROR: "",
    INVESTMENT_START_DATE_STABILITY_ENABLED: False,
    INVESTMENT_START_DATE_STABILITY_DATES: list(START_DATE_STABILITY_TEST_OPTIONS),
    INVESTMENT_START_DATE_STABILITY_RESULTS: {},
    INVESTMENT_START_DATE_STABILITY_SIGNATURE: "",
}

_BOOTSTRAP_EXCLUDED_WIDGET_KEYS = set(STEP1_WIDGET_KEYS) | set(STEP2_WIDGET_KEYS)


def _clone_default(value: Any) -> Any:
    try:
        return deepcopy(value)
    except Exception:
        return value


def bootstrap_session_state(extra_defaults: Mapping[str, Any] | None = None, *, overwrite: bool = False) -> None:
    defaults = dict(DEFAULT_STATE)
    if isinstance(extra_defaults, Mapping):
        defaults.update(dict(extra_defaults))
    for key, value in defaults.items():
        if not overwrite and key in _BOOTSTRAP_EXCLUDED_WIDGET_KEYS:
            continue
        if overwrite or key not in st.session_state:
            st.session_state[key] = _clone_default(value)


def bootstrap_keys(defaults: Mapping[str, Any], *, overwrite: bool = False) -> None:
    if not isinstance(defaults, Mapping):
        return
    for key, value in defaults.items():
        if overwrite or key not in st.session_state:
            st.session_state[key] = _clone_default(value)


def reset_projection_state(*, clear_compare: bool = True) -> None:
    st.session_state[PROJECTION_OPEN] = False
    st.session_state[INVESTMENT_PROJECTION_RESULT] = {}
    st.session_state[INVESTMENT_PROJECTION_SIGNATURE] = None
    if clear_compare:
        st.session_state[INVESTMENT_PROJECTION_COMPARE_ENABLED] = False
        st.session_state[INVESTMENT_PROJECTION_COMPARE_RESULTS] = {}
        st.session_state[INVESTMENT_PROJECTION_COMPARE_SIGNATURE] = None


def reset_plan_state() -> None:
    for key in PLAN_RESET_KEYS:
        if key in DEFAULT_STATE:
            st.session_state[key] = _clone_default(DEFAULT_STATE[key])


def debug_state(keys: list[str] | tuple[str, ...] | None = None) -> dict[str, Any]:
    if keys:
        return {k: st.session_state.get(k) for k in keys}
    return dict(st.session_state)
