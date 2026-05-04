"""Canonical Streamlit ``session_state`` keys for LifeBudget Micro.

This module is the state contract for the app.

Streamlit reruns the script from top to bottom after most user interactions, so
shared state is stored in ``st.session_state``. Defining keys as constants avoids
hard-coded string literals across the UI, reduces typo-related bugs, and makes
cross-module state easier to inspect.

The keys are grouped by workflow area:

- global wizard / flow state;
- Personal Finance Planner inputs and derived values;
- legacy short-term planning state from earlier iterations;
- investment, universe and market-data setup;
- Strategy Engine configuration, diagnostics and suggestions;
- Long-Term Scenario outputs;
- reset groups used by bootstrap and cleanup helpers.

Some keys are retained for backwards compatibility with earlier prototypes or
optional diagnostic/research layers. The submitted app uses the main end-to-end
flow, but these constants remain part of the state contract so old state,
Streamlit reruns and optional panels do not fail unexpectedly.
"""


# ---------------------------------------------------------------------
# Queued state updates
# ---------------------------------------------------------------------
# Buttons, preset applications and recommendation actions sometimes need to
# update widget-owned values. Streamlit can raise errors if widget keys are
# mutated after the widget has already been instantiated, so these updates are
# queued and applied at the start of the next rerun by ui.state.updates.
PENDING_UPDATES = "_pending_session_updates"


# ---------------------------------------------------------------------
# Global flow / wizard state
# ---------------------------------------------------------------------
CURRENT_STEP = "current_step"
USER_INTENT = "user_intent"
STEP2_OPEN = "step2_open"
GOAL_MODE = "goal_mode"
GOAL_SELECTED = "goal_selected"
POSITION_CONFIRMED = "position_confirmed"
CONFIRMED_SNAPSHOT = "confirmed_snapshot"
PLAN_READY = "plan_ready"
PLAN_GENERATED = "plan_generated"
SHOW_PLAN_B = "show_plan_b"
PLANNING_SNAPSHOT = "planning_snapshot"
DOWNSTREAM_INPUTS_DIRTY = "downstream_inputs_dirty"


# ---------------------------------------------------------------------
# Step 1 — Personal Finance Planner widget/input state
# ---------------------------------------------------------------------
STEP1_PENDING_WIDGET_PATCH = "step1_pending_widget_patch"
STEP1_PENDING_SUCCESS_MESSAGE = "step1_pending_success_message"
STEP1_INCOME_AMOUNT = "step1_income_amount"
STEP1_INCOME_PERIOD = "step1_income_period"
STEP1_ESTIMATED_GROSS_INCOME = "step1_estimated_gross_income"
STEP1_ESTIMATED_GROSS_PERIOD = "step1_estimated_gross_period"
STEP1_ESTIMATED_TAX_RATE = "step1_estimated_tax_rate"
STEP1_FIXED_AMOUNT = "step1_fixed_amount"
STEP1_FIXED_PERIOD = "step1_fixed_period"
STEP1_VARIABLE_AMOUNT = "step1_variable_amount"
STEP1_VARIABLE_PERIOD = "step1_variable_period"
STEP1_DISCRETIONARY_AMOUNT = "step1_discretionary_amount"
STEP1_DISCRETIONARY_PERIOD = "step1_discretionary_period"
STEP1_TARGET_WEEKLY_SAVINGS = "step1_target_weekly_savings"
STEP1_FIXED_BREAKDOWN_DF = "step1_fixed_breakdown_df"

# Optional variable-essentials breakdown controls.
STEP1_VAR_UTILITIES_AMOUNT = "step1_var_utilities_amount"
STEP1_VAR_UTILITIES_PERIOD = "step1_var_utilities_period"
STEP1_VAR_GROCERIES_AMOUNT = "step1_var_groceries_amount"
STEP1_VAR_GROCERIES_PERIOD = "step1_var_groceries_period"
STEP1_VAR_HOUSEHOLD_AMOUNT = "step1_var_household_amount"
STEP1_VAR_HOUSEHOLD_PERIOD = "step1_var_household_period"
STEP1_VAR_COMMUTE_DAYS = "step1_var_commute_days"
STEP1_VAR_COMMUTE_COST = "step1_var_commute_cost"
STEP1_VAR_SEASON = "step1_var_season"

# Historical/legacy naming note:
# The constant is Step-1-specific, but the stored string predates the current
# module naming and is kept unchanged for compatibility with existing state.
STEP1_PLANNING_SNAPSHOT_PREVIEW = "planning_snapshot_preview"


# ---------------------------------------------------------------------
# Step 2 / Step 3 — legacy short-term planning state
# ---------------------------------------------------------------------
# These keys support earlier goal-planning and short-term simulation screens.
# They are retained so reset logic, old session state and optional diagnostics
# can still work without breaking the current modular UI.
STEP2_PLANNING_HORIZON_WEEKS = "step2_planning_horizon_weeks"
STEP2_UNCERTAINTY_PRESET = "step2_uncertainty_preset"
STEP2_RANDOM_RUN_NONCE = "step2_random_run_nonce"
STEP2_ENABLE_ONE_OFF_EVENTS = "step2_enable_one_off_events"
STEP2_ONE_OFF_EVENTS_DF = "step2_one_off_events_df"
STEP2_SELECTED_PRESET = "step2_selected_preset"
STEP2_TARGET_INITIALIZED = "step2_target_initialized"
STEP2_TARGET_USER_TOUCHED_INTERNAL = "step2_target_user_touched"
STEP2_BASELINE_WEEKLY_SEEDED = "step2_baseline_weekly_seeded"
STEP2_BUFFER_GOAL_AMOUNT_INPUT = "step2_buffer_goal_amount_input"
STEP2_BUFFER_GOAL_WEEKS_INPUT = "step2_buffer_goal_weeks_input"
STEP2_BUFFER_GOAL_PRIORITY_INPUT = "step2_buffer_goal_priority_input"
STEP2_SUGGESTION_FEEDBACK = "step2_suggestion_feedback"


# ---------------------------------------------------------------------
# Derived Personal Finance values
# ---------------------------------------------------------------------
# These values are calculated from Step 1 inputs and are reused later as the
# contribution bridge for investment testing, long-term projection and reports.
INCOME_W = "income_w"
FIXED_TOTAL_W = "fixed_total_w"
DISCRETIONARY_W = "discretionary_w"
WEEKLY_MARGIN = "weekly_margin"
FIXED_W = "fixed_w"
VAR_W = "var_w"
DISC_W = "disc_w"
FIXED_ITEMS_ROWS_WEEKLY = "fixed_items_rows_weekly"
VARIABLE_ITEMS_ROWS_WEEKLY = "variable_items_rows_weekly"
VARIABLE_TOP_DRIVER = "variable_top_driver"
VARIABLE_PARTS = "variable_parts"
MONTHLY_INCOME_DERIVED = "monthly_income_derived"
ESSENTIAL_SPENDING_MONTHLY_DERIVED = "essential_spending_monthly_derived"
DISCRETIONARY_SPENDING_MONTHLY_DERIVED = "discretionary_spending_monthly_derived"
WEEKLY_SAVINGS_DERIVED = "weekly_savings_derived"


# ---------------------------------------------------------------------
# Legacy baseline / plan outputs
# ---------------------------------------------------------------------
# These keys belong to the earlier planning flow and remain useful for reset
# behaviour, backwards compatibility and any screens that still inspect Plan A
# or Plan B style outputs.
BASELINE_DF = "baseline_df"
BASELINE_READY = "baseline_ready"
BASELINE_SIGNATURE = "baseline_signature"
PLAN_A_DF = "plan_a_df"
PLAN_B_DF = "plan_b_df"
TARGET_A_WEEKLY = "target_a_weekly"
TARGET_A_WEEKLY_USER_TOUCHED = "target_a_weekly_user_touched"
TARGET_B_WEEKLY = "target_b_weekly"


# ---------------------------------------------------------------------
# Investment / projection bridge
# ---------------------------------------------------------------------
# These keys connect Personal Finance, Strategy Engine, Long-Term Scenario and
# Final Report. They represent the user's contribution bridge, investment
# philosophy and projection state.
ENGINE_HAS_RUN = "engine_has_run"
PROJECTION_OPEN = "projection_open"
ENGINE_RESULTS_OPEN = "engine_results_open"
INVESTMENT_SETUP_OPEN = "investment_setup_open"
INVESTMENT_SETUP_GATE_ATTEMPTED = "investment_setup_gate_attempted"
INVESTMENT_CONTEXT = "investment_context"
INVESTMENT_MONTHLY_CONTRIBUTION = "investment_monthly_contribution"
INVESTMENT_WEEKLY_EQUIVALENT = "investment_weekly_equivalent"
INVESTMENT_PHILOSOPHY = "investment_philosophy"
INVESTMENT_PHILOSOPHY_BUNDLE_SUMMARY = "investment_philosophy_bundle_summary"
INVESTMENT_PROJECTION_PROFILE_HINT = "investment_projection_profile_hint"
INVESTMENT_PROJECTION_RESULT = "investment_projection_result"
INVESTMENT_PROJECTION_SIGNATURE = "investment_projection_signature"
INVESTMENT_PROJECTION_COMPARE_ENABLED = "investment_projection_compare_enabled"
INVESTMENT_PROJECTION_COMPARE_HORIZONS = "investment_projection_compare_horizons"
INVESTMENT_PROJECTION_COMPARE_RESULTS = "investment_projection_compare_results"
INVESTMENT_PROJECTION_COMPARE_SIGNATURE = "investment_projection_compare_signature"


# ---------------------------------------------------------------------
# Risk Profile and Asset Universe setup
# ---------------------------------------------------------------------
# These keys define the asset universe available to the Strategy Engine.
# The app can use recommended baskets or a user-edited custom override.
UNIVERSE_SIZE = "universe_size"
UNIVERSE_STRATEGY = "universe_strategy"
UNIVERSE_CUSTOM_ENABLED = "universe_custom_enabled"
UNIVERSE_CUSTOM_ENABLED_SOURCE = "universe_custom_enabled_source"
UNIVERSE_CUSTOM_ENABLED_RECOMMENDED_ASSETS = "universe_custom_enabled_recommended_assets"
CUSTOM_UNIVERSE_TEXT = "custom_universe_text"
RECOMMENDED_UNIVERSE_ASSETS = "recommended_universe_assets"
LAST_USED_UNIVERSE_ASSETS = "last_used_universe_assets"
LAST_MISSING_UNIVERSE_ASSETS = "last_missing_universe_assets"
LAST_RECOMMENDATION_CANDIDATE_ASSETS = "last_recommendation_candidate_assets"
LAST_COMPARISON_UNIVERSE_ASSETS = "last_comparison_universe_assets"
LAST_COMPARISON_MISSING_UNIVERSE_ASSETS = "last_comparison_missing_universe_assets"
UNIVERSE_STRATEGY_LAST_RECOMMENDED = "universe_strategy_last_recommended"
COMPARISON_ENABLED = "comparison_enabled"


# ---------------------------------------------------------------------
# Simplified Strategy Engine setup and governance state
# ---------------------------------------------------------------------
# The final UI exposes a simplified Strategy Engine workspace, but the app
# stores resolved configuration, constraints and governance payloads so the
# result can be explained, diagnosed and reused later.
UNIVERSE_WORKSPACE_MODE = "universe_workspace_mode"
UNIVERSE_ENGINE_SETUP_OPEN = "universe_engine_setup_open"
UNIVERSE_SIMPLE_MODE_ENABLED = "universe_simple_mode_enabled"
UNIVERSE_SIMPLE_STRATEGY_TEMPLATE = "universe_simple_strategy_template"
UNIVERSE_SIMPLE_STYLE_PRESET = "universe_simple_style_preset"
UNIVERSE_SIMPLE_RESOLVED_CONFIG = "universe_simple_resolved_config"
UNIVERSE_SIMPLE_RESOLVED_CONSTRAINTS = "universe_simple_resolved_constraints"
UNIVERSE_SIMPLE_RESOLVED_SUMMARY = "universe_simple_resolved_summary"
UNIVERSE_SIMPLE_GOVERNANCE_PAYLOAD = "universe_simple_governance_payload"
UNIVERSE_SIMPLE_GOVERNED_RESOLUTION = "universe_simple_governed_resolution"

# User-facing semantic/posture sliders for the simplified Strategy Engine UI.
UNIVERSE_SIMPLE_RISK_APPETITE = "universe_simple_risk_appetite"
UNIVERSE_SIMPLE_DIVERSIFICATION = "universe_simple_diversification"
UNIVERSE_SIMPLE_STABILITY = "universe_simple_stability"
UNIVERSE_SIMPLE_TURNOVER_PREF = "universe_simple_turnover_pref"
UNIVERSE_SIMPLE_DRAWDOWN_PROTECTION = "universe_simple_drawdown_protection"
UNIVERSE_SIMPLE_OVERLAY_INTENSITY = "universe_simple_overlay_intensity"
UNIVERSE_SIMPLE_SIGNAL_CONFIDENCE = "universe_simple_signal_confidence"
UNIVERSE_SIMPLE_SIMPLICITY = "universe_simple_simplicity"

# Optional override patch metadata used when semantic presets are adjusted or
# when recommendation actions apply a controlled configuration change.
UNIVERSE_SIMPLE_OVERRIDE_PATCH = "universe_simple_override_patch"
UNIVERSE_SIMPLE_OVERRIDE_SCOPE_KEY = "universe_simple_override_scope_key"
UNIVERSE_SIMPLE_OVERRIDE_SOURCE = "universe_simple_override_source"

# Last engine configuration snapshots and fingerprints. These allow the app to
# distinguish the latest run from a changed setup and keep the sidebar/report
# honest about which result is currently stored.
LAST_PRE_RUN_ENGINE_CONFIG = "last_pre_run_engine_config"
LAST_PRE_RUN_ENGINE_FINGERPRINT = "last_pre_run_engine_fingerprint"
LAST_ENGINE_CONFIG = "last_engine_config"
LAST_ENGINE_PHILOSOPHY = "last_engine_philosophy"
LAST_ENGINE_COHERENCE = "last_engine_coherence"
LAST_APPLIED_ENGINE_CONFIG = "last_applied_engine_config"
LAST_APPLIED_ENGINE_FINGERPRINT = "last_applied_engine_fingerprint"
ADVANCED_GOVERNANCE_LAST_REPAIR_NOTE = "advanced_governance_last_repair_note"
UNIVERSE_PRE_RUN_GOVERNANCE_SNAPSHOT = "universe_pre_run_governance_snapshot"
UNIVERSE_DYNAMIC_SIZE_OPTIONS = "universe_dynamic_size_options"
UNIVERSE_SIZE_OPTIMISATION_RESULTS = "universe_size_optimisation_results"
UNIVERSE_SIZE_OPTIMISATION_SCOPE = "universe_size_optimisation_scope"


# ---------------------------------------------------------------------
# Legacy / optional auto-optimisation and refinement state
# ---------------------------------------------------------------------
# These keys come from earlier experimental recommendation layers. They are
# retained for compatibility with optional diagnostics and future extension,
# but the submitted UI presents suggestions as controlled educational checks
# rather than automatic financial advice.
UNIVERSE_SIMPLE_AUTO_OPTIMIZE_ENABLED = "universe_simple_auto_optimize_enabled"
UNIVERSE_SIMPLE_AUTO_OPT_USED = "universe_simple_auto_opt_used"
UNIVERSE_SIMPLE_AUTO_OPT_LAST_RESULT = "universe_simple_auto_opt_last_result"
UNIVERSE_SIMPLE_AUTO_OPT_LAST_FINGERPRINT = "universe_simple_auto_opt_last_fingerprint"
SIMPLE_AUTO_OPT_RESULT = "simple_auto_opt_result"
SIMPLE_AUTO_OPT_BASELINE_PAYLOAD = "simple_auto_opt_baseline_payload"
SIMPLE_AUTO_OPT_BEST_PAYLOAD = "simple_auto_opt_best_payload"
SIMPLE_AUTO_OPT_USED = "simple_auto_opt_used"
UNIVERSE_LOCAL_REFINEMENT_HISTORY = "universe_local_refinement_history"
UNIVERSE_LOCAL_REFINEMENT_HISTORY_LAST_SIGNATURE = "universe_local_refinement_history_last_signature"
UNIVERSE_GLOBAL_SEARCH_RUN_NONCE = "universe_global_search_run_nonce"


# ---------------------------------------------------------------------
# Asset-panel / market-data state
# ---------------------------------------------------------------------
# The hosted deployment uses prepared/cached market-data panels for reliability.
# These keys still support local/live Yahoo configuration and uploaded-file
# state where available.
ASSET_SOURCE_MODE = "asset_source_mode"
ASSET_START_DATE = "asset_start_date"
ASSET_END_DATE = "asset_end_date"
ASSET_RETURN_FREQUENCY = "asset_return_frequency"
ASSET_AUTO_ADJUST = "asset_auto_adjust"
ASSET_UPLOADED_FILE = "asset_uploaded_file"
ASSET_PANEL_DF = "asset_panel_df"
ASSET_PANEL_READY = "asset_panel_ready"
ASSET_PANEL_SOURCE_LABEL = "asset_panel_source_label"
ASSET_PANEL_ERROR = "asset_panel_error"


# ---------------------------------------------------------------------
# Strategy robustness / start-date stability diagnostics
# ---------------------------------------------------------------------
# These keys support optional reliability checks in the Strategy Engine.
# They test sensitivity to historical start windows; they do not make the
# backtest predictive.
INVESTMENT_START_DATE_STABILITY_ENABLED = "investment_start_date_stability_enabled"
INVESTMENT_START_DATE_STABILITY_DATES = "investment_start_date_stability_dates"
INVESTMENT_START_DATE_STABILITY_RESULTS = "investment_start_date_stability_results"
INVESTMENT_START_DATE_STABILITY_SIGNATURE = "investment_start_date_stability_signature"
STEP5_START_DATE_HISTORY = "step5_start_date_history"

# Default historical start windows for optional robustness diagnostics.
START_DATE_STABILITY_TEST_OPTIONS = ["2005-01-01", "2010-01-01", "2016-01-01"]


# ---------------------------------------------------------------------
# Key groups used by bootstrap/reset helpers
# ---------------------------------------------------------------------
# These lists define reset boundaries. Keeping them explicit makes it easier
# to reset only the relevant part of the workflow without wiping the entire
# Streamlit session.
FLOW_KEYS = [
    CURRENT_STEP,
    USER_INTENT,
    STEP2_OPEN,
    GOAL_MODE,
    GOAL_SELECTED,
    POSITION_CONFIRMED,
    CONFIRMED_SNAPSHOT,
    PLAN_READY,
    PLAN_GENERATED,
    SHOW_PLAN_B,
    PLANNING_SNAPSHOT,
    DOWNSTREAM_INPUTS_DIRTY,
]

# Widget-owned Step 1 keys.
# bootstrap.py excludes these from normal bootstrap writes because Streamlit can
# raise errors when widget state is mutated after widget creation.
STEP1_WIDGET_KEYS = [
    STEP1_INCOME_AMOUNT,
    STEP1_INCOME_PERIOD,
    STEP1_ESTIMATED_GROSS_INCOME,
    STEP1_ESTIMATED_GROSS_PERIOD,
    STEP1_ESTIMATED_TAX_RATE,
    STEP1_FIXED_AMOUNT,
    STEP1_FIXED_PERIOD,
    STEP1_VARIABLE_AMOUNT,
    STEP1_VARIABLE_PERIOD,
    STEP1_DISCRETIONARY_AMOUNT,
    STEP1_DISCRETIONARY_PERIOD,
    STEP1_TARGET_WEEKLY_SAVINGS,
    STEP1_FIXED_BREAKDOWN_DF,
    STEP1_VAR_UTILITIES_AMOUNT,
    STEP1_VAR_UTILITIES_PERIOD,
    STEP1_VAR_GROCERIES_AMOUNT,
    STEP1_VAR_GROCERIES_PERIOD,
    STEP1_VAR_HOUSEHOLD_AMOUNT,
    STEP1_VAR_HOUSEHOLD_PERIOD,
    STEP1_VAR_COMMUTE_DAYS,
    STEP1_VAR_COMMUTE_COST,
    STEP1_VAR_SEASON,
]
STEP1_PRIVATE_KEYS = [
    STEP1_PENDING_WIDGET_PATCH,
    STEP1_PENDING_SUCCESS_MESSAGE,
    STEP1_PLANNING_SNAPSHOT_PREVIEW,
]
STEP1_KEYS = STEP1_WIDGET_KEYS + STEP1_PRIVATE_KEYS

# Derived finance values calculated from Step 1 inputs and shared downstream.
STEP1_DERIVED_KEYS = [
    INCOME_W,
    FIXED_TOTAL_W,
    DISCRETIONARY_W,
    WEEKLY_MARGIN,
    FIXED_W,
    VAR_W,
    DISC_W,
    FIXED_ITEMS_ROWS_WEEKLY,
    VARIABLE_ITEMS_ROWS_WEEKLY,
    VARIABLE_TOP_DRIVER,
    VARIABLE_PARTS,
    MONTHLY_INCOME_DERIVED,
    ESSENTIAL_SPENDING_MONTHLY_DERIVED,
    DISCRETIONARY_SPENDING_MONTHLY_DERIVED,
    WEEKLY_SAVINGS_DERIVED,
]

# Widget-owned Step 2 keys retained for compatibility with the earlier planning
# flow.
STEP2_WIDGET_KEYS = [
    STEP2_PLANNING_HORIZON_WEEKS,
    STEP2_UNCERTAINTY_PRESET,
    STEP2_RANDOM_RUN_NONCE,
    STEP2_ENABLE_ONE_OFF_EVENTS,
    STEP2_ONE_OFF_EVENTS_DF,
    STEP2_BUFFER_GOAL_AMOUNT_INPUT,
    STEP2_BUFFER_GOAL_WEEKS_INPUT,
    STEP2_BUFFER_GOAL_PRIORITY_INPUT,
]
STEP2_PRIVATE_KEYS = [
    STEP2_SELECTED_PRESET,
    STEP2_TARGET_INITIALIZED,
    STEP2_TARGET_USER_TOUCHED_INTERNAL,
    STEP2_BASELINE_WEEKLY_SEEDED,
    STEP2_SUGGESTION_FEEDBACK,
]
STEP2_KEYS = STEP2_WIDGET_KEYS + STEP2_PRIVATE_KEYS

BASELINE_KEYS = [
    BASELINE_DF,
    BASELINE_READY,
    BASELINE_SIGNATURE,
    PLAN_A_DF,
    PLAN_B_DF,
    TARGET_A_WEEKLY,
    TARGET_A_WEEKLY_USER_TOUCHED,
    TARGET_B_WEEKLY,
]

PROJECTION_KEYS = [
    PROJECTION_OPEN,
    INVESTMENT_PROJECTION_RESULT,
    INVESTMENT_PROJECTION_SIGNATURE,
    INVESTMENT_PROJECTION_COMPARE_ENABLED,
    INVESTMENT_PROJECTION_COMPARE_HORIZONS,
    INVESTMENT_PROJECTION_COMPARE_RESULTS,
    INVESTMENT_PROJECTION_COMPARE_SIGNATURE,
]

STEP1_RESET_KEYS = STEP1_WIDGET_KEYS + STEP1_PRIVATE_KEYS + STEP1_DERIVED_KEYS
STEP2_RESET_KEYS = STEP2_WIDGET_KEYS + STEP2_PRIVATE_KEYS
PLAN_RESET_KEYS = FLOW_KEYS + BASELINE_KEYS + STEP1_RESET_KEYS + STEP2_RESET_KEYS


# ---------------------------------------------------------------------
# Step 5 semantic Strategy Engine setup keys
# ---------------------------------------------------------------------
# These support user-facing Strategy Engine controls: template, style and broad
# posture sliders. They are separate from lower-level engine overrides.
STEP5_TEMPLATE = "step5_template"
STEP5_STYLE = "step5_style"
STEP5_SEM_RISK_APPETITE = "step5_sem_risk_appetite"
STEP5_SEM_DRAWDOWN_PROTECTION = "step5_sem_drawdown_protection"
STEP5_SEM_DIVERSIFICATION = "step5_sem_diversification"
STEP5_SEM_OVERLAY_INTENSITY = "step5_sem_overlay_intensity"
STEP5_SEM_STABILITY = "step5_sem_stability"
STEP5_SEM_CONFIDENCE_SIGNAL = "step5_sem_confidence_signal"
STEP5_SEM_TURNOVER_STYLE = "step5_sem_turnover_style"
STEP5_SEM_SIMPLICITY = "step5_sem_simplicity"
STEP5_SEMANTIC_USER_TOUCHED = "step5_semantic_user_touched"
STEP5_SEMANTIC_LAST_SEED_SIGNATURE = "step5_semantic_last_seed_signature"