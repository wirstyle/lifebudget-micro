# app.py (FULL)
# ------------------------------------------------------------
# LifeBudget Micro — main UI / orchestration layer
#
# Key concepts:
# - Step 1: describe current position (no behaviour change)
# - Step 2: scenarios are SAVINGS TARGETS, not category cuts
# - Coverage order: remaining margin → discretionary
# - Essentials are never auto-modified (MVP scope)
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.baseline import generate_baseline, validate_baseline_df

from src.compounder import (
    simulate_scenario_df_with_seed_offset,
    apply_shock_map_to_df,
)

from src.explain import (
    ExplanationInputs,
    build_explanation,
    compute_reflection_metrics,
    build_human_reflection_text,
)

from src.expenses import (
    total_weekly_from_items,
    default_fixed_items_rows,
    variable_essentials_weekly_total,
    discretionary_preset_value,
    clean_events,
    events_to_weekly_shock_map,
)

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(page_title="LifeBudget Micro", layout="centered")

# -----------------------
# Helpers: period -> weekly
# -----------------------
PERIOD_TO_WEEK = {
    "Weekly": 1.0,
    "Monthly": 12.0 / 52.0,
    "Yearly": 1.0 / 52.0,
}

def to_weekly(amount: float, period: str) -> float:
    return float(amount) * float(PERIOD_TO_WEEK[period])

def weekly_to_period(amount_w: float, period: str) -> float:
    """Convert weekly -> Weekly/Monthly/Yearly (inverse of to_weekly)."""
    if period == "Weekly":
        return float(amount_w)
    if period == "Monthly":
        return float(amount_w) * (52.0 / 12.0)
    if period == "Yearly":
        return float(amount_w) * 52.0
    raise ValueError(f"Unknown period: {period}")

# ------------------------------------------------------------
# Session state helper
# ------------------------------------------------------------
def ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

def make_baseline_signature():
    return {
        "income_w": round(float(st.session_state["income_w"]), 6),
        "fixed_total_w": round(float(st.session_state["fixed_total_w"]), 6),
        "discretionary_w": round(float(st.session_state["discretionary_w"]), 6),
        "weeks": int(st.session_state["weeks"]),
        "shock_enabled": bool(st.session_state.get("shock_enabled", False)),
        "shock_events": tuple(
            (float(r.get("amount", 0.0)), int(r.get("week", 0)))
            for r in st.session_state.get("shock_events_rows", [])
            if float(r.get("amount", 0.0)) > 0
        ),
    }

# ------------------------------------------------------------
# Session state defaults
# ------------------------------------------------------------
ss("baseline_df", None)
ss("scenario_a", None)
ss("scenario_b", None)
ss("seed", 42)
ss("baseline_ready", False)
ss("baseline_signature", None)
ss("weeks", 12)

ss("saved_a_msg", False)
ss("saved_b_msg", False)
ss("cleared_msg", False)

# Raw inputs (widget-backed)
ss("income_raw", 460.0)
ss("income_period", "Weekly")
ss("fixed_essential_raw", 185.0)
ss("fixed_essential_period", "Weekly")
ss("variable_essential_raw", 80.0)
ss("variable_essential_period", "Weekly")
ss("discretionary_raw", 35.0)
ss("discretionary_period", "Weekly")

# Estimator state (optional)
ss("gross_annual_est", 30000.0)
ss("net_ratio_pct", 70)
ss("income_est_applied", False)

# Weekly normalised (computed each run in Step 1)
ss("income_w", 0.0)
ss("fixed_total_w", 0.0)
ss("discretionary_w", 0.0)
ss("weekly_margin", 0.0)

# Shock events (new model)
ss("shock_enabled", False)
ss("shock_events_rows", [
    {"name": "", "amount": 0.0, "week": 1},
    {"name": "", "amount": 0.0, "week": 1},
    {"name": "", "amount": 0.0, "week": 1},
])

# ✅ Expense builder (single source of truth = list[dict])
ss("fixed_items_rows", default_fixed_items_rows())

# Variable essentials breakdown
ss("var_utilities_base", 0.0)
ss("var_utilities_period", "Weekly")
ss("var_season", "Normal")
ss("var_commute_days", 0)
ss("var_commute_cost", 0.0)
ss("var_groceries", 0.0)
ss("var_groceries_period", "Weekly")
ss("var_household", 0.0)
ss("var_household_period", "Weekly")

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title("LifeBudget Micro")
st.subheader("Short-term financial clarity")
st.caption("Describe your current position first. Explore change later.")

# ------------------------------------------------------------
# Step 1 helper callbacks (MUST be defined before buttons use them)
# ------------------------------------------------------------
def apply_fixed_total(total_fixed_weekly: float):
    st.session_state["fixed_essential_raw"] = float(total_fixed_weekly)
    st.session_state["fixed_essential_period"] = "Weekly"
    st.toast("Fixed essentials updated", icon="✅")

def apply_variable_total(total_var_weekly: float):
    st.session_state["variable_essential_raw"] = float(total_var_weekly)
    st.session_state["variable_essential_period"] = "Weekly"
    st.toast("Variable essentials updated", icon="✅")

def apply_estimated_income(net_weekly: float):
    st.session_state["income_raw"] = float(net_weekly)
    st.session_state["income_period"] = "Weekly"
    st.session_state["income_est_applied"] = True
    st.toast("Income updated from estimator", icon="✅")

def apply_discretionary_preset(preset_name: str):
    st.session_state["discretionary_raw"] = float(discretionary_preset_value(preset_name))
    st.session_state["discretionary_period"] = "Weekly"
    st.toast(f"Preset applied: {preset_name}", icon="✅")

# ============================================================
# STEP 1 — CURRENT POSITION
# ============================================================
st.header("Step 1 — Understand your current position")

# Income
c1, c2 = st.columns([2, 1])
with c1:
    st.number_input(
        "Take-home income (£)",
        min_value=0.0,
        step=10.0,
        key="income_raw",
        help="Net (after tax) income. Use the estimator below if you only know your gross income.",
    )
with c2:
    st.selectbox(
        "Income period",
        ["Weekly", "Monthly", "Yearly"],
        key="income_period",
        help="How often you receive this income.",
    )

# ------------------------------------------------------------
# Optional: take-home estimator (rough)
# - Slider is full width and uses %
# - Avoids Streamlit warning by not providing both a default and a session_state assignment
# ------------------------------------------------------------
with st.expander("Estimate take-home income (optional)"):
    st.caption(
        "Quick estimator to produce a usable *net* figure. "
        "This is a rough estimate (you can override anytime)."
    )

    st.number_input(
        "Gross income (£/year)",
        min_value=0.0,
        step=500.0,
        key="gross_annual_est",
        help="Gross annual income before tax/deductions.",
    )

    # Slider full width, in %
    st.slider(
        "Estimated net ratio (%)",
        min_value=40,
        max_value=90,
        step=1,
        key="net_ratio_pct",
        format="%d%%",
        help="If unsure, 65–75% is a common ballpark depending on your situation.",
    )

    net_ratio = float(st.session_state["net_ratio_pct"]) / 100.0
    net_annual = float(st.session_state["gross_annual_est"]) * net_ratio
    net_weekly = net_annual / 52.0
    net_monthly = net_annual / 12.0

    st.markdown(
        f"**Estimated take-home:** £{net_weekly:,.2f}/week · £{net_monthly:,.2f}/month · £{net_annual:,.0f}/year"
    )

    st.button(
        "Use this estimate as my Take-home income",
        key="use_income_est_btn",
        on_click=apply_estimated_income,
        args=(net_weekly,),
    )

# ------------------------------------------------------------
# Fixed essentials
# ------------------------------------------------------------
e1, e2 = st.columns([2, 1])
with e1:
    st.number_input(
        "Fixed essential commitments (£)",
        min_value=0.0,
        step=10.0,
        key="fixed_essential_raw",
        help="Recurring essentials you cannot easily change (rent, council tax, insurance, etc.).",
    )
with e2:
    st.selectbox(
        "Period",
        ["Weekly", "Monthly", "Yearly"],
        key="fixed_essential_period",
        help="How often you pay these fixed essentials.",
    )

with st.expander("Build my fixed expenses (optional)"):
    # IMPORTANT: give the editor its own widget key (so Streamlit owns that widget state)
    fixed_rows = st.data_editor(
        st.session_state["fixed_items_rows"],
        hide_index=True,
        num_rows="dynamic",
        key="fixed_items_editor",
        column_config={
            "name": st.column_config.TextColumn("Name"),
            "amount": st.column_config.NumberColumn("Amount (£)", min_value=0.0),
            "period": st.column_config.SelectboxColumn(
                "Period",
                options=["Weekly", "Monthly", "Yearly"],
            ),
        },
        use_container_width=True,
    )

    # Persist back to your storage key (NOT the widget key)
    st.session_state["fixed_items_rows"] = fixed_rows

    # total_weekly_from_items expects list[dict] → fixed_rows is already that
    total_fixed = total_weekly_from_items(fixed_rows)
    st.markdown(f"**Estimated total:** £{total_fixed:,.2f}/week")

    st.button(
        "Use this total for Fixed essentials",
        key="use_fixed_total_btn",
        on_click=apply_fixed_total,
        args=(total_fixed,),
    )

# ------------------------------------------------------------
# Variable essentials
# ------------------------------------------------------------
e3, e4 = st.columns([2, 1])
with e3:
    st.number_input(
        "Variable essentials (£)",
        min_value=0.0,
        step=5.0,
        key="variable_essential_raw",
        help="Essentials that vary week-to-week (utilities, groceries, commuting, household basics).",
    )
with e4:
    st.selectbox(
        "Period ",
        ["Weekly", "Monthly", "Yearly"],
        key="variable_essential_period",
        help="How often you spend this amount.",
    )

with st.expander("Rough breakdown for variable essentials (optional)"):
    st.number_input("Utilities baseline (£)", min_value=0.0, step=5.0, key="var_utilities_base")
    st.selectbox("Utilities period", ["Weekly", "Monthly", "Yearly"], key="var_utilities_period")
    st.selectbox("Season", ["Winter", "Normal", "Summer"], key="var_season")

    st.slider("Commute days/week", 0, 7, key="var_commute_days")
    st.number_input("Cost per commute day (£)", min_value=0.0, step=0.5, key="var_commute_cost")

    st.number_input("Groceries (£)", min_value=0.0, step=5.0, key="var_groceries")
    st.selectbox("Groceries period", ["Weekly", "Monthly", "Yearly"], key="var_groceries_period")

    st.number_input("Household basics (£)", min_value=0.0, step=5.0, key="var_household")
    st.selectbox("Household period", ["Weekly", "Monthly", "Yearly"], key="var_household_period")

    parts = variable_essentials_weekly_total(
        utilities_base=st.session_state["var_utilities_base"],
        utilities_period=st.session_state["var_utilities_period"],
        season=st.session_state["var_season"],
        commute_days=st.session_state["var_commute_days"],
        commute_cost_per_day=st.session_state["var_commute_cost"],
        groceries=st.session_state["var_groceries"],
        groceries_period=st.session_state["var_groceries_period"],
        household=st.session_state["var_household"],
        household_period=st.session_state["var_household_period"],
    )

    st.markdown(f"**Estimated total:** £{parts['total_weekly']:,.2f}/week")

    st.button(
        "Use this total for Variable essentials",
        key="use_variable_total_btn",
        on_click=apply_variable_total,
        args=(parts["total_weekly"],),
    )

# ------------------------------------------------------------
# Discretionary
# ------------------------------------------------------------
e5, e6 = st.columns([2, 1])
with e5:
    st.number_input(
        "Discretionary spending (£)",
        min_value=0.0,
        step=5.0,
        key="discretionary_raw",
        help="Flexible spending you can reduce if needed (social, shopping, entertainment, etc.).",
    )
with e6:
    st.selectbox(
        "Period  ",
        ["Weekly", "Monthly", "Yearly"],
        key="discretionary_period",
        help="How often you spend this discretionary amount.",
    )

p1, p2, p3 = st.columns(3)
for col, name in zip([p1, p2, p3], ["Quiet week", "Typical", "Social-heavy"]):
    with col:
        st.button(
            name,
            key=f"disc_preset_{name}",
            on_click=apply_discretionary_preset,
            args=(name,),
        )

# ------------------------------------------------------------
# Weekly snapshot + comparison (restored)
# ------------------------------------------------------------
income_w = to_weekly(st.session_state["income_raw"], st.session_state["income_period"])
fixed_w  = to_weekly(st.session_state["fixed_essential_raw"], st.session_state["fixed_essential_period"])
var_w    = to_weekly(st.session_state["variable_essential_raw"], st.session_state["variable_essential_period"])
disc_w   = to_weekly(st.session_state["discretionary_raw"], st.session_state["discretionary_period"])

fixed_total_w = fixed_w + var_w
margin_w = income_w - fixed_total_w - disc_w

# Persist weekly-normalised values (used by Step 2)
st.session_state["income_w"] = income_w
st.session_state["fixed_total_w"] = fixed_total_w
st.session_state["discretionary_w"] = disc_w
st.session_state["weekly_margin"] = margin_w


# ============================================================
# 1) GRAPH FIRST
# ============================================================
st.subheader("Weekly composition (stacked)")

income_w = float(income_w)
fixed_w = float(fixed_w)
var_w = float(var_w)
disc_w = float(disc_w)
margin_w = float(margin_w)

if margin_w >= 0:
    segments = [
        ("Fixed", fixed_w),
        ("Variable", var_w),
        ("Discretionary", disc_w),
        ("Margin", margin_w),
    ]
else:
    segments = [
        ("Fixed", fixed_w),
        ("Variable", var_w),
        ("Discretionary", disc_w),
        ("Deficit", abs(margin_w)),
    ]

COLOR_MAP = {
    "Fixed": "#4C72B0",
    "Variable": "#55A868",
    "Discretionary": "#DD8452",
    "Margin": "#64B5CD",
    "Deficit": "#C44E52",
}

stack_total = sum(v for _, v in segments)

fig, ax = plt.subplots(figsize=(9.5, 3.0))

left = 0.0
for label, value in segments:
    ax.barh(
        y=["Weekly"],
        width=[value],
        left=[left],
        color=COLOR_MAP[label],
    )
    left += value

ax.axvline(income_w, linestyle="--", linewidth=1.5)
ax.set_xlabel("£ per week")
ax.set_title("Weekly allocation (stacked)", pad=10)
ax.legend().remove()

max_x = max(income_w, stack_total)
ax.set_xlim(0, max_x * 1.05 if max_x > 0 else 1)

def _pct(v: float) -> float:
    return (v / income_w * 100.0) if income_w > 0 else 0.0

fig.subplots_adjust(bottom=0.48)

x_cols = [0.22, 0.62]
y_rows = [0.14, 0.06]

def _draw_item(i: int, label: str, value: float):
    row = 0 if i < 2 else 1
    col = i % 2
    x = x_cols[col]
    y = y_rows[row]

    fig.text(
        x - 0.045, y,
        "■",
        ha="right",
        va="center",
        fontsize=15,
        color=COLOR_MAP[label],
    )

    fig.text(
        x, y,
        f"{label}: £{value:,.0f} ({_pct(value):.0f}%)",
        ha="left",
        va="center",
        fontsize=12,
        color="black",
    )

for i, (label, value) in enumerate(segments[:4]):
    _draw_item(i, label, value)

st.pyplot(fig, clear_figure=True)

# ============================================================
# 2) TEXT EQUIVALENCES (lightweight)
# ============================================================
st.caption("Equivalences (approx.)")

income_m = weekly_to_period(income_w, "Monthly")
income_y = weekly_to_period(income_w, "Yearly")

ess_w = fixed_total_w
ess_m = weekly_to_period(ess_w, "Monthly")
ess_y = weekly_to_period(ess_w, "Yearly")

disc_m = weekly_to_period(disc_w, "Monthly")
disc_y = weekly_to_period(disc_w, "Yearly")

margin_m = weekly_to_period(margin_w, "Monthly")
margin_y = weekly_to_period(margin_w, "Yearly")

st.write(
    f"Income: **£{income_w:,.0f}/w** ≈ **£{income_m:,.0f}/mo** ≈ **£{income_y:,.0f}/yr**"
)
st.write(
    f"Margin: **£{margin_w:,.0f}/w** ≈ **£{margin_m:,.0f}/mo** ≈ **£{margin_y:,.0f}/yr**"
)

# Optional: one more line (keeps it useful without becoming a table)
st.write(
    f"Essentials: **£{ess_w:,.0f}/w** · Discretionary: **£{disc_w:,.0f}/w**"
)

# ============================================================
# 3) FULL TABLE (optional)
# ============================================================
with st.expander("See full breakdown (Weekly / Monthly / Yearly)"):
    rows = []
    for p in ["Weekly", "Monthly", "Yearly"]:
        rows.append({
            "Period": p,
            "Income": weekly_to_period(income_w, p),
            "Essentials (fixed+variable)": weekly_to_period(ess_w, p),
            "Discretionary": weekly_to_period(disc_w, p),
            "Margin": weekly_to_period(margin_w, p),
        })

    cmp_df = pd.DataFrame(rows)
    st.dataframe(
        cmp_df.style.format({
            "Income": "£{:,.2f}",
            "Essentials (fixed+variable)": "£{:,.2f}",
            "Discretionary": "£{:,.2f}",
            "Margin": "£{:,.2f}",
        }),
        use_container_width=True,
    )


# ============================================================
# Step 2 — Scenarios (A & B savings targets)
# ============================================================
st.divider()
st.subheader("Step 2 — Explore behavioural change")
st.caption(
    "Define two scenarios (A and B) with different weekly savings targets. "
    "If your target exceeds your remaining margin, the difference must come from cutting "
    "**discretionary** spending (the most controllable category). Essentials are not changed automatically."
)

# -----------------------
# ✅ Baseline status state
# -----------------------
if "baseline_ready" not in st.session_state:
    st.session_state["baseline_ready"] = False
if "baseline_signature" not in st.session_state:
    st.session_state["baseline_signature"] = None

def mark_baseline_stale():
    # Don’t delete the baseline df; just mark it as outdated to preserve traceability.
    st.session_state["baseline_ready"] = False


# -----------------------
# ✅ Badges (persistent)
# -----------------------
def saved_badge(text: str):
    st.markdown(
        f"""
        <div style="
            width: 100%;
            box-sizing: border-box;
            padding: 0.65rem 0.9rem;
            border-radius: 0.6rem;
            background: rgba(46, 204, 113, 0.14);
            border: 1px solid rgba(46, 204, 113, 0.35);
            color: rgba(22, 101, 52, 1);
            font-weight: 600;
            margin-top: 0.55rem;
        ">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_badge(text: str):
    st.markdown(
        f"""
        <div style="
            width: 100%;
            box-sizing: border-box;
            padding: 0.65rem 0.9rem;
            border-radius: 0.6rem;
            background: rgba(59, 130, 246, 0.10);
            border: 1px solid rgba(59, 130, 246, 0.25);
            color: rgba(30, 64, 175, 1);
            font-weight: 600;
            margin-top: 0.55rem;
        ">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def warning_badge(text: str):
    st.markdown(
        f"""
        <div style="
            width: 100%;
            box-sizing: border-box;
            padding: 0.65rem 0.9rem;
            border-radius: 0.6rem;
            background: rgba(245, 158, 11, 0.12);
            border: 1px solid rgba(245, 158, 11, 0.28);
            color: rgba(146, 64, 14, 1);
            font-weight: 600;
            margin-top: 0.55rem;
        ">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------
# ✅ Savings target policy resolver (margin -> discretionary)
# -----------------------
def resolve_savings_target(*, target: float, weekly_margin: float, discretionary: float) -> dict:
    """
    Interpret Scenario A/B as a *savings target* per week.

    Coverage order:
      1) remaining margin (no sacrifice)
      2) discretionary spending (requires cut)

    We do NOT touch variable essentials nor fixed essentials (by design, MVP).
    """
    t = max(float(target), 0.0)
    m = float(weekly_margin)
    d = max(float(discretionary), 0.0)

    from_margin = min(t, max(m, 0.0))  # if margin is negative, contributes 0
    remaining = t - from_margin

    from_discretionary = min(remaining, d)
    uncovered = remaining - from_discretionary

    return {
        "target": t,
        "from_margin": from_margin,
        "from_discretionary": from_discretionary,  # real behavioural cut (what Compounder should receive)
        "uncovered": uncovered,
    }


# -----------------------
# ✅ Clear scenarios (avoid stale saved A/B when settings change)
# -----------------------
def clear_ab_silent():
    st.session_state["scenario_a"] = None
    st.session_state["scenario_b"] = None
    st.session_state["saved_a_msg"] = False
    st.session_state["saved_b_msg"] = False
    # no cleared badge here; it's automatic housekeeping


# -----------------------
# Weeks (changing weeks makes baseline stale)
# -----------------------
weeks = st.slider(
    "Planning horizon (weeks)",
    min_value=4,
    max_value=52,
    value=12,
    help="Short-term horizon to keep outputs interpretable.",
    key="weeks",
    on_change=mark_baseline_stale,
)

# -----------------------
# Optional one-off events (up to 3)  ✅ PRIMERO el editor
# -----------------------
with st.expander("Optional — One-off events (unexpected expenses)"):
    st.caption(
        "Model up to three one-off unexpected expenses applied once during the planning horizon "
        "(e.g., repair, fine, unusual night out). This is **not** expense tracking."
    )

    shock_enabled = st.checkbox(
        "Enable one-off events",
        key="shock_enabled",
        on_change=mark_baseline_stale,
    )

    shock_rows = st.data_editor(
        st.session_state.get("shock_events_rows", []),
        key="shock_events_editor",
        hide_index=True,
        num_rows=3,
        disabled=not shock_enabled,
        column_config={
            "name": st.column_config.TextColumn("Event (optional)"),
            "amount": st.column_config.NumberColumn("Amount (£)", min_value=0.0),
            "week": st.column_config.NumberColumn("Week", min_value=1, max_value=int(weeks), step=1),
        },
        use_container_width=True,
    )

    # Persist back to your storage key
    st.session_state["shock_events_rows"] = shock_rows

    if shock_enabled:
        active = [r for r in shock_rows if float(r.get("amount", 0) or 0) > 0]
        if len(active) > 0:
            st.info(f"{len(active)} one-off event(s) active.")

# -----------------------
# Build shock_map (shared by baseline & scenarios)
# -----------------------
shock_map = {}

if st.session_state.get("shock_enabled", False):
    rows = clean_events(
        st.session_state.get("shock_events_rows", []),
        weeks=int(weeks),
    )
    if rows:
        shock_map = events_to_weekly_shock_map(rows)

# -----------------------
# Button: generate baseline (persist + signature)
# -----------------------
if st.button("Generate baseline trajectory", key="generate_baseline_btn"):
    baseline_raw = generate_baseline(
        float(st.session_state["income_w"]),
        float(st.session_state["fixed_total_w"]),
        float(st.session_state["discretionary_w"]),
        int(weeks),
    ).round(2)

    # Schema validation
    try:
        validate_baseline_df(baseline_raw)
    except (ValueError, KeyError) as e:
        st.error(f"Baseline output validation failed: {e}")
        st.stop()

    # Apply one-off events (if any)
    baseline_adj = apply_shock_map_to_df(
        baseline_raw,
        shock_map=shock_map,
        value_cols=("Balance",),
    )

    # Persist
    st.session_state["baseline_df"] = baseline_adj
    st.session_state["baseline_ready"] = True
    st.session_state["baseline_signature"] = make_baseline_signature()

# -----------------------
# Baseline status indicator (no chart here)
# -----------------------
if st.session_state.get("baseline_df") is not None:
    current_sig = make_baseline_signature()
    saved_sig = st.session_state.get("baseline_signature", None)

    if st.session_state.get("baseline_ready", False) and saved_sig == current_sig:
        saved_badge(f"Baseline trajectory generated ({int(weeks)} weeks). Ready for scenario comparison.")
    else:
        warning_badge("Baseline trajectory exists, but inputs have changed. Re-generate baseline to update.")

# ============================================================
# ✅ Presets only (Option A)
# ============================================================
PRESETS = {
    "Quick estimate (default)": {"variability_pct": 30, "iterations": 100},
    "Typical spending": {"variability_pct": 30, "iterations": 200},
    "Unpredictable weeks": {"variability_pct": 50, "iterations": 200},
    "Stress test": {"variability_pct": 50, "iterations": 500},
}

if "preset_name" not in st.session_state:
    st.session_state["preset_name"] = "Quick estimate (default)"


def apply_preset():
    st.session_state["preset_name"] = st.session_state["preset_select"]
    clear_ab_silent()


top_left, top_right = st.columns([2, 1])

with top_left:
    st.selectbox(
        "Preset",
        list(PRESETS.keys()),
        index=list(PRESETS.keys()).index(st.session_state["preset_name"]),
        help="One-click uncertainty setup (beginner-friendly).",
        key="preset_select",
        on_change=apply_preset,
    )

with top_right:
    st.markdown("**Randomness**")
    st.caption("For fair A vs B comparison, randomness is kept consistent unless you redraw.")
    if st.button("Try another random run", use_container_width=True, key="reroll_seed"):
        st.session_state["seed"] += 1
        clear_ab_silent()
        st.toast("New random draw applied.", icon="🎲")

# Effective sim settings from preset
preset_cfg = PRESETS[st.session_state["preset_name"]]
iterations = int(preset_cfg["iterations"])
variability_pct = int(preset_cfg["variability_pct"])
variability_frac = float(variability_pct) / 100.0
seed = int(st.session_state["seed"])

st.caption(
    f"Preset applied: **{st.session_state['preset_name']}** → "
    f"**{iterations} simulations**, **{variability_pct}% unpredictability**."
)

# -----------------------
# Scenario savings targets (NEW semantics)
# -----------------------
disc = float(st.session_state.get("discretionary_w", 0.0))
margin = float(st.session_state.get("weekly_margin", 0.0))

colA, colB = st.columns(2)
with colA:
    target_a = st.number_input(
        "Scenario A — Savings target (£/week)",
        min_value=0.0,
        value=10.0,
        step=5.0,
        key="target_a",
        on_change=clear_ab_silent,
        help=(
            "This is the savings amount you want to achieve each week. "
            "The app tries to cover it first from your remaining margin, then (if needed) "
            "by cutting discretionary spending."
        ),
    )
with colB:
    target_b = st.number_input(
        "Scenario B — Savings target (£/week)",
        min_value=0.0,
        value=20.0,
        step=5.0,
        key="target_b",
        on_change=clear_ab_silent,
        help="A second weekly savings target to compare against Scenario A.",
    )

res_a = resolve_savings_target(target=float(target_a), weekly_margin=float(margin), discretionary=float(disc))
res_b = resolve_savings_target(target=float(target_b), weekly_margin=float(margin), discretionary=float(disc))

st.markdown("**How your target is covered (policy):** remaining margin → discretionary cut (essentials untouched).")
cA1, cB1 = st.columns(2)
with cA1:
    st.caption("Scenario A breakdown")
    st.write(
        f"- Target: **£{res_a['target']:,.0f}/w**\n"
        f"- Covered by margin: **£{res_a['from_margin']:,.0f}/w**\n"
        f"- Requires discretionary cut: **£{res_a['from_discretionary']:,.0f}/w**"
    )
with cB1:
    st.caption("Scenario B breakdown")
    st.write(
        f"- Target: **£{res_b['target']:,.0f}/w**\n"
        f"- Covered by margin: **£{res_b['from_margin']:,.0f}/w**\n"
        f"- Requires discretionary cut: **£{res_b['from_discretionary']:,.0f}/w**"
    )


def savings_status(res: dict, name: str) -> bool:
    if res["uncovered"] > 0:
        st.error(
            f"{name}: target cannot be fully covered without cutting essentials "
            f"(shortfall ≈ £{res['uncovered']:,.0f}/w). "
            "In this MVP, essentials are not adjusted automatically — reduce the target."
        )
        return False
    if res["from_discretionary"] > 0:
        st.warning(
            f"{name}: reaching this target requires cutting discretionary spending by "
            f"≈ £{res['from_discretionary']:,.0f}/w."
        )
    else:
        st.info(f"{name}: fully covered by remaining margin (no spending cuts needed).")
    return True


ok_a = savings_status(res_a, "Scenario A")
ok_b = savings_status(res_b, "Scenario B")

# -----------------------
# Save / Clear + persistent messages
# -----------------------
st.caption("Save Scenario A and Scenario B to compare outcomes side-by-side.")


def clear_ab():
    st.session_state["scenario_a"] = None
    st.session_state["scenario_b"] = None
    st.session_state["saved_a_msg"] = False
    st.session_state["saved_b_msg"] = False
    st.session_state["cleared_msg"] = True


b1, b2, b3 = st.columns(3)

with b1:
    clicked_a = st.button("Save Scenario A", use_container_width=True, key="save_a")
    if clicked_a:
        # ✅ Require baseline exists AND is up-to-date
        if st.session_state.get("baseline_df") is None:
            st.error("Generate the baseline trajectory first (Step 2).")
        elif not st.session_state.get("baseline_ready", False) or st.session_state.get("baseline_signature") != make_baseline_signature():
            st.error("Baseline exists but is out of date. Re-generate baseline before saving scenarios.")
        elif not ok_a:
            st.error("Scenario A cannot be saved because the savings target is not feasible under current MVP rules.")
        else:
            # IMPORTANT: pass only the real discretionary cut to Compounder
            df_a, params_a = simulate_scenario_df_with_seed_offset(
                income=float(st.session_state["income_w"]),
                fixed_expenses=float(st.session_state["fixed_total_w"]),
                variable_expenses=float(st.session_state["discretionary_w"]),
                delta_savings=float(res_a["from_discretionary"]),
                weeks=int(weeks),
                iterations=int(iterations),
                seed=int(seed),
                seed_offset=0,
                variability_frac=float(variability_frac),

            )

            # Add UI traceability fields
            params_a["preset"] = str(st.session_state["preset_name"])
            params_a["variability_pct"] = int(variability_pct)

            # Add NEW semantic traceability fields
            params_a["target_savings"] = float(res_a["target"])
            params_a["from_margin"] = float(res_a["from_margin"])
            params_a["from_discretionary"] = float(res_a["from_discretionary"])
            params_a["uncovered"] = float(res_a["uncovered"])

            st.session_state["scenario_a"] = {"df": df_a.round(2), "params": params_a}
            st.session_state["saved_a_msg"] = True

    if st.session_state.get("saved_a_msg", False):
        saved_badge("Scenario A saved.")

with b2:
    clicked_b = st.button("Save Scenario B", use_container_width=True, key="save_b")
    if clicked_b:
        # ✅ Require baseline exists AND is up-to-date
        if st.session_state.get("baseline_df") is None:
            st.error("Generate the baseline trajectory first (Step 2).")
        elif not st.session_state.get("baseline_ready", False) or st.session_state.get("baseline_signature") != make_baseline_signature():
            st.error("Baseline exists but is out of date. Re-generate baseline before saving scenarios.")
        elif not ok_b:
            st.error("Scenario B cannot be saved because the savings target is not feasible under current MVP rules.")
        else:
            df_b, params_b = simulate_scenario_df_with_seed_offset(
                income=float(st.session_state["income_w"]),
                fixed_expenses=float(st.session_state["fixed_total_w"]),
                variable_expenses=float(st.session_state["discretionary_w"]),
                delta_savings=float(res_b["from_discretionary"]),
                weeks=int(weeks),
                iterations=int(iterations),
                seed=int(seed),
                seed_offset=1,
                variability_frac=float(variability_frac),
            )

            # Add UI traceability fields
            params_b["preset"] = str(st.session_state["preset_name"])
            params_b["variability_pct"] = int(variability_pct)

            # Add NEW semantic traceability fields
            params_b["target_savings"] = float(res_b["target"])
            params_b["from_margin"] = float(res_b["from_margin"])
            params_b["from_discretionary"] = float(res_b["from_discretionary"])
            params_b["uncovered"] = float(res_b["uncovered"])

            st.session_state["scenario_b"] = {"df": df_b.round(2), "params": params_b}
            st.session_state["saved_b_msg"] = True

    if st.session_state.get("saved_b_msg", False):
        saved_badge("Scenario B saved.")

with b3:
    st.button("Clear A/B", use_container_width=True, key="clear_ab", on_click=clear_ab)

with b3:
    if st.session_state.get("cleared_msg", False):
        info_badge("Scenario A and B cleared.")
        st.session_state["cleared_msg"] = False

# ============================================================
# Step 3 — Compare
# ============================================================
st.divider()
st.subheader("Step 3 — Compare outcomes")
st.caption("Compare your baseline trajectory with alternative scenarios under uncertainty.")

baseline_df = st.session_state.get("baseline_df")
scenario_a = st.session_state.get("scenario_a")
scenario_b = st.session_state.get("scenario_b")

# --- Small helper: validate schema to avoid weird runtime errors ---
def assert_cols(df: pd.DataFrame, required: set, name: str):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.error(f"{name} is missing or empty.")
        st.stop()
    missing = required - set(df.columns)
    if missing:
        st.error(f"{name} missing columns: {sorted(missing)}. Found: {df.columns.tolist()}")
        st.stop()

# --- Guardrails: order of operations ---
ready_for_compare = True

if baseline_df is None:
    st.info("Generate the baseline trajectory in Step 2 to enable comparison.")
    ready_for_compare = False
elif not st.session_state.get("baseline_ready", False) or st.session_state.get("baseline_signature") != make_baseline_signature():
    st.warning("Baseline exists but inputs have changed. Re-generate baseline in Step 2 to update comparison.")
    ready_for_compare = False
elif scenario_a is None or scenario_b is None:
    st.info("Save both Scenario A and Scenario B to compare.")
    ready_for_compare = False

df_a = None
df_b = None

if ready_for_compare:
    # --- Pull scenario dataframes ---
    df_a = scenario_a.get("df")
    df_b = scenario_b.get("df")

    # --- Validate expected schemas ---
    assert_cols(baseline_df, {"Week", "Balance"}, "Baseline dataframe")
    assert_cols(df_a, {"Week", "Mean", "Lower", "Upper"}, "Scenario A dataframe")
    assert_cols(df_b, {"Week", "Mean", "Lower", "Upper"}, "Scenario B dataframe")

    # --- Plot ---
    fig, ax = plt.subplots()

    ax.plot(baseline_df["Week"], baseline_df["Balance"], label="Baseline (no change)", linestyle="--")

    ax.plot(df_a["Week"], df_a["Mean"], label="Scenario A (mean)")
    ax.fill_between(df_a["Week"], df_a["Lower"], df_a["Upper"], alpha=0.2, label="A band (10–90%)")

    ax.plot(df_b["Week"], df_b["Mean"], label="Scenario B (mean)")
    ax.fill_between(df_b["Week"], df_b["Lower"], df_b["Upper"], alpha=0.2, label="B band (10–90%)")

    ax.set_xlabel("Week")
    ax.set_ylabel("Balance (£)")
    ax.set_title("Trajectory comparison")
    ax.legend()
    st.pyplot(fig)

    # --- Summary stats ---
    def final_stats(df):
        final_mean = float(df["Mean"].iloc[-1])
        final_low = float(df["Lower"].iloc[-1])
        final_up = float(df["Upper"].iloc[-1])
        return final_mean, final_low, final_up

    base_final = float(baseline_df["Balance"].iloc[-1])
    a_mean, a_low, a_up = final_stats(df_a)
    b_mean, b_low, b_up = final_stats(df_b)

    summary = pd.DataFrame(
        [
            {"Scenario": "Baseline", "Final balance (mean)": round(base_final, 2), "Final range (10–90%)": "—"},
            {"Scenario": "Scenario A", "Final balance (mean)": round(a_mean, 2), "Final range (10–90%)": f"{a_low:.2f} – {a_up:.2f}"},
            {"Scenario": "Scenario B", "Final balance (mean)": round(b_mean, 2), "Final range (10–90%)": f"{b_low:.2f} – {b_up:.2f}"},
        ]
    )

    st.subheader("Summary")
    st.dataframe(summary, use_container_width=True)

# ============================================================
# Step 4 — Reflect on impact (HUMAN-FIRST)
# ============================================================
st.divider()
st.subheader("Step 4 — Reflect on impact")
st.caption("Plain-English summary: what your weekly targets could mean for your money.")

if not ready_for_compare:
    st.info("Complete Step 3 (baseline + saved Scenario A and B) to see the reflection summary.")
else:
    # Pull final numbers (UI/orchestration only)
    params_a = scenario_a.get("params", {})
    params_b = scenario_b.get("params", {})

    base_final = float(baseline_df["Balance"].iloc[-1])

    a_mean = float(df_a["Mean"].iloc[-1])
    a_low = float(df_a["Lower"].iloc[-1])
    a_up = float(df_a["Upper"].iloc[-1])

    b_mean = float(df_b["Mean"].iloc[-1])
    b_low = float(df_b["Lower"].iloc[-1])
    b_up = float(df_b["Upper"].iloc[-1])

    weeks_n = int(st.session_state["weeks"])

    # New semantic values (stored in params)
    target_a_w = float(params_a.get("target_savings", 0.0))
    target_b_w = float(params_b.get("target_savings", 0.0))

    margin_a_w = float(params_a.get("from_margin", 0.0))
    margin_b_w = float(params_b.get("from_margin", 0.0))

    cut_a_w = float(params_a.get("from_discretionary", params_a.get("delta_savings", 0.0)))
    cut_b_w = float(params_b.get("from_discretionary", params_b.get("delta_savings", 0.0)))

    # Centralised Step 4 computation + copy
    # NOTE: To avoid hard-breaks if your explain.py is not updated yet, we try the new signature first,
    # then fall back to the old signature.
    try:
        metrics = compute_reflection_metrics(
            weeks=weeks_n,
            base_final=base_final,
            a_mean=a_mean, a_low=a_low, a_high=a_up,
            b_mean=b_mean, b_low=b_low, b_high=b_up,
            target_a_weekly=target_a_w,
            target_b_weekly=target_b_w,
            cut_a_weekly=cut_a_w,
            cut_b_weekly=cut_b_w,
            margin_a_weekly=margin_a_w,
            margin_b_weekly=margin_b_w,
        )
    except TypeError:
        # Old signature fallback (uses the REAL cut, not the target)
        metrics = compute_reflection_metrics(
            weeks=weeks_n,
            base_final=base_final,
            a_mean=a_mean, a_low=a_low, a_high=a_up,
            b_mean=b_mean, b_low=b_low, b_high=b_up,
            delta_a_weekly=cut_a_w,
            delta_b_weekly=cut_b_w,
        )

    events_active = (
        bool(st.session_state.get("shock_enabled", False))
        and any(
            float(r.get("amount", 0.0)) > 0
            for r in st.session_state.get("shock_events_rows", [])
        )
    )

    # Build reflection text (robust to old signature)
    try:
        text = build_human_reflection_text(metrics, shocks_active=events_active)
    except TypeError:
        text = build_human_reflection_text(metrics)

          # --- 1) Human-first summary ---
    st.markdown("### What you get if you follow each plan")
    st.markdown(text.get("what_you_get", ""))

    st.markdown("### Which one should you pick?")
    if getattr(metrics, "winner", "Tie") == "Tie":
        st.info(text.get("pick_primary", ""))
    else:
        st.success(text.get("pick_primary", ""))
        if text.get("pick_secondary"):
            st.warning(text.get("pick_secondary", ""))

    # --- 2) Shock note ---
    if text.get("shock_note"):
        st.info(text.get("shock_note", ""))

    # --- 3) Variability note ---
    st.markdown(text.get("variability", ""))

    # --- 4) Technical details hidden (for you / marking) ---
    exp_inputs = ExplanationInputs(
        income=float(st.session_state["income_w"]),
        fixed_expenses=float(st.session_state["fixed_total_w"]),
        variable_expenses=float(st.session_state["discretionary_w"]),
        weeks=int(st.session_state["weeks"]),
        # Explanation talks in targets (human intent)
        delta_a=float(target_a_w),
        delta_b=float(target_b_w),

        # Use app-level truth (don’t rely on params_a having these keys)
        variability_pct=float(variability_pct),
        seed=int(st.session_state.get("seed", 0)),
        iters=int(iterations),
    )

    explanation_text = build_explanation(
        base_final=base_final,
        a_final_mean=a_mean,
        a_final_low=a_low,
        a_final_high=a_up,
        b_final_mean=b_mean,
        b_final_low=b_low,
        b_final_high=b_up,
        inputs=exp_inputs,
    )


    preset_name = str(params_a.get("preset", st.session_state.get("preset_name", "Preset")))
    iters = int(params_a.get("iterations", 0))
    var_pct = int(params_a.get("variability_pct", 0))

    fairness_note = (
        "A and B use the same uncertainty preset; Scenario B uses a controlled seed offset "
        "to keep randomness comparable."
    )

    with st.expander("Technical details (optional)"):
        st.markdown("### Model explanation (technical)")
        st.markdown(explanation_text)

        # Add explicit semantic traceability in the technical panel
        st.markdown(
            f"""
**Scenario semantics (traceability):**
- Scenario A target: **£{target_a_w:,.0f}/w** = margin **£{margin_a_w:,.0f}/w** + discretionary cut **£{cut_a_w:,.0f}/w**
- Scenario B target: **£{target_b_w:,.0f}/w** = margin **£{margin_b_w:,.0f}/w** + discretionary cut **£{cut_b_w:,.0f}/w**
""".strip()
        )

        st.caption(f"Preset: **{preset_name}** ({iters} sims, {var_pct}% variability).")
        st.caption(fairness_note)
