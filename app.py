# app.py (FULL)
# ------------------------------------------------------------
# LifeBudget Micro — main UI / orchestration layer
#
# Key concepts:
# - Step 1: describe current position (no behaviour change)
# - Step 2: scenarios are TARGET SAVINGS (use Margin first, then cut discretionary if needed)
# - Baseline already keeps any positive Margin as savings by default
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
    build_structural_deficit_tips,
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
    required = ["income_w", "fixed_total_w", "discretionary_w"]
    missing = [k for k in required if k not in st.session_state]

    if missing:
        raise RuntimeError(
            f"make_baseline_signature() called before weekly values exist: {missing}"
        )

    return {
        "income_w": round(float(st.session_state["income_w"]), 6),
        "fixed_total_w": round(float(st.session_state["fixed_total_w"]), 6),
        "discretionary_w": round(float(st.session_state["discretionary_w"]), 6),
        "weeks": int(st.session_state.get("weeks", 12)),
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
ss("weeks", 12)
ss("baseline_ready", False)
ss("baseline_signature", None)

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
ss("discretionary_period_prev", st.session_state.get("discretionary_period", "Weekly"))
ss("position_confirmed", False)

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title("LifeBudget Micro")
st.subheader("Short-term financial clarity")
st.caption("Describe your current position first. Explore change later.")

# ------------------------------------------------------------
# Step 1 helper callbacks (MUST be defined before buttons use them)
# ------------------------------------------------------------
def invalidate_position():
    """Any change to Step 1 inputs invalidates the confirmed snapshot."""
    st.session_state["position_confirmed"] = False

def apply_fixed_total(total_fixed_weekly: float):
    st.session_state["fixed_essential_raw"] = float(total_fixed_weekly)
    st.session_state["fixed_essential_period"] = "Weekly"
    invalidate_position()
    st.toast("Fixed essentials updated", icon="✅")

def apply_variable_total(total_var_weekly: float):
    st.session_state["variable_essential_raw"] = float(total_var_weekly)
    st.session_state["variable_essential_period"] = "Weekly"
    invalidate_position()
    st.toast("Variable essentials updated", icon="✅")

def apply_estimated_income(net_weekly: float):
    st.session_state["income_raw"] = float(net_weekly)
    st.session_state["income_period"] = "Weekly"
    st.session_state["income_est_applied"] = True
    invalidate_position()
    st.toast("Income updated from estimator", icon="✅")

def apply_discretionary_preset(preset_name: str):
    st.session_state["discretionary_raw"] = float(discretionary_preset_value(preset_name))
    st.session_state["discretionary_period"] = "Weekly"
    st.session_state["discretionary_period_prev"] = "Weekly"
    invalidate_position()
    st.toast(f"Preset applied: {preset_name}", icon="✅")

def on_discretionary_period_change():
    prev = st.session_state.get("discretionary_period_prev", "Weekly")
    curr = st.session_state.get("discretionary_period", "Weekly")

    raw = float(st.session_state.get("discretionary_raw", 0.0))
    weekly = to_weekly(raw, prev)                 # prev -> weekly
    new_raw = weekly_to_period(weekly, curr)      # weekly -> curr

    st.session_state["discretionary_raw"] = round(float(new_raw), 2)
    st.session_state["discretionary_period_prev"] = curr
    invalidate_position()

def confirm_position():
    st.session_state["position_confirmed"] = True

    st.session_state["baseline_ready"] = False
    st.session_state["baseline_df"] = None
    st.session_state["baseline_signature"] = None   # ✅ add

    st.session_state["scenario_a"] = None
    st.session_state["scenario_b"] = None

    st.session_state["saved_a_msg"] = False
    st.session_state["saved_b_msg"] = False
    st.session_state["cleared_msg"] = False         # ✅ optional

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
        on_change=invalidate_position,
    )
with c2:
    st.selectbox(
        "Income period",
        ["Weekly", "Monthly", "Yearly"],
        key="income_period",
        help="How often you receive this income.",
        on_change=invalidate_position,
    )

# ------------------------------------------------------------
# Optional: take-home estimator (rough)
# ------------------------------------------------------------
with st.expander("Estimate take-home income (optional)"):
    st.caption(
        "Quick estimator to produce a usable *net* figure. "
        "This is a **rough approximation** based on a single percentage and does not account for "
        "tax bands, National Insurance, pension contributions, student loans, or other deductions. "
        "For an exact take-home amount, refer to your payslip or official HMRC guidance."
    )

    st.number_input(
        "Gross income (£/year)",
        min_value=0.0,
        step=500.0,
        key="gross_annual_est",
        help="Gross annual income before tax/deductions.",
    )

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
        on_change=invalidate_position,
    )
with e2:
    st.selectbox(
        "Period",
        ["Weekly", "Monthly", "Yearly"],
        key="fixed_essential_period",
        help="How often you pay these fixed essentials.",
        on_change=invalidate_position,
    )

with st.expander("Build my fixed expenses (optional)"):
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

    st.session_state["fixed_items_rows"] = fixed_rows
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
        on_change=invalidate_position,
    )
with e4:
    st.selectbox(
        "Period ",
        ["Weekly", "Monthly", "Yearly"],
        key="variable_essential_period",
        help="How often you spend this amount.",
        on_change=invalidate_position,
    )

with st.expander("Rough breakdown for variable essentials (optional)"):
    st.caption(
        "Utilities are treated as baseline essentials. This usually includes household electricity and gas, "
        "which are commonly paid as a fixed monthly amount under a tariff plan. "
        "Although these costs may change over time, they are typically not adjustable week-by-week, "
        "so the model keeps them fixed and focuses behavioural change on discretionary spending."
    )

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

    # ✅ Save raw parts (optional, useful for debugging / future UI)
    st.session_state["variable_parts"] = parts

    # ✅ NEW: label map for cleaner tip labels
    LABEL_MAP = {
        "Utilities Base": "Utilities",
        "Commute": "Commuting",
        "Commute Days": "Commuting",
        "Commute Cost Per Day": "Commuting",
        "Groceries": "Groceries",
        "Household": "Household basics",
    }

    # ✅ NEW: normalised rows for smart tips
    variable_items_weekly = []
    for k, v in parts.items():
        if k.endswith("_weekly") and k != "total_weekly":
            raw_label = k.replace("_weekly", "").replace("_", " ").title()
            label = LABEL_MAP.get(raw_label, raw_label)  # ✅ apply nicer label
            if float(v) > 0:
                variable_items_weekly.append({"name": label, "weekly": float(v)})

    # Keep stable ordering (highest first)
    variable_items_weekly.sort(key=lambda r: r["weekly"], reverse=True)

    # Optional: if multiple keys map to the same label, merge them (prevents duplicates)
    merged = {}
    for row in variable_items_weekly:
        merged[row["name"]] = merged.get(row["name"], 0.0) + float(row["weekly"])

    variable_items_weekly = [{"name": k, "weekly": v} for k, v in merged.items()]
    variable_items_weekly.sort(key=lambda r: r["weekly"], reverse=True)

    st.session_state["variable_items_rows_weekly"] = variable_items_weekly
    st.session_state["variable_top_driver"] = variable_items_weekly[0] if variable_items_weekly else None

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
        on_change=invalidate_position,
    )
with e6:
    st.selectbox(
        "Period",
        ["Weekly", "Monthly", "Yearly"],
        key="discretionary_period",
        help="How often you spend this discretionary amount.",
        on_change=on_discretionary_period_change,
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
# Confirm gate
# ------------------------------------------------------------
st.button(
    "Confirm my current position",
    use_container_width=True,
    key="confirm_position_btn",
    on_click=confirm_position,
)

if not st.session_state.get("position_confirmed", False):
    st.info("Fill in your income and expenses, then confirm to see your baseline breakdown.")
    st.stop()

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
st.session_state["income_w"] = float(income_w)
st.session_state["fixed_total_w"] = float(fixed_total_w)
st.session_state["discretionary_w"] = float(disc_w)   # (esto lo sigues usando para el simulador)
st.session_state["weekly_margin"] = float(margin_w)

st.session_state["fixed_w"] = float(fixed_w)
st.session_state["var_w"] = float(var_w)
st.session_state["disc_w"] = float(disc_w)

# ------------------------------------------------------------
# NEW: persist fixed items with weekly-normalised values
# (used for intelligent structural deficit tips)
# ------------------------------------------------------------
def _rows_with_weekly(rows):
    out = []
    for r in (rows or []):
        try:
            name = str(r.get("name", "")).strip()
            amount = float(r.get("amount", 0.0) or 0.0)
            period = str(r.get("period", "Weekly"))
            weekly = float(to_weekly(amount, period))
            if weekly > 0:
                out.append({
                    "name": name,
                    "weekly": weekly,
                })
        except Exception:
            continue
    return out

st.session_state["fixed_items_rows_weekly"] = _rows_with_weekly(
    st.session_state.get("fixed_items_rows", [])
)

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

segments = [(label, value) for label, value in segments if value > 0]

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
    f"Leftover (Margin): **£{margin_w:,.0f}/w** ≈ **£{margin_m:,.0f}/mo** ≈ **£{margin_y:,.0f}/yr**"
)
st.write(
    f"Essentials: **£{ess_w:,.0f}/w** · Discretionary: **£{disc_w:,.0f}/w**"
)

# IMPORTANT: baseline assumption (clarity) — UPDATED for Option B
if margin_w >= 0:
    st.info(
        "In this version, any **positive Margin** (leftover money) is assumed to **stay in your balance** "
        "(i.e., it becomes savings by default). "
        "Step 2 scenarios set **target savings** amounts. The model uses your Margin first; "
        "only the remainder requires cutting **discretionary** spending (essentials are not changed automatically)."
    )
else:
    deficit_w = abs(float(margin_w))
    st.warning(
        f"Right now you have a **weekly deficit** of about **£{deficit_w:,.0f}/w** "
        "(your essentials + discretionary are higher than your income). "
        "Step 2 scenarios can still be used, but the **first priority is to reach break-even** "
        "(Margin ≥ £0/w). In this MVP, only **discretionary** is adjusted automatically — "
        "if the deficit exceeds discretionary, structural changes are required."
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
    cmp_df_display = cmp_df.reset_index(drop=True)
    st.dataframe(
        cmp_df_display.style.format({
            "Income": "£{:,.2f}",
            "Essentials (fixed+variable)": "£{:,.2f}",
            "Discretionary": "£{:,.2f}",
            "Margin": "£{:,.2f}",
        }),
        use_container_width=True,
    )

# ============================================================
# # Step 2 — Scenarios (A & B target savings)
# ============================================================
st.divider()
st.subheader("Step 2 — Explore behavioural change")

baseline_margin = max(float(margin_w), 0.0)

if margin_w >= 0:
    st.caption(
        "Baseline already keeps any positive **Margin** as savings by default. "
        "Below, set two scenarios (A and B) as **target weekly savings** amounts. "
        "The model uses your Margin first; only the remainder requires cutting "
        "**discretionary** spending. Essentials are not changed automatically."
    )
else:
    st.caption(
        "Below, set two scenarios (A and B) as **target weekly savings** amounts. "
        "These targets describe what you aim to save **after reaching break-even**. "
        "In the current situation, the priority is reducing spending and/or increasing income."
    )

# -----------------------
# ✅ Baseline status state
# -----------------------
if "baseline_ready" not in st.session_state:
    st.session_state["baseline_ready"] = False
if "baseline_signature" not in st.session_state:
    st.session_state["baseline_signature"] = None

def mark_baseline_stale():
    st.session_state["baseline_ready"] = False
    st.session_state["baseline_df"] = None   # opcional (strict)

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

def clear_ab_silent():
    st.session_state["scenario_a"] = None
    st.session_state["scenario_b"] = None
    st.session_state["saved_a_msg"] = False
    st.session_state["saved_b_msg"] = False

# -----------------------
# Weeks
# -----------------------
weeks = st.slider(
    "Planning horizon (weeks)",
    min_value=4,
    max_value=52,
    value=st.session_state.get("weeks", 12),
    key="weeks",
    help="Short-term horizon to keep outputs interpretable.",
    on_change=mark_baseline_stale,
)

# -----------------------
# One-off events
# -----------------------
with st.expander("Optional — One-off events (unexpected expenses)"):
    st.checkbox(
        "Enable one-off events",
        key="shock_enabled",
        on_change=mark_baseline_stale,
    )

    shock_rows = st.data_editor(
        st.session_state.get("shock_events_rows", []),
        key="shock_events_editor",
        hide_index=True,
        num_rows=3,
        disabled=not st.session_state.get("shock_enabled", False),
        column_config={
            "name": st.column_config.TextColumn("Event (optional)"),
            "amount": st.column_config.NumberColumn("Amount (£)", min_value=0.0),
            "week": st.column_config.NumberColumn("Week", min_value=1, max_value=int(weeks)),
        },
        use_container_width=True,
    )

    st.session_state["shock_events_rows"] = shock_rows


shock_map = {}
if st.session_state.get("shock_enabled", False):
    rows = clean_events(st.session_state.get("shock_events_rows", []), weeks=int(weeks))
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

    try:
        validate_baseline_df(baseline_raw)
    except (ValueError, KeyError) as e:
        st.error(f"Baseline output validation failed: {e}")
        st.stop()

    baseline_adj = apply_shock_map_to_df(
        baseline_raw,
        shock_map=shock_map,
        value_cols=("Balance",),
    )

    st.session_state["baseline_df"] = baseline_adj
    st.session_state["baseline_ready"] = True
    st.session_state["baseline_signature"] = make_baseline_signature()

# -----------------------
# Baseline status indicator
# -----------------------
if st.session_state.get("baseline_df") is not None:
    current_sig = make_baseline_signature()
    saved_sig = st.session_state.get("baseline_signature", None)

    if st.session_state.get("baseline_ready", False) and saved_sig == current_sig:
        saved_badge(f"Baseline trajectory generated ({int(weeks)} weeks). Ready for scenario comparison.")
    else:
        warning_badge("Baseline trajectory exists, but inputs have changed. Re-generate baseline to update.")

# ============================================================
# ✅ Presets (uncertainty)
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
# Scenario TARGET savings (Option B)
# -----------------------
disc = float(st.session_state.get("discretionary_w", 0.0))
margin = float(st.session_state.get("weekly_margin", 0.0))
baseline_margin = max(float(margin), 0.0)

# -----------------------------------------
# Deficit mode: recoverable vs structural
# -----------------------------------------
max_possible_margin = float(margin) + float(disc)
structural_deficit = max_possible_margin < 0
can_save_scenarios = not structural_deficit

# ✅ Single breakdown snapshot (one source of truth)
breakdown = {
    "income_w": float(st.session_state.get("income_w", 0.0)),
    "fixed_w": float(st.session_state.get("fixed_w", 0.0)),
    "var_w": float(st.session_state.get("var_w", 0.0)),
    "disc_w": float(st.session_state.get("disc_w", 0.0)),
    "fixed_total_w": float(st.session_state.get("fixed_total_w", 0.0)),
    "discretionary_w": float(st.session_state.get("discretionary_w", 0.0)),  # alias seguro
    "margin_w": float(st.session_state.get("weekly_margin", 0.0)),

    # ✅ Itemised rows (para tips menos genéricos) — van DENTRO del breakdown
    # (usa las keys que ya estás guardando en Step 1)
    "fixed_items_rows": st.session_state.get("fixed_items_rows_weekly", []),
    "variable_items_rows_weekly": st.session_state.get("variable_items_rows_weekly", []),

    # ✅ opcional (si lo guardas)
    "variable_parts": st.session_state.get("variable_parts", {}),
    "fixed_items_raw_rows": st.session_state.get("fixed_items_rows", []),  # por si quieres debug/UI luego
}

if structural_deficit:
    st.error(
        f"Your current situation shows a **structural deficit** of about "
        f"**£{abs(float(breakdown['margin_w'])):,.0f}/w**. "
        "Even eliminating all discretionary spending would not reach break-even.\n\n"
        "Scenarios below represent **longer-term goals**, not actions you can take immediately. "
        "Short-term solutions likely require **income changes or major essential cost reductions**."
    )

    # ✅ Smart tips right where the user hits the wall
    with st.expander("Tips to fix this structural deficit (based on your data)", expanded=True):
        tips_md = build_structural_deficit_tips(
            breakdown=breakdown,
            deficit_w=abs(float(breakdown["margin_w"])),
        )
        st.markdown(tips_md)

elif margin < 0:
    st.info(
        f"Current situation: **deficit £{abs(float(margin)):,.0f}/w**. "
        "Targets in Step 2 should be interpreted as **minimum savings commitments** once you reach break-even. "
        "For now, the key is reducing spending and/or increasing income to close the deficit."
    )
else:
    st.info(
        f"Baseline savings from your current Margin: **£{baseline_margin:,.0f}/w**. "
        "Set a **target savings** amount for each scenario. "
        "The model uses your Margin first; only the remainder requires cutting discretionary spending."
    )

# Max target you can model via discretionary cuts
if margin >= 0:
    max_target = max(baseline_margin + disc, 0.0)
else:
    max_target = max(disc, 0.0)

colA, colB = st.columns(2)
with colA:
    target_a = st.number_input(
        "Scenario A — Target savings (£/week)",
        min_value=0.0,
        max_value=max_target,
        value=min(120.0, max_target),
        step=5.0,
        key="target_a",
        on_change=clear_ab_silent,
        disabled=structural_deficit,
    )
with colB:
    target_b = st.number_input(
        "Scenario B — Target savings (£/week)",
        min_value=0.0,
        max_value=max_target,
        value=min(160.0, max_target),
        step=5.0,
        key="target_b",
        on_change=clear_ab_silent,
        disabled=structural_deficit,
    )

st.markdown("**Policy (Option B):** targets use your Margin first; only the remainder requires cutting discretionary.")

def target_status(target: float, name: str) -> tuple[bool, float, float, float]:
    """
    Returns:
      ok, cut_needed, from_margin, from_discretionary
    """
    t = float(target)

    # In deficit, targets are interpreted as post-break-even commitments.
    in_deficit_now = float(margin) < 0

    # How much must be cut from discretionary to reach the target?
    cut_needed = max(0.0, t - baseline_margin)

    # Split for explanation
    from_margin = min(t, baseline_margin)
    from_discretionary = cut_needed

    # Feasibility cap:
    # - If margin>=0: supported by margin + discretionary
    # - If margin<0: supported by discretionary only
    cap = (baseline_margin + disc) if float(margin) >= 0 else disc

    if t > cap + 1e-9:
        if float(margin) >= 0:
            st.error(
                f"{name}: target £{t:,.0f}/w exceeds what you can support "
                f"(Margin £{baseline_margin:,.0f}/w + Discretionary £{disc:,.0f}/w)."
            )
        else:
            st.error(
                f"{name}: target £{t:,.0f}/w exceeds your current discretionary (£{disc:,.0f}/w). "
                "While you are in deficit, targets are interpreted as goals **after break-even**."
            )
        return False, cut_needed, from_margin, from_discretionary

    # Status messaging
    if cut_needed <= 1e-9:
        if in_deficit_now:
            st.info(
                f"{name}: this target would be fully covered by margin **after break-even** "
                "(no discretionary cut needed)."
            )
        else:
            st.success(f"{name}: fully covered by your current Margin (no discretionary cut needed).")
    else:
        if in_deficit_now:
            st.warning(
                f"{name}: once you reach break-even, this target would require cutting discretionary "
                f"by ≈ £{cut_needed:,.0f}/w."
            )
        else:
            st.warning(
                f"{name}: £{from_margin:,.0f}/w comes from Margin, "
                f"and requires cutting discretionary by ≈ £{cut_needed:,.0f}/w."
            )

    return True, cut_needed, from_margin, from_discretionary

if structural_deficit:
    ok_a, cut_a_w, from_margin_a, from_disc_a = False, 0.0, 0.0, 0.0
    ok_b, cut_b_w, from_margin_b, from_disc_b = False, 0.0, 0.0, 0.0
else:
    ok_a, cut_a_w, from_margin_a, from_disc_a = target_status(target_a, "Scenario A")
    ok_b, cut_b_w, from_margin_b, from_disc_b = target_status(target_b, "Scenario B")

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
    clicked_a = st.button(
        "Save Scenario A",
        use_container_width=True,
        key="save_a",
        disabled=not can_save_scenarios
    )
    if clicked_a:
        if st.session_state.get("baseline_df") is None:
            st.error("Generate the baseline trajectory first (Step 2).")

        elif (not st.session_state.get("baseline_ready", False)) or (
            st.session_state.get("baseline_signature") != make_baseline_signature()
        ):
            st.error("Baseline exists but is out of date. Re-generate baseline before saving scenarios.")

        elif not ok_a:
            st.error(
                f"Scenario A cannot be saved: target exceeds what you can support "
                f"(Margin £{baseline_margin:,.0f}/w + Discretionary £{disc:,.0f}/w)."
            )

        else:
            target_a_f = float(target_a)
            cut_a_f = float(cut_a_w)

            df_a, params_a = simulate_scenario_df_with_seed_offset(
                income=float(st.session_state["income_w"]),
                fixed_expenses=float(st.session_state["fixed_total_w"]),
                variable_expenses=float(st.session_state["discretionary_w"]),
                delta_savings=cut_a_f,  # ✅ discretionary cut required to reach target
                weeks=int(weeks),
                iterations=int(iterations),
                seed=int(seed),
                seed_offset=0,
                variability_frac=float(variability_frac),
            )

            # Apply one-off shocks (same model as baseline)
            if shock_map:
                df_a = apply_shock_map_to_df(
                    df_a,
                    shock_map=shock_map,
                    value_cols=("Mean", "Lower", "Upper"),
                )

            # --- UI / reproducibility metadata ---
            params_a["preset"] = str(st.session_state["preset_name"])
            params_a["variability_pct"] = int(variability_pct)
            params_a["iterations"] = int(iterations)

            # --- Core financial semantics (Option B: target-first) ---
            params_a["baseline_margin"] = float(baseline_margin)

            params_a["target_savings"] = float(target_a_f)         
            params_a["from_margin"] = float(from_margin_a)          
            params_a["from_discretionary"] = float(from_disc_a)    
            params_a["discretionary_cut"] = float(cut_a_f)          

            # --- Derived / safety ---
            params_a["extra_savings"] = float(from_disc_a)          
            params_a["uncovered"] = 0.0

            st.session_state["scenario_a"] = {"df": df_a.round(2), "params": params_a}
            st.session_state["saved_a_msg"] = True

    if st.session_state.get("saved_a_msg", False):
        saved_badge("Scenario A saved.")

with b2:
    clicked_b = st.button(
        "Save Scenario B",
        use_container_width=True,
        key="save_b",
        disabled=not can_save_scenarios
    )
    if clicked_b:
        if st.session_state.get("baseline_df") is None:
            st.error("Generate the baseline trajectory first (Step 2).")

        elif (not st.session_state.get("baseline_ready", False)) or (
            st.session_state.get("baseline_signature") != make_baseline_signature()
        ):
            st.error("Baseline exists but is out of date. Re-generate baseline before saving scenarios.")

        elif not ok_b:
            st.error(
                f"Scenario B cannot be saved: target exceeds what you can support "
                f"(Margin £{baseline_margin:,.0f}/w + Discretionary £{disc:,.0f}/w)."
            )

        else:
            target_b_f = float(target_b)
            cut_b_f = float(cut_b_w)

            df_b, params_b = simulate_scenario_df_with_seed_offset(
                income=float(st.session_state["income_w"]),
                fixed_expenses=float(st.session_state["fixed_total_w"]),
                variable_expenses=float(st.session_state["discretionary_w"]),
                delta_savings=cut_b_f,  # ✅ discretionary cut required to reach target
                weeks=int(weeks),
                iterations=int(iterations),
                seed=int(seed),
                seed_offset=1,
                variability_frac=float(variability_frac),
            )

            if shock_map:
                df_b = apply_shock_map_to_df(
                    df_b,
                    shock_map=shock_map,
                    value_cols=("Mean", "Lower", "Upper"),
                )

            # --- UI / reproducibility metadata ---
            params_b["preset"] = str(st.session_state["preset_name"])
            params_b["variability_pct"] = int(variability_pct)
            params_b["iterations"] = int(iterations)

            # --- Core financial semantics (Option B: target-first) ---
            params_b["baseline_margin"] = float(baseline_margin)
            params_b["target_savings"] = float(target_b_f)
            params_b["from_margin"] = float(from_margin_b)
            params_b["from_discretionary"] = float(from_disc_b)
            params_b["discretionary_cut"] = float(cut_b_f)

            # --- Derived / safety (optional compatibility) ---
            params_b["extra_savings"] = float(from_disc_b)
            params_b["uncovered"] = 0.0

            st.session_state["scenario_b"] = {"df": df_b.round(2), "params": params_b}
            st.session_state["saved_b_msg"] = True

    if st.session_state.get("saved_b_msg", False):
        saved_badge("Scenario B saved.")


with b3:
    st.button("Clear A/B", use_container_width=True, key="clear_ab", on_click=clear_ab)

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

def assert_cols(df: pd.DataFrame, required: set, name: str):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.error(f"{name} is missing or empty.")
        st.stop()
    missing = required - set(df.columns)
    if missing:
        st.error(f"{name} missing columns: {sorted(missing)}. Found: {df.columns.tolist()}")
        st.stop()

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
    df_a = scenario_a.get("df")
    df_b = scenario_b.get("df")

    assert_cols(baseline_df, {"Week", "Balance"}, "Baseline dataframe")
    assert_cols(df_a, {"Week", "Mean", "Lower", "Upper"}, "Scenario A dataframe")
    assert_cols(df_b, {"Week", "Mean", "Lower", "Upper"}, "Scenario B dataframe")

    baseline_df = baseline_df.sort_values("Week").copy()
    df_a = df_a.sort_values("Week").copy()
    df_b = df_b.sort_values("Week").copy()

    def collapse_weekly(df: pd.DataFrame, value_cols: list[str], mode: str = "last") -> pd.DataFrame:
        """
        Ensures ONE row per Week.
        - mode="last": good for baseline Balance (post-shock, keep the final value for that week)
        - mode="mean": good for simulated bands (averages duplicates if they exist)
        """
        g = df.groupby("Week", as_index=False)[value_cols]
        if mode == "mean":
            return g.mean()
        return g.last()

    baseline_plot = collapse_weekly(baseline_df, ["Balance"], mode="last")
    a_plot = collapse_weekly(df_a, ["Mean", "Lower", "Upper"], mode="mean")
    b_plot = collapse_weekly(df_b, ["Mean", "Lower", "Upper"], mode="mean")

    fig, ax = plt.subplots()

    ax.plot(
        baseline_plot["Week"],
        baseline_plot["Balance"],
        label="Baseline (no change)",
        linestyle="--",
        linewidth=2,
        zorder=5,
    )

    ax.plot(a_plot["Week"], a_plot["Mean"], label="Scenario A (mean)", zorder=4)
    ax.fill_between(
        a_plot["Week"], a_plot["Lower"], a_plot["Upper"],
        alpha=0.2, label="A band (10–90%)", zorder=1
    )

    ax.plot(b_plot["Week"], b_plot["Mean"], label="Scenario B (mean)", zorder=4)
    ax.fill_between(
        b_plot["Week"], b_plot["Lower"], b_plot["Upper"],
        alpha=0.2, label="B band (10–90%)", zorder=1
    )

    ax.set_xlabel("Week")
    ax.set_ylabel("Balance (£)")
    ax.set_title("Trajectory comparison")
    ax.legend()
    st.pyplot(fig)

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
    st.dataframe(summary, use_container_width=True, hide_index=True)

# ============================================================
# Step 4 — Reflect on impact (HUMAN-FIRST)
# ============================================================
st.divider()
st.subheader("Step 4 — Reflect on impact")
st.caption("Plain-English summary: what your weekly targets could mean for your money.")

st.caption(
    "Scenarios reflect **minimum savings commitments**, not spending caps. "
    "If a target is already covered by your current margin, the scenario does not change the baseline."
)

if not ready_for_compare:
    st.info("Complete Step 3 (baseline + saved Scenario A and B) to see the reflection summary.")

else:
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

    # Total weekly saving intent (user target)
    target_a_w = float(params_a.get("target_savings", 0.0))
    target_b_w = float(params_b.get("target_savings", 0.0))

    margin_a_w = float(params_a.get("from_margin", 0.0))
    margin_b_w = float(params_b.get("from_margin", 0.0))

    cut_a_w = float(params_a.get("from_discretionary", params_a.get("delta_savings", 0.0)))
    cut_b_w = float(params_b.get("from_discretionary", params_b.get("delta_savings", 0.0)))

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

    shock_events = (
        st.session_state.get("shock_events_rows", [])
        if st.session_state.get("shock_enabled", False)
        else []
    )

    # ✅ NEW: pass breakdown to reflection builder (for structural deficit insights)
    breakdown = {
        "income_w": float(st.session_state.get("income_w", 0.0)),
        "fixed_total_w": float(st.session_state.get("fixed_total_w", 0.0)),
        "fixed_w": float(to_weekly(st.session_state.get("fixed_essential_raw", 0.0),
                              st.session_state.get("fixed_essential_period", "Weekly"))),
        "var_w": float(to_weekly(st.session_state.get("variable_essential_raw", 0.0),
                            st.session_state.get("variable_essential_period", "Weekly"))),
        "disc_w": float(st.session_state.get("discretionary_w", 0.0)),
        "margin_w": float(st.session_state.get("weekly_margin", 0.0)),
    }

    max_possible_margin = float(st.session_state.get("weekly_margin", 0.0)) +              float(st.session_state.get("discretionary_w", 0.0))
    is_structural_deficit = max_possible_margin < 0

    text = build_human_reflection_text(
        metrics,
        shock_events=shock_events,
        structural_deficit=structural_deficit,
        breakdown=breakdown, 
    )

    # ✅ NEW: structural deficit block (only shown if explain.py provides it)
    if text.get("structural_deficit"):
        st.markdown(text["structural_deficit"])

    st.markdown("### What you get if you follow each plan")
    st.markdown(text.get("what_you_get", ""))

    st.markdown("### Which one should you pick?")
    if getattr(metrics, "winner", "Tie") == "Tie":
        st.info(text.get("pick_primary", ""))
    else:
        st.success(text.get("pick_primary", ""))
        if text.get("pick_secondary"):
            st.warning(text.get("pick_secondary", ""))

    if text.get("shock_note"):
        st.info(text.get("shock_note", ""))

    st.markdown(text.get("variability", ""))

    # --- Technical details hidden (for you / marking) ---
    exp_inputs = ExplanationInputs(
        income=float(st.session_state["income_w"]),
        fixed_expenses=float(st.session_state["fixed_total_w"]),
        variable_expenses=float(st.session_state["discretionary_w"]),
        weeks=int(st.session_state["weeks"]),
        # ✅ Explanation deltas represent the discretionary cut (behaviour change), not the total target
        delta_a=float(cut_a_w),
        delta_b=float(cut_b_w),
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

        st.markdown(
            f"""
**Scenario semantics (traceability):**
- Baseline margin saved by default: **£{baseline_margin:,.0f}/w**
- Scenario A discretionary cut required: **£{cut_a_w:,.0f}/w** → target savings **£{target_a_w:,.0f}/w**
- Scenario B discretionary cut required: **£{cut_b_w:,.0f}/w** → target savings **£{target_b_w:,.0f}/w**
""".strip()
        )

        st.caption(f"Preset: **{preset_name}** ({iters} sims, {var_pct}% variability).")
        st.caption(fairness_note)
