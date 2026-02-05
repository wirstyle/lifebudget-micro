# app.py (FULL) — integrated with the 2 mini-implementations:
# 1) shock_enabled checkbox + disabled inputs when off (clean UX / no accidental shocks)
# 2) baseline schema validation + robust shock application (clear error if baseline output shape changes)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.baseline import generate_baseline
from src.compounder import simulate_scenario
from src.explain import ExplanationInputs, build_explanation  # Milestone 4

st.set_page_config(page_title="LifeBudget Micro", layout="centered")

# -----------------------
# Helpers: period -> weekly
# -----------------------
PERIOD_TO_WEEK = {
    "Weekly": 1.0,
    "Monthly": 12.0 / 52.0,   # approx conversion for MVP
    "Yearly": 1.0 / 52.0,
}

def to_weekly(amount: float, period: str) -> float:
    return float(amount) * float(PERIOD_TO_WEEK[period])

# -----------------------
# ✅ Helper: apply one-off shock to any trajectory
# -----------------------
def apply_one_off_shock_to_series(values, shock_amount: float, shock_week: int):
    """
    values: list/np array of balances length == weeks
    shock is applied ONCE at week shock_week, which means:
    - balance decreases by shock_amount from that week onward (cumulative effect)
    """
    if shock_amount <= 0:
        return values
    start_idx = max(int(shock_week) - 1, 0)
    return [
        float(v) - float(shock_amount) if i >= start_idx else float(v)
        for i, v in enumerate(values)
    ]

def apply_one_off_shock_to_df(
    df: pd.DataFrame,
    shock_amount: float,
    shock_week: int,
    week_col: str = "Week",
    value_cols: tuple = ("Balance",),
):
    """
    Applies the same cumulative shock to one or multiple columns in a dataframe.

    Robustness:
    - We apply by row index (week 1 -> row 0), so Week column is not strictly required
      for the operation, but we keep it for clarity.
    - If expected value cols are missing, we raise a clear error.
    """
    if df is None or df.empty or shock_amount <= 0:
        return df

    missing_vals = [c for c in value_cols if c not in df.columns]
    if missing_vals:
        raise KeyError(
            f"Baseline dataframe missing expected columns {missing_vals}. "
            f"Found columns: {df.columns.tolist()}"
        )

    df2 = df.copy()
    start_idx = max(int(shock_week) - 1, 0)

    for col in value_cols:
        vals = df2[col].tolist()
        df2[col] = [
            float(v) - float(shock_amount) if i >= start_idx else float(v)
            for i, v in enumerate(vals)
        ]
    return df2

# -----------------------
# Session state defaults
# -----------------------
if "baseline_df" not in st.session_state:
    st.session_state["baseline_df"] = None

if "scenario_a" not in st.session_state:
    st.session_state["scenario_a"] = None

if "scenario_b" not in st.session_state:
    st.session_state["scenario_b"] = None

if "seed" not in st.session_state:
    st.session_state["seed"] = 42

if "saved_a_msg" not in st.session_state:
    st.session_state["saved_a_msg"] = False
if "saved_b_msg" not in st.session_state:
    st.session_state["saved_b_msg"] = False

if "cleared_msg" not in st.session_state:
    st.session_state["cleared_msg"] = False

# Store weekly-normalised inputs for cross-step coherence
if "income_w" not in st.session_state:
    st.session_state["income_w"] = 460.0
if "fixed_total_w" not in st.session_state:
    st.session_state["fixed_total_w"] = 185.0 + 80.0
if "discretionary_w" not in st.session_state:
    st.session_state["discretionary_w"] = 35.0
if "weekly_margin" not in st.session_state:
    st.session_state["weekly_margin"] = (
        st.session_state["income_w"]
        - st.session_state["fixed_total_w"]
        - st.session_state["discretionary_w"]
    )

# ✅ Single source of truth for widgets
if "income_raw" not in st.session_state:
    st.session_state["income_raw"] = 460.0
if "income_period" not in st.session_state:
    st.session_state["income_period"] = "Weekly"

# Optional helper defaults
if "gross_annual" not in st.session_state:
    st.session_state["gross_annual"] = 30000.0
if "takehome_rate" not in st.session_state:
    st.session_state["takehome_rate"] = 75

# ✅ One-off shock defaults (Step 2)
if "shock_enabled" not in st.session_state:
    st.session_state["shock_enabled"] = False
if "shock_amount" not in st.session_state:
    st.session_state["shock_amount"] = 0.0
if "shock_week" not in st.session_state:
    st.session_state["shock_week"] = 1

# -----------------------
# UI: Header
# -----------------------
st.title("LifeBudget Micro")
st.subheader("Short-term financial clarity")
st.write(
    "Use this tool to understand your current allocation, explore behavioural changes, "
    "and compare outcomes under uncertainty."
)

# ============================================================
# Step 1 — Current position (no projections here)
# ============================================================
st.header("Step 1 — Understand your current position")

# ✅ Framing (2 lines, no more)
st.caption(
    "This step captures your current financial situation.\n"
    "No saving or behavioural changes are applied yet."
)

def apply_gross_estimate_to_income(est_monthly: float):
    st.session_state["income_raw"] = float(est_monthly)
    st.session_state["income_period"] = "Monthly"
    st.toast("Take-home income updated using rough estimate.", icon="✅")

# --- Income (take-home) ---
c1, c2 = st.columns([2, 1])
with c1:
    income_raw = st.number_input(
        "Take-home income (£)",
        min_value=0.0,
        step=10.0,
        help=(
            "Use take-home (net) income. If you only know annual gross salary, use a salary calculator "
            "and paste the take-home amount here — or use the optional rough helper below."
        ),
        key="income_raw",
    )

with c2:
    income_period = st.selectbox(
        "Income period",
        ["Weekly", "Monthly", "Yearly"],
        key="income_period",
    )

# OPTIONAL helper (Option B): gross -> net rough estimate
with st.expander("Only know your annual gross salary? (Optional rough estimate)"):
    st.caption(
        "This helper provides a **rough** gross-to-net estimate using a simple percentage. "
        "**It is not tax advice** and may be inaccurate (tax code, NI, pension, student loan, etc.). "
        "For accuracy, use a salary calculator and enter your real take-home income."
    )

    gross_annual = st.number_input(
        "Annual gross salary (£)",
        min_value=0.0,
        step=1000.0,
        key="gross_annual",
    )

    takehome_rate = st.slider(
        "Assumed take-home rate (%)",
        min_value=60,
        max_value=85,
        step=1,
        help="A simple approximation. Real take-home depends on tax code, NI, pension, student loan, etc.",
        key="takehome_rate",
    )

    est_net_annual = float(gross_annual) * (float(takehome_rate) / 100.0)
    est_net_monthly = est_net_annual / 12.0
    est_net_weekly = est_net_annual / 52.0

    st.write(
        f"Estimated take-home: **£{est_net_monthly:,.0f}/month** "
        f"(≈ **£{est_net_weekly:,.0f}/week**)"
    )

    st.button(
        "Use this estimate as my take-home income",
        key="apply_gross_estimate",
        on_click=apply_gross_estimate_to_income,
        kwargs={"est_monthly": float(est_net_monthly)},
    )

st.subheader("Spending (3 categories)")

# Fixed essential
e1, e2 = st.columns([2, 1])
with e1:
    fixed_essential_raw = st.number_input(
        "Fixed essential commitments (£)",
        min_value=0.0,
        value=185.0,
        step=10.0,
        help="Hard-to-change costs in the short term (e.g., rent, council tax, utilities, mobile plan, basic internet).",
        key="fixed_essential_raw",
    )
with e2:
    fixed_essential_period = st.selectbox(
        "Period",
        ["Weekly", "Monthly", "Yearly"],
        index=0,
        key="fixed_essential_period",
    )

# Variable essentials
e3, e4 = st.columns([2, 1])
with e3:
    variable_essential_raw = st.number_input(
        "Variable essentials (£)",
        min_value=0.0,
        value=80.0,
        step=5.0,
        help="Necessary spending that fluctuates (e.g., groceries, transport, household basics).",
        key="variable_essential_raw",
    )
with e4:
    variable_essential_period = st.selectbox(
        "Period ",
        ["Weekly", "Monthly", "Yearly"],
        index=0,
        key="variable_essential_period",
    )

# Discretionary
e5, e6 = st.columns([2, 1])
with e5:
    discretionary_raw = st.number_input(
        "Discretionary spending (£)",
        min_value=0.0,
        value=35.0,
        step=5.0,
        help="Spending you have the most control over (e.g., takeaways, leisure, drinks, streaming, non-essential shopping).",
        key="discretionary_raw",
    )
with e6:
    discretionary_period = st.selectbox(
        "Period  ",
        ["Weekly", "Monthly", "Yearly"],
        index=0,
        key="discretionary_period",
    )

# Convert to weekly
income_w = to_weekly(float(income_raw), str(income_period))
fixed_essential_w = to_weekly(float(fixed_essential_raw), str(fixed_essential_period))
variable_essential_w = to_weekly(float(variable_essential_raw), str(variable_essential_period))
discretionary_w = to_weekly(float(discretionary_raw), str(discretionary_period))

fixed_total_w = fixed_essential_w + variable_essential_w
weekly_margin = income_w - fixed_total_w - discretionary_w

st.session_state["income_w"] = float(income_w)
st.session_state["fixed_total_w"] = float(fixed_total_w)
st.session_state["discretionary_w"] = float(discretionary_w)
st.session_state["weekly_margin"] = float(weekly_margin)

st.caption(
    f"Weekly equivalents → Income: £{income_w:,.2f} | "
    f"Fixed essential: £{fixed_essential_w:,.2f} | "
    f"Variable essentials: £{variable_essential_w:,.2f} | "
    f"Discretionary: £{discretionary_w:,.2f}"
)

# ============================================================
# ✅ Step 1: "Landing page" default + expander for details
# ============================================================

spent_w = fixed_total_w + discretionary_w
remaining_w = income_w - spent_w

st.markdown(
    f"**Weekly snapshot:** Income **£{income_w:,.0f}** · Expenses **£{spent_w:,.0f}** · "
    f"{'Margin' if remaining_w >= 0 else 'Deficit'} **£{remaining_w:,.0f}/w**"
)

# --- helper formatting + scale equivalents (kept local to Step 1) ---
def money(x: float) -> str:
    return f"£{x:,.0f}"

def week_to_month(x_w: float) -> float:
    return float(x_w) * (52.0 / 12.0)

def week_to_year(x_w: float) -> float:
    return float(x_w) * 52.0

income_m = week_to_month(income_w)
exp_m    = week_to_month(spent_w)
margin_m = week_to_month(remaining_w)

income_y = week_to_year(income_w)
exp_y    = week_to_year(spent_w)
margin_y = week_to_year(remaining_w)

with st.expander("See breakdown and equivalents", expanded=False):
    st.subheader("Current weekly allocation")

    segments = {
        "Fixed essential": float(fixed_essential_w),
        "Variable essentials": float(variable_essential_w),
        "Discretionary": float(discretionary_w),
    }
    if remaining_w >= 0:
        segments["Remaining margin"] = float(remaining_w)
    else:
        segments["Deficit (overspend)"] = float(abs(remaining_w))

    # Legend labels WITH numbers (weekly)
    legend_labels = {k: f"{k} ({money(v)}/w)" for k, v in segments.items()}

    fig_alloc, ax_alloc = plt.subplots(figsize=(7, 1.4))
    left = 0.0
    for label, val in segments.items():
        ax_alloc.barh(["Allocation"], [val], left=left, label=legend_labels[label])
        left += float(val)

    ax_alloc.set_xlabel("£ per week")
    ax_alloc.set_yticks([])

    ax_alloc.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.55),
        ncol=2,
        frameon=False
    )

    fig_alloc.subplots_adjust(bottom=0.42)
    st.pyplot(fig_alloc)

    st.markdown(
        f"""
**Note on scale:**  
The proportions above remain the same at different time scales.

- **Monthly equivalent (approx):**
  - Income: **{money(income_m)}**
  - Expenses: **{money(exp_m)}**
  - Remaining margin: **{money(margin_m)}**

- **Yearly equivalent:**
  - Income: **{money(income_y)}**
  - Expenses: **{money(exp_y)}**
  - Remaining margin: **{money(margin_y)}**
""".strip()
    )

# ✅ CTA (clean, no pressure) — end of Step 1
st.markdown("**Next:** explore what happens if you change something (Step 2).")

# ============================================================
# Step 2 — Scenarios (A & B deltas)
# ============================================================
st.divider()
st.subheader("Step 2 — Explore behavioural change")
st.caption(
    "Define two scenarios (A and B) with different weekly behaviour adjustments. "
    "Uncertainty is applied to **discretionary** spending (the most controllable category)."
)

# -----------------------
# ✅ Baseline status state
# -----------------------
if "baseline_ready" not in st.session_state:
    st.session_state["baseline_ready"] = False
if "baseline_signature" not in st.session_state:
    st.session_state["baseline_signature"] = None

def make_baseline_signature():
    return {
        "income_w": round(float(st.session_state["income_w"]), 6),
        "fixed_total_w": round(float(st.session_state["fixed_total_w"]), 6),
        "discretionary_w": round(float(st.session_state["discretionary_w"]), 6),
        "weeks": int(st.session_state["weeks"]),
        "shock_enabled": bool(st.session_state.get("shock_enabled", False)),
        "shock_amount": round(float(st.session_state.get("shock_amount", 0.0)), 6),
        "shock_week": int(st.session_state.get("shock_week", 1)),
    }

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
# Optional one-off event (shock)
# -----------------------
with st.expander("Optional — One-off event (unexpected expense)"):
    st.caption(
        "Model a single unexpected expense applied once during the planning horizon "
        "(e.g., a night out, repair, fine). This is **not** expense tracking."
    )

    shock_enabled = st.checkbox(
        "Enable one-off event",
        key="shock_enabled",
        help="Turn on to apply a single unexpected expense once during the horizon.",
        on_change=mark_baseline_stale,
    )

    shock_amount = st.number_input(
        "Unexpected expense amount (£)",
        min_value=0.0,
        step=5.0,
        key="shock_amount",
        disabled=not shock_enabled,
        on_change=mark_baseline_stale,
    )

    shock_week = st.slider(
        "Week of event",
        min_value=1,
        max_value=int(weeks),
        value=min(int(st.session_state.get("shock_week", 1)), int(weeks)),
        key="shock_week",
        disabled=not shock_enabled,
        help="The expense is applied once and reduces balance from that week onward.",
        on_change=mark_baseline_stale,
    )

    if shock_enabled and float(st.session_state["shock_amount"]) > 0:
        st.info(
            f"Active: £{float(st.session_state['shock_amount']):,.2f} at week {int(st.session_state['shock_week'])}."
        )

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

    # ✅ Robustness: baseline must be a DataFrame with Week + Balance
    if not isinstance(baseline_raw, pd.DataFrame):
        st.error(f"generate_baseline must return a DataFrame, got: {type(baseline_raw)}")
        st.stop()

    required_cols = {"Week", "Balance"}
    if not required_cols.issubset(set(baseline_raw.columns)):
        st.error(
            "Baseline output has an unexpected schema.\n\n"
            f"Expected columns: {sorted(required_cols)}\n"
            f"Got columns: {baseline_raw.columns.tolist()}\n\n"
            "Fix src/baseline.py to output Week + Balance."
        )
        st.stop()

    # ✅ Apply shock ONLY if enabled + amount > 0
    if st.session_state.get("shock_enabled", False) and float(st.session_state.get("shock_amount", 0.0)) > 0:
        baseline_adj = apply_one_off_shock_to_df(
            baseline_raw,
            shock_amount=float(st.session_state["shock_amount"]),
            shock_week=int(st.session_state["shock_week"]),
            week_col="Week",
            value_cols=("Balance",),
        )
        st.session_state["baseline_df"] = baseline_adj
    else:
        st.session_state["baseline_df"] = baseline_raw

    # ✅ Persistent indicator + traceability signature
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
    # settings changed → saved A/B now stale
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
# Scenario deltas
# -----------------------
colA, colB = st.columns(2)
with colA:
    delta_a = st.number_input(
        "Scenario A — Reduce discretionary spending by (£/week)",
        min_value=0.0,
        value=10.0,
        step=5.0,
        key="delta_a",
        help="Behavioural change (spending less). This reduces discretionary spending each week by the chosen amount.",
    )
with colB:
    delta_b = st.number_input(
        "Scenario B — Reduce discretionary spending by (£/week)",
        min_value=0.0,
        value=20.0,
        step=5.0,
        key="delta_b",
        help="Test a different behavioural change from Scenario A (e.g., more aggressive or more conservative).",
    )

disc = float(st.session_state.get("discretionary_w", 0.0))
if delta_a > disc:
    st.warning("Scenario A exceeds your current discretionary spending. Discretionary will be clamped to £0 in some runs.")
if delta_b > disc:
    st.warning("Scenario B exceeds your current discretionary spending. Discretionary will be clamped to £0 in some runs.")

def run_scenario_and_build_df(delta_savings: float, seed_offset: int = 0):
    income_eff = float(st.session_state["income_w"])
    fixed_eff = float(st.session_state["fixed_total_w"])
    disc_eff = float(st.session_state["discretionary_w"])

    mean, lower, upper = simulate_scenario(
        income=income_eff,
        fixed_expenses=fixed_eff,
        variable_expenses=disc_eff,
        delta_savings=float(delta_savings),
        weeks=int(weeks),
        iterations=int(iterations),
        seed=int(seed) + int(seed_offset),
        variability_frac=float(variability_frac),
    )

    # ✅ Apply the one-off shock to scenario bands ONLY if enabled + amount > 0
    if st.session_state.get("shock_enabled", False) and float(st.session_state.get("shock_amount", 0.0)) > 0:
        shock_amount_eff = float(st.session_state["shock_amount"])
        shock_week_eff = int(st.session_state["shock_week"])

        mean  = apply_one_off_shock_to_series(mean,  shock_amount_eff, shock_week_eff)
        lower = apply_one_off_shock_to_series(lower, shock_amount_eff, shock_week_eff)
        upper = apply_one_off_shock_to_series(upper, shock_amount_eff, shock_week_eff)
    else:
        shock_amount_eff = 0.0
        shock_week_eff = 1

    scenario_df = pd.DataFrame(
        {
            "Week": range(1, int(weeks) + 1),
            "Mean balance": mean,
            "Lower bound": lower,
            "Upper bound": upper,
        }
    ).round(2)

    params = {
        "delta_savings": float(delta_savings),
        "iterations": int(iterations),
        "variability_pct": int(variability_pct),
        "variability_frac": float(variability_frac),
        "seed": int(seed) + int(seed_offset),
        "weeks": int(weeks),
        "preset": str(st.session_state["preset_name"]),
        "income_w": float(income_eff),
        "fixed_total_w": float(fixed_eff),
        "discretionary_w": float(disc_eff),
        "shock_enabled": bool(st.session_state.get("shock_enabled", False)),
        "shock_amount": float(shock_amount_eff),
        "shock_week": int(shock_week_eff),
    }

    return scenario_df, params

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
        else:
            df_a, params_a = run_scenario_and_build_df(delta_savings=delta_a, seed_offset=0)
            st.session_state["scenario_a"] = {"df": df_a, "params": params_a}
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
        else:
            df_b, params_b = run_scenario_and_build_df(delta_savings=delta_b, seed_offset=1)
            st.session_state["scenario_b"] = {"df": df_b, "params": params_b}
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
st.caption("Compare your baseline trajectory with alternative behavioural scenarios under uncertainty.")

baseline_df = st.session_state.get("baseline_df")
scenario_a  = st.session_state.get("scenario_a")
scenario_b  = st.session_state.get("scenario_b")

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
if baseline_df is None:
    st.info("Generate the baseline trajectory in Step 2 to enable comparison.")
elif not st.session_state.get("baseline_ready", False) or st.session_state.get("baseline_signature") != make_baseline_signature():
    st.warning("Baseline exists but inputs have changed. Re-generate baseline in Step 2 to update comparison.")
elif scenario_a is None or scenario_b is None:
    st.info("Save both Scenario A and Scenario B to compare.")
else:
    # --- Pull scenario dataframes ---
    df_a = scenario_a.get("df")
    df_b = scenario_b.get("df")

    # --- Validate expected schemas ---
    assert_cols(baseline_df, {"Week", "Balance"}, "Baseline dataframe")
    assert_cols(df_a, {"Week", "Mean balance", "Lower bound", "Upper bound"}, "Scenario A dataframe")
    assert_cols(df_b, {"Week", "Mean balance", "Lower bound", "Upper bound"}, "Scenario B dataframe")

    # --- Plot ---
    fig, ax = plt.subplots()

    ax.plot(baseline_df["Week"], baseline_df["Balance"], label="Baseline (no change)", linestyle="--")

    ax.plot(df_a["Week"], df_a["Mean balance"], label="Scenario A (mean)")
    ax.fill_between(df_a["Week"], df_a["Lower bound"], df_a["Upper bound"], alpha=0.2, label="A band (10–90%)")

    ax.plot(df_b["Week"], df_b["Mean balance"], label="Scenario B (mean)")
    ax.fill_between(df_b["Week"], df_b["Lower bound"], df_b["Upper bound"], alpha=0.2, label="B band (10–90%)")

    ax.set_xlabel("Week")
    ax.set_ylabel("Balance (£)")
    ax.set_title("Trajectory comparison")
    ax.legend()
    st.pyplot(fig)

    # --- Summary stats ---
    def final_stats(df):
        final_mean = float(df["Mean balance"].iloc[-1])
        final_low  = float(df["Lower bound"].iloc[-1])
        final_up   = float(df["Upper bound"].iloc[-1])
        return final_mean, final_low, final_up

    base_final = float(baseline_df["Balance"].iloc[-1])
    a_mean, a_low, a_up = final_stats(df_a)
    b_mean, b_low, b_up = final_stats(df_b)

    summary = pd.DataFrame(
        [
            {"Scenario": "Baseline",   "Final balance (mean)": round(base_final, 2), "Final range (10–90%)": "—"},
            {"Scenario": "Scenario A", "Final balance (mean)": round(a_mean, 2),     "Final range (10–90%)": f"{a_low:.2f} – {a_up:.2f}"},
            {"Scenario": "Scenario B", "Final balance (mean)": round(b_mean, 2),     "Final range (10–90%)": f"{b_low:.2f} – {b_up:.2f}"},
        ]
    )

    st.subheader("Summary")
    st.dataframe(summary, use_container_width=True)

# ============================================================
# Step 4 — Reflect on impact (HUMAN-FIRST)
# ============================================================
st.divider()
st.subheader("Step 4 — Reflect on impact")
st.caption("Plain-English summary: what your weekly change could mean for your money.")

# --- helpers ---
def fmt_gbp(x: float) -> str:
    return f"£{x:,.0f}"

def pct_change(new: float, base: float) -> float:
    if base == 0:
        return 0.0
    return ((new - base) / base) * 100.0

def band_width_start_end(df):
    w0 = float(df["Upper bound"].iloc[0] - df["Lower bound"].iloc[0])
    w1 = float(df["Upper bound"].iloc[-1] - df["Lower bound"].iloc[-1])
    return w0, w1

# --- uncertainty widths (for optional tech + a tiny human note) ---
width_a_start, width_a_end = band_width_start_end(df_a)
width_b_start, width_b_end = band_width_start_end(df_b)

params_a = scenario_a.get("params", {})

# --- key numbers (human) ---
weeks_n = int(st.session_state["weeks"])
delta_a_w = float(scenario_a["params"]["delta_savings"])
delta_b_w = float(scenario_b["params"]["delta_savings"])

a_gain = float(a_mean - base_final)
b_gain = float(b_mean - base_final)
gap_ab = float(b_mean - a_mean)

a_pct = pct_change(float(a_mean), float(base_final))
b_pct = pct_change(float(b_mean), float(base_final))

winner = "Scenario A" if a_mean > b_mean else "Scenario B" if b_mean > a_mean else "Tie"
safer = "Scenario A" if a_low > b_low else "Scenario B" if b_low > a_low else "Tie"

# --- 1) Human-first summary ---
st.markdown("### What you get if you follow each plan")

st.markdown(
    f"- If you **change nothing**, you end around **{fmt_gbp(base_final)}** after **{weeks_n} weeks**."
)

st.markdown(
    f"- **Scenario A**: spend **{fmt_gbp(delta_a_w)}/week less** → end around **{fmt_gbp(a_mean)}** "
    f"(≈ **{fmt_gbp(a_gain)} more** than baseline, **{a_pct:.1f}%**)."
)

st.markdown(
    f"- **Scenario B**: spend **{fmt_gbp(delta_b_w)}/week less** → end around **{fmt_gbp(b_mean)}** "
    f"(≈ **{fmt_gbp(b_gain)} more** than baseline, **{b_pct:.1f}%**)."
)

# Direct answer to “which one is better?”
st.markdown("### Which one should you pick?")

if winner == "Tie":
    st.info(
        "Both scenarios are very close on the expected outcome. "
        "Pick the one that feels easier to stick to every week."
    )
else:
    # winner text
    if winner == "Scenario A":
        st.success(
            f"On average, **Scenario A** leaves you with more money. "
            f"The gap vs Scenario B is about **{fmt_gbp(abs(gap_ab))}**."
        )
    else:
        st.success(
            f"On average, **Scenario B** leaves you with more money. "
            f"The gap vs Scenario A is about **{fmt_gbp(abs(gap_ab))}**."
        )

    # safety / worst-case cue (still simple)
    if safer != "Tie" and safer != winner:
        st.warning(
            f"However, the **safer worst-case** (lower-bound) looks like **{safer}**. "
            "So one option wins on average, but the other is slightly more resilient in a bad draw."
        )

# --- 2) Shock note (human) ---
if st.session_state.get("shock_enabled", False) and float(st.session_state.get("shock_amount", 0.0)) > 0:
    shock_amount = float(st.session_state["shock_amount"])
    shock_week = int(st.session_state["shock_week"])
    st.info(
        f"One-off event included: **{fmt_gbp(shock_amount)}** happens in **week {shock_week}** "
        f"(it reduces the balance from that week onward)."
    )

# --- 3) Tiny uncertainty note (human, not nerdy) ---
# Keep it simple: "results may vary" + optionally show ranges at the end
st.markdown("### Results can vary (because real weeks aren’t identical)")

st.markdown(
    f"- Scenario A likely ends somewhere around **{fmt_gbp(a_low)} to {fmt_gbp(a_up)}**.\n"
    f"- Scenario B likely ends somewhere around **{fmt_gbp(b_low)} to {fmt_gbp(b_up)}**."
)

# --- 4) Technical details hidden (for you / marking) ---
# Build explanation text as before, but do not force user to read it.
exp_inputs = ExplanationInputs(
    income=float(st.session_state["income_w"]),
    fixed_expenses=float(st.session_state["fixed_total_w"]),
    variable_expenses=float(st.session_state["discretionary_w"]),
    weeks=int(st.session_state["weeks"]),
    delta_a=float(scenario_a["params"]["delta_savings"]),
    delta_b=float(scenario_b["params"]["delta_savings"]),
    variability_pct=float(params_a.get("variability_pct", 0)),
    seed=int(params_a.get("seed", st.session_state.get("seed", 0))),
    iters=int(params_a.get("iterations", 0)),
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

shock_note = ""
if st.session_state.get("shock_enabled", False) and float(st.session_state.get("shock_amount", 0.0)) > 0:
    shock_amount = float(st.session_state["shock_amount"])
    shock_week = int(st.session_state["shock_week"])
    shock_note = (
        f"\n\n**One-off event applied:** £{shock_amount:,.2f} at week {shock_week} "
        f"(reduces balance from that week onward)."
    )

preset_name = str(params_a.get("preset", st.session_state.get("preset_name", "Preset")))
iters = int(params_a.get("iterations", 0))
var_pct = int(params_a.get("variability_pct", 0))

widens_a = (width_a_end - width_a_start) > 0
widens_b = (width_b_end - width_b_start) > 0

if widens_a or widens_b:
    extra_insight = (
        f"Uncertainty band widens over time (A: £{width_a_start:,.2f} → £{width_a_end:,.2f}; "
        f"B: £{width_b_start:,.2f} → £{width_b_end:,.2f}) under preset **{preset_name}** "
        f"({iters} sims, {var_pct}% variability)."
    )
else:
    extra_insight = (
        f"Uncertainty does not widen noticeably under preset **{preset_name}** "
        f"({iters} sims, {var_pct}% variability)."
    )

fairness_note = (
    "A and B use the same uncertainty preset; Scenario B uses a controlled seed offset "
    "to keep randomness comparable."
)

with st.expander("Technical details (optional)"):
    st.markdown("### Model explanation (technical)")
    st.markdown(explanation_text + shock_note)
    st.markdown("**Additional insight:** " + extra_insight)
    st.caption(fairness_note)
