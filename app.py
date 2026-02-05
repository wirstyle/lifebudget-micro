# app.py (FULL) — refactored to move scenario-building + shocks into src/compounder.py
# ✅ Keeps the same UX features:
# 1) shock_enabled checkbox + disabled inputs when off (clean UX / no accidental shocks)
# 2) baseline schema validation + robust shock application (clear error if baseline output shape changes)
#
# Change implemented in this version:
# - Uses validate_baseline_df() from src/baseline.py (single source of truth)
# - Removes the duplicated manual checks (isinstance + required_cols) in app.py
#
# New change (this version):
# - Step 4 logic is centralized in src/explain.py (app.py keeps UI/orchestration only)
#
# ✅ New conceptual change implemented here:
# - Scenario A/B inputs represent a *savings target* (£/week), NOT a direct discretionary reduction.
# - Coverage order (policy): remaining margin → discretionary cut
# - Essentials are NOT adjusted automatically (MVP scope).
# - Scenario is blocked if target > (margin + discretionary).
# - Compounder receives ONLY the real discretionary cut (from_discretionary), not the target.

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.baseline import generate_baseline, validate_baseline_df
from src.compounder import (
    simulate_scenario_df_with_seed_offset,
    apply_one_off_shock_to_df,  # for baseline shock application
)
from src.explain import (
    ExplanationInputs,
    build_explanation,
    compute_reflection_metrics,
    build_human_reflection_text,
)

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

    # ✅ Centralised schema validation (single source of truth)
    try:
        validate_baseline_df(baseline_raw)
    except (ValueError, KeyError) as e:
        st.error(f"Baseline output validation failed: {e}")
        st.stop()

    # ✅ Apply shock ONLY if enabled + amount > 0
    if st.session_state.get("shock_enabled", False) and float(st.session_state.get("shock_amount", 0.0)) > 0:
        baseline_adj = apply_one_off_shock_to_df(
            baseline_raw,
            shock_amount=float(st.session_state["shock_amount"]),
            shock_week=int(st.session_state["shock_week"]),
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
                shock_enabled=bool(st.session_state.get("shock_enabled", False)),
                shock_amount=float(st.session_state.get("shock_amount", 0.0)),
                shock_week=int(st.session_state.get("shock_week", 1)),
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
                shock_enabled=bool(st.session_state.get("shock_enabled", False)),
                shock_amount=float(st.session_state.get("shock_amount", 0.0)),
                shock_week=int(st.session_state.get("shock_week", 1)),
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
        final_low  = float(df["Lower"].iloc[-1])
        final_up   = float(df["Upper"].iloc[-1])
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
st.caption("Plain-English summary: what your weekly targets could mean for your money.")

if not ready_for_compare:
    st.info("Complete Step 3 (baseline + saved Scenario A and B) to see the reflection summary.")
else:
    # Pull final numbers (UI/orchestration only)
    params_a = scenario_a.get("params", {})
    params_b = scenario_b.get("params", {})

    base_final = float(baseline_df["Balance"].iloc[-1])

    a_mean = float(df_a["Mean"].iloc[-1])
    a_low  = float(df_a["Lower"].iloc[-1])
    a_up   = float(df_a["Upper"].iloc[-1])

    b_mean = float(df_b["Mean"].iloc[-1])
    b_low  = float(df_b["Lower"].iloc[-1])
    b_up   = float(df_b["Upper"].iloc[-1])

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

    text = build_human_reflection_text(
        metrics,
        shock_enabled=bool(st.session_state.get("shock_enabled", False)),
        shock_amount=float(st.session_state.get("shock_amount", 0.0)),
        shock_week=int(st.session_state.get("shock_week", 1)),
    )

    # --- 1) Human-first summary ---
    st.markdown("### What you get if you follow each plan")
    st.markdown(text["what_you_get"])

    st.markdown("### Which one should you pick?")
    if getattr(metrics, "winner", "Tie") == "Tie":
        st.info(text["pick_primary"])
    else:
        st.success(text["pick_primary"])
        if text.get("pick_secondary"):
            st.warning(text["pick_secondary"])

    # --- 2) Shock note ---
    if text.get("shock_note"):
        st.info(text["shock_note"])

    # --- 3) Variability note ---
    st.markdown(text["variability"])

    # --- 4) Technical details hidden (for you / marking) ---
    exp_inputs = ExplanationInputs(
        income=float(st.session_state["income_w"]),
        fixed_expenses=float(st.session_state["fixed_total_w"]),
        variable_expenses=float(st.session_state["discretionary_w"]),
        weeks=int(st.session_state["weeks"]),
        # Explanation talks in targets (human intent)
        delta_a=float(target_a_w),
        delta_b=float(target_b_w),
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

    fairness_note = (
        "A and B use the same uncertainty preset; Scenario B uses a controlled seed offset "
        "to keep randomness comparable."
    )

    with st.expander("Technical details (optional)"):
        st.markdown("### Model explanation (technical)")
        st.markdown(explanation_text + shock_note)

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
