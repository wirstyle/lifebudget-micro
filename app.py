# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.baseline import generate_baseline
from src.compounder import simulate_scenario
from src.explain import ExplanationInputs, build_explanation  # Milestone 4

st.set_page_config(page_title="LifeBudget Micro", layout="centered")

# -----------------------
# Session state defaults
# -----------------------
if "baseline_df" not in st.session_state:
    st.session_state["baseline_df"] = None

if "scenario_a" not in st.session_state:
    st.session_state["scenario_a"] = None  # dict: {"df":..., "params":...}

if "scenario_b" not in st.session_state:
    st.session_state["scenario_b"] = None  # dict: {"df":..., "params":...}

# Hidden seed for reproducible randomness (UX-friendly)
if "seed" not in st.session_state:
    st.session_state["seed"] = 42

# -----------------------
# UI: Header
# -----------------------
st.title("LifeBudget Micro")
st.subheader("Short-term financial clarity")
st.write(
    "Use this tool to understand your baseline trajectory, explore behavioural changes, "
    "and compare outcomes under uncertainty."
)

# -----------------------
# Step 1 — Baseline
# -----------------------
st.header("Step 1 — Establish your baseline")
st.caption(
    "Define your current weekly situation to understand your starting point "
    "(income minus fixed + variable expenses)."
)

income = st.number_input(
    "Weekly income (£)",
    min_value=0.0,
    value=460.0,
    step=10.0,
    help="Your total weekly take-home income (simplified for MVP).",
    key="income",
)
fixed = st.number_input(
    "Weekly fixed expenses (£)",
    min_value=0.0,
    value=185.0,
    step=10.0,
    help="Stable weekly costs (e.g., rent, bills, subscriptions).",
    key="fixed",
)
variable = st.number_input(
    "Weekly variable expenses (£)",
    min_value=0.0,
    value=115.0,
    step=10.0,
    help="Flexible costs that vary week-to-week (e.g., food, travel, leisure).",
    key="variable",
)

weeks = st.slider(
    "Projection period (weeks)",
    min_value=4,
    max_value=52,
    value=12,
    help="Short-term horizon to keep projections interpretable.",
    key="weeks",
)

if st.button("Generate baseline", key="generate_baseline_btn"):
    st.session_state["baseline_df"] = generate_baseline(income, fixed, variable, weeks).round(2)

st.subheader("Baseline Projection")

if st.session_state["baseline_df"] is not None:
    st.caption("Each row represents one week. Balance accumulates weekly savings over time.")

    df_display = st.session_state["baseline_df"].copy()

    try:
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    except TypeError:
        st.dataframe(df_display.style.hide(axis="index"), use_container_width=True)

    st.subheader("Projected Balance Over Time")
    st.line_chart(st.session_state["baseline_df"].set_index("Week")["Balance"])
else:
    st.info("Generate a baseline to unlock scenario exploration and comparisons.")

# -----------------------
# Step 2 — Scenarios (A & B deltas)
# -----------------------
st.divider()
st.subheader("Step 2 — Explore behavioural change")
st.caption(
    "Define two scenarios (A and B) with different weekly savings adjustments. "
    "Results reflect uncertainty in week-to-week variable spending."
)

# --- Presets (1-click) — 100% aligned to conceptual controls ---
PRESETS = {
    "Quick estimate (default)": {"predictability": "Typical", "detail": "Fast"},
    "Typical spending": {"predictability": "Typical", "detail": "Balanced"},
    "Unpredictable weeks": {"predictability": "Chaotic", "detail": "Balanced"},
    "Stress test": {"predictability": "Chaotic", "detail": "High confidence"},
}

PREDICTABILITY_MAP = {"Stable": 10, "Typical": 30, "Chaotic": 50}
DETAIL_MAP = {"Fast": 100, "Balanced": 200, "High confidence": 500}

# Keep UI state
if "preset_name" not in st.session_state:
    st.session_state["preset_name"] = "Quick estimate (default)"
if "predictability" not in st.session_state:
    st.session_state["predictability"] = PRESETS[st.session_state["preset_name"]]["predictability"]
if "detail" not in st.session_state:
    st.session_state["detail"] = PRESETS[st.session_state["preset_name"]]["detail"]

# Layout: preset + seed button
top_left, top_right = st.columns([2, 1])

with top_left:
    preset_name = st.selectbox(
        "Preset",
        list(PRESETS.keys()),
        index=list(PRESETS.keys()).index(st.session_state["preset_name"]),
        help="One-click setup based on the same two settings below.",
        key="preset_select",
    )

    # If user changes preset, apply it directly to the conceptual controls
    if preset_name != st.session_state["preset_name"]:
        st.session_state["preset_name"] = preset_name
        st.session_state["predictability"] = PRESETS[preset_name]["predictability"]
        st.session_state["detail"] = PRESETS[preset_name]["detail"]

with top_right:
    st.markdown("**Randomness**")
    st.caption("For fair A vs B comparison, randomness is kept consistent unless you redraw.")
    if st.button("Try another random run", use_container_width=True, key="reroll_seed"):
        st.session_state["seed"] += 1
        st.toast("New random draw applied.", icon="🎲")

# Two conceptual controls
c1, c2 = st.columns(2)

with c1:
    predictability = st.radio(
        "How predictable are your weekly expenses?",
        list(PREDICTABILITY_MAP.keys()),
        index=list(PREDICTABILITY_MAP.keys()).index(st.session_state["predictability"]),
        help="Stable = small weekly changes. Chaotic = large swings.",
        horizontal=True,
        key="predictability_radio",
    )

with c2:
    detail = st.radio(
        "Result detail",
        list(DETAIL_MAP.keys()),
        index=list(DETAIL_MAP.keys()).index(st.session_state["detail"]),
        help="Higher detail runs more simulations for a smoother, more stable summary.",
        horizontal=True,
        key="detail_radio",
    )

# Update state
st.session_state["predictability"] = predictability
st.session_state["detail"] = detail

# Match preset?
preset_target = PRESETS[st.session_state["preset_name"]]
matches_preset = (
    st.session_state["predictability"] == preset_target["predictability"]
    and st.session_state["detail"] == preset_target["detail"]
)

# Final effective parameters
iterations = int(DETAIL_MAP[st.session_state["detail"]])
variability_pct = int(PREDICTABILITY_MAP[st.session_state["predictability"]])
seed = int(st.session_state["seed"])

if matches_preset:
    st.caption(
        f"Preset applied: **{st.session_state['preset_name']}** "
        f"→ **{iterations} simulations**, **{variability_pct}% unpredictability**."
    )
else:
    st.caption(
        f"Custom settings (based on **{st.session_state['preset_name']}**) "
        f"→ **{iterations} simulations**, **{variability_pct}% unpredictability**."
    )

# Scenario deltas
colA, colB = st.columns(2)

with colA:
    delta_a = st.number_input(
        "Scenario A — Additional weekly savings (£)",
        min_value=0.0,
        value=10.0,
        step=5.0,
        key="delta_a",
        help=(
            "Represents behavioural change (spending less), not extra income. "
            "This reduces your variable expenses each week by the chosen amount."
        ),
    )

with colB:
    delta_b = st.number_input(
        "Scenario B — Additional weekly savings (£)",
        min_value=0.0,
        value=20.0,
        step=5.0,
        key="delta_b",
        help=(
            "Use this to test a different behavioural change from Scenario A "
            "(e.g., a more aggressive or more conservative saving plan)."
        ),
    )

# Defensive UX warnings
if delta_a > variable:
    st.warning(
        "Scenario A savings exceed your current variable expenses. "
        "Variable spending will be clamped to £0 in some runs."
    )
if delta_b > variable:
    st.warning(
        "Scenario B savings exceed your current variable expenses. "
        "Variable spending will be clamped to £0 in some runs."
    )

def run_scenario_and_build_df(delta_savings: float, seed_offset: int = 0):
    """
    Runs the simulation and returns:
    - scenario_df: Mean + uncertainty bounds
    - params: traceability info for examiner + explanation layer
    """
    variability_frac = float(variability_pct) / 100.0  # 30 -> 0.30

    mean, lower, upper = simulate_scenario(
        income=float(income),
        fixed_expenses=float(fixed),
        variable_expenses=float(variable),
        delta_savings=float(delta_savings),
        weeks=int(weeks),
        iterations=int(iterations),
        seed=int(seed) + int(seed_offset),
        variability_frac=float(variability_frac),
    )

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
        "predictability": str(st.session_state["predictability"]),
        "detail": str(st.session_state["detail"]),
    }

    return scenario_df, params

# ✅ RESTORED: Save / Clear buttons
st.caption("Save Scenario A and Scenario B to compare outcomes side-by-side.")
b1, b2, b3 = st.columns(3)

with b1:
    if st.button("Save Scenario A", use_container_width=True, key="save_a"):
        if st.session_state["baseline_df"] is None:
            st.error("Generate the baseline first.")
        else:
            df_a, params_a = run_scenario_and_build_df(delta_savings=delta_a, seed_offset=0)
            st.session_state["scenario_a"] = {"df": df_a, "params": params_a}
            st.success("Scenario A saved.")

with b2:
    if st.button("Save Scenario B", use_container_width=True, key="save_b"):
        if st.session_state["baseline_df"] is None:
            st.error("Generate the baseline first.")
        else:
            # Seed offset keeps B reproducible but avoids accidental identical draws
            df_b, params_b = run_scenario_and_build_df(delta_savings=delta_b, seed_offset=1)
            st.session_state["scenario_b"] = {"df": df_b, "params": params_b}
            st.success("Scenario B saved.")

with b3:
    if st.button("Clear A/B", use_container_width=True, key="clear_ab"):
        st.session_state["scenario_a"] = None
        st.session_state["scenario_b"] = None
        st.info("Scenario A and B cleared.")

# Optional trace for debugging (keep or remove)
if st.session_state["scenario_a"] is not None:
    st.caption(f"Scenario A params: {st.session_state['scenario_a']['params']}")
if st.session_state["scenario_b"] is not None:
    st.caption(f"Scenario B params: {st.session_state['scenario_b']['params']}")

# -----------------------
# Step 3 — Compare
# -----------------------
st.divider()
st.subheader("Step 3 — Compare outcomes")
st.caption("Compare your current trajectory with alternative behavioural scenarios under uncertainty.")

baseline_df = st.session_state["baseline_df"]
scenario_a = st.session_state["scenario_a"]
scenario_b = st.session_state["scenario_b"]

if baseline_df is None:
    st.info("Generate baseline to enable comparison.")
elif scenario_a is None or scenario_b is None:
    st.info("Save both Scenario A and Scenario B to compare.")
else:
    df_a = scenario_a["df"]
    df_b = scenario_b["df"]

    fig, ax = plt.subplots()

    ax.plot(baseline_df["Week"], baseline_df["Balance"], label="Baseline", linestyle="--")

    ax.plot(df_a["Week"], df_a["Mean balance"], label="Scenario A (mean)")
    ax.fill_between(df_a["Week"], df_a["Lower bound"], df_a["Upper bound"], alpha=0.2, label="A band (10–90%)")

    ax.plot(df_b["Week"], df_b["Mean balance"], label="Scenario B (mean)")
    ax.fill_between(df_b["Week"], df_b["Lower bound"], df_b["Upper bound"], alpha=0.2, label="B band (10–90%)")

    ax.set_xlabel("Week")
    ax.set_ylabel("Balance (£)")
    ax.set_title("Budget projection comparison")
    ax.legend()
    st.pyplot(fig)

    def final_stats(df):
        final_mean = float(df["Mean balance"].iloc[-1])
        final_low = float(df["Lower bound"].iloc[-1])
        final_up = float(df["Upper bound"].iloc[-1])
        return final_mean, final_low, final_up

    base_final = float(baseline_df["Balance"].iloc[-1])
    a_mean, a_low, a_up = final_stats(df_a)
    b_mean, b_low, b_up = final_stats(df_b)

    summary = pd.DataFrame(
        [
            {"Scenario": "Baseline", "Final balance (mean)": round(base_final, 2), "Final range (10–90%)": "—"},
            {"Scenario": "Scenario A", "Final balance (mean)": round(a_mean, 2),
             "Final range (10–90%)": f"{a_low:.2f} – {a_up:.2f}"},
            {"Scenario": "Scenario B", "Final balance (mean)": round(b_mean, 2),
             "Final range (10–90%)": f"{b_low:.2f} – {b_up:.2f}"},
        ]
    )

    st.subheader("Summary")
    st.dataframe(summary, use_container_width=True)

    # -----------------------
    # Step 4 — Explainability
    # -----------------------
    st.subheader("Step 4 — Reflect on impact")
    st.caption("Understand why outcomes differ and what is driving change.")

    def band_width(df, idx):
        return float(df["Upper bound"].iloc[idx] - df["Lower bound"].iloc[idx])

    width_a_start = band_width(df_a, 0)
    width_a_end = band_width(df_a, -1)
    width_b_start = band_width(df_b, 0)
    width_b_end = band_width(df_b, -1)

    params_a = scenario_a["params"]
    params_b = scenario_b["params"]

    exp_inputs = ExplanationInputs(
        income=float(income),
        fixed_expenses=float(fixed),
        variable_expenses=float(variable),
        weeks=int(weeks),
        delta_a=float(params_a["delta_savings"]),
        delta_b=float(params_b["delta_savings"]),
        variability_pct=float(params_a["variability_pct"]),  # trace via A
        seed=int(params_a["seed"]),
        iters=int(params_a["iterations"]),
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

    if (width_a_end - width_a_start) > 0 or (width_b_end - width_b_start) > 0:
        extra_insight = (
            f"Uncertainty tends to widen over time (A band width: £{width_a_start:,.2f} → £{width_a_end:,.2f}; "
            f"B band width: £{width_b_start:,.2f} → £{width_b_end:,.2f}), reflecting compounding weekly variability."
        )
    else:
        extra_insight = (
            "Uncertainty does not widen noticeably over time in this run, indicating relatively stable variability under the chosen settings."
        )

    st.markdown("### Interpretation")
    st.markdown(explanation_text + "\n\n" + "**Additional insight:** " + extra_insight)
