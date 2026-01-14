import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.baseline import generate_baseline
from src.compounder import simulate_scenario

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

# -----------------------
# UI: Header
# -----------------------
st.title("LifeBudget Micro")
st.subheader("Short-term financial clarity")
st.write("Enter your monthly data to generate a baseline projection.")

# -----------------------
# UI: Inputs
# -----------------------
st.header("Monthly Inputs")

income = st.number_input("Monthly income (£)", min_value=0.0, value=2000.0, step=50.0)
fixed = st.number_input("Fixed expenses (£)", min_value=0.0, value=800.0, step=50.0)
variable = st.number_input("Variable expenses (£)", min_value=0.0, value=500.0, step=50.0)

months = st.slider("Projection period (months)", min_value=3, max_value=12, value=6)

# -----------------------
# Baseline (BudgetMind)
# -----------------------
if st.button("Generate baseline"):
    baseline_df = generate_baseline(income, fixed, variable, months).round(2)
    st.session_state["baseline_df"] = baseline_df

st.subheader("Baseline Projection")

if st.session_state["baseline_df"] is not None:
    st.dataframe(st.session_state["baseline_df"], use_container_width=True)
    st.subheader("Projected Balance Over Time")
    st.line_chart(st.session_state["baseline_df"].set_index("Month")["Balance"])
else:
    st.info("Generate a baseline to unlock scenario comparisons.")

# -----------------------
# Scenario Simulation (Compounder+)
# -----------------------
st.divider()
st.subheader("Scenario Simulation (Compounder+)")

delta = st.number_input(
    "Additional monthly savings (£)",
    min_value=0.0,
    value=50.0,
    step=10.0,
    help="This scenario reduces your variable expenses by this amount (saving more each month).",
)

iterations = st.slider(
    "Monte Carlo iterations",
    min_value=50,
    max_value=500,
    value=200,
    step=50,
)

variability_pct = st.slider(
    "Spending variability (%)",
    min_value=0,
    max_value=30,
    value=10,
    step=5,
)

seed = st.number_input(
    "Random seed (reproducibility)",
    min_value=0,
    value=42,
    step=1,
)

# Simple validation warning
if delta > variable:
    st.warning(
        "Your additional savings exceed your current variable expenses. "
        "In some simulations, variable spending will be clamped to £0."
    )

# Helper to run simulation and build a DF
def run_scenario_and_build_df():
    mean, lower, upper = simulate_scenario(
        income=income,
        fixed_expenses=fixed,
        variable_expenses=variable,
        delta_savings=delta,
        months=months,
        iterations=int(iterations),
        seed=int(seed),
        variability_pct=float(variability_pct / 100.0),  # convert 10 -> 0.10
    )

    scenario_df = pd.DataFrame(
        {
            "Month": range(1, months + 1),
            "Mean balance": mean,
            "Lower bound": lower,
            "Upper bound": upper,
        }
    ).round(2)

    params = {
        "delta_savings": float(delta),
        "iterations": int(iterations),
        "variability_pct": int(variability_pct),
        "seed": int(seed),
        "months": int(months),
    }
    return scenario_df, params

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Save Scenario A", use_container_width=True):
        if st.session_state["baseline_df"] is None:
            st.error("Generate the baseline first.")
        else:
            df_a, params_a = run_scenario_and_build_df()
            st.session_state["scenario_a"] = {"df": df_a, "params": params_a}
            st.success("Scenario A saved.")

with col2:
    if st.button("Save Scenario B", use_container_width=True):
        if st.session_state["baseline_df"] is None:
            st.error("Generate the baseline first.")
        else:
            df_b, params_b = run_scenario_and_build_df()
            st.session_state["scenario_b"] = {"df": df_b, "params": params_b}
            st.success("Scenario B saved.")

with col3:
    if st.button("Clear A/B", use_container_width=True):
        st.session_state["scenario_a"] = None
        st.session_state["scenario_b"] = None
        st.info("Scenario A and B cleared.")

# Show stored scenarios (helps examiner)
if st.session_state["scenario_a"] is not None:
    st.caption(f"Scenario A params: {st.session_state['scenario_a']['params']}")
if st.session_state["scenario_b"] is not None:
    st.caption(f"Scenario B params: {st.session_state['scenario_b']['params']}")

# -----------------------
# Comparison section (FR-4 / FR-6)
# -----------------------
st.divider()
st.subheader("Comparison (Baseline vs Scenario A vs Scenario B)")

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

    # Plot: baseline + A mean + B mean + bands
    fig, ax = plt.subplots()

    ax.plot(baseline_df["Month"], baseline_df["Balance"], label="Baseline", linestyle="--")
    ax.plot(df_a["Month"], df_a["Mean balance"], label="Scenario A (mean)")
    ax.fill_between(df_a["Month"], df_a["Lower bound"], df_a["Upper bound"], alpha=0.2, label="A band (10–90%)")

    ax.plot(df_b["Month"], df_b["Mean balance"], label="Scenario B (mean)")
    ax.fill_between(df_b["Month"], df_b["Lower bound"], df_b["Upper bound"], alpha=0.2, label="B band (10–90%)")

    ax.set_xlabel("Month")
    ax.set_ylabel("Balance (£)")
    ax.set_title("Budget projection comparison")
    ax.legend()

    st.pyplot(fig)

    # Summary table
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
            {"Scenario": "Scenario A", "Final balance (mean)": round(a_mean, 2), "Final range (10–90%)": f"{a_low:.2f} – {a_up:.2f}"},
            {"Scenario": "Scenario B", "Final balance (mean)": round(b_mean, 2), "Final range (10–90%)": f"{b_low:.2f} – {b_up:.2f}"},
        ]
    )

    st.subheader("Summary")
    st.dataframe(summary, use_container_width=True)

    # Simple interpretation
    diff_a = a_mean - base_final
    diff_b = b_mean - base_final
    better = "Scenario A" if diff_a > diff_b else "Scenario B"

    st.markdown(
        f"**Interpretation:** Compared to the baseline, "
        f"Scenario A changes the final mean balance by **£{diff_a:.2f}**, "
        f"and Scenario B changes it by **£{diff_b:.2f}**. "
        f"Based on mean outcomes, **{better}** produces the higher projected balance. "
        f"The shaded bands show uncertainty driven by variable spending fluctuations."
    )
