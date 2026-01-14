import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.baseline import generate_baseline
from src.compounder import simulate_scenario
from src.explain import ExplanationInputs, build_explanation  # <-- Milestone 4

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
def run_scenario_and_build_df(seed_offset: int = 0):
    """
    seed_offset helps avoid scenarios being identical when user saves A and B
    with same parameters by accident.
    """
    mean, lower, upper = simulate_scenario(
        income=income,
        fixed_expenses=fixed,
        variable_expenses=variable,
        delta_savings=delta,
        months=months,
        iterations=int(iterations),
        seed=int(seed) + int(seed_offset),
        variability_pct=float(variability_pct / 100.0),  # convert 10 -> 0.10 (fraction)
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
        "variability_pct": int(variability_pct),  # keep as percent for explanation text
        "seed": int(seed) + int(seed_offset),
        "months": int(months),
    }
    return scenario_df, params

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Save Scenario A", use_container_width=True):
        if st.session_state["baseline_df"] is None:
            st.error("Generate the baseline first.")
        else:
            df_a, params_a = run_scenario_and_build_df(seed_offset=0)
            st.session_state["scenario_a"] = {"df": df_a, "params": params_a}
            st.success("Scenario A saved.")

with col2:
    if st.button("Save Scenario B", use_container_width=True):
        if st.session_state["baseline_df"] is None:
            st.error("Generate the baseline first.")
        else:
            # Offset seed by 1 so B is reproducible but not identical to A by accident
            df_b, params_b = run_scenario_and_build_df(seed_offset=1)
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

    # -----------------------
    # Milestone 4: Explainability / Reasoning layer
    # -----------------------
    # extra: check whether uncertainty band widens over time
    def band_width(df, idx):
        return float(df["Upper bound"].iloc[idx] - df["Lower bound"].iloc[idx])

    width_a_start = band_width(df_a, 0)
    width_a_end = band_width(df_a, -1)
    width_b_start = band_width(df_b, 0)
    width_b_end = band_width(df_b, -1)

    # Pull params saved for A/B (so explanation references the actual saved scenarios)
    params_a = scenario_a["params"]
    params_b = scenario_b["params"]

    exp_inputs = ExplanationInputs(
        income=float(income),
        fixed_expenses=float(fixed),
        variable_expenses=float(variable),
        months=int(months),
        delta_a=float(params_a["delta_savings"]),
        delta_b=float(params_b["delta_savings"]),
        variability_pct=float(params_a["variability_pct"]),  # they should match, but we use A
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

    # Add one extra smart sentence about widening uncertainty over time
    widening_a = width_a_end - width_a_start
    widening_b = width_b_end - width_b_start

    extra_lines = []
    if widening_a > 0 or widening_b > 0:
        extra_lines.append(
            f"Uncertainty tends to widen over time (A band width: £{width_a_start:,.2f} → £{width_a_end:,.2f}; "
            f"B band width: £{width_b_start:,.2f} → £{width_b_end:,.2f}), which reflects compounding monthly variability."
        )
    else:
        extra_lines.append(
            "Uncertainty does not widen noticeably over time in this run, indicating relatively stable variability under the chosen settings."
        )

    st.markdown("### Interpretation")
    st.markdown(explanation_text + "\n\n" + "**Additional insight:** " + extra_lines[0])
