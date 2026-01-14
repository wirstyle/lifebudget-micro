import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.baseline import generate_baseline
from src.compounder import simulate_scenario

st.set_page_config(page_title="LifeBudget Micro", layout="centered")

st.title("LifeBudget Micro")
st.subheader("Short-term financial clarity")

st.write("Enter your monthly data to generate a baseline projection.")

st.header("Monthly Inputs")

income = st.number_input("Monthly income (£)", min_value=0.0, value=2000.0, step=50.0)
fixed = st.number_input("Fixed expenses (£)", min_value=0.0, value=800.0, step=50.0)
variable = st.number_input("Variable expenses (£)", min_value=0.0, value=500.0, step=50.0)

months = st.slider("Projection period (months)", min_value=3, max_value=12, value=6)

# -----------------------------
# Baseline (BudgetMind)
# -----------------------------
if st.button("Generate baseline"):
    df = generate_baseline(income, fixed, variable, months)

    st.subheader("Baseline Projection")
    # If you want to hide index (optional):
    # st.dataframe(df, hide_index=True)
    st.dataframe(df)

    st.subheader("Projected Balance Over Time")
    st.line_chart(df.set_index("Month")["Balance"])

st.divider()

# -----------------------------
# Scenario Simulation (Compounder+)
# -----------------------------
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
) / 100.0

seed = st.number_input(
    "Random seed (reproducibility)",
    min_value=0,
    value=42,
    step=1,
)

if st.button("Run scenario simulation"):
    mean, lower, upper = simulate_scenario(
        income=income,
        fixed_expenses=fixed,
        variable_expenses=variable,
        delta_savings=delta,
        months=months,
        iterations=int(iterations),
        seed=int(seed),
        variability_pct=float(variability_pct),
    )

    scenario_df = pd.DataFrame({
        "Month": range(1, months + 1),
        "Mean balance": mean,
        "Lower bound": lower,
        "Upper bound": upper
    })

    st.subheader("Scenario Projection")
    st.dataframe(scenario_df)

    # Plot with uncertainty band
    fig, ax = plt.subplots()
    ax.plot(scenario_df["Month"], scenario_df["Mean balance"], label="Scenario mean")
    ax.fill_between(
        scenario_df["Month"],
        scenario_df["Lower bound"],
        scenario_df["Upper bound"],
        alpha=0.3,
        label="Uncertainty band (10–90%)"
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Balance (£)")
    ax.set_title("Scenario Simulation – Compounder+")
    ax.legend()

    st.pyplot(fig)

    st.markdown(
        f"**Interpretation:** Increasing your savings by **£{delta:.0f}/month** "
        f"raises the projected balance over time. The shaded area shows uncertainty "
        f"caused by fluctuations in variable spending (±{int(variability_pct*100)}%)."
    )
