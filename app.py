import streamlit as st
from src.baseline import generate_baseline

st.set_page_config(page_title="LifeBudget Micro", layout="centered")

st.title("LifeBudget Micro")
st.subheader("Short-term financial clarity")

st.write("Enter your monthly data to generate a baseline projection.")

st.header("Monthly Inputs")

income = st.number_input("Monthly income (£)", min_value=0.0, value=2000.0, step=50.0)
fixed = st.number_input("Fixed expenses (£)", min_value=0.0, value=800.0, step=50.0)
variable = st.number_input("Variable expenses (£)", min_value=0.0, value=500.0, step=50.0)

months = st.slider("Projection period (months)", min_value=3, max_value=12, value=6)

if st.button("Generate baseline"):
    df = generate_baseline(income, fixed, variable, months)

    st.subheader("Baseline Projection")
    st.dataframe(df)

    st.subheader("Projected Balance Over Time")
    st.line_chart(df.set_index("Month")["Balance"])
