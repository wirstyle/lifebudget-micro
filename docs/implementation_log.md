14 Jan 2026 – Initial prototype and baseline implementation



Goal:

Set up project environment and implement a first runnable version of the LifeBudget Micro prototype with a deterministic baseline projection.



Implemented:



Created project structure (app.py, requirements.txt, src/baseline.py).



Implemented generate\_baseline() function to calculate monthly savings and cumulative balance over a 3–12 month horizon.



Built Streamlit UI to collect monthly income, fixed expenses, and variable expenses, and render a table and balance chart.



Issues encountered:



streamlit command not recognised in Windows due to PATH configuration.



First-time Streamlit setup required email prompt and firewall permission confirmation.



Fix / Decisions:



Ran Streamlit via python -m streamlit run app.py to bypass PATH issue.



Proceeded without enabling network access, as local execution was sufficient for development.



Evidence:



Working Streamlit interface showing baseline table and projected balance chart.



Initial git commit: feat: initial streamlit app with deterministic baseline projection.

