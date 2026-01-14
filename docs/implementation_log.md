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


### 2026-01-14 — Compounder+ scenario simulation added

**Goal:**  
Introduce scenario-based simulation to move beyond deterministic projections.

**Work done:**  
- Implemented Monte Carlo–style simulation for scenario analysis.  
- Added Gaussian noise to variable expenses.  
- Fixed random seed for reproducibility.  
- Introduced "additional monthly savings" as scenario parameter.

**Issues encountered:**  
- Initial deterministic approach failed to capture real-world variability.

**Decisions / Fix:**  
- Rejected deterministic model in favour of stochastic simulation.  
- Chose additional savings parameter to align with behavioural focus.

**Evidence:**  
- Scenario simulation visible in Streamlit UI.  
- Commit: feat: add compounder scenario simulation



\### 2026-01-14 — Milestone 2: Compounder+ scenario simulation (Monte Carlo) + repo hygiene



\*\*Goal:\*\*  

Extend the baseline prototype with scenario-based simulation (Compounder+) to support short-term “what-if” analysis with uncertainty.



\*\*Work done:\*\*  

\- Implemented Monte Carlo–style simulation for variable expenses (multiple iterations).  

\- Added scenario parameter: additional monthly savings (modelled as reduced variable spending).  

\- Generated uncertainty bounds using 10th–90th percentiles and plotted uncertainty band.  

\- Improved interpretability by comparing baseline vs scenario on the same chart.  

\- Added user controls for iterations, spending variability, and random seed (reproducibility).  

\- Created a `.gitignore` to prevent committing Python cache files and environment folders.



\*\*Issues encountered:\*\*  

\- Needed to ensure stable and repeatable outputs for assessment evidence; without a fixed seed results varied between runs.



\*\*Decisions / Fix:\*\*  

\- Used a fixed default random seed (user-adjustable) to make scenario results reproducible.  

\- Kept the simulation lightweight (no ML training) to align with MVP scope and ensure fast UI response times.



\*\*Evidence:\*\*  

\- Streamlit UI shows baseline projection and scenario simulation with uncertainty band and interpretation text.  

\- Commits:  

&nbsp; - `feat: add baseline vs scenario comparison + polish scenario output`  

&nbsp; - (optional) `chore: add gitignore`



2026-01-14 — Milestone 4: Explainability / Reasoning layer

Goal:
Add an explainability layer to interpret baseline vs scenario outcomes and articulate why changes occur, not just what changes.

Work done:

Implemented an explainability module (src/explain.py) to generate human-readable interpretations.

Added logic to compare baseline, Scenario A and Scenario B final balances.

Identified dominant drivers (e.g. variable spending vs fixed expenses).

Generated narrative explanations such as which scenario performs best and why, and how variability affects uncertainty bands.

Integrated explanation output directly into the Streamlit UI.

Issues encountered:

Early explanations were too generic and did not clearly reference user inputs.

Risk of over-complicating with ML-based explainability (e.g. SHAP) for MVP scope.

Decisions / Fix:

Rejected SHAP/ML explainability in favour of rule-based reasoning aligned with project scope.

Focused on behavioural drivers (savings, variable expenses, compounding effect).

Prioritised clarity and interpretability over mathematical complexity.

Evidence:

Streamlit UI now displays automatic interpretation below scenario comparison.

Commit: feat: add explainability layer for scenario comparison

