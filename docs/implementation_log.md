### 14-01-2026 – Milestone 0: Initial prototype and baseline implementation

**Goal:**  
Set up project environment and implement a first runnable version of the LifeBudget Micro prototype with a deterministic baseline projection.

**Work done:**  
- Created project structure (app.py, requirements.txt, src/baseline.py).
- Implemented generate\_baseline() function to calculate monthly savings and cumulative balance over a 3–12 month horizon.
- Built Streamlit UI to collect monthly income, fixed expenses, and variable expenses, and render a table and balance chart.

**Issues encountered:**  
- Streamlit command not recognised in Windows due to PATH configuration.
- First-time Streamlit setup required email prompt and firewall permission confirmation.

**Decisions / Fix:**  
- Ran Streamlit via python -m streamlit run app.py to bypass PATH issue.
- Proceeded without enabling network access, as local execution was sufficient for development.

**Evidence:**  
- Working Streamlit interface showing baseline table and projected balance chart.
- Initial git commit: feat: initial streamlit app with deterministic baseline projection.

### 14-01-2026 — Milestone 1: Compounder+ scenario simulation added

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

### 14-01-2026 — Milestone 2: Compounder+ scenario simulation (Monte Carlo) + repo hygiene

**Goal:**  
Extend the baseline prototype with scenario-based simulation (Compounder+) to support short-term “what-if” analysis with uncertainty.

**Work done:**  
- Implemented Monte Carlo–style simulation for variable expenses (multiple iterations).  
- Added scenario parameter: additional monthly savings (modelled as reduced variable spending).  
- Generated uncertainty bounds using 10th–90th percentiles and plotted uncertainty band.  
- Improved interpretability by comparing baseline vs scenario on the same chart.  
- Added user controls for iterations, spending variability, and random seed (reproducibility).  
- Created a `.gitignore` to prevent committing Python cache files and environment folders.

**Issues encountered:**  
- Needed to ensure stable and repeatable outputs for assessment evidence; without a fixed seed results varied between runs.

**Decisions / Fix:**  
- Used a fixed default random seed (user-adjustable) to make scenario results reproducible.  
- Kept the simulation lightweight (no ML training) to align with MVP scope and ensure fast UI response times.

**Evidence:**  
- Streamlit UI shows baseline projection and scenario simulation with uncertainty band and interpretation text.  
- Commits:  
	- `feat: add baseline vs scenario comparison + polish scenario output`  
	- (optional) `chore: add gitignore`

### 14-01-2026 — Milestone 3: Scenario comparison (Baseline vs A vs B)

**Goal:**  
Enable direct comparison between baseline projection and multiple scenarios to support informed decision-making.

**Work done:**  
- Added support for storing two independent scenarios (A and B) in session state.  
- Implemented comparison view showing Baseline vs Scenario A vs Scenario B on the same chart.  
- Plotted mean trajectories and uncertainty bands (10–90%) for both scenarios.  
- Added summary table showing final mean balance and uncertainty range for each case.  
- Implemented automatic comparison logic to identify which scenario performs best.

**Issues encountered:**  
- Streamlit state was being reset between interactions, causing scenarios to be lost.  
- Early versions could not reliably compare scenarios without regenerating baseline.

**Decisions / Fix:**  
- Introduced `st.session_state` to persist baseline and scenarios across UI interactions.  
- Required baseline generation before enabling scenario comparison to avoid inconsistent states.  
- Designed comparison view to remain simple and interpretable (no cluttered multi-metric dashboard).

**Evidence:**  
- Streamlit UI displays combined chart: Baseline (dashed) vs Scenario A vs Scenario B with uncertainty bands.  
- Summary table shows final balances and ranges for each case.  
- Commit: feat: add baseline vs scenario A/B comparison


### 14-01-2026 — Milestone 4: Explainability / Reasoning layer

**Goal:**
Add an explainability layer to interpret baseline vs scenario outcomes and articulate why changes occur, not just what changes.

**Work done:**
- Implemented an explainability module (src/explain.py) to generate human-readable interpretations.
- Added logic to compare baseline, Scenario A and Scenario B final balances.
- Identified dominant drivers (e.g. variable spending vs fixed expenses).
- Generated narrative explanations such as which scenario performs best and why, and how variability affects uncertainty bands.
- Integrated explanation output directly into the Streamlit UI.

**Issues encountered:**
- Early explanations were too generic and did not clearly reference user inputs.
- Risk of over-complicating with ML-based explainability (e.g. SHAP) for MVP scope.

**Decisions / Fix:**
- Rejected SHAP/ML explainability in favour of rule-based reasoning aligned with project scope.
- Focused on behavioural drivers (savings, variable expenses, compounding effect).
- Prioritised clarity and interpretability over mathematical complexity.

**Evidence:**
- Streamlit UI now displays automatic interpretation below scenario comparison.
- Commit: feat: add explainability layer for scenario comparison

### 14-01-2026 — Milestone 5: UX, interaction design and narrative layer

**Goal:**  
Transform the technical prototype into a guided, interpretable decision-support interface by improving interaction flow, microcopy, and narrative structure.

**Work done:**  
- Reframed the interface into a step-based flow:  
  1. Establish baseline  
  2. Explore behavioural change  
  3. Compare outcomes  
  4. Reflect on impact  
- Added contextual microcopy and helper text to all key inputs (e.g. additional savings, variability, iterations).  
- Clarified the meaning of scenario parameters (e.g. behavioural change vs extra income).  
- Introduced progressive disclosure (scenarios and comparison only visible after baseline generation).  
- Added descriptive captions to explain purpose of each stage and reduce cognitive load.  
- Aligned interface language with project narrative (short-term clarity, behavioural impact, uncertainty).

**Issues encountered:**  
- Early versions of the UI were technically functional but lacked guidance, which risked user confusion.  
- Some parameters (e.g. variability, Monte Carlo iterations) were not self-explanatory for non-technical users.

**Decisions / Fix:**  
- Structured the interface around a mental model progression (baseline → intervention → comparison → reflection).  
- Used microcopy and captions instead of additional UI complexity to maintain simplicity.  
- Prioritised clarity and interpretability over dense feature sets to align with MVP scope.

**Evidence:**  
- Streamlit UI now displays step-based sections with guidance and narrative framing.  
- Input fields include explanatory helper text.  
- Commit: feat: improve UI narrative and interaction flow (Milestone 5)

### 14-01-2026 — Streamlit rerun behaviour and unintended recomputation

**Goal:**  
Prevent unintended recomputation of scenarios and inconsistent outputs caused by Streamlit reruns.

**Work done:**  
- Investigated why scenario outputs were changing when interacting with unrelated UI elements.  
- Traced issue to Streamlit’s automatic script rerun behaviour.  
- Identified that scenario generation was being triggered implicitly.

**Issues encountered:**  
- Scenario results changed when adjusting sliders or interacting with the UI.  
- Comparison outputs became unreliable due to regenerated values.

**Decisions / Fix:**  
- Introduced explicit “Save Scenario A/B” actions.  
- Moved scenario data into `st.session_state` to prevent accidental overwrites.  
- Ensured that comparison only uses stored scenarios, not freshly computed ones.

**Evidence:**  
- Scenario outputs remain stable across UI interactions.  
- Commit: fix: prevent scenario recomputation via session state


### 14-01-2026 — Shift from deterministic modelling to stochastic simulation

**Goal:**  
Improve realism of financial projections by accounting for uncertainty in variable expenses.

**Work done:**  
- Replaced initial deterministic scenario approach with Monte Carlo–style simulation.  
- Introduced Gaussian noise to model month-to-month variability.  
- Added percentile aggregation (10–90%) to represent uncertainty bands.

**Issues encountered:**  
- Deterministic projections gave a false sense of precision.  
- Early scenario outputs looked identical across months, reducing interpretive value.

**Decisions / Fix:**  
- Rejected purely deterministic modelling in favour of lightweight stochastic simulation.  
- Accepted reduced mathematical sophistication in exchange for interpretability and stability.

**Evidence:**  
- Uncertainty bands visible in scenario plots.  
- Commit: refactor: replace deterministic scenarios with Monte Carlo simulation


### 14-01-2026 — Scope control and feature rejection

**Goal:**  
Maintain focus on short-term clarity and behavioural reasoning within MVP constraints.

**Work done:**  
- Considered adding database persistence and long-term forecasting.  
- Evaluated complexity vs value for early-stage users.

**Issues encountered:**  
- Risk of over-engineering the system and diluting behavioural focus.  
- Increased technical complexity without clear benefit to core objective.

**Decisions / Fix:**  
- Rejected database integration to keep the system lightweight and session-based.  
- Limited projection horizon to 12 months to preserve interpretability.  
- Deferred complex financial modelling (e.g. interest, debt dynamics) to future work.

**Evidence:**  
- Architecture remains modular and lightweight.  
- Design decisions documented in portfolio.

