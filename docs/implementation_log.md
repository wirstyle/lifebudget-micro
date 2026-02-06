### 14-01-2026 – Milestone 0: Initial prototype and baseline implementation

**Goal:**  
Set up project environment and implement a first runnable version of the LifeBudget Micro prototype with a deterministic baseline projection.

**Work done:**  
- Created project structure (app.py, requirements.txt, src/baseline.py).
- Implemented generate\_baseline() function to calculate monthly savings and cumulative balance over a short-term horizon (later refined to weekly).


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
- Introduced an "additional savings" parameter (initially monthly, later refined to weekly).

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

### 15-01-2026 — Milestone 7: Technical portfolio documentation

**Goal:**  
Produce technical documentation explaining the implemented system for other developers.

**Work done:**  
- Wrote system overview, architecture, flow, and design decisions.  
- Documented implementation details and execution steps.  
- Explicitly described limitations and scope constraints.

**Decisions / Fix:**  
- Chose Word/PDF format over Markdown to align with academic assessment expectations.  
- Structured documentation around reasoning flow rather than module listing.

**Evidence:**  
- LifeBudget_Micro_Technical_Portfolio.docx created.

### 15-01-2026 — Parameter Semantics Fix: Variability Consistency (Compounder+)

**Goal:**  
Ensure semantic and numerical consistency between UI controls, simulation logic, and explanation layer for spending variability.

**Work done:**  
- Identified inconsistency between percentage-based UI input and fractional interpretation in simulation logic.
- Standardised variability handling by converting UI percentage input to fractional form before simulation.
- Renamed internal parameter to variability_frac to reflect correct semantics.
- Updated simulation docstrings and defaults for clarity.
- Integrated consistent parameter usage into the explainability layer.
- Added defensive clamping to prevent negative effective variable spending.

**Decisions / Fix:**  
- Chose to keep UI in percentage form for user comprehension and convert internally to fraction for correct mathematical behaviour.
- Prioritised interpretability and numerical stability over adding additional model complexity.
- Ensured traceability of scenario assumptions in the explanation layer.

**Evidence:**  
- Updated compounder.py, app.py, and explain.py with aligned parameter semantics.
- Verified correct behaviour via reproducible runs using fixed random seed.

### 15-01-2026 — Scenario Design Refactor: Behavioural Differentiation for A/B Comparison

**Goal:**  
Ensure that Scenario A and Scenario B represent genuinely different behavioural assumptions rather than superficial stochastic variation.

**Work done:**  
- Introduced explicit delta_a and delta_b parameters to represent different savings behaviours (initially monthly, later refined to weekly).
- Updated UI flow to allow separate behavioural inputs for each scenario.
- Ensured scenario storage preserves distinct behavioural parameters.
- Updated explanation layer to reference behavioural differences directly.

**Decisions / Fix:**  
- Rejected seed-only differentiation as conceptually weak and potentially misleading.
- Prioritised behavioural interpretability over minimal code change.
- Ensured A/B comparison reflects meaningful user decisions rather than random variation.

**Evidence:**  
- Updated app.py scenario saving logic.
- Updated ExplanationInputs to include delta_a and delta_b.
- Verified explanation output reflects correct behavioural assumptions.

### 15-01-2026 — Explainability Layer Refactor: Driver Logic and Assumption Traceability

**Goal:**  
Improve technical correctness and interpretability of the explanation layer.

**Work done:**  
- Added explicit assumption traceability (“A saves £X/month, B saves £Y/month”).
- Replaced heuristic surplus driver logic with case-based behavioural reasoning.
- Added differentiation between equal, unequal, and zero behavioural changes.
- Integrated uncertainty band comparison for technical clarity.
- Ensured explanation text aligns with actual simulation parameters.

**Decisions / Fix:**  
- Chose explicit reasoning branches over heuristic shortcuts to avoid misleading interpretations.
- Prioritised technical accuracy over brevity in explanation generation.
- Ensured explanation logic is robust to edge cases (e.g. equal deltas, zero deltas).

**Evidence:**  
- Refactored build_explanation() in explain.py.
- Tested explanation output across multiple scenario configurations.

### 03-02-2026 — Refactor: Temporal Granularity Shift from Monthly to Weekly

**Goal:**
Improve short-term interpretability by aligning the system’s temporal resolution with weekly budgeting behaviour.

**Work done:**
- Refactored the deterministic baseline projection (BudgetMind) from monthly to weekly calculations.
- Updated the projection horizon to operate over weeks rather than months.
- Refactored the Compounder+ Monte Carlo simulation to use weekly time steps and week-indexed outputs.
- Updated the explainability layer to reflect weekly semantics (weeks, £/week, final week, weekly surplus).
- Updated the Streamlit UI to accept weekly inputs (income, fixed expenses, variable expenses) and use a weekly projection slider.
- Updated tables, charts, and comparison views to use Week-based indexing and labels.
- Updated uncertainty insight text to refer to compounding weekly variability.

**Decisions / Fix:**
- Chose a full system-wide weekly refactor rather than internal monthly-to-weekly conversion to avoid mixed-unit semantics.
- Preserved the original model structure and behavioural assumptions, changing only the temporal granularity to maintain MVP scope and conceptual clarity.
- Prioritised interpretability and consistency across UI, simulation logic, and explanation text.

**Evidence:**
- Updated modules: src/baseline.py, src/compounder.py, src/explain.py, app.py.
- Streamlit UI displays baseline and scenarios using weekly tables, charts, and interpretive text.
- Commit: refactor: switch projections from monthly to weekly across system.

### 03-02-2026 — Note: Historical consistency after weekly refactor

**Note:**  
Earlier log entries describe the initial monthly-based implementation. The system has since been refactored to operate fully on a weekly basis. Previous milestones remain historically accurate and document the evolution of the prototype.

### 03-02-2026 — Design Insight: Temporal granularity as a UX decision

**Goal:**  
Reflect on the impact of temporal resolution (monthly vs weekly) as a design decision rather than a purely technical choice.

**Work done:**  
- Evaluated how users reason about everyday financial decisions.
- Identified that weekly framing improved immediacy and behavioural interpretability.

**Decisions / Fix:**  
- Treated temporal granularity as part of the interaction design, not just a modelling parameter.
- Prioritised user reasoning and clarity over conventional monthly budgeting conventions.

**Evidence:**  
- Weekly-based UI, simulation, and explanation layer.
- Reflected in Evaluation and Technical Portfolio documentation.

### 04-02-2026 — UX Refinement: Baseline table cleanup, user-friendly uncertainty controls, and persistent scenario save feedback

**Goal:**  
Reduce confusion for non-technical users and improve interaction clarity in the scenario workflow, while keeping outputs reproducible and comparisons consistent.

**Work done:**  
- **Baseline table cleanup:** Hid the implicit DataFrame index column in the baseline projection table to avoid confusing users with an extra unlabeled column.  
- **User-friendly uncertainty controls:** Replaced technical parameters (Monte Carlo iterations, spending variability %, random seed) with:
  - A **Preset selector** (one-click configurations aligned to conceptual settings),
  - Two **conceptual controls**:
    - **Expense predictability** (Stable / Typical / Chaotic) → mapped to spending variability (%),
    - **Result detail** (Fast / Balanced / High confidence) → mapped to number of simulation iterations,
  - A **“Try another random run”** button that increments an internal random seed, enabling controlled re-draws without exposing technical jargon.
- **Preset–concept alignment:** Redefined presets so they are **fully consistent** with the two conceptual controls, ensuring no hidden or contradictory parameter values.
- **Scenario workflow stability:** Preserved explicit **Save Scenario A/B** actions and stored scenarios in `st.session_state` to prevent unintended recomputation caused by Streamlit reruns.
- **Persistent save feedback (new):** Implemented **two independent green success messages** (“Scenario A saved” and “Scenario B saved”) that remain visible after saving and **do not overwrite each other**. Messages are cleared only when the user presses **Clear A/B** (and when regenerating the baseline to avoid inconsistent state carry-over).
- **Widget stability:** Added explicit widget keys where required to reduce UI state collisions during Streamlit reruns.

**Issues encountered:**  
- Streamlit reruns caused transient success messages to disappear when interacting with other controls (e.g., saving Scenario B removed Scenario A’s message).
- Non-technical users struggled with simulation jargon and misinterpreted “random seed” as a parameter they needed to understand or manually adjust.

**Decisions / Fix:**  
- Prioritised **conceptual mental models** over technical exposure: users interact with predictability and result detail rather than raw Monte Carlo parameters.
- Kept randomness **reproducible by default** via a fixed internal seed, while providing an explicit re-draw action for exploratory analysis.
- Chose **persistent, state-driven feedback** over ephemeral notifications to support a clearer and more reassuring scenario-saving workflow.

**Evidence:**  
- Updated Streamlit UI showing presets, conceptual controls, and re-draw functionality, with technical parameters mapped internally.
- Saving Scenario A and Scenario B produces persistent green confirmation messages until **Clear A/B** is pressed.
- Changes implemented primarily in `app.py`, including session state flags, preset–control mapping, and persistent banner logic.


### 04-02-2026 — UX Refinement: Preset-driven controls, Advanced Options expander, and persistent scenario save feedback

**Goal:**  
Improve Step 2 usability and conceptual clarity by hiding technical jargon behind presets and an optional “Advanced options” section, while keeping scenarios reproducible and the save workflow reassuring.

**Work done:**  
- **Preset-driven conceptual controls (auto-sync):**  
  - Implemented a preset selector where choosing a preset automatically updates the conceptual radio controls:  
    - **How predictable are your weekly expenses?** (Stable / Typical / Chaotic)  
    - **Result detail** (Fast / Balanced / High confidence)  
  - Achieved via `st.session_state` widget-key control + `on_change` callback to directly set radio values.

- **Advanced Options (collapsible expander):**  
  - Added a hideable section using `st.expander("Advanced options")` (the “>” style control).  
  - Placed the **conceptual radios at the top of the expander** (as requested) so users see the mental model first, then the technical controls.

- **Technical sliders restored (but optional):**  
  - Reintroduced the previously-removed technical parameters inside Advanced Options:  
    - **Monte Carlo simulations (iterations)**  
    - **Weekly spending variability (%)**  
    - **Random seed**  
  - Added an “Enable technical overrides” checkbox so these controls only override behaviour when explicitly enabled.
  - When overrides are disabled, sliders stay **synchronised** with the conceptual controls (coherence by default).

- **Randomness UX maintained:**  
  - Kept “Try another random run” (seed increment) for controlled redraws, while preserving reproducibility for A vs B comparisons.  
  - Ensured advanced seed remains aligned unless overrides are enabled.

- **Persistent Save feedback (green banners):**  
  - Implemented two independent success messages (“Scenario A saved.” / “Scenario B saved.”) stored in `st.session_state`.  
  - Messages **do not overwrite each other** and remain visible until the user presses **Clear A/B**, which clears both banners and both saved scenarios.

- **Traceability improvements:**  
  - Stored extra metadata inside scenario params (preset name, conceptual selections, advanced override flag) for easier examiner traceability and debugging.

**Issues encountered:**  
- Preset selection did not automatically update the radio widgets due to Streamlit rerun/state behaviour.  
- Technical controls created coherence risk (users could unknowingly diverge from the conceptual model).  
- Success banners were ephemeral and disappeared after subsequent interactions.

**Decisions / Fix:**  
- Used widget-key–driven state (`predictability_radio`, `detail_radio`) and a preset `on_change` callback to guarantee automatic radio updates.  
- Made advanced controls opt-in via “Enable technical overrides” to prevent silent contradictions with the conceptual model.  
- Chose persistent, state-driven success banners cleared only by **Clear A/B** to support a stable and reassuring scenario workflow.

**Evidence:**  
- Step 2 shows presets + randomness button, with Advanced Options collapsible section containing conceptual radios and optional technical sliders.  
- Selecting a preset auto-selects the correct radio options.  
- Saving A and B produces two persistent green success messages that only disappear after pressing Clear A/B.  
- Primary changes in `app.py` (session state flags, preset callbacks, expander + override logic).

### 04-02-2026 — UX Refinement: Persistent scenario save feedback and Clear A/B behaviour

**Goal:**  
Improve clarity and consistency of the scenario-saving workflow by providing stable, non-intrusive feedback that reflects the actual state of saved scenarios.

**Work done:**  
- Refactored Scenario A/B save feedback to use **persistent, state-driven visual badges** instead of transient Streamlit messages.
- Implemented **per-scenario confirmation badges** (“Scenario A saved”, “Scenario B saved”) displayed:
  - Directly **under each corresponding Save button**,
  - With **full column width**, visually aligned with the buttons.
- Ensured that saving one scenario **does not overwrite or remove** the confirmation message of the other.
- Introduced a dedicated **Clear A/B callback** that:
  - Clears both stored scenarios from `st.session_state`,
  - Explicitly resets both save confirmation flags,
  - Guarantees that all green confirmation messages disappear **in the same interaction**.
- Added an optional, non-persistent **“Scenario A and B cleared”** informational message to confirm reset action without cluttering the interface.

**Issues encountered:**  
- Streamlit’s rerun behaviour caused success messages to reappear or stack inconsistently across interactions.
- Default `st.success()` messages were unsuitable for persistent, per-button feedback.
- Clearing scenarios did not reliably remove previously rendered confirmation messages.

**Decisions / Fix:**  
- Replaced ephemeral Streamlit alerts with **custom HTML badges** rendered conditionally from `st.session_state`.
- Centralised scenario clearing logic in a single `clear_ab()` callback to ensure atomic state reset.
- Treated save confirmations as **state indicators**, not notifications, aligning UI feedback with actual system state.
- Prioritised visual consistency and spatial proximity (message appears directly under the triggering control).

**Evidence:**  
- Streamlit UI shows stable “Scenario A saved” and “Scenario B saved” badges that persist across reruns.
- Pressing **Clear A/B** removes both badges immediately and clears stored scenarios.
- Changes implemented in `app.py` within the Scenario Save / Clear section.
- Commit: `feat: persistent per-scenario save feedback and reliable Clear A/B reset`

### 04-02-2026 — UX Refinement: Step 1 clarity (framing + numeric legend labels + scale note without extra charts)

**Goal:**  
Increase interpretability of Step 1 (“current position”) without contaminating it with projection meaning, while making the allocation output more readable and self-explanatory.

**Work done:**  
- **Framing microcopy (2 lines):** Added a short caption to explicitly state that Step 1 captures the current situation and that **no savings / behavioural changes** are applied yet.  
- **Clean CTA (no pressure):** Added a simple transition line at the end of Step 1 (“Next: explore what happens if you change something (Step 2)”) to guide progression without persuasion or gamification.  
- **Numeric labels in the stacked bar legend:** Updated legend labels to include **weekly amounts** alongside category names (e.g., “Fixed essential (£185/w)”), improving immediate readability without introducing additional tables or projections.  
- **Scale clarification without duplicate charts:** Rejected monthly/yearly duplicate allocation charts (same proportions, redundant visuals). Instead, added a **structured note** that provides monthly and yearly numeric equivalents while keeping Step 1 visually minimal.  
- **Structured presentation for clarity:** Rendered the scale note using an indented bullet layout (Markdown lists) to reduce visual chaos and improve scanability.

**Issues encountered:**  
- The stacked bar alone communicated proportions well, but users still needed quick access to **actual numeric values** per category.  
- Adding monthly/yearly charts would create redundant visuals and increase cognitive load without adding new insight (same proportional structure).

**Decisions / Fix:**  
- Kept Step 1 deliberately “diagnostic” and non-temporal; avoided projections, tables, or timelines in this stage.  
- Chose **legend numbers + a short structured scale note** as the most defensible solution: it increases clarity while preserving Step 1’s conceptual boundary.  
- Used Markdown (not caption-only text) to support clean indentation and a more “report-like” structure.

**Evidence:**  
- Step 1 now displays:  
  - Input fields → stacked allocation bar → legend with £/w values → structured monthly/yearly equivalents → CTA to Step 2.  
- Implementation updated in `app.py` (Step 1 section): legend label formatting, scale note generation, and microcopy/CTA additions.

### 05-02-2026 — UX + State Robustness: Step 1 progressive disclosure, safer ordering, and human-first Step 4

**Goal:**  
Reduce “dashboard noise” on launch, prevent confusing out-of-order states, and make the interpretation layer genuinely readable (decision-support, not technical narration).

**Work done:**  
- **Step 1 progressive disclosure (launch cleanliness):**  
  - Moved the **allocation chart + “Note on scale”** into a collapsible **expander** so the app opens like a cleaner “landing page” instead of a full dashboard.  
  - Renamed the expander label to a more human microcopy option (e.g., **“See breakdown and equivalents”** / **“Show detailed breakdown”**).

- **Workflow stability / order-of-operations guardrails:**  
  - Strengthened Step 3–4 gating so **comparison + explainability only render when prerequisites exist** (baseline up-to-date + both scenarios saved).  
  - Added lightweight schema checks and “baseline out of date” warnings to avoid errors when users click buttons in the “wrong” order.

- **Step 4 rewrite (human-first interpretation):**  
  - Replaced technical narrative (“uncertainty widens”, “seed offsets”, Monte Carlo jargon) with plain-English statements like:  
    - “If you spend £X/week less, you end with about £Y more after Z weeks.”  
  - Kept technical details (seed, iterations, variability, band widening) inside a **collapsed expander** for assessment traceability without forcing it onto users.

**Issues encountered:**  
- **Streamlit rerun/state behaviour** caused confusing intermediate states and warnings when users interacted out of sequence.  
- Step 4 text was “technically correct” but **not interpretable** for non-technical users, undermining the purpose of a decision-support tool.  
- Step 1 displayed chart + scale notes immediately on load, creating **visual clutter** before the user had context.

**Decisions / Fix:**  
- Prioritised **progressive disclosure**: show essential inputs first, hide detail until requested.  
- Treated Step 4 as **product output**, not developer commentary: user sees decisions and consequences; technical explanation becomes optional evidence.  
- Enforced prerequisite checks to keep the interface stable under reruns and to reduce user confusion during exploration.

**Evidence:**  
- Step 1 launches cleanly; allocation breakdown is available via expander (“See breakdown and equivalents”).  
- Step 3–4 no longer throw confusing states when used out of order; warnings guide the user to regenerate baseline or save missing scenarios.  
- Step 4 displays a short, human-readable summary; technical details are accessible only via an optional expander.

- Commits (suggested labels):  
  - `feat: progressive disclosure for Step 1 breakdown + scale note`  
  - `fix: enforce Step 3/4 prerequisites + schema guardrails`  
  - `refactor: rewrite Step 4 interpretation to human-first + hide technical details`

### 05-02-2026 — Refactor + Modularisation: Step 4 explainability moved into `src/explain.py`

**Goal:**  
Reduce complexity in `app.py` by extracting Step 4 (“Reflect on impact”) into a dedicated explainability module, while keeping the Streamlit UI human-first and the technical rationale available for assessment traceability.

**Work done:**  
- **Modularised Step 4 explainability:**  
  - Introduced / expanded `src/explain.py` to hold all reasoning logic and explanation text generation.  
  - Added an `ExplanationInputs` dataclass to capture the full assumption context (income, expenses, weeks, deltas, variability, seed, iterations).  
- **Moved computation out of UI layer:**  
  - Removed Step 4 helper functions from `app.py` (e.g., currency formatting, percentage change, band-width calculations, winner/safer logic).  
  - Centralised comparison metrics (baseline vs A vs B) and uncertainty reasoning inside `build_explanation()`.  
- **Improved traceability and consistency:**  
  - Ensured the explanation explicitly references the actual scenario assumptions (A saves £X/week vs B saves £Y/week).  
  - Included uncertainty-band interpretation based on the 10–90% ranges at the final week.  
- **Streamlit integration preserved:**  
  - `app.py` now only prepares inputs + final-week values and calls `build_explanation()` to render explanation output.  
  - Kept the human-first narrative in the main Step 4 text, with technical explanation placed under a collapsible expander.

**Issues encountered:**  
- Step 4 had become the largest and most cluttered block in `app.py`, mixing UI rendering with business logic and increasing maintenance risk.  
- Early explanation variants risked drifting from true simulation parameters if assumptions were not passed explicitly.

**Decisions / Fix:**  
- Enforced a clean separation of concerns: **`app.py` orchestrates UI and state; `src/explain.py` owns explanation logic**.  
- Used a structured dataclass (`ExplanationInputs`) to prevent missing/implicit assumptions and guarantee reproducibility/traceability.  
- Accepted a small amount of duplication in displayed text (human summary + technical expander) to satisfy both usability and marking requirements.

**Evidence:**  
- `src/explain.py` contains the explanation engine (`ExplanationInputs`, `build_explanation()` plus internal helper functions).  
- `app.py` Step 4 is visibly shorter and now delegates reasoning to the module.  
- UI shows a plain-English reflection summary with an optional “Technical details” expander containing the generated explanation.  
- Suggested commit message: `refactor: move Step 4 explanation logic into src/explain.py`

### 05-02-2026 — Semantics Alignment: “Savings target” vs “discretionary cut” (Compounder+ + Reflection rewrite)

**Goal:**  
Resolve a conceptual mismatch where the UI described scenario inputs as “weekly savings”, while the simulation engine actually applies the parameter as a **real reduction (cut) to the controllable spending bucket**. Update the reflection layer to explain *targets*, *existing margin coverage*, and *required discretionary cuts*.

**Work done:**  
- **Compounder+ semantics clarification (`src/compounder.py`):**  
  - Updated the `simulate_scenario()` docstring to explicitly state that `delta_savings` is interpreted as a **REAL weekly spend reduction (“cut”)** applied to the controllable spending bucket.  
  - Added an internal alias `delta_cut = float(delta_savings)` before simulation loops to make the model intent explicit.  
  - Ensured the computation reads clearly: `effective_variable = variable_expenses - delta_cut + noise`, with defensive clamping at £0.

- **Human-first reflection upgrade (`src/explain.py`):**  
  - Extended `ReflectionMetrics` + `compute_reflection_metrics()` to carry explicit semantics fields:
    - `target_a_weekly`, `target_b_weekly` (user intent)  
    - `margin_a_weekly`, `margin_b_weekly` (covered by current surplus)  
    - `cut_a_weekly`, `cut_b_weekly` (required discretionary reduction)  
  - Rewrote `build_human_reflection_text()` to describe:
    - the **weekly target**,
    - how much is **covered by existing margin**,
    - and whether the plan **requires a discretionary cut** (or not).

- **UI orchestration (`app.py`):**  
  - Kept scenario simulation inputs compatible (no breaking imports).  
  - Ensured Step 4 passes targets + derived cut/margin values to the reflection layer while keeping the technical explanation available in the optional expander.

**Issues encountered:**  
- The parameter name `delta_savings` and earlier text implied “extra savings” even though the simulation operationally applies a **spend cut**.  
- Step 4 wording could mislead users into thinking savings always require cutting spending, even when current margin already covers the target.

**Decisions / Fix:**  
- Maintained the external parameter name `delta_savings` for backwards compatibility, but **reframed its meaning** explicitly in documentation and internal naming (`delta_cut`).  
- Treated the “savings” input as a **target** in the UX, and made the system compute whether it is:
  - **fully covered by existing margin**, or
  - **partly covered and requires a discretionary cut**.  
- Prioritised alignment of **model semantics ↔ UI mental model** over adding new features.

**Evidence:**  
- `src/compounder.py` now documents and implements `delta_savings` as a “cut” with `delta_cut` aliasing.  
- Step 4 output explicitly states “target”, “covered by margin”, and “requires cutting discretionary by £X/w” where applicable.  
- Suggested commit: `refactor: align savings target semantics with discretionary cut + upgrade human reflection text`

### 05-02-2026 — Step 2 bugfix + UI semantics: Savings target coverage vs discretionary cut (and breakdown clarity)

**Goal:**  
Fix confusing Scenario A/B behaviour when testing savings targets, and align the UI narrative with the actual simulation semantics (margin coverage first, then discretionary cuts), without changing the underlying Compounder+ model.

**Work done:**  
- Implemented a clear **savings target resolver** (`resolve_savings_target`) that splits a weekly target into:
  - amount covered by **remaining margin** (no sacrifice),
  - amount requiring a **discretionary cut** (behavioural change),
  - and any **uncovered shortfall** (would require cutting essentials, disallowed in MVP).
- Updated Step 2 to **block saving scenarios** when the target is not feasible under MVP rules (i.e., uncovered > 0).
- Improved Step 2 breakdown display to avoid misleading output by ensuring the UI explicitly communicates:
  - **target**, **covered by margin**, **required discretionary cut**, and **shortfall** (when present).
- Confirmed Compounder+ semantics: the simulation parameter (`delta_savings`) is operationally a **real cut** applied to the controllable spending bucket:
  - `effective_variable = variable_expenses - delta_savings + noise` (clamped at 0).

**Issues encountered:**  
- During testing, Scenario B appeared inconsistent: the UI could show:
  - “Covered by margin: £150/w” and “Requires discretionary cut: £0/w”
  - while simultaneously raising a shortfall error (e.g., “shortfall ≈ £50/w”).
- Root cause: the breakdown did not explicitly display the **uncovered** value, so users interpreted “£0 discretionary cut” as “fully feasible”.

**Decisions / Fix:**  
- Treated the UI input as a **savings target** (user intent), but passed only the **behavioural cut** component (`from_discretionary`) into simulation to preserve correct model semantics.
- Prioritised **breakdown transparency** over adding more controls: if discretionary available is £0, the UI must make that visible and show the shortfall explicitly.
- Maintained MVP rule: **essentials untouched**; infeasible targets are rejected with a clear message.

**Evidence:**  
- Step 2 UI shows policy line: “remaining margin → discretionary cut (essentials untouched)”, plus per-scenario breakdown.
- If a target exceeds (margin + discretionary), the UI flags the **shortfall** and prevents saving that scenario.
- Behaviour validated with test case where Scenario B triggers uncovered shortfall while showing margin coverage correctly.

