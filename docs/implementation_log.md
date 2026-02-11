# Implementation Log — LifeBudget Micro

---

## 14-01-2026 — Milestones 0–5: From baseline prototype to behavioural decision tool

**Goal:**  
Design and implement a runnable MVP of *LifeBudget Micro*, evolving from a deterministic financial baseline into a behavioural, uncertainty-aware decision-support tool.

### Work done

**Baseline prototype (Milestone 0)**  
- Set up project structure (`app.py`, `src/baseline.py`, `requirements.txt`).  
- Implemented a deterministic baseline projection computing cumulative balance over a short horizon.  
- Built initial Streamlit UI for income and expense inputs with table + balance chart.

**Scenario simulation & uncertainty (Milestones 1–2)**  
- Introduced *Compounder+*: Monte Carlo–style simulation for variable expenses.  
- Added Gaussian noise to model real-world spending variability.  
- Implemented percentile aggregation (10–90%) to visualise uncertainty bands.  
- Added controls for variability, iterations, and random seed (reproducibility).  
- Introduced additional savings parameter (later refined semantically).  
- Added `.gitignore` for repository hygiene.

**Scenario comparison (Milestone 3)**  
- Implemented Scenario A and Scenario B stored independently in `st.session_state`.  
- Built comparison view: Baseline vs A vs B (mean trajectories + uncertainty bands).  
- Added summary table of final balances and ranges.  
- Implemented logic to identify the best-performing scenario.

**Explainability layer (Milestone 4)**  
- Implemented a rule-based explainability module (`src/explain.py`).  
- Generated human-readable explanations answering *why* scenarios differ, not just *what*.  
- Identified dominant behavioural drivers (spending reduction, compounding, uncertainty).  
- Integrated explanations directly into the UI below the comparison view.

**UX and narrative layer (Milestone 5)**  
- Reframed the app into a step-based flow:
  1. Current position  
  2. Explore behavioural change  
  3. Compare outcomes  
  4. Reflect on impact  
- Added microcopy, captions, and helper text to reduce cognitive load.  
- Implemented progressive disclosure (scenarios only visible after baseline).  
- Aligned interface language with behavioural decision-making rather than financial jargon.

### Issues encountered
- Deterministic projections gave a false sense of precision.  
- Streamlit reruns caused unintended recomputation and inconsistent outputs.  
- Early UI lacked guidance, risking misinterpretation by non-technical users.

### Decisions / Fixes
- Rejected purely deterministic modelling in favour of lightweight stochastic simulation.  
- Introduced explicit “Save Scenario A/B” actions and session-state persistence.  
- Prioritised interpretability and behavioural clarity over model sophistication.

### Evidence
- Streamlit UI shows baseline, scenarios, uncertainty bands, and explanations.  
- Commits:
  - `feat: initial streamlit app with deterministic baseline`
  - `feat: add compounder scenario simulation`
  - `feat: add baseline vs scenario A/B comparison`
  - `feat: add explainability layer`
  - `feat: improve UI narrative and interaction flow`

---

## 15-01-2026 — Technical documentation and semantic stabilisation

**Goal:**  
Consolidate system documentation and correct semantic inconsistencies before further iteration.

### Work done
- Produced a technical portfolio document describing architecture, flow, and limitations.  
- Identified and fixed a mismatch between UI variability (%) and internal fractional logic.  
- Standardised variability handling across UI, simulation, and explainability layers.  
- Refactored scenario semantics to ensure A/B differences reflect behavioural intent.

### Decisions / Fixes
- Preserved percentage-based UI inputs for usability, converting internally for correctness.  
- Rejected seed-only differentiation between scenarios as conceptually weak.

### Evidence
- Updated `compounder.py`, `explain.py`, and `app.py`.  
- Technical portfolio document created.

---

## 03-02-2026 — System-wide refactor: Monthly → Weekly temporal granularity

**Goal:**  
Improve short-term interpretability by aligning the system with weekly budgeting behaviour.

### Work done
- Refactored baseline, simulation, explainability, and UI to operate fully on weekly units.  
- Updated charts, tables, labels, and explanations to use week-based semantics.  
- Preserved behavioural logic while changing temporal resolution.

### Decisions / Fixes
- Chose a full system-wide refactor to avoid mixed-unit ambiguity.  
- Treated temporal granularity as a UX decision, not just a technical one.

### Evidence
- Updated modules: `baseline.py`, `compounder.py`, `explain.py`, `app.py`.  
- Weekly projections visible across the UI.

---

## 04-02-2026 — UX refinement: Presets, advanced options, and scenario workflow clarity

**Goal:**  
Make Step 2 usable for non-technical users while preserving reproducibility and traceability.

### Work done
- Introduced preset-driven conceptual controls (predictability, result detail).  
- Added an **Advanced options** expander with optional technical overrides.  
- Implemented controlled randomness via “Try another random run”.  
- Added persistent per-scenario save feedback (Scenario A / B saved).  
- Implemented a robust Clear A/B action to reset state cleanly.  
- Cleaned Step 1 framing, legend labels, and scale notes without adding redundant charts.

### Decisions / Fixes
- Prioritised mental models over exposing Monte Carlo jargon.  
- Used state-driven feedback instead of ephemeral notifications.  
- Applied progressive disclosure to reduce launch-time clutter.

### Evidence
- Step 2 shows presets + expander-based advanced options.  
- Scenario save confirmations persist across reruns.  
- Changes primarily in `app.py`.

---

## 05-02-2026 — Behavioural semantics, guardrails, and explainability refactor

**Goal:**  
Align model semantics, UI language, and explanation logic around *targets vs real cuts* and stabilise workflow under Streamlit reruns.

### Work done
- Clarified that “savings targets” are user intent, but simulation applies **real discretionary cuts**.  
- Implemented a savings-target resolver (margin → discretionary → uncovered).  
- Blocked infeasible scenarios (essentials untouched).  
- Rewrote Step 4 as a **human-first reflection**, hiding technical details in an expander.  
- Modularised Step 4 logic into `src/explain.py`.  
- Added prerequisite gating for Steps 3–4 to prevent invalid states.

### Decisions / Fixes
- Preserved backwards compatibility while correcting semantics internally.  
- Enforced separation of concerns: UI orchestration vs reasoning logic.  
- Treated Step 4 as product output, not developer commentary.

### Evidence
- Step 4 shows plain-English summaries with optional technical detail.  
- `src/explain.py` contains all reflection logic.  
- Suggested commits:
  - `refactor: move Step 4 explanation logic into src/explain.py`
  - `refactor: align savings target semantics with discretionary cut`

---

## 05-02-2026 — Recovery & Git hygiene incident

**Goal:**  
Recover a working state after an accidental stash workflow and harden repository practices.

### Work done
- Recovered stashed files and recreated untracked helpers.  
- Properly committed `src/expenses.py` (weekly totals + one-off events).  

### Decisions / Fixes
- Ensured critical helpers are always tracked before stashing.  
- Committed recovered files immediately to avoid repeat loss.

### Evidence
- `src/expenses.py` committed (205 insertions).  
- Repository returned to stable state.

---

## 06-02-2026 — Stabilisation checkpoint: multi-event shocks and Step 4 alignment

### Goal
Stabilise the codebase after recent refactors and prepare a clean Git checkpoint, ensuring conceptual and behavioural consistency across baseline, scenarios, and reflection.

---

### Work done
- Fully integrated **multi-event one-off shocks** using a unified `shock_map` model.
- Applied one-off shocks **consistently** to:
  - Baseline trajectory
  - Scenario A
  - Scenario B
- Removed legacy single-event logic (`shock_amount`, `shock_week`) and related stale imports.
- Updated **Step 4 (Reflection)** to align with multi-event semantics, deprecating legacy shock arguments.
- Ensured compatibility during refactors via defensive function signatures.

---

### Decisions / Fixes
- Standardised `shock_map` as the **only authoritative representation** of one-off events.
- Treated shocks as **exogenous events** applied post-simulation, identically across baseline and scenarios.
- Maintained backward compatibility temporarily to avoid hard breaks during refactor.
- Identified a non-blocking Streamlit widget warning related to duplicated `weeks` initialisation.
- Deferred cosmetic Streamlit warning cleanup to a later polish pass to avoid destabilising the current checkpoint.

---

### Evidence
- Application runs end-to-end (Steps 1–4) with:
  - Baseline generation
  - Scenario A/B simulation
  - Multi-event shocks
  - Human-readable reflection
- Behaviour verified visually and logically:
  - Baseline and scenarios respond consistently to shocks.
  - Reflection text reflects the actual model state (no legacy assumptions).
- Modified files ready for clean commit:
  - `app.py`
  - `src/expenses.py`
  - `src/explain.py`
  - `docs/implementation_log.md`

---

## 08-02-2026 — Semantic consolidation, state hygiene, and Option A finalisation

**Goal:**  
Finalise behavioural semantics for scenarios, remove transitional logic, and harden application state management after rapid iteration (06–08 Feb).

---

### Work done

**Scenario semantics finalisation (Option A)**  
- Fully consolidated *Option A* semantics:
  - Baseline automatically saves any **positive margin**.
  - Scenarios A/B represent **extra weekly savings** achieved exclusively via **discretionary cuts**.
  - Essentials remain untouched by design.
- Removed the transitional *savings target resolver* logic after confirming it was no longer needed under Option A.
- Ensured simulation inputs (`delta_savings`) always represent **real behavioural cuts**, not abstract targets.

**State hygiene & Streamlit robustness**  
- Cleaned up session-state transitions to avoid stale or misleading state:
  - `confirm_position()` now:
    - Confirms the Step 1 snapshot.
    - Invalidates `baseline_ready`.
    - Clears `baseline_df`.
    - Clears Scenario A/B and related UI flags.
    - Resets `baseline_signature` to prevent stale comparisons.
- Added defensive guards to `make_baseline_signature()` to ensure it is only called once weekly-normalised values exist.
- Ensured baseline freshness checks rely on **both**:
  - `baseline_df` existence, and
  - signature equality against current inputs.

**Scenario save logic stabilisation**  
- Unified Scenario A and B save logic:
  - Consistent parameter sets stored for both scenarios.
  - Explicit semantic traceability fields added:
    - `baseline_margin`
    - `extra_savings`
    - `discretionary_cut`
    - `target_savings` (for downstream compatibility)
    - `from_margin`, `from_discretionary`, `uncovered`
- Fixed a subtle bug where Scenario B metadata mistakenly wrote into `params_a`.

**Explainability & reflection alignment**  
- Confirmed Step 4 (Reflection) consumes **only semantic truth**, not UI artefacts:
  - Weekly saving intent = margin + discretionary cut.
  - Reflection text now aligns exactly with the simulation model.
- Ensured technical explanation (`build_explanation`) uses **cuts**, not targets, as deltas.

---

### Decisions / Fixes

- Chose to **delete** legacy and transitional logic rather than keep unused abstractions:
  - Reduced cognitive load for future refactors.
  - Improved auditability for marking.
- Treated Step 1 confirmation as a **logical snapshot boundary**, not a cosmetic action.
- Prioritised explicit state invalidation over implicit reruns to avoid Streamlit edge cases.
- Accepted minor duplication (e.g. repeated guards) in favour of clarity and debuggability.

---

### Evidence

- Application runs end-to-end with clean semantics:
  - Step 1 snapshot → Step 2 baseline → Step 2 scenarios → Step 3 comparison → Step 4 reflection.
- No stale state persists across:
  - Position changes
  - Preset changes
  - Random seed rerolls
- Key files modified:
  - `app.py`
  - `src/explain.py`
- Suggested commit:
  - `refactor: finalise Option A semantics and clean scenario state handling`


---

## 11-02-2026 — Structural Deficit Intelligence, Dynamic Leverage Analysis & UI Modular Consolidation

### Goal
Transform structural deficit handling from a generic warning into a data-driven, behaviour-aware diagnostic system integrated coherently across Step 2 and Step 4.

---

## Work Done

### Structural Deficit Architecture Redesign

- Introduced a single **breakdown snapshot (weekly-normalised)** as the authoritative state object:

  - `income_w`
  - `fixed_total_w`
  - `fixed_w`
  - `var_w`
  - `disc_w`
  - `margin_w`

- Eliminated duplicated breakdown constructions across Step 2.
- Standardised deficit detection: structural_deficit = (margin_w + disc_w) < 0

---

## Intelligent Structural Deficit Tips Engine (`src/explain.py`)

### Core Upgrade

- Replaced generic advisory text with **data-driven leverage diagnostics**.
- Implemented dynamic ranking of:
  - Fixed expense items
  - Variable essential items (normalised weekly)

---

## Introduced Multi-Level Behavioural Insight

- % of deficit covered by largest item
- Whether single-item reduction could eliminate shortfall
- Top-two combined leverage analysis
- Quantified break-even improvement requirement

---

## Psychologically Clear Leverage Framing

- “X equals Y% of the deficit”
- “Reducing this item by £Z would eliminate the shortfall”

This framing shifts the output from abstract budgeting advice to **actionable behavioural leverage**.

---

## Contextual Comparison Logic

- Essentials % of income compared against reference ranges (50–65%)
- Provides interpretive context without enforcing rigid rules

---

## Financial Safety

- Explicit disclaimer that tips are not personalised financial advice

---

# Variable Essentials Normalisation

- Implemented clean weekly normalisation pipeline
- Extracted `_weekly` keys from `variable_essentials_weekly_total()`
- Renamed labels via mapping for human-readable output  
  (e.g., `"Commute Weekly"` → `"Commuting"`)

### Persisted in Session State:
st.session_state["variable_items_rows_weekly"]
st.session_state["variable_top_driver"]

## Leverage Prioritisation & UI Consolidation

### Leverage Ordering

- Sorted expense items by magnitude
- Prioritised highest-impact items first
- Ensured behavioural focus on largest financial drivers

This guarantees that the most structurally relevant items appear at the top of the diagnostic output.

---

# UI Integration (Step 2)

Consolidated structural deficit display into a single expander:

with st.expander("Tips to fix this structural deficit...", expanded=True):
    st.markdown(build_structural_deficit_tips(...))

---

## Improvements

- Removed duplicated expanders
- Removed redundant function calls
- Simplified conditional rendering logic
- Reduced UI complexity in Step 2
- Improved structural clarity and maintainability

---

# Separation of Concerns

- `app.py` → Data orchestration & session-state management  
- `explain.py` → Behavioural reasoning engine  

This enforces a clean architectural boundary between:
- State handling
- Financial computation
- Behavioural interpretation

---

# Step 4 Compatibility

- Confirmed `build_human_reflection_text()` can optionally consume the same breakdown object
- Maintained modular design to allow:

  - Automatic reinforcement of deficit advice in Step 4
  - Future behavioural layering without architectural changes
  - Reuse of the same leverage intelligence across UI layers

---

# Decisions / Fixes

- Rejected static or generic advice
- Implemented fully dynamic item-based leverage modelling
- Avoided passing fragmented parameters (e.g., `variable_items_weekly=...`)
- Standardised to a single source-of-truth breakdown dictionary
- Removed redundant conditional blocks and duplicated expanders
- Replaced ambiguous leverage phrasing (“multiple of deficit”) with clearer percentage framing
- Prioritised behavioural clarity over financial complexity

---

# Behavioural Design Upgrade

The application now explicitly distinguishes between:

## Recoverable Deficit  
Can be fixed via discretionary cuts.

## Structural Deficit  
Requires income changes or essential-cost restructuring.

---

## Structural Deficit Output Now Includes

- Ranked drivers (top 1–3 weekly items)
- % of deficit covered by largest item
- Top-two combined leverage analysis
- Quantified break-even requirement
- Behavioural guidance framed as leverage, not blame

---

# Conceptual Shift

From:

> “Budget simulator”

To:

> “Behaviour-aware financial decision support tool”

---

# Evidence

## Step 2

- Structural deficit automatically triggers dynamic leverage analysis
- No duplicated expanders
- Tips update instantly based on user input

## `src/explain.py`

- Fully modular `build_structural_deficit_tips()`
- Robust leverage logic (single + top-two analysis)
- Defensive ratio guards for edge cases

---

# State Stability Verified

- No circular imports
- No `unexpected keyword argument` errors
- Clean session-state dependency chain
