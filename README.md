# LifeBudget Micro

LifeBudget Micro is an educational Streamlit prototype for personal finance planning and investment scenario exploration. It helps users estimate savings capacity, build an educational investment strategy, run historical backtests, and compare long-term savings-only versus savings-plus-investing scenarios.

> **Educational notice:** This project is not financial advice, investment advice, or a personal suitability assessment. Investment outputs are historical backtests or educational proxies and do not guarantee future outcomes.


## Intended Users

LifeBudget Micro is designed for students, early-career users, and individual learners who want to understand how personal budgeting, saving capacity, historical investment testing, and long-term scenario planning can be connected in one educational workflow.

It is also intended for assessors and technical reviewers who need to verify a working end-to-end software prototype: what input is provided, what action the system performs, what output is produced, and which limitations affect interpretation.

The system is not designed for real financial advice, regulated suitability assessment, or production investment planning.

## Core Features

- Personal finance planner for budget baseline, savings capacity, and short-term feasibility checks.
- Risk profile and asset universe setup for preparing an educational market panel.
- Strategy Engine with walk-forward historical backtesting and risk/return metrics.
- Optional strategy suggestions, reliability checks, and robustness diagnostics.
- Long-term scenario explorer using Monte Carlo-style projections.
- Final report / insights summary connecting the full workflow.

## Running the App Online

The app can be accessed through the hosted Streamlit deployment:

https://lifebudget-micro.streamlit.app/

This is the recommended way to test the submitted prototype quickly. The online version may take longer during market panel generation or strategy backtesting because it runs on hosted Streamlit resources.

## Device Compatibility

LifeBudget Micro is a web-based Streamlit app and can be opened from a browser on a desktop, laptop, tablet, or mobile device.

The recommended testing environment is a desktop or laptop browser. The app includes charts, tables, expandable diagnostics, and sidebar navigation, which are easier to review on a larger screen.

Mobile browser access may work for quick inspection, but the prototype has primarily been tested and optimised for desktop/laptop use.

## Repository and Final Submission Snapshot

Repository link:

```text
https://github.com/wirstyle/lifebudget-micro
```

The hosted Streamlit app is deployed from the `deploy-lifbudget-demo` branch. The final assessment version should be labelled with a clear Git tag or GitHub release so the marker can identify the exact submitted snapshot. The suggested tag should point to the same commit used by the deployed `deploy-lifbudget-demo` branch.

```text
Deployed branch: deploy-lifbudget-demo
Final assessment tag: assignment-3-final
```

Example Git commands to create the final tag from the deployed branch:

```bash
git checkout deploy-lifbudget-demo
git pull origin deploy-lifbudget-demo
git tag -a assignment-3-final -m "Assignment 3 final submission snapshot"
git push origin assignment-3-final
```

If a different tag or release name is used, replace the suggested label above with the exact submitted version label. The important point is that the tag identifies the same commit used by the online deployed branch.

## Running the App Locally

```bash
git clone https://github.com/wirstyle/lifebudget-micro.git
cd lifebudget-micro
pip install -r requirements.txt
streamlit run app.py
```

If the app does not open automatically, copy the local Streamlit URL shown in the terminal into your browser.

## Recommended Quick Test Flow

1. Open the app online or run it locally.
2. Accept the educational notice on the home screen.
3. Open **Personal Finance Planner**.
4. Enter or keep sample weekly income, spending, and savings target values.
5. Review the cash-flow summary, free margin, savings target, feasibility check, and short-term path chart.
6. Continue to **Investment Strategy Lab**.
7. Select a risk profile and prepare the asset universe / market panel.
8. Run the Strategy Engine.
9. Review historical risk/return metrics, benchmark context, reliability diagnostics, and optional improvement checks.
10. Continue to **Long-Term Scenario Explorer**.
11. Generate a long-term projection using the available monthly contribution.
12. Open the final report / insights summary.


## Assessment Evidence Path

For assessment, the most important demonstration is the working end-to-end path rather than every optional diagnostic. The recommended evidence path is:

- **Input:** enter or keep sample weekly finance values and a savings target.
- **Action:** prepare the investment universe, run the Strategy Engine, and generate a long-term scenario.
- **Output/result:** review the budget feasibility output, historical strategy metrics, benchmark context, projection chart, and final report.
- **Limitations:** explain that the results are educational backtests, proxies, and scenarios rather than financial advice, forecasts, or guarantees.

This mirrors the expected build-phase demonstration: the marker can see the system running, follow the input-action-output chain, and understand the main limitations.

## User Manual Scope

The full PDF User Guide is structured as a user manual rather than only a click-by-click tutorial. It explains:

- what each module does;
- what the user can set;
- which inputs are required or optional;
- what outputs are produced;
- how to interpret the results;
- known limitations and caveats.

It documents the main user-relevant configuration routes without listing every possible internal diagnostic state.


## Common Interface Elements

The app uses a consistent sidebar and expandable help structure across modules:

- **Home button:** returns to the main module selection screen. It is disabled on the home screen because the user is already there.
- **Current state summaries:** show relevant saved information such as the finance plan, selected risk profile, prepared market panel, or latest strategy result.
- **Q&A expanders:** answer common user questions in plain language.
- **Terms expanders:** define key concepts used in the current module.
- **Diagnostics expanders:** provide technical or audit-style details such as panel status, timings, reliability checks, or engine information.
- **Assumptions and limitations expanders:** explain caveats behind educational simulations, backtests, and projections.

Colour-coded messages are also used consistently: blue for information, green for successful actions, and yellow for assumptions, caveats, or limitations.

## Strategy Engine Example Result

A representative Strategy Engine run in the guide uses the following historical backtest metrics for the current setup:

- CAGR: `9.73%`
- Volatility: `11.71%`
- Max drawdown: `-19.43%`
- Sharpe: `0.85`

The guide also explains optional improvement checks. In the example, a Phase 1 preset candidate (`Balanced Risk-Controlled + Defensive`) improves the displayed trade-off to approximately CAGR `9.93%`, volatility `11.61%`, MaxDD `-18.97%`, and Sharpe `0.88`. A Phase 2 engine-tuning candidate improves the example to approximately CAGR `10.00%`, volatility `11.57%`, MaxDD `-18.83%`, and Sharpe `0.89`. Later phases can apply a universe-mix change or keep the current universe size if no tested size alternative passes the acceptance gate.

The example reliability panel reports `Moderate-to-Strong` overall reliability and `Strong` start-date robustness. These are historical educational results only, not forecasts or recommendations.

## Long-Term Scenario Example

The User Guide also documents the Long-Term Scenario Explorer. In the example shown in the guide, the module uses the contribution bridge from Personal Finance Planner and the completed Strategy Engine return path. It compares the selected investment/proxy path with a savings-only baseline across horizons such as 1, 20, 30, and 50 years.

Example values from the guide include:

- Contribution bridge: approximately `£208/month` (`£48/week`)
- 1-year median final wealth: approximately `£2,660`
- 20-year median final wealth: approximately `£152,118`
- 30-year median final wealth: approximately `£450,517`
- 50-year median final wealth: approximately `£3,316,735`

These are scenario outputs under the selected assumptions. They are not forecasts, guarantees, or financial recommendations.

## Final Report Example

The final report summarises the personal finance plan, investment setup, tested Strategy Engine result, Long-Term Scenario comparison, and decision-support interpretation. In the example shown in the guide, the report highlights:

- Monthly contribution: approximately `£208/month`
- 50-year savings-only baseline: approximately `£124,800`
- 50-year investment/proxy median: approximately `£3,316,735`
- 50-year modelled difference: approximately `£3,191,935`
- Strategy snapshot: Sharpe `0.99`, CAGR `10.2%`, drawdown severity `18.9%`

The report frames these values as decision-support outputs. A higher modelled median is not automatically a better decision because drawdown, uncertainty, model risk, evidence quality, and user tolerance still matter.

## Test Credentials or Sample Inputs

No login credentials are required.

The app does not require users to upload private financial data. Users can enter sample budget values directly in the interface. Market data is prepared through the built-in asset panel workflow.

Suggested Personal Finance Planner sample values for testing:

- Take-home income: `£460/week`
- Fixed essentials: `£185/week`
- Variable essentials: `£80/week`
- Discretionary spending: `£85/week`
- Free margin produced by the app: approximately `£110/week`
- Example savings target: `£32/week`
- Example short-term horizon: `52 weeks`

These values are only sample inputs for testing the prototype flow. They are not financial recommendations.

## Known Limitations and Future Work

LifeBudget Micro is a working educational prototype, not a production financial planning platform.

### Current limitations

- LifeBudget Micro is not financial advice, investment advice, or a personal suitability assessment.
- The app is a historical what-if and scenario-comparison prototype. It shows how selected strategies would have behaved over available historical data; it does not predict future market outcomes.
- Investment outputs are based on historical backtests, educational proxies, and scenario assumptions. Historical performance does not guarantee future performance.
- The prototype does not fully model tax, ISA or pension rules, platform fees, fund charges, bid-ask spreads, inflation, currency effects, debt, liquidity needs, asset availability, or individual risk tolerance.
- The hosted Streamlit version uses prepared/cached market-data panels to keep the online deployment reliable. A production version should use a more robust API-based data layer.
- The long-term projection works at strategy-return level and does not yet show month-by-month asset-level contribution allocation.
- Mobile browser access may work for quick inspection, but the prototype is primarily tested and optimised for desktop/laptop use.
- The system focuses on demonstrating core project logic and behaviour rather than production-grade security, persistence, or large-scale deployment.

### Future development opportunities

- Rebuild the frontend with a production web framework such as React, HTML, CSS, and JavaScript to improve responsiveness across desktop and mobile.
- Replace cached CSV panels with robust market-data APIs, multiple data sources, fallback handling, and validation checks.
- Add user accounts and database storage so users can save, reload, compare, and export plans and strategy setups.
- Expand advanced strategy controls with richer parameter governance while keeping the default interface simple.
- Extend governance layers such as coherence checks, stability checks, and guided apply/reset controls.
- Add recommendation-diversity safeguards to reduce convergence risk where many users are pushed toward overly similar setups.
- Explore a forward-looking research mode using scenario variables such as inflation, interest-rate regimes, central-bank policy, and geopolitical stress assumptions.
- Improve long-term projections with asset-level contribution allocation, rebalancing, fees, inflation, and alternative market regimes.

## User Guide

A fuller PDF user guide is included with the submission. It explains each module, expected inputs, outputs, interpretation guidance, troubleshooting, and the recommended demo flow.
