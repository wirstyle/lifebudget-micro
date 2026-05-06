# LifeBudget Micro

LifeBudget Micro is an educational decision-support Streamlit prototype for personal finance planning and investment scenario exploration. It helps users connect weekly budget choices with estimated savings capacity, compare savings-only outcomes with investment/proxy scenarios, and understand the risks and limitations behind those comparisons.

Unlike a simple savings calculator, LifeBudget Micro links estimated contribution capacity with a tested investment-style return path or clearly labelled educational proxy, then translates those assumptions into probabilistic long-term scenario ranges. The prototype demonstrates an end-to-end workflow: financial inputs produce budget feasibility outputs, feed an educational Strategy Engine, and support long-term scenario projections and final insights.

> **Educational notice:** All outputs should be interpreted as educational planning, backtesting, and scenario-comparison outputs only. They are not financial advice, investment advice, a regulated suitability assessment, or a production investment-planning service.


## Audience and Scope

This tool is intended for general users who want to build financial understanding around saving, surplus income, and long-term planning choices. It helps users explore questions such as: how much could I save, what happens if I only save in cash, what changes if I test an investment-style strategy, and what risks or limitations should I consider?

This guide is also intended for assessors and technical reviewers who need to run the prototype, follow the main workflow, and understand how it demonstrates a working software artefact.

## Core Features

### Core end-to-end workflow features

- Personal finance planning: budget inputs normalised to weekly terms, savings capacity estimation, and short-term target feasibility checks.
- Risk profile and asset universe setup: user-selected risk profile, asset universe selection, funding bridge confirmation, and market-data panel preparation.
- Strategy Engine: walk-forward historical backtesting with risk/return metrics and benchmark context.
- Long-term scenario projection: Monte Carlo-style comparison of savings-only and savings-plus-investing outcomes.
- Final report / insights: a summary view that connects personal finance inputs, investment strategy outputs, projections, and limitations.

### Optional analysis and transparency features

- Strategy improvement checks: optional post-run suggestions that compare alternative presets, tuning choices, universe mixes, or universe sizes.
- Reliability and robustness checks: educational diagnostics that help users understand whether a strategy result appears stable or sensitive to assumptions.

## Setup, Access, and Test Inputs

### Access and Run Options

**Recommended option: hosted online app**

The quickest way to test LifeBudget Micro is through the hosted Streamlit deployment:

https://lifebudget-micro.streamlit.app/

This is the recommended route for assessment and quick testing. The hosted version may take longer during market-panel preparation or Strategy Engine backtesting because it runs on Streamlit Community Cloud resources.

**Optional local run**

The app can also be run locally from the GitHub repository.

```bash
git clone https://github.com/wirstyle/lifebudget-micro.git
cd lifebudget-micro
pip install -r requirements.txt
streamlit run app.py
```

If the app does not open automatically, copy the local Streamlit URL shown in the terminal into your browser.

### Recommended Test Environment and Sample Inputs

LifeBudget Micro is a web-based Streamlit app. It can be opened in a browser on desktop, laptop, tablet, or mobile devices.

For assessment and full testing, a desktop or laptop browser is recommended. The app includes charts, tables, expandable diagnostics, and sidebar navigation, which are easier to inspect on a larger screen.

Mobile browser access may work for quick inspection, but the prototype has primarily been tested and optimised for desktop/laptop use.

No login credentials are required. The app does not require users to upload private financial data. Users can enter sample budget values directly in the interface. In the hosted version, market data is prepared through the built-in asset panel workflow using the deployment market-data panel.

Suggested Personal Finance Planner sample values for testing:

- Take-home income: `£460/week`
- Fixed essentials: `£185/week`
- Variable essentials: `£80/week`
- Discretionary spending: `£35/week`
- Free margin produced by the app: approximately `£160/week`
- Example savings target: `£48/week`
- Example short-term horizon: `12 weeks`

These values are only sample inputs for testing the prototype flow. They are not financial recommendations.

### Code Repository and Final Version

```text
GitHub repository: https://github.com/wirstyle/lifebudget-micro
Hosted app: https://lifebudget-micro.streamlit.app/
Deployed branch: deploy-lifebudget-demo
Final assessment tag: assignment-3-final
```

The final assessment tag identifies the submitted code snapshot and should match the commit used by the hosted Streamlit deployment.

## Quick Start: Recommended End-to-End Test Flow

1. Open the app online or run it locally.
2. Accept the educational notice on the home screen.
3. Open **Personal Finance Planner**.
4. Enter or keep sample weekly income, spending, and savings target values.
5. Review the cash-flow summary, free margin, savings target, feasibility check, and short-term path chart.
6. Continue to **Investment Strategy Lab**.
7. Select a risk profile, keep or adjust the recommended asset universe, and confirm that the market-data panel is ready.
8. Run the Strategy Engine and wait for the results-ready confirmation.
9. Review historical risk/return metrics, benchmark context, reliability diagnostics, and optional improvement checks.
10. Continue to **Long-Term Scenario Explorer**.
11. Generate a long-term projection using the available monthly contribution.
12. Open the final report / insights summary.


## What This Test Flow Demonstrates

This test flow demonstrates the full input-action-output chain expected from the prototype: the user enters budget values, the app calculates planning outputs, the investment setup prepares a market panel, the Strategy Engine produces historical metrics, and the Long-Term Scenario Explorer generates scenario outputs.

The results should be interpreted as educational backtests, proxies, and projections rather than advice, forecasts, or guarantees.


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

The Strategy Engine is documented as a compact user flow: choose a strategy template and style preset, check the ready-to-run state, optionally adjust fine-tune or suggestion-depth controls, run the portfolio, review historical metrics and benchmark context, optionally compare improvement candidates, and then use the stored result in the Long-Term Scenario Explorer.

A representative baseline Strategy Engine run in the guide uses the following historical backtest metrics for the current setup:

- CAGR: `9.73%`
- Volatility: `11.71%`
- Max drawdown: `-19.43%`
- Sharpe: `0.85`

If the user applies improvement suggestions, the active displayed result can update. In the guide screenshots, early improvement phases can move the active result toward approximately CAGR `9.93%` and then `10.00%`, with slightly lower volatility and drawdown. These remain historical test outputs for comparison inside the app, not forecasts.

Optional improvement checks can compare alternative presets, tuning choices, universe mixes, or universe sizes. The suggestion-depth control is configured before the run, while the checks themselves appear after the run. These checks preserve user control: the app can recommend a rerun-tested change, but the user can apply it, keep the current setup, or skip a phase if the trade-off is not strong enough. A skipped phase is not a failure; it can mean that no tested candidate materially improved the overall trade-off.

The guide also describes optional diagnostics, reliability, and start-date robustness checks. After a run, the sidebar can show a latest-result summary, timing snapshot, suggestion overhead, and reliability/robustness controls. These tools help assess transparency and sensitivity, but they remain historical educational checks, not forecasts or recommendations.

## Long-Term Scenario Example

The User Guide also documents the Long-Term Scenario Explorer. In the example shown in the guide, the module uses the contribution bridge from Personal Finance Planner and the completed Strategy Engine return path. It compares the selected investment/proxy path with a savings-only baseline across horizons such as 1, 20, 30, and 50 years.

Example values from the guide include:

- Contribution bridge: approximately `£208/month` (`£48/week`)
- 1-year median final wealth: approximately `£2,660`
- 20-year median final wealth: approximately `£154,777`
- 30-year median final wealth: approximately `£430,953`
- 50-year median final wealth: approximately `£3,234,437`

These are scenario outputs under the selected assumptions. In the guide screenshot, the current one-year projection also shows approximately P10 `£2,413`, P90 `£2,880`, loss vs contributions `25.00%`, and goal probability `3.00%`. These outputs are not forecasts, guarantees, or financial recommendations.

## Final Report Example

The final report summarises the personal finance plan, investment setup, tested Strategy Engine result, Long-Term Scenario comparison, and decision-support interpretation. In the example shown in the guide, the report highlights:

- Monthly contribution: approximately `£208/month`
- 50-year savings-only baseline: approximately `£124,800`
- 50-year investment/proxy median: approximately `£3,234,437`
- 50-year modelled difference: approximately `£3,109,637`
- Strategy snapshot: Sharpe `0.89`, CAGR `10.0%`, drawdown severity `18.8%`

The report also shows the cross-horizon comparison used in the final readout: 1-year, 20-year, 30-year, and 50-year horizons. The 50-year example compares a savings-only baseline of approximately `£124,800` with an investment/proxy median of approximately `£3,234,437`, while still showing the downside and upside scenario range.

The report frames these values as decision-support outputs. A higher modelled median is not automatically a better decision because drawdown, uncertainty, model risk, evidence quality, and user tolerance still matter.

## Known Limitations and Future Work

LifeBudget Micro is a working educational prototype, not a production financial planning platform. The submitted version demonstrates the core end-to-end workflow while leaving several production, regulatory, data, and research extensions outside the current scope.

### Current limitations

- LifeBudget Micro is an educational prototype, not financial advice, investment advice, or a personal suitability assessment.
- The app is a historical backtesting and scenario-comparison prototype. It shows how selected strategies would have behaved over available historical data; it does not predict future market outcomes.
- Investment outputs are based on historical backtests, educational proxies, and scenario assumptions. Historical performance does not guarantee future performance.
- The prototype does not fully model tax, ISA or pension rules, platform fees, fund charges, bid-ask spreads, inflation, currency effects, debt, liquidity needs, asset availability, or individual risk tolerance.
- The hosted Streamlit version uses prepared or cached market-data panels to keep the online deployment reliable. Market data availability may vary depending on the source and data preparation pipeline.
- Hosted execution may be slower than local execution because market-panel preparation, backtesting, optional suggestion checks, and robustness diagnostics run on hosted Streamlit resources.
- The long-term projection works at strategy-return level and does not yet show month-by-month asset-level contribution allocation.
- Mobile browser access may work for quick inspection, but the prototype is primarily tested and optimised for desktop or laptop use.
- The system focuses on demonstrating core project logic and behaviour rather than production-grade financial planning, security, persistence, compliance, or large-scale deployment.

### Future development opportunities

- A production frontend could be rebuilt with a dedicated web framework to improve responsiveness across desktop and mobile devices.
- A production data layer should replace cached CSV panels with robust API-based market-data pipelines, multiple data sources, fallback handling, and validation checks.
- Persistent user accounts and a database layer could allow users to save, reload, compare, and export plans, strategy setups, and scenario results.
- The current strategy controls could be expanded with richer parameter governance while keeping the default user interface simple and understandable.
- Additional governance layers, such as coherence checks, stability checks, recommendation diversity checks, and guided apply/reset controls, could be extended further in a future version.
- The current historical model could be extended into a forward-looking research mode using scenario variables such as inflation assumptions, interest-rate environments, central-bank policy regimes, and geopolitical stress scenarios.
- The Long-Term Scenario engine could be improved with more realistic contribution allocation, rebalancing, fees, inflation, alternative market regimes, and asset-level contribution modelling.
- Future work could include multiple functional datasets from different providers rather than relying on one prepared cached market-data source for the hosted demo.

## User Guide

A fuller PDF user guide is included with the submission. It explains each module, expected inputs, outputs, interpretation guidance, troubleshooting, and the recommended demo flow.
