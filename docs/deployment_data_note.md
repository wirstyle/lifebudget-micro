# Deployment market-data note

The local development pipeline can retrieve historical market data through yfinance/Yahoo Finance. The deployed Streamlit demo uses compressed cached Yahoo-generated panels produced by the same local pipeline. This avoids public-cloud Yahoo rate limits and makes the demonstration reproducible.

The cached bundle contains 106 assets from the application strategy universe. The deployed demo therefore supports universe sizes up to 100 assets. Larger research sizes such as 150 and 250 require a larger live or externally hosted data source.

Cached files expected in `data/`:

- `deployment_daily_returns.csv.gz` — raw daily return panel used for diagnostics and daily-to-monthly rebuilds.
- `deployment_weekly_returns.csv.gz` — raw weekly return panel used for diagnostics.
- `deployment_asset_panel.csv.gz` — prepared monthly engine panel used as the stable fallback.

The investment engine and Step 6 projections use historical strategy returns generated from the validated Step 4 panel. Outputs are educational scenario simulations, not financial advice, not a live fund track record, and not forecasts.
