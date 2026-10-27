# Dataset Validation Notes

The notebooks use public data directly rather than synthetic substitutes for the research projects/mocks.

Each loader:
1. downloads from the embedded public URL on first run;
2. caches a local CSV under `data/cache/`;
3. asserts expected columns;
4. asserts a minimum row count.

This prevents a wrong HTML/error page or materially wrong dataset from being silently treated as data.

## Project / mock datasets
- UCI Superconductivity: canonical UCI regression dataset; 21,263 superconductors and 81 engineered features plus critical temperature.
- S&P 500 five-year OHLCV: `all_stocks_5yr.csv` from Plotly's public dataset repository; this is the common Kaggle-style S&P 500 panel.
- PJM hourly energy: public GitHub copy of the widely used Kaggle PJM hourly energy-consumption dataset.
- UCI Adult Census Income: public UCI-derived dataset used widely for binary classification.
- UCI Air Quality: public UCI dataset with real missing-value sentinels and sensor/time structure.
- Ames Housing: public GitHub copy of the Kaggle House Prices training set.
- BTC-USD: public daily OHLCV history file.
- UCI Appliances Energy Prediction: canonical UCI CSV.

## Important
A public mirror can disappear. If a notebook URL ever fails:
- use the canonical named dataset;
- keep the task brief and split rules unchanged;
- do not replace it with a synthetic dataset.
