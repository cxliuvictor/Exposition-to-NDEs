# Interview Cheat Sheet

## First 20 minutes
```python
df.head(); df.shape; df.info(); df.isna().sum(); df.duplicated().sum(); df.nunique(); df.describe()
```
Then: target? one row? numeric/categorical? time/group structure? leakage? target plot? validation?

## Regression
Baseline → Ridge → RandomForest/HistGradientBoosting.

## Time series
Sort; no shuffle; lag; past-only rolling; persistence baseline.

## If stuck
Use internet docs for syntax. Do not spend 20 minutes reconstructing an API.

## Final 30 minutes
Stop adding models. Make the notebook readable and add conclusions.
