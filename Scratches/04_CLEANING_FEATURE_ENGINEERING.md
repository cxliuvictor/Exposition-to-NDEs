# Cleaning + Feature Engineering Cheatsheet

## Compact audit table

```python
audit = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "missing_n": df.isna().sum(),
    "missing_pct": df.isna().mean(),
    "nunique": df.nunique(dropna=False)
}).sort_values("missing_pct", ascending=False)
```

## Constants

```python
constant_cols = [
    c for c in df.columns
    if df[c].nunique(dropna=False) <= 1
]
```

## Near constants

```python
near_constant = []

for c in df.columns:
    top_freq = df[c].value_counts(normalize=True, dropna=False).iloc[0]
    if top_freq > 0.99:
        near_constant.append(c)
```

## Infinite values

```python
num_cols = df.select_dtypes(include=np.number).columns

inf_counts = {
    c: np.isinf(df[c]).sum()
    for c in num_cols
}
```

Replace:

```python
df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
```

## Sentinel values

```python
(df["feature"] == -999).sum()
```

```python
df.loc[df["feature"] == -999, "feature"] = np.nan
```

## Impossible values

```python
df.loc[df["age"] < 0]
df.loc[df["probability"] > 1]
df.loc[df["price"] < 0]
```

Do not automatically delete: determine whether it is a real extreme, a coding error, a sentinel, or a unit problem.

## Outliers

```python
df["x"].quantile([0.01,0.05,0.5,0.95,0.99])
```

IQR:

```python
q1 = df["x"].quantile(0.25)
q3 = df["x"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
```

Winsorise:

```python
lo, hi = df["x"].quantile([0.01,0.99])
df["x_clip"] = df["x"].clip(lo, hi)
```

## Log transform

```python
df["log_x"] = np.log1p(df["x"])
```

Use for positive/right-skewed variables.

## Cyclic time

Hour:

```python
df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
```

Weekday:

```python
df["dow_sin"] = np.sin(2*np.pi*df["weekday"]/7)
df["dow_cos"] = np.cos(2*np.pi*df["weekday"]/7)
```

Month:

```python
df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
```

## Smooth event proximity

```python
d = df["days_to_event"]
df["event_decay"] = np.exp(-d / 3)
```

Sigmoid-style:

```python
df["event_proximity"] = 1 / (1 + np.exp((d - 2) / 1.5))
```

## Rolling physical effects

```python
df["temp_roll_6"] = (
    df["temp"]
      .shift(1)
      .rolling(6)
      .mean()
)
```

Use when the target responds with inertia rather than only to the instantaneous observation.

## Interactions

```python
df["temp_x_hum"] = df["temp"] * df["humidity"]
```

```python
df["momentum_per_vol"] = df["momentum"] / (df["vol"] + 1e-8)
```

## Rank transforms

Global:

```python
df["rank_x"] = df["x"].rank(pct=True)
```

Cross-sectional:

```python
df["rank_x"] = df.groupby("date")["x"].rank(pct=True)
```

## Missing indicators

```python
df["x_missing"] = df["x"].isna().astype(int)
```

Then impute `x`.

## Correlated variables

```python
corr = df[numeric_cols].corr().abs()
```

Possible responses:

- keep both with Ridge
- remove one for simplicity
- retain both for a tree model
- use domain knowledge

Do not automatically delete every highly correlated variable.

## Leakage audit

For every feature ask:

> Could I know this value at prediction time?

Common leakage:

- future return
- post-event information
- target-derived aggregate
- final outcome data used before the event
- preprocessing fit on full data
- feature selection using the test target

## Train-only preprocessing

Manual pattern:

```python
median = train["x"].median()
train["x"] = train["x"].fillna(median)
valid["x"] = valid["x"].fillna(median)
test["x"] = test["x"].fillna(median)
```

Prefer sklearn Pipelines when modelling.
