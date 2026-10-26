# Time-Series Forecasting Cheatsheet

## Sort

```python
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
```

## Create future target

```python
df["target_next"] = df["target"].shift(-1)
```

At time `t`, features should only use information known by `t`.

## Lags

```python
df["lag1"] = df["target"].shift(1)
df["lag24"] = df["target"].shift(24)
df["lag168"] = df["target"].shift(168)
```

## Rolling statistics

Past-only:

```python
df["roll24_mean"] = (
    df["target"]
      .shift(1)
      .rolling(24)
      .mean()
)

df["roll24_std"] = (
    df["target"]
      .shift(1)
      .rolling(24)
      .std()
)
```

## Exponentially weighted mean

```python
df["ewm24"] = (
    df["target"]
      .shift(1)
      .ewm(span=24)
      .mean()
)
```

## Cyclic time

```python
df["hour"] = df["date"].dt.hour

df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
```

```python
df["weekday"] = df["date"].dt.dayofweek

df["dow_sin"] = np.sin(2*np.pi*df["weekday"]/7)
df["dow_cos"] = np.cos(2*np.pi*df["weekday"]/7)
```

## Chronological 80/10/10

```python
n = len(df)
i1 = int(0.8*n)
i2 = int(0.9*n)

train = df.iloc[:i1]
valid = df.iloc[i1:i2]
test = df.iloc[i2:]
```

## Persistence baseline

```python
pred = valid["lag1"]
```

## Seasonal naive

Daily hourly seasonality:

```python
pred = valid["lag24"]
```

Weekly:

```python
pred = valid["lag168"]
```

Always compare ML against these if they make sense.

## TimeSeriesSplit

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, valid_idx in tscv.split(X):
    X_tr = X.iloc[train_idx]
    X_va = X.iloc[valid_idx]
```

## Rolling temperature / physical inertia

```python
df["temp_roll6"] = (
    df["temp"]
      .shift(1)
      .rolling(6)
      .mean()
)
```

Use when the target responds with inertia.

## Error by hour

```python
tmp = pd.DataFrame({
    "hour": valid["hour"],
    "y": y_valid,
    "pred": pred
})

tmp["error"] = tmp["y"] - tmp["pred"]

rmse_by_hour = (
    tmp.groupby("hour")
       .apply(lambda g: np.sqrt(np.mean(g["error"]**2)))
)
```

## Error by target quantile

```python
tmp["bucket"] = pd.qcut(tmp["y"], 5, duplicates="drop")
```

## Leakage trap

Potentially wrong:

```python
df["roll24"] = df["target"].rolling(24).mean()
```

if current target is unavailable at forecast time.

Safer:

```python
df["roll24"] = df["target"].shift(1).rolling(24).mean()
```

## Good onsite workflow

```text
persistence
→ seasonal naive
→ Ridge
→ HistGradientBoosting
→ error by hour/period/target bucket
→ frozen OOS
```
