# 06 — Time Series: Kaggle's six-lesson logic, in code

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.DataFrame({
    "date": pd.date_range("2025-01-01", periods=200, freq="D"),
    "sales": 100 + np.arange(200)*0.2 + 8*np.sin(2*np.pi*np.arange(200)/7)
})
df = df.sort_values("date").set_index("date")
```

## 1. Time-step feature — [K: Lesson 1]

**What it does:** Uses how far through the sample an observation is to model a long-run trend.

```python
df["time"] = np.arange(len(df))
trend_model = LinearRegression()
trend_model.fit(df[["time"]], df["sales"])
trend_pred = trend_model.predict(df[["time"]])
```

**Caution:** Trend estimated over the full sample is only for illustration; for forecasting, fit on the training period.

## 2. Lag feature — [K: Lesson 1]

**What it does:** Gives the model previous observed values to help predict the next one.

```python
df["lag1"] = df["sales"].shift(1)
df["lag7"] = df["sales"].shift(7)
```

**Trap:** If predicting today's sales at the start of today, today's sales cannot appear as a feature.

## 3. Rolling mean for trend — [K: Lesson 2]

**What it does:** Smooths out short-term fluctuations so the longer-run direction is easier to see.

```python
df["trend7"] = df["sales"].shift(1).rolling(7).mean()
```

**Time dummy:** `np.arange(len(df))` is another simple way to represent trend, often with linear regression.

## 4. Seasonal indicators — [K: Lesson 3]

**What they do:** Let the model learn distinct average effects for different weekdays or months.

```python
df["weekday"] = df.index.dayofweek
weekday_flags = pd.get_dummies(df["weekday"], prefix="day", drop_first=True)
```

## 5. Fourier / sine-cosine features — [K: Lesson 3]

**What they do:** Describe a repeating cycle smoothly using waves rather than a separate indicator for each period.

```python
n = np.arange(len(df))
df["weekly_sin"] = np.sin(2*np.pi*n/7)
df["weekly_cos"] = np.cos(2*np.pi*n/7)
```

**Extra useful case:** Encode hour-of-day as `sin(2*pi*hour/24)` and `cos(2*pi*hour/24)` so 23:00 and 00:00 are adjacent.

## 6. Time series as features / lag embedding — [K: Lesson 4]

**What it does:** Creates several past-value columns so a conventional ML model can learn a short recent-history pattern.

```python
for lag in [1, 2, 3, 7, 14]:
    df["lag_%d" % lag] = df["sales"].shift(lag)
feature_cols = ["time", "weekday", "weekly_sin", "weekly_cos", "lag_1", "lag_7"]
model_df = df.dropna(subset=feature_cols + ["sales"]).copy()
```

## 7. Hybrid models — [K: Lesson 5]

**What they do:** Fit a simple model to large-scale structure, then a second model to what the first one fails to explain.

```python
from sklearn.ensemble import RandomForestRegressor
# Example training-only fragment (fit stages only on the training portion):
# trend.fit(X_train_time, y_train)
# residual = y_train - trend.predict(X_train_time)
# second = RandomForestRegressor(random_state=42).fit(X_train_features, residual)
# final_prediction = trend.predict(X_valid_time) + second.predict(X_valid_features)
```

**Not a default:** Start with a straightforward one-model baseline; try a hybrid only if there is enough time and a clear reason.

## 8. Multi-step forecasting strategies — [K: Lesson 6]

**What they do:** Handle forecasts for several future time steps rather than just one.

- **Recursive:** Predict one step, feed prediction back as an input, repeat. Simple but errors compound.
- **Direct:** Fit a separate forecast for each horizon. More models but no predicted-target feedback.
- **Multi-output:** Predict a vector of future values in one model.
- **DirRec / mixed strategies:** Blend direct/recursive ideas.

**Onsite default:** For a one-step task, build only a one-step forecast. Expand only if asked.

## 9. Chronological split + baseline — [+ decisive]

**What it does:** Evaluates a forecasting model on later observations and compares against simply using known history.

```python
model_df = df.dropna().copy()
i = int(0.8 * len(model_df))
train, valid = model_df.iloc[:i], model_df.iloc[i:]

# If predicting today's sales from values known BEFORE today:
persistence = valid["lag1"]
weekly_naive = valid["lag7"]
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(valid["sales"], persistence))
print(mean_absolute_error(valid["sales"], weekly_naive))
```

**Target definition trap:** If instead you form `target_next = sales.shift(-1)` and forecast tomorrow after observing today, the persistence baseline is **today's** sales (`sales`), not `lag1`; the comparable weekly value for tomorrow is six rows before today (`shift(6)` for daily observations).

## 10. `TimeSeriesSplit`, EWM, group lags — [+]

**What they do:** Repeatedly validate past-to-future predictions, smooth observations with an exponential decay, or create lags within entities.

```python
from sklearn.model_selection import TimeSeriesSplit
cv = TimeSeriesSplit(n_splits=4)
df["ewm7_past"] = df["sales"].shift(1).ewm(span=7).mean()
# If multiple shops/assets, sort by entity and timestamp first:
# panel["lag1"] = panel.groupby("shop")["sales"].shift(1)
```

**Pitfalls:** Check missing timestamps/DST, leakage from overlapping windows, and whether features from future exogenous variables would actually be available.
