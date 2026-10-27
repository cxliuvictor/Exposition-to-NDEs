# 04 — Cleaning and feature engineering

**Course basis:** Pandas lesson 5 (data types and missing values); Intermediate ML lessons 2, 3, 4 and 7 (missing, categorical, pipelines, leakage). **[+]** items address onsite-specific edge cases and feature hypotheses.

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({
    "age": [25, 41, 35, 41, 80],
    "income": [50000, np.nan, 72000, np.nan, -999],
    "city": ["London", "Paris", "unknown", "Paris", "London"],
    "month": [12, 1, 2, 3, 4],
    "target": [0, 1, 0, 1, 1]
})
```

## 1. Audit missingness and dtypes — [K]

**What it does:** Shows how many records would be affected if you drop or impute each field.

```python
pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "missing": df.isna().sum(),
    "missing_pct": df.isna().mean(),
    "unique": df.nunique(dropna=False)
})
```

## 2. `dropna`, `fillna`, `SimpleImputer` — [K]

**What they do:** Delete incomplete rows/columns or substitute a learnt typical value for missing cells.

```python
df.dropna(subset=["target"])                    # target needs to be known
income_fill = df["income"].replace(-999, np.nan).median()
df["income"].replace(-999, np.nan).fillna(income_fill)
```

**For modelling, prefer:**

```python
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="median")
# imp.fit(X_train[numeric_cols]); imp.transform(X_valid[numeric_cols])
```

**Trap:** `fit` learns the median and should use training observations only. Put it inside a pipeline.

## 3. Handle categorical values — [K]

**What it does:** Turns strings into input the model can understand.

```python
pd.get_dummies(df[["city"]])  # quick exploratory encoding
# In sklearn pipelines, use OneHotEncoder(handle_unknown="ignore") instead.
```

**Choices from Intermediate ML:** drop the feature, ordinal encoding (only when ordering is meaningful or model-compatible), or one-hot encoding.

## 4. `astype`, `to_numeric`, `to_datetime` — [K]/[+]

**What they do:** Repair columns read with the wrong type.

```python
df["age"].astype(float)
pd.to_numeric(pd.Series(["1.4", "?", "2.6"]), errors="coerce")
pd.to_datetime(pd.Series(["2025-01-01", "bad"]), errors="coerce")
```

## 5. Duplicates, constants and sentinel values — [+]

**What they do:** Catch repeated records, useless columns and special codes that look numeric but mean missing.

```python
df.duplicated().sum()
df.duplicated(subset=["age", "month"]).sum()  # only if these should form a key
constant_cols = [c for c in df if df[c].nunique(dropna=False) <= 1]
df["income"] = df["income"].replace(-999, np.nan)
```

**Trap:** A valid repeated customer transaction is not automatically a duplicate error.

## 6. Infinities and impossible values — [+]

**What they do:** Identify numbers that sklearn cannot usually handle or that violate the meaning of the variable.

```python
numeric = df.select_dtypes(include=np.number).columns
inf_count = np.isinf(df[numeric]).sum()
df[numeric] = df[numeric].replace([np.inf, -np.inf], np.nan)
df.loc[df["age"] < 0]
```

## 7. Skew/outliers: log, clip, quantiles — [+]

**What they do:** Reduce the influence of very large observations without assuming they are data errors.

```python
s = pd.Series([0, 1, 2, 4, 100.])
np.log1p(s)                                       # only if all values > -1
lo, hi = s.quantile([0.01, 0.99])
s.clip(lo, hi)
```

**Critical:** A tail may be economically important; measure error in that region before discarding/clipping it.

## 8. Cyclic calendar representation — [K: Time Series seasonality] / [+ direct encoding]

**What it does:** Makes December and January close to each other, unlike the raw integers 12 and 1.

```python
m = df["month"]
df["month_sin"] = np.sin(2*np.pi*m/12)
df["month_cos"] = np.cos(2*np.pi*m/12)
```

## 9. Lags and rolling features — [K: Time Series]

**What they do:** Use recent and seasonal past values to help forecast a future outcome.

```python
series = pd.Series([3, 4, 6, 8, 10, 9], dtype=float)
past1 = series.shift(1)
past3_mean = series.shift(1).rolling(3).mean()
```

## 10. Feature interactions and event distance — [+]

**What they do:** Allow for effects that depend jointly on two variables, or fade with distance from an event.

```python
temperature = np.array([5., 12., 25.])
humidity = np.array([0.8, 0.6, 0.3])
temperature * humidity
days_from_holiday = np.array([0, 1, 3, 10])
np.exp(-days_from_holiday/3.0)
```

**Rule:** State a hypothesis before introducing the transform; compare validation results with/without it.

## 11. Leakage guard — [K: Intermediate ML]

**What it does:** Prevents the model from using information it could not have had at prediction time.

- If deciding **before** a sales call, the call's eventual duration is unavailable.
- If forecasting yesterday-to-today returns from yesterday's close, today's closing price is future information.
- If imputing/scaling/encoding, fit on training only; transform validation/test using those learnt rules.
- If selecting features/hyperparameters/thresholds, use validation rather than final test performance.
