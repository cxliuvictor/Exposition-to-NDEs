# 07 — Statistics and diagnostics: what to check after fitting

**Course basis:** Intro ML measures generalisation mainly through MAE; Intermediate ML uses CV; Time Series examines trend/seasonality. Remaining diagnostics below are **[+] onsite-useful additions**.

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({
    "x": [1., 2., 3., 4., 5., 6.],
    "target": [2., 3., 4., 8., 10., 11.],
    "prediction": [2., 3., 5., 6., 8., 11.]
})
```

## 1. `.corr` (Pearson / Spearman) — [+]

**What it does:** Pearson measures linear association; Spearman measures whether larger values tend to have larger ranks.

```python
df["x"].corr(df["target"])
df["x"].corr(df["target"], method="spearman")
```

**Trap:** Neither implies causation, and low correlation need not imply no nonlinear predictability.

## 2. MAE / RMSE / R² — [K MAE] / [+ others]

**What they do:** MAE shows typical absolute errors; RMSE punishes occasional large errors more; R² measures improvement over predicting the target mean.

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
y, pred = df["target"], df["prediction"]
print("MAE", mean_absolute_error(y, pred))
print("RMSE", mean_squared_error(y, pred)**0.5)
print("R2", r2_score(y, pred))
```

## 3. Mean error and residuals — [+]

**What they do:** Show whether the model systematically predicts too high/low and which examples it misses.

```python
df["error"] = df["target"] - df["prediction"]
print(df["error"].mean())
print(df["error"].quantile([0.05, 0.5, 0.95]))
```

## 4. Target-bucket error — [+]

**What it does:** Checks whether high/low target values have much worse error than the overall average.

```python
df["bucket"] = pd.qcut(df["target"], 3, duplicates="drop")
by_bucket = df.groupby("bucket", observed=True).apply(
    lambda g: np.mean(np.abs(g["target"]-g["prediction"]))
)
print(by_bucket)
```

**Extra interpretation:** When selecting trades or rare opportunities, errors on extreme targets may matter more than overall RMSE.

## 5. Predicted-decile/tercile realised mean — [+]

**What it does:** Tests whether the model ranks observations in a useful order, even if absolute predictions are imperfect.

```python
df["pred_bin"] = pd.qcut(df["prediction"], 3, duplicates="drop")
print(df.groupby("pred_bin", observed=True)["target"].agg(["mean", "size"]))
```

## 6. Out-of-sample stability — [+]

**What it does:** Checks that a result does not rely on one particular segment of a chronological validation period.

```python
validation = df.reset_index(drop=True).copy()  # must already be chronological for this use
# Split the ordered frame and calculate the same metric per block.
for block in np.array_split(np.arange(len(validation)), 3):
    s = validation.iloc[block]
    print(mean_absolute_error(s["target"], s["prediction"]))
```

## 7. Classification metrics — [+]

**What they do:** Precision counts how many predicted positives are correct; recall counts how many real positives you found; F1 combines both; ROC-AUC judges ranking across thresholds.

```python
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
actual = np.array([0, 0, 1, 1, 0, 1])
prob = np.array([0.1, 0.3, 0.6, 0.8, 0.4, 0.7])
label = (prob >= 0.5).astype(int)
print(precision_score(actual, label), recall_score(actual, label), f1_score(actual, label))
print(roc_auc_score(actual, prob))
```

## 8. Simple confidence interval / bootstrap — [+ optional]

**What they do:** Express uncertainty around a sample mean or model comparison instead of treating an observed difference as certain.

```python
x = np.array([3., 4., 5., 6., 7.])
se = x.std(ddof=1)/np.sqrt(len(x))
print((x.mean()-1.96*se, x.mean()+1.96*se))  # rough normal approximation
```

**Important:** Standard IID formulas are not appropriate without adjustment for strongly time-dependent observations.

## 9. Two-sample test and bootstrap — [+ optional]

**What they do:** A t-test compares group means under assumptions; a bootstrap resamples data to estimate sampling uncertainty.

```python
from scipy import stats  # if installed
a = np.array([2., 4., 5., 6., 7.])
b = np.array([1., 3., 3., 4., 5.])
print(stats.ttest_ind(a, b, equal_var=False))
rng = np.random.default_rng(42)
diffs = [rng.choice(a, len(a), replace=True).mean() -
         rng.choice(b, len(b), replace=True).mean() for _ in range(1000)]
print(np.quantile(diffs, [.025, .975]))
```

**Use only when:** Time permits and the dependence assumptions make sense; resampling individual rows is often wrong for time series.
