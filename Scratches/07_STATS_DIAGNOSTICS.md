# Statistics + Diagnostics Cheatsheet

## Pearson correlation

Linear association:

```python
df["x"].corr(df["y"], method="pearson")
```

## Spearman correlation

Monotonic/rank association:

```python
df["x"].corr(df["y"], method="spearman")
```

## Correlation matrix

```python
corr = df.select_dtypes(include=np.number).corr()
```

## Conditional means / decile response

```python
df["bucket"] = pd.qcut(df["feature"], 10, duplicates="drop")

response = (
    df.groupby("bucket", observed=True)["target"]
      .agg(["mean","std","count"])
)
```

A smooth monotonic response can be more informative than one correlation number.

## Standard error of mean

```python
x = df["target"].dropna()
sem = x.std(ddof=1) / np.sqrt(len(x))
```

## Approximate 95% confidence interval

```python
mean = x.mean()
sem = x.std(ddof=1) / np.sqrt(len(x))

lower = mean - 1.96 * sem
upper = mean + 1.96 * sem
```

## scipy tests

```python
from scipy import stats
```

Independent Welch t-test:

```python
stats.ttest_ind(
    group_a,
    group_b,
    equal_var=False,
    nan_policy="omit"
)
```

Paired t-test:

```python
stats.ttest_rel(before, after, nan_policy="omit")
```

Mann-Whitney:

```python
stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
```

In an onsite, effect size + validation often matter more than a p-value.

## Bootstrap mean difference

```python
rng = np.random.default_rng(42)
diffs = []

for _ in range(5000):
    a = rng.choice(group_a, size=len(group_a), replace=True)
    b = rng.choice(group_b, size=len(group_b), replace=True)
    diffs.append(a.mean() - b.mean())

np.quantile(diffs, [0.025,0.975])
```

## Residuals

```python
resid = y - pred

resid.mean()
resid.std()
np.quantile(resid, [0.01,0.05,0.5,0.95,0.99])
```

## Heteroskedasticity clue

```python
plt.scatter(pred, resid, alpha=0.3)
plt.axhline(0, linestyle="--")
```

Fan-shaped residual spread suggests variance changes with prediction level.

## Error by target bucket

```python
tmp = pd.DataFrame({"y": y, "pred": pred})
tmp["bucket"] = pd.qcut(tmp["y"], 5, duplicates="drop")

def rmse(g):
    return np.sqrt(np.mean((g["y"] - g["pred"]) ** 2))

bucket_rmse = tmp.groupby("bucket", observed=True).apply(rmse)
```

## Predicted-decile response

```python
tmp["pred_decile"] = pd.qcut(tmp["pred"], 10, duplicates="drop")

summary = (
    tmp.groupby("pred_decile", observed=True)["y"]
       .agg(["mean","count"])
)
```

Useful for ranking/trading/prioritisation.

## Classification calibration-style check

```python
tmp = pd.DataFrame({"prob": prob, "y": y_valid})
tmp["decile"] = pd.qcut(tmp["prob"], 10, duplicates="drop")

cal = (
    tmp.groupby("decile", observed=True)
       .agg(
           mean_prob=("prob","mean"),
           realised_rate=("y","mean"),
           n=("y","size")
       )
)
```

## Robustness by period

```python
tmp = tmp.sort_values("date").reset_index(drop=True)
tmp["period"] = pd.qcut(
    tmp.index,
    3,
    labels=["early","middle","late"]
)
```

Then recompute your metric by period.

Ask whether performance is concentrated in one regime.
