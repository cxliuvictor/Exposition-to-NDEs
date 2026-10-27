# 03 — Visualisation: simplest useful plots

**Scope:** The four named courses use plots but do not include Kaggle's separate Data Visualization course. These are **[+] onsite essentials**, kept deliberately short.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)
df = pd.DataFrame({
    "date": pd.date_range("2025-01-01", periods=100),
    "x": rng.normal(size=100),
    "y": rng.normal(size=100),
    "group": rng.choice(["A", "B"], size=100)
})
df["target"] = 2*df["x"] + rng.normal(size=100)
```

## 1. `hist` and `df.hist` — [+]

**What they do:** Show a variable's distribution, including skew, gaps, extreme values and possible data errors.

```python
fig, ax = plt.subplots(figsize=(6, 3))
ax.hist(df["target"], bins=25)
ax.set(title="Target distribution", xlabel="Target", ylabel="Count")
plt.show()

df[["x", "y", "target"]].hist(figsize=(9, 5), bins=20)
plt.tight_layout(); plt.show()
```

## 2. `plot` — [+]

**What it does:** Shows whether a variable changes over time or has unusual periods.

```python
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(df["date"], df["target"])
ax.set(title="Target through time", xlabel="Date", ylabel="Target")
plt.show()
```

## 3. `scatter` — [+]

**What it does:** Shows whether a feature and target tend to move together or show a nonlinear shape.

```python
fig, ax = plt.subplots(figsize=(5, 3))
ax.scatter(df["x"], df["target"], alpha=0.45)
ax.set(xlabel="Feature x", ylabel="Target")
plt.show()
```

## 4. Bar chart of group means — [+]

**What it does:** Compares average target values across categories.

```python
df.groupby("group")["target"].mean().plot.bar(title="Mean target by group")
plt.tight_layout(); plt.show()
```

## 5. `subplots` — [+]

**What it does:** Puts several related plots in one compact figure rather than making many separate figures.

```python
fig, axes = plt.subplots(1, 2, figsize=(9, 3))
axes[0].hist(df["x"], bins=20)
axes[1].scatter(df["x"], df["target"], alpha=0.4)
axes[0].set_title("Feature distribution")
axes[1].set_title("Feature vs target")
fig.tight_layout(); plt.show()
```

## 6. Correlation heatmap — [+]

**What it does:** Shows which numerical columns have strong linear relationships, including pairs of redundant predictors.

```python
corr = df.select_dtypes(include=np.number).corr()
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(corr, vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticklabels(corr.columns)
fig.colorbar(im, ax=ax)
fig.tight_layout(); plt.show()
```

## 7. `actual` vs `predicted` and residuals — [+]

**What they do:** Show bias, systematic mistakes and whether errors grow for large targets.

```python
actual = df["target"].to_numpy()
predicted = 1.7 * df["x"].to_numpy()
fig, axes = plt.subplots(1, 2, figsize=(9, 3))
axes[0].scatter(actual, predicted, alpha=0.4)
axes[0].set(xlabel="Actual", ylabel="Predicted")
axes[1].scatter(predicted, actual-predicted, alpha=0.4)
axes[1].axhline(0, ls="--", color="grey")
axes[1].set(xlabel="Predicted", ylabel="Residual")
fig.tight_layout(); plt.show()
```

## 8. Seaborn if already installed — [+ optional]

**What it does:** Offers compact alternatives to matplotlib for standard histograms and heatmaps.

```python
# import seaborn as sns
# sns.histplot(data=df, x="target", bins=30)
# sns.heatmap(df[["x", "y", "target"]].corr(), annot=True)
```

**Rule:** Do not install another plotting library under time pressure just to make the figures prettier.
