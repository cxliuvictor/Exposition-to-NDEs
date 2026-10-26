# Visualisation Cheatsheet

Use plots to answer questions, not decorate the notebook.

```python
import matplotlib.pyplot as plt
```

## Histogram

```python
fig, ax = plt.subplots(figsize=(6,4))
ax.hist(df["target"].dropna(), bins=30)
ax.set_title("Target distribution")
plt.show()
```

## Scatter

```python
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(df["feature"], df["target"], alpha=0.3)
ax.set_xlabel("Feature")
ax.set_ylabel("Target")
plt.show()
```

## Time series

```python
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(df["date"], df["target"])
plt.show()
```

Rolling overlay:

```python
ax.plot(
    df["date"],
    df["target"].rolling(30).mean(),
    label="30-period mean"
)
ax.legend()
```

## Multiple subplots

```python
cols = ["x1","x2","x3","x4"]
fig, axes = plt.subplots(2, 2, figsize=(10,7))

for ax, c in zip(axes.ravel(), cols):
    ax.hist(df[c].dropna(), bins=30)
    ax.set_title(c)

fig.tight_layout()
plt.show()
```

## Boxplot

```python
cats = df["category"].dropna().unique()
groups = [df.loc[df["category"] == c, "target"] for c in cats]

fig, ax = plt.subplots()
ax.boxplot(groups, labels=cats)
plt.show()
```

## Correlation heatmap — matplotlib

```python
num = df.select_dtypes(include="number")
corr = num.corr()

fig, ax = plt.subplots(figsize=(8,7))
im = ax.imshow(corr, vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticks(range(len(corr.columns)))
ax.set_yticklabels(corr.columns)
fig.colorbar(im, ax=ax)
fig.tight_layout()
plt.show()
```

## Actual vs predicted

```python
fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(y_valid, pred, alpha=0.3)

lo = min(y_valid.min(), pred.min())
hi = max(y_valid.max(), pred.max())
ax.plot([lo,hi], [lo,hi], "--")

ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title("Actual vs predicted")
plt.show()
```

## Residual plot

```python
resid = y_valid - pred

fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(pred, resid, alpha=0.3)
ax.axhline(0, linestyle="--")
ax.set_xlabel("Predicted")
ax.set_ylabel("Residual")
plt.show()
```

## Bucket response curve

```python
tmp = pd.DataFrame({
    "feature": df["feature"],
    "target": df["target"]
})

tmp["bucket"] = pd.qcut(tmp["feature"], 10, duplicates="drop")
resp = tmp.groupby("bucket", observed=True)["target"].mean()

fig, ax = plt.subplots()
ax.plot(range(len(resp)), resp.values, marker="o")
ax.set_xlabel("Feature decile")
ax.set_ylabel("Mean target")
plt.show()
```

## Save figure

```python
fig.savefig("figure.png", dpi=180, bbox_inches="tight")
```

# Seaborn — if installed

```python
import seaborn as sns
```

Distribution:

```python
sns.histplot(data=df, x="target", bins=30, kde=True)
```

Scatter:

```python
sns.scatterplot(data=df, x="feature", y="target", alpha=0.4)
```

Boxplot:

```python
sns.boxplot(data=df, x="category", y="target")
```

Heatmap:

```python
sns.heatmap(
    df.select_dtypes(include="number").corr(),
    cmap="coolwarm",
    center=0
)
```

Pairplot — use sparingly:

```python
sns.pairplot(
    df[["x1","x2","x3","target"]].sample(
        min(1000, len(df)),
        random_state=42
    )
)
```

## Interview rule

Prefer 2–4 clear plots tied to hypotheses over a large decorative EDA dump.
