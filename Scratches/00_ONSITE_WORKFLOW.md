# Onsite Workflow

## First 5–10 minutes

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

df.head()
df.shape
df.info()
df.isna().sum().sort_values(ascending=False)
df.duplicated().sum()
df.nunique().sort_values()
df.describe(include="all").T
```

Immediately identify:

- What does **one row** represent?
- What is the **target**?
- Regression or classification?
- Any time/order/group structure?
- Numeric vs categorical columns?
- Missingness?
- Duplicate rows or duplicate keys?
- IDs?
- Possible leakage?
- Target imbalance/skew?
- Any obvious impossible values?

## First plots

### Target distribution

```python
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(df["target"].dropna(), bins=30)
ax.set_title("Target distribution")
plt.show()
```

### Numeric columns quickly

```python
num_cols = df.select_dtypes(include=np.number).columns

df[num_cols].hist(bins=30, figsize=(14, 10))
plt.tight_layout()
plt.show()
```

### Correlation

```python
corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(corr, vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticks(range(len(corr.columns)))
ax.set_yticklabels(corr.columns)
fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()
```

## Decide validation before serious modelling

### IID / generic tabular

```python
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

Classification:

```python
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### Time series

```python
df = df.sort_values("date").reset_index(drop=True)
split = int(0.8 * len(df))
train = df.iloc[:split]
valid = df.iloc[split:]
```

Do not randomly shuffle a forecasting problem unless you have a specific reason.

## Modelling order

Regression:

1. Mean / persistence / seasonal-naive baseline
2. Ridge
3. Random Forest or HistGradientBoosting
4. Diagnostics
5. One or two targeted improvements

Classification:

1. Majority-class baseline
2. Logistic regression
3. Random Forest / tree model
4. Threshold analysis
5. Diagnostics

## Stop modelling before time expires

Leave time to produce:

- one concise model comparison table
- 2–4 useful figures
- main conclusion
- limitations
- next step

A readable notebook is part of the interview.

## Discussion structure

> I first identified the unit of observation and target, then audited missingness, duplicates and leakage. I used [validation method] because [reason]. I established [baseline], then compared [simple model] with [nonlinear model]. The strongest result was [result], but performance deteriorated in [bucket/regime], so my main remaining concern is [limitation].
