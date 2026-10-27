# 00 — Onsite workflow

**Goal:** Turn an unfamiliar dataset into a defensible notebook before you run out of time.

## First 5–10 minutes — [K]

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")  # change filename
print(df.shape)
display(df.head())
df.info()
display(df.describe(include="all").T)
print(df.isna().sum().sort_values(ascending=False).head(15))
print("duplicate rows:", df.duplicated().sum())
print("unique values:", df.nunique().sort_values().head(15))
```

**What it does:** Checks the size, types, missing values, duplication, and rough ranges before you build any model.

Ask: What is **one row**? Which variable is the **target**? Are there dates, repeated entities, ID columns or features that would only be known *after* the target?

## Look at the target immediately — [K]

```python
target = "y"  # replace
fig, ax = plt.subplots(figsize=(6, 3))
ax.hist(df[target].dropna(), bins=30)
ax.set(title="Target distribution", xlabel=target, ylabel="Count")
plt.show()
```

**What it does:** Shows whether the target is skewed, discrete, imbalanced or dominated by extreme values.

## First modelling decision: which validation split? — [K]

```python
from sklearn.model_selection import train_test_split
# Use for independent tabular observations:
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**What it does:** Reserves unseen observations to test whether the model generalises; **do not use** if you are forecasting later dates from earlier ones.

```python
# Use for a forecasting dataset that has already been sorted by time:
i = int(len(df) * 0.8)
train, valid = df.iloc[:i].copy(), df.iloc[i:].copy()
```

**What it does:** Trains on the past and assesses predictions on the future.

## Time budget for a 09:30–15:00 work period — [+]

| Work time | Output that must exist |
|---|---|
| First 60–75 min | Data audit, 3–4 plots, basic cleaning |
| Next 30 min | Target, features, split, baseline |
| Next 90 min | A simple model and one nonlinear model |
| Next 60 min | Metric comparison, residuals, targeted improvement |
| Final 30–45 min | Clean notebook, conclusions, caveats |

**If behind:** Keep the baseline, one tree/random forest model, one comparison table and a conclusion. A completed basic analysis beats an unfinished advanced one.

## Notebook conclusion template — [+]

```markdown
## Findings
- **Question:** What was being predicted and from which information?
- **Data:** What quality problems or transformations mattered?
- **Validation:** Why was this split appropriate?
- **Baseline vs model:** State both values using the same metric.
- **Failure:** Which subgroup, tail or time period was hardest?
- **Next step:** One plausible improvement, not a list of ten.
```

**What it does:** Gives the researcher enough context to discuss your decisions without reading every cell.
