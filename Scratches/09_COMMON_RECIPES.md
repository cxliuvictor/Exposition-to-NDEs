# 09 — Reusable recipes for actual interview questions

## Recipe A — “How many rows satisfy two conditions?” — [K]

**What it does:** Creates a True/False mask and counts the True values.

```python
import pandas as pd
people = pd.DataFrame({"age": [45, 58, 73], "bmi": [21., 19., 18.]})
print(((people["age"] > 50) & (people["bmi"] < 20)).sum())
```

## Recipe B — “Which groups have the highest average target?” — [K]

**What it does:** Compares target levels across categories.

```python
g = pd.DataFrame({"region": ["N", "S", "N"], "target": [10, 20, 30]})
print(g.groupby("region")["target"].agg(["size", "mean"]).sort_values("mean", ascending=False))
```

## Recipe C — “Build a simple full regression pipeline” — [K]

**What it does:** Fits a baseline, decision tree and forest to the same train/validation split so model performance is directly comparable.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Replace X/y with your real cleaned numeric features and target.
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.2, random_state=42)
# models = {
#     "tree": DecisionTreeRegressor(max_leaf_nodes=30, random_state=42),
#     "forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
# }
# baseline_mae = mean_absolute_error(y_valid, np.repeat(y_train.mean(), len(y_valid)))
# for name, model in models.items():
#     model.fit(X_train, y_train)
#     print(name, mean_absolute_error(y_valid, model.predict(X_valid)))
```

## Recipe D — “My dataset has strings and missing values” — [K]

**What it does:** Fits imputation and category encoding using *only training data* within a reusable pipeline.

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Assume X is a DataFrame and y is a Series, already split.
# nums = X_train.select_dtypes(include="number").columns.tolist()
# cats = X_train.select_dtypes(exclude="number").columns.tolist()
# pre = ColumnTransformer([
#     ("num", SimpleImputer(strategy="median"), nums),
#     ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
#                       ("oh", OneHotEncoder(handle_unknown="ignore"))]), cats)
# ])
# pipe = Pipeline([("preprocess", pre), ("model", RandomForestRegressor(n_estimators=100, random_state=42))])
# pipe.fit(X_train, y_train)
# pred = pipe.predict(X_valid)
```

## Recipe E — “It has dates. Is it a forecasting problem?” — [K]

**What it does:** Makes a past-only feature and compares it to a naïve recent-value forecast.

```python
series = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=10),
                       "y": [10, 11, 12, 10, 14, 13, 15, 16, 14, 18]})
series = series.sort_values("date")
series["lag1"] = series["y"].shift(1)
series["roll3"] = series["y"].shift(1).rolling(3).mean()
# Split by date, NOT with random shuffling.
```

## Recipe F — “Where does the model fail?” — [+]

**What it does:** Compares absolute error for low, middle and high targets.

```python
results = pd.DataFrame({"actual": [10, 12, 25, 40, 50, 60],
                        "pred": [11, 11, 22, 35, 42, 48]})
results["bucket"] = pd.qcut(results["actual"], 3, duplicates="drop")
print(results.groupby("bucket", observed=True).apply(
    lambda d: (d["actual"] - d["pred"]).abs().mean()))
```

## Recipe G — “How should I present this?” — [+]

**What it does:** Reduces the final notebook into a small number of findings rather than a chronology of every command.

```markdown
## Objective
Predict **[target]** from information available at **[time]**.

## Data
[N rows], [M features]. Main issue: [missing/leaky/duplicates].

## Model comparison
| Model | Validation MAE | Comment |
|---|---:|---|
| Baseline | ... | ... |
| Tree | ... | ... |
| Random Forest | ... | ... |

## Conclusion
- Main improvement: ...
- Weak region/subgroup: ...
- Limitation: ...
- If I had more time: ...
```

## Recipe H — “I have 25 minutes left” — [+]

**What it does:** Forces a finishable answer: stop feature engineering, run one baseline/model, write one comparison, create one plot and state limitations.

**Stop adding algorithms; tidy the notebook so the interviewer can follow it.**
