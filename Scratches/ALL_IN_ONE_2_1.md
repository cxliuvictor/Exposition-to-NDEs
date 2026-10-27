# Simplified Dataset Interview Pipeline
**Original-inspired, Current-adapted | Minimal but methodologically complete**

Use this as a **top-to-bottom workflow**, not as a library reference.

The core idea is:

**load → understand → clean → split → EDA → baseline → simple model → nonlinear model → diagnose → freeze → OOS → conclude**

Only take branches that match the dataset.

---

# 0. Start simple

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

target = "target"       # change this
time_col = None         # e.g. "date" if time-ordered, otherwise None
id_cols = []            # e.g. ["customer_id"]
drop_cols = []          # leakage / unusable columns
```

**What this does:** loads the data and records the few dataset-specific choices you need.

Do **not** import every sklearn class at the start. Import things when you need them.

---

# 1. Understand the dataset

```python
df.head()
df.shape
df.info()

print(df.isna().sum().sort_values(ascending=False).head(20))
print("Duplicates:", df.duplicated().sum())

display(df.describe(include="all").T)
```

Also inspect:

```python
print(df[target].value_counts(dropna=False).head(20))
```

## Questions to answer before modelling

- What does one row represent?
- What exactly is the target?
- Regression or classification?
- Is there a time axis?
- Are rows independent?
- Are there repeated people/assets/accounts?
- Any ID columns?
- Any feature that would only be known **after** the target?
- Missing values?
- Categorical variables?

**Main rule:** a wrong target definition or leaky feature matters much more than choosing the wrong model.

---

# 2. Do conservative cleaning

Remove rows with no target:

```python
df = df.loc[df[target].notna()].copy()
```

Replace infinities:

```python
num_cols_now = df.select_dtypes(include=np.number).columns

df[num_cols_now] = df[num_cols_now].replace(
    [np.inf, -np.inf],
    np.nan
)
```

Check constants:

```python
constant_cols = [
    c for c in df.columns
    if df[c].nunique(dropna=False) <= 1
]

print("Constant columns:", constant_cols)
```

Build the candidate feature list:

```python
excluded = [target] + id_cols + drop_cols + constant_cols

features = [
    c for c in df.columns
    if c not in excluded
]
```

## Only if needed

### Sentinel value

```python
df["x"] = df["x"].replace(-999, np.nan)
```

### Numeric stored as text

```python
df["x"] = pd.to_numeric(
    df["x"],
    errors="coerce"
)
```

### Duplicate key

```python
df.duplicated(
    subset=["asset", "date"]
).sum()
```

Do **not** delete strange observations just because they are extreme.

---

# 3. Decide the validation structure

This is one of the most important decisions.

## 3A. Ordinary IID / generic tabular data

```python
from sklearn.model_selection import train_test_split

X = df[features].copy()
y = df[target].copy()

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y if y.nunique() == 2 else None
)

X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp if y.nunique() == 2 else None
)
```

This gives roughly:

```text
70% train
15% validation
15% test
```

**Why:** train learns model parameters, validation chooses the model, test is used once at the end.

If the sample is small or the task is explicitly simpler, a train/validation split alone is acceptable.

---

## 3B. Time-series / forecasting data

Time series is still a regression/classification task, but the **information set and split change**.

First sort:

```python
df[time_col] = pd.to_datetime(df[time_col])
df = df.sort_values(time_col).reset_index(drop=True)
```

Example: predict next-period `load`:

```python
df["target_next"] = df["load"].shift(-1)

df["lag1"] = df["load"].shift(1)
df["lag24"] = df["load"].shift(24)

df["roll24"] = (
    df["load"]
      .shift(1)
      .rolling(24)
      .mean()
)

df["hour"] = df[time_col].dt.hour

df["hour_sin"] = np.sin(
    2*np.pi*df["hour"]/24
)

df["hour_cos"] = np.cos(
    2*np.pi*df["hour"]/24
)

df = df.dropna().copy()

target = "target_next"
```

Then split chronologically:

```python
features = [
    c for c in df.columns
    if c not in [target, time_col]
]

X = df[features]
y = df[target]

n = len(df)

train_end = int(0.70 * n)
valid_end = int(0.85 * n)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_valid = X.iloc[train_end:valid_end]
y_valid = y.iloc[train_end:valid_end]

X_test = X.iloc[valid_end:]
y_test = y.iloc[valid_end:]
```

**Rule:** do not randomly shuffle forecasting data.

---

# 4. Do small, useful EDA — not endless EDA

Only use **training labels** for target-driven exploration.

```python
train_eda = df.loc[X_train.index].copy()
```

## Target

### Regression

```python
train_eda[target].hist(
    bins=30,
    figsize=(6, 4)
)

plt.title("Target distribution")
plt.show()
```

### Classification

```python
train_eda[target].value_counts().plot.bar(
    figsize=(5, 4)
)

plt.title("Class counts")
plt.show()
```

---

## Numeric relationships

For regression:

```python
num_features = [
    c for c in features
    if pd.api.types.is_numeric_dtype(train_eda[c])
]

corr = (
    train_eda[num_features]
    .corrwith(train_eda[target])
    .sort_values(
        key=np.abs,
        ascending=False
    )
)

display(corr.head(10))
```

For classification, group means / positive rates are often easier to interpret:

```python
train_eda.groupby("category")[target].agg(
    ["mean", "count"]
)
```

---

## Probe one promising feature

```python
feature = "x"

tmp = train_eda[[feature, target]].dropna().copy()

tmp["bucket"] = pd.qcut(
    tmp[feature],
    8,
    duplicates="drop"
)

display(
    tmp.groupby(
        "bucket",
        observed=True
    )[target]
    .agg(["mean", "count"])
)
```

**What this tells you:** whether the target changes monotonically, nonlinearly, or barely at all as the feature changes.

Stop after you have **2–3 useful observations**.

---

# 5. Build preprocessing only if the data needs it

Identify numeric and categorical columns:

```python
numeric_cols = X_train.select_dtypes(
    include=np.number
).columns.tolist()

categorical_cols = X_train.select_dtypes(
    exclude=np.number
).columns.tolist()
```

If the data is already numeric and complete, you may skip the more elaborate preprocessing.

Otherwise:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
```

Numeric preprocessing:

```python
numeric_pipe = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),
    (
        "scaler",
        StandardScaler()
    )
])
```

Categorical preprocessing:

```python
categorical_pipe = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),
    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])
```

Combine them:

```python
preprocess = ColumnTransformer([
    (
        "num",
        numeric_pipe,
        numeric_cols
    ),
    (
        "cat",
        categorical_pipe,
        categorical_cols
    )
])
```

**What this does:** learns medians, scaling and category encoding from the training data only, preventing leakage.

---

# 6A. REGRESSION / FORECASTING BRANCH

---

## 6A.1 Baseline

Always establish something simple first.

```python
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

baseline = DummyRegressor(
    strategy="mean"
)

baseline.fit(
    X_train,
    y_train
)

baseline_pred = baseline.predict(
    X_valid
)
```

Metrics:

```python
def regression_metrics(y_true, pred):
    return {
        "RMSE":
            mean_squared_error(
                y_true,
                pred
            ) ** 0.5,
        "MAE":
            mean_absolute_error(
                y_true,
                pred
            ),
        "R2":
            r2_score(
                y_true,
                pred
            )
    }
```

```python
results = [{
    "model": "Baseline",
    **regression_metrics(
        y_valid,
        baseline_pred
    )
}]
```

### Forecasting baseline

For time series, also compare against something like:

```python
persistence_pred = X_valid["lag1"]
```

or an appropriate seasonal lag.

A complicated model should beat the naive time-series baseline.

---

## 6A.2 Simple model: Ridge

```python
from sklearn.linear_model import Ridge
```

With preprocessing:

```python
ridge = Pipeline([
    (
        "preprocess",
        preprocess
    ),
    (
        "model",
        Ridge(alpha=1.0)
    )
])

ridge.fit(
    X_train,
    y_train
)

ridge_pred = ridge.predict(
    X_valid
)

results.append({
    "model": "Ridge",
    **regression_metrics(
        y_valid,
        ridge_pred
    )
})
```

**Idea:** Ridge is a regularised linear model. It is a stable benchmark for mostly linear structure.

---

## 6A.3 Nonlinear model

For a simple generic choice:

```python
from sklearn.ensemble import RandomForestRegressor
```

```python
forest = Pipeline([
    (
        "preprocess",
        preprocess
    ),
    (
        "model",
        RandomForestRegressor(
            n_estimators=150,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
    )
])

forest.fit(
    X_train,
    y_train
)

forest_pred = forest.predict(
    X_valid
)

results.append({
    "model": "Random Forest",
    **regression_metrics(
        y_valid,
        forest_pred
    )
})
```

Comparison:

```python
comparison = (
    pd.DataFrame(results)
      .sort_values("RMSE")
)

display(comparison)
```

**Interpretation:**

- Ridge better → structure may be mostly linear / sample may be noisy.
- Random Forest better → nonlinearities / interactions may matter.
- Neither beats baseline → weak signal, poor features, or wrong setup.

Do not automatically add more models.

---

# 7A. Diagnose the winning regression model

Suppose the Random Forest wins:

```python
chosen_model = forest
chosen_pred = forest_pred
```

## Actual vs predicted

```python
plt.figure(figsize=(5, 5))

plt.scatter(
    y_valid,
    chosen_pred,
    alpha=0.3
)

lo = min(
    y_valid.min(),
    chosen_pred.min()
)

hi = max(
    y_valid.max(),
    chosen_pred.max()
)

plt.plot(
    [lo, hi],
    [lo, hi],
    "--"
)

plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs predicted")
plt.show()
```

## Residuals

```python
residual = (
    np.asarray(y_valid)
    -
    np.asarray(chosen_pred)
)

plt.figure(figsize=(6, 4))

plt.scatter(
    chosen_pred,
    residual,
    alpha=0.3
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Residuals")
plt.show()
```

## Error by target bucket

```python
errors = pd.DataFrame({
    "actual": np.asarray(y_valid),
    "pred": np.asarray(chosen_pred)
})

errors["bucket"] = pd.qcut(
    errors["actual"],
    5,
    duplicates="drop"
)

display(
    errors.groupby(
        "bucket",
        observed=True
    )
    .apply(
        lambda g:
        pd.Series({
            "n": len(g),
            "RMSE":
                np.sqrt(
                    np.mean(
                        (
                            g["actual"]
                            -
                            g["pred"]
                        ) ** 2
                    )
                )
        })
    )
)
```

**Question:** where does the model fail?

That is often more interesting than another 0.5% RMSE improvement.

---

# 8A. Try at most one justified improvement

Only change the model/features if you can say:

> “Because the data suggests X, I tried Y.”

Examples:

Strong skew:

```python
df["log_x"] = np.log1p(
    df["x"].clip(lower=0)
)
```

Cyclic time:

```python
df["hour_sin"] = np.sin(
    2*np.pi*df["hour"]/24
)

df["hour_cos"] = np.cos(
    2*np.pi*df["hour"]/24
)
```

Delayed effect:

```python
df["roll6"] = (
    df["x"]
      .shift(1)
      .rolling(6)
      .mean()
)
```

Then re-run the same validation comparison.

Do **not** generate dozens of arbitrary features.

---

# 9A. Optional robustness check

If you have time:

```python
from sklearn.model_selection import (
    KFold,
    TimeSeriesSplit,
    cross_val_score
)

cv = (
    TimeSeriesSplit(n_splits=4)
    if time_col is not None
    else
    KFold(
        n_splits=4,
        shuffle=True,
        random_state=42
    )
)

scores = cross_val_score(
    chosen_model,
    X_train,
    y_train,
    cv=cv,
    scoring="neg_mean_squared_error"
)

rmse_scores = np.sqrt(-scores)

print(rmse_scores)
print(
    "Mean:",
    rmse_scores.mean(),
    "SD:",
    rmse_scores.std()
)
```

Large variation suggests instability.

If time is tight, this is **optional**.

---

# 10A. Freeze and evaluate OOS once

After model/features/hyperparameters are chosen:

```python
from sklearn.base import clone

X_fit = pd.concat([
    X_train,
    X_valid
])

y_fit = pd.concat([
    y_train,
    y_valid
])

final_model = clone(
    chosen_model
)

final_model.fit(
    X_fit,
    y_fit
)

test_pred = final_model.predict(
    X_test
)

print(
    regression_metrics(
        y_test,
        test_pred
    )
)
```

**Rule:** after seeing test performance, do not tune again and still call it untouched OOS.

---

# 6B. CLASSIFICATION BRANCH

Use this instead of the regression branch.

---

## 6B.1 Check class balance

```python
print(
    y_train.value_counts()
)

print(
    y_train.value_counts(
        normalize=True
    )
)
```

If 95% of observations are class 0, 95% accuracy can be useless.

---

## 6B.2 Baseline

```python
from sklearn.dummy import DummyClassifier

baseline = DummyClassifier(
    strategy="prior"
)

baseline.fit(
    X_train,
    y_train
)

baseline_prob = baseline.predict_proba(
    X_valid
)[:, 1]
```

---

## 6B.3 Simple model: Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

logistic = Pipeline([
    (
        "preprocess",
        preprocess
    ),
    (
        "model",
        LogisticRegression(
            max_iter=2000
        )
    )
])

logistic.fit(
    X_train,
    y_train
)

logistic_prob = logistic.predict_proba(
    X_valid
)[:, 1]
```

**Idea:** Logistic Regression gives a relatively simple probability score.

---

## 6B.4 Nonlinear model

```python
from sklearn.ensemble import RandomForestClassifier

forest = Pipeline([
    (
        "preprocess",
        preprocess
    ),
    (
        "model",
        RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
    )
])

forest.fit(
    X_train,
    y_train
)

forest_prob = forest.predict_proba(
    X_valid
)[:, 1]
```

---

## 6B.5 Compare probability quality

```python
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

results = []

for name, prob in [
    ("Baseline", baseline_prob),
    ("Logistic", logistic_prob),
    ("Random Forest", forest_prob)
]:
    results.append({
        "model": name,
        "ROC_AUC":
            roc_auc_score(
                y_valid,
                prob
            ),
        "Average_precision":
            average_precision_score(
                y_valid,
                prob
            )
    })

classification_table = (
    pd.DataFrame(results)
      .sort_values(
          "Average_precision",
          ascending=False
      )
)

display(classification_table)
```

**ROC-AUC:** how well positives rank above negatives.

**Average precision:** especially useful when positives are rare.

Pick the model based on the task objective, not blindly on accuracy.

---

# 7B. Choose a threshold

Suppose Random Forest wins:

```python
chosen_classifier = forest
chosen_prob = forest_prob
```

A probability is not yet an action.

At threshold 0.5:

```python
pred = (
    chosen_prob >= 0.5
).astype(int)
```

Metrics:

```python
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

print(
    "Precision:",
    precision_score(
        y_valid,
        pred,
        zero_division=0
    )
)

print(
    "Recall:",
    recall_score(
        y_valid,
        pred,
        zero_division=0
    )
)

print(
    "F1:",
    f1_score(
        y_valid,
        pred,
        zero_division=0
    )
)

print(
    confusion_matrix(
        y_valid,
        pred
    )
)
```

## Small threshold sweep

```python
rows = []

for threshold in [
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7
]:

    pred = (
        chosen_prob >= threshold
    ).astype(int)

    rows.append({
        "threshold": threshold,
        "precision":
            precision_score(
                y_valid,
                pred,
                zero_division=0
            ),
        "recall":
            recall_score(
                y_valid,
                pred,
                zero_division=0
            ),
        "F1":
            f1_score(
                y_valid,
                pred,
                zero_division=0
            )
    })

threshold_table = pd.DataFrame(
    rows
)

display(threshold_table)
```

Choose a threshold based on the stated cost/goal.

If no costs are provided, 0.5 plus a short sensitivity check is reasonable.

---

# 8B. Classification diagnostics

## Predicted-probability buckets

```python
calibration = pd.DataFrame({
    "prob": chosen_prob,
    "actual": np.asarray(y_valid)
})

calibration["bucket"] = pd.qcut(
    calibration["prob"],
    10,
    duplicates="drop"
)

display(
    calibration.groupby(
        "bucket",
        observed=True
    )
    .agg(
        n=("actual", "size"),
        mean_prob=("prob", "mean"),
        realised_rate=("actual", "mean")
    )
)
```

**Question:** do larger predicted probabilities correspond to higher realised positive rates?

Also inspect whether false positives / false negatives cluster in a meaningful subgroup if one exists.

---

# 9B. Optional robustness

```python
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score
)

cv = StratifiedKFold(
    n_splits=4,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    chosen_classifier,
    X_train,
    y_train,
    cv=cv,
    scoring="average_precision"
)

print(scores)
print(
    scores.mean(),
    scores.std()
)
```

For time-ordered classification, use chronological validation instead.

---

# 10B. Freeze model + threshold and evaluate OOS

```python
from sklearn.base import clone

X_fit = pd.concat([
    X_train,
    X_valid
])

y_fit = pd.concat([
    y_train,
    y_valid
])

final_classifier = clone(
    chosen_classifier
)

final_classifier.fit(
    X_fit,
    y_fit
)

test_prob = final_classifier.predict_proba(
    X_test
)[:, 1]

test_pred = (
    test_prob >= chosen_threshold
).astype(int)
```

Then report:

```python
print(
    "ROC-AUC:",
    roc_auc_score(
        y_test,
        test_prob
    )
)

print(
    "Average precision:",
    average_precision_score(
        y_test,
        test_prob
    )
)

print(
    "Confusion matrix:"
)

print(
    confusion_matrix(
        y_test,
        test_pred
    )
)
```

Do not tune the threshold after seeing test labels.

---

# 11. Optional Original-style decision branch

Use this **only if the objective is ranking / trade selection / PnL**.

Suppose the regression prediction is an expected opportunity value:

```python
score = np.asarray(
    chosen_pred
)

realised = np.asarray(
    y_valid
)
```

Trade only the top fraction:

```python
fraction = 0.20

cutoff = np.quantile(
    score,
    1 - fraction
)

take = (
    score >= cutoff
)

print(
    "Trades:",
    take.sum()
)

print(
    "Average realised outcome:",
    realised[take].mean()
)

print(
    "Total realised outcome:",
    realised[take].sum()
)

print(
    "Hit rate:",
    (
        realised[take] > 0
    ).mean()
)
```

**Idea:** prediction accuracy and decision quality can be different.

Only include this if it matches the actual interview objective.

---

# 12. Finish with a short research summary

Add a final Markdown cell:

```markdown
## Research summary

**Objective.**
Predict [target] for [unit of observation].

**Data / quality.**
The main issue was [missingness / leakage / class imbalance / time ordering / duplicates].

**Validation.**
I used [random / chronological / group-aware] splitting because [reason].

**Baseline.**
The baseline was [mean / persistence / majority class].

**Models.**
I compared [Ridge / Logistic] against [Random Forest].

**Result.**
The best validation model was [model], with [metric].

**Diagnostics.**
Performance was weakest for [tail / subgroup / period].

**OOS.**
After freezing the procedure, test performance was [metric].

**Limitation / next step.**
The main remaining concern is [issue]. I would next test [one specific idea].
```

---

# If time is running out

Do **not** add more models.

Minimum credible pipeline:

```text
1. Load + audit
2. Leakage / missing check
3. Correct split
4. 2–3 EDA observations
5. Baseline
6. Simple model
7. Nonlinear model
8. Validation comparison
9. One diagnostic
10. Final conclusion
```

If there is enough time:

```text
11. Robustness / CV
12. Freeze and OOS test
```

---

# Overall logic

Remember the logic:

```text
UNDERSTAND
→ CLEAN
→ SPLIT CORRECTLY
→ EXPLORE
→ BASELINE
→ SIMPLE MODEL
→ NONLINEAR MODEL
→ VALIDATE
→ DIAGNOSE
→ FREEZE
→ OOS
→ EXPLAIN
```

Use your other cheatsheet only when you forget syntax.

A simple, complete, correctly validated analysis is stronger than a sophisticated unfinished one.
