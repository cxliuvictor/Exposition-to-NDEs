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

# ============================================================
# SQUAREPOINT BOAT DATA TASK — END-TO-END CODE TEMPLATE
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PART 0 — LOAD DATA
# ============================================================

bd = pd.read_csv("BoatData_V1.csv")

print(bd.head())
print(bd.shape)
print(bd.info())


# ============================================================
# PART 1.1 — EXTRACT NUMERICAL FEATURES FROM TEXT
# Example:
# "753 ft² 70 m²"  -> extract 70 m²
# If only ft² exists, convert it to m²
# ============================================================

# ----- MAINSAIL AREA -----

m2 = (
    bd["Mainsail"]
    .str.extract(r'(\d+(?:\.\d+)?)\s*m(?:²|\^2|2)')[0]
    .astype(float)
)

ft2 = (
    bd["Mainsail"]
    .str.extract(r'(\d+(?:\.\d+)?)\s*ft(?:²|\^2|2)')[0]
    .astype(float)
)

bd["Mainsail_m2"] = m2.fillna(ft2 * 0.092903)


# ----- DRAFT -----
# If values contain metres, e.g. "4.8 6'11 2.1 m"

bd["Draft_m"] = (
    bd["Draft"]
    .str.extract(r'(\d+(?:\.\d+)?)\s*m')[0]
    .astype(float)
)


# ----- BEAM -----

bd["Beam_m"] = (
    bd["Beam"]
    .str.extract(r'(\d+(?:\.\d+)?)\s*m')[0]
    .astype(float)
)


# ----- HULL LENGTH -----

bd["Hull_length_m"] = (
    bd["Hull length"]
    .str.extract(r'(\d+(?:\.\d+)?)\s*m')[0]
    .astype(float)
)


# ============================================================
# PART 1.1a — AVERAGE MAINSAIL AREA OF BENETEAU BOATS
# Adjust "Model"/"Variant" depending on where BENETEAU appears
# ============================================================

beneteau_mean = bd.loc[
    bd["Model"].str.contains("BENETEAU", case=False, na=False),
    "Mainsail_m2"
].mean()

print("Average Beneteau mainsail area:", beneteau_mean)


# ============================================================
# PART 1.1b — CATEGORY WITH HIGHEST MEDIAN DRAFT
# ============================================================

draft_by_category = (
    bd.groupby("Category")["Draft_m"]
      .median()
      .sort_values(ascending=False)
)

print(draft_by_category)

print(
    "Category with highest median draft:",
    draft_by_category.index[0]
)


# ============================================================
# PART 1.2 — INVESTIGATE / CLEAN OTHER COLUMNS
# ============================================================

# Missing values
print(
    bd.isna()
      .sum()
      .sort_values(ascending=False)
)

# Number of unique values
print(
    bd.nunique()
      .sort_values()
)

# Exact duplicates
print(
    "Exact duplicates:",
    bd.duplicated().sum()
)

# Numeric summary
print(
    bd.describe().T
)

# Categorical summaries
for col in bd.select_dtypes(include="object").columns:
    print("\nCOLUMN:", col)
    print(
        bd[col]
        .value_counts(dropna=False)
        .head(10)
    )


# ------------------------------------------------------------
# Optional: remove exact duplicates if clearly genuine duplicates
# ------------------------------------------------------------

# bd = bd.drop_duplicates()


# ------------------------------------------------------------
# Ensure price is numeric
# ------------------------------------------------------------

bd["Listing Price"] = pd.to_numeric(
    bd["Listing Price"],
    errors="coerce"
)


# ------------------------------------------------------------
# Replace infinite values
# ------------------------------------------------------------

numeric_cols_now = bd.select_dtypes(include=np.number).columns

bd[numeric_cols_now] = bd[numeric_cols_now].replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# PART 1.3 — CORRELATION BETWEEN NUMERICAL FEATURES
# ============================================================

num = bd.select_dtypes(include=np.number)

corr = num.corr()

print(corr)


# ------------------------------------------------------------
# Heatmap
# ------------------------------------------------------------

plt.figure(figsize=(12, 9))

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0
)

plt.title("Correlation matrix")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Rank strongest UNIQUE correlation pairs
# ------------------------------------------------------------

upper_triangle = np.triu(
    np.ones(corr.shape),
    k=1
).astype(bool)

pairs = (
    corr.abs()
        .where(upper_triangle)
        .stack()
        .sort_values(ascending=False)
)

print("Top correlations:")
print(pairs.head(10))


# ------------------------------------------------------------
# Scatterplot strongest pair
# ------------------------------------------------------------

x, y = pairs.index[0]

sns.scatterplot(
    data=bd,
    x=x,
    y=y,
    alpha=0.4
)

plt.title(f"{x} vs {y}")
plt.show()


# ============================================================
# PART 1.4 — DISTRIBUTION OF LISTING PRICES
# ============================================================

sns.histplot(
    bd["Listing Price"].dropna(),
    bins=40
)

plt.title("Distribution of Listing Prices")
plt.show()


# ------------------------------------------------------------
# Log-price distribution — useful if heavily right-skewed
# ------------------------------------------------------------

sns.histplot(
    np.log1p(
        bd["Listing Price"].dropna()
    ),
    bins=40
)

plt.xlabel("log(1 + Listing Price)")
plt.title("Log Listing Price")
plt.show()


# ============================================================
# DEFINE "EXPENSIVE" BOATS
# Example = top 10% by listing price
# ============================================================

cutoff = bd["Listing Price"].quantile(0.90)

expensive = bd.loc[
    bd["Listing Price"] >= cutoff
].copy()

print("Expensive boat cutoff:", cutoff)


# ------------------------------------------------------------
# Common categorical characteristics
# ------------------------------------------------------------

print(
    expensive["Category"]
    .value_counts()
    .head(10)
)

print(
    expensive["Geographic Region"]
    .value_counts()
    .head(10)
)

print(
    expensive["Hull type"]
    .value_counts()
    .head(10)
)


# ------------------------------------------------------------
# Compare numeric characteristics:
# all boats vs expensive boats
# ------------------------------------------------------------

comparison = pd.DataFrame({
    "All boats":
        bd.select_dtypes(include=np.number).median(),

    "Expensive boats":
        expensive.select_dtypes(include=np.number).median()
})

print(comparison)


# ============================================================
# PART 1.5 — GEOGRAPHIC REGION VS LISTING PRICE
# ============================================================

region_price = (
    bd.groupby("Geographic Region")["Listing Price"]
      .agg(
          count="count",
          mean="mean",
          median="median"
      )
      .sort_values(
          "median",
          ascending=False
      )
)

print(region_price)


# ------------------------------------------------------------
# Boxplot
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

sns.boxplot(
    data=bd,
    x="Geographic Region",
    y="Listing Price"
)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Log-price version if price is heavily skewed
# ------------------------------------------------------------

bd["log_price"] = np.log1p(
    bd["Listing Price"]
)

plt.figure(figsize=(10, 5))

sns.boxplot(
    data=bd,
    x="Geographic Region",
    y="log_price"
)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Check whether regional differences persist WITHIN categories
# ------------------------------------------------------------

category_region_price = (
    bd.groupby(
        ["Category", "Geographic Region"]
    )["Listing Price"]
    .median()
    .unstack()
)

print(category_region_price)


# ============================================================
# PART 1.6 — MEAN / MEDIAN PRICE BY YEAR BUILT
# ============================================================

year_price = (
    bd.groupby("Year")["Listing Price"]
      .agg(
          count="count",
          mean="mean",
          median="median"
      )
      .reset_index()
)

print(year_price)


# ------------------------------------------------------------
# Optional: ignore years with tiny sample sizes
# ------------------------------------------------------------

year_price_clean = year_price.loc[
    year_price["count"] >= 5
]


# ------------------------------------------------------------
# Plot trend
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    year_price_clean["Year"],
    year_price_clean["mean"],
    label="Mean"
)

plt.plot(
    year_price_clean["Year"],
    year_price_clean["median"],
    label="Median"
)

plt.xlabel("Year built")
plt.ylabel("Listing price")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# ============================================================
# PART 2 — BUILD A PRICING MODEL
# ============================================================
# ============================================================


# ============================================================
# PART 2.1a — CHOOSE FEATURES
# ============================================================

features = [
    "Year",
    "Hull_length_m",
    "Beam_m",
    "Draft_m",
    "Mainsail_m2",
    "Geographic Region",
    "Category",
    "Hull type",
    "Model"
]

# Keep only features that actually exist
features = [
    col for col in features
    if col in bd.columns
]

X = bd[features].copy()
y = bd["Listing Price"].copy()


# Remove rows with missing target
mask = y.notna()

X = X.loc[mask]
y = y.loc[mask]


# ============================================================
# PART 2.1b — TRAIN / TEST SPLIT
# ============================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# IDENTIFY NUMERIC / CATEGORICAL COLUMNS
# ============================================================

numeric_cols = (
    X_train
    .select_dtypes(include=np.number)
    .columns
    .tolist()
)

categorical_cols = (
    X_train
    .select_dtypes(exclude=np.number)
    .columns
    .tolist()
)

print("Numeric:", numeric_cols)
print("Categorical:", categorical_cols)


# ============================================================
# PREPROCESSING
# ============================================================

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


# ------------------------------------------------------------
# Numeric:
# fill missing values with training median
# ------------------------------------------------------------

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    )
])


# ------------------------------------------------------------
# Categorical:
# fill missing values + one-hot encode
#
# handle_unknown="ignore" is especially useful because
# unseen boats may contain categories/makes not in training
# ------------------------------------------------------------

categorical_pipeline = Pipeline([
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


# ------------------------------------------------------------
# Combine numeric + categorical preprocessing
# ------------------------------------------------------------

preprocess = ColumnTransformer([
    (
        "num",
        numeric_pipeline,
        numeric_cols
    ),
    (
        "cat",
        categorical_pipeline,
        categorical_cols
    )
])


# ============================================================
# BASELINE MODEL
# ============================================================

from sklearn.dummy import DummyRegressor

baseline = DummyRegressor(
    strategy="median"
)

baseline.fit(
    X_train,
    y_train
)

baseline_pred = baseline.predict(
    X_test
)


# ============================================================
# MAIN MODEL — RANDOM FOREST
# ============================================================

from sklearn.ensemble import RandomForestRegressor

model = Pipeline([
    (
        "preprocess",
        preprocess
    ),
    (
        "model",
        RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
    )
])

model.fit(
    X_train,
    y_train
)

pred = model.predict(
    X_test
)


# ============================================================
# PART 2.1c — EVALUATE MODEL
# ============================================================

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def regression_metrics(y_true, pred):
    return {
        "MAE":
            mean_absolute_error(
                y_true,
                pred
            ),

        "RMSE":
            mean_squared_error(
                y_true,
                pred
            ) ** 0.5,

        "R2":
            r2_score(
                y_true,
                pred
            )
    }


print(
    "BASELINE:",
    regression_metrics(
        y_test,
        baseline_pred
    )
)

print(
    "RANDOM FOREST:",
    regression_metrics(
        y_test,
        pred
    )
)


# ------------------------------------------------------------
# Actual vs predicted
# ------------------------------------------------------------

plt.figure(figsize=(6, 6))

plt.scatter(
    y_test,
    pred,
    alpha=0.4
)

lo = min(
    y_test.min(),
    pred.min()
)

hi = max(
    y_test.max(),
    pred.max()
)

plt.plot(
    [lo, hi],
    [lo, hi],
    "--"
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.show()


# ------------------------------------------------------------
# Residuals
# ------------------------------------------------------------

residuals = (
    np.asarray(y_test)
    -
    np.asarray(pred)
)

plt.figure(figsize=(7, 4))

plt.scatter(
    pred,
    residuals,
    alpha=0.4
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Predicted price")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.show()


# ============================================================
# PART 2.1d — FEATURE IMPORTANCE
# ============================================================

from sklearn.inspection import permutation_importance

importance = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="neg_mean_absolute_error",
    n_repeats=5,
    random_state=42
)

importance_series = pd.Series(
    importance.importances_mean,
    index=X_test.columns
).sort_values(
    ascending=False
)

print(importance_series)


# ------------------------------------------------------------
# Plot feature importance
# ------------------------------------------------------------

importance_series.head(15).sort_values().plot.barh(
    figsize=(8, 5)
)

plt.xlabel("Permutation importance")
plt.tight_layout()
plt.show()


# ============================================================
# PART 2.1e — PREDICT PRICES FOR 10 UNSEEN BOATS
# ============================================================

new = pd.read_csv(
    "BoatData_NoPrice_V1.csv"
)


# ------------------------------------------------------------
# APPLY THE SAME NUMERIC EXTRACTION TO NEW DATA
# ------------------------------------------------------------

if "Mainsail" in new.columns:

    m2 = (
        new["Mainsail"]
        .str.extract(
            r'(\d+(?:\.\d+)?)\s*m(?:²|\^2|2)'
        )[0]
        .astype(float)
    )

    ft2 = (
        new["Mainsail"]
        .str.extract(
            r'(\d+(?:\.\d+)?)\s*ft(?:²|\^2|2)'
        )[0]
        .astype(float)
    )

    new["Mainsail_m2"] = (
        m2.fillna(
            ft2 * 0.092903
        )
    )


if "Draft" in new.columns:

    new["Draft_m"] = (
        new["Draft"]
        .str.extract(
            r'(\d+(?:\.\d+)?)\s*m'
        )[0]
        .astype(float)
    )


if "Beam" in new.columns:

    new["Beam_m"] = (
        new["Beam"]
        .str.extract(
            r'(\d+(?:\.\d+)?)\s*m'
        )[0]
        .astype(float)
    )


if "Hull length" in new.columns:

    new["Hull_length_m"] = (
        new["Hull length"]
        .str.extract(
            r'(\d+(?:\.\d+)?)\s*m'
        )[0]
        .astype(float)
    )


# ------------------------------------------------------------
# Predict
# ------------------------------------------------------------

new["Predicted Price"] = model.predict(
    new[features]
)

print(
    new["Predicted Price"]
)


# ------------------------------------------------------------
# Average + median predicted price
# ------------------------------------------------------------

print(
    "Average predicted price:",
    new["Predicted Price"].mean()
)

print(
    "Median predicted price:",
    new["Predicted Price"].median()
)


# ============================================================
# OPTIONAL — SHOW FINAL PREDICTIONS
# ============================================================

display(
    new[
        features
        +
        ["Predicted Price"]
    ]
)


# ============================================================
# VERY SHORT VERBAL METHODOLOGY SUMMARY
# ============================================================

# Part 1:
# - Standardise numerical features extracted from text into one unit.
# - Audit missingness, duplicates, data types and suspicious values.
# - Study numeric relationships using correlations + scatterplots.
# - Study price distribution and characteristics of expensive boats.
# - Compare pricing geographically and across build years.

# Part 2:
# - Predict Listing Price using physical characteristics,
#   boat type/model and geographic information.
# - Use train/test split.
# - Impute missing numeric/categorical values.
# - One-hot encode categorical variables.
# - Compare Random Forest against a simple median-price baseline.
# - Evaluate with MAE, RMSE and R².
# - Use permutation importance to understand influential features.
# - Apply identical cleaning + trained pipeline to unseen boats.
