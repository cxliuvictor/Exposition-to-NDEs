# sklearn Modelling + Validation Cheatsheet

## Imports

```python
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    LogisticRegression,
    HuberRegressor
)

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from sklearn.inspection import permutation_importance
```

## Train / validation split

Regression:

```python
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

## Regression baseline

```python
baseline = np.repeat(y_train.mean(), len(y_valid))

rmse = mean_squared_error(y_valid, baseline) ** 0.5
mae = mean_absolute_error(y_valid, baseline)
r2 = r2_score(y_valid, baseline)
```

## Classification baseline

```python
majority = y_train.mode()[0]
baseline = np.repeat(majority, len(y_valid))
accuracy_score(y_valid, baseline)
```

Accuracy may be weak for imbalanced targets.

## Ridge

```python
ridge = Pipeline([
    ("scale", StandardScaler()),
    ("model", Ridge(alpha=1.0))
])

ridge.fit(X_train, y_train)
pred = ridge.predict(X_valid)
```

Why scale? Ridge penalises coefficient magnitude, so feature units should not determine regularisation strength.

## Small Ridge alpha search

```python
rows = []

for alpha in [0.01,0.1,1,10,100]:
    model = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha))
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_valid)

    rows.append({
        "alpha": alpha,
        "rmse": mean_squared_error(y_valid, pred) ** 0.5
    })

pd.DataFrame(rows)
```

Do not run a giant search onsite.

## Huber regression

```python
huber = Pipeline([
    ("scale", StandardScaler()),
    ("model", HuberRegressor())
])

huber.fit(X_train, y_train)
```

Useful when large residuals/outliers dominate least squares.

## Random Forest

Regression:

```python
rf = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
```

Classification:

```python
rf = RandomForestClassifier(
    n_estimators=200,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)
```

## HistGradientBoosting

```python
gbm = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=200,
    max_leaf_nodes=31,
    random_state=42
)

gbm.fit(X_train, y_train)
```

Good sklearn-native nonlinear baseline.

## Mixed numeric + categorical pipeline

```python
numeric_cols = ["age", "balance"]
categorical_cols = ["job", "marital"]
```

Numeric pipeline:

```python
num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])
```

Categorical pipeline:

```python
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
```

Combine:

```python
preprocessor = ColumnTransformer([
    ("num", num_pipe, numeric_cols),
    ("cat", cat_pipe, categorical_cols)
])
```

Full regression pipeline:

```python
model = Pipeline([
    ("preprocess", preprocessor),
    ("model", Ridge(alpha=1.0))
])
```

## Logistic regression

```python
clf = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=2000))
])

clf.fit(X_train, y_train)
prob = clf.predict_proba(X_valid)[:,1]
```

## Classification threshold

```python
threshold = 0.3
pred = (prob >= threshold).astype(int)
```

```python
precision_score(y_valid, pred)
recall_score(y_valid, pred)
f1_score(y_valid, pred)
roc_auc_score(y_valid, prob)
average_precision_score(y_valid, prob)
```

## Threshold sweep

```python
rows = []

for t in [0.1,0.2,0.3,0.4,0.5]:
    pred = (prob >= t).astype(int)

    rows.append({
        "threshold": t,
        "precision": precision_score(y_valid, pred, zero_division=0),
        "recall": recall_score(y_valid, pred, zero_division=0),
        "f1": f1_score(y_valid, pred, zero_division=0)
    })

pd.DataFrame(rows)
```

Select threshold on validation, never test.

## KFold

```python
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Use for IID-type data.

## StratifiedKFold

```python
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Maintains class proportions.

## TimeSeriesSplit

```python
tscv = TimeSeriesSplit(n_splits=5)
```

Use for ordered forecasting.

## Cross-validation score

```python
scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="neg_mean_squared_error"
)

rmse = np.sqrt(-scores)
print(rmse.mean(), rmse.std())
```

## Model comparison table

```python
results = []

results.append({
    "model": "Baseline",
    "rmse": baseline_rmse,
    "mae": baseline_mae
})

results.append({
    "model": "Ridge",
    "rmse": ridge_rmse,
    "mae": ridge_mae
})

results.append({
    "model": "GBM",
    "rmse": gbm_rmse,
    "mae": gbm_mae
})

pd.DataFrame(results).sort_values("rmse")
```

## Target-bucketed errors

```python
tmp = pd.DataFrame({
    "y": y_valid,
    "pred": pred
})

tmp["bucket"] = pd.qcut(tmp["y"], 5, duplicates="drop")

bucket_rmse = (
    tmp.groupby("bucket", observed=True)
       .apply(
           lambda g: np.sqrt(
               np.mean((g["y"] - g["pred"]) ** 2)
           )
       )
)
```

Useful when overall RMSE hides failure in important regions.

## Permutation importance

```python
pi = permutation_importance(
    model,
    X_valid,
    y_valid,
    n_repeats=10,
    random_state=42,
    scoring="neg_mean_squared_error"
)

importance = pd.Series(
    pi.importances_mean,
    index=X_valid.columns
).sort_values(ascending=False)

importance.head(10)
```

Caution: correlated features can substitute for one another.

## SHAP concept

Permutation importance:

> What happens to model performance if this feature is scrambled?

SHAP:

> How much did each feature contribute to this specific prediction relative to a baseline?

Do not spend onsite time installing SHAP unless it clearly helps.

## Freeze before test

Write down:

- features
- preprocessing
- model
- hyperparameters
- threshold
- metric

Then evaluate test once. Do not tune after seeing test performance.
