# 05 — sklearn: model, validate and compare with minimal code

**Course basis:** Intro to ML (decision trees, MAE, under/overfitting, random forests), Intermediate ML (imputation, categorical encoding, pipelines, CV, XGBoost, leakage).

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

sample = pd.DataFrame({
    "size": [55, 65, 75, 85, 95, 110, 125, 150, 175, 200, 230, 260],
    "age":  [15, 20, 18, 10, 8, 5, 11, 4, 6, 3, 2, 1],
    "price": [160, 190, 215, 240, 270, 315, 340, 400, 475, 540, 590, 680]
})
X = sample[["size", "age"]]
y = sample["price"]
```

## 1. Separate X and y — [K]

**What it does:** `X` contains information known at prediction time; `y` is the answer the model should learn to predict.

```python
X = sample.drop(columns=["price"])
y = sample["price"]
```

## 2. `train_test_split` — [K]

**What it does:** Hides some labelled examples so you can check generalisation instead of measuring memorisation.

```python
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.25, random_state=42
)
```

**Trap:** This random split is for non-temporal, approximately independent observations; for time data see `06`.

## 3. Baseline mean prediction — [+ but crucial]

**What it does:** Gives a simple target to beat before investing in more complex models.

```python
baseline = np.repeat(y_train.mean(), len(y_valid))
print("Baseline MAE:", mean_absolute_error(y_valid, baseline))
```

## 4. `DecisionTreeRegressor` — [K]

**What it does:** Learns if/else rules from the training examples and predicts an average target value inside each leaf.

```python
tree = DecisionTreeRegressor(max_leaf_nodes=5, random_state=42)
tree.fit(X_train, y_train)
pred_tree = tree.predict(X_valid)
print("Tree MAE:", mean_absolute_error(y_valid, pred_tree))
```

## 5. `max_leaf_nodes`: underfitting vs overfitting — [K]

**What it does:** Limits how detailed the tree's rules can become; too small misses patterns, too large memorises noise.

```python
for n_leaves in [2, 5, 10, 20]:
    model = DecisionTreeRegressor(max_leaf_nodes=n_leaves, random_state=42)
    model.fit(X_train, y_train)
    print(n_leaves, mean_absolute_error(y_valid, model.predict(X_valid)))
```

**Trap:** Pick complexity on validation, not on your final untouched test set.

## 6. `RandomForestRegressor` — [K]

**What it does:** Averages many decision trees so the prediction depends less on one noisy fitted tree.

```python
forest = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
forest.fit(X_train, y_train)
pred_forest = forest.predict(X_valid)
print("Forest MAE:", mean_absolute_error(y_valid, pred_forest))
```

## 7. MAE, RMSE and R² — [K MAE] / [+ RMSE/R²]

**What they do:** MAE measures typical absolute error; RMSE penalises large errors more; R² compares squared error with predicting the mean.

```python
from sklearn.metrics import r2_score
mae = mean_absolute_error(y_valid, pred_forest)
rmse = np.sqrt(mean_squared_error(y_valid, pred_forest))
r2 = r2_score(y_valid, pred_forest)
print(mae, rmse, r2)
```

## 8. Three missing-data choices — [K: Intermediate ML]

**What they do:** Either remove columns with missing values, replace missing values, or replace and add a missingness indicator.

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median", add_indicator=True)
# imputer.fit(X_train_num)          # learn median ONLY from training
# X_valid_imp = imputer.transform(X_valid_num)
```

**Trap:** A `Pipeline` is safer because it fits preprocessing within each training fold.

## 9. One-hot encoding + `ColumnTransformer` + `Pipeline` — [K]

**What they do:** Clean/encode different column types and fit a model as one repeatable sequence, without leaking validation information.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

mixed = pd.DataFrame({"size": [70, 80, 90, 100], "city": ["A", "B", "A", "C"]})
numerical = ["size"]
categorical = ["city"]
preprocess = ColumnTransformer(transformers=[
    ("num", SimpleImputer(strategy="median"), numerical),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])
pipe = Pipeline(steps=[("preprocess", preprocess),
                       ("model", RandomForestRegressor(n_estimators=50, random_state=42))])
# pipe.fit(X_train_mixed, y_train)    # use the SAME column names as above
# pipe.predict(X_valid_mixed)
```

**Simpler Kaggle alternative:** For just numeric columns, `SimpleImputer` followed by a forest is enough. Use the mixed-column pipeline when categorical variables actually matter.

## 10. `cross_val_score` — [K]

**What it does:** Repeats fitting/validation over several partitions rather than relying on one lucky split.

```python
from sklearn.model_selection import cross_val_score
scores = -cross_val_score(
    RandomForestRegressor(n_estimators=100, random_state=42),
    X, y, cv=3, scoring="neg_mean_absolute_error"
)
print(scores, scores.mean())
```

**Note:** `cv=3` normally uses non-shuffled K-fold for regression. When you need shuffled IID folds use `KFold`, and for forecasting use `TimeSeriesSplit`.

## 11. `XGBRegressor` — [K: Intermediate ML] / optional installation

**What it does:** Adds trees sequentially, each intended to fix mistakes left by earlier trees.

```python
# Only if xgboost is already installed and compatible:
# from xgboost import XGBRegressor
# model = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
# model.fit(X_train, y_train)
```

**Fallback [+]:** `HistGradientBoostingRegressor` provides a sklearn-native boosted-tree model, but check missing/categorical handling for your installed version.

```python
from sklearn.ensemble import HistGradientBoostingRegressor
gbm = HistGradientBoostingRegressor(max_iter=100, random_state=42)
gbm.fit(X_train, y_train)
```

## 12. Classification: logistic baseline, forest and threshold — [+]

**What they do:** Predict a class probability and decide how high the probability must be before labelling an observation positive.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
Xc = pd.DataFrame({"x": [-3, -2, -1, 0, 1, 2, 3, 4]})
yc = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
clf = Pipeline([("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000))])
clf.fit(Xc, yc)
prob = clf.predict_proba(Xc)[:, 1]
labels = (prob >= 0.5).astype(int)
print(accuracy_score(yc, labels), roc_auc_score(yc, prob))  # training score: syntax demo, NOT validation
```

**Extra metric guide:** Precision = fraction of predicted positives that were correct; recall = fraction of actual positives recovered. For a rare class, PR-AUC can be more informative than accuracy.

## 13. Feature importance, buckets and final test discipline — [+]

**What they do:** Identify which predictors a model relies on and whether large target values are especially difficult.

```python
from sklearn.inspection import permutation_importance
imp = permutation_importance(forest, X_valid, y_valid, n_repeats=3,
                             scoring="neg_mean_absolute_error", random_state=42)
print(pd.Series(imp.importances_mean, index=X_valid.columns).sort_values(ascending=False))

errors = pd.DataFrame({"actual": y_valid.to_numpy(), "pred": pred_forest})
errors["bucket"] = pd.qcut(errors["actual"], q=3, duplicates="drop")
print(errors.groupby("bucket", observed=True).apply(
    lambda g: np.mean(np.abs(g["actual"] - g["pred"]))))
```

**Rule:** Use train/CV to fit, validation to choose model settings, and untouched test data **once** to estimate final performance.
