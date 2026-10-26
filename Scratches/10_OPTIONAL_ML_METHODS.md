# Optional ML Methods

These are secondary tools. Do not reach for them before a baseline + simple model + validation are working.

## RobustScaler

Useful when numeric features have heavy tails/outliers.

```python
from sklearn.preprocessing import RobustScaler

model = Pipeline([
    ("scale", RobustScaler()),
    ("model", Ridge(alpha=1.0))
])
```

`RobustScaler` uses median/interquartile-range-type scaling rather than mean/std.

## MinMaxScaler

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
```

Maps training feature range approximately to `[0,1]`. Usually unnecessary for trees.

## Lasso

```python
from sklearn.linear_model import Lasso

model = Pipeline([
    ("scale", StandardScaler()),
    ("model", Lasso(alpha=0.01, max_iter=5000))
])
```

L1 penalty can set coefficients to zero. Useful for sparse feature selection, but can be unstable with correlated features.

## Elastic Net

```python
from sklearn.linear_model import ElasticNet

model = Pipeline([
    ("scale", StandardScaler()),
    ("model", ElasticNet(
        alpha=0.01,
        l1_ratio=0.5,
        max_iter=5000
    ))
])
```

Combines Ridge and Lasso penalties.

## Polynomial / interaction features

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(
    degree=2,
    include_bias=False,
    interaction_only=True
)
```

Pipeline:

```python
model = Pipeline([
    ("poly", PolynomialFeatures(
        degree=2,
        include_bias=False,
        interaction_only=True
    )),
    ("scale", StandardScaler()),
    ("ridge", Ridge(alpha=10))
])
```

Can explode dimensionality. Use only for small feature sets or targeted interactions.

## Mutual information

Regression:

```python
from sklearn.feature_selection import mutual_info_regression

mi = mutual_info_regression(
    X_train,
    y_train,
    random_state=42
)

mi = pd.Series(
    mi,
    index=X_train.columns
).sort_values(ascending=False)
```

Classification:

```python
from sklearn.feature_selection import mutual_info_classif
```

Captures some nonlinear dependence, but still use validation before selecting features.

## SelectKBest

```python
from sklearn.feature_selection import SelectKBest, f_regression

selector = SelectKBest(
    score_func=f_regression,
    k=10
)
```

Avoid selecting features using test data.

## PCA

```python
from sklearn.decomposition import PCA

model = Pipeline([
    ("scale", StandardScaler()),
    ("pca", PCA(n_components=0.95)),
    ("ridge", Ridge(alpha=1.0))
])
```

Useful with many strongly correlated numeric variables. Costs interpretability.

Explained variance:

```python
pca = PCA().fit(X_scaled)
np.cumsum(pca.explained_variance_ratio_)
```

## Simple decision tree

```python
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=20,
    random_state=42
)
```

Useful for interpreting nonlinear splits, but a single tree often overfits.

## GradientBoostingRegressor

Alternative if `HistGradientBoostingRegressor` is inconvenient:

```python
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
```

## XGBoost — only if installed / clearly worthwhile

```python
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

Do not burn interview time debugging package installation if sklearn boosting is sufficient.

## statsmodels OLS — if coefficient inference matters

```python
import statsmodels.api as sm

X_ols = sm.add_constant(X)
model = sm.OLS(y, X_ols, missing="drop").fit()
print(model.summary())
```

Useful if you specifically care about coefficients, standard errors and classical regression diagnostics. Not required for ordinary predictive modelling.

## Variance Inflation Factor (VIF)

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_num = X_numeric.dropna().copy()

vif = pd.Series(
    [
        variance_inflation_factor(X_num.values, i)
        for i in range(X_num.shape[1])
    ],
    index=X_num.columns
)
```

Can diagnose multicollinearity, but regularisation often makes this less urgent in predictive work.

## KMeans — only if clustering is genuinely useful

```python
from sklearn.cluster import KMeans

km = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

clusters = km.fit_predict(X_scaled)
```

Do not add clustering just to create features unless EDA gives a reason.
