# 10 — Optional tools: know they exist; use only if justified

**Scope:** This file is deliberately **[+] beyond the four-course minimum**, except XGBoost itself is covered in Intermediate ML. It preserves the previous cheatsheet's optional syllabus without making it compulsory.

## 1. Ridge / Lasso — [+]

**What they do:** Fit simple linear relationships while penalising large coefficients (Ridge) or encouraging some to zero (Lasso).

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
ridge = make_pipeline(StandardScaler(), Ridge(alpha=1))
lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.1))
# ridge.fit(X_train, y_train)
```

**Use when:** Small/noisy data; need interpretable, restrained alternative to flexible trees.

## 2. `HuberRegressor` / `RobustScaler` — [+]

**What they do:** Reduce the influence of outliers when fitting a linear model or calculating feature scale.

```python
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import RobustScaler
robust = make_pipeline(RobustScaler(), HuberRegressor())
```

## 3. `permutation_importance` — [+]

**What it does:** Measures how much a fitted model's score deteriorates when one feature is randomly shuffled.

```python
from sklearn.inspection import permutation_importance
# perm = permutation_importance(model, X_valid, y_valid, n_repeats=3, random_state=42)
```

**Trap:** Correlated predictors can substitute for one another, making individual importance look weak.

## 4. PCA / mutual information — [+]

**What they do:** PCA compresses correlated numeric variables into new axes; mutual information screens for statistical dependence beyond linear correlation.

```python
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
# Fit inside CV/pipeline; never fit PCA using test rows.
# pca = PCA(n_components=5)
# scores = mutual_info_regression(X_train_numeric, y_train, random_state=42)
```

## 5. Classification probability/threshold and calibration — [+]

**What they do:** Turn a probability model into a decision, then check whether predicted probabilities resemble actual frequencies.

```python
import pandas as pd
p = pd.Series([.1, .2, .5, .7, .9])
labels = (p >= .4).astype(int)
```

**Pick the threshold using validation and the costs of false positives vs false negatives.**

## 6. SHAP (concept only) — [+]

**What it does:** Allocates a prediction's departure from a reference value across individual input features.

**Contrast:** Permutation importance describes loss of overall predictive performance after shuffling a feature; SHAP explains contributions for particular predictions. Don't install it mid-interview without a clear need.

## 7. Simple trading signal metrics — [+]

**What they do:** Compare a predictive score with the realised economic return and quantify risk/costs.

```python
import numpy as np
r = np.array([.01, -.02, .015, .005])
sharpe = np.sqrt(252) * r.mean() / r.std(ddof=1)
wealth = np.cumprod(1+r)
drawdown = wealth/np.maximum.accumulate(wealth) - 1
print(sharpe, drawdown.min())
```

**Only when relevant:** If the task is generic house-price or demand forecasting, do not invent a trading strategy.

## 8. XGBoost — [K: Intermediate ML] / optional library

**What it does:** Sequentially builds many trees, concentrating on mistakes left by previous trees.

```python
# from xgboost import XGBRegressor
# xgb = XGBRegressor(n_estimators=200, learning_rate=.05, random_state=42)
```

**Fallback:** `sklearn.ensemble.HistGradientBoostingRegressor` avoids package-install risk.

## 9. ElasticNet, polynomial features, statsmodels — [+ optional]

**What they do:** ElasticNet combines Ridge/Lasso penalties; polynomial features add squares/interactions; statsmodels can provide classical regression summaries.

```python
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
model = make_pipeline(StandardScaler(), ElasticNet(alpha=.1, l1_ratio=.5, max_iter=3000))
poly = PolynomialFeatures(degree=2, include_bias=False)
# Fit poly inside train-only pipeline; beware exploding feature count.
# import statsmodels.api as sm    # optional package
# sm.OLS(y_train, sm.add_constant(X_train_numeric)).fit().summary()
```

**Not a priority:** Use only if the task asks for coefficient inference or the basic models are insufficient.
