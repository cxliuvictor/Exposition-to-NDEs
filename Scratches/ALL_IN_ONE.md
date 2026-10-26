# Quant Onsite Cheatsheets — All in One

---

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

---

# Pandas Cheatsheet

```python
import pandas as pd
import numpy as np
```

## 1. Inspection

```python
df.head()
df.tail()
df.sample(5, random_state=42)

df.shape
df.columns
df.dtypes
df.info()

df.describe()
df.describe(include="all")

df.isna().sum()
df.isna().mean().sort_values(ascending=False)

df.nunique().sort_values()
df.duplicated().sum()
```

## 2. Selecting / filtering

```python
df["Sales"]
df[["Sales", "Returns"]]

df.loc[df["Sales"] > 1000]

df.loc[
    (df["Sales"] > 1000) & (df["Returns"] < 10),
    ["Region", "Sales", "Returns"]
]

df.iloc[:10, :5]
```

Conditional count:

```python
((df["age"] > 50) & (df["bmi"] < 20)).sum()
```

Fraction satisfying condition:

```python
((df["age"] > 50) & (df["bmi"] < 20)).mean()
```

Readable simple conditions:

```python
df.query("age > 50 and bmi < 20")
```

## 3. Sorting

```python
df.sort_values("Sales", ascending=False)

df.sort_values(
    ["Region", "Sales"],
    ascending=[True, False]
)
```

## 4. GroupBy + `agg`

Representative data:

```python
df = pd.DataFrame({
    "Region": ["North","North","South","South","East","East","West","West"],
    "Sales": [1200,1500,900,1100,2400,3100,1800,1400],
    "Returns": [5,12,0,8,22,15,7,np.nan],
    "Customer_Sat": [4.2,4.5,3.8,4.0,4.9,4.7,3.5,4.1]
})
```

Preferred named aggregation:

```python
summary = (
    df.groupby("Region")
      .agg(
          Total_Sales=("Sales", "sum"),
          Avg_Sales=("Sales", "mean"),
          Max_Returns=("Returns", "max"),
          Missing_Returns=("Returns", lambda s: s.isna().sum())
      )
      .reset_index()
)
```

Multiple functions per column:

```python
summary = df.groupby("Region").agg({
    "Sales": ["sum", "mean"],
    "Returns": ["max", "mean"],
    "Customer_Sat": ["mean", "std"]
})
```

Flatten MultiIndex columns:

```python
summary.columns = [
    "_".join(map(str, col)).strip("_")
    for col in summary.columns
]
summary = summary.reset_index()
```

Whole DataFrame aggregation:

```python
df.agg({
    "Sales": ["sum", "mean"],
    "Returns": ["sum", "mean"]
})
```

## 5. `transform`

Use when output must align with the original rows.

```python
df["Region_Mean_Sales"] = (
    df.groupby("Region")["Sales"]
      .transform("mean")
)

df["Sales_Demeaned"] = (
    df["Sales"] - df["Region_Mean_Sales"]
)
```

Group z-score:

```python
df["Sales_Z"] = (
    df.groupby("Region")["Sales"]
      .transform(lambda s: (s - s.mean()) / s.std())
)
```

- `agg` → one row per group
- `transform` → same length as original DataFrame

## 6. `apply`

Flexible but often slower:

```python
result = (
    df.groupby("Region")
      .apply(lambda g: g.nlargest(1, "Sales"))
)
```

Prefer vectorised/groupby operations when possible.

## 7. `value_counts`

```python
df["Region"].value_counts()
df["Region"].value_counts(normalize=True)
df[["Region","Category"]].value_counts()
```

## 8. Missing values

```python
df.isna().sum()
```

Drop:

```python
df.dropna()
df.dropna(subset=["target"])
```

Fill numeric:

```python
df["Returns"] = df["Returns"].fillna(df["Returns"].median())
```

Fill by group:

```python
df["Returns"] = df["Returns"].fillna(
    df.groupby("Region")["Returns"].transform("median")
)
```

Categorical:

```python
df["Category"] = df["Category"].fillna("Missing")
```

## 9. Duplicates

```python
df.duplicated().sum()
df[df.duplicated(keep=False)]
df.drop_duplicates()
df.drop_duplicates(subset=["customer_id", "date"], keep="last")
```

Check duplicate keys, not only exact duplicate rows.

## 10. Merge

```python
out = left.merge(
    right,
    on="customer_id",
    how="left",
    validate="many_to_one"
)
```

Useful `validate=` values:

```python
"one_to_one"
"one_to_many"
"many_to_one"
"many_to_many"
```

Multiple keys:

```python
out = left.merge(
    right,
    on=["asset", "date"],
    how="left",
    validate="one_to_one"
)
```

Always compare row count before/after:

```python
print(len(left), len(out))
```

## 11. Join

Convenient on index:

```python
left = left.set_index("id")
right = right.set_index("id")
out = left.join(right, how="left")
```

## 12. Concatenate

Rows:

```python
pd.concat([a, b], ignore_index=True)
```

Columns:

```python
pd.concat([a, b], axis=1)
```

## 13. Cross product / Cartesian join

```python
regions = pd.DataFrame({"Region": ["North","South"]})
scenarios = pd.DataFrame({"Scenario": ["Base","Stress","Bull"]})

grid = regions.merge(scenarios, how="cross")
```

Expected rows:

```python
len(regions) * len(scenarios)
```

## 14. `pivot`

Requires unique index-column pairs:

```python
wide = df.pivot(
    index="date",
    columns="asset",
    values="return"
)
```

## 15. `pivot_table`

Safer if duplicates exist because it aggregates:

```python
wide = pd.pivot_table(
    df,
    index="date",
    columns="asset",
    values="return",
    aggfunc="mean"
)
```

## 16. `melt`

Wide → long:

```python
long = wide.reset_index().melt(
    id_vars="date",
    var_name="asset",
    value_name="return"
)
```

## 17. `explode`

```python
df2 = pd.DataFrame({
    "id": [1,2],
    "tags": ["A|B", "B|C|D"]
})

long = (
    df2.assign(tags=df2["tags"].str.split("|"))
       .explode("tags")
)
```

## 18. Datetime

```python
df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["weekday"] = df["date"].dt.dayofweek
df["hour"] = df["date"].dt.hour
```

Difference:

```python
df["days_since"] = (df["date"] - df["previous_date"]).dt.days
```

## 19. Shift

```python
df["lag1"] = df["target"].shift(1)
```

Panel:

```python
df["lag1"] = df.groupby("asset")["return"].shift(1)
```

## 20. Rolling

```python
df["roll20_mean"] = (
    df["return"]
      .shift(1)
      .rolling(20)
      .mean()
)
```

Panel:

```python
df["roll20"] = (
    df.groupby("asset")["return"]
      .transform(lambda s: s.shift(1).rolling(20).mean())
)
```

## 21. Expanding

```python
df["expanding_mean"] = (
    df["target"]
      .shift(1)
      .expanding()
      .mean()
)
```

## 22. Difference / percentage change

```python
df["change"] = df["price"].diff()
df["return"] = df["price"].pct_change()
```

Panel:

```python
df["return"] = df.groupby("asset")["price"].pct_change()
```

## 23. Ranking

```python
df["rank_pct"] = df["score"].rank(pct=True)
```

Cross-sectional:

```python
df["cs_rank"] = df.groupby("date")["signal"].rank(pct=True)
```

## 24. Binning

Fixed bins:

```python
df["age_bucket"] = pd.cut(
    df["age"],
    bins=[0,30,40,50,60,100]
)
```

Equal-frequency:

```python
df["decile"] = pd.qcut(df["score"], 10, duplicates="drop")
```

## 25. Strings

```python
df["name"].str.lower()
df["name"].str.contains("abc", case=False, na=False)
df["name"].str.replace("old", "new", regex=False)
df["name"].str.extract(r"(\d{4})")
df["tags"].str.split("|")
```

## 26. Map / replace

```python
mapping = {"jan":1, "feb":2, "mar":3}
df["month_num"] = df["month"].map(mapping)
```

```python
df["status"] = df["status"].replace({"Y":1, "N":0})
```

## 27. Key checks

```python
df["id"].is_unique
```

```python
df.duplicated(subset=["asset","date"]).sum()
```

```python
df.memory_usage(deep=True)
```

---

# NumPy Cheatsheet

```python
import numpy as np
```

## Arrays / shape

```python
x = np.array([1,2,3])
X = df[["x1","x2","x3"]].to_numpy()

X.shape
X.ndim
X.size
```

## Indexing

```python
X[0]
X[:, 0]
X[:10, :3]

mask = X[:,0] > 0
X[mask]
```

## Aggregations and axes

```python
X.mean(axis=0)   # column means
X.mean(axis=1)   # row means

np.nanmean(X, axis=0)
np.nanstd(X, axis=0)
```

## Broadcasting / standardisation

```python
mu = np.nanmean(X, axis=0)
sd = np.nanstd(X, axis=0)
Z = (X - mu) / sd
```

Row-wise:

```python
mu_row = X.mean(axis=1, keepdims=True)
sd_row = X.std(axis=1, keepdims=True)
Z_row = (X - mu_row) / sd_row
```

## `where`

```python
signal = np.where(returns > 0, 1, -1)
```

Three-way:

```python
position = np.where(
    signal > 1,
    1,
    np.where(signal < -1, -1, 0)
)
```

## `select`

```python
conditions = [score < -1, score > 1]
choices = [-1, 1]
position = np.select(conditions, choices, default=0)
```

## Clip / quantiles

```python
lo, hi = np.quantile(x, [0.01, 0.99])
x_clip = np.clip(x, lo, hi)
```

Column quantiles:

```python
np.quantile(X, [0.01,0.5,0.99], axis=0)
```

## Sorting / ranking

```python
idx = np.argsort(x)
top_idx = np.argsort(-x)[:10]
```

## Difference / cumulative

```python
np.diff(x)
np.cumsum(x)
np.cumprod(1 + returns)
```

## Correlation / covariance

```python
np.corrcoef(X, rowvar=False)
np.cov(X, rowvar=False)
```

## Reproducible randomness

```python
rng = np.random.default_rng(42)
x = rng.normal(0, 1, size=1000)
```

## Log / sigmoid / tanh

```python
log_x = np.log1p(x)
sigmoid = 1 / (1 + np.exp(-x))
tanh_x = np.tanh(x)
```

## Cyclic encoding

```python
hour_sin = np.sin(2*np.pi*hour/24)
hour_cos = np.cos(2*np.pi*hour/24)
```

## Interactions

```python
interaction = x1 * x2
ratio = x1 / (x2 + 1e-8)
spread = x1 - x2
```

## Cross-sectional z-score

For matrix `(time, assets)`:

```python
mu = np.nanmean(signal, axis=1, keepdims=True)
sd = np.nanstd(signal, axis=1, keepdims=True)
z = (signal - mu) / sd
```

## Sharpe

```python
sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns, ddof=1)
```

## Max drawdown

```python
wealth = np.cumprod(1 + returns)
peak = np.maximum.accumulate(wealth)
drawdown = wealth / peak - 1
max_drawdown = drawdown.min()
```

## Common array mistake

If `x` is an array, do not write:

```python
if x > 0:
    ...
```

Use a boolean mask:

```python
mask = x > 0
```

---

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

---

# Cleaning + Feature Engineering Cheatsheet

## Compact audit table

```python
audit = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "missing_n": df.isna().sum(),
    "missing_pct": df.isna().mean(),
    "nunique": df.nunique(dropna=False)
}).sort_values("missing_pct", ascending=False)
```

## Constants

```python
constant_cols = [
    c for c in df.columns
    if df[c].nunique(dropna=False) <= 1
]
```

## Near constants

```python
near_constant = []

for c in df.columns:
    top_freq = df[c].value_counts(normalize=True, dropna=False).iloc[0]
    if top_freq > 0.99:
        near_constant.append(c)
```

## Infinite values

```python
num_cols = df.select_dtypes(include=np.number).columns

inf_counts = {
    c: np.isinf(df[c]).sum()
    for c in num_cols
}
```

Replace:

```python
df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
```

## Sentinel values

```python
(df["feature"] == -999).sum()
```

```python
df.loc[df["feature"] == -999, "feature"] = np.nan
```

## Impossible values

```python
df.loc[df["age"] < 0]
df.loc[df["probability"] > 1]
df.loc[df["price"] < 0]
```

Do not automatically delete: determine whether it is a real extreme, a coding error, a sentinel, or a unit problem.

## Outliers

```python
df["x"].quantile([0.01,0.05,0.5,0.95,0.99])
```

IQR:

```python
q1 = df["x"].quantile(0.25)
q3 = df["x"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
```

Winsorise:

```python
lo, hi = df["x"].quantile([0.01,0.99])
df["x_clip"] = df["x"].clip(lo, hi)
```

## Log transform

```python
df["log_x"] = np.log1p(df["x"])
```

Use for positive/right-skewed variables.

## Cyclic time

Hour:

```python
df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
```

Weekday:

```python
df["dow_sin"] = np.sin(2*np.pi*df["weekday"]/7)
df["dow_cos"] = np.cos(2*np.pi*df["weekday"]/7)
```

Month:

```python
df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
```

## Smooth event proximity

```python
d = df["days_to_event"]
df["event_decay"] = np.exp(-d / 3)
```

Sigmoid-style:

```python
df["event_proximity"] = 1 / (1 + np.exp((d - 2) / 1.5))
```

## Rolling physical effects

```python
df["temp_roll_6"] = (
    df["temp"]
      .shift(1)
      .rolling(6)
      .mean()
)
```

Use when the target responds with inertia rather than only to the instantaneous observation.

## Interactions

```python
df["temp_x_hum"] = df["temp"] * df["humidity"]
```

```python
df["momentum_per_vol"] = df["momentum"] / (df["vol"] + 1e-8)
```

## Rank transforms

Global:

```python
df["rank_x"] = df["x"].rank(pct=True)
```

Cross-sectional:

```python
df["rank_x"] = df.groupby("date")["x"].rank(pct=True)
```

## Missing indicators

```python
df["x_missing"] = df["x"].isna().astype(int)
```

Then impute `x`.

## Correlated variables

```python
corr = df[numeric_cols].corr().abs()
```

Possible responses:

- keep both with Ridge
- remove one for simplicity
- retain both for a tree model
- use domain knowledge

Do not automatically delete every highly correlated variable.

## Leakage audit

For every feature ask:

> Could I know this value at prediction time?

Common leakage:

- future return
- post-event information
- target-derived aggregate
- final outcome data used before the event
- preprocessing fit on full data
- feature selection using the test target

## Train-only preprocessing

Manual pattern:

```python
median = train["x"].median()
train["x"] = train["x"].fillna(median)
valid["x"] = valid["x"].fillna(median)
test["x"] = test["x"].fillna(median)
```

Prefer sklearn Pipelines when modelling.

---

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

---

# Time-Series Forecasting Cheatsheet

## Sort

```python
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
```

## Create future target

```python
df["target_next"] = df["target"].shift(-1)
```

At time `t`, features should only use information known by `t`.

## Lags

```python
df["lag1"] = df["target"].shift(1)
df["lag24"] = df["target"].shift(24)
df["lag168"] = df["target"].shift(168)
```

## Rolling statistics

Past-only:

```python
df["roll24_mean"] = (
    df["target"]
      .shift(1)
      .rolling(24)
      .mean()
)

df["roll24_std"] = (
    df["target"]
      .shift(1)
      .rolling(24)
      .std()
)
```

## Exponentially weighted mean

```python
df["ewm24"] = (
    df["target"]
      .shift(1)
      .ewm(span=24)
      .mean()
)
```

## Cyclic time

```python
df["hour"] = df["date"].dt.hour

df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
```

```python
df["weekday"] = df["date"].dt.dayofweek

df["dow_sin"] = np.sin(2*np.pi*df["weekday"]/7)
df["dow_cos"] = np.cos(2*np.pi*df["weekday"]/7)
```

## Chronological 80/10/10

```python
n = len(df)
i1 = int(0.8*n)
i2 = int(0.9*n)

train = df.iloc[:i1]
valid = df.iloc[i1:i2]
test = df.iloc[i2:]
```

## Persistence baseline

```python
pred = valid["lag1"]
```

## Seasonal naive

Daily hourly seasonality:

```python
pred = valid["lag24"]
```

Weekly:

```python
pred = valid["lag168"]
```

Always compare ML against these if they make sense.

## TimeSeriesSplit

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, valid_idx in tscv.split(X):
    X_tr = X.iloc[train_idx]
    X_va = X.iloc[valid_idx]
```

## Rolling temperature / physical inertia

```python
df["temp_roll6"] = (
    df["temp"]
      .shift(1)
      .rolling(6)
      .mean()
)
```

Use when the target responds with inertia.

## Error by hour

```python
tmp = pd.DataFrame({
    "hour": valid["hour"],
    "y": y_valid,
    "pred": pred
})

tmp["error"] = tmp["y"] - tmp["pred"]

rmse_by_hour = (
    tmp.groupby("hour")
       .apply(lambda g: np.sqrt(np.mean(g["error"]**2)))
)
```

## Error by target quantile

```python
tmp["bucket"] = pd.qcut(tmp["y"], 5, duplicates="drop")
```

## Leakage trap

Potentially wrong:

```python
df["roll24"] = df["target"].rolling(24).mean()
```

if current target is unavailable at forecast time.

Safer:

```python
df["roll24"] = df["target"].shift(1).rolling(24).mean()
```

## Good onsite workflow

```text
persistence
→ seasonal naive
→ Ridge
→ HistGradientBoosting
→ error by hour/period/target bucket
→ frozen OOS
```

---

# Statistics + Diagnostics Cheatsheet

## Pearson correlation

Linear association:

```python
df["x"].corr(df["y"], method="pearson")
```

## Spearman correlation

Monotonic/rank association:

```python
df["x"].corr(df["y"], method="spearman")
```

## Correlation matrix

```python
corr = df.select_dtypes(include=np.number).corr()
```

## Conditional means / decile response

```python
df["bucket"] = pd.qcut(df["feature"], 10, duplicates="drop")

response = (
    df.groupby("bucket", observed=True)["target"]
      .agg(["mean","std","count"])
)
```

A smooth monotonic response can be more informative than one correlation number.

## Standard error of mean

```python
x = df["target"].dropna()
sem = x.std(ddof=1) / np.sqrt(len(x))
```

## Approximate 95% confidence interval

```python
mean = x.mean()
sem = x.std(ddof=1) / np.sqrt(len(x))

lower = mean - 1.96 * sem
upper = mean + 1.96 * sem
```

## scipy tests

```python
from scipy import stats
```

Independent Welch t-test:

```python
stats.ttest_ind(
    group_a,
    group_b,
    equal_var=False,
    nan_policy="omit"
)
```

Paired t-test:

```python
stats.ttest_rel(before, after, nan_policy="omit")
```

Mann-Whitney:

```python
stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
```

In an onsite, effect size + validation often matter more than a p-value.

## Bootstrap mean difference

```python
rng = np.random.default_rng(42)
diffs = []

for _ in range(5000):
    a = rng.choice(group_a, size=len(group_a), replace=True)
    b = rng.choice(group_b, size=len(group_b), replace=True)
    diffs.append(a.mean() - b.mean())

np.quantile(diffs, [0.025,0.975])
```

## Residuals

```python
resid = y - pred

resid.mean()
resid.std()
np.quantile(resid, [0.01,0.05,0.5,0.95,0.99])
```

## Heteroskedasticity clue

```python
plt.scatter(pred, resid, alpha=0.3)
plt.axhline(0, linestyle="--")
```

Fan-shaped residual spread suggests variance changes with prediction level.

## Error by target bucket

```python
tmp = pd.DataFrame({"y": y, "pred": pred})
tmp["bucket"] = pd.qcut(tmp["y"], 5, duplicates="drop")

def rmse(g):
    return np.sqrt(np.mean((g["y"] - g["pred"]) ** 2))

bucket_rmse = tmp.groupby("bucket", observed=True).apply(rmse)
```

## Predicted-decile response

```python
tmp["pred_decile"] = pd.qcut(tmp["pred"], 10, duplicates="drop")

summary = (
    tmp.groupby("pred_decile", observed=True)["y"]
       .agg(["mean","count"])
)
```

Useful for ranking/trading/prioritisation.

## Classification calibration-style check

```python
tmp = pd.DataFrame({"prob": prob, "y": y_valid})
tmp["decile"] = pd.qcut(tmp["prob"], 10, duplicates="drop")

cal = (
    tmp.groupby("decile", observed=True)
       .agg(
           mean_prob=("prob","mean"),
           realised_rate=("y","mean"),
           n=("y","size")
       )
)
```

## Robustness by period

```python
tmp = tmp.sort_values("date").reset_index(drop=True)
tmp["period"] = pd.qcut(
    tmp.index,
    3,
    labels=["early","middle","late"]
)
```

Then recompute your metric by period.

Ask whether performance is concentrated in one regime.

---

# Jupyter + Debugging Cheatsheet

## Environment

```python
import os
import sys

print(sys.version)
print(sys.executable)
print(os.getcwd())
```

Files:

```python
os.listdir()
```

## Package versions

```python
import numpy as np
import pandas as pd
import sklearn

print(np.__version__)
print(pd.__version__)
print(sklearn.__version__)
```

## Install package

Inside Jupyter:

```python
%pip install seaborn
```

Prefer `%pip` to `!pip`. Restart kernel if required.

Do not install packages unless there is real benefit.

## Timing

```python
%timeit np.mean(x)
```

## Variables currently defined

```python
%whos
```

## CSV delimiter / encoding problems

```python
df = pd.read_csv("file.csv")
```

Semicolon:

```python
df = pd.read_csv("file.csv", sep=";")
```

Encoding:

```python
df = pd.read_csv("file.csv", encoding="latin1")
```

## KeyError

```python
df.columns.tolist()
```

Strip whitespace:

```python
df.columns = df.columns.str.strip()
```

## Date parsing failed

```python
dates = pd.to_datetime(df["date"], errors="coerce")
```

Inspect bad values:

```python
df.loc[dates.isna(), "date"]
```

## Numeric stored as string

```python
df["x"] = pd.to_numeric(df["x"], errors="coerce")
```

## Shape mismatch

```python
print(
    X_train.shape,
    X_valid.shape,
    y_train.shape,
    y_valid.shape
)
```

## Merge unexpectedly increased rows

```python
print(left.shape, right.shape)
print(left["id"].duplicated().sum())
print(right["id"].duplicated().sum())
```

Then use the expected `validate=` cardinality.

## Model rejects NaNs

```python
X_train.isna().sum().sort_values(ascending=False).head()
```

Use:

```python
SimpleImputer(strategy="median")
```

## Model rejects strings

```python
X_train.select_dtypes(include="object").columns
```

Use `OneHotEncoder` or a suitable categorical encoding.

## Infinite values

```python
np.isinf(
    X_train.select_dtypes(include=np.number)
).sum()
```

Replace:

```python
X = X.replace([np.inf,-np.inf], np.nan)
```

## Convergence warning

For logistic/Huber, try scaling and:

```python
max_iter=2000
```

## Memory issue

Sample:

```python
df_small = df.sample(100000, random_state=42)
```

Keep relevant columns only:

```python
df = df[needed_cols].copy()
```

Avoid giant many-to-many merges.

## Restart discipline

If notebook state becomes confusing:

1. restart kernel
2. run all
3. fix first failing cell
4. repeat

Final notebook should reproduce top-to-bottom.

---

# Common Onsite Recipes

## 1. How many observations satisfy X and Y?

```python
count = (
    (df["age"] > 50)
    &
    (df["bmi"] < 20)
).sum()
```

## 2. Summarise target by category

```python
summary = (
    df.groupby("category")
      .agg(
          n=("target","size"),
          target_mean=("target","mean"),
          target_std=("target","std")
      )
      .sort_values("target_mean", ascending=False)
)
```

## 3. Strongest simple relationships

```python
num = df.select_dtypes(include=np.number)

pearson = (
    num.corr()["target"]
       .drop("target")
       .sort_values(key=np.abs, ascending=False)
)

spearman = (
    num.corr(method="spearman")["target"]
       .drop("target")
       .sort_values(key=np.abs, ascending=False)
)
```

## 4. Clean modelling table

```python
target = "y"

X = df.drop(columns=[target])
y = df[target]

numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()
```

## 5. Regression benchmark

```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

baseline = np.repeat(y_train.mean(), len(y_valid))
baseline_rmse = mean_squared_error(y_valid, baseline) ** 0.5

ridge = Pipeline([
    ("scale", StandardScaler()),
    ("model", Ridge(alpha=1))
])

ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_valid)
ridge_rmse = mean_squared_error(y_valid, ridge_pred) ** 0.5
```

## 6. Classification benchmark

```python
clf.fit(X_train, y_train)
prob = clf.predict_proba(X_valid)[:,1]

print(roc_auc_score(y_valid, prob))
print(average_precision_score(y_valid, prob))
```

## 7. Does a nonlinear model add anything?

```python
results = pd.DataFrame([
    {"Model":"Baseline", "RMSE":baseline_rmse},
    {"Model":"Ridge", "RMSE":ridge_rmse},
    {"Model":"GBM", "RMSE":gbm_rmse}
])

results.sort_values("RMSE")
```

Then ask whether improvement is:

- meaningful in size
- stable across folds/time
- present in important target buckets
- worth the extra complexity

## 8. Try plausible transforms

```python
x = df["feature"]

candidates = pd.DataFrame({
    "raw": x,
    "log1p": np.log1p(x.clip(lower=0)),
    "sqrt": np.sqrt(x.clip(lower=0)),
    "rank": x.rank(pct=True)
})

for c in candidates.columns:
    print(
        c,
        candidates[c].corr(df["target"], method="spearman")
    )
```

Do not choose a transform solely from in-sample correlation. Validate it.

## 9. Time-series forecast setup

```python
df = df.sort_values("date").reset_index(drop=True)

df["target_next"] = df["target"].shift(-1)
df["lag1"] = df["target"].shift(1)
df["lag24"] = df["target"].shift(24)
df["roll24"] = df["target"].shift(1).rolling(24).mean()

model_df = df.dropna().copy()
```

Then split chronologically.

## 10. Prediction deciles

```python
tmp = pd.DataFrame({
    "pred": pred,
    "actual": y_valid
})

tmp["decile"] = pd.qcut(
    tmp["pred"],
    10,
    labels=False,
    duplicates="drop"
)

summary = (
    tmp.groupby("decile")
       .agg(
           n=("actual","size"),
           mean_pred=("pred","mean"),
           mean_actual=("actual","mean")
       )
)
```

## 11. Trade top predicted opportunities

```python
tmp = pd.DataFrame({
    "pred": pred,
    "pnl": realised_pnl
})

threshold = tmp["pred"].quantile(0.8)
trades = tmp[tmp["pred"] >= threshold]

print(trades["pnl"].sum())
print(trades["pnl"].mean())
print((trades["pnl"] > 0).mean())
```

Select threshold on validation only.

## 12. Transaction costs

```python
gross = position * future_return

turnover = np.abs(
    position - np.roll(position, 1)
)
turnover[0] = np.abs(position[0])

cost_bps = 10
net = gross - turnover * cost_bps / 10000
```

## 13. Final model table + conclusion

```python
results = pd.DataFrame(results).sort_values("RMSE")
display(results)
```

Then write concise markdown:

```markdown
## Conclusion

- Best validation model: ...
- Improvement versus baseline: ...
- Main useful features: ...
- Weakest region: ...
- Main concern: ...
- Next step: ...
```

## 14. If running out of time

Stop adding features/models.

Do:

1. baseline
2. one simple model
3. one nonlinear model
4. one comparison table
5. actual-vs-predicted
6. one error slice
7. conclusions

That is much better than five unfinished models.

---

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
