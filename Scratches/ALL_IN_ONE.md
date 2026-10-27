# All-in-one: Kaggle-aligned Onsite Cheatsheets

**Use Ctrl+F to find a method.** `[K]` = aligned with named Kaggle courses; `[+]` = extra for onsite.

- 00_ONSITE_WORKFLOW
- 01_PANDAS
- 02_NUMPY
- 03_VISUALISATION
- 04_CLEANING_FEATURE_ENGINEERING
- 05_SKLEARN_MODELLING_VALIDATION
- 06_TIME_SERIES
- 07_STATS_DIAGNOSTICS
- 08_JUPYTER_DEBUGGING
- 09_COMMON_RECIPES
- 10_OPTIONAL_ML_METHODS

---

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

---

# 01 — Pandas: course-style essentials, then necessary extras

```python
import pandas as pd
import numpy as np

sales = pd.DataFrame({
    "region": ["North", "North", "South", "South", "East"],
    "revenue": [120, 150, 90, 110, 240],
    "returns": [5, np.nan, 0, 8, 2],
    "category": ["A", "B", "A", "B", "A"]
})
```

## 1. `DataFrame`, `Series`, `read_csv`, `to_csv` — [K]

**What they do:** A Series is one labelled column, a DataFrame is a table, and CSV methods load/save that table.

```python
pd.Series([2, 4, 6], name="value")
pd.DataFrame({"x": [1, 2], "y": [3, 4]})
# sales = pd.read_csv("sales.csv", index_col=0)  # only if first column really is the index
sales.to_csv("sales_output.csv", index=False)
```

## 2. `head`, `shape`, `info`, `describe`, `value_counts`, `unique` — [K]

**What they do:** Quickly display rows, sizes, types, summary statistics, and the most frequent categories.

```python
sales.head()
sales.shape
sales.info()
sales["revenue"].describe()
sales["region"].unique()
sales["region"].value_counts()
sales["region"].value_counts(normalize=True)  # proportions
```

## 3. `iloc`, `loc`, boolean filtering — [K]

**What they do:** `iloc` selects by integer position; `loc` selects by labels or conditions.

```python
sales.iloc[:3, :2]                                     # first 3 rows, first 2 cols
sales.loc[:, ["region", "revenue"]]                    # named columns
sales.loc[(sales["revenue"] > 100) & (sales["returns"].fillna(0) < 5)]
((sales["revenue"] > 100) & (sales["category"] == "A")).sum()  # count
```

**Trap:** Use parentheses around comparisons and `&` / `|`, not Python `and` / `or`, with Series.

## 4. Column assignment, `rename`, `astype`, `set_index` — [K]

**What they do:** Create a field, rename fields, convert types and make a column the row index.

```python
sales["net"] = sales["revenue"] - sales["returns"].fillna(0)
sales = sales.rename(columns={"revenue": "sales_amount"})
sales["sales_amount"] = sales["sales_amount"].astype(float)
sales_by_region = sales.set_index("region")
```

## 5. Summary methods and `map` — [K]

**What they do:** Aggregate a column or transform each value using a simple rule.

```python
sales["sales_amount"].mean()
sales["sales_amount"].median()
sales["sales_amount"].max()
sales["category"].map({"A": 1, "B": 0})          # translate categories
sales["sales_amount"].map(lambda x: x / 100)      # one value at a time
```

## 6. `apply(axis=1)` — [K]

**What it does:** Computes a value from *multiple columns in a row* when direct arithmetic is awkward.

```python
sales.apply(lambda row: row["sales_amount"] - row["returns"]
            if pd.notna(row["returns"]) else row["sales_amount"], axis=1)
# Prefer direct vectorised arithmetic for simple calculations.
```

## 7. `groupby`, `agg` — [K]

**What they do:** Split a table into groups, then produce one summary for each group.

```python
sales.groupby("region")["sales_amount"].mean()
summary = sales.groupby("region").agg(
    n=("sales_amount", "size"),
    total=("sales_amount", "sum"),
    average=("sales_amount", "mean"),
    missing_returns=("returns", lambda s: s.isna().sum())
).reset_index()
summary
```

**Alternative:** `sales.groupby("region").agg({"sales_amount": ["sum", "mean"]})` is compact but creates two-level column names.

## 8. `groupby(...).apply`, `transform` — [K]/[+]

**What they do:** `apply` runs custom logic per group; `transform` returns a value for *each original row*, rather than a smaller summary table.

```python
sales.groupby("region").apply(lambda g: g["sales_amount"].max())
sales["relative_sales"] = (
    sales["sales_amount"] - sales.groupby("region")["sales_amount"].transform("mean")
)
```

**Shortcut:** `agg` = one row per group; `transform` = same number of rows as before.

## 9. `sort_values`, `sort_index`, `idxmax` — [K]

**What they do:** Order observations and locate the row label of an extreme value.

```python
sales.sort_values(["region", "sales_amount"], ascending=[True, False])
sales.sort_index()
sales.loc[sales["sales_amount"].idxmax()]
```

## 10. `isna`, `notna`, `fillna`, `dropna`, `replace` — [K]

**What they do:** Detect, fill, remove or recode missing/bad entries.

```python
sales.isna().sum()
sales["returns"].fillna(sales["returns"].median())
sales.dropna(subset=["returns"])
sales["category"].replace({"A": "Core", "B": "Other"})
```

**Modelling trap:** Fit imputation values on training data, not on the full dataset (see `05`).

## 11. `concat` — [K]

**What it does:** Stacks tables with matching columns vertically (or side-by-side with `axis=1`).

```python
north = sales[sales["region"] == "North"]
other = sales[sales["region"] != "North"]
combined = pd.concat([north, other], ignore_index=True)
```

## 12. `join` — [K]

**What it does:** Combines tables by their row indexes. This is the simple index-based combining style emphasised in Kaggle Pandas.

```python
left = pd.DataFrame({"id": [1, 2], "sales": [100, 150]}).set_index("id")
right = pd.DataFrame({"id": [1, 2], "segment": ["A", "B"]}).set_index("id")
left.join(right, how="left")
```

**Rule:** `concat` stacks; `join` matches indexes.

## 13. `merge(validate=...)` — [+ onsite-critical]

**What it does:** Joins on *ordinary columns* rather than manually setting indexes; `validate` catches unexpected duplicate matches.

```python
left = pd.DataFrame({"id": [1, 2, 2], "sales": [20, 30, 40]})
right = pd.DataFrame({"id": [1, 2], "segment": ["A", "B"]})
result = left.merge(right, on="id", how="left", validate="many_to_one")
assert len(result) == len(left)
```

**When to use:** Combining transaction rows with a unique customer/asset lookup table. Although Kaggle introduces `join` as a convenient alternative, explicit keys and validation are often safer in an unfamiliar dataset.

## 14. `pivot_table`, `melt`, `explode` — [+ onsite-useful]

**What they do:** `pivot_table` turns categories into columns, `melt` reverses wide-to-long, and `explode` separates a list into several rows.

```python
sales.pivot_table(index="region", columns="category", values="sales_amount", aggfunc="mean")
wide = pd.DataFrame({"id": [1, 2], "Jan": [10, 20], "Feb": [11, 22]})
wide.melt(id_vars="id", var_name="month", value_name="sales")
tags = pd.DataFrame({"id": [1, 2], "tags": ["a|b", "b|c"]})
tags.assign(tags=tags["tags"].str.split("|")).explode("tags")
```

**Trap:** `pivot` requires unique row/column keys; `pivot_table` can aggregate repeated keys.

## 15. `pd.to_datetime`, `.dt`, `shift`, `rolling`, `qcut`, `rank` — [+ because forecasting may appear]

**What they do:** Convert dates, extract calendar properties, create past values, summarise recent history and make bins/ranks.

```python
ts = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=40), "y": np.arange(40.0)})
ts["month"] = ts["date"].dt.month
ts["lag1"] = ts["y"].shift(1)
ts["mean7_past"] = ts["y"].shift(1).rolling(7).mean()
ts["quartile"] = pd.qcut(ts["y"], 4, labels=False)
ts["rank_pct"] = ts["y"].rank(pct=True)
```

**Trap:** `shift(1)` is needed only when the current observation would otherwise be unavailable at prediction time. Define *when* the forecast is made before adding lags.

## 16. `pd.cut` vs `pd.qcut` — [+]

**What they do:** `cut` makes fixed-size/rule-based intervals; `qcut` makes roughly equal-count bins.

```python
pd.cut(sales["sales_amount"], bins=[0, 100, 200, 1000])
pd.qcut(sales["sales_amount"], q=2, duplicates="drop")
```

## 17. Cross join / `pct_change` / `expanding` — [+ rarely necessary]

**What they do:** Cross joins create every combination; percentage change converts price levels to returns; expanding windows summarise all earlier rows.

```python
a = pd.DataFrame({"region": ["N", "S"]})
b = pd.DataFrame({"scenario": ["base", "stress"]})
a.merge(b, how="cross")  # if pandas is older, use a temporary key column instead
prices = pd.Series([100., 103., 102., 105.])
prices.pct_change()
prices.shift(1).expanding().mean()
```

**When useful:** Only if a prompt asks for all group-scenario combinations, returns, or historical cumulative statistics.

---

# 02 — NumPy: only the array operations that make pandas/ML easier

**Scope:** NumPy methods are an **addition [+]**; the four specified Kaggle courses use NumPy but do not teach a separate dedicated NumPy syllabus.

```python
import numpy as np
x = np.array([2., 4., 8., 16.])
X = np.array([[2., 4.], [6., 8.], [10., 12.]])
```

## 1. `array`, `.shape`, slicing — [+]

**What they do:** Turn values into numerical arrays, check their dimensions and select parts of them.

```python
X.shape     # (3 rows, 2 columns)
X[:, 0]     # all rows, first column
X[1, :]     # second row
```

## 2. `mean`, `std`, `nanmean`, `quantile` — [+]

**What they do:** Summarise numerical values, optionally ignoring missing numbers.

```python
X.mean(axis=0)                       # average of each column
X.mean(axis=1)                       # average of each row
np.nanmean(np.array([2., np.nan, 6.]))
np.quantile(x, [0.1, 0.5, 0.9])
```

**Memory aid:** `axis=0` aggregates down the rows; `axis=1` aggregates across columns.

## 3. Masks and `np.where` — [+]

**What they do:** Select elements matching a condition or assign two alternative values without a loop.

```python
x[x > 5]
np.where(x > 5, "large", "small")
```

## 4. Broadcasting — [+]

**What it does:** Automatically stretches column-level numbers across every row, making vectorised calculations easy.

```python
mu, sd = X.mean(axis=0), X.std(axis=0)
Z = (X - mu) / sd             # standardise columns
```

**Trap:** Check `.shape` if a subtraction gives unexpected results.

## 5. `clip`, `log1p`, `sqrt` — [+]

**What they do:** Compress extreme values and transform positive skewed variables.

```python
np.clip(x, 3, 12)
np.log1p(x)                  # computes log(1+x), safe at x=0
np.sqrt(x)
```

**Decision rule:** Log is often sensible for counts/volumes; do not apply it blindly to negative observations.

## 6. `argsort`, `cumsum`, `diff` — [+]

**What they do:** Find the order of observations, accumulate values and compute changes.

```python
np.argsort(-x)[:2]           # indices of top two
np.cumsum(x)
np.diff(x)
```

## 7. `corrcoef`, random generator — [+]

**What they do:** Measure simple linear relationships and create reproducible random samples.

```python
np.corrcoef(X, rowvar=False)   # correlations between columns
rng = np.random.default_rng(42)
noise = rng.normal(size=100)
```

## 8. Sine/cosine, sigmoid and interactions — [+]

**What they do:** Represent circular time, flatten out an effect that saturates, or let two features influence one another.

```python
hour = np.array([23, 0, 1])
sin_hour = np.sin(2*np.pi*hour/24)
cos_hour = np.cos(2*np.pi*hour/24)
z = np.array([-2., 0., 2.])
sigmoid = 1/(1 + np.exp(-z))
interaction = X[:, 0] * X[:, 1]
```

## 9. NaNs, infinities and safe division — [+]

**What they do:** Spot invalid numeric values before sklearn fitting.

```python
bad = np.array([1., np.nan, np.inf])
np.isnan(bad)
np.isinf(bad)
np.isfinite(bad)
ratio = X[:, 0] / np.maximum(X[:, 1], 1e-8)
```

**Trap:** Adding `1e-8` is not a principled treatment of genuinely zero or negative denominators. Inspect the data first.

## 10. `np.select`, `cumprod` — [+]

**What they do:** `select` handles three or more conditional outcomes; `cumprod` compounds a sequence of gross returns.

```python
score = np.array([-2., 0., 3.])
np.select([score < -1, score > 1], [-1, 1], default=0)
returns = np.array([.01, -.02, .015])
np.cumprod(1 + returns)
```

---

# 03 — Visualisation: simplest useful plots

**Scope:** The four named courses use plots but do not include Kaggle's separate Data Visualization course. These are **[+] onsite essentials**, kept deliberately short.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)
df = pd.DataFrame({
    "date": pd.date_range("2025-01-01", periods=100),
    "x": rng.normal(size=100),
    "y": rng.normal(size=100),
    "group": rng.choice(["A", "B"], size=100)
})
df["target"] = 2*df["x"] + rng.normal(size=100)
```

## 1. `hist` and `df.hist` — [+]

**What they do:** Show a variable's distribution, including skew, gaps, extreme values and possible data errors.

```python
fig, ax = plt.subplots(figsize=(6, 3))
ax.hist(df["target"], bins=25)
ax.set(title="Target distribution", xlabel="Target", ylabel="Count")
plt.show()

df[["x", "y", "target"]].hist(figsize=(9, 5), bins=20)
plt.tight_layout(); plt.show()
```

## 2. `plot` — [+]

**What it does:** Shows whether a variable changes over time or has unusual periods.

```python
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(df["date"], df["target"])
ax.set(title="Target through time", xlabel="Date", ylabel="Target")
plt.show()
```

## 3. `scatter` — [+]

**What it does:** Shows whether a feature and target tend to move together or show a nonlinear shape.

```python
fig, ax = plt.subplots(figsize=(5, 3))
ax.scatter(df["x"], df["target"], alpha=0.45)
ax.set(xlabel="Feature x", ylabel="Target")
plt.show()
```

## 4. Bar chart of group means — [+]

**What it does:** Compares average target values across categories.

```python
df.groupby("group")["target"].mean().plot.bar(title="Mean target by group")
plt.tight_layout(); plt.show()
```

## 5. `subplots` — [+]

**What it does:** Puts several related plots in one compact figure rather than making many separate figures.

```python
fig, axes = plt.subplots(1, 2, figsize=(9, 3))
axes[0].hist(df["x"], bins=20)
axes[1].scatter(df["x"], df["target"], alpha=0.4)
axes[0].set_title("Feature distribution")
axes[1].set_title("Feature vs target")
fig.tight_layout(); plt.show()
```

## 6. Correlation heatmap — [+]

**What it does:** Shows which numerical columns have strong linear relationships, including pairs of redundant predictors.

```python
corr = df.select_dtypes(include=np.number).corr()
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(corr, vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticklabels(corr.columns)
fig.colorbar(im, ax=ax)
fig.tight_layout(); plt.show()
```

## 7. `actual` vs `predicted` and residuals — [+]

**What they do:** Show bias, systematic mistakes and whether errors grow for large targets.

```python
actual = df["target"].to_numpy()
predicted = 1.7 * df["x"].to_numpy()
fig, axes = plt.subplots(1, 2, figsize=(9, 3))
axes[0].scatter(actual, predicted, alpha=0.4)
axes[0].set(xlabel="Actual", ylabel="Predicted")
axes[1].scatter(predicted, actual-predicted, alpha=0.4)
axes[1].axhline(0, ls="--", color="grey")
axes[1].set(xlabel="Predicted", ylabel="Residual")
fig.tight_layout(); plt.show()
```

## 8. Seaborn if already installed — [+ optional]

**What it does:** Offers compact alternatives to matplotlib for standard histograms and heatmaps.

```python
# import seaborn as sns
# sns.histplot(data=df, x="target", bins=30)
# sns.heatmap(df[["x", "y", "target"]].corr(), annot=True)
```

**Rule:** Do not install another plotting library under time pressure just to make the figures prettier.

---

# 04 — Cleaning and feature engineering

**Course basis:** Pandas lesson 5 (data types and missing values); Intermediate ML lessons 2, 3, 4 and 7 (missing, categorical, pipelines, leakage). **[+]** items address onsite-specific edge cases and feature hypotheses.

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({
    "age": [25, 41, 35, 41, 80],
    "income": [50000, np.nan, 72000, np.nan, -999],
    "city": ["London", "Paris", "unknown", "Paris", "London"],
    "month": [12, 1, 2, 3, 4],
    "target": [0, 1, 0, 1, 1]
})
```

## 1. Audit missingness and dtypes — [K]

**What it does:** Shows how many records would be affected if you drop or impute each field.

```python
pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "missing": df.isna().sum(),
    "missing_pct": df.isna().mean(),
    "unique": df.nunique(dropna=False)
})
```

## 2. `dropna`, `fillna`, `SimpleImputer` — [K]

**What they do:** Delete incomplete rows/columns or substitute a learnt typical value for missing cells.

```python
df.dropna(subset=["target"])                    # target needs to be known
income_fill = df["income"].replace(-999, np.nan).median()
df["income"].replace(-999, np.nan).fillna(income_fill)
```

**For modelling, prefer:**

```python
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="median")
# imp.fit(X_train[numeric_cols]); imp.transform(X_valid[numeric_cols])
```

**Trap:** `fit` learns the median and should use training observations only. Put it inside a pipeline.

## 3. Handle categorical values — [K]

**What it does:** Turns strings into input the model can understand.

```python
pd.get_dummies(df[["city"]])  # quick exploratory encoding
# In sklearn pipelines, use OneHotEncoder(handle_unknown="ignore") instead.
```

**Choices from Intermediate ML:** drop the feature, ordinal encoding (only when ordering is meaningful or model-compatible), or one-hot encoding.

## 4. `astype`, `to_numeric`, `to_datetime` — [K]/[+]

**What they do:** Repair columns read with the wrong type.

```python
df["age"].astype(float)
pd.to_numeric(pd.Series(["1.4", "?", "2.6"]), errors="coerce")
pd.to_datetime(pd.Series(["2025-01-01", "bad"]), errors="coerce")
```

## 5. Duplicates, constants and sentinel values — [+]

**What they do:** Catch repeated records, useless columns and special codes that look numeric but mean missing.

```python
df.duplicated().sum()
df.duplicated(subset=["age", "month"]).sum()  # only if these should form a key
constant_cols = [c for c in df if df[c].nunique(dropna=False) <= 1]
df["income"] = df["income"].replace(-999, np.nan)
```

**Trap:** A valid repeated customer transaction is not automatically a duplicate error.

## 6. Infinities and impossible values — [+]

**What they do:** Identify numbers that sklearn cannot usually handle or that violate the meaning of the variable.

```python
numeric = df.select_dtypes(include=np.number).columns
inf_count = np.isinf(df[numeric]).sum()
df[numeric] = df[numeric].replace([np.inf, -np.inf], np.nan)
df.loc[df["age"] < 0]
```

## 7. Skew/outliers: log, clip, quantiles — [+]

**What they do:** Reduce the influence of very large observations without assuming they are data errors.

```python
s = pd.Series([0, 1, 2, 4, 100.])
np.log1p(s)                                       # only if all values > -1
lo, hi = s.quantile([0.01, 0.99])
s.clip(lo, hi)
```

**Critical:** A tail may be economically important; measure error in that region before discarding/clipping it.

## 8. Cyclic calendar representation — [K: Time Series seasonality] / [+ direct encoding]

**What it does:** Makes December and January close to each other, unlike the raw integers 12 and 1.

```python
m = df["month"]
df["month_sin"] = np.sin(2*np.pi*m/12)
df["month_cos"] = np.cos(2*np.pi*m/12)
```

## 9. Lags and rolling features — [K: Time Series]

**What they do:** Use recent and seasonal past values to help forecast a future outcome.

```python
series = pd.Series([3, 4, 6, 8, 10, 9], dtype=float)
past1 = series.shift(1)
past3_mean = series.shift(1).rolling(3).mean()
```

## 10. Feature interactions and event distance — [+]

**What they do:** Allow for effects that depend jointly on two variables, or fade with distance from an event.

```python
temperature = np.array([5., 12., 25.])
humidity = np.array([0.8, 0.6, 0.3])
temperature * humidity
days_from_holiday = np.array([0, 1, 3, 10])
np.exp(-days_from_holiday/3.0)
```

**Rule:** State a hypothesis before introducing the transform; compare validation results with/without it.

## 11. Leakage guard — [K: Intermediate ML]

**What it does:** Prevents the model from using information it could not have had at prediction time.

- If deciding **before** a sales call, the call's eventual duration is unavailable.
- If forecasting yesterday-to-today returns from yesterday's close, today's closing price is future information.
- If imputing/scaling/encoding, fit on training only; transform validation/test using those learnt rules.
- If selecting features/hyperparameters/thresholds, use validation rather than final test performance.

---

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

---

# 06 — Time Series: Kaggle's six-lesson logic, in code

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.DataFrame({
    "date": pd.date_range("2025-01-01", periods=200, freq="D"),
    "sales": 100 + np.arange(200)*0.2 + 8*np.sin(2*np.pi*np.arange(200)/7)
})
df = df.sort_values("date").set_index("date")
```

## 1. Time-step feature — [K: Lesson 1]

**What it does:** Uses how far through the sample an observation is to model a long-run trend.

```python
df["time"] = np.arange(len(df))
trend_model = LinearRegression()
trend_model.fit(df[["time"]], df["sales"])
trend_pred = trend_model.predict(df[["time"]])
```

**Caution:** Trend estimated over the full sample is only for illustration; for forecasting, fit on the training period.

## 2. Lag feature — [K: Lesson 1]

**What it does:** Gives the model previous observed values to help predict the next one.

```python
df["lag1"] = df["sales"].shift(1)
df["lag7"] = df["sales"].shift(7)
```

**Trap:** If predicting today's sales at the start of today, today's sales cannot appear as a feature.

## 3. Rolling mean for trend — [K: Lesson 2]

**What it does:** Smooths out short-term fluctuations so the longer-run direction is easier to see.

```python
df["trend7"] = df["sales"].shift(1).rolling(7).mean()
```

**Time dummy:** `np.arange(len(df))` is another simple way to represent trend, often with linear regression.

## 4. Seasonal indicators — [K: Lesson 3]

**What they do:** Let the model learn distinct average effects for different weekdays or months.

```python
df["weekday"] = df.index.dayofweek
weekday_flags = pd.get_dummies(df["weekday"], prefix="day", drop_first=True)
```

## 5. Fourier / sine-cosine features — [K: Lesson 3]

**What they do:** Describe a repeating cycle smoothly using waves rather than a separate indicator for each period.

```python
n = np.arange(len(df))
df["weekly_sin"] = np.sin(2*np.pi*n/7)
df["weekly_cos"] = np.cos(2*np.pi*n/7)
```

**Extra useful case:** Encode hour-of-day as `sin(2*pi*hour/24)` and `cos(2*pi*hour/24)` so 23:00 and 00:00 are adjacent.

## 6. Time series as features / lag embedding — [K: Lesson 4]

**What it does:** Creates several past-value columns so a conventional ML model can learn a short recent-history pattern.

```python
for lag in [1, 2, 3, 7, 14]:
    df["lag_%d" % lag] = df["sales"].shift(lag)
feature_cols = ["time", "weekday", "weekly_sin", "weekly_cos", "lag_1", "lag_7"]
model_df = df.dropna(subset=feature_cols + ["sales"]).copy()
```

## 7. Hybrid models — [K: Lesson 5]

**What they do:** Fit a simple model to large-scale structure, then a second model to what the first one fails to explain.

```python
from sklearn.ensemble import RandomForestRegressor
# Example training-only fragment (fit stages only on the training portion):
# trend.fit(X_train_time, y_train)
# residual = y_train - trend.predict(X_train_time)
# second = RandomForestRegressor(random_state=42).fit(X_train_features, residual)
# final_prediction = trend.predict(X_valid_time) + second.predict(X_valid_features)
```

**Not a default:** Start with a straightforward one-model baseline; try a hybrid only if there is enough time and a clear reason.

## 8. Multi-step forecasting strategies — [K: Lesson 6]

**What they do:** Handle forecasts for several future time steps rather than just one.

- **Recursive:** Predict one step, feed prediction back as an input, repeat. Simple but errors compound.
- **Direct:** Fit a separate forecast for each horizon. More models but no predicted-target feedback.
- **Multi-output:** Predict a vector of future values in one model.
- **DirRec / mixed strategies:** Blend direct/recursive ideas.

**Onsite default:** For a one-step task, build only a one-step forecast. Expand only if asked.

## 9. Chronological split + baseline — [+ decisive]

**What it does:** Evaluates a forecasting model on later observations and compares against simply using known history.

```python
model_df = df.dropna().copy()
i = int(0.8 * len(model_df))
train, valid = model_df.iloc[:i], model_df.iloc[i:]

# If predicting today's sales from values known BEFORE today:
persistence = valid["lag1"]
weekly_naive = valid["lag7"]
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(valid["sales"], persistence))
print(mean_absolute_error(valid["sales"], weekly_naive))
```

**Target definition trap:** If instead you form `target_next = sales.shift(-1)` and forecast tomorrow after observing today, the persistence baseline is **today's** sales (`sales`), not `lag1`; the comparable weekly value for tomorrow is six rows before today (`shift(6)` for daily observations).

## 10. `TimeSeriesSplit`, EWM, group lags — [+]

**What they do:** Repeatedly validate past-to-future predictions, smooth observations with an exponential decay, or create lags within entities.

```python
from sklearn.model_selection import TimeSeriesSplit
cv = TimeSeriesSplit(n_splits=4)
df["ewm7_past"] = df["sales"].shift(1).ewm(span=7).mean()
# If multiple shops/assets, sort by entity and timestamp first:
# panel["lag1"] = panel.groupby("shop")["sales"].shift(1)
```

**Pitfalls:** Check missing timestamps/DST, leakage from overlapping windows, and whether features from future exogenous variables would actually be available.

---

# 07 — Statistics and diagnostics: what to check after fitting

**Course basis:** Intro ML measures generalisation mainly through MAE; Intermediate ML uses CV; Time Series examines trend/seasonality. Remaining diagnostics below are **[+] onsite-useful additions**.

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({
    "x": [1., 2., 3., 4., 5., 6.],
    "target": [2., 3., 4., 8., 10., 11.],
    "prediction": [2., 3., 5., 6., 8., 11.]
})
```

## 1. `.corr` (Pearson / Spearman) — [+]

**What it does:** Pearson measures linear association; Spearman measures whether larger values tend to have larger ranks.

```python
df["x"].corr(df["target"])
df["x"].corr(df["target"], method="spearman")
```

**Trap:** Neither implies causation, and low correlation need not imply no nonlinear predictability.

## 2. MAE / RMSE / R² — [K MAE] / [+ others]

**What they do:** MAE shows typical absolute errors; RMSE punishes occasional large errors more; R² measures improvement over predicting the target mean.

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
y, pred = df["target"], df["prediction"]
print("MAE", mean_absolute_error(y, pred))
print("RMSE", mean_squared_error(y, pred)**0.5)
print("R2", r2_score(y, pred))
```

## 3. Mean error and residuals — [+]

**What they do:** Show whether the model systematically predicts too high/low and which examples it misses.

```python
df["error"] = df["target"] - df["prediction"]
print(df["error"].mean())
print(df["error"].quantile([0.05, 0.5, 0.95]))
```

## 4. Target-bucket error — [+]

**What it does:** Checks whether high/low target values have much worse error than the overall average.

```python
df["bucket"] = pd.qcut(df["target"], 3, duplicates="drop")
by_bucket = df.groupby("bucket", observed=True).apply(
    lambda g: np.mean(np.abs(g["target"]-g["prediction"]))
)
print(by_bucket)
```

**Extra interpretation:** When selecting trades or rare opportunities, errors on extreme targets may matter more than overall RMSE.

## 5. Predicted-decile/tercile realised mean — [+]

**What it does:** Tests whether the model ranks observations in a useful order, even if absolute predictions are imperfect.

```python
df["pred_bin"] = pd.qcut(df["prediction"], 3, duplicates="drop")
print(df.groupby("pred_bin", observed=True)["target"].agg(["mean", "size"]))
```

## 6. Out-of-sample stability — [+]

**What it does:** Checks that a result does not rely on one particular segment of a chronological validation period.

```python
validation = df.reset_index(drop=True).copy()  # must already be chronological for this use
# Split the ordered frame and calculate the same metric per block.
for block in np.array_split(np.arange(len(validation)), 3):
    s = validation.iloc[block]
    print(mean_absolute_error(s["target"], s["prediction"]))
```

## 7. Classification metrics — [+]

**What they do:** Precision counts how many predicted positives are correct; recall counts how many real positives you found; F1 combines both; ROC-AUC judges ranking across thresholds.

```python
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
actual = np.array([0, 0, 1, 1, 0, 1])
prob = np.array([0.1, 0.3, 0.6, 0.8, 0.4, 0.7])
label = (prob >= 0.5).astype(int)
print(precision_score(actual, label), recall_score(actual, label), f1_score(actual, label))
print(roc_auc_score(actual, prob))
```

## 8. Simple confidence interval / bootstrap — [+ optional]

**What they do:** Express uncertainty around a sample mean or model comparison instead of treating an observed difference as certain.

```python
x = np.array([3., 4., 5., 6., 7.])
se = x.std(ddof=1)/np.sqrt(len(x))
print((x.mean()-1.96*se, x.mean()+1.96*se))  # rough normal approximation
```

**Important:** Standard IID formulas are not appropriate without adjustment for strongly time-dependent observations.

## 9. Two-sample test and bootstrap — [+ optional]

**What they do:** A t-test compares group means under assumptions; a bootstrap resamples data to estimate sampling uncertainty.

```python
from scipy import stats  # if installed
a = np.array([2., 4., 5., 6., 7.])
b = np.array([1., 3., 3., 4., 5.])
print(stats.ttest_ind(a, b, equal_var=False))
rng = np.random.default_rng(42)
diffs = [rng.choice(a, len(a), replace=True).mean() -
         rng.choice(b, len(b), replace=True).mean() for _ in range(1000)]
print(np.quantile(diffs, [.025, .975]))
```

**Use only when:** Time permits and the dependence assumptions make sense; resampling individual rows is often wrong for time series.

---

# 08 — Python and Jupyter survival (fast-reference)

**Scope:** [+] Onsite addition. The named Kaggle courses assume some Python knowledge, so this fills the specific gap of being rusty in Python itself.

## 1. Lists, dictionaries and loops — [+]

**What they do:** Store groups of values or repeat the same calculation over a small list of columns.

```python
cols = ["age", "income"]
weights = {"age": 0.2, "income": 0.8}
for col in cols:
    print(col)
```

## 2. Functions and comprehensions — [+]

**What they do:** Package a calculation for reuse or construct a list by filtering another list.

```python
def mae(y, pred):
    return abs(y - pred).mean()

columns = ["age", "income", "id"]
features = [c for c in columns if c != "id"]
```

## 3. Check Python, packages and working directory — [+]

**What it does:** Tells you which kernel you're actually using and why Jupyter cannot find a file.

```python
import os, sys, pandas as pd, sklearn
print(sys.version)
print(sys.executable)
print(os.getcwd())
print(pd.__version__, sklearn.__version__)
print(os.listdir("."))
```

## 4. Package install / speed test — [+]

**What they do:** `%pip` installs in the active notebook environment; `%timeit` helps compare slow vs fast approaches.

```python
# These lines run in Jupyter only:
# %pip install seaborn
# %timeit sum(range(1000))
```

**Rule:** Avoid installing optional libraries unless they provide a decisive benefit.

## 5. Read CSV with uncommon delimiter, decimal or encoding — [+]

**What it does:** Corrects files that otherwise load as one giant column or fail due to text encoding.

```python
# pd.read_csv("data.csv", sep=";")
# pd.read_csv("data.csv", decimal=",")
# pd.read_csv("data.csv", encoding="latin1")
```

## 6. Fix common errors — [+]

**What they do:** Reveal the mismatch instead of guessing the cause.

```python
# KeyError: inspect actual names
# print(df.columns.tolist())
# df.columns = df.columns.str.strip()

# ValueError: could not convert string to float
# print(X.select_dtypes(exclude="number").columns)

# Model says NaN is present
# print(X.isna().sum().sort_values(ascending=False).head())

# Wrong shapes
# print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
```

## 7. Restart / run all — [+]

**What it does:** Confirms the final notebook works from a clean kernel rather than relying on variables created out of order.

Use **Kernel → Restart and Run All** before concluding, if time permits. Save a copy of your work first.

## 8. High-value internet documentation links — [+]

- [Pandas API](https://pandas.pydata.org/docs/reference/index.html)
- [NumPy reference](https://numpy.org/doc/stable/reference/)
- [scikit-learn user guide](https://scikit-learn.org/stable/user_guide.html)
- [Matplotlib examples](https://matplotlib.org/stable/gallery/index.html)

**Best search:** `site:pandas.pydata.org DataFrame groupby agg`, not a broad search for a full solution to the entire problem.

---

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

---

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

---

