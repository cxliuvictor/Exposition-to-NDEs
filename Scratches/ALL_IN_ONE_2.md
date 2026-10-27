# The Dataset Interview: One Continuous Research Pipeline

**Original-inspired, current-adapted | Python 3.8-compatible patterns | Pandas → sklearn → out-of-sample conclusion**

This is designed to be read **top to bottom in a Jupyter notebook**, not as disconnected documentation for separate libraries. Each step gives a small explanation, code you can adapt, what to inspect, and what to say to the interviewer. The default route is **tabular regression / forecasting**. Whenever the task is classification, a genuine time series, or trade selection, take the small branch marked **IF** at the point where it becomes relevant.

**Source and adaptation.** The flow follows your uploaded Original project reconstruction: inspect data → preserve a validation period → target distribution / temporal consistency → univariate Pearson/Spearman and response buckets → simple rule, Ridge and boosted trees → compare predictive and decision metrics → validation-block robustness → feature shift → freeze and evaluate OOS. Your Original exercise had predominantly numerical features and a PnL objective. This guide adds missing/categorical data, appropriate split selection, baseline models, and regression/classification decision points. It does **not** assume your onsite dataset will be financial.

**Copying convention:** MAIN PATH cells are written to run **in order**; change only the configuration in Step 0 for your file. IF branches are alternatives or additions, **not** extra cells you must always run. Code that performs operations is not commented out. Comment lines explain what happens. The bundled demo uses sklearn's real Diabetes regression dataset if you have no CSV to practise with.

## A 5.5-hour target pace

| Working time | Goal | Output |
|---|---|---|
| 0:00–0:15 | Load, understand target and one-row meaning | Basic audit |
| 0:15–1:00 | Quality checks + 3–5 diagnostic plots | Cleaning decisions + hypotheses |
| 1:00–1:25 | Split; make modelling matrix; define baseline | Defensible methodology |
| 1:25–2:35 | Baseline → Ridge → one nonlinear model | Comparison table |
| 2:35–3:40 | Diagnose failure and try **one** targeted improvement | Error slices / importance |
| 3:40–4:40 | Validate robustness; freeze; assess OOS | Final test estimate |
| Last 30–40 min | Organise notebook and write conclusions | Coherent presentation |

These are **ceilings, not mandatory durations**: if you finish cleaning earlier, start modelling earlier. If time is running out, skip tuning and importance; preserve the baseline, evaluation and conclusions.

---

## Step 0 — Configure once and load the dataset

**Why:** centralise the only things that differ from one dataset to another; you should not have to rewrite your pipeline in six later cells.

**STARTING CELL — run this first:**

```python
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split, TimeSeriesSplit, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.base import clone

# -------- CHANGE THIS CONFIGURATION FOR THE ACTUAL INTERVIEW --------
USE_DEMO = True                     # Set False for the onsite dataset
DATA_PATH = 'data.csv'              # Local CSV, only used if USE_DEMO=False
TARGET = 'target'                   # Replace with the actual target column
TASK = 'regression'                 # Or 'classification' for binary outcomes
TIME_COL = None                     # E.g. 'date' if chronological; otherwise None
ID_COLS = []                        # E.g. ['customer_id', 'trade_id']; don't model IDs
DROP_COLS = []                      # Unavailable-at-prediction-time / leakage fields
RANDOM_STATE = 42
# -------------------------------------------------------------------

if USE_DEMO:
    # Real datasets included with sklearn, so this guide works without downloads.
    if TASK == 'regression':
        df = load_diabetes(as_frame=True).frame.copy()
    else:
        df = load_breast_cancer(as_frame=True).frame.copy()
else:
    df = pd.read_csv(DATA_PATH)

print('Data shape:', df.shape)
print('Columns:', df.columns.tolist())
display(df.head())
```

`pd.read_csv()` loads a CSV into a DataFrame (a spreadsheet-like table). `display(df.head())` previews the first five observations. `TIME_COL=None` means **do not assume** observations are chronologically dependent.

**IF your CSV has a different delimiter or thousands/decimal convention:**

```python
# Use this instead of the pd.read_csv(DATA_PATH) line in Step 0 when needed.
df = pd.read_csv(DATA_PATH, sep=';')
```

`sep=';'` matters for files where the default comma separator would read everything as one column. Check `df.shape` immediately.

**IF you were given separate labelled train and held-out test files:** load them as `train_df` and `test_df`; make a **development/validation split only inside `train_df`**. Do not select models using the held-out test labels, even if they are visible. You can take the main path by setting `df = train_df.copy()` and replacing its final internal test split with the supplied `test_df`.

## Step 1 — Understand the target and perform a five-minute audit

**Why:** modelling without knowing what one row means, which data are missing, or whether the target leaks into the features is usually a bigger failure than picking a suboptimal model.

**MAIN PATH:**

```python
assert TARGET in df.columns, 'Target column not found: {}'.format(TARGET)
print('Rows / columns:', df.shape)
print('Column types:')
print(df.dtypes.value_counts())
print('Full-row duplicates:', df.duplicated().sum())
print('Missing target:', df[TARGET].isna().sum())
print('Target dtype:', df[TARGET].dtype)
print('Detailed target distribution after the split (Step 4), using train only')

# One row per feature. This is quicker to scan than df.info() alone.
audit = pd.DataFrame({
    'dtype': df.dtypes.astype(str),
    'missing_n': df.isna().sum(),
    'missing_pct': df.isna().mean(),
    'n_unique': df.nunique(dropna=False)
}).sort_values(['missing_pct', 'n_unique'], ascending=[False, True])
display(audit)
```

- `isna().sum()` counts blanks in each column; `isna().mean()` shows their percentage as fractions.
- `nunique()` counts distinct values. One unique value suggests a constant, and thousands of unique IDs may signal a metadata field rather than a useful predictor.
- `duplicated()` detects **identical entire rows**; it does **not** tell you whether an identifier has multiple records.
- `describe()` gives a distribution's centre, dispersion and extremes. Inspect the **training target** after splitting; keep the future test target unexamined.

**Questions to answer in Markdown immediately:** What is one observation? What exactly is predicted? At what moment would you have the features? Do rows represent different people/contracts, or repeated observations of the same entity? Is there a time axis? Is the objective error minimisation, ranking, or a decision?

**IF you have an expected unique key (e.g., asset/date):**

```python
KEYS = ['asset', 'date']  # Replace with meaningful identifying columns
print('Duplicate keys:', df.duplicated(subset=KEYS).sum())
display(df.loc[df.duplicated(subset=KEYS, keep=False)].head(10))
```

This catches **duplicate observations at the same identifier/time** even when other fields differ. Do not delete them until you understand why they occur.

**IF the target is categorical/binary:** inspect `df[TARGET].value_counts(dropna=False, normalize=True)` instead of treating its numeric summary as a meaningful regression target.

## Step 2 — Remove unmistakable problems, but don't “clean away” genuine data

**Why:** the strongest general default is *conservative cleaning*. Keep unusual but valid observations. Treat unknown/missing/impossible values explicitly, and learn imputation statistics on the training set later.

**MAIN PATH:**

```python
# Never train on rows where the outcome is unknown.
df = df.loc[df[TARGET].notna()].copy()

# Only remove exact duplicate full rows when those truly represent repeated records.
# This main-path default leaves duplicates in place for inspection.
print('Exact duplicates still to assess:', df.duplicated().sum())

# Replace infinities with missing values in numeric predictors.
numeric_now = df.select_dtypes(include=np.number).columns.drop(TARGET, errors='ignore')
df[numeric_now] = df[numeric_now].replace([np.inf, -np.inf], np.nan)

# Keep a list of feature exclusions; they will be applied before modelling.
excluded = set([TARGET] + ID_COLS + DROP_COLS)
features = [c for c in df.columns if c not in excluded]
print('Number of candidate features:', len(features))
```

`notna()` removes rows with no label. `.copy()` avoids ambiguous slice modifications. `.replace([np.inf,-np.inf],np.nan)` makes infinite values treatable by standard imputers.

**IF a numeric value is a documented sentinel such as `-999` or `-200`:**

```python
# Example only: do this after confirming -999 means missing for this column.
FEATURE = 'sensor_value'
df[FEATURE] = df[FEATURE].replace(-999, np.nan)
```

**IF you want to inspect constants or near constants:**

```python
constants = [c for c in features if df[c].nunique(dropna=False) <= 1]
print('Constant predictors:', constants)

near_constants = []
for c in features:
    proportions = df[c].value_counts(dropna=False, normalize=True)
    if len(proportions) and proportions.iloc[0] >= 0.99:
        near_constants.append(c)
print('At least 99% one value:', near_constants)
```

Drop a constant predictor when appropriate by adding it to `DROP_COLS` and rebuilding `features`. Treat these pre-split checks as **data-quality flags**; any feature selection based on predictive usefulness belongs on training data. A near-constant column is not automatically useless: rare-event flags might be valuable.

**IF some numeric numbers are stored as strings:**

```python
FEATURE = 'amount'  # Replace with relevant column
converted = pd.to_numeric(df[FEATURE], errors='coerce')
print('Newly unparseable values:', (converted.isna() & df[FEATURE].notna()).sum())
df[FEATURE] = converted
```

`pd.to_numeric(..., errors='coerce')` converts valid numbers and replaces malformed text with `NaN`.

**IF you need simple group operations while investigating:**

```python
# Illustration: replace column names for your dataset.
summary = df.groupby('group').agg(
    n=('outcome', 'size'),
    mean_outcome=('outcome', 'mean'),
    missing_x=('x', lambda s: s.isna().sum())
).reset_index()
```

`groupby('group')` divides the table into categories; **named `agg`** computes one summary row per category. For a value aligned to *every original row*, use `df.groupby('group')['x'].transform('mean')` instead. Don't run the illustration unless those columns exist.

**IF the dataset has multiple related tables:**

```python
# Left-join a customer-level summary into a transaction table.
# 'many_to_one' means transactions repeat customer IDs, summaries do not.
combined = transactions.merge(
    customer_summary, on='customer_id', how='left', validate='many_to_one'
)
assert len(combined) == len(transactions)
```

`join()` is usually shorter when the lookup table is already indexed by the key: `transactions.join(customer_summary.set_index('customer_id'), on='customer_id')`. Use `merge()` when you need explicit keys and `validate=` to catch accidental row multiplication. `pd.concat([a,b], ignore_index=True)` instead **stacks** tables with the same columns. `pivot_table`, `melt` and `explode` are only needed if the dataset actually requires reshaping; don't spend onsite time demonstrating them gratuitously.

**Important leakage check:** Any feature measured after the prediction event is unavailable and must be excluded, even if it has fantastic correlation. Examples: tomorrow's prices, the full duration of a call *before* deciding whom to call, aggregate statistics computed from future labels, or target-derived fields.

## Step 3 — Define what is known at prediction time; split once

**Why:** a good validation design is usually more valuable than an extra model. Your Original notebook sensibly used a chronological last-20% validation period; that would be inappropriate by default for generic unordered IID data.

**IF you need next-hour/next-day prediction from an observed series:** create the shifted future label **now, before making the split**, then remove the missing final label. The *label* uses the future because that is what you are predicting; *features* must not.

```python
# Example: known target at time t, predict its value at t+1.
# Run this BEFORE the MAIN PATH split below, on a correctly sorted series.
df['y_next'] = df['load'].shift(-1)
df['lag_1'] = df['load'].shift(1)
df['lag_24'] = df['load'].shift(24)
df['roll_24'] = df['load'].shift(1).rolling(24).mean()
df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
df = df.dropna(subset=['y_next', 'lag_24', 'roll_24']).copy()
TARGET = 'y_next'
```

`shift(1)` makes previous observations available. `rolling(24)` averages 24 past points. **Important distinction:** if you predict `y_{t+1}` after observing `y_t`, the current `y_t` is a valid feature; if you predict `y_t` before it occurs, it is not. For multi-asset panels, build lags using `.groupby('asset')['value'].shift(1)` after sorting by asset/time. If labels overlap the train/validation boundary at longer horizons, consider a temporal gap/embargo.


**IF this is binary classification and the target contains text:** map the two labels to `0` and `1` **before splitting and before target-correlation plots**. For example, if the labels are literally `no` and `yes`:

```python
df[TARGET] = df[TARGET].map({'no': 0, 'yes': 1})
assert df[TARGET].notna().all(), 'Check unexpected target labels'
```

Change the mapping to match your actual labels. Leave numeric 0/1 targets untouched.

**MAIN PATH — use a training set, validation set and untouched test set:**

```python
# Recalculate after deliberate exclusions/feature creation.
excluded = set([TARGET] + ID_COLS + DROP_COLS)
features = [c for c in df.columns if c not in excluded]

if TIME_COL is not None:
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors='coerce')
    df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)
    features = [c for c in features if c != TIME_COL]

X = df[features].copy()
y = df[TARGET].copy()

if TIME_COL is None:
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE,
        stratify=y if TASK == 'classification' else None
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_dev, y_dev, test_size=0.15/0.85, random_state=RANDOM_STATE,
        stratify=y_dev if TASK == 'classification' else None
    )
else:
    n = len(X)
    train_end, valid_end = int(0.70*n), int(0.85*n)
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_valid, y_valid = X.iloc[train_end:valid_end], y.iloc[train_end:valid_end]
    X_test, y_test = X.iloc[valid_end:], y.iloc[valid_end:]

print('Train:', X_train.shape, 'Valid:', X_valid.shape, 'Test:', X_test.shape)
```

`train_test_split` randomly divides IID observations. The time branch uses `.iloc` so **later periods are never used to train earlier forecasts**. The default here is 70/15/15, not a universal optimal ratio. For very small data use cross-validation rather than burning too much sample on a test set. For groups such as multiple rows per patient/company, use group-aware splitting instead of ordinary random rows.

**IF the same person/company appears repeatedly:** splitting rows at random can put nearly identical observations on both sides. One simple alternative for an IID **grouped** dataset is `GroupShuffleSplit` (and then a second group split inside the development set):

```python
from sklearn.model_selection import GroupShuffleSplit

# 'person_id' must be a real existing column; keep it out of model features.
groups = df['person_id']
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
dev_idx, test_idx = next(gss.split(X, y, groups=groups))
X_dev, X_test = X.iloc[dev_idx], X.iloc[test_idx]
y_dev, y_test = y.iloc[dev_idx], y.iloc[test_idx]
# Repeat GroupShuffleSplit inside X_dev, with groups.iloc[dev_idx],
# to get train/validation partitions without shared group identities.
```

This is an **alternative to**, not an addition after, the ordinary random split. The code presumes the group IDs exist and is intended for adaptation, not automatic execution on the demo dataset.


For a binary outcome, use stratification in the IID split (the code below does that when `TASK="classification"`). The regression modelling cells in Steps 6–8 must then be replaced with the classification branch in Step 7.


## Step 4 — Get the shape of the signal before choosing a model

This reproduces the strongest EDA components of your Original notebook **using training labels only**, but under a strict budget. In that notebook you checked the target PnL distribution, chronological blocks, Pearson/Spearman correlations, and quantile-bucket conditional responses. For the onsite, produce **two or three useful observations**, not dozens of plots.

**MAIN PATH — target distribution and numeric correlations:**

```python
train_eda = df.loc[X_train.index].copy()

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.hist(train_eda[TARGET].dropna(), bins=35)
ax.set(title='Target distribution', xlabel=TARGET, ylabel='Count')
fig.tight_layout()
plt.show()

num_features = [c for c in features if pd.api.types.is_numeric_dtype(train_eda[c])]
pearson = train_eda[num_features].corrwith(train_eda[TARGET], method='pearson')
spearman = train_eda[num_features].corrwith(train_eda[TARGET], method='spearman')
correlations = pd.DataFrame({'pearson': pearson, 'spearman': spearman})
correlations['largest_abs'] = correlations.abs().max(axis=1)
display(correlations.sort_values('largest_abs', ascending=False).head(10))
```

For **binary classification**, treat the histogram as a two-bar class count (use `train_eda[TARGET].value_counts().plot.bar()` instead) and interpret Pearson correlation cautiously; training-group positive rates and category counts are often more informative. `corrwith()` calculates correlation between each numeric feature and the target. Pearson detects **linear** relationships; Spearman detects **monotonic rank** relationships. A low correlation does not prove that the feature is useless (think U-shaped effects and interactions). All plots and correlation screens here use the **training partition**; leave validation and test labels out of feature discovery.

**MAIN PATH — inspect one plausible feature as a conditional response curve:**

```python
if len(num_features):
    strongest_feature = correlations['largest_abs'].idxmax()
    probe = train_eda[[strongest_feature, TARGET]].dropna().copy()
    probe['bucket'] = pd.qcut(probe[strongest_feature], q=8, duplicates='drop')
    bucket_stats = probe.groupby('bucket', observed=True)[TARGET].agg(['mean','size'])
    display(bucket_stats)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(np.arange(len(bucket_stats)), bucket_stats['mean'], marker='o')
    ax.set(xlabel='Increasing feature buckets', ylabel='Mean target',
           title='Conditional response: {}'.format(strongest_feature))
    fig.tight_layout()
    plt.show()
```

`qcut()` divides observations into roughly equally populated quantile buckets. Grouping then estimates **average target conditional on each bucket**. It reveals nonlinear shapes that one correlation number may miss. This is descriptive; confirm any proposed transform on validation.

**IF you have categories:**

```python
# Replace 'category' with an actual categorical predictor.
category_summary = train_eda.groupby('category')[TARGET].agg(['mean', 'median', 'count'])
display(category_summary.sort_values('count', ascending=False).head(12))
```

For a binary outcome, the conditional mean is the **observed positive rate**. Check that a high category mean is not coming from two observations.

**IF you have time order:**

```python
# Run after parsing/sorting the real time column, not on arbitrary row order.
time_train = train_eda.sort_values(TIME_COL).copy()

fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(time_train[TIME_COL], time_train[TARGET], alpha=0.65)
ax.set(title='Target through time', xlabel=TIME_COL, ylabel=TARGET)
fig.tight_layout()
plt.show()

# Descriptive time-block check (NOT a validation score).
blocks = pd.qcut(np.arange(len(time_train)), q=5, labels=False)
display(time_train.groupby(blocks)[TARGET].agg(['mean', 'std', 'size']))
```

The block table asks: **is the distribution changing through time?** A model that works only during one regime is risky. Note: target observations need not be available at time of prediction, so use **training-period targets** for research decisions.

### A small, justified feature idea is enough

If the plot suggests right skew, try `np.log1p(x)` for nonnegative measurements. If the effect is seasonal, try `sin(2πh/24)` and `cos(2πh/24)`. If it depends on recent history, try lags or rolling averages. If two fields interact, try their product. **Do not add all four merely because the recipes exist.**

## Step 5 — Make a clean modelling matrix without leaking information

**Why:** fitting a median, scaler or encoding to the complete dataset allows future/validation information into preprocessing. A `Pipeline` trains these transformations *only* on `X_train` during `.fit()` and reuses them for `.predict()`.

**MAIN PATH — includes numeric/categorical support:**

```python
numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
category_cols = [c for c in X_train.columns if c not in numeric_cols]
print('Numeric:', len(numeric_cols), 'Categorical:', len(category_cols))

numeric_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])
category_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocess = ColumnTransformer([
    ('num', numeric_pipeline, numeric_cols),
    ('cat', category_pipeline, category_cols)
])
```

- `SimpleImputer` fills missing numbers using **training medians**, and categories using **training modes**.
- `StandardScaler` centres and rescales numeric columns, particularly helpful for Ridge regularisation.
- `OneHotEncoder` converts categories into indicator columns; `handle_unknown='ignore'` prevents errors when a previously unseen category appears at validation/test.
- `ColumnTransformer` applies the appropriate processing to each set of columns.

**Caution:** a very high-cardinality column (e.g., unique customer ID) can create too many one-hot variables. Exclude ID-like data when appropriate. For extremely wide sparse designs, prefer Ridge/RandomForest over algorithms that require dense input.

**IF the numeric feature is positive and severely right-skewed:** consider `np.log1p(x)`, but transform **after** defining the training/validation protocol and justify it. **IF missingness itself is informative:** add an imputer's missing indicator (`SimpleImputer(add_indicator=True, strategy='median')`) or a separate `isna().astype(int)` flag built without labels. **IF your dataset is already clean and entirely numeric:** the pipeline above still works, and no categorical handling is needed.

## Choose your continuation: regression/forecasting or binary classification

The shared preparation is now complete. **For regression or time-series forecasting**, continue immediately with Steps 6–10 below. **For binary classification**, skip to [C6–C10: the complete classification continuation](#c6--c10-the-complete-classification-continuation) after the regression path. Both follow the same logic: baseline → two candidate models → diagnostics → robustness/selection → final OOS test.

Time series is **not a separate model family**: it is a forecasting version of the regression problem, requiring chronological validation, causal features and appropriate naive baselines. Classification can also have time or group dependence, in which case the common split rules still apply.

## Step 6 — Establish a baseline first (regression and forecasting)

The Original notebook compared a simple one-feature rule with Ridge and gradient boosting. For generic regression, start even simpler: the **training mean**. With a genuine time series, persistence or seasonal-naive is often stronger.

**MAIN PATH:**

```python
def reg_metrics(y_true, y_pred):
    return {
        'RMSE': mean_squared_error(y_true, y_pred) ** 0.5,
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'Pred/target corr': pd.Series(np.asarray(y_pred)).corr(
            pd.Series(np.asarray(y_true)), method='pearson'
        )
    }

baseline_model = DummyRegressor(strategy='mean')
baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_valid)
results = [{'model': 'Train mean', **reg_metrics(y_valid, baseline_pred)}]
display(pd.DataFrame(results))
```

`RMSE` (square-root average squared residual) penalises large errors; `MAE` is average absolute error; `R²` compares to a constant-mean prediction; prediction/target correlation indicates association but **not calibration or magnitude accuracy**. A constant prediction has undefined correlation; this is expected.

**IF next-step time series:** compare against the latest-known value and a seasonal-naive prediction. With `target_next = load.shift(-1)` at row `t` (predicting `load[t+1]`), **persistence is `load[t]`**, while a same-hour-previous-day baseline is **`load[t-23]`**, not `load[t-24]`. The exact lag must be derived from the forecast horizon, not memorised. Never compare a next-hour forecast to a baseline that accidentally includes the next-hour observed value.

**IF the target is binary:** follow the complete C6–C10 path below instead of executing the regression baseline. Start with a `DummyClassifier`, prevalence, ROC-AUC and average precision.

## Step 7 — Fit only two candidate model families (regression and forecasting)

One simple regularised model, one nonlinear alternative. This is typically enough to demonstrate research judgement before the interviewer discussion. Avoid large hyperparameter grids.

**MAIN PATH — Ridge, then boosted trees for numeric-only data or Random Forest for mixed data:**

```python
ridge = Pipeline([
    ('preprocess', clone(preprocess)),
    ('model', Ridge(alpha=1.0))
])

if len(category_cols) == 0:
    nonlinear_estimator = HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=160, max_leaf_nodes=20,
        l2_regularization=1.0, random_state=RANDOM_STATE
    )
else:
    # RandomForest accepts the sparse matrices often returned by OneHotEncoder.
    nonlinear_estimator = RandomForestRegressor(
        n_estimators=150, min_samples_leaf=3,
        random_state=RANDOM_STATE, n_jobs=-1
    )

nonlinear = Pipeline([
    ('preprocess', clone(preprocess)),
    ('model', nonlinear_estimator)
])

models = {'Train mean': baseline_model, 'Ridge': ridge, 'Nonlinear': nonlinear}
valid_preds = {'Train mean': baseline_pred}
for name in ['Ridge', 'Nonlinear']:
    model = models[name]
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    valid_preds[name] = pred
    results.append({'model': name, **reg_metrics(y_valid, pred)})

comparison = pd.DataFrame(results).sort_values('RMSE')
display(comparison)
```

`fit()` learns model parameters from training data. `predict()` applies the trained model to untouched validation features. Ridge learns a **regularised linear relationship** (typically stable/interpretable), while boosted trees / Random Forest can capture **nonlinear thresholds and interactions**. Complexity does not guarantee improvement.

**What to say:** “I started with a mean baseline to establish difficulty, then a regularised linear model, then one nonlinear model to test whether interactions or thresholds matter. I selected on validation, not on training fit.”

## Step 8 — Diagnose *where* a regression model succeeds or fails

Your Original notebook correctly went beyond RMSE to prediction/target correlation and decision value. For a generic regression, residual and target-bucket plots are the simplest informative additions.

**REGRESSION PATH — choose the validation winner, then inspect it:**

```python
chosen_name = comparison.iloc[0]['model']
chosen_model = models[chosen_name]
chosen_pred = np.asarray(valid_preds[chosen_name])
truth = np.asarray(y_valid)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(truth, chosen_pred, alpha=0.4, s=12)
lims = [min(truth.min(), chosen_pred.min()), max(truth.max(), chosen_pred.max())]
axes[0].plot(lims, lims, '--', color='gray')
axes[0].set(xlabel='Observed', ylabel='Predicted', title='Actual versus predicted')
axes[1].scatter(chosen_pred, truth-chosen_pred, alpha=0.4, s=12)
axes[1].axhline(0, linestyle='--', color='gray')
axes[1].set(xlabel='Predicted', ylabel='Residual', title='Residuals')
fig.tight_layout()
plt.show()
```

Actual-v-predicted shows calibration/compression of extremes. Residual-v-predicted reveals systematic under/overprediction, nonlinearity and changing error spread.

**REGRESSION PATH — error conditional on the *actual* target:**

```python
errors = pd.DataFrame({'actual': truth, 'prediction': chosen_pred})
errors['error'] = errors['actual'] - errors['prediction']
errors['target_bucket'] = pd.qcut(errors['actual'], 5, duplicates='drop')
error_by_bucket = errors.groupby('target_bucket', observed=True).agg(
    n=('error', 'size'),
    MAE=('error', lambda s: np.abs(s).mean()),
    RMSE=('error', lambda s: np.sqrt(np.mean(s ** 2)))
)
display(error_by_bucket)
```

Target buckets tell you whether the model is particularly poor on **rare or extreme outcomes**. If tails matter economically, this may be more important than small global RMSE differences. Avoid interpreting bucket error based on a handful of samples.

**IF prediction is mostly about ranking opportunities:**

```python
ranking = pd.DataFrame({'score': chosen_pred, 'outcome': truth})
ranking['predicted_bucket'] = pd.qcut(ranking['score'], 10, duplicates='drop')
display(ranking.groupby('predicted_bucket', observed=True)['outcome']
               .agg(['mean', 'size']))
print('Spearman rank correlation:', ranking['score'].corr(ranking['outcome'], method='spearman'))
```

This asks whether larger predictions actually identify larger realised outcomes. It is **not** a trading PnL analysis unless the outcome is economically meaningful after costs.

### IF the objective is Original-style PnL / trade selection

Your Original task selected a fraction of opportunities by predicted PnL. Replicate that **only when the target is trade-level PnL** (or can be translated into PnL defensibly).

```python
# Validation only: score from chosen model, realised per-trade PnL = y_valid.
score = np.asarray(chosen_pred)
realised_pnl = np.asarray(y_valid)
fractions = [0.10, 0.20, 0.30, 0.50]
trade_rows = []
for frac in fractions:
    cutoff = np.quantile(score, 1-frac)
    take = score >= cutoff
    trade_rows.append({
        'fraction': frac,
        'n_trades': int(take.sum()),
        'mean_pnl': realised_pnl[take].mean(),
        'total_pnl': realised_pnl[take].sum(),
        'hit_rate': np.mean(realised_pnl[take] > 0)
    })
trade_table = pd.DataFrame(trade_rows)
display(trade_table)
```

A smaller fraction may improve average PnL per trade but reduce total opportunities. Compare **trade all** and **trade nothing**. If transactions have costs, subtract them. Choose the fraction on validation only. Trying 20 fractions and taking the single maximum is more prone to selection bias; favour a stable plateau across nearby thresholds.

## Step 9 — Check whether the regression result is fragile

This is where research judgement becomes visible. The Original reconstruction used validation-time blocks and test/train feature-shift diagnostics. Preserve those when relevant, but keep the exercise proportional.

**REGRESSION PATH — limited CV on training data (optional when time is tight):**

```python
cv = (TimeSeriesSplit(n_splits=4) if TIME_COL is not None
      else KFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE))

cv_scores = cross_val_score(
    clone(chosen_model), X_train, y_train,
    cv=cv, scoring='neg_mean_squared_error'
)
cv_rmse = np.sqrt(-cv_scores)
print('Training-fold RMSE:', cv_rmse.round(4))
print('Mean RMSE:', cv_rmse.mean().round(4),
      '| Fold sd:', cv_rmse.std().round(4))
```

`cross_val_score()` retrains an estimator on multiple training folds. Large variation between folds suggests instability. For time series, `TimeSeriesSplit` repeatedly trains on the past and validates on a later block. **Do not** run shuffled folds through forecasting data.

**REGRESSION PATH — examine feature distribution shift *without test labels*:**

```python
numeric_cols = X_train.select_dtypes(include=np.number).columns
if len(numeric_cols):
    train_mean = X_train[numeric_cols].mean()
    test_mean = X_test[numeric_cols].mean()
    train_sd = X_train[numeric_cols].std().replace(0, np.nan)
    shift = ((test_mean-train_mean).abs()/train_sd).sort_values(ascending=False)
    display(shift.head(10).rename('standardised_mean_shift').to_frame())
```

This is a quick check, not a statistical proof of drift. Inspect shifts caused by missing values or changing populations. **Using unlabeled test features to diagnose deployment shift is distinct from tuning with test labels**, but do not repeatedly change the final model based on the test sample without acknowledging that you are adapting to it.

**IF time order matters:** compute RMSE/MAE on 3–5 chronological validation sub-blocks. Watch whether the last validation period deteriorates sharply.

**IF you have 5–10 minutes spare:**

```python
importance = permutation_importance(
    chosen_model, X_valid, y_valid, n_repeats=5,
    random_state=RANDOM_STATE, scoring='neg_mean_squared_error'
)
importance_table = pd.Series(
    importance.importances_mean, index=X_valid.columns
).sort_values(ascending=False)
display(importance_table.head(10).to_frame('importance'))
```

Permutation importance measures how much the score worsens when one column is scrambled. It works through the fitted **entire pipeline**, not just the model. Correlated features can act as substitutes, so their individual permutation scores can be deceptively small. SHAP is not needed unless explicitly asked.

### How to decide whether to try one more feature/model

Try a change only if you can write a sentence beginning **“Because the plot/structure suggests …”**: because seasonality is strong → cyclic encodings; because target reacts gradually → lagged rolling average; because marginal influence saturates → `log1p`/`tanh`; because Ridge residuals curve → nonlinear model. Validate the *change* and keep it only if it helps. Otherwise stop.

## Step 10 — Freeze the regression procedure; evaluate OOS only once

Your Original notebook's final test logic is instructive, with one important correction: **refit whichever model was chosen**, rather than always cloning the gradient-boosting model. Also freeze the feature set, preprocessing choices and any selection fraction before reading test labels.

**MAIN PATH:**

```python
# At this point chosen_name, hyperparameters and features are fixed.
X_fit = pd.concat([X_train, X_valid], axis=0)
y_fit = pd.concat([y_train, y_valid], axis=0)
final_model = clone(models[chosen_name])
final_model.fit(X_fit, y_fit)

test_pred = final_model.predict(X_test)
test_metrics = reg_metrics(y_test, test_pred)
print('Final model:', chosen_name)
print('Validation performance:', reg_metrics(y_valid, valid_preds[chosen_name]))
print('Final OOS performance:', test_metrics)
```

Refitting train+validation **after model selection is finished** increases the available training sample without using test labels. The test metric now estimates the whole selected/retrained procedure, while the validation metric corresponds to an earlier smaller training fit; they need not match exactly.

**IF the true test labels were not supplied:** don't invent an OOS metric. Produce predictions and report validation diagnostics with that limitation.

**IF you chose a trade fraction:** apply the **same fraction** to the test scores and compute realised test PnL only if test labels are available. A cohort-wise top-fraction rule assumes that all cohort scores are known when making selections. For sequential live decisions, freeze an absolute or training-based threshold instead and account for execution timing/costs.

## C6–C10: the complete classification continuation

Use this path **instead of regression Steps 6–10** when the target is genuinely categorical, especially a binary outcome such as default/no default or subscribe/no subscribe. It uses the **same cleaned features and train/validation/test split from Steps 0–5**, but changes the predictions, baselines, metrics, thresholds and diagnostics.

### C6 — Understand the classes and establish a proper baseline

**What/why:** first establish the positive prevalence. When only 5% of observations are positive, predicting “no” always gives 95% accuracy without identifying a single positive case. Compare **probabilities**, not only hard labels.

```python
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, log_loss, brier_score_loss)

# Map any text labels to 0/1 back in Step 3; positive class is always 1.
assert set(np.unique(y_train)) == {0, 1}, 'This path expects binary 0/1 labels'
print('Training class counts:\n', y_train.value_counts())
print('Positive rate:', y_train.mean())

majority = DummyClassifier(strategy='prior')
majority.fit(X_train, y_train)
prob_baseline = majority.predict_proba(X_valid)[:, 1]
```

`DummyClassifier(strategy='prior')` predicts the **training-set class frequencies** for everyone. Its constant score cannot distinguish positive and negative observations. This is a stronger reference than merely reporting accuracy.

**Choose the right metrics:**

- **ROC-AUC:** how well positive cases rank above negatives, across all thresholds. A random-ranking model is around 0.5.
- **Average precision (AP):** summarises the precision–recall curve; especially useful when positives are rare. Its no-skill reference is roughly the positive prevalence. AP is a particular PR summary, not identical to every numerical definition of “PR-AUC”.
- **Precision:** of those predicted positive, what fraction truly were positive?
- **Recall:** of all true positives, what fraction did we identify?
- **F1:** balances precision and recall; it does **not** represent asymmetric business costs.
- **Log loss / Brier score:** evaluate probability quality, not merely ranking. Lower is better.

### C7 — Compare a simple interpretable classifier with a nonlinear one

**What/why:** logistic regression estimates a probability using a linear function of the features' transformed values. Random Forest aggregates decision trees, allowing nonlinear thresholds and interactions. Both are fitted through the **same training-only preprocessing** from Step 5.

```python
classifiers = {
    'Prior baseline': majority,
    'Logistic': Pipeline([
        ('preprocess', clone(preprocess)),
        ('model', LogisticRegression(max_iter=2000))
    ]),
    'RandomForest': Pipeline([
        ('preprocess', clone(preprocess)),
        ('model', RandomForestClassifier(
            n_estimators=150, min_samples_leaf=3,
            random_state=RANDOM_STATE, n_jobs=-1
        ))
    ])
}

valid_prob = {'Prior baseline': prob_baseline}
for name in ['Logistic', 'RandomForest']:
    classifiers[name].fit(X_train, y_train)
    valid_prob[name] = classifiers[name].predict_proba(X_valid)[:, 1]

def class_metrics(y_true, probability, threshold=0.5):
    prediction = (np.asarray(probability) >= threshold).astype(int)
    return {
        'ROC_AUC': roc_auc_score(y_true, probability),
        'Avg_precision': average_precision_score(y_true, probability),
        'Log_loss': log_loss(y_true, probability, labels=[0, 1]),
        'Accuracy': accuracy_score(y_true, prediction),
        'Precision': precision_score(y_true, prediction, zero_division=0),
        'Recall': recall_score(y_true, prediction, zero_division=0),
        'F1': f1_score(y_true, prediction, zero_division=0)
    }

classification_table = pd.DataFrame([
    {'model': name, **class_metrics(y_valid, prob)}
    for name, prob in valid_prob.items()
]).sort_values('Avg_precision', ascending=False)
display(classification_table)
```

The comparison above uses **threshold-independent measures (ROC-AUC/AP/log loss)** alongside threshold-0.5 metrics. Do not pick the winner using validation accuracy if the outcome is highly imbalanced. For a rare-positive identification task, AP is a reasonable default selection metric; for an actual business decision, use its explicit cost/benefit when possible.

**IF the problem has a known asymmetry:** e.g. missing fraud costs more than reviewing a false alert. A model with lower accuracy may still be better because it finds more important positive cases. This is why the next step chooses an *action threshold* separately from the model.

### C8 — Inspect the selected model, then choose an action threshold

**What/why:** probabilities are **scores**, while labels/actions result from a threshold. Moving the threshold changes the trade-off between false alarms and missed positives. Choose both model and threshold using **validation only**.

```python
# Selection metric: choose based on the stated objective, not by habit.
chosen_class_name = classification_table.iloc[0]['model']
chosen_class_model = classifiers[chosen_class_name]
chosen_prob = np.asarray(valid_prob[chosen_class_name])

# Study how realised positive rates vary with the predicted score.
calibration = pd.DataFrame({
    'prob': chosen_prob, 'actual': np.asarray(y_valid)
})
calibration['bucket'] = pd.qcut(calibration['prob'], 10, duplicates='drop')
calibration_table = calibration.groupby('bucket', observed=True).agg(
    n=('actual', 'size'),
    mean_prediction=('prob', 'mean'),
    realised_positive_rate=('actual', 'mean')
)
display(calibration_table)

if len(calibration_table):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(calibration_table['mean_prediction'],
            calibration_table['realised_positive_rate'], marker='o')
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set(xlabel='Average predicted probability',
           ylabel='Observed positive rate', title='Probability calibration by bucket')
    fig.tight_layout()
    plt.show()
```

A calibration table asks whether, for example, cases assigned ~20% probability are positive roughly 20% of the time. A model may rank correctly but be poorly calibrated. In a very small dataset, bins with few observations will be noisy. If a constant baseline model is chosen, `qcut` may produce no meaningful bins; report its constant base rate instead of attempting to interpret a calibration curve.

**Choose a threshold** with a clear objective. The default below maximises validation F1. If the task supplies false-positive and false-negative costs, optimise those costs instead.

```python
threshold_grid = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
rows = []
for threshold in threshold_grid:
    pred = (chosen_prob >= threshold).astype(int)
    rows.append({
        'threshold': threshold,
        'precision': precision_score(y_valid, pred, zero_division=0),
        'recall': recall_score(y_valid, pred, zero_division=0),
        'F1': f1_score(y_valid, pred, zero_division=0),
        'false_positives': int(((pred == 1) & (np.asarray(y_valid) == 0)).sum()),
        'false_negatives': int(((pred == 0) & (np.asarray(y_valid) == 1)).sum())
    })
threshold_table = pd.DataFrame(rows)
display(threshold_table)
chosen_threshold = float(threshold_table.loc[
    threshold_table['F1'].idxmax(), 'threshold'
])
print('Frozen model/threshold:', chosen_class_name, chosen_threshold)

validation_labels = (chosen_prob >= chosen_threshold).astype(int)
print('Validation confusion matrix (rows true 0/1, columns predicted 0/1):')
print(confusion_matrix(y_valid, validation_labels, labels=[0, 1]))
```

`confusion_matrix` makes the errors concrete: top-left = true negatives, top-right = false positives, bottom-left = false negatives, bottom-right = true positives. A high recall generally requires lowering the threshold, which usually creates more false positives.

**IF business costs are known**, calculate an actual decision loss rather than relying on F1:

```python
COST_FP, COST_FN = 1.0, 5.0   # Replace with the stated business costs.
threshold_table['decision_cost'] = (
    COST_FP * threshold_table['false_positives']
    + COST_FN * threshold_table['false_negatives']
)
print(threshold_table[['threshold', 'decision_cost']])
# For cost-based selection, set chosen_threshold from decision_cost.idxmin()
```

The code evaluates both criteria; **do not automatically change the selected threshold** unless cost minimisation is your declared objective. Any new choice must be made now, on validation, before testing.

### C9 — Check robustness rather than searching dozens of models

**What/why:** a single favourable split can be luck. Check how performance varies across training folds, and where validation failures concentrate. Keep this short if time is limited.

```python
from sklearn.model_selection import StratifiedKFold
cv = (TimeSeriesSplit(n_splits=4) if TIME_COL is not None
      else StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE))

cv_scores = cross_val_score(
    clone(chosen_class_model), X_train, y_train,
    cv=cv, scoring='average_precision'
)
print('Training-fold average precision:', cv_scores.round(3))
print('Mean / standard deviation:', cv_scores.mean(), cv_scores.std())
```

For **time-ordered classification**, `TimeSeriesSplit` still avoids training on future periods. But check that every fold contains both classes, otherwise ROC-AUC/AP or even logistic fitting may be undefined. For data with repeated people/accounts, prefer a **group-aware split** to avoid putting the same entity in train and test.

**IF you have useful categories such as region, sex, product or time period**, measure where false negatives / false positives concentrate, using validation data only. The snippet below works when `GROUP_FEATURE` is a retained column.

```python
# Use a meaningful categorical feature if one exists.
if category_cols:
    group_feature = category_cols[0]
    error_view = pd.DataFrame({
        'group': X_valid[group_feature].astype(str).to_numpy(),
        'true': np.asarray(y_valid),
        'predicted': validation_labels
    })
    error_view['false_negative'] = ((error_view['true'] == 1) &
                                    (error_view['predicted'] == 0))
    error_view['false_positive'] = ((error_view['true'] == 0) &
                                    (error_view['predicted'] == 1))
    display(error_view.groupby('group').agg(
        n=('true', 'size'),
        positive_rate=('true', 'mean'),
        fn_count=('false_negative', 'sum'),
        fp_count=('false_positive', 'sum')
    ).sort_values('n', ascending=False).head(12))
```

This runs only if you retained categorical features. Choose a substantive grouping variable instead of automatically treating an ID or a continuous measurement as a subgroup. For numeric-only data, use predicted-probability deciles and consider whether errors concentrate at extreme scores.

**IF you want feature importance and have a few minutes:**

```python
importance = permutation_importance(
    chosen_class_model, X_valid, y_valid, n_repeats=5,
    random_state=RANDOM_STATE, scoring='average_precision'
)
display(pd.Series(importance.importances_mean, index=X_valid.columns)
          .sort_values(ascending=False).head(10).rename('AP_drop').to_frame())
```

Permutation importance measures how much predictive quality drops when one validation feature is shuffled. It works on the **whole pipeline**; correlated features can substitute for one another. It does not prove causality.

### C10 — Freeze the classification procedure and evaluate OOS once

**What/why:** the test labels are for the final assessment, not another tuning set. At this point, the model family, features, preprocessing and threshold have already been selected using training/validation. Refit the selected **pipeline** on all development data, keeping the **same threshold**.

```python
final_classifier = clone(chosen_class_model)
final_classifier.fit(pd.concat([X_train, X_valid]),
                     pd.concat([y_train, y_valid]))

prob_test = final_classifier.predict_proba(X_test)[:, 1]
test_metrics = class_metrics(y_test, prob_test, threshold=chosen_threshold)
print('Frozen model:', chosen_class_name)
print('Frozen threshold:', chosen_threshold)
print('Validation:', class_metrics(y_valid, chosen_prob, threshold=chosen_threshold))
print('Final test:', test_metrics)
print('Test confusion matrix:\n', confusion_matrix(
    y_test, (prob_test >= chosen_threshold).astype(int), labels=[0, 1]
))
```

Once you have looked at test performance, **do not** revise the threshold/model based on test errors and still describe the revised estimate as untouched OOS. If no test labels exist, produce the probabilities and frozen decisions without claiming a final OOS score.

The classification path now rejoins the shared **Step 11: notebook conclusion and interviewer defence**. Present prevalence, baseline AP, model comparison, the reason for your selected threshold, confusion-matrix trade-offs, the strongest failure group and the untouched OOS result. Avoid talking only about accuracy.

---

## Step 11 — Turn the notebook into your presentation

End with one readable Markdown summary: problem → data issue → baseline → model comparison → main finding → limitations. **Do not leave the interviewer staring at 30 unlabeled cells.**

**Closing structure and complete it from your results:**

> ### Research summary
> **Objective.** Predict [outcome] for [unit of observation], using information available at [decision time].
>
> **Dataset.** [Rows × columns], [date range/groups if relevant]. The biggest issue was [missingness, duplication, leakage, drift or class imbalance].
>
> **Method.** I used [IID/chronological/group] train/validation/test because [reason]. The reference baseline was [mean/persistence/majority class]. I compared [simple model] with [nonlinear model].
>
> **Finding.** On validation, [metric and model comparison]. On untouched test, [metric if available]. [One useful feature/hypothesis] appeared important.
>
> **Weakness.** Performance was worse for [tail, group, time block or other segment]. [Potential issue: small sample, label leakage, distribution shift, multiple testing, costs].
>
> **Next steps.** [One specific additional experiment], rather than generic “tune more”.

**Likely questions:** Why that metric? What was the naive baseline? Why Ridge? Did boosted trees truly help? Was the split appropriate? Could any predictor leak the outcome? How stable were results across subgroups or time? Were extreme outcomes harder? What did you reject? If the objective is trading, how do prediction accuracy, rank quality and net trading PnL differ?

**Likely follow-up:** Explain what you actually built, what failed, and what commercial or market intuition it demonstrates. You can connect to your options/neural-SDE benchmarking or market-signal research, but keep the answer about this dataset grounded in *this* dataset.

---

## At-a-glance recovery: if you are stuck or nearly out of time

| Situation | Smallest credible response |
|---|---|
| Can't remember pandas syntax | Search docs; do not spend 15 minutes on one `.groupby` argument |
| Many missing/categorical features | Use `SimpleImputer` + `OneHotEncoder` inside `ColumnTransformer` |
| Unknown prediction objective | Clarify regression versus classification versus ranking **before** fitting |
| Numerical regression, little time | Train mean → Ridge → one nonlinear sklearn model → RMSE/MAE |
| Imbalanced binary labels | Stratified split → logistic → ROC-AUC/average precision → threshold |
| Time series | Sort → causal lags → future holdout → persistence/seasonal baseline |
| No strong signals | Show a valid negative result instead of inventing value |
| Validation looks far better than OOS | Check leakage, repeated tuning and distribution shift; report it |
| 20 minutes remaining | Stop modelling; output comparison table, 2 plots and 6-line summary |

**Research rule to remember:** a simple, correctly validated conclusion is stronger than a sophisticated but unfinished or leaky model.
