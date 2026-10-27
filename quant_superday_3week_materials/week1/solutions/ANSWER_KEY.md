# Week 1 Answer Key and Code Patterns

Use only after attempting the exercises. Exact numeric outputs may vary slightly if a public mirror changes, but the code patterns and reasoning should remain valid.

---

# Notebook 1 — Pandas Transformations

## Audit
```python
for name, x in {"movies": movies, "ratings": ratings, "tags": tags}.items():
    print(name, x.shape)
    print(x.dtypes)
    print(x.isna().sum())
    print("duplicates", x.duplicated().sum())
```

## Conditional counts
```python
user_stats = ratings.groupby("userId").agg(
    n=("rating","size"),
    mean_rating=("rating","mean")
)
heavy = user_stats.query("n >= 100")
n_heavy_below_35 = (heavy["mean_rating"] < 3.5).sum()

movie_stats = ratings.groupby("movieId").agg(
    n=("rating","size"),
    mean_rating=("rating","mean")
)
strong_movies = movie_stats.query("n >= 50 and mean_rating > 4.0")
```

## Named agg / transform
```python
user_summary = ratings.groupby("userId").agg(
    n=("rating","size"),
    mean=("rating","mean"),
    median=("rating","median"),
    sd=("rating","std"),
    last_date=("datetime","max"),
).reset_index()

ratings["user_mean"] = ratings.groupby("userId")["rating"].transform("mean")
ratings["user_sd"] = ratings.groupby("userId")["rating"].transform("std")
ratings["demeaned"] = ratings["rating"] - ratings["user_mean"]
ratings["z"] = ratings["demeaned"] / ratings["user_sd"]
```
`agg` reduces each group; `transform` returns an aligned value for each original row.

## Explode
```python
mg = movies.assign(genres=movies["genres"].str.split("|")).explode("genres")
genre_counts = mg["genres"].value_counts()

rg = ratings.merge(mg[["movieId","genres"]], on="movieId",
                   how="left", validate="many_to_many")
genre_stats = rg.groupby("genres").agg(
    n=("rating","size"),
    mean_rating=("rating","mean")
).sort_values("n", ascending=False)
```
The many-to-many row increase is expected because one movie can have multiple genres.

## Merge validation
```python
ratings_movies = ratings.merge(
    movies, on="movieId", how="left", validate="many_to_one"
)
movie_summary = ratings.groupby("movieId").agg(
    n_ratings=("rating","size"),
    avg_rating=("rating","mean")
).reset_index()

movies2 = movies.merge(movie_summary, on="movieId",
                       how="left", validate="one_to_one")
```

## Join
```python
movies_idx = movies.set_index("movieId")
summary_idx = movie_summary.set_index("movieId")
joined = movies_idx.join(summary_idx, how="left", validate="one_to_one")
```

## Cross join
```python
buckets = pd.DataFrame({"rating_bucket":["low","mid","high"]})
top_genres = pd.DataFrame({"genre": mg["genres"].value_counts().head(5).index})
grid = buckets.merge(top_genres, how="cross")
assert len(grid) == 15
```

## Pivot / melt
```python
top30 = ratings["movieId"].value_counts().head(30).index
sub = ratings[ratings["movieId"].isin(top30)]
wide = sub.pivot_table(index="userId", columns="movieId",
                       values="rating", aggfunc="mean")
long = wide.reset_index().melt(
    id_vars="userId", var_name="movieId", value_name="rating"
).dropna()
```

## Rank / qcut / shift / rolling
```python
ratings["rating_pct"] = ratings.groupby("userId")["rating"].rank(pct=True)

movie_stats = movie_stats.reset_index()
movie_stats["pop_q"] = pd.qcut(movie_stats["n"], 5, labels=False,
                               duplicates="drop") + 1

r = ratings.sort_values(["userId","datetime"]).copy()
r["prev_rating"] = r.groupby("userId")["rating"].shift(1)
r["roll10_past"] = r.groupby("userId")["rating"].transform(
    lambda s: s.shift(1).rolling(10, min_periods=3).mean()
)
```

---

# Notebook 2 — NumPy + Transformations

## Axis / standardisation
```python
cols = ["temp","atemp","hum","windspeed","cnt"]
X = df[cols].to_numpy(float)

col_mu = np.nanmean(X, axis=0)
col_sd = np.nanstd(X, axis=0)
X_col_z = (X-col_mu)/col_sd

row_mu = np.nanmean(X, axis=1, keepdims=True)
row_sd = np.nanstd(X, axis=1, keepdims=True)
X_row_z = (X-row_mu)/row_sd
```

## Masks / where / select
```python
q90 = np.quantile(df["cnt"], .90)
high = np.where(df["cnt"].to_numpy() > q90, 1, 0)

weather = df["weathersit"].to_numpy()
severity = np.select(
    [weather <= 1, weather == 2, weather >= 3],
    [0, 1, 2],
    default=np.nan
)
```

## Cyclic encoding
```python
h = df["hr"].to_numpy()
h_sin = np.sin(2*np.pi*h/24)
h_cos = np.cos(2*np.pi*h/24)
```
Both sine and cosine are needed because one component alone maps distinct positions to the same value.

## Sigmoid
```python
z = (df["temp"]-df["temp"].mean())/df["temp"].std()
sigmoid = 1/(1+np.exp(-z))
```

## Lags / rolling
```python
s = df["cnt"]
df["lag1"] = s.shift(1)
df["lag24"] = s.shift(24)
df["roll24_mean"] = s.shift(1).rolling(24).mean()
df["roll24_std"] = s.shift(1).rolling(24).std()
```
For a prediction made at the start of period \(t\), `shift(1)` prevents current-period target information entering a rolling predictor.

---

# Notebook 3 — EDA / Matplotlib

A compact numeric grid:
```python
num = wine.select_dtypes(include=np.number).columns
fig, axes = plt.subplots(3, 4, figsize=(13, 9))
for ax, c in zip(axes.ravel(), num):
    ax.hist(wine[c].dropna(), bins=30)
    ax.set_title(c)
for ax in axes.ravel()[len(num):]:
    ax.axis("off")
fig.tight_layout()
```

Correlation:
```python
corr = wine[num].corr()
fig, ax = plt.subplots(figsize=(9,8))
im = ax.imshow(corr, vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(num))); ax.set_yticks(range(len(num)))
ax.set_xticklabels(num, rotation=90); ax.set_yticklabels(num)
fig.colorbar(im, ax=ax)
fig.tight_layout()
```

Bucket plot:
```python
tmp = wine[["alcohol","quality"]].copy()
tmp["bucket"] = pd.qcut(tmp["alcohol"], 10, duplicates="drop")
resp = tmp.groupby("bucket", observed=True)["quality"].mean()
resp.plot(marker="o")
```

The key interpretation point: use plots to decide what you need to test, not as an endpoint.

---

# Notebook 4 — Cleaning / Feature Engineering

## Audit table
```python
rows = []
for c in messy.columns:
    s = messy[c]
    numeric = pd.api.types.is_numeric_dtype(s)
    rows.append({
        "column": c,
        "dtype": str(s.dtype),
        "missing_n": s.isna().sum(),
        "missing_pct": s.isna().mean(),
        "unique": s.nunique(dropna=False),
        "inf_n": np.isinf(s).sum() if numeric else 0,
        "min": s.replace([np.inf,-np.inf],np.nan).min() if numeric else np.nan,
        "max": s.replace([np.inf,-np.inf],np.nan).max() if numeric else np.nan,
    })
audit = pd.DataFrame(rows)
```

## Constants / duplicate columns
```python
constant = [c for c in messy if messy[c].nunique(dropna=False) <= 1]

dup_cols = []
cols = list(messy.columns)
for i,c1 in enumerate(cols):
    for c2 in cols[i+1:]:
        if messy[c1].equals(messy[c2]):
            dup_cols.append((c1,c2))
```

## Sentinel / infinity
```python
messy.loc[messy["campaign"].eq(-999), "campaign"] = np.nan
messy["duration"] = messy["duration"].replace([np.inf,-np.inf], np.nan)
```

## Month cyclic
```python
month_map = dict(zip(
    ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"],
    range(1,13)
))
m = messy["month"].map(month_map)
messy["month_sin"] = np.sin(2*np.pi*m/12)
messy["month_cos"] = np.cos(2*np.pi*m/12)
```

## Leakage point
`duration` is the length of the current call. If the decision is made *before* the call, the eventual duration is unavailable. A model using it may look excellent but is not operationally valid.

---

# Notebook 5 — Regression / Validation

## Future target
```python
df["target"] = df["cnt"].shift(-1)
```
If predicting next-hour demand at the end of the current hour, current `cnt` can be valid. If predicting before current-hour demand is observed, it is not.

## Cyclic / lag / rolling
```python
dt = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
df["hour_sin"] = np.sin(2*np.pi*df["hr"]/24)
df["hour_cos"] = np.cos(2*np.pi*df["hr"]/24)
df["lag1"] = df["cnt"].shift(1)
df["lag24"] = df["cnt"].shift(24)
df["lag168"] = df["cnt"].shift(168)
df["roll24_mean"] = df["cnt"].shift(1).rolling(24).mean()
df["roll168_mean"] = df["cnt"].shift(1).rolling(168).mean()
```

## Chronological split
```python
n = len(model_df)
i1, i2 = int(.8*n), int(.9*n)
train = model_df.iloc[:i1]
valid = model_df.iloc[i1:i2]
test  = model_df.iloc[i2:]
```

## Metrics
```python
def reg_metrics(y, p):
    return {
        "rmse": mean_squared_error(y,p)**0.5,
        "mae": mean_absolute_error(y,p),
        "r2": r2_score(y,p)
    }
```

## Ridge
```python
ridge = Pipeline([
    ("scale", StandardScaler()),
    ("model", Ridge(alpha=1.0))
])
ridge.fit(X_train, y_train)
```

## Huber
```python
huber = Pipeline([
    ("scale", StandardScaler()),
    ("model", HuberRegressor())
])
```

## TimeSeriesSplit
```python
tscv = TimeSeriesSplit(n_splits=5)
for tr_idx, va_idx in tscv.split(X_train):
    ...
```

Shuffled K-fold lets the model learn from later periods when predicting earlier periods. That can be inappropriate if relationships drift or features contain temporal dependence.

## Target-bucket error
```python
out = pd.DataFrame({"y":y_valid, "pred":pred})
out["bucket"] = pd.qcut(out["y"], 5, duplicates="drop")
bucket_rmse = out.groupby("bucket", observed=True).apply(
    lambda g: np.sqrt(np.mean((g["y"]-g["pred"])**2))
)
```

---

# Notebook 6 — Classification

## ColumnTransformer
```python
num_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])
cat_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("oh", OneHotEncoder(handle_unknown="ignore"))
])
pre = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])
clf = Pipeline([
    ("pre", pre),
    ("model", LogisticRegression(max_iter=2000))
])
```

## Probability metrics
```python
prob = clf.predict_proba(X_valid)[:,1]
roc = roc_auc_score(y_valid, prob)
pr  = average_precision_score(y_valid, prob)
```

## Thresholds
```python
rows = []
for t in [0.1,0.2,0.3,0.4,0.5]:
    pred = (prob >= t).astype(int)
    rows.append({
        "threshold":t,
        "precision":precision_score(y_valid,pred,zero_division=0),
        "recall":recall_score(y_valid,pred,zero_division=0),
        "f1":f1_score(y_valid,pred,zero_division=0)
    })
pd.DataFrame(rows)
```

If false negatives cost more, lower thresholds may be rational even if precision falls.

## Permutation importance
Permutation importance answers: “how much does predictive performance deteriorate if this feature is scrambled?” It is global and can understate importance when highly correlated features substitute for one another.

SHAP explains each prediction relative to a baseline by allocating output contributions across features. It answers a different question.

---

# Closed-book test

Do not seek an exact answer key. The pass condition is methodological:
- correct data timing;
- no leakage;
- safe joins;
- appropriate validation;
- a baseline;
- a simple model and one nonlinear model;
- diagnostic metrics beyond one aggregate score;
- clear final conclusion.
