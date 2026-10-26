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
