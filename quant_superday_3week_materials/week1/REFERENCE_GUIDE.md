# Week 1 master reference guide

Use while learning. By the closed-book test, you should not need it.

## Pandas
```python
df.shape; df.dtypes; df.info(); df.describe(include="all")
df.isna().sum(); df.duplicated().sum(); df.nunique()
df.loc[mask, ["a","b"]]; df.iloc[:10,:5]
df.sort_values(["date","score"], ascending=[True,False])
df.drop_duplicates(subset=["id","date"])
df.groupby("g").agg(n=("x","size"), mean_x=("x","mean")).reset_index()
df["demeaned"] = df["x"] - df.groupby("g")["x"].transform("mean")
df["lag1"] = df.groupby("asset")["ret"].shift(1)
df["roll20"] = df.groupby("asset")["ret"].transform(lambda s: s.shift(1).rolling(20).mean())
df.merge(other, on="id", how="left", validate="many_to_one")
left.join(right, how="left")
pd.concat([a,b], ignore_index=True)
a.merge(b, how="cross")
df.assign(genres=df["genres"].str.split("|")).explode("genres")
df.pivot_table(index="date", columns="asset", values="ret", aggfunc="mean")
wide.reset_index().melt(id_vars="date")
pd.qcut(df["score"], 10, labels=False, duplicates="drop")
df.groupby("date")["signal"].rank(pct=True)
```

## NumPy
```python
x.shape; x.ndim; x[:,0]; x[mask]
x.mean(axis=0); x.std(axis=1)
np.where(c,a,b); np.select([c1,c2],[v1,v2], default=0)
np.clip(x, lo, hi); np.quantile(x,[.01,.5,.99],axis=0)
np.argsort(x,axis=1); np.diff(x,axis=0); np.cumsum(x,axis=0)
np.corrcoef(x,rowvar=False); np.nanmean(x,axis=0)
```
Broadcasting: `(T,N)-(N,)` is columnwise; `(T,N)-(T,1)` is rowwise.

## Transformations
- right-skewed positive: `log1p`
- extreme tails: clip/winsorise, rank, robust loss/scaler
- cyclic period \(P\): `sin(2*pi*x/P)`, `cos(2*pi*x/P)`
- saturation: sigmoid or `tanh`
- persistence: lags and past-only rolling statistics
- interactions: product/ratio/spread only with a hypothesis
- event proximity: smooth decay such as `exp(-distance/tau)`

## Matplotlib
```python
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(x,y); ax.scatter(x,y,alpha=.3); ax.hist(x,bins=40)
ax.set(title="...", xlabel="...", ylabel="...")
fig.tight_layout()
fig.savefig("figure.png", dpi=180, bbox_inches="tight")
```

## Cleaning
Check types/date parsing, NaN/sentinels/inf, duplicates, constants, IDs, leakage, impossible values, outliers, high-cardinality categoricals, distribution shift. Fit preprocessing on training only.

## sklearn
Regression: mean/persistence baseline → Ridge/Huber → sklearn tree/boosting.
Classification: majority baseline → logistic → tree model.
Mixed columns: `ColumnTransformer` + `SimpleImputer` + `StandardScaler` + `OneHotEncoder(handle_unknown="ignore")`.

## Validation
IID: K-fold or train/valid/test. Time series: chronological split + `TimeSeriesSplit`, never shuffle. Freeze model/hyperparameters/threshold before OOS test.

Regression: RMSE, MAE, R², target-bucketed errors, residuals, rank correlation.
Classification: precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, threshold sensitivity.

## Importance
Permutation importance = loss in predictive score after shuffling a feature.
SHAP = per-observation attribution relative to a baseline.

## Trading
Prediction quality != strategy quality. Check ranking, thresholds, costs, turnover, hit rate, tail loss, cumulative PnL, drawdown and time stability. Align signals with future returns correctly.
