# Python / Pandas / NumPy / Matplotlib Emergency Reference

## Python you actually need
```python
x = 10
cols = ["age", "bmi"]
lookup = {"jan": 1, "feb": 2}
for c in cols:
    print(c, df[c].mean())
def rmse(y, pred):
    return ((y-pred)**2).mean()**0.5
```

## Immediate audit
```python
df.head(); df.shape; df.columns; df.dtypes; df.info()
df.describe(); df.describe(include="all")
df.isna().sum(); df.isna().mean().sort_values(ascending=False)
df.duplicated().sum(); df.nunique().sort_values()
```
Compact audit:
```python
audit = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "missing_n": df.isna().sum(),
    "missing_pct": df.isna().mean(),
    "nunique": df.nunique(dropna=False)
})
```

## Filter / select
```python
df["age"]
df[["age","bmi"]]
df.loc[(df["age"] > 50) & (df["bmi"] < 20)]
df.iloc[:10,:5]
((df["age"] > 50) & (df["bmi"] < 20)).sum()
```
Use `&` / `|`, not Python `and` / `or`, for Series conditions.

## Sort / duplicates
```python
df.sort_values("date")
df.sort_values(["asset","date"])
df.drop_duplicates()
df.drop_duplicates(subset=["id","date"])
df.reset_index(drop=True)
```

## Groupby
```python
df.groupby("job")["income"].mean()
summary=(df.groupby("job")
    .agg(n=("income","size"), mean_income=("income","mean"), median_age=("age","median"))
    .reset_index())
df["job_mean"] = df.groupby("job")["income"].transform("mean")
```
`agg` reduces groups; `transform` returns one aligned value per original row.

## Merge / join / concat / cross
```python
out=left.merge(right,on="id",how="left",validate="many_to_one")
out=left.set_index("id").join(right.set_index("id"),how="left")
pd.concat([a,b],ignore_index=True)
grid=a.merge(b,how="cross")
```
Always ask what one row should represent after a merge.

## Pivot / melt / explode
```python
wide=pd.pivot_table(df,index="date",columns="asset",values="return",aggfunc="mean")
long=wide.reset_index().melt(id_vars="date",var_name="asset",value_name="return")
tmp=df.assign(tags=df["tags"].str.split("|")).explode("tags")
```

## Missing values
```python
df.dropna(subset=["target"])
df["x"].fillna(df["x"].median())
df["cat"].fillna("missing")
```
For modelling, fit imputation on training data only.

## Dates
```python
df["date"]=pd.to_datetime(df["date"])
df["month"]=df["date"].dt.month
df["dow"]=df["date"].dt.dayofweek
df["hour"]=df["date"].dt.hour
```
Cyclic:
```python
df["hour_sin"]=np.sin(2*np.pi*df["hour"]/24)
df["hour_cos"]=np.cos(2*np.pi*df["hour"]/24)
```

## Lags / rolling
```python
df["lag1"]=df["target"].shift(1)
df["lag24"]=df["target"].shift(24)
df["roll24"]=df["target"].shift(1).rolling(24).mean()
df["asset_lag1"]=df.groupby("asset")["return"].shift(1)
```

## Bins / ranks
```python
pd.cut(df["x"], bins=[0,10,20,30])
pd.qcut(df["x"],10,duplicates="drop")
df["rank"]=df["score"].rank(pct=True)
df["cs_rank"]=df.groupby("date")["score"].rank(pct=True)
```

## NumPy essentials
```python
x=df[["a","b","c"]].to_numpy(float)
x.shape; x.ndim; x[:,0]; x[:10]
x.mean(axis=0); x.mean(axis=1)
mask=x[:,0]>0; x[mask]
np.where(x>0,1,-1)
np.clip(x,-3,3)
np.quantile(x,[.01,.5,.99])
mu=x.mean(axis=0); sd=x.std(axis=0); z=(x-mu)/sd
np.corrcoef(x,rowvar=False)
```

## Useful transforms
```python
np.log1p(x)
lo,hi=df["x"].quantile([.01,.99]); df["x_clip"]=df["x"].clip(lo,hi)
z=(df["x"]-df["x"].mean())/df["x"].std(); sigmoid=1/(1+np.exp(-z))
df["interaction"]=df["temp"]*df["humidity"]
```
Transform because of a hypothesis, not because it exists.

## Matplotlib
```python
fig,ax=plt.subplots(figsize=(7,4)); ax.hist(df["target"].dropna(),bins=30); plt.show()
fig,ax=plt.subplots(figsize=(7,4)); ax.scatter(df["x"],df["target"],alpha=.3); plt.show()
fig,ax=plt.subplots(figsize=(10,4)); ax.plot(df["date"],df["target"]); plt.show()
fig,axes=plt.subplots(2,2,figsize=(10,7)); fig.tight_layout()
```
Correlation heatmap:
```python
corr=df.select_dtypes(include=np.number).corr()
fig,ax=plt.subplots(figsize=(8,7)); im=ax.imshow(corr,vmin=-1,vmax=1,aspect="auto"); fig.colorbar(im,ax=ax)
```

## First 20 minutes tomorrow
```python
df.head(); df.shape; df.info(); df.isna().sum(); df.duplicated().sum(); df.nunique(); df.describe()
```
Then identify target, unit of observation, numeric/categorical/time columns, leakage risks, target distribution and validation design.
