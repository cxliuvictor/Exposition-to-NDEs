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
