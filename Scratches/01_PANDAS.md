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
