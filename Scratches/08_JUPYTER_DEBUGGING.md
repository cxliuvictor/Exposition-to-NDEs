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
