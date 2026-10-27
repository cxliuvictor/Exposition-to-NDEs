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
