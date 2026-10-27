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
