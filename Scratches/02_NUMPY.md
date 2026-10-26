# NumPy Cheatsheet

```python
import numpy as np
```

## Arrays / shape

```python
x = np.array([1,2,3])
X = df[["x1","x2","x3"]].to_numpy()

X.shape
X.ndim
X.size
```

## Indexing

```python
X[0]
X[:, 0]
X[:10, :3]

mask = X[:,0] > 0
X[mask]
```

## Aggregations and axes

```python
X.mean(axis=0)   # column means
X.mean(axis=1)   # row means

np.nanmean(X, axis=0)
np.nanstd(X, axis=0)
```

## Broadcasting / standardisation

```python
mu = np.nanmean(X, axis=0)
sd = np.nanstd(X, axis=0)
Z = (X - mu) / sd
```

Row-wise:

```python
mu_row = X.mean(axis=1, keepdims=True)
sd_row = X.std(axis=1, keepdims=True)
Z_row = (X - mu_row) / sd_row
```

## `where`

```python
signal = np.where(returns > 0, 1, -1)
```

Three-way:

```python
position = np.where(
    signal > 1,
    1,
    np.where(signal < -1, -1, 0)
)
```

## `select`

```python
conditions = [score < -1, score > 1]
choices = [-1, 1]
position = np.select(conditions, choices, default=0)
```

## Clip / quantiles

```python
lo, hi = np.quantile(x, [0.01, 0.99])
x_clip = np.clip(x, lo, hi)
```

Column quantiles:

```python
np.quantile(X, [0.01,0.5,0.99], axis=0)
```

## Sorting / ranking

```python
idx = np.argsort(x)
top_idx = np.argsort(-x)[:10]
```

## Difference / cumulative

```python
np.diff(x)
np.cumsum(x)
np.cumprod(1 + returns)
```

## Correlation / covariance

```python
np.corrcoef(X, rowvar=False)
np.cov(X, rowvar=False)
```

## Reproducible randomness

```python
rng = np.random.default_rng(42)
x = rng.normal(0, 1, size=1000)
```

## Log / sigmoid / tanh

```python
log_x = np.log1p(x)
sigmoid = 1 / (1 + np.exp(-x))
tanh_x = np.tanh(x)
```

## Cyclic encoding

```python
hour_sin = np.sin(2*np.pi*hour/24)
hour_cos = np.cos(2*np.pi*hour/24)
```

## Interactions

```python
interaction = x1 * x2
ratio = x1 / (x2 + 1e-8)
spread = x1 - x2
```

## Cross-sectional z-score

For matrix `(time, assets)`:

```python
mu = np.nanmean(signal, axis=1, keepdims=True)
sd = np.nanstd(signal, axis=1, keepdims=True)
z = (signal - mu) / sd
```

## Sharpe

```python
sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns, ddof=1)
```

## Max drawdown

```python
wealth = np.cumprod(1 + returns)
peak = np.maximum.accumulate(wealth)
drawdown = wealth / peak - 1
max_drawdown = drawdown.min()
```

## Common array mistake

If `x` is an array, do not write:

```python
if x > 0:
    ...
```

Use a boolean mask:

```python
mask = x > 0
```
