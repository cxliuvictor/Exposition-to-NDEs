# Feature Transformation Playbook

The hard interview skill is not knowing many transforms. It is generating a small number of plausible representations from the *structure of the variable*.

## 1. Ask what the variable physically means

Before transforming \(x\), ask:
- is it positive and multiplicative?
- bounded?
- cyclic?
- a time distance?
- an accumulated quantity?
- a noisy instantaneous measurement?
- a relative rather than absolute quantity?
- likely to interact with another variable?
- heavy-tailed?
- a category disguised as an integer?

## 2. Common patterns

### Positive/right-skewed
Examples: volume, income, market cap, counts.

Try:
```python
np.log1p(x)
```
Reason: ratios often matter more than absolute differences and tails are compressed.

### Cyclic
Examples: hour, weekday, month, angle.

Use both:
```python
sin = np.sin(2*np.pi*x/P)
cos = np.cos(2*np.pi*x/P)
```
One component alone loses position information.

### Saturating response
Examples: an effect rises quickly and then plateaus.

Try:
```python
1/(1+np.exp(-z))
np.tanh(z)
```

### Event proximity
A hard dummy may waste information around an event.

Examples:
```python
np.exp(-distance/tau)
1/(1+np.exp((distance-c)/scale))
```

### Persistence / delayed physical response
Examples: demand vs temperature, volatility, flow/pressure, customer activity.

Try:
- lag 1 / lag seasonal period
- rolling mean
- exponentially weighted mean
- rolling std/min/max

Always make the feature past-only.

### Cross-sectional relative signal
Raw level may be incomparable between entities.

Try:
- date-wise rank
- date-wise z-score
- demean by group/sector/date

### Heavy tails / outliers
Do not automatically delete them.

Try:
- clipping/winsorisation
- rank transform
- robust model/loss
- log if domain allows

Then explicitly inspect performance on the tail you compressed.

### Interactions
Only add if there is a mechanism.

Examples:
- temperature × humidity
- momentum / volatility
- signal × liquidity
- distance × event type

Linear models need explicit interactions; tree models can learn many interactions themselves.

## 3. Selection discipline

For each candidate transform:
1. state the hypothesis in one sentence;
2. build it using training/past data only;
3. compare against a simple untransformed baseline;
4. validate on the right split;
5. keep it only if it improves robustness or interpretability.

Avoid "feature soup": dozens of arbitrary transforms create multiple-testing risk and make the final presentation indefensible.
