# Notes on Stochastic Calculus

## Ito's Lemma
For a function $f(t, X_t)$ where $X_t$ follows
$$
Ito’s lemma states:
df = \left(\frac{\partial f}{\partial t} + \mu \frac{\partial f}{\partial X} + \frac{1}{2}\sigma^2\frac{\partial^2 f}{\partial X^2}\right)dt + \sigma \frac{\partial f}{\partial X} dW_t
$$
```python
import numpy as np
print("Hello stochastic world")