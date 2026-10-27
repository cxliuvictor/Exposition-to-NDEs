# Python 3.8 / onsite environment notes

Prefer conservative syntax and core libraries.

## Safe defaults
- NumPy
- pandas
- matplotlib
- scikit-learn
- Python stdlib

## Avoid depending on
- very new Python syntax
- a recent pandas method unless you know it exists
- XGBoost / LightGBM / SHAP / Optuna being preinstalled
- GPU libraries

If a non-standard package would save only a few minutes, do not install it.

## Useful startup cell
```python
import os, sys
import numpy as np
import pandas as pd
import matplotlib
import sklearn

print(sys.version)
print(sys.executable)
print(os.getcwd())
print(np.__version__)
print(pd.__version__)
print(matplotlib.__version__)
print(sklearn.__version__)
```

## Jupyter actions to know
- Shift+Enter
- interrupt
- restart kernel
- restart and run all
- clear outputs
- `%timeit`
- `%pip install package` if genuinely necessary

A clean notebook should run top-to-bottom after restarting the kernel.
