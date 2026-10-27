# Kaggle-aligned Quant Dataset Interview Cheatsheets

**Purpose:** A compact, searchable, Python 3.8-friendly reference for an onsite Jupyter forecasting / supervised learning exercise. Use `Ctrl+F` in `ALL_IN_ONE.md` or open the relevant topic file.

**Notation:** **[K]** = method or topic aligned with the four specified Kaggle Learn courses. **[+]** = extra, kept because it can resolve a likely onsite obstacle; these additions are **not claimed to be covered by those four courses**.

## Open these files

| File | Find this when you need... |
|---|---|
| [00 Workflow](00_ONSITE_WORKFLOW.md) | A timed plan and the first 15 lines of a new notebook |
| [01 Pandas](01_PANDAS.md) | Select/filter, summarise, map/apply, group/sort, join/concat; extra merge/pivot |
| [02 NumPy](02_NUMPY.md) | Masks, arrays, broadcasting and simple feature transforms |
| [03 Plots](03_VISUALISATION.md) | First plots, conditional relationships, residuals |
| [04 Cleaning/features](04_CLEANING_FEATURE_ENGINEERING.md) | Missing/unknown/sentinel values, encoding, transformations, leakage |
| [05 sklearn](05_SKLEARN_MODELLING_VALIDATION.md) | Split, decision tree/random forest, pipeline, CV, metrics, boosting |
| [06 Time series](06_TIME_SERIES.md) | Lag/time-step/trend/seasonal/Fourier, chronological evaluation |
| [07 Statistics](07_STATS_DIAGNOSTICS.md) | Interpret metrics, correlation, errors by target bucket |
| [08 Jupyter/debug](08_JUPYTER_DEBUGGING.md) | Python survival, errors, paths, environment, package versions |
| [09 End-to-end recipes](09_COMMON_RECIPES.md) | Minimal tabular and forecasting pipelines, final notebook summary |
| [10 Optional methods](10_OPTIONAL_ML_METHODS.md) | Huber, Ridge, importance, PCA, SHAP concept, selected trading tools |
| [ALL_IN_ONE](ALL_IN_ONE.md) | Entire pack as one searchable Markdown file |

## How this differs from the earlier pack

1. **Start with the Kaggle course vocabulary and simplest syntax.** For example, `groupby`, `value_counts`, `map`, `apply`, `sort_values`, `concat`, and index-based `join` come before more elaborate recipes.
2. **One sentence of plain-English explanation comes before each code snippet.** Each file contains a tiny self-contained dataset so the examples can be run immediately.
3. **Onsite additions are deliberately small and labelled [+].** The highest-value ones are `merge(validate=...)`, `pivot_table`, `melt`, leakage-safe pipelines, target-bucket errors, and correct target/lag alignment.
4. **Keep complexity proportional to the task.** A mean/persistence baseline plus tree/random forest and sensible validation is more valuable than ten algorithms that you cannot explain.

## Kaggle course mapping and provenance

- [Pandas](https://www.kaggle.com/learn/pandas): Creating/reading/writing; indexing/selecting/assigning; summary functions/maps; grouping/sorting; data types/missing values; renaming/combining. Representative lesson material on `join` and `concat` is also preserved in a [public lesson mirror](https://plsms.github.io/kaggle/learn/3%20Pandas/6rf.%20Renaming%20and%20combining%20reference.html).
- [Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning): basic exploration, decision trees, validation, under/overfitting, random forests and competition workflow.
- [Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning): missing values, categorical variables, pipelines, cross-validation, XGBoost, data leakage.
- [Time Series](https://www.kaggle.com/learn/time-series): linear regression with time series; trend; seasonality; time series as features; hybrid models; forecasting with ML.

**Source limitation:** The Kaggle course landing pages and several representative published lesson transcripts/mirrors were accessible. Not every current tutorial cell was accessible directly. These are independently written condensed teaching notes, *not* verbatim reproductions or a claim that every individual method is in Kaggle Learn. `[+]` explicitly marks material beyond the four-course coverage.

**Compatibility:** Examples use conservative syntax suitable for Python 3.8 and standard pandas/NumPy/scikit-learn. Some APIs depend on the installed library version; verify with `pd.__version__`, `sklearn.__version__` if needed.

**Exam rules:** According to the company FAQ previously supplied, ordinary internet resources were allowed and AI assistance was prohibited. Follow any additional instructions given on the day; a public repo is not automatically exempt from newer restrictions.
