# 3-Week Quant Dataset Interview Preparation Pack

This pack is designed for a Squarepoint-style onsite dataset interview:
- Jupyter / Python
- standard NumPy, pandas, matplotlib, scikit-learn
- internet documentation allowed, but practise without AI
- generic datasets, often but not always time series
- several hours of independent research followed by a presentation and defence

The progression is deliberately:
**fluency → research judgement → realistic mocks**.

---

# WEEK 1 — Fluency and fundamentals

Goal: by the end of the week, the mechanics should be automatic enough that you can spend interview time thinking about the data rather than recalling syntax.

## Day 0 / warm-up
`week1/00_environment_and_jupyter.ipynb`
- Jupyter kernel controls
- working directory / relative paths
- package versions
- `%timeit`
- figure export for PowerPoint
- package-install judgement

## Day 1
`week1/01_pandas_transformations.ipynb`
- filtering / conditional counts
- groupby / named agg / transform
- merge / join / concat
- explode
- pivot / pivot_table / melt
- Cartesian products
- ranks / qcut
- shift / rolling
- merge-cardinality traps

## Day 2
`week1/02_numpy_and_feature_transforms.ipynb`
- shape / axes / slicing
- broadcasting
- masks / where / select
- quantiles / clipping
- argsort / vectorisation
- log / sigmoid / tanh
- cyclic sine/cosine encodings
- interactions
- lags / rolling transforms

## Day 3
`week1/03_eda_and_matplotlib.ipynb`
- first-5-minute audit
- histogram/subplot grids
- scatter / box / bucket plots
- correlation heatmap
- outlier views
- time-series plots
- one presentation-quality figure

## Day 4
`week1/04_cleaning_and_feature_engineering.ipynb`
- NaN / sentinel / inf
- constants / near constants
- duplicate rows / duplicate columns
- categorical `unknown`
- leakage
- IDs
- outliers / skew
- correlated features
- train-only preprocessing

## Day 5
`week1/05_sklearn_regression_and_validation.ipynb`
- next-step prediction
- mean/persistence/seasonal-naive baselines
- Ridge
- Huber
- HistGradientBoosting
- 80/10/10 chronological split
- KFold vs TimeSeriesSplit
- target-bucket errors
- residual diagnostics
- permutation importance
- frozen OOS test

## Day 6
`week1/06_sklearn_classification_and_interpretability.ipynb`
- imbalanced classification
- ColumnTransformer
- logistic regression
- nonlinear classifier
- precision/recall/F1
- ROC-AUC / PR-AUC
- threshold selection
- permutation importance vs SHAP concept
- frozen OOS test

## Day 7
`week1/07_closed_book_mastery_test.ipynb`
- 120-minute no-AI/no-reference test
- redo weak sections until you can finish at least ~80% without leakage or validation errors

Use:
- `week1/REFERENCE_GUIDE.md`
- `week1/MASTERY_CHECKLIST.md`
- `week1/solutions/ANSWER_KEY.md`

The answer key is for checking *after* a genuine attempt, not for following line-by-line.

---

# WEEK 2 — Five 2.5–3 hour research projects

Every project uses a real public dataset and requires a small presentation.

## Project 1 — Anonymous tabular regression
`week2/01_anonymous_tabular_superconductivity.ipynb`
- UCI Superconductivity
- ~80 anonymous numeric descriptors
- IID split + K-fold
- Ridge vs nonlinear model
- motivated transforms
- target-bucket errors
- permutation importance
- 3-slide mini-deck

## Project 2 — Cross-sectional equity research
`week2/02_cross_sectional_sp500.ipynb`
- S&P 500 constituent daily OHLCV panel
- momentum / reversal / volatility / liquidity features
- cross-sectional rank/z-score
- chronological validation
- daily IC / decile spread
- pooled Ridge / nonlinear model
- long-short portfolio
- turnover / costs / drawdown / regime stability

## Project 3 — Demand forecasting and feature engineering
`week2/03_pjm_demand_feature_engineering.ipynb`
- PJM hourly electricity demand
- cyclic calendar transforms
- lags / rolling statistics
- holiday / bridge-day proximity
- seasonal naive baselines
- TimeSeriesSplit
- feature ablations
- error slices

## Project 4 — Generic mixed-type classification
`week2/04_adult_income_classification.ipynb`
- UCI Adult Census Income
- missing categorical values
- redundant representations
- logistic pipeline
- nonlinear comparison
- threshold / asymmetric cost
- PR-AUC / ROC-AUC
- probability-decile calibration

## Project 5 — Messy sensor time series
`week2/05_air_quality_sensor_regression.ipynb`
- UCI Air Quality
- real `-200` sentinels
- missingness and sensor drift
- cyclic / lag / rolling features
- chronological validation
- Ridge / Huber / nonlinear model
- high-pollution error slices
- drift robustness

### Week 2 rule
For each project:
1. set a timer;
2. no AI;
3. stop at the stated time;
4. make 3–4 slides;
5. give a 5-minute verbal summary;
6. use `reference/INTERVIEWER_QUESTION_BANK.md` to defend it.

Do not automatically reuse the same features/model. The point is to decide what is appropriate for each dataset.

---

# WEEK 3 — Realistic mocks

## Mock 1 — 3 hours
`week3/01_mock_3h_ames_housing.ipynb`
- Kaggle Ames House Prices
- generic mixed-type regression
- hard stop
- 4 slides
- 5-minute presentation

## Mock 2 — 4 hours
`week3/02_mock_4h_btc_trade_selection.ipynb`
- real BTC-USD OHLCV
- long / short / no-trade decision
- simple technical/statistical baselines
- ML only if it adds OOS value
- transaction-cost sensitivity
- frozen threshold
- 4–5 slides
- 7-minute defence

## Mock 3 — full 09:00–14:00 dress rehearsal
`week3/03_full_day_mock_appliances_energy.ipynb`
- UCI Appliances Energy Prediction
- 10-minute multivariate time series
- 10–11 checkpoint
- 12–1 lunch
- stop modelling 13:35
- presentation at 14:00
- 5–6 slides
- immediate defence

After Mock 3:
`week3/CLOSING_INTERVIEW_DRILL.md`

---

# KAGGLE LEARN BALANCE

Do not finish courses before starting projects. Alternate course concepts with notebooks.

## Essential
- **Pandas — 100%**
- **Intermediate Machine Learning — 100%**
- **Time Series — 100%**

## Selective
- **Data Cleaning — missing values, scaling/normalisation, parsing dates**
- **Feature Engineering — mutual information, creating features, PCA, target encoding**
- **Data Visualization — line/scatter/distributions/plot choice**
- **Machine Learning Explainability — permutation importance; SHAP concepts only**

## Approximate allocation
- Week 1: ~70–75% notebooks, ~25–30% Kaggle Learn
- Week 2: ~80–85% projects, ~15–20% finishing/refreshing Kaggle
- Week 3: almost entirely timed mocks + review

---

# Interview research workflow

Use `reference/RESEARCH_WORKFLOW_9_TO_2.md`.

The core loop is:

**understand → audit → plot → clean → hypothesise → transform → baseline → model → validate → stress-test → translate to decision → present**

By the first interviewer checkpoint you should already have:
- a coherent description of the data;
- main quality/leakage concerns;
- a baseline;
- at least one plausible relationship;
- a validation design;
- a next-step plan.

At ~13:35 in the full mock, stop adding models and build the presentation.

---

# Data policy

The project/mock notebooks load public real datasets from GitHub/UCI and cache them locally under `data/cache/` after the first successful run.

Each loader checks expected columns and a minimum row count so that a broken or wrong download fails loudly rather than silently changing the exercise.

See:
`reference/DATASET_SOURCES.md`

If a mirror ever disappears, retrieve the named canonical dataset and keep the same project brief.
