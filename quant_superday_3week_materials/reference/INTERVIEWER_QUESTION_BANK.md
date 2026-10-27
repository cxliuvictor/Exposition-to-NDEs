# Dataset-interview defence question bank

## Data / cleaning
- What is one row?
- Which columns can only be known after the target?
- Why did you impute rather than drop?
- Could your merge have multiplied rows?
- Did you find duplicate keys?
- How did you treat sentinel values?
- Why is that outlier invalid rather than merely unusual?
- Is there distribution shift?

## Features
- Why that transform?
- Why sine and cosine rather than the raw month?
- Why that rolling window?
- Is your rolling feature past-only?
- Why log rather than rank-transform?
- Why remove/retain correlated variables?
- Which interactions have a plausible mechanism?
- What would happen if the relationship changes regime?

## Validation
- Why random K-fold / why not?
- Why TimeSeriesSplit?
- What is your untouched test set?
- Which decisions were made on validation?
- How variable are fold scores?
- Does performance survive the last period?
- How would you detect leakage from preprocessing?

## Models
- Why Ridge?
- Why Huber?
- Why a tree/GBM?
- What does regularization buy you?
- Why did the nonlinear model improve?
- How much tuning did you do?
- Could a simpler model be preferable?

## Metrics
- Why RMSE?
- Why MAE?
- What do target-bucketed errors reveal?
- Why use Spearman?
- Why PR-AUC rather than accuracy?
- What threshold would you choose if false negatives cost 5×?

## Interpretation
- What does permutation importance measure?
- Why can importance be unstable with correlated variables?
- How is SHAP different?
- Which feature is genuinely causal? (Usually: you cannot claim causality.)

## Trading
- When exactly is your signal observed?
- Which future return does it earn?
- Are costs included?
- What creates turnover?
- Is performance concentrated in a few trades?
- Why top 20% rather than top 10%?
- Does the threshold survive OOS?
- What is the simple non-ML baseline?

## Presentation
- What is the one result you want me to remember?
- What did not work?
- What is your biggest caveat?
- What would you do with another two hours?
