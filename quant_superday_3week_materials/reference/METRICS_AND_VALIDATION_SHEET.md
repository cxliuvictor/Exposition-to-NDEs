# Metrics and Validation Sheet

## Regression
\[
\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_i(y_i-\hat y_i)^2}
\]
Penalises large errors heavily.

\[
\mathrm{MAE}=\frac{1}{n}\sum_i|y_i-\hat y_i|
\]
More robust to large residuals.

\[
R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}
\]
Relative to mean prediction.

Also inspect:
- residual plots
- target-bucket RMSE/MAE
- period/regime errors
- Pearson/Spearman between prediction and target
- monotonic realised target by prediction quantile

## Huber
Quadratic for small residuals, approximately linear for large ones. Useful when you want robustness without completely ignoring tail observations.

## Classification
\[
\text{precision}=\frac{TP}{TP+FP},\quad
\text{recall}=\frac{TP}{TP+FN}
\]

\[
F_1=2\frac{\text{precision}\cdot\text{recall}}
{\text{precision}+\text{recall}}
\]

ROC-AUC measures ranking across FPR/TPR tradeoffs.
PR-AUC is often more informative under strong class imbalance.

Threshold choice is a *decision problem* and belongs on validation, not test.

## Trading
Daily Sharpe:
\[
\text{Sharpe}\approx \sqrt{252}\frac{\bar r}{s_r}
\]

Turnover:
\[
\mathrm{TO}_t = \sum_i |w_{i,t}-w_{i,t-1}|
\]

A simple cost model:
\[
r^{net}_t = r^{gross}_t - c\,\mathrm{TO}_t
\]

Drawdown from cumulative wealth/PnL:
\[
DD_t = W_t-\max_{s\le t}W_s
\]

Also report:
- hit rate
- number of trades
- mean/median trade PnL
- worst trade/day
- concentration of PnL
- subperiod Sharpe
- sensitivity to costs and thresholds

## Validation choice
IID tabular: stratified/random split or K-fold can be reasonable.
Time/order/group dependence: use chronological/group-aware validation.

Never:
- fit scaler/imputer on all data;
- select features with test target;
- tune threshold on test;
- shuffle a temporal prediction problem without justification.
