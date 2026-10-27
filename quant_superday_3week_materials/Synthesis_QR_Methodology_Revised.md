# Anchor, Correct, Guard: Forecasting Tomorrow on a Stock Panel with XGBoost

Forecasting tomorrow's value of a financial quantity looks like routine regression, but persistence, changing levels and shared market shocks complicate the task. For positive, heavy-tailed quantities, a few large stocks and extreme days can dominate raw RMSE. Performance must also generalise across several datasets. This procedure therefore follows three principles: anchor and correct, train against the scored metric, and retain a conservative fallback. The anchor is a simple forecast from the target's own history; XGBoost predicts departures from it.

The pipeline cleans and aligns the panel, constructs anchors and features, validates through time, improves the model through budgeted experiments, and deploys the selected procedure. Every forecast must be point-in-time defensible. The implementation agent works on the initial history; subsequent adaptation must be encoded in the program, without further LLM decisions.

## Understanding and cleaning the data

Use column names and observed distributions to establish the target's meaning, support and persistence. Apparent arithmetic identities guide feature construction, but do not automatically justify multiplying component forecasts: the conditional mean of a product also depends on conditional covariance. Unusually predictive columns trigger an alignment audit, not automatic removal.

Index the panel by trading sessions. Features at session t predict the same stock's target at the next market session, not its next observed row after a gap. At close T, train only on pairs whose outcomes are already available through T. Resolve duplicate stock-date records, preserve genuine zeros and convert identifiable missing-value sentinels to NaN. Leave missing features for XGBoost to handle; never interpolate using subsequent observations. Exclude unavailable labels from training, without removing stocks requiring forecasts or conditioning panel membership on future survival. Do not clip valid target extremes. Estimate all learned preprocessing quantities within each training window.

## The anchor

Trees extrapolate poorly beyond observed feature ranges. An anchor carries the current stock-specific level while trees learn corrections. Candidate anchors are yesterday's value, means over five, twenty-two and sixty-six sessions, their equal-weight blend, the stock's expanding historical mean, and the pooled training mean. The blend is the default, replaced when another candidate improves chronological validation RMSE. Score all anchors separately from the official baseline; beating an anchor does not necessarily imply positive official skill.

Use only observations available at each forecast date, allowing shorter trailing windows for young stocks. When recent observations are missing, use the last observed target with a staleness feature; without any stock history, use the pooled training mean. This ensures every stock in today's panel receives a forecast.

## Features

Compute stock histories after sorting by stock and date. Persistence features follow the multiscale idea of Corsi's HAR model: recent values and averages over a trading week, month and quarter, with ratios and logs only where defined. Calendar features describe the forecast session: weekday, calendar gap, month and quarter ends, holidays and neighbouring sessions. Add exchange-specific expiry or rebalance indicators only when their schedules are known in advance, rather than assuming all events occur on third Fridays.

Spike recency counts sessions since a realised target-to-anchor ratio exceeded its own historical 99th percentile, estimated using earlier observations. For signed targets, use additive departures. Market and sector features average realised departures from the previous session's anchor, capturing common movements without requiring PCA. Retain supplied raw features, adding cross-sectional ranks and trailing sixty-six-session z-scores where meaningful, together with history length, staleness and departure volatility. Undefined normalisations remain missing. Correlations and importance plots inform diagnostics; they do not exclude features that may work through nonlinearities or interactions. Evaluate feature families through chronological ablation.

## Validation that mirrors deployment

Reserve the final 15% of initial-history sessions as a final selection block; divide the preceding 30% into three validation blocks with expanding training histories. Split on unique dates. Each training pair must have its label observed before the simulated forecast; backward-looking features may legitimately use earlier sessions across a boundary. Choose boosting rounds using the last 10% of training sessions for early stopping, then refit on the whole training window using that round count. Replay the production refit schedule and fallback rules during validation.

Use the supplied scoring aggregation; if unspecified, minimise raw squared errors pooled over observed stock-day labels, reporting pooled RMSE and daily loss diagnostics. If the official constant baseline is undefined, report both historical-mean benchmarks explicitly without claiming exact official skill. Preserve stock-day weighting when comparing candidates. Shared shocks and serial dependence require uncertainty estimates from blocks of consecutive dates, not independent-row tests. Use a moving-block bootstrap with twenty-session blocks to assess paired loss differences; treat intervals as descriptive evidence after model selection, not multiplicity-adjusted significance tests.

Before experiments, verify baseline arithmetic, label alignment, complete prediction coverage, equality of features under future-data truncation with identical fitted preprocessing, and agreement between offline and daily replay predictions. Freeze the scorer and these checks.

## Fitting and the experiment loop

The metric shapes the loss. For strictly positive targets and anchors, default to predicting the ratio r=y/A with squared error and weights A squared, normalised to mean one before tuning regularisation. The identity (y-A*r_hat)^2=A^2*(r-r_hat)^2 aligns this data-fit term with raw squared error. Reconstruct forecasts as A*r_hat. For other targets, fit y-A with squared error and reconstruct A+correction. Neither formulation needs an additional anchor through base_margin.

For positive targets, also test raw-target Tweedie regression with variance power 1.5 and log(A) as base_margin, supplied during both fitting and prediction. Its log link produces A*exp(correction). Always compare reconstructed forecasts in raw units. Avoid treating exponentiated log-regression forecasts as conditional means; retransformation would require an additional, potentially conditional calibration.

Use CPU histogram trees, learning rate 0.05, depth six, and row and column subsampling of 0.8. Cap boosting at 2,000 rounds with fifty-round early stopping. Improvement follows an ordered queue: target formulation, feature-family ablations, depth three versus six, stronger leaf regularisation, and training weights with a one-year half-life versus equal weights. The leaf threshold controls summed Hessians, not independent dates; use min_child_weight ten and compare L2 penalties one and ten. Limit search to fifteen configurations including tuning trials and the core model defined below. Screen on the latest validation block, then replay the two best configurations and the core model across all three blocks. Choose the lowest pooled squared error; prefer a simpler candidate if its paired bootstrap interval against the best includes zero. Record fold results and investigate concentrated losses rather than requiring improvement on every fold.

Each experiment logs its hypothesis before execution and its configuration, data cutoff, scores, runtime and decision afterwards. Where supported, independent feature experiments run in isolated parallel waves, with a shared thirty-two-core budget and one integration step. An independent review checks timing and forecast reconstruction before integration. Keep concise persistent records so context limits do not erase decisions. Documentation lookup resolves implementation details; the specified candidate set bounds modelling exploration. Cache rolling features and profile refit and prediction costs over a full evaluation year.

## Deployment and guardrails

The final forecast is A+lambda*(M-A), where M is the reconstructed model forecast. Fit one lambda between zero and one by minimising squared error on walk-forward predictions; it defaults to zero if the model adds no improvement. This provides shrinkage, not a guarantee of future outperformance. Enforce only known support bounds, such as nonnegativity, rather than clipping predictions to historical extremes.

Before opening the final selection block, freeze three candidates: the selected pipeline, a core version with persistence and calendar features, and the anchor alone. Fit shrinkage separately for each model using the earlier validation predictions. Select the lowest-RMSE candidate on the final block, preferring the simpler one under the same bootstrap rule. This block performs selection; its score is not an unbiased final performance estimate. No new candidates are introduced afterwards.

Refit the selected model on all available initial history, including the final block, with frozen hyperparameters and shrinkage. During evaluation, update features daily and refit every twenty sessions using newly available labels. Persist model and rolling state between dates. Missing anchors or nonfinite model outputs invoke the defined historical fallback.

The procedure adapts through current anchors, daily features and scheduled refitting. Its purpose is a reproducible improvement over simple forecasts, supported by chronological evidence and explicit failure handling.
