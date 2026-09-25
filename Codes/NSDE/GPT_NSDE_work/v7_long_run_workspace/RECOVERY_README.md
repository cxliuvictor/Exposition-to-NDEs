# Neural LV v7 experiment recovery bundle

This bundle captures the interrupted common-shock training and evaluation search.

## Frozen protocol

- Fixed model architecture: shared 4x50 ReLU trunk with maturity-specific Softplus heads.
- Primary selection: pooled OTM-on-both-dates next-day pricing RMSE.
- Tie-break: lowest pure Neural-LV hedge RMSE among pipelines within 2% of the best pricing RMSE.
- OptionMetrics delta is a comparator only; it is not blended into the Neural-LV hedge.
- Development dates: 2024-03-01 -> 2024-03-08 (five transitions).
- Untouched confirmation dates: 2024-03-08 -> 2024-03-28 (fourteen transitions).

## Completed results

### Common-shock v7 baseline

- 20,001 OTM contract-transition observations.
- Pooled next-day pricing RMSE: 4.4090076732.
- Pooled next-day pricing MAE: 3.5609592609.
- Pure Neural-LV hedge RMSE: 2.1050740252.
- OptionMetrics market-delta hedge RMSE: 1.9004002570.

### Robust Huber, 2024-03-04 -> 2024-03-05

- Huber threshold: 3 index points.
- 62 phases run; monitor best update 37; independent rerank selected update 59.
- Reranked same-day RMSE: 1.7293309553.
- Next-day OTM pricing RMSE: 5.0799764585.
- Pure Neural-LV hedge RMSE: 4.1593404055.

### Higher-patience MSE, 2024-03-04 -> 2024-03-05

- 100 phases run; monitor best update 93; independent rerank selected update 84.
- Reranked same-day RMSE: 1.7204639301.
- Next-day OTM pricing RMSE: 4.8510954019.
- Pure Neural-LV hedge RMSE: 4.1379688082.

## Interrupted runs

- `robust_mse8k`: March 4 did not reach a saved date boundary and must restart.
- `robust_twobatch`: March 4 did not reach a saved date boundary and must restart.
- `robust_twostage`: newly launched; no saved date boundary and must restart.
- `robust_mse` and `robust_huber3` have complete March 4 checkpoints. Their incomplete
  March 5 work was not saved; resume by loading the March 4 checkpoint and starting March 5.

## Important files

- `search_pipeline.py`: development training variants and robust checkpoint reranking.
- `evaluate_baseline_common.py`: reevaluates saved v7 models using common evaluation shocks.
- `analyze_search.py`: pooled pricing-first comparison and pure-NLV hedge tie-break.
- `run_frozen_test.py`: untouched later-March evaluation after the winner is frozen.
- `audit_v7.py`: residual, Monte Carlo checkpoint-noise, and delta-convention audits.
- `delta_stability.py`: bump-size and path-count delta stability audit.

The Python environment used Python 3.12 with PyTorch 2.9.1 CPU, NumPy 2.2.6,
pandas 2.2.2, and SciPy 1.14.1. The virtual environment itself is intentionally
excluded from the bundles.
