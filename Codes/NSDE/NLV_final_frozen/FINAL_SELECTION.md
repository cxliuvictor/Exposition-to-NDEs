# Final NLV selection

Frozen model: `sabate_grid192`.

Contrast ablation retained in Git: `random_maturity` (seed 12345), because its next-day pricing is superficially competitive while its learned-CV hedge RMSE is 57.2% worse and frozen-physical hedge RMSE is 24.4% worse than `sabate_grid192` on the same seed. This makes it the clearest demonstration that pricing performance does not imply hedging quality.

Do not resume NLV tuning on the March 1–8 development window. The next model-development stage is NSLV.
