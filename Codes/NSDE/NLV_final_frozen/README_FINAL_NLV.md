# Frozen Neural Local Volatility pipeline

## 1. Final specification

The frozen Neural Local Volatility (NLV) model is `sabate_grid192`.

It combines the most useful structural features of the Sabaté-style NLV implementation with the one numerical ablation that improved the final pricing/hedging frontier:

- one independent local-volatility network per maturity interval;
- 4 hidden layers of width 50 with ReLU activations;
- Softplus output to keep local volatility positive;
- MSE calibration loss;
- 8,192 Monte Carlo paths per stochastic-gradient step, replayed in 512-path microbatches;
- 400 contracts per optimisation step;
- 15 paired calibration updates per rolling date;
- 10 SDE steps and 10 control-variate (CV) steps per paired update;
- Adam for both SDE and CV networks;
- SDE learning rate `1e-3`, multiplied by `0.2` after updates 8 and 12;
- CV learning rate `1e-3`;
- uniform contract sampling and contract weighting;
- fixed update 15 rather than checkpoint selection or early stopping;
- tamed Euler simulation;
- surrogate gradient through the tamed denominator;
- **192 steps/year plus exact option maturities**;
- no temporal penalty and no CV warm-start between dates;
- 40-step deterministic function distillation for rolling coefficient transfer;
- 50,000 paths for development evaluation and 100,000 paths for later confirmation/benchmark evaluation.

The frozen configuration is encoded directly in `nlv_core_final.py`; the runner intentionally exposes no architecture/configuration flag.

## 2. Relation to the Sabaté implementation

The final model is best described as a **Sabaté-structured, budget-matched NLV with a finer 192-step/year discretisation**.

It is Sabaté-like in the aspects that matter most structurally: independent maturity networks, 4×50 ReLU volatility networks with positive output, tamed Euler, MSE calibration, a learned martingale/control-variate hedge, and a high initial SDE learning rate with aggressive decay.

It is not a literal reproduction of the paper experiment. The original synthetic-data experiment used 40,000 paths per SGD iteration, 96 steps/year, a much longer training budget, and 400,000 evaluation paths. The present pipeline instead solves a daily rolling real-SPX problem under a fixed realistic compute budget. The 192-step grid is an empirical improvement selected from the frozen development ablations, so it is deliberately less paper-faithful than `sabate_budget_matched`.

## 3. Market data and cleaning

The pipeline reads four SPX files:

1. option quotes;
2. forward prices;
3. SPX spot prices;
4. zero-coupon yield curves.

The SPX identifier is `secid = 108105`. The base option filter keeps:

- European options only;
- contract size 100 and `cfadj = 1`;
- calls and puts;
- non-negative bid, ask strictly above bid and positive midpoint;
- 7–365 calendar days to expiry;
- market IV between `1e-4` and `5.0`;
- absolute market delta between `0.05` and `0.95`;
- relative bid-ask spread at most `25%`;
- absolute log-forward-moneyness `|log(K/F)| <= 0.30`;
- prices satisfying elementary discounted no-arbitrage bounds.

Spot, forward and interpolated zero rates are merged into each option row. For each contract,

\[
T = \frac{\text{days to expiry}}{365},\qquad
D=e^{-rT},\qquad
q=r-\frac{\log(F/S)}{T},\qquad
k=\log(K/F).
\]

The OTM surface uses the call when `K >= F` and the put otherwise. In a narrow ATM band `|k| <= 0.05`, if both call and put are available, the tighter-spread side is used. A date/expiry/settlement slice must contain at least eight usable points.

## 4. Frozen development protocol

The GPU model-selection window is March 1–8, 2024, with transitions:

- March 1 → March 4;
- March 4 → March 5;
- March 5 → March 6;
- March 6 → March 7;
- March 7 → March 8.

The primary population contains contracts that are OTM under the above rule on **both** dates and that match by the same option ID, call/put flag, strike, expiry and settlement convention. The frozen development population contains 20,001 matched contract observations across the five transitions.

The protocol hash fixes the dates, contract population, next-day join, hedge accounting convention, evaluation path count and evaluation shock grid. This prevents later ablations from silently changing the test population.

## 5. Risk-neutral model and normalisation

For deterministic rates/dividends, the spot model is

\[
dS_t=(r_t-q_t)S_t\,dt+S_t\sigma_{\theta,d}(t,X_t)\,dW_t.
\]

The implementation works in forward-normalised coordinates

\[
X_t=\frac{S_t}{F_d(t)},
\]

so under the pricing measure

\[
dX_t=X_t\sigma_{\theta,d}(t,X_t)\,dW_t.
\]

This removes deterministic carry from the simulated state. Option prices are reconstructed with the appropriate forward and discount factor for each expiry.

## 6. Final volatility architecture

Each maturity interval has its **own** network. If the ordered maturity boundaries are

\[
0<T_1<\cdots<T_m,
\]

then interval `j` uses a separate function

\[
\sigma_{\theta,j}(t,x).
\]

Each network receives `(t, x)` and has:

- 4 hidden layers;
- width 50;
- ReLU hidden activations;
- a one-dimensional Softplus output.

Weights use the Sabaté-style Xavier normal initialisation with gain `1.5`.

This differs from the earlier GPU reference, which shared a 4×50 trunk across maturities and used maturity-specific output heads. The independent architecture is more faithful to the released Sabaté NLV structure.

## 7. Time discretisation

The frozen model uses a base grid of 192 steps/year and inserts every observed option maturity exactly into the simulation grid.

With

\[
a_n=X_n\sigma_\theta(t_n,X_n),
\]

the tamed Euler step is

\[
X_{n+1}
=
X_n+
\frac{a_n\Delta W_n}{1+|a_n|\sqrt{\Delta t}}.
\]

For the frozen `surrogate` gradient, the `a_n` inside the taming denominator is detached when differentiating. The forward simulation itself is unchanged; this only modifies the training gradient.

The 192-step grid was selected because the standalone `grid192` ablation produced the strongest pricing improvement. When combined with the Sabaté-structured model, the three-seed final confirmation improved pricing without sacrificing the large hedging gain of the Sabaté-budget model.

## 8. March-1 anchor

The pipeline does **not** recalibrate March 1 from scratch.

It loads the frozen CPU checkpoint `initial_full_otm_refinement_20240301.pt`. If the checkpoint architecture is the earlier shared-trunk/maturity-head form, it is converted function-preservingly into independent interval networks by copying the shared trunk and the corresponding maturity head into each independent network.

The March-1 coefficient model is then kept fixed. Only the learned CV/hedge bank is fitted for 15 CV updates. This avoids spending the GPU budget rediscovering an already completed initial calibration while ensuring that the final hedge representation is compatible with the frozen GPU model.

## 9. Rolling transfer between dates

A daily recalibration cannot simply reuse the old network with the time origin reset. Both calendar time and the forward-normalised state change.

When rolling from date `d0` to `d1=d0+e`, a new-date state `x` at time-to-maturity `u` corresponds to the old state

\[
x_{\rm old}
=x\,\frac{F_{d_1}(u)}{F_{d_0}(u+e)}.
\]

The previous volatility function is therefore treated as a teacher in the new coordinates. The process is:

1. match each new maturity interval to the old interval at absolute time `u+e`;
2. shift the time input of the copied network by the elapsed calendar time;
3. apply the forward-ratio state map above;
4. distil the resulting teacher function into the new same-shaped network.

Distillation uses:

- 40 Adam steps;
- learning rate `5e-4`;
- 41 state probes from `0.70` to `1.30`;
- 3 time probes per maturity interval;
- AM/PM-settlement weights from the current contract population.

The transfer is accepted only if its post-distillation MSE is finite and does not deteriorate beyond the explicit tolerance.

## 10. Contract sampler and Monte Carlo budget

At each optimisation step:

- 400 eligible contracts are drawn uniformly without replacement where possible;
- contract weighting is uniform;
- 8,192 simulated paths are used;
- paths are processed in microbatches of 512 to control GPU memory.

The microbatching is designed to reconstruct the full 8,192-path objective rather than to optimise separate 512-path objectives.

## 11. Learned control variate / hedge

For each maturity interval there is a separate hedge network bank. Each per-option hedge network has:

- 3 hidden layers;
- width 30;
- ReLU activations;
- one output coefficient per option in the current calibration set.

For a discounted payoff estimator `Phi`, the network supplies a stochastic-integral control variate. Schematically,

\[
\Phi^{cv}
=
\Phi-\sum_n h_\xi(t_n,X_n)\,\Delta \bar S_n.
\]

The expectation is unchanged while the Monte Carlo variance can be reduced. The learned integrand is also retained as one model-native hedge convention.

Importantly, the final empirical study showed that stronger Monte Carlo variance reduction does **not** automatically imply lower realised one-day hedging error. The CV network therefore plays two related but distinct roles: variance reduction during calibration and a candidate hedge evaluated out of sample.

## 12. Alternating SDE/CV training

Each rolling date receives 15 paired updates. Each paired update contains:

1. 10 SDE coefficient optimisation steps;
2. 10 CV-network optimisation steps.

### SDE step

First, the code makes a no-gradient pass over all 8,192 paths to estimate the controlled option-price means. With target midprices `p_i`, the frozen loss is a spot-normalised weighted MSE,

\[
L(\theta)
=
\frac{1}{S_0^2}
\frac{1}{M}
\sum_{i=1}^M
\left(\widehat p_i(\theta)-p_i\right)^2.
\]

It then replays the **same shocks** in 512-path chunks and accumulates the gradient using the already-computed residual coefficients. This preserves the intended full-path estimator while avoiding storage of the whole 8,192-path computation graph.

The model gradient norm is clipped at 5.

### CV step

The coefficient model is frozen and the CV bank is trained to minimise the variance of the controlled estimator around its estimated mean. The CV gradient norm is clipped at 3.

### Learning-rate schedule

The SDE Adam rate is:

- updates 1–8: `1e-3`;
- updates 9–12: `2e-4`;
- updates 13–15: `4e-5`.

The CV Adam rate remains `1e-3`.

No bounded checkpoint selection or early stopping is used in the frozen model: update 15 is always retained. This is deliberate because the earlier checkpoint-selection ablation improved next-day pricing while materially damaging hedge quality.

## 13. Same-day pricing

For a calibrated date, terminal normalised states are simulated to every listed maturity. European call/put payoffs are evaluated contract by contract and multiplied by `D(T)F(T)` to return to price units.

Same-day diagnostics include RMSE, MAE and bid-ask coverage. These are secondary diagnostics; the main project objective is next-day performance.

## 14. Conditional next-day repricing

The model calibrated on date `d` is frozen. For the next trading date `d+1`, the pipeline uses the **realised next-day spot and deterministic curve** and asks how the frozen model reprices the matched option contracts.

This is therefore conditional next-day repricing, not a forecast of tomorrow's spot or yield curve.

The next-day pricing error is

\[
e^{price}_i
=\widehat C_{d+1,i}-C^{mkt}_{d+1,i}.
\]

The primary pricing metric is pooled OTM-on-both-dates RMSE in index points, with MAE, relative MAE, bid-ask coverage and implied-volatility diagnostics reported alongside it.

## 15. Delta and hedge conventions

Four hedge conventions are exported separately.

### 15.1 Learned-CV hedge

The learned control-variate integrand at the current state is interpreted in share-delta units. This is the default model-native headline hedge.

### 15.2 Frozen-physical delta

The local-volatility function is frozen and the initial state is bumped with common Brownian shocks. A central finite difference gives the sensitivity of the frozen model price to the physical spot perturbation. The routine bump is 5 bp.

This is the co-primary model-native hedge diagnostic.

### 15.3 Sticky-normalised delta

This is the historical re-centred-distribution heuristic retained from earlier versions. It is reported because it performs strongly empirically, but it is not labelled the true frozen-model delta.

### 15.4 OptionMetrics delta

The matched market delta is retained as an external benchmark. It does not depend on the NLV configuration.

## 16. One-day self-financing hedge accounting

All delta conventions use the same overnight accounting.

With current option price `C0`, spot `S0`, hedge `delta`, next-day prices `C1,S1`, overnight discount factor `D01`, and integrated dividend yield `Q01`, the one-period hedge error is implemented as

\[
E
=(C_1-C_0)
-\delta(S_1-S_0)
-\delta S_0\left(e^{Q_{01}}-1\right)
-(C_0-\delta S_0)\left(D_{01}^{-1}-1\right).
\]

Thus the stock position, dividend approximation and financing of the residual cash account are all treated consistently across model and market deltas.

## 17. Common-random-number evaluation

Evaluation uses antithetic Brownian paths and a common fine Brownian grid with 384 shock steps. The increments are aggregated onto each model's simulation grid so that discretisation comparisons use coupled shocks rather than unrelated Monte Carlo samples.

Development evaluation uses 50,000 paths. The frozen configuration retains 100,000 paths for later benchmark/confirmation evaluation and a quarantined optional 400,000-path audit budget.

## 18. Checkpointing, resumption and provenance

The production pipeline stores:

- model and CV states;
- both optimisers;
- sampler state;
- Python, NumPy, Torch and CUDA RNG states;
- completed update count;
- monitoring/training history;
- transfer diagnostics;
- configuration, data, protocol, CPU-anchor and executable-code digests.

In-progress checkpoints are written atomically. Repeating the same run identity resumes compatible work rather than silently mixing experiments.

The corrected `run_identity` canonicalises the configuration through JSON before comparing manifests, preventing Python tuple/list representation differences from creating false stale-cache failures.

The final runner marks the numerical run `completed` **before** optional report/plot subprocesses. A reporting or Matplotlib failure therefore cannot invalidate a completed expensive calibration.

## 19. Final model-selection evidence

The final configuration was not chosen on same-day fit alone. The selection criterion was the joint next-day pricing / one-day hedging frontier after the full NLV ablation programme.

Across confirmation seeds `12345`, `22345`, and `32345`, the frozen `sabate_grid192` model obtained mean ± sample-SD:

| Metric | Final NLV |
|---|---:|
| OTM next-day pricing RMSE | **4.1014 ± 0.2527** |
| OTM next-day pricing MAE | **3.2983 ± 0.1808** |
| Learned-CV hedge RMSE | **3.7680 ± 0.0742** |
| Frozen-physical hedge RMSE | **3.6483 ± 0.0473** |
| Sticky-normalised hedge RMSE | **2.0893 ± 0.0094** |
| OptionMetrics hedge RMSE | **1.8847** |
| Same-day OTM pricing RMSE | **2.2894 ± 0.0374** |
| Mean CV variance-reduction ratio | **1.7488 ± 0.0054** |

The two later seeds ran on RTX6000 while seed 12345 ran on the earlier final cohort hardware, so runtime is not treated as a hardware-invariant scientific metric. The scientific data/protocol/source implementation was held fixed.

The model still does not beat the OptionMetrics market delta benchmark. That negative result is retained rather than hidden by further tuning.

## 20. Contrast ablation: `random_maturity`

For the repository contrast, `random_maturity` is retained because it is a particularly informative failure rather than merely a slower or slightly worse variant.

On the common seed `12345`:

| Metric | Frozen `sabate_grid192` | `random_maturity` | Contrast |
|---|---:|---:|---:|
| OTM next-day pricing RMSE | 4.3931 | 4.2992 | random maturity 2.1% lower |
| Learned-CV hedge RMSE | **3.8453** | **6.0430** | random maturity **57.2% worse** |
| Frozen-physical hedge RMSE | **3.6621** | **4.5548** | random maturity **24.4% worse** |
| Sticky-normalised hedge RMSE | **2.0816** | 2.1140 | random maturity 1.6% worse |
| Same-day OTM pricing RMSE | **2.2596** | 3.3834 | random maturity 49.7% worse |

This is a useful contrast because next-day pricing alone would not flag it as a failure on this seed: its pricing RMSE is slightly lower than the frozen final model. Its learned-CV hedge, however, deteriorates dramatically, and the physical hedge also becomes materially worse. This directly illustrates one of the central empirical conclusions of the project: **good option-price fit or next-day repricing does not guarantee a good hedge**.

`random_maturity` samples one maturity group at a time rather than drawing contracts uniformly across the whole current surface. Under the short 15-update rolling budget, this appears to make the learned hedge particularly unstable across dates.

## 21. Files in this final record

- `nlv_core_final.py`: complete frozen numerical implementation;
- `run_nlv_final.py`: single-configuration CLI runner with crash-safe post-processing;
- `results/final_three_seed_results.csv`: the three final confirmation seeds;
- `results/final_three_seed_aggregate.csv`: mean/SD summary;
- `results/final_sabate_grid192/`: seed-12345 final run metadata, daily summary and compact report tables;
- `plots/sabate_grid192/`: original production plots for the frozen final model;
- `results/contrast_random_maturity/`: seed-12345 contrast outputs;
- `plots/random_maturity/`: daily pricing/hedging plots reconstructed from its saved daily summary;
- `plots/contrast/`: direct final-vs-random-maturity comparison plots;
- `results/summary_by_configuration_all.csv` and `completed_runs_all.csv`: the wider ablation record used for selection.

The deliberately large contract-level `diagnostic_contracts.csv` and training checkpoints are not included in the Git-oriented record. They remain in the HPC archive if exact contract-level regeneration is later required.

## 22. Reproduction command

With the four SPX CSV files and the CPU anchor in `data/`:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python run_nlv_final.py --mode development --seed 12345
```

Use `--check-inputs` first for a cheap login-node validation. Research runs require a CUDA allocation; do not run the expensive calibration on an HPC login node.

## 23. Status

The NLV architecture is now frozen. No additional NLV hyperparameter search should be performed on the March 1–8 development window. Future work should use this model as the fixed NLV comparator when implementing and evaluating Neural Stochastic Local Volatility (NSLV) under the same matched-data and hedging protocol.
