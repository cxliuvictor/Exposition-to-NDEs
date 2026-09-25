from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from audit_v7 import load_notebook_environment


ROOT = Path(__file__).resolve().parent
BASE_CACHE = ROOT / "cache_original" / "neural_lv_v7_final_safe_cache"
RUN_ROOT = ROOT / "search_runs"
RUN_ROOT.mkdir(exist_ok=True)


@dataclass(frozen=True)
class SearchConfig:
    name: str
    objective: str = "mse"
    huber_delta_dollars: float = 3.0
    paths_per_step: int = 4096
    steps_per_phase: int = 10
    contracts_per_step: int = 400
    max_sde_updates: int = 100
    early_min_sde: int = 25
    early_patience_sde: int = 25
    monitor_paths: int = 32768
    sde_lr: float = 1e-4
    hedge_lr: float = 1e-3
    screen_top_k: int = 18
    screen_paths: int = 32768
    screen_seeds: int = 2
    final_top_k: int = 5
    final_paths: int = 100000
    final_seeds: int = 3
    eval_paths: int = 100000
    seed_offset: int = 0
    two_stage: bool = False
    polish_max_sde_updates: int = 60
    polish_early_min_sde: int = 15
    polish_early_patience_sde: int = 20


VARIANTS = {
    "robust_mse": SearchConfig(name="robust_mse"),
    "robust_mse8k": SearchConfig(
        name="robust_mse8k", paths_per_step=8192
    ),
    "robust_huber3": SearchConfig(
        name="robust_huber3", objective="huber", huber_delta_dollars=3.0,
    ),
    "robust_twobatch": SearchConfig(
        name="robust_twobatch", objective="two_batch_mse", paths_per_step=4096,
    ),
    "robust_twostage": SearchConfig(
        name="robust_twostage", objective="mse", two_stage=True,
    ),
}


def cpu_state_dict(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def set_grad(module, value):
    for parameter in module.parameters():
        parameter.requires_grad_(value)


def load_anchor(ns):
    payload = torch.load(
        BASE_CACHE / "forecast_full_20240301_resume.pt",
        map_location="cpu", weights_only=False,
    )
    model = ns["MaturityHeadNeuralLocalVol"](payload["boundaries"]).cpu()
    model.load_state_dict(payload["final_model_state"], strict=True)
    model.eval()
    set_grad(model, False)
    return model


def robust_scores(ns, ctx, candidates, paths, seeds, seed_base):
    """Score candidate price surfaces using independent MC seeds and common random numbers."""
    model = ns["MaturityHeadNeuralLocalVol"](ctx.maturity_boundaries).cpu().eval()
    per_seed = []
    square_sums = {int(c["update"]): 0.0 for c in candidates}
    abs_sums = {int(c["update"]): 0.0 for c in candidates}
    count = len(ctx.contracts) * seeds
    for seed_index in range(seeds):
        z = ns["antithetic_normals"](
            paths, len(ctx.grid_np) - 1, seed_base + seed_index
        )
        for candidate in candidates:
            update = int(candidate["update"])
            model.load_state_dict(candidate["state"], strict=True)
            diag = ns["objective_diagnostics"](model, ctx, z)
            errors = diag["table"]["raw_error"].to_numpy(float)
            square_sums[update] += float(np.sum(errors * errors))
            abs_sums[update] += float(np.sum(np.abs(errors)))
            per_seed.append({
                "update": update,
                "seed_index": seed_index,
                "paths": paths,
                "rmse": float(np.sqrt(np.mean(errors * errors))),
                "mae": float(np.mean(np.abs(errors))),
            })
        del z
        gc.collect()
    summary = pd.DataFrame([
        {
            "update": update,
            "aggregate_rmse": math.sqrt(square_sums[update] / count),
            "aggregate_mae": abs_sums[update] / count,
        }
        for update in square_sums
    ]).sort_values(["aggregate_rmse", "aggregate_mae"]).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1)
    return summary, pd.DataFrame(per_seed)


def training_loss(ns, model, hedge, ctx, ids, objective, cfg):
    ids_tensor = torch.as_tensor(ids, dtype=torch.long)
    n_steps = len(ctx.grid_np) - 1
    if objective == "two_batch_mse":
        z1 = torch.randn(cfg.paths_per_step, n_steps, dtype=torch.float32)
        z2 = torch.randn(cfg.paths_per_step, n_steps, dtype=torch.float32)
        _, cv1 = ns["simulate_estimators"](model, hedge, ctx, ids, z1, "sde")
        _, cv2 = ns["simulate_estimators"](model, hedge, ctx, ids, z2, "sde")
        r1 = (cv1.mean(0) - ctx.target[ids_tensor]) / ctx.spot
        r2 = (cv2.mean(0) - ctx.target[ids_tensor]) / ctx.spot
        # Independent products remove the positive Monte Carlo variance term in squared loss.
        loss = (r1 * r2).mean()
        return loss, (z1, z2, cv1, cv2)

    z = torch.randn(cfg.paths_per_step, n_steps, dtype=torch.float32)
    _, cv = ns["simulate_estimators"](model, hedge, ctx, ids, z, "sde")
    residual_dollars = cv.mean(0) - ctx.target[ids_tensor]
    if objective == "mse":
        loss = (residual_dollars / ctx.spot).square().mean()
    elif objective == "huber":
        delta = float(cfg.huber_delta_dollars)
        absolute = residual_dollars.abs()
        huber = torch.where(
            absolute <= delta,
            0.5 * residual_dollars.square(),
            delta * (absolute - 0.5 * delta),
        )
        loss = huber.mean() / (ctx.spot * ctx.spot)
    else:
        raise ValueError(f"Unknown objective {objective}")
    return loss, (z, cv)


def train_day(ns, cfg, date, previous_model, previous_date, output_dir,
              contracts=None, artifact_prefix=""):
    selected, _, _ = ns["daily_split"](date)
    training_contracts = selected if contracts is None else contracts
    ctx = ns["build_calibration_context"](date, training_contracts)
    model = ns["MaturityHeadNeuralLocalVol"](ctx.maturity_boundaries).cpu()
    ns["transfer_rolling_model"](previous_model, model, previous_date, date)
    hedge = ns["IntervalHedgeBank"](ctx.maturity_boundaries, len(ctx.contracts)).cpu()
    opt_model = torch.optim.Adam(model.parameters(), lr=cfg.sde_lr)
    opt_hedge = torch.optim.Adam(hedge.parameters(), lr=cfg.hedge_lr)
    sampler = ns["UniformContractSampler"](
        ctx, cfg.contracts_per_step,
        ns["SEED"] + cfg.seed_offset + int(pd.Timestamp(date).day) * 101,
    )
    monitor_seed = ns["SEED"] + cfg.seed_offset + int(pd.Timestamp(date).day) * 1009
    z_monitor = ns["antithetic_normals"](
        cfg.monitor_paths, len(ctx.grid_np) - 1, monitor_seed
    )
    history = []
    candidates = []

    def record(update, train_value=np.nan):
        model.eval()
        set_grad(model, False)
        diagnostic = ns["objective_diagnostics"](model, ctx, z_monitor)
        score = float(diagnostic["raw_rmse"])
        candidates.append({"update": int(update), "monitor_rmse": score, "state": cpu_state_dict(model)})
        history.append({
            "update": int(update), "train_loss": float(train_value),
            "monitor_rmse": score,
        })
        return score

    torch.manual_seed(ns["SEED"] + cfg.seed_offset + int(pd.Timestamp(date).day) * 17)
    best = record(0)
    best_update = 0
    bad = 0
    start = time.time()

    for update in range(1, cfg.max_sde_updates + 1):
        model.train()
        hedge.eval()
        set_grad(model, True)
        set_grad(hedge, False)
        model_losses = []
        for _ in range(cfg.steps_per_phase):
            ids = sampler.draw()
            opt_model.zero_grad(set_to_none=True)
            loss, scratch = training_loss(ns, model, hedge, ctx, ids, cfg.objective, cfg)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt_model.step()
            model_losses.append(float(loss.detach()))
            del loss, scratch

        score = record(update, np.mean(model_losses))
        if score < best:
            best, best_update, bad = score, update, 0
        else:
            bad += 1

        if update >= cfg.early_min_sde and bad >= cfg.early_patience_sde:
            break

        # Retain v7's alternating variance-reduction hedge phase.
        model.eval()
        hedge.train()
        set_grad(model, False)
        set_grad(hedge, True)
        for _ in range(cfg.steps_per_phase):
            ids = sampler.draw()
            z = torch.randn(cfg.paths_per_step, len(ctx.grid_np) - 1, dtype=torch.float32)
            opt_hedge.zero_grad(set_to_none=True)
            _, cv = ns["simulate_estimators"](model, hedge, ctx, ids, z, "hedge")
            variance = torch.var(cv / ctx.spot, dim=0, unbiased=False).mean()
            variance.backward()
            nn.utils.clip_grad_norm_(hedge.parameters(), 3.0)
            opt_hedge.step()
            del z, cv, variance
        gc.collect()

    del z_monitor, opt_model, opt_hedge, hedge
    gc.collect()

    history_frame = pd.DataFrame(history)
    # Monitor top set plus a regular time-grid guard against a noisy monitor omitting regions.
    monitor_ids = set(history_frame.nsmallest(cfg.screen_top_k, "monitor_rmse")["update"].astype(int))
    monitor_ids.update(history_frame.loc[history_frame["update"] % 5 == 0, "update"].astype(int))
    monitor_ids.add(best_update)
    screened = [c for c in candidates if c["update"] in monitor_ids]
    screen, screen_seed_rows = robust_scores(
        ns, ctx, screened, cfg.screen_paths, cfg.screen_seeds,
        ns["SEED"] + cfg.seed_offset + 700_000 + int(pd.Timestamp(date).day) * 100,
    )
    finalist_ids = set(screen.head(cfg.final_top_k)["update"].astype(int))
    finalists = [c for c in screened if c["update"] in finalist_ids]
    final, final_seed_rows = robust_scores(
        ns, ctx, finalists, cfg.final_paths, cfg.final_seeds,
        ns["SEED"] + cfg.seed_offset + 900_000 + int(pd.Timestamp(date).day) * 100,
    )
    chosen_update = int(final.iloc[0]["update"])
    chosen = next(c for c in finalists if c["update"] == chosen_update)
    model.load_state_dict(chosen["state"], strict=True)
    model.eval()
    set_grad(model, False)

    stamp = f"{artifact_prefix}{pd.Timestamp(date):%Y%m%d}"
    history_frame.to_csv(output_dir / f"{stamp}_history.csv", index=False)
    screen.to_csv(output_dir / f"{stamp}_screen.csv", index=False)
    screen_seed_rows.to_csv(output_dir / f"{stamp}_screen_seeds.csv", index=False)
    final.to_csv(output_dir / f"{stamp}_final_rerank.csv", index=False)
    final_seed_rows.to_csv(output_dir / f"{stamp}_final_rerank_seeds.csv", index=False)
    torch.save({
        "config": asdict(cfg), "date": str(pd.Timestamp(date).date()),
        "previous_date": str(pd.Timestamp(previous_date).date()),
        "boundaries": ctx.maturity_boundaries,
        "chosen_update": chosen_update,
        "final_model_state": cpu_state_dict(model),
        "candidate_states": candidates,
    }, output_dir / f"{stamp}_checkpoint.pt")

    meta = {
        "date": pd.Timestamp(date), "updates_run": int(history_frame["update"].max()),
        "monitor_best_update": int(best_update), "chosen_update": chosen_update,
        "chosen_rerank_rmse": float(final.iloc[0].aggregate_rmse),
        "elapsed_seconds": time.time() - start,
        "stage": artifact_prefix.rstrip("_") or "forecast_full",
    }
    del candidates, screened, finalists
    gc.collect()
    return model, ctx, meta


def evaluate_one_transition(ns, model, date, next_date, cfg, output_dir, index):
    # Use common evaluation randomness across pipeline variants.  Training
    # randomness is handled separately; changing a method must not also change
    # the Monte Carlo sample on which its next-day metrics are compared.
    state = ns["evaluate_transition_state"](
        model, date, paths=cfg.eval_paths,
        seed=ns["SEED"] + 1_200_000 + index * 100,
    )
    summary, contracts = ns["evaluate_transition"](
        model, date, next_date, state, paths=cfg.eval_paths,
        seed=ns["SEED"] + 1_300_000 + index * 100,
    )
    contracts.to_csv(output_dir / f"transition_{pd.Timestamp(date):%Y%m%d}.csv", index=False)
    del state, contracts
    gc.collect()
    return summary


def run_development(variant):
    cfg = VARIANTS[variant]
    output_dir = RUN_ROOT / variant
    output_dir.mkdir(exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    ns = load_notebook_environment()
    ns["stage_log"] = lambda message: None
    ns["FINAL_PATH_CHUNK"] = 2000
    torch.set_num_threads(1)

    # Same March-1 anchor and first transition for every pipeline; subsequent dates differ.
    previous = load_anchor(ns)
    previous_date = pd.Timestamp("2024-03-01")
    dates = [pd.Timestamp(x) for x in ["2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07"]]
    next_dates = [pd.Timestamp(x) for x in ["2024-03-05", "2024-03-06", "2024-03-07", "2024-03-08"]]
    summaries = []
    metadata = []
    for index, (date, next_date) in enumerate(zip(dates, next_dates), start=1):
        print(f"[{variant}] train {date.date()} -> evaluate {next_date.date()}", flush=True)
        if cfg.two_stage:
            _, fit_contracts, _ = ns["daily_split"](date)
            fit_model, fit_ctx, fit_meta = train_day(
                ns, cfg, date, previous, previous_date, output_dir,
                contracts=fit_contracts, artifact_prefix="fit80_",
            )
            polish_cfg = replace(
                cfg,
                two_stage=False,
                max_sde_updates=cfg.polish_max_sde_updates,
                early_min_sde=cfg.polish_early_min_sde,
                early_patience_sde=cfg.polish_early_patience_sde,
            )
            model, ctx, meta = train_day(
                ns, polish_cfg, date, fit_model, date, output_dir,
            )
            meta.update({
                "fit80_updates_run": fit_meta["updates_run"],
                "fit80_monitor_best_update": fit_meta["monitor_best_update"],
                "fit80_chosen_update": fit_meta["chosen_update"],
                "fit80_chosen_rerank_rmse": fit_meta["chosen_rerank_rmse"],
                "fit80_elapsed_seconds": fit_meta["elapsed_seconds"],
            })
            del fit_model, fit_ctx
        else:
            model, ctx, meta = train_day(ns, cfg, date, previous, previous_date, output_dir)
        summary = evaluate_one_transition(ns, model, date, next_date, cfg, output_dir, index)
        summaries.append(summary)
        metadata.append(meta)
        pd.DataFrame(summaries).to_csv(output_dir / "development_summary_partial.csv", index=False)
        pd.DataFrame(metadata).to_csv(output_dir / "training_metadata_partial.csv", index=False)
        del ctx, previous
        previous, previous_date = model, date
        gc.collect()

    # Prepend the common first transition from the uploaded v7 cache.
    first = pd.read_csv(BASE_CACHE / "v7_first_transition_summary.csv").iloc[0].to_dict()
    combined = pd.concat([pd.DataFrame([first]), pd.DataFrame(summaries)], ignore_index=True)
    combined.to_csv(output_dir / "development_summary.csv", index=False)
    pd.DataFrame(metadata).to_csv(output_dir / "training_metadata.csv", index=False)
    print(f"[{variant}] complete", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    args = parser.parse_args()
    run_development(args.variant)


if __name__ == "__main__":
    main()
