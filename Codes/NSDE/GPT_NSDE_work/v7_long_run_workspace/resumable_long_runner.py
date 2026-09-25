
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

import search_pipeline as sp
from audit_v7 import load_notebook_environment


SAVE_EVERY_UPDATES = 5
NUM_THREADS = 3


def atomic_torch_save(payload, path: Path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_finished_model(ns, checkpoint_path: Path):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ns["MaturityHeadNeuralLocalVol"](payload["boundaries"]).cpu()
    model.load_state_dict(payload["final_model_state"], strict=True)
    model.eval()
    sp.set_grad(model, False)
    return model


def _save_middate(progress_path, cfg, date, previous_date, model, hedge,
                  opt_model, opt_hedge, sampler, history, candidates,
                  best, best_update, bad, last_update, elapsed_so_far):
    payload = {
        "version": "v7_long_runs_middate_v1",
        "config": sp.asdict(cfg),
        "date": str(pd.Timestamp(date).date()),
        "previous_date": str(pd.Timestamp(previous_date).date()),
        "model_state": sp.cpu_state_dict(model),
        "hedge_state": {k: v.detach().cpu().clone() for k, v in hedge.state_dict().items()},
        "opt_model": opt_model.state_dict(),
        "opt_hedge": opt_hedge.state_dict(),
        "sampler_rng_state": sampler.rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "history": history,
        "candidates": candidates,
        "best": float(best),
        "best_update": int(best_update),
        "bad": int(bad),
        "last_update": int(last_update),
        "elapsed_so_far": float(elapsed_so_far),
    }
    atomic_torch_save(payload, progress_path)


def train_day_resumable(ns, cfg, date, previous_model, previous_date, output_dir,
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

    stamp = f"{artifact_prefix}{pd.Timestamp(date):%Y%m%d}"
    progress_path = output_dir / f"{stamp}_inprogress.pt"

    history = []
    candidates = []

    def record(update, train_value=np.nan):
        model.eval()
        sp.set_grad(model, False)
        diagnostic = ns["objective_diagnostics"](model, ctx, z_monitor)
        score = float(diagnostic["raw_rmse"])
        candidates.append({
            "update": int(update),
            "monitor_rmse": score,
            "state": sp.cpu_state_dict(model),
        })
        history.append({
            "update": int(update),
            "train_loss": float(train_value),
            "monitor_rmse": score,
        })
        return score

    start_time = time.time()
    elapsed_before = 0.0
    start_update = 0

    if progress_path.exists():
        p = torch.load(progress_path, map_location="cpu", weights_only=False)
        if p.get("config", {}).get("name") != cfg.name:
            raise RuntimeError(f"Config mismatch in {progress_path}")
        model.load_state_dict(p["model_state"], strict=True)
        hedge.load_state_dict(p["hedge_state"], strict=True)
        opt_model.load_state_dict(p["opt_model"])
        opt_hedge.load_state_dict(p["opt_hedge"])
        sampler.rng.bit_generator.state = p["sampler_rng_state"]
        torch.set_rng_state(p["torch_rng_state"])
        history = list(p["history"])
        candidates = list(p["candidates"])
        best = float(p["best"])
        best_update = int(p["best_update"])
        bad = int(p["bad"])
        start_update = int(p["last_update"])
        elapsed_before = float(p.get("elapsed_so_far", 0.0))
        print(
            f"[resume] {cfg.name} {pd.Timestamp(date).date()} "
            f"from completed update {start_update}",
            flush=True,
        )
        del p
        gc.collect()
    else:
        torch.manual_seed(ns["SEED"] + cfg.seed_offset + int(pd.Timestamp(date).day) * 17)
        best = record(0)
        best_update = 0
        bad = 0

    update_bar = tqdm(
        range(start_update + 1, cfg.max_sde_updates + 1),
        total=cfg.max_sde_updates,
        initial=start_update,
        desc=f"{cfg.name} {pd.Timestamp(date):%Y-%m-%d}",
        unit="SDE",
        dynamic_ncols=True,
        mininterval=1.0,
    )

    for update in update_bar:
        model.train()
        hedge.eval()
        sp.set_grad(model, True)
        sp.set_grad(hedge, False)
        model_losses = []

        for _ in range(cfg.steps_per_phase):
            ids = sampler.draw()
            opt_model.zero_grad(set_to_none=True)
            loss, scratch = sp.training_loss(ns, model, hedge, ctx, ids, cfg.objective, cfg)
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

        stop_now = update >= cfg.early_min_sde and bad >= cfg.early_patience_sde

        # Match original search protocol: skip hedge phase if early stop has fired.
        if not stop_now:
            model.eval()
            hedge.train()
            sp.set_grad(model, False)
            sp.set_grad(hedge, True)
            for _ in range(cfg.steps_per_phase):
                ids = sampler.draw()
                z = torch.randn(
                    cfg.paths_per_step, len(ctx.grid_np) - 1, dtype=torch.float32
                )
                opt_hedge.zero_grad(set_to_none=True)
                _, cv = ns["simulate_estimators"](model, hedge, ctx, ids, z, "hedge")
                variance = torch.var(cv / ctx.spot, dim=0, unbiased=False).mean()
                variance.backward()
                nn.utils.clip_grad_norm_(hedge.parameters(), 3.0)
                opt_hedge.step()
                del z, cv, variance

        elapsed = elapsed_before + (time.time() - start_time)
        if (
            update % SAVE_EVERY_UPDATES == 0
            or stop_now
            or update == cfg.max_sde_updates
        ):
            _save_middate(
                progress_path, cfg, date, previous_date,
                model, hedge, opt_model, opt_hedge, sampler,
                history, candidates, best, best_update, bad, update, elapsed,
            )
            print(
                f"[checkpoint] {cfg.name} {pd.Timestamp(date).date()} "
                f"update={update} monitor={score:.4f} best={best:.4f}",
                flush=True,
            )

        if stop_now:
            print(
                f"[early stop] {cfg.name} {pd.Timestamp(date).date()} "
                f"at update {update}; best monitor update {best_update}",
                flush=True,
            )
            break

        gc.collect()

    del z_monitor, opt_model, opt_hedge, hedge
    gc.collect()

    history_frame = pd.DataFrame(history)

    # Same robust checkpoint selection as the recovered search.
    monitor_ids = set(
        history_frame.nsmallest(cfg.screen_top_k, "monitor_rmse")["update"].astype(int)
    )
    monitor_ids.update(
        history_frame.loc[history_frame["update"] % 5 == 0, "update"].astype(int)
    )
    monitor_ids.add(best_update)
    screened = [c for c in candidates if c["update"] in monitor_ids]

    screen, screen_seed_rows = sp.robust_scores(
        ns, ctx, screened, cfg.screen_paths, cfg.screen_seeds,
        ns["SEED"] + cfg.seed_offset + 700_000 + int(pd.Timestamp(date).day) * 100,
    )
    finalist_ids = set(screen.head(cfg.final_top_k)["update"].astype(int))
    finalists = [c for c in screened if c["update"] in finalist_ids]

    final, final_seed_rows = sp.robust_scores(
        ns, ctx, finalists, cfg.final_paths, cfg.final_seeds,
        ns["SEED"] + cfg.seed_offset + 900_000 + int(pd.Timestamp(date).day) * 100,
    )

    chosen_update = int(final.iloc[0]["update"])
    chosen = next(c for c in finalists if c["update"] == chosen_update)
    model.load_state_dict(chosen["state"], strict=True)
    model.eval()
    sp.set_grad(model, False)

    history_frame.to_csv(output_dir / f"{stamp}_history.csv", index=False)
    screen.to_csv(output_dir / f"{stamp}_screen.csv", index=False)
    screen_seed_rows.to_csv(output_dir / f"{stamp}_screen_seeds.csv", index=False)
    final.to_csv(output_dir / f"{stamp}_final_rerank.csv", index=False)
    final_seed_rows.to_csv(output_dir / f"{stamp}_final_rerank_seeds.csv", index=False)

    torch.save({
        "config": sp.asdict(cfg),
        "date": str(pd.Timestamp(date).date()),
        "previous_date": str(pd.Timestamp(previous_date).date()),
        "boundaries": ctx.maturity_boundaries,
        "chosen_update": chosen_update,
        "final_model_state": sp.cpu_state_dict(model),
        "candidate_states": candidates,
    }, output_dir / f"{stamp}_checkpoint.pt")

    progress_path.unlink(missing_ok=True)

    meta = {
        "date": pd.Timestamp(date),
        "updates_run": int(history_frame["update"].max()),
        "monitor_best_update": int(best_update),
        "chosen_update": chosen_update,
        "chosen_rerank_rmse": float(final.iloc[0].aggregate_rmse),
        "elapsed_seconds": elapsed_before + (time.time() - start_time),
        "stage": artifact_prefix.rstrip("_") or "forecast_full",
    }
    del candidates, screened, finalists
    gc.collect()
    return model, ctx, meta


def _existing_row(path: Path, date: pd.Timestamp):
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        return None
    frame["date"] = pd.to_datetime(frame["date"])
    hit = frame.loc[frame["date"] == pd.Timestamp(date)]
    return None if hit.empty else hit.iloc[-1].to_dict()


def run_development_resumable(variant):
    cfg = sp.VARIANTS[variant]
    output_dir = sp.RUN_ROOT / variant
    output_dir.mkdir(exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(sp.asdict(cfg), indent=2), encoding="utf-8"
    )

    ns = load_notebook_environment()
    ns["stage_log"] = lambda message: None
    ns["FINAL_PATH_CHUNK"] = 2000
    torch.set_num_threads(NUM_THREADS)

    previous = sp.load_anchor(ns)
    previous_date = pd.Timestamp("2024-03-01")

    dates = [pd.Timestamp(x) for x in [
        "2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07"
    ]]
    next_dates = [pd.Timestamp(x) for x in [
        "2024-03-05", "2024-03-06", "2024-03-07", "2024-03-08"
    ]]

    partial_summary_path = output_dir / "development_summary_partial.csv"
    partial_meta_path = output_dir / "training_metadata_partial.csv"
    summaries, metadata = [], []

    for index, (date, next_date) in enumerate(zip(dates, next_dates), start=1):
        checkpoint = output_dir / f"{date:%Y%m%d}_checkpoint.pt"
        transition = output_dir / f"transition_{date:%Y%m%d}.csv"

        if checkpoint.exists() and transition.exists():
            print(f"[skip completed] {variant} {date.date()}", flush=True)
            existing_s = _existing_row(partial_summary_path, date)
            existing_m = _existing_row(partial_meta_path, date)
            if existing_s is None or existing_m is None:
                raise RuntimeError(
                    f"Completed artifacts exist for {date.date()} but partial summary/meta is missing."
                )
            summaries.append(existing_s)
            metadata.append(existing_m)
            del previous
            previous = load_finished_model(ns, checkpoint)
            previous_date = date
            gc.collect()
            continue

        print(
            f"[{variant}] train {date.date()} -> evaluate {next_date.date()}",
            flush=True,
        )

        if cfg.two_stage:
            _, fit_contracts, _ = ns["daily_split"](date)
            fit_model, fit_ctx, fit_meta = train_day_resumable(
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
            model, ctx, meta = train_day_resumable(
                ns, polish_cfg, date, fit_model, date, output_dir
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
            model, ctx, meta = train_day_resumable(
                ns, cfg, date, previous, previous_date, output_dir
            )

        summary = sp.evaluate_one_transition(
            ns, model, date, next_date, cfg, output_dir, index
        )
        summaries.append(summary)
        metadata.append(meta)

        pd.DataFrame(summaries).to_csv(partial_summary_path, index=False)
        pd.DataFrame(metadata).to_csv(partial_meta_path, index=False)

        del ctx, previous
        previous, previous_date = model, date
        gc.collect()

    first = pd.read_csv(sp.BASE_CACHE / "v7_first_transition_summary.csv").iloc[0].to_dict()
    combined = pd.concat(
        [pd.DataFrame([first]), pd.DataFrame(summaries)], ignore_index=True
    )
    combined.to_csv(output_dir / "development_summary.csv", index=False)
    pd.DataFrame(metadata).to_csv(output_dir / "training_metadata.csv", index=False)
    print(f"[{variant}] complete", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(sp.VARIANTS), required=True)
    args = parser.parse_args()
    run_development_resumable(args.variant)


if __name__ == "__main__":
    main()
