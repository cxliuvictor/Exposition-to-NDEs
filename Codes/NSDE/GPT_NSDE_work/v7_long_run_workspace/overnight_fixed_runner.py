
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

import search_pipeline as sp
from audit_v7 import load_notebook_environment


SAVE_EVERY_UPDATES = 2
NUM_THREADS = 3

PRESETS = {
    "capped15_endpoint": {
        "objective": "mse",
        "updates": 15,
        "paths_per_step": 4096,
        "seed_offset": 0,
    },
    "capped20_twobatch": {
        "objective": "two_batch_mse",
        "updates": 20,
        "paths_per_step": 4096,
        # Separate training stream from capped MSE while keeping evaluation
        # randomness common across variants.
        "seed_offset": 300_000,
    },
}


def atomic_torch_save(payload, path: Path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def cpu_optimizer_state(state):
    """Move tensor-valued Adam state to CPU for a portable resume file."""
    out = {}
    for key, value in state.items():
        if isinstance(value, dict):
            out[key] = cpu_optimizer_state(value)
        elif isinstance(value, list):
            out[key] = [
                item.detach().cpu().clone() if torch.is_tensor(item) else item
                for item in value
            ]
        elif torch.is_tensor(value):
            out[key] = value.detach().cpu().clone()
        else:
            out[key] = value
    return out


def load_finished_model(ns, checkpoint_path: Path):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ns["MaturityHeadNeuralLocalVol"](payload["boundaries"]).cpu()
    model.load_state_dict(payload["final_model_state"], strict=True)
    model.eval()
    sp.set_grad(model, False)
    return model


def existing_row(path: Path, date: pd.Timestamp):
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        return None
    frame["date"] = pd.to_datetime(frame["date"])
    hit = frame.loc[frame["date"] == pd.Timestamp(date)]
    return None if hit.empty else hit.iloc[-1].to_dict()


def save_middate(
    path, signature, date, previous_date, model, hedge,
    opt_model, opt_hedge, sampler, history, update, elapsed
):
    payload = {
        "version": "v7_overnight_fixed_middate_v1",
        "signature": signature,
        "date": str(pd.Timestamp(date).date()),
        "previous_date": str(pd.Timestamp(previous_date).date()),
        "model_state": sp.cpu_state_dict(model),
        "hedge_state": {
            k: v.detach().cpu().clone() for k, v in hedge.state_dict().items()
        },
        "opt_model": cpu_optimizer_state(opt_model.state_dict()),
        "opt_hedge": cpu_optimizer_state(opt_hedge.state_dict()),
        "sampler_rng_state": sampler.rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "history": history,
        "last_update": int(update),
        "elapsed_seconds": float(elapsed),
    }
    atomic_torch_save(payload, path)


def train_fixed_endpoint(
    ns, *, name, objective, updates, paths_per_step, seed_offset,
    date, previous_model, previous_date, output_dir
):
    cfg = sp.SearchConfig(
        name=name,
        objective=objective,
        paths_per_step=paths_per_step,
        steps_per_phase=10,
        contracts_per_step=400,
        max_sde_updates=updates,
        early_min_sde=updates + 1,          # fixed endpoint: no early stopping
        early_patience_sde=updates + 1,
        monitor_paths=0,                    # monitor is intentionally not a selector
        sde_lr=1e-4,
        hedge_lr=1e-3,
        eval_paths=100000,
        seed_offset=seed_offset,
    )
    signature = {
        "name": name,
        "objective": objective,
        "updates": updates,
        "paths_per_step": paths_per_step,
        "steps_per_phase": cfg.steps_per_phase,
        "contracts_per_step": cfg.contracts_per_step,
        "sde_lr": cfg.sde_lr,
        "hedge_lr": cfg.hedge_lr,
        "seed_offset": seed_offset,
    }

    selected, _, _ = ns["daily_split"](date)
    ctx = ns["build_calibration_context"](date, selected)

    # Preserve the construction/order of the recovered search implementation.
    model = ns["MaturityHeadNeuralLocalVol"](ctx.maturity_boundaries).cpu()
    ns["transfer_rolling_model"](previous_model, model, previous_date, date)
    hedge = ns["IntervalHedgeBank"](ctx.maturity_boundaries, len(ctx.contracts)).cpu()

    opt_model = torch.optim.Adam(model.parameters(), lr=cfg.sde_lr)
    opt_hedge = torch.optim.Adam(hedge.parameters(), lr=cfg.hedge_lr)
    sampler = ns["UniformContractSampler"](
        ctx,
        cfg.contracts_per_step,
        ns["SEED"] + seed_offset + int(pd.Timestamp(date).day) * 101,
    )

    stamp = f"{pd.Timestamp(date):%Y%m%d}"
    progress_path = output_dir / f"{stamp}_inprogress.pt"
    history = []
    start_update = 0
    elapsed_before = 0.0

    if progress_path.exists():
        p = torch.load(progress_path, map_location="cpu", weights_only=False)
        if p.get("signature") != signature:
            raise RuntimeError(
                f"Resume signature mismatch for {progress_path}.\n"
                f"saved={p.get('signature')}\ncurrent={signature}"
            )
        model.load_state_dict(p["model_state"], strict=True)
        hedge.load_state_dict(p["hedge_state"], strict=True)
        opt_model.load_state_dict(p["opt_model"])
        opt_hedge.load_state_dict(p["opt_hedge"])
        sampler.rng.bit_generator.state = p["sampler_rng_state"]
        torch.set_rng_state(p["torch_rng_state"])
        history = list(p.get("history", []))
        start_update = int(p["last_update"])
        elapsed_before = float(p.get("elapsed_seconds", 0.0))
        print(
            f"[resume] {name} {pd.Timestamp(date).date()} "
            f"from update {start_update}/{updates}",
            flush=True,
        )
        del p
        gc.collect()
    else:
        # Same date-based training seed convention as the recovered search.
        torch.manual_seed(
            ns["SEED"] + seed_offset + int(pd.Timestamp(date).day) * 17
        )

    wall_start = time.time()

    update_bar = tqdm(
        range(start_update + 1, updates + 1),
        total=updates,
        initial=start_update,
        desc=f"{name} {pd.Timestamp(date):%m-%d} SDE updates",
        unit="upd",
        leave=False,
        dynamic_ncols=True,
    )
    for update in update_bar:
        # ----- SDE/model phase -----
        model.train()
        hedge.eval()
        sp.set_grad(model, True)
        sp.set_grad(hedge, False)
        model_losses = []

        for _ in range(cfg.steps_per_phase):
            ids = sampler.draw()
            opt_model.zero_grad(set_to_none=True)
            loss, scratch = sp.training_loss(
                ns, model, hedge, ctx, ids, objective, cfg
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt_model.step()
            model_losses.append(float(loss.detach()))
            del loss, scratch

        # ----- Sabate-style learned control-variate / hedge phase -----
        model.eval()
        hedge.train()
        sp.set_grad(model, False)
        sp.set_grad(hedge, True)
        hedge_losses = []

        for _ in range(cfg.steps_per_phase):
            ids = sampler.draw()
            z = torch.randn(
                cfg.paths_per_step,
                len(ctx.grid_np) - 1,
                dtype=torch.float32,
            )
            opt_hedge.zero_grad(set_to_none=True)
            _, cv = ns["simulate_estimators"](
                model, hedge, ctx, ids, z, "hedge"
            )
            variance = torch.var(
                cv / ctx.spot, dim=0, unbiased=False
            ).mean()
            variance.backward()
            nn.utils.clip_grad_norm_(hedge.parameters(), 3.0)
            opt_hedge.step()
            hedge_losses.append(float(variance.detach()))
            del z, cv, variance

        history.append({
            "update": update,
            "train_loss": float(np.mean(model_losses)),
            "hedge_variance_loss": float(np.mean(hedge_losses)),
        })
        update_bar.set_postfix(
            train=f"{history[-1]['train_loss']:.3g}",
            hedge=f"{history[-1]['hedge_variance_loss']:.3g}",
            refresh=False,
        )

        elapsed = elapsed_before + (time.time() - wall_start)
        if (
            update % SAVE_EVERY_UPDATES == 0
            or update == updates
        ):
            save_middate(
                progress_path, signature, date, previous_date,
                model, hedge, opt_model, opt_hedge, sampler,
                history, update, elapsed,
            )
            print(
                f"[checkpoint] {name} {pd.Timestamp(date).date()} "
                f"{update}/{updates} | train={history[-1]['train_loss']:.6g}",
                flush=True,
            )
        gc.collect()

    model.eval()
    sp.set_grad(model, False)

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(
        output_dir / f"{stamp}_history.csv", index=False
    )

    checkpoint = {
        "version": "v7_overnight_fixed_final_v1",
        "config": signature,
        "date": str(pd.Timestamp(date).date()),
        "previous_date": str(pd.Timestamp(previous_date).date()),
        "boundaries": ctx.maturity_boundaries,
        "endpoint_update": int(updates),
        "final_model_state": sp.cpu_state_dict(model),
    }
    atomic_torch_save(checkpoint, output_dir / f"{stamp}_checkpoint.pt")
    progress_path.unlink(missing_ok=True)

    meta = {
        "date": pd.Timestamp(date),
        "endpoint_update": int(updates),
        "objective": objective,
        "paths_per_step": int(paths_per_step),
        "elapsed_seconds": elapsed_before + (time.time() - wall_start),
    }

    del hedge, opt_model, opt_hedge
    gc.collect()
    return model, ctx, meta


def weighted_pooled(frame: pd.DataFrame):
    n = frame["otm_n"].to_numpy(float)
    pricing = frame["otm_pricing_RMSE"].to_numpy(float)
    hedge = frame["otm_NLV_hedge_RMSE"].to_numpy(float)
    market = frame["otm_market_delta_hedge_RMSE"].to_numpy(float)
    return {
        "otm_n": int(n.sum()),
        "pooled_otm_pricing_RMSE": float(
            np.sqrt(np.sum(n * pricing**2) / np.sum(n))
        ),
        "pooled_otm_NLV_hedge_RMSE": float(
            np.sqrt(np.sum(n * hedge**2) / np.sum(n))
        ),
        "pooled_otm_market_delta_hedge_RMSE": float(
            np.sqrt(np.sum(n * market**2) / np.sum(n))
        ),
        "weighted_otm_pricing_MAE": float(
            np.sum(n * frame["otm_pricing_MAE"].to_numpy(float)) / np.sum(n)
        ),
    }


def run_variant(preset_name: str):
    if preset_name not in PRESETS:
        raise KeyError(preset_name)
    preset = PRESETS[preset_name]

    output_dir = sp.RUN_ROOT / preset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(preset, indent=2), encoding="utf-8"
    )

    ns = load_notebook_environment()
    ns["stage_log"] = lambda message: None
    ns["FINAL_PATH_CHUNK"] = 2000
    torch.set_num_threads(NUM_THREADS)

    previous = sp.load_anchor(ns)
    previous_date = pd.Timestamp("2024-03-01")

    dates = [
        pd.Timestamp("2024-03-04"),
        pd.Timestamp("2024-03-05"),
        pd.Timestamp("2024-03-06"),
        pd.Timestamp("2024-03-07"),
    ]
    next_dates = [
        pd.Timestamp("2024-03-05"),
        pd.Timestamp("2024-03-06"),
        pd.Timestamp("2024-03-07"),
        pd.Timestamp("2024-03-08"),
    ]

    partial_summary_path = output_dir / "development_summary_partial.csv"
    partial_meta_path = output_dir / "training_metadata_partial.csv"
    summaries = []
    metadata = []

    date_pairs = list(zip(dates, next_dates))
    date_bar = tqdm(
        date_pairs,
        total=len(date_pairs),
        desc=f"{preset_name} rolling dates",
        unit="date",
        dynamic_ncols=True,
    )
    for index, (date, next_date) in enumerate(date_bar, start=1):
        date_bar.set_postfix_str(f"{date:%m-%d}->{next_date:%m-%d}", refresh=False)
        checkpoint_path = output_dir / f"{date:%Y%m%d}_checkpoint.pt"

        existing_summary = existing_row(partial_summary_path, date)
        existing_meta = existing_row(partial_meta_path, date)

        if checkpoint_path.exists():
            print(
                f"[reuse endpoint] {preset_name} {date.date()}",
                flush=True,
            )
            model = load_finished_model(ns, checkpoint_path)
            ctx = None
            if existing_meta is None:
                payload = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                meta = {
                    "date": date,
                    "endpoint_update": int(payload["endpoint_update"]),
                    "objective": preset["objective"],
                    "paths_per_step": int(preset["paths_per_step"]),
                    "elapsed_seconds": np.nan,
                }
                del payload
            else:
                meta = existing_meta
        else:
            print(
                f"[{preset_name}] train {date.date()} "
                f"({preset['updates']} fixed updates)",
                flush=True,
            )
            model, ctx, meta = train_fixed_endpoint(
                ns,
                name=preset_name,
                objective=preset["objective"],
                updates=int(preset["updates"]),
                paths_per_step=int(preset["paths_per_step"]),
                seed_offset=int(preset["seed_offset"]),
                date=date,
                previous_model=previous,
                previous_date=previous_date,
                output_dir=output_dir,
            )

        if existing_summary is None:
            print(
                f"[{preset_name}] evaluate {date.date()} -> "
                f"{next_date.date()} at 100k common-shock paths",
                flush=True,
            )
            summary = sp.evaluate_one_transition(
                ns,
                model,
                date,
                next_date,
                sp.SearchConfig(
                    name=preset_name,
                    eval_paths=100000,
                    seed_offset=int(preset["seed_offset"]),
                ),
                output_dir,
                index,
            )
        else:
            summary = existing_summary

        summaries.append(summary)
        metadata.append(meta)

        pd.DataFrame(summaries).to_csv(
            partial_summary_path, index=False
        )
        pd.DataFrame(metadata).to_csv(
            partial_meta_path, index=False
        )

        if ctx is not None:
            del ctx
        del previous
        previous = model
        previous_date = date
        gc.collect()

    date_bar.close()

    first = pd.read_csv(
        sp.BASE_CACHE / "v7_first_transition_summary.csv"
    ).iloc[0].to_dict()
    combined = pd.concat(
        [pd.DataFrame([first]), pd.DataFrame(summaries)],
        ignore_index=True,
    )
    combined.to_csv(
        output_dir / "development_summary.csv", index=False
    )
    pd.DataFrame(metadata).to_csv(
        output_dir / "training_metadata.csv", index=False
    )

    pooled = weighted_pooled(combined)
    pd.DataFrame([{
        "variant": preset_name,
        **pooled,
    }]).to_csv(output_dir / "pooled_summary.csv", index=False)

    print(
        f"\n[{preset_name}] COMPLETE | "
        f"pricing={pooled['pooled_otm_pricing_RMSE']:.6f} | "
        f"NLV hedge={pooled['pooled_otm_NLV_hedge_RMSE']:.6f} | "
        f"n={pooled['otm_n']}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        required=True,
        choices=sorted(PRESETS),
    )
    args = parser.parse_args()
    run_variant(args.preset)


if __name__ == "__main__":
    main()
