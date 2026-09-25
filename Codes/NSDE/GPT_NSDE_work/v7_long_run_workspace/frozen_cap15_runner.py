
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import search_pipeline as sp
import overnight_fixed_runner as ofr
from audit_v7 import load_notebook_environment


def load_model(ns, checkpoint_path: Path):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ns["MaturityHeadNeuralLocalVol"](payload["boundaries"]).cpu()
    model.load_state_dict(payload["final_model_state"], strict=True)
    model.eval()
    sp.set_grad(model, False)
    return model


def existing_row(path: Path, date: pd.Timestamp):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"])
    hit = df.loc[df["date"] == pd.Timestamp(date)]
    return None if hit.empty else hit.iloc[-1].to_dict()


def pooled(df: pd.DataFrame):
    n = df["otm_n"].to_numpy(float)
    p = df["otm_pricing_RMSE"].to_numpy(float)
    h = df["otm_NLV_hedge_RMSE"].to_numpy(float)
    m = df["otm_market_delta_hedge_RMSE"].to_numpy(float)
    out = {
        "otm_n": int(n.sum()),
        "pooled_otm_pricing_RMSE": float(np.sqrt(np.sum(n * p**2) / np.sum(n))),
        "pooled_otm_NLV_hedge_RMSE": float(np.sqrt(np.sum(n * h**2) / np.sum(n))),
        "pooled_otm_market_delta_hedge_RMSE": float(np.sqrt(np.sum(n * m**2) / np.sum(n))),
    }
    if "otm_pricing_MAE" in df.columns:
        out["weighted_otm_pricing_MAE"] = float(
            np.sum(n * df["otm_pricing_MAE"].to_numpy(float)) / np.sum(n)
        )
    return out


def train_confirmation():
    ns = load_notebook_environment()
    ns["stage_log"] = lambda message: tqdm.write(str(message))
    ns["FINAL_PATH_CHUNK"] = 2000
    torch.set_num_threads(3)

    dev_dir = sp.RUN_ROOT / "capped15_endpoint"
    dev_last = dev_dir / "20240307_checkpoint.pt"
    if not dev_last.exists():
        raise FileNotFoundError(
            f"Missing {dev_last}. Run/extract the completed capped15_endpoint development branch first."
        )

    out = sp.RUN_ROOT / "frozen_test_capped15_endpoint"
    out.mkdir(parents=True, exist_ok=True)

    previous = load_model(ns, dev_last)
    previous_date = pd.Timestamp("2024-03-07")

    trading_dates = [pd.Timestamp(x) for x in ns["trading_dates"]]
    start = trading_dates.index(pd.Timestamp("2024-03-08"))
    training_dates = trading_dates[start:-1]
    target_dates = trading_dates[start+1:]
    assert len(training_dates) == 14 and len(target_dates) == 14

    partial_summary = out / "unseen_summary_partial.csv"
    partial_meta = out / "training_metadata_partial.csv"
    summaries, metadata = [], []

    transitions = list(zip(training_dates, target_dates))
    progress = tqdm(
        transitions,
        total=len(transitions),
        desc="Frozen test: train + 100k eval",
        unit="transition",
        dynamic_ncols=True,
        mininterval=0.5,
    )

    for index, (date, target) in enumerate(progress, start=1):
        progress.set_description(f"{date:%m-%d} -> {target:%m-%d}")
        progress.set_postfix_str("checking checkpoint")
        cp = out / f"{date:%Y%m%d}_checkpoint.pt"
        old_summary = existing_row(partial_summary, date)
        old_meta = existing_row(partial_meta, date)

        if cp.exists():
            progress.set_postfix_str("reuse checkpoint")
            tqdm.write(f"[reuse] {date.date()} endpoint checkpoint")
            model = load_model(ns, cp)
            ctx = None
            if old_meta is None:
                payload = torch.load(cp, map_location="cpu", weights_only=False)
                meta = {
                    "date": date,
                    "endpoint_update": int(payload["endpoint_update"]),
                    "objective": "mse",
                    "paths_per_step": 4096,
                    "elapsed_seconds": np.nan,
                }
                del payload
            else:
                meta = old_meta
        else:
            progress.set_postfix_str("training cap15 MSE")
            tqdm.write(f"[train] {date.date()} -> {target.date()} | cap15 MSE")
            model, ctx, meta = ofr.train_fixed_endpoint(
                ns,
                name="frozen_test_capped15_endpoint",
                objective="mse",
                updates=15,
                paths_per_step=4096,
                seed_offset=0,
                date=date,
                previous_model=previous,
                previous_date=previous_date,
                output_dir=out,
            )

        if old_summary is None:
            progress.set_postfix_str("evaluating 100k")
            tqdm.write(f"[eval 100k] {date.date()} -> {target.date()}")
            cfg = sp.SearchConfig(
                name="frozen_test_capped15_endpoint",
                eval_paths=100000,
                seed_offset=0,
            )
            summary = sp.evaluate_one_transition(
                ns, model, date, target, cfg, out, index
            )
        else:
            progress.set_postfix_str("reuse 100k result")
            summary = old_summary

        summaries.append(summary)
        metadata.append(meta)
        pd.DataFrame(summaries).to_csv(partial_summary, index=False)
        pd.DataFrame(metadata).to_csv(partial_meta, index=False)

        if ctx is not None:
            del ctx
        del previous
        previous = model
        previous_date = date
        gc.collect()

    sdf = pd.DataFrame(summaries)
    sdf.to_csv(out / "unseen_summary.csv", index=False)
    pd.DataFrame(metadata).to_csv(out / "training_metadata.csv", index=False)
    p = pooled(sdf)
    pd.DataFrame([{"variant": "frozen_test_capped15_endpoint", **p}]).to_csv(
        out / "pooled_summary.csv", index=False
    )
    print("\nPRIMARY FROZEN CONFIRMATION COMPLETE")
    print(p)


def eval_400k():
    ns = load_notebook_environment()
    ns["stage_log"] = lambda message: tqdm.write(str(message))
    ns["FINAL_PATH_CHUNK"] = 2000
    torch.set_num_threads(3)

    src = sp.RUN_ROOT / "frozen_test_capped15_endpoint"
    if not (src / "unseen_summary.csv").exists():
        raise FileNotFoundError("Run the primary 100k frozen confirmation first.")

    out = sp.RUN_ROOT / "frozen_test_capped15_endpoint_eval400k"
    out.mkdir(parents=True, exist_ok=True)

    trading_dates = [pd.Timestamp(x) for x in ns["trading_dates"]]
    start = trading_dates.index(pd.Timestamp("2024-03-08"))
    training_dates = trading_dates[start:-1]
    target_dates = trading_dates[start+1:]

    partial = out / "unseen_summary_400k_partial.csv"
    summaries = []

    transitions = list(zip(training_dates, target_dates))
    progress = tqdm(
        transitions,
        total=len(transitions),
        desc="400k robustness eval",
        unit="transition",
        dynamic_ncols=True,
        mininterval=0.5,
    )

    for index, (date, target) in enumerate(progress, start=1):
        progress.set_description(f"400k {date:%m-%d} -> {target:%m-%d}")
        progress.set_postfix_str("checking cached result")
        old = existing_row(partial, date)
        if old is not None:
            progress.set_postfix_str("reuse 400k result")
            summaries.append(old)
            tqdm.write(f"[reuse 400k] {date.date()} -> {target.date()}")
            continue

        cp = src / f"{date:%Y%m%d}_checkpoint.pt"
        model = load_model(ns, cp)
        cfg = sp.SearchConfig(
            name="frozen_test_capped15_endpoint_eval400k",
            eval_paths=400000,
            seed_offset=0,
        )
        progress.set_postfix_str("evaluating 400k")
        tqdm.write(f"[eval 400k] {date.date()} -> {target.date()}")
        s = sp.evaluate_one_transition(ns, model, date, target, cfg, out, index)
        summaries.append(s)
        pd.DataFrame(summaries).to_csv(partial, index=False)
        del model
        gc.collect()

    sdf = pd.DataFrame(summaries)
    sdf.to_csv(out / "unseen_summary_400k.csv", index=False)
    p = pooled(sdf)
    pd.DataFrame([{
        "variant": "frozen_test_capped15_endpoint_eval400k",
        **p
    }]).to_csv(out / "pooled_summary_400k.csv", index=False)
    print("\n400K ROBUSTNESS EVALUATION COMPLETE")
    print(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train100k", "eval400k"], required=True)
    args = ap.parse_args()
    if args.mode == "train100k":
        train_confirmation()
    else:
        eval_400k()


if __name__ == "__main__":
    main()
