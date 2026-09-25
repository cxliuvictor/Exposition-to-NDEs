from __future__ import annotations

import argparse
import gc
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

from audit_v7 import load_notebook_environment
from search_pipeline import RUN_ROOT, VARIANTS, evaluate_one_transition, set_grad, train_day


def load_development_model(ns, variant):
    payload = torch.load(
        RUN_ROOT / variant / "20240307_checkpoint.pt",
        map_location="cpu", weights_only=False,
    )
    model = ns["MaturityHeadNeuralLocalVol"](payload["boundaries"]).cpu()
    model.load_state_dict(payload["final_model_state"], strict=True)
    model.eval()
    set_grad(model, False)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    args = parser.parse_args()
    cfg = VARIANTS[args.variant]
    output = RUN_ROOT / f"frozen_test_{args.variant}"
    output.mkdir(exist_ok=True)

    ns = load_notebook_environment()
    ns["stage_log"] = lambda message: None
    ns["FINAL_PATH_CHUNK"] = 2000
    torch.set_num_threads(1)
    previous = load_development_model(ns, args.variant)
    previous_date = pd.Timestamp("2024-03-07")

    trading_dates = [pd.Timestamp(x) for x in ns["trading_dates"]]
    start = trading_dates.index(pd.Timestamp("2024-03-08"))
    training_dates = trading_dates[start:-1]
    target_dates = trading_dates[start + 1:]
    summaries, metadata = [], []
    for index, (date, target) in enumerate(zip(training_dates, target_dates), start=1):
        print(f"[frozen {args.variant}] train {date.date()} -> evaluate {target.date()}", flush=True)
        if cfg.two_stage:
            _, fit_contracts, _ = ns["daily_split"](date)
            fit_model, fit_ctx, fit_meta = train_day(
                ns, cfg, date, previous, previous_date, output,
                contracts=fit_contracts, artifact_prefix="fit80_",
            )
            polish_cfg = replace(
                cfg,
                two_stage=False,
                max_sde_updates=cfg.polish_max_sde_updates,
                early_min_sde=cfg.polish_early_min_sde,
                early_patience_sde=cfg.polish_early_patience_sde,
            )
            model, ctx, meta = train_day(ns, polish_cfg, date, fit_model, date, output)
            meta.update({
                "fit80_updates_run": fit_meta["updates_run"],
                "fit80_chosen_update": fit_meta["chosen_update"],
                "fit80_chosen_rerank_rmse": fit_meta["chosen_rerank_rmse"],
                "fit80_elapsed_seconds": fit_meta["elapsed_seconds"],
            })
            del fit_model, fit_ctx
        else:
            model, ctx, meta = train_day(ns, cfg, date, previous, previous_date, output)
        summary = evaluate_one_transition(ns, model, date, target, cfg, output, index)
        summaries.append(summary)
        metadata.append(meta)
        pd.DataFrame(summaries).to_csv(output / "unseen_summary_partial.csv", index=False)
        pd.DataFrame(metadata).to_csv(output / "training_metadata_partial.csv", index=False)
        del previous, ctx
        previous, previous_date = model, date
        gc.collect()

    pd.DataFrame(summaries).to_csv(output / "unseen_summary.csv", index=False)
    pd.DataFrame(metadata).to_csv(output / "training_metadata.csv", index=False)
    print(f"Frozen test complete: {output}", flush=True)


if __name__ == "__main__":
    main()
