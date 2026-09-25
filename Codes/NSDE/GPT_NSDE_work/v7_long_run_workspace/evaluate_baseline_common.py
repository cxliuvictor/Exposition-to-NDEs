from __future__ import annotations

import gc
from pathlib import Path

import pandas as pd
import torch

from audit_v7 import load_notebook_environment
from search_pipeline import BASE_CACHE, RUN_ROOT, VARIANTS, evaluate_one_transition, set_grad


OUTPUT = RUN_ROOT / "v7_baseline_common"


def load_forecast_model(ns, date):
    date = pd.Timestamp(date)
    payload = torch.load(
        BASE_CACHE / f"forecast_full_{date:%Y%m%d}_resume.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = ns["MaturityHeadNeuralLocalVol"](payload["boundaries"]).cpu()
    model.load_state_dict(payload["final_model_state"], strict=True)
    model.eval()
    set_grad(model, False)
    return model


def main():
    OUTPUT.mkdir(exist_ok=True)
    ns = load_notebook_environment()
    ns["stage_log"] = lambda message: None
    ns["FINAL_PATH_CHUNK"] = 2000
    torch.set_num_threads(1)
    # Only evaluation fields are used.  The robust-MSE configuration supplies
    # the common 100k path count and the shared evaluation seeds.
    cfg = VARIANTS["robust_mse"]
    dates = pd.to_datetime([
        "2024-03-01", "2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07"
    ])
    next_dates = pd.to_datetime([
        "2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07", "2024-03-08"
    ])
    summaries = []
    for index, (date, next_date) in enumerate(zip(dates, next_dates), start=1):
        print(f"[v7 baseline common] {date.date()} -> {next_date.date()}", flush=True)
        model = load_forecast_model(ns, date)
        summary = evaluate_one_transition(ns, model, date, next_date, cfg, OUTPUT, index)
        summaries.append(summary)
        pd.DataFrame(summaries).to_csv(OUTPUT / "development_summary_partial.csv", index=False)
        del model
        gc.collect()
    pd.DataFrame(summaries).to_csv(OUTPUT / "development_summary.csv", index=False)


if __name__ == "__main__":
    main()
