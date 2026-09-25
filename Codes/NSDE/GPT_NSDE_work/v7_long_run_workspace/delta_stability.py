from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from audit_v7 import CACHE, OUT, load_model, load_notebook_environment


def main():
    ns = load_notebook_environment()
    ns["stage_log"] = lambda message: None
    torch.set_num_threads(1)
    contracts = pd.read_csv(CACHE / "v7_final_next_day_contract_results.csv")
    contracts = contracts.loc[contracts["otm_on_both_dates"].astype(bool)].copy()
    rows = []
    contract_rows = []
    for date, target in contracts.groupby("date"):
        date = pd.Timestamp(date)
        model, _ = load_model(ns, date)
        full = ns["quotes_all"].loc[ns["quotes_all"]["date"] == date].copy().reset_index(drop=True)
        ids = set(target["optionid"].astype(int))
        current = full.loc[full["optionid"].astype(int).isin(ids)].copy().reset_index(drop=True)
        for seed_index in range(2):
            samples_max = ns["simulate_same_day_x"](
                model, current["T"].unique(), 200000, 2000,
                ns["SEED"] + 1_600_000 + date.day * 100 + seed_index,
                progress=False,
            )
            for paths in (50000, 100000, 200000):
                pairs = paths // 2
                samples = {key: value[:pairs].copy() for key, value in samples_max.items()}
                delta = ns["delta_rows_from_terminal"](
                    current, samples, bump_bps=(1, 2, 5, 10, 20)
                )
                merged = target.merge(
                    delta[["optionid", "delta_1bp", "delta_2bp", "delta_5bp", "delta_10bp", "delta_20bp"]]
                    .drop_duplicates("optionid"),
                    on="optionid", how="left", validate="many_to_one",
                )
                for bps in (1, 2, 5, 10, 20):
                    error = merged["dP_market"].to_numpy(float) - merged[f"delta_{bps}bp"].to_numpy(float) * merged["dS"].to_numpy(float)
                    rows.append({
                        "date": date, "seed_index": seed_index, "paths": paths,
                        "bump_bps": bps, "hedge_rmse": float(np.sqrt(np.mean(error * error))),
                        "hedge_mae": float(np.mean(np.abs(error))),
                    })
                if paths == 200000:
                    keep = merged[["date", "optionid", "dP_market", "dS", "delta_market", "delta_1bp", "delta_2bp", "delta_5bp", "delta_10bp", "delta_20bp"]].copy()
                    keep["seed_index"] = seed_index
                    contract_rows.append(keep)
                del samples, delta, merged
            del samples_max
            gc.collect()
        del model, full, current
        gc.collect()
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "delta_bump_path_stability.csv", index=False)
    all_contracts = pd.concat(contract_rows, ignore_index=True)
    all_contracts.to_csv(OUT / "delta_bump_contracts_200k.csv", index=False)
    pooled = detail.groupby(["paths", "bump_bps"]).agg(
        mean_daily_rmse=("hedge_rmse", "mean"),
        sd_daily_rmse=("hedge_rmse", "std"),
    ).reset_index()
    pooled.to_csv(OUT / "delta_bump_path_summary.csv", index=False)
    print(pooled.to_string(index=False))


if __name__ == "__main__":
    main()
