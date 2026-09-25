from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = ROOT / "cache_original" / "neural_lv_v7_final_safe_cache"
NOTEBOOK = ROOT / "05_Neural_LV_v7_final_revision_crash_safe.ipynb"
OUT = ROOT / "audit_results"
OUT.mkdir(exist_ok=True)


def load_notebook_environment():
    """Execute definition/data cells only; do not train or mutate uploaded checkpoints."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    module = types.ModuleType("nlv_audit")
    namespace = module.__dict__
    sys.modules[module.__name__] = module
    original = Path.cwd()
    os.chdir(DATA)
    try:
        for index in (2, 4, 6, 8, 10, 12, 14):
            source = "".join(notebook["cells"][index]["source"])
            # The diagnostic build opens append-only logs in its local cache. Remove that
            # block for a read-only audit; everything after stage_log's initial call is unused.
            if index == 2 and "# Persistent diagnostics:" in source:
                source = source.split("# Persistent diagnostics:", 1)[0]
            exec(compile(source, f"notebook_cell_{index}", "exec"), namespace)
    finally:
        os.chdir(original)
    return namespace


def load_model(ns, date):
    date = pd.Timestamp(date)
    checkpoint = CACHE / f"forecast_full_{date:%Y%m%d}_resume.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ns["MaturityHeadNeuralLocalVol"](payload["boundaries"]).cpu().eval()
    model.load_state_dict(payload["final_model_state"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def residual_audit():
    frame = pd.read_csv(CACHE / "v7_final_next_day_contract_results.csv")
    frame["date"] = pd.to_datetime(frame["date"])
    frame["next_date"] = pd.to_datetime(frame["next_date"])
    frame = frame.loc[frame["otm_on_both_dates"].astype(bool)].copy()
    frame["sq_error"] = frame["next_day_pricing_error"] ** 2
    frame["abs_error"] = frame["next_day_pricing_error"].abs()
    frame["abs_k"] = frame["k_next"].abs()
    frame["days_next"] = frame["T_next"] * 365.0
    frame["maturity_bucket"] = pd.cut(
        frame["days_next"], [0, 30, 60, 120, 240, 400], right=True,
        labels=["0-30", "31-60", "61-120", "121-240", "241-400"],
    )
    frame["moneyness_bucket"] = pd.cut(
        frame["k_next"], [-np.inf, -0.10, -0.05, 0, 0.05, 0.10, np.inf],
        labels=["<-10%", "-10:-5%", "-5:0%", "0:5%", "5:10%", ">10%"],
    )

    total_squared = frame["sq_error"].sum()
    ordered = frame.sort_values("sq_error", ascending=False)
    concentration = []
    for fraction in (0.001, 0.005, 0.01, 0.05, 0.10):
        count = max(1, int(math.ceil(len(frame) * fraction)))
        concentration.append({
            "top_fraction": fraction,
            "contracts": count,
            "squared_error_share": ordered.head(count)["sq_error"].sum() / total_squared,
        })

    groups = []
    for dimension in ("date", "maturity_bucket", "moneyness_bucket", "cp_flag"):
        for key, group in frame.groupby(dimension, observed=True):
            groups.append({
                "dimension": dimension,
                "group": str(key),
                "n": len(group),
                "bias": group["next_day_pricing_error"].mean(),
                "mae": group["abs_error"].mean(),
                "rmse": np.sqrt(group["sq_error"].mean()),
                "squared_error_share": group["sq_error"].sum() / total_squared,
            })

    # Huber influence diagnostics only; no claim that a threshold is optimal.
    huber = []
    error = frame["next_day_pricing_error"].to_numpy(float)
    for delta in (0.5, 1.0, 2.0, 3.0, 5.0):
        absolute = np.abs(error)
        loss = np.where(absolute <= delta, 0.5 * error**2, delta * (absolute - 0.5 * delta))
        huber.append({
            "delta": delta,
            "mean_huber": loss.mean(),
            "fraction_linear_region": np.mean(absolute > delta),
            "gradient_mass_top_1pct": np.sort(np.minimum(absolute, delta))[-max(1, len(error)//100):].sum()
            / np.minimum(absolute, delta).sum(),
        })

    pd.DataFrame(concentration).to_csv(OUT / "residual_concentration.csv", index=False)
    pd.DataFrame(groups).to_csv(OUT / "residual_groups.csv", index=False)
    pd.DataFrame(huber).to_csv(OUT / "huber_influence.csv", index=False)
    ordered.head(250).to_csv(OUT / "largest_next_day_errors.csv", index=False)


@torch.no_grad()
def repeated_checkpoint_scores(ns, repeats=8, paths=16384):
    rows = []
    dates = pd.to_datetime(["2024-03-01", "2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07"])
    for date in dates:
        model, payload = load_model(ns, date)
        selected, _, _ = ns["daily_split"](date)
        context = ns["build_calibration_context"](date, selected)
        for repeat in range(repeats):
            normals = ns["antithetic_normals"](
                paths, len(context.grid_np) - 1, 910_000 + repeat
            )
            diagnostic = ns["objective_diagnostics"](model, context, normals)
            rows.append({
                "date": date,
                "repeat": repeat,
                "paths": paths,
                "raw_rmse": diagnostic["raw_rmse"],
                "raw_mae": diagnostic["raw_mae"],
                "selected_update": min(
                    payload.get("rerank_records", []),
                    key=lambda row: row["rerank_raw_rmse"],
                )["sde_update"],
            })
            del normals, diagnostic
        del model, payload, context
        gc.collect()
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "checkpoint_mc_repeats.csv", index=False)
    summary = result.groupby("date").agg(
        rmse_mean=("raw_rmse", "mean"),
        rmse_sd=("raw_rmse", "std"),
        rmse_min=("raw_rmse", "min"),
        rmse_max=("raw_rmse", "max"),
        mae_mean=("raw_mae", "mean"),
    )
    summary.to_csv(OUT / "checkpoint_mc_repeat_summary.csv")


@torch.no_grad()
def simulate_fixed_coordinate_bumps(ns, model, maturities, total_paths, chunk_paths, seed, epsilon):
    """Hold sigma(t,x) fixed in the original normalised state coordinate.

    The original forward curve remains the numeraire scale and x_0 is 1 +/- epsilon.
    This contrasts with v7's sticky-normalised-moneyness bump, where x_0 remains one
    and the forward/payoff scale is bumped.
    """
    grid = ns["build_time_grid"](maturities)
    maturity_index = {
        ns["tkey"](maturity): int(ns["nearest_grid_indices"](grid, [maturity])[0])
        for maturity in np.unique(np.asarray(maturities, float))
    }
    by_index = {}
    for key, index in maturity_index.items():
        by_index.setdefault(index, []).append(key)
    total_pairs = total_paths // 2
    chunk_pairs = max(1, chunk_paths // 2)
    output = {
        sign: {key: np.empty((total_pairs, 2), np.float32) for key in maturity_index}
        for sign in (-1, 1)
    }
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    done = 0
    while done < total_pairs:
        count = min(chunk_pairs, total_pairs - done)
        half = torch.randn(count, len(grid) - 1, generator=generator, dtype=torch.float32)
        shocks = torch.cat([half, -half], dim=0)
        states = {
            -1: torch.full((2 * count, 1), 1.0 - epsilon, dtype=torch.float32),
            1: torch.full((2 * count, 1), 1.0 + epsilon, dtype=torch.float32),
        }
        for step in range(len(grid) - 1):
            time = float(grid[step])
            dt = float(grid[step + 1] - grid[step])
            dw = math.sqrt(dt) * shocks[:, step:step + 1]
            for sign in (-1, 1):
                state = states[sign]
                sigma = model(torch.full_like(state, time), state)
                coefficient = state * sigma
                states[sign] = state + coefficient * dw / (
                    1.0 + torch.abs(coefficient) * math.sqrt(dt)
                )
            if step + 1 in by_index:
                lower, upper = done, done + count
                for sign in (-1, 1):
                    array = states[sign].squeeze(1).numpy()
                    for key in by_index[step + 1]:
                        output[sign][key][lower:upper, 0] = array[:count]
                        output[sign][key][lower:upper, 1] = array[count:]
        done += count
        del half, shocks, states
    return output


def prices_from_original_scale(ns, rows, samples):
    rows = rows.reset_index(drop=True)
    prices = np.empty(len(rows), float)
    for maturity, indices in rows.groupby("T").groups.items():
        positions = np.asarray(list(indices), int)
        group = rows.loc[indices]
        payoff = ns["empirical_payoff_mean"](
            samples[ns["tkey"](maturity)].reshape(-1),
            group["K"].to_numpy(float) / group["F"].to_numpy(float),
            group["cp_flag"].to_numpy(),
        )
        prices[positions] = group["D"].to_numpy(float) * group["F"].to_numpy(float) * payoff
    return prices


@torch.no_grad()
def fixed_coordinate_delta_audit(ns, paths=50000, epsilon=5e-4):
    contract_results = pd.read_csv(CACHE / "v7_final_next_day_contract_results.csv")
    contract_results["date"] = pd.to_datetime(contract_results["date"])
    contract_results["next_date"] = pd.to_datetime(contract_results["next_date"])
    rows_out = []
    summary = []
    for date, group in contract_results.groupby("date"):
        date = pd.Timestamp(date)
        model, _ = load_model(ns, date)
        current = ns["quotes_all"].loc[ns["quotes_all"]["date"] == date].copy()
        ids = set(group["optionid"].astype(int))
        current = current.loc[current["optionid"].astype(int).isin(ids)].copy().reset_index(drop=True)
        bumps = simulate_fixed_coordinate_bumps(
            ns, model, current["T"].unique(), paths, 2000,
            930_000 + date.day, epsilon,
        )
        price_down = prices_from_original_scale(ns, current, bumps[-1])
        price_up = prices_from_original_scale(ns, current, bumps[1])
        current["fixed_coordinate_delta"] = (
            (price_up - price_down) / (2.0 * epsilon * current["S"].to_numpy(float))
        )
        merged = group.merge(
            current[["optionid", "fixed_coordinate_delta"]].drop_duplicates("optionid"),
            on="optionid", how="left", validate="many_to_one",
        )
        merged["fixed_coordinate_hedge_error"] = (
            merged["dP_market"] - merged["fixed_coordinate_delta"] * merged["dS"]
        )
        primary = merged.loc[merged["otm_on_both_dates"].astype(bool)].copy()
        def rmse(column):
            values = primary[column].dropna().to_numpy(float)
            return float(np.sqrt(np.mean(values**2)))
        summary.append({
            "date": date,
            "next_date": primary["next_date"].iloc[0],
            "n": len(primary),
            "fixed_coordinate_hedge_rmse": rmse("fixed_coordinate_hedge_error"),
            "sticky_normalised_hedge_rmse": rmse("hedge_error"),
            "market_delta_hedge_rmse": rmse("market_delta_hedge_error"),
            "fixed_vs_sticky_delta_rmse": float(np.sqrt(np.mean(
                (primary["fixed_coordinate_delta"] - primary["neural_lv_delta"])**2
            ))),
        })
        rows_out.append(merged)
        del model, bumps, price_down, price_up, current, merged, primary
        gc.collect()
    pd.concat(rows_out, ignore_index=True).to_csv(OUT / "delta_convention_contracts.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "delta_convention_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-model-audits", action="store_true")
    args = parser.parse_args()
    residual_audit()
    if not args.skip_model_audits:
        torch.set_num_threads(1)
        namespace = load_notebook_environment()
        repeated_checkpoint_scores(namespace)
        fixed_coordinate_delta_audit(namespace)
    print(f"Audit results written to {OUT}")


if __name__ == "__main__":
    main()
