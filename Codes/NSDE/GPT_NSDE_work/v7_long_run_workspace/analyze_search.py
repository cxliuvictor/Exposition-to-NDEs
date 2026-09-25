from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from search_pipeline import BASE_CACHE, RUN_ROOT, VARIANTS


OUT = RUN_ROOT / "comparison"
OUT.mkdir(exist_ok=True)


def rmse(values):
    values = np.asarray(values, float)
    return float(np.sqrt(np.mean(values * values)))


def load_contracts(variant):
    # March 1 -> 4 is common to every pipeline and is re-evaluated once using
    # the same evaluation seed convention as the later transitions.
    first = pd.read_csv(RUN_ROOT / "v7_baseline_common" / "transition_20240301.csv")
    later = [
        pd.read_csv(RUN_ROOT / variant / f"transition_202403{day:02d}.csv")
        for day in (4, 5, 6, 7)
    ]
    frame = pd.concat([first] + later, ignore_index=True)
    frame["variant"] = variant
    return frame.loc[frame["otm_on_both_dates"].astype(bool)].copy()


def metrics(frame, blend_weight=None):
    price_error = frame["next_day_pricing_error"].to_numpy(float)
    nlv_error = frame["hedge_error"].to_numpy(float)
    market_error = frame["market_delta_hedge_error"].to_numpy(float)
    out = {
        "n": len(frame),
        "pricing_rmse": rmse(price_error),
        "pricing_mae": float(np.mean(np.abs(price_error))),
        "nlv_hedge_rmse": rmse(nlv_error),
        "market_hedge_rmse": rmse(market_error),
    }
    if blend_weight is not None:
        blended_delta = (
            blend_weight * frame["neural_lv_delta"].to_numpy(float)
            + (1.0 - blend_weight) * frame["delta_market"].to_numpy(float)
        )
        blended_error = frame["dP_market"].to_numpy(float) - blended_delta * frame["dS"].to_numpy(float)
        out["blend_weight_nlv"] = blend_weight
        out["blended_hedge_rmse"] = rmse(blended_error)
        out["blended_hedge_mae"] = float(np.mean(np.abs(blended_error)))
    return out


def optimal_blend(frame):
    # delta_blend = delta_market + w * (delta_nlv - delta_market)
    base_error = (
        frame["dP_market"].to_numpy(float)
        - frame["delta_market"].to_numpy(float) * frame["dS"].to_numpy(float)
    )
    exposure = frame["dS"].to_numpy(float) * (
        frame["neural_lv_delta"].to_numpy(float)
        - frame["delta_market"].to_numpy(float)
    )
    weight = float(np.sum(exposure * base_error) / np.sum(exposure * exposure))
    return float(np.clip(weight, 0.0, 1.0))


def main():
    frames = {}
    # Re-evaluated v7 baseline uses the same MC shocks as every new pipeline.
    baseline_parts = [
        pd.read_csv(RUN_ROOT / "v7_baseline_common" / f"transition_202403{day:02d}.csv")
        for day in (1, 4, 5, 6, 7)
    ]
    baseline = pd.concat(baseline_parts, ignore_index=True)
    frames["v7_baseline"] = baseline.loc[baseline["otm_on_both_dates"].astype(bool)].copy()
    for variant in VARIANTS:
        frames[variant] = load_contracts(variant)

    rows = []
    daily = []
    for name, frame in frames.items():
        weight = optimal_blend(frame)
        row = {"pipeline": name, **metrics(frame, weight)}
        rows.append(row)
        for date, group in frame.groupby("date"):
            daily.append({"pipeline": name, "date": date, **metrics(group, weight)})

    comparison = pd.DataFrame(rows).sort_values("pricing_rmse").reset_index(drop=True)
    best_price = float(comparison["pricing_rmse"].min())
    comparison["within_2pct_best_pricing"] = comparison["pricing_rmse"] <= 1.02 * best_price
    eligible = comparison.loc[comparison["within_2pct_best_pricing"]].copy()
    # Frozen selection rule agreed before the search: price first, then the
    # *pure* NLV hedge among pipelines within 2% of the best pricing RMSE.
    # The market/NLV blend is retained only as an explicitly secondary
    # diagnostic and must not influence pipeline selection.
    winner = eligible.sort_values(["nlv_hedge_rmse", "pricing_rmse"]).iloc[0]["pipeline"]
    comparison["selected_by_rule"] = comparison["pipeline"].eq(winner)
    comparison.to_csv(OUT / "development_pipeline_comparison.csv", index=False)
    pd.DataFrame(daily).to_csv(OUT / "development_daily_comparison.csv", index=False)
    (OUT / "frozen_pipeline.txt").write_text(str(winner) + "\n", encoding="utf-8")
    print(comparison.to_string(index=False))
    print("\nFrozen winner:", winner)


if __name__ == "__main__":
    main()
