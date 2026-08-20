from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CELLS = ("H00", "H01", "H02", "H03")


def main() -> None:
    parser = argparse.ArgumentParser(description="Five-seed HSR x RD factorial contrasts")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.metrics)
    required = {"cell", "seed"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    if set(frame["cell"]) != set(CELLS):
        raise ValueError("Factorial input must contain exactly H00, H01, H02, and H03")
    counts = frame.groupby("cell")["seed"].nunique()
    if not (counts == 5).all():
        raise ValueError("Each factorial cell must contain five seeds")
    metric_columns = [
        column
        for column in frame.select_dtypes("number").columns
        if column not in {"seed", "best_epoch", "n_seeds"}
    ]
    effects = []
    for seed in sorted(frame["seed"].unique()):
        seed_frame = frame.loc[frame["seed"].eq(seed)].set_index("cell")
        if set(seed_frame.index) != set(CELLS):
            raise ValueError(f"Incomplete factorial grid for seed {seed}")
        for metric in metric_columns:
            value = {cell: float(seed_frame.loc[cell, metric]) for cell in CELLS}
            hsr_rd0 = value["H01"] - value["H00"]
            hsr_rd1 = value["H03"] - value["H02"]
            rd_hsr0 = value["H02"] - value["H00"]
            rd_hsr1 = value["H03"] - value["H01"]
            effects.append(
                {
                    "seed": seed,
                    "metric": metric,
                    **value,
                    "hsr_simple_at_rd0": hsr_rd0,
                    "hsr_simple_at_rd1": hsr_rd1,
                    "hsr_average_main_effect": (hsr_rd0 + hsr_rd1) / 2.0,
                    "rd_simple_at_hsr0": rd_hsr0,
                    "rd_simple_at_hsr1": rd_hsr1,
                    "rd_average_main_effect": (rd_hsr0 + rd_hsr1) / 2.0,
                    "hsr_by_rd_interaction": value["H03"] - value["H01"] - value["H02"] + value["H00"],
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    effects_frame = pd.DataFrame(effects)
    effects_frame.to_csv(args.output_dir / "factorial_effects_by_seed.csv", index=False)
    effects_frame.groupby("metric", as_index=False).agg(
        hsr_average_main_effect_mean=("hsr_average_main_effect", "mean"),
        hsr_average_main_effect_sd=("hsr_average_main_effect", "std"),
        rd_average_main_effect_mean=("rd_average_main_effect", "mean"),
        rd_average_main_effect_sd=("rd_average_main_effect", "std"),
        interaction_mean=("hsr_by_rd_interaction", "mean"),
        interaction_sd=("hsr_by_rd_interaction", "std"),
    ).to_csv(args.output_dir / "factorial_effects_summary.csv", index=False)


if __name__ == "__main__":
    main()
