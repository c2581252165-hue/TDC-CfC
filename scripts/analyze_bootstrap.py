from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpmf.bootstrap import paired_macro_rmse_bootstrap  # noqa: E402


def parse_prediction(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("prediction must be MODEL=NPZ")
    name, path = value.split("=", 1)
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired dependence-aware test bootstrap")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--fixed-sites", type=Path, required=True)
    parser.add_argument("--prediction", action="append", type=parse_prediction, required=True)
    parser.add_argument("--reference-model", default="TDC-CfC")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=50_000)
    parser.add_argument("--excluded-month", default="2025-10")
    parser.add_argument("--minimum-month-support", type=int, default=2_879)
    args = parser.parse_args()
    metadata = pd.read_parquet(args.metadata)
    fixed_sites = pd.read_csv(args.fixed_sites)
    observed = None
    predictions = {}
    for name, path in args.prediction:
        arrays = np.load(path, allow_pickle=False)
        candidate_observed = np.asarray(arrays["observed"], dtype=np.float64)
        if observed is None:
            observed = candidate_observed
        elif not np.array_equal(observed, candidate_observed):
            raise RuntimeError(f"Observed arrays differ for {name}")
        predictions[name] = np.asarray(arrays["predicted"], dtype=np.float64)
    assert observed is not None
    rows = []
    for spatial in (False, True):
        for block_length in (2, 3):
            rows.append(
                paired_macro_rmse_bootstrap(
                    metadata,
                    observed,
                    predictions,
                    fixed_sites,
                    reference_model=args.reference_model,
                    repetitions=args.repetitions,
                    temporal_block_length=block_length,
                    spatial_resampling=spatial,
                    seed=20260809 + block_length + (100 if spatial else 0),
                    excluded_month=args.excluded_month,
                    minimum_month_support=args.minimum_month_support,
                )
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
