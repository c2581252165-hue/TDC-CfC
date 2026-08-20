from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_POINT_COUNT = 9_596


def parse_prediction(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("prediction must be MODEL=NPZ")
    name, path = value.split("=", 1)
    return name, Path(path)


def macro_rmse(metadata: pd.DataFrame, observed: np.ndarray, predicted: np.ndarray, threshold: int) -> float:
    cells = []
    for _, row_indices in metadata.groupby("target_year_month", sort=True).indices.items():
        selected = np.asarray(row_indices, dtype=int)
        if len(selected) < threshold:
            continue
        cells.extend(
            np.sqrt(np.mean((predicted[selected] - observed[selected]) ** 2, axis=0)).tolist()
        )
    return float(np.mean(cells)) if cells else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-one-month-out and support-threshold checks")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--prediction", action="append", type=parse_prediction, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata = pd.read_parquet(args.metadata)
    observed = None
    predictions = {}
    for name, path in args.prediction:
        payload = np.load(path, allow_pickle=False)
        candidate = np.asarray(payload["observed"], dtype=np.float64)
        if observed is None:
            observed = candidate
        elif not np.array_equal(observed, candidate):
            raise RuntimeError(f"Observed arrays differ for {name}")
        predictions[name] = np.asarray(payload["predicted"], dtype=np.float64)
    assert observed is not None
    if len(metadata) != len(observed):
        raise ValueError("metadata and predictions have different sample counts")
    primary_threshold = math.ceil(EXPECTED_POINT_COUNT * 0.30)
    month_text = metadata["target_year_month"].astype(str)
    month_counts = month_text.value_counts()
    eligible_months = sorted(month_counts[month_counts.ge(primary_threshold)].index)
    eligible = month_text.isin(eligible_months).to_numpy()
    eligible_metadata = metadata.loc[eligible].reset_index(drop=True)
    eligible_observed = observed[eligible]
    eligible_predictions = {
        model: prediction[eligible] for model, prediction in predictions.items()
    }
    lomo_rows = []
    for omitted in eligible_months:
        keep = eligible_metadata["target_year_month"].astype(str).ne(omitted).to_numpy()
        for model, prediction in eligible_predictions.items():
            lomo_rows.append(
                {
                    "omitted_month": omitted,
                    "model": model,
                    "macro_rmse": macro_rmse(
                        eligible_metadata.loc[keep].reset_index(drop=True),
                        eligible_observed[keep],
                        prediction[keep],
                        primary_threshold,
                    ),
                }
            )
    threshold_rows = []
    for fraction in (0.20, 0.30, 0.40, 0.50):
        threshold = math.ceil(EXPECTED_POINT_COUNT * fraction)
        for model, prediction in predictions.items():
            threshold_rows.append(
                {
                    "support_fraction": fraction,
                    "minimum_month_n": threshold,
                    "model": model,
                    "macro_rmse": macro_rmse(metadata, observed, prediction, threshold),
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(lomo_rows).to_csv(args.output_dir / "leave_one_month_out.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(args.output_dir / "support_thresholds.csv", index=False)


if __name__ == "__main__":
    main()
