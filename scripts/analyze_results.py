from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BANDS = ("B2", "B3", "B4", "B8", "B11", "B12")


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=np.abs(denominator) > 1e-8,
    )


def indices(values: np.ndarray) -> dict[str, np.ndarray]:
    b2, b3, b4, b8, b11, b12 = np.asarray(values, dtype=np.float64).T
    return {
        "NDVI": safe_ratio(b8 - b4, b8 + b4),
        "NDWI": safe_ratio(b3 - b8, b3 + b8),
        "NDMI": safe_ratio(b8 - b11, b8 + b11),
        "MNDWI": safe_ratio(b3 - b11, b3 + b11),
        "NBR": safe_ratio(b8 - b12, b8 + b12),
    }


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted = observed[valid], predicted[valid]
    error = predicted - observed
    denominator = np.sum((observed - observed.mean()) ** 2)
    return {
        "n": int(len(observed)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "r2": float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else np.nan,
        "correlation": float(np.corrcoef(observed, predicted)[0, 1])
        if np.std(observed) > 0 and np.std(predicted) > 0
        else np.nan,
    }


def attribution_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    pointmonth: np.ndarray,
    strongest_simple: np.ndarray,
) -> dict[str, float]:
    valid = (
        np.isfinite(observed)
        & np.isfinite(predicted)
        & np.isfinite(pointmonth)
        & np.isfinite(strongest_simple)
    )
    observed = observed[valid]
    predicted = predicted[valid]
    pointmonth = pointmonth[valid]
    strongest_simple = strongest_simple[valid]
    result = metrics(observed, predicted)
    pointmonth_rmse = np.sqrt(np.mean((pointmonth - observed) ** 2))
    strongest_rmse = np.sqrt(np.mean((strongest_simple - observed) ** 2))
    observed_anomaly = observed - pointmonth
    predicted_anomaly = predicted - pointmonth
    anomaly_error = predicted_anomaly - observed_anomaly
    denominator = np.sum((observed_anomaly - observed_anomaly.mean()) ** 2)
    nonzero = observed_anomaly != 0
    result.update(
        skill_vs_pointmonth=float(1.0 - result["rmse"] / pointmonth_rmse),
        skill_vs_strongest_simple=float(1.0 - result["rmse"] / strongest_rmse),
        anomaly_r2=float(1.0 - np.sum(anomaly_error**2) / denominator)
        if denominator > 0
        else np.nan,
        anomaly_correlation=float(np.corrcoef(observed_anomaly, predicted_anomaly)[0, 1])
        if np.std(observed_anomaly) > 0 and np.std(predicted_anomaly) > 0
        else np.nan,
        anomaly_sign_accuracy=float(
            np.mean(np.sign(observed_anomaly[nonzero]) == np.sign(predicted_anomaly[nonzero]))
        )
        if nonzero.any()
        else np.nan,
    )
    return result


def parse_prediction(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("prediction must be MODEL=NPZ")
    name, path = value.split("=", 1)
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derived indices, month-band gains, and condition-stratified metrics"
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--prediction", action="append", type=parse_prediction, required=True)
    parser.add_argument("--reference-model")
    parser.add_argument("--stratum", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata = pd.read_parquet(args.metadata)
    arrays = {}
    observed = None
    pointmonth = None
    strongest_simple = None
    for name, path in args.prediction:
        payload = np.load(path, allow_pickle=False)
        current_observed = np.asarray(payload["observed"], dtype=np.float64)
        if observed is None:
            observed = current_observed
        elif not np.array_equal(observed, current_observed):
            raise RuntimeError(f"Observed arrays differ for {name}")
        current_pointmonth = np.asarray(payload["pointmonth"], dtype=np.float64)
        current_strongest = np.asarray(payload["strongest_simple"], dtype=np.float64)
        if pointmonth is None:
            pointmonth, strongest_simple = current_pointmonth, current_strongest
        elif not np.array_equal(pointmonth, current_pointmonth) or not np.array_equal(
            strongest_simple, current_strongest
        ):
            raise RuntimeError(f"Reference arrays differ for {name}")
        arrays[name] = np.asarray(payload["predicted"], dtype=np.float64)
    assert observed is not None
    assert pointmonth is not None and strongest_simple is not None
    if len(metadata) != len(observed):
        raise ValueError("metadata and prediction arrays have different lengths")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    index_rows = []
    observed_indices = indices(observed)
    for model, prediction in arrays.items():
        for index_name, predicted_index in indices(prediction).items():
            index_rows.append(
                {"model": model, "index": index_name, **metrics(observed_indices[index_name], predicted_index)}
            )
    pd.DataFrame(index_rows).to_csv(args.output_dir / "derived_index_metrics.csv", index=False)
    cell_rows = []
    for model, prediction in arrays.items():
        for month, row_indices in metadata.groupby("target_year_month", sort=True).indices.items():
            selected = np.asarray(row_indices, dtype=int)
            for band_index, band in enumerate(BANDS):
                cell_rows.append(
                    {
                        "model": model,
                        "target_year_month": month,
                        "band": band,
                        **metrics(observed[selected, band_index], prediction[selected, band_index]),
                    }
                )
    cell_frame = pd.DataFrame(cell_rows)
    if args.reference_model:
        reference = cell_frame.loc[
            cell_frame["model"].eq(args.reference_model),
            ["target_year_month", "band", "rmse"],
        ].rename(columns={"rmse": "reference_rmse"})
        cell_frame = cell_frame.merge(reference, on=["target_year_month", "band"], validate="many_to_one")
        cell_frame["rmse_gain_vs_reference"] = cell_frame["reference_rmse"] - cell_frame["rmse"]
    cell_frame.to_csv(args.output_dir / "month_band_metrics.csv", index=False)
    strata_rows = []
    for column in args.stratum:
        if column not in metadata.columns:
            raise KeyError(f"Stratum column absent from metadata: {column}")
        for level, row_indices in metadata.groupby(column, dropna=False, sort=True).indices.items():
            selected = np.asarray(row_indices, dtype=int)
            for model, prediction in arrays.items():
                strata_rows.append(
                    {
                        "stratum": column,
                        "level": level,
                        "model": model,
                        **attribution_metrics(
                            observed[selected].ravel(),
                            prediction[selected].ravel(),
                            pointmonth[selected].ravel(),
                            strongest_simple[selected].ravel(),
                        ),
                    }
                )
    pd.DataFrame(strata_rows).to_csv(args.output_dir / "stratified_metrics.csv", index=False)


if __name__ == "__main__":
    main()
