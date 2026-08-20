from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BANDS = ("B2", "B3", "B4", "B8", "B11", "B12")


def dependence_summary(anomalies: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    valid = np.isfinite(anomalies).all(axis=1)
    values = anomalies[valid]
    if len(values) < 2:
        raise ValueError("Too few complete anomaly rows for spectral dependence")
    scale = values.std(axis=0)
    if np.any(scale <= 0):
        raise ValueError("A target band has zero anomaly variance")
    standardized = (values - values.mean(axis=0)) / scale
    correlation = np.corrcoef(standardized, rowvar=False)
    singular_values = np.linalg.svd(standardized, full_matrices=False, compute_uv=False)
    explained = singular_values**2 / np.sum(singular_values**2)
    upper = correlation[np.triu_indices(len(BANDS), k=1)]
    return correlation, explained, {
        "n": int(len(values)),
        "mean_pairwise_correlation": float(upper.mean()),
        "minimum_pairwise_correlation": float(upper.min()),
        "maximum_pairwise_correlation": float(upper.max()),
        "pc1_explained_variance_ratio": float(explained[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bandwise intervention sensitivity and target-anomaly dependence"
    )
    parser.add_argument("--band-metrics", type=Path, required=True)
    parser.add_argument("--observed", type=Path, required=True, help="NPZ with observed and pointmonth")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--stratum", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metrics = pd.read_csv(args.band_metrics)
    natural = metrics.loc[metrics["scenario"].eq("natural"), ["model", "seed", "band", "anomaly_r2"]]
    sensitivity = metrics.merge(
        natural,
        on=["model", "seed", "band"],
        suffixes=("", "_natural"),
        validate="many_to_one",
    )
    sensitivity["delta_anomaly_r2"] = (
        sensitivity["anomaly_r2_natural"] - sensitivity["anomaly_r2"]
    )
    summary = sensitivity.groupby(["model", "scenario"], as_index=False).agg(
        band_sensitivity_mean=("delta_anomaly_r2", "mean"),
        band_sensitivity_sd=("delta_anomaly_r2", "std"),
        band_sensitivity_range=("delta_anomaly_r2", lambda x: float(x.max() - x.min())),
    )
    arrays = np.load(args.observed, allow_pickle=False)
    anomalies = np.asarray(arrays["observed"], dtype=np.float64) - np.asarray(
        arrays["pointmonth"], dtype=np.float64
    )
    correlation, explained, overall = dependence_summary(anomalies)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sensitivity.to_csv(args.output_dir / "bandwise_sensitivity.csv", index=False)
    summary.to_csv(args.output_dir / "bandwise_sensitivity_summary.csv", index=False)
    pd.DataFrame(correlation, index=BANDS, columns=BANDS).to_csv(
        args.output_dir / "target_anomaly_correlation.csv"
    )
    pd.DataFrame(
        {"principal_component": np.arange(1, len(explained) + 1), "explained_variance_ratio": explained}
    ).to_csv(args.output_dir / "target_anomaly_pca.csv", index=False)
    summary_rows = [{"stratum": "overall", "level": "all", **overall}]
    if args.stratum:
        if args.metadata is None:
            raise ValueError("--metadata is required when --stratum is used")
        metadata = pd.read_parquet(args.metadata)
        if len(metadata) != len(anomalies):
            raise ValueError("metadata and observed arrays have different sample counts")
        for column in args.stratum:
            if column not in metadata.columns:
                raise KeyError(f"Stratum column absent from metadata: {column}")
            for level, row_indices in metadata.groupby(column, dropna=False, sort=True).indices.items():
                selected = np.asarray(row_indices, dtype=int)
                _, _, result = dependence_summary(anomalies[selected])
                summary_rows.append({"stratum": column, "level": level, **result})
    pd.DataFrame(summary_rows).to_csv(
        args.output_dir / "target_anomaly_dependence_summary.csv", index=False
    )


if __name__ == "__main__":
    main()
