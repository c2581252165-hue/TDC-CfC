from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .constants import EXPECTED_POINT_COUNT, MIN_POINT_MONTHS_FOR_R2, TARGET_BANDS


PRIMARY_EVALUABLE_MONTH_FRACTION = 0.30
PRIMARY_EVALUABLE_MONTH_N = math.ceil(EXPECTED_POINT_COUNT * PRIMARY_EVALUABLE_MONTH_FRACTION)
SENSITIVITY_MONTH_FRACTIONS = (0.10, 0.20, 0.30, 0.40)

METRIC_COLUMNS = (
    "rmse",
    "mae",
    "bias",
    "r2",
    "pred_obs_std_ratio",
    "skill_vs_pointmonth",
    "skill_vs_strongest_simple",
    "anomaly_rmse",
    "anomaly_mae",
    "anomaly_bias",
    "anomaly_r2",
    "anomaly_correlation",
    "anomaly_sign_accuracy",
)


def _metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    pointmonth_reference: np.ndarray,
    strongest_reference: np.ndarray,
    *,
    min_r2_n: int = 2,
) -> dict[str, float]:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    pointmonth_reference = np.asarray(pointmonth_reference, dtype=np.float64)
    strongest_reference = np.asarray(strongest_reference, dtype=np.float64)
    valid = np.isfinite(observed) & np.isfinite(predicted) & np.isfinite(pointmonth_reference) & np.isfinite(strongest_reference)
    if valid.sum() == 0:
        return {key: float("nan") for key in METRIC_COLUMNS}
    observed = observed[valid]
    predicted = predicted[valid]
    pointmonth_reference = pointmonth_reference[valid]
    strongest_reference = strongest_reference[valid]
    error = predicted - observed
    rmse = float(np.sqrt(np.mean(error**2)))
    absolute_denominator = float(np.sum((observed - observed.mean()) ** 2))
    observed_std = float(np.std(observed))
    pointmonth_rmse = float(np.sqrt(np.mean((pointmonth_reference - observed) ** 2)))
    strongest_rmse = float(np.sqrt(np.mean((strongest_reference - observed) ** 2)))
    observed_anomaly = observed - pointmonth_reference
    predicted_anomaly = predicted - pointmonth_reference
    anomaly_error = predicted_anomaly - observed_anomaly
    anomaly_denominator = float(np.sum((observed_anomaly - observed_anomaly.mean()) ** 2))
    anomaly_correlation = float("nan")
    if len(observed) >= min_r2_n and np.std(observed_anomaly) > 0 and np.std(predicted_anomaly) > 0:
        anomaly_correlation = float(np.corrcoef(observed_anomaly, predicted_anomaly)[0, 1])
    nonzero_anomaly = observed_anomaly != 0
    sign_accuracy = float(np.mean(np.sign(observed_anomaly[nonzero_anomaly]) == np.sign(predicted_anomaly[nonzero_anomaly]))) if nonzero_anomaly.any() else float("nan")
    return {
        "rmse": rmse,
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "r2": float(1.0 - np.sum(error**2) / absolute_denominator) if len(observed) >= min_r2_n and absolute_denominator > 0 else float("nan"),
        "pred_obs_std_ratio": float(np.std(predicted) / observed_std) if observed_std > 0 else float("nan"),
        "skill_vs_pointmonth": 1.0 - rmse / pointmonth_rmse if pointmonth_rmse > 0 else float("nan"),
        "skill_vs_strongest_simple": 1.0 - rmse / strongest_rmse if strongest_rmse > 0 else float("nan"),
        "anomaly_rmse": float(np.sqrt(np.mean(anomaly_error**2))),
        "anomaly_mae": float(np.mean(np.abs(anomaly_error))),
        "anomaly_bias": float(np.mean(anomaly_error)),
        "anomaly_r2": float(1.0 - np.sum(anomaly_error**2) / anomaly_denominator) if len(observed) >= min_r2_n and anomaly_denominator > 0 else float("nan"),
        "anomaly_correlation": anomaly_correlation,
        "anomaly_sign_accuracy": sign_accuracy,
    }


def _macro_row(grid: pd.DataFrame, base: dict, *, aggregation: str, min_month_n: int, weighted: bool = False) -> dict:
    selected = grid.loc[grid["n"] >= min_month_n]
    row = {
        **base,
        "aggregation": aggregation,
        "minimum_month_n": int(min_month_n),
        "n_months": int(selected["target_year_month"].nunique()),
        "n_bands": 6,
    }
    for column in METRIC_COLUMNS:
        values = selected[column].to_numpy(dtype=np.float64)
        valid = np.isfinite(values)
        if not valid.any():
            row[column] = float("nan")
        elif weighted:
            weights = selected["n"].to_numpy(dtype=np.float64)
            row[column] = float(np.average(values[valid], weights=weights[valid]))
        else:
            row[column] = float(np.mean(values[valid]))
    return row


def _joint_spectral_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    valid = np.isfinite(observed).all(axis=1) & np.isfinite(predicted).all(axis=1)
    observed, predicted = observed[valid], predicted[valid]
    if len(observed) == 0:
        return {
            "n": 0,
            "sam_degrees_mean": float("nan"),
            "sam_degrees_median": float("nan"),
            "sam_degrees_p90": float("nan"),
            "spectral_rmse_mean": float("nan"),
            "prediction_lt0_fraction": float("nan"),
            "prediction_gt1_fraction": float("nan"),
            "prediction_gt1_5_fraction": float("nan"),
        }
    denominator = np.linalg.norm(observed, axis=1) * np.linalg.norm(predicted, axis=1)
    sam_valid = denominator > 0
    angles = np.full(len(observed), np.nan, dtype=np.float64)
    cosine = np.sum(observed[sam_valid] * predicted[sam_valid], axis=1) / denominator[sam_valid]
    angles[sam_valid] = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    spectral_rmse = np.sqrt(np.mean((predicted - observed) ** 2, axis=1))
    return {
        "n": int(len(observed)),
        "sam_degrees_mean": float(np.nanmean(angles)),
        "sam_degrees_median": float(np.nanmedian(angles)),
        "sam_degrees_p90": float(np.nanpercentile(angles, 90)),
        "spectral_rmse_mean": float(np.mean(spectral_rmse)),
        "prediction_lt0_fraction": float(np.mean(predicted < 0)),
        "prediction_gt1_fraction": float(np.mean(predicted > 1)),
        "prediction_gt1_5_fraction": float(np.mean(predicted > 1.5)),
    }


def evaluate_predictions(
    metadata: pd.DataFrame,
    observed: np.ndarray,
    predicted: np.ndarray,
    pointmonth_reference: np.ndarray,
    strongest_simple_reference: np.ndarray | None = None,
    *,
    model_id: str,
    seed: int | None,
    split: str,
    include_by_point: bool = True,
    min_evaluable_month_n: int | None = None,
) -> dict[str, pd.DataFrame]:
    observed = np.asarray(observed, dtype=np.float32)
    predicted = np.asarray(predicted, dtype=np.float32)
    pointmonth_reference = np.asarray(pointmonth_reference, dtype=np.float32)
    strongest = pointmonth_reference if strongest_simple_reference is None else np.asarray(strongest_simple_reference, dtype=np.float32)
    if observed.shape != predicted.shape or observed.shape != pointmonth_reference.shape or observed.shape != strongest.shape or observed.ndim != 2 or observed.shape[1] != 6:
        raise ValueError("Observed, predicted and reference arrays must share [sample, 6]")
    if len(metadata) != len(observed):
        raise ValueError("Metadata and prediction sample counts differ")
    minimum_n = PRIMARY_EVALUABLE_MONTH_N if min_evaluable_month_n is None else int(min_evaluable_month_n)
    if minimum_n < 1:
        raise ValueError("min_evaluable_month_n must be positive")
    base = {"model_id": model_id, "seed": seed, "split": split}
    band_rows: list[dict] = []
    month_rows: list[dict] = []
    grid_rows: list[dict] = []
    point_rows: list[dict] = []
    joint_rows: list[dict] = [{**base, "target_year_month": "ALL", **_joint_spectral_metrics(observed, predicted)}]
    month_groups = metadata.groupby("target_year_month", sort=True).indices
    for band_index, band in enumerate(TARGET_BANDS):
        band_rows.append({**base, "band": band, "n": len(observed), **_metrics(observed[:, band_index], predicted[:, band_index], pointmonth_reference[:, band_index], strongest[:, band_index])})
        for month, group_index in month_groups.items():
            selected = np.asarray(group_index, dtype=np.int64)
            grid_rows.append({
                **base,
                "target_year_month": month,
                "band": band,
                "n": len(selected),
                "month_evaluable": len(selected) >= minimum_n,
                **_metrics(observed[selected, band_index], predicted[selected, band_index], pointmonth_reference[selected, band_index], strongest[selected, band_index]),
            })
    for month, group_index in month_groups.items():
        selected = np.asarray(group_index, dtype=np.int64)
        month_rows.append({
            **base,
            "target_year_month": month,
            "n": len(selected),
            "month_evaluable": len(selected) >= minimum_n,
            **_metrics(observed[selected].ravel(), predicted[selected].ravel(), pointmonth_reference[selected].ravel(), strongest[selected].ravel()),
        })
        joint_rows.append({**base, "target_year_month": month, **_joint_spectral_metrics(observed[selected], predicted[selected])})
    if include_by_point and "coord_point_id" in metadata:
        for point_id, group_index in metadata.groupby("coord_point_id", sort=True).indices.items():
            selected = np.asarray(group_index, dtype=np.int64)
            for band_index, band in enumerate(TARGET_BANDS):
                point_rows.append({**base, "coord_point_id": point_id, "band": band, "n_months": len(selected), **_metrics(observed[selected, band_index], predicted[selected, band_index], pointmonth_reference[selected, band_index], strongest[selected, band_index], min_r2_n=MIN_POINT_MONTHS_FOR_R2)})
    grid = pd.DataFrame(grid_rows)
    macro = pd.DataFrame([_macro_row(grid, base, aggregation="evaluable_month_band_macro", min_month_n=minimum_n)])
    macro_all = pd.DataFrame([_macro_row(grid, base, aggregation="all_month_band_macro", min_month_n=1)])
    macro_weighted = pd.DataFrame([_macro_row(grid, base, aggregation="support_weighted_month_band_sensitivity", min_month_n=1, weighted=True)])
    sensitivity_rows = []
    for fraction in SENSITIVITY_MONTH_FRACTIONS:
        threshold = math.ceil(EXPECTED_POINT_COUNT * fraction)
        row = _macro_row(grid, base, aggregation="coverage_threshold_sensitivity", min_month_n=threshold)
        row["minimum_expected_point_fraction"] = fraction
        sensitivity_rows.append(row)
    pooled = pd.DataFrame([{**base, "n": len(observed), **_metrics(observed.ravel(), predicted.ravel(), pointmonth_reference.ravel(), strongest.ravel())}])
    return {
        "pooled": pooled,
        "by_band": pd.DataFrame(band_rows),
        "by_month": pd.DataFrame(month_rows),
        "month_band": grid,
        "by_point": pd.DataFrame(point_rows),
        "macro": macro,
        "macro_all_months": macro_all,
        "macro_support_weighted": macro_weighted,
        "macro_sensitivity": pd.DataFrame(sensitivity_rows),
        "joint_spectral": pd.DataFrame(joint_rows),
    }
