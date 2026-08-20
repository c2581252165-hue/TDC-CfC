"""Paired temporal and spatiotemporal bootstrap used for test comparisons."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


COORD_RE = re.compile(r"^utm50n_500m_x(?P<x>-?\d+)_y(?P<y>-?\d+)$")


def coordinate_blocks(fixed_sites: pd.DataFrame) -> dict[str, str]:
    parsed = fixed_sites["coord_point_id"].astype(str).map(COORD_RE.match)
    if parsed.isna().any():
        raise ValueError("coord_point_id does not follow the fixed-site registry contract")
    x = parsed.map(lambda match: int(match.group("x"))).to_numpy()
    y = parsed.map(lambda match: int(match.group("y"))).to_numpy()
    xb = pd.qcut(x, q=4, labels=False, duplicates="drop").astype(int)
    yb = pd.qcut(y, q=4, labels=False, duplicates="drop").astype(int)
    return dict(
        zip(
            fixed_sites["coord_point_id"].astype(str),
            [f"x{left}_y{right}" for left, right in zip(xb, yb)],
        )
    )


def _moving_month_indices(
    rng: np.random.Generator, block_length: int, n_months: int
) -> np.ndarray:
    output: list[int] = []
    while len(output) < n_months:
        start = int(rng.integers(0, n_months))
        output.extend((start + np.arange(block_length)) % n_months)
    return np.asarray(output[:n_months], dtype=int)


def paired_macro_rmse_bootstrap(
    metadata: pd.DataFrame,
    observed: np.ndarray,
    predictions: dict[str, np.ndarray],
    fixed_sites: pd.DataFrame,
    *,
    reference_model: str,
    repetitions: int = 50_000,
    temporal_block_length: int = 2,
    spatial_resampling: bool = True,
    seed: int = 20260809,
    excluded_month: str | None = "2025-10",
    minimum_month_support: int = 2_879,
) -> pd.DataFrame:
    """Return paired equal-weight month-by-band RMSE distributions.

    Spatial blocks are the prespecified 4x4 coordinate-quantile blocks.  The
    same month and spatial multiplicities are applied to every model and band.
    """
    observed = np.asarray(observed, dtype=np.float64)
    if observed.ndim != 2 or observed.shape[1] != 6:
        raise ValueError("observed must have shape [sample, 6]")
    if len(metadata) != len(observed):
        raise ValueError("metadata and observed arrays have different lengths")
    if reference_model not in predictions:
        raise KeyError(reference_model)
    for name, values in predictions.items():
        if np.asarray(values).shape != observed.shape:
            raise ValueError(f"Prediction shape mismatch for {name}")
    mapping = coordinate_blocks(fixed_sites)
    work = metadata.copy().reset_index(drop=True)
    month_text = work["target_year_month"].astype(str)
    month_counts = month_text.value_counts()
    eligible = month_text.map(month_counts).ge(int(minimum_month_support))
    if excluded_month is not None:
        eligible &= month_text.ne(str(excluded_month))
    if not bool(eligible.any()):
        raise ValueError("No months satisfy the formal support contract")
    selected = np.flatnonzero(eligible.to_numpy())
    work = work.iloc[selected].reset_index(drop=True)
    observed = observed[selected]
    predictions = {
        name: np.asarray(values)[selected]
        for name, values in predictions.items()
    }
    work["spatial_block"] = work["coord_point_id"].astype(str).map(mapping)
    if work["spatial_block"].isna().any():
        raise ValueError("Prediction metadata contains a site absent from the registry")
    months = sorted(work["target_year_month"].astype(str).unique())
    block_names = sorted(work["spatial_block"].unique())
    month_index = work["target_year_month"].astype(str).map({m: i for i, m in enumerate(months)}).to_numpy()
    spatial_index = work["spatial_block"].map({b: i for i, b in enumerate(block_names)}).to_numpy()
    n_months, n_blocks = len(months), len(block_names)
    counts = np.zeros((n_months, 6, n_blocks), dtype=np.float64)
    sse = {
        name: np.zeros((n_months, 6, n_blocks), dtype=np.float64)
        for name in predictions
    }
    for month in range(n_months):
        rows = month_index == month
        for band in range(6):
            np.add.at(counts[month, band], spatial_index[rows], 1.0)
            for name, values in predictions.items():
                error = np.asarray(values, dtype=np.float64)[rows, band] - observed[rows, band]
                np.add.at(sse[name][month, band], spatial_index[rows], error * error)
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(repetitions, dtype=np.float64) for name in predictions}
    for replicate in range(repetitions):
        sampled_months = _moving_month_indices(rng, temporal_block_length, n_months)
        cell_values = {name: [] for name in predictions}
        for month in sampled_months:
            multiplicity = (
                rng.multinomial(n_blocks, np.full(n_blocks, 1.0 / n_blocks))
                if spatial_resampling
                else np.ones(n_blocks, dtype=int)
            )
            denominator = counts[month] @ multiplicity
            for name in predictions:
                rmse = np.sqrt(
                    np.divide(
                        sse[name][month] @ multiplicity,
                        denominator,
                        out=np.full(6, np.nan),
                        where=denominator > 0,
                    )
                )
                cell_values[name].append(rmse)
        for name in predictions:
            draws[name][replicate] = float(np.nanmean(np.vstack(cell_values[name])))
    reference = draws[reference_model]
    rows = []
    for name, values in draws.items():
        gain = values - reference
        rows.append(
            {
                "model": name,
                "reference_model": reference_model,
                "repetitions": repetitions,
                "temporal_block_length": temporal_block_length,
                "spatial_resampling": spatial_resampling,
                "excluded_month": excluded_month,
                "minimum_month_support": minimum_month_support,
                "eligible_months": n_months,
                "rmse_median": float(np.median(values)),
                "rmse_ci_low": float(np.quantile(values, 0.025)),
                "rmse_ci_high": float(np.quantile(values, 0.975)),
                "gain_median": float(np.median(gain)),
                "gain_ci_low": float(np.quantile(gain, 0.025)),
                "gain_ci_high": float(np.quantile(gain, 0.975)),
                "positive_gain_probability": float(np.mean(gain > 0)),
            }
        )
    return pd.DataFrame(rows)


__all__ = ["coordinate_blocks", "paired_macro_rmse_bootstrap"]
