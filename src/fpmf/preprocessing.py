from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import warnings

import numpy as np

from .utils import write_json


def _nan_mean_std(
    values: np.ndarray, axes: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(values, axis=axes, dtype=np.float64)
        std = np.nanstd(values, axis=axes, dtype=np.float64)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


@dataclass(frozen=True)
class FoldPreprocessor:
    x_mean: list[float]
    x_scale: list[float]
    y_mean: list[float]
    y_scale: list[float]
    fitted_split: str
    training_sample_count: int

    @classmethod
    def fit(
        cls,
        store: "TrainingStoreLike",
        sample_indices: np.ndarray,
        fitted_split: str,
    ) -> "FoldPreprocessor":
        indices = np.asarray(sample_indices, dtype=np.int64)
        if indices.size == 0:
            raise ValueError("Cannot fit a preprocessor on zero samples")
        x_mean, x_scale = _nan_mean_std(
            np.asarray(store.x_values[indices], dtype=np.float32), axes=(0, 1)
        )
        y_mean, y_scale = _nan_mean_std(
            np.asarray(store.y[indices], dtype=np.float32), axes=(0,)
        )
        return cls(
            x_mean=x_mean.tolist(),
            x_scale=x_scale.tolist(),
            y_mean=y_mean.tolist(),
            y_scale=y_scale.tolist(),
            fitted_split=fitted_split,
            training_sample_count=int(indices.size),
        )

    def transform_x(self, values: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.x_mean, dtype=np.float32)
        scale = np.asarray(self.x_scale, dtype=np.float32)
        transformed = (np.asarray(values, dtype=np.float32) - mean) / scale
        return np.nan_to_num(
            transformed, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)

    def transform_y(self, values: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.y_mean, dtype=np.float32)
        scale = np.asarray(self.y_scale, dtype=np.float32)
        return ((np.asarray(values, dtype=np.float32) - mean) / scale).astype(
            np.float32
        )

    def inverse_y(self, values: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.y_mean, dtype=np.float32)
        scale = np.asarray(self.y_scale, dtype=np.float32)
        return (np.asarray(values, dtype=np.float32) * scale + mean).astype(np.float32)

    def save(self, path: str | Path) -> None:
        write_json(path, asdict(self))

    @classmethod
    def load(cls, path: str | Path) -> "FoldPreprocessor":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            x_mean=payload["x_mean"],
            x_scale=payload["x_scale"],
            y_mean=payload["y_mean"],
            y_scale=payload["y_scale"],
            fitted_split=payload["fitted_split"],
            training_sample_count=int(payload["training_sample_count"]),
        )


class TrainingStoreLike:
    x_values: np.ndarray
    y: np.ndarray
