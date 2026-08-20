from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .constants import TARGET_FROM_HISTORY

HGB_ESTIMATOR_CONFIG = {
    "learning_rate": 0.08,
    "max_iter": 250,
    "max_leaf_nodes": 31,
    "l2_regularization": 1.0,
    "early_stopping": False,
}


def _month_of_year(month_ordinal: np.ndarray) -> np.ndarray:
    return np.mod(np.asarray(month_ordinal, dtype=np.int64), 12)


def _fallback_chain(primary: np.ndarray, point_mean: np.ndarray, global_mean: np.ndarray) -> np.ndarray:
    output = np.asarray(primary, dtype=np.float32).copy()
    missing = ~np.isfinite(output)
    output[missing] = point_mean[missing]
    missing = ~np.isfinite(output)
    if np.any(missing):
        output[missing] = np.broadcast_to(global_mean, output.shape)[missing]
    return output


@dataclass
class SimpleBaselineSuite:
    global_mean: np.ndarray
    point_mean: np.ndarray
    point_month: np.ndarray
    ar_phi: np.ndarray

    @classmethod
    def fit(cls, store, train_indices: np.ndarray) -> "SimpleBaselineSuite":
        indices = np.asarray(train_indices, dtype=np.int64)
        y = np.asarray(store.y[indices], dtype=np.float64)
        point = np.asarray(store.point_indices[indices], dtype=np.int64)
        target_ordinals = np.asarray(store.month_ordinals[store.target_month_indices[indices]], dtype=np.int64)
        moy = _month_of_year(target_ordinals)
        point_count = int(store.manifest["point_count"])
        global_mean = np.nanmean(y, axis=0)
        point_sum = np.zeros((point_count, 6), dtype=np.float64)
        point_n = np.zeros((point_count, 6), dtype=np.float64)
        pm_sum = np.zeros((point_count, 12, 6), dtype=np.float64)
        pm_n = np.zeros((point_count, 12, 6), dtype=np.float64)
        for band in range(6):
            np.add.at(point_sum[:, band], point, y[:, band])
            np.add.at(point_n[:, band], point, np.isfinite(y[:, band]))
            np.add.at(pm_sum[:, :, band], (point, moy), y[:, band])
            np.add.at(pm_n[:, :, band], (point, moy), np.isfinite(y[:, band]))
        point_mean = np.divide(point_sum, point_n, out=np.full_like(point_sum, np.nan), where=point_n > 0)
        point_mean = _fallback_chain(point_mean, np.broadcast_to(global_mean, point_mean.shape), global_mean)
        point_month = np.divide(pm_sum, pm_n, out=np.full_like(pm_sum, np.nan), where=pm_n > 0)
        point_month = _fallback_chain(point_month, np.broadcast_to(point_mean[:, None, :], point_month.shape), global_mean)
        order = np.lexsort((target_ordinals, point))
        point_o, ordinal_o, y_o = point[order], target_ordinals[order], y[order]
        anomaly = y_o - point_month[point_o, _month_of_year(ordinal_o), :]
        same_next = (point_o[1:] == point_o[:-1]) & (ordinal_o[1:] == ordinal_o[:-1] + 1)
        previous, current = anomaly[:-1][same_next], anomaly[1:][same_next]
        numerator = np.nansum(previous * current, axis=0)
        denominator = np.nansum(previous * previous, axis=0)
        ar_phi = np.divide(numerator, denominator, out=np.zeros(6), where=denominator > 1e-12)
        return cls(global_mean.astype(np.float32), point_mean.astype(np.float32), point_month.astype(np.float32), np.clip(ar_phi, -0.99, 0.99).astype(np.float32))

    def predict_all(self, store, indices: np.ndarray) -> dict[str, np.ndarray]:
        indices = np.asarray(indices, dtype=np.int64)
        point = np.asarray(store.point_indices[indices], dtype=np.int64)
        target_ordinals = np.asarray(store.month_ordinals[store.target_month_indices[indices]], dtype=np.int64)
        moy = _month_of_year(target_ordinals)
        x = np.asarray(store.x_values[indices], dtype=np.float32)
        mask = np.asarray(store.x_mask[indices], dtype=bool)
        point_mean = self.point_mean[point]
        point_month = self.point_month[point, moy]
        last_valid = np.full((len(indices), 6), np.nan, dtype=np.float32)
        unresolved = np.ones(len(indices), dtype=bool)
        for lag in range(x.shape[1] - 1, -1, -1):
            take = unresolved & mask[:, lag]
            last_valid[take] = x[take, lag][:, TARGET_FROM_HISTORY]
            unresolved[take] = False
        last_valid = _fallback_chain(last_valid, point_mean, self.global_mean)
        lag12 = x[:, 0, TARGET_FROM_HISTORY]
        lag12 = np.where(mask[:, 0, None], lag12, point_month)
        lag12 = _fallback_chain(lag12, point_mean, self.global_mean)
        previous_climatology = self.point_month[point, _month_of_year(target_ordinals - 1)]
        ar = point_month + (last_valid - previous_climatology) * self.ar_phi[None, :]
        return {
            "B01_POINT_MEAN": point_mean.astype(np.float32),
            "B02_POINT_MONTH": point_month.astype(np.float32),
            "B03_LAST_VALID": last_valid.astype(np.float32),
            "B04_LAG12": lag12.astype(np.float32),
            "B05_POINT_MONTH_AR1": ar.astype(np.float32),
        }


def flattened_features(
    store,
    indices: np.ndarray,
    preprocessor,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    x = preprocessor.transform_x(np.asarray(store.x_values[indices], dtype=np.float32))
    mask = np.asarray(store.x_mask[indices], dtype=np.float32)
    age = np.asarray(store.x_age[indices], dtype=np.float32) / 12.0
    ordinals = np.asarray(store.month_ordinals[store.history_month_indices[indices]], dtype=np.int64)
    angle = 2.0 * np.pi * (_month_of_year(ordinals).astype(np.float32) / 12.0)
    calendar = np.stack([np.sin(angle), np.cos(angle)], axis=-1)
    base = np.concatenate([x, mask[..., None], age[..., None], calendar], axis=-1)
    # The paper HGB used the common 53-feature monthly token contract.  Its
    # 39 non-Sentinel-2 auxiliary positions were fixed to zero in the S2-only
    # experiment, but remain present so released/retrained assets have the
    # exact feature shape used in the reported run.
    zeros = np.zeros((*base.shape[:2], 39), dtype=np.float32)
    tokens = np.concatenate([base, zeros], axis=-1)
    if tokens.shape[1:] != (12, 53):
        raise RuntimeError(f"Unexpected HGB token contract: {tokens.shape}")
    return tokens.reshape(len(indices), -1).astype(np.float32)

class SklearnBaseline:
    def __init__(self, kind: str, random_state: int):
        self.kind, self.random_state, self.model = kind, int(random_state), None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "SklearnBaseline":
        if self.kind == "ridge":
            from sklearn.linear_model import Ridge
            self.model = Ridge(alpha=1.0)
        elif self.kind == "hgb":
            from sklearn.ensemble import HistGradientBoostingRegressor
            from sklearn.multioutput import MultiOutputRegressor
            base = HistGradientBoostingRegressor(**HGB_ESTIMATOR_CONFIG, random_state=self.random_state)
            self.model = MultiOutputRegressor(base, n_jobs=1)
        else:
            raise ValueError(f"Unknown sklearn baseline: {self.kind}")
        self.model.fit(features, targets)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Baseline is not fitted")
        return np.asarray(self.model.predict(features), dtype=np.float32)
