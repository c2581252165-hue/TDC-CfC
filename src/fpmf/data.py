from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import (
    EXPECTED_MONTH_COUNT,
    EXPECTED_POINT_COUNT,
    HISTORY_BANDS,
    HISTORY_COLUMNS,
    HISTORY_MIN_VALID_ACQUISITIONS,
    LOOKBACK,
    OVERLAP_DIFFERENCE_LIMIT,
    TARGET_COLUMNS,
    TARGET_MIN_VALID_ACQUISITIONS,
)
from .preprocessing import FoldPreprocessor
from .utils import calendar_features_from_ordinals, month_ordinal, sha256_file, stable_json_hash, write_json


PANEL_REQUIRED_COLUMNS = (
    "coord_point_id",
    "year_month",
    *HISTORY_COLUMNS,
    "S2_target_common_valid_acquisition_count",
    "S2_overlap_max_abs_difference",
)


def _validate_month_continuity(months: list[str]) -> None:
    ordinals = [month_ordinal(month) for month in months]
    expected = list(range(ordinals[0], ordinals[0] + len(ordinals)))
    if ordinals != expected:
        raise ValueError(f"Panel months are not continuous: {months}")


def _history_age(valid: np.ndarray) -> np.ndarray:
    if valid.ndim != 2:
        raise ValueError("valid must have shape [point, month]")
    age = np.zeros(valid.shape, dtype=np.float32)
    for month_index in range(valid.shape[1]):
        if month_index == 0:
            age[:, month_index] = np.where(valid[:, month_index], 0.0, 1.0)
        else:
            age[:, month_index] = np.where(valid[:, month_index], 0.0, age[:, month_index - 1] + 1.0)
    return age


def materialize_training_store(
    panel_csv: str | Path,
    split_csv: str | Path,
    output_dir: str | Path,
    *,
    expected_point_count: int = EXPECTED_POINT_COUNT,
    expected_month_count: int = EXPECTED_MONTH_COUNT,
) -> dict:
    """Build an immutable, memory-mapped 12-to-1 training store.

    The function never modifies the source panel and refuses to overwrite an
    existing output. Target-quality values are written only to a sidecar.
    """

    panel_path = Path(panel_csv).resolve()
    split_path = Path(split_csv).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing training store: {destination}")
    staging = destination.with_name(destination.name + ".building")
    if staging.exists():
        raise FileExistsError(f"Stale staging directory exists; inspect it manually: {staging}")
    staging.mkdir(parents=True)

    panel = pd.read_csv(
        panel_path,
        usecols=list(PANEL_REQUIRED_COLUMNS),
        dtype={"coord_point_id": "string", "year_month": "string"},
    )
    missing = sorted(set(PANEL_REQUIRED_COLUMNS) - set(panel.columns))
    if missing:
        raise ValueError(f"Panel is missing required columns: {missing}")
    if panel.duplicated(["coord_point_id", "year_month"]).any():
        raise ValueError("Duplicate coord_point_id + year_month keys detected")

    months = sorted(panel["year_month"].astype(str).unique().tolist())
    _validate_month_continuity(months)
    if expected_month_count and len(months) != expected_month_count:
        raise ValueError(f"Expected {expected_month_count} months, found {len(months)}")
    point_ids = sorted(panel["coord_point_id"].astype(str).unique().tolist())
    if expected_point_count and len(point_ids) != expected_point_count:
        raise ValueError(f"Expected {expected_point_count} points, found {len(point_ids)}")
    expected_rows = len(months) * len(point_ids)
    if len(panel) != expected_rows:
        raise ValueError(f"Panel is not a complete point-month rectangle: {len(panel)} != {expected_rows}")

    panel = panel.sort_values(["year_month", "coord_point_id"], kind="mergesort").reset_index(drop=True)
    month_codes = pd.Categorical(panel["year_month"], categories=months, ordered=True).codes
    point_codes = pd.Categorical(panel["coord_point_id"], categories=point_ids, ordered=True).codes
    expected_month_codes = np.repeat(np.arange(len(months)), len(point_ids))
    expected_point_codes = np.tile(np.arange(len(point_ids)), len(months))
    if not np.array_equal(month_codes, expected_month_codes) or not np.array_equal(point_codes, expected_point_codes):
        raise ValueError("Point set or point order changes between months")

    values_month_point = panel[list(HISTORY_COLUMNS)].to_numpy(np.float32).reshape(len(months), len(point_ids), 10)
    values = np.transpose(values_month_point, (1, 0, 2)).copy()
    valid_count = panel["S2_target_common_valid_acquisition_count"].to_numpy(np.float32).reshape(len(months), len(point_ids)).T
    overlap_source = panel["S2_overlap_max_abs_difference"].to_numpy(np.float64).reshape(len(months), len(point_ids)).T
    overlap_ok = ~np.isfinite(overlap_source) | (overlap_source < OVERLAP_DIFFERENCE_LIMIT)
    overlap = overlap_source.copy()
    finite_all = np.isfinite(values).all(axis=2)
    nonzero_all = (values != 0.0).all(axis=2)
    history_valid = finite_all & nonzero_all & (valid_count >= HISTORY_MIN_VALID_ACQUISITIONS) & overlap_ok

    clean_values = values.copy()
    clean_values[~history_valid, :] = np.nan

    target_indices = np.asarray([HISTORY_BANDS.index(column.removeprefix("S2_")) for column in TARGET_COLUMNS], dtype=np.int64)
    target_values = values[:, :, target_indices]
    target_valid = (
        np.isfinite(target_values).all(axis=2)
        & (target_values != 0.0).all(axis=2)
        & (valid_count >= TARGET_MIN_VALID_ACQUISITIONS)
        & overlap_ok
    )

    split_table = pd.read_csv(split_path, dtype={"target_year_month": "string", "split": "string"})
    split_map = dict(zip(split_table["target_year_month"].astype(str), split_table["split"].astype(str)))
    split_valid_n = dict(zip(split_table["target_year_month"].astype(str), split_table["target_valid_n"].astype(int)))
    unknown_splits = sorted(set(split_map.values()) - {"train", "validation", "test"})
    if unknown_splits:
        raise ValueError(f"Unknown split labels: {unknown_splits}")

    target_month_positions = [index for index, month in enumerate(months) if index >= LOOKBACK and month in split_map]
    if not target_month_positions:
        raise ValueError("No target months have both 12-month history and a split assignment")
    sample_count = int(sum(target_valid[:, index].sum() for index in target_month_positions))
    if sample_count <= 0:
        raise ValueError("No valid supervised samples")

    x_values_mm = np.lib.format.open_memmap(staging / "x_values.npy", mode="w+", dtype=np.float32, shape=(sample_count, LOOKBACK, 10))
    x_mask_mm = np.lib.format.open_memmap(staging / "x_mask.npy", mode="w+", dtype=np.uint8, shape=(sample_count, LOOKBACK))
    x_age_mm = np.lib.format.open_memmap(staging / "x_age.npy", mode="w+", dtype=np.float32, shape=(sample_count, LOOKBACK))
    history_month_mm = np.lib.format.open_memmap(staging / "history_month_indices.npy", mode="w+", dtype=np.int16, shape=(sample_count, LOOKBACK))
    y_mm = np.lib.format.open_memmap(staging / "y.npy", mode="w+", dtype=np.float32, shape=(sample_count, 6))
    point_index_mm = np.lib.format.open_memmap(staging / "point_indices.npy", mode="w+", dtype=np.int32, shape=(sample_count,))
    target_month_mm = np.lib.format.open_memmap(staging / "target_month_indices.npy", mode="w+", dtype=np.int16, shape=(sample_count,))

    metadata_parts: list[pd.DataFrame] = []
    sidecar_parts: list[pd.DataFrame] = []
    cursor = 0
    split_counts = {"train": 0, "validation": 0, "test": 0}
    per_month_counts: dict[str, int] = {}
    for target_month_index in target_month_positions:
        month = months[target_month_index]
        point_index = np.flatnonzero(target_valid[:, target_month_index]).astype(np.int32)
        count = int(point_index.size)
        start, stop = cursor, cursor + count
        history_slice = slice(target_month_index - LOOKBACK, target_month_index)
        x_values_mm[start:stop] = clean_values[point_index, history_slice, :]
        window_valid = history_valid[point_index, history_slice]
        x_mask_mm[start:stop] = window_valid.astype(np.uint8)
        x_age_mm[start:stop] = _history_age(window_valid)
        history_month_mm[start:stop] = np.arange(target_month_index - LOOKBACK, target_month_index, dtype=np.int16)
        y_mm[start:stop] = target_values[point_index, target_month_index, :]
        point_index_mm[start:stop] = point_index
        target_month_mm[start:stop] = target_month_index

        selected_ids = np.asarray(point_ids, dtype=object)[point_index]
        sample_ids = [f"{point_id}__{month}" for point_id in selected_ids]
        split_name = split_map[month]
        metadata_parts.append(
            pd.DataFrame(
                {
                    "sample_id": sample_ids,
                    "coord_point_id": selected_ids,
                    "point_index": point_index,
                    "target_year_month": month,
                    "target_month_index": target_month_index,
                    "split": split_name,
                }
            )
        )
        selected_y = target_values[point_index, target_month_index, :]
        sidecar_parts.append(
            pd.DataFrame(
                {
                    "sample_id": sample_ids,
                    "target_common_valid_acquisition_count": valid_count[point_index, target_month_index],
                    "target_overlap_max_abs_difference": overlap[point_index, target_month_index],
                    "target_reflectance_gt1_flag": (selected_y > 1.0).any(axis=1),
                    "target_reflectance_gt1_2_flag": (selected_y > 1.2).any(axis=1),
                }
            )
        )
        split_counts[split_name] += count
        per_month_counts[month] = count
        expected_valid = split_valid_n.get(month)
        if expected_valid is not None and expected_valid != count:
            raise ValueError(f"Target-valid count mismatch for {month}: split={expected_valid}, materialized={count}")
        cursor = stop

    if cursor != sample_count:
        raise RuntimeError(f"Internal sample-count mismatch: {cursor} != {sample_count}")
    for array in (x_values_mm, x_mask_mm, x_age_mm, history_month_mm, y_mm, point_index_mm, target_month_mm):
        array.flush()
    del array  # release the final Windows memmap handle before directory publication
    del x_values_mm, x_mask_mm, x_age_mm, history_month_mm, y_mm, point_index_mm, target_month_mm

    metadata = pd.concat(metadata_parts, ignore_index=True)
    sidecar = pd.concat(sidecar_parts, ignore_index=True)
    if metadata["sample_id"].duplicated().any():
        raise RuntimeError("Duplicate sample IDs produced")
    metadata.to_parquet(staging / "metadata.parquet", index=False)
    sidecar.to_parquet(staging / "target_quality_sidecar.parquet", index=False)
    month_ordinals = np.asarray([month_ordinal(month) for month in months], dtype=np.int32)
    np.save(staging / "month_ordinals.npy", month_ordinals)
    pd.DataFrame({"month_index": np.arange(len(months)), "year_month": months, "month_ordinal": month_ordinals}).to_parquet(
        staging / "month_registry.parquet", index=False
    )
    pd.DataFrame({"point_index": np.arange(len(point_ids)), "coord_point_id": point_ids}).to_parquet(
        staging / "point_registry.parquet", index=False
    )

    files = sorted(path for path in staging.iterdir() if path.is_file())
    file_records = [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files]
    module_dir = Path(__file__).resolve().parent
    builder_files = [module_dir / name for name in ("data.py", "constants.py", "preprocessing.py", "utils.py")]
    contract = {
        "schema_version": "s2_core_12to1_train_ready_v2_1",
        "builder_files": [{"path": str(item), "sha256": sha256_file(item)} for item in builder_files],
        "source_panel": str(panel_path),
        "source_panel_sha256": sha256_file(panel_path),
        "split_file": str(split_path),
        "split_file_sha256": sha256_file(split_path),
        "history_bands": list(HISTORY_BANDS),
        "target_bands": [column.removeprefix("S2_") for column in TARGET_COLUMNS],
        "lookback_months": LOOKBACK,
        "forecast_horizon_months": 1,
        "age_generation": "window_local_causal_from_x_mask",
        "history_valid_rule": {
            "all_ten_finite": True,
            "all_ten_nonzero": True,
            "common_valid_acquisition_count_min": HISTORY_MIN_VALID_ACQUISITIONS,
            "overlap_difference_lt_or_missing": OVERLAP_DIFFERENCE_LIMIT,
        },
        "target_valid_rule": {
            "all_six_finite": True,
            "all_six_nonzero": True,
            "common_valid_acquisition_count_min": TARGET_MIN_VALID_ACQUISITIONS,
            "overlap_difference_lt_or_missing": OVERLAP_DIFFERENCE_LIMIT,
            "threshold_comparison_dtype": "float64_source_precision",
        },
        "sample_count": sample_count,
        "split_counts": split_counts,
        "per_target_month_counts": per_month_counts,
        "point_count": len(point_ids),
        "month_count": len(months),
        "model_feature_tensors": ["x_values", "x_mask", "x_age", "history_month_indices"],
        "metadata_only_fields": ["sample_id", "coord_point_id", "point_index", "target_year_month", "split"],
        "target_quality_is_sidecar_only": True,
        "files": file_records,
    }
    contract["contract_hash"] = stable_json_hash(contract)
    write_json(staging / "manifest.json", contract)
    staging.replace(destination)
    return contract


class TrainingStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.metadata = pd.read_parquet(self.root / "metadata.parquet")
        self.x_values = np.load(self.root / "x_values.npy", mmap_mode="r")
        self.x_mask = np.load(self.root / "x_mask.npy", mmap_mode="r")
        self.x_age = np.load(self.root / "x_age.npy", mmap_mode="r")
        self.history_month_indices = np.load(self.root / "history_month_indices.npy", mmap_mode="r")
        self.y = np.load(self.root / "y.npy", mmap_mode="r")
        self.point_indices = np.load(self.root / "point_indices.npy", mmap_mode="r")
        self.target_month_indices = np.load(self.root / "target_month_indices.npy", mmap_mode="r")
        self.month_ordinals = np.load(self.root / "month_ordinals.npy", mmap_mode="r")
        expected = len(self.metadata)
        arrays = (self.x_values, self.x_mask, self.x_age, self.history_month_indices, self.y, self.point_indices, self.target_month_indices)
        if any(len(array) != expected for array in arrays):
            raise ValueError("Training store arrays and metadata have inconsistent sample counts")

    def close(self) -> None:
        for name in (
            "x_values", "x_mask", "x_age", "history_month_indices", "y",
            "point_indices", "target_month_indices", "month_ordinals",
        ):
            array = getattr(self, name, None)
            memory_map = getattr(array, "_mmap", None)
            if memory_map is not None:
                memory_map.close()

    def __enter__(self) -> "TrainingStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def indices(self, splits: str | Iterable[str], *, allow_test: bool = False) -> np.ndarray:
        requested = {splits} if isinstance(splits, str) else set(splits)
        if "test" in requested and not allow_test:
            raise PermissionError("Test split is sealed. Explicit allow_test=True is required after final configuration freeze.")
        unknown = requested - {"train", "validation", "test"}
        if unknown:
            raise ValueError(f"Unknown splits: {sorted(unknown)}")
        return np.flatnonzero(self.metadata["split"].isin(requested).to_numpy()).astype(np.int64)

    def dataset(
        self,
        splits: str | Iterable[str],
        preprocessor: FoldPreprocessor,
        *,
        allow_test: bool = False,
        include_targets: bool = True,
    ) -> "ForecastDataset":
        return ForecastDataset(
            self,
            self.indices(splits, allow_test=allow_test),
            preprocessor,
            include_targets=include_targets,
        )


class ForecastDataset:
    def __init__(
        self,
        store: TrainingStore,
        sample_indices: np.ndarray,
        preprocessor: FoldPreprocessor,
        *,
        include_targets: bool = True,
    ):
        self.store = store
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64)
        self.preprocessor = preprocessor
        self.include_targets = bool(include_targets)

    def __len__(self) -> int:
        return int(self.sample_indices.size)

    def __getitem__(self, item: int) -> dict:
        import torch

        sample_index = int(self.sample_indices[item])
        x_raw = np.asarray(self.store.x_values[sample_index], dtype=np.float32)
        mask = np.asarray(self.store.x_mask[sample_index], dtype=np.float32)
        age = np.asarray(self.store.x_age[sample_index], dtype=np.float32) / float(LOOKBACK)
        history_month_indices = np.asarray(self.store.history_month_indices[sample_index], dtype=np.int64)
        ordinals = np.asarray(self.store.month_ordinals[history_month_indices], dtype=np.int64)
        calendar = calendar_features_from_ordinals(ordinals)
        result = {
            "sample_index": torch.tensor(sample_index, dtype=torch.int64),
            "x_value": torch.from_numpy(self.preprocessor.transform_x(x_raw)),
            "x_mask": torch.from_numpy(mask),
            "x_age": torch.from_numpy(age),
            "x_cal": torch.from_numpy(calendar),
        }
        if self.include_targets:
            y_raw = np.asarray(self.store.y[sample_index], dtype=np.float32)
            result["y"] = torch.from_numpy(self.preprocessor.transform_y(y_raw))
            result["y_raw"] = torch.from_numpy(y_raw.copy())
        return result
