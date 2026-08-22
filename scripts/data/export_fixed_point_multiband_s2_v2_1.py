"""Sentinel-2 exporter for the fixed-point next-month panel.

The default invocation is read-only ``--plan-only``. The exporter writes
same-month observations; target shifting happens only in the local sample
builder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import namedtuple
from datetime import date
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "fixed_point_multiband_s2_v2_1"
RUN_ID = "fixed_point_s2_2021_01_2026_05_v2_1"
FROZEN_START = "2021-01"
FROZEN_END = "2026-05"
GEE_PROJECT = ""
REGION_ASSET = ""
POINT_ASSET = ""
BOUNDARY_GEOJSON = ""
DRIVE_FOLDER = "huanghua_fixed_point_s2_v2_1"

S2_SR_DATASET = "COPERNICUS/S2_SR_HARMONIZED"
S2_CLOUD_DATASET = "COPERNICUS/S2_CLOUD_PROBABILITY"
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
TARGET_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
ANALYSIS_CRS = "EPSG:32650"
ANALYSIS_CRS_TRANSFORM = [20, 0, 0, 0, -20, 0]
ANALYSIS_RESOLUTION_M = 20
SAMPLING_CRS = "EPSG:32650"
SAMPLING_CRS_TRANSFORM = [500, 0, 0, 0, -500, 0]
SAMPLING_SPACING_M = 500
SAMPLING_INCLUSION_RULE = "pixel_center_inside_region"
SAMPLING_FRAME_ID = "huanghua_utm50n_500m_center_v1"
EXPORT_SHARD_COUNT = 2
EXPORT_SHARD_SPLIT_EASTING_M = 540000
EXPORT_SHARDING_ROLE = "execution_only"
ZERO_COUNT_ENCODING = "explicit_zero"
SOURCE_DATATAKE_PROPERTY = "DATATAKE_IDENTIFIER"
TEMPORAL_OBSERVATION_UNIT = "canonical_acquisition_key"
S2_CLOUD_PROBABILITY_THRESHOLD = 65
LAST_VALID_LOOKBACK_MONTHS = 12
TARGET_MAIN_MIN_COMMON_COUNT = 2
TARGET_COVERAGE_MIN_COMMON_COUNT = 1
TARGET_HIGH_CONFIDENCE_MIN_COMMON_COUNT = 3
IQR_MIN_COUNT = 2
MIN_CLOUD_MATCH_FRACTION = 0.99
PILOT_MONTHS = ["2021-01", "2025-08", "2025-11"]
OVERLAP_MERGE_RULE = "granule_median"
ACQUISITION_TIME_TOLERANCE_SECONDS = 120

SCRIPT_PATH = Path(__file__).resolve()

Indicator = namedtuple("Indicator", "name group meaning role unit")


def month_range(start: str, end: str) -> list[tuple[int, int]]:
    sy, sm = (int(value) for value in start.split("-"))
    ey, em = (int(value) for value in end.split("-"))
    if (sy, sm) > (ey, em) or not (1 <= sm <= 12 and 1 <= em <= 12):
        raise ValueError(f"invalid inclusive month range: {start}..{end}")
    result: list[tuple[int, int]] = []
    year, month = sy, sm
    while (year, month) <= (ey, em):
        result.append((year, month))
        month += 1
        if month == 13:
            year += 1
            month = 1
    return result


def planned_indicators() -> list[Indicator]:
    rows = [
        Indicator("coord_point_id", "metadata", "deterministic UTM grid primary key", "metadata_only", "string"),
        Indicator("grid_x", "metadata", "500 m grid column index", "metadata_only", "integer"),
        Indicator("grid_y", "metadata", "500 m grid row index", "metadata_only", "integer"),
        Indicator("easting_m", "metadata", "EPSG:32650 point-centre easting", "metadata_only", "metre"),
        Indicator("northing_m", "metadata", "EPSG:32650 point-centre northing", "metadata_only", "metre"),
        Indicator("longitude", "metadata", "longitude for site linkage", "metadata_only", "degree"),
        Indicator("latitude", "metadata", "latitude for site linkage", "metadata_only", "degree"),
        Indicator("year", "metadata", "observation year", "metadata_only", "year"),
        Indicator("month", "metadata", "observation month", "metadata_only", "month"),
        Indicator("year_month", "metadata", "observation calendar month", "metadata_only", "YYYY-MM"),
    ]
    targets = set(TARGET_BANDS)
    for band in S2_BANDS:
        role = "history_input_and_target_source" if band in targets else "history_input"
        rows.append(Indicator(f"S2_{band}", "s2_reflectance", f"monthly median {band} surface reflectance", role, "reflectance"))
        rows.append(Indicator(f"S2_{band}_valid_count", "s2_quality", f"clear acquisition count for {band}", "quality_sidecar", "count"))
        rows.append(Indicator(f"S2_{band}_IQR", "s2_quality", f"within-month {band} P75-P25; undefined below two acquisitions", "quality_sidecar", "reflectance"))

    rows.extend([
        Indicator("S2_raw_observation_count", "s2_quality", "raw independent acquisitions available at the point", "history_quality", "count"),
        Indicator("S2_cloud_matched_observation_count", "s2_quality", "raw acquisitions with cloud-probability match", "history_quality", "count"),
        Indicator("S2_clear_observation_count", "s2_quality", "clear independent acquisitions at the point", "history_quality", "count"),
        Indicator("S2_cloud_match_fraction", "s2_quality", "matched divided by raw point acquisitions", "history_quality", "fraction"),
        Indicator("S2_clear_fraction_of_matched", "s2_quality", "clear divided by cloud-matched point acquisitions", "history_quality", "fraction"),
        Indicator("S2_usable_fraction_of_raw", "s2_quality", "clear divided by raw point acquisitions", "history_quality", "fraction"),
        Indicator("S2_target_common_valid_acquisition_count", "s2_quality", "clear acquisitions jointly valid for all six targets", "quality_sidecar", "count"),
        Indicator("S2_cloud_probability_mean", "s2_quality", "mean matched cloud probability", "quality_sidecar", "percent"),
        Indicator("S2_cloud_probability_p50", "s2_quality", "median matched cloud probability", "quality_sidecar", "percent"),
        Indicator("S2_cloud_probability_p90", "s2_quality", "90th percentile matched cloud probability", "quality_sidecar", "percent"),
        Indicator("S2_cloud_probability_near_threshold_fraction_50_80", "s2_quality", "fraction of matched acquisitions with cloud probability in [50,80]", "quality_sidecar", "fraction"),



        Indicator("S2_first_valid_time_utc_ms", "s2_quality", "first clear acquisition time in month", "quality_sidecar", "UTC milliseconds"),
        Indicator("S2_last_valid_time_utc_ms", "s2_quality", "last clear acquisition time in month", "quality_sidecar", "UTC milliseconds"),
        Indicator("S2_median_valid_time_utc_ms", "s2_quality", "median clear acquisition time in month", "quality_sidecar", "UTC milliseconds"),
        Indicator("S2_valid_acquisition_span_days", "s2_quality", "last minus first clear acquisition time", "quality_sidecar", "day"),
        Indicator("S2_overlap_acquisition_count", "s2_quality", "clear acquisitions formed from more than one valid granule at the point", "quality_metadata", "count"),
        Indicator("S2_overlap_max_abs_difference", "s2_quality", "maximum within-acquisition cross-granule absolute reflectance range", "quality_metadata", "reflectance"),


        Indicator("S2_days_since_last_valid", "s2_quality", "days from month end to most recent clear B4 acquisition in 12-month lookback", "history_quality", "day"),
        Indicator("S2_last_valid_left_censored", "s2_quality", "one when no valid acquisition exists in the finite lookback", "quality_sidecar", "binary"),
        Indicator("S2_granule_count", "s2_manifest", "AOI-filtered SR granules before acquisition merge", "acquisition_metadata", "count"),
        Indicator("S2_acquisition_count", "s2_manifest", "unique DATATAKE_IDENTIFIER acquisitions", "acquisition_metadata", "count"),
        Indicator("S2_cloud_matched_granule_count", "s2_manifest", "SR granules exactly matched to cloud probability", "acquisition_metadata", "count"),
        Indicator("S2_cloud_unmatched_granule_count", "s2_manifest", "SR granules missing an exact cloud match", "acquisition_metadata", "count"),
    ])
    return rows


def sampling_frame_sha256() -> str:
    payload = {
        "spatial_source": spatial_source_descriptor(),
        "sampling_frame_id": SAMPLING_FRAME_ID,
        "sampling_crs": SAMPLING_CRS,
        "sampling_crs_transform": SAMPLING_CRS_TRANSFORM,
        "sampling_inclusion_rule": SAMPLING_INCLUSION_RULE,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def spatial_source_descriptor() -> dict[str, str]:
    """Return the configured spatial source without contacting Earth Engine."""
    if POINT_ASSET:
        return {"type": "point_asset", "value": POINT_ASSET}
    if REGION_ASSET:
        return {"type": "region_asset", "value": REGION_ASSET}
    if BOUNDARY_GEOJSON:
        path = Path(BOUNDARY_GEOJSON).resolve()
        return {"type": "boundary_geojson", "value": str(path), "sha256": file_sha256(path)}
    return {"type": "unconfigured", "value": ""}


def load_region(ee: Any) -> Any:
    """Load a region used only to bound Sentinel-2 collection queries."""
    if POINT_ASSET:
        return ee.FeatureCollection(POINT_ASSET)
    if REGION_ASSET:
        return ee.FeatureCollection(REGION_ASSET)
    if BOUNDARY_GEOJSON:
        payload = json.loads(Path(BOUNDARY_GEOJSON).read_text(encoding="utf-8"))
        if payload.get("type") == "FeatureCollection":
            features = payload.get("features", [])
        elif payload.get("type") == "Feature":
            features = [payload]
        else:
            features = [{"type": "Feature", "properties": {}, "geometry": payload}]
        if not features:
            raise ValueError("boundary GeoJSON contains no features")
        return ee.FeatureCollection(features)
    raise RuntimeError("configure --point-asset, --region-asset, or --boundary-geojson")


def plan_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "start": FROZEN_START,
        "end": FROZEN_END,
        "datasets": [S2_SR_DATASET, S2_CLOUD_DATASET],
        "analysis_crs": ANALYSIS_CRS,
        "analysis_crs_transform": ANALYSIS_CRS_TRANSFORM,
        "analysis_resolution_m": ANALYSIS_RESOLUTION_M,
        "sampling_frame_id": SAMPLING_FRAME_ID,
        "sampling_crs": SAMPLING_CRS,
        "sampling_crs_transform": SAMPLING_CRS_TRANSFORM,
        "sampling_spacing_m": SAMPLING_SPACING_M,
        "sampling_inclusion_rule": SAMPLING_INCLUSION_RULE,
        "spatial_source": spatial_source_descriptor(),
        "sampling_frame_sha256": sampling_frame_sha256(),
        "export_shard_count": EXPORT_SHARD_COUNT,
        "export_shard_split_easting_m": EXPORT_SHARD_SPLIT_EASTING_M,
        "export_sharding_role": EXPORT_SHARDING_ROLE,
        "zero_count_encoding": ZERO_COUNT_ENCODING,
        "temporal_observation_unit": TEMPORAL_OBSERVATION_UNIT,
        "source_datatake_property": SOURCE_DATATAKE_PROPERTY,
        "cloud_probability_threshold": S2_CLOUD_PROBABILITY_THRESHOLD,
        "last_valid_lookback_months": LAST_VALID_LOOKBACK_MONTHS,
        "target_main_min_common_count": TARGET_MAIN_MIN_COMMON_COUNT,
        "target_coverage_min_common_count": TARGET_COVERAGE_MIN_COMMON_COUNT,
        "target_high_confidence_min_common_count": TARGET_HIGH_CONFIDENCE_MIN_COMMON_COUNT,
        "iqr_min_count": IQR_MIN_COUNT,
        "minimum_cloud_match_fraction": MIN_CLOUD_MATCH_FRACTION,
        "pilot_months": PILOT_MONTHS,
        "overlap_merge_rule": OVERLAP_MERGE_RULE,
        "acquisition_time_tolerance_seconds": ACQUISITION_TIME_TOLERANCE_SECONDS,
        "history_bands": S2_BANDS,
        "target_bands": TARGET_BANDS,
        "indicators": [row._asdict() for row in planned_indicators()],
    }


def plan_sha256() -> str:
    raw = json.dumps(plan_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_grid_aligned(transform: list[float]) -> bool:
    if len(transform) != 6:
        return False
    x_scale, x_shear, x_origin, y_shear, y_scale, y_origin = transform
    tolerance = 1e-8
    return (
        abs(x_scale - 20) < tolerance
        and abs(y_scale + 20) < tolerance
        and abs(x_shear) < tolerance
        and abs(y_shear) < tolerance
        and abs(x_origin / 20 - round(x_origin / 20)) < tolerance
        and abs(y_origin / 20 - round(y_origin / 20)) < tolerance
    )


def _source_grid_compatible(transform: Any) -> bool:
    if not isinstance(transform, list) or len(transform) != 6:
        return False
    x_scale, x_shear, x_origin, y_shear, y_scale, y_origin = transform
    tolerance = 1e-8
    return (
        abs(abs(x_scale) - abs(y_scale)) < tolerance
        and abs(x_scale) in {10, 20}
        and abs(x_shear) < tolerance
        and abs(y_shear) < tolerance
        and abs(x_origin / 20 - round(x_origin / 20)) < tolerance
        and abs(y_origin / 20 - round(y_origin / 20)) < tolerance
    )


def canonical_acquisition_key(value: str) -> str:
    text = str(value)
    key = re.sub(r"_N\d+[.]\d+$", "", text)
    if not key:
        raise ValueError(f"invalid DATATAKE_IDENTIFIER: {value!r}")
    return key


def evaluate_preflight_manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate immutable metadata and acquisition-key consistency before export."""
    expected_months = {f"{year:04d}-{month:02d}" for year, month in month_range(FROZEN_START, FROZEN_END)}
    observed_months = {row.get("year_month") for row in records if row.get("year_month")}
    system_indices = [row.get("system_index") for row in records]
    duplicate_system_index_count = len(system_indices) - len(set(system_indices))
    missing_datatake_count = sum(not row.get("datatake_identifier") for row in records)
    matched_count = sum(bool(row.get("cloud_matched")) for row in records)
    match_fraction = matched_count / len(records) if records else 0.0

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        value = row.get("datatake_identifier")
        if value:
            groups.setdefault(canonical_acquisition_key(str(value)), []).append(row)

    conflicts: list[dict[str, Any]] = []
    tolerance_ms = ACQUISITION_TIME_TOLERANCE_SECONDS * 1000
    for key, members in groups.items():
        times = [int(row["system_time_start"]) for row in members if row.get("system_time_start") is not None]
        spacecraft = {str(row.get("spacecraft_name")) for row in members if row.get("spacecraft_name") is not None}
        orbits = {str(row.get("sensing_orbit_number")) for row in members if row.get("sensing_orbit_number") is not None}
        time_span_ms = max(times) - min(times) if times else None
        if (time_span_ms is not None and time_span_ms > tolerance_ms) or len(spacecraft) > 1 or len(orbits) > 1:
            conflicts.append({
                "canonical_acquisition_key": key,
                "system_indices": [row.get("system_index") for row in members],
                "time_span_seconds": None if time_span_ms is None else time_span_ms / 1000,
                "spacecraft_names": sorted(spacecraft),
                "sensing_orbit_numbers": sorted(orbits),
            })

    month_stats: dict[str, dict[str, Any]] = {}
    for month in sorted(observed_months):
        subset = [row for row in records if row.get("year_month") == month]
        month_matched = sum(bool(row.get("cloud_matched")) for row in subset)
        month_stats[str(month)] = {
            "raw_granule_count": len(subset),
            "cloud_matched_granule_count": month_matched,
            "cloud_unmatched_granule_count": len(subset) - month_matched,
            "cloud_match_fraction": month_matched / len(subset) if subset else None,
        }

    incompatible_grids = [
        row.get("system_index") for row in records
        if not _source_grid_compatible(row.get("b2_transform"))
        or not _source_grid_compatible(row.get("b11_transform"))
    ]
    failed_checks: list[str] = []
    if observed_months != expected_months:
        failed_checks.append("month_coverage")
    if missing_datatake_count:
        failed_checks.append("datatake_complete")
    if duplicate_system_index_count:
        failed_checks.append("system_index_unique")
    if match_fraction < MIN_CLOUD_MATCH_FRACTION:
        failed_checks.append("cloud_match_fraction")
    if conflicts:
        failed_checks.append("acquisition_consistency")
    if incompatible_grids:
        failed_checks.append("grid_alignment")

    return {
        "passed": not failed_checks,
        "failed_checks": failed_checks,
        "warnings": [
            f"monthly_cloud_match_below_{MIN_CLOUD_MATCH_FRACTION:.2f}:{month}"
            for month, stats in month_stats.items()
            if stats["cloud_match_fraction"] is not None and stats["cloud_match_fraction"] < MIN_CLOUD_MATCH_FRACTION
        ],
        "raw_granule_count": len(records),
        "cloud_matched_granule_count": matched_count,
        "cloud_unmatched_granule_count": len(records) - matched_count,
        "cloud_match_fraction": match_fraction,
        "cloud_match_by_month": month_stats,
        "acquisition_count": len(groups),
        "multi_granule_acquisition_count": sum(len(members) > 1 for members in groups.values()),
        "acquisition_consistency_conflicts": conflicts,
        "duplicate_system_index_count": duplicate_system_index_count,
        "missing_datatake_count": missing_datatake_count,
        "missing_months": sorted(expected_months - observed_months),
        "unexpected_months": sorted(observed_months - expected_months),
        "incompatible_grid_system_indices": incompatible_grids,
    }


if __name__ == "__main__":
    import gee_s2_v2_1_runtime

    raise SystemExit(gee_s2_v2_1_runtime.main())
