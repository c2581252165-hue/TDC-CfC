"""Earth Engine runtime for export_fixed_point_multiband_s2_v2_1.py."""
from __future__ import annotations

import argparse
import importlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import export_fixed_point_multiband_s2_v2_1 as cfg

RUNTIME_PATH = Path(__file__).resolve()
ROOT = RUNTIME_PATH.parents[2]


def load_ee() -> Any:
    return importlib.import_module("ee")


def next_month(year: int, month: int) -> tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


def iso_month_bounds(year: int, month: int) -> tuple[str, str]:
    end_year, end_month = next_month(year, month)
    return f"{year:04d}-{month:02d}-01", f"{end_year:04d}-{end_month:02d}-01"


def selected_months(single_month: str | None) -> list[tuple[int, int]]:
    months = cfg.month_range(cfg.FROZEN_START, cfg.FROZEN_END)
    if single_month is None:
        return months
    selected = cfg.month_range(single_month, single_month)
    if selected[0] not in months:
        raise ValueError(f"month {single_month} is outside the configured range")
    return selected


def build_sampling_grid(ee: Any) -> Any:
    if cfg.POINT_ASSET:
        return ee.FeatureCollection(cfg.POINT_ASSET)

    region = cfg.load_region(ee)
    projection = ee.Projection(cfg.SAMPLING_CRS, cfg.SAMPLING_CRS_TRANSFORM)
    sampled = ee.Image.pixelCoordinates(projection).sample(
        region=region.geometry(),
        projection=projection,
        geometries=True,
        tileScale=4,
    )

    def annotate(feature: Any) -> Any:
        feature = ee.Feature(feature)
        grid_x = ee.Number(feature.get("x")).toInt()
        grid_y = ee.Number(feature.get("y")).toInt()
        centre = feature.geometry().transform(cfg.SAMPLING_CRS, 0.1)
        xy = ee.List(centre.coordinates())
        easting = ee.Number(xy.get(0))
        northing = ee.Number(xy.get(1))
        lonlat = ee.List(centre.transform("EPSG:4326", 0.1).coordinates())
        point_id = (
            ee.String("utm50n_500m_x")
            .cat(grid_x.format("%d"))
            .cat("_y")
            .cat(grid_y.format("%d"))
        )
        x_error = easting.subtract(10).divide(20).round().multiply(20).add(10).subtract(easting).abs()
        y_error = northing.subtract(10).divide(20).round().multiply(20).add(10).subtract(northing).abs()
        return ee.Feature(centre, {
            "coord_point_id": point_id,
            "grid_x": grid_x,
            "grid_y": grid_y,
            "easting_m": easting,
            "northing_m": northing,
            "longitude": lonlat.get(0),
            "latitude": lonlat.get(1),
            "analysis_center_alignment_error_m": x_error.max(y_error),
        })

    return ee.FeatureCollection(sampled.map(annotate))


def build_export_shards(ee: Any, points: Any) -> list[dict[str, Any]]:
    split = cfg.EXPORT_SHARD_SPLIT_EASTING_M
    return [
        {"shard_id": "part01of02", "predicate": f"easting_m < {split}", "points": points.filter(ee.Filter.lt("easting_m", split))},
        {"shard_id": "part02of02", "predicate": f"easting_m >= {split}", "points": points.filter(ee.Filter.gte("easting_m", split))},
    ]


def shard_descriptions(base: str) -> list[str]:
    return [f"{base}_part01of02", f"{base}_part02of02"]


def profile_sampling_grid(ee: Any, points: Any) -> dict[str, Any]:
    split = cfg.EXPORT_SHARD_SPLIT_EASTING_M
    stats = ee.Dictionary({
        "count": points.size(),
        "unique_ids": points.aggregate_count_distinct("coord_point_id"),
        "alignment_error": points.aggregate_max("analysis_center_alignment_error_m"),
        "part01of02": points.filter(ee.Filter.lt("easting_m", split)).size(),
        "part02of02": points.filter(ee.Filter.gte("easting_m", split)).size(),
    }).getInfo()
    count = int(stats["count"])
    unique_ids = int(stats["unique_ids"])
    alignment_error = float(stats.get("alignment_error") or 0)
    source = cfg.spatial_source_descriptor()
    asset_id = cfg.POINT_ASSET or cfg.REGION_ASSET
    asset = ee.data.getAsset(asset_id) if asset_id else {}
    return {
        "sampling_frame_id": cfg.SAMPLING_FRAME_ID,
        "sampling_frame_sha256": cfg.sampling_frame_sha256(),
        "spatial_source": source,
        "source_asset_type": asset.get("type"),
        "source_asset_update_time": asset.get("updateTime"),
        "sampling_crs": cfg.SAMPLING_CRS,
        "sampling_crs_transform": cfg.SAMPLING_CRS_TRANSFORM,
        "inclusion_rule": cfg.SAMPLING_INCLUSION_RULE,
        "row_count": count,
        "unique_point_id_count": unique_ids,
        "max_analysis_center_alignment_error_m": alignment_error,
        "export_shard_split_easting_m": split,
        "export_shard_counts": {"part01of02": int(stats["part01of02"]), "part02of02": int(stats["part02of02"])},
    }


def raw_collection(ee: Any, aoi: Any, start: Any, end: Any) -> Any:
    return ee.ImageCollection(cfg.S2_SR_DATASET).filterBounds(aoi).filterDate(start, end)


def joined_collection(ee: Any, aoi: Any, start: Any, end: Any) -> Any:
    sr = raw_collection(ee, aoi, start, end)
    clouds = ee.ImageCollection(cfg.S2_CLOUD_DATASET).filterBounds(aoi).filterDate(start, end)
    joined = ee.Join.saveFirst("cloud_probability").apply(
        primary=sr,
        secondary=clouds,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )
    return ee.ImageCollection(joined).filter(ee.Filter.notNull(["cloud_probability"]))


def _to_reference(image: Any) -> Any:
    return image.reproject(crs=cfg.ANALYSIS_CRS, crsTransform=cfg.ANALYSIS_CRS_TRANSFORM)


def _support_at_20m(ee: Any, mask: Any) -> Any:
    support = (
        ee.Image(mask)
        .unmask(0)
        .toFloat()
        .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=16, bestEffort=False)
    )
    return _to_reference(support).gte(0.999)


def _raw_10m_mask(ee: Any, image: Any) -> Any:
    all_band_mask = image.select(cfg.S2_BANDS).mask().reduce(ee.Reducer.min())
    edge_mask = image.select("B8A").mask().And(image.select("B9").mask())
    return all_band_mask.And(edge_mask)


def _with_acquisition_properties(ee: Any, prepared: Any, source: Any) -> Any:
    canonical = ee.String(source.get(cfg.SOURCE_DATATAKE_PROPERTY)).replace(
        "_N[0-9]+[.][0-9]+$", ""
    )
    return prepared.copyProperties(source, source.propertyNames()).set(
        cfg.TEMPORAL_OBSERVATION_UNIT, canonical
    )


def prepare_raw_granule(ee: Any, image: Any) -> Any:
    raw_support = _support_at_20m(ee, _raw_10m_mask(ee, image))
    flag = _to_reference(ee.Image.constant(1).rename("raw_flag")).updateMask(raw_support)
    return _with_acquisition_properties(ee, flag, image)


def prepare_joined_granule(ee: Any, image: Any) -> Any:
    cloud = ee.Image(image.get("cloud_probability")).select("probability")
    raw_mask = _raw_10m_mask(ee, image)
    scl = image.select("SCL")
    scl_good = (
        scl.neq(0)
        .And(scl.neq(1))
        .And(scl.neq(3))
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )
    clear_mask = raw_mask.And(scl_good).And(cloud.lt(cfg.S2_CLOUD_PROBABILITY_THRESHOLD))
    raw_support = _support_at_20m(ee, raw_mask)
    clear_support = _support_at_20m(ee, clear_mask)

    parts = []
    ten_metre = {"B2", "B3", "B4", "B8"}
    for band in cfg.S2_BANDS:
        source = image.select(band).multiply(0.0001).updateMask(clear_mask)
        if band in ten_metre:
            source = source.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=16, bestEffort=False)
        parts.append(_to_reference(source).updateMask(clear_support).rename(f"S2_{band}"))

    cloud_20m = (
        cloud.updateMask(raw_mask)
        .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=16, bestEffort=False)
    )
    parts.extend(
        [
            _to_reference(cloud_20m).updateMask(raw_support).rename("cloud_probability_value"),
            _to_reference(cloud_20m.gte(50).And(cloud_20m.lte(80))).updateMask(raw_support).rename("cloud_probability_near_threshold_50_80"),

            _to_reference(ee.Image.constant(1).rename("matched_flag")).updateMask(raw_support),
            _to_reference(ee.Image.constant(1).rename("clear_flag")).updateMask(clear_support),
            _to_reference(ee.Image.constant(image.date().millis()).toDouble().rename("valid_millis")).updateMask(clear_support),
        ]
    )
    return _with_acquisition_properties(ee, ee.Image.cat(parts), image)


def prepare_last_valid_granule(ee: Any, image: Any) -> Any:
    """Build only the clear-support timestamp needed by the 12-month lookback."""
    cloud = ee.Image(image.get("cloud_probability")).select("probability")
    raw_mask = _raw_10m_mask(ee, image)
    scl = image.select("SCL")
    scl_good = (
        scl.neq(0)
        .And(scl.neq(1))
        .And(scl.neq(3))
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )
    clear_support = _support_at_20m(
        ee,
        raw_mask.And(scl_good).And(cloud.lt(cfg.S2_CLOUD_PROBABILITY_THRESHOLD)),
    )
    millis = _to_reference(
        ee.Image.constant(image.date().millis()).toDouble().rename("valid_millis")
    ).updateMask(clear_support)
    return _with_acquisition_properties(ee, millis, image)


def merge_by_acquisition(ee: Any, collection: Any, overlap_audit: bool = False) -> Any:
    """Merge granules deterministically; optionally retain overlap disagreement."""
    key = cfg.TEMPORAL_OBSERVATION_UNIT
    seeds = collection.distinct([key])
    grouped = ee.Join.saveAll("granules").apply(
        primary=seeds,
        secondary=collection,
        condition=ee.Filter.equals(leftField=key, rightField=key),
    )

    def merge_one(element: Any) -> Any:
        seed = ee.Image(element)
        members = ee.ImageCollection.fromImages(ee.List(seed.get("granules")))
        merged = members.median()
        if overlap_audit:
            reflectance_names = [f"S2_{band}" for band in cfg.S2_BANDS]
            minmax = members.select(reflectance_names).reduce(ee.Reducer.minMax())
            ranges = [
                minmax.select(f"S2_{band}_max").subtract(minmax.select(f"S2_{band}_min"))
                for band in cfg.S2_BANDS
            ]
            max_difference = ee.Image.cat(ranges).reduce(ee.Reducer.max()).rename(
                "acquisition_overlap_max_abs_difference"
            )
            overlap_count = members.select("S2_B2").count().rename(
                "acquisition_overlap_granule_count"
            )
            overlap_flag = overlap_count.gt(1).rename("acquisition_overlap_flag")
            merged = merged.addBands([overlap_count, overlap_flag, max_difference])
        return merged.set(
            {
                key: seed.get(key),
                "system:time_start": members.aggregate_min("system:time_start"),
                "granules_in_acquisition": members.size(),
            }
        )

    return ee.ImageCollection(grouped.map(merge_one))


def acquisition_collections(ee: Any, aoi: Any, start: Any, end: Any) -> tuple[Any, Any, Any, Any]:
    raw = raw_collection(ee, aoi, start, end)
    joined = joined_collection(ee, aoi, start, end)
    raw_prepared = raw.map(lambda image: prepare_raw_granule(ee, image))
    joined_prepared = joined.map(lambda image: prepare_joined_granule(ee, image))
    return raw, joined, merge_by_acquisition(ee, raw_prepared), merge_by_acquisition(ee, joined_prepared, overlap_audit=True)


def last_valid_acquisitions(ee: Any, aoi: Any, start: Any, end: Any) -> Any:
    joined = joined_collection(ee, aoi, start, end)
    prepared = joined.map(lambda image: prepare_last_valid_granule(ee, image))
    return merge_by_acquisition(ee, prepared)


def _constant(ee: Any, value: Any, name: str) -> Any:
    return _to_reference(ee.Image.constant(value).rename(name))


def zero_safe_count(collection: Any, source_band: str, output_band: str) -> Any:
    return collection.select(source_band).count().rename(output_band)


def month_image(ee: Any, year: int, month: int, aoi: Any) -> Any:
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    lookback_start = start.advance(1 - cfg.LAST_VALID_LOOKBACK_MONTHS, "month")
    raw, joined, raw_acq, clear_acq = acquisition_collections(ee, aoi, start, end)
    last_valid_history = last_valid_acquisitions(ee, aoi, lookback_start, end)

    reflectance_names = [f"S2_{band}" for band in cfg.S2_BANDS]
    median = clear_acq.select(reflectance_names).median()
    counts = ee.Image.cat(
        [zero_safe_count(clear_acq, f"S2_{band}", f"S2_{band}_valid_count") for band in cfg.S2_BANDS]
    )
    percentile = clear_acq.select(reflectance_names).reduce(ee.Reducer.percentile([25, 75]))
    iqr_parts = []
    for band in cfg.S2_BANDS:
        count = counts.select(f"S2_{band}_valid_count")
        iqr = (
            percentile.select(f"S2_{band}_p75")
            .subtract(percentile.select(f"S2_{band}_p25"))
            .rename(f"S2_{band}_IQR")
            .updateMask(count.gte(cfg.IQR_MIN_COUNT))
        )
        iqr_parts.append(iqr)
    iqr_image = ee.Image.cat(iqr_parts)
    raw_count = zero_safe_count(raw_acq, "raw_flag", "S2_raw_observation_count")
    matched_count = zero_safe_count(clear_acq, "matched_flag", "S2_cloud_matched_observation_count")
    clear_count = zero_safe_count(clear_acq, "clear_flag", "S2_clear_observation_count")
    match_fraction = matched_count.divide(raw_count).updateMask(raw_count.gt(0)).rename("S2_cloud_match_fraction")
    clear_fraction = clear_count.divide(matched_count).updateMask(matched_count.gt(0)).rename("S2_clear_fraction_of_matched")
    usable_fraction = clear_count.divide(raw_count).updateMask(raw_count.gt(0)).rename("S2_usable_fraction_of_raw")
    target_common_count = clear_count.rename("S2_target_common_valid_acquisition_count")
    cloud_mean = clear_acq.select("cloud_probability_value").mean().rename("S2_cloud_probability_mean")
    cloud_percentiles = clear_acq.select("cloud_probability_value").reduce(ee.Reducer.percentile([50, 90]))
    cloud_p50 = cloud_percentiles.select("cloud_probability_value_p50").rename("S2_cloud_probability_p50")
    cloud_p90 = cloud_percentiles.select("cloud_probability_value_p90").rename("S2_cloud_probability_p90")
    cloud_near = clear_acq.select("cloud_probability_near_threshold_50_80").mean().rename(
        "S2_cloud_probability_near_threshold_fraction_50_80"
    )
    first_valid = clear_acq.select("valid_millis").min().rename("S2_first_valid_time_utc_ms")
    last_valid_month = clear_acq.select("valid_millis").max().rename("S2_last_valid_time_utc_ms")
    median_valid = clear_acq.select("valid_millis").median().rename("S2_median_valid_time_utc_ms")
    valid_span = last_valid_month.subtract(first_valid).divide(86_400_000).rename(
        "S2_valid_acquisition_span_days"
    )
    overlap_count = clear_acq.select("acquisition_overlap_flag").sum().rename("S2_overlap_acquisition_count")
    overlap_max = clear_acq.select("acquisition_overlap_max_abs_difference").max().rename(
        "S2_overlap_max_abs_difference"
    )

    last_millis = last_valid_history.select("valid_millis").max()
    has_last = last_valid_history.select("valid_millis").count().gt(0)
    days_since = (
        _constant(ee, end.millis(), "month_end_millis")
        .toDouble()
        .subtract(last_millis)
        .divide(86_400_000)
        .rename("S2_days_since_last_valid")
        .updateMask(has_last)
    )
    left_censored = has_last.Not().rename("S2_last_valid_left_censored")

    manifest = ee.Image.cat(
        [
            _constant(ee, raw.size(), "S2_granule_count"),
            _constant(ee, raw_acq.size(), "S2_acquisition_count"),
            _constant(ee, joined.size(), "S2_cloud_matched_granule_count"),
            _constant(ee, raw.size().subtract(joined.size()), "S2_cloud_unmatched_granule_count"),
        ]
    )
    return ee.Image.cat(
        [
            median,
            counts,
            iqr_image,
            raw_count,
            matched_count,
            clear_count,
            match_fraction,
            clear_fraction,
            usable_fraction,
            target_common_count,
            cloud_mean,
            cloud_p50,
            cloud_p90,
            cloud_near,
            first_valid,
            last_valid_month,
            median_valid,
            valid_span,
            overlap_count,
            overlap_max,
            days_since,
            left_censored,
            manifest,
        ]
    ).setDefaultProjection(crs=cfg.ANALYSIS_CRS, crsTransform=cfg.ANALYSIS_CRS_TRANSFORM)


def reduce_to_points(ee: Any, image: Any, points: Any, year: int, month: int) -> Any:
    rows = image.reduceRegions(
        collection=points,
        reducer=ee.Reducer.first(),
        crs=cfg.ANALYSIS_CRS,
        crsTransform=cfg.ANALYSIS_CRS_TRANSFORM,
        tileScale=4,
    )
    year_month = f"{year:04d}-{month:02d}"
    return rows.map(
        lambda feature: feature.set(
            {
                "year": year,
                "month": month,
                "year_month": year_month,
                "schema_version": cfg.SCHEMA_VERSION,
                "run_id": cfg.RUN_ID,
            }
        ).setGeometry(None)
    )


def start_export(ee: Any, rows: Any, description: str) -> dict[str, Any]:
    selectors = [row.name for row in cfg.planned_indicators()] + ["schema_version", "run_id"]
    task = ee.batch.Export.table.toDrive(
        collection=rows,
        description=description,
        folder=cfg.DRIVE_FOLDER,
        fileNamePrefix=description,
        fileFormat="CSV",
        selectors=selectors,
    )
    task.start()
    return {"task_id": task.id, "state": task.status().get("state", "UNKNOWN")}


def _full_date_range() -> tuple[str, str]:
    start = f"{cfg.FROZEN_START}-01"
    ey, em = cfg.month_range(cfg.FROZEN_END, cfg.FROZEN_END)[0]
    ny, nm = next_month(ey, em)
    return start, f"{ny:04d}-{nm:02d}-01"


def collect_preflight_manifest(ee: Any, aoi: Any) -> list[dict[str, Any]]:
    start, end = _full_date_range()
    raw = raw_collection(ee, aoi, start, end)
    joined = joined_collection(ee, aoi, start, end)
    properties = [
        "system:index",
        "system:time_start",
        cfg.SOURCE_DATATAKE_PROPERTY,
        "MGRS_TILE",
        "PRODUCT_ID",
        "PROCESSING_BASELINE",
        "GENERATION_TIME",
        "SPACECRAFT_NAME",
        "SENSING_ORBIT_NUMBER",
        "DATASTRIP_ID",
        "CLOUDY_PIXEL_PERCENTAGE",
        "NODATA_PIXEL_PERCENTAGE",
    ]
    arrays = {prop: raw.aggregate_array(prop).getInfo() for prop in properties}
    matched_indices = set(joined.aggregate_array("system:index").getInfo())
    tiles = sorted(set(arrays["MGRS_TILE"]))
    transforms = {}
    for tile in tiles:
        image = ee.Image(raw.filter(ee.Filter.eq("MGRS_TILE", tile)).first())
        b2_projection = image.select("B2").projection().getInfo()
        b11_projection = image.select("B11").projection().getInfo()
        transforms[tile] = {
            "b2_crs": b2_projection["crs"],
            "b2_transform": b2_projection["transform"],
            "b11_crs": b11_projection["crs"],
            "b11_transform": b11_projection["transform"],
        }
    records = []
    for index in range(len(arrays["system:index"])):
        timestamp_ms = arrays["system:time_start"][index]
        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        tile = arrays["MGRS_TILE"][index]
        record = {
            "system_index": arrays["system:index"][index],
            "system_time_start": timestamp_ms,
            "year_month": dt.strftime("%Y-%m"),
            "datatake_identifier": arrays[cfg.SOURCE_DATATAKE_PROPERTY][index],
            "mgrs_tile": tile,
            "product_id": arrays["PRODUCT_ID"][index],
            "processing_baseline": arrays["PROCESSING_BASELINE"][index],
            "generation_time": arrays["GENERATION_TIME"][index],
            "spacecraft_name": arrays["SPACECRAFT_NAME"][index],
            "sensing_orbit_number": arrays["SENSING_ORBIT_NUMBER"][index],
            "datastrip_id": arrays["DATASTRIP_ID"][index],
            "cloudy_pixel_percentage": arrays["CLOUDY_PIXEL_PERCENTAGE"][index],
            "nodata_pixel_percentage": arrays["NODATA_PIXEL_PERCENTAGE"][index],
            "cloud_system_index": arrays["system:index"][index] if arrays["system:index"][index] in matched_indices else None,
            "cloud_matched": arrays["system:index"][index] in matched_indices,
            **transforms[tile],
        }
        records.append(record)
    return records


def run_preflight(ee: Any, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    region = cfg.load_region(ee)
    aoi = region.geometry()
    records = collect_preflight_manifest(ee, aoi)
    evaluation = cfg.evaluate_preflight_manifest(records)
    points_profile = profile_sampling_grid(ee, build_sampling_grid(ee))
    if points_profile["row_count"] <= 0:
        evaluation["failed_checks"].append("sampling_frame_empty")
    if points_profile["row_count"] != points_profile["unique_point_id_count"]:
        evaluation["failed_checks"].append("sampling_frame_duplicate_ids")
    if points_profile["max_analysis_center_alignment_error_m"] > 1e-6:
        evaluation["failed_checks"].append("sampling_frame_not_nested_on_20m_centres")
    shard_counts = points_profile["export_shard_counts"]
    if any(value <= 0 for value in shard_counts.values()) or sum(shard_counts.values()) != points_profile["row_count"]:
        evaluation["failed_checks"].append("export_shards_not_complete_partition")
    evaluation["failed_checks"] = sorted(set(evaluation["failed_checks"]))
    evaluation["passed"] = not evaluation["failed_checks"]
    manifest_path = report_dir / "s2_granule_manifest.json"
    manifest_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        **evaluation,
        "schema_version": cfg.SCHEMA_VERSION,
        "run_id": cfg.RUN_ID,
        "plan_sha256": cfg.plan_sha256(),
        "script_sha256": cfg.file_sha256(cfg.SCRIPT_PATH),
        "runtime_sha256": cfg.file_sha256(RUNTIME_PATH),
        "sampling_frame": points_profile,
        "manifest_path": str(manifest_path),
        "manifest_sha256": cfg.file_sha256(manifest_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path = report_dir / "preflight_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def submission_matches_current(row: dict[str, Any], plan_sha256: str, runtime_sha256: str) -> bool:
    return row.get("plan_sha256") == plan_sha256 and row.get("runtime_sha256") == runtime_sha256


def submitted_descriptions_by_state(ee: Any, path: Path) -> tuple[set[str], set[str]]:
    if not path.exists():
        return set(), set()
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("task_id") and row.get("description"):
                latest[row["description"]] = row
    current_plan = cfg.plan_sha256()
    current_runtime = cfg.file_sha256(RUNTIME_PATH)
    latest = {description: row for description, row in latest.items() if submission_matches_current(row, current_plan, current_runtime)}
    if not latest:
        return set(), set()
    statuses = ee.data.getTaskStatus([row["task_id"] for row in latest.values()])
    by_id = {str(row.get("id") or row.get("task_id") or row.get("name")): row for row in statuses}
    protected, completed = set(), set()
    for description, row in latest.items():
        state = by_id.get(str(row["task_id"]), {}).get("state", "UNKNOWN")
        if state not in {"FAILED", "CANCELLED", "CANCEL_REQUESTED"}:
            protected.add(description)
        if state == "COMPLETED":
            completed.add(description)
    return protected, completed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--submit", action="store_true")
    parser.add_argument("--gee-project", required=True)
    spatial = parser.add_mutually_exclusive_group(required=True)
    spatial.add_argument(
        "--point-asset",
        help="Earth Engine table asset created from data/points/point_registry.csv",
    )
    spatial.add_argument("--region-asset", help="authorized Earth Engine boundary asset")
    spatial.add_argument("--boundary-geojson", type=Path, help="authorized local GeoJSON boundary")
    parser.add_argument("--month", help="optional pilot month YYYY-MM")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "outputs/gee")
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--max-new-tasks", type=int)
    parser.add_argument("--submission-delay-seconds", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg.GEE_PROJECT = args.gee_project
    cfg.POINT_ASSET = args.point_asset or ""
    cfg.REGION_ASSET = args.region_asset or ""
    cfg.BOUNDARY_GEOJSON = str(args.boundary_geojson.resolve()) if args.boundary_geojson else ""
    if not args.preflight and not args.submit:
        print(json.dumps({"plan": cfg.plan_payload(), "plan_sha256": cfg.plan_sha256()}, ensure_ascii=False, indent=2))
        return 0

    if args.submit:
        if args.preflight_report is None or not args.preflight_report.exists():
            raise RuntimeError("--submit requires a passing --preflight-report")
        bound_report = json.loads(args.preflight_report.read_text(encoding="utf-8"))
        if not bound_report.get("passed"):
            raise RuntimeError("preflight report did not pass")
        if bound_report.get("plan_sha256") != cfg.plan_sha256():
            raise RuntimeError("preflight report does not match the current acquisition plan")
        if bound_report.get("runtime_sha256") != cfg.file_sha256(RUNTIME_PATH):
            raise RuntimeError("runtime module changed after preflight; rerun preflight")

    ee = load_ee()
    ee.Initialize(project=cfg.GEE_PROJECT)
    if args.preflight:
        report_path = run_preflight(ee, args.report_dir)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print(json.dumps({"report_path": str(report_path), "report_sha256": cfg.file_sha256(report_path), **report}, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 2

    months = selected_months(args.month)
    points = build_sampling_grid(ee)
    points_profile = bound_report["sampling_frame"]
    shards = build_export_shards(ee, points)
    aoi = cfg.load_region(ee).geometry()
    log_path = args.report_dir / "submission_log.jsonl"
    protected, completed = submitted_descriptions_by_state(ee, log_path)
    submitted = 0
    stop = False
    for year, month in months:
        base = f"{cfg.RUN_ID}_{year:04d}_{month:02d}"
        if base in completed:
            continue
        image = month_image(ee, year, month, aoi)
        for shard in shards:
            description = f"{base}_{shard['shard_id']}"
            if description in protected:
                continue
            if args.max_new_tasks is not None and submitted >= args.max_new_tasks:
                stop = True
                break
            try:
                rows = reduce_to_points(ee, image, shard["points"], year, month)
                task = start_export(ee, rows, description)
                row = {
                    **task,
                    "description": description,
                    "year_month": f"{year:04d}-{month:02d}",
                    "shard_id": shard["shard_id"],
                    "shard_count": cfg.EXPORT_SHARD_COUNT,
                    "shard_predicate": shard["predicate"],
                    "shard_point_count": points_profile["export_shard_counts"][shard["shard_id"]],
                    "plan_sha256": cfg.plan_sha256(),
                    "script_sha256": cfg.file_sha256(cfg.SCRIPT_PATH),
                    "runtime_sha256": cfg.file_sha256(RUNTIME_PATH),
                    "sampling_frame_sha256": points_profile["sampling_frame_sha256"],
                    "point_count": points_profile["row_count"],
                    "preflight_report_sha256": cfg.file_sha256(args.preflight_report),
                    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            except Exception as exc:
                row = {
                    "task_id": None,
                    "state": "FAILED_TO_START",
                    "description": description,
                    "year_month": f"{year:04d}-{month:02d}",
                    "shard_id": shard["shard_id"],
                    "error": repr(exc),
                    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                append_jsonl(log_path, row)
                raise
            append_jsonl(log_path, row)
            print(json.dumps(row, ensure_ascii=False))
            protected.add(description)
            submitted += 1
            time.sleep(args.submission_delay_seconds)
        if stop:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
