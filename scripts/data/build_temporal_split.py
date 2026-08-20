"""Build target-month split counts from the assembled monthly panel."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMNS = ("S2_B2", "S2_B3", "S2_B4", "S2_B8", "S2_B11", "S2_B12")
COUNT_COLUMN = "S2_target_common_valid_acquisition_count"
OVERLAP_COLUMN = "S2_overlap_max_abs_difference"
OVERLAP_LIMIT = 0.05


def split_name(year: int) -> str:
    if year in (2022, 2023):
        return "train"
    if year == 2024:
        return "validation"
    if year in (2025, 2026):
        return "test"
    raise ValueError(f"Target year {year} is outside the released protocol")


def build_split(panel: pd.DataFrame) -> pd.DataFrame:
    required = {"coord_point_id", "year_month", COUNT_COLUMN, OVERLAP_COLUMN, *TARGET_COLUMNS}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Panel lacks required columns: {sorted(missing)}")
    if panel.duplicated(["coord_point_id", "year_month"]).any():
        raise ValueError("Panel contains duplicate point-month rows")

    rows: list[dict[str, object]] = []
    for month, frame in panel.groupby("year_month", sort=True):
        month = str(month)
        year = int(month[:4])
        if year < 2022:
            continue
        values = frame[list(TARGET_COLUMNS)].to_numpy(float)
        counts = pd.to_numeric(frame[COUNT_COLUMN], errors="coerce").to_numpy(float)
        overlap = pd.to_numeric(frame[OVERLAP_COLUMN], errors="coerce").to_numpy(float)
        valid = (
            np.isfinite(values).all(axis=1)
            & (values != 0).all(axis=1)
            & (counts >= 2)
            & (~np.isfinite(overlap) | (overlap < OVERLAP_LIMIT))
        )
        rows.append(
            {
                "target_year_month": month,
                "split": split_name(year),
                "target_valid_n": int(valid.sum()),
            }
        )
    result = pd.DataFrame(rows)
    expected = [str(value) for value in pd.period_range("2022-01", "2026-05", freq="M")]
    if result["target_year_month"].tolist() != expected:
        raise ValueError("Panel does not cover every released target month from 2022-01 to 2026-05")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        help="optional released split CSV; fail if regenerated counts differ",
    )
    args = parser.parse_args()
    result = build_split(pd.read_csv(args.panel))
    if args.reference:
        reference = pd.read_csv(args.reference)
        pd.testing.assert_frame_equal(result, reference, check_dtype=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    totals = result.groupby("split")["target_valid_n"].sum().to_dict()
    print(f"PASS: {len(result)} target months, sample totals={totals}")


if __name__ == "__main__":
    main()
