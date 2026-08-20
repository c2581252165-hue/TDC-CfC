"""Merge downloaded monthly Earth Engine CSV exports into one panel."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY = "coord_point_id"
EXPECTED_START = "2021-01"
EXPECTED_END = "2026-05"


def expected_months(start: str, end: str) -> list[str]:
    return [str(value) for value in pd.period_range(start, end, freq="M")]


def read_exports(input_dir: Path) -> dict[str, list[pd.DataFrame]]:
    grouped: dict[str, list[pd.DataFrame]] = {}
    files = sorted(input_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV exports found below {input_dir}")
    for path in files:
        frame = pd.read_csv(path)
        missing = {KEY, "year_month"} - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} lacks required columns: {sorted(missing)}")
        months = frame["year_month"].dropna().astype(str).unique().tolist()
        if len(months) != 1:
            raise ValueError(f"{path.name} must contain exactly one year_month, found {months}")
        frame = frame.drop(columns=[column for column in (".geo", "system:index") if column in frame])
        grouped.setdefault(months[0], []).append(frame)
    return grouped


def merge_exports(input_dir: Path, expected_points: int) -> pd.DataFrame:
    grouped = read_exports(input_dir)
    wanted = expected_months(EXPECTED_START, EXPECTED_END)
    missing_months = sorted(set(wanted) - set(grouped))
    unexpected_months = sorted(set(grouped) - set(wanted))
    if missing_months or unexpected_months:
        raise ValueError(
            f"Month coverage mismatch; missing={missing_months}, unexpected={unexpected_months}"
        )

    monthly: list[pd.DataFrame] = []
    reference_ids: set[str] | None = None
    reference_columns: list[str] | None = None
    for month in wanted:
        frame = pd.concat(grouped[month], ignore_index=True)
        if frame[KEY].isna().any() or frame[KEY].astype(str).duplicated().any():
            raise ValueError(f"{month} contains missing or duplicate {KEY} values")
        if len(frame) != expected_points:
            raise ValueError(f"{month} has {len(frame)} points; expected {expected_points}")
        ids = set(frame[KEY].astype(str))
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ValueError(f"{month} does not contain the same fixed-point registry")
        columns = frame.columns.tolist()
        if reference_columns is None:
            reference_columns = columns
        elif columns != reference_columns:
            raise ValueError(f"{month} column order/schema differs from the first month")
        count_columns = [column for column in frame if column.endswith("_count")]
        frame[count_columns] = frame[count_columns].fillna(0)
        monthly.append(frame.sort_values(KEY, kind="stable"))

    panel = pd.concat(monthly, ignore_index=True)
    if len(panel) != expected_points * len(wanted):
        raise RuntimeError("Merged panel size violates the point-by-month contract")
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-points", type=int, default=9596)
    args = parser.parse_args()
    panel = merge_exports(args.input_dir, args.expected_points)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.output, index=False)
    print(
        f"PASS: {len(panel)} rows, {panel['year_month'].nunique()} months, "
        f"{panel[KEY].nunique()} fixed points -> {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
