from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def test_fixed_point_registry_contract() -> None:
    points = pd.read_parquet(ROOT / "data/points/point_registry.parquet")
    required = {
        "point_index",
        "coord_point_id",
        "grid_x",
        "grid_y",
        "easting_m",
        "northing_m",
        "longitude",
        "latitude",
    }
    assert len(points) == 9_596
    assert required == set(points.columns)
    assert points["point_index"].is_unique
    assert points["coord_point_id"].is_unique
    assert points[["longitude", "latitude"]].notna().all().all()


def test_released_temporal_split_contract() -> None:
    split = pd.read_csv(ROOT / "data/temporal_split_v1.csv")
    assert split.columns.tolist() == ["target_year_month", "split", "target_valid_n"]
    assert len(split) == 53
    assert split.groupby("split")["target_valid_n"].sum().to_dict() == {
        "test": 145_706,
        "train": 203_579,
        "validation": 89_774,
    }
def test_release_contains_no_model_weights() -> None:
    weight_suffixes = {
        ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".joblib", ".pkl", ".pickle"
    }
    bundled = [
        path.relative_to(ROOT)
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in weight_suffixes
    ]
    assert bundled == []
