from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpmf.classical import SklearnBaseline, flattened_features  # noqa: E402
from fpmf.data import TrainingStore  # noqa: E402
from fpmf.preprocessing import FoldPreprocessor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the paper HistGradientBoosting control")
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=438344685)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with TrainingStore(args.store) as store:
        train_indices = store.indices("train")
        preprocessor = FoldPreprocessor.fit(store, train_indices, fitted_split="train")
        features = flattened_features(store, train_indices, preprocessor)
        targets = preprocessor.transform_y(np.asarray(store.y[train_indices], dtype=np.float32))
        model = SklearnBaseline("hgb", args.seed).fit(features, targets)
        preprocessor.save(args.output / "preprocessor.json")
        joblib.dump(model, args.output / "hgb.joblib")


if __name__ == "__main__":
    main()
