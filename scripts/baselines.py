from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpmf.classical import SimpleBaselineSuite  # noqa: E402
from fpmf.data import TrainingStore  # noqa: E402
from fpmf.evaluation import evaluate_predictions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate interpretable skill controls")
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    with TrainingStore(args.store) as store:
        train_indices = (
            store.indices(("train", "validation"))
            if args.split == "test"
            else store.indices("train")
        )
        eval_indices = store.indices(args.split, allow_test=args.split == "test")
        suite = SimpleBaselineSuite.fit(store, train_indices)
        predictions = suite.predict_all(store, eval_indices)
        observed = np.asarray(store.y[eval_indices], dtype=np.float32)
        metadata = store.metadata.iloc[eval_indices].reset_index(drop=True)
        pointmonth = predictions["B02_POINT_MONTH"]
        ar1 = predictions["B05_POINT_MONTH_AR1"]
        for model_id, predicted in predictions.items():
            frames = evaluate_predictions(
                metadata,
                observed,
                predicted,
                pointmonth,
                ar1,
                model_id=model_id,
                seed=None,
                split=args.split,
            )
            model_dir = args.output / model_id
            model_dir.mkdir()
            for name, frame in frames.items():
                frame.to_csv(model_dir / f"metrics_{name}.csv", index=False)


if __name__ == "__main__":
    main()
