from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpmf.classical import SimpleBaselineSuite  # noqa: E402
from fpmf.data import TrainingStore  # noqa: E402
from fpmf.evaluation import evaluate_predictions  # noqa: E402
from fpmf.models.factory import build_model  # noqa: E402
from fpmf.preprocessing import FoldPreprocessor  # noqa: E402


def parse_run(value: str) -> tuple[str, int, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("run must be MODEL_ID=SEED=RUN_DIR")
    return parts[0], int(parts[1]), Path(parts[2])


@torch.inference_mode()
def predict(model, loader, preprocessor, device):
    indices, predictions = [], []
    for batch in loader:
        indices.append(batch["sample_index"].numpy())
        moved = {key: value.to(device) for key, value in batch.items() if key.startswith("x_")}
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            predictions.append(
                preprocessor.inverse_y(model(moved).prediction.float().cpu().numpy())
            )
    sample_index = np.concatenate(indices)
    values = np.concatenate(predictions)
    order = np.argsort(sample_index)
    return sample_index[order], values[order]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one model or a seed ensemble")
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    model_ids = {model_id for model_id, _, _ in args.run}
    if len(model_ids) != 1:
        raise ValueError("All ensemble runs must use the same model_id")
    model_id = next(iter(model_ids))
    device = torch.device(args.device)
    with TrainingStore(args.store) as store:
        evaluation_indices = store.indices(args.split, allow_test=args.split == "test")
        fit_indices = (
            store.indices(("train", "validation"))
            if args.split == "test"
            else store.indices("train")
        )
        references = SimpleBaselineSuite.fit(store, fit_indices).predict_all(
            store, evaluation_indices
        )
        seed_predictions = []
        for _, seed, run_dir in args.run:
            preprocessor = FoldPreprocessor.load(run_dir / "preprocessor.json")
            dataset = store.dataset(
                args.split,
                preprocessor,
                allow_test=args.split == "test",
            )
            loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)
            model = build_model(model_id).to(device)
            checkpoint_name = "final_refit.pt" if args.split == "test" else "best.pt"
            checkpoint = torch.load(run_dir / checkpoint_name, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.eval()
            sample_index, prediction = predict(model, loader, preprocessor, device)
            if not np.array_equal(sample_index, evaluation_indices):
                raise RuntimeError(f"Prediction alignment failed for seed {seed}")
            seed_predictions.append(prediction)
        ensemble = np.mean(seed_predictions, axis=0)
        observed = np.asarray(store.y[evaluation_indices], dtype=np.float32)
        metadata = store.metadata.iloc[evaluation_indices].reset_index(drop=True)
        frames = evaluate_predictions(
            metadata,
            observed,
            ensemble,
            references["B02_POINT_MONTH"],
            references["B05_POINT_MONTH_AR1"],
            model_id=f"{model_id}__ENSEMBLE",
            seed=None,
            split=args.split,
        )
        args.output.mkdir(parents=True, exist_ok=False)
        for name, frame in frames.items():
            frame.to_csv(args.output / f"metrics_{name}.csv", index=False)
        np.savez_compressed(
            args.output / "evaluation_arrays.npz",
            sample_index=evaluation_indices,
            observed=observed,
            predicted=ensemble,
            pointmonth=references["B02_POINT_MONTH"],
            strongest_simple=references["B05_POINT_MONTH_AR1"],
        )


if __name__ == "__main__":
    main()
