from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpmf.classical import SimpleBaselineSuite  # noqa: E402
from fpmf.data import TrainingStore  # noqa: E402
from fpmf.evaluation import evaluate_predictions  # noqa: E402
from fpmf.interventions import tdc_parameter_intervention  # noqa: E402
from fpmf.models.factory import build_model  # noqa: E402
from fpmf.preprocessing import FoldPreprocessor  # noqa: E402


@torch.inference_mode()
def predict(model, loader, preprocessor, device):
    indices, predictions = [], []
    for batch in loader:
        indices.append(batch["sample_index"].numpy())
        moved = {key: value.to(device) for key, value in batch.items() if key.startswith("x_")}
        prediction = model(moved).prediction.float().cpu().numpy()
        predictions.append(preprocessor.inverse_y(prediction))
    sample_index = np.concatenate(indices)
    predicted = np.concatenate(predictions)
    order = np.argsort(sample_index)
    return sample_index[order], predicted[order]


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen H03 RD direction and zeroing checks")
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint_name = "final_refit.pt" if (args.run_dir / "final_refit.pt").is_file() else "best.pt"
    checkpoint = torch.load(args.run_dir / checkpoint_name, map_location="cpu", weights_only=False)
    preprocessor = FoldPreprocessor.load(args.run_dir / "preprocessor.json")
    model = build_model("H03_HSR_RD_TIMEMIX_CFC_211K").to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    rows = []
    with TrainingStore(args.store) as store:
        evaluation_indices = store.indices(args.split, allow_test=args.split == "test")
        fit_indices = store.indices(("train", "validation")) if args.split == "test" else store.indices("train")
        references = SimpleBaselineSuite.fit(store, fit_indices).predict_all(store, evaluation_indices)
        observed = np.asarray(store.y[evaluation_indices], dtype=np.float32)
        metadata = store.metadata.iloc[evaluation_indices].reset_index(drop=True)
        loader = DataLoader(
            store.dataset(args.split, preprocessor, allow_test=args.split == "test"),
            batch_size=1024,
            shuffle=False,
            num_workers=0,
        )
        for scenario in ("full", "rd_zero", "rd_reverse"):
            with tdc_parameter_intervention(model, scenario):
                sample_index, prediction = predict(model, loader, preprocessor, device)
            if not np.array_equal(sample_index, evaluation_indices):
                raise RuntimeError("Prediction/sample alignment failed")
            metrics = evaluate_predictions(
                metadata,
                observed,
                prediction,
                references["B02_POINT_MONTH"],
                references["B05_POINT_MONTH_AR1"],
                model_id="H03_HSR_RD_TIMEMIX_CFC_211K",
                seed=args.seed,
                split=args.split,
                include_by_point=False,
            )
            row = metrics["macro"].iloc[0].to_dict()
            row["scenario"] = scenario
            rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
