from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpmf.models.factory import MODEL_ID, build_model  # noqa: E402
from fpmf.preprocessing import FoldPreprocessor  # noqa: E402


BANDS = ("B2", "B3", "B4", "B8", "B11", "B12")
def main() -> None:
    parser = argparse.ArgumentParser(description="Run trained TDC-CfC models on an NPZ batch")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="NPZ file containing x_value, x_mask, x_age, and x_cal",
    )
    parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more training/refit directories containing a checkpoint and preprocessor.json",
    )
    parser.add_argument("--expected", type=Path, default=None)
    parser.add_argument("--tolerance", type=float, default=2.0e-6)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/example_predictions.csv")
    args = parser.parse_args()

    source = np.load(args.input)
    batch = {
        name: torch.from_numpy(source[name])
        for name in ("x_value", "x_mask", "x_age", "x_cal")
    }
    predictions = []
    for run_dir in args.run_dirs:
        checkpoint_path = (
            run_dir / "final_refit.pt"
            if (run_dir / "final_refit.pt").is_file()
            else run_dir / "best.pt"
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if checkpoint["model_id"] != MODEL_ID:
            raise RuntimeError(f"Unexpected model identity: {checkpoint['model_id']}")
        model = build_model().eval()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        if sum(parameter.numel() for parameter in model.parameters()) != 211_326:
            raise RuntimeError("Model parameter count differs from the released contract")
        with torch.inference_mode():
            standardized = model(batch).prediction.cpu().numpy()
        preprocessor = FoldPreprocessor.load(run_dir / "preprocessor.json")
        predictions.append(preprocessor.inverse_y(standardized))

    ensemble = np.mean(predictions, axis=0)
    if ensemble.ndim != 2 or ensemble.shape[1] != 6 or not np.isfinite(ensemble).all():
        raise RuntimeError("Prediction output is invalid")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample_index = source["sample_index"] if "sample_index" in source else np.arange(len(ensemble))
    output = pd.DataFrame({"sample_index": sample_index})
    for index, band in enumerate(BANDS):
        output[f"predicted_{band}"] = ensemble[:, index]
    output.to_csv(args.output, index=False, float_format="%.8f")
    if args.expected is not None:
        expected = pd.read_csv(args.expected)
        expected_array = expected[[f"predicted_{band}" for band in BANDS]].to_numpy()
        difference = np.abs(ensemble - expected_array)
        maximum = float(difference.max())
        if difference.shape != ensemble.shape or not np.isfinite(difference).all():
            raise RuntimeError("Prediction comparison is invalid")
        if maximum > args.tolerance:
            raise RuntimeError(
                f"Inference mismatch: max absolute difference {maximum:.3e} exceeds "
                f"{args.tolerance:.3e}"
            )
        print(f"PASS: max_abs_diff={maximum:.3e}")
    print(f"WROTE: {len(output)} examples from {len(args.run_dirs)} model run(s) to {args.output}")


if __name__ == "__main__":
    main()
