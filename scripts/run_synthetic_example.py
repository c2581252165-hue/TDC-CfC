"""Run TDC-CfC on a fully synthetic, checkpoint-free input batch.

This smoke example verifies the released 12-to-1 input contract and model
forward path. It does not reproduce trained predictions or paper metrics.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpmf.interventions import recompute_age  # noqa: E402
from fpmf.models.factory import build_model  # noqa: E402


TARGET_BANDS = ("B2", "B3", "B4", "B8", "B11", "B12")
REFERENCE_TOLERANCE = 2.0e-5


def load_synthetic_batch(path: Path) -> tuple[dict[str, torch.Tensor], np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"sample_index", "x_value", "x_mask", "x_age", "x_cal"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"Synthetic input is missing keys: {sorted(missing)}")
        sample_index = payload["sample_index"].astype(np.int64, copy=False)
        arrays = {
            key: payload[key].astype(np.float32, copy=False)
            for key in ("x_value", "x_mask", "x_age", "x_cal")
        }

    size = sample_index.shape[0]
    expected_shapes = {
        "x_value": (size, 12, 10),
        "x_mask": (size, 12),
        "x_age": (size, 12),
        "x_cal": (size, 12, 2),
    }
    for key, expected in expected_shapes.items():
        if arrays[key].shape != expected:
            raise ValueError(f"{key} has shape {arrays[key].shape}; expected {expected}")
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"{key} contains a non-finite value")

    mask = arrays["x_mask"]
    if not np.isin(mask, (0.0, 1.0)).all():
        raise ValueError("x_mask must contain only 0 and 1")
    if not np.all(arrays["x_value"] * (1.0 - mask[..., None]) == 0.0):
        raise ValueError("Missing history positions must be zero in x_value")

    tensors = {key: torch.from_numpy(value.copy()) for key, value in arrays.items()}
    if not torch.allclose(tensors["x_age"], recompute_age(tensors["x_mask"]), atol=1e-7):
        raise ValueError("x_age is inconsistent with the causal mask contract")
    return tensors, sample_index


def build_reference_model() -> torch.nn.Module:
    """Create a deterministic untrained state without loading a checkpoint."""
    model = build_model().cpu().eval()
    with torch.no_grad():
        for parameter_index, parameter in enumerate(model.parameters()):
            flat = torch.arange(parameter.numel(), dtype=torch.float32)
            values = 0.02 * torch.sin(flat * 0.013 + float(parameter_index))
            parameter.copy_(values.reshape_as(parameter).to(parameter.dtype))
    return model


def predict(path: Path) -> tuple[np.ndarray, np.ndarray]:
    torch.set_num_threads(1)
    batch, sample_index = load_synthetic_batch(path)
    model = build_reference_model()
    with torch.no_grad():
        prediction = model(batch).prediction.cpu().numpy()
    if prediction.shape != (sample_index.shape[0], len(TARGET_BANDS)):
        raise RuntimeError(f"Unexpected prediction shape: {prediction.shape}")
    if not np.isfinite(prediction).all():
        raise RuntimeError("Synthetic forward pass produced a non-finite prediction")
    return sample_index, prediction


def read_expected(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    sample_index = np.asarray([int(row["sample_index"]) for row in rows], dtype=np.int64)
    prediction = np.asarray(
        [[float(row[f"standardized_{band}"]) for band in TARGET_BANDS] for row in rows],
        dtype=np.float32,
    )
    return sample_index, prediction


def write_predictions(path: Path, sample_index: np.ndarray, prediction: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", *(f"standardized_{band}" for band in TARGET_BANDS)])
        for index, row in zip(sample_index.tolist(), prediction.tolist(), strict=True):
            writer.writerow([index, *(f"{value:.9f}" for value in row)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "examples" / "synthetic_input.npz",
        help="Synthetic NPZ following the released 12-to-1 input contract.",
    )
    parser.add_argument(
        "--expected",
        type=Path,
        default=ROOT / "examples" / "expected_untrained_predictions.csv",
        help="Committed checkpoint-free reference output.",
    )
    parser.add_argument("--output", type=Path, help="Optional path for generated CSV output.")
    args = parser.parse_args()

    sample_index, prediction = predict(args.input)
    expected_index, expected_prediction = read_expected(args.expected)
    if not np.array_equal(sample_index, expected_index):
        raise RuntimeError("Synthetic sample indices differ from the reference output")
    maximum_error = float(np.max(np.abs(prediction - expected_prediction)))
    if maximum_error > REFERENCE_TOLERANCE:
        raise RuntimeError(
            f"Synthetic reference mismatch: max absolute error {maximum_error:.3e} "
            f"> {REFERENCE_TOLERANCE:.3e}"
        )
    if args.output is not None:
        write_predictions(args.output, sample_index, prediction)

    print("PASS: checkpoint-free synthetic TDC-CfC forward example")
    print(f"samples={len(sample_index)}, output_shape={prediction.shape}, max_error={maximum_error:.3e}")
    print("This smoke example is synthetic and is not a paper-performance result.")


if __name__ == "__main__":
    main()
