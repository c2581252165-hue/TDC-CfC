"""Create the deterministic synthetic input and checkpoint-free reference output."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fpmf.interventions import recompute_age  # noqa: E402
from run_synthetic_example import predict, write_predictions  # noqa: E402


def main() -> None:
    output_dir = ROOT / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_count = 8
    time = np.arange(12, dtype=np.float32)[None, :, None]
    band = np.arange(10, dtype=np.float32)[None, None, :]
    sample = np.arange(sample_count, dtype=np.float32)[:, None, None]
    x_value = (
        0.45 * np.sin(2.0 * np.pi * (time + sample) / 12.0)
        + 0.12 * np.cos((band + 1.0) * 0.43)
        + 0.015 * sample
        + 0.01 * time * (band + 1.0) / 10.0
    ).astype(np.float32)

    x_mask = np.ones((sample_count, 12), dtype=np.float32)
    missing_positions = {
        1: (11,),
        2: (9, 10, 11),
        3: (0, 4, 8),
        4: (1, 3, 5, 7, 9),
        5: (2, 3, 6, 10),
        6: (0, 1, 2, 11),
        7: (5, 6, 7),
    }
    for row, positions in missing_positions.items():
        x_mask[row, list(positions)] = 0.0
    x_value *= x_mask[..., None]
    x_age = recompute_age(torch.from_numpy(x_mask)).numpy().astype(np.float32)

    target_month = np.arange(sample_count, dtype=np.int64) % 12
    history_month = (target_month[:, None] + np.arange(12, dtype=np.int64)[None, :]) % 12
    angle = 2.0 * np.pi * history_month.astype(np.float32) / 12.0
    x_cal = np.stack((np.sin(angle), np.cos(angle)), axis=-1).astype(np.float32)
    sample_index = np.arange(1000, 1000 + sample_count, dtype=np.int64)

    input_path = output_dir / "synthetic_input.npz"
    np.savez_compressed(
        input_path,
        sample_index=sample_index,
        x_value=x_value,
        x_mask=x_mask,
        x_age=x_age,
        x_cal=x_cal,
    )
    predicted_index, prediction = predict(input_path)
    write_predictions(
        output_dir / "expected_untrained_predictions.csv",
        predicted_index,
        prediction,
    )
    print(f"Wrote {input_path}")


if __name__ == "__main__":
    main()
