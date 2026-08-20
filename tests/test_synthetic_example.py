from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_input_contract() -> None:
    path = ROOT / "examples" / "synthetic_input.npz"
    with np.load(path, allow_pickle=False) as payload:
        assert set(payload.files) == {"sample_index", "x_value", "x_mask", "x_age", "x_cal"}
        assert payload["x_value"].shape == (8, 12, 10)
        assert payload["x_mask"].shape == (8, 12)
        assert payload["x_age"].shape == (8, 12)
        assert payload["x_cal"].shape == (8, 12, 2)
        assert np.all(payload["x_value"] * (1.0 - payload["x_mask"][..., None]) == 0.0)


def test_synthetic_example_runs_without_checkpoint(tmp_path: Path) -> None:
    output = tmp_path / "synthetic_predictions.csv"
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_synthetic_example.py"), "--output", str(output)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "checkpoint-free synthetic TDC-CfC forward example" in completed.stdout
    assert output.is_file()
