from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpmf.models.cfc import ClosedFormCfCCell  # noqa: E402
from fpmf.models.factory import build_model  # noqa: E402


def batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(20260820)
    return {
        "x_value": torch.randn(3, 12, 10),
        "x_mask": torch.tensor([[1.0] * 12, [1.0] * 9 + [0.0] * 3, [0.0, 1.0] * 6]),
        "x_age": torch.linspace(0, 1, 12).repeat(3, 1),
        "x_cal": torch.randn(3, 12, 2),
    }


def test_tdc_cfc_shape_and_parameter_count() -> None:
    model = build_model().eval()
    with torch.no_grad():
        output = model(batch())
    assert output.prediction.shape == (3, 6)
    assert torch.isfinite(output.prediction).all()
    assert sum(parameter.numel() for parameter in model.parameters()) == 211_326


def test_prediction_does_not_read_target_fields() -> None:
    model = build_model().eval()
    common = batch()
    first = {**common, "y": torch.zeros(3, 6), "target_quality": torch.zeros(3, 4)}
    second = {
        **common,
        "y": torch.full((3, 6), 999.0),
        "target_quality": torch.full((3, 4), -999.0),
    }
    with torch.no_grad():
        first_prediction = model(first).prediction
        second_prediction = model(second).prediction
    assert torch.equal(first_prediction, second_prediction)


def test_declared_cfc_equation() -> None:
    torch.manual_seed(23)
    cell = ClosedFormCfCCell(5, 7).eval()
    inputs = torch.randn(3, 5)
    hidden = torch.randn(3, 7)
    with torch.no_grad():
        joined = cell.backbone(torch.cat([inputs, hidden], dim=-1))
        first = torch.tanh(cell.ff1(joined))
        second = torch.tanh(cell.ff2(joined))
        gate = torch.sigmoid(cell.time_a(joined) + cell.time_b(joined))
        expected = gate * first + (1.0 - gate) * second
        actual = cell(inputs, hidden, dt=1.0)
    assert torch.equal(actual, expected)
