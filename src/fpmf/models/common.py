from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class ForecastOutput:
    prediction: torch.Tensor
    auxiliary_losses: dict[str, torch.Tensor] = field(default_factory=dict)
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)


BASE_STEP_FEATURE_DIM = 14
def step_features(
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    parts = [
        batch["x_value"],
        batch["x_mask"].unsqueeze(-1),
        batch["x_age"].unsqueeze(-1),
        batch["x_cal"],
    ]
    return torch.cat(parts, dim=-1)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def assert_parameter_ceiling(model: nn.Module, ceiling: int = 1_000_000) -> None:
    count = count_trainable_parameters(model)
    if count >= ceiling:
        raise ValueError(f"Model has {count:,} trainable parameters; ceiling is {ceiling:,}")
