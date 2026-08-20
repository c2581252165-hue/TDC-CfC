"""Generic model classes and capacity controls reported in the paper."""

from __future__ import annotations

import math

import torch
from torch import nn

from .cfc import RecurrentBackbone
from .cfc_mmrnn import LandsatMixedMemoryCfCForecaster
from .common import BASE_STEP_FEATURE_DIM, ForecastOutput, MLP, step_features


def _next_month_calendar(calendar: torch.Tensor) -> torch.Tensor:
    delta = math.pi / 6.0
    last_sin, last_cos = calendar[:, -1, 0], calendar[:, -1, 1]
    return torch.stack(
        (
            last_sin * math.cos(delta) + last_cos * math.sin(delta),
            last_cos * math.cos(delta) - last_sin * math.sin(delta),
        ),
        dim=-1,
    )


class PlainGRUForecaster(nn.Module):
    """The 19,910-parameter GRU used in cross-model information analyses."""

    scientific_name = "Plain GRU forecaster"

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = RecurrentBackbone(BASE_STEP_FEATURE_DIM, hidden_dim, "gru")
        self.head = MLP(hidden_dim, hidden_dim, 6, dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> ForecastOutput:
        _, hidden = self.backbone(step_features(batch))
        return ForecastOutput(self.head(hidden))


class TemporalTransformerForecaster(nn.Module):
    """Causal SITS Transformer used in the capacity-matched comparison."""

    scientific_name = "SITS causal temporal Transformer forecaster"

    def __init__(
        self,
        d_model: int,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.history_projection = nn.Linear(BASE_STEP_FEATURE_DIM, d_model)
        self.target_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.target_calendar_projection = nn.Linear(2, d_model, bias=False)
        position = torch.arange(13, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        encoding = torch.zeros(13, d_model)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("position_encoding", encoding.unsqueeze(0), persistent=False)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = MLP(d_model, d_model, 6, dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> ForecastOutput:
        history = self.history_projection(step_features(batch))
        target = self.target_token.expand(history.shape[0], -1, -1)
        target = target + self.target_calendar_projection(
            _next_month_calendar(batch["x_cal"])
        ).unsqueeze(1)
        tokens = torch.cat((history, target), dim=1)
        tokens = tokens + self.position_encoding[:, : tokens.shape[1]].to(tokens.dtype)
        causal_mask = torch.triu(
            torch.ones(
                tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=tokens.device
            ),
            diagonal=1,
        )
        encoded = self.encoder(tokens, mask=causal_mask)
        return ForecastOutput(self.head(encoded[:, -1]))


PAPER_COMPARATOR_BUILDERS = {
    "B08_GRU_COMMON": PlainGRUForecaster,
    "C20_SITS_CAPACITY_219K": lambda: TemporalTransformerForecaster(d_model=76),
    "C21_SITS_CAPACITY_242K": lambda: TemporalTransformerForecaster(d_model=80),
    "C22_CFC_MMRNN_CAPACITY_211K": lambda: LandsatMixedMemoryCfCForecaster(
        hidden_dim=128, backbone_layers=3, backbone_units=144, dropout=0.1
    ),
    "C23_CFC_MMRNN_CAPACITY_247K": lambda: LandsatMixedMemoryCfCForecaster(
        hidden_dim=128, backbone_layers=3, backbone_units=172, dropout=0.1
    ),
}


__all__ = [
    "PAPER_COMPARATOR_BUILDERS",
    "PlainGRUForecaster",
    "TemporalTransformerForecaster",
]
