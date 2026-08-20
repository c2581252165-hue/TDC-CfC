from __future__ import annotations

import torch
from torch import nn

from .common import (
    BASE_STEP_FEATURE_DIM,
    ForecastOutput,
    step_features,
)


LANDSAT_CFC_MMRNN_DOI = "10.3390/s25051622"


class _MixedMemoryLSTM(nn.Module):
    """Official CfC mixed-memory LSTM equations with forget bias fixed to one."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_kernel = nn.Parameter(torch.empty(input_dim, 4 * hidden_dim))
        self.recurrent_kernel = nn.Parameter(torch.empty(hidden_dim, 4 * hidden_dim))
        self.bias = nn.Parameter(torch.zeros(4 * hidden_dim))
        self.forget_bias = 1.0
        nn.init.xavier_uniform_(self.input_kernel)
        nn.init.orthogonal_(self.recurrent_kernel)

    def forward(
        self,
        inputs: torch.Tensor,
        output_state: torch.Tensor,
        cell_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gates = inputs @ self.input_kernel + output_state @ self.recurrent_kernel + self.bias
        candidate, input_gate, forget_gate, output_gate = gates.chunk(4, dim=-1)
        new_cell = (
            cell_state * torch.sigmoid(forget_gate + self.forget_bias)
            + torch.tanh(candidate) * torch.sigmoid(input_gate)
        )
        return torch.tanh(new_cell) * torch.sigmoid(output_gate), new_cell


class _LandsatPaperCfCCell(nn.Module):
    """CfC cell configured as the TensorFlow model reported by the Landsat paper."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        backbone_layers: int = 3,
        backbone_units: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.backbone_layers = backbone_layers
        self.backbone_units = backbone_units
        self.backbone_activation = "relu"
        self.backbone_dropout = dropout

        layers: list[nn.Module] = []
        width = input_dim + hidden_dim
        for _ in range(backbone_layers):
            layers.extend(
                (
                    nn.Linear(width, backbone_units),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
            width = backbone_units
        self.backbone = nn.Sequential(*layers)
        self.ff1 = nn.Linear(backbone_units, hidden_dim)
        self.ff2 = nn.Linear(backbone_units, hidden_dim)
        self.time_a = nn.Linear(backbone_units, hidden_dim)
        self.time_b = nn.Linear(backbone_units, hidden_dim)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _lecun_tanh(value: torch.Tensor) -> torch.Tensor:
        return 1.7159 * torch.tanh(0.666 * value)

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor,
        elapsed: torch.Tensor,
    ) -> torch.Tensor:
        features = self.backbone(torch.cat((inputs, hidden), dim=-1))
        first = self._lecun_tanh(self.ff1(features))
        second = self._lecun_tanh(self.ff2(features))
        interpolation = torch.sigmoid(-self.time_a(features) * elapsed + self.time_b(features))
        return first * (1.0 - interpolation) + interpolation * second


class LandsatMixedMemoryCfCForecaster(nn.Module):
    """Task-adapted reproduction of the Landsat CFC-mmRNN architecture.

    The architecture follows the paper's 128 recurrent units, three 128-unit
    ReLU backbone layers, 0.1 backbone dropout, and official mixed-memory
    LSTM-to-CfC recurrence. The present task remains a regular monthly 12-to-1
    forecast, so elapsed time is one month and the common paper loss,
    optimizer, epochs, and six-band readout are intentionally retained.
    """

    paper_doi = LANDSAT_CFC_MMRNN_DOI
    paper_framework = "TensorFlow 2.15 architecture reproduced in PyTorch"
    temporal_adaptation = "regular_monthly_dt_1"
    scientific_name = "Task-adapted Landsat CfC-mmRNN forecaster"
    state_operator_family = "standard_cfc_cell_with_mixed_memory"
    uses_standard_cfc_cell = True

    def __init__(
        self,
        hidden_dim: int = 128,
        backbone_layers: int = 3,
        backbone_units: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        input_dim = BASE_STEP_FEATURE_DIM
        self.memory = _MixedMemoryLSTM(input_dim, hidden_dim)
        self.cfc = _LandsatPaperCfCCell(
            input_dim,
            hidden_dim=hidden_dim,
            backbone_layers=backbone_layers,
            backbone_units=backbone_units,
            dropout=dropout,
        )
        self.readout = nn.Linear(hidden_dim, 6)

    def forward(self, batch: dict[str, torch.Tensor]) -> ForecastOutput:
        sequence = step_features(batch)
        hidden = sequence.new_zeros(sequence.shape[0], self.hidden_dim)
        cell = sequence.new_zeros(sequence.shape[0], self.hidden_dim)
        elapsed = sequence.new_ones(sequence.shape[0], 1)
        for step in range(sequence.shape[1]):
            mixed_hidden, cell = self.memory(sequence[:, step], hidden, cell)
            hidden = self.cfc(sequence[:, step], mixed_hidden, elapsed)
        return ForecastOutput(self.readout(hidden))
