from __future__ import annotations

import torch
from torch import nn


class ClosedFormCfCCell(nn.Module):
    """Equation-level closed-form CfC-style cell with monthly ``dt=1``."""

    def __init__(self, input_dim: int, hidden_dim: int, backbone_dim: int | None = None):
        super().__init__()
        width = backbone_dim or hidden_dim
        self.hidden_dim = hidden_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, width),
            nn.Tanh(),
        )
        self.ff1 = nn.Linear(width, hidden_dim)
        self.ff2 = nn.Linear(width, hidden_dim)
        self.time_a = nn.Linear(width, hidden_dim)
        self.time_b = nn.Linear(width, hidden_dim)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor, *, dt: float = 1.0) -> torch.Tensor:
        if float(dt) != 1.0:
            raise ValueError("Monthly CfC updates require dt=1")
        joined = self.backbone(torch.cat([x, hidden], dim=-1))
        ff1 = torch.tanh(self.ff1(joined))
        ff2 = torch.tanh(self.ff2(joined))
        gate = torch.sigmoid(self.time_a(joined) * float(dt) + self.time_b(joined))
        return gate * ff1 + (1.0 - gate) * ff2


class RecurrentBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kind: str):
        super().__init__()
        if kind not in {"gru", "cfc"}:
            raise ValueError(f"Unknown recurrent kind: {kind}")
        self.kind = kind
        self.hidden_dim = hidden_dim
        self.cell = nn.GRUCell(input_dim, hidden_dim) if kind == "gru" else ClosedFormCfCCell(input_dim, hidden_dim)

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = sequence.new_zeros(sequence.shape[0], self.hidden_dim)
        states = []
        for step in range(sequence.shape[1]):
            if self.kind == "gru":
                hidden = self.cell(sequence[:, step], hidden)
            else:
                hidden = self.cell(sequence[:, step], hidden, dt=1.0)
            states.append(hidden)
        return torch.stack(states, dim=1), hidden
