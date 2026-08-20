"""Ordered recurrent backbone used by TDC-CfC.

This file contains only the mixed-memory CfC/GRU trajectory and the shared
six-band readout used by the paper.  Historical search families were
deliberately excluded from the public release.
"""

from __future__ import annotations

import torch
from torch import nn

from .cfc_mmrnn import _LandsatPaperCfCCell, _MixedMemoryLSTM
from .common import BASE_STEP_FEATURE_DIM, ForecastOutput, step_features
from .model_blocks import TARGET_INDICES, history_reliability, normalized_sensor_metadata


class _TrajectoryEncoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        dropout: float,
        operator: str = "cfc",
        backbone_units: int = 128,
    ) -> None:
        super().__init__()
        if operator not in {"cfc", "gru"}:
            raise ValueError(f"Unsupported trajectory operator: {operator}")
        self.hidden_dim = hidden_dim
        self.operator = operator
        if operator == "cfc":
            self.memory = _MixedMemoryLSTM(BASE_STEP_FEATURE_DIM, hidden_dim)
            self.cfc = _LandsatPaperCfCCell(
                BASE_STEP_FEATURE_DIM,
                hidden_dim=hidden_dim,
                backbone_layers=3,
                backbone_units=backbone_units,
                dropout=dropout,
            )
            self.gru = None
        else:
            self.memory = None
            self.cfc = None
            self.gru = nn.GRU(
                BASE_STEP_FEATURE_DIM,
                hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=dropout,
            )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        sequence = step_features(batch)
        if self.operator == "gru":
            states, _ = self.gru(sequence)
            return states
        hidden = sequence.new_zeros(sequence.shape[0], self.hidden_dim)
        cell = sequence.new_zeros(sequence.shape[0], self.hidden_dim)
        elapsed = sequence.new_ones(sequence.shape[0], 1)
        states: list[torch.Tensor] = []
        for step in range(sequence.shape[1]):
            mixed_hidden, cell = self.memory(sequence[:, step], hidden, cell)
            hidden = self.cfc(sequence[:, step], mixed_hidden, elapsed)
            states.append(hidden)
        return torch.stack(states, dim=1)


class ControlledDTG(nn.Module):
    """Parameter-stable ordered backbone and shared target-band readout."""

    scientific_name = "Controlled ordered recurrent forecaster"

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        mode: str = "no_dtg",
        operator: str = "cfc",
        backbone_units: int = 128,
        target_indices: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        allowed = {"full", "no_dtg", "last_only", "pool_only", "fixed_gate", "shared_gate", "uniform_pool"}
        if mode not in allowed:
            raise ValueError(f"Unsupported readout mode: {mode}")
        self.mode = mode
        self.operator = operator
        self.trajectory = _TrajectoryEncoder(
            hidden_dim=hidden_dim,
            dropout=dropout,
            operator=operator,
            backbone_units=backbone_units,
        )
        indices = TARGET_INDICES.tolist() if target_indices is None else list(target_indices)
        if not indices:
            raise ValueError("At least one target band is required")
        self.output_dim = len(indices)
        self.base_readout = nn.Linear(hidden_dim, self.output_dim)
        self.register_buffer("sensor_metadata", normalized_sensor_metadata(), persistent=True)
        self.register_buffer("target_indices", torch.tensor(indices, dtype=torch.long), persistent=True)
        self.query = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 32))
        self.trajectory_gate = nn.Sequential(
            nn.Linear(32 + hidden_dim * 2, 64), nn.GELU(), nn.Linear(64, 1)
        )
        self.residual_head = nn.Sequential(
            nn.Linear(32 + hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )
        self.residual_scale = nn.Parameter(torch.zeros(self.output_dim))
        self.state_operator_family = "standard_cfc_cell_with_mixed_memory" if operator == "cfc" else "two_layer_gru"
        self.uses_standard_cfc_cell = operator == "cfc"

    def forward(self, batch: dict[str, torch.Tensor]) -> ForecastOutput:
        states = self.trajectory(batch)
        last = states[:, -1]
        reliability = history_reliability(batch["x_mask"], batch["x_age"])
        pool_weight = torch.ones_like(reliability) if self.mode == "uniform_pool" else reliability
        pooled = (states * pool_weight.unsqueeze(-1)).sum(dim=1) / pool_weight.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        metadata = self.sensor_metadata.index_select(0, self.target_indices)
        queries = self.query(metadata).unsqueeze(0).expand(states.shape[0], -1, -1)
        last_by_band = last.unsqueeze(1).expand(-1, self.output_dim, -1)
        pooled_by_band = pooled.unsqueeze(1).expand(-1, self.output_dim, -1)
        learned_gate = torch.sigmoid(
            self.trajectory_gate(torch.cat((queries, last_by_band, pooled_by_band), dim=-1))
        )
        if self.mode == "last_only":
            gate = torch.ones_like(learned_gate)
        elif self.mode == "pool_only":
            gate = torch.zeros_like(learned_gate)
        elif self.mode == "fixed_gate":
            gate = torch.full_like(learned_gate, 0.5)
        elif self.mode == "shared_gate":
            shared_query = queries.mean(dim=1, keepdim=True)
            shared = torch.sigmoid(
                self.trajectory_gate(
                    torch.cat((shared_query, last.unsqueeze(1), pooled.unsqueeze(1)), dim=-1)
                )
            )
            gate = shared.expand(-1, self.output_dim, -1)
        else:
            gate = learned_gate
        fused = gate * last_by_band + (1.0 - gate) * pooled_by_band
        residual = self.residual_head(torch.cat((queries, fused), dim=-1)).squeeze(-1)
        base = self.base_readout(last)
        scale = torch.tanh(self.residual_scale).unsqueeze(0)
        prediction = base if self.mode == "no_dtg" else base + scale * residual
        return ForecastOutput(
            prediction,
            diagnostics={
                "base_prediction": base,
                "trajectory_reliability": reliability,
                "trajectory_pool_weight": pool_weight,
                "trajectory_gate": gate.squeeze(-1),
                "learned_trajectory_gate": learned_gate.squeeze(-1),
                "residual_prediction": residual,
                "residual_scale": scale.expand_as(residual),
            },
        )


__all__ = ["ControlledDTG"]
