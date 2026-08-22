from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .model_blocks import (
    history_reliability,
    next_month_calendar,
    normalized_sensor_metadata,
)
from .common import ForecastOutput
from .paper_backbone import ControlledDTG


INPUT_SPECTRAL_GROUPS = (0, 0, 0, 1, 1, 1, 1, 1, 2, 2)
REFERENCE_WINDOWS = (1, 3, 6, 12)


def _copy_batch(
    batch: dict[str, torch.Tensor], **updates: torch.Tensor
) -> dict[str, torch.Tensor]:
    copied = dict(batch)
    copied.update(updates)
    return copied


def _small_nonzero_raw_scale(value: float = 0.1) -> torch.Tensor:
    if not 0.0 < value < 1.0:
        raise ValueError("shrink initialization must be inside (0, 1)")
    return torch.tensor(math.atanh(value), dtype=torch.float32)


class HSRRDTimeMixCfC(nn.Module):
    """K3-anchored TimeMix with optional shrunk routing and causal deviation.

    HSR adds group- and wavelength-conditioned residual routes around a global
    route. RD injects a deterministic, strictly causal deviation correction in
    observation space before the recurrent update. The correction is expressed
    as a deviation from the causal reference.

    Added branch output layers are zero initialized while their outer shrink
    magnitudes start small and non-zero, preserving the H00 initialization
    while keeping residual branches trainable.
    """

    scientific_name = "Hierarchically Shrunk Reliability-Deviation TimeMix CfC-mmRNN"
    state_operator_family = "standard_cfc_cell_with_mixed_memory_and_causal_timemix"
    uses_standard_cfc_cell = True
    auxiliary_loss_contract = "none"
    fixed_step_continuous_time_claim = False

    def __init__(
        self,
        *,
        hsr_enabled: bool,
        rd_enabled: bool,
        backbone_units: int,
        hidden_dim: int = 100,
        route_width: int = 16,
        deviation_width: int = 32,
        dropout: float = 0.1,
        target_indices: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if min(backbone_units, hidden_dim, route_width, deviation_width) <= 0:
            raise ValueError("all HSR/RD widths must be positive")
        self.hsr_enabled = bool(hsr_enabled)
        self.rd_enabled = bool(rd_enabled)
        # Post-training intervention selector; it is not a model parameter or buffer.
        self.inference_intervention = "none"

        # Common modules are constructed before optional branches. Under the
        # same seed, all four cells therefore share the same initial anchor.
        self.mix = nn.ModuleList(
            nn.Conv1d(10, 10, kernel_size=kernel, groups=10)
            for kernel in (1, 3, 5)
        )
        self.time_scale = nn.Parameter(torch.zeros(1))
        self.global_route_delta = nn.Parameter(torch.zeros(2))
        self.register_buffer(
            "sensor_metadata", normalized_sensor_metadata(), persistent=True
        )
        self.register_buffer(
            "spectral_groups",
            torch.tensor(INPUT_SPECTRAL_GROUPS, dtype=torch.long),
            persistent=True,
        )
        self.base = ControlledDTG(
            hidden_dim=hidden_dim,
            dropout=dropout,
            mode="no_dtg",
            operator="cfc",
            backbone_units=backbone_units,
            target_indices=target_indices,
        )

        if self.hsr_enabled:
            self.group_route = self._zero_output_mlp(7, route_width, 2)
            self.wavelength_route = self._zero_output_mlp(7, route_width, 2)
            self.group_shrink_raw = nn.Parameter(_small_nonzero_raw_scale())
            self.wavelength_shrink_raw = nn.Parameter(_small_nonzero_raw_scale())
        else:
            self.group_route = None
            self.wavelength_route = None
            self.register_parameter("group_shrink_raw", None)
            self.register_parameter("wavelength_shrink_raw", None)

        if self.rd_enabled:
            self.deviation_mixer = self._zero_output_mlp(
                2 * len(REFERENCE_WINDOWS) * 10,
                deviation_width,
                10,
            )
            self.deviation_shrink_raw = nn.Parameter(_small_nonzero_raw_scale())
        else:
            self.deviation_mixer = None
            self.register_parameter("deviation_shrink_raw", None)

    @staticmethod
    def _zero_output_mlp(
        input_width: int, hidden_width: int, output_width: int
    ) -> nn.Sequential:
        network = nn.Sequential(
            nn.Linear(input_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, output_width),
        )
        nn.init.zeros_(network[-1].weight)
        nn.init.zeros_(network[-1].bias)
        return network

    @staticmethod
    def _validate_batch(batch: dict[str, torch.Tensor]) -> None:
        values = batch["x_value"]
        mask = batch["x_mask"]
        age = batch["x_age"]
        calendar = batch["x_cal"]
        if values.ndim != 3 or values.shape[-1] != 10:
            raise ValueError("HSR/RD models expect x_value [batch,time,10]")
        if mask.shape != values.shape[:2] or age.shape != values.shape[:2]:
            raise ValueError("HSR/RD mask/age axes do not match x_value")
        if calendar.shape != (*values.shape[:2], 2):
            raise ValueError("HSR/RD models expect x_cal [batch,time,2]")

    def _causal_scales(self, value: torch.Tensor) -> torch.Tensor:
        sequence = value.transpose(1, 2)
        scales = []
        for layer in self.mix:
            left = layer.kernel_size[0] - 1
            scales.append(layer(F.pad(sequence, (left, 0))).transpose(1, 2))
        return torch.stack(scales, dim=2)

    def _route_context(
        self,
        batch: dict[str, torch.Tensor],
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reliability = history_reliability(batch["x_mask"], batch["x_age"])
        coverage = reliability.mean(dim=1, keepdim=True)
        future_calendar = next_month_calendar(batch["x_cal"])
        batch_size, steps, _ = value.shape
        calendar = future_calendar[:, None, None, :].expand(-1, steps, 10, -1)
        rel = reliability[:, :, None, None].expand(-1, -1, 10, -1)
        cov = coverage[:, None, None, :].expand(-1, steps, 10, -1)
        metadata = self.sensor_metadata[None, None, :, :].expand(
            batch_size, steps, -1, -1
        )
        magnitude = value.abs().unsqueeze(-1)
        return (
            torch.cat((metadata, calendar, rel, cov, magnitude), dim=-1),
            reliability,
            coverage,
        )

    def _route_delta(
        self,
        batch: dict[str, torch.Tensor],
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, steps, bands = value.shape
        raw = self.global_route_delta.view(1, 1, 1, 2).expand(
            batch_size, steps, bands, -1
        )
        context, reliability, coverage = self._route_context(batch, value)
        if self.hsr_enabled:
            grouped = []
            for group in range(3):
                selector = self.spectral_groups == group
                grouped.append(context[:, :, selector].mean(dim=2))
            group_raw = self.group_route(torch.stack(grouped, dim=2)).index_select(
                2, self.spectral_groups
            )
            wavelength_raw = self.wavelength_route(context)
            rel = reliability[:, :, None, None]
            raw = raw + rel * (
                torch.tanh(self.group_shrink_raw) * group_raw
                + torch.tanh(self.wavelength_shrink_raw) * wavelength_raw
            )
        return 0.5 * torch.tanh(raw), reliability, coverage

    @staticmethod
    def causal_deviations(
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return deviations from strictly previous-window deterministic means."""

        deviations = []
        availability = []
        for step in range(value.shape[1]):
            step_deviations = []
            step_availability = []
            for window in REFERENCE_WINDOWS:
                start = max(0, step - window)
                prior_mask = mask[:, start:step]
                denominator = prior_mask.sum(dim=1, keepdim=True)
                reference = (
                    value[:, start:step] * prior_mask.unsqueeze(-1)
                ).sum(dim=1) / denominator.clamp_min(1.0)
                available = denominator > 0
                current_valid = mask[:, step : step + 1] > 0
                valid = available & current_valid
                step_deviations.append((value[:, step] - reference) * valid)
                step_availability.append(valid.squeeze(1))
            deviations.append(torch.stack(step_deviations, dim=1))
            availability.append(torch.stack(step_availability, dim=1))
        return torch.stack(deviations, dim=1), torch.stack(availability, dim=1)

    def _deviation_injection(
        self,
        value: torch.Tensor,
        mask: torch.Tensor,
        reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        deviations, availability = self.causal_deviations(value, mask)
        if not self.rd_enabled:
            return torch.zeros_like(value), deviations, availability
        if self.inference_intervention == "deviation_sign_flip":
            deviations = -deviations
        features = torch.cat((deviations, deviations.abs()), dim=2).flatten(2)
        correction = self.deviation_mixer(features)
        gate = reliability.unsqueeze(-1) * mask.unsqueeze(-1)
        correction = torch.tanh(self.deviation_shrink_raw) * gate * correction
        if self.inference_intervention == "rd_zero":
            correction = torch.zeros_like(correction)
        return correction, deviations, availability

    def forward(self, batch: dict[str, torch.Tensor]) -> ForecastOutput:
        self._validate_batch(batch)
        mask = batch["x_mask"]
        value = batch["x_value"] * mask.unsqueeze(-1)
        scales = self._causal_scales(value)
        route_delta, reliability, coverage = self._route_delta(batch, value)
        k1, k3, k5 = scales.unbind(dim=2)
        mixed = (
            k3
            + route_delta[..., 0] * (k1 - k3)
            + route_delta[..., 1] * (k5 - k3)
        )
        enhanced = value + torch.tanh(self.time_scale) * mixed
        deviation_correction, deviations, availability = self._deviation_injection(
            value, mask, reliability
        )
        enhanced = enhanced + deviation_correction
        output = self.base(_copy_batch(batch, x_value=enhanced))
        diagnostics = dict(output.diagnostics)
        diagnostics.update(
            {
                "time_mix_scales": scales,
                "time_mix_anchor": k3,
                "time_mix_delta": mixed,
                "time_mix_scale": torch.tanh(self.time_scale).expand(value.shape[0], 1),
                "route_delta": route_delta,
                "history_reliability": reliability,
                "history_coverage": coverage,
                "causal_deviation_history": deviations,
                "causal_deviation_available": availability.to(value.dtype),
                "deviation_correction": deviation_correction,
            }
        )
        return ForecastOutput(
            output.prediction,
            auxiliary_losses={},
            diagnostics=diagnostics,
        )


def _build(
    hsr_enabled: bool,
    rd_enabled: bool,
    *,
    backbone_units: int,
) -> HSRRDTimeMixCfC:
    return HSRRDTimeMixCfC(
        hsr_enabled=hsr_enabled,
        rd_enabled=rd_enabled,
        backbone_units=backbone_units,
    )


# Backbone widths reproduce the parameter-matched models reported in the paper.
HSR_RD_TIMEMIX_BUILDERS = {
    "H00_UNIFORM_TIMEMIX_CFC_211K": lambda: _build(False, False, backbone_units=164),
    "H01_HSR_TIMEMIX_CFC_211K": lambda: _build(True, False, backbone_units=162),
    "H02_RD_TIMEMIX_CFC_211K": lambda: _build(False, True, backbone_units=151),
    "H03_HSR_RD_TIMEMIX_CFC_211K": lambda: _build(True, True, backbone_units=149),
}


__all__ = [
    "HSR_RD_TIMEMIX_BUILDERS",
    "HSRRDTimeMixCfC",
    "INPUT_SPECTRAL_GROUPS",
    "REFERENCE_WINDOWS",
]
