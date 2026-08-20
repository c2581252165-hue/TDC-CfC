from __future__ import annotations

import torch

from .tdc_cfc_base import HSRRDTimeMixCfC, _copy_batch
from .common import ForecastOutput


class IdentifiableHSRRDTimeMixCfC(HSRRDTimeMixCfC):
    """HSR/RD v2 with independently identifiable residual route injection."""

    scientific_name = "Identifiable Hierarchically Shrunk Reliability-Deviation TimeMix CfC-mmRNN"

    def _route_components(
        self,
        batch: dict[str, torch.Tensor],
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, steps, bands = value.shape
        global_raw = self.global_route_delta.view(1, 1, 1, 2).expand(
            batch_size, steps, bands, -1
        )
        context, reliability, coverage = self._route_context(batch, value)
        hsr_raw = torch.zeros_like(global_raw)
        if self.hsr_enabled:
            grouped = []
            for group in range(3):
                selector = self.spectral_groups == group
                grouped.append(context[:, :, selector].mean(dim=2))
            group_raw = self.group_route(torch.stack(grouped, dim=2)).index_select(
                2, self.spectral_groups
            )
            wavelength_raw = self.wavelength_route(context)
            if self.inference_intervention == "group_zero":
                group_raw = torch.zeros_like(group_raw)
            if self.inference_intervention == "wavelength_zero":
                wavelength_raw = torch.zeros_like(wavelength_raw)
            hsr_raw = reliability[:, :, None, None] * (
                torch.tanh(self.group_shrink_raw) * group_raw
                + torch.tanh(self.wavelength_shrink_raw) * wavelength_raw
            )
            if self.inference_intervention == "hsr_zero":
                hsr_raw = torch.zeros_like(hsr_raw)
        hsr_delta = 0.5 * torch.tanh(hsr_raw)
        if self.inference_intervention == "route_roll":
            hsr_delta = torch.roll(hsr_delta, shifts=1, dims=2)
        return (
            0.5 * torch.tanh(global_raw),
            hsr_delta,
            reliability,
            coverage,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> ForecastOutput:
        self._validate_batch(batch)
        mask = batch["x_mask"]
        value = batch["x_value"] * mask.unsqueeze(-1)
        scales = self._causal_scales(value)
        global_delta, hsr_delta, reliability, coverage = self._route_components(
            batch, value
        )
        k1, k3, k5 = scales.unbind(dim=2)
        global_mixed = (
            k3
            + global_delta[..., 0] * (k1 - k3)
            + global_delta[..., 1] * (k5 - k3)
        )
        hsr_correction = (
            hsr_delta[..., 0] * (k1 - k3)
            + hsr_delta[..., 1] * (k5 - k3)
        )
        enhanced = value + torch.tanh(self.time_scale) * global_mixed + hsr_correction
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
                "time_mix_delta": global_mixed,
                "time_mix_scale": torch.tanh(self.time_scale).expand(value.shape[0], 1),
                "global_route_delta": global_delta,
                "hsr_route_delta": hsr_delta,
                "route_delta": global_delta + hsr_delta,
                "hsr_input_correction": hsr_correction,
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


def _build(hsr: bool, rd: bool) -> IdentifiableHSRRDTimeMixCfC:
    return IdentifiableHSRRDTimeMixCfC(
        hsr_enabled=hsr,
        rd_enabled=rd,
        backbone_units=164,
        hidden_dim=100,
        route_width=8,
        deviation_width=8,
    )


HSR_RD_TIMEMIX_V2_BUILDERS = {
    "H00_UNIFORM_TIMEMIX_CFC_211K": lambda: _build(False, False),
    "H01_HSR_TIMEMIX_CFC_211K": lambda: _build(True, False),
    "H02_RD_TIMEMIX_CFC_211K": lambda: _build(False, True),
    "H03_HSR_RD_TIMEMIX_CFC_211K": lambda: _build(True, True),
}


__all__ = ["HSR_RD_TIMEMIX_V2_BUILDERS", "IdentifiableHSRRDTimeMixCfC"]
