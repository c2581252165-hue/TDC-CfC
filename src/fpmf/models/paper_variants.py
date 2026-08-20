"""TDC-CfC structural variants reported in the paper.

These builders preserve the released H03 implementation and change exactly
one reported representation or sharing contract.  They are training models,
not post-hoc masking aliases.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .common import ForecastOutput
from .paper_backbone import ControlledDTG
from .tdc_cfc import IdentifiableHSRRDTimeMixCfC


TARGET_HISTORY_INDICES = (0, 1, 2, 6, 8, 9)


def recompute_window_age(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 2 or mask.shape[1] != 12:
        raise ValueError("mask must have shape [batch, 12]")
    valid = mask > 0
    age = torch.zeros_like(mask, dtype=torch.float32)
    age[:, 0] = (~valid[:, 0]).to(age.dtype) / 12.0
    for step in range(1, 12):
        age[:, step] = torch.where(
            valid[:, step],
            torch.zeros_like(age[:, step]),
            (age[:, step - 1] + 1.0 / 12.0).clamp_max(1.0),
        )
    return age


@dataclass(frozen=True)
class TDCVariantSpec:
    history_months: int = 12
    target_phase: bool = True
    explicit_missingness: bool = True
    history_bands: str = "all10"
    backbone: str = "cfc"

    def validate(self) -> "TDCVariantSpec":
        if self.history_months not in {6, 9, 12}:
            raise ValueError("history_months must be 6, 9, or 12")
        if self.history_bands not in {"all10", "target6"}:
            raise ValueError("history_bands must be all10 or target6")
        if self.backbone not in {"cfc", "gru"}:
            raise ValueError("backbone must be cfc or gru")
        return self


class TDCRepresentationVariant(IdentifiableHSRRDTimeMixCfC):
    """Full H03 with one representation contract changed before training."""

    def __init__(self, spec: TDCVariantSpec) -> None:
        self.variant_spec = spec.validate()
        super().__init__(
            hsr_enabled=True,
            rd_enabled=True,
            backbone_units=164,
            hidden_dim=100,
            route_width=8,
            deviation_width=8,
        )
        if spec.backbone == "gru":
            # Exact parameter-matched TDC-GRU reported in the paper.
            self.base = ControlledDTG(
                hidden_dim=138,
                dropout=0.1,
                mode="no_dtg",
                operator="gru",
                backbone_units=164,
            )
        target_gate = torch.zeros(10)
        target_gate[list(TARGET_HISTORY_INDICES)] = 1.0
        self.register_buffer("target_history_gate", target_gate, persistent=False)

    def _prepare(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output = dict(batch)
        value = batch["x_value"].clone()
        mask = batch["x_mask"].clone()
        age = batch["x_age"].clone()
        calendar = batch["x_cal"].clone()
        if self.variant_spec.history_months < 12:
            end = 12 - self.variant_spec.history_months
            value[:, :end] = 0.0
            mask[:, :end] = 0.0
            age = recompute_window_age(mask)
        if not self.variant_spec.target_phase:
            calendar.zero_()
        if not self.variant_spec.explicit_missingness:
            mask.fill_(1.0)
            age.zero_()
        if self.variant_spec.history_bands == "target6":
            value = value * self.target_history_gate.view(1, 1, -1)
        output.update(x_value=value, x_mask=mask, x_age=age, x_cal=calendar)
        return output

    def forward(self, batch: dict[str, torch.Tensor]) -> ForecastOutput:
        return super().forward(self._prepare(batch))


class IndependentSixTDC(nn.Module):
    """Six disjoint one-band TDC paths used in the sharing comparison."""

    def __init__(self, *, width_matched: bool) -> None:
        super().__init__()
        hidden, units = ((100, 164) if width_matched else (38, 44))
        self.paths = nn.ModuleList(
            self._build_path(target_index, hidden=hidden, units=units)
            for target_index in TARGET_HISTORY_INDICES
        )

    @staticmethod
    def _build_path(
        target_index: int,
        *,
        hidden: int,
        units: int,
    ) -> IdentifiableHSRRDTimeMixCfC:
        """Reproduce the exact constructor order used by the paper runs.

        The frozen training implementation first instantiated the full H03
        path and then replaced its shared six-band recurrent/readout block by
        a target-specific block.  Constructing the narrow path directly is
        mathematically equivalent after loading a checkpoint, but consumes a
        different part of the seeded random stream and therefore does not
        reproduce training initialization.  Keeping the original two-stage
        construction makes fresh training bitwise aligned at initialization.
        """
        path = IdentifiableHSRRDTimeMixCfC(
            hsr_enabled=True,
            rd_enabled=True,
            backbone_units=164,
            hidden_dim=100,
            route_width=8,
            deviation_width=8,
        )
        path.base = ControlledDTG(
            hidden_dim=hidden,
            dropout=0.1,
            mode="no_dtg",
            operator="cfc",
            backbone_units=units,
            target_indices=(target_index,),
        )
        return path

    def forward(self, batch: dict[str, torch.Tensor]) -> ForecastOutput:
        outputs = [path(batch) for path in self.paths]
        return ForecastOutput(
            prediction=torch.cat([output.prediction for output in outputs], dim=-1),
            auxiliary_losses={},
            diagnostics={},
        )


PAPER_VARIANT_BUILDERS = {
    "TDC_HISTORY_6": lambda: TDCRepresentationVariant(TDCVariantSpec(history_months=6)),
    "TDC_HISTORY_9": lambda: TDCRepresentationVariant(TDCVariantSpec(history_months=9)),
    "TDC_TARGET_PHASE_OFF": lambda: TDCRepresentationVariant(TDCVariantSpec(target_phase=False)),
    "TDC_EXPLICIT_MISSINGNESS_OFF": lambda: TDCRepresentationVariant(
        TDCVariantSpec(explicit_missingness=False)
    ),
    "TDC_TARGET6_HISTORY": lambda: TDCRepresentationVariant(
        TDCVariantSpec(history_bands="target6")
    ),
    "TDC_GRU_212K": lambda: TDCRepresentationVariant(TDCVariantSpec(backbone="gru")),
    "TDC_INDEPENDENT6_TOTAL_MATCH": lambda: IndependentSixTDC(width_matched=False),
    "TDC_INDEPENDENT6_WIDTH_MATCH": lambda: IndependentSixTDC(width_matched=True),
}


__all__ = [
    "IndependentSixTDC",
    "PAPER_VARIANT_BUILDERS",
    "TDCRepresentationVariant",
    "TDCVariantSpec",
    "recompute_window_age",
]
