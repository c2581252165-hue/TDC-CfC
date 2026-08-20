"""Model registry restricted to configurations reported in the paper."""

from __future__ import annotations

from torch import nn

from .comparators import PAPER_COMPARATOR_BUILDERS
from .tdc_cfc import HSR_RD_TIMEMIX_V2_BUILDERS
from .paper_variants import PAPER_VARIANT_BUILDERS


MODEL_ID = "H03_HSR_RD_TIMEMIX_CFC_211K"


PAPER_MODEL_BUILDERS = {
    **HSR_RD_TIMEMIX_V2_BUILDERS,
    **PAPER_COMPARATOR_BUILDERS,
    **PAPER_VARIANT_BUILDERS,
}


def build_model(model_id: str = MODEL_ID) -> nn.Module:
    """Build an exact neural configuration used in a reported comparison."""
    try:
        return PAPER_MODEL_BUILDERS[model_id]()
    except KeyError as error:
        raise KeyError(f"Unknown paper model id: {model_id}") from error


def list_model_ids() -> list[str]:
    return sorted(PAPER_MODEL_BUILDERS)


def model_requires_s2_only(model_id: str) -> bool:
    if model_id not in PAPER_MODEL_BUILDERS:
        raise KeyError(f"Unknown paper model id: {model_id}")
    return True


__all__ = [
    "MODEL_ID",
    "PAPER_MODEL_BUILDERS",
    "build_model",
    "list_model_ids",
    "model_requires_s2_only",
]
