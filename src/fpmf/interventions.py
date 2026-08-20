"""Controlled history interventions used in the paper."""

from __future__ import annotations

import numpy as np
import torch
from contextlib import contextmanager


SHUFFLE_SEEDS = (20260727, 20260728, 20260729, 20260730, 20260731)
TDC_FIXED_PERMUTATION = (7, 0, 10, 3, 5, 1, 11, 8, 2, 9, 4, 6)
MISSINGNESS_RATES = (10, 20, 30, 40, 50)
RECENT_HORIZONS = (1, 2, 3, 4, 5)
CROSS_MODEL_STRESS_HASH_SEED = 20260816
TDC_STRESS_HASH_SEED = 20260727


def fixed_permutations() -> list[list[int]]:
    orders = [np.random.default_rng(seed).permutation(12).tolist() for seed in SHUFFLE_SEEDS]
    if len({tuple(order) for order in orders}) != len(orders):
        raise RuntimeError("Fixed temporal permutations are not unique")
    if any(order == list(range(12)) for order in orders):
        raise RuntimeError("A fixed temporal permutation is the identity")
    return orders


def tdc_fixed_permutation() -> list[int]:
    """Return the single fixed month order used by the final TDC-CfC check."""
    order = list(TDC_FIXED_PERMUTATION)
    if sorted(order) != list(range(12)) or order == list(range(12)):
        raise RuntimeError("Invalid TDC-CfC fixed temporal permutation")
    return order


def recompute_age(mask: torch.Tensor) -> torch.Tensor:
    """Recompute window-local observation age, normalized by 12 months."""
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


def _deterministic_uniform(
    sample_index: torch.Tensor,
    steps: int,
    *,
    stress_seed: int,
) -> torch.Tensor:
    sample = sample_index.to(torch.int64).view(-1, 1)
    time = torch.arange(steps, dtype=torch.int64, device=sample.device).view(1, -1)
    hashed = (sample * 1103515245 + time * 12345 + int(stress_seed)) & 0x7FFFFFFF
    return hashed.to(torch.float32) / float(0x7FFFFFFF)


def apply_history_intervention(
    batch: dict[str, torch.Tensor],
    scenario: str,
    *,
    permutation: list[int] | None = None,
    stress_seed: int = CROSS_MODEL_STRESS_HASH_SEED,
) -> dict[str, torch.Tensor]:
    """Apply a paper intervention without changing the target or future data."""
    if scenario in {"natural", "D0_original", "A0_natural"}:
        return dict(batch)
    output = dict(batch)
    value = batch["x_value"].clone()
    mask = batch["x_mask"].clone()
    calendar = batch["x_cal"].clone()
    if scenario in {"previous_year_removed", "D1_prev_year_removed"}:
        mask[:, 0] = 0.0
    elif scenario.startswith("recent_"):
        count = int(scenario.rsplit("_", 1)[1])
        if count not in RECENT_HORIZONS:
            raise ValueError(f"Unsupported recent horizon: {count}")
        mask[:, -count:] = 0.0
    elif scenario.startswith("missingness_"):
        rate = int(scenario.rsplit("_", 1)[1])
        if rate not in MISSINGNESS_RATES:
            raise ValueError(f"Unsupported missingness rate: {rate}")
        keep = _deterministic_uniform(
            batch["sample_index"], mask.shape[1], stress_seed=stress_seed
        ) >= rate / 100.0
        mask = mask * keep.to(mask.dtype)
    elif scenario in {"temporal_shuffle", "D4_temporal_shuffle"}:
        if permutation is None:
            raise ValueError("Temporal shuffle requires an explicit fixed permutation")
        order = torch.as_tensor(permutation, dtype=torch.int64, device=value.device)
        value = value.index_select(1, order)
        mask = mask.index_select(1, order)
        calendar = calendar.index_select(1, order)
        output["x_age"] = batch["x_age"].index_select(1, order)
    else:
        raise ValueError(f"Unknown intervention: {scenario}")
    output["x_value"] = value * mask.unsqueeze(-1)
    output["x_mask"] = mask
    if scenario not in {"temporal_shuffle", "D4_temporal_shuffle"}:
        output["x_age"] = recompute_age(mask)
    output["x_cal"] = calendar
    return output


def assert_nested_missingness(batch: dict[str, torch.Tensor]) -> None:
    previous = None
    for rate in MISSINGNESS_RATES:
        changed = apply_history_intervention(
            batch,
            f"missingness_{rate}",
            stress_seed=CROSS_MODEL_STRESS_HASH_SEED,
        )
        if torch.any(changed["x_mask"] > batch["x_mask"]):
            raise AssertionError("Artificial missingness restored a native missing month")
        if previous is not None and torch.any(changed["x_mask"] > previous):
            raise AssertionError("Missingness masks are not nested")
        if not torch.equal(changed["x_age"], recompute_age(changed["x_mask"])):
            raise AssertionError("Observation age was not recomputed causally")
        previous = changed["x_mask"]


def apply_tdc_input_ablation(
    batch: dict[str, torch.Tensor], scenario: str
) -> dict[str, torch.Tensor]:
    """Input-side TDC-CfC verification ablations reported in the paper."""
    if scenario == "full":
        return dict(batch)
    if scenario in {"history_6", "history_9"}:
        keep = int(scenario.rsplit("_", 1)[1])
        output = dict(batch)
        mask = batch["x_mask"].clone()
        mask[:, : 12 - keep] = 0.0
        output["x_mask"] = mask
        output["x_value"] = batch["x_value"] * mask.unsqueeze(-1)
        output["x_age"] = recompute_age(mask)
        return output
    if scenario == "target_phase_off":
        output = dict(batch)
        output["x_cal"] = torch.zeros_like(batch["x_cal"])
        return output
    if scenario == "explicit_missingness_off":
        output = dict(batch)
        output["x_mask"] = torch.ones_like(batch["x_mask"])
        output["x_age"] = torch.zeros_like(batch["x_age"])
        return output
    if scenario == "target6_history_only":
        output = dict(batch)
        gate = batch["x_value"].new_zeros(10)
        gate[torch.tensor((0, 1, 2, 6, 8, 9), device=gate.device)] = 1.0
        output["x_value"] = batch["x_value"] * gate.view(1, 1, -1)
        return output
    raise ValueError(f"Unknown TDC input ablation: {scenario}")


@contextmanager
def tdc_parameter_intervention(model: torch.nn.Module, scenario: str):
    """Temporarily alter learned corrections without modifying a checkpoint."""
    if scenario == "full":
        yield model
        return
    exact_scenarios = {
        "rd_zero": "rd_zero",
        "rd_reverse": "deviation_sign_flip",
        "deviation_sign_flip": "deviation_sign_flip",
        "hsr_zero": "hsr_zero",
        "group_zero": "group_zero",
        "wavelength_zero": "wavelength_zero",
        "route_roll": "route_roll",
    }
    if scenario in exact_scenarios:
        if not hasattr(model, "inference_intervention"):
            raise ValueError("Model does not expose the paper inference intervention contract")
        original = model.inference_intervention
        model.inference_intervention = exact_scenarios[scenario]
        try:
            yield model
        finally:
            model.inference_intervention = original
        return
    changes: list[tuple[torch.Tensor, torch.Tensor]] = []

    def replace(parameter: torch.Tensor | None, value: torch.Tensor) -> None:
        if parameter is None:
            raise ValueError(f"Model does not contain a parameter required by {scenario}")
        changes.append((parameter, parameter.detach().clone()))
        parameter.data.copy_(value.to(parameter))

    try:
        if scenario == "complete_no_timemix":
            replace(model.time_scale, torch.zeros_like(model.time_scale))
            replace(model.group_shrink_raw, torch.zeros_like(model.group_shrink_raw))
            replace(model.wavelength_shrink_raw, torch.zeros_like(model.wavelength_shrink_raw))
        else:
            raise ValueError(f"Unknown TDC parameter intervention: {scenario}")
        yield model
    finally:
        for parameter, original in reversed(changes):
            parameter.data.copy_(original)


__all__ = [
    "CROSS_MODEL_STRESS_HASH_SEED",
    "MISSINGNESS_RATES",
    "RECENT_HORIZONS",
    "SHUFFLE_SEEDS",
    "TDC_FIXED_PERMUTATION",
    "TDC_STRESS_HASH_SEED",
    "apply_history_intervention",
    "apply_tdc_input_ablation",
    "assert_nested_missingness",
    "fixed_permutations",
    "tdc_fixed_permutation",
    "recompute_age",
    "tdc_parameter_intervention",
]
