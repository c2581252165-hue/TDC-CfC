from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fpmf.bootstrap import paired_macro_rmse_bootstrap
from fpmf.interventions import (
    CROSS_MODEL_STRESS_HASH_SEED,
    SHUFFLE_SEEDS,
    TDC_FIXED_PERMUTATION,
    TDC_STRESS_HASH_SEED,
    apply_history_intervention,
    assert_nested_missingness,
    fixed_permutations,
    recompute_age,
)
from fpmf.models.factory import build_model
from fpmf.training import build_lr_scheduler


EXPECTED_PARAMETERS = {
    "B08_GRU_COMMON": 19_910,
    "C20_SITS_CAPACITY_219K": 218_582,
    "C21_SITS_CAPACITY_242K": 241_926,
    "C22_CFC_MMRNN_CAPACITY_211K": 210_582,
    "C23_CFC_MMRNN_CAPACITY_247K": 246_674,
    "H00_UNIFORM_TIMEMIX_CFC_211K": 210_421,
    "H01_HSR_TIMEMIX_CFC_211K": 210_587,
    "H02_RD_TIMEMIX_CFC_211K": 211_160,
    "H03_HSR_RD_TIMEMIX_CFC_211K": 211_326,
    "TDC_GRU_212K": 212_718,
    "TDC_INDEPENDENT6_TOTAL_MATCH": 210_348,
    "TDC_INDEPENDENT6_WIDTH_MATCH": 1_264_896,
}


ROOT = Path(__file__).resolve().parents[1]


def _batch(size: int = 4) -> dict[str, torch.Tensor]:
    mask = torch.ones(size, 12)
    return {
        "sample_index": torch.arange(size),
        "x_value": torch.randn(size, 12, 10),
        "x_mask": mask,
        "x_age": recompute_age(mask),
        "x_cal": torch.randn(size, 12, 2),
    }


def test_exact_reported_parameter_counts() -> None:
    for model_id, expected in EXPECTED_PARAMETERS.items():
        actual = sum(parameter.numel() for parameter in build_model(model_id).parameters())
        assert actual == expected, (model_id, actual, expected)


def test_interventions_update_value_mask_and_age_together() -> None:
    batch = _batch()
    changed = apply_history_intervention(batch, "recent_3")
    assert torch.all(changed["x_mask"][:, -3:] == 0)
    assert torch.all(changed["x_value"][:, -3:] == 0)
    assert torch.equal(changed["x_age"], recompute_age(changed["x_mask"]))
    assert_nested_missingness(batch)


def test_final_intervention_randomization_contract() -> None:
    assert SHUFFLE_SEEDS == (20260727, 20260728, 20260729, 20260730, 20260731)
    assert TDC_FIXED_PERMUTATION == (7, 0, 10, 3, 5, 1, 11, 8, 2, 9, 4, 6)
    assert CROSS_MODEL_STRESS_HASH_SEED == 20260816
    assert TDC_STRESS_HASH_SEED == 20260727


def test_temporal_shuffle_moves_complete_tokens() -> None:
    batch = _batch()
    order = fixed_permutations()[0]
    changed = apply_history_intervention(batch, "temporal_shuffle", permutation=order)
    index = torch.tensor(order)
    for key in ("x_value", "x_mask", "x_age", "x_cal"):
        assert torch.equal(changed[key], batch[key].index_select(1, index))


def test_selected_epoch_refit_keeps_sixty_epoch_schedule() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=3e-4)
    scheduler = build_lr_scheduler(
        optimizer,
        steps_per_epoch=10,
        schedule_total_epochs=60,
        schedule="linear_warmup_cosine",
        warmup_fraction=0.05,
        warmup_start_factor=0.10,
        min_lr_ratio=0.01,
    )
    for _ in range(14 * 10):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] > 3e-6


def test_reported_training_contract() -> None:
    contract = json.loads(
        (ROOT / "configs" / "paper_experiments.json").read_text(encoding="utf-8")
    )
    training = contract["training"]
    assert training["num_workers"] == 4
    assert training["maximum_epochs"] == 60
    assert training["final_refit_epochs"] == {
        "C20_SITS_CAPACITY_219K": 7,
        "C21_SITS_CAPACITY_242K": 6,
        "C22_CFC_MMRNN_CAPACITY_211K": 14,
        "C23_CFC_MMRNN_CAPACITY_247K": 19,
        "H00_UNIFORM_TIMEMIX_CFC_211K": 11,
        "H01_HSR_TIMEMIX_CFC_211K": 25,
        "H02_RD_TIMEMIX_CFC_211K": 13,
        "H03_HSR_RD_TIMEMIX_CFC_211K": 14,
    }
    assert training["best_validation_epochs"]["H03_HSR_RD_TIMEMIX_CFC_211K"] == {
        "438344685": 25,
        "293280205": 11,
        "353421717": 14,
    }
    lineage = contract["frozen_training_lineage"]
    assert lineage["source_bundle_sha256"] == (
        "D7E3FD7B5EBE8DC3489A6FB5FFDB95BEFD94A7F70CD26676651F9E129DC49464"
    )
    assert lineage["model_source_sha256"] == (
        "811B1F49FBC4280319985A8A16C3727E791D5B2B23553F81A1061854D425C78A"
    )
    assert lineage["training_core_sha256"] == (
        "522D9BCF4E14DEE4FE2383375C4F702D6251E4CE877E976C3DA10FDAC921AEE0"
    )
    assert lineage["training_entry_sha256"] == (
        "8DAADE091869BB7204C11DE6CFA0674559D94681D81F736459DBFB891C6C6EAF"
    )


def test_small_paired_bootstrap_contract() -> None:
    site_ids = [f"utm50n_500m_x{x}_y{y}" for x in range(4) for y in range(4)]
    fixed = pd.DataFrame({"coord_point_id": site_ids})
    metadata = pd.DataFrame(
        {
            "coord_point_id": site_ids * 2,
            "target_year_month": ["2025-01"] * 16 + ["2025-02"] * 16,
        }
    )
    observed = np.zeros((32, 6), dtype=float)
    predictions = {
        "TDC-CfC": np.full((32, 6), 0.1),
        "Comparator": np.full((32, 6), 0.2),
    }
    result = paired_macro_rmse_bootstrap(
        metadata,
        observed,
        predictions,
        fixed,
        reference_model="TDC-CfC",
        repetitions=10,
        excluded_month=None,
        minimum_month_support=1,
    )
    comparator = result.loc[result["model"].eq("Comparator")].iloc[0]
    assert comparator["positive_gain_probability"] == 1.0
