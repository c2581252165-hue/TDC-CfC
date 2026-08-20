from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import joblib
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpmf.classical import SimpleBaselineSuite  # noqa: E402
from fpmf.data import TrainingStore  # noqa: E402
from fpmf.evaluation import evaluate_predictions  # noqa: E402
from fpmf.interventions import (  # noqa: E402
    CROSS_MODEL_STRESS_HASH_SEED,
    MISSINGNESS_RATES,
    RECENT_HORIZONS,
    TDC_STRESS_HASH_SEED,
    apply_history_intervention,
    fixed_permutations,
    tdc_fixed_permutation,
)
from fpmf.models.factory import build_model  # noqa: E402
from fpmf.preprocessing import FoldPreprocessor  # noqa: E402


def parse_run(value: str) -> tuple[str, str, int, Path]:
    parts = value.split("=", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("run must be NAME=MODEL_ID=SEED=RUN_DIR")
    return parts[0], parts[1], int(parts[2]), Path(parts[3])


def parse_hgb_run(value: str) -> tuple[str, int, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("hgb-run must be NAME=SEED=RUN_DIR")
    return parts[0], int(parts[1]), Path(parts[2])


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    preprocessor: FoldPreprocessor,
    device: torch.device,
    scenario: str,
    permutation: list[int] | None,
    stress_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices, predictions = [], []
    for batch in loader:
        changed = apply_history_intervention(
            batch,
            scenario,
            permutation=permutation,
            stress_seed=stress_seed,
        )
        indices.append(changed["sample_index"].numpy())
        moved = {key: value.to(device) for key, value in changed.items() if key.startswith("x_")}
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            output = model(moved).prediction
        predictions.append(preprocessor.inverse_y(output.float().cpu().numpy()))
    sample_index = np.concatenate(indices)
    prediction = np.concatenate(predictions)
    order = np.argsort(sample_index)
    return sample_index[order], prediction[order]


def checkpoint_path(run_dir: Path) -> Path:
    for name in ("best.pt", "final_refit.pt"):
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No best.pt or final_refit.pt in {run_dir}")


def hgb_checkpoint_path(run_dir: Path) -> Path:
    for name in ("hgb.joblib", "model.joblib"):
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No hgb.joblib or model.joblib in {run_dir}")


def natural_strata(store: TrainingStore, indices: np.ndarray) -> list[tuple[str, str, np.ndarray]]:
    mask = np.asarray(store.x_mask[indices], dtype=np.float32)
    age = np.asarray(store.x_age[indices], dtype=np.float32)
    coverage = mask.sum(axis=1).astype(int)
    last_age = age[:, -1].astype(int)
    return [
        ("coverage_valid_months", "le_9", coverage <= 9),
        ("coverage_valid_months", "10", coverage == 10),
        ("coverage_valid_months", "11", coverage == 11),
        ("coverage_valid_months", "12", coverage == 12),
        ("last_observation_age_months", "0", last_age == 0),
        ("last_observation_age_months", "1", last_age == 1),
        ("last_observation_age_months", "ge_2", last_age >= 2),
    ]


def hgb_features(batch: dict[str, torch.Tensor]) -> np.ndarray:
    base = torch.cat(
        (
            batch["x_value"],
            batch["x_mask"].unsqueeze(-1),
            batch["x_age"].unsqueeze(-1),
            batch["x_cal"],
        ),
        dim=-1,
    )
    zeros = torch.zeros((*base.shape[:2], 39), dtype=base.dtype)
    tokens = torch.cat((base, zeros), dim=-1)
    if tokens.shape[1:] != (12, 53):
        raise RuntimeError(f"Unexpected HGB token shape: {tokens.shape}")
    return tokens.reshape(tokens.shape[0], -1).numpy().astype(np.float32)


def predict_hgb(
    model,
    loader: DataLoader,
    preprocessor: FoldPreprocessor,
    scenario: str,
    permutation: list[int] | None,
    stress_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices, predictions = [], []
    for batch in loader:
        changed = apply_history_intervention(
            batch,
            scenario,
            permutation=permutation,
            stress_seed=stress_seed,
        )
        indices.append(changed["sample_index"].numpy())
        normalized = np.asarray(model.predict(hgb_features(changed)), dtype=np.float32)
        predictions.append(preprocessor.inverse_y(normalized))
    sample_index = np.concatenate(indices)
    prediction = np.concatenate(predictions)
    order = np.argsort(sample_index)
    return sample_index[order], prediction[order]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-model or TDC-CfC frozen-weight historical interventions"
    )
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--hgb-run", action="append", type=parse_hgb_run, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--protocol",
        choices=("cross-model", "tdc"),
        default="cross-model",
        help="Use the five cross-model shuffles or the final single TDC-CfC shuffle",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--stress-seed",
        type=int,
        default=None,
        help="Override the protocol-specific deterministic missingness seed",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if args.protocol == "cross-model":
        shuffle_scenarios = [
            (f"temporal_shuffle_{index + 1}", order)
            for index, order in enumerate(fixed_permutations())
        ]
        stress_seed = (
            CROSS_MODEL_STRESS_HASH_SEED if args.stress_seed is None else args.stress_seed
        )
    else:
        shuffle_scenarios = [("temporal_shuffle_fixed", tdc_fixed_permutation())]
        stress_seed = TDC_STRESS_HASH_SEED if args.stress_seed is None else args.stress_seed
    scenarios: list[tuple[str, list[int] | None]] = [
        ("natural", None),
        ("previous_year_removed", None),
        *[(f"recent_{count}", None) for count in RECENT_HORIZONS],
        *shuffle_scenarios,
        *[(f"missingness_{rate}", None) for rate in MISSINGNESS_RATES],
    ]
    macro_rows, band_rows, month_band_rows, strata_rows = [], [], [], []
    with TrainingStore(args.store) as store:
        evaluation_indices = store.indices(args.split, allow_test=args.split == "test")
        baseline_fit_indices = store.indices(("train", "validation")) if args.split == "test" else store.indices("train")
        references = SimpleBaselineSuite.fit(store, baseline_fit_indices).predict_all(
            store, evaluation_indices
        )
        observed = np.asarray(store.y[evaluation_indices], dtype=np.float32)
        metadata = store.metadata.iloc[evaluation_indices].reset_index(drop=True)
        strata = natural_strata(store, evaluation_indices)
        for display_name, model_id, seed, run_dir in args.run:
            preprocessor = FoldPreprocessor.load(run_dir / "preprocessor.json")
            dataset = store.dataset(
                args.split,
                preprocessor,
                allow_test=args.split == "test",
            )
            loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)
            model = build_model(model_id).to(device)
            checkpoint = torch.load(checkpoint_path(run_dir), map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.eval()
            for scenario_name, permutation in scenarios:
                base_scenario = "temporal_shuffle" if scenario_name.startswith("temporal_shuffle_") else scenario_name
                sample_index, prediction = predict(
                    model,
                    loader,
                    preprocessor,
                    device,
                    base_scenario,
                    permutation,
                    stress_seed,
                )
                if not np.array_equal(sample_index, evaluation_indices):
                    raise RuntimeError("Prediction/sample alignment failed")
                metrics = evaluate_predictions(
                    metadata,
                    observed,
                    prediction,
                    references["B02_POINT_MONTH"],
                    references["B05_POINT_MONTH_AR1"],
                    model_id=model_id,
                    seed=seed,
                    split=args.split,
                    include_by_point=False,
                )
                if scenario_name == "natural":
                    for stratum_type, stratum, selected in strata:
                        if int(selected.sum()) < 50:
                            raise RuntimeError(
                                f"Natural stratum too small: {stratum_type}/{stratum}"
                            )
                        stratum_metrics = evaluate_predictions(
                            metadata.loc[selected].reset_index(drop=True),
                            observed[selected],
                            prediction[selected],
                            references["B02_POINT_MONTH"][selected],
                            references["B05_POINT_MONTH_AR1"][selected],
                            model_id=model_id,
                            seed=seed,
                            split=args.split,
                            include_by_point=False,
                        )["macro_all_months"].iloc[0].to_dict()
                        strata_rows.append(
                            {
                                "model": display_name,
                                "seed": seed,
                                "stratum_type": stratum_type,
                                "stratum": stratum,
                                "sample_count": int(selected.sum()),
                                **stratum_metrics,
                            }
                        )
                for frame_name, target in (
                    ("macro", macro_rows),
                    ("by_band", band_rows),
                    ("month_band", month_band_rows),
                ):
                    frame = metrics[frame_name].copy()
                    frame.insert(0, "scenario", scenario_name)
                    frame.insert(0, "model", display_name)
                    target.append(frame)
        for display_name, seed, run_dir in args.hgb_run:
            preprocessor = FoldPreprocessor.load(run_dir / "preprocessor.json")
            dataset = store.dataset(
                args.split,
                preprocessor,
                allow_test=args.split == "test",
            )
            loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)
            model = joblib.load(hgb_checkpoint_path(run_dir))
            for scenario_name, permutation in scenarios:
                base_scenario = "temporal_shuffle" if scenario_name.startswith("temporal_shuffle_") else scenario_name
                sample_index, prediction = predict_hgb(
                    model,
                    loader,
                    preprocessor,
                    base_scenario,
                    permutation,
                    stress_seed,
                )
                if not np.array_equal(sample_index, evaluation_indices):
                    raise RuntimeError("HGB prediction/sample alignment failed")
                metrics = evaluate_predictions(
                    metadata,
                    observed,
                    prediction,
                    references["B02_POINT_MONTH"],
                    references["B05_POINT_MONTH_AR1"],
                    model_id="B07_HGB",
                    seed=seed,
                    split=args.split,
                    include_by_point=False,
                )
                if scenario_name == "natural":
                    for stratum_type, stratum, selected in strata:
                        if int(selected.sum()) < 50:
                            raise RuntimeError(
                                f"Natural stratum too small: {stratum_type}/{stratum}"
                            )
                        stratum_metrics = evaluate_predictions(
                            metadata.loc[selected].reset_index(drop=True),
                            observed[selected],
                            prediction[selected],
                            references["B02_POINT_MONTH"][selected],
                            references["B05_POINT_MONTH_AR1"][selected],
                            model_id="B07_HGB",
                            seed=seed,
                            split=args.split,
                            include_by_point=False,
                        )["macro_all_months"].iloc[0].to_dict()
                        strata_rows.append(
                            {
                                "model": display_name,
                                "seed": seed,
                                "stratum_type": stratum_type,
                                "stratum": stratum,
                                "sample_count": int(selected.sum()),
                                **stratum_metrics,
                            }
                        )
                for frame_name, target in (
                    ("macro", macro_rows),
                    ("by_band", band_rows),
                    ("month_band", month_band_rows),
                ):
                    frame = metrics[frame_name].copy()
                    frame.insert(0, "scenario", scenario_name)
                    frame.insert(0, "model", display_name)
                    target.append(frame)
    pd.concat(macro_rows, ignore_index=True).to_csv(args.output / "macro_metrics.csv", index=False)
    pd.concat(band_rows, ignore_index=True).to_csv(args.output / "band_metrics.csv", index=False)
    pd.concat(month_band_rows, ignore_index=True).to_csv(
        args.output / "month_band_metrics.csv", index=False
    )
    pd.DataFrame(strata_rows).to_csv(
        args.output / "natural_coverage_age_metrics.csv", index=False
    )


if __name__ == "__main__":
    main()
