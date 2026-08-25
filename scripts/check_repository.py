from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    failures: list[str] = []

    excluded_directories = {
        "checkpoints",
        "data/example",
        "outputs",
        "results",
        "runs",
    }
    for relative in sorted(excluded_directories):
        if (ROOT / relative).exists():
            failures.append(f"generated asset directory included: {relative}")

    required = {
        "scripts/train.py",
        "scripts/refit.py",
        "scripts/evaluate.py",
        "scripts/analyze_interventions.py",
        "scripts/analyze_factorial.py",
        "scripts/analyze_spectral_structure.py",
        "scripts/analyze_bootstrap.py",
        "scripts/analyze_results.py",
        "scripts/analyze_robustness.py",
        "scripts/analyze_tdc_components.py",
        "scripts/train_hgb.py",
        "scripts/data/export_fixed_point_multiband_s2_v2_1.py",
        "scripts/materialize.py",
        "scripts/create_synthetic_example.py",
        "scripts/run_synthetic_example.py",
        "src/fpmf/models/paper_variants.py",
        "docs/PAPER_CODE_MAP.md",
        "docs/SYNTHETIC_EXAMPLE.md",
        "configs/paper_experiments.json",
        "data/points/point_registry.csv",
        "data/panel/training_monthly_panel.csv.gz",
        "data/static/fixed_site_static_metadata.csv",
        "data/README.md",
        "data/temporal_split_v1.csv",
        "examples/synthetic_input.npz",
        "examples/expected_untrained_predictions.csv",
    }
    for relative in sorted(required):
        if not (ROOT / relative).exists():
            failures.append(f"missing paper asset: {relative}")

    configuration_path = ROOT / "configs/paper_experiments.json"
    if configuration_path.is_file():
        configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
        from fpmf.interventions import (
            CROSS_MODEL_STRESS_HASH_SEED,
            SHUFFLE_SEEDS,
            TDC_FIXED_PERMUTATION,
            TDC_STRESS_HASH_SEED,
        )
        from fpmf.models.factory import build_model

        for model_id, expected in configuration["models"].items():
            actual = sum(parameter.numel() for parameter in build_model(model_id).parameters())
            if actual != expected:
                failures.append(
                    f"parameter-count mismatch: {model_id}={actual}, expected={expected}"
                )
        intervention = configuration["frozen_history_interventions"]
        if tuple(intervention["shuffle_seeds"]) != SHUFFLE_SEEDS:
            failures.append("shuffle-seed mismatch")
        if tuple(intervention["tdc_fixed_permutation"]) != TDC_FIXED_PERMUTATION:
            failures.append("TDC permutation mismatch")
        if intervention["cross_model_missingness_hash_seed"] != CROSS_MODEL_STRESS_HASH_SEED:
            failures.append("cross-model missingness seed mismatch")
        if intervention["tdc_missingness_hash_seed"] != TDC_STRESS_HASH_SEED:
            failures.append("TDC missingness seed mismatch")

    weight_suffixes = {
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".onnx",
        ".joblib",
        ".pkl",
        ".pickle",
    }
    bundled_weights = sorted(
        path.relative_to(ROOT)
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in weight_suffixes
    )
    failures.extend(f"model weight included: {path}" for path in bundled_weights)

    if failures:
        print("FAIL")
        print("\n".join(f"- {failure}" for failure in failures))
        raise SystemExit(1)
    print("PASS: repository contents and paper configuration are consistent")


if __name__ == "__main__":
    main()
