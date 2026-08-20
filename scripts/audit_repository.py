from __future__ import annotations

from pathlib import Path
import json
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".csv", ".cff", ".yml", ".yaml"}
FORBIDDEN = (
    "GITHUB_REPOSITORY_URL_REQUIRED",
    "LICENSE_SELECTION_REQUIRED",
    "candidate_pool_",
    "adjudication",
    "retrospective_provenance",
    "include_multisource",
    "x_source_present",
)
FORBIDDEN_FILES = {"multisource.py", "timemix.py"}


def main() -> None:
    failures = []
    forbidden_release_paths = {
        "checkpoints",
        "data/example",
        "outputs",
        "results",
        "runs",
    }
    for relative in sorted(forbidden_release_paths):
        if (ROOT / relative).exists():
            failures.append(f"generated or unpublished asset directory: {relative}")
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.name in FORBIDDEN_FILES:
            failures.append(f"forbidden legacy file: {path.relative_to(ROOT)}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in FORBIDDEN:
            if token in text and path.name != Path(__file__).name:
                failures.append(f"forbidden token {token!r}: {path.relative_to(ROOT)}")
        if re.search(r"(?:^|[\\/])(?:__pycache__|\.pytest_cache)(?:[\\/]|$)", str(path)):
            failures.append(f"cache file: {path.relative_to(ROOT)}")
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
        "data/temporal_split_v1.csv",
        "examples/synthetic_input.npz",
        "examples/expected_untrained_predictions.csv",
    }
    for relative in sorted(required):
        if not (ROOT / relative).exists():
            failures.append(f"missing required paper asset: {relative}")
    configuration_path = ROOT / "configs/paper_experiments.json"
    if configuration_path.is_file():
        configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
        from fpmf.models.factory import build_model
        from fpmf.interventions import (
            CROSS_MODEL_STRESS_HASH_SEED,
            SHUFFLE_SEEDS,
            TDC_FIXED_PERMUTATION,
            TDC_STRESS_HASH_SEED,
        )

        for model_id, expected in configuration["models"].items():
            actual = sum(parameter.numel() for parameter in build_model(model_id).parameters())
            if actual != expected:
                failures.append(f"parameter-count drift: {model_id}={actual}, expected={expected}")
        intervention = configuration["frozen_history_interventions"]
        if tuple(intervention["shuffle_seeds"]) != SHUFFLE_SEEDS:
            failures.append("shuffle-seed contract drift")
        if tuple(intervention["tdc_fixed_permutation"]) != TDC_FIXED_PERMUTATION:
            failures.append("TDC fixed-permutation contract drift")
        if intervention["cross_model_missingness_hash_seed"] != CROSS_MODEL_STRESS_HASH_SEED:
            failures.append("cross-model missingness seed drift")
        if intervention["tdc_missingness_hash_seed"] != TDC_STRESS_HASH_SEED:
            failures.append("TDC missingness seed drift")
    bundled_weights = sorted(
        path.relative_to(ROOT)
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower()
        in {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".joblib", ".pkl", ".pickle"}
    )
    if bundled_weights:
        failures.extend(f"bundled model asset: {path}" for path in bundled_weights)
    if failures:
        print("FAIL")
        print("\n".join(f"- {failure}" for failure in failures))
        raise SystemExit(1)
    print("PASS: repository contains the paper pipeline and no known legacy assets")


if __name__ == "__main__":
    main()
