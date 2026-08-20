from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpmf.data import TrainingStore  # noqa: E402
from fpmf.models.factory import MODEL_ID, list_model_ids  # noqa: E402
from fpmf.training import TrainConfig, refit_neural_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed-epoch 2022--2024 refit using the original 60-epoch LR trajectory"
    )
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=438344685)
    parser.add_argument("--model", choices=list_model_ids(), default=MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = TrainConfig(
        model_id=args.model,
        seed=args.seed,
        learning_rate=3e-4,
        epochs=60,
        schedule_total_epochs=60,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    with TrainingStore(args.store) as store:
        summary = refit_neural_model(
            store, config, args.output, selected_epochs=args.selected_epochs
        )
    print(summary)


if __name__ == "__main__":
    main()
