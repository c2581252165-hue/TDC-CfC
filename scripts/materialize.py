from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpmf.data import materialize_training_store  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the 12-to-1 training store")
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = materialize_training_store(args.panel, args.split, args.output)
    print(f"materialized {contract['sample_count']} samples at {args.output.resolve()}")


if __name__ == "__main__":
    main()
