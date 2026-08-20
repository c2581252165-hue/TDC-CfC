from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: str | Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def stable_json_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest().upper()


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def worker_seed(worker_id: int) -> None:
    import torch

    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def month_ordinal(year_month: str) -> int:
    year, month = (int(part) for part in year_month.split("-"))
    return year * 12 + month - 1


def ordinal_to_year_month(value: int) -> str:
    year, month0 = divmod(int(value), 12)
    return f"{year:04d}-{month0 + 1:02d}"


def calendar_features_from_ordinals(ordinals: np.ndarray) -> np.ndarray:
    month0 = np.asarray(ordinals, dtype=np.int64) % 12
    angle = 2.0 * np.pi * month0 / 12.0
    return np.stack((np.sin(angle), np.cos(angle)), axis=-1).astype(np.float32)
