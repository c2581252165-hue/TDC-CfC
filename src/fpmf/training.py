from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .classical import SimpleBaselineSuite
from .data import TrainingStore
from .evaluation import evaluate_predictions
from .models.factory import build_model
from .preprocessing import FoldPreprocessor
from .utils import worker_seed, set_global_seed, stable_json_hash, write_json


@dataclass(frozen=True)
class TrainConfig:
    model_id: str
    seed: int
    data_variant: str = "S2_ONLY"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    lr_schedule: str = "linear_warmup_cosine"
    warmup_fraction: float = 0.05
    warmup_start_factor: float = 0.10
    min_lr_ratio: float = 0.01
    schedule_total_epochs: int = 60
    early_stopping: bool = False
    batch_size: int = 1024
    epochs: int = 60
    patience: int = 10
    gradient_clip: float = 1.0
    num_workers: int = 4
    device: str = "cuda"
    report_every: int = 10
    max_train_samples: int | None = None
    max_validation_samples: int | None = None


def _limited_dataset(dataset, limit: int | None):
    if limit is None or len(dataset) <= limit:
        return dataset
    return Subset(dataset, list(range(int(limit))))


def _loader(dataset, config: TrainConfig, *, shuffle: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
        worker_init_fn=worker_seed,
        generator=generator,
        persistent_workers=config.num_workers > 0,
    )


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    steps_per_epoch: int,
    schedule_total_epochs: int,
    schedule: str,
    warmup_fraction: float,
    warmup_start_factor: float,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    if schedule != "linear_warmup_cosine":
        raise ValueError(f"Unsupported learning-rate schedule: {schedule}")
    if steps_per_epoch < 1 or schedule_total_epochs < 1:
        raise ValueError("Learning-rate schedule requires positive steps and epochs")
    if not 0.0 < warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in (0, 1)")
    if not 0.0 < warmup_start_factor <= 1.0:
        raise ValueError("warmup_start_factor must be in (0, 1]")
    if not 0.0 <= min_lr_ratio < 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1)")
    total_steps = steps_per_epoch * schedule_total_epochs
    warmup_steps = max(1, math.ceil(total_steps * warmup_fraction))
    decay_steps = max(1, total_steps - warmup_steps)

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            progress = step / max(1, warmup_steps - 1)
            return warmup_start_factor + (1.0 - warmup_start_factor) * progress
        progress = min(1.0, (step - warmup_steps) / decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def _capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(payload: dict) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch"])
    if torch.cuda.is_available() and payload.get("cuda") is not None:
        torch.cuda.set_rng_state_all(payload["cuda"])


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    config: TrainConfig,
    data_contract_hash: str,
    epoch: int,
    best_score: float,
    best_epoch: int,
    stale: int,
    best_state: dict[str, torch.Tensor] | None,
    history: list[dict],
    train_loader: DataLoader,
    validation_loader: DataLoader,
) -> dict:
    return {
        "checkpoint_kind": "latest_resumable",
        "model_id": config.model_id,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": scheduler.state_dict(),
        "grad_scaler_state_dict": scaler.state_dict(),
        "config": asdict(config),
        "config_hash": stable_json_hash(asdict(config)),
        "data_contract_hash": data_contract_hash,
        "epoch": epoch,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "stale": stale,
        "best_state_dict": best_state,
        "history": history,
        "rng_state": _capture_rng_state(),
        "train_loader_generator_state": train_loader.generator.get_state(),
        "validation_loader_generator_state": validation_loader.generator.get_state(),
    }


@torch.inference_mode()
def predict_neural(model: nn.Module, loader: DataLoader, preprocessor: FoldPreprocessor, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    sample_indices, predictions = [], []
    use_amp = device.type == "cuda"
    for batch in loader:
        indices = batch["sample_index"].numpy()
        moved = {key: value.to(device, non_blocking=True) for key, value in batch.items() if key not in {"sample_index", "y_raw"}}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            output = model(moved)
        predictions.append(preprocessor.inverse_y(output.prediction.float().cpu().numpy()))
        sample_indices.append(indices)
    return np.concatenate(sample_indices), np.concatenate(predictions)


def train_neural_model(
    store: TrainingStore,
    config: TrainConfig,
    run_dir: str | Path,
    *,
    resume: bool = False,
) -> dict:
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if config.report_every < 1:
        raise ValueError("report_every must be at least 1")
    if config.schedule_total_epochs < config.epochs:
        raise ValueError("schedule_total_epochs cannot be shorter than epochs")
    if config.data_variant != "S2_ONLY":
        raise ValueError("The paper uses Sentinel-2 history only")
    device = torch.device(config.device)
    run_path = Path(run_dir)
    if resume:
        if not run_path.is_dir() or not (run_path / "latest.pt").exists():
            raise FileNotFoundError("--resume requires an existing run directory with latest.pt")
        if (run_path / "summary.json").exists():
            raise RuntimeError("Completed runs are immutable and cannot be resumed")
    else:
        run_path.mkdir(parents=True, exist_ok=False)
    set_global_seed(config.seed)
    train_indices = store.indices("train")
    validation_indices = store.indices("validation")
    preprocessor = FoldPreprocessor.fit(store, train_indices, fitted_split="train")
    if not resume:
        preprocessor.save(run_path / "preprocessor.json")
    baseline = SimpleBaselineSuite.fit(store, train_indices)
    baseline_validation = baseline.predict_all(store, validation_indices)
    pointmonth_validation = baseline_validation["B02_POINT_MONTH"]
    strongest_validation = baseline_validation["B05_POINT_MONTH_AR1"]
    train_dataset = _limited_dataset(
        store.dataset(
            "train",
            preprocessor,
        ),
        config.max_train_samples,
    )
    validation_dataset = _limited_dataset(
        store.dataset(
            "validation",
            preprocessor,
        ),
        config.max_validation_samples,
    )
    train_loader = _loader(train_dataset, config, shuffle=True)
    validation_loader = _loader(validation_dataset, config, shuffle=False)
    model = build_model(config.model_id).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=config.weight_decay,
        amsgrad=False,
    )
    scheduler = build_lr_scheduler(
        optimizer,
        steps_per_epoch=len(train_loader),
        schedule_total_epochs=config.schedule_total_epochs,
        schedule=config.lr_schedule,
        warmup_fraction=config.warmup_fraction,
        warmup_start_factor=config.warmup_start_factor,
        min_lr_ratio=config.min_lr_ratio,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score, best_epoch, stale = float("-inf"), -1, 0
    best_state = None
    history: list[dict] = []
    start_epoch = 0
    if resume:
        latest = torch.load(run_path / "latest.pt", map_location="cpu", weights_only=False)
        expected_config_hash = stable_json_hash(asdict(config))
        if latest.get("config_hash") != expected_config_hash:
            raise RuntimeError("Resume config hash mismatch")
        if latest.get("data_contract_hash") != store.manifest["contract_hash"]:
            raise RuntimeError("Resume data contract hash mismatch")
        model.load_state_dict(latest["state_dict"])
        optimizer.load_state_dict(latest["optimizer_state_dict"])
        if "lr_scheduler_state_dict" not in latest:
            raise RuntimeError("Resume checkpoint lacks learning-rate scheduler state")
        scheduler.load_state_dict(latest["lr_scheduler_state_dict"])
        scaler.load_state_dict(latest["grad_scaler_state_dict"])
        start_epoch = int(latest["epoch"])
        best_score = float(latest["best_score"])
        best_epoch = int(latest["best_epoch"])
        stale = int(latest["stale"])
        best_state = latest["best_state_dict"]
        history = list(latest["history"])
        train_loader.generator.set_state(latest["train_loader_generator_state"])
        validation_loader.generator.set_state(latest["validation_loader_generator_state"])
        _restore_rng_state(latest["rng_state"])
        print(f"RESUME {config.model_id}: epoch={start_epoch}, best_epoch={best_epoch}, stale={stale}", flush=True)
    started = time.time()
    for epoch in range(start_epoch, config.epochs):
        model.train()
        loss_sum, sample_count = 0.0, 0
        epoch_learning_rates: list[float] = []
        optimizer_steps_attempted = 0
        optimizer_steps_succeeded = 0
        optimizer_steps_skipped = 0
        scaler_scales: list[float] = []
        for batch in train_loader:
            moved = {key: value.to(device, non_blocking=True) for key, value in batch.items() if key not in {"sample_index", "y_raw"}}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(moved)
                loss = torch.nn.functional.huber_loss(output.prediction, moved["y"], delta=1.0)
                if output.auxiliary_losses:
                    loss = loss + sum(output.auxiliary_losses.values(), loss.new_zeros(()))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip, error_if_nonfinite=False
            )
            gradient_norm_finite = bool(torch.isfinite(gradient_norm))
            learning_rate_used = float(optimizer.param_groups[0]["lr"])
            scale_before = float(scaler.get_scale())
            if scale_before <= 0 or not np.isfinite(scale_before):
                raise FloatingPointError(
                    f"Invalid AMP scale before optimizer step for {config.model_id} "
                    f"at epoch {epoch + 1}: {scale_before}"
                )
            optimizer_steps_attempted += 1
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            scaler_scales.extend((scale_before, scale_after))
            if scale_after <= 0 or not np.isfinite(scale_after):
                raise FloatingPointError(
                    f"Invalid AMP scale after optimizer step for {config.model_id} "
                    f"at epoch {epoch + 1}: {scale_after}"
                )
            if scale_after < scale_before:
                optimizer_steps_skipped += 1
            elif not gradient_norm_finite:
                raise FloatingPointError(
                    f"Non-finite gradients were not skipped by GradScaler for "
                    f"{config.model_id} at epoch {epoch + 1}"
                )
            else:
                optimizer_steps_succeeded += 1
                epoch_learning_rates.append(learning_rate_used)
                scheduler.step()
            batch_size = moved["y"].shape[0]
            loss_sum += float(loss.detach()) * batch_size
            sample_count += batch_size
        if optimizer_steps_succeeded == 0:
            raise FloatingPointError(
                f"Zero successful optimizer updates for {config.model_id} at epoch {epoch + 1}; "
                f"attempted={optimizer_steps_attempted}, skipped={optimizer_steps_skipped}"
            )
        sample_index, validation_prediction = predict_neural(model, validation_loader, preprocessor, device)
        order = np.argsort(sample_index)
        sample_index, validation_prediction = sample_index[order], validation_prediction[order]
        expected_indices = validation_indices[: len(validation_dataset)]
        if not np.array_equal(sample_index, expected_indices):
            raise RuntimeError("Validation prediction/sample alignment failed")
        validation_observed = np.asarray(store.y[sample_index], dtype=np.float32)
        validation_metadata = store.metadata.iloc[sample_index].reset_index(drop=True)
        reference = pointmonth_validation[: len(sample_index)]
        strongest_reference = strongest_validation[: len(sample_index)]
        metrics = evaluate_predictions(
            validation_metadata,
            validation_observed,
            validation_prediction,
            reference,
            strongest_reference,
            model_id=f"{config.model_id}__{config.data_variant}",
            seed=config.seed,
            split="validation",
            include_by_point=False,
            min_evaluable_month_n=1 if config.max_validation_samples is not None else None,
        )
        score = float(metrics["macro"].iloc[0]["skill_vs_pointmonth"])
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / max(sample_count, 1),
            "validation_month_band_macro_skill_vs_pointmonth": score,
            "validation_month_band_macro_rmse": float(metrics["macro"].iloc[0]["rmse"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "learning_rate_min": float(min(epoch_learning_rates)),
            "learning_rate_max": float(max(epoch_learning_rates)),
            "optimizer_steps_attempted": optimizer_steps_attempted,
            "optimizer_steps_succeeded": optimizer_steps_succeeded,
            "optimizer_steps_skipped": optimizer_steps_skipped,
            "amp_scale_min": float(min(scaler_scales)),
            "amp_scale_end": float(scaler_scales[-1]),
            "elapsed_seconds": time.time() - started,
        }
        history.append(epoch_record)
        improved = np.isfinite(score) and score > best_score + 1e-8
        if improved:
            best_score, best_epoch, stale = score, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
            _atomic_torch_save(
                {
                    "checkpoint_kind": "best_validation",
                    "model_id": config.model_id,
                    "state_dict": best_state,
                    "config": asdict(config),
                    "config_hash": stable_json_hash(asdict(config)),
                    "data_contract_hash": store.manifest["contract_hash"],
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                },
                run_path / "best.pt",
            )
        else:
            stale += 1
        write_json(run_path / "history.json", history)
        write_json(
            run_path / "progress.json",
            {
                "status": "running",
                "model_id": config.model_id,
                "data_variant": config.data_variant,
                "epoch": epoch + 1,
                "max_epochs": config.epochs,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "stale_epochs": stale,
                "learning_rate": epoch_record["learning_rate"],
                "test_opened": False,
            },
        )
        latest_payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            data_contract_hash=store.manifest["contract_hash"],
            epoch=epoch + 1,
            best_score=best_score,
            best_epoch=best_epoch,
            stale=stale,
            best_state=best_state,
            history=history,
            train_loader=train_loader,
            validation_loader=validation_loader,
        )
        _atomic_torch_save(latest_payload, run_path / "latest.pt")
        should_stop = config.early_stopping and stale >= config.patience
        if epoch == start_epoch or (epoch + 1) % config.report_every == 0 or epoch + 1 == config.epochs or should_stop:
            print(
                f"[{config.model_id}::{config.data_variant}] epoch {epoch + 1:03d}/{config.epochs:03d} "
                f"loss={epoch_record['train_loss']:.6f} val_skill={score:.6f} "
                f"best={best_score:.6f}@{best_epoch} lr={epoch_record['learning_rate']:.3e} "
                f"stale={stale}/{'off' if not config.early_stopping else config.patience}",
                flush=True,
            )
        if should_stop:
            break
    if best_state is None:
        raise RuntimeError("No finite best model was produced")
    model.load_state_dict(best_state)
    sample_index, prediction = predict_neural(model, validation_loader, preprocessor, device)
    order = np.argsort(sample_index)
    sample_index, prediction = sample_index[order], prediction[order]
    observed = np.asarray(store.y[sample_index], dtype=np.float32)
    metadata = store.metadata.iloc[sample_index].reset_index(drop=True)
    reference = pointmonth_validation[: len(sample_index)]
    strongest_reference = strongest_validation[: len(sample_index)]
    final_metrics = evaluate_predictions(metadata, observed, prediction, reference, strongest_reference, model_id=f"{config.model_id}__{config.data_variant}", seed=config.seed, split="validation")
    for name, frame in final_metrics.items():
        frame.to_parquet(run_path / f"metrics_{name}.parquet", index=False)
    np.savez_compressed(run_path / "validation_predictions.npz", sample_index=sample_index, observed=observed, predicted=prediction, pointmonth=reference, strongest_simple=strongest_reference)
    summary = {
        "status": "completed",
        "config": asdict(config),
        "config_hash": stable_json_hash(asdict(config)),
        "data_contract_hash": store.manifest["contract_hash"],
        "best_epoch": best_epoch,
        "best_validation_month_band_macro_skill_vs_pointmonth": best_score,
        "selection_metric": "validation month-band macro skill versus PointMonth",
        "loss": "per-band train-scale-normalized equal-weight Huber(delta=1)",
        "optimizer_contract": {
            "name": "AdamW",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": config.weight_decay,
            "amsgrad": False,
            "lr_schedule": config.lr_schedule,
            "warmup_fraction": config.warmup_fraction,
            "warmup_start_factor": config.warmup_start_factor,
            "min_lr_ratio": config.min_lr_ratio,
            "schedule_total_epochs": config.schedule_total_epochs,
            "early_stopping": config.early_stopping,
        },
        "elapsed_seconds": time.time() - started,
        "epochs_ran": len(history),
        "run_artifacts": ["history.json", "progress.json", "latest.pt", "best.pt"],
        "test_opened": False,
    }
    write_json(run_path / "progress.json", {"status": "completed", "best_epoch": best_epoch, "epochs_ran": len(history), "test_opened": False})
    write_json(run_path / "summary.json", summary)
    return summary


def refit_neural_model(
    store: TrainingStore,
    config: TrainConfig,
    run_dir: str | Path,
    *,
    selected_epochs: int,
) -> dict:
    """Reinitialize and refit a selected model on 2022--2024.

    The optimizer follows the original 60-epoch learning-rate trajectory and
    stops after the validation-selected epoch count.  This distinction is
    essential: a 14-epoch refit is not a newly compressed 14-epoch schedule.
    """
    if selected_epochs < 1 or selected_epochs > config.schedule_total_epochs:
        raise ValueError("selected_epochs must lie within the fixed schedule")
    if config.data_variant != "S2_ONLY":
        raise ValueError("The paper uses Sentinel-2 history only")
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=False)
    device = torch.device(config.device)
    set_global_seed(config.seed)
    refit_indices = store.indices(("train", "validation"))
    preprocessor = FoldPreprocessor.fit(
        store, refit_indices, fitted_split="train+validation"
    )
    preprocessor.save(run_path / "preprocessor.json")
    dataset = store.dataset(
        ("train", "validation"),
        preprocessor,
    )
    refit_config = TrainConfig(
        **{
            **asdict(config),
            "epochs": int(selected_epochs),
            "early_stopping": False,
        }
    )
    loader = _loader(dataset, refit_config, shuffle=True)
    model = build_model(config.model_id).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=config.weight_decay,
        amsgrad=False,
    )
    scheduler = build_lr_scheduler(
        optimizer,
        steps_per_epoch=len(loader),
        schedule_total_epochs=config.schedule_total_epochs,
        schedule=config.lr_schedule,
        warmup_fraction=config.warmup_fraction,
        warmup_start_factor=config.warmup_start_factor,
        min_lr_ratio=config.min_lr_ratio,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    for epoch in range(selected_epochs):
        model.train()
        total_loss, total_n, successful_steps = 0.0, 0, 0
        for batch in loader:
            moved = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
                if key not in {"sample_index", "y_raw"}
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model(moved)
                loss = torch.nn.functional.huber_loss(
                    output.prediction, moved["y"], delta=1.0
                )
                loss = loss + sum(
                    output.auxiliary_losses.values(),
                    loss.new_zeros(()),
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite refit loss at epoch {epoch + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            if float(scaler.get_scale()) >= scale_before:
                scheduler.step()
                successful_steps += 1
            batch_n = moved["y"].shape[0]
            total_loss += float(loss.detach()) * batch_n
            total_n += batch_n
        if successful_steps == 0:
            raise FloatingPointError("No successful optimizer step in refit epoch")
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / max(total_n, 1),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "successful_optimizer_steps": successful_steps,
            }
        )
        write_json(run_path / "history.json", history)
    checkpoint = {
        "checkpoint_kind": "fixed_epoch_train_validation_refit",
        "model_id": config.model_id,
        "state_dict": model.state_dict(),
        "seed": config.seed,
        "selected_epochs": selected_epochs,
        "schedule_total_epochs": config.schedule_total_epochs,
        "config": asdict(config),
        "data_contract_hash": store.manifest["contract_hash"],
        "refit_splits": ["train", "validation"],
        "test_opened": False,
    }
    _atomic_torch_save(checkpoint, run_path / "final_refit.pt")
    summary = {
        "status": "completed",
        "model_id": config.model_id,
        "seed": config.seed,
        "selected_epochs": selected_epochs,
        "schedule_total_epochs": config.schedule_total_epochs,
        "learning_rate": config.learning_rate,
        "refit_sample_count": int(len(refit_indices)),
        "checkpoint": "final_refit.pt",
        "test_opened": False,
    }
    write_json(run_path / "summary.json", summary)
    return summary


__all__ = [
    "TrainConfig",
    "build_lr_scheduler",
    "predict_neural",
    "refit_neural_model",
    "train_neural_model",
]
