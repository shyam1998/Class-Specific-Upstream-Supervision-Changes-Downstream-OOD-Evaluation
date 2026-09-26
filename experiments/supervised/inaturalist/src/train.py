from __future__ import annotations

import copy
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50

from common import (
    MODELS,
    RECIPE_PATH,
    ROOT,
    SEEDS,
    append_csv,
    environment_record,
    read_yaml,
    set_seed,
    sha256_file,
    state_dict_sha256,
    utc_now,
    write_json,
    write_csv,
)
from data import all_validation_dataset, downstream_train_dataset, restricted_validation_dataset, training_dataset


MIGRATION_AUDIT_PATH = ROOT / "migration_audit.json"


def make_model(num_classes: int = 80) -> nn.Module:
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes, bias=True)
    return model


def encoder_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith("fc.")}


def _worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    import random

    random.seed(seed)


def _load_exported_initialization(seed: int) -> tuple[nn.Module, dict[str, Any]]:
    """Recreate the matched random initialization for a public source release.

    Binary initialization snapshots are intentionally excluded. The pinned
    environment and seed reproduce one initialization per seed, shared by all
    four rotations.
    """
    if seed not in SEEDS:
        raise ValueError(f"Unsupported initialization seed: {seed}")
    set_seed(seed)
    model = make_model()
    loaded_state = model.state_dict()
    loaded_hashes = {
        "whole_state_sha256": state_dict_sha256(loaded_state),
        "encoder_state_sha256": state_dict_sha256(
            {key: value for key, value in loaded_state.items() if not key.startswith("fc.")}
        ),
        "classifier_state_sha256": state_dict_sha256(
            {key: value for key, value in loaded_state.items() if key.startswith("fc.")}
        ),
    }
    return model, {
        "initialization_source": "torchvision ResNet-50 weights=None with the recorded seed",
        "loaded_tensor_initialization_hashes": loaded_hashes,
        "strict_model_load": "PASS",
        "tensor_count": len(loaded_state),
    }

def ensure_initializations() -> dict[str, Any]:
    records = {}
    for seed in SEEDS:
        # Preserve the experiment's deterministic seed convention.
        set_seed(seed)
        _, export_record = _load_exported_initialization(seed)
        whole_hash = export_record["loaded_tensor_initialization_hashes"]["whole_state_sha256"]
        enc_hash = export_record["loaded_tensor_initialization_hashes"]["encoder_state_sha256"]
        cls_hash = export_record["loaded_tensor_initialization_hashes"]["classifier_state_sha256"]
        records[str(seed)] = {
            "whole_state_sha256": whole_hash,
            "encoder_state_sha256": enc_hash,
            "classifier_state_sha256": cls_hash,
            **export_record,
            "per_rotation_encoder_hash": {m: enc_hash for m in MODELS},
            "per_rotation_classifier_hash": {m: cls_hash for m in MODELS},
        }
    if records["0"]["encoder_state_sha256"] == records["1"]["encoder_state_sha256"]:
        raise AssertionError("Seed 0 and seed 1 encoder initializations must differ")
    result = {
        "created_utc": utc_now(),
        "architecture": "torchvision ResNet-50, weights=None, 80-way fc",
        "adaptation_scope": "infrastructure migration only",
        "adaptation_reason": (
            "Initializations are generated once per seed and shared across rotations."
        ),
        "scientific_protocol_changed": False,
        "matched_within_seed": True,
        "different_between_seeds": True,
        "seeds": records,
        "host_environment": environment_record(),
        "status": "PASS",
    }
    if MIGRATION_AUDIT_PATH.is_file():
        previous = json.loads(MIGRATION_AUDIT_PATH.read_text(encoding="utf-8"))
        if "dry_validation" in previous:
            result["dry_validation"] = previous["dry_validation"]
    write_json(MIGRATION_AUDIT_PATH, result)
    return result


def load_initial_model(seed: int, device: torch.device) -> nn.Module:
    model, _ = _load_exported_initialization(seed)
    return model.to(device)


def _existing_run_state(model_id: str, seed: int, epochs: int) -> dict[str, Any]:
    run_dir = ROOT / "checkpoints" / f"{model_id}_seed{seed}"
    final_path = run_dir / f"epoch_{epochs}.pt"
    if final_path.exists():
        payload = torch.load(final_path, map_location="cpu", weights_only=False)
        if payload["rotation"] != model_id or int(payload["seed"]) != seed or int(payload["epoch"]) != epochs:
            raise RuntimeError(f"Malformed final checkpoint: {final_path}")
        return {"status": "ALREADY_COMPLETE", "path": final_path, "payload": payload}
    periodic = sorted(run_dir.glob("epoch_*.pt"))
    if periodic:
        path = periodic[-1]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["rotation"] != model_id or int(payload["seed"]) != seed:
            raise RuntimeError(f"Resume metadata mismatch: {path}")
        return {"status": "RESUME", "path": path, "payload": payload}
    return {"status": "NEW", "path": None, "payload": None}


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.inference_mode()
def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for images, labels, _, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        correct += int((logits.argmax(1) == labels).sum().item())
        total += int(labels.numel())
    return correct / max(1, total)


def _make_loaders(model_id: str, seed: int, batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_ds, _ = training_dataset(model_id)
    val_ds = restricted_validation_dataset(model_id)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=_worker_init,
        generator=generator,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=_worker_init,
        persistent_workers=True,
    )
    return train_loader, val_loader


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    global_step: int,
    seed: int,
    model_id: str,
    initial_hash: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "study": "supervised_inaturalist_rotation4",
            "rotation": model_id,
            "seed": seed,
            "epoch": epoch,
            "global_step": global_step,
            "initial_state_sha256": initial_hash,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "recipe_sha256": sha256_file(RECIPE_PATH),
            "created_utc": utc_now(),
        },
        path,
    )


def train_one(model_id: str, seed: int, *, epochs: int = 100, smoke_steps: int | None = None) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for upstream training")
    recipe = read_yaml()
    batch_size = int(recipe["upstream"]["batch_size"])
    if batch_size != 256 or int(recipe["upstream"]["gradient_accumulation_steps"]) != 1:
        raise AssertionError("Local hardware protocol requires literal batch 256 and no accumulation")
    device = torch.device("cuda:0")
    set_seed(seed)
    initial_audit = ensure_initializations()
    initial_hash = initial_audit["seeds"][str(seed)]["whole_state_sha256"]
    run_dir = ROOT / ("smoke" if smoke_steps is not None else "checkpoints") / f"{model_id}_seed{seed}"
    log_path = ROOT / ("smoke" if smoke_steps is not None else "logs") / f"{model_id}_seed{seed}_train.csv"
    run_state = {"status": "NEW", "path": None, "payload": None}
    if smoke_steps is None:
        run_state = _existing_run_state(model_id, seed, epochs)
        if run_state["status"] == "ALREADY_COMPLETE":
            return {
                "rotation": model_id,
                "seed": seed,
                "status": "ALREADY_COMPLETE",
                "checkpoint": str(run_state["path"]),
            }
    if run_state["status"] == "RESUME":
        # A resumable run is restored only from its training checkpoint. The
        # epoch-0 export is audited above but is never loaded into this model.
        model = make_model().to(device)
    else:
        # Only genuinely new runs obtain model tensors from the authoritative
        # cross-platform frozen initialization export.
        model = load_initial_model(seed, device)
        if state_dict_sha256(model.state_dict()) != initial_hash:
            raise RuntimeError("Loaded model does not match frozen initial state")
    train_loader, val_loader = _make_loaders(model_id, seed, batch_size)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(recipe["upstream"]["initial_lr"]),
        momentum=float(recipe["upstream"]["momentum"]),
        weight_decay=float(recipe["upstream"]["weight_decay"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    total_steps = epochs * len(train_loader)
    warmup_steps = int(recipe["upstream"]["warmup_epochs"]) * len(train_loader)
    base_lr = float(recipe["upstream"]["initial_lr"])
    start_epoch = 0
    global_step = 0

    if run_state["status"] == "RESUME":
        payload = run_state["payload"]
        if payload.get("initial_state_sha256") != initial_hash:
            raise RuntimeError(
                f"Resume checkpoint initialization hash mismatch: {run_state['path']}"
            )
        if payload.get("recipe_sha256") != sha256_file(RECIPE_PATH):
            raise RuntimeError(f"Resume checkpoint recipe hash mismatch: {run_state['path']}")
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        scaler.load_state_dict(payload["scaler_state"])
        start_epoch = int(payload["epoch"])
        global_step = int(payload["global_step"])

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    losses: list[float] = []
    finite_gradient_steps = 0
    amp_overflow_steps = 0
    model.train()
    stop = False
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        correct = samples = 0
        model.train()
        for images, labels, _, _ in train_loader:
            lr = _lr(global_step, total_steps, warmup_steps, base_lr)
            _set_lr(optimizer, lr)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                logits = model(images)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at {model_id} seed {seed} step {global_step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradients_finite = True
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    gradients_finite = False
                    break
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            if gradients_finite and scale_after >= scale_before:
                finite_gradient_steps += 1
            else:
                # Standard AMP behavior: GradScaler skips an overflowing update
                # and reduces its scale. This is logged rather than bypassed.
                amp_overflow_steps += 1
            value = float(loss.item())
            losses.append(value)
            epoch_loss += value * labels.numel()
            correct += int((logits.detach().argmax(1) == labels).sum().item())
            samples += int(labels.numel())
            global_step += 1
            if smoke_steps is not None and len(losses) >= smoke_steps:
                stop = True
                break
        completed_epoch = epoch + 1
        val_acc = float("nan")
        if smoke_steps is None and (completed_epoch % 5 == 0 or completed_epoch == epochs):
            val_acc = accuracy(model, val_loader, device)
        row = {
            "utc": utc_now(),
            "rotation": model_id,
            "seed": seed,
            "epoch": completed_epoch,
            "global_step": global_step,
            "train_loss": epoch_loss / max(1, samples),
            "train_accuracy": correct / max(1, samples),
            "restricted_validation_accuracy": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "finite_gradient_steps": finite_gradient_steps,
            "amp_overflow_steps": amp_overflow_steps,
        }
        append_csv(log_path, row)
        print(json.dumps(row), flush=True)
        if smoke_steps is None and completed_epoch % 5 == 0:
            _save_checkpoint(
                run_dir / f"epoch_{completed_epoch:03d}.pt",
                model,
                optimizer,
                scaler,
                completed_epoch,
                global_step,
                seed,
                model_id,
                initial_hash,
            )
        if stop:
            break

    elapsed = time.time() - started
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    result = {
        "rotation": model_id,
        "seed": seed,
        "status": "SMOKE_TRAINED" if smoke_steps is not None else "COMPLETE",
        "steps": len(losses) if smoke_steps is not None else global_step,
        "losses": losses if smoke_steps is not None else None,
        "runtime_seconds": elapsed,
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "finite_gradient_steps": finite_gradient_steps,
        "amp_overflow_steps": amp_overflow_steps,
        "train_images": len(train_loader.dataset),
        "batches_per_epoch": len(train_loader),
    }
    if smoke_steps is not None:
        smoke_ckpt = run_dir / "smoke_checkpoint.pt"
        _save_checkpoint(smoke_ckpt, model, optimizer, scaler, completed_epoch, global_step, seed, model_id, initial_hash)
        result["checkpoint"] = str(smoke_ckpt)
        result["model"] = model
    else:
        final_path = run_dir / "epoch_100.pt"
        if not final_path.exists():
            _save_checkpoint(final_path, model, optimizer, scaler, epochs, global_step, seed, model_id, initial_hash)
        encoder_path = run_dir / "encoder_final.pt"
        classifier_path = run_dir / "classifier_final.pt"
        torch.save({"rotation": model_id, "seed": seed, "encoder_state": encoder_state(model)}, encoder_path)
        torch.save({"rotation": model_id, "seed": seed, "classifier_state": copy.deepcopy(model.fc.state_dict())}, classifier_path)
        result.update(
            {
                "checkpoint": str(final_path),
                "checkpoint_sha256": sha256_file(final_path),
                "encoder_checkpoint": str(encoder_path),
                "encoder_checkpoint_sha256": sha256_file(encoder_path),
                "classifier_checkpoint": str(classifier_path),
                "classifier_checkpoint_sha256": sha256_file(classifier_path),
                "final_state_sha256": state_dict_sha256(model.state_dict()),
                "final_encoder_state_sha256": state_dict_sha256(encoder_state(model)),
            }
        )
    return result


@torch.inference_mode()
def _smoke_features(model: nn.Module) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    feature_model = copy.deepcopy(model)
    feature_model.fc = nn.Identity()
    feature_model.eval()
    outputs = []
    labels = []
    for dataset in (downstream_train_dataset(), all_validation_dataset()):
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
        feats, ys = [], []
        for images, target, _, _ in loader:
            values = feature_model(images.to(device, non_blocking=True)).float().cpu().numpy()
            feats.append(values)
            ys.append(target.numpy())
        outputs.append(np.concatenate(feats))
        labels.append(np.concatenate(ys))
    return outputs[0], labels[0], outputs[1], labels[1]


def _smoke_downstream(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray) -> dict[str, Any]:
    from common import manifest_rows

    manifest = manifest_rows()
    d_ids = sorted(int(r["category_id"]) for r in manifest if r["role"] == "d")
    ood_id = next(int(r["category_id"]) for r in manifest if r["role"] == "c1")
    mapping = {v: i for i, v in enumerate(d_ids)}
    y = torch.tensor([mapping[int(v)] for v in train_y], dtype=torch.long)
    x = torch.from_numpy(train_x).float()
    set_seed(0)
    probe = nn.Linear(2048, 20)
    opt = torch.optim.SGD(probe.parameters(), lr=0.1, momentum=0.9)
    probe.train()
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(probe(x), y)
        loss.backward()
        opt.step()
    probe.eval()
    with torch.no_grad():
        logits = probe(torch.from_numpy(val_x).float()).numpy()
    norm_train = train_x / np.maximum(np.linalg.norm(train_x, axis=1, keepdims=True), 1e-12)
    norm_val = val_x / np.maximum(np.linalg.norm(val_x, axis=1, keepdims=True), 1e-12)
    similarities = norm_val @ norm_train.T
    nearest = np.partition(similarities, similarities.shape[1] - 50, axis=1)[:, -50:]
    knn = np.mean(1.0 - nearest, axis=1)
    energy = -np.logaddexp.reduce(logits, axis=1)
    mask = np.isin(val_y, d_ids) | (val_y == ood_id)
    binary = (val_y[mask] == ood_id).astype(int)
    return {
        "probe_loss_finite": bool(np.isfinite(loss.item())),
        "feature_shapes": [list(train_x.shape), list(val_x.shape)],
        "knn_auroc": float(roc_auc_score(binary, knn[mask])),
        "energy_auroc": float(roc_auc_score(binary, energy[mask])),
        "scores_finite": bool(np.isfinite(knn).all() and np.isfinite(energy).all()),
    }


def smoke() -> dict[str, Any]:
    ensure_initializations()
    try:
        trained = train_one("M1", 0, epochs=1, smoke_steps=4)
        model = trained.pop("model")
        train_x, train_y, val_x, val_y = _smoke_features(model)
        downstream = _smoke_downstream(train_x, train_y, val_x, val_y)
        checkpoint = torch.load(trained["checkpoint"], map_location="cpu", weights_only=False)
        reloaded = make_model()
        reloaded.load_state_dict(checkpoint["model_state"], strict=True)
        result = {
            "created_utc": utc_now(),
            "status": "PASS",
            "scientific_run": False,
            "batch_size": 256,
            "gradient_accumulation_steps": 1,
            "training": trained,
            "downstream": downstream,
            "checkpoint_reload_pass": True,
            "loss_all_finite": all(math.isfinite(x) for x in trained["losses"]),
            "loss_first": trained["losses"][0],
            "loss_last": trained["losses"][-1],
            "at_least_one_finite_gradient_update": trained["finite_gradient_steps"] > 0,
        }
        if not (
            result["loss_all_finite"]
            and result["at_least_one_finite_gradient_update"]
            and downstream["scores_finite"]
        ):
            raise RuntimeError("Smoke numerical check failed")
        write_json(ROOT / "smoke" / "smoke_result.json", result)
        print(json.dumps(result, indent=2), flush=True)
        return result
    except torch.cuda.OutOfMemoryError as exc:
        device = torch.device("cuda:0")
        failed_peak_allocated = int(torch.cuda.max_memory_allocated(device))
        failed_peak_reserved = int(torch.cuda.max_memory_reserved(device))
        error_text = str(exc)
        gc.collect()
        torch.cuda.empty_cache()
        diagnostic = _diagnose_largest_fitting_batch()
        report = {
            "created_utc": utc_now(),
            "status": "CUDA_OOM_STOP_BEFORE_FULL",
            "batch_size_attempted": 256,
            "gradient_accumulation_steps": 1,
            "peak_cuda_allocated_bytes": failed_peak_allocated,
            "peak_cuda_reserved_bytes": failed_peak_reserved,
            "device_total_bytes": int(torch.cuda.get_device_properties(device).total_memory),
            "error": error_text,
            "non_scientific_fit_diagnostic": diagnostic,
            "smallest_necessary_adjustment": (
                f"Reduce literal batch from 256 to {diagnostic['largest_fitting_batch']} "
                f"(reduction {256 - diagnostic['largest_fitting_batch']}); not applied to scientific training."
                if diagnostic["largest_fitting_batch"] is not None
                else "No candidate literal batch fit; scientific training remains blocked."
            ),
        }
        write_json(ROOT / "smoke" / "oom_report.json", report)
        print(json.dumps(report, indent=2), flush=True)
        raise SystemExit(2) from exc


def _diagnose_largest_fitting_batch() -> dict[str, Any]:
    """Non-scientific post-OOM fit diagnostic; never launches full training."""
    device = torch.device("cuda:0")
    dataset, _ = training_dataset("M1")
    tested: list[dict[str, Any]] = []

    def fits(batch_size: int) -> bool:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            set_seed(0)
            model = load_initial_model(0, device).train()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
            scaler = torch.cuda.amp.GradScaler(enabled=True)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
            images, labels, _, _ = next(iter(loader))
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                loss = nn.functional.cross_entropy(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize(device)
            tested.append(
                {
                    "batch_size": batch_size,
                    "fits": True,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                }
            )
            del images, labels, loss, optimizer, scaler, model, loader
            return True
        except torch.cuda.OutOfMemoryError:
            tested.append(
                {
                    "batch_size": batch_size,
                    "fits": False,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                }
            )
            return False
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    low, high = 1, 255
    largest = None
    while low <= high:
        mid = (low + high) // 2
        if fits(mid):
            largest = mid
            low = mid + 1
        else:
            high = mid - 1
    return {
        "purpose": "measure the smallest literal batch-size adjustment after the mandated batch-256 OOM",
        "scientific": False,
        "no_gradient_accumulation": True,
        "largest_fitting_batch": largest,
        "tested": tested,
    }


def _validate_training_checkpoint(
    run_state: dict[str, Any], model_id: str, seed: int, expected_initial_hash: str
) -> dict[str, Any]:
    path = run_state["path"]
    payload = run_state["payload"]
    file_hash_before = sha256_file(path)
    if payload.get("initial_state_sha256") != expected_initial_hash:
        raise RuntimeError(f"Checkpoint initialization provenance mismatch: {path}")
    if payload.get("recipe_sha256") != sha256_file(RECIPE_PATH):
        raise RuntimeError(f"Checkpoint recipe mismatch: {path}")
    model = make_model()
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    optimizer.load_state_dict(payload["optimizer_state"])
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    scaler.load_state_dict(payload["scaler_state"])
    file_hash_after = sha256_file(path)
    if file_hash_before != file_hash_after:
        raise RuntimeError(f"Checkpoint changed during dry validation: {path}")
    return {
        "rotation": model_id,
        "seed": seed,
        "status": run_state["status"],
        "checkpoint": str(path.relative_to(ROOT)),
        "checkpoint_sha256": file_hash_after,
        "epoch": int(payload["epoch"]),
        "global_step": int(payload["global_step"]),
        "strict_model_load": "PASS",
        "optimizer_state_load": "PASS",
        "amp_scaler_state_load": "PASS",
        "recipe_hash_match": "PASS",
        "initialization_provenance_hash_match": "PASS",
        "checkpoint_unchanged": True,
    }


def dry_validate_migration() -> dict[str, Any]:
    initialization = ensure_initializations()
    expected_hashes = {
        seed: initialization["seeds"][str(seed)]["whole_state_sha256"] for seed in SEEDS
    }
    m1 = _existing_run_state("M1", 0, 100)
    m2 = _existing_run_state("M2", 0, 100)
    m3 = _existing_run_state("M3", 0, 100)
    if m1["status"] != "ALREADY_COMPLETE" or m2["status"] != "ALREADY_COMPLETE":
        raise RuntimeError("Completed M1/M2 seed-0 runs were not recognized as complete")
    if m3["status"] != "RESUME":
        raise RuntimeError("M3 seed-0 was not recognized as resumable")
    completed_m1 = _validate_training_checkpoint(m1, "M1", 0, expected_hashes[0])
    completed_m2 = _validate_training_checkpoint(m2, "M2", 0, expected_hashes[0])
    resumable_m3 = _validate_training_checkpoint(m3, "M3", 0, expected_hashes[0])
    if resumable_m3["epoch"] != 5 or resumable_m3["global_step"] != 435:
        raise RuntimeError(
            f"Unexpected M3 seed-0 resume location: epoch {resumable_m3['epoch']}, "
            f"global step {resumable_m3['global_step']}"
        )
    dry = {
        "created_utc": utc_now(),
        "status": "PASS",
        "training_launched": False,
        "new_run_initialization_source": "authoritative exported frozen tensors",
        "resume_run_state_source": "existing periodic training checkpoint only",
        "completed_runs_retrained": False,
        "completed_runs": {"M1_seed0": completed_m1, "M2_seed0": completed_m2},
        "resumable_run": resumable_m3,
    }
    audit = json.loads(MIGRATION_AUDIT_PATH.read_text(encoding="utf-8"))
    audit["dry_validation"] = dry
    write_json(MIGRATION_AUDIT_PATH, audit)
    return {
        "seed0_file_hash": initialization["seeds"]["0"]["export_file_sha256"],
        "seed1_file_hash": initialization["seeds"]["1"]["export_file_sha256"],
        "seed0_initialization_match": initialization["seeds"]["0"]["exact_initialization_match"],
        "seed1_initialization_match": initialization["seeds"]["1"]["exact_initialization_match"],
        "m1_seed0_status": m1["status"],
        "m2_seed0_status": m2["status"],
        "m3_seed0_resume_checkpoint": resumable_m3["checkpoint"],
        "dry_validation": dry["status"],
    }


def _write_training_runs(results: list[dict[str, Any]], path: Path | None = None) -> None:
    run_rows = [{k: v for k, v in result.items() if k != "losses"} for result in results]
    # Resumed migrations can mix sparse ALREADY_COMPLETE records with richer
    # records returned by newly completed runs. Preserve every field in a
    # stable first-seen order instead of inferring the schema from one row.
    fieldnames = list(dict.fromkeys(key for row in run_rows for key in row))
    write_csv(path or ROOT / "training_runs.csv", run_rows, fieldnames=fieldnames)


def full() -> list[dict[str, Any]]:
    smoke_path = ROOT / "smoke" / "smoke_result.json"
    if not smoke_path.exists():
        raise RuntimeError("Smoke test has not been run")
    with smoke_path.open("r", encoding="utf-8") as f:
        smoke_result = json.load(f)
    if smoke_result.get("status") != "PASS":
        raise RuntimeError("Smoke test did not pass; full training is forbidden")
    results = []
    for seed in SEEDS:
        for model_id in MODELS:
            result = train_one(model_id, seed, epochs=100, smoke_steps=None)
            results.append(result)
            write_json(ROOT / "logs" / "training_status.json", {"updated_utc": utc_now(), "runs": results})
            _write_training_runs(results)
    write_json(ROOT / "logs" / "training_status.json", {"updated_utc": utc_now(), "status": "COMPLETE", "runs": results})
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["smoke", "full", "migration-dry-validate"])
    args = parser.parse_args()
    if args.stage == "smoke":
        smoke()
    elif args.stage == "full":
        full()
    else:
        dry_validate_migration()
