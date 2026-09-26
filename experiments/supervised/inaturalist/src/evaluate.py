from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import (
    MODELS,
    ROOT,
    SEEDS,
    manifest_rows,
    set_seed,
    sha256_file,
    state_dict_sha256,
    utc_now,
    write_csv,
    write_json,
)
from data import all_validation_dataset, downstream_train_dataset, restricted_validation_dataset, upstream_train_eval_dataset
from train import make_model


def _final_checkpoint(model_id: str, seed: int) -> Path:
    path = ROOT / "checkpoints" / f"{model_id}_seed{seed}" / "epoch_100.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing final upstream checkpoint: {path}")
    return path


def _load_model(model_id: str, seed: int, device: torch.device) -> tuple[nn.Module, dict[str, Any], str]:
    path = _final_checkpoint(model_id, seed)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["rotation"] != model_id or int(payload["seed"]) != seed or int(payload["epoch"]) != 100:
        raise RuntimeError(f"Final checkpoint metadata mismatch: {path}")
    model = make_model()
    model.load_state_dict(payload["model_state"], strict=True)
    model_hash = state_dict_sha256(model.state_dict())
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload, model_hash


@torch.inference_mode()
def _classification_metrics(model: nn.Module, dataset, device: torch.device) -> tuple[dict[str, float], dict[int, tuple[int, int]]]:
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    model.eval()
    total = top1 = top5 = 0
    loss_sum = 0.0
    per_label: dict[int, list[int]] = {}
    for images, target, _, _ in loader:
        target = target.to(device, non_blocking=True)
        logits = model(images.to(device, non_blocking=True))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite upstream logits")
        loss_sum += float(nn.functional.cross_entropy(logits, target, reduction="sum").item())
        prediction = logits.argmax(1)
        hit = prediction.eq(target)
        top1 += int(hit.sum().item())
        top5 += int(logits.topk(5, dim=1).indices.eq(target[:, None]).any(1).sum().item())
        total += int(target.numel())
        for label, ok in zip(target.cpu().tolist(), hit.cpu().tolist()):
            values = per_label.setdefault(int(label), [0, 0])
            values[0] += int(ok)
            values[1] += 1
    return {"top1": top1 / total, "top5": top5 / total, "cross_entropy": loss_sum / total, "n": total}, {
        key: (value[0], value[1]) for key, value in per_label.items()
    }


def evaluate_upstream_classification() -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for upstream classification evaluation")
    from common import read_csv

    device = torch.device("cuda:0")
    run_rows, species_rows = [], []
    for seed in SEEDS:
        for model_id in MODELS:
            model, _, before = _load_model(model_id, seed, device)
            train_metrics, _ = _classification_metrics(model, upstream_train_eval_dataset(model_id), device)
            val_metrics, per_label = _classification_metrics(model, restricted_validation_dataset(model_id), device)
            after = state_dict_sha256(model.state_dict())
            if before != after:
                raise RuntimeError(f"Model mutated during upstream evaluation: {model_id} seed {seed}")
            rotation = {int(r["local_training_label"]): r for r in read_csv(ROOT / f"rotation_{model_id}.csv")}
            macro = float(np.mean([correct / count for correct, count in per_label.values()]))
            if abs(macro - val_metrics["top1"]) > 1e-12:
                raise AssertionError("Balanced validation macro and micro accuracy disagree")
            run_rows.append({
                "seed": seed, "rotation": model_id, "train_images": int(train_metrics["n"]),
                "train_top1": train_metrics["top1"], "train_top5": train_metrics["top5"],
                "train_cross_entropy": train_metrics["cross_entropy"], "validation_images": int(val_metrics["n"]),
                "validation_top1": val_metrics["top1"], "validation_macro_top1": macro,
                "validation_top5": val_metrics["top5"], "validation_cross_entropy": val_metrics["cross_entropy"],
                "model_state_unchanged": True,
            })
            for label, (correct, count) in sorted(per_label.items()):
                source = rotation[label]
                species_rows.append({"seed": seed, "rotation": model_id, "local_label": label,
                                     "category_id": source["category_id"], "scientific_name": source["scientific_name"],
                                     "correct": correct, "n": count, "top1": correct / count})
            del model
            torch.cuda.empty_cache()
    write_csv(ROOT / "upstream_classification_runs.csv", run_rows)
    write_csv(ROOT / "upstream_per_species_accuracy.csv", species_rows)
    return run_rows


@torch.inference_mode()
def _extract(model: nn.Module, dataset, device: torch.device) -> dict[str, Any]:
    classifier = model.fc
    model.fc = nn.Identity()
    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    features, labels, image_ids, paths = [], [], [], []
    try:
        for images, target, ids, names in loader:
            values = model(images.to(device, non_blocking=True)).float().cpu().numpy()
            features.append(values)
            labels.append(target.numpy())
            image_ids.append(ids.numpy())
            paths.extend(list(names))
    finally:
        model.fc = classifier
    array = np.concatenate(features).astype(np.float32, copy=False)
    if array.shape[1] != 2048 or not np.isfinite(array).all():
        raise RuntimeError(f"Invalid frozen feature tensor: {array.shape}")
    return {
        "features": array,
        "category_ids": np.concatenate(labels).astype(np.int64),
        "image_ids": np.concatenate(image_ids).astype(np.int64),
        "paths": np.asarray(paths),
    }


def extract_all_features() -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for frozen feature extraction")
    device = torch.device("cuda:0")
    output_records = []
    train_ds = downstream_train_dataset()
    val_ds = all_validation_dataset()
    expected_train_ids = None
    expected_val_ids = None
    for seed in SEEDS:
        for model_id in MODELS:
            out_path = ROOT / "frozen_features" / f"{model_id}_seed{seed}.npz"
            meta_path = ROOT / "frozen_features" / f"{model_id}_seed{seed}.json"
            if out_path.is_file() and meta_path.is_file():
                with meta_path.open("r", encoding="utf-8") as f:
                    record = json.load(f)
                if record["rotation"] != model_id or int(record["seed"]) != seed:
                    raise RuntimeError(f"Feature cache metadata mismatch: {meta_path}")
                output_records.append(record)
                continue
            model, _, model_hash = _load_model(model_id, seed, device)
            before_hash = state_dict_sha256(model.state_dict())
            train = _extract(model, train_ds, device)
            val = _extract(model, val_ds, device)
            after_hash = state_dict_sha256(model.state_dict())
            if before_hash != after_hash or before_hash != model_hash:
                raise RuntimeError(f"Encoder/model state mutated during feature extraction: {model_id} seed {seed}")
            if train["features"].shape != (1000, 2048) or val["features"].shape != (1000, 2048):
                raise RuntimeError(f"Unexpected feature shapes for {model_id} seed {seed}")
            if expected_train_ids is None:
                expected_train_ids = train["image_ids"].copy()
                expected_val_ids = val["image_ids"].copy()
            if not np.array_equal(train["image_ids"], expected_train_ids) or not np.array_equal(val["image_ids"], expected_val_ids):
                raise RuntimeError("Evaluation image IDs are not fixed across model/seed conditions")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                out_path,
                train_features=train["features"],
                train_category_ids=train["category_ids"],
                train_image_ids=train["image_ids"],
                train_paths=train["paths"],
                val_features=val["features"],
                val_category_ids=val["category_ids"],
                val_image_ids=val["image_ids"],
                val_paths=val["paths"],
            )
            record = {
                "created_utc": utc_now(),
                "rotation": model_id,
                "seed": seed,
                "source_checkpoint": str(_final_checkpoint(model_id, seed)),
                "source_checkpoint_sha256": sha256_file(_final_checkpoint(model_id, seed)),
                "model_state_sha256_before": before_hash,
                "model_state_sha256_after": after_hash,
                "state_unchanged": True,
                "feature_file": str(out_path),
                "feature_file_sha256": sha256_file(out_path),
                "train_shape": list(train["features"].shape),
                "validation_shape": list(val["features"].shape),
                "train_finite": True,
                "validation_finite": True,
            }
            write_json(meta_path, record)
            output_records.append(record)
            del model
            torch.cuda.empty_cache()
    write_json(ROOT / "frozen_features" / "inventory.json", output_records)
    return output_records


def _load_features(model_id: str, seed: int) -> dict[str, np.ndarray]:
    path = ROOT / "frozen_features" / f"{model_id}_seed{seed}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _probe_initial(seed: int) -> tuple[dict[str, torch.Tensor], str]:
    set_seed(seed)
    probe = nn.Linear(2048, 20, bias=True)
    state = copy.deepcopy(probe.state_dict())
    return state, state_dict_sha256(state)


def _train_probe(model_id: str, seed: int, features: dict[str, np.ndarray]) -> tuple[nn.Module, dict[str, Any]]:
    manifest = manifest_rows()
    d_ids = sorted(int(r["category_id"]) for r in manifest if r["role"] == "d")
    label_map = {value: index for index, value in enumerate(d_ids)}
    y_values = features["train_category_ids"]
    if set(map(int, np.unique(y_values))) != set(d_ids):
        raise RuntimeError("Probe feature bank is not exactly the 20 d species")
    x = torch.from_numpy(features["train_features"]).float()
    y = torch.tensor([label_map[int(v)] for v in y_values], dtype=torch.long)
    initial_state, initial_hash = _probe_initial(seed)
    probe = nn.Linear(2048, 20, bias=True)
    probe.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.SGD(probe.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(x, y), batch_size=512, shuffle=True, generator=generator, num_workers=0)
    logs = []
    for epoch in range(50):
        probe.train()
        loss_sum = 0.0
        correct = count = 0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = probe(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite probe loss: {model_id} seed {seed}")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * yb.numel()
            correct += int((logits.detach().argmax(1) == yb).sum().item())
            count += int(yb.numel())
        scheduler.step()
        logs.append({"epoch": epoch + 1, "loss": loss_sum / count, "accuracy": correct / count, "lr": scheduler.get_last_lr()[0]})
    probe.eval()
    path = ROOT / "probes" / f"probe_{model_id}_seed{seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "rotation": model_id,
            "upstream_seed": seed,
            "probe_seed": seed,
            "initial_state_sha256": initial_hash,
            "model_state": probe.state_dict(),
            "recipe": {
                "input_dim": 2048,
                "classes": 20,
                "epochs": 50,
                "batch_size": 512,
                "optimizer": "SGD",
                "lr": 0.1,
                "momentum": 0.9,
                "weight_decay": 0.0,
                "schedule": "CosineAnnealingLR(T_max=50), stepped by epoch",
            },
            "logs": logs,
        },
        path,
    )
    record = {
        "rotation": model_id,
        "seed": seed,
        "probe_seed": seed,
        "initial_state_sha256": initial_hash,
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "final_train_loss": logs[-1]["loss"],
        "final_train_accuracy": logs[-1]["accuracy"],
    }
    return probe, record


def _knn_scores(train: np.ndarray, query: np.ndarray, k: int = 50) -> np.ndarray:
    train_norm = train / np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
    query_norm = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    similarity = query_norm @ train_norm.T
    nearest = np.partition(similarity, similarity.shape[1] - k, axis=1)[:, -k:]
    return np.mean(1.0 - nearest, axis=1).astype(np.float64)


def evaluate_all() -> list[dict[str, Any]]:
    evaluate_upstream_classification()
    extract_all_features()
    manifest = manifest_rows()
    d_ids = {int(r["category_id"]) for r in manifest if r["role"] == "d"}
    ood_rows = [r for r in manifest if r["role"] != "d"]
    auroc_rows = []
    probe_records = []
    fixed_train_ids = fixed_val_ids = None
    for seed in SEEDS:
        init_hashes = []
        for model_id in MODELS:
            data = _load_features(model_id, seed)
            if fixed_train_ids is None:
                fixed_train_ids = data["train_image_ids"].copy()
                fixed_val_ids = data["val_image_ids"].copy()
            if not np.array_equal(data["train_image_ids"], fixed_train_ids) or not np.array_equal(data["val_image_ids"], fixed_val_ids):
                raise RuntimeError("Feature-cache sample IDs vary across conditions")
            probe, probe_record = _train_probe(model_id, seed, data)
            probe_records.append(probe_record)
            init_hashes.append(probe_record["initial_state_sha256"])
            with torch.inference_mode():
                logits = probe(torch.from_numpy(data["val_features"]).float()).numpy()
            knn = _knn_scores(data["train_features"], data["val_features"], k=50)
            energy = -np.logaddexp.reduce(logits.astype(np.float64), axis=1)
            shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
            softmax = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
            msp = 1.0 - softmax.max(axis=1)
            if not (np.isfinite(knn).all() and np.isfinite(energy).all() and np.isfinite(msp).all()):
                raise FloatingPointError(f"Non-finite OOD scores: {model_id} seed {seed}")
            scores_path = ROOT / "scores" / f"{model_id}_seed{seed}.npz"
            scores_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                scores_path,
                image_ids=data["val_image_ids"],
                category_ids=data["val_category_ids"],
                knn=knn,
                energy=energy,
                msp=msp,
                logits=logits.astype(np.float32),
            )
            categories = data["val_category_ids"]
            id_mask = np.isin(categories, list(d_ids))
            if int(id_mask.sum()) != 200:
                raise AssertionError("ID validation sample count is not 200")
            for row in ood_rows:
                category_id = int(row["category_id"])
                ood_mask = categories == category_id
                if int(ood_mask.sum()) != 10:
                    raise AssertionError(f"OOD sample count is not 10 for category {category_id}")
                mask = id_mask | ood_mask
                y = ood_mask[mask].astype(np.int64)
                values = {
                    "auroc_knn": float(roc_auc_score(y, knn[mask])),
                    "auroc_energy": float(roc_auc_score(y, energy[mask])),
                    "auroc_msp": float(roc_auc_score(y, msp[mask])),
                }
                if not all(0.0 <= v <= 1.0 for v in values.values()):
                    raise AssertionError(f"AUROC outside [0,1]: {values}")
                auroc_rows.append(
                    {
                        "seed": seed,
                        "rotation": model_id,
                        "genus_group_id": row["group_id"],
                        "genus_name": row["genus"],
                        "official_category_id": category_id,
                        "scientific_species_name": row["scientific_name"],
                        "common_name": row["common_name"],
                        "role": row["role"],
                        "withheld_rotation": row["withheld_model"],
                        "provenance_state": "withheld" if row["withheld_model"] == model_id else "supervised",
                        **values,
                        "id_validation_images": 200,
                        "ood_validation_images": 10,
                        "score_file": str(scores_path),
                    }
                )
        if len(set(init_hashes)) != 1:
            raise AssertionError(f"Probe initialization was not matched across M1-M4 for upstream seed {seed}")
    write_csv(ROOT / "per_state_aurocs.csv", auroc_rows)
    write_json(ROOT / "probe_training_audit.json", {"created_utc": utc_now(), "matched_within_seed": True, "runs": probe_records})
    return auroc_rows


if __name__ == "__main__":
    print(json.dumps({"rows": len(evaluate_all()), "status": "COMPLETE"}, indent=2))
