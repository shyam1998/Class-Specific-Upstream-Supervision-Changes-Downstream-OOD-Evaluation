from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .common import (MODELS, ROOT, SEEDS, assert_protected_unchanged, atomic_torch_save,
                     manifest_rows, now, read_json, sha256_file, state_dict_sha256,
                     torch_load, write_csv, write_json)
from .data import downstream_eval_dataset, downstream_reference_dataset
from .model import make_backbone
from .train import _checkpoint_path, _checkpoint_valid


def _feature_paths(rotation: str, seed: int):
    return ROOT / "features" / f"{rotation}_seed{seed}.npz", ROOT / "features" / f"{rotation}_seed{seed}.json"


def _extract_dataset(encoder, dataset):
    import torch
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4,
                        pin_memory=True, persistent_workers=True)
    features, categories, image_ids, paths = [], [], [], []
    with torch.inference_mode():
        for images, category, identifiers, names in loader:
            value = encoder(images.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)).float().cpu()
            if value.ndim != 2 or value.shape[1] != 2048 or not torch.isfinite(value).all():
                raise RuntimeError("Invalid 2048-D pre-projector feature tensor")
            features.append(value.numpy()); categories.append(category.numpy()); image_ids.append(identifiers.numpy()); paths.extend(names)
    return {"features": np.concatenate(features).astype(np.float32, copy=False),
            "category_ids": np.concatenate(categories).astype(np.int64, copy=False),
            "image_ids": np.concatenate(image_ids).astype(np.int64, copy=False),
            "paths": np.asarray(paths, dtype=str)}


def extract_features(rotation: str, seed: int):
    import torch
    feature_path, metadata_path = _feature_paths(rotation, seed)
    if feature_path.is_file() and metadata_path.is_file():
        metadata = read_json(metadata_path)
        if sha256_file(feature_path) != metadata["feature_file_sha256"]:
            raise RuntimeError(f"Feature cache hash mismatch: {feature_path}")
        return feature_path
    checkpoint_path = _checkpoint_path(rotation, seed, 200)
    checkpoint = _checkpoint_valid(checkpoint_path, rotation, seed, required_epoch=200)
    encoder = make_backbone(); encoder.load_state_dict(checkpoint["encoder_state"], strict=True)
    before = state_dict_sha256(encoder.state_dict())
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.cuda().eval()
    reference = _extract_dataset(encoder, downstream_reference_dataset())
    evaluation = _extract_dataset(encoder, downstream_eval_dataset())
    after = state_dict_sha256(encoder.state_dict())
    if before != after:
        raise RuntimeError("Encoder parameter or BatchNorm-buffer mutation during extraction")
    if reference["features"].shape != (1000, 2048) or evaluation["features"].shape != (1000, 2048):
        raise RuntimeError("Unexpected frozen feature-cache shape")
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(feature_path,
             reference_features=reference["features"], reference_category_ids=reference["category_ids"],
             reference_image_ids=reference["image_ids"], reference_paths=reference["paths"],
             eval_features=evaluation["features"], eval_category_ids=evaluation["category_ids"],
             eval_image_ids=evaluation["image_ids"], eval_paths=evaluation["paths"])
    metadata = {"created_utc": now(), "rotation": rotation, "seed": seed,
                "source_checkpoint": str(checkpoint_path), "source_checkpoint_sha256": sha256_file(checkpoint_path),
                "encoder_sha256_before": before, "encoder_sha256_after": after,
                "parameters_and_bn_unchanged": True, "feature_dim": 2048,
                "reference_shape": list(reference["features"].shape), "evaluation_shape": list(evaluation["features"].shape),
                "feature_file": str(feature_path), "feature_file_sha256": sha256_file(feature_path)}
    write_json(metadata_path, metadata)
    del encoder
    torch.cuda.empty_cache()
    return feature_path


def extract_all_features():
    assert_protected_unchanged()
    identity = None
    records = []
    for seed in SEEDS:
        for rotation in MODELS:
            path = extract_features(rotation, seed)
            with np.load(path, allow_pickle=False) as data:
                signature = (tuple(data["reference_image_ids"].tolist()), tuple(data["eval_image_ids"].tolist()))
            if identity is None:
                identity = signature
            elif signature != identity:
                raise RuntimeError("Downstream image identities differ across rotations/seeds")
            records.append(read_json(path.with_suffix(".json")))
    write_json(ROOT / "features" / "inventory.json", {"status": "PASS", "fixed_image_identity": True, "runs": records})


def _load_features(rotation: str, seed: int):
    path, metadata_path = _feature_paths(rotation, seed)
    metadata = read_json(metadata_path)
    if sha256_file(path) != metadata["feature_file_sha256"]:
        raise RuntimeError(f"Feature cache changed: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _probe_initial(seed: int):
    import torch
    from torch import nn
    torch.manual_seed(10000 + seed); torch.cuda.manual_seed_all(10000 + seed)
    probe = nn.Linear(2048, 20, bias=True)
    state = copy.deepcopy(probe.state_dict())
    return state, state_dict_sha256(state)


def train_probe(rotation: str, seed: int, data):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    output = ROOT / "probes" / f"{rotation}_seed{seed}.pt"
    if output.is_file():
        payload = torch_load(output)
        if payload["rotation"] != rotation or int(payload["upstream_seed"]) != seed:
            raise RuntimeError(f"Probe metadata mismatch: {output}")
        return payload, output
    d_rows = sorted((row for row in manifest_rows() if row["role"] == "d"), key=lambda row: int(row["category_id"]))
    d_ids = [int(row["category_id"]) for row in d_rows]
    mapping = {category: index for index, category in enumerate(d_ids)}
    if set(data["reference_category_ids"].tolist()) != set(d_ids):
        raise RuntimeError("Probe bank is not exactly the 20 d species")
    x = torch.from_numpy(data["reference_features"]).float()
    y = torch.tensor([mapping[int(value)] for value in data["reference_category_ids"]], dtype=torch.long)
    initial, initial_hash = _probe_initial(seed)
    probe = nn.Linear(2048, 20, bias=True); probe.load_state_dict(initial, strict=True)
    optimizer = torch.optim.SGD(probe.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.0)
    loader = DataLoader(TensorDataset(x, y), batch_size=512, shuffle=True,
                        generator=torch.Generator().manual_seed(10000 + seed), num_workers=0)
    logs = []
    for epoch in range(50):
        probe.train(); loss_sum = 0.0; correct = total = 0
        for features, target in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = probe(features); loss = nn.functional.cross_entropy(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite probe loss")
            loss.backward(); optimizer.step()
            loss_sum += float(loss) * len(target); correct += int((logits.detach().argmax(1) == target).sum()); total += len(target)
        scheduler.step()
        logs.append({"epoch": epoch + 1, "loss": loss_sum / total, "train_accuracy": correct / total,
                     "lr_after_epoch": scheduler.get_last_lr()[0]})
    probe.eval()
    eval_ids = data["eval_category_ids"]
    mask = np.isin(eval_ids, d_ids)
    targets = torch.tensor([mapping[int(value)] for value in eval_ids[mask]], dtype=torch.long)
    with torch.inference_mode():
        predictions = probe(torch.from_numpy(data["eval_features"][mask]).float()).argmax(1)
    accuracy = float((predictions == targets).float().mean())
    payload = {"rotation": rotation, "upstream_seed": seed, "probe_seed": 10000 + seed,
               "initial_state_sha256": initial_hash, "final_state_sha256": state_dict_sha256(probe.state_dict()),
               "model_state": probe.state_dict(), "downstream_id_eval_accuracy": accuracy,
               "recipe": {"input_dim": 2048, "classes": 20, "epochs": 50, "batch_size": 512,
                          "optimizer": "SGD", "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0,
                          "scheduler": "CosineAnnealingLR(T_max=50), stepped by epoch"}, "logs": logs}
    atomic_torch_save(payload, output)
    payload["checkpoint_sha256"] = sha256_file(output)
    return payload, output


def _knn_scores(reference, query, k=50):
    reference = reference / np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-12)
    query = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    similarity = query @ reference.T
    nearest = np.partition(similarity, similarity.shape[1] - k, axis=1)[:, -k:]
    return np.mean(1.0 - nearest, axis=1).astype(np.float64)


def evaluate_all():
    import torch
    assert_protected_unchanged()
    extract_all_features()
    manifest = manifest_rows()
    d_ids = {int(row["category_id"]) for row in manifest if row["role"] == "d"}
    candidates = [row for row in manifest if row["role"] != "d"]
    state_rows, probe_rows = [], []
    fixed_signature = None
    for seed in SEEDS:
        hashes = []
        for rotation in MODELS:
            data = _load_features(rotation, seed)
            signature = (tuple(data["reference_image_ids"]), tuple(data["eval_image_ids"]))
            if fixed_signature is None: fixed_signature = signature
            elif signature != fixed_signature: raise RuntimeError("Evaluation identity changed across conditions")
            probe_payload, probe_path = train_probe(rotation, seed, data)
            hashes.append(probe_payload["initial_state_sha256"])
            probe = torch.nn.Linear(2048, 20); probe.load_state_dict(probe_payload["model_state"]); probe.eval()
            with torch.inference_mode():
                logits = probe(torch.from_numpy(data["eval_features"]).float()).numpy().astype(np.float64)
            knn = _knn_scores(data["reference_features"], data["eval_features"], 50)
            energy = -np.logaddexp.reduce(logits, axis=1)
            shifted = logits - logits.max(axis=1, keepdims=True)
            probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
            msp = 1.0 - probabilities.max(axis=1)
            if not (np.isfinite(knn).all() and np.isfinite(energy).all() and np.isfinite(msp).all()):
                raise RuntimeError("Non-finite detector score")
            score_path = ROOT / "features" / f"scores_{rotation}_seed{seed}.npz"
            np.savez(score_path, image_ids=data["eval_image_ids"], category_ids=data["eval_category_ids"],
                     knn=knn, energy=energy, msp=msp, logits=logits.astype(np.float32))
            probe_rows.append({"rotation": rotation, "seed": seed, "probe_seed": 10000 + seed,
                               "initial_state_sha256": probe_payload["initial_state_sha256"],
                               "final_state_sha256": probe_payload["final_state_sha256"],
                               "downstream_id_eval_accuracy": probe_payload["downstream_id_eval_accuracy"],
                               "checkpoint": str(probe_path), "checkpoint_sha256": sha256_file(probe_path)})
            categories = data["eval_category_ids"]
            id_mask = np.isin(categories, list(d_ids))
            if int(id_mask.sum()) != 200: raise RuntimeError("Expected 200 downstream-ID evaluation images")
            for row in candidates:
                category = int(row["category_id"]); ood_mask = categories == category
                if int(ood_mask.sum()) != 10: raise RuntimeError(f"Expected 10 evaluation images for {category}")
                mask = id_mask | ood_mask; target = ood_mask[mask].astype(np.int64)
                state_rows.append({"genus_group_id": row["group_id"], "genus_name": row["genus"],
                    "official_category_id": category, "scientific_species_name": row["scientific_name"],
                    "common_name": row["common_name"], "role": row["role"], "absent_model": row["withheld_model"],
                    "seed": seed, "rotation": rotation,
                    "present_or_absent": "absent" if row["withheld_model"] == rotation else "present",
                    "auroc_knn": float(roc_auc_score(target, knn[mask])),
                    "auroc_energy": float(roc_auc_score(target, energy[mask])),
                    "auroc_msp": float(roc_auc_score(target, msp[mask])),
                    "id_eval_images": 200, "ood_eval_images": 10, "score_file": str(score_path),
                    "score_file_sha256": sha256_file(score_path)})
        if len(set(hashes)) != 1:
            raise RuntimeError(f"Probe initialization mismatch within upstream seed {seed}")
    if len(state_rows) != 640:
        raise RuntimeError(f"Expected 640 per-state AUROC rows, got {len(state_rows)}")
    write_csv(ROOT / "per_state_aurocs.csv", state_rows)
    write_csv(ROOT / "probe_runs.csv", probe_rows)
    return state_rows


if __name__ == "__main__":
    print(json.dumps({"rows": len(evaluate_all()), "status": "COMPLETE"}, indent=2))
