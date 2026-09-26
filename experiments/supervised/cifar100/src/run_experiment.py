"""Run the isolated eight-model all-subclass leave-one-out experiment."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .common import (
    CifarResNet18,
    cloned_state_dict,
    encoder_state_dict,
    environment_record,
    eval_subset,
    file_hash,
    pretrain_dataset,
    safe_load,
    seed_everything,
    seed_worker,
    state_hash,
    upsert_csv,
    write_json_immutable,
)
from .design import MODEL_WITHHELD, freeze_design


PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
CONFIG_PATH = PROJECT / "config.json"
CHECKPOINTS = PROJECT / "checkpoints"
FEATURES = PROJECT / "features"
LOGS = PROJECT / "logs"


def load_config() -> dict:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["project_dir"] = str(PROJECT)
    cfg["repo_root"] = str(REPO)
    cfg["data_dir"] = str(Path(os.environ.get("CIFAR100_ROOT", PROJECT / cfg["data_dir"])).resolve())
    cfg["source_semantic_manifest"] = str(
        (PROJECT / cfg["source_semantic_manifest"]).resolve()
    )
    return cfg


def run_name(model_id: str, seed: int) -> str:
    return f"{model_id.lower()}_seed{seed}"


def ensure_directories() -> None:
    for directory in (CHECKPOINTS, FEATURES, LOGS):
        directory.mkdir(parents=True, exist_ok=True)


def initial_state_audit(cfg: dict) -> dict:
    per_run = []
    per_seed = {}
    for seed in cfg["seeds"]:
        model_hashes = {}
        encoder_hashes = {}
        for model_id in cfg["models"]:
            seed_everything(seed, cfg["deterministic"])
            model = CifarResNet18(80)
            full_hash = state_hash(cloned_state_dict(model))
            encoder_hash = state_hash(encoder_state_dict(model))
            model_hashes[model_id] = full_hash
            encoder_hashes[model_id] = encoder_hash
            per_run.append(
                {
                    "seed": seed,
                    "model": model_id,
                    "initial_state_dict_sha256": full_hash,
                    "initial_encoder_sha256": encoder_hash,
                }
            )
            del model
        per_seed[str(seed)] = {
            "model_hashes": model_hashes,
            "encoder_hashes": encoder_hashes,
            "all_four_model_hashes_identical": len(set(model_hashes.values())) == 1,
            "all_four_encoder_hashes_identical": len(set(encoder_hashes.values())) == 1,
        }
    seed_representatives = [next(iter(per_seed[str(seed)]["model_hashes"].values())) for seed in cfg["seeds"]]
    audit = {
        "per_run": per_run,
        "per_seed": per_seed,
        "same_seed_models_identical": all(
            item["all_four_model_hashes_identical"] and item["all_four_encoder_hashes_identical"]
            for item in per_seed.values()
        ),
        "different_seeds_different": len(set(seed_representatives)) == len(seed_representatives),
    }
    audit["all_passed"] = audit["same_seed_models_identical"] and audit["different_seeds_different"]
    return audit


def source_hashes() -> dict:
    paths = [
        CONFIG_PATH,
        PROJECT / "manifest.json",
        PROJECT / "manifest.csv",
        Path(load_config()["source_semantic_manifest"]),
        Path(__file__),
        PROJECT / "src" / "common.py",
        PROJECT / "src" / "design.py",
        REPO / "confirmation2_interference" / "src" / "train_pretrain.py",
        REPO / "confirmation2_interference" / "src" / "data.py",
        REPO / "confirmation2_interference" / "src" / "models.py",
        REPO / "confirmation2_interference" / "src" / "extract_features.py",
        REPO / "confirmation2_interference" / "src" / "evaluate_ood.py",
    ]
    return {str(path.resolve()): file_hash(path) for path in paths}


def preflight(cfg: dict) -> tuple[dict, dict]:
    ensure_directories()
    manifest, design_invariants = freeze_design(
        PROJECT,
        Path(cfg["source_semantic_manifest"]),
        Path(cfg["data_dir"]),
    )
    print(pd.read_csv(PROJECT / "manifest.csv").to_string(index=False), flush=True)
    init = initial_state_audit(cfg)
    invariants = {
        **design_invariants,
        "initialization_audit": init,
        "all_passed": bool(design_invariants["all_passed"] and init["all_passed"]),
    }
    invariants["status"] = "PASS" if invariants["all_passed"] else "FAIL"
    write_json_immutable(PROJECT / "invariants.json", invariants)
    if not invariants["all_passed"]:
        raise RuntimeError(f"STOP BEFORE TRAINING: {invariants}")
    preregistration = {
        "frozen_before_training": True,
        "config_sha256": file_hash(CONFIG_PATH),
        "manifest_json_sha256": file_hash(PROJECT / "manifest.json"),
        "manifest_csv_sha256": file_hash(PROJECT / "manifest.csv"),
        "invariants_sha256": file_hash(PROJECT / "invariants.json"),
        "source_hashes": source_hashes(),
        "environment": environment_record(REPO),
        "seeds": cfg["seeds"],
        "bootstrap_seed": cfg["bootstrap_seed"],
        "decision_rule": cfg["decision_rule"],
        "exact_command": cfg["exact_command"],
    }
    write_json_immutable(PROJECT / "preregistration_snapshot.json", preregistration)
    print("PRETRAINING INVARIANTS: PASS", flush=True)
    for seed, record in init["per_seed"].items():
        digest = next(iter(record["model_hashes"].values()))
        print(f"seed={seed} matched_initial_state_dict_sha256={digest}", flush=True)
    return manifest, invariants


def supervised_accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for images, targets in loader:
            targets = targets.to(device)
            correct += (model(images.to(device)).argmax(1) == targets).sum().item()
            total += targets.numel()
    return correct / total


def expected_init_hash(invariants: dict, seed: int, model_id: str, encoder: bool = False) -> str:
    key = "encoder_hashes" if encoder else "model_hashes"
    return invariants["initialization_audit"]["per_seed"][str(seed)][key][model_id]


def train_one(cfg: dict, manifest: dict, invariants: dict, model_id: str, seed: int) -> dict:
    selected = sorted(manifest["model_pretraining_classes"][model_id])
    if len(selected) != 80:
        raise AssertionError("Every model must contain exactly 80 fine classes")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = seed_everything(seed, cfg["deterministic"])
    model = CifarResNet18(80).to(device)
    initial_model_hash = state_hash(cloned_state_dict(model))
    initial_encoder_hash = state_hash(encoder_state_dict(model))
    if initial_model_hash != expected_init_hash(invariants, seed, model_id):
        raise RuntimeError(f"Initial full-model hash mismatch for {model_id}, seed {seed}")
    if initial_encoder_hash != expected_init_hash(invariants, seed, model_id, encoder=True):
        raise RuntimeError(f"Initial encoder hash mismatch for {model_id}, seed {seed}")

    optimizer = SGD(
        model.parameters(),
        lr=cfg["learning_rate"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    amp_enabled = bool(cfg["amp_on_cuda"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    run = run_name(model_id, seed)
    full_path = CHECKPOINTS / f"pretrain_{run}.pt"
    encoder_path = CHECKPOINTS / f"encoder_{run}.pt"
    start_epoch = 0
    history = []
    if full_path.exists() and cfg["resume"]:
        saved = safe_load(full_path)
        if saved.get("identity") != [model_id, seed]:
            raise RuntimeError(f"Checkpoint identity mismatch: {full_path}")
        if saved.get("classes") != selected or saved.get("target_epochs") != cfg["epochs"]:
            raise RuntimeError(f"Checkpoint recipe/class mismatch: {full_path}")
        if saved.get("initial_state_dict_sha256") != initial_model_hash:
            raise RuntimeError(f"Checkpoint initialization mismatch: {full_path}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        generator.set_state(saved["data_generator_state"])
        start_epoch = int(saved["epoch"])
        history = list(saved["history"])
        print(f"{run} resuming at epoch {start_epoch}/{cfg['epochs']}", flush=True)

    dataset = pretrain_dataset(cfg["data_dir"], selected)
    if len(dataset) != 40_000:
        raise AssertionError(f"Expected 40,000 pretraining samples, got {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=cfg["num_workers"],
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
    )
    criterion = nn.CrossEntropyLoss()
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        correct = total = 0
        loss_sum = 0.0
        began = time.time()
        for images, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            images, targets = images.to(device), targets.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            correct += (logits.argmax(1) == targets).sum().item()
            total += targets.numel()
            loss_sum += loss.item() * targets.numel()
        scheduler.step()
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": loss_sum / total,
                "train_acc": correct / total,
                "seconds": time.time() - began,
            }
        )
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "data_generator_state": generator.get_state(),
                "epoch": epoch + 1,
                "target_epochs": cfg["epochs"],
                "history": history,
                "identity": [model_id, seed],
                "classes": selected,
                "initial_state_dict_sha256": initial_model_hash,
                "initial_encoder_sha256": initial_encoder_hash,
                "recipe_config_sha256": file_hash(CONFIG_PATH),
            },
            full_path,
        )
        (LOGS / f"{run}.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        latest = history[-1]
        print(
            f"{run} epoch={epoch + 1}/{cfg['epochs']} "
            f"loss={latest['train_loss']:.4f} acc={latest['train_acc']:.4f} "
            f"seconds={latest['seconds']:.1f}",
            flush=True,
        )

    restricted_test = eval_subset(cfg["data_dir"], False, selected, remap=True)
    restricted_test_acc = supervised_accuracy(
        model,
        DataLoader(
            restricted_test,
            batch_size=cfg["feature_batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"],
            pin_memory=device.type == "cuda",
        ),
        device,
    )
    torch.save(
        {
            "encoder": encoder_state_dict(model),
            "feature_dim": 512,
            "identity": [model_id, seed],
            "classes": selected,
            "initial_state_dict_sha256": initial_model_hash,
            "initial_encoder_sha256": initial_encoder_hash,
        },
        encoder_path,
    )
    final = history[-1]
    row = {
        "model": model_id,
        "seed": seed,
        "completed_epochs": len(history),
        "target_epochs": cfg["epochs"],
        "train_loss": final["train_loss"],
        "train_acc": final["train_acc"],
        "restricted_test_acc": restricted_test_acc,
        "n_classes": len(selected),
        "n_examples": len(dataset),
        "initial_state_dict_sha256": initial_model_hash,
        "initial_encoder_sha256": initial_encoder_hash,
        "checkpoint_sha256": file_hash(full_path),
        "encoder_sha256": file_hash(encoder_path),
        "checkpoint": str(full_path.resolve()),
        "encoder": str(encoder_path.resolve()),
        "status": "COMPLETE",
    }
    upsert_csv(PROJECT / "training_runs.csv", [row], ("model", "seed"))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def collect_features(encoder, loader, device):
    encoder.eval()
    encoder.requires_grad_(False)
    features, labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            features.append(torch.flatten(encoder(images.to(device)), 1).cpu())
            labels.append(targets.cpu())
    return torch.cat(features), torch.cat(labels)


def extract_one(cfg: dict, manifest: dict, model_id: str, seed: int) -> Path:
    run = run_name(model_id, seed)
    output = FEATURES / f"features_{run}.pt"
    if output.exists() and cfg["resume"]:
        saved = safe_load(output)
        if saved.get("identity") != [model_id, seed]:
            raise RuntimeError(f"Feature-cache identity mismatch: {output}")
        return output
    selected = sorted(manifest["model_pretraining_classes"][model_id])
    encoder_file = CHECKPOINTS / f"encoder_{run}.pt"
    saved_encoder = safe_load(encoder_file)
    if saved_encoder.get("classes") != selected or saved_encoder.get("identity") != [model_id, seed]:
        raise RuntimeError(f"Encoder checkpoint mismatch: {encoder_file}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed, cfg["deterministic"])
    model = CifarResNet18(80).to(device)
    model.encoder.load_state_dict(saved_encoder["encoder"])
    model.encoder.eval()
    model.encoder.requires_grad_(False)
    before = state_hash(encoder_state_dict(model))
    d_classes = set(manifest["downstream_id_classes"])
    all_classes = set(range(100))
    train = eval_subset(cfg["data_dir"], True, d_classes, remap=False)
    test = eval_subset(cfg["data_dir"], False, all_classes, remap=False)
    args = {
        "batch_size": cfg["feature_batch_size"],
        "shuffle": False,
        "num_workers": cfg["num_workers"],
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    train_x, train_y = collect_features(model.encoder, DataLoader(train, **args), device)
    test_x, test_y = collect_features(model.encoder, DataLoader(test, **args), device)
    after = state_hash(encoder_state_dict(model))
    if before != after:
        raise RuntimeError(f"Encoder/BatchNorm state mutated during feature extraction: {run}")
    if train_x.shape != (10_000, 512) or test_x.shape != (10_000, 512):
        raise AssertionError(f"Unexpected feature shapes for {run}")
    torch.save(
        {
            "train_id_features": train_x,
            "train_id_labels": train_y,
            "train_id_indices": torch.tensor(train.indices),
            "test_features": test_x,
            "test_labels": test_y,
            "test_indices": torch.tensor(test.indices),
            "feature_interface": "pooled_encoder_penultimate",
            "feature_dim": 512,
            "identity": [model_id, seed],
            "encoder_state_sha256_before": before,
            "encoder_state_sha256_after": after,
            "encoder_eval": True,
            "parameters_frozen": True,
        },
        output,
    )
    print(f"{run} frozen features saved: {output}", flush=True)
    del model, train_x, test_x
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def knn_scores(query: torch.Tensor, reference: torch.Tensor, device, k: int = 50, chunk: int = 256):
    reference = F.normalize(reference, dim=1).to(device)
    values = []
    with torch.no_grad():
        for start in range(0, len(query), chunk):
            normalized = F.normalize(query[start : start + chunk], dim=1).to(device)
            distances = 1 - normalized @ reference.T
            values.append(distances.topk(k, largest=False, dim=1).values.mean(1).cpu())
    return torch.cat(values).numpy()


def ood_auroc(id_scores, ood_scores) -> float:
    labels = np.r_[np.zeros(len(id_scores)), np.ones(len(ood_scores))]
    return float(roc_auc_score(labels, np.r_[id_scores, ood_scores]))


def candidate_lookup(manifest: dict) -> dict:
    result = {}
    for group in manifest["groups"]:
        for role in ("c1", "c2", "c3", "c4"):
            item = group[role]
            result[item["fine_id"]] = {
                "coarse_id": group["coarse_id"],
                "coarse_name": group["coarse_name"],
                "fine_id": item["fine_id"],
                "fine_name": item["fine_name"],
                "new_role": role,
                "original_role": item["original_role"],
                "withheld_model": item["withheld_model"],
            }
    return result


def evaluate_one(cfg: dict, manifest: dict, model_id: str, seed: int) -> list[dict]:
    run = run_name(model_id, seed)
    data = safe_load(FEATURES / f"features_{run}.pt")
    if data.get("identity") != [model_id, seed] or not data.get("encoder_eval") or not data.get("parameters_frozen"):
        raise RuntimeError(f"Invalid frozen feature cache: {run}")
    if data["encoder_state_sha256_before"] != data["encoder_state_sha256_after"]:
        raise RuntimeError(f"Encoder state mutation recorded: {run}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores = knn_scores(
        data["test_features"],
        data["train_id_features"],
        device,
        k=cfg["knn_k"],
        chunk=256,
    )
    test_y = data["test_labels"].numpy()
    d_classes = set(manifest["downstream_id_classes"])
    id_mask = np.isin(test_y, list(d_classes))
    if id_mask.sum() != 2_000:
        raise AssertionError("Expected 2,000 downstream-ID test examples")
    lookup = candidate_lookup(manifest)
    selected = set(manifest["model_pretraining_classes"][model_id])
    rows = []
    for fine_id in sorted(lookup):
        item = lookup[fine_id]
        mask = test_y == fine_id
        if mask.sum() != 100:
            raise AssertionError(f"Expected 100 OOD test samples for class {fine_id}")
        rows.append(
            {
                **item,
                "model": model_id,
                "seed": seed,
                "supervised_upstream": fine_id in selected,
                "auroc_knn": ood_auroc(scores[id_mask], scores[mask]),
                "mean_knn_score": float(scores[mask].mean()),
                "k": cfg["knn_k"],
                "distance": cfg["knn_distance"],
            }
        )
    upsert_csv(PROJECT / "class_level_metrics.csv", rows, ("model", "seed", "fine_id"))
    print(f"{run} kNN evaluation complete", flush=True)
    return rows


def class_deltas(raw: pd.DataFrame, manifest: dict, cfg: dict) -> pd.DataFrame:
    lookup = candidate_lookup(manifest)
    rows = []
    for seed in cfg["seeds"]:
        seed_rows = raw[raw.seed == seed]
        for fine_id in sorted(lookup):
            item = lookup[fine_id]
            values = seed_rows[seed_rows.fine_id == fine_id].set_index("model").auroc_knn.to_dict()
            if set(values) != set(cfg["models"]):
                raise RuntimeError(f"Missing raw AUROC for class={fine_id}, seed={seed}")
            withheld_model = item["withheld_model"]
            supervised_models = [model for model in cfg["models"] if model != withheld_model]
            supervised_values = [values[model] for model in supervised_models]
            if len(supervised_values) != 3:
                raise AssertionError("Every class must have exactly three supervised values")
            row = {
                **item,
                "seed": seed,
                "auroc_withheld": values[withheld_model],
                "auroc_supervised_mean": float(np.mean(supervised_values)),
            }
            for model in cfg["models"]:
                row[f"raw_auroc_{model.lower()}"] = values[model]
                row[f"auroc_supervised_{model.lower()}"] = (
                    np.nan if model == withheld_model else values[model]
                )
            row["delta_knn"] = row["auroc_withheld"] - row["auroc_supervised_mean"]
            rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != 160 or result.fine_id.nunique() != 80:
        raise AssertionError("Expected 160 class-by-seed delta records over 80 classes")
    return result


def summarize_and_bootstrap(delta: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, np.ndarray]:
    identifiers = [
        "coarse_id",
        "coarse_name",
        "fine_id",
        "fine_name",
        "new_role",
        "original_role",
        "withheld_model",
    ]
    per_class = delta.groupby(identifiers, as_index=False).agg(
        mean_delta_knn=("delta_knn", "mean"),
        mean_auroc_withheld=("auroc_withheld", "mean"),
        mean_auroc_supervised=("auroc_supervised_mean", "mean"),
        seed_sd_delta_knn=("delta_knn", "std"),
        n_seeds=("seed", "nunique"),
    )
    if len(per_class) != 80 or set(per_class.n_seeds) != {2}:
        raise AssertionError("Primary aggregation must have 80 class effects averaged over two seeds")
    role_summary = per_class.groupby("original_role", as_index=False).agg(
        n_classes=("mean_delta_knn", "size"),
        mean_delta_knn=("mean_delta_knn", "mean"),
        median_delta_knn=("mean_delta_knn", "median"),
        std_delta_knn=("mean_delta_knn", "std"),
        min_delta_knn=("mean_delta_knn", "min"),
        max_delta_knn=("mean_delta_knn", "max"),
        negative_classes=("mean_delta_knn", lambda values: int((values < 0).sum())),
        positive_classes=("mean_delta_knn", lambda values: int((values > 0).sum())),
        zero_classes=("mean_delta_knn", lambda values: int((values == 0).sum())),
    )
    group_summary = per_class.groupby(["coarse_id", "coarse_name"], as_index=False).agg(
        n_classes=("mean_delta_knn", "size"),
        mean_delta_knn=("mean_delta_knn", "mean"),
        median_delta_knn=("mean_delta_knn", "median"),
        min_delta_knn=("mean_delta_knn", "min"),
        max_delta_knn=("mean_delta_knn", "max"),
        negative_classes=("mean_delta_knn", lambda values: int((values < 0).sum())),
    )
    if set(group_summary.n_classes) != {4}:
        raise AssertionError("Each bootstrap cluster must retain four candidate classes")
    group_values = group_summary.sort_values("coarse_id").mean_delta_knn.to_numpy()
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    sampled = rng.integers(0, 20, size=(cfg["bootstrap_draws"], 20))
    bootstrap = group_values[sampled].mean(axis=1)
    effects = per_class.mean_delta_knn.to_numpy()
    summary = {
        "n_classes": 80,
        "n_coarse_groups": 20,
        "n_seeds": 2,
        "mean_delta_knn": float(effects.mean()),
        "median_delta_knn": float(np.median(effects)),
        "std_delta_knn": float(effects.std(ddof=1)),
        "min_delta_knn": float(effects.min()),
        "max_delta_knn": float(effects.max()),
        "negative_classes": int((effects < 0).sum()),
        "positive_classes": int((effects > 0).sum()),
        "zero_classes": int((effects == 0).sum()),
        "cluster_bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)),
        "cluster_bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975)),
        "bootstrap_draws": cfg["bootstrap_draws"],
        "bootstrap_seed": cfg["bootstrap_seed"],
        "bootstrap_unit": cfg["bootstrap_unit"],
    }
    return per_class, role_summary, group_summary, summary, bootstrap


def compare_old(per_class: pd.DataFrame, cfg: dict) -> dict:
    old_path = REPO / "confirmation2_interference" / "artifacts" / "full" / "class_level_metrics.csv"
    old = pd.read_csv(old_path)
    keys = ["coarse_class", "coarse_class_index", "intervention_role", "fine_class", "fine_class_index"]
    old_classes = old.groupby(keys, as_index=False).delta_knn.mean()
    old_roles = old_classes.groupby("intervention_role").agg(
        n_classes=("delta_knn", "size"),
        mean_delta_knn=("delta_knn", "mean"),
        std_delta_knn=("delta_knn", "std"),
        negative_classes=("delta_knn", lambda values: int((values < 0).sum())),
    )
    return {
        "old_ab_source": str(old_path.resolve()),
        "old_ab_source_sha256": file_hash(old_path),
        "old_ab_mean_delta_knn": float(old_classes.delta_knn.mean()),
        "old_ab_std_delta_knn": float(old_classes.delta_knn.std(ddof=1)),
        "old_ab_negative_classes": int((old_classes.delta_knn < 0).sum()),
        "old_ab_n_classes": len(old_classes),
        "old_ab_role_specific": {
            role: {key: (int(value) if key in ("n_classes", "negative_classes") else float(value)) for key, value in row.items()}
            for role, row in old_roles.to_dict(orient="index").items()
        },
        "old_ab_r1_r2_note": "r1/r2 were common pretraining context, not intervention OOD classes, so old A/B treatment effects do not exist for them.",
        "new_minus_old_mean_delta": float(per_class.mean_delta_knn.mean() - old_classes.delta_knn.mean()),
    }


def verdict(summary: dict, cfg: dict) -> tuple[str, dict]:
    rule = cfg["decision_rule"]
    criteria = {
        "mean_delta_at_most_minus_0_03": summary["mean_delta_knn"] <= rule["mean_delta_max"],
        "cluster_ci_upper_below_zero": summary["cluster_bootstrap_ci95_high"] < rule["cluster_ci_upper_strictly_below"],
        "at_least_60_of_80_negative": summary["negative_classes"] >= rule["minimum_negative_classes"],
    }
    return (rule["replicated_label"] if all(criteria.values()) else rule["otherwise_label"]), criteria


def design_assessment(final_verdict: str, role_summary: pd.DataFrame) -> tuple[str, str]:
    roles_all_negative = bool((role_summary.mean_delta_knn < 0).all())
    roles_most_negative = bool((role_summary.negative_classes >= 15).all())
    if final_verdict == "LOO_EFFECT_REPLICATED" and roles_all_negative and roles_most_negative:
        return (
            "STRONGLY_CONTRADICTED",
            "The effect meets the preregistered 80-class LOO rule and is consistently negative across all four former role strata, including the previous r1/r2 context classes.",
        )
    if final_verdict == "LOO_EFFECT_REPLICATED":
        return (
            "WEAKENED",
            "The 80-class LOO effect replicates, but former-role heterogeneity prevents the strongest rejection of pair-selection sensitivity.",
        )
    return (
        "SUPPORTED",
        "The all-subclass LOO experiment does not meet its preregistered replication rule.",
    )


def plots(per_class: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    ordered = per_class.sort_values(["coarse_id", "new_role"]).reset_index(drop=True)
    colors = plt.cm.tab20(ordered.coarse_id.to_numpy() % 20)
    fig, ax = plt.subplots(figsize=(22, 8))
    ax.scatter(np.arange(80), ordered.mean_delta_knn, c=colors, s=38)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(np.arange(80))
    ax.set_xticklabels(ordered.fine_name, rotation=90, fontsize=7)
    ax.set_ylabel("Delta kNN AUROC (withheld − supervised mean)")
    ax.set_title("All 80 candidate downstream-OOD classes")
    fig.tight_layout()
    fig.savefig(PROJECT / "delta_knn_all_80.png", bbox_inches="tight")
    plt.close(fig)

    roles = ["a", "b", "r1", "r2"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    values = [per_class.loc[per_class.original_role == role, "mean_delta_knn"].to_numpy() for role in roles]
    ax.boxplot(values, labels=roles, showfliers=True)
    jitter_rng = np.random.default_rng(20260908)
    for index, role_values in enumerate(values, start=1):
        ax.scatter(index + jitter_rng.uniform(-0.10, 0.10, len(role_values)), role_values, s=25, alpha=0.75)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Original confirmation role")
    ax.set_ylabel("Delta kNN AUROC")
    ax.set_title("Treatment effects by original role")
    fig.tight_layout()
    fig.savefig(PROJECT / "delta_knn_by_original_role.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 6.5))
    role_colors = {"a": "#4C78A8", "b": "#F58518", "r1": "#54A24B", "r2": "#E45756"}
    offsets = {"a": -0.24, "b": -0.08, "r1": 0.08, "r2": 0.24}
    groups = ordered[["coarse_id", "coarse_name"]].drop_duplicates().sort_values("coarse_id")
    for role in roles:
        subset = ordered[ordered.original_role == role].sort_values("coarse_id")
        ax.scatter(subset.coarse_id + offsets[role], subset.mean_delta_knn, label=role, color=role_colors[role], s=36)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(groups.coarse_id)
    ax.set_xticklabels(groups.coarse_name, rotation=55, ha="right")
    ax.set_ylabel("Delta kNN AUROC")
    ax.set_title("Four candidate classes within each semantic superclass")
    ax.legend(title="Original role", frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(PROJECT / "delta_knn_by_superclass.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7))
    for role in roles:
        subset = per_class[per_class.original_role == role]
        ax.scatter(
            subset.mean_auroc_supervised,
            subset.mean_auroc_withheld,
            label=role,
            s=38,
            alpha=0.8,
            color=role_colors[role],
        )
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Supervised mean kNN AUROC")
    ax.set_ylabel("Withheld kNN AUROC")
    ax.set_title("Withheld versus supervised OOD detectability")
    ax.legend(title="Original role", frameon=False)
    fig.tight_layout()
    fig.savefig(PROJECT / "withheld_vs_supervised_auroc.png", bbox_inches="tight")
    plt.close(fig)


def finalize(cfg: dict, manifest: dict, invariants: dict) -> dict:
    training = pd.read_csv(PROJECT / "training_runs.csv")
    expected_pairs = {(model, seed) for model in cfg["models"] for seed in cfg["seeds"]}
    actual_pairs = set(map(tuple, training[["model", "seed"]].to_numpy()))
    if actual_pairs != expected_pairs or not (training.status == "COMPLETE").all() or not (training.completed_epochs == 100).all():
        raise RuntimeError("8/8 upstream training runs are not complete")
    for seed in cfg["seeds"]:
        rows = training[training.seed == seed]
        if rows.initial_state_dict_sha256.nunique() != 1 or rows.initial_encoder_sha256.nunique() != 1:
            raise RuntimeError(f"Matched initialization failed in completed runs for seed={seed}")
    if training.groupby("seed").initial_state_dict_sha256.first().nunique() != 2:
        raise RuntimeError("Different seeds unexpectedly share initialization")

    raw = pd.read_csv(PROJECT / "class_level_metrics.csv")
    if len(raw) != 640:
        raise RuntimeError("Expected 640 raw model × seed × class AUROCs")
    delta = class_deltas(raw, manifest, cfg)
    delta.to_csv(PROJECT / "class_level_delta.csv", index=False)
    per_class, roles, groups, primary, bootstrap = summarize_and_bootstrap(delta, cfg)
    roles.to_csv(PROJECT / "role_summary.csv", index=False)
    groups.to_csv(PROJECT / "group_summary.csv", index=False)
    pd.DataFrame({"draw": np.arange(cfg["bootstrap_draws"]), "mean_delta_knn": bootstrap}).to_csv(
        PROJECT / "bootstrap_samples.csv", index=False
    )
    bootstrap_summary = {
        "unit": cfg["bootstrap_unit"],
        "draws": cfg["bootstrap_draws"],
        "seed": cfg["bootstrap_seed"],
        "observed_mean": primary["mean_delta_knn"],
        "ci95": [primary["cluster_bootstrap_ci95_low"], primary["cluster_bootstrap_ci95_high"]],
        "bootstrap_mean": float(bootstrap.mean()),
        "bootstrap_median": float(np.median(bootstrap)),
        "bootstrap_min": float(bootstrap.min()),
        "bootstrap_max": float(bootstrap.max()),
    }
    (PROJECT / "bootstrap_summary.json").write_text(json.dumps(bootstrap_summary, indent=2) + "\n", encoding="utf-8")
    final_verdict, criteria = verdict(primary, cfg)
    design_label, design_reason = design_assessment(final_verdict, roles)
    comparison = compare_old(per_class, cfg)
    checkpoint_hashes = {
        str(path.resolve()): file_hash(path) for path in sorted(CHECKPOINTS.glob("*.pt"))
    }
    feature_hashes = {
        str(path.resolve()): file_hash(path) for path in sorted(FEATURES.glob("*.pt"))
    }
    snapshot = json.loads((PROJECT / "preregistration_snapshot.json").read_text(encoding="utf-8"))
    current_source_hashes = source_hashes()
    source_unchanged = snapshot["source_hashes"] == current_source_hashes
    if not source_unchanged:
        raise RuntimeError("Frozen source/config/manifest hash changed during experiment")
    summary = {
        "training_completion": "8/8",
        "invariant_status": invariants["status"],
        "primary": primary,
        "role_summary": roles.to_dict(orient="records"),
        "group_summary": groups.to_dict(orient="records"),
        "comparison_to_existing_ab": comparison,
        "decision_criteria": criteria,
        "verdict": final_verdict,
        "design_objection_assessment": design_label,
        "design_objection_reason": design_reason,
        "reproducibility": {
            "environment": environment_record(REPO),
            "seeds": cfg["seeds"],
            "bootstrap_seed": cfg["bootstrap_seed"],
            "exact_command": cfg["exact_command"],
            "source_hashes": current_source_hashes,
            "source_hashes_unchanged": source_unchanged,
            "checkpoint_hashes": checkpoint_hashes,
            "feature_hashes": feature_hashes,
        },
    }
    (PROJECT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (PROJECT / "verdict.txt").write_text(
        "\n".join(
            [
                final_verdict,
                f"mean_delta_knn={primary['mean_delta_knn']:.9f}",
                f"semantic_group_bootstrap_95CI=[{primary['cluster_bootstrap_ci95_low']:.9f}, {primary['cluster_bootstrap_ci95_high']:.9f}]",
                f"negative_classes={primary['negative_classes']}/80",
                f"design_objection={design_label}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    plots(per_class)
    write_report(summary, roles)
    return summary


def write_report(summary: dict, roles: pd.DataFrame) -> None:
    primary = summary["primary"]
    comparison = summary["comparison_to_existing_ab"]
    lines = [
        "# All-subclass leave-one-out CIFAR-100 experiment",
        "",
        "## Scientific question",
        "",
        "This isolated experiment tests whether the same-class upstream-supervision effect persists when all four non-downstream-ID fine classes in every CIFAR-100 superclass receive leave-one-out treatment, eliminating the original arbitrary A/B candidate-pair restriction.",
        "",
        "## Frozen design and execution",
        "",
        "The downstream-ID class `d` is unchanged from confirmation1/confirmation2. Original roles `a,b,r1,r2` map deterministically to `c1,c2,c3,c4`. M1–M4 each withhold one corresponding candidate per superclass, contain exactly 80 classes/40,000 training images, and use the unchanged supervised ResNet-18 recipe. Seeds are 0 and 1; within each seed all four models have identical initial full-network and encoder hashes. Eight upstream models were trained; no downstream linear probes were required for the frozen-feature kNN endpoint.",
        "",
        "All encoders were frozen, set to eval mode, and evaluated under no-grad. Encoder state hashes before and after extraction were identical, including BatchNorm buffers. kNN uses the unchanged k=50 mean cosine distance to the 10,000-image downstream-ID training reference bank; OOD is positive in AUROC.",
        "",
        "## Primary result",
        "",
        "For each class and seed, the withheld-model AUROC was compared with the mean of its three supervised-model AUROCs. Seeds were then averaged within each of 80 candidate classes. The primary CI resamples 20 semantic superclasses and retains all four candidate classes in each sampled cluster.",
        "",
        "| Mean | Median | SD | Min | Max | Negative | Positive | Zero | 95% cluster CI |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        f"| {primary['mean_delta_knn']:.6f} | {primary['median_delta_knn']:.6f} | {primary['std_delta_knn']:.6f} | {primary['min_delta_knn']:.6f} | {primary['max_delta_knn']:.6f} | {primary['negative_classes']}/80 | {primary['positive_classes']}/80 | {primary['zero_classes']}/80 | [{primary['cluster_bootstrap_ci95_low']:.6f}, {primary['cluster_bootstrap_ci95_high']:.6f}] |",
        "",
        "## Original-role strata",
        "",
        "| Original role | n | Mean Delta | Median | SD | Min | Max | Negative |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in roles.sort_values("original_role").itertuples(index=False):
        lines.append(
            f"| {row.original_role} | {row.n_classes} | {row.mean_delta_knn:.6f} | {row.median_delta_knn:.6f} | {row.std_delta_knn:.6f} | {row.min_delta_knn:.6f} | {row.max_delta_knn:.6f} | {row.negative_classes}/20 |"
        )
    lines += [
        "",
        "## Comparison with the frozen A/B result",
        "",
        f"The existing confirmation2 A/B mean was {comparison['old_ab_mean_delta_knn']:.6f} across 40 a/b intervention classes (SD {comparison['old_ab_std_delta_knn']:.6f}; {comparison['old_ab_negative_classes']}/40 negative). The new all-subclass LOO mean is {primary['mean_delta_knn']:.6f} across 80 classes (SD {primary['std_delta_knn']:.6f}; {primary['negative_classes']}/80 negative), a new-minus-old mean difference of {comparison['new_minus_old_mean_delta']:.6f}. Old A/B treatment effects do not exist for r1/r2 because they were fixed common-context classes; their new LOO results are therefore the direct test of whether the earlier role selection mattered.",
        "",
        "## Preregistered decision",
        "",
        f"Criteria: mean ≤ -0.03 = {summary['decision_criteria']['mean_delta_at_most_minus_0_03']}; cluster-CI upper bound < 0 = {summary['decision_criteria']['cluster_ci_upper_below_zero']}; at least 60/80 negative = {summary['decision_criteria']['at_least_60_of_80_negative']}.",
        "",
        f"**{summary['verdict']}**",
        "",
        "## Design objection",
        "",
        f"**{summary['design_objection_assessment']}** — {summary['design_objection_reason']}",
        "",
        "This conclusion is limited to the controlled supervised CIFAR-100 intervention implemented here. It does not claim universality beyond this architecture, recipe, dataset, or treatment definition.",
        "",
        "## Reproducibility",
        "",
        "`summary.json` records package/CUDA/GPU versions, seeds, exact command, source and manifest hashes, and all checkpoint/feature hashes. The frozen pretraining manifest, configuration, decision rule, and source hashes were unchanged from preflight through analysis.",
    ]
    (PROJECT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(summary: dict) -> None:
    primary = summary["primary"]
    roles = {row["original_role"]: row["mean_delta_knn"] for row in summary["role_summary"]}
    comparison = summary["comparison_to_existing_ab"]
    print("", flush=True)
    print("CIFAR-100 design all-subclass LOO", flush=True)
    print("Training completion: 8/8", flush=True)
    print(f"Invariant status: {summary['invariant_status']}", flush=True)
    print(f"Mean Delta_kNN: {primary['mean_delta_knn']:.6f}", flush=True)
    print(
        f"95% semantic-group bootstrap CI: [{primary['cluster_bootstrap_ci95_low']:.6f}, "
        f"{primary['cluster_bootstrap_ci95_high']:.6f}]",
        flush=True,
    )
    print(f"Negative classes: {primary['negative_classes']} / 80", flush=True)
    print(
        "Role means: " + ", ".join(f"{role}={roles[role]:.6f}" for role in ("a", "b", "r1", "r2")),
        flush=True,
    )
    print(
        f"Old A/B mean: {comparison['old_ab_mean_delta_knn']:.6f}; "
        f"new-minus-old: {comparison['new_minus_old_mean_delta']:.6f}",
        flush=True,
    )
    print(f"Verdict: {summary['verdict']}", flush=True)
    print(f"Design A/B-pair-selection objection: {summary['design_objection_assessment']}", flush=True)


def run(preflight_only: bool = False) -> None:
    cfg = load_config()
    manifest, invariants = preflight(cfg)
    if preflight_only:
        print("Preflight-only requested; no training started.", flush=True)
        return
    for seed in cfg["seeds"]:
        for model_id in cfg["models"]:
            train_one(cfg, manifest, invariants, model_id, seed)
            extract_one(cfg, manifest, model_id, seed)
            evaluate_one(cfg, manifest, model_id, seed)
    summary = finalize(cfg, manifest, invariants)
    print_summary(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    run(preflight_only=args.preflight_only)


if __name__ == "__main__":
    main()
