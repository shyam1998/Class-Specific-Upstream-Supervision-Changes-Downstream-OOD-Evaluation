from __future__ import annotations

import copy
import json
import os
import random
import time
from pathlib import Path

from .common import (MODELS, ROOT, ROTATION_TOTALS, SEEDS, assert_protected_unchanged,
                     atomic_torch_save, full_model_sha256, now, read_json, sha256_file,
                     state_dict_sha256, torch_load, write_csv, write_json)
from .data import downstream_eval_dataset, downstream_reference_dataset, upstream_dataset, upstream_rows
from .model import make_backbone, make_model, nt_xent


def _identity():
    return {"config_sha256": sha256_file(ROOT / "config.json"),
            "manifest_sha256": sha256_file(ROOT / "manifest.csv"),
            "preregistration_sha256": sha256_file(ROOT / "preregistration_snapshot.json")}


def _validate_initial(seed: int):
    import torch
    path = ROOT / "checkpoints" / f"initial_seed{seed}.pt"
    if not path.is_file():
        from .common import set_seed
        from .model import make_model
        set_seed(seed)
        model = make_model(False)
        payload = {"metadata": {"encoder_sha256": state_dict_sha256(model.encoder.state_dict()),
                                 "projector_sha256": state_dict_sha256(model.projector.state_dict()),
                                 "full_model_sha256": full_model_sha256(model.encoder.state_dict(), model.projector.state_dict())},
                   "encoder_state": model.encoder.state_dict(), "projector_state": model.projector.state_dict()}
        atomic_torch_save(payload, path)
    payload = torch_load(path)
    metadata = payload["metadata"]
    if state_dict_sha256(payload["encoder_state"]) != metadata["encoder_sha256"]:
        raise RuntimeError("Frozen encoder initialization hash mismatch")
    if state_dict_sha256(payload["projector_state"]) != metadata["projector_sha256"]:
        raise RuntimeError("Frozen projector initialization hash mismatch")
    if full_model_sha256(payload["encoder_state"], payload["projector_state"]) != metadata["full_model_sha256"]:
        raise RuntimeError("Frozen full-model initialization hash mismatch")
    return payload


def _loader(rotation: str, seed: int):
    import torch
    from torch.utils.data import DataLoader
    dataset = upstream_dataset(rotation)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=8,
                        pin_memory=True, persistent_workers=True, drop_last=True,
                        generator=generator)
    return dataset, loader, generator


def _checkpoint_path(rotation: str, seed: int, epoch: int):
    return ROOT / "checkpoints" / f"{rotation}_seed{seed}" / f"epoch_{epoch:03d}.pt"


def _checkpoint_valid(path: Path, rotation: str, seed: int, required_epoch: int | None = None):
    payload = torch_load(path)
    metadata = payload.get("metadata", {})
    expected = {"kind": "simclr_checkpoint", "rotation": rotation, "seed": seed, **_identity()}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"Checkpoint metadata mismatch for {key}: {path}")
    if required_epoch is not None and int(metadata.get("epoch", -1)) != required_epoch:
        raise RuntimeError(f"Checkpoint epoch mismatch: {path}")
    if state_dict_sha256(payload["encoder_state"]) != metadata["encoder_sha256"]:
        raise RuntimeError(f"Checkpoint encoder hash mismatch: {path}")
    if state_dict_sha256(payload["projector_state"]) != metadata["projector_sha256"]:
        raise RuntimeError(f"Checkpoint projector hash mismatch: {path}")
    return payload


def _save_checkpoint(path, rotation, seed, epoch, model, optimizer, scheduler, scaler,
                     generator, global_step, runtime_seconds, mean_loss, resume_history):
    encoder_hash = state_dict_sha256(model.encoder.state_dict())
    projector_hash = state_dict_sha256(model.projector.state_dict())
    metadata = {"kind": "simclr_checkpoint", "rotation": rotation, "seed": seed,
                "epoch": epoch, **_identity(), "encoder_sha256": encoder_hash,
                "projector_sha256": projector_hash,
                "full_model_sha256": full_model_sha256(model.encoder.state_dict(), model.projector.state_dict()),
                "source_images": ROTATION_TOTALS[rotation], "source_batch_size": 256,
                "representations_per_loss": 512, "temperature": 0.5,
                "activation_checkpointing": False, "channels_last": True,
                "runtime_seconds": runtime_seconds, "resume_history": resume_history}
    atomic_torch_save({"metadata": metadata, "encoder_state": model.encoder.state_dict(),
                       "projector_state": model.projector.state_dict(),
                       "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                       "scaler_state": scaler.state_dict(), "data_generator_state": generator.get_state(),
                       "global_step": global_step, "mean_epoch_loss": mean_loss}, path)


def smoke(steps: int = 6):
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    if not (ROOT / "preregistration_snapshot.json").is_file():
        raise RuntimeError("Preregistration must be frozen before smoke")
    assert_protected_unchanged()
    expected_species = {int(row["category_id"]) for row in upstream_rows("M1")}
    if len(expected_species) != 80:
        raise RuntimeError("M1 local selection does not contain 80 species")
    dataset, loader, _ = _loader("M1", 0)
    initial = _validate_initial(0)
    model = make_model(False)
    model.encoder.load_state_dict(initial["encoder_state"])
    model.projector.load_state_dict(initial["projector_state"])
    initial_full_hash = full_model_sha256(model.encoder.state_dict(), model.projector.state_dict())
    initial_parameter_hash = state_dict_sha256(dict(model.named_parameters()))
    model.cuda().to(memory_format=torch.channels_last).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    losses = []; optimizer_steps = 0; amp_overflow_steps = 0
    try:
        for index, ((view1, view2), _species, _image_id, _path) in enumerate(loader):
            if index >= steps:
                break
            if view1.shape[0] != 256:
                raise RuntimeError("Smoke source batch is not exactly 256")
            view1 = view1.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
            view2 = view2.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=True):
                features1, z1 = model(view1)
                _, z2 = model(view2)
                loss, targets = nt_xent(z1, z2, 0.5)
            if features1.shape != (256, 2048) or targets.shape != (512,):
                raise RuntimeError("Smoke feature/NT-Xent dimensions failed")
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            if scaler.get_scale() < scale_before:
                amp_overflow_steps += 1
            else:
                optimizer_steps += 1
            losses.append(float(loss.detach()))
        scheduler.step()
    except torch.cuda.OutOfMemoryError:
        record = {"status": "FAIL_OOM", "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "source_batch_size": 256,
                  "representations_per_loss": 512}
        write_json(ROOT / "smoke" / "smoke_result.json", record)
        raise RuntimeError(f"Literal source batch 256 OOM; smoke record: {record}")
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    final_parameter_hash = state_dict_sha256(dict(model.named_parameters()))
    optimizer_parameter_update = optimizer_steps > 0 and final_parameter_hash != initial_parameter_hash
    checkpoint = ROOT / "smoke" / "checkpoint_reload_test.pt"
    _save_checkpoint(checkpoint, "M1", 0, 0, model, optimizer, scheduler, scaler,
                     torch.Generator().manual_seed(0), len(losses), 0.0, losses[-1], [])
    saved = _checkpoint_valid(checkpoint, "M1", 0, required_epoch=0)
    reloaded = make_model(False)
    reloaded.encoder.load_state_dict(saved["encoder_state"])
    reloaded.projector.load_state_dict(saved["projector_state"])
    exact_reload = (state_dict_sha256(reloaded.encoder.state_dict()) == saved["metadata"]["encoder_sha256"] and
                    state_dict_sha256(reloaded.projector.state_dict()) == saved["metadata"]["projector_sha256"])
    # Deterministic evaluation and feature plumbing use canonical images.
    eval_ds = downstream_reference_dataset()
    first_a = eval_ds[0][0]; first_b = eval_ds[0][0]
    deterministic_eval = torch.equal(first_a, first_b)
    reloaded.encoder.cuda().eval()
    with torch.inference_mode():
        tiny = torch.stack([eval_ds[i][0] for i in range(4)]).cuda().contiguous(memory_format=torch.channels_last)
        features = reloaded.encoder(tiny).float().cpu()
    probe = nn.Linear(2048, 20).eval()
    with torch.inference_mode():
        logits = probe(features)
    # Orientation checks are algebraic engineering checks, not smoke AUROCs.
    ref = np.asarray([[1.0, 0.0], [0.99, 0.01]], dtype=np.float64)
    near = np.asarray([[1.0, 0.0]], dtype=np.float64); far = np.asarray([[-1.0, 0.0]], dtype=np.float64)
    def score(q):
        r = ref / np.linalg.norm(ref, axis=1, keepdims=True); q = q / np.linalg.norm(q, axis=1, keepdims=True)
        return float(np.mean(1.0 - q @ r.T))
    knn_orientation = score(far) > score(near)
    known = torch.tensor([[8.0, 0.0]]); unknown = torch.tensor([[0.0, 0.0]])
    energy_orientation = float(-torch.logsumexp(unknown, 1)) > float(-torch.logsumexp(known, 1))
    msp_orientation = float(1 - unknown.softmax(1).max()) > float(1 - known.softmax(1).max())
    record = {"status": "PASS", "steps": len(losses), "losses": losses,
              "source_pool_images": len(dataset), "source_species": len(expected_species),
              "source_batch_size": 256, "representations_per_loss": 512,
              "semantic_labels_passed_to_nt_xent": False, "amp": True, "channels_last": True,
              "optimizer_steps": optimizer_steps, "amp_overflow_steps": amp_overflow_steps,
              "optimizer_parameter_update": optimizer_parameter_update,
              "activation_checkpointing": False, "peak_allocated_bytes": peak_allocated,
              "peak_reserved_bytes": peak_reserved, "checkpoint_exact_reload": exact_reload,
              "deterministic_evaluation_transform": deterministic_eval,
              "feature_shape": list(features.shape), "probe_logits_shape": list(logits.shape),
              "knn_higher_is_ood": knn_orientation, "energy_higher_is_ood": energy_orientation,
              "msp_higher_is_ood": msp_orientation, "initial_full_model_sha256": initial_full_hash,
              "output_creation": checkpoint.is_file()}
    if not all([optimizer_parameter_update, exact_reload, deterministic_eval, features.shape == (4, 2048),
                logits.shape == (4, 20), knn_orientation, energy_orientation, msp_orientation]):
        record["status"] = "FAIL"
        write_json(ROOT / "smoke" / "smoke_result.json", record)
        raise RuntimeError(f"Smoke invariant failed: {record}")
    write_json(ROOT / "smoke" / "smoke_result.json", record)
    del model, reloaded
    torch.cuda.empty_cache()
    print(json.dumps(record, indent=2))
    return record


def train_one(rotation: str, seed: int):
    import torch
    if read_json(ROOT / "smoke" / "smoke_result.json")["status"] != "PASS":
        raise RuntimeError("Smoke gate has not passed")
    assert_protected_unchanged()
    final_path = _checkpoint_path(rotation, seed, 200)
    if final_path.is_file():
        _checkpoint_valid(final_path, rotation, seed, required_epoch=200)
        print(f"{rotation} seed{seed}: already complete", flush=True)
        return final_path
    dataset, loader, generator = _loader(rotation, seed)
    initial = _validate_initial(seed)
    model = make_model(False)
    model.encoder.load_state_dict(initial["encoder_state"])
    model.projector.load_state_dict(initial["projector_state"])
    if full_model_sha256(model.encoder.state_dict(), model.projector.state_dict()) != initial["metadata"]["full_model_sha256"]:
        raise RuntimeError("Loaded matched initialization mismatch")
    model.cuda().to(memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    start_epoch = 0; global_step = 0; elapsed_before = 0.0; resume_history = []
    run_dir = final_path.parent; run_dir.mkdir(parents=True, exist_ok=True)
    for candidate in sorted(run_dir.glob("epoch_*.pt"), reverse=True):
        try:
            payload = _checkpoint_valid(candidate, rotation, seed)
            epoch = int(payload["metadata"]["epoch"])
            if epoch >= 200:
                continue
            model.encoder.load_state_dict(payload["encoder_state"])
            model.projector.load_state_dict(payload["projector_state"])
            optimizer.load_state_dict(payload["optimizer_state"])
            scheduler.load_state_dict(payload["scheduler_state"])
            scaler.load_state_dict(payload["scaler_state"])
            generator.set_state(payload["data_generator_state"])
            start_epoch = epoch; global_step = int(payload["global_step"])
            elapsed_before = float(payload["metadata"].get("runtime_seconds", 0.0))
            resume_history = list(payload["metadata"].get("resume_history", [])) + [{"resumed_utc": now(), "checkpoint": str(candidate), "checkpoint_sha256": sha256_file(candidate)}]
            print(f"{rotation} seed{seed}: resuming epoch {epoch}", flush=True)
            break
        except Exception as exc:
            print(f"Ignoring invalid resume candidate {candidate}: {exc}", flush=True)
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    status = {"rotation": rotation, "seed": seed, "status": "RUNNING", "start_epoch": start_epoch,
              "source_images": len(dataset), "batches_per_epoch": len(loader), "started_utc": now(),
              "initial_encoder_sha256": initial["metadata"]["encoder_sha256"],
              "initial_projector_sha256": initial["metadata"]["projector_sha256"],
              "resume_history": resume_history}
    write_json(run_dir / "status.json", status)
    log_path = ROOT / "logs" / f"{rotation}_seed{seed}.jsonl"
    started = time.time()
    for epoch_index in range(start_epoch, 200):
        model.train(); epoch_started = time.time(); loss_sum = 0.0; batches = 0
        used_lr = optimizer.param_groups[0]["lr"]
        for (view1, view2), _selection_labels, _image_ids, _paths in loader:
            if view1.shape[0] != 256:
                raise RuntimeError("Scientific source batch changed from 256")
            view1 = view1.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
            view2 = view2.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=True):
                _, z1 = model(view1); _, z2 = model(view2); loss, targets = nt_xent(z1, z2, 0.5)
            if targets.shape != (512,) or not torch.isfinite(loss):
                raise RuntimeError("Invalid true-batch NT-Xent computation")
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            loss_sum += float(loss.detach()); batches += 1; global_step += 1
        scheduler.step()
        epoch = epoch_index + 1
        runtime = elapsed_before + time.time() - started
        record = {"timestamp_utc": now(), "rotation": rotation, "seed": seed, "epoch": epoch,
                  "mean_nt_xent_loss": loss_sum / batches, "lr_used": used_lr,
                  "lr_after_epoch": optimizer.param_groups[0]["lr"], "epoch_seconds": time.time() - epoch_started,
                  "runtime_seconds": runtime, "global_step": global_step, "batches": batches,
                  "source_images": len(dataset), "source_batch_size": 256, "representations_per_loss": 512}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)
        if epoch % 20 == 0 or epoch == 200:
            _save_checkpoint(_checkpoint_path(rotation, seed, epoch), rotation, seed, epoch, model,
                             optimizer, scheduler, scaler, generator, global_step, runtime,
                             record["mean_nt_xent_loss"], resume_history)
    status.update({"status": "COMPLETE", "completed_utc": now(), "final_checkpoint": str(final_path),
                   "final_checkpoint_sha256": sha256_file(final_path), "global_step": global_step,
                   "runtime_seconds": runtime, "resume_history": resume_history})
    write_json(run_dir / "status.json", status)
    refresh_training_tables()
    return final_path


def refresh_training_tables():
    loss_rows = []
    run_rows = []
    for seed in SEEDS:
        for rotation in MODELS:
            log = ROOT / "logs" / f"{rotation}_seed{seed}.jsonl"
            records = [json.loads(line) for line in log.read_text().splitlines() if line.strip()] if log.is_file() else []
            loss_rows.extend(records)
            status_path = ROOT / "checkpoints" / f"{rotation}_seed{seed}" / "status.json"
            status = read_json(status_path) if status_path.is_file() else {"rotation": rotation, "seed": seed, "status": "PENDING"}
            first = records[0] if records else {}; last = records[-1] if records else {}
            run_rows.append({"rotation": rotation, "seed": seed, "status": status["status"],
                             "epochs_completed": int(last.get("epoch", 0)), "source_images": ROTATION_TOTALS[rotation],
                             "batches_per_epoch": ROTATION_TOTALS[rotation] // 256,
                             "optimization_steps": int(last.get("global_step", 0)),
                             "initial_loss": first.get("mean_nt_xent_loss", ""),
                             "final_loss": last.get("mean_nt_xent_loss", ""),
                             "runtime_seconds": last.get("runtime_seconds", ""),
                             "resume_history_json": json.dumps(status.get("resume_history", []), sort_keys=True),
                             "final_checkpoint": status.get("final_checkpoint", ""),
                             "final_checkpoint_sha256": status.get("final_checkpoint_sha256", "")})
    write_csv(ROOT / "training_runs.csv", run_rows)
    if loss_rows:
        write_csv(ROOT / "training_losses.csv", loss_rows)


def train_all():
    for seed in SEEDS:
        for rotation in MODELS:
            train_one(rotation, seed)
    refresh_training_tables()


if __name__ == "__main__":
    train_all()
