from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .common import ROOT, canonical_state_hash, load_run, save_npz_atomic, sha256_array, sha256_file, write_json
from .models import load_encoder_probe, normalization_std, raw_evaluation_dataset


def odin_batch(model, images: torch.Tensor, temperature: float, epsilon: float, std: torch.Tensor):
    images = images.detach().requires_grad_(True)
    logits = model(images)
    predicted = logits.detach().argmax(dim=1)
    loss = F.cross_entropy(logits / temperature, predicted, reduction="sum")
    gradient = torch.autograd.grad(loss, images, only_inputs=True)[0]
    signed = torch.where(gradient >= 0, torch.ones_like(gradient), -torch.ones_like(gradient))
    perturbed = images.detach() - epsilon * signed / std
    with torch.inference_mode():
        perturbed_logits = model(perturbed)
        score = 1.0 - torch.softmax(perturbed_logits / temperature, dim=1).max(dim=1).values
    return score, logits.detach(), gradient.detach(), perturbed.detach()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("cifar100", "imagenet", "inat"))
    parser.add_argument("--model", required=True, choices=("M1", "M2", "M3", "M4"))
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("ODIN requires the audited CUDA device")
    started = time.time()
    data = load_run(args.dataset, args.model, args.seed)
    model, load_metadata = load_encoder_probe(data)
    before_encoder = canonical_state_hash(model.encoder.state_dict())
    before_probe = canonical_state_hash(model.probe.state_dict())
    model = model.to("cuda").eval()
    model.requires_grad_(False)

    dataset = raw_evaluation_dataset(data)
    batch_size = 512 if args.dataset == "cifar100" else 256
    epsilon = 0.002 if args.dataset == "cifar100" else 0.0014
    temperature = 1000.0
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, drop_last=False, persistent_workers=args.workers > 0,
    )
    std = torch.tensor(normalization_std(args.dataset), device="cuda", dtype=torch.float32).view(1, 3, 1, 1)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    scores = []
    observed_positions = []
    first_batch_audit = None
    for batch_index, batch in enumerate(loader):
        if args.dataset == "cifar100":
            images, labels, positions = batch
            expected = data.eval_class_ids[np.asarray(positions)]
            if not np.array_equal(np.asarray([str(x) for x in labels.tolist()]), expected):
                raise RuntimeError("CIFAR DataLoader labels/order mismatch")
        else:
            images, positions = batch
        images = images.to(device="cuda", dtype=torch.float32, non_blocking=True)
        raw_feature_audit = None
        if batch_index == 0:
            with torch.inference_mode():
                raw_features = model.features(images)
            cached_features = torch.from_numpy(data.eval_features[np.asarray(positions)]).to(device="cuda", dtype=torch.float32)
            raw_feature_audit = {
                "max_absolute_error_vs_canonical_cache": float(torch.max(torch.abs(raw_features - cached_features)).item()),
                "minimum_cosine_similarity_vs_canonical_cache": float(F.cosine_similarity(raw_features, cached_features, dim=1).min().item()),
            }
            raw_feature_audit["pass"] = raw_feature_audit["max_absolute_error_vs_canonical_cache"] <= 5e-3 and raw_feature_audit["minimum_cosine_similarity_vs_canonical_cache"] >= 0.99999
            if not raw_feature_audit["pass"]:
                raise RuntimeError(f"Raw-image transform/checkpoint features differ from canonical cache: {raw_feature_audit}")
        score, logits, gradient, perturbed = odin_batch(model, images, temperature, epsilon, std)
        if batch_index == 0:
            with torch.inference_mode():
                zero_second_logits = model(images.detach())
                zero_score = 1.0 - torch.softmax(zero_second_logits / temperature, dim=1).max(dim=1).values
                direct_score = 1.0 - torch.softmax(logits / temperature, dim=1).max(dim=1).values
            zero_error = float(torch.max(torch.abs(zero_score - direct_score)).item())
            channel_step = torch.mean(torch.abs((perturbed - images.detach()) * std), dim=(0, 2, 3)).cpu().numpy()
            first_batch_audit = {
                "epsilon_zero_max_absolute_error": zero_error,
                "epsilon_zero_pass": zero_error <= 1e-7,
                "raw_pixel_step_by_channel": channel_step.tolist(),
                "raw_pixel_step_max_error_from_epsilon": float(np.max(np.abs(channel_step - epsilon))),
                "raw_image_feature_match": raw_feature_audit,
                "gradient_finite": bool(torch.isfinite(gradient).all().item()),
                "score_finite": bool(torch.isfinite(score).all().item()),
            }
            first_batch_audit["channel_scaling_pass"] = first_batch_audit["raw_pixel_step_max_error_from_epsilon"] <= 1e-6
            if not first_batch_audit["epsilon_zero_pass"] or not first_batch_audit["channel_scaling_pass"] or not first_batch_audit["gradient_finite"] or not first_batch_audit["score_finite"]:
                raise RuntimeError(f"ODIN first-batch plumbing failed: {first_batch_audit}")
        scores.append(score.cpu().numpy().astype(np.float32))
        observed_positions.extend(positions.tolist())

    torch.cuda.synchronize()
    output = np.concatenate(scores)
    expected_positions = list(range(len(data.eval_ids)))
    if observed_positions != expected_positions or len(output) != len(data.eval_ids):
        raise RuntimeError("Incomplete or reordered ODIN raw-score coverage")
    if not np.isfinite(output).all():
        raise FloatingPointError("Non-finite ODIN output")
    after_encoder = canonical_state_hash(model.encoder.cpu().state_dict())
    after_probe = canonical_state_hash(model.probe.cpu().state_dict())
    if before_encoder != after_encoder or before_probe != after_probe:
        raise RuntimeError("ODIN mutated encoder/probe state")

    tag = f"{args.model}_seed{args.seed}"
    raw_path = ROOT / "raw_scores" / args.dataset / f"{tag}_odin.npz"
    save_npz_atomic(raw_path, evaluation_ids=data.eval_ids, class_ids=data.eval_class_ids, odin=output)
    metadata = {
        "status": "PASS", "dataset": data.dataset, "dataset_slug": args.dataset,
        "rotation": args.model, "seed": args.seed, "detector": "odin",
        "score_orientation": "higher_is_more_ood", "evaluation_images": len(output),
        "temperature": temperature, "epsilon_raw_pixel_units": epsilon,
        "batch_size": batch_size, "precision": "FP32", "clipping": False,
        "workers": args.workers, "runtime_seconds": time.time() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "evaluation_id_sha256": sha256_array(data.eval_ids),
        "evaluation_class_id_sha256": sha256_array(data.eval_class_ids),
        "raw_score_sha256": sha256_array(output), "raw_file": str(raw_path),
        "raw_file_sha256": sha256_file(raw_path), "first_batch_plumbing": first_batch_audit,
        "checkpoint": str(data.checkpoint_path), "checkpoint_sha256": sha256_file(data.checkpoint_path),
        "probe": str(data.probe_path), "probe_sha256": sha256_file(data.probe_path),
        "load_audit": load_metadata, "encoder_state_sha256_before": before_encoder,
        "encoder_state_sha256_after": after_encoder, "probe_state_sha256_before": before_probe,
        "probe_state_sha256_after": after_probe, "parameters_frozen": True, "model_eval": True,
    }
    write_json(raw_path.with_suffix(".json"), metadata)
    print(json.dumps({"status": "PASS", "dataset": args.dataset, "tag": tag,
                      "runtime_seconds": metadata["runtime_seconds"],
                      "peak_reserved_bytes": metadata["peak_cuda_reserved_bytes"]}))


if __name__ == "__main__":
    main()
