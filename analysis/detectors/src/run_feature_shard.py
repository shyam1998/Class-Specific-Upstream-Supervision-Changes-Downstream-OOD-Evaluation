from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .common import ROOT, load_run, save_npz_atomic, sha256_array, sha256_file, write_json
from .detectors import (
    fit_and_score_all,
    score_gradorth,
    score_mahalanobis,
    score_nci,
    score_neco,
    score_vim,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("cifar100", "imagenet", "inat"))
    parser.add_argument("--model", required=True, choices=("M1", "M2", "M3", "M4"))
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1))
    parser.add_argument("--chunk-size", type=int, default=1024)
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    started = time.time()
    data = load_run(args.dataset, args.model, args.seed)
    tag = f"{args.model}_seed{args.seed}"
    fit_root = ROOT / "fit_states" / args.dataset / tag
    raw_root = ROOT / "raw_scores" / args.dataset
    fit_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    states, scores = fit_and_score_all(
        data.reference_features,
        data.reference_class_ids,
        data.eval_features,
        data.probe_weight,
        data.probe_bias,
        chunk_size=args.chunk_size,
    )

    score_functions = {
        "mahalanobis": lambda x, state, chunk: score_mahalanobis(x, state, chunk),
        "vim": lambda x, state, chunk: score_vim(
            x, data.probe_weight, data.probe_bias, state, chunk
        ),
        "neco": lambda x, state, chunk: score_neco(x, state, chunk),
        "nci": lambda x, state, chunk: score_nci(
            x, data.probe_weight, data.probe_bias, state, chunk
        ),
        "gradorth": lambda x, state, chunk: score_gradorth(
            x, data.probe_weight, data.probe_bias, state, chunk
        ),
    }
    invariance = {}
    check_count = min(257, len(data.eval_features))
    for detector, (arrays, metadata) in states.items():
        fit_path = fit_root / f"{detector}.npz"
        save_npz_atomic(fit_path, **arrays)
        repeated = score_functions[detector](data.eval_features[:check_count], arrays, 31)
        error = float(np.max(np.abs(repeated - scores[detector][:check_count])))
        relative_error = float(
            np.max(
                np.abs(repeated - scores[detector][:check_count])
                / np.maximum(np.abs(scores[detector][:check_count]), 1e-300)
            )
        )
        passed = bool(np.allclose(repeated, scores[detector][:check_count], rtol=1e-12, atol=1e-6))
        invariance[detector] = {
            "records": check_count,
            "alternative_chunk_size": 31,
            "max_absolute_error": error,
            "max_relative_error": relative_error,
            "absolute_tolerance": 1e-6,
            "relative_tolerance": 1e-12,
            "pass": passed,
        }
        if not invariance[detector]["pass"]:
            raise RuntimeError(
                f"Chunk-size invariance failed: {data.dataset}/{tag}/{detector}: {error}"
            )
        metadata = {
            **metadata,
            "dataset": data.dataset,
            "dataset_slug": args.dataset,
            "rotation": args.model,
            "seed": args.seed,
            "score_orientation": "higher_is_more_ood",
            "reference_feature_sha256": sha256_array(data.reference_features),
            "reference_class_id_sha256": sha256_array(data.reference_class_ids),
            "probe_weight_sha256": sha256_array(data.probe_weight),
            "probe_bias_sha256": sha256_array(data.probe_bias),
            "fit_file": str(fit_path),
            "fit_file_sha256": sha256_file(fit_path),
            "chunk_size_invariance": invariance[detector],
        }
        write_json(fit_root / f"{detector}.json", metadata)

    raw_path = raw_root / f"{tag}_feature_detectors.npz"
    save_npz_atomic(
        raw_path,
        evaluation_ids=data.eval_ids,
        class_ids=data.eval_class_ids,
        mahalanobis=scores["mahalanobis"],
        vim=scores["vim"],
        neco=scores["neco"],
        nci=scores["nci"],
        gradorth=scores["gradorth"],
    )
    finite = {name: bool(np.isfinite(value).all()) for name, value in scores.items()}
    if not all(finite.values()):
        raise FloatingPointError(f"Non-finite raw scores: {finite}")
    metadata = {
        "status": "PASS",
        "dataset": data.dataset,
        "dataset_slug": args.dataset,
        "rotation": args.model,
        "seed": args.seed,
        "detectors": sorted(scores),
        "score_orientation": "higher_is_more_ood",
        "evaluation_images": len(data.eval_ids),
        "evaluation_id_sha256": sha256_array(data.eval_ids),
        "evaluation_class_id_sha256": sha256_array(data.eval_class_ids),
        "raw_score_sha256": {name: sha256_array(value) for name, value in scores.items()},
        "finite": finite,
        "raw_file": str(raw_path),
        "raw_file_sha256": sha256_file(raw_path),
        "source_hashes": {
            "checkpoint": sha256_file(data.checkpoint_path),
            "probe": sha256_file(data.probe_path),
            "features": {str(path): sha256_file(path) for path in data.feature_paths},
        },
        "runtime_seconds": time.time() - started,
    }
    write_json(raw_path.with_suffix(".json"), metadata)
    print(
        json.dumps(
            {
                "status": "PASS",
                "dataset": args.dataset,
                "tag": tag,
                "runtime_seconds": metadata["runtime_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
