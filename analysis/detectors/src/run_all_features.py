from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

from .common import MODELS, ROOT, SEEDS, read_json


def valid(slug: str, model: str, seed: int) -> bool:
    path = ROOT / "raw_scores" / slug / f"{model}_seed{seed}_feature_detectors.json"
    return path.is_file() and read_json(path).get("status") == "PASS"


def run(job: tuple[str, str, int]) -> dict:
    slug, model, seed = job
    log_path = ROOT / "logs" / f"feature_{slug}_{model}_seed{seed}.log"
    command = [
        sys.executable,
        "-m",
        "src.run_feature_shard",
        "--dataset",
        slug,
        "--model",
        model,
        "--seed",
        str(seed),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": ".",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "OMP_NESTED": "FALSE",
        }
    )
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(result.stdout, encoding="utf-8")
    return {
        "dataset": slug,
        "model": model,
        "seed": seed,
        "returncode": result.returncode,
        "output": result.stdout[-2000:],
    }


def main() -> None:
    jobs = [
        (slug, model, seed)
        for slug in ("cifar100", "imagenet", "inat")
        for model in MODELS
        for seed in SEEDS
        if not valid(slug, model, seed)
    ]
    failures = []
    print(
        json.dumps({"worker_limit": 4, "pending_jobs": len(jobs), "already_valid": 32 - len(jobs)}),
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(json.dumps(result), flush=True)
            if result["returncode"] != 0:
                failures.append(result)
    if failures:
        raise RuntimeError(f"Feature shard failures: {failures}")
    if not all(
        valid(slug, model, seed)
        for slug in ("cifar100", "imagenet", "inat")
        for model in MODELS
        for seed in SEEDS
    ):
        raise RuntimeError("Feature raw-score coverage incomplete")
    print(json.dumps({"status": "PASS", "completed_jobs": 32}), flush=True)


if __name__ == "__main__":
    main()
