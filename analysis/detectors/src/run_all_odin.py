from __future__ import annotations

import json
import os
import subprocess
import sys

from .common import MODELS, ROOT, SEEDS, read_json


def valid(slug: str, model: str, seed: int) -> bool:
    path = ROOT / "raw_scores" / slug / f"{model}_seed{seed}_odin.json"
    return path.is_file() and read_json(path).get("status") == "PASS"


def main() -> None:
    jobs = [(slug, model, seed) for slug in ("cifar100", "imagenet", "inat") for model in MODELS for seed in SEEDS if not valid(slug, model, seed)]
    print(json.dumps({"gpu_worker_limit": 1, "pending_jobs": len(jobs), "already_valid": 32 - len(jobs)}), flush=True)
    failures = []
    for slug, model, seed in jobs:
        command = [sys.executable, "-m", "src.run_odin", "--dataset", slug, "--model", model, "--seed", str(seed)]
        environment = os.environ.copy(); environment["PYTHONPATH"] = "."
        result = subprocess.run(command, cwd=ROOT, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (ROOT / "logs" / f"odin_{slug}_{model}_seed{seed}.log").write_text(result.stdout, encoding="utf-8")
        record = {"dataset": slug, "model": model, "seed": seed, "returncode": result.returncode, "output": result.stdout[-2000:]}
        print(json.dumps(record), flush=True)
        if result.returncode != 0:
            failures.append(record)
            break
    if failures:
        raise RuntimeError(f"ODIN failure; preserved and stopped: {failures}")
    if not all(valid(slug, model, seed) for slug in ("cifar100", "imagenet", "inat") for model in MODELS for seed in SEEDS):
        raise RuntimeError("ODIN raw-score coverage incomplete")
    print(json.dumps({"status": "PASS", "completed_jobs": 32, "gpu_worker_limit": 1}), flush=True)


if __name__ == "__main__":
    main()
