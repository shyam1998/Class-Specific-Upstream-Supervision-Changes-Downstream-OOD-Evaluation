from __future__ import annotations

import argparse
import json
import subprocess
import sys

from .common import MODELS, ROOT, SEEDS, read_json


def run_tests():
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                   cwd=ROOT, check=True)


def status():
    rows = []
    for seed in SEEDS:
        for model in MODELS:
            path = ROOT / "checkpoints" / f"{model}_seed{seed}" / "status.json"
            rows.append(read_json(path) if path.is_file() else {"rotation": model, "seed": seed, "status": "PENDING"})
    print(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["tests", "smoke", "train", "evaluate", "analyze", "all", "status"])
    args = parser.parse_args()
    if args.stage == "status":
        status(); return
    if args.stage in ("tests", "all"):
        run_tests()
    if args.stage == "tests": return
    if args.stage in ("smoke", "all") and not (ROOT / "smoke" / "smoke_result.json").is_file():
        from .train import smoke
        smoke()
    if args.stage == "smoke": return
    if args.stage in ("train", "all"):
        from .train import train_all
        train_all()
    if args.stage == "train": return
    if args.stage in ("evaluate", "all"):
        from .evaluate import evaluate_all
        evaluate_all()
    if args.stage == "evaluate": return
    if args.stage in ("analyze", "all"):
        from .analyze import analyze
        analyze()


if __name__ == "__main__":
    main()
