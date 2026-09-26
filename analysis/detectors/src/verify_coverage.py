from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np

from .common import MODELS, ROOT, SEEDS, load_run, read_json, sha256_array, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("feature", "all"), required=True)
    args = parser.parse_args()
    expected_detectors = ["mahalanobis", "vim", "neco", "nci", "gradorth"] + (["odin"] if args.phase == "all" else [])
    runs = []
    failures = []
    for slug in ("cifar100", "imagenet", "inat"):
        for model in MODELS:
            for seed in SEEDS:
                data = load_run(slug, model, seed)
                tag = f"{model}_seed{seed}"
                paths = {"feature": ROOT / "raw_scores" / slug / f"{tag}_feature_detectors.npz"}
                if args.phase == "all": paths["odin"] = ROOT / "raw_scores" / slug / f"{tag}_odin.npz"
                run = {"dataset": slug, "rotation": model, "seed": seed, "evaluation_images": len(data.eval_ids), "detectors": {}}
                for kind, path in paths.items():
                    if not path.is_file():
                        failures.append(f"missing:{path}"); continue
                    metadata = read_json(path.with_suffix(".json"))
                    with np.load(path, allow_pickle=False) as source:
                        if not np.array_equal(source["evaluation_ids"], data.eval_ids) or not np.array_equal(source["class_ids"], data.eval_class_ids):
                            failures.append(f"identity:{path}")
                        names = ["odin"] if kind == "odin" else ["mahalanobis", "vim", "neco", "nci", "gradorth"]
                        for detector in names:
                            score = source[detector]
                            okay = len(score) == len(data.eval_ids) and np.isfinite(score).all() and metadata.get("status") == "PASS"
                            if not okay: failures.append(f"coverage:{path}:{detector}")
                            run["detectors"][detector] = {"count": len(score), "finite": bool(np.isfinite(score).all()), "sha256": sha256_array(score)}
                runs.append(run)
    for run in runs:
        if set(run["detectors"]) != set(expected_detectors):
            failures.append(f"detectors:{run['dataset']}:{run['rotation']}:seed{run['seed']}")
    value = {
        "status": "PASS" if not failures else "FAIL", "phase": args.phase,
        "created_utc": datetime.now(timezone.utc).isoformat(), "expected_runs": 32,
        "expected_detectors": expected_detectors, "run_count": len(runs), "failures": failures, "runs": runs,
    }
    path = ROOT / f"raw_score_coverage_{args.phase}.json"
    write_json(path, value)
    print(json.dumps({"status": value["status"], "phase": args.phase, "run_count": len(runs), "failures": failures}))
    if failures: raise SystemExit(1)


if __name__ == "__main__":
    main()
