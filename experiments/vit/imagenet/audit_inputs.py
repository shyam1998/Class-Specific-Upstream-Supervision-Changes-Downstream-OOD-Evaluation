#!/usr/bin/env python3
"""Read-only identity audit for the controlled-ImageNet ViT experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
SOURCE = PROJECT / "detector_suite_inputs/imagenet_supervised/experiment"
CONFIG = ROOT / "configs/config.json"
MODELS = ("M1", "M2", "M3", "M4")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main():
    cfg = json.loads(CONFIG.read_text())
    manifest_path = SOURCE / "manifests/rotation4_v1/semantic_groups_rotation4_v1.csv"
    manifest_json = SOURCE / "manifests/rotation4_v1/semantic_groups_rotation4_v1.json"
    prereg_path = SOURCE / "manifests/rotation4_v1/preregistration.json"
    ref_path = SOURCE / "detector_handoff/id_reference_images.csv"
    eval_path = SOURCE / "detector_handoff/eval_images.csv"
    manifest = rows(manifest_path)
    reference = rows(ref_path)
    evaluation = rows(eval_path)
    roles = {r["role"] for r in manifest}
    groups = {r["group_id"] for r in manifest}
    wnids = {r["wnid"] for r in manifest}
    by_model = {m: {r["wnid"] for r in manifest if r[f"included_{m}"] == "True"} for m in MODELS}
    non_id = [r for r in manifest if r["role"] != "d"]
    data_root = Path(os.environ.get("IMAGENET_ROOT", cfg["data_root"])).expanduser()
    missing = []
    if data_root.is_dir():
        for r in reference + evaluation:
            if not (data_root / r["relative_path"]).is_file():
                missing.append(r["relative_path"])
                if len(missing) >= 100:
                    break
    checks = {
        "canonical_files_present": all(p.is_file() for p in (manifest_path, manifest_json, prereg_path, ref_path, eval_path)),
        "manifest_rows_100": len(manifest) == 100,
        "semantic_groups_20": len(groups) == 20,
        "roles_exact": roles == {"d", "a", "b", "r1", "r2"},
        "unique_classes_100": len(wnids) == 100,
        "downstream_id_classes_20": sum(r["role"] == "d" for r in manifest) == 20,
        "future_ood_classes_80": len(non_id) == 80,
        "each_rotation_80_classes": all(len(x) == 80 for x in by_model.values()),
        "each_ood_withheld_once": all(sum(r["wnid"] not in by_model[m] for m in MODELS) == 1 for r in non_id),
        "each_ood_supervised_three_times": all(sum(r["wnid"] in by_model[m] for m in MODELS) == 3 for r in non_id),
        "seeds_exact": cfg["seeds"] == [0, 1],
        "reference_rows_match_preregistered": len(reference) == 25743,
        "evaluation_rows_5000": len(evaluation) == 5000,
        "evaluation_100_classes_50_each": len({r["wnid"] for r in evaluation}) == 100 and all(sum(x["wnid"] == w for x in evaluation) == 50 for w in wnids),
        "data_root_present": data_root.is_dir(),
        "all_frozen_identity_paths_present": data_root.is_dir() and not missing,
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "BLOCKED_MISSING_DATASET" if not checks["data_root_present"] else "FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "resolved_data_root": str(data_root),
        "missing_identity_paths_sample": missing,
        "counts": {"manifest_rows": len(manifest), "groups": len(groups), "reference_images": len(reference), "evaluation_images": len(evaluation)},
        "hashes": {"manifest_csv": digest(manifest_path), "manifest_json": digest(manifest_json), "preregistration": digest(prereg_path), "id_reference_images": digest(ref_path), "eval_images": digest(eval_path), "config": digest(CONFIG)},
        "no_ood_evaluation": True,
    }
    write(ROOT / "canonical_input_audit.json", payload)
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
