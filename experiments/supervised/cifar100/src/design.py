"""Frozen all-subclass leave-one-out role manifest and pretraining invariants."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from .common import file_hash, raw_cifar, write_immutable, write_json_immutable


ROLE_TO_C = {"a": "c1", "b": "c2", "r1": "c3", "r2": "c4"}
MODEL_WITHHELD = {"M1": "c1", "M2": "c2", "M3": "c3", "M4": "c4"}


def build_manifest(source_path: Path) -> dict:
    source = json.loads(Path(source_path).read_text(encoding="utf-8"))
    grouped = {}
    for row in source["classes"]:
        grouped.setdefault(int(row["coarse_class_index"]), []).append(row)
    groups = []
    for coarse_id in sorted(grouped):
        source_rows = grouped[coarse_id]
        by_role = {row["role"]: row for row in source_rows}
        if set(by_role) != {"d", "a", "b", "r1", "r2"}:
            raise AssertionError(f"Unexpected source roles for coarse group {coarse_id}")
        group = {
            "coarse_id": coarse_id,
            "coarse_name": by_role["d"]["coarse_class"],
            "d": {
                "fine_id": int(by_role["d"]["fine_class_index"]),
                "fine_name": by_role["d"]["fine_class"],
                "original_role": "d",
            },
        }
        for original_role, new_role in ROLE_TO_C.items():
            row = by_role[original_role]
            group[new_role] = {
                "fine_id": int(row["fine_class_index"]),
                "fine_name": row["fine_class"],
                "original_role": original_role,
                "withheld_model": f"M{int(new_role[1:])}",
            }
        group["withheld"] = {
            model: group[new_role]["fine_name"] for model, new_role in MODEL_WITHHELD.items()
        }
        groups.append(group)

    model_classes = {}
    for model, withheld_role in MODEL_WITHHELD.items():
        selected = []
        for group in groups:
            selected.append(group["d"]["fine_id"])
            selected.extend(
                group[role]["fine_id"]
                for role in ("c1", "c2", "c3", "c4")
                if role != withheld_role
            )
        model_classes[model] = sorted(selected)
    return {
        "source_semantic_manifest": str(Path(source_path).resolve()),
        "source_semantic_manifest_sha256": file_hash(source_path),
        "source_confirmation_split_seed": source.get("confirmation_split_seed"),
        "mapping": ROLE_TO_C,
        "model_withheld_roles": MODEL_WITHHELD,
        "groups": groups,
        "model_pretraining_classes": model_classes,
        "downstream_id_classes": sorted(group["d"]["fine_id"] for group in groups),
        "downstream_ood_candidate_classes": sorted(
            group[role]["fine_id"] for group in groups for role in ("c1", "c2", "c3", "c4")
        ),
    }


def manifest_csv(manifest: dict) -> pd.DataFrame:
    rows = []
    for group in manifest["groups"]:
        row = {"coarse_id": group["coarse_id"], "coarse_name": group["coarse_name"]}
        for role in ("d", "c1", "c2", "c3", "c4"):
            row[f"{role}_id"] = group[role]["fine_id"]
            row[f"{role}_name"] = group[role]["fine_name"]
            row[f"{role}_original_role"] = group[role]["original_role"]
        for model in MODEL_WITHHELD:
            new_role = MODEL_WITHHELD[model]
            row[f"{model}_withheld_id"] = group[new_role]["fine_id"]
            row[f"{model}_withheld_name"] = group[new_role]["fine_name"]
        rows.append(row)
    return pd.DataFrame(rows)


def validate_manifest(manifest: dict, data_dir: Path) -> dict:
    groups = manifest["groups"]
    all_roles = [group[role]["fine_id"] for group in groups for role in ("d", "c1", "c2", "c3", "c4")]
    d_classes = set(manifest["downstream_id_classes"])
    candidates = set(manifest["downstream_ood_candidate_classes"])
    model_sets = {model: set(classes) for model, classes in manifest["model_pretraining_classes"].items()}
    train = raw_cifar(data_dir, True, transform=None, download=True)
    target_counts = Counter(map(int, train.targets))
    model_examples = {
        model: sum(target_counts[class_id] for class_id in classes)
        for model, classes in model_sets.items()
    }
    checks = {
        "01_exactly_20_coarse_groups": len(groups) == 20,
        "02_each_group_has_5_distinct_classes": all(
            len({group[role]["fine_id"] for role in ("d", "c1", "c2", "c3", "c4")}) == 5
            for group in groups
        ),
        "03_d_c1_c2_c3_c4_distinct_within_group": all(
            len({group[role]["fine_id"] for role in ("d", "c1", "c2", "c3", "c4")}) == 5
            for group in groups
        ),
        "04_all_100_fine_classes_appear_exactly_once": len(all_roles) == 100 and set(all_roles) == set(range(100)),
        "05_d_identical_across_M1_M4": all(d_classes.issubset(classes) for classes in model_sets.values()),
        "06_each_model_has_exactly_80_classes": all(len(classes) == 80 for classes in model_sets.values()),
        "07_each_model_contains_all_20_d_classes": len(d_classes) == 20 and all(d_classes.issubset(classes) for classes in model_sets.values()),
        "08_each_candidate_absent_from_exactly_one_model": all(sum(c not in classes for classes in model_sets.values()) == 1 for c in candidates),
        "09_each_candidate_present_in_exactly_three_models": all(sum(c in classes for classes in model_sets.values()) == 3 for c in candidates),
        "10_identical_class_and_sample_counts": len(set(map(len, model_sets.values()))) == 1 and set(model_examples.values()) == {40000},
        "11_downstream_id_only_20_d_classes": len(d_classes) == 20 and d_classes.isdisjoint(candidates),
        "12_all_80_candidates_always_downstream_ood": len(candidates) == 80 and candidates == set(range(100)) - d_classes,
    }
    passed = all(checks.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "all_passed": passed,
        "checks": checks,
        "model_class_counts": {model: len(classes) for model, classes in model_sets.items()},
        "model_sample_counts": model_examples,
        "downstream_id_count": len(d_classes),
        "candidate_ood_count": len(candidates),
    }


def freeze_design(project: Path, source_path: Path, data_dir: Path) -> tuple[dict, dict]:
    manifest = build_manifest(source_path)
    invariants = validate_manifest(manifest, data_dir)
    if not invariants["all_passed"]:
        raise RuntimeError(f"STOP BEFORE TRAINING: manifest invariant failure: {invariants}")
    write_json_immutable(project / "manifest.json", manifest)
    csv_text = manifest_csv(manifest).to_csv(index=False, lineterminator="\n")
    write_immutable(project / "manifest.csv", csv_text)
    return manifest, invariants
