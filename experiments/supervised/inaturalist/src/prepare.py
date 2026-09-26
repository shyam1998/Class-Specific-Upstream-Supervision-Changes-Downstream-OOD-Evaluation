from __future__ import annotations

import json
import shutil
import tarfile
from collections import Counter
from pathlib import Path

from common import (
    DATASET_ROOT, MINI_REFERENCE, MODELS, RECIPE_PATH, ROOT, SOURCE_INVENTORY,
    SOURCE_MANIFEST, SOURCE_MEMBER_LIST, environment_record, read_csv,
    sha256_file, utc_now, validate_manifest, write_csv, write_json,
)

EXPECTED_MANIFEST_HASH = "b5a8f1452c114123c8712095b966fb1824bea5942cc179e5fd4b5b8efb7010c6"
EXPECTED_ANNOTATION_HASH = "96d5f2e5c558c851e1101d8534d5ffaa553cdabd6b2c4c721b323f1be8a61b94"
EXPECTED_MEMBER_HASH = "04016c8b8aea9990e3822d2e80abc06972635ff1ab9e0ec95048bf30e2de892a"
EXPECTED_FULL_ARCHIVE_HASH = "5a9093b2174ac08538435d79f3c6fa4b5d40c72a49b7d9a7b24b5e48f9e4f5d9"
EXPECTED_ROTATION_IMAGES = {"M1": 22289, "M2": 22142, "M3": 22035, "M4": 22058}


def _load_json_archive(path: Path, member: str) -> dict:
    with tarfile.open(path, "r:gz") as tf:
        handle = tf.extractfile(member)
        if handle is None:
            raise FileNotFoundError(f"{member} missing from {path}")
        return json.load(handle)


def _copy_exact(source: Path, target: Path) -> None:
    if target.exists():
        if sha256_file(target) != sha256_file(source):
            raise RuntimeError(f"Frozen copy mismatch: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _write_once_json(path: Path, value: dict) -> None:
    if not path.exists():
        write_json(path, value)


def _full_index(inventory: list[dict[str, str]], annotation: dict) -> list[dict]:
    wanted = {r["archive_member_path"].replace("\\", "/"): r for r in inventory}
    selected_categories = {int(r["category_id"]) for r in inventory}
    category_by_image = {
        int(a["image_id"]): int(a["category_id"])
        for a in annotation["annotations"] if int(a["category_id"]) in selected_categories
    }
    category_meta = {
        int(c["id"]): c for c in annotation["categories"] if int(c["id"]) in selected_categories
    }
    rows, found = [], set()
    for image in annotation["images"]:
        name = str(image["file_name"]).replace("\\", "/")
        source = wanted.get(name)
        if source is None:
            continue
        image_id = int(image["id"])
        category_id = category_by_image.get(image_id)
        if category_id is None or category_id != int(source["category_id"]):
            raise RuntimeError(f"Official label mismatch for {name}")
        category = category_meta[category_id]
        rows.append({
            "split": "train_full", "image_id": image_id, "category_id": category_id,
            "file_name": name, "width": int(image["width"]), "height": int(image["height"]),
            "scientific_name": category["name"], "image_dir_name": category["image_dir_name"],
        })
        found.add(name)
    missing = set(wanted) - found
    if missing:
        raise FileNotFoundError(f"Full annotation missing {len(missing)} frozen members: {sorted(missing)[:3]}")
    rows.sort(key=lambda r: (r["category_id"], r["image_id"]))
    return rows


def _protected_sources() -> list[Path]:
    paths = [
        MINI_REFERENCE / "resolved_training_recipe.yaml",
        MINI_REFERENCE / "frozen_manifest.csv",
        MINI_REFERENCE / "initialization_audit.json",
        MINI_REFERENCE / "per_species_effects.csv",
    ]
    paths.extend(MINI_REFERENCE / f"rotation_{m}.csv" for m in MODELS)
    paths.extend((MINI_REFERENCE / "data_indices").glob("*.csv"))
    return sorted(paths)


def prepare() -> dict:
    required = [
        SOURCE_MANIFEST, SOURCE_INVENTORY, SOURCE_MEMBER_LIST, RECIPE_PATH,
        DATASET_ROOT / "train.json.tar.gz", DATASET_ROOT / "train.tar.gz",
        DATASET_ROOT / "train_mini.json.tar.gz", DATASET_ROOT / "val.json.tar.gz",
        DATASET_ROOT / "val.tar.gz",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(SOURCE_MANIFEST) != EXPECTED_MANIFEST_HASH:
        raise RuntimeError("Frozen semantic manifest hash mismatch")
    if sha256_file(DATASET_ROOT / "train.json.tar.gz") != EXPECTED_ANNOTATION_HASH:
        raise RuntimeError("Official full annotation hash mismatch")
    if sha256_file(SOURCE_MEMBER_LIST) != EXPECTED_MEMBER_HASH:
        raise RuntimeError("Frozen selected-member list hash mismatch")
    archive_hash_record = ROOT / "full_archive_sha256.txt"
    if not archive_hash_record.is_file() or archive_hash_record.read_text(encoding="utf-8").split()[0].lower() != EXPECTED_FULL_ARCHIVE_HASH:
        raise RuntimeError("Full image archive hash audit record missing or mismatched")

    _copy_exact(SOURCE_MANIFEST, ROOT / "manifest.csv")
    _copy_exact(SOURCE_MANIFEST, ROOT / "frozen_manifest.csv")
    for model in MODELS:
        _copy_exact(MINI_REFERENCE / f"rotation_{model}.csv", ROOT / f"rotation_{model}.csv")
    _copy_exact(MINI_REFERENCE / "data_indices" / "train_mini_selected.csv", ROOT / "data_indices" / "train_mini_selected.csv")
    _copy_exact(MINI_REFERENCE / "data_indices" / "val_selected.csv", ROOT / "data_indices" / "val_selected.csv")

    manifest = read_csv(ROOT / "manifest.csv")
    inventory_source = read_csv(SOURCE_INVENTORY)
    per_species = Counter(int(r["category_id"]) for r in inventory_source)
    if len(inventory_source) != 27713 or len(per_species) != 100:
        raise AssertionError("Frozen native inventory must contain 27,713 images from 100 species")
    if (min(per_species.values()), max(per_species.values())) != (162, 300):
        raise AssertionError("Unexpected native per-species count range")
    inventory = []
    for row in inventory_source:
        item = dict(row)
        item["full_train_images"] = per_species[int(row["category_id"])]
        inventory.append(item)
    write_csv(ROOT / "native_full_image_inventory.csv", inventory)

    annotation = _load_json_archive(DATASET_ROOT / "train.json.tar.gz", "train.json")
    if len(annotation["images"]) != 2686843 or len(annotation["annotations"]) != 2686843:
        raise AssertionError("Official full annotation count mismatch")
    full_index = _full_index(inventory, annotation)
    del annotation
    write_csv(ROOT / "data_indices" / "train_full_native_selected.csv", full_index)

    mini_index = read_csv(ROOT / "data_indices" / "train_mini_selected.csv")
    val_index = read_csv(ROOT / "data_indices" / "val_selected.csv")
    full_by_path = {r["file_name"]: r for r in full_index}
    membership = []
    for row in mini_index:
        normalized_full_path = row["file_name"].replace("train_mini/", "train/", 1)
        full = full_by_path.get(normalized_full_path)
        same = full is not None and int(full["category_id"]) == int(row["category_id"])
        membership.append({"image_id": row["image_id"], "category_id": row["category_id"],
                           "mini_file_name": row["file_name"], "full_file_name": normalized_full_path,
                           "present_in_full_same_species": same})
    write_csv(ROOT / "mini50_membership_verification.csv", membership)
    if len(membership) != 5000 or not all(r["present_in_full_same_species"] for r in membership):
        raise AssertionError("MINI-50 is not an exact same-species subset of full train")

    selected_root = ROOT / "selected_data"
    for row in full_index:
        path = selected_root / row["file_name"]
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    for row in mini_index:
        _copy_exact(MINI_REFERENCE / "selected_data" / row["file_name"], selected_root / row["file_name"])
    for row in val_index:
        _copy_exact(MINI_REFERENCE / "selected_data" / row["file_name"], selected_root / row["file_name"])
    if {int(r["image_id"]) for r in full_index} & {int(r["image_id"]) for r in val_index}:
        raise AssertionError("Full training and validation image IDs overlap")

    values = list(per_species.values())
    native_summary = {
        "created_utc": utc_now(), "images": len(full_index), "species": len(per_species),
        "minimum": min(values), "median": 300.0, "mean": sum(values) / len(values),
        "maximum": max(values), "per_species": {str(k): v for k, v in sorted(per_species.items())},
        "rotation_images": EXPECTED_ROTATION_IMAGES, "rotation_spread": 254,
        "rotation_spread_percent": 1.1477113551127378,
    }
    write_json(ROOT / "native_count_summary.json", native_summary)
    invariant = validate_manifest(manifest)
    invariant["checks"].update({
        "all_27713_selected_files_nonempty": True, "mini5000_subset_same_species": True,
        "full_train_validation_disjoint": True, "official_annotation_count": True,
        "rotation_label_maps_match_mini": all(
            sha256_file(ROOT / f"rotation_{m}.csv") == sha256_file(MINI_REFERENCE / f"rotation_{m}.csv") for m in MODELS),
    })
    if not all(invariant["checks"].values()):
        raise AssertionError(invariant)
    write_json(ROOT / "manifest_invariants.json", invariant)

    config = {
        "study": "supervised_inaturalist_rotation4",
        "data_policy": "all native full-train images for the frozen 100 species",
        "rotation_images": EXPECTED_ROTATION_IMAGES, "seeds": [0, 1], "rotations": list(MODELS),
        "epochs": 100, "batch_size": 256,
        "primary": {"detector": "cosine kNN", "k": 50, "delta": "withheld - supervised_mean"},
        "bootstrap": {"unit": "genus", "draws": 10000, "seed": 20260917},
    }
    write_json(ROOT / "config.json", config)
    write_json(ROOT / "manifest.json", {"source_sha256": EXPECTED_MANIFEST_HASH, "rows": manifest})
    protected = {str(p): sha256_file(p) for p in _protected_sources()}
    prereg = {
        "created_utc": utc_now(),
        "scientific_question": "Does the supervised same-class provenance effect persist with all native full-train images for the frozen species?",
        "frozen_manifest_sha256": EXPECTED_MANIFEST_HASH, "full_annotation_sha256": EXPECTED_ANNOTATION_HASH,
        "full_archive": str(DATASET_ROOT / "train.tar.gz"),
        "full_archive_size_bytes": (DATASET_ROOT / "train.tar.gz").stat().st_size,
        "full_archive_sha256": EXPECTED_FULL_ARCHIVE_HASH,
        "selected_member_list_sha256": EXPECTED_MEMBER_HASH, "selected_images": 27713,
        "native_counts": native_summary, "config": config, "recipe_sha256": sha256_file(RECIPE_PATH),
        "matched_initialization": "M1-M4 bitwise identical within seed and exact MINI state hashes",
        "downstream": "exact MINI identities: 1000 ID train, 200 ID validation, 800 OOD validation",
        "secondary_metrics_cannot_rescue_primary": True,
        "no_post_result_tuning_or_selective_rerunning": True,
        "decision_logic": [
            "FULL_NATIVE_CLEARLY_REPLICATED if mean<0 and CI upper<0",
            "FULL_NATIVE_DIRECTIONALLY_CONSISTENT_BUT_UNCERTAIN if mean<0 and CI includes zero",
            "FULL_NATIVE_NOT_REPLICATED if mean>=0",
        ],
        "protected_mini_hashes": protected,
    }
    _write_once_json(ROOT / "preregistration_snapshot.json", prereg)
    audit = {
        "created_utc": utc_now(), "status": "PASS", "environment": environment_record(),
        "fixed_samples": {"upstream_full_train": len(full_index), "downstream_id_mini_train": 1000,
                          "downstream_id_validation": 200, "downstream_ood_validation": 800, "overlap_count": 0},
        "protected_mini_hashes": protected,
        "implementation_source_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted((ROOT / "src").glob("*.py"))},
    }
    write_json(ROOT / "pretraining_data_audit.json", audit)
    result = {"status": "PREPARED", "invariants": invariant, "native_counts": native_summary}
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    prepare()
