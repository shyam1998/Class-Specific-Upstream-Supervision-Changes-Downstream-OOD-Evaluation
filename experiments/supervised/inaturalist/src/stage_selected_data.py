from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(os.environ.get("INAT_ARCHIVE_ROOT", "data/inaturalist"))
DESTINATION_ROOT = ROOT / "selected_data"
AUDIT_PATH = ROOT / "selected_data_staging_audit.json"
INDEX_FILES = {
    "full_train": ROOT / "data_indices" / "train_full_native_selected.csv",
    "mini_train": ROOT / "data_indices" / "train_mini_selected.csv",
    "validation": ROOT / "data_indices" / "val_selected.csv",
}
EXPECTED_COUNTS = {"full_train": 27_713, "mini_train": 5_000, "validation": 1_000}
EXPECTED_ARCHIVE_HASHES = {
    "train.tar.gz": "5a9093b2174ac08538435d79f3c6fa4b5d40c72a49b7d9a7b24b5e48f9e4f5d9",
    "train.json.tar.gz": "96d5f2e5c558c851e1101d8534d5ffaa553cdabd6b2c4c721b323f1be8a61b94",
}
PROTECTED_FILES = [
    ROOT / "frozen_manifest.csv",
    ROOT / "manifest.json",
    ROOT / "config.json",
    ROOT / "preregistration_snapshot.json",
    ROOT / "resolved_training_recipe.yaml",
    ROOT / "manifest_invariants.json",
    ROOT / "native_count_summary.json",
    ROOT / "pretraining_data_audit.json",
    ROOT / "initialization_audit.json",
    *(ROOT / f"rotation_M{number}.csv" for number in range(1, 5)),
    *INDEX_FILES.values(),
]


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_index(name: str, rows: list[dict[str, str]], prefix: str) -> None:
    if len(rows) != EXPECTED_COUNTS[name]:
        raise RuntimeError(f"{name}: expected {EXPECTED_COUNTS[name]} rows, got {len(rows)}")
    paths = [row["file_name"].replace("\\", "/") for row in rows]
    if len(paths) != len(set(paths)):
        raise RuntimeError(f"{name}: duplicate indexed paths")
    for row, relative in zip(rows, paths):
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 3:
            raise RuntimeError(f"{name}: unsafe or malformed frozen path: {relative}")
        if pure.parts[0] != prefix:
            raise RuntimeError(f"{name}: unexpected path prefix: {relative}")
        if pure.parts[1] != row["image_dir_name"]:
            raise RuntimeError(f"{name}: frozen directory metadata mismatch: {relative}")
        try:
            path_category = int(pure.parts[1].split("_", 1)[0])
        except Exception as exc:
            raise RuntimeError(f"{name}: cannot parse species ID from {relative}") from exc
        if path_category != int(row["category_id"]):
            raise RuntimeError(f"{name}: path/species mismatch: {relative}")


def selective_extract(archive: Path, members: list[str], destination: Path) -> dict[str, Any]:
    missing = [member for member in members if not (destination / member).is_file()]
    if not missing:
        return {"archive": str(archive), "requested": len(members), "extracted_now": 0, "command": None}
    # If a prior extraction was interrupted, its final existing file could be
    # partial. Re-request the entire frozen member list so tar fully
    # reconstructs every selected member from the authoritative archive.
    requested_members = members
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix="inat_members_", suffix=".txt", dir=ROOT, delete=False
    ) as handle:
        list_path = Path(handle.name)
        for member in requested_members:
            handle.write(member + "\n")
    command = [
        "tar", "--extract", "--use-compress-program=pigz", f"--file={archive}", f"--directory={destination}",
        "--no-recursion", "--verbatim-files-from", f"--files-from={list_path}",
        "--no-same-owner", "--no-same-permissions",
    ]
    try:
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Selective extraction failed for {archive} with exit {completed.returncode}: "
                f"{completed.stderr[-4000:]}"
            )
    finally:
        list_path.unlink(missing_ok=True)
    unresolved = [member for member in members if not (destination / member).is_file()]
    if unresolved:
        raise RuntimeError(f"Archive did not resolve {len(unresolved)} exact members; first: {unresolved[0]}")
    return {
        "archive": str(archive), "requested": len(members), "extracted_now": len(requested_members),
        "command": " ".join(str(value) for value in command[:-3] + command[-2:]),
    }


def decode_one(path: Path) -> tuple[str, str] | None:
    try:
        with Image.open(path) as image:
            image.load()
            image.convert("RGB")
        return None
    except Exception as exc:
        return str(path), repr(exc)


def stage() -> dict[str, Any]:
    rows = {name: read_index(path) for name, path in INDEX_FILES.items()}
    validate_index("full_train", rows["full_train"], "train")
    validate_index("mini_train", rows["mini_train"], "train_mini")
    validate_index("validation", rows["validation"], "val")
    protected_before = {str(path.relative_to(ROOT)): sha256_file(path) for path in PROTECTED_FILES}

    full_by_path = {row["file_name"]: row for row in rows["full_train"]}
    mini_archive_members: list[str] = []
    for mini in rows["mini_train"]:
        member = mini["file_name"].replace("train_mini/", "train/", 1)
        full = full_by_path.get(member)
        if full is None:
            raise RuntimeError(f"Frozen MINI identity is absent from native-full index: {mini['file_name']}")
        identity = (mini["image_id"], mini["category_id"], Path(mini["file_name"]).name)
        full_identity = (full["image_id"], full["category_id"], Path(full["file_name"]).name)
        if identity != full_identity:
            raise RuntimeError(f"Frozen MINI/native-full identity mismatch: {mini['file_name']}")
        mini_archive_members.append(member)

    full_paths = [row["file_name"] for row in rows["full_train"]]
    mini_paths = [row["file_name"] for row in rows["mini_train"]]
    validation_paths = [row["file_name"] for row in rows["validation"]]
    all_literal_paths = full_paths + mini_paths + validation_paths
    if len(all_literal_paths) != len(set(all_literal_paths)):
        raise RuntimeError("Duplicate literal path across frozen indexes")
    train_ids = {row["image_id"] for row in rows["full_train"]} | {row["image_id"] for row in rows["mini_train"]}
    validation_ids = {row["image_id"] for row in rows["validation"]}
    train_val_overlap = len(train_ids & validation_ids)
    if train_val_overlap:
        raise RuntimeError(f"Frozen train/validation identity overlap: {train_val_overlap}")

    archives = {
        "train.tar.gz": SOURCE_ROOT / "train.tar.gz",
        "train.json.tar.gz": SOURCE_ROOT / "train.json.tar.gz",
        "val.tar.gz": SOURCE_ROOT / "val.tar.gz",
    }
    for path in archives.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    archive_records = {}
    for name, path in archives.items():
        actual_hash = sha256_file(path)
        expected_hash = EXPECTED_ARCHIVE_HASHES.get(name)
        if expected_hash is not None and actual_hash != expected_hash:
            raise RuntimeError(f"Official source archive hash mismatch for {path}")
        archive_records[name] = {
            "path": str(path), "size_bytes": path.stat().st_size,
            "sha256": actual_hash, "expected_sha256": expected_hash,
            "expected_hash_match": None if expected_hash is None else True,
            "used_for": {
                "train.tar.gz": ["native-full train", "MINI train exact identity subset"],
                "train.json.tar.gz": ["official full-train annotation provenance"],
                "val.tar.gz": ["validation"],
            }[name],
        }

    DESTINATION_ROOT.mkdir(parents=True, exist_ok=True)
    expected_set = set(all_literal_paths)
    unexpected_existing = [
        str(path.relative_to(DESTINATION_ROOT)).replace(os.sep, "/")
        for path in DESTINATION_ROOT.rglob("*") if path.is_file()
        and str(path.relative_to(DESTINATION_ROOT)).replace(os.sep, "/") not in expected_set
    ]
    if unexpected_existing:
        raise RuntimeError(f"Unexpected pre-existing selected_data file: {unexpected_existing[0]}")

    extraction = {
        "full_train": selective_extract(archives["train.tar.gz"], full_paths, DESTINATION_ROOT),
        "validation": selective_extract(archives["val.tar.gz"], validation_paths, DESTINATION_ROOT),
    }
    linked_now = 0
    for mini_path, member in zip(mini_paths, mini_archive_members):
        source = DESTINATION_ROOT / member
        destination = DESTINATION_ROOT / mini_path
        if not source.is_file():
            raise RuntimeError(f"Exact MINI source identity was not extracted: {member}")
        if destination.exists():
            if not destination.is_file() or not os.path.samefile(source, destination):
                raise RuntimeError(f"Existing MINI destination is not the exact staged identity: {mini_path}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, destination)
        linked_now += 1
    extraction["mini_train"] = {
        "archive": str(archives["train.tar.gz"]),
        "archive_member_mapping": "train_mini/... -> exact train/... member",
        "identity_validation": "image_id, category_id, and filename exact",
        "hardlinks_created_now": linked_now,
        "requested": len(mini_paths),
    }

    missing = [relative for relative in all_literal_paths if not (DESTINATION_ROOT / relative).is_file()]
    staged_counts = {
        name: sum((DESTINATION_ROOT / row["file_name"]).is_file() for row in split_rows)
        for name, split_rows in rows.items()
    }
    decode_paths = [DESTINATION_ROOT / relative for relative in all_literal_paths]
    with ThreadPoolExecutor(max_workers=16) as executor:
        decode_errors = [value for value in executor.map(decode_one, decode_paths, chunksize=16) if value is not None]
    protected_after = {str(path.relative_to(ROOT)): sha256_file(path) for path in PROTECTED_FILES}
    protected_unchanged = protected_before == protected_after
    status = "PASS" if (
        staged_counts == EXPECTED_COUNTS and not missing and not decode_errors
        and train_val_overlap == 0 and protected_unchanged
    ) else "FAIL"
    audit = {
        "status": status,
        "destination_root": str(DESTINATION_ROOT),
        "source_archives": archive_records,
        "authoritative_indexes": {
            name: {
                "path": str(path), "file_sha256": sha256_file(path),
                "frozen_path_list_sha256": sha256_lines([row["file_name"] for row in rows[name]]),
                "rows": len(rows[name]), "unique_paths": len({row["file_name"] for row in rows[name]}),
            }
            for name, path in INDEX_FILES.items()
        },
        "extraction": extraction,
        "selected_file_counts": staged_counts,
        "missing_files": len(missing), "missing_examples": missing[:10],
        "decode_errors": len(decode_errors), "decode_error_examples": decode_errors[:10],
        "duplicate_indexed_paths": len(all_literal_paths) - len(set(all_literal_paths)),
        "train_validation_overlap": train_val_overlap,
        "species_identity_checks": {
            "path_category_matches_frozen_category_id": True,
            "path_directory_matches_frozen_image_dir_name": True,
            "mini_to_full_exact_identity_matches": len(mini_paths),
            "mini_to_full_identity_mismatches": 0,
        },
        "protected_artifact_hashes_before": protected_before,
        "protected_artifact_hashes_after": protected_after,
        "protected_artifacts_unchanged": protected_unchanged,
        "training_launched": False,
    }
    with AUDIT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    if status != "PASS":
        raise RuntimeError(f"Selected-data staging validation failed; see {AUDIT_PATH}")
    return {
        "full_train_staged": staged_counts["full_train"],
        "mini_train_staged": staged_counts["mini_train"],
        "validation_staged": staged_counts["validation"],
        "missing_files": len(missing), "decode_errors": len(decode_errors),
        "train_val_overlap": train_val_overlap, "selected_data_root": str(DESTINATION_ROOT),
        "staging_audit": str(AUDIT_PATH), "resume_ready": True,
    }


def main() -> None:
    result = stage()
    print(f"FULL_TRAIN_STAGED: {result['full_train_staged']}")
    print(f"MINI_TRAIN_STAGED: {result['mini_train_staged']}")
    print(f"VALIDATION_STAGED: {result['validation_staged']}")
    print(f"MISSING_FILES: {result['missing_files']}")
    print(f"DECODE_ERRORS: {result['decode_errors']}")
    print(f"TRAIN_VAL_OVERLAP: {result['train_val_overlap']}")
    print(f"SELECTED_DATA_ROOT: {result['selected_data_root']}")
    print(f"STAGING_AUDIT: {result['staging_audit']}")
    print(f"RESUME_READY: {'PASS' if result['resume_ready'] else 'FAIL'}")


if __name__ == "__main__":
    main()
