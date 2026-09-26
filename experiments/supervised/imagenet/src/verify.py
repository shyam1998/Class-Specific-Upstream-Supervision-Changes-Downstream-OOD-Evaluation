from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .data import sha256_file


def read_hash_record(path: Path) -> str:
    parts = Path(path).read_text().strip().split()
    if not parts or len(parts[0]) != 64:
        raise ValueError(f"Invalid SHA256 record: {path}")
    return parts[0]


def verify_preregistration(path: Path, hash_path: Path) -> dict:
    expected = read_hash_record(hash_path)
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"Preregistration hash mismatch: expected {expected}, got {actual}")
    payload = json.loads(path.read_text())
    # The digest is supplied at runtime: embedding a file's own SHA256 in that
    # file is not well-defined.
    payload["preregistration_sha256"] = actual
    return payload


def verify_manifest_hash(manifest_path: Path, expected: str) -> None:
    actual = sha256_file(manifest_path)
    if actual != expected:
        raise RuntimeError(f"Manifest hash mismatch: expected {expected}, got {actual}")


def verify_design(payload: dict) -> None:
    groups = payload.get("groups", [])
    classes = [item for group in groups for item in group.get("classes", [])]
    if len(groups) != 20 or len(classes) != 100:
        raise ValueError("Expected 20 groups and 100 classes")


def artifact_metadata(kind: str, condition: str, seed: int, prereg_hash: str,
                      manifest_hash: str, **extra) -> dict:
    return {"kind": kind, "condition": condition, "seed": seed,
            "preregistration_sha256": prereg_hash, "manifest_sha256": manifest_hash, **extra}


def safe_torch_load(path: Path, expected: dict, map_location="cpu"):
    import torch
    payload = torch.load(path, map_location=map_location, weights_only=False)
    metadata = payload.get("metadata", {})
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"Unsafe artifact {path}: {key}={metadata.get(key)!r}, expected {value!r}")
    return payload


def atomic_torch_save(payload, path: Path) -> None:
    import os
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)
