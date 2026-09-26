from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
PROJECT = REPO
CANONICAL = REPO / "experiments/supervised/inaturalist"
REFERENCE = REPO / "experiments/simclr/imagenet"
SELECTED_DATA = Path(os.environ.get("INAT_ROOT", REPO / "data/inaturalist"))
MODELS = ("M1", "M2", "M3", "M4")
SEEDS = (0, 1)
ROLES = ("d", "c1", "c2", "c3", "c4")
WITHHELD = {"c1": "M1", "c2": "M2", "c3": "M3", "c4": "M4"}
ROTATION_TOTALS = {"M1": 22289, "M2": 22142, "M3": 22035, "M4": 22058}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields = fieldnames or list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def path_list_sha256(rows: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for value in sorted(row["file_name"] for row in rows):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def state_dict_sha256(state_dict) -> str:
    """Exact hash convention imported from the ImageNet SimCLR reference."""
    digest = hashlib.sha256()
    for name, tensor in state_dict.items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def full_model_sha256(encoder_state, projector_state) -> str:
    merged = {**{f"encoder.{k}": v for k, v in encoder_state.items()},
              **{f"projector.{k}": v for k, v in projector_state.items()}}
    return state_dict_sha256(merged)


def set_seed(seed: int) -> None:
    import torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_torch_save(value: Any, path: Path) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    os.replace(temporary, path)


def torch_load(path: Path) -> Any:
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def environment_record() -> dict[str, Any]:
    import PIL
    import torch
    import torchvision
    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {"name": properties.name, "total_memory_bytes": properties.total_memory,
               "compute_capability": [properties.major, properties.minor]}
    return {"recorded_utc": now(), "python": sys.version, "torch": torch.__version__,
            "torchvision": torchvision.__version__, "compiled_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(), "gpu": gpu, "PIL": PIL.__version__,
            "persistent_mount": "data", "git_commit": git_commit(),
            "git_backed": git_commit() is not None}


def manifest_rows() -> list[dict[str, str]]:
    return read_csv(ROOT / "manifest.csv")


def protected_hashes() -> dict[str, str]:
    paths = [
        CANONICAL / "frozen_manifest.csv", CANONICAL / "manifest.json",
        CANONICAL / "data_indices/train_full_native_selected.csv",
        CANONICAL / "data_indices/train_mini_selected.csv",
        CANONICAL / "data_indices/val_selected.csv",
        CANONICAL / "per_state_aurocs.csv", CANONICAL / "per_species_effects_by_seed.csv",
        CANONICAL / "per_species_effects.csv", CANONICAL / "per_group_effects.csv",
        CANONICAL / "bootstrap_primary.csv", CANONICAL / "bootstrap_detector_gap.csv",
        CANONICAL / "final_report.md", CANONICAL / "verdict.txt",
    ]
    paths.extend(sorted(path for path in REFERENCE.rglob("*") if path.is_file()))
    return {str(path): sha256_file(path) for path in paths}


def assert_protected_unchanged() -> None:
    required = [ROOT / "manifest.csv", ROOT / "train_image_selection.csv",
                ROOT / "downstream_reference_selection.csv", ROOT / "downstream_eval_selection.csv"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen protocol files: {missing}")
