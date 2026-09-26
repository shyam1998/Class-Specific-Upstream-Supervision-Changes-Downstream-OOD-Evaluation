from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
# Required by PyTorch for deterministic CUDA GEMM on CUDA >= 10.2. This must
# be set before importing torch and does not alter the scientific recipe.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
DATASET_ROOT = Path(os.environ.get("INAT_ROOT", REPO / "data/inaturalist"))
SOURCE_MANIFEST = ROOT / "frozen_manifest.csv"
SOURCE_INVENTORY = ROOT / "data_indices/train_full_native_selected.csv"
SOURCE_MEMBER_LIST = SOURCE_INVENTORY.with_name("full_native_selected_members.txt")
MINI_REFERENCE = ROOT
RECIPE_PATH = ROOT / "resolved_training_recipe.yaml"
PYTHON_EXE = Path(sys.executable)

MODELS = ("M1", "M2", "M3", "M4")
SEEDS = (0, 1)
ROLES = ("d", "c1", "c2", "c3", "c4")
OOD_ROLES = ("c1", "c2", "c3", "c4")
WITHHELD = {"c1": "M1", "c2": "M2", "c3": "M3", "c4": "M4"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_yaml(path: Path = RECIPE_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"Cannot infer CSV schema for empty rows: {path}")
        fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(tensor.dtype).encode("ascii"))
        h.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        h.update(tensor.numpy().tobytes(order="C"))
    return h.hexdigest()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def environment_record() -> dict[str, Any]:
    import PIL
    import pandas
    import scipy
    import sklearn
    import torchvision

    gpu = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {
            "name": props.name,
            "total_memory_bytes": props.total_memory,
            "compute_capability": [props.major, props.minor],
        }
    return {
        "created_utc": utc_now(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "PIL": PIL.__version__,
        "yaml": yaml.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
        "git_commit": git_commit(),
    }


def manifest_rows() -> list[dict[str, str]]:
    return read_csv(ROOT / "frozen_manifest.csv")


def validate_manifest(rows: list[dict[str, str]]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    groups = sorted({r["group_id"] for r in rows})
    category_ids = [int(r["category_id"]) for r in rows]
    checks["exactly_20_groups"] = len(groups) == 20
    checks["exactly_100_rows"] = len(rows) == 100
    checks["exactly_100_distinct_species"] = len(set(category_ids)) == 100
    checks["roles_per_group"] = all(
        sorted(r["role"] for r in rows if r["group_id"] == g) == sorted(ROLES) for g in groups
    )
    checks["frozen_mini_and_validation_counts"] = all(
        int(r["mini_train_images"]) == 50 and int(r["validation_images"]) == 10 for r in rows
    )
    checks["withheld_mapping_frozen"] = all(
        (r["role"] == "d" and r["withheld_model"] == "NONE")
        or (r["role"] in OOD_ROLES and r["withheld_model"] == WITHHELD[r["role"]])
        for r in rows
    )
    full_counts = {
        int(r["category_id"]): int(r["full_train_images"])
        for r in read_csv(ROOT / "native_full_image_inventory.csv")
    }
    checks["full_counts_cover_manifest"] = set(full_counts) == set(category_ids)
    rotation_counts: dict[str, Any] = {}
    for model in MODELS:
        present = [r for r in rows if r["withheld_model"] != model]
        d_rows = [r for r in present if r["role"] == "d"]
        rotation_counts[model] = {
            "species": len(present),
            "d_species": len(d_rows),
            "train_images": sum(full_counts[int(r["category_id"])] for r in present),
        }
    expected_images = {"M1": 22289, "M2": 22142, "M3": 22035, "M4": 22058}
    checks["rotations_exactly_80_species_native_totals"] = all(
        rotation_counts[m] == {"species": 80, "d_species": 20, "train_images": expected_images[m]}
        for m in MODELS
    )
    checks["every_ood_withheld_once_present_three"] = all(
        sum(r["withheld_model"] == m for m in MODELS) == 1
        and sum(r["withheld_model"] != m for m in MODELS) == 3
        for r in rows if r["role"] in OOD_ROLES
    )
    checks["d_present_all_rotations"] = all(r["withheld_model"] == "NONE" for r in rows if r["role"] == "d")
    if not all(bool(v) for k, v in checks.items() if k != "rotation_counts"):
        raise AssertionError(f"Frozen manifest invariant failed: {checks}")
    return {"checks": checks, "rotation_counts": rotation_counts, "status": "PASS"}


def rotation_rows(rows: list[dict[str, str]], model: str) -> list[dict[str, Any]]:
    selected = [dict(r) for r in rows if r["withheld_model"] != model]
    selected.sort(key=lambda r: (r["group_id"], ROLES.index(r["role"])))
    for local_label, row in enumerate(selected):
        row["rotation"] = model
        row["local_training_label"] = local_label
        row["present_upstream"] = True
    return selected


def lr_at_step(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
