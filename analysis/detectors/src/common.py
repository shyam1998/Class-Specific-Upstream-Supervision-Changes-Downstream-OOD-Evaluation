from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
PROJECT = Path(os.environ.get("PROVENANCE_OUTPUT_ROOT", REPO / "outputs"))
CIFAR_RUN_ROOT = Path(
    os.environ.get("CIFAR_SUPERVISED_ROOT", REPO / "experiments/supervised/cifar100")
)
IMAGENET_RUN_ROOT = Path(
    os.environ.get("IMAGENET_SUPERVISED_ROOT", REPO / "experiments/supervised/imagenet")
)
INAT_RUN_ROOT = Path(
    os.environ.get("INAT_SUPERVISED_ROOT", REPO / "experiments/supervised/inaturalist")
)

MODELS = ("M1", "M2", "M3", "M4")
SEEDS = (0, 1)
DETECTORS_NEW = ("mahalanobis", "vim", "neco", "nci", "gradorth", "odin")
DETECTORS_EXISTING = ("knn", "msp", "energy")
DETECTORS_ALL = DETECTORS_EXISTING + DETECTORS_NEW

DATASET_SLUGS = {
    "CIFAR-100": "cifar100",
    "controlled ImageNet": "imagenet",
    "iNaturalist 2021 FULL-native": "inat",
}
SLUG_DATASETS = {value: key for key, value in DATASET_SLUGS.items()}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows and not fieldnames:
        raise ValueError(f"Cannot infer columns for empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    columns = fieldnames or list(rows[0])
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def canonical_state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def save_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


@dataclass
class RunData:
    dataset: str
    slug: str
    model: str
    seed: int
    feature_dim: int
    reference_features: np.ndarray
    reference_class_ids: np.ndarray
    reference_ids: np.ndarray
    eval_features: np.ndarray
    eval_class_ids: np.ndarray
    eval_ids: np.ndarray
    d_classes: list[str]
    candidates: list[dict[str, str]]
    probe_weight: np.ndarray
    probe_bias: np.ndarray
    checkpoint_path: Path
    probe_path: Path
    feature_paths: list[Path]

    def validate(self) -> None:
        if self.model not in MODELS or self.seed not in SEEDS:
            raise ValueError((self.model, self.seed))
        if self.reference_features.shape != (len(self.reference_ids), self.feature_dim):
            raise RuntimeError(
                f"Reference shape mismatch: {self.dataset}/{self.model}/seed{self.seed}"
            )
        if self.eval_features.shape != (len(self.eval_ids), self.feature_dim):
            raise RuntimeError(
                f"Evaluation shape mismatch: {self.dataset}/{self.model}/seed{self.seed}"
            )
        if len(self.reference_class_ids) != len(self.reference_ids) or len(
            self.eval_class_ids
        ) != len(self.eval_ids):
            raise RuntimeError("Identity length mismatch")
        if len(set(self.reference_ids.tolist())) != len(self.reference_ids):
            raise RuntimeError("Duplicate reference identity")
        if len(set(self.eval_ids.tolist())) != len(self.eval_ids):
            raise RuntimeError("Duplicate evaluation identity")
        if self.probe_weight.shape != (20, self.feature_dim) or self.probe_bias.shape != (20,):
            raise RuntimeError(
                f"Probe shape mismatch: {self.probe_weight.shape}/{self.probe_bias.shape}"
            )
        if len(self.d_classes) != 20 or len(self.candidates) != 80:
            raise RuntimeError("Expected 20 downstream-ID and 80 future-OOD classes")
        if set(self.reference_class_ids.tolist()) != set(self.d_classes):
            raise RuntimeError("Reference classes do not equal frozen downstream-ID classes")
        if not set(self.d_classes).isdisjoint({item["class_id"] for item in self.candidates}):
            raise RuntimeError("ID/OOD class overlap")
        for item in self.candidates:
            if item["withheld_model"] not in MODELS:
                raise RuntimeError(f"Missing withheld model: {item}")
            if np.sum(self.eval_class_ids == item["class_id"]) == 0:
                raise RuntimeError(f"No evaluation images for candidate {item['class_id']}")
        if (
            not np.isfinite(self.reference_features).all()
            or not np.isfinite(self.eval_features).all()
        ):
            raise RuntimeError("Non-finite frozen features")


def _probe_numpy(state: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    weight = state["weight"].detach().cpu().numpy().astype(np.float32, copy=True)
    bias = state["bias"].detach().cpu().numpy().astype(np.float32, copy=True)
    return weight, bias


def _load_cifar(model: str, seed: int) -> RunData:
    root = CIFAR_RUN_ROOT
    lower = model.lower()
    feature_path = root / f"experiment/features/features_{lower}_seed{seed}.pt"
    probe_path = root / f"detector_audit/probes/linear_{lower}_seed{seed}.pt"
    checkpoint_path = root / f"experiment/checkpoints/encoder_{lower}_seed{seed}.pt"
    feature = torch.load(feature_path, map_location="cpu", weights_only=False)
    probe = torch.load(probe_path, map_location="cpu", weights_only=False)
    manifest = read_json(root / "experiment/manifest.json")
    candidates = []
    for group in manifest["groups"]:
        for role in ("c1", "c2", "c3", "c4"):
            item = group[role]
            candidates.append(
                {
                    "class_id": str(item["fine_id"]),
                    "class_name": item["fine_name"],
                    "group_id": str(group["coarse_id"]),
                    "group_name": group["coarse_name"],
                    "role": role,
                    "withheld_model": item["withheld_model"],
                }
            )
    weight, bias = _probe_numpy(probe["head"])
    value = RunData(
        dataset="CIFAR-100",
        slug="cifar100",
        model=model,
        seed=seed,
        feature_dim=512,
        reference_features=feature["train_id_features"].cpu().numpy(),
        reference_class_ids=np.asarray([str(x) for x in feature["train_id_labels"].tolist()]),
        reference_ids=np.asarray([f"train:{int(x)}" for x in feature["train_id_indices"].tolist()]),
        eval_features=feature["test_features"].cpu().numpy(),
        eval_class_ids=np.asarray([str(x) for x in feature["test_labels"].tolist()]),
        eval_ids=np.asarray([f"test:{int(x)}" for x in feature["test_indices"].tolist()]),
        d_classes=[str(x) for x in manifest["downstream_id_classes"]],
        candidates=candidates,
        probe_weight=weight,
        probe_bias=bias,
        checkpoint_path=checkpoint_path,
        probe_path=probe_path,
        feature_paths=[feature_path],
    )
    value.validate()
    return value


def _load_imagenet(model: str, seed: int) -> RunData:
    root = IMAGENET_RUN_ROOT
    tag = f"{model}_seed{seed}"
    train_path = next((root / "features/rotation4_v1").glob(f"{tag}_train_*.pt"))
    val_path = next((root / "features/rotation4_v1").glob(f"{tag}_val_*.pt"))
    probe_path = next((root / "checkpoints/rotation4_v1").glob(f"probe_{tag}_*.pt"))
    checkpoint_path = root / f"checkpoints/rotation4_v1/{tag}/epoch_100.pt"
    train = torch.load(train_path, map_location="cpu", weights_only=False)
    val = torch.load(val_path, map_location="cpu", weights_only=False)
    probe = torch.load(probe_path, map_location="cpu", weights_only=False)
    manifest = read_csv(root / "manifests/rotation4_v1/semantic_groups_rotation4_v1.csv")
    role_map = {"a": "c1", "b": "c2", "r1": "c3", "r2": "c4"}
    candidates = [
        {
            "class_id": row["wnid"],
            "class_name": row["class_name"],
            "group_id": row["group_id"],
            "group_name": row["semantic_parent"],
            "role": role_map[row["role"]],
            "withheld_model": row["withheld_model"],
        }
        for row in manifest
        if row["role"] != "d"
    ]
    d_classes = sorted(row["wnid"] for row in manifest if row["role"] == "d")
    reference_rows = sorted(
        read_csv(root / "detector_handoff/id_reference_images.csv"),
        key=lambda row: int(row["imagefolder_index"]),
    )
    eval_rows = sorted(
        read_csv(root / "detector_handoff/eval_images.csv"),
        key=lambda row: int(row["imagefolder_index"]),
    )
    train_order = train["metadata"]["wnid_order"]
    val_order = val["metadata"]["wnid_order"]
    reference_class_ids = np.asarray(train_order)[train["labels"].cpu().numpy()]
    eval_class_ids = np.asarray(val_order)[val["labels"].cpu().numpy()]
    if reference_class_ids.tolist() != [row["wnid"] for row in reference_rows]:
        raise RuntimeError(f"ImageNet reference identity mismatch: {tag}")
    if eval_class_ids.tolist() != [row["wnid"] for row in eval_rows]:
        raise RuntimeError(f"ImageNet evaluation identity mismatch: {tag}")
    weight, bias = _probe_numpy(probe["model_state"])
    value = RunData(
        dataset="controlled ImageNet",
        slug="imagenet",
        model=model,
        seed=seed,
        feature_dim=2048,
        reference_features=train["features"].cpu().numpy(),
        reference_class_ids=reference_class_ids,
        reference_ids=np.asarray([row["relative_path"] for row in reference_rows]),
        eval_features=val["features"].cpu().numpy(),
        eval_class_ids=eval_class_ids,
        eval_ids=np.asarray([row["relative_path"] for row in eval_rows]),
        d_classes=d_classes,
        candidates=candidates,
        probe_weight=weight,
        probe_bias=bias,
        checkpoint_path=checkpoint_path,
        probe_path=probe_path,
        feature_paths=[train_path, val_path],
    )
    value.validate()
    return value


def _load_inat(model: str, seed: int) -> RunData:
    root = INAT_RUN_ROOT
    feature_path = root / f"frozen_features/{model}_seed{seed}.npz"
    probe_path = root / f"probes/probe_{model}_seed{seed}.pt"
    checkpoint_path = root / f"checkpoints/{model}_seed{seed}/epoch_100.pt"
    with np.load(feature_path, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    probe = torch.load(probe_path, map_location="cpu", weights_only=False)
    manifest = read_csv(root / "frozen_manifest.csv")
    candidates = [
        {
            "class_id": row["category_id"],
            "class_name": row["scientific_name"],
            "group_id": row["group_id"],
            "group_name": row["parent_name"],
            "role": row["role"],
            "withheld_model": row["withheld_model"],
        }
        for row in manifest
        if row["role"] != "d"
    ]
    d_classes = [row["category_id"] for row in manifest if row["role"] == "d"]
    weight, bias = _probe_numpy(probe["model_state"])
    value = RunData(
        dataset="iNaturalist 2021 FULL-native",
        slug="inat",
        model=model,
        seed=seed,
        feature_dim=2048,
        reference_features=arrays["train_features"],
        reference_class_ids=np.asarray([str(x) for x in arrays["train_category_ids"]]),
        reference_ids=arrays["train_paths"].astype(str),
        eval_features=arrays["val_features"],
        eval_class_ids=np.asarray([str(x) for x in arrays["val_category_ids"]]),
        eval_ids=arrays["val_paths"].astype(str),
        d_classes=d_classes,
        candidates=candidates,
        probe_weight=weight,
        probe_bias=bias,
        checkpoint_path=checkpoint_path,
        probe_path=probe_path,
        feature_paths=[feature_path],
    )
    value.validate()
    return value


def load_run(slug: str, model: str, seed: int) -> RunData:
    functions = {"cifar100": _load_cifar, "imagenet": _load_imagenet, "inat": _load_inat}
    if slug not in functions:
        raise ValueError(slug)
    return functions[slug](model, seed)


def environment() -> dict[str, Any]:
    import pandas
    import scipy
    import sklearn
    import torchvision
    from PIL import __version__ as pillow_version

    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "pandas": pandas.__version__,
        "pillow": pillow_version,
        "compiled_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_bytes": (
            torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
        ),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
    }
