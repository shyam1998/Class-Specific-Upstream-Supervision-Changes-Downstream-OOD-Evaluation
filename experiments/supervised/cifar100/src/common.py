"""Exact confirmation2 CIFAR model, data views, and reproducibility helpers."""

from __future__ import annotations

import hashlib
import io
import json
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torchvision.models import resnet18


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


class CifarResNet18(nn.Module):
    def __init__(self, num_classes: int = 80):
        super().__init__()
        net = resnet18(weights=None, num_classes=num_classes)
        net.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        self.encoder = nn.Sequential(*list(net.children())[:-1])
        self.classifier = net.fc
        self.feature_dim = 512

    def features(self, x):
        return torch.flatten(self.encoder(x), 1)

    def forward(self, x):
        return self.classifier(self.features(x))


def encoder_state_dict(model: CifarResNet18) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.encoder.state_dict().items()}


def cloned_state_dict(model: nn.Module) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def supervised_transform():
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )


def eval_transform():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)]
    )


def raw_cifar(data_dir, train: bool, transform=None, download: bool = True):
    dataset = CIFAR100(str(Path(data_dir)), train=train, transform=transform, download=download)
    expected = {name: index for index, name in enumerate(dataset.classes)}
    if len(dataset.classes) != 100 or dataset.class_to_idx != expected:
        raise RuntimeError("Unexpected CIFAR-100 fine-class order")
    return dataset


class ClassSubset(Dataset):
    def __init__(self, dataset, classes, remap: bool = True):
        self.dataset = dataset
        self.classes = sorted(map(int, classes))
        selected = set(self.classes)
        self.indices = [index for index, target in enumerate(dataset.targets) if target in selected]
        self.mapping = {label: index for index, label in enumerate(self.classes)} if remap else None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.dataset[self.indices[index]]
        return image, self.mapping[label] if self.mapping is not None else label


def pretrain_dataset(data_dir, classes):
    return ClassSubset(raw_cifar(data_dir, True, supervised_transform()), classes, remap=True)


def eval_subset(data_dir, train: bool, classes, remap: bool = False):
    return ClassSubset(raw_cifar(data_dir, train, eval_transform()), classes, remap=remap)


def seed_everything(seed: int, deterministic: bool = True) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return torch.Generator().manual_seed(seed)


def seed_worker(_worker_id):
    value = torch.initial_seed() % (2**32)
    random.seed(value)
    np.random.seed(value)


def state_hash(state: dict) -> str:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_load(path: Path):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Required checkpoint missing: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def write_immutable(path: Path, content: str) -> None:
    path = Path(path)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"Frozen experiment file differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json_immutable(path: Path, value) -> None:
    write_immutable(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def upsert_csv(path: Path, rows: list[dict], keys: tuple[str, ...]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not old.empty:
        new_keys = set(map(tuple, new[list(keys)].astype(str).to_numpy()))
        keep = [tuple(row) not in new_keys for row in old[list(keys)].astype(str).to_numpy()]
        old = old.loc[keep]
    pd.concat([old, new], ignore_index=True).to_csv(path, index=False)


def git_commit(project_root: Path):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_record(project_root: Path) -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": git_commit(project_root),
    }
