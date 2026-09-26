from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from common import DATASET_ROOT, ROOT, read_csv


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform() -> Callable:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform() -> Callable:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class IndexedImageDataset(Dataset):
    def __init__(self, rows: list[dict], transform: Callable, label_map: dict[int, int] | None = None):
        self.rows = rows
        self.transform = transform
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = DATASET_ROOT / row["file_name"]
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        category_id = int(row["category_id"])
        label = self.label_map[category_id] if self.label_map is not None else category_id
        return tensor, int(label), int(row["image_id"]), str(row["file_name"])


def indexed_rows(split: str) -> list[dict[str, str]]:
    names = {
        "full_train": "train_full_native_selected.csv",
        "mini_train": "train_mini_selected.csv",
        "val": "val_selected.csv",
    }
    if split not in names:
        raise ValueError(f"Unknown indexed split: {split}")
    name = names[split]
    return read_csv(ROOT / "data_indices" / name)


def training_dataset(model: str) -> tuple[IndexedImageDataset, dict[int, int]]:
    rotation = read_csv(ROOT / f"rotation_{model}.csv")
    label_map = {int(r["category_id"]): int(r["local_training_label"]) for r in rotation}
    rows = [r for r in indexed_rows("full_train") if int(r["category_id"]) in label_map]
    rows.sort(key=lambda r: (int(r["category_id"]), int(r["image_id"])))
    expected = {"M1": 22289, "M2": 22142, "M3": 22035, "M4": 22058}[model]
    if len(rows) != expected:
        raise AssertionError(f"{model}: expected {expected} training rows, got {len(rows)}")
    return IndexedImageDataset(rows, train_transform(), label_map), label_map


def upstream_train_eval_dataset(model: str) -> IndexedImageDataset:
    rotation = read_csv(ROOT / f"rotation_{model}.csv")
    label_map = {int(r["category_id"]): int(r["local_training_label"]) for r in rotation}
    rows = [r for r in indexed_rows("full_train") if int(r["category_id"]) in label_map]
    rows.sort(key=lambda r: (int(r["category_id"]), int(r["image_id"])))
    expected = {"M1": 22289, "M2": 22142, "M3": 22035, "M4": 22058}[model]
    if len(rows) != expected:
        raise AssertionError(f"{model}: expected {expected} deterministic train rows, got {len(rows)}")
    return IndexedImageDataset(rows, eval_transform(), label_map)


def restricted_validation_dataset(model: str) -> IndexedImageDataset:
    rotation = read_csv(ROOT / f"rotation_{model}.csv")
    label_map = {int(r["category_id"]): int(r["local_training_label"]) for r in rotation}
    rows = [r for r in indexed_rows("val") if int(r["category_id"]) in label_map]
    rows.sort(key=lambda r: (int(r["category_id"]), int(r["image_id"])))
    if len(rows) != 800:
        raise AssertionError(f"{model}: expected 800 restricted validation rows, got {len(rows)}")
    return IndexedImageDataset(rows, eval_transform(), label_map)


def downstream_train_dataset() -> IndexedImageDataset:
    manifest = read_csv(ROOT / "frozen_manifest.csv")
    d_ids = {int(r["category_id"]) for r in manifest if r["role"] == "d"}
    rows = [r for r in indexed_rows("mini_train") if int(r["category_id"]) in d_ids]
    rows.sort(key=lambda r: (int(r["category_id"]), int(r["image_id"])))
    if len(rows) != 1000:
        raise AssertionError(f"Expected 1000 downstream-ID train rows, got {len(rows)}")
    return IndexedImageDataset(rows, eval_transform(), None)


def all_validation_dataset() -> IndexedImageDataset:
    rows = indexed_rows("val")
    rows.sort(key=lambda r: (int(r["category_id"]), int(r["image_id"])))
    if len(rows) != 1000:
        raise AssertionError(f"Expected 1000 selected validation rows, got {len(rows)}")
    return IndexedImageDataset(rows, eval_transform(), None)
