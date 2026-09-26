from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .common import ROOT, SELECTED_DATA, ROTATION_TOTALS, read_csv

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def simclr_transform():
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(), T.Normalize(MEAN, STD),
    ])


def eval_transform():
    return T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])


class TwoView:
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, image):
        return self.transform(image), self.transform(image)


class IndexedDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(SELECTED_DATA / row["file_name"]) as image:
            value = self.transform(image.convert("RGB"))
        return value, int(row["category_id"]), int(row["image_id"]), row["file_name"]


def upstream_rows(rotation: str):
    manifest = read_csv(ROOT / "manifest.csv")
    present = {int(row["category_id"]) for row in manifest if row["withheld_model"] != rotation}
    rows = [row for row in read_csv(ROOT / "train_image_selection.csv") if int(row["category_id"]) in present]
    rows.sort(key=lambda row: (int(row["category_id"]), int(row["image_id"])))
    if len(rows) != ROTATION_TOTALS[rotation]:
        raise RuntimeError(f"{rotation} pool has {len(rows)}, expected {ROTATION_TOTALS[rotation]}")
    return rows


def upstream_dataset(rotation: str):
    return IndexedDataset(upstream_rows(rotation), TwoView(simclr_transform()))


def downstream_reference_dataset():
    rows = read_csv(ROOT / "downstream_reference_selection.csv")
    rows.sort(key=lambda row: (int(row["category_id"]), int(row["image_id"])))
    if len(rows) != 1000:
        raise RuntimeError("Downstream reference bank is not 1,000 images")
    return IndexedDataset(rows, eval_transform())


def downstream_eval_dataset():
    rows = read_csv(ROOT / "downstream_eval_selection.csv")
    rows.sort(key=lambda row: (int(row["category_id"]), int(row["image_id"])))
    if len(rows) != 1000:
        raise RuntimeError("Downstream evaluation set is not 1,000 images")
    return IndexedDataset(rows, eval_transform())
