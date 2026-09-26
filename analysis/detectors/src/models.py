from __future__ import annotations

from pathlib import Path
import os
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torchvision.models import resnet18, resnet50

from .common import RunData, canonical_state_hash


CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR_STD = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class EncoderProbe(nn.Module):
    def __init__(self, encoder: nn.Module, probe: nn.Linear):
        super().__init__()
        self.encoder = encoder
        self.probe = probe

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.probe(self.features(x))


class IndexedCIFAR(Dataset):
    def __init__(self, indices: list[int], transform: Callable):
        self.dataset = CIFAR100(
            os.environ.get("CIFAR100_ROOT", "data/cifar100"),
            train=False, transform=transform, download=False,
        )
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, label = self.dataset[self.indices[index]]
        return image, int(label), index


class IndexedPaths(Dataset):
    def __init__(self, root: Path, paths: list[str], transform: Callable):
        self.root = root
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.root / self.paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, index


def cifar_eval_transform() -> Callable:
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])


def r50_eval_transform() -> Callable:
    return transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _cifar_encoder() -> nn.Module:
    network = resnet18(weights=None, num_classes=80)
    network.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    network.maxpool = nn.Identity()
    return nn.Sequential(*list(network.children())[:-1])


def _r50_encoder_from_full(network: nn.Module) -> nn.Module:
    return nn.Sequential(*list(network.children())[:-1])


def load_encoder_probe(data: RunData) -> tuple[EncoderProbe, dict]:
    checkpoint = torch.load(data.checkpoint_path, map_location="cpu", weights_only=False)
    if data.slug == "cifar100":
        if checkpoint.get("identity") != [data.model, data.seed] or int(checkpoint.get("feature_dim", -1)) != 512:
            raise RuntimeError(f"CIFAR checkpoint identity mismatch: {data.checkpoint_path}")
        encoder = _cifar_encoder()
        encoder.load_state_dict(checkpoint["encoder"], strict=True)
        checkpoint_state_hash = canonical_state_hash(checkpoint["encoder"])
    else:
        network = resnet50(weights=None)
        network.fc = nn.Linear(2048, 80, bias=True)
        network.load_state_dict(checkpoint["model_state"], strict=True)
        if data.slug == "imagenet":
            metadata = checkpoint["metadata"]
            valid = metadata.get("rotation_model") == data.model and int(metadata.get("seed", -1)) == data.seed and int(metadata.get("epoch", -1)) == 100
        else:
            valid = checkpoint.get("rotation") == data.model and int(checkpoint.get("seed", -1)) == data.seed and int(checkpoint.get("epoch", -1)) == 100
        if not valid:
            raise RuntimeError(f"R50 checkpoint identity mismatch: {data.checkpoint_path}")
        encoder = _r50_encoder_from_full(network)
        checkpoint_state_hash = canonical_state_hash(checkpoint["model_state"])

    probe = nn.Linear(data.feature_dim, 20, bias=True)
    probe_state = {
        "weight": torch.from_numpy(data.probe_weight.copy()),
        "bias": torch.from_numpy(data.probe_bias.copy()),
    }
    probe.load_state_dict(probe_state, strict=True)
    model = EncoderProbe(encoder, probe).eval()
    model.requires_grad_(False)
    metadata = {
        "checkpoint_state_sha256": checkpoint_state_hash,
        "encoder_state_sha256": canonical_state_hash(model.encoder.state_dict()),
        "probe_state_sha256": canonical_state_hash(model.probe.state_dict()),
        "strict_checkpoint_load": True,
        "strict_probe_load": True,
    }
    return model, metadata


def raw_evaluation_dataset(data: RunData) -> Dataset:
    if data.slug == "cifar100":
        indices = [int(item.split(":", 1)[1]) for item in data.eval_ids.tolist()]
        dataset = IndexedCIFAR(indices, cifar_eval_transform())
        observed_labels = np.asarray([dataset.dataset.targets[index] for index in indices]).astype(str)
        if not np.array_equal(observed_labels, data.eval_class_ids):
            raise RuntimeError("CIFAR raw evaluation labels differ from canonical feature identities")
        return dataset
    roots = {
        "imagenet": Path(os.environ.get("IMAGENET_ROOT", "data/imagenet")),
        "inat": Path(os.environ.get("INAT_ROOT", "data/inaturalist")),
    }
    dataset = IndexedPaths(roots[data.slug], data.eval_ids.tolist(), r50_eval_transform())
    missing = [item for item in data.eval_ids.tolist() if not (roots[data.slug] / item).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} raw evaluation images; first={missing[0]}")
    return dataset


def normalization_std(slug: str) -> tuple[float, float, float]:
    return CIFAR_STD if slug == "cifar100" else IMAGENET_STD
