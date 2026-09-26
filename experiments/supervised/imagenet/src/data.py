from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPEG"}
WNID_TAR_RE = re.compile(r"^(n\d{8})\.tar$")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_dataset_root(configured: str | None = None) -> Path | None:
    """Check only the explicitly permitted, bounded ImageNet locations."""
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    if os.environ.get("IMAGENET_ROOT"):
        candidates.append(Path(os.environ["IMAGENET_ROOT"]).expanduser())
    home = Path.home()
    candidates.extend([
        home / "imagenet",
        home / "data" / "imagenet",
        home / "datasets" / "imagenet",
        Path("/data/imagenet"),
        Path("/scratch") / os.environ.get("USER", "") / "imagenet",
    ])
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in seen and candidate.exists():
            return candidate
        seen.add(candidate)
    return None


def _image_files(path: Path) -> Iterable[Path]:
    for item in path.iterdir():
        if item.is_file() and item.suffix.lower() in {x.lower() for x in IMAGE_EXTENSIONS}:
            yield item


def inspect_imagenet(root: Path, count_images: bool = True, val_directory: str = "val") -> dict:
    root = root.resolve()
    train = root / "train"
    val = root / val_directory
    if not train.is_dir() or not val.is_dir():
        raise ValueError(f"ImageNet root must contain train/ and val/: {root}")
    train_classes = sorted(p.name for p in train.iterdir() if p.is_dir() and p.name.startswith("n"))
    val_classes = sorted(p.name for p in val.iterdir() if p.is_dir() and p.name.startswith("n"))
    flat_val = not val_classes and any(_image_files(val))
    result = {
        "root": str(root),
        "val_directory": val_directory,
        "train_classes": len(train_classes),
        "val_classes": len(val_classes),
        "flat_val": flat_val,
        "train_wnids": train_classes,
        "val_wnids": val_classes,
    }
    if count_images:
        result["train_images"] = sum(1 for w in train_classes for _ in _image_files(train / w))
        result["val_images"] = (sum(1 for w in val_classes for _ in _image_files(val / w))
                                if val_classes else sum(1 for _ in _image_files(val)))
    return result


def require_imagenet(root: Path, val_directory: str = "val") -> dict:
    info = inspect_imagenet(root, val_directory=val_directory)
    errors = []
    if info["train_classes"] != 1000:
        errors.append(f"train has {info['train_classes']} WNID directories, expected 1000")
    if not info["flat_val"] and info["val_classes"] != 1000:
        errors.append(f"val has {info['val_classes']} WNID directories, expected 1000")
    if info.get("train_images") != 1_281_167:
        errors.append(f"train has {info.get('train_images')} images, expected 1281167")
    if info.get("val_images") != 50_000:
        errors.append(f"val has {info.get('val_images')} images, expected 50000")
    if errors:
        raise ValueError("Invalid/incomplete ILSVRC2012 dataset: " + "; ".join(errors))
    return info


def devkit_index_to_wnid(metadata: Path) -> dict[int, str]:
    metadata = Path(metadata)
    if metadata.suffix.lower() == ".json":
        mapping = {int(k): v for k, v in json.loads(metadata.read_text()).items()}
    else:
        from scipy.io import loadmat
        raw = loadmat(metadata, squeeze_me=True, struct_as_record=False)["synsets"]
        mapping = {int(item.ILSVRC2012_ID): str(item.WNID) for item in raw
                   if int(item.num_children) == 0 and str(item.WNID).startswith("n")}
    if len(mapping) != 1000:
        raise ValueError(f"Devkit metadata yielded {len(mapping)} leaf mappings, expected 1000")
    return mapping


def prepare_flat_validation(
    root: Path, ground_truth: Path, metadata: Path, output: Path | None = None,
    selected_wnids: set[str] | None = None, flat_directory: str = "val",
) -> Path:
    """Create a non-destructive symlink tree for official flat ILSVRC validation.

    metadata may be the official devkit meta.mat, or a JSON mapping of integer
    ILSVRC class indices (1..1000) to WNIDs. The ground-truth file is the
    official 50,000-line sequence of those indices.
    Existing source files are never moved or renamed.
    """
    root, ground_truth, metadata = map(Path, (root, ground_truth, metadata))
    src = root / flat_directory
    output = output or (root / "val_imagefolder")
    index_to_wnid = devkit_index_to_wnid(metadata)
    labels = [int(x.strip()) for x in ground_truth.read_text().splitlines() if x.strip()]
    images = sorted(_image_files(src))
    if len(images) != 50_000 or len(labels) != 50_000:
        raise ValueError(f"Expected 50,000 images and labels; got {len(images)} and {len(labels)}")
    if set(labels) - set(index_to_wnid):
        raise ValueError("Ground truth contains class indices absent from metadata mapping")
    selected_wnids = selected_wnids or set(index_to_wnid.values())
    if not selected_wnids <= set(index_to_wnid.values()):
        raise ValueError("Selected WNID is absent from official devkit leaf mapping")
    output.mkdir(parents=True, exist_ok=True)
    for image, label in zip(images, labels, strict=True):
        wnid = index_to_wnid[label]
        if wnid not in selected_wnids: continue
        target_dir = output / wnid
        target_dir.mkdir(exist_ok=True)
        link = target_dir / image.name
        if link.exists() or link.is_symlink():
            if link.resolve() != image.resolve():
                raise FileExistsError(f"Conflicting link: {link}")
        else:
            link.symlink_to(image.resolve())
    return output


def verify_validation_mapping(flat_dir: Path, val_tree: Path, ground_truth: Path,
                              metadata: Path, selected_wnids: set[str]) -> dict[str, int]:
    index_to_wnid = devkit_index_to_wnid(metadata)
    labels = [int(x) for x in Path(ground_truth).read_text().splitlines() if x.strip()]
    images = sorted(_image_files(Path(flat_dir)))
    if len(images) != len(labels) or len(images) != 50_000:
        raise ValueError("Official validation image/ground-truth sequence is not 50,000 entries")
    expected = {(index_to_wnid[label], image.name): image.resolve()
                for image, label in zip(images, labels, strict=True)
                if index_to_wnid[label] in selected_wnids}
    actual = {}
    for wnid in selected_wnids:
        directory = Path(val_tree) / wnid
        if not directory.is_dir(): raise ValueError(f"Missing selected validation directory: {wnid}")
        for link in directory.iterdir():
            if not link.is_symlink(): raise ValueError(f"Validation entry is not a symlink: {link}")
            actual[(wnid, link.name)] = link.resolve()
    if actual != expected: raise ValueError("Selected validation symlinks disagree with official devkit mapping")
    return {w: sum(1 for key in actual if key[0] == w) for w in sorted(selected_wnids)}


def transforms(train: bool):
    from torchvision import transforms as T
    normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(), normalize])
    return T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_archive_wnids(archive: Path) -> tuple[list[str], dict[str, tarfile.TarInfo]]:
    """Read and strictly validate the 1,000 class-tar members without extraction."""
    archive = Path(archive)
    if not archive.is_file(): raise FileNotFoundError(archive)
    members: dict[str, tarfile.TarInfo] = {}
    with tarfile.open(archive, "r:") as outer:
        for item in outer.getmembers():
            name = Path(item.name).name
            match = WNID_TAR_RE.fullmatch(name)
            if not match:
                raise ValueError(f"Unexpected outer training archive member: {item.name}")
            if not item.isfile() or item.name != name:
                raise ValueError(f"Unsafe/non-file outer member: {item.name}")
            wnid = match.group(1)
            if wnid in members: raise ValueError(f"Duplicate WNID tar member: {wnid}")
            members[wnid] = item
    if len(members) != 1000:
        raise ValueError(f"Training archive has {len(members)} WNID tar members, expected exactly 1000")
    return sorted(members), members


def selective_extract_train(archive: Path, selected_wnids: set[str], output_root: Path) -> dict[str, int]:
    """Extract only frozen class tarballs, with path traversal and partial-output guards."""
    if len(selected_wnids) != 100: raise ValueError("Selective extraction requires exactly 100 WNIDs")
    wnids, members = training_archive_wnids(archive)
    missing = selected_wnids - set(wnids)
    if missing: raise ValueError(f"Selected WNIDs absent from train archive: {sorted(missing)}")
    output_root = Path(output_root); output_root.mkdir(parents=True, exist_ok=True)
    counts = {}
    with tarfile.open(archive, "r:") as outer:
        indexed = {Path(m.name).stem: m for m in outer.getmembers()}
        for wnid in sorted(selected_wnids):
            target = output_root / wnid
            if target.is_dir() and not (output_root / f".{wnid}.partial").exists():
                existing = sum(1 for _ in _image_files(target))
                if existing == 0: raise RuntimeError(f"Existing selected class directory is empty: {target}")
                counts[wnid] = existing; continue
            partial = output_root / f".{wnid}.partial"
            if partial.exists(): raise RuntimeError(f"Incomplete prior extraction requires inspection: {partial}")
            partial.mkdir()
            inner_file = outer.extractfile(indexed[wnid])
            if inner_file is None: raise RuntimeError(f"Cannot read inner archive for {wnid}")
            count = 0
            with tarfile.open(fileobj=inner_file, mode="r|") as inner:
                for item in inner:
                    name = Path(item.name)
                    if not item.isfile() or name.name != item.name or name.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                        raise ValueError(f"Unsafe/unexpected member in {wnid}.tar: {item.name}")
                    source = inner.extractfile(item)
                    if source is None: raise RuntimeError(f"Cannot read {item.name}")
                    destination = partial / name.name
                    with destination.open("xb") as handle:
                        for chunk in iter(lambda: source.read(1024 * 1024), b""): handle.write(chunk)
                    count += 1
            if count == 0: raise RuntimeError(f"No images extracted for {wnid}")
            os.replace(partial, target); counts[wnid] = count
    return counts


def extract_flat_validation(archive: Path, output: Path) -> int:
    """Extract official validation JPEGs once into a guarded flat staging directory."""
    archive, output = Path(archive), Path(output)
    if output.is_dir():
        count = sum(1 for _ in _image_files(output))
        if count == 50_000: return count
        raise RuntimeError(f"Existing flat validation staging has {count}/50000 images: {output}")
    partial = output.with_name("." + output.name + ".partial")
    if partial.exists(): raise RuntimeError(f"Incomplete prior validation extraction: {partial}")
    partial.mkdir(parents=True)
    count = 0
    with tarfile.open(archive, "r:") as source:
        for item in source:
            name = Path(item.name)
            if not item.isfile() or name.name != item.name or name.suffix.lower() not in {".jpg", ".jpeg"}:
                raise ValueError(f"Unsafe/unexpected validation member: {item.name}")
            stream = source.extractfile(item)
            if stream is None: raise RuntimeError(f"Cannot read {item.name}")
            with (partial / name.name).open("xb") as handle:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""): handle.write(chunk)
            count += 1
    if count != 50_000: raise ValueError(f"Validation archive has {count} images, expected 50000")
    os.replace(partial, output); return count


def extract_devkit_metadata(archive: Path, output: Path) -> tuple[Path, Path]:
    """Extract only official meta.mat and validation ground truth."""
    archive, output = Path(archive), Path(output); output.mkdir(parents=True, exist_ok=True)
    wanted = {"meta.mat": output / "meta.mat",
              "ILSVRC2012_validation_ground_truth.txt": output / "ILSVRC2012_validation_ground_truth.txt"}
    with tarfile.open(archive, "r:gz") as source:
        found = set()
        for item in source:
            basename = Path(item.name).name
            if basename not in wanted: continue
            stream = source.extractfile(item)
            if stream is None: raise RuntimeError(f"Cannot read devkit member {item.name}")
            destination = wanted[basename]
            if not destination.exists():
                with destination.open("xb") as handle:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""): handle.write(chunk)
            found.add(basename)
    if found != set(wanted): raise ValueError(f"Devkit missing required metadata: {set(wanted)-found}")
    return wanted["meta.mat"], wanted["ILSVRC2012_validation_ground_truth.txt"]


def verify_selected_tree(root: Path, selected_wnids: set[str]) -> dict:
    root = Path(root); train = root / "train"; val = root / "val"
    if len(selected_wnids) != 100: raise ValueError("Expected 100 selected WNIDs")
    train_dirs = {p.name for p in train.iterdir() if p.is_dir()} if train.is_dir() else set()
    val_dirs = {p.name for p in val.iterdir() if p.is_dir()} if val.is_dir() else set()
    if not selected_wnids <= train_dirs: raise ValueError(f"Selected train WNID directories missing: {selected_wnids-train_dirs}")
    if not selected_wnids <= val_dirs: raise ValueError(f"Selected val WNID directories missing: {selected_wnids-val_dirs}")
    train_counts = {w: sum(1 for _ in _image_files(train / w)) for w in sorted(selected_wnids)}
    val_counts = {w: sum(1 for _ in _image_files(val / w)) for w in sorted(selected_wnids)}
    if any(v == 0 for v in train_counts.values()): raise ValueError("At least one selected train class is empty")
    if any(v != 50 for v in val_counts.values()): raise ValueError("Every selected validation class must have 50 images")
    return {"selected_classes": 100, "train_images": sum(train_counts.values()),
            "val_images": sum(val_counts.values()), "train_counts_by_wnid": train_counts,
            "val_counts_by_wnid": val_counts,
            "available_train_directories": len(train_dirs),
            "available_val_directories": len(val_dirs)}


class RemappedImageFolder:
    """Factory wrapper limiting ImageFolder to WNIDs and fixed output indices."""

    @staticmethod
    def build(root: Path, wnid_to_index: dict[str, int], transform=None):
        from torchvision.datasets import ImageFolder
        dataset = ImageFolder(root, transform=transform)
        samples = [(path, wnid_to_index[dataset.classes[target]])
                   for path, target in dataset.samples
                   if dataset.classes[target] in wnid_to_index]
        dataset.samples = samples
        dataset.imgs = samples
        dataset.targets = [target for _, target in samples]
        dataset.classes = [w for w, _ in sorted(wnid_to_index.items(), key=lambda x: x[1])]
        dataset.class_to_idx = dict(wnid_to_index)
        return dataset
