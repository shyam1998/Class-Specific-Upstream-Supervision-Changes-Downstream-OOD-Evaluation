# Upstream class provenance and OOD detection

Anonymous reproduction code for the ICLR 2027 submission.

## Experiments

The fixed grouped rotating leave-one-out design uses 20 semantic groups, one downstream-ID class and four candidate future-OOD classes per group, four rotations (`M1`–`M4`), and seeds 0 and 1. Implementations are organized as follows:

- `experiments/supervised/`: supervised ResNet baselines for CIFAR-100, controlled ImageNet, and iNaturalist 2021 FULL-native.
- `experiments/simclr/`: SimCLR ResNet experiments for CIFAR-100, controlled ImageNet, and iNaturalist.
- `experiments/vit/`: from-scratch ViT architecture controls for all three datasets. CIFAR uses the stronger-regularized ViT recipe.
- `analysis/detectors/`: the common nine-detector implementation and paired class/group aggregation.
- `analysis/geometry/`: detector-independent centroid geometry analysis.
- `manifests/`: the frozen semantic groups and rotation assignments.

Raw datasets are not redistributed. Dataset licenses and download instructions are provided by CIFAR-100, ImageNet, and iNaturalist.

## Setup

```bash
conda env create -f environment.yml
conda activate provenance-ood
export PYTHONPATH="$PWD"
export CIFAR100_ROOT=/path/to/cifar100
export IMAGENET_ROOT=/path/to/ilsvrc2012
export INAT_ROOT=/path/to/inaturalist-selected-data
```

`IMAGENET_ROOT` must contain `train/` and `val/` ImageFolder trees. `INAT_ROOT` must contain the relative image paths listed in the frozen selection CSVs. All model constructors set `weights=None`; no external pretrained weights are used.

## Reproduction

Run commands from the repository root. Each launcher supports stage-level execution; use `--help` or the positional stage choices shown below.

```bash
# Supervised CIFAR-100 ResNet-18
python -m experiments.supervised.cifar100.src.run_experiment --help

# Supervised controlled-ImageNet ResNet-50
python -m experiments.supervised.imagenet.src.run_rotation4 --stage full

# Supervised iNaturalist ResNet-50
(cd experiments/supervised/inaturalist && python src/run_experiment.py all)

# CIFAR-100, controlled-ImageNet, and iNaturalist SimCLR
python -m experiments.simclr.cifar100.run all
python -m experiments.simclr.imagenet.src.run_simclr_rotation4 --stage full
python -m experiments.simclr.inaturalist.src.run_experiment all

# ViT controls
# Each ViT run uses the same staged interface:
python -m experiments.vit.cifar100.v2.run preflight
for seed in 0 1; do for model in M1 M2 M3 M4; do
  python -m experiments.vit.cifar100.v2.run train --model "$model" --seed "$seed"
done; done
python -m experiments.vit.cifar100.v2.run classification-audit
for seed in 0 1; do for model in M1 M2 M3 M4; do
  python -m experiments.vit.cifar100.v2.run extract --model "$model" --seed "$seed"
  python -m experiments.vit.cifar100.v2.run evaluate --model "$model" --seed "$seed"
done; done
python -m experiments.vit.cifar100.v2.run aggregate

# Replace the module above with experiments.vit.imagenet.run_vit or
# experiments.vit.inaturalist.run_vit for the corresponding dataset.
```

The classification-quality gates in the ViT launchers run before focal-OOD evaluation. The detector suite reuses each trained encoder and downstream probe across kNN, Energy, MSP, Mahalanobis, ViM, NECO, NCI, GradOrth, and ODIN. Detector fitting uses downstream-ID data only.

For detector replay after feature/probe generation:

```bash
python -m analysis.detectors.src.run_all_features
python -m analysis.detectors.src.run_all_odin
python -m analysis.detectors.src.verify_coverage
python -m analysis.detectors.src.analyze
```

Set `CIFAR_SUPERVISED_ROOT`, `IMAGENET_SUPERVISED_ROOT`, and `INAT_SUPERVISED_ROOT` if artifacts are outside their experiment directories. Set `PROVENANCE_OUTPUT_ROOT` to choose the analysis output directory.

## Validation

```bash
python -m compileall -q .
pytest -q analysis/detectors/tests
```
