from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from .common import (
    DATASET_SLUGS,
    DETECTORS_ALL,
    DETECTORS_NEW,
    MODELS,
    PROJECT,
    ROOT,
    SEEDS,
    load_run,
    read_csv,
    read_json,
    sha256_file,
    write_csv,
    write_json,
)

DISPLAY = {
    "knn": "kNN",
    "msp": "MSP",
    "energy": "Energy",
    "mahalanobis": "Mahalanobis",
    "vim": "ViM",
    "neco": "NECO",
    "nci": "NCI",
    "gradorth": "GradOrth",
    "odin": "ODIN",
}
DISPLAY_ORDER = ("knn", "msp", "energy", "mahalanobis", "vim", "neco", "nci", "gradorth", "odin")


def auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    labels = np.r_[np.zeros(len(id_scores), dtype=np.int8), np.ones(len(ood_scores), dtype=np.int8)]
    return float(roc_auc_score(labels, np.r_[id_scores, ood_scores]))


def candidate_map(slug: str) -> dict[str, dict[str, str]]:
    return {item["class_id"]: item for item in load_run(slug, "M1", 0).candidates}


def baseline_rows(slug: str) -> list[dict[str, Any]]:
    candidates = candidate_map(slug)
    output = []
    if slug == "cifar100":
        path = (
            Path(os.environ.get("CIFAR_SUPERVISED_ROOT", PROJECT / "cifar100"))
            / "detector_audit/detector_model_class_raw.csv"
        )
        source = read_csv(path)
        normalized = [
            (str(row["fine_id"]), int(row["seed"]), row["model"], row["state"], row)
            for row in source
        ]
        columns = {"knn": "auroc_knn", "energy": "auroc_energy", "msp": "auroc_msp"}
    elif slug == "imagenet":
        path = (
            Path(os.environ.get("IMAGENET_SUPERVISED_ROOT", PROJECT / "imagenet"))
            / "artifacts/rotation4_v1/model_class_aurocs.csv"
        )
        source = read_csv(path)
        normalized = [
            (row["wnid"], int(row["seed"]), row["rotation_model"], row["treatment_state"], row)
            for row in source
        ]
        columns = {"knn": "auroc_knn", "energy": "auroc_energy", "msp": "auroc_msp"}
    elif slug == "inat":
        path = (
            Path(os.environ.get("INAT_SUPERVISED_ROOT", PROJECT / "inaturalist"))
            / "per_state_aurocs.csv"
        )
        source = read_csv(path)
        normalized = [
            (
                row["official_category_id"],
                int(row["seed"]),
                row["rotation"],
                row["provenance_state"],
                row,
            )
            for row in source
        ]
        columns = {"knn": "auroc_knn", "energy": "auroc_energy", "msp": "auroc_msp"}
    else:
        raise ValueError(slug)
    dataset = load_run(slug, "M1", 0).dataset
    for class_id, seed, rotation, state, row in normalized:
        if class_id not in candidates:
            continue
        item = candidates[class_id]
        for detector, column in columns.items():
            output.append(
                {
                    "dataset": dataset,
                    "dataset_slug": slug,
                    "detector": detector,
                    "group_id": item["group_id"],
                    "group_name": item["group_name"],
                    "class_id": class_id,
                    "class_name": item["class_name"],
                    "role": item["role"],
                    "withheld_model": item["withheld_model"],
                    "seed": seed,
                    "rotation": rotation,
                    "state": "withheld" if state in ("withheld", "absent") else "present",
                    "id_eval_images": "",
                    "ood_eval_images": "",
                    "auroc": float(row[column]),
                    "score_source": f"canonical stored per-state artifact: {path}",
                }
            )
    return output


def new_rows(slug: str) -> list[dict[str, Any]]:
    output = []
    for model in MODELS:
        for seed in SEEDS:
            data = load_run(slug, model, seed)
            tag = f"{model}_seed{seed}"
            feature_path = ROOT / "raw_scores" / slug / f"{tag}_feature_detectors.npz"
            odin_path = ROOT / "raw_scores" / slug / f"{tag}_odin.npz"
            with np.load(feature_path, allow_pickle=False) as source:
                scores = {
                    detector: source[detector].copy()
                    for detector in DETECTORS_NEW
                    if detector != "odin"
                }
                if not np.array_equal(source["evaluation_ids"], data.eval_ids):
                    raise RuntimeError(f"Feature-score identity mismatch: {slug}/{tag}")
            with np.load(odin_path, allow_pickle=False) as source:
                scores["odin"] = source["odin"].copy()
                if not np.array_equal(source["evaluation_ids"], data.eval_ids):
                    raise RuntimeError(f"ODIN-score identity mismatch: {slug}/{tag}")
            id_mask = np.isin(data.eval_class_ids, data.d_classes)
            for item in data.candidates:
                ood_mask = data.eval_class_ids == item["class_id"]
                if not id_mask.any() or not ood_mask.any():
                    raise RuntimeError(f"Empty ID/OOD comparison: {slug}/{tag}/{item['class_id']}")
                for detector in DETECTORS_NEW:
                    output.append(
                        {
                            "dataset": data.dataset,
                            "dataset_slug": slug,
                            "detector": detector,
                            "group_id": item["group_id"],
                            "group_name": item["group_name"],
                            "class_id": item["class_id"],
                            "class_name": item["class_name"],
                            "role": item["role"],
                            "withheld_model": item["withheld_model"],
                            "seed": seed,
                            "rotation": model,
                            "state": "withheld" if item["withheld_model"] == model else "present",
                            "id_eval_images": int(id_mask.sum()),
                            "ood_eval_images": int(ood_mask.sum()),
                            "auroc": auroc(scores[detector][id_mask], scores[detector][ood_mask]),
                            "score_source": (
                                "genuine computation from"
                                f" {feature_path if detector != 'odin' else odin_path}"
                            ),
                        }
                    )
    return output


def effects(state_rows: list[dict[str, Any]]):
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in state_rows:
        grouped[
            (
                row["dataset"],
                row["dataset_slug"],
                row["detector"],
                row["group_id"],
                row["group_name"],
                row["class_id"],
                row["class_name"],
                row["role"],
                row["withheld_model"],
                int(row["seed"]),
            )
        ].append(row)
    seed_rows = []
    for key, rows in sorted(grouped.items()):
        (
            dataset,
            slug,
            detector,
            group_id,
            group_name,
            class_id,
            class_name,
            role,
            withheld_model,
            seed,
        ) = key
        if len(rows) != 4 or {row["rotation"] for row in rows} != set(MODELS):
            raise RuntimeError(f"Incomplete states: {key}")
        withheld = [row for row in rows if row["state"] == "withheld"]
        present = [row for row in rows if row["state"] == "present"]
        if len(withheld) != 1 or len(present) != 3:
            raise RuntimeError(f"Invalid treatment states: {key}")
        by_rotation = {row["rotation"]: float(row["auroc"]) for row in rows}
        present_mean = float(np.mean([row["auroc"] for row in present]))
        seed_rows.append(
            {
                "dataset": dataset,
                "dataset_slug": slug,
                "detector": detector,
                "group_id": group_id,
                "group_name": group_name,
                "class_id": class_id,
                "class_name": class_name,
                "role": role,
                "withheld_model": withheld_model,
                "seed": seed,
                "auroc_withheld": float(withheld[0]["auroc"]),
                "auroc_present_mean": present_mean,
                **{f"auroc_{model}": by_rotation[model] for model in MODELS},
                "delta": float(withheld[0]["auroc"] - present_mean),
            }
        )

    class_grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        class_grouped[
            (
                row["dataset"],
                row["dataset_slug"],
                row["detector"],
                row["group_id"],
                row["group_name"],
                row["class_id"],
                row["class_name"],
                row["role"],
                row["withheld_model"],
            )
        ].append(row)
    class_rows = []
    for key, rows in sorted(class_grouped.items()):
        if len(rows) != 2 or {row["seed"] for row in rows} != set(SEEDS):
            raise RuntimeError(f"Incomplete seed effects: {key}")
        (
            dataset,
            slug,
            detector,
            group_id,
            group_name,
            class_id,
            class_name,
            role,
            withheld_model,
        ) = key
        class_rows.append(
            {
                "dataset": dataset,
                "dataset_slug": slug,
                "detector": detector,
                "group_id": group_id,
                "group_name": group_name,
                "class_id": class_id,
                "class_name": class_name,
                "role": role,
                "withheld_model": withheld_model,
                "delta_seed0": next(row["delta"] for row in rows if row["seed"] == 0),
                "delta_seed1": next(row["delta"] for row in rows if row["seed"] == 1),
                "mean_auroc_withheld": float(np.mean([row["auroc_withheld"] for row in rows])),
                "mean_auroc_present": float(np.mean([row["auroc_present_mean"] for row in rows])),
                "delta": float(np.mean([row["delta"] for row in rows])),
            }
        )

    group_grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in class_rows:
        group_grouped[
            (
                row["dataset"],
                row["dataset_slug"],
                row["detector"],
                row["group_id"],
                row["group_name"],
            )
        ].append(row)
    group_rows = []
    for key, rows in sorted(group_grouped.items()):
        if len(rows) != 4:
            raise RuntimeError(f"Group does not retain four c classes: {key}")
        dataset, slug, detector, group_id, group_name = key
        group_rows.append(
            {
                "dataset": dataset,
                "dataset_slug": slug,
                "detector": detector,
                "group_id": group_id,
                "group_name": group_name,
                "classes": 4,
                "mean_delta": float(np.mean([row["delta"] for row in rows])),
            }
        )
    return seed_rows, class_rows, group_rows


def summarize_and_bootstrap(seed_rows, class_rows, group_rows):
    seeds = {
        "CIFAR-100": 240521,
        "controlled ImageNet": 240522,
        "iNaturalist 2021 FULL-native": 240523,
    }
    class_by: dict[tuple, list[dict]] = defaultdict(list)
    group_by: dict[tuple, list[dict]] = defaultdict(list)
    seed_by: dict[tuple, list[dict]] = defaultdict(list)
    for row in class_rows:
        class_by[(row["dataset"], row["dataset_slug"], row["detector"])].append(row)
    for row in group_rows:
        group_by[(row["dataset"], row["dataset_slug"], row["detector"])].append(row)
    for row in seed_rows:
        seed_by[(row["dataset"], row["dataset_slug"], row["detector"], row["seed"])].append(row)
    summary_rows = []
    bootstrap_rows = []
    seed_summary = []
    nested = {}
    for key in sorted(class_by):
        dataset, slug, detector = key
        classes = class_by[key]
        groups = sorted(group_by[key], key=lambda row: row["group_id"])
        if len(classes) != 80 or len(groups) != 20:
            raise RuntimeError(f"Aggregation count mismatch: {key}: {len(classes)}/{len(groups)}")
        values = np.asarray([row["delta"] for row in classes], dtype=np.float64)
        group_values = np.asarray([row["mean_delta"] for row in groups], dtype=np.float64)
        rng = np.random.default_rng(seeds[dataset])
        draws = group_values[rng.integers(0, 20, size=(10000, 20))].mean(axis=1)
        low, high = np.percentile(draws, [2.5, 97.5])
        item = {
            "dataset": dataset,
            "dataset_slug": slug,
            "detector": detector,
            "detector_display": DISPLAY[detector],
            "classes": 80,
            "groups": 20,
            "mean_delta": float(values.mean()),
            "median_delta": float(np.median(values)),
            "sd_delta": float(values.std(ddof=0)),
            "min_delta": float(values.min()),
            "max_delta": float(values.max()),
            "negative_classes": int(np.sum(values < 0)),
            "positive_classes": int(np.sum(values > 0)),
            "zero_classes": int(np.sum(values == 0)),
            "negative_groups": int(np.sum(group_values < 0)),
            "positive_groups": int(np.sum(group_values > 0)),
            "zero_groups": int(np.sum(group_values == 0)),
            "bootstrap_draws": 10000,
            "bootstrap_seed": seeds[dataset],
            "ci95_low": float(low),
            "ci95_high": float(high),
        }
        summary_rows.append(item)
        sample_path = ROOT / "bootstrap_samples" / f"{slug}_{detector}.csv"
        write_csv(
            sample_path,
            [{"draw": index + 1, "mean_delta": float(value)} for index, value in enumerate(draws)],
        )
        bootstrap_rows.append(
            {
                key: item[key]
                for key in (
                    "dataset",
                    "dataset_slug",
                    "detector",
                    "detector_display",
                    "groups",
                    "bootstrap_draws",
                    "bootstrap_seed",
                    "mean_delta",
                    "ci95_low",
                    "ci95_high",
                )
            }
        )
        nested.setdefault(slug, {})[detector] = {
            **item,
            "samples_file": str(sample_path),
            "samples_sha256": sha256_file(sample_path),
        }
    for key, rows in sorted(seed_by.items()):
        dataset, slug, detector, seed = key
        values = np.asarray([row["delta"] for row in rows])
        seed_summary.append(
            {
                "dataset": dataset,
                "dataset_slug": slug,
                "detector": detector,
                "seed": seed,
                "classes": len(values),
                "mean_delta": float(values.mean()),
                "median_delta": float(np.median(values)),
                "sd_delta": float(values.std(ddof=0)),
                "negative": int(np.sum(values < 0)),
                "positive": int(np.sum(values > 0)),
                "zero": int(np.sum(values == 0)),
            }
        )
    return summary_rows, bootstrap_rows, seed_summary, nested


def cross_tables(summary_rows: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    lookup = {(row["detector"], row["dataset_slug"]): row for row in summary_rows}
    columns = ["detector", "CIFAR-100", "controlled ImageNet", "iNaturalist"]
    means = []
    cis = []
    labels = {"cifar100": "CIFAR-100", "imagenet": "controlled ImageNet", "inat": "iNaturalist"}
    for detector in DISPLAY_ORDER:
        means.append(
            {
                "detector": DISPLAY[detector],
                **{label: lookup[(detector, slug)]["mean_delta"] for slug, label in labels.items()},
            }
        )
        cis.append(
            {
                "detector": DISPLAY[detector],
                **{
                    label: (
                        f"[{lookup[(detector, slug)]['ci95_low']:.10f},"
                        f" {lookup[(detector, slug)]['ci95_high']:.10f}]"
                    )
                    for slug, label in labels.items()
                },
            }
        )
    return means, cis


def sign_heterogeneity(class_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in class_rows:
        grouped[
            (
                row["dataset"],
                row["dataset_slug"],
                row["group_id"],
                row["group_name"],
                row["class_id"],
                row["class_name"],
            )
        ].append(row)
    output = []
    for key, rows in sorted(grouped.items()):
        dataset, slug, group_id, group_name, class_id, class_name = key
        values = {row["detector"]: row["delta"] for row in rows}
        if set(values) != set(DETECTORS_ALL):
            raise RuntimeError(f"Detector coverage mismatch for class: {key}")
        signs = {
            detector: (
                "negative"
                if values[detector] < 0
                else "positive" if values[detector] > 0 else "zero"
            )
            for detector in DISPLAY_ORDER
        }
        output.append(
            {
                "dataset": dataset,
                "dataset_slug": slug,
                "group_id": group_id,
                "group_name": group_name,
                "class_id": class_id,
                "class_name": class_name,
                **{f"delta_{detector}": values[detector] for detector in DISPLAY_ORDER},
                **{f"sign_{detector}": signs[detector] for detector in DISPLAY_ORDER},
                "distinct_signs": len(set(signs.values())),
                "sign_disagreement": len(set(signs.values())) > 1,
            }
        )
    return output


def fit_metadata_summary() -> dict:
    output = {"feature_detectors": [], "odin": []}
    for slug in ("cifar100", "imagenet", "inat"):
        for model in MODELS:
            for seed in SEEDS:
                tag = f"{model}_seed{seed}"
                for detector in ("mahalanobis", "vim", "neco", "nci", "gradorth"):
                    path = ROOT / "fit_states" / slug / tag / f"{detector}.json"
                    output["feature_detectors"].append(read_json(path))
                output["odin"].append(read_json(ROOT / "raw_scores" / slug / f"{tag}_odin.json"))
    return output


def main() -> None:
    coverage = read_json(ROOT / "raw_score_coverage_all.json")
    if coverage["status"] != "PASS":
        raise RuntimeError("Raw-score coverage gate is not PASS")
    state_rows = []
    for slug in ("cifar100", "imagenet", "inat"):
        state_rows.extend(baseline_rows(slug))
        state_rows.extend(new_rows(slug))
    expected = 3 * 9 * 80 * 2 * 4
    if len(state_rows) != expected:
        raise RuntimeError(f"Expected {expected} per-state rows, got {len(state_rows)}")
    seed_rows, class_rows, group_rows = effects(state_rows)
    summary_rows, bootstrap_rows, seed_summary, nested = summarize_and_bootstrap(
        seed_rows, class_rows, group_rows
    )
    mean_table, ci_table = cross_tables(summary_rows)
    heterogeneity = sign_heterogeneity(class_rows)

    write_csv(ROOT / "per_state_aurocs.csv", state_rows)
    write_csv(ROOT / "seed_level_effects.csv", seed_rows)
    write_csv(ROOT / "class_level_effects.csv", class_rows)
    write_csv(ROOT / "group_level_effects.csv", group_rows)
    write_csv(ROOT / "seed_specific_summary.csv", seed_summary)
    write_csv(ROOT / "bootstrap_summary.csv", bootstrap_rows)
    write_json(ROOT / "bootstrap_summary.json", {"status": "PASS", "datasets": nested})
    write_csv(ROOT / "detector_dataset_summary.csv", summary_rows)
    write_csv(ROOT / "detector_dataset_ci.csv", ci_table)
    write_csv(ROOT / "cross_detector_table.csv", mean_table)
    write_csv(ROOT / "cross_detector_sign_heterogeneity.csv", heterogeneity)
    write_json(ROOT / "fit_metadata_summary.json", fit_metadata_summary())

    reversals = {}
    ranges = {}
    for slug in ("cifar100", "imagenet", "inat"):
        rows = [row for row in summary_rows if row["dataset_slug"] == slug]
        values = np.asarray([row["mean_delta"] for row in rows])
        ranges[slug] = {
            "minimum_detector_mean_delta": float(values.min()),
            "maximum_detector_mean_delta": float(values.max()),
            "range": float(values.max() - values.min()),
        }
    result = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "new_detector_dataset_cells_complete": 18,
        "new_detector_dataset_cells_expected": 18,
        "state_rows": len(state_rows),
        "seed_effect_rows": len(seed_rows),
        "class_effect_rows": len(class_rows),
        "group_effect_rows": len(group_rows),
        "bootstrap_cells": len(bootstrap_rows),
        "raw_score_coverage": "PASS",
        "canonical_knn_energy_reversals_unchanged": reversals,
        "dataset_detector_mean_ranges": ranges,
        "sign_disagreement_classes": {
            slug: sum(
                row["sign_disagreement"] for row in heterogeneity if row["dataset_slug"] == slug
            )
            for slug in ("cifar100", "imagenet", "inat")
        },
        "summary": summary_rows,
    }
    write_json(ROOT / "analysis_summary.json", result)
    print(
        json.dumps(
            {
                "status": "PASS",
                "new_cells": 18,
                "per_state_rows": len(state_rows),
                "bootstrap_cells": len(bootstrap_rows),
            }
        )
    )


if __name__ == "__main__":
    main()
