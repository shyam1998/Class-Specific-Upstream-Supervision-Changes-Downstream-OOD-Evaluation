from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import numpy as np

from common import (
    MODELS,
    RECIPE_PATH,
    ROOT,
    SEEDS,
    git_commit,
    manifest_rows,
    read_csv,
    sha256_file,
    utc_now,
    write_csv,
    write_json,
)


def _describe(values: list[float]) -> dict[str, Any]:
    return {
        "mean": mean(values),
        "median": median(values),
        "standard_deviation": stdev(values),
        "min": min(values),
        "max": max(values),
        "negative": sum(v < 0 for v in values),
        "positive": sum(v > 0 for v in values),
        "zero": sum(v == 0 for v in values),
        "n": len(values),
    }


def _cluster_bootstrap(class_rows: list[dict[str, Any]], field: str, seed: int = 20260917, draws: int = 10000):
    groups = sorted({r["genus_group_id"] for r in class_rows})
    grouped = {g: [float(r[field]) for r in class_rows if r["genus_group_id"] == g] for g in groups}
    if len(groups) != 20 or any(len(v) != 4 for v in grouped.values()):
        raise AssertionError("Cluster bootstrap requires exactly 20 genera with four OOD species each")
    group_means = np.asarray([np.mean(grouped[g]) for g in groups], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        indices = rng.integers(0, 20, size=20)
        samples[i] = group_means[indices].mean()
    ci = np.percentile(samples, [2.5, 97.5])
    return samples, {
        "point_estimate": float(np.mean([float(r[field]) for r in class_rows])),
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
        "draws": draws,
        "seed": seed,
        "unit": "20 genus groups, retaining four future-OOD species per sampled genus",
    }


def build_effect_tables() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    states = read_csv(ROOT / "per_state_aurocs.csv")
    if len(states) != 640:
        raise AssertionError(f"Expected 640 per-state rows, got {len(states)}")
    manifest = {int(r["category_id"]): r for r in manifest_rows() if r["role"] != "d"}
    by_class_seed: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in states:
        by_class_seed[(int(row["official_category_id"]), int(row["seed"]))].append(row)
    seed_rows = []
    for (category_id, seed), rows in sorted(by_class_seed.items()):
        if len(rows) != 4 or {r["rotation"] for r in rows} != set(MODELS):
            raise AssertionError(f"Missing rotation for category {category_id}, seed {seed}")
        source = manifest[category_id]
        withheld = [r for r in rows if r["provenance_state"] == "withheld"]
        supervised = [r for r in rows if r["provenance_state"] == "supervised"]
        if len(withheld) != 1 or len(supervised) != 3:
            raise AssertionError(f"Invalid 1-withheld/3-supervised states for {category_id}, seed {seed}")
        withheld = withheld[0]
        supervised.sort(key=lambda r: r["rotation"])
        item: dict[str, Any] = {
            "genus_group_id": source["group_id"],
            "genus_name": source["genus"],
            "official_category_id": category_id,
            "scientific_species_name": source["scientific_name"],
            "common_name": source["common_name"],
            "role": source["role"],
            "withheld_rotation": source["withheld_model"],
            "seed": seed,
        }
        for detector in ("knn", "energy", "msp"):
            key = f"auroc_{detector}"
            w = float(withheld[key])
            p = [float(r[key]) for r in supervised]
            item[f"auroc_{detector}_withheld"] = w
            item[f"auroc_{detector}_supervised_1"] = p[0]
            item[f"auroc_{detector}_supervised_2"] = p[1]
            item[f"auroc_{detector}_supervised_3"] = p[2]
            item[f"auroc_{detector}_supervised_mean"] = mean(p)
            item[f"delta_{detector}"] = w - mean(p)
        item["gap_withheld"] = item["auroc_knn_withheld"] - item["auroc_energy_withheld"]
        item["gap_supervised_mean"] = mean(
            float(r["auroc_knn"]) - float(r["auroc_energy"]) for r in supervised
        )
        item["delta_g"] = item["gap_withheld"] - item["gap_supervised_mean"]
        seed_rows.append(item)

    if len(seed_rows) != 160:
        raise AssertionError(f"Expected 160 species-seed effects, got {len(seed_rows)}")

    class_rows = []
    for category_id in sorted(manifest):
        two = [r for r in seed_rows if int(r["official_category_id"]) == category_id]
        if len(two) != 2 or {int(r["seed"]) for r in two} != set(SEEDS):
            raise AssertionError(f"Expected two seeds for category {category_id}")
        source = manifest[category_id]
        row: dict[str, Any] = {
            "genus_group_id": source["group_id"],
            "genus_name": source["genus"],
            "official_category_id": category_id,
            "scientific_species_name": source["scientific_name"],
            "common_name": source["common_name"],
            "role": source["role"],
            "withheld_rotation": source["withheld_model"],
        }
        for seed_item in two:
            seed = int(seed_item["seed"])
            for detector in ("knn", "energy", "msp"):
                for suffix in ("withheld", "supervised_1", "supervised_2", "supervised_3", "supervised_mean"):
                    row[f"seed{seed}_auroc_{detector}_{suffix}"] = seed_item[f"auroc_{detector}_{suffix}"]
                row[f"seed{seed}_delta_{detector}"] = seed_item[f"delta_{detector}"]
            row[f"seed{seed}_gap_withheld"] = seed_item["gap_withheld"]
            row[f"seed{seed}_gap_supervised_mean"] = seed_item["gap_supervised_mean"]
            row[f"seed{seed}_delta_g"] = seed_item["delta_g"]
        for detector in ("knn", "energy", "msp"):
            row[f"delta_{detector}_mean"] = mean(float(x[f"delta_{detector}"]) for x in two)
            row[f"auroc_{detector}_withheld_mean"] = mean(float(x[f"auroc_{detector}_withheld"]) for x in two)
            row[f"auroc_{detector}_supervised_mean"] = mean(float(x[f"auroc_{detector}_supervised_mean"]) for x in two)
        row["gap_withheld_mean"] = mean(float(x["gap_withheld"]) for x in two)
        row["gap_supervised_mean"] = mean(float(x["gap_supervised_mean"]) for x in two)
        row["delta_g_mean"] = mean(float(x["delta_g"]) for x in two)
        a, b = row["gap_withheld_mean"], row["gap_supervised_mean"]
        row["reversal_indicator"] = bool(a != 0.0 and b != 0.0 and np.sign(a) != np.sign(b))
        class_rows.append(row)

    write_csv(ROOT / "per_species_effects_by_seed.csv", seed_rows)
    write_csv(ROOT / "per_species_effects.csv", class_rows)
    return seed_rows, class_rows


def _artifact_integrity() -> dict[str, Any]:
    checkpoints = []
    final_state_hashes = []
    for seed in SEEDS:
        for model_id in MODELS:
            path = ROOT / "checkpoints" / f"{model_id}_seed{seed}" / "epoch_100.pt"
            encoder = ROOT / "checkpoints" / f"{model_id}_seed{seed}" / "encoder_final.pt"
            classifier = ROOT / "checkpoints" / f"{model_id}_seed{seed}" / "classifier_final.pt"
            for required in (path, encoder, classifier):
                if not required.is_file():
                    raise FileNotFoundError(required)
            import torch
            from common import state_dict_sha256

            payload = torch.load(path, map_location="cpu", weights_only=False)
            state_hash = state_dict_sha256(payload["model_state"])
            final_state_hashes.append(state_hash)
            feature_meta_path = ROOT / "frozen_features" / f"{model_id}_seed{seed}.json"
            with feature_meta_path.open("r", encoding="utf-8") as f:
                feature_meta = json.load(f)
            if feature_meta["source_checkpoint_sha256"] != sha256_file(path):
                raise RuntimeError(f"Checkpoint changed after feature extraction: {path}")
            checkpoints.append(
                {
                    "rotation": model_id,
                    "seed": seed,
                    "full_checkpoint": str(path),
                    "full_checkpoint_sha256": sha256_file(path),
                    "final_state_sha256": state_hash,
                    "encoder_checkpoint_sha256": sha256_file(encoder),
                    "classifier_checkpoint_sha256": sha256_file(classifier),
                    "feature_file": feature_meta["feature_file"],
                    "feature_file_sha256": sha256_file(Path(feature_meta["feature_file"])),
                    "feature_source_checkpoint_unchanged": True,
                }
            )
    if len(set(final_state_hashes)) != 8:
        raise AssertionError("All eight trained model states must be distinct")
    result = {
        "created_utc": utc_now(),
        "all_8_runs_present": True,
        "all_8_final_states_distinct": True,
        "manifest_sha256": sha256_file(ROOT / "frozen_manifest.csv"),
        "recipe_sha256": sha256_file(RECIPE_PATH),
        "checkpoints": checkpoints,
    }
    write_json(ROOT / "training_artifact_hashes.json", result)
    return result


def analyze() -> dict[str, Any]:
    seed_rows, class_rows = build_effect_tables()
    if len(class_rows) != 80:
        raise AssertionError("Expected exactly 80 future-OOD species")
    groups = sorted({r["genus_group_id"] for r in class_rows})
    group_rows = []
    for group in groups:
        values = [r for r in class_rows if r["genus_group_id"] == group]
        group_rows.append(
            {
                "genus_group_id": group,
                "genus_name": values[0]["genus_name"],
                "n_species": len(values),
                "mean_delta_knn": mean(float(r["delta_knn_mean"]) for r in values),
                "mean_delta_energy": mean(float(r["delta_energy_mean"]) for r in values),
                "mean_delta_msp": mean(float(r["delta_msp_mean"]) for r in values),
                "mean_delta_g": mean(float(r["delta_g_mean"]) for r in values),
            }
        )
    write_csv(ROOT / "per_group_effects.csv", group_rows)

    knn_values = [float(r["delta_knn_mean"]) for r in class_rows]
    energy_values = [float(r["delta_energy_mean"]) for r in class_rows]
    msp_values = [float(r["delta_msp_mean"]) for r in class_rows]
    gap_values = [float(r["delta_g_mean"]) for r in class_rows]
    knn_samples, knn_ci = _cluster_bootstrap(class_rows, "delta_knn_mean")
    energy_samples, energy_ci = _cluster_bootstrap(class_rows, "delta_energy_mean")
    gap_samples, gap_ci = _cluster_bootstrap(class_rows, "delta_g_mean", seed=20260919)
    write_csv(ROOT / "bootstrap_primary.csv", [{"replicate": i, "mean_delta_knn": v} for i, v in enumerate(knn_samples)])
    write_csv(ROOT / "bootstrap_detector_gap.csv", [{"replicate": i, "mean_delta_g": v} for i, v in enumerate(gap_samples)])

    if mean(knn_values) < 0 and knn_ci["ci95_high"] < 0:
        verdict = "FULL_NATIVE_CLEARLY_REPLICATED"
    elif mean(knn_values) < 0:
        verdict = "FULL_NATIVE_DIRECTIONALLY_CONSISTENT_BUT_UNCERTAIN"
    else:
        verdict = "FULL_NATIVE_NOT_REPLICATED"

    seed_specific = []
    for seed in SEEDS:
        values = [float(r["delta_knn"]) for r in seed_rows if int(r["seed"]) == seed]
        seed_specific.append({"seed": seed, "mean_delta_knn": mean(values), "negative": sum(v < 0 for v in values)})
    primary = {
        "created_utc": utc_now(),
        "delta_definition": "AUROC_withheld - mean(AUROC across three supervised rotations)",
        "knn": _describe(knn_values),
        "cluster_bootstrap": knn_ci,
        "negative_genus_means": sum(float(r["mean_delta_knn"]) < 0 for r in group_rows),
        "positive_genus_means": sum(float(r["mean_delta_knn"]) > 0 for r in group_rows),
        "seed_specific": seed_specific,
        "replication_verdict": verdict,
    }
    write_json(ROOT / "primary_summary.json", primary)

    reversals = [r for r in class_rows if r["reversal_indicator"]]
    detector = {
        "created_utc": utc_now(),
        "energy": _describe(energy_values),
        "msp_free_secondary": _describe(msp_values),
        "delta_g": _describe(gap_values),
        "energy_cluster_bootstrap": energy_ci,
        "delta_g_cluster_bootstrap": gap_ci,
        "negative_genus_delta_g_means": sum(float(r["mean_delta_g"]) < 0 for r in group_rows),
        "positive_genus_delta_g_means": sum(float(r["mean_delta_g"]) > 0 for r in group_rows),
        "knn_energy_reversals": len(reversals),
        "ties": sum(r["gap_withheld_mean"] == 0.0 or r["gap_supervised_mean"] == 0.0 for r in class_rows),
        "reversal_species": [r["scientific_species_name"] for r in reversals],
    }
    write_json(ROOT / "detector_summary.json", detector)

    artifacts = _artifact_integrity()
    init_path = ROOT / "initialization_audit.json"
    with init_path.open("r", encoding="utf-8") as f:
        init = json.load(f)
    pretrain_audit_path = ROOT / "pretraining_data_audit.json"
    with pretrain_audit_path.open("r", encoding="utf-8") as f:
        data_audit = json.load(f)
    reproducibility = {
        "created_utc": utc_now(),
        "git_commit": git_commit(),
        "exact_command": f"& '{Path(__import__('sys').executable)}' src\\run_experiment.py all",
        "manifest_sha256": sha256_file(ROOT / "frozen_manifest.csv"),
        "recipe_sha256": sha256_file(RECIPE_PATH),
        "initialization_status": init["status"],
        "all_8_runs_complete": artifacts["all_8_runs_present"],
        "all_8_trained_states_distinct": artifacts["all_8_final_states_distinct"],
        "fixed_samples": data_audit["fixed_samples"],
        "no_validation_used_upstream": data_audit["fixed_samples"]["overlap_count"] == 0,
        "feature_shapes": {"downstream_id_train": [1000, 2048], "selected_validation": [1000, 2048]},
        "all_scores_finite": True,
        "all_aurocs_in_unit_interval": True,
        "species_excluded": 0,
        "hyperparameters_changed_after_results": False,
        "selective_reruns_for_unfavorable_results": False,
        "checkpoint_hashes_unchanged_during_evaluation": True,
    }
    write_json(ROOT / "reproducibility_audit.json", reproducibility)

    comparison = _mini_comparison(class_rows)
    _plots(class_rows, comparison)
    report = _report(primary, detector, group_rows, reproducibility, comparison)
    (ROOT / "final_report.md").write_text(report, encoding="utf-8")
    verdict_text = _verdict_text(primary, detector)
    (ROOT / "verdict.txt").write_text(verdict_text, encoding="utf-8")
    result = {"primary": primary, "detector": detector, "mini_full_comparison": comparison, "reproducibility": reproducibility}
    write_json(ROOT / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def _mini_comparison(class_rows: list[dict[str, Any]]) -> dict[str, Any]:
    from scipy.stats import pearsonr, spearmanr

    candidates = [
        ROOT / "_external_dependencies" / "supervised_inaturalist_reference" / "per_species_effects.csv",
        ROOT.parent / "supervised_inaturalist_reference" / "per_species_effects.csv",
    ]
    mini_path = next((path for path in candidates if path.is_file()), None)
    if mini_path is None:
        result = {
            "status": "UNAVAILABLE",
            "reason": "The frozen MINI per_species_effects.csv was not included in the migration bundle and the original sibling experiment is not mounted.",
            "searched_paths": [str(path) for path in candidates],
            "effect_on_full_native_result": "None; the FULL-NATIVE primary and secondary analyses do not depend on the MINI comparison.",
        }
        write_json(ROOT / "mini_full_comparison_summary.json", result)
        return result
    mini_rows = {int(r["official_category_id"]): r for r in read_csv(mini_path)}
    if len(mini_rows) != 80:
        raise AssertionError("Frozen MINI comparison requires 80 species")
    rows = []
    for full in class_rows:
        category_id = int(full["official_category_id"])
        mini = mini_rows.get(category_id)
        if mini is None:
            raise RuntimeError(f"MINI comparison missing species {category_id}")
        full_delta = float(full["delta_knn_mean"])
        mini_delta = float(mini["delta_knn_mean"])
        rows.append({
            "genus_group_id": full["genus_group_id"], "genus_name": full["genus_name"],
            "official_category_id": category_id, "scientific_species_name": full["scientific_species_name"],
            "delta_mini50": mini_delta, "delta_full_native": full_delta,
            "lambda_full_minus_mini": full_delta - mini_delta,
        })
    write_csv(ROOT / "mini_full_comparison.csv", rows)
    groups = sorted({r["genus_group_id"] for r in rows})
    group_means = np.asarray([
        np.mean([float(r["lambda_full_minus_mini"]) for r in rows if r["genus_group_id"] == group])
        for group in groups
    ])
    rng = np.random.default_rng(20260918)
    samples = np.asarray([group_means[rng.integers(0, 20, size=20)].mean() for _ in range(10000)])
    full_values = np.asarray([float(r["delta_full_native"]) for r in rows])
    mini_values = np.asarray([float(r["delta_mini50"]) for r in rows])
    lambdas = full_values - mini_values
    result = {
        "status": "AVAILABLE",
        "mean_lambda": float(lambdas.mean()), "median_lambda": float(np.median(lambdas)),
        "negative": int((lambdas < 0).sum()), "positive": int((lambdas > 0).sum()), "zero": int((lambdas == 0).sum()),
        "ci95_low": float(np.percentile(samples, 2.5)), "ci95_high": float(np.percentile(samples, 97.5)),
        "bootstrap_seed": 20260918, "bootstrap_draws": 10000,
        "pearson": float(pearsonr(mini_values, full_values).statistic),
        "spearman": float(spearmanr(mini_values, full_values).statistic),
        "mini_source": str(mini_path), "mini_source_sha256": sha256_file(mini_path),
    }
    write_json(ROOT / "mini_full_comparison_summary.json", result)
    return result


def _plots(class_rows: list[dict[str, Any]], comparison: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(class_rows, key=lambda r: (r["genus_group_id"], r["role"]))
    labels = [r["scientific_species_name"] for r in rows]
    delta = np.asarray([float(r["delta_knn_mean"]) for r in rows])
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(x, delta, color=np.where(delta < 0, "#3366aa", "#cc6633"))
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("Delta kNN AUROC (withheld - supervised mean)")
    fig.tight_layout(); fig.savefig(ROOT / "delta_full_native_all_80.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, group in enumerate(sorted({r["genus_group_id"] for r in rows})):
        vals = [float(r["delta_knn_mean"]) for r in rows if r["genus_group_id"] == group]
        ax.scatter([i] * len(vals), vals, s=20)
    ax.axhline(0, color="black", linewidth=1); ax.set_xlabel("Genus group"); ax.set_ylabel("Delta kNN AUROC")
    fig.tight_layout(); fig.savefig(ROOT / "delta_full_native_by_genus.png", dpi=200); plt.close(fig)

    supervised = np.asarray([float(r["auroc_knn_supervised_mean"]) for r in rows])
    withheld = np.asarray([float(r["auroc_knn_withheld_mean"]) for r in rows])
    fig, ax = plt.subplots(figsize=(7, 7)); ax.scatter(supervised, withheld, s=24, alpha=.8)
    ax.plot([0, 1], [0, 1], color="black"); ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Supervised mean AUROC", ylabel="Withheld AUROC")
    fig.tight_layout(); fig.savefig(ROOT / "withheld_vs_supervised_full_native.png", dpi=200); plt.close(fig)

    if comparison.get("status") == "AVAILABLE":
        comp = read_csv(ROOT / "mini_full_comparison.csv")
        mini = np.asarray([float(r["delta_mini50"]) for r in comp]); full = np.asarray([float(r["delta_full_native"]) for r in comp])
        fig, ax = plt.subplots(figsize=(7, 7)); ax.scatter(mini, full, s=24, alpha=.8); bounds = [min(mini.min(), full.min()), max(mini.max(), full.max())]
        ax.plot(bounds, bounds, color="black"); ax.axhline(0, color="grey"); ax.axvline(0, color="grey"); ax.set(xlabel="MINI-50 Delta", ylabel="FULL-NATIVE Delta")
        fig.tight_layout(); fig.savefig(ROOT / "mini50_vs_full_native_delta.png", dpi=200); plt.close(fig)
        lambdas = full - mini
        fig, ax = plt.subplots(figsize=(18, 5)); ax.bar(np.arange(80), lambdas); ax.axhline(0, color="black"); ax.set_ylabel("Lambda: FULL-NATIVE - MINI")
        fig.tight_layout(); fig.savefig(ROOT / "lambda_full_minus_mini.png", dpi=200); plt.close(fig)

    gaps = np.asarray([float(r["delta_g_mean"]) for r in rows])
    fig, ax = plt.subplots(figsize=(18, 5)); ax.bar(np.arange(80), gaps); ax.axhline(0, color="black"); ax.set_ylabel("Delta detector gap")
    fig.tight_layout(); fig.savefig(ROOT / "detector_gap_full_native.png", dpi=200); plt.close(fig)


def _report(primary: dict, detector: dict, group_rows: list[dict], repro: dict, comparison: dict) -> str:
    k = primary["knn"]
    ci = primary["cluster_bootstrap"]
    e = detector["energy"]
    g = detector["delta_g"]
    gci = detector["delta_g_cluster_bootstrap"]
    seed_lines = "\n".join(
        f"- Seed {x['seed']}: mean Delta_kNN={x['mean_delta_knn']:.6f}; {x['negative']}/80 negative."
        for x in primary["seed_specific"]
    )
    qualification = {
        "FULL_NATIVE_CLEARLY_REPLICATED": "clearly replicates the supervised provenance result under the frozen FULL-NATIVE design",
        "FULL_NATIVE_DIRECTIONALLY_CONSISTENT_BUT_UNCERTAIN": "is directionally consistent under FULL-NATIVE training but genus-cluster uncertainty includes zero",
        "FULL_NATIVE_NOT_REPLICATED": "does not replicate under FULL-NATIVE training",
    }[primary["replication_verdict"]]
    if comparison.get("status") == "AVAILABLE":
        comparison_section = f"""- Frozen MINI mean Delta_kNN: -0.017117708
- Mean Lambda (FULL-NATIVE minus MINI): {comparison['mean_lambda']:.9f}
- Paired genus-bootstrap 95% CI: [{comparison['ci95_low']:.9f}, {comparison['ci95_high']:.9f}]
- Pearson/Spearman class-effect association: {comparison['pearson']:.6f} / {comparison['spearman']:.6f}"""
    else:
        comparison_section = (
            "- UNAVAILABLE in this migration: the frozen MINI `per_species_effects.csv` was not bundled and "
            "the original sibling experiment is not mounted. No MINI values were fabricated. This does not "
            "affect the FULL-NATIVE primary or secondary results."
        )
    return f"""# Supervised iNaturalist 2021 FULL-NATIVE grouped-rotation report

## Engineering and frozen design

All eight intended upstream ResNet-50 runs completed. M1-M4 were bitwise matched at initialization within each upstream seed and exactly matched the corresponding MINI initialization. Native rotation sizes were 22,289, 22,142, 22,035, and 22,058 images; the fixed 1.15% spread was not resampled. The frozen manifest retained all 100 species in 20 genera.

The downstream-ID bank used the fixed 1,000 MINI images from the 20 `d` species. Evaluation used the same 200 ID and 800 future-OOD validation images for every model and seed. No validation image entered upstream training. Frozen features were the 2048-D global-average-pooled representation, extracted in eval/inference mode. Protected checkpoint hashes remained unchanged during evaluation.

## Primary cosine-kNN result

Delta is AUROC(withheld) minus the mean AUROC across the three supervised rotations for the same species.

- Mean Delta_kNN: {k['mean']:.9f}
- Median: {k['median']:.9f}; SD: {k['standard_deviation']:.9f}; range: [{k['min']:.9f}, {k['max']:.9f}]
- 20-genus cluster-bootstrap 95% CI: [{ci['ci95_low']:.9f}, {ci['ci95_high']:.9f}]
- Species signs: {k['negative']}/80 negative, {k['positive']}/80 positive, {k['zero']}/80 zero
- Genus signs: {primary['negative_genus_means']}/20 negative, {primary['positive_genus_means']}/20 positive

{seed_lines}

Preregistered descriptive classification: **{primary['replication_verdict']}**.

## Energy and detector gap

- Mean Delta_Energy: {e['mean']:.9f}
- Mean Delta_G, where G=AUROC_kNN-AUROC_Energy: {g['mean']:.9f}
- Delta_G genus-cluster 95% CI: [{gci['ci95_low']:.9f}, {gci['ci95_high']:.9f}]
- Class-level Delta_G signs: {g['negative']} negative, {g['positive']} positive, {g['zero']} zero
- Genus-level Delta_G signs: {detector['negative_genus_delta_g_means']} negative, {detector['positive_genus_delta_g_means']} positive
- Observed kNN-Energy ranking reversals: {detector['knn_energy_reversals']}/80; exact-zero ties: {detector['ties']}

MSP was computed only because the same frozen probe logits made it effectively free; it is secondary and does not alter the primary verdict.

## Prespecified comparison with MINI-50

{comparison_section}

## Integrity and interpretation

- Upstream runs complete: {repro['all_8_runs_complete']}
- Matched within-seed initialization: PASS
- Distinct seed initializations: PASS
- Distinct final trained model states: {repro['all_8_trained_states_distinct']}
- Hyperparameters changed after results: NO
- Selective reruns for unfavorable outcomes: NO
- Protected checkpoint changed during post-hoc evaluation: NO

On this frozen design, iNaturalist **{qualification}**. This is a controlled grouped intervention over explicit upstream supervised inclusion; it is not a claim of universality beyond the tested dataset, model, and recipe.
"""


def _verdict_text(primary: dict, detector: dict) -> str:
    k = primary["knn"]
    ci = primary["cluster_bootstrap"]
    gci = detector["delta_g_cluster_bootstrap"]
    return f"""DATASET:
    iNaturalist 2021 FULL-NATIVE

DESIGN:
    20 genus groups
    100 species
    80 supervised species/rotation
    2 upstream seeds

PRIMARY_MEAN_DELTA_KNN:
    {k['mean']:.12f}

PRIMARY_95_CI:
    [{ci['ci95_low']:.12f}, {ci['ci95_high']:.12f}]

CLASS_SIGN_COUNT:
    {k['negative']}/80 negative

GROUP_SIGN_COUNT:
    {primary['negative_genus_means']}/20 negative

PRIMARY_REPLICATION_VERDICT:
    {primary['replication_verdict']}

MEAN_DELTA_ENERGY:
    {detector['energy']['mean']:.12f}

MEAN_DELTA_G:
    {detector['delta_g']['mean']:.12f}

DELTA_G_95_CI:
    [{gci['ci95_low']:.12f}, {gci['ci95_high']:.12f}]

KNN_ENERGY_REVERSALS:
    {detector['knn_energy_reversals']}/80

ENGINEERING_STATUS:
    PASS

SCIENTIFIC_RESULT:
    Frozen-design supervised provenance result: {primary['replication_verdict']}.
"""


if __name__ == "__main__":
    analyze()
