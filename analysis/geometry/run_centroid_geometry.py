#!/usr/bin/env python3
"""Detector-independent centroid geometry versus primary kNN detectability."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
RESULTS = Path(os.environ.get("DETECTOR_RESULTS_ROOT", REPO / "outputs/detectors"))
sys.path.insert(0, str(REPO))
from analysis.detectors.src.common import MODELS, SEEDS, load_run, sha256_array, sha256_file  # noqa: E402

DATASETS = (
    ("cifar100", "CIFAR-100", "#0072B2"),
    ("imagenet", "controlled ImageNet", "#D55E00"),
    ("inat", "iNaturalist 2021 FULL-native", "#009E73"),
)
N_BOOTSTRAP = 100_000
BOOTSTRAP_SEED = 20260907


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def unit_rows(features: np.ndarray) -> np.ndarray:
    value = features.astype(np.float64, copy=False)
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    if not np.isfinite(value).all() or np.any(norms <= 0):
        raise RuntimeError("Non-finite or zero-norm feature")
    return value / norms


def mean_direction(normalized_features: np.ndarray) -> np.ndarray:
    centroid = normalized_features.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError("Undefined normalized class centroid")
    return centroid / norm


def state_geometry(run) -> list[dict]:
    reference = unit_rows(run.reference_features)
    evaluation = unit_rows(run.eval_features)
    id_centroids = {
        class_id: mean_direction(reference[run.reference_class_ids == class_id])
        for class_id in run.d_classes
    }
    if any(np.sum(run.reference_class_ids == class_id) == 0 for class_id in run.d_classes):
        raise RuntimeError("Missing downstream-ID reference class")
    id_ids = np.asarray(sorted(id_centroids))
    id_matrix = np.stack([id_centroids[class_id] for class_id in id_ids])
    pooled_id_centroid = mean_direction(reference)
    rows = []
    for item in run.candidates:
        mask = run.eval_class_ids == item["class_id"]
        focal = mean_direction(evaluation[mask])
        distances = 1.0 - id_matrix @ focal
        nearest_index = int(np.argmin(distances))
        rows.append({
            "dataset": run.dataset, "dataset_slug": run.slug, "seed": run.seed,
            "model": run.model, "semantic_group_id": item["group_id"],
            "semantic_group": item["group_name"], "class_id": item["class_id"],
            "class_name": item["class_name"], "withheld_model": item["withheld_model"],
            "state": "withheld" if run.model == item["withheld_model"] else "supervised",
            "n_ood_images": int(mask.sum()),
            "nearest_id_class_id": id_ids[nearest_index],
            "nearest_id_class_cosine_distance": float(distances[nearest_index]),
            "pooled_id_centroid_cosine_distance": float(1.0 - focal @ pooled_id_centroid),
        })
    return rows


def collect_state_geometry():
    rows, audits = [], []
    for slug, dataset, _ in DATASETS:
        cross_seed_hashes = {"reference": set(), "evaluation": set()}
        for seed in SEEDS:
            expected = None
            for model in MODELS:
                print(f"[{slug} seed {seed}] {model}", flush=True)
                run = load_run(slug, model, seed)
                signature = (run.reference_ids, run.reference_class_ids, run.eval_ids,
                             run.eval_class_ids, run.candidates, run.d_classes)
                if expected is None:
                    expected = signature
                else:
                    for observed, first in zip(signature[:4], expected[:4]):
                        if not np.array_equal(observed, first):
                            raise RuntimeError("Cross-model image identity or class-label mismatch")
                    if signature[4:] != expected[4:]:
                        raise RuntimeError("Cross-model rotation design mismatch")
                if set(run.reference_ids.tolist()) & set(run.eval_ids.tolist()):
                    raise RuntimeError("Reference/evaluation image overlap")
                new_rows = state_geometry(run)
                if len(new_rows) != 80:
                    raise RuntimeError("Expected 80 focal-class centroid measurements")
                rows.extend(new_rows)
                audits.append({
                    "dataset": dataset, "dataset_slug": slug, "seed": seed, "model": model,
                    "reference_identity_sha256": sha256_array(run.reference_ids),
                    "evaluation_identity_sha256": sha256_array(run.eval_ids),
                    "feature_files": {str(p): sha256_file(p) for p in run.feature_paths},
                })
                cross_seed_hashes["reference"].add(sha256_array(run.reference_ids))
                cross_seed_hashes["evaluation"].add(sha256_array(run.eval_ids))
                del run
        if any(len(value) != 1 for value in cross_seed_hashes.values()):
            raise RuntimeError(f"Cross-seed image identities differ for {slug}")
    return pd.DataFrame(rows), audits


def aggregate(state: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "dataset_slug", "semantic_group_id", "semantic_group",
            "class_id", "class_name", "withheld_model"]
    seed_rows = []
    for identity, frame in state.groupby(keys + ["seed"], sort=False):
        withheld = frame[frame.state == "withheld"]
        supervised = frame[frame.state == "supervised"]
        if len(withheld) != 1 or len(supervised) != 3 or set(frame.model) != set(MODELS):
            raise RuntimeError("Invalid one-withheld/three-supervised state structure")
        row = dict(zip(keys + ["seed"], identity))
        row.update({
            "n_ood_images": int(withheld.n_ood_images.iloc[0]),
            "nearest_distance_withheld": withheld.nearest_id_class_cosine_distance.iloc[0],
            "nearest_distance_supervised": supervised.nearest_id_class_cosine_distance.mean(),
            "delta_nearest_id_centroid": supervised.nearest_id_class_cosine_distance.mean()
                                         - withheld.nearest_id_class_cosine_distance.iloc[0],
            "pooled_distance_withheld": withheld.pooled_id_centroid_cosine_distance.iloc[0],
            "pooled_distance_supervised": supervised.pooled_id_centroid_cosine_distance.mean(),
            "delta_pooled_id_centroid": supervised.pooled_id_centroid_cosine_distance.mean()
                                        - withheld.pooled_id_centroid_cosine_distance.iloc[0],
            "nearest_id_withheld": withheld.nearest_id_class_id.iloc[0],
            "nearest_ids_supervised": ";".join(supervised.sort_values("model").nearest_id_class_id),
        })
        seed_rows.append(row)
    seed_table = pd.DataFrame(seed_rows)
    if not (seed_table.groupby(["dataset_slug", "seed"]).size() == 80).all():
        raise RuntimeError("Expected 80 class-level effects per dataset/seed")
    numeric = ["n_ood_images", "nearest_distance_withheld", "nearest_distance_supervised",
               "delta_nearest_id_centroid", "pooled_distance_withheld",
               "pooled_distance_supervised", "delta_pooled_id_centroid"]
    wide = seed_table.pivot(index=keys, columns="seed", values=numeric)
    wide.columns = [f"{name}_seed{seed}" for name, seed in wide.columns]
    wide = wide.reset_index()
    for metric in ("delta_nearest_id_centroid", "delta_pooled_id_centroid"):
        wide[metric] = (wide[f"{metric}_seed0"] + wide[f"{metric}_seed1"]) / 2.0
    for column in ("n_ood_images_seed0", "n_ood_images_seed1"):
        wide[column] = wide[column].astype(int)
    return seed_table, wide


def join_primary(wide: pd.DataFrame) -> pd.DataFrame:
    primary = pd.read_csv(RESULTS / "class_level_effects.csv",
                          dtype={"class_id": str, "group_id": str}, float_precision="round_trip")
    primary = primary[(primary.detector == "knn") &
                      primary.dataset_slug.isin([x[0] for x in DATASETS])]
    primary = primary[["dataset_slug", "class_id", "group_id", "group_name", "class_name",
                       "withheld_model", "delta_seed0", "delta_seed1", "delta"]].rename(columns={
                           "group_id": "check_group_id", "group_name": "check_group",
                           "class_name": "check_class", "withheld_model": "check_withheld",
                           "delta_seed0": "delta_knn_seed0", "delta_seed1": "delta_knn_seed1",
                           "delta": "delta_knn"})
    merged = wide.merge(primary, on=["dataset_slug", "class_id"], validate="one_to_one",
                        how="outer", indicator=True)
    if len(merged) != 240 or set(merged._merge) != {"both"}:
        raise RuntimeError("Incomplete centroid/primary effect join")
    for left, right in [("semantic_group_id", "check_group_id"),
                        ("semantic_group", "check_group"), ("class_name", "check_class"),
                        ("withheld_model", "check_withheld")]:
        if not (merged[left].astype(str) == merged[right].astype(str)).all():
            raise RuntimeError(f"Join metadata mismatch: {left}")
    return merged.drop(columns=["_merge", "check_group_id", "check_group", "check_class",
                                "check_withheld"]).sort_values(
                                    ["dataset_slug", "semantic_group_id", "class_id"], kind="stable")


def cluster_bootstrap(frame, x_column, rng, n_bootstrap):
    groups = frame.semantic_group_id.unique()
    if len(groups) != 20 or any(sum(frame.semantic_group_id == g) != 4 for g in groups):
        raise RuntimeError("Expected 20 groups of four classes")
    values = {g: frame.loc[frame.semantic_group_id == g, [x_column, "delta_knn"]].to_numpy()
              for g in groups}
    draws = np.empty(n_bootstrap)
    for start in range(0, n_bootstrap, 1000):
        sample = rng.integers(0, 20, size=(min(1000, n_bootstrap - start), 20))
        for offset, indices in enumerate(sample):
            xy = np.concatenate([values[groups[i]] for i in indices])
            draws[start + offset] = np.corrcoef(xy[:, 0], xy[:, 1])[0, 1]
    if not np.isfinite(draws).all():
        raise RuntimeError("Non-finite bootstrap result")
    return draws, np.quantile(draws, [.025, .975])


def analyze(merged, n_bootstrap):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summaries, all_draws = [], []
    metrics = [("nearest_id_class_centroid", "delta_nearest_id_centroid"),
               ("pooled_id_centroid", "delta_pooled_id_centroid")]
    for slug, dataset, _ in DATASETS:
        frame = merged[merged.dataset_slug == slug]
        for metric, column in metrics:
            pearson = pearsonr(frame[column], frame.delta_knn)
            spearman = spearmanr(frame[column], frame.delta_knn)
            draws, ci = cluster_bootstrap(frame, column, rng, n_bootstrap)
            summaries.append({"dataset": dataset, "dataset_slug": slug, "geometry": metric,
                              "n_classes": 80, "n_semantic_groups": 20,
                              "pearson_r": pearson.statistic,
                              "pearson_p_two_sided_iid_classes": pearson.pvalue,
                              "pearson_group_bootstrap_ci_low": ci[0],
                              "pearson_group_bootstrap_ci_high": ci[1],
                              "spearman_rho": spearman.statistic,
                              "spearman_p_two_sided_iid_classes": spearman.pvalue,
                              "bootstrap_replicates": n_bootstrap})
            all_draws.append(pd.DataFrame({"dataset": dataset, "dataset_slug": slug,
                                           "geometry": metric,
                                           "bootstrap_replicate": np.arange(n_bootstrap),
                                           "pearson_r": draws}))
    return pd.DataFrame(summaries), pd.concat(all_draws, ignore_index=True)


def plot_panel(ax, frame, result, color, dataset):
    x = frame.delta_nearest_id_centroid
    y = frame.delta_knn
    ax.axhline(0, color="#777777", lw=.8, ls=":")
    ax.axvline(0, color="#777777", lw=.8, ls=":")
    ax.scatter(x, y, s=34, facecolor=color, edgecolor="white", linewidth=.55, alpha=.86, zorder=2)
    slope, intercept = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 100)
    ax.plot(xx, intercept + slope * xx, color="#222222", lw=1.25)
    ax.set_title(dataset, fontsize=13)
    ax.set_xlabel("Change in distance to nearest\ndownstream-ID class centroid\n(supervised − withheld)")
    ax.set_ylabel("Change in kNN AUROC\n(withheld − supervised)")
    ax.text(.04, .04, f"Pearson r = {result.pearson_r:+.3f}\n"
            f"95% CI [{result.pearson_group_bootstrap_ci_low:+.3f}, "
            f"{result.pearson_group_bootstrap_ci_high:+.3f}]\n"
            f"Spearman ρ = {result.spearman_rho:+.3f}", transform=ax.transAxes, va="bottom",
            fontsize=10.5, bbox={"boxstyle": "round,pad=.35", "facecolor": "white",
                               "edgecolor": "#cccccc", "alpha": .92})
    ax.grid(alpha=.16, lw=.5)
    ax.set_axisbelow(True)


def draw(merged, results, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.labelsize": 11.5, "xtick.labelsize": 10.5,
                         "ytick.labelsize": 10.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    title = "Classes that move farther from downstream ID in representation space tend to become easier to detect as OOD"
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.1), layout="constrained")
    for ax, (slug, dataset, color) in zip(axes, DATASETS):
        result = results[(results.dataset_slug == slug) &
                         (results.geometry == "nearest_id_class_centroid")].iloc[0]
        plot_panel(ax, merged[merged.dataset_slug == slug], result, color, dataset)
    fig.suptitle(title, fontsize=15)
    fig.savefig(output / "centroid_geometry_vs_knn_all_datasets.pdf")
    fig.savefig(output / "centroid_geometry_vs_knn_all_datasets.png", dpi=300)
    plt.close(fig)
    for slug, dataset, color in DATASETS:
        fig, ax = plt.subplots(figsize=(6.3, 5.5), layout="constrained")
        result = results[(results.dataset_slug == slug) &
                         (results.geometry == "nearest_id_class_centroid")].iloc[0]
        plot_panel(ax, merged[merged.dataset_slug == slug], result, color, dataset)
        fig.suptitle(title, fontsize=13)
        fig.savefig(output / f"{slug}_centroid_geometry_vs_knn.pdf")
        fig.savefig(output / f"{slug}_centroid_geometry_vs_knn.png", dpi=300)
        plt.close(fig)


def main():
    state, audits = collect_state_geometry()
    seed_table, wide = aggregate(state)
    merged = join_primary(wide)
    results, draws = analyze(merged, N_BOOTSTRAP)
    state.to_csv(ROOT / "per_state_centroid_geometry.csv", index=False, float_format="%.17g")
    seed_table.to_csv(ROOT / "seed_level_centroid_geometry.csv", index=False, float_format="%.17g")
    merged.to_csv(ROOT / "class_centroid_geometry_knn_effects.csv", index=False, float_format="%.17g")
    results.to_csv(ROOT / "centroid_correlations.csv", index=False, float_format="%.17g")
    draws.to_csv(ROOT / "centroid_group_bootstrap_draws.csv.gz", index=False,
                 float_format="%.17g", compression="gzip")
    draw(merged, results, ROOT)
    # Re-read the principal table and assert all manuscript-facing invariants.
    check = pd.read_csv(ROOT / "class_centroid_geometry_knn_effects.csv",
                        dtype={"class_id": str, "semantic_group_id": str}, float_precision="round_trip")
    if len(check) != 240 or set(check.dataset_slug) != {x[0] for x in DATASETS}:
        raise RuntimeError("Invalid manuscript output dataset/count")
    validation = {"status": "PASS", "rows": 240, "datasets": 3, "classes_per_dataset": 80,
                  "groups_per_dataset": 20,
                  "reference_eval_disjoint": True, "cross_model_and_seed_identities_matched": True}
    write_json(ROOT / "validation.json", validation)
    outputs = [p for p in ROOT.iterdir() if p.is_file() and p.name != "analysis_metadata.json"]
    write_json(ROOT / "analysis_metadata.json", {
        "status": "complete", "primary_geometry": "nearest downstream-ID class centroid cosine distance",
        "feature_preprocessing": "L2-normalize every image feature; average within class; L2-normalize resulting mean direction",
        "focal_centroid_images": "all fixed focal-class OOD test images",
        "id_centroid_images": "fixed downstream-ID training images, separately for each of 20 ID classes",
        "nearest_operation": "minimum cosine distance over the 20 class centroids; no image-neighbor search",
        "supervised_aggregation": "average three scalar centroid distances after computing each within its encoder",
        "seed_aggregation": "compute class effect within seed, then average seeds 0 and 1",
        "sensitivity_geometry": "cosine distance to pooled downstream-ID training mean direction",
        "kNN_independence": "geometry does not use kNN scores, nearest images, k=50, or held-out ID scores",
        "bootstrap": {"method": "20-semantic-group percentile cluster bootstrap",
                      "replicates": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED},
        "input_audits": audits, "validation": validation,
        "outputs": {p.name: sha256_file(p) for p in outputs},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "script_sha256": sha256_file(Path(__file__)),
    })
    print(results.to_string(index=False))
    print(f"Saved detector-independent centroid analysis to {ROOT}")


if __name__ == "__main__":
    main()
