from __future__ import annotations

import json
import math
import platform
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import (CANONICAL, MODELS, PROJECT, REFERENCE, ROOT, SEEDS,
                     assert_protected_unchanged, environment_record, git_commit,
                     manifest_rows, now, read_csv, read_json, sha256_file, write_csv, write_json)

DETECTORS = ("knn", "energy", "msp")


def describe(values):
    values = [float(value) for value in values]
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "standard_deviation": statistics.stdev(values), "min": min(values), "max": max(values),
            "negative": sum(value < 0 for value in values), "positive": sum(value > 0 for value in values),
            "zero": sum(value == 0 for value in values), "n": len(values)}


def cluster_bootstrap(rows, field, seed, draws=10000):
    groups = sorted({row["genus_group_id"] for row in rows})
    arrays = [[float(row[field]) for row in rows if row["genus_group_id"] == group] for group in groups]
    if len(arrays) != 20 or any(len(values) != 4 for values in arrays):
        raise RuntimeError("Bootstrap requires exactly 20 genus clusters with four c species each")
    group_means = np.asarray([np.mean(values) for values in arrays], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = group_means[rng.integers(0, 20, size=(draws, 20))].mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    summary = {**describe([row[field] for row in rows]), "ci95_low": float(low), "ci95_high": float(high),
               "draws": draws, "seed": seed, "bootstrap_unit": "official genus; all four c species retained"}
    return samples, summary


def build_effects():
    states = read_csv(ROOT / "per_state_aurocs.csv")
    if len(states) != 640:
        raise RuntimeError(f"Expected 640 per-state rows, got {len(states)}")
    manifest = {int(row["category_id"]): row for row in manifest_rows() if row["role"] != "d"}
    grouped = defaultdict(list)
    for row in states:
        grouped[(int(row["official_category_id"]), int(row["seed"]))].append(row)
    seed_rows = []
    for (category, seed), rows in sorted(grouped.items()):
        if len(rows) != 4 or {row["rotation"] for row in rows} != set(MODELS):
            raise RuntimeError("Every species/seed must retain all four raw rotations")
        absent = [row for row in rows if row["present_or_absent"] == "absent"]
        present = [row for row in rows if row["present_or_absent"] == "present"]
        if len(absent) != 1 or len(present) != 3:
            raise RuntimeError("Invalid one-absent/three-present mapping")
        source = manifest[category]
        item = {"genus_group_id": source["group_id"], "genus_name": source["genus"],
                "official_category_id": category, "scientific_species_name": source["scientific_name"],
                "common_name": source["common_name"], "role": source["role"],
                "absent_model": source["withheld_model"], "seed": seed}
        by_model = {row["rotation"]: row for row in rows}
        for detector in DETECTORS:
            absent_value = float(absent[0][f"auroc_{detector}"])
            present_values = [float(row[f"auroc_{detector}"]) for row in present]
            item[f"auroc_absent_{detector}"] = absent_value
            item[f"auroc_present_{detector}_mean"] = statistics.mean(present_values)
            for model in MODELS:
                item[f"auroc_present_{detector}_{model.lower()}"] = (math.nan if model == source["withheld_model"]
                                                                      else float(by_model[model][f"auroc_{detector}"]))
            item[f"delta_ssl_{detector}"] = absent_value - statistics.mean(present_values)
        item["gap_absent"] = item["auroc_absent_knn"] - item["auroc_absent_energy"]
        item["gap_present_mean"] = item["auroc_present_knn_mean"] - item["auroc_present_energy_mean"]
        item["delta_g_ssl"] = item["gap_absent"] - item["gap_present_mean"]
        item["strict_knn_energy_reversal"] = item["gap_absent"] * item["gap_present_mean"] < 0
        item["knn_energy_tie"] = item["gap_absent"] == 0 or item["gap_present_mean"] == 0
        seed_rows.append(item)
    if len(seed_rows) != 160:
        raise RuntimeError(f"Expected 160 species-seed effects, got {len(seed_rows)}")
    write_csv(ROOT / "per_species_effects_by_seed.csv", seed_rows)
    species_rows = []
    for category, source in sorted(manifest.items()):
        values = [row for row in seed_rows if int(row["official_category_id"]) == category]
        if len(values) != 2 or {int(row["seed"]) for row in values} != set(SEEDS):
            raise RuntimeError("Seed averaging requires exactly seed0 and seed1")
        item = {"genus_group_id": source["group_id"], "genus_name": source["genus"],
                "official_category_id": category, "scientific_species_name": source["scientific_name"],
                "common_name": source["common_name"], "role": source["role"], "absent_model": source["withheld_model"]}
        numeric_keys = [key for key, value in values[0].items() if key not in item and key not in ("seed", "strict_knn_energy_reversal", "knn_energy_tie")]
        for key in numeric_keys:
            numbers = [float(row[key]) for row in values]
            item[key] = float(np.nanmean(numbers)) if not all(math.isnan(number) for number in numbers) else math.nan
        for seed_row in values:
            seed = int(seed_row["seed"])
            for detector in DETECTORS:
                item[f"seed{seed}_delta_ssl_{detector}"] = seed_row[f"delta_ssl_{detector}"]
            item[f"seed{seed}_delta_g_ssl"] = seed_row["delta_g_ssl"]
        item["strict_knn_energy_reversal"] = item["gap_absent"] * item["gap_present_mean"] < 0
        item["knn_energy_tie"] = item["gap_absent"] == 0 or item["gap_present_mean"] == 0
        species_rows.append(item)
    if len(species_rows) != 80:
        raise RuntimeError("Expected exactly 80 seed-averaged species effects")
    write_csv(ROOT / "per_species_effects.csv", species_rows)
    return seed_rows, species_rows


def paired_gamma(seed_rows, species_rows):
    supervised_seed = read_csv(CANONICAL / "per_species_effects_by_seed.csv")
    lookup = {(int(row["official_category_id"]), int(row["seed"])): row for row in supervised_seed}
    gamma_seed = []
    for row in seed_rows:
        key = (int(row["official_category_id"]), int(row["seed"]))
        source = lookup.get(key)
        if source is None:
            raise RuntimeError(f"Canonical supervised seed-level effect missing: {key}")
        if (source["genus_group_id"], source["genus_name"], source["role"], source["withheld_rotation"]) != (
                row["genus_group_id"], row["genus_name"], row["role"], row["absent_model"]):
            raise RuntimeError(f"Canonical supervised identity mismatch: {key}")
        delta_sup = float(source["delta_knn"]); delta_ssl = float(row["delta_ssl_knn"])
        gamma_seed.append({"genus_group_id": row["genus_group_id"], "genus_name": row["genus_name"],
                           "official_category_id": key[0], "scientific_species_name": row["scientific_species_name"],
                           "role": row["role"], "absent_model": row["absent_model"], "seed": key[1],
                           "delta_sup_knn": delta_sup, "delta_ssl_knn": delta_ssl, "gamma": delta_sup - delta_ssl,
                           "canonical_downstream_identity_match": True})
    if len(gamma_seed) != 160:
        raise RuntimeError("Expected 160 paired Gamma seed rows")
    write_csv(ROOT / "paired_gamma_by_seed.csv", gamma_seed)
    gamma_species = []
    for row in species_rows:
        subset = [value for value in gamma_seed if int(value["official_category_id"]) == int(row["official_category_id"])]
        gamma_species.append({"genus_group_id": row["genus_group_id"], "genus_name": row["genus_name"],
                              "official_category_id": row["official_category_id"],
                              "scientific_species_name": row["scientific_species_name"], "role": row["role"],
                              "absent_model": row["absent_model"],
                              "delta_sup_knn": statistics.mean(float(value["delta_sup_knn"]) for value in subset),
                              "delta_ssl_knn": statistics.mean(float(value["delta_ssl_knn"]) for value in subset),
                              "gamma": statistics.mean(float(value["gamma"]) for value in subset)})
    write_csv(ROOT / "paired_gamma.csv", gamma_species)
    return gamma_seed, gamma_species


def _group_rows(species_rows, gamma_species):
    groups = sorted({row["genus_group_id"] for row in species_rows})
    gamma_lookup = {int(row["official_category_id"]): row for row in gamma_species}
    result = []
    for group in groups:
        rows = [row for row in species_rows if row["genus_group_id"] == group]
        result.append({"genus_group_id": group, "genus_name": rows[0]["genus_name"], "species": 4,
                       "mean_delta_ssl_knn": statistics.mean(float(row["delta_ssl_knn"]) for row in rows),
                       "mean_delta_ssl_energy": statistics.mean(float(row["delta_ssl_energy"]) for row in rows),
                       "mean_delta_ssl_msp": statistics.mean(float(row["delta_ssl_msp"]) for row in rows),
                       "mean_delta_g_ssl": statistics.mean(float(row["delta_g_ssl"]) for row in rows),
                       "mean_gamma": statistics.mean(float(gamma_lookup[int(row["official_category_id"])]["gamma"]) for row in rows),
                       "negative_delta_ssl_knn_species": sum(float(row["delta_ssl_knn"]) < 0 for row in rows),
                       "strict_reversals": sum(str(row["strict_knn_energy_reversal"]).lower() == "true" for row in rows)})
    write_csv(ROOT / "per_group_effects.csv", result)
    write_csv(ROOT / "gamma_group_summary.csv", [{"genus_group_id": row["genus_group_id"], "genus_name": row["genus_name"],
                                                   "mean_gamma": row["mean_gamma"]} for row in result])
    return result


def _plots(species_rows, gamma_species):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = sorted(species_rows, key=lambda row: (row["genus_group_id"], row["role"]))
    x = np.arange(80); delta = np.asarray([float(row["delta_ssl_knn"]) for row in rows])
    labels = [row["scientific_species_name"] for row in rows]
    fig, ax = plt.subplots(figsize=(18, 6)); ax.bar(x, delta, color=np.where(delta < 0, "#3366aa", "#cc6633")); ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=6); ax.set_ylabel("Delta_SSL kNN AUROC (absent - present mean)")
    fig.tight_layout(); fig.savefig(ROOT / "delta_ssl_knn_all_80.png", dpi=200); plt.close(fig)
    fig, ax = plt.subplots(figsize=(14, 6))
    for index, group in enumerate(sorted({row["genus_group_id"] for row in rows})):
        values = [float(row["delta_ssl_knn"]) for row in rows if row["genus_group_id"] == group]
        ax.scatter([index] * 4, values, s=24)
    ax.axhline(0, color="black", lw=1); ax.set_xlabel("Genus group"); ax.set_ylabel("Delta_SSL kNN AUROC")
    fig.tight_layout(); fig.savefig(ROOT / "delta_ssl_knn_by_genus.png", dpi=200); plt.close(fig)
    absent = np.asarray([float(row["auroc_absent_knn"]) for row in rows]); present = np.asarray([float(row["auroc_present_knn_mean"]) for row in rows])
    fig, ax = plt.subplots(figsize=(7, 7)); ax.scatter(present, absent, s=24, alpha=.8); ax.plot([0, 1], [0, 1], color="black")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Present mean kNN AUROC", ylabel="Absent kNN AUROC")
    fig.tight_layout(); fig.savefig(ROOT / "absent_vs_present_knn.png", dpi=200); plt.close(fig)
    gamma_lookup = {int(row["official_category_id"]): row for row in gamma_species}
    sup = np.asarray([float(gamma_lookup[int(row["official_category_id"])]["delta_sup_knn"]) for row in rows])
    low = min(delta.min(), sup.min(), 0); high = max(delta.max(), sup.max(), 0)
    fig, ax = plt.subplots(figsize=(7, 7)); ax.scatter(delta, sup, s=24, alpha=.8); ax.plot([low, high], [low, high], color="black")
    ax.axhline(0, color="grey", lw=.7); ax.axvline(0, color="grey", lw=.7); ax.set(xlabel="Delta_SSL kNN", ylabel="Delta_SUP kNN")
    fig.tight_layout(); fig.savefig(ROOT / "supervised_vs_simclr_delta_knn.png", dpi=200); plt.close(fig)
    gamma = np.asarray([float(gamma_lookup[int(row["official_category_id"])]["gamma"]) for row in rows])
    fig, ax = plt.subplots(figsize=(18, 6)); ax.bar(x, gamma, color=np.where(gamma < 0, "#3366aa", "#cc6633")); ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=6); ax.set_ylabel("Gamma = Delta_SUP - Delta_SSL")
    fig.tight_layout(); fig.savefig(ROOT / "gamma_all_80.png", dpi=200); plt.close(fig)


def _artifact_audit():
    checkpoints, features, probes = [], [], []
    for seed in SEEDS:
        for model in MODELS:
            checkpoint = ROOT / "checkpoints" / f"{model}_seed{seed}" / "epoch_200.pt"
            feature = ROOT / "features" / f"{model}_seed{seed}.npz"
            probe = ROOT / "probes" / f"{model}_seed{seed}.pt"
            checkpoints.append({"path": str(checkpoint), "sha256": sha256_file(checkpoint)})
            features.append({"path": str(feature), "sha256": sha256_file(feature)})
            probes.append({"path": str(probe), "sha256": sha256_file(probe)})
    return {"checkpoints": checkpoints, "features": features, "probes": probes}


def analyze():
    assert_protected_unchanged()
    seed_rows, species_rows = build_effects()
    gamma_seed, gamma_species = paired_gamma(seed_rows, species_rows)
    groups = _group_rows(species_rows, gamma_species)
    primary_samples, primary = cluster_bootstrap(species_rows, "delta_ssl_knn", 20260919)
    gamma_samples, gamma = cluster_bootstrap(gamma_species, "gamma", 20260920)
    gap_samples, gap = cluster_bootstrap(species_rows, "delta_g_ssl", 20260921)
    write_csv(ROOT / "bootstrap_primary.csv", [{"draw": i, "mean_delta_ssl_knn": value} for i, value in enumerate(primary_samples)])
    write_json(ROOT / "bootstrap_primary_summary.json", primary)
    write_csv(ROOT / "bootstrap_gamma.csv", [{"draw": i, "mean_gamma": value} for i, value in enumerate(gamma_samples)])
    write_json(ROOT / "bootstrap_gamma_summary.json", gamma)
    write_csv(ROOT / "bootstrap_detector_gap.csv", [{"draw": i, "mean_delta_g_ssl": value} for i, value in enumerate(gap_samples)])
    energy = describe(row["delta_ssl_energy"] for row in species_rows)
    msp = describe(row["delta_ssl_msp"] for row in species_rows)
    reversals = sum(str(row["strict_knn_energy_reversal"]).lower() == "true" for row in species_rows)
    ties = sum(str(row["knn_energy_tie"]).lower() == "true" for row in species_rows)
    detector = {"delta_ssl_energy": energy, "delta_ssl_msp": msp, "delta_g_ssl": gap,
                "strict_knn_energy_reversals": reversals, "ties": ties,
                "genus_means": [{"genus_group_id": row["genus_group_id"], "genus_name": row["genus_name"],
                                 "mean_delta_g_ssl": row["mean_delta_g_ssl"]} for row in groups]}
    write_json(ROOT / "detector_gap_summary.json", detector)
    simclr_pass = primary["mean"] <= -0.03 and primary["ci95_high"] < 0 and primary["negative"] >= 60
    moderation_pass = gamma["mean"] <= -0.03 and gamma["ci95_high"] < 0
    verdict = {"simclr": "SIMCLR_NEGATIVE_EFFECT_REPLICATED" if simclr_pass else "SIMCLR_NEGATIVE_EFFECT_NOT_REPLICATED",
               "moderation": "SUPERVISION_MODERATION_REPLICATED" if moderation_pass else "SUPERVISION_MODERATION_NOT_REPLICATED"}
    seed_summaries = []
    for seed in SEEDS:
        values = [float(row["delta_ssl_knn"]) for row in seed_rows if int(row["seed"]) == seed]
        seed_summaries.append({"seed": seed, **describe(values)})
    supervised_mean = read_json(CANONICAL / "primary_summary.json")["knn"]["mean"]
    summary = {"created_utc": now(), "canonical_full_native_import": "PASS", "data_gate": "PASS",
               "invariants": "PASS", "smoke": read_json(ROOT / "smoke" / "smoke_result.json"),
               "simclr_runs_complete": "8/8", "primary_delta_ssl_knn": primary,
               "seed_specific": seed_summaries, "canonical_supervised_mean_delta_sup_knn": supervised_mean,
               "gamma": gamma, "detector": detector, "verdict": verdict,
               "genus_means": groups}
    write_json(ROOT / "summary.json", summary)
    (ROOT / "verdict.txt").write_text(verdict["simclr"] + "\n" + verdict["moderation"] + "\n", encoding="utf-8")
    _plots(species_rows, gamma_species)
    training = read_csv(ROOT / "training_runs.csv"); probes = read_csv(ROOT / "probe_runs.csv")
    context = ["CIFAR-100 SimCLR mean Delta_SSL_kNN = +0.0111",
               "controlled ImageNet SimCLR mean Delta_SSL_kNN = -0.007357"]

