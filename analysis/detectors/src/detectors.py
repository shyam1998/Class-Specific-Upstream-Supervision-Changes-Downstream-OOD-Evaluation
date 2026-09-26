from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import EmpiricalCovariance, empirical_covariance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def probe_logits(features: np.ndarray, weight: np.ndarray, bias: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
    outputs = []
    weight_t = torch.from_numpy(np.asarray(weight, dtype=np.float32))
    bias_t = torch.from_numpy(np.asarray(bias, dtype=np.float32))
    with torch.inference_mode():
        for start in range(0, len(features), chunk_size):
            current = torch.from_numpy(np.asarray(features[start:start + chunk_size], dtype=np.float32))
            outputs.append(F.linear(current, weight_t, bias_t).numpy())
    return np.concatenate(outputs, axis=0)


def logsumexp(values: np.ndarray) -> np.ndarray:
    maximum = np.max(values, axis=1)
    return maximum + np.log(np.exp(values - maximum[:, None]).sum(axis=1))


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def state_hash(arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _spectral_rank(covariance: np.ndarray) -> tuple[int, float, np.ndarray]:
    eigenvalues = np.linalg.eigvalsh(covariance)
    largest = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    tolerance = float(max(covariance.shape) * np.finfo(covariance.dtype).eps * largest)
    rank = int(np.sum(eigenvalues > tolerance))
    return rank, tolerance, eigenvalues


def fit_mahalanobis(reference: np.ndarray, labels: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    x = np.asarray(reference, dtype=np.float64)
    classes = np.unique(labels.astype(str))
    means = np.stack([x[labels.astype(str) == item].mean(axis=0) for item in classes])
    lookup = {item: index for index, item in enumerate(classes.tolist())}
    residuals = x - means[np.asarray([lookup[item] for item in labels.astype(str)], dtype=np.int64)]
    estimator = EmpiricalCovariance(assume_centered=True, store_precision=True).fit(residuals)
    covariance = np.asarray(estimator.covariance_, dtype=np.float64)
    precision = np.asarray(estimator.precision_, dtype=np.float64)
    rank, tolerance, eigenvalues = _spectral_rank(covariance)
    arrays = {"classes": classes, "means": means, "covariance": covariance, "precision": precision, "eigenvalues": eigenvalues}
    metadata = {
        "detector": "mahalanobis", "fit_dtype": "float64", "assume_centered": True,
        "class_count": int(len(classes)), "reference_count": int(len(x)), "feature_dim": int(x.shape[1]),
        "numerical_rank": rank, "rank_tolerance": tolerance, "precision_finite": bool(np.isfinite(precision).all()),
        "shrinkage": None, "input_preprocessing": None, "feature_normalization": None,
    }
    if not metadata["precision_finite"]:
        raise FloatingPointError("Mahalanobis precision is non-finite")
    metadata["fit_state_sha256"] = state_hash(arrays, metadata)
    return arrays, metadata


def score_mahalanobis(features: np.ndarray, state: dict[str, np.ndarray], chunk_size: int = 1024) -> np.ndarray:
    means = state["means"]
    precision = state["precision"]
    output = np.empty(len(features), dtype=np.float64)
    for start in range(0, len(features), chunk_size):
        x = np.asarray(features[start:start + chunk_size], dtype=np.float64)
        best = np.full(len(x), np.inf, dtype=np.float64)
        for mean in means:
            diff = x - mean
            distance = np.einsum("ij,ij->i", diff @ precision, diff, optimize=True)
            best = np.minimum(best, distance)
        output[start:start + len(x)] = best
    if not np.isfinite(output).all():
        raise FloatingPointError("Non-finite Mahalanobis score")
    return output


def vim_dimension(feature_dim: int) -> int:
    if feature_dim >= 2048:
        return 1000
    if feature_dim >= 768:
        return 512
    return feature_dim // 2


def fit_vim(reference: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    x = np.asarray(reference, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64)
    origin = -np.linalg.pinv(w, rcond=1e-15) @ b
    centered = x - origin
    covariance = empirical_covariance(centered, assume_centered=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    dimension = vim_dimension(x.shape[1])
    if dimension >= x.shape[1]:
        raise ValueError("ViM principal dimension leaves no residual subspace")
    residual_basis = eigenvectors[:, : x.shape[1] - dimension]
    residual_norm = np.linalg.norm(centered @ residual_basis, axis=1)
    logits = probe_logits(reference, weight, bias).astype(np.float64)
    denominator = float(residual_norm.mean())
    if not np.isfinite(denominator) or denominator <= 0:
        raise FloatingPointError("Invalid ViM ID residual denominator")
    alpha = float(np.max(logits, axis=1).mean() / denominator)
    arrays = {
        "origin": origin, "covariance_eigenvalues": eigenvalues,
        "residual_basis": residual_basis, "alpha": np.asarray(alpha, dtype=np.float64),
    }
    metadata = {
        "detector": "vim", "fit_dtype": "float64", "feature_dim": int(x.shape[1]),
        "reference_count": int(len(x)), "principal_dimension": dimension,
        "residual_dimension": int(residual_basis.shape[1]), "pinv_rcond": 1e-15,
        "alpha": alpha, "mean_id_residual_norm": denominator,
        "mean_id_max_logit": float(np.max(logits, axis=1).mean()),
        "eigenvalue_min": float(eigenvalues[0]), "eigenvalue_max": float(eigenvalues[-1]),
    }
    metadata["fit_state_sha256"] = state_hash(arrays, metadata)
    return arrays, metadata


def score_vim(features: np.ndarray, weight: np.ndarray, bias: np.ndarray, state: dict[str, np.ndarray], chunk_size: int = 1024) -> np.ndarray:
    origin = state["origin"]
    residual_basis = state["residual_basis"]
    alpha = float(state["alpha"])
    output = np.empty(len(features), dtype=np.float64)
    for start in range(0, len(features), chunk_size):
        x = np.asarray(features[start:start + chunk_size], dtype=np.float64)
        residual = np.linalg.norm((x - origin) @ residual_basis, axis=1)
        logits = probe_logits(features[start:start + chunk_size], weight, bias).astype(np.float64)
        output[start:start + len(x)] = alpha * residual - logsumexp(logits)
    if not np.isfinite(output).all():
        raise FloatingPointError("Non-finite ViM score")
    return output


def fit_neco(reference: np.ndarray, threshold: float = 0.90) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    x = np.asarray(reference, dtype=np.float64)
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(x)
    standardized = scaler.transform(x)
    pca = PCA(n_components=None, svd_solver="covariance_eigh", copy=True).fit(standardized)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    dimension = int(np.searchsorted(cumulative, threshold, side="left") + 1)
    if dimension > len(cumulative) or cumulative[dimension - 1] < threshold:
        raise RuntimeError("NECO PCA could not reach frozen explained-variance threshold")
    arrays = {
        "scaler_mean": scaler.mean_.astype(np.float64), "scaler_scale": scaler.scale_.astype(np.float64),
        "pca_mean": pca.mean_.astype(np.float64), "pca_components": pca.components_[:dimension].astype(np.float64),
        "explained_variance": pca.explained_variance_.astype(np.float64),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float64),
    }
    metadata = {
        "detector": "neco", "fit_dtype": "float64", "feature_dim": int(x.shape[1]),
        "reference_count": int(len(x)), "pca_solver": "covariance_eigh",
        "variance_threshold": threshold, "pca_dimension": dimension,
        "cumulative_explained_variance": float(cumulative[dimension - 1]),
        "maxlogit_multiplication": False, "architecture_branch": "ResNet",
        "constant_coordinate_count": int(np.sum(scaler.var_ == 0)),
    }
    metadata["fit_state_sha256"] = state_hash(arrays, metadata)
    return arrays, metadata


def score_neco(features: np.ndarray, state: dict[str, np.ndarray], chunk_size: int = 1024) -> np.ndarray:
    output = np.empty(len(features), dtype=np.float64)
    for start in range(0, len(features), chunk_size):
        x = np.asarray(features[start:start + chunk_size], dtype=np.float64)
        standardized = (x - state["scaler_mean"]) / state["scaler_scale"]
        centered = standardized - state["pca_mean"]
        denominator = np.linalg.norm(standardized, axis=1)
        if np.any(denominator == 0):
            raise FloatingPointError("Zero NECO standardized feature norm")
        projected = centered @ state["pca_components"].T
        output[start:start + len(x)] = -np.linalg.norm(projected, axis=1) / denominator
    if not np.isfinite(output).all():
        raise FloatingPointError("Non-finite NECO score")
    return output


def nci_alpha(feature_dim: int) -> float:
    if feature_dim == 512:
        return 1e-2
    if feature_dim == 2048:
        return 1e-3
    raise ValueError(f"No frozen NCI alpha for feature dimension {feature_dim}")


def fit_nci(reference: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    x = np.asarray(reference, dtype=np.float64)
    alpha = nci_alpha(x.shape[1])
    arrays = {"global_mean": x.mean(axis=0), "alpha": np.asarray(alpha, dtype=np.float64)}
    metadata = {
        "detector": "nci", "fit_dtype": "float64", "feature_dim": int(x.shape[1]),
        "reference_count": int(len(x)), "alpha": alpha, "norm_filter": "L1",
        "alpha_selection": "frozen architecture-regime transfer; no sweep",
    }
    metadata["fit_state_sha256"] = state_hash(arrays, metadata)
    return arrays, metadata


def score_nci(features: np.ndarray, weight: np.ndarray, bias: np.ndarray, state: dict[str, np.ndarray], chunk_size: int = 1024) -> np.ndarray:
    output = np.empty(len(features), dtype=np.float64)
    mean = state["global_mean"]
    alpha = float(state["alpha"])
    w = np.asarray(weight, dtype=np.float64)
    for start in range(0, len(features), chunk_size):
        original = features[start:start + chunk_size]
        x = np.asarray(original, dtype=np.float64)
        logits = probe_logits(original, weight, bias)
        predicted = np.argmax(logits, axis=1)
        centered = x - mean
        denominator = np.linalg.norm(centered, axis=1)
        if np.any(denominator == 0):
            raise FloatingPointError("Zero NCI centered-feature norm")
        projection = np.einsum("ij,ij->i", w[predicted], centered) / denominator
        id_score = projection + alpha * np.linalg.norm(x, ord=1, axis=1)
        output[start:start + len(x)] = -id_score
    if not np.isfinite(output).all():
        raise FloatingPointError("Non-finite NCI score")
    return output


def fit_gradorth(reference: np.ndarray, threshold: float = 0.97) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    x = np.asarray(reference, dtype=np.float64)
    _, singular_values, vt = np.linalg.svd(x, full_matrices=False)
    energy = singular_values ** 2
    cumulative = np.cumsum(energy) / energy.sum()
    rank = int(np.searchsorted(cumulative, threshold, side="left") + 1)
    basis = vt[:rank].T.copy()
    arrays = {"basis": basis, "singular_values": singular_values}
    metadata = {
        "detector": "gradorth", "fit_dtype": "float64", "feature_dim": int(x.shape[1]),
        "reference_count": int(len(x)), "representation_centered": False,
        "svd_energy_threshold": threshold, "retained_rank": rank,
        "retained_energy_fraction": float(cumulative[rank - 1]),
        "target_distribution": "uniform_20", "gradient_scope": "probe_weight_only",
        "reference_policy": "all canonical downstream-ID references",
    }
    metadata["fit_state_sha256"] = state_hash(arrays, metadata)
    return arrays, metadata


def score_gradorth(features: np.ndarray, weight: np.ndarray, bias: np.ndarray, state: dict[str, np.ndarray], chunk_size: int = 1024) -> np.ndarray:
    output = np.empty(len(features), dtype=np.float64)
    uniform = np.full(20, 1.0 / 20.0, dtype=np.float64)
    for start in range(0, len(features), chunk_size):
        original = features[start:start + chunk_size]
        x = np.asarray(original, dtype=np.float64)
        probabilities = softmax(probe_logits(original, weight, bias).astype(np.float64))
        error_norm = np.linalg.norm(probabilities - uniform, axis=1)
        projection_norm = np.linalg.norm(x @ state["basis"], axis=1)
        output[start:start + len(x)] = -(error_norm * projection_norm)
    if not np.isfinite(output).all():
        raise FloatingPointError("Non-finite GradOrth score")
    return output


FIT_FUNCTIONS = {
    "mahalanobis": fit_mahalanobis,
    "vim": fit_vim,
    "neco": fit_neco,
    "nci": fit_nci,
    "gradorth": fit_gradorth,
}


def fit_and_score_all(reference: np.ndarray, labels: np.ndarray, evaluation: np.ndarray,
                      weight: np.ndarray, bias: np.ndarray, chunk_size: int = 1024):
    states: dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]] = {}
    scores: dict[str, np.ndarray] = {}

    states["mahalanobis"] = fit_mahalanobis(reference, labels)
    scores["mahalanobis"] = score_mahalanobis(evaluation, states["mahalanobis"][0], chunk_size)

    states["vim"] = fit_vim(reference, weight, bias)
    scores["vim"] = score_vim(evaluation, weight, bias, states["vim"][0], chunk_size)

    states["neco"] = fit_neco(reference)
    scores["neco"] = score_neco(evaluation, states["neco"][0], chunk_size)

    states["nci"] = fit_nci(reference)
    scores["nci"] = score_nci(evaluation, weight, bias, states["nci"][0], chunk_size)

    states["gradorth"] = fit_gradorth(reference)
    scores["gradorth"] = score_gradorth(evaluation, weight, bias, states["gradorth"][0], chunk_size)
    return states, scores
