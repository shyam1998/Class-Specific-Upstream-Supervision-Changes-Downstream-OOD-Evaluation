from __future__ import annotations

import hashlib
import os
import unittest

import numpy as np
import torch
from torch import nn

from analysis.detectors.src.common import canonical_state_hash, load_run
from analysis.detectors.src.detectors import (
    fit_gradorth,
    fit_mahalanobis,
    fit_nci,
    fit_neco,
    fit_vim,
    nci_alpha,
    probe_logits,
    score_gradorth,
    score_mahalanobis,
    score_nci,
    score_neco,
    score_vim,
)
from analysis.detectors.src.models import load_encoder_probe
from analysis.detectors.src.run_odin import odin_batch

ARTIFACT_TESTS = all(os.environ.get(name) for name in
                     ("CIFAR_SUPERVISED_ROOT", "IMAGENET_SUPERVISED_ROOT", "INAT_SUPERVISED_ROOT"))


class DetectorProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(12037)
        cls.reference = rng.normal(size=(80, 16)).astype(np.float32)
        cls.labels = np.asarray([f"c{i % 4}" for i in range(80)])
        cls.evaluation = rng.normal(size=(23, 16)).astype(np.float32)
        cls.weight = rng.normal(scale=0.2, size=(20, 16)).astype(np.float32)
        cls.bias = rng.normal(scale=0.1, size=20).astype(np.float32)

    def test_score_orientation_toys(self):
        # Mahalanobis: a far point has a larger distance.
        reference = np.asarray([[-1.1, 0], [-0.9, 0.1], [0.9, -0.1], [1.1, 0]], dtype=np.float32)
        labels = np.asarray(["a", "a", "b", "b"])
        state, _ = fit_mahalanobis(reference, labels)
        values = score_mahalanobis(np.asarray([[1.0, 0], [1.0, 8.0]], dtype=np.float32), state)
        self.assertGreater(values[1], values[0])

        # ViM: with a fixed positive alpha and equal logits, residual magnitude raises OOD score.
        vim_state = {"origin": np.zeros(2), "residual_basis": np.asarray([[0.0], [1.0]]), "alpha": np.asarray(1.0)}
        w = np.zeros((20, 2), dtype=np.float32); b = np.zeros(20, dtype=np.float32)
        values = score_vim(np.asarray([[1.0, 0.0], [1.0, 3.0]], dtype=np.float32), w, b, vim_state)
        self.assertGreater(values[1], values[0])

        # NECO: a vector outside the retained subspace approaches zero and is more OOD-oriented.
        neco_state = {
            "scaler_mean": np.zeros(2), "scaler_scale": np.ones(2), "pca_mean": np.zeros(2),
            "pca_components": np.asarray([[1.0, 0.0]]),
        }
        values = score_neco(np.asarray([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32), neco_state)
        self.assertGreater(values[1], values[0])

        # NCI and GradOrth are stored as negatives of their ID-oriented scores.
        nci_state = {"global_mean": np.zeros(2), "alpha": np.asarray(0.0)}
        w = np.zeros((20, 2), dtype=np.float32); w[0, 0] = 2.0
        b = np.zeros(20, dtype=np.float32); b[0] = 1.0
        values = score_nci(np.asarray([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32), w, b, nci_state)
        self.assertGreater(values[1], values[0])
        grad_state = {"basis": np.asarray([[1.0], [0.0]])}
        values = score_gradorth(np.asarray([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32), w, b, grad_state)
        self.assertGreater(values[1], values[0])

        # ODIN/MSP orientation: lower maximum probability produces a larger OOD score.
        confidence = np.asarray([0.99, 0.55])
        ood_score = 1.0 - confidence
        self.assertGreater(ood_score[1], ood_score[0])

    def test_fit_repeat_determinism_and_finite_values(self):
        fitters = {
            "mahalanobis": lambda: fit_mahalanobis(self.reference, self.labels),
            "vim": lambda: fit_vim(self.reference, self.weight, self.bias),
            "neco": lambda: fit_neco(self.reference),
            "nci": lambda: fit_nci(np.pad(self.reference, ((0, 0), (0, 496)))),
            "gradorth": lambda: fit_gradorth(self.reference),
        }
        for name, fit in fitters.items():
            first_arrays, first_meta = fit()
            second_arrays, second_meta = fit()
            self.assertEqual(first_meta["fit_state_sha256"], second_meta["fit_state_sha256"], name)
            for array in first_arrays.values():
                if np.asarray(array).dtype.kind not in "OUS":
                    self.assertTrue(np.isfinite(array).all(), name)

    def test_chunk_size_invariance(self):
        state, _ = fit_mahalanobis(self.reference, self.labels)
        np.testing.assert_allclose(score_mahalanobis(self.evaluation, state, 3), score_mahalanobis(self.evaluation, state, 19), rtol=0, atol=1e-6)
        state, _ = fit_vim(self.reference, self.weight, self.bias)
        np.testing.assert_allclose(score_vim(self.evaluation, self.weight, self.bias, state, 3), score_vim(self.evaluation, self.weight, self.bias, state, 19), rtol=0, atol=1e-6)
        state, _ = fit_neco(self.reference)
        np.testing.assert_allclose(score_neco(self.evaluation, state, 3), score_neco(self.evaluation, state, 19), rtol=0, atol=1e-6)
        padded_ref = np.pad(self.reference, ((0, 0), (0, 496))); padded_eval = np.pad(self.evaluation, ((0, 0), (0, 496)))
        padded_weight = np.pad(self.weight, ((0, 0), (0, 496)))
        state, _ = fit_nci(padded_ref)
        np.testing.assert_allclose(score_nci(padded_eval, padded_weight, self.bias, state, 3), score_nci(padded_eval, padded_weight, self.bias, state, 19), rtol=0, atol=1e-6)
        state, _ = fit_gradorth(self.reference)
        np.testing.assert_allclose(score_gradorth(self.evaluation, self.weight, self.bias, state, 3), score_gradorth(self.evaluation, self.weight, self.bias, state, 19), rtol=0, atol=1e-6)

    def test_mahalanobis_numerical_rank_audit(self):
        base = np.arange(40, dtype=np.float64).reshape(20, 2)
        reference = np.c_[base, base[:, :1] * 2]
        labels = np.asarray(["a"] * 10 + ["b"] * 10)
        _, metadata = fit_mahalanobis(reference, labels)
        self.assertLess(metadata["numerical_rank"], 3)
        self.assertGreater(metadata["rank_tolerance"], 0)
        self.assertTrue(metadata["precision_finite"])

    def test_vim_neco_nci_fixed_audits(self):
        _, vim = fit_vim(self.reference, self.weight, self.bias)
        self.assertEqual(vim["principal_dimension"], 8)
        self.assertEqual(vim["residual_dimension"], 8)
        self.assertTrue(np.isfinite(vim["alpha"]))
        _, neco = fit_neco(self.reference)
        self.assertGreaterEqual(neco["cumulative_explained_variance"], 0.90)
        self.assertEqual(neco["pca_solver"], "covariance_eigh")
        self.assertEqual(nci_alpha(512), 0.01)
        self.assertEqual(nci_alpha(2048), 0.001)

    def test_gradorth_closed_form_equals_autograd_projection(self):
        torch.manual_seed(19)
        h = torch.randn(7, 11, dtype=torch.float64)
        weight = torch.randn(20, 11, dtype=torch.float64, requires_grad=True)
        bias = torch.randn(20, dtype=torch.float64, requires_grad=True)
        q, _ = torch.linalg.qr(torch.randn(11, 5, dtype=torch.float64))
        values = []
        for i in range(len(h)):
            probabilities = torch.softmax(weight @ h[i] + bias, dim=0)
            loss = -(torch.full((20,), 1 / 20, dtype=torch.float64) * torch.log(probabilities)).sum()
            gradient = torch.autograd.grad(loss, weight, retain_graph=True)[0]
            projected = gradient @ q @ q.T
            values.append(torch.linalg.matrix_norm(projected))
        explicit = torch.stack(values)
        probabilities = torch.softmax(h @ weight.detach().T + bias.detach(), dim=1)
        closed = torch.linalg.vector_norm(probabilities - 1 / 20, dim=1) * torch.linalg.vector_norm(h @ q, dim=1)
        torch.testing.assert_close(explicit, closed, rtol=1e-12, atol=1e-12)

    def test_odin_sign_channel_scaling_and_epsilon_zero(self):
        torch.manual_seed(23)
        model = nn.Sequential(nn.Flatten(), nn.Linear(12, 20, bias=True)).eval()
        model.requires_grad_(False)
        x = torch.randn(4, 3, 2, 2)
        std = torch.tensor([0.2, 0.4, 0.8]).view(1, 3, 1, 1)
        score, logits, gradient, perturbed = odin_batch(model, x, 1000.0, 0.002, std)
        signed = torch.where(gradient >= 0, torch.ones_like(gradient), -torch.ones_like(gradient))
        torch.testing.assert_close(perturbed, x - 0.002 * signed / std, rtol=0, atol=0)
        zero_score, _, _, zero_perturbed = odin_batch(model, x, 1000.0, 0.0, std)
        direct = 1.0 - torch.softmax(logits / 1000.0, dim=1).max(dim=1).values
        torch.testing.assert_close(zero_perturbed, x, rtol=0, atol=0)
        torch.testing.assert_close(zero_score, direct, rtol=1e-6, atol=1e-7)
        self.assertTrue(torch.isfinite(score).all())

    @unittest.skipUnless(ARTIFACT_TESTS, "requires saved experiment artifacts")
    def test_probe_logit_reconstruction(self):
        data = load_run("cifar100", "M1", 0)
        reconstructed = probe_logits(data.eval_features, data.probe_weight, data.probe_bias)
        root = os.path.join(os.environ["CIFAR_SUPERVISED_ROOT"], "detector_audit/scores/scores_m1_seed0.npz")
        with np.load(root, allow_pickle=False) as source:
            maximum = float(np.max(np.abs(reconstructed - source["logits"])))
        self.assertLessEqual(maximum, 1e-6)

    @unittest.skipUnless(ARTIFACT_TESTS, "requires saved experiment artifacts")
    def test_strict_checkpoint_load_and_no_mutation(self):
        for slug in ("cifar100", "imagenet"):
            data = load_run(slug, "M1", 0)
            model, metadata = load_encoder_probe(data)
            before = canonical_state_hash(model.state_dict())
            model.eval(); model.requires_grad_(False)
            after = canonical_state_hash(model.state_dict())
            self.assertEqual(before, after)
            self.assertTrue(metadata["strict_checkpoint_load"] and metadata["strict_probe_load"])

    @unittest.skipUnless(ARTIFACT_TESTS, "requires saved experiment artifacts")
    def test_canonical_identity_adapters(self):
        for slug in ("cifar100", "imagenet", "inat"):
            data = load_run(slug, "M1", 0)
            self.assertEqual(len(data.d_classes), 20)
            self.assertEqual(len(data.candidates), 80)
            self.assertEqual({item["withheld_model"] for item in data.candidates}, {"M1", "M2", "M3", "M4"})


if __name__ == "__main__":
    unittest.main()
