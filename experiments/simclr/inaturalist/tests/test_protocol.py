import math
import unittest

import torch

from experiments.simclr.inaturalist.src.common import MODELS, ROOT, ROTATION_TOTALS, SEEDS, full_model_sha256, read_csv, read_json, state_dict_sha256, torch_load
from experiments.simclr.inaturalist.src.data import simclr_transform, upstream_rows
from experiments.simclr.inaturalist.src.model import make_model, nt_xent


class ProtocolTests(unittest.TestCase):
    def test_frozen_design(self):
        manifest = read_csv(ROOT / "manifest.csv")
        self.assertEqual(len(manifest), 100)
        self.assertEqual(len({row["group_id"] for row in manifest}), 20)
        self.assertEqual(sum(row["role"] == "d" for row in manifest), 20)
        self.assertEqual(sum(row["role"] != "d" for row in manifest), 80)
        for model in MODELS:
            self.assertEqual(len(upstream_rows(model)), ROTATION_TOTALS[model])

    def test_exact_reference_augmentation(self):
        names = [type(item).__name__ for item in simclr_transform().transforms]
        self.assertEqual(names, ["RandomResizedCrop", "RandomHorizontalFlip", "RandomApply",
                                 "RandomGrayscale", "ToTensor", "Normalize"])
        self.assertNotIn("GaussianBlur", names)

    def test_model_and_objective(self):
        model = make_model(False)
        self.assertEqual(model.projector[0].in_features, 2048)
        self.assertEqual(model.projector[0].out_features, 512)
        self.assertIsInstance(model.projector[1], torch.nn.BatchNorm1d)
        self.assertEqual(model.projector[-1].out_features, 128)
        z1 = torch.nn.functional.normalize(torch.randn(8, 128), dim=1)
        z2 = torch.nn.functional.normalize(torch.randn(8, 128), dim=1)
        loss, targets = nt_xent(z1, z2, 0.5)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(tuple(targets.shape), (16,))

    @unittest.skipUnless((ROOT / "initialization_audit.json").is_file(), "requires generated initialization artifacts")
    def test_initialization_audit(self):
        records = read_json(ROOT / "initialization_audit.json")
        self.assertEqual(records["status"], "PASS")
        hashes = []
        for seed in SEEDS:
            payload = torch_load(ROOT / "checkpoints" / f"initial_seed{seed}.pt")
            metadata = payload["metadata"]
            self.assertEqual(state_dict_sha256(payload["encoder_state"]), metadata["encoder_sha256"])
            self.assertEqual(state_dict_sha256(payload["projector_state"]), metadata["projector_sha256"])
            self.assertEqual(full_model_sha256(payload["encoder_state"], payload["projector_state"]), metadata["full_model_sha256"])
            hashes.append(metadata["full_model_sha256"])
        self.assertNotEqual(hashes[0], hashes[1])

    def test_preregistered_protocol(self):
        config = read_json(ROOT / "config.json")
        self.assertEqual(config["simclr"]["source_batch_size"], 256)
        self.assertEqual(config["simclr"]["representations_per_loss"], 512)
        self.assertEqual(config["simclr"]["epochs"], 200)
        self.assertEqual(config["simclr"]["temperature"], 0.5)
        self.assertEqual(config["downstream"]["probe"]["batch_size"], 512)
        self.assertEqual(config["bootstrap"]["primary_seed"], 20260919)
        self.assertEqual(config["bootstrap"]["gamma_seed"], 20260920)
        self.assertEqual(config["bootstrap"]["detector_gap_seed"], 20260921)
        self.assertTrue(read_json(ROOT / "preregistration_snapshot.json")["labels_in_objective"] is False)


if __name__ == "__main__":
    unittest.main()
