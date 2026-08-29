import unittest
from pathlib import Path

from config_loader import load_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigLoaderTests(unittest.TestCase):
    def test_dataset_configs_inherit_base_values(self):
        for dataset in ("synapse", "acdc", "isic"):
            config = load_config(str(ROOT / "configs" / f"{dataset}.yaml"))
            self.assertEqual(config["data"]["dataset_name"], dataset)
            self.assertEqual(config["data"]["image_size"], [224, 224])
            self.assertEqual(config["training"]["learning_rate"], 3e-4)
            self.assertIn("teacher", config["model"])
            self.assertIn("student", config["model"])

    def test_split_seed_is_fixed_across_training_seeds(self):
        config = load_config(str(ROOT / "configs" / "synapse.yaml"))
        self.assertEqual(config["data"]["split"]["seed"], 2027)
        self.assertEqual(config["reproducibility"]["seed"], 2027)

    def test_synapse_has_background_plus_eight_organs(self):
        config = load_config(str(ROOT / "configs" / "synapse.yaml"))
        self.assertEqual(config["data"]["num_classes"], 9)
        self.assertEqual(config["model"]["student"]["num_classes"], 9)


if __name__ == "__main__":
    unittest.main()
