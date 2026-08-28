import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class ReportedResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "results" / "reported_results.yaml").open("r", encoding="utf-8") as stream:
            cls.results = yaml.safe_load(stream)

    def test_headline_values_match_manuscript(self):
        expected = {
            "synapse": (86.45, 12.34, "nsd", 78.56),
            "acdc": (91.89, 5.67, "nsd", 91.34),
            "isic2018": (90.89, 9.45, "jaccard", 83.12),
        }
        for dataset, (dice, hd95, extra_name, extra_value) in expected.items():
            row = self.results[dataset]["methods"]["hiermedkd"]
            self.assertEqual(row["dice"], dice)
            self.assertEqual(row["hd95_mm"] if dataset != "isic2018" else row["hd95_px"], hd95)
            self.assertEqual(row[extra_name], extra_value)

    def test_entropy_gate_ablation_matches_manuscript(self):
        full = self.results["synapse_ablation"]["full"]
        ungated = self.results["synapse_ablation"]["without_entropy_gate"]
        self.assertEqual((full["dice"], full["hd95_mm"]), (86.45, 12.34))
        self.assertEqual((ungated["dice"], ungated["hd95_mm"]), (85.34, 13.12))


if __name__ == "__main__":
    unittest.main()
