import json
import unittest
from pathlib import Path

import numpy as np

from errors.error_detection_4params import FEATURE_COUNT, build_templates, main
from features import feature_extraction as fe


class ErrorDetectionTests(unittest.TestCase):
    def setUp(self):
        fixture_path = (
            Path(__file__).parents[2] / "features" / "testing" / "fixtures" / "A_example.json"
        )
        self.payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        self.actual_features = fe.extract_static_phonological_features(
            self.payload["landmarks"],
            self.payload["image_width"],
            self.payload["image_height"],
        )
        self.assertEqual(self.actual_features.shape, (FEATURE_COUNT,))

        # Synthetic reference data for exercising detection; replace with real
        # labeled reference samples before interpreting the result as ASL feedback.
        self.base = np.array(
            [
                0.86, 0.73, 0.01, 0.11, 0.06, 0.09, 0.08,
                0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.30,
                0.33, 0.36, 0.39, 0.42, 0.45, 0.48, 0.51,
                0.54, 0.57, 0.60, 0.63, 0.66, 0.69, 0.72,
                0.75, 0.78, 0.81, 0.84, 0.87, 0.90, 0.93,
                0.96, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
                0.85, 0.90, 0.95, 0.12, 0.18, 0.24, 0.30,
                0.36, 0.42,
            ],
            dtype=float,
        )
        self.assertEqual(self.base.shape, (FEATURE_COUNT,))

        rng = np.random.default_rng(42)
        reference_samples = self.base + rng.normal(
            loc=0.0, scale=0.008, size=(20, FEATURE_COUNT)
        )
        labels = np.array(["A"] * len(reference_samples))
        self.templates = build_templates(reference_samples, labels)

    def test_a_example_landmarks_produce_json_error_detection_output(self):
        result = main(self.actual_features, self.payload["target_label"], self.templates)

        self.assertEqual(
            set(result["components"]), {"handshape", "orientation", "location"}
        )
        self.assertTrue(all(
            len(component["deviations"]) <= 1
            for component in result["components"].values()
        ))
        serialized = json.dumps(result, ensure_ascii=False, indent=2)
        print(serialized)


if __name__ == "__main__":
    unittest.main()
