import json
import unittest
from pathlib import Path

import numpy as np

from errors.error_detection_static import FEATURE_COUNT, GROUPS, main
from errors.handshape import HANDSHAPE_FEATURES
from errors.orientation import ORIENTATION_FEATURES
from features import feature_extraction as fe


class ErrorDetectionTests(unittest.TestCase):
    def setUp(self):
        self.assertEqual(len(HANDSHAPE_FEATURES), 40)
        self.assertEqual(len(ORIENTATION_FEATURES), 4)
        np.testing.assert_array_equal(GROUPS["handshape"], np.arange(0, 40))
        np.testing.assert_array_equal(GROUPS["orientation"], np.arange(40, 44))
        np.testing.assert_array_equal(GROUPS["location"], np.arange(44, 46))

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
        self.base = np.linspace(0.1, 0.9, FEATURE_COUNT)
        self.assertEqual(self.base.shape, (FEATURE_COUNT,))

        self.templates = {
            "A": (self.base, np.full(FEATURE_COUNT, 0.008, dtype=float))
        }

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
