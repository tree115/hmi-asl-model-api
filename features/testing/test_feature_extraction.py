import json
import unittest
from pathlib import Path

import numpy as np

from features import feature_extraction as fe


class StaticPhonologicalFeatureTests(unittest.TestCase):

    def setUp(self):
        fixture_path = Path(__file__).parent / "fixtures" / "A_example.json"
        self.payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        self.landmarks = np.asarray(self.payload["landmarks"], dtype=float)
        self.width = self.payload["image_width"]
        self.height = self.payload["image_height"]

    def test_extracts_features_in_detector_order(self):
        features = fe.extract_static_phonological_features(
            self.landmarks, self.width, self.height
        )

        print("\n========== FEATURE EXTRACTION ==========")
        print(f"Target: {self.payload.get('target_label', 'A')}")
        print(f"Image size: {self.width} x {self.height}")
        print(f"Number of landmarks: {len(self.landmarks)}")
        print(f"Feature shape: {features.shape}")
        print("\nFeature values:")
        for i, value in enumerate(features):
            print(f"Feature [{i:02d}]: {value:.6f}")

        self.assertEqual(features.shape, (58,))
        self.assertTrue(np.isfinite(features).all())

        xy = self.landmarks[:, :2]
        frame = (xy * np.array([self.width, self.height], dtype=float))[
            np.newaxis, :, :
        ]

        expected = np.concatenate(
            [
                fe.calc_angles_per_frame(frame)[0],
                fe.calc_distance_per_frame(frame)[0],
                (xy[[4, 8, 12, 16, 20]] - xy[0]).reshape(-1),
                fe.calc_hand_orientation_2d(frame)[0],
                xy[0],
            ]
        )

        np.testing.assert_allclose(features, expected)

        print("\nFeature extraction: PASSED")

    def test_uses_left_hand_orientation_when_requested(self):
        right_features = fe.extract_static_phonological_features(
            self.landmarks, self.width, self.height
        )

        left_features = fe.extract_static_phonological_features(
            self.landmarks, self.width, self.height, is_left=True
        )

        print("\n========== HAND ORIENTATION ==========")
        print(f"Right hand orientation: {right_features[54]:.6f}")
        print(f"Left hand orientation:  {left_features[54]:.6f}")
        print(f"Difference: {right_features[54] - left_features[54]:.6f}")

        np.testing.assert_allclose(right_features[:54], left_features[:54])
        self.assertAlmostEqual(right_features[54], -left_features[54])
        np.testing.assert_allclose(right_features[55:], left_features[55:])

        print("Left/right orientation: PASSED")

    def test_rejects_invalid_landmarks_or_frame_dimensions(self):
        print("\n========== INVALID INPUT TESTS ==========")

        with self.assertRaises(ValueError):
            fe.extract_static_phonological_features(
                self.landmarks[:20], self.width, self.height
            )
        print("20 landmarks: Rejected correctly")

        with self.assertRaises(ValueError):
            fe.extract_static_phonological_features(self.landmarks, 0, self.height)
        print("Invalid image width: Rejected correctly")

        with self.assertRaises(ValueError):
            fe.extract_static_phonological_features(
                np.full_like(self.landmarks, np.nan), self.width, self.height
            )
        print("NaN landmarks: Rejected correctly")


if __name__ == "__main__":
    unittest.main(verbosity=2)
