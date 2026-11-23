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

        self.assertEqual(features.shape, (46,))
        self.assertTrue(np.isfinite(features).all())

        xy = self.landmarks[:, :2]
        frame = (xy * np.array([self.width, self.height], dtype=float))[
            np.newaxis, :, :
        ]

        expected = np.concatenate(
            [
                fe.calc_angles_per_frame(frame)[0],
                fe.calc_distance_per_frame(frame)[0],
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
        print(f"Right hand orientation: {right_features[40:44]}")
        print(f"Left hand orientation:  {left_features[40:44]}")

        np.testing.assert_allclose(right_features, left_features)

        print("Left/right orientation: PASSED")

    def test_static_feature_matrix_uses_the_same_46_feature_order(self):
        frame = self.landmarks[np.newaxis, :, :2]
        features = fe.get_static_features({"r_hand": frame}, "r_hand")

        self.assertEqual(features.shape, (1, 46))
        expected = np.concatenate(
            [
                fe.calc_angles_per_frame(frame),
                fe.calc_distance_per_frame(frame),
                fe.calc_hand_orientation_2d(frame),
                frame[:, 0, :],
            ],
            axis=1,
        )
        np.testing.assert_allclose(features, expected)

    def test_preprocess_then_extract_returns_the_46_feature_vector(self):
        normalized_landmarks = fe.preprocess_hand(
            self.landmarks, self.width, self.height
        )
        features = fe.extract_static_phonological_features(normalized_landmarks)
        raw_features = fe.extract_static_phonological_features(
            self.landmarks, self.width, self.height
        )

        self.assertEqual(normalized_landmarks.shape, (21, 2))
        self.assertEqual(features.shape, (46,))
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_allclose(features[:44], raw_features[:44], atol=1e-6)

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
