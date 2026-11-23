"""Kiểm thử Feedback cấp 1 với các trường hợp lỗi khác nhau.

Chạy: python -m unittest feedback.testing.test_feedback_level1 -v
"""

import json
import unittest

from feedback.feedback_generation import (
    LEVEL1_MESSAGES,
    SIGN_HANDSHAPE_TIPS,
    component_results_from_detection,
    generate_feedback,
)
from feedback.feedback_manager import FeedbackManager


def dev(feature, delta, unit, finger=None, reference=0.0, actual=None, z=2.5):
    d = {
        "feature": feature,
        "reference": reference,
        "actual": reference + delta if actual is None else actual,
        "delta": delta,
        "unit": unit,
        "z_score": z,
    }
    if finger:
        d["finger"] = finger
    return d


def detection(handshape=(), orientation=(), location=(), scores=None):
    scores = scores or {}
    groups = {"handshape": handshape, "orientation": orientation, "location": location}
    return {
        "components": {
            g: {"score": scores.get(g, 40.0 if devs else 95.0), "deviations": list(devs)}
            for g, devs in groups.items()
        },
        "score": 70.0,
    }


CORRECT = detection()
NO_HAND = {
    "score": None,
    "components": {g: {"score": None, "deviations": []} for g in ("handshape", "orientation", "location")},
    "errors": ["no_hand"],
}


class Level1MessageTests(unittest.TestCase):
    def test_each_component_has_level1_message(self):
        for comp in ("H", "L", "O"):
            fb = generate_feedback(comp, 1, None, target="A", version="V1")
            self.assertEqual(fb["message"], LEVEL1_MESSAGES["V1"][comp])
            self.assertEqual(fb["level"], 1)
            self.assertTrue(fb["hint"])

    def test_movement_not_applicable_in_v1(self):
        self.assertIsNone(generate_feedback("M", 1, None, version="V1"))
        fb = generate_feedback("M", 1, None, version="V2")
        self.assertEqual(fb["message"], "Bạn đang di chuyển bàn tay chưa đúng cách.")

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            generate_feedback("X", 1)
        with self.assertRaises(ValueError):
            generate_feedback("H", 4)
        with self.assertRaises(ValueError):
            generate_feedback("H", 1, version="V3")

    def test_output_is_json_serializable(self):
        info = {"score": 40.0, "deviations": [dev("index_pip_angle", -30.0, "degree", "index")]}
        json.dumps(generate_feedback("H", 1, info, target="D"), ensure_ascii=False)


class HandshapeHintTests(unittest.TestCase):
    def hint(self, d, target="A"):
        return generate_feedback("H", 1, {"score": 40.0, "deviations": [d]}, target=target)

    def test_joint_more_bent_than_reference_asks_to_straighten(self):
        fb = self.hint(dev("index_pip_angle", -30.0, "degree", "index", reference=170.0))
        self.assertEqual(fb["hint"], "Hãy duỗi ngón trỏ thẳng hơn.")
        self.assertEqual(fb["hint_source"], "deviation")

    def test_joint_straighter_than_reference_asks_to_bend(self):
        fb = self.hint(dev("ring_dip_angle", 25.0, "degree", "ring", reference=120.0))
        self.assertEqual(fb["hint"], "Hãy gập ngón áp út lại nhiều hơn.")

    def test_spread_angle(self):
        self.assertEqual(
            self.hint(dev("index_middle_angle", 15.0, "degree", "index_middle"))["hint"],
            "Hãy khép ngón trỏ và ngón giữa lại gần nhau hơn.",
        )
        self.assertEqual(
            self.hint(dev("index_middle_angle", -15.0, "degree", "index_middle"))["hint"],
            "Hãy tách ngón trỏ và ngón giữa ra xa nhau hơn.",
        )

    def test_wrist_distance(self):
        self.assertEqual(
            self.hint(dev("wrist_pinky_distance", -0.4, "palm_ratio", "pinky"))["hint"],
            "Hãy duỗi ngón út ra xa hơn.",
        )
        self.assertEqual(
            self.hint(dev("wrist_index_distance", 0.4, "palm_ratio", "index"))["hint"],
            "Hãy co ngón trỏ lại gần lòng bàn tay hơn.",
        )

    def test_fingertip_pair_distance(self):
        self.assertEqual(
            self.hint(dev("thumb_index_distance", 0.3, "palm_ratio", "thumb_index"))["hint"],
            "Hãy đưa đầu ngón cái và đầu ngón trỏ lại gần nhau hơn.",
        )

    def test_tiny_deviation_falls_back_to_sign_library(self):
        fb = self.hint(dev("index_pip_angle", -2.0, "degree", "index"), target="d")
        self.assertEqual(fb["hint"], SIGN_HANDSHAPE_TIPS["D"])
        self.assertEqual(fb["hint_source"], "sign_library")

    def test_unknown_sign_without_details_uses_generic(self):
        fb = generate_feedback("H", 1, {"score": 50.0, "deviations": []}, target="HELLO")
        self.assertEqual(fb["hint_source"], "generic")

    def test_all_static_letters_have_tip(self):
        static = set("ABCDEFGHIKLMNOPQRSTUVWXY")
        self.assertEqual(set(SIGN_HANDSHAPE_TIPS), static)


class OrientationAndLocationHintTests(unittest.TestCase):
    def hint(self, comp, d):
        return generate_feedback(comp, 1, {"score": 40.0, "deviations": [d]}, target="A")["hint"]

    def test_roll(self):
        self.assertIn("tay phải", self.hint("O", dev("palm_roll_sin", 0.4, "unitless")))
        self.assertIn("tay trái", self.hint("O", dev("palm_roll_sin", -0.4, "unitless")))
        self.assertIn("thẳng đứng", self.hint("O", dev("palm_roll_cos", -0.3, "unitless")))

    def test_flipped_palm(self):
        h = self.hint("O", dev("palm_horizontal_axis_x", -1.6, "unitless", reference=0.8, actual=-0.8))
        self.assertIn("lật cổ tay", h)

    def test_location(self):
        self.assertEqual(self.hint("L", dev("wrist_y", 0.2, "normalized_coordinate")), "Hãy nâng cổ tay lên cao hơn.")
        self.assertIn("tay phải", self.hint("L", dev("wrist_x", 0.2, "normalized_coordinate")))


class DetectionMappingTests(unittest.TestCase):
    def test_no_hand_is_not_evaluated(self):
        self.assertEqual(component_results_from_detection(NO_HAND), {})
        self.assertEqual(component_results_from_detection(None), {})

    def test_deviation_means_wrong(self):
        det = detection(handshape=[dev("index_pip_angle", -30.0, "degree", "index")])
        self.assertEqual(component_results_from_detection(det), {"H": False, "O": True, "L": True})

    def test_low_score_threshold(self):
        det = detection(scores={"orientation": 50.0})
        self.assertFalse(component_results_from_detection(det, threshold_score=60)["O"])


class ManagerTests(unittest.TestCase):
    H_WRONG = detection(handshape=[dev("index_pip_angle", -30.0, "degree", "index")])

    def test_no_hand_does_not_touch_history(self):
        m = FeedbackManager()
        out = m.process_attempt("p01", "A", NO_HAND)
        self.assertEqual(out["status"], "not_evaluated")
        self.assertEqual(m.store.get("p01", "A"), {})

    def test_correct_attempt_has_no_feedback(self):
        out = FeedbackManager().process_attempt("p01", "A", CORRECT)
        self.assertEqual(out["status"], "correct")
        self.assertEqual(out["feedbacks"], [])

    def test_first_error_gets_level1(self):
        out = FeedbackManager().process_attempt("p01", "D", self.H_WRONG)
        self.assertEqual(out["status"], "has_errors")
        self.assertEqual(len(out["feedbacks"]), 1)
        fb = out["feedbacks"][0]
        self.assertEqual((fb["component"], fb["level"]), ("H", 1))
        self.assertEqual(fb["hint"], "Hãy duỗi ngón trỏ thẳng hơn.")

    def test_fixed_mode_stays_level1_while_adaptive_escalates(self):
        fixed, adaptive = FeedbackManager(), FeedbackManager()
        for _ in range(3):
            f = fixed.process_attempt("p01", "A", self.H_WRONG, mode="fixed")
            a = adaptive.process_attempt("p01", "A", self.H_WRONG, mode="adaptive")
        self.assertEqual(f["feedbacks"][0]["level"], 1)
        self.assertGreater(a["feedbacks"][0]["level"], 1)

    def test_history_is_separate_per_sign(self):
        m = FeedbackManager()
        for _ in range(3):
            m.process_attempt("p01", "A", self.H_WRONG)
        out = m.process_attempt("p01", "B", self.H_WRONG)
        self.assertEqual(out["feedbacks"][0]["level"], 1)

    def test_multiple_errors_sorted_by_severity_and_limited(self):
        det = detection(
            handshape=[dev("index_pip_angle", -30.0, "degree", "index")],
            orientation=[dev("palm_roll_sin", 0.4, "unitless")],
            scores={"handshape": 60.0, "orientation": 20.0},
        )
        out = FeedbackManager().process_attempt("p01", "A", det)
        self.assertEqual([f["component"] for f in out["feedbacks"]], ["O", "H"])
        out = FeedbackManager().process_attempt("p01", "A", det, max_feedbacks=1)
        self.assertEqual(len(out["feedbacks"]), 1)


if __name__ == "__main__":
    unittest.main()
