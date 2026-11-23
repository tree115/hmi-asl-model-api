"""Kiểm thử API /errors/detect và /feedback bằng mẫu chuẩn tổng hợp có kiểm soát.

Chạy: python -m unittest api.testing.test_feedback_api -v

Mẫu chuẩn được dựng từ chính fixture chữ A, nên:
- target A  -> không lệch -> "correct"
- target B  -> giống A nhưng góc PIP ngón trỏ lớn hơn 54° -> lỗi H, gợi ý duỗi ngón trỏ
Như vậy test chỉ kiểm tra luồng API, không phụ thuộc chất lượng mẫu chuẩn thật.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from api.main import app
from errors.templates import save_templates
from features import feature_extraction as fe

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = json.loads(
    (ROOT / "features/testing/fixtures/A_example.json").read_text(encoding="utf-8")
)
INDEX_PIP = 2  # vị trí index_pip_angle trong vector 46 đặc trưng


def fixture_features():
    normalized = fe.preprocess_hand(
        FIXTURE["landmarks"], FIXTURE["image_width"], FIXTURE["image_height"]
    )
    return fe.extract_static_phonological_features(normalized)


class FeedbackApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        x = fixture_features()
        std = np.full_like(x, 0.05)
        b_mean = x.copy()
        b_mean[INDEX_PIP] += 0.3  # 0.3 * 180 = 54°
        path = Path(cls.tmp.name) / "templates.npz"
        save_templates(path, {"A": (x, std), "B": (b_mean, std)})
        cls.old_env = os.environ.get("AI_ERROR_TEMPLATE_DATA")
        os.environ["AI_ERROR_TEMPLATE_DATA"] = str(path)
        cls.client_cm = TestClient(app)
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_cm.__exit__(None, None, None)
        if cls.old_env is None:
            os.environ.pop("AI_ERROR_TEMPLATE_DATA", None)
        else:
            os.environ["AI_ERROR_TEMPLATE_DATA"] = cls.old_env
        cls.tmp.cleanup()

    def post_feedback(self, target="B", **extra):
        body = dict(FIXTURE, target_label=target, **extra)
        return self.client.post("/api/v1/ai/feedback", json=body)

    def test_capabilities_advertise_feedback(self):
        caps = self.client.get("/api/v1/ai/capabilities").json()
        self.assertEqual(caps["feedback_endpoint"], "/api/v1/ai/feedback")
        self.assertTrue(caps["adaptive_feedback"])
        self.assertTrue(caps["dynamic_word_recognition"])

    def test_word_endpoint_present_after_merge(self):
        self.assertEqual(self.client.post("/api/v1/ai/predict/word", json={}).status_code, 422)

    def test_detect_reports_status_and_uppercases_target(self):
        r = self.client.post("/api/v1/ai/errors/detect", json=dict(FIXTURE, target_label="a"))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["target_label"], "A")
        self.assertEqual(r.json()["status"], "evaluated")

    def test_correct_attempt(self):
        r = self.post_feedback("A")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "correct")
        self.assertEqual(body["feedbacks"], [])
        self.assertEqual(body["history"]["H"]["last_result"], "correct")

    def test_wrong_attempt_gives_level1_hint(self):
        body = self.post_feedback("b").json()
        self.assertEqual(body["status"], "has_errors")
        self.assertEqual(body["target_label"], "B")
        fb = body["feedbacks"][0]
        self.assertEqual((fb["component"], fb["level"]), ("H", 1))
        self.assertEqual(fb["message"], "Hình dạng bàn tay của bạn chưa giống mẫu.")
        self.assertEqual(fb["hint"], "Hãy duỗi ngón trỏ thẳng hơn.")

    def test_history_round_trip_escalates_only_in_adaptive(self):
        for mode, expected in (("adaptive", [1, 1, 2]), ("fixed", [1, 1, 1])):
            history, levels = {}, []
            for _ in range(3):
                body = self.post_feedback("B", mode=mode, history=history).json()
                levels.append(body["feedbacks"][0]["level"])
                history = body["history"]
            self.assertEqual(levels, expected, mode)

    def test_not_evaluated_cases_keep_history(self):
        history = self.post_feedback("B").json()["history"]
        for target, extra, reason in (
            ("J", {}, "unsupported_target"),
            ("B", {"landmarks": None}, "no_hand"),
        ):
            body = self.post_feedback(target, history=history, **extra).json()
            self.assertEqual(body["status"], "not_evaluated")
            self.assertEqual(body["reason"], reason)
            self.assertEqual(body["history"], history)

    def test_invalid_history_rejected(self):
        r = self.post_feedback("B", history={"H": {"level": 5}})
        self.assertEqual(r.status_code, 422)
        r = self.post_feedback("B", history={"X": {}})
        self.assertEqual(r.status_code, 422)

    def test_missing_template_returns_503(self):
        saved = os.environ.pop("AI_ERROR_TEMPLATE_DATA")
        try:
            r = self.post_feedback("B")
            self.assertEqual(r.status_code, 503)
            self.assertEqual(r.json()["detail"]["code"], "template_data_unavailable")
        finally:
            os.environ["AI_ERROR_TEMPLATE_DATA"] = saved


if __name__ == "__main__":
    unittest.main()
