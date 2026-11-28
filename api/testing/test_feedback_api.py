"""Kiểm thử API /errors/detect và /feedback bằng mẫu chuẩn tổng hợp có kiểm soát.

Chạy: python -m unittest api.testing.test_feedback_api -v

Mẫu chuẩn được dựng từ chính fixture chữ A (classifier nhận ra là A), nên:
- target A -> classifier đúng -> "correct"
- target B -> classifier ra A (sai) + mẫu chuẩn B giống A nhưng góc PIP ngón trỏ
  lớn hơn 54° -> lỗi H từ so sánh đặc trưng, gợi ý duỗi ngón trỏ
- target C -> classifier sai, không có mẫu chuẩn C -> lỗi H chỉ từ classifier
Test kiểm tra luồng API, không phụ thuộc chất lượng mẫu chuẩn thật.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.main import app
from errors.templates import save_templates
from features import feature_extraction as fe
from integration.backend_proxy import create_ai_router

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


def attempt_body(target, frames=15, landmarks="fixture", **extra):
    lms = FIXTURE["landmarks"] if landmarks == "fixture" else landmarks
    return {
        "client_attempt_id": "test-1",
        "target_label": target,
        "image_width": FIXTURE["image_width"],
        "image_height": FIXTURE["image_height"],
        "frames": [
            {"handedness": "Right", "timestamp_ms": i * 80, "landmarks": lms}
            for i in range(frames)
        ],
        **extra,
    }


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

    def post(self, target="B", **extra):
        r = self.client.post("/api/v1/ai/feedback", json=attempt_body(target, **extra))
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    # ---- capabilities / routes ----
    def test_capabilities_advertise_feedback(self):
        caps = self.client.get("/api/v1/ai/capabilities").json()
        self.assertEqual(caps["feedback_endpoint"], "/api/v1/ai/feedback")
        self.assertTrue(caps["adaptive_feedback"])
        self.assertTrue(caps["dynamic_word_recognition"])

    def test_word_endpoint_present_after_merge(self):
        self.assertEqual(self.client.post("/api/v1/ai/predict/word", json={}).status_code, 422)

    def test_backend_proxy_exposes_all_ai_routes(self):
        backend = FastAPI()
        backend.include_router(create_ai_router())
        paths = backend.openapi()["paths"]
        for path in (
            "/api/v1/ai/feedback",
            "/api/v1/ai/errors/detect",
            "/api/v1/ai/predict/attempt",
            "/api/v1/ai/predict/word",
            "/api/v1/ai/words/capabilities",
        ):
            self.assertIn(path, paths)

    # ---- /errors/detect (một khung hình) ----
    def test_detect_reports_status_and_uppercases_target(self):
        r = self.client.post("/api/v1/ai/errors/detect", json=dict(FIXTURE, target_label="a"))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["target_label"], "A")
        self.assertEqual(r.json()["status"], "evaluated")

    # ---- /feedback ----
    def test_correct_attempt(self):
        body = self.post("a")
        self.assertEqual(body["status"], "correct")
        self.assertTrue(body["is_correct"])
        self.assertEqual(body["recognition"]["predicted_label"], "A")
        self.assertEqual(body["feedbacks"], [])
        self.assertEqual(body["errors"], [])
        self.assertIsNone(body["feedback_level"])
        self.assertEqual(body["history"]["H"]["last_result"], "correct")

    def test_wrong_attempt_with_feature_diagnosis(self):
        body = self.post("b")
        self.assertEqual(body["status"], "has_errors")
        self.assertFalse(body["is_correct"])
        self.assertEqual(body["diagnosis_source"], "feature_templates")
        fb = body["feedbacks"][0]
        self.assertEqual((fb["component"], fb["error_type"], fb["level"]), ("H", "handshape", 1))
        self.assertEqual(fb["hint"], "Hãy duỗi ngón trỏ thẳng hơn.")
        self.assertEqual(body["feedback_level"], 1)
        err = body["errors"][0]
        self.assertEqual(err["type"], "handshape")
        self.assertIn(err["severity"], ("low", "medium", "high"))

    def test_wrong_attempt_without_template_uses_classifier(self):
        body = self.post("C")
        self.assertEqual(body["diagnosis_source"], "classifier_only")
        self.assertEqual(body["results"], {"H": "wrong"})
        self.assertEqual(body["feedbacks"][0]["hint_source"], "sign_library")

    def test_history_round_trip_escalates_only_in_adaptive(self):
        for mode, expected in (("adaptive", [1, 1, 2]), ("fixed", [1, 1, 1])):
            history, levels = {}, []
            for _ in range(3):
                body = self.post("B", mode=mode, history=history)
                levels.append(body["feedback_level"])
                history = body["history"]
            self.assertEqual(levels, expected, mode)

    def test_not_evaluated_cases_keep_history(self):
        history = self.post("B")["history"]
        for target, extra, reason in (
            ("HELLO", {}, "unsupported_target"),
            ("B", {"landmarks": None}, "no_hand"),
        ):
            body = self.post(target, history=history, **extra)
            self.assertEqual(body["status"], "not_evaluated")
            self.assertEqual(body["reason"], reason)
            self.assertIsNone(body["is_correct"])
            self.assertEqual(body["errors"], [])
            self.assertEqual(body["history"], history)

    def test_missing_template_falls_back_to_classifier(self):
        saved = os.environ["AI_ERROR_TEMPLATE_DATA"]
        os.environ["AI_ERROR_TEMPLATE_DATA"] = str(Path(self.tmp.name) / "missing.npz")
        try:
            body = self.post("B")
            self.assertEqual(body["diagnosis_source"], "classifier_only")
            self.assertEqual(body["status"], "has_errors")
            self.assertEqual(self.post("A")["status"], "correct")
        finally:
            os.environ["AI_ERROR_TEMPLATE_DATA"] = saved

    # ---- lịch sử lỗi (tuần 6) ----
    def test_feedback_returns_history_summary(self):
        history = {}
        for _ in range(2):
            body = self.post("B", history=history)
            history = body["history"]
        rows = {r["component"]: r for r in body["history_summary"]}
        self.assertEqual(list(rows), ["H", "L", "O", "M"])
        self.assertEqual(rows["H"]["progress"], "not_improving")
        self.assertEqual(rows["H"]["recent"], ["wrong", "wrong"])
        self.assertEqual(rows["H"]["feedbacks_shown"], 2)
        self.assertEqual(rows["M"]["progress_label"], "Chưa có dữ liệu")

        body = self.post("A", history=history)
        h = {r["component"]: r for r in body["history_summary"]}["H"]
        self.assertEqual(h["progress"], "improving")
        self.assertEqual(h["corrections"], 1)
        self.assertEqual(body["history"]["H"]["fixed_after_level"], {"1": 1})

    def test_history_summary_endpoint(self):
        history = self.post("B")["history"]
        r = self.client.post("/api/v1/ai/feedback/history-summary", json={"history": history})
        self.assertEqual(r.status_code, 200)
        rows = {row["component"]: row for row in r.json()["rows"]}
        self.assertEqual(rows["H"]["progress"], "new_error")
        empty = self.client.post("/api/v1/ai/feedback/history-summary", json={}).json()
        self.assertTrue(all(row["progress"] == "no_data" for row in empty["rows"]))

    def test_old_history_format_accepted(self):
        old = {"H": {"level": 1, "wrong_streak": 1, "correct_streak": 0, "total_wrong": 1,
                     "total_correct": 0, "last_result": "wrong", "needs_review": False}}
        body = self.post("B", history=old)
        self.assertEqual(body["history"]["H"]["total_wrong"], 2)

    def test_invalid_requests_rejected(self):
        for body in (
            attempt_body("B", history={"H": {"level": 5}}),
            attempt_body("B", history={"X": {}}),
            attempt_body("B", mode="other"),
            attempt_body("B", frames=2),
        ):
            self.assertEqual(self.client.post("/api/v1/ai/feedback", json=body).status_code, 422)


if __name__ == "__main__":
    unittest.main()
