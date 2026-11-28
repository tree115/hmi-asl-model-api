"""Kiểm thử quản lý lịch sử lỗi và đánh giá tiến bộ (tuần 6).

Chạy: python -m unittest feedback.testing.test_history_summary -v
"""

import unittest

from feedback.feedback_manager import evaluate_attempt
from feedback.history_summary import (
    PROGRESS_LABELS,
    RECENT_LIMIT,
    evaluate_progress,
    record_attempt,
    summarize_history,
)


def detection(handshape_wrong=False, orientation_wrong=False):
    def comp(wrong):
        devs = [{"feature": "index_pip_angle", "finger": "index", "delta": -30.0,
                 "unit": "degree", "z_score": 3.5}] if wrong else []
        return {"score": 40.0 if wrong else 95.0, "deviations": devs}

    return {
        "components": {
            "handshape": comp(handshape_wrong),
            "orientation": comp(orientation_wrong),
            "location": comp(False),
        },
        "score": 70.0,
    }


class ProgressRuleTests(unittest.TestCase):
    """Bảng ví dụ của Tuyền: H chưa cải thiện, L lỗi mới, O đang tiến bộ, M chưa có dữ liệu."""

    def test_rules(self):
        self.assertEqual(evaluate_progress({}), "no_data")
        self.assertEqual(
            evaluate_progress({"total_wrong": 1, "total_correct": 0, "last_result": "wrong"}),
            "new_error",
        )
        self.assertEqual(
            evaluate_progress({"total_wrong": 4, "total_correct": 0, "last_result": "wrong"}),
            "not_improving",
        )
        self.assertEqual(
            evaluate_progress({"total_wrong": 2, "total_correct": 2, "last_result": "correct"}),
            "improving",
        )
        self.assertEqual(
            evaluate_progress({"total_wrong": 0, "total_correct": 3, "last_result": "correct"}),
            "stable",
        )

    def test_summary_has_all_components_in_order(self):
        rows = summarize_history({"H": {"level": 3, "wrong_streak": 2, "total_wrong": 5,
                                        "last_result": "wrong"}})
        self.assertEqual([r["component"] for r in rows], ["H", "L", "O", "M"])
        self.assertEqual(rows[0]["progress_label"], PROGRESS_LABELS["not_improving"])
        self.assertEqual(rows[3]["progress"], "no_data")
        self.assertEqual(rows[3]["name"], "Chuyển động")


class RecordAttemptTests(unittest.TestCase):
    def test_counts_feedback_and_correction_by_level(self):
        h = record_attempt({}, {"H": False}, {"H": 1})
        h = record_attempt(h, {"H": False}, {"H": 2})
        h = record_attempt(h, {"H": True}, {})
        s = h["H"]
        self.assertEqual(s["attempts"], 3)
        self.assertEqual(s["recent"], ["wrong", "wrong", "correct"])
        self.assertEqual(s["feedbacks_shown"], 2)
        self.assertEqual(s["last_feedback_level"], 2)
        self.assertEqual(s["corrections"], 1)
        self.assertEqual(s["fixed_after_level"], {"2": 1})

    def test_correct_without_prior_error_is_not_a_correction(self):
        h = record_attempt({}, {"H": True}, {})
        self.assertEqual(h["H"]["corrections"], 0)

    def test_recent_is_capped(self):
        h = {}
        for _ in range(RECENT_LIMIT + 5):
            h = record_attempt(h, {"H": True}, {})
        self.assertEqual(len(h["H"]["recent"]), RECENT_LIMIT)

    def test_input_not_mutated(self):
        original = {"H": {"attempts": 1, "recent": ["wrong"]}}
        record_attempt(original, {"H": True}, {})
        self.assertEqual(original, {"H": {"attempts": 1, "recent": ["wrong"]}})


class ManagerIntegrationTests(unittest.TestCase):
    def test_example_scenario_matches_table(self):
        history = {}
        # H sai liên tiếp, O sai rồi đúng 2 lần.
        steps = [(True, True), (True, False), (True, False)]
        for h_wrong, o_wrong in steps:
            out = evaluate_attempt(history, "A", detection(h_wrong, o_wrong))
            history = out["history"]
        rows = {r["component"]: r for r in out["history_summary"]}
        self.assertEqual(rows["H"]["progress"], "not_improving")
        self.assertEqual(rows["O"]["progress"], "improving")
        self.assertEqual(rows["L"]["progress"], "stable")
        self.assertEqual(rows["M"]["progress"], "no_data")
        self.assertEqual(rows["O"]["corrections"], 1)
        self.assertEqual(rows["H"]["feedbacks_shown"], 3)

    def test_not_evaluated_keeps_history_and_returns_summary(self):
        history = evaluate_attempt({}, "A", detection(True))["history"]
        out = evaluate_attempt(history, "A", {"errors": ["no_hand"]})
        self.assertEqual(out["history"], history)
        self.assertEqual(len(out["history_summary"]), 4)

    def test_old_history_without_tracking_fields_still_works(self):
        old = {"H": {"level": 1, "wrong_streak": 1, "correct_streak": 0, "total_wrong": 1,
                     "total_correct": 0, "last_result": "wrong", "needs_review": False}}
        out = evaluate_attempt(old, "A", detection(False))
        self.assertEqual(out["history"]["H"]["attempts"], 1)
        self.assertEqual(out["history"]["H"]["recent"], ["correct"])


if __name__ == "__main__":
    unittest.main()
