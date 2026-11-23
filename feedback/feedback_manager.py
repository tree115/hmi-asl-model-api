"""FeedbackManager: điều phối error detection -> adaptation -> generation -> FE.

- Lịch sử được theo dõi theo cặp (người học, ký hiệu mục tiêu), mỗi thành
  phần H/L/O/M có bộ đếm riêng (báo cáo mục 4.2.3).
- mode="fixed": luôn trả Cấp 1 (điều kiện baseline); lịch sử vẫn được cập nhật
  để log, nhưng không dùng để chọn cấp.
- mode="adaptive": cấp lấy từ feedback_adaptation.
- Lượt không đánh giá được (no_hand...) không cập nhật lịch sử.

Một lượt có thể sai nhiều thành phần. Việc hiển thị hết hay chỉ ưu tiên một
thành phần là quyết định sư phạm, đặt ở đây (tham số max_feedbacks).

Hai cách dùng:
- evaluate_attempt(history, ...): hàm thuần, nhận lịch sử cũ và trả lịch sử
  mới. API /api/v1/ai/feedback dùng hàm này; BE lưu lịch sử.
- FeedbackManager: bọc evaluate_attempt với một kho lưu (mặc định trong RAM),
  tiện cho demo/test.
"""

from feedback.feedback_adaptation import update_error_history
from feedback.feedback_generation import (
    component_results_from_detection,
    error_infos_from_detection,
    generate_feedback,
)

MODES = ("fixed", "adaptive")


def evaluate_attempt(
    history,
    sign_id,
    detection,
    mode="adaptive",
    version="V1",
    initial_levels=None,
    max_feedbacks=None,
    generator=generate_feedback,
    component_results=None,
):
    """Xử lý một lượt thử. Không sửa history truyền vào.

    component_results: truyền vào để ghi đè kết luận đúng/sai suy từ detection
    (ví dụ khi classifier đã xác nhận đúng chữ mục tiêu).

    Trả về dict gồm status, results, feedbacks, needs_review và history
    (lịch sử mới để lưu lại cho lượt sau).
    """
    if mode not in MODES:
        raise ValueError(f"mode phải là một trong {MODES}, nhận {mode!r}")
    history = history or {}

    if component_results is None:
        component_results = component_results_from_detection(detection)
    if not component_results:
        reasons = (detection or {}).get("errors") or ["not_evaluated"]
        return {
            "status": "not_evaluated",
            "reason": reasons[0],
            "results": {},
            "feedbacks": [],
            "needs_review": [],
            "history": history,
        }

    out = update_error_history(history, component_results, initial_levels)

    error_infos = error_infos_from_detection(detection)
    feedbacks = []
    for component, level in out["feedback_levels"].items():
        chosen_level = 1 if mode == "fixed" else level
        item = generator(
            component,
            chosen_level,
            error_infos.get(component),
            target=sign_id,
            version=version,
        )
        if item is not None:
            feedbacks.append(item)

    # Ưu tiên thành phần có điểm thấp nhất (sai nặng nhất) lên đầu.
    def _score(item):
        info = error_infos.get(item["component"]) or {}
        score = info.get("score")
        return score if score is not None else 100.0

    feedbacks.sort(key=_score)
    if max_feedbacks is not None:
        feedbacks = feedbacks[:max_feedbacks]

    all_correct = all(component_results.values())
    return {
        "status": "correct" if all_correct else "has_errors",
        "reason": None,
        "results": out["results"],  # đúng/sai từng thành phần
        "feedbacks": feedbacks,  # chỉ các thành phần sai
        "needs_review": [] if mode == "fixed" else out["needs_review"],
        "history": out["updated_history"],
    }


class InMemoryHistoryStore:
    """Lưu lịch sử trong bộ nhớ, đúng với giới hạn 'chỉ tính trong phiên'."""

    def __init__(self):
        self._data = {}

    def get(self, learner_id, sign_id):
        return self._data.get((learner_id, sign_id), {})

    def save(self, learner_id, sign_id, history):
        self._data[(learner_id, sign_id)] = history


class FeedbackManager:
    def __init__(self, history_store=None, generator=generate_feedback):
        self.store = history_store or InMemoryHistoryStore()  # cần get/save
        self.generate = generator  # tiêm vào được -> dễ thay, dễ test

    def process_attempt(
        self,
        learner_id,
        sign_id,
        detection,
        mode="adaptive",
        version="V1",
        initial_levels=None,
        max_feedbacks=None,
    ):
        """detection: JSON trả về từ POST /api/v1/ai/errors/detect."""
        out = evaluate_attempt(
            self.store.get(learner_id, sign_id),
            sign_id,
            detection,
            mode=mode,
            version=version,
            initial_levels=initial_levels,
            max_feedbacks=max_feedbacks,
            generator=self.generate,
        )
        if out["status"] != "not_evaluated":
            self.store.save(learner_id, sign_id, out["history"])
        return out
