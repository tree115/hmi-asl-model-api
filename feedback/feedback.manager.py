from feedback import feedback_adaptation
from feedback import feedback_generation

# xử lí trường hợp người dùng đúng thì làm gì
# ghép kết quả và giao tiếp với I/O
# có thể dùng để đưa input vào và lưu lại lịch sử sau khi đưa ra feedback
"""
Một lượt có thể sai nhiều thành phần (ví dụ H và M cùng sai). 
Manager là nơi quyết định hiển thị hết hay ưu tiên một thành phần (chẳng hạn chỉ feedback thành phần sai nặng nhất để người học không bị quá tải). 
Đây là quyết định sư phạm, nên đặt ở manager chứ không đặt ở hai module kia
"""


from error_history import update_error_history


class FeedbackManager:
    def __init__(self, history_store, generator=generate_feedback):
        self.store = history_store  # DB hoặc JSON, chỉ cần get/save
        self.generate = generator  # tiêm vào được -> dễ thay, dễ test

    def process_attempt(
        self,
        learner_id,
        sign_id,
        component_results,
        initial_levels=None,
        error_infos=None,
    ):
        error_infos = error_infos or {}
        history = self.store.get(learner_id)

        out = update_error_history(history, component_results, initial_levels)

        feedbacks = {}
        for comp, level in out["feedback_levels"].items():
            feedbacks[comp] = self.generate(comp, level, error_infos.get(comp))

        self.store.save(learner_id, out["updated_history"])

        return {
            "results": out["results"],  # đúng/sai từng thành phần
            "feedbacks": feedbacks,  # chỉ các thành phần sai
            "needs_review": out["needs_review"],
        }
