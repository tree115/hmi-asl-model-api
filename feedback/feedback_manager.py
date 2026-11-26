from feedback import feedback_adaptation
from feedback import feedback_generation


# vấn đề: chưa có lịch sử lỗi người dùng -> sử lại code liên quan
# tùy vào hàm feedback generator trả về cái gì thì mình sửa lại output của cái đó
# kết nối với Fe
# thêm: xử lí quá cấp
class FeedbackManager:
    """
    Điều phối quy trình tạo feedback cho một lượt luyện tập.

    Workflow:
        1. Nhận JSON lỗi từ Error Detection.
        2. Lấy lịch sử luyện tập của người dùng.
        3. Gửi lỗi + lịch sử cho Feedback Adaptation.
        4. Nhận cấp độ feedback từ Adaptation.
        5. Gửi thông tin lỗi + cấp độ cho Feedback Generation.
        6. Tổng hợp kết quả và trả về FE.

    Lưu ý:
        - Adaptation chịu trách nhiệm cập nhật lịch sử lỗi.
        - Generation chịu trách nhiệm tạo nội dung feedback.
        - Manager chỉ điều phối và tổng hợp kết quả.
    """

    def __init__(
        self,
        history_store,
        adaptation=None,
        generator=None,
    ):
        self.store = history_store

        self.adaptation = adaptation or feedback_adaptation.update_error_history

        self.generator = generator or feedback_generation.generate_feedback  # sửa

    def process_attempt(
        self,
        learner_id: str,
        sign_id: str,
        error_json: dict,
    ) -> dict:
        """
        Xử lý một lượt luyện tập.

        Args:
            learner_id: ID người học.
            sign_id: ID ký hiệu đang luyện tập.
            error_json: JSON kết quả phân tích lỗi từ Error Detection.

        Returns:
            dict: Kết quả luyện tập và feedback gửi về FE.
        """

        # 1. Lấy lịch sử lỗi của người dùng.
        history = self.store.get(learner_id) or {}

        # 2. Kiểm tra kết quả nhận diện.
        is_correct = error_json.get("is_correct", False)

        # 3. Gửi lỗi và lịch sử cho Adaptation.
        adaptation_result = self.adaptation(
            error_json=error_json,
            history=history,
            sign_id=sign_id,
        )

        # 4. Nhận lịch sử đã cập nhật và cấp độ feedback.
        updated_history = adaptation_result["updated_history"]
        feedback_levels = adaptation_result.get("feedback_levels", {})

        # 5. Lưu lịch sử mới.
        self.store.save(learner_id, updated_history)

        # 6. Nếu người dùng thực hiện đúng, không cần tạo feedback sửa lỗi.
        if is_correct:
            return {
                "learner_id": learner_id,
                "sign_id": sign_id,
                "is_correct": True,
                "feedbacks": {},
                "message": "Correct! Well done.",  # sửa tùy vào nên feedback gì cho người dùng mà ấy
            }

        # 7. Tạo feedback theo cấp độ Adaptation đã quyết định.
        feedbacks = {}

        errors = error_json.get("errors", {})

        for component, level in feedback_levels.items():
            error_info = errors.get(component)

            feedbacks[component] = self.generator(
                component=component,
                level=level,
                error_info=error_info,
            )

        # 8. Tổng hợp kết quả trả về FE.
        return {
            "learner_id": learner_id,
            "sign_id": sign_id,
            "is_correct": False,
            "feedbacks": feedbacks,
            "needs_review": adaptation_result.get(
                "needs_review", False
            ),  # cần xử lí
        }
