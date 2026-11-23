# xây dựng feedback cấp 1 2 3
# thiết kế dữ liệu lịch sử lỗi
# xây dựng xử lí lỗi cho mô hình động + trích đặc trưng khác + hàm chuẩn hóa khác
# xây hàm manager trung gian
# tải data về
# kiểm thử cho feedback
# kết hợp classification + feedback
# cần xử lí xem khi vượt cấp 3 sẽ như thế nào -> gợi ý bài dễ hơn
# Phản hồi có giúp người học cải thiện không?
# chuẩn hóa xoay - tuy nhiên có những chữ cần xoay
# các chữ cái dễ nhầm lẫn

"""
1. initial_levels: Hiện tại mặc định L1. Nếu muốn chọn cấp ban đầu theo mức độ nghiêm trọng thì tầng điều phối có thể truyền cấp đã xác định vào đây. Module này không cần biết cách tính severity.
2. previous_history: Phải lấy từ lịch sử của đúng người học và đúng ký hiệu đang luyện tập.
3. Error JSON: Giữ nguyên ở tầng điều phối để chuyển sang Feedback Generation.
"""

from copy import deepcopy

# ---------- Cấu hình ----------
LEVEL_MIN = 1
LEVEL_MAX = 3

WRONG_THRESHOLD = 3  # sai liên tiếp đủ số này -> tăng 1 cấp
CORRECT_THRESHOLD = 2  # đúng liên tiếp đủ số này -> giảm 1 cấp
STUCK_THRESHOLD = 2  # sai liên tiếp ở cấp tối đa đủ số này -> cần can thiệp

COMPONENTS = ("H", "L", "O", "M")  # Handshape, Location, Orientation, Movement


# tạm vì chưa có lịch sử lỗi
def default_component_history() -> dict:
    """Khởi tạo lịch sử cho một thành phần."""
    return {
        "level": None,  # chưa từng có lỗi -> chưa có cấp
        "wrong_streak": 0,  # sai liên tiếp hiện tại
        "correct_streak": 0,  # đúng liên tiếp hiện tại
        "total_wrong": 0,  # tổng số lần sai tích lũy (không bị xóa khi làm đúng)
        "total_correct": 0,  # tổng số lần đúng tích lũy
        "last_result": None,  # "wrong" | "correct" | None
        "needs_review": False,  # sai mãi ở cấp tối đa -> gợi ý can thiệp
    }


def initial_level_from_severity(severity: float | None) -> int:
    """
    Chọn cấp ban đầu theo mức độ lỗi (severity trong [0, 1]).
    Thay ngưỡng bên dưới theo thang đo thực tế của dự án.
    """
    if severity is None:
        return LEVEL_MIN
    if severity < 0.34:
        return 1
    if severity < 0.67:
        return 2
    return 3


def update_component_history(state: dict, is_correct: bool) -> dict:
    """
    Cập nhật chuỗi đúng/sai của một thành phần.
    Không phân tích nội dung lỗi.
    """
    if not is_correct:
        state["total_wrong"] += 1
        state["correct_streak"] = 0

        if state["last_result"] == "correct":
            # Trước đúng, nay sai -> bắt đầu lại chuỗi sai, giữ nguyên cấp
            state["wrong_streak"] = 1
        else:
            state["wrong_streak"] += 1

        state["last_result"] = "wrong"
    else:
        state["total_correct"] += 1
        state["wrong_streak"] = 0  # chỉ đặt lại chuỗi, KHÔNG xóa total_wrong
        state["correct_streak"] += 1
        state["last_result"] = "correct"

    return state


def select_feedback_level(state: dict, initial_level: int = LEVEL_MIN) -> int:
    """
    Quyết định cấp feedback dựa trên lịch sử.

    - Lần đầu mắc lỗi: dùng initial_level (tính từ mức độ lỗi).
    - Sai liên tiếp đạt ngưỡng -> tăng 1 cấp.
    - Đúng liên tiếp đạt ngưỡng -> giảm 1 cấp.
    - Chưa đạt ngưỡng -> giữ nguyên.
    """
    initial_level = max(LEVEL_MIN, min(initial_level, LEVEL_MAX))

    # Lần đầu tiên có lỗi ở thành phần này
    if state["level"] is None:
        # Chỉ gán cấp khi thật sự có lỗi; nếu đúng từ đầu thì chưa cần cấp
        if state["last_result"] == "wrong":
            state["level"] = initial_level
        return state["level"]

    if state["wrong_streak"] >= WRONG_THRESHOLD:
        state["level"] = min(state["level"] + 1, LEVEL_MAX)
        state["wrong_streak"] = 0
        state["correct_streak"] = 0

    elif state["correct_streak"] >= CORRECT_THRESHOLD:
        state["level"] = max(state["level"] - 1, LEVEL_MIN)
        state["wrong_streak"] = 0
        state["correct_streak"] = 0

    return state["level"]


def update_error_history(
    previous_history: dict,
    component_results: dict,
    initial_levels: dict | None = None,
) -> dict:
    """
    Điều phối cập nhật lịch sử H, L, O, M.

    Input:
        previous_history:  lịch sử trước đó (đọc từ DB/file).
        component_results: {"H": True/False, ...} đúng/sai ở lượt hiện tại.
        initial_levels:    cấp ban đầu theo từng thành phần (từ mức độ lỗi).

    Output:
        feedback_levels: cấp feedback cho các thành phần đang sai.
        results:         trạng thái từng thành phần ("correct"/"wrong").
        needs_review:    các thành phần sai kéo dài ở cấp tối đa.
        updated_history: lịch sử sau khi cập nhật (để lưu lại).
    """
    history = deepcopy(previous_history)
    initial_levels = initial_levels or {}

    feedback_levels = {}
    results = {}
    needs_review = []

    for component in COMPONENTS:
        if component not in component_results:
            continue  # không đánh giá ở lượt này -> giữ nguyên lịch sử

        is_correct = component_results[component]
        state = history.setdefault(component, default_component_history())

        # Bước 1: cập nhật chuỗi đúng/sai
        update_component_history(state, is_correct)

        # Bước 2: chọn cấp feedback
        level = select_feedback_level(
            state, initial_level=initial_levels.get(component, LEVEL_MIN)
        )

        results[component] = "correct" if is_correct else "wrong"

        if not is_correct:
            feedback_levels[component] = level

            # Sai kéo dài dù đã ở cấp cao nhất
            state["needs_review"] = (
                level == LEVEL_MAX and state["wrong_streak"] >= STUCK_THRESHOLD
            )
            if state["needs_review"]:
                needs_review.append(component)
        else:
            state["needs_review"] = False

    return {
        "feedback_levels": feedback_levels,
        "results": results,
        "needs_review": needs_review,
        "updated_history": history,
    }
