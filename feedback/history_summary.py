"""Quản lý lịch sử lỗi: ghi thêm thông tin sau mỗi lượt và đánh giá tiến bộ.

feedback_adaptation.py giữ các bộ đếm dùng để chọn cấp (level, streak, total).
Module này bổ sung, cho mỗi thành phần H/L/O/M:

- attempts:            số lượt đã được đánh giá
- recent:              kết quả các lượt gần nhất (cũ -> mới), tối đa RECENT_LIMIT
- last_feedback_level: cấp feedback đã hiển thị ở lượt sai gần nhất
- feedbacks_shown:     số lần đã hiển thị feedback cho thành phần này
- corrections:         số lần sửa được lỗi (lượt sai có feedback -> lượt sau đúng)
- fixed_after_level:   corrections tách theo cấp feedback ngay trước khi sửa được

và đánh giá tiến bộ theo bảng của Tuyền:
    Chưa có dữ liệu / Lỗi mới / Chưa cải thiện / Đang tiến bộ / Ổn định.
"""

from copy import deepcopy

COMPONENT_ORDER = ("H", "L", "O", "M")
COMPONENT_NAMES = {
    "H": "Hình dạng bàn tay",
    "L": "Vị trí bàn tay",
    "O": "Hướng bàn tay",
    "M": "Chuyển động",
}
COMPONENT_CODES = {
    "H": "handshape",
    "L": "location",
    "O": "orientation",
    "M": "movement",
}

RECENT_LIMIT = 10

PROGRESS_LABELS = {
    "no_data": "Chưa có dữ liệu",
    "new_error": "Lỗi mới",
    "not_improving": "Chưa cải thiện",
    "improving": "Đang tiến bộ",
    "stable": "Ổn định",
}

TRACKING_DEFAULTS = {
    "attempts": 0,
    "recent": [],
    "last_feedback_level": None,
    "feedbacks_shown": 0,
    "corrections": 0,
    "fixed_after_level": {},
}


def record_attempt(history, component_results, shown_levels):
    """Ghi thêm thông tin theo dõi vào history (đã được adaptation cập nhật).

    component_results: {"H": True/False, ...} của lượt này.
    shown_levels:      {"H": 2, ...} cấp feedback thực sự hiển thị ở lượt này.
    Trả về bản sao history đã cập nhật.
    """
    history = deepcopy(history)
    for component, is_correct in component_results.items():
        state = history.setdefault(component, {})
        for key, default in TRACKING_DEFAULTS.items():
            if key not in state or state[key] is None and default is not None:
                state[key] = deepcopy(default)

        previous = state["recent"][-1] if state["recent"] else None
        state["attempts"] += 1
        state["recent"] = (state["recent"] + ["correct" if is_correct else "wrong"])[
            -RECENT_LIMIT:
        ]

        if is_correct:
            # Sửa được lỗi: lượt trước sai và đã có feedback hiển thị.
            if previous == "wrong" and state["last_feedback_level"] is not None:
                state["corrections"] += 1
                key = str(state["last_feedback_level"])
                state["fixed_after_level"][key] = state["fixed_after_level"].get(key, 0) + 1
        elif component in shown_levels:
            state["feedbacks_shown"] += 1
            state["last_feedback_level"] = shown_levels[component]
    return history


def evaluate_progress(state):
    """Trả về mã tiến bộ của một thành phần từ trạng thái lịch sử."""
    if not state:
        return "no_data"
    total_wrong = state.get("total_wrong", 0)
    total_correct = state.get("total_correct", 0)
    if total_wrong + total_correct == 0:
        return "no_data"
    if state.get("last_result") == "correct":
        return "stable" if total_wrong == 0 else "improving"
    return "new_error" if total_wrong == 1 else "not_improving"


def summarize_history(history, components=COMPONENT_ORDER):
    """Bảng lịch sử lỗi để FE hiển thị: mỗi thành phần một dòng."""
    rows = []
    for component in components:
        state = (history or {}).get(component) or {}
        progress = evaluate_progress(state)
        rows.append(
            {
                "component": component,
                "error_type": COMPONENT_CODES[component],
                "name": COMPONENT_NAMES[component],
                "level": state.get("level"),
                "wrong_streak": state.get("wrong_streak", 0),
                "correct_streak": state.get("correct_streak", 0),
                "total_wrong": state.get("total_wrong", 0),
                "total_correct": state.get("total_correct", 0),
                "attempts": state.get("attempts", 0),
                "recent": list(state.get("recent") or []),
                "feedbacks_shown": state.get("feedbacks_shown", 0),
                "corrections": state.get("corrections", 0),
                "needs_review": bool(state.get("needs_review", False)),
                "progress": progress,
                "progress_label": PROGRESS_LABELS[progress],
            }
        )
    return rows
