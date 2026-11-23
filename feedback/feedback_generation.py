"""Feedback Generation: biến kết quả error detection + cấp độ thành nội dung phản hồi.

Input (do FeedbackManager truyền vào):
    component:  "H" | "L" | "O" | "M"  (thành phần lỗi)
    level:      1 | 2 | 3              (từ feedback_adaptation)
    error_info: phần "components[<group>]" trong JSON của error detection,
                ví dụ {"score": 40.0, "deviations": [{"feature": ..., "delta": ...}]}
    target:     chữ/từ mục tiêu, ví dụ "A"
    version:    "V1" (ký hiệu tĩnh) | "V2" (ký hiệu động)

Output: dict JSON-serializable để manager gửi lên FE (xem generate_feedback).

Hiện mới hiện thực đầy đủ Cấp 1 (text + gợi ý sửa ngắn). Cấp 2/3 trả về cùng
nội dung text của Cấp 1 và để trống "visual"/"steps" cho tới khi có nội dung
hình ảnh/hướng dẫn từng bước.

Không dùng LLM: nội dung sinh theo quy tắc cố định để kết quả tái hiện được và
so sánh công bằng giữa điều kiện Fixed và Adaptive (báo cáo mục 5.4).
"""

COMPONENTS = ("H", "L", "O", "M")

# Tên nhóm trong JSON error detection -> mã thành phần dùng ở adaptation.
GROUP_TO_COMPONENT = {
    "handshape": "H",
    "location": "L",
    "orientation": "O",
    "movement": "M",
}
COMPONENT_TO_GROUP = {v: k for k, v in GROUP_TO_COMPONENT.items()}

# ---------- Cấp 1: câu báo thành phần sai ----------
LEVEL1_MESSAGES = {
    "V1": {
        "H": "Hình dạng bàn tay của bạn chưa giống mẫu.",
        "L": "Bạn đang đặt bàn tay chưa đúng vị trí.",
        "O": "Lòng bàn tay của bạn đang hướng chưa đúng.",
        # V1 không đánh giá chuyển động.
    },
    "V2": {
        "H": "Hình dạng bàn tay của bạn chưa giống mẫu.",
        "L": "Bạn đang đặt bàn tay chưa đúng vị trí.",
        "O": "Lòng bàn tay của bạn đang hướng chưa đúng.",
        "M": "Bạn đang di chuyển bàn tay chưa đúng cách.",
    },
}

# Gợi ý chung khi không có chi tiết sai lệch nào dùng được.
GENERIC_HINTS = {
    "H": "Hãy so sánh các ngón tay của bạn với hình mẫu và tạo lại hình bàn tay.",
    "L": "Hãy đặt bàn tay ở vị trí giống như trong hình mẫu.",
    "O": "Hãy xoay cổ tay để lòng bàn tay hướng giống như hình mẫu.",
    "M": "Hãy xem lại mẫu và di chuyển bàn tay theo đúng hướng như mẫu.",
}

# Mô tả ngắn cách tạo hình tay cho các chữ cái tĩnh ASL (V1).
SIGN_HANDSHAPE_TIPS = {
    "A": "Nắm tay lại, ngón cái áp sát bên cạnh ngón trỏ.",
    "B": "Duỗi thẳng bốn ngón và khép sát nhau, ngón cái gập ngang lòng bàn tay.",
    "C": "Cong các ngón và ngón cái để bàn tay tạo thành hình chữ C.",
    "D": "Duỗi thẳng ngón trỏ, các ngón còn lại cong lại chạm vào đầu ngón cái.",
    "E": "Gập cong các đầu ngón xuống, ngón cái gập ngang phía dưới các đầu ngón.",
    "F": "Chạm đầu ngón trỏ vào ngón cái thành vòng tròn, ba ngón còn lại duỗi thẳng.",
    "G": "Duỗi ngón trỏ và ngón cái song song nằm ngang, các ngón còn lại nắm lại.",
    "H": "Duỗi ngón trỏ và ngón giữa khép sát, nằm ngang; các ngón còn lại nắm lại.",
    "I": "Duỗi thẳng ngón út, các ngón còn lại nắm lại.",
    "K": "Duỗi ngón trỏ và ngón giữa hướng lên thành chữ V, ngón cái đặt giữa hai ngón.",
    "L": "Duỗi ngón trỏ hướng lên và ngón cái sang ngang để tạo chữ L.",
    "M": "Nắm tay, đặt ngón cái nằm dưới ngón trỏ, ngón giữa và ngón áp út.",
    "N": "Nắm tay, đặt ngón cái nằm dưới ngón trỏ và ngón giữa.",
    "O": "Cong tất cả các ngón chạm vào đầu ngón cái để tạo thành chữ O.",
    "P": "Tạo hình giống chữ K nhưng hướng các ngón xuống dưới.",
    "Q": "Tạo hình giống chữ G nhưng hướng ngón trỏ và ngón cái xuống dưới.",
    "R": "Bắt chéo ngón giữa lên trên ngón trỏ, các ngón còn lại nắm lại.",
    "S": "Nắm tay, ngón cái vắt ngang phía trước các ngón.",
    "T": "Nắm tay, kẹp ngón cái vào giữa ngón trỏ và ngón giữa.",
    "U": "Duỗi ngón trỏ và ngón giữa khép sát nhau, hướng lên.",
    "V": "Duỗi ngón trỏ và ngón giữa tách ra thành chữ V.",
    "W": "Duỗi ngón trỏ, ngón giữa và ngón áp út tách ra thành chữ W.",
    "X": "Cong ngón trỏ như móc câu, các ngón còn lại nắm lại.",
    "Y": "Duỗi ngón cái và ngón út, các ngón còn lại nắm lại.",
}

FINGER_NAMES = {
    "thumb": "ngón cái",
    "index": "ngón trỏ",
    "middle": "ngón giữa",
    "ring": "ngón áp út",
    "pinky": "ngón út",
}

# Các góc đo độ "mở" giữa hai ngón (lớn hơn = tách xa hơn),
# khác với góc khớp (180° = khớp duỗi thẳng, nhỏ hơn = gập).
SPREAD_ANGLE_FEATURES = {
    "thumb_index_spread_angle",
    "index_pinky_spread_angle",
    "ring_pinky_angle",
    "index_middle_angle",
    "thumb_pinky_angle",
    "index_pinky_angle",
}

# Bỏ qua sai lệch quá nhỏ để không đưa ra hướng dẫn vô nghĩa.
MIN_DELTA = {"degree": 5.0, "palm_ratio": 0.05, "unitless": 0.05}


def _finger_pair(code):
    """'thumb_index' -> ('ngón cái', 'ngón trỏ')."""
    first, _, second = (code or "").partition("_")
    return FINGER_NAMES.get(first, first), FINGER_NAMES.get(second, second)


def _handshape_hint(dev):
    feature = dev.get("feature", "")
    delta = dev.get("delta")
    finger = dev.get("finger")
    unit = dev.get("unit")
    if delta is None or abs(delta) < MIN_DELTA.get(unit, 0.0):
        return None

    if unit == "degree":
        if feature in SPREAD_ANGLE_FEATURES:
            a, b = _finger_pair(finger)
            if delta > 0:
                return f"Hãy khép {a} và {b} lại gần nhau hơn."
            return f"Hãy tách {a} và {b} ra xa nhau hơn."
        name = FINGER_NAMES.get(finger, finger)
        # Góc khớp nhỏ hơn mẫu -> đang gập nhiều hơn mẫu.
        if delta < 0:
            return f"Hãy duỗi {name} thẳng hơn."
        return f"Hãy gập {name} lại nhiều hơn."

    if unit == "palm_ratio":
        if feature.startswith("wrist_"):
            name = FINGER_NAMES.get(finger, finger)
            # Đầu ngón gần cổ tay hơn mẫu -> ngón đang co lại.
            if delta < 0:
                return f"Hãy duỗi {name} ra xa hơn."
            return f"Hãy co {name} lại gần lòng bàn tay hơn."
        a, b = _finger_pair(finger)
        if delta > 0:
            return f"Hãy đưa đầu {a} và đầu {b} lại gần nhau hơn."
        return f"Hãy tách đầu {a} và đầu {b} ra xa nhau hơn."

    return None


def _orientation_hint(dev):
    feature = dev.get("feature", "")
    delta = dev.get("delta")
    ref = dev.get("reference")
    actual = dev.get("actual")
    if delta is None or abs(delta) < MIN_DELTA["unitless"]:
        return None

    if feature == "palm_roll_sin":
        # roll > 0: các ngón chỉ sang phải trong ảnh camera (chưa lật gương),
        # tức sang phía tay TRÁI của người học. Giả định bàn tay hướng lên.
        if delta > 0:
            return "Hãy nghiêng các đầu ngón tay sang phía tay phải của bạn một chút."
        return "Hãy nghiêng các đầu ngón tay sang phía tay trái của bạn một chút."
    if feature == "palm_roll_cos":
        if delta < 0:
            return "Hãy dựng bàn tay thẳng đứng hơn, giống hình mẫu."
        return "Hãy nghiêng bàn tay nhiều hơn, giống hình mẫu."
    if feature.startswith("palm_horizontal_axis"):
        if ref is not None and actual is not None and ref * actual < 0:
            return "Hãy lật cổ tay lại: có thể bạn đang để ngược mặt lòng bàn tay so với mẫu."
        return "Hãy xoay cổ tay nhẹ để lòng bàn tay hướng giống hình mẫu."
    return None


def _location_hint(dev):
    feature = dev.get("feature", "")
    delta = dev.get("delta")
    if delta is None or abs(delta) < MIN_DELTA["unitless"]:
        return None
    # Tọa độ ảnh: x tăng sang phải ảnh (= phía tay trái người học), y tăng xuống dưới.
    if feature == "wrist_x":
        side = "phải" if delta > 0 else "trái"
        return f"Hãy dịch cổ tay sang phía tay {side} của bạn một chút."
    if feature == "wrist_y":
        return "Hãy nâng cổ tay lên cao hơn." if delta > 0 else "Hãy hạ cổ tay xuống thấp hơn."
    return None


_HINT_BUILDERS = {"H": _handshape_hint, "O": _orientation_hint, "L": _location_hint}


def build_hint(component, error_info=None, target=None):
    """Trả về (hint, source). source cho biết gợi ý lấy từ đâu (để log/đánh giá)."""
    deviations = (error_info or {}).get("deviations") or []
    builder = _HINT_BUILDERS.get(component)
    if builder:
        for dev in deviations:
            hint = builder(dev)
            if hint:
                return hint, "deviation"

    if component == "H" and target and target.upper() in SIGN_HANDSHAPE_TIPS:
        return SIGN_HANDSHAPE_TIPS[target.upper()], "sign_library"

    return GENERIC_HINTS[component], "generic"


def generate_feedback(component, level, error_info=None, target=None, version="V1"):
    """Sinh nội dung phản hồi cho MỘT thành phần đang sai.

    Trả về None nếu thành phần không áp dụng cho phiên bản (ví dụ M ở V1).
    """
    if component not in COMPONENTS:
        raise ValueError(f"Thành phần không hợp lệ: {component!r}")
    if version not in LEVEL1_MESSAGES:
        raise ValueError(f"Phiên bản không hợp lệ: {version!r}")
    if level not in (1, 2, 3):
        raise ValueError(f"Cấp độ phải là 1, 2 hoặc 3, nhận {level!r}")

    message = LEVEL1_MESSAGES[version].get(component)
    if message is None:
        return None

    hint, hint_source = build_hint(component, error_info, target)
    deviations = (error_info or {}).get("deviations") or []

    return {
        "component": component,
        "level": level,
        "message": message,
        "hint": hint,
        "hint_source": hint_source,
        "details": deviations[0] if deviations else None,
        # Cấp 2/3 chưa có nội dung trực quan; FE hiển thị text như Cấp 1.
        "visual": None,
        "steps": None,
    }


def component_results_from_detection(detection, threshold_score=None):
    """Chuyển JSON error detection -> {"H": True/False, ...} cho feedback_adaptation.

    Quy tắc: một thành phần SAI khi có ít nhất một đặc trưng lệch |z| > 2
    (error detection chỉ đưa vào "deviations" những đặc trưng như vậy), hoặc
    khi điểm thành phần thấp hơn threshold_score (nếu truyền vào).

    Trả về {} khi không đánh giá được (no_hand, thiếu dữ liệu): theo báo cáo
    mục 4.2.3 lượt này không được tính đúng/sai và không cập nhật lịch sử.
    """
    # errors: no_hand / invalid_landmarks / unsupported_target
    if not detection or detection.get("errors"):
        return {}

    results = {}
    for group, info in (detection.get("components") or {}).items():
        component = GROUP_TO_COMPONENT.get(group)
        if component is None or info is None or info.get("score") is None:
            continue
        wrong = bool(info.get("deviations"))
        if threshold_score is not None and info["score"] < threshold_score:
            wrong = True
        results[component] = not wrong
    return results


def error_infos_from_detection(detection):
    """{"H": components["handshape"], ...} để truyền vào generate_feedback."""
    infos = {}
    for group, info in ((detection or {}).get("components") or {}).items():
        component = GROUP_TO_COMPONENT.get(group)
        if component is not None:
            infos[component] = info
    return infos
