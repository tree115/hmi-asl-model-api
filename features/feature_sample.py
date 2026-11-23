import numpy as np
from features import feature_extraction
import cv2, numpy as np, mediapipe as mp
from pathlib import Path
from tqdm import tqdm
from errors.handshape import HANDSHAPE_FEATURES
from errors.orientation import ORIENTATION_FEATURES

GROUPS = {
    "handshape": np.arange(0, 40),  # 25 góc, 15 khoảng cách
    "orientation": np.arange(40, 44),  # 4 đặc trưng hướng bàn tay
    "location": np.arange(44, 46),  # tọa độ cổ tay x, y
}
FEATURE_COUNT = 46
IDX = np.arange(FEATURE_COUNT)

# Trọng số khi gộp điểm tổng; movement chưa được tính.
WEIGHTS = {"handshape": 0.5, "orientation": 0.2, "location": 0.15}

# Sàn std ở đơn vị đặc trưng: góc đã chia 180, khoảng cách đã chuẩn hóa,
# và tọa độ được lấy trực tiếp từ landmarks.
STD_MIN = np.full(FEATURE_COUNT, 0.05)
STD_MIN[:25] = 0.02
STD_MIN[44:46] = 0.02  # tọa độ cổ tay

FEATURES = {
    **HANDSHAPE_FEATURES,
    **{index + 40: feature for index, feature in ORIENTATION_FEATURES.items()},
    44: {"feature": "wrist_x", "unit": "normalized_coordinate", "scale": 1.0},
    45: {"feature": "wrist_y", "unit": "normalized_coordinate", "scale": 1.0},
}


hands = mp.solutions.hands.Hands(
    static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
)

# còn cần phải chuẩn hóa nữa bây


def extract_from_image(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_hand_landmarks:
        return None
    lm = np.array([[p.x, p.y, p.z] for p in res.multi_hand_landmarks[0].landmark])
    norm_landmarks = feature_extraction.preprocess_hand(
        lm, img.shape[1], img.shape[0]
    )
    return feature_extraction.extract_static_phonological_features(norm_landmarks)


def build_dataset(root, per_class=500, skip=("nothing",)):
    X, y = [], []
    for cls_dir in sorted(Path(root).iterdir()):
        if not cls_dir.is_dir() or cls_dir.name in skip:
            continue
        files = sorted(cls_dir.glob("*.jpg"))[:per_class]
        for f in tqdm(files, desc=cls_dir.name):
            feat = extract_from_image(f)
            if feat is not None:
                X.append(feat)
                y.append(cls_dir.name)
    return np.array(X), np.array(y)


def build_templates(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or X.shape[1] != FEATURE_COUNT:
        raise ValueError(f"X phải có dạng (số_mẫu, {FEATURE_COUNT})")
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError("y phải là vector nhãn có cùng số phần tử với X")

    templates = {}
    for c in np.unique(y):
        Xc = X[y == c][:, IDX]
        Xc = Xc[np.isfinite(Xc).all(axis=1)]
        if not len(Xc):
            raise ValueError(f"Không có mẫu hợp lệ cho nhãn {c!r}")

        floor = STD_MIN[IDX]
        mean, std = Xc.mean(0), np.maximum(Xc.std(0), floor)
        z = np.abs(Xc - mean) / std
        Xc = Xc[(z > 3).mean(1) < 0.1]  # lọc nhiễu
        if not len(Xc):
            raise ValueError(f"Tất cả mẫu của nhãn {c!r} bị loại khi lọc nhiễu")
        templates[c] = (Xc.mean(0), np.maximum(Xc.std(0), floor))
    return templates


def save_templates(path, templates):
    """Persist prebuilt templates for reuse by the API."""
    if not templates:
        raise ValueError("templates không được để trống")

    labels = np.asarray(list(templates), dtype=str)
    means = np.stack([templates[label][0] for label in labels]).astype(float)
    stds = np.stack([templates[label][1] for label in labels]).astype(float)
    if means.shape[1:] != (FEATURE_COUNT,) or stds.shape != means.shape:
        raise ValueError(
            f"Mỗi template phải chứa mean/std có {FEATURE_COUNT} đặc trưng"
        )

    np.savez_compressed(path, labels=labels, means=means, stds=stds)


X, y = build_dataset("asl_alphabet_train/asl_alphabet_train", per_class=500)
np.save("X.npy", X)
np.save("y.npy", y)

templates = build_templates(X, y)

save_templates("asl_alphabet_train/asl_alphabet_train", templates)
