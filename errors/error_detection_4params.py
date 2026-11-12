import numpy as np

from errors.handshape import HANDSHAPE_FEATURES
from errors.orientation import ORIENTATION_FEATURES

# Thứ tự ghép khớp với get_handshape_features,
# get_orientation_features và get_location_features trong feature_extraction.py.
GROUPS = {
    "handshape": np.arange(0, 50),  # 25 góc, 15 khoảng cách, 10 tọa độ đầu ngón
    "orientation": np.arange(50, 56),  # 6 đặc trưng hướng bàn tay
    "location": np.arange(56, 58),  # tọa độ cổ tay x, y
}
FEATURE_COUNT = 58
IDX = np.arange(FEATURE_COUNT)

# Trọng số khi gộp điểm tổng; movement chưa được tính.
WEIGHTS = {"handshape": 0.5, "orientation": 0.2, "location": 0.15}

# Sàn std ở đơn vị đặc trưng: góc đã chia 180, khoảng cách đã chuẩn hóa,
# và tọa độ được lấy trực tiếp từ landmarks.
STD_MIN = np.full(FEATURE_COUNT, 0.05)
STD_MIN[:25] = 0.02
STD_MIN[40:50] = 0.02  # tọa độ đầu ngón tương đối với cổ tay
STD_MIN[56:58] = 0.02  # tọa độ cổ tay

FEATURES = {
    **HANDSHAPE_FEATURES,
    **{index + 50: feature for index, feature in ORIENTATION_FEATURES.items()},
    56: {"feature": "wrist_x", "unit": "normalized_coordinate", "scale": 1.0},
    57: {"feature": "wrist_y", "unit": "normalized_coordinate", "scale": 1.0},
}


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


def load_templates(path):
    """Load templates previously persisted by save_templates()."""
    with np.load(path, allow_pickle=False) as data:
        required = {"labels", "means", "stds"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Template file thiếu mảng: {', '.join(sorted(missing))}")

        labels = data["labels"]
        means = np.asarray(data["means"], dtype=float)
        stds = np.asarray(data["stds"], dtype=float)

    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("labels trong template file phải là vector không rỗng")
    if means.shape != (len(labels), FEATURE_COUNT) or stds.shape != means.shape:
        raise ValueError(
            f"means/stds trong template file phải có dạng (số_nhãn, {FEATURE_COUNT})"
        )
    if len(set(labels.tolist())) != len(labels):
        raise ValueError("labels trong template file không được trùng")
    if not np.isfinite(means).all() or not np.isfinite(stds).all() or (stds <= 0).any():
        raise ValueError("Template file chứa mean/std không hợp lệ")

    return {
        str(label): (mean.copy(), std.copy())
        for label, mean, std in zip(labels, means, stds)
    }


def _score(z):
    """z: vector z-score của 1 nhóm -> điểm 0-100."""
    rmse = np.sqrt(np.mean(np.clip(np.abs(z), 0, 4) ** 2))
    return 100 * (1 - min(rmse, 4) / 4)


def detect_errors(
    x,
    target,
    templates,
    names=None,
    use=("handshape", "orientation", "location"),
):
    x = np.asarray(x, dtype=float)
    if x.shape != (FEATURE_COUNT,):
        raise ValueError(f"x phải có đúng {FEATURE_COUNT} đặc trưng")

    unknown_groups = set(use) - GROUPS.keys()
    if unknown_groups:
        raise ValueError(
            f"Nhóm đặc trưng chưa được hỗ trợ: {', '.join(sorted(unknown_groups))}"
        )
    if not use:
        raise ValueError("use phải chứa ít nhất một nhóm đặc trưng")

    if not np.isfinite(x).all():
        return {
            "score": None,
            "components": {group: {"score": None, "deviations": []} for group in use},
            "errors": ["no_hand"],
        }

    mean, std = templates[target]
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    if mean.shape != (FEATURE_COUNT,) or std.shape != (FEATURE_COUNT,):
        raise ValueError(
            f"Template của {target!r} phải có đúng {FEATURE_COUNT} đặc trưng"
        )
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError(f"Template của {target!r} chứa mean/std không hợp lệ")

    z = (x - mean) / std

    result = {"components": {}}
    for g in use:
        ks = GROUPS[g]
        zg = z[ks]
        component_deviations = []
        bad = [k for k in np.argsort(-np.abs(zg)) if abs(zg[k]) > 2]
        if bad:
            k = bad[0]
            gi = int(ks[k])
            metadata = FEATURES[gi]
            scale = metadata["scale"]
            deviation = {
                "feature": (names or {}).get(gi, metadata["feature"]),
            }
            if "finger" in metadata:
                deviation["finger"] = metadata["finger"]
            deviation.update(
                {
                    "reference": round(float(mean[gi] * scale), 4),
                    "actual": round(float(x[gi] * scale), 4),
                    "delta": round(float((x[gi] - mean[gi]) * scale), 4),
                    "unit": metadata["unit"],
                    "z_score": round(float(zg[k]), 2),
                }
            )
            component_deviations.append(deviation)

        result["components"][g] = {
            "score": round(float(_score(zg)), 1),
            "deviations": component_deviations,
        }

    w = np.array([WEIGHTS[g] for g in use])
    s = np.array([result["components"][g]["score"] for g in use])
    result["score"] = round(float((w * s).sum() / w.sum()), 1)
    return result


def main(
    features,
    target,
    templates,
    names=None,
    use=("handshape", "orientation", "location"),
):
    """Detect feature deviations using templates prepared and saved separately."""
    return detect_errors(features, target, templates, names=names, use=use)


# Ví dụ output: mỗi thành phần chỉ giữ đặc trưng có |z_score| lớn nhất.
"""
{
  "components": {
    "handshape": {
      "score": 40.0,
      "deviations": [{
        "feature": "index_pip_angle",
        "finger": "index",
        "reference": 30.0,
        "actual": 45.0,
        "delta": 15.0,
        "unit": "degree",
        "z_score": 2.4
      }]
    },
    "orientation": {"score": 88.4, "deviations": []},
    "location": {"score": 80.1, "deviations": []}
  },
  "score": 58.5
}
"""
