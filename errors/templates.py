"""Xây dựng và lưu mẫu chuẩn (templates) cho error_detection_static.

Tách ra từ features/feature_sample.py để có thể import mà không chạy pipeline.
"""

import numpy as np

from errors.error_detection_static import FEATURE_COUNT, IDX, STD_MIN


def build_templates(X, y, z_outlier=3.0, max_outlier_ratio=0.1):
    """X: (số_mẫu, 46), y: nhãn. Trả về {nhãn: (mean, std)}."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or X.shape[1] != FEATURE_COUNT:
        raise ValueError(f"X phải có dạng (số_mẫu, {FEATURE_COUNT})")
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError("y phải là vector nhãn có cùng số phần tử với X")

    floor = STD_MIN[IDX]
    templates = {}
    for c in np.unique(y):
        Xc = X[y == c][:, IDX]
        Xc = Xc[np.isfinite(Xc).all(axis=1)]
        if not len(Xc):
            raise ValueError(f"Không có mẫu hợp lệ cho nhãn {c!r}")

        mean, std = Xc.mean(0), np.maximum(Xc.std(0), floor)
        z = np.abs(Xc - mean) / std
        Xc = Xc[(z > z_outlier).mean(1) < max_outlier_ratio]  # lọc nhiễu
        if not len(Xc):
            raise ValueError(f"Tất cả mẫu của nhãn {c!r} bị loại khi lọc nhiễu")
        templates[str(c)] = (Xc.mean(0), np.maximum(Xc.std(0), floor))
    return templates


def save_templates(path, templates):
    """Lưu templates thành .npz (labels, means, stds) để API nạp lại."""
    if not templates:
        raise ValueError("templates không được để trống")

    labels = np.asarray(list(templates), dtype=str)
    means = np.stack([templates[label][0] for label in labels]).astype(float)
    stds = np.stack([templates[label][1] for label in labels]).astype(float)
    if means.shape[1:] != (FEATURE_COUNT,) or stds.shape != means.shape:
        raise ValueError(f"Mỗi template phải chứa mean/std có {FEATURE_COUNT} đặc trưng")

    np.savez_compressed(path, labels=labels, means=means, stds=stds)
