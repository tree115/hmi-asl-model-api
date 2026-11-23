"""Chạy MediaPipe trên dataset ảnh ASL Alphabet để tạo mẫu chuẩn cho /errors/detect.

Dataset: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
Sau khi giải nén, thư mục train có dạng <data>/A/*.jpg, <data>/B/*.jpg, ...

Ví dụ:
    python -m scripts.build_error_templates ^
        --data D:/datasets/asl_alphabet_train/asl_alphabet_train ^
        --out data/error_templates.npz --per-class 500

Kết quả:
    <out>             file .npz (labels, means, stds) -> đặt AI_ERROR_TEMPLATE_DATA trỏ tới
    <out>.features.npz X (đặc trưng 46 chiều), y (nhãn), paths -> dùng lại, không chạy lại MediaPipe
    In ra số ảnh dùng được / bị bỏ theo từng chữ.

Pipeline giống hệt lúc chạy API: landmarks MediaPipe (tọa độ ảnh chuẩn hóa)
-> features.feature_extraction.preprocess_hand -> extract_static_phonological_features.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from errors.templates import build_templates, save_templates
from features import feature_extraction as fe

# J, Z là chữ động; del/nothing/space không phải ký hiệu chữ cái.
DEFAULT_SKIP = ("J", "Z", "del", "nothing", "space")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def extract_features(hands, image_bgr, flip=False):
    """Trả về (vector 46 chiều, lý do bỏ) cho một ảnh."""
    if flip:
        image_bgr = cv2.flip(image_bgr, 1)
    result = hands.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    if not result.multi_hand_landmarks:
        return None, "no_hand"
    landmarks = np.array(
        [[p.x, p.y, p.z] for p in result.multi_hand_landmarks[0].landmark], dtype=float
    )
    height, width = image_bgr.shape[:2]
    try:
        normalized = fe.preprocess_hand(landmarks, width, height)
        return fe.extract_static_phonological_features(normalized), None
    except ValueError:
        return None, "invalid_landmarks"


def build_dataset(root, per_class, skip, flip=False):
    import mediapipe as mp

    root = Path(root)
    class_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name not in skip)
    if not class_dirs:
        print(f"Không tìm thấy thư mục nhãn nào trong {root}")
        sys.exit(1)

    X, y, paths = [], [], []
    stats = {}
    with mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
    ) as hands:
        for class_dir in class_dirs:
            files = sorted(f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
            if per_class and len(files) > per_class:
                # Lấy ngẫu nhiên (cố định seed) thay vì N file đầu: ảnh dataset
                # là các frame liên tiếp nên N file đầu gần như trùng nhau.
                rng = np.random.default_rng(0)
                files = sorted(rng.choice(files, per_class, replace=False))
            reasons = Counter()
            for f in files:
                image = cv2.imread(str(f))
                if image is None:
                    reasons["unreadable"] += 1
                    continue
                feat, reason = extract_features(hands, image, flip=flip)
                if feat is None:
                    reasons[reason] += 1
                    continue
                X.append(feat)
                y.append(class_dir.name.upper())
                paths.append(str(f))
            used = len(files) - sum(reasons.values())
            stats[class_dir.name] = (used, len(files), dict(reasons))
            print(f"{class_dir.name:>8}: dùng {used}/{len(files)}  bỏ {dict(reasons)}")
    return np.asarray(X, dtype=float), np.asarray(y), np.asarray(paths), stats


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # console Windows mặc định cp1252
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="Thư mục chứa các thư mục con A, B, C, ...")
    parser.add_argument("--out", default="data/error_templates.npz")
    parser.add_argument("--per-class", type=int, default=500, help="Số ảnh tối đa mỗi chữ (0 = tất cả)")
    parser.add_argument("--skip", nargs="*", default=list(DEFAULT_SKIP))
    parser.add_argument(
        "--flip",
        action="store_true",
        help="Lật ngang ảnh trước khi trích (dùng nếu dataset là ảnh gương so với webcam của FE)",
    )
    parser.add_argument("--min-samples", type=int, default=30, help="Cảnh báo nếu một chữ có ít mẫu hơn")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    X, y, paths, stats = build_dataset(args.data, args.per_class, set(args.skip), flip=args.flip)
    if not len(X):
        print("Không trích được mẫu nào.")
        sys.exit(1)
    np.savez_compressed(out.with_suffix(".features.npz"), X=X, y=y, paths=paths)

    templates = build_templates(X, y)
    save_templates(out, templates)

    few = [c for c, (used, _, _) in stats.items() if used < args.min_samples]
    print(f"\nĐã lưu {len(templates)} mẫu chuẩn -> {out}")
    print(f"Đặc trưng thô -> {out.with_suffix('.features.npz')}")
    if few:
        print(f"CẢNH BÁO: các chữ có ít hơn {args.min_samples} mẫu: {', '.join(few)}")
    print(f"\nĐặt biến môi trường cho API:\n  $env:AI_ERROR_TEMPLATE_DATA = '{out.resolve()}'")


if __name__ == "__main__":
    main()
