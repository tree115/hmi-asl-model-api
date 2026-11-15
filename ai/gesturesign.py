"""Server-side inference for the pretrained GestureSign word checkpoint."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import onnxruntime as ort

from .gesturesign_features import landmarks as lm
from .gesturesign_features.features import FEATURE_VERSION, sequence_features
from .gesturesign_features.resample import fit_length, resample


ASSETS = Path(__file__).resolve().parents[1] / "models" / "gesturesign"


def label_key(label):
    return re.sub(r"[\s_-]+", "", label.casefold())


class GestureSign:
    def __init__(self):
        self.meta = json.loads((ASSETS / "model-meta.json").read_text(encoding="utf-8"))
        self.labels = json.loads((ASSETS / "labels.json").read_text(encoding="utf-8"))["signs"]
        model = (ASSETS / "model.onnx").read_bytes()
        digest = hashlib.sha256(model).hexdigest()
        if self.meta["version"] != f"default-{digest[:10]}" or self.meta["featureVersion"] != FEATURE_VERSION:
            raise ValueError("GestureSign checkpoint or feature version mismatch")
        if len(self.labels) != self.meta["classes"] or any(row["index"] != i for i, row in enumerate(self.labels)):
            raise ValueError("GestureSign label table does not match the checkpoint")
        self.shipped = np.array([row["index"] for row in self.labels if row["shipped"]])
        self.targets = {
            label_key(row[key]): row["label"]
            for row in self.labels if row["shipped"]
            for key in ("sign", "label")
        }
        self.session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])

    def predict(self, frames, fps):
        points = np.full((len(frames), lm.N_POINTS, 3), np.nan, dtype=np.float32)
        for index, frame in enumerate(frames):
            for field, part in (("face", "face"), ("pose", "pose"),
                                ("left_hand", "left_hand"), ("right_hand", "right_hand")):
                values = getattr(frame, field)
                if values is not None:
                    start = lm.TYPE_OFFSETS[part]
                    points[index, start:start + lm.TYPE_SIZES[part]] = [
                        point if point is not None else (np.nan, np.nan, np.nan)
                        for point in values
                    ]
        hands = np.concatenate((points[:, lm.LEFT_HAND_START:lm.POSE_START, 0],
                                points[:, lm.RIGHT_HAND_START:, 0]), axis=1)
        visible = np.isfinite(hands).any(axis=1)
        usable = int(visible.sum())
        if not usable:
            return "no_hand", usable, []
        shoulders = points[visible][:, [lm.pose(11), lm.pose(12)], :2]
        in_view = ((shoulders >= -0.02) & (shoulders <= 1.02)).all(axis=(1, 2))
        if int(in_view.sum()) * 2 < usable:
            return "uncertain", usable, []
        fitted = fit_length(resample(points, fps, self.meta["fps"]), self.meta["maxFrames"])
        if len(fitted) < self.meta["minFrames"]:
            return "uncertain", usable, []
        features = sequence_features(fitted, self.meta["featureSet"])[None, ...]
        io = self.meta["io"]
        logits = self.session.run([io["outputs"][0]], {io["input"]: features})[0].reshape(-1)
        calibration = self.meta["calibration"]
        logits = logits[self.shipped].astype(np.float64) / calibration["temperature"]
        if not np.isfinite(logits).all():
            return "uncertain", usable, []
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        top = np.argsort(-probabilities, kind="stable")[:3]
        candidates = [
            (self.labels[self.shipped[i]]["label"], float(probabilities[i])) for i in top
        ]
        status = "recognized" if candidates[0][1] >= calibration["notSureBelow"] else "uncertain"
        return status, usable, candidates
