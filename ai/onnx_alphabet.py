"""Frozen skeleton CNN using 21 hand landmarks, following its author's renderer."""
import hashlib
import importlib.util
import json

import numpy as np
import onnxruntime as ort

from ai.paths import ROOT


class OnnxAlphabetClassifier:
    def __init__(self, threshold=0.5):
        directory = ROOT / "models/onnx_alphabet"
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for entry in manifest["files"]:
            if hashlib.sha256((directory / entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
                raise RuntimeError("ONNX asset checksum mismatch")
        spec = importlib.util.spec_from_file_location("frozen_alphabet_preprocessing", directory / "preprocessing.py")
        renderer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(renderer)
        self.renderer = renderer
        self.labels = (directory / "class_names.txt").read_text().strip().upper().split(",")
        if self.labels != list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            raise RuntimeError("Unexpected class ordering for frozen ONNX model")
        self.supported_targets = [label for label in self.labels if label not in {"J", "Z"}]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(directory / "asl_cnn_model.onnx"), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        if self.session.get_inputs()[0].shape[1:] != [96, 96, 3]:
            raise RuntimeError("Unexpected ONNX input dimensions")
        self.threshold = float(threshold)
        if not 0 <= self.threshold <= 1:
            raise ValueError("Invalid confidence threshold")
        self.provider = "onnx_alphabet"
        self.feature_version = "skeleton_rgb96_canvas192_fill07_pixel_aspect_v1"
        self.version = "onnx-alphabet-" + manifest["revision"][:12]
        self.revision = manifest["revision"]

    def features(self, landmarks, width, height):
        points = np.asarray(landmarks, dtype=np.float32)
        if points.shape != (21, 3) or not np.isfinite(points).all() or width <= 0 or height <= 0:
            raise ValueError("Invalid hand landmarks or image dimensions")
        if np.linalg.norm(points[9] - points[0]) < 1e-6:
            raise ValueError("Degenerate hand")
        pixels = points[:, :2] * np.array([width, height], dtype=np.float32)
        if np.ptp(pixels, axis=0).max() < 1e-6:
            raise ValueError("Degenerate projected hand")
        normalized = self.renderer.normalize_landmarks(pixels)
        return self.renderer.render_skeleton(normalized).astype(np.float32)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 4 or X.shape[1:] != (96, 96, 3) or not np.isfinite(X).all():
            raise ValueError("Expected RGB skeleton images [N,96,96,3]")
        logits = self.session.run(["logits"], {self.input_name: X})[0]
        logits -= logits.max(axis=1, keepdims=True)
        values = np.exp(logits)
        return values / values.sum(axis=1, keepdims=True)
