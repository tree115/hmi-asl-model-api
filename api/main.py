"""FastAPI boundary for versioned static ASL recognition model adapters."""
from contextlib import asynccontextmanager
import os
import threading
import time
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import numpy as np
from pydantic import BaseModel, Field, FiniteFloat, model_validator

from ai.onnx_alphabet import OnnxAlphabetClassifier


class LandmarkRequest(BaseModel):
    target_label: str = Field(min_length=1, max_length=16)
    landmarks: Annotated[list[tuple[FiniteFloat, FiniteFloat, FiniteFloat]], Field(min_length=21, max_length=21)] | None
    image_width: int = Field(gt=0, le=8192)
    image_height: int = Field(gt=0, le=8192)
    coordinate_space: Literal["mediapipe_image_normalized_v1"] = "mediapipe_image_normalized_v1"
    mirrored: Literal[False] = False


class Candidate(BaseModel):
    label: str
    score: float


class Frame(BaseModel):
    handedness: Literal["Left", "Right"] | None = None
    timestamp_ms: float = Field(ge=0, allow_inf_nan=False)
    landmarks: Annotated[list[tuple[FiniteFloat, FiniteFloat, FiniteFloat]], Field(min_length=21, max_length=21)] | None


class AttemptRequest(BaseModel):
    client_attempt_id: str = Field(min_length=1, max_length=80)
    target_label: str = Field(min_length=1, max_length=16)
    image_width: int = Field(gt=0, le=8192)
    image_height: int = Field(gt=0, le=8192)
    coordinate_space: Literal["mediapipe_image_normalized_v1"] = "mediapipe_image_normalized_v1"
    mirrored: Literal[False] = False
    frames: list[Frame] = Field(min_length=4, max_length=180)

    @model_validator(mode="after")
    def timestamps(self):
        times = [f.timestamp_ms for f in self.frames]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("Frame timestamps must be strictly increasing")
        if not 750 <= times[-1] - times[0] <= 5000:
            raise ValueError("Capture window must span 750..5000 ms")
        return self


class Prediction(BaseModel):
    request_id: str
    model_version: str
    feature_version: str
    status: Literal["recognized", "uncertain", "no_hand", "invalid_landmarks", "unsupported_target"]
    target_label: str
    predicted_label: str | None = None
    score: float | None = Field(default=None, description="Legacy alias of model_confidence; NOT a learner skill score")
    predicted_sign: str | None = None
    model_confidence: float | None = Field(default=None, description="Classifier output, not a calibrated skill or correctness probability")
    valid: bool = False
    matches_target: bool | None = Field(default=None, description="Accepted predicted label matches target; null when assessment is unavailable")
    top3: list[Candidate] = Field(default_factory=list)
    error_type: None = None
    diagnosis_supported: Literal[False] = False
    update_error_history: Literal[False] = False
    processing_ms: float
    recognition_method: str = "static_classifier"
    motion_details: dict | None = None


class AttemptPrediction(Prediction):
    client_attempt_id: str
    frames_received: int
    frames_usable: int
    accepted_frames: int | None
    winning_votes: int | None
    stability: float | None
    aggregation_policy: str = "experimental_3_accepted_min_75pct_all_frames_v1"


class Engine:
    def __init__(self):
        self.provider = "onnx_alphabet"
        requested = os.environ.get("AI_MODEL_PROVIDER", "onnx_alphabet")
        if requested != self.provider:
            raise RuntimeError("This API-only package supports onnx_alphabet only")
        self.classifier = OnnxAlphabetClassifier(threshold=float(os.environ.get("AI_CONFIDENCE_THRESHOLD", "0.5")))
        self.labels = self.classifier.labels
        self.targets = list(self.classifier.supported_targets)
        self.feature_builder = self.classifier.features
        self.feature_version = self.classifier.feature_version
        self.threshold = self.classifier.threshold
        self.version = self.classifier.version + "+motion-v4"
        self.lock = threading.Lock()

    def result(self, target, status, start, **values):
        return Prediction(request_id=str(uuid4()), model_version=self.version,
                          feature_version=self.feature_version, target_label=target,
                          status=status, processing_ms=(time.perf_counter() - start) * 1000,
                          predicted_sign=values.get("predicted_label"), model_confidence=values.get("score"), **values)

    def classify(self, feature, target, start):
        scores = self.classifier.predict_proba(feature[None])[0]
        order = np.argsort(scores)[::-1]
        label = str(self.labels[order[0]])
        score = float(scores[order[0]])
        valid = score >= self.threshold and label in self.targets
        return self.result(target, "recognized" if valid else "uncertain", start,
                           predicted_label=label, score=score, valid=valid,
                           matches_target=(label == target) if valid else None,
                           top3=[Candidate(label=str(self.labels[i]), score=float(scores[i])) for i in order[:3]])

    def landmarks(self, payload):
        start = time.perf_counter()
        target = payload.target_label.strip().upper()
        if target not in self.targets:
            return self.result(target, "unsupported_target", start)
        if payload.landmarks is None:
            return self.result(target, "no_hand", start)
        try:
            feature = self.feature_builder(payload.landmarks, payload.image_width, payload.image_height)
        except ValueError:
            return self.result(target, "invalid_landmarks", start)
        with self.lock:
            return self.classify(feature, target, start)

    def attempt(self, payload):
        start = time.perf_counter()
        target = payload.target_label.strip().upper()
        total = len(payload.frames)
        def response(status, usable=0, accepted=0, votes=0, **values):
            result = self.result(target, status, start, **values)
            dynamic = result.recognition_method.startswith("trajectory_rules_")
            return AttemptPrediction(**result.model_dump(), client_attempt_id=payload.client_attempt_id,
                                     frames_received=total, frames_usable=usable, accepted_frames=None if dynamic else accepted,
                                     winning_votes=None if dynamic else votes, stability=None if dynamic else votes / total,
                                     aggregation_policy=result.recognition_method if dynamic else "experimental_3_accepted_min_75pct_all_frames_v1")
        if target in {"J", "Z"} and self.provider == "onnx_alphabet":
            from ai.motion import recognize_motion
            outcome = recognize_motion(payload, self.classifier, self.threshold)
            return response(**outcome)
        if target not in self.targets:
            return response("unsupported_target")
        features = []
        for frame in payload.frames:
            if frame.landmarks is not None:
                try:
                    features.append(self.feature_builder(frame.landmarks, payload.image_width, payload.image_height))
                except ValueError:
                    pass
        if not features:
            return response("no_hand" if all(f.landmarks is None for f in payload.frames) else "invalid_landmarks")
        with self.lock:
            probabilities = self.classifier.predict_proba(np.asarray(features))
        order = probabilities.argmax(axis=1)
        maxima = probabilities.max(axis=1)
        accepted = (maxima >= self.threshold) & np.isin(np.array(self.labels)[order], self.targets)
        counts = np.bincount(order[accepted], minlength=len(self.labels))
        winner = int(counts.argmax())
        votes = int(counts[winner])
        if votes == 0:
            return response("uncertain", usable=len(features))
        # Only confidences for the winning accepted label are averaged.
        score = float(maxima[accepted & (order == winner)].mean())
        valid = votes >= 3 and votes / total >= 0.75
        label = str(self.labels[winner])
        means = probabilities.mean(axis=0)
        top = np.argsort(means)[::-1][:3]
        return response("recognized" if valid else "uncertain", usable=len(features),
                        accepted=int(accepted.sum()), votes=votes,
                        predicted_label=label, score=score, valid=valid,
                        matches_target=(label == target) if valid else None,
                        top3=[Candidate(label=str(self.labels[i]), score=float(means[i])) for i in top])


@asynccontextmanager
async def lifespan(app):
    app.state.engine = Engine()
    yield


app = FastAPI(title="ASL Practice AI", version="0.4.0", lifespan=lifespan)
origins = os.environ.get("AI_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173,http://127.0.0.1:5173").split(",")
app.add_middleware(CORSMiddleware, allow_origins=[v.strip() for v in origins if v.strip()],
                   allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    # Invalid NaN/Infinity inputs must not crash JSON serialization of errors.
    return JSONResponse(status_code=422, content={"detail": [{"loc": list(e["loc"]), "type": e["type"], "msg": e["msg"]} for e in exc.errors()]})


@app.get("/health")
def health():
    return {"status": "ready", "model_version": app.state.engine.version,
            "model_provider": app.state.engine.provider, "pilot": True, "deployment_stage": "experimental"}


@app.get("/api/v1/ai/capabilities")
def capabilities():
    engine = app.state.engine
    return {"supported_targets": engine.targets + (["J", "Z"] if engine.provider == "onnx_alphabet" else []),
            "static_targets": engine.targets,
            "dynamic_letter_targets": ["J", "Z"] if engine.provider == "onnx_alphabet" else [],
            "dynamic_letter_method": "experimental_trajectory_rules_v4",
            "dynamic_letter_modes": {"J": "handshape_and_pinky_trajectory", "Z": "onnx_DZ_handshape_and_filtered_index_trajectory"},
            "dynamic_input_requirements": {"endpoint": "attempt", "min_frames": 15, "max_gap_ms": 250, "handedness_required": True}, "model_output_labels": engine.labels,
            "recognition_scope": "static_and_dynamic_alphabet" if engine.provider == "onnx_alphabet" else "static_alphabet", "dynamic_word_recognition": False,
            "input_types": ["landmarks", "attempt_frames"],
            "coordinate_space": "mediapipe_image_normalized_v1", "mirrored": False,
            "feature_version": engine.feature_version, "model_version": engine.version,
            "model_provider": engine.provider, "confidence_threshold": engine.threshold,
            "deployment_stage": "experimental", "skill_scoring": False,
            "error_diagnosis": False, "adaptive_feedback": False, "pilot": True,
            "recommended_for_integration_trial": engine.provider == "onnx_alphabet",
            "webcam_validation_complete": False,
            "pretrained_adoption_recommended": False,
            "legacy_score_semantics": "classifier confidence, never learner skill score"}


@app.post("/api/v1/ai/predict/landmarks", response_model=Prediction)
def landmark_prediction(payload: LandmarkRequest):
    return app.state.engine.landmarks(payload)


@app.post("/api/v1/ai/predict/attempt", response_model=AttemptPrediction)
def attempt_prediction(payload: AttemptRequest):
    return app.state.engine.attempt(payload)
