"""FastAPI boundary for versioned static ASL recognition model adapters."""

from contextlib import asynccontextmanager
from functools import lru_cache
import os
import threading
import time
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from errors.error_detection_static import (
    FEATURE_COUNT,
    load_templates,
    main as detect_feature_errors,
)
from features import feature_extraction as fe
from feedback.feedback_manager import evaluate_attempt
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from ai.gesturesign import GestureSign, label_key
from ai.onnx_alphabet import OnnxAlphabetClassifier


# kiểm tra dữ liệu đầu vào
class LandmarkRequest(BaseModel):  # nhận diện tư thế tĩnh
    target_label: str = Field(
        min_length=1, max_length=16
    )  # chữ người dùng đang tập
    landmarks: (
        Annotated[
            list[tuple[FiniteFloat, FiniteFloat, FiniteFloat]],
            Field(min_length=21, max_length=21),
        ]
        | None
    )  # danh sách các landmark
    image_width: int = Field(gt=0, le=8192)
    image_height: int = Field(gt=0, le=8192)
    coordinate_space: Literal["mediapipe_image_normalized_v1"] = (
        "mediapipe_image_normalized_v1"
    )
    mirrored: Literal[False] = False


class Candidate(BaseModel):
    label: str
    score: float


class Frame(BaseModel):  # chuỗi nhiều Frame để nhận diện tư thế động J/Z
    handedness: Literal["Left", "Right"] | None = None
    timestamp_ms: float = Field(ge=0, allow_inf_nan=False)
    landmarks: (
        Annotated[
            list[tuple[FiniteFloat, FiniteFloat, FiniteFloat]],
            Field(min_length=21, max_length=21),
        ]
        | None
    )


class AttemptRequest(BaseModel):  # thông tin lượt tập và danh sách Frame
    client_attempt_id: str = Field(min_length=1, max_length=80)
    target_label: str = Field(min_length=1, max_length=16)
    image_width: int = Field(gt=0, le=8192)
    image_height: int = Field(gt=0, le=8192)
    coordinate_space: Literal["mediapipe_image_normalized_v1"] = (
        "mediapipe_image_normalized_v1"
    )
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
    status: Literal[
        "recognized", "uncertain", "no_hand", "invalid_landmarks", "unsupported_target"
    ]
    target_label: str
    predicted_label: str | None = None  # chữ dự đoán bởi mô hình
    score: float | None = Field(
        default=None,
        description="Legacy alias of model_confidence; NOT a learner skill score",
    )
    predicted_sign: str | None = None
    model_confidence: float | None = Field(
        default=None,
        description="Classifier output, not a calibrated skill or correctness probability",
    )
    valid: bool = False
    matches_target: bool | None = (
        Field(  # dự đoán khớp với chữ người dùng đang tập
            default=None,
            description="Accepted predicted label matches target; null when assessment is unavailable",
        )
    )
    top3: list[Candidate] = Field(default_factory=list)  # 3 nhãn có điểm cao nhất
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


# trong model đã có chuẩn hóa landmark nên nhận từ FE vào đó r landmark được chuẩn hóa
class Engine:  # khởi tạo mô hình nhận diện chữ cái bằng onnx_alphabet
    def __init__(self):
        self.provider = "onnx_alphabet"  # model đã được huấn luyện
        requested = os.environ.get("AI_MODEL_PROVIDER", "onnx_alphabet")
        if requested != self.provider:
            raise RuntimeError("This API-only package supports onnx_alphabet only")
        self.classifier = OnnxAlphabetClassifier(
            threshold=float(os.environ.get("AI_CONFIDENCE_THRESHOLD", "0.5"))
        )
        self.labels = self.classifier.labels
        self.targets = list(self.classifier.supported_targets)
        self.feature_builder = self.classifier.features
        self.feature_version = self.classifier.feature_version
        self.threshold = self.classifier.threshold
        self.version = self.classifier.version + "+motion-v4"
        self.lock = threading.Lock()

    def result(self, target, status, start, **values):
        return Prediction(
            request_id=str(uuid4()),
            model_version=self.version,
            feature_version=self.feature_version,
            target_label=target,
            status=status,
            processing_ms=(time.perf_counter() - start) * 1000,
            predicted_sign=values.get("predicted_label"),
            model_confidence=values.get("score"),
            **values,
        )

    # nhận diện 1 frame tĩnh
    # chuẩn hóa chữ mục tiêu thành chữ hoa và so sánh với chữ dự đoán bởi mô hình
    def classify(self, feature, target, start):
        scores = self.classifier.predict_proba(feature[None])[
            0
        ]  # dự đoán điểm số cho các nhãn
        order = np.argsort(scores)[::-1]
        label = str(self.labels[order[0]])
        score = float(scores[order[0]])
        valid = score >= self.threshold and label in self.targets
        # trả về kết quả dự đoán với thông tin chi tiết về nhãn dự đoán, điểm số, và các nhãn hàng đầu
        return self.result(
            target,
            "recognized" if valid else "uncertain",
            start,
            predicted_label=label,
            score=score,
            valid=valid,
            matches_target=(label == target) if valid else None,
            top3=[
                Candidate(label=str(self.labels[i]), score=float(scores[i]))
                for i in order[:3]
            ],
        )

    def landmarks(self, payload):
        start = time.perf_counter()
        target = payload.target_label.strip().upper()
        # trường hợp ngoại lê khi nhận diện
        if target not in self.targets:
            return self.result(
                target, "unsupported_target", start
            )  # chữ k được hỗ trợ
        if payload.landmarks is None:
            return self.result(target, "no_hand", start)
        try:
            feature = self.feature_builder(
                payload.landmarks, payload.image_width, payload.image_height
            )
        except ValueError:
            return self.result(target, "invalid_landmarks", start)
        with self.lock:
            return self.classify(feature, target, start)

    def attempt(self, payload):  # nhận diện nhiều Frame cho J/Z
        start = time.perf_counter()
        target = payload.target_label.strip().upper()
        total = len(payload.frames)

        def response(status, usable=0, accepted=0, votes=0, **values):
            result = self.result(target, status, start, **values)
            dynamic = result.recognition_method.startswith("trajectory_rules_")
            return AttemptPrediction(
                **result.model_dump(),
                client_attempt_id=payload.client_attempt_id,
                frames_received=total,
                frames_usable=usable,
                accepted_frames=None if dynamic else accepted,
                winning_votes=None if dynamic else votes,
                stability=None if dynamic else votes / total,
                aggregation_policy=(
                    result.recognition_method
                    if dynamic
                    else "experimental_3_accepted_min_75pct_all_frames_v1"
                ),
            )

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
                    features.append(
                        self.feature_builder(
                            frame.landmarks, payload.image_width, payload.image_height
                        )
                    )
                except ValueError:
                    pass
        if not features:
            return response(
                "no_hand"
                if all(f.landmarks is None for f in payload.frames)
                else "invalid_landmarks"
            )
        with self.lock:
            probabilities = self.classifier.predict_proba(np.asarray(features))
        order = probabilities.argmax(axis=1)
        maxima = probabilities.max(axis=1)
        accepted = (maxima >= self.threshold) & np.isin(
            np.array(self.labels)[order], self.targets
        )
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
        return response(
            "recognized" if valid else "uncertain",
            usable=len(features),
            accepted=int(accepted.sum()),
            votes=votes,
            predicted_label=label,
            score=score,
            valid=valid,
            matches_target=(label == target) if valid else None,
            top3=[
                Candidate(label=str(self.labels[i]), score=float(means[i])) for i in top
            ],
        )


# sửa thêm 3 hàm dưới đi từ file feature extraction
class Features:
    def feature_extract(payload: LandmarkRequest):
        if payload.landmarks is None:
            return np.full(FEATURE_COUNT, np.nan, dtype=float)
        try:
            normalized_landmarks = fe.preprocess_hand(
                payload.landmarks,
                payload.image_width,
                payload.image_height,
            )
            return fe.extract_static_phonological_features(normalized_landmarks)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_landmarks", "message": str(exc)},
            ) from exc


class Error:
    @staticmethod
    @lru_cache(maxsize=4)
    def _load_templates(path, modified_time_ns):
        return load_templates(path)

    @staticmethod
    def error_detection(features, target):
        if not np.isfinite(features).all():
            return {
                "score": None,
                "components": {
                    group: {"score": None, "deviations": []}
                    for group in ("handshape", "orientation", "location")
                },
                "errors": ["no_hand"],
            }

        template_path = os.environ.get("AI_ERROR_TEMPLATE_DATA")
        if not template_path:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "template_data_unavailable",
                    "message": "Set AI_ERROR_TEMPLATE_DATA to a prebuilt template .npz file.",
                },
            )

        path = Path(template_path)
        if not path.is_file():
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "template_data_not_found",
                    "message": f"Template file does not exist: {path}",
                },
            )

        try:
            templates = Error._load_templates(
                str(path.resolve()), path.stat().st_mtime_ns
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=500,
                detail={"code": "invalid_template_data", "message": str(exc)},
            ) from exc

        try:
            return detect_feature_errors(features, target, templates)
        except KeyError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "template_target_not_found",
                    "message": f"No reference template for target {target!r}.",
                },
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_features", "message": str(exc)},
            ) from exc


@asynccontextmanager
async def lifespan(app):  # khởi tạo Engine và cấu hình FastAPI liên quan đến FE
    app.state.engine = Engine()
    app.state.word_engine = GestureSign()
    yield


app = FastAPI(title="ASL Practice AI", version="0.4.0", lifespan=lifespan)
origins = os.environ.get(
    "AI_CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://127.0.0.1:5173",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[v.strip() for v in origins if v.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    # Invalid NaN/Infinity inputs must not crash JSON serialization of errors.
    return JSONResponse(
        status_code=422,
        content={
            "detail": [
                {"loc": list(e["loc"]), "type": e["type"], "msg": e["msg"]}
                for e in exc.errors()
            ]
        },
    )


@app.get("/health")
def health():
    return {
        "status": "ready",
        "model_version": app.state.engine.version,
        "model_provider": app.state.engine.provider,
        "pilot": True,
        "deployment_stage": "experimental",
    }


@app.get("/api/v1/ai/capabilities")
def capabilities():  # trả về thông tin cho FE/BE biết service này hỗ trợ gì
    engine = app.state.engine
    return {  # trả về JSOn gồm
        "supported_targets": engine.targets
        + (["J", "Z"] if engine.provider == "onnx_alphabet" else []),
        "static_targets": engine.targets,
        "dynamic_letter_targets": (
            ["J", "Z"] if engine.provider == "onnx_alphabet" else []
        ),
        "dynamic_letter_method": "experimental_trajectory_rules_v4",
        "dynamic_letter_modes": {
            "J": "handshape_and_pinky_trajectory",
            "Z": "onnx_DZ_handshape_and_filtered_index_trajectory",
        },
        "dynamic_input_requirements": {
            "endpoint": "attempt",
            "min_frames": 15,
            "max_gap_ms": 250,
            "handedness_required": True,
        },
        "model_output_labels": engine.labels,
        "recognition_scope": (
            "static_and_dynamic_alphabet"
            if engine.provider == "onnx_alphabet"
            else "static_alphabet"
        ),
        "dynamic_word_recognition": True,
        "word_capabilities_endpoint": "/api/v1/ai/words/capabilities",
        "input_types": ["landmarks", "attempt_frames"],
        "coordinate_space": "mediapipe_image_normalized_v1",
        "mirrored": False,
        "feature_version": engine.feature_version,
        "model_version": engine.version,
        "model_provider": engine.provider,
        "confidence_threshold": engine.threshold,
        "deployment_stage": "experimental",
        # các giới hạn như chưa hỗ trợ nhận diện từ, chấm điểm kỹ năng, chẩn đoán lỗi hay xác nhận đã kiểm thử webcam.
        "skill_scoring": False,  # chấm điểm kĩ năng người dùng
        "error_diagnosis": False,  # detect còn thử nghiệm, chưa đủ tin cậy
        "error_detection_endpoint": "/api/v1/ai/errors/detect",
        "adaptive_feedback": True,  # cấp phản hồi theo lịch sử lỗi
        "feedback_endpoint": "/api/v1/ai/feedback",
        "feedback_levels_implemented": [1],  # cấp 2/3 hiện trả text như cấp 1
        "pilot": True,  # phản hồi từ người dùng để cải thiện sản phẩm
        "recommended_for_integration_trial": engine.provider == "onnx_alphabet",
        "webcam_validation_complete": False,  # kiểm tra webcam đã hoàn tất
        "pretrained_adoption_recommended": False,
        "legacy_score_semantics": "classifier confidence, never learner skill score",
    }


# FE gửi landmarks vừa lấy được lên BE thông qua API
@app.post("/api/v1/ai/predict/landmarks", response_model=Prediction)
def landmark_prediction(payload: LandmarkRequest):
    return app.state.engine.landmarks(payload)


def _not_evaluated(status):
    """Kết quả detection khi lượt này không đánh giá được (không tính đúng/sai)."""
    return {
        "status": status,
        "score": None,
        "components": {
            group: {"score": None, "deviations": []}
            for group in ("handshape", "orientation", "location")
        },
        "errors": [status],
    }


def run_error_detection(payload: LandmarkRequest):
    """landmarks -> đặc trưng -> so với mẫu chuẩn. Trả về (chữ mục tiêu, detection)."""
    target = payload.target_label.strip().upper()
    if target not in app.state.engine.targets:  # J/Z và nhãn lạ chưa có mẫu tĩnh
        return target, _not_evaluated("unsupported_target")
    if payload.landmarks is None:
        return target, _not_evaluated("no_hand")
    try:
        features = Features.feature_extract(payload)
    except HTTPException:
        return target, _not_evaluated("invalid_landmarks")
    detection = Error.error_detection(features, target)
    detection.setdefault(
        "status", "no_hand" if detection.get("errors") else "evaluated"
    )
    return target, detection


@app.post("/api/v1/ai/errors/detect", response_model=dict)
def detect_errors(payload: LandmarkRequest):
    target, detection = run_error_detection(payload)
    return {"target_label": target, **detection}


class ComponentHistory(BaseModel):
    """Trạng thái lịch sử của một thành phần (H/L/O/M), do BE lưu và gửi lại."""

    model_config = ConfigDict(extra="forbid")
    level: int | None = Field(default=None, ge=1, le=3)
    wrong_streak: int = Field(default=0, ge=0)
    correct_streak: int = Field(default=0, ge=0)
    total_wrong: int = Field(default=0, ge=0)
    total_correct: int = Field(default=0, ge=0)
    last_result: Literal["wrong", "correct"] | None = None
    needs_review: bool = False


class FeedbackRequest(LandmarkRequest):
    mode: Literal["fixed", "adaptive"] = "adaptive"
    history: dict[Literal["H", "L", "O", "M"], ComponentHistory] = Field(
        default_factory=dict,
        description=(
            "Lịch sử của đúng (người học, chữ mục tiêu, phiên) do lượt trước "
            "trả về trong trường history; gửi {} cho lượt đầu tiên."
        ),
    )
    max_feedbacks: int | None = Field(default=None, ge=1, le=4)


# Lịch sử do BE lưu: AI Server không giữ trạng thái giữa các request.
@app.post("/api/v1/ai/feedback", response_model=dict)
def feedback(payload: FeedbackRequest):
    target, detection = run_error_detection(payload)
    history = {k: v.model_dump() for k, v in payload.history.items()}
    out = evaluate_attempt(
        history,
        target,
        detection,
        mode=payload.mode,
        version="V1",
        max_feedbacks=payload.max_feedbacks,
    )
    return {
        "request_id": str(uuid4()),
        "target_label": target,
        "mode": payload.mode,
        **out,
        "detection": detection,
    }


# riêng cho J/Z
@app.post("/api/v1/ai/predict/attempt", response_model=AttemptPrediction)
def attempt_prediction(payload: AttemptRequest):
    return app.state.engine.attempt(payload)


Point = tuple[Annotated[FiniteFloat, Field(ge=-100, le=100)],
              Annotated[FiniteFloat, Field(ge=-100, le=100)],
              Annotated[FiniteFloat, Field(ge=-100, le=100)]]
Hand = Annotated[list[Point | None], Field(min_length=21, max_length=21)]


class WordFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")
    left_hand: Hand | None = None
    right_hand: Hand | None = None
    pose: Annotated[list[Point | None], Field(min_length=33, max_length=33)] | None = None
    face: Annotated[list[Point | None], Field(min_length=468, max_length=468)] | None = None


class WordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    client_attempt_id: str = Field(min_length=1, max_length=80)
    target_label: str = Field(min_length=1, max_length=80)
    fps: FiniteFloat = Field(ge=1, le=120)
    coordinate_space: Literal["mediapipe_holistic_image_normalized_v1"] = "mediapipe_holistic_image_normalized_v1"
    mirrored: Literal[False] = False
    frames: list[WordFrame] = Field(min_length=1, max_length=600)

    @model_validator(mode="after")
    def capture_duration(self):
        if (len(self.frames) - 1) / self.fps > 5.1:
            raise ValueError("Word clip must span at most 5 seconds")
        return self


class WordPrediction(Prediction):
    client_attempt_id: str
    frames_received: int
    frames_usable: int


@app.get("/api/v1/ai/words/capabilities")
def word_capabilities():
    engine = app.state.word_engine
    return {
        "model_version": engine.meta["version"],
        "feature_version": str(engine.meta["featureVersion"]),
        "supported_targets": [row["label"] for row in engine.labels if row["shipped"]],
        "coordinate_space": "mediapipe_holistic_image_normalized_v1",
        "mirrored": False,
        "max_duration_ms": 5000,
        "confidence_threshold": engine.meta["calibration"]["notSureBelow"],
    }


@app.post("/api/v1/ai/predict/word", response_model=WordPrediction)
def word_prediction(payload: WordRequest):
    started = time.perf_counter()
    engine = app.state.word_engine
    target = engine.targets.get(label_key(payload.target_label))
    status, usable, guesses = ("unsupported_target", 0, []) if target is None else engine.predict(
        payload.frames, payload.fps
    )
    valid = status == "recognized"
    label, score = guesses[0] if valid else (None, None)
    return WordPrediction(
        request_id=str(uuid4()), client_attempt_id=payload.client_attempt_id,
        target_label=target or payload.target_label, status=status, valid=valid,
        predicted_label=label, predicted_sign=label, score=score, model_confidence=score,
        matches_target=(label == target) if valid else None,
        top3=[Candidate(label=name, score=prob) for name, prob in guesses],
        model_version=engine.meta["version"], feature_version=str(engine.meta["featureVersion"]),
        recognition_method="gesturesign_onnx", frames_received=len(payload.frames),
        frames_usable=usable, processing_ms=(time.perf_counter() - started) * 1000,
    )
