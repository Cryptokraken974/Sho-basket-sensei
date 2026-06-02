"""Real video detection + tracking analysis.

This wires actual computer-vision inference (YOLO detection + ByteTrack
tracking) into the same persistence/overlay contracts the synthetic demo uses.

The heavy runtime dependencies (``ultralytics``, ``opencv-python``, ``torch``)
are intentionally **not** part of the package's declared dependencies: they are
large, platform/GPU specific, and cannot be resolved in every environment. The
concrete analyzer therefore imports them lazily and raises
:class:`AnalyzerUnavailable` with install guidance when they are missing. All
orchestration here is dependency-injected so it can be exercised with a fake
analyzer in tests and CI without the models installed.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from basketvision_coach.cv_pipeline import (
    BoundingBox,
    DetectionCandidate,
    InMemoryVisionStore,
    JobProgress,
)

# COCO class ids produced by stock YOLO weights, mapped to our class names.
COCO_PERSON = 0
COCO_SPORTS_BALL = 32
COCO_TO_CLASS = {COCO_PERSON: "player", COCO_SPORTS_BALL: "ball"}

CV_INSTALL_HINT = (
    "Real analysis needs the CV runtime. Install it with "
    "`pip install ultralytics opencv-python-headless` (GPU/MPS recommended)."
)


class AnalyzerUnavailable(RuntimeError):
    """Raised when the optional CV runtime is not installed."""


def resolve_device(explicit: str | None = None) -> str:
    """Pick the torch device: explicit arg, then ``DEVICE`` env, then auto-detect.

    Auto-detection prefers CUDA, then Apple Silicon's MPS (Metal), then CPU, so
    it runs on an M-series Mac out of the box. Falls back to ``cpu`` when torch
    is not installed (the CV runtime is optional).
    """

    requested = explicit if explicit is not None else os.environ.get("DEVICE")
    if requested:
        return requested.strip().lower()
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():  # pragma: no cover - needs a CUDA gpu
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():  # pragma: no cover - needs Apple gpu
        return "mps"
    return "cpu"  # pragma: no cover - needs torch installed


@dataclass(frozen=True, slots=True)
class DetectedObject:
    class_name: str
    confidence: float
    # Top-left x, y plus width, height in the analyzed frame's pixel space.
    x: float
    y: float
    width: float
    height: float
    track_id: int | None = None


@dataclass(frozen=True, slots=True)
class FrameDetections:
    frame_idx: int
    ts_s: float
    objects: tuple[DetectedObject, ...] = ()


@dataclass(frozen=True, slots=True)
class VideoMeta:
    width: int
    height: int
    fps: float


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    detection_run_id: str
    tracking_run_id: str
    source_width: int
    source_height: int
    fps: float
    frame_count: int
    object_count: int
    engine: str


class VideoAnalyzer(Protocol):
    """Contract for a detector+tracker over a local video file."""

    engine: str

    def probe(self, video_path: Path) -> VideoMeta: ...

    def detect_and_track(self, video_path: Path) -> Iterable[FrameDetections]: ...


def run_analysis(
    store: InMemoryVisionStore,
    *,
    video_id: int,
    video_path: Path,
    analyzer: VideoAnalyzer,
) -> AnalysisResult:
    """Run ``analyzer`` over ``video_path`` and persist detection + track runs."""

    meta = analyzer.probe(video_path)
    detection_run = store.create_run(
        video_id=str(video_id),
        run_type="detection",
        model_versions_json={"detector": {"engine": analyzer.engine}},
        params_json={"source": video_path.name},
    )
    tracking_run = store.create_run(
        video_id=str(video_id),
        run_type="tracking",
        model_versions_json={"tracker": {"engine": "bytetrack"}},
        params_json={"detection_run_id": detection_run.run_id},
    )

    frame_count = 0
    object_count = 0
    for frame in analyzer.detect_and_track(video_path):
        frame_count = max(frame_count, frame.frame_idx + 1)
        for obj in frame.objects:
            record = store.add_detection(
                detection_run.run_id,
                DetectionCandidate(
                    frame_idx=frame.frame_idx,
                    ts_s=frame.ts_s,
                    class_name=obj.class_name,
                    confidence=obj.confidence,
                    bbox=BoundingBox(obj.x, obj.y, obj.width, obj.height),
                ),
            )
            object_count += 1
            if obj.track_id is not None and obj.class_name == "player":
                store.add_track(tracking_run.run_id, record, obj.track_id)

    store.finish_run(detection_run.run_id, "succeeded")
    store.finish_run(tracking_run.run_id, "succeeded")
    last_frame = frame_count - 1 if frame_count else None
    for run_id, stage in (
        (detection_run.run_id, "detection:complete"),
        (tracking_run.run_id, "tracking:complete"),
    ):
        store.save_progress(
            JobProgress(
                run_id=run_id,
                stage=stage,
                frame_start=0 if frame_count else None,
                frame_end=last_frame,
                processing_fps=meta.fps,
                warnings=(),
                status="succeeded",
            )
        )

    return AnalysisResult(
        detection_run_id=detection_run.run_id,
        tracking_run_id=tracking_run.run_id,
        source_width=meta.width,
        source_height=meta.height,
        fps=meta.fps,
        frame_count=frame_count,
        object_count=object_count,
        engine=analyzer.engine,
    )


@dataclass
class YoloByteTrackAnalyzer:
    """Stock-YOLO detection + ByteTrack tracking via the Ultralytics runtime.

    Uses pretrained COCO weights, so it needs no training to find players
    (``person``) and the ball (``sports ball``).
    """

    model_name: str = "yolov8n.pt"
    confidence: float = 0.25
    device: str | None = None
    engine: str = field(default="yolov8+bytetrack")
    _fps: float = field(default=25.0, init=False, repr=False)

    def _load_model(self) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - exercised only with deps
            raise AnalyzerUnavailable(CV_INSTALL_HINT) from exc
        return YOLO(self.model_name)

    def probe(self, video_path: Path) -> VideoMeta:  # pragma: no cover - needs opencv
        try:
            import cv2
        except ImportError as exc:
            raise AnalyzerUnavailable(CV_INSTALL_HINT) from exc
        capture = cv2.VideoCapture(str(video_path))
        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
        finally:
            capture.release()
        self._fps = fps
        return VideoMeta(width=width, height=height, fps=fps)

    def detect_and_track(
        self, video_path: Path
    ) -> Iterable[FrameDetections]:  # pragma: no cover - needs models
        model = self._load_model()
        results = model.track(
            source=str(video_path),
            stream=True,
            persist=True,
            classes=list(COCO_TO_CLASS),
            conf=self.confidence,
            tracker="bytetrack.yaml",
            device=resolve_device(self.device),
            verbose=False,
        )
        for frame_idx, result in enumerate(results):
            objects: list[DetectedObject] = []
            boxes = getattr(result, "boxes", None)
            if boxes is not None:
                for box in _iter_boxes(boxes):
                    class_name = COCO_TO_CLASS.get(box.cls)
                    if class_name is None:
                        continue
                    objects.append(
                        DetectedObject(
                            class_name=class_name,
                            confidence=box.conf,
                            x=box.x,
                            y=box.y,
                            width=box.width,
                            height=box.height,
                            track_id=box.track_id,
                        )
                    )
            yield FrameDetections(
                frame_idx=frame_idx,
                ts_s=frame_idx / self._fps if self._fps else 0.0,
                objects=tuple(objects),
            )


@dataclass(frozen=True, slots=True)
class _Box:
    cls: int
    conf: float
    x: float
    y: float
    width: float
    height: float
    track_id: int | None


def _iter_boxes(boxes: object) -> Iterable[_Box]:  # pragma: no cover - needs models
    cls = boxes.cls.tolist()  # type: ignore[attr-defined]
    conf = boxes.conf.tolist()  # type: ignore[attr-defined]
    xywh = boxes.xywh.tolist()  # type: ignore[attr-defined]
    ids = boxes.id.tolist() if boxes.id is not None else [None] * len(cls)  # type: ignore[attr-defined]
    for class_id, confidence, (cx, cy, w, h), track_id in zip(cls, conf, xywh, ids, strict=True):
        yield _Box(
            cls=int(class_id),
            conf=float(confidence),
            x=float(cx) - float(w) / 2,
            y=float(cy) - float(h) / 2,
            width=float(w),
            height=float(h),
            track_id=int(track_id) if track_id is not None else None,
        )


def load_default_analyzer() -> VideoAnalyzer:
    """Return the default real analyzer, or raise :class:`AnalyzerUnavailable`."""

    analyzer = YoloByteTrackAnalyzer()
    # Surface a missing runtime eagerly with a friendly message.
    analyzer._load_model()
    return analyzer
