"""Versioned computer-vision processing pipeline primitives.

This module provides a deterministic, dependency-light implementation of the
storage contracts and worker orchestration needed by the future YOLO/BoT-SORT
runtime.  The detector input is intentionally injectable so tests and API
workers can persist real YOLO-style results without requiring model weights in
this package.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from hashlib import sha256
from math import hypot
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast
from uuid import uuid4

from basketvision_coach.court_calibration import ProjectedCourtPoint

RunStatus = Literal["queued", "running", "succeeded", "failed"]
Device = Literal["cpu", "mps"]

SUPPORTED_CLASSES = frozenset({"player", "ball", "hoop", "backboard", "referee"})

if TYPE_CHECKING:
    from basketvision_coach.shots import ShotCandidateDetectionSpec, ShotEvent


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Pixel-space object bounding box."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("bounding box width and height must be positive")

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)


@dataclass(frozen=True, slots=True)
class CourtPoint:
    """Optional calibrated court-space coordinates."""

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Version metadata recorded for a CV model."""

    name: str
    version: str
    weights_hash: str | None = None

    @classmethod
    def with_weights_bytes(cls, name: str, version: str, weights: bytes) -> ModelSpec:
        return cls(name=name, version=version, weights_hash=f"sha256:{sha256(weights).hexdigest()}")

    def to_json(self) -> dict[str, str]:
        data = {"name": self.name, "version": self.version}
        if self.weights_hash is not None:
            data["weights_hash"] = self.weights_hash
        return data


@dataclass(frozen=True, slots=True)
class DetectionCandidate:
    """YOLO-style detector output before persistence."""

    frame_idx: int
    ts_s: float
    class_name: str
    confidence: float
    bbox: BoundingBox
    court_coordinates: CourtPoint | None = None


@dataclass(frozen=True, slots=True)
class DetectionRecord:
    """Persisted detector output scoped to a processing run."""

    detection_id: str
    run_id: str
    frame_idx: int
    ts_s: float
    class_name: str
    confidence: float
    bbox: BoundingBox
    court_coordinates: CourtPoint | None = None


@dataclass(frozen=True, slots=True)
class TrackRecord:
    """Persisted per-frame tracking output scoped to a processing run."""

    track_row_id: str
    run_id: str
    source_detection_id: str
    track_id: int
    frame_idx: int
    ts_s: float
    class_name: str
    confidence: float
    bbox: BoundingBox
    court_coordinates: CourtPoint | None = None


@dataclass(frozen=True, slots=True)
class ProcessingRun:
    """Versioned processing run row."""

    run_id: str
    video_id: str
    run_type: str
    model_versions_json: dict[str, Any]
    params_json: dict[str, Any]
    status: RunStatus
    started_at: datetime
    finished_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class JobProgress:
    """API-facing job progress snapshot."""

    run_id: str
    stage: str
    frame_start: int | None
    frame_end: int | None
    processing_fps: float
    warnings: tuple[str, ...]
    status: RunStatus


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Run result returned by pipeline jobs."""

    run_id: str
    progress: JobProgress


@dataclass(frozen=True, slots=True)
class DetectionJobSpec:
    """Inputs for a detector processing run."""

    video_id: str
    model: ModelSpec
    confidence_threshold: float
    frame_detections: list[list[DetectionCandidate]]
    device: Device = "cpu"
    court_calibration_id: str | None = None


@dataclass(frozen=True, slots=True)
class TrackingJobSpec:
    """Inputs for a player tracking processing run."""

    video_id: str
    detection_run_id: str
    tracker_name: str = "bytetrack"
    tracker_config: dict[str, Any] = field(default_factory=lambda: {"max_distance_px": 50.0})
    court_calibration_id: str | None = None


@dataclass(frozen=True, slots=True)
class OverlayOptions:
    """UI toggles for CV overlays on video playback."""

    show_detection_boxes: bool = True
    show_track_trails: bool = True
    show_ball_path: bool = True

    def visible_layers(self) -> tuple[str, ...]:
        layers: list[str] = []
        if self.show_detection_boxes:
            layers.append("detection_boxes")
        if self.show_track_trails:
            layers.append("track_trails")
        if self.show_ball_path:
            layers.append("ball_path")
        return tuple(layers)

    def with_layer(self, layer: str, visible: bool) -> OverlayOptions:
        layer_to_field = {
            "detection_boxes": "show_detection_boxes",
            "track_trails": "show_track_trails",
            "ball_path": "show_ball_path",
        }
        try:
            return replace(self, **{layer_to_field[layer]: visible})
        except KeyError as exc:
            raise ValueError(f"unknown overlay layer: {layer}") from exc


@dataclass(frozen=True, slots=True)
class WorkerConnections:
    """Connection details used by a native worker against Dockerized services."""

    api_url: str
    redis_url: str
    shared_storage_path: str


@dataclass(frozen=True, slots=True)
class WorkerRuntime:
    """Worker runtime configuration, including macOS native DEVICE support."""

    device: Device
    connections: WorkerConnections

    @classmethod
    def from_env(cls, env: dict[str, str], connections: WorkerConnections) -> WorkerRuntime:
        raw_device = env.get("DEVICE", "cpu")
        if raw_device not in {"cpu", "mps"}:
            raise ValueError("DEVICE must be either 'cpu' or 'mps'")
        return cls(device=cast(Device, raw_device), connections=connections)

    @property
    def connects_to_dockerized_stack(self) -> bool:
        return all(
            (
                self.connections.api_url.startswith(("http://", "https://")),
                self.connections.redis_url.startswith("redis://"),
                bool(self.connections.shared_storage_path),
            )
        )


class CalibrationProjector(Protocol):
    """Projection contract supplied by the calibration API/service."""

    def project_point(
        self, *, video_id: str, image_x: float, image_y: float, ts_s: float
    ) -> ProjectedCourtPoint: ...


class InMemoryVisionStore:
    """Small repository that mirrors the eventual API database contract."""

    def __init__(self) -> None:
        self._runs: dict[str, ProcessingRun] = {}
        self._detections: dict[str, list[DetectionRecord]] = {}
        self._tracks: dict[str, list[TrackRecord]] = {}
        self._progress: dict[str, JobProgress] = {}
        self._shot_events: dict[str, ShotEvent] = {}

    def create_run(
        self,
        *,
        video_id: str,
        run_type: str,
        model_versions_json: dict[str, Any],
        params_json: dict[str, Any],
    ) -> ProcessingRun:
        run = ProcessingRun(
            run_id=str(uuid4()),
            video_id=video_id,
            run_type=run_type,
            model_versions_json=model_versions_json,
            params_json=params_json,
            status="running",
            started_at=datetime.now(UTC),
        )
        self._runs[run.run_id] = run
        return run

    def finish_run(self, run_id: str, status: RunStatus) -> ProcessingRun:
        run = self.get_run(run_id)
        finished = replace(run, status=status, finished_at=datetime.now(UTC))
        self._runs[run_id] = finished
        return finished

    def get_run(self, run_id: str) -> ProcessingRun:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise KeyError(f"unknown processing run: {run_id}") from exc

    def add_detection(self, run_id: str, detection: DetectionCandidate) -> DetectionRecord:
        record = DetectionRecord(
            detection_id=str(uuid4()),
            run_id=run_id,
            frame_idx=detection.frame_idx,
            ts_s=detection.ts_s,
            class_name=detection.class_name,
            confidence=detection.confidence,
            bbox=detection.bbox,
            court_coordinates=detection.court_coordinates,
        )
        self._detections.setdefault(run_id, []).append(record)
        return record

    def list_detections(self, run_id: str) -> list[DetectionRecord]:
        return list(self._detections.get(run_id, ()))

    def add_track(self, run_id: str, source: DetectionRecord, track_id: int) -> TrackRecord:
        record = TrackRecord(
            track_row_id=str(uuid4()),
            run_id=run_id,
            source_detection_id=source.detection_id,
            track_id=track_id,
            frame_idx=source.frame_idx,
            ts_s=source.ts_s,
            class_name=source.class_name,
            confidence=source.confidence,
            bbox=source.bbox,
            court_coordinates=source.court_coordinates,
        )
        self._tracks.setdefault(run_id, []).append(record)
        return record

    def list_tracks(self, run_id: str) -> list[TrackRecord]:
        return list(self._tracks.get(run_id, ()))

    def save_progress(self, progress: JobProgress) -> None:
        self._progress[progress.run_id] = progress

    def get_progress(self, run_id: str) -> JobProgress:
        return self._progress[run_id]

    def detect_shot_attempts(self, spec: ShotCandidateDetectionSpec) -> list[ShotEvent]:
        from basketvision_coach.shots import detect_shot_attempts

        events = detect_shot_attempts(
            detections=self.list_detections(spec.detection_run_id),
            tracks=self.list_tracks(spec.tracking_run_id),
            spec=spec,
        )
        for event in events:
            self._shot_events[event.event_id] = event
        return events

    def list_shot_events(self) -> list[ShotEvent]:
        return list(self._shot_events.values())

    def get_shot_event(self, event_id: str) -> ShotEvent:
        try:
            return self._shot_events[event_id]
        except KeyError as exc:
            raise KeyError(f"unknown shot event: {event_id}") from exc

    def update_shot_event(self, event: ShotEvent) -> ShotEvent:
        self.get_shot_event(event.event_id)
        self._shot_events[event.event_id] = event
        return event

    def lock_shot_event(self, event_id: str) -> ShotEvent:
        from dataclasses import replace

        event = self.get_shot_event(event_id)
        if event.status != "reviewed":
            raise ValueError("only reviewed shot events can be locked")
        return self.update_shot_event(replace(event, status="locked"))


class VisionPipeline:
    """Runs detector and tracker jobs while preserving versioned outputs."""

    def __init__(
        self, store: InMemoryVisionStore, calibration_service: CalibrationProjector | None = None
    ) -> None:
        self._store = store
        self._calibration_service = calibration_service

    def run_detection(self, spec: DetectionJobSpec) -> PipelineResult:
        start = perf_counter()
        run = self._store.create_run(
            video_id=spec.video_id,
            run_type="detection",
            model_versions_json={"detector": spec.model.to_json()},
            params_json={
                "confidence_threshold": spec.confidence_threshold,
                "device": spec.device,
                "court_calibration_id": spec.court_calibration_id,
                "classes": sorted(SUPPORTED_CLASSES),
            },
        )
        warnings: list[str] = []
        dropped_count = 0
        frame_indices: list[int] = []
        for frame in spec.frame_detections:
            for candidate in frame:
                if candidate.class_name not in SUPPORTED_CLASSES:
                    warnings.append(f"ignored unsupported class {candidate.class_name!r}")
                    continue
                frame_indices.append(candidate.frame_idx)
                if candidate.confidence < spec.confidence_threshold:
                    dropped_count += 1
                    continue
                self._store.add_detection(
                    run.run_id, self._with_projected_coordinates(spec.video_id, candidate)
                )
        if dropped_count:
            warnings.append(f"dropped {dropped_count} detections below confidence threshold")
        self._store.finish_run(run.run_id, "succeeded")
        progress = self._progress(
            run_id=run.run_id,
            stage="detection:complete",
            frame_indices=frame_indices,
            processed_frames=max(len(spec.frame_detections), 1),
            warnings=tuple(warnings),
            start=start,
        )
        return PipelineResult(run_id=run.run_id, progress=progress)

    def run_tracking(self, spec: TrackingJobSpec) -> PipelineResult:
        start = perf_counter()
        run = self._store.create_run(
            video_id=spec.video_id,
            run_type="tracking",
            model_versions_json={
                "tracker": {"name": spec.tracker_name, "version": "deterministic-1"}
            },
            params_json={
                "detection_run_id": spec.detection_run_id,
                "tracker_config": spec.tracker_config,
                "court_calibration_id": spec.court_calibration_id,
            },
        )
        detections = sorted(
            (
                d
                for d in self._store.list_detections(spec.detection_run_id)
                if d.class_name == "player"
            ),
            key=lambda item: (item.frame_idx, item.bbox.x, item.bbox.y),
        )
        max_distance = float(spec.tracker_config.get("max_distance_px", 50.0))
        next_track_id = 1
        active_centers: dict[int, tuple[float, float]] = {}
        frame_indices: list[int] = []
        for detection in detections:
            frame_indices.append(detection.frame_idx)
            center = detection.bbox.center
            track_id = _nearest_track(center, active_centers, max_distance)
            if track_id is None:
                track_id = next_track_id
                next_track_id += 1
            active_centers[track_id] = center
            self._store.add_track(run.run_id, detection, track_id)
        self._store.finish_run(run.run_id, "succeeded")
        progress = self._progress(
            run_id=run.run_id,
            stage="tracking:complete",
            frame_indices=frame_indices,
            processed_frames=len(set(frame_indices)) or 1,
            warnings=(),
            start=start,
        )
        return PipelineResult(run_id=run.run_id, progress=progress)

    def get_job_progress(self, run_id: str) -> JobProgress:
        return self._store.get_progress(run_id)

    def _with_projected_coordinates(
        self, video_id: str, candidate: DetectionCandidate
    ) -> DetectionCandidate:
        if candidate.court_coordinates is not None or self._calibration_service is None:
            return candidate
        projected = self._calibration_service.project_point(
            video_id=video_id,
            image_x=candidate.bbox.center[0],
            image_y=candidate.bbox.center[1],
            ts_s=candidate.ts_s,
        )
        return replace(
            candidate,
            court_coordinates=CourtPoint(x=projected.court_x, y=projected.court_y),
        )

    def _progress(
        self,
        *,
        run_id: str,
        stage: str,
        frame_indices: list[int],
        processed_frames: int,
        warnings: tuple[str, ...],
        start: float,
    ) -> JobProgress:
        elapsed = max(perf_counter() - start, 1e-9)
        progress = JobProgress(
            run_id=run_id,
            stage=stage,
            frame_start=min(frame_indices) if frame_indices else None,
            frame_end=max(frame_indices) if frame_indices else None,
            processing_fps=processed_frames / elapsed,
            warnings=warnings,
            status=self._store.get_run(run_id).status,
        )
        self._store.save_progress(progress)
        return progress


def _nearest_track(
    center: tuple[float, float], active_centers: dict[int, tuple[float, float]], max_distance: float
) -> int | None:
    best_track_id: int | None = None
    best_distance = max_distance
    for track_id, prior_center in active_centers.items():
        distance = hypot(center[0] - prior_center[0], center[1] - prior_center[1])
        if distance <= best_distance:
            best_track_id = track_id
            best_distance = distance
    return best_track_id
