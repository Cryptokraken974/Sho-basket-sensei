"""BasketVision Coach public API."""

from basketvision_coach.coach import CoachFeedback, ShotObservation, assess_shot
from basketvision_coach.cv_pipeline import (
    BoundingBox,
    CourtPoint,
    DetectionCandidate,
    DetectionJobSpec,
    DetectionRecord,
    InMemoryVisionStore,
    JobProgress,
    ModelSpec,
    OverlayOptions,
    PipelineResult,
    ProcessingRun,
    TrackingJobSpec,
    TrackRecord,
    VisionPipeline,
    WorkerConnections,
    WorkerRuntime,
)

__all__ = [
    "BoundingBox",
    "CoachFeedback",
    "CourtPoint",
    "DetectionCandidate",
    "DetectionJobSpec",
    "DetectionRecord",
    "InMemoryVisionStore",
    "JobProgress",
    "ModelSpec",
    "OverlayOptions",
    "PipelineResult",
    "ProcessingRun",
    "ShotObservation",
    "TrackRecord",
    "TrackingJobSpec",
    "VisionPipeline",
    "WorkerConnections",
    "WorkerRuntime",
    "assess_shot",
]
