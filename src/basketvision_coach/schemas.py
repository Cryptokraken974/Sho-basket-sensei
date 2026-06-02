"""Pydantic request/response schemas for the BasketVision Coach HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GameCreate(BaseModel):
    name: str


class SampleIngest(BaseModel):
    filename: str


class GameRead(BaseModel):
    id: int
    name: str
    created_at: str

    model_config = ConfigDict(from_attributes=True)


class VideoRead(BaseModel):
    id: int
    game_id: int
    state: str
    original_path: str
    proxy_path: str | None
    proxy_url: str | None
    original_url: str | None
    hls_path: str | None
    fps: float | None
    width: int | None
    height: int | None
    duration_s: float | None
    error_message: str | None
    ts_mapping: str


# --- Court calibration ---------------------------------------------------------


class CalibrationPointPairIn(BaseModel):
    landmark: str
    image_x: float
    image_y: float
    court_x: float
    court_y: float


class CalibrationCreate(BaseModel):
    point_pairs: list[CalibrationPointPairIn] = Field(min_length=4)
    valid_from_s: float = 0.0
    valid_to_s: float | None = None
    created_by: str


class CalibrationRead(BaseModel):
    calibration_id: str
    video_id: str
    image_points_json: list[dict[str, float | str]]
    court_points_json: list[dict[str, float | str]]
    homography_json: list[list[float]]
    valid_from_s: float
    valid_to_s: float | None
    created_by: str


class ProjectRequest(BaseModel):
    image_x: float
    image_y: float
    ts_s: float = 0.0


class ProjectedRead(BaseModel):
    calibration_id: str
    court_x: float
    court_y: float


class ProjectablePointIn(BaseModel):
    label: str
    image_x: float
    image_y: float


class OverlayRequest(BaseModel):
    ts_s: float = 0.0
    test_points: list[ProjectablePointIn] = Field(default_factory=list)


# --- CV pipeline (detection / tracking) ----------------------------------------


class BoundingBoxIn(BaseModel):
    x: float
    y: float
    width: float
    height: float


class CourtPointIn(BaseModel):
    x: float
    y: float


class DetectionCandidateIn(BaseModel):
    frame_idx: int
    ts_s: float
    class_name: str
    confidence: float
    bbox: BoundingBoxIn
    court_coordinates: CourtPointIn | None = None


class ModelSpecIn(BaseModel):
    name: str
    version: str
    weights_hash: str | None = None


class DetectionRunCreate(BaseModel):
    model: ModelSpecIn
    confidence_threshold: float = 0.25
    device: str = "cpu"
    court_calibration_id: str | None = None
    frames: list[list[DetectionCandidateIn]] = Field(default_factory=list)

    model_config = ConfigDict(protected_namespaces=())


class TrackingRunCreate(BaseModel):
    detection_run_id: str
    tracker_name: str = "bytetrack"
    tracker_config: dict[str, Any] = Field(default_factory=lambda: {"max_distance_px": 50.0})
    court_calibration_id: str | None = None


class JobProgressRead(BaseModel):
    run_id: str
    stage: str
    frame_start: int | None
    frame_end: int | None
    processing_fps: float
    warnings: list[str]
    status: str


class RunRead(BaseModel):
    run_id: str
    progress: JobProgressRead


class DemoAnalysisRead(BaseModel):
    detection_run_id: str
    tracking_run_id: str
    source_width: float
    source_height: float
    fps: float
    frame_count: int


class AnalysisRead(BaseModel):
    detection_run_id: str
    tracking_run_id: str
    source_width: float
    source_height: float
    fps: float
    frame_count: int
    object_count: int
    engine: str


class ClickPromptIn(BaseModel):
    object_label: str
    frame_idx: int
    x: float
    y: float


class ClickTrackRequest(BaseModel):
    prompts: list[ClickPromptIn] = Field(min_length=1)


# --- Manual review / event tagging --------------------------------------------


class ReviewEventCreateIn(BaseModel):
    type: str
    start_s: float
    end_s: float | None = None
    team_id: str | None = None
    player_id: str | None = None
    confidence: float = 1.0
    status: str = "reviewed"


class ReviewEventUpdate(BaseModel):
    type: str | None = None
    start_s: float | None = None
    end_s: float | None = None
    team_id: str | None = None
    player_id: str | None = None
    status: str | None = None


class ReviewEventRead(BaseModel):
    id: int
    video_id: str
    type: str
    start_s: float
    end_s: float
    team_id: str | None
    player_id: str | None
    confidence: float
    source: str
    status: str
    reviewed: bool


class ReportRead(BaseModel):
    total_tagged: int
    official_event_count: int
    excluded_unreviewed_event_count: int
    by_type: dict[str, int]
    by_status: dict[str, int]
    makes: int
    misses: int
    shooting_pct: float | None
    missing_data: dict[str, str]
    event_types: list[str]


class BoundingBoxRead(BaseModel):
    x: float
    y: float
    width: float
    height: float


class CourtPointRead(BaseModel):
    x: float
    y: float


class DetectionRecordRead(BaseModel):
    detection_id: str
    run_id: str
    frame_idx: int
    ts_s: float
    class_name: str
    confidence: float
    bbox: BoundingBoxRead
    court_coordinates: CourtPointRead | None = None

    model_config = ConfigDict(from_attributes=True)


class TrackRecordRead(BaseModel):
    track_row_id: str
    run_id: str
    track_id: int
    frame_idx: int
    ts_s: float
    class_name: str
    confidence: float
    bbox: BoundingBoxRead
    court_coordinates: CourtPointRead | None = None

    model_config = ConfigDict(from_attributes=True)


# --- Identity review (roster / track identities) -------------------------------


class TeamCreate(BaseModel):
    name: str
    id: str | None = None
    jersey_color: str | None = None


class TeamRead(BaseModel):
    id: str
    name: str
    jersey_color: str | None


class PlayerCreate(BaseModel):
    name: str
    team_id: str
    id: str | None = None
    jersey_number: str | None = None


class PlayerRead(BaseModel):
    id: str
    name: str
    team_id: str
    jersey_number: str | None


class TrackIdentityAssign(BaseModel):
    track_id: str
    player_id: str | None = None
    team_id: str | None = None
    user: str
    start_s: float | None = None
    end_s: float | None = None
    confidence: float = 1.0
    source: str = "manual"


class TrackIdentityRead(BaseModel):
    id: int | None
    track_id: str
    player_id: str | None
    team_id: str | None
    confidence: float
    start_s: float | None
    end_s: float | None
    source: str
    reviewed: bool
