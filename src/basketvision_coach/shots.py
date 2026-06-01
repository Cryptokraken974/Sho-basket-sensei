"""Semi-automatic shot candidate detection and human review primitives."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from math import hypot
from typing import Literal, Protocol
from uuid import uuid4

from basketvision_coach.cv_pipeline import DetectionRecord, TrackRecord

ShotEventStatus = Literal["needs_review", "reviewed", "locked", "rejected"]
ShotEventSource = Literal["model", "manual"]


class ShotResult(StrEnum):
    """Human-confirmed shot outcome."""

    MADE = "make"
    MISSED = "miss"


class ShotValue(StrEnum):
    """Human-confirmed shot value."""

    TWO = "2PT"
    THREE = "3PT"


@dataclass(frozen=True, slots=True)
class ClipPreview:
    """Small video window that lets reviewers inspect a candidate."""

    video_id: str
    start_s: float
    end_s: float


@dataclass(frozen=True, slots=True)
class ShotCandidateDetectionSpec:
    """Inputs for detecting model-sourced shot attempt review candidates."""

    video_id: str
    detection_run_id: str
    tracking_run_id: str
    frame_rate: float
    track_team_candidates: dict[int, str] | None = None
    clip_padding_s: float = 2.0
    min_confidence: float = 0.55


@dataclass(frozen=True, slots=True)
class ShotEvent:
    """A shot_attempt event that is inert until a human reviews it."""

    event_id: str
    video_id: str
    event_type: Literal["shot_attempt"]
    timestamp_s: float
    frame_idx: int
    confidence: float
    court_x: float
    court_y: float
    source: ShotEventSource
    status: ShotEventStatus
    clip_preview: ClipPreview
    shooter_candidate: str | None = None
    team_candidate: str | None = None
    shooter_id: str | None = None
    team_id: str | None = None
    result: ShotResult | None = None
    shot_value: ShotValue | None = None
    period: int | None = None
    shot_type: str | None = None
    reviewed_by: str | None = None
    zone: str | None = None


@dataclass(frozen=True, slots=True)
class ShotReviewDecision:
    """Human-entered final shot fields."""

    reviewer_id: str
    shooter_id: str
    team_id: str
    result: ShotResult
    shot_value: ShotValue
    court_x: float
    court_y: float
    period: int | None = None
    shot_type: str | None = None


class _ShotEventStore(Protocol):
    def get_shot_event(self, event_id: str) -> ShotEvent: ...

    def update_shot_event(self, event: ShotEvent) -> ShotEvent: ...


class ShotReviewPanel:
    """Review UI façade for accepting, editing, and rejecting candidates."""

    def __init__(self, store: _ShotEventStore) -> None:
        self._store = store

    def accept(self, event_id: str, decision: ShotReviewDecision) -> ShotEvent:
        return self._store.update_shot_event(
            _apply_review(self._store.get_shot_event(event_id), decision)
        )

    def edit(self, event_id: str, decision: ShotReviewDecision) -> ShotEvent:
        event = self._store.get_shot_event(event_id)
        if event.status == "rejected":
            event = replace(event, status="needs_review")
        return self._store.update_shot_event(_apply_review(event, decision))

    def reject(self, event_id: str, reviewer_id: str) -> ShotEvent:
        event = self._store.get_shot_event(event_id)
        return self._store.update_shot_event(
            replace(
                event,
                status="rejected",
                result=None,
                shot_value=None,
                reviewed_by=reviewer_id,
                shooter_id=None,
                team_id=None,
                shot_type=None,
                zone=None,
            )
        )


def detect_shot_attempts(
    *,
    detections: list[DetectionRecord],
    tracks: list[TrackRecord],
    spec: ShotCandidateDetectionSpec,
) -> list[ShotEvent]:
    """Find model shot candidates from motion and nearby hoop/player context."""

    _ = (
        spec.frame_rate
    )  # frame indexes are already timestamped; retained in the contract for workers.
    balls = sorted(
        (item for item in detections if item.class_name == "ball"), key=lambda item: item.frame_idx
    )
    hoops = sorted(
        (item for item in detections if item.class_name == "hoop"), key=lambda item: item.frame_idx
    )
    if len(balls) < 3 or not hoops:
        return []

    events: list[ShotEvent] = []
    consumed_frames: set[int] = set()
    for index in range(1, len(balls) - 1):
        previous_ball = balls[index - 1]
        ball = balls[index]
        next_ball = balls[index + 1]
        if ball.frame_idx in consumed_frames:
            continue
        continuity = _track_continuity_score(previous_ball, ball, next_ball)
        if continuity == 0:
            continue
        motion = _shot_motion_score(previous_ball, ball, next_ball)
        if motion == 0:
            continue
        hoop = _nearest_hoop(ball, hoops)
        hoop_score = _hoop_region_score(ball, hoop)
        if hoop_score == 0:
            continue
        court_score = _court_location_score(ball, hoop)
        release_ball = balls[max(0, index - 3)]
        shooter_track, proximity_score = _nearest_shooter(release_ball, tracks)
        confidence = round(
            min(
                0.99,
                0.27
                + 0.23 * motion
                + 0.23 * hoop_score
                + 0.12 * court_score
                + 0.12 * proximity_score
                + 0.06 * continuity,
            ),
            2,
        )
        if confidence < spec.min_confidence:
            continue
        court_point = ball.court_coordinates or hoop.court_coordinates
        if court_point is None:
            continue
        shooter_candidate = f"track:{shooter_track.track_id}" if shooter_track is not None else None
        team_candidate = (
            spec.track_team_candidates.get(shooter_track.track_id)
            if spec.track_team_candidates is not None and shooter_track is not None
            else None
        )
        events.append(
            ShotEvent(
                event_id=str(uuid4()),
                video_id=spec.video_id,
                event_type="shot_attempt",
                timestamp_s=ball.ts_s,
                frame_idx=ball.frame_idx,
                confidence=confidence,
                court_x=court_point.x,
                court_y=court_point.y,
                source="model",
                status="needs_review",
                clip_preview=ClipPreview(
                    video_id=spec.video_id,
                    start_s=max(0.0, round(ball.ts_s - spec.clip_padding_s, 3)),
                    end_s=round(ball.ts_s + spec.clip_padding_s, 3),
                ),
                shooter_candidate=shooter_candidate,
                team_candidate=team_candidate,
                zone=_shot_zone(court_point.x, court_point.y),
            )
        )
        consumed_frames.update({previous_ball.frame_idx, ball.frame_idx, next_ball.frame_idx})
    return events


def reviewed_shots_for_chart(events: list[ShotEvent]) -> list[ShotEvent]:
    """Default shot chart filter: only human-reviewed or locked shots affect reports."""

    return [event for event in events if event.status in {"reviewed", "locked"}]


def export_shot_chart_rows(events: list[ShotEvent]) -> list[dict[str, object]]:
    """Export reviewed chart rows with the MVP reporting columns."""

    return [
        {
            "player": event.shooter_id,
            "team": event.team_id,
            "period": event.period,
            "timestamp": event.timestamp_s,
            "court_x": event.court_x,
            "court_y": event.court_y,
            "zone": event.zone or _shot_zone(event.court_x, event.court_y),
            "2PT/3PT": event.shot_value.value if event.shot_value else None,
            "make/miss": event.result.value if event.result else None,
            "source": event.source,
            "status": event.status,
        }
        for event in reviewed_shots_for_chart(events)
    ]


def _apply_review(event: ShotEvent, decision: ShotReviewDecision) -> ShotEvent:
    return replace(
        event,
        status="reviewed",
        shooter_id=decision.shooter_id,
        team_id=decision.team_id,
        result=decision.result,
        shot_value=decision.shot_value,
        court_x=decision.court_x,
        court_y=decision.court_y,
        period=decision.period,
        shot_type=decision.shot_type,
        reviewed_by=decision.reviewer_id,
        zone=_shot_zone(decision.court_x, decision.court_y),
    )


def _track_continuity_score(
    previous_ball: DetectionRecord, ball: DetectionRecord, next_ball: DetectionRecord
) -> float:
    if previous_ball.frame_idx >= ball.frame_idx or ball.frame_idx >= next_ball.frame_idx:
        return 0.0
    gap = max(next_ball.frame_idx - previous_ball.frame_idx, 1)
    return max(0.0, 1.0 - (gap - 2) * 0.25)


def _shot_motion_score(
    previous_ball: DetectionRecord, ball: DetectionRecord, next_ball: DetectionRecord
) -> float:
    prev_x, prev_y = previous_ball.bbox.center
    ball_x, ball_y = ball.bbox.center
    next_x, next_y = next_ball.bbox.center
    horizontal_progress = ball_x >= prev_x and next_x >= ball_x
    apex_or_descent = (ball_y < prev_y and next_y > ball_y) or (prev_y < ball_y < next_y)
    if not (horizontal_progress and apex_or_descent):
        return 0.0
    vertical_change = abs(prev_y - ball_y) + abs(next_y - ball_y)
    return min(1.0, vertical_change / 80.0)


def _nearest_hoop(ball: DetectionRecord, hoops: list[DetectionRecord]) -> DetectionRecord:
    return min(hoops, key=lambda hoop: abs(hoop.frame_idx - ball.frame_idx))


def _hoop_region_score(ball: DetectionRecord, hoop: DetectionRecord) -> float:
    ball_x, ball_y = ball.bbox.center
    hoop_x, hoop_y = hoop.bbox.center
    distance = hypot(ball_x - hoop_x, ball_y - hoop_y)
    region_radius = max(60.0, hoop.bbox.width * 2.0)
    if distance > region_radius:
        return 0.0
    return max(0.0, 1.0 - distance / region_radius)


def _court_location_score(ball: DetectionRecord, hoop: DetectionRecord) -> float:
    if ball.court_coordinates is None:
        return 0.0
    if hoop.court_coordinates is None:
        return 0.5
    distance = hypot(
        ball.court_coordinates.x - hoop.court_coordinates.x,
        ball.court_coordinates.y - hoop.court_coordinates.y,
    )
    return max(0.0, min(1.0, 1.0 - distance / 12.0))


def _nearest_shooter(
    ball: DetectionRecord, tracks: list[TrackRecord]
) -> tuple[TrackRecord | None, float]:
    eligible = [track for track in tracks if track.frame_idx <= ball.frame_idx]
    if not eligible:
        return None, 0.0
    by_track: dict[int, TrackRecord] = {}
    for track in eligible:
        existing = by_track.get(track.track_id)
        if existing is None or track.frame_idx > existing.frame_idx:
            by_track[track.track_id] = track
    ball_x, ball_y = ball.bbox.center
    nearest = min(
        by_track.values(),
        key=lambda track: (
            hypot(ball_x - track.bbox.center[0], ball_y - track.bbox.center[1])
            + abs(ball.frame_idx - track.frame_idx) * 20
        ),
    )
    distance = hypot(ball_x - nearest.bbox.center[0], ball_y - nearest.bbox.center[1])
    return nearest, max(0.0, min(1.0, 1.0 - distance / 180.0))


def _shot_zone(court_x: float, court_y: float) -> str:
    if court_y <= 5 and court_x >= 22:
        return "right-corner-3"
    if court_y <= 5 and court_x <= 3:
        return "left-corner-3"
    if court_x >= 19 or court_x <= 6:
        return "wing-3"
    if court_y >= 19:
        return "paint"
    return "midrange"
