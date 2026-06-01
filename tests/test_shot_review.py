from __future__ import annotations

from basketvision_coach.cv_pipeline import (
    BoundingBox,
    CourtPoint,
    DetectionCandidate,
    DetectionJobSpec,
    InMemoryVisionStore,
    ModelSpec,
    TrackingJobSpec,
    VisionPipeline,
)
from basketvision_coach.shots import (
    ShotCandidateDetectionSpec,
    ShotResult,
    ShotReviewDecision,
    ShotReviewPanel,
    ShotValue,
    export_shot_chart_rows,
    reviewed_shots_for_chart,
)


def _seed_processed_video() -> tuple[InMemoryVisionStore, str, str]:
    store = InMemoryVisionStore()
    pipeline = VisionPipeline(store=store)
    detections = pipeline.run_detection(
        DetectionJobSpec(
            video_id="game-7",
            model=ModelSpec(name="yolo-basket", version="2026.06.1"),
            confidence_threshold=0.5,
            frame_detections=[
                [
                    DetectionCandidate(
                        0, 0.00, "player", 0.95, BoundingBox(40, 220, 35, 70), CourtPoint(4.0, 20.0)
                    ),
                    DetectionCandidate(
                        0,
                        0.00,
                        "player",
                        0.94,
                        BoundingBox(250, 220, 35, 70),
                        CourtPoint(12.0, 18.0),
                    ),
                    DetectionCandidate(
                        0, 0.00, "hoop", 0.99, BoundingBox(315, 90, 45, 25), CourtPoint(14.0, 25.0)
                    ),
                    DetectionCandidate(
                        0, 0.00, "ball", 0.90, BoundingBox(72, 198, 10, 10), CourtPoint(5.0, 20.5)
                    ),
                ],
                [
                    DetectionCandidate(
                        1, 0.04, "player", 0.95, BoundingBox(42, 220, 35, 70), CourtPoint(4.1, 20.0)
                    ),
                    DetectionCandidate(
                        1,
                        0.04,
                        "player",
                        0.94,
                        BoundingBox(250, 220, 35, 70),
                        CourtPoint(12.0, 18.0),
                    ),
                    DetectionCandidate(
                        1, 0.04, "hoop", 0.99, BoundingBox(315, 90, 45, 25), CourtPoint(14.0, 25.0)
                    ),
                    DetectionCandidate(
                        1, 0.04, "ball", 0.91, BoundingBox(92, 158, 10, 10), CourtPoint(6.0, 21.4)
                    ),
                ],
                [
                    DetectionCandidate(
                        2, 0.08, "player", 0.95, BoundingBox(44, 220, 35, 70), CourtPoint(4.2, 20.0)
                    ),
                    DetectionCandidate(
                        2,
                        0.08,
                        "player",
                        0.94,
                        BoundingBox(250, 220, 35, 70),
                        CourtPoint(12.0, 18.0),
                    ),
                    DetectionCandidate(
                        2, 0.08, "hoop", 0.99, BoundingBox(315, 90, 45, 25), CourtPoint(14.0, 25.0)
                    ),
                    DetectionCandidate(
                        2, 0.08, "ball", 0.92, BoundingBox(180, 82, 10, 10), CourtPoint(10.5, 23.5)
                    ),
                ],
                [
                    DetectionCandidate(
                        3, 0.12, "player", 0.95, BoundingBox(46, 220, 35, 70), CourtPoint(4.3, 20.0)
                    ),
                    DetectionCandidate(
                        3,
                        0.12,
                        "player",
                        0.94,
                        BoundingBox(250, 220, 35, 70),
                        CourtPoint(12.0, 18.0),
                    ),
                    DetectionCandidate(
                        3, 0.12, "hoop", 0.99, BoundingBox(315, 90, 45, 25), CourtPoint(14.0, 25.0)
                    ),
                    DetectionCandidate(
                        3, 0.12, "ball", 0.91, BoundingBox(304, 95, 10, 10), CourtPoint(13.8, 24.2)
                    ),
                ],
                [
                    DetectionCandidate(
                        4, 0.16, "player", 0.95, BoundingBox(48, 220, 35, 70), CourtPoint(4.4, 20.0)
                    ),
                    DetectionCandidate(
                        4,
                        0.16,
                        "player",
                        0.94,
                        BoundingBox(250, 220, 35, 70),
                        CourtPoint(12.0, 18.0),
                    ),
                    DetectionCandidate(
                        4, 0.16, "hoop", 0.99, BoundingBox(315, 90, 45, 25), CourtPoint(14.0, 25.0)
                    ),
                    DetectionCandidate(
                        4, 0.16, "ball", 0.89, BoundingBox(326, 120, 10, 10), CourtPoint(14.1, 24.6)
                    ),
                ],
            ],
        )
    )
    tracking = pipeline.run_tracking(
        TrackingJobSpec(
            video_id="game-7",
            detection_run_id=detections.run_id,
            tracker_config={"max_distance_px": 35},
        )
    )
    return store, detections.run_id, tracking.run_id


def test_pipeline_generates_needs_review_shot_attempt_candidates_from_motion_context() -> None:
    store, detection_run_id, tracking_run_id = _seed_processed_video()

    events = store.detect_shot_attempts(
        ShotCandidateDetectionSpec(
            video_id="game-7",
            detection_run_id=detection_run_id,
            tracking_run_id=tracking_run_id,
            frame_rate=25.0,
            track_team_candidates={1: "home", 2: "away"},
        )
    )

    assert len(events) == 1
    event = events[0]
    assert event.event_type == "shot_attempt"
    assert event.source == "model"
    assert event.status == "needs_review"
    assert event.result is None, "make/miss is not final before human review"
    assert event.confidence >= 0.70
    assert event.shooter_candidate == "track:1"
    assert event.team_candidate == "home"
    assert event.court_x == 13.8
    assert event.court_y == 24.2
    assert event.timestamp_s == 0.12
    assert event.clip_preview.video_id == "game-7"
    assert event.clip_preview.start_s < event.timestamp_s < event.clip_preview.end_s


def test_review_panel_requires_human_make_miss_and_supports_accept_edit_reject() -> None:
    store, detection_run_id, tracking_run_id = _seed_processed_video()
    event = store.detect_shot_attempts(
        ShotCandidateDetectionSpec("game-7", detection_run_id, tracking_run_id, frame_rate=25.0)
    )[0]
    panel = ShotReviewPanel(store)

    accepted = panel.accept(
        event.event_id,
        ShotReviewDecision(
            reviewer_id="coach-1",
            shooter_id="p12",
            team_id="home",
            result=ShotResult.MADE,
            shot_value=ShotValue.THREE,
            court_x=22.1,
            court_y=4.0,
            period=2,
            shot_type="catch-and-shoot",
        ),
    )

    assert accepted.status == "reviewed"
    assert accepted.shooter_id == "p12"
    assert accepted.team_id == "home"
    assert accepted.result is ShotResult.MADE
    assert accepted.shot_value is ShotValue.THREE
    assert accepted.court_x == 22.1
    assert accepted.period == 2
    assert accepted.shot_type == "catch-and-shoot"
    assert accepted.reviewed_by == "coach-1"

    edited = panel.edit(
        event.event_id,
        ShotReviewDecision(
            reviewer_id="coach-2",
            shooter_id="p99",
            team_id="away",
            result=ShotResult.MISSED,
            shot_value=ShotValue.TWO,
            court_x=5.0,
            court_y=6.0,
        ),
    )
    assert edited.shooter_id == "p99"
    assert edited.result is ShotResult.MISSED
    assert edited.status == "reviewed"

    rejected = panel.reject(event.event_id, reviewer_id="coach-2")
    assert rejected.status == "rejected"
    assert rejected.result is None


def test_shot_chart_defaults_to_reviewed_or_locked_shots_and_exports_required_columns() -> None:
    store, detection_run_id, tracking_run_id = _seed_processed_video()
    pending = store.detect_shot_attempts(
        ShotCandidateDetectionSpec("game-7", detection_run_id, tracking_run_id, frame_rate=25.0)
    )[0]
    panel = ShotReviewPanel(store)

    assert reviewed_shots_for_chart(store.list_shot_events()) == []

    reviewed = panel.accept(
        pending.event_id,
        ShotReviewDecision(
            reviewer_id="coach-1",
            shooter_id="p12",
            team_id="home",
            result=ShotResult.MADE,
            shot_value=ShotValue.THREE,
            court_x=23.0,
            court_y=3.5,
            period=4,
        ),
    )
    locked = store.lock_shot_event(reviewed.event_id)

    chart_rows = reviewed_shots_for_chart(store.list_shot_events())
    assert chart_rows == [locked]

    exported = export_shot_chart_rows(chart_rows)
    assert exported == [
        {
            "player": "p12",
            "team": "home",
            "period": 4,
            "timestamp": 0.12,
            "court_x": 23.0,
            "court_y": 3.5,
            "zone": "right-corner-3",
            "2PT/3PT": "3PT",
            "make/miss": "make",
            "source": "model",
            "status": "locked",
        }
    ]
