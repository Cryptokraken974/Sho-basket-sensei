from __future__ import annotations

from basketvision_coach.cv_pipeline import (
    BoundingBox,
    CourtPoint,
    DetectionCandidate,
    DetectionJobSpec,
    InMemoryVisionStore,
    ModelSpec,
    OverlayOptions,
    TrackingJobSpec,
    VisionPipeline,
    WorkerConnections,
    WorkerRuntime,
)


def test_detection_job_creates_versioned_run_and_appends_detections() -> None:
    store = InMemoryVisionStore()
    pipeline = VisionPipeline(store=store)
    model = ModelSpec(name="yolo-basket", version="2026.06.1", weights_hash="sha256:abc")
    frames = [
        [
            DetectionCandidate(
                frame_idx=0,
                ts_s=0.0,
                class_name="player",
                confidence=0.91,
                bbox=BoundingBox(x=10, y=20, width=30, height=40),
                court_coordinates=CourtPoint(x=4.5, y=6.0),
            ),
            DetectionCandidate(
                frame_idx=0,
                ts_s=0.0,
                class_name="ball",
                confidence=0.74,
                bbox=BoundingBox(x=100, y=22, width=8, height=8),
            ),
        ]
    ]

    first = pipeline.run_detection(
        DetectionJobSpec(
            video_id="game-1",
            model=model,
            confidence_threshold=0.5,
            frame_detections=frames,
            device="cpu",
            court_calibration_id="court-42",
        )
    )
    second = pipeline.run_detection(
        DetectionJobSpec(
            video_id="game-1",
            model=ModelSpec(name="yolo-basket", version="2026.07.0"),
            confidence_threshold=0.6,
            frame_detections=frames,
            device="mps",
        )
    )

    assert first.run_id != second.run_id
    assert store.get_run(first.run_id).run_type == "detection"
    assert store.get_run(first.run_id).model_versions_json == {
        "detector": {"name": "yolo-basket", "version": "2026.06.1", "weights_hash": "sha256:abc"}
    }
    assert store.get_run(first.run_id).params_json["confidence_threshold"] == 0.5
    assert store.get_run(first.run_id).params_json["court_calibration_id"] == "court-42"
    assert store.get_run(first.run_id).status == "succeeded"
    assert store.get_run(first.run_id).finished_at is not None
    assert [d.class_name for d in store.list_detections(first.run_id)] == ["player", "ball"]
    assert len(store.list_detections(second.run_id)) == 2
    assert len(store.list_detections(first.run_id)) == 2, "new model runs preserve earlier outputs"


def test_tracking_job_consumes_detection_run_and_writes_stable_player_tracks() -> None:
    store = InMemoryVisionStore()
    pipeline = VisionPipeline(store=store)
    detection = pipeline.run_detection(
        DetectionJobSpec(
            video_id="game-1",
            model=ModelSpec(name="yolo-basket", version="2026.06.1"),
            confidence_threshold=0.5,
            frame_detections=[
                [
                    DetectionCandidate(0, 0.0, "player", 0.9, BoundingBox(10, 10, 20, 40)),
                    DetectionCandidate(0, 0.0, "player", 0.88, BoundingBox(100, 10, 20, 40)),
                    DetectionCandidate(0, 0.0, "ball", 0.77, BoundingBox(50, 4, 8, 8)),
                ],
                [
                    DetectionCandidate(1, 0.04, "player", 0.93, BoundingBox(12, 10, 20, 40)),
                    DetectionCandidate(1, 0.04, "player", 0.84, BoundingBox(103, 10, 20, 40)),
                ],
            ],
        )
    )

    tracking = pipeline.run_tracking(
        TrackingJobSpec(
            video_id="game-1",
            detection_run_id=detection.run_id,
            tracker_name="bytetrack",
            tracker_config={"max_distance_px": 30, "lost_buffer": 2},
        )
    )

    tracks = store.list_tracks(tracking.run_id)
    assert [track.class_name for track in tracks] == ["player", "player", "player", "player"]
    assert [track.track_id for track in tracks if track.frame_idx == 0] == [1, 2]
    assert [track.track_id for track in tracks if track.frame_idx == 1] == [1, 2]
    assert store.get_run(tracking.run_id).params_json["detection_run_id"] == detection.run_id
    assert store.get_run(tracking.run_id).params_json["tracker_config"] == {
        "max_distance_px": 30,
        "lost_buffer": 2,
    }


def test_job_progress_includes_stage_bounds_fps_warnings_and_status() -> None:
    store = InMemoryVisionStore()
    pipeline = VisionPipeline(store=store)

    result = pipeline.run_detection(
        DetectionJobSpec(
            video_id="game-1",
            model=ModelSpec(name="yolo-basket", version="2026.06.1"),
            confidence_threshold=0.95,
            frame_detections=[
                [DetectionCandidate(7, 0.28, "referee", 0.5, BoundingBox(1, 2, 3, 4))]
            ],
        )
    )

    progress = pipeline.get_job_progress(result.run_id)
    assert progress.stage == "detection:complete"
    assert progress.frame_start == 7
    assert progress.frame_end == 7
    assert progress.processing_fps > 0
    assert progress.warnings == ("dropped 1 detections below confidence threshold",)
    assert progress.status == "succeeded"


def test_overlay_options_can_show_and_hide_detections_tracks_and_ball_path() -> None:
    options = OverlayOptions(
        show_detection_boxes=True,
        show_track_trails=False,
        show_ball_path=True,
    )

    assert options.visible_layers() == ("detection_boxes", "ball_path")
    assert options.with_layer("track_trails", True).visible_layers() == (
        "detection_boxes",
        "track_trails",
        "ball_path",
    )
    assert options.with_layer("detection_boxes", False).visible_layers() == ("ball_path",)


def test_worker_runtime_supports_cpu_and_mps_with_dockerized_dependencies() -> None:
    connections = WorkerConnections(
        api_url="http://localhost:8000",
        redis_url="redis://localhost:6379/0",
        shared_storage_path="/Volumes/basketvision/shared",
    )

    cpu_runtime = WorkerRuntime.from_env({"DEVICE": "cpu"}, connections)
    mps_runtime = WorkerRuntime.from_env({"DEVICE": "mps"}, connections)

    assert cpu_runtime.device == "cpu"
    assert mps_runtime.device == "mps"
    assert mps_runtime.connects_to_dockerized_stack is True
