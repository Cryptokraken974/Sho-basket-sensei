from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from basketvision_coach.api import create_app
from basketvision_coach.db import Base
from basketvision_coach.video import FfmpegMetadata, VideoIngestService


def make_service(tmp_path: Path) -> VideoIngestService:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    def fake_transcode(source: Path, proxy: Path, hls_dir: Path | None) -> FfmpegMetadata:
        proxy.write_bytes(b"proxy mp4")
        return FfmpegMetadata(duration_s=1.0, fps=30.0, width=1280, height=720)

    return VideoIngestService(
        session_factory=session_factory,
        data_root=tmp_path,
        transcode_runner=fake_transcode,
        create_hls=False,
    )


def client_for(tmp_path: Path) -> TestClient:
    return TestClient(create_app(service=make_service(tmp_path)))


def square_point_pairs() -> list[dict[str, object]]:
    landmarks = active_landmarks()
    coords = [
        (100.0, 100.0, 0.0, 0.0),
        (500.0, 100.0, 94.0, 0.0),
        (500.0, 400.0, 94.0, 50.0),
        (100.0, 400.0, 0.0, 50.0),
    ]
    return [
        {
            "landmark": landmarks[i],
            "image_x": ix,
            "image_y": iy,
            "court_x": cx,
            "court_y": cy,
        }
        for i, (ix, iy, cx, cy) in enumerate(coords)
    ]


def active_landmarks() -> list[str]:
    from basketvision_coach.court_calibration import active_supported_landmark_labels

    return list(active_supported_landmark_labels())


def test_landmarks_endpoint_lists_supported_labels(tmp_path: Path) -> None:
    client = client_for(tmp_path)
    response = client.get("/api/landmarks")
    assert response.status_code == 200
    labels = response.json()
    assert "center_circle_center" in labels


def test_calibration_create_list_and_project(tmp_path: Path) -> None:
    client = client_for(tmp_path)
    payload = {"point_pairs": square_point_pairs(), "created_by": "coach"}
    created = client.post("/api/videos/1/calibrations", json=payload)
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["video_id"] == "1"
    assert len(body["homography_json"]) == 3

    listed = client.get("/api/videos/1/calibrations")
    assert listed.status_code == 200
    assert len(listed.json()) == 1

    projected = client.post(
        "/api/videos/1/project",
        json={"image_x": 100.0, "image_y": 100.0, "ts_s": 0.0},
    )
    assert projected.status_code == 200
    pj = projected.json()
    assert abs(pj["court_x"] - 0.0) < 1e-6
    assert abs(pj["court_y"] - 0.0) < 1e-6


def test_calibration_requires_four_points(tmp_path: Path) -> None:
    client = client_for(tmp_path)
    payload = {"point_pairs": square_point_pairs()[:3], "created_by": "coach"}
    response = client.post("/api/videos/1/calibrations", json=payload)
    assert response.status_code == 422  # pydantic min_length


def test_project_without_calibration_returns_404(tmp_path: Path) -> None:
    client = client_for(tmp_path)
    response = client.post("/api/videos/9/project", json={"image_x": 1.0, "image_y": 1.0})
    assert response.status_code == 404


def test_detection_and_tracking_job_flow(tmp_path: Path) -> None:
    client = client_for(tmp_path)
    detection_payload = {
        "model": {"name": "yolo", "version": "1.0"},
        "confidence_threshold": 0.25,
        "frames": [
            [
                {
                    "frame_idx": 0,
                    "ts_s": 0.0,
                    "class_name": "player",
                    "confidence": 0.9,
                    "bbox": {"x": 10.0, "y": 10.0, "width": 20.0, "height": 40.0},
                },
                {
                    "frame_idx": 0,
                    "ts_s": 0.0,
                    "class_name": "ball",
                    "confidence": 0.1,
                    "bbox": {"x": 50.0, "y": 50.0, "width": 5.0, "height": 5.0},
                },
            ],
            [
                {
                    "frame_idx": 1,
                    "ts_s": 0.033,
                    "class_name": "player",
                    "confidence": 0.8,
                    "bbox": {"x": 12.0, "y": 11.0, "width": 20.0, "height": 40.0},
                }
            ],
        ],
    }
    det = client.post("/api/videos/1/detections", json=detection_payload)
    assert det.status_code == 201, det.text
    det_body = det.json()
    detection_run_id = det_body["run_id"]
    assert det_body["progress"]["status"] == "succeeded"
    # one low-confidence ball detection should be dropped -> warning present
    assert any("confidence" in w for w in det_body["progress"]["warnings"])

    track = client.post(
        "/api/videos/1/tracking",
        json={"detection_run_id": detection_run_id},
    )
    assert track.status_code == 201, track.text
    tracking_run_id = track.json()["run_id"]

    progress = client.get(f"/api/runs/{tracking_run_id}/progress")
    assert progress.status_code == 200
    assert progress.json()["stage"] == "tracking:complete"


def test_run_progress_missing_returns_404(tmp_path: Path) -> None:
    client = client_for(tmp_path)
    response = client.get("/api/runs/does-not-exist/progress")
    assert response.status_code == 404


def test_detection_rejects_bad_device(tmp_path: Path) -> None:
    client = client_for(tmp_path)
    response = client.post(
        "/api/videos/1/detections",
        json={"model": {"name": "yolo", "version": "1.0"}, "device": "cuda", "frames": []},
    )
    assert response.status_code == 400
