from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from basketvision_coach.api import create_app
from basketvision_coach.db import Base
from basketvision_coach.demo_data import generate_demo_frames
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
        return FfmpegMetadata(duration_s=4.0, fps=25.0, width=1280, height=720)

    return VideoIngestService(
        session_factory=session_factory,
        data_root=tmp_path,
        transcode_runner=fake_transcode,
        create_hls=False,
    )


def upload_video(client: TestClient) -> int:
    game_id = client.post("/api/games", json={"name": "Demo game"}).json()["id"]
    resp = client.post(
        f"/api/games/{game_id}/video",
        files={"file": ("clip.mp4", b"\x00\x00\x00\x18ftypmp42rest", "video/mp4")},
    )
    video_id: int = resp.json()["id"]
    return video_id


def test_generate_demo_frames_has_players_and_ball() -> None:
    frames = generate_demo_frames(duration_s=2.0, fps=10.0, num_players=5)
    assert len(frames) == 20
    first = frames[0]
    classes = [c.class_name for c in first]
    assert classes.count("player") == 5
    assert classes.count("ball") == 1
    # frame indices and timestamps are consistent with fps
    assert {c.frame_idx for c in first} == {0}


def test_demo_frames_default_on_nonpositive_inputs() -> None:
    frames = generate_demo_frames(duration_s=0.0, fps=0.0)
    assert len(frames) == int(8.0 * 25.0)


def test_video_read_exposes_original_url(tmp_path: Path) -> None:
    client = TestClient(create_app(service=make_service(tmp_path)))
    video_id = upload_video(client)
    body = client.get(f"/api/videos/{video_id}").json()
    assert body["original_url"].endswith("/original.mp4")
    assert client.get(body["original_url"]).status_code == 200


def test_demo_analysis_runs_detection_and_tracking(tmp_path: Path) -> None:
    client = TestClient(create_app(service=make_service(tmp_path)))
    video_id = upload_video(client)

    resp = client.post(f"/api/videos/{video_id}/demo-analysis?duration_s=2.0")
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["detection_run_id"] != body["tracking_run_id"]
    assert body["frame_count"] == int(2.0 * 25.0)
    assert body["source_width"] == 1000.0

    detections = client.get(f"/api/runs/{body['detection_run_id']}/detections").json()
    assert any(d["class_name"] == "ball" for d in detections)
    assert any(d["class_name"] == "player" for d in detections)

    tracks = client.get(f"/api/runs/{body['tracking_run_id']}/tracks").json()
    assert len(tracks) > 0
    # tracking assigns stable integer track ids to players
    assert all(t["class_name"] == "player" for t in tracks)
    assert max(t["track_id"] for t in tracks) >= 1


def test_demo_analysis_missing_video_returns_404(tmp_path: Path) -> None:
    client = TestClient(create_app(service=make_service(tmp_path)))
    assert client.post("/api/videos/999/demo-analysis").status_code == 404
