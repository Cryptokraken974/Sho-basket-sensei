from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from basketvision_coach.analysis import (
    DetectedObject,
    FrameDetections,
    VideoMeta,
    run_analysis,
)
from basketvision_coach.api import create_app
from basketvision_coach.cv_pipeline import InMemoryVisionStore
from basketvision_coach.db import Base
from basketvision_coach.video import FfmpegMetadata, VideoIngestService


class FakeAnalyzer:
    """Deterministic stand-in for the YOLO+ByteTrack analyzer (no torch)."""

    engine = "fake"

    def probe(self, video_path: Path) -> VideoMeta:
        return VideoMeta(width=1280, height=720, fps=25.0)

    def detect_and_track(self, video_path: Path) -> Iterable[FrameDetections]:
        for frame_idx in range(3):
            yield FrameDetections(
                frame_idx=frame_idx,
                ts_s=frame_idx / 25.0,
                objects=(
                    DetectedObject("player", 0.9, 10.0 + frame_idx, 20.0, 40.0, 90.0, track_id=1),
                    DetectedObject("player", 0.8, 200.0, 30.0, 40.0, 90.0, track_id=2),
                    DetectedObject("ball", 0.95, 100.0 + frame_idx * 5, 50.0, 16.0, 16.0),
                ),
            )


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


def upload_ready_video(client: TestClient) -> int:
    game_id = client.post("/api/games", json={"name": "Analysis game"}).json()["id"]
    video_id: int = client.post(
        f"/api/games/{game_id}/video",
        files={"file": ("clip.mp4", b"\x00\x00\x00\x18ftypmp42rest", "video/mp4")},
    ).json()["id"]
    return video_id


def test_run_analysis_persists_detection_and_track_runs(tmp_path: Path) -> None:
    store = InMemoryVisionStore()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"data")
    result = run_analysis(store, video_id=1, video_path=video, analyzer=FakeAnalyzer())

    assert result.engine == "fake"
    assert result.frame_count == 3
    assert result.object_count == 9  # 3 frames x 3 objects
    assert result.source_width == 1280

    detections = store.list_detections(result.detection_run_id)
    assert sum(1 for d in detections if d.class_name == "ball") == 3
    # only players carry track ids -> tracking run gets player rows
    tracks = store.list_tracks(result.tracking_run_id)
    assert len(tracks) == 6
    assert {t.track_id for t in tracks} == {1, 2}


def test_analysis_endpoint_with_injected_analyzer(tmp_path: Path) -> None:
    app = create_app(service=make_service(tmp_path), analyzer=FakeAnalyzer())
    client = TestClient(app)
    video_id = upload_ready_video(client)

    resp = client.post(f"/api/videos/{video_id}/analysis")
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["engine"] == "fake"
    assert body["object_count"] == 9

    tracks = client.get(f"/api/runs/{body['tracking_run_id']}/tracks").json()
    assert len(tracks) == 6


def test_analysis_endpoint_reports_missing_runtime(tmp_path: Path) -> None:
    # No analyzer injected and the CV runtime (ultralytics) is not installed here,
    # so the endpoint should surface an actionable 503 rather than crashing.
    app = create_app(service=make_service(tmp_path))
    client = TestClient(app)
    video_id = upload_ready_video(client)

    resp = client.post(f"/api/videos/{video_id}/analysis")
    assert resp.status_code == 503
    assert "install" in resp.json()["detail"].lower()


def test_analysis_endpoint_missing_video(tmp_path: Path) -> None:
    app = create_app(service=make_service(tmp_path), analyzer=FakeAnalyzer())
    client = TestClient(app)
    assert client.post("/api/videos/4242/analysis").status_code == 404
