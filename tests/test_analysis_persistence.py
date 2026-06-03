from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from basketvision_coach.analysis import DetectedObject, FrameDetections, VideoMeta
from basketvision_coach.analysis_store import (
    analysis_available,
    load_analysis,
    restore_into_store,
    save_analysis,
    snapshot,
)
from basketvision_coach.api import create_app
from basketvision_coach.cv_pipeline import InMemoryVisionStore
from basketvision_coach.db import Base
from basketvision_coach.video import FfmpegMetadata, VideoIngestService


class FakeAnalyzer:
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
                    DetectedObject("ball", 0.95, 100.0 + frame_idx, 50.0, 16.0, 16.0),
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


def upload_video(client: TestClient) -> int:
    game_id = client.post("/api/games", json={"name": "g"}).json()["id"]
    return int(
        client.post(
            f"/api/games/{game_id}/video",
            files={"file": ("clip.mp4", b"\x00\x00\x00\x18ftypmp42rest", "video/mp4")},
        ).json()["id"]
    )


def test_snapshot_save_load_restore_roundtrip(tmp_path: Path) -> None:
    store = InMemoryVisionStore()
    from basketvision_coach.analysis import run_analysis

    result = run_analysis(store, video_id=5, video_path=tmp_path / "v.mp4", analyzer=FakeAnalyzer())
    snap = snapshot(
        store,
        video_id=5,
        detection_run_id=result.detection_run_id,
        tracking_run_id=result.tracking_run_id,
        source_width=1280.0,
        source_height=720.0,
        fps=25.0,
        engine="fake",
    )
    save_analysis(tmp_path, 5, snap)
    assert analysis_available(tmp_path, 5)

    loaded = load_analysis(tmp_path, 5)
    assert loaded is not None and loaded["object_count"] == len(snap["detections"])

    fresh = InMemoryVisionStore()
    det_run, trk_run = restore_into_store(fresh, loaded)
    assert len(fresh.list_detections(det_run)) == len(snap["detections"])
    assert len(fresh.list_tracks(trk_run)) == len(snap["tracks"])


def test_analysis_persists_and_restores_via_api(tmp_path: Path) -> None:
    client = TestClient(create_app(service=make_service(tmp_path), analyzer=FakeAnalyzer()))
    video_id = upload_video(client)

    assert client.get(f"/api/videos/{video_id}/analysis-available").json()["available"] is False

    run = client.post(f"/api/videos/{video_id}/analysis")
    assert run.status_code == 201, run.text

    status = client.get(f"/api/videos/{video_id}/analysis-available").json()
    assert status["available"] is True
    assert status["object_count"] >= 1

    restored = client.post(f"/api/videos/{video_id}/restore-analysis")
    assert restored.status_code == 200, restored.text
    body = restored.json()
    dets = client.get(f"/api/runs/{body['detection_run_id']}/detections").json()
    assert any(d["class_name"] == "ball" for d in dets)


def test_restore_without_saved_analysis_404(tmp_path: Path) -> None:
    client = TestClient(create_app(service=make_service(tmp_path), analyzer=FakeAnalyzer()))
    video_id = upload_video(client)
    assert client.post(f"/api/videos/{video_id}/restore-analysis").status_code == 404
