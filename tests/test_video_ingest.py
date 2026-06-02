from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from basketvision_coach.api import create_app
from basketvision_coach.db import Base
from basketvision_coach.video import (
    FfmpegMetadata,
    VideoIngestService,
    frame_idx_to_ts_s,
    ts_s_to_frame_idx,
)
from basketvision_coach.video_models import VideoState


def make_service(tmp_path: Path, metadata: FfmpegMetadata | None = None) -> VideoIngestService:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    def fake_transcode(source: Path, proxy: Path, hls_dir: Path | None) -> FfmpegMetadata:
        assert source.read_bytes().startswith(b"\x00\x00\x00\x18ftypmp42")
        proxy.write_bytes(b"proxy mp4")
        if hls_dir is not None:
            hls_dir.mkdir(parents=True, exist_ok=True)
            (hls_dir / "index.m3u8").write_text("#EXTM3U\n", encoding="utf-8")
        return metadata or FfmpegMetadata(duration_s=12.5, fps=29.97, width=1920, height=1080)

    return VideoIngestService(
        session_factory=session_factory,
        data_root=tmp_path / "data",
        transcode_runner=fake_transcode,
    )


def tiny_mp4_bytes() -> bytes:
    return b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"raw-video-payload"


def test_ingesting_mp4_preserves_original_and_records_processed_metadata(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    original = tiny_mp4_bytes()
    original_hash = hashlib.sha256(original).hexdigest()

    game = service.create_game("Free throw form")
    video = service.ingest_game_video(game.id, "shot.mp4", original, run_transcode=True)

    stored = service.get_video(video.id)
    assert stored is not None
    assert stored.state is VideoState.READY
    assert stored.error_message is None
    assert stored.original_path.endswith("original.mp4")
    assert Path(stored.original_path).read_bytes() == original
    assert hashlib.sha256(Path(stored.original_path).read_bytes()).hexdigest() == original_hash
    assert Path(stored.proxy_path or "").name == "proxy_720p.mp4"
    assert Path(stored.proxy_path or "").read_bytes() == b"proxy mp4"
    assert stored.duration_s == 12.5
    assert stored.fps == 29.97
    assert stored.width == 1920
    assert stored.height == 1080
    assert stored.created_at is not None


def test_failed_transcode_surfaces_error_state_and_keeps_original(tmp_path: Path) -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    def failing_transcode(source: Path, proxy: Path, hls_dir: Path | None) -> FfmpegMetadata:
        raise RuntimeError("ffmpeg exited 1")

    service = VideoIngestService(
        session_factory=session_factory,
        data_root=tmp_path / "data",
        transcode_runner=failing_transcode,
    )
    game = service.create_game("Broken upload")
    video = service.ingest_game_video(game.id, "shot.mp4", tiny_mp4_bytes(), run_transcode=True)

    stored = service.get_video(video.id)
    assert stored is not None
    assert stored.state is VideoState.FAILED
    assert stored.error_message == "ffmpeg exited 1"
    assert Path(stored.original_path).exists()
    assert stored.proxy_path is None


def test_timestamp_mapping_uses_stored_fps_deterministically() -> None:
    assert frame_idx_to_ts_s(150, fps=30.0) == 5.0
    assert frame_idx_to_ts_s(3, fps=29.97) == 3 / 29.97
    assert ts_s_to_frame_idx(5.0, fps=30.0) == 150
    assert ts_s_to_frame_idx(3 / 29.97, fps=29.97) == 3


def test_api_upload_creates_video_record_and_player_uses_proxy(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    app = create_app(service=service)
    client = TestClient(app)

    game_response = client.post("/api/games", json={"name": "Corner threes"})
    assert game_response.status_code == 201
    game_id = game_response.json()["id"]

    upload_response = client.post(
        f"/api/games/{game_id}/video",
        files={"file": ("shot.mp4", tiny_mp4_bytes(), "video/mp4")},
    )
    assert upload_response.status_code == 201
    body = upload_response.json()
    assert body["state"] == "ready"
    assert body["proxy_url"].endswith("/proxy_720p.mp4")
    assert body["ts_mapping"] == "ts_s = frame_idx / fps"

    page = client.get(f"/games/{game_id}")
    assert page.status_code == 200
    assert "Corner threes" in page.text
    assert "<video" in page.text
    assert "/static/app.js" in page.text

    playable = client.get(body["proxy_url"])
    assert playable.status_code == 200
    assert playable.content == b"proxy mp4"


def test_upload_rejects_non_mp4(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    app = create_app(service=service)
    client = TestClient(app)
    game_id = client.post("/api/games", json={"name": "Bad upload"}).json()["id"]

    response = client.post(
        f"/api/games/{game_id}/video",
        files={"file": ("clip.mov", b"not mp4", "video/quicktime")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only MP4 uploads are supported"
