from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from basketvision_coach.api import create_app
from basketvision_coach.db import Base
from basketvision_coach.samples import list_sample_videos, resolve_sample
from basketvision_coach.video import FfmpegMetadata, VideoIngestService

MP4_BYTES = b"\x00\x00\x00\x18ftypmp42rest-of-clip"


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
        data_root=tmp_path / "data",
        transcode_runner=fake_transcode,
        create_hls=False,
    )


def make_samples_dir(tmp_path: Path) -> Path:
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "Game 1.mp4").write_bytes(MP4_BYTES)
    (samples / "notes.txt").write_text("ignore me")
    return samples


def test_list_and_resolve_helpers(tmp_path: Path) -> None:
    samples = make_samples_dir(tmp_path)
    assert list_sample_videos(samples) == ["Game 1.mp4"]
    assert resolve_sample("Game 1.mp4", samples) == samples / "Game 1.mp4"
    # path traversal is stripped to the basename and not found
    assert resolve_sample("../secrets.mp4", samples) is None
    assert resolve_sample("missing.mp4", samples) is None


def test_samples_listed_and_ingestable_from_ui(tmp_path: Path) -> None:
    samples = make_samples_dir(tmp_path)
    client = TestClient(
        create_app(service=make_service(tmp_path), samples_directory=samples)
    )

    assert client.get("/api/samples").json() == ["Game 1.mp4"]

    game_id = client.post("/api/games", json={"name": "Sample game"}).json()["id"]
    resp = client.post(
        f"/api/games/{game_id}/video/from-sample", json={"filename": "Game 1.mp4"}
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["game_id"] == game_id


def test_ingest_unknown_sample_404(tmp_path: Path) -> None:
    samples = make_samples_dir(tmp_path)
    client = TestClient(
        create_app(service=make_service(tmp_path), samples_directory=samples)
    )
    game_id = client.post("/api/games", json={"name": "g"}).json()["id"]
    resp = client.post(
        f"/api/games/{game_id}/video/from-sample", json={"filename": "nope.mp4"}
    )
    assert resp.status_code == 404


def test_no_samples_dir_returns_empty(tmp_path: Path) -> None:
    client = TestClient(
        create_app(service=make_service(tmp_path), samples_directory=tmp_path / "missing")
    )
    assert client.get("/api/samples").json() == []
