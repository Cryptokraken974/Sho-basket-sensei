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


def upload_ready_video(client: TestClient, game_id: int) -> None:
    client.post(
        f"/api/games/{game_id}/video",
        files={"file": ("shot.mp4", b"\x00\x00\x00\x18ftypmp42rest", "video/mp4")},
    )


def test_index_renders_with_create_form(tmp_path: Path) -> None:
    response = client_for(tmp_path).get("/")
    assert response.status_code == 200
    assert "Games" in response.text
    assert 'action="/games"' in response.text


def test_create_game_form_htmx_returns_partial(tmp_path: Path) -> None:
    client = client_for(tmp_path)
    response = client.post("/games", data={"name": "Pickup run"}, headers={"HX-Request": "true"})
    assert response.status_code == 200
    assert "Pickup run" in response.text
    assert 'id="game-list"' in response.text
    # the partial should not include the full page chrome
    assert "<html" not in response.text


def test_create_game_form_full_page(tmp_path: Path) -> None:
    client = client_for(tmp_path)
    response = client.post("/games", data={"name": "Full court"})
    assert response.status_code == 200
    assert "Full court" in response.text
    assert "<html" in response.text


def test_game_page_renders_player_scaffold(tmp_path: Path) -> None:
    client = client_for(tmp_path)
    game_id = client.post("/api/games", json={"name": "Game A"}).json()["id"]
    response = client.get(f"/games/{game_id}")
    assert response.status_code == 200
    assert "Game A" in response.text
    assert "bvGamePage" in response.text
    assert "/static/app.js" in response.text


def test_calibration_requires_ready_video(tmp_path: Path) -> None:
    client = client_for(tmp_path)
    game_id = client.post("/api/games", json={"name": "No video"}).json()["id"]
    response = client.get(f"/games/{game_id}/calibrate")
    assert response.status_code == 409


def test_calibration_page_renders_for_ready_video(tmp_path: Path) -> None:
    client = client_for(tmp_path)
    game_id = client.post("/api/games", json={"name": "Ready game"}).json()["id"]
    upload_ready_video(client, game_id)
    response = client.get(f"/games/{game_id}/calibrate")
    assert response.status_code == 200
    assert "bvCalibration" in response.text
    assert "Court calibration" in response.text


def test_static_assets_served(tmp_path: Path) -> None:
    client = client_for(tmp_path)
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/app.css").status_code == 200
