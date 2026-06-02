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
        return FfmpegMetadata(duration_s=4.0, fps=25.0, width=1280, height=720)

    return VideoIngestService(
        session_factory=session_factory,
        data_root=tmp_path,
        transcode_runner=fake_transcode,
        create_hls=False,
    )


def client_for(tmp_path: Path) -> TestClient:
    return TestClient(create_app(service=make_service(tmp_path)))


def upload_video(client: TestClient) -> tuple[int, int]:
    game_id = client.post("/api/games", json={"name": "Review game"}).json()["id"]
    video_id = client.post(
        f"/api/games/{game_id}/video",
        files={"file": ("clip.mp4", b"\x00\x00\x00\x18ftypmp42rest", "video/mp4")},
    ).json()["id"]
    return game_id, video_id


def test_tag_list_and_lifecycle(tmp_path: Path) -> None:
    client = client_for(tmp_path)
    _, video_id = upload_video(client)

    created = client.post(
        f"/api/videos/{video_id}/events",
        json={"type": "make", "start_s": 12.5, "player_id": "p1"},
    )
    assert created.status_code == 201, created.text
    event = created.json()
    assert event["type"] == "make"
    assert event["end_s"] == 12.5  # defaults to start
    assert event["status"] == "reviewed"

    listed = client.get(f"/api/videos/{video_id}/events").json()
    assert len(listed) == 1

    rejected = client.post(f"/api/events/{event['id']}/reject")
    assert rejected.json()["status"] == "rejected"
    accepted = client.post(f"/api/events/{event['id']}/accept")
    assert accepted.json()["status"] == "reviewed"

    patched = client.patch(f"/api/events/{event['id']}", json={"type": "miss"})
    assert patched.json()["type"] == "miss"

    assert client.delete(f"/api/events/{event['id']}").status_code == 204
    assert client.get(f"/api/videos/{video_id}/events").json() == []


def test_create_event_rejects_bad_type(tmp_path: Path) -> None:
    client = client_for(tmp_path)
    _, video_id = upload_video(client)
    resp = client.post(
        f"/api/videos/{video_id}/events", json={"type": "dunkzilla", "start_s": 1.0}
    )
    assert resp.status_code == 400


def test_report_counts_and_shooting(tmp_path: Path) -> None:
    client = client_for(tmp_path)
    _, video_id = upload_video(client)
    for payload in (
        {"type": "make", "start_s": 1.0},
        {"type": "make", "start_s": 2.0},
        {"type": "miss", "start_s": 3.0},
        {"type": "turnover", "start_s": 4.0, "status": "needs_review"},
    ):
        client.post(f"/api/videos/{video_id}/events", json=payload)

    report = client.get(f"/api/videos/{video_id}/report").json()
    assert report["total_tagged"] == 4
    assert report["official_event_count"] == 3  # the needs_review one is excluded
    assert report["excluded_unreviewed_event_count"] == 1
    assert report["by_type"]["make"] == 2
    assert report["makes"] == 2 and report["misses"] == 1
    assert report["shooting_pct"] == 66.7


def test_events_csv_export(tmp_path: Path) -> None:
    client = client_for(tmp_path)
    _, video_id = upload_video(client)
    client.post(f"/api/videos/{video_id}/events", json={"type": "make", "start_s": 1.0})
    resp = client.get(f"/api/videos/{video_id}/events.csv")
    assert resp.status_code == 200
    assert "make" in resp.text


def test_review_and_report_pages_render(tmp_path: Path) -> None:
    client = client_for(tmp_path)
    game_id, _ = upload_video(client)
    review = client.get(f"/games/{game_id}/review")
    assert review.status_code == 200
    assert "bvReview" in review.text
    assert "Tag events" in review.text

    report = client.get(f"/games/{game_id}/report")
    assert report.status_code == 200
    assert "bvReport" in report.text


def test_gamenav_links_present(tmp_path: Path) -> None:
    client = client_for(tmp_path)
    game_id, _ = upload_video(client)
    page = client.get(f"/games/{game_id}")
    assert f'href="/games/{game_id}/review"' in page.text
    assert f'href="/games/{game_id}/report"' in page.text
