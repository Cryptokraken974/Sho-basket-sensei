from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from basketvision_coach.analysis import DetectedObject, FrameDetections, VideoMeta
from basketvision_coach.api import create_app
from basketvision_coach.click_track import (
    ClickPrompt,
    mask_bbox,
    run_click_tracking,
)
from basketvision_coach.cv_pipeline import InMemoryVisionStore
from basketvision_coach.db import Base
from basketvision_coach.video import FfmpegMetadata, VideoIngestService


class FakeClickTracker:
    """Stand-in for SAM 2: emits one box per prompt label per frame."""

    engine = "fake-sam2"

    def probe(self, video_path: Path) -> VideoMeta:
        return VideoMeta(width=1280, height=720, fps=25.0)

    def track(
        self, video_path: Path, prompts: Sequence[ClickPrompt]
    ) -> Iterable[FrameDetections]:
        labels: list[str] = []
        for prompt in prompts:
            if prompt.object_label not in labels:
                labels.append(prompt.object_label)
        for frame_idx in range(2):
            objects = tuple(
                DetectedObject(
                    class_name="ball" if label.lower() == "ball" else "player",
                    confidence=1.0,
                    x=10.0 + frame_idx + i,
                    y=20.0,
                    width=30.0,
                    height=60.0,
                    track_id=i + 1,
                )
                for i, label in enumerate(labels)
            )
            yield FrameDetections(frame_idx=frame_idx, ts_s=frame_idx / 25.0, objects=objects)


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
    game_id = client.post("/api/games", json={"name": "Click game"}).json()["id"]
    video_id: int = client.post(
        f"/api/games/{game_id}/video",
        files={"file": ("clip.mp4", b"\x00\x00\x00\x18ftypmp42rest", "video/mp4")},
    ).json()["id"]
    return video_id


def test_mask_bbox_encloses_truthy_cells() -> None:
    mask = [
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ]
    assert mask_bbox(mask) == (1.0, 1.0, 2.0, 2.0)


def test_mask_bbox_empty_is_none() -> None:
    assert mask_bbox([[0, 0], [0, 0]]) is None


def test_run_click_tracking_requires_prompts(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    with pytest.raises(ValueError):
        run_click_tracking(
            InMemoryVisionStore(),
            video_id=1,
            video_path=video,
            prompts=[],
            tracker=FakeClickTracker(),
        )


def test_run_click_tracking_persists_runs(tmp_path: Path) -> None:
    store = InMemoryVisionStore()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    prompts = [
        ClickPrompt("Player 7", frame_idx=0, x=100.0, y=200.0),
        ClickPrompt("ball", frame_idx=0, x=300.0, y=120.0),
    ]
    result = run_click_tracking(
        store, video_id=1, video_path=video, prompts=prompts, tracker=FakeClickTracker()
    )
    assert result.engine == "fake-sam2"
    assert result.frame_count == 2
    # two labels x two frames
    assert len(store.list_tracks(result.tracking_run_id)) == 4
    assert {t.track_id for t in store.list_tracks(result.tracking_run_id)} == {1, 2}


def test_click_track_endpoint_with_injected_tracker(tmp_path: Path) -> None:
    app = create_app(service=make_service(tmp_path), click_tracker=FakeClickTracker())
    client = TestClient(app)
    video_id = upload_video(client)

    resp = client.post(
        f"/api/videos/{video_id}/click-track",
        json={"prompts": [{"object_label": "Player 7", "frame_idx": 0, "x": 50.0, "y": 60.0}]},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["engine"] == "fake-sam2"
    tracks = client.get(f"/api/runs/{body['tracking_run_id']}/tracks").json()
    assert len(tracks) == 2


def test_click_track_requires_at_least_one_prompt(tmp_path: Path) -> None:
    app = create_app(service=make_service(tmp_path), click_tracker=FakeClickTracker())
    client = TestClient(app)
    video_id = upload_video(client)
    resp = client.post(f"/api/videos/{video_id}/click-track", json={"prompts": []})
    assert resp.status_code == 422  # pydantic min_length


def test_click_track_reports_missing_runtime(tmp_path: Path) -> None:
    app = create_app(service=make_service(tmp_path))  # no tracker; sam2 not installed
    client = TestClient(app)
    video_id = upload_video(client)
    resp = client.post(
        f"/api/videos/{video_id}/click-track",
        json={"prompts": [{"object_label": "p", "frame_idx": 0, "x": 1.0, "y": 1.0}]},
    )
    assert resp.status_code == 503
    assert "sam 2" in resp.json()["detail"].lower()
