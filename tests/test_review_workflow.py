from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from basketvision_coach.review import (
    EVENT_TYPES,
    ClipExporter,
    EventStatus,
    ReviewEventCreate,
    ReviewStore,
    render_review_page,
)


def test_manual_event_defaults_to_reviewed_and_can_be_updated_and_deleted(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path / "review.sqlite3")

    event = store.create_event(
        ReviewEventCreate(
            video_id="game-1",
            type="make",
            start_s=10.0,
            end_s=11.5,
            team_id="home",
            player_id="p23",
        )
    )

    assert event.source == "manual"
    assert event.status == EventStatus.REVIEWED
    assert event.reviewed is True
    assert event.confidence == 1.0

    edited = store.update_event(
        event.id, type="miss", status=EventStatus.NEEDS_REVIEW, reviewed=False
    )
    assert edited.type == "miss"
    assert edited.status == EventStatus.NEEDS_REVIEW
    assert edited.reviewed is False

    accepted = store.accept_event(event.id)
    assert accepted.status == EventStatus.REVIEWED
    assert accepted.reviewed is True

    rejected = store.reject_event(event.id)
    assert rejected.status == EventStatus.REJECTED
    assert rejected.reviewed is False

    store.delete_event(event.id)
    assert store.list_events("game-1") == []


def test_event_type_validation_and_page_contains_keyboard_shortcuts(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path / "review.sqlite3")
    page = render_review_page(video_id="game-1", video_url="/media/game.mp4", events=[])

    for event_type in EVENT_TYPES:
        assert event_type in page

    assert "data-shortcut" in page
    assert "Space" in page
    assert "ArrowLeft" in page
    assert "ArrowRight" in page
    assert "seekToEvent" in page
    assert "createManualEvent" in page

    try:
        store.create_event(
            ReviewEventCreate(video_id="game-1", type="bad-tag", start_s=0.0, end_s=1.0)
        )
    except ValueError as exc:
        assert "unsupported event type" in str(exc)
    else:  # pragma: no cover - explicit assertion path
        raise AssertionError("invalid event type should fail")


def test_clip_export_records_playable_clip_paths(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path / "review.sqlite3")
    event = store.create_event(
        ReviewEventCreate(video_id="game-1", type="shot", start_s=5.0, end_s=7.0)
    )
    calls: list[list[str]] = []

    def fake_runner(command: list[str]) -> None:
        calls.append(command)
        Path(command[-1]).write_bytes(b"fake mp4")

    exporter = ClipExporter(store, runner=fake_runner)
    clips = exporter.export_clips(
        video_id="game-1",
        source_video=tmp_path / "source.mp4",
        event_ids=[event.id],
        output_dir=tmp_path / "clips",
        pre_roll_s=2.0,
        post_roll_s=3.0,
    )

    assert len(clips) == 1
    assert clips[0].event_id == event.id
    assert Path(clips[0].path).read_bytes() == b"fake mp4"
    assert "-ss" in calls[0]
    assert calls[0][calls[0].index("-ss") + 1] == "3.000"
    assert "-t" in calls[0]
    assert calls[0][calls[0].index("-t") + 1] == "7.000"

    with sqlite3.connect(tmp_path / "review.sqlite3") as conn:
        rows = conn.execute("select event_id, path from clips").fetchall()
    assert rows == [(event.id, clips[0].path)]


def test_csv_export_defaults_to_reviewed_and_locked_events(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path / "review.sqlite3")
    reviewed = store.create_event(
        ReviewEventCreate(video_id="game-1", type="make", start_s=1.0, end_s=2.0)
    )
    locked = store.create_event(
        ReviewEventCreate(video_id="game-1", type="rebound", start_s=3.0, end_s=4.0)
    )
    store.lock_event(locked.id)
    auto = store.create_event(
        ReviewEventCreate(
            video_id="game-1",
            type="turnover",
            start_s=5.0,
            end_s=6.0,
            source="auto",
            status=EventStatus.NEEDS_REVIEW,
            reviewed=False,
            confidence=0.42,
        )
    )
    rejected = store.create_event(
        ReviewEventCreate(video_id="game-1", type="miss", start_s=7.0, end_s=8.0)
    )
    store.reject_event(rejected.id)

    csv_path = store.export_events_csv("game-1", tmp_path / "events.csv")
    rows = list(csv.DictReader(csv_path.read_text().splitlines()))

    assert [int(row["id"]) for row in rows] == [reviewed.id, locked.id]
    assert {row["status"] for row in rows} == {"reviewed", "locked"}

    all_path = store.export_events_csv("game-1", tmp_path / "events-all.csv", include_all=True)
    all_rows = list(csv.DictReader(all_path.read_text().splitlines()))
    assert [int(row["id"]) for row in all_rows] == [reviewed.id, locked.id, auto.id, rejected.id]
