# ruff: noqa: E501
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from basketvision_coach.api import create_app
from basketvision_coach.db import Base
from basketvision_coach.identity import (
    IdentityReviewService,
    PlayerStatObservation,
    SpacingObservation,
)
from basketvision_coach.video import FfmpegMetadata, VideoIngestService


def make_session_factory() -> sessionmaker[Session]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def make_video_service(tmp_path: Path, session_factory: sessionmaker[Session]) -> VideoIngestService:
    def fake_transcode(source: Path, proxy: Path, hls_dir: Path | None) -> FfmpegMetadata:
        proxy.write_bytes(b"proxy")
        return FfmpegMetadata(duration_s=20.0, fps=25.0, width=1280, height=720)

    return VideoIngestService(
        session_factory=session_factory,
        data_root=tmp_path / "data",
        transcode_runner=fake_transcode,
    )


def test_manual_assignment_persists_track_identity_and_audits_changes() -> None:
    session_factory = make_session_factory()
    service = IdentityReviewService(session_factory=session_factory)
    team = service.create_team("Home", jersey_color="#ffffff")
    first_player = service.create_player("Alice", team.id, jersey_number="7")
    second_player = service.create_player("Bianca", team.id, jersey_number="8")

    created = service.assign_track_segment(
        track_id="track-42",
        player_id=first_player.id,
        team_id=team.id,
        user="reviewer@example.com",
        start_s=2.0,
        end_s=9.5,
        confidence=0.88,
        source="model_plus_manual",
    )
    assert created.reviewed is True
    assert created.source == "model_plus_manual"

    updated = service.assign_track_segment(
        track_id="track-42",
        player_id=second_player.id,
        team_id=team.id,
        user="lead@example.com",
        start_s=2.0,
        end_s=9.5,
        confidence=1.0,
        source="manual",
    )

    assignments = service.list_track_identities(track_id="track-42")
    assert [assignment.id for assignment in assignments] == [updated.id]
    assert assignments[0].player_id == second_player.id
    assert assignments[0].team_id == team.id
    assert assignments[0].reviewed is True

    corrections = service.list_corrections(entity_type="track_identity")
    correction_rows = [(c.field, c.old_value, c.new_value, c.user) for c in corrections]
    assert correction_rows == [
        ("player_id", str(first_player.id), str(second_player.id), "lead@example.com"),
        ("confidence", "0.88", "1.0", "lead@example.com"),
        ("source", "model_plus_manual", "manual", "lead@example.com"),
    ]
    assert all(c.changed_at.tzinfo is not None for c in corrections)


def test_reports_prefer_reviewed_identity_and_team_spacing_allows_team_only_tracks() -> None:
    session_factory = make_session_factory()
    service = IdentityReviewService(session_factory=session_factory)
    home = service.create_team("Home")
    away = service.create_team("Away")
    home_player = service.create_player("Alice", home.id)
    away_player = service.create_player("Cara", away.id)

    service.seed_team_identity_from_jersey_color(
        track_id="ambiguous-home",
        team_id=home.id,
        confidence=0.63,
    )
    service.assign_track_segment(
        track_id="track-1",
        player_id=home_player.id,
        team_id=home.id,
        user="reviewer",
        start_s=0.0,
        end_s=10.0,
    )
    service.assign_track_segment(
        track_id="track-1",
        player_id=away_player.id,
        team_id=away.id,
        user="reviewer",
        start_s=10.0,
        end_s=20.0,
    )

    stats = service.player_stat_report(
        [
            PlayerStatObservation(track_id="track-1", time_s=4.0, points=2, assists=1),
            PlayerStatObservation(track_id="track-1", time_s=12.0, points=3, rebounds=1),
            PlayerStatObservation(track_id="ambiguous-home", time_s=6.0, points=99),
        ]
    )
    assert stats == {
        home_player.id: {"points": 2, "rebounds": 0, "assists": 1},
        away_player.id: {"points": 3, "rebounds": 1, "assists": 0},
    }

    spacing = service.team_spacing_report(
        [
            SpacingObservation(track_id="ambiguous-home", time_s=4.0, spacing_m=5.0),
            SpacingObservation(track_id="track-1", time_s=4.0, spacing_m=3.0),
            SpacingObservation(track_id="unknown", time_s=4.0, spacing_m=99.0),
        ]
    )
    assert spacing == {home.id: 4.0}


def test_tracks_page_exposes_manual_review_form_and_api_accepts_bounded_assignment(
    tmp_path: Path,
) -> None:
    session_factory = make_session_factory()
    video_service = make_video_service(tmp_path, session_factory)
    identity_service = IdentityReviewService(session_factory=session_factory)
    team = identity_service.create_team("Home", jersey_color="white")
    player = identity_service.create_player("Alice Example", team.id, jersey_number="12")
    game = video_service.create_game("Review game")

    app = create_app(video_service=video_service, identity_service=identity_service)
    client = TestClient(app)

    page = client.get(f"/games/{game.id}/tracks")
    assert page.status_code == 200
    assert "data-track-id=\"track-1\"" in page.text
    assert "Assign selected track segment" in page.text
    assert "Alice Example" in page.text

    response = client.post(
        "/api/track-identities",
        json={
            "track_id": "track-1",
            "player_id": player.id,
            "team_id": team.id,
            "user": "reviewer@example.com",
            "start_s": 3.25,
            "end_s": 8.75,
            "confidence": 0.97,
            "source": "manual",
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["track_id"] == "track-1"
    assert body["player_id"] == player.id
    assert body["team_id"] == team.id
    assert body["reviewed"] is True
    assert body["start_s"] == 3.25
    assert body["end_s"] == 8.75
