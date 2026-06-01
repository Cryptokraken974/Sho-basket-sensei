from __future__ import annotations

from basketvision_coach import (
    IdentityReviewStore,
    PlayerStatObservation,
    SpacingObservation,
    TrackIdentity,
)


def test_reviewer_assigns_full_track_to_roster_player_and_team() -> None:
    store = IdentityReviewStore()

    assignment = store.assign_track_segment(
        track_id="track-7",
        player_id="player-23",
        team_id="home",
        user="coach@example.com",
    )

    assert assignment == TrackIdentity(
        track_id="track-7",
        player_id="player-23",
        team_id="home",
        confidence=1.0,
        start_s=None,
        end_s=None,
        source="manual",
        reviewed=True,
    )
    assert store.track_identities == [assignment]
    assert store.corrections == []


def test_reviewer_assigns_bounded_segment_and_changing_it_is_audited() -> None:
    store = IdentityReviewStore()
    store.assign_track_segment(
        track_id="track-9",
        player_id="player-9",
        team_id="away",
        user="analyst",
        start_s=12.5,
        end_s=18.0,
        confidence=0.84,
        source="model_plus_manual",
    )

    updated = store.assign_track_segment(
        track_id="track-9",
        player_id="player-10",
        team_id="away",
        user="lead-reviewer",
        start_s=12.5,
        end_s=18.0,
        confidence=0.95,
        source="manual",
    )

    assert updated.player_id == "player-10"
    assert updated.team_id == "away"
    assert updated.confidence == 0.95
    assert updated.source == "manual"
    assert updated.reviewed is True
    correction_rows = [
        (c.entity_type, c.field, c.old_value, c.new_value, c.user) for c in store.corrections
    ]
    assert correction_rows == [
        ("track_identity", "player_id", "player-9", "player-10", "lead-reviewer"),
        ("track_identity", "confidence", 0.84, 0.95, "lead-reviewer"),
        ("track_identity", "source", "model_plus_manual", "manual", "lead-reviewer"),
    ]
    assert all(c.changed_at.tzinfo is not None for c in store.corrections)


def test_reports_attribute_player_stats_with_reviewed_identity_for_observation_time() -> None:
    store = IdentityReviewStore()
    store.assign_track_segment(
        track_id="track-11",
        player_id="player-a",
        team_id="home",
        user="reviewer",
        start_s=0,
        end_s=30,
    )
    store.assign_track_segment(
        track_id="track-11",
        player_id="player-b",
        team_id="home",
        user="reviewer",
        start_s=30,
        end_s=60,
    )

    report = store.player_stat_report(
        [
            PlayerStatObservation(track_id="track-11", time_s=10, points=2, rebounds=0, assists=1),
            PlayerStatObservation(track_id="track-11", time_s=40, points=3, rebounds=1, assists=0),
            PlayerStatObservation(
                track_id="unknown", time_s=10, points=99, rebounds=99, assists=99
            ),
        ]
    )

    assert report == {
        "player-a": {"points": 2, "rebounds": 0, "assists": 1},
        "player-b": {"points": 3, "rebounds": 1, "assists": 0},
    }


def test_unassigned_tracks_with_team_identity_contribute_to_team_spacing() -> None:
    store = IdentityReviewStore()
    store.assign_track_segment(
        track_id="track-unassigned-home",
        player_id=None,
        team_id="home",
        user="reviewer",
    )
    store.assign_track_segment(
        track_id="track-away",
        player_id="away-2",
        team_id="away",
        user="reviewer",
    )

    spacing = store.team_spacing_report(
        [
            SpacingObservation(track_id="track-unassigned-home", time_s=5, spacing_m=4.5),
            SpacingObservation(track_id="track-away", time_s=5, spacing_m=3.0),
            SpacingObservation(track_id="unknown", time_s=5, spacing_m=1.0),
        ]
    )

    assert spacing == {"home": 4.5, "away": 3.0}
