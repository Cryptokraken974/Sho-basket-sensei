from __future__ import annotations

import json

from basketvision_coach.coaching_reports import (
    ClipFilter,
    CoachingReportInput,
    ReviewStatus,
    build_coaching_report,
    export_report_csv,
    export_report_json,
    filter_clips,
    render_clip_library_html,
    render_printable_report_html,
    sample_clips,
    sample_events,
    sample_tracks,
)


def make_input(**overrides: object) -> CoachingReportInput:
    defaults: dict[str, object] = {
        "game_id": "game-xyz",
        "final_score": {"home": 100, "away": 90},
        "events": sample_events(),
        "clips": sample_clips(),
        "tracks": sample_tracks(),
        "active_calibration_id": "cal-default",
    }
    defaults.update(overrides)
    return CoachingReportInput(**defaults)  # type: ignore[arg-type]


def test_official_totals_exclude_unreviewed_events() -> None:
    report = build_coaching_report(make_input())
    # 4 reviewed/locked events out of 5; one AUTO event excluded.
    assert report.official_event_count == 4
    assert report.excluded_unreviewed_event_count == 1
    assert "shot-auto" not in report.official_event_ids


def test_clip_collections_only_include_reviewed() -> None:
    report = build_coaching_report(make_input())
    assert [c.clip_id for c in report.turnover_clips] == ["clip-turnover"]
    assert [c.clip_id for c in report.top_fast_break_clips] == []


def test_missing_data_messages_surface_gaps() -> None:
    report = build_coaching_report(
        make_input(final_score=None, active_calibration_id=None, tracks=())
    )
    assert "final_score" in report.missing_data
    assert "calibration" in report.missing_data
    assert "tracks" in report.missing_data


def test_filter_clips_matches_all_criteria() -> None:
    clips = sample_clips()
    filtered = filter_clips(
        clips,
        ClipFilter(
            event_type="shot",
            player_id="p1",
            team_id="home",
            status=ReviewStatus.REVIEWED,
            min_confidence=0.9,
            tags=("paint-touch",),
        ),
    )
    assert len(filtered) == 1
    assert filtered[0].clip_id == "clip-shot-reviewed"


def test_exports_and_rendering_are_well_formed() -> None:
    report = build_coaching_report(make_input())
    payload = json.loads(export_report_json(report))
    assert payload["official_event_count"] == 4

    csv = export_report_csv(report)
    assert csv.splitlines()[0] == "Type,Team,Zone,Value,EventID"

    assert "Coaching Report game-xyz" in render_printable_report_html(report)
    assert "Clip Library" in render_clip_library_html(sample_clips())
    assert render_clip_library_html(()) == "<p>No clips match the current filter criteria.</p>"
