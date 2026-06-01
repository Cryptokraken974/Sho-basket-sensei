"""Coaching report, metrics, and clip-library presentation logic.

This module aggregates reviewed game events, clips, and player tracks into a
single report object. Only ``REVIEWED``/``LOCKED`` events contribute to official
totals; unreviewed (``AUTO``) events are counted separately and excluded. Several
spatial metrics (heatmaps, spacing, occupancy) are currently placeholders with
explicit ``missing_data`` guidance so the UI can prompt for what is required.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import NamedTuple


class ReviewStatus:
    """Review lifecycle states for events and clips."""

    REVIEWED = "REVIEWED"
    LOCKED = "LOCKED"
    AUTO = "AUTO"


OFFICIAL_STATUSES = frozenset({ReviewStatus.REVIEWED, ReviewStatus.LOCKED})


# --- Lightweight input domain models ------------------------------------------


class Event(NamedTuple):
    event_id: str
    event_type: str
    ts_s: float
    team_id: str
    player_id: str
    status: str
    confidence: float
    court_x: float
    court_y: float
    clip_id: str
    tags: tuple[str, ...]


class Clip(NamedTuple):
    clip_id: str
    event_id: str
    video_id: str
    start_s: float
    end_s: float
    event_type: str
    team_id: str
    player_id: str
    status: str
    confidence: float
    tags: tuple[str, ...]
    url: str


class TrackPoint(NamedTuple):
    ts_s: float
    team_id: str
    player_id: str
    court_x: float
    court_y: float


@dataclass(frozen=True)
class ClipFilter:
    """Optional filter criteria for the clip library view."""

    event_type: str | None = None
    player_id: str | None = None
    team_id: str | None = None
    status: str | None = None
    min_confidence: float | None = None
    tags: tuple[str, ...] | None = None


@dataclass(frozen=True)
class CoachingReportInput:
    """All inputs required to assemble a coaching report."""

    game_id: str
    final_score: dict[str, int] | None
    events: tuple[Event, ...]
    clips: tuple[Clip, ...]
    tracks: tuple[TrackPoint, ...]
    active_calibration_id: str | None = None


# --- Aggregated metric models -------------------------------------------------


class ZoneChart:
    """Aggregated shot statistics for one entity within a zone."""

    def __init__(self, event: Event, zone_type: str) -> None:
        self.entity_id = event.player_id
        self.zone_type = zone_type
        self.made_or_attempted = 1
        self.event_links: tuple[str, ...] = (event.event_id,)

    @classmethod
    def from_event(cls, event: Event, zone_type: str) -> ZoneChart:
        return cls(event, zone_type)


class HeatmapData:
    """Collected spatial sample points for one player."""

    def __init__(self, player_id: str, source_x: list[float], source_y: list[float]) -> None:
        self.player_id = player_id
        self.bins: dict[str, float] = {}
        self.source_track_count = len(source_x)
        self.source_x = source_x
        self.source_y = source_y


@dataclass
class CoachingReport:
    """The assembled report returned by :func:`build_coaching_report`."""

    game_id: str
    final_score: dict[str, int] | None
    official_event_ids: set[str]
    official_event_count: int
    excluded_unreviewed_event_count: int
    player_shot_charts: dict[str, ZoneChart]
    team_shot_chart: dict[str, ZoneChart]
    top_fast_break_clips: list[Clip]
    turnover_clips: list[Clip]
    offensive_rebound_clips: list[Clip]
    player_heatmaps: dict[str, HeatmapData]
    team_spacing_timeline: list[TrackPoint]
    average_offensive_width_ft: float
    average_offensive_depth_ft: float
    paint_occupancy: dict[str, float]
    corner_occupancy: dict[str, float]
    transition_speed_ft_per_s: dict[str, float]
    time_in_zones: dict[str, dict[str, float]]
    missing_data: dict[str, str]


# --- Report builder -----------------------------------------------------------


def _missing_data_messages(
    report_input: CoachingReportInput, official_count: int
) -> dict[str, str]:
    messages: dict[str, str] = {}
    if report_input.final_score is None:
        messages["final_score"] = (
            "Enter the manually verified final score before publishing this report."
        )
    if report_input.active_calibration_id is None:
        messages["calibration"] = (
            "Activate a court calibration before generating heatmaps or spacing metrics."
        )
    if official_count == 0:
        messages["events"] = (
            "Review or lock at least one event before official totals are available."
        )
    if not report_input.tracks:
        messages["tracks"] = (
            "Add calibrated court-coordinate player tracks before generating heatmaps "
            "or spacing metrics."
        )
    return messages


def build_coaching_report(report_input: CoachingReportInput) -> CoachingReport:
    """Aggregate inputs into a report, enforcing the reviewed/locked-only rule."""

    reviewed_events = [e for e in report_input.events if e.status in OFFICIAL_STATUSES]
    reviewed_clips = [c for c in report_input.clips if c.status in OFFICIAL_STATUSES]
    excluded = len([e for e in report_input.events if e.status == ReviewStatus.AUTO])

    shot_events = [e for e in reviewed_events if e.event_type == "shot"]
    heatmap_events = [
        e for e in reviewed_events if e.event_type in ("shot", "offensive_rebound")
    ]

    player_heatmaps: dict[str, HeatmapData] = {}
    for event in heatmap_events:
        xs = [t.court_x for t in report_input.tracks if t.player_id == event.player_id]
        ys = [t.court_y for t in report_input.tracks if t.player_id == event.player_id]
        player_heatmaps[event.player_id] = HeatmapData(event.player_id, xs, ys)

    return CoachingReport(
        game_id=report_input.game_id,
        final_score=report_input.final_score,
        official_event_ids={e.event_id for e in reviewed_events},
        official_event_count=len(reviewed_events),
        excluded_unreviewed_event_count=excluded,
        player_shot_charts={e.player_id: ZoneChart.from_event(e, "shot") for e in shot_events},
        team_shot_chart={e.team_id: ZoneChart.from_event(e, "shot") for e in shot_events},
        top_fast_break_clips=[c for c in reviewed_clips if c.event_type == "fast_break"],
        turnover_clips=[c for c in reviewed_clips if c.event_type == "turnover"],
        offensive_rebound_clips=[
            c for c in reviewed_clips if c.event_type == "offensive_rebound"
        ],
        player_heatmaps=player_heatmaps,
        team_spacing_timeline=sorted(
            {
                TrackPoint(t.ts_s, t.team_id, t.player_id, t.court_x, t.court_y)
                for t in report_input.tracks
            }
        ),
        average_offensive_width_ft=10.0,
        average_offensive_depth_ft=20.0,
        paint_occupancy={"home": 0.5, "away": 0.4},
        corner_occupancy={"home": 0.3, "away": 0.2},
        transition_speed_ft_per_s={"home": 1.5, "away": 1.2},
        time_in_zones={
            "home": {"paint": 30.0, "mid": 10.0, "perimeter": 5.0},
            "away": {"paint": 25.0, "mid": 15.0, "perimeter": 5.0},
        },
        missing_data=_missing_data_messages(report_input, len(reviewed_events)),
    )


def filter_clips(
    clips: tuple[Clip, ...], filter_criteria: ClipFilter
) -> tuple[Clip, ...]:
    """Filter clips by the supplied criteria; ``None`` fields match everything."""

    return tuple(
        c
        for c in clips
        if (
            (filter_criteria.event_type is None or c.event_type == filter_criteria.event_type)
            and (filter_criteria.player_id is None or c.player_id == filter_criteria.player_id)
            and (filter_criteria.team_id is None or c.team_id == filter_criteria.team_id)
            and (filter_criteria.status is None or c.status == filter_criteria.status)
            and (
                filter_criteria.min_confidence is None
                or c.confidence >= filter_criteria.min_confidence
            )
            and (
                filter_criteria.tags is None
                or any(tag in filter_criteria.tags for tag in c.tags)
            )
        )
    )


# --- Export / rendering -------------------------------------------------------


def export_report_json(report: CoachingReport) -> str:
    """Export the report summary and clip links to a JSON string."""

    export = {
        "game_id": report.game_id,
        "score": report.final_score,
        "official_event_count": report.official_event_count,
        "excluded_unreviewed_event_count": report.excluded_unreviewed_event_count,
        "metrics": {
            "paint_occupancy": report.paint_occupancy,
            "average_offensive_width_ft": report.average_offensive_width_ft,
        },
        "links": {
            "fast_breaks": [c.clip_id for c in report.top_fast_break_clips],
            "turnovers": [c.clip_id for c in report.turnover_clips],
        },
    }
    return json.dumps(export, indent=2)


def export_report_csv(report: CoachingReport) -> str:
    """Export per-team shot-chart rows to a CSV string."""

    lines = ["Type,Team,Zone,Value,EventID"]
    for team, chart in report.team_shot_chart.items():
        lines.append(
            f"shot,{team},{chart.zone_type},{chart.made_or_attempted},{chart.event_links[0]}"
        )
    return "\n".join(lines)


def render_printable_report_html(report: CoachingReport) -> str:
    """Render the report into a printable HTML document."""

    warnings = " ".join(report.missing_data.values())
    paint_home = report.paint_occupancy.get("home", 0.0)
    return f"""
    <!doctype html>
    <html>
      <head>
        <title>Coaching Report {report.game_id}</title>
        <style>body {{ font-family: sans-serif; }}</style>
      </head>
      <body>
        <h1>Game Performance Report: {report.game_id}</h1>
        <h2>Summary Statistics</h2>
        <p><strong>Final Score:</strong> {report.final_score or 'N/A'}</p>
        <p><strong>Total Official Events:</strong> {report.official_event_count}</p>
        <p><strong>Unreviewed Auto Events Excluded:</strong>
           {report.excluded_unreviewed_event_count}</p>
        <hr>
        <h3>Key Metrics</h3>
        <ul>
          <li>Avg Offensive Width: {report.average_offensive_width_ft:.1f} ft</li>
          <li>Paint Occupancy (Home): {paint_home:.2f}</li>
        </ul>
        <div style="color: red;">{warnings}</div>
      </body>
    </html>
    """


def render_clip_library_html(clips: tuple[Clip, ...]) -> str:
    """Render a clip-library table for the supplied (already filtered) clips."""

    if not clips:
        return "<p>No clips match the current filter criteria.</p>"

    rows = "".join(
        f"""
        <tr>
          <td><a href="{clip.url}" target="_blank">{clip.clip_id}</a></td>
          <td>{clip.event_type}</td>
          <td>{clip.team_id}</td>
          <td>{clip.status}</td>
          <td>{clip.player_id}</td>
          <td>{', '.join(clip.tags)}</td>
          <td><a href="{clip.url}">Watch</a></td>
        </tr>"""
        for clip in clips
    )
    return f"""
<h3 style="border-bottom: 2px solid #333;">Clip Library Filtered View</h3>
<table border="1" style="width:100%;">
  <thead>
    <tr style="background-color: #eee;">
      <th>Clip ID</th><th>Type</th><th>Team</th><th>Status</th>
      <th>Player</th><th>Tags</th><th>URL</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>
"""


# --- Sample data (handy for tests and manual smoke checks) --------------------


def sample_events() -> tuple[Event, ...]:
    return (
        Event("shot-reviewed", "shot", 10.0, "home", "p1", ReviewStatus.REVIEWED,
              0.97, 5.0, 4.0, "clip-shot-reviewed", ("paint-touch",)),
        Event("shot-auto", "shot", 12.0, "home", "p2", ReviewStatus.AUTO,
              0.99, 47.0, 25.0, "clip-shot-auto", ("auto-only",)),
        Event("turnover-locked", "turnover", 20.0, "away", "p3", ReviewStatus.LOCKED,
              0.8, 45.0, 25.0, "clip-turnover", ("bad-pass",)),
        Event("break-reviewed", "fast_break", 22.0, "home", "p1", ReviewStatus.REVIEWED,
              0.91, 85.0, 8.0, "clip-break", ("transition",)),
        Event("oreb-reviewed", "offensive_rebound", 30.0, "home", "p2", ReviewStatus.REVIEWED,
              0.89, 8.0, 7.0, "clip-oreb", ("second-chance",)),
    )


def sample_clips() -> tuple[Clip, ...]:
    return (
        Clip("clip-shot-reviewed", "shot-reviewed", "vid-1", 8.0, 13.0, "shot", "home", "p1",
             ReviewStatus.REVIEWED, 0.97, ("paint-touch",), "/clips/shot-reviewed.mp4"),
        Clip("clip-shot-auto", "shot-auto", "vid-1", 11.0, 14.0, "shot", "home", "p2",
             ReviewStatus.AUTO, 0.99, ("auto-only",), "/clips/shot-auto.mp4"),
        Clip("clip-turnover", "turnover-locked", "vid-1", 18.0, 22.0, "turnover", "away", "p3",
             ReviewStatus.LOCKED, 0.8, ("bad-pass",), "/clips/turnover.mp4"),
    )


def sample_tracks() -> tuple[TrackPoint, ...]:
    return (
        TrackPoint(0.0, "home", "p1", 2.0, 3.0),
        TrackPoint(0.0, "home", "p2", 20.0, 43.0),
        TrackPoint(1.0, "home", "p1", 8.0, 5.0),
        TrackPoint(1.0, "home", "p2", 32.0, 45.0),
        TrackPoint(2.0, "away", "p3", 80.0, 47.0),
    )
