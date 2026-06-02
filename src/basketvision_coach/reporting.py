"""Aggregate manual review events into a coach-facing report.

Bridges the manual tagging store (:mod:`basketvision_coach.review`) and the
report builder (:mod:`basketvision_coach.coaching_reports`): reviewed/locked
events count toward official totals, ``needs_review`` events are excluded, and
rejected events are dropped entirely.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from basketvision_coach.coaching_reports import (
    CoachingReportInput,
    Event,
    ReviewStatus,
    build_coaching_report,
)
from basketvision_coach.review import EVENT_TYPES, EventStatus, ReviewEvent

_STATUS_MAP = {
    EventStatus.REVIEWED: ReviewStatus.REVIEWED,
    EventStatus.LOCKED: ReviewStatus.LOCKED,
    EventStatus.NEEDS_REVIEW: ReviewStatus.AUTO,
}


def build_review_report(video_id: str, events: Sequence[ReviewEvent]) -> dict[str, Any]:
    """Return a report summary dict for ``events`` tagged on ``video_id``."""

    active = [event for event in events if event.status != EventStatus.REJECTED]
    report = build_coaching_report(
        CoachingReportInput(
            game_id=video_id,
            final_score=None,
            events=tuple(
                Event(
                    event_id=str(event.id),
                    event_type=event.type,
                    ts_s=event.start_s,
                    team_id=event.team_id or "",
                    player_id=event.player_id or "",
                    status=_STATUS_MAP.get(event.status, ReviewStatus.AUTO),
                    confidence=event.confidence,
                    court_x=0.0,
                    court_y=0.0,
                    clip_id=str(event.id),
                    tags=(),
                )
                for event in active
            ),
            clips=(),
            tracks=(),
        )
    )

    by_type = Counter(event.type for event in active)
    by_status = Counter(event.status.value for event in events)
    makes = by_type.get("make", 0)
    misses = by_type.get("miss", 0)
    attempts = makes + misses
    return {
        "total_tagged": len(events),
        "official_event_count": report.official_event_count,
        "excluded_unreviewed_event_count": report.excluded_unreviewed_event_count,
        "by_type": {event_type: by_type.get(event_type, 0) for event_type in EVENT_TYPES},
        "by_status": dict(by_status),
        "makes": makes,
        "misses": misses,
        "shooting_pct": round(makes / attempts * 100, 1) if attempts else None,
        "missing_data": report.missing_data,
        "event_types": list(EVENT_TYPES),
    }
