"""Identity review primitives for assigning tracks to teams and players."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

IdentitySource = Literal["model", "model_plus_manual", "manual"]
StatLine = dict[str, int]


@dataclass(frozen=True, slots=True)
class TrackIdentity:
    """Reviewed or model-seeded identity for a track or track segment."""

    track_id: str
    player_id: str | None
    team_id: str | None
    confidence: float
    start_s: float | None
    end_s: float | None
    source: IdentitySource
    reviewed: bool

    def __post_init__(self) -> None:
        if not self.track_id:
            raise ValueError("track_id is required")
        if self.team_id is None and self.player_id is None:
            raise ValueError("player_id or team_id is required")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if (self.start_s is None) != (self.end_s is None):
            raise ValueError("start_s and end_s must both be set for bounded assignments")
        if self.start_s is not None and self.end_s is not None and self.start_s >= self.end_s:
            raise ValueError("start_s must be before end_s")

    def covers(self, time_s: float) -> bool:
        """Return whether this identity covers an observation timestamp."""

        if self.start_s is None or self.end_s is None:
            return True
        return self.start_s <= time_s < self.end_s

    @property
    def segment_key(self) -> tuple[str, float | None, float | None]:
        """Stable key for the assigned track segment."""

        return (self.track_id, self.start_s, self.end_s)


@dataclass(frozen=True, slots=True)
class Correction:
    """Audit record for a changed reviewed identity field."""

    entity_type: str
    entity_id: str
    field: str
    old_value: object
    new_value: object
    user: str
    changed_at: datetime


@dataclass(frozen=True, slots=True)
class PlayerStatObservation:
    """Track-level stat event emitted by the vision or event pipeline."""

    track_id: str
    time_s: float
    points: int = 0
    rebounds: int = 0
    assists: int = 0


@dataclass(frozen=True, slots=True)
class SpacingObservation:
    """Track-level spacing measurement used for team spacing reports."""

    track_id: str
    time_s: float
    spacing_m: float


class IdentityReviewStore:
    """In-memory review table facade for track identities and corrections.

    The scaffold has no database layer yet, so this class models the tables the
    API/UI layer will persist later: ``track_identities`` and ``corrections``.
    Review tools can call ``assign_track_segment`` after a click on a full track
    or after selecting a bounded interval on the event/tracks page.
    """

    def __init__(self) -> None:
        self.track_identities: list[TrackIdentity] = []
        self.corrections: list[Correction] = []

    def assign_track_segment(
        self,
        *,
        track_id: str,
        player_id: str | None,
        team_id: str | None,
        user: str,
        start_s: float | None = None,
        end_s: float | None = None,
        confidence: float = 1.0,
        source: IdentitySource = "manual",
    ) -> TrackIdentity:
        """Create or update a reviewed manual assignment for a track segment."""

        if not user:
            raise ValueError("user is required for auditable manual assignments")
        if source not in ("manual", "model_plus_manual"):
            raise ValueError(
                "manual review assignments must use manual or model_plus_manual source"
            )

        new_identity = TrackIdentity(
            track_id=track_id,
            player_id=player_id,
            team_id=team_id,
            confidence=confidence,
            start_s=start_s,
            end_s=end_s,
            source=source,
            reviewed=True,
        )
        existing_index = self._find_segment_index(new_identity.segment_key)
        if existing_index is None:
            self.track_identities.append(new_identity)
            return new_identity

        old_identity = self.track_identities[existing_index]
        self.track_identities[existing_index] = new_identity
        self._record_corrections(old_identity, new_identity, user=user)
        return new_identity

    def seed_team_identity_from_jersey_color(
        self,
        *,
        track_id: str,
        team_id: str,
        confidence: float,
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> TrackIdentity:
        """Initialize team identity from jersey classification before review."""

        identity = TrackIdentity(
            track_id=track_id,
            player_id=None,
            team_id=team_id,
            confidence=confidence,
            start_s=start_s,
            end_s=end_s,
            source="model",
            reviewed=False,
        )
        self.track_identities.append(identity)
        return identity

    def player_stat_report(self, observations: list[PlayerStatObservation]) -> dict[str, StatLine]:
        """Attribute track-level stat events to reviewed player identities."""

        totals: dict[str, StatLine] = defaultdict(
            lambda: {"points": 0, "rebounds": 0, "assists": 0}
        )
        for observation in observations:
            identity = self.reviewed_identity_for(observation.track_id, observation.time_s)
            if identity is None or identity.player_id is None:
                continue
            totals[identity.player_id]["points"] += observation.points
            totals[identity.player_id]["rebounds"] += observation.rebounds
            totals[identity.player_id]["assists"] += observation.assists
        return dict(totals)

    def team_spacing_report(self, observations: list[SpacingObservation]) -> dict[str, float]:
        """Average spacing by team, including tracks with team-only identities."""

        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for observation in observations:
            identity = self.identity_for(observation.track_id, observation.time_s)
            if identity is None or identity.team_id is None:
                continue
            totals[identity.team_id] += observation.spacing_m
            counts[identity.team_id] += 1
        return {team_id: totals[team_id] / counts[team_id] for team_id in totals}

    def reviewed_identity_for(self, track_id: str, time_s: float) -> TrackIdentity | None:
        """Return the reviewed assignment for a track at a timestamp, if any."""

        return self._best_identity(track_id, time_s, reviewed_only=True)

    def identity_for(self, track_id: str, time_s: float) -> TrackIdentity | None:
        """Return the best known identity for a track at a timestamp."""

        reviewed = self._best_identity(track_id, time_s, reviewed_only=True)
        if reviewed is not None:
            return reviewed
        return self._best_identity(track_id, time_s, reviewed_only=False)

    def _find_segment_index(
        self, segment_key: tuple[str, float | None, float | None]
    ) -> int | None:
        for index, identity in enumerate(self.track_identities):
            if identity.segment_key == segment_key:
                return index
        return None

    def _record_corrections(self, old: TrackIdentity, new: TrackIdentity, *, user: str) -> None:
        changed_at = datetime.now(UTC)
        for field in (
            "player_id",
            "team_id",
            "confidence",
            "start_s",
            "end_s",
            "source",
            "reviewed",
        ):
            old_value = getattr(old, field)
            new_value = getattr(new, field)
            if old_value == new_value:
                continue
            self.corrections.append(
                Correction(
                    entity_type="track_identity",
                    entity_id=old.track_id,
                    field=field,
                    old_value=old_value,
                    new_value=new_value,
                    user=user,
                    changed_at=changed_at,
                )
            )

    def _best_identity(
        self, track_id: str, time_s: float, *, reviewed_only: bool
    ) -> TrackIdentity | None:
        candidates = [
            identity
            for identity in self.track_identities
            if identity.track_id == track_id
            and identity.covers(time_s)
            and (identity.reviewed or not reviewed_only)
        ]
        if not candidates:
            return None
        return max(candidates, key=_identity_specificity)


def _identity_specificity(identity: TrackIdentity) -> tuple[int, float]:
    bounded = int(identity.start_s is not None and identity.end_s is not None)
    return (bounded, identity.confidence)
