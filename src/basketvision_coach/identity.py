# ruff: noqa: E501
"""Identity review primitives for assigning tracks to teams and players."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from basketvision_coach.identity_models import CorrectionRow, Player, Team, TrackIdentityRow

IdentitySource = Literal["model", "model_plus_manual", "manual"]
StatLine = dict[str, int]


@dataclass(frozen=True, slots=True)
class RosterTeam:
    id: str
    name: str
    jersey_color: str | None = None


@dataclass(frozen=True, slots=True)
class RosterPlayer:
    id: str
    name: str
    team_id: str
    jersey_number: str | None = None


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
    id: int | None = None

    def __post_init__(self) -> None:
        validate_identity_segment(
            track_id=self.track_id,
            player_id=self.player_id,
            team_id=self.team_id,
            confidence=self.confidence,
            start_s=self.start_s,
            end_s=self.end_s,
        )

    def covers(self, time_s: float) -> bool:
        if self.start_s is None or self.end_s is None:
            return True
        return self.start_s <= time_s < self.end_s

    @property
    def segment_key(self) -> tuple[str, float | None, float | None]:
        return (self.track_id, self.start_s, self.end_s)


@dataclass(frozen=True, slots=True)
class Correction:
    """Audit record for a changed reviewed identity field."""

    entity_type: str
    entity_id: str
    field: str
    old_value: str | None
    new_value: str | None
    user: str
    changed_at: datetime
    id: int | None = None


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


class IdentityReviewService:
    """Repository/service for ``track_identities`` and auditable corrections."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def create_team(self, name: str, *, team_id: str | None = None, jersey_color: str | None = None) -> RosterTeam:
        if not name:
            raise ValueError("team name is required")
        normalized_id = team_id or _slug(name)
        with self._session_factory() as session:
            row = Team(id=normalized_id, name=name, jersey_color=jersey_color)
            session.add(row)
            session.commit()
            return _team_to_dataclass(row)

    def list_teams(self) -> list[RosterTeam]:
        with self._session_factory() as session:
            return [_team_to_dataclass(row) for row in session.scalars(select(Team).order_by(Team.name))]

    def create_player(
        self,
        name: str,
        team_id: str,
        *,
        player_id: str | None = None,
        jersey_number: str | None = None,
    ) -> RosterPlayer:
        if not name:
            raise ValueError("player name is required")
        normalized_id = player_id or _slug(f"{team_id}-{jersey_number or name}")
        with self._session_factory() as session:
            if session.get(Team, team_id) is None:
                raise LookupError(f"team not found: {team_id}")
            row = Player(id=normalized_id, name=name, team_id=team_id, jersey_number=jersey_number)
            session.add(row)
            session.commit()
            return _player_to_dataclass(row)

    def list_players(self, *, team_id: str | None = None) -> list[RosterPlayer]:
        stmt = select(Player).order_by(Player.team_id, Player.jersey_number, Player.name)
        if team_id is not None:
            stmt = stmt.where(Player.team_id == team_id)
        with self._session_factory() as session:
            return [_player_to_dataclass(row) for row in session.scalars(stmt)]

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
            raise ValueError("manual review assignments must use manual or model_plus_manual source")
        validate_identity_segment(
            track_id=track_id,
            player_id=player_id,
            team_id=team_id,
            confidence=confidence,
            start_s=start_s,
            end_s=end_s,
        )
        with self._session_factory() as session:
            self._validate_roster_refs(session, player_id=player_id, team_id=team_id)
            existing = _find_segment(session, track_id=track_id, start_s=start_s, end_s=end_s)
            if existing is None:
                row = TrackIdentityRow(
                    track_id=track_id,
                    player_id=player_id,
                    team_id=team_id,
                    confidence=confidence,
                    start_s=start_s,
                    end_s=end_s,
                    source=source,
                    reviewed=True,
                )
                session.add(row)
                session.commit()
                return _identity_to_dataclass(row)

            old_values = _identity_field_values(existing)
            existing.player_id = player_id
            existing.team_id = team_id
            existing.confidence = confidence
            existing.start_s = start_s
            existing.end_s = end_s
            existing.source = source
            existing.reviewed = True
            existing.updated_at = datetime.now(UTC)
            self._record_corrections(session, existing, old_values, user=user)
            session.commit()
            return _identity_to_dataclass(existing)

    def seed_team_identity_from_jersey_color(
        self,
        *,
        track_id: str,
        team_id: str,
        confidence: float,
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> TrackIdentity:
        """Initialize a team-only identity from jersey classification before review."""

        validate_identity_segment(
            track_id=track_id,
            player_id=None,
            team_id=team_id,
            confidence=confidence,
            start_s=start_s,
            end_s=end_s,
        )
        with self._session_factory() as session:
            self._validate_roster_refs(session, player_id=None, team_id=team_id)
            existing = _find_segment(session, track_id=track_id, start_s=start_s, end_s=end_s)
            if existing is None:
                row = TrackIdentityRow(
                    track_id=track_id,
                    player_id=None,
                    team_id=team_id,
                    confidence=confidence,
                    start_s=start_s,
                    end_s=end_s,
                    source="model",
                    reviewed=False,
                )
                session.add(row)
                session.commit()
                return _identity_to_dataclass(row)
            existing.team_id = team_id
            existing.confidence = confidence
            existing.source = "model"
            existing.reviewed = False
            session.commit()
            return _identity_to_dataclass(existing)

    def list_track_identities(self, *, track_id: str | None = None) -> list[TrackIdentity]:
        stmt = select(TrackIdentityRow).order_by(TrackIdentityRow.track_id, TrackIdentityRow.start_s)
        if track_id is not None:
            stmt = stmt.where(TrackIdentityRow.track_id == track_id)
        with self._session_factory() as session:
            return [_identity_to_dataclass(row) for row in session.scalars(stmt)]

    def list_corrections(self, *, entity_type: str | None = None) -> list[Correction]:
        stmt = select(CorrectionRow).order_by(CorrectionRow.id)
        if entity_type is not None:
            stmt = stmt.where(CorrectionRow.entity_type == entity_type)
        with self._session_factory() as session:
            return [_correction_to_dataclass(row) for row in session.scalars(stmt)]

    def reviewed_identity_for(self, track_id: str, time_s: float) -> TrackIdentity | None:
        return self._best_identity(track_id, time_s, reviewed_only=True)

    def identity_for(self, track_id: str, time_s: float) -> TrackIdentity | None:
        reviewed = self._best_identity(track_id, time_s, reviewed_only=True)
        if reviewed is not None:
            return reviewed
        return self._best_identity(track_id, time_s, reviewed_only=False)

    def player_stat_report(self, observations: Iterable[PlayerStatObservation]) -> dict[str, StatLine]:
        """Attribute player stats only through reviewed identity assignments."""

        totals: dict[str, StatLine] = defaultdict(lambda: {"points": 0, "rebounds": 0, "assists": 0})
        for observation in observations:
            identity = self.reviewed_identity_for(observation.track_id, observation.time_s)
            if identity is None or identity.player_id is None:
                continue
            totals[identity.player_id]["points"] += observation.points
            totals[identity.player_id]["rebounds"] += observation.rebounds
            totals[identity.player_id]["assists"] += observation.assists
        return dict(totals)

    def team_spacing_report(self, observations: Iterable[SpacingObservation]) -> dict[str, float]:
        """Average spacing by team, including team-known tracks without players."""

        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for observation in observations:
            identity = self.identity_for(observation.track_id, observation.time_s)
            if identity is None or identity.team_id is None:
                continue
            totals[identity.team_id] += observation.spacing_m
            counts[identity.team_id] += 1
        return {team_id: totals[team_id] / counts[team_id] for team_id in totals}

    def _best_identity(self, track_id: str, time_s: float, *, reviewed_only: bool) -> TrackIdentity | None:
        with self._session_factory() as session:
            rows = list(session.scalars(select(TrackIdentityRow).where(TrackIdentityRow.track_id == track_id)))
            candidates = [
                _identity_to_dataclass(row)
                for row in rows
                if _row_covers(row, time_s) and (row.reviewed or not reviewed_only)
            ]
        if not candidates:
            return None
        return max(candidates, key=_identity_specificity)

    def _validate_roster_refs(
        self, session: Session, *, player_id: str | None, team_id: str | None
    ) -> None:
        if team_id is not None and session.get(Team, team_id) is None:
            raise LookupError(f"team not found: {team_id}")
        if player_id is None:
            return
        player = session.get(Player, player_id)
        if player is None:
            raise LookupError(f"player not found: {player_id}")
        if team_id is not None and player.team_id != team_id:
            raise ValueError("player_id must belong to team_id")

    def _record_corrections(
        self,
        session: Session,
        row: TrackIdentityRow,
        old_values: dict[str, str | None],
        *,
        user: str,
    ) -> None:
        changed_at = datetime.now(UTC)
        new_values = _identity_field_values(row)
        for field in ("player_id", "team_id", "confidence", "start_s", "end_s", "source", "reviewed"):
            if old_values[field] == new_values[field]:
                continue
            session.add(
                CorrectionRow(
                    entity_type="track_identity",
                    entity_id=row.track_id,
                    field=field,
                    old_value=old_values[field],
                    new_value=new_values[field],
                    user=user,
                    changed_at=changed_at,
                )
            )


def validate_identity_segment(
    *,
    track_id: str,
    player_id: str | None,
    team_id: str | None,
    confidence: float,
    start_s: float | None,
    end_s: float | None,
) -> None:
    if not track_id:
        raise ValueError("track_id is required")
    if team_id is None and player_id is None:
        raise ValueError("player_id or team_id is required")
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    if (start_s is None) != (end_s is None):
        raise ValueError("start_s and end_s must both be set for bounded assignments")
    if start_s is not None and end_s is not None and start_s >= end_s:
        raise ValueError("start_s must be before end_s")


def _find_segment(
    session: Session, *, track_id: str, start_s: float | None, end_s: float | None
) -> TrackIdentityRow | None:
    rows = session.scalars(select(TrackIdentityRow).where(TrackIdentityRow.track_id == track_id))
    for row in rows:
        if row.start_s == start_s and row.end_s == end_s:
            return row
    return None


def _row_covers(row: TrackIdentityRow, time_s: float) -> bool:
    if row.start_s is None or row.end_s is None:
        return True
    return row.start_s <= time_s < row.end_s


def _identity_specificity(identity: TrackIdentity) -> tuple[int, float]:
    bounded = int(identity.start_s is not None and identity.end_s is not None)
    return (bounded, identity.confidence)


def _identity_field_values(row: TrackIdentityRow) -> dict[str, str | None]:
    return {
        "player_id": row.player_id,
        "team_id": row.team_id,
        "confidence": str(row.confidence),
        "start_s": None if row.start_s is None else str(row.start_s),
        "end_s": None if row.end_s is None else str(row.end_s),
        "source": row.source,
        "reviewed": str(row.reviewed),
    }


def _identity_to_dataclass(row: TrackIdentityRow) -> TrackIdentity:
    return TrackIdentity(
        id=row.id,
        track_id=row.track_id,
        player_id=row.player_id,
        team_id=row.team_id,
        confidence=row.confidence,
        start_s=row.start_s,
        end_s=row.end_s,
        source=_coerce_source(row.source),
        reviewed=row.reviewed,
    )


def _correction_to_dataclass(row: CorrectionRow) -> Correction:
    changed_at = row.changed_at
    if changed_at.tzinfo is None:
        changed_at = changed_at.replace(tzinfo=UTC)
    return Correction(
        id=row.id,
        entity_type=row.entity_type,
        entity_id=row.entity_id,
        field=row.field,
        old_value=row.old_value,
        new_value=row.new_value,
        user=row.user,
        changed_at=changed_at,
    )


def _team_to_dataclass(row: Team) -> RosterTeam:
    return RosterTeam(id=row.id, name=row.name, jersey_color=row.jersey_color)


def _player_to_dataclass(row: Player) -> RosterPlayer:
    return RosterPlayer(
        id=row.id,
        name=row.name,
        team_id=row.team_id,
        jersey_number=row.jersey_number,
    )


def _coerce_source(value: str) -> IdentitySource:
    if value not in {"model", "model_plus_manual", "manual"}:
        raise ValueError(f"unknown identity source: {value}")
    return value  # type: ignore[return-value]


def _slug(value: str) -> str:
    slug = "-".join("".join(ch.lower() if ch.isalnum() else " " for ch in value).split())
    if not slug:
        raise ValueError("could not derive identifier")
    return slug
