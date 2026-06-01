"""Manual video review, event tagging, and clip export workflows."""

from __future__ import annotations

import csv
import html
import sqlite3
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, cast

EVENT_TYPES: tuple[str, ...] = (
    "shot",
    "make",
    "miss",
    "turnover",
    "rebound",
    "fast_break",
    "defensive_error",
    "good_action",
)
EVENT_TYPE_SHORTCUTS: dict[str, str] = {
    "shot": "1",
    "make": "2",
    "miss": "3",
    "turnover": "4",
    "rebound": "5",
    "fast_break": "6",
    "defensive_error": "7",
    "good_action": "8",
}
CSV_FIELDS: tuple[str, ...] = (
    "id",
    "video_id",
    "type",
    "start_s",
    "end_s",
    "team_id",
    "player_id",
    "confidence",
    "source",
    "status",
    "reviewed",
)


class EventStatus(StrEnum):
    """Review lifecycle state for a tagged event."""

    NEEDS_REVIEW = "needs_review"
    REVIEWED = "reviewed"
    REJECTED = "rejected"
    LOCKED = "locked"


@dataclass(frozen=True, slots=True)
class ReviewEventCreate:
    """Input for creating a review event."""

    video_id: str
    type: str
    start_s: float
    end_s: float
    team_id: str | None = None
    player_id: str | None = None
    confidence: float = 1.0
    source: str = "manual"
    status: EventStatus = EventStatus.REVIEWED
    reviewed: bool = True


@dataclass(frozen=True, slots=True)
class ReviewEvent:
    """A tagged event on a video timeline."""

    id: int
    video_id: str
    type: str
    start_s: float
    end_s: float
    team_id: str | None
    player_id: str | None
    confidence: float
    source: str
    status: EventStatus
    reviewed: bool


@dataclass(frozen=True, slots=True)
class ClipRecord:
    """A generated video clip for a review event."""

    id: int
    event_id: int
    path: str
    pre_roll_s: float
    post_roll_s: float


class CommandRunner(Protocol):
    def __call__(self, command: list[str]) -> None: ...


def _default_runner(command: list[str]) -> None:
    subprocess.run(command, check=True)


class ReviewStore:
    """SQLite-backed store for manual review events and exported clips."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def create_event(self, event: ReviewEventCreate) -> ReviewEvent:
        self._validate_event_values(
            type=event.type,
            start_s=event.start_s,
            end_s=event.end_s,
            confidence=event.confidence,
        )
        with self._connect() as conn:
            cursor = conn.execute(
                """
                insert into events (
                    video_id, type, start_s, end_s, team_id, player_id,
                    confidence, source, status, reviewed
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.video_id,
                    event.type,
                    event.start_s,
                    event.end_s,
                    event.team_id,
                    event.player_id,
                    event.confidence,
                    event.source,
                    event.status.value,
                    int(event.reviewed),
                ),
            )
            event_id = _lastrowid(cursor)
        return self.get_event(event_id)

    def get_event(self, event_id: int) -> ReviewEvent:
        with self._connect() as conn:
            row = conn.execute("select * from events where id = ?", (event_id,)).fetchone()
        if row is None:
            raise KeyError(f"event {event_id} does not exist")
        return _event_from_row(row)

    def list_events(self, video_id: str) -> list[ReviewEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                "select * from events where video_id = ? order by start_s, id", (video_id,)
            ).fetchall()
        return [_event_from_row(row) for row in rows]

    def update_event(self, event_id: int, **changes: object) -> ReviewEvent:
        if not changes:
            return self.get_event(event_id)

        allowed = {
            "type",
            "start_s",
            "end_s",
            "team_id",
            "player_id",
            "confidence",
            "source",
            "status",
            "reviewed",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"unsupported event fields: {', '.join(sorted(unknown))}")

        current = self.get_event(event_id)
        merged_type = cast(str, changes.get("type", current.type))
        merged_start_s = cast(float, changes.get("start_s", current.start_s))
        merged_end_s = cast(float, changes.get("end_s", current.end_s))
        merged_confidence = cast(float, changes.get("confidence", current.confidence))
        self._validate_event_values(
            type=merged_type,
            start_s=merged_start_s,
            end_s=merged_end_s,
            confidence=merged_confidence,
        )

        assignments: list[str] = []
        values: list[object] = []
        for key, value in changes.items():
            assignments.append(f"{key} = ?")
            if isinstance(value, EventStatus):
                values.append(value.value)
            elif key == "reviewed":
                values.append(int(bool(value)))
            else:
                values.append(value)
        values.append(event_id)

        with self._connect() as conn:
            conn.execute(f"update events set {', '.join(assignments)} where id = ?", values)
        return self.get_event(event_id)

    def accept_event(self, event_id: int) -> ReviewEvent:
        return self.update_event(event_id, status=EventStatus.REVIEWED, reviewed=True)

    def reject_event(self, event_id: int) -> ReviewEvent:
        return self.update_event(event_id, status=EventStatus.REJECTED, reviewed=False)

    def lock_event(self, event_id: int) -> ReviewEvent:
        return self.update_event(event_id, status=EventStatus.LOCKED, reviewed=True)

    def delete_event(self, event_id: int) -> None:
        with self._connect() as conn:
            conn.execute("delete from clips where event_id = ?", (event_id,))
            conn.execute("delete from events where id = ?", (event_id,))

    def add_clip(
        self, event_id: int, path: str | Path, pre_roll_s: float, post_roll_s: float
    ) -> ClipRecord:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                insert into clips (event_id, path, pre_roll_s, post_roll_s)
                values (?, ?, ?, ?)
                """,
                (event_id, str(path), pre_roll_s, post_roll_s),
            )
            clip_id = _lastrowid(cursor)
        return ClipRecord(
            id=clip_id,
            event_id=event_id,
            path=str(path),
            pre_roll_s=pre_roll_s,
            post_roll_s=post_roll_s,
        )

    def export_events_csv(
        self, video_id: str, output_path: str | Path, *, include_all: bool = False
    ) -> Path:
        events = self.list_events(video_id)
        if not include_all:
            events = [
                event
                for event in events
                if event.status in {EventStatus.REVIEWED, EventStatus.LOCKED}
                and event.source != "auto"
                and event.reviewed
            ]

        csv_path = Path(output_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for event in events:
                writer.writerow(_event_to_csv_row(event))
        return csv_path

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                create table if not exists events (
                    id integer primary key autoincrement,
                    video_id text not null,
                    type text not null,
                    start_s real not null,
                    end_s real not null,
                    team_id text,
                    player_id text,
                    confidence real not null,
                    source text not null,
                    status text not null,
                    reviewed integer not null
                );

                create table if not exists clips (
                    id integer primary key autoincrement,
                    event_id integer not null references events(id) on delete cascade,
                    path text not null,
                    pre_roll_s real not null,
                    post_roll_s real not null
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma foreign_keys = on")
        return conn

    @staticmethod
    def _validate_event_values(
        *, type: str, start_s: float, end_s: float, confidence: float
    ) -> None:
        if type not in EVENT_TYPES:
            raise ValueError(f"unsupported event type: {type}")
        if start_s < 0:
            raise ValueError("start_s must be non-negative")
        if end_s < start_s:
            raise ValueError("end_s must be greater than or equal to start_s")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")


class ClipExporter:
    """Generate event clips with ffmpeg and persist clip records."""

    def __init__(self, store: ReviewStore, *, runner: CommandRunner = _default_runner) -> None:
        self.store = store
        self.runner = runner

    def export_clips(
        self,
        *,
        video_id: str,
        source_video: str | Path,
        event_ids: Iterable[int],
        output_dir: str | Path,
        pre_roll_s: float = 2.0,
        post_roll_s: float = 2.0,
    ) -> list[ClipRecord]:
        if pre_roll_s < 0 or post_roll_s < 0:
            raise ValueError("pre-roll and post-roll must be non-negative")

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        clips: list[ClipRecord] = []
        for event_id in event_ids:
            event = self.store.get_event(event_id)
            if event.video_id != video_id:
                raise ValueError(f"event {event_id} does not belong to video {video_id}")
            start = max(0.0, event.start_s - pre_roll_s)
            duration = event.end_s - start + post_roll_s
            clip_path = destination / f"{video_id}-event-{event.id}-{event.type}.mp4"
            command = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(source_video),
                "-t",
                f"{duration:.3f}",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(clip_path),
            ]
            self.runner(command)
            if not clip_path.exists():
                raise FileNotFoundError(f"clip exporter did not create {clip_path}")
            clips.append(self.store.add_clip(event.id, clip_path, pre_roll_s, post_roll_s))
        return clips


def render_review_page(video_id: str, video_url: str, events: Sequence[ReviewEvent]) -> str:
    """Render a self-contained browser review page with synchronized tagging controls."""

    event_buttons = "\n".join(
        f'<button type="button" data-shortcut="{html.escape(shortcut)}" '
        f'onclick="createManualEvent(\'{html.escape(event_type)}\')">'
        f'{html.escape(shortcut)} · {html.escape(event_type.replace("_", " ").title())}</button>'
        for event_type, shortcut in EVENT_TYPE_SHORTCUTS.items()
    )
    event_rows = "\n".join(_render_event_row(event) for event in events)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>BasketVision Review · {html.escape(video_id)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 1rem; background: #111; color: #f5f5f5; }}
    video {{ width: min(100%, 960px); display: block; background: #000; }}
    button {{ margin: .25rem; padding: .45rem .7rem; }}
    tr.active {{ background: #273b62; }}
    table {{ border-collapse: collapse; width: min(100%, 960px); margin-top: 1rem; }}
    td, th {{ border-bottom: 1px solid #333; padding: .45rem; text-align: left; }}
  </style>
</head>
<body data-video-id="{html.escape(video_id)}">
  <h1>Manual Review</h1>
  <video id="player" controls src="{html.escape(video_url)}"></video>
  <section aria-label="keyboard shortcuts">
    <p>
      Shortcuts: Space play/pause, ArrowLeft frame back, ArrowRight frame forward,
      J/K jump, 1-8 quick tags.
    </p>
    {event_buttons}
  </section>
  <table aria-label="event list">
    <thead><tr><th>Time</th><th>Type</th><th>Status</th><th>Player</th><th>Actions</th></tr></thead>
    <tbody id="event-list">{event_rows}</tbody>
  </table>
  <script>
    const player = document.getElementById('player');
    const shortcutTypes = {EVENT_TYPE_SHORTCUTS!r};
    function seekToEvent(startSeconds) {{
      player.currentTime = Number(startSeconds);
      player.focus();
    }}
    function activeRows() {{
      const now = player.currentTime;
      document.querySelectorAll('[data-start-s]').forEach(row => {{
        const start = Number(row.dataset.startS);
        const end = Number(row.dataset.endS);
        row.classList.toggle('active', now >= start && now <= end);
      }});
    }}
    function createManualEvent(type) {{
      const start = Math.max(0, player.currentTime - 1);
      const end = player.currentTime + 1;
      const payload = {{
        type,
        start_s: start,
        end_s: end,
        source: 'manual',
        status: 'reviewed',
        reviewed: true
      }};
      document.dispatchEvent(new CustomEvent('basketvision:create-event', {{detail: payload}}));
      return payload;
    }}
    function frameStep(delta) {{
      player.currentTime = Math.max(0, player.currentTime + delta / 30);
    }}
    function jump(delta) {{ player.currentTime = Math.max(0, player.currentTime + delta); }}
    document.addEventListener('keydown', event => {{
      if (event.target && ['INPUT', 'SELECT', 'TEXTAREA'].includes(event.target.tagName)) return;
      if (event.code === 'Space') {{
        event.preventDefault();
        player.paused ? player.play() : player.pause();
      }}
      if (event.code === 'ArrowLeft') {{ event.preventDefault(); frameStep(-1); }}
      if (event.code === 'ArrowRight') {{ event.preventDefault(); frameStep(1); }}
      if (event.key === 'j' || event.key === 'J') jump(-5);
      if (event.key === 'k' || event.key === 'K') jump(5);
      if (shortcutTypes[event.key]) createManualEvent(shortcutTypes[event.key]);
    }});
    player.addEventListener('timeupdate', activeRows);
  </script>
</body>
</html>"""


def _render_event_row(event: ReviewEvent) -> str:
    return (
        f'<tr data-start-s="{event.start_s:.3f}" data-end-s="{event.end_s:.3f}" '
        f'onclick="seekToEvent({event.start_s:.3f})">'
        f"<td>{event.start_s:.2f}-{event.end_s:.2f}</td>"
        f"<td>{html.escape(event.type)}</td>"
        f"<td>{html.escape(event.status.value)}</td>"
        f"<td>{html.escape(event.player_id or '')}</td>"
        f"<td>accept · reject · edit · delete</td>"
        "</tr>"
    )


def _lastrowid(cursor: sqlite3.Cursor) -> int:
    if cursor.lastrowid is None:
        raise RuntimeError("sqlite did not return a new row id")
    return cursor.lastrowid


def _event_from_row(row: sqlite3.Row) -> ReviewEvent:
    return ReviewEvent(
        id=int(row["id"]),
        video_id=str(row["video_id"]),
        type=str(row["type"]),
        start_s=float(row["start_s"]),
        end_s=float(row["end_s"]),
        team_id=row["team_id"] if row["team_id"] is None else str(row["team_id"]),
        player_id=row["player_id"] if row["player_id"] is None else str(row["player_id"]),
        confidence=float(row["confidence"]),
        source=str(row["source"]),
        status=EventStatus(str(row["status"])),
        reviewed=bool(row["reviewed"]),
    )


def _event_to_csv_row(event: ReviewEvent) -> dict[str, object]:
    return {
        "id": event.id,
        "video_id": event.video_id,
        "type": event.type,
        "start_s": f"{event.start_s:.3f}",
        "end_s": f"{event.end_s:.3f}",
        "team_id": event.team_id or "",
        "player_id": event.player_id or "",
        "confidence": f"{event.confidence:.3f}",
        "source": event.source,
        "status": event.status.value,
        "reviewed": str(event.reviewed).lower(),
    }
