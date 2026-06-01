# ruff: noqa: E501
from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, ConfigDict

from basketvision_coach.db import build_session_factory
from basketvision_coach.identity import (
    IdentityReviewService,
    RosterPlayer,
    RosterTeam,
    TrackIdentity,
)
from basketvision_coach.video import VideoIngestService
from basketvision_coach.video_models import Game, Video, VideoState


class GameCreate(BaseModel):
    name: str


class GameRead(BaseModel):
    id: int
    name: str
    created_at: str

    model_config = ConfigDict(from_attributes=True)


class VideoRead(BaseModel):
    id: int
    game_id: int
    state: str
    original_path: str
    proxy_path: str | None
    proxy_url: str | None
    hls_path: str | None
    fps: float | None
    width: int | None
    height: int | None
    duration_s: float | None
    error_message: str | None
    ts_mapping: str


class TeamCreate(BaseModel):
    name: str
    id: str | None = None
    jersey_color: str | None = None


class TeamRead(BaseModel):
    id: str
    name: str
    jersey_color: str | None


class PlayerCreate(BaseModel):
    name: str
    team_id: str
    id: str | None = None
    jersey_number: str | None = None


class PlayerRead(BaseModel):
    id: str
    name: str
    team_id: str
    jersey_number: str | None


class TrackIdentityAssign(BaseModel):
    track_id: str
    player_id: str | None = None
    team_id: str | None = None
    user: str
    start_s: float | None = None
    end_s: float | None = None
    confidence: float = 1.0
    source: str = "manual"


class TrackIdentityRead(BaseModel):
    id: int | None
    track_id: str
    player_id: str | None
    team_id: str | None
    confidence: float
    start_s: float | None
    end_s: float | None
    source: str
    reviewed: bool


def default_session_factory() -> object:
    return build_session_factory()


def default_video_service() -> VideoIngestService:
    return VideoIngestService(session_factory=build_session_factory())


def video_to_read(video: Video) -> VideoRead:
    proxy_url = f"/api/videos/{video.id}/proxy_720p.mp4" if video.proxy_path else None
    return VideoRead(
        id=video.id,
        game_id=video.game_id,
        state=video.state.value,
        original_path=video.original_path,
        proxy_path=video.proxy_path,
        proxy_url=proxy_url,
        hls_path=video.hls_path,
        fps=video.fps,
        width=video.width,
        height=video.height,
        duration_s=video.duration_s,
        error_message=video.error_message,
        ts_mapping="ts_s = frame_idx / fps",
    )


def team_to_read(team: RosterTeam) -> TeamRead:
    return TeamRead(id=team.id, name=team.name, jersey_color=team.jersey_color)


def player_to_read(player: RosterPlayer) -> PlayerRead:
    return PlayerRead(
        id=player.id,
        name=player.name,
        team_id=player.team_id,
        jersey_number=player.jersey_number,
    )


def identity_to_read(identity: TrackIdentity) -> TrackIdentityRead:
    return TrackIdentityRead(
        id=identity.id,
        track_id=identity.track_id,
        player_id=identity.player_id,
        team_id=identity.team_id,
        confidence=identity.confidence,
        start_s=identity.start_s,
        end_s=identity.end_s,
        source=identity.source,
        reviewed=identity.reviewed,
    )


def create_app(
    service: VideoIngestService | None = None,
    *,
    video_service: VideoIngestService | None = None,
    identity_service: IdentityReviewService | None = None,
) -> FastAPI:
    if service is not None and video_service is not None:
        raise ValueError("pass either service or video_service, not both")
    if service is not None:
        video_service = service
    if video_service is None or identity_service is None:
        session_factory = build_session_factory()
        video_service = video_service or VideoIngestService(session_factory=session_factory)
        identity_service = identity_service or IdentityReviewService(session_factory=session_factory)
    app = FastAPI(title="BasketVision Coach")

    @app.post("/api/games", response_model=GameRead, status_code=201)
    def create_game(payload: GameCreate) -> GameRead:
        try:
            game = video_service.create_game(payload.name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return GameRead(id=game.id, name=game.name, created_at=game.created_at.isoformat())

    @app.get("/api/games", response_model=list[GameRead])
    def list_games() -> list[GameRead]:
        return [
            GameRead(id=game.id, name=game.name, created_at=game.created_at.isoformat())
            for game in video_service.list_games()
        ]

    @app.post("/api/games/{game_id}/video", response_model=VideoRead, status_code=201)
    async def upload_video(
        game_id: int,
        background_tasks: BackgroundTasks,
        file: Annotated[UploadFile, File()],
        process_inline: bool = True,
    ) -> VideoRead:
        if not file.filename or Path(file.filename).suffix.lower() != ".mp4":
            raise HTTPException(status_code=400, detail="Only MP4 uploads are supported")
        contents = await file.read()
        try:
            video = video_service.ingest_game_video(
                game_id,
                file.filename,
                contents,
                run_transcode=process_inline,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not process_inline:
            background_tasks.add_task(video_service.transcode_video, video.id)
        return video_to_read(video)

    @app.get("/api/games/{game_id}/video", response_model=VideoRead)
    def get_latest_video(game_id: int) -> VideoRead:
        video = video_service.latest_video_for_game(game_id)
        if video is None:
            raise HTTPException(status_code=404, detail="No video has been uploaded for this game")
        return video_to_read(video)

    @app.get("/api/videos/{video_id}/proxy_720p.mp4")
    def get_proxy(video_id: int) -> FileResponse:
        video = video_service.get_video(video_id)
        if video is None:
            raise HTTPException(status_code=404, detail="Video not found")
        if video.state is not VideoState.READY or video.proxy_path is None:
            raise HTTPException(status_code=409, detail="Video proxy is not ready")
        proxy = Path(video.proxy_path)
        if not proxy.exists():
            raise HTTPException(status_code=404, detail="Proxy file is missing")
        return FileResponse(proxy, media_type="video/mp4", filename="proxy_720p.mp4")

    @app.post("/api/roster/teams", response_model=TeamRead, status_code=201)
    def create_team(payload: TeamCreate) -> TeamRead:
        try:
            team = identity_service.create_team(
                payload.name,
                team_id=payload.id,
                jersey_color=payload.jersey_color,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return team_to_read(team)

    @app.get("/api/roster/teams", response_model=list[TeamRead])
    def list_teams() -> list[TeamRead]:
        return [team_to_read(team) for team in identity_service.list_teams()]

    @app.post("/api/roster/players", response_model=PlayerRead, status_code=201)
    def create_player(payload: PlayerCreate) -> PlayerRead:
        try:
            player = identity_service.create_player(
                payload.name,
                payload.team_id,
                player_id=payload.id,
                jersey_number=payload.jersey_number,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return player_to_read(player)

    @app.get("/api/roster/players", response_model=list[PlayerRead])
    def list_players(team_id: str | None = None) -> list[PlayerRead]:
        return [player_to_read(player) for player in identity_service.list_players(team_id=team_id)]

    @app.post("/api/track-identities", response_model=TrackIdentityRead, status_code=201)
    def assign_track_identity(payload: TrackIdentityAssign) -> TrackIdentityRead:
        try:
            identity = identity_service.assign_track_segment(
                track_id=payload.track_id,
                player_id=payload.player_id,
                team_id=payload.team_id,
                user=payload.user,
                start_s=payload.start_s,
                end_s=payload.end_s,
                confidence=payload.confidence,
                source=payload.source,  # type: ignore[arg-type]
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return identity_to_read(identity)

    @app.get("/api/track-identities", response_model=list[TrackIdentityRead])
    def list_track_identities(track_id: str | None = None) -> list[TrackIdentityRead]:
        return [
            identity_to_read(identity)
            for identity in identity_service.list_track_identities(track_id=track_id)
        ]

    @app.get("/games/{game_id}/tracks", response_class=HTMLResponse)
    def tracks_page(game_id: int) -> HTMLResponse:
        game = video_service.get_game(game_id)
        if game is None:
            raise HTTPException(status_code=404, detail="Game not found")
        video = video_service.latest_video_for_game(game_id)
        return HTMLResponse(render_tracks_page(game, video, identity_service))

    @app.get("/games/{game_id}", response_class=HTMLResponse)
    def game_page(game_id: int) -> HTMLResponse:
        game = video_service.get_game(game_id)
        if game is None:
            raise HTTPException(status_code=404, detail="Game not found")
        video = video_service.latest_video_for_game(game_id)
        html = render_game_page(game, video)
        return HTMLResponse(html)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        games = video_service.list_games()
        items = "".join(f'<li><a href="/games/{game.id}">{escape(game.name)}</a></li>' for game in games)
        return HTMLResponse(f"<h1>BasketVision Coach</h1><ul>{items}</ul>")

    return app


def escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_game_page(game: Game, video: Video | None) -> str:
    if video is None:
        state = "No upload"
        player = "<p>No video uploaded yet.</p>"
        metadata = ""
    else:
        state = video.state.value
        if video.state is VideoState.READY and video.proxy_path:
            proxy_url = f"/api/videos/{video.id}/proxy_720p.mp4"
            player = f'<video controls src="{proxy_url}" style="max-width: 100%;"></video>'
        elif video.state is VideoState.FAILED:
            player = f"<p>Transcode failed: {escape(video.error_message or 'unknown error')}</p>"
        else:
            player = "<p>Video is processing. Refresh shortly.</p>"
        metadata = (
            f"<dl><dt>Duration</dt><dd>{video.duration_s or ''}</dd>"
            f"<dt>FPS</dt><dd>{video.fps or ''}</dd>"
            f"<dt>Dimensions</dt><dd>{video.width or ''}x{video.height or ''}</dd>"
            "<dt>Timestamp mapping</dt><dd>ts_s = frame_idx / fps</dd></dl>"
        )
    return f"""
    <!doctype html>
    <html>
      <head><title>{escape(game.name)} - BasketVision Coach</title></head>
      <body>
        <h1>{escape(game.name)}</h1>
        <p>Processing state: <strong>{escape(state)}</strong></p>
        {player}
        {metadata}
        <p><a href="/games/{game.id}/tracks">Review tracked players</a></p>
      </body>
    </html>
    """


def render_tracks_page(
    game: Game,
    video: Video | None,
    identity_service: IdentityReviewService,
) -> str:
    teams = identity_service.list_teams()
    players = identity_service.list_players()
    assignments = identity_service.list_track_identities()
    duration = video.duration_s if video is not None and video.duration_s else 20.0
    track_buttons = "".join(
        f'<button type="button" class="track" data-track-id="track-{track_id}">Track {track_id}</button>'
        for track_id in range(1, 6)
    )
    player_options = "<option value="">Unassigned player</option>" + "".join(
        f'<option value="{escape(player.id)}">{escape(player.name)} ({escape(player.team_id)})</option>'
        for player in players
    )
    team_options = "".join(
        f'<option value="{escape(team.id)}">{escape(team.name)}</option>' for team in teams
    )
    assignment_rows = "".join(
        "<tr>"
        f"<td>{escape(identity.track_id)}</td>"
        f"<td>{escape(identity.player_id or 'unassigned')}</td>"
        f"<td>{escape(identity.team_id or 'unknown')}</td>"
        f"<td>{escape(identity.start_s if identity.start_s is not None else 'full')}</td>"
        f"<td>{escape(identity.end_s if identity.end_s is not None else 'track')}</td>"
        f"<td>{escape(identity.source)}</td>"
        "</tr>"
        for identity in assignments
    ) or '<tr><td colspan="6">No reviewed identities yet.</td></tr>'
    return f"""
    <!doctype html>
    <html>
      <head>
        <title>{escape(game.name)} tracks - BasketVision Coach</title>
        <style>.track[aria-pressed="true"] {{ outline: 3px solid #0a7; }}</style>
      </head>
      <body>
        <h1>{escape(game.name)} tracked player review</h1>
        <p>Click a track, optionally bound the segment by seconds, then assign a roster player and team.</p>
        <section aria-label="Tracked players">
          {track_buttons}
        </section>
        <form id="identity-form" method="post" action="/api/track-identities">
          <h2>Assign selected track segment</h2>
          <label>Track ID <input name="track_id" id="track_id" value="track-1" required></label>
          <label>Start seconds <input name="start_s" id="start_s" type="number" step="0.01" min="0" max="{escape(duration)}"></label>
          <label>End seconds <input name="end_s" id="end_s" type="number" step="0.01" min="0" max="{escape(duration)}"></label>
          <label>Player <select name="player_id" id="player_id">{player_options}</select></label>
          <label>Team <select name="team_id" id="team_id" required>{team_options}</select></label>
          <label>Reviewer <input name="user" id="user" value="reviewer" required></label>
          <input name="confidence" id="confidence" type="hidden" value="1.0">
          <input name="source" id="source" type="hidden" value="manual">
          <button type="submit">Save assignment</button>
        </form>
        <table>
          <thead><tr><th>Track</th><th>Player</th><th>Team</th><th>Start</th><th>End</th><th>Source</th></tr></thead>
          <tbody>{assignment_rows}</tbody>
        </table>
        <script>
          const form = document.getElementById('identity-form');
          document.querySelectorAll('.track').forEach((button) => {{
            button.addEventListener('click', () => {{
              document.querySelectorAll('.track').forEach((b) => b.setAttribute('aria-pressed', 'false'));
              button.setAttribute('aria-pressed', 'true');
              document.getElementById('track_id').value = button.dataset.trackId;
            }});
          }});
          form.addEventListener('submit', async (event) => {{
            event.preventDefault();
            const data = Object.fromEntries(new FormData(form).entries());
            for (const key of ['start_s', 'end_s']) {{ if (data[key] === '') data[key] = null; else data[key] = Number(data[key]); }}
            if (data.player_id === '') data.player_id = null;
            data.confidence = Number(data.confidence);
            const response = await fetch('/api/track-identities', {{
              method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify(data)
            }});
            if (!response.ok) alert(await response.text()); else window.location.reload();
          }});
        </script>
      </body>
    </html>
    """


app = create_app()
