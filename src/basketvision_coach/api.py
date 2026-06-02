from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, ConfigDict

from basketvision_coach.db import build_session_factory
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


def default_service() -> VideoIngestService:
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


def create_app(service: VideoIngestService | None = None) -> FastAPI:
    video_service = service or default_service()
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
        items = "".join(
            f'<li><a href="/games/{game.id}">{escape(game.name)}</a></li>' for game in games
        )
        return HTMLResponse(f"<h1>BasketVision Coach</h1><ul>{items}</ul>")

    return app


def escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
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
      </body>
    </html>
    """


app = create_app()
