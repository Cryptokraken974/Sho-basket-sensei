from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from basketvision_coach.analysis import (
    AnalyzerUnavailable,
    VideoAnalyzer,
    load_default_analyzer,
    run_analysis,
)
from basketvision_coach.court_calibration import (
    CalibrationPointPair,
    CourtCalibrationService,
    CourtLandmark,
    ImagePoint,
    ProjectablePoint,
    active_supported_landmark_labels,
)
from basketvision_coach.cv_pipeline import (
    BoundingBox,
    CourtPoint,
    DetectionCandidate,
    DetectionJobSpec,
    InMemoryVisionStore,
    ModelSpec,
    TrackingJobSpec,
    VisionPipeline,
)
from basketvision_coach.db import build_session_factory
from basketvision_coach.demo_data import DEMO_HEIGHT, DEMO_WIDTH, generate_demo_frames
from basketvision_coach.identity import (
    IdentityReviewService,
    RosterPlayer,
    RosterTeam,
    TrackIdentity,
)
from basketvision_coach.schemas import (
    AnalysisRead,
    CalibrationCreate,
    CalibrationRead,
    DemoAnalysisRead,
    DetectionRecordRead,
    DetectionRunCreate,
    GameCreate,
    GameRead,
    JobProgressRead,
    OverlayRequest,
    PlayerCreate,
    PlayerRead,
    ProjectedRead,
    ProjectRequest,
    RunRead,
    TeamCreate,
    TeamRead,
    TrackIdentityAssign,
    TrackIdentityRead,
    TrackingRunCreate,
    TrackRecordRead,
    VideoRead,
)
from basketvision_coach.video import VideoIngestService
from basketvision_coach.video_models import Video, VideoState

_PACKAGE_DIR = Path(__file__).resolve().parent
_TEMPLATES = Jinja2Templates(directory=str(_PACKAGE_DIR / "templates"))
_STATIC_DIR = _PACKAGE_DIR / "static"


def default_service() -> VideoIngestService:
    return VideoIngestService(session_factory=build_session_factory())


def video_to_read(video: Video) -> VideoRead:
    proxy_url = f"/api/videos/{video.id}/proxy_720p.mp4" if video.proxy_path else None
    original_url = f"/api/videos/{video.id}/original.mp4" if video.original_path else None
    return VideoRead(
        id=video.id,
        game_id=video.game_id,
        state=video.state.value,
        original_path=video.original_path,
        proxy_url=proxy_url,
        original_url=original_url,
        proxy_path=video.proxy_path,
        hls_path=video.hls_path,
        fps=video.fps,
        width=video.width,
        height=video.height,
        duration_s=video.duration_s,
        error_message=video.error_message,
        ts_mapping="ts_s = frame_idx / fps",
    )


def _calibration_to_read(record: object) -> CalibrationRead:
    return CalibrationRead.model_validate(record, from_attributes=True)


def _progress_to_read(progress: object) -> JobProgressRead:
    return JobProgressRead.model_validate(progress, from_attributes=True)


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
    calibration_service: CourtCalibrationService | None = None,
    vision_store: InMemoryVisionStore | None = None,
    identity_service: IdentityReviewService | None = None,
    analyzer: VideoAnalyzer | None = None,
) -> FastAPI:
    if service is not None and video_service is not None:
        raise ValueError("pass either service or video_service, not both")
    video_service = service or video_service or default_service()
    calibration = calibration_service or CourtCalibrationService()
    store = vision_store or InMemoryVisionStore()
    identity = identity_service or IdentityReviewService(
        session_factory=video_service.session_factory
    )
    app = FastAPI(title="BasketVision Coach")
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # --- Games ----------------------------------------------------------------

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

    # --- Video ingest ---------------------------------------------------------

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

    @app.get("/api/videos/{video_id}", response_model=VideoRead)
    def get_video(video_id: int) -> VideoRead:
        video = video_service.get_video(video_id)
        if video is None:
            raise HTTPException(status_code=404, detail="Video not found")
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

    @app.get("/api/videos/{video_id}/original.mp4")
    def get_original(video_id: int) -> FileResponse:
        video = video_service.get_video(video_id)
        if video is None or not video.original_path:
            raise HTTPException(status_code=404, detail="Video not found")
        original = Path(video.original_path)
        if not original.exists():
            raise HTTPException(status_code=404, detail="Original file is missing")
        return FileResponse(original, media_type="video/mp4", filename="original.mp4")

    # --- Court calibration ----------------------------------------------------

    @app.get("/api/landmarks", response_model=list[str])
    def list_landmarks() -> list[str]:
        return list(active_supported_landmark_labels())

    @app.post(
        "/api/videos/{video_id}/calibrations",
        response_model=CalibrationRead,
        status_code=201,
    )
    def create_calibration(video_id: int, payload: CalibrationCreate) -> CalibrationRead:
        supported = set(active_supported_landmark_labels())
        pairs: list[CalibrationPointPair] = []
        for pair in payload.point_pairs:
            if pair.landmark not in supported:
                raise HTTPException(
                    status_code=400, detail=f"unsupported landmark: {pair.landmark}"
                )
            pairs.append(
                CalibrationPointPair(
                    landmark=CourtLandmark(pair.landmark),
                    image=ImagePoint(pair.image_x, pair.image_y),
                    court_x=pair.court_x,
                    court_y=pair.court_y,
                )
            )
        try:
            record = calibration.create_calibration(
                video_id=str(video_id),
                point_pairs=pairs,
                valid_from_s=payload.valid_from_s,
                valid_to_s=payload.valid_to_s,
                created_by=payload.created_by,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _calibration_to_read(record)

    @app.get("/api/videos/{video_id}/calibrations", response_model=list[CalibrationRead])
    def list_calibrations(video_id: int) -> list[CalibrationRead]:
        return [_calibration_to_read(rec) for rec in calibration.list_calibrations(str(video_id))]

    @app.post("/api/videos/{video_id}/project", response_model=ProjectedRead)
    def project_point(video_id: int, payload: ProjectRequest) -> ProjectedRead:
        try:
            projected = calibration.project_point(
                video_id=str(video_id),
                image_x=payload.image_x,
                image_y=payload.image_y,
                ts_s=payload.ts_s,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return ProjectedRead(
            calibration_id=projected.calibration_id,
            court_x=projected.court_x,
            court_y=projected.court_y,
        )

    @app.post("/api/videos/{video_id}/overlay")
    def overlay_payload(video_id: int, payload: OverlayRequest) -> dict[str, object]:
        test_points = tuple(
            ProjectablePoint(label=p.label, image_x=p.image_x, image_y=p.image_y)
            for p in payload.test_points
        )
        try:
            return calibration.overlay_projection_payload(
                video_id=str(video_id),
                ts_s=payload.ts_s,
                test_points=test_points,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    # --- CV pipeline (detection / tracking jobs) ------------------------------

    @app.post("/api/videos/{video_id}/detections", response_model=RunRead, status_code=201)
    def run_detection(video_id: int, payload: DetectionRunCreate) -> RunRead:
        if payload.device not in {"cpu", "mps"}:
            raise HTTPException(status_code=400, detail="device must be 'cpu' or 'mps'")
        frames: list[list[DetectionCandidate]] = []
        for frame in payload.frames:
            candidates: list[DetectionCandidate] = []
            for c in frame:
                try:
                    bbox = BoundingBox(c.bbox.x, c.bbox.y, c.bbox.width, c.bbox.height)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                court = (
                    CourtPoint(c.court_coordinates.x, c.court_coordinates.y)
                    if c.court_coordinates is not None
                    else None
                )
                candidates.append(
                    DetectionCandidate(
                        frame_idx=c.frame_idx,
                        ts_s=c.ts_s,
                        class_name=c.class_name,
                        confidence=c.confidence,
                        bbox=bbox,
                        court_coordinates=court,
                    )
                )
            frames.append(candidates)
        spec = DetectionJobSpec(
            video_id=str(video_id),
            model=ModelSpec(
                name=payload.model.name,
                version=payload.model.version,
                weights_hash=payload.model.weights_hash,
            ),
            confidence_threshold=payload.confidence_threshold,
            frame_detections=frames,
            device=payload.device,  # type: ignore[arg-type]
            court_calibration_id=payload.court_calibration_id,
        )
        # Only project detections into court space when the request opts in with a
        # calibration; otherwise the projector would raise for videos without one.
        projector = calibration if payload.court_calibration_id else None
        pipeline = VisionPipeline(store, calibration_service=projector)
        try:
            result = pipeline.run_detection(spec)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return RunRead(run_id=result.run_id, progress=_progress_to_read(result.progress))

    @app.post("/api/videos/{video_id}/tracking", response_model=RunRead, status_code=201)
    def run_tracking(video_id: int, payload: TrackingRunCreate) -> RunRead:
        spec = TrackingJobSpec(
            video_id=str(video_id),
            detection_run_id=payload.detection_run_id,
            tracker_name=payload.tracker_name,
            tracker_config=payload.tracker_config,
            court_calibration_id=payload.court_calibration_id,
        )
        pipeline = VisionPipeline(store, calibration_service=calibration)
        try:
            result = pipeline.run_tracking(spec)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="detection run not found") from exc
        return RunRead(run_id=result.run_id, progress=_progress_to_read(result.progress))

    @app.post(
        "/api/videos/{video_id}/demo-analysis",
        response_model=DemoAnalysisRead,
        status_code=201,
    )
    def run_demo_analysis(
        video_id: int, duration_s: float | None = None, fps: float | None = None
    ) -> DemoAnalysisRead:
        """Fabricate detections and run the real detection+tracking pipeline.

        This does not look at video pixels (the project has no trained detector);
        it demonstrates the tracking and overlay system end to end on synthetic data.
        """
        video = video_service.get_video(video_id)
        if video is None:
            raise HTTPException(status_code=404, detail="Video not found")
        clip_duration = duration_s or video.duration_s or 8.0
        clip_fps = fps or video.fps or 25.0
        frames = generate_demo_frames(duration_s=clip_duration, fps=clip_fps)

        pipeline = VisionPipeline(store)
        detection = pipeline.run_detection(
            DetectionJobSpec(
                video_id=str(video_id),
                model=ModelSpec(name="demo-synthetic", version="1.0"),
                confidence_threshold=0.25,
                frame_detections=frames,
            )
        )
        tracking = pipeline.run_tracking(
            TrackingJobSpec(video_id=str(video_id), detection_run_id=detection.run_id)
        )
        return DemoAnalysisRead(
            detection_run_id=detection.run_id,
            tracking_run_id=tracking.run_id,
            source_width=DEMO_WIDTH,
            source_height=DEMO_HEIGHT,
            fps=clip_fps,
            frame_count=len(frames),
        )

    @app.post(
        "/api/videos/{video_id}/analysis",
        response_model=AnalysisRead,
        status_code=201,
    )
    def run_real_analysis(video_id: int) -> AnalysisRead:
        """Run YOLO detection + ByteTrack tracking over the stored video file."""
        video = video_service.get_video(video_id)
        if video is None:
            raise HTTPException(status_code=404, detail="Video not found")
        # Analyze the same file the UI plays so overlay coordinates align.
        source = video.proxy_path if video.proxy_path else video.original_path
        if not source or not Path(source).exists():
            raise HTTPException(status_code=409, detail="No playable video file to analyze")
        try:
            chosen = analyzer or load_default_analyzer()
        except AnalyzerUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        result = run_analysis(
            store, video_id=video_id, video_path=Path(source), analyzer=chosen
        )
        return AnalysisRead(
            detection_run_id=result.detection_run_id,
            tracking_run_id=result.tracking_run_id,
            source_width=result.source_width,
            source_height=result.source_height,
            fps=result.fps,
            frame_count=result.frame_count,
            object_count=result.object_count,
            engine=result.engine,
        )

    @app.get("/api/runs/{run_id}/progress", response_model=JobProgressRead)
    def get_run_progress(run_id: str) -> JobProgressRead:
        try:
            return _progress_to_read(store.get_progress(run_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc

    @app.get("/api/runs/{run_id}/detections", response_model=list[DetectionRecordRead])
    def list_run_detections(run_id: str) -> list[DetectionRecordRead]:
        return [
            DetectionRecordRead.model_validate(record, from_attributes=True)
            for record in store.list_detections(run_id)
        ]

    @app.get("/api/runs/{run_id}/tracks", response_model=list[TrackRecordRead])
    def list_run_tracks(run_id: str) -> list[TrackRecordRead]:
        return [
            TrackRecordRead.model_validate(record, from_attributes=True)
            for record in store.list_tracks(run_id)
        ]

    # --- Identity review (roster / track identities) --------------------------

    @app.post("/api/roster/teams", response_model=TeamRead, status_code=201)
    def create_team(payload: TeamCreate) -> TeamRead:
        try:
            team = identity.create_team(
                payload.name, team_id=payload.id, jersey_color=payload.jersey_color
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return team_to_read(team)

    @app.get("/api/roster/teams", response_model=list[TeamRead])
    def list_teams() -> list[TeamRead]:
        return [team_to_read(team) for team in identity.list_teams()]

    @app.post("/api/roster/players", response_model=PlayerRead, status_code=201)
    def create_player(payload: PlayerCreate) -> PlayerRead:
        try:
            player = identity.create_player(
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
        return [player_to_read(p) for p in identity.list_players(team_id=team_id)]

    @app.post("/api/track-identities", response_model=TrackIdentityRead, status_code=201)
    def assign_track_identity(payload: TrackIdentityAssign) -> TrackIdentityRead:
        try:
            assigned = identity.assign_track_segment(
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
        return identity_to_read(assigned)

    @app.get("/api/track-identities", response_model=list[TrackIdentityRead])
    def list_track_identities(track_id: str | None = None) -> list[TrackIdentityRead]:
        return [
            identity_to_read(item)
            for item in identity.list_track_identities(track_id=track_id)
        ]

    # --- HTML pages (Jinja2 + htmx + Alpine) ----------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        games = video_service.list_games()
        return _TEMPLATES.TemplateResponse(request, "index.html", {"games": games})

    @app.get("/games/{game_id}/tracks", response_class=HTMLResponse)
    def tracks_page(request: Request, game_id: int) -> HTMLResponse:
        game = video_service.get_game(game_id)
        if game is None:
            raise HTTPException(status_code=404, detail="Game not found")
        video = video_service.latest_video_for_game(game_id)
        duration = video.duration_s if video and video.duration_s else 20.0
        return _TEMPLATES.TemplateResponse(
            request,
            "tracks.html",
            {
                "game": game,
                "duration": duration,
                "teams": identity.list_teams(),
                "players": identity.list_players(),
                "assignments": identity.list_track_identities(),
                "track_ids": list(range(1, 6)),
            },
        )

    @app.post("/games", response_class=HTMLResponse)
    def create_game_form(request: Request, name: Annotated[str, Form()]) -> HTMLResponse:
        try:
            video_service.create_game(name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        games = video_service.list_games()
        # htmx swaps in just the refreshed list; a full-page form post still works too.
        if request.headers.get("HX-Request"):
            return _TEMPLATES.TemplateResponse(request, "_game_list.html", {"games": games})
        return _TEMPLATES.TemplateResponse(request, "index.html", {"games": games})

    @app.get("/games/{game_id}", response_class=HTMLResponse)
    def game_page(request: Request, game_id: int) -> HTMLResponse:
        game = video_service.get_game(game_id)
        if game is None:
            raise HTTPException(status_code=404, detail="Game not found")
        video = video_service.latest_video_for_game(game_id)
        return _TEMPLATES.TemplateResponse(
            request,
            "game.html",
            {"game": game, "video": video_to_read(video) if video else None},
        )

    @app.get("/games/{game_id}/calibrate", response_class=HTMLResponse)
    def calibration_page(request: Request, game_id: int) -> HTMLResponse:
        game = video_service.get_game(game_id)
        if game is None:
            raise HTTPException(status_code=404, detail="Game not found")
        video = video_service.latest_video_for_game(game_id)
        if video is None or video.state is not VideoState.READY:
            raise HTTPException(status_code=409, detail="A ready video is required to calibrate")
        return _TEMPLATES.TemplateResponse(
            request,
            "calibration.html",
            {"game": game, "video": video_to_read(video)},
        )

    return app


app = create_app()
