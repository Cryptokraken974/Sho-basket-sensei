from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from basketvision_coach.video_models import Game, Video, VideoState


@dataclass(frozen=True, slots=True)
class FfmpegMetadata:
    duration_s: float
    fps: float
    width: int
    height: int


TranscodeRunner = Callable[[Path, Path, Path | None], FfmpegMetadata]


def frame_idx_to_ts_s(frame_idx: int, fps: float) -> float:
    if frame_idx < 0:
        raise ValueError("frame_idx must be non-negative")
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    return frame_idx / fps


def ts_s_to_frame_idx(ts_s: float, fps: float) -> int:
    if ts_s < 0:
        raise ValueError("ts_s must be non-negative")
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    return round(ts_s * fps)


def parse_rational(value: str) -> float:
    if "/" not in value:
        return float(value)
    numerator, denominator = value.split("/", 1)
    denominator_float = float(denominator)
    if denominator_float == 0:
        raise ValueError("ffprobe returned an invalid zero-denominator frame rate")
    return float(numerator) / denominator_float


def _run_ffmpeg(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run an ffmpeg/ffprobe command, surfacing stderr on failure."""

    try:
        completed = subprocess.run(command, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{command[0]} not found; install ffmpeg to enable transcoding"
        ) from exc
    if completed.returncode != 0:
        tail = "\n".join((completed.stderr or "").strip().splitlines()[-6:])
        tool = command[0]
        raise RuntimeError(
            f"{tool} exited {completed.returncode}: {tail or 'no stderr output'}"
        )
    return completed


def probe_metadata(source: Path) -> FfmpegMetadata:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,duration",
        "-of",
        "json",
        str(source),
    ]
    completed = _run_ffmpeg(command)
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError("ffprobe found no video stream")
    stream = streams[0]
    duration_raw = stream.get("duration")
    if duration_raw in (None, "N/A"):
        duration_raw = "0"
    return FfmpegMetadata(
        duration_s=float(duration_raw),
        fps=parse_rational(str(stream["r_frame_rate"])),
        width=int(stream["width"]),
        height=int(stream["height"]),
    )


# Cap height at 720p without upscaling. The comma in min() is escaped so ffmpeg's
# filtergraph parser does not split on it, and both dimensions are forced even
# (libx264 rejects odd width/height): width via -2, height via trunc(ih/2)*2.
PROXY_SCALE_FILTER = r"scale=-2:min(720\,trunc(ih/2)*2)"


def _proxy_command(source: Path, proxy_path: Path) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vf",
        PROXY_SCALE_FILTER,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(proxy_path),
    ]


def _hls_command(proxy_path: Path, hls_dir: Path) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(proxy_path),
        "-codec",
        "copy",
        "-start_number",
        "0",
        "-hls_time",
        "4",
        "-hls_playlist_type",
        "vod",
        str(hls_dir / "index.m3u8"),
    ]


def run_ffmpeg_transcode(
    source: Path, proxy_path: Path, hls_dir: Path | None = None
) -> FfmpegMetadata:
    proxy_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(_proxy_command(source, proxy_path))

    if hls_dir is not None:
        hls_dir.mkdir(parents=True, exist_ok=True)
        _run_ffmpeg(_hls_command(proxy_path, hls_dir))

    return probe_metadata(source)



class VideoIngestService:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        data_root: Path = Path("data"),
        transcode_runner: TranscodeRunner = run_ffmpeg_transcode,
        create_hls: bool = True,
    ) -> None:
        self.session_factory = session_factory
        self.data_root = data_root
        self.video_root = data_root / "videos"
        self.transcode_runner = transcode_runner
        self.create_hls = create_hls

    @contextmanager
    def session_scope(self) -> Iterator[Session]:
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_game(self, name: str) -> Game:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("game name is required")
        with self.session_scope() as session_obj:
            session = session_obj
            game = Game(name=clean_name)
            session.add(game)
            session.flush()
            session.refresh(game)
            return game

    def list_games(self) -> list[Game]:
        with self.session_scope() as session_obj:
            session = session_obj
            return list(session.scalars(select(Game).order_by(Game.created_at.desc())).all())

    def get_game(self, game_id: int) -> Game | None:
        with self.session_scope() as session_obj:
            session = session_obj
            return session.get(Game, game_id)

    def get_video(self, video_id: int) -> Video | None:
        with self.session_scope() as session_obj:
            session = session_obj
            return session.get(Video, video_id)

    def latest_video_for_game(self, game_id: int) -> Video | None:
        with self.session_scope() as session_obj:
            session = session_obj
            return session.scalars(
                select(Video).where(Video.game_id == game_id).order_by(Video.created_at.desc())
            ).first()

    def ingest_game_video(
        self,
        game_id: int,
        filename: str,
        content: bytes | BinaryIO,
        *,
        run_transcode: bool = False,
    ) -> Video:
        suffix = Path(filename).suffix.lower()
        if suffix != ".mp4":
            raise ValueError("Only MP4 uploads are supported")

        with self.session_scope() as session_obj:
            session = session_obj
            game = session.get(Game, game_id)
            if game is None:
                raise LookupError(f"Game {game_id} does not exist")
            video = Video(game_id=game_id, original_path="")
            session.add(video)
            session.flush()
            destination_dir = self.video_root / str(video.id)
            destination_dir.mkdir(parents=True, exist_ok=True)
            original_path = destination_dir / "original.mp4"
            self._write_original(content, original_path)
            video.original_path = str(original_path)
            video.state = VideoState.UPLOADED
            session.flush()
            session.refresh(video)
            video_id = video.id

        if run_transcode:
            self.transcode_video(video_id)

        found = self.get_video(video_id)
        if found is None:
            raise RuntimeError("Video disappeared after ingest")
        return found

    def transcode_video(self, video_id: int) -> None:
        with self.session_scope() as session_obj:
            session = session_obj
            video = session.get(Video, video_id)
            if video is None:
                raise LookupError(f"Video {video_id} does not exist")
            video.state = VideoState.PROCESSING
            video.error_message = None
            original_path = Path(video.original_path)
            proxy_path = original_path.parent / "proxy_720p.mp4"
            hls_dir = original_path.parent / "hls" if self.create_hls else None

        try:
            metadata = self.transcode_runner(original_path, proxy_path, hls_dir)
        except Exception as exc:
            with self.session_scope() as session_obj:
                session = session_obj
                failed_video = session.get(Video, video_id)
                if failed_video is not None:
                    failed_video.state = VideoState.FAILED
                    failed_video.error_message = str(exc)
                    failed_video.proxy_path = None
            return

        with self.session_scope() as session_obj:
            session = session_obj
            ready_video = session.get(Video, video_id)
            if ready_video is None:
                raise LookupError(f"Video {video_id} does not exist")
            ready_video.proxy_path = str(proxy_path)
            ready_video.hls_path = str(hls_dir / "index.m3u8") if hls_dir is not None else None
            ready_video.duration_s = metadata.duration_s
            ready_video.fps = metadata.fps
            ready_video.width = metadata.width
            ready_video.height = metadata.height
            ready_video.state = VideoState.READY
            ready_video.error_message = None

    @staticmethod
    def _write_original(content: bytes | BinaryIO, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            destination.write_bytes(content)
            return
        with destination.open("wb") as output:
            shutil.copyfileobj(content, output)
