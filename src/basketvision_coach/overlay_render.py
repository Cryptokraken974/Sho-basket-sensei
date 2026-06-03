"""Burn detection/track overlays into a copy of the analyzed video for download.

Reads the same source the detections were computed on, draws the selected
overlay layers (boxes / track trails / ball path) per frame with OpenCV, then
re-encodes to browser-friendly H.264 with ffmpeg. Requires the optional CV
runtime (opencv) plus ffmpeg.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

TRAIL_WINDOW = 30  # frames of history drawn for track trails


class _HasBox(Protocol):
    frame_idx: int
    class_name: str
    bbox: Any


@dataclass(frozen=True, slots=True)
class OverlayLayers:
    detection_boxes: bool = True
    track_trails: bool = True
    ball_path: bool = True


@dataclass(frozen=True, slots=True)
class OverlayIndex:
    detections_by_frame: dict[int, list[Any]]
    ball_path: list[tuple[int, float, float]]
    track_points: dict[int, list[tuple[int, float, float]]]


def _center(bbox: Any) -> tuple[float, float]:
    return (bbox.x + bbox.width / 2.0, bbox.y + bbox.height / 2.0)


def build_overlay_index(detections: list[Any], tracks: list[Any]) -> OverlayIndex:
    """Group detections/tracks by frame and track for efficient per-frame drawing."""

    detections_by_frame: dict[int, list[Any]] = defaultdict(list)
    ball_path: list[tuple[int, float, float]] = []
    for det in detections:
        detections_by_frame[det.frame_idx].append(det)
        if det.class_name == "ball":
            cx, cy = _center(det.bbox)
            ball_path.append((det.frame_idx, cx, cy))
    ball_path.sort(key=lambda item: item[0])

    track_points: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for trk in tracks:
        cx, cy = _center(trk.bbox)
        track_points[trk.track_id].append((trk.frame_idx, cx, cy))
    for points in track_points.values():
        points.sort(key=lambda item: item[0])

    return OverlayIndex(
        detections_by_frame=dict(detections_by_frame),
        ball_path=ball_path,
        track_points=dict(track_points),
    )


def render_overlay_video(
    *,
    source: Path,
    detections: list[Any],
    tracks: list[Any],
    layers: OverlayLayers,
    out_path: Path,
) -> Path:  # pragma: no cover - needs opencv + ffmpeg
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("overlay export needs opencv (install the CV runtime)") from exc

    index = build_overlay_index(detections, tracks)
    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        raw_path = Path(tmp.name)
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    green, orange, yellow = (60, 207, 60), (60, 122, 255), (66, 176, 245)  # BGR
    frame_idx = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if layers.detection_boxes:
                for det in index.detections_by_frame.get(frame_idx, ()):
                    b = det.bbox
                    p1 = (int(b.x), int(b.y))
                    p2 = (int(b.x + b.width), int(b.y + b.height))
                    cv2.rectangle(frame, p1, p2, green, 2)
                    cv2.putText(
                        frame,
                        f"{det.class_name} {det.confidence:.2f}",
                        (int(b.x), max(0, int(b.y) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        green,
                        1,
                        cv2.LINE_AA,
                    )
            if layers.track_trails:
                for points in index.track_points.values():
                    recent = [
                        (x, y)
                        for (f, x, y) in points
                        if frame_idx - TRAIL_WINDOW <= f <= frame_idx
                    ]
                    for i in range(1, len(recent)):
                        cv2.line(
                            frame,
                            (int(recent[i - 1][0]), int(recent[i - 1][1])),
                            (int(recent[i][0]), int(recent[i][1])),
                            orange,
                            2,
                        )
            if layers.ball_path:
                seen = [(x, y) for (f, x, y) in index.ball_path if f <= frame_idx]
                for i in range(1, len(seen)):
                    cv2.line(
                        frame,
                        (int(seen[i - 1][0]), int(seen[i - 1][1])),
                        (int(seen[i][0]), int(seen[i][1])),
                        yellow,
                        2,
                    )
            writer.write(frame)
            frame_idx += 1
    finally:
        capture.release()
        writer.release()

    # Re-encode to H.264 for broad browser playback.
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_path),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    raw_path.unlink(missing_ok=True)
    if completed.returncode != 0:
        tail = "\n".join((completed.stderr or "").strip().splitlines()[-6:])
        raise RuntimeError(f"ffmpeg overlay encode failed: {tail or completed.returncode}")
    return out_path
