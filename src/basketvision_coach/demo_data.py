"""Synthetic detection generator for the demo analysis.

The project ships no trained object detector (that needs model weights and a
heavy runtime). To let a coach *see* what the tracking and overlay system does
on a real uploaded video, this module fabricates plausible per-frame detections
— a handful of players moving along smooth paths plus a ball with a shooting arc
— in a fixed virtual coordinate space. Those detections are fed through the real
:class:`~basketvision_coach.cv_pipeline.VisionPipeline`, so tracking, trails, and
ball-path overlays are exercised end to end.
"""

from __future__ import annotations

from math import cos, pi, sin

from basketvision_coach.cv_pipeline import BoundingBox, DetectionCandidate

# Fixed virtual frame the demo detections are drawn in. The UI scales this onto
# whatever resolution the played video reports, so alignment does not depend on
# the real proxy/original dimensions.
DEMO_WIDTH = 1000.0
DEMO_HEIGHT = 1000.0

_PLAYER_BOX = (44.0, 96.0)
_BALL_BOX = (24.0, 24.0)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def generate_demo_frames(
    *, duration_s: float, fps: float, num_players: int = 5
) -> list[list[DetectionCandidate]]:
    """Return deterministic per-frame detections for ``duration_s`` at ``fps``."""

    if duration_s <= 0:
        duration_s = 8.0
    if fps <= 0:
        fps = 25.0
    frame_count = max(int(round(duration_s * fps)), 1)
    frames: list[list[DetectionCandidate]] = []

    for frame_idx in range(frame_count):
        ts_s = frame_idx / fps
        phase = frame_idx / frame_count  # 0 -> 1 across the clip
        candidates: list[DetectionCandidate] = []

        for player in range(num_players):
            offset = player / num_players
            # Smooth, continuous Lissajous-style motion so tracks stay stable.
            cx = 0.5 + 0.32 * sin(2 * pi * (phase + offset))
            cy = 0.35 + 0.25 * cos(2 * pi * (phase * 1.3 + offset)) + 0.15 * offset
            px = _clamp(cx * DEMO_WIDTH, 0, DEMO_WIDTH - _PLAYER_BOX[0])
            py = _clamp(cy * DEMO_HEIGHT, 0, DEMO_HEIGHT - _PLAYER_BOX[1])
            candidates.append(
                DetectionCandidate(
                    frame_idx=frame_idx,
                    ts_s=ts_s,
                    class_name="player",
                    confidence=0.82 + 0.1 * (player % 2),
                    bbox=BoundingBox(px, py, _PLAYER_BOX[0], _PLAYER_BOX[1]),
                )
            )

        # Ball travels left-to-right with a parabolic shooting arc.
        bx = (0.15 + 0.7 * phase) * DEMO_WIDTH
        by = (0.7 - 1.8 * phase * (1 - phase)) * DEMO_HEIGHT
        candidates.append(
            DetectionCandidate(
                frame_idx=frame_idx,
                ts_s=ts_s,
                class_name="ball",
                confidence=0.9,
                bbox=BoundingBox(
                    _clamp(bx, 0, DEMO_WIDTH - _BALL_BOX[0]),
                    _clamp(by, 0, DEMO_HEIGHT - _BALL_BOX[1]),
                    _BALL_BOX[0],
                    _BALL_BOX[1],
                ),
            )
        )
        frames.append(candidates)

    return frames
