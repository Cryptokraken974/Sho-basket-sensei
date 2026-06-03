from __future__ import annotations

from dataclasses import dataclass

from basketvision_coach.overlay_render import OverlayLayers, build_overlay_index


@dataclass(frozen=True)
class _Box:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class _Det:
    frame_idx: int
    class_name: str
    bbox: _Box


@dataclass(frozen=True)
class _Trk:
    frame_idx: int
    track_id: int
    bbox: _Box


def test_overlay_layers_defaults() -> None:
    layers = OverlayLayers()
    assert layers.detection_boxes and layers.track_trails and layers.ball_path


def test_build_overlay_index_groups_and_sorts() -> None:
    detections = [
        _Det(0, "player", _Box(10, 10, 20, 40)),
        _Det(0, "ball", _Box(50, 50, 10, 10)),
        _Det(1, "ball", _Box(60, 40, 10, 10)),
    ]
    tracks = [
        _Trk(1, 7, _Box(12, 11, 20, 40)),
        _Trk(0, 7, _Box(10, 10, 20, 40)),
    ]
    index = build_overlay_index(detections, tracks)

    # detections grouped by frame
    assert len(index.detections_by_frame[0]) == 2
    assert len(index.detections_by_frame[1]) == 1

    # ball path = ball centers, sorted by frame
    assert [f for (f, _x, _y) in index.ball_path] == [0, 1]
    assert index.ball_path[0][1] == 55.0  # 50 + 10/2

    # track 7 centers sorted by frame
    centers = index.track_points[7]
    assert [f for (f, _x, _y) in centers] == [0, 1]
