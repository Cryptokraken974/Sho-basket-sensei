"""Persist YOLO analysis results so they survive restarts and can be reloaded
(or hand-edited) without re-running the detector.

A snapshot is a JSON file per video under ``<DATA_ROOT>/analysis/<id>.json`` with
the detections, tracks, and source metadata. It can be restored back into an
``InMemoryVisionStore`` to drive the overlay and annotated-video export.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from basketvision_coach.cv_pipeline import (
    BoundingBox,
    DetectionCandidate,
    DetectionRecord,
    InMemoryVisionStore,
)


def analysis_path(data_root: Path, video_id: int) -> Path:
    return data_root / "analysis" / f"{video_id}.json"


def analysis_available(data_root: Path, video_id: int) -> bool:
    return analysis_path(data_root, video_id).is_file()


def _bbox_json(bbox: BoundingBox) -> dict[str, float]:
    return {"x": bbox.x, "y": bbox.y, "width": bbox.width, "height": bbox.height}


def snapshot(
    store: InMemoryVisionStore,
    *,
    video_id: int,
    detection_run_id: str,
    tracking_run_id: str,
    source_width: float,
    source_height: float,
    fps: float,
    engine: str,
) -> dict[str, Any]:
    """Build a serializable snapshot of a run's detections and tracks."""

    detection_records = store.list_detections(detection_run_id)
    frame_count = max((d.frame_idx for d in detection_records), default=-1) + 1
    detections = [
        {
            "frame_idx": d.frame_idx,
            "ts_s": d.ts_s,
            "class_name": d.class_name,
            "confidence": d.confidence,
            "bbox": _bbox_json(d.bbox),
        }
        for d in detection_records
    ]
    tracks = [
        {
            "frame_idx": t.frame_idx,
            "ts_s": t.ts_s,
            "class_name": t.class_name,
            "confidence": t.confidence,
            "track_id": t.track_id,
            "bbox": _bbox_json(t.bbox),
        }
        for t in store.list_tracks(tracking_run_id)
    ]
    return {
        "video_id": video_id,
        "engine": engine,
        "source_width": source_width,
        "source_height": source_height,
        "fps": fps,
        "frame_count": frame_count,
        "object_count": len(detections),
        "detections": detections,
        "tracks": tracks,
    }


def save_analysis(data_root: Path, video_id: int, payload: dict[str, Any]) -> Path:
    path = analysis_path(data_root, video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_analysis(data_root: Path, video_id: int) -> dict[str, Any] | None:
    path = analysis_path(data_root, video_id)
    if not path.is_file():
        return None
    loaded: dict[str, Any] = json.loads(path.read_text())
    return loaded


def restore_into_store(store: InMemoryVisionStore, payload: dict[str, Any]) -> tuple[str, str]:
    """Recreate detection + tracking runs from a snapshot; return their run ids."""

    video_id = str(payload.get("video_id", ""))
    detection_run = store.create_run(
        video_id=video_id,
        run_type="detection",
        model_versions_json={"detector": {"engine": payload.get("engine", "restored")}},
        params_json={"source": "restored"},
    )
    tracking_run = store.create_run(
        video_id=video_id,
        run_type="tracking",
        model_versions_json={"tracker": {"engine": "restored"}},
        params_json={"detection_run_id": detection_run.run_id},
    )
    for d in payload.get("detections", []):
        store.add_detection(
            detection_run.run_id,
            DetectionCandidate(
                frame_idx=d["frame_idx"],
                ts_s=d["ts_s"],
                class_name=d["class_name"],
                confidence=d["confidence"],
                bbox=BoundingBox(**d["bbox"]),
            ),
        )
    for t in payload.get("tracks", []):
        source = DetectionRecord(
            detection_id="restored",
            run_id=detection_run.run_id,
            frame_idx=t["frame_idx"],
            ts_s=t["ts_s"],
            class_name=t["class_name"],
            confidence=t["confidence"],
            bbox=BoundingBox(**t["bbox"]),
        )
        store.add_track(tracking_run.run_id, source, t["track_id"])
    store.finish_run(detection_run.run_id, "succeeded")
    store.finish_run(tracking_run.run_id, "succeeded")
    return detection_run.run_id, tracking_run.run_id
