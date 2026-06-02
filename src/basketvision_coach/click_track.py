"""Click-to-track: follow a clicked object through a whole video (SAM 2).

A coach clicks an object on one frame and gives it a label; SAM 2's video
predictor propagates that object's mask across every frame. No training is
required — SAM 2 is a promptable foundation model. As with the YOLO analyzer the
heavy runtime is lazy-imported and dependency-injected, so the orchestration and
geometry helpers are unit-tested with a fake tracker, while the real model path
runs where ``sam2``/``torch`` are installed.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from basketvision_coach.analysis import (
    AnalysisResult,
    AnalyzerUnavailable,
    DetectedObject,
    FrameDetections,
    VideoMeta,
    resolve_device,
)
from basketvision_coach.cv_pipeline import (
    BoundingBox,
    DetectionCandidate,
    InMemoryVisionStore,
    JobProgress,
)

SAM_INSTALL_HINT = (
    "Click-to-track needs SAM 2. Install it with `pip install sam2 torch "
    "opencv-python-headless` (GPU/MPS recommended) and download a SAM 2 checkpoint."
)


@dataclass(frozen=True, slots=True)
class ClickPrompt:
    """A single foreground click selecting an object to track."""

    object_label: str
    frame_idx: int
    x: float
    y: float


class ClickTracker(Protocol):
    """Contract for propagating clicked objects across a video."""

    engine: str

    def probe(self, video_path: Path) -> VideoMeta: ...

    def track(
        self, video_path: Path, prompts: Sequence[ClickPrompt]
    ) -> Iterable[FrameDetections]: ...


def mask_bbox(rows: Iterable[Iterable[object]]) -> tuple[float, float, float, float] | None:
    """Return ``(x, y, width, height)`` enclosing truthy mask cells, or ``None``.

    Pure and array-agnostic so it is unit-testable without numpy; the real SAM 2
    path converts its mask once and passes the rows in.
    """

    min_x = min_y = None
    max_x = max_y = None
    for y, row in enumerate(rows):
        for x, value in enumerate(row):
            if not value:
                continue
            if min_x is None or x < min_x:
                min_x = x
            if max_x is None or x > max_x:
                max_x = x
            if min_y is None or y < min_y:
                min_y = y
            if max_y is None or y > max_y:
                max_y = y
    if min_x is None or min_y is None or max_x is None or max_y is None:
        return None
    return (float(min_x), float(min_y), float(max_x - min_x + 1), float(max_y - min_y + 1))


def _label_track_ids(prompts: Sequence[ClickPrompt]) -> dict[str, int]:
    """Assign a stable integer track id to each distinct object label."""

    ids: dict[str, int] = {}
    for prompt in prompts:
        if prompt.object_label not in ids:
            ids[prompt.object_label] = len(ids) + 1
    return ids


def run_click_tracking(
    store: InMemoryVisionStore,
    *,
    video_id: int,
    video_path: Path,
    prompts: Sequence[ClickPrompt],
    tracker: ClickTracker,
) -> AnalysisResult:
    """Propagate clicked objects across ``video_path`` and persist runs."""

    if not prompts:
        raise ValueError("at least one click prompt is required")

    meta = tracker.probe(video_path)
    detection_run = store.create_run(
        video_id=str(video_id),
        run_type="detection",
        model_versions_json={"detector": {"engine": tracker.engine}},
        params_json={"prompts": len(prompts), "source": video_path.name},
    )
    tracking_run = store.create_run(
        video_id=str(video_id),
        run_type="tracking",
        model_versions_json={"tracker": {"engine": tracker.engine}},
        params_json={"detection_run_id": detection_run.run_id},
    )

    frame_count = 0
    object_count = 0
    for frame in tracker.track(video_path, prompts):
        frame_count = max(frame_count, frame.frame_idx + 1)
        for obj in frame.objects:
            record = store.add_detection(
                detection_run.run_id,
                DetectionCandidate(
                    frame_idx=frame.frame_idx,
                    ts_s=frame.ts_s,
                    class_name=obj.class_name,
                    confidence=obj.confidence,
                    bbox=BoundingBox(obj.x, obj.y, obj.width, obj.height),
                ),
            )
            object_count += 1
            if obj.track_id is not None:
                store.add_track(tracking_run.run_id, record, obj.track_id)

    store.finish_run(detection_run.run_id, "succeeded")
    store.finish_run(tracking_run.run_id, "succeeded")
    last_frame = frame_count - 1 if frame_count else None
    for run_id, stage in (
        (detection_run.run_id, "detection:complete"),
        (tracking_run.run_id, "tracking:complete"),
    ):
        store.save_progress(
            JobProgress(
                run_id=run_id,
                stage=stage,
                frame_start=0 if frame_count else None,
                frame_end=last_frame,
                processing_fps=meta.fps,
                warnings=(),
                status="succeeded",
            )
        )

    return AnalysisResult(
        detection_run_id=detection_run.run_id,
        tracking_run_id=tracking_run.run_id,
        source_width=meta.width,
        source_height=meta.height,
        fps=meta.fps,
        frame_count=frame_count,
        object_count=object_count,
        engine=tracker.engine,
    )


@dataclass
class Sam2ClickTracker:
    """SAM 2 video predictor wrapper (real implementation, lazy-imported).

    This targets the ``sam2`` package's video predictor API. Exact calls can vary
    by SAM 2 release/checkpoint, so adapt as needed for your install; it is marked
    no-cover because the model cannot run in CI.
    """

    checkpoint: str = "sam2_hiera_small.pt"
    model_cfg: str = "sam2_hiera_s.yaml"
    device: str | None = None
    engine: str = "sam2"

    def _build_predictor(self) -> Any:  # pragma: no cover - needs sam2/torch
        device = resolve_device(self.device)
        # Several SAM 2 ops lack Metal kernels; let them fall back to CPU on MPS.
        if device == "mps":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as exc:
            raise AnalyzerUnavailable(SAM_INSTALL_HINT) from exc
        return build_sam2_video_predictor(self.model_cfg, self.checkpoint, device=device)

    def probe(self, video_path: Path) -> VideoMeta:  # pragma: no cover - needs opencv
        try:
            import cv2
        except ImportError as exc:
            raise AnalyzerUnavailable(SAM_INSTALL_HINT) from exc
        capture = cv2.VideoCapture(str(video_path))
        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
        finally:
            capture.release()
        return VideoMeta(width=width, height=height, fps=fps)

    def track(
        self, video_path: Path, prompts: Sequence[ClickPrompt]
    ) -> Iterable[FrameDetections]:  # pragma: no cover - needs models
        predictor = self._build_predictor()
        track_ids = _label_track_ids(prompts)
        labels_by_id = {v: k for k, v in track_ids.items()}
        fps = self.probe(video_path).fps

        state = predictor.init_state(video_path=str(video_path))
        for prompt in prompts:
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=prompt.frame_idx,
                obj_id=track_ids[prompt.object_label],
                points=[[prompt.x, prompt.y]],
                labels=[1],
            )

        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            objects: list[DetectedObject] = []
            for obj_id, logits in zip(obj_ids, mask_logits, strict=True):
                mask = (logits[0] > 0.0).tolist()
                box = mask_bbox(mask)
                if box is None:
                    continue
                label = labels_by_id.get(int(obj_id), str(obj_id))
                objects.append(
                    DetectedObject(
                        class_name="ball" if label.lower() == "ball" else "player",
                        confidence=1.0,
                        x=box[0],
                        y=box[1],
                        width=box[2],
                        height=box[3],
                        track_id=int(obj_id),
                    )
                )
            yield FrameDetections(
                frame_idx=frame_idx,
                ts_s=frame_idx / fps if fps else 0.0,
                objects=tuple(objects),
            )


def load_default_click_tracker() -> ClickTracker:
    """Return the SAM 2 tracker, or raise :class:`AnalyzerUnavailable`."""

    tracker = Sam2ClickTracker()
    tracker._build_predictor()
    return tracker
