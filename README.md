# BasketVision Coach

BasketVision Coach is a Python scaffold for turning basketball shot measurements from video or pose-estimation pipelines into concise coaching feedback.

The initial package focuses on a small, typed domain model that can be expanded as the vision pipeline matures:

- capture normalized shot observations
- score release angle, elbow alignment, follow-through, and balance
- run versioned CV detection jobs for player, ball, hoop, backboard, and referee objects
- run deterministic player tracking jobs from persisted detections
- calibrate video pixels to court coordinates using coach-labeled court landmarks
- expose job progress and overlay toggles for API/UI integration
- expose a lightweight CLI for smoke testing

## Quick start

```bash
uv sync --extra dev
uv run pytest
uv run mypy src tests
uv run ruff check .
```

## CV processing pipeline

`basketvision_coach.cv_pipeline` contains dependency-light contracts for the planned video worker:

- detection runs persist `processing_run` metadata with model name/version, optional weights hash, confidence threshold, device, calibration ID, status, and timestamps
- detection records store frame index, timestamp, class, confidence, bounding box, and optional court coordinates without deleting earlier runs
- tracking runs consume a detection run and write per-frame player tracks with stable `track_id` values for that tracking run
- progress snapshots include stage, frame bounds, processing FPS, warnings, and status for API responses
- `OverlayOptions` lets the UI independently show or hide detection boxes, track trails, and ball path overlays
- `WorkerRuntime` validates `DEVICE=cpu` or `DEVICE=mps` and carries native-worker connections to the Dockerized API, Redis, and shared storage

## Court calibration and projection

`basketvision_coach.court_calibration` models the calibration API/UI contract:

- the calibration page starts from a `FrameCapture` and lets a coach place labeled landmarks on the captured frame
- supported landmark labels include baseline/sideline intersections, free-throw line intersections, center-circle references, and three-point arc references
- `CourtCalibrationService.create_calibration` validates at least four unique point pairs, computes a homography, and appends a `court_calibrations`-style record with `image_points_json`, `court_points_json`, `homography_json`, `valid_from_s`, `valid_to_s`, and `created_by`
- `project_point` uses the active calibration for a video timestamp to return `court_x` and `court_y`
- `overlay_projection_payload` returns a court map plus projected test points for the video overlay
- replacing calibration means appending another validity-ranged record; prior records remain queryable

## CLI

```bash
uv run basketvision-coach --release-angle 49 --elbow-alignment 0.82 --follow-through 0.9 --balance 0.76
```

## Project layout

```text
src/basketvision_coach/   package code
tests/                    focused tests
```
