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

These checks also run in CI (`.github/workflows/ci.yml`) in a clean environment
so missing runtime dependencies are caught before merge.

## Running the web app

```bash
# ffmpeg/ffprobe are required for real video transcoding
uv run uvicorn basketvision_coach.api:app --reload
```

Then open <http://localhost:8000/> (interactive API docs at `/docs`). uvicorn
binds `127.0.0.1:8000` by default.

### Configuration

| Env var | Default | Purpose |
|---|---|---|
| `DATA_ROOT` | `data` | Directory for the SQLite db and stored/transcoded video assets. |
| `DATABASE_URL` | `sqlite:///<DATA_ROOT>/basketvision.db` | Override the database (e.g. Postgres). |

### Changing host / port

`uvicorn` controls the bind address; there is no port baked into the code:

```bash
uv run uvicorn basketvision_coach.api:app --port 9000              # http://127.0.0.1:9000/
uv run uvicorn basketvision_coach.api:app --host 0.0.0.0 --port 9000   # reachable from other hosts
```

For platforms that inject a `PORT` env var (Render, Railway, Cloud Run, Heroku):

```bash
uv run uvicorn basketvision_coach.api:app --host 0.0.0.0 --port ${PORT:-8000}
```

### Production-style (multiple workers)

```bash
uv run uvicorn basketvision_coach.api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker

A `Dockerfile` (with ffmpeg) and `docker-compose.yml` (persistent `/data`
volume) are included:

```bash
docker build -t sensei .
docker run --rm -p 8000:8000 -v sensei-data:/data sensei

# or, with compose (host port overridable):
docker compose up --build
SENSEI_PORT=9000 docker compose up        # serve on http://localhost:9000/
```

The container reads `PORT` (default 8000) and stores everything under the
`/data` volume. The optional CV runtime (YOLO/SAM 2) is not baked into the
image to keep it light — add it in a derived image if you need those buttons.

### UI

The web UI is server-rendered with Jinja2 and progressively enhanced with
htmx (form posts) and Alpine.js (client state). Pages:

- **`/`** — list and create games.
- **`/games/{id}`** — upload an MP4, watch background processing update live,
  then play the 720p proxy (or the original directly if `ffmpeg` is not
  installed) with a CV overlay canvas (detection boxes, track trails, ball
  path) and per-layer toggles. Click **Run demo analysis** to fabricate
  synthetic detections and run them through the real detection + tracking
  pipeline so the overlays render over your video — handy for seeing the
  system work before a trained detector is wired in. You can also paste a
  detection/tracking run id from the CV worker to load a real run.
- **`/games/{id}/calibrate`** — capture a frame, click court landmarks to map
  pixels to court coordinates, and save a calibration.

### Real detection + tracking (optional CV runtime)

The **Run analysis (YOLO)** button runs stock YOLO detection + ByteTrack
tracking over the uploaded video and draws the resulting boxes/trails/ball path
— **no model training required** (pretrained COCO weights already detect
`person` and `sports ball`). The heavy runtime is not a declared dependency, so
install it where you have a GPU/MPS and network access:

```bash
pip install ultralytics opencv-python-headless
```

Without it, the button returns a clear "install the CV runtime" message and you
can still use **Run demo (synthetic)**. The analyzer is dependency-injected
(`create_app(analyzer=...)`), so it is covered in tests with a fake and swappable
for other engines.

**Click-to-track (SAM 2).** On the game page you can also type a label, enable
click mode, and click an object in the video to follow it through the whole clip
— SAM 2 propagates the selection across every frame, no training required. This
needs the SAM 2 runtime:

```bash
pip install sam2 torch opencv-python-headless   # plus a SAM 2 checkpoint
```

Like the YOLO analyzer it is lazy-imported and dependency-injected
(`create_app(click_tracker=...)`); without it the endpoint returns an actionable
503. Tracked objects feed the same overlay and identity-review tools, so you can
assign each tracked object to a roster player/team and correct any ID swaps.

### HTTP API

In addition to the pages, the app exposes JSON endpoints consumed by the UI
and CV worker: game/video CRUD and proxy streaming, court calibration
(`/api/landmarks`, `/api/videos/{id}/calibrations`, `/project`, `/overlay`),
and CV jobs (`/api/videos/{id}/detections`, `/tracking`,
`/api/runs/{run_id}/progress|detections|tracks`).

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
