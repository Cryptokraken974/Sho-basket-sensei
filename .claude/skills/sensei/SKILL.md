---
name: sensei
description: >
  Deploy, run, configure, manage, and troubleshoot the BasketVision Coach
  (Sho-basket-sensei) FastAPI app. Use when asked to start/serve/run/build the
  app, install or enable the CV runtime (YOLO/SAM 2) on a Mac,
  change the port/host, set env vars (DATA_ROOT, DATABASE_URL, DEVICE,
  YOLO_MODEL, SAM2_CHECKPOINT, SAM2_CONFIG, SAMPLES_DIR), run the
  test/lint/type gates, run the CV preflight, reset data, or debug startup,
  ffmpeg, GPU/MPS, transcode, or CV-runtime errors.
---

# BasketVision Coach — ops skill

FastAPI app. ASGI entrypoint: `basketvision_coach.api:app`. Package manager: `uv`.
Default URL: **http://localhost:8000/** (API docs at `/docs`).

## Prerequisites

```bash
brew install uv ffmpeg     # uv runs the app; ffmpeg transcodes 720p proxies
```

`ffmpeg` is optional — without it, playback falls back to the uploaded original,
and the demo/review/report/calibration/roster features still work.

## Build / install (core app)

```bash
git pull origin main
uv sync --extra dev        # installs the package + dev deps from uv.lock
```

## Run

```bash
# development (auto-reload)
uv run uvicorn basketvision_coach.api:app --reload

# production-style
uv run uvicorn basketvision_coach.api:app --host 0.0.0.0 --port 8000 --workers 4

# change the port
uv run uvicorn basketvision_coach.api:app --port 9000
```

URL: http://localhost:8000/ (or chosen port). Bind `--host 0.0.0.0` for LAN access.

## CV runtime (YOLO + SAM 2) — optional, native only

Heavy deps are NOT in `uv.lock` (kept light), so install them natively into the
project venv to enable the **Run analysis (YOLO)** and
**Click-to-track (SAM 2)** buttons.

> ⚠️ **`uv sync` DELETES the CV libs.** `uv sync` is exact — it removes anything
> not in `uv.lock` (which excludes ultralytics/torch/sam2 on purpose). So
> `git pull && uv sync --extra dev` strips the CV runtime. **Reinstall the CV
> deps after any `uv sync`,** and run the CV server with the venv python directly
> (below) so nothing re-syncs underneath it. If "Run analysis" reports the CV
> runtime is missing despite installing it, a `uv sync` removed it — reinstall.

GOTCHA: plain `uv pip install` can target a different venv when `VIRTUAL_ENV` is
set in the shell. Always target the project venv explicitly:

```bash
uv pip install --python .venv/bin/python ultralytics opencv-python-headless
uv pip install --python .venv/bin/python torch
uv pip install --python .venv/bin/python "git+https://github.com/facebookresearch/sam2.git"
```

Download a SAM 2 checkpoint (small = good balance on Apple Silicon):

```bash
mkdir -p models
curl -L -o models/sam2.1_hiera_small.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

YOLO weights (`yolov8n.pt`) download automatically on first analysis.
Model weights are git-ignored (`models/`, `*.pt`) — never commit them
(the SAM checkpoint is ~184MB and GitHub rejects files over 100MB).

Verified working stack (Apple Silicon, M-series): torch 2.12, ultralytics 8.4.60,
opencv 4.13, sam-2 1.0, with checkpoint `models/sam2.1_hiera_small.pt` and config
`configs/sam2.1/sam2.1_hiera_s.yaml`.

### Preflight (always run this before launching with CV)

```bash
DEVICE=mps SAM2_CHECKPOINT=models/sam2.1_hiera_small.pt \
  .venv/bin/python -m basketvision_coach.checkcv
```

Aim for: device `mps`, torch/ultralytics/opencv/sam2 ✓, checkpoint found, and
both features `READY`. Each ✗ prints the exact `uv pip install` to run. (Use the
venv python, not `uv run`, so the check can't trigger a re-sync.)

### Run with CV enabled (Apple Silicon / MPS)

Use the venv python directly (NOT `uv run`, which re-syncs and can strip the CV
libs):

```bash
DEVICE=mps \
SAM2_CHECKPOINT=models/sam2.1_hiera_small.pt \
SAM2_CONFIG=configs/sam2.1/sam2.1_hiera_s.yaml \
YOLO_MODEL=yolov8n.pt \
.venv/bin/python -m uvicorn basketvision_coach.api:app --host 0.0.0.0 --port 8000
```

`PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically on MPS. SAM 2 holds all
frames in memory — test on SHORT clips first. CV models load lazily on first
request (not at startup), so config/checkpoint errors surface when a button is
first clicked, not when the server boots.

## Configuration (env vars)

| Var | Default | Purpose |
|---|---|---|
| `DATA_ROOT` | `data` | Dir for SQLite db + stored/transcoded assets. |
| `DATABASE_URL` | `sqlite:///<DATA_ROOT>/basketvision.db` | Override db (e.g. Postgres). |
| `DEVICE` | auto (`cuda`→`mps`→`cpu`) | torch device for YOLO/SAM 2. |
| `YOLO_MODEL` | `yolov8n.pt` | YOLO weights for the analysis endpoint. |
| `SAM2_CHECKPOINT` | `sam2_hiera_small.pt` | Path to the SAM 2 checkpoint. |
| `SAM2_CONFIG` | `sam2_hiera_s.yaml` | SAM 2 config matching the checkpoint. |
| `SAMPLES_DIR` | `samples` | Dir of bundled `.mp4` clips offered in the UI. |

## Quality gates (keep green before committing)

```bash
uv run ruff check .
uv run mypy src tests
uv run pytest
```

CI (`.github/workflows/ci.yml`) runs all three in a clean env on every PR.

## Data & persistence

- Persisted: games, videos, roster/teams/players, track identities (main db),
  manual review events (`<DATA_ROOT>/review.db`).
- In-memory (reset on restart): court calibrations and detection/tracking runs.
- Reset local state: stop the server and `rm -rf data/`.

## Routes / features

- `/` — games list + create; per-game tabs below.
- `/games/{id}` — upload OR **load a bundled sample**, playback, overlays,
  **Run demo** (synthetic, no CV), **Run analysis (YOLO)**,
  **Click-to-track (SAM 2)**, **Retry transcode**, jobs panel.
- `/games/{id}/review` — tag events at the current time (keys 1–8), timeline,
  accept/reject/delete, CSV export. No CV needed.
- `/games/{id}/report` — official counts, shooting %, by-type bars, gaps.
- `/games/{id}/calibrate` — click court landmarks → save calibration.
- `/games/{id}/tracks` — roster + assign tracked objects to player/team.
- `/docs` — OpenAPI UI.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` on import | `uv sync --extra dev`. |
| Transcode failed / exit non-zero | Read the surfaced ffmpeg stderr; ensure `ffmpeg` is installed; click **Retry transcode**. Original playback still works. |
| CV deps "not installed" but you installed them | A `uv sync` removed them (exact sync, not in lock) OR you hit a different venv. Reinstall with `uv pip install --python .venv/bin/python ...` and run via `.venv/bin/python -m uvicorn`. |
| YOLO/SAM 2 button → 503 | Install the CV runtime; run the preflight. |
| SAM 2 config/checkpoint error on first click | Match `SAM2_CONFIG` to the installed package version (e.g. `configs/sam2.1/sam2.1_hiera_s.yaml`) and `SAM2_CHECKPOINT` to the downloaded file. |
| SAM 2 slow / OOM on Mac | Use `DEVICE=mps` and short clips. |
| Port already in use | run with `--port <N>`. |

## Conventions

- Keep `ruff`, `mypy`, and `pytest` green before committing.
- Do NOT add the CV runtime (torch/ultralytics/sam2) to `pyproject`/`uv.lock`,
  and never commit model weights (`models/`, `*.pt`).
