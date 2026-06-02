# BasketVision Coach — runnable container image.
#
# Includes ffmpeg for 720p proxy transcoding. Video assets and the SQLite
# database live under /data (mount a volume to persist them). The server honors
# the PORT env var so it slots into Render/Railway/Cloud Run/Heroku-style hosts.
#
#   docker build -t sensei .
#   docker run --rm -p 8000:8000 -v sensei-data:/data sensei
#
# Note: the optional CV runtime (ultralytics / sam2 / torch) is NOT installed
# here to keep the image light; add it in a derived image if you want the
# YOLO/SAM 2 buttons to work.
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

ENV DATA_ROOT=/data \
    PORT=8000
VOLUME ["/data"]
EXPOSE 8000

# Shell form so ${PORT} is expanded at runtime.
CMD uv run --no-dev uvicorn basketvision_coach.api:app --host 0.0.0.0 --port ${PORT:-8000}
