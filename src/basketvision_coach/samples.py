"""Discovery of bundled sample videos shipped in the repo's ``samples/`` folder.

Lets the UI offer "load a sample" so a coach can try the app without their own
footage. The directory is ``samples`` by default and overridable with the
``SAMPLES_DIR`` env var.
"""

from __future__ import annotations

import os
from pathlib import Path


def samples_dir() -> Path:
    return Path(os.environ.get("SAMPLES_DIR", "samples"))


def list_sample_videos(directory: Path | None = None) -> list[str]:
    """Return the names of ``.mp4`` clips available in the samples directory."""

    directory = directory or samples_dir()
    if not directory.is_dir():
        return []
    return sorted(
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".mp4"
    )


def resolve_sample(filename: str, directory: Path | None = None) -> Path | None:
    """Resolve a sample by name within the samples directory (no path traversal)."""

    directory = directory or samples_dir()
    # Strip any directory components so a request can't escape the samples dir.
    candidate = directory / Path(filename).name
    if candidate.is_file() and candidate.suffix.lower() == ".mp4":
        return candidate
    return None
