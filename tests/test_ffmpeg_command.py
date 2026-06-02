from __future__ import annotations

from pathlib import Path

import pytest

from basketvision_coach.video import (
    PROXY_SCALE_FILTER,
    _proxy_command,
    _run_ffmpeg,
)


def test_proxy_scale_filter_escapes_comma_and_forces_even() -> None:
    # ffmpeg splits filtergraphs on unescaped commas, so the min() comma must be
    # escaped; both dimensions must be even for libx264.
    assert PROXY_SCALE_FILTER == r"scale=-2:min(720\,trunc(ih/2)*2)"
    assert "\\," in PROXY_SCALE_FILTER


def test_proxy_command_uses_the_filter_and_even_pixfmt() -> None:
    cmd = _proxy_command(Path("/in/original.mp4"), Path("/out/proxy.mp4"))
    assert "-vf" in cmd
    assert cmd[cmd.index("-vf") + 1] == PROXY_SCALE_FILTER
    assert "yuv420p" in cmd  # broad player compatibility
    assert cmd[-1] == "/out/proxy.mp4"


def test_run_ffmpeg_reports_missing_binary() -> None:
    # A missing binary should raise a clear RuntimeError, not a bare FileNotFoundError.
    with pytest.raises(RuntimeError) as excinfo:
        _run_ffmpeg(["ffmpeg-not-installed-xyz", "-version"])
    message = str(excinfo.value)
    assert "ffmpeg-not-installed-xyz" in message
    assert "install ffmpeg" in message
