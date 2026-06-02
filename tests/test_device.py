from __future__ import annotations

import pytest

from basketvision_coach.analysis import resolve_device


def test_explicit_device_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEVICE", "cuda")
    assert resolve_device("mps") == "mps"


def test_explicit_device_is_normalized() -> None:
    assert resolve_device("  MPS ") == "mps"


def test_device_env_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEVICE", "CPU")
    assert resolve_device() == "cpu"


def test_auto_detect_without_torch_is_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    # torch is not installed in CI, so auto-detection should fall back to cpu.
    monkeypatch.delenv("DEVICE", raising=False)
    assert resolve_device() == "cpu"
