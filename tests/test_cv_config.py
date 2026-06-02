from __future__ import annotations

import pytest

from basketvision_coach.analysis import build_default_analyzer
from basketvision_coach.click_track import build_default_click_tracker


def test_yolo_model_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("YOLO_MODEL", raising=False)
    assert build_default_analyzer().model_name == "yolov8n.pt"


def test_yolo_model_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YOLO_MODEL", "yolov8m.pt")
    assert build_default_analyzer().model_name == "yolov8m.pt"


def test_sam2_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SAM2_CHECKPOINT", raising=False)
    monkeypatch.delenv("SAM2_CONFIG", raising=False)
    tracker = build_default_click_tracker()
    assert tracker.checkpoint == "sam2_hiera_small.pt"
    assert tracker.model_cfg == "sam2_hiera_s.yaml"


def test_sam2_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAM2_CHECKPOINT", "/models/sam2.1_hiera_small.pt")
    monkeypatch.setenv("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_s.yaml")
    tracker = build_default_click_tracker()
    assert tracker.checkpoint == "/models/sam2.1_hiera_small.pt"
    assert tracker.model_cfg == "configs/sam2.1/sam2.1_hiera_s.yaml"
