"""CV preflight check: ``uv run python -m basketvision_coach.checkcv``.

Reports the resolved torch device and whether the optional CV runtime
(torch / ultralytics / opencv / sam2) and the configured SAM 2 checkpoint/config
are available — so the YOLO and click-to-track features can be validated in one
command before launching the server.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from basketvision_coach.analysis import resolve_device

_MARK = {"ok": "✓", "fail": "✗", "warn": "–"}


@dataclass(frozen=True, slots=True)
class Check:
    label: str
    status: str  # "ok" | "fail" | "warn"
    detail: str
    hint: str = ""


def _module_installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def check_module(import_name: str, label: str, hint: str) -> Check:
    if _module_installed(import_name):
        return Check(label, "ok", "installed")
    return Check(label, "fail", "not installed", hint)


def check_torch_mps() -> Check:
    if not _module_installed("torch"):
        return Check("torch MPS (Apple GPU)", "warn", "torch not installed yet")
    import torch  # noqa: PLC0415 - optional dependency, imported on demand

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return Check("torch MPS (Apple GPU)", "ok", "available")
    return Check(
        "torch MPS (Apple GPU)",
        "warn",
        "not available (CPU only)",
        "expected on Apple Silicon; CPU still works but is slower",
    )


def check_path_env(env: str, default: str, label: str) -> Check:
    value = os.environ.get(env, default)
    path = Path(value)
    if path.is_file():
        return Check(label, "ok", f"{value} (found)")
    if path.suffix in {".yaml", ".yml"}:
        return Check(label, "warn", f"{value} (not a local file; may resolve via the sam2 package)")
    return Check(label, "fail", f"{value} (missing)", f"set {env}=/path/to/file")


def check_device() -> Check:
    device = resolve_device()
    hint = "" if device != "cpu" else "set DEVICE=mps on Apple Silicon for GPU acceleration"
    return Check("resolved device", "ok", device, hint)


def run_checks() -> list[Check]:
    return [
        check_device(),
        check_module("torch", "torch", "uv pip install torch"),
        check_torch_mps(),
        check_module(
            "ultralytics",
            "ultralytics (YOLO)",
            "uv pip install ultralytics opencv-python-headless",
        ),
        check_module("cv2", "opencv (cv2)", "uv pip install opencv-python-headless"),
        check_module(
            "sam2",
            "sam2",
            'uv pip install "git+https://github.com/facebookresearch/sam2.git"',
        ),
        check_path_env("SAM2_CHECKPOINT", "sam2_hiera_small.pt", "SAM2_CHECKPOINT"),
        check_path_env("SAM2_CONFIG", "sam2_hiera_s.yaml", "SAM2_CONFIG"),
    ]


def _installed(checks: list[Check], label: str) -> bool:
    return any(c.label == label and c.status == "ok" for c in checks)


def main() -> int:
    checks = run_checks()
    print("BasketVision Coach — CV preflight\n")
    for check in checks:
        line = f"  {_MARK[check.status]} {check.label:<24} {check.detail}"
        print(line)
        if check.hint and check.status != "ok":
            print(f"      → {check.hint}")

    yolo_ready = _installed(checks, "torch") and _installed(checks, "ultralytics (YOLO)")
    sam_ready = (
        _installed(checks, "torch")
        and _installed(checks, "sam2")
        and _installed(checks, "SAM2_CHECKPOINT")
    )
    print("\nSummary")
    print(f"  Run analysis (YOLO):   {'READY' if yolo_ready else 'not ready'}")
    print(f"  Click-to-track (SAM2): {'READY' if sam_ready else 'not ready'}")
    print("\nThe demo, review, report, calibration and roster features work without any of this.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
