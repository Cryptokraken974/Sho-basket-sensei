from __future__ import annotations

from pathlib import Path

import pytest

from basketvision_coach.checkcv import (
    check_module,
    check_path_env,
    main,
    run_checks,
)


def test_check_module_reports_missing() -> None:
    check = check_module("a_module_that_is_not_installed_xyz", "ghost", "install it")
    assert check.status == "fail"
    assert check.hint == "install it"


def test_check_module_reports_present() -> None:
    # 'json' is always importable.
    check = check_module("json", "json", "n/a")
    assert check.status == "ok"


def test_check_path_env_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"x")
    monkeypatch.setenv("SAM2_CHECKPOINT", str(ckpt))
    check = check_path_env("SAM2_CHECKPOINT", "default.pt", "SAM2_CHECKPOINT")
    assert check.status == "ok"


def test_check_path_env_missing_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAM2_CHECKPOINT", "/nope/model.pt")
    check = check_path_env("SAM2_CHECKPOINT", "default.pt", "SAM2_CHECKPOINT")
    assert check.status == "fail"


def test_check_path_env_config_is_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_s.yaml")
    check = check_path_env("SAM2_CONFIG", "sam2_hiera_s.yaml", "SAM2_CONFIG")
    assert check.status == "warn"  # not a local file, but may resolve via the package


def test_run_checks_and_main(capsys: pytest.CaptureFixture[str]) -> None:
    checks = run_checks()
    labels = {c.label for c in checks}
    assert {"resolved device", "torch", "ultralytics (YOLO)", "sam2"} <= labels

    assert main() == 0
    out = capsys.readouterr().out
    assert "CV preflight" in out
    assert "Run analysis (YOLO)" in out
