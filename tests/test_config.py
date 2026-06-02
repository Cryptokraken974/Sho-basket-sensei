from __future__ import annotations

from pathlib import Path

import pytest

from basketvision_coach.db import data_root_from_env, database_url_from_env


def test_data_root_defaults_to_local_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATA_ROOT", raising=False)
    assert data_root_from_env() == Path("data")


def test_data_root_honors_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_ROOT", "/srv/sensei")
    assert data_root_from_env() == Path("/srv/sensei")


def test_database_url_defaults_under_data_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DATA_ROOT", raising=False)
    assert database_url_from_env() == "sqlite:///data/basketvision.db"


def test_database_url_follows_data_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("DATA_ROOT", "/srv/sensei")
    assert database_url_from_env() == "sqlite:////srv/sensei/basketvision.db"


def test_explicit_database_url_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://user@host/db")
    monkeypatch.setenv("DATA_ROOT", "/srv/sensei")
    assert database_url_from_env() == "postgresql://user@host/db"
