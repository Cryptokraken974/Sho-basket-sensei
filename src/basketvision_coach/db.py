from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


def data_root_from_env() -> Path:
    """Root directory for stored video assets and the default SQLite database.

    Defaults to ``./data`` so the database and video files live together and are
    writable without root; override with ``DATA_ROOT`` (e.g. a mounted volume).
    """

    return Path(os.environ.get("DATA_ROOT", "data"))


def database_url_from_env() -> str:
    explicit = os.environ.get("DATABASE_URL")
    if explicit:
        return explicit
    return f"sqlite:///{data_root_from_env() / 'basketvision.db'}"


def _ensure_sqlite_parent(url: str) -> None:
    """Create the parent directory for a file-backed SQLite database if needed."""

    prefix = "sqlite:///"
    if not url.startswith(prefix):
        return
    path_part = url[len(prefix) :]
    if not path_part or path_part == ":memory:":
        return
    parent = Path(path_part).expanduser().parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def build_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    url = database_url or database_url_from_env()
    connect_args: dict[str, object] = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        _ensure_sqlite_parent(url)
    engine = create_engine(url, connect_args=connect_args)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def session_scope(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
