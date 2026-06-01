from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


def database_url_from_env() -> str:
    return os.environ.get("DATABASE_URL", "sqlite:///data/basketvision.db")


def build_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    url = database_url or database_url_from_env()
    connect_args: dict[str, object] = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        if ":memory:" not in url:
            database_path = url.removeprefix("sqlite+pysqlite:///").removeprefix("sqlite:///")
            if database_path:
                Path(database_path).parent.mkdir(parents=True, exist_ok=True)
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
