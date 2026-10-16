from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, current_app, g

SCHEMA = Path(__file__).with_name("schema.sql")
BUSY_TIMEOUT_SECONDS = 30.0


def connect(path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=BUSY_TIMEOUT_SECONDS)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def database_path(app: Flask) -> Path:
    path = Path(app.config["DATABASE"])
    return path if path.is_absolute() else Path(app.instance_path) / path


def open_database(app: Flask) -> sqlite3.Connection:
    return connect(database_path(app))


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = open_database(current_app)
    return g.db


def close_db(_error: BaseException | None = None) -> None:
    connection = g.pop("db", None)
    if connection is not None:
        connection.close()


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA.read_text(encoding="utf-8"))


def init_app(app: Flask) -> None:
    app.teardown_appcontext(close_db)
    with closing(open_database(app)) as connection:
        create_schema(connection)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
