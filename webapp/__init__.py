from __future__ import annotations

import secrets
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path

from flask import Flask

from webapp import database
from webapp.formatting import register_filters
from webapp.jobs import RunQueue
from webapp.repositories.runs import RunRepository
from webapp.security import init_csrf
from webapp.storage import Storage
from webapp.views import register_views

DEFAULTS = {
    "DATABASE": "consolidator.sqlite3",
    "MAX_CONTENT_LENGTH": 256 * 1024 * 1024,
    "PREVIEW_ROWS": 100,
    "SOURCE_PREVIEW_ROWS": 20,
    "DUPLICATE_ROWS": 100,
    "REVIEW_PAGE_SIZE": 20,
    "RUN_WORKERS": 1,
}
SECRET_FILE = "secret_key"


def create_app(config: Mapping[str, object] | None = None, instance_path: str | Path | None = None) -> Flask:
    app = Flask(__name__, instance_path=str(instance_path) if instance_path else None)
    app.config.from_mapping(DEFAULTS)
    app.config.from_mapping(config or {})
    instance = Path(app.instance_path)
    instance.mkdir(parents=True, exist_ok=True)
    if not app.config.get("SECRET_KEY"):
        app.config["SECRET_KEY"] = _secret_key(instance / SECRET_FILE)
    database.init_app(app)
    with closing(database.open_database(app)) as db:
        RunRepository(db).interrupt_unfinished()
    app.extensions["storage"] = Storage(instance)
    app.extensions["run_queue"] = RunQueue(int(app.config["RUN_WORKERS"]))
    init_csrf(app)
    register_filters(app)
    register_views(app)
    return app


def _secret_key(path: Path) -> str:
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    key = secrets.token_hex(32)
    path.write_text(key, encoding="utf-8")
    return key
