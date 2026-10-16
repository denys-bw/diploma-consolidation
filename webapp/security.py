from __future__ import annotations

import hmac
import secrets

from flask import Flask, abort, request, session

TOKEN_FIELD = "csrf_token"
TOKEN_HEADER = "X-CSRF-Token"
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
STALE_FORM = "Сторінку відкрито надто давно або в іншому вікні. Оновіть її й повторіть дію."


def csrf_token() -> str:
    token = session.get(TOKEN_FIELD)
    if not token:
        token = secrets.token_urlsafe(32)
        session[TOKEN_FIELD] = token
    return token


def init_csrf(app: Flask) -> None:
    app.jinja_env.globals["csrf_token"] = csrf_token

    @app.before_request
    def verify_csrf_token() -> None:
        if request.method in SAFE_METHODS:
            return
        expected = session.get(TOKEN_FIELD, "")
        sent = request.form.get(TOKEN_FIELD) or request.headers.get(TOKEN_HEADER, "")
        if not expected or not hmac.compare_digest(str(sent), str(expected)):
            abort(400, description=STALE_FORM)
