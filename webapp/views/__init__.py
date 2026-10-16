from __future__ import annotations

from flask import Flask, render_template
from werkzeug.exceptions import HTTPException

from webapp.views import mapping, projects, reviews, runs, sources
from webapp.views.common import project_progress

ERROR_TITLES = {
    400: "Запит не вдалося виконати",
    404: "Сторінку не знайдено",
    405: "Дія недоступна",
    413: "Файл завеликий",
    500: "Внутрішня помилка",
}
ERROR_DESCRIPTIONS = {
    400: "Застосунок не зміг обробити запит. Поверніться на попередню сторінку й повторіть дію.",
    404: "Адреса неправильна або сторінку вже видалено.",
    405: "Цю дію не можна виконати за такою адресою.",
    500: "Під час обробки запиту сталася помилка. Подробиці записано в журнал застосунку.",
}


def register_views(app: Flask) -> None:
    for module in (projects, sources, mapping, runs, reviews):
        app.register_blueprint(module.blueprint)
    app.jinja_env.globals["project_progress"] = project_progress

    @app.errorhandler(HTTPException)
    def http_error(error: HTTPException) -> tuple[str, int]:
        code = error.code or 500
        description = error.description
        if description == type(error).description:
            description = ERROR_DESCRIPTIONS.get(code, description)
        if code == 413:
            limit = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
            description = f"Сумарний розмір файлів перевищує {limit} МБ. Завантажте файли частинами."
        title = ERROR_TITLES.get(code, error.name)
        return render_template("errors.html", code=code, title=title, description=description), code
