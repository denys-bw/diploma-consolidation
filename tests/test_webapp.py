import io
import json
from contextlib import closing
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.consolidation import CHICAGO_SOURCE_CHECKSUMS
from benchmarks.dataset import DATA
from consolidator.imputation import AdaptiveImputer
from consolidator.io import read_bytes
from consolidator.linkage import CoTraining, FellegiSunterScorer
from consolidator.pipeline import Pipeline
from webapp import create_app
from webapp import options as run_options
from webapp.database import connect, database_path, timestamp
from webapp.formatting import duration, file_size, number, percent, points
from webapp.options import (
    AHP_PAIRS,
    DEFAULT_PRESET,
    BlockingMode,
    DuplicateMode,
    ImputationMode,
    ModelMode,
    OptionsError,
    RunOptions,
)
from webapp.repositories.runs import INTERRUPTED, RunRepository
from webapp.services import runs as run_service

TOKEN = "test-token"
RUN_TIMEOUT = 300
FODORS = """name,addr,city,phone
arnie morton's of chicago,435 s. la cienega blv.,los angeles,310/246-1501
art's delicatessen,12224 ventura blvd.,studio city,818/762-1221
hotel bel-air,701 stone canyon rd.,bel air,310/472-1211
cafe bizou,14016 ventura blvd.,sherman oaks,818/788-3536
campanile,624 s. la brea ave.,los angeles,213/938-1447
chinois on main,2709 main st.,santa monica,
"""
ZAGATS = [
    {
        "name": "arnie morton's of chicago",
        "address": "435 s. la cienega blvd.",
        "city": "los angeles",
        "phone": "310-246-1501",
    },
    {"name": "art's deli", "address": "12224 ventura blvd.", "city": "studio city", "phone": "818-762-1221"},
    {"name": "bel-air hotel", "address": "701 stone canyon rd.", "city": "bel air", "phone": "310-472-1211"},
    {"name": "cafe bizou", "address": "14016 ventura blvd.", "city": "sherman oaks", "phone": "818-788-3536"},
    {"name": "patina", "address": "5955 melrose ave.", "city": "los angeles", "phone": "213-467-1108"},
]
FILES = {"fodors.csv": FODORS.encode("utf-8"), "zagats.json": json.dumps(ZAGATS).encode("utf-8")}
PLACES = """site,id,name,city
fodors,1,arnie morton's of chicago,los angeles
fodors,2,art's delicatessen,studio city
zagats,1,arnie morton's of chicago,los angeles
zagats,2,art's deli,studio city
zagats,3,patina,los angeles
"""
SETTINGS = {
    "policy": "dominant",
    "phone_region": "",
    "phone_format": "999-999-9999",
    "duplicates": "auto",
    "blocking": "tokens",
    "match_probability": "0.5",
    "false_match_rate": "5",
    "missed_match_rate": "5",
    "imputation": "auto",
    "neighbours": "",
    **{f"ahp_{left}_{right}": value for (left, right), value in zip(AHP_PAIRS, DEFAULT_PRESET)},
}
BLOCKING_LABELS = {
    "keys": "стандартне за ключами",
    "sorted": "сортоване сусідство (w=",
    "full": "без блокування",
    "tokens": "TF-IDF, найближчих сусідів: адаптивно",
}
NAMES_FODORS = "\n".join(
    ["name,city", *(",".join(line.split(",")[0::2][:2]) for line in FODORS.strip().splitlines()[1:])]
)
NAMES_ZAGATS = [{"name": record["name"], "city": record["city"]} for record in ZAGATS]
NAME_FILES = {
    "fodors.csv": (NAMES_FODORS + "\n").encode("utf-8"),
    "zagats.json": json.dumps(NAMES_ZAGATS).encode("utf-8"),
}


@pytest.fixture
def app(tmp_path):
    application = create_app({"TESTING": True}, instance_path=tmp_path / "instance")
    yield application
    application.extensions["run_queue"].shutdown()


@pytest.fixture
def client(app):
    test_client = app.test_client()
    with test_client.session_transaction() as session:
        session["csrf_token"] = TOKEN
    return test_client


def post(client, url, data=None, **options):
    return client.post(url, data={"csrf_token": TOKEN, **(data or {})}, **options)


def upload(client, project_id, files):
    payload = {"files": [(io.BytesIO(content), name) for name, content in files.items()]}
    return post(client, f"/projects/{project_id}/files", payload, content_type="multipart/form-data")


def create_project(client, name="Ресторани"):
    response = post(client, "/projects", {"name": name, "description": "Fodor's і Zagat's"})
    assert response.status_code == 302
    return int(response.headers["Location"].rsplit("/", 1)[1])


def project_with_sources(client, files=FILES):
    project_id = create_project(client)
    assert upload(client, project_id, files).status_code == 302
    return project_id


def start_run(client, project_id, **changes):
    response = post(client, f"/projects/{project_id}/settings", {**SETTINGS, **changes, "action": "run"})
    assert response.status_code == 302, response.get_data(as_text=True)
    return int(response.headers["Location"].rsplit("/", 1)[1])


def finished(app, client, run_id):
    assert app.extensions["run_queue"].join(RUN_TIMEOUT)
    return client.get(f"/runs/{run_id}/status").get_json()


def query(app, sql, *parameters):
    with closing(connect(database_path(app))) as db:
        return db.execute(sql, parameters).fetchall()


def text(response):
    return response.get_data(as_text=True)


def mapping_form(rows, action="save"):
    form = {"rows": str(len(rows)), "action": action}
    for index, (current, name, column_type, role, selected) in enumerate(rows):
        form[f"current_{index}"] = current
        form[f"name_{index}"] = name
        form[f"type_{index}"] = column_type
        form[f"role_{index}"] = role
        if selected:
            form[f"selected_{index}"] = "on"
    return form


def current_rows(app, roles=None):
    attributes = query(app, "SELECT name, type FROM attribute ORDER BY position")
    return [
        (row["name"], row["name"], row["type"], (roles or {}).get(row["name"], "attribute"), False)
        for row in attributes
    ]


def review_form(pairs, action="save", page=1):
    form = {"pairs": str(len(pairs)), "page": str(page), "action": action}
    for row, (left, right, decision) in enumerate(pairs):
        form[f"left_source_{row}"], form[f"left_key_{row}"] = left
        form[f"right_source_{row}"], form[f"right_key_{row}"] = right
        form[f"decision_{row}"] = decision
    return form


def labelled_files():
    lines = FODORS.strip().splitlines()
    fodors = "\n".join(f"{line},{label}" for line, label in zip(lines, ["ref", "a", "b", "c", "d", "e", "f"]))
    zagats = [{**record, "ref": label} for record, label in zip(ZAGATS, "abcdg")]
    return {"fodors.csv": (fodors + "\n").encode("utf-8"), "zagats.json": json.dumps(zagats).encode("utf-8")}


def test_project_is_created_and_listed(client):
    assert "Проєктів ще немає" in text(client.get("/"))
    project_id = create_project(client, "Каталог товарів")
    page = text(client.get("/"))
    assert "Каталог товарів" in page
    assert f"/projects/{project_id}" in page


def test_project_name_is_required(client):
    response = post(client, "/projects", {"name": "   "}, follow_redirects=True)
    assert "Вкажіть назву проєкту." in text(response)


def test_forms_without_token_are_rejected(client):
    response = client.post("/projects", data={"name": "Без токена"})
    assert response.status_code == 400
    assert "Оновіть її й повторіть дію" in text(response)


def test_unknown_pages_are_explained_in_ukrainian(client):
    response = client.get("/projects/999")
    assert response.status_code == 404
    assert "Проєкт не знайдено." in text(response)
    assert "Адреса неправильна" in text(client.get("/nothing-here"))


def test_too_large_upload_is_explained(app, client):
    app.config["MAX_CONTENT_LENGTH"] = 1024
    project_id = create_project(client)
    response = upload(client, project_id, {"big.csv": b"a,b\n" + b"1,2\n" * 1000})
    assert response.status_code == 413
    assert "Сумарний розмір файлів перевищує" in text(response)


def test_upload_reads_sources_and_matches_columns(app, client):
    project_id = create_project(client)
    response = upload(client, project_id, FILES)
    assert response.status_code == 302
    page = text(client.get(f"/projects/{project_id}"))
    assert "Додано джерел: 2 (fodors, zagats)" in page
    sources = query(
        app,
        "SELECT source.name, source_table.format, record_count, column_count "
        "FROM source JOIN source_table ON source_table.id = source.table_id ORDER BY source.id",
    )
    assert [tuple(row) for row in sources] == [("fodors", "CSV", 6, 4), ("zagats", "JSON", 5, 4)]
    attributes = query(app, "SELECT name, type, role FROM attribute ORDER BY position")
    assert [tuple(row) for row in attributes] == [
        ("name", "text", "attribute"),
        ("addr", "text", "attribute"),
        ("city", "text", "attribute"),
        ("phone", "phone", "attribute"),
    ]
    columns = query(
        app,
        "SELECT column_name FROM attribute_column JOIN attribute ON attribute.id = attribute_column.attribute_id "
        "WHERE attribute.name = 'addr' ORDER BY source_id",
    )
    assert [row["column_name"] for row in columns] == ["addr", "address"]
    assert len(list((Path(app.instance_path) / "uploads").iterdir())) == 2


def test_unreadable_upload_is_reported_and_not_stored(app, client):
    project_id = create_project(client)
    response = upload(client, project_id, {"table.xls": b"not a spreadsheet"})
    page = text(client.get(response.headers["Location"]))
    assert "«table.xls»: Формат файлу «table.xls» не підтримується: збережіть файл у форматі .xlsx" in page
    assert query(app, "SELECT count(*) AS count FROM source_file")[0]["count"] == 0
    assert not (Path(app.instance_path) / "uploads").exists()


def test_same_source_names_get_suffixes(app, client):
    project_id = project_with_sources(client)
    upload(client, project_id, {"fodors.csv": FODORS.encode("utf-8")})
    names = [row["name"] for row in query(app, "SELECT name FROM source ORDER BY id")]
    assert names == ["fodors", "zagats", "fodors (2)"]


def test_source_page_shows_profile_and_numbered_rows(client):
    project_id = project_with_sources(client)
    page = text(client.get(f"/projects/{project_id}/sources/1"))
    assert "Профіль колонок" in page
    assert "Викидів (IQR)" in page and "Викидів (ізоляційний ліс)" in page
    assert "кодування: utf-8" in page
    assert "роздільник: кома" in page
    assert "№ рядка" in page
    assert '<td class="num muted">1</td>' in page
    assert "310/246-1501" in page
    assert client.get(f"/projects/{project_id}/sources/99").status_code == 404


def test_table_is_split_into_sources_with_own_keys(app, client):
    project_id = project_with_sources(client, {"places.csv": PLACES.encode("utf-8")})
    response = post(
        client,
        f"/projects/{project_id}/sources/1/table",
        {"key_column": "id", "split_column": "site"},
        follow_redirects=True,
    )
    assert "Таблицю «places» оновлено. Джерел: 2, атрибутів: 2." in text(response)
    sources = query(app, "SELECT id, part, name, record_count, column_count FROM source ORDER BY id")
    assert [tuple(row)[1:] for row in sources] == [("fodors", "fodors", 2, 2), ("zagats", "zagats", 3, 2)]
    page = text(client.get(f"/projects/{project_id}/sources/{sources[1]['id']}"))
    assert ">id</th>" in page
    assert '<td class="num muted">3</td>' in page
    assert "виділено з таблиці: places" in page
    attributes = [row["name"] for row in query(app, "SELECT name FROM attribute ORDER BY position")]
    assert attributes == ["name", "city"]
    run_id = start_run(client, project_id)
    assert finished(app, client, run_id)["status"] == "finished"
    assert query(app, "SELECT record_count FROM report")[0]["record_count"] == 5


def test_wrong_table_structure_is_explained(app, client):
    project_id = project_with_sources(client, {"places.csv": PLACES.encode("utf-8")})
    url = f"/projects/{project_id}/sources/1/table"
    response = post(client, url, {"key_column": "site"}, follow_redirects=True)
    assert "Колонка «site» не може бути ключем джерела «places»: повторюються значення fodors, zagats" in text(response)
    response = post(client, url, {"key_column": "id"}, follow_redirects=True)
    assert "Колонка «id» не може бути ключем джерела «places»: повторюються значення 1, 2" in text(response)
    response = post(client, url, {"key_column": "id", "split_column": "id"}, follow_redirects=True)
    assert "Одна колонка не може бути одночасно ідентифікатором запису і ознакою джерела." in text(response)
    assert [row["name"] for row in query(app, "SELECT name FROM source")] == ["places"]


def test_source_can_be_switched_off(app, client):
    project_id = project_with_sources(client)
    post(client, f"/projects/{project_id}/sources/2/toggle")
    assert query(app, "SELECT included FROM source WHERE id = 2")[0]["included"] == 0
    names = [row["name"] for row in query(app, "SELECT name FROM attribute ORDER BY position")]
    assert names == ["name", "addr", "city", "phone"]
    owners = query(app, "SELECT DISTINCT source_id FROM attribute_column")
    assert [row["source_id"] for row in owners] == [1]


def test_mapping_edits_are_saved(app, client):
    project_id = project_with_sources(client)
    rows = [
        ("name", "Назва", "text", "attribute", False),
        ("addr", "Адреса", "text", "attribute", False),
        ("city", "Місто", "text", "ignored", False),
        ("phone", "Телефон", "phone", "attribute", False),
    ]
    response = post(client, f"/projects/{project_id}/mapping", mapping_form(rows), follow_redirects=True)
    assert "Зміни зіставлення збережено." in text(response)
    saved = query(app, "SELECT name, role FROM attribute ORDER BY position")
    assert [tuple(row) for row in saved] == [
        ("Назва", "attribute"),
        ("Адреса", "attribute"),
        ("Місто", "ignored"),
        ("Телефон", "attribute"),
    ]


def test_mapping_rejects_repeated_names_and_second_reference(app, client):
    project_id = project_with_sources(client)
    rows = current_rows(app)
    rows[3] = ("phone", "city", "phone", "attribute", False)
    response = post(client, f"/projects/{project_id}/mapping", mapping_form(rows), follow_redirects=True)
    assert "Назви атрибутів мають бути різними" in text(response)
    rows = current_rows(app, {"name": "reference", "city": "reference"})
    response = post(client, f"/projects/{project_id}/mapping", mapping_form(rows), follow_redirects=True)
    assert "Еталоном для оцінювання може бути лише один атрибут." in text(response)
    saved = query(app, "SELECT name, role FROM attribute ORDER BY position")
    assert [tuple(row) for row in saved] == [(name, "attribute") for name in ("name", "addr", "city", "phone")]


def test_columns_can_be_detached_and_merged_back(app, client):
    project_id = project_with_sources(client)
    response = post(client, f"/projects/{project_id}/mapping", mapping_form(current_rows(app), "detach:1:2"))
    assert response.status_code == 302
    attributes = [row["name"] for row in query(app, "SELECT name FROM attribute ORDER BY position")]
    assert attributes == ["name", "addr", "city", "phone", "address"]
    rows = [(*row[:4], row[0] in ("addr", "address")) for row in current_rows(app)]
    response = post(client, f"/projects/{project_id}/mapping", mapping_form(rows, "merge"), follow_redirects=True)
    assert "Атрибути об&#39;єднано в «addr»." in text(response)
    attributes = [row["name"] for row in query(app, "SELECT name FROM attribute ORDER BY position")]
    assert attributes == ["name", "addr", "city", "phone"]


def test_columns_of_one_source_merge_into_a_composite_field(app, client):
    project_id = project_with_sources(client)
    rows = [(*row[:4], row[0] in ("addr", "city")) for row in current_rows(app)]
    response = post(client, f"/projects/{project_id}/mapping", mapping_form(rows, "merge"), follow_redirects=True)
    assert "Атрибути об&#39;єднано в «addr»." in text(response)
    columns = {row["column_name"] for row in query(app, "SELECT column_name FROM attribute_column")}
    assert "addr + city" in columns
    run_id = start_run(client, project_id)
    assert finished(app, client, run_id)["status"] == "finished"


def test_mapping_can_be_rebuilt_with_another_significance_level(app, client):
    project_id = project_with_sources(client)
    post(client, f"/projects/{project_id}/mapping", mapping_form(current_rows(app), "detach:1:2"))
    response = post(
        client,
        f"/projects/{project_id}/mapping",
        {"action": "rebuild", "schema_alpha": "0,01"},
        follow_redirects=True,
    )
    assert "Зіставлення перебудовано автоматично" in text(response)
    options = json.loads(query(app, "SELECT options FROM project")[0]["options"])
    assert options["schema_alpha"] == pytest.approx(0.01)


def test_settings_page_offers_default_weights(client):
    project_id = project_with_sources(client)
    page = text(client.get(f"/projects/{project_id}/settings"))
    assert 'data-weight-value="uniqueness">0,637<' in page
    assert "data-ahp-cr>0,033<" in page
    assert "Пріоритет джерел" in page
    assert 'name="transliterate"' in page


def test_invalid_settings_are_explained_and_kept_in_form(client):
    project_id = project_with_sources(client)
    data = {**SETTINGS, "phone_format": "abc", "match_probability": "2", "action": "save"}
    response = post(client, f"/projects/{project_id}/settings", data)
    page = text(response)
    assert response.status_code == 400
    assert "Шаблон телефону може містити лише цифри" in page
    assert "Поріг імовірності збігу: значення має бути від 0,01 до 0,99." in page
    assert 'value="abc"' in page


def test_tolerated_error_rates_are_validated(client):
    project_id = project_with_sources(client)
    data = {**SETTINGS, "false_match_rate": "60", "action": "save"}
    response = post(client, f"/projects/{project_id}/settings", data)
    assert response.status_code == 400
    assert "Допустима частка хибних збігів, %: значення має бути від 0 до 50." in text(response)


def test_inconsistent_judgements_are_rejected(client):
    project_id = project_with_sources(client)
    data = {
        **SETTINGS,
        "ahp_completeness_uniqueness": "9",
        "ahp_completeness_consistency": "1/9",
        "ahp_uniqueness_consistency": "9",
        "action": "save",
    }
    response = post(client, f"/projects/{project_id}/settings", data)
    assert response.status_code == 400
    assert "Судження МАІ суперечливі" in text(response)


def test_ahp_preview_reproduces_default_judgements(client):
    query_string = {f"ahp_{left}_{right}": value for (left, right), value in zip(AHP_PAIRS, DEFAULT_PRESET)}
    result = client.get("/api/ahp", query_string=query_string).get_json()
    assert result["weights"]["completeness"] == pytest.approx(0.2583, abs=1e-4)
    assert result["weights"]["uniqueness"] == pytest.approx(0.6370, abs=1e-4)
    assert result["weights"]["consistency"] == pytest.approx(0.1047, abs=1e-4)
    assert result["consistency_ratio"] == pytest.approx(0.0332, abs=1e-4)
    assert result["consistent"] is True
    assert client.get("/api/ahp", query_string={"ahp_completeness_uniqueness": "10"}).status_code == 400


def test_settings_are_saved_with_priorities_and_rules(app, client):
    project_id = project_with_sources(client)
    data = {
        **SETTINGS,
        "duplicates": "across_clean",
        "phone_region": "US",
        "blocking": "sorted",
        "window": "5",
        "transliterate": "on",
        "attribute_rows": "2",
        "attribute_0": "name",
        "resolution_0": "longest",
        "attribute_1": "addr",
        "resolution_1": "",
        "priority_1": "2",
        "priority_2": "1",
        "action": "save",
    }
    response = post(client, f"/projects/{project_id}/settings", data, follow_redirects=True)
    assert "Налаштування збережено." in text(response)
    resolutions = query(app, "SELECT name, resolution FROM attribute ORDER BY position")
    assert [tuple(row) for row in resolutions][:2] == [("name", "longest"), ("addr", None)]
    priorities = query(app, "SELECT id, priority FROM source ORDER BY id")
    assert [tuple(row) for row in priorities] == [(1, 2), (2, 1)]
    options = RunOptions.from_dict(json.loads(query(app, "SELECT options FROM project")[0]["options"]))
    assert options.duplicates is DuplicateMode.ACROSS_CLEAN
    assert options.blocking is BlockingMode.SORTED
    assert options.window == 5
    assert options.transliterate is True
    assert options.phone_region == "US"


def test_run_builds_report_equal_to_library_result(app, client):
    project_id = project_with_sources(client)
    run_id = start_run(client, project_id)
    state = finished(app, client, run_id)
    assert state["status"] == "finished", state
    sources = [source.numbered() for name, content in FILES.items() for source in read_bytes(content, name)]
    settings = RunOptions.from_form(SETTINGS).pipeline_settings(source_priority=["fodors", "zagats"])
    expected = Pipeline(settings).run(sources)
    report = query(app, "SELECT * FROM report")[0]
    assert report["index_before"] == pytest.approx(expected.index_before)
    assert report["index_after"] == pytest.approx(expected.index_after)
    assert report["entity_count"] == expected.fusion.n_entities
    assert report["merged_records"] == expected.fusion.merged_records
    assert report["record_count"] == 11
    scores = {row["criterion"]: tuple(row)[2:] for row in query(app, "SELECT * FROM quality_score")}
    assert scores["uniqueness"] == pytest.approx(
        (settings.weights["uniqueness"], expected.quality_before["uniqueness"], expected.quality_after["uniqueness"])
    )
    page = text(client.get(f"/runs/{run_id}"))
    assert "Звіт обробки №" in page
    assert "Консолідовані дані (CSV)" in page
    assert "Профіль результату" in page
    assert f"{expected.index_after * 100:.2f}".replace(".", ",") in page
    assert "Оцінювання за еталоном" not in page
    assert "Пар, щодо яких дві моделі не згодні" in page
    reliability = expected.linkage.reliability()
    assert ("Пошук дублікатів надійний." in page) is reliability.reliable


def test_report_files_can_be_downloaded(app, client):
    project_id = project_with_sources(client)
    run_id = start_run(client, project_id)
    finished(app, client, run_id)
    consolidated = client.get(f"/runs/{run_id}/download/consolidated")
    assert consolidated.status_code == 200
    assert "attachment" in consolidated.headers["Content-Disposition"]
    table = pd.read_csv(io.BytesIO(consolidated.data), dtype=str, keep_default_na=False, encoding="utf-8-sig")
    assert list(table.columns) == ["Сутність", "Джерела", "Записів", "name", "addr", "city", "phone"]
    assert table["phone"].str.fullmatch(r"(\d{3}-\d{3}-\d{4})?").all()
    workbook = client.get(f"/runs/{run_id}/download/report")
    assert workbook.data[:2] == b"PK"
    database = client.get(f"/runs/{run_id}/download/database")
    assert database.data.startswith(b"SQLite format 3")
    summary = client.get(f"/runs/{run_id}/download/summary").get_json()
    assert summary["summary"]["record_count"] == 11
    assert client.get(f"/runs/{run_id}/download/unknown").status_code == 404


def test_entity_page_shows_source_records(app, client):
    project_id = project_with_sources(client)
    run_id = start_run(client, project_id)
    finished(app, client, run_id)
    page = text(client.get(f"/runs/{run_id}/entities/0"))
    assert "Консолідований запис" in page
    assert "fodors" in page and "zagats" in page
    assert "310/246-1501" in page and "310-246-1501" in page
    assert client.get(f"/runs/{run_id}/entities/500").status_code == 404


def test_reference_column_evaluates_found_duplicates(app, client):
    project_id = project_with_sources(client, labelled_files())
    rows = current_rows(app, {"ref": "reference"})
    post(client, f"/projects/{project_id}/mapping", mapping_form(rows))
    assert "Еталон: «ref»" in text(client.get(f"/projects/{project_id}"))
    run_id = start_run(client, project_id)
    assert finished(app, client, run_id)["status"] == "finished"
    evaluation = query(app, "SELECT reference, true_entities, true_pairs, found_pairs, correct_pairs FROM evaluation")
    assert tuple(evaluation[0]) == ("ref", 7, 4, 4, 4)
    page = text(client.get(f"/runs/{run_id}"))
    assert "Оцінювання за еталоном" in page
    assert percent(1.0, 2) in page
    consolidated = client.get(f"/runs/{run_id}/download/consolidated")
    assert "ref" not in pd.read_csv(io.BytesIO(consolidated.data), encoding="utf-8-sig").columns


def test_review_answers_are_used_in_next_run(app, client):
    app.config["REVIEW_PAGE_SIZE"] = 2
    project_id = project_with_sources(client, NAME_FILES)
    run_id = start_run(client, project_id, duplicates="across", phone_format="")
    assert finished(app, client, run_id)["status"] == "finished"
    report = query(app, "SELECT reviewed_pairs FROM report WHERE run_id = ?", run_id)
    assert report[0]["reviewed_pairs"] == 0
    first = text(client.get(f"/runs/{run_id}/review"))
    assert "Пара 2" in first and "Пара 3" not in first
    assert "Зберегти й далі" in first
    assert "Свідчення" in first
    assert "за " in first or "проти " in first
    assert "Пара 3" in text(client.get(f"/runs/{run_id}/review?page=2"))
    pairs = [
        (("fodors", "1"), ("zagats", "1"), "match"),
        (("zagats", "2"), ("fodors", "2"), "different"),
    ]
    response = post(client, f"/runs/{run_id}/review", review_form(pairs, "next"))
    assert response.headers["Location"].endswith(f"/runs/{run_id}/review?page=2")
    saved = query(app, "SELECT left_source, left_key, right_source, right_key, is_match FROM pair_review ORDER BY 2")
    assert [tuple(row) for row in saved] == [("fodors", "1", "zagats", "1", 1), ("fodors", "2", "zagats", "2", 0)]
    response = post(client, f"/runs/{run_id}/review", review_form(pairs, "rerun"))
    rerun = int(response.headers["Location"].rsplit("/", 1)[1])
    assert rerun != run_id
    assert finished(app, client, rerun)["status"] == "finished"
    report = query(app, "SELECT reviewed_pairs FROM report WHERE run_id = ?", rerun)
    assert report[0]["reviewed_pairs"] == 2
    page = text(client.get(f"/runs/{rerun}/review"))
    assert 'value="match" checked' in page and 'value="different" checked' in page
    post(client, f"/runs/{rerun}/review", review_form([(("fodors", "1"), ("zagats", "1"), "")]))
    assert len(query(app, "SELECT * FROM pair_review")) == 1


def test_empty_review_zone_is_explained(app, client):
    project_id = project_with_sources(client)
    run_id = start_run(client, project_id)
    finished(app, client, run_id)
    assert "Пар для ручної перевірки немає" in text(client.get(f"/runs/{run_id}/review"))


@pytest.mark.parametrize("blocking", sorted(BLOCKING_LABELS))
def test_runs_use_selected_blocking(app, client, blocking):
    project_id = project_with_sources(client)
    run_id = start_run(client, project_id, blocking=blocking)
    assert finished(app, client, run_id)["status"] == "finished"
    assert BLOCKING_LABELS[blocking] in text(client.get(f"/runs/{run_id}"))


def test_run_without_duplicate_search_keeps_every_record(app, client):
    project_id = project_with_sources(client)
    run_id = start_run(client, project_id, duplicates="none", imputation="none")
    assert finished(app, client, run_id)["status"] == "finished"
    report = query(app, "SELECT entity_count, candidate_pairs FROM report WHERE run_id = ?", run_id)[0]
    assert tuple(report) == (11, 0)


def test_full_comparison_limit_is_checked_before_run(client, monkeypatch):
    monkeypatch.setattr(run_options, "MAX_FULL_PAIRS", 10)
    project_id = project_with_sources(client)
    response = post(client, f"/projects/{project_id}/settings", {**SETTINGS, "blocking": "full", "action": "run"})
    page = text(client.get(response.headers["Location"]))
    assert "Без блокування довелося б порівняти 55 пар записів, а допустимо не більше 10." in page


def test_single_source_needs_search_inside_sources(client):
    project_id = project_with_sources(client, {"fodors.csv": FODORS.encode("utf-8")})
    response = post(client, f"/projects/{project_id}/settings", {**SETTINGS, "duplicates": "across", "action": "run"})
    assert "У проєкті одне джерело" in text(client.get(response.headers["Location"]))
    assert RunOptions().duplicates is DuplicateMode.AUTO


def test_failed_run_shows_the_reason(app, client, monkeypatch):
    def broken(*arguments, **options):
        raise ValueError("Колонка «phone» не має жодного значення")

    monkeypatch.setattr(run_service.Pipeline, "run", broken)
    project_id = project_with_sources(client)
    run_id = start_run(client, project_id)
    assert finished(app, client, run_id)["status"] == "failed"
    page = text(client.get(f"/runs/{run_id}"))
    assert "Обробку не завершено" in page
    assert "Колонка «phone» не має жодного значення" in page
    assert not (Path(app.instance_path) / "runs" / str(run_id)).exists()


def test_unfinished_runs_are_marked_failed_after_restart(app):
    with closing(connect(database_path(app))) as db, db:
        db.execute(
            "INSERT INTO project (name, created_at, updated_at) VALUES ('Старий', ?, ?)", (timestamp(), timestamp())
        )
        db.execute(
            "INSERT INTO run (project_id, status, options, created_at) VALUES (1, 'running', '{}', ?)", (timestamp(),)
        )
    restarted = create_app({"TESTING": True}, instance_path=app.instance_path)
    with closing(connect(database_path(restarted))) as db:
        run = RunRepository(db).get(1)
    restarted.extensions["run_queue"].shutdown()
    assert run.status.value == "failed"
    assert run.error == INTERRUPTED


def test_runs_and_projects_are_deleted_with_their_files(app, client):
    project_id = project_with_sources(client)
    run_id = start_run(client, project_id)
    finished(app, client, run_id)
    instance = Path(app.instance_path)
    assert (instance / "runs" / str(run_id) / "consolidated.csv").is_file()
    post(client, f"/runs/{run_id}/delete")
    assert not (instance / "runs" / str(run_id)).exists()
    response = post(client, f"/projects/{project_id}/delete", follow_redirects=True)
    assert "видалено разом із файлами й звітами" in text(response)
    assert list((instance / "uploads").iterdir()) == []
    assert query(app, "SELECT count(*) AS count FROM source")[0]["count"] == 0


def test_run_options_round_trip_and_validation():
    form = {**SETTINGS, "imputation": "knn", "neighbours": "7", "blocking": "sorted", "window": "4"}
    options = RunOptions.from_form({**form, "phone_region": "us", "false_match_rate": "2,5"})
    assert options.imputation is ImputationMode.KNN
    assert options.phone_region == "US"
    assert RunOptions.from_dict(options.to_dict()) == options
    assert options.imputer().neighbours == 7
    assert options.blocker().window == 4
    assert options.linkage_settings().decision.false_match_rate == pytest.approx(0.025)
    assert isinstance(RunOptions().imputer(), AdaptiveImputer)
    assert RunOptions().blocker().neighbours is None
    assert isinstance(options.linkage_settings().scorer, CoTraining)
    plain = RunOptions.from_form({**SETTINGS, "model": "fellegi_sunter"})
    assert isinstance(plain.linkage_settings().scorer, FellegiSunterScorer)
    assert RunOptions.from_dict(plain.to_dict()).model is ModelMode.FELLEGI_SUNTER
    assert RunOptions.from_form({"duplicates": "none"}).linkage_settings() is None
    assert RunOptions.from_dict({"duplicates": "unknown"}) == RunOptions()
    with pytest.raises(OptionsError) as error:
        RunOptions.from_form({"phone_region": "XX", "neighbours": "1.5"})
    assert len(error.value.messages) == 2


def test_numbers_are_formatted_for_ukrainian_readers():
    assert number(1234567) == "1\u00a0234\u00a0567"
    assert number("0.2583") == "0,2583"
    assert number(0.25834, 3) == "0,258"
    assert percent(0.8525, 2) == "85,25\u00a0%"
    assert points(0.1412) == "+14,1\u00a0п.\u00a0п."
    assert duration(75) == "1\u00a0хв 15\u00a0с"
    assert file_size(2048) == "2,0\u00a0КБ"


CHICAGO_FILES = [DATA / relative for relative in sorted(CHICAGO_SOURCE_CHECKSUMS)]


@pytest.mark.skipif(not all(path.is_file() for path in CHICAGO_FILES), reason="немає підготовлених джерел Chicago")
def test_chicago_sources_in_four_formats_are_consolidated(app, client):
    project_id = create_project(client, "Chicago")
    upload(client, project_id, {path.name: path.read_bytes() for path in CHICAGO_FILES})
    sources = query(
        app,
        "SELECT format, count(*) AS count FROM source JOIN source_table ON source_table.id = source.table_id "
        "GROUP BY format ORDER BY format",
    )
    assert {row["format"]: row["count"] for row in sources} == {"CSV": 2, "JSON": 2, "SQLite": 3, "XLSX": 2}
    run_id = start_run(client, project_id, duplicates="everywhere", phone_format="")
    assert finished(app, client, run_id)["status"] == "finished"
    report = query(app, "SELECT record_count, entity_count, index_before, index_after FROM report")[0]
    assert report["record_count"] == 3337
    assert report["entity_count"] < 3337
    assert report["index_after"] > report["index_before"]
    assert "Сутностей після злиття" in text(client.get(f"/runs/{run_id}"))


def test_one_layout_is_applied_to_all_tables(app, client):
    files = {"a.csv": b"id,name\n1,x\n2,y\n", "b.csv": b"id,name\n1,x\n3,z\n", "c.csv": b"name\nq\n"}
    project_id = project_with_sources(client, files)
    url = f"/projects/{project_id}/tables"
    response = post(client, url, {"key_column": "column:id", "split_column": "keep"}, follow_redirects=True)
    page = text(response)
    assert "Структуру оновлено для таблиць: 2." in page
    assert "Без змін, бо в таблиці немає вибраної колонки: c." in page
    assert '<option value="keep" selected>різні — не змінювати</option>' in page
    keys = [row["key_column"] for row in query(app, "SELECT key_column FROM source_table ORDER BY id")]
    assert keys == ["id", "id", None]
    post(client, url, {"key_column": "none", "split_column": "keep"})
    keys = [row["key_column"] for row in query(app, "SELECT key_column FROM source_table ORDER BY id")]
    assert keys == [None, None, None]
    response = post(client, url, {"key_column": "column:name", "split_column": "column:name"}, follow_redirects=True)
    assert "Одна колонка не може бути одночасно ідентифікатором запису і ознакою джерела." in text(response)
