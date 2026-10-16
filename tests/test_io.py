import gzip
import io
import json
import sqlite3
import zipfile

import pytest
from openpyxl import Workbook

from consolidator.io import ReadOptions, UnsupportedFormatError, read_bytes, read_file, read_files
from consolidator.io.delimited import detect_delimiter, parse_delimited
from consolidator.io.encoding import decode_bytes, guess_single_byte_encoding
from consolidator.io.names import stem_of, unique_column_names


def test_utf8_with_bom_is_recognised():
    decoded = decode_bytes("﻿назва;місто\n".encode())
    assert decoded.encoding == "utf-8-sig"
    assert decoded.text.startswith("назва")


def test_cp1251_is_told_apart_from_cp1252():
    ukrainian = "Київ, вулиця Хрещатик, будинок 22".encode("cp1251")
    western = "Müller, Straße 5, Köln".encode("cp1252")
    assert guess_single_byte_encoding(ukrainian) == "cp1251"
    assert guess_single_byte_encoding(western) == "cp1252"
    assert decode_bytes(ukrainian).text == "Київ, вулиця Хрещатик, будинок 22"


def test_mixed_utf8_and_single_byte_lines_are_decoded_line_by_line():
    data = "name\nSjöström\n".encode() + "Wäsch\n".encode("cp1252")
    decoded = decode_bytes(data)
    assert decoded.encoding == "utf-8+cp1252"
    assert decoded.text.splitlines() == ["name", "Sjöström", "Wäsch"]


@pytest.mark.parametrize(
    ("text", "delimiter"),
    [
        ("a,b,c\n1,2,3\n4,5,6\n", ","),
        ("a;b;c\n1,5;2;3\n4;5,5;6\n", ";"),
        ("a\tb\n1\t2\n", "\t"),
        ("a|b|c\nx|y|z\n", "|"),
    ],
)
def test_delimiter_detection(text, delimiter):
    assert detect_delimiter(text) == delimiter


def test_quoted_fields_with_line_breaks_are_kept_whole():
    table = parse_delimited('id,name\n1,"Chicago Commons\nAssociation"\n2,"Smith, John"\n')
    assert table.frame.shape == (2, 2)
    assert table.frame.loc[0, "name"] == "Chicago Commons\nAssociation"
    assert table.frame.loc[1, "name"] == "Smith, John"


def test_unbalanced_quotes_fall_back_to_literal_quotes():
    text = 'id|title\n1|"Gone Fishin\' part one\n2|plain\n3|"quoted" start\n'
    table = parse_delimited(text)
    assert table.quoting == "none"
    assert table.frame["title"].tolist() == ["\"Gone Fishin' part one", "plain", '"quoted" start']


def test_values_stay_text_and_empty_cells_stay_empty():
    frame = parse_delimited("zip,phone,price\n00123,0671234567,\n").frame
    assert frame.iloc[0].tolist() == ["00123", "0671234567", ""]


def test_column_names_are_made_unique():
    assert unique_column_names(["name", "", "name", " city ", None]) == [
        "name",
        "column_2",
        "name_2",
        "city",
        "column_5",
    ]


def test_stem_of_handles_double_extensions():
    assert stem_of("data/offers.json.gz") == "offers"
    assert stem_of("Abt.csv") == "Abt"


def test_json_array_with_nested_objects_and_lists():
    document = [
        {"id": 1, "name": "Disc", "artist": {"name": "Band", "country": "UA"}, "tracks": ["One", "Two"]},
        {"id": 2, "name": None, "available": True, "tracks": []},
    ]
    source = read_bytes(json.dumps(document).encode("utf-8"), "discs.json")[0]
    frame = source.frame
    assert list(frame.columns) == ["id", "name", "artist.name", "artist.country", "tracks", "available"]
    assert frame.loc[0, "tracks"] == "One | Two"
    assert frame.loc[1, "name"] == ""
    assert frame.loc[1, "available"] == "true"
    assert source.format == "JSON"


def test_json_object_with_record_list_inside():
    document = {"meta": {"count": 2}, "data": {"items": [{"a": 1}, {"a": 2}]}}
    frame = read_bytes(json.dumps(document).encode(), "export.json")[0].frame
    assert frame["a"].tolist() == ["1", "2"]


def test_json_lines_are_detected_inside_json_file():
    data = b'{"id": 1, "title": "A"}\n{"id": 2, "title": "B"}\n'
    source = read_bytes(data, "offers.json")[0]
    assert source.format == "JSON Lines"
    assert source.frame["title"].tolist() == ["A", "B"]


def test_nested_xml_records_with_attributes_and_repeated_elements():
    xml = b"""<?xml version="1.0"?>
    <discs>
      <disc id="7"><artist>Band</artist><dtitle>Long ti</dtitle><dtitle>tle</dtitle>
        <tracks><title> One </title><title>Two</title></tracks></disc>
      <disc id="8"><artist>Other</artist><dtitle>Short</dtitle><tracks><title>Solo</title></tracks></disc>
    </discs>"""
    joined = read_bytes(xml, "cd.xml")[0].frame
    assert joined.loc[0, "id"] == "7"
    assert joined.loc[0, "dtitle"] == "Long ti | tle"
    assert joined.loc[0, "tracks.title"] == "One | Two"
    concatenated = read_bytes(xml, "cd.xml", options=ReadOptions(concatenate=frozenset({"dtitle"})))[0].frame
    assert concatenated.loc[0, "dtitle"] == "Long title"


def test_xlsx_sheets_become_sources(tmp_path):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Clients"
    sheet.append(["name", "zip", "amount"])
    sheet.append(["Ольга", 1234, 12.5])
    sheet.append(["Петро", None, 3.0])
    second = workbook.create_sheet("Empty")
    second.append([None])
    path = tmp_path / "clients.xlsx"
    workbook.save(path)
    sources = read_file(path)
    assert [source.name for source in sources] == ["clients"]
    assert sources[0].frame.values.tolist() == [["Ольга", "1234", "12.5"], ["Петро", "", "3"]]


def test_sqlite_tables_become_sources(tmp_path):
    path = tmp_path / "registry.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE firms (code TEXT, name TEXT)")
        connection.execute("CREATE TABLE people (name TEXT, age INTEGER)")
        connection.execute("INSERT INTO firms VALUES ('00123456', 'ТОВ Ромашка')")
        connection.execute("INSERT INTO people VALUES ('Іван', 30), ('Олена', NULL)")
    connection.close()
    sources = {source.name: source for source in read_file(path)}
    assert set(sources) == {"firms", "people"}
    assert sources["firms"].frame.iloc[0].tolist() == ["00123456", "ТОВ Ромашка"]
    assert sources["people"].frame["age"].tolist() == ["30", ""]
    from_bytes = read_bytes(path.read_bytes(), "copy.db")
    assert {source.name for source in from_bytes} == {"firms", "people"}


def test_zip_archive_with_several_formats():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("a.csv", "id,name\n1,A\n")
        archive.writestr("nested/b.json", '[{"id": 2}]')
        archive.writestr("__MACOSX/a.csv", "ignored")
        archive.writestr("readme.md", "ignored")
    sources = read_bytes(buffer.getvalue(), "bundle.zip")
    assert [source.name for source in sources] == ["a", "b"]
    assert sources[0].origin == "bundle.zip/a.csv"


def test_gzip_is_unpacked_by_inner_extension():
    sources = read_bytes(gzip.compress(b'{"x": 1}\n{"x": 2}\n'), "part.json.gz")
    assert sources[0].frame["x"].tolist() == ["1", "2"]


def test_unsupported_format_has_a_hint():
    with pytest.raises(UnsupportedFormatError, match="xlsx"):
        read_bytes(b"\xd0\xcf\x11\xe0", "old.xls")


def test_duplicate_source_names_are_numbered(tmp_path):
    first = tmp_path / "a" / "data.csv"
    second = tmp_path / "b" / "data.csv"
    for path in (first, second):
        path.parent.mkdir()
        path.write_text("x\n1\n", encoding="utf-8")
    assert [source.name for source in read_files([first, second])] == ["data", "data (2)"]


def test_keyed_by_rejects_duplicates():
    source = read_bytes(b"id,x\n1,a\n1,b\n", "t.csv")[0]
    with pytest.raises(ValueError, match="ключем"):
        source.keyed_by("id")
    with pytest.raises(ValueError, match="порожні значення"):
        read_bytes(b"id,x\n1,a\n,b\n", "t.csv")[0].keyed_by("id")
    keyed = read_bytes(b"id,x\n1,a\n2,b\n", "t.csv")[0].keyed_by("id")
    assert keyed.frame.index.tolist() == ["1", "2"]
    assert keyed.columns == ["x"]


def test_table_is_split_into_sources_by_column():
    table = read_bytes(b"src,flight,dep\naa,AA-3859,7:10 a.m.\nua,AA-3859,7:10 a.m.\naa,AA-1733,7:45 p.m.\n", "t.csv")[
        0
    ]
    parts = table.split_by("src")
    assert [part.name for part in parts] == ["aa", "ua"]
    assert parts[0].frame.index.tolist() == [0, 2]
    assert parts[0].columns == ["flight", "dep"]
    assert parts[1].details["split_by"] == "src"
    with pytest.raises(KeyError):
        table.split_by("missing")
    gaps = read_bytes(b"src,x\naa,1\n,2\n", "g.csv")[0]
    with pytest.raises(ValueError):
        gaps.split_by("src")


def test_split_sources_keep_only_columns_they_fill():
    table = read_bytes(b"src,flight,dep\naa,AA-3859,\nua,AA-3859,7:10 a.m.\naa,AA-1733, \n", "t.csv")[0]
    aa, ua = table.split_by("src")
    assert aa.columns == ["flight"]
    assert ua.columns == ["flight", "dep"]
