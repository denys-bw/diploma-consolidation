from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter, deque
from collections.abc import Collection, Iterable, Mapping
from typing import Any

import pandas as pd

from consolidator.io.encoding import decode_bytes
from consolidator.io.formats import RawFile, ReadOptions, TableFormat
from consolidator.io.names import stem_of, unique_column_names
from consolidator.io.source import Source

LIST_SEPARATOR = " | "
NESTING_SEPARATOR = "."
MAX_SEARCH_DEPTH = 3


class JsonFormat(TableFormat):
    suffixes = (".json",)

    def read(self, file: RawFile, options: ReadOptions) -> list[Source]:
        decoded = decode_bytes(file.data)
        frame, label = parse_json(decoded.text)
        return [Source(stem_of(file.name), frame, file.origin, label, {"encoding": decoded.encoding})]


class JsonLinesFormat(TableFormat):
    suffixes = (".jsonl", ".ndjson")

    def read(self, file: RawFile, options: ReadOptions) -> list[Source]:
        decoded = decode_bytes(file.data)
        frame = parse_json_lines(decoded.text)
        return [Source(stem_of(file.name), frame, file.origin, "JSON Lines", {"encoding": decoded.encoding})]


class XmlFormat(TableFormat):
    suffixes = (".xml",)

    def read(self, file: RawFile, options: ReadOptions) -> list[Source]:
        frame = parse_xml(file.data, options.concatenate)
        return [Source(stem_of(file.name), frame, file.origin, "XML")]


def parse_json(text: str) -> tuple[pd.DataFrame, str]:
    try:
        document: Any = json.loads(text)
    except json.JSONDecodeError as error:
        if error.msg != "Extra data":
            raise ValueError(f"Некоректний JSON: {error.msg} (рядок {error.lineno}, позиція {error.colno})") from error
        return parse_json_lines(text), "JSON Lines"
    return records_to_frame(_json_records(document)), "JSON"


def parse_json_lines(text: str) -> pd.DataFrame:
    records: list[Any] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Некоректний JSON у рядку {number}: {error.msg}") from error
    return records_to_frame(_json_records(records))


def parse_xml(data: bytes, concatenate: Collection[str] = ()) -> pd.DataFrame:
    try:
        root = ET.fromstring(data)
    except ET.ParseError as error:
        raise ValueError(f"Некоректний XML: {error}") from error
    container, tag = _record_container(root)
    tags = frozenset(concatenate)
    return records_to_frame(_element_record(element, tags) for element in container if _local(element.tag) == tag)


def records_to_frame(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    rows = [flatten(record) for record in records]
    columns: dict[str, None] = {}
    for row in rows:
        columns.update(dict.fromkeys(row))
    names = list(columns)
    frame = pd.DataFrame([[row.get(name, "") for name in names] for row in rows], columns=names, dtype=object)
    frame.columns = unique_column_names(frame.columns)
    return frame


def flatten(record: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    flat: dict[str, str] = {}
    for key, value in record.items():
        name = f"{prefix}{NESTING_SEPARATOR}{key}" if prefix else str(key)
        flat.update(_flatten_value(value, name))
    return flat


def scalar_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _flatten_value(value: Any, name: str) -> dict[str, str]:
    if isinstance(value, Mapping):
        return flatten(value, name)
    if isinstance(value, list):
        return _flatten_list(value, name)
    return {name: scalar_text(value)}


def _flatten_list(values: list[Any], name: str) -> dict[str, str]:
    if all(not isinstance(value, (Mapping, list)) for value in values):
        return {name: LIST_SEPARATOR.join(scalar_text(value).strip() for value in values)}
    return _join_items([_flatten_value(value, name) for value in values])


def _join_items(items: list[dict[str, str]]) -> dict[str, str]:
    keys: dict[str, None] = {}
    for item in items:
        keys.update(dict.fromkeys(item))
    if len(items) == 1:
        return {key: items[0].get(key, "") for key in keys}
    return {key: LIST_SEPARATOR.join(item.get(key, "").strip() for item in items) for key in keys}


def _json_records(document: Any) -> list[Mapping[str, Any]]:
    if isinstance(document, list):
        return [item if isinstance(item, Mapping) else {"value": item} for item in document]
    if isinstance(document, Mapping):
        found = _largest_record_list(document, MAX_SEARCH_DEPTH)
        return found if found is not None else [document]
    return [{"value": document}]


def _largest_record_list(document: Mapping[str, Any], depth: int) -> list[Mapping[str, Any]] | None:
    best: list[Mapping[str, Any]] | None = None
    for value in document.values():
        candidate = None
        if isinstance(value, list) and value and all(isinstance(item, Mapping) for item in value):
            candidate = value
        elif isinstance(value, Mapping) and depth > 1:
            candidate = _largest_record_list(value, depth - 1)
        if candidate is not None and (best is None or len(candidate) > len(best)):
            best = candidate
    return best


def _local(tag: object) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _record_container(root: ET.Element) -> tuple[ET.Element, str]:
    best_count, best_element, best_tag = 0, root, ""
    queue: deque[tuple[ET.Element, int]] = deque([(root, 0)])
    while queue:
        element, depth = queue.popleft()
        children = list(element)
        if not children:
            continue
        tag, count = Counter(_local(child.tag) for child in children).most_common(1)[0]
        if count > best_count:
            best_count, best_element, best_tag = count, element, tag
        if depth < MAX_SEARCH_DEPTH:
            queue.extend((child, depth + 1) for child in children)
    if best_count <= 1 and best_element is root:
        wrapper = ET.Element("records")
        wrapper.append(root)
        return wrapper, _local(root.tag)
    return best_element, best_tag


def _element_record(element: ET.Element, concatenate: frozenset[str], prefix: str = "") -> dict[str, str]:
    record = {f"{prefix}{_local(name)}": value for name, value in element.attrib.items()}
    children = list(element)
    if not children:
        text = element.text or ""
        if text.strip() or not record:
            record[prefix.rstrip(NESTING_SEPARATOR) or "value"] = text
        return record
    groups: dict[str, list[ET.Element]] = {}
    for child in children:
        groups.setdefault(_local(child.tag), []).append(child)
    for tag, members in groups.items():
        name = f"{prefix}{tag}"
        if all(len(member) == 0 and not member.attrib for member in members):
            record[name] = _join_texts([member.text or "" for member in members], tag in concatenate)
        else:
            record.update(
                _join_items([_element_record(member, concatenate, f"{name}{NESTING_SEPARATOR}") for member in members])
            )
    return record


def _join_texts(texts: list[str], concatenate: bool) -> str:
    if len(texts) == 1:
        return texts[0]
    if concatenate:
        return "".join(texts)
    return LIST_SEPARATOR.join(text.strip() for text in texts)
