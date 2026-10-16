from __future__ import annotations

from consolidator.profiling.types import ColumnType

FAMILIES = {
    ColumnType.TEXT: "text",
    ColumnType.CATEGORY: "text",
    ColumnType.NUMBER: "number",
    ColumnType.MEASURE: "number",
    ColumnType.YEAR: "date",
    ColumnType.DATE: "date",
    ColumnType.DATETIME: "date",
    ColumnType.IDENTIFIER: "key",
    ColumnType.CODE: "key",
    ColumnType.TIME: "time",
    ColumnType.PHONE: "phone",
    ColumnType.EMAIL: "email",
    ColumnType.URL: "url",
    ColumnType.BOOLEAN: "boolean",
}
FORMAT_FAMILIES = frozenset({"phone", "email", "url"})
TEXT_FAMILY = "text"
RELATED = frozenset({frozenset({"key", "number"}), frozenset({"key", "text"}), frozenset({"date", "number"})})


def family(column_type: ColumnType) -> str:
    return FAMILIES.get(column_type, "empty")


def compatible(left: ColumnType, right: ColumnType) -> bool:
    a, b = family(left), family(right)
    if "empty" in (a, b):
        return False
    return a == b or frozenset({a, b}) in RELATED
