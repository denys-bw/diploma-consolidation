from __future__ import annotations

import csv
import io
from collections import Counter
from dataclasses import dataclass

import pandas as pd

from consolidator.io.encoding import decode_bytes
from consolidator.io.formats import RawFile, ReadOptions, TableFormat
from consolidator.io.names import stem_of, unique_column_names
from consolidator.io.source import Source

DELIMITERS = (",", ";", "\t", "|")
SUFFIX_DELIMITERS = {".tsv": "\t", ".tab": "\t", ".psv": "|"}
SAMPLE_CHARACTERS = 200_000
SAMPLE_LINES = 200
QUOTING_NAMES = {csv.QUOTE_MINIMAL: "standard", csv.QUOTE_NONE: "none"}

csv.field_size_limit(2**31 - 1)


class DelimitedFormat(TableFormat):
    suffixes = (".csv", ".txt", ".tsv", ".tab", ".psv")

    def read(self, file: RawFile, options: ReadOptions) -> list[Source]:
        decoded = decode_bytes(file.data)
        table = parse_delimited(decoded.text, options.delimiter or SUFFIX_DELIMITERS.get(file.suffix))
        details = {"encoding": decoded.encoding, "delimiter": table.delimiter, "quoting": table.quoting}
        label = "TSV" if table.delimiter == "\t" else "CSV"
        return [Source(stem_of(file.name), table.frame, file.origin, label, details)]


@dataclass(frozen=True, eq=False)
class DelimitedTable:
    frame: pd.DataFrame
    delimiter: str
    quoting: str


def parse_delimited(text: str, delimiter: str | None = None) -> DelimitedTable:
    delimiter = delimiter or detect_delimiter(text)
    modes = [csv.QUOTE_MINIMAL, csv.QUOTE_NONE] if '"' in text else [csv.QUOTE_MINIMAL]
    parsed = [(quoting, _read_rows(text, delimiter, quoting)) for quoting in modes]
    quoting, rows = min(parsed, key=lambda item: (_count_irregular(item[1]), -len(item[1]), modes.index(item[0])))
    return DelimitedTable(_build_frame(rows), delimiter, QUOTING_NAMES[quoting])


def detect_delimiter(text: str) -> str:
    sample = "\n".join(text[:SAMPLE_CHARACTERS].splitlines()[:SAMPLE_LINES])
    best, best_score = ",", (0.0, 0)
    for delimiter in DELIMITERS:
        widths = [len(row) for row in csv.reader(io.StringIO(sample, newline=""), delimiter=delimiter) if row]
        if not widths:
            continue
        width, count = Counter(widths).most_common(1)[0]
        score = (count / len(widths), width)
        if width > 1 and score > best_score:
            best, best_score = delimiter, score
    return best


def _read_rows(text: str, delimiter: str, quoting: int) -> list[list[str]]:
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, quoting=quoting)
    return [row for row in reader if any(field.strip() for field in row)]


def _count_irregular(rows: list[list[str]]) -> int:
    if not rows:
        return 0
    width = len(rows[0])
    return sum(_is_irregular(row, width) for row in rows[1:])


def _is_irregular(row: list[str], width: int) -> bool:
    if len(row) < width:
        return True
    return any(row[width:])


def _build_frame(rows: list[list[str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    header, body = rows[0], rows[1:]
    width = len(header)
    if body:
        width = max(width, Counter(len(row) for row in body).most_common(1)[0][0])
    columns = unique_column_names(header + [""] * (width - len(header)))
    fitted = []
    for number, row in enumerate(body, start=1):
        if len(row) > width:
            if any(row[width:]):
                raise ValueError(f"Запис {number} має {len(row)} полів, а заголовок — {width}")
            row = row[:width]
        fitted.append(row + [""] * (width - len(row)))
    return pd.DataFrame(fitted, columns=columns, dtype=object)
