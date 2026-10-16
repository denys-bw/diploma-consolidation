from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from consolidator.io.archives import GzipFormat, ZipFormat
from consolidator.io.database import SqliteFormat
from consolidator.io.delimited import DelimitedFormat
from consolidator.io.documents import JsonFormat, JsonLinesFormat, XmlFormat
from consolidator.io.formats import ArchiveFormat, FileFormat, RawFile, ReadOptions, suffix_of
from consolidator.io.names import unique_source_names
from consolidator.io.source import Source
from consolidator.io.spreadsheet import SpreadsheetFormat

FORMATS: tuple[FileFormat, ...] = (
    GzipFormat(),
    ZipFormat(),
    SqliteFormat(),
    SpreadsheetFormat(),
    XmlFormat(),
    JsonFormat(),
    JsonLinesFormat(),
    DelimitedFormat(),
)
SUPPORTED_SUFFIXES = frozenset(suffix for file_format in FORMATS for suffix in file_format.suffixes)
UNSUPPORTED_HINTS = {
    ".xls": "збережіть файл у форматі .xlsx",
    ".ods": "збережіть файл у форматі .xlsx",
    ".rdata": "перетворіть файл на CSV",
    ".rds": "перетворіть файл на CSV",
}
DEFAULT_OPTIONS = ReadOptions()


class UnsupportedFormatError(ValueError):
    pass


def read_files(paths: Iterable[str | Path], options: ReadOptions = DEFAULT_OPTIONS) -> list[Source]:
    return with_unique_names([source for path in paths for source in read_file(path, options)])


def read_file(path: str | Path, options: ReadOptions = DEFAULT_OPTIONS) -> list[Source]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Файл не знайдено: {path}")
    return read_bytes(path.read_bytes(), path.name, str(path), options)


def read_bytes(
    data: bytes, filename: str, origin: str | None = None, options: ReadOptions = DEFAULT_OPTIONS
) -> list[Source]:
    return _read(RawFile(filename, data, origin or filename), options)


def is_supported(filename: str) -> bool:
    return suffix_of(filename) in SUPPORTED_SUFFIXES


def with_unique_names(sources: list[Source]) -> list[Source]:
    names = unique_source_names(source.name for source in sources)
    return [source if source.name == name else source.renamed(name) for source, name in zip(sources, names)]


def _read(file: RawFile, options: ReadOptions) -> list[Source]:
    file_format = _format_of(file)
    if not isinstance(file_format, ArchiveFormat):
        return file_format.read(file, options)
    members = [member for member in file_format.unpack(file) if is_supported(member.name)]
    sources = [source for member in members for source in _read(member, options)]
    if not sources:
        raise UnsupportedFormatError(f"В архіві «{file.origin}» немає файлів підтримуваних форматів")
    return with_unique_names(sources)


def _format_of(file: RawFile) -> FileFormat:
    for file_format in FORMATS:
        if file_format.recognizes(file):
            return file_format
    hint = UNSUPPORTED_HINTS.get(file.suffix)
    raise UnsupportedFormatError(f"Формат файлу «{file.name}» не підтримується" + (f": {hint}" if hint else ""))
