from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

from werkzeug.datastructures import FileStorage

from consolidator.io import Source, read_bytes
from consolidator.profiling import profile_source
from consolidator.reporting import profile_table
from webapp.models import SourceDraft, SourceInfo, SourceTable, TableDraft, UploadedFile
from webapp.repositories import Repositories
from webapp.storage import Storage

logger = logging.getLogger(__name__)


class SourceError(ValueError):
    pass


class Keep(Enum):
    CURRENT = "current"


@dataclass(frozen=True)
class UploadOutcome:
    added: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LayoutOutcome:
    updated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class SourceService:
    def __init__(self, repositories: Repositories, storage: Storage) -> None:
        self.repositories = repositories
        self.storage = storage

    def add_uploads(self, project_id: int, uploads: Iterable[FileStorage]) -> UploadOutcome:
        outcome = UploadOutcome()
        taken = self.repositories.sources.names(project_id)
        for upload in uploads:
            filename = (upload.filename or "").strip()
            if not filename:
                continue
            data = upload.read()
            try:
                tables = read_bytes(data, filename)
            except Exception as error:
                logger.warning("Не вдалося прочитати файл %s", filename, exc_info=True)
                outcome.errors.append(f"«{filename}»: {_reason(error)}")
                continue
            filled = [(position, table) for position, table in enumerate(tables) if table.n_records and table.columns]
            if not filled:
                outcome.errors.append(f"«{filename}»: у файлі не знайдено жодної таблиці із записами.")
                continue
            file_id = self.repositories.sources.add_file(project_id, filename, self.storage.save_upload(data, filename))
            for position, table in filled:
                draft = TableDraft(position, table.name, table.format, tuple(table.columns), _details(table))
                table_id = self.repositories.sources.add_table(file_id, draft)
                drafts = _drafts(derive(table, None, None), taken)
                self.repositories.sources.replace_sources(project_id, table_id, drafts)
                taken.update(source.name for source in drafts)
                outcome.added.extend(source.name for source in drafts)
        return outcome

    def configure(
        self, project_id: int, table: SourceTable, key_column: str | None, split_column: str | None
    ) -> list[str]:
        if key_column and key_column == split_column:
            raise SourceError("Одна колонка не може бути одночасно ідентифікатором запису і ознакою джерела.")
        unknown = [column for column in (key_column, split_column) if column and column not in table.columns]
        if unknown:
            raise SourceError(f"У таблиці «{table.name}» немає колонки «{unknown[0]}».")
        parts = derive(self._table(project_id, table), key_column, split_column)
        own = {source.name for source in table.sources}
        drafts = _drafts(parts, self.repositories.sources.names(project_id) - own)
        self.repositories.sources.set_table_options(table.id, key_column, split_column)
        self.repositories.sources.replace_sources(project_id, table.id, drafts)
        return [draft.name for draft in drafts]

    def configure_all(
        self, project_id: int, key_column: str | None | Keep, split_column: str | None | Keep
    ) -> LayoutOutcome:
        outcome = LayoutOutcome()
        for table in self.repositories.sources.tables(project_id):
            key = table.key_column if key_column is Keep.CURRENT else key_column
            split = table.split_column if split_column is Keep.CURRENT else split_column
            if (key, split) == (table.key_column, table.split_column):
                continue
            if any(column and column not in table.columns for column in (key, split)):
                outcome.skipped.append(table.name)
                continue
            try:
                self.configure(project_id, table, key, split)
            except SourceError as error:
                outcome.errors.append(f"«{table.name}»: {error}")
                continue
            outcome.updated.append(table.name)
        return outcome

    def load_sources(self, project_id: int) -> list[Source]:
        sources = []
        for uploaded in self.repositories.sources.files(project_id):
            tables = [table for table in uploaded.tables if any(info.included for info in table.sources)]
            if not tables:
                continue
            parsed = self._parse(uploaded)
            for table in tables:
                parts = dict(derive(parsed[table.position].renamed(table.name), table.key_column, table.split_column))
                sources.extend(parts[info.part].renamed(info.name) for info in table.sources if info.included)
        return sources

    def load_source(self, project_id: int, info: SourceInfo) -> Source:
        table = self.repositories.sources.table(project_id, info.table_id)
        parts = dict(derive(self._table(project_id, table), table.key_column, table.split_column))
        return parts[info.part].renamed(info.name)

    def remove_file(self, uploaded: UploadedFile) -> None:
        self.repositories.sources.delete_file(uploaded.id)
        self.storage.delete_upload(uploaded.stored_name)

    def toggle(self, info: SourceInfo) -> None:
        self.repositories.sources.set_included(info.id, not info.included)

    def _table(self, project_id: int, table: SourceTable) -> Source:
        uploaded = self.repositories.sources.file(project_id, table.file_id)
        if uploaded is None:
            raise SourceError(f"Файл таблиці «{table.name}» не знайдено.")
        return self._parse(uploaded)[table.position].renamed(table.name)

    def _parse(self, uploaded: UploadedFile) -> list[Source]:
        return read_bytes(self.storage.read_upload(uploaded.stored_name), uploaded.original_name)


def derive(table: Source, key_column: str | None, split_column: str | None) -> list[tuple[str, Source]]:
    try:
        prepared = table if key_column else table.numbered()
        parts = prepared.split_by(split_column) if split_column else [prepared]
        keyed = [part.keyed_by(key_column) for part in parts] if key_column else parts
    except (KeyError, ValueError) as error:
        raise SourceError(str(error).strip("'\"")) from error
    return [(part.name if split_column else "", part) for part in keyed]


def _drafts(parts: list[tuple[str, Source]], taken: set[str]) -> list[SourceDraft]:
    drafts = []
    names = set(taken)
    for part, source in parts:
        name = _unique(source.name, names)
        names.add(name)
        profile = profile_source(source)
        drafts.append(
            SourceDraft(
                part=part,
                name=name,
                record_count=source.n_records,
                column_count=len(source.columns),
                completeness=profile.completeness,
                profile=profile_table(profile).to_dict(orient="records"),
            )
        )
    return drafts


def _details(table: Source) -> dict[str, str]:
    return {key: str(value) for key, value in table.details.items()}


def _unique(base: str, taken: set[str]) -> str:
    name, suffix = base, 2
    while name in taken:
        name, suffix = f"{base} ({suffix})", suffix + 1
    return name


def _reason(error: Exception) -> str:
    if isinstance(error, ValueError):
        return str(error)
    return f"файл пошкоджений або має інший формат, ніж вказує розширення ({type(error).__name__}: {error})"
