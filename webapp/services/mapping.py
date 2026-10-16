from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace

from consolidator.profiling import ColumnType
from consolidator.schema import Attribute, ColumnRef, SchemaMapping, match_sources
from webapp.models import AttributeRole, AttributeState, SourceInfo
from webapp.repositories import Repositories
from webapp.services.sources import SourceService

STALE = "Зіставлення змінилося в іншому вікні. Оновіть сторінку й повторіть зміни."


class MappingError(ValueError):
    pass


@dataclass(frozen=True)
class AttributeEdit:
    current: str
    name: str
    type: ColumnType
    role: AttributeRole


class MappingService:
    def __init__(self, repositories: Repositories, sources: SourceService) -> None:
        self.repositories = repositories
        self.sources = sources

    def rebuild(self, project_id: int, alpha: float) -> int:
        infos = self.repositories.sources.sources(project_id, included_only=True)
        states = []
        if infos:
            mapping = match_sources(self.sources.load_sources(project_id), alpha)
            states = _states(mapping, _ids(infos), {})
        self.repositories.attributes.replace(project_id, states)
        return len(states)

    def mapping(self, project_id: int) -> SchemaMapping:
        infos = self.repositories.sources.sources(project_id, included_only=True)
        return to_mapping(self.repositories.attributes.load(project_id), infos, {AttributeRole.ATTRIBUTE})

    def reference(self, project_id: int) -> Attribute | None:
        infos = self.repositories.sources.sources(project_id, included_only=True)
        found = to_mapping(self.repositories.attributes.load(project_id), infos, {AttributeRole.REFERENCE})
        return found.attributes[0] if found.attributes else None

    def save_edits(self, project_id: int, edits: Sequence[AttributeEdit]) -> list[AttributeState]:
        states = {state.name: state for state in self.repositories.attributes.load(project_id)}
        if len(edits) != len(states) or any(edit.current not in states for edit in edits):
            raise MappingError(STALE)
        names = [edit.name.strip() for edit in edits]
        if not all(names):
            raise MappingError("Назва атрибута не може бути порожньою.")
        repeated = [name for name, count in Counter(names).items() if count > 1]
        if repeated:
            raise MappingError(f"Назви атрибутів мають бути різними: «{repeated[0]}» трапляється кілька разів.")
        if sum(edit.role is AttributeRole.REFERENCE for edit in edits) > 1:
            raise MappingError("Еталоном для оцінювання може бути лише один атрибут.")
        updated = [_edited(states[edit.current], edit, name) for edit, name in zip(edits, names)]
        self.repositories.attributes.replace(project_id, updated)
        return updated

    def merge(self, project_id: int, names: Collection[str]) -> str:
        if len(names) < 2:
            raise MappingError("Позначте щонайменше два атрибути, які треба об'єднати.")
        infos, states, mapping = self._current(project_id)
        merged = mapping.merged([state.name for state in states if state.name in names])
        self.repositories.attributes.replace(project_id, _states(merged, _ids(infos), _by_name(states)))
        return next(state.name for state in states if state.name in names)

    def detach(self, project_id: int, name: str, source_id: int) -> str:
        infos, states, mapping = self._current(project_id)
        source = next((info for info in infos if info.id == source_id), None)
        attribute = next((item for item in mapping.attributes if item.name == name), None)
        if source is None or attribute is None or attribute.column_in(source.name) is None:
            raise MappingError("Колонку не знайдено: оновіть сторінку.")
        if len(attribute.columns) == 1:
            raise MappingError(f"Атрибут «{name}» має лише одну колонку — від'єднувати нічого.")
        column = attribute.column_in(source.name)
        new_name = _unique(column, set(mapping.names))
        detached = mapping.detached(name, ColumnRef(source.name, column), new_name)
        self.repositories.attributes.replace(project_id, _states(detached, _ids(infos), _by_name(states)))
        return new_name

    def _current(self, project_id: int) -> tuple[list[SourceInfo], list[AttributeState], SchemaMapping]:
        infos = self.repositories.sources.sources(project_id, included_only=True)
        states = self.repositories.attributes.load(project_id)
        return infos, states, to_mapping(states, infos, set(AttributeRole))


def to_mapping(
    states: Sequence[AttributeState], infos: Sequence[SourceInfo], roles: Collection[AttributeRole]
) -> SchemaMapping:
    names = {info.id: info.name for info in infos}
    attributes = []
    for state in states:
        columns = tuple(ColumnRef(names[owner], column) for owner, column in state.columns if owner in names)
        if columns and state.role in roles:
            attributes.append(Attribute(state.name, state.type, columns, state.confidence))
    return SchemaMapping(tuple(attributes))


def _edited(state: AttributeState, edit: AttributeEdit, name: str) -> AttributeState:
    resolution = state.resolution if edit.type is state.type else None
    return replace(state, name=name, type=edit.type, role=edit.role, resolution=resolution)


def _states(
    mapping: SchemaMapping, ids: Mapping[str, int], previous: Mapping[str, AttributeState]
) -> list[AttributeState]:
    states = []
    for attribute in mapping.attributes:
        old = previous.get(attribute.name)
        same = old is not None and old.type is attribute.type
        states.append(
            AttributeState(
                name=attribute.name,
                type=attribute.type,
                confidence=attribute.confidence,
                role=old.role if same else AttributeRole.ATTRIBUTE,
                resolution=old.resolution if same else None,
                columns=tuple((ids[ref.source], ref.column) for ref in attribute.columns),
            )
        )
    return states


def _ids(infos: Sequence[SourceInfo]) -> dict[str, int]:
    return {info.name: info.id for info in infos}


def _by_name(states: Sequence[AttributeState]) -> dict[str, AttributeState]:
    return {state.name: state for state in states}


def _unique(name: str, taken: set[str]) -> str:
    candidate, suffix = name, 2
    while candidate in taken:
        candidate, suffix = f"{name}_{suffix}", suffix + 1
    return candidate
