from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

import pandas as pd

from consolidator.profiling.types import ColumnType

COMPOSITE_SEPARATOR = " + "
SPACES = r"\s{2,}"


@dataclass(frozen=True)
class ColumnRef:
    source: str
    column: str

    @classmethod
    def composite(cls, source: str, columns: Sequence[str]) -> ColumnRef:
        return cls(source, COMPOSITE_SEPARATOR.join(columns))

    def parts(self, available: Collection[str]) -> tuple[str, ...]:
        if self.column in available or COMPOSITE_SEPARATOR not in self.column:
            return (self.column,)
        return tuple(self.column.split(COMPOSITE_SEPARATOR))


@dataclass(frozen=True)
class Attribute:
    name: str
    type: ColumnType
    columns: tuple[ColumnRef, ...]
    confidence: float = 1.0

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(ref.source for ref in self.columns)

    def column_in(self, source: str) -> str | None:
        for ref in self.columns:
            if ref.source == source:
                return ref.column
        return None


@dataclass(frozen=True)
class SchemaMapping:
    attributes: tuple[Attribute, ...]

    @property
    def names(self) -> list[str]:
        return [attribute.name for attribute in self.attributes]

    @property
    def types(self) -> dict[str, ColumnType]:
        return {attribute.name: attribute.type for attribute in self.attributes}

    def attribute(self, name: str) -> Attribute:
        for attribute in self.attributes:
            if attribute.name == name:
                return attribute
        raise KeyError(f"Немає атрибута «{name}»")

    def apply(self, frames: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        result = {}
        for source, frame in frames.items():
            columns = {
                attribute.name: joined_values(frame, ref.parts(frame.columns))
                for attribute in self.attributes
                for ref in attribute.columns
                if ref.source == source
            }
            result[source] = pd.DataFrame(columns, index=frame.index)
        return result

    def renamed(self, name: str, new_name: str) -> SchemaMapping:
        if new_name in self.names and new_name != name:
            raise ValueError(f"Атрибут «{new_name}» уже існує")
        return self._replaced(name, lambda attribute: replace(attribute, name=new_name))

    def merged(self, names: Iterable[str], new_name: str | None = None) -> SchemaMapping:
        selected = [self.attribute(name) for name in names]
        if len(selected) < 2:
            return self
        grouped: dict[str, list[str]] = {}
        for ref in (ref for attribute in selected for ref in attribute.columns):
            grouped.setdefault(ref.source, []).append(ref.column)
        columns = tuple(ColumnRef.composite(source, names) for source, names in grouped.items())
        merged = Attribute(new_name or selected[0].name, selected[0].type, columns, 1.0)
        kept = [attribute for attribute in self.attributes if attribute not in selected]
        position = self.attributes.index(selected[0])
        kept.insert(min(position, len(kept)), merged)
        return SchemaMapping(tuple(kept))

    def detached(self, name: str, ref: ColumnRef, new_name: str) -> SchemaMapping:
        attribute = self.attribute(name)
        if ref not in attribute.columns or len(attribute.columns) == 1:
            return self
        if new_name in self.names:
            raise ValueError(f"Атрибут «{new_name}» уже існує")
        remaining = replace(attribute, columns=tuple(item for item in attribute.columns if item != ref))
        detached = Attribute(new_name, attribute.type, (ref,), 1.0)
        attributes = [remaining if item is attribute else item for item in self.attributes]
        return SchemaMapping((*attributes, detached))

    def to_frame(self, sources: Iterable[str]) -> pd.DataFrame:
        sources = list(sources)
        rows = []
        for attribute in self.attributes:
            row = {
                "attribute": attribute.name,
                "type": attribute.type.value,
                "confidence": round(attribute.confidence, 3),
            }
            row.update({source: attribute.column_in(source) or "" for source in sources})
            rows.append(row)
        return pd.DataFrame(rows, columns=["attribute", "type", "confidence", *sources])

    def _replaced(self, name: str, change) -> SchemaMapping:
        found = False
        attributes = []
        for attribute in self.attributes:
            if attribute.name == name:
                attribute, found = change(attribute), True
            attributes.append(attribute)
        if not found:
            raise KeyError(f"Немає атрибута «{name}»")
        return SchemaMapping(tuple(attributes))


def joined_values(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    if len(columns) == 1:
        return frame[columns[0]]
    parts = [frame[column].fillna("").astype(str).str.strip() for column in columns]
    joined = parts[0]
    for part in parts[1:]:
        joined = joined.str.cat(part, sep=" ")
    return joined.str.replace(SPACES, " ", regex=True).str.strip()
