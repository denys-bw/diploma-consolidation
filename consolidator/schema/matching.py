from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from consolidator.io.source import Source
from consolidator.profiling.profile import TableProfile, profile_source
from consolidator.profiling.types import ColumnType, infer_column_type
from consolidator.schema.evidence import DuplicateEvidence, PairEvidence
from consolidator.schema.families import FORMAT_FAMILIES, TEXT_FAMILY, compatible, family
from consolidator.schema.mapping import Attribute, ColumnRef, SchemaMapping, joined_values
from consolidator.schema.names import NameRelation, name_relation, positional_names, readable_name
from consolidator.statistics import SIGNIFICANCE, holm


@dataclass
class Cluster:
    refs: list[ColumnRef]
    confidence: float = 1.0
    evidence: list[float] = field(default_factory=list)

    @property
    def single(self) -> bool:
        return len(self.refs) == 1

    def add(self, ref: ColumnRef, effect: float) -> None:
        self.refs.append(ref)
        self.evidence.append(max(effect, 0.0))
        self.confidence = min(self.evidence)


def match_sources(
    sources: Sequence[Source],
    alpha: float = SIGNIFICANCE,
    profiles: Mapping[str, TableProfile] | None = None,
) -> SchemaMapping:
    frames = {source.name: source.frame for source in sources}
    column_sets = {frozenset(map(str, source.columns)) for source in sources}
    if len(column_sets) == 1 and not positional_names([str(column) for column in sources[0].columns]):
        combined = pd.concat(list(frames.values()), ignore_index=True)
        shared = {str(column): infer_column_type(combined[column]).type for column in combined.columns}
        return identity_mapping(frames, dict.fromkeys(frames, shared))
    profiles = profiles or {source.name: profile_source(source) for source in sources}
    return match_schemas(frames, {name: profiles[name].types for name in frames}, alpha)


def match_schemas(
    frames: Mapping[str, pd.DataFrame],
    types: Mapping[str, Mapping[str, ColumnType]],
    alpha: float = SIGNIFICANCE,
) -> SchemaMapping:
    frames = {source: frame.rename(columns=str) for source, frame in frames.items()}
    order = sorted(frames, key=lambda source: -len(frames[source].columns))
    clusters = [Cluster([ColumnRef(order[0], column)]) for column in frames[order[0]].columns]
    for source in order[1:]:
        SourceMatcher(frames, types, clusters, source, alpha).apply()
    return SchemaMapping(tuple(_attributes(clusters, frames, list(frames))))


class SourceMatcher:
    def __init__(
        self,
        frames: Mapping[str, pd.DataFrame],
        types: Mapping[str, Mapping[str, ColumnType]],
        clusters: list[Cluster],
        source: str,
        alpha: float,
    ) -> None:
        self.frames = frames
        self.types = types
        self.clusters = clusters
        self.source = source
        self.alpha = alpha
        self.columns = list(frames[source].columns)
        self.labels = [str(index) for index in range(len(clusters))]
        self.left_types = {column: types[source][column] for column in self.columns}
        self.right_types = {label: self._cluster_type(cluster) for label, cluster in zip(self.labels, clusters)}
        self.evidence = DuplicateEvidence(
            frames[source][self.columns], self._integrated(), self.left_types, self.right_types
        )
        self.positional_left = positional_names(self.columns)
        self.tests = [
            [self.evidence.correspondence((column,), (label,)) for label in self.labels] for column in self.columns
        ]

    def apply(self) -> None:
        assigned = self._by_values()
        self._by_names(assigned)
        self._by_format(assigned)
        absorbed_left, absorbed_right = self._composites(assigned)
        for row, column in enumerate(self.columns):
            if row in assigned or row in absorbed_left:
                continue
            self.clusters.append(Cluster([ColumnRef(self.source, column)]))
        for row, (target, effect, extension) in assigned.items():
            columns = [self.columns[row]] if extension is None else self._ordered([self.columns[row], extension])
            ref = ColumnRef(self.source, columns[0]) if len(columns) == 1 else ColumnRef.composite(self.source, columns)
            self.clusters[target].add(ref, effect)
        for target in sorted(absorbed_right, reverse=True):
            del self.clusters[target]

    def _by_values(self) -> dict[int, tuple[int, float, str | None]]:
        if not self.columns or not self.labels:
            return {}
        p_values = np.array([[test.p_value for test in row] for row in self.tests])
        significant = holm(p_values.ravel(), self.alpha).reshape(p_values.shape)
        effects = np.array([[test.effect for test in row] for row in self.tests])
        significant &= effects > 0
        strength = np.where(significant, effects, 0.0)
        rows, columns = linear_sum_assignment(-strength)
        row_best, column_best = strength.max(axis=1), strength.max(axis=0)
        return {
            int(row): (int(column), self.tests[row][column].effect, None)
            for row, column in zip(rows, columns)
            if significant[row, column]
            and strength[row, column] >= row_best[row]
            and strength[row, column] >= column_best[column]
        }

    def _by_names(self, assigned: dict[int, tuple[int, float, str | None]]) -> None:
        taken = {target for target, _, _ in assigned.values()}
        rows = [row for row in range(len(self.columns)) if row not in assigned]
        targets = [target for target in range(len(self.clusters)) if target not in taken]
        relation = np.zeros((len(self.columns), len(self.clusters)), dtype=np.int64)
        for row in rows:
            column = self.columns[row]
            if column in self.positional_left:
                continue
            for target in targets:
                if compatible(self.left_types[column], self.right_types[self.labels[target]]):
                    relation[row, target] = int(self._name_relation(column, self.clusters[target]))
        for row in rows:
            best = relation[row].max()
            if best == NameRelation.NONE:
                continue
            chosen = np.flatnonzero(relation[row] == best)
            if (
                len(chosen) == 1
                and np.flatnonzero(relation[:, chosen[0]] == best).tolist() == [row]
                and relation[:, chosen[0]].max() == best
            ):
                target = int(chosen[0])
                assigned[row] = (target, self.tests[row][target].effect, None)

    def _by_format(self, assigned: dict[int, tuple[int, float, str | None]]) -> None:
        taken = {target for target, _, _ in assigned.values()}
        for kind in sorted(FORMAT_FAMILIES):
            rows = [
                row
                for row, column in enumerate(self.columns)
                if row not in assigned and family(self.left_types[column]) == kind
            ]
            targets = [
                target
                for target, label in enumerate(self.labels)
                if target not in taken and family(self.right_types[label]) == kind
            ]
            if len(rows) == 1 and len(targets) == 1:
                assigned[rows[0]] = (targets[0], self.tests[rows[0]][targets[0]].effect, None)
                taken.add(targets[0])

    def _composites(self, assigned: dict[int, tuple[int, float, str | None]]) -> tuple[set[int], set[int]]:
        taken = {target for target, _, _ in assigned.values()}
        free_rows = [
            row
            for row, column in enumerate(self.columns)
            if row not in assigned and family(self.left_types[column]) == TEXT_FAMILY
        ]
        free_targets = [
            target
            for target, cluster in enumerate(self.clusters)
            if target not in taken and cluster.single and family(self.right_types[self.labels[target]]) == TEXT_FAMILY
        ]
        options: list[tuple[int, str, int, PairEvidence]] = []
        for row, (target, _, _) in assigned.items():
            column, label = self.columns[row], self.labels[target]
            if family(self.left_types[column]) != TEXT_FAMILY or family(self.right_types[label]) != TEXT_FAMILY:
                continue
            base = ((column,), (label,))
            for other in free_rows:
                extended = (tuple(self._ordered([column, self.columns[other]])), (label,))
                options.append((row, "left", other, self.evidence.improvement(base, extended)))
            if self.clusters[target].single:
                owner = self.clusters[target].refs[0].source
                for other in free_targets:
                    if self.clusters[other].refs[0].source != owner:
                        continue
                    extended = ((column,), tuple(self._right_order([target, other])))
                    options.append((row, "right", other, self.evidence.improvement(base, extended)))
        if not options:
            return set(), set()
        accepted = holm(np.array([evidence.p_value for *_, evidence in options]), self.alpha)
        best: dict[int, tuple[str, int, PairEvidence]] = {}
        for (row, side, other, evidence), significant in zip(options, accepted):
            if significant and evidence.effect > 0 and (row not in best or evidence.effect > best[row][2].effect):
                best[row] = (side, other, evidence)
        absorbed_left: set[int] = set()
        absorbed_right: set[int] = set()
        for row, (side, other, _) in sorted(best.items(), key=lambda item: -item[1][2].effect):
            target, effect, _ = assigned[row]
            if side == "left" and other not in absorbed_left:
                assigned[row] = (target, effect, self.columns[other])
                absorbed_left.add(other)
            elif side == "right" and other not in absorbed_right and target not in absorbed_right:
                self._extend_cluster(target, other)
                absorbed_right.add(other)
        return absorbed_left, absorbed_right

    def _extend_cluster(self, target: int, other: int) -> None:
        base, addition = self.clusters[target].refs[0], self.clusters[other].refs[0]
        columns = [column for column in self.frames[base.source].columns if column in (base.column, addition.column)]
        self.clusters[target].refs[0] = ColumnRef.composite(base.source, columns)

    def _ordered(self, columns: Sequence[str]) -> list[str]:
        return [column for column in self.columns if column in columns]

    def _right_order(self, targets: Sequence[int]) -> list[str]:
        refs = {target: self.clusters[target].refs[0] for target in targets}
        position = {column: index for index, column in enumerate(self.frames[refs[targets[0]].source].columns)}
        return [self.labels[target] for target in sorted(targets, key=lambda target: position[refs[target].column])]

    def _name_relation(self, column: str, cluster: Cluster) -> NameRelation:
        relations = [
            name_relation(column, ref.column)
            for ref in cluster.refs
            if ref.column not in positional_names(list(self.frames[ref.source].columns))
        ]
        return max(relations, default=NameRelation.NONE)

    def _cluster_type(self, cluster: Cluster) -> ColumnType:
        ref = cluster.refs[0]
        parts = ref.parts(self.frames[ref.source].columns)
        return self.types[ref.source].get(parts[0], ColumnType.TEXT)

    def _integrated(self) -> pd.DataFrame:
        sources = list(dict.fromkeys(ref.source for cluster in self.clusters for ref in cluster.refs))
        parts = []
        for source in sources:
            frame = self.frames[source]
            columns = {}
            for label, cluster in zip(self.labels, self.clusters):
                ref = next((ref for ref in cluster.refs if ref.source == source), None)
                columns[label] = (
                    joined_values(frame, list(ref.parts(frame.columns))).fillna("").astype(str)
                    if ref is not None
                    else pd.Series("", index=frame.index)
                )
            parts.append(pd.DataFrame(columns, index=frame.index))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=self.labels)


def identity_mapping(
    frames: Mapping[str, pd.DataFrame], types: Mapping[str, Mapping[str, ColumnType]]
) -> SchemaMapping:
    columns: dict[str, list[ColumnRef]] = {}
    for source, frame in frames.items():
        for column in frame.columns:
            columns.setdefault(str(column), []).append(ColumnRef(source, str(column)))
    attributes = []
    for name, refs in columns.items():
        declared = {types[ref.source][ref.column] for ref in refs}
        if len(declared) == 1:
            column_type = declared.pop()
        else:
            column_type = infer_column_type(pd.concat([frames[ref.source][ref.column] for ref in refs])).type
        attributes.append(Attribute(name, column_type, tuple(refs)))
    return SchemaMapping(tuple(attributes))


def _attributes(clusters: Sequence[Cluster], frames: Mapping[str, pd.DataFrame], order: list[str]) -> list[Attribute]:
    position = {source: index for index, source in enumerate(order)}
    names: set[str] = set()
    attributes = []
    for cluster in clusters:
        refs = tuple(sorted(cluster.refs, key=lambda ref: position[ref.source]))
        values = [joined_values(frames[ref.source], list(ref.parts(frames[ref.source].columns))) for ref in refs]
        pooled = infer_column_type(pd.concat(values, ignore_index=True)).type
        name = _unique(_display_name(refs[0]), names)
        attributes.append(Attribute(name, pooled, refs, cluster.confidence))
    return attributes


def _display_name(ref: ColumnRef) -> str:
    if "://" in ref.column:
        return readable_name(ref.column).replace(" ", "_") or ref.column
    return ref.column


def _unique(name: str, taken: set[str]) -> str:
    candidate, suffix = name, 2
    while candidate in taken:
        candidate, suffix = f"{name}_{suffix}", suffix + 1
    taken.add(candidate)
    return candidate
