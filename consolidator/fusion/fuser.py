from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from consolidator.fusion.resolvers import NO_VALUE, ConflictResolver, FusionContext, default_resolver
from consolidator.fusion.truth import PRIOR_ACCURACY, Claims, discover_truth
from consolidator.linkage.records import RecordSet
from consolidator.profiling.types import ColumnType
from consolidator.values.text import comparison_keys


class Fuser:
    def __init__(
        self, resolvers: Mapping[str, ConflictResolver] | None = None, source_priority: Sequence[str] = ()
    ) -> None:
        self.resolvers = dict(resolvers or {})
        self.source_priority = tuple(source_priority)

    def fuse(self, records: RecordSet, labels: np.ndarray) -> FusionResult:
        labels = np.asarray(labels)
        if len(labels) != records.size:
            raise ValueError("Кількість міток кластерів не збігається з кількістю записів")
        plan = {attribute: self.resolver_for(attribute, records) for attribute in records.attributes}
        entity_ids = np.unique(labels, return_inverse=True)[1].ravel().astype(np.int64)
        n_entities = int(entity_ids.max()) + 1 if len(entity_ids) else 0
        order = np.lexsort((np.arange(records.size), self._source_ranks(records), entity_ids))
        keys = {attribute: comparison_keys(records.values(attribute)) for attribute in records.attributes}
        voted = [attribute for attribute, resolver in plan.items() if resolver.uses_source_accuracy]
        accuracy, truth_scores = _discover_truth(records, entity_ids, keys, voted)
        context = FusionContext(records, entity_ids, n_entities, order, keys, truth_scores)
        chosen = {attribute: resolver.choose(attribute, context) for attribute, resolver in plan.items()}
        return FusionResult(
            records=records,
            entity_ids=entity_ids,
            chosen=chosen,
            keys=keys,
            resolvers=plan,
            source_accuracy=dict(zip(records.source_names, accuracy.tolist())),
        )

    def resolver_for(self, attribute: str, records: RecordSet) -> ConflictResolver:
        return self.resolvers.get(attribute) or default_resolver(records.types.get(attribute, ColumnType.TEXT))

    def _source_ranks(self, records: RecordSet) -> np.ndarray:
        priority = {name: rank for rank, name in enumerate(self.source_priority or records.source_names)}
        ranks = np.array([priority.get(name, len(priority)) for name in records.source_names], dtype=np.int64)
        return ranks[records.source_codes] if len(ranks) else np.empty(0, dtype=np.int64)


@dataclass(frozen=True, eq=False)
class FusionResult:
    records: RecordSet
    entity_ids: np.ndarray
    chosen: dict[str, np.ndarray]
    keys: dict[str, np.ndarray]
    resolvers: dict[str, ConflictResolver]
    source_accuracy: dict[str, float]

    @property
    def n_entities(self) -> int:
        return int(self.entity_ids.max()) + 1 if len(self.entity_ids) else 0

    @property
    def merged_records(self) -> int:
        return self.records.size - self.n_entities

    @cached_property
    def membership(self) -> pd.DataFrame:
        names = np.asarray(self.records.source_names, dtype=object)
        return pd.DataFrame(
            {"entity": self.entity_ids, "source": names[self.records.source_codes], "key": self.records.keys}
        )

    @cached_property
    def entities(self) -> pd.DataFrame:
        frame = pd.DataFrame(
            {attribute: self._entity_values(attribute) for attribute in self.records.attributes},
            columns=self.records.attributes,
        )
        frame.insert(0, "sources", self._entity_sources())
        frame.insert(1, "records", np.bincount(self.entity_ids, minlength=self.n_entities))
        frame.index.name = "entity"
        return frame

    @cached_property
    def provenance(self) -> pd.DataFrame:
        names = np.append(np.asarray(self.records.source_names, dtype=object), "")
        columns = {}
        for attribute, chosen in self.chosen.items():
            codes = np.where(chosen == NO_VALUE, len(names) - 1, self.records.source_codes[np.maximum(chosen, 0)])
            columns[attribute] = names[codes]
        frame = pd.DataFrame(columns, columns=self.records.attributes)
        frame.index.name = "entity"
        return frame

    @cached_property
    def conflicts(self) -> pd.DataFrame:
        rows = []
        for attribute in self.records.attributes:
            keys = self.keys[attribute]
            present = keys != ""
            if not present.any():
                continue
            frame = pd.DataFrame({"entity": self.entity_ids[present], "key": keys[present]})
            counts = frame.groupby("entity")["key"].agg(["size", "nunique"])
            disputed = counts[counts["nunique"] > 1]
            chosen = self._entity_values(attribute)
            rows.extend(
                {
                    "entity": int(entity),
                    "attribute": attribute,
                    "values": int(size),
                    "distinct": int(distinct),
                    "chosen": chosen[int(entity)],
                }
                for entity, size, distinct in zip(disputed.index, disputed["size"], disputed["nunique"])
            )
        return pd.DataFrame(rows, columns=["entity", "attribute", "values", "distinct", "chosen"])

    def record_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(
            {attribute: self._entity_values(attribute)[self.entity_ids] for attribute in self.records.attributes},
            columns=self.records.attributes,
        )
        frame.insert(0, "source", self.membership["source"].to_numpy())
        frame.insert(1, "key", self.records.keys)
        return frame

    def conflicting_cells(self) -> pd.DataFrame:
        columns = {}
        for attribute in self.records.attributes:
            own, fused = self.keys[attribute], self._fused_keys(attribute)
            columns[attribute] = (own != "") & (fused != "") & (own != fused)
        return pd.DataFrame(columns, columns=self.records.attributes)

    def filled_cells(self) -> pd.DataFrame:
        columns = {
            attribute: (self.keys[attribute] == "") & (self._fused_keys(attribute) != "")
            for attribute in self.records.attributes
        }
        return pd.DataFrame(columns, columns=self.records.attributes)

    def _entity_values(self, attribute: str) -> np.ndarray:
        chosen = self.chosen[attribute]
        values = self.records.values(attribute)
        return np.where(chosen == NO_VALUE, "", values[np.maximum(chosen, 0)]).astype(object)

    def _fused_keys(self, attribute: str) -> np.ndarray:
        chosen = self.chosen[attribute][self.entity_ids]
        return np.where(chosen == NO_VALUE, "", self.keys[attribute][np.maximum(chosen, 0)])

    def _entity_sources(self) -> np.ndarray:
        names = np.asarray(self.records.source_names, dtype=object)
        pairs = pd.DataFrame({"entity": self.entity_ids, "code": self.records.source_codes}).drop_duplicates()
        pairs = pairs.sort_values(["entity", "code"], kind="stable")
        joined = pairs.groupby("entity")["code"].agg(lambda codes: ", ".join(names[codes.to_numpy()]))
        return joined.reindex(range(self.n_entities), fill_value="").to_numpy(dtype=object)


def _discover_truth(
    records: RecordSet, entity_ids: np.ndarray, keys: Mapping[str, np.ndarray], attributes: Sequence[str]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not attributes:
        return np.full(len(records.source_names), PRIOR_ACCURACY), {}
    items, values, sources, slices = [], [], [], {}
    offset = 0
    for index, attribute in enumerate(attributes):
        positions = np.flatnonzero(keys[attribute] != "")
        items.append(entity_ids[positions] * len(attributes) + index)
        values.append(pd.factorize(keys[attribute][positions])[0])
        sources.append(records.source_codes[positions])
        slices[attribute] = (positions, slice(offset, offset + len(positions)))
        offset += len(positions)
    claims = Claims(
        items=np.concatenate(items),
        values=np.concatenate(values),
        sources=np.concatenate(sources),
        n_sources=len(records.source_names),
    )
    estimate = discover_truth(claims)
    scores = {}
    for attribute, (positions, part) in slices.items():
        column = np.full(records.size, -np.inf)
        column[positions] = estimate.scores[part]
        scores[attribute] = column
    return estimate.accuracy, scores
