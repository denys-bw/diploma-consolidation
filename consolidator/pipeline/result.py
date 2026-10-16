from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from consolidator.fusion import FusionResult
from consolidator.imputation import ImputationResult
from consolidator.io import Source
from consolidator.linkage import LinkageResult, RecordSet
from consolidator.pipeline.context import PipelineContext
from consolidator.pipeline.settings import PipelineSettings
from consolidator.profiling import TableProfile
from consolidator.quality import QualityScores
from consolidator.schema import SchemaMapping
from consolidator.standardization import StandardizedTable


@dataclass(frozen=True, eq=False)
class PreparedData:
    sources: tuple[Source, ...]
    profiles: dict[str, TableProfile]
    mapping: SchemaMapping
    frames: dict[str, pd.DataFrame]
    standardized: dict[str, StandardizedTable]
    records: RecordSet
    timings: dict[str, float]

    @classmethod
    def from_context(cls, context: PipelineContext) -> PreparedData:
        return cls(
            sources=context.sources,
            profiles=context.profiles,
            mapping=context.mapping,
            frames=context.frames,
            standardized=context.standardized,
            records=context.records,
            timings=dict(context.timings),
        )


@dataclass(frozen=True, eq=False)
class PipelineResult:
    settings: PipelineSettings
    prepared: PreparedData
    linkage: LinkageResult | None
    fusion: FusionResult
    imputation: ImputationResult | None
    consolidated: pd.DataFrame
    provenance: pd.DataFrame
    applicable: pd.DataFrame
    quality_before: QualityScores
    quality_after: QualityScores
    timings: dict[str, float]

    @classmethod
    def from_context(cls, prepared: PreparedData, context: PipelineContext) -> PipelineResult:
        return cls(
            settings=context.settings,
            prepared=prepared,
            linkage=context.linkage,
            fusion=context.fusion,
            imputation=context.imputation,
            consolidated=context.consolidated,
            provenance=context.provenance,
            applicable=context.applicable,
            quality_before=context.quality_before,
            quality_after=context.quality_after,
            timings=dict(context.timings),
        )

    @property
    def sources(self) -> tuple[Source, ...]:
        return self.prepared.sources

    @property
    def profiles(self) -> dict[str, TableProfile]:
        return self.prepared.profiles

    @property
    def mapping(self) -> SchemaMapping:
        return self.prepared.mapping

    @property
    def standardized(self) -> dict[str, StandardizedTable]:
        return self.prepared.standardized

    @property
    def records(self) -> RecordSet:
        return self.prepared.records

    @property
    def attributes(self) -> list[str]:
        return self.mapping.names

    @property
    def index_before(self) -> float:
        return self.quality_before.index(self.settings.weights)

    @property
    def index_after(self) -> float:
        return self.quality_after.index(self.settings.weights)

    def record_values(self) -> pd.DataFrame:
        values = self.consolidated[self.attributes].to_numpy(dtype=object)[self.fusion.entity_ids]
        frame = pd.DataFrame(values, columns=self.attributes)
        frame.insert(0, "source", self.fusion.membership["source"].to_numpy())
        frame.insert(1, "key", self.records.keys)
        frame.insert(2, "entity", self.fusion.entity_ids)
        return frame
