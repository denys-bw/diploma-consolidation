from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from consolidator.fusion import Fuser
from consolidator.imputation import impute
from consolidator.linkage import Linker, RecordSet
from consolidator.pipeline.context import PipelineContext
from consolidator.profiling import profile_source
from consolidator.quality import DataSnapshot, assess
from consolidator.schema import match_sources
from consolidator.standardization import standardize_frame, standardize_frames


class Stage(ABC):
    name = ""

    @abstractmethod
    def run(self, context: PipelineContext) -> None:
        raise NotImplementedError


class ProfilingStage(Stage):
    name = "profiling"

    def run(self, context: PipelineContext) -> None:
        for source in context.sources:
            if source.name not in context.profiles:
                context.profiles[source.name] = profile_source(source)


class SchemaStage(Stage):
    name = "schema"

    def run(self, context: PipelineContext) -> None:
        if context.mapping is None:
            context.mapping = match_sources(context.sources, context.settings.schema_alpha, context.profiles)
        context.frames = context.mapping.apply({source.name: source.frame for source in context.sources})


class StandardizationStage(Stage):
    name = "standardization"

    def run(self, context: PipelineContext) -> None:
        types = context.mapping.types
        tables = standardize_frames(
            list(context.frames.values()), types, context.settings.standardization, list(context.frames)
        )
        context.standardized = dict(zip(context.frames, tables))
        context.records = RecordSet.from_frames(
            {name: table.frame for name, table in context.standardized.items()}, types
        )


class LinkageStage(Stage):
    name = "linkage"

    def run(self, context: PipelineContext) -> None:
        settings = context.settings.linkage
        if settings is None:
            context.labels = np.arange(context.records.size)
            return
        context.linkage = Linker(settings).run(context.records, context.review)
        context.labels = context.linkage.labels


class FusionStage(Stage):
    name = "fusion"

    def run(self, context: PipelineContext) -> None:
        settings = context.settings
        fusion = Fuser(settings.resolvers, settings.source_priority).fuse(context.records, context.labels)
        context.fusion = fusion
        context.consolidated = fusion.entities.copy()
        context.provenance = fusion.provenance.copy()
        context.applicable = _applicable_cells(context)


class ImputationStage(Stage):
    name = "imputation"

    def run(self, context: PipelineContext) -> None:
        imputer = context.settings.imputer
        if imputer is None:
            return
        attributes = context.attributes
        imputation = impute(
            context.consolidated[attributes],
            context.mapping.types,
            imputer,
            context.settings.standardization,
            context.applicable,
        )
        context.imputation = imputation
        context.consolidated[attributes] = imputation.frame
        context.provenance = context.provenance.mask(imputation.imputed, imputation.mark)


class QualityStage(Stage):
    name = "quality"

    def run(self, context: PipelineContext) -> None:
        attributes, types = context.attributes, context.mapping.types
        entities = context.fusion.n_entities
        before = DataSnapshot(
            frames=[frame.reindex(columns=attributes) for frame in context.frames.values()],
            statuses=[table.status for table in context.standardized.values()],
            types=types,
            records=context.records.size,
            entities=entities,
        )
        consolidated = context.consolidated[attributes]
        final = standardize_frame(consolidated, types, context.settings.standardization)
        after = DataSnapshot([consolidated], [final.status], types, entities, entities)
        context.quality_before = assess(before)
        context.quality_after = assess(after)


PREPARATION: tuple[Stage, ...] = (ProfilingStage(), SchemaStage(), StandardizationStage())
CONSOLIDATION: tuple[Stage, ...] = (LinkageStage(), FusionStage(), ImputationStage(), QualityStage())


def _applicable_cells(context: PipelineContext) -> pd.DataFrame:
    records, fusion = context.records, context.fusion
    names = np.asarray(records.source_names, dtype=object)
    columns = {}
    for attribute in context.mapping.attributes:
        provided = np.isin(names, attribute.sources)
        members = provided[records.source_codes].astype(np.float64)
        columns[attribute.name] = np.bincount(fusion.entity_ids, weights=members, minlength=fusion.n_entities) > 0
    return pd.DataFrame(columns, index=fusion.entities.index, columns=context.attributes)
