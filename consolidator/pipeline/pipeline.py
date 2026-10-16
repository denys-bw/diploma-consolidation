from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Protocol

from consolidator.io import Source, with_unique_names
from consolidator.linkage import ClericalReview
from consolidator.pipeline.context import PipelineContext
from consolidator.pipeline.result import PipelineResult, PreparedData
from consolidator.pipeline.settings import PipelineSettings
from consolidator.pipeline.stages import CONSOLIDATION, PREPARATION, Stage
from consolidator.profiling import TableProfile
from consolidator.schema import SchemaMapping, match_sources


class StageObserver(Protocol):
    def stage_started(self, stage: str) -> None: ...


class Pipeline:
    def __init__(self, settings: PipelineSettings | None = None, observers: Sequence[StageObserver] = ()) -> None:
        self.settings = settings or PipelineSettings()
        self.observers = tuple(observers)

    def match(self, sources: Sequence[Source], profiles: Mapping[str, TableProfile] | None = None) -> SchemaMapping:
        return match_sources(sources, self.settings.schema_alpha, profiles)

    def prepare(
        self,
        sources: Sequence[Source],
        mapping: SchemaMapping | None = None,
        profiles: Mapping[str, TableProfile] | None = None,
    ) -> PreparedData:
        context = PipelineContext(
            self.settings, tuple(with_unique_names(list(sources))), profiles=dict(profiles or {}), mapping=mapping
        )
        self._execute(PREPARATION, context)
        return PreparedData.from_context(context)

    def consolidate(self, prepared: PreparedData, review: ClericalReview | None = None) -> PipelineResult:
        context = PipelineContext(
            self.settings,
            prepared.sources,
            profiles=prepared.profiles,
            mapping=prepared.mapping,
            review=review or ClericalReview(),
            frames=prepared.frames,
            standardized=prepared.standardized,
            records=prepared.records,
            timings=dict(prepared.timings),
        )
        self._execute(CONSOLIDATION, context)
        return PipelineResult.from_context(prepared, context)

    def run(
        self,
        sources: Sequence[Source],
        mapping: SchemaMapping | None = None,
        profiles: Mapping[str, TableProfile] | None = None,
        review: ClericalReview | None = None,
    ) -> PipelineResult:
        return self.consolidate(self.prepare(sources, mapping, profiles), review)

    def _execute(self, stages: Sequence[Stage], context: PipelineContext) -> None:
        for stage in stages:
            for observer in self.observers:
                observer.stage_started(stage.name)
            started = time.perf_counter()
            stage.run(context)
            context.timings[stage.name] = time.perf_counter() - started
