from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from consolidator.fusion import FusionResult
from consolidator.imputation import ImputationResult
from consolidator.io import Source
from consolidator.linkage import ClericalReview, LinkageResult, RecordSet
from consolidator.pipeline.settings import PipelineSettings
from consolidator.profiling import TableProfile
from consolidator.quality import QualityScores
from consolidator.schema import SchemaMapping
from consolidator.standardization import StandardizedTable


@dataclass(eq=False)
class PipelineContext:
    settings: PipelineSettings
    sources: tuple[Source, ...]
    profiles: dict[str, TableProfile] = field(default_factory=dict)
    mapping: SchemaMapping | None = None
    review: ClericalReview = field(default_factory=ClericalReview)
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    standardized: dict[str, StandardizedTable] = field(default_factory=dict)
    records: RecordSet | None = None
    linkage: LinkageResult | None = None
    labels: np.ndarray | None = None
    fusion: FusionResult | None = None
    consolidated: pd.DataFrame | None = None
    provenance: pd.DataFrame | None = None
    applicable: pd.DataFrame | None = None
    imputation: ImputationResult | None = None
    quality_before: QualityScores | None = None
    quality_after: QualityScores | None = None
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def attributes(self) -> list[str]:
        return self.mapping.names
