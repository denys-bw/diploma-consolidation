from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from consolidator.fusion import ConflictResolver
from consolidator.imputation import AdaptiveImputer, Imputer
from consolidator.linkage import LinkageSettings
from consolidator.quality import QualityWeights
from consolidator.standardization import StandardizationSettings
from consolidator.statistics import SIGNIFICANCE


@dataclass(frozen=True)
class PipelineSettings:
    standardization: StandardizationSettings = field(default_factory=StandardizationSettings)
    schema_alpha: float = SIGNIFICANCE
    linkage: LinkageSettings | None = field(default_factory=LinkageSettings)
    resolvers: Mapping[str, ConflictResolver] = field(default_factory=dict)
    source_priority: tuple[str, ...] = ()
    imputer: Imputer | None = field(default_factory=AdaptiveImputer)
    weights: QualityWeights = field(default_factory=QualityWeights.default)
