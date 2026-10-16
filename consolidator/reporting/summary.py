from __future__ import annotations

from dataclasses import asdict, dataclass

from consolidator.linkage import MatchStatus
from consolidator.pipeline import PipelineResult
from consolidator.standardization import CellStatus


@dataclass(frozen=True)
class RunSummary:
    source_count: int
    record_count: int
    attribute_count: int
    entity_count: int
    merged_records: int
    candidate_pairs: int
    matched_pairs: int
    possible_pairs: int
    reviewed_pairs: int
    unified_cells: int
    unrecognized_cells: int
    ambiguous_cells: int
    conflicting_cells: int
    filled_cells: int
    imputed_cells: int
    weights: dict[str, float]
    quality_before: dict[str, float]
    quality_after: dict[str, float]
    index_before: float
    index_after: float
    seconds: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def summarize(result: PipelineResult) -> RunSummary:
    statuses = [table.status.to_numpy() for table in result.standardized.values()]
    linkage = result.linkage
    return RunSummary(
        source_count=len(result.sources),
        record_count=result.records.size,
        attribute_count=len(result.attributes),
        entity_count=result.fusion.n_entities,
        merged_records=result.fusion.merged_records,
        candidate_pairs=len(linkage.candidates) if linkage is not None else 0,
        matched_pairs=int(linkage.accepted.sum()) if linkage is not None else 0,
        possible_pairs=linkage.count(MatchStatus.POSSIBLE) if linkage is not None else 0,
        reviewed_pairs=int((linkage.reviewed >= 0).sum()) if linkage is not None else 0,
        unified_cells=sum(int((codes == CellStatus.CHANGED).sum()) for codes in statuses),
        unrecognized_cells=sum(int((codes == CellStatus.INVALID).sum()) for codes in statuses),
        ambiguous_cells=sum(int((codes == CellStatus.AMBIGUOUS).sum()) for codes in statuses),
        conflicting_cells=int(result.fusion.conflicting_cells().to_numpy().sum()),
        filled_cells=int(result.fusion.filled_cells().to_numpy().sum()),
        imputed_cells=result.imputation.count if result.imputation is not None else 0,
        weights=dict(result.settings.weights),
        quality_before=dict(result.quality_before),
        quality_after=dict(result.quality_after),
        index_before=result.index_before,
        index_after=result.index_after,
        seconds=sum(result.timings.values()),
    )
