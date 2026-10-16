from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from benchmarks import Dataset
from consolidator.evaluation import evaluate_positions
from consolidator.linkage import Blocker, CoTraining, DecisionRule, LinkageSettings, Linker, PairScope, RecordSet
from consolidator.statistics import MAJORITY, PRIOR_SCALE, SIGNIFICANCE
from experiments.common import (
    MAIN_DATASETS,
    available,
    gold_positions,
    linkage_settings,
    linked_pairs,
    pipeline_settings,
    prepare,
    save,
)

DATASETS = MAIN_DATASETS
MEAN = "середнє F1"

Change = Callable[[LinkageSettings, float], LinkageSettings]


@dataclass(frozen=True)
class Parameter:
    label: str
    values: tuple[float, ...]
    default: float
    change: Change | None = None


@dataclass(frozen=True, eq=False)
class FixedCandidates(Blocker):
    pairs: np.ndarray

    @property
    def label(self) -> str:
        return "пари-кандидати з типового запуску"

    def candidates(self, records: RecordSet, scope: PairScope) -> np.ndarray:
        return self.pairs


def _threshold(settings: LinkageSettings, value: float) -> LinkageSettings:
    decision = settings.decision
    return replace(settings, decision=DecisionRule(value, decision.false_match_rate, decision.missed_match_rate))


def _prior_scale(settings: LinkageSettings, value: float) -> LinkageSettings:
    return replace(settings, scorer=CoTraining(prior_scale=value))


PARAMETERS = (
    Parameter("поріг імовірності збігу (правило Баєса)", (0.4, MAJORITY, 0.6), MAJORITY, _threshold),
    Parameter(
        "масштаб апріорного розподілу логістичної регресії",
        (PRIOR_SCALE / 2, PRIOR_SCALE, PRIOR_SCALE * 2),
        PRIOR_SCALE,
        _prior_scale,
    ),
    Parameter("рівень значущості для зіставлення колонок", (0.01, SIGNIFICANCE, 0.10), SIGNIFICANCE),
)


def main() -> None:
    scores: dict[tuple[str, float], dict[str, float]] = {}
    for dataset in available(DATASETS):
        print(f"[чутливість] {dataset.title}")
        records = prepare(dataset).records
        gold = gold_positions(dataset, records)
        settings = linkage_settings()
        baseline = Linker(settings).run(records)
        fixed = replace(settings, blocker=FixedCandidates(baseline.candidates))
        default = evaluate_positions(linked_pairs(dataset, baseline), gold, records.size).f1
        for parameter in PARAMETERS:
            for value in parameter.values:
                if value == parameter.default:
                    f1 = default
                elif parameter.change is None:
                    f1 = _with_schema_alpha(dataset, value)
                else:
                    result = Linker(parameter.change(fixed, value)).run(records)
                    f1 = evaluate_positions(linked_pairs(dataset, result), gold, records.size).f1
                scores.setdefault((parameter.label, value), {})[dataset.name] = f1
    rows = []
    for parameter in PARAMETERS:
        for value in parameter.values:
            row = {"параметр": parameter.label, "значення": value, "типове": value == parameter.default}
            row.update(scores[parameter.label, value])
            row[MEAN] = float(np.mean(list(scores[parameter.label, value].values())))
            rows.append(row)
    save(pd.DataFrame(rows), "e10_sensitivity.csv")


def _with_schema_alpha(dataset: Dataset, value: float) -> float:
    records = prepare(dataset, pipeline_settings(schema_alpha=value)).records
    result = Linker(linkage_settings()).run(records)
    return evaluate_positions(linked_pairs(dataset, result), gold_positions(dataset, records), records.size).f1


if __name__ == "__main__":
    main()
