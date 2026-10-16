from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from consolidator.evaluation import evaluate_positions
from consolidator.linkage import (
    ClericalReview,
    CoTraining,
    LinkageResult,
    LinkageSettings,
    Linker,
    PairScores,
    RecordSet,
    ScoringTask,
)
from experiments.common import available, gold_positions, linkage_settings, linked_pairs, prepare, save

DATASETS = ["restaurants", "chicago_sources", "dblp_acm", "abt_buy", "settlements", "musicbrainz"]
BUDGETS = (10, 20, 30, 50)
BATCH = 10
LEARNING = "модель навчається на відповідях"
FIXING = "відповіді лише фіксують перевірені пари"


class ReviewWithoutLearning(CoTraining):
    def score(self, task: ScoringTask) -> PairScores:
        return self.unsupervised(task)


def main() -> None:
    rows = []
    for dataset in available(DATASETS):
        print(f"[активне навчання] {dataset.title}")
        records = prepare(dataset).records
        gold = gold_positions(dataset, records)
        truth = {tuple(pair) for pair in np.sort(gold, axis=1).tolist()}
        settings = linkage_settings()
        for mode, scorer in ((LEARNING, CoTraining()), (FIXING, ReviewWithoutLearning())):
            for answers, result in _session(records, truth, replace(settings, scorer=scorer)):
                metrics = evaluate_positions(linked_pairs(dataset, result), gold, records.size)
                rows.append(
                    {
                        "набір": dataset.name,
                        "режим": mode,
                        "відповідей": answers,
                        "precision": metrics.precision,
                        "recall": metrics.recall,
                        "F1": metrics.f1,
                    }
                )
    save(pd.DataFrame(rows), "e8_active_learning.csv")


def _session(records: RecordSet, truth: set[tuple[int, int]], settings: LinkageSettings):
    answers: dict[tuple[int, int], bool] = {}
    result = Linker(settings).run(records)
    yield 0, result
    for budget in BUDGETS:
        while len(answers) < budget:
            batch = _next_pairs(result, answers, min(BATCH, budget - len(answers)))
            if not batch:
                break
            answers.update((pair, pair in truth) for pair in batch)
            review = ClericalReview.of(
                (records.record_id(left), records.record_id(right), is_match)
                for (left, right), is_match in answers.items()
            )
            result = Linker(settings).run(records, review)
        yield budget, result


def _next_pairs(result: LinkageResult, answers: dict[tuple[int, int], bool], limit: int) -> list[tuple[int, int]]:
    chosen = []
    for index in result.review_queue(extended=True):
        pair = tuple(int(value) for value in result.candidates[int(index)])
        if pair not in answers:
            chosen.append(pair)
        if len(chosen) == limit:
            break
    return chosen


if __name__ == "__main__":
    main()
