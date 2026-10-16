from __future__ import annotations

import time

import pandas as pd

from benchmarks import Dataset, NeedsDownload
from benchmarks.multisource import NC_VOTERS_SAMPLE, load_nc_voters
from consolidator.evaluation import evaluate_blocking, evaluate_positions
from consolidator.linkage import Linker, PairScope
from consolidator.reporting import NEEDS_REVIEW, RELIABLE
from experiments.common import gold_positions, linkage_settings, linked_pairs, prepare, save

SHARES = (None, 0.1, 0.3, 0.5)
ENTITIES = NC_VOTERS_SAMPLE
NATURAL = "природна частка"


def main() -> None:
    rows = []
    for share in SHARES:
        try:
            dataset = load_nc_voters(ENTITIES, share)
        except NeedsDownload as error:
            print(f"[пропущено] nc_voters: {error}")
            return
        print(f"[частка дублікатів] {dataset.title}")
        rows.append(_row(dataset, share))
    save(pd.DataFrame(rows), "e11_duplicate_share.csv")


def _row(dataset: Dataset, share: float | None) -> dict[str, object]:
    started = time.perf_counter()
    records = prepare(dataset).records
    result = Linker(linkage_settings()).run(records)
    elapsed = time.perf_counter() - started
    gold = gold_positions(dataset, records)
    metrics = evaluate_positions(linked_pairs(dataset, result), gold, records.size)
    blocking = evaluate_blocking(result.candidates, gold, records.size, records.total_pairs(dataset.scope))
    reliability = result.reliability()
    accepted = result.accepted
    return {
        "людей з копіями": NATURAL if share is None else share,
        "людей": ENTITIES,
        "записів": records.size,
        "пар дублікатів в еталоні": len(dataset.gold),
        "повтори всередині джерел (рішення програми)": "так" if result.scope is PairScope.ALL else "ні",
        "PC блокування": blocking.pairs_completeness,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "F1": metrics.f1,
        "оцінка програми": RELIABLE if reliability.reliable else NEEDS_REVIEW,
        "очікувано хибних збігів (модель)": reliability.expected_false,
        "очікувано пропущених збігів (модель)": reliability.expected_missed,
        "прийнято пар": int(accepted.sum()),
        "час, с": elapsed,
    }


if __name__ == "__main__":
    main()
