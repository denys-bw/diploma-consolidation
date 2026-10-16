from pathlib import Path

import pandas as pd

from benchmarks import REGISTRY, NeedsDownload, check

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    rows = []
    for name, loader in REGISTRY.items():
        try:
            dataset = loader()
        except NeedsDownload as error:
            print(f"[потрібно скачати] {name}: {error}")
            continue
        problems = check(dataset)
        row = dataset.summary()
        row["перевірка"] = "OK" if not problems else "; ".join(problems)
        rows.append(row)
        status = "OK" if not problems else "РОЗБІЖНІСТЬ"
        notes = f" ({dataset.notes})" if dataset.notes else ""
        details = "" if not problems else f" — {row['перевірка']}"
        print(
            f"[{status}] {dataset.title}: {row['записів']}; еталон — {row['еталон']} {row['одиниця еталону']}{notes}{details}"
        )
    RESULTS.mkdir(exist_ok=True)
    output = RESULTS / "e0_datasets.csv"
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    print(f"\nТаблицю збережено: {output}")


if __name__ == "__main__":
    main()
