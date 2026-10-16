from __future__ import annotations

from itertools import combinations

from benchmarks.dataset import DATA, RAW, Dataset, Task, keyed, pairs_from_clusters, record_pair, require
from consolidator.io import read_file, read_files

CHICAGO_FILE = RAW / "consolidation" / "chicago" / "chicago_early_childhood.csv"
CHICAGO_CHECKSUMS = {
    "raw/consolidation/chicago/chicago_early_childhood.csv": "13cbb7b939c4b201f92b09b6e800ae034f1f783d825dda7095fdaccb2793edaa",
}
CHICAGO_SOURCE_CHECKSUMS = {
    "prepared/chicago_sources/CPS_Early_Childhood_Portal_scrape.json": (
        "01f70d8328c7d7db2776f72f4760e50020416bd274eac6158c6cbd63a31a1604"
    ),
    "prepared/chicago_sources/DFSS_AgencySiteLies_2012.xlsx": (
        "6362ca05b82a63f044dfd4c07f38d10fef774d5b7c2a60fbfd9013077999e9b6"
    ),
    "prepared/chicago_sources/ECE Chicago Find a School scrape.json": (
        "0842ba46f09d65e066edab1f51881ca83bef1a27a8403f3b6751d16991340baf"
    ),
    "prepared/chicago_sources/NAEYC_accreditation.csv": (
        "7e31fdd393809daefa6646fb07f7c765bcc2a76ef7223f56905d3d80f54df4e6"
    ),
    "prepared/chicago_sources/chapin_dfss_providers_2011_070212.xlsx": (
        "504f188c583b2285d0483470db2fd055e0daf491add38fdbb75531a67d44e5ce"
    ),
    "prepared/chicago_sources/purple_binder_early_childhood.csv": (
        "574e0152b122622c57fb516c74ee01d08cb5c73490a6ebd1839ede5f8844e549"
    ),
    "prepared/chicago_sources/state_registries.sqlite": (
        "68f0394030c3461095b9ff5e9c85d87379b2fdf5e1c49e31922316845ca03d17"
    ),
}


def load_chicago() -> Dataset:
    frame = keyed(read_file(require(CHICAGO_FILE))[0].frame, "Id")
    truth = frame.pop("True Id")
    return Dataset(
        name="chicago",
        title="Chicago Early Childhood Locations",
        group="consolidation",
        domain="організації: назва, адреса, індекс, телефон",
        task=Task.DEDUP,
        file_format="CSV (9 джерел в одному файлі)",
        sources={"chicago": frame},
        gold=pairs_from_clusters("chicago", truth),
        expected={"records": 3337, "gold": 6608},
        origin="dedupe-examples (DataMade), приклад csv_example",
        notes="9 реальних джерел (колонка Source), 1162 заклади",
        checksums=CHICAGO_CHECKSUMS,
    )


def load_chicago_sources() -> Dataset:
    original = read_file(require(CHICAGO_FILE))[0].frame
    paths = [DATA / relative for relative in sorted(CHICAGO_SOURCE_CHECKSUMS)]
    sources = {source.name: keyed(source.frame, "Id") for source in read_files(map(require, paths))}
    owner = {key: name for name, frame in sources.items() for key in frame.index}
    truth = original.set_index(original["Id"].astype(str))["True Id"]
    gold = set()
    for _, group in truth.groupby(truth):
        keys = sorted(group.index)
        gold.update(record_pair((owner[a], a), (owner[b], b)) for a, b in combinations(keys, 2))
    return Dataset(
        name="chicago_sources",
        title="Chicago: 9 джерел у JSON, Excel, CSV і SQLite",
        group="consolidation",
        domain="організації: назва, адреса, індекс, телефон",
        task=Task.DEDUP,
        file_format="2 JSON, 2 XLSX, 2 CSV, 3 таблиці SQLite",
        sources=sources,
        gold=frozenset(gold),
        expected={"records": 3337, "gold": 6608},
        origin="створено скриптом scripts/prepare_chicago_sources.py з оригінального CSV",
        notes="ті самі записи й еталон, що й у chicago",
        checksums=CHICAGO_CHECKSUMS | CHICAGO_SOURCE_CHECKSUMS,
    )
