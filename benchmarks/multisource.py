from __future__ import annotations

import json
import tarfile
import zipfile
from functools import lru_cache
from itertools import combinations

import numpy as np
import pandas as pd

from benchmarks.dataset import RAW, Dataset, Task, keyed, record_pair, require_download
from consolidator.io import Source, read_bytes

FOLDER = RAW / "multisource"
LEIPZIG_PAGE = "https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution"
LEIPZIG_ORIGIN = "групування сутностей з кількох джерел, Лейпцизький університет (Saeedi, Peukert, Rahm, ADBIS 2017)"

SETTLEMENTS_ARCHIVE = "geographicalSettelments.zip"
SETTLEMENTS_RECORDS = "geographicalSettelments/settlements.json"
SETTLEMENTS_CLUSTERS = "geographicalSettelments/combinedSettlements(PerfectMatch).json"
SETTLEMENTS_SOURCE_COLUMN = "data.ontology"
SETTLEMENT_SOURCES = {
    "http://dbpedia.org/": "dbpedia",
    "http://rdf.freebase.com/": "freebase",
    "http://data.nytimes.com/": "nytimes",
    "http://sws.geonames.org/": "geonames",
}
SETTLEMENTS_CHECKSUM = "76a5b8d261607f6c77a9cd79eaf4b0bb60e8b4ebfad3802b4bd01a02d97cc02e"

MUSICBRAINZ_FILE = "musicbrainz-20-A01.csv.dapo"
MUSICBRAINZ_CHECKSUM = "527a94f24f7e813a9bc3fef35a635f13e195516966b308140a0dd2926afbb97d"
MUSICBRAINZ_KEY = "TID"
MUSICBRAINZ_ENTITY = "CID"
MUSICBRAINZ_SOURCE = "SourceID"
MUSICBRAINZ_SERVICE_COLUMNS = ("CTID", "id")

NC_VOTERS_ARCHIVE = "5Party-ocp20.tar.gz"
NC_VOTERS_CHECKSUM = "4cecc0ddba08793befbc448696268bcf8a26e22e7fd7ff20a3cba327d9e1fd6c"
NC_VOTERS_ENTITY = "recid"
NC_VOTERS_SOURCES = 5
NC_VOTERS_RECORDS_PER_SOURCE = 1_000_000
NC_VOTERS_PAIRS = 3_331_384
NC_VOTERS_SAMPLE = 20_000
SAMPLE_SEED = 0

SOURCE = "__source"
KEY = "__key"
ENTITY = "__entity"


def load_settlements() -> Dataset:
    require_download(FOLDER, [SETTLEMENTS_ARCHIVE], [LEIPZIG_PAGE])
    with zipfile.ZipFile(FOLDER / SETTLEMENTS_ARCHIVE) as zipped:
        table = read_bytes(zipped.read(SETTLEMENTS_RECORDS), SETTLEMENTS_RECORDS)[0]
        clusters = [json.loads(line) for line in zipped.read(SETTLEMENTS_CLUSTERS).decode("utf-8").splitlines() if line]
    source = Source(table.name, keyed(table.frame, "id"), LEIPZIG_ORIGIN, "JSON Lines")
    sources = {SETTLEMENT_SOURCES[part.name]: part.frame for part in source.split_by(SETTLEMENTS_SOURCE_COLUMN)}
    owner = {key: name for name, frame in sources.items() for key in frame.index}
    gold = set()
    for cluster in clusters:
        members = sorted({str(member) for member in cluster["data"]["clusteredVertices"]})
        gold.update(record_pair((owner[a], a), (owner[b], b)) for a, b in combinations(members, 2))
    return Dataset(
        name="settlements",
        title="Geographic Settlements",
        group="multisource",
        domain="населені пункти: назва, координати, тип",
        task=Task.MULTI,
        file_format="JSON Lines (вкладені поля) в архіві zip; джерело запису — поле ontology",
        sources=sources,
        gold=frozenset(gold),
        expected={"dbpedia": 774, "freebase": 776, "nytimes": 755, "geonames": 749, "gold": 4391},
        origin=LEIPZIG_ORIGIN,
        notes="4 джерела (DBpedia, Freebase, NYTimes, Geonames), 820 груп; назви різними мовами й у різному написанні",
        clean_sources=True,
        checksums={f"raw/multisource/{SETTLEMENTS_ARCHIVE}": SETTLEMENTS_CHECKSUM},
    )


def load_musicbrainz() -> Dataset:
    require_download(FOLDER, [MUSICBRAINZ_FILE], [LEIPZIG_PAGE])
    frame = read_bytes((FOLDER / MUSICBRAINZ_FILE).read_bytes(), MUSICBRAINZ_FILE.removesuffix(".dapo"))[0].frame
    table = pd.DataFrame(
        {
            SOURCE: "source_" + frame[MUSICBRAINZ_SOURCE].astype(str),
            KEY: frame[MUSICBRAINZ_KEY].astype(str),
            ENTITY: frame[MUSICBRAINZ_ENTITY].astype(str),
        }
    ).join(frame.drop(columns=[MUSICBRAINZ_SOURCE, MUSICBRAINZ_KEY, MUSICBRAINZ_ENTITY, *MUSICBRAINZ_SERVICE_COLUMNS]))
    sources, gold = split_sources(table)
    return Dataset(
        name="musicbrainz",
        title="MusicBrainz 20K",
        group="multisource",
        domain="пісні: назва, виконавець, альбом, рік, тривалість, мова",
        task=Task.MULTI,
        file_format="CSV (джерело запису — колонка SourceID)",
        sources=sources,
        gold=gold,
        expected={
            **{f"source_{index}": size for index, size in enumerate((3827, 3933, 3926, 3861, 3828), 1)},
            "gold": 16250,
        },
        origin=f"{LEIPZIG_ORIGIN}; дублікати згенеровано DAPO (Hildebrandt et al., IEEE Trans. Big Data)",
        notes="10 000 пісень, у половини — копії в 2–5 джерелах з сильними помилками; формати полів у джерелах різні",
        clean_sources=True,
        checksums={f"raw/multisource/{MUSICBRAINZ_FILE}": MUSICBRAINZ_CHECKSUM},
    )


def load_nc_voters(entities: int | None = None, duplicated: float | None = None, seed: int = SAMPLE_SEED) -> Dataset:
    table = nc_voters_table()
    name, title, expected = "nc_voters", "North Carolina Voters 5M", {}
    if entities is None:
        expected = {**dict.fromkeys(_nc_voters_sources(), NC_VOTERS_RECORDS_PER_SOURCE), "gold": NC_VOTERS_PAIRS}
    else:
        table = table[table[ENTITY].isin(sample_entities(table[ENTITY], entities, duplicated, seed))]
        share = "природна частка" if duplicated is None else f"{duplicated:.0%} з копіями"
        name = f"nc_voters_{entities}" if duplicated is None else f"nc_voters_{entities}_{round(duplicated * 100)}"
        title = f"North Carolina Voters: {entities} людей, {share}"
    sources, gold = split_sources(table)
    return Dataset(
        name=name,
        title=title,
        group="multisource",
        domain="люди: ім'я, прізвище, населений пункт, поштовий індекс",
        task=Task.MULTI,
        file_format="5 CSV-файлів в архіві tar.gz",
        sources=sources,
        gold=gold,
        expected=expected,
        origin=f"{LEIPZIG_ORIGIN}; копії згенеровано GeCo (Tran, Vatsalan, Christen, CIKM 2013), 20 % помилок",
        notes="кожне джерело без внутрішніх повторів; частина копій точні, частина з помилками",
        clean_sources=True,
        checksums={f"raw/multisource/{NC_VOTERS_ARCHIVE}": NC_VOTERS_CHECKSUM},
    )


def load_nc_voters_sample() -> Dataset:
    return load_nc_voters(NC_VOTERS_SAMPLE)


@lru_cache(maxsize=1)
def nc_voters_table() -> pd.DataFrame:
    require_download(FOLDER, [NC_VOTERS_ARCHIVE], [LEIPZIG_PAGE])
    parts = []
    with tarfile.open(FOLDER / NC_VOTERS_ARCHIVE, "r:gz") as archive:
        members = sorted((member for member in archive.getmembers() if member.isfile()), key=lambda item: item.name)
        for name, member in zip(_nc_voters_sources(), members):
            handle = archive.extractfile(member)
            if handle is None:
                continue
            frame = pd.read_csv(handle, dtype=str, keep_default_na=False)
            entity = frame.pop(NC_VOTERS_ENTITY)
            keys = pd.Series(np.arange(len(frame)).astype(str), index=frame.index)
            parts.append(pd.DataFrame({SOURCE: name, KEY: keys, ENTITY: entity}).join(frame))
    return pd.concat(parts, ignore_index=True)


def sample_entities(entities: pd.Series, count: int, duplicated: float | None, seed: int) -> np.ndarray:
    sizes = entities.value_counts().sort_index()
    generator = np.random.default_rng(seed)
    if duplicated is None:
        return generator.choice(sizes.index.to_numpy(), min(count, len(sizes)), replace=False)
    repeated = sizes.index[sizes > 1].to_numpy()
    single = sizes.index[sizes == 1].to_numpy()
    with_copies = round(count * duplicated)
    if with_copies > len(repeated) or count - with_copies > len(single):
        raise ValueError(f"у наборі немає {with_copies} об'єктів з копіями і {count - with_copies} без копій")
    return np.concatenate(
        [
            generator.choice(repeated, with_copies, replace=False),
            generator.choice(single, count - with_copies, replace=False),
        ]
    )


def _nc_voters_sources() -> list[str]:
    return [f"ncvr_{index}" for index in range(NC_VOTERS_SOURCES)]


def split_sources(table: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], frozenset]:
    sources = {}
    for name, part in table.groupby(SOURCE, sort=True):
        frame = part.drop(columns=[SOURCE, ENTITY]).set_index(KEY)
        frame.index.name = None
        sources[str(name)] = frame
    return sources, _gold(table)


def _gold(table: pd.DataFrame) -> frozenset:
    codes = pd.factorize(table[ENTITY])[0]
    order = np.argsort(codes, kind="stable")
    ordered = codes[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]]) if len(ordered) else np.zeros(0, dtype=np.int64)
    sizes = np.diff(np.r_[starts, len(ordered)])
    names, keys = table[SOURCE].to_numpy(), table[KEY].to_numpy()
    gold = set()
    for size in np.unique(sizes[sizes > 1]):
        members = order[starts[sizes == size][:, None] + np.arange(size)]
        for a, b in combinations(range(int(size)), 2):
            gold.update(
                record_pair((names[left], keys[left]), (names[right], keys[right]))
                for left, right in zip(members[:, a].tolist(), members[:, b].tolist())
            )
    return frozenset(gold)
