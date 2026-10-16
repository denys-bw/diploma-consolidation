from __future__ import annotations

from collections.abc import Callable

from benchmarks.consolidation import load_chicago
from benchmarks.dataset import DATA, Dataset, Task, sha256_of
from benchmarks.linkage import load_abt_buy, load_amazon_google, load_dblp_acm, load_dblp_scholar, load_restaurants
from benchmarks.multisource import load_musicbrainz, load_nc_voters, load_nc_voters_sample, load_settlements
from benchmarks.quality import load_beers, load_flights

ERROR_RATE_TOLERANCE = 0.01

REGISTRY: dict[str, Callable[[], Dataset]] = {
    "chicago": load_chicago,
    "restaurants": load_restaurants,
    "abt_buy": load_abt_buy,
    "amazon_google": load_amazon_google,
    "dblp_acm": load_dblp_acm,
    "dblp_scholar": load_dblp_scholar,
    "flights": load_flights,
    "beers": load_beers,
    "settlements": load_settlements,
    "musicbrainz": load_musicbrainz,
    "nc_voters": load_nc_voters,
    "nc_voters_sample": load_nc_voters_sample,
}


def load_dataset(name: str) -> Dataset:
    if name not in REGISTRY:
        raise KeyError(f"Невідомий набір «{name}». Доступні: {', '.join(REGISTRY)}")
    return REGISTRY[name]()


def check(dataset: Dataset) -> list[str]:
    actual: dict[str, object] = {"gold": len(dataset.gold)}
    if dataset.task is Task.CLEAN:
        dirty = dataset.sources["dirty"]
        actual.update(rows=dirty.shape[0], cols=dirty.shape[1])
    elif dataset.task is Task.DEDUP:
        actual.update(records=dataset.n_records)
    else:
        actual.update({source: len(frame) for source, frame in dataset.sources.items()})
    if "published" in dataset.extra:
        actual["published"] = {key: len(pairs) for key, pairs in dataset.extra["published"].items()}
    problems = [
        f"{relative}: контрольна сума SHA-256 не збігається з очікуваною"
        for relative, checksum in dataset.checksums.items()
        if sha256_of(DATA / relative) != checksum
    ]
    for key, value in dataset.expected.items():
        if key == "error_rate":
            rate = len(dataset.gold) / dataset.sources["dirty"].size
            if abs(rate - value) > ERROR_RATE_TOLERANCE:
                problems.append(f"частка помилок {rate:.3f}, у статті {value}")
        elif actual.get(key) != value:
            problems.append(f"{key}: очікувалось {value}, отримано {actual.get(key)}")
    if dataset.task is not Task.CLEAN:
        known = {(source, str(key)) for source, frame in dataset.sources.items() for key in frame.index}
        missing = sum(1 for left, right in dataset.gold if left not in known or right not in known)
        if missing:
            problems.append(f"{missing} еталонних пар посилаються на відсутні записи")
    return problems
