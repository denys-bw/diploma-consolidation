from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from pathlib import Path

import pandas as pd

from consolidator.linkage.records import PairScope, RecordId

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"

GROUPS = {
    "consolidation": "консолідація з різних джерел",
    "linkage": "класичні бенчмарки зв'язування",
    "quality": "якість даних: помилки і чиста версія",
    "multisource": "групування сутностей з кількох джерел",
}

RecordPair = tuple[RecordId, RecordId]


class Task(str, Enum):
    LINK = "link"
    MULTI = "multi"
    DEDUP = "dedup"
    CLEAN = "clean"

    @property
    def label(self) -> str:
        return {
            Task.LINK: "зв'язування двох джерел",
            Task.MULTI: "консолідація кількох джерел",
            Task.DEDUP: "дублікати в одній таблиці",
            Task.CLEAN: "виявлення і виправлення помилок",
        }[self]


class NeedsDownload(FileNotFoundError):
    pass


@dataclass(frozen=True, eq=False)
class Dataset:
    name: str
    title: str
    group: str
    domain: str
    task: Task
    file_format: str
    sources: dict[str, pd.DataFrame]
    gold: frozenset
    expected: dict
    origin: str
    notes: str = ""
    extra: dict = field(default_factory=dict)
    clean_sources: bool = False
    checksums: dict[str, str] = field(default_factory=dict)

    @property
    def scope(self) -> PairScope:
        return PairScope.CROSS if self.task in (Task.LINK, Task.MULTI) else PairScope.ALL

    @property
    def n_records(self) -> int:
        if self.task is Task.CLEAN:
            return len(self.sources["dirty"])
        return sum(len(frame) for frame in self.sources.values())

    @property
    def n_all_pairs(self) -> int:
        sizes = [len(frame) for frame in self.sources.values()]
        if self.task in (Task.LINK, Task.MULTI):
            return sum(a * b for a, b in combinations(sizes, 2))
        if self.task is Task.DEDUP:
            total = sum(sizes)
            return total * (total - 1) // 2
        return 0

    def summary(self) -> dict[str, object]:
        if self.task is Task.CLEAN:
            dirty = self.sources["dirty"]
            records, unit = f"{dirty.shape[0]} рядків × {dirty.shape[1]} колонок", "клітинок з помилками"
        else:
            records, unit = " + ".join(str(len(frame)) for frame in self.sources.values()), "пар дублікатів"
        return {
            "набір": self.name,
            "група": GROUPS[self.group],
            "назва": self.title,
            "домен": self.domain,
            "задача": self.task.label,
            "формат": self.file_format,
            "записів": records,
            "еталон": len(self.gold),
            "одиниця еталону": unit,
            "пар без блокування": self.n_all_pairs or "",
            "джерела без внутрішніх дублікатів": "так" if self.clean_sources else "ні",
        }


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record_pair(left: RecordId, right: RecordId) -> RecordPair:
    return (left, right) if left <= right else (right, left)


def pairs_from_clusters(source: str, entities: pd.Series) -> frozenset:
    pairs = set()
    for _, group in entities.groupby(entities):
        keys = sorted(str(key) for key in group.index)
        pairs.update(record_pair((source, a), (source, b)) for a, b in combinations(keys, 2))
    return frozenset(pairs)


def require(path: Path) -> Path:
    if not path.exists():
        raise NeedsDownload(f"Немає файлу {path}")
    return path


def require_download(folder: Path, files: Iterable[str], urls: Iterable[str]) -> None:
    missing = [name for name in files if not (folder / name).exists()]
    if missing:
        raise NeedsDownload(
            f"у папці {folder.relative_to(ROOT).as_posix()} бракує: {', '.join(missing)}. Скачати: {' ; '.join(urls)}"
        )


def keyed(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    keys = frame[column].astype(str)
    result = frame.drop(columns=column)
    result.index = pd.Index(keys.to_numpy(), name=column)
    return result
