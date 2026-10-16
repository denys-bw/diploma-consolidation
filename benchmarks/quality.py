from __future__ import annotations

from benchmarks.dataset import RAW, Dataset, Task, require
from consolidator.evaluation.cells import error_cells
from consolidator.io import read_file


def load_flights() -> Dataset:
    return _cleaning(
        "flights",
        "Flights",
        "авіарейси: час вильоту і прильоту за даними 38 сайтів",
        {"rows": 2376, "cols": 7, "error_rate": 0.30, "gold": 4920},
        "Li, Dong et al. (2012); підготовка — Mahdavi, Abedjan et al. (Raha, 2019; Baran, 2020)",
        "38 джерел (колонка src) по-різному пишуть той самий час; є пропуски",
        {
            "clean.csv": "0acfcfd8985b06fdd363965c9e8d9522c43e7589a93d79ae7dc311e1c37fdf3b",
            "dirty.csv": "1b5c1afa10aa0e7c20fd7e14d05c56772715b2771aa0f5fa67ed1709e1eecd46",
        },
        clean_sources=True,
        layout={"source_column": "src", "row_column": "tuple_id", "entity_column": "flight"},
    )


def load_beers() -> Dataset:
    return _cleaning(
        "beers",
        "Beers",
        "крафтове пиво: об'єм, міцність, гіркота, пивоварня, місто",
        {"rows": 2410, "cols": 11, "error_rate": 0.16, "gold": 4362},
        "сайт craftcans.com; підготовка — Mahdavi, Abedjan et al. (Raha, 2019; Baran, 2020)",
        "одиниці виміру записано по-різному («12.0 oz», «0.09%»), пропуски «N/A», місто разом зі штатом",
        {
            "clean.csv": "373227df59ad197e154dd5149125789e415019535c7223355e9486ee1b3b93de",
            "dirty.csv": "7110bf4931a9445a1675e544d6c996817c739136239f8a2b02e088c7ec0a1f68",
        },
    )


def _cleaning(
    name: str,
    title: str,
    domain: str,
    expected: dict,
    origin: str,
    notes: str,
    checksums: dict[str, str],
    clean_sources: bool = False,
    layout: dict[str, str] | None = None,
) -> Dataset:
    folder = RAW / "quality" / name
    dirty = read_file(require(folder / "dirty.csv"))[0].frame
    clean = read_file(require(folder / "clean.csv"))[0].frame
    if dirty.shape != clean.shape:
        raise ValueError(f"{name}: брудна і чиста версії мають різну форму")
    clean.columns = dirty.columns
    return Dataset(
        name=name,
        title=title,
        group="quality",
        domain=domain,
        task=Task.CLEAN,
        file_format="CSV",
        sources={"dirty": dirty, "clean": clean},
        gold=frozenset(error_cells(dirty, clean)),
        expected=expected,
        origin=origin,
        notes=notes,
        clean_sources=clean_sources,
        extra={"layout": layout or {}},
        checksums={f"raw/quality/{name}/{file}": checksum for file, checksum in checksums.items()},
    )
