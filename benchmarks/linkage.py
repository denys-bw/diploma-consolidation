from __future__ import annotations

import zipfile

from benchmarks.dataset import RAW, Dataset, Task, keyed, record_pair, require, require_download
from consolidator.io import read_bytes, read_file

LEIPZIG_PAGE = "https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution"
RESTAURANT_CHECKSUMS = {
    "raw/linkage/restaurants/restaurants.csv": "495e2797f01a630c7024a6382f1bf107b628d4c7cfcf3c49c2f5cba9157b1b4e",
    "raw/linkage/restaurants/restaurants_gold.csv": "5e468a3bb50a6979be7bd0765f867b3097ee1e42c0adff62f88c515ab667582b",
}
LEIPZIG_ORIGIN = "бенчмарки Лейпцизького університету (Köpcke, Thor, Rahm, 2010); офіційний архів з сайту університету"


def load_restaurants() -> Dataset:
    folder = RAW / "linkage" / "restaurants"
    records = keyed(read_file(require(folder / "restaurants.csv"))[0].frame, "id").drop(columns=["class"])
    gold_frame = read_file(require(folder / "restaurants_gold.csv"))[0].frame
    is_fodors = records["phone"].str.contains("/", regex=False)
    fodors, zagats = records[is_fodors], records[~is_fodors]
    owner = dict.fromkeys(fodors.index, "fodors") | dict.fromkeys(zagats.index, "zagats")
    gold = frozenset(
        record_pair((owner[str(a)], str(a)), (owner[str(b)], str(b)))
        for a, b in zip(gold_frame["id1"], gold_frame["id2"])
    )
    return Dataset(
        name="restaurants",
        title="Restaurants (Fodors-Zagats)",
        group="linkage",
        domain="ресторани: назва, адреса, місто, телефон, тип кухні",
        task=Task.LINK,
        file_format="CSV (перетворено з RData пакета restaurant)",
        sources={"fodors": fodors, "zagats": zagats},
        gold=gold,
        expected={"fodors": 533, "zagats": 331, "gold": 112},
        origin="пакет restaurant (cleanzr, CC0); оригінал — Tejada, Knoblock, Minton (2001)",
        notes="джерело запису визначено за форматом телефону: 533 Fodor's / 331 Zagat's",
        clean_sources=True,
        checksums=RESTAURANT_CHECKSUMS,
    )


def load_abt_buy() -> Dataset:
    return _leipzig(
        "abt_buy",
        "Abt-Buy.zip",
        ("abt", "Abt.csv"),
        ("buy", "Buy.csv"),
        "abt_buy_perfectMapping.csv",
        "Abt-Buy",
        "товари двох інтернет-магазинів: назва, опис, ціна",
        {"abt": 1081, "buy": 1092, "gold": 1097},
        "b9ff2937d97371b7a00b0a97033213bfbdee24e8f1a0718a41ea8364472a8ff8",
        clean_sources=True,
    )


def load_amazon_google() -> Dataset:
    return _leipzig(
        "amazon_google",
        "Amazon-GoogleProducts.zip",
        ("amazon", "Amazon.csv"),
        ("google", "GoogleProducts.csv"),
        "Amzon_GoogleProducts_perfectMapping.csv",
        "Amazon-Google",
        "товари: назва, опис, виробник, ціна",
        {"amazon": 1363, "google": 3226, "gold": 1300},
        "fcd72e223d51ce5b0d6b1d68034a438ba95cf2940431e209a57910b067cc4811",
    )


def load_dblp_acm() -> Dataset:
    return _leipzig(
        "dblp_acm",
        "DBLP-ACM.zip",
        ("dblp", "DBLP2.csv"),
        ("acm", "ACM.csv"),
        "DBLP-ACM_perfectMapping.csv",
        "DBLP-ACM",
        "наукові статті: назва, автори, видання, рік",
        {"dblp": 2616, "acm": 2294, "gold": 2224},
        "e37e7ed8e06722499e6d9e94583fdec279207b13ce6270a37595b6a5d594bc40",
        clean_sources=True,
        notes="2616 записів DBLP — як в офіційному архіві й у таблиці 1 статті Köpcke et al. (2010); "
        "на вебсторінці набору одруківка: 2614",
    )


def load_dblp_scholar() -> Dataset:
    return _leipzig(
        "dblp_scholar",
        "DBLP-Scholar.zip",
        ("dblp", "DBLP1.csv"),
        ("scholar", "Scholar.csv"),
        "DBLP-Scholar_perfectMapping.csv",
        "DBLP-Scholar",
        "наукові статті у великому масштабі",
        {"dblp": 2616, "scholar": 64263, "gold": 5347},
        "80b9ddbe89d6b78ab583f801e86a8c451e94dbf1a57ba88dfbd180245cff138b",
    )


def _leipzig(
    name: str,
    archive: str,
    left: tuple[str, str],
    right: tuple[str, str],
    mapping_file: str,
    title: str,
    domain: str,
    expected: dict,
    checksum: str,
    clean_sources: bool = False,
    notes: str = "",
) -> Dataset:
    folder = RAW / "linkage" / name
    require_download(folder, [archive], [LEIPZIG_PAGE])
    with zipfile.ZipFile(folder / archive) as zipped:
        sources = {
            source: keyed(read_bytes(zipped.read(member), member)[0].frame, "id") for source, member in (left, right)
        }
        mapping = read_bytes(zipped.read(mapping_file), mapping_file)[0].frame
    gold = frozenset(
        record_pair((left[0], str(a)), (right[0], str(b))) for a, b in mapping.iloc[:, :2].itertuples(index=False)
    )
    return Dataset(
        name=name,
        title=title,
        group="linkage",
        domain=domain,
        task=Task.LINK,
        file_format="CSV в архіві zip",
        sources=sources,
        gold=gold,
        expected=expected,
        origin=LEIPZIG_ORIGIN,
        notes=notes,
        clean_sources=clean_sources,
        checksums={f"raw/linkage/{name}/{archive}": checksum},
    )
