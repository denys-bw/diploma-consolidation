from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyreadr


def convert(rdata_folder: Path, output_folder: Path) -> None:
    records = pyreadr.read_r(rdata_folder / "restaurant.RData")["restaurant"]
    gold = pyreadr.read_r(rdata_folder / "restaurant_gold.RData")["restaurant_gold"]
    records = records.astype({"id": int, "class": int})
    gold = pd.DataFrame(
        {
            "class": gold["class"].astype(str).str.strip("'").astype(int),
            "id1": gold["id_1"].astype(int),
            "id2": gold["id_2"].astype(int),
        }
    )
    output_folder.mkdir(parents=True, exist_ok=True)
    records.to_csv(output_folder / "restaurants.csv", index=False)
    gold.to_csv(output_folder / "restaurants_gold.csv", index=False)
    print(f"restaurants.csv: {len(records)} записів; restaurants_gold.csv: {len(gold)} пар -> {output_folder}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Використання: python -m scripts.convert_restaurants_rdata <папка з .RData> <папка для CSV>")
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
