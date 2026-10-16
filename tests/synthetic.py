from __future__ import annotations

import numpy as np
import pandas as pd

SYLLABLES = ["ka", "lo", "mi", "ra", "to", "ve", "sa", "ni", "do", "be", "ru", "an", "el", "or", "ta", "zi"]
STREETS = ["Main", "Oak", "Pine", "Maple", "Cedar", "Elm", "Lake", "Hill", "River", "Park"]
CITIES = ["Springfield", "Riverton", "Lakeside", "Fairview", "Greenville", "Bristol"]


def entities(count: int, seed: int = 0) -> pd.DataFrame:
    generator = np.random.default_rng(seed)

    def word(size: int) -> str:
        return "".join(generator.choice(SYLLABLES, size)).capitalize()

    return pd.DataFrame(
        {
            "name": [f"{word(2)} {word(3)}" for _ in range(count)],
            "street": [f"{generator.integers(1, 999)} {generator.choice(STREETS)} St" for _ in range(count)],
            "city": [str(generator.choice(CITIES)) for _ in range(count)],
            "price": [f"{generator.uniform(5, 500):.2f}" for _ in range(count)],
            "phone": [
                f"({generator.integers(200, 999)}) {generator.integers(200, 999)}-{generator.integers(1000, 9999)}"
                for _ in range(count)
            ],
        }
    )


def typo(generator: np.random.Generator, text: str) -> str:
    if len(text) < 4:
        return text
    position = int(generator.integers(1, len(text) - 1))
    return text[:position] + text[position + 1 :]


def linked_sources(
    count: int, shared: float = 0.5, seed: int = 0, internal: float = 0.0
) -> tuple[dict[str, pd.DataFrame], set[tuple[tuple[str, str], tuple[str, str]]]]:
    generator = np.random.default_rng(seed + 1)
    base = entities(count, seed)
    overlap = int(count * shared)
    left_ids = list(range((count + overlap) // 2))
    right_ids = list(range((count - overlap) // 2, count))
    left = base.iloc[left_ids].copy()
    left.index = [f"a{index}" for index in left_ids]
    right = base.iloc[right_ids].copy()
    right.index = [f"b{index}" for index in right_ids]
    right["name"] = [typo(generator, value) if generator.random() < 0.5 else value for value in right["name"]]
    right["street"] = [value.replace(" St", " Street") for value in right["street"]]
    members: dict[int, list[tuple[str, str]]] = {}
    for index in left_ids:
        members.setdefault(index, []).append(("left", f"a{index}"))
    for index in right_ids:
        members.setdefault(index, []).append(("right", f"b{index}"))
    if internal:
        repeated = generator.choice(left_ids, int(len(left_ids) * internal), replace=False)
        copies = base.iloc[repeated].copy()
        copies["name"] = [typo(generator, value) for value in copies["name"]]
        copies.index = [f"a{index}x" for index in repeated]
        left = pd.concat([left, copies])
        for index in repeated:
            members[int(index)].append(("left", f"a{index}x"))
    gold = set()
    for records in members.values():
        for first in range(len(records)):
            for second in range(first + 1, len(records)):
                gold.add(tuple(sorted((records[first], records[second]))))
    return {"left": left, "right": right}, gold
