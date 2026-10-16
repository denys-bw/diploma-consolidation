from __future__ import annotations

from collections.abc import Iterable


def unique_column_names(names: Iterable[object]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for position, raw in enumerate(names, start=1):
        base = str(raw if raw is not None else "").replace("﻿", "").strip() or f"column_{position}"
        name, suffix = base, 2
        while name in seen:
            name, suffix = f"{base}_{suffix}", suffix + 1
        seen.add(name)
        result.append(name)
    return result


def unique_source_names(names: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for base in names:
        name, suffix = base, 2
        while name in seen:
            name, suffix = f"{base} ({suffix})", suffix + 1
        seen.add(name)
        result.append(name)
    return result


def stem_of(filename: str) -> str:
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    for suffix in (".gz", ".zip"):
        if name.lower().endswith(suffix) and name.count(".") > 1:
            name = name[: -len(suffix)]
    return name.rsplit(".", 1)[0] if "." in name else name
