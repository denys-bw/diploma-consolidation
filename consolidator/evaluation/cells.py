from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

from consolidator.evaluation.pairs import MatchMetrics

Cell = tuple[int, str]


def error_cells(dirty: pd.DataFrame, clean: pd.DataFrame) -> set[Cell]:
    if dirty.shape != clean.shape:
        raise ValueError("Брудна і чиста таблиці мають різну форму")
    rows, columns = (dirty.to_numpy(dtype=object) != clean.to_numpy(dtype=object)).nonzero()
    return {(int(row), str(dirty.columns[column])) for row, column in zip(rows, columns)}


def evaluate_detection(detected: Iterable[Cell], errors: Iterable[Cell]) -> MatchMetrics:
    detected_set, error_set = set(detected), set(errors)
    true_positives = len(detected_set & error_set)
    return MatchMetrics(true_positives, len(detected_set) - true_positives, len(error_set) - true_positives)


def evaluate_correction(repairs: Mapping[Cell, str], dirty: pd.DataFrame, clean: pd.DataFrame) -> MatchMetrics:
    errors = error_cells(dirty, clean)
    position = {column: index for index, column in enumerate(dirty.columns)}
    clean_values = clean.to_numpy(dtype=object)
    dirty_values = dirty.to_numpy(dtype=object)
    changed = {cell: value for cell, value in repairs.items() if value != dirty_values[cell[0], position[cell[1]]]}
    correct = sum(
        1
        for (row, column), value in changed.items()
        if value == clean_values[row, position[column]] and (row, column) in errors
    )
    return MatchMetrics(correct, len(changed) - correct, len(errors) - correct)
