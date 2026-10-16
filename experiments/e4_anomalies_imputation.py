from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from benchmarks import Dataset
from consolidator.evaluation import MatchMetrics, evaluate_detection
from consolidator.fusion import Fuser, MajorityVote, WeightedVote
from consolidator.imputation import AdaptiveImputer, KnnImputer, MeanImputer, impute
from consolidator.pipeline import Pipeline, PipelineResult
from consolidator.profiling import OUTLIER_DETECTORS, ColumnType, detect_outliers, quantitative_frame
from consolidator.reporting.labels import OUTLIER_LABELS
from consolidator.standardization import FormatPolicy, StandardizationSettings
from consolidator.values.missing import missing_mask
from experiments.common import available, pipeline_settings, save, sources_of

FLIGHTS = "flights"
GAPS_DATASET = "chicago_sources"
LIBRARY_NEIGHBOURS = KNNImputer().n_neighbors
IMPUTERS = {
    "середнє значення колонки": MeanImputer(),
    f"k найближчих сусідів, k = {LIBRARY_NEIGHBOURS} (типове значення scikit-learn)": KnnImputer(LIBRARY_NEIGHBOURS),
    "метод і k обрано перехресною перевіркою (типово в програмі)": AdaptiveImputer(),
}


def main() -> None:
    for dataset in available([FLIGHTS]):
        result = run_flights(dataset)
        detection, imputation = flights_tables(dataset, result)
        save(detection, "e4_flights_detection.csv")
        save(imputation, "e4_flights_imputation.csv")
    for dataset in available([GAPS_DATASET]):
        save(chicago_gaps(dataset), "e4_chicago_gaps.csv")


def run_flights(dataset: Dataset) -> PipelineResult:
    settings = pipeline_settings(standardization=StandardizationSettings(policy=FormatPolicy.DOMINANT))
    return Pipeline(settings).run(sources_of(dataset))


def flights_tables(dataset: Dataset, result: PipelineResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = result.records
    times = [attribute for attribute in result.attributes if records.types[attribute] is ColumnType.TIME]
    rows = records.keys.astype(int)
    clean = dataset.sources["clean"][times].iloc[rows].reset_index(drop=True)
    observed = records.frame[times].reset_index(drop=True)
    present = observed != ""
    wrong = present & (observed != clean)
    errors = _cells(wrong)
    groups = result.fusion.entity_ids
    minutes = quantitative_frame(observed, dict.fromkeys(times, ColumnType.TIME))
    detection = []
    for detector in OUTLIER_DETECTORS:
        for scope, labels in (("уся колонка", None), ("у межах сутності", groups)):
            flags = detect_outliers(minutes, detector, labels) & present
            detection.append(
                _detection_row(f"{OUTLIER_LABELS[detector.key]}, {scope}", evaluate_detection(_cells(flags), errors))
            )
    for resolver, label in ((MajorityVote(), "більшість голосів"), (WeightedVote(), "зважене голосування")):
        fusion = Fuser(dict.fromkeys(times, resolver)).fuse(records, groups)
        conflicts = fusion.conflicting_cells()[times].reset_index(drop=True)
        detection.append(
            _detection_row(
                f"розбіжність із узгодженим значенням: {label}", evaluate_detection(_cells(conflicts), errors)
            )
        )
    detection = pd.DataFrame(detection)
    detection.insert(1, "помилкових значень", len(errors))

    missing = ~present
    candidates = {}
    settings = result.settings.standardization
    for label, imputer in IMPUTERS.items():
        candidates[label] = impute(observed, dict.fromkeys(times, ColumnType.TIME), imputer, settings).frame
    candidates["узгоджене значення дублікатів"] = result.fusion.record_frame()[times].reset_index(drop=True)
    pipeline = "дублікати, потім відновлення, обране перехресною перевіркою (конвеєр)"
    candidates[pipeline] = result.record_values()[times].reset_index(drop=True)
    truth = quantitative_frame(clean, dict.fromkeys(times, ColumnType.TIME))
    imputation = pd.DataFrame(
        [_imputation_row(label, filled, missing, clean, truth, times) for label, filled in candidates.items()]
    )
    imputation.insert(1, "пропусків", int(missing.to_numpy().sum()))
    return detection, imputation


def chicago_gaps(dataset: Dataset) -> pd.DataFrame:
    result = Pipeline(pipeline_settings()).run(sources_of(dataset))
    records = result.records
    names = np.asarray(records.source_names, dtype=object)
    fused = result.fusion.record_frame()
    imputed = result.imputation.counts() if result.imputation is not None else pd.Series(dtype=int)
    rows = []
    for attribute in result.attributes:
        provided = np.isin(names, result.mapping.attribute(attribute).sources)[records.source_codes]
        gaps = provided & (records.values(attribute) == "")
        restored = gaps & (fused[attribute].to_numpy() != "")
        applicable = result.applicable[attribute].to_numpy()
        remaining = applicable & missing_mask(result.consolidated[attribute])
        rows.append(
            {
                "атрибут": attribute,
                "тип": result.mapping.types[attribute].value,
                "джерел з атрибутом": len(result.mapping.attribute(attribute).sources),
                "записів з атрибутом": int(provided.sum()),
                "пропусків у записах": int(gaps.sum()),
                "заповнено значенням дубліката": int(restored.sum()),
                "відновлено статистично": int(imputed.get(attribute, 0)),
                "сутностей з атрибутом": int(applicable.sum()),
                "пропусків у консолідованій таблиці": int(remaining.sum()),
            }
        )
    return pd.DataFrame(rows)


def _detection_row(label: str, metrics: MatchMetrics) -> dict:
    return {
        "метод": label,
        "позначено": metrics.found,
        "правильно позначено": metrics.true_positives,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "F1": metrics.f1,
    }


def _imputation_row(
    label: str,
    filled: pd.DataFrame,
    missing: pd.DataFrame,
    clean: pd.DataFrame,
    truth: pd.DataFrame,
    times: list[str],
) -> dict:
    target = missing & (filled != "")
    exact = int((target & (filled == clean)).to_numpy().sum())
    estimates = quantitative_frame(filled, dict.fromkeys(times, ColumnType.TIME))
    errors = (estimates - truth).abs().where(target).stack()
    total = int(missing.to_numpy().sum())
    return {
        "метод": label,
        "заповнено": int(target.to_numpy().sum()),
        "точно відновлено": exact,
        "частка точно відновлених": exact / total if total else 0.0,
        "середня похибка, хв": float(errors.mean()) if len(errors) else np.nan,
        "медіанна похибка, хв": float(errors.median()) if len(errors) else np.nan,
    }


def _cells(flags: pd.DataFrame) -> set[tuple[int, str]]:
    return {(int(row), str(column)) for column in flags.columns for row in np.flatnonzero(flags[column].to_numpy())}


if __name__ == "__main__":
    main()
