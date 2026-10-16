from __future__ import annotations

import numpy as np
import pandas as pd

from consolidator.linkage import LinkageResult, MatchStatus
from consolidator.pipeline import PipelineResult
from consolidator.profiling import OUTLIER_DETECTORS, ColumnType, TableProfile, profile_frame
from consolidator.reporting.labels import (
    CLUSTERER_LABELS,
    CRITERION_LABELS,
    IMPUTER_LABELS,
    INDEX_LABEL,
    MATCH_STATUS_LABELS,
    OUTLIER_LABELS,
    POLICY_CHOICE_LABELS,
    POLICY_LABELS,
    REASON_LABELS,
    RESOLVER_LABELS,
    SCORER_LABELS,
    STAGE_LABELS,
    STATUS_LABELS,
    TYPE_LABELS,
    provenance_label,
)
from consolidator.standardization import CellStatus, standardizer_for
from consolidator.values.numbers import format_number

DECIMALS = 4
RESULT_PROFILE = "консолідовані дані"


def consolidated_table(result: PipelineResult) -> pd.DataFrame:
    frame = result.consolidated.rename(columns={"sources": "Джерела", "records": "Записів"})
    return frame.reset_index().rename(columns={"entity": "Сутність"})


def provenance_table(result: PipelineResult) -> pd.DataFrame:
    frame = result.provenance.map(provenance_label)
    return frame.reset_index().rename(columns={"entity": "Сутність"})


def records_table(result: PipelineResult) -> pd.DataFrame:
    frames = [frame.reindex(columns=result.attributes) for frame in result.prepared.frames.values()]
    values = pd.concat(frames, ignore_index=True).fillna("") if frames else pd.DataFrame(columns=result.attributes)
    membership = result.fusion.membership.rename(columns={"entity": "Сутність", "source": "Джерело", "key": "Запис"})
    return pd.concat([membership.reset_index(drop=True), values], axis=1)


def sources_table(result: PipelineResult) -> pd.DataFrame:
    rows = [
        {
            "Джерело": source.name,
            "Формат": source.format,
            "Походження": source.origin,
            "Записів": source.n_records,
            "Колонок": len(source.columns),
            "Повнота": round(result.profiles[source.name].completeness, DECIMALS),
        }
        for source in result.sources
    ]
    return pd.DataFrame(rows, columns=["Джерело", "Формат", "Походження", "Записів", "Колонок", "Повнота"])


def profile_table(profile: TableProfile) -> pd.DataFrame:
    rows = []
    for column in profile.columns:
        summary = column.numeric
        row = {
            "Колонка": column.name,
            "Тип": TYPE_LABELS[column.type],
            "Частка значень цього типу": round(column.type_share, 3),
            "Пропусків": column.missing,
            "З них прихованих": column.disguised_missing,
            "Різних значень": column.distinct,
            "Повнота": round(column.completeness, DECIMALS),
            "Найчастіше значення": column.top_values[0][0] if column.top_values else "",
            "Переважний шаблон": column.top_patterns[0][0] if column.top_patterns else "",
            "Частка шаблону": round(column.dominant_pattern_share, 3),
            "Мінімум": _readable(column.type, summary.minimum if summary else None),
            "Медіана": _readable(column.type, summary.median if summary else None),
            "Максимум": _readable(column.type, summary.maximum if summary else None),
            "Нижня межа IQR": _readable(column.type, summary.lower_fence if summary else None),
            "Верхня межа IQR": _readable(column.type, summary.upper_fence if summary else None),
        }
        for detector in OUTLIER_DETECTORS:
            row[f"Викидів ({OUTLIER_LABELS[detector.key]})"] = column.outliers.get(detector.key)
        rows.append(row)
    return pd.DataFrame(rows)


def result_profile_table(result: PipelineResult) -> pd.DataFrame:
    return profile_table(profile_frame(result.consolidated[result.attributes], RESULT_PROFILE))


def mapping_table(result: PipelineResult) -> pd.DataFrame:
    frame = result.mapping.to_frame(source.name for source in result.sources)
    frame["type"] = frame["type"].map(lambda value: TYPE_LABELS[ColumnType(value)])
    return frame.rename(columns={"attribute": "Атрибут", "type": "Тип", "confidence": "Впевненість"})


def standardization_table(result: PipelineResult) -> pd.DataFrame:
    counts = {name: dict.fromkeys(CellStatus, 0) for name in result.attributes}
    parameters: dict[str, dict[str, str]] = {}
    for table in result.standardized.values():
        parameters.update(table.parameters)
        for column in table.status.columns:
            codes = table.status[column].to_numpy()
            for status in CellStatus:
                counts[column][status] += int((codes == status).sum())
    rows = []
    for attribute in result.attributes:
        row = {"Атрибут": attribute, "Тип": TYPE_LABELS[result.mapping.types[attribute]]}
        row.update({STATUS_LABELS[status]: counts[attribute][status] for status in CellStatus})
        row["Параметри"] = "; ".join(f"{key}={value}" for key, value in parameters.get(attribute, {}).items() if value)
        rows.append(row)
    return pd.DataFrame(rows)


def linkage_table(result: PipelineResult) -> pd.DataFrame:
    linkage, records = result.linkage, result.records
    if linkage is None:
        rows = [("Записів", records.size), ("Пошук дублікатів", "вимкнено"), ("Сутностей", records.size)]
        return pd.DataFrame(rows, columns=["Показник", "Значення"])
    total = records.total_pairs(linkage.scope)
    candidates = len(linkage.candidates)
    compared = ", ".join(
        f"{attribute} ({metric})"
        for attribute, metric in zip(linkage.comparison.attributes, linkage.comparison.metrics)
    )
    upper, lower = linkage.weight_thresholds if linkage.model is not None else (np.nan, np.nan)
    chosen = "; ".join(
        f"{attribute}: {choice.metric}" for attribute, choice in linkage.metric_choices.items() if choice.scores
    )
    rows = [
        ("Записів", records.size),
        ("Пошук дублікатів", POLICY_CHOICE_LABELS[linkage.settings.policy]),
        ("Повтори всередині джерел", _internal_duplicates(linkage)),
        ("Можливих пар записів", total),
        ("Блокування", linkage.blocker.label),
        ("Пар-кандидатів після блокування", candidates),
        ("Скорочення кількості порівнянь (RR)", round(1.0 - candidates / total, 6) if total else 0.0),
        ("Атрибути порівняння", compared),
        ("Метрики, обрані програмою за даними", chosen or "—"),
        ("Модель", SCORER_LABELS[linkage.settings.scorer.key]),
        ("Групування", CLUSTERER_LABELS[linkage.clusterer.key]),
        ("Ітерацій EM", linkage.model.iterations if linkage.model else 0),
        ("Кроків спільного навчання", linkage.training_rounds),
        ("Апріорна частка збігів серед кандидатів", round(linkage.prior, DECIMALS)),
        ("Верхній поріг ваги пари", round(upper, DECIMALS)),
        ("Нижній поріг ваги пари", round(lower, DECIMALS)),
        *((f"Пар у зоні «{MATCH_STATUS_LABELS[status]}»", linkage.count(status)) for status in reversed(MatchStatus)),
        ("Пар, щодо яких дві моделі не згодні", int(linkage.contested.sum())),
        ("Пар перевірено вручну", int((linkage.reviewed >= 0).sum())),
        ("Пар визнано дублікатами", int(linkage.accepted.sum())),
        ("Очікувано хибних збігів серед них (оцінка моделі)", round(linkage.expected_false_matches(), 1)),
        ("Очікувано пропущених збігів серед кандидатів (оцінка моделі)", round(linkage.expected_missed_matches(), 1)),
        ("Сутностей", linkage.entities),
        ("Об'єднано записів", records.size - linkage.entities),
    ]
    return pd.DataFrame(rows, columns=["Показник", "Значення"])


RELIABLE = "надійний"
NEEDS_REVIEW = "варто перевірити вручну"
RELIABILITY_ROWS = {
    "verdict": "Висновок",
    "reasons": "Чому",
    "accepted": "Пар визнано дублікатами",
    "false_share": "Очікувана частка хибних збігів серед них",
    "false_tolerance": "Допустима частка хибних збігів",
    "missed_share": "Очікувана частка пропущених дублікатів",
    "missed_tolerance": "Допустима частка пропущених дублікатів",
    "contested": "Пар, щодо яких дві моделі не згодні",
    "possible": "Пар у зоні ручної перевірки",
    "suggested": "Рекомендовано перевірити пар",
    "rounds": "Кроків спільного навчання",
    "one_to_one": "Обмеження «не більше одного запису з кожного джерела» під час навчання",
    "corrected": "Модель уточнено відповідями людини",
}


def reliability_table(result: PipelineResult) -> pd.DataFrame:
    linkage = result.linkage
    if linkage is None:
        return pd.DataFrame(columns=["Показник", "Значення"])
    reliability = linkage.reliability()
    values = {
        "verdict": RELIABLE if reliability.reliable else NEEDS_REVIEW,
        "reasons": "; ".join(REASON_LABELS[reason] for reason in reliability.reasons) or "—",
        "accepted": reliability.accepted,
        "false_share": round(reliability.false_share, DECIMALS),
        "false_tolerance": reliability.false_match_rate,
        "missed_share": round(reliability.missed_share, DECIMALS),
        "missed_tolerance": reliability.missed_match_rate,
        "contested": reliability.contested,
        "possible": reliability.possible,
        "suggested": reliability.suggested_reviews,
        "rounds": reliability.training_rounds,
        "one_to_one": "так" if reliability.one_to_one else "ні",
        "corrected": "так" if reliability.corrected else "ні",
    }
    return pd.DataFrame(
        [(label, values[key]) for key, label in RELIABILITY_ROWS.items()], columns=["Показник", "Значення"]
    )


def model_table(result: PipelineResult) -> pd.DataFrame:
    model = result.linkage.model if result.linkage is not None else None
    if model is None:
        return pd.DataFrame(columns=["Атрибут", "Рівень подібності", "m", "u", "Вага log2(m/u)"])
    frame = model.parameters_frame()
    frame[["m", "u", "weight"]] = frame[["m", "u", "weight"]].round(6)
    return frame.rename(columns={"attribute": "Атрибут", "level": "Рівень подібності", "weight": "Вага log2(m/u)"})


def review_table(result: PipelineResult) -> pd.DataFrame:
    columns = ["Джерело 1", "Запис 1", "Джерело 2", "Запис 2", "Імовірність збігу", "Вага пари", "Рішення"]
    linkage = result.linkage
    if linkage is None:
        return pd.DataFrame(columns=columns)
    records = result.records
    rows = []
    for index in linkage.review_queue():
        left, right = linkage.candidates[index]
        reviewed = int(linkage.reviewed[index])
        decision = "дублікати" if linkage.accepted[index] else "різні записи"
        rows.append(
            (
                *records.record_id(int(left)),
                *records.record_id(int(right)),
                round(float(linkage.probabilities[index]), DECIMALS),
                round(float(linkage.weights[index]), DECIMALS),
                decision + (" (перевірено вручну)" if reviewed >= 0 else ""),
            )
        )
    return pd.DataFrame(rows, columns=columns)


def fusion_table(result: PipelineResult) -> pd.DataFrame:
    fusion = result.fusion
    conflicts = fusion.conflicting_cells().sum()
    filled = fusion.filled_cells().sum()
    disputed = fusion.conflicts.groupby("attribute").size() if len(fusion.conflicts) else pd.Series(dtype=int)
    imputed = result.imputation.counts() if result.imputation is not None else pd.Series(dtype=int)
    rows = [
        {
            "Атрибут": attribute,
            "Правило вибору значення": RESOLVER_LABELS[fusion.resolvers[attribute].key],
            "Сутностей з різними значеннями": int(disputed.get(attribute, 0)),
            "Значень замінено узгодженим": int(conflicts.get(attribute, 0)),
            "Пропусків заповнено з інших джерел": int(filled.get(attribute, 0)),
            "Пропусків відновлено статистично": int(imputed.get(attribute, 0)),
        }
        for attribute in result.attributes
    ]
    return pd.DataFrame(rows)


def conflicts_table(result: PipelineResult) -> pd.DataFrame:
    return result.fusion.conflicts.rename(
        columns={
            "entity": "Сутність",
            "attribute": "Атрибут",
            "values": "Значень",
            "distinct": "Різних",
            "chosen": "Обране значення",
        }
    )


def source_accuracy_table(result: PipelineResult) -> pd.DataFrame:
    sizes = dict(zip(result.records.source_names, result.records.source_sizes.tolist()))
    rows = [
        {"Джерело": name, "Записів": sizes.get(name, 0), "Оцінена точність": round(accuracy, DECIMALS)}
        for name, accuracy in result.fusion.source_accuracy.items()
    ]
    return pd.DataFrame(rows, columns=["Джерело", "Записів", "Оцінена точність"])


def quality_table(result: PipelineResult) -> pd.DataFrame:
    weights, before, after = result.settings.weights, result.quality_before, result.quality_after
    rows = [
        {
            "Критерій": CRITERION_LABELS.get(criterion, criterion),
            "Вага": round(weights[criterion], DECIMALS),
            "До обробки": round(before[criterion], DECIMALS),
            "Після обробки": round(after[criterion], DECIMALS),
            "Зміна": round(after[criterion] - before[criterion], DECIMALS),
        }
        for criterion in before
    ]
    rows.append(
        {
            "Критерій": INDEX_LABEL,
            "Вага": round(sum(weights[criterion] for criterion in before), DECIMALS),
            "До обробки": round(result.index_before, DECIMALS),
            "Після обробки": round(result.index_after, DECIMALS),
            "Зміна": round(result.index_after - result.index_before, DECIMALS),
        }
    )
    return pd.DataFrame(rows)


def timings_table(result: PipelineResult) -> pd.DataFrame:
    rows = [(STAGE_LABELS.get(stage, stage), round(seconds, 3)) for stage, seconds in result.timings.items()]
    rows.append(("Усього", round(sum(result.timings.values()), 3)))
    return pd.DataFrame(rows, columns=["Етап", "Час, с"])


def settings_table(result: PipelineResult) -> pd.DataFrame:
    settings = result.settings
    standardization = settings.standardization
    rows = [
        ("Формати", POLICY_LABELS[standardization.policy]),
        ("Регіон телефонів", standardization.phone_region or "визначається за даними"),
        ("Формат телефонів", standardization.phone_format or "за правилом форматів"),
        ("Транслітерація кирилиці", "так" if standardization.transliterate else "ні"),
        ("Рівень значущості для зіставлення колонок", settings.schema_alpha),
    ]
    linkage = settings.linkage
    if linkage is None:
        rows.append(("Пошук дублікатів", "вимкнено"))
    else:
        decision = linkage.decision
        blocker = result.linkage.blocker if result.linkage is not None else linkage.blocker
        rows += [
            ("Пошук дублікатів", POLICY_CHOICE_LABELS[linkage.policy]),
            ("Блокування", blocker.label),
            ("Модель", SCORER_LABELS[linkage.scorer.key]),
            ("Поріг імовірності збігу", decision.match_probability),
            ("Допустима частка хибних збігів", decision.false_match_rate),
            ("Допустима частка пропущених дублікатів", decision.missed_match_rate),
        ]
    imputer = settings.imputer
    rows.append(("Відновлення пропусків", IMPUTER_LABELS[imputer.key] if imputer is not None else "вимкнено"))
    chosen = result.imputation.imputer if result.imputation is not None else None
    if chosen is not None and imputer is not None and chosen.key != imputer.key:
        rows.append(("Обраний метод відновлення", f"{IMPUTER_LABELS[chosen.key]} ({chosen.label})"))
    elif chosen is not None:
        rows.append(("Метод відновлення", chosen.label))
    return pd.DataFrame(rows, columns=["Параметр", "Значення"])


def _internal_duplicates(linkage: LinkageResult) -> str:
    structure = linkage.structure
    if len(linkage.records.source_names) == 1:
        return "одне джерело: шукаються повтори всередині нього"
    answer = "немає" if structure.clean else "є"
    if not structure.decided:
        return f"{answer} (задано в налаштуваннях)"
    if structure.evidence is None or not np.isfinite(structure.evidence):
        return f"{answer} (визначено програмою)"
    return f"{answer} (визначено програмою за критерієм BIC: {format_number(round(structure.evidence, 1))})"


def _readable(column_type: ColumnType, value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return ""
    if column_type in (ColumnType.NUMBER, ColumnType.MEASURE):
        return format_number(value)
    return standardizer_for(column_type).render(value) or ""
