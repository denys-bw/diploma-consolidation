from __future__ import annotations

from consolidator.fusion import Longest, MajorityVote, Median, SourcePriority, WeightedVote
from consolidator.imputation import IMPUTED_MARK, AdaptiveImputer, KnnImputer, MeanImputer
from consolidator.linkage import (
    AverageLinkage,
    ConnectedComponents,
    CoTraining,
    DuplicatePolicy,
    FellegiSunterScorer,
    MatchStatus,
    OneToOneAssignment,
)
from consolidator.linkage.reliability import FALSE_MATCHES, FEW_ATTRIBUTES, MISSED_MATCHES
from consolidator.profiling import ColumnType, IsolationForestDetector, TukeyFences
from consolidator.standardization import CellStatus, FormatPolicy

TYPE_LABELS = {
    ColumnType.EMPTY: "порожня",
    ColumnType.IDENTIFIER: "ідентифікатор",
    ColumnType.CODE: "код",
    ColumnType.BOOLEAN: "так/ні",
    ColumnType.CATEGORY: "категорія",
    ColumnType.NUMBER: "число",
    ColumnType.MEASURE: "величина з одиницею",
    ColumnType.YEAR: "рік",
    ColumnType.DATE: "дата",
    ColumnType.DATETIME: "дата й час",
    ColumnType.TIME: "час",
    ColumnType.PHONE: "телефон",
    ColumnType.EMAIL: "електронна пошта",
    ColumnType.URL: "веб-адреса",
    ColumnType.TEXT: "короткий текст",
}
STATUS_LABELS = {
    CellStatus.MISSING: "пропуск",
    CellStatus.UNCHANGED: "без змін",
    CellStatus.CHANGED: "уніфіковано",
    CellStatus.INVALID: "не розпізнано",
    CellStatus.AMBIGUOUS: "неоднозначне",
}
CRITERION_LABELS = {
    "completeness": "Повнота",
    "uniqueness": "Унікальність",
    "consistency": "Узгодженість форматів",
}
INDEX_LABEL = "Інтегральний індекс якості"
OUTLIER_LABELS = {
    TukeyFences.key: "IQR",
    IsolationForestDetector.key: "ізоляційний ліс",
}
RESOLVER_LABELS = {
    WeightedVote.key: "зважене голосування",
    MajorityVote.key: "голосування більшістю",
    Median.key: "медіана",
    Longest.key: "найповніше значення",
    SourcePriority.key: "значення пріоритетного джерела",
}
IMPUTER_LABELS = {
    AdaptiveImputer.key: "метод вибирається перехресною перевіркою",
    MeanImputer.key: "середнє значення",
    KnnImputer.key: "k найближчих сусідів",
}
CLUSTERER_LABELS = {
    AverageLinkage.key: "середній зв'язок",
    ConnectedComponents.key: "зв'язні компоненти",
    OneToOneAssignment.key: "не більше одного запису з кожного джерела",
}
SCORER_LABELS = {
    CoTraining.key: "Фелегі–Зунтер і логістична регресія, що вчаться одна в одної",
    FellegiSunterScorer.key: "Фелегі–Зунтер",
}
MATCH_STATUS_LABELS = {
    MatchStatus.MATCH: "збіг",
    MatchStatus.POSSIBLE: "можливий збіг",
    MatchStatus.NON_MATCH: "незбіг",
}
POLICY_LABELS = {
    FormatPolicy.STANDARD: "міжнародні стандарти (ISO 8601, E.164)",
    FormatPolicy.DOMINANT: "переважний формат даних",
}
POLICY_CHOICE_LABELS = {
    DuplicatePolicy.AUTO: "автоматично",
    DuplicatePolicy.EVERYWHERE: "у джерелах і між ними",
    DuplicatePolicy.ACROSS: "лише між джерелами",
    DuplicatePolicy.ACROSS_CLEAN: "лише між джерелами, джерела без внутрішніх повторів",
}
REASON_LABELS = {
    FEW_ATTRIBUTES: (
        "для порівняння менше трьох атрибутів, тому без відповідей людини модель не може однозначно "
        "відокремити дублікати від схожих різних записів"
    ),
    FALSE_MATCHES: "очікувана частка хибних збігів більша за допустиму",
    MISSED_MATCHES: "очікувана частка пропущених дублікатів більша за допустиму",
}
STAGE_LABELS = {
    "profiling": "Профілювання",
    "schema": "Зіставлення схем",
    "standardization": "Уніфікація форматів",
    "linkage": "Зв'язування записів",
    "fusion": "Злиття дублікатів",
    "imputation": "Відновлення пропусків",
    "quality": "Оцінювання якості",
}


def provenance_label(value: str) -> str:
    if not value.startswith(IMPUTED_MARK):
        return value
    return "відновлено: " + IMPUTER_LABELS.get(value.removeprefix(IMPUTED_MARK), value)
