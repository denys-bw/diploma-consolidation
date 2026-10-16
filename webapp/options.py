from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from fractions import Fraction

from consolidator.fusion import RESOLVERS
from consolidator.imputation import AdaptiveImputer, Imputer, KnnImputer, MeanImputer
from consolidator.linkage import (
    MAX_FULL_PAIRS,
    AutomaticKeys,
    Blocker,
    CoTraining,
    DecisionRule,
    DuplicatePolicy,
    FellegiSunterScorer,
    FullComparison,
    LinkageSettings,
    PairScope,
    PairScorer,
    TokenBlocking,
    pair_count,
)
from consolidator.pipeline import PipelineSettings
from consolidator.quality import (
    CONSISTENCY_LIMIT,
    CRITERION_KEYS,
    DEFAULT_JUDGEMENTS,
    AhpResult,
    QualityWeights,
    analyse,
    pairwise_matrix,
)
from consolidator.reporting.labels import CRITERION_LABELS
from consolidator.standardization import FormatPolicy, StandardizationSettings
from consolidator.statistics import SIGNIFICANCE
from consolidator.values.phones import MAX_DIGITS, MIN_DIGITS, SUPPORTED_REGIONS
from webapp.formatting import number

DEFAULT_DECISION = DecisionRule()
AHP_PAIRS = (("completeness", "uniqueness"), ("completeness", "consistency"), ("uniqueness", "consistency"))
SAATY_VALUES = (*(str(value) for value in range(9, 0, -1)), *(f"1/{value}" for value in range(2, 10)))
SAATY_ADVERBS = {3: "помірно", 5: "істотно", 7: "значно", 9: "абсолютно"}
PHONE_FORMAT_CHARACTERS = frozenset("0123456789 +-()./")
PROBABILITY_RANGE = (0.01, 0.99)
RATE_RANGE = (0.0, 0.5)
ALPHA_RANGE = (0.001, 0.2)
NEIGHBOURS_RANGE = (1, 1000)
WINDOW_RANGE = (2, 1000)
MATCH_LABEL = "Поріг імовірності збігу"
FALSE_LABEL = "Допустима частка хибних збігів"
MISSED_LABEL = "Допустима частка пропущених дублікатів"
ALPHA_LABEL = "Рівень значущості"
COMMON_REGIONS = {
    "UA": "Україна",
    "US": "США",
    "CA": "Канада",
    "GB": "Велика Британія",
    "DE": "Німеччина",
    "PL": "Польща",
    "FR": "Франція",
    "IT": "Італія",
    "ES": "Іспанія",
    "MD": "Молдова",
    "RO": "Румунія",
    "CZ": "Чехія",
}


class DuplicateMode(str, Enum):
    AUTO = "auto"
    EVERYWHERE = "everywhere"
    ACROSS = "across"
    ACROSS_CLEAN = "across_clean"
    NONE = "none"

    @property
    def policy(self) -> DuplicatePolicy | None:
        return None if self is DuplicateMode.NONE else DuplicatePolicy(self.value)


class BlockingMode(str, Enum):
    TOKENS = "tokens"
    KEYS = "keys"
    SORTED = "sorted"
    FULL = "full"


class ModelMode(str, Enum):
    CO_TRAINING = "co_training"
    FELLEGI_SUNTER = "fellegi_sunter"


class ImputationMode(str, Enum):
    AUTO = "auto"
    KNN = "knn"
    MEAN = "mean"
    NONE = "none"


@dataclass(frozen=True)
class Choice:
    value: str
    title: str
    hint: str


POLICY_CHOICES = (
    Choice(
        FormatPolicy.STANDARD.value,
        "Міжнародні стандарти",
        "ISO 8601 для дат і часу, E.164 для телефонів, числа з десятковою крапкою.",
    ),
    Choice(
        FormatPolicy.DOMINANT.value,
        "Переважний формат даних",
        "Кожна колонка приводиться до формату, який у ній переважає.",
    ),
)
DUPLICATE_CHOICES = (
    Choice(
        DuplicateMode.AUTO.value,
        "Автоматично",
        "Програма сама перевіряє, чи є повтори всередині джерел, і вибирає спосіб групування.",
    ),
    Choice(
        DuplicateMode.EVERYWHERE.value,
        "У джерелах і між ними",
        "Дублікати шукаються і всередині джерел, і між ними.",
    ),
    Choice(
        DuplicateMode.ACROSS.value,
        "Між джерелами",
        "Порівнюються лише записи з різних джерел.",
    ),
    Choice(
        DuplicateMode.ACROSS_CLEAN.value,
        "Між джерелами без внутрішніх повторів",
        "Джерела вважаються очищеними: з кожного не більше одного запису на сутність.",
    ),
    Choice(
        DuplicateMode.NONE.value,
        "Не шукати",
        "Лише уніфікація, відновлення пропусків і оцінка якості.",
    ),
)
BLOCKING_CHOICES = (
    Choice(
        BlockingMode.TOKENS.value,
        "TF-IDF і найближчі сусіди",
        "Найсхожіші записи за рідкісними словами.",
    ),
    Choice(
        BlockingMode.KEYS.value,
        "Стандартне за ключами",
        "Записи зі спільним значенням ключа.",
    ),
    Choice(
        BlockingMode.SORTED.value,
        "Сортоване сусідство",
        "Ковзне вікно за відсортованими ключами.",
    ),
    Choice(
        BlockingMode.FULL.value,
        "Без блокування",
        "Усі пари записів, до 5 млн.",
    ),
)
MODEL_CHOICES = (
    Choice(
        ModelMode.CO_TRAINING.value,
        "Дві моделі, що вчаться одна в одної",
        "Фелегі–Зунтер і логістична регресія за кількома метриками подібності всіх атрибутів.",
    ),
    Choice(ModelMode.FELLEGI_SUNTER.value, "Лише Фелегі–Зунтер", "Класична модель з EM без розмітки."),
)
IMPUTATION_CHOICES = (
    Choice(
        ImputationMode.AUTO.value,
        "Автоматично",
        "Програма перехресною перевіркою вибирає між середнім і k найближчими сусідами та саме k.",
    ),
    Choice(
        ImputationMode.KNN.value,
        "k найближчих сусідів",
        "Середнє k найсхожіших записів.",
    ),
    Choice(ImputationMode.MEAN.value, "Середнє значення", "Середнє значення атрибута."),
    Choice(ImputationMode.NONE.value, "Не відновлювати", "Пропуски лишаються."),
)


class OptionsError(ValueError):
    def __init__(self, messages: Sequence[str]) -> None:
        super().__init__("; ".join(messages))
        self.messages = list(messages)


def saaty_label(value: str, left: str, right: str) -> str:
    ratio = Fraction(value)
    if ratio == 1:
        return "1 — однаково важливі"
    leader, strength = (left, ratio.numerator) if ratio > 1 else (right, ratio.denominator)
    name = _lowercase(CRITERION_LABELS[leader])
    adverb = SAATY_ADVERBS.get(strength)
    if adverb is None:
        return f"{value} — {name} важливіша (проміжна оцінка)"
    return f"{value} — {name} {adverb} важливіша"


def judgement_values(judgements: Mapping[tuple[str, str], float]) -> tuple[str, ...]:
    values = []
    for left, right in AHP_PAIRS:
        if (left, right) in judgements:
            values.append(str(Fraction(judgements[left, right]).limit_denominator(9)))
        else:
            values.append(str(1 / Fraction(judgements[right, left]).limit_denominator(9)))
    return tuple(values)


DEFAULT_PRESET = judgement_values(DEFAULT_JUDGEMENTS)
EQUAL_PRESET = ("1",) * len(AHP_PAIRS)


def ahp_for(judgements: Sequence[str]) -> AhpResult:
    values = {pair: float(Fraction(value)) for pair, value in zip(AHP_PAIRS, judgements)}
    return analyse(CRITERION_KEYS, pairwise_matrix(CRITERION_KEYS, values))


@dataclass(frozen=True)
class RunOptions:
    policy: FormatPolicy = FormatPolicy.STANDARD
    phone_region: str = ""
    phone_format: str = ""
    transliterate: bool = False
    duplicates: DuplicateMode = DuplicateMode.AUTO
    blocking: BlockingMode = BlockingMode.TOKENS
    window: int = 0
    model: ModelMode = ModelMode.CO_TRAINING
    match_probability: float = DEFAULT_DECISION.match_probability
    false_match_rate: float = DEFAULT_DECISION.false_match_rate
    missed_match_rate: float = DEFAULT_DECISION.missed_match_rate
    imputation: ImputationMode = ImputationMode.AUTO
    neighbours: int = 0
    judgements: tuple[str, ...] = DEFAULT_PRESET
    schema_alpha: float = SIGNIFICANCE

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> RunOptions:
        try:
            return cls(**{name: parse(data[name]) for name, parse in FIELDS.items() if name in data})
        except (TypeError, ValueError):
            return cls()

    @classmethod
    def from_form(cls, form: Mapping[str, str], base: RunOptions | None = None) -> RunOptions:
        base = base or cls()
        errors: list[str] = []
        options = cls(
            policy=_choice(form, "policy", FormatPolicy, base.policy, errors),
            phone_region=_region(form.get("phone_region", base.phone_region), errors),
            phone_format=_phone_format(form.get("phone_format", base.phone_format), errors),
            transliterate="transliterate" in form,
            duplicates=_choice(form, "duplicates", DuplicateMode, base.duplicates, errors),
            blocking=_choice(form, "blocking", BlockingMode, base.blocking, errors),
            window=_optional_integer(form, "window", WINDOW_RANGE, "Розмір вікна", errors),
            model=_choice(form, "model", ModelMode, base.model, errors),
            match_probability=_number(
                form, "match_probability", base.match_probability, PROBABILITY_RANGE, MATCH_LABEL, errors
            ),
            false_match_rate=_percent(form, "false_match_rate", base.false_match_rate, FALSE_LABEL, errors),
            missed_match_rate=_percent(form, "missed_match_rate", base.missed_match_rate, MISSED_LABEL, errors),
            imputation=_choice(form, "imputation", ImputationMode, base.imputation, errors),
            neighbours=_optional_integer(form, "neighbours", NEIGHBOURS_RANGE, "Кількість сусідів", errors),
            judgements=_judgements(form, base.judgements, errors),
            schema_alpha=base.schema_alpha,
        )
        if not errors:
            errors.extend(options._conflicts())
        if errors:
            raise OptionsError(errors)
        return options

    def with_schema_alpha(self, raw: str) -> RunOptions:
        errors: list[str] = []
        alpha = _number({"schema_alpha": raw}, "schema_alpha", self.schema_alpha, ALPHA_RANGE, ALPHA_LABEL, errors)
        if errors:
            raise OptionsError(errors)
        return replace(self, schema_alpha=alpha)

    def to_dict(self) -> dict[str, object]:
        return {name: _plain(getattr(self, name)) for name in FIELDS}

    @property
    def scope(self) -> PairScope:
        if self.duplicates in (DuplicateMode.ACROSS, DuplicateMode.ACROSS_CLEAN):
            return PairScope.CROSS
        return PairScope.ALL

    def problems(self, sizes: Sequence[int]) -> list[str]:
        problems = []
        if not sizes:
            problems.append("У проєкті немає жодного увімкненого джерела.")
        if len(sizes) == 1 and self.duplicates in (DuplicateMode.ACROSS, DuplicateMode.ACROSS_CLEAN):
            problems.append(
                "У проєкті одне джерело, тому пар записів з різних джерел немає: "
                "виберіть пошук дублікатів «У джерелах і між ними»."
            )
        if self.duplicates is not DuplicateMode.NONE and self.blocking is BlockingMode.FULL:
            total = pair_count(sizes, self.scope)
            if total > MAX_FULL_PAIRS:
                problems.append(
                    f"Без блокування довелося б порівняти {number(total)} пар записів, а допустимо не більше "
                    f"{number(MAX_FULL_PAIRS)}. Виберіть один із методів блокування."
                )
        return problems

    def ahp(self) -> AhpResult:
        return ahp_for(self.judgements)

    def weights(self) -> QualityWeights:
        return QualityWeights.from_ahp(self.ahp())

    def blocker(self) -> Blocker:
        if self.blocking is BlockingMode.KEYS:
            return AutomaticKeys()
        if self.blocking is BlockingMode.SORTED:
            return AutomaticKeys(sorted_window=True, window=self.window or None)
        if self.blocking is BlockingMode.FULL:
            return FullComparison()
        return TokenBlocking()

    def scorer(self) -> PairScorer:
        return FellegiSunterScorer() if self.model is ModelMode.FELLEGI_SUNTER else CoTraining()

    def imputer(self) -> Imputer | None:
        if self.imputation is ImputationMode.AUTO:
            return AdaptiveImputer()
        if self.imputation is ImputationMode.KNN:
            return KnnImputer(self.neighbours) if self.neighbours else AdaptiveImputer()
        return MeanImputer() if self.imputation is ImputationMode.MEAN else None

    def linkage_settings(self) -> LinkageSettings | None:
        if self.duplicates is DuplicateMode.NONE:
            return None
        return LinkageSettings(
            policy=self.duplicates.policy,
            blocker=self.blocker(),
            decision=DecisionRule(self.match_probability, self.false_match_rate, self.missed_match_rate),
            scorer=self.scorer(),
        )

    def pipeline_settings(
        self, resolutions: Mapping[str, str] | None = None, source_priority: Sequence[str] = ()
    ) -> PipelineSettings:
        standardization = StandardizationSettings(
            policy=self.policy,
            phone_region=self.phone_region or None,
            phone_format=self.phone_format or None,
            transliterate=self.transliterate,
        )
        return PipelineSettings(
            standardization=standardization,
            schema_alpha=self.schema_alpha,
            linkage=self.linkage_settings(),
            resolvers={attribute: RESOLVERS[key] for attribute, key in (resolutions or {}).items()},
            source_priority=tuple(source_priority),
            imputer=self.imputer(),
            weights=self.weights(),
        )

    def _conflicts(self) -> list[str]:
        problems = []
        ahp = self.ahp()
        if not ahp.consistent:
            problems.append(
                f"Судження МАІ суперечливі: відношення узгодженості CR = {_decimal(round(ahp.consistency_ratio, 3))} "
                f"не менше {_decimal(CONSISTENCY_LIMIT)}. Перегляньте попарні порівняння."
            )
        return problems


FIELDS: dict[str, Callable[[object], object]] = {
    "policy": FormatPolicy,
    "phone_region": str,
    "phone_format": str,
    "transliterate": bool,
    "duplicates": DuplicateMode,
    "blocking": BlockingMode,
    "window": int,
    "model": ModelMode,
    "match_probability": float,
    "false_match_rate": float,
    "missed_match_rate": float,
    "imputation": ImputationMode,
    "neighbours": int,
    "judgements": tuple,
    "schema_alpha": float,
}


def _lowercase(label: str) -> str:
    return label[:1].lower() + label[1:]


def _plain(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return list(value)
    return value


def _choice(form: Mapping[str, str], name: str, kind: type[Enum], default: Enum, errors: list[str]) -> Enum:
    raw = form.get(name)
    if raw is None:
        return default
    try:
        return kind(raw)
    except ValueError:
        errors.append(f"Невідоме значення параметра «{name}»: {raw}")
        return default


def _number(
    form: Mapping[str, str],
    name: str,
    default: float,
    bounds: tuple[float, float],
    label: str,
    errors: list[str],
) -> float:
    raw = form.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw.strip().replace(",", "."))
    except ValueError:
        errors.append(f"{label}: «{raw}» не є числом.")
        return default
    low, high = bounds
    if not low <= value <= high:
        errors.append(f"{label}: значення має бути від {_decimal(low)} до {_decimal(high)}.")
        return default
    return value


def _integer(
    form: Mapping[str, str],
    name: str,
    default: int,
    bounds: tuple[int, int],
    label: str,
    errors: list[str],
) -> int:
    value = _number(form, name, default, bounds, label, errors)
    if not float(value).is_integer():
        errors.append(f"{label}: потрібне ціле число.")
        return default
    return int(value)


def _optional_integer(
    form: Mapping[str, str], name: str, bounds: tuple[int, int], label: str, errors: list[str]
) -> int:
    raw = form.get(name)
    if raw is None or not raw.strip():
        return 0
    return _integer(form, name, 0, bounds, label, errors)


def _percent(form: Mapping[str, str], name: str, default: float, label: str, errors: list[str]) -> float:
    raw = form.get(name)
    if raw is None or not raw.strip():
        return default
    low, high = RATE_RANGE
    value = _number(form, name, default * 100, (low * 100, high * 100), f"{label}, %", errors)
    return value / 100


def _decimal(value: float) -> str:
    return f"{value:g}".replace(".", ",")


def _region(raw: str, errors: list[str]) -> str:
    region = raw.strip().upper()
    if region and region not in SUPPORTED_REGIONS:
        errors.append(f"Невідомий код регіону «{raw.strip()}»: потрібен двобуквений код ISO 3166, наприклад UA або US.")
        return ""
    return region


def _phone_format(raw: str, errors: list[str]) -> str:
    template = raw.strip()
    if not template:
        return ""
    if not set(template) <= PHONE_FORMAT_CHARACTERS:
        errors.append("Шаблон телефону може містити лише цифри, пробіли та символи + - ( ) . /")
        return ""
    if not MIN_DIGITS <= template.count("9") <= MAX_DIGITS:
        errors.append(
            f"Шаблон телефону має містити від {MIN_DIGITS} до {MAX_DIGITS} цифр «9» — по одній на кожну цифру номера."
        )
        return ""
    return template


def _judgements(form: Mapping[str, str], default: tuple[str, ...], errors: list[str]) -> tuple[str, ...]:
    values = []
    for (left, right), fallback in zip(AHP_PAIRS, default):
        raw = form.get(f"ahp_{left}_{right}", fallback)
        if raw not in SAATY_VALUES:
            errors.append(f"Недопустима оцінка попарного порівняння: {raw}")
            raw = fallback
        values.append(raw)
    return tuple(values)
