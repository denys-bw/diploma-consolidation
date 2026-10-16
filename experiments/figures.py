from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullLocator
from matplotlib.transforms import offset_copy

from experiments.common import CO_TRAINING, CSV_ENCODING, RESULTS

SURFACE = "#ffffff"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
DEEMPHASIS = "#c3c2b7"
DPI = 200
LEARNING = "навчається"
TITLE_X = 0.02
WIDTH = 7.0

DATASET_LABELS = {
    "restaurants": "Restaurants",
    "abt_buy": "Abt-Buy",
    "amazon_google": "Amazon-Google",
    "dblp_acm": "DBLP-ACM",
    "dblp_scholar": "DBLP-Scholar",
    "chicago_sources": "Chicago",
    "settlements": "Geographic Settlements",
    "musicbrainz": "MusicBrainz 20K",
    "nc_voters_sample": "NC Voters (вибірка)",
    "flights": "Flights",
    "beers": "Beers",
}
SHARE_SERIES = (("F1", "F1"), ("precision", "precision"), ("recall", "recall"))
SHARE_MARKERS = ("o", "s", "^")
CHOSEN = "program_choice"
METRIC_LABELS = {
    CHOSEN: "метрика, обрана програмою для атрибута",
    "exact": "точний збіг",
    "jaro_winkler": "Джаро–Вінклер",
    "levenshtein": "Левенштейн",
    "token_set": "множини слів",
    "jaccard": "Жаккар (слова)",
    "tfidf": "TF-IDF (слова)",
    "tfidf_char": "TF-IDF (3-грами символів)",
}


def main() -> None:
    for draw in FIGURES:
        path = draw()
        if path is not None:
            print(path.relative_to(RESULTS.parent).as_posix())


def similarity() -> Path | None:
    metrics, thresholds = _read("e1_metrics.csv"), _read("e1_decision_threshold.csv")
    if metrics is None or thresholds is None:
        return None
    text = metrics[metrics["тип"] == "text"]
    precision = text.pivot_table(index=["набір", "атрибут"], columns="метрика", values="AP").dropna()
    chosen = text[text["обрано програмою"]].set_index(["набір", "атрибут"])["AP"].reindex(precision.index)
    precision = precision[chosen.notna()]
    if precision.empty:
        return None
    means = precision.mean()
    means[CHOSEN] = chosen.dropna().mean()
    means = means.sort_values(kind="stable")
    thresholds = thresholds.sort_values("F1 при найкращому порозі", kind="stable")
    figure, (top, bottom) = _panels(len(means), len(thresholds))
    positions = np.arange(len(means))
    colors = [SERIES[0] if metric == CHOSEN else DEEMPHASIS for metric in means.index]
    top.barh(positions, means, height=0.6, color=colors)
    top.set_yticks(positions, [METRIC_LABELS.get(metric, metric) for metric in means.index])
    _value_labels(top, positions, means)
    top.set_xlabel("середня точність ранжування пар (AP)")
    top.xaxis.set_major_formatter(FuncFormatter(lambda value, _: _number(value, 2)))
    cases, datasets = len(precision), precision.index.get_level_values(0).nunique()
    _panel_title(top, f"а) текстові атрибути: {cases}; набори даних: {datasets}; синім — вибір програми без розмітки")
    _dumbbell(
        bottom,
        thresholds["набір"],
        thresholds["F1 при p ≥ 0,5"],
        thresholds["F1 при найкращому порозі"],
        ("поріг 0,5 без розмітки", "найкращий поріг, обраний за еталоном"),
        "F1 класифікації пар до групування",
    )
    _panel_title(bottom, "б) поріг апостеріорної ймовірності збігу: 0,5 проти найкращого за еталоном")
    return _save(
        figure,
        "Метрики подібності й поріг рішення",
        "відповідь на перше питання дослідження",
        "e1_similarity.png",
    )


def models() -> Path | None:
    table = _read("e2_models.csv")
    if table is None:
        return None
    fs = table[table["модель"] == "Фелегі–Зунтер + групування"]
    trained = table[table["модель"] == CO_TRAINING][["набір", "сценарій", "F1"]]
    linear = table[table["модель"].str.startswith("лінійна з ручними вагами")].copy()
    linear["лише назва"] = linear["модель"].str.contains("лише назва")
    linear = linear.sort_values("лише назва", kind="stable").drop_duplicates(["набір", "сценарій"])
    rows = fs.merge(linear, on=["набір", "сценарій"], suffixes=(" fs", " linear"))
    rows = rows.merge(trained.rename(columns={"F1": "F1 trained"}), on=["набір", "сценарій"])
    labels = [
        f"{DATASET_LABELS.get(name, name)}: {scenario}" for name, scenario in zip(rows["набір"], rows["сценарій"])
    ]
    figure, axes = _canvas(len(rows), per_row=0.7)
    positions = np.arange(len(rows))[::-1]
    bars = (
        ("F1 trained", "Фелегі–Зунтер і логістична регресія, без розмітки", 0.27),
        ("F1 fs", "лише Фелегі–Зунтер, без розмітки", 0.0),
        ("F1 linear", "лінійна модель, поріг обрано за еталоном", -0.27),
    )
    for color, (column, label, offset) in zip(SERIES, bars):
        axes.barh(positions + offset, rows[column], height=0.25, color=color, label=label)
    axes.set_yticks(positions, labels)
    axes.set_xlim(0, 1.05)
    axes.set_xlabel("F1 знайдених пар дублікатів")
    _legend(axes)
    return _save(
        figure,
        "Імовірнісні моделі проти лінійної моделі з ручними вагами",
        "сценарії з телефоном, без телефону і без атрибутів-категорій",
        "e2_models.png",
    )


def scaling() -> Path | None:
    table = _read("e3_scaling.csv")
    if table is None or "записів" not in table:
        return None
    table = table.dropna(subset=["записів"])
    if table.empty:
        return None
    figure, axes = plt.subplots(figsize=(WIDTH, 3.4), facecolor=SURFACE)
    _style(axes)
    records = table["записів"]
    for column, color, label in (
        ("обробка з блокуванням, с", SERIES[0], "уся обробка з блокуванням TF-IDF (виміряно)"),
        ("порівняння всіх пар (оцінка), с", SERIES[1], "порівняння всіх пар без блокування (оцінка)"),
    ):
        axes.plot(records, table[column], color=color, linewidth=2, marker="o", markersize=5, label=label)
        axes.annotate(
            f"{_number(table[column].iloc[-1], 0)} с",
            (records.iloc[-1], table[column].iloc[-1]),
            textcoords="offset points",
            xytext=(7, 0),
            va="center",
            fontsize=8,
            color=SECONDARY,
        )
    axes.set_yscale("log")
    axes.yaxis.set_minor_locator(NullLocator())
    axes.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _number(value, 0 if value >= 1 else 1)))
    axes.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}".replace(",", " ")))
    axes.set_xlabel("записів в обох джерелах")
    axes.set_ylabel("час, с (логарифмічна шкала)")
    _legend(axes)
    return _save(
        figure,
        "Час зв'язування NC Voters залежно від обсягу даних",
        "кількість людей подвоюється, доки очікуваний час кроку менший за 30 хв",
        "e3_scaling.png",
    )


def flights() -> Path | None:
    detection, imputation = _read("e4_flights_detection.csv"), _read("e4_flights_imputation.csv")
    if detection is None or imputation is None:
        return None
    detection = detection.sort_values("F1", kind="stable")
    imputation = imputation.sort_values("частка точно відновлених", kind="stable")
    figure, (top, bottom) = _panels(len(detection), len(imputation))
    _bars(top, detection["метод"], detection["F1"], ["зважене" in method for method in detection["метод"]])
    top.set_xlabel("F1 виявлення неправильних значень")
    errors = int(detection["помилкових значень"].iloc[0])
    _panel_title(top, f"а) виявлення {errors} неправильних значень часу; синім — метод цієї роботи")
    imputed = imputation["частка точно відновлених"]
    _bars(bottom, imputation["метод"], imputed, ["дублікат" in method for method in imputation["метод"]])
    bottom.set_xlabel("частка пропусків, відновлених точно")
    gaps = int(imputation["пропусків"].iloc[0])
    _panel_title(bottom, f"б) відновлення {gaps} пропущених значень часу; синім — значення дублікатів")
    return _save(
        figure,
        "Неправильні значення й пропуски у Flights",
        "38 сайтів по-різному повідомляють час тих самих рейсів",
        "e4_flights.png",
    )


def quality_index() -> Path | None:
    table = _read("e5_quality.csv")
    if table is None:
        return None
    column = "індекс, ваги МАІ"
    before = table[table["етап"] == "до обробки"].set_index("набір")[column]
    after = table[table["етап"] == "після обробки"].set_index("набір")[column]
    order = list(after.sort_values(kind="stable").index)
    figure, axes = _canvas(len(order))
    _dumbbell(axes, order, before[order], after[order], ("до обробки", "після обробки"), "інтегральний індекс якості")
    return _save(
        figure,
        "Інтегральний індекс якості до і після обробки",
        "ваги критеріїв за МАІ; унікальність — за еталоном",
        "e5_quality_index.png",
    )


def diagnostics() -> Path | None:
    table = _read("e7_diagnostics.csv")
    if table is None:
        return None
    table = table.sort_values("дві моделі, найкращий поріг", kind="stable")
    table["стеля"] = table[["логістична регресія з повною розміткою", "градієнтний бустинг з повною розміткою"]].max(
        axis=1
    )
    figure, axes = _canvas(len(table), per_row=0.5)
    positions = np.arange(len(table))
    columns = (
        ("Фелегі–Зунтер, найкращий поріг", SERIES[1], "лише Фелегі–Зунтер, без розмітки"),
        ("дві моделі, найкращий поріг", SERIES[0], "дві моделі, без розмітки"),
        ("стеля", SERIES[2], "стеля: ті самі ознаки з повною розміткою"),
    )
    low = table[[column for column, _, _ in columns]].min(axis=1)
    high = table[[column for column, _, _ in columns]].max(axis=1)
    axes.hlines(positions, low, high, color=BASELINE, linewidth=2)
    for column, color, label in columns:
        axes.scatter(
            table[column], positions, s=42, color=color, zorder=3, label=label, edgecolors=SURFACE, linewidths=1.5
        )
    axes.set_yticks(positions, [DATASET_LABELS.get(name, name) for name in table["набір"]])
    axes.set_xlim(0, 1.05)
    axes.set_xlabel("F1 класифікації пар за найкращого порогу, до групування")
    _legend(axes)
    return _save(
        figure,
        "Де втрачається якість пошуку дублікатів",
        "відстань до зеленої точки — що могла б дати модель, а не ознаки",
        "e7_diagnostics.png",
    )


def active_learning() -> Path | None:
    table = _read("e8_active_learning.csv")
    if table is None:
        return None
    names = list(dict.fromkeys(table["набір"]))
    columns = 3
    rows = int(np.ceil(len(names) / columns))
    figure, grid = plt.subplots(
        rows,
        columns,
        figsize=(WIDTH, 1.2 + 2.0 * rows),
        facecolor=SURFACE,
        layout="constrained",
        sharex=True,
        squeeze=False,
    )
    for axes, name in zip(grid.ravel(), names):
        _style(axes)
        subset = table[table["набір"] == name]
        modes = sorted(subset["режим"].unique(), key=lambda mode: LEARNING not in mode)
        for mode, color in zip(modes, (SERIES[0], DEEMPHASIS)):
            line = subset[subset["режим"] == mode]
            axes.plot(line["відповідей"], line["F1"], color=color, linewidth=2, marker="o", markersize=4, label=mode)
        axes.set_title(DATASET_LABELS.get(name, name), fontsize=9, color=INK, loc="left")
        axes.yaxis.set_major_locator(MaxNLocator(4))
        axes.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _number(value, 3)))
        axes.xaxis.set_major_formatter(FuncFormatter(lambda value, _: _number(value, 0)))
    for axes in grid.ravel()[len(names) :]:
        axes.set_visible(False)
    for axes in grid[-1]:
        axes.set_xlabel("відповідей людини")
    handles, labels = grid.ravel()[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower left", ncol=2, frameon=False, fontsize=8, labelcolor=SECONDARY)
    return _save(
        figure,
        "Навчання на відповідях людини",
        "F1 після групування; програма сама вибирає найсумнівніші пари; шкала F1 у кожної панелі своя",
        "e8_active_learning.png",
    )


def robustness() -> Path | None:
    table = _read("e9_robustness.csv")
    if table is None:
        return None
    conditions = list(dict.fromkeys(table["умова"]))
    names = list(dict.fromkeys(table["набір"]))
    figure, axes = _canvas(len(names), per_row=0.25 * len(conditions) + 0.2)
    height = 0.8 / len(conditions)
    positions = np.arange(len(names))[::-1]
    for index, (condition, color) in enumerate(zip(conditions, SERIES)):
        subset = table[table["умова"] == condition].set_index("набір").reindex(names)
        offsets = positions + 0.4 - height * (index + 0.5)
        axes.barh(offsets, subset["F1"].fillna(0), height=height * 0.9, color=color, label=condition)
    axes.set_yticks(positions, [DATASET_LABELS.get(name, name) for name in names])
    axes.set_xlim(0, 1.05)
    axes.set_xlabel("F1 знайдених пар дублікатів")
    _legend(axes)
    return _save(
        figure,
        "Стійкість до несхожих джерел",
        "справжні дані в умовах, як при злитті баз різних компаній",
        "e9_robustness.png",
    )


def sensitivity() -> Path | None:
    table = _read("e10_sensitivity.csv")
    if table is None:
        return None
    datasets = [column for column in table.columns if column in DATASET_LABELS]
    defaults = table[table["типове"]].set_index("параметр")
    changed = table[~table["типове"]].reset_index(drop=True)
    shifts = (changed[datasets].to_numpy() - defaults.loc[changed["параметр"], datasets].to_numpy()) * 100
    figure, axes = _canvas(len(changed), per_row=0.36)
    positions = np.arange(len(changed))[::-1]
    axes.axvline(0, color=BASELINE, linewidth=1)
    axes.hlines(
        positions,
        shifts.min(axis=1),
        shifts.max(axis=1),
        color=DEEMPHASIS,
        linewidth=2,
        label="розкид по окремих наборах",
    )
    means = shifts.mean(axis=1)
    axes.scatter(means, positions, s=42, color=SERIES[0], zorder=3, edgecolors=SURFACE, linewidths=1.5, label="середнє")
    for position, mean in zip(positions, means):
        axes.annotate(
            _number(mean, 1),
            (mean, position),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=7,
            color=SECONDARY,
        )
    labels = [
        f"{parameter} = {_number(value, 2).rstrip('0').rstrip(',')}"
        for parameter, value in zip(changed["параметр"], changed["значення"])
    ]
    axes.set_yticks(positions, labels)
    axes.xaxis.set_major_formatter(FuncFormatter(lambda value, _: _number(value, 0)))
    axes.set_xlabel("зміна F1 відносно типових параметрів, п. п.")
    _legend(axes)
    return _save(
        figure,
        "Чутливість до параметрів",
        f"змінюється один параметр, решта типові; наборів даних: {len(datasets)}",
        "e10_sensitivity.png",
    )


def duplicate_share() -> Path | None:
    table = _read("e11_duplicate_share.csv")
    if table is None:
        return None
    shares = pd.to_numeric(table["людей з копіями"], errors="coerce")
    table = table[shares.notna()].assign(частка=shares[shares.notna()] * 100).sort_values("частка")
    if table.empty:
        return None
    figure, axes = plt.subplots(figsize=(WIDTH, 3.4), facecolor=SURFACE)
    _style(axes)
    for (column, label), color, marker in zip(SHARE_SERIES, SERIES, SHARE_MARKERS):
        axes.plot(table["частка"], table[column], color=color, linewidth=2, marker=marker, markersize=6, label=label)
        axes.annotate(
            f"{label} {_number(table[column].iloc[-1], 3)}",
            (table["частка"].iloc[-1], table[column].iloc[-1]),
            textcoords="offset points",
            xytext=(8, 0),
            va="center",
            fontsize=8,
            color=SECONDARY,
        )
    axes.set_xticks(table["частка"], [f"{_number(value, 0)} %" for value in table["частка"]])
    axes.set_ylim(0, 1.05)
    axes.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _number(value, 1)))
    axes.set_xlabel("частка людей, які мають копії в інших джерелах")
    _legend(axes)
    return _save(
        figure,
        "Якість пошуку дублікатів залежно від їхньої частки",
        "NC Voters: ті самі люди й помилки, змінюється лише частка людей з копіями",
        "e11_duplicate_share.png",
    )


def _dumbbell(axes, names, left, right, labels, xlabel) -> None:
    names, left, right = list(names), np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    positions = np.arange(len(names))
    axes.hlines(positions, np.minimum(left, right), np.maximum(left, right), color=BASELINE, linewidth=2)
    for values, color, label in ((left, SERIES[1], labels[0]), (right, SERIES[0], labels[1])):
        axes.scatter(values, positions, s=42, color=color, zorder=3, label=label, edgecolors=SURFACE, linewidths=1.5)
    for position, value in zip(positions, right):
        axes.annotate(
            _number(value, 2),
            (value, position),
            textcoords="offset points",
            xytext=(8, 0),
            va="center",
            fontsize=8,
            color=SECONDARY,
        )
    axes.set_yticks(positions, [DATASET_LABELS.get(name, name) for name in names])
    axes.set_xlim(0, 1.08)
    axes.set_xlabel(xlabel)
    _legend(axes)


def _bars(axes, names, values, highlighted) -> None:
    positions = np.arange(len(values))
    axes.barh(positions, values, height=0.6, color=[SERIES[0] if flag else DEEMPHASIS for flag in highlighted])
    axes.set_yticks(positions, list(names))
    _value_labels(axes, positions, values)
    axes.set_xlim(0, 1.1)


def _panels(top_rows: int, bottom_rows: int, per_row: float = 0.42):
    figure, panels = plt.subplots(
        2,
        1,
        figsize=(WIDTH, 2.2 + per_row * (top_rows + bottom_rows)),
        facecolor=SURFACE,
        layout="constrained",
        gridspec_kw={"height_ratios": [top_rows + 1.5, bottom_rows + 1.5]},
    )
    figure.get_layout_engine().set(h_pad=0.25, hspace=0.08)
    for axes in panels:
        _style(axes)
        axes.grid(axis="y", visible=False)
    return figure, panels


def _panel_title(axes, text: str) -> None:
    axes.set_title(" ", fontsize=9, pad=12)
    axes.annotate(
        text,
        xy=(TITLE_X, 1),
        xycoords=("figure fraction", "axes fraction"),
        xytext=(0, 12),
        textcoords="offset points",
        fontsize=9,
        color=INK,
    )


def _canvas(rows: int, per_row: float = 0.42):
    figure, axes = plt.subplots(figsize=(WIDTH, 0.9 + per_row * rows), facecolor=SURFACE)
    _style(axes)
    axes.grid(axis="y", visible=False)
    return figure, axes


def _style(axes) -> None:
    axes.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(BASELINE)
    axes.tick_params(colors=MUTED, labelcolor=SECONDARY, labelsize=8, length=0)
    for label in (axes.xaxis.label, axes.yaxis.label):
        label.set_color(SECONDARY)
        label.set_size(9)
    axes.grid(color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.xaxis.set_major_formatter(FuncFormatter(lambda value, _: _number(value, 1)))


def _legend(axes) -> None:
    axes.legend(
        loc="upper left",
        bbox_to_anchor=(0, -0.16),
        ncol=2,
        frameon=False,
        fontsize=8,
        labelcolor=SECONDARY,
        handlelength=1.2,
        columnspacing=1.5,
    )


def _value_labels(axes, positions, values) -> None:
    for position, value in zip(positions, values):
        axes.annotate(
            _number(value, 2),
            (value, position),
            textcoords="offset points",
            xytext=(4, 0),
            va="center",
            fontsize=8,
            color=SECONDARY,
        )


def _number(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")


def _read(name: str) -> pd.DataFrame | None:
    path = RESULTS / name
    if not path.exists():
        print(f"[пропущено] немає {name}")
        return None
    return pd.read_csv(path, encoding=CSV_ENCODING)


def _save(figure, title: str, subtitle: str, name: str) -> Path:
    if figure.get_layout_engine() is None:
        figure.tight_layout()
    subtitle_place = offset_copy(figure.transFigure, fig=figure, y=6, units="points")
    title_place = offset_copy(figure.transFigure, fig=figure, y=20, units="points")
    figure.text(TITLE_X, 1, subtitle, transform=subtitle_place, fontsize=8, color=MUTED, va="bottom", ha="left")
    figure.text(TITLE_X, 1, title, transform=title_place, fontsize=11, color=INK, va="bottom", ha="left")
    path = RESULTS / name
    figure.savefig(path, dpi=DPI, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)
    return path


FIGURES: tuple[Callable[[], Path | None], ...] = (
    similarity,
    models,
    scaling,
    flights,
    quality_index,
    diagnostics,
    active_learning,
    robustness,
    sensitivity,
    duplicate_share,
)


if __name__ == "__main__":
    main()
