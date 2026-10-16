import pytest

from consolidator.values.transliteration import transliterate

OFFICIAL_EXAMPLES = [
    ("Алушта", "Alushta"),
    ("Андрій", "Andrii"),
    ("Борщагівка", "Borshchahivka"),
    ("Борисенко", "Borysenko"),
    ("Вінниця", "Vinnytsia"),
    ("Володимир", "Volodymyr"),
    ("Гадяч", "Hadiach"),
    ("Богдан", "Bohdan"),
    ("Згурський", "Zghurskyi"),
    ("Ґалаґан", "Galagan"),
    ("Ґорґани", "Gorgany"),
    ("Донецьк", "Donetsk"),
    ("Дмитро", "Dmytro"),
    ("Рівне", "Rivne"),
    ("Олег", "Oleh"),
    ("Есмань", "Esman"),
    ("Єнакієве", "Yenakiieve"),
    ("Гаєвич", "Haievych"),
    ("Короп'є", "Koropie"),
    ("Житомир", "Zhytomyr"),
    ("Жанна", "Zhanna"),
    ("Жежелів", "Zhezheliv"),
    ("Закарпаття", "Zakarpattia"),
    ("Казимирчук", "Kazymyrchuk"),
    ("Медвин", "Medvyn"),
    ("Михайленко", "Mykhailenko"),
    ("Іванків", "Ivankiv"),
    ("Іващенко", "Ivashchenko"),
    ("Їжакевич", "Yizhakevych"),
    ("Кадиївка", "Kadyivka"),
    ("Мар'їне", "Marine"),
    ("Йосипівка", "Yosypivka"),
    ("Стрий", "Stryi"),
    ("Олексій", "Oleksii"),
    ("Київ", "Kyiv"),
    ("Коваленко", "Kovalenko"),
    ("Лебедин", "Lebedyn"),
    ("Леонід", "Leonid"),
    ("Миколаїв", "Mykolaiv"),
    ("Маринич", "Marynych"),
    ("Ніжин", "Nizhyn"),
    ("Наталія", "Nataliia"),
    ("Одеса", "Odesa"),
    ("Онищенко", "Onyshchenko"),
    ("Полтава", "Poltava"),
    ("Петро", "Petro"),
    ("Решетилівка", "Reshetylivka"),
    ("Рибчинський", "Rybchynskyi"),
    ("Суми", "Sumy"),
    ("Соломія", "Solomiia"),
    ("Тернопіль", "Ternopil"),
    ("Троць", "Trots"),
    ("Ужгород", "Uzhhorod"),
    ("Уляна", "Uliana"),
    ("Фастів", "Fastiv"),
    ("Філіпчук", "Filipchuk"),
    ("Харків", "Kharkiv"),
    ("Христина", "Khrystyna"),
    ("Біла Церква", "Bila Tserkva"),
    ("Стеценко", "Stetsenko"),
    ("Чернівці", "Chernivtsi"),
    ("Шевченко", "Shevchenko"),
    ("Шостка", "Shostka"),
    ("Кишеньки", "Kyshenky"),
    ("Щербухи", "Shcherbukhy"),
    ("Гоща", "Hoshcha"),
    ("Гаращенко", "Harashchenko"),
    ("Юрій", "Yurii"),
    ("Корюківка", "Koriukivka"),
    ("Яготин", "Yahotyn"),
    ("Ярошенко", "Yaroshenko"),
    ("Костянтин", "Kostiantyn"),
    ("Знам'янка", "Znamianka"),
    ("Феодосія", "Feodosiia"),
    ("Згорани", "Zghorany"),
    ("Розгон", "Rozghon"),
]


@pytest.mark.parametrize(("ukrainian", "latin"), OFFICIAL_EXAMPLES)
def test_official_examples_of_resolution_55(ukrainian, latin):
    assert transliterate(ukrainian) == latin


def test_all_official_examples_are_covered():
    assert len(OFFICIAL_EXAMPLES) == 76


@pytest.mark.parametrize("apostrophe", ["'", "’", "ʼ"])
def test_apostrophe_variants_are_dropped_inside_words(apostrophe):
    assert transliterate(f"Знам{apostrophe}янка") == "Znamianka"


def test_upper_case_words_stay_upper_case():
    assert transliterate("ЩЕРБУХИ") == "SHCHERBUKHY"
    assert transliterate("ЗГОРАНИ") == "ZGHORANY"


def test_latin_text_and_digits_are_unchanged():
    assert transliterate("Kyiv 01001, вул. Хрещатик 22") == "Kyiv 01001, vul. Khreshchatyk 22"
    assert transliterate("Only latin") == "Only latin"
