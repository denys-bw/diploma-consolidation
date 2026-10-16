import numpy as np
import pandas as pd

from consolidator.io import Source
from consolidator.profiling import ColumnType, profile_frame
from consolidator.schema import (
    ColumnRef,
    DuplicateEvidence,
    NameRelation,
    identity_mapping,
    match_schemas,
    match_sources,
    name_relation,
    positional_names,
    readable_name,
)

SYLLABLES = ["ka", "lo", "mi", "ra", "to", "ve", "sa", "ni", "do", "be", "ru", "an", "el", "or", "ta", "zi"]
STREETS = ["Main", "Oak", "Pine", "Maple", "Cedar", "Elm", "Lake", "Hill", "River", "Park"]
CITIES = ["Springfield", "Riverton", "Lakeside", "Fairview", "Greenville", "Bristol"]


def types_of(frames):
    return {source: profile_frame(frame).types for source, frame in frames.items()}


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


def test_readable_names():
    assert readable_name("https://www.scads.de/movieBenchmark/ontology/startYear") == "start year"
    assert readable_name("Purple_binder-service.type") == "purple binder service type"
    assert readable_name("Телефон") == "telefon"


def test_name_relations_are_structural():
    assert name_relation("Phone", "phone") is NameRelation.SAME
    assert name_relation("first_name", "NameFirst") is NameRelation.WORDS
    assert name_relation("addr", "address") is NameRelation.ABBREVIATES
    assert name_relation("phone", "phone number") is NameRelation.CONTAINS
    assert name_relation("title", "name") is NameRelation.NONE
    assert name_relation("dob", "Date of birth") is NameRelation.ABBREVIATES
    assert name_relation("E-mail", "email") is NameRelation.WORDS
    assert positional_names(["column 1", "column 2", "name"]) == {"column 1", "column 2"}


def test_duplicates_reveal_corresponding_columns():
    base = entities(200)
    left = base.iloc[:150][["name", "street", "price"]]
    right = base.iloc[50:][["phone", "street", "name"]].set_axis(["tel", "address", "title"], axis=1)
    types = {column: ColumnType.TEXT for column in [*left.columns, *right.columns]}
    evidence = DuplicateEvidence(left, right, types, types)
    assert evidence.size >= 50
    assert evidence.correspondence(("name",), ("title",)).p_value < 1e-6
    assert evidence.correspondence(("name",), ("address",)).p_value > 0.01


def test_columns_with_different_names_are_matched_by_values():
    base = entities(120, seed=1)
    left = base[["name", "price", "city"]]
    right = base.iloc[::-1][["city", "name", "price"]].set_axis(["town", "title", "cost"], axis=1)
    frames = {"left": left, "right": right.reset_index(drop=True)}
    mapping = match_schemas(frames, types_of(frames))
    assert mapping.attribute("name").column_in("right") == "title"
    assert mapping.attribute("price").column_in("right") == "cost"
    assert mapping.attribute("city").column_in("right") == "town"
    assert all(attribute.confidence > 0.5 for attribute in mapping.attributes)


def test_without_shared_records_only_equal_names_and_unique_formats_are_matched():
    first, second = entities(80, seed=2), entities(80, seed=3)
    frames = {
        "a": first[["name", "phone", "city"]],
        "b": second[["name", "phone", "city"]].set_axis(["name", "tel", "town"], axis=1),
    }
    mapping = match_schemas(frames, types_of(frames))
    pairs = {tuple(ref.column for ref in attribute.columns) for attribute in mapping.attributes}
    assert ("name", "name") in pairs
    assert ("phone", "tel") in pairs


def test_unrelated_numeric_columns_are_not_merged():
    frames = {
        "a": pd.DataFrame({"deathYear": [f"19{value:02d}-01-01" for value in range(40)]}),
        "b": pd.DataFrame({"release_year": [f"19{value:02d}-06-01" for value in range(40, 80)]}),
    }
    mapping = match_schemas(frames, types_of(frames))
    assert len(mapping.attributes) == 2


def test_mapping_edits():
    frames = {"a": pd.DataFrame({"x": ["1"], "y": ["2"]}), "b": pd.DataFrame({"x": ["3"], "z": ["4"]})}
    types = {"a": {"x": ColumnType.TEXT, "y": ColumnType.TEXT}, "b": {"x": ColumnType.TEXT, "z": ColumnType.TEXT}}
    mapping = identity_mapping(frames, types)
    assert mapping.names == ["x", "y", "z"]
    merged = mapping.merged(["y", "z"], "yz")
    assert merged.attribute("yz").columns == (ColumnRef("a", "y"), ColumnRef("b", "z"))
    renamed = merged.renamed("x", "key")
    detached = renamed.detached("key", ColumnRef("b", "x"), "key_b")
    assert detached.attribute("key_b").sources == ("b",)
    composite = mapping.merged(["x", "y"], "xy")
    assert composite.attribute("xy").columns == (ColumnRef("a", "x + y"), ColumnRef("b", "x"))
    assert composite.apply(frames)["a"]["xy"].tolist() == ["1 2"]
    applied = merged.apply(frames)
    assert list(applied["b"].columns) == ["x", "yz"]
    assert merged.to_frame(frames).columns.tolist() == ["attribute", "type", "confidence", "a", "b"]


def test_time_column_with_a_few_remarks_is_still_matched():
    times = [f"{hour}:{minute:02d} a.m." for hour, minute in zip(range(1, 11), range(5, 55, 5))]
    flights = [f"AA-{number}" for number in range(len(times))]
    remarks = [*times[:8], "Contact Airline", "Contact Airline"]
    clean = Source("a", pd.DataFrame({"flight": flights, "dep": times}), "", "")
    noisy = Source("b", pd.DataFrame({"flight": flights, "dep": remarks, "gate": ["B1"] * len(times)}), "", "")
    mapping = match_sources([clean, noisy])
    assert {attribute.name: len(attribute.columns) for attribute in mapping.attributes} == {
        "flight": 2,
        "dep": 2,
        "gate": 1,
    }
    assert mapping.types["dep"] is ColumnType.TIME


def test_columns_are_matched_by_values_when_names_say_nothing():
    base = entities(100, seed=4)
    frames = {
        "a": base[["name", "phone"]],
        "b": base.iloc[::-1][["phone", "name"]].set_axis(["column 1", "column 2"], axis=1).reset_index(drop=True),
    }
    mapping = match_schemas(frames, types_of(frames))
    assert {tuple(ref.column for ref in attribute.columns) for attribute in mapping.attributes} == {
        ("name", "column 2"),
        ("phone", "column 1"),
    }


def test_one_column_is_matched_with_two_joined_columns():
    base = entities(120, seed=5)
    frames = {
        "a": base[["name", "street", "city"]],
        "b": pd.DataFrame({"name": base["name"], "location": base["street"] + " " + base["city"]}),
    }
    mapping = match_schemas(frames, types_of(frames))
    joined = next(attribute for attribute in mapping.attributes if ColumnRef("b", "location") in attribute.columns)
    assert ColumnRef("a", "street + city") in joined.columns
    assert mapping.apply(frames)["a"][joined.name].iloc[0] == f"{base['street'][0]} {base['city'][0]}"


def test_a_column_goes_to_the_counterpart_it_resembles_most():
    generator = np.random.default_rng(3)
    first = generator.choice(["Ivan", "Olena", "Petro", "Maria", "Taras", "Iryna", "Oleh", "Anna"], 300)
    last = generator.choice(["Petrenko", "Kovalenko", "Bondarenko", "Melnyk", "Boiko", "Moroz", "Lysenko"], 300)
    mail = [f"{a.lower()}.{b.lower()}{generator.integers(1, 99)}@mail.example" for a, b in zip(first, last)]
    left = pd.DataFrame({"full name": [f"{a} {b}" for a, b in zip(first, last)], "contact": mail}).iloc[:200]
    right = pd.DataFrame({"client": [f"{b.upper()} {a}" for a, b in zip(first, last)], "address": mail}).iloc[100:]
    frames = {"left": left, "right": right}
    mapping = match_schemas(frames, types_of(frames))
    groups = {frozenset(ref.column for ref in attribute.columns) for attribute in mapping.attributes}
    assert frozenset({"full name", "client"}) in groups
    assert frozenset({"contact", "address"}) in groups
