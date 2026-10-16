from consolidator.schema.evidence import DuplicateEvidence, PairEvidence
from consolidator.schema.families import compatible, family
from consolidator.schema.mapping import Attribute, ColumnRef, SchemaMapping
from consolidator.schema.matching import identity_mapping, match_schemas, match_sources
from consolidator.schema.names import NameRelation, name_relation, positional_names, readable_name

__all__ = [
    "Attribute",
    "ColumnRef",
    "DuplicateEvidence",
    "NameRelation",
    "PairEvidence",
    "SchemaMapping",
    "compatible",
    "family",
    "identity_mapping",
    "match_schemas",
    "match_sources",
    "name_relation",
    "positional_names",
    "readable_name",
]
