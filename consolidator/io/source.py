from __future__ import annotations

from dataclasses import dataclass, field, replace

import pandas as pd

ROW_NUMBER = "№ рядка"


@dataclass(frozen=True, eq=False)
class Source:
    name: str
    frame: pd.DataFrame
    origin: str
    format: str
    details: dict[str, str] = field(default_factory=dict)

    @property
    def columns(self) -> list[str]:
        return [str(column) for column in self.frame.columns]

    @property
    def n_records(self) -> int:
        return len(self.frame)

    def renamed(self, name: str) -> Source:
        return replace(self, name=name)

    def split_by(self, column: str) -> list[Source]:
        if column not in self.frame.columns:
            raise KeyError(f"У джерелі «{self.name}» немає колонки «{column}»")
        labels = self.frame[column].fillna("").astype(str).str.strip()
        if (labels == "").any():
            raise ValueError(f"Колонка «{column}» має порожні значення, тому не визначає джерело кожного запису")
        frame = self.frame.drop(columns=column)
        parts = []
        for label in pd.unique(labels):
            rows = frame[(labels == label).to_numpy()]
            parts.append(
                replace(
                    self,
                    name=label,
                    frame=rows[filled_columns(rows)],
                    details={**self.details, "split_from": self.name, "split_by": column},
                )
            )
        return parts

    def numbered(self) -> Source:
        return replace(self, frame=self.frame.set_axis(pd.RangeIndex(1, len(self.frame) + 1, name=ROW_NUMBER)))

    def keyed_by(self, column: str) -> Source:
        if column not in self.frame.columns:
            raise KeyError(f"У джерелі «{self.name}» немає колонки «{column}»")
        keys = self.frame[column].fillna("").astype(str).str.strip()
        if (keys == "").any():
            raise ValueError(f"Колонка «{column}» має порожні значення, тому не може бути ключем джерела «{self.name}»")
        duplicated = keys[keys.duplicated()].unique()
        if len(duplicated):
            raise ValueError(
                f"Колонка «{column}» не може бути ключем джерела «{self.name}»: "
                f"повторюються значення {', '.join(map(str, duplicated[:5]))}"
            )
        frame = self.frame.drop(columns=column)
        frame.index = pd.Index(keys.to_numpy(), name=column)
        return replace(self, frame=frame)


def filled_columns(frame: pd.DataFrame) -> list[str]:
    text = frame.fillna("").astype(str)
    return [column for column in frame.columns if text[column].str.strip().ne("").any()]
