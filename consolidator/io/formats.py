from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import PurePosixPath

from consolidator.io.source import Source


@dataclass(frozen=True)
class RawFile:
    name: str
    data: bytes
    origin: str

    @property
    def suffix(self) -> str:
        return suffix_of(self.name)


@dataclass(frozen=True)
class ReadOptions:
    delimiter: str | None = None
    concatenate: frozenset[str] = frozenset()


class FileFormat(ABC):
    suffixes: tuple[str, ...] = ()

    def recognizes(self, file: RawFile) -> bool:
        return file.suffix in self.suffixes


class TableFormat(FileFormat):
    @abstractmethod
    def read(self, file: RawFile, options: ReadOptions) -> list[Source]:
        raise NotImplementedError


class ArchiveFormat(FileFormat):
    @abstractmethod
    def unpack(self, file: RawFile) -> list[RawFile]:
        raise NotImplementedError


def suffix_of(filename: str) -> str:
    return PurePosixPath(filename.replace("\\", "/")).suffix.lower()
