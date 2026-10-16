from __future__ import annotations

import gzip
import io
import zipfile
from pathlib import PurePosixPath

from consolidator.io.formats import ArchiveFormat, RawFile

MAX_UNPACKED_BYTES = 1 << 30
HIDDEN_PREFIXES = (".", "__MACOSX", "~$")


class ZipFormat(ArchiveFormat):
    suffixes = (".zip",)

    def unpack(self, file: RawFile) -> list[RawFile]:
        with zipfile.ZipFile(io.BytesIO(file.data)) as archive:
            members = [info for info in archive.infolist() if _is_visible_file(info)]
            if sum(info.file_size for info in members) > MAX_UNPACKED_BYTES:
                raise ValueError("Архів завеликий: після розпакування понад 1 ГБ")
            return [RawFile(info.filename, archive.read(info), f"{file.origin}/{info.filename}") for info in members]


class GzipFormat(ArchiveFormat):
    suffixes = (".gz",)

    def unpack(self, file: RawFile) -> list[RawFile]:
        return [RawFile(file.name[: -len(file.suffix)], gzip.decompress(file.data), file.origin)]


def _is_visible_file(info: zipfile.ZipInfo) -> bool:
    if info.is_dir():
        return False
    return not any(part.startswith(HIDDEN_PREFIXES) for part in PurePosixPath(info.filename).parts)
