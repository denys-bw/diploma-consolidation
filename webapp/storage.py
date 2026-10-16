from __future__ import annotations

import hashlib
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

UPLOADS = "uploads"
RUNS = "runs"
SAFE_SUFFIX = re.compile(r"^\.[a-z0-9]{1,10}$")


@dataclass(frozen=True)
class StoredFile:
    name: str
    size: int
    sha256: str


class Storage:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.uploads = self.root / UPLOADS
        self.runs = self.root / RUNS

    def save_upload(self, data: bytes, original_name: str) -> StoredFile:
        self.uploads.mkdir(parents=True, exist_ok=True)
        suffix = PurePosixPath(original_name.replace("\\", "/")).suffix.lower()
        name = uuid.uuid4().hex + (suffix if SAFE_SUFFIX.match(suffix) else "")
        (self.uploads / name).write_bytes(data)
        return StoredFile(name, len(data), hashlib.sha256(data).hexdigest())

    def read_upload(self, name: str) -> bytes:
        return (self.uploads / name).read_bytes()

    def delete_upload(self, name: str) -> None:
        (self.uploads / name).unlink(missing_ok=True)

    def run_directory(self, run_id: int) -> Path:
        return self.runs / str(run_id)

    def delete_run(self, run_id: int) -> None:
        shutil.rmtree(self.run_directory(run_id), ignore_errors=True)
