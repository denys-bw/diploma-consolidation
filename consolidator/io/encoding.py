from __future__ import annotations

import codecs
from dataclasses import dataclass

import numpy as np

BYTE_ORDER_MARKS = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)
CYRILLIC_RUN_SHARE = 0.5
SAMPLE_BYTES = 1 << 20
ERROR_HANDLER = "consolidator.latin1"


def _latin1_fallback(error: UnicodeError) -> tuple[str, int]:
    if not isinstance(error, UnicodeDecodeError):
        raise error
    return error.object[error.start : error.end].decode("latin-1"), error.end


codecs.register_error(ERROR_HANDLER, _latin1_fallback)


@dataclass(frozen=True)
class DecodedText:
    text: str
    encoding: str


def decode_bytes(data: bytes) -> DecodedText:
    for mark, encoding in BYTE_ORDER_MARKS:
        if data.startswith(mark):
            return DecodedText(data.decode(encoding), encoding)
    if _is_utf8(data):
        return DecodedText(data.decode("utf-8"), "utf-8")
    lines = data.splitlines(keepends=True)
    valid = [_is_utf8(line) for line in lines]
    fallback = guess_single_byte_encoding(b"".join(line for line, ok in zip(lines, valid) if not ok))
    if any(ok and not line.isascii() for line, ok in zip(lines, valid)):
        parts = (line.decode("utf-8") if ok else decode_single_byte(line, fallback) for line, ok in zip(lines, valid))
        return DecodedText("".join(parts), f"utf-8+{fallback}")
    return DecodedText(decode_single_byte(data, fallback), fallback)


def guess_single_byte_encoding(data: bytes) -> str:
    sample = np.frombuffer(data[:SAMPLE_BYTES], dtype=np.uint8)
    high = sample >= 0x80
    if not high.any():
        return "cp1252"
    neighbour = np.zeros_like(high)
    neighbour[1:] |= high[:-1]
    neighbour[:-1] |= high[1:]
    run_share = (high & neighbour).sum() / high.sum()
    return "cp1251" if run_share >= CYRILLIC_RUN_SHARE else "cp1252"


def decode_single_byte(data: bytes, encoding: str) -> str:
    return data.decode(encoding, errors=ERROR_HANDLER)


def _is_utf8(data: bytes) -> bool:
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True
