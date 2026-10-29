"""
Fruity Wrapper (VST-хост FL): блоб события 213.

Формат (по PyFLP): u32 тип, затем записи ``u32 id, u64 size, data``. Текстовые записи —
UTF-8: 51 FourCC, 54 имя VST, 55 путь к .dll/.vst3, 56 вендор; 53 — состояние плагина
(VST chunk, хранится как есть, не декодируется).

Достоверность: обрамление записей — средняя (по PyFLP, проверено на GGate и Auto-Tune Evo);
содержимое VST chunk не декодировано.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

from app.fl.plugins._common import PluginDecodeError, require_len

INTERNAL_NAME = "Fruity Wrapper"
RECORD_NAMES = {1: "midi", 2: "flags", 30: "io", 31: "inputs", 32: "outputs", 50: "plugin_info",
                51: "fourcc", 52: "guid", 53: "state", 54: "name", 55: "path", 56: "vendor"}
TEXT_RECORDS = frozenset({51, 54, 55, 56})
_RECORD_HEADER = struct.Struct("<IQ")


@dataclass(frozen=True)
class WrapperInfo:
    kind: int
    name: str | None
    path: str | None
    vendor: str | None
    fourcc: str | None
    state: bytes
    records: tuple[tuple[int, bytes], ...]


def decode(blob: bytes) -> WrapperInfo:
    require_len(blob, 4, INTERNAL_NAME)
    kind = struct.unpack_from("<I", blob)[0]
    records: list[tuple[int, bytes]] = []
    pos = 4
    while pos + _RECORD_HEADER.size <= len(blob):
        rid, size = _RECORD_HEADER.unpack_from(blob, pos)
        pos += _RECORD_HEADER.size
        if pos + size > len(blob):
            raise PluginDecodeError(f"{INTERNAL_NAME}: record {rid} declares {size} bytes past the blob end")
        records.append((rid, blob[pos:pos + size]))
        pos += size
    texts = {rid: data.decode("utf-8", "replace").rstrip("\x00") for rid, data in records if rid in TEXT_RECORDS}
    state = next((data for rid, data in records if rid == 53), b"")
    return WrapperInfo(kind=kind, name=texts.get(54), path=texts.get(55), vendor=texts.get(56),
                       fourcc=texts.get(51), state=state, records=tuple(records))
