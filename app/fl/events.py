"""
Чтение потока событий FL Studio (.flp проекты и .fst пресеты).

Формат: чанк ``FLhd`` (u32 длина, i16 format, u16 channels, u16 ppq), затем чанк ``FLdt``
(u32 длина) с событиями. Размер данных события задаётся его id:

* 0–63    — 1 байт;
* 64–127  — 2 байта (u16 LE);
* 128–191 — 4 байта (u32 LE);
* 192–255 — varint-длина (7 бит на байт, LE) и столько же байт данных.

Файлы считаются недоверенными: любые выходы за границы дают ``FLParseError``, код из файла
не исполняется, данные остаются байтами.

Известная особенность: в .fst пресетах вставки микшера из FL 26 (format 64) заголовочное
событие 172 записано с 3 байтами вместо 4. Строгий разбор при этом не падает, а «съезжает»
и случайно выравнивается к концу чанка, теряя 204/236/201. Поэтому для событий из
``SHORT_DWORD_CANDIDATES`` читатель выбирает 3 байта, если после них стоит корректная
UTF-16 строка, а 4-байтное прочтение ведёт на событие без данных. Срабатывания
записываются в ``FLFile.quirks``.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

BYTE_MAX = 63
WORD_MAX = 127
DWORD_MAX = 191

# Id 4-байтных событий, которые встречались укороченными до 3 байт (см. docstring модуля).
SHORT_DWORD_CANDIDATES = frozenset({172})


class FLParseError(ValueError):
    """Файл не является корректным потоком событий FL Studio."""


@dataclass(frozen=True)
class Event:
    """Одно событие потока: id, значение (int для фиксированных размеров, bytes для данных) и смещение."""

    id: int
    value: int | bytes
    offset: int

    @property
    def is_data(self) -> bool:
        return self.id > DWORD_MAX

    @property
    def data(self) -> bytes:
        """Данные события переменной длины; для фиксированных событий — ошибка."""
        if not isinstance(self.value, bytes):
            raise TypeError(f"event {self.id} carries an int, not bytes")
        return self.value

    @property
    def text(self) -> str:
        return decode_text(self.data)


@dataclass(frozen=True)
class FLFile:
    """Разобранный файл: поля заголовка, события и список применённых обходов (offset, описание)."""

    format: int
    channels: int
    ppq: int
    events: tuple[Event, ...]
    quirks: tuple[tuple[int, str], ...] = field(default=())


def decode_text(data: bytes) -> str:
    """Строка FL: UTF-16 LE с нулевым терминатором (старые версии — однобайтная)."""
    if len(data) % 2 == 0:
        try:
            return data.decode("utf-16-le").rstrip("\x00")
        except UnicodeDecodeError:
            pass
    return data.decode("latin-1").rstrip("\x00")


def _read_varint(data: bytes, pos: int, end: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        if pos >= end:
            raise FLParseError(f"truncated varint at {pos}")
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7
        if shift > 35:
            raise FLParseError(f"varint too long at {pos}")


def _is_utf16_text_event(data: bytes, pos: int, end: int) -> bool:
    """Начинается ли в ``pos`` событие с данными, похожими на UTF-16 строку с терминатором."""
    if pos >= end or data[pos] <= DWORD_MAX:
        return False
    try:
        size, start = _read_varint(data, pos + 1, end)
    except FLParseError:
        return False
    payload = data[start:start + size]
    if size < 4 or size % 2 or start + size > end or payload[-2:] != b"\x00\x00":
        return False
    return decode_text(payload).isprintable()


def _dword_size(eid: int, data: bytes, pos: int, end: int) -> int:
    if eid not in SHORT_DWORD_CANDIDATES:
        return 4
    full_read_hits_data_event = pos + 4 < end and data[pos + 4] > DWORD_MAX
    if not full_read_hits_data_event and _is_utf16_text_event(data, pos + 3, end):
        return 3
    return 4


def parse_events(data: bytes, start: int, end: int) -> tuple[list[Event], list[tuple[int, str]]]:
    """Разбирает события в ``data[start:end]``; возвращает события и сработавшие обходы."""
    events: list[Event] = []
    quirks: list[tuple[int, str]] = []
    pos = start
    while pos < end:
        offset = pos
        eid = data[pos]
        pos += 1
        if eid <= WORD_MAX:
            size = 1 if eid <= BYTE_MAX else 2
            if pos + size > end:
                raise FLParseError(f"event {eid} at {offset} overruns the data chunk")
            value: int | bytes = int.from_bytes(data[pos:pos + size], "little")
        elif eid <= DWORD_MAX:
            size = _dword_size(eid, data, pos, end)
            if pos + size > end:
                raise FLParseError(f"event {eid} at {offset} overruns the data chunk")
            if size != 4:
                quirks.append((offset, f"event {eid} read as {size} bytes"))
            value = int.from_bytes(data[pos:pos + size], "little")
        else:
            size, pos = _read_varint(data, pos, end)
            if pos + size > end:
                raise FLParseError(f"event {eid} at {offset} declares {size} bytes past the data chunk")
            value = data[pos:pos + size]
        pos += size
        events.append(Event(eid, value, offset))
    return events, quirks


def parse_bytes(data: bytes) -> FLFile:
    """Разбирает содержимое .flp/.fst файла."""
    if len(data) < 14 or data[:4] != b"FLhd":
        raise FLParseError("missing FLhd header")
    header_len = struct.unpack_from("<I", data, 4)[0]
    if header_len < 6:
        raise FLParseError(f"FLhd header too short: {header_len}")
    fmt, channels, ppq = struct.unpack_from("<hHH", data, 8)
    pos = 8 + header_len
    if data[pos:pos + 4] != b"FLdt" or pos + 8 > len(data):
        raise FLParseError("missing FLdt chunk")
    data_len = struct.unpack_from("<I", data, pos + 4)[0]
    start = pos + 8
    end = start + data_len
    if end > len(data):
        raise FLParseError(f"FLdt declares {data_len} bytes, file has {len(data) - start}")
    events, quirks = parse_events(data, start, end)
    return FLFile(fmt, channels, ppq, tuple(events), tuple(quirks))


def read_fl_file(path: str | Path) -> FLFile:
    """Читает и разбирает .flp/.fst файл."""
    with open(Path(path), "rb") as f:
        return parse_bytes(f.read())
