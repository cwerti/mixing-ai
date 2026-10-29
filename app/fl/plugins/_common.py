"""Общие помощники декодеров блобов плагинов (событие 213)."""
from __future__ import annotations

import struct


class PluginDecodeError(ValueError):
    """Блоб плагина короче или иной структуры, чем ожидает декодер."""


def int32s(blob: bytes, signed: bool = True) -> tuple[int, ...]:
    """Блоб как последовательность int32 LE (хвост меньше 4 байт отбрасывается)."""
    n = len(blob) // 4
    return struct.unpack(f"<{n}{'i' if signed else 'I'}", blob[: n * 4])


def require_len(blob: bytes, minimum: int, plugin: str) -> None:
    if len(blob) < minimum:
        raise PluginDecodeError(f"{plugin}: blob has {len(blob)} bytes, need at least {minimum}")
