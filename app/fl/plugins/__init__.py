"""
Декодеры блобов (событие 213) стоковых плагинов FL Studio по внутреннему имени плагина.

Неизвестные плагины не декодируются: в слоте остаются сырые байты.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.fl.plugins import blood_overdrive, eq2, soundgoodizer, stereo_enhancer, wrapper
from app.fl.plugins._common import PluginDecodeError

DECODERS: dict[str, Callable[[bytes], Any]] = {
    eq2.INTERNAL_NAME: eq2.decode,
    soundgoodizer.INTERNAL_NAME: soundgoodizer.decode,
    blood_overdrive.INTERNAL_NAME: blood_overdrive.decode,
    stereo_enhancer.INTERNAL_NAME: stereo_enhancer.decode,
    wrapper.INTERNAL_NAME: wrapper.decode,
}

__all__ = ["DECODERS", "PluginDecodeError", "decode_plugin"]


def decode_plugin(internal_name: str, blob: bytes) -> Any | None:
    """Декодирует блоб известного плагина; ``None`` для неизвестных. Ошибки раскладки — ``PluginDecodeError``."""
    decoder = DECODERS.get(internal_name)
    return decoder(blob) if decoder else None
