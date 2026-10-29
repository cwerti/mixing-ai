"""
Fruity Stereo Enhancer: блоб события 213 (раскладка из PyFLP ``FruityStereoEnhancerEvent``).

24 байта, int32 LE: pan (-128..127), volume (256 = 0 dB, 0..320), stereo_separation
(-96 = 100% разделение/шире .. +96 = 100% моно), phase_offset (±512), effect_position
(0 pre, 1 post), phase_inversion (0 нет, 1 левый, 2 правый).

Достоверность: раскладка — высокая; громкость в дБ предполагает линейную шкалу (вероятно);
phase_offset -> ±500 мс и знак канала (плюс = задержка правого) — догадка.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from app.fl.plugins._common import int32s, require_len

INTERNAL_NAME = "Fruity Stereo Enhancer"
POSITIONS = {0: "pre", 1: "post"}
INVERSIONS = {0: "none", 1: "left", 2: "right"}


@dataclass(frozen=True)
class StereoEnhancer:
    pan: float
    """-1 (лево) .. ~+1 (право)."""
    volume_db: float | None
    """None при нулевой громкости (-inf)."""
    separation: float
    """-1 = максимально широко, 0 = без изменений, +1 = моно."""
    phase_offset_ms: float
    """Положительное — задержан правый канал (догадка)."""
    effect_position: str
    phase_inversion: str
    raw: tuple[int, ...]


def decode(blob: bytes) -> StereoEnhancer:
    require_len(blob, 24, INTERNAL_NAME)
    raw = int32s(blob)[:6]
    pan, volume, separation, phase, position, inversion = raw
    gain = volume / 256
    return StereoEnhancer(
        pan=pan / 128, volume_db=20 * math.log10(gain) if gain > 0 else None, separation=separation / 96,
        phase_offset_ms=phase / 512 * 500, effect_position=POSITIONS.get(position, f"unknown({position})"),
        phase_inversion=INVERSIONS.get(inversion, f"unknown({inversion})"), raw=raw,
    )
