"""
Fruity Blood Overdrive: блоб события 213 (раскладка из PyFLP ``FruityBloodOverdriveEvent``).

36 байт: u32 маркер переработанного плагина, затем pre_band, color, pre_amp, x100 (bool),
post_filter, post_gain и 8 неизвестных байт; 32 байта — то же без маркера. Ручки 0..10000.

Достоверность: раскладка — высокая (сверено с PyFLP); значения отдаются как положение ручки
0..1, перевод в Гц/дБ неизвестен. post_gain: 1.0 = единичное усиление после искажения
(PyFLP показывает это как -1..0) — не проверено на слух.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.fl.plugins._common import int32s, require_len

INTERNAL_NAME = "Fruity Blood Overdrive"
KNOB_SCALE = 10000
MARKED_LEN = 36


@dataclass(frozen=True)
class BloodOverdrive:
    """Ручки в долях 0..1."""

    pre_band: float
    color: float
    pre_amp: float
    x100: bool
    post_filter: float
    post_gain: float
    raw: tuple[int, ...]


def decode(blob: bytes) -> BloodOverdrive:
    offset = 4 if len(blob) == MARKED_LEN else 0
    require_len(blob, offset + 24, INTERNAL_NAME)
    raw = int32s(blob[offset:], signed=False)
    pre_band, color, pre_amp, x100, post_filter, post_gain = raw[:6]
    return BloodOverdrive(
        pre_band=pre_band / KNOB_SCALE, color=color / KNOB_SCALE, pre_amp=pre_amp / KNOB_SCALE,
        x100=bool(x100), post_filter=post_filter / KNOB_SCALE, post_gain=post_gain / KNOB_SCALE, raw=raw,
    )
