"""
Fruity Parametric EQ 2: блоб события 213, int32 LE (версии данных 6 и 8 совпадают по раскладке).

Раскладка:
    [0]      версия данных
    [1:8]    уровень полосы, 0.01 dB (знаковый)
    [8:15]   частота, 0..65536 -> 20 * 1000 ** (v / 65536) Гц
    [15:22]  ширина, 0..65536 -> 0.1 + 4.9 * v / 65536 октав
    [22:29]  тип: 0 off, 1 LP, 2 BP, 3 HP, 4 notch, 5 low shelf, 6 peak, 7 high shelf
    [29:36]  порядок, -3..3 (0 = 2-й порядок, 12 dB/oct)
    [36]     основной уровень (та же шкала, что и у полос)
    [37:73]  запасной банк «Compare» с той же раскладкой из 36 int
    [73:]    глобальные опции (не декодированы, хранятся как есть)

Достоверность:
    подтверждено — раскладка, коды типов, частота, усиление (raw/100, упор ровно в -18 dB);
    вероятно     — ширина в октавах / Q для peak, notch, band pass; модуль порядка (|n| -> 4/6/8 полюсов);
    догадка      — знак порядка (отрицательный = steep), main_level_db;
    только raw   — ширина на полках и LP/HP (резонанс без физической единицы), глобальные опции.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from app.fl.plugins._common import int32s, require_len

INTERNAL_NAME = "Fruity Parametric EQ 2"
BAND_COUNT = 7
FULL_SCALE = 65536
BANK_INTS = 36
MIN_INTS = 1 + 2 * BANK_INTS

BAND_TYPES = ("off", "low pass", "band pass", "high pass", "notch", "low shelf", "peaking", "high shelf")
GAINLESS_TYPES = frozenset({"off", "low pass", "band pass", "high pass", "notch"})


def freq_hz(raw: int) -> float:
    return 20.0 * 1000.0 ** (raw / FULL_SCALE)


def width_octaves(raw: int) -> float:
    return 0.1 + 4.9 * raw / FULL_SCALE


def q_from_octaves(octaves: float) -> float:
    return 1.0 / (2.0 * math.sinh(math.log(2) / 2 * octaves))


def slope_db_per_oct(order: int) -> int:
    """Крутизна среза LP/HP: 0 -> 12 dB/oct, |1|/|2|/|3| -> 24/36/48 dB/oct (вероятно)."""
    poles = {0: 2, 1: 4, 2: 6, 3: 8}.get(abs(order), abs(order))
    return poles * 6


@dataclass(frozen=True)
class EQ2Band:
    """Полоса эквалайзера в физических единицах; ``raw_*`` — исходные значения."""

    index: int
    type: str
    freq_hz: float
    gain_db: float
    width_oct: float
    q: float
    order: int
    raw_level: int
    raw_freq: int
    raw_width: int
    raw_type: int

    @property
    def gain_applies(self) -> bool:
        return self.type not in GAINLESS_TYPES

    @property
    def is_active(self) -> bool:
        """Влияет ли полоса на звук: не off и не нулевой peak/shelf."""
        if self.type == "off":
            return False
        return not (self.gain_applies and self.raw_level == 0)

    @property
    def slope_db_per_oct(self) -> int:
        return slope_db_per_oct(self.order)


@dataclass(frozen=True)
class ParametricEQ2:
    data_version: int
    bands: tuple[EQ2Band, ...]
    main_level_db: float
    spare_bands: tuple[EQ2Band, ...]
    global_raw: tuple[int, ...]

    @property
    def active_bands(self) -> tuple[EQ2Band, ...]:
        return tuple(b for b in self.bands if b.is_active)


def _decode_bank(ints: tuple[int, ...], base: int) -> tuple[tuple[EQ2Band, ...], float]:
    level, freq, width, kind, order = (ints[base + k * BAND_COUNT: base + (k + 1) * BAND_COUNT] for k in range(5))
    bands = []
    for i in range(BAND_COUNT):
        name = BAND_TYPES[kind[i]] if 0 <= kind[i] < len(BAND_TYPES) else f"unknown({kind[i]})"
        octaves = width_octaves(width[i])
        bands.append(EQ2Band(
            index=i + 1, type=name, freq_hz=freq_hz(freq[i]), gain_db=level[i] / 100, width_oct=octaves,
            q=q_from_octaves(octaves), order=order[i], raw_level=level[i], raw_freq=freq[i],
            raw_width=width[i], raw_type=kind[i],
        ))
    return tuple(bands), ints[base + 5 * BAND_COUNT] / 100


def decode(blob: bytes) -> ParametricEQ2:
    require_len(blob, MIN_INTS * 4, INTERNAL_NAME)
    ints = int32s(blob)
    bands, main_level = _decode_bank(ints, 1)
    spare, _ = _decode_bank(ints, 1 + BANK_INTS)
    return ParametricEQ2(
        data_version=ints[0], bands=bands, main_level_db=main_level, spare_bands=spare,
        global_raw=ints[MIN_INTS:],
    )


def freq_raw(hz: float) -> int:
    """Обратное к freq_hz: частота в Гц -> 0..65536 (20 Гц – 20 кГц)."""
    hz = min(max(hz, 20.0), 20000.0)
    return round(FULL_SCALE * math.log(hz / 20.0) / math.log(1000.0))


def width_raw_for_q(q: float) -> int:
    """Ширина peak-полосы по Q (через октавы; формула ширины «вероятная», см. модуль)."""
    octaves = 2.0 / math.log(2) * math.asinh(1.0 / (2.0 * q))
    return round(min(max((octaves - 0.1) / 4.9, 0.0), 1.0) * FULL_SCALE)


def encode_bands(blob: bytes, bands: list[tuple[float, float]], peak_q: float | None = None) -> bytes:
    """
    Блоб EQ 2 с новыми частотами и усилениями первых len(bands) полос основного банка; остальное (типы,
    порядок, банк Compare, глобальные опции) берется из исходного блоба. bands — [(Гц, дБ), ...],
    усиление ограничено ±18 dB. peak_q — ширина peak-полос (как у DSPEngine), None — оставить как есть.
    """
    eq = decode(blob)
    ints = list(int32s(blob))
    for i, (hz, gain_db) in enumerate(bands[:BAND_COUNT]):
        ints[1 + i] = round(min(max(gain_db, -18.0), 18.0) * 100)
        ints[1 + BAND_COUNT + i] = freq_raw(hz)
        if peak_q is not None and eq.bands[i].type == "peaking":
            ints[1 + 2 * BAND_COUNT + i] = width_raw_for_q(peak_q)
    head = struct.pack(f"<{len(ints)}i", *ints)
    return head + blob[len(head):]


def summary(eq: ParametricEQ2) -> str:
    """Короткое описание активных полос, например ``HIGH PASS 12 dB/oct 97 Hz, -2.63 dB peaking 247 Hz Q0.67``."""
    parts = []
    for b in eq.active_bands:
        f = f"{b.freq_hz / 1000:.2f} kHz" if b.freq_hz >= 1000 else f"{b.freq_hz:.0f} Hz"
        if b.type in ("low pass", "high pass"):
            parts.append(f"{b.type.upper()} {b.slope_db_per_oct} dB/oct {f}")
        elif b.type in ("band pass", "notch"):
            parts.append(f"{b.type} {f} {b.width_oct:.2f} oct")
        else:
            q = f" Q{b.q:.2f}" if b.type == "peaking" else ""
            parts.append(f"{b.gain_db:+.2f} dB {b.type} {f}{q}")
    if eq.main_level_db:
        parts.append(f"main {eq.main_level_db:+.2f} dB")
    return ", ".join(parts) or "flat"
