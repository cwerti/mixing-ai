"""
Soundgoodizer: блоб события 213 (раскладка из PyFLP ``SoundgoodizerEvent``).

12 байт: u32 маркер версии, u32 режим (0..3 = A..D), u32 amount (0..1000); 8 байт — без маркера.

Достоверность: раскладка — высокая (сверено с PyFLP); проценты предполагают линейное
отображение ручки (PyFLP называет кривую логарифмической) — не проверено.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.fl.plugins._common import int32s, require_len

INTERNAL_NAME = "Soundgoodizer"
MODES = "ABCD"


@dataclass(frozen=True)
class Soundgoodizer:
    mode: str
    amount_raw: int

    @property
    def amount(self) -> float:
        """Положение ручки 0..1."""
        return self.amount_raw / 1000

    @property
    def amount_percent(self) -> float:
        return self.amount_raw / 10


def decode(blob: bytes) -> Soundgoodizer:
    require_len(blob, 8, INTERNAL_NAME)
    ints = int32s(blob, signed=False)
    if len(blob) >= 12:
        ints = ints[1:]
    mode, amount = ints[:2]
    return Soundgoodizer(mode=MODES[mode] if mode < len(MODES) else f"unknown({mode})", amount_raw=amount)
