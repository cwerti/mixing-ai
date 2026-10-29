import math
import struct
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.fl.plugins import (
    PluginDecodeError, blood_overdrive, decode_plugin, eq2, soundgoodizer, stereo_enhancer, wrapper,
)


def eq2_blob(bands: dict[int, tuple[int, int, int, int, int]], main_level: int = 0, version: int = 8) -> bytes:
    """Синтетический блоб EQ2: {номер полосы 0..6: (level, freq, width, type, order)}."""
    bank = [0] * 36
    for i, (level, freq, width, kind, order) in bands.items():
        for k, value in enumerate((level, freq, width, kind, order)):
            bank[k * 7 + i] = value
    bank[35] = main_level
    tail = [0] * 15
    return struct.pack(f"<{1 + 72 + len(tail)}i", version, *bank, *bank, *tail)


def test_eq2_units():
    blob = eq2_blob({
        0: (0, 0, 0, 3, 0),                     # HP 20 Hz, 12 dB/oct
        1: (-263, 32768, 32768, 6, 0),          # peak -2.63 dB, ~632 Hz, 2.55 oct
        2: (1800, 65536, 65536, 7, 0),          # high shelf +18 dB at 20 kHz
        3: (0, 32768, 0, 6, 0),                 # 0 dB peak -> не активна
        6: (0, 49152, 0, 1, -3),                # LP, order -3 -> 48 dB/oct
    }, main_level=-150)
    eq = eq2.decode(blob)
    assert eq.data_version == 8 and len(eq.bands) == 7
    hp, peak, shelf, flat, *_rest, lp = eq.bands
    assert hp.type == "high pass" and hp.freq_hz == pytest.approx(20.0) and hp.slope_db_per_oct == 12
    assert peak.type == "peaking" and peak.gain_db == pytest.approx(-2.63)
    assert peak.freq_hz == pytest.approx(20 * math.sqrt(1000)) and peak.width_oct == pytest.approx(2.55)
    assert peak.q == pytest.approx(1 / (2 * math.sinh(math.log(2) / 2 * 2.55)))
    assert shelf.freq_hz == pytest.approx(20000.0) and shelf.gain_db == pytest.approx(18.0)
    assert lp.type == "low pass" and lp.slope_db_per_oct == 48
    assert not flat.is_active and [b.index for b in eq.active_bands] == [1, 2, 3, 7]
    assert eq.main_level_db == pytest.approx(-1.5)
    assert eq.spare_bands == eq.bands and len(eq.global_raw) == 15
    assert eq2.summary(eq).startswith("HIGH PASS 12 dB/oct 20 Hz, -2.63 dB peaking 632 Hz")


def test_eq2_too_short():
    with pytest.raises(PluginDecodeError):
        eq2.decode(bytes(100))


def test_soundgoodizer_with_and_without_marker():
    sg = soundgoodizer.decode(struct.pack("<3I", 3, 2, 368))
    assert sg.mode == "C" and sg.amount_raw == 368 and sg.amount_percent == pytest.approx(36.8)
    assert soundgoodizer.decode(struct.pack("<2I", 1, 500)).mode == "B"


def test_blood_overdrive_marker():
    values = (100, 6500, 2000, 1, 3000, 10000, 0, 0)
    marked = blood_overdrive.decode(struct.pack("<9I", 3, *values))
    plain = blood_overdrive.decode(struct.pack("<8I", *values))
    assert marked == plain
    assert marked.color == pytest.approx(0.65) and marked.x100 and marked.post_gain == pytest.approx(1.0)


def test_stereo_enhancer():
    se = stereo_enhancer.decode(struct.pack("<6i", -64, 128, -96, 248, 1, 2))
    assert se.pan == pytest.approx(-0.5) and se.volume_db == pytest.approx(20 * math.log10(0.5))
    assert se.separation == pytest.approx(-1.0) and se.phase_offset_ms == pytest.approx(242.1875)
    assert se.effect_position == "post" and se.phase_inversion == "right"
    assert stereo_enhancer.decode(struct.pack("<6i", 0, 0, 0, 0, 0, 0)).volume_db is None


def _record(rid: int, data: bytes) -> bytes:
    return struct.pack("<IQ", rid, len(data)) + data


def test_wrapper_records():
    path = b"C:\\VST\\GGate.dll"
    blob = struct.pack("<I", 10) + _record(54, b"GGate\x00") + _record(55, path) + _record(53, b"\x01\x02")
    info = wrapper.decode(blob)
    assert (info.kind, info.name, info.vendor, info.state) == (10, "GGate", None, b"\x01\x02")
    assert info.path == path.decode()
    with pytest.raises(PluginDecodeError):
        wrapper.decode(struct.pack("<I", 10) + struct.pack("<IQ", 54, 1000) + b"x")


def test_unknown_plugin_not_decoded():
    assert decode_plugin("Fruity Limiter", b"\x07\x00\x00\x00") is None
