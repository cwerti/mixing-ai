import struct
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.bridge.event_parser import parse_fl_events
from app.fl.events import FLParseError, parse_bytes, read_fl_file

CHE = Path(__file__).parent.parent / "data/raw/presets/che/CHE_PRESET_ALL_STOCK_PLUGINS__PROD.KENZO_.fst"


def varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out)


def fl_bytes(body: bytes, fmt: int = 0) -> bytes:
    return b"FLhd" + struct.pack("<IhHH", 6, fmt, 1, 96) + b"FLdt" + struct.pack("<I", len(body)) + body


def text_event(eid: int, s: str) -> bytes:
    data = (s + "\x00").encode("utf-16-le")
    return bytes([eid]) + varint(len(data)) + data


def test_event_size_ranges():
    """0–63 байт, 64–127 слово, 128–191 dword, 192–255 varint + данные."""
    body = (bytes([0, 0x7F]) + bytes([63, 1])
            + bytes([64]) + struct.pack("<H", 0xBEEF) + bytes([127]) + struct.pack("<H", 2)
            + bytes([128]) + struct.pack("<I", 0xDEADBEEF) + bytes([191]) + struct.pack("<I", 3)
            + bytes([192]) + varint(3) + b"abc" + bytes([255]) + varint(200) + bytes(range(200)))
    f = parse_bytes(fl_bytes(body, fmt=5))
    assert f.format == 5 and f.channels == 1 and f.ppq == 96
    assert [(e.id, e.value) for e in f.events] == [
        (0, 0x7F), (63, 1), (64, 0xBEEF), (127, 2), (128, 0xDEADBEEF), (191, 3),
        (192, b"abc"), (255, bytes(range(200))),
    ]
    assert f.events[-1].offset == len(body) - 200 - 3 + 22  # смещение от начала файла
    assert f.quirks == ()


def test_text_decoding():
    f = parse_bytes(fl_bytes(text_event(204, "Main Voc Part 1")))
    assert f.events[0].text == "Main Voc Part 1"


@pytest.mark.parametrize("body", [
    bytes([70, 1]),                      # слово без второго байта
    bytes([130, 1, 2]),                  # dword обрезан
    bytes([213]) + varint(10) + b"x",    # данные длиннее чанка
    bytes([213, 0x80, 0x80]),            # незавершённый varint
])
def test_truncated_streams_raise(body):
    with pytest.raises(FLParseError):
        parse_bytes(fl_bytes(body))


def test_bad_header_raises():
    with pytest.raises(FLParseError):
        parse_bytes(b"RIFF" + bytes(20))
    with pytest.raises(FLParseError):
        parse_bytes(fl_bytes(b"")[:-4] + struct.pack("<I", 100))


def test_short_172_quirk():
    """Укороченное до 3 байт событие 172 перед UTF-16 строкой читается как 3 байта."""
    body = (bytes([28, 1, 172, 1, 1, 0]) + text_event(192, "FL Studio 26")
            + text_event(204, "LEAD BUS") + bytes([236, 0]))
    f = parse_bytes(fl_bytes(body, fmt=64))
    assert [e.id for e in f.events] == [28, 172, 192, 204, 236]
    assert f.events[1].value == 0x0101
    assert len(f.quirks) == 1


def test_normal_172_not_shortened():
    body = bytes([172]) + struct.pack("<I", 7) + text_event(192, "FL Studio")
    f = parse_bytes(fl_bytes(body))
    assert [(e.id, e.value) for e in f.events][:1] == [(172, 7)]
    assert f.quirks == ()


@pytest.mark.skipif(not CHE.exists(), reason="CHE preset not available (data/ is gitignored)")
def test_che_header_resync():
    f = read_fl_file(CHE)
    assert f.format == 64
    ids = [e.id for e in f.events]
    assert ids[:6] == [199, 159, 169, 28, 172, 192]
    assert f.events[5].text.startswith("FL Studio 26")
    assert 204 in ids and 236 in ids


def test_bridge_event_parser_redirect(tmp_path):
    path = tmp_path / "x.fst"
    path.write_bytes(fl_bytes(bytes([64]) + struct.pack("<H", 300) + bytes([150]) + struct.pack("<I", 70000)))
    assert parse_fl_events(path) == [(64, 300), (150, 70000)]
    bad = tmp_path / "bad.fst"
    bad.write_bytes(b"not an fl file")
    assert parse_fl_events(bad) == []
