import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.fl import load_project, parse_bytes
from app.fl.plugins import blood_overdrive, eq2, soundgoodizer, stereo_enhancer
from app.fl.project import build_project
from test_fl_events import fl_bytes, text_event, varint

PRESETS = Path(__file__).parent.parent / "data/raw/presets"
CHE = PRESETS / "che/CHE_PRESET_ALL_STOCK_PLUGINS__PROD.KENZO_.fst"
NINOO = PRESETS / "ninoo/Stock Plugins Free Hyperpop Vocal Preset With Harmonies By Ninoo.flp"
DECODED = PRESETS / "decoded"

needs_che = pytest.mark.skipif(not CHE.exists(), reason="CHE preset not available (data/ is gitignored)")
needs_ninoo = pytest.mark.skipif(not NINOO.exists(), reason="Ninoo preset not available (data/ is gitignored)")


def data_event(eid: int, data: bytes) -> bytes:
    return bytes([eid]) + varint(len(data)) + data


def slot(name: str, data: bytes, index: int, display: str | None = None) -> bytes:
    out = text_event(201, name) + (text_event(203, display) if display else b"") + data_event(213, data)
    return out + bytes([98]) + struct.pack("<H", index)


def test_synthetic_inserts_names_precede_flags():
    """Имя (204) стоит перед 236 своей вставки; плагин канала до микшера не попадает в слот."""
    body = (slot("Channel Synth", b"\x00", 0)[:-3]           # генератор канала без 98
            + bytes([236, 0]) + bytes([98, 0, 0])             # мастер, пустой слот 0
            + bytes([147]) + struct.pack("<I", 0)
            + text_event(204, "Vox") + bytes([236, 0])
            + slot("Soundgoodizer", struct.pack("<3I", 3, 2, 368), 0)
            + slot("Fruity Mystery", b"\xde\xad", 1, display="My FX")
            + slot("Fruity Stereo Enhancer", b"\x00" * 4, 2)  # обрезанный блоб
            + bytes([147]) + struct.pack("<I", 0))
    project = build_project(parse_bytes(fl_bytes(body)))
    master, vox = project.inserts
    assert master.name is None and master.slots == []
    assert vox.index == 1 and vox.name == "Vox"
    assert vox.chain == ["Soundgoodizer", "Fruity Mystery", "Fruity Stereo Enhancer"]
    sg, mystery, broken = vox.slots
    assert sg.decoded == soundgoodizer.Soundgoodizer("C", 368)
    assert mystery.decoded is None and mystery.data == b"\xde\xad" and mystery.display_name == "My FX"
    assert broken.decoded is None and "Stereo Enhancer" in broken.decode_error
    assert [p.internal_name for p in project.other_plugins] == ["Channel Synth"]


@needs_che
def test_che_lead_bus_chain():
    project = load_project(CHE)
    lead = project.insert_by_name("LEAD BUS")
    # Слот 0 — Pitcher: старый разбор терял его вместе с событиями после рассинхронизации на 172.
    assert lead.chain[0] == "Pitcher" and lead.slots[0].index == 0
    assert lead.chain[1:] == ["Fruity Wrapper(GGate)", "Fruity Parametric EQ 2", "Fruity Limiter", "Fruity Limiter",
                              "Fruity Parametric EQ 2", "Soundgoodizer", "Maximus"]
    assert lead.slots[1].vst_path.endswith("GGate.dll")
    eq = lead.slots[2].decoded
    assert lead.slots[2].index == 2 and isinstance(eq, eq2.ParametricEQ2)
    hp = eq.active_bands[0]
    assert hp.type == "high pass" and hp.freq_hz == pytest.approx(97, abs=1) and hp.slope_db_per_oct == 12
    assert lead.slots[3].decoded is None and len(lead.slots[3].data) == 169  # Limiter: сырые байты


@needs_ninoo
def test_ninoo_main_voc():
    project = load_project(NINOO)
    insert = project.inserts[1]
    assert insert.name == "Main Voc Part 1"
    assert insert.slots[0].vst_name == "Auto-Tune Evo VST"
    sg = next(s for s in insert.slots if s.internal_name == "Soundgoodizer")
    assert sg.index == 7 and sg.decoded.mode == "C" and sg.decoded.amount_raw == 368
    bo = next(s for s in insert.slots if s.internal_name == "Fruity Blood Overdrive").decoded
    assert isinstance(bo, blood_overdrive.BloodOverdrive) and bo.color == pytest.approx(0.65)
    se = project.insert_by_name("Back Voc Part 2").slots[-1].decoded
    assert isinstance(se, stereo_enhancer.StereoEnhancer) and se.effect_position == "post"


def _all_slots(*paths: Path):
    for path in paths:
        for insert in load_project(path).inserts:
            for s in insert.slots:
                yield path.name, insert, s


@needs_che
@needs_ninoo
@pytest.mark.skipif(not (DECODED / "eq2.json").exists(), reason="decoded reference not available")
def test_eq2_matches_reference_json():
    ref = json.loads((DECODED / "eq2.json").read_text(encoding="utf-8"))
    ours = [(src, s) for src, _, s in _all_slots(CHE, NINOO) if s.internal_name == eq2.INTERNAL_NAME]
    assert len(ours) == len(ref)
    for (src, s), r in zip(ours, ref):
        assert (src, s.index) == (r["source"], r["slot"])
        assert list(struct.unpack(f"<{len(s.data) // 4}i", s.data[: len(s.data) // 4 * 4])) == r["raw_ints"]
        for band, rb in zip(s.decoded.bands, r["active"]["bands"]):
            assert band.type == rb["type"]
            assert band.freq_hz == pytest.approx(rb["freq_hz"], abs=0.06)
            assert band.gain_db == pytest.approx(rb["gain_db"])
            assert band.width_oct == pytest.approx(rb["width_oct"], abs=1e-3)
            assert band.order == rb["order"]


@needs_che
@needs_ninoo
@pytest.mark.skipif(not (DECODED / "pyflp_plugins.json").exists(), reason="decoded reference not available")
def test_plugins_match_pyflp_reference_json():
    ref = json.loads((DECODED / "pyflp_plugins.json").read_text(encoding="utf-8"))
    ours = {(src, i.index if src == NINOO.name else 0, s.index): s for src, i, s in _all_slots(CHE, NINOO)}
    for r in ref:
        s = ours[(r["source"], r["insert"], r["slot"])]
        assert s.internal_name == r["plugin"] and len(s.data) == r["raw_size"]
        d = r["decoded"]
        if r["plugin"] == "Soundgoodizer":
            assert (s.decoded.mode, s.decoded.amount_raw) == (d["mode"]["value"], d["amount"]["raw"])
        elif r["plugin"] == "Fruity Blood Overdrive":
            assert s.decoded.raw[:6] == tuple(d[k]["raw"] for k in
                                              ("pre_band", "color", "pre_amp", "x100", "post_filter", "post_gain"))
        elif r["plugin"] == "Fruity Stereo Enhancer":
            assert s.decoded.raw == tuple(d[k]["raw"] for k in ("pan", "volume", "stereo_separation", "phase_offset",
                                                                 "effect_position", "phase_inversion"))
        elif r["plugin"] == "Fruity Wrapper":
            names = [x.get("text") for x in d["records"] if x["id"] == 54]
            assert [s.vst_name] == names
