from pathlib import Path

import pytest

from app.fl.events import read_fl_file
from app.fl.plugins import eq2
from app.presets.full_preset_generator import FullPresetGenerator

TEMPLATE = Path("FlProject/template.fst")


@pytest.mark.skipif(not TEMPLATE.exists(), reason="нет шаблона FlProject/template.fst")
def test_exported_eq_decodes_back_to_requested_bands(tmp_path):
    bands = [(80.0, -3.0), (250.0, 2.5), (500.0, -1.0), (1000.0, 0.5), (3000.0, 4.0), (8000.0, -2.0), (12000.0, 1.5)]
    path = FullPresetGenerator(str(TEMPLATE)).generate_vocal_chain(bands, "out.fst", output_dir=tmp_path)

    blobs = [e.data for e in read_fl_file(path).events if e.id == 213]
    decoded = eq2.decode(blobs[0])
    assert len(path.read_bytes()) == len(TEMPLATE.read_bytes())
    for band, (hz, gain_db) in zip(decoded.bands, bands):
        assert band.freq_hz == pytest.approx(hz, rel=0.002)
        assert band.gain_db == pytest.approx(gain_db, abs=0.01)
    assert [b.type for b in decoded.bands] == ["low shelf"] + ["peaking"] * 5 + ["high shelf"]
    assert decoded.bands[1].q == pytest.approx(FullPresetGenerator.PEAK_Q, rel=0.01)


def test_gain_is_clamped_to_eq2_range():
    template = eq2.encode_bands(bytes(4 * eq2.MIN_INTS), [(1000.0, 30.0)])
    assert eq2.decode(template).bands[0].gain_db == 18.0
