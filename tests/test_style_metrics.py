import sys
from pathlib import Path

import numpy as np
import pytest
from pedalboard import Compressor, Pedalboard, Reverb

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.audio.vocal_enhancer import VocalEnhancer
from app.eval.style_metrics import extract_style_features, style_distance
from synthetic_audio import synthetic_vocal as _synthetic_vocal

SR = 22050


def synthetic_vocal(seconds: float = 8.0, seed: int = 0, detune_cents: float = 0.0) -> np.ndarray:
    return _synthetic_vocal(SR, seconds=seconds, seed=seed, detune_cents=detune_cents)


def test_reverb_slows_decay():
    """Реверб ограничивает скорость спада громкости после слогов."""
    dry = synthetic_vocal()
    wet = Pedalboard([Reverb(room_size=0.8, wet_level=0.3, dry_level=0.8)])(dry[np.newaxis, :], SR)[0]
    f_dry = extract_style_features(dry, SR, with_pitch=False)
    f_wet = extract_style_features(wet, SR, with_pitch=False)
    assert f_wet["decay_fast_db"] > f_dry["decay_fast_db"] + 5.0


def test_compressor_narrows_dynamics():
    """Компрессор уменьшает разброс громкости между слогами."""
    dry = synthetic_vocal()
    board = Pedalboard([Compressor(threshold_db=-35.0, ratio=8.0, attack_ms=2.0, release_ms=80.0)])
    comp = board(dry[np.newaxis, :], SR)[0]
    f_dry = extract_style_features(dry, SR, with_pitch=False)
    f_comp = extract_style_features(comp, SR, with_pitch=False)
    assert f_comp["dr_p95_p10_db"] < f_dry["dr_p95_p10_db"] - 3.0


def test_stereo_enhancer_adds_width():
    """Эффект Хааса дает ненулевую ширину, моно-сигнал — нулевую."""
    dry = synthetic_vocal()
    wide = VocalEnhancer(sr=SR).apply_stereo_enhancer(dry, delay_ms=15.0, width=1.0)
    assert extract_style_features(dry, SR, with_pitch=False)["width_global"] == 0.0
    assert extract_style_features(wide, SR, with_pitch=False)["width_global"] > 0.1


def test_autotune_reduces_grid_deviation():
    """Жесткий тюн притягивает расстроенные ноты к сетке."""
    detuned = synthetic_vocal(detune_cents=30.0, seed=1)
    tuned = VocalEnhancer(sr=SR).apply_autotune(detuned, key_index=0, scale="major", speed=1.0)
    cents_before = extract_style_features(detuned, SR)["pitch_cents"]
    cents_after = extract_style_features(tuned, SR)["pitch_cents"]
    assert cents_after < cents_before - 5.0


def test_style_distance_zero_for_identical_and_signed():
    """Расстояние до самого себя равно нулю, а разница по признаку несет знак «результат - референс»."""
    dry = synthetic_vocal()
    wet = Pedalboard([Reverb(room_size=0.8, wet_level=0.3, dry_level=0.8)])(dry[np.newaxis, :], SR)[0]
    f_dry = extract_style_features(dry, SR, with_pitch=False)
    f_wet = extract_style_features(wet, SR, with_pitch=False)

    total_same, _ = style_distance(f_dry, f_dry)
    total_diff, per_feature = style_distance(f_wet, f_dry)
    assert total_same == pytest.approx(0.0)
    assert total_diff > 0.5
    assert per_feature["decay_fast_db"] > 0
