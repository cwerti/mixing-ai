"""Слой гармонии/дабла: высота, уровень, отставание, панорама и то, что он не трогает модель и подбор."""
import sys
from pathlib import Path

import librosa
import numpy as np
import pytest
import torch

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.audio.dsp_engine import DSPEngine
from app.audio.vocal_enhancer import VocalEnhancer
from app.dataset.dataset_schema import (
    CHAIN_ORDER, MODEL_PLUGINS, PARAMETER_VECTOR_LENGTH, RENDER_ONLY_PLUGINS, RENDER_ORDER, ChainConfig, PluginConfig
)
from app.mixing.chain_optimizer import SEARCHABLE_PLUGINS, ChainOptimizer, physical_to_normalized
from synthetic_audio import synthetic_vocal

SR = 44100
F0 = 220.0
CHECKPOINT = Path(__file__).parent.parent / "data" / "models" / "v4" / "best_model.pth"


def _tone(f0: float = F0, seconds: float = 2.0, harmonics: int = 6) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.1 * sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, harmonics + 1))).astype(np.float32)


def _f0_hz(y: np.ndarray) -> float:
    """Основной тон по YIN (пик спектра не годится: с сохранением формант основной тон слабее гармоник)."""
    return float(np.median(librosa.yin(y[SR // 2:SR // 2 + SR], fmin=60.0, fmax=1500.0, sr=SR)))


def _rms_db(y: np.ndarray) -> float:
    return float(20.0 * np.log10(np.sqrt(np.mean(y ** 2))))


def _harmony(**phys: float) -> PluginConfig:
    return PluginConfig(name="harmony", params={k: physical_to_normalized("harmony", k, v) for k, v in phys.items()})


@pytest.mark.parametrize("semitones", [3, 5, 7, 12, -12])
def test_layer_is_shifted_by_interval(semitones):
    y = _tone()
    layer = VocalEnhancer(sr=SR).apply_harmony(y, semitones=semitones, level_db=-8.0) - y
    assert _f0_hz(layer) == pytest.approx(F0 * 2.0 ** (semitones / 12.0), rel=0.005)


@pytest.mark.parametrize("level_db", [-3.0, -8.0, -18.0])
def test_layer_level_relative_to_lead(level_db):
    y = _tone()
    layer = VocalEnhancer(sr=SR).apply_harmony(y, semitones=5, level_db=level_db) - y
    assert _rms_db(layer) - _rms_db(y) == pytest.approx(level_db, abs=0.3)


def test_mix_zero_is_identity():
    y = _tone()
    assert np.array_equal(VocalEnhancer(sr=SR).apply_harmony(y, semitones=7, mix=0.0), y)

    dsp = DSPEngine()
    plain = dsp.apply_chain(y, SR, ChainConfig(plugins=[]))
    with_off_layer = dsp.apply_chain(y, SR, ChainConfig(plugins=[_harmony(semitones=7, mix=0.0)]))
    assert np.array_equal(plain, with_off_layer)


def test_delay_shifts_layer_in_time():
    enhancer, y = VocalEnhancer(sr=SR), _tone()
    on_time = enhancer.apply_harmony(y, semitones=3) - y
    late = enhancer.apply_harmony(y, semitones=3, delay_ms=20.0) - y
    lag = int(round(0.020 * SR))
    assert np.allclose(late[lag:], on_time[:-lag], atol=1e-6)
    assert np.allclose(late[:lag], 0.0)


def test_detune_moves_pitch_by_cents():
    y = _tone(seconds=3.0)
    layer = VocalEnhancer(sr=SR).apply_harmony(y, semitones=0, detune_cents=25.0) - y
    # +25 центов от 220 Гц = 223.2 Гц
    assert _f0_hz(layer) == pytest.approx(F0 * 2.0 ** (0.25 / 12.0), abs=1.0)


def test_formant_shift_moves_spectral_envelope():
    """Центроид спектра растет со сдвигом формант: с сохранением он ближе всего к исходному голосу."""
    y = _tone(f0=110.0, harmonics=60)
    from scipy.signal import butter, sosfilt
    y = sosfilt(butter(2, [700.0, 1400.0], btype="bandpass", fs=SR, output="sos"), y).astype(np.float32)
    enhancer = VocalEnhancer(sr=SR)

    def centroid(sig: np.ndarray) -> float:
        spectrum = np.abs(np.fft.rfft(sig[SR // 2:SR // 2 + SR]))
        freqs = np.fft.rfftfreq(SR, 1.0 / SR)
        return float(np.sum(freqs * spectrum) / np.sum(spectrum))

    shifted = [centroid(enhancer.apply_harmony(y, semitones=7, formant_shift=f) - y) for f in (0.0, 3.0, 7.0)]
    assert centroid(y) < shifted[0] < shifted[1] < shifted[2]
    assert shifted[0] - centroid(y) < 0.5 * (shifted[2] - centroid(y))


def test_width_makes_two_panned_voices():
    y = _tone()
    out = VocalEnhancer(sr=SR).apply_harmony(y, semitones=12, level_db=-6.0, delay_ms=15.0, detune_cents=10.0,
                                             width=1.0)
    assert out.shape == (2, len(y))
    layer = out - y
    assert _rms_db(layer[0] - layer[1]) > _rms_db(layer[0]) - 3.0
    assert _rms_db(layer) - _rms_db(y) == pytest.approx(-6.0, abs=0.5)


def test_dsp_engine_renders_harmony_in_chain():
    y = synthetic_vocal(SR, seconds=3.0)
    dsp = DSPEngine()
    base = [PluginConfig(name="compressor", params={"threshold_db": 0.4, "ratio": 0.3, "attack_ms": 0.2,
                                                    "release_ms": 0.3})]
    plain = dsp.apply_chain(y, SR, ChainConfig(plugins=base))
    layered = dsp.apply_chain(y, SR, ChainConfig(plugins=base + [_harmony(semitones=3, level_db=-8.0)]))
    assert layered.shape == plain.shape
    assert np.all(np.isfinite(layered))
    assert np.max(np.abs(layered)) <= 1.0
    assert not np.allclose(layered, plain)


def test_semitones_are_rounded_to_whole_interval():
    y = _tone()
    dsp = DSPEngine()
    exact = dsp.apply_chain(y, SR, ChainConfig(plugins=[_harmony(semitones=3.0)]))
    off_grid = dsp.apply_chain(y, SR, ChainConfig(plugins=[_harmony(semitones=3.3)]))
    assert np.allclose(exact, off_grid, atol=1e-6)


def test_harmony_is_render_only():
    assert "harmony" in RENDER_ONLY_PLUGINS
    assert "harmony" not in MODEL_PLUGINS and "harmony" not in SEARCHABLE_PLUGINS
    assert MODEL_PLUGINS is CHAIN_ORDER and len(MODEL_PLUGINS) == 11
    assert [p for p in RENDER_ORDER if p not in RENDER_ONLY_PLUGINS] == CHAIN_ORDER
    # Параллельная шина стоит до общих send-эффектов
    assert RENDER_ORDER.index("harmony") < RENDER_ORDER.index("reverb")


def test_optimizer_keeps_fixed_harmony_without_searching_it():
    dry = synthetic_vocal(22050, seconds=4.0)
    ref = synthetic_vocal(22050, seconds=4.0, seed=1)
    layer = _harmony(semitones=3, level_db=-8.0)
    result = ChainOptimizer(sr=22050, log=lambda _: None).optimize(
        dry, ref, budget_s=2.0, excerpt_s=3.0, allow_tune=False, patience_renders=10, fixed_plugins=[layer])
    names = [p.name for p in result.chain.plugins]
    assert names.count("harmony") == 1
    assert result.chain.plugins[names.index("harmony")].params == layer.params
    assert names == sorted(names, key=RENDER_ORDER.index)

    with pytest.raises(ValueError):
        ChainOptimizer(sr=22050, log=lambda _: None).optimize(
            dry, ref, budget_s=1.0, fixed_plugins=[PluginConfig(name="reverb", params={})])


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="нет обученного чекпоинта v4")
def test_trained_checkpoint_loads_strictly_and_predicts():
    """Головы модели не изменились: чекпоинт v4 грузится без фильтрации по форме и предсказывает 11 плагинов."""
    from app.ml.model import MixingAIModel
    from app.ml.pipeline import MixingAIMLPipeline

    state = torch.load(CHECKPOINT, map_location="cpu")["model_state_dict"]
    MixingAIModel().load_state_dict(state, strict=True)

    y = synthetic_vocal(SR, seconds=4.0)
    plugins, probs, params = MixingAIMLPipeline(CHECKPOINT).predict(y, y)
    assert probs.shape == (len(MODEL_PLUGINS),)
    assert params.shape == (PARAMETER_VECTOR_LENGTH,)
    assert "harmony" not in plugins


def test_stacked_layers_are_built_from_the_lead():
    """Второй слой строится из лидера, а не из лидера с первым слоем: сумма равна сумме отдельных слоев."""
    y = _tone()
    dsp = DSPEngine()
    third, octave = _harmony(semitones=3, level_db=-12.0), _harmony(semitones=12, level_db=-12.0)
    only_third = dsp.apply_chain(y, SR, ChainConfig(plugins=[third])) - dsp.apply_chain(y, SR, ChainConfig())
    only_octave = dsp.apply_chain(y, SR, ChainConfig(plugins=[octave])) - dsp.apply_chain(y, SR, ChainConfig())
    both = dsp.apply_chain(y, SR, ChainConfig(plugins=[third, octave])) - dsp.apply_chain(y, SR, ChainConfig())
    assert np.allclose(both, only_third + only_octave, atol=2e-3)
