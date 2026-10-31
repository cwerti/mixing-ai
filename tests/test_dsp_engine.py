import pytest
import numpy as np
from app.audio.dsp_engine import DSPEngine
from app.dataset.dataset_schema import ChainConfig, PluginConfig

def test_norm_denorm_value():
    dsp = DSPEngine()
    
    # Normal bounds
    assert dsp.denormalize_value(0.0, -10.0, 10.0) == -10.0
    assert dsp.denormalize_value(0.5, -10.0, 10.0) == 0.0
    assert dsp.denormalize_value(1.0, -10.0, 10.0) == 10.0
    
    # Clamping
    assert dsp.denormalize_value(1.5, -10.0, 10.0) == 10.0
    assert dsp.denormalize_value(-0.5, -10.0, 10.0) == -10.0
    
    # Normalize
    assert dsp.normalize_value(0.0, -10.0, 10.0) == 0.5
    assert dsp.normalize_value(10.0, -10.0, 10.0) == 1.0
    assert dsp.normalize_value(-10.0, -10.0, 10.0) == 0.0

def test_apply_chain_length():
    dsp = DSPEngine()
    sr = 44100
    # Generate 1s of silence
    y = np.zeros(sr, dtype=np.float32)
    
    # Empty chain
    chain = ChainConfig()
    y_out = dsp.apply_chain(y, sr, chain)
    assert len(y_out) == len(y)
    
    # Reverb chain
    chain_reverb = ChainConfig(plugins=[
        PluginConfig(name="reverb", params={"room_size": 0.5, "damping": 0.5, "wet_level": 0.2, "dry_level": 0.8})
    ])
    y_out_reverb = dsp.apply_chain(y, sr, chain_reverb)
    assert len(y_out_reverb) == len(y)
    
    # EQ + Compressor + Delay chain
    chain_full = ChainConfig(plugins=[
        PluginConfig(name="eq", params={
            "band1_freq_hz": 0.1, "band1_gain_db": 0.5,
            "band2_freq_hz": 0.2, "band2_gain_db": 0.5,
            "band3_freq_hz": 0.3, "band3_gain_db": 0.5,
            "band4_freq_hz": 0.5, "band4_gain_db": 0.5,
            "band5_freq_hz": 0.6, "band5_gain_db": 0.5,
            "band6_freq_hz": 0.8, "band6_gain_db": 0.5,
            "band7_freq_hz": 0.9, "band7_gain_db": 0.5
        }),
        PluginConfig(name="compressor", params={
            "threshold_db": 0.5, "ratio": 0.2, "attack_ms": 0.1, "release_ms": 0.2
        }),
        PluginConfig(name="delay", params={
            "delay_seconds": 0.3, "feedback": 0.2, "mix": 0.1
        })
    ])
    y_out_full = dsp.apply_chain(y, sr, chain_full)
    assert len(y_out_full) == len(y)

def test_vst_fallback_chain():
    dsp = DSPEngine()
    sr = 44100
    y = np.zeros(sr, dtype=np.float32)
    
    # Chain with VST3 plugins that will trigger fallback
    chain = ChainConfig(plugins=[
        PluginConfig(name="fabfilter_pro_q_3", params={
            "band1_freq": 0.2, "band1_gain": 0.5,
            "band2_freq": 0.5, "band2_gain": 0.5
        }),
        PluginConfig(name="fabfilter_pro_c_2", params={
            "threshold": 0.5, "ratio": 0.2
        }),
        PluginConfig(name="valhalla_vintage_verb", params={
            "mix": 0.15, "decay": 0.5
        })
    ])
    
    y_out = dsp.apply_chain(y, sr, chain)
    assert len(y_out) == len(y)



def _low_band_energy(y, sr, cutoff_hz=200.0):
    spectrum = np.abs(np.fft.rfft(np.asarray(y).mean(axis=0) if np.ndim(y) > 1 else y)) ** 2
    freqs = np.fft.rfftfreq(np.shape(y)[-1], 1 / sr)
    return spectrum[freqs < cutoff_hz].sum()


def test_reverb_send_low_cut_keeps_lows_out_of_the_tail():
    dsp = DSPEngine()
    sr = 44100
    t = np.arange(sr) / sr
    y = (0.3 * np.sin(2 * np.pi * 80 * t) * (t < 0.3)).astype(np.float32)  # низкий тон, затем тишина (хвост)
    params = {"room_size": 0.9, "damping": 0.3, "wet_level": 1.0, "dry_level": 0.0}
    no_cut = dsp.apply_chain(y, sr, ChainConfig([PluginConfig("reverb", {**params, "low_cut_hz": 0.0})]))
    cut = dsp.apply_chain(y, sr, ChainConfig([PluginConfig("reverb", {**params, "low_cut_hz": 0.5})]))
    tail = slice(int(0.4 * sr), None)
    assert _low_band_energy(cut[..., tail], sr) < 0.1 * _low_band_energy(no_cut[..., tail], sr)


def test_reverb_without_send_params_matches_serial_reverb():
    dsp = DSPEngine()
    sr = 44100
    t = np.arange(sr // 2) / sr
    y = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)  # фильтры на краях не трогают середину
    params = {"room_size": 0.5, "damping": 0.5, "wet_level": 0.5, "dry_level": 0.5}
    serial = dsp.apply_chain(y, sr, ChainConfig([PluginConfig("reverb", params)]))
    send = dsp.apply_chain(y, sr, ChainConfig([PluginConfig("reverb", {**params, "predelay_ms": 0.0,
                                                                        "low_cut_hz": 0.0, "high_cut_hz": 1.0})]))
    # Без predelay и с фильтрами на краях диапазона send-шина звучит как прежний последовательный Reverb
    assert np.allclose(serial, send, atol=0.02)


def test_decay_reverb_tail_follows_rt60():
    from app.audio.dsp_engine import decay_reverb
    sr = 16000
    impulse = np.zeros(sr * 8, dtype=np.float32)
    impulse[0] = 1.0
    for rt60 in (1.0, 4.0):
        tail = decay_reverb(impulse, sr, decay_s=rt60, damping=0.0)
        energy = np.convolve(tail ** 2, np.ones(sr // 20) / (sr // 20), "same")
        level_db = 10 * np.log10(energy[[int(0.1 * sr), int(0.1 * sr + rt60 / 2 * sr)]] + 1e-20)
        # За RT60/2 хвост падает на ~30 дБ
        assert 25.0 < level_db[0] - level_db[1] < 35.0


def test_send_delay_spreads_repeats_and_darkens_them():
    from app.audio.dsp_engine import send_delay
    sr = 16000
    impulse = np.zeros(sr * 2, dtype=np.float32)
    impulse[0] = 1.0
    wet = send_delay(impulse, sr, delay_s=0.3, feedback=0.5, spread=0.1, high_cut_hz=2000.0)
    # Первый повтор слева через 0.33 с, справа через 0.27 с
    assert abs(np.argmax(np.abs(wet[0])) / sr - 0.33) < 0.01
    assert abs(np.argmax(np.abs(wet[1])) / sr - 0.27) < 0.01

    def high_share(seg):
        spectrum = np.abs(np.fft.rfft(seg)) ** 2
        return spectrum[np.fft.rfftfreq(len(seg), 1 / sr) > 3000].sum() / spectrum.sum()

    first = wet[0, int(0.32 * sr):int(0.5 * sr)]
    second = wet[0, int(0.65 * sr):int(0.85 * sr)]
    # Срез стоит в петле обратной связи: каждый следующий повтор темнее
    assert high_share(second) < high_share(first)


def test_delay_send_keeps_dry_and_zero_mix_is_transparent():
    dsp = DSPEngine()
    sr = 44100
    t = np.arange(sr) / sr
    y = (0.2 * np.sin(2 * np.pi * 300 * t) * (t < 0.2)).astype(np.float32)
    params = {"delay_seconds": 0.5, "feedback": 0.5, "stereo_spread": 0.0, "low_cut_hz": 0.0, "high_cut_hz": 1.0}
    silent = dsp.apply_chain(y, sr, ChainConfig([PluginConfig("delay", {**params, "mix": 0.0})]),
                             final_limiter=False)
    assert np.allclose(silent, np.vstack([y, y]), atol=1e-5)
    echoed = dsp.apply_chain(y, sr, ChainConfig([PluginConfig("delay", {**params, "mix": 1.0})]),
                             final_limiter=False)
    # Сухая часть не ослаблена (send, а не кроссфейд), в паузе слышен повтор
    assert np.allclose(echoed[:, :int(0.15 * sr)], np.vstack([y, y])[:, :int(0.15 * sr)], atol=1e-5)
    assert np.abs(echoed[:, int(0.3 * sr):int(0.6 * sr)]).max() > 0.05


def test_stereo_enhancer_high_cut_narrows_top():
    from app.audio.vocal_enhancer import VocalEnhancer
    sr = 44100
    rng = np.random.default_rng(0)
    y = (0.1 * rng.standard_normal(sr)).astype(np.float32)
    enhancer = VocalEnhancer(sr=sr)

    def side_high(out):
        side = 0.5 * (out[0] - out[1])
        spectrum = np.abs(np.fft.rfft(side)) ** 2
        return spectrum[np.fft.rfftfreq(len(side), 1 / sr) > 6000].sum()

    full = enhancer.apply_stereo_enhancer(y, delay_ms=15.0, width=1.0)
    cut = enhancer.apply_stereo_enhancer(y, delay_ms=15.0, width=1.0, high_cut_hz=3000.0)
    assert side_high(cut) < 0.05 * side_high(full)
