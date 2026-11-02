import numpy as np
from pedalboard import Chorus, Pedalboard, PitchShift

from app.audio.dsp_engine import decay_reverb, send_delay
from app.eval.effect_fingerprint import fingerprint, fingerprint_distance, fingerprint_vector

SR = 48000
FAST = {"sr": SR, "ir_seconds": 3.0, "sine_seconds": 2.5}


def _reverb(decay_s: float, damping: float):
    def render(x, sr):
        # Один канал ИХ на оба канала — иначе шум второго канала зависит от длины входа (сдвинутый импульс)
        return x + 0.5 * decay_reverb(x.mean(axis=0), sr, decay_s, damping)
    return render


def _delay(delay_s: float, spread: float = 0.0):
    return lambda x, sr: x + 0.5 * send_delay(x, sr, delay_s, 0.4, spread)


def test_reverb_decay_and_darkness():
    dark = fingerprint(_reverb(1.5, 1.0), **FAST)
    bright = fingerprint(_reverb(1.5, 0.0), **FAST)
    assert 1.2 < dark["rt60_500"] < 1.8
    # Темный хвост двухнаклонный (верх проваливается за ~0.1 с, утечка фильтра гаснет с серединой), поэтому
    # темнота видна по уровню верха в хвосте, а не по RT60 полосы
    assert dark["tail_db_8000"] < bright["tail_db_8000"] - 6.0
    assert dark["echo_strength"] < 0.4 and dark["ned_mid"] > 0.8


def test_delay_period_per_channel():
    f = fingerprint(_delay(0.32, spread=0.05), **FAST)
    assert abs(f["echo_period_l_ms"] - 336.0) <= 10.0
    assert abs(f["echo_period_r_ms"] - 304.0) <= 10.0
    assert f["echo_strength"] > 0.6
    assert abs(f["predelay_ms"] - 304.0) <= 10.0


def test_modulation_pitch_and_saturation_are_separated():
    chorus = fingerprint(lambda x, sr: Pedalboard([Chorus(rate_hz=1.0, depth=0.4, mix=0.5)])(x, sr), **FAST)
    shimmer = fingerprint(lambda x, sr: x + 0.5 * Pedalboard([PitchShift(12)])(x, sr), **FAST)
    clip = fingerprint(lambda x, sr: np.tanh(4.0 * x) / 4.0, **FAST)
    static = fingerprint(_reverb(1.5, 0.5), **FAST)

    assert chorus["mod_spread_db"] > static["mod_spread_db"] + 30.0
    assert chorus["ir_repeatability"] < 0.9 < static["ir_repeatability"]
    assert shimmer["pitch_up_db"] > -15.0 > static["pitch_up_db"]
    assert clip["saturation_db"] > 20.0 and clip["compression_db"] > 3.0
    assert static["saturation_db"] == 0.0 and abs(static["compression_db"]) < 0.1


def test_distance_orders_similar_effects_closer():
    a, b = fingerprint(_reverb(1.5, 0.9), **FAST), fingerprint(_reverb(1.7, 0.9), **FAST)
    echo = fingerprint(_delay(0.32), **FAST)
    assert fingerprint_distance(a, b) < fingerprint_distance(a, echo)
    assert np.isfinite(fingerprint_vector(echo)).all()


def test_identity_is_silent_tail():
    f = fingerprint(lambda x, sr: x, **FAST)
    assert f.get("silent_tail") and f["drr_db"] == 60.0
    assert np.isfinite(fingerprint_vector(f)).all()
