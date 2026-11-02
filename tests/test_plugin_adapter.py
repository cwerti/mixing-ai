import numpy as np

from app.audio.dsp_engine import decay_reverb, send_delay
from app.eval.effect_fingerprint import fingerprint, fingerprint_distance
from app.eval.plugin_adapter import (SM_KNOBS, SM_MODES, fast_fingerprint, knobs_to_unit, probe_signal,
                                     surrogate_input, unit_to_params)

SR = 48000


def _reverb(x, sr):
    return x + 0.5 * decay_reverb(x.mean(axis=0), sr, 1.2, 0.6)


def _delay(x, sr):
    return x + 0.5 * send_delay(x, sr, 0.3, 0.4, 0.05)


def test_knob_roundtrip_and_bounds():
    u = np.linspace(0.0, 1.0, len(SM_KNOBS))
    params = unit_to_params("Hydra", u)
    assert params["mode"] == "Hydra" and params["delaysync"] == "Msec"
    assert np.allclose(knobs_to_unit(params), u, atol=1e-9)
    # Значение вне диапазона прижимается к границе
    assert knobs_to_unit({"mix": 500.0})[0] == 1.0
    x = surrogate_input("Sirius", u)
    assert x.shape == (len(SM_MODES) + len(SM_KNOBS),) and x[SM_MODES.index("Sirius")] == 1.0


def test_probe_layout():
    x, pos = probe_signal(SR)
    assert x[0, pos["ir"]] > 0 and x[0, pos["ir2"]] > 0
    assert pos["ir2"] - pos["n_ir"] == int(0.317 * SR)
    sine = x[0, pos["sine-6"] + pos["n_sine"] // 2:pos["sine-6"] + pos["n_sine"]]
    assert abs(20 * np.log10(np.sqrt(np.mean(sine ** 2))) + 6.0) < 0.2


def test_fast_fingerprint_matches_full_for_static_effects():
    for effect in (_reverb, _delay):
        fast, rmap = fast_fingerprint(effect, SR)
        full = fingerprint(effect, sr=SR, ir_seconds=8.0)
        assert fingerprint_distance(fast, full) < 0.15
        assert rmap.shape[0] == 2
