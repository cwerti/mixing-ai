import numpy as np

from app.eval.reference_quality import reference_quality

SR = 44100


def _voice(seconds: float = 4.0, f0: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    y = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
    envelope = (np.sin(2 * np.pi * 0.5 * t) > -0.3).astype(np.float32)  # фразы и паузы
    return (0.2 * y * envelope).astype(np.float32)


def test_clean_harmonic_vocal_is_reliable():
    q = reference_quality(_voice(), SR)
    assert q["reliable"], q["reasons"]


def test_noise_wash_is_flagged():
    rng = np.random.default_rng(0)
    y = _voice() + 0.3 * rng.standard_normal(int(4.0 * SR)).astype(np.float32)
    q = reference_quality(y, SR)
    assert not q["reliable"]
    assert q["flatness"] > 0.01


def test_bass_leak_is_flagged():
    t = np.arange(int(4.0 * SR)) / SR
    y = _voice() + 1.0 * np.sin(2 * np.pi * 60 * t).astype(np.float32)
    q = reference_quality(np.vstack([y, y]), SR)
    assert not q["reliable"]
    assert q["low_share"] > 0.3


def test_silence_is_not_reliable():
    assert not reference_quality(np.zeros(SR, dtype=np.float32), SR)["reliable"]
