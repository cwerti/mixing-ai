import numpy as np

from app.audio.dsp_engine import decay_reverb, send_delay
from app.eval.blind_fingerprint import blind_fingerprint, find_word_ends

SR = 44100


def _phrases(seconds: float = 24.0, seed: int = 0) -> np.ndarray:
    """Слоги-ноты 0.3–0.6 с с паузами 0.5–1.0 с: в паузах слышен только хвост эффекта."""
    rng = np.random.default_rng(seed)
    y = np.zeros(int(seconds * SR), dtype=np.float32)
    pos = int(0.3 * SR)
    while pos < len(y) - SR:
        length = int(rng.uniform(0.3, 0.6) * SR)
        t = np.arange(length) / SR
        f0 = rng.uniform(150.0, 400.0)
        tone = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 20))
        noise = rng.standard_normal(length) * 0.05  # согласные: верх для детектора концов слов
        env = np.minimum(1.0, np.minimum(t / 0.01, (length / SR - t) / 0.01))
        y[pos:pos + length] = (0.2 * (tone + noise) * env).astype(np.float32)
        pos += length + int(rng.uniform(0.5, 1.0) * SR)
    return np.stack([y, y])


def test_word_ends_found_in_dry_phrases():
    import librosa
    from app.eval.blind_fingerprint import _band_db
    x = _phrases()
    spec = np.abs(librosa.stft(x.mean(axis=0), n_fft=2048, hop_length=441)) ** 2
    ends = find_word_ends(_band_db(spec, librosa.fft_frequencies(sr=SR, n_fft=2048), 1000.0, 8000.0))
    assert len(ends) >= 15


def test_reverb_decay_and_darkness_recovered_blind():
    x = _phrases()
    short = blind_fingerprint(x + 0.4 * decay_reverb(x.mean(axis=0), SR, 1.0, 0.5)[None], SR)
    long_dark = blind_fingerprint(x + 0.4 * decay_reverb(x.mean(axis=0), SR, 3.0, 1.0)[None], SR)
    long_bright = blind_fingerprint(x + 0.4 * decay_reverb(x.mean(axis=0), SR, 3.0, 0.0)[None], SR)
    assert short["confidence"]["n_ends"] >= 10
    assert 0.6 < short["rt60_s"] < 1.6
    assert 2.0 < long_dark["rt60_s"] < 4.5
    # Истина по импульсу: -10.3 дБ у темного против -1.9 у яркого на 8 кГц
    assert long_dark["tail_db_8000"] < long_bright["tail_db_8000"] - 5.0


def test_delay_period_and_width_recovered_blind():
    x = _phrases(seed=1)
    wet = x + 0.5 * send_delay(x, SR, 0.3, 0.4, spread=0.05)
    f = blind_fingerprint(wet, SR)
    assert abs(f["echo_period_l_ms"] - 315.0) <= 15.0
    assert abs(f["echo_period_r_ms"] - 285.0) <= 15.0
    mono = blind_fingerprint(x + 0.4 * decay_reverb(x.mean(axis=0), SR, 1.0, 0.5)[None], SR)
    assert f["side_mid_db"] > mono["side_mid_db"] + 20.0
