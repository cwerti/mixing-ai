"""Синтетический вокал для тестов метрики стиля и оптимизатора цепочки."""
import numpy as np


def synthetic_vocal(sr: int, seconds: float = 8.0, seed: int = 0, detune_cents: float = 0.0) -> np.ndarray:
    """Гармонический «голос»: слоги по 150-400 мс с паузами, разной громкостью и нотами гаммы до мажор."""
    rng = np.random.default_rng(seed)
    y = np.zeros(int(seconds * sr), dtype=np.float32)
    pos = int(0.2 * sr)
    scale = [0, 2, 4, 5, 7, 9, 11]
    while pos < len(y) - sr // 2:
        length = int(rng.uniform(0.15, 0.4) * sr)
        midi = 57 + rng.choice(scale) + rng.normal(0.0, detune_cents / 100.0)
        f0 = 440.0 * 2.0 ** ((midi - 69) / 12.0)
        t = np.arange(length) / sr
        tone = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 12))
        envelope = np.minimum(1.0, np.minimum(t / 0.01, (length / sr - t) / 0.01))
        gain = 10.0 ** (rng.uniform(-18.0, 0.0) / 20.0)
        y[pos:pos + length] += (0.15 * gain * tone * envelope).astype(np.float32)
        pos += length + int(rng.uniform(0.03, 0.25) * sr)
    return y
