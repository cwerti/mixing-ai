"""
Проверка, годится ли выделенный вокал референса для подбора. После Demucs в «вокале» бывает много чужого:
барабаны, бас или весь микс. По такому референсу и тон, и пространство подбираются мимо. На 62 парах Cambridge
у 10 таких референсов полная цепочка ухудшила log-mel к истине в среднем на 1.4 дБ, у остальных 52 улучшила на
0.5 дБ.

Признаки считаются по звучащим кадрам моно-сигнала:
- spectral flatness (медиана): шумовая каша вместо гармоник голоса;
- доля энергии ниже 150 Гц: в вокал протек бас или бочка.
"""
from __future__ import annotations

import numpy as np

SR = 22050
N_FFT = 2048
HOP = 512
ACTIVE_DB = -35.0  # кадр звучит, если его RMS не ниже пика на столько дБ
LOW_HZ = 150.0
MAX_FLATNESS = 0.01
MAX_LOW_SHARE = 0.3


def reference_quality(y: np.ndarray, sr: int) -> dict:
    """
    y — моно (n,) или многоканальный (каналы, n) сигнал. Возвращает признаки и флаг reliable с причинами
    (reasons — список строк для пользователя).
    """
    import librosa
    mono = librosa.to_mono(np.asarray(y, dtype=np.float32))
    if sr != SR:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=SR)
    spec = np.abs(librosa.stft(mono, n_fft=N_FFT, hop_length=HOP))
    rms = librosa.feature.rms(S=spec, frame_length=N_FFT)[0]
    active = rms > rms.max() * 10 ** (ACTIVE_DB / 20) if rms.max() > 0 else np.zeros_like(rms, dtype=bool)
    if not active.any():
        return {"flatness": 0.0, "low_share": 0.0, "reliable": False, "reasons": ["референс беззвучный"]}
    power = spec[:, active] ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    low_share = float(power[freqs < LOW_HZ].sum() / max(power.sum(), 1e-20))
    flatness = float(np.median(librosa.feature.spectral_flatness(S=spec)[0][active]))
    reasons = []
    if flatness > MAX_FLATNESS:
        reasons.append(f"в вокале много шума или чужих инструментов (flatness {flatness:.3f} > {MAX_FLATNESS})")
    if low_share > MAX_LOW_SHARE:
        reasons.append(f"в вокал протек бас или бочка ({low_share:.0%} энергии ниже {LOW_HZ:.0f} Гц)")
    return {"flatness": flatness, "low_share": low_share, "reliable": not reasons, "reasons": reasons}
