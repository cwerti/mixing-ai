"""
Отпечаток эффекта: что эффект делает со звуком, независимо от того, какой плагин и с какими ручками это сделал.

Эффект — черный ящик `render(x, sr) -> y` (стерео (2, n) на входе). Его прогоняют через тестовые сигналы и меряют:

- импульс → затухание по полосам (RT60, уровень хвоста по полосам — темный/яркий хвост), предзадержка, время
  до пика энергии (swell — медленно нарастающий хвост), плотность эха (дискретные повторы против диффузного
  хвоста, Abel & Huang 2006), периодичность огибающей (темповый дилей и время L/R), стерео (Side/Mid, IACC
  в начале и в хвосте), отношение прямого звука к хвосту;
- тот же импульс со сдвигом → повторяемость отклика (модулированный ревер / хорус меняют отклик во времени);
- синус на трех уровнях → модуляция (боковые полосы вокруг тона), сдвиг высоты (энергия на ½f, 1.5f, 2f —
  shimmer), нелинейность (гармоники на громком и тихом сигнале), сжатие (разница усиления громкого и тихого).

Зачем: модель и подбор должны говорить на одном языке с любыми плагинами пользователя (Valhalla, стоковые FL,
наш DSP). Тот же отпечаток снимается с нашей цепочки, поэтому язык проверяется на нас самих.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.signal import butter, sosfilt

Effect = Callable[[np.ndarray, int], np.ndarray]

BANDS_HZ = (125, 250, 500, 1000, 2000, 4000, 8000)
SINE_HZ = 500.0
# Сдвиг второго импульса для проверки повторяемости: не кратен типичным периодам LFO и темпа
SHIFT_S = 0.317
# Окно прямого звука: всё, что раньше, считается сухим сигналом/латентностью, позже — эффектом
DIRECT_S = 0.002
_ERFC_1_SQRT2 = 0.3173  # доля выборок гауссова шума за пределами ±1σ — нормировка плотности эха


def _as_2ch(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        return np.stack([y, y])
    if y.shape[0] != 2 and y.shape[-1] == 2:
        y = y.T
    return y[:2] if y.shape[0] >= 2 else np.vstack([y, y])


def _render(effect: Effect, x: np.ndarray, sr: int) -> np.ndarray:
    out = _as_2ch(effect(x.astype(np.float32), sr))
    n = x.shape[-1]
    if out.shape[1] < n:
        out = np.pad(out, ((0, 0), (0, n - out.shape[1])))
    return out[:, :n]


def _db(x: float, floor: float = 1e-20) -> float:
    return float(10.0 * np.log10(max(float(x), floor)))


def _frame_energy(y: np.ndarray, sr: int, frame_s: float) -> np.ndarray:
    hop = max(1, int(frame_s * sr))
    n = len(y) // hop
    return (y[:n * hop] ** 2).reshape(n, hop).sum(axis=1)


def _band(y: np.ndarray, sr: int, center: float) -> np.ndarray:
    lo, hi = center / np.sqrt(2.0), min(center * np.sqrt(2.0), 0.45 * sr)
    return sosfilt(butter(4, [lo, hi], "bandpass", fs=sr, output="sos"), y)


def decay_time_s(y: np.ndarray, sr: int, frame_s: float = 0.02) -> float:
    """
    RT60 по наклону огибающей. Если хвост успел затухнуть на 45 дБ в окне — по кривой Шрёдера (T20, при нехватке
    динамики T10), иначе (хвост длиннее окна, Шрёдер на обрезке занижает) — линейной регрессией огибающей в дБ
    от ее пика до конца окна.
    """
    e = _frame_energy(y, sr, frame_s)
    if e.size < 3 or e.max() <= 0:
        return 0.0
    peak = int(np.argmax(e))
    e_db = 10.0 * np.log10(e / e.max() + 1e-20)
    tail = e_db[-max(1, int(0.5 / frame_s)):]
    if tail.max() < -45.0:
        edc = np.cumsum(e[::-1])[::-1]
        edc_db = 10.0 * np.log10(edc / edc[0] + 1e-20)
        t = np.arange(len(edc)) * frame_s
        for lo, hi in ((-5.0, -25.0), (-5.0, -15.0)):
            sel = (edc_db <= lo) & (edc_db >= hi)
            if sel.sum() >= 3:
                slope = np.polyfit(t[sel], edc_db[sel], 1)[0]
                return float(np.clip(-60.0 / slope, 0.0, 120.0)) if slope < 0 else 120.0
    sel = np.arange(len(e_db)) >= peak
    sel &= e_db > -80.0
    if sel.sum() < 3:
        return 0.0
    t = np.arange(len(e_db)) * frame_s
    slope = np.polyfit(t[sel], e_db[sel], 1)[0]
    return float(np.clip(-60.0 / slope, 0.0, 120.0)) if slope < 0 else 120.0


def echo_density(y: np.ndarray, sr: int, window_s: float = 0.02) -> np.ndarray:
    """Нормированная плотность эха по окнам: ~1 у диффузного (гауссова) хвоста, ≪1 у отдельных повторов."""
    w = max(8, int(window_s * sr))
    n = len(y) // w
    if n == 0:
        return np.zeros(0)
    frames = y[:n * w].reshape(n, w)
    std = frames.std(axis=1, keepdims=True)
    outside = (np.abs(frames - frames.mean(axis=1, keepdims=True)) > std).mean(axis=1)
    return np.where(std[:, 0] > 0, outside / _ERFC_1_SQRT2, 0.0)


def periodicity(y: np.ndarray, sr: int, min_s: float = 0.04, max_s: float = 1.5,
                frame_s: float = 0.005) -> tuple[float, float]:
    """
    Период и сила повторов: автокорреляция огибающей в дБ после вычитания линейного затухания (экспонента
    без повторов дает остаток-шум, темповый дилей — пики на кратных периоду лагах).
    """
    e = _frame_energy(y, sr, frame_s)
    if e.size < 4 or e.max() <= 0:
        return 0.0, 0.0
    e_db = 10.0 * np.log10(e / e.max() + 1e-8)
    alive = np.nonzero(e_db > -60.0)[0]
    if alive.size < 4:
        return 0.0, 0.0
    seg = e_db[alive[0]:alive[-1] + 1]
    t = np.arange(len(seg))
    # Квадратичный тренд: у хвоста из двух полос с разным затуханием огибающая в дБ выпуклая, а не прямая
    resid = seg - np.polyval(np.polyfit(t, seg, 2), t)
    resid -= resid.mean()
    if not np.any(resid):
        return 0.0, 0.0
    ac = np.correlate(resid, resid, "full")[len(resid) - 1:]
    ac /= ac[0]
    lo, hi = int(min_s / frame_s), min(int(max_s / frame_s), len(ac) - 2)
    if hi <= lo:
        return 0.0, 0.0
    # Только локальные максимумы: у гладкого остатка автокорреляция монотонно спадает и максимум на краю — не период
    peaks = [i for i in range(lo, hi) if ac[i] >= ac[i - 1] and ac[i] > ac[i + 1]]
    if not peaks:
        return 0.0, 0.0
    lag = max(peaks, key=lambda i: ac[i])
    # Сила — высота пика над провалом перед ним: у пачки повторов провал отрицательный, у шума пики мелкие
    strength = float(np.clip(ac[lag] - ac[1:lag].min(), 0.0, 1.0))
    return float(lag * frame_s * 1000.0), strength


def _iacc(left: np.ndarray, right: np.ndarray, sr: int, max_lag_s: float = 0.001) -> float:
    denom = np.sqrt(np.sum(left ** 2) * np.sum(right ** 2))
    if denom <= 0:
        return 1.0
    lag = int(max_lag_s * sr)
    full = np.correlate(left, right, "full")
    mid = len(right) - 1
    return float(np.max(np.abs(full[mid - lag:mid + lag + 1])) / denom)


def _band_energy(spec: np.ndarray, freqs: np.ndarray, center: float, half_width: float,
                 exclude: float = 0.0) -> float:
    d = np.abs(freqs - center)
    return float(spec[(d <= half_width) & (d >= exclude)].sum())


def impulse_features(h: np.ndarray, sr: int) -> dict:
    """Признаки отклика на импульс h (2, n), импульс в отсчете 0."""
    mid, side = h.mean(axis=0), 0.5 * (h[0] - h[1])
    d = max(1, int(DIRECT_S * sr))
    tail_mid, tail = mid[d:], h[:, d:]
    direct_e, tail_e = float(np.sum(mid[:d] ** 2)), float(np.sum(tail_mid ** 2))
    f: dict = {"drr_db": float(np.clip(_db(direct_e) - _db(tail_e), -60.0, 60.0))}
    if tail_e <= 1e-14:
        return f | {"silent_tail": True}

    env = _frame_energy(tail_mid, sr, 0.005)
    env_db = 10.0 * np.log10(env / env.max() + 1e-20)
    first = int(np.argmax(env_db > -20.0))
    f["predelay_ms"] = (first * 0.005 + DIRECT_S) * 1000.0
    smooth = _frame_energy(tail_mid, sr, 0.05)
    f["peak_ms"] = (float(np.argmax(smooth)) * 0.05 + DIRECT_S) * 1000.0
    f["rt60_s"] = decay_time_s(tail_mid, sr)

    band_e = []
    for c in BANDS_HZ:
        b = _band(tail_mid, sr, c)
        f[f"rt60_{c}"] = decay_time_s(b, sr)
        band_e.append(np.sum(b ** 2))
    total = max(sum(band_e), 1e-20)
    for c, e in zip(BANDS_HZ, band_e):
        f[f"tail_db_{c}"] = _db(e / total)

    # Плотность эха и стерео — только пока хвост выше -50 дБ, дальше считается шум хвоста окна
    alive_frames = np.nonzero(env_db > -50.0)[0]
    alive_end = (alive_frames[-1] + 1) * int(0.005 * sr) if alive_frames.size else len(tail_mid)
    start = first * int(0.005 * sr)
    ned = echo_density(tail_mid[start:alive_end], sr)
    w = 0.02
    for name, (a, b) in {"ned_early": (0.0, 0.05), "ned_mid": (0.05, 0.2), "ned_late": (0.2, 1.0)}.items():
        seg = ned[int(a / w):int(b / w)]
        f[name] = float(seg.mean()) if seg.size else float("nan")
    smooth_ned = np.convolve(ned, np.ones(3) / 3.0, "same") if ned.size >= 3 else ned
    mixed = np.nonzero(smooth_ned >= 0.9)[0]
    f["mixing_ms"] = float(mixed[0] * w * 1000.0) if mixed.size else float(len(ned) * w * 1000.0)

    for ch, name in ((0, "l"), (1, "r")):
        period, strength = periodicity(tail[ch][:alive_end], sr)
        f[f"echo_period_{name}_ms"], f[f"echo_strength_{name}"] = period, strength
    f["echo_strength"] = 0.5 * (f["echo_strength_l"] + f["echo_strength_r"])

    f["side_mid_db"] = _db(np.sum(side[d:] ** 2)) - _db(tail_e)
    early = slice(start, start + int(0.08 * sr))
    late = slice(start + int(0.08 * sr), min(alive_end, start + int(1.0 * sr)))
    f["iacc_early"] = _iacc(tail[0][early], tail[1][early], sr)
    f["iacc_late"] = _iacc(tail[0][late], tail[1][late], sr) if late.stop > late.start + d else f["iacc_early"]
    return f


def sine_features(outputs: dict[float, np.ndarray], sr: int, levels_db: dict[float, float]) -> dict:
    """
    Признаки откликов на синус SINE_HZ: outputs — {уровень дБFS: выход (2, n)}, анализируется последняя половина
    (хвост ревера успел накопиться). levels_db — входные RMS тех же синусов.
    """
    f: dict = {}
    low, high = min(outputs), max(outputs)
    for level, y in outputs.items():
        mid = y.mean(axis=0)
        seg = mid[len(mid) // 2:]
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
        fund = _band_energy(spec, freqs, SINE_HZ, 1.5)
        main = max(_band_energy(spec, freqs, SINE_HZ, 60.0), 1e-20)
        rms_out = np.sqrt(np.mean(seg ** 2))
        f[f"gain_db@{level:g}"] = 20.0 * np.log10(max(rms_out, 1e-12)) - levels_db[level]
        if level == low:
            # Тихий синус: нелинейность почти не работает, энергия на ½f / 1.5f / 2f — сдвиг высоты
            f["mod_spread_db"] = _db(_band_energy(spec, freqs, SINE_HZ, 60.0, exclude=3.0)) - _db(max(fund, 1e-20))
            for name, mult in (("pitch_down_db", 0.5), ("pitch_fifth_db", 1.5), ("pitch_up_db", 2.0)):
                f[name] = _db(_band_energy(spec, freqs, SINE_HZ * mult, 30.0)) - _db(main)
        harm = sum(_band_energy(spec, freqs, SINE_HZ * k, 30.0) for k in (2, 3, 4, 5))
        f[f"harm_db@{level:g}"] = _db(harm) - _db(main)
    f["compression_db"] = f[f"gain_db@{low:g}"] - f[f"gain_db@{high:g}"]
    # Рост гармоник с уровнем — сатурация; постоянные гармоники при росте уровня — сдвиг высоты, а не нелинейность
    f["saturation_db"] = max(0.0, f[f"harm_db@{high:g}"] - max(f[f"harm_db@{low:g}"], -100.0))
    return f


def fingerprint(effect: Effect, sr: int = 48000, ir_seconds: float = 6.0, sine_seconds: float = 4.0,
                sine_levels_dbfs: tuple[float, ...] = (-36.0, -18.0, -6.0), impulse_amp: float = 0.5) -> dict:
    """Полный отпечаток эффекта. Детерминирован, если детерминирован сам эффект."""
    n = int(ir_seconds * sr)
    x = np.zeros((2, n), dtype=np.float32)
    x[:, 0] = impulse_amp
    h = _render(effect, x, sr) / impulse_amp
    f = impulse_features(h, sr)

    shift = int(SHIFT_S * sr)
    x2 = np.zeros((2, n + shift), dtype=np.float32)
    x2[:, shift] = impulse_amp
    h2 = _render(effect, x2, sr)[:, shift:] / impulse_amp
    win = slice(0, min(n, int(1.0 * sr)))
    a, b = h[:, win].ravel(), h2[:, win].ravel()
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    f["ir_repeatability"] = float(np.dot(a, b) / denom) if denom > 0 else 1.0

    m = int(sine_seconds * sr)
    t = np.arange(m) / sr
    # Плавный вход за 1 с: резкое начало синуса — широкополосный щелчок, его хвост в длинном ревере
    # выглядел бы как модуляция (боковые полосы)
    fade = 0.5 - 0.5 * np.cos(np.pi * np.minimum(1.0, t / 1.0))
    outputs, levels = {}, {}
    for level in sine_levels_dbfs:
        amp = 10.0 ** (level / 20.0) * np.sqrt(2.0)
        s = (amp * np.sin(2 * np.pi * SINE_HZ * t) * fade).astype(np.float32)
        outputs[level] = _render(effect, np.stack([s, s]), sr)
        levels[level] = level
    f.update(sine_features(outputs, sr, levels))
    return f


# Ключи вектора отпечатка и их масштабы: признак / масштаб ≈ «одна заметная на слух разница»
FINGERPRINT_SCALES: dict[str, float] = {
    "drr_db": 6.0, "predelay_ms": 20.0, "peak_ms": 100.0,
    **{f"rt60_{c}": 0.3 for c in BANDS_HZ},  # в логарифме: 0.3 ≈ ×1.35
    **{f"tail_db_{c}": 6.0 for c in BANDS_HZ},
    "ned_early": 0.25, "ned_mid": 0.25, "ned_late": 0.25,
    "echo_strength": 0.2, "side_mid_db": 4.0, "iacc_early": 0.2, "iacc_late": 0.2,
    "ir_repeatability": 0.15, "mod_spread_db": 10.0,
    "pitch_down_db": 10.0, "pitch_fifth_db": 10.0, "pitch_up_db": 10.0,
    "compression_db": 2.0, "saturation_db": 10.0,
}
_LOG_KEYS = {f"rt60_{c}" for c in BANDS_HZ}
_DB_FLOOR = -80.0


def fingerprint_vector(f: dict) -> np.ndarray:
    """Вектор в масштабированных единицах (для расстояний и обучения); отсутствующие признаки — 0."""
    out = []
    for key, scale in FINGERPRINT_SCALES.items():
        v = f.get(key)
        if v is None or not np.isfinite(v):
            v = 0.0
        elif key in _LOG_KEYS:
            v = np.log(max(v, 0.05))
        elif key.endswith("_db"):
            v = float(np.clip(v, _DB_FLOOR, -_DB_FLOOR))
        out.append(float(v) / scale)
    return np.asarray(out)


def fingerprint_distance(a: dict, b: dict) -> float:
    return float(np.sqrt(np.mean((fingerprint_vector(a) - fingerprint_vector(b)) ** 2)))
