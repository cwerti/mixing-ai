"""
Слепая оценка отпечатка эффекта по одному мокрому вокалу (без его сухого).

Референс в реальной работе — сведенный вокал чужой песни: сухого нет, тестовые сигналы через его цепочку не
прогнать. Но в паузах после концов слов звучит то, что эффект делает с только что спетым: хвост ревера, повторы
дилея, их ширина. Отсюда признаки с теми же ключами, что у app/eval/effect_fingerprint.fingerprint():

- концы слов: резкий спад верха (1–8 кГц) с последующей паузой без нового слога;
- по полосам BANDS_HZ: наклон спада в паузе → rt60_<c>, уровень хвоста в начале паузы относительно ноты →
  отношение wet/dry по полосе → tail_db_<c> (спектр хвоста, как у импульса) и drr_db (широкополосно);
- повторы: самоподобие спектра кадров паузы с кадрами на лаг раньше (дилей повторяет рисунок высоты фразы,
  диффузный хвост — только размазывает) → echo_period_{l,r}_ms, echo_strength (своя шкала, сравнивать рангом);
- стерео в паузах: side_mid_db, iacc_late;
- shimmer: энергия на ½f0 и 1.5f0 в хвосте относительно f0 ноты перед паузой, прирост 2f0 хвоста против ноты;
- модуляция: гармоничность устойчивых нот (хорус/детюн размывают гармоники) — косвенный признак.

Предзадержка, нарастание хвоста, сатурация и компрессия здесь не оцениваются (нужен сухой сигнал или другие
признаки). Кроме значений возвращается уверенность: число концов слов, суммарная длина пауз, разброс.
"""
from __future__ import annotations

import numpy as np
import librosa

from app.eval.effect_fingerprint import BANDS_HZ

HOP_S = 0.01
N_FFT = 2048
DETECT_HZ = (1000.0, 8000.0)
# Резкий спад: прямой звук гаснет за десятки мс, хвост эффекта — медленно. 4 дБ хватает и для очень мокрых миксов,
# где хвост в начале паузы лишь на 3–6 дБ тише ноты
DROP_DB, DROP_FRAMES = 4.0, 4
MIN_PAUSE_S, MAX_PAUSE_S = 0.3, 2.0
# Начало хвоста — через 40 мс после спада: окно STFT (46 мс) и естественное затухание голоса
SKIP_FRAMES = 4
TRIM_FRAMES = 5
ECHO_LAGS_S = (0.04, 1.5)


def _to_2ch(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        return np.stack([y, y])
    if y.shape[0] != 2 and y.shape[-1] == 2:
        y = y.T
    return y[:2] if y.shape[0] >= 2 else np.vstack([y, y])


def _band_db(power: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return 10.0 * np.log10(power[(freqs >= lo) & (freqs < hi)].sum(axis=0) + 1e-12)


def find_word_ends(det_db: np.ndarray, hop_s: float = HOP_S,
                   onset_db: np.ndarray | None = None) -> list[tuple[int, int, float]]:
    """
    Концы слов: (кадр ноты перед спадом, кадр начала паузы, длина паузы в кадрах). Пауза кончается, когда
    детектор за 30 мс вырастает до уровня ноты −6 дБ (новый слог) или через MAX_PAUSE_S. onset_db — второй
    детектор начала слога (широкая полоса): у темного длинного хвоста рост верха на новом слоге маскируется.
    """
    top = np.percentile(det_db, 95)
    floor = np.percentile(det_db, 5)
    min_pause, max_pause = int(MIN_PAUSE_S / hop_s), int(MAX_PAUSE_S / hop_s)
    ends, i, n = [], 3, len(det_db)
    while i < n - min_pause:
        note = det_db[i - 3:i].max()
        if note > top - 25.0 and note - floor > 15.0 and det_db[i + DROP_FRAMES] < note - DROP_DB:
            start = i + SKIP_FRAMES
            j = start
            note_on = onset_db[max(0, i - 3):i].max() if onset_db is not None else 0.0
            while j < min(n - 3, start + max_pause):
                if onset_db is not None and onset_db[j + 3] > onset_db[j] + 3.0 and onset_db[j + 3] > note_on - 10.0:
                    break
                # Новый слог — подъем почти до уровня ноты. Мелкие подъемы (повторы дилея, шум) пауза терпит:
                # иначе дилей обрывал бы каждую паузу на первом повторе
                if det_db[j + 3] > note - 6.0 and det_db[j + 3] > det_db[j] + 3.0:
                    break
                j += 1
            # Последние кадры паузы отбрасываются: окно STFT (±23 мс) захватывает начало следующего слога
            j = max(start, j - TRIM_FRAMES)
            if j - start >= min_pause:
                ends.append((i - 1, start, j - start))
                i = j
                continue
        i += 1
    return ends


def _fit_slope(t: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    a, b = np.polyfit(t, v, 1)
    return float(a), float(b)


def _echo_similarity(logspec: np.ndarray, ends: list, lags: np.ndarray) -> np.ndarray:
    """Среднее по кадрам пауз сходство спектра кадра с кадром на лаг раньше (корреляция по частотам)."""
    z = logspec - logspec.mean(axis=0, keepdims=True)
    z /= np.linalg.norm(z, axis=0, keepdims=True) + 1e-9
    frames = np.concatenate([np.arange(s, s + p) for _, s, p in ends]) if ends else np.zeros(0, int)
    sims = np.full(len(lags), np.nan)
    for k, lag in enumerate(lags):
        f = frames[frames - lag >= 0]
        if f.size:
            sims[k] = float(np.mean(np.sum(z[:, f] * z[:, f - lag], axis=0)))
    return sims


def _echo_peak(sims: np.ndarray, lags: np.ndarray, hop_s: float) -> tuple[float, float]:
    """Период — самый выраженный локальный максимум сходства после вычитания гладкого спада (как periodicity)."""
    ok = np.isfinite(sims)
    if ok.sum() < 5:
        return 0.0, 0.0
    x = lags[ok].astype(float)
    s = sims[ok] - np.polyval(np.polyfit(x, sims[ok], 2), x)
    peaks = [i for i in range(1, len(s) - 1) if s[i] >= s[i - 1] and s[i] > s[i + 1]]
    if not peaks:
        return 0.0, 0.0
    prominence = {i: s[i] - s[:i].min() for i in peaks}
    strongest = max(prominence.values())
    # Пики на кратных периоду лагах почти так же сильны — берется наименьший лаг среди сильных
    best = min(i for i in peaks if prominence[i] >= 0.6 * strongest)
    return float(x[best] * hop_s * 1000.0), float(max(prominence[best], 0.0))


def _iacc(left: np.ndarray, right: np.ndarray, sr: int) -> float:
    lag = int(0.001 * sr)
    denom = np.sqrt(np.sum(left ** 2) * np.sum(right ** 2))
    if denom <= 0:
        return 1.0
    full = np.correlate(left, right, "full")
    mid = len(right) - 1
    return float(np.max(np.abs(full[mid - lag:mid + lag + 1])) / denom)


def _harmonic_db(spec: np.ndarray, freqs: np.ndarray, f: float, rel: float = 0.03) -> float:
    sel = np.abs(freqs - f) <= max(rel * f, 3.0 * (freqs[1] - freqs[0]))
    return float(10.0 * np.log10(spec[sel].sum() + 1e-20))


def blind_fingerprint(wet: np.ndarray, sr: int) -> dict:
    """Оценка отпечатка по мокрому вокалу. Ключи — подмножество fingerprint() плюс 'confidence'."""
    y = _to_2ch(wet)
    mid, side = y.mean(axis=0), 0.5 * (y[0] - y[1])
    hop = int(HOP_S * sr)
    spec = np.abs(librosa.stft(mid, n_fft=N_FFT, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    det = _band_db(spec, freqs, *DETECT_HZ)
    ends = find_word_ends(det, onset_db=_band_db(spec, freqs, 100.0, 8000.0))
    f: dict = {"confidence": {"n_ends": len(ends), "pause_s": float(sum(p for *_, p in ends) * HOP_S)}}
    if len(ends) < 3:
        return f

    # --- Затухание и спектр хвоста по полосам ---
    band_rows = {c: [] for c in BANDS_HZ}
    for c in BANDS_HZ:
        e = _band_db(spec, freqs, c / np.sqrt(2.0), min(c * np.sqrt(2.0), 0.45 * sr))
        noise = np.percentile(e, 1)
        for note_i, start, pause in ends:
            note = e[max(0, note_i - 5):note_i + 1].max()
            seg = e[start:start + pause]
            # Обрезка по первому подходу сглаженной огибающей к шуму, а не маска «выше шума»: маска в конце
            # паузы пропускает только всплески вверх и делает наклон длинных хвостов пологим
            smooth = 10.0 * np.log10(np.convolve(10.0 ** (seg / 10.0), np.ones(7) / 7.0, "same") + 1e-12)
            low = np.nonzero(smooth < noise + 6.0)[0]
            cut = int(low[0]) if low.size else len(seg)
            if cut < 8:
                continue
            t = np.arange(cut) * HOP_S
            slope, icpt = _fit_slope(t, seg[:cut])
            # Уровень хвоста в момент спада (экстраполяция назад на SKIP_FRAMES) относительно ноты
            level = icpt - slope * SKIP_FRAMES * HOP_S - note
            band_rows[c].append((slope, min(level, -0.5), pause))
    gains, spread = {}, {}
    for c, rows in band_rows.items():
        if len(rows) < 3:
            continue
        slopes = np.array([r[0] for r in rows])
        levels = np.array([r[1] for r in rows])
        w = np.array([r[2] for r in rows], dtype=float)
        slope = float(np.median(slopes))
        f[f"rt60_{c}"] = float(np.clip(-60.0 / slope, 0.05, 120.0)) if slope < 0 else 120.0
        spread[f"rt60_{c}"] = float(np.subtract(*np.percentile(-60.0 / np.minimum(slopes, -0.5), [75, 25])))
        r = 10.0 ** (np.average(levels, weights=w) / 10.0)
        gains[c] = r / max(1.0 - r, 1e-3)  # wet/dry по мощности: нота = сухой + мокрый, начало паузы — мокрый
    if gains:
        weighted = {c: g * c for c, g in gains.items()}  # энергия импульса в октавной полосе ∝ ее ширине
        total = sum(weighted.values())
        for c, v in weighted.items():
            f[f"tail_db_{c}"] = float(10.0 * np.log10(v / total))
        mids = [gains[c] for c in (500, 1000, 2000) if c in gains]
        if mids:
            f["drr_db"] = float(np.clip(-10.0 * np.log10(np.mean(mids)), -60.0, 60.0))
        rts = [f[f"rt60_{c}"] for c in (500, 1000, 2000) if f"rt60_{c}" in f]
        if rts:
            f["rt60_s"] = float(np.median(rts))
    f["confidence"]["rt60_iqr_s"] = spread

    # --- Повторы по каналам ---
    lags = np.arange(int(ECHO_LAGS_S[0] / HOP_S), int(ECHO_LAGS_S[1] / HOP_S))
    sel = (freqs >= 300.0) & (freqs < 4000.0)
    for ch, name in ((0, "l"), (1, "r")):
        s_ch = np.log(np.abs(librosa.stft(y[ch], n_fft=N_FFT, hop_length=hop))[sel] ** 2 + 1e-10)
        period, strength = _echo_peak(_echo_similarity(s_ch, ends, lags), lags, HOP_S)
        f[f"echo_period_{name}_ms"], f[f"echo_strength_{name}"] = period, strength
    f["echo_strength"] = 0.5 * (f["echo_strength_l"] + f["echo_strength_r"])

    # --- Стерео в паузах ---
    segs = [slice(start * hop, (start + pause) * hop) for _, start, pause in ends]
    mid_e = sum(float(np.sum(mid[s] ** 2)) for s in segs)
    side_e = sum(float(np.sum(side[s] ** 2)) for s in segs)
    f["side_mid_db"] = float(10.0 * np.log10(max(side_e, 1e-20) / max(mid_e, 1e-20)))
    late = [slice(s.start + int(0.08 * sr), s.stop) for s in segs if s.stop - s.start > int(0.15 * sr)]
    if late:
        vals = [_iacc(y[0, s], y[1, s], sr) for s in late]
        lens = [s.stop - s.start for s in late]
        f["iacc_late"] = float(np.average(vals, weights=lens))

    # --- Shimmer: высоты в хвосте, которых не было в ноте ---
    long_fft = 8192
    up, down, fifth = [], [], []
    for note_i, start, pause in ends:
        a, b = max(0, (note_i - 20) * hop), (note_i + 1) * hop
        if b - a < long_fft // 2:
            continue
        f0 = librosa.yin(mid[a:b], fmin=70.0, fmax=800.0, sr=sr, frame_length=2048)
        f0 = float(np.median(f0))
        tail = mid[start * hop:start * hop + min(pause * hop, int(0.6 * sr))]
        if len(tail) < long_fft // 2 or not np.isfinite(f0):
            continue
        sp_note = np.abs(np.fft.rfft(mid[a:b] * np.hanning(b - a), long_fft)) ** 2
        sp_tail = np.abs(np.fft.rfft(tail * np.hanning(len(tail)), long_fft)) ** 2
        fr = np.fft.rfftfreq(long_fft, 1.0 / sr)
        base_t = _harmonic_db(sp_tail, fr, f0)
        down.append(_harmonic_db(sp_tail, fr, 0.5 * f0) - base_t)
        fifth.append(_harmonic_db(sp_tail, fr, 1.5 * f0) - base_t)
        up.append((_harmonic_db(sp_tail, fr, 2.0 * f0) - base_t)
                  - (_harmonic_db(sp_note, fr, 2.0 * f0) - _harmonic_db(sp_note, fr, f0)))
    if up:
        f["pitch_up_db"], f["pitch_down_db"] = float(np.median(up)), float(np.median(down))
        f["pitch_fifth_db"] = float(np.median(fifth))

    # --- Модуляция (косвенно): гармоничность нот — хорус и детюн ее снижают ---
    frames = librosa.util.frame(mid[:len(mid) // hop * hop], frame_length=N_FFT, hop_length=hop * 4)
    energy = np.sum(frames ** 2, axis=0)
    loud = frames[:, energy > np.percentile(energy, 70)]
    if loud.shape[1]:
        ac = np.array([np.correlate(x, x, "full")[N_FFT - 1:] for x in loud[:, :400].T])
        ac /= ac[:, :1] + 1e-12
        lo, hi = int(sr / 800.0), int(sr / 70.0)
        harm = np.median(ac[:, lo:hi].max(axis=1))
        f["harmonicity"] = float(harm)
    return f
