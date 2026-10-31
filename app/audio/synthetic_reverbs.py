"""
Алгоритмические ревербераторы для банка импульсных характеристик (только генерация данных, не путь рендера
оптимизатора).

Реальные инженеры почти не используют свёртку с залами: на вокале стоят алгоритмические плейты и холлы
(Valhalla, Lexicon, стоковые Reeverb/Luxeverb), пружины и гейтованные ревербы. Здесь — три классические
топологии, из которых строятся синтетические ИХ (scripts/build_ir_cache.py):
  - plate_ir: плейт Даторро (J. Dattorro, «Effect Design, Part 1», JAES 1997) — входные диффузоры и
    «бак» из двух ветвей с затуханием и демпфированием;
  - fdn_ir: сеть задержек с обратной связью (Jot, 1991) на 8/16 линий с матрицей Адамара, двухполосным
    затуханием (свой RT60 для низа и верха) и входными диффузорами;
  - spring_ir: пружина — дисперсионная цепочка фазовращателей в петле задержки (характерный «чирп»).
Сами ИХ потом просто свёртываются с сигналом (app/audio/effect_variants.py), поэтому скорость рекурсии
важна только при построении кэша.
"""
import numpy as np
from numba import njit
from scipy.signal import butter, sosfilt

# Длины линий плейта Даторро в сэмплах при 29761 Гц (из статьи), масштабируются под sr и «размер»
_DATTORRO_SR = 29761.0
_DATTORRO_INPUT_AP = (142, 107, 379, 277)
_DATTORRO_TANK = (672, 4453, 1800, 3720, 908, 4217, 2656, 3163)
# Отводы выхода (левый и правый каналы), из статьи: (линия, позиция, знак)
_DATTORRO_TAPS_L = ((1, 266, 1.0), (1, 2974, 1.0), (2, 1913, -1.0), (3, 1996, 1.0), (5, 1990, -1.0),
                    (6, 187, -1.0), (7, 1066, -1.0))
_DATTORRO_TAPS_R = ((5, 353, 1.0), (5, 3627, 1.0), (6, 1228, -1.0), (7, 2673, 1.0), (1, 2111, -1.0),
                    (2, 335, -1.0), (3, 121, -1.0))


@njit(cache=True)
def _plate_kernel(n: int, input_ap: np.ndarray, tank: np.ndarray, taps_l: np.ndarray, taps_r: np.ndarray,
                  decay: float, damping: float, bandwidth: float, in_diff1: float, in_diff2: float,
                  dec_diff1: float, dec_diff2: float, mod_depth: int, mod_rate: float, sr: float) -> np.ndarray:
    out = np.zeros((2, n))
    # Входные диффузоры
    ap_buf = np.zeros((4, int(input_ap.max()) + 1))
    ap_idx = np.zeros(4, dtype=np.int64)
    # Бак: 8 линий (ap1, delay1, ap2, delay2 для каждой ветви)
    max_len = int(tank.max()) + mod_depth + 2
    bufs = np.zeros((8, max_len))
    widx = np.zeros(8, dtype=np.int64)
    bw_state = 0.0
    damp_a = 0.0
    damp_b = 0.0
    fb_a = 0.0
    fb_b = 0.0
    for t in range(n):
        x = 1.0 if t == 0 else 0.0
        bw_state = bandwidth * x + (1.0 - bandwidth) * bw_state
        s = bw_state
        for i in range(4):
            d = int(input_ap[i])
            j = ap_idx[i]
            delayed = ap_buf[i, j]
            g = in_diff1 if i < 2 else in_diff2
            v = s + g * delayed
            # Фазовращатель Шрёдера: y = -g·v + x[n-d], v = x + g·v[n-d]
            s = delayed - g * v
            ap_buf[i, j] = v
            ap_idx[i] = (j + 1) % (d + 1)
        # Две ветви бака, перекрестная обратная связь
        mod = mod_depth * np.sin(2.0 * np.pi * mod_rate * t / sr)
        for branch in range(2):
            base = branch * 4
            inp = s + (fb_b if branch == 0 else fb_a)
            # Модулированный фазовращатель (decay diffusion 1, знак минус как у Даторро)
            d1 = tank[base] + (mod if branch == 0 else -mod)
            ln = max_len
            rd = (widx[base] - int(d1) + ln) % ln
            delayed = bufs[base, rd]
            v = inp - (-dec_diff1) * delayed
            y = delayed + (-dec_diff1) * v
            bufs[base, widx[base]] = v
            widx[base] = (widx[base] + 1) % ln
            # Линия задержки 1
            bufs[base + 1, widx[base + 1]] = y
            rd = (widx[base + 1] - int(tank[base + 1]) + ln) % ln
            y = bufs[base + 1, rd]
            widx[base + 1] = (widx[base + 1] + 1) % ln
            # Демпфирование и затухание
            if branch == 0:
                damp_a = (1.0 - damping) * y + damping * damp_a
                y = damp_a * decay
            else:
                damp_b = (1.0 - damping) * y + damping * damp_b
                y = damp_b * decay
            # Фазовращатель decay diffusion 2
            rd = (widx[base + 2] - int(tank[base + 2]) + ln) % ln
            delayed = bufs[base + 2, rd]
            v = y + dec_diff2 * delayed
            y = delayed - dec_diff2 * v
            bufs[base + 2, widx[base + 2]] = v
            widx[base + 2] = (widx[base + 2] + 1) % ln
            # Линия задержки 2
            bufs[base + 3, widx[base + 3]] = y
            rd = (widx[base + 3] - int(tank[base + 3]) + ln) % ln
            y = bufs[base + 3, rd]
            widx[base + 3] = (widx[base + 3] + 1) % ln
            if branch == 0:
                fb_a = y * decay
            else:
                fb_b = y * decay
        # Отводы выхода: позиция отвода — отставание от текущей точки записи линии
        acc_l = 0.0
        acc_r = 0.0
        for k in range(taps_l.shape[0]):
            line = int(taps_l[k, 0])
            pos = (widx[line] - 1 - int(taps_l[k, 1]) + 2 * max_len) % max_len
            acc_l += taps_l[k, 2] * bufs[line, pos]
        for k in range(taps_r.shape[0]):
            line = int(taps_r[k, 0])
            pos = (widx[line] - 1 - int(taps_r[k, 1]) + 2 * max_len) % max_len
            acc_r += taps_r[k, 2] * bufs[line, pos]
        out[0, t] = acc_l
        out[1, t] = acc_r
    return out


def plate_ir(sr: int, seconds: float, decay: float, damping: float, size: float = 1.0, bandwidth: float = 0.9995,
             mod_depth_samples: int = 8, mod_rate_hz: float = 1.0) -> np.ndarray:
    """
    Стерео ИХ плейта Даторро (2, n).

    Args:
        decay: Коэффициент затухания бака (0.3–0.95; 0.5 ≈ 1.5 с, 0.9 ≈ 6 с при size 1).
        damping: Демпфирование верха в баке (0 — светлый, 0.8 — темный).
        size: Масштаб длин линий (0.5–1.6).
    """
    scale = sr / _DATTORRO_SR * size
    input_ap = np.array([max(1, int(d * scale)) for d in _DATTORRO_INPUT_AP], dtype=np.int64)
    tank = np.array([max(2, int(d * scale)) for d in _DATTORRO_TANK], dtype=np.float64)
    # Отводы: строки (линия бака, позиция, знак). Линии в ядре: 0 ap1A, 1 del1A, 2 ap2A, 3 del2A, 4..7 — ветвь B
    tank_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
    taps_l = np.array([(tank_map[line], min(int(pos * scale), int(tank[line]) - 1), sign)
                       for line, pos, sign in _DATTORRO_TAPS_L], dtype=np.float64)
    taps_r = np.array([(tank_map[line], min(int(pos * scale), int(tank[line]) - 1), sign)
                       for line, pos, sign in _DATTORRO_TAPS_R], dtype=np.float64)
    n = int(seconds * sr)
    ir = _plate_kernel(n, input_ap, tank, taps_l, taps_r, float(decay), float(damping), float(bandwidth),
                       0.75, 0.625, 0.7, 0.5, int(mod_depth_samples * sr / _DATTORRO_SR), float(mod_rate_hz),
                       float(sr))
    return ir.astype(np.float32)


@njit(cache=True)
def _fdn_kernel(n: int, delays: np.ndarray, g_low: np.ndarray, g_high: np.ndarray, split_a: float,
                matrix: np.ndarray, out_l: np.ndarray, out_r: np.ndarray, in_gain: np.ndarray) -> np.ndarray:
    m = delays.shape[0]
    max_len = int(delays.max()) + 1
    bufs = np.zeros((m, max_len))
    widx = np.zeros(m, dtype=np.int64)
    lp = np.zeros(m)
    outs = np.zeros(m)
    out = np.zeros((2, n))
    for t in range(n):
        x = 1.0 if t == 0 else 0.0
        for i in range(m):
            rd = (widx[i] - int(delays[i]) + max_len) % max_len
            v = bufs[i, rd]
            # Двухполосное затухание: низ (одним полюсом) и верх с разными коэффициентами
            lp[i] = split_a * lp[i] + (1.0 - split_a) * v
            outs[i] = g_low[i] * lp[i] + g_high[i] * (v - lp[i])
        acc_l = 0.0
        acc_r = 0.0
        for i in range(m):
            acc_l += out_l[i] * outs[i]
            acc_r += out_r[i] * outs[i]
        out[0, t] = acc_l
        out[1, t] = acc_r
        for i in range(m):
            s = in_gain[i] * x
            for j in range(m):
                s += matrix[i, j] * outs[j]
            bufs[i, widx[i]] = s
            widx[i] = (widx[i] + 1) % max_len
    return out


def _hadamard(m: int) -> np.ndarray:
    h = np.array([[1.0]])
    while h.shape[0] < m:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(m)


def fdn_ir(sr: int, seconds: float, rt60_low: float, rt60_high: float, rng: np.random.Generator,
           n_lines: int = 8, size_ms: tuple[float, float] = (25.0, 90.0), crossover_hz: float = 3000.0,
           diffusion: float = 0.6) -> np.ndarray:
    """
    Стерео ИХ FDN (2, n): линии длиной size_ms (взаимно простые), RT60 отдельно для низа и верха.
    Входные диффузоры (4 фазовращателя) сглаживают ранние отражения, как у алгоритмических холлов.
    """
    lo, hi = size_ms
    delays = np.sort(rng.uniform(lo, hi, n_lines)) * sr / 1000.0
    delays = np.array([_next_prime(int(d)) for d in delays], dtype=np.float64)
    g_low = 10.0 ** (-3.0 * delays / (sr * rt60_low))
    g_high = 10.0 ** (-3.0 * delays / (sr * max(rt60_high, 0.05)))
    split_a = float(np.exp(-2.0 * np.pi * crossover_hz / sr))
    out_l = rng.choice([-1.0, 1.0], n_lines) * rng.uniform(0.5, 1.0, n_lines)
    out_r = rng.choice([-1.0, 1.0], n_lines) * rng.uniform(0.5, 1.0, n_lines)
    in_gain = rng.choice([-1.0, 1.0], n_lines)
    ir = _fdn_kernel(int(seconds * sr), delays, g_low, g_high, split_a, _hadamard(n_lines), out_l, out_r, in_gain)
    if diffusion > 0:
        for d_ms in rng.uniform(1.0, 12.0, 4):
            ir = schroeder_allpass(ir, int(d_ms * sr / 1000.0), diffusion)
    return ir.astype(np.float32)


@njit(cache=True)
def _allpass(x: np.ndarray, d: int, g: float) -> np.ndarray:
    y = np.zeros_like(x)
    buf = np.zeros(d)
    idx = 0
    for t in range(x.shape[0]):
        delayed = buf[idx]
        v = x[t] + g * delayed
        y[t] = delayed - g * v
        buf[idx] = v
        idx = (idx + 1) % d
    return y


def schroeder_allpass(x: np.ndarray, d: int, g: float) -> np.ndarray:
    """Фазовращатель Шрёдера по последней оси."""
    d = max(1, int(d))
    if x.ndim == 1:
        return _allpass(x.astype(np.float64), d, g)
    return np.stack([_allpass(ch.astype(np.float64), d, g) for ch in x])


@njit(cache=True)
def _spring_kernel(n: int, loop_delay: int, n_stages: int, a: float, feedback: float, lp_a: float) -> np.ndarray:
    out = np.zeros(n)
    buf = np.zeros(loop_delay + 1)
    widx = 0
    # Состояния цепочки первого порядка allpass: y = a*x + x1 - a*y1
    x1 = np.zeros(n_stages)
    y1 = np.zeros(n_stages)
    lp = 0.0
    for t in range(n):
        x = 1.0 if t == 0 else 0.0
        rd = (widx - loop_delay + loop_delay + 1) % (loop_delay + 1)
        s = x + feedback * buf[rd]
        for k in range(n_stages):
            y = a * s + x1[k] - a * y1[k]
            x1[k] = s
            y1[k] = y
            s = y
        lp = lp_a * lp + (1.0 - lp_a) * s
        out[t] = lp
        buf[widx] = lp
        widx = (widx + 1) % (loop_delay + 1)
    return out


def spring_ir(sr: int, seconds: float, rng: np.random.Generator, rt60: float = 2.5) -> np.ndarray:
    """
    Стерео ИХ пружинного ревербератора: две пружины разной длины (L/R), дисперсия — 60–120 фазовращателей
    первого порядка, петля 25–60 мс; верх срезан на 3.5–5.5 кГц, низ — на 100–250 Гц.
    """
    chans = []
    for _ in range(2):
        loop_ms = rng.uniform(25.0, 60.0)
        loop = int(loop_ms * sr / 1000.0)
        feedback = 10.0 ** (-3.0 * loop / (sr * rt60))
        lp_a = float(np.exp(-2.0 * np.pi * rng.uniform(3500.0, 5500.0) / sr))
        ir = _spring_kernel(int(seconds * sr), loop, int(rng.integers(60, 121)), float(rng.uniform(0.55, 0.75)),
                            float(feedback), lp_a)
        ir = sosfilt(butter(2, rng.uniform(100.0, 250.0), "high", fs=sr, output="sos"), ir)
        chans.append(ir)
    return np.stack(chans).astype(np.float32)


def _next_prime(k: int) -> int:
    k = max(k, 2)
    while any(k % p == 0 for p in range(2, int(k ** 0.5) + 1)):
        k += 1
    return k


def estimate_rt60(ir: np.ndarray, sr: int) -> float:
    """
    RT60 по кривой Шрёдера (обратное интегрирование энергии): прямая по участку −5…−25 дБ (T20 × 3),
    если затухание мельче — по −5…−15 дБ (T10 × 6). Возвращает NaN, если ИХ слишком короткая.
    """
    mono = ir.mean(axis=0) if ir.ndim == 2 else ir
    energy = np.cumsum((mono.astype(np.float64) ** 2)[::-1])[::-1]
    if energy[0] <= 0:
        return float("nan")
    edc = 10.0 * np.log10(np.maximum(energy / energy[0], 1e-30))
    for top, bottom in ((-5.0, -25.0), (-5.0, -15.0)):
        idx = np.flatnonzero((edc <= top) & (edc >= bottom))
        if len(idx) > 10:
            slope = np.polyfit(idx / sr, edc[idx], 1)[0]  # дБ/с
            if slope < 0:
                return float(-60.0 / slope)
    return float("nan")
