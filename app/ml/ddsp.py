"""
Дифференцируемые версии эффектов тона DSPEngine (torch): нужны, чтобы учить сеть по звуку — ошибка между рендером
предсказанной цепочки и референсом прокидывается в параметры. Каждый эффект повторяет pedalboard-реализацию
DSPEngine (проверка — tests/test_ddsp.py); параметры — нормализованные [0..1] из DATAGEN_PLUGINS, как в метке.

Сигнал — тензор (batch, channels, samples); параметры — тензоры (batch,).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from app.dataset.dataset_schema import DATAGEN_PLUGINS

EQ_KINDS = ("low_shelf", "peak", "peak", "peak", "peak", "peak", "high_shelf")
EQ_Q = (0.707, 1.0, 1.0, 1.0, 1.0, 1.0, 0.707)


def denormalize(plugin: str, param: str, value: torch.Tensor) -> torch.Tensor:
    """Нормализованный параметр [0..1] → физический (как DSPEngine.denormalize_value, без обрезки градиента)."""
    lo, hi = DATAGEN_PLUGINS[plugin]["params"][param]
    return lo + value * (hi - lo)


def biquad_coefficients(kind: str, freq: torch.Tensor, gain_db: torch.Tensor, q: float, sr: int
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Коэффициенты (b, a) формы (batch, 3) — те же формулы, что juce::dsp::IIR::Coefficients (makePeakFilter,
    makeLowShelf, makeHighShelf), которыми pedalboard строит PeakFilter / LowShelfFilter / HighShelfFilter.
    """
    a_gain = torch.pow(10.0, gain_db / 40.0)
    omega = 2.0 * math.pi * freq / sr
    cos_o, sin_o = torch.cos(omega), torch.sin(omega)
    if kind == "peak":
        alpha = sin_o / (2.0 * q)
        b = torch.stack([1 + alpha * a_gain, -2 * cos_o, 1 - alpha * a_gain], dim=-1)
        a = torch.stack([1 + alpha / a_gain, -2 * cos_o, 1 - alpha / a_gain], dim=-1)
        return b, a
    am1, ap1 = a_gain - 1, a_gain + 1
    beta = sin_o * torch.sqrt(a_gain) / q
    am1_cos = am1 * cos_o
    if kind == "low_shelf":
        b = torch.stack([a_gain * (ap1 - am1_cos + beta), a_gain * 2 * (am1 - ap1 * cos_o),
                         a_gain * (ap1 - am1_cos - beta)], dim=-1)
        a = torch.stack([ap1 + am1_cos + beta, -2 * (am1 + ap1 * cos_o), ap1 + am1_cos - beta], dim=-1)
        return b, a
    if kind == "high_shelf":
        b = torch.stack([a_gain * (ap1 + am1_cos + beta), a_gain * -2 * (am1 + ap1 * cos_o),
                         a_gain * (ap1 + am1_cos - beta)], dim=-1)
        a = torch.stack([ap1 - am1_cos + beta, 2 * (am1 - ap1 * cos_o), ap1 - am1_cos - beta], dim=-1)
        return b, a
    raise ValueError(kind)


def biquad_response(b: torch.Tensor, a: torch.Tensor, n_fft: int) -> torch.Tensor:
    """Комплексная частотная характеристика (batch, n_fft // 2 + 1) биквада на сетке rfft."""
    w = torch.linspace(0, math.pi, n_fft // 2 + 1, device=b.device, dtype=b.dtype)
    z = torch.exp(-1j * w)  # z^-1
    z2 = z * z
    num = b[:, 0:1] + b[:, 1:2] * z + b[:, 2:3] * z2
    den = a[:, 0:1] + a[:, 1:2] * z + a[:, 2:3] * z2
    return num / den


def eq_response(params: dict[str, torch.Tensor], sr: int, n_fft: int, plugin: str = "eq") -> torch.Tensor:
    """Частотная характеристика 7-полосного EQ DSPEngine (batch, n_fft // 2 + 1)."""
    h = None
    for i, (kind, q) in enumerate(zip(EQ_KINDS, EQ_Q), start=1):
        freq = denormalize("eq", f"band{i}_freq_hz", params[f"band{i}_freq_hz"])
        gain = denormalize("eq", f"band{i}_gain_db", params[f"band{i}_gain_db"])
        b, a = biquad_coefficients(kind, freq.double(), gain.double(), q, sr)
        band = biquad_response(b, a, n_fft)
        h = band if h is None else h * band
    return h


def apply_eq(x: torch.Tensor, params: dict[str, torch.Tensor], sr: int) -> torch.Tensor:
    """
    EQ как произведение спектров: БИХ-фильтры pedalboard затухают быстро, поэтому при дополнении нулями на
    полсекунды результат совпадает с рекурсивной фильтрацией с точностью до хвоста за концом сигнала.
    """
    n = x.shape[-1]
    n_fft = 1 << math.ceil(math.log2(n + sr // 2))
    h = eq_response(params, sr, n_fft).to(torch.complex64 if x.dtype == torch.float32 else torch.complex128)
    spec = torch.fft.rfft(x, n=n_fft)
    return torch.fft.irfft(spec * h[:, None, :], n=n_fft)[..., :n]


# --- Компрессор -----------------------------------------------------------------------------------------------

_KERNELS = None


def _kernels():
    global _KERNELS
    if _KERNELS is None:
        from numba import njit, prange

        @njit(parallel=True, cache=True)
        def forward(x, ca, cr):
            rows, n = x.shape
            y = np.empty_like(x)
            attack = np.zeros(x.shape, dtype=np.bool_)
            for r in prange(rows):
                prev = 0.0
                for t in range(n):
                    is_attack = x[r, t] > prev
                    c = ca[r] if is_attack else cr[r]
                    prev = x[r, t] + c * (prev - x[r, t])
                    y[r, t] = prev
                    attack[r, t] = is_attack
            return y, attack

        @njit(parallel=True, cache=True)
        def backward(grad_y, x, y, attack, ca, cr):
            """Обратный проход при зафиксированных фазах атаки/спада: та же рекурсия во времени назад."""
            rows, n = x.shape
            grad_x = np.empty_like(x)
            grad_ca = np.zeros(rows)
            grad_cr = np.zeros(rows)
            for r in prange(rows):
                acc = 0.0
                for t in range(n - 1, -1, -1):
                    c = ca[r] if attack[r, t] else cr[r]
                    acc += grad_y[r, t]
                    grad_x[r, t] = (1.0 - c) * acc
                    prev = y[r, t - 1] if t > 0 else 0.0
                    if attack[r, t]:
                        grad_ca[r] += (prev - x[r, t]) * acc
                    else:
                        grad_cr[r] += (prev - x[r, t]) * acc
                    acc *= c
            return grad_x, grad_ca, grad_cr

        _KERNELS = (forward, backward)
    return _KERNELS


class Ballistics(torch.autograd.Function):
    """Огибающая (rows, samples) с коэффициентами атаки и спада на строку; считается на CPU (numba)."""

    @staticmethod
    def forward(ctx, x, coef_attack, coef_release):
        xn = x.detach().double().cpu().numpy()
        ca, cr = coef_attack.detach().double().cpu().numpy(), coef_release.detach().double().cpu().numpy()
        y, attack = _kernels()[0](xn, ca, cr)
        ctx.save_for_backward(x, coef_attack, coef_release)
        ctx.arrays = (xn, y, attack, ca, cr)
        return torch.from_numpy(y).to(device=x.device, dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_y):
        x, coef_attack, coef_release = ctx.saved_tensors
        xn, y, attack, ca, cr = ctx.arrays
        gx, gca, gcr = _kernels()[1](grad_y.detach().double().cpu().numpy(), xn, y, attack, ca, cr)

        def back(a, like):
            return torch.from_numpy(a).to(device=like.device, dtype=like.dtype)
        return back(gx, x), back(gca, coef_attack), back(gcr, coef_release)


def time_coefficient(ms: torch.Tensor, sr: int) -> torch.Tensor:
    """Коэффициент сглаживания juce::dsp::BallisticsFilter для времени в мс."""
    return torch.exp(-2.0 * math.pi * 1000.0 / (sr * ms))


def compress(x: torch.Tensor, threshold_db: torch.Tensor, ratio: torch.Tensor, attack_ms: torch.Tensor,
             release_ms: torch.Tensor, sr: int) -> torch.Tensor:
    """
    juce::dsp::Compressor (pedalboard Compressor): каналы независимо, пиковая огибающая, жесткое колено;
    усиление (env / порог)^(1/ratio − 1) выше порога. Параметры — физические тензоры (batch,).
    """
    batch, channels, n = x.shape
    rows = lambda v: v.repeat_interleave(channels)  # noqa: E731
    env = Ballistics.apply(x.abs().reshape(batch * channels, n), rows(time_coefficient(attack_ms, sr)),
                           rows(time_coefficient(release_ms, sr))).reshape(batch, channels, n)
    thr = torch.pow(10.0, threshold_db / 20.0)[:, None, None]
    over = env / thr
    gain = torch.where(over > 1.0, torch.pow(over.clamp_min(1.0), (1.0 / ratio - 1.0)[:, None, None]),
                       torch.ones_like(over))
    return x * gain


def apply_compressor(x: torch.Tensor, params: dict[str, torch.Tensor], sr: int) -> torch.Tensor:
    """Компрессор DSPEngine: compress + автоматический частичный makeup (как в DSPEngine.apply_chain)."""
    thr = denormalize("compressor", "threshold_db", params["threshold_db"])
    ratio = denormalize("compressor", "ratio", params["ratio"])
    y = compress(x, thr, ratio, denormalize("compressor", "attack_ms", params["attack_ms"]),
                 denormalize("compressor", "release_ms", params["release_ms"]), sr)
    makeup_db = torch.clamp((-10.0 - thr) * (1.0 - 1.0 / ratio.clamp_min(1.0)) * 0.45, 0.0, 12.0)
    makeup_db = torch.where(makeup_db > 0.1, makeup_db, torch.zeros_like(makeup_db))
    return y * torch.pow(10.0, makeup_db / 20.0)[:, None, None]


# --- Фильтры первого порядка, де-эссер, дисторшн -----------------------------------------------------------------

def first_order_response(kind: str, freq: float, sr: int, n_fft: int, device=None) -> torch.Tensor:
    """АЧХ juce makeFirstOrderLowPass / HighPass (pedalboard LowpassFilter / HighpassFilter, 6 дБ/окт)."""
    n = math.tan(math.pi * freq / sr)
    b = (n, n) if kind == "lowpass" else (1.0, -1.0)
    a = (n + 1.0, n - 1.0)
    w = torch.linspace(0, math.pi, n_fft // 2 + 1, device=device, dtype=torch.float64)
    z = torch.exp(-1j * w)
    return (b[0] + b[1] * z) / (a[0] + a[1] * z)


def _filter(x: torch.Tensor, h_fn) -> torch.Tensor:
    n = x.shape[-1]
    n_fft = 1 << math.ceil(math.log2(n + 4096))
    h = h_fn(n_fft).to(torch.complex64 if x.dtype == torch.float32 else torch.complex128)
    return torch.fft.irfft(torch.fft.rfft(x, n=n_fft) * h, n=n_fft)[..., :n]


# Полоса сибилянтов де-эссера: в ней и детектор, и ослабление. Воздух выше нее не трогается: прежний де-эссер
# (раздел полос первого порядка на 4.2 кГц, сжатие всего верха) ловил гласные склоном 2–4 кГц и на пороге −31 дБ
# срезал −6 дБ выше 10 кГц, почти не трогая сибилянты
DEESS_BAND_HZ = (4500.0, 11000.0)
DEESS_DETECT_BOOST_DB = 24.0  # пики полосы сибилянтов на −23 LUFS ~−32…−42 дБ: +24 дБ — в диапазон порога
DEESS_ATTACK_MS, DEESS_RELEASE_MS = 1.5, 35.0


def _deess_band_response(sr: int, n_fft: int, device) -> torch.Tensor:
    """АЧХ полосы сибилянтов (1, n_fft // 2 + 1): Баттерворт 4-го порядка с двух сторон, нулевая фаза."""
    lo = torch.tensor([DEESS_BAND_HZ[0]], device=device, dtype=torch.float64)
    hi = torch.tensor([DEESS_BAND_HZ[1]], device=device, dtype=torch.float64)
    return (butterworth_response("highpass", lo, sr, n_fft) * butterworth_response("lowpass", hi, sr, n_fft)).abs()


def deess(x: torch.Tensor, threshold_db: torch.Tensor, ratio: torch.Tensor, sr: int) -> torch.Tensor:
    """
    Split-band де-эссер с физическими параметрами (batch,): детектор — пиковая огибающая полосы сибилянтов
    (каналы связаны, + DEESS_DETECT_BOOST_DB), ослабление выше порога (жесткое колено) — только этой полосы,
    остальной сигнал проходит как есть. Это же считает DSPEngine.apply_chain (одна реализация для рендера и обучения по звуку).
    """
    band = _filter(x, lambda n_fft: _deess_band_response(sr, n_fft, x.device))
    boost = 10.0 ** (DEESS_DETECT_BOOST_DB / 20.0)
    detector = band.abs().amax(dim=1) * boost
    ones = torch.ones_like(threshold_db, dtype=torch.float64)
    env = Ballistics.apply(detector, time_coefficient(DEESS_ATTACK_MS * ones, sr),
                           time_coefficient(DEESS_RELEASE_MS * ones, sr))
    over = env / torch.pow(10.0, threshold_db / 20.0)[:, None]
    gain = torch.where(over > 1.0, torch.pow(over.clamp_min(1.0), (1.0 / ratio - 1.0)[:, None]),
                       torch.ones_like(over))
    return x + band * (gain[:, None, :] - 1.0)


def apply_deesser(x: torch.Tensor, params: dict[str, torch.Tensor], sr: int) -> torch.Tensor:
    """Де-эссер DSPEngine (deess) с нормализованными параметрами."""
    return deess(x, denormalize("deesser", "threshold_db", params["threshold_db"]),
                 denormalize("deesser", "ratio", params["ratio"]), sr)


def apply_distortion(x: torch.Tensor, params: dict[str, torch.Tensor], sr: int) -> torch.Tensor:
    """Параллельная сатурация DSPEngine: tanh с drive, полоса 150 Гц – 5.5 кГц, wet 22–38 %."""
    norm = params["drive_db"]
    gain = torch.pow(10.0, denormalize("distortion", "drive_db", norm) / 20.0)[:, None, None]
    wet = torch.tanh(x * gain)
    wet = _filter(wet, lambda n_fft: first_order_response("highpass", 150.0, sr, n_fft, x.device)
                  * first_order_response("lowpass", 5500.0, sr, n_fft, x.device))
    mix = (0.22 + 0.16 * norm.clamp(0.0, 1.0))[:, None, None]
    return (1.0 - mix) * x + mix * wet


# --- Мультибэнд (OTT) и экситер ----------------------------------------------------------------------------------

def butterworth_response(kind: str, freq: torch.Tensor, sr: int, n_fft: int) -> torch.Tensor:
    """
    АЧХ scipy.signal.butter(4, freq, kind) (batch, n_fft // 2 + 1): два биквада RBJ с добротностями
    4-го порядка Баттерворта — то же билинейное преобразование с предыскажением на частоте среза.
    """
    freq = freq.double()
    omega = 2.0 * math.pi * freq / sr
    cos_o, sin_o = torch.cos(omega), torch.sin(omega)
    h = None
    for q in (1.0 / (2 * math.cos(math.pi / 8)), 1.0 / (2 * math.cos(3 * math.pi / 8))):
        alpha = sin_o / (2.0 * q)
        if kind == "lowpass":
            b = torch.stack([(1 - cos_o) / 2, 1 - cos_o, (1 - cos_o) / 2], dim=-1)
        else:
            b = torch.stack([(1 + cos_o) / 2, -(1 + cos_o), (1 + cos_o) / 2], dim=-1)
        a = torch.stack([1 + alpha, -2 * cos_o, 1 - alpha], dim=-1)
        band = biquad_response(b, a, n_fft)
        h = band if h is None else h * band
    return h


_OTT_BANDS = (  # те же пороги, что VocalEnhancer.apply_multiband_compressor
    (-26.0, 5.0, -40.0, 3.2, 4.5),
    (-22.0, 4.5, -34.0, 3.0, 2.5),
    (-30.0, 5.5, -44.0, 4.0, 7.5),
)


def apply_multiband(x: torch.Tensor, params: dict[str, torch.Tensor], sr: int) -> torch.Tensor:
    """OTT DSPEngine: 3 полосы (150 Гц / 2.5 кГц), RMS 12 мс по Mid, вниз и вверх, затем подмес depth."""
    batch = x.shape[0]
    ones = torch.ones(batch, device=x.device, dtype=torch.float64)
    low = _filter(x, lambda n_fft: butterworth_response("lowpass", 150.0 * ones, sr, n_fft)[:, None, :])
    high = _filter(x, lambda n_fft: butterworth_response("highpass", 2500.0 * ones, sr, n_fft)[:, None, :])
    bands = (low, x - low - high, high)
    win = max(1, int(0.012 * sr))
    kernel = torch.ones(1, 1, win, device=x.device, dtype=x.dtype) / win
    out = torch.zeros_like(x)
    for band, (t_down, r_down, t_up, r_up, makeup) in zip(bands, _OTT_BANDS):
        power = band.mean(dim=1, keepdim=True).pow(2)
        # signal.convolve(..., mode="same") с четным окном: выход сдвинут на (win − 1) // 2 отсчетов
        env_sq = torch.nn.functional.conv1d(power, kernel, padding=win - 1)[..., (win - 1) - (win - 1) // 2:]
        env_sq = env_sq[..., :x.shape[-1]]
        env_db = 20 * torch.log10(torch.sqrt(env_sq.clamp_min(1e-10)) + 1e-8)
        gain_db = makeup + torch.where(env_db > t_down, (t_down - env_db) * (1 - 1 / r_down),
                                       torch.zeros_like(env_db))
        up = torch.clamp((t_up - env_db) * (1 - 1 / r_up) * 0.65, max=12.0)
        gain_db = gain_db + torch.where((env_db < t_up) & (env_db > -58.0), up, torch.zeros_like(env_db))
        out = out + band * torch.pow(10.0, gain_db / 20.0)
    depth = denormalize("multiband_compressor", "depth", params["depth"])[:, None, None]
    return (1 - depth) * x + depth * out


def apply_exciter(x: torch.Tensor, params: dict[str, torch.Tensor], sr: int) -> torch.Tensor:
    """Экситер DSPEngine (VocalEnhancer.apply_exciter, drive 2): ВЧ выше среза → tanh-гармоники → подмес."""
    cutoff = denormalize("exciter", "cutoff_hz", params["cutoff_hz"]).clamp(3500.0, sr * 0.44)
    mix = denormalize("exciter", "mix", params["mix"])
    hp = lambda n_fft: butterworth_response("highpass", cutoff, sr, n_fft)[:, None, :]  # noqa: E731
    high = _filter(x, hp)
    flat = high.abs().flatten(1)
    p98 = torch.quantile(flat, 0.98, dim=1)[:, None, None] + 1e-6
    norm = high / p98
    drive = 2.0
    harmonics = torch.tanh(norm * (drive * 1.8)) + 0.35 * torch.tanh(norm.pow(2) * drive)
    air = _filter(harmonics, hp)
    rms = lambda v: torch.sqrt(v.pow(2).flatten(1).mean(dim=1))[:, None, None] + 1e-8  # noqa: E731
    target = torch.maximum(rms(high) * 2.2, rms(x) * 0.28)
    air = air * (target / rms(air))
    return x + (mix * 1.6)[:, None, None] * air


# --- Цепочка тона ------------------------------------------------------------------------------------------------

TONE_EFFECTS = {  # в порядке CHAIN_ORDER; тюн (PSOLA) и хорус (рекурсия с модуляцией) не дифференцируются
    "compressor": apply_compressor,
    "deesser": apply_deesser,
    "multiband_compressor": apply_multiband,
    "distortion": apply_distortion,
}


def render_tone_chain(x: torch.Tensor, params: dict[str, dict[str, torch.Tensor]], sr: int,
                      on: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
    """
    EQ и эффекты тона в порядке CHAIN_ORDER. on — вероятность включения эффекта (batch,): выход эффекта
    смешивается с его входом, так что градиент доходит и до решения «включать». on=None — все из params.
    Выходной лимитер DSPEngine не повторяется: сравнение идет на нормализованном уровне.
    """
    y = apply_eq(x, params["eq"], sr) if "eq" in params else x
    for name, fx in TONE_EFFECTS.items():
        if name not in params:
            continue
        wet = fx(y, params[name], sr)
        if on is None or name not in on:
            y = wet
        else:
            gate = on[name][:, None, None].to(y.dtype)
            y = y + gate * (wet - y)
    return y
