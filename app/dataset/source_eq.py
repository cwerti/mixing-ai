"""
EQ обучающих цепочек, зависящий от исходника: как у инженера, который подтягивает тональный баланс голоса к
целевому. На сведениях Cambridge изменение полосы «сухой → микс» отрицательно связано с ее уровнем в сухом
(низ < 150 Гц и 2–5 кГц: корреляция −0.5), поэтому одна и та же кривая EQ с чужого трека на другом голосе дает
случайный результат. Цель — спектр случайного настоящего сведенного вокала (scripts/calibrate_generator.py eq).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

REAL_EQ_PATH = Path(__file__).resolve().parents[1] / "data" / "real_eq_curves.json"
# Сетка спектра (1/6 октавы): выше 16 кГц вокал Demucs ненадежен
GRID_HZ = np.geomspace(40.0, 16000.0, 53)
# Опорные частоты полос EQ генератора внутри диапазонов DATAGEN_PLUGINS["eq"] (полоса 1 и 7 — полки)
EQ_CENTERS_HZ = (80.0, 160.0, 450.0, 1300.0, 3000.0, 7000.0, 11000.0)
_MID = (GRID_HZ >= 200) & (GRID_HZ <= 5000)
# Потолок и штраф EQ генератора: без них подгонка под чужой спектр упирала 23% полос в ±12 дБ с соседями
# разного знака (у EQ «сухой → сведение» Cambridge таких 10%) — на слух бубнящий низ и «гребенка»
SOURCE_EQ_LIMIT_DB = 9.0
SOURCE_EQ_RIDGE = 2.0


def ltas_db(y: np.ndarray, sr: int) -> np.ndarray:
    """Средний спектр Mid на GRID_HZ (сглаживание 1/6 октавы), относительно среднего в 200 Гц – 5 кГц."""
    from scipy.signal import welch
    mid = np.mean(y, axis=0) if y.ndim == 2 else y
    f, p = welch(mid, sr, nperseg=min(16384, len(mid)))
    out = np.empty(len(GRID_HZ))
    for i, fc in enumerate(GRID_HZ):
        band = (f >= fc * 2 ** (-1 / 12)) & (f < fc * 2 ** (1 / 12))
        power = np.mean(p[band]) if band.any() else np.interp(fc, f, p)
        out[i] = 10 * np.log10(power + 1e-20)
    return out - out[_MID].mean()


def eq_response_db(freqs: np.ndarray, gains: np.ndarray, sr: int) -> np.ndarray:
    """АЧХ 7-полосного EQ DSPEngine на GRID_HZ без общего уровня (те же биквады, что app.ml.ddsp)."""
    import torch
    from app.ml.ddsp import EQ_KINDS, EQ_Q, biquad_coefficients
    w = torch.from_numpy(2 * np.pi * GRID_HZ / sr)
    z = torch.exp(-1j * w)
    h = torch.ones(len(GRID_HZ), dtype=torch.complex128)
    for kind, q, fc, g in zip(EQ_KINDS, EQ_Q, freqs, gains):
        b, a = biquad_coefficients(kind, torch.tensor([float(fc)], dtype=torch.float64),
                                   torch.tensor([float(g)], dtype=torch.float64), q, sr)
        h = h * (b[0, 0] + b[0, 1] * z + b[0, 2] * z * z) / (a[0, 0] + a[0, 1] * z + a[0, 2] * z * z)
    resp = 20 * np.log10(np.abs(h.numpy()) + 1e-12)
    return resp - resp[_MID].mean()


def fit_eq_gains(curve: np.ndarray, freqs: np.ndarray, sr: int, steps: int = 4, ridge: float = 0.5,
                 limit: float = 12.0) -> tuple[np.ndarray, float]:
    """
    Усиления полос (дБ, ±limit), чья АЧХ ближе всего к кривой (обе без общего уровня): Гаусс-Ньютон со слабой
    тягой к нулю. Возвращает усиления и остаточную ошибку (средний модуль, дБ).
    """
    gains = np.zeros(len(freqs))
    for _ in range(steps):
        base = eq_response_db(freqs, gains, sr)
        jac = np.stack([eq_response_db(freqs, gains + (np.arange(len(gains)) == i), sr) - base
                        for i in range(len(gains))], axis=1)
        reg = ridge * np.eye(len(gains))
        gains = np.clip(gains + np.linalg.solve(jac.T @ jac + reg, jac.T @ (curve - base) - reg @ gains),
                        -limit, limit)
    return gains, float(np.mean(np.abs(curve - eq_response_db(freqs, gains, sr))))


def smooth_octave(curve: np.ndarray, octaves: float = 1.0) -> np.ndarray:
    """
    Сглаживание кривой по сетке GRID_HZ (окно в octaves): тональный баланс без высоты и формант конкретного
    голоса. Разница спектров двух разных голосов на сетке 1/6 октавы — это в основном их высота (±8 дБ), а не
    то, что переносит EQ.
    """
    half = int(round(octaves * 6 / 2))
    padded = np.pad(curve, half, mode="edge")
    return np.convolve(padded, np.ones(2 * half + 1) / (2 * half + 1), mode="valid")


_TARGETS: tuple[list[np.ndarray], np.ndarray] | None = None


def real_targets() -> list[np.ndarray]:
    """Спектры настоящих сведенных вокалов (ltas_db) — цели EQ."""
    return _load_targets()[0]


def target_bias() -> np.ndarray:
    """
    Поправка цели EQ (дБ на GRID_HZ): систематическое расхождение спектра генератора с настоящими сведениями
    на тех же сухих (scripts/calibrate_generator.py compare --update-bias). Перенос спектра чужого сведения в
    логарифмической мере не воспроизводит энергетический баланс: инженер прибирает низкую середину, которая
    доминирует в энергии, — и 2–5 кГц у генератора оставались на 2.5 дБ ниже настоящих.
    """
    return _load_targets()[1]


def _load_targets() -> tuple[list[np.ndarray], np.ndarray]:
    global _TARGETS
    if _TARGETS is None:
        data = json.loads(REAL_EQ_PATH.read_text(encoding="utf-8"))
        _TARGETS = ([np.asarray(t, dtype=float) for t in data["targets_db"]],
                    np.asarray(data.get("bias_db", np.zeros(len(GRID_HZ))), dtype=float))
    return _TARGETS


def source_aware_eq(y: np.ndarray, sr: int, rng: random.Random, render=None, passes: int = 2
                    ) -> dict[str, float]:
    """
    Нормализованные параметры EQ, которые подтягивают спектр голоса к спектру случайного настоящего сведения на
    долю 0.4–1.0 (инженер не всегда выравнивает до конца), с частотами ±1/3 октавы от опорных.

    render(params) — рендер цепочки с этим EQ: тогда EQ выставляется по результату всей цепочки за passes
    поправок (как на слух), иначе — по самому y. Динамические эффекты после EQ частично гасят его подъем, и
    поправка по сухому не доходит до цели.
    """
    from app.audio.dsp_engine import DSPEngine
    from app.dataset.dataset_schema import DATAGEN_PLUGINS
    ranges = DATAGEN_PLUGINS["eq"]["params"]
    freqs = np.array([np.clip(fc * 2 ** rng.uniform(-1 / 3, 1 / 3), *ranges[f"band{i}_freq_hz"])
                      for i, fc in enumerate(EQ_CENTERS_HZ, start=1)])
    amount, target = rng.uniform(0.4, 1.0), rng.choice(real_targets())

    def to_params(gains: np.ndarray) -> dict[str, float]:
        params = {}
        for i, (fc, g) in enumerate(zip(freqs, gains), start=1):
            params[f"band{i}_freq_hz"] = DSPEngine.normalize_value(float(fc), *ranges[f"band{i}_freq_hz"])
            params[f"band{i}_gain_db"] = DSPEngine.normalize_value(float(g), *ranges[f"band{i}_gain_db"])
        return params

    gains = np.zeros(len(freqs))
    start = smooth_octave(ltas_db(render(to_params(gains)) if render else y, sr))
    goal = start + amount * (smooth_octave(target) - start) + target_bias()
    fit = dict(ridge=SOURCE_EQ_RIDGE, limit=SOURCE_EQ_LIMIT_DB)
    gains, _ = fit_eq_gains(goal - start, freqs, sr, **fit)
    for _ in range(passes if render else 0):
        residual = goal - smooth_octave(ltas_db(render(to_params(gains)), sr))
        step, _ = fit_eq_gains(residual, freqs, sr, **fit)
        gains = np.clip(gains + step, -SOURCE_EQ_LIMIT_DB, SOURCE_EQ_LIMIT_DB)
    return to_params(gains)
