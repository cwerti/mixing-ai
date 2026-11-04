"""
Подбор пресета плагина под мокрый референс без сухого оригинала референса (рабочий сценарий).

1. Слепая сеть описывает референс (медиана по окнам 8 с).
2. Предотбор: K пресетов, чьи отпечатки (сняты заранее тестовыми сигналами) ближе всего к описанию.
3. Двусторонняя проверка: сухой пользователя рендерится через каждого кандидата, та же сеть описывает рендер, и
   выбирается кандидат с описанием ближе всего к описанию референса. Ошибки сети (например, ритм исполнения,
   принятый за повторы) входят в обе стороны и частично сокращаются; на шаге 2 они бьют по выбору напрямую.

Сравниваются только признаки, которые сеть восстанавливает надежно (docs/research/blind_fingerprint_net.md).
"""
from __future__ import annotations

import numpy as np

from app.eval.blind_fingerprint_net import KEYS, SECONDS, SR, predict, target_vector

# Признаки, которые сеть восстанавливает на отложенных эффектах со Spearman ≥ ~0.6
RELIABLE = ("rt60_125", "rt60_250", "rt60_500", "rt60_1000", "rt60_2000", "rt60_4000", "rt60_8000",
            "predelay_ms", "drr_db", "side_mid_db", "iacc_late")
# Семь полос RT60 вместе весят как два признака, иначе длина хвоста забивает баланс сухой/мокрый и ширину
WEIGHTS = np.array([2.0 / 7] * 7 + [1.0] * 4)
_IDX = [KEYS.index(k) for k in RELIABLE]
# Больше окон на описание не берется: на длинном треке шаг растет. 30 с при шаге 1 с — это 23 окна, так что
# короткие фрагменты (бенчмарк Cambridge) описываются как раньше, а 80 с — в три раза быстрее.
MAX_WINDOWS = 24


def blind_estimate(model, wet: np.ndarray, hop_s: float = 1.0, max_windows: int = MAX_WINDOWS) -> dict:
    """
    Описание мокрого сигнала (2, n) при SR: медиана предсказаний по окнам 8 с с шагом hop_s (шаг растет, если
    окон получается больше max_windows).
    """
    n = int(SECONDS * SR)
    if wet.shape[1] < n:
        wet = np.pad(wet, ((0, 0), (0, n - wet.shape[1])))
    hop = max(1, int(hop_s * SR), int(np.ceil((wet.shape[1] - n) / max(1, max_windows - 1))))
    starts = range(0, wet.shape[1] - n + 1, hop)
    preds = [predict(model, wet[:, s:s + n], SR) for s in starts]
    return {k: float(np.median([p[k] for p in preds])) for k in preds[0]}


def description_distance(a: dict, b: dict) -> float:
    """Взвешенное RMS по надежным признакам; признак, не определенный хотя бы у одного, не учитывается."""
    va, ma = target_vector(a)
    vb, mb = target_vector(b)
    w = WEIGHTS * ma[_IDX] * mb[_IDX]
    return float(np.sqrt(np.sum(w * (va[_IDX] - vb[_IDX]) ** 2) / max(w.sum(), 1e-9)))


def preselect(fingerprints: list[dict], target: dict, k: int) -> list[int]:
    """Индексы k отпечатков, ближайших к описанию target."""
    d = [description_distance(fp, target) for fp in fingerprints]
    return [int(i) for i in np.argsort(d)[:k]]


def symmetric_pick(model, renders: list[np.ndarray], target: dict) -> tuple[int, list[float], list[dict]]:
    """(лучший индекс, расстояния, описания рендеров): описание каждого рендера сравнивается с описанием референса."""
    descs = [blind_estimate(model, r) for r in renders]
    best, dists = pick_by_descriptions(descs, target)
    return best, dists, descs


def pick_by_descriptions(descs: list[dict], target: dict) -> tuple[int, list[float]]:
    """Двусторонняя проверка по готовым описаниям рендеров: (лучший индекс, расстояния до описания референса)."""
    dists = [description_distance(d, target) for d in descs]
    return int(np.argmin(dists)), dists
