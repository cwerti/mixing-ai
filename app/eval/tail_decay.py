"""
Хвост пространства после конца фразы: самый прямой признак длины и громкости ревера. Слепая сеть смотрит на окна
внутри фраз, где хвост замаскирован голосом, и может недооценить его (на треке пользователя RT60 0.6-0.7 с вместо
~1.2 с). Тогда выбор уходит в короткий «амбиенс» без хвоста: ошибка профиля 30 дБ против 3-5 дБ у ревербов,
которые и по log-mel к сведению ближе всего.

Конец фразы ищется по самому сигналу (сухой оригинал не нужен): пауза — уровень Mid ниже пика фразы на PAUSE_DB
не меньше PAUSE_S; конец фразы — последний кадр перед паузой, где уровень еще в пределах END_DB от уровня фразы.
Референс и пробный рендер кандидата меряются одинаково, поэтому сдвиг найденной точки частично сокращается.

Профиль — уровни Mid и Side через POINTS_S после конца фразы относительно уровня фразы (медиана по концам фраз).
"""
from __future__ import annotations

import numpy as np

HOP_S = 0.01
WIN_S = 0.05
PHRASE_S = 1.0  # уровень фразы — медиана громких кадров за столько секунд до конца
PAUSE_DB = 50.0
PAUSE_S = 1.2
END_DB = 6.0
POINTS_S = (0.1, 0.3, 0.6, 1.0)
FLOOR_DB = -70.0
# Ошибка профиля в единицах описания. Бенчмарк Cambridge (21 пара с паузами в референсе): без хвоста log-mel 9.10,
# с весом 6/4/3/1 дБ — 8.74/8.73/8.75/8.74 (плато), лучше сухого 19/19/20/20 из 21; хвост ранжирует кандидатов по
# log-mel лучше описания сети (медиана Spearman 0.8 против 0.6).
TAIL_UNIT_DB = 3.0


def _envelope_db(x: np.ndarray, sr: int) -> np.ndarray:
    hop, win = int(HOP_S * sr), int(WIN_S * sr)
    n = max(0, (len(x) - win) // hop + 1)
    if n == 0:
        return np.zeros(0)
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop][:n]
    return 10.0 * np.log10(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-20)


def phrase_ends(y: np.ndarray, sr: int) -> list[tuple[int, float]]:
    """
    Концы фраз перед паузами: (кадр конца, уровень фразы в дБ) по огибающей Mid с шагом HOP_S. Пауза — не меньше
    PAUSE_S тише пика всего сигнала на PAUSE_DB (у выделенного вокала между фразами тишина или слабый призвук);
    конец фразы — последний кадр перед паузой не тише уровня фразы на END_DB.
    """
    mid = np.mean(y, axis=0) if y.ndim == 2 else y
    env = _envelope_db(mid, sr)
    if not len(env):
        return []
    quiet = env < env.max() - PAUSE_DB
    phrase_frames, pause_frames = int(PHRASE_S / HOP_S), int(PAUSE_S / HOP_S)
    ends = []
    k = 0
    while k < len(env):
        if not quiet[k] or (k > 0 and quiet[k - 1]):
            k += 1
            continue
        run = k
        while run < len(env) and quiet[run]:
            run += 1
        if run - k >= pause_frames and k > phrase_frames:
            loud = np.flatnonzero(~quiet[:k])
            body = env[loud[-phrase_frames:]] if len(loud) else np.zeros(0)
            if len(body) >= phrase_frames // 2:
                level = float(np.percentile(body, 90))
                above = np.flatnonzero(env[:k] >= level - END_DB)
                end = int(above[-1]) if len(above) else -1
                if end >= 0 and k - end <= int((max(POINTS_S) + 1.0) / HOP_S):
                    ends.append((end, level))
        k = run
    return ends


def decay_profile(y: np.ndarray, sr: int) -> np.ndarray | None:
    """Профиль хвоста (Mid по POINTS_S, затем Side) в дБ относительно уровня фразы; None — пауз после фраз нет."""
    if y.ndim != 2:
        y = np.vstack([y, y])
    mid, side = np.mean(y, axis=0), (y[0] - y[1]) / 2
    env_mid, env_side = _envelope_db(mid, sr), _envelope_db(side, sr)
    profiles = []
    for end, level in phrase_ends(y, sr):
        idx = [end + int(round(t / HOP_S)) for t in POINTS_S]
        values = [env[k] - level for env in (env_mid, env_side) for k in idx]
        profiles.append(np.maximum(values, FLOOR_DB))
    return np.median(profiles, axis=0) if profiles else None


def profile_at(y: np.ndarray, sr: int, end: int, level: float | None = None) -> np.ndarray:
    """
    Профиль хвоста от известного кадра конца фразы end (проба кандидата: конец найден по сухой пробе, у рендера с
    длинным хвостом пауза может так и не наступить). level — уровень фразы; None — по самому сигналу.
    """
    if y.ndim != 2:
        y = np.vstack([y, y])
    env_mid, env_side = _envelope_db(np.mean(y, axis=0), sr), _envelope_db((y[0] - y[1]) / 2, sr)
    if level is None:
        level = float(np.percentile(env_mid[max(0, end - int(PHRASE_S / HOP_S)):end + 1], 90))
    idx = [min(len(env_mid) - 1, end + int(round(t / HOP_S))) for t in POINTS_S]
    return np.maximum([env[k] - level for env in (env_mid, env_side) for k in idx], FLOOR_DB)


def profile_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Средняя абсолютная разница профилей в дБ."""
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def probe_segment(y: np.ndarray, sr: int, before_s: float = 3.0, after_s: float = 2.5) -> np.ndarray | None:
    """Кусок сигнала вокруг первого конца фразы перед паузой — проба хвоста для рендера кандидатов."""
    ends = phrase_ends(y, sr)
    if not ends:
        return None
    center = int(ends[0][0] * HOP_S * sr)
    a, b = max(0, center - int(before_s * sr)), min(y.shape[-1], center + int(after_s * sr))
    return y[..., a:b]
