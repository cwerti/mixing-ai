"""
Подбор цепочки сведения по референсу методом analysis-by-synthesis.

Нейросеть дает начальное приближение, а окончательные параметры подбираются по обратной связи:
фрагмент исходника рендерится текущей цепочкой, меряется расстояние по стилю до референса
(app/eval/style_metrics.py), и CMA-ES крутит ручки, пока расстояние не перестанет падать.
Решение «включать ли плагин» принимают правила по разнице признаков, а не регрессия: включенный
плагин в DATAGEN_PLUGINS всегда слышен, поэтому лишний плагин нельзя «выкрутить в ноль».
"""
import time
from dataclasses import dataclass, field
from typing import Callable

import cma
import numpy as np
import pyloudnorm as pyln

from app.audio.audio_processor import SpectralMatcher
from app.audio.dsp_engine import DSPEngine
from app.audio.vocal_enhancer import VocalEnhancer
from app.dataset.dataset_schema import (
    CHAIN_ORDER, DATAGEN_PLUGINS, PARAM_LAYOUT, PLUGIN_ON_PROB, RENDER_ORDER, ChainConfig, PluginConfig
)
from app.eval.style_metrics import DEFAULT_WEIGHTS, extract_style_features, style_distance

# Уровень, на котором сухой вокал подается в цепочку при генерации датасета (пороги плагинов рассчитаны на него)
CHAIN_INPUT_LUFS = -23.0

# Направление «интенсивности» параметра: +1 — больше значение = сильнее эффект, -1 — наоборот.
# Используется для L1-регуляризации: при равном расстоянии по стилю выигрывает более мягкая обработка.
PARAM_INTENSITY = {
    ("compressor", "threshold_db"): -1, ("compressor", "ratio"): 1,
    ("deesser", "threshold_db"): -1, ("deesser", "ratio"): 1,
    ("multiband_compressor", "depth"): 1,
    ("distortion", "drive_db"): 1,
    ("chorus", "depth"): 1, ("chorus", "mix"): 1, ("chorus", "feedback"): 1,
    ("stereo_enhancer", "width"): 1,
    ("reverb", "wet_level"): 1, ("reverb", "room_size"): 1,
    ("delay", "mix"): 1, ("delay", "feedback"): 1,
    ("exciter", "mix"): 1,
}

# Шаг CMA-ES без подсказки сети и границы шага по уверенности сети (ошибка параметра на валидации)
DEFAULT_SIGMA = 0.2
PRIOR_SIGMA_RANGE = (0.05, DEFAULT_SIGMA)

# Параметры, которых метрика стиля не слышит (в ней нет признаков модуляции): CMA-ES крутит их до упора ради
# мелких сдвигов спектра, и хорус начинает «плавать». Они не подбираются, а берутся у модели или по умолчанию.
UNMEASURED_PARAMS = {("chorus", "rate_hz"), ("chorus", "depth"), ("chorus", "feedback")}
# Плагины, в которых модель уверена настолько, что подбор их не выключает (компрессию метрика видит плохо)
MODEL_KEEP_PROB = 0.9

# Метрики целевой функции CMA-ES: style — ручная style_distance (по умолчанию), embedding — обученный
# кодировщик стиля обработки (app/eval/style_embedding.py), combined — их сумма в общих единицах
METRICS = ("style", "embedding", "combined")

# Плагины, которые подбирает оптимизатор (EQ считается аналитически, автотюн — отдельно вне цикла)
SEARCHABLE_PLUGINS = [name for name in CHAIN_ORDER if name in PARAM_LAYOUT and name != "pitch_corrector"]

# Параметры, которые подбирает только оптимизатор (модель их не предсказывает), со стартовыми нормализованными
# значениями. Старт нейтральный (без predelay и фильтров, звучит как прежний Reverb): predelay и срезы
# (по роликам low cut 400–700 Гц, темный хвост) поиск находит сам, а старт с ними проигрывал полному реверу.
OPTIMIZER_EXTRA_PARAMS = {
    "reverb": {"predelay_ms": 0.0, "low_cut_hz": 0.0, "high_cut_hz": 1.0},
}


def _search_params(plugin: str) -> list[str]:
    return PARAM_LAYOUT[plugin] + list(OPTIMIZER_EXTRA_PARAMS.get(plugin, {}))


@dataclass
class OptimizationResult:
    """Итог подбора цепочки."""
    chain: ChainConfig
    eq_bands: list[tuple[float, float]]
    pitch_speed: float | None
    distance_source: float
    distance_prior: float | None
    distance_final: float
    report: list[str] = field(default_factory=list)
    # Сходимость CMA-ES: (число рендеров, лучшая целевая функция на этот момент) после каждого поколения
    history: list[tuple[int, float]] = field(default_factory=list)


def normalize_lufs(y: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    """Нормализует громкость сигнала (1D или (channels, samples)) к целевому LUFS."""
    meter = pyln.Meter(sr)
    data = y.T if y.ndim == 2 else y
    loudness = meter.integrated_loudness(data)
    if not np.isfinite(loudness):
        return y
    normalized = pyln.normalize.loudness(data, loudness, target_lufs)
    return (normalized.T if y.ndim == 2 else normalized).astype(np.float32)


def to_stereo(y: np.ndarray) -> np.ndarray:
    """Приводит сигнал к форме (2, samples)."""
    if y.ndim == 1:
        return np.vstack([y, y]).astype(np.float32)
    return (y if y.shape[0] == 2 else y.T).astype(np.float32)


def select_excerpt(y: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    """Выбирает самый громкий фрагмент заданной длины (шаг поиска 0.5 с)."""
    n = int(seconds * sr)
    if y.shape[-1] <= n:
        return y
    energy = np.cumsum(np.concatenate([[0.0], np.mean(y, axis=0) ** 2]))
    starts = np.arange(0, y.shape[-1] - n, sr // 2)
    best = int(starts[np.argmax(energy[starts + n] - energy[starts])])
    return y[:, best:best + n]


def physical_to_normalized(plugin: str, param: str, value: float) -> float:
    """Переводит физическое значение параметра в нормализованное [0..1] по DATAGEN_PLUGINS."""
    low, high = DATAGEN_PLUGINS[plugin]["params"][param]
    return float(np.clip((value - low) / (high - low), 0.0, 1.0)) if high != low else 0.0


def eq_plugin(eq_bands: list[tuple[float, float]]) -> PluginConfig:
    """Собирает PluginConfig эквалайзера из списка (частота, гейн)."""
    params = {}
    for i, (freq, gain) in enumerate(eq_bands, start=1):
        params[f"band{i}_freq_hz"] = physical_to_normalized("eq", f"band{i}_freq_hz", freq)
        params[f"band{i}_gain_db"] = physical_to_normalized("eq", f"band{i}_gain_db", gain)
    return PluginConfig(name="eq", params=params)


def select_plugins(src: dict, ref: dict, model_probs: dict[str, float] | None) -> tuple[list[str], list[str]]:
    """
    Решает, какие плагины нужны, по разнице признаков исходника и референса.
    Вероятности модели используются для эффектов, которые плохо измеряются напрямую (хорус, дисторшн, экситер).

    Returns:
        tuple: (плагины в порядке CHAIN_ORDER, человекочитаемые причины).
    """
    probs = model_probs or {}
    chosen, reasons = set(), []

    decay_gain = ref["decay_fast_db"] - src["decay_fast_db"]
    if decay_gain > 3.0:
        chosen.add("reverb")
        reasons.append(f"reverb: спад у референса медленнее на {decay_gain:.1f} дБ")
    if ref["delay_peak"] - src["delay_peak"] > 0.05 and decay_gain > 1.5:
        chosen.add("delay")
        reasons.append(f"delay: повторы в огибающей {src['delay_peak']:.2f} -> {ref['delay_peak']:.2f}")

    width_gain = ref["width_global"] - src["width_global"]
    notes_width_gain = ref["width_notes_db"] - src["width_notes_db"]
    if width_gain > 0.05 and "reverb" not in chosen:
        chosen.add("stereo_enhancer")
        reasons.append(f"stereo_enhancer: ширина {src['width_global']:.2f} -> {ref['width_global']:.2f} без хвостов")
    elif notes_width_gain > 3.0:
        # Реверб расширяет в основном паузы (хвост), а широкий голос на нотах дает только расширитель
        chosen.add("stereo_enhancer")
        reasons.append(f"stereo_enhancer: ширина на нотах {src['width_notes_db']:.1f} -> "
                       f"{ref['width_notes_db']:.1f} дБ Side/Mid")

    dynamics_cut = src["dr_p95_p10_db"] - ref["dr_p95_p10_db"]
    crest_cut = src["crest_db"] - ref["crest_db"]
    if dynamics_cut > 3.0 or crest_cut > 2.0:
        chosen.add("compressor")
        reasons.append(f"compressor: разброс громкости -{dynamics_cut:.1f} дБ, пик-фактор -{crest_cut:.1f} дБ")
    if dynamics_cut > 8.0:
        chosen.add("multiband_compressor")
        reasons.append("multiband_compressor: динамика референса сжата очень сильно")

    sibilance_cut = src["sib_p90_db"] - ref["sib_p90_db"]
    if sibilance_cut > 3.0:
        chosen.add("deesser")
        reasons.append(f"deesser: сибилянты тише на {sibilance_cut:.1f} дБ")

    for name in ("chorus", "distortion", "exciter"):
        if probs.get(name, 0.0) > 0.8:
            chosen.add(name)
            reasons.append(f"{name}: уверенность модели {probs[name]:.2f}")

    return [name for name in SEARCHABLE_PLUGINS if name in chosen], reasons


def _uses_prior(plugin: str, model_probs: dict[str, float] | None,
                model_params: dict[str, dict[str, float]] | None) -> bool:
    """
    Можно ли стартовать с параметров модели. Регрессия учится только на активных плагинах, поэтому параметры
    плагина, который сама модель считает выключенным, — шум, а не подсказка.
    """
    if not model_params or plugin not in model_params:
        return False
    return model_probs is None or model_probs.get(plugin, 0.0) > PLUGIN_ON_PROB


def _default_params(plugin: str) -> dict[str, float]:
    """Мягкая середина диапазона: умеренная обработка без подсказки модели."""
    start = {}
    for param in PARAM_LAYOUT[plugin]:
        direction = PARAM_INTENSITY.get((plugin, param), 0)
        start[param] = 0.25 if direction > 0 else 0.75 if direction < 0 else 0.5
    return {**start, **OPTIMIZER_EXTRA_PARAMS.get(plugin, {})}


def _initial_params(plugin: str, model_probs: dict[str, float] | None,
                    model_params: dict[str, dict[str, float]] | None) -> dict[str, float]:
    """Стартовые нормализованные параметры: предсказание модели или мягкая середина диапазона."""
    if _uses_prior(plugin, model_probs, model_params):
        return {**OPTIMIZER_EXTRA_PARAMS.get(plugin, {}),
                **{k: float(np.clip(v, 0.0, 1.0)) for k, v in model_params[plugin].items()}}
    return _default_params(plugin)


def search_stds(layout: list[tuple[str, str]], prior_plugins: set[str],
                model_param_std: dict[str, dict[str, float]] | None) -> list[float]:
    """
    Начальный шаг CMA-ES по каждой координате: у параметров со стартом от модели — ее типичная ошибка
    на валидации (уверенная сеть сужает поиск), у остальных — DEFAULT_SIGMA.
    """
    stds = []
    for plugin, param in layout:
        std = (model_param_std or {}).get(plugin, {}).get(param) if plugin in prior_plugins else None
        stds.append(DEFAULT_SIGMA if std is None else float(np.clip(std, *PRIOR_SIGMA_RANGE)))
    return stds


def _stagnated(history: list[tuple[int, float]], patience_renders: int, min_improvement: float) -> bool:
    """True, если за patience_renders последних рендеров целевая функция улучшилась меньше min_improvement."""
    renders, best = history[-1]
    if renders < patience_renders:
        return False
    before = next(value for n, value in reversed(history) if n <= renders - patience_renders)
    return before - best < min_improvement


class ChainOptimizer:
    """Подбирает плагины и параметры цепочки, минимизируя расстояние по стилю до референса."""

    def __init__(self, sr: int = 44100, weights: dict | None = None, intensity_penalty: float = 0.15,
                 log: Callable[[str], None] = print, metric: str = "style", encoder=None):
        """
        Args:
            metric: Целевая функция подбора, одна из METRICS. Для embedding/combined нужен обученный
                кодировщик (encoder или чекпоинт по умолчанию data/models/style_encoder/style_encoder.pt).
            encoder: app.eval.style_embedding.StyleEncoder; None — загрузить чекпоинт по умолчанию.
        """
        if metric not in METRICS:
            raise ValueError(f"Неизвестная метрика {metric!r}, допустимы {METRICS}")
        self.metric = metric
        self.encoder = encoder
        if metric != "style" and encoder is None:
            from app.eval.style_embedding import load_encoder
            self.encoder = load_encoder()
        self.sr = sr
        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        self.intensity_penalty = intensity_penalty
        self.log = log
        self.dsp = DSPEngine()
        self.enhancer = VocalEnhancer(sr=sr)
        self.matcher = SpectralMatcher(sr=sr)

    def _render(self, base: np.ndarray, eq_bands, plugins: dict[str, dict[str, float]],
                gate_threshold_db: float | None, fixed: tuple[PluginConfig, ...] = ()) -> np.ndarray:
        chain = ChainConfig(plugins=[eq_plugin(eq_bands)] if eq_bands else [])
        chain.plugins += [PluginConfig(name=name, params=params) for name, params in plugins.items()]
        chain.plugins += list(fixed)
        chain.plugins.sort(key=lambda plugin: RENDER_ORDER.index(plugin.name))
        return self.dsp.apply_chain(base, self.sr, chain, gate_threshold_db=gate_threshold_db)

    def _reference_target(self, y_ref: np.ndarray, ref_features: dict) -> dict:
        """Что нужно метрике от референса: признаки style_distance и (для embedding/combined) эмбеддинг."""
        target = {"features": ref_features}
        if self.metric != "style":
            target["embedding"] = self.encoder.embed(y_ref, self.sr)
        return target

    def _distance(self, y: np.ndarray, target: dict) -> float:
        if self.metric == "style":
            return style_distance(extract_style_features(y, self.sr, with_pitch=False), target["features"],
                                  self.weights)[0]
        from app.eval.style_embedding import EMBEDDING_SCALE, combined_distance
        emb = self.encoder.distance(self.encoder.embed(y, self.sr), target["embedding"])
        if self.metric == "embedding":
            return EMBEDDING_SCALE * emb
        style = style_distance(extract_style_features(y, self.sr, with_pitch=False), target["features"],
                               self.weights)[0]
        return combined_distance(style, emb)

    def _intensity(self, plugins: dict[str, dict[str, float]]) -> float:
        values = [v if PARAM_INTENSITY[(p, k)] > 0 else 1.0 - v
                  for p, params in plugins.items() for k, v in params.items() if (p, k) in PARAM_INTENSITY]
        return float(np.mean(values)) if values else 0.0

    def _choose_pitch_speed(self, excerpt: np.ndarray, src_cents: float, ref_cents: float,
                            key_index: int, scale: str) -> float | None:
        """Подбирает speed автотюна, при котором отклонение от нот совпадает с референсом (вне цикла CMA-ES)."""
        if not (np.isfinite(src_cents) and np.isfinite(ref_cents)) or ref_cents > src_cents - 4.0:
            return None
        best_speed, best_gap = None, abs(src_cents - ref_cents)
        for speed in (0.0, 0.33, 0.67, 1.0):
            phys = DATAGEN_PLUGINS["pitch_corrector"]["params"]["speed"]
            tuned = self.enhancer.apply_autotune(excerpt, key_index=key_index, scale=scale,
                                                 speed=phys[0] + speed * (phys[1] - phys[0]))
            cents = extract_style_features(tuned, self.sr)["pitch_cents"]
            self.log(f"    автотюн speed={speed:.2f}: {cents:.1f} центов (референс {ref_cents:.1f})")
            if np.isfinite(cents) and abs(cents - ref_cents) < best_gap:
                best_speed, best_gap = speed, abs(cents - ref_cents)
        return best_speed

    def optimize(self, y_src: np.ndarray, y_ref: np.ndarray, *, model_probs: dict[str, float] | None = None,
                 model_params: dict[str, dict[str, float]] | None = None,
                 model_param_std: dict[str, dict[str, float]] | None = None, key_index: int = 0,
                 scale: str = "major", gate_threshold_db: float | None = None, budget_s: float = 60.0,
                 excerpt_s: float = 12.0, seed: int = 0, patience_renders: int | None = None,
                 min_improvement: float = 0.01, allow_tune: bool = True,
                 tune_speed: float | None = None,
                 fixed_plugins: list[PluginConfig] | None = None) -> OptimizationResult:
        """
        Подбирает цепочку для исходника y_src под референс y_ref.

        Args:
            y_src: Сухой вокал (1D или стерео) на любом уровне громкости.
            y_ref: Референсный вокал.
            model_probs: Вероятности плагинов от MixingAIModel (по имени).
            model_params: Нормализованные параметры плагинов от MixingAIModel (стартовая точка поиска).
            model_param_std: Ошибка параметров модели на валидации (начальный шаг поиска вокруг ее старта).
            key_index, scale: Тональность референса для автотюна.
            gate_threshold_db: Порог Noise Gate для сигнала на уровне CHAIN_INPUT_LUFS.
            budget_s: Ограничение времени на CMA-ES.
            excerpt_s: Длина фрагмента исходника для рендеров в цикле.
            patience_renders: Остановить CMA-ES, если за столько рендеров целевая функция улучшилась меньше,
                чем на min_improvement (None — работать до бюджета или сходимости).
            allow_tune: Разрешить автотюн. PSOLA пересинтезирует сигнал и на переходах нот может давать
                ВЧ-артефакты, поэтому пользователь может его запретить.
            tune_speed: Нормализованная сила тюна [0..1] вместо подбора по отклонению от нот референса
                (на слух выбор может расходиться с метрикой).
            fixed_plugins: Плагины, заданные цепочкой/архетипом явно (например, harmony): рендерятся во всех
                пробах и попадают в итоговую цепочку как есть, но не подбираются и не выкидываются упрощением.
        """
        report = []
        fixed = tuple(fixed_plugins or ())
        clashing = [p.name for p in fixed if p.name in SEARCHABLE_PLUGINS or p.name in ("eq", "pitch_corrector")]
        if clashing:
            raise ValueError(f"fixed_plugins не должны пересекаться с подбираемыми плагинами: {clashing}")

        def note(message: str):
            report.append(message)
            self.log(message)

        src = to_stereo(normalize_lufs(y_src, self.sr, CHAIN_INPUT_LUFS))
        ref_features = extract_style_features(y_ref, self.sr)
        src_features = extract_style_features(src, self.sr)
        target = self._reference_target(y_ref, ref_features)

        excerpt = select_excerpt(src, self.sr, excerpt_s)
        excerpt = np.vstack([self.enhancer.suppress_resonances(ch) for ch in excerpt]).astype(np.float32)
        distance_source = self._distance(excerpt, target)

        plugins, reasons = select_plugins(src_features, ref_features, model_probs)
        note("[*] Выбор плагинов по разнице признаков:")
        for reason in reasons or ["ничего, кроме EQ, не требуется"]:
            note(f"    {reason}")

        pitch_speed = None
        if allow_tune and tune_speed is not None:
            pitch_speed = float(np.clip(tune_speed, 0.0, 1.0))
        elif allow_tune:
            pitch_speed = self._choose_pitch_speed(excerpt, src_features["pitch_cents"], ref_features["pitch_cents"],
                                                   key_index, scale)
        if pitch_speed is not None:
            phys = DATAGEN_PLUGINS["pitch_corrector"]["params"]["speed"]
            excerpt = self.enhancer.apply_autotune(excerpt, key_index=key_index, scale=scale,
                                                   speed=phys[0] + pitch_speed * (phys[1] - phys[0]))
            note(f"[+] Автотюн: speed={pitch_speed:.2f} (нормализованный)")

        eq_bands = self.matcher.match_eq(src, y_ref)

        distance_prior = None
        if model_probs and model_params:
            prior_plugins = {p: model_params[p] for p in SEARCHABLE_PLUGINS
                             if model_probs.get(p, 0.0) > PLUGIN_ON_PROB and p in model_params}
            distance_prior = self._distance(self._render(excerpt, eq_bands, prior_plugins, gate_threshold_db,
                                                         fixed), target)
            note(f"[*] Цепочка модели {list(prior_plugins)}: расстояние {distance_prior:.2f}")

        kept = {p for p in SEARCHABLE_PLUGINS if (model_probs or {}).get(p, 0.0) > MODEL_KEEP_PROB
                and _uses_prior(p, model_probs, model_params)}
        if kept - set(plugins):
            note(f"[*] По уверенности модели добавлены: {sorted(kept - set(plugins))}")
        plugins = [p for p in SEARCHABLE_PLUGINS if p in set(plugins) | kept]
        prior_plugins = {p for p in plugins if _uses_prior(p, model_probs, model_params)}
        current = {p: _initial_params(p, model_probs, model_params) for p in plugins}
        layout = [(p, k) for p in plugins for k in _search_params(p) if (p, k) not in UNMEASURED_PARAMS]

        def unpack(x) -> dict[str, dict[str, float]]:
            result = {p: {k: v for k, v in current[p].items() if (p, k) in UNMEASURED_PARAMS} for p in plugins}
            for (p, k), value in zip(layout, x):
                result[p][k] = float(value)
            return result

        def objective(params: dict[str, dict[str, float]]) -> float:
            y = self._render(excerpt, eq_bands, params, gate_threshold_db, fixed)
            return self._distance(y, target) + self.intensity_penalty * self._intensity(params)

        best_score = objective(current)
        history = [(1, best_score)]
        if prior_plugins:
            # Страховка от промаха модели: если мягкий старт без нее лучше, поиск идет от него с широким шагом
            fallback = {p: {**_default_params(p), **{k: v for k, v in current[p].items()
                                                     if (p, k) in UNMEASURED_PARAMS}} for p in plugins}
            fallback_score = objective(fallback)
            history.append((2, min(best_score, fallback_score)))
            if fallback_score < best_score:
                note(f"[*] Старт модели хуже мягкого ({best_score:.2f} против {fallback_score:.2f}), он отброшен")
                current, best_score, prior_plugins = fallback, fallback_score, set()
        if layout:
            x0 = [current[p][k] for p, k in layout]
            stds = search_stds(layout, prior_plugins, model_param_std)
            options = {"bounds": [0.0, 1.0], "seed": seed + 1, "verbose": -9, "maxfevals": 2000,
                       "CMA_stds": [std / DEFAULT_SIGMA for std in stds]}
            if prior_plugins:
                note(f"[*] Старт от модели для {sorted(prior_plugins)}, шаг поиска {np.mean(stds):.3f}")
            es = cma.CMAEvolutionStrategy(x0, DEFAULT_SIGMA, options)
            started = time.perf_counter()
            evaluations = history[-1][0]
            while not es.stop() and time.perf_counter() - started < budget_s:
                candidates = es.ask()
                scores = [objective(unpack(x)) for x in candidates]
                es.tell(candidates, scores)
                evaluations += len(candidates)
                history.append((evaluations, min(history[-1][1], es.result.fbest)))
                if patience_renders and _stagnated(history, patience_renders, min_improvement):
                    note(f"[*] CMA-ES остановлен: за {patience_renders} рендеров улучшение меньше {min_improvement}")
                    break
            if es.result.fbest < best_score:
                best_score = es.result.fbest
                current = unpack(es.result.xbest)
            note(f"[*] CMA-ES: {evaluations} рендеров за {time.perf_counter() - started:.0f} с, "
                 f"целевая функция {best_score:.2f}")

        # Упрощение: убираем плагины, без которых результат не хуже (лишняя обработка = риск артефактов)
        for name in [p for p in current if p not in kept]:
            reduced = {p: v for p, v in current.items() if p != name}
            score = objective(reduced)
            if score <= best_score + 0.02:
                note(f"[-] {name} убран: без него {score:.2f} против {best_score:.2f}")
                current, best_score = reduced, min(best_score, score)

        # EQ пересчитывается после остальных эффектов (реверб, хорус и экситер меняют спектр) и по всему треку:
        # самый громкий фрагмент по тембру отличается от трека в целом, и EQ по нему переобучается
        # Рендер идет настоящей финальной цепочкой (EQ стоит до автотюна, а PSOLA пересинтезирует область
        # основного тона), поэтому EQ уточняется итеративно по остатку разницы спектров
        def build_chain(bands) -> ChainConfig:
            chain = ChainConfig(plugins=[eq_plugin(bands)])
            if pitch_speed is not None:
                chain.plugins.append(PluginConfig(name="pitch_corrector", params={"speed": pitch_speed}))
            chain.plugins += [PluginConfig(name=name, params=current[name]) for name in current]
            chain.plugins += list(fixed)
            chain.plugins.sort(key=lambda plugin: RENDER_ORDER.index(plugin.name))
            return chain

        full_base = np.vstack([self.enhancer.suppress_resonances(ch) for ch in src]).astype(np.float32)
        ref_spectrum = self.matcher.long_term_spectrum_db(y_ref)
        eq_bands = [(f, 0.0) for f in SpectralMatcher.BAND_CENTERS_HZ]
        for _ in range(3):
            rendered = self.dsp.apply_chain(full_base, self.sr, build_chain(eq_bands),
                                            gate_threshold_db=gate_threshold_db,
                                            target_key_index=key_index, target_scale=scale)
            residual = ref_spectrum - self.matcher.long_term_spectrum_db(rendered)
            eq_bands = self.matcher.fit_eq_to_delta(residual + self.matcher.eq_curve_db(eq_bands))

        chain = build_chain(eq_bands)
        distance_final = self._distance(self._render(excerpt, eq_bands, current, gate_threshold_db, fixed),
                                        target)

        note(f"[+] Расстояние до референса: исходник {distance_source:.2f}"
             + (f", модель {distance_prior:.2f}" if distance_prior is not None else "")
             + f", подбор {distance_final:.2f}")
        return OptimizationResult(chain, eq_bands, pitch_speed, distance_source, distance_prior,
                                  distance_final, report, history)
