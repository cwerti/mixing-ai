"""
Доменная рандомизация: альтернативные реализации эффектов палитры — ТОЛЬКО для генерации данных
(датасет модели и банк кодировщика стиля). Путь рендера оптимизатора (DSPEngine.apply_chain) не меняется:
он должен оставаться детерминированным и быстрым.

Зачем: модель и кодировщик стиля учились только на нашем DSP (Freeverb, pedalboard Compressor, tanh-дисторшн),
а инженеры работают другими плагинами (Valhalla, FabFilter, стоковые FL). На реальных парах модель не видит
реверб Whiptails и путает «что сделано» с «каким алгоритмом». Здесь каждый плагин цепочки с вероятностью
1 − own_prob рендерится случайной другой реализацией того же эффекта, а метка остается прежней (плагин схемы +
нормализованные параметры, соответствие параметров приблизительное — смысл в разнообразии).

Реализации (VARIANTS): реверб — Freeverb/RT60-свёртка как send с другими фильтрами, свёртка с реальной или
синтетической алгоритмической ИХ (банк data/raw/irs/_cache, scripts/build_ir_cache.py); компрессор —
feed-forward RMS/пик с мягким коленом, feedback «опто» с программно-зависимым release; сатурация — tanh, ламповая
асимметрия, лента с потерей верха, жесткий клип, биткраш; хорус — многоголосый стерео, флэнжер с обратной
связью, «dimension»; дилей — пинг-понг, фильтрованная обратная связь (лента), разные времена L/R; EQ — другие
добротности и линейно-фазовая кривая; де-эссер, OTT, стерео и экситер — по 2–3 варианта. Установленные VST3
(scripts/vst_inventory.py → data/raw/vst_inventory.json) добавляются как еще одна реализация своей категории.
Тюн, слои harmony и подавление резонансов всегда рендерятся нашим DSPEngine.
"""
from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
from numba import njit
from pedalboard import (Bitcrush, Chorus, Clipping, Compressor, Delay, Gain, HighpassFilter, HighShelfFilter,
                        Limiter, LowpassFilter, LowShelfFilter, PeakFilter, Pedalboard, PitchShift, Reverb)
from scipy.signal import butter, fftconvolve, sosfilt

from app.audio.dsp_engine import DSPEngine, decay_reverb
from app.dataset.dataset_schema import ChainConfig, PluginConfig

IR_CACHE_DIR = Path("data/raw/irs/_cache")
VST_INVENTORY = Path("data/raw/vst_inventory.json")
# Доля плагинов, которые рендерятся нашей реализацией (как у оптимизатора и в прежних датасетах)
OWN_PROB = 0.35
# RT60 Freeverb (pedalboard Reverb) по room_size, замер по кривой Шрёдера при damping 0.5 (от damping почти
# не зависит): нужен, чтобы подобрать ИХ с похожей длиной хвоста
FREEVERB_RT60 = ((0.15, 0.6), (0.3, 0.73), (0.5, 1.04), (0.7, 1.66), (0.85, 2.89), (0.95, 5.4), (0.99, 8.05))
# Категория VST (scripts/vst_inventory.py) для плагина схемы
VST_CATEGORY = {"reverb": "reverb", "delay": "delay", "compressor": "compressor", "distortion": "saturation",
                "chorus": "modulation", "eq": "eq", "multiband_compressor": "multiband"}


def freeverb_rt60(room_size: float) -> float:
    rooms, rts = zip(*FREEVERB_RT60)
    return float(np.exp(np.interp(room_size, rooms, np.log(rts))))


def _sos(kind: str, freq: float, sr: int, order: int = 2) -> np.ndarray:
    return butter(order, float(np.clip(freq, 20.0, 0.45 * sr)), kind, fs=sr, output="sos")


def _rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y, dtype=np.float64))) + 1e-12)


def _match_rms(y: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return (y * (_rms(ref) / _rms(y))).astype(np.float32)


# ---------------------------------------------------------------------------------------------------------
# Динамика: общий вычислитель усиления (numba)
# ---------------------------------------------------------------------------------------------------------

@njit(cache=True)
def _gain_curve_db(level_db: float, thresh: float, ratio: float, knee: float, up_thresh: float, up_ratio: float,
                   up_max: float) -> float:
    """Статическая кривая (дБ усиления): нисходящая компрессия с мягким коленом + опционально восходящая."""
    over = level_db - thresh
    gain = 0.0
    slope = 1.0 - 1.0 / ratio
    if knee > 0.0 and abs(over) <= knee / 2.0:
        gain = -slope * (over + knee / 2.0) ** 2 / (2.0 * knee)
    elif over > knee / 2.0:
        gain = -slope * over
    if up_ratio > 1.0 and level_db < up_thresh and level_db > -70.0:
        gain += min(up_max, (up_thresh - level_db) * (1.0 - 1.0 / up_ratio))
    return gain


@njit(cache=True)
def _feedforward_gain(det_db: np.ndarray, thresh: float, ratio: float, knee: float, att: float, rel: float,
                      up_thresh: float, up_ratio: float, up_max: float) -> np.ndarray:
    """Сглаженное усиление (дБ) по уровню детектора: атака, когда ослабление растет, иначе release."""
    n = det_db.shape[0]
    out = np.empty(n)
    g = 0.0
    for t in range(n):
        target = _gain_curve_db(det_db[t], thresh, ratio, knee, up_thresh, up_ratio, up_max)
        coef = att if target < g else rel
        g = coef * g + (1.0 - coef) * target
        out[t] = g
    return out


@njit(cache=True)
def _feedback_opto(x: np.ndarray, thresh: float, ratio: float, att: float, rel_fast: float, rel_slow: float) -> np.ndarray:
    """
    Компрессор с обратной связью (детектор на выходе) и программно-зависимым release (опто/LA-2A-подобный):
    чем глубже и дольше ослабление, тем медленнее восстановление. x — (2, n), возвращает усиление (линейное).
    """
    n = x.shape[1]
    gain = np.empty(n)
    g_db = 0.0
    env = 0.0
    memory = 0.0
    for t in range(n):
        lin = 10.0 ** (g_db / 20.0)
        level = max(abs(x[0, t]), abs(x[1, t])) * lin
        env = max(level, 0.999 * env)
        level_db = 20.0 * np.log10(env + 1e-9)
        target = 0.0
        if level_db > thresh:
            target = -(level_db - thresh) * (1.0 - 1.0 / ratio)
        memory = 0.9995 * memory + 0.0005 * min(-g_db, 20.0)
        rel = rel_fast + (rel_slow - rel_fast) * min(memory / 6.0, 1.0)
        coef = att if target < g_db else rel
        g_db = coef * g_db + (1.0 - coef) * target
        gain[t] = 10.0 ** (g_db / 20.0)
    return gain


def _coef(ms: float, sr: int) -> float:
    return float(np.exp(-1.0 / max(ms * 1e-3 * sr, 1.0)))


def _detector_db(y: np.ndarray, sr: int, mode: str, window_ms: float) -> np.ndarray:
    """Уровень детектора (дБ) по стерео-связке: rms — скользящее среднее квадрата, peak — максимум |x|."""
    from scipy.signal import lfilter
    if mode == "rms":
        a = _coef(window_ms, sr)
        power = lfilter([1.0 - a], [1.0, -a], np.mean(np.square(y, dtype=np.float64), axis=0))
        return 10.0 * np.log10(np.maximum(power, 1e-12)) + 3.0  # +3 дБ: RMS синуса к пику
    return 20.0 * np.log10(np.max(np.abs(y), axis=0).astype(np.float64) + 1e-9)


def _makeup_db(thresh: float, ratio: float) -> float:
    # Та же автокомпенсация, что у компрессора DSPEngine
    return float(np.clip((-10.0 - thresh) * (1.0 - 1.0 / max(ratio, 1.0)) * 0.45, 0.0, 12.0))


def compress(y: np.ndarray, sr: int, thresh: float, ratio: float, attack_ms: float, release_ms: float,
             knee_db: float = 6.0, detector: str = "rms", window_ms: float = 10.0, makeup_db: float = 0.0,
             up_thresh: float = -200.0, up_ratio: float = 1.0, up_max: float = 0.0,
             sidechain: np.ndarray | None = None) -> np.ndarray:
    """Feed-forward компрессор со стерео-связкой (detector: rms | peak), опционально восходящий и с сайдчейном."""
    det = _detector_db(sidechain if sidechain is not None else y, sr, detector, window_ms)
    g_db = _feedforward_gain(det, thresh, ratio, knee_db, _coef(attack_ms, sr), _coef(release_ms, sr),
                             up_thresh, up_ratio, up_max)
    return (y * (10.0 ** ((g_db + makeup_db) / 20.0))[None, :]).astype(np.float32)


# ---------------------------------------------------------------------------------------------------------
# Модуляция: дробная задержка (векторно) и флэнжер с обратной связью (numba)
# ---------------------------------------------------------------------------------------------------------

def _mod_delay(x: np.ndarray, delay_samples: np.ndarray) -> np.ndarray:
    """x[n - d[n]] с линейной интерполяцией (d >= 0)."""
    n = np.arange(len(x), dtype=np.float64)
    return np.interp(n - delay_samples, n, x, left=0.0).astype(np.float32)


@njit(cache=True)
def _flanger_kernel(x: np.ndarray, delay: np.ndarray, feedback: float) -> np.ndarray:
    n = x.shape[0]
    size = int(delay.max()) + 4
    buf = np.zeros(size)
    out = np.zeros(n)
    w = 0
    for t in range(n):
        d = delay[t]
        pos = w - d
        i0 = int(np.floor(pos))
        frac = pos - i0
        a = buf[i0 % size]
        b = buf[(i0 + 1) % size]
        v = (1.0 - frac) * a + frac * b
        out[t] = v
        buf[w % size] = x[t] + feedback * v
        w += 1
    return out


# ---------------------------------------------------------------------------------------------------------
# Банк импульсных характеристик
# ---------------------------------------------------------------------------------------------------------

class IRBank:
    """
    ИХ из кэша scripts/build_ir_cache.py: index.json (метаданные) + irs.npy (все ИХ подряд, (2, N) float16,
    memmap — страницы общие для всех процессов-генераторов). Прямой звук ослаблен, энергия канала = 4 (как у
    decay_reverb), 44.1 кГц. Виды: external (скачанные библиотеки) и synthetic (наши алгоритмические).
    """

    def __init__(self, cache_dir: Path = IR_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        index = json.loads((self.cache_dir / "index.json").read_text(encoding="utf-8"))
        self.items: list[dict] = index["irs"]
        self.sr = int(index["sr"])
        self.data = np.load(self.cache_dir / "irs.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.items)

    def kinds(self) -> set[str]:
        return {it["kind"] for it in self.items}

    def get(self, i: int) -> np.ndarray:
        it = self.items[i]
        return np.asarray(self.data[:, it["offset"]:it["offset"] + it["length"]], dtype=np.float32)

    def sample(self, rng: random.Random, target_rt60: float | None = None, kind: str | None = None,
               max_ratio: float = 1.7) -> tuple[np.ndarray, dict]:
        """
        Случайная ИХ (kind: external | synthetic | None — любая), по возможности с RT60 в max_ratio раз от цели.
        Сначала равновероятно выбирается источник (библиотека или топология), потом ИХ в нем — иначе большая
        библиотека одного зала вытеснила бы остальные.
        """
        pool = [i for i, it in enumerate(self.items) if kind is None or it["kind"] == kind]
        if not pool:
            raise ValueError(f"В банке нет ИХ вида {kind}")
        if target_rt60 is not None:
            close = [i for i in pool if abs(np.log(self.items[i]["rt60"] / target_rt60)) <= np.log(max_ratio)]
            if len(close) < 3:
                close = sorted(pool, key=lambda i: abs(np.log(self.items[i]["rt60"] / target_rt60)))[:5]
            pool = close
        sources = sorted({self.items[i]["source"] for i in pool})
        source = rng.choice(sources)
        i = rng.choice([i for i in pool if self.items[i]["source"] == source])
        return self.get(i), self.items[i]


@lru_cache(maxsize=2)
def default_ir_bank(cache_dir: str = str(IR_CACHE_DIR)) -> IRBank | None:
    """Банк ИХ процесса (None, если кэш еще не построен)."""
    path = Path(cache_dir)
    if not (path / "index.json").exists() or not (path / "irs.npy").exists():
        return None
    return IRBank(path)


# ---------------------------------------------------------------------------------------------------------
# Установленные VST3
# ---------------------------------------------------------------------------------------------------------

# Ключевые слова имен параметров -> какой параметр схемы ими управлять (нормализованное значение 0..1)
_VST_PARAM_HINTS = {
    "reverb": {"mix|wet": "wet_level", "decay|size|time|length|feedback": "room_size", "damp|high ?cut|tone": "damping"},
    "delay": {"mix|wet": "mix", "feedback|regen": "feedback", "time|delay": "delay_seconds"},
    "compressor": {"thresh": "threshold_db", "ratio": "ratio", "attack": "attack_ms", "release": "release_ms"},
    "saturation": {"drive|gain|satur|amount": "drive_db"},
    "modulation": {"rate|speed": "rate_hz", "depth|width": "depth", "mix|wet": "mix", "feedback": "feedback"},
    "multiband": {"depth|amount|mix": "depth"},
}


class VSTPool:
    """VST3 из описи со статусом ok; плагины грузятся лениво и кэшируются в процессе."""

    def __init__(self, inventory: Path = VST_INVENTORY):
        self.by_category: dict[str, list[dict]] = {}
        if inventory.exists():
            for row in json.loads(inventory.read_text(encoding="utf-8")).get("plugins", []):
                if row.get("status") == "ok" and row.get("category"):
                    self.by_category.setdefault(row["category"], []).append(row)
        self._loaded: dict[str, object] = {}
        self._broken: set[str] = set()

    def has(self, category: str) -> bool:
        return bool([r for r in self.by_category.get(category, []) if r["path"] not in self._broken])

    def render(self, y: np.ndarray, sr: int, category: str, norm: dict, rng: random.Random) -> np.ndarray:
        """
        Случайный плагин категории: параметры с понятными именами — по нормализованным параметрам схемы
        с разбросом ±0.15, прочие непрерывные — случайно в средней части диапазона.
        """
        import pedalboard
        row = rng.choice([r for r in self.by_category[category] if r["path"] not in self._broken])
        try:
            plugin = self._loaded.get(row["path"]) or pedalboard.load_plugin(row["path"])
            self._loaded[row["path"]] = plugin
            hints = _VST_PARAM_HINTS.get(category, {})
            import re
            for name, param in plugin.parameters.items():
                if getattr(param, "type", float) is not float:
                    continue
                target = next((norm.get(key) for pattern, key in hints.items() if re.search(pattern, name, re.I)),
                              None)
                value = (float(np.clip(target + rng.uniform(-0.15, 0.15), 0.0, 1.0)) if target is not None
                         else rng.uniform(0.3, 0.7))
                try:
                    param.raw_value = value
                except Exception:  # параметр только для чтения или с дискретными шагами
                    continue
            out = plugin.process(y, sr, reset=True)
            if not np.isfinite(out).all():
                raise ValueError("nan")
            return np.asarray(out, dtype=np.float32)
        except Exception:
            self._broken.add(row["path"])
            raise


@lru_cache(maxsize=1)
def default_vst_pool() -> VSTPool:
    return VSTPool()


# ---------------------------------------------------------------------------------------------------------
# Реализации эффектов
# ---------------------------------------------------------------------------------------------------------

class EffectVariants:
    """
    Рендер цепочки, где каждый плагин с вероятностью 1 − own_prob заменяется случайной альтернативной
    реализацией. Метка (цепочка) не меняется. Выход всегда стерео (2, N).

    Пример:
        variants = EffectVariants(sr=44100)
        y_wet, used = variants.apply_chain(y, chain, random.Random(0), target_key_index=0, target_scale="minor")
    """

    def __init__(self, sr: int = 44100, own_prob: float = OWN_PROB, ir_bank: IRBank | None | str = "default",
                 vst_pool: VSTPool | None | str = "default", dsp: DSPEngine | None = None):
        self.sr = sr
        self.own_prob = own_prob
        self.ir_bank = default_ir_bank() if ir_bank == "default" else ir_bank
        self.vst_pool = default_vst_pool() if vst_pool == "default" else vst_pool
        self.dsp = dsp or DSPEngine()
        # Варианты: имя -> (вес, функция). VST добавляются в available(), если в описи есть плагин категории.
        self.variants: dict[str, dict[str, tuple[float, Callable]]] = {
            "eq": {"eq_random_q": (1.0, self._eq_random_q), "eq_linear_phase": (1.0, self._eq_linear_phase)},
            "compressor": {"comp_ff_rms": (1.0, self._comp_ff_rms), "comp_ff_peak": (1.0, self._comp_ff_peak),
                           "comp_fb_opto": (1.0, self._comp_fb_opto), "comp_pedalboard_jitter": (0.5, self._comp_pb)},
            "deesser": {"deess_dynamic_band": (1.0, self._deess_band), "deess_wideband": (1.0, self._deess_wide)},
            "multiband_compressor": {"mb_upward_downward": (1.0, self._mb_ott), "mb_downward": (1.0, self._mb_down)},
            "distortion": {"sat_tanh_insert": (1.0, self._sat_tanh), "sat_tube": (1.0, self._sat_tube),
                           "sat_tape": (1.0, self._sat_tape), "sat_hardclip": (0.6, self._sat_clip),
                           "sat_bitcrush": (0.3, self._sat_bitcrush)},
            "chorus": {"chorus_multivoice": (1.0, self._chorus_multi), "chorus_flanger": (0.7, self._flanger),
                       "chorus_dimension": (0.7, self._dimension), "chorus_pedalboard_jitter": (0.4, self._chorus_pb)},
            "stereo_enhancer": {"stereo_microshift": (1.0, self._microshift),
                                "stereo_decorrelator": (1.0, self._decorrelator), "stereo_haas_lr": (0.6, self._haas_lr)},
            "reverb": {"rev_ir_external": (1.4, self._rev_ir_external), "rev_ir_algorithmic": (1.4, self._rev_ir_synth),
                       "rev_freeverb_send": (0.6, self._rev_freeverb_send), "rev_decay_send": (0.6, self._rev_decay_send)},
            "delay": {"delay_pingpong": (1.0, self._delay_pingpong), "delay_tape": (1.0, self._delay_tape),
                      "delay_stereo_offset": (0.7, self._delay_offset)},
            "exciter": {"exc_air_shelf": (1.0, self._exc_shelf), "exc_rectifier": (1.0, self._exc_rectify)},
        }

    # --- выбор реализации ----------------------------------------------------------------------------------

    def available(self, name: str) -> dict[str, tuple[float, Callable]]:
        """Доступные альтернативные реализации плагина (без нашей), включая VST3 категории."""
        out = dict(self.variants.get(name, {}))
        if self.ir_bank is None or "external" not in self.ir_bank.kinds():
            out.pop("rev_ir_external", None)
        if self.ir_bank is None or "synthetic" not in self.ir_bank.kinds():
            out.pop("rev_ir_algorithmic", None)
        category = VST_CATEGORY.get(name)
        if self.vst_pool is not None and category and self.vst_pool.has(category):
            out["vst_" + category] = (1.0, lambda y, phys, norm, r, c=category: self.vst_pool.render(
                y, self.sr, c, norm, r))
        return out

    def choose(self, name: str, rng: random.Random) -> str:
        """Имя реализации: "own" с вероятностью own_prob (или если альтернатив нет), иначе случайная по весам."""
        options = self.available(name)
        if not options or rng.random() < self.own_prob:
            return "own"
        names = sorted(options)
        return rng.choices(names, weights=[options[n][0] for n in names])[0]

    def render_plugin(self, y: np.ndarray, plugin: PluginConfig, impl: str, rng: random.Random) -> np.ndarray:
        """Один плагин заданной альтернативной реализацией (y — (2, N))."""
        phys = self.dsp.get_physical_params(plugin)
        return self.available(plugin.name)[impl][1](y, phys, plugin.params, rng)

    def apply_chain(self, y: np.ndarray, chain: ChainConfig, rng: random.Random, target_key_index: int | None = None,
                    target_scale: str | None = None, force: dict[str, str] | None = None
                    ) -> tuple[np.ndarray, list[str]]:
        """
        Рендер цепочки со случайными реализациями. Подряд идущие плагины с нашей реализацией рендерятся одним
        вызовом DSPEngine.apply_chain (без выходного лимитера), так что слои harmony и экситер ведут себя как
        обычно. В конце — тот же лимитер −1 дБ, что у DSPEngine.

        Args:
            force: {плагин: реализация} — зафиксировать выбор (для тестов и прослушивания).

        Returns:
            (стерео (2, N), список "плагин:реализация" в порядке цепочки).
        """
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = np.vstack([y, y])
        used: list[str] = []
        pending: list[PluginConfig] = []

        def flush(signal: np.ndarray) -> np.ndarray:
            if not pending:
                return signal
            out = self.dsp.apply_chain(signal, self.sr, ChainConfig(plugins=list(pending)),
                                       target_key_index=target_key_index, target_scale=target_scale,
                                       final_limiter=False)
            pending.clear()
            out = np.asarray(out, dtype=np.float32)
            return np.vstack([out, out]) if out.ndim == 1 else out

        for plugin in chain.plugins:
            impl = (force or {}).get(plugin.name) or self.choose(plugin.name, rng)
            if impl != "own":
                y = flush(y)
                try:
                    y_new = self.render_plugin(y, plugin, impl, rng)
                    if y_new.shape != y.shape or not np.isfinite(y_new).all():
                        raise ValueError(f"{impl}: форма {y_new.shape} или не конечные значения")
                    y = y_new
                except Exception:
                    impl = "own"  # сломавшийся вариант (например, VST) заменяется нашей реализацией
            if impl == "own":
                pending.append(plugin)
            used.append(f"{plugin.name}:{impl}")
        y = flush(y)
        y = Pedalboard([Limiter(threshold_db=-1.0, release_ms=10.0)])(y, self.sr)
        return np.nan_to_num(np.asarray(y, dtype=np.float32)), used

    # --- EQ ------------------------------------------------------------------------------------------------

    def _eq_bands(self, phys: dict) -> list[tuple[str, float, float]]:
        kinds = ["low_shelf"] + ["peak"] * 5 + ["high_shelf"]
        return [(k, phys.get(f"band{i}_freq_hz", 1000.0), phys.get(f"band{i}_gain_db", 0.0))
                for i, k in enumerate(kinds, start=1)]

    def _eq_random_q(self, y, phys, norm, rng):
        """Те же полосы с другой добротностью (у FabFilter/Fruity EQ2 Q по умолчанию разный) и ±20 % усиления."""
        effects = []
        for kind, freq, gain in self._eq_bands(phys):
            gain *= rng.uniform(0.8, 1.2)
            freq = min(freq, 0.45 * self.sr)
            if kind == "low_shelf":
                effects.append(LowShelfFilter(cutoff_frequency_hz=freq, gain_db=gain, q=rng.uniform(0.5, 1.0)))
            elif kind == "high_shelf":
                effects.append(HighShelfFilter(cutoff_frequency_hz=freq, gain_db=gain, q=rng.uniform(0.5, 1.0)))
            else:
                effects.append(PeakFilter(cutoff_frequency_hz=freq, gain_db=gain, q=rng.uniform(0.4, 2.5)))
        return Pedalboard(effects)(y, self.sr)

    def _eq_linear_phase(self, y, phys, norm, rng):
        """Линейно-фазовая кривая в частотной области: колокола — гауссианы по log f, полки — логистические."""
        n = 1 << int(np.ceil(np.log2(y.shape[1] + 1)))
        f = np.fft.rfftfreq(n, 1.0 / self.sr)
        logf = np.log2(np.maximum(f, 10.0))
        curve = np.zeros_like(f)
        for kind, freq, gain in self._eq_bands(phys):
            gain *= rng.uniform(0.8, 1.2)
            x = logf - np.log2(freq)
            if kind == "peak":
                curve += gain * np.exp(-0.5 * (x / rng.uniform(0.25, 0.9)) ** 2)
            else:
                sign = -1.0 if kind == "low_shelf" else 1.0
                curve += gain / (1.0 + np.exp(-sign * x / rng.uniform(0.15, 0.5)))
        spec = np.fft.rfft(y, n=n, axis=1) * (10.0 ** (curve / 20.0))[None, :]
        return np.fft.irfft(spec, n=n, axis=1)[:, :y.shape[1]].astype(np.float32)

    # --- компрессор ----------------------------------------------------------------------------------------

    @staticmethod
    def _comp_settings(phys, rng):
        thresh = phys.get("threshold_db", -20.0) + rng.uniform(-3.0, 3.0)
        ratio = max(1.2, phys.get("ratio", 4.0) * rng.uniform(0.75, 1.33))
        attack = phys.get("attack_ms", 10.0) * rng.uniform(0.5, 2.0)
        release = phys.get("release_ms", 100.0) * rng.uniform(0.5, 2.0)
        return thresh, ratio, attack, release

    def _comp_ff_rms(self, y, phys, norm, rng):
        thresh, ratio, attack, release = self._comp_settings(phys, rng)
        return compress(y, self.sr, thresh, ratio, attack, release, knee_db=rng.uniform(0.0, 12.0), detector="rms",
                        window_ms=rng.uniform(3.0, 50.0), makeup_db=_makeup_db(thresh, ratio) + rng.uniform(-2, 2))

    def _comp_ff_peak(self, y, phys, norm, rng):
        thresh, ratio, attack, release = self._comp_settings(phys, rng)
        return compress(y, self.sr, thresh, ratio, attack, release, knee_db=rng.uniform(0.0, 6.0), detector="peak",
                        makeup_db=_makeup_db(thresh, ratio) + rng.uniform(-2, 2))

    def _comp_fb_opto(self, y, phys, norm, rng):
        thresh, ratio, attack, release = self._comp_settings(phys, rng)
        # Пиковый детектор с обратной связью видит уровень выше RMS: порог сдвинут на +6 дБ
        gain = _feedback_opto(y.astype(np.float64), thresh + 6.0, ratio, _coef(max(attack, 5.0), self.sr),
                              _coef(release, self.sr), _coef(release * rng.uniform(3.0, 10.0), self.sr))
        makeup = 10.0 ** ((_makeup_db(thresh, ratio) + rng.uniform(-2, 2)) / 20.0)
        return (y * gain[None, :] * makeup).astype(np.float32)

    def _comp_pb(self, y, phys, norm, rng):
        thresh, ratio, attack, release = self._comp_settings(phys, rng)
        board = Pedalboard([Compressor(threshold_db=thresh, ratio=ratio, attack_ms=attack, release_ms=release),
                            Gain(gain_db=_makeup_db(thresh, ratio))])
        return board(y, self.sr)

    # --- де-эссер и OTT ------------------------------------------------------------------------------------

    def _deess_band(self, y, phys, norm, rng):
        """Динамический EQ: сжимается только полоса сибилянтов (граница 4.5–8 кГц), сайдчейн +14 дБ как у DSPEngine."""
        split = rng.uniform(4500.0, 8000.0)
        high = sosfilt(_sos("high", split, self.sr, 4), y, axis=1).astype(np.float32)
        low = y - high
        thresh = phys.get("threshold_db", -20.0) + rng.uniform(-3.0, 3.0)
        ratio = phys.get("ratio", 4.0) * rng.uniform(0.75, 1.33)
        boost = 10.0 ** (14.0 / 20.0)
        comp = compress(high * boost, self.sr, thresh, ratio, rng.uniform(0.5, 3.0), rng.uniform(20.0, 80.0),
                        knee_db=rng.uniform(0.0, 6.0), detector="peak") / boost
        return (low + comp).astype(np.float32)

    def _deess_wide(self, y, phys, norm, rng):
        """Широкополосный де-эссер: детектор по ВЧ-полосе, ослабляется весь сигнал."""
        side = sosfilt(_sos("high", rng.uniform(4000.0, 7000.0), self.sr, 4), y, axis=1) * 10.0 ** (14.0 / 20.0)
        thresh = phys.get("threshold_db", -20.0) + rng.uniform(-2.0, 4.0)
        ratio = phys.get("ratio", 4.0) * rng.uniform(0.6, 1.0)
        return compress(y, self.sr, thresh, ratio, rng.uniform(0.5, 3.0), rng.uniform(20.0, 80.0), knee_db=3.0,
                        detector="peak", sidechain=side.astype(np.float32))

    def _split3(self, y, rng):
        lo_f, hi_f = rng.uniform(80.0, 250.0), rng.uniform(1800.0, 5000.0)
        low = sosfilt(_sos("low", lo_f, self.sr, 4), y, axis=1).astype(np.float32)
        high = sosfilt(_sos("high", hi_f, self.sr, 4), y, axis=1).astype(np.float32)
        return [low, y - low - high, high]

    def _mb_ott(self, y, phys, norm, rng):
        """3 полосы с другими частотами раздела, нисходящая + восходящая компрессия со сглаживанием, depth-микс."""
        depth = phys.get("depth", 0.5)
        out = np.zeros_like(y)
        for band, (td, tu, mk) in zip(self._split3(y, rng), ((-26.0, -40.0, 4.5), (-22.0, -34.0, 2.5),
                                                             (-30.0, -44.0, 7.5))):
            out += compress(band, self.sr, td + rng.uniform(-4, 4), rng.uniform(3.0, 8.0), rng.uniform(0.5, 5.0),
                            rng.uniform(30.0, 150.0), knee_db=rng.uniform(0.0, 6.0), detector="rms", window_ms=5.0,
                            makeup_db=mk + rng.uniform(-2, 2), up_thresh=tu + rng.uniform(-4, 4),
                            up_ratio=rng.uniform(2.0, 4.0), up_max=rng.uniform(8.0, 18.0))
        return ((1.0 - depth) * y + depth * out).astype(np.float32)

    def _mb_down(self, y, phys, norm, rng):
        """Многополосная только нисходящая компрессия (Maximus/C6-подобная) с компенсацией, глубина -> ratio."""
        depth = phys.get("depth", 0.5)
        out = np.zeros_like(y)
        for band in self._split3(y, rng):
            level = 20.0 * np.log10(_rms(band))
            thresh = level + rng.uniform(-4.0, 4.0)
            ratio = 1.5 + 8.0 * depth * rng.uniform(0.7, 1.3)
            out += compress(band, self.sr, thresh, ratio, rng.uniform(1.0, 10.0), rng.uniform(40.0, 200.0),
                            knee_db=6.0, detector="rms", window_ms=10.0, makeup_db=_makeup_db(thresh, ratio))
        return _match_rms(out, y)

    # --- сатурация -----------------------------------------------------------------------------------------

    @staticmethod
    def _drive_gain(phys, rng, lo=0.4, hi=0.9) -> float:
        # Нормируем вход к пику, потом усиливаем на долю drive: 0 дБ — почти чисто, 20 дБ — явный перегруз
        return 10.0 ** (phys.get("drive_db", 10.0) * rng.uniform(lo, hi) / 20.0)

    @staticmethod
    def _peak(y) -> float:
        return float(np.percentile(np.abs(y), 99.9) + 1e-9)

    def _sat_tanh(self, y, phys, norm, rng):
        peak = self._peak(y)
        g = self._drive_gain(phys, rng) * rng.uniform(0.5, 1.2)
        wet = _match_rms(np.tanh(g * y / peak), y)
        mix = rng.uniform(0.5, 1.0)
        return ((1.0 - mix) * y + mix * wet).astype(np.float32)

    def _sat_tube(self, y, phys, norm, rng):
        """Асимметричная (ламповая) кривая: четные гармоники, DC убирается ФВЧ 20 Гц."""
        peak = self._peak(y)
        g = self._drive_gain(phys, rng) * rng.uniform(0.5, 1.2)
        bias = rng.uniform(0.1, 0.4)
        wet = np.tanh(g * y / peak + bias) - np.tanh(bias)
        wet = sosfilt(_sos("high", 20.0, self.sr), wet, axis=1)
        wet = _match_rms(wet, y)
        mix = rng.uniform(0.3, 1.0)
        return ((1.0 - mix) * y + mix * wet).astype(np.float32)

    def _sat_tape(self, y, phys, norm, rng):
        """Лента: arctan-сатурация, потеря верха (тем больше, чем сильнее drive), подъем низа (head bump)."""
        peak = self._peak(y)
        drive = phys.get("drive_db", 10.0)
        g = self._drive_gain(phys, rng, 0.3, 0.7)
        wet = np.arctan(g * y / peak) * (2.0 / np.pi)
        board = Pedalboard([LowpassFilter(cutoff_frequency_hz=min(0.45 * self.sr, rng.uniform(9000.0, 16000.0)
                                                                  * (1.0 - drive / 45.0))),
                            LowShelfFilter(cutoff_frequency_hz=rng.uniform(70.0, 130.0), gain_db=rng.uniform(0.5, 2.5),
                                           q=0.7)])
        return _match_rms(board(wet.astype(np.float32), self.sr), y)

    def _sat_clip(self, y, phys, norm, rng):
        peak = self._peak(y)
        drive = phys.get("drive_db", 10.0)
        thr_db = 20.0 * np.log10(peak) - drive * rng.uniform(0.3, 0.6)
        board = Pedalboard([Clipping(threshold_db=float(thr_db)),
                            LowpassFilter(cutoff_frequency_hz=min(0.45 * self.sr, rng.uniform(8000.0, 14000.0)))])
        wet = _match_rms(board(y, self.sr), y)
        mix = rng.uniform(0.3, 0.8)
        return ((1.0 - mix) * y + mix * wet).astype(np.float32)

    def _sat_bitcrush(self, y, phys, norm, rng):
        drive = phys.get("drive_db", 10.0)
        wet = Bitcrush(bit_depth=float(np.clip(16.0 - drive * rng.uniform(0.35, 0.55), 5.0, 16.0)))(y, self.sr)
        mix = rng.uniform(0.15, 0.5)
        return ((1.0 - mix) * y + mix * _match_rms(wet, y)).astype(np.float32)

    # --- модуляция -----------------------------------------------------------------------------------------

    def _lfo(self, n: int, rate: float, phase: float, shape: str) -> np.ndarray:
        t = np.arange(n) / self.sr
        ph = 2.0 * np.pi * rate * t + phase
        if shape == "triangle":
            return (2.0 / np.pi) * np.arcsin(np.sin(ph))
        return np.sin(ph)

    def _chorus_multi(self, y, phys, norm, rng):
        """2–4 голоса по стерео, базовая задержка 8–25 мс, LFO синус/треугольник с разнесенными фазами."""
        rate = phys.get("rate_hz", 1.0) * rng.uniform(0.7, 1.4)
        depth_ms = phys.get("depth", 0.25) * rng.uniform(4.0, 12.0)
        mix = phys.get("mix", 0.2) * rng.uniform(0.8, 1.5)
        voices = rng.randint(2, 4)
        shape = rng.choice(["sine", "triangle"])
        mono = y.mean(axis=0)
        wet = np.zeros_like(y)
        for v in range(voices):
            base = rng.uniform(8.0, 25.0)
            d = (base + depth_ms * 0.5 * (1.0 + self._lfo(len(mono), rate * rng.uniform(0.9, 1.1),
                                                          2.0 * np.pi * v / voices, shape))) * self.sr / 1000.0
            voice = _mod_delay(mono, d)
            pan = (v / max(voices - 1, 1)) if voices > 1 else 0.5
            wet[0] += voice * np.cos(pan * np.pi / 2) * 1.41 / voices
            wet[1] += voice * np.sin(pan * np.pi / 2) * 1.41 / voices
        return ((1.0 - 0.5 * mix) * y + mix * wet).astype(np.float32)

    def _flanger(self, y, phys, norm, rng):
        """Флэнжер: задержка 0.5–4 мс с модуляцией и обратной связью (numba), каналы в противофазе LFO."""
        rate = phys.get("rate_hz", 1.0) * rng.uniform(0.2, 0.8)
        fb = float(np.clip(phys.get("feedback", 0.2) * rng.uniform(1.0, 1.8), 0.0, 0.85)) * rng.choice([1, -1])
        mix = phys.get("mix", 0.2) * rng.uniform(0.8, 1.4)
        base, depth = rng.uniform(0.5, 2.0), rng.uniform(0.5, 2.5)
        out = np.empty_like(y)
        for ch in range(2):
            d = (base + depth * 0.5 * (1.0 + self._lfo(y.shape[1], rate, np.pi * ch, "triangle"))) * self.sr / 1000.0
            out[ch] = _flanger_kernel(y[ch].astype(np.float64), d, fb)
        return ((1.0 - 0.5 * mix) * y + mix * out).astype(np.float32)

    def _dimension(self, y, phys, norm, rng):
        """«Dimension D»: два голоса в противофазе, медленный LFO, маленькая глубина, без обратной связи."""
        rate = rng.uniform(0.2, 0.8)
        mix = float(np.clip(phys.get("mix", 0.2) * rng.uniform(1.0, 2.0), 0.1, 0.6))
        mono = y.mean(axis=0)
        wet = np.empty_like(y)
        for ch in range(2):
            d = (rng.uniform(6.0, 12.0) + rng.uniform(0.3, 1.5) * self._lfo(len(mono), rate, np.pi * ch, "sine")
                 ) * self.sr / 1000.0
            wet[ch] = _mod_delay(mono, d)
        return ((1.0 - 0.5 * mix) * y + mix * wet).astype(np.float32)

    def _chorus_pb(self, y, phys, norm, rng):
        return Chorus(rate_hz=phys.get("rate_hz", 1.0) * rng.uniform(0.6, 1.6),
                      depth=float(np.clip(phys.get("depth", 0.25) * rng.uniform(0.6, 1.6), 0.0, 1.0)),
                      centre_delay_ms=rng.uniform(5.0, 20.0),
                      feedback=float(np.clip(phys.get("feedback", 0.0) * rng.uniform(0.5, 1.5), 0.0, 0.9)),
                      mix=float(np.clip(phys.get("mix", 0.2) * rng.uniform(0.8, 1.4), 0.0, 1.0)))(y, self.sr)

    # --- стерео --------------------------------------------------------------------------------------------

    def _microshift(self, y, phys, norm, rng):
        """MicroShift/Doubler: L и R — копии со сдвигом ±5–15 центов и задержкой, подмешанные к сухому."""
        width = phys.get("width", 0.5)
        cents = rng.uniform(5.0, 15.0)
        mono = y.mean(axis=0)
        out = y.copy()
        for ch, sign in ((0, 1.0), (1, -1.0)):
            shifted = PitchShift(semitones=sign * cents / 100.0)(mono[None, :].astype(np.float32), self.sr)[0]
            d = int(rng.uniform(5.0, 25.0) * self.sr / 1000.0)
            shifted = np.concatenate([np.zeros(d, np.float32), shifted[:len(mono) - d]])
            out[ch] = out[ch] + width * rng.uniform(0.3, 0.6) * shifted
        return out.astype(np.float32)

    def _decorrelator(self, y, phys, norm, rng):
        """Side из Mid через цепочку фазовращателей (декоррелятор стерео-имиджеров) выше 150–400 Гц."""
        from app.audio.synthetic_reverbs import schroeder_allpass
        width = phys.get("width", 0.5)
        mid = y.mean(axis=0)
        side = mid.astype(np.float64)
        for d_ms in [rng.uniform(1.0, 9.0) for _ in range(rng.randint(3, 6))]:
            side = schroeder_allpass(side, int(d_ms * self.sr / 1000.0), rng.uniform(0.5, 0.7))
        side = sosfilt(_sos("high", rng.uniform(150.0, 400.0), self.sr, 4), side)
        side = side * width * rng.uniform(0.4, 0.9)
        return np.vstack([y[0] + side, y[1] - side]).astype(np.float32)

    def _haas_lr(self, y, phys, norm, rng):
        """Классический Хаас: один канал задержан (в моно-сумме — гребенка, как у простых плагинов)."""
        d = int(phys.get("delay_ms", 15.0) * rng.uniform(0.7, 1.3) * self.sr / 1000.0)
        width = phys.get("width", 0.5)
        out = y.copy()
        ch = rng.randint(0, 1)
        delayed = np.concatenate([np.zeros(d, np.float32), y[ch][:y.shape[1] - d]])
        out[ch] = (1.0 - 0.5 * width) * y[ch] + 0.5 * width * delayed
        return out.astype(np.float32)

    # --- реверб --------------------------------------------------------------------------------------------

    def _rt60_target(self, phys) -> float:
        if "decay_s" in phys:
            return float(phys["decay_s"])
        return freeverb_rt60(phys.get("room_size", 0.5))

    def _send_filters(self, phys, rng) -> list:
        """Фильтры и predelay мокрого сигнала: из параметров send-шины, если заданы, иначе случайные."""
        nyq = 0.45 * self.sr
        low = phys.get("low_cut_hz", rng.choice([20.0, rng.uniform(80.0, 600.0)]))
        damping = phys.get("damping", 0.5)
        high = phys.get("high_cut_hz", rng.uniform(4000.0, 18000.0) * (1.0 - 0.5 * damping))
        predelay = phys.get("predelay_ms", rng.choice([0.0, rng.uniform(5.0, 80.0)]))
        effects = [HighpassFilter(cutoff_frequency_hz=min(low, nyq)), LowpassFilter(cutoff_frequency_hz=min(high, nyq))]
        if predelay > 0.5:
            effects.insert(0, Delay(delay_seconds=predelay / 1000.0, feedback=0.0, mix=1.0))
        return effects

    def _mix_reverb(self, y, wet, phys, rng) -> np.ndarray:
        # Тот же баланс, что у send-шины DSPEngine: 2·dry·x + wet_level·wet (масштаб JUCE Reverb)
        wet_level = phys.get("wet_level", 0.2) * rng.uniform(0.7, 1.4)
        return (2.0 * phys.get("dry_level", 0.9) * y + wet_level * wet).astype(np.float32)

    def _convolve(self, x: np.ndarray, ir: np.ndarray) -> np.ndarray:
        """Моно-посыл -> стерео хвост (ИХ (2, n) или (1, n)), длина как у x."""
        mono = x.mean(axis=0)
        ir = ir[:, :len(mono)]
        if ir.shape[0] == 1:
            ir = np.vstack([ir, ir])
        return np.stack([fftconvolve(mono, ir[c])[:len(mono)] for c in range(2)]).astype(np.float32)

    def _rev_ir(self, y, phys, rng, kind):
        target = self._rt60_target(phys) * rng.uniform(0.75, 1.33)
        ir, _ = self.ir_bank.sample(rng, target_rt60=target, kind=kind)
        send = Pedalboard(self._send_filters(phys, rng))(y, self.sr)
        return self._mix_reverb(y, self._convolve(send, ir), phys, rng)

    def _rev_ir_external(self, y, phys, norm, rng):
        return self._rev_ir(y, phys, rng, "external")

    def _rev_ir_synth(self, y, phys, norm, rng):
        return self._rev_ir(y, phys, rng, "synthetic")

    def _rev_freeverb_send(self, y, phys, norm, rng):
        room = float(np.clip(phys.get("room_size", 0.5) + rng.uniform(-0.1, 0.1), 0.0, 1.0))
        send = Pedalboard(self._send_filters(phys, rng) + [
            Reverb(room_size=room, damping=float(np.clip(phys.get("damping", 0.5) + rng.uniform(-0.2, 0.2), 0, 1)),
                   wet_level=1.0, dry_level=0.0, width=rng.uniform(0.5, 1.0))])
        return self._mix_reverb(y, send(y, self.sr), phys, rng)

    def _rev_decay_send(self, y, phys, norm, rng):
        send = Pedalboard(self._send_filters(phys, rng))(y, self.sr)
        wet = decay_reverb(send, self.sr, self._rt60_target(phys) * rng.uniform(0.75, 1.33),
                           float(np.clip(phys.get("damping", 0.5) + rng.uniform(-0.2, 0.2), 0, 1)),
                           seed=rng.randint(0, 2 ** 31 - 1))
        return self._mix_reverb(y, wet, phys, rng)

    # --- дилей ---------------------------------------------------------------------------------------------

    def _taps(self, x: np.ndarray, delay_s: float, feedback: float, filt: list | None, saturate: float = 0.0,
              max_taps: int = 24) -> list[np.ndarray]:
        """Повторы x: k-й задержан на k·delay, ослаблен feedback^(k-1) и k раз пропущен через filt."""
        d = int(delay_s * self.sr)
        n = x.shape[-1]
        taps = []
        cur = x.astype(np.float32)
        for k in range(1, max_taps + 1):
            if k * d >= n or feedback ** (k - 1) < 1e-3:
                break
            if filt is not None and k > 1:
                cur = Pedalboard(filt)(cur, self.sr)
            if saturate > 0 and k > 1:
                peak = float(np.max(np.abs(cur)) + 1e-9)
                cur = (np.tanh(saturate * cur / peak) * peak / np.tanh(saturate)).astype(np.float32)
            tap = np.zeros_like(cur)
            tap[..., k * d:] = cur[..., :n - k * d] * feedback ** (k - 1)
            taps.append(tap)
        return taps

    def _delay_settings(self, phys, rng):
        return (phys.get("delay_seconds", 0.25) * rng.uniform(0.97, 1.03),
                float(np.clip(phys.get("feedback", 0.3) * rng.uniform(0.8, 1.25), 0.0, 0.9)),
                phys.get("mix", 0.15) * rng.uniform(0.8, 1.5))

    def _delay_filters(self, rng) -> list:
        return [HighpassFilter(cutoff_frequency_hz=rng.uniform(100.0, 500.0)),
                LowpassFilter(cutoff_frequency_hz=min(0.45 * self.sr, rng.uniform(2500.0, 9000.0)))]

    def _delay_pingpong(self, y, phys, norm, rng):
        delay, fb, mix = self._delay_settings(phys, rng)
        mono = y.mean(axis=0)
        wet = np.zeros_like(y)
        first = rng.randint(0, 1)
        for k, tap in enumerate(self._taps(mono, delay, fb, self._delay_filters(rng) if rng.random() < 0.7 else None)):
            wet[(first + k) % 2] += tap
        return (y + mix * wet).astype(np.float32)

    def _delay_tape(self, y, phys, norm, rng):
        """Ленточное эхо: повторы темнеют и насыщаются, в центре."""
        delay, fb, mix = self._delay_settings(phys, rng)
        taps = self._taps(y, delay, fb, self._delay_filters(rng), saturate=rng.uniform(0.5, 2.0))
        wet = np.sum(taps, axis=0) if taps else np.zeros_like(y)
        return ((1.0 - 0.5 * mix) * y + mix * wet).astype(np.float32)

    def _delay_offset(self, y, phys, norm, rng):
        """Разные времена L/R (вторая сторона ×1.25–1.6, «пунктир»), фильтрованная обратная связь."""
        delay, fb, mix = self._delay_settings(phys, rng)
        mono = y.mean(axis=0)
        filt = self._delay_filters(rng)
        wet = np.zeros_like(y)
        for ch, factor in ((0, 1.0), (1, rng.uniform(1.25, 1.6))):
            taps = self._taps(mono, delay * factor, fb, filt)
            if taps:
                wet[ch] = np.sum(taps, axis=0)
        if rng.random() < 0.5:
            wet = wet[::-1]
        return (y + mix * wet).astype(np.float32)

    # --- экситер -------------------------------------------------------------------------------------------

    def _exc_shelf(self, y, phys, norm, rng):
        """«Air»-полка (Maag/Pultec-подобная) с легкой сатурацией верха вместо генерации гармоник."""
        gain = phys.get("mix", 0.2) * rng.uniform(12.0, 25.0)
        out = HighShelfFilter(cutoff_frequency_hz=min(phys.get("cutoff_hz", 7000.0) * rng.uniform(0.7, 1.5),
                                                      0.45 * self.sr), gain_db=gain, q=0.6)(y, self.sr)
        return out.astype(np.float32)

    def _exc_rectify(self, y, phys, norm, rng):
        """Гармоники выпрямителем |x| и tanh на ВЧ-полосе (Aphex-подобно), подмес по mix."""
        cutoff = phys.get("cutoff_hz", 7000.0) * rng.uniform(0.6, 1.2)
        band = sosfilt(_sos("high", cutoff, self.sr, 4), y, axis=1)
        peak = float(np.percentile(np.abs(band), 99) + 1e-9)
        harm = rng.uniform(0.3, 1.0) * np.abs(band / peak) + np.tanh(3.0 * band / peak)
        harm = sosfilt(_sos("high", cutoff, self.sr, 4), harm, axis=1)
        harm = harm * (max(_rms(band) * 2.0, _rms(y) * 0.25) / _rms(harm))
        return (y + phys.get("mix", 0.2) * rng.uniform(0.8, 1.8) * harm).astype(np.float32)


def all_variant_names() -> dict[str, list[str]]:
    """Список альтернативных реализаций по плагинам (без VST) — для документации и тестов."""
    v = EffectVariants(ir_bank=None, vst_pool=None)
    return {name: sorted(opts) for name, opts in v.variants.items()}
