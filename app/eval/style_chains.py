"""
Реалистичные цепочки и пул сухого контента для обучения кодировщика стиля (app/eval/style_embedding.py).

В отличие от _generate_balanced_random_config (балансировка классов для модели v4), здесь распределение
ближе к реальной работе: много слабой обработки (почти плоский EQ, легкая компрессия, слабая сатурация,
короткий реверб), иногда «ничего», плюс архетипы из декодированных FL-пресетов (data/raw/presets/decoded/,
data/processed/archetypes/) с разбросом параметров. Кодировщик должен различать и тонкую обработку.

Песни реального теста (scripts/real_pairs.py) исключаются из пула целиком.
"""
import json
import random
from pathlib import Path

import numpy as np

from app.audio.dsp_engine import DSPEngine
from app.dataset.dataset_generator import DatasetGenerator, is_dry_source
from app.dataset.dataset_schema import DATAGEN_PLUGINS, ChainConfig, PluginConfig

# Песни тестовых реальных пар: не попадают ни в обучение, ни в валидацию кодировщика
TEST_SONG_PREFIXES = ("Albert_Kader_Whiptails", "oxeye_Sorry", "Ronnie_Jones")
EQ2_JSON = Path("data/raw/presets/decoded/eq2.json")


def is_test_song(path: Path) -> bool:
    return path.stem.startswith(TEST_SONG_PREFIXES)


def content_pool(dry_dir: str | Path = "data/raw/dry_vocals/cambridge", val_ratio: float = 0.1,
                 seed: int = 7) -> tuple[list[Path], list[Path]]:
    """Сухие файлы без тестовых песен, разбитые на train/val по песне (song_key)."""
    files = sorted(f for ext in ("*.wav", "*.flac") for f in Path(dry_dir).rglob(ext)
                   if is_dry_source(f) and not is_test_song(f))
    groups: dict[str, list[Path]] = {}
    for f in files:
        groups.setdefault(DatasetGenerator.song_key(f), []).append(f)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_val = max(1, int(round(len(keys) * val_ratio)))
    val = [f for k in keys[:n_val] for f in groups[k]]
    train = [f for k in keys[n_val:] for f in groups[k]]
    return train, val


def _norm(name: str, **phys: float) -> PluginConfig:
    ranges = DATAGEN_PLUGINS[name]["params"]
    return PluginConfig(name, {k: DSPEngine.normalize_value(v, *ranges[k]) for k, v in phys.items()})


def _load_decoded_eqs() -> list[list[dict]]:
    if not EQ2_JSON.exists():
        return []
    return [e["active"]["bands"] for e in json.loads(EQ2_JSON.read_text(encoding="utf-8"))
            if e.get("active", {}).get("bands")]


_DECODED_EQS: list[list[dict]] | None = None


def decoded_eqs() -> list[list[dict]]:
    global _DECODED_EQS
    if _DECODED_EQS is None:
        _DECODED_EQS = _load_decoded_eqs()
    return _DECODED_EQS


def palette_eq_from_bands(bands: list[dict], gain_scale: float = 1.0) -> PluginConfig:
    """Полосы EQ2 на палитру eq: полоса i -> полоса i, HP/LP -> полка −12 дБ (как в render_archetypes.py)."""
    ranges = DATAGEN_PLUGINS["eq"]["params"]
    params = {}
    for i, b in enumerate(bands[:7], start=1):
        gain = -12.0 if b["type"] in ("high pass", "low pass") else b["gain_db"] * gain_scale
        params[f"band{i}_freq_hz"] = DSPEngine.normalize_value(b["freq_hz"], *ranges[f"band{i}_freq_hz"])
        params[f"band{i}_gain_db"] = DSPEngine.normalize_value(gain, *ranges[f"band{i}_gain_db"])
    return PluginConfig("eq", params)


def random_eq(rng: random.Random, gain_std_db: float) -> PluginConfig:
    ranges = DATAGEN_PLUGINS["eq"]["params"]
    params = {}
    for i in range(1, 8):
        params[f"band{i}_freq_hz"] = rng.uniform(0.1, 0.9)
        gain = float(np.clip(rng.gauss(0.0, gain_std_db), -12.0, 12.0))
        params[f"band{i}_gain_db"] = DSPEngine.normalize_value(gain, *ranges[f"band{i}_gain_db"])
    return PluginConfig("eq", params)


def _sample_eq(rng: random.Random) -> PluginConfig:
    kind = rng.random()
    if kind < 0.35:
        return random_eq(rng, rng.uniform(0.3, 1.2))  # почти плоский
    if kind < 0.70:
        return random_eq(rng, rng.uniform(1.5, 4.0))  # умеренный
    if kind < 0.82:
        eq = random_eq(rng, 1.0)  # сильная правка 1-3 полос
        for i in rng.sample(range(1, 8), rng.randint(1, 3)):
            eq.params[f"band{i}_gain_db"] = rng.choice([rng.uniform(0.0, 0.25), rng.uniform(0.75, 1.0)])
        return eq
    eqs = decoded_eqs()
    if eqs:
        return palette_eq_from_bands(rng.choice(eqs), gain_scale=rng.uniform(0.5, 1.2))
    return random_eq(rng, 2.0)


def _sample_compressor(rng: random.Random) -> PluginConfig:
    level = rng.random()
    if level < 0.5:  # легкая
        return _norm("compressor", threshold_db=rng.uniform(-20, -10), ratio=rng.uniform(1.8, 3.0),
                     attack_ms=rng.uniform(5, 40), release_ms=rng.uniform(60, 300))
    if level < 0.85:
        return _norm("compressor", threshold_db=rng.uniform(-28, -16), ratio=rng.uniform(3.0, 6.0),
                     attack_ms=rng.uniform(2, 20), release_ms=rng.uniform(40, 200))
    return _norm("compressor", threshold_db=rng.uniform(-36, -26), ratio=rng.uniform(6.0, 10.0),
                 attack_ms=rng.uniform(1, 6), release_ms=rng.uniform(20, 100))


def _sample_reverb(rng: random.Random) -> PluginConfig:
    if rng.random() < 0.5:  # короткая комната / слабый посыл
        return _norm("reverb", room_size=rng.uniform(0.15, 0.5), damping=rng.uniform(0.3, 0.9),
                     wet_level=rng.uniform(0.06, 0.15), dry_level=rng.uniform(0.85, 1.0))
    return _norm("reverb", room_size=rng.uniform(0.5, 0.95), damping=rng.uniform(0.1, 0.7),
                 wet_level=rng.uniform(0.12, 0.36), dry_level=rng.uniform(0.65, 0.95))


def sample_realistic_chain(rng: random.Random) -> ChainConfig:
    """Случайная цепочка с реалистичным распределением (в т.ч. «ничего» и архетипы пресетов)."""
    u = rng.random()
    if u < 0.07:
        return ChainConfig()  # только выходной лимитер DSPEngine
    if u < 0.14:
        return archetype_chain(rng.choice(["che", "ninoo"]), rng)

    plugins: list[PluginConfig] = []
    if rng.random() < 0.15:
        plugins.append(_norm("resonance_suppressor", threshold_db=rng.uniform(5, 10),
                             max_attenuation_db=rng.uniform(3, 12)))
    if rng.random() < 0.85:
        plugins.append(_sample_eq(rng))
    if rng.random() < 0.15:
        plugins.append(_norm("pitch_corrector", speed=rng.choice([rng.uniform(0.25, 0.6), rng.uniform(0.6, 1.0)])))
    if rng.random() < 0.6:
        plugins.append(_sample_compressor(rng))
    if rng.random() < 0.25:
        plugins.append(_norm("deesser", threshold_db=rng.uniform(-36, -12), ratio=rng.uniform(2, 8)))
    if rng.random() < 0.2:
        plugins.append(_norm("multiband_compressor", depth=rng.choice([rng.uniform(0.2, 0.4), rng.uniform(0.4, 0.85)])))
    if rng.random() < 0.2:
        plugins.append(_norm("distortion", drive_db=rng.choice([rng.uniform(4, 8), rng.uniform(8, 20)])))
    if rng.random() < 0.12:
        plugins.append(_norm("chorus", rate_hz=rng.uniform(0.4, 2.0), depth=rng.uniform(0.1, 0.35),
                             feedback=rng.uniform(0.0, 0.3), mix=rng.uniform(0.12, 0.4)))
    if rng.random() < 0.15:
        plugins.append(_norm("stereo_enhancer", delay_ms=rng.uniform(6, 25), width=rng.uniform(0.25, 1.0)))
    if rng.random() < 0.45:
        plugins.append(_sample_reverb(rng))
    if rng.random() < 0.25:
        plugins.append(_norm("delay", delay_seconds=rng.uniform(0.08, 0.5), feedback=rng.uniform(0.1, 0.5),
                             mix=rng.uniform(0.08, 0.3)))
    if rng.random() < 0.2:
        plugins.append(_norm("exciter", mix=rng.uniform(0.1, 0.3), cutoff_hz=rng.uniform(5500, 10000)))
    return ChainConfig(plugins=plugins)


def archetype_chain(name: str, rng: random.Random | None = None) -> ChainConfig:
    """
    Палитровые версии архетипов CHE и Ninoo (как che_palette / ninoo_palette в render_archetypes.py,
    без гейта и гармонии). С rng — с разбросом параметров вокруг предположенных значений.
    """
    def j(value: float, rel: float = 0.25) -> float:
        return value * (1.0 + rng.uniform(-rel, rel)) if rng else value

    eqs = {e["source"] + str(e["insert"]) + str(e["slot"]): e["active"]["bands"]
           for e in (json.loads(EQ2_JSON.read_text(encoding="utf-8")) if EQ2_JSON.exists() else [])
           if e.get("active", {}).get("bands")}

    def eq(key_part: str, insert, slot: int) -> list[PluginConfig]:
        for k, bands in eqs.items():
            if key_part in k and k.endswith(f"{insert}{slot}"):
                return [palette_eq_from_bands(bands, gain_scale=j(1.0, 0.3))]
        return []

    maximus = dict(threshold_db=j(-12.0), ratio=j(3.0), attack_ms=j(5.0), release_ms=j(100.0))
    if name == "che":
        plugins = [*eq("CHE", None, 2),
                   _norm("compressor", threshold_db=j(-20.0), ratio=j(4.0), attack_ms=j(3.0), release_ms=j(60.0)),
                   _norm("compressor", threshold_db=j(-14.0), ratio=j(6.0), attack_ms=j(1.0), release_ms=j(50.0)),
                   *eq("CHE", None, 5),
                   _norm("multiband_compressor", depth=0.104),
                   _norm("compressor", **maximus)]
    elif name == "ninoo":
        plugins = [_norm("pitch_corrector", speed=1.0),
                   *eq("Hyperpop", 1, 1), *eq("Hyperpop", 1, 2), *eq("Hyperpop", 1, 3),
                   _norm("distortion", drive_db=4.0),
                   _norm("compressor", **maximus),
                   _norm("multiband_compressor", depth=j(0.30)), _norm("multiband_compressor", depth=j(0.368)),
                   *eq("Hyperpop", 1, 8),
                   _norm("multiband_compressor", depth=j(0.30)),
                   _norm("compressor", **maximus),
                   _norm("chorus", rate_hz=0.6, depth=0.25, feedback=0.0, mix=j(0.3)),
                   _norm("reverb", room_size=j(0.6), damping=0.5, wet_level=float(np.clip(j(0.15), 0.06, 0.36)),
                         dry_level=1.0),
                   _norm("delay", delay_seconds=j(0.43), feedback=j(0.3), mix=float(np.clip(j(0.2), 0.08, 0.35)))]
    else:
        raise ValueError(f"Неизвестный архетип: {name}")
    return ChainConfig(plugins=plugins)


def render_chain(x: np.ndarray, chain: ChainConfig, dsp: DSPEngine | None = None, variants=None,
                 rng: random.Random | None = None, sr: int = 44100) -> np.ndarray:
    """
    Рендер цепочки кодировщика для стерео x (2, N). Подавление резонансов DSPEngine работает только с моно,
    поэтому оно применяется по каналам до остальной цепочки.

    Args:
        variants: EffectVariants (app/audio/effect_variants.py) — доменная рандомизация: остальная цепочка
            рендерится случайными реализациями эффектов (выбор задает rng; один и тот же seed дает ту же
            реализацию на любом содержании). None — нашим DSPEngine, как у оптимизатора.
    """
    from app.audio.vocal_enhancer import VocalEnhancer
    dsp = dsp or DSPEngine()
    rest = [p for p in chain.plugins if p.name != "resonance_suppressor"]
    if len(rest) != len(chain.plugins):
        phys = dsp.get_physical_params(next(p for p in chain.plugins if p.name == "resonance_suppressor"))
        enhancer = VocalEnhancer(sr=sr)
        x = np.vstack([enhancer.suppress_resonances(ch, threshold_db=phys["threshold_db"],
                                                    max_attenuation_db=phys["max_attenuation_db"])
                       for ch in x]).astype(np.float32)
    if variants is None:
        return dsp.apply_chain(x, sr, ChainConfig(plugins=rest))
    return variants.apply_chain(x, ChainConfig(plugins=rest), rng or random.Random(0))[0]


def chain_flags(chain: ChainConfig) -> np.ndarray:
    """Какие типы плагинов есть в цепочке (вспомогательная цель обучения)."""
    return np.array([float(any(p.name == name for p in chain.plugins)) for name in FLAG_PLUGINS], dtype=np.float32)


FLAG_PLUGINS = ["resonance_suppressor", "eq", "pitch_corrector", "compressor", "deesser", "multiband_compressor",
                "distortion", "chorus", "stereo_enhancer", "reverb", "delay", "exciter"]
