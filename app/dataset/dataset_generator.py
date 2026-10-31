import os
import random
import json
from typing import List, Dict, Tuple
import librosa
import numpy as np
import soundfile as sf
import scipy.signal as signal
from pathlib import Path
from tqdm import tqdm

from app.dataset.dataset_schema import (
    DatasetSample, ChainConfig, PluginConfig, CHAIN_ORDER, DATAGEN_PLUGINS, LEGACY_PARAM_RANGES,
    PARAM_LAYOUT, PARAM_OFFSETS, PARAMETER_VECTOR_LENGTH, RENDER_ORDER, model_param_ranges
)
from app.audio.dsp_engine import DSPEngine
from app.audio.audio_processor import AudioProcessor
from app.mixing.chain_optimizer import normalize_lufs, to_stereo
from app.ml.features import MODEL_LUFS, signal_features, window_seconds

# Файлы из пула сухих вокалов Cambridge, которые на самом деле уже обработаны инженером (их сухие версии
# лежат рядом). Как «сухой» исходник или голос референса они учат модель обрабатывать уже обработанное,
# а в реальном бенчмарке служат истинной целью (scripts/real_pairs.py), поэтому из пула исключаются.
PROCESSED_VOCAL_STEMS = frozenset({
    "Albert_Kader_Whiptails_12_Vox_Wet",  # полная обработка; сухой дубль — _11_Vox_Dry
    "Ronnie_Jones_Why_Dont_You_Stay_23_LeadVoxTapeEffect",  # ленточный эффект; сухой — _24_LeadVoxRaw
    # Тюн; сухие версии — те же имена с суффиксом _NoPitchProcessing
    "oxeye_Sorry_12_LeadVox",
    "oxeye_Sorry_14_LeadVoxDT",
    "oxeye_Sorry_16_BackingVox1",
    "oxeye_Sorry_18_BackingVox2",
    "oxeye_Sorry_20_BackingVox3",
    "oxeye_Sorry_22_BackingVox4",
})


def is_dry_source(path: Path) -> bool:
    """Подходит ли файл как сухой исходник: не служебный файл macOS, не пустой и не из списка обработанных."""
    try:
        return (not path.name.startswith("._") and path.stem not in PROCESSED_VOCAL_STEMS
                and path.stat().st_size > 1000)
    except OSError:
        return False


# Песни реальных тестовых пар (scripts/real_pairs.py, docs/research/real_pairs.md): начиная с dataset_v4 они
# не попадают ни в train, ни в val целиком, включая сухие дубли, иначе модель заранее слышит голоса теста.
TEST_SONG_PREFIXES = ("Albert_Kader_Whiptails", "oxeye_Sorry", "Ronnie_Jones_Why_Dont_You_Stay")


def is_test_song(path: Path) -> bool:
    """Файл из песни реальных тестовых пар."""
    return path.stem.startswith(TEST_SONG_PREFIXES)


def _is_audio_file(path: Path) -> bool:
    """Аудиофайл пула без проверки на обработанность (так файлы отбирались при разбиении dataset_v3)."""
    try:
        return not path.name.startswith("._") and path.stat().st_size > 1000
    except OSError:
        return False


def _legacy_reverb_norm(param: str, value: float) -> float:
    """
    Нормализованное значение реверба, заданное в прежних диапазонах (wet_level 0.06–0.36, room_size 0.15–0.95),
    -> в текущих DATAGEN_PLUGINS. Потолки подняли для оптимизатора (wet до 0.60, room до 0.99), а генераторы
    сэмплируют нормализованные значения: без пересчета физический wet датасета сместился бы до ~0.57.
    Пересчет сохраняет прежнее физическое распределение, метка — в текущих диапазонах (как и рендер).
    """
    lo, hi = LEGACY_PARAM_RANGES[("reverb", param)]
    return DSPEngine.normalize_value(lo + value * (hi - lo), *DATAGEN_PLUGINS["reverb"]["params"][param])


# Стили, где иногда бывает «большой» мокрый реверб выше прежнего потолка 0.36 (эмбиент-вокал pluggnb/rage/hyperpop)
_BIG_REVERB_STYLES = ("pluggnb_rnb", "rage", "hyperpop")
_BIG_REVERB_PROB = 0.12


def _generate_balanced_random_config(rng: random.Random = random) -> ChainConfig:
    """
    Генерирует стилистически осмысленную и при этом сбалансированную по классам цепочку плагинов.
    Значения параметров [0.0..1.0] покрывают весь музыкальный диапазон DATAGEN_PLUGINS без скрытого обрезания.
    """
    chain = ChainConfig()

    styles = ["pop", "hyperpop", "hiphop", "indie", "rock_metal", "rnb_modern", "electronic", "cinematic", "dry_clean"]
    selected_style = rng.choices(
        styles,
        weights=[0.16, 0.16, 0.14, 0.14, 0.12, 0.12, 0.08, 0.05, 0.03],
        k=1
    )[0]

    active_plugins = ["eq"]  # Эквалайзер активен всегда

    # Сбалансированные вероятности активации (каждый из 10 эффектов представлен в 35%..65% датасета)
    style_prob = {
        "pop": {
            "pitch_corrector": 0.55, "compressor": 0.75, "deesser": 0.65, "multiband_compressor": 0.35,
            "distortion": 0.25, "chorus": 0.45, "stereo_enhancer": 0.55, "reverb": 0.75, "delay": 0.60,
            "exciter": 0.60
        },
        "hyperpop": {
            "pitch_corrector": 0.90, "compressor": 0.65, "deesser": 0.75, "multiband_compressor": 0.85,
            "distortion": 0.75, "chorus": 0.65, "stereo_enhancer": 0.80, "reverb": 0.65, "delay": 0.75,
            "exciter": 0.80
        },
        "hiphop": {
            "pitch_corrector": 0.75, "compressor": 0.80, "deesser": 0.70, "multiband_compressor": 0.50,
            "distortion": 0.45, "chorus": 0.30, "stereo_enhancer": 0.55, "reverb": 0.55, "delay": 0.65,
            "exciter": 0.55
        },
        "indie": {
            "pitch_corrector": 0.20, "compressor": 0.55, "deesser": 0.45, "multiband_compressor": 0.25,
            "distortion": 0.50, "chorus": 0.75, "stereo_enhancer": 0.50, "reverb": 0.80, "delay": 0.55,
            "exciter": 0.35
        },
        "rock_metal": {
            "pitch_corrector": 0.15, "compressor": 0.85, "deesser": 0.65, "multiband_compressor": 0.60,
            "distortion": 0.85, "chorus": 0.25, "stereo_enhancer": 0.40, "reverb": 0.45, "delay": 0.40,
            "exciter": 0.50
        },
        "rnb_modern": {
            "pitch_corrector": 0.70, "compressor": 0.70, "deesser": 0.65, "multiband_compressor": 0.45,
            "distortion": 0.20, "chorus": 0.55, "stereo_enhancer": 0.70, "reverb": 0.80, "delay": 0.65,
            "exciter": 0.65
        },
        "electronic": {
            "pitch_corrector": 0.80, "compressor": 0.60, "deesser": 0.60, "multiband_compressor": 0.75,
            "distortion": 0.60, "chorus": 0.70, "stereo_enhancer": 0.75, "reverb": 0.70, "delay": 0.70,
            "exciter": 0.70
        },
        "cinematic": {
            "pitch_corrector": 0.20, "compressor": 0.50, "deesser": 0.40, "multiband_compressor": 0.30,
            "distortion": 0.15, "chorus": 0.60, "stereo_enhancer": 0.75, "reverb": 0.90, "delay": 0.75,
            "exciter": 0.45
        },
        "dry_clean": {
            "pitch_corrector": 0.0, "compressor": 0.25, "deesser": 0.25, "multiband_compressor": 0.0,
            "distortion": 0.0, "chorus": 0.0, "stereo_enhancer": 0.0, "reverb": 0.0, "delay": 0.0,
            "exciter": 0.0
        }
    }

    probs = style_prob[selected_style]
    for name, p in probs.items():
        if rng.random() < p:
            active_plugins.append(name)

    active_plugins = [p for p in CHAIN_ORDER if p in active_plugins]

    for name in active_plugins:
        plugin_range = DATAGEN_PLUGINS[name]["params"]
        params = {}

        if name == "eq":
            # На инференсе EQ рассчитывается аналитически через SpectralMatcher.
            # В синтетическом датасете задаем умеренную вариацию АЧХ (±1.8 дБ вокруг 0.5),
            # чтобы случайные ±12 дБ провалы/подъемы не маскировали De-esser, Exciter и OTT.
            for p_name in plugin_range:
                if "gain" in p_name:
                    params[p_name] = float(rng.uniform(0.42, 0.58))
                else:
                    params[p_name] = float(rng.uniform(0.25, 0.75))
        elif name == "compressor":
            # Порог всегда в рабочей зоне [-36..-13.6 дБ], чтобы компрессор реально срабатывал
            thresh = float(rng.uniform(0.05, 0.80))
            intensity = 1.0 - thresh
            params["threshold_db"] = thresh
            params["ratio"] = float(np.clip(0.20 + 0.75 * intensity + rng.uniform(-0.12, 0.12), 0.0, 1.0))
            params["attack_ms"] = float(np.clip(0.80 - 0.65 * intensity + rng.uniform(-0.12, 0.12), 0.0, 1.0))
            params["release_ms"] = float(np.clip(0.75 - 0.55 * intensity + rng.uniform(-0.12, 0.12), 0.0, 1.0))
        elif name == "deesser":
            thresh = float(rng.uniform(0.05, 0.80))
            intensity = 1.0 - thresh
            params["threshold_db"] = thresh
            params["ratio"] = float(np.clip(0.20 + 0.75 * intensity + rng.uniform(-0.12, 0.12), 0.0, 1.0))
        elif name == "chorus":
            mix_val = float(rng.uniform(0.10, 0.95))
            rate_val = float(rng.uniform(0.08, 0.92))
            params["mix"] = mix_val
            params["rate_hz"] = rate_val
            params["depth"] = float(np.clip(0.20 + 0.65 * mix_val + rng.uniform(-0.12, 0.12), 0.0, 1.0))
            params["feedback"] = float(np.clip(0.15 + 0.65 * mix_val + rng.uniform(-0.12, 0.12), 0.0, 1.0))
        elif name == "reverb":
            wet_val = float(rng.uniform(0.08, 0.95))
            room_val = float( np.clip(0.20 + 0.70 * wet_val + rng.uniform(-0.18, 0.18), 0.0, 1.0) )
            damp_val = float(rng.uniform(0.08, 0.92))
            # Dry-уровень физически связан с Wet-уровнем (баланс Dry/Wet), чтобы избежать вырождения при нормализации
            dry_val = float(np.clip(0.95 - 0.65 * wet_val + rng.uniform(-0.08, 0.08), 0.0, 1.0))
            # wet/room сэмплируются в прежних диапазонах (физическое распределение как у dataset_v3)
            params["room_size"] = _legacy_reverb_norm("room_size", room_val)
            params["damping"] = damp_val
            params["wet_level"] = _legacy_reverb_norm("wet_level", wet_val)
            params["dry_level"] = dry_val
        elif name == "delay":
            dly_sec = float(rng.uniform(0.05, 0.95))
            mix_val = float(rng.uniform(0.10, 0.95))
            fb_val = float(np.clip(0.18 + 0.70 * mix_val + rng.uniform(-0.12, 0.12), 0.0, 1.0))
            params["delay_seconds"] = dly_sec
            params["feedback"] = fb_val
            params["mix"] = mix_val
        elif name == "exciter":
            mix_val = float(rng.uniform(0.10, 0.95))
            cutoff_val = float(rng.uniform(0.05, 0.95))
            params["mix"] = mix_val
            params["cutoff_hz"] = cutoff_val
        else:
            for p_name in plugin_range:
                val = rng.uniform(0.05, 0.95)
                if rng.random() < 0.30:
                    if selected_style in ("hyperpop", "electronic") and name in ("pitch_corrector", "multiband_compressor"):
                        val = rng.uniform(0.55, 0.98)
                    elif selected_style == "rock_metal" and name == "distortion":
                        val = rng.uniform(0.50, 0.98)
                params[p_name] = float(val)

        chain.plugins.append(PluginConfig(name=name, params=params))

    return chain


# Стили реалистичного генератора (dataset_v4, модель v5): вес стиля и вероятность включения каждого эффекта.
# Опора — разобранные пресеты (data/raw/presets/harvest/INDEX.md, §3: EQ 5/6, компрессия 5/6, тюн 4/6, реверб
# 4/6, де-эссер в стоковых цепях почти всегда через Maximus) и рецепты роликов (docs/research/videos/SUMMARY.md,
# TIPS.md: явный дисторшн только в rage, хорус «на ширину» ~7 %, stereo_enhancer на лиде не ставят).
# В отличие от _generate_balanced_random_config (~50 % на каждый эффект, модели v3/v4) редкие эффекты здесь
# действительно редкие, а сатурация и хорус чаще слабые. Маргинальные частоты — realistic_on_rates().
REALISTIC_STYLES: Dict[str, Tuple[float, Dict[str, float]]] = {
    # Стоковый «западный» рэп/трэп: Pitcher → EQ → Fruity Compressor → Maximus de-ess → Reeverb 2 → Delay 3
    "stock_west": (0.20, {
        "pitch_corrector": 0.65, "compressor": 0.95, "deesser": 0.75, "multiband_compressor": 0.15,
        "distortion": 0.12, "chorus": 0.05, "stereo_enhancer": 0.08, "reverb": 0.85, "delay": 0.70,
        "exciter": 0.45}),
    # Стоковые «русские» пресеты: две ступени компрессии, Blood Overdrive/Soundgoodizer как соус
    "stock_ru": (0.15, {
        "pitch_corrector": 0.80, "compressor": 0.95, "deesser": 0.70, "multiband_compressor": 0.35,
        "distortion": 0.35, "chorus": 0.10, "stereo_enhancer": 0.08, "reverb": 0.85, "delay": 0.65,
        "exciter": 0.55}),
    # Rage / новая школа: жесткий тюн, два сатуратора, хорус ~7 %
    "rage": (0.14, {
        "pitch_corrector": 0.95, "compressor": 0.90, "deesser": 0.70, "multiband_compressor": 0.35,
        "distortion": 0.55, "chorus": 0.30, "stereo_enhancer": 0.10, "reverb": 0.85, "delay": 0.70,
        "exciter": 0.55}),
    # Hyperpop: OTT, сатурация на всех слоях, модуляция
    "hyperpop": (0.10, {
        "pitch_corrector": 0.95, "compressor": 0.85, "deesser": 0.60, "multiband_compressor": 0.65,
        "distortion": 0.55, "chorus": 0.40, "stereo_enhancer": 0.25, "reverb": 0.80, "delay": 0.70,
        "exciter": 0.65}),
    # Pluggnb / R&B: мягкая динамика, длинный темный реверб, хорус на ширину
    "pluggnb_rnb": (0.12, {
        "pitch_corrector": 0.85, "compressor": 0.90, "deesser": 0.65, "multiband_compressor": 0.20,
        "distortion": 0.08, "chorus": 0.35, "stereo_enhancer": 0.20, "reverb": 0.90, "delay": 0.60,
        "exciter": 0.45}),
    "pop": (0.13, {
        "pitch_corrector": 0.55, "compressor": 0.90, "deesser": 0.75, "multiband_compressor": 0.25,
        "distortion": 0.08, "chorus": 0.15, "stereo_enhancer": 0.20, "reverb": 0.85, "delay": 0.55,
        "exciter": 0.50}),
    "indie_rock": (0.10, {
        "pitch_corrector": 0.20, "compressor": 0.90, "deesser": 0.55, "multiband_compressor": 0.25,
        "distortion": 0.40, "chorus": 0.25, "stereo_enhancer": 0.15, "reverb": 0.75, "delay": 0.45,
        "exciter": 0.35}),
    # Почти без обработки: учит «ничего не включать», когда референс близок к сухому
    "dry_clean": (0.06, {
        "pitch_corrector": 0.10, "compressor": 0.60, "deesser": 0.40, "multiband_compressor": 0.0,
        "distortion": 0.0, "chorus": 0.0, "stereo_enhancer": 0.0, "reverb": 0.20, "delay": 0.05,
        "exciter": 0.10}),
}

# Стили, где тюн жесткий (Retune 0 / Pitcher Fast, 8 источников в SUMMARY.md)
_HARD_TUNE_STYLES = ("rage", "hyperpop", "pluggnb_rnb", "stock_ru")
# Доля сэмплов, где параметр берется равномерно по всему диапазону, а не из реалистичного prior: регрессия
# должна видеть весь диапазон, иначе на необычных референсах она стягивается к prior.
_PARAM_COVERAGE_SHARE = 0.3


def realistic_on_rates() -> Dict[str, float]:
    """Ожидаемая частота включения каждого эффекта по смеси REALISTIC_STYLES (EQ включен всегда)."""
    total = sum(w for w, _ in REALISTIC_STYLES.values())
    rates = {"eq": 1.0}
    for name in CHAIN_ORDER[1:]:
        rates[name] = sum(w * probs[name] for w, probs in REALISTIC_STYLES.values()) / total
    return rates


def _realistic_params(name: str, style: str, rng: random.Random) -> Dict[str, float]:
    """Нормализованные параметры эффекта для реалистичного генератора (prior по рецептам + доля равномерных)."""
    u = rng.uniform
    covered = rng.random() < _PARAM_COVERAGE_SHARE
    if name == "eq":
        # Как в сбалансированном генераторе: EQ на инференсе считается аналитически (SpectralMatcher)
        return {p: float(u(0.42, 0.58) if "gain" in p else u(0.25, 0.75)) for p in DATAGEN_PLUGINS["eq"]["params"]}
    if name == "pitch_corrector":
        # Retune 0 / Pitcher Fast в новой школе, Medium (0.5–0.7 физ.) в «натуральном»; ниже ~0.4 физ. не бывает
        if covered:
            return {"speed": u(0.0, 1.0)}
        hard = style in _HARD_TUNE_STYLES and rng.random() < 0.75
        return {"speed": u(0.8, 1.0) if hard else u(0.2, 0.65)}
    if name == "compressor":
        thresh = u(0.05, 0.80)
        intensity = 1.0 - thresh
        if covered:
            ratio = np.clip(0.20 + 0.75 * intensity + u(-0.12, 0.12), 0.0, 1.0)
            attack = np.clip(0.80 - 0.65 * intensity + u(-0.12, 0.12), 0.0, 1.0)
            release = np.clip(0.75 - 0.55 * intensity + u(-0.12, 0.12), 0.0, 1.0)
        else:
            # Ratio 2.5–5 (prior 3.5), attack 10–30 мс, release 170–330 мс (TIPS.md, п. 4)
            ratio, attack, release = u(0.08, 0.40) + 0.1 * intensity, u(0.11, 0.37), u(0.26, 0.53)
        return {"threshold_db": thresh, "ratio": float(ratio), "attack_ms": float(attack),
                "release_ms": float(release)}
    if name == "deesser":
        thresh = u(0.05, 0.80)
        return {"threshold_db": thresh,
                "ratio": float(np.clip(0.20 + 0.75 * (1.0 - thresh) + u(-0.12, 0.12), 0.0, 1.0))}
    if name == "multiband_compressor":
        return {"depth": u(0.5, 1.0) if style == "hyperpop" and not covered else u(0.0, 1.0)}
    if name == "distortion":
        # Слабая сатурация (0–4 дБ, «соус» Blood Overdrive/Soundgoodizer) — самый частый случай; жесткий
        # перегруз в основном в rage/hyperpop/rock
        heavy_style = style in ("rage", "hyperpop", "indie_rock")
        tier = rng.choices(("weak", "medium", "strong"), weights=(0.30, 0.40, 0.30) if heavy_style
                           else (0.55, 0.35, 0.10))[0]
        if covered:
            tier = "any"
        drive = {"weak": (0.0, 0.2), "medium": (0.2, 0.55), "strong": (0.55, 1.0), "any": (0.0, 1.0)}[tier]
        return {"drive_db": u(*drive)}
    if name == "chorus":
        if not covered and rng.random() < 0.5:
            # Легкий хорус «на ширину»: mix 5–11 %, малая глубина
            return {"rate_hz": u(0.05, 0.6), "depth": u(0.0, 0.35), "feedback": u(0.0, 0.25), "mix": u(0.0, 0.15)}
        mix = u(0.15, 0.95)
        return {"rate_hz": u(0.08, 0.92), "depth": float(np.clip(0.20 + 0.65 * mix + u(-0.12, 0.12), 0.0, 1.0)),
                "feedback": float(np.clip(0.15 + 0.65 * mix + u(-0.12, 0.12), 0.0, 1.0)), "mix": mix}
    if name == "stereo_enhancer":
        return {"delay_ms": u(0.05, 0.95), "width": u(0.05, 0.95)}
    if name == "reverb":
        wet = u(0.08, 0.95)
        if covered:
            room, damp = float(np.clip(0.20 + 0.70 * wet + u(-0.18, 0.18), 0.0, 1.0)), u(0.08, 0.92)
        else:
            # Длинный (2.3–4 с) и темный (срез верха 6–8 кГц) реверб: большая комната, высокое затухание
            room, damp = u(0.5, 0.95), u(0.55, 0.95)
        dry = float(np.clip(0.95 - 0.65 * wet + u(-0.08, 0.08), 0.0, 1.0))
        # Физически как в dataset_v4 (wet 0.08–0.35, room 0.15–0.95), иногда в «больших» стилях — wet 0.36–0.55
        wet_norm = _legacy_reverb_norm("wet_level", wet)
        if style in _BIG_REVERB_STYLES and rng.random() < _BIG_REVERB_PROB:
            wet_norm = DSPEngine.normalize_value(u(0.36, 0.55), *DATAGEN_PLUGINS["reverb"]["params"]["wet_level"])
        return {"room_size": _legacy_reverb_norm("room_size", room), "damping": damp, "wet_level": wet_norm,
                "dry_level": dry}
    if name == "delay":
        lo, hi = DATAGEN_PLUGINS["delay"]["params"]["delay_seconds"]
        if covered:
            delay = u(0.05, 0.95)
        else:
            # Синхронизация с темпом: 1/4 (4 источника), 1/8, реже 1/16 при 120–165 BPM
            beats = rng.choices((1.0, 0.5, 0.25), weights=(0.45, 0.40, 0.15))[0]
            delay = DSPEngine.normalize_value(60.0 / u(120.0, 165.0) * beats, lo, hi)
        mix = u(0.10, 0.95)
        return {"delay_seconds": float(delay), "mix": mix,
                "feedback": float(np.clip(0.18 + 0.70 * mix + u(-0.12, 0.12), 0.0, 1.0))}
    if name == "exciter":
        return {"mix": u(0.05, 0.95), "cutoff_hz": u(0.05, 0.95)}
    raise ValueError(f"Нет реалистичного prior для плагина {name}")


def _generate_realistic_config(rng: random.Random = random) -> ChainConfig:
    """
    Цепочка с реалистичными частотами включения и параметрами (dataset_v4, модель v5): см. REALISTIC_STYLES.
    Размерности выхода модели те же (MODEL_PLUGINS / PARAM_LAYOUT), меняется только распределение.
    """
    styles = list(REALISTIC_STYLES)
    style = rng.choices(styles, weights=[REALISTIC_STYLES[s][0] for s in styles], k=1)[0]
    probs = REALISTIC_STYLES[style][1]
    active = ["eq"] + [name for name in CHAIN_ORDER[1:] if rng.random() < probs[name]]
    return ChainConfig(plugins=[PluginConfig(name=name, params={k: float(v) for k, v in
                                                                 _realistic_params(name, style, rng).items()})
                                for name in CHAIN_ORDER if name in active])


def _harmony(semitones: float, level_db: float, delay_ms: float, detune_cents: float,
             width: float) -> PluginConfig:
    """Слой harmony с физическими параметрами (форманты сохранены, фейдер слоя открыт)."""
    ranges = DATAGEN_PLUGINS["harmony"]["params"]
    phys = {"semitones": semitones, "formant_shift": 0.0, "level_db": level_db, "delay_ms": delay_ms,
            "detune_cents": detune_cents, "width": width, "mix": 1.0}
    return PluginConfig("harmony", {k: DSPEngine.normalize_value(v, *ranges[k]) for k, v in phys.items()})


def sample_reference_layers(rng: random.Random) -> List[PluginConfig]:
    """
    Даблы, бэки и гармонии в референсе (аугментация только на стороне референса, модель их не предсказывает).
    Реальный референс — вокал из микса, где слои дают модуляцию и ширину, похожие на хорус; без них модель
    путала такую ширину с chorus (docs/research/calibration.md, §4). Уровни и панорама — SUMMARY.md, §8:
    стерео-даблы ±100 %, около −5 дБ (до −15); моно-дабл под лидом −10…−15 дБ; гармонии ±80–90 %, −7…−8 дБ.
    """
    u = rng.uniform
    kinds = ["stereo_double", "mono_double", "harmony"]
    first = rng.choices(kinds, weights=(0.50, 0.20, 0.30))[0]
    chosen = [first]
    if rng.random() < 0.3:
        chosen.append(rng.choice([k for k in kinds if k != first]))
    layers = []
    for kind in chosen:
        if kind == "stereo_double":
            layers.append(_harmony(0.0, u(-12.0, -3.0), u(8.0, 30.0), u(5.0, 15.0), u(0.85, 1.0)))
        elif kind == "mono_double":
            layers.append(_harmony(0.0, u(-15.0, -10.0), u(5.0, 25.0), u(3.0, 12.0), 0.0))
        else:
            interval = rng.choice((3.0, 4.0, 5.0, 7.0, -5.0, 12.0, -12.0))
            layers.append(_harmony(interval, u(-10.0, -6.0), u(0.0, 15.0), u(0.0, 8.0), u(0.75, 0.9)))
    return layers


def with_reference_layers(chain: ChainConfig, layers: List[PluginConfig]) -> ChainConfig:
    """Цепочка рендера референса: слои harmony встают на свое место в RENDER_ORDER (перед реверб/дилей)."""
    plugins = list(chain.plugins) + list(layers)
    order = {name: i for i, name in enumerate(RENDER_ORDER)}
    return ChainConfig(plugins=sorted(plugins, key=lambda p: order.get(p.name, len(order))))


CHAIN_PRIORS = {"realistic": _generate_realistic_config, "balanced": _generate_balanced_random_config}


class DatasetGenerator:
    """
    Генератор синтетического датасета v2 для обучения Mixing-AI.
    Формирует пары (Dry, Wet) исключительно на чистых студийных вокалах (Cambridge MT / VCTK),
    делит исходники по трекам/певцам на train (85%) и val (15%) до нарезки
    и сохраняет стерео Mid/Side мел-спектрограммы + 18 физических DSP-дескрипторов.
    """

    def __init__(self, dry_dir: Path, output_dir: Path, sr: int = 44100):
        self.dry_dir = Path(dry_dir)
        self.output_dir = Path(output_dir)
        self.sr = sr
        self.dsp = DSPEngine()
        self.audio_processor = AudioProcessor(sr=sr)

    def _generate_random_config(self, prior: str = "realistic") -> ChainConfig:
        return CHAIN_PRIORS[prior]()

    def _flatten_params(self, chain_config: ChainConfig) -> List[float]:
        vec = [0.0] * PARAMETER_VECTOR_LENGTH
        for plugin in chain_config.plugins:
            if plugin.name not in PARAM_LAYOUT:
                continue
            offset = PARAM_OFFSETS[plugin.name]
            p_names = PARAM_LAYOUT[plugin.name]
            for i, p_name in enumerate(p_names):
                vec[offset + i] = plugin.params.get(p_name, 0.0)
        return vec

    @staticmethod
    def song_key(f: Path) -> str:
        """Ключ песни/вокалиста: префикс из двух первых частей имени файла."""
        parts = f.stem.split("_")
        return "_".join(parts[:2]) if len(parts) >= 2 else f.stem

    @staticmethod
    def _pick_reference_file(pool: List[Path], src_file: Path, cross_ratio: float, rng: random.Random) -> Path:
        """
        Выбирает файл, из которого рендерится референс. С вероятностью cross_ratio — другой голос из другой
        песни того же сплита (как в реальной задаче), иначе — тот же файл (референс из того же фрагмента).
        """
        if rng.random() >= cross_ratio:
            return src_file
        src_key = DatasetGenerator.song_key(src_file)
        others = ([f for f in pool if DatasetGenerator.song_key(f) != src_key]
                  or [f for f in pool if f != src_file] or [src_file])
        return rng.choice(others)

    @staticmethod
    def _split_files_by_group(files: List[Path], val_ratio: float = 0.15) -> Tuple[List[Path], List[Path]]:
        """
        Разделяет список исходных файлов на train и val по песне/вокалисту (префикс файла или родительская папка),
        чтобы исключить утечку одного и того же голоса между train и val.
        """
        if len(files) <= 1:
            return files, files

        groups: Dict[str, List[Path]] = {}
        for f in sorted(files):
            groups.setdefault(DatasetGenerator.song_key(f), []).append(f)

        group_keys = sorted(groups.keys())
        rng = random.Random(42)
        rng.shuffle(group_keys)

        if len(group_keys) == 1:
            sorted_f = sorted(files)
            n_val = max(1, int(round(len(sorted_f) * val_ratio)))
            return sorted_f[:-n_val], sorted_f[-n_val:]

        n_val_groups = max(1, int(round(len(group_keys) * val_ratio)))
        val_keys = set(group_keys[:n_val_groups])

        train_files, val_files = [], []
        for k, f_list in groups.items():
            if k in val_keys:
                val_files.extend(f_list)
            else:
                train_files.extend(f_list)

        if not train_files:
            train_files = val_files
        if not val_files:
            val_files = train_files[:1]
        return train_files, val_files

    def generate(self, max_samples: int = 1000, sc_ratio: float = 0.0, save_all_wav: bool = False,
                 cross_ratio: float = 1.0, seed: int = 0, prior: str = "realistic", harmony_ratio: float = 0.35,
                 exclude_test_songs: bool = True, domain_rand_own_prob: float | None = None,
                 workers: int | None = None, resume: bool = False) -> List[DatasetSample]:
        """
        Запускает параллельную генерацию датасета на всех ядрах CPU (без использования SoundCloud).

        Args:
            cross_ratio: Доля пар, где референс рендерится из другого голоса другой песни (реальная задача);
                остальные пары — референс из того же фрагмента, что и исходник (поведение датасета v1).
            seed: Seed выбора файлов, пар и цепочек.
            prior: Распределение цепочек: "realistic" (dataset_v4, частоты как в реальных цепях) или
                "balanced" (dataset_v3, ~50 % на эффект).
            harmony_ratio: Доля референсов с даблами/гармониями (рендер-плагин harmony, без метки).
            exclude_test_songs: Не брать песни реальных тестовых пар (TEST_SONG_PREFIXES) ни в train, ни в val.
            domain_rand_own_prob: Доменная рандомизация рендера референса (dataset_v5): None — все плагины нашим
                DSPEngine, как раньше; число — доля плагинов, оставленных нашей реализации, остальные рендерятся
                случайной альтернативной (app/audio/effect_variants.py). Выбор реализаций идет из отдельного
                генератора, поэтому цепочки, файлы и сплит те же, что без рандомизации при том же seed.
            workers: Число процессов (по умолчанию все ядра).
            resume: Дописать прерванную генерацию: сэмплы, уже записанные в index.jsonl, пропускаются (сэмпл
                детерминирован своим номером и seed, поэтому результат тот же, что у непрерывного прогона);
                оборванная последняя строка отбрасывается.
        """
        if prior not in CHAIN_PRIORS:
            raise ValueError(f"Неизвестный prior цепочек: {prior}")
        task_rng = random.Random(seed)
        dry_out_dir = self.output_dir / "audio" / "dry"
        wet_out_dir = self.output_dir / "audio" / "wet"
        features_out_dir = self.output_dir / "features"

        dry_out_dir.mkdir(parents=True, exist_ok=True)
        wet_out_dir.mkdir(parents=True, exist_ok=True)
        features_out_dir.mkdir(parents=True, exist_ok=True)

        cambridge_dir = self.dry_dir / "cambridge"
        vctk_dir = self.dry_dir / "vctk"

        # Разбиение по песням считается на всех аудиофайлах пула, как у dataset_v3, а обработанные стемы и
        # песни теста убираются уже после него: так val-песни v4 и v5 совпадают, и модели сравнимы на одном val.
        is_valid_audio = _is_audio_file

        def usable(f: Path) -> bool:
            return is_dry_source(f) and not (exclude_test_songs and is_test_song(f))

        cambridge_files: List[Path] = []
        vctk_files: List[Path] = []
        for ext in ["*.wav", "*.flac"]:
            if cambridge_dir.exists():
                cambridge_files.extend([f for f in cambridge_dir.rglob(ext) if is_valid_audio(f)])
            if vctk_dir.exists():
                vctk_files.extend([f for f in vctk_dir.rglob(ext) if is_valid_audio(f)])

        all_sources: Dict[str, List[Path]] = {}
        if cambridge_files:
            all_sources["cambridge"] = cambridge_files
        elif vctk_files:
            all_sources["vctk"] = vctk_files
        else:
            any_audio = []
            for ext in ["*.wav", "*.flac"]:
                any_audio.extend([
                    f for f in self.dry_dir.rglob(ext)
                    if is_valid_audio(f) and "soundcloud" not in str(f).lower()
                ])
            if any_audio:
                all_sources["general"] = any_audio
            else:
                print(f"[-] Исходные сухие файлы не найдены в директории: {self.dry_dir}")
                return []

        print("[*] Доступные студийные источники сухого вокала (SoundCloud отключен):")
        source_splits: Dict[str, Tuple[List[Path], List[Path]]] = {}
        for s_name, s_list in all_sources.items():
            tr_f, val_f = self._split_files_by_group(s_list, val_ratio=0.15)
            tr_f, val_f = [f for f in tr_f if usable(f)], [f for f in val_f if usable(f)]
            if not tr_f and not val_f:
                continue
            tr_f, val_f = tr_f or val_f, val_f or tr_f[:1]
            source_splits[s_name] = (tr_f, val_f)
            print(f"    - {s_name}: всего {len(s_list)} файлов, после исключения обработанных и тестовых песен "
                  f"Train: {len(tr_f)} | Val: {len(val_f)}")
        if not source_splits:
            print(f"[-] После исключения обработанных стемов и тестовых песен сухих файлов не осталось: {self.dry_dir}")
            return []

        tasks = []
        source_keys = list(source_splits.keys())
        n_val_samples = max(1, int(round(max_samples * 0.15))) if max_samples >= 5 else 0
        n_train_samples = max_samples - n_val_samples

        for i in range(max_samples):
            split_name = "train" if i < n_train_samples else "val"
            src_type = task_rng.choice(source_keys)
            tr_pool, val_pool = source_splits[src_type]
            pool = tr_pool if split_name == "train" else val_pool
            src_file = task_rng.choice(pool)
            ref_file = self._pick_reference_file(pool, src_file, cross_ratio, task_rng)
            # Сохраняем WAV для первых 50 сэмплов (или для всех при малом тесте max_samples <= 50)
            write_wav = save_all_wav or (max_samples <= 50) or (i < 30)
            tasks.append((i, src_file, ref_file, src_type, split_name, self.sr, str(self.output_dir), write_wav,
                          seed, prior, harmony_ratio, domain_rand_own_prob))

        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        num_cores = workers or max(1, multiprocessing.cpu_count())

        samples = []
        index_file = self.output_dir / "index.jsonl"
        if resume and index_file.exists():
            for line in index_file.read_text(encoding="utf-8").splitlines():
                try:
                    samples.append(DatasetSample.from_jsonl_line(line))
                except (json.JSONDecodeError, TypeError):
                    continue  # строка, оборванная при прерывании
            with open(index_file, "w", encoding="utf-8", newline="\n") as f_index:
                f_index.writelines(s.to_jsonl_line() + "\n" for s in samples)
            done = {int(s.id) for s in samples}
            tasks = [t for t in tasks if t[0] not in done]
            print(f"[*] Продолжение: готово {len(done)}, осталось {len(tasks)}")
        print(f"[*] Запуск параллельной генерации {len(tasks)} сэмплов на {num_cores} процессах...")

        with open(index_file, "a" if resume else "w", encoding="utf-8") as f_index:
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                for result_line in tqdm(executor.map(_generate_sample_worker, tasks), total=len(tasks)):
                    if result_line is not None:
                        f_index.write(result_line + "\n")
                        f_index.flush()
                        samples.append(DatasetSample.from_jsonl_line(result_line))

        stats = {
            "total_samples": len(samples),
            "train_samples": sum(1 for s in samples if s.split == "train"),
            "val_samples": sum(1 for s in samples if s.split == "val"),
            "sr": self.sr,
            "cross_ratio": cross_ratio,
            "prior": prior,
            "harmony_ratio": harmony_ratio,
            "exclude_test_songs": list(TEST_SONG_PREFIXES) if exclude_test_songs else [],
            "with_reference_layers": sum(1 for s in samples if s.ref_layers),
            "domain_rand_own_prob": domain_rand_own_prob,
            # Диапазоны, в которых нормализованы метки (train_ml копирует их в чекпоинт)
            "param_ranges": model_param_ranges(),
            "ref_impl_counts": _impl_counts(samples),
            "chain_order": CHAIN_ORDER,
            "param_layout": PARAM_LAYOUT
        }
        with open(self.output_dir / "stats.json", "w", encoding="utf-8") as f_stats:
            json.dump(stats, f_stats, indent=4)

        print(f"[+] Генерация датасета успешно завершена ({len(samples)} сэмплов). Сохранено в: {self.output_dir}")
        return samples


def _impl_counts(samples: List[DatasetSample]) -> Dict[str, int]:
    """Сколько раз каждая реализация плагина встретилась в рендерах референсов (для stats.json)."""
    counts: Dict[str, int] = {}
    for s in samples:
        for impl in s.ref_impls:
            counts[impl] = counts.get(impl, 0) + 1
    return dict(sorted(counts.items()))


_VARIANTS = None


def _effect_variants(sr: int, own_prob: float):
    """EffectVariants процесса-воркера (банк ИХ и VST грузятся один раз на процесс)."""
    global _VARIANTS
    if _VARIANTS is None or _VARIANTS.own_prob != own_prob or _VARIANTS.sr != sr:
        from app.audio.effect_variants import EffectVariants
        _VARIANTS = EffectVariants(sr=sr, own_prob=own_prob)
    return _VARIANTS


def _load_chunk(path: Path, sr: int, seconds: float, rng: random.Random) -> np.ndarray:
    """
    Стерео-фрагмент (2, N) заданной длины на уровне MODEL_LUFS: самый громкий из 4 случайных окон файла
    без тишины по краям; короткий файл дополняется тишиной.
    """
    y, _ = librosa.load(path, sr=sr, mono=False)
    y = to_stereo(y)
    _, (start, end) = librosa.effects.trim(librosa.to_mono(y), top_db=60.0)
    y = y[:, start:end]

    n = int(seconds * sr)
    if y.shape[1] > n:
        starts = [rng.randint(0, y.shape[1] - n) for _ in range(4)]
        best = max(starts, key=lambda s0: float(np.mean(y[:, s0:s0 + n] ** 2)))
        y = y[:, best:best + n]
    elif y.shape[1] < n:
        y = np.pad(y, ((0, 0), (0, n - y.shape[1])))
    return normalize_lufs(y, sr, MODEL_LUFS)


def _generate_sample_worker(task) -> str | None:
    """
    Автономный воркер генерации одного сэмпла (Dry/Wet + Mid/Side Mel + 18 DSP дескрипторов).
    Исходник — фрагмент src_file, референс — цепочка, примененная к фрагменту ref_file (другой голос при
    cross-парах). Оба сигнала стерео и подготовлены так же, как на инференсе (app/ml/features.py).
    """
    idx, src_file, ref_file, source_name, split_name, sr, output_dir_str, write_wav, seed, prior, harmony_ratio = task[:11]
    own_prob = task[11] if len(task) > 11 else None

    import warnings
    warnings.filterwarnings("ignore")

    output_dir = Path(output_dir_str)
    dry_out_dir = output_dir / "audio" / "dry"
    wet_out_dir = output_dir / "audio" / "wet"
    features_out_dir = output_dir / "features"

    audio_processor = AudioProcessor(sr=sr)
    dsp = DSPEngine()
    rng = random.Random(100000 + seed * 1000003 + idx * 7919)

    try:
        # 1. Фрагменты длиной в окно модели (~6 с, 520 мел-кадров) на -23 LUFS
        duration = window_seconds(sr)
        y_dry = _load_chunk(src_file, sr, duration, rng)
        y_ref_dry = y_dry if ref_file == src_file else _load_chunk(ref_file, sr, duration, rng)

        # 2. Генерация цепочки плагинов (метка модели) и, с вероятностью harmony_ratio, слоев даблов/гармоний,
        # которые есть только в рендере референса и в метку не входят
        chain = CHAIN_PRIORS[prior](rng)
        layers = sample_reference_layers(rng) if rng.random() < harmony_ratio else []
        render_chain = with_reference_layers(chain, layers) if layers else chain

        # 3. Оффлайн-рендеринг в стерео (реверб дает ширину) БЕЗ скрытого обрезания параметров
        rand_key = rng.randint(0, 11)
        rand_scale = rng.choice(["major", "minor"])
        impls: List[str] = []
        if own_prob is None:
            y_wet = dsp.apply_chain(
                y_ref_dry,
                sr,
                render_chain,
                target_key_index=rand_key,
                target_scale=rand_scale,
                apply_safety_clamps=False
            )
        else:
            # Отдельный генератор: остальная случайность сэмпла (и метка) та же, что без рандомизации
            variant_rng = random.Random(200000 + seed * 1000003 + idx * 7919)
            y_wet, impls = _effect_variants(sr, own_prob).apply_chain(
                y_ref_dry, render_chain, variant_rng, target_key_index=rand_key, target_scale=rand_scale)

        # Легкая имитация полосы пропускания Demucs/MP3 на 25% тренировочных референсов (Domain Adaptation)
        y_wet_for_features = y_wet
        if split_name == "train" and rng.random() < 0.25:
            cutoff_demucs = rng.uniform(15500.0, 18000.0)
            sos_lp = signal.butter(4, cutoff_demucs, btype='lowpass', fs=sr, output='sos')
            y_wet_for_features = signal.sosfilt(sos_lp, y_wet).astype(np.float32)
        # Референс на инференсе нормализуется к -23 LUFS, поэтому и признаки считаются на этом уровне
        y_wet_for_features = normalize_lufs(y_wet_for_features, sr, MODEL_LUFS)

        # 4. Опциональное сохранение аудиофайлов на диск
        sample_id = f"{idx:06d}"
        dry_name = f"{source_name}_{sample_id}_dry.wav"
        wet_name = f"{source_name}_{sample_id}_wet.wav"

        dry_path = dry_out_dir / dry_name
        wet_path = wet_out_dir / wet_name

        if write_wav:
            sf.write(dry_path, y_dry.T if y_dry.ndim == 2 else y_dry, sr)
            sf.write(wet_path, y_wet.T if y_wet.ndim == 2 else y_wet, sr)

        # 5. Извлечение двухканальных Mid/Side Мел-спектрограмм (2, 128, T) и 18 физических DSP-признаков
        mel_dry, desc_dry = signal_features(y_dry, audio_processor)
        mel_wet, desc_wet = signal_features(y_wet_for_features, audio_processor)

        # Мел-спектрограммы в [-1, 1] хранятся в float16: точности хватает, а датасет вдвое меньше
        feature_file = features_out_dir / f"{sample_id}.npz"
        np.savez_compressed(
            feature_file,
            mel_dry=mel_dry.astype(np.float16),
            mel_wet=mel_wet.astype(np.float16),
            desc_dry=desc_dry,
            desc_wet=desc_wet
        )

        # 6. Сохранение метаданных сэмпла
        chain_list = [p.name for p in chain.plugins]
        chain_onehot = [1 if p in chain_list else 0 for p in CHAIN_ORDER]

        params_dict = {plugin.name: plugin.params for plugin in chain.plugins}

        flat_vector = [0.0] * PARAMETER_VECTOR_LENGTH
        for plugin in chain.plugins:
            if plugin.name not in PARAM_LAYOUT:
                continue
            offset = PARAM_OFFSETS[plugin.name]
            p_names = PARAM_LAYOUT[plugin.name]
            for j, p_name in enumerate(p_names):
                flat_vector[offset + j] = plugin.params.get(p_name, 0.0)

        sample = DatasetSample(
            id=sample_id,
            dry_path=str(Path("audio/dry") / dry_name),
            wet_path=str(Path("audio/wet") / wet_name),
            source=source_name,
            duration_sec=duration,
            chain=chain_list,
            chain_onehot=chain_onehot,
            params=params_dict,
            params_vector=flat_vector,
            split=split_name,
            src_file=str(src_file),
            ref_file=str(ref_file),
            ref_layers=[layer.params for layer in layers],
            ref_impls=impls
        )
        return sample.to_jsonl_line()
    except Exception as e:
        print(f"\n[-] Ошибка генерации сэмпла {idx} из файла {src_file.name}: {e}")
        return None
