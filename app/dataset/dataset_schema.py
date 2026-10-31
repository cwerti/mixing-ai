from dataclasses import dataclass, field
from typing import Dict, List, Any
import json

# Порядок применения плагинов к аудиосигналу (11 классов: аналитический EQ + 10 нейросетевых эффектов стиля).
# resonance_suppressor и noise_gate вынесены в детерминированную пре-очистку исходника (Source Hygiene).
CHAIN_ORDER = [
    "eq",
    "pitch_corrector",
    "compressor",
    "deesser",
    "multiband_compressor",
    "distortion",
    "chorus",
    "stereo_enhancer",
    "reverb",
    "delay",
    "exciter"
]

# Плагины, которые предсказывает модель: порядок и длина задают размерности ее голов (классы и вектор
# параметров), поэтому список нельзя менять без переобучения — чекпоинт data/models/v4 обучен на этих 11.
MODEL_PLUGINS = CHAIN_ORDER

# Плагины только для рендера: DSPEngine их обрабатывает, но модель их не предсказывает, а оптимизатор
# не подбирает (в метрике стиля нет признаков высоты слоя). Включаются, только если цепочка/архетип
# задает их явно (ChainOptimizer.optimize(fixed_plugins=...)).
RENDER_ONLY_PLUGINS = ["harmony"]

# Полный порядок рендера. Гармония — параллельная шина: сдвигается уже обработанный инсертами лидера сигнал
# (EQ, тюн, компрессия, сатурация), а сумма лидера и слоя идет в общие реверб и дилей, как send-эффекты в миксе.
_SEND_START = CHAIN_ORDER.index("reverb")
RENDER_ORDER = CHAIN_ORDER[:_SEND_START] + RENDER_ONLY_PLUGINS + CHAIN_ORDER[_SEND_START:]

# Порог вероятности модели, с которого плагин включается в цепочку. Один на все места, где из вероятностей
# собирается цепочка, иначе бенчмарк, process_vocal и оптимизатор меряли бы разные решения модели.
# Перебор 0.5–0.8 на 20 синтетических парах (режим model): выше порога ложных включений меньше, но пропусков
# больше, и distance_true растет (0.789 -> 0.814), поэтому порог 0.5.
PLUGIN_ON_PROB = 0.5

# Количество явных физических DSP-дескрипторов одного аудиосигнала (AudioProcessor.extract_mixing_descriptors)
NUM_DSP_DESCRIPTORS = 18

# Музыкально выверенные и гарантированно слышимые диапазоны параметров плагинов.
# Нижние границы выбраны так, чтобы активный плагин (1 в one-hot) всегда вносил реальный вклад в сигнал,
# а верхние границы не требовали скрытого обрезания (label в датасете == 100% реальный рендер).
DATAGEN_PLUGINS = {
    "eq": {
        "fl_name": "Fruity Parametric EQ 2",
        "params": {
            "band1_freq_hz": (20.0, 100.0),       # Саб-бас / Низкочастотный гул
            "band1_gain_db": (-12.0, 12.0),
            "band2_freq_hz": (80.0, 350.0),       # Бас / теплота вокала
            "band2_gain_db": (-12.0, 12.0),
            "band3_freq_hz": (200.0, 1000.0),     # Низкая середина / тело вокала
            "band3_gain_db": (-12.0, 12.0),
            "band4_freq_hz": (600.0, 3000.0),     # Средние частоты / присутствие
            "band4_gain_db": (-12.0, 12.0),
            "band5_freq_hz": (1500.0, 6000.0),    # Высокая середина / разборчивость
            "band5_gain_db": (-12.0, 12.0),
            "band6_freq_hz": (4000.0, 12000.0),   # Высокие частоты / сибилянты
            "band6_gain_db": (-12.0, 12.0),
            "band7_freq_hz": (8000.0, 20000.0),   # Воздух / блеск
            "band7_gain_db": (-12.0, 12.0),
        }
    },
    "pitch_corrector": {
        "fl_name": "Pitcher",
        "params": {
            "speed": (0.25, 1.0),                 # От мягкой коррекции (0.25) до жесткого T-Pain/Hyperpop тюна (1.0)
        }
    },
    "compressor": {
        "fl_name": "Fruity Compressor",
        "params": {
            "threshold_db": (-36.0, -8.0),        # Гарантированное срабатывание на нормализованном к -23 LUFS вокале
            "ratio": (1.8, 10.0),
            "attack_ms": (1.0, 80.0),
            "release_ms": (20.0, 600.0),
        }
    },
    "deesser": {
        "fl_name": "Fruity Limiter (De-esser)",
        "params": {
            "threshold_db": (-36.0, -10.0),
            "ratio": (2.0, 8.0),
        }
    },
    "multiband_compressor": {
        "fl_name": "Fruity Multiband Compressor",
        "params": {
            "depth": (0.20, 0.85),                # Отчетливо слышимая 3-полосная OTT-динамика
        }
    },
    "distortion": {
        "fl_name": "Fruity Fast Dist",
        "params": {
            # От почти чистой сатурации до яркого перегруза. Нижняя граница 0 дБ (была 4 дБ до модели v5): в стоковых
            # цепях сатурация — «соус» (Blood Overdrive почти выключен, Soundgoodizer ~10 %), а с минимумом 4 дБ
            # почти чистой сатурации в обучении не было как класса (docs/research/calibration.md, §4)
            "drive_db": (0.0, 20.0),
        }
    },
    "chorus": {
        "fl_name": "Fruity Chorus",
        "params": {
            "rate_hz": (0.4, 4.0),
            "depth": (0.10, 0.45),
            "feedback": (0.0, 0.40),
            # Нижняя граница 0.05 (была 0.12 до модели v5): хорус «на ширину» подмешивают около 7 %
            # (docs/research/videos/SUMMARY.md, п. 7 «Что это значит для Mixing-AI»)
            "mix": (0.05, 0.45),
        }
    },
    "stereo_enhancer": {
        "fl_name": "Fruity Stereo Enhancer",
        "params": {
            "delay_ms": (6.0, 25.0),
            "width": (0.25, 1.0),
            # Срез верха у расширения (параметр оптимизатора, не модели): у реальных миксов верх уже середины
            # (Whiptails: Side на 3–8 кГц на 33 дБ ниже Mid, у Хааса без среза — на 16 дБ)
            "high_cut_hz": (2000.0, 20000.0),
        }
    },
    "reverb": {
        "fl_name": "Fruity Reeverb 2",
        "params": {
            "room_size": (0.15, 0.99),
            "damping": (0.10, 0.90),
            # Потолок поднят с 0.36: на Whiptails оптимизатор упирался в него, а хвост в паузах был на 30 дБ тише
            "wet_level": (0.06, 0.60),            # Реальный физический уровень Wet без скрытого дополнительного деления
            "dry_level": (0.65, 1.0),
            # Параметры send-шины (REVERB_SEND_PARAMS): не входят в PARAM_LAYOUT модели, их подбирает оптимизатор
            "predelay_ms": (0.0, 80.0),
            "low_cut_hz": (20.0, 1000.0),
            "high_cut_hz": (3000.0, 20000.0),
            "decay_s": (0.3, 20.0),               # RT60 ревера со свёрткой (decay_reverb); без него — Freeverb
        }
    },
    "delay": {
        "fl_name": "Fruity Delay 3",
        "params": {
            "delay_seconds": (0.08, 0.50),
            "feedback": (0.10, 0.55),
            # Нижняя граница 0 (была 0.08 до модели v7): в send-режиме это уровень повторов, и тихие повторы
            # «на подложку» должны быть в диапазоне
            "mix": (0.0, 0.35),
            # Параметры send-шины (DELAY_SEND_PARAMS), подбирает оптимизатор: разнос времени L/R (левый ×(1+s),
            # правый ×(1-s), Whiptails: 356/322 мс) и фильтры на повторах
            "stereo_spread": (0.0, 0.25),
            "low_cut_hz": (20.0, 1000.0),
            "high_cut_hz": (1500.0, 20000.0),
        }
    },
    "exciter": {
        "fl_name": "Fruity Fast Dist (Exciter)",
        "params": {
            "mix": (0.10, 0.35),
            "cutoff_hz": (5500.0, 10000.0),
        }
    },
    # Слой гармонии/дабла (только рендер, см. RENDER_ONLY_PLUGINS). Интервал в полутонах округляется до целого
    # в DSPEngine: гармония — это музыкальный интервал (+3, +5, +7, ±12), а «между нотами» отвечает detune_cents.
    # formant_shift — сдвиг формант относительно исходного голоса (0 = форманты сохранены, = semitones — без
    # сохранения, «бурундук»). mix — общий фейдер слоя поверх level_db (0 = слоя нет, сигнал не меняется).
    "harmony": {
        "fl_name": "Pitch Shifter (harmony layer)",
        "params": {
            "semitones": (-12.0, 12.0),
            "formant_shift": (-6.0, 6.0),
            "level_db": (-24.0, 0.0),             # Уровень слоя относительно лидера (по RMS)
            "delay_ms": (0.0, 30.0),              # Отставание дабла от лидера (5..30 мс — ручной дабл)
            "detune_cents": (0.0, 25.0),
            "width": (0.0, 1.0),                  # 0 — слой в центре, 1 — два голоса по краям (L/R)
            "mix": (0.0, 1.0),
        }
    },
    # Детерминированный модуль пре-очистки исходника (не предсказывается по чужому референсу)
    "resonance_suppressor": {
        "fl_name": "Fruity Limiter (De-res)",
        "params": {
            "threshold_db": (5.0, 10.0),
            "max_attenuation_db": (3.0, 15.0),
        }
    }
}

# Диапазоны, которые поменялись с моделью v5 (до нее drive_db 4–20 дБ, chorus.mix 0.12–0.45). Чекпоинты до v5
# не хранят диапазонов (ключ "param_ranges" в .pth), их нормализованные предсказания этих параметров
# пересчитываются в текущие диапазоны (MixingAIMLPipeline), иначе v4 рендерил бы другой физический drive.
LEGACY_PARAM_RANGES = {
    ("distortion", "drive_db"): (4.0, 20.0),
    ("chorus", "mix"): (0.12, 0.45),
    ("reverb", "room_size"): (0.15, 0.95),
    ("reverb", "wet_level"): (0.06, 0.36),
    ("delay", "mix"): (0.08, 0.35),
}

# Параметры ревера как send-шины: если хоть один задан, DSPEngine рендерит ревер параллельно с predelay и
# фильтрами на мокром сигнале; без них — прежний последовательный Reverb (цепочки модели звучат как раньше).
REVERB_SEND_PARAMS = ("predelay_ms", "low_cut_hz", "high_cut_hz", "decay_s")

# Параметры дилея как send-шины: если хоть один задан, повторы идут параллельно сухому (сухой не ослабляется),
# с разносом L/R и фильтрами в петле обратной связи; без них — прежний Delay из pedalboard.
DELAY_SEND_PARAMS = ("stereo_spread", "low_cut_hz", "high_cut_hz")

# Официальная структура размещения параметров в плоском векторе обучения нейросети (24 параметра).
# Эквалайзер рассчитывается через SpectralMatcher, а подавитель резонансов — по исходнику.
PARAM_LAYOUT = {
    "pitch_corrector": ["speed"],
    "compressor": ["threshold_db", "ratio", "attack_ms", "release_ms"],
    "deesser": ["threshold_db", "ratio"],
    "multiband_compressor": ["depth"],
    "distortion": ["drive_db"],
    "chorus": ["rate_hz", "depth", "feedback", "mix"],
    "stereo_enhancer": ["delay_ms", "width"],
    "reverb": ["room_size", "damping", "wet_level", "dry_level"],
    "delay": ["delay_seconds", "feedback", "mix"],
    "exciter": ["mix", "cutoff_hz"],
}

# Динамический расчет смещений и общей длины вектора параметров
PARAM_OFFSETS = {}
current_offset = 0
for plugin_name in CHAIN_ORDER:
    if plugin_name in PARAM_LAYOUT:
        PARAM_OFFSETS[plugin_name] = current_offset
        current_offset += len(PARAM_LAYOUT[plugin_name])
PARAMETER_VECTOR_LENGTH = current_offset


def model_param_ranges() -> Dict[str, Dict[str, List[float]]]:
    """Физические диапазоны параметров модели (сохраняются в чекпоинт, чтобы знать, в каких он обучен)."""
    return {name: {p: list(DATAGEN_PLUGINS[name]["params"][p]) for p in layout} for name, layout in PARAM_LAYOUT.items()}


# dataset_v4 сгенерирован до подъема потолков реверба (wet_level 0.06–0.36, room_size 0.15–0.95), но уже с
# диапазонами drive/chorus.mix модели v5; в его stats.json диапазоны не записаны
DATASET_V4_PARAM_RANGES = {name: {p: list(LEGACY_PARAM_RANGES[(name, p)]) if name in ("reverb", "delay") and
                                  (name, p) in LEGACY_PARAM_RANGES else list(DATAGEN_PLUGINS[name]["params"][p])
                                  for p in layout} for name, layout in PARAM_LAYOUT.items()}


def dataset_param_ranges(dataset_dir) -> tuple[Dict[str, Dict[str, List[float]]], str]:
    """
    Диапазоны, в которых нормализованы метки датасета: stats.json["param_ranges"] (пишется генератором с
    датасета v5); для dataset_v4 — DATASET_V4_PARAM_RANGES; иначе — текущие (источник "schema", с этим нужно
    предупреждение: если схема менялась после генерации, чекпоинт сохранит неверные диапазоны).
    """
    from pathlib import Path
    stats_path = Path(dataset_dir) / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if "param_ranges" in stats:
            return stats["param_ranges"], "stats.json"
    if Path(dataset_dir).name == "dataset_v4":
        return DATASET_V4_PARAM_RANGES, "dataset_v4 (до подъема потолков реверба)"
    return model_param_ranges(), "schema"


def remap_param_vector(vec, ranges: Dict[str, Dict[str, List[float]]] | None):
    """
    Переводит нормализованный вектор параметров модели из диапазонов, в которых она обучена, в текущие
    DATAGEN_PLUGINS. ranges = None — чекпоинт до v5 (диапазоны LEGACY_PARAM_RANGES для измененных параметров).
    """
    out = vec.copy()
    for name, layout in PARAM_LAYOUT.items():
        for i, p in enumerate(layout):
            old = ranges[name][p] if ranges is not None else LEGACY_PARAM_RANGES.get((name, p))
            new = DATAGEN_PLUGINS[name]["params"][p]
            if old is None or tuple(old) == tuple(new):
                continue
            phys = old[0] + out[PARAM_OFFSETS[name] + i] * (old[1] - old[0])
            out[PARAM_OFFSETS[name] + i] = min(max((phys - new[0]) / (new[1] - new[0]), 0.0), 1.0)
    return out


@dataclass
class PluginConfig:
    """Конфигурация параметров отдельного плагина (нормализованные значения в [0..1])."""
    name: str
    params: Dict[str, float] = field(default_factory=dict)


@dataclass
class ChainConfig:
    """Конфигурация цепочки эффектов."""
    plugins: List[PluginConfig] = field(default_factory=list)


@dataclass
class DatasetSample:
    """Описывает единичный сэмпл датасета с аудиофайлами, сплитом и параметрами обработки."""
    id: str
    dry_path: str
    wet_path: str
    source: str
    duration_sec: float
    chain: List[str]
    chain_onehot: List[int]
    params: Dict[str, Dict[str, float]]
    params_vector: List[float]
    split: str = "train"
    # Исходные сухие файлы исходника и референса (у датасета v1 совпадают и не записаны)
    src_file: str = ""
    ref_file: str = ""
    # Слои harmony (даблы/гармонии), добавленные только в рендер референса: в метку модели не входят
    # (нормализованные параметры, как у DATAGEN_PLUGINS["harmony"]; у датасетов до v4 поля нет)
    ref_layers: List[Dict[str, float]] = field(default_factory=list)
    # Реализации плагинов в рендере референса при доменной рандомизации ("плагин:реализация",
    # app/audio/effect_variants.py); пусто — все плагины нашим DSPEngine (датасеты до v5)
    ref_impls: List[str] = field(default_factory=list)

    def to_jsonl_line(self) -> str:
        """Сериализует объект сэмпла в строку JSONL."""
        return json.dumps(self.__dict__)

    @classmethod
    def from_jsonl_line(cls, line: str) -> "DatasetSample":
        """Десериализует объект сэмпла из строки JSONL (с поддержкой старых записей без поля split)."""
        data = json.loads(line)
        if "split" not in data:
            data["split"] = "train"
        return cls(**data)
