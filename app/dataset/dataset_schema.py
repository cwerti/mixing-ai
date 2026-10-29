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
            "drive_db": (4.0, 20.0),              # От теплой сатурации до яркого перегруза
        }
    },
    "chorus": {
        "fl_name": "Fruity Chorus",
        "params": {
            "rate_hz": (0.4, 4.0),
            "depth": (0.10, 0.45),
            "feedback": (0.0, 0.40),
            "mix": (0.12, 0.45),
        }
    },
    "stereo_enhancer": {
        "fl_name": "Fruity Stereo Enhancer",
        "params": {
            "delay_ms": (6.0, 25.0),
            "width": (0.25, 1.0),
        }
    },
    "reverb": {
        "fl_name": "Fruity Reeverb 2",
        "params": {
            "room_size": (0.15, 0.95),
            "damping": (0.10, 0.90),
            "wet_level": (0.06, 0.36),            # Реальный физический уровень Wet без скрытого дополнительного деления
            "dry_level": (0.65, 1.0),
        }
    },
    "delay": {
        "fl_name": "Fruity Delay 3",
        "params": {
            "delay_seconds": (0.08, 0.50),
            "feedback": (0.10, 0.55),
            "mix": (0.08, 0.35),
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
