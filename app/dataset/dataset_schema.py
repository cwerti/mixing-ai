from dataclasses import dataclass, field
from typing import Dict, List, Any
import json

# Порядок применения плагинов к аудиосигналу.
# Фиксированный порядок гарантирует простоту обработки некоммутативности эффектов.
CHAIN_ORDER = ["eq", "compressor", "deesser", "distortion", "chorus", "reverb", "delay", "resonance_suppressor", "exciter"]

# Безопасные физические и нормализованные диапазоны параметров плагинов для генерации датасета.
# Обновлено до 7 полос эквалайзера с акустически перекрывающимися частотными диапазонами.
DATAGEN_PLUGINS = {
    "eq": {
        "fl_name": "Fruity Parametric EQ 2",
        "params": {
            "band1_freq_hz": (10.0, 100.0),       # Саб-бас / Низкочастотный гул
            "band1_gain_db": (-12.0, 12.0),
            "band2_freq_hz": (80.0, 350.0),       # Бас / теплота вокала
            "band2_gain_db": (-12.0, 12.0),
            "band3_freq_hz": (200.0, 1000.0),     # Низкая середина / тело вокала
            "band3_gain_db": (-12.0, 12.0),
            "band4_freq_hz": (600.0, 3000.0),     # Средние частоты / гнусавость
            "band4_gain_db": (-12.0, 12.0),
            "band5_freq_hz": (1500.0, 6000.0),    # Высокая середина / разборчивость и присутствие
            "band5_gain_db": (-12.0, 12.0),
            "band6_freq_hz": (4000.0, 12000.0),   # Высокие частоты / сибилянты
            "band6_gain_db": (-12.0, 12.0),
            "band7_freq_hz": (8000.0, 20000.0),   # Воздух / блеск
            "band7_gain_db": (-12.0, 12.0),
        }
    },
    "compressor": {
        "fl_name": "Fruity Compressor",
        "params": {
            "threshold_db": (-40.0, 0.0),
            "ratio": (1.0, 10.0),
            "attack_ms": (0.1, 100.0),
            "release_ms": (10.0, 1000.0),
        }
    },
    "deesser": {
        "fl_name": "Fruity Limiter (De-esser)",
        "params": {
            "threshold_db": (-40.0, 0.0),
            "ratio": (1.0, 6.0),
        }
    },
    "distortion": {
        "fl_name": "Fruity Fast Dist",
        "params": {
            "drive_db": (0.0, 20.0),              # От легкой теплоты/сатурации до перегруза
        }
    },
    "chorus": {
        "fl_name": "Fruity Chorus",
        "params": {
            "rate_hz": (0.5, 5.0),
            "depth": (0.0, 1.0),
            "feedback": (0.0, 0.5),
            "mix": (0.0, 0.8),
        }
    },
    "reverb": {
        "fl_name": "Fruity Reeverb 2",
        "params": {
            "room_size": (0.0, 1.0),
            "damping": (0.0, 1.0),
            "wet_level": (0.0, 0.4),              # Ограничено для предотвращения чрезмерного "замыливания" вокала
            "dry_level": (0.6, 1.0),
        }
    },
    "delay": {
        "fl_name": "Fruity Delay 3",
        "params": {
            "delay_seconds": (0.05, 0.5),
            "feedback": (0.0, 0.5),
            "mix": (0.0, 0.3),
        }
    },
    "resonance_suppressor": {
        "fl_name": "Fruity Limiter (De-res)",
        "params": {}
    },
    "exciter": {
        "fl_name": "Fruity Fast Dist (Exciter)",
        "params": {}
    }
}

# Официальная структура размещения параметров в плоском векторе
# Эквалайзер исключен из вектора обучения нейросети, так как вычисляется математически.
PARAM_LAYOUT = {
    "compressor": ["threshold_db", "ratio", "attack_ms", "release_ms"],
    "deesser": ["threshold_db", "ratio"],
    "distortion": ["drive_db"],
    "chorus": ["rate_hz", "depth", "feedback", "mix"],
    "reverb": ["room_size", "damping", "wet_level", "dry_level"],
    "delay": ["delay_seconds", "feedback", "mix"]
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
    name: str  # Короткое имя плагина (например, "eq", "compressor")
    params: Dict[str, float] = field(default_factory=dict)  # Карта: имя параметра -> значение

@dataclass
class ChainConfig:
    """Конфигурация цепочки эффектов."""
    plugins: List[PluginConfig] = field(default_factory=list)

@dataclass
class DatasetSample:
    """Описывает единичный сэмпл датасета с аудиофайлами и их параметрами обработки."""
    id: str
    dry_path: str
    wet_path: str
    source: str  # Источник данных ("vctk" or "soundcloud")
    duration_sec: float
    chain: List[str]  # Список активных плагинов в цепочке
    chain_onehot: List[int]  # Мультилейбл кодирование активных плагинов (one-hot)
    params: Dict[str, Dict[str, float]]  # Значения параметров по плагинам
    params_vector: List[float]  # Плоский вектор параметров (неактивные заполнены 0.0)

    def to_jsonl_line(self) -> str:
        """Сериализует объект сэмпла в строку JSONL."""
        return json.dumps(self.__dict__)

    @classmethod
    def from_jsonl_line(cls, line: str) -> "DatasetSample":
        """Десериализует объект сэмпла из строки JSONL."""
        data = json.loads(line)
        return cls(**data)
