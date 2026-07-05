from dataclasses import dataclass, field
from typing import Dict, List, Any
import json

# Order in which plugins are applied to the audio signal.
# Fixed order guarantees that non-commutativity is handled simply.
CHAIN_ORDER = ["eq", "compressor", "reverb", "delay"]

# Safe ranges for randomized parameter generation (both physical and normalized)
DATAGEN_PLUGINS = {
    "eq": {
        "fl_name": "Fruity Parametric EQ 2",
        "params": {
            "band1_freq_hz": (80.0, 300.0),       # Bass band freq
            "band1_gain_db": (-12.0, 12.0),       # Bass band gain
            "band2_freq_hz": (300.0, 2000.0),     # Mid band freq
            "band2_gain_db": (-12.0, 12.0),       # Mid band gain
            "band3_freq_hz": (2000.0, 12000.0),   # High band freq
            "band3_gain_db": (-12.0, 12.0),       # High band gain
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
    "reverb": {
        "fl_name": "Fruity Reeverb 2",
        "params": {
            "room_size": (0.0, 1.0),
            "damping": (0.0, 1.0),
            "wet_level": (0.0, 0.4),              # Restrict to avoid complete washing out
            "dry_level": (0.6, 1.0),
        }
    },
    "delay": {
        "fl_name": "Fruity Delay 3",
        "params": {
            "delay_seconds": (0.05, 0.5),
            "feedback": (0.0, 0.5),
            "mix": (0.0, 0.3),                    # Restrict to keep it reasonable
        }
    }
}

@dataclass
class PluginConfig:
    name: str  # Short name (e.g. "eq", "compressor")
    params: Dict[str, float] = field(default_factory=dict)  # normalized parameters in [0.0, 1.0]

@dataclass
class ChainConfig:
    plugins: List[PluginConfig] = field(default_factory=list)

@dataclass
class DatasetSample:
    id: str
    dry_path: str
    wet_path: str
    source: str  # "vctk" or "soundcloud"
    duration_sec: float
    chain: List[str]  # Active plugin names
    chain_onehot: List[int]  # One-hot/multi-label encoding for the active plugins
    params: Dict[str, Dict[str, float]]  # Plugin short name -> parameter name -> normalized value
    params_vector: List[float]  # Flattened vector containing all parameters (zeroed for inactive ones)

    def to_jsonl_line(self) -> str:
        return json.dumps(self.__dict__)

    @classmethod
    def from_jsonl_line(cls, line: str) -> "DatasetSample":
        data = json.loads(line)
        return cls(**data)
