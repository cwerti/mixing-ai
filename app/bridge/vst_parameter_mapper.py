import json
import math
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

class VSTParameterMapper:
    """
    Класс для загрузки конфигурационных файлов плагинов (JSON)
    и преобразования абстрактных параметров (например, частоты в Гц)
    в нормализованные значения [0..1] и обратно.
    """

    def __init__(self, config_name: str, config_dir: Optional[Path] = None):
        """
        Инициализирует маппер параметров для плагина.
        
        Args:
            config_name: Имя плагина или прямой путь к конфигурационному файлу.
            config_dir: Папка с конфигурациями плагинов. По умолчанию app/data/plugin_configs/.
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "data" / "plugin_configs"
        
        config_path = Path(config_name)
        if not config_path.exists():
            config_path = config_dir / f"{config_name}.json"
            if not config_path.exists():
                raise FileNotFoundError(f"Файл конфигурации для плагина '{config_name}' не найден по пути: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config: Dict[str, Any] = json.load(f)

        self.plugin_id = self.config.get("plugin_id")
        self.display_name = self.config.get("display_name")
        self.category = self.config.get("category")
        self.vst_type = self.config.get("vst_type")
        self.capabilities = self.config.get("capabilities", {})
        self.mappings = self.config.get("parameter_mapping", {})

    def get_param_index(self, param_name: str) -> Optional[int]:
        """
        Возвращает индекс параметра (MIDI CC или внутренний ID) по имени параметра.
        
        Args:
            param_name: Имя параметра (например, 'band1_freq').
            
        Returns:
            Optional[int]: Индекс параметра или None, если параметр не найден.
        """
        param_cfg = self.mappings.get(param_name)
        return param_cfg.get("index") if param_cfg else None

    def map_physical_to_normalized(self, param_name: str, phys_val: float) -> float:
        """
        Преобразует физическое значение параметра (например, 440 Гц или -12 dB) в нормализованное [0.0..1.0].
        
        Args:
            param_name: Имя параметра.
            phys_val: Физическое значение параметра.
            
        Returns:
            float: Нормализованное значение параметра.
        """
        param_cfg = self.mappings.get(param_name)
        if not param_cfg:
            raise KeyError(f"Параметр '{param_name}' не найден в маппинге конфигурации для {self.plugin_id}.")

        mapping_type = param_cfg.get("type", "float")
        if mapping_type == "bool":
            return 1.0 if phys_val else 0.0

        range_cfg = param_cfg.get("range_mapping", {})
        min_phys = float(range_cfg.get("min_phys", 0.0))
        max_phys = float(range_cfg.get("max_phys", 1.0))
        mapping_mode = range_cfg.get("type", "linear")

        # Ограничиваем физическое значение рамками диапазона
        min_v = min(min_phys, max_phys)
        max_v = max(min_phys, max_phys)
        phys_val = float(np.clip(phys_val, min_v, max_v))

        if mapping_mode == "linear":
            denom = max_phys - min_phys
            return (phys_val - min_phys) / denom if denom != 0.0 else 0.0
        
        elif mapping_mode in ("logarithmic", "exponential"):
            if min_phys <= 0.0 or max_phys <= 0.0:
                raise ValueError("Границы логарифмического/экспоненциального маппинга должны быть строго положительными.")
            return math.log(phys_val / min_phys) / math.log(max_phys / min_phys)
        
        return 0.0

    def map_normalized_to_physical(self, param_name: str, norm_val: float) -> float:
        """
        Преобразует нормализованное значение [0.0..1.0] в соответствующее физическое значение (например, Гц, dB).
        
        Args:
            param_name: Имя параметра.
            norm_val: Нормализованное значение [0..1].
            
        Returns:
            float: Физическое значение параметра.
        """
        param_cfg = self.mappings.get(param_name)
        if not param_cfg:
            raise KeyError(f"Параметр '{param_name}' не найден в маппинге конфигурации для {self.plugin_id}.")

        mapping_type = param_cfg.get("type", "float")
        if mapping_type == "bool":
            return 1.0 if norm_val >= 0.5 else 0.0

        range_cfg = param_cfg.get("range_mapping", {})
        min_phys = float(range_cfg.get("min_phys", 0.0))
        max_phys = float(range_cfg.get("max_phys", 1.0))
        mapping_mode = range_cfg.get("type", "linear")

        norm_val = float(np.clip(norm_val, 0.0, 1.0))

        if mapping_mode == "linear":
            return min_phys + norm_val * (max_phys - min_phys)
        
        elif mapping_mode in ("logarithmic", "exponential"):
            if min_phys <= 0.0 or max_phys <= 0.0:
                raise ValueError("Границы логарифмического/экспоненциального маппинга должны быть строго положительными.")
            return min_phys * (max_phys / min_phys) ** norm_val

        return 0.0
