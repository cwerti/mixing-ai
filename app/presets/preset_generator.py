import numpy as np
from pathlib import Path

class PresetGenerator:
    """
    Класс для эвристического маппинга параметров эквалайзера.
    Предоставляет функции расчета частот и усилений для Fruity Parametric EQ 2
    на основе кривой разницы АЧХ (delta).
    """

    def __init__(self):
        # Константы для Fruity Parametric EQ 2
        self.MIN_FREQ = 10.0
        self.MAX_FREQ = 20000.0
        self.MIN_GAIN = -18.0
        self.MAX_GAIN = 18.0

    def freq_to_param(self, freq: float) -> float:
        """
        Маппинг физической частоты в нормализованный параметр FL Studio (логарифмическая шкала).
        
        Args:
            freq: Частота в Гц.
            
        Returns:
            float: Нормализованный параметр [0..1].
        """
        freq = np.clip(freq, self.MIN_FREQ, self.MAX_FREQ)
        return (np.log10(freq) - np.log10(self.MIN_FREQ)) / (np.log10(self.MAX_FREQ) - np.log10(self.MIN_FREQ))

    def gain_to_param(self, gain: float) -> float:
        """
        Маппинг усиления в децибелах в нормализованный параметр FL Studio (линейная шкала).
        
        Args:
            gain: Усиление в dB.
            
        Returns:
            float: Нормализованный параметр [0..1].
        """
        gain = np.clip(gain, self.MIN_GAIN, self.MAX_GAIN)
        return (gain - self.MIN_GAIN) / (self.MAX_GAIN - self.MIN_GAIN)

    def extract_key_bands(self, freqs: np.ndarray, delta: np.ndarray, num_bands: int = 7) -> list:
        """
        Извлекает ключевые полосы (частоту и гейн) из сглаженной кривой дельты АЧХ.
        Использует логарифмическую сетку частот от 50 Гц до 16000 Гц для равномерного
        распределения полос эквалайзера (бас, середина, верха).
        
        Args:
            freqs: Частотная ось.
            delta: Дельта АЧХ в dB.
            num_bands: Количество выделяемых полос (по умолчанию 7).
            
        Returns:
            list: Список кортежей [(freq, gain), ...] для эквалайзера.
        """
        mask = (freqs >= 50) & (freqs <= 16000)
        f_filtered = freqs[mask]
        d_filtered = delta[mask]
        
        if len(f_filtered) == 0:
            return [(1000.0, 0.0)] * num_bands

        # Логарифмические границы для деления спектра на 7 полос
        log_boundaries = np.geomspace(50.0, 16000.0, num_bands + 1)
        bands = []
        
        for i in range(num_bands):
            f_min = log_boundaries[i]
            f_max = log_boundaries[i+1]
            
            # Маска для выбора частот из текущего логарифмического интервала
            seg_mask = (f_filtered >= f_min) & (f_filtered <= f_max)
            if np.sum(seg_mask) > 0:
                segment_d = d_filtered[seg_mask]
                # Центральная геометрическая (логарифмическая) частота интервала
                avg_freq = np.sqrt(f_min * f_max)
                avg_gain = np.mean(segment_d)
            else:
                avg_freq = np.sqrt(f_min * f_max)
                avg_gain = 0.0
                
            bands.append((avg_freq, avg_gain))
        return bands

    def generate_one_liner(self, bands: list) -> str:
        """
        Генерирует однострочный скрипт Python для прямой вставки настроек в FL Studio 25.
        
        Args:
            bands: Список кортежей полос [(freq, gain), ...].
            
        Returns:
            str: Скрипт-однострочник.
        """
        band_data = ",".join([f"({f:.0f}, {g:.2f})" for f, g in bands])
        
        # Используем проверенный синтаксис FL Studio Python API: plugins.setParamValue(val, idx, tr, slot)
        script = (
            f"import plugins, mixer, math; tr = mixer.trackNumber(); "
            f"b = [{band_data}]; "
            f"[(plugins.setParamValue((math.log10(f)-1)/3.301, i*5, tr, 0), "
            f"plugins.setParamValue((g+18)/36, i*5+1, tr, 0)) "
            f"for i, (f, g) in enumerate(b) if plugins.isValid(tr, 0)]"
        )
        return script
