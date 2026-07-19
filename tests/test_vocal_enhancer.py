import pytest
import numpy as np
import scipy.signal as signal
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.audio.vocal_enhancer import VocalEnhancer

def test_vocal_enhancer_shapes():
    """Проверяет, что выходные массивы функций имеют правильный размер и тип данных."""
    sr = 22050
    enhancer = VocalEnhancer(sr=sr)
    
    # 2 секунды белого шума
    y = np.random.normal(0, 0.1, sr * 2)
    
    y_clean = enhancer.suppress_resonances(y)
    assert len(y_clean) == len(y)
    assert isinstance(y_clean, np.ndarray)
    
    y_excited = enhancer.apply_exciter(y)
    assert len(y_excited) == len(y)
    assert isinstance(y_excited, np.ndarray)

def test_suppress_resonances_attenuation():
    """Проверяет, что алгоритм действительно глушит сильные узкие резонансы."""
    sr = 22050
    enhancer = VocalEnhancer(sr=sr)
    
    # Создаем белый шум с подмешанным сильным синусоидальным свистом на частоте 1500 Гц
    t = np.linspace(0, 1.0, sr)
    noise = np.random.normal(0, 0.05, sr)
    whistle = 0.5 * np.sin(2 * np.pi * 1500 * t)
    y = noise + whistle
    
    # Измеряем амплитуду на частоте 1500 Гц до обработки
    f, psd_before = signal.welch(y, sr, nperseg=512)
    idx_1500 = np.abs(f - 1500).argmin()
    val_before = psd_before[idx_1500]
    
    # Обрабатываем
    y_clean = enhancer.suppress_resonances(y, threshold_db=5.0, max_attenuation_db=15.0)
    
    # Измеряем амплитуду после обработки
    _, psd_after = signal.welch(y_clean, sr, nperseg=512)
    val_after = psd_after[idx_1500]
    
    # Убеждаемся, что пик на 1500 Гц ослаблен
    assert val_after < val_before * 0.1  # Громкость пика должна упасть как минимум в 10 раз

def test_exciter_generates_harmonics():
    """Проверяет, что экситер действительно генерирует новые высокие частоты."""
    sr = 44100
    enhancer = VocalEnhancer(sr=sr)
    
    # Синусоида 1000 Гц (частот выше 1000 Гц изначально нет вообще)
    t = np.linspace(0, 0.5, sr // 2)
    y = np.sin(2 * np.pi * 1000 * t)
    
    # Фильтруем оригинальный сигнал ВЧ-фильтром (выше 10 кГц), чтобы убедиться, что там пусто
    sos = signal.butter(4, 10000.0, btype='highpass', fs=sr, output='sos')
    y_high_orig = signal.sosfilt(sos, y)
    rms_before = np.sqrt(np.mean(y_high_orig**2))
    
    # Применяем экситер с низкой частотой среза, чтобы насытить синусоиду
    y_excited = enhancer.apply_exciter(y, cutoff_hz=800, drive=5.0, mix=0.2)
    
    # Измеряем ВЧ-энергию выше 10 кГц после экситера
    y_high_excited = signal.sosfilt(sos, y_excited)
    rms_after = np.sqrt(np.mean(y_high_excited**2))
    
    # Должен произойти значительный прирост высоких частот за счет сгенерированных гармоник
    assert rms_after > rms_before + 1e-4
