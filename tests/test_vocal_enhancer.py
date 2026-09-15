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

def test_stereo_enhancer():
    """Проверяет корректность расчета стерео-ширины и работу стерео-расширителя Хааса."""
    sr = 22050
    enhancer = VocalEnhancer(sr=sr)
    
    # 1. Тест моно-сигнала
    y_mono = np.random.normal(0, 0.1, sr)
    width_mono = enhancer.measure_stereo_width(y_mono)
    assert width_mono == 0.0
    
    # 2. Тест стерео-расширения
    # Применяем экситер стерео к моно-сигналу
    y_stereo = enhancer.apply_stereo_enhancer(y_mono, delay_ms=15.0, width=1.0)
    
    # Должна получиться двумерная стерео-матрица (2, n_samples)
    assert y_stereo.ndim == 2
    assert y_stereo.shape == (2, len(y_mono))
    
    # Левый и правый каналы не должны быть идентичными
    assert not np.array_equal(y_stereo[0], y_stereo[1])
    
    # Измеряем ширину полученного стерео-сигнала
    width_stereo = enhancer.measure_stereo_width(y_stereo)
    assert width_stereo > 0.1  # Должна быть зафиксирована стерео-ширина

def test_key_detection_and_ott():
    """Проверяет детектор тональности и эмуляцию многополосного OTT компрессора."""
    sr = 22050
    enhancer = VocalEnhancer(sr=sr)
    
    # 1. Тест многополосного компрессора (OTT)
    y_mono = np.random.normal(0, 0.1, sr)
    y_ott = enhancer.apply_multiband_compressor(y_mono, depth=0.5)
    assert y_ott.shape == y_mono.shape
    
    # Проверяем на стерео-сигнале
    y_stereo = np.vstack([y_mono, y_mono * 0.9])
    y_stereo_ott = enhancer.apply_multiband_compressor(y_stereo, depth=0.5)
    assert y_stereo_ott.shape == y_stereo.shape
    
    # 2. Тест детектора тональности (проверка вызовов без исключений)
    # Генерируем синусоиду ля-мажор / A4 (440 Гц)
    t = np.linspace(0, 0.5, sr // 2)
    y_sine = np.sin(2 * np.pi * 440.0 * t)
    
    key, scale = enhancer.detect_key_and_scale(y_sine)
    assert isinstance(key, int)
    assert key in range(12)
    assert scale in ('major', 'minor')


