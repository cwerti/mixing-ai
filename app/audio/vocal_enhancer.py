import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage
import librosa

class VocalEnhancer:
    """
    Класс для математического улучшения качества вокала без использования машинного обучения.
    Содержит алгоритмы спектрального подавления резонансов и гармонического возбуждения высоких частот (Air Exciter).
    """
    
    def __init__(self, sr: int = 44100):
        self.sr = sr

    def suppress_resonances(self, y: np.ndarray, threshold_db: float = 7.0, max_attenuation_db: float = 12.0) -> np.ndarray:
        """
        Динамическое подавление узкополосных спектральных резонансов (свист, комнатный гул).
        Анализирует кадры STFT, находит пики относительно медианной огибающей спектра
        и ослабляет их в частотной области.
        
        Args:
            y: Входной аудиосигнал (одномерный массив).
            threshold_db: Порог превышения пика над огибающей в дБ для детекции резонанса.
            max_attenuation_db: Максимальное ослабление резонанса в дБ.
            
        Returns:
            np.ndarray: Очищенный аудиосигнал.
        """
        if len(y) == 0:
            return y
            
        # Параметры STFT
        n_fft = 2048
        hop_length = 512
        
        # Получаем комплексный спектр
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # 1. Вычисляем сглаженную спектральную огибающую с помощью медианного фильтра по оси частот.
        # Это сглаживает спектральные пики, но сохраняет широкие форманты вокала.
        # Размер окна 17 бинов (~360 Гц при n_fft=2048 и sr=44100)
        smoothed_mag = ndimage.median_filter(magnitude, size=(17, 1), mode='reflect')
        
        # 2. Вычисляем отношение в дБ
        ratio_db = 20 * np.log10((magnitude + 1e-8) / (smoothed_mag + 1e-8))
        
        # 3. Находим маску резонансных частот (где отношение превышает порог)
        resonance_mask = ratio_db > threshold_db
        
        # 4. Рассчитываем коэффициенты ослабления для каждого бина
        attenuation_factors = np.ones_like(magnitude)
        excess = ratio_db[resonance_mask] - threshold_db
        attenuation_db = np.minimum(excess, max_attenuation_db)
        
        # Переводим дБ в линейный масштаб
        attenuation_factors[resonance_mask] = 10 ** (-attenuation_db / 20.0)
        
        # Применяем ослабление к спектру
        new_magnitude = magnitude * attenuation_factors
        
        # 5. Восстанавливаем комплексный спектр и делаем обратный STFT
        new_stft = new_magnitude * np.exp(1j * phase)
        y_clean = librosa.istft(new_stft, hop_length=hop_length, length=len(y))
        
        return y_clean

    def apply_exciter(self, y: np.ndarray, cutoff_hz: float = 7000.0, drive: float = 2.0, mix: float = 0.15) -> np.ndarray:
        """
        Гармонический экситер для генерации новых высоких частот (эффект Air / воздуха).
        Выделяет высокие частоты, насыщает их для генерации гармоник и подмешивает обратно.
        
        Args:
            y: Входной аудиосигнал.
            cutoff_hz: Частота среза ВЧ-фильтра для выделения насыщаемой полосы (Гц).
            drive: Сила сатурации (усиление сигнала перед нелинейным ограничением).
            mix: Коэффициент подмешивания сгенерированных гармоник (0.0 - 1.0).
            
        Returns:
            np.ndarray: Обогащенный гармониками сигнал.
        """
        if len(y) == 0 or mix <= 0.0:
            return y
            
        # 1. Выделяем высокие частоты с помощью фильтра Баттерворта ВЧ
        sos_high = signal.butter(4, cutoff_hz, btype='highpass', fs=self.sr, output='sos')
        y_high = signal.sosfilt(sos_high, y)
        
        # 2. Генерируем новые гармоники нелинейным искажением (кубическая сатурация + tanh)
        # Это создает новые спектральные составляющие в диапазоне 10-20 кГц
        y_harmonics = np.tanh(y_high * drive)
        
        # 3. Фильтруем результат еще раз, чтобы убрать просочившиеся низкие частоты (интермодуляцию)
        # Оставляем только чистый "воздух" выше 10 кГц
        sos_air = signal.butter(4, 10000.0, btype='highpass', fs=self.sr, output='sos')
        y_air = signal.sosfilt(sos_air, y_harmonics)
        
        # Нормализуем уровень гармоник по RMS, чтобы микс был предсказуемым
        rms_orig = np.sqrt(np.mean(y_high**2)) + 1e-8
        rms_air = np.sqrt(np.mean(y_air**2)) + 1e-8
        y_air = y_air * (rms_orig / rms_air)
        
        # 4. Смешиваем с исходным сигналом
        y_excited = y + mix * y_air
        
        return y_excited
