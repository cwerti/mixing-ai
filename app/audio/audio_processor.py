import librosa
import numpy as np
import pyloudnorm as pyln
from pathlib import Path

def db_to_amplitude(db: float) -> float:
    """
    Конвертирует значение в децибелах (dB) в линейную амплитуду.
    
    Args:
        db: Значение в децибелах.
        
    Returns:
        float: Линейное значение амплитуды.
    """
    return 10 ** (db / 20.0)

class AudioProcessor:
    """
    Класс для обработки аудиосигнала вокала.
    Предоставляет функции загрузки, удаления тишины, нормализации по LUFS
    и извлечения спектральных характеристик вокала.
    """

    def __init__(self, sr: int = 44100):
        """
        Инициализирует аудиопроцессор с заданной частотой дискретизации.
        
        Args:
            sr: Частота дискретизации аудио. По умолчанию 44100 Гц.
        """
        self.sr = sr

    def load_audio(self, file_path: str | Path, target_lufs: float = -23.0, trim_silence: bool = True, top_db: float = 60.0) -> np.ndarray:
        """
        Загружает аудиофайл, опционально обрезает тишину в начале и конце
        и нормализует интегрированную громкость к целевому уровню LUFS.
        
        Args:
            file_path: Путь к аудиофайлу.
            target_lufs: Целевой уровень громкости в LUFS. По умолчанию -23.0.
            trim_silence: Обрезать ли тишину на концах аудиозаписи.
            top_db: Порог в dB ниже пика для определения тишины.
            
        Returns:
            np.ndarray: Одномерный массив нормализованного аудиосигнала.
        """
        y, sr = librosa.load(file_path, sr=self.sr)
        if trim_silence:
            y, _ = librosa.effects.trim(y, top_db=top_db)
        
        # Нормализация громкости по стандарту LUFS с помощью pyloudnorm
        meter = pyln.Meter(self.sr)
        try:
            loudness = meter.integrated_loudness(y)
            if not np.isnan(loudness) and not np.isinf(loudness):
                y = pyln.normalize.loudness(y, loudness, target_lufs)
        except Exception as e:
            # Откат к пиковой нормализации, если расчет LUFS не удался (например, тихий/короткий файл)
            print(f"LUFS нормализация не удалась ({e}), переход на пиковую нормализацию.")
            y = librosa.util.normalize(y) * db_to_amplitude(-1.0)
            
        return y

    def get_spectral_envelope(self, y: np.ndarray, n_fft: int = 2048, window: str = 'hann') -> np.ndarray:
        """
        Вычисляет усредненный по времени спектр мощности (спектральную огибающую) в dB.
        Максимальное значение спектра нормализуется к 0 dB.
        
        Args:
            y: Аудиосигнал.
            n_fft: Длина окна FFT.
            window: Тип оконной функции.
            
        Returns:
            np.ndarray: Спектральная огибающая вокала.
        """
        if y.ndim == 2:
            y = np.mean(y, axis=0)
        S = np.abs(librosa.stft(y, n_fft=n_fft, window=window))
        # Среднее значение по временной шкале
        avg_spectrum = np.mean(S, axis=1)
        # Перевод в децибелы с привязкой к максимуму
        avg_db = librosa.amplitude_to_db(avg_spectrum, ref=np.max)
        return avg_db

    def get_mel_spectrogram(self, y: np.ndarray, n_fft: int = 2048, hop_length: int = 512, n_mels: int = 128, window: str = 'hann', pitch_normalize: bool = False) -> np.ndarray:
        """
        Вычисляет Мел-спектрограмму в децибелах (dB).
        Максимум нормализуется к 0 dB.
        
        Args:
            y: Аудиосигнал.
            n_fft: Длина окна FFT.
            hop_length: Шаг сдвига окна.
            n_mels: Количество Мел-фильтров.
            window: Тип оконной функции.
            pitch_normalize: Если True, нормализует высоту тона (для Nightcore референсов).
            
        Returns:
            np.ndarray: Мел-спектрограмма.
        """
        if y.ndim == 2:
            y = np.mean(y, axis=0)
            
        if pitch_normalize:
            try:
                f0, _, voiced_prob = librosa.pyin(
                    y, 
                    fmin=librosa.note_to_hz('C2'), 
                    fmax=librosa.note_to_hz('C6'), 
                    sr=self.sr
                )
                valid_f0 = f0[(voiced_prob > 0.4) & ~np.isnan(f0) & (f0 > 0)]
                if len(valid_f0) > 0:
                    median_pitch = np.median(valid_f0)
                    if median_pitch > 310.0:
                        n_steps = 12.0 * np.log2(220.0 / median_pitch)
                        print(f"[*] Nightcore-детектор: обнаружен высокий тон ({median_pitch:.1f} Гц). Сдвигаем на {n_steps:.1f} полутонов вниз для анализа.")
                        y = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)
            except Exception as e:
                print(f"[!] Ошибка pitch normalization: {e}")
                
        S = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, window=window
        )
        return librosa.power_to_db(S, ref=np.max)

    def extract_features(self, y: np.ndarray) -> dict:
        """
        Извлекает ключевые характеристики аудиосигнала (динамический диапазон, пик-фактор, центроид).
        
        Args:
            y: Аудиосигнал.
            
        Returns:
            dict: Словарь с извлеченными признаками вокала.
        """
        if y.ndim == 2:
            y = np.mean(y, axis=0)
        # 1. Спектральный центроид (показывает тембральную яркость)
        centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr)
        mean_centroid = float(np.mean(centroid))

        # 2. Пик-фактор (Crest Factor): отношение максимального пика к RMS энергии
        rms_val = librosa.feature.rms(y=y)
        mean_rms = float(np.mean(rms_val))
        peak = float(np.max(np.abs(y)))
        crest_factor = (peak / mean_rms) if mean_rms > 1e-6 else 0.0

        # 3. Динамический диапазон (оценка разницы между 95-м и 5-м перцентилями RMS в dB)
        rms_db = librosa.amplitude_to_db(rms_val, ref=np.max)
        dynamic_range = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))

        return {
            "spectral_centroid_mean": mean_centroid,
            "crest_factor": crest_factor,
            "dynamic_range_db": dynamic_range,
            "rms_mean": mean_rms
        }

    def estimate_noise_gate_threshold(self, y: np.ndarray) -> float:
        """
        Математически оценивает шум в тихих паузах и рассчитывает
        оптимальный порог для Noise Gate в дБ.
        
        Args:
            y: Входной аудиосигнал.
            
        Returns:
            float: Оптимальный порог гейта в дБ (от -70.0 до -35.0).
        """
        frame_length = int(0.02 * self.sr)
        hop_length = frame_length // 2
        
        # Считаем RMS энергию
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        rms_db = 20 * np.log10(rms + 1e-6)
        
        # Находим 10-й перцентиль (уровень шума)
        noise_floor = float(np.percentile(rms_db, 10))
        
        # Задаем порог гейта чуть выше уровня шума (+6 дБ)
        threshold = noise_floor + 6.0
        
        # Ограничиваем в безопасных пределах
        threshold = np.clip(threshold, -70.0, -35.0)
        
        # Если запись идеально чистая, отключаем гейт (ставим -70 дБ)
        if noise_floor < -62.0:
            return -70.0
            
        return float(threshold)

class SpectralMatcher:
    """
    Класс для математического сопоставления АЧХ (Spectral Matching).
    Вычисляет разницу спектров сухого вокала и референса и строит
    оптимальную 7-полосную кривую для эквалайзера.
    """
    def __init__(self, sr: int = 44100):
        self.sr = sr

    def match_eq(self, y_dry: np.ndarray, y_ref: np.ndarray) -> list[tuple[float, float]]:
        """
        Сравнивает спектры y_dry и y_ref и рассчитывает физические частоты и гейны
        для 7 полос эквалайзера.
        
        Returns:
            list[tuple[float, float]]: Список из 7 кортежей (частота_гц, гейн_дб).
        """
        n_fft = 2048
        # Вычисление спектрограммы
        S_dry = np.abs(librosa.stft(y_dry, n_fft=n_fft))
        S_ref = np.abs(librosa.stft(y_ref, n_fft=n_fft))
        
        # Среднее значение по времени
        avg_dry = np.mean(S_dry, axis=1)
        avg_ref = np.mean(S_ref, axis=1)
        
        # Перевод в децибелы
        db_dry = 20 * np.log10(avg_dry + 1e-6)
        db_ref = 20 * np.log10(avg_ref + 1e-6)
        
        # Разница АЧХ (Delta)
        delta = db_ref - db_dry
        
        # Центрируем дельту, убирая изменение общей громкости
        delta = delta - np.mean(delta)
        
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
        
        # 7 непересекающихся диапазонов частот эквалайзера
        bands_ranges = [
            (10.0, 100.0),
            (80.0, 350.0),
            (200.0, 1000.0),
            (600.0, 3000.0),
            (1500.0, 6000.0),
            (4000.0, 12000.0),
            (8000.0, 20000.0)
        ]
        
        eq_bands = []
        for flow, fhigh in bands_ranges:
            indices = np.where((freqs >= flow) & (freqs <= fhigh))[0]
            if len(indices) == 0:
                freq = np.sqrt(flow * fhigh)
                gain = 0.0
            else:
                gain = float(np.mean(delta[indices]))
                # Поиск пиковой частоты в диапазоне
                peak_idx = indices[np.argmax(np.abs(delta[indices]))]
                freq = float(freqs[peak_idx])
                
            # Ограничение гейна в безопасном диапазоне
            gain = float(np.clip(gain, -12.0, 12.0))
            eq_bands.append((freq, gain))
            
        return eq_bands

if __name__ == "__main__":
    processor = AudioProcessor()
    print("Модуль AudioProcessor успешно инициализирован.")
