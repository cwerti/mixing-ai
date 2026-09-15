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

    def apply_stereo_enhancer(self, y: np.ndarray, delay_ms: float = 18.0, width: float = 1.0) -> np.ndarray:
        """
        Преобразует моно-сигнал в стерео с использованием задержки Хааса.
        Задерживает правый канал относительно левого, создавая ощущение пространства.
        Сохраняет моносовместимость низких частот.
        
        Args:
            y: Входной сигнал (1D моно-массив или 2D стерео-массив).
            delay_ms: Задержка в миллисекундах (обычно 12..25 мс для эффекта Хааса).
            width: Степень расширения стереобазы (0.0..1.0).
            
        Returns:
            np.ndarray: Стерео-аудиосигнал формы (2, n_samples).
        """
        # Приведение к стерео-массиву формы (2, n_samples)
        if y.ndim == 1:
            y_stereo = np.vstack([y, y])
        elif y.ndim == 2:
            if y.shape[0] == 1:
                y_stereo = np.vstack([y[0], y[0]])
            else:
                y_stereo = y.copy()
        else:
            raise ValueError(f"Неподдерживаемая размерность сигнала: {y.ndim}")
            
        if delay_ms <= 0.0 or width <= 0.0:
            return y_stereo
            
        delay_samples = int((delay_ms / 1000.0) * self.sr)
        if delay_samples <= 0 or y_stereo.shape[1] <= delay_samples:
            return y_stereo
            
        # Левый канал оставляем сухим, правый задерживаем
        left = y_stereo[0]
        right_delayed = np.zeros_like(y_stereo[1])
        right_delayed[delay_samples:] = y_stereo[1][:-delay_samples]
        
        # Разделяем задержанный сигнал на СЧ/ВЧ и НЧ (чтобы низ остался в моно и не потерял фазу)
        sos_hp = signal.butter(4, 150.0, btype='highpass', fs=self.sr, output='sos')
        right_high = signal.sosfilt(sos_hp, right_delayed)
        right_low = right_delayed - right_high # НЧ-составляющая
        
        # Смешиваем задержанный СЧ/ВЧ сигнал с исходным
        # Левый: исходный (моно)
        # Правый: НЧ от исходного (без задержки) + СЧ/ВЧ от задержанного
        y_stereo[1] = (right_low + (1 - width * 0.5) * left + (width * 0.5) * right_high)
        
        return y_stereo

    @staticmethod
    def measure_stereo_width(y: np.ndarray) -> float:
        """
        Вычисляет стерео-ширину сигнала.
        Возвращает значение от 0.0 (полное моно) до 1.0 (широкое стерео).
        
        Args:
            y: Входной сигнал (1D или 2D).
            
        Returns:
            float: Коэффициент стерео-ширины.
        """
        # Если сигнал одномерный или содержит 1 канал - это 100% моно
        if y.ndim == 1 or y.shape[0] == 1:
            return 0.0
            
        left = y[0]
        right = y[1]
        
        # Получаем Mid и Side составляющие
        mid = (left + right) / 2.0
        side = (left - right) / 2.0
        
        rms_mid = np.sqrt(np.mean(mid**2)) + 1e-8
        rms_side = np.sqrt(np.mean(side**2)) + 1e-8
        
        # Рассчитываем стерео-ширину на основе пропорции Side к общей энергии
        width = rms_side / (rms_mid + rms_side)
        
        # Нормализуем так, чтобы равное распределение Mid/Side энергии давало 1.0
        return float(np.clip(width * 2.0, 0.0, 1.0))

    def detect_key_and_scale(self, y: np.ndarray) -> tuple[int, str]:
        """
        Математический детектор тональности вокала (Key & Scale).
        Использует pYIN для F0 и сравнивает Pitch Class Histogram
        с профилями Krumhansl-Schmuckler.
        
        Args:
            y: Входной аудиосигнал.
            
        Returns:
            tuple[int, str]: (индекс тоники 0-11: C-B, лад 'major' или 'minor').
        """
        # Превращаем в моно
        if y.ndim == 2:
            y = np.mean(y, axis=0)
            
        if len(y) == 0:
            return 0, 'minor'
            
        try:
            # Извлекаем F0 (Pitch) с помощью алгоритма pYIN
            # Нормализуем пиковую амплитуду для надежного анализа
            peak_val = np.max(np.abs(y))
            y_detect = y / peak_val if peak_val > 1e-4 else y
            
            f0, _, voiced_prob = librosa.pyin(
                y_detect, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C6'), 
                sr=self.sr
            )
            voiced_flag = (voiced_prob > 0.35)
            
            f0_clean = f0[voiced_flag & ~np.isnan(f0)]
            if len(f0_clean) == 0:
                return 0, 'minor'
                
            # Переводим частоты в MIDI-ноты и строим гистограмму нот (0-11)
            midi_notes = librosa.hz_to_midi(f0_clean)
            pitch_classes = np.round(midi_notes).astype(int) % 12
            histogram, _ = np.histogram(pitch_classes, bins=np.arange(13), density=True)
            
            # Профили Krumhansl-Schmuckler для сопоставления ладов
            major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
            
            major_profile /= np.sum(major_profile)
            minor_profile /= np.sum(minor_profile)
            
            best_corr = -1.0
            best_key = 0
            best_scale = 'minor'
            
            for key in range(12):
                shifted_hist = np.roll(histogram, -key)
                
                corr_maj = np.corrcoef(shifted_hist, major_profile)[0, 1]
                if corr_maj > best_corr:
                    best_corr = corr_maj
                    best_key = key
                    best_scale = 'major'
                    
                corr_min = np.corrcoef(shifted_hist, minor_profile)[0, 1]
                if corr_min > best_corr:
                    best_corr = corr_min
                    best_key = key
                    best_scale = 'minor'
                    
            return best_key, best_scale
        except Exception:
            return 0, 'minor'

    def apply_multiband_compressor(self, y: np.ndarray, depth: float = 0.5) -> np.ndarray:
        """
        Эмулирует OTT (3-полосный агрессивный Upward/Downward компрессор).
        Разделяет сигнал на Low (<150 Hz), Mid (150-2500 Hz), High (>2500 Hz),
        применяет быстрое восходящее и нисходящее сжатие и суммирует обратно.
        
        Args:
            y: Входной аудиосигнал.
            depth: Интенсивность сжатия (mix) от 0.0 до 1.0 (0.5 = 50% OTT).
            
        Returns:
            np.ndarray: Сжатый аудиосигнал.
        """
        if len(y) == 0 or depth <= 0.0:
            return y
            
        is_stereo = (y.ndim == 2 and y.shape[0] == 2)
        
        # 1. Создаем кроссоверные фильтры Баттерворта
        sos_low = signal.butter(4, 150.0, btype='lowpass', fs=self.sr, output='sos')
        sos_high = signal.butter(4, 2500.0, btype='highpass', fs=self.sr, output='sos')
        
        # Разделяем на 3 полосы
        y_low = signal.sosfilt(sos_low, y)
        y_high = signal.sosfilt(sos_high, y)
        y_mid = y - y_low - y_high
        
        bands = [y_low, y_mid, y_high]
        compressed_bands = []
        
        # Параметры компрессии для полос (Low, Mid, High)
        # Пороги (дБ) и соотношения (Ratios)
        threshold_down = -20.0  # Нисходящий порог (пики)
        ratio_down = 4.0
        
        threshold_up = -35.0    # Восходящий порог (тихие детали)
        ratio_up = 2.5          # Коэффициент усиления тихих мест (Upward)
        
        for band in bands:
            # Вычисляем огибающую RMS
            # Создаем сглаживающее окно 15 мс
            window_size = int(0.015 * self.sr)
            window = np.ones(window_size) / window_size
            
            # Для стерео берем среднюю огибающую по обоим каналам
            if is_stereo:
                mono_band = np.mean(band, axis=0)
            else:
                mono_band = band
                
            env_squared = signal.convolve(mono_band**2, window, mode='same')
            env = np.sqrt(np.maximum(env_squared, 1e-10))
            env_db = 20 * np.log10(env + 1e-8)
            
            # Рассчитываем гейн (дБ) для каждого сэмпла
            gain_db = np.zeros_like(env_db)
            
            # 1. Downward Compression (сжимаем громкие пики)
            over_threshold = env_db > threshold_down
            gain_db[over_threshold] += (threshold_down - env_db[over_threshold]) * (1.0 - 1.0 / ratio_down)
            
            # 2. Upward Compression (вытягиваем тихие хвосты/детали)
            under_threshold = env_db < threshold_up
            # Вытаскиваем тихий сигнал с плавным затуханием на абсолютной тишине
            gate_mask = env_db > -60.0
            upward_mask = under_threshold & gate_mask
            gain_db[upward_mask] += (threshold_up - env_db[upward_mask]) * (ratio_up - 1.0) * 0.15
            
            # Переводим гейн в линейный масштаб
            gain_linear = 10 ** (gain_db / 20.0)
            
            # Применяем гейн к полосе
            if is_stereo:
                comp_band = np.vstack([band[0] * gain_linear, band[1] * gain_linear])
            else:
                comp_band = band * gain_linear
                
            compressed_bands.append(comp_band)
            
        # Суммируем полосы обратно
        y_compressed = compressed_bands[0] + compressed_bands[1] + compressed_bands[2]
        
        # Подмешиваем обработанный сигнал к сухому на основе глубины (Depth Mix)
        y_out = (1.0 - depth) * y + depth * y_compressed
        
        return y_out

    def apply_autotune(self, y: np.ndarray, key_index: int = 0, scale: str = 'minor', speed: float = 0.85) -> np.ndarray:
        """
        Математический автотюн (Pitch Correction) вокального трека с использованием
        профессионального алгоритма PSOLA (Pitch-Synchronous Overlap-Add) библиотеки Praat.
        Определяет высоту тона f0 на всей дорожке вокала, квантует ее в соответствии с гаммой
        и ресинтезирует сигнал без разрывов фазы, тремоло и металлического призвука.
        
        Args:
            y: Входной аудиосигнал.
            key_index: Индекс тоники гаммы (0-11: C-B).
            scale: Лад ('major' или 'minor').
            speed: Скорость подтяжки (0.0..1.0, где 1.0 - мгновенный робо-эффект).
            
        Returns:
            np.ndarray: Скорректированный аудиосигнал.
        """
        import psola
        
        is_stereo = (y.ndim == 2 and y.shape[0] == 2)
        y_mono = np.mean(y, axis=0) if is_stereo else y
        
        if len(y_mono) == 0:
            return y
            
        scale_intervals = {
            'major': [0, 2, 4, 5, 7, 9, 11],
            'minor': [0, 2, 3, 5, 7, 8, 10]
        }
        allowed_intervals = scale_intervals.get(scale, scale_intervals['minor'])
        allowed_classes = [(key_index + interval) % 12 for interval in allowed_intervals]
        
        # 1. Оцениваем F0 один раз для всей дорожки вокала
        peak_val = np.max(np.abs(y_mono))
        y_mono_detect = y_mono / peak_val if peak_val > 1e-4 else y_mono
        
        fmin = librosa.note_to_hz('C2')
        fmax = librosa.note_to_hz('C6')
        
        f0, _, voiced_prob = librosa.pyin(
            y_mono_detect, 
            fmin=fmin, 
            fmax=fmax, 
            sr=self.sr
        )
        voiced_flag = (voiced_prob > 0.35) & (~np.isnan(f0)) & (f0 > 0)
        
        # 2. Строим целевой контур частоты
        target_pitch = np.copy(f0)
        
        for t in range(len(f0)):
            if not voiced_flag[t]:
                target_pitch[t] = np.nan
                continue
                
            f_curr = f0[t]
            midi = 12.0 * np.log2(f_curr / 440.0) + 69.0
            midi_round = int(round(midi))
            
            octave = midi_round // 12
            note_class = midi_round % 12
            
            best_diff = 12.0
            target_class = allowed_classes[0]
            for c in allowed_classes:
                diff = min(abs(note_class - c), 12 - abs(note_class - c))
                if diff < best_diff:
                    best_diff = diff
                    target_class = c
                    
            target_midi = octave * 12 + target_class
            
            if abs(target_midi - midi) > 6.0:
                if target_midi < midi:
                    target_midi += 12
                else:
                    target_midi -= 12
                    
            effective_speed = np.power(speed, 0.3) if speed > 0.1 else speed
            target_midi_clamped = midi + effective_speed * (target_midi - midi)
            
            target_pitch[t] = 440.0 * (2.0 ** ((target_midi_clamped - 69.0) / 12.0))
            
        n_voiced = np.sum(voiced_flag)
        total_f = len(f0)
        pct = (n_voiced / total_f * 100.0) if total_f > 0 else 0.0
        print(f"[DEBUG] Autotune (PSOLA): detected {n_voiced} voiced frames out of {total_f} ({pct:.1f}%)")
        
        # 3. Синтезируем с помощью PSOLA
        if is_stereo:
            y_tuned_l = psola.vocode(y[0], self.sr, target_pitch=target_pitch, fmin=int(fmin), fmax=int(fmax))
            y_tuned_r = psola.vocode(y[1], self.sr, target_pitch=target_pitch, fmin=int(fmin), fmax=int(fmax))
            y_tuned = np.vstack([y_tuned_l, y_tuned_r])
        else:
            y_tuned = psola.vocode(y_mono, self.sr, target_pitch=target_pitch, fmin=int(fmin), fmax=int(fmax))
            
        return y_tuned



