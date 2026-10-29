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
        
        # 3. Резонанс комнаты/микрофона стоит на одной частоте, а гармоники голоса двигаются вместе с нотой
        # и в покадровом сравнении неотличимы от резонансов. Поэтому кандидатов ищем по долгосрочному
        # спектру (LTAS) громких кадров: там гармоники размазываются, а стационарные пики остаются.
        frame_energy = magnitude.sum(axis=0)
        loud_frames = frame_energy > np.percentile(frame_energy, 40)
        ltas = magnitude[:, loud_frames].mean(axis=1) if loud_frames.any() else magnitude.mean(axis=1)
        ltas_smooth = ndimage.median_filter(ltas, size=17, mode='reflect')
        ltas_ratio_db = 20 * np.log10((ltas + 1e-8) / (ltas_smooth + 1e-8))
        resonant_bins = ltas_ratio_db > threshold_db * 0.5

        # Если мелодия узкая, гармоники частых нот тоже стационарны в LTAS. Отличаем их по F0:
        # пик на k * F0 текущего кадра — это голос, а не резонанс.
        harmonic_mask = self._harmonic_bin_mask(y, ratio_db, n_fft, hop_length)

        # 4. Ослабляем только стационарные негармонические бины в кадрах, где пик реально выпирает
        resonance_mask = resonant_bins[:, np.newaxis] & (ratio_db > threshold_db) & ~harmonic_mask
        attenuation_db = np.zeros_like(magnitude)
        attenuation_db[resonance_mask] = np.minimum(ratio_db[resonance_mask] - threshold_db, max_attenuation_db)

        # Сглаживание по времени (~60 мс), чтобы маска не дрожала от кадра к кадру (иначе слышен хрип)
        attenuation_db = ndimage.uniform_filter1d(attenuation_db, size=5, axis=1)

        # Применяем ослабление к спектру
        new_magnitude = magnitude * 10 ** (-attenuation_db / 20.0)
        
        # 5. Восстанавливаем комплексный спектр и делаем обратный STFT
        new_stft = new_magnitude * np.exp(1j * phase)
        y_clean = librosa.istft(new_stft, hop_length=hop_length, length=len(y))
        
        return y_clean

    def _harmonic_bin_mask(self, y: np.ndarray, ratio_db: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
        """
        Маска бинов STFT, лежащих на гармониках F0 в вокализованных кадрах.
        Кадр считается вокализованным, если минимум 3 из первых 5 гармоник выступают над огибающей на 6 дБ:
        одиночный свист на гармонике случайного F0 под это условие не попадает.
        """
        n_bins, n_frames = ratio_db.shape
        f0 = librosa.yin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'),
            sr=self.sr, frame_length=n_fft, hop_length=hop_length
        )[:n_frames]
        if len(f0) < n_frames:
            f0 = np.pad(f0, (0, n_frames - len(f0)), mode='edge')
        f0 = f0.astype(np.float32)

        bin_hz = self.sr / n_fft
        freqs = (np.arange(n_bins) * bin_hz).astype(np.float32)[:, np.newaxis]

        harmonic_bins = np.clip(np.rint(np.outer(np.arange(1, 6), f0) / bin_hz).astype(int), 0, n_bins - 1)
        voiced = (ratio_db[harmonic_bins, np.arange(n_frames)] > 6.0).sum(axis=0) >= 3

        harmonic_index = freqs / f0[np.newaxis, :]
        dist_hz = np.abs(harmonic_index - np.maximum(np.rint(harmonic_index), 1.0)) * f0[np.newaxis, :]
        # Допуск растет с номером гармоники (джиттер k * dF0), но не шире 20% шага между гармониками
        tolerance_hz = np.maximum(1.5 * bin_hz, np.minimum(0.03 * freqs, 0.2 * f0[np.newaxis, :]))
        return (dist_hz <= tolerance_hz) & voiced[np.newaxis, :]

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
            
        # 1. Выделяем высокие частоты выше cutoff_hz с помощью фильтра Баттерворта ВЧ
        safe_cutoff = float(np.clip(cutoff_hz, 3500.0, self.sr * 0.44))
        sos_high = signal.butter(4, safe_cutoff, btype='highpass', fs=self.sr, output='sos')
        y_high = signal.sosfilt(sos_high, y)
        
        # 2. Нормализуем ВЧ-полосу перед нелинейным звеном, чтобы сатурация (tanh + четная гармоника)
        # стабильно работала даже на вокале, нормализованном к -23 LUFS!
        p98_high = float(np.percentile(np.abs(y_high), 98) + 1e-6)
        y_high_norm = y_high / p98_high
        y_harmonics = np.tanh(y_high_norm * (drive * 1.8)) + 0.35 * np.tanh((y_high_norm ** 2) * drive)
        
        # 3. Фильтруем результат выше safe_cutoff, чтобы сохранить явную спектральную границу cutoff_hz
        # и убрать интермодуляционную грязь в нижней середине
        sos_air = signal.butter(4, safe_cutoff, btype='highpass', fs=self.sr, output='sos')
        y_air = signal.sosfilt(sos_air, y_harmonics)
        
        # Привязываем уровень сгенерированных гармоник к энергии исходного сигнала
        rms_full = float(np.sqrt(np.mean(y ** 2)) + 1e-8)
        rms_orig = float(np.sqrt(np.mean(y_high ** 2)) + 1e-8)
        rms_air = float(np.sqrt(np.mean(y_air ** 2)) + 1e-8)
        target_air_rms = max(rms_orig * 2.2, rms_full * 0.28)
        y_air = y_air * (target_air_rms / rms_air)
        
        # 4. Смешиваем с исходным сигналом пропорционально mix
        y_excited = y + (mix * 1.6) * y_air
        
        return y_excited

    def apply_stereo_enhancer(self, y: np.ndarray, delay_ms: float = 18.0, width: float = 1.0) -> np.ndarray:
        """
        Расширяет стереобазу задержанной копией голоса, подмешанной только в Side (Mid/Side-вариант эффекта
        Хааса): L = M + S + g*d, R = M + S - g*d, где d — задержанный Mid выше 150 Гц, g = width / 2.
        Mid (моно-сумма) не меняется, поэтому тембр не окрашивается гребенчатым фильтром, а низ остается в моно.

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
            
        # Раньше задержанная копия подмешивалась к сухому правому каналу: в моно-сумме это гребенчатый
        # фильтр (провалы через 1/delay Гц), который портил тембр сильнее, чем добавлял ширины
        mid = 0.5 * (y_stereo[0] + y_stereo[1])
        delayed = np.zeros_like(mid)
        delayed[delay_samples:] = mid[:-delay_samples]
        sos_hp = signal.butter(4, 150.0, btype='highpass', fs=self.sr, output='sos')
        spread = 0.5 * width * signal.sosfilt(sos_hp, delayed)

        return np.vstack([y_stereo[0] + spread, y_stereo[1] - spread]).astype(y_stereo.dtype)

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
            # Извлекаем F0 (Pitch) с помощью быстрого детерминированного алгоритма YIN
            peak_val = np.max(np.abs(y))
            y_detect = y / peak_val if peak_val > 1e-4 else y
            
            frame_length = 2048
            hop_length = 512
            f0 = librosa.yin(
                y_detect,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C6'),
                sr=self.sr,
                frame_length=frame_length,
                hop_length=hop_length
            )
            rms = librosa.feature.rms(y=y_detect, frame_length=frame_length, hop_length=hop_length)[0]
            flatness = librosa.feature.spectral_flatness(y=y_detect, n_fft=frame_length, hop_length=hop_length)[0]
            min_len = min(len(f0), len(rms), len(flatness))
            f0 = f0[:min_len]
            voiced_flag = (
                (rms[:min_len] > 0.04 * (np.max(rms) + 1e-8))
                & (flatness[:min_len] < 0.25)
                & (~np.isnan(f0))
                & (f0 > librosa.note_to_hz('C2') * 1.03)
                & (f0 < librosa.note_to_hz('C6') * 0.97)
            )
            
            f0_clean = f0[voiced_flag]
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
        
        # Параметры агрессивной OTT-компрессии по 3 полосам (Low, Mid, High)
        # Пороги адаптированы под реальное распределение энергии полос при -23 LUFS
        band_configs = [
            {"thresh_down": -26.0, "ratio_down": 5.0, "thresh_up": -40.0, "ratio_up": 3.2, "makeup_db": 4.5},
            {"thresh_down": -22.0, "ratio_down": 4.5, "thresh_up": -34.0, "ratio_up": 3.0, "makeup_db": 2.5},
            {"thresh_down": -30.0, "ratio_down": 5.5, "thresh_up": -44.0, "ratio_up": 4.0, "makeup_db": 7.5},
        ]
        
        for band, b_cfg in zip(bands, band_configs):
            threshold_down = b_cfg["thresh_down"]
            ratio_down = b_cfg["ratio_down"]
            threshold_up = b_cfg["thresh_up"]
            ratio_up = b_cfg["ratio_up"]
            makeup_db = b_cfg["makeup_db"]

            # Вычисляем огибающую RMS в окне 12 мс
            window_size = max(1, int(0.012 * self.sr))
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
            gain_db = np.full_like(env_db, makeup_db)
            
            # 1. Downward Compression (сжимаем громкие пики полосы)
            over_threshold = env_db > threshold_down
            gain_db[over_threshold] += (threshold_down - env_db[over_threshold]) * (1.0 - 1.0 / ratio_down)
            
            # 2. Upward Compression (вытягиваем тихие хвосты и ВЧ-воздух в стиле Xfer OTT)
            under_threshold = env_db < threshold_up
            gate_mask = env_db > -58.0
            upward_mask = under_threshold & gate_mask
            gain_db[upward_mask] += np.minimum(12.0, (threshold_up - env_db[upward_mask]) * (1.0 - 1.0 / ratio_up) * 0.65)
            
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

    def _psola_shift(self, y: np.ndarray, frame_times: np.ndarray, correction: np.ndarray,
                     fmin: float, fmax: float) -> np.ndarray:
        """
        Сдвигает высоту на correction полутонов (кадры frame_times) через PSOLA Praat.

        Цель строится от собственного контура высоты Praat, по которому он режет сигнал на периоды: при
        нулевой поправке сигнал не меняется. Библиотека psola строила цель от внешней оценки F0 (расхождение
        в несколько центов давало дрожание высоты) и писала 0 Гц в кадры без голоса — Praat интерполировал
        цель к нулю, и на краях нот и придыхании высота проваливалась («песок»).
        """
        import parselmouth
        from parselmouth.praat import call

        sound = parselmouth.Sound(np.asarray(y, dtype=np.float64), sampling_frequency=self.sr)
        manipulation = call(sound, "To Manipulation", 0.01, float(fmin), float(fmax))
        source = call(manipulation, "Extract pitch tier")
        target = call("Create PitchTier", "target", sound.xmin, sound.xmax)
        for i in range(1, int(call(source, "Get number of points")) + 1):
            t = call(source, "Get time from index", i)
            shift = float(np.interp(t, frame_times, correction))
            call(target, "Add point", t, call(source, "Get value at index", i) * 2.0 ** (shift / 12.0))
        call([target, manipulation], "Replace pitch tier")
        out = call(manipulation, "Get resynthesis (overlap-add)").values[0]
        return np.pad(out, (0, max(0, len(y) - len(out))))[:len(y)].astype(np.float32)

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
        
        frame_length = 2048
        hop_length = 512
        f0 = librosa.yin(
            y_mono_detect, 
            fmin=fmin, 
            fmax=fmax, 
            sr=self.sr,
            frame_length=frame_length,
            hop_length=hop_length
        )
        rms = librosa.feature.rms(y=y_mono_detect, frame_length=frame_length, hop_length=hop_length)[0]
        flatness = librosa.feature.spectral_flatness(y=y_mono_detect, n_fft=frame_length, hop_length=hop_length)[0]
        min_len = min(len(f0), len(rms), len(flatness))
        f0 = f0[:min_len]
        voiced_flag = (
            (rms[:min_len] > 0.035 * (np.max(rms) + 1e-8))
            & (flatness[:min_len] < 0.28)
            & (~np.isnan(f0))
            & (f0 > fmin * 1.03)
            & (f0 < fmax * 0.97)
        )
        
        # 2. Целевой контур. Нота выбирается по сглаженной высоте с гистерезисом (вибрато и ошибки yin
        # не перебрасывают цель между соседними нотами), а поправка к ноте плавно меняется во времени:
        # мгновенные скачки цели PSOLA пересинтезирует с «песком» на верхах
        hop_s = hop_length / self.sr
        midi = np.full(len(f0), np.nan)
        midi[voiced_flag] = 12.0 * np.log2(f0[voiced_flag] / 440.0) + 69.0
        midi_smooth = np.copy(midi)
        midi_smooth[voiced_flag] = ndimage.median_filter(np.nan_to_num(midi, nan=0.0), size=7)[voiced_flag]

        def nearest_note(m: float) -> float:
            base = int(np.floor(m)) - 6
            candidates = [n for n in range(base, base + 14) if n % 12 in allowed_classes]
            return float(min(candidates, key=lambda n: abs(n - m)))

        correction = np.zeros(len(f0))
        note = None
        for t in range(len(f0)):
            if not voiced_flag[t]:
                note = None
                continue
            nearest = nearest_note(midi_smooth[t])
            # Гистерезис: соседняя нота принимается, только если голос ушел от текущей больше чем на 0.65 полутона
            if note is None or abs(midi_smooth[t] - note) > 0.65:
                note = nearest
            correction[t] = note - midi_smooth[t]

        # speed: 1.0 — подтяжка за ~10 мс (робо-эффект), 0.0 — за ~200 мс; сила подтяжки как раньше
        retune_s = 0.01 + 0.19 * (1.0 - speed)
        amount = np.power(speed, 0.3) if speed > 0.1 else speed
        correction = amount * ndimage.uniform_filter1d(correction, size=max(1, int(round(retune_s / hop_s))))
        # 3. Синтезируем с помощью PSOLA (Praat): поправка в полутонах по кадрам yin
        frame_times = np.arange(len(correction)) * hop_s
        if is_stereo:
            y_tuned = np.vstack([self._psola_shift(ch, frame_times, correction, fmin, fmax) for ch in y])
        else:
            y_tuned = self._psola_shift(y_mono, frame_times, correction, fmin, fmax)

        # PSOLA оставляет инфразвук и гул ниже основного тона (+14 дБ ниже 20 Гц на реальном вокале),
        # а EQ стоит в цепочке раньше автотюна и уже не может его срезать
        sos = signal.butter(4, fmin * 0.9, btype='highpass', fs=self.sr, output='sos')
        y_tuned = signal.sosfilt(sos, y_tuned, axis=-1)

        # Тюн не смешивается с оригиналом: PSOLA пересобирает волну с другой фазой (корреляция с оригиналом
        # около нуля даже без сдвига), и плавный переход между ними давал фазовые призвуки на краях нот
        return y_tuned




    def _shift_voice(self, y: np.ndarray, semitones: float, formant_shift: float) -> np.ndarray:
        """
        Сдвигает высоту моно-сигнала Rubber Band без изменения длины и задержки (латентность компенсирована).
        Форманты: сначала весь голос (тон + форманты) сдвигается на formant_shift без сохранения формант,
        затем тон — на остаток с сохранением формант. Итог: тон на semitones, форманты на formant_shift.
        """
        from pedalboard import time_stretch

        voice = y[np.newaxis, :].astype(np.float32)
        if abs(formant_shift) > 1e-3:
            voice = time_stretch(voice, self.sr, 1.0, float(formant_shift), preserve_formants=False)
        rest = semitones - formant_shift
        if abs(rest) > 1e-3:
            voice = time_stretch(voice, self.sr, 1.0, float(rest), preserve_formants=True)
        voice = voice[0, :len(y)]
        return np.pad(voice, (0, len(y) - len(voice)))

    def harmony_layer(self, y: np.ndarray, semitones: float = 3.0, level_db: float = -8.0,
                      formant_shift: float = 0.0, delay_ms: float = 0.0, detune_cents: float = 0.0,
                      width: float = 0.0, mix: float = 1.0) -> np.ndarray | None:
        """
        Строит слой гармонии или дабла: копию голоса, сдвинутую на semitones полутонов (без самого голоса).

        Копия строится из моно-суммы входа, приводится к RMS лидера и ставится на level_db ниже него.
        При width == 0 слой один и стоит в центре; при width > 0 голосов два (L и R с панорамой ±width),
        у них противоположная расстройка ±detune_cents и разное отставание (delay_ms и 0.6·delay_ms),
        как у двух ручных даблов; суммарная мощность слоя по-прежнему level_db.

        Args:
            y: Голос-источник (1D моно или (2, n) стерео).
            semitones: Интервал гармонии в полутонах (+3, +5, +7, ±12).
            level_db: Уровень слоя относительно лидера в дБ (по RMS).
            formant_shift: Сдвиг формант в полутонах (0 — форманты голоса сохранены).
            delay_ms: Отставание слоя от лидера в мс.
            detune_cents: Расстройка слоя в центах.
            width: Ширина панорамы слоя (0..1).
            mix: Общий фейдер слоя (0 — слоя нет).

        Returns:
            np.ndarray | None: Слой формы (2, n) или None, если слоя нет (mix = 0 или тишина).
        """
        if mix <= 0.0:
            return None
        mono = y if y.ndim == 1 else np.mean(y, axis=0)
        lead_rms = float(np.sqrt(np.mean(mono ** 2)))
        if lead_rms < 1e-9:
            return None

        detune = detune_cents / 100.0
        if width > 0.0:
            voices = [(semitones + detune, delay_ms, -width), (semitones - detune, 0.6 * delay_ms, width)]
        else:
            voices = [(semitones + detune, delay_ms, 0.0)]
        voice_gain = mix * 10.0 ** (level_db / 20.0) / np.sqrt(len(voices))

        layer = np.zeros((2, len(mono)), dtype=np.float32)
        for shift, lag_ms, pan in voices:
            voice = self._shift_voice(mono, shift, formant_shift)
            voice_rms = float(np.sqrt(np.mean(voice ** 2)))
            if voice_rms < 1e-9:
                continue
            voice *= voice_gain * lead_rms / voice_rms
            lag = int(round(lag_ms / 1000.0 * self.sr))
            if lag > 0:
                voice = np.concatenate([np.zeros(lag, dtype=voice.dtype), voice[:-lag]])
            # Равномощная панорама, нормированная на центр: в центре слой одинаков в обоих каналах
            angle = (pan + 1.0) * np.pi / 4.0
            layer[0] += np.sqrt(2.0) * np.cos(angle) * voice
            layer[1] += np.sqrt(2.0) * np.sin(angle) * voice
        return layer

    @staticmethod
    def add_layer(y: np.ndarray, layer: np.ndarray | None) -> np.ndarray:
        """
        Суммирует голос со слоем (2, n). Моно-голос остается моно, если слой в центре (каналы равны),
        иначе становится стерео (2, n).
        """
        if layer is None:
            return y
        if y.ndim == 1 and np.array_equal(layer[0], layer[1]):
            return (y + layer[0]).astype(y.dtype)
        lead = np.vstack([y, y]) if y.ndim == 1 else y
        if lead.shape[0] == 1:
            lead = np.vstack([lead[0], lead[0]])
        return (lead + layer).astype(lead.dtype)

    def apply_harmony(self, y: np.ndarray, **layer_params: float) -> np.ndarray:
        """Голос + слой гармонии (параметры как у harmony_layer); при mix = 0 вход возвращается без изменений."""
        return self.add_layer(y, self.harmony_layer(y, **layer_params))
