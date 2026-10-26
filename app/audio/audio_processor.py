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
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    y = pyln.normalize.loudness(y, loudness, target_lufs)
                # Защита от цифрового клиппинга (True Ceiling)
                peak = np.max(np.abs(y))
                if peak > 1.0:
                    y = y / peak * 0.99
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
                f0 = librosa.yin(
                    y,
                    fmin=librosa.note_to_hz('C2'),
                    fmax=librosa.note_to_hz('C6'),
                    sr=self.sr,
                    frame_length=2048,
                    hop_length=512
                )
                rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
                min_len = min(len(f0), len(rms))
                valid_f0 = f0[:min_len][(rms[:min_len] > 0.04 * (np.max(rms) + 1e-8)) & ~np.isnan(f0[:min_len]) & (f0[:min_len] > 80.0)]
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

    def get_stereo_mel_spectrogram(
        self,
        y: np.ndarray,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        window: str = 'hann',
        pitch_normalize: bool = False
    ) -> np.ndarray:
        """
        Вычисляет двухканальную (Mid + Side) Мел-спектрограмму, нормализованную в диапазон [-1.0, 1.0].
        Канал Side нормируется относительно пиковой мощности канала Mid, сохраняя точный баланс Mid/Side в дБ.

        Args:
            y: Входной аудиосигнал (1D моно или 2D стерео формы (2, N) или (N, 2)).
            n_fft: Длина окна FFT.
            hop_length: Шаг сдвига окна.
            n_mels: Количество Мел-фильтров.
            window: Тип оконной функции.
            pitch_normalize: Если True, компенсирует экстремальный питч-шифт (Nightcore).

        Returns:
            np.ndarray: Массив формы (2, n_mels, T) со значениями в [-1.0, 1.0] (float32).
        """
        if y.ndim == 2:
            if y.shape[0] == 2:
                left, right = y[0], y[1]
            elif y.shape[1] == 2:
                left, right = y[:, 0], y[:, 1]
            else:
                left = np.mean(y, axis=0)
                right = left
            mid = 0.5 * (left + right)
            side = 0.5 * (left - right)
        else:
            mid = y
            side = np.zeros_like(mid)

        if pitch_normalize and len(mid) > 0:
            try:
                f0 = librosa.yin(
                    mid,
                    fmin=librosa.note_to_hz('C2'),
                    fmax=librosa.note_to_hz('C6'),
                    sr=self.sr,
                    frame_length=2048,
                    hop_length=512
                )
                rms = librosa.feature.rms(y=mid, frame_length=2048, hop_length=512)[0]
                min_len = min(len(f0), len(rms))
                valid_f0 = f0[:min_len][(rms[:min_len] > 0.04 * (np.max(rms) + 1e-8)) & ~np.isnan(f0[:min_len]) & (f0[:min_len] > 80.0)]
                if len(valid_f0) > 0:
                    median_pitch = float(np.median(valid_f0))
                    if median_pitch > 310.0:
                        n_steps = 12.0 * np.log2(220.0 / median_pitch)
                        mid = librosa.effects.pitch_shift(mid, sr=self.sr, n_steps=n_steps)
                        if np.max(np.abs(side)) > 1e-6:
                            side = librosa.effects.pitch_shift(side, sr=self.sr, n_steps=n_steps)
            except Exception:
                pass

        S_mid = librosa.feature.melspectrogram(
            y=mid, sr=self.sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, window=window
        )
        ref_power = float(max(np.max(S_mid), 1e-10))
        db_mid = librosa.power_to_db(S_mid, ref=ref_power, top_db=80.0)

        if np.max(np.abs(side)) > 1e-6:
            S_side = librosa.feature.melspectrogram(
                y=side, sr=self.sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, window=window
            )
            db_side = np.clip(10.0 * np.log10( np.maximum(S_side, 1e-12) / ref_power ), -80.0, 0.0)
        else:
            db_side = np.full_like(db_mid, -80.0)

        mel_mid_norm = np.clip((db_mid + 40.0) / 40.0, -1.0, 1.0)
        mel_side_norm = np.clip((db_side + 40.0) / 40.0, -1.0, 1.0)
        return np.stack([mel_mid_norm, mel_side_norm], axis=0).astype(np.float32)

    def extract_mixing_descriptors(self, y: np.ndarray) -> np.ndarray:
        """
        Извлекает 18 явных физических DSP-дескрипторов сигнала (динамика, гармоники, стерео, пространство, питч),
        нормализованных в диапазон [-1.0, 1.0] для прямой подачи в регрессионную и классификационную ветки модели.

        Args:
            y: Входной аудиосигнал (1D моно или 2D стерео).

        Returns:
            np.ndarray: Вектор из 18 признаков (float32) в диапазоне [-1.0, 1.0].
        """
        import scipy.signal as signal

        def norm_range(val: float, low: float, high: float) -> float:
            return float(np.clip(2.0 * (val - low) / (max(high - low, 1e-8)) - 1.0, -1.0, 1.0))

        if y.ndim == 2:
            if y.shape[0] == 2:
                left, right = y[0], y[1]
            elif y.shape[1] == 2:
                left, right = y[:, 0], y[:, 1]
            else:
                left = np.mean(y, axis=0)
                right = left
            mid = 0.5 * (left + right)
            side = 0.5 * (left - right)
            is_stereo = True
        else:
            left = y
            right = y
            mid = y
            side = np.zeros_like(mid)
            is_stereo = False

        if len(mid) < 2048 or np.max(np.abs(mid)) < 1e-6:
            return np.zeros(18, dtype=np.float32)

        # 1-5: Динамика и компрессия (Compressor / Multiband OTT)
        frame_len = 1024
        hop_len = 512
        rms_frames = librosa.feature.rms(y=mid, frame_length=frame_len, hop_length=hop_len)[0]
        peak_global = float(np.max(np.abs(mid)) + 1e-8)
        rms_global = float(np.sqrt(np.mean(mid ** 2)) + 1e-8)
        crest_db = 20.0 * np.log10(peak_global / rms_global)

        rms_db = 20.0 * np.log10(rms_frames / (np.max(rms_frames) + 1e-8) + 1e-8)
        active_mask = rms_db > -45.0
        active_rms_db = rms_db[active_mask] if np.sum(active_mask) >= 4 else rms_db

        dr_p95_p10 = float(np.percentile(active_rms_db, 95) - np.percentile(active_rms_db, 10))
        dr_p99_p90 = float(np.percentile(active_rms_db, 99) - np.percentile(active_rms_db, 90))
        rms_std = float(np.std(active_rms_db))
        env_mod_rate = float(np.mean(np.abs(np.diff(active_rms_db)))) if len(active_rms_db) > 1 else 0.0

        f1_crest = norm_range(crest_db, 4.0, 24.0)
        f2_dr = norm_range(dr_p95_p10, 2.0, 28.0)
        f3_trans = norm_range(dr_p99_p90, 0.3, 8.0)
        f4_rms_std = norm_range(rms_std, 1.0, 10.0)
        f5_env_mod = norm_range(env_mod_rate, 0.3, 4.5)

        # 6-10: Спектр, сибилянты, сатурация и экситер (De-esser / Distortion / Exciter / OTT)
        stft_mag = np.abs(librosa.stft(mid, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=2048)

        body_mask = (freqs >= 200.0) & (freqs <= 3000.0)
        lowmid_mask = (freqs >= 150.0) & (freqs <= 800.0)
        highmid_mask = (freqs >= 1500.0) & (freqs <= 5000.0)
        sib_mask = (freqs >= 4500.0) & (freqs <= 10000.0)
        air_mask = (freqs >= 10000.0) & (freqs <= 18000.0)
        flat_mask = (freqs >= 1000.0) & (freqs <= 8000.0)

        body_energy_t = np.mean(stft_mag[body_mask, :] ** 2, axis=0) + 1e-12
        sib_energy_t = np.mean(stft_mag[sib_mask, :] ** 2, axis=0) + 1e-12
        sib_ratio_frames_db = 10.0 * np.log10(sib_energy_t / body_energy_t)
        sib_p90_db = float(np.percentile(sib_ratio_frames_db[active_mask], 90)) if np.sum(active_mask) >= 4 else float(np.percentile(sib_ratio_frames_db, 90))

        body_mean = float(np.mean(body_energy_t))
        air_mean = float(np.mean(stft_mag[air_mask, :] ** 2) + 1e-12)
        air_ratio_db = 10.0 * np.log10(air_mean / body_mean)

        lowmid_mean = float(np.mean(stft_mag[lowmid_mask, :] ** 2) + 1e-12)
        highmid_mean = float(np.mean(stft_mag[highmid_mask, :] ** 2) + 1e-12)
        highmid_ratio_db = 10.0 * np.log10(highmid_mean / lowmid_mean)

        sub_mag = stft_mag[flat_mask, :] + 1e-10
        geom_mean = np.exp(np.mean(np.log(sub_mag), axis=0))
        arith_mean = np.mean(sub_mag, axis=0) + 1e-10
        flatness_db = float(20.0 * np.log10(np.mean(geom_mean / arith_mean) + 1e-8))

        # Спектральный центроид ВЧ-полосы (4.5..16 кГц) — напрямую реагирует на exciter.cutoff_hz и reverb.damping
        hf_mask = (freqs >= 4500.0) & (freqs <= 16000.0)
        hf_power = np.mean(stft_mag[hf_mask, :] ** 2, axis=1) + 1e-12
        hf_centroid_hz = float(np.sum(freqs[hf_mask] * hf_power) / np.sum(hf_power))

        f6_sib = norm_range(sib_p90_db, -24.0, 4.0)
        f7_air = norm_range(air_ratio_db, -42.0, -6.0)
        f8_hmid = norm_range(highmid_ratio_db, -22.0, 8.0)
        f9_flat = norm_range(flatness_db, -28.0, -4.0)
        f10_crest_hb = norm_range(hf_centroid_hz, 5500.0, 12500.0)

        # 11-14: Стерео-ширина, задержка Хааса и фазовая декорреляция (Stereo Enhancer / Chorus / Reverb)
        if is_stereo and np.max(np.abs(side)) > 1e-6:
            rms_m = float(np.sqrt(np.mean(mid ** 2)) + 1e-8)
            rms_s = float(np.sqrt(np.mean(side ** 2)) + 1e-8)
            width_global = min(1.0, 2.0 * rms_s / (rms_m + rms_s))

            sos_hp150 = signal.butter(4, 150.0, btype='highpass', fs=self.sr, output='sos')
            mid_hp = signal.sosfilt(sos_hp150, mid)
            side_hp = signal.sosfilt(sos_hp150, side)
            rms_m_hp = float(np.sqrt(np.mean(mid_hp ** 2)) + 1e-8)
            rms_s_hp = float(np.sqrt(np.mean(side_hp ** 2)) + 1e-8)
            width_high = min(1.0, 2.0 * rms_s_hp / (rms_m_hp + rms_s_hp))

            # Взаимная корреляция L и R в окне задержек до 30 мс
            max_lag = int(0.030 * self.sr)
            l_norm = left - np.mean(left)
            r_norm = right - np.mean(right)
            denom = float(np.sqrt(np.sum(l_norm ** 2) * np.sum(r_norm ** 2)) + 1e-8)
            xcorr_full = signal.fftconvolve(r_norm, l_norm[::-1], mode='full') / denom
            center = len(l_norm) - 1
            window_xcorr = xcorr_full[max(0, center - max_lag) : min(len(xcorr_full), center + max_lag + 1)]
            peak_idx = int(np.argmax(np.abs(window_xcorr)))
            lag_samples = abs(peak_idx - (center - max(0, center - max_lag)))
            lag_ms = (lag_samples / self.sr) * 1000.0
            max_corr = float(np.clip(np.max(np.abs(window_xcorr)), 0.0, 1.0))
            decorrelation = 1.0 - max_corr
        else:
            width_global = 0.0
            width_high = 0.0
            lag_ms = 0.0
            decorrelation = 0.0

        f11_w_glob = norm_range(width_global, 0.0, 0.8)
        f12_w_high = norm_range(width_high, 0.0, 0.85)
        f13_haas_lag = norm_range(lag_ms, 0.0, 26.0)
        f14_decorr = norm_range(decorrelation, 0.0, 0.5)

        # 15-17: Пространство, хвост реверберации и эхо дилея (Reverb / Delay)
        # Оценка энергии хвостов после спадов фраз (переход из активного вокала в паузу)
        diffs_db = np.diff(rms_db)
        tail_levels = []
        for idx_frame in range(len(diffs_db) - 8):
            if rms_db[idx_frame] > -20.0 and rms_db[idx_frame + 2] < rms_db[idx_frame] - 6.0:
                tail_drop = rms_db[idx_frame + 6] - rms_db[idx_frame]
                tail_levels.append(tail_drop)
        reverb_tail_db = float(np.median(tail_levels)) if tail_levels else float(np.percentile(rms_db, 20) - np.percentile(rms_db, 80))

        # Автокорреляция высокочастотной огибающей (лаги 60 мс .. 520 мс) для детекции Delay
        env_diff = np.maximum(0.0, np.diff(rms_frames, prepend=rms_frames[0]))
        env_centered = env_diff - np.mean(env_diff)
        env_var = float(np.sum(env_centered ** 2) + 1e-8)
        acorr = signal.fftconvolve(env_centered, env_centered[::-1], mode='full')[len(env_centered) - 1 :] / env_var
        min_delay_lag = max(2, int(0.065 * self.sr / hop_len))
        max_delay_lag = min(len(acorr) - 1, int(0.520 * self.sr / hop_len))
        if max_delay_lag > min_delay_lag:
            acorr_slice = acorr[min_delay_lag : max_delay_lag + 1]
            best_lag_idx = int(np.argmax(acorr_slice))
            delay_peak = float(acorr_slice[best_lag_idx])
            delay_lag_sec = float((min_delay_lag + best_lag_idx) * hop_len / self.sr)
        else:
            delay_peak = 0.0
            delay_lag_sec = 0.08

        f15_rev_tail = norm_range(reverb_tail_db, -32.0, -4.0)
        f16_dly_peak = norm_range(delay_peak, 0.02, 0.45)
        f17_dly_lag = norm_range(delay_lag_sec, 0.08, 0.50)

        # 18: Квантование высоты тона F0 к полутоновой сетке в центах (Pitch Corrector)
        try:
            f0 = librosa.yin(
                mid,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C6'),
                sr=self.sr,
                frame_length=2048,
                hop_length=1024
            )
            rms_f0 = librosa.feature.rms(y=mid, frame_length=2048, hop_length=1024)[0]
            flat_f0 = librosa.feature.spectral_flatness(y=mid, n_fft=2048, hop_length=1024)[0]
            m_len = min(len(f0), len(rms_f0), len(flat_f0))
            v_mask = (
                (rms_f0[:m_len] > 0.05 * (np.max(rms_f0) + 1e-8))
                & (flat_f0[:m_len] < 0.22)
                & (~np.isnan(f0[:m_len]))
                & (f0[:m_len] > 85.0)
                & (f0[:m_len] < 950.0)
            )
            if np.sum(v_mask) >= 4:
                midi_vals = 12.0 * np.log2(f0[:m_len][v_mask] / 440.0) + 69.0
                # Проверяем минимальное среднее отклонение по всем 12 мажорным/минорным гаммам или хроматической сетке
                cent_dev = np.abs(midi_vals - np.round(midi_vals)) * 100.0
                median_cents = float(np.median(cent_dev))
            else:
                median_cents = 20.0
        except Exception:
            median_cents = 20.0

        # Меньше центов отклонения -> ближе к +1.0 (сильный автотюн), больше центов -> ближе к -1.0 (живое вибрато)
        f18_pitch_quant = -norm_range(median_cents, 3.0, 25.0)

        return np.array([
            f1_crest, f2_dr, f3_trans, f4_rms_std, f5_env_mod,
            f6_sib, f7_air, f8_hmid, f9_flat, f10_crest_hb,
            f11_w_glob, f12_w_high, f13_haas_lag, f14_decorr,
            f15_rev_tail, f16_dly_peak, f17_dly_lag, f18_pitch_quant
        ], dtype=np.float32)

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
