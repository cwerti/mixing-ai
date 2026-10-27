"""
Единая подготовка входов MixingAIModel для генератора датасета и инференса.

Модель должна получать на инференсе ровно то, на чем училась: стерео Mid/Side (иначе ширина референса не видна),
громкий фрагмент длиной в окно модели на -23 LUFS, мел-спектрограмму без pitch_normalize.
"""
import librosa
import numpy as np

from app.audio.audio_processor import AudioProcessor
from app.mixing.chain_optimizer import normalize_lufs, select_excerpt, to_stereo

MODEL_LUFS = -23.0
MEL_HOP = 512
N_MELS = 128
DEFAULT_MAX_LEN = 520


def window_seconds(sr: int, max_len: int = DEFAULT_MAX_LEN) -> float:
    """Длительность сигнала, которая помещается в max_len кадров мел-спектрограммы."""
    return max_len * MEL_HOP / sr


def prepare_model_audio(y: np.ndarray, sr: int, max_len: int = DEFAULT_MAX_LEN) -> np.ndarray:
    """Стерео (2, N): без тишины по краям, самый громкий фрагмент длиной в окно модели, -23 LUFS."""
    y = to_stereo(y)
    _, (start, end) = librosa.effects.trim(librosa.to_mono(y), top_db=60.0)
    y = select_excerpt(y[:, start:end], sr, window_seconds(sr, max_len))
    return normalize_lufs(y, sr, MODEL_LUFS)


def pad_mel(mel: np.ndarray, max_len: int = DEFAULT_MAX_LEN) -> np.ndarray:
    """Приводит мел-спектрограмму (2, n_mels, T) к длине max_len, дополняя тишиной (-1.0)."""
    channels, n_mels, t = mel.shape
    if t >= max_len:
        return mel[:, :, :max_len].astype(np.float32)
    padded = np.full((channels, n_mels, max_len), -1.0, dtype=np.float32)
    padded[:, :, :t] = mel
    return padded


def build_model_inputs(mel_src: np.ndarray, mel_ref: np.ndarray, desc_src: np.ndarray,
                       desc_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Собирает входные массивы для MixingAIModel v2:
      - x_spec: (6, 128, T) -> [mel_src (2), mel_ref (2), (mel_ref - mel_src) / 2.0 (2)]
      - x_desc: (54,) -> [desc_src (18), desc_ref (18), (desc_ref - desc_src) / 2.0 (18)]
    """
    delta_mel = np.clip((mel_ref - mel_src) * 0.5, -1.0, 1.0)
    x_spec = np.concatenate([mel_src, mel_ref, delta_mel], axis=0).astype(np.float32)

    delta_desc = np.clip((desc_ref - desc_src) * 0.5, -1.0, 1.0)
    x_desc = np.concatenate([desc_src, desc_ref, delta_desc], axis=0).astype(np.float32)
    return x_spec, x_desc


def signal_features(y: np.ndarray, processor: AudioProcessor) -> tuple[np.ndarray, np.ndarray]:
    """Мел-спектрограмма Mid/Side (2, 128, T) и 18 DSP-дескрипторов подготовленного сигнала."""
    return (processor.get_stereo_mel_spectrogram(y, n_mels=N_MELS),
            processor.extract_mixing_descriptors(y))


def model_inputs(y_src: np.ndarray, y_ref: np.ndarray, processor: AudioProcessor,
                 max_len: int = DEFAULT_MAX_LEN) -> tuple[np.ndarray, np.ndarray]:
    """
    Входы модели (x_spec (6, 128, max_len), x_desc (54,)) по сырым сигналам исходника и референса
    любой длины, громкости и канальности.
    """
    mel_src, desc_src = signal_features(prepare_model_audio(y_src, processor.sr, max_len), processor)
    mel_ref, desc_ref = signal_features(prepare_model_audio(y_ref, processor.sr, max_len), processor)
    return build_model_inputs(pad_mel(mel_src, max_len), pad_mel(mel_ref, max_len), desc_src, desc_ref)
