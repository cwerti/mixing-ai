"""
Проверка пользовательского аудио на входе (CLI, пайплайн, GUI) до тяжелой обработки.

Короткий, тихий или битый файл должен давать понятное сообщение, а не трассировку из глубины librosa/pyloudnorm.
"""
from pathlib import Path

import librosa
import numpy as np

MIN_DURATION_S = 1.0
# -60 dBFS: ниже этого в записи нет полезного сигнала (тишина или шум цифрового нуля)
MIN_RMS = 1e-3


class AudioInputError(ValueError):
    """Входной аудиофайл непригоден для анализа (не читается, слишком короткий, тихий или с NaN/Inf)."""


def check_audio(y: np.ndarray, sr: int, label: str = "Аудио", min_duration_s: float = MIN_DURATION_S) -> None:
    """Проверяет уже загруженный сигнал (1D или (channels, samples)); бросает AudioInputError с причиной."""
    if y.size == 0:
        raise AudioInputError(f"{label}: файл не содержит аудиоданных.")
    if not np.all(np.isfinite(y)):
        raise AudioInputError(f"{label}: в файле есть некорректные отсчеты (NaN/Inf), файл поврежден.")
    duration = y.shape[-1] / sr
    if duration < min_duration_s:
        raise AudioInputError(
            f"{label}: запись слишком короткая ({duration:.2f} с), нужно не меньше {min_duration_s:.1f} с.")
    rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
    if rms < MIN_RMS:
        raise AudioInputError(f"{label}: запись пустая или почти беззвучная (RMS {rms:.1e}).")


def load_checked_audio(path: str | Path, sr: int, label: str = "Аудио", mono: bool = False,
                       min_duration_s: float = MIN_DURATION_S) -> np.ndarray:
    """Загружает файл через librosa и проверяет его; любые ошибки чтения превращаются в AudioInputError."""
    path = Path(path)
    if not path.is_file():
        raise AudioInputError(f"{label}: файл не найден: {path}")
    try:
        y, _ = librosa.load(path, sr=sr, mono=mono)
    except Exception as e:  # soundfile/audioread бросают разные типы на битых и неподдерживаемых файлах
        raise AudioInputError(f"{label}: не удалось прочитать файл {path.name} ({type(e).__name__}: {e}).") from e
    check_audio(y, sr, label, min_duration_s)
    return y
