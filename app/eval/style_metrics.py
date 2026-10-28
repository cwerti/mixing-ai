"""
Метрика стиля сведения, не зависящая от исполнения.

Референс в реальной работе — вокал из чужой песни, поэтому сравнивать результат с ним по сэмплам нельзя.
Сравниваются физические характеристики стиля: долгосрочный спектр, динамика, количество пространства,
ширина и степень тюна. Все величины в дБ / центах / долях, без нормировки в [-1, 1].
"""
import numpy as np
import librosa
from scipy import signal

from app.audio.audio_processor import AudioProcessor

# Терцоктавные полосы долгосрочного спектра (центральные частоты, Гц)
LTAS_CENTERS_HZ = 1000.0 * 2.0 ** (np.arange(-14, 11) / 3.0)  # 63 Гц .. 10 кГц
LTAS_EDGE_FACTOR = 2.0 ** (1.0 / 6.0)

# Масштаб «заметной на слух разницы» для каждого признака: разница в 1 масштаб = 1 единица расстояния
FEATURE_SCALES = {
    "ltas_mae_db": 1.5,
    "crest_db": 3.0,
    "dr_p95_p10_db": 3.0,
    "dr_p99_p90_db": 1.5,
    "decay_fast_db": 2.0,
    "reverb_tail_db": 3.0,
    "delay_peak": 0.1,
    "width_global": 0.05,
    "width_high": 0.05,
    "width_notes_db": 1.5,
    "width_tails_db": 3.0,
    "tail_brightness_db": 3.0,
    "sib_p90_db": 3.0,
    "air_ratio_db": 3.0,
    "pitch_cents": 5.0,
}

DEFAULT_WEIGHTS = {
    "ltas_mae_db": 2.0,
    "crest_db": 1.0,
    "dr_p95_p10_db": 1.0,
    "dr_p99_p90_db": 0.5,
    "decay_fast_db": 2.0,
    "reverb_tail_db": 1.0,
    "delay_peak": 0.5,
    "width_global": 1.0,
    "width_high": 0.5,
    "width_notes_db": 1.0,
    "width_tails_db": 0.5,
    "tail_brightness_db": 0.5,
    "sib_p90_db": 1.0,
    "air_ratio_db": 1.0,
    "pitch_cents": 1.0,
}

_RAW_KEYS = [
    "crest_db", "dr_p95_p10_db", "dr_p99_p90_db", "reverb_tail_db", "delay_peak",
    "width_global", "width_high", "sib_p90_db", "air_ratio_db",
]


def _to_channels(y: np.ndarray) -> np.ndarray:
    """Приводит сигнал к форме (channels, samples)."""
    if y.ndim == 1:
        return y[np.newaxis, :]
    return y if y.shape[0] <= 2 else y.T


def ltas_bands_db(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Долгосрочный спектр по терцоктавным полосам в дБ относительно общей энергии.
    Учитываются только активные кадры (громче -40 дБ от P95), чтобы паузы не размывали тембр.
    """
    mid = _to_channels(y).mean(axis=0)
    power = np.abs(librosa.stft(mid, n_fft=4096, hop_length=1024)) ** 2
    frame_db = 10.0 * np.log10(power.sum(axis=0) + 1e-12)
    active = frame_db > np.percentile(frame_db, 95) - 40.0
    spectrum = power[:, active].mean(axis=1) if active.any() else power.mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)

    bands = np.array([
        spectrum[(freqs >= fc / LTAS_EDGE_FACTOR) & (freqs < fc * LTAS_EDGE_FACTOR)].sum()
        for fc in LTAS_CENTERS_HZ
    ])
    return 10.0 * np.log10(bands / (spectrum.sum() + 1e-12) + 1e-12)


def fastest_decay_db(y: np.ndarray, sr: int) -> float:
    """
    Насколько быстро может падать громкость: P2 изменения уровня за ~46 мс по кадрам, начинающимся
    не тише P95 - 30 дБ. У сухого голоса окончания слогов обрываются резко (-20..-25 дБ), реверб и дилей
    ограничивают спад своим затуханием (-8..-13 дБ). Пауз для расчета не требуется, а цифровая тишина
    вырезанных пауз и гейт не влияют на результат (уровень снизу ограничен P95 - 60 дБ).
    """
    hop = 512
    mid = _to_channels(y).mean(axis=0)
    level = 20.0 * np.log10(librosa.feature.rms(y=mid, frame_length=1024, hop_length=hop)[0] + 1e-12)
    audible = level[level > -100.0]
    if len(audible) < 16:
        return 0.0
    p95 = np.percentile(audible, 95)
    level = np.maximum(level, p95 - 60.0)
    lag = max(1, int(round(0.046 * sr / hop)))
    change = level[lag:] - level[:-lag]
    from_phrase = level[:-lag] > p95 - 30.0
    return float(np.percentile(change[from_phrase], 2)) if from_phrase.sum() >= 16 else 0.0


# Нижняя граница Side/Mid: у моно-сигнала Side = 0, и без нее разница с любым стерео-референсом бесконечна
WIDTH_FLOOR_DB = -30.0


def width_and_tails(y: np.ndarray, sr: int) -> dict:
    """
    Пространство раздельно на нотах и в паузах. Средняя ширина по треку их смешивает: широкий хвост реверба
    в паузах маскирует узкий голос на нотах, хотя на слух это разное пространство (голос «в центре
    с хвостом» против широкого голоса).

    Returns:
        dict: width_notes_db / width_tails_db — Side/Mid (выше 150 Гц) на громких кадрах (выше P60)
            и в паузах (P10..P35, не тише P95 - 60 дБ); tail_brightness_db — доля 1-8 кГц в паузах.
    """
    channels = _to_channels(y)
    if channels.shape[0] == 1:
        channels = np.vstack([channels, channels])
    sos = signal.butter(4, 150.0, btype="highpass", fs=sr, output="sos")
    mid = signal.sosfilt(sos, 0.5 * (channels[0] + channels[1]))
    side = signal.sosfilt(sos, 0.5 * (channels[0] - channels[1]))

    hop = 1024
    rms_mid = librosa.feature.rms(y=mid, frame_length=2048, hop_length=hop)[0]
    rms_side = librosa.feature.rms(y=side, frame_length=2048, hop_length=hop)[0]
    level = 20.0 * np.log10(rms_mid + 1e-12)
    audible = level > np.percentile(level, 95) - 60.0
    notes = level > np.percentile(level[audible], 60)
    tails = audible & (level >= np.percentile(level[audible], 10)) & (level <= np.percentile(level[audible], 35))

    def width_db(frames: np.ndarray) -> float:
        if not frames.any():
            return WIDTH_FLOOR_DB
        ratio = np.sqrt(np.mean(rms_side[frames] ** 2) / (np.mean(rms_mid[frames] ** 2) + 1e-20))
        return float(max(WIDTH_FLOOR_DB, 20.0 * np.log10(ratio + 1e-12)))

    power = np.abs(librosa.stft(mid, n_fft=2048, hop_length=hop)) ** 2 + np.abs(
        librosa.stft(side, n_fft=2048, hop_length=hop)) ** 2
    n = min(power.shape[1], len(tails))
    tail_power = power[:, :n][:, tails[:n]].sum(axis=1) if tails[:n].any() else power.sum(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    bright = tail_power[(freqs >= 1000.0) & (freqs < 8000.0)].sum() / (tail_power.sum() + 1e-12)
    return {"width_notes_db": width_db(notes), "width_tails_db": width_db(tails),
            "tail_brightness_db": float(10.0 * np.log10(bright + 1e-12))}


def pitch_grid_deviation_cents(y: np.ndarray, sr: int, max_seconds: float = 60.0) -> float:
    """
    Медианное отклонение F0 от полутоновой сетки в центах с поправкой на общую расстройку трека.
    Живой вокал: ~15-25 центов, мягкий тюн: ~8-12, жесткий тюн: < 5.
    """
    mid = _to_channels(y).mean(axis=0)[: int(max_seconds * sr)]
    mid = librosa.resample(mid, orig_sr=sr, target_sr=22050) if sr != 22050 else mid
    # pyin надежно определяет вокализацию, но квантует F0 с шагом 10 центов, поэтому частоту берем из yin,
    # отбрасывая кадры, где оценки расходятся больше чем на полтона (октавные ошибки)
    f0_pyin, voiced, voiced_prob = librosa.pyin(mid, fmin=65.0, fmax=1000.0, sr=22050,
                                                frame_length=2048, hop_length=256)
    f0_yin = librosa.yin(mid, fmin=65.0, fmax=1000.0, sr=22050, frame_length=2048, hop_length=256)
    n = min(len(f0_pyin), len(f0_yin))
    agree = np.abs(1200.0 * np.log2(np.nan_to_num(f0_pyin[:n], nan=1.0) / f0_yin[:n])) < 50.0
    f0 = f0_yin[:n][voiced[:n] & (voiced_prob[:n] > 0.7) & agree]
    if len(f0) < 20:
        return float("nan")
    offset = ((1200.0 * np.log2(f0 / 440.0) + 50.0) % 100.0) - 50.0
    detune = np.median(offset)
    return float(np.median(np.abs(((offset - detune + 50.0) % 100.0) - 50.0)))


def extract_style_features(y: np.ndarray, sr: int, with_pitch: bool = True) -> dict:
    """
    Извлекает признаки стиля сведения.

    Args:
        y: Сигнал (моно 1D или стерео (2, N) / (N, 2)).
        sr: Частота дискретизации.
        with_pitch: Считать ли отклонение от нотной сетки (pyin медленный, в цикле оптимизации не нужен).

    Returns:
        dict: Скалярные признаки и массив "ltas_db".
    """
    raw = AudioProcessor(sr=sr).raw_mixing_descriptors(_to_channels(y))
    if raw is None:
        raise ValueError("Сигнал слишком короткий или тихий для анализа стиля")

    features = {key: float(raw[key]) for key in _RAW_KEYS}
    features["ltas_db"] = ltas_bands_db(y, sr)
    features["decay_fast_db"] = fastest_decay_db(y, sr)
    features.update(width_and_tails(y, sr))
    if with_pitch:
        features["pitch_cents"] = pitch_grid_deviation_cents(y, sr)
    return features


def style_distance(features: dict, reference: dict, weights: dict | None = None) -> tuple[float, dict]:
    """
    Взвешенное расстояние по стилю. Каждая разница делится на масштаб заметности из FEATURE_SCALES.

    Returns:
        tuple: (итоговое расстояние, {признак: разница в единицах масштаба со знаком "результат - референс"}).
    """
    weights = DEFAULT_WEIGHTS if weights is None else weights
    per_feature = {}
    for name, weight in weights.items():
        if weight <= 0:
            continue
        if name == "ltas_mae_db":
            if "ltas_db" not in features or "ltas_db" not in reference:
                continue
            diff = float(np.mean(np.abs(features["ltas_db"] - reference["ltas_db"])))
        else:
            if name not in features or name not in reference:
                continue
            if np.isnan(features[name]) or np.isnan(reference[name]):
                continue
            diff = features[name] - reference[name]
        per_feature[name] = diff / FEATURE_SCALES[name]

    total_weight = sum(weights[name] for name in per_feature)
    total = sum(weights[name] * abs(value) for name, value in per_feature.items()) / max(total_weight, 1e-8)
    return float(total), per_feature
