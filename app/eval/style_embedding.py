"""
Обученная метрика стиля обработки: кодировщик одного сигнала -> нормированный эмбеддинг, расстояние = 1 - cos.

Ручная метрика (style_metrics.style_distance) в основном меряет само спетое (содержание фрагмента), а не
обработку: на реальных парах истинный wet до референса из другой части песни дальше, чем dry до wet.
Кодировщик обучен контрастно (scripts/train_style_encoder.py): одинаковая цепочка на разном содержании
(другие части песни, другие певцы) — близко, разные цепочки на том же содержании — далеко. Сухого
референса на инференсе нет, поэтому кодировщик смотрит на один сигнал.

Вход: log-mel Mid и Side (64 полосы, окно 2048, шаг 512) после выравнивания громкости к -23 LUFS;
длинный сигнал режется на окна ~3 с с шагом ~1.5 с, эмбеддинги окон усредняются.
"""
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
import torch
from torch import nn

from app.mixing.chain_optimizer import normalize_lufs, to_stereo

SR = 44100
N_FFT, HOP, N_MELS = 2048, 512, 64
FMIN, FMAX = 30.0, 20000.0
FEATURE_LUFS = -23.0
WINDOW_FRAMES = 256  # ~3 с
WINDOW_HOP_FRAMES = 128
DB_FLOOR = -100.0
DEFAULT_CHECKPOINT = Path("data/models/style_encoder/style_encoder.pt")
# Перевод косинусного расстояния в единицы, сопоставимые со style_distance (штраф за интенсивность 0.15,
# порог упрощения 0.02 и min_improvement 0.01 в ChainOptimizer рассчитаны на них). Калибровка — в
# docs/research/style_metric_v2.md.
EMBEDDING_SCALE = 10.0
COMBINED_STYLE_WEIGHT = 0.5


@lru_cache(maxsize=4)
def _mel_basis(sr: int) -> np.ndarray:
    return librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX).astype(np.float32)


def style_features(y: np.ndarray, sr: int = SR, normalize: bool = True) -> np.ndarray:
    """
    log-mel Mid/Side (2, N_MELS, T), масштабированный к ~[-3, 3]: (дБ + 40) / 20.

    Args:
        y: Сигнал 1D или (2, N) / (N, 2).
        normalize: Выровнять громкость к FEATURE_LUFS (кодировщик обучен на выровненных сигналах).
    """
    y = to_stereo(np.asarray(y, dtype=np.float32))
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    if normalize:
        y = normalize_lufs(y, SR, FEATURE_LUFS)
    mid, side = 0.5 * (y[0] + y[1]), 0.5 * (y[0] - y[1])
    basis = _mel_basis(SR)
    out = []
    for ch in (mid, side):
        power = np.abs(librosa.stft(ch, n_fft=N_FFT, hop_length=HOP)) ** 2
        db = 10.0 * np.log10(np.maximum(basis @ power, 1e-10))
        out.append((np.maximum(db, DB_FLOOR) + 40.0) / 20.0)
    return np.stack(out).astype(np.float32)


def _conv(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class StyleNet(nn.Module):
    """
    Небольшая CNN по (2, N_MELS, T): свертки сохраняют ось частот (EQ частотно-специфичен), по времени —
    статистический пулинг (среднее и стандартное отклонение), который отбрасывает порядок нот.
    """

    def __init__(self, n_mels: int = N_MELS, width: int = 32, embed_dim: int = 128, n_flags: int = 12):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            _conv(2, w), _conv(w, w), nn.MaxPool2d(2),
            _conv(w, 2 * w), _conv(2 * w, 2 * w), nn.MaxPool2d(2),
            _conv(2 * w, 4 * w), _conv(4 * w, 4 * w), nn.MaxPool2d(2),
            _conv(4 * w, 8 * w), nn.MaxPool2d((2, 1)),
        )
        f_out = n_mels // 16
        self.proj = nn.Sequential(nn.Linear(2 * 8 * w * f_out, 256), nn.ReLU(inplace=True))
        self.embed = nn.Linear(256, embed_dim)
        self.flags = nn.Linear(256, n_flags)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)  # (B, C, F, T)
        stats = torch.cat([h.mean(dim=3), h.std(dim=3)], dim=1)  # (B, 2C, F)
        return self.proj(stats.flatten(1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return nn.functional.normalize(self.embed(h), dim=1), self.flags(h)


def windows(features: np.ndarray, frames: int = WINDOW_FRAMES, hop: int = WINDOW_HOP_FRAMES) -> np.ndarray:
    """Окна (n, 2, N_MELS, frames); короткий сигнал повторяется до длины окна."""
    t = features.shape[-1]
    if t < frames:
        features = np.concatenate([features] * int(np.ceil(frames / max(t, 1))), axis=-1)[..., :frames]
        t = frames
    starts = list(range(0, t - frames + 1, hop))
    if starts[-1] != t - frames:
        starts.append(t - frames)
    return np.stack([features[..., s:s + frames] for s in starts])


class StyleEncoder:
    """Загруженный кодировщик стиля: embed(y) -> единичный вектор, distance(a, b) = 1 - cos."""

    def __init__(self, checkpoint: str | Path = DEFAULT_CHECKPOINT, device: str = "cpu",
                 net: StyleNet | None = None):
        self.device = torch.device(device)
        if net is None:
            state = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
            net = StyleNet(**state.get("config", {}))
            net.load_state_dict(state["model"])
        self.net = net.to(self.device).eval()

    @torch.no_grad()
    def embed_features(self, features: np.ndarray) -> np.ndarray:
        batch = torch.from_numpy(windows(features)).to(self.device)
        emb, _ = self.net(batch)
        mean = emb.mean(dim=0)
        return (mean / mean.norm().clamp_min(1e-8)).cpu().numpy()

    def embed(self, y: np.ndarray, sr: int = SR) -> np.ndarray:
        return self.embed_features(style_features(y, sr))

    @staticmethod
    def distance(a: np.ndarray, b: np.ndarray) -> float:
        """Косинусное расстояние между эмбеддингами (0 — одинаково, 2 — противоположно)."""
        return float(max(0.0, 1.0 - float(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))


@lru_cache(maxsize=2)
def load_encoder(checkpoint: str = str(DEFAULT_CHECKPOINT), device: str = "cpu") -> StyleEncoder:
    """Кодировщик из чекпоинта (кэшируется на процесс)."""
    return StyleEncoder(checkpoint, device=device)


def combined_distance(style: float, embedding: float) -> float:
    """Комбинация ручной метрики и косинусного расстояния кодировщика в единицах style_distance."""
    return COMBINED_STYLE_WEIGHT * style + (1.0 - COMBINED_STYLE_WEIGHT) * EMBEDDING_SCALE * embedding


def embedding_distance(y: np.ndarray, ref: np.ndarray, sr: int = SR, encoder: StyleEncoder | None = None) -> float:
    """Расстояние по стилю обработки между сигналом и референсом (1 - cos эмбеддингов)."""
    encoder = encoder or load_encoder()
    return encoder.distance(encoder.embed(y, sr), encoder.embed(ref, sr))
