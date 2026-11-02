"""
Выученный эмбеддинг отпечатка эффекта: расстояние в нем должно предсказывать разницу на слух (log-mel мокрых
рендеров одного и того же сухого вокала) лучше, чем ручное fingerprint_distance.

Вход сети — ручные признаки отпечатка (fingerprint_vector) плюс энергетическая карта отклика на импульс: энергия
Mid и Side по 7 полосам в логарифмически растущих окнах времени (0–8 с). Карта несет то, чего нет в признаках:
форму нарастания и спада хвоста по полосам (диффузию, окраску), на которой ручные признаки путают режимы
Supermassive (docs/research/effect_fingerprint.md).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.signal import butter, sosfilt
from torch import nn

from app.eval.effect_fingerprint import BANDS_HZ, fingerprint_vector

# Границы окон карты: первые 2 мс — прямой звук, дальше логарифмически до 8 с
MAP_EDGES_S = np.concatenate([[0.0, 0.002], np.geomspace(0.01, 8.0, 20)])
MAP_FLOOR_DB = -80.0


def impulse_response(effect, sr: int = 48000, seconds: float = 8.0, amp: float = 0.5) -> np.ndarray:
    """Отклик (2, n) на стерео-импульс амплитуды amp, нормированный к единичному импульсу."""
    from app.eval.effect_fingerprint import _render
    x = np.zeros((2, int(seconds * sr)), dtype=np.float32)
    x[:, 0] = amp
    return _render(effect, x, sr) / amp


def response_map(h: np.ndarray, sr: int) -> np.ndarray:
    """
    Карта (2, полосы, окна) в дБ относительно полной энергии отклика: [Mid, Side] × BANDS_HZ × MAP_EDGES_S.
    Энергия окна делится на его длину — иначе длинные поздние окна выглядели бы громче ранних.
    """
    mid, side = h.mean(axis=0), 0.5 * (h[0] - h[1])
    total = max(float(np.sum(mid ** 2) + np.sum(side ** 2)), 1e-20)
    edges = (MAP_EDGES_S * sr).astype(int)
    out = np.full((2, len(BANDS_HZ), len(edges) - 1), MAP_FLOOR_DB, dtype=np.float32)
    for c, sig in enumerate((mid, side)):
        for b, center in enumerate(BANDS_HZ):
            lo, hi = center / np.sqrt(2.0), min(center * np.sqrt(2.0), 0.45 * sr)
            band = sosfilt(butter(4, [lo, hi], "bandpass", fs=sr, output="sos"), sig)
            for k in range(len(edges) - 1):
                a, z = edges[k], min(edges[k + 1], len(band))
                if z <= a:
                    continue
                e = float(np.sum(band[a:z] ** 2)) / ((z - a) / sr) / total
                out[c, b, k] = max(10.0 * np.log10(e + 1e-20), MAP_FLOOR_DB)
    return out


def embedding_input(fp: dict, rmap: np.ndarray) -> np.ndarray:
    """Вход сети: признаки отпечатка (уже масштабированы) + карта в единицах 20 дБ."""
    return np.concatenate([fingerprint_vector(fp), (rmap.ravel() - MAP_FLOOR_DB / 2) / 20.0]).astype(np.float32)


class FingerprintEmbedder(nn.Module):
    """MLP: вход → эмбеддинг; расстояние между эффектами — евклидово между эмбеддингами (в дБ log-mel)."""

    def __init__(self, in_dim: int, hidden: int = 128, dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.config = {"in_dim": in_dim, "hidden": hidden, "dim": dim, "dropout": dropout}
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def embed(model: FingerprintEmbedder, inputs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(((inputs - mean) / std).astype(np.float32))
        return model(x).numpy()


def save_embedder(path, model: FingerprintEmbedder, mean: np.ndarray, std: np.ndarray, meta: dict) -> None:
    torch.save({"state": model.state_dict(), "mean": mean, "std": std, "config": model.config, "meta": meta}, path)


def load_embedder(path) -> tuple[FingerprintEmbedder, np.ndarray, np.ndarray]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = FingerprintEmbedder(**ckpt["config"])
    model.load_state_dict(ckpt["state"], strict=True)
    model.eval()
    return model, ckpt["mean"], ckpt["std"]
