"""
Слепая оценка отпечатка эффекта сетью: мокрый вокал (без сухого) → вектор fingerprint_vector + период повторов.

Ручной оценщик (app/eval/blind_fingerprint.py) восстанавливает по концам слов ширину хвоста, декорреляцию L/R и
период дилея, но не спектр хвоста, DRR, shimmer и модуляцию; на очень мокрых миксах он почти не находит концов
слов (docs/research/blind_fingerprint.md). Сеть учится на синтетике, где истина бесплатна: эффект с известным
отпечатком (fingerprint()) рендерится на разных сухих вокалах Cambridge.

Вход: log-mel четырех каналов (L, R, Mid, Side; 64 полосы, hop 512 при 44.1 кГц, 8 с) в дБ относительно максимума Mid
+ автокорреляция спектрального потока по каналам L/R до 1.5 с (явная подсказка для периода повторов: сверточной
сети с глобальным пулингом трудно мерить задержку в сотни миллисекунд).
Выход: масштабированный вектор признаков FINGERPRINT_SCALES (как fingerprint_vector), логиты периода L/R по лагам
автокорреляции (1D-свертка по самой автокорреляции: сдвиг пика = сдвиг ответа; регрессия периода из глобального
вектора попадала в ±10 мс лишь в 8–14 % случаев) и логит «есть повторы» (echo_strength истины > ECHO_THRESHOLD).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from app.eval.effect_fingerprint import FINGERPRINT_SCALES, _DB_FLOOR, _LOG_KEYS

SR = 44100
SECONDS = 8.0
N_MELS = 64
HOP = 512
FRAMES = int(SECONDS * SR) // HOP + 1
MAX_LAG_S = 1.5
N_LAGS = int(MAX_LAG_S * SR / HOP)
MIN_LAG = 3  # меньшие лаги — гладкость самой огибающей, не повторы
ECHO_THRESHOLD = 0.5
# Признаки вектора отпечатка + общий RT60 (у ручного оценщика он один из немногих восстанавливаемых)
SCALES = {**FINGERPRINT_SCALES, "rt60_s": 0.3}
LOG_KEYS = _LOG_KEYS | {"rt60_s"}
KEYS = tuple(SCALES)


# ------------------------------------------------------------------------------------------------- цели

def target_vector(fp: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Масштабированный вектор признаков и маска «признак определен». В отличие от fingerprint_vector, пропуски не
    подменяются нулем: у хоруса или компрессора нет хвоста, и rt60/ned для них не определены — их не учим.
    """
    vec, mask = np.zeros(len(KEYS), np.float32), np.zeros(len(KEYS), np.float32)
    for i, key in enumerate(KEYS):
        v = fp.get(key)
        if v is None or not np.isfinite(v):
            continue
        if key in LOG_KEYS:
            v = np.log(max(v, 0.05))
        elif key.endswith("_db"):
            v = float(np.clip(v, _DB_FLOOR, -_DB_FLOOR))
        vec[i], mask[i] = float(v) / SCALES[key], 1.0
    return vec, mask


def vector_to_dict(vec: np.ndarray) -> dict:
    """Обратное преобразование масштабированного вектора в признаки отпечатка."""
    out = {}
    for i, key in enumerate(KEYS):
        v = float(vec[i]) * SCALES[key]
        out[key] = float(np.exp(v)) if key in LOG_KEYS else v
    return out


LAG_S = HOP / SR


def echo_targets(fp: dict) -> tuple[float, np.ndarray]:
    """(есть повторы 0/1, период L/R в лагах автокорреляции — дробный индекс кадра)."""
    has = float(fp.get("echo_strength", 0.0) or 0.0) > ECHO_THRESHOLD
    periods = [float(fp.get(f"echo_period_{c}_ms", 0.0) or 0.0) / 1000.0 / LAG_S for c in ("l", "r")]
    periods = np.clip(np.asarray(periods, np.float32), MIN_LAG, N_LAGS - 1)
    if not has:
        periods[:] = MIN_LAG
    return float(has), periods


def lag_from_logits(logits: np.ndarray) -> float:
    """Дробный лаг максимума: argmax + парабола по соседям (шаг лага 11.6 мс, нужна точность ±10 мс)."""
    i = int(np.argmax(logits))
    if 0 < i < len(logits) - 1:
        a, b, c = logits[i - 1], logits[i], logits[i + 1]
        den = a - 2 * b + c
        if den < 0:
            return float(i + 0.5 * (a - c) / den)
    return float(i)


# ------------------------------------------------------------------------------------------------- вход

# Каналы подсказки по лагам: поток L, R (+ сходство спектра «нота → хвост» L, R при with_lag_similarity).
# Обученная модель (docs/research/blind_fingerprint_net.md) — 2 канала; 4-канальный вариант не обучен.
N_ACF = 2
N_ACF_LAG_SIM = 4


def lag_similarity(mel: np.ndarray) -> np.ndarray:
    """
    (2, N_LAGS) — для каналов L/R: косинусное сходство спектра кадра t и кадра t+лаг, взвешенное спадом энергии
    между ними (кадр t+лаг тише — это пауза или хвост после ноты). Повтор дилея — копия ноты в паузе, поэтому
    сходство пиковое именно на его лаге; автокорреляция потока путает его с ритмом самого вокала.
    """
    out = np.zeros((2, N_LAGS), np.float32)
    for ch in (0, 1):
        x = mel[ch].astype(np.float32)
        energy = 10.0 * np.log10(np.mean(10.0 ** (x / 10.0), axis=0) + 1e-12)
        z = x - x.mean(axis=0, keepdims=True)
        z /= np.linalg.norm(z, axis=0, keepdims=True) + 1e-6
        for lag in range(MIN_LAG, N_LAGS):
            sim = np.sum(z[:, :-lag] * z[:, lag:], axis=0)
            w = np.clip(energy[:-lag] - energy[lag:], 0.0, 30.0) * (energy[:-lag] > -50.0)
            out[ch, lag] = float(np.sum(w * sim) / (np.sum(w) + 1e-6))
    return out

def _mel_basis() -> np.ndarray:
    import librosa
    return librosa.filters.mel(sr=SR, n_fft=2048, n_mels=N_MELS).astype(np.float32)


_MEL = None


def wet_features(wet: np.ndarray, sr: int = SR,
                 with_lag_similarity: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    (mel (4, N_MELS, FRAMES) в дБ относительно максимума Mid, порог -80; acf (N_ACF или N_ACF_LAG_SIM, N_LAGS)).
    Вход стерео (2, n) или моно; длина приводится к SECONDS (обрезка/дополнение тишиной).
    """
    global _MEL
    import librosa
    y = np.asarray(wet, dtype=np.float32)
    if y.ndim == 1:
        y = np.stack([y, y])
    if y.shape[0] != 2 and y.shape[-1] == 2:
        y = y.T
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    n = int(SECONDS * SR)
    y = y[:, :n] if y.shape[1] >= n else np.pad(y, ((0, 0), (0, n - y.shape[1])))
    if _MEL is None:
        _MEL = _mel_basis()
    chans = [y[0], y[1], 0.5 * (y[0] + y[1]), 0.5 * (y[0] - y[1])]
    mels = []
    for ch in chans:
        spec = np.abs(librosa.stft(ch, n_fft=2048, hop_length=HOP)) ** 2
        mels.append(_MEL @ spec)
    mels = 10.0 * np.log10(np.stack(mels) + 1e-12)
    mels = np.maximum(mels - mels[2].max(), -80.0)

    acf = []
    for ch in (0, 1):
        flux = np.maximum(np.diff(mels[ch], axis=1), 0.0).sum(axis=0)
        flux = flux - flux.mean()
        full = np.correlate(flux, flux, "full")[len(flux) - 1:]
        full = full / (full[0] + 1e-9)
        acf.append(full[:N_LAGS])
    acf = np.stack(acf).astype(np.float32)
    if with_lag_similarity:
        acf = np.concatenate([acf, lag_similarity(mels)])
    acf[:, :MIN_LAG] = 0.0
    return mels.astype(np.float32), acf


# ------------------------------------------------------------------------------------------------- модель

class BlindFingerprintNet(nn.Module):
    """Свертка по log-mel (частота × время) + MLP по автокорреляции → признаки, период, «есть повторы»."""

    def __init__(self, n_out: int = len(KEYS), n_lags: int = N_LAGS, width: int = 48, n_acf: int = N_ACF):
        super().__init__()
        self.n_acf = n_acf

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.GELU(),
                                 nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.GELU(),
                                 nn.MaxPool2d(2))
        self.conv = nn.Sequential(block(4, width), block(width, 2 * width), block(2 * width, 4 * width),
                                  block(4 * width, 4 * width))
        self.acf = nn.Sequential(nn.Linear(n_acf * n_lags, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU())
        feat = 4 * width * 2 + 128
        self.trunk = nn.Sequential(nn.Linear(feat, 384), nn.GELU(), nn.Dropout(0.2), nn.Linear(384, 256), nn.GELU())
        self.reg = nn.Linear(256, n_out)
        # Период: свертка по лагам автокорреляции (вход — автокорреляции L и R), выход — логиты L и R по лагам
        self.period = nn.Sequential(nn.Conv1d(n_acf, 32, 7, padding=3), nn.GELU(), nn.Conv1d(32, 32, 7, padding=3),
                                    nn.GELU(), nn.Conv1d(32, 2, 5, padding=2))
        self.echo = nn.Linear(256, 1)

    def forward(self, mel: torch.Tensor, acf: torch.Tensor):
        h = self.conv(mel / 40.0 + 1.0)
        h = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        h = torch.cat([h, self.acf(acf.flatten(1))], dim=1)
        h = self.trunk(h)
        period = self.period(acf * 4.0)
        period[..., :MIN_LAG] = -30.0  # лаги короче MIN_LAG — гладкость огибающей, не повторы
        return self.reg(h), period, self.echo(h).squeeze(1)


def save_model(path, model: BlindFingerprintNet, meta: dict) -> None:
    torch.save({"state": model.state_dict(), "meta": {**meta, "n_acf": model.n_acf}}, path)


def load_model(path, device: str = "cpu") -> BlindFingerprintNet:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = BlindFingerprintNet(n_acf=ckpt["meta"].get("n_acf", N_ACF))
    model.load_state_dict(ckpt["state"], strict=True)
    return model.eval().to(device)


@torch.no_grad()
def predict(model: BlindFingerprintNet, wet: np.ndarray, sr: int = SR) -> dict:
    """Отпечаток мокрого вокала: признаки fingerprint + echo_period_{l,r}_ms + echo_prob."""
    mel, acf = wet_features(wet, sr, with_lag_similarity=model.n_acf == N_ACF_LAG_SIM)
    device = next(model.parameters()).device
    reg, period, echo = model(torch.from_numpy(mel)[None].to(device), torch.from_numpy(acf)[None].to(device))
    out = vector_to_dict(reg[0].cpu().numpy())
    logits = period[0].cpu().numpy()
    out["echo_period_l_ms"], out["echo_period_r_ms"] = (lag_from_logits(logits[c]) * LAG_S * 1000.0 for c in (0, 1))
    out["echo_prob"] = float(torch.sigmoid(echo[0]))
    return out
