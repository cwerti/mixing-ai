"""
AFx-Rep (Steinmetz et al., ST-ITO, 2024; Apache-2.0) как готовый эмбеддинг аудиоэффектов — базовая линия
для scripts/eval_style_metric.py --afx <путь к afx-rep.ckpt>.

Веса: huggingface.co/csteinmetz1/afx-rep (afx-rep.ckpt, 1.16 ГБ — Lightning-чекпоинт с состоянием
оптимизатора; самой сети ~80 М параметров). Архитектура — PANNs Cnn14 с отдельными головами Mid и Side
(st_ito/models/panns.py), здесь переписана без torchlibrosa/pytorch_lightning: STFT берется из весов
чекпоинта (conv_real/conv_imag), log-mel -> minmax-нормировка [-80, 40] дБ -> [-1, 1]. Эмбеддинг одного
сигнала, как в ST-ITO: concat(L2(mid), L2(side)); окна по 262144 сэмпла (48 кГц) усредняются.
"""
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

AFX_SR = 48000
N_FFT, HOP = 2048, 1024
WINDOW = 262144


class _ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.bn1, self.bn2 = nn.BatchNorm2d(c_out), nn.BatchNorm2d(c_out)

    def forward(self, x, pool):
        x = F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))
        return F.avg_pool2d(x, pool)


class _Cnn14(nn.Module):
    def __init__(self, embed_dim: int = 512, mel_bins: int = 128):
        super().__init__()
        self.stft_real = nn.Parameter(torch.zeros(N_FFT // 2 + 1, 1, N_FFT), requires_grad=False)
        self.stft_imag = nn.Parameter(torch.zeros(N_FFT // 2 + 1, 1, N_FFT), requires_grad=False)
        self.melW = nn.Parameter(torch.zeros(N_FFT // 2 + 1, mel_bins), requires_grad=False)
        widths = [1, 64, 128, 256, 512, 1024, 2048]
        self.blocks = nn.ModuleList(_ConvBlock(a, b) for a, b in zip(widths[:-1], widths[1:]))
        self.fc_mid = nn.Linear(2048, embed_dim)
        self.fc_side = nn.Linear(2048, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2, N) L/R -> (B, 2 * embed_dim), concat нормированных Mid и Side."""
        b = x.shape[0]
        ms = torch.stack([(x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2], dim=1).reshape(b * 2, 1, -1)
        ms = F.pad(ms, (N_FFT // 2, N_FFT // 2), mode="reflect")
        power = F.conv1d(ms, self.stft_real, stride=HOP) ** 2 + F.conv1d(ms, self.stft_imag, stride=HOP) ** 2
        mel = torch.matmul(power.transpose(1, 2), self.melW)  # (2B, T, mel)
        h = 10.0 * torch.log10(mel.clamp_min(1e-10))
        h = ((h.clamp(-80.0, 40.0) + 80.0) / 120.0 * 2.0 - 1.0).unsqueeze(1)
        for i, block in enumerate(self.blocks):
            h = block(h, (2, 2) if i < 5 else (1, 1))
        h = h.mean(dim=3)
        h = h.max(dim=2).values + h.mean(dim=2)
        h = h.view(b, 2, -1)
        mid, side = self.fc_mid(h[:, 0]), self.fc_side(h[:, 1])
        return torch.cat([F.normalize(mid, dim=-1), F.normalize(side, dim=-1)], dim=-1)


def _load_state(path: Path) -> dict:
    sd = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
    out = {}
    for k, v in sd.items():
        if not k.startswith("encoder."):
            continue
        k = k[len("encoder."):]
        if k.startswith("bn0"):
            continue  # input_norm=minmax: bn0 не используется
        k = (k.replace("spectrogram_extractor.stft.conv_real.weight", "stft_real")
             .replace("spectrogram_extractor.stft.conv_imag.weight", "stft_imag")
             .replace("logmel_extractor.melW", "melW"))
        if k.startswith("conv_block"):
            idx = int(k[len("conv_block")]) - 1
            k = f"blocks.{idx}" + k[len("conv_blockN"):]
        out[k] = v
    return out


class AFxRep:
    """embed(y, sr) -> единичный вектор (среднее по окнам, перенормированное); distance как у StyleEncoder."""

    def __init__(self, checkpoint: str | Path, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = _Cnn14()
        self.net.load_state_dict(_load_state(Path(checkpoint)), strict=True)
        self.net = self.net.to(self.device).eval()

    @torch.no_grad()
    def embed(self, y: np.ndarray, sr: int) -> np.ndarray:
        from app.mixing.chain_optimizer import normalize_lufs, to_stereo
        y = normalize_lufs(to_stereo(np.asarray(y, dtype=np.float32)), sr, -23.0)
        y = librosa.resample(y, orig_sr=sr, target_sr=AFX_SR)
        if y.shape[1] < WINDOW:
            y = np.tile(y, (1, int(np.ceil(WINDOW / y.shape[1]))))[:, :WINDOW]
        starts = list(range(0, y.shape[1] - WINDOW + 1, WINDOW // 2))
        if starts[-1] != y.shape[1] - WINDOW:
            starts.append(y.shape[1] - WINDOW)
        emb = []
        for s in starts:
            x = torch.from_numpy(np.ascontiguousarray(y[:, s:s + WINDOW])).unsqueeze(0).to(self.device)
            emb.append(self.net(x)[0].cpu().numpy())
        mean = np.mean(emb, axis=0)
        return mean / (np.linalg.norm(mean) + 1e-12)
