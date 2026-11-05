"""
Обучение сети по звуку для эффектов тона (компрессор, де-эссер, мультибэнд, дисторшн).

Ручки этих эффектов не идентифицируются по готовому звуку: разные настройки компрессора дают почти одинаковый
результат, и регрессия к одной «правильной» точке учит сеть среднему. Здесь ошибка считается по звуку:
предсказанная цепочка тона, примененная к сухому фрагменту исходника дифференцируемыми эффектами
(app/ml/ddsp.py), сравнивается с той же цепочкой с истинными ручками на том же фрагменте. Любые настройки,
звучащие как истинные, одинаково хороши.

EQ в обоих путях берется из метки (teacher forcing): на инференсе его ставит подгонка спектра, а не сеть, и
так ошибка по звуку отражает только динамику и сатурацию. Тюн, хорус, ширина, ревер, дилей и экситер учатся
прежней регрессией — они не дифференцируются или не входят в тон.
"""
from __future__ import annotations

import random

import numpy as np
import torch

from app.dataset.dataset_generator import load_chunk_at
from app.dataset.dataset_schema import CHAIN_ORDER, DATAGEN_PLUGINS, PARAM_LAYOUT, PARAM_OFFSETS
from app.ml.dataset import MixingAIDataset
from app.ml.ddsp import TONE_EFFECTS, render_tone_chain

TONE_NAMES = tuple(TONE_EFFECTS)
EQ_PARAMS = tuple(DATAGEN_PLUGINS["eq"]["params"])
# Нейтральный EQ (усиления 0 дБ при симметричном диапазоне) для сэмплов, где EQ выключен
_EQ_FLAT = np.array([0.5] * len(EQ_PARAMS), dtype=np.float32)


def tone_mask() -> torch.Tensor:
    """Маска позиций вектора параметров, которые учатся по звуку."""
    mask = torch.zeros(sum(len(v) for v in PARAM_LAYOUT.values()))
    for name in TONE_NAMES:
        mask[PARAM_OFFSETS[name]:PARAM_OFFSETS[name] + len(PARAM_LAYOUT[name])] = 1.0
    return mask


class ToneDataset(MixingAIDataset):
    """
    MixingAIDataset (входы сети и метка) + сухой фрагмент исходника длиной crop_s и параметры EQ метки.
    Фрагмент перечитывается по координатам src_chunk (датасеты v6+), аудио в датасете не хранится.
    """

    def __init__(self, dataset_dir, split: str, sr: int = 44100, crop_s: float = 3.0, **kwargs):
        super().__init__(dataset_dir, split=split, **kwargs)
        self.samples = [s for s in self.samples if s.get("src_chunk")]
        if not self.samples:
            raise ValueError(f"В {dataset_dir} нет сэмплов с src_chunk (нужен датасет v6+)")
        self.sr = sr
        self.crop = int(crop_s * sr)

    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        sample = self.samples[idx]
        audio = load_chunk_at(sample["src_chunk"], self.sr)
        if audio.shape[1] > self.crop:
            # Обучение — случайное окно, валидация — фиксированное из середины (сравнимые эпохи)
            start = (random.randint(0, audio.shape[1] - self.crop) if self.augment
                     else (audio.shape[1] - self.crop) // 2)
            audio = audio[:, start:start + self.crop]
        eq = sample["params"].get("eq")
        eq_vec = np.array([eq[p] for p in EQ_PARAMS], dtype=np.float32) if eq else _EQ_FLAT
        return (*item, torch.from_numpy(np.ascontiguousarray(audio)), torch.from_numpy(eq_vec))


def tone_params(vector: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
    """Вектор параметров модели (batch, P) → параметры эффектов тона для render_tone_chain."""
    return {name: {p: vector[:, PARAM_OFFSETS[name] + j] for j, p in enumerate(PARAM_LAYOUT[name])}
            for name in TONE_NAMES}


def render(audio: torch.Tensor, eq_vec: torch.Tensor, vector: torch.Tensor, on: torch.Tensor, sr: int
           ) -> torch.Tensor:
    """Цепочка тона на сухом фрагменте: EQ метки, эффекты тона с параметрами vector и вероятностями on."""
    params = tone_params(vector)
    params["eq"] = {p: eq_vec[:, j] for j, p in enumerate(EQ_PARAMS)}
    gates = {name: on[:, CHAIN_ORDER.index(name)] for name in TONE_NAMES}
    return render_tone_chain(audio, params, sr, on=gates)


def spectral_loss(pred: torch.Tensor, target: torch.Tensor, ffts: tuple[int, ...] = (2048, 512, 128)
                  ) -> torch.Tensor:
    """
    Многомасштабная ошибка спектра Mid и Side (лог-амплитуда, L1): длинное окно — тон, короткое — атаки и
    огибающая компрессора. Ошибка на сэмпл (batch,).
    """
    def ms(x):
        return torch.stack([x.mean(dim=1), 0.5 * (x[:, 0] - x[:, 1])], dim=1).flatten(0, 1)

    p, t = ms(pred), ms(target)
    total = 0.0
    for n_fft in ffts:
        window = torch.hann_window(n_fft, device=pred.device, dtype=pred.dtype)
        sp = torch.stft(p, n_fft, n_fft // 4, window=window, return_complex=True).abs()
        st = torch.stft(t, n_fft, n_fft // 4, window=window, return_complex=True).abs()
        err = (torch.log(sp + 1e-4) - torch.log(st + 1e-4)).abs().mean(dim=(1, 2))
        total = total + err.reshape(pred.shape[0], 2).mean(dim=1)
    return total / len(ffts)
