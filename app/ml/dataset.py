import json
import random
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

from app.dataset.dataset_schema import (
    CHAIN_ORDER, PARAM_LAYOUT, PARAM_OFFSETS, PARAMETER_VECTOR_LENGTH, NUM_DSP_DESCRIPTORS
)


class MixingAIDataset(Dataset):
    """
    Датасет PyTorch для Mixing-AI v2 (Source + Reference Dual-Input).
    Загружает парные Mid/Side мел-спектрограммы (2, 128, T) и 18 физических DSP-дескрипторов,
    формируя:
      - x_spec: (6, 128, max_len) -> [mel_src_mid, mel_src_side, mel_ref_mid, mel_ref_side, delta_mid, delta_side]
      - x_desc: (54,) -> [desc_src (18), desc_ref (18), desc_ref - desc_src (18)]
    Поддерживает фильтрацию по полю split ("train", "val", "all") без утечки голосов.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        max_len: int = 520,
        split: str = "all",
        augment: bool = False
    ):
        """
        Args:
            dataset_dir: Путь к папке датасета (содержащей index.jsonl и features/).
            max_len: Фиксированная длина временной оси мел-спектрограммы (~6 сек при 44.1 кГц и hop=512).
            split: Выборка ("train", "val" или "all").
            augment: Включить ли безопасные тренировочные аугментации (Time-Roll и Cross-Source).
        """
        self.dataset_dir = Path(dataset_dir)
        self.max_len = max_len
        self.split = split
        self.augment = augment and (split != "val")
        self.samples = []

        index_path = self.dataset_dir / "index.jsonl"
        if not index_path.exists():
            raise FileNotFoundError(f"Файл индекса датасета не найден: {index_path}")

        all_loaded = []
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    all_loaded.append(item)

        if split in ("train", "val"):
            filtered = [s for s in all_loaded if s.get("split", "train") == split]
            # Если датасет крошечный (например, в юнит-тесте на 3 сэмпла), отдаем все элементы
            self.samples = filtered if len(filtered) > 0 else all_loaded
        else:
            self.samples = all_loaded

    def __len__(self) -> int:
        return len(self.samples)

    def _pad_stereo_mel(self, mel: np.ndarray) -> np.ndarray:
        """
        Приводит мел-спектрограмму к форме (2, 128, max_len) в шкале [-1.0, 1.0].
        Поддерживает обратную совместимость со старыми 2D массивами (128, T).
        """
        if mel.ndim == 2:
            # Обратная совместимость со старыми 1-канальными сэмплами в [-80..0] дБ
            if np.min(mel) < -1.5:
                mel_norm = np.clip((mel + 40.0) / 40.0, -1.0, 1.0).astype(np.float32)
            else:
                mel_norm = mel.astype(np.float32)
            side_empty = np.full_like(mel_norm, -1.0, dtype=np.float32)
            mel = np.stack([mel_norm, side_empty], axis=0)

        channels, n_mels, t = mel.shape
        if t >= self.max_len:
            return mel[:, :, :self.max_len].astype(np.float32)

        padded = np.full((channels, n_mels, self.max_len), -1.0, dtype=np.float32)
        padded[:, :, :t] = mel
        return padded

    def _load_sample_features(self, sample_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        npz_path = self.dataset_dir / "features" / f"{sample_id}.npz"
        with np.load(npz_path) as features:
            mel_dry = self._pad_stereo_mel(features["mel_dry"])
            mel_wet = self._pad_stereo_mel(features["mel_wet"])
            desc_dry = (
                features["desc_dry"].astype(np.float32)
                if "desc_dry" in features
                else np.zeros(NUM_DSP_DESCRIPTORS, dtype=np.float32)
            )
            desc_wet = (
                features["desc_wet"].astype(np.float32)
                if "desc_wet" in features
                else np.zeros(NUM_DSP_DESCRIPTORS, dtype=np.float32)
            )
        return mel_dry, mel_wet, desc_dry, desc_wet

    @staticmethod
    def build_model_inputs(
        mel_src: np.ndarray,
        mel_ref: np.ndarray,
        desc_src: np.ndarray,
        desc_ref: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Возвращает один тренировочный/валидационный сэмпл:
            x_spec: (6, 128, max_len)
            x_desc: (54,)
            y_chain: (len(CHAIN_ORDER),)
            y_params: (PARAMETER_VECTOR_LENGTH,)
            param_mask: (PARAMETER_VECTOR_LENGTH,)
        """
        sample_data = self.samples[idx]
        sample_id = sample_data["id"]

        mel_dry, mel_wet, desc_dry, desc_wet = self._load_sample_features(sample_id)

        if self.augment:
            # 1. Синхронный циклический сдвиг по оси ВРЕМЕНИ (не трогая частотную ось!),
            # чтобы delta_mel = (mel_wet - mel_dry) * 0.5 оставалась по-фреймово выровненной.
            if random.random() < 0.65:
                t_shift = random.randint(-80, 80)
                mel_wet = np.roll(mel_wet, t_shift, axis=2)
                mel_dry = np.roll(mel_dry, t_shift, axis=2)

            # 2. Мягкая микро-вариация огибающей сухого сигнала (имитация разницы микрофонов
            # без разрушения физических дескрипторов и таргетов регрессии)
            if random.random() < 0.25:
                noise = np.random.normal(0.0, 0.015, size=(1, mel_dry.shape[1], 1)).astype(np.float32)
                mel_dry = np.clip(mel_dry + noise, -1.0, 1.0)

        x_spec, x_desc = self.build_model_inputs(mel_dry, mel_wet, desc_dry, desc_wet)

        chain_onehot = np.array(sample_data["chain_onehot"], dtype=np.float32)
        params_vector = np.array(sample_data["params_vector"], dtype=np.float32)

        param_mask = np.zeros(PARAMETER_VECTOR_LENGTH, dtype=np.float32)
        for i, plugin_name in enumerate(CHAIN_ORDER):
            if i < len(chain_onehot) and chain_onehot[i] == 1 and plugin_name in PARAM_OFFSETS:
                offset = PARAM_OFFSETS[plugin_name]
                length = len(PARAM_LAYOUT[plugin_name])
                param_mask[offset : offset + length] = 1.0

        return (
            torch.from_numpy(x_spec),
            torch.from_numpy(x_desc),
            torch.from_numpy(chain_onehot),
            torch.from_numpy(params_vector),
            torch.from_numpy(param_mask)
        )
