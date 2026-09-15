import json
import torch
import random
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

from app.dataset.dataset_schema import (
    CHAIN_ORDER, PARAM_LAYOUT, PARAM_OFFSETS, PARAMETER_VECTOR_LENGTH
)

class MixingAIDataset(Dataset):
    """
    Класс датасета для PyTorch.
    Загружает сэмплы из метаданных index.jsonl и предвычисленные 
    спектрограммы Мелов (.npz), приводя их к фиксированной длине для батчинга.
    """

    def __init__(self, dataset_dir: str | Path, max_len: int = 700):
        """
        Инициализирует датасет.
        
        Args:
            dataset_dir: Путь к папке датасета (содержащей index.jsonl и папку features/).
            max_len: Фиксированная длина временной оси для батчинга спектрограмм.
        """
        self.dataset_dir = Path(dataset_dir)
        self.max_len = max_len
        self.samples = []
        
        index_path = self.dataset_dir / "index.jsonl"
        if not index_path.exists():
            raise FileNotFoundError(f"Файл индекса датасета не найден: {index_path}")
            
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def _pad_spectrogram(self, mel: np.ndarray) -> np.ndarray:
        """
        Дополняет или обрезает спектрограмму по оси времени до длины max_len.
        Дополнение выполняется минимальным значением спектра (фоновым шумом).
        
        Args:
            mel: Двумерный массив спектрограммы Мелов (128, T).
            
        Returns:
            np.ndarray: Спектрограмма фиксированной формы (128, max_len).
        """
        n_mels, t = mel.shape
        if t >= self.max_len:
            return mel[:, :self.max_len]
        
        # Заполняем пустоту минимальным значением (тишиной в шкале dB)
        pad_val = mel.min()
        padded = np.full((n_mels, self.max_len), pad_val, dtype=np.float32)
        padded[:, :t] = mel
        return padded

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Возвращает один тренировочный сэмпл.
        
        Returns:
            x: Тензор спектрограмм Dry и Wet вокала, форма (2, 128, max_len).
            y_chain: Тензор активности плагинов (one-hot), форма (4,).
            y_params: Тензор целевых параметров плагинов, форма (25,).
            param_mask: Маска активности параметров для Masked Loss, форма (25,).
        """
        sample_data = self.samples[idx]
        sample_id = sample_data["id"]
        
        # 1. Загрузка спектрограмм из NPZ
        npz_path = self.dataset_dir / "features" / f"{sample_id}.npz"
        features = np.load(npz_path)
        mel_dry = features["mel_dry"]
        mel_wet = features["mel_wet"]
        
        # 2. Выравнивание длины
        mel_dry_padded = self._pad_spectrogram(mel_dry)
        mel_wet_padded = self._pad_spectrogram(mel_wet)
        
        # Возвращаем только спектрограмму Wet (1 канал)
        # Аугментация высоты тона (Frequency Roll) для инвариантности к Pitch-Shift (Nightcore / Высокий тон)
        if random.random() < 0.6:
            shift = random.randint(-16, 16) # Сдвиг до +-16 бинов (~+-8 полутонов)
            mel_wet_padded = np.roll(mel_wet_padded, shift, axis=0)
            if shift > 0:
                mel_wet_padded[:shift, :] = mel_wet_padded.min()
            elif shift < 0:
                mel_wet_padded[shift:, :] = mel_wet_padded.min()
                
        x = mel_wet_padded[np.newaxis, :, :] # Форма (1, 128, max_len)
        
        # 3. Подготовка таргетов
        chain_onehot = np.array(sample_data["chain_onehot"], dtype=np.float32)
        params_vector = np.array(sample_data["params_vector"], dtype=np.float32)
        
        # 4. Динамическое построение маски активных параметров
        param_mask = np.zeros(PARAMETER_VECTOR_LENGTH, dtype=np.float32)
        for i, plugin_name in enumerate(CHAIN_ORDER):
            if chain_onehot[i] == 1 and plugin_name in PARAM_OFFSETS:
                offset = PARAM_OFFSETS[plugin_name]
                length = len(PARAM_LAYOUT[plugin_name])
                param_mask[offset : offset + length] = 1.0
                
        return (
            torch.from_numpy(x),
            torch.from_numpy(chain_onehot),
            torch.from_numpy(params_vector),
            torch.from_numpy(param_mask)
        )
