import os
import tempfile
import json
import numpy as np
import pytest
import soundfile as sf
import torch
from pathlib import Path

from app.dataset.dataset_generator import DatasetGenerator
from app.dataset.dataset_schema import PARAMETER_VECTOR_LENGTH, CHAIN_ORDER
from app.ml.dataset import MixingAIDataset

@pytest.fixture
def temp_ml_dataset():
    """Создает временное окружение с нарезанными файлами и сгенерированным мини-датасетом."""
    temp_dir = Path(tempfile.gettempdir()) / "test_mixing_ai_ml_dataset"
    dry_dir = temp_dir / "raw" / "dry_vocals"
    output_dir = temp_dir / "processed" / "dataset_v1"
    
    dry_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Синтезируем вокал (1 секунда синуса)
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y_synth = 0.3 * np.sin(2 * np.pi * 440 * t)
    
    vctk_dir = dry_dir / "vctk"
    vctk_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем 2 сухих файла
    sf.write(vctk_dir / "vctk_p225_001.wav", y_synth, sr)
    sf.write(vctk_dir / "vctk_p225_002.wav", y_synth, sr)
    
    # Запускаем генератор датасета на 3 сэмпла
    generator = DatasetGenerator(dry_dir=dry_dir, output_dir=output_dir, sr=sr)
    generator.generate(max_samples=3, sc_ratio=0.0)
    
    yield output_dir
    
    # Очистка
    if temp_dir.exists():
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_ml_dataset_loading(temp_ml_dataset):
    # Инициализация датасета PyTorch
    dataset = MixingAIDataset(temp_ml_dataset, max_len=700)
    
    assert len(dataset) == 3
    
    # Получение первого элемента
    x, y_chain, y_params, param_mask = dataset[0]
    
    # Проверка типов возвращаемых данных
    assert isinstance(x, torch.Tensor)
    assert isinstance(y_chain, torch.Tensor)
    assert isinstance(y_params, torch.Tensor)
    assert isinstance(param_mask, torch.Tensor)
    
    # Проверка размерностей тензоров
    assert x.shape == (1, 128, 700)
    assert y_chain.shape == (len(CHAIN_ORDER),)
    assert y_params.shape == (PARAMETER_VECTOR_LENGTH,)
    assert param_mask.shape == (PARAMETER_VECTOR_LENGTH,)
    
    # Проверка маскирования
    # Маска должна содержать 1.0 только для параметров активных плагинов
    # Убеждаемся, что маска бинарна
    for val in param_mask:
        assert val.item() in (0.0, 1.0)
