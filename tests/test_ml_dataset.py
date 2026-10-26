import os
import tempfile
import numpy as np
import pytest
import soundfile as sf
import torch
from pathlib import Path

from app.dataset.dataset_generator import DatasetGenerator
from app.dataset.dataset_schema import PARAMETER_VECTOR_LENGTH, CHAIN_ORDER, NUM_DSP_DESCRIPTORS
from app.ml.dataset import MixingAIDataset


@pytest.fixture
def temp_ml_dataset():
    """Создает временное окружение с нарезанными файлами и сгенерированным мини-датасетом."""
    temp_dir = Path(tempfile.gettempdir()) / "test_mixing_ai_ml_dataset"
    dry_dir = temp_dir / "raw" / "dry_vocals"
    output_dir = temp_dir / "processed" / "dataset_v1"

    dry_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y_synth = 0.3 * np.sin(2 * np.pi * 440 * t)

    vctk_dir = dry_dir / "vctk"
    vctk_dir.mkdir(parents=True, exist_ok=True)

    sf.write(vctk_dir / "vctk_p225_001.wav", y_synth, sr)
    sf.write(vctk_dir / "vctk_p225_002.wav", y_synth, sr)

    generator = DatasetGenerator(dry_dir=dry_dir, output_dir=output_dir, sr=sr)
    generator.generate(max_samples=3, sc_ratio=0.0)

    yield output_dir

    if temp_dir.exists():
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_ml_dataset_loading(temp_ml_dataset):
    dataset = MixingAIDataset(temp_ml_dataset, max_len=520, split="all", augment=False)

    assert len(dataset) == 3

    x_spec, x_desc, y_chain, y_params, param_mask = dataset[0]

    assert isinstance(x_spec, torch.Tensor)
    assert isinstance(x_desc, torch.Tensor)
    assert isinstance(y_chain, torch.Tensor)
    assert isinstance(y_params, torch.Tensor)
    assert isinstance(param_mask, torch.Tensor)

    # 6 каналов: [src_mid, src_side, ref_mid, ref_side, delta_mid, delta_side]
    assert x_spec.shape == (6, 128, 520)
    # 54 дескриптора: [src(18), ref(18), delta(18)]
    assert x_desc.shape == (NUM_DSP_DESCRIPTORS * 3,)
    assert y_chain.shape == (len(CHAIN_ORDER),)
    assert y_params.shape == (PARAMETER_VECTOR_LENGTH,)
    assert param_mask.shape == (PARAMETER_VECTOR_LENGTH,)

    for val in param_mask:
        assert val.item() in (0.0, 1.0)
