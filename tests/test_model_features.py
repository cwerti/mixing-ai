import random
import sys
from pathlib import Path

import numpy as np
import pyloudnorm as pyln

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.audio.audio_processor import AudioProcessor
from app.dataset.dataset_generator import DatasetGenerator
from app.ml.features import MODEL_LUFS, model_inputs, prepare_model_audio, window_seconds
from synthetic_audio import synthetic_vocal

SR = 22050


def test_prepare_model_audio_takes_loud_stereo_window():
    """Моно-вход становится стерео, длина равна окну модели, громкость -23 LUFS, берется громкая часть."""
    quiet = 0.02 * synthetic_vocal(SR, seconds=10.0, seed=1)
    loud = synthetic_vocal(SR, seconds=8.0, seed=2)
    y = np.concatenate([quiet, loud])

    out = prepare_model_audio(y, SR)

    assert out.shape == (2, int(window_seconds(SR) * SR))
    assert abs(pyln.Meter(SR).integrated_loudness(out.T) - MODEL_LUFS) < 0.5
    # Окно должно лежать в громкой второй половине: до нормализации ее пики в 50 раз выше
    ratio = np.abs(out).max() / np.abs(prepare_model_audio(quiet, SR)).max()
    assert 0.5 < ratio < 2.0


def test_model_inputs_keep_reference_width():
    """Ширина референса доходит до модели: Side-канал стерео-референса не пустой, у моно — тишина."""
    processor = AudioProcessor(sr=SR)
    src = synthetic_vocal(SR, seconds=8.0, seed=3)
    rng = np.random.default_rng(0)
    wide_ref = np.vstack([src, np.roll(src, 300) + 0.01 * rng.standard_normal(len(src))]).astype(np.float32)

    x_spec, x_desc = model_inputs(src, wide_ref, processor)
    x_spec_mono, _ = model_inputs(src, src, processor)

    assert x_spec.shape == (6, 128, 520) and x_desc.shape == (54,)
    assert x_spec[3].mean() > x_spec_mono[3].mean() + 0.1  # канал ref_side


def test_pick_reference_file_prefers_other_song():
    """При cross_ratio=1 референс из другой песни, при 0 — тот же файл."""
    pool = [Path(f"Song_A_{i}.wav") for i in range(3)] + [Path("Song_B_1.wav"), Path("Song_C_1.wav")]
    rng = random.Random(0)
    for _ in range(20):
        ref = DatasetGenerator._pick_reference_file(pool, pool[0], 1.0, rng)
        assert DatasetGenerator.song_key(ref) != "Song_A"
    assert DatasetGenerator._pick_reference_file(pool, pool[0], 0.0, rng) == pool[0]
    # Одна песня в пуле: берется другой файл той же песни
    assert DatasetGenerator._pick_reference_file(pool[:3], pool[0], 1.0, rng) != pool[0]
