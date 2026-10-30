import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.eval.style_chains import (
    FLAG_PLUGINS, archetype_chain, chain_flags, content_pool, is_test_song, sample_realistic_chain,
)
from app.eval.style_embedding import (
    EMBEDDING_SCALE, N_MELS, WINDOW_FRAMES, StyleEncoder, StyleNet, combined_distance, embedding_distance,
    style_features, windows,
)
from app.mixing.chain_optimizer import CHAIN_INPUT_LUFS, ChainOptimizer, normalize_lufs, to_stereo
from synthetic_audio import synthetic_vocal

SR = 44100


@pytest.fixture(scope="module")
def encoder() -> StyleEncoder:
    """Кодировщик со случайными весами: API не зависит от обученного чекпоинта."""
    torch.manual_seed(0)
    return StyleEncoder(net=StyleNet(width=8, embed_dim=16))


def test_features_shape_mono_and_stereo():
    y = synthetic_vocal(SR, seconds=3.0, seed=1)
    mono = style_features(y, SR)
    stereo = style_features(np.vstack([y, 0.5 * y]), SR)
    assert mono.shape == stereo.shape == (2, N_MELS, len(y) // 512 + 1)
    assert mono.dtype == np.float32
    # У моно Side пустой (пол), у стерео с разными каналами — нет
    assert mono[1].max() < stereo[1].max()


def test_windows_pad_short_and_cover_long():
    short = np.zeros((2, N_MELS, 50), dtype=np.float32)
    assert windows(short).shape == (1, 2, N_MELS, WINDOW_FRAMES)
    long = np.zeros((2, N_MELS, 1000), dtype=np.float32)
    w = windows(long)
    assert w.shape[1:] == (2, N_MELS, WINDOW_FRAMES) and len(w) >= 1000 // WINDOW_FRAMES


def test_embedding_shape_norm_determinism(encoder):
    y = synthetic_vocal(SR, seconds=4.0, seed=2)
    a, b = encoder.embed(y, SR), encoder.embed(y, SR)
    assert a.shape == (16,)
    assert np.isclose(np.linalg.norm(a), 1.0, atol=1e-5)
    np.testing.assert_array_equal(a, b)


def test_distance_symmetric_and_identity(encoder):
    y1 = synthetic_vocal(SR, seconds=4.0, seed=3)
    y2 = synthetic_vocal(SR, seconds=4.0, seed=4)
    assert embedding_distance(y1, y1, SR, encoder) == pytest.approx(0.0, abs=1e-6)
    d12, d21 = embedding_distance(y1, y2, SR, encoder), embedding_distance(y2, y1, SR, encoder)
    assert d12 == pytest.approx(d21, abs=1e-6)
    assert 0.0 <= d12 <= 2.0


def test_embedding_ignores_loudness(encoder):
    """Признаки считаются после выравнивания громкости: уровень входа не меняет эмбеддинг."""
    y = synthetic_vocal(SR, seconds=4.0, seed=5)
    assert embedding_distance(y, 0.3 * y, SR, encoder) == pytest.approx(0.0, abs=1e-4)


def test_combined_distance_units():
    assert combined_distance(0.0, 0.0) == 0.0
    assert combined_distance(1.0, 0.0) < combined_distance(1.0, 0.1)
    assert combined_distance(0.0, 0.1) == pytest.approx(0.5 * EMBEDDING_SCALE * 0.1)


def test_realistic_chains_and_flags():
    import random
    rng = random.Random(0)
    chains = [sample_realistic_chain(rng) for _ in range(200)]
    assert any(not c.plugins for c in chains)  # бывает «ничего не делать»
    flags = np.stack([chain_flags(c) for c in chains])
    assert flags.shape == (200, len(FLAG_PLUGINS)) and 0.5 < flags[:, FLAG_PLUGINS.index("eq")].mean() < 1.0
    for chain in chains + [archetype_chain("che"), archetype_chain("ninoo", rng)]:
        for p in chain.plugins:
            assert all(0.0 <= v <= 1.0 for v in p.params.values()), (p.name, p.params)


def test_test_songs_excluded_from_pool():
    assert is_test_song(Path("oxeye_Sorry_13_LeadVox_NoPitchProcessing.wav"))
    train, val = content_pool()
    if not train:
        pytest.skip("Пул сухих вокалов не скачан")
    assert not any(is_test_song(f) for f in train + val)
    from app.dataset.dataset_generator import DatasetGenerator
    assert {DatasetGenerator.song_key(f) for f in train}.isdisjoint(DatasetGenerator.song_key(f) for f in val)


def test_optimizer_metric_switch(encoder):
    dry = synthetic_vocal(22050, seconds=6.0, seed=6)
    ref = normalize_lufs(to_stereo(dry), 22050, CHAIN_INPUT_LUFS)
    with pytest.raises(ValueError):
        ChainOptimizer(sr=22050, metric="bogus")
    assert ChainOptimizer(sr=22050).metric == "style"  # по умолчанию прежнее поведение
    for metric in ("embedding", "combined"):
        opt = ChainOptimizer(sr=22050, log=lambda _: None, metric=metric, encoder=encoder)
        result = opt.optimize(dry, ref, budget_s=2.0, excerpt_s=6.0, allow_tune=False)
        assert np.isfinite(result.distance_final) and result.distance_final >= 0.0
        assert result.chain.plugins[0].name == "eq"
