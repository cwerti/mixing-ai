import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.audio.dsp_engine import DSPEngine
from app.dataset.dataset_schema import ChainConfig, PluginConfig
from app.mixing.chain_optimizer import (
    CHAIN_INPUT_LUFS, ChainOptimizer, normalize_lufs, physical_to_normalized, select_excerpt, to_stereo,
)
from synthetic_audio import synthetic_vocal

SR = 22050


def _render_reference(dry: np.ndarray) -> np.ndarray:
    """Референс: тот же голос через известную цепочку (компрессор + реверб) на уровне входа цепочки."""
    chain = ChainConfig(plugins=[
        PluginConfig(name="compressor", params={"threshold_db": 0.2, "ratio": 0.6, "attack_ms": 0.1,
                                                "release_ms": 0.3}),
        PluginConfig(name="reverb", params={"room_size": 0.6, "damping": 0.5,
                                            "wet_level": physical_to_normalized("reverb", "wet_level", 0.2),
                                            "dry_level": 0.8}),
    ])
    stereo = normalize_lufs(to_stereo(dry), SR, CHAIN_INPUT_LUFS)
    return DSPEngine().apply_chain(stereo, SR, chain)


def test_optimizer_finds_reverb_and_reduces_distance():
    """Оптимизатор включает реверб, подбирает слышимый wet и минимум вдвое сокращает расстояние до референса."""
    dry = synthetic_vocal(SR, seconds=8.0, seed=3)
    ref = _render_reference(dry)

    result = ChainOptimizer(sr=SR, log=lambda _: None).optimize(dry, ref, budget_s=15.0, excerpt_s=8.0)

    names = [plugin.name for plugin in result.chain.plugins]
    assert names[0] == "eq"
    assert "reverb" in names
    assert "pitch_corrector" not in names  # ноты синтетики уже на сетке, тюн не нужен
    assert result.distance_final < result.distance_source / 2

    reverb = next(plugin for plugin in result.chain.plugins if plugin.name == "reverb")
    assert 0.05 < reverb.params["wet_level"] < 0.8


def test_optimizer_keeps_dry_reference_dry():
    """Если референс сухой, пространственные эффекты не добавляются."""
    dry = synthetic_vocal(SR, seconds=8.0, seed=4)
    ref = normalize_lufs(to_stereo(dry), SR, CHAIN_INPUT_LUFS)

    result = ChainOptimizer(sr=SR, log=lambda _: None).optimize(dry, ref, budget_s=5.0, excerpt_s=8.0)

    names = {plugin.name for plugin in result.chain.plugins}
    assert not names & {"reverb", "delay", "chorus", "stereo_enhancer"}


def test_select_excerpt_picks_loudest_part():
    """Фрагмент выбирается по максимальной энергии."""
    quiet = 0.01 * np.ones((2, SR * 4), dtype=np.float32)
    loud = 0.5 * np.ones((2, SR * 2), dtype=np.float32)
    y = np.concatenate([quiet, loud, quiet], axis=1)
    excerpt = select_excerpt(y, SR, seconds=2.0)
    assert excerpt.shape == (2, SR * 2)
    assert excerpt.mean() > 0.4
