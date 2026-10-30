import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.audio.dsp_engine import DSPEngine
from app.dataset.dataset_schema import PLUGIN_ON_PROB, ChainConfig, PluginConfig
from app.mixing.chain_optimizer import (
    CHAIN_INPUT_LUFS, DEFAULT_SIGMA, ChainOptimizer, _initial_params, _stagnated, normalize_lufs,
    physical_to_normalized, search_stds, select_excerpt, select_plugins, to_stereo,
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


def test_prior_start_only_for_plugins_the_model_enables():
    """Параметры модели берутся как старт только у плагинов, которые она считает активными."""
    params = {"reverb": {"room_size": 0.9, "damping": 0.1, "wet_level": 0.7, "dry_level": 0.2}}
    assert _initial_params("reverb", {"reverb": 0.9}, params)["wet_level"] == 0.7
    assert _initial_params("reverb", {"reverb": 0.2}, params)["wet_level"] == 0.25
    # граница та же, что при сборке цепочки моделью (PLUGIN_ON_PROB)
    assert _initial_params("reverb", {"reverb": PLUGIN_ON_PROB + 0.01}, params)["wet_level"] == 0.7
    assert _initial_params("reverb", {"reverb": PLUGIN_ON_PROB - 0.01}, params)["wet_level"] == 0.25


def test_search_stds_narrow_around_confident_prior():
    """Шаг поиска у параметров со стартом от модели — ее ошибка в пределах PRIOR_SIGMA_RANGE, у прочих — DEFAULT_SIGMA."""
    layout = [("reverb", "wet_level"), ("reverb", "damping"), ("delay", "mix")]
    std = {"reverb": {"wet_level": 0.01, "damping": 0.12}, "delay": {"mix": 0.1}}
    assert search_stds(layout, {"reverb"}, std) == [0.05, 0.12, DEFAULT_SIGMA]
    assert search_stds(layout, {"reverb"}, None) == [DEFAULT_SIGMA] * 3


def test_stagnated_after_patience_without_progress():
    """Остановка, только когда за patience рендеров улучшение меньше порога."""
    history = [(1, 3.0), (11, 2.0), (21, 1.995), (31, 1.99)]
    assert _stagnated(history, patience_renders=20, min_improvement=0.02)
    assert not _stagnated(history, patience_renders=30, min_improvement=0.02)
    assert not _stagnated(history[:2], patience_renders=20, min_improvement=0.02)


def test_optimizer_drops_misleading_prior_and_stops_early():
    """Заведомо неверный старт модели (сухой реверб) отбрасывается, а ранняя остановка экономит рендеры."""
    dry = synthetic_vocal(SR, seconds=8.0, seed=3)
    ref = _render_reference(dry)
    probs = {"reverb": 0.9, "compressor": 0.9}
    params = {"reverb": {"room_size": 0.0, "damping": 1.0, "wet_level": 0.0, "dry_level": 1.0},
              "compressor": {"threshold_db": 1.0, "ratio": 0.0, "attack_ms": 0.5, "release_ms": 0.5}}

    result = ChainOptimizer(sr=SR, log=lambda _: None).optimize(
        dry, ref, model_probs=probs, model_params=params, budget_s=15.0, excerpt_s=8.0, patience_renders=40)

    assert any("отброшен" in line for line in result.report)
    assert any("остановлен" in line for line in result.report) or result.history[-1][0] < 400
    assert result.distance_final < result.distance_source / 2


def test_enhancer_added_with_reverb_when_notes_are_narrow():
    """Реверб не дает ширины на нотах, поэтому при узком голосе расширитель ставится вместе с ним."""
    base = {"decay_fast_db": -20.0, "delay_peak": 0.1, "width_global": 0.0, "dr_p95_p10_db": 20.0,
            "crest_db": 15.0, "sib_p90_db": -8.0, "width_notes_db": -30.0}
    ref = dict(base, decay_fast_db=-12.0, width_global=0.3, width_notes_db=-15.0)
    assert {"reverb", "stereo_enhancer"} <= set(select_plugins(base, ref, None)[0])
    assert "stereo_enhancer" not in select_plugins(base, dict(ref, width_notes_db=-28.0), None)[0]


def test_optimizer_keeps_confident_plugins_and_unmeasured_params():
    """Плагин, в котором модель уверена, не выкидывается, а скорость и глубина хоруса остаются как у модели."""
    dry = synthetic_vocal(SR, seconds=8.0, seed=3)
    ref = _render_reference(dry)
    probs = {"chorus": 0.95, "compressor": 0.95}
    params = {"chorus": {"rate_hz": 0.1, "depth": 0.2, "feedback": 0.3, "mix": 0.3},
              "compressor": {"threshold_db": 0.2, "ratio": 0.6, "attack_ms": 0.1, "release_ms": 0.3}}

    result = ChainOptimizer(sr=SR, log=lambda _: None).optimize(
        dry, ref, model_probs=probs, model_params=params, budget_s=5.0, excerpt_s=8.0, patience_renders=40)

    plugins = {plugin.name: plugin.params for plugin in result.chain.plugins}
    assert {"chorus", "compressor"} <= set(plugins)
    assert (plugins["chorus"]["rate_hz"], plugins["chorus"]["depth"], plugins["chorus"]["feedback"]) == (0.1, 0.2, 0.3)


def test_optimizer_searches_reverb_send_params_model_does_not_predict():
    from app.mixing.chain_optimizer import OPTIMIZER_EXTRA_PARAMS, _default_params, _initial_params
    from app.dataset.dataset_schema import PARAM_LAYOUT
    extra = set(OPTIMIZER_EXTRA_PARAMS["reverb"])
    assert not extra & set(PARAM_LAYOUT["reverb"])
    assert extra <= set(_default_params("reverb"))
    model_params = {"reverb": {k: 0.5 for k in PARAM_LAYOUT["reverb"]}}
    assert extra <= set(_initial_params("reverb", {"reverb": 0.99}, model_params))


def test_stagnated_at_exactly_patience_renders_does_not_crash():
    from app.mixing.chain_optimizer import _stagnated
    assert _stagnated([(1, 5.0), (10, 4.0)], 10, 0.01) is False
    assert _stagnated([(1, 5.0), (11, 4.995)], 10, 0.01) is True


def test_start_params_omit_superseded_room_size():
    from app.mixing.chain_optimizer import _default_params, _initial_params
    from app.dataset.dataset_schema import PARAM_LAYOUT
    model_params = {"reverb": {k: 0.5 for k in PARAM_LAYOUT["reverb"]}}
    assert "room_size" not in _default_params("reverb")
    assert "room_size" not in _initial_params("reverb", {"reverb": 0.99}, model_params)
