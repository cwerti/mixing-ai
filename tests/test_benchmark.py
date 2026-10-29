import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.dataset.dataset_schema import CHAIN_ORDER, PARAM_LAYOUT, PLUGIN_ON_PROB, ChainConfig, PluginConfig
from app.eval.benchmark import ChainBenchmark, make_case, renders_to_fraction, renders_to_level, summarize
from app.mixing.chain_optimizer import physical_to_normalized
from synthetic_audio import synthetic_vocal

SR = 22050


def _reverb_chain() -> ChainConfig:
    return ChainConfig(plugins=[
        PluginConfig(name="reverb", params={"room_size": 0.6, "damping": 0.5,
                                            "wet_level": physical_to_normalized("reverb", "wet_level", 0.2),
                                            "dry_level": 0.8}),
    ])


def test_renders_to_fraction():
    """Считается первый момент, когда пройдено 90% улучшения целевой функции."""
    history = [(1, 10.0), (9, 6.0), (17, 2.5), (25, 2.0)]
    assert renders_to_fraction(history) == 17
    assert renders_to_fraction([(1, 3.0), (9, 3.0)]) == 1
    assert renders_to_fraction([]) is None


def test_run_case_cross_performance():
    """Пара из разных «голосов»: oracle и optimizer ближе к референсу, чем исходник; режимы без модели пропущены."""
    case = make_case("pair", synthetic_vocal(SR, seconds=8.0, seed=1), synthetic_vocal(SR, seconds=8.0, seed=2),
                     _reverb_chain(), SR)
    bench = ChainBenchmark(sr=SR, prior=None, with_pitch=False, log=lambda _: None)

    rows = bench.run_case(case, budget_s=5.0, excerpt_s=8.0)
    distance = {row["mode"]: row["distance"] for row in rows}

    assert set(distance) == {"source", "oracle", "optimizer"}
    assert distance["oracle"] < distance["source"]
    assert distance["optimizer"] < distance["source"]
    optimizer_row = next(row for row in rows if row["mode"] == "optimizer")
    assert optimizer_row["renders"] >= optimizer_row["renders_to_90"] >= 1

    true_distance = {row["mode"]: row["distance_true"] for row in rows}
    assert true_distance["oracle"] == 0.0
    assert true_distance["optimizer"] < true_distance["source"]

    summary = summarize(rows)
    assert "| optimizer | 1 |" in summary
    assert "| oracle | 1 | " in summary and "| 0.00 |" in summary


def test_oracle_rendered_with_mode_gate():
    """Истинная цепочка рендерится с тем же гейтом, что и режимы, иначе distance_true меряет гейт, а не цепочку."""
    case = make_case("pair", synthetic_vocal(SR, seconds=6.0, seed=1), synthetic_vocal(SR, seconds=6.0, seed=2),
                     _reverb_chain(), SR)
    bench = ChainBenchmark(sr=SR, prior=None, with_pitch=False, log=lambda _: None)
    gates = []
    render = bench.render
    bench.render = lambda c, chain, gate: gates.append(gate) or render(c, chain, gate)

    bench.run_case(case, modes=("optimizer",), budget_s=2.0, excerpt_s=6.0)

    assert len(gates) == 2 and gates[0] is not None and gates[0] == gates[1]


def test_model_mode_uses_plugin_on_threshold():
    """Режим model включает плагин только выше PLUGIN_ON_PROB: чуть ниже порога плагина в цепочке нет."""
    case = make_case("pair", synthetic_vocal(SR, seconds=4.0, seed=1), synthetic_vocal(SR, seconds=4.0, seed=2),
                     _reverb_chain(), SR)
    probs = {name: 0.0 for name in CHAIN_ORDER}
    probs.update(reverb=PLUGIN_ON_PROB + 0.01, deesser=PLUGIN_ON_PROB - 0.01)
    params = {name: {p: 0.5 for p in PARAM_LAYOUT[name]} for name in ("reverb", "deesser")}
    bench = ChainBenchmark(sr=SR, prior=lambda src, ref: (probs, params, None), with_pitch=False,
                           log=lambda _: None)

    row = bench.run_case(case, modes=("model",), with_baselines=False)[0]

    assert row["plugins"] == ["eq", "reverb"]


def test_summarize_head_to_head():
    """Сравнение с моделью и без считается по парам (пара, seed) с допуском 0.05."""
    rows = [
        {"case": "a", "seed": 0, "mode": "source", "distance": 4.0, "renders_to_90": None, "seconds": 0},
        {"case": "a", "seed": 0, "mode": "optimizer", "distance": 2.0, "renders_to_90": 40, "seconds": 60},
        {"case": "a", "seed": 0, "mode": "model+optimizer", "distance": 1.5, "renders_to_90": 20, "seconds": 60},
        {"case": "a", "seed": 1, "mode": "optimizer", "distance": 2.2, "renders_to_90": 50, "seconds": 60},
        {"case": "a", "seed": 1, "mode": "model+optimizer", "distance": 2.21, "renders_to_90": 30, "seconds": 60},
    ]
    summary = summarize(rows)
    assert "лучше 1, хуже 0, вничью (±0.05) 1 из 2" in summary
    assert "| optimizer | 2 | 2.10 |" in summary


def test_summarize_speed_to_optimizer_level():
    """Ускорение считается по рендерам до итогового уровня optimizer на той же паре."""
    rows = [
        {"case": "a", "seed": 0, "mode": "optimizer", "distance": 2.0, "renders_to_90": 40, "seconds": 60,
         "history": [(1, 4.0), (100, 2.5), (200, 2.0)]},
        {"case": "a", "seed": 0, "mode": "model+optimizer", "distance": 2.0, "renders_to_90": 20, "seconds": 60,
         "history": [(2, 2.6), (50, 2.01), (200, 1.9)]},
    ]
    assert renders_to_level(rows[1]["history"], 2.02) == 50
    assert renders_to_level(rows[1]["history"], 1.0) is None
    assert "дошел в 1 из 1 за 50 рендеров, ускорение 4.0x" in summarize(rows)
