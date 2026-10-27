import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.dataset.dataset_schema import ChainConfig, PluginConfig
from app.eval.benchmark import ChainBenchmark, make_case, renders_to_fraction, summarize
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

    summary = summarize(rows)
    assert "| optimizer | 1 |" in summary


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
