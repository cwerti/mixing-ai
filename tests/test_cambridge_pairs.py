import numpy as np

from scripts.cambridge_pairs import SR, global_lag, sections


def _phrases(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    y = np.zeros(n, dtype=np.float32)
    t = 0.0
    while t < seconds - 1.0:
        dur = rng.uniform(0.3, 1.5)
        a, b = int(t * SR), int(min(seconds, t + dur) * SR)
        y[a:b] = rng.standard_normal(b - a) * rng.uniform(0.1, 0.5)
        t += dur + rng.uniform(0.2, 1.0)
    return np.vstack([y, y])


def test_global_lag_finds_large_offset():
    dry = _phrases(40.0)
    shift = int(3.27 * SR)
    wet = np.pad(dry, ((0, 0), (shift, 0)))[:, : dry.shape[1]]
    lag, peak = global_lag(dry, wet)
    assert abs(lag - shift) <= 441
    assert peak > 0.5


def test_sections_are_disjoint_and_inside():
    dry = _phrases(90.0, seed=1)
    a, b = sections(dry, seconds=30.0)
    n = int(30.0 * SR)
    assert b >= a + n
    assert b + n <= dry.shape[1]
    assert sections(dry[:, : int(50 * SR)], seconds=30.0) is None
