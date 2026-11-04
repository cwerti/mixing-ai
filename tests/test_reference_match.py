from app.eval.reference_match import RELIABLE, description_distance, preselect


def _desc(rt60: float, drr: float, width: float) -> dict:
    d = {k: rt60 for k in RELIABLE if k.startswith("rt60_")}
    return d | {"predelay_ms": 20.0, "drr_db": drr, "side_mid_db": width, "iacc_late": 0.5}


def test_distance_is_zero_for_identical_and_grows_with_difference():
    a = _desc(2.0, 10.0, -6.0)
    assert description_distance(a, a) == 0.0
    assert description_distance(a, _desc(2.0, 4.0, -6.0)) > description_distance(a, _desc(2.0, 8.0, -6.0)) > 0.0


def test_tail_length_does_not_drown_dry_wet_balance():
    # Сдвиг на одну единицу масштаба во всех семи полосах RT60 весит как два признака (√2 от сдвига DRR на
    # единицу), а не семь (√7): ln(1.35) ≈ 0.3 — единица масштаба RT60, 6 дБ — единица DRR
    target = _desc(2.0, 10.0, -6.0)
    longer_tail = _desc(2.0 * 1.35, 10.0, -6.0)
    wetter = _desc(2.0, 10.0 - 6.0, -6.0)
    ratio = description_distance(target, longer_tail) / description_distance(target, wetter)
    assert 1.3 < ratio < 1.5


def test_missing_feature_is_ignored():
    a = _desc(2.0, 10.0, -6.0)
    b = dict(a)
    del b["predelay_ms"]
    assert description_distance(a, b) == 0.0


def test_preselect_orders_by_distance():
    target = _desc(2.0, 10.0, -6.0)
    pool = [_desc(5.0, 0.0, -20.0), _desc(2.0, 9.0, -6.0), _desc(3.0, 10.0, -6.0)]
    assert preselect(pool, target, 2) == [1, 2]


def test_blind_estimate_caps_windows_on_long_signal(monkeypatch):
    import numpy as np

    import app.eval.reference_match as rm
    calls = []
    monkeypatch.setattr(rm, "predict", lambda model, wet, sr: calls.append(wet.shape[1]) or {"drr_db": 1.0})
    long = np.zeros((2, int(90 * rm.SR)), dtype=np.float32)
    short = np.zeros((2, int(30 * rm.SR)), dtype=np.float32)
    rm.blind_estimate(None, long)
    assert 2 <= len(calls) <= rm.MAX_WINDOWS
    calls.clear()
    rm.blind_estimate(None, short)
    assert len(calls) == 23  # 30 с при шаге 1 с — как до ограничения
