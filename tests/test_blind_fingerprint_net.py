import numpy as np
import torch

from app.audio.dsp_engine import send_delay
from app.eval.blind_fingerprint_net import (FRAMES, KEYS, N_ACF, N_ACF_LAG_SIM, N_LAGS, N_MELS, MIN_LAG, SR,
                                            BlindFingerprintNet, predict, target_vector, vector_to_dict,
                                            wet_features)


def _pulses(seconds: float = 8.0) -> np.ndarray:
    """Короткие «слоги» разной высоты раз в 0.9 с — сухой вход без собственной периодичности около 300 мс."""
    y = np.zeros(int(seconds * SR), dtype=np.float32)
    t = np.arange(int(0.12 * SR)) / SR
    for k, start in enumerate(np.arange(0.2, seconds - 0.5, 0.9)):
        i = int(start * SR)
        f0 = 220.0 * 2.0 ** ((k * 5 % 12) / 12.0)
        y[i:i + len(t)] += 0.3 * np.sin(2 * np.pi * f0 * t) * np.hanning(len(t))
    return y


def test_features_shape_and_echo_hint():
    dry = _pulses()
    wet = np.stack([dry, dry]) + 0.6 * send_delay(np.stack([dry, dry]), SR, 0.30, 0.5)
    mel, acf = wet_features(wet, SR, with_lag_similarity=True)
    assert mel.shape == (4, N_MELS, FRAMES) and acf.shape == (N_ACF_LAG_SIM, N_LAGS)
    assert mel.max() <= 0.0 and mel.min() >= -80.0
    assert np.all(acf[:, :MIN_LAG] == 0.0)
    # Пик автокорреляции потока — на лаге повтора (300 мс ≈ 26 кадров по 11.6 мс)
    lag_s = (MIN_LAG + int(np.argmax(acf[0, MIN_LAG:int(0.6 * SR / 512)]))) * 512 / SR
    assert abs(lag_s - 0.30) < 0.03
    lag_sim = (MIN_LAG + int(np.argmax(acf[2, MIN_LAG:]))) * 512 / SR
    assert abs(lag_sim - 0.30) < 0.03


def test_lag_interpolation():
    from app.eval.blind_fingerprint_net import lag_from_logits
    lags = np.arange(40, dtype=float)
    assert abs(lag_from_logits(-(lags - 27.4) ** 2) - 27.4) < 1e-6


def test_target_roundtrip_and_mask():
    fp = {"rt60_1000": 2.0, "tail_db_8000": -12.0, "side_mid_db": -3.0, "rt60_s": 1.5, "drr_db": float("nan")}
    vec, mask = target_vector(fp)
    assert vec.shape == (len(KEYS),) and mask.sum() == 4
    back = vector_to_dict(vec)
    assert abs(back["rt60_1000"] - 2.0) < 1e-4 and abs(back["tail_db_8000"] + 12.0) < 1e-4
    assert abs(back["rt60_s"] - 1.5) < 1e-4


def test_model_forward_and_predict_are_deterministic():
    torch.manual_seed(0)
    model = BlindFingerprintNet().eval()
    reg, period, echo = model(torch.zeros(2, 4, N_MELS, FRAMES) - 40.0, torch.zeros(2, N_ACF, N_LAGS))
    assert reg.shape == (2, len(KEYS)) and period.shape == (2, 2, N_LAGS) and echo.shape == (2,)
    wet = np.stack([_pulses(), _pulses()])
    a, b = predict(model, wet, SR), predict(model, wet, SR)
    assert a == b and 0.0 <= a["echo_prob"] <= 1.0 and set(KEYS) <= set(a)
