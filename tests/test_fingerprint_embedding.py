import numpy as np
import torch

from app.audio.dsp_engine import decay_reverb, send_delay
from app.eval.effect_fingerprint import BANDS_HZ, fingerprint
from app.eval.fingerprint_embedding import (MAP_EDGES_S, FingerprintEmbedder, embed, embedding_input,
                                            impulse_response, load_embedder, response_map, save_embedder)

SR = 48000


def _reverb(x, sr):
    return x + 0.5 * decay_reverb(x.mean(axis=0), sr, 1.0, 0.8)


def test_response_map_shape_and_decay():
    h = impulse_response(_reverb, sr=SR, seconds=3.0)
    rmap = response_map(h, SR)
    assert rmap.shape == (2, len(BANDS_HZ), len(MAP_EDGES_S) - 1)
    mid_1k = rmap[0, BANDS_HZ.index(1000)]
    # Хвост 1 с: окно около 0.1 с заметно громче окна около 2 с
    t = 0.5 * (MAP_EDGES_S[:-1] + MAP_EDGES_S[1:])
    assert mid_1k[np.argmin(np.abs(t - 0.1))] > mid_1k[np.argmin(np.abs(t - 2.0))] + 40.0
    # Моно-ревер: Side на полу
    assert rmap[1].max() <= -79.0


def test_delay_and_reverb_maps_differ():
    echo = response_map(impulse_response(lambda x, sr: x + 0.5 * send_delay(x, sr, 0.3, 0.4), SR, 3.0), SR)
    rev = response_map(impulse_response(_reverb, SR, 3.0), SR)
    assert np.abs(echo - rev).mean() > 3.0


def test_embedder_deterministic_and_roundtrip(tmp_path):
    fp = fingerprint(_reverb, sr=SR, ir_seconds=2.0, sine_seconds=2.0)
    x = embedding_input(fp, response_map(impulse_response(_reverb, SR, 2.0), SR))[None]
    inputs = np.vstack([x, x + 0.1])
    torch.manual_seed(0)
    model = FingerprintEmbedder(inputs.shape[1], dim=8)
    mean, std = inputs.mean(axis=0), inputs.std(axis=0) + 1e-3
    e1, e2 = embed(model, inputs, mean, std), embed(model, inputs, mean, std)
    assert e1.shape == (2, 8) and np.array_equal(e1, e2)
    path = tmp_path / "emb.pt"
    save_embedder(path, model, mean, std, {"test": True})
    loaded, m2, s2 = load_embedder(path)
    assert np.allclose(embed(loaded, inputs, m2, s2), e1)
