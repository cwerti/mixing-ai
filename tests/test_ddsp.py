import numpy as np
import torch

from app.audio.dsp_engine import DSPEngine
from app.dataset.dataset_schema import ChainConfig, PluginConfig
from app.ml.ddsp import apply_eq

SR = 44100


def _eq_params(rng: np.random.Generator) -> dict[str, float]:
    params = {}
    for i in range(1, 8):
        params[f"band{i}_freq_hz"] = float(rng.uniform(0.0, 1.0))
        params[f"band{i}_gain_db"] = float(rng.uniform(0.0, 1.0))
    return params


def test_eq_matches_dsp_engine():
    """Дифференцируемый EQ звучит как EQ DSPEngine (pedalboard) — до сотых долей уровня сигнала."""
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((2, SR)) * 0.1).astype(np.float32)
    for _ in range(3):
        params = _eq_params(rng)
        ref = DSPEngine().apply_chain(x, SR, ChainConfig(plugins=[PluginConfig("eq", params)]),
                                      final_limiter=False)
        y = apply_eq(torch.from_numpy(x)[None], {k: torch.tensor([v]) for k, v in params.items()}, SR)[0].numpy()
        err = np.sqrt(np.mean((y - ref) ** 2)) / np.sqrt(np.mean(ref ** 2))
        assert err < 0.01, (params, err)


def test_eq_gradient_reaches_gains():
    """Градиент ошибки по звуку доходит до усилений полос."""
    x = torch.randn(1, 2, SR // 2) * 0.1
    params = {f"band{i}_{p}": torch.tensor([0.5], requires_grad=True) for i in range(1, 8)
              for p in ("freq_hz", "gain_db")}
    apply_eq(x, params, SR).pow(2).mean().backward()
    assert all(params[f"band{i}_gain_db"].grad.abs().item() > 0 for i in range(1, 8))


def _render_one(name: str, params: dict[str, float], x: np.ndarray) -> np.ndarray:
    return DSPEngine().apply_chain(x, SR, ChainConfig(plugins=[PluginConfig(name, params)]), final_limiter=False)


def test_compressor_matches_dsp_engine():
    """Дифференцируемый компрессор звучит как компрессор DSPEngine (с автоматическим makeup)."""
    from app.ml.ddsp import apply_compressor
    rng = np.random.default_rng(1)
    t = np.arange(SR) / SR
    x = (rng.standard_normal((2, SR)) * (0.02 + 0.3 * ((t > 0.3) & (t < 0.6)))).astype(np.float32)
    for _ in range(3):
        params = {k: float(rng.uniform(0.0, 1.0)) for k in ("threshold_db", "ratio", "attack_ms", "release_ms")}
        ref = _render_one("compressor", params, x)
        y = apply_compressor(torch.from_numpy(x)[None].double(),
                             {k: torch.tensor([v], dtype=torch.float64) for k, v in params.items()}, SR)[0].numpy()
        err = np.sqrt(np.mean((y - ref) ** 2)) / np.sqrt(np.mean(ref ** 2))
        assert err < 0.01, (params, err)


def test_compressor_gradients_match_finite_differences():
    """Аналитический обратный проход огибающей совпадает с численной производной по всем ручкам."""
    from app.ml.ddsp import apply_compressor
    rng = np.random.default_rng(2)
    t = np.arange(SR // 4) / SR
    x = torch.from_numpy(rng.standard_normal((1, 1, SR // 4)) * (0.05 + 0.4 * ((t > 0.08) & (t < 0.16))))
    base = {"threshold_db": 0.3, "ratio": 0.5, "attack_ms": 0.2, "release_ms": 0.4}
    params = {k: torch.tensor([v], dtype=torch.float64, requires_grad=True) for k, v in base.items()}
    loss = lambda p: apply_compressor(x, p, SR).pow(2).mean()  # noqa: E731
    loss(params).backward()
    for k in base:
        eps = 1e-4
        hi = {n: torch.tensor([v + (eps if n == k else 0.0)], dtype=torch.float64) for n, v in base.items()}
        lo = {n: torch.tensor([v - (eps if n == k else 0.0)], dtype=torch.float64) for n, v in base.items()}
        numeric = (loss(hi) - loss(lo)).item() / (2 * eps)
        assert abs(params[k].grad.item() - numeric) < 0.05 * abs(numeric) + 1e-9, (k, params[k].grad.item(), numeric)


def _check_matches(name: str, fn, keys: tuple[str, ...], seed: int, tol: float = 0.01) -> None:
    rng = np.random.default_rng(seed)
    t = np.arange(SR) / SR
    x = (rng.standard_normal((2, SR)) * (0.02 + 0.3 * ((t > 0.3) & (t < 0.6)))).astype(np.float32)
    for _ in range(3):
        params = {k: float(rng.uniform(0.0, 1.0)) for k in keys}
        ref = _render_one(name, params, x)
        y = fn(torch.from_numpy(x)[None].double(),
               {k: torch.tensor([v], dtype=torch.float64) for k, v in params.items()}, SR)[0].numpy()
        err = np.sqrt(np.mean((y - ref) ** 2)) / np.sqrt(np.mean(ref ** 2))
        assert err < tol, (name, params, err)


def test_deesser_matches_dsp_engine():
    from app.ml.ddsp import apply_deesser
    _check_matches("deesser", apply_deesser, ("threshold_db", "ratio"), seed=3)


def test_distortion_matches_dsp_engine():
    from app.ml.ddsp import apply_distortion
    _check_matches("distortion", apply_distortion, ("drive_db",), seed=4)


def test_multiband_matches_dsp_engine():
    from app.ml.ddsp import apply_multiband
    _check_matches("multiband_compressor", apply_multiband, ("depth",), seed=5)


def test_exciter_matches_dsp_engine():
    from app.ml.ddsp import apply_exciter
    _check_matches("exciter", apply_exciter, ("mix", "cutoff_hz"), seed=6)


def test_tone_chain_matches_dsp_engine_and_soft_gate_interpolates():
    """Цепочка EQ → компрессор → де-эссер → дисторшн совпадает с DSPEngine; on=0 выключает эффект."""
    from app.ml.ddsp import render_tone_chain
    rng = np.random.default_rng(7)
    t = np.arange(SR) / SR
    x = (rng.standard_normal((2, SR)) * (0.02 + 0.3 * ((t > 0.3) & (t < 0.6)))).astype(np.float32)
    plugins = {"eq": _eq_params(rng), "compressor": {"threshold_db": 0.3, "ratio": 0.6, "attack_ms": 0.2,
                                                     "release_ms": 0.4},
               "deesser": {"threshold_db": 0.3, "ratio": 0.7}, "distortion": {"drive_db": 0.4}}
    ref = DSPEngine().apply_chain(x, SR, ChainConfig(plugins=[PluginConfig(n, p) for n, p in plugins.items()]),
                                  final_limiter=False)
    tensors = {n: {k: torch.tensor([v], dtype=torch.float64) for k, v in p.items()} for n, p in plugins.items()}
    xt = torch.from_numpy(x)[None].double()
    y = render_tone_chain(xt, tensors, SR)[0].numpy()
    assert np.sqrt(np.mean((y - ref) ** 2)) / np.sqrt(np.mean(ref ** 2)) < 0.02

    off = render_tone_chain(xt, tensors, SR, on={"compressor": torch.zeros(1), "deesser": torch.zeros(1),
                                                 "distortion": torch.zeros(1)})
    from app.ml.ddsp import apply_eq
    assert torch.allclose(off, apply_eq(xt, tensors["eq"], SR))


def test_deesser_cuts_sibilance_not_vowels_or_air():
    """Де-эссер ослабляет всплески в полосе сибилянтов, не трогая гласные (<4 кГц) и воздух (>12 кГц)."""
    from scipy.signal import butter, sosfilt

    from app.dataset.dataset_schema import DATAGEN_PLUGINS

    rng = np.random.default_rng(5)
    t = np.arange(SR) / SR
    vowel = 0.2 * sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((220, 440, 660, 880, 1760), 1))
    sib = sosfilt(butter(4, (5500, 9000), "bandpass", fs=SR, output="sos"), rng.standard_normal(SR)) * 0.3
    sib[: SR // 2] = 0.0  # «с» только во второй половине
    air = sosfilt(butter(4, 13000, "highpass", fs=SR, output="sos"), rng.standard_normal(SR)) * 0.01
    x = np.vstack([vowel + sib + air] * 2).astype(np.float32)
    lo, hi = DATAGEN_PLUGINS["deesser"]["params"]["threshold_db"]
    chain = ChainConfig(plugins=[PluginConfig("deesser", {"threshold_db": (-28.0 - lo) / (hi - lo), "ratio": 1.0})])
    y = DSPEngine().apply_chain(x, SR, chain, final_limiter=False)

    def band_db(s, lo_hz, hi_hz, part):
        seg = s[0, SR // 2:] if part else s[0, : SR // 2]
        spec = np.abs(np.fft.rfft(seg)) ** 2
        f = np.fft.rfftfreq(len(seg), 1 / SR)
        return 10 * np.log10(spec[(f >= lo_hz) & (f < hi_hz)].sum())

    assert band_db(x, 5500, 9000, 1) - band_db(y, 5500, 9000, 1) > 6.0  # сибилянты ослаблены
    assert abs(band_db(x, 100, 3000, 0) - band_db(y, 100, 3000, 0)) < 0.2  # гласная без «с» не тронута
    assert abs(band_db(x, 13000, 20000, 1) - band_db(y, 13000, 20000, 1)) < 1.0  # воздух почти не тронут
