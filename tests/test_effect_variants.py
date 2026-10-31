"""Доменная рандомизация: альтернативные реализации эффектов (app/audio/effect_variants.py) и генератор."""
import json
import random

import numpy as np
import pytest
import soundfile as sf

from app.audio.dsp_engine import DSPEngine
from app.audio.effect_variants import EffectVariants, IRBank, VSTPool, default_vst_pool, freeverb_rt60
from app.audio.synthetic_reverbs import estimate_rt60, fdn_ir, plate_ir, spring_ir
from app.dataset.dataset_schema import DATAGEN_PLUGINS, REVERB_SEND_PARAMS, ChainConfig, PluginConfig
from tests.synthetic_audio import synthetic_vocal

SR = 44100


@pytest.fixture(scope="module")
def vocal() -> np.ndarray:
    y = synthetic_vocal(SR, seconds=2.0, seed=1)
    y = y if y.ndim == 2 else np.vstack([y, y])
    return (0.1 * y / (np.abs(y).max() + 1e-9)).astype(np.float32)


@pytest.fixture(scope="module")
def ir_bank(tmp_path_factory) -> IRBank:
    """Маленький банк: четыре «внешние» ИХ (экспоненциальный шум) и одна синтетическая (FDN)."""
    out = tmp_path_factory.mktemp("ir_cache")
    rng = np.random.default_rng(0)
    irs, items, offset = [], [], 0
    for i, (kind, rt60) in enumerate([("external", 0.8), ("external", 2.2), ("external", 2.5), ("external", 2.9),
                                      ("synthetic", 1.5)]):
        n = int(rt60 * 1.2 * SR)
        if kind == "synthetic":
            ir = fdn_ir(SR, rt60 * 1.2, rt60, rt60 / 2, rng)
        else:
            ir = rng.standard_normal((2, n)) * np.exp(-6.91 * np.arange(n) / SR / rt60)
        ir = (2.0 * ir / np.sqrt(np.mean((ir ** 2).sum(axis=1)))).astype(np.float32)
        irs.append(ir)
        items.append({"kind": kind, "source": f"{kind}_{i}", "name": str(i), "rt60": estimate_rt60(ir, SR),
                      "offset": offset, "length": ir.shape[1]})
        offset += ir.shape[1]
    np.save(out / "irs.npy", np.concatenate(irs, axis=1).astype(np.float16))
    (out / "index.json").write_text(json.dumps({"sr": SR, "irs": items}), encoding="utf-8")
    return IRBank(out)


def mid_params(name: str) -> dict:
    return {p: 0.5 for p in DATAGEN_PLUGINS[name]["params"] if p not in REVERB_SEND_PARAMS}


def test_every_variant_renders_finite_stereo(vocal, ir_bank):
    """Каждая альтернативная реализация каждого плагина дает конечный стерео-сигнал той же длины."""
    variants = EffectVariants(sr=SR, ir_bank=ir_bank, vst_pool=None)
    rng = random.Random(0)
    checked = 0
    for name in variants.variants:
        plugin = PluginConfig(name, mid_params(name))
        for impl in variants.available(name):
            out = variants.render_plugin(vocal, plugin, impl, rng)
            assert out.shape == vocal.shape, (name, impl)
            assert np.isfinite(out).all(), (name, impl)
            assert np.abs(out).max() < 20.0, (name, impl)
            checked += 1
    assert checked >= 30


def test_reverb_ir_variants_need_bank(vocal):
    """Без кэша ИХ свёрточные варианты недоступны, остальные реверберы остаются."""
    variants = EffectVariants(sr=SR, ir_bank=None, vst_pool=None)
    options = variants.available("reverb")
    assert "rev_ir_external" not in options and "rev_ir_algorithmic" not in options
    assert {"rev_freeverb_send", "rev_decay_send"} <= set(options)


def test_ir_convolution_respects_wet_and_dry(vocal, ir_bank):
    """wet_level 0 — чистый сухой сигнал 2·dry·x; больше wet — больше энергии хвоста после конца фразы."""
    variants = EffectVariants(sr=SR, ir_bank=ir_bank, vst_pool=None)
    x = vocal.copy()
    x[:, SR:] = 0.0  # тишина во второй секунде: там только хвост
    ranges = DATAGEN_PLUGINS["reverb"]["params"]
    dry_norm = 0.5
    dry_phys = ranges["dry_level"][0] + dry_norm * (ranges["dry_level"][1] - ranges["dry_level"][0])

    def render(wet_phys: float, seed: int = 3) -> np.ndarray:
        plugin = PluginConfig("reverb", {"room_size": 0.6, "damping": 0.5, "dry_level": dry_norm,
                                         "wet_level": DSPEngine.normalize_value(wet_phys, *ranges["wet_level"])})
        phys = variants.dsp.get_physical_params(plugin)
        phys["wet_level"] = wet_phys  # ровно 0 вне диапазона схемы
        return variants._rev_ir_external(x, phys, plugin.params, random.Random(seed))

    np.testing.assert_allclose(render(0.0), 2.0 * dry_phys * x, atol=1e-6)
    tail = slice(int(1.1 * SR), None)
    low, high = render(0.1), render(0.5)
    assert np.sum(high[:, tail] ** 2) > 10.0 * np.sum(low[:, tail] ** 2) > 0.0


def test_ir_bank_prefers_matching_rt60(ir_bank):
    rng = random.Random(0)
    picks = [ir_bank.sample(rng, target_rt60=2.5, kind="external")[1]["rt60"] for _ in range(20)]
    assert min(picks) > 1.5
    ir, meta = ir_bank.sample(rng, kind="synthetic")
    assert ir.shape[0] == 2 and ir.dtype == np.float32 and meta["kind"] == "synthetic"


def test_own_only_matches_dsp_engine(vocal):
    """own_prob = 1: рандомизация выключена, результат совпадает с DSPEngine.apply_chain."""
    chain = ChainConfig([PluginConfig("eq", mid_params("eq")), PluginConfig("compressor", mid_params("compressor")),
                         PluginConfig("chorus", mid_params("chorus")), PluginConfig("reverb", mid_params("reverb")),
                         PluginConfig("delay", mid_params("delay")), PluginConfig("exciter", mid_params("exciter"))])
    variants = EffectVariants(sr=SR, own_prob=1.0, ir_bank=None, vst_pool=None)
    out, used = variants.apply_chain(vocal, chain, random.Random(0))
    expected = DSPEngine().apply_chain(vocal, SR, chain)
    assert all(u.endswith(":own") for u in used)
    # Лимитер стоит после экситера, а у DSPEngine — до него: на тихом сигнале он ничего не делает
    np.testing.assert_allclose(out, expected, atol=1e-4)


def test_randomized_chain_is_reproducible_and_varied(vocal, ir_bank):
    chain = ChainConfig([PluginConfig(n, mid_params(n)) for n in ("compressor", "distortion", "reverb", "delay")])
    variants = EffectVariants(sr=SR, own_prob=0.0, ir_bank=ir_bank, vst_pool=None)
    a, used_a = variants.apply_chain(vocal, chain, random.Random(7))
    b, used_b = variants.apply_chain(vocal, chain, random.Random(7))
    assert used_a == used_b and np.array_equal(a, b)
    impls = {tuple(variants.apply_chain(vocal, chain, random.Random(s))[1]) for s in range(8)}
    assert len(impls) > 1 and not any(u.endswith(":own") for u in used_a)


def test_vst_path_skipped_without_plugins(tmp_path, vocal):
    """Пустая опись — VST-вариантов нет; если на машине есть рабочие VST3, они рендерят без ошибок."""
    empty = VSTPool(tmp_path / "missing.json")
    assert not empty.has("reverb")
    variants = EffectVariants(sr=SR, ir_bank=None, vst_pool=empty)
    assert not any(k.startswith("vst_") for k in variants.available("reverb"))

    pool = default_vst_pool()
    categories = [c for c in pool.by_category if pool.has(c)]
    if not categories:
        pytest.skip("В data/raw/vst_inventory.json нет загружаемых VST3 (scripts/vst_inventory.py)")
    out = pool.render(vocal, SR, categories[0], {}, random.Random(0))
    assert out.shape[0] == 2 and np.isfinite(out).all()


def test_synthetic_reverbs_have_requested_decay():
    rng = np.random.default_rng(0)
    for ir, lo, hi in ((plate_ir(SR, 4.0, 0.6, 0.3), 1.0, 4.0), (fdn_ir(SR, 4.0, 2.0, 1.0, rng), 0.8, 3.0),
                       (spring_ir(SR, 3.0, rng, rt60=2.0), 0.8, 3.0)):
        assert ir.shape[0] == 2 and np.isfinite(ir).all()
        assert lo < estimate_rt60(ir, SR) < hi
    assert freeverb_rt60(0.15) < freeverb_rt60(0.7) < freeverb_rt60(0.99)


def test_generator_domain_rand_option(tmp_path):
    """Опция генератора: импл-метки в сэмплах, метки цепочек те же, диапазоны записаны в stats.json."""
    from app.dataset.dataset_generator import DatasetGenerator
    from app.dataset.dataset_schema import model_param_ranges
    t = np.arange(int(0.5 * SR)) / SR
    cambridge = tmp_path / "dry" / "cambridge"
    cambridge.mkdir(parents=True)
    for i, name in enumerate(["Song_A_01_LeadVox", "Song_B_01_LeadVox", "Song_C_01_LeadVox"]):
        sf.write(cambridge / f"{name}.wav", 0.2 * np.sin(2 * np.pi * (200 + 40 * i) * t), SR)

    plain = DatasetGenerator(tmp_path / "dry", tmp_path / "plain", sr=SR).generate(max_samples=6, seed=3)
    rand = DatasetGenerator(tmp_path / "dry", tmp_path / "rand", sr=SR).generate(
        max_samples=6, seed=3, domain_rand_own_prob=0.0)
    assert len(plain) == len(rand) == 6
    by_id = {s.id: s for s in plain}
    for s in rand:
        assert s.chain == by_id[s.id].chain and s.params_vector == by_id[s.id].params_vector
        assert len(s.ref_impls) == len(s.chain) + len(s.ref_layers)
    assert all(not s.ref_impls for s in plain)
    stats = json.loads((tmp_path / "rand" / "stats.json").read_text(encoding="utf-8"))
    assert stats["domain_rand_own_prob"] == 0.0 and stats["param_ranges"] == model_param_ranges()
    assert sum(stats["ref_impl_counts"].values()) == sum(len(s.ref_impls) for s in rand)


def test_dataset_param_ranges(tmp_path):
    """Диапазоны меток берутся из stats.json, для dataset_v4 — прежние потолки реверба, иначе — схема."""
    from app.dataset.dataset_schema import LEGACY_PARAM_RANGES, dataset_param_ranges, model_param_ranges
    (tmp_path / "dataset_v4").mkdir()
    ranges, source = dataset_param_ranges(tmp_path / "dataset_v4")
    assert ranges["reverb"]["wet_level"] == list(LEGACY_PARAM_RANGES[("reverb", "wet_level")])
    assert ranges["distortion"] == model_param_ranges()["distortion"]
    other = tmp_path / "other"
    other.mkdir()
    assert dataset_param_ranges(other)[1] == "schema"
    (other / "stats.json").write_text(json.dumps({"param_ranges": {"x": {"y": [0, 1]}}}), encoding="utf-8")
    assert dataset_param_ranges(other) == ({"x": {"y": [0, 1]}}, "stats.json")


def test_generator_reverb_wet_keeps_physical_distribution():
    """После подъема потолка wet до 0.60 физический wet генератора остается в основном 0.08–0.35."""
    from app.dataset.dataset_generator import _generate_balanced_random_config, _generate_realistic_config
    lo, hi = DATAGEN_PLUGINS["reverb"]["params"]["wet_level"]
    rng = random.Random(0)
    for gen, max_share in ((_generate_realistic_config, 0.1), (_generate_balanced_random_config, 0.0)):
        wet = np.array([lo + p.params["wet_level"] * (hi - lo) for _ in range(3000) for p in gen(rng).plugins
                        if p.name == "reverb"])
        assert wet.min() >= 0.06 and np.median(wet) < 0.26
        assert np.mean(wet > 0.36) <= max_share


def test_generator_resume_matches_full_run(tmp_path):
    """Прерванная генерация дописывается: оборванная строка отбрасывается, итог как у непрерывного прогона."""
    from app.dataset.dataset_generator import DatasetGenerator
    t = np.arange(int(0.5 * SR)) / SR
    cambridge = tmp_path / "dry" / "cambridge"
    cambridge.mkdir(parents=True)
    for i, name in enumerate(["Song_A_01_LeadVox", "Song_B_01_LeadVox", "Song_C_01_LeadVox"]):
        sf.write(cambridge / f"{name}.wav", 0.2 * np.sin(2 * np.pi * (200 + 40 * i) * t), SR)
    full = DatasetGenerator(tmp_path / "dry", tmp_path / "full", sr=SR).generate(max_samples=6, seed=1, workers=2)
    out = tmp_path / "part"
    out.mkdir()
    lines = (tmp_path / "full" / "index.jsonl").read_text(encoding="utf-8").splitlines()
    (out / "index.jsonl").write_text("\n".join(lines[:3]) + "\n" + lines[3][:40], encoding="utf-8")
    resumed = DatasetGenerator(tmp_path / "dry", out, sr=SR).generate(max_samples=6, seed=1, workers=2, resume=True)
    key = lambda s: s.id  # noqa: E731
    assert [(s.id, s.params_vector) for s in sorted(resumed, key=key)] == \
        [(s.id, s.params_vector) for s in sorted(full, key=key)]
    assert len((out / "index.jsonl").read_text(encoding="utf-8").splitlines()) == 6
