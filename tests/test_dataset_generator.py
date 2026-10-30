import os
import tempfile
import json
import numpy as np
import pytest
import soundfile as sf
from pathlib import Path
from app.dataset.dataset_generator import DatasetGenerator
from app.dataset.dataset_schema import PARAMETER_VECTOR_LENGTH

@pytest.fixture
def temp_dataset_env():
    """Sets up a temporary folder with mock dry WAV files."""
    temp_dir = Path(tempfile.gettempdir()) / "test_mixing_ai_dataset"
    dry_dir = temp_dir / "raw" / "dry_vocals"
    output_dir = temp_dir / "processed" / "dataset_v1"
    
    dry_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create 3 mock dry vocal files
    sr = 44100
    t = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False)
    y1 = 0.3 * np.sin(2 * np.pi * 220 * t)
    y2 = 0.2 * np.sin(2 * np.pi * 440 * t)
    
    vctk_dir = dry_dir / "vctk"
    sc_dir = dry_dir / "soundcloud"
    vctk_dir.mkdir(parents=True, exist_ok=True)
    sc_dir.mkdir(parents=True, exist_ok=True)
    
    sf.write(vctk_dir / "vctk_p225_001.wav", y1, sr)
    sf.write(vctk_dir / "vctk_p225_002.wav", y2, sr)
    sf.write(sc_dir / "sc_track1.wav", y1, sr)
    
    yield dry_dir, output_dir
    
    # Cleanup
    if temp_dir.exists():
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_dataset_generator(temp_dataset_env):
    dry_dir, output_dir = temp_dataset_env
    
    generator = DatasetGenerator(dry_dir=dry_dir, output_dir=output_dir, sr=44100)
    samples = generator.generate(max_samples=5)
    
    # We requested 5 samples, should generate 5
    assert len(samples) == 5
    
    # Check index.jsonl exists
    index_file = output_dir / "index.jsonl"
    assert index_file.exists()
    
    # Validate structure of JSONL entries
    with open(index_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 5
        
        for line in lines:
            data = json.loads(line)
            assert "id" in data
            assert "dry_path" in data
            assert "wet_path" in data
            assert "chain" in data
            assert "chain_onehot" in data
            assert "params" in data
            assert "params_vector" in data
            
            from app.dataset.dataset_schema import CHAIN_ORDER
            assert len(data["chain_onehot"]) == len(CHAIN_ORDER)  # EQ, Comp, Dist, Chorus, Reverb, Delay
            assert len(data["params_vector"]) == PARAMETER_VECTOR_LENGTH  # Fixed parameter vector layout
            
            # Check files exist on disk
            assert (output_dir / data["dry_path"]).exists()
            assert (output_dir / data["wet_path"]).exists()
            
            # Check feature npz exists
            npz_path = output_dir / "features" / f"{data['id']}.npz"
            assert npz_path.exists()
            
            # Try to load features and verify
            loaded = np.load(npz_path)
            assert "mel_dry" in loaded
            assert "mel_wet" in loaded
            assert "desc_dry" in loaded
            assert "desc_wet" in loaded
            assert loaded["mel_dry"].shape[:2] == (2, 128)
            assert loaded["mel_wet"].shape[:2] == (2, 128)
            assert loaded["desc_dry"].shape == (18,)
            assert loaded["desc_wet"].shape == (18,)
            assert data.get("split") in ("train", "val")

    # Validate stats.json exists
    stats_file = output_dir / "stats.json"
    assert stats_file.exists()
    with open(stats_file, "r") as fs:
        stats = json.load(fs)
        assert stats["total_samples"] == 5
        assert stats["sr"] == 44100


def test_is_dry_source_excludes_processed_vocals(tmp_path):
    """Обработанные инженером файлы из пула Cambridge не считаются сухими, их сухие версии — считаются."""
    from app.dataset.dataset_generator import PROCESSED_VOCAL_STEMS, is_dry_source

    assert "Albert_Kader_Whiptails_12_Vox_Wet" in PROCESSED_VOCAL_STEMS
    sr = 44100
    tone = 0.2 * np.sin(2 * np.pi * 220 * np.arange(sr // 10) / sr)
    dry_names = ["Albert_Kader_Whiptails_11_Vox_Dry", "oxeye_Sorry_13_LeadVox_NoPitchProcessing",
                 "Ronnie_Jones_Why_Dont_You_Stay_24_LeadVoxRaw"]
    for name in [*PROCESSED_VOCAL_STEMS, *dry_names]:
        sf.write(tmp_path / f"{name}.wav", tone, sr)

    for name in PROCESSED_VOCAL_STEMS:
        assert not is_dry_source(tmp_path / f"{name}.wav")
    for name in dry_names:
        assert is_dry_source(tmp_path / f"{name}.wav")
    assert not is_dry_source(tmp_path / "missing.wav")


def test_generator_never_uses_processed_vocals(tmp_path):
    """Генератор не берет обработанные файлы ни как исходник, ни как голос референса."""
    sr = 44100
    t = np.arange(int(0.5 * sr)) / sr
    cambridge = tmp_path / "dry" / "cambridge"
    cambridge.mkdir(parents=True)
    sf.write(cambridge / "Albert_Kader_Whiptails_11_Vox_Dry.wav", 0.3 * np.sin(2 * np.pi * 220 * t), sr)
    sf.write(cambridge / "Albert_Kader_Whiptails_12_Vox_Wet.wav", 0.3 * np.sin(2 * np.pi * 220 * t), sr)
    sf.write(cambridge / "Other_Song_01_LeadVox.wav", 0.2 * np.sin(2 * np.pi * 330 * t), sr)

    generator = DatasetGenerator(dry_dir=tmp_path / "dry", output_dir=tmp_path / "out", sr=sr)
    generator.generate(max_samples=6)

    lines = (tmp_path / "out" / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert lines
    used = {Path(json.loads(line)[key]).name for line in lines for key in ("src_file", "ref_file")}
    assert "Albert_Kader_Whiptails_12_Vox_Wet.wav" not in used


def test_realistic_prior_rates_follow_real_chains():
    """Реалистичный prior: динамика, тюн и реверб частые, chorus/distortion/stereo_enhancer редкие."""
    import random
    from app.dataset.dataset_generator import _generate_realistic_config, realistic_on_rates

    rates = realistic_on_rates()
    assert rates["eq"] == 1.0
    assert rates["compressor"] > 0.85 and rates["reverb"] > 0.75 and rates["pitch_corrector"] > 0.6
    assert rates["chorus"] < 0.25 and rates["distortion"] < 0.3 and rates["stereo_enhancer"] < 0.2

    rng = random.Random(0)
    chains = [_generate_realistic_config(rng) for _ in range(3000)]
    chorus = [p.params for c in chains for p in c.plugins if p.name == "chorus"]
    drive = [p.params["drive_db"] for c in chains for p in c.plugins if p.name == "distortion"]
    assert abs(len(chorus) / len(chains) - rates["chorus"]) < 0.03
    # Слабые настройки: почти чистая сатурация (< 4 дБ) и легкий хорус (mix < 11 %) — заметная доля
    assert np.mean(np.array(drive) < 0.2) > 0.3
    assert np.mean([c["mix"] < 0.15 for c in chorus]) > 0.3
    for chain in chains[:200]:
        for plugin in chain.plugins:
            assert all(0.0 <= v <= 1.0 for v in plugin.params.values()), plugin


def test_reference_layers_render_only():
    """Слои harmony встают перед реверб/дилей в рендере референса и не попадают в метку модели."""
    import random
    from app.dataset.dataset_generator import _generate_realistic_config, sample_reference_layers, with_reference_layers
    from app.dataset.dataset_schema import MODEL_PLUGINS, RENDER_ORDER

    rng = random.Random(3)
    for _ in range(50):
        chain = _generate_realistic_config(rng)
        layers = sample_reference_layers(rng)
        assert 1 <= len(layers) <= 2 and all(layer.name == "harmony" for layer in layers)
        names = [p.name for p in with_reference_layers(chain, layers).plugins]
        assert [n for n in names if n != "harmony"] == [p.name for p in chain.plugins]
        assert names == sorted(names, key=RENDER_ORDER.index)
        assert "harmony" not in MODEL_PLUGINS


def test_generator_excludes_test_songs_and_stores_layers(tmp_path):
    """Песни реальных тестовых пар не попадают ни в train, ни в val; слои референса пишутся отдельно от метки."""
    sr = 44100
    t = np.arange(int(0.5 * sr)) / sr
    cambridge = tmp_path / "dry" / "cambridge"
    cambridge.mkdir(parents=True)
    for i, name in enumerate(["oxeye_Sorry_13_LeadVox_NoPitchProcessing", "Ronnie_Jones_Why_Dont_You_Stay_24_LeadVoxRaw",
                              "Song_A_01_LeadVox", "Song_B_01_LeadVox", "Song_C_01_LeadVox"]):
        sf.write(cambridge / f"{name}.wav", 0.2 * np.sin(2 * np.pi * (200 + 40 * i) * t), sr)

    generator = DatasetGenerator(dry_dir=tmp_path / "dry", output_dir=tmp_path / "out", sr=sr)
    samples = generator.generate(max_samples=8, harmony_ratio=1.0)

    assert len(samples) == 8
    used = {Path(getattr(s, key)).stem for s in samples for key in ("src_file", "ref_file")}
    assert not any(stem.startswith(("oxeye_Sorry", "Ronnie_Jones")) for stem in used)
    assert all(s.ref_layers and "harmony" not in s.chain and "harmony" not in s.params for s in samples)
    stats = json.loads((tmp_path / "out" / "stats.json").read_text(encoding="utf-8"))
    assert stats["prior"] == "realistic" and stats["with_reference_layers"] == 8


def test_remap_param_vector_keeps_physical_values_of_old_checkpoints():
    """Предсказания чекпоинта до v5 пересчитываются в текущие диапазоны с тем же физическим значением."""
    from app.dataset.dataset_schema import (
        DATAGEN_PLUGINS, LEGACY_PARAM_RANGES, PARAM_OFFSETS, PARAMETER_VECTOR_LENGTH, model_param_ranges,
        remap_param_vector,
    )

    vec = np.linspace(0.0, 1.0, PARAMETER_VECTOR_LENGTH).astype(np.float32)
    assert np.array_equal(remap_param_vector(vec, model_param_ranges()), vec)
    out = remap_param_vector(vec, None)
    changed = {i for i in range(PARAMETER_VECTOR_LENGTH) if out[i] != vec[i]}
    expected = {PARAM_OFFSETS["distortion"], PARAM_OFFSETS["chorus"] + 3}
    assert changed <= expected
    for (plugin, param), (lo, hi) in LEGACY_PARAM_RANGES.items():
        idx = PARAM_OFFSETS[plugin] + (0 if plugin == "distortion" else 3)
        new_lo, new_hi = DATAGEN_PLUGINS[plugin]["params"][param]
        assert lo + vec[idx] * (hi - lo) == pytest.approx(new_lo + out[idx] * (new_hi - new_lo), abs=1e-5)
