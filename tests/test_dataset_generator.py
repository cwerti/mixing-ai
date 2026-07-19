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
            assert loaded["mel_dry"].shape[0] == 128
            assert loaded["mel_wet"].shape[0] == 128

    # Validate stats.json exists
    stats_file = output_dir / "stats.json"
    assert stats_file.exists()
    with open(stats_file, "r") as fs:
        stats = json.load(fs)
        assert stats["total_samples"] == 5
        assert stats["sr"] == 44100
