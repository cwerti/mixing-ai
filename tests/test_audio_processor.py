import os
import tempfile
import numpy as np
import pytest
import soundfile as sf
import librosa
from app.audio.audio_processor import AudioProcessor, db_to_amplitude

@pytest.fixture
def temp_audio_file():
    """Generates a temporary mono WAV file with 1 second of sine wave + silence at ends."""
    sr = 44100
    t = np.linspace(0, 1, sr, endpoint=False)
    # Sine wave of 440 Hz
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Pad with silence (0.2s at start, 0.2s at end)
    silence = np.zeros(int(0.2 * sr))
    audio = np.concatenate([silence, sine, silence])
    
    # Save to a temp file
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, "test_synth_voice.wav")
    sf.write(file_path, audio, sr)
    
    yield file_path
    
    # Cleanup
    if os.path.exists(file_path):
        os.remove(file_path)

def test_load_audio_trim_and_normalize(temp_audio_file):
    processor = AudioProcessor(sr=44100)
    
    # Load with silence trimming and LUFS normalization
    y = processor.load_audio(temp_audio_file, target_lufs=-23.0, trim_silence=True)
    
    assert len(y) > 0
    # Original length with padding: 1.4s * 44100 = 61740 samples.
    # Trimmed length should be around 1.0s * 44100 = 44100 samples (+/- small margin).
    assert len(y) < 55000
    
    # Test that LUFS integrated loudness is close to -23.0 dB
    import pyloudnorm as pyln
    meter = pyln.Meter(processor.sr)
    loudness = meter.integrated_loudness(y)
    assert pytest.approx(loudness, abs=1.0) == -23.0

def test_spectral_envelope(temp_audio_file):
    processor = AudioProcessor(sr=44100)
    y = processor.load_audio(temp_audio_file, trim_silence=False)
    
    env = processor.get_spectral_envelope(y, n_fft=2048, window='hann')
    assert isinstance(env, np.ndarray)
    assert len(env) == 1025  # n_fft // 2 + 1
    
    # Check max is normalized to 0 dB (ref=np.max in code)
    assert np.max(env) == pytest.approx(0.0, abs=1e-5)

def test_mel_spectrogram(temp_audio_file):
    processor = AudioProcessor(sr=44100)
    y = processor.load_audio(temp_audio_file, trim_silence=False)
    
    mel = processor.get_mel_spectrogram(y, n_fft=2048, n_mels=64)
    assert isinstance(mel, np.ndarray)
    assert mel.shape[0] == 64
    assert np.max(mel) == pytest.approx(0.0, abs=1e-5)

def test_extract_features(temp_audio_file):
    processor = AudioProcessor(sr=44100)
    y = processor.load_audio(temp_audio_file, trim_silence=False)
    
    features = processor.extract_features(y)
    assert isinstance(features, dict)
    assert "spectral_centroid_mean" in features
    assert "crest_factor" in features
    assert "dynamic_range_db" in features
    assert "rms_mean" in features
    
    assert features["crest_factor"] > 1.0
    assert features["dynamic_range_db"] >= 0.0
