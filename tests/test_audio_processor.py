import os
import tempfile
import numpy as np
import pytest
import soundfile as sf
import librosa
from app.audio.audio_processor import AudioProcessor, SpectralMatcher, db_to_amplitude

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

def test_stereo_mel_and_mixing_descriptors(temp_audio_file):
    processor = AudioProcessor(sr=44100)
    y_mono = processor.load_audio(temp_audio_file, trim_silence=False)
    y_stereo = np.vstack([y_mono, np.roll(y_mono, 400) * 0.8])

    stereo_mel = processor.get_stereo_mel_spectrogram(y_stereo, n_mels=128)
    assert stereo_mel.ndim == 3
    assert stereo_mel.shape[0] == 2
    assert stereo_mel.shape[1] == 128
    assert np.all(stereo_mel >= -1.0) and np.all(stereo_mel <= 1.0)

    desc_mono = processor.extract_mixing_descriptors(y_mono)
    desc_stereo = processor.extract_mixing_descriptors(y_stereo)
    assert desc_mono.shape == (18,)
    assert desc_stereo.shape == (18,)
    assert np.all(desc_mono >= -1.0) and np.all(desc_mono <= 1.0)
    assert np.all(desc_stereo >= -1.0) and np.all(desc_stereo <= 1.0)
    # Стерео-сигнал с задержкой должен иметь большую стерео-ширину (признак 10), чем моно
    assert desc_stereo[10] > desc_mono[10]



def _phrased_noise(sr: int, seconds: float = 6.0, seed: int = 0) -> np.ndarray:
    """Розовый шум, нарезанный на «фразы» с паузами цифровой тишины."""
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.normal(0, 1, int(sr * seconds)))
    freqs = np.fft.rfftfreq(int(sr * seconds), d=1.0 / sr)
    pink = np.fft.irfft(spectrum / np.sqrt(np.maximum(freqs, 20.0)), n=int(sr * seconds))
    gate = (np.arange(len(pink)) // int(0.4 * sr)) % 3 != 2
    return (0.1 * pink / np.std(pink) * gate).astype(np.float32)


def test_match_eq_recovers_known_curve():
    """EQ, подобранный по паре (сухой, сухой + известный EQ), восстанавливает известные усиления полос."""
    from pedalboard import HighShelfFilter, LowShelfFilter, PeakFilter, Pedalboard

    sr = 44100
    matcher = SpectralMatcher(sr=sr)
    true_gains = [-4.0, 3.0, 0.0, -2.0, 4.0, 0.0, 5.0]
    centers = SpectralMatcher.BAND_CENTERS_HZ
    board = Pedalboard(
        [LowShelfFilter(cutoff_frequency_hz=centers[0], gain_db=true_gains[0], q=0.707)]
        + [PeakFilter(cutoff_frequency_hz=f, gain_db=g, q=1.0) for f, g in zip(centers[1:6], true_gains[1:6])]
        + [HighShelfFilter(cutoff_frequency_hz=centers[6], gain_db=true_gains[6], q=0.707)]
    )
    dry = _phrased_noise(sr)
    ref = 0.5 * board(dry[np.newaxis, :], sr)[0]  # другая общая громкость не должна попасть в EQ

    bands = matcher.match_eq(dry, ref)

    # Частоты полос подбираются, поэтому сравниваем итоговую АЧХ EQ, а не гейны по полосам
    grid = matcher._grid_hz
    fitted = sum(g * matcher.band_response_db(i, f) for i, (f, g) in enumerate(bands))
    expected = sum(g * matcher.band_response_db(i, f) for i, (f, g) in enumerate(zip(centers, true_gains)))
    vocal_range = (grid >= 100) & (grid <= 12000)
    error = (fitted - expected)[vocal_range]
    assert np.max(np.abs(error - error.mean())) < 1.0  # общий сдвиг громкости в EQ не входит
    assert all(low <= f <= high for (f, _), (low, high) in zip(bands, SpectralMatcher.BAND_RANGES_HZ))


def test_eq_limits_register_region():
    """Разница +15 дБ на низах (регистр другого голоса) ограничивается, середина и верх копируются полностью."""
    matcher = SpectralMatcher(sr=44100)
    grid = matcher._grid_hz
    delta = np.where(grid < 250, 15.0, 0.0) + np.where(grid > 3000, -6.0, 0.0) - 20.0  # -20: общая громкость
    limited = matcher.limit_register_delta(delta)
    assert np.allclose(limited[grid < 250] + 20.0, matcher.low_gain_limit_db)
    assert np.allclose(limited[grid > 3000], delta[grid > 3000])
    assert SpectralMatcher(sr=44100, low_gain_limit_db=None).limit_register_delta(delta) is delta


def test_match_eq_flat_for_identical_signals():
    """Для одинаковых сигналов EQ остается плоским."""
    sr = 44100
    dry = _phrased_noise(sr)
    bands = SpectralMatcher(sr=sr).match_eq(dry, dry.copy())
    assert all(abs(gain) < 0.5 for _, gain in bands)
