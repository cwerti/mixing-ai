import librosa
import numpy as np
import pyloudnorm as pyln
from pathlib import Path

def db_to_amplitude(db):
    """Converts decibels to linear amplitude."""
    return 10 ** (db / 20.0)

class AudioProcessor:
    def __init__(self, sr=44100):
        self.sr = sr

    def load_audio(self, file_path, target_lufs=-23.0, trim_silence=True, top_db=60):
        """Loads, trims silence, and LUFS-normalizes audio."""
        y, sr = librosa.load(file_path, sr=self.sr)
        if trim_silence:
            y, _ = librosa.effects.trim(y, top_db=top_db)
        
        # LUFS normalization using pyloudnorm
        meter = pyln.Meter(self.sr)
        try:
            loudness = meter.integrated_loudness(y)
            if not np.isnan(loudness) and not np.isinf(loudness):
                y = pyln.normalize.loudness(y, loudness, target_lufs)
        except Exception as e:
            # Fallback to peak normalization if LUFS fails (e.g. extremely short or silent audio)
            print(f"LUFS normalization failed ({e}), falling back to peak normalization.")
            y = librosa.util.normalize(y) * db_to_amplitude(-1.0)
            
        return y

    def get_spectral_envelope(self, y, n_fft=2048, window='hann'):
        """Computes the mean magnitude spectrum (spectral envelope) in dB."""
        S = np.abs(librosa.stft(y, n_fft=n_fft, window=window))
        # Time average
        avg_spectrum = np.mean(S, axis=1)
        # Convert to dB
        avg_db = librosa.amplitude_to_db(avg_spectrum, ref=np.max)
        return avg_db

    def get_mel_spectrogram(self, y, n_fft=2048, hop_length=512, n_mels=128, window='hann'):
        """Computes Mel-spectrogram in dB."""
        S = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, window=window
        )
        return librosa.power_to_db(S, ref=np.max)

    def extract_features(self, y):
        """Extracts advanced audio features (dynamic range, crest factor, spectral centroid)."""
        # 1. Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr)
        mean_centroid = float(np.mean(centroid))

        # 2. Crest Factor: ratio of peak amplitude to RMS
        rms_val = librosa.feature.rms(y=y)
        mean_rms = float(np.mean(rms_val))
        peak = float(np.max(np.abs(y)))
        crest_factor = (peak / mean_rms) if mean_rms > 1e-6 else 0.0

        # 3. Dynamic Range (simple estimate: difference between peak and RMS floor in dB)
        rms_db = librosa.amplitude_to_db(rms_val, ref=np.max)
        dynamic_range = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))

        return {
            "spectral_centroid_mean": mean_centroid,
            "crest_factor": crest_factor,
            "dynamic_range_db": dynamic_range,
            "rms_mean": mean_rms
        }

if __name__ == "__main__":
    # Self-test block
    processor = AudioProcessor()
    print("AudioProcessor class successfully loaded.")
