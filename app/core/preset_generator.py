import numpy as np
from pathlib import Path

class PresetGenerator:
    def __init__(self):
        # Константы для Fruity Parametric EQ 2
        self.MIN_FREQ = 10.0
        self.MAX_FREQ = 20000.0
        self.MIN_GAIN = -18.0
        self.MAX_GAIN = 18.0

    def freq_to_param(self, freq):
        freq = np.clip(freq, self.MIN_FREQ, self.MAX_FREQ)
        return (np.log10(freq) - np.log10(self.MIN_FREQ)) / (np.log10(self.MAX_FREQ) - np.log10(self.MIN_FREQ))

    def gain_to_param(self, gain):
        gain = np.clip(gain, self.MIN_GAIN, self.MAX_GAIN)
        return (gain - self.MIN_GAIN) / (self.MAX_GAIN - self.MIN_GAIN)

    def extract_key_bands(self, freqs, delta, num_bands=7):
        mask = (freqs >= 50) & (freqs <= 16000)
        f_filtered = freqs[mask]
        d_filtered = delta[mask]
        
        if len(f_filtered) == 0:
            return [(1000.0, 0.0)] * num_bands

        indices = np.linspace(0, len(f_filtered) - 1, num_bands + 1).astype(int)
        bands = []
        for i in range(num_bands):
            idx_start = indices[i]
            idx_end = indices[i+1]
            if idx_start == idx_end and i < num_bands - 1:
                idx_end = idx_start + 1
            
            segment_f = f_filtered[idx_start:idx_end]
            segment_d = d_filtered[idx_start:idx_end]
            
            avg_freq = segment_f[len(segment_f)//2]
            avg_gain = np.mean(segment_d)
            bands.append((avg_freq, avg_gain))
        return bands

    def generate_one_liner(self, bands):
        """Генерирует проверенный однострочник для FL Studio 25."""
        band_data = ",".join([f"({f:.0f}, {g:.2f})" for f, g in bands])
        
        # Используем проверенный синтаксис: plugins.setParamValue(val, idx, tr, slot)
        script = (
            f"import plugins, mixer, math; tr = mixer.trackNumber(); "
            f"b = [{band_data}]; "
            f"[(plugins.setParamValue((math.log10(f)-1)/3.301, i*5, tr, 0), "
            f"plugins.setParamValue((g+18)/36, i*5+1, tr, 0)) "
            f"for i, (f, g) in enumerate(b) if plugins.isValid(tr, 0)]"
        )
        return script
