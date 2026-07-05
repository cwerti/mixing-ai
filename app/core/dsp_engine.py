import numpy as np
from pedalboard import Pedalboard, Compressor, Reverb, Delay, LowShelfFilter, PeakFilter, HighShelfFilter
from app.core.dataset_schema import ChainConfig, PluginConfig, DATAGEN_PLUGINS

class DSPEngine:
    """Wraps Pedalboard to process audio using configurations with normalized [0..1] parameters."""

    @staticmethod
    def denormalize_value(norm_val: float, min_val: float, max_val: float) -> float:
        """Converts [0..1] normalized value to physical range [min_val..max_val]."""
        clamped = np.clip(norm_val, 0.0, 1.0)
        return float(min_val + clamped * (max_val - min_val))

    @staticmethod
    def normalize_value(phys_val: float, min_val: float, max_val: float) -> float:
        """Converts physical value in [min_val..max_val] to [0..1]."""
        clamped = np.clip(phys_val, min_val, max_val)
        denom = max_val - min_val
        return float((clamped - min_val) / denom if denom > 0 else 0.0)

    def get_physical_params(self, plugin_config: PluginConfig) -> dict:
        """Returns physical parameters mapped from [0..1] configs."""
        plugin_name = plugin_config.name
        if plugin_name not in DATAGEN_PLUGINS:
            raise ValueError(f"Plugin '{plugin_name}' not supported by DSPEngine.")
        
        ranges = DATAGEN_PLUGINS[plugin_name]["params"]
        phys_params = {}
        for p_name, val in plugin_config.params.items():
            if p_name in ranges:
                min_v, max_v = ranges[p_name]
                phys_params[p_name] = self.denormalize_value(val, min_v, max_v)
            else:
                # Keep parameter as is if not in randomized list
                phys_params[p_name] = val
        return phys_params

    def apply_chain(self, y: np.ndarray, sr: int, chain_config: ChainConfig) -> np.ndarray:
        """Applies a list of plugin configurations to the audio array."""
        effects = []
        
        for plugin in chain_config.plugins:
            phys = self.get_physical_params(plugin)
            
            if plugin.name == "eq":
                # Create a 3-band EQ chain
                # Band 1: Low Shelf
                effects.append(LowShelfFilter(
                    cutoff_frequency_hz=phys.get("band1_freq_hz", 150.0),
                    gain_db=phys.get("band1_gain_db", 0.0),
                    q=0.707
                ))
                # Band 2: Peak Filter (Mid)
                effects.append(PeakFilter(
                    cutoff_frequency_hz=phys.get("band2_freq_hz", 1000.0),
                    gain_db=phys.get("band2_gain_db", 0.0),
                    q=1.0
                ))
                # Band 3: High Shelf
                effects.append(HighShelfFilter(
                    cutoff_frequency_hz=phys.get("band3_freq_hz", 5000.0),
                    gain_db=phys.get("band3_gain_db", 0.0),
                    q=0.707
                ))
                
            elif plugin.name == "compressor":
                effects.append(Compressor(
                    threshold_db=phys.get("threshold_db", -20.0),
                    ratio=phys.get("ratio", 4.0),
                    attack_ms=phys.get("attack_ms", 2.0),
                    release_ms=phys.get("release_ms", 100.0)
                ))
                
            elif plugin.name == "reverb":
                effects.append(Reverb(
                    room_size=phys.get("room_size", 0.5),
                    damping=phys.get("damping", 0.5),
                    wet_level=phys.get("wet_level", 0.1),
                    dry_level=phys.get("dry_level", 0.9)
                ))
                
            elif plugin.name == "delay":
                effects.append(Delay(
                    delay_seconds=phys.get("delay_seconds", 0.25),
                    feedback=phys.get("feedback", 0.2),
                    mix=phys.get("mix", 0.1)
                ))
                
        if not effects:
            return y.copy()
            
        board = Pedalboard(effects)
        # Pedalboard outputs float32 mono or stereo array depending on input
        # We ensure it processes correctly
        try:
            return board(y, sr)
        except Exception as e:
            print(f"[-] Pedalboard processing error: {e}")
            return y.copy()
