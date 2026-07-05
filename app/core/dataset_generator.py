import os
import random
import json
from typing import List, Dict
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from app.core.dataset_schema import DatasetSample, ChainConfig, PluginConfig, CHAIN_ORDER, DATAGEN_PLUGINS
from app.core.dsp_engine import DSPEngine
from app.core.audio_processor import AudioProcessor

class DatasetGenerator:
    """Generates synthetic dataset using dry vocals, randomized DSP chains, and feature extraction."""

    def __init__(self, dry_dir: Path, output_dir: Path, sr: int = 44100):
        self.dry_dir = Path(dry_dir)
        self.output_dir = Path(output_dir)
        self.sr = sr
        self.dsp = DSPEngine()
        self.audio_processor = AudioProcessor(sr=sr)

        # Flat parameters mapping
        self.param_indices = {
            "eq": ["band1_freq_hz", "band1_gain_db", "band2_freq_hz", "band2_gain_db", "band3_freq_hz", "band3_gain_db"],
            "compressor": ["threshold_db", "ratio", "attack_ms", "release_ms"],
            "reverb": ["room_size", "damping", "wet_level", "dry_level"],
            "delay": ["delay_seconds", "feedback", "mix"]
        }

    def _generate_random_config(self) -> ChainConfig:
        """Generates a randomized plugin chain and parameter configuration."""
        chain = ChainConfig()
        
        # Decide active plugins
        active_plugins = []
        rand_val = random.random()
        
        if rand_val < 0.2:
            # 20% EQ only
            active_plugins = ["eq"]
        elif rand_val < 0.4:
            # 20% EQ + Compressor
            active_plugins = ["eq", "compressor"]
        elif rand_val < 0.6:
            # 20% EQ + Compressor + Reverb
            active_plugins = ["eq", "compressor", "reverb"]
        elif rand_val < 0.8:
            # 20% Full chain
            active_plugins = ["eq", "compressor", "reverb", "delay"]
        else:
            # 20% Random subset (non-empty)
            num_plugins = random.randint(1, len(CHAIN_ORDER))
            active_plugins = random.sample(CHAIN_ORDER, num_plugins)
            
        # Sort active plugins according to global CHAIN_ORDER
        active_plugins = [p for p in CHAIN_ORDER if p in active_plugins]
        
        for name in active_plugins:
            plugin_range = DATAGEN_PLUGINS[name]["params"]
            params = {}
            for p_name in plugin_range:
                # Generate random normalized parameter in [0.0, 1.0]
                params[p_name] = float(random.random())
            
            chain.plugins.append(PluginConfig(name=name, params=params))
            
        return chain

    def _flatten_params(self, chain_config: ChainConfig) -> List[float]:
        """Flattens plugin parameters to a fixed-size vector of length 17."""
        # 17 float elements initialized to 0.0
        vec = [0.0] * 17
        
        # Mapping index offsets
        offset_map = {
            "eq": 0,
            "compressor": 6,
            "reverb": 10,
            "delay": 14
        }
        
        for plugin in chain_config.plugins:
            offset = offset_map[plugin.name]
            p_names = self.param_indices[plugin.name]
            for i, p_name in enumerate(p_names):
                vec[offset + i] = plugin.params.get(p_name, 0.0)
                
        return vec

    def generate(self, max_samples: int = 1000, sc_ratio: float = 0.5) -> List[DatasetSample]:
        """Runs the dataset generation loop over files in dry_dir maintaining the sc_ratio."""
        # Prepare subdirectories
        dry_out_dir = self.output_dir / "audio" / "dry"
        wet_out_dir = self.output_dir / "audio" / "wet"
        features_out_dir = self.output_dir / "features"
        
        dry_out_dir.mkdir(parents=True, exist_ok=True)
        wet_out_dir.mkdir(parents=True, exist_ok=True)
        features_out_dir.mkdir(parents=True, exist_ok=True)

        # Separate dry files by source folder
        vctk_dir = self.dry_dir / "vctk"
        sc_dir = self.dry_dir / "soundcloud"
        
        vctk_files = []
        sc_files = []
        for ext in ["*.wav", "*.mp3", "*.flac"]:
            if vctk_dir.exists():
                vctk_files.extend(list(vctk_dir.rglob(ext)))
            if sc_dir.exists():
                sc_files.extend(list(sc_dir.rglob(ext)))

        if not vctk_files and not sc_files:
            print(f"[-] No dry audio files found in: {self.dry_dir}")
            return []

        # Determine target count per source based on sc_ratio
        if sc_files and vctk_files:
            sc_target = int(max_samples * sc_ratio)
            vctk_target = max_samples - sc_target
        elif sc_files:
            print("[!] VCTK source files not found. Generating 100% from SoundCloud.")
            sc_target = max_samples
            vctk_target = 0
        else:
            print("[!] SoundCloud source files not found. Generating 100% from VCTK.")
            sc_target = 0
            vctk_target = max_samples

        print(f"[*] Target distribution: SoundCloud: {sc_target} samples, VCTK: {vctk_target} samples.")
        
        # Build sources pool
        source_pool = (["soundcloud"] * sc_target) + (["vctk"] * vctk_target)
        random.shuffle(source_pool)

        samples = []
        index_file = self.output_dir / "index.jsonl"
        
        # Write to JSONL directly in a loop
        with open(index_file, "w", encoding="utf-8") as f_index:
            for i, src_type in enumerate(tqdm(source_pool)):
                # Choose file from correct pool
                if src_type == "soundcloud":
                    src_file = random.choice(sc_files)
                    source_name = "soundcloud"
                else:
                    src_file = random.choice(vctk_files)
                    source_name = "vctk"
                
                try:
                    # 2. Load the dry audio (LUFS-normalized to target -23dB)
                    y_dry = self.audio_processor.load_audio(src_file, target_lufs=-23.0, trim_silence=True)
                    duration = float(len(y_dry) / self.sr)
                    
                    # 3. Generate randomized DSP config
                    chain_config = self._generate_random_config()
                    
                    # 4. Process audio via DSPEngine
                    y_wet = self.dsp.apply_chain(y_dry, self.sr, chain_config)
                    
                    # 5. Define output names
                    sample_id = f"{i:06d}"
                    dry_name = f"{source_name}_{sample_id}_dry.wav"
                    wet_name = f"{source_name}_{sample_id}_wet.wav"
                    
                    dry_path = dry_out_dir / dry_name
                    wet_path = wet_out_dir / wet_name
                    
                    # Save WAV files
                    sf.write(dry_path, y_dry, self.sr)
                    sf.write(wet_path, y_wet, self.sr)
                    
                    # 6. Extract spectral envelope / Mel spectrograms
                    mel_dry = self.audio_processor.get_mel_spectrogram(y_dry, n_mels=128)
                    mel_wet = self.audio_processor.get_mel_spectrogram(y_wet, n_mels=128)
                    
                    # Save предвычисленные спектрограммы (Mel)
                    feature_file = features_out_dir / f"{sample_id}.npz"
                    np.savez_compressed(feature_file, mel_dry=mel_dry, mel_wet=mel_wet)
                    
                    # 7. Construct sample record
                    chain_list = [p.name for p in chain_config.plugins]
                    chain_onehot = [1 if p in chain_list else 0 for p in CHAIN_ORDER]
                    
                    params_dict = {}
                    for plugin in chain_config.plugins:
                        params_dict[plugin.name] = plugin.params
                        
                    flat_vector = self._flatten_params(chain_config)
                    
                    sample = DatasetSample(
                        id=sample_id,
                        dry_path=str(Path("audio/dry") / dry_name),
                        wet_path=str(Path("audio/wet") / wet_name),
                        source=source_name,
                        duration_sec=duration,
                        chain=chain_list,
                        chain_onehot=chain_onehot,
                        params=params_dict,
                        params_vector=flat_vector
                    )
                    
                    # Write to index.jsonl
                    f_index.write(sample.to_jsonl_line() + "\n")
                    samples.append(sample)
                    
                except Exception as e:
                    # Skip corrupt files silently or log them
                    print(f"\n[-] Failed to generate sample {i} from {src_file.name}: {e}")
                    continue
                    
        # Write dataset stats
        stats = {
            "total_samples": len(samples),
            "sr": self.sr,
            "chain_order": CHAIN_ORDER,
            "param_layout": self.param_indices
        }
        with open(self.output_dir / "stats.json", "w", encoding="utf-8") as f_stats:
            json.dump(stats, f_stats, indent=4)
            
        print(f"[+] Dataset generation completed successfully. Output path: {self.output_dir}")
        return samples
