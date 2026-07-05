import numpy as np
import time
import librosa
from pathlib import Path
from app.core.audio_processor import AudioProcessor
from app.core.bridge_client import BridgeClient
from app.core.plugin_manager import PluginLoader

class MixingAIPipeline:
    def __init__(self, port_name: str = 'Mixing Ai'):
        self.processor = AudioProcessor()
        self.client = BridgeClient(port_name=port_name)
        self.loader = PluginLoader(self.client)

    def run(self, source_path: str = "data/raw/source/source.wav", ref_path: str = "data/raw/reference/reference.wav"):
        print("\n=== Mixing-AI Integrated Pipeline ===")
        
        # 1. Load and analyze audio
        src_file = Path(source_path)
        ref_file = Path(ref_path)
        
        if src_file.exists() and ref_file.exists():
            print(f"[*] Processing audio files:\n  Source: {src_file}\n  Reference: {ref_file}")
            # Real analysis
            y_src = self.processor.load_audio(src_file)
            y_ref = self.processor.load_audio(ref_file)
            
            env_src = self.processor.get_spectral_envelope(y_src)
            env_ref = self.processor.get_spectral_envelope(y_ref)
            delta = env_ref - env_src
            freqs = librosa.fft_frequencies(sr=self.processor.sr, n_fft=2048)
            
            # Extract advanced features for printout / future use
            features_src = self.processor.extract_features(y_src)
            features_ref = self.processor.extract_features(y_ref)
            print(f"[*] Source crest factor: {features_src['crest_factor']:.2f}, dynamic range: {features_src['dynamic_range_db']:.2f} dB")
            print(f"[*] Reference crest factor: {features_ref['crest_factor']:.2f}, dynamic range: {features_ref['dynamic_range_db']:.2f} dB")
        else:
            print("[-] Source or Reference audio file not found. Falling back to mock data.")
            freqs = np.linspace(20, 20000, 1024)
            delta = np.random.uniform(-5, 5, 1024)

        # 2. Connect to FL Studio
        if not self.client.connect_midi():
            print("[-] MIDI connection failed. Make sure virtual MIDI port is active and FL Studio is running.")
            return False

        # 3. Prepare mixer track
        if not self.client.prepare_empty_track():
            print("[-] Failed to prepare empty mixer track.")
            return False

        # 4. Load plugins
        chain = ["Fruity Parametric EQ 2", "Fruity Limiter"]
        for plugin in chain:
            if not self.loader.load(plugin):
                print(f"[-] Failed to load plugin: {plugin}")
            time.sleep(2.0)

        # 5. Wait for plugin initialization
        print("[*] Waiting for plugins to initialize...")
        time.sleep(2.0) 

        # 6. Apply spectral EQ correction settings
        print("[*] Setting EQ parameters...")
        if self.client.set_eq_params(freqs, delta, "Fruity Parametric EQ 2"):
            print("[+] SUCCESS: Vocal chain and EQ parameters applied to FL Studio.")
            return True
        else:
            print("[-] Failed to apply EQ parameters.")
            return False

if __name__ == "__main__":
    pipeline = MixingAIPipeline()
    pipeline.run()
