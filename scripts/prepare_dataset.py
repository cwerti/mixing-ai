import argparse
import sys
from pathlib import Path
import soundfile as sf
import numpy as np

# Adjust path to find app module
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.core.vocal_collector import VocalCollector
from app.core.dataset_generator import DatasetGenerator

def main():
    parser = argparse.ArgumentParser(description="Mixing-AI Dataset Preparation CLI")
    parser.add_argument("--max-samples", type=int, default=100, help="Maximum number of dataset samples to generate")
    parser.add_argument("--output-dir", type=str, default="data/processed/dataset_v1", help="Path to store processed dataset")
    parser.add_argument("--soundcloud-urls", type=str, default="data/raw/soundcloud_urls.txt", help="Path to SoundCloud URLs text file")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    urls_file = Path(args.soundcloud_urls)
    
    raw_dry_dir = Path("data/raw/dry_vocals")
    raw_dry_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Check if we have urls_file and download
    collector = VocalCollector(sr=44100)
    if urls_file.exists():
        print("[*] Running SoundCloud vocal collection pipeline...")
        download_dir = Path("data/raw/soundcloud_temp")
        downloaded = collector.download_soundcloud(urls_file, download_dir)
        
        # Extract vocals using Demucs and slice them
        for f_path in downloaded:
            separated = collector.separate_vocals(f_path, Path("data/raw/soundcloud_separated"))
            if separated:
                collector.slice_audio(
                    audio_path=separated,
                    output_dir=raw_dry_dir / "soundcloud",
                    prefix=f"sc_{f_path.stem}"
                )
        # Clean up temporary downloads
        if download_dir.exists():
            import shutil
            shutil.rmtree(download_dir, ignore_errors=True)
            
    # 2. Check if we have VCTK or other dry vocals
    # Fallback option: if no dry vocals found, generate a basic synthetic dry voice from project audio
    dry_files = list(raw_dry_dir.rglob("*.wav")) + list(raw_dry_dir.rglob("*.mp3"))
    
    if not dry_files:
        print("[!] No dry vocal files found in data/raw/dry_vocals/.")
        print("[*] Generating synthetic seed voice from data/raw/source/source.wav for testing...")
        
        src_wav = Path("data/raw/source/source.wav")
        if src_wav.exists():
            collector.slice_audio(
                audio_path=src_wav,
                output_dir=raw_dry_dir / "vctk",
                prefix="vctk_synth_seed"
            )
            dry_files = list(raw_dry_dir.rglob("*.wav"))
        else:
            print("[-] Error: Seed file data/raw/source/source.wav not found. Please add a wav file to start.")
            sys.exit(1)

    print(f"[+] Total dry vocal sources available for generation: {len(dry_files)}")
    
    # 3. Generate dataset
    generator = DatasetGenerator(dry_dir=raw_dry_dir, output_dir=output_dir, sr=44100)
    generator.generate(max_samples=args.max_samples)

if __name__ == "__main__":
    main()
