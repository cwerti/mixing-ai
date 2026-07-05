import argparse
import sys
from pathlib import Path
import soundfile as sf
import numpy as np

# Prevent Windows console UnicodeEncodeError crashes when printing non-ASCII filenames
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(errors='replace')

# Adjust path to find app module
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.core.vocal_collector import VocalCollector
from app.core.dataset_generator import DatasetGenerator

def main():
    # Dynamically inject WinGet FFmpeg directory into PATH to bypass Windows environment refresh issues
    try:
        import os
        winget_base = Path(os.path.expanduser("~")) / "AppData/Local/Microsoft/WinGet/Packages"
        if winget_base.exists():
            ffmpeg_bins = list(winget_base.rglob("ffmpeg.exe"))
            if ffmpeg_bins:
                ffmpeg_bin_dir = ffmpeg_bins[0].parent
                os.environ["PATH"] = str(ffmpeg_bin_dir.absolute()) + os.pathsep + os.environ["PATH"]
                print(f"[+] Programmatically added FFmpeg to PATH: {ffmpeg_bin_dir}")
    except Exception as e:
        print(f"[!] Warning during dynamic FFmpeg path injection: {e}")

    parser = argparse.ArgumentParser(description="Mixing-AI Dataset Preparation CLI")
    parser.add_argument("--max-samples", type=int, default=100, help="Maximum number of dataset samples to generate")
    parser.add_argument("--output-dir", type=str, default="data/processed/dataset_v1", help="Path to store processed dataset")
    parser.add_argument("--soundcloud-urls", type=str, default="data/raw/soundcloud_urls.txt", help="Path to SoundCloud URLs text file")
    parser.add_argument("--vctk-dir", type=str, default=None, help="Path to local VCTK dataset folder")
    parser.add_argument("--vctk-limit", type=int, default=10, help="Maximum VCTK source files to slice (limits disk space usage)")
    parser.add_argument("--sc-ratio", type=float, default=0.5, help="Percentage ratio of SoundCloud vs VCTK vocals (0.0 to 1.0)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    urls_file = Path(args.soundcloud_urls)
    
    raw_dry_dir = Path("data/raw/dry_vocals")
    raw_dry_dir.mkdir(parents=True, exist_ok=True)
    
    collector = VocalCollector(sr=44100)

    # 1. Process local VCTK folder if provided
    vctk_dest = raw_dry_dir / "vctk"
    if args.vctk_dir:
        vctk_src_dir = Path(args.vctk_dir)
        if vctk_src_dir.exists():
            print(f"[*] Processing VCTK source files from: {vctk_src_dir}...")
            vctk_wavs = []
            for ext in ["*.wav", "*.flac"]:
                vctk_wavs.extend(list(vctk_src_dir.rglob(ext)))
                
            # Limit VCTK files to slice
            if len(vctk_wavs) > args.vctk_limit:
                import random
                vctk_wavs = random.sample(vctk_wavs, args.vctk_limit)
                
            print(f"[*] Slicing {len(vctk_wavs)} selected VCTK files...")
            vctk_dest.mkdir(parents=True, exist_ok=True)
            for f_path in vctk_wavs:
                collector.slice_audio(
                    audio_path=f_path,
                    output_dir=vctk_dest,
                    prefix=f"vctk_{f_path.stem}"
                )
        else:
            print(f"[-] VCTK directory not found: {args.vctk_dir}")
            
    # 2. Check if we have urls_file and download
    if urls_file.exists():
        print("[*] Running SoundCloud vocal collection pipeline...")
        download_dir = Path("data/raw/soundcloud_raw")
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
            
    # 3. Fallback option: if no dry vocals found anywhere, generate basic synthetic voice from project audio
    vctk_files = list(vctk_dest.glob("**/*.wav")) if vctk_dest.exists() else []
    sc_dest = raw_dry_dir / "soundcloud"
    sc_files = list(sc_dest.glob("**/*.wav")) if sc_dest.exists() else []
    
    if not vctk_files and not sc_files:
        print("[!] No dry vocal files found in data/raw/dry_vocals/.")
        print("[*] Generating synthetic seed voice from data/raw/source/source.wav...")
        
        src_wav = Path("data/raw/source/source.wav")
        if src_wav.exists():
            vctk_dest.mkdir(parents=True, exist_ok=True)
            collector.slice_audio(
                audio_path=src_wav,
                output_dir=vctk_dest,
                prefix="vctk_synth_seed"
            )
        else:
            print("[-] Error: Seed file data/raw/source/source.wav not found. Please add a wav file to start.")
            sys.exit(1)
            
    # 4. Generate dataset
    generator = DatasetGenerator(dry_dir=raw_dry_dir, output_dir=output_dir, sr=44100)
    generator.generate(max_samples=args.max_samples, sc_ratio=args.sc_ratio)

if __name__ == "__main__":
    main()
