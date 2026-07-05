import os
import sys
import subprocess
import shutil
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import List, Optional

# Prevent Windows console UnicodeEncodeError crashes when printing non-ASCII filenames
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(errors='replace')

class VocalCollector:
    """Handles downloading from SoundCloud, vocal extraction via Demucs, and audio slicing."""
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
        # Dynamic path injection for WinGet-installed FFmpeg (handles Windows shell restart issue)
        try:
            import os
            winget_base = Path(os.path.expanduser("~")) / "AppData/Local/Microsoft/WinGet/Packages"
            if winget_base.exists():
                ffmpeg_bins = list(winget_base.rglob("ffmpeg.exe"))
                if ffmpeg_bins:
                    ffmpeg_bin_dir = ffmpeg_bins[0].parent
                    if str(ffmpeg_bin_dir.absolute()) not in os.environ["PATH"]:
                        os.environ["PATH"] = str(ffmpeg_bin_dir.absolute()) + os.pathsep + os.environ["PATH"]
        except Exception:
            pass

    def download_soundcloud(self, urls_file: Path, download_dir: Path) -> List[Path]:
        """Downloads SoundCloud tracks listed in urls_file using yt-dlp."""
        download_dir.mkdir(parents=True, exist_ok=True)
        downloaded_files = []
        
        if not urls_file.exists():
            print(f"[-] SoundCloud URLs file not found at: {urls_file}")
            return downloaded_files

        with open(urls_file, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

        print(f"[*] Found {len(urls)} SoundCloud URLs to download.")
        archive_file = download_dir.parent / "soundcloud_download_archive.txt"
        for url in urls:
            print(f"[*] Downloading track/playlist: {url}...")
            try:
                # Use yt-dlp to download and convert to wav
                # Output filename format: download_dir/title.wav
                out_tmpl = str(download_dir / "%(title)s.%(ext)s")
                cmd = [
                    "yt-dlp",
                    "-x", "--audio-format", "wav",
                    "--yes-playlist",
                    "--ignore-errors",
                    "--download-archive", str(archive_file.absolute()),
                    "-o", out_tmpl,
                    url
                ]
                subprocess.run(cmd, check=True)
                print(f"[+] Successfully processed {url}")
            except Exception as e:
                print(f"[-] Failed to download {url}: {e}")

        # Gather all downloaded wav files
        for f_path in download_dir.glob("*.wav"):
            downloaded_files.append(f_path)
            
        return downloaded_files

    def separate_vocals(self, audio_path: Path, separation_dir: Path) -> Optional[Path]:
        """Extracts vocals from target audio_path using Demucs CLI."""
        separation_dir.mkdir(parents=True, exist_ok=True)
        
        song_name = audio_path.stem
        vocals_file = separation_dir / "htdemucs" / song_name / "vocals.wav"
        
        if vocals_file.exists():
            print(f"[+] Demucs separated vocals already exist at: {vocals_file}. Skipping separation.")
            return vocals_file
            
        print(f"[*] Running Demucs vocal separation on: {audio_path.name}...")
        
        try:
            # Command: demucs --two-stems=vocals -o <separation_dir> <audio_path>
            cmd = [
                "demucs",
                "--two-stems=vocals",
                "-o", str(separation_dir.absolute()),
                str(audio_path.absolute())
            ]
            subprocess.run(cmd, check=True)
            
            # Demucs places results in separation_dir/htdemucs/<song_name>/vocals.wav
            song_name = audio_path.stem
            vocals_file = separation_dir / "htdemucs" / song_name / "vocals.wav"
            
            if vocals_file.exists():
                print(f"[+] Demucs separated vocals saved to: {vocals_file}")
                return vocals_file
            else:
                print(f"[-] Could not find Demucs output at: {vocals_file}")
                return None
        except Exception as e:
            print(f"[-] Demucs vocal separation failed: {e}")
            return None

    def slice_audio(self, audio_path: Path, output_dir: Path, prefix: str, segment_sec: float = 8.0, min_rms: float = 0.015, min_vocal_ratio: float = 0.5) -> List[Path]:
        """Slices an audio file into fixed-length segments, skipping parts with low voice activity."""
        output_dir.mkdir(parents=True, exist_ok=True)
        sliced_paths = []
        
        try:
            y, sr = librosa.load(audio_path, sr=self.sr)
            seg_samples = int(segment_sec * sr)
            total_samples = len(y)
            num_segments = total_samples // seg_samples
            
            print(f"[*] Slicing {audio_path.name} into segments (Voice Activity filter: min_vocal_ratio={min_vocal_ratio})...")
            
            # Frame-level RMS parameters (50ms frames, 25ms overlap)
            frame_len = int(0.05 * sr)
            hop_len = int(0.025 * sr)
            
            # Handle short audio files by padding
            if total_samples < seg_samples:
                # Ensure we have at least one frame length of samples to avoid empty frame check
                if total_samples >= frame_len:
                    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop_len)
                    frame_rms = np.sqrt(np.mean(frames**2, axis=0))
                    vocal_frames = np.sum(frame_rms > min_rms)
                    vocal_ratio = vocal_frames / len(frame_rms)
                    
                    if vocal_ratio >= min_vocal_ratio:
                        padded = np.zeros(seg_samples, dtype=np.float32)
                        padded[:total_samples] = y
                        segment_path = output_dir / f"{prefix}_short.wav"
                        sf.write(segment_path, padded, sr)
                        sliced_paths.append(segment_path)
                return sliced_paths
                
            for i in range(num_segments):
                start = i * seg_samples
                end = start + seg_samples
                chunk = y[start:end]
                
                # Compute frame-level RMS
                frames = librosa.util.frame(chunk, frame_length=frame_len, hop_length=hop_len)
                frame_rms = np.sqrt(np.mean(frames**2, axis=0))
                
                # Check ratio of frames exceeding threshold
                vocal_frames = np.sum(frame_rms > min_rms)
                vocal_ratio = vocal_frames / len(frame_rms)
                
                if vocal_ratio < min_vocal_ratio:
                    continue  # skip segment due to insufficient vocal density
                
                segment_path = output_dir / f"{prefix}_{i:03d}.wav"
                sf.write(segment_path, chunk, sr)
                sliced_paths.append(segment_path)
                
            return sliced_paths
        except Exception as e:
            print(f"[-] Slicing failed for {audio_path}: {e}")
            return []
