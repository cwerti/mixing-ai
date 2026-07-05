import os
import sys
import urllib.request
from pathlib import Path
from tqdm import tqdm

def download_vctk_subset(output_dir: Path, num_files_per_speaker: int = 50):
    """Downloads a tiny subset of VCTK dataset (speakers p225 and p226) from Hugging Face mirror."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    speakers = ["p225", "p226"]
    print(f"[*] Downloading a tiny VCTK subset ({num_files_per_speaker} files for speakers {', '.join(speakers)}) from Hugging Face...")
    
    downloaded_count = 0
    for speaker in speakers:
        speaker_dir = output_dir / speaker
        speaker_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[*] Downloading files for speaker {speaker}...")
        for i in tqdm(range(1, num_files_per_speaker + 1)):
            file_id = f"{i:03d}"
            filename = f"{speaker}_{file_id}.wav"
            url = f"https://huggingface.co/datasets/vctk/resolve/main/wav48/{speaker}/{filename}"
            dest_path = speaker_dir / filename
            
            # Skip if already downloaded
            if dest_path.exists():
                downloaded_count += 1
                continue
                
            try:
                # Download file
                urllib.request.urlretrieve(url, dest_path)
                downloaded_count += 1
            except Exception as e:
                # Some file IDs might not exist, skip them silently
                continue

    print(f"[+] Download complete! Successfully prepared {downloaded_count} VCTK files at: {output_dir}")

def main():
    dest_dir = Path("data/raw/vctk_subset")
    download_vctk_subset(dest_dir, num_files_per_speaker=50)
    print("\n[+] To use this subset, run dataset generation with:")
    print("poetry run python scripts/prepare_dataset.py --vctk-dir data/raw/vctk_subset --max-samples 100")

if __name__ == "__main__":
    main()
