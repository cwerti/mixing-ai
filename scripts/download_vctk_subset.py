import os
import sys
from pathlib import Path
from tqdm import tqdm

# Prevent console encoding crashes
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')

def download_vctk_subset(output_dir: Path, target_samples: int = 100):
    """Downloads VCTK samples directly using Hugging Face datasets streaming mode with decoding disabled."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # We import datasets here to ensure it only loads when function is called
    from datasets import load_dataset, Audio
    
    print("[*] Loading VCTK dataset in streaming mode from Hugging Face (kth-tmh/vctk)...")
    try:
        ds = load_dataset("kth-tmh/vctk", name="default", split="train", streaming=True)
        # Disable audio decoding to bypass torchcodec DLL loading errors on Windows
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception as e:
        print(f"[-] Failed to load dataset: {e}")
        return

    print(f"[*] Streaming and saving first {target_samples} samples as FLAC files for speakers p225 and p226...")
    
    count = 0
    progress_bar = tqdm(total=target_samples)
    
    for item in ds:
        speaker_id = item.get("speaker_id")
        text_id = item.get("text_id")
        audio = item.get("audio")
        
        # Limit to speakers p225 and p226
        if speaker_id not in ["p225", "p226"]:
            continue
            
        speaker_dir = output_dir / speaker_id
        speaker_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw flac bytes directly to disk
        filename = f"{speaker_id}_{text_id}.flac"
        dest_path = speaker_dir / filename
        
        if not dest_path.exists():
            try:
                with open(dest_path, "wb") as f_out:
                    f_out.write(audio["bytes"])
            except Exception as e:
                print(f"\n[-] Failed to write {dest_path.name}: {e}")
                continue
                
        count += 1
        progress_bar.update(1)
        
        if count >= target_samples:
            break
            
    progress_bar.close()
    print(f"[+] VCTK subset download complete. Successfully downloaded {count} files to: {output_dir}")

def main():
    dest_dir = Path("data/raw/vctk_subset")
    download_vctk_subset(dest_dir, target_samples=100)
    print("\n[+] To use this subset, run dataset generation with:")
    print("poetry run python scripts/prepare_dataset.py --vctk-dir data/raw/vctk_subset --max-samples 100")

if __name__ == "__main__":
    main()
