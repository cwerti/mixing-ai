import argparse
import sys
from pathlib import Path
import soundfile as sf
import numpy as np

# Предотвращение сбоев кодировки вывода в консоль Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(errors='replace')

# Добавление корневой директории в пути импорта Python
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.audio.vocal_collector import VocalCollector
from app.dataset.dataset_generator import DatasetGenerator

def main():
    """
    Основная оркестрация CLI-скрипта сборки сырых данных
    и генерации обучающего датасета.
    """
    # Добавление пути FFmpeg из WinGet в PATH
    try:
        import os
        winget_base = Path(os.path.expanduser("~")) / "AppData/Local/Microsoft/WinGet/Packages"
        if winget_base.exists():
            ffmpeg_bins = list(winget_base.rglob("ffmpeg.exe"))
            if ffmpeg_bins:
                ffmpeg_bin_dir = ffmpeg_bins[0].parent
                os.environ["PATH"] = str(ffmpeg_bin_dir.absolute()) + os.pathsep + os.environ["PATH"]
                print(f"[+] FFmpeg добавлен в PATH: {ffmpeg_bin_dir}")
    except Exception as e:
        print(f"[!] Предупреждение при поиске FFmpeg: {e}")

    parser = argparse.ArgumentParser(description="Скрипт подготовки датасета Mixing-AI")
    parser.add_argument("--max-samples", type=int, default=100, help="Максимальное количество сэмплов датасета")
    parser.add_argument("--output-dir", type=str, default="data/processed/dataset_v1", help="Папка для сохранения датасета")
    parser.add_argument("--soundcloud-urls", type=str, default="data/raw/soundcloud_urls.txt", help="Файл со ссылками SoundCloud")
    parser.add_argument("--vctk-dir", type=str, default=None, help="Путь к локальной папке датасета VCTK")
    parser.add_argument("--vctk-limit", type=int, default=10, help="Лимит нарезаемых файлов VCTK для экономии места")
    parser.add_argument("--sc-ratio", type=float, default=0.5, help="Пропорция вокала SoundCloud по отношению к VCTK (0.0 - 1.0)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    urls_file = Path(args.soundcloud_urls)
    
    raw_dry_dir = Path("data/raw/dry_vocals")
    raw_dry_dir.mkdir(parents=True, exist_ok=True)
    
    collector = VocalCollector(sr=44100)

    # 1. Обработка локальных файлов VCTK (если указана папка)
    vctk_dest = raw_dry_dir / "vctk"
    if args.vctk_dir:
        vctk_src_dir = Path(args.vctk_dir)
        if vst_src_dir := vctk_src_dir.exists():
            print(f"[*] Обработка исходных файлов VCTK из: {vctk_src_dir}...")
            vctk_wavs = []
            for ext in ["*.wav", "*.flac"]:
                candidates = list(vctk_src_dir.rglob(ext))
                for f in candidates:
                    f_name_lower = f.name.lower()
                    parents_lower = [p.name.lower() for p in f.parents]
                    
                    # Разрешаем вокальные ключевые слова или файлы из папки vctk
                    is_vocal = (
                        any(kw in f_name_lower for kw in ["vocal", "vox", "mic", "sing", "lead", "backing", "voice", "dry"])
                        or "vctk" in parents_lower
                    )
                    
                    # Исключаем инструменты
                    is_instrument = any(kw in f_name_lower for kw in [
                        "kick", "snare", "drum", "guitar", "bass", "synth", "piano", 
                        "loop", "beat", "instrumental", "organs", "strings", "brass",
                        "horns", "perc", "ride", "cymbal", "hihat"
                    ])
                    
                    if is_vocal and not is_instrument:
                        vctk_wavs.append(f)
                
            # Выборка случайного подмножества для ограничения дискового пространства
            if len(vctk_wavs) > args.vctk_limit:
                import random
                vctk_wavs = random.sample(vctk_wavs, args.vctk_limit)
                
            print(f"[*] Нарезка {len(vctk_wavs)} выбранных файлов VCTK...")
            vctk_dest.mkdir(parents=True, exist_ok=True)
            for f_path in vctk_wavs:
                collector.slice_audio(
                    audio_path=f_path,
                    output_dir=vctk_dest,
                    prefix=f"vctk_{f_path.stem}"
                )
        else:
            print(f"[-] Директория VCTK не найдена: {args.vctk_dir}")
            
    # 2. Скачивание и обработка с SoundCloud
    if urls_file.exists():
        print("[*] Запуск сбора вокала с SoundCloud...")
        download_dir = Path("data/raw/soundcloud_raw")
        downloaded = collector.download_soundcloud(urls_file, download_dir)
        
        # Выделение вокала через Demucs и последующая нарезка
        for f_path in downloaded:
            separated = collector.separate_vocals(f_path, Path("data/raw/soundcloud_separated"))
            if separated:
                collector.slice_audio(
                    audio_path=separated,
                    output_dir=raw_dry_dir / "soundcloud",
                    prefix=f"sc_{f_path.stem}"
                )
            
    # 3. Резервный вариант (Seed): если сухой вокал не найден, нарезаем тестовый файл source.wav
    vctk_files = list(vctk_dest.glob("**/*.wav")) if vctk_dest.exists() else []
    sc_dest = raw_dry_dir / "soundcloud"
    sc_files = list(sc_dest.glob("**/*.wav")) if sc_dest.exists() else []
    
    if not vctk_files and not sc_files:
        print("[!] В папке data/raw/dry_vocals/ отсутствует сухой вокал.")
        print("[*] Генерация базовых сегментов из файла data/raw/source/source.wav...")
        
        src_wav = Path("data/raw/source/source.wav")
        if src_wav.exists():
            vctk_dest.mkdir(parents=True, exist_ok=True)
            collector.slice_audio(
                audio_path=src_wav,
                output_dir=vctk_dest,
                prefix="vctk_synth_seed"
            )
        else:
            print("[-] Ошибка: Файл-источник data/raw/source/source.wav не найден. Пожалуйста, добавьте тестовый аудиофайл.")
            sys.exit(1)
            
    # 4. Запуск генерации итогового датасета
    generator = DatasetGenerator(dry_dir=raw_dry_dir, output_dir=output_dir, sr=44100)
    generator.generate(max_samples=args.max_samples, sc_ratio=args.sc_ratio)

if __name__ == "__main__":
    main()
