import argparse
import sys
from pathlib import Path

# Настройка кодировки консоли Windows для предотвращения сбоев вывода
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')

# Добавление корня проекта в пути импорта Python
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.audio.vocal_collector import VocalCollector
from app.dataset.dataset_generator import DatasetGenerator


def main():
    parser = argparse.ArgumentParser(description="Генератор датасета вокала для Mixing-AI (на базе Cambridge MT / VCTK)")
    parser.add_argument("--max-samples", type=int, default=5000, help="Максимальное количество генерируемых сэмплов")
    parser.add_argument("--output-dir", type=str, default="data/processed/dataset_v6", help="Папка сохранения датасета")
    parser.add_argument("--vctk-dir", type=str, default=None, help="Путь к локальной папке датасета VCTK (опционально)")
    parser.add_argument("--vctk-limit", type=int, default=10, help="Лимит нарезаемых файлов VCTK")
    parser.add_argument("--cross-ratio", type=float, default=1.0,
                        help="Доля пар, где референс — другой голос другой песни (1.0 = как в реальной задаче)")
    parser.add_argument("--seed", type=int, default=0, help="Seed выбора файлов, пар и цепочек")
    parser.add_argument("--save-all-wav", action="store_true", help="Сохранять WAV для всех сэмплов (по умолчанию только первые 30)")
    parser.add_argument("--prior", choices=["realistic", "balanced"], default="realistic",
                        help="Распределение цепочек: realistic (dataset_v4+) или balanced (как dataset_v3)")
    parser.add_argument("--harmony-ratio", type=float, default=0.35,
                        help="Доля референсов с даблами/гармониями (только рендер, без метки)")
    parser.add_argument("--include-test-songs", action="store_true",
                        help="Не исключать песни реальных тестовых пар (поведение до dataset_v4)")
    parser.add_argument("--workers", type=int, default=None, help="Число процессов (по умолчанию все ядра)")
    parser.add_argument("--resume", action="store_true",
                        help="Дописать прерванную генерацию (сэмплы из index.jsonl пропускаются)")
    parser.add_argument("--domain-rand", type=float, default=None, metavar="OWN_PROB",
                        help="Доменная рандомизация рендера референса (dataset_v5): доля плагинов нашей "
                             "реализации, например 0.35 (app/audio/effect_variants.py). Без флага — как раньше")

    args = parser.parse_args()

    raw_dry_dir = Path("data/raw/dry_vocals")
    output_dir = Path(args.output_dir)

    raw_dry_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    collector = VocalCollector(sr=44100)

    # 1. Опциональная обработка локального корпуса VCTK
    if args.vctk_dir:
        vctk_path = Path(args.vctk_dir)
        if vctk_path.exists():
            print(f"[*] Нарезка локальных файлов VCTK из {vctk_path}...")
            vctk_files = list(vctk_path.glob("**/*.wav")) + list(vctk_path.glob("**/*.flac"))
            for vf in vctk_files[:args.vctk_limit]:
                collector.slice_audio(vf, raw_dry_dir / "vctk", prefix=f"vctk_{vf.stem}", target_sec=7.0)
        else:
            print(f"[-] Директория VCTK не найдена: {args.vctk_dir}")

    # 2. Проверка наличия студийного сухого вокала (Cambridge MT / VCTK)
    cambridge_dest = raw_dry_dir / "cambridge"
    cambridge_files = list(cambridge_dest.glob("**/*.wav")) if cambridge_dest.exists() else []
    vctk_dest = raw_dry_dir / "vctk"
    vctk_files = list(vctk_dest.glob("**/*.wav")) if vctk_dest.exists() else []

    if not cambridge_files and not vctk_files:
        print("[!] В папке data/raw/dry_vocals/ отсутствует сухой вокал Cambridge/VCTK.")
        source_fallback = Path("data/raw/source/source.wav")
        if source_fallback.exists():
            print("[*] Генерация базовых сегментов из файла data/raw/source/source.wav...")
            collector.slice_audio(source_fallback, vctk_dest, prefix="local_source", target_sec=6.0)
        else:
            print("[-] Исходный файл data/raw/source/source.wav также не найден.")
            sys.exit(1)

    # 3. Запуск генерации итогового датасета (без использования SoundCloud)
    generator = DatasetGenerator(dry_dir=raw_dry_dir, output_dir=output_dir, sr=44100)
    generator.generate(max_samples=args.max_samples, save_all_wav=args.save_all_wav,
                       cross_ratio=args.cross_ratio, seed=args.seed, prior=args.prior,
                       harmony_ratio=args.harmony_ratio, exclude_test_songs=not args.include_test_songs,
                       domain_rand_own_prob=args.domain_rand, workers=args.workers, resume=args.resume)


if __name__ == "__main__":
    main()
