"""
Бенчмарк подбора цепочки: optimizer (без сети) vs model+optimizer vs model (без подбора).

Пример:
    python scripts/benchmark_chain.py --n-synthetic 20 --budget 60 --seeds 0 1
    python scripts/benchmark_chain.py --n-synthetic 0 --manifest data/raw/benchmark/manifest.json

Результаты: <out>/results.jsonl (строка на пару/режим/seed) и <out>/summary.md (сводная таблица).
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Настройка кодировки консоли Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(errors='replace')

# Добавление корня проекта в пути импорта Python
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.eval.benchmark import MODES, ChainBenchmark, manifest_cases, summarize, synthetic_cases

SR = 44100


def main(args):
    out_dir = Path(args.out) if args.out else Path("data/runs/benchmark") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    prior = None
    modes = tuple(args.modes)
    if any(m.startswith("model") for m in modes):
        if Path(args.model_path).exists():
            from app.ml.pipeline import MixingAIMLPipeline
            prior = MixingAIMLPipeline(args.model_path).predict_prior
        else:
            print(f"[!] Веса модели не найдены ({args.model_path}), режимы с моделью пропускаются")

    cases = []
    if args.n_synthetic > 0:
        print(f"[*] Сборка {args.n_synthetic} синтетических пар из {args.dry_dir}...")
        cases += synthetic_cases(args.dry_dir, args.n_synthetic, sr=SR, seconds=args.seconds, seed=args.case_seed)
    if args.manifest:
        cases += manifest_cases(args.manifest, sr=SR, seconds=args.seconds)
    if not cases:
        print("[-] Нет пар для бенчмарка: задайте --n-synthetic и/или --manifest")
        sys.exit(1)

    bench = ChainBenchmark(sr=SR, prior=prior, with_pitch=not args.no_pitch)
    audio_dir = out_dir / "audio" if args.save_audio else None
    rows = []
    with open(out_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for case in cases:
            for i, seed in enumerate(args.seeds):
                for row in bench.run_case(case, modes, budget_s=args.budget, seed=seed, audio_dir=audio_dir,
                                          with_baselines=i == 0, patience_renders=args.patience):
                    rows.append(row)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()

    summary = summarize(rows)
    config = (f"Пар: {len(cases)}, seeds: {args.seeds}, бюджет CMA-ES: {args.budget:.0f} с, "
              f"ранняя остановка: {args.patience or 'нет'}, "
              f"фрагмент: {args.seconds:.0f} с, тюн в метрике: {not args.no_pitch}")
    (out_dir / "summary.md").write_text(f"# Бенчмарк цепочки\n\n{config}\n\n{summary}\n", encoding="utf-8")
    print(f"\n{config}\n\n{summary}\n\n[+] Результаты: {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Бенчмарк подбора цепочки сведения по референсу")
    parser.add_argument("--dry-dir", type=str, default="data/raw/dry_vocals", help="Сухие вокалы для синтетических пар")
    parser.add_argument("--n-synthetic", type=int, default=20, help="Число синтетических пар из разных песен")
    parser.add_argument("--manifest", type=str, default=None,
                        help='JSON с реальными парами: [{"name": ..., "source": ..., "ref": ...}]')
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES), help="Режимы для сравнения")
    parser.add_argument("--model-path", type=str, default="data/models/v3/best_model.pth", help="Веса MixingAIModel")
    parser.add_argument("--budget", type=float, default=60.0, help="Время на CMA-ES на одну пару, секунд")
    parser.add_argument("--patience", type=int, default=None,
                        help="Остановить CMA-ES, если за столько рендеров улучшение меньше 0.01")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="Seed CMA-ES (несколько — для разброса)")
    parser.add_argument("--case-seed", type=int, default=0, help="Seed выбора синтетических пар и цепочек")
    parser.add_argument("--seconds", type=float, default=30.0, help="Длина фрагмента исходника и референса")
    parser.add_argument("--no-pitch", action="store_true", help="Не учитывать тюн в метрике (быстрее)")
    parser.add_argument("--save-audio", action="store_true", help="Сохранять рендеры каждого режима")
    parser.add_argument("--out", type=str, default=None, help="Каталог результатов (по умолчанию data/runs/benchmark/<время>)")
    main(parser.parse_args())
