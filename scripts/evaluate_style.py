import argparse
import sys
from pathlib import Path

import librosa
import numpy as np

# Настройка кодировки консоли Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(errors='replace')

# Добавление корня проекта в пути импорта Python
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.eval.style_metrics import DEFAULT_WEIGHTS, LTAS_CENTERS_HZ, extract_style_features, style_distance

SR = 44100

ROWS = [
    ("crest_db", "Пик-фактор, дБ"),
    ("dr_p95_p10_db", "Разброс громкости фраз, дБ"),
    ("dr_p99_p90_db", "Пики внутри фраз, дБ"),
    ("decay_fast_db", "Самый быстрый спад, дБ/46мс"),
    ("reverb_tail_db", "Хвост после фраз, дБ"),
    ("delay_peak", "Повторы дилея (автокорр.)"),
    ("width_global", "Ширина (side/mid)"),
    ("width_high", "Ширина выше 150 Гц"),
    ("sib_p90_db", "Сибилянты P90, дБ"),
    ("air_ratio_db", "Воздух 10-18 кГц, дБ"),
    ("pitch_cents", "Отклонение от нот, центы"),
]


def load(path: Path) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=False)
    return y


def evaluate(args):
    """Печатает таблицу признаков стиля и расстояние до референса для исходника и каждого результата."""
    files = {"source": Path(args.source), "ref": Path(args.ref)}
    for i, out in enumerate(args.outputs, start=1):
        files[f"out{i}"] = Path(out)
        print(f"  out{i} = {out}")

    features = {}
    for name, path in files.items():
        print(f"[*] Анализ {name}: {path}")
        features[name] = extract_style_features(load(path), SR, with_pitch=not args.no_pitch)

    names = list(files)
    width = 14
    print("\n" + "Признак".ljust(30) + "".join(n.rjust(width) for n in names))
    for key, title in ROWS:
        if key not in features["ref"]:
            continue
        print(title.ljust(30) + "".join(f"{features[n][key]:{width}.2f}" for n in names))

    print("\nДолгосрочный спектр, дБ относительно референса (полосы с разницей > 2 дБ):")
    for i, fc in enumerate(LTAS_CENTERS_HZ):
        diffs = [features[n]["ltas_db"][i] - features["ref"]["ltas_db"][i] for n in names if n != "ref"]
        if max(abs(d) for d in diffs) > 2.0:
            print(f"  {fc:7.0f} Гц".ljust(30) + "".join(f"{d:{width}.1f}" for d in diffs))

    print("\nРасстояние до референса (меньше = ближе; 1.0 = заметная на слух разница по каждому признаку):")
    for name in names:
        if name == "ref":
            continue
        total, per_feature = style_distance(features[name], features["ref"], DEFAULT_WEIGHTS)
        worst = sorted(per_feature.items(), key=lambda kv: -abs(kv[1]))[:3]
        worst_str = ", ".join(f"{k} {v:+.1f}" for k, v in worst)
        print(f"  {name:18s} {total:6.2f}   хуже всего: {worst_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сравнение стиля сведения результатов с референсом")
    parser.add_argument("--source", type=str, required=True, help="Сухой вокал")
    parser.add_argument("--ref", type=str, required=True, help="Референс (вокал)")
    parser.add_argument("--outputs", type=str, nargs="*", default=[], help="Результаты обработки для сравнения")
    parser.add_argument("--no-pitch", action="store_true", help="Не считать отклонение от нот (быстрее)")
    evaluate(parser.parse_args())
