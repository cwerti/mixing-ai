"""
Проверка языка отпечатков: совпадает ли близость отпечатков (app/eval/effect_fingerprint.py) с близостью на слух.

Каждая строка jsonl из scripts/effect_fingerprint.py (поле source: VST3 + настройки или наш DSP) рендерится на
сухом вокале. Для всех пар считается расстояние отпечатков и log-mel между рендерами (дБ, симметрично, по
каналам, пол -80 дБ от максимума). Печатаются:
- Spearman по парам;
- ранг ближайшего по отпечатку среди соседей по звуку (0 — совпал с ближайшим по звуку);
- промахи — пары, близкие по отпечатку и далекие по звуку.

Примеры:
    python scripts/fingerprint_vs_audio.py data/runs/fingerprint/supermassive_default.jsonl
    python scripts/fingerprint_vs_audio.py data/runs/fingerprint/supermassive_*.jsonl --combined
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.eval.effect_fingerprint import fingerprint_distance  # noqa: E402

DEFAULT_DRY = Path("data/processed/real_pairs_v10/whiptails_vox/dry.wav")


def load_rows(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        stem = Path(path).stem
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["label"] = f"{stem}/{row['name'].split(':')[-1]}"
                rows.append(row)
    return rows


def render_row(row: dict, y: np.ndarray, sr: int) -> np.ndarray:
    """Рендер по полю source; VST — на свежем экземпляре (reset не сбрасывает состояние части плагинов)."""
    source = row.get("source")
    if not source:
        raise ValueError(f"{row['name']}: нет поля source — перегенерируйте jsonl scripts/effect_fingerprint.py")
    if "vst" in source:
        import pedalboard
        plugin = pedalboard.load_plugin(source["vst"])
        for key, value in source["settings"].items():
            setattr(plugin, key, value)
        return np.asarray(plugin.process(y, sr, reset=True), dtype=np.float32)
    from scripts.effect_fingerprint import dsp_effect
    return np.asarray(dsp_effect(source["dsp"], source["params"])(y, sr), dtype=np.float32)


def log_mel(y: np.ndarray, sr: int) -> list[np.ndarray]:
    return [librosa.power_to_db(librosa.feature.melspectrogram(y=c, sr=sr, n_fft=2048, hop_length=512, n_mels=80),
                                amin=1e-10) for c in y]


def logmel_distance(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    """Симметричная ошибка log-mel (дБ): пол -80 дБ от максимума каждого из двух рендеров по очереди."""
    def one(x, ref):
        floor = ref.max() - 80.0
        return float(np.mean(np.abs(np.maximum(x, floor) - np.maximum(ref, floor))))
    return float(np.mean([0.5 * (one(x, z) + one(z, x)) for x, z in zip(a, b)]))


def evaluate(rows: list[dict], mels: dict[str, list[np.ndarray]], title: str, misses: int = 5) -> dict:
    n = len(rows)
    d_fp = np.full((n, n), np.inf)
    d_au = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            d_fp[i, j] = d_fp[j, i] = fingerprint_distance(rows[i]["fingerprint"], rows[j]["fingerprint"])
            d_au[i, j] = d_au[j, i] = logmel_distance(mels[rows[i]["label"]], mels[rows[j]["label"]])
    iu = np.triu_indices(n, 1)
    rho = float(spearmanr(d_fp[iu], d_au[iu])[0])
    ranks = [int((d_au[i] < d_au[i, int(np.argmin(d_fp[i]))]).sum()) for i in range(n)]
    print(f"\n== {title}: {n} эффектов, {len(iu[0])} пар")
    print(f"Spearman(отпечаток, log-mel) = {rho:.3f}")
    print(f"log-mel между эффектами: мин {d_au[iu].min():.2f}, медиана {np.median(d_au[iu]):.2f}, "
          f"макс {d_au[iu].max():.2f} дБ")
    print(f"ранг ближайшего по отпечатку среди соседей по звуку: среднее {np.mean(ranks):.2f}, "
          f"совпал {sum(r == 0 for r in ranks)}/{n}")
    # Промах: по отпечатку в нижней четверти пар, а по звуку — дальше медианы ближайших соседей по звуку
    fp_q = np.quantile(d_fp[iu], 0.25)
    nn_audio = np.median(d_au.min(axis=1))
    bad = [(d_au[i, j], d_fp[i, j], rows[i]["label"], rows[j]["label"]) for i, j in zip(*iu)
           if d_fp[i, j] <= fp_q and d_au[i, j] > 2.0 * nn_audio]
    bad.sort(key=lambda t: t[0] - 3.0 * t[1], reverse=True)
    print(f"промахи (отпечаток ≤ {fp_q:.2f}, звук > {2.0 * nn_audio:.2f} дБ): {len(bad)}")
    for au, fp, a, b in bad[:misses]:
        print(f"  отпечаток {fp:4.2f}  звук {au:5.2f} дБ  {a} / {b}")
    return {"title": title, "n": n, "spearman": rho, "mean_rank": float(np.mean(ranks)),
            "rank0": int(sum(r == 0 for r in ranks)), "misses": len(bad)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jsonl", nargs="+")
    parser.add_argument("--dry", default=str(DEFAULT_DRY))
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--combined", action="store_true", help="плюс оценка по всем файлам вместе")
    parser.add_argument("--out", default=None, help="json со сводкой")
    args = parser.parse_args()

    y, sr = sf.read(args.dry, dtype="float32", always_2d=True)
    y = y.T[:, :int(args.seconds * sr)]
    if y.shape[0] == 1:
        y = np.vstack([y, y])

    groups = {Path(p).stem: load_rows([p]) for p in args.jsonl}
    mels: dict[str, list[np.ndarray]] = {}
    for rows in groups.values():
        for row in rows:
            mels[row["label"]] = log_mel(render_row(row, y, sr), sr)
            print(f"[+] {row['label']}", flush=True)

    summary = [evaluate(rows, mels, name) for name, rows in groups.items()]
    if args.combined and len(groups) > 1:
        summary.append(evaluate([r for rows in groups.values() for r in rows], mels, "все вместе"))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
