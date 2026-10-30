"""
Проверка метрик стиля на реальных парах dry/wet (scripts/real_pairs.py) и на синтетике.

Метрики: style (ручная style_distance), embedding (обученный кодировщик, 1 - cos), combined (их сумма в
единицах style_distance, app/eval/style_embedding.combined_distance), afx (AFx-Rep, если передан --afx) и
emb+afx (среднее косинусных расстояний кодировщика и AFx-Rep). Косинусные расстояния умножены на EMBEDDING_SCALE.

Что считается:
  (a) доля пар, где d(wet A, ref B) < d(dry A, ref B) — истинный результат ближе к референсу из другой части
      песни, чем сухой исходник;
  (b) отношение «пола» d(wet A, wet B) к разрыву обработки d(dry A, wet A) (меньше 1 — метрика видит
      обработку сильнее, чем смену содержания);
  (c) попарное согласие и τ Кендалла с прямой ошибкой log-mel по выходам режимов из
      data/runs/benchmark/real_pairs.jsonl (dry/model/optimizer/model+opt, отдельно — вместе с wet_true);
  (d) синтетика на отложенных песнях: та же цепочка на другом фрагменте ближе, чем другая цепочка
      на том же фрагменте (доля троек).

Пример:
    python scripts/eval_style_metric.py --out data/runs/style_encoder/eval.json
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
sys.path.append(str(Path(__file__).parent.parent.absolute()))
sys.path.append(str(Path(__file__).parent.absolute()))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

from app.eval.style_embedding import EMBEDDING_SCALE, StyleEncoder, combined_distance  # noqa: E402
from app.eval.style_metrics import extract_style_features, style_distance  # noqa: E402

SR = 44100
MODE_FILES = {"source": "source", "model": "model", "optimizer": "optimizer", "model+optimizer": "model_optimizer"}


class Metrics:
    """Все метрики сразу; признаки и эмбеддинги кэшируются по ключу сигнала."""

    def __init__(self, encoder: StyleEncoder, afx=None, with_pitch: bool = True, style_cache: Path | None = None):
        self.encoder, self.afx, self.with_pitch = encoder, afx, with_pitch
        self.cache: dict[str, dict] = {}
        # Признаки style_distance (pyin медленный) не зависят от кодировщика и кэшируются на диске
        self.style_cache_path = style_cache
        self.style_cache: dict[str, dict] = {}
        if style_cache is not None and style_cache.exists():
            import pickle
            self.style_cache = pickle.loads(style_cache.read_bytes())

    def save_style_cache(self) -> None:
        if self.style_cache_path is not None:
            import pickle
            self.style_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.style_cache_path.write_bytes(pickle.dumps(self.style_cache))

    def describe(self, key: str, y: np.ndarray) -> dict:
        if key not in self.cache:
            if key not in self.style_cache:
                self.style_cache[key] = extract_style_features(y, SR, with_pitch=self.with_pitch)
            d = {"style": self.style_cache[key], "emb": self.encoder.embed(y, SR)}
            if self.afx is not None:
                d["afx"] = self.afx.embed(y, SR)
            self.cache[key] = d
        return self.cache[key]

    def distances(self, a: dict, b: dict) -> dict:
        style = style_distance(a["style"], b["style"])[0]
        emb = StyleEncoder.distance(a["emb"], b["emb"])
        out = {"style": style, "embedding": EMBEDDING_SCALE * emb, "combined": combined_distance(style, emb)}
        if "afx" in a:
            afx = StyleEncoder.distance(a["afx"], b["afx"])
            out["afx"] = EMBEDDING_SCALE * afx
            # Кодировщик слышит тюн и стерео, AFx-Rep — спектр: среднее двух косинусных расстояний
            out["emb+afx"] = EMBEDDING_SCALE * 0.5 * (emb + afx)
        return out


def ms_logmel_db(x: np.ndarray, target: np.ndarray) -> float:
    """
    Прямая ошибка log-mel по Mid и Side (дБ) после выравнивания громкости и задержки. В отличие от logmel_db по
    L/R, видит появление или пропажу стерео (у моно-цели Side пуст, а лишняя ширина дает ошибку в Side).
    """
    import librosa
    from real_pairs import align, find_lag
    from app.mixing.chain_optimizer import CHAIN_INPUT_LUFS, normalize_lufs, to_stereo
    x = normalize_lufs(to_stereo(x), SR, CHAIN_INPUT_LUFS)
    target = normalize_lufs(to_stereo(target), SR, CHAIN_INPUT_LUFS)
    lag, _, _ = find_lag(target, x, max_lag_s=0.1)
    target, x = align(target, x, lag)

    def ms(y):
        mid, side = 0.5 * (y[0] + y[1]), 0.5 * (y[0] - y[1])
        return [librosa.power_to_db(librosa.feature.melspectrogram(y=c, sr=SR, n_fft=2048, hop_length=512, n_mels=80),
                                    ref=1.0, amin=1e-10) for c in (mid, side)]
    (xm, xs), (tm, ts) = ms(x), ms(target)
    floor = tm.max() - 80.0  # общий пол от Mid цели: пустой Side цели не дает бесконечной ошибки
    return float(np.mean([np.mean(np.abs(np.maximum(a, floor) - np.maximum(b, floor))) for a, b in ((xm, tm), (xs, ts))]))


def real_pairs_eval(metrics: Metrics, jsonl: Path, audio_dir: Path) -> dict:
    from real_pairs import PAIRS, build_cases
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    logmel = {(r["case"], r["mode"]): r["logmel_db"] for r in rows}
    f0 = {(r["case"], r["mode"]): r["f0_err_cents"] for r in rows}
    per_pair, ranking = {}, {}
    for name, dry_name, wet_name in PAIRS:
        cases = build_cases(name, dry_name, wet_name)
        (same_case, target), (cross_case, _) = cases
        dry = metrics.describe(f"{name}/dry", same_case.src)
        wet_a = metrics.describe(f"{name}/wetA", target)
        wet_b = metrics.describe(f"{name}/wetB", cross_case.ref)
        floor, dry_ref, gap = (metrics.distances(wet_a, wet_b), metrics.distances(dry, wet_b),
                               metrics.distances(dry, wet_a))
        per_pair[name] = {"floor_wetA_wetB": floor, "dry_vs_wetB": dry_ref, "gap_dry_wetA": gap}
        print(f"[*] {name}: " + " | ".join(f"{m}: пол {floor[m]:.3f}, dry→B {dry_ref[m]:.3f}, dry↔A {gap[m]:.3f}"
                                          for m in floor), flush=True)
        for case in (same_case, cross_case):
            ref = metrics.describe(f"{case.name}/ref", case.ref)
            values = {}
            for mode, file_mode in MODE_FILES.items():
                path = audio_dir / case.name / f"{case.name}_{file_mode}_s0.wav"
                if path.exists() and (case.name, mode) in logmel:
                    y = sf.read(path)[0].T
                    direct = {"logmel": logmel[(case.name, mode)], "f0": f0[(case.name, mode)],
                              "mslogmel": ms_logmel_db(y, target)}
                    values[mode] = (metrics.distances(metrics.describe(str(path), y), ref), direct, y)
            values["wet_true"] = (metrics.distances(wet_a, ref), {"logmel": 0.0, "f0": 0.0, "mslogmel": 0.0}, target)
            ranking[case.name] = values
    return {"per_pair": per_pair, "ranking": ranking}


def summarize_real(result: dict, metric_names: list[str]) -> dict:
    per_pair = result["per_pair"]
    out = {}
    for m in metric_names:
        a = [p["floor_wetA_wetB"][m] < p["dry_vs_wetB"][m] for p in per_pair.values()]
        b = [p["floor_wetA_wetB"][m] / max(p["gap_dry_wetA"][m], 1e-9) for p in per_pair.values()]
        out[m] = {"a_wet_closer_than_dry": f"{sum(a)}/{len(a)}", "a_frac": float(np.mean(a)),
                  "b_floor_over_gap_median": float(np.median(b)), "b_floor_over_gap": [round(v, 2) for v in b]}
        for variant, with_true, direct_key in [(v, t, d) for d in ("logmel", "mslogmel", "f0")
                                               for v in ("same", "cross") for t in (False, True)]:
            tie = 0.5 if direct_key == "f0" else 0.05
            agree, total, taus = 0, 0, []
            for case, values in result["ranking"].items():
                if not case.endswith(variant):
                    continue
                modes = [k for k in values if with_true or k != "wet_true"]
                modes = [k for k in modes if np.isfinite(values[k][1][direct_key])]
                md = [values[k][0][m] for k in modes]
                lm = [values[k][1][direct_key] for k in modes]
                for i, j in combinations(range(len(modes)), 2):
                    same_audio = (values[modes[i]][2].shape == values[modes[j]][2].shape
                                  and np.allclose(values[modes[i]][2], values[modes[j]][2]))
                    if abs(lm[i] - lm[j]) < tie or same_audio:
                        continue
                    total += 1
                    agree += (md[i] - md[j]) * (lm[i] - lm[j]) > 0
                tau = kendalltau(md, lm).statistic if len(modes) >= 2 else float("nan")
                if np.isfinite(tau):
                    taus.append(tau)
            key = f"c_{direct_key}_{variant}" + ("_with_true" if with_true else "")
            out[m][key] = {"agree": f"{agree}/{total}", "agree_frac": agree / max(total, 1),
                           "tau_mean": float(np.mean(taus)) if taus else float("nan")}
    return out


def synthetic_eval(metrics: Metrics, n_blocks: int, workers: int) -> dict:
    """Тройки на отложенных песнях: d(та же цепочка, другой фрагмент) < d(другая цепочка, тот же фрагмент)."""
    from functools import partial

    from app.eval.style_chains import content_pool
    from train_style_encoder import CHAINS, CONTENTS, make_block
    _, val = content_pool()
    files = [str(f) for f in val]
    tasks = [(2_000_000 + i, files) for i in range(n_blocks)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        blocks = [b for b in pool.map(partial(make_block, keep_audio=True), tasks) if b is not None]
    wins: dict[str, int] = {}
    total = 0
    used = 0
    for bi, block in enumerate(blocks):
        try:
            desc = [[metrics.describe(f"syn{bi}/{i}/{k}", block["audio"][i][k]) for k in range(CONTENTS)]
                    for i in range(CHAINS)]
        except ValueError:  # тихий фрагмент: ручная метрика его не анализирует
            continue
        used += 1
        for i in range(CHAINS):
            for k in range(CONTENTS):
                for l in range(CONTENTS):
                    if l == k:
                        continue
                    pos = metrics.distances(desc[i][k], desc[i][l])
                    for j in range(CHAINS):
                        if j == i:
                            continue
                        neg = metrics.distances(desc[i][k], desc[j][k])
                        total += 1
                        for m in pos:
                            wins[m] = wins.get(m, 0) + (pos[m] < neg[m])
    return {m: w / max(total, 1) for m, w in wins.items()} | {"triplets": total, "blocks": used}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Проверка метрик стиля")
    parser.add_argument("--encoder", default="data/models/style_encoder/style_encoder.pt")
    parser.add_argument("--afx", default=None, help="Чекпоинт AFx-Rep (scripts/afx_rep.py), если скачан")
    parser.add_argument("--jsonl", default="data/runs/benchmark/real_pairs.jsonl")
    parser.add_argument("--audio-dir", default="data/runs/benchmark/real_pairs_audio")
    parser.add_argument("--synthetic-blocks", type=int, default=24)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", default="data/runs/style_encoder/eval.json")
    parser.add_argument("--style-cache", default="data/runs/style_encoder/style_features_cache.pkl")
    parser.add_argument("--no-synthetic", action="store_true")
    args = parser.parse_args()

    afx = None
    if args.afx:
        from afx_rep import AFxRep
        afx = AFxRep(args.afx)
    metrics = Metrics(StyleEncoder(args.encoder), afx, style_cache=Path(args.style_cache))
    names = ["style", "embedding", "combined"] + (["afx", "emb+afx"] if afx else [])
    real = real_pairs_eval(metrics, Path(args.jsonl), Path(args.audio_dir))
    metrics.save_style_cache()
    summary = {"real": summarize_real(real, names), "per_pair": real["per_pair"],
               "ranking": {case: {mode: {"metrics": v[0], "direct": v[1]} for mode, v in values.items()}
                           for case, values in real["ranking"].items()}}
    print(json.dumps(summary["real"], ensure_ascii=False, indent=1))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    if not args.no_synthetic:
        metrics.with_pitch = False  # 4-секундные фрагменты, pyin медленный
        summary["synthetic"] = synthetic_eval(metrics, args.synthetic_blocks, args.workers)
        metrics.save_style_cache()
        print(json.dumps(summary["synthetic"], indent=1))
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"[+] {args.out}")
