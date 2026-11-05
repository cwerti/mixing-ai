"""
Калибровка генератора обучающих цепочек по измеримому результату обработки настоящих сведений.

Ручки эффектов по готовому звуку не восстанавливаются (scripts/real_chain_prior.py), а результат — меряется
надежно: как обработка меняет спектр, динамику, сибилянты, ширину и хвост после фразы. Здесь он сравнивается у
настоящих сведений Cambridge (сухой кусок A → вокал того же куска из микса) и у генератора на тех же сухих
кусках, с тем же путем рендера, что у датасета (реализации эффектов EffectVariants, слои даблов/гармоний).

    python scripts/calibrate_generator.py eq        # спектры сведенного вокала (цели EQ) → app/data/real_eq_curves.json
    python scripts/calibrate_generator.py compare   # таблица «настоящее сведение / генератор» (разница с сухим)

Оговорка: вокал сведения выделен Demucs, который частично съедает хвосты и низ — реальный ревер чуть больше,
а срез низа чуть меньше, чем в таблице.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.mixing.chain_optimizer import normalize_lufs, to_stereo  # noqa: E402
from app.dataset.source_eq import EQ_CENTERS_HZ, GRID_HZ, REAL_EQ_PATH, fit_eq_gains, ltas_db  # noqa: E402
from app.ml.features import MODEL_LUFS  # noqa: E402

SR = 44100
PAIRS = Path("data/processed/cambridge_pairs")
OUT = PAIRS / "real_prior"
KEYS = ("crest_db", "dr_p95_p10_db", "sib_p90_db", "air_ratio_db", "width_notes_db", "width_tails_db",
        "decay_fast_db")
BANDS = {"<150": (0, 150), "150-500": (150, 500), "2-5k": (2000, 5000), "5-10k": (5000, 10000),
         ">10k": (10000, 20000)}
WORKERS = 8  # половина процессора


def _load(path: Path) -> np.ndarray:
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == SR, (path, sr)
    return y.T


def _ok_pairs() -> list[dict]:
    rows = [json.loads(line) for line in (PAIRS / "pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    return [r for r in rows if r["status"] == "ok"]


def _eq_job(row: dict) -> dict:
    """Спектр сведенного вокала (цель EQ генератора) и, для диагностики, подгонка EQ «сухой → сведение»."""
    folder = Path(row["folder"])
    target = ltas_db(_load(folder / "truth_A.wav"), SR)
    gains, err = fit_eq_gains(target - ltas_db(_load(folder / "dry_A.wav"), SR), np.array(EQ_CENTERS_HZ), SR)
    return {"name": row["name"], "target_db": [round(float(v), 2) for v in target],
            "gains_db": [round(float(g), 2) for g in gains], "fit_err_db": round(err, 2)}


def eq(args) -> None:
    from app.eval.reference_quality import reference_quality
    rows = [r for r in _ok_pairs() if reference_quality(_load(Path(r["folder"]) / "truth_A.wav"), SR)["reliable"]]
    with ProcessPoolExecutor(WORKERS) as ex:
        fits = list(ex.map(_eq_job, rows))
    gains = np.array([f["gains_db"] for f in fits])
    print(f"пар {len(fits)}, ошибка подгонки «сухой → сведение» {np.median([f['fit_err_db'] for f in fits]):.2f} дБ")
    print("полоса      Гц    медиана [10%..90%] дБ")
    for i, fc in enumerate(EQ_CENTERS_HZ):
        q = np.percentile(gains[:, i], (10, 50, 90))
        print(f"  band{i + 1} {fc:7.0f}  {q[1]:+5.1f} [{q[0]:+5.1f}..{q[2]:+5.1f}]")
    REAL_EQ_PATH.write_text(json.dumps({
        "source": "Cambridge Multitracks: вокал из микса (Demucs) и сухой кусок того же места, "
                  "scripts/calibrate_generator.py eq",
        "grid_hz": [round(float(f), 2) for f in GRID_HZ], "centers_hz": list(EQ_CENTERS_HZ),
        "targets_db": [f.pop("target_db") for f in fits], "curves": fits}, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"→ {REAL_EQ_PATH}")


def describe(y: np.ndarray) -> dict:
    """Измеримый результат: динамика, сибилянты, воздух, ширина, баланс полос и хвост после фразы."""
    from app.eval.style_metrics import extract_style_features
    from app.eval.tail_decay import decay_profile
    y = normalize_lufs(to_stereo(y), SR, MODEL_LUFS)
    feats = extract_style_features(y, SR, with_pitch=False)
    out = {k: float(feats[k]) for k in KEYS}
    m = np.mean(y, axis=0)
    spec = np.abs(np.fft.rfft(m * np.hanning(len(m)))) ** 2
    f = np.fft.rfftfreq(len(m), 1 / SR)
    total = spec[(f > 150) & (f < 5000)].sum()
    out.update({k: float(10 * np.log10(spec[(f >= a) & (f < b)].sum() / total + 1e-12))
                for k, (a, b) in BANDS.items()})
    tail = decay_profile(y, SR)
    out["tail_mid_0.3s"] = float(tail[1]) if tail is not None else float("nan")
    out["_ltas"] = ltas_db(y, SR).tolist()  # не признак таблицы: для поправки EQ генератора (--update-bias)
    return out


def _real_job(row: dict) -> dict:
    folder = Path(row["folder"])
    return {"name": row["name"], "dry": describe(_load(folder / "dry_A.wav")),
            "truth": describe(_load(folder / "truth_A.wav"))}


def _gen_job(job: tuple) -> list[dict]:
    """Рендер как в _generate_sample_worker датасета: тот же prior, слои даблов и реализации эффектов."""
    folder, seed, n, harmony_ratio, own_prob = job
    from app.audio.dsp_engine import DSPEngine
    from app.dataset.dataset_generator import (
        CHAIN_PRIORS, _effect_variants, adapt_chain_to_source, sample_reference_layers, with_reference_layers,
    )
    dry = normalize_lufs(to_stereo(_load(Path(folder) / "dry_A.wav")), SR, MODEL_LUFS)
    rng = random.Random(seed)
    out = []
    for k in range(n):
        chain = adapt_chain_to_source(CHAIN_PRIORS["realistic"](rng), dry, SR, rng, "realistic")
        layers = sample_reference_layers(rng) if rng.random() < harmony_ratio else []
        chain = with_reference_layers(chain, layers) if layers else chain
        key, scale = rng.randint(0, 11), rng.choice(["major", "minor"])
        if own_prob >= 1.0:
            y = DSPEngine().apply_chain(dry, SR, chain, target_key_index=key, target_scale=scale,
                                        apply_safety_clamps=False)
        else:
            y, _ = _effect_variants(SR, own_prob).apply_chain(dry, chain, random.Random(seed * 31 + k),
                                                              target_key_index=key, target_scale=scale)
        out.append(describe(y))
    return out


def _truth_ltas(folder: str) -> list[float]:
    return ltas_db(normalize_lufs(to_stereo(_load(Path(folder) / "truth_A.wav")), SR, MODEL_LUFS), SR).tolist()


def update_bias(rows: list[dict], gens: list[list[dict]]) -> None:
    """
    Систематическое расхождение спектра генератора с настоящим сведением на тех же сухих (медиана по парам,
    сглаживание до октавы) добавляется к поправке цели EQ генератора (bias_db в real_eq_curves.json).
    """
    from app.dataset.source_eq import smooth_octave
    with ProcessPoolExecutor(WORKERS) as ex:
        truths = list(ex.map(_truth_ltas, [r["folder"] for r in rows]))
    residual = np.median([np.array(t) - np.array(d["_ltas"]) for t, ds in zip(truths, gens) for d in ds], axis=0)
    residual = smooth_octave(residual)
    data = json.loads(REAL_EQ_PATH.read_text(encoding="utf-8"))
    bias = np.array(data.get("bias_db", np.zeros(len(GRID_HZ)))) + residual
    data["bias_db"] = [round(float(v), 2) for v in bias]
    REAL_EQ_PATH.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")
    idx = [int(np.argmin(np.abs(GRID_HZ - f))) for f in (60, 120, 300, 1000, 3000, 8000, 14000)]
    print("поправка цели EQ (60/120/300/1k/3k/8k/14k Гц):", np.round(bias[idx], 1), f"→ {REAL_EQ_PATH}")


def _fmt(a: np.ndarray) -> str:
    return f"{np.nanmedian(a):6.1f} [{np.nanpercentile(a, 10):6.1f}..{np.nanpercentile(a, 90):6.1f}]"


def _test_pairs() -> set[str]:
    """Пары замороженного честного теста (app/data/test_songs.json): по ним генератор не калибруется."""
    path = Path("app/data/test_songs.json")
    return set(json.loads(path.read_text(encoding="utf-8"))["pairs"]) if path.exists() else set()


def compare(args) -> None:
    rows = _ok_pairs()
    if not args.with_test_pairs:
        test = _test_pairs()
        rows = [r for r in rows if r["name"] not in test]
    real_path = OUT / ("outcomes_real.json" if args.with_test_pairs else "outcomes_real_calib.json")
    if real_path.exists():
        real = json.loads(real_path.read_text(encoding="utf-8"))
    else:
        with ProcessPoolExecutor(WORKERS) as ex:
            real = list(ex.map(_real_job, rows))
        real_path.write_text(json.dumps(real, ensure_ascii=False), encoding="utf-8")
    with ProcessPoolExecutor(WORKERS) as ex:
        gens = list(ex.map(_gen_job, [(r["folder"], args.seed + i, args.n, args.harmony_ratio, args.own_prob)
                                      for i, r in enumerate(rows)]))
    print(f"пар {len(rows)} × {args.n} цепочек; разница с сухим: медиана [10%..90%]; сдвиг — "
          f"(медиана генератора − реальная) / разброс реальных (90%−10%)")
    print(f"{'признак':16s} {'настоящее сведение':>26s} {'генератор':>26s}  сдвиг")
    summary = {}
    for k in [k for k in real[0]["dry"] if not k.startswith("_")]:
        r = np.array([p["truth"][k] - p["dry"][k] for p in real])
        g = np.array([d[k] - p["dry"][k] for p, ds in zip(real, gens) for d in ds])
        spread = max(np.nanpercentile(r, 90) - np.nanpercentile(r, 10), 1e-6)
        shift = (np.nanmedian(g) - np.nanmedian(r)) / spread
        summary[k] = {"real": [float(np.nanpercentile(r, q)) for q in (10, 50, 90)],
                      "gen": [float(np.nanpercentile(g, q)) for q in (10, 50, 90)], "shift": float(shift)}
        print(f"{k:16s} {_fmt(r):>26s} {_fmt(g):>26s}  {shift:+5.2f}")
    print(f"средний |сдвиг|: {np.mean([abs(s['shift']) for s in summary.values()]):.2f}")
    (OUT / f"calibration_{args.tag}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                                       encoding="utf-8")
    if args.update_bias:
        update_bias(rows, gens)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("eq").set_defaults(func=eq)
    p = sub.add_parser("compare")
    p.add_argument("--n", type=int, default=3, help="цепочек генератора на пару")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--harmony-ratio", type=float, default=0.35, help="как dataset_v5")
    p.add_argument("--own-prob", type=float, default=0.35, help="как dataset_v5; 1.0 — только наш DSP")
    p.add_argument("--tag", default="current")
    p.add_argument("--with-test-pairs", action="store_true",
                   help="калибровать и по парам честного теста (по умолчанию только по калибровочной половине)")
    p.add_argument("--update-bias", action="store_true",
                   help="добавить расхождение спектра с настоящими сведениями к поправке цели EQ генератора")
    p.set_defaults(func=compare)
    OUT.mkdir(parents=True, exist_ok=True)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
