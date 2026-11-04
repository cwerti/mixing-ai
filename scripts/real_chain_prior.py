"""
Распределение цепочек тона по реальным сведениям: какие эффекты и с какими значениями на самом деле стоят на
сведенном вокале. Опора генератора обучающих цепочек (REALISTIC_STYLES) — три открытых пресета и рецепты роликов;
здесь то же самое меряется на настоящих миксах Cambridge.

На каждой паре scripts/cambridge_pairs.py подбирается цепочка «сухой кусок A → вокал того же куска A из готового
микса» (один и тот же материал — подбор ближе всего к тому, что сделал инженер). Все плагины тона проверяются
принудительно, лишние убирает упрощение оптимизатора: набор решают данные, а не правила select_plugins. Модель как
стартовая точка не используется, чтобы распределение не повторяло ее же prior.

Подбор неоднозначен (разные ручки дают похожий звук), поэтому сначала проверка на синтетике: сухой кусок
пропускается через известную случайную цепочку, и смотрим, насколько подбор ее находит.

Итог (2026-11-05, 12 синтетических цепочек): набор эффектов не восстанавливается — компрессор найден 5 из 10,
лишние мультибэнд/дисторшн/хорус в 4-5 из 12 при хорошем совпадении звука (log-mel 2.65 → 1.48). Кроме того, какой
из взаимозаменяемых эффектов останется, решает порядок упрощения (CHAIN_ORDER) и штраф за интенсивность, а слабо
слышимые ручки остаются у стартовых значений. Частоты и квантили отсюда — диагностика подбора, а не распределение
для генератора; генератор калибруется по измеримому результату обработки (спектр, хвост, динамика).

    python scripts/real_chain_prior.py recover --n 12   # синтетика: восстанавливается ли известная цепочка
    python scripts/real_chain_prior.py fit              # реальные пары → real_prior/fits.jsonl
    python scripts/real_chain_prior.py report           # частоты включения и квантили параметров
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mix_full import MAX_RENDERS, SR, TONE_EXCLUDE, TONE_PENALTY, tone_chain  # noqa: E402

from app.mixing.chain_optimizer import CHAIN_INPUT_LUFS, SEARCHABLE_PLUGINS, normalize_lufs, to_stereo  # noqa: E402

PAIRS = Path("data/processed/cambridge_pairs")
OUT = PAIRS / "real_prior"
FORCE = tuple(p for p in SEARCHABLE_PLUGINS if p not in TONE_EXCLUDE)
THREADS = 2  # потоков CMA-ES на пару; процессов --workers // THREADS (половина процессора — 8 потоков)


def _load(path: Path) -> np.ndarray:
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == SR, (path, sr)
    return y.T


def _key(y: np.ndarray) -> dict:
    import librosa
    from app.audio.vocal_enhancer import VocalEnhancer
    key_index, scale = VocalEnhancer(sr=SR).detect_key_and_scale(librosa.to_mono(y))
    return {"key_index": key_index, "scale": scale}


def _init_worker(threads: int) -> None:
    """Сеть метрики в каждом процессе берет не больше threads потоков — иначе процессы делят все ядра."""
    import torch
    torch.set_num_threads(threads)


def _synthetic_target(dry: np.ndarray, chain: dict) -> np.ndarray:
    """
    Цель синтетики: известная цепочка на том же очищенном исходнике и с тем же гейтом, что рендерит оптимизатор,
    — иначе подбор добирает недостающую очистку деэссером и компрессором и «лишние» эффекты завышены.
    """
    import librosa
    from app.audio.audio_processor import AudioProcessor
    from app.audio.dsp_engine import DSPEngine
    from app.audio.vocal_enhancer import VocalEnhancer
    from app.dataset.dataset_schema import ChainConfig
    src = normalize_lufs(to_stereo(dry), SR, CHAIN_INPUT_LUFS)
    gate = AudioProcessor(sr=SR).estimate_noise_gate_threshold(librosa.to_mono(src))
    enhancer = VocalEnhancer(sr=SR)
    cleaned = np.vstack([enhancer.suppress_resonances(ch) for ch in src]).astype(np.float32)
    return DSPEngine().apply_chain(cleaned, SR, ChainConfig(**chain), gate_threshold_db=gate)


def _fit(job: dict) -> dict:
    """Пара грузится в процессе-работнике: в очередь уходит только путь, а не массивы всех пар сразу."""
    folder = Path(job["folder"])
    dry = _load(folder / "dry_A.wav")
    extra = dict(job.get("extra", {}))
    if "chain" in job:
        target = _synthetic_target(dry, job["chain"])
    else:
        from app.eval.reference_quality import reference_quality
        target = _load(folder / "truth_A.wav")
        q = reference_quality(target, SR)
        extra.update(reliable=q["reliable"], quality_reasons=q["reasons"])
    y, info = tone_chain(dry, target, _key(target), max_renders=MAX_RENDERS, intensity_penalty=TONE_PENALTY,
                         workers=THREADS, force_plugins=FORCE)
    from real_pairs import logmel_db
    from match_reference import loud
    info.pop("report")
    return {"name": job["name"], **extra, **info, "logmel_dry": logmel_db(loud(dry), loud(target)),
            "logmel_fit": logmel_db(loud(y), loud(target))}


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _run(jobs: list[dict], path: Path, workers: int) -> None:
    """
    Пары считаются в процессах и пишутся по мере готовности. Ошибка одной пары не роняет прогон: пишется строка
    с error, и при перезапуске пара пропускается (чтобы пересчитать — удалить ее строку).
    """
    done = {r["name"] for r in _read_rows(path)}
    jobs = [j for j in jobs if j["name"] not in done]
    print(f"[*] пар к подбору: {len(jobs)} (готово {len(done)})", flush=True)
    pool = ProcessPoolExecutor(max(1, workers // THREADS), initializer=_init_worker, initargs=(THREADS,))
    with pool as ex, open(path, "a", encoding="utf-8", newline="\n") as f:
        futures = {ex.submit(_fit, job): job["name"] for job in jobs}
        for future in as_completed(futures):
            try:
                row = future.result()
                print(f"[+] {row['name']}: {', '.join(p['name'] for p in row['plugins'])}; log-mel "
                      f"{row['logmel_dry']:.2f} → {row['logmel_fit']:.2f}", flush=True)
            except Exception as e:  # noqa: BLE001 — пара пропускается, причина пишется в результат
                row = {"name": futures[future], "error": f"{type(e).__name__}: {e}"}
                print(f"[!] {row['name']}: {row['error']}", flush=True)
            f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
            f.flush()


def _ok_pairs(limit: int | None = None) -> list[dict]:
    rows = [r for r in _read_rows(PAIRS / "pairs.jsonl") if r["status"] == "ok"]
    return rows[:limit] if limit else rows


def recover(args) -> None:
    """Синтетика: сухие куски A пар через известную цепочку тона из генератора обучающих данных."""
    from app.dataset.dataset_generator import CHAIN_PRIORS
    rng = random.Random(0)
    jobs = []
    for r in _ok_pairs(args.n):
        chain = CHAIN_PRIORS["realistic"](rng)
        plugins = [{"name": p.name, "params": p.params} for p in chain.plugins if p.name in FORCE + ("eq",)]
        truth = [p for p in plugins if p["name"] != "eq"]
        jobs.append({"name": r["name"], "folder": r["folder"], "chain": {"plugins": plugins},
                     "extra": {"true_plugins": truth}})
    _run(jobs, OUT / "recover.jsonl", args.workers)


def fit(args) -> None:
    jobs = [{"name": r["name"], "folder": r["folder"], "extra": {"genre": r["genre"]}} for r in _ok_pairs()]
    _run(jobs, OUT / "fits.jsonl", args.workers)


def _summarize(rows: list[dict]) -> dict:
    """Частота включения каждого эффекта и квантили (10/50/90 %) его нормализованных параметров."""
    out = {}
    for name in FORCE:
        on = [p["params"] for r in rows for p in r["plugins"] if p["name"] == name]
        params = {k: [float(np.percentile([o[k] for o in on], q)) for q in (10, 50, 90)]
                  for k in (on[0] if on else {})}
        out[name] = {"rate": len(on) / max(1, len(rows)), "params": params}
    speeds = [r["pitch_speed"] for r in rows if r.get("pitch_speed") is not None]
    out["pitch_corrector"] = {"rate": len(speeds) / max(1, len(rows)),
                              "params": {"speed": [float(np.percentile(speeds, q)) for q in (10, 50, 90)]}
                              if speeds else {}}
    return out


def report(args) -> None:
    from app.dataset.dataset_generator import realistic_on_rates
    rec_path, fit_path = OUT / "recover.jsonl", OUT / "fits.jsonl"
    rec = [r for r in _read_rows(rec_path) if "error" not in r]
    if rec:
        print(f"Синтетика, {len(rec)} цепочек: эффект включен / найден; ошибка ручек |подбор − истина| "
              f"(норм. 0..1, медиана по найденным)")
        for name in FORCE:
            truth = [next((p["params"] for p in r["true_plugins"] if p["name"] == name), None) for r in rec]
            found = [next((p["params"] for p in r["plugins"] if p["name"] == name), None) for r in rec]
            hit = sum(t is not None and f is not None for t, f in zip(truth, found))
            false = sum(f is not None and t is None for t, f in zip(truth, found))
            errors = {}
            for t, f in zip(truth, found):
                for k in (t or {}).keys() & (f or {}).keys():
                    errors.setdefault(k, []).append(abs(f[k] - t[k]))
            params = "; ".join(f"{k} {np.median(v):.2f}" for k, v in errors.items())
            print(f"  {name:22s} был {sum(t is not None for t in truth):2d}, найден {hit:2d}, лишний {false:2d}"
                  f"  {params}")
        print(f"  log-mel к цели: сухой {np.mean([r['logmel_dry'] for r in rec]):.2f}, "
              f"подбор {np.mean([r['logmel_fit'] for r in rec]):.2f}")
    fits = [r for r in _read_rows(fit_path) if "error" not in r]
    if fits:
        clean = [r for r in fits if r["reliable"] and r["logmel_fit"] < r["logmel_dry"]]
        stats = _summarize(clean)
        prior = realistic_on_rates()
        print(f"\nРеальные пары: {len(fits)}, в распределении {len(clean)} (чистый референс и подбор лучше сухого)")
        print(f"  log-mel: сухой {np.mean([r['logmel_dry'] for r in clean]):.2f}, "
              f"подбор {np.mean([r['logmel_fit'] for r in clean]):.2f}")
        print(f"  {'эффект':22s} {'реально':>8s} {'генератор':>10s}  параметры (10/50/90 %)")
        for name, s in stats.items():
            params = "; ".join(f"{k} " + "/".join(f"{v:.2f}" for v in qs) for k, qs in s["params"].items())
            print(f"  {name:22s} {s['rate']:8.0%} {prior.get(name, 0):10.0%}  {params}")
        (OUT / "distribution.json").write_text(json.dumps({"pairs": [r["name"] for r in clean], "effects": stats},
                                                          ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, func in (("recover", recover), ("fit", fit), ("report", report)):
        p = sub.add_parser(name)
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--n", type=int, default=12)
        p.set_defaults(func=func)
    OUT.mkdir(parents=True, exist_ok=True)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
