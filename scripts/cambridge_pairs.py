"""
Реальные пары из Cambridge: сухой лид-вокал (data/raw/dry_vocals/cambridge) + вокал, выделенный Demucs из готового
микса (data/raw/cambridge_mixes → data/processed/cambridge_pairs/sep/htdemucs/<микс>/vocals.wav).

    python scripts/cambridge_pairs.py pairs              # сопоставить, выровнять, нарезать A/B → pairs.jsonl
    python scripts/cambridge_pairs.py match [--limit N]  # подбор пресета для каждой пары → match/<пара>/, summary.json
    python scripts/cambridge_pairs.py match --refine 6   # + уменьшенный mix, сухой → match_refine/, summary_refine.json
    python scripts/cambridge_pairs.py full               # полная цепочка (тон DSP + Supermassive) → full/, summary_full.json

Протокол cross: референс — выделенный вокал части B песни, сухой — часть A, истина — выделенный вокал части A.
Так референс не совпадает по материалу с обрабатываемым сухим, как в реальной работе. Истина — выход Demucs,
не чистый мокрый стем: часть хвоста ревера и бэки в ней искажены одинаково с референсом.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.eval.blind_fingerprint_net import SR  # noqa: E402

MIXES = Path("data/raw/cambridge_mixes")
DRYS = Path("data/raw/dry_vocals/cambridge")
OUT = Path("data/processed/cambridge_pairs")
SEP = OUT / "sep" / "htdemucs"
SECTION_S = 30.0
MIN_SECTION_S = 10.0  # слепая сеть смотрит окнами по 8 с
MIN_CORR = 0.5        # корреляция огибающих сухого и выделенного вокала на перекрытии
ENV_HOP = 441  # 10 мс при 44.1 кГц


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def pick_dry(artist: str, title: str) -> Path | None:
    """Сухой лид-вокал песни: имя файла начинается с артиста и названия; лид важнее бэков, без дублей."""
    prefix = _key(artist) + _key(title)
    files = [p for p in DRYS.glob("*.wav") if _key(p.stem).startswith(prefix)]
    if not files:
        return None

    def score(p: Path) -> tuple:
        n = p.stem.lower()
        bad = any(w in n for w in ("backing", "bv", "harm", "double", "dt", "room", "scratch", "wet", "choir"))
        return (bad, "lead" not in n and "ld" not in n, n)
    return sorted(files, key=score)[0]


def _load(path: Path) -> np.ndarray:
    import librosa
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.T if y.shape[1] > 1 else np.vstack([y[:, 0], y[:, 0]])
    return librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y


def _envelope(y: np.ndarray) -> np.ndarray:
    mono = y.mean(axis=0)
    n = len(mono) // ENV_HOP
    e = np.sqrt(np.mean(mono[:n * ENV_HOP].reshape(n, ENV_HOP) ** 2, axis=1))
    env = np.log(e + 1e-4)
    return env - env.mean()


def global_lag(dry: np.ndarray, wet: np.ndarray) -> tuple[int, float]:
    """
    Сдвиг wet относительно dry (сэмплы, >0 — wet позже) по огибающим на всей длине и корреляция Пирсона огибающих
    на перекрытии. Сухие стемы Cambridge часто — отрывок (медиана 26 с) целой песни: пик, нормированный по полным
    длинам, у них занижен, поэтому качество меряется только там, где оба сигнала есть.
    """
    a, b = _envelope(dry), _envelope(wet)
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    corr = np.fft.irfft(np.fft.rfft(b, n) * np.conj(np.fft.rfft(a, n)), n)
    best = int(np.argmax(corr))
    lag = best if best < n // 2 else best - n
    x, y = (a, b[lag:]) if lag >= 0 else (a[-lag:], b)
    m = min(len(x), len(y))
    if m < 100:
        return lag * ENV_HOP, 0.0
    return lag * ENV_HOP, float(np.corrcoef(x[:m], y[:m])[0, 1])


def sections(dry: np.ndarray, seconds: float = SECTION_S) -> tuple[int, int] | None:
    """Начала частей A (первая половина) и B (вторая) с наибольшей долей активного вокала; None — песня коротка."""
    n = int(seconds * SR)
    if dry.shape[1] < 2 * n:
        return None
    env = _envelope(dry)
    active = (env > env.max() - 4.0).astype(float)  # огибающая в натуральном логарифме: −4 ≈ −35 дБ
    per = n // ENV_HOP
    csum = np.concatenate([[0.0], np.cumsum(active)])
    half = len(active) // 2
    starts_a = np.arange(0, max(1, half - per), 50)
    starts_b = np.arange(half, max(half + 1, len(active) - per), 50)
    if not len(starts_a) or not len(starts_b):
        return None
    a = int(starts_a[np.argmax(csum[starts_a + per] - csum[starts_a])])
    b = int(starts_b[np.argmax(csum[np.minimum(starts_b + per, len(active))] - csum[starts_b])])
    if b < a + per:
        b = a + per
    if (b + per) * ENV_HOP > dry.shape[1]:
        return None
    return a * ENV_HOP, b * ENV_HOP


def make_pairs(args) -> None:
    import soundfile as sf
    manifest = json.loads((MIXES / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for m in manifest:
        name = Path(m["filename"]).stem
        vocals = SEP / name / "vocals.wav"
        dry_path = pick_dry(m["artist"], m["title"])
        row = {"name": name, "artist": m["artist"], "title": m["title"], "genre": m["genre"],
               "dry": str(dry_path) if dry_path else None}
        if dry_path is None or not vocals.exists():
            rows.append(row | {"status": "нет сухого" if dry_path is None else "нет разделения"})
            continue
        dry, wet = _load(dry_path), _load(vocals)
        lag, peak = global_lag(dry, wet)
        # Выравнивание: wet сдвигается так, чтобы совпасть с dry
        if lag >= 0:
            wet = wet[:, lag:]
        else:
            dry = dry[:, -lag:]
        n = min(dry.shape[1], wet.shape[1])
        dry, wet = dry[:, :n], wet[:, :n]
        seconds = min(SECTION_S, n / SR / 2)  # отрывок короче минуты — части короче, но не меньше MIN_SECTION_S
        sec = sections(dry, seconds) if seconds >= MIN_SECTION_S else None
        row |= {"lag_s": lag / SR, "env_corr": peak, "section_s": seconds}
        if peak < MIN_CORR or sec is None:
            rows.append(row | {"status": "слабое совпадение" if peak < MIN_CORR else "коротко"})
            continue
        a, b = sec
        L = int(seconds * SR)
        folder = OUT / "pairs" / name
        folder.mkdir(parents=True, exist_ok=True)
        sf.write(folder / "dry_A.wav", dry[:, a:a + L].T, SR, subtype="PCM_24")
        sf.write(folder / "truth_A.wav", wet[:, a:a + L].T, SR, subtype="PCM_24")
        sf.write(folder / "ref_B.wav", wet[:, b:b + L].T, SR, subtype="PCM_24")
        rows.append(row | {"status": "ok", "a_s": a / SR, "b_s": b / SR, "folder": str(folder)})
        print(f"[+] {name}: сдвиг {lag / SR:+.2f} с, пик {peak:.2f}, A {a / SR:.0f} с, B {b / SR:.0f} с", flush=True)
    (OUT / "pairs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                     encoding="utf-8")
    from collections import Counter
    print(Counter(r["status"] for r in rows))


def match(args) -> None:
    from match_reference import NET, load_presets, run_match
    from app.eval.blind_fingerprint_net import load_model
    rows = [json.loads(line) for line in (OUT / "pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["status"] == "ok"][: args.limit or None]
    model, presets = load_model(NET), load_presets()
    suffix = "_refine" if args.refine else ""
    summary_path = OUT / f"summary{suffix}.json"
    done = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    for r in rows:
        if r["name"] in done:
            continue
        folder = Path(r["folder"])
        res = run_match(_load(folder / "dry_A.wav"), _load(folder / "ref_B.wav"), _load(folder / "truth_A.wav"),
                        OUT / f"match{suffix}" / r["name"], model, presets, args.k, args.workers,
                        header=[f"{r['artist']} — {r['title']} ({r['genre']})", f"сухой {r['dry']}"],
                        refine=args.refine)
        done[r["name"]] = res | {"genre": r["genre"]}
        summary_path.write_text(json.dumps(done, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    keys = ("logmel_dry", "logmel_by_description", "logmel_symmetric", "logmel_oracle", "logmel_median")
    print(f"пар {len(done)}: " + ", ".join(f"{k[7:]} {np.mean([d[k] for d in done.values()]):.2f}" for k in keys))


def full(args) -> None:
    """
    Полная цепочка scripts/mix_full.py на каждой паре: тон подбирается параллельно по парам (кэш full_tone/),
    затем пространство — по паре за раз (рендеры Supermassive параллельно внутри пары).
    """
    from concurrent.futures import ProcessPoolExecutor
    from mix_full import (MAX_RENDERS, MODEL, TONE_PENALTY, UNRELIABLE_PENALTY, _tone_job, full_chain, tone_only,
                          tone_prior)
    from app.eval.reference_quality import reference_quality
    from match_reference import NET, load_presets
    from app.eval.blind_fingerprint_net import load_model
    from app.ml.pipeline import MixingAIMLPipeline
    rows = [json.loads(line) for line in (OUT / "pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["status"] == "ok"][: args.limit or None]
    suffix = f"_{args.tag}" if args.tag else ""
    cache = OUT / f"full_tone{suffix}"
    cache.mkdir(parents=True, exist_ok=True)
    audio = {r["name"]: tuple(_load(Path(r["folder"]) / f) for f in ("dry_A.wav", "ref_B.wav", "truth_A.wav"))
             for r in rows}
    quality = {n: reference_quality(a[1], SR) for n, a in audio.items()}
    todo = [r["name"] for r in rows if not (cache / f"{r['name']}.npz").exists()]
    if todo:
        pipeline = MixingAIMLPipeline(MODEL)
        jobs = [(audio[n][0], audio[n][1], tone_prior(pipeline, audio[n][0], audio[n][1]), 0, MAX_RENDERS,
                 TONE_PENALTY if quality[n]["reliable"] else UNRELIABLE_PENALTY) for n in todo]
        del pipeline
        with ProcessPoolExecutor(args.workers) as ex:
            for n, (y, info) in zip(todo, ex.map(_tone_job, jobs)):
                np.savez(cache / f"{n}.npz", y=y, info=json.dumps(info, ensure_ascii=False, default=float))
                print(f"[+] тон {n}: " + ", ".join(p["name"] for p in info["plugins"]), flush=True)
    blind, presets = load_model(NET), load_presets()
    summary_path = OUT / f"summary_full{suffix}.json"
    done = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    for r in rows:
        if r["name"] in done:
            continue
        z = np.load(cache / f"{r['name']}.npz")
        dry, ref, truth = audio[r["name"]]
        tone = (z["y"], json.loads(str(z["info"])))
        header = [f"{r['artist']} — {r['title']} ({r['genre']})", f"сухой {r['dry']}"]
        out = OUT / f"full{suffix}" / r["name"]
        q = quality[r["name"]]
        if q["reliable"]:
            res = full_chain(dry, ref, out, tone=tone, blind_model=blind, presets=presets, truth=truth,
                             workers=args.workers, header=header)
        else:
            res = tone_only(dry, out, tone=tone, reasons=q["reasons"], truth=truth, header=header)
        done[r["name"]] = res | {"genre": r["genre"], "reference_reliable": q["reliable"]}
        summary_path.write_text(json.dumps(done, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    keys = ("logmel_dry", "logmel_tone", "logmel_symmetric")
    for title, part in (("все", list(done.values())),
                        ("чистые референсы", [d for d in done.values() if d.get("reference_reliable", True)])):
        better = sum(d["logmel_symmetric"] < d["logmel_dry"] for d in part)
        print(f"{title}, пар {len(part)}: " + ", ".join(f"{k[7:]} {np.mean([d[k] for d in part]):.2f}" for k in keys)
              + f", итог лучше сухого в {better}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pairs")
    m = sub.add_parser("match")
    m.add_argument("--limit", type=int, default=0)
    m.add_argument("--k", type=int, default=24)
    m.add_argument("--workers", type=int, default=18)
    m.add_argument("--refine", type=int, default=0, help="второй круг: mix ×0.5/×0.25 для N лучших, сухой, без дилеев")
    f = sub.add_parser("full")
    f.add_argument("--limit", type=int, default=0)
    f.add_argument("--workers", type=int, default=18)
    f.add_argument("--tag", default="", help="суффикс папок и сводки, чтобы не затирать прошлый вариант")
    args = parser.parse_args()
    {"pairs": make_pairs, "match": match, "full": full}[args.cmd](args)


if __name__ == "__main__":
    main()
