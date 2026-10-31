"""
Кэш импульсных характеристик для доменной рандомизации реверба (app/audio/effect_variants.py, IRBank).

Источники — папки data/raw/irs/<source>/ (скачанные библиотеки с README о лицензии, см.
docs/research/domain_randomization.md) и синтетические алгоритмические ИХ (app/audio/synthetic_reverbs.py):
плейт Даторро, FDN-холлы/комнаты, пружины, гейтованный реверб.

Обработка каждой ИХ: только WAV (данные из интернета не исполняются), каналы: стерео как есть, B-format и
многоканальные массивы — канал 0 (моно); ресемплинг в 44.1 кГц; обрезка до начала (порог −20 дБ от пика минус
1 мс); прямой звук приглушается окном 0→5 мс (в посыл идет почти только отражение, сухой сигнал микшируется
отдельно); хвост до −60 дБ по кривой Шрёдера, не длиннее 12 с, затухание 50 мс; энергия канала = 4 (как у
decay_reverb). RT60 — по кривой Шрёдера (estimate_rt60). Слишком короткие (< 0.12 с, кабинеты) пропускаются.

Выход: data/raw/irs/_cache/irs.npy ((2, N) float16, все ИХ подряд) и index.json.

Пример:
    python scripts/build_ir_cache.py
    python scripts/build_ir_cache.py --synthetic 0     # только скачанные
"""
import argparse
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
sys.path.append(str(Path(__file__).parent.parent.absolute()))

import numpy as np  # noqa: E402

IRS_DIR = Path("data/raw/irs")
CACHE_DIR = IRS_DIR / "_cache"
SR = 44100
MAX_SECONDS = 12.0

# Скачанные библиотеки: папка -> (лицензия, фильтр путей WAV, максимум файлов). Ely Lady Chapel не берется:
# LICENCE.txt внутри архива — CC BY-NC-SA 4.0 (см. data/raw/irs/ely_lady_chapel/README.md).
SOURCES = {
    "voxengo": ("Voxengo IR licence (royalty-free use, no redistribution)", r"^(?!.*Direct Cabinet).*\.wav$", None),
    "york_vr_game_spaces": ("CC BY 4.0", r"-001\.wav$", None),
    "ccrma_shoebox": ("CC BY 4.0", r"^(?!.*Listening Study).*\.wav$", None),
    "detmold_srir": ("CC BY 4.0", r"\.wav$", None),
}


def load_ir(path: Path) -> np.ndarray:
    """WAV -> (каналы, n) при SR: стерео остается стерео, остальное — канал 0."""
    import soundfile as sf
    import soxr
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    data = data.T
    data = data if data.shape[0] == 2 else data[:1]
    if sr != SR:
        data = np.stack([soxr.resample(ch, sr, SR) for ch in data])
    return data.astype(np.float32)


def prepare_ir(ir: np.ndarray, sr: int = SR) -> tuple[np.ndarray, float] | None:
    """Обрезка, приглушение прямого звука, хвост до −60 дБ, нормировка; (ИХ (2, n), RT60) или None."""
    from app.audio.synthetic_reverbs import estimate_rt60
    if ir.shape[0] == 1:
        ir = np.vstack([ir, ir])
    mono = np.abs(ir).max(axis=0)
    if not np.isfinite(ir).all() or mono.max() <= 0:
        return None
    onset = max(0, int(np.argmax(mono > 0.1 * mono.max())) - int(0.001 * sr))
    ir = ir[:, onset:]
    fade_in = min(int(0.005 * sr), ir.shape[1])
    ir[:, :fade_in] *= 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade_in))[None, :]
    energy = np.cumsum((ir.astype(np.float64) ** 2).sum(axis=0)[::-1])[::-1]
    if energy[0] <= 0:
        return None
    edc = 10 * np.log10(np.maximum(energy / energy[0], 1e-30))
    end = int(np.argmax(edc < -60.0)) if (edc < -60.0).any() else ir.shape[1]
    end = min(end, int(MAX_SECONDS * sr))
    ir = ir[:, :end]
    fade = min(int(0.05 * sr), ir.shape[1] // 4)
    if fade > 0:
        ir[:, -fade:] *= np.linspace(1.0, 0.0, fade)[None, :]
    rt60 = estimate_rt60(ir, sr)
    if not np.isfinite(rt60) or rt60 < 0.12 or ir.shape[1] < int(0.05 * sr):
        return None
    ir = 2.0 * ir / np.sqrt(np.mean((ir.astype(np.float64) ** 2).sum(axis=1)))
    return ir.astype(np.float32), float(rt60)


def external_items(max_per_source: dict[str, int | None]) -> list[tuple[str, str, Path]]:
    items = []
    for source, (licence, pattern, limit) in SOURCES.items():
        root = IRS_DIR / source / "wav"
        if not root.exists():
            print(f"    [-] нет {root}")
            continue
        files = sorted(p for p in root.rglob("*") if p.suffix.lower() == ".wav"
                       and re.search(pattern, str(p.relative_to(root)).replace("\\", "/"), re.I))
        limit = max_per_source.get(source, limit)
        if limit is not None and len(files) > limit:
            # Равномерная подвыборка (детерминированная), чтобы покрыть все залы/позиции
            files = [files[i] for i in np.linspace(0, len(files) - 1, limit).round().astype(int)]
        items += [(source, licence, f) for f in files]
    return items


def _external_worker(item: tuple[str, str, str]) -> dict | None:
    source, licence, path = item
    try:
        prepared = prepare_ir(load_ir(Path(path)))
    except Exception as e:  # битый или экзотический WAV
        print(f"    [-] {path}: {e}")
        return None
    if prepared is None:
        return None
    ir, rt60 = prepared
    return {"ir": ir, "meta": {"kind": "external", "source": source, "licence": licence,
                               "name": Path(path).stem, "file": str(Path(path).relative_to(IRS_DIR)), "rt60": rt60}}


def synthetic_specs(n: int, seed: int = 0) -> list[dict]:
    """Параметры синтетических ИХ: плейты 35 %, FDN 45 %, пружины 12 %, гейт 8 %; RT60 лог-равномерно 0.3–10 с."""
    rng = np.random.default_rng(seed)
    specs = []
    kinds = rng.choice(["plate", "fdn", "spring", "gated"], size=n, p=[0.35, 0.45, 0.12, 0.08])
    for i, kind in enumerate(kinds):
        specs.append({"kind": str(kind), "seed": int(seed * 100000 + i), "rt60": float(np.exp(rng.uniform(
            np.log(0.3), np.log(10.0))))})
    return specs


def _synthetic_worker(spec: dict) -> dict | None:
    from app.audio.synthetic_reverbs import fdn_ir, plate_ir, spring_ir
    rng = np.random.default_rng(spec["seed"])
    rt60 = spec["rt60"]
    seconds = min(MAX_SECONDS, 1.3 * rt60 + 0.3)
    kind = spec["kind"]
    if kind == "plate":
        # decay бака -> RT60 при size 1 (замер): 0.4 -> 1.3 с, 0.6 -> 2.1, 0.8 -> 4.7, 0.9 -> 10
        size = float(rng.uniform(0.6, 1.5))
        decay = float(np.clip(np.interp(np.log(rt60 / size), np.log([0.6, 1.31, 2.13, 4.68, 10.1]),
                                        [0.15, 0.4, 0.6, 0.8, 0.9]), 0.1, 0.93))
        ir = plate_ir(SR, seconds, decay, float(rng.uniform(0.0, 0.8)), size=size,
                      bandwidth=float(rng.uniform(0.6, 0.9995)), mod_depth_samples=int(rng.integers(0, 16)),
                      mod_rate_hz=float(rng.uniform(0.3, 2.0)))
        params = {"decay": decay, "size": size}
    elif kind == "fdn":
        lines = int(rng.choice([8, 16]))
        size_lo = float(rng.uniform(8.0, 40.0))
        ir = fdn_ir(SR, seconds, rt60, rt60 * float(rng.uniform(0.25, 1.0)), rng, n_lines=lines,
                    size_ms=(size_lo, size_lo * float(rng.uniform(1.5, 4.0))),
                    crossover_hz=float(rng.uniform(1500.0, 6000.0)), diffusion=float(rng.uniform(0.0, 0.75)))
        params = {"lines": lines, "size_ms": size_lo}
    elif kind == "spring":
        rt60 = float(np.clip(rt60, 1.0, 5.0))
        ir = spring_ir(SR, min(MAX_SECONDS, 1.3 * rt60 + 0.3), rng, rt60=rt60)
        params = {}
    else:
        # Гейтованный реверб: плотный хвост 1.5–4 с, обрезанный через 150–450 мс
        long_rt = float(rng.uniform(1.5, 4.0))
        ir = fdn_ir(SR, 1.0, long_rt, long_rt * 0.5, rng, n_lines=16, size_ms=(10.0, 40.0), diffusion=0.7)
        hold = int(rng.uniform(0.15, 0.45) * SR)
        env = np.ones(ir.shape[1], dtype=np.float32)
        env[hold:] = np.exp(-np.arange(ir.shape[1] - hold) / (0.01 * SR))
        ir = ir * env[None, :]
        params = {"hold_s": hold / SR}
    prepared = prepare_ir(ir)
    if prepared is None:
        return None
    ir, measured = prepared
    return {"ir": ir, "meta": {"kind": "synthetic", "source": f"synthetic_{kind}", "licence": "own code",
                               "name": f"{kind}_{spec['seed']}", "rt60": measured, "target_rt60": spec["rt60"],
                               **params}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Кэш ИХ для доменной рандомизации реверба")
    parser.add_argument("--synthetic", type=int, default=200, help="Сколько синтетических ИХ сгенерировать")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", default=str(CACHE_DIR))
    args = parser.parse_args()
    started = time.perf_counter()
    items = external_items({})
    print(f"[*] Скачанных WAV к обработке: {len(items)}; синтетических: {args.synthetic}")
    results = []
    with ProcessPoolExecutor(args.workers) as pool:
        results += list(pool.map(_external_worker, [(s, lic, str(p)) for s, lic, p in items], chunksize=4))
        results += list(pool.map(_synthetic_worker, synthetic_specs(args.synthetic), chunksize=2))
    results = [r for r in results if r is not None]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total = sum(r["ir"].shape[1] for r in results)
    data = np.lib.format.open_memmap(out / "irs.npy", mode="w+", dtype=np.float16, shape=(2, total))
    index, offset = [], 0
    for r in results:
        n = r["ir"].shape[1]
        data[:, offset:offset + n] = r["ir"].astype(np.float16)
        index.append({**r["meta"], "offset": offset, "length": n})
        offset += n
    data.flush()
    del data
    by_source: dict[str, list[float]] = {}
    for it in index:
        by_source.setdefault(it["source"], []).append(it["rt60"])
    summary = {s: {"count": len(v), "rt60_median": float(np.median(v)), "rt60_min": float(min(v)),
                   "rt60_max": float(max(v))} for s, v in sorted(by_source.items())}
    with open(out / "index.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"sr": SR, "dtype": "float16", "created": time.strftime("%Y-%m-%d %H:%M"), "summary": summary,
                   "irs": index}, f, ensure_ascii=False, indent=1)
    for s, v in summary.items():
        print(f"    {s:28s} {v['count']:4d}  RT60 медиана {v['rt60_median']:.2f} с ({v['rt60_min']:.2f}–{v['rt60_max']:.2f})")
    print(f"[+] {out}: {len(index)} ИХ, {total / SR / 60:.1f} мин, {(out / 'irs.npy').stat().st_size / 1e6:.0f} МБ, "
          f"{time.perf_counter() - started:.0f} с")


if __name__ == "__main__":
    main()
