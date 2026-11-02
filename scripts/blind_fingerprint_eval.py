"""
Проверка слепой оценки отпечатка (app/eval/blind_fingerprint.py): насколько отпечаток, оцененный только по мокрому
вокалу, совпадает с истинным.

1. Синтетика: случайный эффект (наш DSP: ревер / дилей / ревер+дилей / хорус+ревер; Valhalla Supermassive со
   случайным режимом и ручками, свежий экземпляр на каждый рендер) на сухом вокале из реальных пар →
   blind_fingerprint(wet) против fingerprint(effect).
2. Реальные пары Cambridge: «истинный» отпечаток — impulse_features ИХ, оцененной деконволюцией Винера wet/dry
   (линейное приближение: компрессия и EQ в него не влезают, доля объясненной энергии печатается);
   слепой — только по wet.

    python scripts/blind_fingerprint_eval.py --synthetic 40 --real
Результаты: data/runs/fingerprint/blind/{synthetic,real}.jsonl и таблица ошибок по признакам.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.audio.dsp_engine import decay_reverb, send_delay  # noqa: E402
from app.eval.blind_fingerprint import blind_fingerprint  # noqa: E402
from app.eval.effect_fingerprint import BANDS_HZ, fingerprint, impulse_features  # noqa: E402
from scripts.real_pairs import PAIRS, SR, load  # noqa: E402

OUT_DIR = Path("data/runs/fingerprint/blind")
SUPERMASSIVE = Path("C:/Program Files/Common Files/VST3/ValhallaSupermassive.vst3")
# Сухие вокалы с естественными паузами (Whiptails — нарезанная петля без пауз длиннее 0.3 с, не годится)
DRY_PAIRS = ("ronnie_leadvox_tape", "oxeye_leadvox", "oxeye_backing1", "oxeye_backing3")

# Признаки сравнения: (ключ, тип). log — ошибка в октавах (|log2 оценка/истина|), db — в дБ, rank — только ранг
COMPARE = ([("rt60_s", "log")] + [(f"rt60_{c}", "log") for c in BANDS_HZ]
           + [(f"tail_db_{c}", "db") for c in BANDS_HZ]
           + [("drr_db", "db"), ("side_mid_db", "db"), ("iacc_late", "lin"), ("echo_strength", "rank"),
              ("pitch_up_db", "rank"), ("pitch_down_db", "rank")])


def _lowpass(y: np.ndarray, sr: int, hz: float) -> np.ndarray:
    from scipy.signal import butter, sosfilt
    return sosfilt(butter(2, min(hz, 0.45 * sr), "low", fs=sr, output="sos"), y, axis=-1)


def random_dsp_effect(rng: random.Random):
    """Случайный эффект нашего DSP: описание и render(x, sr) (детерминирован, ИХ ревера одна на оба канала)."""
    kind = rng.choice(["reverb", "delay", "reverb+delay", "chorus+reverb"])
    p = {"kind": kind, "decay_s": rng.uniform(0.4, 8.0), "damping": rng.uniform(0.0, 1.0),
         "rev_wet": rng.uniform(0.1, 0.6), "predelay_s": rng.uniform(0.0, 0.08), "rev_hc": rng.uniform(3000, 20000),
         "delay_s": rng.uniform(0.1, 0.5), "feedback": rng.uniform(0.1, 0.6), "spread": rng.uniform(0.0, 0.08),
         "dly_hc": rng.uniform(1500, 20000), "dly_wet": rng.uniform(0.15, 0.6), "chorus_mix": rng.uniform(0.2, 0.6)}

    def render(x: np.ndarray, sr: int) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        out = x.copy()
        src = x
        if "chorus" in kind:
            from pedalboard import Chorus, Pedalboard
            src = Pedalboard([Chorus(rate_hz=0.8, depth=0.35, mix=p["chorus_mix"])])(x, sr)
            out = src.copy()
        if "reverb" in kind:
            wet = decay_reverb(src.mean(axis=0), sr, p["decay_s"], p["damping"])
            wet = _lowpass(wet, sr, p["rev_hc"])
            d = int(p["predelay_s"] * sr)
            wet = np.concatenate([np.zeros(d, np.float32), wet[:len(wet) - d]]) if d else wet
            out = out + p["rev_wet"] * wet[None, :]
        if "delay" in kind:
            out = out + p["dly_wet"] * send_delay(src, sr, p["delay_s"], p["feedback"], p["spread"],
                                                  high_cut_hz=p["dly_hc"])
        return out.astype(np.float32)
    return p, render


def random_supermassive(rng: random.Random):
    import pedalboard
    probe = pedalboard.load_plugin(str(SUPERMASSIVE))
    modes = list(dict.fromkeys(probe.parameters["mode"].valid_values))
    p = {"kind": "supermassive", "mode": rng.choice(modes), "mix": rng.uniform(15, 50),
         "delay_ms": rng.choice([0.0, rng.uniform(0, 80), rng.uniform(100, 450)]), "feedback": rng.uniform(20, 85),
         "density": rng.choice([0.0, 50.0, 100.0]), "width": rng.uniform(0, 100), "highcut": rng.uniform(2000, 20000),
         "lowcut": rng.uniform(10, 300), "moddepth": rng.uniform(0, 100)}

    def render(x: np.ndarray, sr: int) -> np.ndarray:
        # Свежий экземпляр: reset=True не сбрасывает состояние Supermassive (docs/research/effect_fingerprint.md)
        plugin = pedalboard.load_plugin(str(SUPERMASSIVE))
        for key, value in p.items():
            if key != "kind":
                setattr(plugin, key, value)
        return plugin.process(np.asarray(x, dtype=np.float32), sr, reset=True)
    return p, render


def deconvolve(dry: np.ndarray, wet: np.ndarray, sr: int, ir_s: float = 6.0) -> tuple[np.ndarray, float]:
    """ИХ wet ≈ dry * h по каналам wet (Винер), прямой звук сдвинут в отсчет 0; доля объясненной энергии."""
    n = min(dry.shape[-1], wet.shape[-1])
    x = dry[:, :n].mean(axis=0)
    length, pre = int(ir_s * sr), int(0.05 * sr)
    nfft = 1 << int(np.ceil(np.log2(n + length)))
    X = np.fft.rfft(x, nfft)
    pxx = np.abs(X) ** 2
    lam = 1e-3 * pxx.mean()
    hs, explained = [], []
    for ch in range(2):
        h = np.fft.irfft(np.fft.rfft(wet[ch, :n], nfft) * np.conj(X) / (pxx + lam), nfft)
        h = np.concatenate([h[-pre:], h[:length]])
        pred = np.fft.irfft(np.fft.rfft(h[pre:], nfft) * X, nfft)[:n]
        explained.append(1.0 - np.sum((wet[ch, :n] - pred) ** 2) / np.sum(wet[ch, :n] ** 2))
        hs.append(h)
    h = np.stack(hs)
    peak = int(np.argmax(np.abs(h.mean(axis=0))[:2 * pre]))
    return h[:, peak:], float(np.mean(explained))


def error_table(rows: list[dict]) -> list[dict]:
    """Ошибка по признакам: число пар, Spearman, медиана ошибки в единицах типа."""
    table = []
    for key, kind in COMPARE:
        pairs = [(r["true"].get(key), r["blind"].get(key)) for r in rows]
        pairs = [(t, b) for t, b in pairs if isinstance(t, (int, float)) and isinstance(b, (int, float))
                 and np.isfinite(t) and np.isfinite(b)]
        if kind == "db":
            pairs = [(max(t, -60.0), max(b, -60.0)) for t, b in pairs]
        if len(pairs) < 4:
            table.append({"key": key, "n": len(pairs)})
            continue
        t, b = map(np.array, zip(*pairs))
        rho = float(spearmanr(t, b)[0]) if np.ptp(t) > 0 and np.ptp(b) > 0 else float("nan")
        if kind == "log":
            err = np.abs(np.log2(np.maximum(b, 0.05) / np.maximum(t, 0.05)))
        elif kind in ("db", "lin"):
            err = np.abs(b - t)
        else:
            err = np.full(len(t), np.nan)
        table.append({"key": key, "n": len(pairs), "spearman": rho, "median_err": float(np.nanmedian(err))
                      if np.isfinite(err).any() else float("nan"), "kind": kind})
    # Период повторов: доля попаданий ±10 мс там, где повторы явные (истинная сила > 0.5)
    hits = []
    for r in rows:
        if r["true"].get("echo_strength", 0) > 0.5:
            for ch in ("l", "r"):
                t, b = r["true"].get(f"echo_period_{ch}_ms"), r["blind"].get(f"echo_period_{ch}_ms")
                if t and b is not None:
                    hits.append(min(abs(b - t), abs(b - 2 * t), abs(2 * b - t)) <= 10.0)
    table.append({"key": "echo_period (±10 мс, кратные)", "n": len(hits),
                  "hit_rate": float(np.mean(hits)) if hits else float("nan")})
    return table


def print_table(title: str, table: list[dict]) -> None:
    print(f"\n{title}")
    print(f"{'признак':32s} {'n':>4s} {'Spearman':>9s} {'медиана ошибки':>15s}")
    for row in table:
        if "hit_rate" in row:
            print(f"{row['key']:32s} {row['n']:4d} {'':9s} попаданий {row['hit_rate']:.0%}")
            continue
        rho = row.get("spearman", float("nan"))
        err = row.get("median_err", float("nan"))
        unit = {"log": " окт", "db": " дБ", "lin": "", "rank": ""}.get(row.get("kind", ""), "")
        err_s = f"{err:.2f}{unit}" if np.isfinite(err) else "-"
        print(f"{row['key']:32s} {row['n']:4d} {rho:9.2f} {err_s:>15s}")


def _jsonable(d: dict) -> dict:
    return json.loads(json.dumps(d, default=lambda o: float(o) if isinstance(o, np.floating) else str(o)))


def run_synthetic(n: int, seed: int, vst_share: float) -> list[dict]:
    rng = random.Random(seed)
    dries = {name: load(dry) for name, dry, _ in PAIRS if name in DRY_PAIRS}
    rows = []
    use_vst = SUPERMASSIVE.exists()
    for i in range(n):
        if use_vst and rng.random() < vst_share:
            params, effect = random_supermassive(rng)
        else:
            params, effect = random_dsp_effect(rng)
        vocal = rng.choice(sorted(dries))
        wet = effect(dries[vocal], SR)
        row = {"i": i, "vocal": vocal, "params": params, "true": fingerprint(effect, sr=SR, ir_seconds=8.0),
               "blind": blind_fingerprint(wet, SR)}
        rows.append(_jsonable(row))
        conf = row["blind"]["confidence"]
        print(f"[{i + 1}/{n}] {params['kind']:14s} {vocal:20s} концов слов {conf['n_ends']:3d}", flush=True)
    return rows


def run_real() -> list[dict]:
    rows = []
    for name, dry_name, wet_name in PAIRS:
        dry, wet = load(dry_name), load(wet_name)
        h, explained = deconvolve(dry, wet, SR)
        row = {"pair": name, "explained": explained, "true": impulse_features(h, SR), "blind": blind_fingerprint(wet, SR)}
        rows.append(_jsonable(row))
        print(f"[+] {name:22s} объяснено линейной ИХ {explained:5.2f}  концов слов "
              f"{row['blind']['confidence']['n_ends']}", flush=True)
    return rows


def print_real(rows: list[dict]) -> None:
    keys = ["drr_db", "rt60_s", "rt60_500", "rt60_4000", "tail_db_8000", "side_mid_db", "iacc_late",
            "echo_period_l_ms", "echo_period_r_ms", "echo_strength"]
    print(f"\n{'пара':22s} {'признак':18s} {'истина(ИХ)':>11s} {'слепая':>9s}")
    for r in rows:
        for k in keys:
            t, b = r["true"].get(k), r["blind"].get(k)
            fmt = lambda v: f"{v:9.2f}" if isinstance(v, (int, float)) and np.isfinite(v) else "        -"
            print(f"{r['pair']:22s} {k:18s} {fmt(t):>11s} {fmt(b)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synthetic", type=int, default=0, help="число случайных эффектов")
    parser.add_argument("--vst-share", type=float, default=0.35, help="доля Supermassive среди синтетики")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--real", action="store_true", help="8 реальных пар Cambridge")
    parser.add_argument("--reuse", action="store_true", help="не пересчитывать, взять jsonl прошлых прогонов")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.synthetic or args.reuse:
        path = OUT_DIR / "synthetic.jsonl"
        if args.reuse and path.exists():
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            rows = run_synthetic(args.synthetic, args.seed, args.vst_share)
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print_table(f"Синтетика: {len(rows)} эффектов (все)", error_table(rows))
        for label, pick in (("наш DSP", lambda r: r["params"]["kind"] != "supermassive"),
                            ("Supermassive", lambda r: r["params"]["kind"] == "supermassive")):
            part = [r for r in rows if pick(r)]
            if part:
                print_table(f"Синтетика: {label} ({len(part)})", error_table(part))
    if args.real or args.reuse:
        path = OUT_DIR / "real.jsonl"
        if args.reuse and path.exists() and not args.real:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            rows = run_real()
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print_real(rows)
        print_table(f"Реальные пары ({len(rows)}), истина — деконволюция", error_table(rows))


if __name__ == "__main__":
    main()
