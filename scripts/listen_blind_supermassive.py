"""
Прослушивание цепочки «мокрый референс → слепая сеть → отпечаток → готовая настройка Supermassive» на Whiptails.

Сеть видит только мокрый вокал (окна по 8 с, медиана). По предсказанному отпечатку выбирается ближайший из
готовых эффектов Supermassive набора эмбеддинга (только признаки, которые сеть восстанавливает надежно:
RT60 по полосам, predelay, DRR, ширина). Для сравнения — лучший из того же набора по log-mel к истинному wet
(потолок выбора из готовых: подсматривает в ответ, в реальной работе недоступен).

    python scripts/listen_blind_supermassive.py --workers 16

Результат: data/processed/listen_blind_sm/whiptails/ (все файлы −18 LUFS) и README.txt с числами.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from real_pairs import align, find_lag, logmel_db  # noqa: E402
from app.eval.blind_fingerprint_net import KEYS, SECONDS, SR, load_model, predict, target_vector  # noqa: E402
from app.eval.plugin_adapter import sm_render  # noqa: E402
from app.mixing.chain_optimizer import normalize_lufs, to_stereo  # noqa: E402

RAW = Path("data/raw/dry_vocals/cambridge")
DRY = RAW / "Albert_Kader_Whiptails_11_Vox_Dry.wav"
WET = RAW / "Albert_Kader_Whiptails_12_Vox_Wet.wav"
POOL = Path("data/runs/fingerprint/embedding/effects")
NET = Path("data/runs/fingerprint/blind_net/blind_net.pt")
OUT = Path("data/processed/listen_blind_sm/whiptails")
# Признаки, которые сеть восстанавливает на отложенных эффектах со Spearman ≥ ~0.6 (blind_fingerprint_net.md)
RELIABLE = ("rt60_125", "rt60_250", "rt60_500", "rt60_1000", "rt60_2000", "rt60_4000", "rt60_8000",
            "predelay_ms", "drr_db", "side_mid_db", "iacc_late")
LUFS = -18.0


def load(path: Path) -> np.ndarray:
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == SR, (path, sr)
    return to_stereo(y[:, 0] if y.shape[1] == 1 else y.T)


def loud(y: np.ndarray) -> np.ndarray:
    y = normalize_lufs(y, SR, LUFS)
    peak = float(np.max(np.abs(y)))
    return y * (0.89 / peak) if peak > 0.89 else y


def blind_estimate(wet: np.ndarray) -> dict:
    model = load_model(NET)
    n = int(SECONDS * SR)
    starts = range(0, max(1, wet.shape[1] - n + 1), SR)
    preds = [predict(model, wet[:, s:s + n], SR) for s in starts]
    return {k: float(np.median([p[k] for p in preds])) for k in preds[0]}


def pool_sm() -> list[dict]:
    rows = []
    for path in sorted(POOL.glob("*.npz")):
        meta = json.loads(str(np.load(path, allow_pickle=True)["meta"]))
        if meta["spec"]["kind"] == "vst":
            rows.append(meta)
    return rows


def _render_job(args: tuple[dict, np.ndarray]) -> np.ndarray:
    params, x = args
    return sm_render(params, x, SR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    dry, wet = load(DRY), load(WET)
    lag, _, _ = find_lag(dry, wet)
    dry, wet = align(dry, wet, lag)
    dry_in = normalize_lufs(dry, SR, LUFS)

    est = blind_estimate(wet)
    target, _ = target_vector(est)
    idx = [KEYS.index(k) for k in RELIABLE]
    pool = pool_sm()
    dists = []
    for meta in pool:
        vec, mask = target_vector(meta["fingerprint"])
        m = mask[idx]
        dists.append(float(np.sqrt(np.sum(m * (vec[idx] - target[idx]) ** 2) / max(m.sum(), 1.0))))
    blind_i = int(np.argmin(dists))

    print(f"рендер {len(pool)} готовых эффектов Supermassive на сухом ({args.workers} процессов)...", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        renders = list(ex.map(_render_job, [(m["spec"]["params"], dry_in) for m in pool]))
    errs = [logmel_db(loud(r), loud(wet)) for r in renders]
    order = np.argsort(errs)
    oracle_i = int(order[0])
    blind_rank = int(np.where(order == blind_i)[0][0]) + 1

    import soundfile as sf
    OUT.mkdir(parents=True, exist_ok=True)
    files = {"1_dry.wav": dry, "2_reference_wet.wav": wet, "3_blind_supermassive.wav": renders[blind_i],
             "4_best_of_pool_supermassive.wav": renders[oracle_i]}
    for name, y in files.items():
        sf.write(OUT / name, loud(y).T, SR, subtype="PCM_24")

    def show(i: int) -> str:
        p = pool[i]["spec"]["params"]
        return ", ".join(f"{k}={v:.0f}" if isinstance(v, float) else f"{k}={v}" for k, v in p.items())

    fp_true = {k: est[k] for k in RELIABLE}
    lines = [
        "Whiptails: мокрый референс -> слепая сеть -> ближайшая готовая настройка Supermassive",
        f"Ошибка log-mel к 2_reference_wet (дБ, меньше — ближе): сухой {logmel_db(loud(dry), loud(wet)):.2f}, "
        f"слепой выбор {errs[blind_i]:.2f} (место {blind_rank} из {len(pool)}), лучший из набора {errs[oracle_i]:.2f}, "
        f"медиана набора {float(np.median(errs)):.2f}",
        "",
        "Слепая оценка по мокрому (надежные признаки): " + ", ".join(f"{k}={v:.2f}" for k, v in fp_true.items()),
        f"  период повторов L/R {est['echo_period_l_ms']:.0f}/{est['echo_period_r_ms']:.0f} мс (сеть, ненадежно), "
        f"вероятность повторов {est['echo_prob']:.2f}",
        "",
        f"3_blind_supermassive: {show(blind_i)}",
        f"4_best_of_pool_supermassive: {show(oracle_i)}",
    ]
    (OUT / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[+] {OUT.resolve()}")


if __name__ == "__main__":
    main()
