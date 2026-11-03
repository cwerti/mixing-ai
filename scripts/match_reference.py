"""
Подбор пресета Supermassive под мокрый референс: слепая сеть → предотбор по кэшу пресетов → двусторонняя проверка.
Рабочий сценарий: сухой оригинал референса не нужен. Если есть истинный результат для этого сухого (--truth),
считается log-mel и лучший из предотобранных по нему (потолок, подсматривает в ответ).

    python scripts/match_reference.py --dry dry.wav --ref wet.wav [--truth wet.wav] --out data/processed/match/<имя>

Пресеты: data/runs/fingerprint/presets/presets.jsonl (scripts/profile_presets.py). В папке результата: сухой,
референс, выбор по описанию (шаг 2), выбор после двусторонней проверки (шаг 3), [потолок], README.txt и
spectrograms.png.
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

from spectrogram_compare import compare  # noqa: E402
from app.eval.blind_fingerprint_net import SR, load_model  # noqa: E402
from app.eval.plugin_adapter import sm_render  # noqa: E402
from app.eval.reference_match import RELIABLE, blind_estimate, preselect, symmetric_pick  # noqa: E402
from app.mixing.chain_optimizer import normalize_lufs, select_excerpt, to_stereo  # noqa: E402

PRESETS = Path("data/runs/fingerprint/presets/presets.jsonl")
NET = Path("data/runs/fingerprint/blind_net/blind_net.pt")
LUFS = -18.0
ECHO_PROB_MIN = 0.5  # ниже — сеть не слышит повторов, дилеи в кандидаты не берутся
MIX_FACTORS = (0.5, 0.25)


def load(path: Path, seconds: float | None = None) -> np.ndarray:
    import librosa
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = to_stereo(y[:, 0] if y.shape[1] == 1 else y.T)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return select_excerpt(y, SR, seconds) if seconds else y


def loud(y: np.ndarray) -> np.ndarray:
    y = normalize_lufs(y, SR, LUFS)
    peak = float(np.max(np.abs(y)))
    return y * (0.89 / peak) if peak > 0.89 else y


def _render_job(args: tuple[dict, np.ndarray]) -> np.ndarray:
    params, x = args
    return sm_render(params, x, SR)


def run_match(dry: np.ndarray, ref: np.ndarray, truth: np.ndarray | None, out: Path, model, presets: list[dict],
              k: int = 24, workers: int = 18, header: list[str] | None = None, refine: int = 0) -> dict:
    """
    Подбор для одной пары (массивы (2, n) при SR, truth — истина для этого же сухого или None). Пишет wav, README и
    спектрограммы в out; возвращает сводку (выборы и log-mel, если есть истина).

    refine > 0 — второй круг: дилеи (ECHOES) не берутся, если сеть не слышит повторов; refine лучших после
    двусторонней проверки рендерятся еще с mix ×0.5 и ×0.25, и в выбор входит сухой без эффекта. В инженерных миксах
    ревер на вокале часто тише родного mix пресета, а без этих вариантов любой выбор добавляет лишнее.
    """
    dry_in = normalize_lufs(dry, SR, LUFS)
    target = blind_estimate(model, ref)
    pool = list(range(len(presets)))
    if refine and target["echo_prob"] < ECHO_PROB_MIN:
        pool = [i for i in pool if presets[i]["category"] != "ECHOES"]
    cand = [pool[j] for j in preselect([presets[i]["fingerprint"] for i in pool], target, k)]
    labels = [f"{presets[i]['category']}/{presets[i]['name']}" for i in cand]
    with ProcessPoolExecutor(workers) as ex:
        renders = list(ex.map(_render_job, [({"raw": presets[i]["raw"]}, dry_in) for i in cand]))
        best, dists, _ = symmetric_pick(model, renders, target)
        if refine:
            top = [int(i) for i in np.argsort(dists)[:refine]]
            jobs = [(cand[i], f) for i in top for f in MIX_FACTORS]
            renders += list(ex.map(_render_job, [({"raw": presets[i]["raw"] | {"mix": presets[i]["raw"]["mix"] * f}},
                                                  dry_in) for i, f in jobs]))
            labels += [f"{presets[i]['category']}/{presets[i]['name']} mix×{f:g}" for i, f in jobs]
            renders.append(dry_in)
            labels.append("без эффекта")
            best, dists, _ = symmetric_pick(model, renders, target)

    import soundfile as sf
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.wav"):
        old.unlink()
    name = labels.__getitem__
    files = {"1_dry.wav": dry, "2_reference.wav": ref, "3_by_description.wav": renders[0],
             "4_symmetric_check.wav": renders[best]}
    summary = {"by_description": name(0), "symmetric": name(best), "symmetric_rank": best + 1,
               "target": {k_: target[k_] for k_ in RELIABLE}}
    lines = list(header or []) + [
        "Описание референса (слепая сеть): " + ", ".join(f"{k_}={target[k_]:.2f}" for k_ in RELIABLE),
        f"  период повторов L/R {target['echo_period_l_ms']:.0f}/{target['echo_period_r_ms']:.0f} мс, "
        f"вероятность повторов {target['echo_prob']:.2f}", "",
        f"3_by_description: {name(0)}", f"4_symmetric_check: {name(best)} (место по описанию {best + 1})"]
    if truth is not None:
        from real_pairs import logmel_db
        errs = [logmel_db(loud(r), loud(truth)) for r in renders]
        oracle = int(np.argmin(errs))
        files["5_best_of_candidates_by_truth.wav"] = renders[oracle]
        files["6_truth.wav"] = truth
        summary |= {"logmel_dry": logmel_db(loud(dry), loud(truth)), "logmel_by_description": errs[0],
                    "logmel_symmetric": errs[best], "logmel_oracle": errs[oracle],
                    "logmel_median": float(np.median(errs)), "oracle": name(oracle)}
        lines += [f"5_best_of_candidates_by_truth: {name(oracle)}", "",
                  f"log-mel к истине (дБ): сухой {summary['logmel_dry']:.2f}, по описанию {errs[0]:.2f}, "
                  f"двусторонняя {errs[best]:.2f}, лучший из {len(renders)} {errs[oracle]:.2f}, "
                  f"медиана кандидатов {summary['logmel_median']:.2f}"]
    lines += ["", "кандидаты (по описанию; расстояние описаний рендера и референса):"]
    lines += [f"  {i + 1:>2}. {name(i):<56} {dists[i]:.3f}" for i in range(len(renders))]
    for fname, y in files.items():
        sf.write(out / fname, loud(y).T, SR, subtype="PCM_24")
    (out / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    compare(out)
    return summary


def load_presets() -> list[dict]:
    return [json.loads(line) for line in PRESETS.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry", type=Path, required=True)
    parser.add_argument("--ref", type=Path, required=True)
    parser.add_argument("--truth", type=Path, help="истинный мокрый для этого же сухого (только для оценки)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", type=int, default=24, help="сколько пресетов проверять рендером")
    parser.add_argument("--seconds", type=float, default=30.0, help="длина фрагмента сухого")
    parser.add_argument("--workers", type=int, default=18)
    args = parser.parse_args()

    ref = load(args.ref)
    if args.truth:  # с истиной сухой и истина режутся одним окном, чтобы log-mel сравнивал одно и то же
        from real_pairs import align, find_lag
        dry, truth = load(args.dry), load(args.truth)
        lag, _, _ = find_lag(dry, truth)
        dry, truth = align(dry, truth, lag)
        n = min(dry.shape[1], int(args.seconds * SR))
        dry, truth = dry[:, :n], truth[:, :n]
    else:
        dry, truth = load(args.dry, args.seconds), None
    run_match(dry, ref, truth, args.out, load_model(NET), load_presets(), args.k, args.workers,
              header=[f"сухой {args.dry}", f"референс {args.ref}"])


if __name__ == "__main__":
    main()
