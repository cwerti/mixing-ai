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
from app.eval.reference_match import RELIABLE, blind_estimate, description_distance, preselect  # noqa: E402
from app.eval.tail_decay import (  # noqa: E402
    TAIL_UNIT_DB, decay_profile, phrase_ends, probe_segment, profile_at, profile_distance,
)
from app.mixing.chain_optimizer import normalize_lufs, select_excerpt, to_stereo  # noqa: E402

PRESETS = Path("data/runs/fingerprint/presets/presets.jsonl")
NET = Path("data/runs/fingerprint/blind_net/blind_net.pt")
LUFS = -18.0
ECHO_PROB_MIN = 0.5  # ниже — сеть не слышит повторов, дилеи в кандидаты не берутся
MIX_FACTORS = (0.5, 0.25)
# Supermassive сам многопоточный: на 20 ядрах 12 рендеров по 30 с идут 26 с в 12 процессах и 12 с в 3-4
SM_PROCESSES = 4


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


_worker_model = None


def _init_worker(model, threads: int) -> None:
    """Слепая сеть в каждом процессе: рендер и его описание считаются там же, где рендер."""
    import torch
    global _worker_model
    torch.set_num_threads(threads)
    _worker_model = model


def _render_job(args: tuple[dict, np.ndarray, tuple[np.ndarray, int] | None]
                ) -> tuple[np.ndarray, dict, np.ndarray | None]:
    """Рендер, его описание сетью и профиль хвоста на пробе от ее конца фразы (если проба есть)."""
    params, x, probe = args
    y = sm_render(params, x, SR)
    tail = profile_at(sm_render(params, probe[0], SR), SR, probe[1]) if probe is not None else None
    return y, blind_estimate(_worker_model, y), tail


def tail_probe(dry: np.ndarray) -> np.ndarray:
    """
    Проба хвоста: кусок сухого с концом фразы перед паузой; если пауз нет — самый громкий кусок 3 с, оборванный
    в тишину 2.5 с (короткое затухание, чтобы не щелкало).
    """
    probe = probe_segment(dry, SR)
    if probe is not None:
        return probe
    voice = select_excerpt(dry, SR, 3.0).copy()
    fade = int(0.02 * SR)
    voice[:, -fade:] *= np.linspace(1.0, 0.0, fade, dtype=voice.dtype)
    return np.concatenate([voice, np.zeros((voice.shape[0], int(2.5 * SR)), dtype=voice.dtype)], axis=1)


def _scores(descs: list[dict], tails: list[np.ndarray | None], target: dict,
            tail_target: np.ndarray | None) -> tuple[int, list[float], list[float | None]]:
    """Двусторонняя проверка: расстояние описаний плюс ошибка профиля хвоста в единицах TAIL_UNIT_DB."""
    dists, tail_errs = [], []
    for desc, tail in zip(descs, tails):
        err = profile_distance(tail, tail_target) if tail is not None and tail_target is not None else None
        tail_errs.append(err)
        dists.append(description_distance(desc, target) + (err / TAIL_UNIT_DB if err is not None else 0.0))
    return int(np.argmin(dists)), dists, tail_errs


def run_match(dry: np.ndarray, ref: np.ndarray, truth: np.ndarray | None, out: Path, model, presets: list[dict],
              k: int = 24, workers: int = 18, header: list[str] | None = None, refine: int = 0,
              spectrograms: bool = True, probe: np.ndarray | None = None) -> dict:
    """
    Подбор для одной пары (массивы (2, n) при SR, truth — истина для этого же сухого или None). Пишет wav, README и
    спектрограммы в out; возвращает сводку (выборы и log-mel, если есть истина).

    spectrograms=False — не строить spectrograms.png (вызывающий допишет файлы и построит сам).

    Если у референса есть концы фраз перед паузами, в проверку входит профиль хвоста (app.eval.tail_decay): кандидат
    рендерится еще и на пробе (probe или tail_probe(dry)) — куске с концом фразы. Сеть смотрит окна внутри фраз и
    может недооценить хвост, тогда без этого выбирается короткий «амбиенс».

    refine > 0 — второй круг: дилеи (ECHOES) не берутся, если сеть не слышит повторов; refine лучших после
    двусторонней проверки рендерятся еще с mix ×0.5 и ×0.25, и в выбор входит сухой без эффекта. В инженерных миксах
    ревер на вокале часто тише родного mix пресета, а без этих вариантов любой выбор добавляет лишнее.
    """
    dry_in = normalize_lufs(dry, SR, LUFS)
    target = blind_estimate(model, ref)
    tail_target = decay_profile(ref, SR)
    probe_in = None
    if tail_target is not None:
        x = normalize_lufs(probe if probe is not None else tail_probe(dry), SR, LUFS)
        ends = phrase_ends(x, SR)
        probe_in = (x, ends[0][0]) if ends else None
    pool = list(range(len(presets)))
    if refine and target["echo_prob"] < ECHO_PROB_MIN:
        pool = [i for i in pool if presets[i]["category"] != "ECHOES"]
    cand = [pool[j] for j in preselect([presets[i]["fingerprint"] for i in pool], target, k)]
    labels = [f"{presets[i]['category']}/{presets[i]['name']}" for i in cand]
    picks: list[tuple[int, float] | None] = [(i, 1.0) for i in cand]  # (пресет, множитель mix); None — без эффекта
    processes = max(1, min(workers, SM_PROCESSES))
    threads = max(1, workers // processes)
    with ProcessPoolExecutor(processes, initializer=_init_worker, initargs=(model, threads)) as ex:
        renders, descs, tails = map(list, zip(*ex.map(
            _render_job, [({"raw": presets[i]["raw"]}, dry_in, probe_in) for i in cand])))
        best, dists, tail_errs = _scores(descs, tails, target, tail_target)
        if refine:
            top = [int(i) for i in np.argsort(dists)[:refine]]
            jobs = [(cand[i], f) for i in top for f in MIX_FACTORS]
            for y, desc, tail in ex.map(_render_job, [
                    ({"raw": presets[i]["raw"] | {"mix": presets[i]["raw"]["mix"] * f}}, dry_in, probe_in)
                    for i, f in jobs]):
                renders.append(y)
                descs.append(desc)
                tails.append(tail)
            labels += [f"{presets[i]['category']}/{presets[i]['name']} mix×{f:g}" for i, f in jobs]
            picks += jobs
            renders.append(dry_in)
            descs.append(blind_estimate(model, dry_in))
            tails.append(profile_at(probe_in[0], SR, probe_in[1]) if probe_in is not None else None)
            labels.append("без эффекта")
            picks.append(None)
            best, dists, tail_errs = _scores(descs, tails, target, tail_target)

    import soundfile as sf
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.wav"):
        old.unlink()
    name = labels.__getitem__
    files = {"1_dry.wav": dry, "2_reference.wav": ref, "3_by_description.wav": renders[0],
             "4_symmetric_check.wav": renders[best]}
    chosen = picks[best]
    summary = {"by_description": name(0), "symmetric": name(best), "symmetric_rank": best + 1,
               "symmetric_preset": None if chosen is None else presets[chosen[0]]["path"],
               "symmetric_mix_factor": None if chosen is None else chosen[1],
               "symmetric_raw": None if chosen is None else
               presets[chosen[0]]["raw"] | {"mix": presets[chosen[0]]["raw"]["mix"] * chosen[1]},
               "target": {k_: target[k_] for k_ in RELIABLE},
               "tail_target": None if tail_target is None else [float(v) for v in tail_target],
               "symmetric_tail_error": tail_errs[best]}
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
                    "logmel_median": float(np.median(errs)), "oracle": name(oracle),
                    "candidates": [{"label": name(i), "description": description_distance(descs[i], target),
                                    "tail_error": tail_errs[i], "logmel": errs[i]} for i in range(len(renders))]}
        lines += [f"5_best_of_candidates_by_truth: {name(oracle)}", "",
                  f"log-mel к истине (дБ): сухой {summary['logmel_dry']:.2f}, по описанию {errs[0]:.2f}, "
                  f"двусторонняя {errs[best]:.2f}, лучший из {len(renders)} {errs[oracle]:.2f}, "
                  f"медиана кандидатов {summary['logmel_median']:.2f}"]
    if tail_target is not None:
        lines += ["", "Хвост референса после конца фразы (дБ к уровню фразы через 0.1/0.3/0.6/1.0 с): Mid "
                  + " ".join(f"{v:.0f}" for v in tail_target[:4]) + ", Side " + " ".join(f"{v:.0f}" for v in tail_target[4:])]
    else:
        lines += ["", "У референса нет пауз после фраз — хвост в выборе не участвует"]
    lines += ["", "кандидаты (по описанию; итог проверки = расстояние описаний + ошибка хвоста / "
              f"{TAIL_UNIT_DB:.0f} дБ):"]
    lines += [f"  {i + 1:>2}. {name(i):<56} {dists[i]:.3f}"
              + (f"  хвост {tail_errs[i]:.1f} дБ" if tail_errs[i] is not None else "") for i in range(len(renders))]
    for fname, y in files.items():
        sf.write(out / fname, loud(y).T, SR, subtype="PCM_24")
    (out / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if spectrograms:
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
