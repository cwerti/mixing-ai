"""
Честная проверка на замороженной половине реальных пар Cambridge (app/data/test_songs.json: песен нет ни в
train, ни в val). Исходник — сухой фрагмент A, референс — вокал микса из другого места B той же песни
(Demucs), цель — вокал микса на фрагменте A. Результат каждой модели сравнивается с целью по аспектам:

  тон        — средний |Δ| баланса полос (<150, 150–500, 2–5k, 5–10k, >10k относительно 150 Гц – 5 кГц), дБ
  динамика   — |Δ| crest и разброса громкости p95–p10, дБ
  сибилянты  — |Δ| sib_p90, дБ;  воздух — |Δ| air_ratio, дБ
  ширина     — |Δ| ширины на нотах и на хвостах, дБ
  простр.    — |Δ| спада после фраз (decay_fast, хвост через 0.3 с), дБ
  log-mel    — прямая ошибка спектрограммы относительно цели (дБ, оба на MODEL_LUFS)

Режим dry — без обработки (масштаб ошибок). Тюн не применяется: тональность песни неизвестна.

    python scripts/evaluate_real.py --models data/models/v6/best_model.pth data/models/v7_tone/best_model.pth
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.dataset.dataset_schema import CHAIN_ORDER, PLUGIN_ON_PROB  # noqa: E402
from app.mixing.chain_optimizer import normalize_lufs, to_stereo  # noqa: E402
from app.ml.features import MODEL_LUFS  # noqa: E402
from scripts.calibrate_generator import PAIRS, SR, _load, describe  # noqa: E402
from scripts.real_pairs import logmel_db  # noqa: E402

TEST_SONGS = Path("app/data/test_songs.json")
ASPECTS = {
    "тон": ("<150", "150-500", "2-5k", "5-10k", ">10k"),
    "динамика": ("crest_db", "dr_p95_p10_db"),
    "сибилянты": ("sib_p90_db",),
    "воздух": ("air_ratio_db",),
    "ширина": ("width_notes_db", "width_tails_db"),
    "простр.": ("decay_fast_db", "tail_mid_0.3s"),
}


def aspect_errors(result: np.ndarray, truth: np.ndarray, truth_desc: dict) -> dict[str, float]:
    d = describe(result)
    out = {a: float(np.nanmean([abs(d[k] - truth_desc[k]) for k in keys])) for a, keys in ASPECTS.items()}
    out["log-mel"] = logmel_db(normalize_lufs(result, SR, MODEL_LUFS), normalize_lufs(truth, SR, MODEL_LUFS))
    return out


def model_render(pipeline, matcher, dsp, enhancer, dry: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, list]:
    """Режим «только модель», как в app/eval/benchmark.py: EQ подгонкой спектра, остальное — сеть."""
    from app.ml.pipeline import model_chain, model_params_by_plugin
    _, probs, vec = pipeline.predict(dry, ref)
    plugins = [n for i, n in enumerate(CHAIN_ORDER) if n not in ("eq", "pitch_corrector") and probs[i] > PLUGIN_ON_PROB]
    chain = model_chain(plugins, matcher.match_eq(dry, ref), model_params_by_plugin(vec))
    cleaned = np.vstack([enhancer.suppress_resonances(ch) for ch in dry]).astype(np.float32)
    return dsp.apply_chain(cleaned, SR, chain), plugins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["data/models/v6/best_model.pth"])
    parser.add_argument("--out", default="data/runs/benchmark/real_test.jsonl")
    args = parser.parse_args()

    from app.audio.dsp_engine import DSPEngine
    from app.audio.audio_processor import SpectralMatcher
    from app.audio.vocal_enhancer import VocalEnhancer
    from app.ml.pipeline import MixingAIMLPipeline
    names = json.loads(TEST_SONGS.read_text(encoding="utf-8"))["pairs"]
    pipelines = {Path(m).parent.name: MixingAIMLPipeline(m) for m in args.models}
    matcher, dsp, enhancer = SpectralMatcher(sr=SR), DSPEngine(), VocalEnhancer(sr=SR)
    rows = []
    for name in names:
        folder = PAIRS / "pairs" / name
        dry = normalize_lufs(to_stereo(_load(folder / "dry_A.wav")), SR, MODEL_LUFS)
        ref, truth = _load(folder / "ref_B.wav"), _load(folder / "truth_A.wav")
        truth_desc = describe(truth)
        rows.append({"pair": name, "mode": "dry", **aspect_errors(dry, truth, truth_desc)})
        for tag, pipe in pipelines.items():
            y, plugins = model_render(pipe, matcher, dsp, enhancer, dry, ref)
            rows.append({"pair": name, "mode": tag, "plugins": plugins, **aspect_errors(y, truth, truth_desc)})
        print(name, " | ".join(f"{r['mode']} {r['log-mel']:.2f}" for r in rows if r["pair"] == name), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    cols = list(ASPECTS) + ["log-mel"]
    print(f"\n{len(names)} пар; медиана |ошибки| относительно настоящего микса (меньше — лучше)")
    print(f"{'режим':10s}" + "".join(f"{c:>11s}" for c in cols))
    for mode in ["dry", *pipelines]:
        sel = [r for r in rows if r["mode"] == mode]
        print(f"{mode:10s}" + "".join(f"{np.nanmedian([r[c] for r in sel]):11.2f}" for c in cols))
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
