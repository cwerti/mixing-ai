"""
Полная цепочка альфы: тон и динамика — наш DSP (EQ, тюн, компрессор, де-эссер…, подбор CMA-ES без ревера и
дилея), пространство — настоящий Valhalla Supermassive (слепая сеть → пресеты → двусторонняя проверка с
уменьшенным mix). Supermassive открывается в FL Studio тем же плагином, поэтому пространство в превью и в FL
совпадает; тон пока рендерится нашим DSP (переезд на VST3 — следующий шаг).

    python scripts/mix_full.py --dry vocal.wav --ref reference_vocal.wav --out data/processed/mix/<имя>

Референс — уже выделенный вокал (для полного микса: scripts/process_vocal.py --extract-vocals или Demucs).
В папке результата: preview.wav (итог), tone.wav (без пространства), supermassive.vpreset, chain.json,
README.txt, spectrograms.png и файлы двусторонней проверки.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.mixing.chain_optimizer import CHAIN_INPUT_LUFS, normalize_lufs, to_stereo  # noqa: E402

SR = 44100
SPACE_PLUGINS = ("reverb", "delay")  # подбираются Supermassive после тона
# Не подбираются в тоне. Расширитель на моно-вокале до ревера копирует всю гармонику в Side (Side/Mid тона
# −15 дБ против −25…−30 у истины на Cambridge), а ширину и так дает Supermassive. Эксайтер ухудшил log-mel тона
# в среднем на +1.4 дБ (6 пар из 62).
TONE_EXCLUDE = SPACE_PLUGINS + ("stereo_enhancer", "exciter")
MODEL = Path("data/models/v6/best_model.pth")
MAX_RENDERS = 400  # по числу рендеров, а не по времени: итог не зависит от загрузки машины


def tone_prior(pipeline, dry: np.ndarray, ref: np.ndarray) -> dict:
    """Стартовая цепочка модели и тональность референса (считается в главном процессе: модель на GPU)."""
    import librosa
    from app.audio.vocal_enhancer import VocalEnhancer
    key_index, scale = VocalEnhancer(sr=SR).detect_key_and_scale(librosa.to_mono(ref))
    probs, params, std = pipeline.predict_prior(dry, ref)
    return {"model_probs": probs, "model_params": params, "model_param_std": std,
            "key_index": key_index, "scale": scale}


def tone_chain(dry: np.ndarray, ref: np.ndarray, prior: dict, seed: int = 0,
               max_renders: int = MAX_RENDERS) -> tuple[np.ndarray, dict]:
    """
    Тон и динамика без пространства. Возвращает рендер всего сухого (на уровне CHAIN_INPUT_LUFS, как в
    process_vocal.py) и описание цепочки для chain.json.
    """
    import librosa
    from app.audio.audio_processor import AudioProcessor
    from app.audio.dsp_engine import DSPEngine
    from app.audio.vocal_enhancer import VocalEnhancer
    from app.mixing.chain_optimizer import ChainOptimizer
    src = normalize_lufs(to_stereo(dry), SR, CHAIN_INPUT_LUFS)
    gate = AudioProcessor(sr=SR).estimate_noise_gate_threshold(librosa.to_mono(src))
    result = ChainOptimizer(sr=SR, metric="embedding", log=lambda _: None).optimize(
        src, ref, **prior, gate_threshold_db=gate, budget_s=float("inf"), seed=seed, patience_renders=60,
        max_renders=max_renders, exclude_plugins=TONE_EXCLUDE)
    enhancer = VocalEnhancer(sr=SR)
    cleaned = np.vstack([enhancer.suppress_resonances(ch) for ch in src]).astype(np.float32)
    y = DSPEngine().apply_chain(cleaned, SR, result.chain, gate_threshold_db=gate,
                                target_key_index=prior["key_index"], target_scale=prior["scale"])
    info = {"plugins": [{"name": p.name, "params": p.params} for p in result.chain.plugins],
            "eq_bands": [[float(f), float(g)] for f, g in result.eq_bands], "pitch_speed": result.pitch_speed,
            "gate_db": gate, "report": result.report}
    return y, info


def _tone_job(args: tuple) -> tuple[np.ndarray, dict]:
    return tone_chain(*args)


def full_chain(dry: np.ndarray, ref: np.ndarray, out: Path, *, tone: tuple[np.ndarray, dict], blind_model,
               presets: list[dict], truth: np.ndarray | None = None, workers: int = 18,
               header: list[str] | None = None) -> dict:
    """
    Пространство поверх готового тона и запись папки результата. tone — (рендер, описание) из tone_chain.
    Возвращает сводку run_match плюс log-mel сухого и тона к истине (если она есть).
    """
    import soundfile as sf
    from match_reference import loud, run_match
    from spectrogram_compare import compare
    from app.eval.plugin_adapter import write_vpreset
    y_tone, tone_info = tone
    summary = run_match(y_tone, ref, truth, out, blind_model, presets, k=24, workers=workers, header=header,
                        refine=6)
    # run_match назвал вход «1_dry» — здесь это тон; исходный сухой и итог пишутся рядом
    (out / "1_dry.wav").rename(out / "1_tone.wav")
    sf.write(out / "0_dry.wav", loud(dry).T, SR, subtype="PCM_24")
    final = out / "4_symmetric_check.wav"
    (out / "preview.wav").write_bytes(final.read_bytes())
    if summary["symmetric_preset"]:
        write_vpreset(summary["symmetric_preset"], out / "supermassive.vpreset", "Mixing-AI " + out.name,
                      summary["symmetric_mix_factor"])
    if truth is not None:
        from real_pairs import logmel_db
        summary["logmel_tone"] = summary.pop("logmel_dry")  # run_match мерил свой вход, то есть тон
        summary["logmel_dry"] = logmel_db(loud(dry), loud(truth))
    chain = {"tone": tone_info, "space": {"plugin": "Valhalla Supermassive", "preset": summary["symmetric"],
                                          "source": summary["symmetric_preset"],
                                          "mix_factor": summary["symmetric_mix_factor"]}}
    (out / "chain.json").write_text(json.dumps(chain, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    readme = out / "README.txt"
    tone_lines = [f"  {p['name']}: " + ", ".join(f"{k}={v:.2f}" for k, v in p["params"].items())
                  for p in tone_info["plugins"]]
    readme.write_text("\n".join(
        ["Полная цепочка Mixing-AI", "", "0_dry — исходный сухой, 1_tone — после тона, preview — итог "
         "(тон + Supermassive).", "", "Тон (наш DSP, нормализованные параметры):", *tone_lines, "",
         f"Пространство: Supermassive, {summary['symmetric']}",
         "  в FL: вставить Valhalla Supermassive после цепочки тона, Preset → Load → supermassive.vpreset"
         if summary["symmetric_preset"] else "  без эффекта", "", "--- двусторонняя проверка ---",
         readme.read_text(encoding="utf-8")]), encoding="utf-8")
    compare(out)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry", type=Path, required=True)
    parser.add_argument("--ref", type=Path, required=True, help="выделенный вокал референса")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=18)
    args = parser.parse_args()
    from match_reference import NET, load, load_presets
    from app.eval.blind_fingerprint_net import load_model
    from app.ml.pipeline import MixingAIMLPipeline
    dry, ref = load(args.dry), load(args.ref)
    prior = tone_prior(MixingAIMLPipeline(MODEL), dry, ref)
    tone = tone_chain(dry, ref, prior)
    full_chain(dry, ref, args.out, tone=tone, blind_model=load_model(NET), presets=load_presets(),
               workers=args.workers, header=[f"сухой {args.dry}", f"референс {args.ref}"])
    print(f"[+] {args.out / 'preview.wav'}")


if __name__ == "__main__":
    main()
