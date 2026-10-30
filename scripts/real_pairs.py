"""
Бенчмарк на реальных парах «сухой дубль ↔ тот же дубль после обработки инженером» (Cambridge multitracks).

В отличие от синтетических пар здесь есть истинная цель: обработанный файл того же дубля. Поэтому кроме
style_distance(результат, wet) считается прямая ошибка относительно выровненного по времени wet того же
фрагмента (многоразрешающий STFT и log-mel) — она валидна и когда референсом служит другая часть песни.

Варианты пар:
  - same: исходник = сухой фрагмент A, референс = wet того же фрагмента A (легкий случай);
  - cross: исходник = сухой фрагмент A, референс = wet другого фрагмента B той же песни (ближе к реальной работе).

Пример:
    python scripts/real_pairs.py --analyze
    python scripts/real_pairs.py --budget 60 --patience 60
    python scripts/real_pairs.py --metric embedding --modes model+optimizer --out data/runs/benchmark/real_pairs_v2.jsonl         --listen-dir data/processed/real_pairs_v2
"""
import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

sys.path.append(str(Path(__file__).parent.parent.absolute()))

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import pyloudnorm as pyln  # noqa: E402
import soundfile as sf  # noqa: E402

from app.eval.benchmark import MODES, BenchmarkCase, ChainBenchmark  # noqa: E402
from app.eval.style_metrics import extract_style_features, style_distance  # noqa: E402
from app.mixing.chain_optimizer import CHAIN_INPUT_LUFS, normalize_lufs, to_stereo  # noqa: E402

SR = 44100
CAMBRIDGE = Path("data/raw/dry_vocals/cambridge")
# (имя пары, сухой файл, обработанный файл)
PAIRS = [
    ("whiptails_vox", "Albert_Kader_Whiptails_11_Vox_Dry", "Albert_Kader_Whiptails_12_Vox_Wet"),
    ("ronnie_leadvox_tape", "Ronnie_Jones_Why_Dont_You_Stay_24_LeadVoxRaw",
     "Ronnie_Jones_Why_Dont_You_Stay_23_LeadVoxTapeEffect"),
    ("oxeye_leadvox", "oxeye_Sorry_13_LeadVox_NoPitchProcessing", "oxeye_Sorry_12_LeadVox"),
    ("oxeye_leadvox_dt", "oxeye_Sorry_15_LeadVoxDT_NoPitchProcessing", "oxeye_Sorry_14_LeadVoxDT"),
    ("oxeye_backing1", "oxeye_Sorry_17_BackingVox1_NoPitchProcessing", "oxeye_Sorry_16_BackingVox1"),
    ("oxeye_backing2", "oxeye_Sorry_19_BackingVox2_NoPitchProcessing", "oxeye_Sorry_18_BackingVox2"),
    ("oxeye_backing3", "oxeye_Sorry_21_BackingVox3_NoPitchProcessing", "oxeye_Sorry_20_BackingVox3"),
    ("oxeye_backing4", "oxeye_Sorry_23_BackingVox4_NoPitchProcessing", "oxeye_Sorry_22_BackingVox4"),
]
MAX_LAG_S = 0.5


def load(name: str) -> np.ndarray:
    y, _ = librosa.load(CAMBRIDGE / f"{name}.wav", sr=SR, mono=False)
    return to_stereo(y)


def lufs(y: np.ndarray) -> float:
    return float(pyln.Meter(SR).integrated_loudness(to_stereo(y).T))


def _xcorr_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Лаг b относительно a (>0 — b запаздывает) по максимуму |взаимной корреляции| и нормированный пик."""
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    corr = np.fft.irfft(np.fft.rfft(b, n) * np.conj(np.fft.rfft(a, n)), n)
    lags = np.concatenate([np.arange(0, max_lag + 1), np.arange(-max_lag, 0)])
    values = np.concatenate([corr[:max_lag + 1], corr[-max_lag:]])
    best = int(np.argmax(np.abs(values)))
    return int(lags[best]), float(values[best] / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def find_lag(dry: np.ndarray, wet: np.ndarray, max_lag_s: float = MAX_LAG_S) -> tuple[int, float, str]:
    """
    Сдвиг wet относительно dry в сэмплах (>0 — wet запаздывает), пик корреляции и способ оценки.

    Корреляция формы волны точна до сэмпла, но тюн и реверб разрушают фазу (у пар oxeye пик ~0.1 и ложный
    лаг до 45 мс). Поэтому при слабом пике (< 0.5) лаг берется по корреляции лог-огибающей (шаг 64 сэмпла).
    """
    lag, peak = _xcorr_lag(dry.mean(axis=0), wet.mean(axis=0), int(max_lag_s * SR))
    if abs(peak) >= 0.5:
        return lag, peak, "waveform"
    hop = 64

    def envelope(y):
        rms = librosa.feature.rms(y=y.mean(axis=0), frame_length=512, hop_length=hop)[0]
        env = np.log(rms + 1e-4)
        return env - env.mean()

    env_lag, env_peak = _xcorr_lag(envelope(dry), envelope(wet), int(max_lag_s * SR / hop))
    return env_lag * hop, env_peak, "envelope"


def align(dry: np.ndarray, wet: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Сдвигает wet на lag и обрезает оба сигнала до общей длины."""
    if lag > 0:
        wet = wet[:, lag:]
    elif lag < 0:
        dry = dry[:, -lag:]
    n = min(dry.shape[1], wet.shape[1])
    return dry[:, :n], wet[:, :n]


def active_halves(y: np.ndarray) -> tuple[slice, slice]:
    """Делит трек на две части с примерно равной энергией голоса: A — до медианы энергии, B — после."""
    energy = np.cumsum(y.mean(axis=0) ** 2)
    mid = int(np.searchsorted(energy, energy[-1] / 2))
    return slice(0, mid), slice(mid, y.shape[1])


def mrstft_db(x: np.ndarray, y: np.ndarray) -> float:
    """Многоразрешающая ошибка лог-спектра (дБ, среднее |Δ| по активным бинам) по Mid, окна 512/2048/8192."""
    a, b = x.mean(axis=0), y.mean(axis=0)
    errs = []
    for n_fft in (512, 2048, 8192):
        sa = 20 * np.log10(np.abs(librosa.stft(a, n_fft=n_fft, hop_length=n_fft // 4)) + 1e-5)
        sb = 20 * np.log10(np.abs(librosa.stft(b, n_fft=n_fft, hop_length=n_fft // 4)) + 1e-5)
        floor = sb.max() - 80.0
        errs.append(float(np.mean(np.abs(np.maximum(sa, floor) - np.maximum(sb, floor)))))
    return float(np.mean(errs))


def logmel_db(x: np.ndarray, y: np.ndarray) -> float:
    """Ошибка log-mel (дБ, 80 полос, окно 2048) по каждому каналу отдельно — учитывает и стерео."""
    errs = []
    for ca, cb in zip(to_stereo(x), to_stereo(y)):
        ma = librosa.power_to_db(librosa.feature.melspectrogram(y=ca, sr=SR, n_fft=2048, hop_length=512, n_mels=80),
                                 ref=1.0, amin=1e-10)
        mb = librosa.power_to_db(librosa.feature.melspectrogram(y=cb, sr=SR, n_fft=2048, hop_length=512, n_mels=80),
                                 ref=1.0, amin=1e-10)
        floor = mb.max() - 80.0
        errs.append(float(np.mean(np.abs(np.maximum(ma, floor) - np.maximum(mb, floor)))))
    return float(np.mean(errs))


F0_SR, F0_HOP = 22050, 256


def f0_track(y: np.ndarray, seconds: float = 40.0) -> np.ndarray:
    """
    F0 по кадрам (NaN — не озвучено). Вокализация от pyin, частота от yin (pyin квантует F0 по 10 центов,
    а разница тюна измеряется единицами центов), кадры с расхождением > 60 центов отбрасываются — как в
    style_metrics.pitch_grid_deviation_cents.
    """
    mono = librosa.resample(to_stereo(y).mean(axis=0)[: int(seconds * SR)], orig_sr=SR, target_sr=F0_SR)
    f0_pyin, voiced, prob = librosa.pyin(mono, fmin=65.0, fmax=1000.0, sr=F0_SR, frame_length=2048,
                                         hop_length=F0_HOP, resolution=0.5)
    f0_yin = librosa.yin(mono, fmin=65.0, fmax=1000.0, sr=F0_SR, frame_length=2048, hop_length=F0_HOP)
    n = min(len(f0_pyin), len(f0_yin))
    agree = np.abs(1200.0 * np.log2(np.nan_to_num(f0_pyin[:n], nan=1.0) / f0_yin[:n])) < 60.0
    return np.where(voiced[:n] & (prob[:n] > 0.7) & agree, f0_yin[:n], np.nan)


def f0_difference_cents(f0_a: np.ndarray, f0_b: np.ndarray, lag_frames: int = 0) -> np.ndarray:
    """|ΔF0| в центах на кадрах, озвученных в обоих треках (b сдвинут на lag_frames), без октавных ошибок."""
    if lag_frames > 0:
        f0_b = f0_b[lag_frames:]
    elif lag_frames < 0:
        f0_a = f0_a[-lag_frames:]
    n = min(len(f0_a), len(f0_b))
    both = ~np.isnan(f0_a[:n]) & ~np.isnan(f0_b[:n])
    diff = np.abs(1200.0 * np.log2(f0_b[:n][both] / f0_a[:n][both]))
    return diff[diff < 300.0]


def direct_errors(result: np.ndarray, target: np.ndarray, target_f0: np.ndarray | None = None) -> dict:
    """
    Прямая ошибка результата против истинного wet того же фрагмента после выравнивания громкости и сдвига:
    mrstft_db / logmel_db — средняя |Δ| лог-спектра в дБ, f0_err_cents — медиана |ΔF0| (ошибка тюна).
    """
    result = normalize_lufs(to_stereo(result), SR, CHAIN_INPUT_LUFS)
    target = normalize_lufs(to_stereo(target), SR, CHAIN_INPUT_LUFS)
    # Задержка, которую могла внести цепочка (дилей, PSOLA, фазовый EQ)
    lag, _, _ = find_lag(target, result, max_lag_s=0.1)
    target_aligned, result_aligned = align(target, result, lag)
    target_f0 = f0_track(target) if target_f0 is None else target_f0
    f0_diff = f0_difference_cents(target_f0, f0_track(result), int(round(lag / SR * F0_SR / F0_HOP)))
    return {"mrstft_db": mrstft_db(result_aligned, target_aligned), "logmel_db": logmel_db(result_aligned, target_aligned),
            "f0_err_cents": float(np.median(f0_diff)) if len(f0_diff) >= 20 else float("nan"),
            "lag_ms": lag / SR * 1e3}


def pitch_compare(dry: np.ndarray, wet: np.ndarray) -> dict:
    """
    Стабильность тона: насколько wet сдвинут относительно dry по F0 на общих озвученных кадрах
    и какая доля кадров каждого сигнала лежит ближе 10 центов к полутоновой сетке (после общей расстройки).
    """
    fd, fw = f0_track(dry), f0_track(wet)
    shift = f0_difference_cents(fd, fw)

    def near_grid(track):
        cents = 1200 * np.log2(track[~np.isnan(track)] / 440.0)
        if len(cents) < 20:
            return float("nan")
        offset = ((cents + 50) % 100) - 50
        offset = ((offset - np.median(offset) + 50) % 100) - 50
        return float(np.mean(np.abs(offset) < 10))

    return {"f0_shift_median_abs_cents": float(np.median(shift)) if len(shift) >= 20 else float("nan"),
            "f0_shift_p90_abs_cents": float(np.percentile(shift, 90)) if len(shift) >= 20 else float("nan"),
            "near_grid_10c_dry": near_grid(fd), "near_grid_10c_wet": near_grid(fw)}


def describe(dry: np.ndarray, wet: np.ndarray) -> dict:
    """Чем отличается wet от dry: громкость, спектр, динамика, тюн, стерео (те же признаки, что у метрики)."""
    fd = extract_style_features(normalize_lufs(dry, SR, CHAIN_INPUT_LUFS), SR)
    fw = extract_style_features(normalize_lufs(wet, SR, CHAIN_INPUT_LUFS), SR)
    distance, per_feature = style_distance(fd, fw)
    diff_ltas = fw["ltas_db"] - fd["ltas_db"]
    side = wet[0] - wet[1]
    return {
        "lufs_dry": lufs(dry), "lufs_wet": lufs(wet),
        "style_distance_dry_vs_wet": distance,
        "per_feature": {k: round(v, 2) for k, v in per_feature.items()},
        "ltas_diff_db_63_250_1k_4k_10k": [round(float(diff_ltas[i]), 1) for i in (0, 6, 12, 18, 24)],
        "pitch_cents_dry": fd["pitch_cents"], "pitch_cents_wet": fw["pitch_cents"],
        "crest_dry": fd["crest_db"], "crest_wet": fw["crest_db"],
        "decay_fast_dry": fd["decay_fast_db"], "decay_fast_wet": fw["decay_fast_db"],
        "stereo_side_to_mid_db": float(20 * np.log10(np.std(side) / (np.std(wet[0] + wet[1]) + 1e-12) + 1e-12)),
        "direct_dry_vs_wet": direct_errors(dry, wet),
        **pitch_compare(dry, wet),
    }


def analyze() -> dict:
    report = {}
    for name, dry_name, wet_name in PAIRS:
        dry, wet = load(dry_name), load(wet_name)
        lag, peak, method = find_lag(dry, wet)
        dry_a, wet_a = align(dry, wet, lag)
        info = {"lag_samples": lag, "xcorr_peak": peak, "lag_method": method, **describe(dry_a, wet_a)}
        report[name] = info
        print(name, json.dumps(info, ensure_ascii=False, default=float))
    return report


def build_cases(name: str, dry_name: str, wet_name: str) -> list[tuple[BenchmarkCase, np.ndarray]]:
    """Пары same и cross для одной реальной пары. Возвращает (case, истинный wet фрагмента исходника)."""
    from app.audio.vocal_enhancer import VocalEnhancer
    dry, wet = load(dry_name), load(wet_name)
    lag, _, _ = find_lag(dry, wet)
    dry, wet = align(dry, wet, lag)
    part_a, part_b = active_halves(dry)
    src = normalize_lufs(dry[:, part_a], SR, CHAIN_INPUT_LUFS)
    target = wet[:, part_a]
    enhancer = VocalEnhancer(sr=SR)
    cases = []
    for variant, ref in (("same", wet[:, part_a]), ("cross", wet[:, part_b])):
        key_index, scale = enhancer.detect_key_and_scale(librosa.to_mono(ref))
        cases.append((BenchmarkCase(f"{name}__{variant}", src, ref, key_index, scale), target))
    return cases


def run(args) -> None:
    from app.ml.pipeline import MixingAIMLPipeline
    pipeline = MixingAIMLPipeline(args.model_path)
    bench = ChainBenchmark(sr=SR, prior=pipeline.predict_prior, metric=args.metric)
    encoder = None
    if Path(args.encoder).exists():
        from app.eval.style_embedding import StyleEncoder
        encoder = StyleEncoder(args.encoder)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    listen_root = Path(args.listen_dir)
    pairs = [p for p in PAIRS if not args.only or p[0] in args.only]
    with open(out, "a" if args.only else "w", encoding="utf-8") as f:
        for name, dry_name, wet_name in pairs:
            cases = build_cases(name, dry_name, wet_name)
            target_f0 = f0_track(cases[0][1])  # истинный wet фрагмента исходника общий для same и cross
            for case, target in cases:
                variant = case.name.split("__")[1]
                audio_dir = Path(args.audio_tmp) / case.name
                rows = bench.run_case(case, tuple(args.modes), budget_s=args.budget, seed=0, audio_dir=audio_dir,
                                      patience_renders=args.patience)
                # Потолок метрики: истинный wet фрагмента исходника против референса
                ref_features = extract_style_features(case.ref, SR)
                wet_distance, wet_per_feature = style_distance(extract_style_features(target, SR), ref_features)
                rows.append({"case": case.name, "mode": "wet_true", "seed": 0, "distance": wet_distance,
                             "per_feature": wet_per_feature, "plugins": [], "history": []})
                ref_embedding = encoder.embed(case.ref, SR) if encoder is not None else None
                for row in rows:
                    y = (target if row["mode"] == "wet_true" else
                         sf.read(audio_dir / f"{case.name}_{row['mode'].replace('+', '_')}_s0.wav")[0].T)
                    row.update({"pair": name, "variant": variant, "dry_file": dry_name, "wet_file": wet_name,
                                "metric": args.metric, **direct_errors(y, target, target_f0)})
                    if encoder is not None:
                        row["embedding_distance"] = encoder.distance(encoder.embed(y, SR), ref_embedding)
                    row.pop("history", None)
                    f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
                    f.flush()
                    print(f"    {row['mode']:16s} style {row['distance']:.2f} | mrstft {row['mrstft_db']:.2f} дБ"
                          f" | logmel {row['logmel_db']:.2f} дБ | f0 {row['f0_err_cents']:.1f} ц")
                # Основная папка — cross (реальный сценарий), same_take/ — легкий случай
                folder = listen_root / name if variant == "cross" else listen_root / name / "same_take"
                compare_dir = Path(args.compare_audio_tmp) / case.name if args.compare_audio_tmp else None
                write_listening(folder, audio_dir, case, target, rows, compare_dir)


def chain_text(chain: list[dict]) -> str:
    """Цепочка строкой: плагин(параметр=нормализованное значение, ...)."""
    return "; ".join(f"{p['name']}(" + ", ".join(f"{k}={v:.2f}" for k, v in p["params"].items()) + ")"
                     for p in chain)


def write_listening(folder: Path, audio_dir: Path, case: BenchmarkCase, target: np.ndarray, rows: list[dict],
                    compare_dir: Path | None = None):
    """
    Папка прослушивания: все файлы на одной громкости (-18 LUFS; при перегрузе пик опускается до -0.1 дБFS).
    compare_dir — рендеры прошлого прогона (например, с метрикой style) для A/B: файлы с суффиксом _style.
    """
    folder.mkdir(parents=True, exist_ok=True)
    tracks = {"dry": case.src, "wet_true": target}
    for mode, file_mode in (("model", "model"), ("model+optimizer", "model_optimizer"),
                            ("optimizer", "optimizer_only")):
        path = audio_dir / f"{case.name}_{mode.replace('+', '_')}_s0.wav"
        if path.exists():
            tracks[file_mode] = sf.read(path)[0].T
    if compare_dir is not None:
        for mode, file_mode in (("model+optimizer", "model_optimizer"), ("optimizer", "optimizer_only")):
            path = compare_dir / f"{case.name}_{mode.replace('+', '_')}_s0.wav"
            if path.exists():
                tracks[f"{file_mode}_style"] = sf.read(path)[0].T
    tracks["reference_used"] = case.ref
    for file_mode, y in tracks.items():
        y = normalize_lufs(to_stereo(np.asarray(y, dtype=np.float32)), SR, -18.0)
        peak = np.max(np.abs(y))
        sf.write(folder / f"{file_mode}.wav", (y / max(peak, 1.0) * 0.99 if peak > 0.99 else y).T, SR)
    with open(folder / "chains.txt", "w", encoding="utf-8") as f:
        for row in rows:
            if row.get("chain"):
                f.write(f"[{row['mode']}] style {row['distance']:.2f} | mrstft {row['mrstft_db']:.2f} дБ | "
                        f"logmel {row['logmel_db']:.2f} дБ | f0 {row['f0_err_cents']:.1f} ц\n"
                        f"  {chain_text(row['chain'])}\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Бенчмарк на реальных парах dry/wet")
    parser.add_argument("--analyze", action="store_true", help="Только анализ пар (сдвиг, громкость, отличия)")
    parser.add_argument("--model-path", default="data/models/v5/best_model.pth")
    parser.add_argument("--budget", type=float, default=60.0)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--only", nargs="*", default=None, help="Прогнать только эти пары (дописать в jsonl)")
    parser.add_argument("--out", default="data/runs/benchmark/real_pairs.jsonl")
    parser.add_argument("--audio-tmp", default=None,
                        help="Папка рендеров (по умолчанию real_pairs_audio, для другой метрики — с ее суффиксом)")
    parser.add_argument("--listen-dir", default="data/processed/real_pairs")
    parser.add_argument("--metric", choices=["style", "embedding", "combined", "embedding_tail"], default="style",
                        help="Целевая функция подбора (ChainOptimizer metric)")
    parser.add_argument("--compare-audio-tmp", default=None,
                        help="Рендеры прошлого прогона для A/B в папке прослушивания (файлы *_style.wav)")
    parser.add_argument("--modes", nargs="*", default=list(MODES), choices=list(MODES))
    parser.add_argument("--encoder", default="data/models/style_encoder/style_encoder.pt",
                        help="Чекпоинт кодировщика стиля для колонки embedding_distance (если есть)")
    parsed = parser.parse_args()
    if parsed.audio_tmp is None:
        parsed.audio_tmp = "data/runs/benchmark/real_pairs_audio" + ("" if parsed.metric == "style" else f"_{parsed.metric}")
    if parsed.analyze:
        result = analyze()
        Path("data/runs/benchmark").mkdir(parents=True, exist_ok=True)
        Path("data/runs/benchmark/real_pairs_analysis.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    else:
        run(parsed)
