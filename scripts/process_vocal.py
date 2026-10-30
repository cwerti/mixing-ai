"""
Mixing-AI: Скрипт автоматического сведения вокала по референсу (Style Transfer).
Анализирует сухой вокал и референс, прогоняет через обученную модель нейросети на GPU,
вычисляет оптимальные параметры обработки и генерирует:
1. Нативный пресет FL Studio (.fst) для переноса на дорожку микшера.
2. Готовый обработанный аудиофайл (.wav) для мгновенного прослушивания результата.
"""

import argparse
import sys
import shutil
import subprocess
from pathlib import Path
import torch
import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln

# Добавление корня проекта в пути импорта Python
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.audio.audio_processor import AudioProcessor, SpectralMatcher
from app.audio.dsp_engine import DSPEngine
from app.audio.vocal_enhancer import VocalEnhancer
from app.dataset.dataset_schema import (
    CHAIN_ORDER, DATAGEN_PLUGINS, PLUGIN_ON_PROB, ChainConfig, remap_param_vector
)
from app.eval.style_metrics import extract_style_features, style_distance
from app.mixing.chain_optimizer import CHAIN_INPUT_LUFS, ChainOptimizer, normalize_lufs, to_stereo
from app.ml.model import MixingAIModel
from app.ml.features import model_inputs
from app.ml.pipeline import load_param_std, model_chain, model_params_by_plugin
from app.presets.full_preset_generator import FullPresetGenerator

def extract_vocal_stem(reference_mix_path: Path, output_dir: Path) -> Path:
    """
    Извлекает изолированный сухой/мокрый вокал из полной песни-референса с помощью Demucs на GPU.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[*] Запуск нейросети Demucs для извлечения вокала из трека ({device.upper()})...")
    
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems=vocals",
        "-d", device,
        "-o", str(output_dir.absolute()),
        str(reference_mix_path.absolute())
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"[-] Ошибка запуска Demucs: {e}")
        sys.exit(1)
        
    stem_name = reference_mix_path.stem
    isolated_vocals = output_dir / "htdemucs" / stem_name / "vocals.wav"
    if not isolated_vocals.exists():
        # Резервный поиск
        found = list(output_dir.glob("**/vocals.wav"))
        if found:
            isolated_vocals = found[0]
        else:
            print(f"[-] Ошибка: Не удалось найти извлеченный файл vocals.wav в {output_dir}")
            sys.exit(1)
            
    print(f"[+] Вокал успешно изолирован: {isolated_vocals}")
    return isolated_vocals

def print_chain(chain: ChainConfig):
    """Печатает цепочку в физических единицах."""
    print("\n[*] Итоговая цепочка:")
    for plugin in chain.plugins:
        ranges = DATAGEN_PLUGINS[plugin.name]["params"]
        values = ", ".join(
            f"{p}={ranges[p][0] + v * (ranges[p][1] - ranges[p][0]):.2f}" for p, v in plugin.params.items()
        )
        print(f"    [{plugin.name.upper()}] {values}")


def process_vocal(args):
    source_file = Path(args.source)
    ref_file = Path(args.ref)
    model_file = Path(args.model_path)
    output_fst = Path(args.output_fst)
    output_wav = Path(args.output_wav)

    print("=========================================================")
    print("      MIXING-AI: АВТОМАТИЧЕСКИЙ ВОКАЛЬНЫЙ АССИСТЕНТ     ")
    print("=========================================================")
    
    if not source_file.exists():
        print(f"[-] Ошибка: Исходный файл вокала не найден: {source_file}")
        sys.exit(1)
    if not ref_file.exists():
        print(f"[-] Ошибка: Файл референса не найден: {ref_file}")
        sys.exit(1)
    if not model_file.exists():
        print(f"[-] Ошибка: Файл весов модели не найден: {model_file}")
        sys.exit(1)

    temp_dir = Path("data/raw/temp_user_processing")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Извлечение вокала из референса, если требуется
    ref_vocal_path = ref_file
    if args.extract_vocals or ref_file.suffix.lower() in [".mp3", ".m4a", ".flac"]:
        print(f"[*] Референс определен как полный микс/песня ({ref_file.name}).")
        ref_vocal_path = extract_vocal_stem(ref_file, temp_dir / "demucs_ref")

    print(f"\n[+] Исходный сухой вокал: {source_file}")
    print(f"[+] Референс для сведения: {ref_vocal_path}")

    # 2. Аудио-анализ
    processor = AudioProcessor(sr=44100)
    dsp = DSPEngine()
    enhancer = VocalEnhancer(sr=44100)

    print("\n[*] Загрузка и анализ аудиосигналов...")
    sr = processor.sr
    y_ref_norm = processor.load_audio(ref_vocal_path, target_lufs=-23.0, trim_silence=True)

    # Детекция тональности референса
    ref_key, ref_scale = enhancer.detect_key_and_scale(y_ref_norm)
    key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    print(f"[+] Распознанная тональность трека: {key_names[ref_key]} {ref_scale}")

    # 3. Входы модели: стерео Mid/Side мел-спектрограммы и 18 DSP-дескрипторов громкого фрагмента
    #    (та же подготовка, что в генераторе датасета)
    x_spec, x_desc = model_inputs(librosa.load(source_file, sr=sr, mono=False)[0],
                                  librosa.load(ref_vocal_path, sr=sr, mono=False)[0], processor, args.max_len)
    x_spec_tensor = torch.from_numpy(x_spec).unsqueeze(0)
    x_desc_tensor = torch.from_numpy(x_desc).unsqueeze(0)

    # 4. Загрузка нейросети на GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Запуск двухвходового инференса модели (Source + Ref) на {str(device).upper()}...")
    model = MixingAIModel().to(device)
    checkpoint = torch.load(model_file, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    model_dict = model.state_dict()
    model_dict.update({k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape})
    model.load_state_dict(model_dict)
    model.eval()

    with torch.no_grad():
        x_spec_tensor = x_spec_tensor.to(device)
        x_desc_tensor = x_desc_tensor.to(device)
        logits, pred_params = model(x_spec_tensor, None, x_desc_tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        # Чекпоинты до v5 обучены в прежних диапазонах drive/chorus.mix (LEGACY_PARAM_RANGES)
        pred_params_vec = remap_param_vector(pred_params.squeeze(0).cpu().numpy(), checkpoint.get("param_ranges"))

    model_probs = {name: float(probs[i]) for i, name in enumerate(CHAIN_ORDER)}
    model_params = model_params_by_plugin(pred_params_vec)
    model_plugins = [name for name in CHAIN_ORDER if name != "eq" and model_probs[name] > PLUGIN_ON_PROB]
    print(f"[+] Цепочка, предложенная моделью: {model_plugins}")

    # 5. Сухой вокал подается в цепочку на том же уровне, что и при генерации датасета:
    # пороги компрессора, де-эссера и дисторшна в DATAGEN_PLUGINS рассчитаны на -23 LUFS
    y_src_chain = normalize_lufs(to_stereo(librosa.load(source_file, sr=sr, mono=False)[0]), sr, CHAIN_INPUT_LUFS)
    y_ref_full, _ = librosa.load(ref_vocal_path, sr=sr, mono=False)
    gate_thresh = processor.estimate_noise_gate_threshold(librosa.to_mono(y_src_chain))
    print(f"[+] Оптимальный порог Noise Gate: {gate_thresh:.1f} дБ")

    # 6. Подбор цепочки: модель дает начальное приближение, оптимизатор подгоняет звук под референс
    if args.no_optimize:
        eq_bands = SpectralMatcher(sr=sr).match_eq(y_src_chain, y_ref_full)
        chain_config = model_chain(model_plugins, eq_bands, model_params)
    else:
        print(f"\n[*] Подбор цепочки по референсу (бюджет {args.budget:.0f} с)...")
        result = ChainOptimizer(sr=sr).optimize(
            y_src_chain, y_ref_full,
            model_probs=model_probs, model_params=model_params, model_param_std=load_param_std(model_file),
            key_index=ref_key, scale=ref_scale,
            gate_threshold_db=gate_thresh, budget_s=args.budget, patience_renders=args.patience or None,
            allow_tune=not args.no_tune, tune_speed=args.tune_speed,
        )
        chain_config, eq_bands = result.chain, result.eq_bands
    print_chain(chain_config)

    # 7. Генерация нативного .fst пресета
    output_fst.parent.mkdir(parents=True, exist_ok=True)
    try:
        generator = FullPresetGenerator()
        generator.generate_vocal_chain(eq_bands, output_name=output_fst.name)
        # Копируем пресет в целевое место, если имя отличается
        src_fst = Path("data/processed") / output_fst.name
        if src_fst.exists() and src_fst.resolve() != output_fst.resolve():
            shutil.copy(src_fst, output_fst)
        print(f"\n[+] НА ТИВНЫЙ ПРЕСЕТ FL STUDIO СГЕНЕРИРОВАН:")
        print(f"    --> {output_fst.resolve()}")
    except Exception as e:
        print(f"[!] Ошибка сборки .fst пресета: {e}")

    # 8. Рендеринг аудио (если включен)
    if not args.no_audio_render:
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n[*] Рендеринг обработанного аудио ({output_wav.name})...")
        
        # Очистка стационарных резонансов (по каналам: функция работает с моно-сигналом)
        y_cleaned = np.vstack([
            enhancer.suppress_resonances(ch, threshold_db=7.0, max_attenuation_db=12.0) for ch in y_src_chain
        ])

        # Цепочка рендерится в стерео, чтобы реверб давал ширину, как в референсе
        y_out = dsp.apply_chain(
            y_cleaned, sr, chain_config,
            gate_threshold_db=gate_thresh,
            target_key_index=ref_key,
            target_scale=ref_scale
        )

        # Громкость подгоняем под референс, чтобы A/B сравнение было честным
        meter = pyln.Meter(sr)
        target_lufs = meter.integrated_loudness(y_ref_full.T if y_ref_full.ndim == 2 else y_ref_full)
        loudness = meter.integrated_loudness(y_out.T)
        if np.isfinite(loudness) and np.isfinite(target_lufs):
            y_out = pyln.normalize.loudness(y_out.T, loudness, target_lufs).T
            # Вместо лимитера опускаем весь сигнал, если пики вылезли за 0 дБ: лимитер давит пики (у вокала
            # они в середине) и смещает тембр на 1-1.5 дБ, а общий гейн тембр не меняет
            peak = float(np.max(np.abs(y_out)))
            if peak > 0.99:
                y_out = y_out * (0.99 / peak)

        sf.write(output_wav, y_out.T, sr)
        print(f"[+] ОБРАБОТАННЫЙ АУДИОФАЙЛ СОХРАНЕН:")
        print(f"    --> {output_wav.resolve()}")

        ref_features = extract_style_features(y_ref_full, sr, with_pitch=False)
        distance_out, per_feature = style_distance(extract_style_features(y_out, sr, with_pitch=False), ref_features)
        distance_src, _ = style_distance(extract_style_features(y_src_chain, sr, with_pitch=False), ref_features)
        worst = ", ".join(f"{k} {v:+.1f}" for k, v in sorted(per_feature.items(), key=lambda kv: -abs(kv[1]))[:3])
        print(f"[+] Расстояние по стилю до референса (весь трек): исходник {distance_src:.2f} -> результат "
              f"{distance_out:.2f}; сильнее всего отличается: {worst}")

    print("\n=========================================================")
    print("                 СВЕДЕНИЕ УСПЕШНО ЗАВЕРШЕНО!            ")
    print("=========================================================")
    print("Инструкция по использованию в FL Studio:")
    print("1. Откройте ваш проект FL Studio.")
    print("2. Перетащите файл пресета (.fst) прямо на дорожку микшера с вокалом.")
    print("3. Либо прослушайте созданный WAV-файл для оценки звучания.")
    print("=========================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mixing-AI: Перенос стиля сведения на вокал")
    parser.add_argument("--source", type=str, required=True, help="Путь к вашему сухому вокалу (WAV)")
    parser.add_argument("--ref", type=str, required=True, help="Путь к референсному треку (WAV или MP3)")
    parser.add_argument("--extract-vocals", action="store_true", help="Принудительно изолировать вокал из референса через Demucs")
    parser.add_argument("--model-path", type=str, default="data/models/v4/best_model.pth", help="Путь к весам обученной нейросети")
    parser.add_argument("--output-fst", type=str, default="data/processed/vocal_mastering.fst", help="Путь сохранения пресета FL Studio")
    parser.add_argument("--output-wav", type=str, default="data/processed/mixed_vocal_preview.wav", help="Путь сохранения WAV превью")
    parser.add_argument("--no-audio-render", action="store_true", help="Отключить генерацию WAV превью (быстрее, только .fst)")
    parser.add_argument("--max-len", type=int, default=520, help="Длина мел-спектрограммы (как при обучении)")
    parser.add_argument("--no-optimize", action="store_true", help="Использовать цепочку модели без подбора по референсу")
    parser.add_argument("--budget", type=float, default=60.0, help="Время на подбор параметров, секунд")
    parser.add_argument("--no-tune", action="store_true",
                        help="Не добавлять автотюн (если в референсе его нет или слышны артефакты)")
    parser.add_argument("--tune-speed", type=float, default=None,
                        help="Сила тюна 0..1 вручную (по умолчанию подбирается по отклонению от нот референса)")
    parser.add_argument("--patience", type=int, default=60,
                        help="Остановить подбор, если за столько рендеров улучшение меньше 0.01 (0 — до конца бюджета)")

    args = parser.parse_args()
    process_vocal(args)
