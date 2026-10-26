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
    CHAIN_ORDER, PARAM_LAYOUT, PARAM_OFFSETS, DATAGEN_PLUGINS, ChainConfig, PluginConfig
)
from app.ml.model import MixingAIModel
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
    y_src_full, sr = librosa.load(source_file, sr=44100)
    y_src_norm = processor.load_audio(source_file, target_lufs=-23.0, trim_silence=True)
    y_ref_norm = processor.load_audio(ref_vocal_path, target_lufs=-23.0, trim_silence=True)

    # Детекция тональности референса
    ref_key, ref_scale = enhancer.detect_key_and_scale(y_ref_norm)
    key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    print(f"[+] Распознанная тональность трека: {key_names[ref_key]} {ref_scale}")

    # Порог Noise Gate
    gate_thresh = processor.estimate_noise_gate_threshold(y_src_full)
    print(f"[+] Оптимальный порог Noise Gate: {gate_thresh:.1f} дБ")

    # 3. Вычисление двухканальных Mid/Side Мел-спектрограмм и 18 физических DSP-дескрипторов
    from app.ml.dataset import MixingAIDataset

    def pad_stereo(mel: np.ndarray, max_len: int) -> np.ndarray:
        c, n_mels, t = mel.shape
        if t >= max_len:
            return mel[:, :, :max_len].astype(np.float32)
        padded = np.full((c, n_mels, max_len), -1.0, dtype=np.float32)
        padded[:, :, :t] = mel
        return padded

    mel_src_padded = pad_stereo(processor.get_stereo_mel_spectrogram(y_src_norm, n_mels=128), args.max_len)
    mel_ref_padded = pad_stereo(processor.get_stereo_mel_spectrogram(y_ref_norm, n_mels=128, pitch_normalize=True), args.max_len)
    desc_src = processor.extract_mixing_descriptors(y_src_norm)
    desc_ref = processor.extract_mixing_descriptors(y_ref_norm)

    x_spec, x_desc = MixingAIDataset.build_model_inputs(mel_src_padded, mel_ref_padded, desc_src, desc_ref)
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
        pred_params_vec = pred_params.squeeze(0).cpu().numpy()

    # Активные плагины
    active_plugins = [CHAIN_ORDER[i] for i, prob in enumerate(probs) if prob > 0.5]
    if "eq" not in active_plugins:
        active_plugins.insert(0, "eq")
    print(f"[+] Активированная цепочка эффектов: {active_plugins}")

    # 5. Математический расчет EQ (Spectral Matching)
    matcher = SpectralMatcher(sr=sr)
    eq_bands = matcher.match_eq(y_src_norm, y_ref_norm)
    print("\n[*] Расчет 7-полосного Fruity Parametric EQ 2:")
    for idx, (f, g) in enumerate(eq_bands):
        print(f"    Полоса {idx+1}: {f:6.1f} Гц | {g:+5.2f} дБ")

    # 6. Реконструкция физических параметров плагинов
    chain_config = ChainConfig()
    print("\n[*] Предсказанные параметры плагинов:")
    for plugin_name in active_plugins:
        plugin_params = {}
        if plugin_name == "eq":
            for i, (f, g) in enumerate(eq_bands):
                min_f, max_f = DATAGEN_PLUGINS["eq"]["params"][f"band{i+1}_freq_hz"]
                min_g, max_g = DATAGEN_PLUGINS["eq"]["params"][f"band{i+1}_gain_db"]
                f_norm = (f - min_f) / (max_f - min_f) if (max_f - min_f) != 0 else 0.0
                g_norm = (g - min_g) / (max_g - min_g) if (max_g - min_g) != 0 else 0.0
                plugin_params[f"band{i+1}_freq_hz"] = float(np.clip(f_norm, 0.0, 1.0))
                plugin_params[f"band{i+1}_gain_db"] = float(np.clip(g_norm, 0.0, 1.0))
        elif plugin_name in PARAM_OFFSETS:
            layout = PARAM_LAYOUT[plugin_name]
            offset = PARAM_OFFSETS[plugin_name]
            ranges = DATAGEN_PLUGINS[plugin_name]["params"]
            for idx, p_name in enumerate(layout):
                val = float(pred_params_vec[offset + idx])
                # Плавные защитные лимиты для деликатного сведения
                if plugin_name == "chorus" and p_name == "depth": val = min(val, 0.20)
                elif plugin_name == "distortion" and p_name == "drive_db": val = min(val, 0.35)
                elif plugin_name == "reverb" and p_name == "wet_level": val = min(val, 0.40)
                elif plugin_name == "delay" and p_name == "mix": val = min(val, 0.45)
                plugin_params[p_name] = val
                
                # Физическое значение для вывода в консоль
                min_v, max_v = ranges[p_name]
                phys_v = min_v + val * (max_v - min_v)
                print(f"    [{plugin_name.upper()}] {p_name}: {phys_v:.2f}")

        chain_config.plugins.append(PluginConfig(name=plugin_name, params=plugin_params))

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
        
        # Очистка резонансов
        y_cleaned = enhancer.suppress_resonances(y_src_full, threshold_db=7.0, max_attenuation_db=12.0)
        
        # Применение DSP-цепочки
        y_out = dsp.apply_chain(
            y_cleaned, sr, chain_config,
            gate_threshold_db=gate_thresh,
            target_key_index=ref_key,
            target_scale=ref_scale
        )
        # Легкий вокальный экситер (присутствие и воздух)
        y_out = enhancer.apply_exciter(y_out, mix=0.10)

        # Нормализация громкости к -23 LUFS
        meter = pyln.Meter(sr)
        try:
            y_temp = y_out.T if y_out.ndim == 2 else y_out
            loudness = meter.integrated_loudness(y_temp)
            if not np.isnan(loudness) and not np.isinf(loudness):
                y_temp = pyln.normalize.loudness(y_temp, loudness, -23.0)
                y_out = y_temp.T if y_out.ndim == 2 else y_temp
        except Exception as e:
            pass

        sf.write(output_wav, y_out.T if y_out.ndim == 2 else y_out, sr)
        print(f"[+] ОБРАБОТАННЫЙ АУДИОФАЙЛ СОХРАНЕН:")
        print(f"    --> {output_wav.resolve()}")

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
    parser.add_argument("--model-path", type=str, default="data/models/best_model.pth", help="Путь к весам обученной нейросети")
    parser.add_argument("--output-fst", type=str, default="data/processed/vocal_mastering.fst", help="Путь сохранения пресета FL Studio")
    parser.add_argument("--output-wav", type=str, default="data/processed/mixed_vocal_preview.wav", help="Путь сохранения WAV превью")
    parser.add_argument("--no-audio-render", action="store_true", help="Отключить генерацию WAV превью (быстрее, только .fst)")
    parser.add_argument("--max-len", type=int, default=700, help="Длина мел-спектрограммы")

    args = parser.parse_args()
    process_vocal(args)
