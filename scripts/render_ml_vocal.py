import argparse
import sys
import torch
import numpy as np
import soundfile as sf
from pathlib import Path

# Добавление корня проекта в пути импорта Python
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.audio.audio_processor import AudioProcessor
from app.audio.dsp_engine import DSPEngine
from app.audio.vocal_enhancer import VocalEnhancer
from app.dataset.dataset_schema import (
    CHAIN_ORDER, PARAM_LAYOUT, PARAM_OFFSETS, ChainConfig, PluginConfig
)
from app.ml.model import MixingAIModel

def render_vocal(source_path: str | Path, ref_path: str | Path, output_path: str | Path, model_path: str | Path, max_len: int = 700):
    """
    Выполняет инференс модели на исходном и референсном вокале,
    реконструирует цепочку эффектов на основе предсказанных параметров
    и рендерит обработанное аудио (res.wav) через оффлайн-движок Pedalboard.
    """
    src_file = Path(source_path)
    ref_file = Path(ref_path)
    out_file = Path(output_path)
    model_file = Path(model_path)
    
    # Проверки файлов
    if not src_file.exists():
        print(f"[-] Ошибка: Исходный файл не найден по пути: {src_file}")
        return
    if not ref_file.exists():
        print(f"[-] Ошибка: Референсный файл не найден по пути: {ref_file}")
        return
    if not model_file.exists():
        print(f"[-] Ошибка: Веса модели не найдены по пути: {model_file}")
        return

    print("[*] Загрузка и подготовка аудио для анализа...")
    processor = AudioProcessor(sr=44100)
    dsp = DSPEngine()
    
    # 1. Загрузка исходных сигналов (сохраняем оригинальный source для обработки, без VAD обрезки)
    # Загружаем сухой вокал без VAD-фильтра для финального рендеринга
    y_src_full, _ = librosa_load = librosa = librosa_load_fn = (None, None)
    import librosa
    y_src_full, sr = librosa.load(src_file, sr=44100)
    
    # Рассчитываем порог Noise Gate на основе шума в паузах исходного файла
    gate_thresh = processor.estimate_noise_gate_threshold(y_src_full)
    print(f"[+] Автоматически определенный порог Noise Gate: {gate_thresh:.1f} дБ")
    
    # Для анализа модели загружаем нормализованные и обрезанные участки
    y_src_norm = processor.load_audio(src_file, target_lufs=-23.0, trim_silence=True)
    y_ref_norm = processor.load_audio(ref_file, target_lufs=-23.0, trim_silence=True)
    
    # 2. Вычисление Мел-спектрограмм
    mel_dry = processor.get_mel_spectrogram(y_src_norm, n_mels=128)
    mel_wet = processor.get_mel_spectrogram(y_ref_norm, n_mels=128)
    
    # Выравнивание длины спектрограмм
    def pad_spec(mel, length):
        n_mels, t = mel.shape
        if t >= length:
            return mel[:, :length]
        pad_val = mel.min()
        padded = np.full((n_mels, length), pad_val, dtype=np.float32)
        padded[:, :t] = mel
        return padded

    mel_dry_pad = pad_spec(mel_dry, max_len)
    mel_wet_pad = pad_spec(mel_wet, max_len)
    
    mel_delta_pad = mel_wet_pad - mel_dry_pad
    x = np.stack([mel_dry_pad, mel_wet_pad, mel_delta_pad], axis=0) # Форма: (3, 128, max_len)
    
    # 3. Запуск предсказания нейросети
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MixingAIModel().to(device)
    checkpoint = torch.load(model_file, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    model_dict = model.state_dict()
    
    filtered_dict = {}
    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            filtered_dict[k] = v
        else:
            print(f"[!] Предупреждение: Несовпадение размерностей для слоя {k}, пропуск.")
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    model.eval()
    
    x_tensor = torch.from_numpy(x).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits, pred_params = model(x_tensor, None)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        pred_params_vec = pred_params.squeeze(0).cpu().numpy()
        
    active_plugins = [CHAIN_ORDER[i] for i, prob in enumerate(probs) if prob > 0.5]
    if "eq" not in active_plugins:
        active_plugins.insert(0, "eq")
    print(f"[+] Итоговая цепочка плагинов: {active_plugins}")
    
    # 3.5. Вычисляем оптимальный эквалайзер математически (Spectral Matching)
    from app.audio.audio_processor import SpectralMatcher
    from app.dataset.dataset_schema import DATAGEN_PLUGINS
    matcher = SpectralMatcher(sr=sr)
    eq_bands = matcher.match_eq(y_src_norm, y_ref_norm)
    print("[+] Математически вычисленная АЧХ (Hybrid EQ):")
    for idx, (f, g) in enumerate(eq_bands):
        print(f"  Полоса {idx+1}: {f:6.1f} Гц | {g:+5.2f} дБ")
    
    # 4. Реконструкция ChainConfig на основе вектора предсказанных параметров [0..1]
    chain_config = ChainConfig()
    
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
        else:
            if plugin_name in PARAM_OFFSETS:
                layout = PARAM_LAYOUT[plugin_name]
                offset = PARAM_OFFSETS[plugin_name]
                for idx, p_name in enumerate(layout):
                    val = float(pred_params_vec[offset + idx])
                    
                    # Safety Limiter: Ограничиваем экстремальные параметры во избежание мемного звука
                    if plugin_name == "chorus":
                        if p_name == "depth":
                            val = min(val, 0.15)       # Ограничение глубины качания питча до 15%
                        elif p_name == "mix":
                            val = min(val, 0.20)       # Ограничение подмеса хоруса до 20%
                        elif p_name == "rate_hz":
                            val = min(val, 0.25)       # Ограничение скорости LFO (около 1.6 Гц)
                    elif plugin_name == "distortion":
                        if p_name == "drive_db":
                            val = min(val, 0.40)       # Ограничение драйва сатуратора до 8 дБ
                    elif plugin_name == "reverb":
                        if p_name == "wet_level":
                            val = min(val, 0.45)       # Не даем вокалу утонуть в эхе (wet_level до 0.18)
                    elif plugin_name == "delay":
                        if p_name == "mix":
                            val = min(val, 0.50)       # Ограничение громкости дилея до 15%
                            
                    plugin_params[p_name] = val
            
        chain_config.plugins.append(PluginConfig(name=plugin_name, params=plugin_params))
        
    # Выводим информацию по применяемым параметрам
    print("[*] Восстановленная цепочка эффектов:")
    for plugin in chain_config.plugins:
        phys_params = dsp.get_physical_params(plugin)
        print(f"  Плагин '{plugin.name}':")
        for k, v in phys_params.items():
            print(f"    - {k}: {v:.3f}")
            
    # 5. Оффлайн рендеринг обработанного аудио с математическим улучшением
    print(f"[*] Обработка звука через DSPEngine (подавление резонансов + экситер)...")
    enhancer = VocalEnhancer(sr=sr)
    y_src_cleaned = enhancer.suppress_resonances(y_src_full, threshold_db=7.0, max_attenuation_db=12.0)
    
    y_processed = dsp.apply_chain(y_src_cleaned, sr, chain_config, gate_threshold_db=gate_thresh)
    
    y_processed = enhancer.apply_exciter(y_processed, mix=0.12)
    
    # Нормализуем громкость выходного сигнала под громкость референса (-23.0 LUFS)
    import pyloudnorm as pyln
    meter = pyln.Meter(sr)
    try:
        processed_loudness = meter.integrated_loudness(y_processed)
        if not np.isnan(processed_loudness) and not np.isinf(processed_loudness):
            y_processed = pyln.normalize.loudness(y_processed, processed_loudness, -23.0)
            print("[+] Громкость выходного сигнала нормализована к -23.0 LUFS.")
    except Exception as e:
        print(f"[!] Предупреждение при нормализации громкости: {e}")
        
    # Сохранение результата
    out_file.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_file, y_processed, sr)
    print(f"[+] Успех! Обработанный вокал успешно сохранен в: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скрипт E2E рендеринга вокала по предсказаниям модели")
    parser.add_argument("--source", type=str, default="data/raw/source/source.wav", help="Путь к сухому вокалу")
    parser.add_argument("--ref", type=str, default="data/raw/reference/reference.wav", help="Путь к референсному вокалу")
    parser.add_argument("--output", type=str, default="data/processed/res.wav", help="Путь для сохранения обработанного файла")
    parser.add_argument("--model-path", type=str, default="data/models/best_model.pth", help="Путь к файлу весов модели")
    parser.add_argument("--max-len", type=int, default=700, help="Длина спектрограммы")
    
    args = parser.parse_args()
    render_vocal(args.source, args.ref, args.output, args.model_path, args.max_len)
