import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import librosa

# Добавление корня проекта в пути импорта Python
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.audio.audio_processor import AudioProcessor
from app.dataset.dataset_schema import (
    CHAIN_ORDER, PARAM_LAYOUT, PARAM_OFFSETS, DATAGEN_PLUGINS, PARAMETER_VECTOR_LENGTH
)
from app.ml.model import MixingAIModel
from app.presets.full_preset_generator import FullPresetGenerator

class MixingAIMLPipeline:
    """
    Инференс-пайплайн Mixing-AI.
    Загружает обученную модель, извлекает спектральные характеристики 
    из входного (сухого) вокала и референса, запускает предсказание ML
    и генерирует файл пресета .fst для FL Studio.
    """

    def __init__(self, model_path: str | Path, max_len: int = 700):
        """
        Инициализирует пайплайн инференса.
        
        Args:
            model_path: Путь к файлу сохраненных весов модели (.pth).
            max_len: Фиксированная длина спектрограммы временной оси.
        """
        self.max_len = max_len
        self.processor = AudioProcessor(sr=44100)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Создание и загрузка весов модели
        self.model = MixingAIModel().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"]
        model_dict = self.model.state_dict()
        
        filtered_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                filtered_dict[k] = v
            else:
                print(f"[!] Предупреждение: Несовпадение размерностей для слоя {k}, пропуск.")
        model_dict.update(filtered_dict)
        self.model.load_state_dict(model_dict)
        self.model.eval()
        print(f"[+] Модель успешно загружена на {self.device} (эпоха обучения: {checkpoint.get('epoch', 'N/A')})")

    def _pad_spectrogram(self, mel: np.ndarray) -> np.ndarray:
        """
        Дополняет или обрезает спектрограмму по оси времени до max_len.
        
        Args:
            mel: Мел-спектрограмма.
            
        Returns:
            np.ndarray: Спектрограмма фиксированной формы (128, max_len).
        """
        n_mels, t = mel.shape
        if t >= self.max_len:
            return mel[:, :self.max_len]
        
        pad_val = mel.min()
        padded = np.full((n_mels, self.max_len), pad_val, dtype=np.float32)
        padded[:, :t] = mel
        return padded

    def run(self, source_path: str | Path, ref_path: str | Path, output_preset_name: str = "mixing_ai_predicted.fst") -> Path | None:
        """
        Запускает полный процесс ML-подбора плагинов и параметров.
        
        Args:
            source_path: Путь к сухому вокалу пользователя.
            ref_path: Путь к референсному вокалу.
            output_preset_name: Имя генерируемого пресета .fst.
            
        Returns:
            Optional[Path]: Путь к сгенерированному файлу пресета или None в случае ошибки.
        """
        print(f"\n[*] Запуск инференса модели для:\n  Dry: {source_path}\n  Ref: {ref_path}")
        
        # 1. Загрузка и нормализация аудио
        y_src = self.processor.load_audio(source_path, target_lufs=-23.0, trim_silence=True)
        y_ref = self.processor.load_audio(ref_path, target_lufs=-23.0, trim_silence=True)
        
        # 2. Вычисление Мел-спектрограмм
        mel_dry = self.processor.get_mel_spectrogram(y_src, n_mels=128)
        mel_wet = self.processor.get_mel_spectrogram(y_ref, n_mels=128, pitch_normalize=True)
        
        # 3. Выравнивание размеров спектрограмм
        mel_dry_padded = self._pad_spectrogram(mel_dry)
        mel_wet_padded = self._pad_spectrogram(mel_wet)
        
        # Формирование батч-тензора ввода: форма (1, 1, 128, max_len)
        x = mel_wet_padded[np.newaxis, :, :]
        x_tensor = torch.from_numpy(x).unsqueeze(0).to(self.device)
        
        # 4. Прогон через модель
        with torch.no_grad():
            logits, pred_params = self.model(x_tensor, None)
            
            # Определение активных плагинов
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            active_plugins = [CHAIN_ORDER[i] for i, prob in enumerate(probs) if prob > 0.5]
            
            # Извлечение вектора параметров
            pred_params_vec = pred_params.squeeze(0).cpu().numpy()

        print(f"[+] Распознанные активные плагины (вероятность > 50%): {active_plugins}")
        
        # 5. Денормализация параметров EQ для экспорта пресета .fst
        eq_bands = []
        eq_ranges = DATAGEN_PLUGINS["eq"]["params"]
        eq_offset = PARAM_OFFSETS["eq"]
        
        for i in range(7):
            f_norm = pred_params_vec[eq_offset + i * 2]
            g_norm = pred_params_vec[eq_offset + i * 2 + 1]
            
            min_f, max_f = eq_ranges[f"band{i+1}_freq_hz"]
            min_g, max_g = eq_ranges[f"band{i+1}_gain_db"]
            
            f_phys = min_f + f_norm * (max_f - min_f)
            g_phys = min_g + g_norm * (max_g - min_g)
            eq_bands.append((f_phys, g_phys))
            
        print("[*] Предсказанные параметры 7-полосного EQ:")
        for idx, (f, g) in enumerate(eq_bands):
            print(f"  Полоса {idx+1}: {f:6.1f} Гц | {g:+5.2f} дБ")
            
        # Вывод параметров остальных плагинов динамически
        for plugin_name in CHAIN_ORDER:
            if plugin_name != "eq" and plugin_name in active_plugins:
                if plugin_name in PARAM_LAYOUT:
                    print(f"[*] Предсказанные параметры {plugin_name.upper()}:")
                    layout = PARAM_LAYOUT[plugin_name]
                    offset = PARAM_OFFSETS[plugin_name]
                    ranges = DATAGEN_PLUGINS[plugin_name]["params"]
                    
                    for idx, p_name in enumerate(layout):
                        val_norm = pred_params_vec[offset + idx]
                        min_v, max_v = ranges[p_name]
                        val_phys = min_v + val_norm * (max_v - min_v)
                        
                        unit = ""
                        if "freq" in p_name: unit = " Гц"
                        elif "gain" in p_name or "db" in p_name: unit = " дБ"
                        elif "ms" in p_name: unit = " мс"
                        elif "seconds" in p_name: unit = " сек"
                        
                        if plugin_name in ("reverb", "delay", "chorus") and any(kw in p_name for kw in ("size", "damp", "level", "feedback", "mix", "depth")):
                            print(f"  - {p_name}: {val_phys * 100:.1f}%")
                        elif "ratio" in p_name:
                            print(f"  - {p_name}: {val_phys:.2f}:1")
                        else:
                            print(f"  - {p_name}: {val_phys:.2f}{unit}")
                else:
                    print(f"[*] Плагин {plugin_name.upper()} активен (без дополнительных параметров).")
            
        # 6. Сборка бинарного пресета .fst на основе шаблона
        try:
            generator = FullPresetGenerator()
            preset_path = generator.generate_vocal_chain(eq_bands, output_preset_name)
            return preset_path
        except Exception as e:
            print(f"[-] Ошибка генерации пресета .fst: {e}")
            return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скрипт инференса Mixing-AI Pipeline")
    parser.add_argument("--source", type=str, default="data/raw/source/source.wav", help="Путь к сухому вокалу")
    parser.add_argument("--ref", type=str, default="data/raw/reference/reference.wav", help="Путь к референсному вокалу")
    parser.add_argument("--model-path", type=str, default="data/models/best_model.pth", help="Путь к весам модели")
    parser.add_argument("--output-name", type=str, default="mixing_ai_predicted.fst", help="Имя выходного пресета .fst")
    
    args = parser.parse_args()
    
    # Проверка существования файлов перед запуском
    if not Path(args.model_path).exists():
        print(f"[-] Ошибка: Веса модели не найдены по пути: {args.model_path}. Запустите сначала обучение train_ml.py.")
        sys.exit(1)
        
    pipeline = MixingAIMLPipeline(args.model_path)
    pipeline.run(args.source, args.ref, args.output_name)
