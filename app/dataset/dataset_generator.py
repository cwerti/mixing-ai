import os
import random
import json
from typing import List, Dict
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

from app.dataset.dataset_schema import (
    DatasetSample, ChainConfig, PluginConfig, CHAIN_ORDER, DATAGEN_PLUGINS,
    PARAM_LAYOUT, PARAM_OFFSETS, PARAMETER_VECTOR_LENGTH
)
from app.audio.dsp_engine import DSPEngine
from app.audio.audio_processor import AudioProcessor

class DatasetGenerator:
    """
    Класс для автоматической генерации синтетического датасета.
    Применяет случайные цепочки эффектов с рандомными параметрами к сухому вокалу,
    рендерит обработанные файлы и сохраняет метаданные и спектрограммы.
    """

    def __init__(self, dry_dir: Path, output_dir: Path, sr: int = 44100):
        """
        Инициализирует генератор датасета.
        
        Args:
            dry_dir: Папка с исходными сухими вокальными сэмплами.
            output_dir: Папка для сохранения сгенерированного датасета.
            sr: Частота дискретизации аудио. По умолчанию 44100 Гц.
        """
        self.dry_dir = Path(dry_dir)
        self.output_dir = Path(output_dir)
        self.sr = sr
        self.dsp = DSPEngine()
        self.audio_processor = AudioProcessor(sr=sr)

    def _generate_random_config(self) -> ChainConfig:
        """
        Генерирует случайную цепочку плагинов и случайные нормализованные параметры [0..1] для них.
        
        Returns:
            ChainConfig: Конфигурация случайной цепочки эффектов.
        """
        chain = ChainConfig()
        
        # Определяем стилистические профили генерации цепей
        styles = ["pop", "hyperpop", "hiphop", "indie", "metal", "speech", "acoustic", "cinematic", "dry_only"]
        selected_style = random.choices(
            styles, 
            weights=[0.20, 0.20, 0.15, 0.15, 0.10, 0.08, 0.07, 0.05, 0.05], 
            k=1
        )[0]
        
        active_plugins = ["eq"]  # Эквалайзер активен всегда
        
        # Задаем вероятности активации плагинов для каждого стиля
        style_prob = {
            "pop": {
                "pitch_corrector": 0.2, "compressor": 0.9, "deesser": 0.8, "multiband_compressor": 0.1,
                "distortion": 0.05, "chorus": 0.3, "stereo_enhancer": 0.4, "reverb": 0.85, "delay": 0.7,
                "resonance_suppressor": 0.4, "exciter": 0.6
            },
            "hyperpop": {
                "pitch_corrector": 0.95, "compressor": 0.8, "deesser": 0.9, "multiband_compressor": 0.9,
                "distortion": 0.8, "chorus": 0.5, "stereo_enhancer": 0.8, "reverb": 0.8, "delay": 0.85,
                "resonance_suppressor": 0.6, "exciter": 0.8
            },
            "hiphop": {
                "pitch_corrector": 0.8, "compressor": 0.9, "deesser": 0.85, "multiband_compressor": 0.4,
                "distortion": 0.2, "chorus": 0.2, "stereo_enhancer": 0.6, "reverb": 0.7, "delay": 0.6,
                "resonance_suppressor": 0.5, "exciter": 0.5
            },
            "indie": {
                "pitch_corrector": 0.1, "compressor": 0.7, "deesser": 0.6, "multiband_compressor": 0.05,
                "distortion": 0.3, "chorus": 0.75, "stereo_enhancer": 0.3, "reverb": 0.9, "delay": 0.5,
                "resonance_suppressor": 0.3, "exciter": 0.3
            },
            "metal": {
                "pitch_corrector": 0.05, "compressor": 0.95, "deesser": 0.8, "multiband_compressor": 0.5,
                "distortion": 0.9, "chorus": 0.1, "stereo_enhancer": 0.2, "reverb": 0.4, "delay": 0.2,
                "resonance_suppressor": 0.7, "exciter": 0.4
            },
            "speech": {
                "pitch_corrector": 0.0, "compressor": 0.95, "deesser": 0.9, "multiband_compressor": 0.05,
                "distortion": 0.0, "chorus": 0.0, "stereo_enhancer": 0.0, "reverb": 0.05, "delay": 0.0,
                "resonance_suppressor": 0.6, "exciter": 0.3
            },
            "acoustic": {
                "pitch_corrector": 0.0, "compressor": 0.3, "deesser": 0.5, "multiband_compressor": 0.0,
                "distortion": 0.0, "chorus": 0.0, "stereo_enhancer": 0.05, "reverb": 0.8, "delay": 0.05,
                "resonance_suppressor": 0.2, "exciter": 0.2
            },
            "cinematic": {
                "pitch_corrector": 0.1, "compressor": 0.5, "deesser": 0.6, "multiband_compressor": 0.2,
                "distortion": 0.05, "chorus": 0.6, "stereo_enhancer": 0.7, "reverb": 0.95, "delay": 0.8,
                "resonance_suppressor": 0.3, "exciter": 0.4
            },
            "dry_only": {
                "pitch_corrector": 0.0, "compressor": 0.0, "deesser": 0.0, "multiband_compressor": 0.0,
                "distortion": 0.0, "chorus": 0.0, "stereo_enhancer": 0.0, "reverb": 0.0, "delay": 0.0,
                "resonance_suppressor": 0.0, "exciter": 0.0
            }
        }
        
        probs = style_prob[selected_style]
        for name, p in probs.items():
            if random.random() < p:
                active_plugins.append(name)
                
        # Логические связки и правила сведения:
        # 1. Если OTT-компрессия (multiband_compressor) активна, классический компрессор ставим реже
        if "multiband_compressor" in active_plugins and selected_style in ("indie", "pop", "speech") and "compressor" in active_plugins:
            if random.random() < 0.6:
                active_plugins.remove("compressor")
                
        # 2. Не пускаем сильный дисторшн без деэссера (во избежание мемного сибилянтного свиста)
        if "distortion" in active_plugins and "deesser" not in active_plugins:
            active_plugins.append("deesser")
            
        # 3. Не включаем Haas стерео-расширитель без пространства (reverb/delay), чтобы не было фазовой пустоты в моно
        if "reverb" not in active_plugins and "delay" not in active_plugins and "stereo_enhancer" in active_plugins:
            if random.random() < 0.7 and selected_style != "hyperpop" and selected_style != "cinematic":
                active_plugins.remove("stereo_enhancer")

        # Сортировка цепочки эффектов по фиксированному CHAIN_ORDER
        active_plugins = [p for p in CHAIN_ORDER if p in active_plugins]
        
        for name in active_plugins:
            plugin_range = DATAGEN_PLUGINS[name]["params"]
            params = {}
            for p_name in plugin_range:
                val = random.random()
                
                # Тюнинг физического смысла параметров под выбранные стили
                if selected_style == "hyperpop":
                    if name == "multiband_compressor" and p_name == "depth":
                        val = random.uniform(0.4, 0.9)
                    elif name == "pitch_corrector" and p_name == "speed":
                        val = random.uniform(0.8, 1.0)
                    elif name == "distortion" and p_name == "drive_db":
                        val = random.uniform(0.3, 0.7)
                elif selected_style == "pop":
                    if name == "reverb" and p_name == "wet_level":
                        val = random.uniform(0.1, 0.3)
                    elif name == "distortion" and p_name == "drive_db":
                        val = random.uniform(0.01, 0.15)
                elif selected_style == "metal":
                    if name == "compressor" and p_name == "ratio":
                        val = random.uniform(0.7, 1.0)
                    elif name == "distortion" and p_name == "drive_db":
                        val = random.uniform(0.5, 0.9)
                elif selected_style == "speech":
                    if name == "compressor" and p_name == "ratio":
                        val = random.uniform(0.1, 0.4)
                    elif name == "reverb" and p_name == "wet_level":
                        val = random.uniform(0.01, 0.05)
                elif selected_style == "acoustic":
                    if name == "compressor" and p_name == "ratio":
                        val = random.uniform(0.0, 0.2)
                    elif name == "reverb" and p_name == "wet_level":
                        val = random.uniform(0.05, 0.15)
                elif selected_style == "cinematic":
                    if name == "reverb" and p_name == "wet_level":
                        val = random.uniform(0.25, 0.40)
                    elif name == "reverb" and p_name == "room_size":
                        val = random.uniform(0.8, 1.0)
                    elif name == "delay" and p_name == "feedback":
                        val = random.uniform(0.4, 0.7)
                        
                params[p_name] = float(val)
            
            chain.plugins.append(PluginConfig(name=name, params=params))
            
        return chain

    def _flatten_params(self, chain_config: ChainConfig) -> List[float]:
        """
        Преобразует параметры активных плагинов цепочки в плоский вектор динамически вычисляемой длины.
        
        Args:
            chain_config: Конфигурация цепочки эффектов.
            
        Returns:
            List[float]: Плоский вектор параметров с занулением для неактивных плагинов.
        """
        vec = [0.0] * PARAMETER_VECTOR_LENGTH
        
        for plugin in chain_config.plugins:
            if plugin.name not in PARAM_LAYOUT:
                continue
            offset = PARAM_OFFSETS[plugin.name]
            p_names = PARAM_LAYOUT[plugin.name]
            for i, p_name in enumerate(p_names):
                vec[offset + i] = plugin.params.get(p_name, 0.0)
                
        return vec

    def generate(self, max_samples: int = 1000, sc_ratio: float = 0.5) -> List[DatasetSample]:
        """
        Запускает цикл параллельной генерации датасета на основе файлов в папке dry_dir.
        Использует все доступные ядра процессора (Multiprocessing).
        
        Args:
            max_samples: Максимальное количество сэмплов для генерации.
            sc_ratio: Пропорция сэмплов SoundCloud по отношению к VCTK (от 0.0 до 1.0).
            
        Returns:
            List[DatasetSample]: Список сгенерированных объектов сэмплов.
        """
        dry_out_dir = self.output_dir / "audio" / "dry"
        wet_out_dir = self.output_dir / "audio" / "wet"
        features_out_dir = self.output_dir / "features"
        
        dry_out_dir.mkdir(parents=True, exist_ok=True)
        wet_out_dir.mkdir(parents=True, exist_ok=True)
        features_out_dir.mkdir(parents=True, exist_ok=True)

        vctk_dir = self.dry_dir / "vctk"
        sc_dir = self.dry_dir / "soundcloud"
        
        vctk_files = []
        sc_files = []
        for ext in ["*.wav", "*.mp3", "*.flac"]:
            if vctk_dir.exists():
                vctk_files.extend(list(vctk_dir.rglob(ext)))
            if sc_dir.exists():
                sc_files.extend(list(sc_dir.rglob(ext)))

        if not vctk_files and not sc_files:
            print(f"[-] Исходные сухие файлы не найдены в директории: {self.dry_dir}")
            return []

        # Расчет распределения источников аудио в соответствии с sc_ratio
        if sc_files and vctk_files:
            sc_target = int(max_samples * sc_ratio)
            vctk_target = max_samples - sc_target
        elif sc_files:
            print("[!] Файлы VCTK не найдены. Генерация 100% сэмплов из SoundCloud.")
            sc_target = max_samples
            vctk_target = 0
        else:
            print("[!] Файлы SoundCloud не найдены. Генерация 100% сэмплов из VCTK.")
            sc_target = 0
            vctk_target = max_samples

        print(f"[*] Целевое распределение: SoundCloud: {sc_target} сэмплов, VCTK: {vctk_target} сэмплов.")
        
        source_pool = (["soundcloud"] * sc_target) + (["vctk"] * vctk_target)
        random.shuffle(source_pool)

        # Подготовка задач для параллельного выполнения
        tasks = []
        for i, src_type in enumerate(source_pool):
            if src_type == "soundcloud":
                src_file = random.choice(sc_files)
                source_name = "soundcloud"
            else:
                src_file = random.choice(vctk_files)
                source_name = "vctk"
            tasks.append((i, src_file, source_name, self.sr, str(self.output_dir)))

        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        num_cores = multiprocessing.cpu_count()
        print(f"[*] Запуск параллельной генерации на {num_cores} ядрах процессора...")

        samples = []
        index_file = self.output_dir / "index.jsonl"
        
        with open(index_file, "w", encoding="utf-8") as f_index:
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                # tqdm отображает прогресс-бар по мере выполнения воркеров
                for result_line in tqdm(executor.map(_generate_sample_worker, tasks), total=len(tasks)):
                    if result_line is not None:
                        f_index.write(result_line + "\n")
                        samples.append(DatasetSample.from_jsonl_line(result_line))
                        
        # Запись общей статистики датасета
        stats = {
            "total_samples": len(samples),
            "sr": self.sr,
            "chain_order": CHAIN_ORDER,
            "param_layout": PARAM_LAYOUT
        }
        with open(self.output_dir / "stats.json", "w", encoding="utf-8") as f_stats:
            json.dump(stats, f_stats, indent=4)
            
        print(f"[+] Генерация датасета успешно завершена. Результаты сохранены в: {self.output_dir}")
        return samples


def _generate_sample_worker(task) -> str | None:
    """
    Автономный воркер для обработки одного аудиофайла.
    Выполняется в отдельном процессе Windows (без PicklingError).
    """
    idx, src_file, source_name, sr, output_dir_str = task
    
    import random
    import numpy as np
    import soundfile as sf
    from pathlib import Path
    
    from app.audio.audio_processor import AudioProcessor
    from app.audio.dsp_engine import DSPEngine
    from app.dataset.dataset_schema import (
        CHAIN_ORDER, DATAGEN_PLUGINS, PARAM_LAYOUT, PARAM_OFFSETS, 
        PARAMETER_VECTOR_LENGTH, ChainConfig, PluginConfig, DatasetSample
    )
    
    output_dir = Path(output_dir_str)
    dry_out_dir = output_dir / "audio" / "dry"
    wet_out_dir = output_dir / "audio" / "wet"
    features_out_dir = output_dir / "features"
    
    audio_processor = AudioProcessor(sr=sr)
    dsp = DSPEngine()
    
    try:
        # 1. Загрузка и LUFS-нормализация сухого сигнала
        y_dry = audio_processor.load_audio(src_file, target_lufs=-23.0, trim_silence=True)
        duration = float(len(y_dry) / sr)
        
        # 2. Стилистическая генерация цепочки плагинов
        chain = ChainConfig()
        styles = ["pop", "hyperpop", "hiphop", "indie", "metal", "speech", "acoustic", "cinematic", "dry_only"]
        selected_style = random.choices(
            styles, 
            weights=[0.20, 0.20, 0.15, 0.15, 0.10, 0.08, 0.07, 0.05, 0.05], 
            k=1
        )[0]
        
        active_plugins = ["eq"]  # Эквалайзер активен всегда
        
        style_prob = {
            "pop": {
                "pitch_corrector": 0.2, "compressor": 0.9, "deesser": 0.8, "multiband_compressor": 0.1,
                "distortion": 0.05, "chorus": 0.3, "stereo_enhancer": 0.4, "reverb": 0.85, "delay": 0.7,
                "resonance_suppressor": 0.4, "exciter": 0.6
            },
            "hyperpop": {
                "pitch_corrector": 0.95, "compressor": 0.8, "deesser": 0.9, "multiband_compressor": 0.9,
                "distortion": 0.8, "chorus": 0.5, "stereo_enhancer": 0.8, "reverb": 0.8, "delay": 0.85,
                "resonance_suppressor": 0.6, "exciter": 0.8
            },
            "hiphop": {
                "pitch_corrector": 0.8, "compressor": 0.9, "deesser": 0.85, "multiband_compressor": 0.4,
                "distortion": 0.2, "chorus": 0.2, "stereo_enhancer": 0.6, "reverb": 0.7, "delay": 0.6,
                "resonance_suppressor": 0.5, "exciter": 0.5
            },
            "indie": {
                "pitch_corrector": 0.1, "compressor": 0.7, "deesser": 0.6, "multiband_compressor": 0.05,
                "distortion": 0.3, "chorus": 0.75, "stereo_enhancer": 0.3, "reverb": 0.9, "delay": 0.5,
                "resonance_suppressor": 0.3, "exciter": 0.3
            },
            "metal": {
                "pitch_corrector": 0.05, "compressor": 0.95, "deesser": 0.8, "multiband_compressor": 0.5,
                "distortion": 0.9, "chorus": 0.1, "stereo_enhancer": 0.2, "reverb": 0.4, "delay": 0.2,
                "resonance_suppressor": 0.7, "exciter": 0.4
            },
            "speech": {
                "pitch_corrector": 0.0, "compressor": 0.95, "deesser": 0.9, "multiband_compressor": 0.05,
                "distortion": 0.0, "chorus": 0.0, "stereo_enhancer": 0.0, "reverb": 0.05, "delay": 0.0,
                "resonance_suppressor": 0.6, "exciter": 0.3
            },
            "acoustic": {
                "pitch_corrector": 0.0, "compressor": 0.3, "deesser": 0.5, "multiband_compressor": 0.0,
                "distortion": 0.0, "chorus": 0.0, "stereo_enhancer": 0.05, "reverb": 0.8, "delay": 0.05,
                "resonance_suppressor": 0.2, "exciter": 0.2
            },
            "cinematic": {
                "pitch_corrector": 0.1, "compressor": 0.5, "deesser": 0.6, "multiband_compressor": 0.2,
                "distortion": 0.05, "chorus": 0.6, "stereo_enhancer": 0.7, "reverb": 0.95, "delay": 0.8,
                "resonance_suppressor": 0.3, "exciter": 0.4
            },
            "dry_only": {
                "pitch_corrector": 0.0, "compressor": 0.0, "deesser": 0.0, "multiband_compressor": 0.0,
                "distortion": 0.0, "chorus": 0.0, "stereo_enhancer": 0.0, "reverb": 0.0, "delay": 0.0,
                "resonance_suppressor": 0.0, "exciter": 0.0
            }
        }
        
        probs = style_prob[selected_style]
        for name, p in probs.items():
            if random.random() < p:
                active_plugins.append(name)
                
        # Логика взаимного исключения и зависимостей плагинов
        if "multiband_compressor" in active_plugins and selected_style in ("indie", "pop", "speech") and "compressor" in active_plugins:
            if random.random() < 0.6:
                active_plugins.remove("compressor")
                
        if "distortion" in active_plugins and "deesser" not in active_plugins:
            active_plugins.append("deesser")
            
        if "reverb" not in active_plugins and "delay" not in active_plugins and "stereo_enhancer" in active_plugins:
            if random.random() < 0.7 and selected_style != "hyperpop" and selected_style != "cinematic":
                active_plugins.remove("stereo_enhancer")

        active_plugins = [p for p in CHAIN_ORDER if p in active_plugins]
        
        for name in active_plugins:
            plugin_range = DATAGEN_PLUGINS[name]["params"]
            params = {}
            for p_name in plugin_range:
                val = random.random()
                
                if selected_style == "hyperpop":
                    if name == "multiband_compressor" and p_name == "depth":
                        val = random.uniform(0.4, 0.9)
                    elif name == "pitch_corrector" and p_name == "speed":
                        val = random.uniform(0.8, 1.0)
                    elif name == "distortion" and p_name == "drive_db":
                        val = random.uniform(0.3, 0.7)
                elif selected_style == "pop":
                    if name == "reverb" and p_name == "wet_level":
                        val = random.uniform(0.1, 0.3)
                    elif name == "distortion" and p_name == "drive_db":
                        val = random.uniform(0.01, 0.15)
                elif selected_style == "metal":
                    if name == "compressor" and p_name == "ratio":
                        val = random.uniform(0.7, 1.0)
                    elif name == "distortion" and p_name == "drive_db":
                        val = random.uniform(0.5, 0.9)
                elif selected_style == "speech":
                    if name == "compressor" and p_name == "ratio":
                        val = random.uniform(0.1, 0.4)
                    elif name == "reverb" and p_name == "wet_level":
                        val = random.uniform(0.01, 0.05)
                elif selected_style == "acoustic":
                    if name == "compressor" and p_name == "ratio":
                        val = random.uniform(0.0, 0.2)
                    elif name == "reverb" and p_name == "wet_level":
                        val = random.uniform(0.05, 0.15)
                elif selected_style == "cinematic":
                    if name == "reverb" and p_name == "wet_level":
                        val = random.uniform(0.25, 0.40)
                    elif name == "reverb" and p_name == "room_size":
                        val = random.uniform(0.8, 1.0)
                    elif name == "delay" and p_name == "feedback":
                        val = random.uniform(0.4, 0.7)
                        
                params[p_name] = float(val)
            chain.plugins.append(PluginConfig(name=name, params=params))

        # 3. DSP Рендеринг сигнала в оффлайн-цепочке
        y_wet = dsp.apply_chain(y_dry, sr, chain)

        # 4. Сохранение результатов на диск
        sample_id = f"{idx:06d}"
        dry_name = f"{source_name}_{sample_id}_dry.wav"
        wet_name = f"{source_name}_{sample_id}_wet.wav"
        
        dry_path = dry_out_dir / dry_name
        wet_path = wet_out_dir / wet_name
        
        sf.write(dry_path, y_dry.T if y_dry.ndim == 2 else y_dry, sr)
        sf.write(wet_path, y_wet.T if y_wet.ndim == 2 else y_wet, sr)
        
        # 5. Извлечение Мел-спектрограмм для ML
        mel_dry = audio_processor.get_mel_spectrogram(y_dry, n_mels=128)
        mel_wet = audio_processor.get_mel_spectrogram(y_wet, n_mels=128)
        
        feature_file = features_out_dir / f"{sample_id}.npz"
        np.savez_compressed(feature_file, mel_dry=mel_dry, mel_wet=mel_wet)
        
        # 6. Метаданные сэмпла
        chain_list = [p.name for p in chain.plugins]
        chain_onehot = [1 if p in chain_list else 0 for p in CHAIN_ORDER]
        
        params_dict = {}
        for plugin in chain.plugins:
            params_dict[plugin.name] = plugin.params
            
        # Формирование плоского вектора регрессии
        flat_vector = [0.0] * PARAMETER_VECTOR_LENGTH
        for plugin in chain.plugins:
            if plugin.name not in PARAM_LAYOUT:
                continue
            offset = PARAM_OFFSETS[plugin.name]
            p_names = PARAM_LAYOUT[plugin.name]
            for j, p_name in enumerate(p_names):
                flat_vector[offset + j] = plugin.params.get(p_name, 0.0)
                
        sample = DatasetSample(
            id=sample_id,
            dry_path=str(Path("audio/dry") / dry_name),
            wet_path=str(Path("audio/wet") / wet_name),
            source=source_name,
            duration_sec=duration,
            chain=chain_list,
            chain_onehot=chain_onehot,
            params=params_dict,
            params_vector=flat_vector
        )
        return sample.to_jsonl_line()
    except Exception as e:
        print(f"\n[-] Ошибка генерации сэмпла {idx} из файла {src_file.name}: {e}")
        return None

