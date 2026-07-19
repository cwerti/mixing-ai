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
        
        # Определение активных эффектов по долям вероятностей
        active_plugins = []
        rand_val = random.random()
        
        if rand_val < 0.2:
            # 20% Только эквалайзер
            active_plugins = ["eq"]
        elif rand_val < 0.4:
            # 20% EQ + Compressor
            active_plugins = ["eq", "compressor"]
        elif rand_val < 0.6:
            # 20% EQ + Compressor + Reverb
            active_plugins = ["eq", "compressor", "reverb"]
        elif rand_val < 0.8:
            # 20% Полная цепочка
            active_plugins = ["eq", "compressor", "reverb", "delay"]
        else:
            # 20% Случайный непустой набор плагинов
            num_plugins = random.randint(1, len(CHAIN_ORDER))
            active_plugins = random.sample(CHAIN_ORDER, num_plugins)
            
        # Сортировка цепочки эффектов по фиксированному CHAIN_ORDER
        active_plugins = [p for p in CHAIN_ORDER if p in active_plugins]
        
        for name in active_plugins:
            plugin_range = DATAGEN_PLUGINS[name]["params"]
            params = {}
            for p_name in plugin_range:
                # Генерируем нормализованное значение в диапазоне [0.0, 1.0]
                params[p_name] = float(random.random())
            
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
        Запускает цикл генерации датасета на основе файлов в папке сухих сэмплов dry_dir.
        
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

        samples = []
        index_file = self.output_dir / "index.jsonl"
        
        with open(index_file, "w", encoding="utf-8") as f_index:
            for i, src_type in enumerate(tqdm(source_pool)):
                if src_type == "soundcloud":
                    src_file = random.choice(sc_files)
                    source_name = "soundcloud"
                else:
                    src_file = random.choice(vctk_files)
                    source_name = "vctk"
                
                try:
                    # Загрузка и LUFS-нормализация исходного сухого сигнала
                    y_dry = self.audio_processor.load_audio(src_file, target_lufs=-23.0, trim_silence=True)
                    duration = float(len(y_dry) / self.sr)
                    
                    # Генерация случайной цепи эффектов
                    chain_config = self._generate_random_config()
                    
                    # Оффлайн DSP обработка сигнала
                    y_wet = self.dsp.apply_chain(y_dry, self.sr, chain_config)
                    
                    sample_id = f"{i:06d}"
                    dry_name = f"{source_name}_{sample_id}_dry.wav"
                    wet_name = f"{source_name}_{sample_id}_wet.wav"
                    
                    dry_path = dry_out_dir / dry_name
                    wet_path = wet_out_dir / wet_name
                    
                    # Сохранение WAV-файлов на диск
                    sf.write(dry_path, y_dry, self.sr)
                    sf.write(wet_path, y_wet, self.sr)
                    
                    # Извлечение Мел-спектрограмм для ML
                    mel_dry = self.audio_processor.get_mel_spectrogram(y_dry, n_mels=128)
                    mel_wet = self.audio_processor.get_mel_spectrogram(y_wet, n_mels=128)
                    
                    # Сохранение предвычисленных спектрограмм
                    feature_file = features_out_dir / f"{sample_id}.npz"
                    np.savez_compressed(feature_file, mel_dry=mel_dry, mel_wet=mel_wet)
                    
                    # Формирование метаданных сэмпла
                    chain_list = [p.name for p in chain_config.plugins]
                    chain_onehot = [1 if p in chain_list else 0 for p in CHAIN_ORDER]
                    
                    params_dict = {}
                    for plugin in chain_config.plugins:
                        params_dict[plugin.name] = plugin.params
                        
                    flat_vector = self._flatten_params(chain_config)
                    
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
                    
                    # Запись строки в index.jsonl
                    f_index.write(sample.to_jsonl_line() + "\n")
                    samples.append(sample)
                    
                except Exception as e:
                    print(f"\n[-] Ошибка генерации сэмпла {i} из файла {src_file.name}: {e}")
                    continue
                    
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
