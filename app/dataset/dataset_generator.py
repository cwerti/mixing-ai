import os
import random
import json
from typing import List, Dict, Tuple
import numpy as np
import soundfile as sf
import scipy.signal as signal
from pathlib import Path
from tqdm import tqdm

from app.dataset.dataset_schema import (
    DatasetSample, ChainConfig, PluginConfig, CHAIN_ORDER, DATAGEN_PLUGINS,
    PARAM_LAYOUT, PARAM_OFFSETS, PARAMETER_VECTOR_LENGTH
)
from app.audio.dsp_engine import DSPEngine
from app.audio.audio_processor import AudioProcessor


def _generate_balanced_random_config(rng: random.Random = random) -> ChainConfig:
    """
    Генерирует стилистически осмысленную и при этом сбалансированную по классам цепочку плагинов.
    Значения параметров [0.0..1.0] покрывают весь музыкальный диапазон DATAGEN_PLUGINS без скрытого обрезания.
    """
    chain = ChainConfig()

    styles = ["pop", "hyperpop", "hiphop", "indie", "rock_metal", "rnb_modern", "electronic", "cinematic", "dry_clean"]
    selected_style = rng.choices(
        styles,
        weights=[0.16, 0.16, 0.14, 0.14, 0.12, 0.12, 0.08, 0.05, 0.03],
        k=1
    )[0]

    active_plugins = ["eq"]  # Эквалайзер активен всегда

    # Сбалансированные вероятности активации (каждый из 10 эффектов представлен в 35%..65% датасета)
    style_prob = {
        "pop": {
            "pitch_corrector": 0.55, "compressor": 0.75, "deesser": 0.65, "multiband_compressor": 0.35,
            "distortion": 0.25, "chorus": 0.45, "stereo_enhancer": 0.55, "reverb": 0.75, "delay": 0.60,
            "exciter": 0.60
        },
        "hyperpop": {
            "pitch_corrector": 0.90, "compressor": 0.65, "deesser": 0.75, "multiband_compressor": 0.85,
            "distortion": 0.75, "chorus": 0.65, "stereo_enhancer": 0.80, "reverb": 0.65, "delay": 0.75,
            "exciter": 0.80
        },
        "hiphop": {
            "pitch_corrector": 0.75, "compressor": 0.80, "deesser": 0.70, "multiband_compressor": 0.50,
            "distortion": 0.45, "chorus": 0.30, "stereo_enhancer": 0.55, "reverb": 0.55, "delay": 0.65,
            "exciter": 0.55
        },
        "indie": {
            "pitch_corrector": 0.20, "compressor": 0.55, "deesser": 0.45, "multiband_compressor": 0.25,
            "distortion": 0.50, "chorus": 0.75, "stereo_enhancer": 0.50, "reverb": 0.80, "delay": 0.55,
            "exciter": 0.35
        },
        "rock_metal": {
            "pitch_corrector": 0.15, "compressor": 0.85, "deesser": 0.65, "multiband_compressor": 0.60,
            "distortion": 0.85, "chorus": 0.25, "stereo_enhancer": 0.40, "reverb": 0.45, "delay": 0.40,
            "exciter": 0.50
        },
        "rnb_modern": {
            "pitch_corrector": 0.70, "compressor": 0.70, "deesser": 0.65, "multiband_compressor": 0.45,
            "distortion": 0.20, "chorus": 0.55, "stereo_enhancer": 0.70, "reverb": 0.80, "delay": 0.65,
            "exciter": 0.65
        },
        "electronic": {
            "pitch_corrector": 0.80, "compressor": 0.60, "deesser": 0.60, "multiband_compressor": 0.75,
            "distortion": 0.60, "chorus": 0.70, "stereo_enhancer": 0.75, "reverb": 0.70, "delay": 0.70,
            "exciter": 0.70
        },
        "cinematic": {
            "pitch_corrector": 0.20, "compressor": 0.50, "deesser": 0.40, "multiband_compressor": 0.30,
            "distortion": 0.15, "chorus": 0.60, "stereo_enhancer": 0.75, "reverb": 0.90, "delay": 0.75,
            "exciter": 0.45
        },
        "dry_clean": {
            "pitch_corrector": 0.0, "compressor": 0.25, "deesser": 0.25, "multiband_compressor": 0.0,
            "distortion": 0.0, "chorus": 0.0, "stereo_enhancer": 0.0, "reverb": 0.0, "delay": 0.0,
            "exciter": 0.0
        }
    }

    probs = style_prob[selected_style]
    for name, p in probs.items():
        if rng.random() < p:
            active_plugins.append(name)

    active_plugins = [p for p in CHAIN_ORDER if p in active_plugins]

    for name in active_plugins:
        plugin_range = DATAGEN_PLUGINS[name]["params"]
        params = {}

        if name == "eq":
            # На инференсе EQ рассчитывается аналитически через SpectralMatcher.
            # В синтетическом датасете задаем умеренную вариацию АЧХ (±1.8 дБ вокруг 0.5),
            # чтобы случайные ±12 дБ провалы/подъемы не маскировали De-esser, Exciter и OTT.
            for p_name in plugin_range:
                if "gain" in p_name:
                    params[p_name] = float(rng.uniform(0.42, 0.58))
                else:
                    params[p_name] = float(rng.uniform(0.25, 0.75))
        elif name == "compressor":
            # Порог всегда в рабочей зоне [-36..-13.6 дБ], чтобы компрессор реально срабатывал
            thresh = float(rng.uniform(0.05, 0.80))
            intensity = 1.0 - thresh
            params["threshold_db"] = thresh
            params["ratio"] = float(np.clip(0.20 + 0.75 * intensity + rng.uniform(-0.12, 0.12), 0.0, 1.0))
            params["attack_ms"] = float(np.clip(0.80 - 0.65 * intensity + rng.uniform(-0.12, 0.12), 0.0, 1.0))
            params["release_ms"] = float(np.clip(0.75 - 0.55 * intensity + rng.uniform(-0.12, 0.12), 0.0, 1.0))
        elif name == "deesser":
            thresh = float(rng.uniform(0.05, 0.80))
            intensity = 1.0 - thresh
            params["threshold_db"] = thresh
            params["ratio"] = float(np.clip(0.20 + 0.75 * intensity + rng.uniform(-0.12, 0.12), 0.0, 1.0))
        elif name == "chorus":
            mix_val = float(rng.uniform(0.10, 0.95))
            rate_val = float(rng.uniform(0.08, 0.92))
            params["mix"] = mix_val
            params["rate_hz"] = rate_val
            params["depth"] = float(np.clip(0.20 + 0.65 * mix_val + rng.uniform(-0.12, 0.12), 0.0, 1.0))
            params["feedback"] = float(np.clip(0.15 + 0.65 * mix_val + rng.uniform(-0.12, 0.12), 0.0, 1.0))
        elif name == "reverb":
            wet_val = float(rng.uniform(0.08, 0.95))
            room_val = float( np.clip(0.20 + 0.70 * wet_val + rng.uniform(-0.18, 0.18), 0.0, 1.0) )
            damp_val = float(rng.uniform(0.08, 0.92))
            # Dry-уровень физически связан с Wet-уровнем (баланс Dry/Wet), чтобы избежать вырождения при нормализации
            dry_val = float(np.clip(0.95 - 0.65 * wet_val + rng.uniform(-0.08, 0.08), 0.0, 1.0))
            params["room_size"] = room_val
            params["damping"] = damp_val
            params["wet_level"] = wet_val
            params["dry_level"] = dry_val
        elif name == "delay":
            dly_sec = float(rng.uniform(0.05, 0.95))
            mix_val = float(rng.uniform(0.10, 0.95))
            fb_val = float(np.clip(0.18 + 0.70 * mix_val + rng.uniform(-0.12, 0.12), 0.0, 1.0))
            params["delay_seconds"] = dly_sec
            params["feedback"] = fb_val
            params["mix"] = mix_val
        elif name == "exciter":
            mix_val = float(rng.uniform(0.10, 0.95))
            cutoff_val = float(rng.uniform(0.05, 0.95))
            params["mix"] = mix_val
            params["cutoff_hz"] = cutoff_val
        else:
            for p_name in plugin_range:
                val = rng.uniform(0.05, 0.95)
                if rng.random() < 0.30:
                    if selected_style in ("hyperpop", "electronic") and name in ("pitch_corrector", "multiband_compressor"):
                        val = rng.uniform(0.55, 0.98)
                    elif selected_style == "rock_metal" and name == "distortion":
                        val = rng.uniform(0.50, 0.98)
                params[p_name] = float(val)

        chain.plugins.append(PluginConfig(name=name, params=params))

    return chain


class DatasetGenerator:
    """
    Генератор синтетического датасета v2 для обучения Mixing-AI.
    Формирует пары (Dry, Wet) исключительно на чистых студийных вокалах (Cambridge MT / VCTK),
    делит исходники по трекам/певцам на train (85%) и val (15%) до нарезки
    и сохраняет стерео Mid/Side мел-спектрограммы + 18 физических DSP-дескрипторов.
    """

    def __init__(self, dry_dir: Path, output_dir: Path, sr: int = 44100):
        self.dry_dir = Path(dry_dir)
        self.output_dir = Path(output_dir)
        self.sr = sr
        self.dsp = DSPEngine()
        self.audio_processor = AudioProcessor(sr=sr)

    def _generate_random_config(self) -> ChainConfig:
        return _generate_balanced_random_config()

    def _flatten_params(self, chain_config: ChainConfig) -> List[float]:
        vec = [0.0] * PARAMETER_VECTOR_LENGTH
        for plugin in chain_config.plugins:
            if plugin.name not in PARAM_LAYOUT:
                continue
            offset = PARAM_OFFSETS[plugin.name]
            p_names = PARAM_LAYOUT[plugin.name]
            for i, p_name in enumerate(p_names):
                vec[offset + i] = plugin.params.get(p_name, 0.0)
        return vec

    @staticmethod
    def _split_files_by_group(files: List[Path], val_ratio: float = 0.15) -> Tuple[List[Path], List[Path]]:
        """
        Разделяет список исходных файлов на train и val по песне/вокалисту (префикс файла или родительская папка),
        чтобы исключить утечку одного и того же голоса между train и val.
        """
        if len(files) <= 1:
            return files, files

        groups: Dict[str, List[Path]] = {}
        for f in sorted(files):
            parts = f.stem.split("_")
            group_key = "_".join(parts[:2]) if len(parts) >= 2 else f.stem
            groups.setdefault(group_key, []).append(f)

        group_keys = sorted(groups.keys())
        rng = random.Random(42)
        rng.shuffle(group_keys)

        if len(group_keys) == 1:
            sorted_f = sorted(files)
            n_val = max(1, int(round(len(sorted_f) * val_ratio)))
            return sorted_f[:-n_val], sorted_f[-n_val:]

        n_val_groups = max(1, int(round(len(group_keys) * val_ratio)))
        val_keys = set(group_keys[:n_val_groups])

        train_files, val_files = [], []
        for k, f_list in groups.items():
            if k in val_keys:
                val_files.extend(f_list)
            else:
                train_files.extend(f_list)

        if not train_files:
            train_files = val_files
        if not val_files:
            val_files = train_files[:1]
        return train_files, val_files

    def generate(self, max_samples: int = 1000, sc_ratio: float = 0.0, save_all_wav: bool = False) -> List[DatasetSample]:
        """
        Запускает параллельную генерацию датасета на всех ядрах CPU (без использования SoundCloud).
        """
        dry_out_dir = self.output_dir / "audio" / "dry"
        wet_out_dir = self.output_dir / "audio" / "wet"
        features_out_dir = self.output_dir / "features"

        dry_out_dir.mkdir(parents=True, exist_ok=True)
        wet_out_dir.mkdir(parents=True, exist_ok=True)
        features_out_dir.mkdir(parents=True, exist_ok=True)

        cambridge_dir = self.dry_dir / "cambridge"
        vctk_dir = self.dry_dir / "vctk"

        def is_valid_audio(p: Path) -> bool:
            try:
                return not p.name.startswith("._") and p.stat().st_size > 1000
            except Exception:
                return False

        cambridge_files: List[Path] = []
        vctk_files: List[Path] = []
        for ext in ["*.wav", "*.flac"]:
            if cambridge_dir.exists():
                cambridge_files.extend([f for f in cambridge_dir.rglob(ext) if is_valid_audio(f)])
            if vctk_dir.exists():
                vctk_files.extend([f for f in vctk_dir.rglob(ext) if is_valid_audio(f)])

        all_sources: Dict[str, List[Path]] = {}
        if cambridge_files:
            all_sources["cambridge"] = cambridge_files
        elif vctk_files:
            all_sources["vctk"] = vctk_files
        else:
            any_audio = []
            for ext in ["*.wav", "*.flac"]:
                any_audio.extend([
                    f for f in self.dry_dir.rglob(ext)
                    if is_valid_audio(f) and "soundcloud" not in str(f).lower()
                ])
            if any_audio:
                all_sources["general"] = any_audio
            else:
                print(f"[-] Исходные сухие файлы не найдены в директории: {self.dry_dir}")
                return []

        print("[*] Доступные студийные источники сухого вокала (SoundCloud отключен):")
        source_splits: Dict[str, Tuple[List[Path], List[Path]]] = {}
        for s_name, s_list in all_sources.items():
            tr_f, val_f = self._split_files_by_group(s_list, val_ratio=0.15)
            source_splits[s_name] = (tr_f, val_f)
            print(f"    - {s_name}: всего {len(s_list)} файлов (Train: {len(tr_f)} | Val: {len(val_f)})")

        tasks = []
        source_keys = list(source_splits.keys())
        n_val_samples = max(1, int(round(max_samples * 0.15))) if max_samples >= 5 else 0
        n_train_samples = max_samples - n_val_samples

        for i in range(max_samples):
            split_name = "train" if i < n_train_samples else "val"
            src_type = random.choice(source_keys)
            tr_pool, val_pool = source_splits[src_type]
            pool = tr_pool if split_name == "train" else val_pool
            src_file = random.choice(pool)
            # Сохраняем WAV для первых 50 сэмплов (или для всех при малом тесте max_samples <= 50)
            write_wav = save_all_wav or (max_samples <= 50) or (i < 30)
            tasks.append((i, src_file, src_type, split_name, self.sr, str(self.output_dir), write_wav))

        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        num_cores = max(1, multiprocessing.cpu_count())
        print(f"[*] Запуск параллельной генерации {max_samples} сэмплов на {num_cores} ядрах CPU...")

        samples = []
        index_file = self.output_dir / "index.jsonl"

        with open(index_file, "w", encoding="utf-8") as f_index:
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                for result_line in tqdm(executor.map(_generate_sample_worker, tasks), total=len(tasks)):
                    if result_line is not None:
                        f_index.write(result_line + "\n")
                        f_index.flush()
                        samples.append(DatasetSample.from_jsonl_line(result_line))

        stats = {
            "total_samples": len(samples),
            "train_samples": sum(1 for s in samples if s.split == "train"),
            "val_samples": sum(1 for s in samples if s.split == "val"),
            "sr": self.sr,
            "chain_order": CHAIN_ORDER,
            "param_layout": PARAM_LAYOUT
        }
        with open(self.output_dir / "stats.json", "w", encoding="utf-8") as f_stats:
            json.dump(stats, f_stats, indent=4)

        print(f"[+] Генерация датасета успешно завершена ({len(samples)} сэмплов). Сохранено в: {self.output_dir}")
        return samples


def _generate_sample_worker(task) -> str | None:
    """
    Автономный воркер генерации одного сэмпла (Dry/Wet + Mid/Side Mel + 18 DSP дескрипторов).
    """
    idx, src_file, source_name, split_name, sr, output_dir_str, write_wav = task

    import warnings
    warnings.filterwarnings("ignore")

    output_dir = Path(output_dir_str)
    dry_out_dir = output_dir / "audio" / "dry"
    wet_out_dir = output_dir / "audio" / "wet"
    features_out_dir = output_dir / "features"

    audio_processor = AudioProcessor(sr=sr)
    dsp = DSPEngine()
    rng = random.Random(100000 + idx * 7919)

    try:
        # 1. Загрузка и LUFS-нормализация сухого сигнала
        y_dry = audio_processor.load_audio(src_file, target_lufs=-23.0, trim_silence=True)

        # Оптимизация длины: 6.0 секунд (~517 мел-фреймов) — быстрее в рендере и богато по фразам
        duration = 6.0
        target_samples = int(duration * sr)
        if len(y_dry) > target_samples:
            max_offset = len(y_dry) - target_samples
            best_start = rng.randint(0, max_offset)
            best_rms = -1.0
            for _ in range(4):
                start = rng.randint(0, max_offset)
                chunk = y_dry[start : start + target_samples]
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                if rms > best_rms:
                    best_rms = rms
                    best_start = start
            y_dry = y_dry[best_start : best_start + target_samples]
        elif len(y_dry) < target_samples:
            pad_width = target_samples - len(y_dry)
            y_dry = np.pad(y_dry, (0, pad_width), mode='constant')

        # 2. Генерация сбалансированной цепочки плагинов и честных параметров
        chain = _generate_balanced_random_config(rng)

        # 3. Оффлайн-рендеринг сигнала БЕЗ скрытого обрезания параметров (apply_safety_clamps=False)
        rand_key = rng.randint(0, 11)
        rand_scale = rng.choice(["major", "minor"])
        y_wet = dsp.apply_chain(
            y_dry,
            sr,
            chain,
            target_key_index=rand_key,
            target_scale=rand_scale,
            apply_safety_clamps=False
        )

        # Легкая имитация полосы пропускания Demucs/MP3 на 25% тренировочных референсов (Domain Adaptation)
        y_wet_for_features = y_wet
        if split_name == "train" and rng.random() < 0.25:
            cutoff_demucs = rng.uniform(15500.0, 18000.0)
            sos_lp = signal.butter(4, cutoff_demucs, btype='lowpass', fs=sr, output='sos')
            y_wet_for_features = signal.sosfilt(sos_lp, y_wet).astype(np.float32)

        # 4. Опциональное сохранение аудиофайлов на диск
        sample_id = f"{idx:06d}"
        dry_name = f"{source_name}_{sample_id}_dry.wav"
        wet_name = f"{source_name}_{sample_id}_wet.wav"

        dry_path = dry_out_dir / dry_name
        wet_path = wet_out_dir / wet_name

        if write_wav:
            sf.write(dry_path, y_dry.T if y_dry.ndim == 2 else y_dry, sr)
            sf.write(wet_path, y_wet.T if y_wet.ndim == 2 else y_wet, sr)

        # 5. Извлечение двухканальных Mid/Side Мел-спектрограмм (2, 128, T) и 18 физических DSP-признаков
        mel_dry = audio_processor.get_stereo_mel_spectrogram(y_dry, n_mels=128)
        mel_wet = audio_processor.get_stereo_mel_spectrogram(y_wet_for_features, n_mels=128)
        desc_dry = audio_processor.extract_mixing_descriptors(y_dry)
        desc_wet = audio_processor.extract_mixing_descriptors(y_wet_for_features)

        feature_file = features_out_dir / f"{sample_id}.npz"
        np.savez_compressed(
            feature_file,
            mel_dry=mel_dry,
            mel_wet=mel_wet,
            desc_dry=desc_dry,
            desc_wet=desc_wet
        )

        # 6. Сохранение метаданных сэмпла
        chain_list = [p.name for p in chain.plugins]
        chain_onehot = [1 if p in chain_list else 0 for p in CHAIN_ORDER]

        params_dict = {plugin.name: plugin.params for plugin in chain.plugins}

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
            params_vector=flat_vector,
            split=split_name
        )
        return sample.to_jsonl_line()
    except Exception as e:
        print(f"\n[-] Ошибка генерации сэмпла {idx} из файла {src_file.name}: {e}")
        return None
