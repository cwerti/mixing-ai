"""
Бенчмарк подбора цепочки на парах «сухой вокал + чужой референс».

Синтетические пары собираются так же, как выглядит реальная задача: референс — другой голос из другой песни,
обработанный известной цепочкой. Эта цепочка, примененная к исходнику, дает ориентир oracle — «правильную
обработку». Это не нижняя граница: метрика стиля видит и тембр самих голосов, поэтому подбор с EQ может уйти
ниже oracle, подгоняя один голос под другой. Реальные пары (без oracle) задаются манифестом.

Сравниваются режимы:
  - optimizer: правила + CMA-ES без нейросети;
  - model+optimizer: то же, но стартовая точка, шаг поиска и вероятности берутся у MixingAIModel;
  - model: цепочка модели без подбора (как scripts/process_vocal.py --no-optimize).
"""
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import soundfile as sf

from app.audio.audio_processor import AudioProcessor, SpectralMatcher
from app.audio.dsp_engine import DSPEngine
from app.audio.vocal_enhancer import VocalEnhancer
from app.dataset.dataset_generator import DatasetGenerator, _generate_balanced_random_config, is_dry_source
from app.dataset.dataset_schema import CHAIN_ORDER, PLUGIN_ON_PROB, ChainConfig
from app.eval.style_metrics import extract_style_features, style_distance
from app.mixing.chain_optimizer import CHAIN_INPUT_LUFS, ChainOptimizer, normalize_lufs, select_excerpt, to_stereo

MODES = ("optimizer", "model+optimizer", "model")
# Допуск по целевой функции при сравнении скорости режимов (меньше шума одного рендера)
LEVEL_TOLERANCE = 0.02


@dataclass
class BenchmarkCase:
    """Пара для бенчмарка. src — стерео на уровне CHAIN_INPUT_LUFS, ref — референсный вокал."""
    name: str
    src: np.ndarray
    ref: np.ndarray
    key_index: int
    scale: str
    oracle_chain: ChainConfig | None = None


def load_stereo(path: str | Path, sr: int, seconds: float | None = None) -> np.ndarray:
    """Загружает файл в стерео (2, N), при необходимости оставляя самый громкий фрагмент."""
    y, _ = librosa.load(path, sr=sr, mono=False)
    y = to_stereo(y)
    return select_excerpt(y, sr, seconds) if seconds else y


def make_case(name: str, dry_src: np.ndarray, dry_ref: np.ndarray, chain: ChainConfig, sr: int,
              enhancer: VocalEnhancer | None = None) -> BenchmarkCase:
    """Синтетическая пара: референс = chain(dry_ref), тональность для автотюна берется из голоса референса."""
    enhancer = enhancer or VocalEnhancer(sr=sr)
    dry_ref = normalize_lufs(to_stereo(dry_ref), sr, CHAIN_INPUT_LUFS)
    key_index, scale = enhancer.detect_key_and_scale(librosa.to_mono(dry_ref))
    ref = DSPEngine().apply_chain(dry_ref, sr, chain, target_key_index=key_index, target_scale=scale)
    src = normalize_lufs(to_stereo(dry_src), sr, CHAIN_INPUT_LUFS)
    return BenchmarkCase(name, src, ref, key_index, scale, oracle_chain=chain)


def _song_groups(files: list[Path]) -> dict[str, list[Path]]:
    """Группирует файлы по песне тем же ключом, что и при разбиении датасета на train/val."""
    groups: dict[str, list[Path]] = {}
    for f in sorted(files):
        groups.setdefault(DatasetGenerator.song_key(f), []).append(f)
    return groups


def synthetic_cases(dry_dir: str | Path, n: int, sr: int = 44100, seconds: float = 30.0, seed: int = 0,
                    log: Callable[[str], None] = print) -> list[BenchmarkCase]:
    """
    Собирает n пар из песен валидационной части датасета (тот же split, что при обучении, чтобы модель
    не видела эти голоса). Исходник и референс всегда из разных песен; берутся лид-вокалы, если они есть.
    """
    files = [f for ext in ("*.wav", "*.flac") for f in Path(dry_dir).rglob(ext) if is_dry_source(f)]
    if not files:
        raise FileNotFoundError(f"Сухие вокалы не найдены в {dry_dir}")
    _, val_files = DatasetGenerator._split_files_by_group(files, val_ratio=0.15)
    lead = [f for f in val_files if "lead" in f.stem.lower()]
    groups = _song_groups(lead or val_files)
    if len(groups) < 2:
        raise ValueError("Для пар из разных песен нужно минимум две песни в валидационной части")

    rng = random.Random(seed)
    enhancer = VocalEnhancer(sr=sr)
    cases = []
    song_keys = sorted(groups)
    for i in range(n):
        src_song, ref_song = rng.sample(song_keys, 2)
        src_file, ref_file = rng.choice(groups[src_song]), rng.choice(groups[ref_song])
        chain = _generate_balanced_random_config(rng)
        name = f"syn{i:02d}_{src_file.stem}__{ref_file.stem}"
        log(f"[*] {name}: {[p.name for p in chain.plugins]}")
        cases.append(make_case(name, load_stereo(src_file, sr, seconds), load_stereo(ref_file, sr, seconds),
                               chain, sr, enhancer))
    return cases


def manifest_cases(manifest: str | Path, sr: int = 44100, seconds: float | None = None) -> list[BenchmarkCase]:
    """Реальные пары из JSON-манифеста: [{"name": ..., "source": "...wav", "ref": "...wav"}, ...]."""
    entries = json.loads(Path(manifest).read_text(encoding="utf-8"))
    base = Path(manifest).parent
    enhancer = VocalEnhancer(sr=sr)
    cases = []
    for entry in entries:
        src = normalize_lufs(load_stereo(base / entry["source"], sr, seconds), sr, CHAIN_INPUT_LUFS)
        ref = load_stereo(base / entry["ref"], sr, seconds)
        key_index, scale = enhancer.detect_key_and_scale(librosa.to_mono(ref))
        cases.append(BenchmarkCase(entry["name"], src, ref, key_index, scale))
    return cases


def renders_to_level(history: list[tuple[int, float]], level: float) -> int | None:
    """Сколько рендеров понадобилось, чтобы лучшая целевая функция опустилась до level (None — не дошла)."""
    return next((n for n, value in history if value <= level), None)


def renders_to_fraction(history: list[tuple[int, float]], fraction: float = 0.9) -> int | None:
    """Сколько рендеров понадобилось CMA-ES, чтобы пройти fraction всего улучшения целевой функции."""
    if not history:
        return None
    start, final = history[0][1], history[-1][1]
    if start <= final:
        return history[0][0]
    target = start - fraction * (start - final)
    return next(n for n, value in history if value <= target)


class ChainBenchmark:
    """Прогоняет режимы подбора на парах и меряет итоговое расстояние по стилю на всем исходнике."""

    def __init__(self, sr: int = 44100, prior: Callable | None = None, with_pitch: bool = True,
                 log: Callable[[str], None] = print, metric: str = "style"):
        """
        Args:
            prior: Функция (y_src, y_ref) -> (model_probs, model_params, model_param_std) по сырым сигналам,
                например MixingAIMLPipeline.predict_prior. Без нее режимы с моделью пропускаются.
            with_pitch: Учитывать ли отклонение от нот (pyin медленный).
            metric: Целевая функция подбора в ChainOptimizer (style / embedding / combined). Колонка
                distance в результатах всегда style_distance, чтобы прогоны были сравнимы.
        """
        self.metric = metric
        self.sr = sr
        self.prior = prior
        self.with_pitch = with_pitch
        self.log = log
        self.dsp = DSPEngine()
        self.enhancer = VocalEnhancer(sr=sr)
        self.processor = AudioProcessor(sr=sr)
        self.matcher = SpectralMatcher(sr=sr)

    def render(self, case: BenchmarkCase, chain: ChainConfig, gate_threshold_db: float | None) -> np.ndarray:
        """Рендер исходника как в process_vocal.py: подавление резонансов, затем цепочка."""
        cleaned = np.vstack([self.enhancer.suppress_resonances(ch) for ch in case.src]).astype(np.float32)
        return self.dsp.apply_chain(cleaned, self.sr, chain, gate_threshold_db=gate_threshold_db,
                                    target_key_index=case.key_index, target_scale=case.scale)

    def _distance(self, y: np.ndarray, ref_features: dict) -> tuple[float, dict]:
        return style_distance(extract_style_features(y, self.sr, with_pitch=self.with_pitch), ref_features)

    def run_case(self, case: BenchmarkCase, modes: tuple[str, ...] = MODES, budget_s: float = 60.0,
                 seed: int = 0, excerpt_s: float = 12.0, audio_dir: Path | None = None,
                 with_baselines: bool = True, patience_renders: int | None = None) -> list[dict]:
        """
        Возвращает по строке результатов на каждый режим.

        Args:
            with_baselines: Добавить строки source и oracle (от seed не зависят, достаточно одного раза на пару).
            patience_renders: Ранняя остановка CMA-ES (см. ChainOptimizer.optimize).
        """
        ref_features = extract_style_features(case.ref, self.sr, with_pitch=self.with_pitch)
        gate = self.processor.estimate_noise_gate_threshold(librosa.to_mono(case.src))
        rows = []
        # Расстояние до исходника, обработанного истинной цепочкой: голос тот же, поэтому оно меряет только
        # обработку, без разницы тембров певцов, которую цепочка повторять не должна (например, регистр в EQ).
        # Гейт тот же, что у режимов (и в process_vocal.py): иначе в distance_true попадает сам гейт, а не цепочка
        oracle_y = self.render(case, case.oracle_chain, gate) if case.oracle_chain is not None else None
        true_features = (extract_style_features(oracle_y, self.sr, with_pitch=self.with_pitch)
                         if oracle_y is not None else None)

        def add(mode: str, y: np.ndarray, chain: ChainConfig | None, seconds: float = 0.0,
                history: list[tuple[int, float]] | None = None):
            distance, per_feature = self._distance(y, ref_features)
            rows.append({
                "case": case.name, "mode": mode, "seed": seed, "distance": distance,
                "distance_true": self._distance(y, true_features)[0] if true_features is not None else None,
                "per_feature": per_feature, "seconds": seconds,
                "plugins": [p.name for p in chain.plugins] if chain else [],
                "chain": [{"name": p.name, "params": dict(p.params)} for p in chain.plugins] if chain else [],
                "renders": history[-1][0] if history else None,
                "renders_to_90": renders_to_fraction(history) if history else None,
                "history": history or [],
            })
            self.log(f"    {mode:16s} расстояние {distance:5.2f}"
                     + (f" | {seconds:4.0f} с" if seconds else "")
                     + (f" | {rows[-1]['plugins']}" if chain else ""))
            if audio_dir is not None:
                audio_dir.mkdir(parents=True, exist_ok=True)
                sf.write(audio_dir / f"{case.name}_{mode.replace('+', '_')}_s{seed}.wav", y.T, self.sr)

        self.log(f"[*] {case.name} (тональность {case.key_index} {case.scale}, seed {seed})")
        if with_baselines:
            add("source", case.src, None)
        if with_baselines and oracle_y is not None:
            add("oracle", oracle_y, case.oracle_chain)

        model_probs = model_params = model_param_std = None
        if self.prior is not None and any(m.startswith("model") for m in modes):
            model_probs, model_params, model_param_std = self.prior(case.src, case.ref)

        optimizer = ChainOptimizer(sr=self.sr, log=lambda _: None, metric=self.metric)
        for mode in modes:
            started = time.perf_counter()
            if mode == "model":
                if model_probs is None:
                    continue
                from app.ml.pipeline import model_chain
                plugins = [name for name in CHAIN_ORDER if name != "eq" and model_probs[name] > PLUGIN_ON_PROB]
                chain = model_chain(plugins, self.matcher.match_eq(case.src, case.ref), model_params)
                add(mode, self.render(case, chain, gate), chain, time.perf_counter() - started)
                continue
            use_model = mode == "model+optimizer"
            if use_model and model_probs is None:
                continue
            result = optimizer.optimize(
                case.src, case.ref, model_probs=model_probs if use_model else None,
                model_params=model_params if use_model else None,
                model_param_std=model_param_std if use_model else None, key_index=case.key_index, scale=case.scale,
                gate_threshold_db=gate, budget_s=budget_s, excerpt_s=excerpt_s, seed=seed,
                patience_renders=patience_renders,
            )
            add(mode, self.render(case, result.chain, gate), result.chain, time.perf_counter() - started,
                result.history)
        return rows


def summarize(rows: list[dict]) -> str:
    """Сводная таблица (Markdown) по режимам: расстояния, доля от исходника, скорость сходимости, победы."""
    by_mode: dict[str, list[dict]] = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row)
    source = {r["case"]: r["distance"] for r in by_mode.get("source", [])}

    lines = ["| Режим | N | Расстояние (среднее) | Медиана | От исходника | До истинной цепочки | "
             "Рендеров до 90% | Время, с |",
             "|---|---|---|---|---|---|---|---|"]
    for mode, items in by_mode.items():
        distances = np.array([r["distance"] for r in items])
        ratios = [r["distance"] / source[r["case"]] for r in items if source.get(r["case"], 0) > 0]
        to_true = [r["distance_true"] for r in items if r.get("distance_true") is not None]
        to_90 = [r["renders_to_90"] for r in items if r["renders_to_90"] is not None]
        seconds = [r["seconds"] for r in items if r["seconds"]]
        lines.append(f"| {mode} | {len(items)} | {distances.mean():.2f} | {np.median(distances):.2f} | "
                     + (f"{np.mean(ratios):.0%}" if ratios else "—") + " | "
                     + (f"{np.mean(to_true):.2f}" if to_true else "—") + " | "
                     + (f"{np.mean(to_90):.0f}" if to_90 else "—") + " | "
                     + (f"{np.mean(seconds):.0f}" if seconds else "—") + " |")

    pairs, histories = {}, {}
    for row in rows:
        pairs.setdefault((row["case"], row["seed"]), {})[row["mode"]] = row["distance"]
        if row.get("history"):
            histories.setdefault((row["case"], row["seed"]), {})[row["mode"]] = row["history"]
    head_to_head = [(p["model+optimizer"], p["optimizer"]) for p in pairs.values()
                    if "model+optimizer" in p and "optimizer" in p]
    if head_to_head:
        wins = sum(with_model < without - 0.05 for with_model, without in head_to_head)
        losses = sum(with_model > without + 0.05 for with_model, without in head_to_head)
        lines.append(f"\nmodel+optimizer против optimizer: лучше {wins}, хуже {losses}, "
                     f"вничью (±0.05) {len(head_to_head) - wins - losses} из {len(head_to_head)}")

    # Ускорение: за сколько рендеров каждый режим доходит до итогового уровня optimizer на той же паре
    # (целевая функция одна и та же, поэтому уровни сравнимы, в отличие от «90% собственного улучшения»)
    speed = [(renders_to_level(h["optimizer"], h["optimizer"][-1][1] + LEVEL_TOLERANCE),
              renders_to_level(h["model+optimizer"], h["optimizer"][-1][1] + LEVEL_TOLERANCE))
             for h in histories.values() if "optimizer" in h and "model+optimizer" in h]
    if speed:
        reached = [(base, fast) for base, fast in speed if fast is not None]
        line = (f"\nДо итогового уровня optimizer (±{LEVEL_TOLERANCE}): "
                f"optimizer за {np.median([b for b, _ in speed]):.0f} рендеров (медиана), model+optimizer дошел в {len(reached)} из {len(speed)}")
        if reached:
            ratios = [base / max(fast, 1) for base, fast in reached]
            line += (f" за {np.median([f for _, f in reached]):.0f} рендеров, "
                     f"ускорение {np.median(ratios):.1f}x (медиана по парам)")
        lines.append(line)

    seeds = {r["seed"] for r in rows}
    if len(seeds) > 1:
        lines.append("\nРазброс между seed (среднее по парам стандартное отклонение расстояния):")
        for mode, items in by_mode.items():
            per_case: dict[str, list[float]] = {}
            for r in items:
                per_case.setdefault(r["case"], []).append(r["distance"])
            stds = [np.std(v) for v in per_case.values() if len(v) > 1]
            if stds:
                lines.append(f"  - {mode}: {np.mean(stds):.3f}")
    return "\n".join(lines)
