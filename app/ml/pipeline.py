import argparse
import json
import sys
from pathlib import Path
import librosa
import torch
import numpy as np

# Добавление корня проекта в пути импорта Python
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.audio.audio_processor import AudioProcessor, SpectralMatcher
from app.dataset.dataset_schema import (
    CHAIN_ORDER, PARAM_LAYOUT, PARAM_OFFSETS, DATAGEN_PLUGINS, PLUGIN_ON_PROB, ChainConfig, PluginConfig,
    remap_param_vector
)
from app.mixing.chain_optimizer import eq_plugin
from app.ml.features import model_inputs
from app.ml.model import MixingAIModel
from app.presets.full_preset_generator import FullPresetGenerator


def model_params_by_plugin(pred_params_vec: np.ndarray) -> dict[str, dict[str, float]]:
    """Раскладывает плоский вектор параметров модели по плагинам (нормализованные значения [0..1])."""
    return {
        plugin: {p: float(pred_params_vec[PARAM_OFFSETS[plugin] + i]) for i, p in enumerate(layout)}
        for plugin, layout in PARAM_LAYOUT.items()
    }


def model_chain(plugins: list[str], eq_bands: list[tuple[float, float]],
                model_params: dict[str, dict[str, float]]) -> ChainConfig:
    """Цепочка «как предсказала модель» (режим --no-optimize) с защитными лимитами пространственных эффектов."""
    # drive_db 0.48 = 9.6 дБ при диапазоне 0–20 дБ (тот же физический предел, что 0.35 при прежнем 4–20 дБ)
    limits = {("chorus", "depth"): 0.20, ("distortion", "drive_db"): 0.48,
              ("reverb", "wet_level"): 0.40, ("delay", "mix"): 0.45}
    chain = ChainConfig(plugins=[eq_plugin(eq_bands)])
    for plugin in plugins:
        params = {p: min(v, limits.get((plugin, p), 1.0)) for p, v in model_params[plugin].items()}
        chain.plugins.append(PluginConfig(name=plugin, params=params))
    return chain


def calibration_path(model_path: str | Path) -> Path:
    """Файл калибровки рядом с весами: data/models/v4/best_model.pth -> best_model.calibration.json."""
    return Path(model_path).with_suffix(".calibration.json")


def load_param_std(model_path: str | Path) -> dict[str, dict[str, float]] | None:
    """Ошибка параметров модели на валидации (scripts/calibrate_prior.py) или None, если калибровки нет."""
    path = calibration_path(model_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["param_std"]


class MixingAIMLPipeline:
    """
    Инференс-пайплайн Mixing-AI v2 (Source + Reference Dual-Input).
    Извлекает Mid/Side мел-спектрограммы и 18 физических DSP-дескрипторов
    из входного сухого вокала и референса, запускает предсказание модели
    и генерирует файл пресета .fst для FL Studio.
    """

    def __init__(self, model_path: str | Path, max_len: int = 520):
        self.max_len = max_len
        self.processor = AudioProcessor(sr=44100)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MixingAIModel().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"]
        model_dict = self.model.state_dict()

        filtered_dict = {
            k: v for k, v in state_dict.items()
            if k in model_dict and model_dict[k].shape == v.shape
        }
        model_dict.update(filtered_dict)
        self.model.load_state_dict(model_dict)
        self.model.eval()
        self.param_std = load_param_std(model_path)
        # Диапазоны параметров, в которых обучен чекпоинт (нет у v4 и старше — тогда LEGACY_PARAM_RANGES)
        self.param_ranges = checkpoint.get("param_ranges")
        print(f"[+] Модель Mixing-AI v2 загружена на {self.device} (эпоха: {checkpoint.get('epoch', 'N/A')})")

    def predict(self, y_src: np.ndarray, y_ref: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
        """
        Выполняет двухвходовой инференс (Source + Reference) по сигналам любой длины, громкости и канальности
        (подготовка как в генераторе датасета, см. app/ml/features.py) и возвращает:
          - active_plugins: список имен активных плагинов
          - probs: массив вероятностей по всем плагинам CHAIN_ORDER
          - pred_params_vec: плоский вектор предсказанных параметров [0..1]
        """
        x_spec, x_desc = model_inputs(y_src, y_ref, self.processor, self.max_len)
        x_spec_t = torch.from_numpy(x_spec).unsqueeze(0).to(self.device)
        x_desc_t = torch.from_numpy(x_desc).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, pred_params = self.model(x_spec_t, None, x_desc_t)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            pred_params_vec = remap_param_vector(pred_params.squeeze(0).cpu().numpy(), self.param_ranges)

        active_plugins = [CHAIN_ORDER[i] for i, prob in enumerate(probs) if prob > PLUGIN_ON_PROB]
        if "eq" not in active_plugins:
            active_plugins.insert(0, "eq")

        return active_plugins, probs, pred_params_vec

    def predict_prior(self, y_src: np.ndarray, y_ref: np.ndarray) -> tuple[
            dict[str, float], dict[str, dict[str, float]], dict[str, dict[str, float]] | None]:
        """
        Предсказание в формате ChainOptimizer: вероятности и нормализованные параметры по именам плагинов,
        плюс ошибка параметров на валидации (None без калибровки).
        """
        _, probs, pred_params_vec = self.predict(y_src, y_ref)
        return ({name: float(probs[i]) for i, name in enumerate(CHAIN_ORDER)},
                model_params_by_plugin(pred_params_vec), self.param_std)

    def run(self, source_path: str | Path, ref_path: str | Path, output_preset_name: str = "mixing_ai_predicted.fst") -> Path | None:
        print(f"\n[*] Запуск инференса модели для:\n  Dry: {source_path}\n  Ref: {ref_path}")

        y_src = self.processor.load_audio(source_path, target_lufs=-23.0, trim_silence=True)
        y_ref = self.processor.load_audio(ref_path, target_lufs=-23.0, trim_silence=True)

        active_plugins, _, pred_params_vec = self.predict(librosa.load(source_path, sr=self.processor.sr, mono=False)[0],
                                                          librosa.load(ref_path, sr=self.processor.sr, mono=False)[0])
        print(f"[+] Распознанные активные плагины (вероятность > 50%): {active_plugins}")

        matcher = SpectralMatcher(sr=self.processor.sr)
        eq_bands = matcher.match_eq(y_src, y_ref)

        print("[*] Предсказанные параметры 7-полосного EQ:")
        for idx, (f, g) in enumerate(eq_bands):
            print(f"  Полоса {idx+1}: {f:6.1f} Гц | {g:+5.2f} дБ")

        for plugin_name in CHAIN_ORDER:
            if plugin_name != "eq" and plugin_name in active_plugins and plugin_name in PARAM_LAYOUT:
                print(f"[*] Предсказанные параметры {plugin_name.upper()}:")
                layout = PARAM_LAYOUT[plugin_name]
                offset = PARAM_OFFSETS[plugin_name]
                ranges = DATAGEN_PLUGINS[plugin_name]["params"]

                for idx, p_name in enumerate(layout):
                    val_norm = pred_params_vec[offset + idx]
                    min_v, max_v = ranges[p_name]
                    val_phys = min_v + val_norm * (max_v - min_v)
                    print(f"  - {p_name}: {val_phys:.2f}")

        try:
            generator = FullPresetGenerator()
            return generator.generate_vocal_chain(eq_bands, output_preset_name)
        except Exception as e:
            print(f"[-] Ошибка генерации пресета .fst: {e}")
            return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скрипт инференса Mixing-AI Pipeline v2")
    parser.add_argument("--source", type=str, default="data/raw/source/source.wav", help="Путь к сухому вокалу")
    parser.add_argument("--ref", type=str, default="data/raw/reference/reference.wav", help="Путь к референсному вокалу")
    parser.add_argument("--model-path", type=str, default="data/models/v4/best_model.pth", help="Путь к весам модели")
    parser.add_argument("--output-name", type=str, default="mixing_ai_predicted.fst", help="Имя выходного пресета .fst")

    args = parser.parse_args()

    if not Path(args.model_path).exists():
        print(f"[-] Ошибка: Веса модели не найдены по пути: {args.model_path}.")
        sys.exit(1)

    pipeline = MixingAIMLPipeline(args.model_path)
    pipeline.run(args.source, args.ref, args.output_name)
