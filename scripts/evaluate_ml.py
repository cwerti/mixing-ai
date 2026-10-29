import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split

# Настройка кодировки консоли Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(errors='replace')

# Добавление корня проекта в пути импорта Python
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.ml.dataset import MixingAIDataset
from app.ml.model import MixingAIModel
from app.dataset.dataset_schema import CHAIN_ORDER, PARAM_LAYOUT, PARAM_OFFSETS


def evaluate(args):
    """
    Загружает обученную модель Mixing-AI v2 и вычисляет метрики классификации (Accuracy, Precision, Recall, F1)
    и регрессии параметров (MAE в шкале [0..1]) на независимом валидационном сплите.
    """
    print("=== РАСЧЕТ МЕТРИК И ПРОВЕРКА МОДЕЛИ MIXING-AI v2 ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(args.model_path)

    if not model_path.exists():
        print(f"[-] Файл весов модели не найден: {model_path}")
        return

    val_set = MixingAIDataset(args.dataset_dir, max_len=args.max_len, split="val", augment=False)
    if len(val_set) == 0:
        full_set = MixingAIDataset(args.dataset_dir, max_len=args.max_len, split="all", augment=False)
        train_size = int(0.85 * len(full_set))
        val_size = len(full_set) - train_size
        generator = torch.Generator().manual_seed(42)
        _, val_set = random_split(full_set, [train_size, val_size], generator=generator)

    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
    print(f"[*] Независимых валидационных сэмплов: {len(val_set)}")

    model = MixingAIModel().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_pred_plugins = []
    all_true_plugins = []
    all_pred_params = []
    all_true_params = []
    param_errors = []

    with torch.no_grad():
        for batch_spec, batch_desc, batch_chain, batch_params, batch_mask in val_loader:
            batch_spec = batch_spec.to(device)
            batch_desc = batch_desc.to(device)

            logits, pred_params = model(batch_spec, None, batch_desc)

            probs = torch.sigmoid(logits).cpu().numpy()
            pred_chain = (probs > 0.5).astype(np.float32)
            true_chain = batch_chain.numpy()

            pred_p_np = pred_params.cpu().numpy()
            true_p_np = batch_params.numpy()
            mask_np = batch_mask.numpy()

            for i in range(len(pred_chain)):
                all_pred_plugins.append(pred_chain[i])
                all_true_plugins.append(true_chain[i])
                all_pred_params.append(pred_p_np[i])
                all_true_params.append(true_p_np[i])

                active_diffs = np.abs(pred_p_np[i] - true_p_np[i])[mask_np[i] == 1.0]
                if len(active_diffs) > 0:
                    param_errors.extend(active_diffs.tolist())

    pred_arr = np.array(all_pred_plugins)
    true_arr = np.array(all_true_plugins)

    print("\n--- МЕТРИКИ ТОЧНОСТИ ВЫБОРА ПЛАГИНОВ (Classification Accuracy) ---")
    f1_scores = []
    for idx, plugin_name in enumerate(CHAIN_ORDER):
        acc = np.mean(pred_arr[:, idx] == true_arr[:, idx]) * 100
        tp = np.sum((pred_arr[:, idx] == 1) & (true_arr[:, idx] == 1))
        fp = np.sum((pred_arr[:, idx] == 1) & (true_arr[:, idx] == 0))
        fn = np.sum((pred_arr[:, idx] == 0) & (true_arr[:, idx] == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

        print(f"  {plugin_name:20}: Accuracy: {acc:5.1f}% | Precision: {precision:.2f} | Recall: {recall:.2f} | F1-Score: {f1:.2f}")

    print(f"\n  >>> Сводный Macro F1-Score по всем классам: {np.mean(f1_scores):.3f}")

    mean_mae = float(np.mean(param_errors)) if param_errors else 0.0
    print("\n--- ТОЧНОСТЬ ПАРАМЕТРОВ (Regression Parameter Accuracy) ---")
    print(f"  Средняя Абсолютная Ошибка (MAE) всех параметров: {mean_mae:.4f} (в шкале [0..1] или {mean_mae * 100:.1f}%)")

    for plugin_name in CHAIN_ORDER:
        if plugin_name in PARAM_LAYOUT:
            offset = PARAM_OFFSETS[plugin_name]
            p_names = PARAM_LAYOUT[plugin_name]
            length = len(p_names)

            plugin_errors = []
            idx_in_chain = CHAIN_ORDER.index(plugin_name)

            for true_chain, pred_p, true_p in zip(all_true_plugins, all_pred_params, all_true_params):
                if true_chain[idx_in_chain] == 1.0:
                    for i in range(length):
                        plugin_errors.append(abs(pred_p[offset + i] - true_p[offset + i]))

            if plugin_errors:
                p_mae = float(np.mean(plugin_errors))
                print(f"  - {plugin_name:20}: MAE = {p_mae:.4f} ({p_mae * 100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скрипт проверки метрик модели Mixing-AI v2")
    parser.add_argument("--dataset-dir", type=str, default="data/processed/dataset_v1", help="Путь к датасету")
    parser.add_argument("--model-path", type=str, default="data/models/v4/best_model.pth", help="Путь к чекпоинту модели")
    parser.add_argument("--max-len", type=int, default=520, help="Длина спектрограммы")

    args = parser.parse_args()
    evaluate(args)
