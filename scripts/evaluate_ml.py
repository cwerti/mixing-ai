import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split

# Добавление корня проекта в пути импорта Python
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.ml.dataset import MixingAIDataset
from app.ml.model import MixingAIModel
from app.dataset.dataset_schema import CHAIN_ORDER, PARAM_LAYOUT, PARAM_OFFSETS

def evaluate(args):
    """
    Загружает обученную модель и вычисляет метрики точности
    на валидационном подмножестве данных.
    """
    print("=== РАСЧЕТ МЕТРИК И ПРОВЕРКА МОДЕЛИ MIXING-AI ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(args.model_path)
    
    if not model_path.exists():
        print(f"[-] Файл весов модели не найден: {model_path}")
        print("[*] Пожалуйста, сначала обучите модель с помощью train_ml.py.")
        return

    # 1. Загрузка данных
    dataset = MixingAIDataset(args.dataset_dir, max_len=args.max_len)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    # Используем фиксированный сид для воспроизводимости разделения
    generator = torch.Generator().manual_seed(42)
    _, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    
    print(f"[*] Сэмплов для валидации: {len(val_set)}")
    if len(val_set) == 0:
        print("[-] Ошибка: Валидационная выборка пуста. Увеличьте размер датасета.")
        return

    # 2. Инициализация модели и загрузка чекпоинта
    model = MixingAIModel().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Накопители метрик
    total_samples = 0
    all_pred_plugins = []
    all_true_plugins = []
    all_pred_params = []
    all_true_params = []
    param_errors = []
    
    # 3. Цикл валидации
    with torch.no_grad():
        for batch_x, batch_chain, batch_params, batch_mask in val_loader:
            batch_x = batch_x.to(device)
            batch_chain = batch_chain.to(device)
            batch_params = batch_params.to(device)
            batch_mask = batch_mask.to(device)
            
            # Инференс
            logits, pred_params = model(batch_x, None)
            
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            pred_chain = (probs > 0.5).astype(float)
            true_chain = batch_chain.squeeze(0).cpu().numpy()
            
            all_pred_plugins.append(pred_chain)
            all_true_plugins.append(true_chain)
            
            pred_p_vec = pred_params.squeeze(0).cpu().numpy()
            true_p_vec = batch_params.squeeze(0).cpu().numpy()
            mask_vec = batch_mask.squeeze(0).cpu().numpy()
            
            all_pred_params.append(pred_p_vec)
            all_true_params.append(true_p_vec)
            
            for idx in range(len(mask_vec)):
                if mask_vec[idx] == 1.0:
                    err = abs(pred_p_vec[idx] - true_p_vec[idx])
                    param_errors.append(err)
            
            total_samples += 1

    # 4. Расчет и вывод метрик
    pred_arr = np.array(all_pred_plugins)
    true_arr = np.array(all_true_plugins)
    
    print("\n--- МЕТРИКИ ТОЧНОСТИ ВЫБОРА ПЛАГИНОВ (Classification Accuracy) ---")
    for idx, plugin_name in enumerate(CHAIN_ORDER):
        acc = np.mean(pred_arr[:, idx] == true_arr[:, idx]) * 100
        tp = np.sum((pred_arr[:, idx] == 1) & (true_arr[:, idx] == 1))
        fp = np.sum((pred_arr[:, idx] == 1) & (true_arr[:, idx] == 0))
        fn = np.sum((pred_arr[:, idx] == 0) & (true_arr[:, idx] == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"  {plugin_name:10}: Accuracy: {acc:.1f}% | Precision: {precision:.2f} | Recall: {recall:.2f} | F1-Score: {f1:.2f}")

    mean_mae = np.mean(param_errors) if param_errors else 0.0
    print("\n--- ТОЧНОСТЬ ПАРАМЕТРОВ (Regression Parameter Accuracy) ---")
    print(f"  Средняя Абсолютная Ошибка (MAE) всех параметров: {mean_mae:.4f} (в шкале [0..1] или {mean_mae * 100:.1f}%)")
    
    # Расчет MAE для каждого плагина отдельно (без повторного прогона модели)
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
                p_mae = np.mean(plugin_errors)
                print(f"  - {plugin_name:10}: MAE = {p_mae:.4f} ({p_mae * 100:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скрипт проверки метрик модели Mixing-AI")
    parser.add_argument("--dataset-dir", type=str, default="data/processed/dataset_v1", help="Путь к валидационному датасету")
    parser.add_argument("--model-path", type=str, default="data/models/best_model.pth", help="Путь к чекпоинту модели")
    parser.add_argument("--max-len", type=int, default=700, help="Длина спектрограммы")
    
    args = parser.parse_args()
    evaluate(args)
