"""
Калибровка подсказки модели для ChainOptimizer: типичная ошибка каждого параметра на валидации.

Ошибка считается только на сэмплах, где плагин активен (на остальных регрессия не обучается).
Оптимизатор берет ее как начальный шаг CMA-ES вокруг старта от модели: где сеть точна, поиск уже.

Пример:
    python scripts/calibrate_prior.py --model-path data/models/v4/best_model.pth --dataset-dir data/processed/dataset_v5

Результат: <model>.calibration.json рядом с весами (читает app.ml.pipeline.load_param_std).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Настройка кодировки консоли Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')

# Добавление корня проекта в пути импорта Python
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.dataset.dataset_schema import PARAM_LAYOUT, PARAM_OFFSETS
from app.ml.dataset import MixingAIDataset
from app.ml.pipeline import MixingAIMLPipeline, calibration_path


def main(args):
    pipeline = MixingAIMLPipeline(args.model_path, max_len=args.max_len)
    val_set = MixingAIDataset(args.dataset_dir, max_len=args.max_len, split="val", augment=False)
    loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"[*] Валидация: {len(val_set)} сэмплов")

    squared, counts = None, None
    with torch.no_grad():
        for x_spec, x_desc, _, y_params, mask in loader:
            _, pred = pipeline.model(x_spec.to(pipeline.device), None, x_desc.to(pipeline.device))
            err = (pred.cpu().numpy() - y_params.numpy()) ** 2 * mask.numpy()
            squared = err.sum(axis=0) if squared is None else squared + err.sum(axis=0)
            counts = mask.numpy().sum(axis=0) if counts is None else counts + mask.numpy().sum(axis=0)

    rmse = np.sqrt(squared / np.maximum(counts, 1))
    param_std, lines = {}, []
    for plugin, layout in PARAM_LAYOUT.items():
        for i, param in enumerate(layout):
            idx = PARAM_OFFSETS[plugin] + i
            if counts[idx] < args.min_samples:
                continue
            param_std.setdefault(plugin, {})[param] = round(float(rmse[idx]), 4)
            lines.append(f"  {plugin:22s} {param:16s} RMSE {rmse[idx]:.3f} (n={int(counts[idx])})")

    out = calibration_path(args.model_path)
    out.write_text(json.dumps({"dataset_dir": str(args.dataset_dir), "n_val": len(val_set),
                               "param_std": param_std}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"[+] Калибровка сохранена: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ошибка параметров модели на валидации для шага CMA-ES")
    parser.add_argument("--model-path", type=str, default="data/models/v4/best_model.pth", help="Веса MixingAIModel")
    parser.add_argument("--dataset-dir", type=str, default="data/processed/dataset_v5", help="Датасет с index.jsonl")
    parser.add_argument("--max-len", type=int, default=520, help="Длина мел-спектрограммы, как при обучении")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--min-samples", type=int, default=30,
                        help="Минимум активных сэмплов, чтобы доверять оценке ошибки параметра")
    main(parser.parse_args())
