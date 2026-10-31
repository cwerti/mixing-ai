import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import matplotlib
import matplotlib.pyplot as plt

# Настройка кодировки консоли Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(errors='replace')

# Добавление корня проекта в пути импорта Python
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.dataset.dataset_schema import CHAIN_ORDER, dataset_param_ranges
from app.ml.dataset import MixingAIDataset
from app.ml.model import MixingAIModel, compute_multitask_loss


def draw_and_save_plot(epoch_axis, train_loss_hist, val_loss_hist, save_dir):
    """
    Генерирует и сохраняет график кривых обучения на диск.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_axis, train_loss_hist, label='Train Loss (Обучение)', color='#1f77b4', marker='o')
    plt.plot(epoch_axis, val_loss_hist, label='Val Loss (Валидация)', color='#d62728', marker='x')
    plt.xlabel('Эпоха')
    plt.ylabel('Значение лосса')
    plt.title('Динамика обучения нейросети Mixing-AI v2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    plot_path = Path(save_dir) / "loss_plot.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()


def compute_class_pos_weights(samples: list, device: torch.device) -> torch.Tensor:
    """
    Вычисляет балансировочные веса положительных классов pos_weight = N_neg / N_pos для BCEWithLogitsLoss,
    чтобы модель одинаково хорошо распознавала как частые, так и более редкие плагины.
    """
    num_classes = len(CHAIN_ORDER)
    pos_counts = np.zeros(num_classes, dtype=np.float64)
    total = max(1, len(samples))

    for s in samples:
        onehot = s.get("chain_onehot", [])
        for i in range(min(num_classes, len(onehot))):
            pos_counts[i] += float(onehot[i])

    neg_counts = total - pos_counts
    weights = np.ones(num_classes, dtype=np.float32)
    for i in range(num_classes):
        if pos_counts[i] > 0 and neg_counts[i] > 0:
            weights[i] = float(np.clip(neg_counts[i] / pos_counts[i], 0.4, 3.0))
        else:
            weights[i] = 1.0

    return torch.from_numpy(weights).to(device)


def train(args):
    """
    Запускает цикл обучения модели Mixing-AI v2 (Dual-Input Mid/Side + 18 DSP Descriptors).
    """
    if not args.show_plot:
        matplotlib.use('Agg')

    print("=== ЗАПУСК ОБУЧЕНИЯ ГИБРИДНОЙ МОДЕЛИ MIXING-AI v2 ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Используемое устройство: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tb_dir = Path("data/runs")
    tb_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(tb_dir))

    # Загрузка train и val сплитов без утечки вокалистов
    # Диапазоны берутся из датасета, а не из текущей схемы: схема могла поменяться после генерации
    param_ranges, ranges_source = dataset_param_ranges(args.dataset_dir)
    if ranges_source == "schema":
        print("[!] В stats.json датасета нет param_ranges: в чекпоинт записываются текущие диапазоны схемы. "
              "Если DATAGEN_PLUGINS менялся после генерации, предсказания будут пересчитаны неверно.")
    else:
        print(f"[*] Диапазоны параметров: {ranges_source}")
    train_set = MixingAIDataset(args.dataset_dir, max_len=args.max_len, split="train", augment=True)
    val_set = MixingAIDataset(args.dataset_dir, max_len=args.max_len, split="val", augment=False)

    # Откат к random_split только если датасет старого образца без разметки split="val"
    if len(val_set) == len(train_set) and all(s.get("split", "train") == "train" for s in train_set.samples):
        full_set = MixingAIDataset(args.dataset_dir, max_len=args.max_len, split="all", augment=False)
        train_size = int(0.85 * len(full_set))
        val_size = len(full_set) - train_size
        generator = torch.Generator().manual_seed(42)
        train_set, val_set = random_split(full_set, [train_size, val_size], generator=generator)
        pos_weight = compute_class_pos_weights(full_set.samples, device)
    else:
        pos_weight = compute_class_pos_weights(train_set.samples, device)
    if args.pos_weight == "none":
        # Частоты эффектов в датасете уже реалистичные (dataset_v4+): балансировка N_neg/N_pos (до 3x) вернула бы
        # редким chorus/distortion эффективную априорную частоту ~0.4–0.5 и с ней лишние ложные включения
        pos_weight = None
    print(f"[*] pos_weight: {args.pos_weight}"
          + (f" {[round(w, 2) for w in pos_weight.tolist()]}" if pos_weight is not None else ""))

    if len(train_set) < 4:
        print("[-] Ошибка: Для обучения модели нужно больше сэмплов в датасете.")
        sys.exit(1)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=(len(train_set) > args.batch_size),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    print(f"[*] Train сэмплов: {len(train_set)} | Val сэмплов: {len(val_set)}")

    model = MixingAIModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    patience_counter = 0
    patience = args.patience

    train_loss_hist = []
    val_loss_hist = []
    epoch_axis = []

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        train_class = 0.0
        train_reg = 0.0

        for batch_spec, batch_desc, batch_chain, batch_params, batch_mask in train_loader:
            batch_spec = batch_spec.to(device)
            batch_desc = batch_desc.to(device)
            batch_chain = batch_chain.to(device)
            batch_params = batch_params.to(device)
            batch_mask = batch_mask.to(device)

            optimizer.zero_grad()
            logits, pred_params = model(batch_spec, batch_chain, batch_desc)
            loss, loss_class, loss_reg = compute_multitask_loss(
                logits, pred_params, batch_chain, batch_params, batch_mask, pos_weight=pos_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_loss += loss.item()
            train_class += loss_class.item()
            train_reg += loss_reg.item()

        n_tr = max(1, len(train_loader))
        train_loss /= n_tr
        train_class /= n_tr
        train_reg /= n_tr

        # Валидация
        model.eval()
        val_loss = 0.0
        val_class = 0.0
        val_reg = 0.0
        all_pred_chain = []
        all_true_chain = []
        abs_param_errors = []

        with torch.no_grad():
            for batch_spec, batch_desc, batch_chain, batch_params, batch_mask in val_loader:
                batch_spec = batch_spec.to(device)
                batch_desc = batch_desc.to(device)
                batch_chain = batch_chain.to(device)
                batch_params = batch_params.to(device)
                batch_mask = batch_mask.to(device)

                logits, pred_params = model(batch_spec, None, batch_desc)
                loss, loss_class, loss_reg = compute_multitask_loss(
                    logits, pred_params, batch_chain, batch_params, batch_mask, pos_weight=pos_weight
                )
                val_loss += loss.item()
                val_class += loss_class.item()
                val_reg += loss_reg.item()

                probs = torch.sigmoid(logits).cpu().numpy()
                all_pred_chain.append((probs > 0.5).astype(np.float32))
                all_true_chain.append(batch_chain.cpu().numpy())

                diff_abs = torch.abs(pred_params - batch_params).cpu().numpy()
                mask_np = batch_mask.cpu().numpy()
                active_errs = diff_abs[mask_np == 1.0]
                if len(active_errs) > 0:
                    abs_param_errors.extend(active_errs.tolist())

        n_val = max(1, len(val_loader))
        val_loss /= n_val
        val_class /= n_val
        val_reg /= n_val

        # Расчет Macro-F1 и MAE на валидации
        pred_arr = np.vstack(all_pred_chain) if all_pred_chain else np.zeros((1, len(CHAIN_ORDER)))
        true_arr = np.vstack(all_true_chain) if all_true_chain else np.zeros((1, len(CHAIN_ORDER)))
        f1_list = []
        for idx_c in range(len(CHAIN_ORDER)):
            tp = np.sum((pred_arr[:, idx_c] == 1) & (true_arr[:, idx_c] == 1))
            fp = np.sum((pred_arr[:, idx_c] == 1) & (true_arr[:, idx_c] == 0))
            fn = np.sum((pred_arr[:, idx_c] == 0) & (true_arr[:, idx_c] == 1))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
            f1_c = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1_list.append(f1_c)
        val_macro_f1 = float(np.mean(f1_list))
        val_mae = float(np.mean(abs_param_errors)) if abs_param_errors else 0.0

        scheduler.step()

        tb_writer.add_scalar("Loss/Train_Total", train_loss, epoch)
        tb_writer.add_scalar("Loss/Val_Total", val_loss, epoch)
        tb_writer.add_scalar("Metrics/Val_Macro_F1", val_macro_f1, epoch)
        tb_writer.add_scalar("Metrics/Val_Param_MAE", val_mae, epoch)

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        epoch_axis.append(epoch)

        if not args.show_plot:
            draw_and_save_plot(epoch_axis, train_loss_hist, val_loss_hist, save_dir)

        epoch_duration = time.time() - epoch_start_time
        print(
            f"Эпоха {epoch:02d}/{args.epochs:02d} | LR: {scheduler.get_last_lr()[0]:.6f} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Macro-F1: {val_macro_f1:.3f} | Val MAE: {val_mae:.4f} | {epoch_duration:.1f}с"
        )

        # Целевой скор модели: минимизируем MAE параметров и одновременно максимизируем Macro-F1
        val_score = val_mae - 0.15 * val_macro_f1
        if val_score < best_val_loss:
            best_val_loss = val_score
            patience_counter = 0
            best_model_path = save_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_macro_f1": val_macro_f1,
                "val_mae": val_mae,
                # Диапазоны, в которых нормализованы параметры (MixingAIMLPipeline пересчитывает их в текущие)
                "param_ranges": param_ranges,
                "dataset_dir": str(args.dataset_dir),
                "pos_weight_mode": args.pos_weight
            }, best_model_path)
            print(f"  [+] Новая лучшая модель сохранена (F1={val_macro_f1:.3f}, MAE={val_mae:.4f}) -> {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[!] Ранняя остановка на эпохе {epoch} (patience={patience}).")
                break

    tb_writer.close()
    print(f"\n[+] Обучение завершено! График сохранен в {save_dir}/loss_plot.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение гибридной модели Mixing-AI v2")
    parser.add_argument("--dataset-dir", type=str, default="data/processed/dataset_v5", help="Путь к датасету")
    parser.add_argument("--save-dir", type=str, default="data/models", help="Папка сохранения модели")
    parser.add_argument("--epochs", type=int, default=35, help="Количество эпох")
    parser.add_argument("--patience", type=int, default=8, help="Терпение ранней остановки")
    parser.add_argument("--batch-size", type=int, default=32, help="Размер батча")
    parser.add_argument("--lr", type=float, default=1e-3, help="Начальный Learning Rate")
    parser.add_argument("--max-len", type=int, default=520, help="Длина спектрограммы по времени")
    parser.add_argument("--show-plot", action="store_true", help="Интерактивный график")
    parser.add_argument("--num-workers", type=int, default=0, help="Воркеры DataLoader")
    parser.add_argument("--pos-weight", choices=["balanced", "none"], default="balanced",
                        help="Веса положительных классов BCE: balanced = N_neg/N_pos в [0.4, 3] (v4), none — без весов")

    args = parser.parse_args()
    train(args)
