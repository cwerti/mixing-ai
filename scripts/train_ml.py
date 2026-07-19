import argparse
import sys
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import matplotlib
import matplotlib.pyplot as plt

# Добавление корня проекта в пути импорта Python
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.ml.dataset import MixingAIDataset
from app.ml.model import MixingAIModel, compute_multitask_loss

def draw_and_save_plot(epoch_axis, train_loss_hist, val_loss_hist, save_dir):
    """
    Генерирует и перезаписывает график с кривыми обучения на диск.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_axis, train_loss_hist, label='Train Loss (Обучение)', color='#1f77b4', marker='o')
    plt.plot(epoch_axis, val_loss_hist, label='Val Loss (Валидация)', color='#d62728', marker='x')
    plt.xlabel('Эпоха')
    plt.ylabel('Значение лосса')
    plt.title('Динамика обучения нейросети Mixing-AI')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plot_path = Path(save_dir) / "loss_plot.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

def train(args):
    """
    Запускает цикл обучения с ранней остановкой, планировщиком LR,
    интерактивным окном отображения графиков и логированием в TensorBoard.
    """
    # Если интерактивный режим отключен, используем Headless бэкенд для Matplotlib
    if not args.show_plot:
        matplotlib.use('Agg')
        
    print("=== ЗАПУСК ОБУЧЕНИЯ УЛУЧШЕННОЙ МОДЕЛИ MIXING-AI ===")
    
    # 1. Настройка устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Используемое устройство: {device}")
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Инициализация TensorBoard SummaryWriter
    tb_dir = Path("data/runs")
    tb_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"[*] Логи TensorBoard записываются в: {tb_dir}")
    
    # 2. Загрузка датасета
    dataset = MixingAIDataset(args.dataset_dir, max_len=args.max_len)
    if len(dataset) < 10:
        print("[-] Ошибка: Для обучения улучшенной модели нужно больше сэмплов в датасете.")
        sys.exit(1)
        
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    # Использование фиксированного seed для воспроизводимого сплита
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"[*] Всего сэмплов: {len(dataset)} | Train: {len(train_set)} | Val: {len(val_set)}")
    
    # 3. Инициализация ResNet модели и AdamW
    model = MixingAIModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_loss = float("inf")
    patience_counter = 0
    patience = args.patience
    
    train_loss_hist = []
    val_loss_hist = []
    epoch_axis = []
    
    # Инициализация интерактивного окна Matplotlib, если флаг активен
    fig, ax = None, None
    if args.show_plot:
        plt.ion() # Включение интерактивного режима
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title('Обучение Mixing-AI в реальном времени')
        ax.set_xlabel('Эпоха')
        ax.set_ylabel('Значение лосса')
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.show(block=False)
    
    # 4. Цикл эпох
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        # Обучение
        model.train()
        train_loss = 0.0
        train_class = 0.0
        train_reg = 0.0
        
        for batch_x, batch_chain, batch_params, batch_mask in train_loader:
            batch_x = batch_x.to(device)
            batch_chain = batch_chain.to(device)
            batch_params = batch_params.to(device)
            batch_mask = batch_mask.to(device)
            
            optimizer.zero_grad()
            logits, pred_params = model(batch_x, batch_chain)
            loss, loss_class, loss_reg = compute_multitask_loss(
                logits, pred_params, batch_chain, batch_params, batch_mask
            )
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_class += loss_class.item()
            train_reg += loss_reg.item()
            
        train_loss /= len(train_loader)
        train_class /= len(train_loader)
        train_reg /= len(train_loader)
        
        # Валидация
        model.eval()
        val_loss = 0.0
        val_class = 0.0
        val_reg = 0.0
        
        with torch.no_grad():
            for batch_x, batch_chain, batch_params, batch_mask in val_loader:
                batch_x = batch_x.to(device)
                batch_chain = batch_chain.to(device)
                batch_params = batch_params.to(device)
                batch_mask = batch_mask.to(device)
                
                logits, pred_params = model(batch_x, None)
                loss, loss_class, loss_reg = compute_multitask_loss(
                    logits, pred_params, batch_chain, batch_params, batch_mask
                )
                val_loss += loss.item()
                val_class += loss_class.item()
                val_reg += loss_reg.item()
                
        val_loss /= len(val_loader) if len(val_loader) > 0 else 1
        val_class /= len(val_loader) if len(val_loader) > 0 else 1
        val_reg /= len(val_loader) if len(val_loader) > 0 else 1
        
        scheduler.step()
        
        # Запись метрик в TensorBoard
        tb_writer.add_scalar("Loss/Train_Total", train_loss, epoch)
        tb_writer.add_scalar("Loss/Train_Classification", train_class, epoch)
        tb_writer.add_scalar("Loss/Train_Regression", train_reg, epoch)
        tb_writer.add_scalar("Loss/Val_Total", val_loss, epoch)
        tb_writer.add_scalar("Loss/Val_Classification", val_class, epoch)
        tb_writer.add_scalar("Loss/Val_Regression", val_reg, epoch)
        tb_writer.add_scalar("Learning_Rate", scheduler.get_last_lr()[0], epoch)
        
        # Добавляем в историю
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        epoch_axis.append(epoch)
        
        # Сохранение статического графика на диск
        if not args.show_plot:
            draw_and_save_plot(epoch_axis, train_loss_hist, val_loss_hist, save_dir)
        
        # 5. Отрисовка в интерактивном окне
        if args.show_plot and ax is not None:
            ax.clear()
            ax.plot(epoch_axis, train_loss_hist, label='Train Loss (Обучение)', color='#1f77b4', marker='o')
            ax.plot(epoch_axis, val_loss_hist, label='Val Loss (Валидация)', color='#d62728', marker='x')
            ax.set_xlabel('Эпоха')
            ax.set_ylabel('Значение лосса')
            ax.set_title('Обучение Mixing-AI в реальном времени')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.1) # Пауза для рендеринга окна
        
        epoch_duration = time.time() - epoch_start_time
        print(f"Эпоха {epoch:02d}/{args.epochs:02d} | LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Время: {epoch_duration:.1f} сек")
              
        # 6. Ранняя остановка
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = save_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss
            }, best_model_path)
            print(f"  [+] Новая лучшая модель сохранена в: {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[!] Обучение остановлено досрочно на эпохе {epoch}. Лосс на валидации не улучшался {patience} эпох.")
                break

    tb_writer.close()
    if args.show_plot:
        plt.ioff()
        # Сохраняем финальный график при закрытии окна
        draw_and_save_plot(epoch_axis, train_loss_hist, val_loss_hist, save_dir)
        plt.show()
        
    print(f"\n[+] Обучение успешно завершено! Итоговые графики сохранены в {save_dir}/loss_plot.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скрипт обучения многозадачной модели Mixing-AI")
    parser.add_argument("--dataset-dir", type=str, default="data/processed/dataset_v1", help="Путь к сгенерированному датасету")
    parser.add_argument("--save-dir", type=str, default="data/models", help="Папка для сохранения модели")
    parser.add_argument("--epochs", type=int, default=30, help="Максимальное количество эпох обучения")
    parser.add_argument("--patience", type=int, default=7, help="Количество эпох для ранней остановки (patience)")
    parser.add_argument("--batch-size", type=int, default=16, help="Размер батча")
    parser.add_argument("--lr", type=float, default=1e-3, help="Начальная скорость обучения (learning rate)")
    parser.add_argument("--max-len", type=int, default=700, help="Длина спектрограммы Мелов")
    parser.add_argument("--show-plot", action="store_true", help="Отображать интерактивное окно с графиками во время обучения")
    parser.add_argument("--num-workers", type=int, default=0, help="Количество воркеров для параллельной загрузки данных")
    
    args = parser.parse_args()
    train(args)
