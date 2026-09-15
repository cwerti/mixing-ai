import argparse
import sys
import subprocess
from pathlib import Path

# Предотвращение сбоев кодировки вывода в консоль Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(errors='replace')

def run_cmd(args):
    """
    Запускает команду как дочерний процесс и проверяет код возврата.
    """
    print(f"[*] Выполнение: {' '.join(args)}")
    result = subprocess.run(args, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[-] Ошибка при выполнении шага пайплайна (код {result.returncode})")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Mixing-AI: Автоматический пайплайн сборки датасета и обучения модели.")
    parser.add_argument("--max-samples", type=int, default=1000, help="Максимальное количество сэмплов для генерации.")
    parser.add_argument("--epochs", type=int, default=50, help="Количество эпох обучения.")
    parser.add_argument("--batch-size", type=int, default=16, help="Размер батча для обучения.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Скорость обучения (learning rate).")
    parser.add_argument("--sc-ratio", type=float, default=0.03, help="Пропорция вокала SoundCloud по отношению к VCTK (0.0 - 1.0).")
    parser.add_argument("--patience", type=int, default=7, help="Количество эпох ранней остановки (patience).")
    parser.add_argument("--num-workers", type=int, default=2, help="Количество воркеров для DataLoader.")
    
    args = parser.parse_args()
    
    project_dir = Path(__file__).parent.parent.absolute()
    print("=== ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА MIXING-AI ===")
    print(f"Рабочая директория: {project_dir}")
    
    # Шаг 1: Сборка датасета
    print("\n--- ЭТАП 1: ГЕНЕРАЦИЯ ДАТАСЕТА ---")
    dataset_script = str(project_dir / "scripts" / "prepare_dataset.py")
    dataset_cmd = [
        sys.executable, dataset_script,
        "--max-samples", str(args.max_samples),
        "--sc-ratio", str(args.sc_ratio)
    ]
    run_cmd(dataset_cmd)
    
    # Шаг 2: Обучение модели
    print("\n--- ЭТАП 2: ОБУЧЕНИЕ МОДЕЛИ MIXING-AI ---")
    train_script = str(project_dir / "scripts" / "train_ml.py")
    train_cmd = [
        sys.executable, train_script,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--patience", str(args.patience),
        "--num-workers", str(args.num_workers)
    ]
    run_cmd(train_cmd)
    
    print("\n[+] Пайплайн успешно завершен!")
    print("[*] Обученная модель сохранена в: data/models/best_model.pth")

if __name__ == "__main__":
    main()
