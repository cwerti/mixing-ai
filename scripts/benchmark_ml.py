import time
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.ml.dataset import MixingAIDataset
from app.ml.model import MixingAIModel

def benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Используемое устройство: {device}")
    
    # 1. Тест скорости загрузки данных
    dataset = MixingAIDataset("data/processed/dataset_v3")
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    
    print("[*] Тестирование DataLoader (только чтение с диска)...")
    start = time.time()
    count = 0
    for batch_x, batch_chain, batch_params, batch_mask in loader:
        count += 1
        if count >= 10:  # Измеряем 10 батчей
            break
    duration = time.time() - start
    print(f"[+] Загружено 10 батчей (батч-сайз 64) за {duration:.2f} сек ({duration/10:.3f} сек/батч)")
    
    # 2. Тест скорости вычислений модели на GPU
    print("[*] Тестирование модели (прямой и обратный проходы на GPU)...")
    model = MixingAIModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Генерируем фейковый батч на GPU (размер 16)
    dummy_x = torch.randn(16, 3, 128, 700).to(device)
    dummy_chain = torch.randint(0, 2, (16, 7)).float().to(device)
    dummy_params = torch.rand(16, 18).to(device)
    dummy_mask = torch.ones(16, 18).to(device)
    
    # Разогрев GPU
    for _ in range(5):
        logits, pred_params = model(dummy_x, dummy_chain)
        diff = (pred_params - dummy_params) ** 2
        loss = diff.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    start = time.time()
    steps = 20
    for _ in range(steps):
        logits, pred_params = model(dummy_x, dummy_chain)
        diff = (pred_params - dummy_params) ** 2
        loss = diff.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    duration = time.time() - start
    print(f"[+] Модель выполнила {steps} шагов на GPU за {duration:.2f} сек ({duration/steps*1000:.1f} мс/шаг)")

if __name__ == "__main__":
    benchmark()
