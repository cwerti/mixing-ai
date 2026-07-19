import json
import argparse
import sys
from pathlib import Path
from collections import Counter

def inspect_dataset(dataset_dir: str | Path):
    """
    Анализирует сгенерированный датасет и выводит сводную статистику
    по классам плагинов, распределению параметров и метаданным.
    
    Args:
        dataset_dir: Путь к папке датасета (где лежит index.jsonl).
    """
    dataset_path = Path(dataset_dir)
    index_file = dataset_path / "index.jsonl"
    stats_file = dataset_path / "stats.json"
    
    if not index_file.exists():
        print(f"[-] Файл индекса не найден по пути: {index_file}")
        print("[*] Пожалуйста, сначала сгенерируйте датасет с помощью prepare_dataset.py.")
        return

    print(f"=== АНАЛИЗ ДАТАСЕТА: {dataset_path.name} ===")
    
    # 1. Чтение общей статистики из stats.json
    if stats_file.exists():
        with open(stats_file, "r", encoding="utf-8") as f:
            stats = json.load(f)
            print(f"Частота дискретизации (SR): {stats.get('sr')} Гц")
            print(f"Порядок цепочки эффектов: {stats.get('chain_order')}")
    
    # 2. Чтение index.jsonl
    samples = []
    with open(index_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                
    total_samples = len(samples)
    print(f"Всего сэмплов в датасете: {total_samples}")
    
    if total_samples == 0:
        return
        
    # 3. Подсчет распределения плагинов
    plugin_counts = Counter()
    sources = Counter()
    durations = []
    
    for sample in samples:
        sources[sample["source"]] += 1
        durations.append(sample["duration_sec"])
        for plugin in sample["chain"]:
            plugin_counts[plugin] += 1
            
    print("\n--- Распределение по источникам аудио ---")
    for src, count in sources.items():
        pct = (count / total_samples) * 100
        print(f"  {src}: {count} сэмплов ({pct:.1f}%)")
        
    print("\n--- Распределение активных плагинов в цепочках ---")
    for plugin, count in plugin_counts.items():
        pct = (count / total_samples) * 100
        print(f"  {plugin}: {count} раз(а) ({pct:.1f}%)")
        
    print(f"\nСредняя длительность сэмпла: {sum(durations)/len(durations):.2f} сек.")
    
    # 4. Вывод примера метаданных первого сэмпла
    print("\n--- Пример метаданных сэмпла (ID: 000000) ---")
    sample_zero = samples[0]
    print(json.dumps(sample_zero, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скрипт инспекции сгенерированного датасета")
    parser.add_argument("--dir", type=str, default="data/processed/dataset_v1", help="Путь к директории датасета")
    args = parser.parse_args()
    
    inspect_dataset(args.dir)
