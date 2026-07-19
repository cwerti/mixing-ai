import sys
from pathlib import Path

def compare_files(file1: str | Path, file2: str | Path):
    """
    Побайтово сравнивает два файла пресетов (.fst) или проектов (.flp)
    и выводит смещения и значения отличающихся байт.
    Помогает отлаживать бинарный формат FL Studio.
    
    Args:
        file1: Путь к первому файлу.
        file2: Путь ко второму файлу.
    """
    try:
        with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
            data1 = f1.read()
            data2 = f2.read()
            
        print(f"Файл 1 ({file1}) размер: {len(data1)} байт")
        print(f"Файл 2 ({file2}) размер: {len(data2)} байт")
        
        min_len = min(len(data1), len(data2))
        
        diffs = []
        for i in range(min_len):
            if data1[i] != data2[i]:
                diffs.append((i, data1[i], data2[i]))
                
        print(f"\nОбнаружено побайтовых различий: {len(diffs)}")
        for offset, b1, b2 in diffs[:20]: # Показываем только первые 20 различий
            print(f"Смещение: {offset:4} (0x{offset:04x}) | Файл 1: 0x{b1:02x} ({b1:3}) | Файл 2: 0x{b2:02x} ({b2:3})")
            
        if len(data1) != len(data2):
            print("\nРазмеры файлов отличаются!")
            
    except Exception as e:
        print(f"Ошибка при сравнении файлов: {e}")

if __name__ == "__main__":
    compare_files("FlProject/template.fst", "FlProject/untitled.fst")
