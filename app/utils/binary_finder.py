import struct
from pathlib import Path

def find_eq2_params(file_path: str | Path) -> int:
    """
    Ищет сигнатурные смещения параметров Fruity Parametric EQ 2 внутри .fst файла
    на основе известных значений по умолчанию (например, 0.5 для средних частот
    или 0.05 для низкочастотной полосы).
    
    Args:
        file_path: Путь к файлу пресета эквалайзера.
        
    Returns:
        int: Смещение первого найденного параметра или -1, если совпадения не найдены.
    """
    with open(file_path, 'rb') as f:
        data = f.read()

    # Ищем значение 0.5 (значение Freq для Band 4 по умолчанию)
    target = struct.pack('<f', 0.5)
    pos = data.find(target)
    
    if pos != -1:
        print(f"[FOUND] Значение 0.5 найдено на смещении: {pos}")
        # Параметры в пресете EQ2 идут блоками по 4 байта (тип float32)
        return pos
    else:
        print("[NOT FOUND] Значение 0.5 (float32) не найдено.")
        # Пробуем альтернативный поиск: 0.05 (Band 1 Freq по умолчанию)
        target = struct.pack('<f', 0.05)
        pos = data.find(target)
        if pos != -1:
            print(f"[FOUND] Значение 0.05 найдено на смещении: {pos}")
            return pos
    return -1

if __name__ == "__main__":
    find_eq2_params("FlProject/template.fst")
