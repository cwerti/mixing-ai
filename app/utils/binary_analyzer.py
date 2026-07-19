import struct
from pathlib import Path

def analyze_fst(file_path: str | Path) -> bytes:
    """
    Анализирует бинарную структуру .fst файла пресета,
    находя смещения дефолтных значений float32 (например, 0.5).
    Помогает реверс-инжинирить бинарные пресеты.
    
    Args:
        file_path: Путь к бинарному файлу пресета.
        
    Returns:
        bytes: Сырое бинарное содержимое файла.
    """
    with open(file_path, 'rb') as f:
        data = f.read()
    
    print(f"Размер файла: {len(data)} байт")
    
    # Ищем дефолтное значение 0.5 (стандартное положение многих ручек в FL Studio)
    # В формате float32 (Little Endian) число 0.5 кодируется как 0x3F000000 (байты: 00 00 00 3F)
    target = struct.pack('<f', 0.5)
    offsets = []
    for i in range(len(data) - 3):
        if data[i:i+4] == target:
            offsets.append(i)
    
    print(f"Значение 0.5 (float32) найдено на смещениях: {offsets}")
    return data

if __name__ == "__main__":
    analyze_fst("FlProject/template.fst")
