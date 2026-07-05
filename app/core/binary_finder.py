import struct

def find_eq2_params(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()

    # Ищем 0.5 (Band 4 Freq по умолчанию)
    target = struct.pack('<f', 0.5)
    pos = data.find(target)
    
    if pos != -1:
        print(f"[FOUND] Value 0.5 found at offset: {pos}")
        # Если нашли 0.5, значит рядом должны быть и другие параметры.
        # В EQ2 параметры идут блоками по 4 байта (float32).
        return pos
    else:
        print("[NOT FOUND] Value 0.5 (float32) not found.")
        # Попробуем найти 0.05 (Band 1 Freq)
        target = struct.pack('<f', 0.05)
        pos = data.find(target)
        if pos != -1:
            print(f"[FOUND] Value 0.05 found at offset: {pos}")
            return pos
    return -1

if __name__ == "__main__":
    find_eq2_params("FlProject/template.fst")
