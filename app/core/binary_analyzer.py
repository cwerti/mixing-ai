import struct

def analyze_fst(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
    
    print(f"File size: {len(data)} bytes")
    
    # Ищем значения 0.5 (это стандартный Gain и частоты по умолчанию)
    # В float32 (Little Endian) 0.5 это 00 00 00 3F
    target = struct.pack('<f', 0.5)
    offsets = []
    for i in range(len(data) - 3):
        if data[i:i+4] == target:
            offsets.append(i)
    
    print(f"Found 0.5 (float32) at offsets: {offsets}")
    
    # В EQ2 обычно идет блок параметров для каждой полосы. 
    # Попробуем найти последовательность
    return data

if __name__ == "__main__":
    analyze_fst("FlProject/template.fst")
