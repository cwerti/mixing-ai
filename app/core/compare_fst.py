import sys

def compare_files(file1, file2):
    try:
        with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
            data1 = f1.read()
            data2 = f2.read()
            
        print(f"File 1 ({file1}) size: {len(data1)} bytes")
        print(f"File 2 ({file2}) size: {len(data2)} bytes")
        
        min_len = min(len(data1), len(data2))
        
        diffs = []
        for i in range(min_len):
            if data1[i] != data2[i]:
                diffs.append((i, data1[i], data2[i]))
                
        print(f"\nFound {len(diffs)} byte differences:")
        for offset, b1, b2 in diffs[:20]: # show first 20 diffs
            print(f"Offset: {offset:4} (0x{offset:04x}) | File 1: 0x{b1:02x} ({b1:3}) | File 2: 0x{b2:02x} ({b2:3})")
            
        if len(data1) != len(data2):
            print("\nFiles have different lengths!")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    compare_files("FlProject/template.fst", "FlProject/untitled.fst")
