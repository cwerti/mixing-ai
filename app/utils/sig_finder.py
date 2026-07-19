from pathlib import Path

def find_sig():
    """
    Ищет в файле пресета шаблон сигнатур байт эквалайзера EQ2
    для локализации блока параметров фильтрации.
    """
    template_file = 'FlProject/template.fst'
    if not Path(template_file).exists():
        print(f"[-] Файл шаблона {template_file} не найден.")
        return
        
    data = open(template_file, 'rb').read()
    
    # Сигнатурный паттерн байт EQ 2
    sig = b'\xab\x2a\x00\x00\x1c\x47\x00\x00\x8e\x63\x00\x00\x00\x80\x00\x00\x72\x9c\x00\x00\xe4\xb8\x00\x00\x55\xd5\x00\x00'
    offsets = []
    idx = data.find(sig)
    while idx != -1:
        offsets.append(idx)
        idx = data.find(sig, idx + 1)
        
    print('Сигнатура обнаружена на смещениях:', offsets)

if __name__ == '__main__':
    find_sig()
