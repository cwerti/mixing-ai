import struct
from pathlib import Path

def parse_fl_events(file_path: str | Path) -> list:
    """
    Парсит поток событий формата Image-Line (FL Studio Event stream) из бинарного файла (.flp / .fst).
    Позволяет извлечь параметры автоматизации и состояния загруженных VST-плагинов.
    
    Args:
        file_path: Путь к файлу пресета (.fst) или проекта (.flp).
        
    Returns:
        list: Список кортежей (event_id, value), содержащих события FL.
    """
    with open(file_path, 'rb') as f:
        data = f.read()
    
    # Ищем сигнатурный маркер потока событий FLdt
    pos = data.find(b'FLdt')
    if pos == -1: 
        return []
    
    dt_len = struct.unpack('<I', data[pos+4:pos+8])[0]
    pos += 8
    end_pos = pos + dt_len
    
    events = []
    while pos < end_pos:
        eid = data[pos]
        pos += 1
        
        # Разбор событий разного размера в зависимости от event_id (eid)
        if eid < 128:
            # 1-байтовые данные (события от 0 до 127)
            if pos >= len(data): 
                break
            val = data[pos]
            pos += 1
            events.append((eid, val))
        elif eid < 192:
            # 2-байтовые данные (события от 128 до 191)
            if pos + 2 > len(data): 
                break
            val = struct.unpack('<H', data[pos:pos+2])[0]
            pos += 2
            events.append((eid, val))
        elif eid < 240:
            # 4-байтовые данные (события от 192 до 239)
            if pos + 4 > len(data): 
                break
            val = struct.unpack('<I', data[pos:pos+4])[0]
            pos += 4
            events.append((eid, val))
        else:
            # Текстовые или бинарные данные переменной длины (события от 240 до 255)
            length = 0
            shift = 0
            while pos < len(data):
                b = data[pos]
                pos += 1
                length |= (b & 0x7F) << shift
                if not (b & 0x80):
                    break
                shift += 7
            if pos + length > len(data): 
                break
            val = data[pos:pos+length]
            pos += length
            events.append((eid, val))
            
    return events

if __name__ == "__main__":
    # Локальный тест на файле шаблона
    template_file = "FlProject/template.fst"
    if Path(template_file).exists():
        evs = parse_fl_events(template_file)
        # Параметры состояния плагина в .fst обычно передаются через ID 241 (Plugin State)
        for eid, val in evs:
            if eid == 241:
                print(f"Обнаружен блок Plugin State (Размер: {len(val)} байт)")
                print(val.hex()[:100] + "...")
            else:
                print(f"ID события: {eid:3} | Значение: {val}")
    else:
        print(f"Шаблон {template_file} не найден для локального теста.")
