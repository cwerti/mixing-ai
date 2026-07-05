import struct
from pathlib import Path

def parse_fl_events(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
    
    pos = data.find(b'FLdt')
    if pos == -1: return []
    
    dt_len = struct.unpack('<I', data[pos+4:pos+8])[0]
    pos += 8
    end_pos = pos + dt_len
    
    print(f"Start pos: {pos}, End pos: {end_pos}, Data len: {len(data)}")
    
    events = []
    while pos < end_pos:
        eid = data[pos]
        start_event_pos = pos
        pos += 1
        
        if eid < 128:
            if pos >= len(data): break
            val = data[pos]
            pos += 1
            events.append((eid, val))
        elif eid < 192:
            if pos + 2 > len(data): break
            val = struct.unpack('<H', data[pos:pos+2])[0]
            pos += 2
            events.append((eid, val))
        elif eid < 240:
            if pos + 4 > len(data): break
            val = struct.unpack('<I', data[pos:pos+4])[0]
            pos += 4
            events.append((eid, val))
        else:
            length = 0
            shift = 0
            while pos < len(data):
                b = data[pos]
                pos += 1
                length |= (b & 0x7F) << shift
                if not (b & 0x80):
                    break
                shift += 7
            if pos + length > len(data): break
            val = data[pos:pos+length]
            pos += length
            events.append((eid, val))
        
        # print(f"Parsed ID {eid} at {start_event_pos}, next pos {pos}")
            
    return events

if __name__ == "__main__":
    evs = parse_fl_events("FlProject/template.fst")
    # Параметры плагина в .fst обычно передаются через ID 241 (Plugin State)
    for eid, val in evs:
        if eid == 241:
            print(f"Plugin State Data found (Length: {len(val)})")
            # Печатаем кусок данных состояния плагина
            print(val.hex()[:100])
        else:
            print(f"ID: {eid:3} | Value: {val}")
