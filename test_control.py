import mido
import time

PORT_NAME = 'Mixing Ai' 

def build_vocal_chain():
    try:
        outputs = mido.get_output_names()
        target_port = next((o for o in outputs if PORT_NAME.lower() in o.lower()), None)
        out_port = mido.open_output(target_port)
        print(f"[+] Mixing-AI Bridge Connected")

        # 1. Загружаем цепочку через SysEx
        header = [0x00, 0x4D, 0x41, 0x49]
        
        print("[*] Building Vocal Chain...")
        
        # Слот 1: EQ
        out_port.send(mido.Message('sysex', data=header + list(b"LOAD|1|0|Fruity Parametric EQ 2")))
        time.sleep(1)
        
        # Слот 2: Limiter (Компрессия)
        out_port.send(mido.Message('sysex', data=header + list(b"LOAD|1|1|Fruity Limiter")))
        time.sleep(1)

        print("[*] Applying AI Settings to EQ...")
        # Установим 3 характерные точки для теста
        settings = [
            (0, 0, 40),  # Слот 1, Парам 0 (Freq 1), Value 40
            (0, 1, 80),  # Слот 1, Парам 1 (Gain 1), Value 80
            (0, 5, 64),  # Слот 1, Парам 5 (Freq 2), Value 64
        ]
        
        for slot, param, val in settings:
            # Используем канал MIDI для обозначения слота (0 = Slot 1)
            msg = mido.Message('control_change', channel=slot, control=param, value=val)
            out_port.send(msg)
            time.sleep(0.1)

        print("[+] Vocal Chain is READY!")
        out_port.close()

    except Exception as e:
        print(f"[-] Error: {e}")

if __name__ == "__main__":
    build_vocal_chain()
