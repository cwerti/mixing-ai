import rtmidi
import time
import sys

def main():
    midiout = rtmidi.MidiOut()
    
    # На Windows open_virtual_port часто не работает без драйвера, 
    # поэтому мы используем другой хак: открываем порт и ждем подключения.
    port_name = "Mixing-AI Virtual Port"
    
    try:
        # Пытаемся создать виртуальный порт (нативно)
        midiout.open_virtual_port(port_name)
        print(f"[+] УСПЕХ: Виртуальный порт '{port_name}' создан!")
    except:
        print("[-] Твоя версия Windows блокирует создание виртуальных портов без драйвера.")
        print("[!] План Б: Сейчас я попробую пробиться через Win32 Messaging.")
        return

    print("[*] Сервер активен. ПЕРЕЙДИ В FL STUDIO.")
    print("[*] В списке INPUT должен появиться этот порт. Включи его (Enable).")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        midiout.close_port()
        print("\n[*] Порт закрыт.")

if __name__ == "__main__":
    main()
