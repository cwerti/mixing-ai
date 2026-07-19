import socket
import struct
import time
import threading

class ILRemoteServer:
    """
    Эмулятор сервера Image-Line Remote (IL Remote).
    Позволяет общаться с FL Studio по сети (TCP/UDP) без использования физических 
    или виртуальных MIDI-кабелей, притворяясь мобильным приложением-контроллером.
    """
    
    def __init__(self):
        self.ip = "0.0.0.0"
        self.tcp_port = 9000
        self.udp_port = 9100
        self.running = True

    def start_tcp_handshake(self):
        """
        Запускает TCP-сервер для обработки рукопожатия с FL Studio.
        FL Studio сначала подключается по TCP, чтобы идентифицировать устройство.
        """
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((self.ip, self.tcp_port))
        server.listen(5)
        print(f"[*] Сервер IL Remote TCP запущен на порту {self.tcp_port}")
        
        while self.running:
            try:
                conn, addr = server.accept()
                print(f"[+] FL Studio подключилась с адреса {addr}")
                # Отправляем приветствие в формате JSON: идентифицируем себя как Mixing-AI
                conn.send(b'{"name":"Mixing-AI", "type":"iPad", "version":1}\n')
            except:
                break

    def send_midi(self, status: int, data1: int, data2: int):
        """
        Отправляет MIDI-сообщение в FL Studio по UDP протоколу с заголовком ILR.
        
        Args:
            status: Байт статуса MIDI (например, 0xBF для Control Change).
            data1: Номер контроллера (CC) или клавиши.
            data2: Значение параметра (0-127).
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Магический сигнатурный заголовок Image-Line Remote
        header = b'ILR\x00\x01' 
        packet = header + struct.pack('BBB', status, data1, data2)
        sock.sendto(packet, ("127.0.0.1", self.udp_port))

    def apply_eq_bands(self, bands: list):
        """
        Отправляет настройки полос эквалайзера по сети в FL Studio.
        
        Args:
            bands: Список кортежей полос [(freq, gain), ...]
        """
        print(f"[*] Отправка {len(bands)} полос в FL Studio по сети...")
        # 1. Инициализация (посылаем сигнал пробуждения через CC 0)
        self.send_midi(0xBF, 0, 127)
        time.sleep(0.1)
        
        for i, (f, g) in enumerate(bands):
            import numpy as np
            f_p = (np.log10(f) - np.log10(10)) / (np.log10(20000) - np.log10(10))
            g_p = (g + 18) / 36
            
            f_midi = int(np.clip(f_p * 127, 0, 127))
            g_midi = int(np.clip(g_p * 127, 0, 127))
            
            # CC 1-7: Частота полосы, CC 8-14: Усиление полосы (16-й MIDI канал)
            self.send_midi(0xBF, i+1, f_midi)
            self.send_midi(0xBF, i+8, g_midi)
            time.sleep(0.01)
        print("[+] Настройки успешно отправлены по сети!")

if __name__ == "__main__":
    srv = ILRemoteServer()
    threading.Thread(target=srv.start_tcp_handshake, daemon=True).start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        srv.running = False
