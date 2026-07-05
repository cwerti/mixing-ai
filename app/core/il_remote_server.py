import socket
import struct
import time
import threading

class ILRemoteServer:
    def __init__(self):
        self.ip = "0.0.0.0"
        self.tcp_port = 9000
        self.udp_port = 9100
        self.running = True

    def start_tcp_handshake(self):
        """FL Studio сначала подключается по TCP, чтобы опознать устройство."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((self.ip, self.tcp_port))
        server.listen(5)
        print(f"[*] IL Remote TCP Server on {self.tcp_port}")
        
        while self.running:
            try:
                conn, addr = server.accept()
                # FL ожидает приветствие в формате JSON или спец. байтов
                # Для начала просто держим соединение открытым
                print(f"[+] FL Studio connected from {addr}")
                # Мы 'прикидываемся' контроллером 'Mixing-AI'
                conn.send(b'{"name":"Mixing-AI", "type":"iPad", "version":1}\n')
            except:
                break

    def send_midi(self, status, data1, data2):
        """Отправка MIDI через UDP пакеты с заголовком ILR."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Магический заголовок Image-Line Remote
        header = b'ILR\x00\x01' 
        packet = header + struct.pack('BBB', status, data1, data2)
        sock.sendto(packet, ("127.0.0.1", self.udp_port))

    def apply_eq_bands(self, bands):
        print(f"[*] Sending {len(bands)} bands to FL Studio...")
        # 1. Инициализация (CC 0)
        self.send_midi(0xBF, 0, 127)
        time.sleep(0.1)
        
        for i, (f, g) in enumerate(bands):
            import numpy as np
            f_p = (np.log10(f) - np.log10(10)) / (np.log10(20000) - np.log10(10))
            g_p = (g + 18) / 36
            
            f_midi = int(np.clip(f_p * 127, 0, 127))
            g_midi = int(np.clip(g_p * 127, 0, 127))
            
            # CC 1-7: Freq, CC 8-14: Gain (Channel 16)
            self.send_midi(0xBF, i+1, f_midi)
            self.send_midi(0xBF, i+8, g_midi)
            time.sleep(0.01)
        print("[+] Done!")

if __name__ == "__main__":
    # Тестовый запуск сервера
    srv = ILRemoteServer()
    threading.Thread(target=srv.start_tcp_handshake, daemon=True).start()
    while True: time.sleep(1)
