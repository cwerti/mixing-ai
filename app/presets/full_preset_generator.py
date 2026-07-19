import pyflp
# Совместимость с Python 3.12+ (исправление пустых Enum в PyFLP)
pyflp.EventEnum._member_names_ = ['DUMMY']
pyflp.EventEnum._member_map_ = {'DUMMY': 999}
import numpy as np
import math
import struct
from pathlib import Path

class FullPresetGenerator:
    """
    Класс для автоматической сборки бинарных файлов пресетов FL Studio (.fst) 
    на основе шаблона с использованием библиотеки PyFLP или прямого бинарного патчинга.
    """
    
    def __init__(self, template_path: str = "FlProject/template.fst"):
        """
        Инициализирует генератор полных пресетов.
        
        Args:
            template_path: Путь к файлу шаблона канала (.fst). По умолчанию FlProject/template.fst.
        """
        self.template_path = Path(template_path)
        self.MIN_FREQ = 10.0
        self.MAX_FREQ = 20000.0

    def freq_to_param(self, freq: float) -> float:
        """
        Маппинг физической частоты в значение параметра EQ 2.
        
        Args:
            freq: Частота в Гц.
            
        Returns:
            float: Нормализованное значение параметра.
        """
        freq = np.clip(freq, self.MIN_FREQ, self.MAX_FREQ)
        return (math.log10(freq) - 1.0) / 3.301

    def gain_to_param(self, gain: float) -> float:
        """
        Маппинг усиления в децибелах в значение параметра EQ 2.
        
        Args:
            gain: Усиление в dB.
            
        Returns:
            float: Нормализованное значение параметра.
        """
        gain = np.clip(gain, -18.0, 18.0)
        return (gain + 18.0) / 36.0

    def generate_vocal_chain(self, bands: list, output_name: str = "mixing_ai_vocal_track.fst") -> Path | None:
        """
        Создает новый файл пресета канала микшера (.fst), заполняя параметры
        эквалайзера на основе переданных полос и сохраняя результат в папку обработанных данных.
        
        Args:
            bands: Список полос [(freq, gain), ...].
            output_name: Имя выходного файла пресета.
            
        Returns:
            Path | None: Путь к сохраненному файлу пресета или None в случае ошибки.
        """
        if not self.template_path.exists():
            raise FileNotFoundError(f"Файл шаблона канала не найден по пути: {self.template_path}")

        try:
            # Попытка открыть как проект FLP/FST
            preset = pyflp.parse(str(self.template_path))
            
            # Если объект поддерживает slots (например, в FLP), используем высокоуровневый API
            if hasattr(preset, 'slots'):
                track = preset 
                eq_slot = track.slots[0]
                print(f"[*] Модификация плагина в Слоте 1 через PyFLP: {eq_slot.name}")

                for i, (freq, gain) in enumerate(bands):
                    if i >= 7: 
                        break
                    
                    f_param = self.freq_to_param(freq)
                    g_param = self.gain_to_param(gain)
                    
                    offset = i * 5
                    eq_slot.parameters[offset].value = f_param
                    eq_slot.parameters[offset + 1].value = g_param

                output_path = Path("data/processed") / output_name
                output_path.parent.mkdir(parents=True, exist_ok=True)
                pyflp.save(track, str(output_path))
                print(f"[+] Успешно сгенерирован пресет канала через PyFLP: {output_path}")
                return output_path
                
            else:
                # Откатываемся к надежному низкоуровневому бинарному патчу .fst
                raw_data = self.template_path.read_bytes()
                
                # Поиск сигнатурного паттерна частот в EQ2
                sig = b'\xab\x2a\x00\x00\x1c\x47\x00\x00\x8ec\x00\x00\x00\x80\x00\x00\x72\x9c\x00\x00\xe4\xb8\x00\x00\x55\xd5\x00\x00'
                pos = raw_data.find(sig)
                
                if pos != -1:
                    raw_arr = bytearray(raw_data)
                    for i, (freq, gain) in enumerate(bands):
                        if i >= 7: 
                            break
                        f_param = self.freq_to_param(freq)
                        # Переводим в 16-битный целочисленный параметр (0..65535)
                        f_val = int(np.clip(f_param * 65536, 0, 65535))
                        # Перезаписываем 4-байтовое значение частоты
                        struct.pack_into('<I', raw_arr, pos + i * 4, f_val)
                    
                    output_path = Path("data/processed") / output_name
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(raw_arr)
                    print(f"[+] Успешно сгенерирован пресет .fst через прямой бинарный патч: {output_path}")
                    return output_path
                else:
                    raise ValueError("Сигнатура параметров эквалайзера не найдена в бинарном шаблоне.")

        except Exception as e:
            print(f"[-] Ошибка при генерации пресета: {e}")
            return None

if __name__ == "__main__":
    test_bands = [
        (100, -3), (250, 2), (500, -1), (1000, 0), 
        (3000, 3), (8000, -2), (12000, 1)
    ]
    gen = FullPresetGenerator()
    try:
        gen.generate_vocal_chain(test_bands)
    except Exception as e:
        print(f"Ошибка теста: {e}")
