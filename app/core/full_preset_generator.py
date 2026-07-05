import pyflp
import numpy as np
import math
from pathlib import Path

class FullPresetGenerator:
    def __init__(self, template_path="FlProject/template.fst"):
        self.template_path = Path(template_path)
        self.MIN_FREQ = 10.0
        self.MAX_FREQ = 20000.0

    def freq_to_param(self, freq):
        freq = np.clip(freq, self.MIN_FREQ, self.MAX_FREQ)
        # Маппинг для Fruity Parametric EQ 2 (Логарифмическая шкала)
        return (math.log10(freq) - 1.0) / 3.301

    def gain_to_param(self, gain):
        # Маппинг для Fruity Parametric EQ 2 (-18dB to +18dB -> 0.0 to 1.0)
        gain = np.clip(gain, -18.0, 18.0)
        return (gain + 18.0) / 36.0

    def generate_vocal_chain(self, bands, output_name="mixing_ai_vocal_track.fst"):
        """
        Создает полный пресет микшерного канала на основе шаблона.
        bands: список кортежей [(freq, gain), ...]
        """
        if not self.template_path.exists():
            raise FileNotFoundError(f"Template not found at {self.template_path}")

        # 1. Загружаем шаблон канала (это должен быть .fst файл микшерного трека)
        # В PyFLP .fst файлы открываются так же, как и проекты
        preset = pyflp.parse(str(self.template_path))
        
        # Предполагаем, что в шаблоне в Слот 1 уже стоит EQ 2
        # В объекте пресета (если это Mixer State) мы ищем плагины
        try:
            # Находим первый плагин в слотах (обычно это EQ)
            # Примечание: Структура может меняться в зависимости от версии PyFLP
            # Для v2.x пресет микшера ведет себя как MixerTrack
            track = preset 
            
            # Настройка EQ 2 (Slot 1)
            eq_slot = track.slots[0]
            print(f"[*] Modifying plugin in Slot 1: {eq_slot.name}")

            for i, (freq, gain) in enumerate(bands):
                if i >= 7: break # EQ 2 имеет 7 полос
                
                f_param = self.freq_to_param(freq)
                g_param = self.gain_to_param(gain)
                
                # В EQ 2 параметры: 0-Freq, 1-Gain, 2-BW, 3-Type, 4-Order (для каждой полосы)
                # Полосы идут с шагом 5
                offset = i * 5
                eq_slot.parameters[offset].value = f_param
                eq_slot.parameters[offset + 1].value = g_param
                
            # Можно также настроить компрессор в Слот 2, если он есть в шаблоне
            if len(track.slots) > 1 and track.slots[1].name:
                print(f"[*] Adjusting compression in Slot 2: {track.slots[1].name}")
                # Здесь можно добавить логику для лимитера/компрессора
                pass

            # 3. Сохраняем результат
            output_path = Path("data/processed") / output_name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Сохраняем как новый .fst
            pyflp.save(track, str(output_path))
            print(f"[+] Success! Full channel preset saved to: {output_path}")
            return output_path

        except Exception as e:
            print(f"[-] Error generating preset: {e}")
            return None

if __name__ == "__main__":
    # Тестовые данные (7 полос)
    test_bands = [
        (100, -3), (250, 2), (500, -1), (1000, 0), 
        (3000, 3), (8000, -2), (12000, 1)
    ]
    gen = FullPresetGenerator()
    gen.generate_vocal_chain(test_bands)
