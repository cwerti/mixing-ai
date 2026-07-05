import os
import sys
import platform
import subprocess
import urllib.request
import ctypes
import time
import struct
from pathlib import Path
import numpy as np
import librosa
import mido

from app.core.audio_processor import AudioProcessor
from app.core.preset_generator import PresetGenerator
from app.core.virtual_midi import TeVirtualMIDI

# --- КОНФИГУРАЦИЯ ---
# Прямая ссылка на установщик драйвера (v1.3.0.43)
VIRTUAL_MIDI_URL = "https://www.tobias-erichsen.de/wp-content/uploads/2020/01/teVirtualMIDI_setup.exe"
# Системный путь к DLL после установки
DLL_PATH_64 = Path("C:/Windows/System32/teVirtualMIDI64.dll")

def install_driver_silent():
    """Скрытая установка системного драйвера."""
    if platform.system() != "Windows" or DLL_PATH_64.exists():
        return True

    print("[-] MIDI-драйвер не найден. Начинаю скрытую установку...")
    try:
        setup_exe = Path("teVirtualMIDI_setup.exe")
        # Скачивание напрямую exe
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        
        print(f"[*] Загрузка с {VIRTUAL_MIDI_URL}...")
        urllib.request.urlretrieve(VIRTUAL_MIDI_URL, setup_exe)
        
        print("[*] Установка... Это займет 5 секунд (потребуются права администратора).")
        # /VERYSILENT - флаг InnoSetup для полной невидимости
        subprocess.run([str(setup_exe.absolute()), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/SP-"], check=True)
        
        setup_exe.unlink(missing_ok=True)
        print("[+] Драйвер успешно интегрирован в систему.")
        return True
    except Exception as e:
        print(f"[-] Ошибка установки: {e}")
        return False

def install_fl_bridge():
    """Автоматическая установка моста в FL Studio."""
    try:
        docs = Path(os.path.expanduser("~/Documents"))
        fl_path = docs / "Image-Line/FL Studio/Settings/Hardware/MixingAI_Bridge"
        if not (docs / "Image-Line").exists():
            fl_path = Path(os.path.expanduser("~")) / "OneDrive/Documents/Image-Line/FL Studio/Settings/Hardware/MixingAI_Bridge"
        
        fl_path.mkdir(parents=True, exist_ok=True)
        with open(fl_path / "device_MixingAI.py", "w", encoding="utf-8") as f:
            f.write("""# name=Mixing-AI Super Bridge
import plugins
import mixer
def OnMidiMsg(event):
    if event.status == 191:
        cc, val, tr = event.data1, event.data2 / 127.0, mixer.trackNumber()
        slot = -1
        for s in range(10):
            if mixer.isTrackPluginValid(tr, s) and "EQ 2" in plugins.getPluginName(tr, s):
                slot = s; break
        if slot == -1:
            for s in range(10):
                if not mixer.isTrackPluginValid(tr, s):
                    mixer.loadEffect(s, "Fruity Parametric EQ 2", tr); slot = s; break
        if slot != -1:
            if 1 <= cc <= 7: plugins.setParamValue(val, (cc-1)*5, tr, slot)
            elif 8 <= cc <= 14: plugins.setParamValue(val, (cc-8)*5+1, tr, slot)
            event.handled = True
""")
        print("[+] Мост в FL Studio обновлен.")
    except Exception as e:
        print(f"[-] Ошибка установки моста: {e}")

def main():
    print("=== MIXING-AI PRO: FULL STEALTH AUTOMATION ===")
    
    # 1. Тихая подготовка системы
    if not install_driver_silent():
        print("[!] Запусти скрипт от имени Администратора для первой установки.")
        return

    # 2. Поднимаем невидимый порт
    v_midi = TeVirtualMIDI("Mixing-AI")
    if not v_midi.create_port():
        print("[-] Не удалось создать MIDI-порт. Перезапусти скрипт.")
        return
    
    print("[+] Программный порт 'Mixing-AI' активен.")
    install_fl_bridge()

    # 3. Анализ
    source_path = Path("data/raw/source/source.wav")
    ref_path = Path("data/raw/reference/reference.wav")

    if source_path.exists():
        print("[*] Анализ аудио...")
        processor = AudioProcessor()
        y_s = processor.load_audio(source_path)
        y_r = processor.load_audio(ref_path)
        delta = processor.get_spectral_envelope(y_r) - processor.get_spectral_envelope(y_s)
        bands = PresetGenerator().extract_key_bands(librosa.fft_frequencies(sr=44100, n_fft=2048), delta)

        # 4. Передача (через mido)
        time.sleep(1) # Даем FL Studio увидеть новый порт
        try:
            with mido.open_output("Mixing-AI") as outport:
                print("[*] Настройка FL Studio...")
                # Wake up
                outport.send(mido.Message('control_change', channel=15, control=0, value=127))
                time.sleep(0.5)
                for i, (f, g) in enumerate(bands):
                    f_p, g_p = (np.log10(f)-1)/3.301, (g+18)/36
                    outport.send(mido.Message('control_change', channel=15, control=i+1, value=int(np.clip(f_p*127, 0, 127))))
                    outport.send(mido.Message('control_change', channel=15, control=i+8, value=int(np.clip(g_p*127, 0, 127))))
                print("[SUCCESS] Эквалайзер настроен!")
        except Exception as e:
            print(f"[-] Ошибка отправки: {e}")
            print("[!] Один раз включи 'Mixing-AI' в настройках MIDI в FL Studio.")
    else:
        print("[-] Файлы не найдены.")

    time.sleep(1)
    v_midi.close()

if __name__ == "__main__":
    main()
