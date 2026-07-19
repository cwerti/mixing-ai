import sys
import os
import shutil
import time
import mido
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# Импорт Windows-зависимых библиотек для управления GUI FL Studio
if sys.platform == "win32":
    try:
        import win32gui
        import win32con
        import win32api
        import win32com.client
        HAS_WIN32 = True
    except ImportError:
        HAS_WIN32 = False
else:
    HAS_WIN32 = False

class BridgeClient:
    """
    Универсальный клиент для управления FL Studio.
    Обеспечивает фокусы окон, навигацию по микшеру, отправку SysEx-сообщений
    и управление параметрами плагинов через виртуальный MIDI-порт.
    """
    
    def __init__(self, port_name: str = 'Mixing Ai', max_retries: int = 3, retry_delay: float = 0.5):
        """
        Инициализирует мост с FL Studio.
        
        Args:
            port_name: Имя виртуального MIDI-порта.
            max_retries: Количество попыток подключения к порту.
            retry_delay: Задержка между попытками подключения.
        """
        self.port_name = port_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.midi_out: Optional[mido.ports.BaseOutput] = None
        self.shell: Any = None
        
        if HAS_WIN32:
            try:
                self.shell = win32com.client.Dispatch("WScript.Shell")
            except Exception as e:
                print(f"[!] Предупреждение: Не удалось инициализировать WScript.Shell ({e})")
            
            # Автоматическая установка или обновление MIDI-скрипта FL Studio
            self.install_bridge()

    def install_bridge(self) -> bool:
        """
        Автоматически находит папку настроек FL Studio в Документах (включая OneDrive)
        и копирует туда MIDI-скрипт устройства.
        
        Returns:
            bool: True в случае успешной установки/обновления, иначе False.
        """
        try:
            import ctypes
            from ctypes import wintypes
            
            CSIDL_PERSONAL = 5
            buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
            ctypes.windll.shell32.SHGetFolderPathW(None, CSIDL_PERSONAL, None, 0, buf)
            docs_path = Path(buf.value)
            
            fl_hardware_path = docs_path / "Image-Line" / "FL Studio" / "Settings" / "Hardware"
            if not fl_hardware_path.exists():
                home = Path(os.path.expanduser("~"))
                alt_paths = [
                    home / "OneDrive" / "Документы" / "Image-Line" / "FL Studio" / "Settings" / "Hardware",
                    home / "OneDrive" / "Documents" / "Image-Line" / "FL Studio" / "Settings" / "Hardware",
                ]
                for p in alt_paths:
                    if p.exists():
                        fl_hardware_path = p
                        break
            
            if not fl_hardware_path.exists():
                print("[-] Папка настроек оборудования FL Studio не найдена. Требуется ручная установка скрипта.")
                return False
                
            target_dir = fl_hardware_path / "MixingAI_Bridge"
            target_dir.mkdir(parents=True, exist_ok=True)
            
            source_script = Path("MixingAI_Bridge/device_MixingAI.py")
            if source_script.exists():
                target_file = target_dir / "device_MixingAI.py"
                shutil.copy2(source_script, target_file)
                print(f"[+] MIDI-скрипт успешно установлен/обновлен в: {target_file}")
                return True
            else:
                print("[-] Исходный файл скрипта MixingAI_Bridge/device_MixingAI.py не найден в репозитории.")
                return False
        except Exception as e:
            print(f"[-] Ошибка при установке MIDI-скрипта: {e}")
            return False

    def connect_midi(self) -> bool:
        """
        Устанавливает соединение с виртуальным MIDI-портом вывода (с попытками повтора).
        
        Returns:
            bool: True, если порт успешно открыт, иначе False.
        """
        if self.midi_out:
            return True
            
        for attempt in range(self.max_retries):
            try:
                outputs = mido.get_output_names()
                target = next((o for o in outputs if self.port_name.lower() in o.lower()), None)
                if target:
                    self.midi_out = mido.open_output(target)
                    print(f"[+] MIDI порт подключен: {target}")
                    print("[!] ВАЖНО: В настройках MIDI в FL Studio проверьте, что:")
                    print("    1. Входной порт 'Mixing-AI' включен (Enable).")
                    print("    2. Тип контроллера (Controller type) выбран как 'Mixing-AI Super Bridge'.")
                    return True
            except Exception as e:
                print(f"[-] Попытка подключения к MIDI {attempt + 1} завершилась неудачей: {e}")
            time.sleep(self.retry_delay)
            
        print(f"[-] Не удалось подключиться к MIDI-порту '{self.port_name}' после {self.max_retries} попыток.")
        return False

    def close(self):
        """Закрывает MIDI-соединение."""
        if self.midi_out:
            try:
                self.midi_out.close()
                print("[*] MIDI-соединение закрыто.")
            except Exception as e:
                print(f"[-] Ошибка при закрытии MIDI-порта: {e}")
            self.midi_out = None

    def focus_fl(self) -> bool:
        """
        Находит главное окно FL Studio среди запущенных процессов Windows
        и выводит его на передний план.
        
        Returns:
            bool: True в случае успешной фокусировки, иначе False.
        """
        if not HAS_WIN32:
            print("[!] Библиотеки win32 недоступны. Невозможно сфокусировать FL Studio.")
            return False

        def callback(hwnd, hwnds):
            if win32gui.IsWindowVisible(hwnd) and "FL Studio" in win32gui.GetWindowText(hwnd):
                hwnds.append(hwnd)
                
        hwnds = []
        try:
            win32gui.EnumWindows(callback, hwnds)
        except Exception as e:
            print(f"[-] Ошибка поиска окон: {e}")
            return False
            
        if hwnds:
            hwnd = hwnds[0]
            try:
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnd)
                return True
            except Exception:
                try:
                    win32api.keybd_event(win32con.VK_MENU, 0, 0, 0)
                    win32gui.SetForegroundWindow(hwnd)
                    win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)
                    return True
                except Exception as e:
                    print(f"[-] Не удалось вывести FL Studio на передний план: {e}")
                    return False
        print("[-] Окно FL Studio не найдено.")
        return False

    def send_sysex(self, command: str) -> bool:
        """
        Отправляет команду через SysEx-сообщение с заголовком MAI в FL Studio.
        
        Args:
            command: Строковая команда (например, 'SELECT_EMPTY').
            
        Returns:
            bool: True, если сообщение успешно отправлено, иначе False.
        """
        if not self.connect_midi():
            return False
            
        try:
            header = [0x00, 0x4D, 0x41, 0x49] # Уникальный заголовок MAI
            msg = mido.Message('sysex', data=header + list(command.encode('utf-8')))
            self.midi_out.send(msg)
            return True
        except Exception as e:
            print(f"[-] Ошибка отправки SysEx: {e}")
            return False

    def press_key(self, vk_code: int, delay: float = 0.1):
        """
        Симулирует нажатие клавиши клавиатуры (только для Windows).
        
        Args:
            vk_code: Виртуальный код клавиши (например, win32con.VK_F9).
            delay: Задержка в секундах до отпускания клавиши.
        """
        if not HAS_WIN32:
            print("[!] win32 недоступен. Эмуляция нажатия клавиш невозможна.")
            return
        try:
            win32api.keybd_event(vk_code, 0, 0, 0)
            time.sleep(delay)
            win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)
        except Exception as e:
            print(f"[-] Ошибка эмуляции нажатия клавиши: {e}")

    def paste_text(self, text: str):
        """
        Копирует текст в буфер обмена Windows и симулирует Ctrl+V для вставки.
        Не зависит от раскладки клавиатуры.
        
        Args:
            text: Вставляемый текст.
        """
        if not HAS_WIN32:
            print("[!] win32 недоступен. Эмуляция вставки текста невозможна.")
            return
        import pyperclip
        try:
            old_clip = pyperclip.paste()
        except Exception:
            old_clip = ""

        try:
            pyperclip.copy(text)
            time.sleep(0.1)
            # Зажатие Ctrl (0x11)
            win32api.keybd_event(0x11, 0, 0, 0)
            time.sleep(0.05)
            # Нажатие V (0x56)
            win32api.keybd_event(0x56, 0, 0, 0)
            time.sleep(0.05)
            # Отпускание V
            win32api.keybd_event(0x56, 0, win32con.KEYEVENTF_KEYUP, 0)
            # Отпускание Ctrl
            win32api.keybd_event(0x11, 0, win32con.KEYEVENTF_KEYUP, 0)
            time.sleep(0.1)
        except Exception as e:
            print(f"[-] Ошибка вставки текста: {e}")
        finally:
            try:
                pyperclip.copy(old_clip)
            except Exception:
                pass

    def prepare_empty_track(self) -> bool:
        """
        Фокусирует микшер FL Studio и выбирает первый свободный микшерный канал.
        
        Returns:
            bool: True в случае успеха, иначе False.
        """
        print("[*] Подготовка микшера и выбор свободного трека...")
        if not self.focus_fl():
            return False
        time.sleep(0.5)
        
        self.press_key(win32con.VK_F9)
        time.sleep(0.5)
        
        if self.send_sysex("SELECT_EMPTY"):
            time.sleep(1.0)
            return True
        return False

    def target_plugin(self, plugin_name: str) -> bool:
        """
        Отправляет SysEx команду для выбора или загрузки плагина в FL Studio.
        
        Args:
            plugin_name: Название целевого плагина (например, 'Fruity Parametric EQ 2').
            
        Returns:
            bool: True в случае успеха, иначе False.
        """
        print(f"[*] Выбор плагина: '{plugin_name}'...")
        return self.send_sysex(f"TARGET_PLUGIN|{plugin_name}")

    def set_param(self, slot_index: int, param_index: int, value: float) -> bool:
        """
        Управляет ручкой/параметром плагина с помощью посыла MIDI Control Change (CC).
        
        Args:
            slot_index: Номер слота эффекта (от 0 до 9) в канале микшера или -1 для текущего активного.
            param_index: Индекс параметра плагина (MIDI CC контроллер).
            value: Значение параметра от 0.0 до 1.0.
            
        Returns:
            bool: True в случае успеха, иначе False.
        """
        if not self.connect_midi():
            return False
            
        val_clamped = np.clip(value, 0.0, 1.0)
        val_midi = int(val_clamped * 127)
        
        try:
            if slot_index == -1:
                channel = 15 # Канал 16 зарезервирован под фокус
            else:
                channel = int(np.clip(slot_index, 0, 9))
                
            msg = mido.Message('control_change', channel=channel, control=param_index, value=val_midi)
            self.midi_out.send(msg)
            return True
        except Exception as e:
            print(f"[-] Ошибка отправки MIDI CC параметра: {e}")
            return False

    def set_eq_params(self, freqs: np.ndarray, delta: np.ndarray, plugin_name: str = "Fruity Parametric EQ 2") -> bool:
        """
        Настраивает полосы Fruity Parametric EQ 2 по кривой разницы АЧХ.
        
        Args:
            freqs: Частотная ось.
            delta: Дельта АЧХ в dB.
            plugin_name: Название плагина эквалайзера.
            
        Returns:
            bool: True в случае успеха, иначе False.
        """
        if not self.connect_midi():
            return False
            
        if not self.target_plugin(plugin_name):
            return False
        time.sleep(0.2)
        
        target_freqs = [100, 250, 500, 1000, 3000, 7000, 12000]
        
        try:
            for i, target_f in enumerate(target_freqs):
                idx = (np.abs(freqs - target_f)).argmin()
                gain = np.clip(delta[idx], -18.0, 18.0)
                
                f_val = (np.log10(target_f) - 1.0) / 3.301
                g_val = (gain + 18.0) / 36.0
                
                self.set_param(slot_index=-1, param_index=i*5, value=f_val)
                time.sleep(0.01)
                self.set_param(slot_index=-1, param_index=i*5+1, value=g_val)
                time.sleep(0.01)
            return True
        except Exception as e:
            print(f"[-] Ошибка отправки параметров EQ: {e}")
            return False
