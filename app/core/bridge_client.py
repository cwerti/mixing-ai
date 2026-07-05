import sys
import os
import shutil
import time
import mido
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# Conditional Windows Imports
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
    """Unified client for FL Studio control, mixer navigation, and plugin parameters."""
    def __init__(self, port_name: str = 'Mixing Ai', max_retries: int = 3, retry_delay: float = 0.5):
        self.port_name = port_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.midi_out: Optional[mido.ports.BaseOutput] = None
        self.shell: Any = None
        
        if HAS_WIN32:
            try:
                self.shell = win32com.client.Dispatch("WScript.Shell")
            except Exception as e:
                print(f"[!] Warning: Could not initialize WScript.Shell ({e})")
            
            # Automatically install/update FL Studio MIDI script
            self.install_bridge()

    def install_bridge(self) -> bool:
        """Automatically installs/updates the MIDI script in FL Studio hardware settings."""
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
                print("[-] FL Studio Hardware settings directory not found. Please install the MIDI script manually.")
                return False
                
            target_dir = fl_hardware_path / "MixingAI_Bridge"
            target_dir.mkdir(parents=True, exist_ok=True)
            
            source_script = Path("MixingAI_Bridge/device_MixingAI.py")
            if source_script.exists():
                target_file = target_dir / "device_MixingAI.py"
                shutil.copy2(source_script, target_file)
                print(f"[+] MIDI Script successfully installed/updated to: {target_file}")
                return True
            else:
                print("[-] Source script MixingAI_Bridge/device_MixingAI.py not found in workspace.")
                return False
        except Exception as e:
            print(f"[-] Error installing MIDI script: {e}")
            return False

    def connect_midi(self) -> bool:
        """Connects to the virtual MIDI output port with retries."""
        if self.midi_out:
            return True
            
        for attempt in range(self.max_retries):
            try:
                outputs = mido.get_output_names()
                target = next((o for o in outputs if self.port_name.lower() in o.lower()), None)
                if target:
                    self.midi_out = mido.open_output(target)
                    print(f"[+] MIDI Connected: {target}")
                    print("[!] ВАЖНО: В настройках MIDI в FL Studio убедитесь, что:")
                    print("    1. Входной порт 'Mixing-AI' включен (Enable).")
                    print("    2. Для него выбран Controller type: 'Mixing-AI Super Bridge'.")
                    return True
            except Exception as e:
                print(f"[-] MIDI connection attempt {attempt + 1} failed: {e}")
            time.sleep(self.retry_delay)
            
        print(f"[-] Failed to connect to MIDI port '{self.port_name}' after {self.max_retries} attempts.")
        return False

    def close(self):
        """Closes the MIDI connection."""
        if self.midi_out:
            try:
                self.midi_out.close()
                print("[*] MIDI connection closed.")
            except Exception as e:
                print(f"[-] Error closing MIDI port: {e}")
            self.midi_out = None

    def focus_fl(self) -> bool:
        """Finds and brings the FL Studio window to the foreground."""
        if not HAS_WIN32:
            print("[!] win32 libraries are not available on this platform. Cannot focus FL Studio.")
            return False

        def callback(hwnd, hwnds):
            if win32gui.IsWindowVisible(hwnd) and "FL Studio" in win32gui.GetWindowText(hwnd):
                hwnds.append(hwnd)
                
        hwnds = []
        try:
            win32gui.EnumWindows(callback, hwnds)
        except Exception as e:
            print(f"[-] Error enumerating windows: {e}")
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
                    print(f"[-] Failed to bring FL Studio window to foreground: {e}")
                    return False
        print("[-] FL Studio window not found.")
        return False

    def send_sysex(self, command: str) -> bool:
        """Sends a custom SysEx command with the MAI header to FL Studio."""
        if not self.connect_midi():
            return False
            
        try:
            header = [0x00, 0x4D, 0x41, 0x49] # MAI Header
            msg = mido.Message('sysex', data=header + list(command.encode('utf-8')))
            self.midi_out.send(msg)
            return True
        except Exception as e:
            print(f"[-] Error sending SysEx command: {e}")
            return False

    def press_key(self, vk_code: int, delay: float = 0.1):
        """Simulates a key press (Windows only)."""
        if not HAS_WIN32:
            print("[!] win32 is not available. Cannot press key.")
            return
        try:
            win32api.keybd_event(vk_code, 0, 0, 0)
            time.sleep(delay)
            win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)
        except Exception as e:
            print(f"[-] Error simulating key press: {e}")

    def paste_text(self, text: str):
        """Copies text to clipboard and simulates Ctrl+V to paste (layout-independent)."""
        if not HAS_WIN32:
            print("[!] win32 is not available. Cannot paste text.")
            return
        import pyperclip
        try:
            old_clip = pyperclip.paste()
        except Exception:
            old_clip = ""

        try:
            pyperclip.copy(text)
            time.sleep(0.1)
            # Ctrl (0x11) Down
            win32api.keybd_event(0x11, 0, 0, 0)
            time.sleep(0.05)
            # V (0x56) Down
            win32api.keybd_event(0x56, 0, 0, 0)
            time.sleep(0.05)
            # V Up
            win32api.keybd_event(0x56, 0, win32con.KEYEVENTF_KEYUP, 0)
            # Ctrl Up
            win32api.keybd_event(0x11, 0, win32con.KEYEVENTF_KEYUP, 0)
            time.sleep(0.1)
        except Exception as e:
            print(f"[-] Error pasting text: {e}")
        finally:
            try:
                pyperclip.copy(old_clip)
            except Exception:
                pass

    def prepare_empty_track(self) -> bool:
        """Prepares the mixer and selects the first empty track."""
        print("[*] Preparing mixer and choosing an empty track...")
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
        """Commands FL Studio to find and target the specified plugin in mixer slots."""
        print(f"[*] Targeting plugin: '{plugin_name}'...")
        return self.send_sysex(f"TARGET_PLUGIN|{plugin_name}")

    def set_param(self, slot_index: int, param_index: int, value: float) -> bool:
        """Sets a specific parameter on the currently targeted or slot-specific plugin."""
        if not self.connect_midi():
            return False
            
        val_clamped = np.clip(value, 0.0, 1.0)
        val_midi = int(val_clamped * 127)
        
        try:
            if slot_index == -1:
                channel = 15 # target slot
            else:
                channel = int(np.clip(slot_index, 0, 9))
                
            msg = mido.Message('control_change', channel=channel, control=param_index, value=val_midi)
            self.midi_out.send(msg)
            return True
        except Exception as e:
            print(f"[-] Error sending MIDI param change: {e}")
            return False

    def set_eq_params(self, freqs: np.ndarray, delta: np.ndarray, plugin_name: str = "Fruity Parametric EQ 2") -> bool:
        """Sets parameters for Fruity Parametric EQ 2 using the spectral delta."""
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
            print(f"[-] Error setting EQ parameters: {e}")
            return False
