import ctypes
from typing import Optional, Union, List

class TeVirtualMIDI:
    """
    Обертка над системным драйвером/SDK teVirtualMIDI.
    Используется для динамического программного создания виртуальных MIDI-портов на Windows.
    """
    
    def __init__(self, port_name: str = "Mixing-AI"):
        """
        Инициализирует виртуальный MIDI-порт.
        
        Args:
            port_name: Имя создаваемого порта в системе. По умолчанию "Mixing-AI".
        """
        self.port_name = port_name
        self.handle: Optional[ctypes.c_void_p] = None
        self.dll: Optional[ctypes.WinDLL] = None
        self._load_dll()

    def _load_dll(self):
        """Динамически загружает библиотеку teVirtualMIDI DLL и настраивает сигнатуры функций."""
        try:
            self.dll = ctypes.windll.teVirtualMIDI64
        except (OSError, AttributeError):
            try:
                self.dll = ctypes.windll.teVirtualMIDI
            except (OSError, AttributeError):
                self.dll = None

        if self.dll:
            # Настройка сигнатур функций SDK teVirtualMIDI
            self.dll.virtualMIDICreatePortEx2.restype = ctypes.c_void_p
            self.dll.virtualMIDICreatePortEx2.argtypes = [
                ctypes.c_wchar_p,  # portName
                ctypes.c_void_p,   # callback
                ctypes.c_void_p,   # dwCallbackInstance
                ctypes.c_uint32,   # maxSysexLength
                ctypes.c_uint32    # flags
            ]
            
            self.dll.virtualMIDIClosePort.argtypes = [ctypes.c_void_p]
            
            self.dll.virtualMIDISendData.restype = ctypes.c_bool
            self.dll.virtualMIDISendData.argtypes = [
                ctypes.c_void_p,                # hPort
                ctypes.POINTER(ctypes.c_ubyte), # data
                ctypes.c_uint32                 # length
            ]

    def create_port(self) -> bool:
        """
        Создает виртуальный MIDI-порт в операционной системе Windows.
        
        Returns:
            bool: True в случае успешного создания порта, иначе False.
        """
        if not self.dll:
            print("[-] Библиотека teVirtualMIDI DLL не загружена. Не удалось создать порт.")
            return False
        
        try:
            # Создаем порт с буфером SysEx 65535 байт
            self.handle = self.dll.virtualMIDICreatePortEx2(self.port_name, None, None, 65535, 0)
            return self.handle is not None
        except Exception as e:
            print(f"[-] Ошибка при создании виртуального MIDI порта: {e}")
            return False

    def send_data(self, data: Union[bytes, List[int]]) -> bool:
        """
        Отправляет сырые MIDI-байты через виртуальный порт.
        
        Args:
            data: Данные для отправки (байты или список чисел).
            
        Returns:
            bool: True в случае успешной отправки, иначе False.
        """
        if not self.dll or not self.handle:
            return False
        
        try:
            midi_data = (ctypes.c_ubyte * len(data))(*data)
            return self.dll.virtualMIDISendData(self.handle, midi_data, len(data))
        except Exception as e:
            print(f"[-] Ошибка отправки MIDI данных: {e}")
            return False

    def close(self):
        """Закрывает виртуальный MIDI-порт и освобождает системные дескрипторы."""
        if self.dll and self.handle:
            try:
                self.dll.virtualMIDIClosePort(self.handle)
            except Exception as e:
                print(f"[-] Ошибка при закрытии виртуального MIDI порта: {e}")
            self.handle = None

    def __enter__(self):
        self.create_port()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
