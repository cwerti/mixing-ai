import ctypes
from typing import Optional, Union, List

class TeVirtualMIDI:
    """Wrapper for the teVirtualMIDI SDK DLL to create virtual MIDI ports on Windows."""
    def __init__(self, port_name: str = "Mixing-AI"):
        self.port_name = port_name
        self.handle: Optional[ctypes.c_void_p] = None
        self.dll: Optional[ctypes.WinDLL] = None
        self._load_dll()

    def _load_dll(self):
        """Loads teVirtualMIDI DLL dynamically and configures types."""
        try:
            self.dll = ctypes.windll.teVirtualMIDI64
        except (OSError, AttributeError):
            try:
                self.dll = ctypes.windll.teVirtualMIDI
            except (OSError, AttributeError):
                self.dll = None

        if self.dll:
            # Set up types for SDK functions
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
        """Creates the virtual MIDI port."""
        if not self.dll:
            print("[-] teVirtualMIDI DLL not loaded. Cannot create port.")
            return False
        
        try:
            self.handle = self.dll.virtualMIDICreatePortEx2(self.port_name, None, None, 65535, 0)
            return self.handle is not None
        except Exception as e:
            print(f"[-] Error creating virtual MIDI port: {e}")
            return False

    def send_data(self, data: Union[bytes, List[int]]) -> bool:
        """Sends raw MIDI data bytes through the virtual port."""
        if not self.dll or not self.handle:
            return False
        
        try:
            midi_data = (ctypes.c_ubyte * len(data))(*data)
            return self.dll.virtualMIDISendData(self.handle, midi_data, len(data))
        except Exception as e:
            print(f"[-] Error sending MIDI data: {e}")
            return False

    def close(self):
        """Closes the virtual MIDI port."""
        if self.dll and self.handle:
            try:
                self.dll.virtualMIDIClosePort(self.handle)
            except Exception as e:
                print(f"[-] Error closing virtual MIDI port: {e}")
            self.handle = None

    def __enter__(self):
        self.create_port()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
