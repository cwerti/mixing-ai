from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time
import win32con

@dataclass
class PluginParam:
    index: int
    name: str
    min_val: float = 0.0
    max_val: float = 1.0
    default: float = 0.0

@dataclass
class PluginInfo:
    name: str
    param_count: int
    params: Dict[int, PluginParam] = field(default_factory=dict)

# Common FL Studio native plugins database
PLUGIN_DB: Dict[str, PluginInfo] = {
    "Fruity Parametric EQ 2": PluginInfo(
        name="Fruity Parametric EQ 2",
        param_count=35,
        params={}
    ),
    "Fruity Limiter": PluginInfo(
        name="Fruity Limiter",
        param_count=32,
        params={
            0: PluginParam(0, "Gain", default=0.5),
            1: PluginParam(1, "Ceil", default=1.0),
            2: PluginParam(2, "Thresh", default=1.0),
            3: PluginParam(3, "Attack", default=0.1),
            4: PluginParam(4, "Release", default=0.2),
        }
    ),
    "Fruity Compressor": PluginInfo(
        name="Fruity Compressor",
        param_count=10,
        params={
            0: PluginParam(0, "Threshold", default=0.5),
            1: PluginParam(1, "Ratio", default=0.2),
            2: PluginParam(2, "Gain", default=0.5),
            3: PluginParam(3, "Attack", default=0.1),
            4: PluginParam(4, "Release", default=0.2),
            5: PluginParam(5, "Type", default=0.0),
        }
    ),
    "Fruity Blood Overdrive": PluginInfo(
        name="Fruity Blood Overdrive",
        param_count=6,
        params={
            0: PluginParam(0, "PreAmp", default=0.0),
            1: PluginParam(1, "Dist", default=0.5),
            2: PluginParam(2, "x100", default=0.0),
            3: PluginParam(3, "PostFilter", default=0.5),
            4: PluginParam(4, "Color", default=0.5),
            5: PluginParam(5, "PostAmp", default=0.5),
        }
    ),
    "Soundgoodizer": PluginInfo(
        name="Soundgoodizer",
        param_count=2,
        params={
            0: PluginParam(0, "Blend", default=0.0),
            1: PluginParam(1, "Type", default=0.0),
        }
    ),
    "Fruity Reeverb 2": PluginInfo(
        name="Fruity Reeverb 2",
        param_count=20,
        params={
            0: PluginParam(0, "Decay", default=0.5),
            1: PluginParam(1, "HighCut", default=0.5),
            2: PluginParam(2, "LowCut", default=0.5),
            3: PluginParam(3, "Wet", default=0.2),
            4: PluginParam(4, "Dry", default=0.8),
            5: PluginParam(5, "RoomSize", default=0.5),
        }
    ),
    "Fruity Delay 3": PluginInfo(
        name="Fruity Delay 3",
        param_count=45,
        params={
            0: PluginParam(0, "DelayTime", default=0.5),
            1: PluginParam(1, "Feedback", default=0.3),
            2: PluginParam(2, "WetLevel", default=0.2),
        }
    ),
    "Fruity Stereo Enhancer": PluginInfo(
        name="Fruity Stereo Enhancer",
        param_count=10,
        params={
            0: PluginParam(0, "Stereo Separation", default=0.5),
            1: PluginParam(1, "Phase Inversion", default=0.0),
            2: PluginParam(2, "Volume", default=0.5),
        }
    )
}

# Programmatically populate all 7 bands for Fruity Parametric EQ 2
for band in range(7):
    base = band * 5
    PLUGIN_DB["Fruity Parametric EQ 2"].params[base] = PluginParam(base, f"Band {band+1} Freq")
    PLUGIN_DB["Fruity Parametric EQ 2"].params[base+1] = PluginParam(base+1, f"Band {band+1} Gain", default=0.5)
    PLUGIN_DB["Fruity Parametric EQ 2"].params[base+2] = PluginParam(base+2, f"Band {band+1} BW", default=0.5)
    PLUGIN_DB["Fruity Parametric EQ 2"].params[base+3] = PluginParam(base+3, f"Band {band+1} Type", default=0.5)
    PLUGIN_DB["Fruity Parametric EQ 2"].params[base+4] = PluginParam(base+4, f"Band {band+1} Order", default=0.5)


class PluginLoader:
    """Responsible for loading plugins via FL Studio's Plugin Picker."""
    def __init__(self, controller):
        self.fl = controller

    def load(self, plugin_name: str) -> bool:
        if plugin_name not in PLUGIN_DB:
            print(f"[!] Warning: '{plugin_name}' is not in PLUGIN_DB database.")
            
        print(f"[*] Loading plugin: {plugin_name}...")
        self.fl.focus_fl()
        
        # F8 - Plugin Picker
        self.fl.press_key(win32con.VK_F8)
        time.sleep(1.5)
        
        try:
            self.fl.paste_text(plugin_name)
            time.sleep(0.5)
            self.fl.press_key(win32con.VK_RETURN)
            return True
        except Exception as e:
            print(f"[-] Error loading plugin via paste: {e}")
            return False
