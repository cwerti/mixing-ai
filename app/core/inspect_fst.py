import pyflp
import sys
from pathlib import Path

def inspect_fst(file_path):
    try:
        preset = pyflp.parse(file_path)
        print(f"Preset type: {type(preset)}")
        
        # Для .fst файлов это обычно объект PluginState
        if hasattr(preset, 'plugin'):
             print(f"Plugin name: {preset.plugin.name}")
        
        # Попробуем найти параметры (Event ID в FL Studio)
        # В PyFLP параметры доступны через .events
        print(f"Number of events: {len(preset.events) if hasattr(preset, 'events') else 'N/A'}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_fst("FlProject/template.fst")
