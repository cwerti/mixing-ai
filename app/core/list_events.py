import pyflp
from pathlib import Path

def list_events(file_path):
    try:
        preset = pyflp.parse(file_path)
        print(f"Preset type: {type(preset)}")
        
        # В PyFLP 2.x события хранятся в списке
        # Попробуем получить доступ к событиями плагина
        if hasattr(preset, 'events'):
            print(f"Total events: {len(preset.events)}")
            for i, event in enumerate(preset.events):
                # Попытаемся вывести ID и значение
                try:
                    print(f"Event {i}: ID={event.id}, Value={event.value}")
                except:
                    print(f"Event {i}: {event}")
        else:
            print("No events attribute found.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_events("FlProject/template.fst")
